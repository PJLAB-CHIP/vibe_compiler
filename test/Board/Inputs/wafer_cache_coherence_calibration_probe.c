#include "instr_def.h"
#include "wafer_cache_coherence_calibration_probe_protocol.h"
#include "wafer_tx81_crt.h"

#include <stdint.h>

#define WAFER_CCH_PMU_BASE UINT64_C(0x590000)
#define WAFER_CCH_STABLE_RETRIES 8U
#define WAFER_CCH_SPM_GUARD_VALUE UINT8_C(0x6d)
#define WAFER_CCH_CACHE_INVALIDATE_INPUT UINT32_C(1)
#define WAFER_CCH_CACHE_CLEAN_SOURCE UINT32_C(2)
#define WAFER_CCH_CACHE_INVALIDATE_OUTPUT UINT32_C(4)
#define WAFER_CCH_MATCHING_LOCAL_FENCE UINT32_C(8)

extern int8_t *get_spm_memory_mapping(uint64_t offset);

typedef struct WaferCCHCase {
  uint32_t case_id;
  uint32_t phases;
  uint32_t expected_rdma;
  uint32_t expected_wdma;
  uint32_t output_regions;
  uint32_t payload_bytes;
  uint32_t pair_kind;
  uint32_t schedule;
  uint32_t bank_offset;
  uint32_t mode;
  uint32_t transfer_bytes;
  uint32_t base_relation;
  uint32_t base_translation;
} WaferCCHCase;

typedef struct WaferCCHPMU {
  uint32_t rdma_instructions;
  uint32_t wdma_instructions;
  uint64_t rdma_execution;
  uint64_t wdma_execution;
} WaferCCHPMU;

static const WaferCCHCase wafer_cch_cases[] = {
    {0U, 1U, 0U, 0U, 0U, WAFER_CCH_PAYLOAD_BYTES, 0U, 0U, 0U,
     0U, 0U, 0U, 0U},
    {1U, 1U, 2U, 2U, 2U, WAFER_CCH_PAYLOAD_BYTES, 0U, 0U, 0U,
     0U, 0U, 0U, 0U},
    {2U, 1U, 1U, 1U, 1U, WAFER_CCH_PAYLOAD_BYTES, 0U, 0U, 0U,
     0U, 0U, 0U, 0U},
    {3U, 1U, 1U, 1U, 1U, WAFER_CCH_PAYLOAD_BYTES, 0U, 0U, 0U,
     0U, 0U, 0U, 0U},
};

static const uint32_t wafer_cch_bank_offsets[] = {
    0U, 256U, 512U, 1024U, 2048U, 4096U, 8192U, 16384U, 32768U,
};

static void wafer_cch_cache_range(uint64_t begin, uint32_t bytes,
                                  uint32_t invalidate_only) {
  enum {
    WAFER_CCH_SUPERVISOR_MODE = 1,
    WAFER_CCH_MACHINE_MODE = 3,
    WAFER_CCH_CACHE_LINE_BYTES = 64,
  };
  uintptr_t mode;
  __asm__ volatile("fence" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
  __asm__ volatile("csrr %0, mxstatus" : "=r"(mode));
  mode = (mode >> 30) & 3U;
  for (uintptr_t address = (uintptr_t)begin; address < (uintptr_t)begin + bytes;
       address += WAFER_CCH_CACHE_LINE_BYTES) {
    if (mode == WAFER_CCH_MACHINE_MODE) {
      if (invalidate_only != 0U)
        __asm__ volatile("dcache.ipa %0" : : "r"(address) : "memory");
      else
        __asm__ volatile("dcache.cipa %0" : : "r"(address) : "memory");
    } else if (mode == WAFER_CCH_SUPERVISOR_MODE) {
      if (invalidate_only != 0U)
        __asm__ volatile("dcache.iva %0" : : "r"(address) : "memory");
      else
        __asm__ volatile("dcache.civa %0" : : "r"(address) : "memory");
    }
  }
  __asm__ volatile("sync.is" ::: "memory");
  __asm__ volatile("fence" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
}

static uint32_t wafer_cch_read32(uint32_t offset) {
  return *(const volatile uint32_t *)(uintptr_t)(WAFER_CCH_PMU_BASE + offset);
}

static uint64_t wafer_cch_read64(uint32_t low_offset) {
  uint32_t low = 0;
  uint32_t high_after = 0;
  for (uint32_t retry = 0; retry < WAFER_CCH_STABLE_RETRIES; ++retry) {
    uint32_t high_before = wafer_cch_read32(low_offset + 4U);
    low = wafer_cch_read32(low_offset);
    high_after = wafer_cch_read32(low_offset + 4U);
    if (high_before == high_after)
      break;
  }
  return ((uint64_t)high_after << 32) | low;
}

static WaferCCHPMU wafer_cch_read_pmu(void) {
  WaferCCHPMU result;
  result.rdma_instructions = wafer_cch_read32(GR_PMU_RDMA_INST_NUMS);
  result.wdma_instructions = wafer_cch_read32(GR_PMU_WDMA_INST_NUMS);
  result.rdma_execution = wafer_cch_read64(GR_PMU_RDMA_EXE_TIME);
  result.wdma_execution = wafer_cch_read64(GR_PMU_WDMA_EXE_TIME);
  return result;
}

static uint32_t wafer_cch_conflict_axis_index(
    const uint32_t *values, uint32_t count, uint32_t value,
    uint32_t *index) {
  for (uint32_t candidate = 0; candidate < count; ++candidate) {
    if (values[candidate] == value) {
      *index = candidate;
      return 1U;
    }
  }
  return 0U;
}

static uint32_t wafer_cch_decode_conflict(
    const volatile uint64_t *request, WaferCCHCase *selected) {
  static const uint32_t transfers[] = {256U, 4096U};
  static const uint32_t translations[] = {0U, 4096U};
  uint32_t transfer_index = 0U;
  uint32_t translation_index = 0U;
  uint32_t offset_index = 0U;
  uint32_t pair_kind = (uint32_t)request[WAFER_CCH_REQ_PAIR_KIND];
  uint32_t base_relation =
      (uint32_t)request[WAFER_CCH_REQ_BASE_RELATION];
  uint32_t transfer =
      (uint32_t)request[WAFER_CCH_REQ_TRANSFER_BYTES];
  uint32_t translation =
      (uint32_t)request[WAFER_CCH_REQ_BASE_TRANSLATION];
  uint32_t bank_offset =
      (uint32_t)request[WAFER_CCH_REQ_BANK_OFFSET];
  if ((pair_kind != 1U && pair_kind != 2U) ||
      (base_relation != WAFER_CCH_CONFLICT_BASE_SAME_ALLOCATION &&
       base_relation != WAFER_CCH_CONFLICT_BASE_CROSS_ALLOCATION) ||
      (pair_kind == 2U &&
       base_relation != WAFER_CCH_CONFLICT_BASE_SAME_ALLOCATION) ||
      !wafer_cch_conflict_axis_index(
          transfers, sizeof(transfers) / sizeof(transfers[0]), transfer,
          &transfer_index) ||
      !wafer_cch_conflict_axis_index(
          translations,
          sizeof(translations) / sizeof(translations[0]), translation,
          &translation_index) ||
      !wafer_cch_conflict_axis_index(
          wafer_cch_bank_offsets,
          sizeof(wafer_cch_bank_offsets) /
              sizeof(wafer_cch_bank_offsets[0]),
          bank_offset, &offset_index))
    return 0U;
  uint32_t expected_id = WAFER_CCH_CONFLICT_FIRST_CASE_ID;
  if (pair_kind == 1U) {
    expected_id += transfer_index * 36U;
    expected_id +=
        (base_relation -
         WAFER_CCH_CONFLICT_BASE_SAME_ALLOCATION) *
        18U;
  } else {
    expected_id += 72U + transfer_index * 18U;
  }
  expected_id += translation_index * 9U + offset_index;
  uint32_t expected_rdma =
      pair_kind == 1U
          ? WAFER_CCH_BANK_SEED_RDMA_INSTRUCTIONS +
                WAFER_CCH_CONFLICT_BATCH_ROWS * 2U
          : WAFER_CCH_BANK_SEED_RDMA_INSTRUCTIONS;
  uint32_t expected_wdma =
      pair_kind == 1U
          ? WAFER_CCH_CONFLICT_BATCH_ROWS *
                WAFER_CCH_BANK_READBACK_WDMA_INSTRUCTIONS
          : 4U + WAFER_CCH_CONFLICT_BATCH_ROWS * 2U;
  if (pair_kind == 2U)
    expected_rdma += WAFER_CCH_CONFLICT_BATCH_ROWS * 2U;
  if (request[WAFER_CCH_REQ_CASE] != expected_id ||
      request[WAFER_CCH_REQ_SAMPLE] != 0U ||
      request[WAFER_CCH_REQ_PAYLOAD_BYTES] != transfer ||
      request[WAFER_CCH_REQ_EXPECTED_RDMA] != expected_rdma ||
      request[WAFER_CCH_REQ_EXPECTED_WDMA] != expected_wdma ||
      request[WAFER_CCH_REQ_SCHEDULE] != 0U ||
      request[WAFER_CCH_REQ_REPETITIONS] !=
          WAFER_CCH_CONFLICT_REPETITIONS ||
      request[WAFER_CCH_REQ_BATCH_ROWS] !=
          WAFER_CCH_CONFLICT_BATCH_ROWS ||
      request[WAFER_CCH_REQ_ARCHIVE_BASE] !=
          WAFER_CCH_CONFLICT_ARCHIVE_BASE ||
      request[WAFER_CCH_REQ_ARCHIVE_STRIDE] !=
          WAFER_CCH_CONFLICT_ARCHIVE_STRIDE ||
      WAFER_CCH_CONFLICT_DATA_BASE + translation +
              WAFER_CCH_BANK_REGION_GAP + bank_offset + transfer +
              WAFER_CCH_SPM_GUARD_BYTES >
          WAFER_CCH_RESOURCE_BYTES)
    return 0U;
  selected->case_id = expected_id;
  selected->phases = 1U;
  selected->expected_rdma = expected_rdma;
  selected->expected_wdma = expected_wdma;
  selected->output_regions = 2U;
  selected->payload_bytes = transfer;
  selected->pair_kind = pair_kind;
  selected->schedule = 0U;
  selected->bank_offset = bank_offset;
  selected->mode = WAFER_CCH_CONFLICT_MODE;
  selected->transfer_bytes = transfer;
  selected->base_relation = base_relation;
  selected->base_translation = translation;
  return 1U;
}

static uint32_t wafer_cch_decode(const volatile uint64_t *request,
                                 WaferCCHCase *selected, uint32_t *sample) {
  if (request[WAFER_CCH_REQ_MAGIC] != WAFER_CCH_REQUEST_MAGIC ||
      request[WAFER_CCH_REQ_WORD_COUNT] !=
          (WAFER_CCH_REQUEST_WORDS) ||
      request[WAFER_CCH_REQ_RESOURCE_BYTES] != WAFER_CCH_RESOURCE_BYTES ||
      request[WAFER_CCH_REQ_BODY_OFFSET] != WAFER_CCH_BODY_OFFSET ||
      request[WAFER_CCH_REQ_OUTPUT0_OFFSET] != WAFER_CCH_OUTPUT0_OFFSET ||
      request[WAFER_CCH_REQ_OUTPUT1_OFFSET] != WAFER_CCH_OUTPUT1_OFFSET ||
      request[WAFER_CCH_REQ_GUARD] != WAFER_CCH_REQUEST_GUARD)
    return WAFER_CCH_STATUS_BAD_REQUEST;
  if (request[WAFER_CCH_REQ_MODE] == WAFER_CCH_CONFLICT_MODE) {
    if (!wafer_cch_decode_conflict(request, selected))
      return WAFER_CCH_STATUS_BAD_REQUEST;
    *sample = 0U;
    return WAFER_CCH_STATUS_OK;
  }
  if (request[WAFER_CCH_REQ_MODE] != 0U ||
      request[WAFER_CCH_REQ_TRANSFER_BYTES] != 0U ||
      request[WAFER_CCH_REQ_BASE_RELATION] != 0U ||
      request[WAFER_CCH_REQ_BASE_TRANSLATION] != 0U ||
      request[WAFER_CCH_REQ_REPETITIONS] != 0U ||
      request[WAFER_CCH_REQ_BATCH_ROWS] != 0U ||
      request[WAFER_CCH_REQ_ARCHIVE_BASE] != 0U ||
      request[WAFER_CCH_REQ_ARCHIVE_STRIDE] != 0U)
    return WAFER_CCH_STATUS_BAD_REQUEST;
  uint32_t case_id = (uint32_t)request[WAFER_CCH_REQ_CASE];
  if (case_id < sizeof(wafer_cch_cases) / sizeof(wafer_cch_cases[0])) {
    *selected = wafer_cch_cases[case_id];
  } else {
    const uint32_t schedules = 2U;
    const uint32_t offsets =
        sizeof(wafer_cch_bank_offsets) / sizeof(wafer_cch_bank_offsets[0]);
    const uint32_t cases_per_pair = offsets * schedules;
    uint32_t bank_case = case_id - 4U;
    uint32_t pair_index = bank_case / cases_per_pair;
    uint32_t within_pair = bank_case % cases_per_pair;
    uint32_t offset_index = within_pair / schedules;
    uint32_t schedule = within_pair % schedules + 1U;
    if (pair_index >= 3U || offset_index >= offsets)
      return WAFER_CCH_STATUS_BAD_REQUEST;
    selected->case_id = case_id;
    selected->phases = WAFER_CCH_BANK_PMU_REPETITIONS;
    selected->payload_bytes = WAFER_CCH_BANK_PAYLOAD_BYTES;
    selected->pair_kind = pair_index + 1U;
    selected->schedule = schedule;
    selected->bank_offset = wafer_cch_bank_offsets[offset_index];
    selected->expected_rdma =
        WAFER_CCH_BANK_SEED_RDMA_INSTRUCTIONS +
        (selected->pair_kind == 1U ? 2U
                                   : selected->pair_kind == 3U ? 1U : 0U);
    selected->expected_wdma =
        WAFER_CCH_BANK_READBACK_WDMA_INSTRUCTIONS +
        (selected->pair_kind == 2U ? 2U
                                   : selected->pair_kind == 3U ? 1U : 0U);
    selected->output_regions = 2U;
  }
  *sample = (uint32_t)request[WAFER_CCH_REQ_SAMPLE];
  if (*sample >= selected->phases ||
      request[WAFER_CCH_REQ_PAYLOAD_BYTES] != selected->payload_bytes ||
      request[WAFER_CCH_REQ_EXPECTED_RDMA] != selected->expected_rdma ||
      request[WAFER_CCH_REQ_EXPECTED_WDMA] != selected->expected_wdma ||
      request[WAFER_CCH_REQ_PAIR_KIND] != selected->pair_kind ||
      request[WAFER_CCH_REQ_SCHEDULE] != selected->schedule ||
      request[WAFER_CCH_REQ_BANK_OFFSET] != selected->bank_offset ||
      WAFER_CCH_BODY_OFFSET + WAFER_CCH_BANK_REGION_GAP +
              selected->bank_offset + selected->payload_bytes >
          WAFER_CCH_RESOURCE_BYTES ||
      WAFER_CCH_OUTPUT0_OFFSET + WAFER_CCH_BANK_REGION_GAP +
              selected->bank_offset + selected->payload_bytes >
          WAFER_CCH_RESOURCE_BYTES)
    return WAFER_CCH_STATUS_BAD_REQUEST;
  return WAFER_CCH_STATUS_OK;
}

static uint8_t wafer_cch_pattern(uint32_t case_id, uint32_t sample,
                                 uint32_t index) {
  return (uint8_t)(case_id * 41U + (sample + 1U) * 13U + index * 17U +
                   (index >> 7) * 29U + 3U);
}

static uint8_t wafer_cch_store_pattern(uint32_t case_id, uint32_t sample,
                                       uint32_t index) {
  return (uint8_t)(case_id * 41U + (sample + 101U) * 13U + index * 17U +
                   (index >> 7) * 29U + 3U);
}

static void wafer_cch_fill_ddr(uint64_t address, uint32_t bytes,
                               uint8_t value) {
  volatile uint8_t *mapped = (volatile uint8_t *)(uintptr_t)address;
  for (uint32_t index = 0; index < bytes; ++index)
    mapped[index] = value;
}

static void wafer_cch_store_generated(uint64_t address, uint32_t case_id,
                                      uint32_t sample) {
  volatile uint8_t *mapped = (volatile uint8_t *)(uintptr_t)address;
  for (uint32_t index = 0; index < WAFER_CCH_PAYLOAD_BYTES; ++index)
    mapped[index] = wafer_cch_store_pattern(case_id, sample, index);
  __asm__ volatile("fence" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
}

static uint64_t wafer_cch_mismatch_ddr(uint64_t address, uint32_t case_id,
                                       uint32_t sample,
                                       uint32_t store_pattern) {
  const volatile uint8_t *mapped = (const volatile uint8_t *)(uintptr_t)address;
  uint64_t mismatches = 0;
  for (uint32_t index = 0; index < WAFER_CCH_PAYLOAD_BYTES; ++index) {
    uint8_t expected = store_pattern != 0U
                           ? wafer_cch_store_pattern(case_id, sample, index)
                           : wafer_cch_pattern(case_id, sample, index);
    mismatches += mapped[index] != expected;
  }
  return mismatches;
}

static uint64_t wafer_cch_mismatch_spm(uint64_t address, uint32_t case_id,
                                       uint32_t sample,
                                       uint32_t store_pattern) {
  const volatile uint8_t *mapped =
      (const volatile uint8_t *)(const void *)get_spm_memory_mapping(address);
  uint64_t mismatches = 0;
  for (uint32_t index = 0; index < WAFER_CCH_PAYLOAD_BYTES; ++index) {
    uint8_t expected = store_pattern != 0U
                           ? wafer_cch_store_pattern(case_id, sample, index)
                           : wafer_cch_pattern(case_id, sample, index);
    mismatches += mapped[index] != expected;
  }
  return mismatches;
}

static void wafer_cch_seed_spm(uint64_t address) {
  volatile uint8_t *mapped = (volatile uint8_t *)(void *)get_spm_memory_mapping(
      address - WAFER_CCH_SPM_GUARD_BYTES);
  for (uint32_t index = 0;
       index < WAFER_CCH_PAYLOAD_BYTES + 2U * WAFER_CCH_SPM_GUARD_BYTES;
       ++index)
    mapped[index] = WAFER_CCH_SPM_GUARD_VALUE;
  __asm__ volatile("fence iorw, iorw" ::: "memory");
}

static uint64_t wafer_cch_spm_guard_mismatches_bytes(uint64_t address,
                                                     uint32_t payload_bytes) {
  const volatile uint8_t *before =
      (const volatile uint8_t *)(const void *)get_spm_memory_mapping(
          address - WAFER_CCH_SPM_GUARD_BYTES);
  const volatile uint8_t *after =
      (const volatile uint8_t *)(const void *)get_spm_memory_mapping(
          address + payload_bytes);
  uint64_t mismatches = 0;
  for (uint32_t index = 0; index < WAFER_CCH_SPM_GUARD_BYTES; ++index) {
    mismatches += before[index] != WAFER_CCH_SPM_GUARD_VALUE;
    mismatches += after[index] != WAFER_CCH_SPM_GUARD_VALUE;
  }
  return mismatches;
}

static uint64_t wafer_cch_output_guard_mismatches_at(
    uint64_t output_ddr, const uint32_t *offsets, uint32_t regions,
    uint32_t payload_bytes) {
  uint64_t mismatches = 0;
  for (uint32_t region = 0; region < regions; ++region) {
    const volatile uint8_t *before =
        (const volatile uint8_t *)(uintptr_t)(output_ddr + offsets[region] -
                                              WAFER_CCH_SPM_GUARD_BYTES);
    const volatile uint8_t *after =
        (const volatile uint8_t *)(uintptr_t)(output_ddr + offsets[region] +
                                              payload_bytes);
    for (uint32_t index = 0; index < WAFER_CCH_SPM_GUARD_BYTES; ++index) {
      mismatches += before[index] != WAFER_CCH_OUTPUT_CANARY;
      mismatches += after[index] != WAFER_CCH_OUTPUT_CANARY;
    }
  }
  return mismatches;
}

static uint64_t wafer_cch_output_guard_mismatches(uint64_t output_ddr,
                                                  uint32_t regions) {
  const uint32_t offsets[2] = {WAFER_CCH_OUTPUT0_OFFSET,
                               WAFER_CCH_OUTPUT1_OFFSET};
  return wafer_cch_output_guard_mismatches_at(
      output_ddr, offsets, regions, WAFER_CCH_PAYLOAD_BYTES);
}

static void wafer_cch_init_record(volatile uint64_t *record, uint32_t status) {
  for (uint32_t index = 0; index < WAFER_CCH_RECORD_WORDS; ++index)
    record[index] = 0;
  record[WAFER_CCH_REC_MAGIC] = WAFER_CCH_RECORD_MAGIC;
  record[WAFER_CCH_REC_WORD_COUNT] =
      WAFER_CCH_RECORD_WORDS;
  record[WAFER_CCH_REC_STATUS] = status;
  record[WAFER_CCH_REC_OUTPUT0_OFFSET] = WAFER_CCH_OUTPUT0_OFFSET;
  record[WAFER_CCH_REC_OUTPUT1_OFFSET] = WAFER_CCH_OUTPUT1_OFFSET;
  record[WAFER_CCH_REC_RECORD_GUARD] = WAFER_CCH_RECORD_GUARD;
}

static void wafer_cch_issue_bank_pair(const WaferCCHCase *selected,
                                      uint64_t input0, uint64_t input1,
                                      uint64_t output0, uint64_t output1) {
  if (selected->pair_kind == 1U) {
    wafer_tx81_rdma(input0, WAFER_CCH_SPM_A, selected->payload_bytes,
                    selected->payload_bytes, 0, 0, 0, 1, 1, 1, Fmt_UINT8, 0U);
    if (selected->schedule == 1U)
      wafer_tx81_ncc_join(1U);
    wafer_tx81_rdma(input1, WAFER_CCH_SPM_B, selected->payload_bytes,
                    selected->payload_bytes, 0, 0, 0, 1, 1, 1, Fmt_UINT8, 0U);
  } else if (selected->pair_kind == 2U) {
    wafer_tx81_wdma(WAFER_CCH_SPM_A, output0, selected->payload_bytes,
                    selected->payload_bytes, 0, 0, 0, 1, 1, 1, Fmt_UINT8, 0U);
    if (selected->schedule == 1U)
      wafer_tx81_ncc_join(1U);
    wafer_tx81_wdma(WAFER_CCH_SPM_B, output1, selected->payload_bytes,
                    selected->payload_bytes, 0, 0, 0, 1, 1, 1, Fmt_UINT8, 0U);
  } else {
    wafer_tx81_rdma(input1, WAFER_CCH_SPM_B, selected->payload_bytes,
                    selected->payload_bytes, 0, 0, 0, 1, 1, 1, Fmt_UINT8, 0U);
    if (selected->schedule == 1U)
      wafer_tx81_ncc_join(1U);
    wafer_tx81_wdma(WAFER_CCH_SPM_A, output0, selected->payload_bytes,
                    selected->payload_bytes, 0, 0, 0, 1, 1, 1, Fmt_UINT8, 0U);
  }
}

static void wafer_cch_seed_bank_slots(uint64_t input0, uint64_t input1) {
  wafer_tx81_rdma(input0 - WAFER_CCH_SPM_GUARD_BYTES,
                  WAFER_CCH_SPM_A - WAFER_CCH_SPM_GUARD_BYTES,
                  WAFER_CCH_BANK_SLOT_BYTES, WAFER_CCH_BANK_SLOT_BYTES, 0, 0,
                  0, 1, 1, 1, Fmt_UINT8, 0U);
  wafer_tx81_rdma(input1 - WAFER_CCH_SPM_GUARD_BYTES,
                  WAFER_CCH_SPM_B - WAFER_CCH_SPM_GUARD_BYTES,
                  WAFER_CCH_BANK_SLOT_BYTES, WAFER_CCH_BANK_SLOT_BYTES, 0, 0,
                  0, 1, 1, 1, Fmt_UINT8, 0U);
}

static void wafer_cch_readback_bank_slots(uint64_t output0,
                                          uint64_t output1) {
  wafer_tx81_wdma(WAFER_CCH_SPM_A - WAFER_CCH_SPM_GUARD_BYTES,
                  output0 - WAFER_CCH_SPM_GUARD_BYTES,
                  WAFER_CCH_BANK_SLOT_BYTES, WAFER_CCH_BANK_SLOT_BYTES, 0, 0,
                  0, 1, 1, 1, Fmt_UINT8, 0U);
  wafer_tx81_wdma(WAFER_CCH_SPM_B - WAFER_CCH_SPM_GUARD_BYTES,
                  output1 - WAFER_CCH_SPM_GUARD_BYTES,
                  WAFER_CCH_BANK_SLOT_BYTES, WAFER_CCH_BANK_SLOT_BYTES, 0, 0,
                  0, 1, 1, 1, Fmt_UINT8, 0U);
}

static uint64_t wafer_cch_cycle(void) {
  uint64_t value;
  __asm__ volatile("rdcycle %0" : "=r"(value));
  return value;
}

static void wafer_cch_issue_conflict_pair(
    const WaferCCHCase *selected, uint64_t address_a,
    uint64_t address_b, uint32_t schedule, uint32_t issue_order) {
  uint32_t first = issue_order == 0U ? 0U : 1U;
  for (uint32_t ordinal = 0; ordinal < 2U; ++ordinal) {
    uint32_t lane = ordinal == 0U ? first : (first ^ 1U);
    uint64_t address = lane == 0U ? address_a : address_b;
    uint64_t spm = lane == 0U ? WAFER_CCH_SPM_A : WAFER_CCH_SPM_B;
    if (selected->pair_kind == 1U) {
      wafer_tx81_rdma(address, spm, selected->transfer_bytes,
                      selected->transfer_bytes, 0U, 0U, 0U, 1U, 1U,
                      1U, Fmt_UINT8, 0U);
    } else {
      wafer_tx81_wdma(spm, address, selected->transfer_bytes,
                      selected->transfer_bytes, 0U, 0U, 0U, 1U, 1U,
                      1U, Fmt_UINT8, 0U);
    }
    if (schedule == 1U && ordinal == 0U)
      wafer_tx81_ncc_join(1U);
  }
  wafer_tx81_ncc_join(1U);
}

static uint32_t wafer_cch_conflict_identity(
    uint32_t repetition, uint32_t schedule, uint32_t issue_order) {
  return repetition | (schedule << 8) | (issue_order << 16);
}

static uint8_t wafer_cch_conflict_pattern(uint32_t case_id, uint32_t lane,
                                          uint32_t index) {
  return (uint8_t)(case_id * 31U + 43U + lane * 97U + index * 19U +
                   (index >> 6) * 11U + 5U);
}

static uint64_t wafer_cch_conflict_spm_mismatches(uint64_t address,
                                                   uint32_t case_id,
                                                   uint32_t lane,
                                                   uint32_t transfer_bytes) {
  const volatile uint8_t *mapped =
      (const volatile uint8_t *)(const void *)get_spm_memory_mapping(
          address - WAFER_CCH_SPM_GUARD_BYTES);
  uint64_t mismatches = 0U;
  for (uint32_t index = 0U; index < WAFER_CCH_SPM_GUARD_BYTES; ++index)
    mismatches += mapped[index] != WAFER_CCH_SPM_GUARD_VALUE;
  for (uint32_t index = 0U; index < transfer_bytes; ++index)
    mismatches +=
        mapped[WAFER_CCH_SPM_GUARD_BYTES + index] !=
        wafer_cch_conflict_pattern(case_id, lane, index);
  for (uint32_t index = 0U; index < WAFER_CCH_SPM_GUARD_BYTES; ++index)
    mismatches +=
        mapped[WAFER_CCH_SPM_GUARD_BYTES + transfer_bytes + index] !=
        WAFER_CCH_SPM_GUARD_VALUE;
  return mismatches;
}

static uint32_t wafer_cch_execute_conflict(
    const WaferCCHCase *selected, uint64_t request_ddr,
    uint64_t payload_ddr, uint64_t output_ddr,
    volatile uint64_t *record, WaferCCHPMU *total_before,
    WaferCCHPMU *total_after, uint64_t *base_a, uint64_t *base_b,
    uint64_t *address_a, uint64_t *address_b) {
  uint64_t offset_a =
      WAFER_CCH_CONFLICT_DATA_BASE + selected->base_translation;
  uint64_t offset_b =
      offset_a + WAFER_CCH_BANK_REGION_GAP + selected->bank_offset;
  if (selected->pair_kind == 1U) {
    *base_a =
        selected->base_relation ==
                WAFER_CCH_CONFLICT_BASE_CROSS_ALLOCATION
            ? request_ddr
            : payload_ddr;
    *base_b = payload_ddr;
  } else {
    *base_a = output_ddr;
    *base_b = output_ddr;
  }
  *address_a = *base_a + offset_a;
  *address_b = *base_b + offset_b;

  uint64_t seed_a =
      selected->pair_kind == 1U ? *address_a : payload_ddr + offset_a;
  uint64_t seed_b =
      selected->pair_kind == 1U ? *address_b : payload_ddr + offset_b;
  *total_before = wafer_cch_read_pmu();
  wafer_tx81_rdma(
      seed_a - WAFER_CCH_SPM_GUARD_BYTES,
      WAFER_CCH_SPM_A - WAFER_CCH_SPM_GUARD_BYTES,
      selected->transfer_bytes + 2U * WAFER_CCH_SPM_GUARD_BYTES,
      selected->transfer_bytes + 2U * WAFER_CCH_SPM_GUARD_BYTES,
      0U, 0U, 0U, 1U, 1U, 1U, Fmt_UINT8, 0U);
  wafer_tx81_rdma(
      seed_b - WAFER_CCH_SPM_GUARD_BYTES,
      WAFER_CCH_SPM_B - WAFER_CCH_SPM_GUARD_BYTES,
      selected->transfer_bytes + 2U * WAFER_CCH_SPM_GUARD_BYTES,
      selected->transfer_bytes + 2U * WAFER_CCH_SPM_GUARD_BYTES,
      0U, 0U, 0U, 1U, 1U, 1U, Fmt_UINT8, 0U);
  wafer_tx81_ncc_join(1U);

  if (selected->pair_kind == 2U) {
    const uint64_t spm[2] = {WAFER_CCH_SPM_A, WAFER_CCH_SPM_B};
    const uint64_t address[2] = {*address_a, *address_b};
    for (uint32_t lane = 0; lane < 2U; ++lane) {
      wafer_tx81_wdma(
          spm[lane] - WAFER_CCH_SPM_GUARD_BYTES,
          address[lane] - WAFER_CCH_SPM_GUARD_BYTES,
          WAFER_CCH_SPM_GUARD_BYTES, WAFER_CCH_SPM_GUARD_BYTES,
          0U, 0U, 0U, 1U, 1U, 1U, Fmt_UINT8, 0U);
      wafer_tx81_wdma(
          spm[lane] + selected->transfer_bytes,
          address[lane] + selected->transfer_bytes,
          WAFER_CCH_SPM_GUARD_BYTES, WAFER_CCH_SPM_GUARD_BYTES,
          0U, 0U, 0U, 1U, 1U, 1U, Fmt_UINT8, 0U);
    }
    wafer_tx81_ncc_join(1U);
  }

  uint32_t rows_written = 0U;
  for (uint32_t repetition = 0U;
       repetition < WAFER_CCH_CONFLICT_REPETITIONS; ++repetition) {
    for (uint32_t issue_order = 0U;
         issue_order < WAFER_CCH_CONFLICT_ISSUE_ORDERS; ++issue_order) {
      uint32_t reverse_schedule = (repetition + issue_order) & 1U;
      for (uint32_t schedule_position = 0U;
           schedule_position < WAFER_CCH_CONFLICT_SCHEDULES;
           ++schedule_position, ++rows_written) {
        uint32_t schedule =
            reverse_schedule != 0U
                ? WAFER_CCH_CONFLICT_SCHEDULES - schedule_position
                : schedule_position + 1U;
        uint32_t row_ordinal =
            (repetition * WAFER_CCH_CONFLICT_SCHEDULES + schedule - 1U) *
                WAFER_CCH_CONFLICT_ISSUE_ORDERS +
            issue_order;
        WaferCCHPMU before = wafer_cch_read_pmu();
        uint64_t begin = wafer_cch_cycle();
        wafer_cch_issue_conflict_pair(
            selected, *address_a, *address_b, schedule, issue_order);
        uint64_t plan_cycles = wafer_cch_cycle() - begin;
        WaferCCHPMU after = wafer_cch_read_pmu();
        if (selected->pair_kind == 2U) {
          uint32_t slot_bytes =
              selected->transfer_bytes +
              2U * WAFER_CCH_SPM_GUARD_BYTES;
          wafer_tx81_rdma(
              *address_a - WAFER_CCH_SPM_GUARD_BYTES,
              WAFER_CCH_SPM_A - WAFER_CCH_SPM_GUARD_BYTES,
              slot_bytes, slot_bytes, 0U, 0U, 0U, 1U, 1U, 1U,
              Fmt_UINT8, 0U);
          wafer_tx81_rdma(
              *address_b - WAFER_CCH_SPM_GUARD_BYTES,
              WAFER_CCH_SPM_B - WAFER_CCH_SPM_GUARD_BYTES,
              slot_bytes, slot_bytes, 0U, 0U, 0U, 1U, 1U, 1U,
              Fmt_UINT8, 0U);
          wafer_tx81_ncc_join(1U);
        }
        uint64_t result_mismatches =
            wafer_cch_conflict_spm_mismatches(
                WAFER_CCH_SPM_A, selected->case_id, 0U,
                selected->transfer_bytes) +
            wafer_cch_conflict_spm_mismatches(
                WAFER_CCH_SPM_B, selected->case_id, 1U,
                selected->transfer_bytes);
        uint32_t control =
            (schedule - 1U) * WAFER_CCH_CONFLICT_ISSUE_ORDERS +
            issue_order;
        uint64_t archive0 =
            selected->pair_kind == 1U
                ? WAFER_CCH_CONFLICT_ARCHIVE_BASE +
                      control * WAFER_CCH_CONFLICT_ARCHIVE_STRIDE
                : offset_a - WAFER_CCH_SPM_GUARD_BYTES;
        uint64_t archive1 =
            selected->pair_kind == 1U
                ? archive0 + selected->transfer_bytes +
                      2U * WAFER_CCH_SPM_GUARD_BYTES
                : offset_b - WAFER_CCH_SPM_GUARD_BYTES;
        volatile uint64_t *row =
            record + WAFER_CCH_RECORD_WORDS +
            row_ordinal * WAFER_CCH_CONFLICT_ROW_WORDS;
        row[WAFER_CCH_CONFLICT_ROW_MAGIC_WORD] =
            WAFER_CCH_CONFLICT_ROW_MAGIC;
        row[WAFER_CCH_CONFLICT_ROW_IDENTITY] =
            wafer_cch_conflict_identity(
                repetition, schedule, issue_order);
        row[WAFER_CCH_CONFLICT_ROW_REPETITION] = repetition;
        row[WAFER_CCH_CONFLICT_ROW_SCHEDULE] = schedule;
        row[WAFER_CCH_CONFLICT_ROW_ISSUE_ORDER] = issue_order;
        row[WAFER_CCH_CONFLICT_ROW_RDMA_INST_DELTA] =
            (uint32_t)(after.rdma_instructions -
                       before.rdma_instructions);
        row[WAFER_CCH_CONFLICT_ROW_WDMA_INST_DELTA] =
            (uint32_t)(after.wdma_instructions -
                       before.wdma_instructions);
        row[WAFER_CCH_CONFLICT_ROW_RDMA_EXEC_DELTA] =
            after.rdma_execution - before.rdma_execution;
        row[WAFER_CCH_CONFLICT_ROW_WDMA_EXEC_DELTA] =
            after.wdma_execution - before.wdma_execution;
        row[WAFER_CCH_CONFLICT_ROW_PLAN_CYCLES] = plan_cycles;
        row[WAFER_CCH_CONFLICT_ROW_ARCHIVE0_OFFSET] = archive0;
        row[WAFER_CCH_CONFLICT_ROW_ARCHIVE1_OFFSET] = archive1;
        row[WAFER_CCH_CONFLICT_ROW_DDR_ADDRESS_A] = *address_a;
        row[WAFER_CCH_CONFLICT_ROW_DDR_ADDRESS_B] = *address_b;
        row[WAFER_CCH_CONFLICT_ROW_COMPLETED] = 1U;
        row[WAFER_CCH_CONFLICT_ROW_GUARD_WORD] =
            WAFER_CCH_CONFLICT_ROW_GUARD;
        row[WAFER_CCH_CONFLICT_ROW_RESULT_MISMATCHES] =
            result_mismatches;
        row[WAFER_CCH_CONFLICT_ROW_SCHEDULE_POSITION] =
            schedule_position;
        if (selected->pair_kind == 1U) {
          uint32_t slot_bytes =
              selected->transfer_bytes +
              2U * WAFER_CCH_SPM_GUARD_BYTES;
          wafer_tx81_wdma(
              WAFER_CCH_SPM_A - WAFER_CCH_SPM_GUARD_BYTES,
              output_ddr + archive0, slot_bytes, slot_bytes,
              0U, 0U, 0U, 1U, 1U, 1U, Fmt_UINT8, 0U);
          wafer_tx81_wdma(
              WAFER_CCH_SPM_B - WAFER_CCH_SPM_GUARD_BYTES,
              output_ddr + archive1, slot_bytes, slot_bytes,
              0U, 0U, 0U, 1U, 1U, 1U, Fmt_UINT8, 0U);
          wafer_tx81_ncc_join(1U);
        }
      }
    }
  }
  if (rows_written != WAFER_CCH_CONFLICT_BATCH_ROWS)
    return 0U;
  *total_after = wafer_cch_read_pmu();
  return 1U;
}

__attribute__((visibility("hidden"))) void
wafer_tx81_instruction_family_probe(uint64_t request_ddr, uint64_t payload_ddr,
                                    uint64_t output_ddr) {
  wafer_cch_cache_range(request_ddr, WAFER_CCH_BODY_OFFSET, 1U);
  const volatile uint64_t *request =
      (const volatile uint64_t *)(uintptr_t)request_ddr;
  WaferCCHCase selected = {0};
  uint32_t sample = 0;
  uint32_t status = wafer_cch_decode(request, &selected, &sample);

  wafer_cch_fill_ddr(output_ddr, WAFER_CCH_RESOURCE_BYTES,
                     WAFER_CCH_OUTPUT_CANARY);
  wafer_cch_cache_range(output_ddr, WAFER_CCH_RESOURCE_BYTES, 0U);
  volatile uint64_t *record = (volatile uint64_t *)(uintptr_t)output_ddr;
  wafer_cch_init_record(record, status);
  if (status == WAFER_CCH_STATUS_OK &&
      selected.mode == WAFER_CCH_CONFLICT_MODE) {
    WaferCCHPMU before = {0};
    WaferCCHPMU after = {0};
    uint64_t base_a = 0U;
    uint64_t base_b = 0U;
    uint64_t address_a = 0U;
    uint64_t address_b = 0U;
    if (!wafer_cch_execute_conflict(
            &selected, request_ddr, payload_ddr, output_ddr, record,
            &before, &after, &base_a, &base_b, &address_a,
            &address_b))
      status = WAFER_CCH_STATUS_EXECUTE_FAILED;
    record[WAFER_CCH_REC_CASE] = selected.case_id;
    record[WAFER_CCH_REC_SAMPLE] = sample;
    record[WAFER_CCH_REC_PAYLOAD_BYTES] = selected.payload_bytes;
    record[WAFER_CCH_REC_EXPECTED_RDMA] = selected.expected_rdma;
    record[WAFER_CCH_REC_EXPECTED_WDMA] = selected.expected_wdma;
    record[WAFER_CCH_REC_REQUEST_GUARD] =
        request[WAFER_CCH_REQ_GUARD];
    record[WAFER_CCH_REC_SPM_GUARD_MISMATCHES] = 0U;
    record[WAFER_CCH_REC_OUTPUT_GUARD_MISMATCHES] = 0U;
    record[WAFER_CCH_REC_RDMA_INST_DELTA] =
        (uint32_t)(after.rdma_instructions -
                   before.rdma_instructions);
    record[WAFER_CCH_REC_WDMA_INST_DELTA] =
        (uint32_t)(after.wdma_instructions -
                   before.wdma_instructions);
    record[WAFER_CCH_REC_CACHE_CONTROL_MASK] =
        WAFER_CCH_MATCHING_LOCAL_FENCE;
    record[WAFER_CCH_REC_RDMA_EXEC_DELTA] =
        after.rdma_execution - before.rdma_execution;
    record[WAFER_CCH_REC_WDMA_EXEC_DELTA] =
        after.wdma_execution - before.wdma_execution;
    record[WAFER_CCH_REC_PAIR_KIND] = selected.pair_kind;
    record[WAFER_CCH_REC_SCHEDULE] = 0U;
    record[WAFER_CCH_REC_BANK_OFFSET] = selected.bank_offset;
    record[WAFER_CCH_REC_MODE] = selected.mode;
    record[WAFER_CCH_REC_TRANSFER_BYTES] =
        selected.transfer_bytes;
    record[WAFER_CCH_REC_BASE_RELATION] =
        selected.base_relation;
    record[WAFER_CCH_REC_BASE_TRANSLATION] =
        selected.base_translation;
    record[WAFER_CCH_REC_DDR_BASE_A] = base_a;
    record[WAFER_CCH_REC_DDR_BASE_B] = base_b;
    record[WAFER_CCH_REC_DDR_ADDRESS_A] = address_a;
    record[WAFER_CCH_REC_DDR_ADDRESS_B] = address_b;
    record[WAFER_CCH_REC_STATUS] = status;
    wafer_cch_cache_range(
        output_ddr, WAFER_CCH_CONFLICT_RECORD_BYTES, 0U);
    return;
  }
  if (status == WAFER_CCH_STATUS_OK) {
    uint64_t input = payload_ddr + WAFER_CCH_BODY_OFFSET;
    uint64_t bank_input1 =
        input + WAFER_CCH_BANK_REGION_GAP + selected.bank_offset;
    uint64_t output0 = output_ddr + WAFER_CCH_OUTPUT0_OFFSET;
    uint64_t output1 = output_ddr + WAFER_CCH_OUTPUT1_OFFSET;
    uint64_t bank_output1 =
        output0 + WAFER_CCH_BANK_REGION_GAP + selected.bank_offset;
    uint64_t mismatch_before = 0;
    uint64_t mismatch_after = 0;
    uint32_t cache_control_mask = 0;
    WaferCCHPMU before;

    if (selected.pair_kind != 0U) {
      /*
       * Keep the ordinary DDR pair probe entirely in the NCC completion
       * domain.  The fixed envelope is:
       *
       *   host payload -> two RDMA slot seeds -> measured pair
       *                -> two full-slot WDMA readbacks -> terminal fence
       *
       * Exact/guard validation is therefore a host oracle over the two slot
       * dumps.  The serial schedule retains only its intentional fence
       * between the measured pair operations; the window schedule has no
       * intermediate fence.
       */
      before = wafer_cch_read_pmu();
      wafer_cch_seed_bank_slots(input, bank_input1);
      wafer_cch_issue_bank_pair(&selected, input, bank_input1, output0,
                                bank_output1);
      wafer_cch_readback_bank_slots(output0, bank_output1);
      wafer_tx81_ncc_join(1U);
      cache_control_mask |= WAFER_CCH_MATCHING_LOCAL_FENCE;
    } else {
      wafer_cch_seed_spm(WAFER_CCH_SPM_A);
      wafer_cch_seed_spm(WAFER_CCH_SPM_B);
      before = wafer_cch_read_pmu();
      switch (selected.case_id) {
    case 0U:
      wafer_cch_cache_range(input, WAFER_CCH_PAYLOAD_BYTES, 1U);
      cache_control_mask |= WAFER_CCH_CACHE_INVALIDATE_INPUT;
      mismatch_before =
          wafer_cch_mismatch_ddr(input, selected.case_id, sample, 0U);
      mismatch_after =
          wafer_cch_mismatch_ddr(input, selected.case_id, sample, 0U);
      break;
    case 1U:
      wafer_cch_cache_range(input, WAFER_CCH_PAYLOAD_BYTES, 1U);
      cache_control_mask |= WAFER_CCH_CACHE_INVALIDATE_INPUT;
      wafer_cch_store_generated(input, selected.case_id, sample);
      wafer_tx81_rdma(input, WAFER_CCH_SPM_A, WAFER_CCH_PAYLOAD_BYTES,
                      WAFER_CCH_PAYLOAD_BYTES, 0, 0, 0, 1, 1, 1, Fmt_UINT8, 0U);
      wafer_tx81_ncc_join(1U);
      mismatch_before =
          wafer_cch_mismatch_spm(WAFER_CCH_SPM_A, selected.case_id, sample, 1U);
      wafer_cch_cache_range(input, WAFER_CCH_PAYLOAD_BYTES, 0U);
      cache_control_mask |= WAFER_CCH_CACHE_CLEAN_SOURCE;
      wafer_tx81_rdma(input, WAFER_CCH_SPM_B, WAFER_CCH_PAYLOAD_BYTES,
                      WAFER_CCH_PAYLOAD_BYTES, 0, 0, 0, 1, 1, 1, Fmt_UINT8, 0U);
      wafer_tx81_ncc_join(1U);
      mismatch_after =
          wafer_cch_mismatch_spm(WAFER_CCH_SPM_B, selected.case_id, sample, 1U);
      wafer_tx81_wdma(WAFER_CCH_SPM_A, output0, WAFER_CCH_PAYLOAD_BYTES,
                      WAFER_CCH_PAYLOAD_BYTES, 0, 0, 0, 1, 1, 1, Fmt_UINT8, 0U);
      wafer_tx81_wdma(WAFER_CCH_SPM_B, output1, WAFER_CCH_PAYLOAD_BYTES,
                      WAFER_CCH_PAYLOAD_BYTES, 0, 0, 0, 1, 1, 1, Fmt_UINT8, 0U);
      wafer_tx81_ncc_join(1U);
      cache_control_mask |= WAFER_CCH_MATCHING_LOCAL_FENCE;
      break;
    case 2U:
      wafer_tx81_rdma(input, WAFER_CCH_SPM_A, WAFER_CCH_PAYLOAD_BYTES,
                      WAFER_CCH_PAYLOAD_BYTES, 0, 0, 0, 1, 1, 1, Fmt_UINT8, 0U);
      wafer_tx81_ncc_join(1U);
      (void)wafer_cch_mismatch_ddr(output0, selected.case_id, sample, 0U);
      wafer_tx81_wdma(WAFER_CCH_SPM_A, output0, WAFER_CCH_PAYLOAD_BYTES,
                      WAFER_CCH_PAYLOAD_BYTES, 0, 0, 0, 1, 1, 1, Fmt_UINT8, 0U);
      wafer_tx81_ncc_join(1U);
      cache_control_mask |= WAFER_CCH_MATCHING_LOCAL_FENCE;
      mismatch_before =
          wafer_cch_mismatch_ddr(output0, selected.case_id, sample, 0U);
      wafer_cch_cache_range(output0, WAFER_CCH_PAYLOAD_BYTES, 1U);
      cache_control_mask |= WAFER_CCH_CACHE_INVALIDATE_OUTPUT;
      mismatch_after =
          wafer_cch_mismatch_ddr(output0, selected.case_id, sample, 0U);
      break;
    case 3U:
      wafer_tx81_rdma(input, WAFER_CCH_SPM_A, WAFER_CCH_PAYLOAD_BYTES,
                      WAFER_CCH_PAYLOAD_BYTES, 0, 0, 0, 1, 1, 1, Fmt_UINT8, 0U);
      wafer_tx81_ncc_join(1U);
      wafer_tx81_wdma(WAFER_CCH_SPM_A, output0, WAFER_CCH_PAYLOAD_BYTES,
                      WAFER_CCH_PAYLOAD_BYTES, 0, 0, 0, 1, 1, 1, Fmt_UINT8, 0U);
      wafer_tx81_ncc_join(1U);
      cache_control_mask |= WAFER_CCH_MATCHING_LOCAL_FENCE;
      break;
    default:
      status = WAFER_CCH_STATUS_EXECUTE_FAILED;
      break;
      }
    }

    WaferCCHPMU after = wafer_cch_read_pmu();
    record[WAFER_CCH_REC_CASE] = selected.case_id;
    record[WAFER_CCH_REC_SAMPLE] = sample;
    record[WAFER_CCH_REC_PAYLOAD_BYTES] = selected.payload_bytes;
    record[WAFER_CCH_REC_EXPECTED_RDMA] = selected.expected_rdma;
    record[WAFER_CCH_REC_EXPECTED_WDMA] = selected.expected_wdma;
    record[WAFER_CCH_REC_REQUEST_GUARD] = request[WAFER_CCH_REQ_GUARD];
    record[WAFER_CCH_REC_MISMATCH_BEFORE] = mismatch_before;
    record[WAFER_CCH_REC_MISMATCH_AFTER] = mismatch_after;
    if (selected.pair_kind != 0U) {
      /*
       * The pair path deliberately performs no Kcore/mapped-SPM scan.  The
       * host validates both complete slot dumps, including their guards, and
       * the untouched output canary.  Zero here means no device-side scan was
       * requested, not that a mapped alias supplied the oracle.
       */
      record[WAFER_CCH_REC_SPM_GUARD_MISMATCHES] = 0U;
      record[WAFER_CCH_REC_OUTPUT_GUARD_MISMATCHES] = 0U;
    } else {
      record[WAFER_CCH_REC_SPM_GUARD_MISMATCHES] =
          wafer_cch_spm_guard_mismatches_bytes(WAFER_CCH_SPM_A,
                                               selected.payload_bytes) +
          wafer_cch_spm_guard_mismatches_bytes(WAFER_CCH_SPM_B,
                                               selected.payload_bytes);
      record[WAFER_CCH_REC_OUTPUT_GUARD_MISMATCHES] =
          wafer_cch_output_guard_mismatches(output_ddr,
                                            selected.output_regions);
    }
    record[WAFER_CCH_REC_RDMA_INST_DELTA] =
        (uint32_t)(after.rdma_instructions - before.rdma_instructions);
    record[WAFER_CCH_REC_WDMA_INST_DELTA] =
        (uint32_t)(after.wdma_instructions - before.wdma_instructions);
    record[WAFER_CCH_REC_CACHE_CONTROL_MASK] = cache_control_mask;
    record[WAFER_CCH_REC_RDMA_EXEC_DELTA] =
        after.rdma_execution - before.rdma_execution;
    record[WAFER_CCH_REC_WDMA_EXEC_DELTA] =
        after.wdma_execution - before.wdma_execution;
    record[WAFER_CCH_REC_PAIR_KIND] = selected.pair_kind;
    record[WAFER_CCH_REC_SCHEDULE] = selected.schedule;
    record[WAFER_CCH_REC_BANK_OFFSET] = selected.bank_offset;
    record[WAFER_CCH_REC_STATUS] = status;
  }
  wafer_cch_cache_range(output_ddr, WAFER_CCH_RECORD_WORDS * sizeof(uint64_t),
                        0U);
}
