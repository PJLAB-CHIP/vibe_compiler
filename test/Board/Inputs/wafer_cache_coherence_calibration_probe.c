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
} WaferCCHCase;

typedef struct WaferCCHPMU {
  uint32_t rdma_instructions;
  uint32_t wdma_instructions;
  uint64_t rdma_execution;
  uint64_t wdma_execution;
} WaferCCHPMU;

static const WaferCCHCase wafer_cch_cases[] = {
    {0U, 1U, 0U, 0U, 0U, WAFER_CCH_PAYLOAD_BYTES, 0U, 0U, 0U},
    {1U, 1U, 2U, 2U, 2U, WAFER_CCH_PAYLOAD_BYTES, 0U, 0U, 0U},
    {2U, 1U, 1U, 1U, 1U, WAFER_CCH_PAYLOAD_BYTES, 0U, 0U, 0U},
    {3U, 1U, 1U, 1U, 1U, WAFER_CCH_PAYLOAD_BYTES, 0U, 0U, 0U},
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

static uint32_t wafer_cch_decode(const volatile uint64_t *request,
                                 WaferCCHCase *selected, uint32_t *sample) {
  if (request[WAFER_CCH_REQ_MAGIC] != WAFER_CCH_REQUEST_MAGIC ||
      request[WAFER_CCH_REQ_SCHEMA_AND_WORDS] !=
          (((uint64_t)WAFER_CCH_SCHEMA << 32) | WAFER_CCH_REQUEST_WORDS) ||
      request[WAFER_CCH_REQ_RESOURCE_BYTES] != WAFER_CCH_RESOURCE_BYTES ||
      request[WAFER_CCH_REQ_BODY_OFFSET] != WAFER_CCH_BODY_OFFSET ||
      request[WAFER_CCH_REQ_OUTPUT0_OFFSET] != WAFER_CCH_OUTPUT0_OFFSET ||
      request[WAFER_CCH_REQ_OUTPUT1_OFFSET] != WAFER_CCH_OUTPUT1_OFFSET ||
      request[WAFER_CCH_REQ_GUARD] != WAFER_CCH_REQUEST_GUARD)
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
        selected->pair_kind == 1U ? 2U : selected->pair_kind == 3U ? 1U : 0U;
    selected->expected_wdma =
        selected->pair_kind == 2U ? 2U : selected->pair_kind == 3U ? 1U : 0U;
    selected->output_regions =
        selected->pair_kind == 2U ? 2U : selected->pair_kind == 3U ? 1U : 0U;
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

static uint8_t wafer_cch_bank_pattern(uint32_t case_id, uint32_t sample,
                                      uint32_t lane, uint32_t index) {
  return (uint8_t)(case_id * 31U + (sample + 1U) * 43U + lane * 97U +
                   index * 19U + (index >> 6) * 11U + 5U);
}

static void wafer_cch_seed_spm_bank(uint64_t address, uint32_t case_id,
                                    uint32_t sample, uint32_t lane) {
  volatile uint8_t *mapped =
      (volatile uint8_t *)(void *)get_spm_memory_mapping(address);
  for (uint32_t index = 0; index < WAFER_CCH_BANK_PAYLOAD_BYTES; ++index)
    mapped[index] = wafer_cch_bank_pattern(case_id, sample, lane, index);
  __asm__ volatile("fence iorw, iorw" ::: "memory");
}

static uint64_t wafer_cch_mismatch_spm_bank(uint64_t address,
                                            uint32_t case_id,
                                            uint32_t sample,
                                            uint32_t lane) {
  const volatile uint8_t *mapped =
      (const volatile uint8_t *)(const void *)get_spm_memory_mapping(address);
  uint64_t mismatches = 0;
  for (uint32_t index = 0; index < WAFER_CCH_BANK_PAYLOAD_BYTES; ++index)
    mismatches +=
        mapped[index] != wafer_cch_bank_pattern(case_id, sample, lane, index);
  return mismatches;
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
  record[WAFER_CCH_REC_SCHEMA_AND_WORDS] =
      ((uint64_t)WAFER_CCH_SCHEMA << 32) | WAFER_CCH_RECORD_WORDS;
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
                    selected->payload_bytes, 0, 0, 0, 1, 1, 1, Fmt_UINT8);
    if (selected->schedule == 1U)
      wafer_tx81_local_fence();
    wafer_tx81_rdma(input1, WAFER_CCH_SPM_B, selected->payload_bytes,
                    selected->payload_bytes, 0, 0, 0, 1, 1, 1, Fmt_UINT8);
  } else if (selected->pair_kind == 2U) {
    wafer_tx81_wdma(WAFER_CCH_SPM_A, output0, selected->payload_bytes,
                    selected->payload_bytes, 0, 0, 0, 1, 1, 1, Fmt_UINT8);
    if (selected->schedule == 1U)
      wafer_tx81_local_fence();
    wafer_tx81_wdma(WAFER_CCH_SPM_B, output1, selected->payload_bytes,
                    selected->payload_bytes, 0, 0, 0, 1, 1, 1, Fmt_UINT8);
  } else {
    wafer_tx81_rdma(input1, WAFER_CCH_SPM_B, selected->payload_bytes,
                    selected->payload_bytes, 0, 0, 0, 1, 1, 1, Fmt_UINT8);
    if (selected->schedule == 1U)
      wafer_tx81_local_fence();
    wafer_tx81_wdma(WAFER_CCH_SPM_A, output0, selected->payload_bytes,
                    selected->payload_bytes, 0, 0, 0, 1, 1, 1, Fmt_UINT8);
  }
  wafer_tx81_local_fence();
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
    wafer_cch_seed_spm(WAFER_CCH_SPM_A);
    wafer_cch_seed_spm(WAFER_CCH_SPM_B);
    if (selected.pair_kind == 2U) {
      wafer_cch_seed_spm_bank(WAFER_CCH_SPM_A, selected.case_id, sample, 0U);
      wafer_cch_seed_spm_bank(WAFER_CCH_SPM_B, selected.case_id, sample, 1U);
    } else if (selected.pair_kind == 3U) {
      wafer_cch_seed_spm_bank(WAFER_CCH_SPM_A, selected.case_id, sample, 0U);
    }
    WaferCCHPMU before = wafer_cch_read_pmu();

    if (selected.pair_kind != 0U) {
      wafer_cch_issue_bank_pair(&selected, input, bank_input1, output0,
                                bank_output1);
      mismatch_after =
          wafer_cch_mismatch_spm_bank(WAFER_CCH_SPM_A, selected.case_id,
                                      sample, 0U) +
          wafer_cch_mismatch_spm_bank(WAFER_CCH_SPM_B, selected.case_id,
                                      sample, 1U);
      cache_control_mask |= WAFER_CCH_MATCHING_LOCAL_FENCE;
    } else {
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
                      WAFER_CCH_PAYLOAD_BYTES, 0, 0, 0, 1, 1, 1, Fmt_UINT8);
      wafer_tx81_local_fence();
      mismatch_before =
          wafer_cch_mismatch_spm(WAFER_CCH_SPM_A, selected.case_id, sample, 1U);
      wafer_cch_cache_range(input, WAFER_CCH_PAYLOAD_BYTES, 0U);
      cache_control_mask |= WAFER_CCH_CACHE_CLEAN_SOURCE;
      wafer_tx81_rdma(input, WAFER_CCH_SPM_B, WAFER_CCH_PAYLOAD_BYTES,
                      WAFER_CCH_PAYLOAD_BYTES, 0, 0, 0, 1, 1, 1, Fmt_UINT8);
      wafer_tx81_local_fence();
      mismatch_after =
          wafer_cch_mismatch_spm(WAFER_CCH_SPM_B, selected.case_id, sample, 1U);
      wafer_tx81_wdma(WAFER_CCH_SPM_A, output0, WAFER_CCH_PAYLOAD_BYTES,
                      WAFER_CCH_PAYLOAD_BYTES, 0, 0, 0, 1, 1, 1, Fmt_UINT8);
      wafer_tx81_wdma(WAFER_CCH_SPM_B, output1, WAFER_CCH_PAYLOAD_BYTES,
                      WAFER_CCH_PAYLOAD_BYTES, 0, 0, 0, 1, 1, 1, Fmt_UINT8);
      wafer_tx81_local_fence();
      cache_control_mask |= WAFER_CCH_MATCHING_LOCAL_FENCE;
      break;
    case 2U:
      wafer_tx81_rdma(input, WAFER_CCH_SPM_A, WAFER_CCH_PAYLOAD_BYTES,
                      WAFER_CCH_PAYLOAD_BYTES, 0, 0, 0, 1, 1, 1, Fmt_UINT8);
      wafer_tx81_local_fence();
      (void)wafer_cch_mismatch_ddr(output0, selected.case_id, sample, 0U);
      wafer_tx81_wdma(WAFER_CCH_SPM_A, output0, WAFER_CCH_PAYLOAD_BYTES,
                      WAFER_CCH_PAYLOAD_BYTES, 0, 0, 0, 1, 1, 1, Fmt_UINT8);
      wafer_tx81_local_fence();
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
                      WAFER_CCH_PAYLOAD_BYTES, 0, 0, 0, 1, 1, 1, Fmt_UINT8);
      wafer_tx81_local_fence();
      wafer_tx81_wdma(WAFER_CCH_SPM_A, output0, WAFER_CCH_PAYLOAD_BYTES,
                      WAFER_CCH_PAYLOAD_BYTES, 0, 0, 0, 1, 1, 1, Fmt_UINT8);
      wafer_tx81_local_fence();
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
    record[WAFER_CCH_REC_SPM_GUARD_MISMATCHES] =
        wafer_cch_spm_guard_mismatches_bytes(WAFER_CCH_SPM_A,
                                             selected.payload_bytes) +
        wafer_cch_spm_guard_mismatches_bytes(WAFER_CCH_SPM_B,
                                             selected.payload_bytes);
    if (selected.pair_kind != 0U) {
      const uint32_t bank_output_offsets[2] = {
          WAFER_CCH_OUTPUT0_OFFSET,
          WAFER_CCH_OUTPUT0_OFFSET + WAFER_CCH_BANK_REGION_GAP +
              selected.bank_offset,
      };
      record[WAFER_CCH_REC_OUTPUT_GUARD_MISMATCHES] =
          wafer_cch_output_guard_mismatches_at(
              output_ddr, bank_output_offsets, selected.output_regions,
              selected.payload_bytes);
    } else {
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
