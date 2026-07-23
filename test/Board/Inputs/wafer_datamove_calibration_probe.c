#include "instr_def.h"
#include "wafer_datamove_calibration_probe_protocol.h"
#include "wafer_tx81_crt.h"

#include <stdint.h>

#define WAFER_DMC_PMU_BASE UINT64_C(0x590000)
#define WAFER_DMC_STABLE_RETRIES 8U

extern int8_t *get_spm_memory_mapping(uint64_t offset);

typedef struct WaferDMCCase {
  uint32_t case_id;
  uint32_t input_bytes;
  uint32_t result_bytes;
  uint32_t output_span;
  uint32_t expected_instructions;
} WaferDMCCase;

typedef struct WaferDMCPMU {
  uint32_t instructions;
  uint32_t blocking;
  uint64_t execution;
} WaferDMCPMU;

static const WaferDMCCase wafer_dmc_cases[] = {
    {0U, 3922U, 3922U, 4096U, 1U},     {1U, 3922U, 3922U, 4096U, 53U},
    {2U, 3922U, 3922U, 4096U, 37U},    {3U, 646U, 646U, 768U, 323U},
    {4U, 3922U, 3922U, 4096U, 53U},    {5U, 6460U, 6460U, 6656U, 2U},
    {6U, 6460U, 6460U, 6656U, 2U},     {7U, 23256U, 23256U, 23296U, 2U},
    {8U, 106U, 3922U, 4096U, 1U},      {9U, 74U, 3922U, 4096U, 1U},
    {10U, 16380U, 17152U, 17152U, 2U}, {11U, 17152U, 16380U, 16384U, 2U},
    {12U, 16380U, 17408U, 17408U, 4U}, {13U, 17408U, 16380U, 16384U, 4U},
    {14U, 8320U, 4224U, 4352U, 1U},
};

static void wafer_dmc_cache_range(uint64_t begin, uint32_t bytes,
                                  uint32_t invalidate_only) {
  enum {
    WAFER_DMC_SUPERVISOR_MODE = 1,
    WAFER_DMC_MACHINE_MODE = 3,
    WAFER_DMC_CACHE_LINE_BYTES = 64,
  };
  uintptr_t mode;
  __asm__ volatile("fence" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
  __asm__ volatile("csrr %0, mxstatus" : "=r"(mode));
  mode = (mode >> 30) & 3U;
  for (uintptr_t address = (uintptr_t)begin; address < (uintptr_t)begin + bytes;
       address += WAFER_DMC_CACHE_LINE_BYTES) {
    if (mode == WAFER_DMC_MACHINE_MODE) {
      if (invalidate_only != 0U)
        __asm__ volatile("dcache.ipa %0" : : "r"(address) : "memory");
      else
        __asm__ volatile("dcache.cipa %0" : : "r"(address) : "memory");
    } else if (mode == WAFER_DMC_SUPERVISOR_MODE) {
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

static uint32_t wafer_dmc_read32(uint32_t offset) {
  return *(const volatile uint32_t *)(uintptr_t)(WAFER_DMC_PMU_BASE + offset);
}

static uint64_t wafer_dmc_read64(uint32_t low_offset) {
  uint32_t low = 0;
  uint32_t high_after = 0;
  for (uint32_t retry = 0; retry < WAFER_DMC_STABLE_RETRIES; ++retry) {
    uint32_t high_before = wafer_dmc_read32(low_offset + 4U);
    low = wafer_dmc_read32(low_offset);
    high_after = wafer_dmc_read32(low_offset + 4U);
    if (high_before == high_after)
      break;
  }
  return ((uint64_t)high_after << 32) | low;
}

static WaferDMCPMU wafer_dmc_read_pmu(void) {
  WaferDMCPMU result;
  result.instructions = wafer_dmc_read32(GR_PMU_TDMA_INST_NUMS);
  result.blocking = wafer_dmc_read32(GR_PMU_TDMA_BLOCKING_TIME);
  result.execution = wafer_dmc_read64(GR_PMU_TDMA_EXE_TIME);
  return result;
}

static uint32_t wafer_dmc_decode(const volatile uint64_t *request,
                                 WaferDMCCase *selected) {
  if (request[WAFER_DMC_REQ_MAGIC] != WAFER_DMC_REQUEST_MAGIC ||
      request[WAFER_DMC_REQ_SCHEMA_AND_WORDS] !=
          (((uint64_t)WAFER_DMC_SCHEMA << 32) | WAFER_DMC_REQUEST_WORDS) ||
      request[WAFER_DMC_REQ_RESOURCE_BYTES] != WAFER_DMC_RESOURCE_BYTES ||
      request[WAFER_DMC_REQ_SLOT_BYTES] != WAFER_DMC_SLOT_BYTES ||
      request[WAFER_DMC_REQ_BODY_OFFSET] != WAFER_DMC_BODY_OFFSET ||
      request[WAFER_DMC_REQ_GUARD] != WAFER_DMC_REQUEST_GUARD)
    return WAFER_DMC_STATUS_BAD_REQUEST;
  uint32_t case_id = (uint32_t)request[WAFER_DMC_REQ_CASE];
  if (case_id >= sizeof(wafer_dmc_cases) / sizeof(wafer_dmc_cases[0]))
    return WAFER_DMC_STATUS_BAD_REQUEST;
  *selected = wafer_dmc_cases[case_id];
  if (request[WAFER_DMC_REQ_INPUT_BYTES] != selected->input_bytes ||
      request[WAFER_DMC_REQ_RESULT_BYTES] != selected->result_bytes ||
      request[WAFER_DMC_REQ_OUTPUT_SPAN] != selected->output_span ||
      request[WAFER_DMC_REQ_EXPECTED_INSTRUCTIONS] !=
          selected->expected_instructions ||
      WAFER_DMC_BODY_OFFSET + selected->input_bytes > WAFER_DMC_SLOT_BYTES ||
      WAFER_DMC_BODY_OFFSET + selected->output_span > WAFER_DMC_SLOT_BYTES)
    return WAFER_DMC_STATUS_BAD_REQUEST;
  return WAFER_DMC_STATUS_OK;
}

static void wafer_dmc_fill_output(void) {
  volatile uint8_t *output =
      (volatile uint8_t *)(void *)get_spm_memory_mapping(WAFER_DMC_SPM_OUTPUT);
  for (uint32_t index = 0; index < WAFER_DMC_SLOT_BYTES; ++index)
    output[index] = WAFER_DMC_SLOT_CANARY;
  __asm__ volatile("fence iorw, iorw" ::: "memory");
}

static uint64_t wafer_dmc_guard_mismatches(const WaferDMCCase *selected) {
  const volatile uint8_t *output =
      (const volatile uint8_t *)(const void *)get_spm_memory_mapping(
          WAFER_DMC_SPM_OUTPUT);
  uint64_t mismatches = 0;
  for (uint32_t index = 0; index < WAFER_DMC_BODY_OFFSET; ++index)
    mismatches += output[index] != WAFER_DMC_SLOT_CANARY;
  for (uint32_t index = WAFER_DMC_BODY_OFFSET + selected->output_span;
       index < WAFER_DMC_SLOT_BYTES; ++index)
    mismatches += output[index] != WAFER_DMC_SLOT_CANARY;
  return mismatches;
}

static void wafer_dmc_gather(uint32_t source_offset, uint32_t dest_offset,
                             uint32_t byte_count, uint32_t inner_bytes,
                             uint32_t source_stride0, uint32_t source_stride1,
                             uint32_t source_stride2,
                             uint32_t source_iteration0,
                             uint32_t source_iteration1,
                             uint32_t source_iteration2, uint32_t dest_stride0,
                             uint32_t dest_stride1, uint32_t dest_stride2,
                             uint32_t dest_iteration0, uint32_t dest_iteration1,
                             uint32_t dest_iteration2) {
  wafer_tx81_gather_scatter(
      WAFER_DMC_SPM_INPUT + WAFER_DMC_BODY_OFFSET + source_offset,
      WAFER_DMC_SPM_OUTPUT + WAFER_DMC_BODY_OFFSET + dest_offset, byte_count,
      inner_bytes, source_stride0, source_stride1, source_stride2,
      source_iteration0, source_iteration1, source_iteration2, dest_stride0,
      dest_stride1, dest_stride2, dest_iteration0, dest_iteration1,
      dest_iteration2);
}

static uint32_t wafer_dmc_issue(const WaferDMCCase *selected) {
  switch (selected->case_id) {
  case 0U:
    wafer_dmc_gather(0U, 0U, 3922U, 2U, 106U, 2U, 0U, 37U, 53U, 1U, 2U, 74U, 0U,
                     37U, 53U, 1U);
    break;
  case 1U:
    for (uint32_t column = 0; column < 53U; ++column) {
      wafer_dmc_gather(column * 2U, (52U - column) * 2U, 74U, 2U, 106U, 0U, 0U,
                       37U, 1U, 1U, 106U, 0U, 0U, 37U, 1U, 1U);
      wafer_tx81_local_fence();
    }
    return 1U;
  case 2U:
    for (uint32_t row = 0; row < 37U; ++row) {
      wafer_dmc_gather(row * 106U, (36U - row) * 2U, 106U, 2U, 2U, 0U, 0U, 53U,
                       1U, 1U, 74U, 0U, 0U, 53U, 1U, 1U);
      wafer_tx81_local_fence();
    }
    return 1U;
  case 3U:
    for (uint32_t row = 0; row < 17U; ++row)
      for (uint32_t column = 0; column < 19U; ++column) {
        uint32_t source = (row * 19U + column) * 2U;
        uint32_t dest = ((16U - row) * 19U + (18U - column)) * 2U;
        wafer_dmc_gather(source, dest, 2U, 2U, 0U, 0U, 0U, 1U, 1U, 1U, 0U, 0U,
                         0U, 1U, 1U, 1U);
        wafer_tx81_local_fence();
      }
    return 1U;
  case 4U:
    for (uint32_t column = 0; column < 53U; ++column) {
      wafer_dmc_gather(column * 2U, (52U - column) * 74U, 74U, 2U, 106U, 0U, 0U,
                       37U, 1U, 1U, 2U, 0U, 0U, 37U, 1U, 1U);
      wafer_tx81_local_fence();
    }
    return 1U;
  case 5U:
    for (uint32_t batch = 0; batch < 2U; ++batch)
      wafer_dmc_gather(batch * 3230U, batch * 3230U, 3230U, 2U, 646U, 2U, 38U,
                       5U, 19U, 17U, 2U, 10U, 190U, 5U, 19U, 17U);
    break;
  case 6U:
    for (uint32_t batch = 0; batch < 2U; ++batch)
      wafer_dmc_gather(batch * 3230U, batch * 3230U, 3230U, 2U, 10U, 190U, 2U,
                       19U, 17U, 5U, 2U, 38U, 646U, 19U, 17U, 5U);
    break;
  case 7U:
    wafer_dmc_gather(0U, 0U, 9044U, 14U, 14U, 0U, 0U, 646U, 1U, 1U, 36U, 0U, 0U,
                     646U, 1U, 1U);
    wafer_dmc_gather(9044U, 14U, 14212U, 22U, 22U, 0U, 0U, 646U, 1U, 1U, 36U,
                     0U, 0U, 646U, 1U, 1U);
    break;
  case 8U:
    wafer_dmc_gather(0U, 0U, 3922U, 106U, 0U, 0U, 0U, 37U, 1U, 1U, 106U, 0U, 0U,
                     37U, 1U, 1U);
    break;
  case 9U:
    wafer_dmc_gather(0U, 0U, 3922U, 2U, 0U, 2U, 0U, 53U, 37U, 1U, 2U, 106U, 0U,
                     53U, 37U, 1U);
    break;
  case 10U:
    wafer_dmc_gather(0U, 0U, 16128U, 128U, 130U, 0U, 0U, 126U, 1U, 1U, 128U, 0U,
                     0U, 126U, 1U, 1U);
    wafer_dmc_gather(128U, 16128U, 252U, 2U, 130U, 0U, 0U, 126U, 1U, 1U, 8U, 0U,
                     0U, 126U, 1U, 1U);
    break;
  case 11U:
    wafer_dmc_gather(0U, 0U, 16128U, 128U, 128U, 0U, 0U, 126U, 1U, 1U, 130U, 0U,
                     0U, 126U, 1U, 1U);
    wafer_dmc_gather(16128U, 128U, 252U, 2U, 8U, 0U, 0U, 126U, 1U, 1U, 130U, 0U,
                     0U, 126U, 1U, 1U);
    break;
  case 12U:
    for (uint32_t batch = 0; batch < 2U; ++batch) {
      wafer_dmc_gather(batch * 8190U, batch * 8704U, 8064U, 128U, 130U, 0U, 0U,
                       63U, 1U, 1U, 128U, 0U, 0U, 63U, 1U, 1U);
      wafer_dmc_gather(batch * 8190U + 128U, batch * 8704U + 8064U, 126U, 2U,
                       130U, 0U, 0U, 63U, 1U, 1U, 8U, 0U, 0U, 63U, 1U, 1U);
    }
    break;
  case 13U:
    for (uint32_t batch = 0; batch < 2U; ++batch) {
      wafer_dmc_gather(batch * 8704U, batch * 8190U, 8064U, 128U, 128U, 0U, 0U,
                       63U, 1U, 1U, 130U, 0U, 0U, 63U, 1U, 1U);
      wafer_dmc_gather(batch * 8704U + 8064U, batch * 8190U + 128U, 126U, 2U,
                       8U, 0U, 0U, 63U, 1U, 1U, 130U, 0U, 0U, 63U, 1U, 1U);
    }
    break;
  case 14U:
    wafer_dmc_gather(0U, 0U, 4224U, 2U, 4U, 130U, 0U, 33U, 64U, 1U, 2U, 66U, 0U,
                     33U, 64U, 1U);
    break;
  default:
    return 0U;
  }
  wafer_tx81_local_fence();
  return 1U;
}

static void wafer_dmc_init_record(volatile uint64_t *record, uint32_t status) {
  for (uint32_t index = 0; index < WAFER_DMC_RECORD_WORDS; ++index)
    record[index] = 0;
  record[WAFER_DMC_REC_MAGIC] = WAFER_DMC_RECORD_MAGIC;
  record[WAFER_DMC_REC_SCHEMA_AND_WORDS] =
      ((uint64_t)WAFER_DMC_SCHEMA << 32) | WAFER_DMC_RECORD_WORDS;
  record[WAFER_DMC_REC_STATUS] = status;
  record[WAFER_DMC_REC_OUTPUT_DDR_OFFSET] = WAFER_DMC_OUTPUT_DDR_OFFSET;
  record[WAFER_DMC_REC_SLOT_BYTES] = WAFER_DMC_SLOT_BYTES;
  record[WAFER_DMC_REC_BODY_OFFSET] = WAFER_DMC_BODY_OFFSET;
  record[WAFER_DMC_REC_RECORD_GUARD] = WAFER_DMC_RECORD_GUARD;
}

__attribute__((visibility("hidden"))) void
wafer_tx81_instruction_family_probe(uint64_t request_ddr, uint64_t payload_ddr,
                                    uint64_t output_ddr) {
  wafer_dmc_cache_range(request_ddr, WAFER_DMC_RESOURCE_BYTES, 1U);
  wafer_dmc_cache_range(payload_ddr, WAFER_DMC_RESOURCE_BYTES, 1U);
  const volatile uint64_t *request =
      (const volatile uint64_t *)(uintptr_t)request_ddr;
  volatile uint64_t *record = (volatile uint64_t *)(uintptr_t)output_ddr;
  WaferDMCCase selected = {0};
  uint32_t status = wafer_dmc_decode(request, &selected);
  wafer_dmc_init_record(record, status);
  if (status == WAFER_DMC_STATUS_OK) {
    record[WAFER_DMC_REC_CASE] = selected.case_id;
    record[WAFER_DMC_REC_INPUT_BYTES] = selected.input_bytes;
    record[WAFER_DMC_REC_RESULT_BYTES] = selected.result_bytes;
    record[WAFER_DMC_REC_OUTPUT_SPAN] = selected.output_span;
    record[WAFER_DMC_REC_EXPECTED_INSTRUCTIONS] =
        selected.expected_instructions;
    record[WAFER_DMC_REC_SAMPLE] = request[WAFER_DMC_REQ_SAMPLE];
    record[WAFER_DMC_REC_REQUEST_GUARD] = request[WAFER_DMC_REQ_GUARD];

    wafer_tx81_rdma(payload_ddr, WAFER_DMC_SPM_INPUT, WAFER_DMC_SLOT_BYTES,
                    WAFER_DMC_SLOT_BYTES, 0, 0, 0, 1, 1, 1, Fmt_UINT8);
    wafer_tx81_local_fence();
    wafer_dmc_fill_output();
    WaferDMCPMU before = wafer_dmc_read_pmu();
    if (wafer_dmc_issue(&selected) == 0U) {
      status = WAFER_DMC_STATUS_EXECUTE_FAILED;
    } else {
      WaferDMCPMU after = wafer_dmc_read_pmu();
      record[WAFER_DMC_REC_TDMA_INST_DELTA] =
          (uint32_t)(after.instructions - before.instructions);
      record[WAFER_DMC_REC_TDMA_EXEC_DELTA] =
          after.execution - before.execution;
      record[WAFER_DMC_REC_TDMA_BLOCKING_DELTA] =
          (uint32_t)(after.blocking - before.blocking);
      record[WAFER_DMC_REC_OUTPUT_GUARD_MISMATCHES] =
          wafer_dmc_guard_mismatches(&selected);
      wafer_tx81_wdma(WAFER_DMC_SPM_OUTPUT,
                      output_ddr + WAFER_DMC_OUTPUT_DDR_OFFSET,
                      WAFER_DMC_SLOT_BYTES, WAFER_DMC_SLOT_BYTES, 0, 0, 0, 1, 1,
                      1, Fmt_UINT8);
      wafer_tx81_local_fence();
    }
    record[WAFER_DMC_REC_STATUS] = status;
  }
  wafer_dmc_cache_range(output_ddr, WAFER_DMC_RECORD_WORDS * sizeof(uint64_t),
                        0U);
}
