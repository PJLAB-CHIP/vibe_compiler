#include "instr_def.h"
#include "wafer_spm_calibration_probe_protocol.h"
#include "wafer_tx81_crt.h"

#include <stdint.h>

#define WAFER_SPM_PMU_BASE UINT64_C(0x590000)
#define WAFER_SPM_STABLE_RETRIES 8U
#define WAFER_SPM_GUARD_BYTES 64U

typedef struct WaferSPMCase {
  uint32_t case_id;
  uint64_t address;
  uint32_t transfer_bytes;
  uint32_t kind;
  uint32_t iterations;
  uint32_t slot_stride;
} WaferSPMCase;

typedef struct WaferSPMPMU {
  uint32_t rdma_instructions;
  uint32_t wdma_instructions;
  uint32_t rdma_blocking;
  uint32_t wdma_blocking;
  uint64_t rdma_execution;
  uint64_t wdma_execution;
} WaferSPMPMU;

static const WaferSPMCase wafer_spm_cases[] = {
    {0U, UINT64_C(0x10000), 256U, 0U, 1U, 0U},
    {1U, UINT64_C(0x10000), 8192U, 0U, 1U, 0U},
    {2U, UINT64_C(0x2ef000), 4096U, 0U, 1U, 0U},
    {3U, UINT64_C(0x2eff00), 256U, 0U, 1U, 0U},
    {4U, UINT64_C(0x100000), 4096U, 0U, 1U, 0U},
    {5U, UINT64_C(0x100100), 4096U, 0U, 1U, 0U},
    {6U, UINT64_C(0x100200), 4096U, 0U, 1U, 0U},
    {7U, UINT64_C(0x100400), 4096U, 0U, 1U, 0U},
    {8U, UINT64_C(0x100800), 4096U, 0U, 1U, 0U},
    {9U, UINT64_C(0x101000), 4096U, 0U, 1U, 0U},
    {10U, UINT64_C(0x102000), 4096U, 0U, 1U, 0U},
    {11U, UINT64_C(0x104000), 4096U, 0U, 1U, 0U},
    {12U, UINT64_C(0x108000), 4096U, 0U, 1U, 0U},
    {13U, UINT64_C(0x110000), 4096U, 0U, 1U, 0U},
    {14U, UINT64_C(0x110100), 4096U, 0U, 1U, 0U},
    {15U, UINT64_C(0x180000), 65536U, 0U, 1U, 0U},
    {16U, UINT64_C(0x140000), 4096U, 1U, 1U, 8192U},
    {17U, UINT64_C(0x140000), 4096U, 1U, 4U, 8192U},
    {18U, UINT64_C(0x140000), 4096U, 1U, 5U, 8192U},
    {19U, UINT64_C(0x100040), 4096U, 0U, 1U, 0U},
    {20U, UINT64_C(0x100080), 4096U, 0U, 1U, 0U},
    {21U, UINT64_C(0x1000c0), 4096U, 0U, 1U, 0U},
    {22U, UINT64_C(0x100000), 128U, 0U, 1U, 0U},
    {23U, UINT64_C(0x100000), 384U, 0U, 1U, 0U},
};

static void wafer_spm_cache_range(uint64_t begin, uint32_t bytes,
                                  uint32_t invalidate_only) {
  enum {
    WAFER_SPM_SUPERVISOR_MODE = 1,
    WAFER_SPM_MACHINE_MODE = 3,
    WAFER_SPM_CACHE_LINE_BYTES = 64,
  };
  uintptr_t mode;
  __asm__ volatile("fence" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
  __asm__ volatile("csrr %0, mxstatus" : "=r"(mode));
  mode = (mode >> 30) & 3U;
  for (uintptr_t address = (uintptr_t)begin;
       address < (uintptr_t)begin + bytes;
       address += WAFER_SPM_CACHE_LINE_BYTES) {
    if (mode == WAFER_SPM_MACHINE_MODE) {
      if (invalidate_only != 0)
        __asm__ volatile("dcache.ipa %0" : : "r"(address) : "memory");
      else
        __asm__ volatile("dcache.cipa %0" : : "r"(address) : "memory");
    } else if (mode == WAFER_SPM_SUPERVISOR_MODE) {
      if (invalidate_only != 0)
        __asm__ volatile("dcache.iva %0" : : "r"(address) : "memory");
      else
        __asm__ volatile("dcache.civa %0" : : "r"(address) : "memory");
    }
  }
  __asm__ volatile("sync.is" ::: "memory");
  __asm__ volatile("fence" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
}

static uint32_t wafer_spm_read_pmu32(uint32_t offset) {
  return *(const volatile uint32_t *)(uintptr_t)(WAFER_SPM_PMU_BASE + offset);
}

static uint64_t wafer_spm_read_pmu64(uint32_t low_offset) {
  uint32_t low = 0;
  uint32_t high_after = 0;
  for (uint32_t retry = 0; retry < WAFER_SPM_STABLE_RETRIES; ++retry) {
    uint32_t high_before = wafer_spm_read_pmu32(low_offset + 4U);
    low = wafer_spm_read_pmu32(low_offset);
    high_after = wafer_spm_read_pmu32(low_offset + 4U);
    if (high_before == high_after)
      break;
  }
  return ((uint64_t)high_after << 32) | low;
}

static WaferSPMPMU wafer_spm_read_pmu(void) {
  WaferSPMPMU result;
  result.rdma_instructions = wafer_spm_read_pmu32(GR_PMU_RDMA_INST_NUMS);
  result.wdma_instructions = wafer_spm_read_pmu32(GR_PMU_WDMA_INST_NUMS);
  result.rdma_blocking = wafer_spm_read_pmu32(GR_PMU_RDMA_BLOCKING_TIME);
  result.wdma_blocking = wafer_spm_read_pmu32(GR_PMU_WDMA_BLOCKING_TIME);
  result.rdma_execution = wafer_spm_read_pmu64(GR_PMU_RDMA_EXE_TIME);
  result.wdma_execution = wafer_spm_read_pmu64(GR_PMU_WDMA_EXE_TIME);
  return result;
}

static uint32_t wafer_spm_decode(const volatile uint64_t *request,
                                 WaferSPMCase *selected) {
  if (request[WAFER_SPM_REQ_MAGIC] != WAFER_SPM_REQUEST_MAGIC ||
      request[WAFER_SPM_REQ_SCHEMA_AND_WORDS] !=
          (((uint64_t)WAFER_SPM_SCHEMA << 32) |
           WAFER_SPM_REQUEST_WORDS) ||
      request[WAFER_SPM_REQ_RESOURCE_BYTES] != WAFER_SPM_RESOURCE_BYTES ||
      request[WAFER_SPM_REQ_SLOT_BYTES] != WAFER_SPM_SLOT_BYTES ||
      request[WAFER_SPM_REQ_GUARD] != WAFER_SPM_REQUEST_GUARD)
    return WAFER_SPM_STATUS_BAD_REQUEST;
  uint32_t case_id = (uint32_t)request[WAFER_SPM_REQ_CASE];
  if (case_id >= sizeof(wafer_spm_cases) / sizeof(wafer_spm_cases[0]))
    return WAFER_SPM_STATUS_BAD_REQUEST;
  *selected = wafer_spm_cases[case_id];
  if (request[WAFER_SPM_REQ_ADDRESS] != selected->address ||
      request[WAFER_SPM_REQ_TRANSFER_BYTES] != selected->transfer_bytes ||
      selected->transfer_bytes > WAFER_SPM_SLOT_BYTES ||
      selected->iterations == 0U ||
      selected->iterations * selected->transfer_bytes > WAFER_SPM_SLOT_BYTES ||
      selected->address < WAFER_SPM_ALLOCATABLE_BEGIN ||
      selected->address + selected->transfer_bytes >
          WAFER_SPM_ALLOCATABLE_END)
    return WAFER_SPM_STATUS_BAD_REQUEST;
  if (selected->kind == 1U &&
      selected->address + selected->slot_stride + selected->transfer_bytes >
          WAFER_SPM_ALLOCATABLE_END)
    return WAFER_SPM_STATUS_BAD_REQUEST;
  return WAFER_SPM_STATUS_OK;
}

static void wafer_spm_seed_guards(const WaferSPMCase *selected,
                                  uint64_t canary_ddr) {
  uint32_t slots = selected->kind == 1U && selected->iterations > 1U ? 2U : 1U;
  for (uint32_t slot = 0; slot < slots; ++slot) {
    uint64_t address = selected->address + slot * selected->slot_stride;
    if (address >= WAFER_SPM_ALLOCATABLE_BEGIN + WAFER_SPM_GUARD_BYTES)
      wafer_tx81_rdma_v3(canary_ddr, address - WAFER_SPM_GUARD_BYTES,
                      WAFER_SPM_GUARD_BYTES, WAFER_SPM_GUARD_BYTES, 0, 0, 0, 1,
                      1, 1, Fmt_UINT8, 0U);
    if (address + selected->transfer_bytes + WAFER_SPM_GUARD_BYTES <=
        WAFER_SPM_ALLOCATABLE_END)
      wafer_tx81_rdma_v3(canary_ddr, address + selected->transfer_bytes,
                      WAFER_SPM_GUARD_BYTES, WAFER_SPM_GUARD_BYTES, 0, 0, 0, 1,
                      1, 1, Fmt_UINT8, 0U);
  }
}

static void wafer_spm_readback_guards(const WaferSPMCase *selected,
                                      uint64_t output_ddr) {
  uint64_t readback =
      output_ddr + WAFER_SPM_RECORD_WORDS * sizeof(uint64_t);
  uint32_t slots = selected->kind == 1U && selected->iterations > 1U ? 2U : 1U;
  for (uint32_t slot = 0; slot < slots; ++slot) {
    uint64_t address = selected->address + slot * selected->slot_stride;
    if (address >= WAFER_SPM_ALLOCATABLE_BEGIN + WAFER_SPM_GUARD_BYTES) {
      wafer_tx81_wdma_v3(address - WAFER_SPM_GUARD_BYTES, readback,
                      WAFER_SPM_GUARD_BYTES, WAFER_SPM_GUARD_BYTES, 0, 0, 0, 1,
                      1, 1, Fmt_UINT8, 0U);
      readback += WAFER_SPM_GUARD_BYTES;
    }
    if (address + selected->transfer_bytes + WAFER_SPM_GUARD_BYTES <=
        WAFER_SPM_ALLOCATABLE_END) {
      wafer_tx81_wdma_v3(address + selected->transfer_bytes, readback,
                      WAFER_SPM_GUARD_BYTES, WAFER_SPM_GUARD_BYTES, 0, 0, 0, 1,
                      1, 1, Fmt_UINT8, 0U);
      readback += WAFER_SPM_GUARD_BYTES;
    }
  }
}

static void wafer_spm_init_record(volatile uint64_t *record,
                                  uint32_t status) {
  for (uint32_t index = 0; index < WAFER_SPM_RECORD_WORDS; ++index)
    record[index] = 0;
  record[WAFER_SPM_REC_MAGIC] = WAFER_SPM_RECORD_MAGIC;
  record[WAFER_SPM_REC_SCHEMA_AND_WORDS] =
      ((uint64_t)WAFER_SPM_SCHEMA << 32) | WAFER_SPM_RECORD_WORDS;
  record[WAFER_SPM_REC_STATUS] = status;
  record[WAFER_SPM_REC_OUTPUT_DDR_OFFSET] =
      WAFER_SPM_OUTPUT_DDR_OFFSET;
  record[WAFER_SPM_REC_SLOT_BYTES] = WAFER_SPM_SLOT_BYTES;
  record[WAFER_SPM_REC_RECORD_GUARD] = WAFER_SPM_RECORD_GUARD;
}

__attribute__((visibility("hidden"))) void
wafer_tx81_instruction_family_probe(uint64_t request_ddr,
                                    uint64_t payload_ddr,
                                    uint64_t output_ddr) {
  wafer_spm_cache_range(request_ddr, WAFER_SPM_RESOURCE_BYTES, 1);
  wafer_spm_cache_range(payload_ddr, WAFER_SPM_RESOURCE_BYTES, 1);
  const volatile uint64_t *request =
      (const volatile uint64_t *)(uintptr_t)request_ddr;
  volatile uint64_t *record =
      (volatile uint64_t *)(uintptr_t)output_ddr;
  WaferSPMCase selected = {0};
  uint32_t status = wafer_spm_decode(request, &selected);
  wafer_spm_init_record(record, status);
  if (status == WAFER_SPM_STATUS_OK) {
    record[WAFER_SPM_REC_CASE] = selected.case_id;
    record[WAFER_SPM_REC_ADDRESS] = selected.address;
    record[WAFER_SPM_REC_TRANSFER_BYTES] = selected.transfer_bytes;
    record[WAFER_SPM_REC_SAMPLE] = request[WAFER_SPM_REQ_SAMPLE];
    record[WAFER_SPM_REC_REQUEST_GUARD] =
        request[WAFER_SPM_REQ_GUARD];
    WaferSPMPMU before = wafer_spm_read_pmu();
    wafer_spm_seed_guards(
        &selected,
        request_ddr + WAFER_SPM_REQUEST_WORDS * sizeof(uint64_t));
    for (uint32_t iteration = 0; iteration < selected.iterations;
         ++iteration) {
      uint64_t slot =
          selected.address +
          (selected.kind == 1U ? iteration % 2U * selected.slot_stride : 0U);
      uint64_t payload =
          payload_ddr + (uint64_t)iteration * selected.transfer_bytes;
      uint64_t output =
          output_ddr + WAFER_SPM_OUTPUT_DDR_OFFSET +
          (uint64_t)iteration * selected.transfer_bytes;
      wafer_tx81_rdma_v3(payload, slot, selected.transfer_bytes,
                      selected.transfer_bytes, 0, 0, 0, 1, 1, 1, Fmt_UINT8, 0U);
      wafer_tx81_wdma_v3(slot, output, selected.transfer_bytes,
                      selected.transfer_bytes, 0, 0, 0, 1, 1, 1, Fmt_UINT8, 0U);
    }
    wafer_spm_readback_guards(&selected, output_ddr);
    wafer_tx81_ncc_join(1U);
    WaferSPMPMU after = wafer_spm_read_pmu();
    record[WAFER_SPM_REC_RDMA_INST_DELTA] =
        (uint32_t)(after.rdma_instructions - before.rdma_instructions);
    record[WAFER_SPM_REC_WDMA_INST_DELTA] =
        (uint32_t)(after.wdma_instructions - before.wdma_instructions);
    record[WAFER_SPM_REC_RDMA_EXEC_DELTA] =
        after.rdma_execution - before.rdma_execution;
    record[WAFER_SPM_REC_WDMA_EXEC_DELTA] =
        after.wdma_execution - before.wdma_execution;
    record[WAFER_SPM_REC_RDMA_BLOCKING_DELTA] =
        (uint32_t)(after.rdma_blocking - before.rdma_blocking);
    record[WAFER_SPM_REC_WDMA_BLOCKING_DELTA] =
        (uint32_t)(after.wdma_blocking - before.wdma_blocking);
  }
  wafer_spm_cache_range(output_ddr,
                        WAFER_SPM_RECORD_WORDS * sizeof(uint64_t), 0);
}
