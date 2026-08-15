#include "Wafer/ABI/Tx81DirectDTEStatusABI.h"
#include "instr_adapter_plat.h"
#include "instr_def.h"

#include <stddef.h>
#include <stdint.h>

#define WAFER_TX81_NCC_PMU_BASE UINT64_C(0x590000)
#define WAFER_TX81_NCC_WORKER_COUNT 3
#define WAFER_TX81_PMU_COUNTER_COUNT 8
#define WAFER_TX81_PMU_STABLE_RETRIES 8
#define WAFER_TX81_PROBE_WORD_COUNT 32
#define WAFER_TX81_PROBE_MAGIC UINT64_C(0x3130554d50464157)

#ifndef WAFER_PMU_SLOTS_PER_TILE
#error "WAFER_PMU_SLOTS_PER_TILE must match the package entry layout"
#endif

__attribute__((visibility("hidden")))
const uint64_t wafer_pmu_slots_per_tile = WAFER_PMU_SLOTS_PER_TILE;

enum WaferTx81ProbeWord {
  WAFER_PROBE_MAGIC = 0,
  WAFER_PROBE_WORD_COUNT = 1,
  WAFER_PROBE_STABLE_COUNTER_MASK = 2,
  WAFER_PROBE_PMU_ENABLE = 3,
  WAFER_PROBE_SERIAL_MODE_WORKER0 = 4,
  WAFER_PROBE_SERIAL_MODE_WORKER1 = 5,
  WAFER_PROBE_SERIAL_MODE_WORKER2 = 6,
  WAFER_PROBE_CONTROL_WORKER0 = 7,
  WAFER_PROBE_CONTROL_WORKER1 = 8,
  WAFER_PROBE_CONTROL_WORKER2 = 9,
  WAFER_PROBE_STATISTICS_WINDOW = 10,
  WAFER_PROBE_FULL_EXECUTION = 11,
  WAFER_PROBE_CT_EXECUTION = 12,
  WAFER_PROBE_NE_EXECUTION = 13,
  WAFER_PROBE_RDMA_EXECUTION = 14,
  WAFER_PROBE_WDMA_EXECUTION = 15,
  WAFER_PROBE_TDMA_EXECUTION = 16,
  WAFER_PROBE_SCALAR_EXECUTION = 17,
  WAFER_PROBE_CT_INSTRUCTIONS = 18,
  WAFER_PROBE_NE_INSTRUCTIONS = 19,
  WAFER_PROBE_RDMA_INSTRUCTIONS = 20,
  WAFER_PROBE_WDMA_INSTRUCTIONS = 21,
  WAFER_PROBE_TDMA_INSTRUCTIONS = 22,
  WAFER_PROBE_SCALAR_INSTRUCTIONS = 23,
  WAFER_PROBE_CT_BLOCKING = 24,
  WAFER_PROBE_NE_BLOCKING = 25,
  WAFER_PROBE_RDMA_BLOCKING = 26,
  WAFER_PROBE_WDMA_BLOCKING = 27,
  WAFER_PROBE_TDMA_BLOCKING = 28,
  WAFER_PROBE_SCALAR_BLOCKING = 29,
  WAFER_PROBE_PMU_BASE = 30,
  WAFER_PROBE_NCC_BASE = 31,
};

static const uint32_t
    wafer_tx81_execution_offsets[WAFER_TX81_PMU_COUNTER_COUNT] = {
        GR_PMU_STATISTICS_WINDOW, GR_PMU_FU_EXE_TIME,     GR_PMU_CT_EXE_TIME,
        GR_PMU_NE_EXE_TIME,       GR_PMU_RDMA_EXE_TIME,   GR_PMU_WDMA_EXE_TIME,
        GR_PMU_TDMA_EXE_TIME,     GR_PMU_SCALAR_EXE_TIME,
};

static const uint32_t wafer_tx81_instruction_offsets[6] = {
    GR_PMU_CT_INST_NUMS,   GR_PMU_NE_INST_NUMS,   GR_PMU_RDMA_INST_NUMS,
    GR_PMU_WDMA_INST_NUMS, GR_PMU_TDMA_INST_NUMS, GR_PMU_SCALAR_INST_NUMS,
};

static const uint32_t wafer_tx81_blocking_offsets[6] = {
    GR_PMU_CT_BLOCKING_TIME,   GR_PMU_NE_BLOCKING_TIME,
    GR_PMU_RDMA_BLOCKING_TIME, GR_PMU_WDMA_BLOCKING_TIME,
    GR_PMU_TDMA_BLOCKING_TIME, GR_PMU_SCALAR_BLOCKING_TIME,
};

__attribute__((noinline, visibility("hidden"))) uint32_t
wafer_tx81_probe_read_pmu32(uint32_t offset) {
  const volatile uint32_t *address =
      (const volatile uint32_t *)(uintptr_t)(WAFER_TX81_NCC_PMU_BASE + offset);
  return *address;
}

static uint64_t wafer_tx81_probe_read_pmu64(uint32_t low_offset,
                                            uint64_t stability_bit,
                                            uint64_t *stable_mask) {
  uint32_t low = 0;
  uint32_t high_after = 0;
  for (uint32_t retry = 0; retry < WAFER_TX81_PMU_STABLE_RETRIES; ++retry) {
    uint32_t high_before = wafer_tx81_probe_read_pmu32(low_offset + 4);
    low = wafer_tx81_probe_read_pmu32(low_offset);
    high_after = wafer_tx81_probe_read_pmu32(low_offset + 4);
    if (high_before == high_after) {
      *stable_mask |= stability_bit;
      break;
    }
  }
  return ((uint64_t)high_after << 32) | low;
}

static void wafer_tx81_probe_write_output(uint64_t output_ddr) {
  enum {
    WAFER_TX81_SUPERVISOR_MODE = 1,
    WAFER_TX81_MACHINE_MODE = 3,
  };
  uintptr_t mode;
  __asm__ volatile("fence" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
  __asm__ volatile("csrr %0, mxstatus" : "=r"(mode));
  mode = (mode >> 30) & 3U;
  for (uintptr_t address = output_ddr;
       address < output_ddr + WAFER_TX81_PROBE_WORD_COUNT * sizeof(uint64_t);
       address += WAFER_TX81_DIRECT_DTE_STATUS_CACHE_LINE_BYTES) {
    if (mode == WAFER_TX81_MACHINE_MODE)
      __asm__ volatile("dcache.cipa %0" : : "r"(address) : "memory");
    else if (mode == WAFER_TX81_SUPERVISOR_MODE)
      __asm__ volatile("dcache.civa %0" : : "r"(address) : "memory");
  }
  __asm__ volatile("sync.is" ::: "memory");
  __asm__ volatile("fence" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
}

__attribute__((visibility("hidden"))) void
wafer_tx81_board_probe(uint64_t output_ddr) {
  volatile uint64_t *record = (volatile uint64_t *)(uintptr_t)output_ddr;
  uint64_t stable_mask = 0;

  __asm__ volatile("fence iorw, iorw" ::: "memory");

  record[WAFER_PROBE_MAGIC] = WAFER_TX81_PROBE_MAGIC;
  record[WAFER_PROBE_WORD_COUNT] = WAFER_TX81_PROBE_WORD_COUNT;
  record[WAFER_PROBE_STABLE_COUNTER_MASK] = 0;
  record[WAFER_PROBE_PMU_ENABLE] = wafer_tx81_probe_read_pmu32(GR_PMU_EN);

  for (size_t worker = 0; worker < WAFER_TX81_NCC_WORKER_COUNT; ++worker) {
    record[WAFER_PROBE_SERIAL_MODE_WORKER0 + worker] =
        get_ncc_reg(worker, GR_CSR_SERIAL_MODE_ADDR);
    record[WAFER_PROBE_CONTROL_WORKER0 + worker] =
        get_ncc_reg(worker, GR_CSR_CONTROL_ADDR);
  }

  for (size_t index = 0; index < WAFER_TX81_PMU_COUNTER_COUNT; ++index)
    record[WAFER_PROBE_STATISTICS_WINDOW + index] =
        wafer_tx81_probe_read_pmu64(wafer_tx81_execution_offsets[index],
                                    UINT64_C(1) << index, &stable_mask);

  for (size_t index = 0; index < 6; ++index) {
    record[WAFER_PROBE_CT_INSTRUCTIONS + index] =
        wafer_tx81_probe_read_pmu32(wafer_tx81_instruction_offsets[index]);
    record[WAFER_PROBE_CT_BLOCKING + index] =
        wafer_tx81_probe_read_pmu32(wafer_tx81_blocking_offsets[index]);
  }

  record[WAFER_PROBE_STABLE_COUNTER_MASK] = stable_mask;
  record[WAFER_PROBE_PMU_BASE] = WAFER_TX81_NCC_PMU_BASE;
  record[WAFER_PROBE_NCC_BASE] = NCC_ADDR;

  wafer_tx81_probe_write_output(output_ddr);
}
