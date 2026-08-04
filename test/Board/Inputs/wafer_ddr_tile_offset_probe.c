#include "instr_def.h"
#include "wafer_ddr_tile_offset_probe_protocol.h"
#include "wafer_tx81_crt.h"

#include <stdint.h>

#define WAFER_DDR_TILE_PMU_BASE UINT64_C(0x590000)
#define WAFER_DDR_TILE_PMU_STABLE_RETRIES 8U
#define WAFER_DDR_TILE_CACHE_LINE_BYTES 64U

extern void hrt_barrier(void);

typedef struct WaferDDRTilePMU {
  uint32_t rdma_instructions;
  uint32_t wdma_instructions;
  uint32_t rdma_blocking;
  uint32_t wdma_blocking;
  uint64_t rdma_execution;
  uint64_t wdma_execution;
  uint64_t fu_execution;
  uint64_t statistics_window;
} WaferDDRTilePMU;

static const uint32_t wafer_ddr_tile_offsets[WAFER_DDR_TILE_OFFSET_COUNT] = {
    0U,     256U,   512U,   1024U,   2048U,   4096U,   8192U,
    16384U, 32768U, 65536U, 131072U, 262144U, 524288U, 1048576U,
};

static const uint32_t
    wafer_ddr_tile_conflict_bases[WAFER_DDR_TILE_CONFLICT_BASE_COUNT] = {
        0x20000U,
        0x40000U,
};

static const uint32_t
    wafer_ddr_tile_conflict_offsets[WAFER_DDR_TILE_CONFLICT_OFFSET_COUNT] = {
        0U,
        4096U,
        32768U,
};

static const uint32_t
    wafer_ddr_tile_conflict_transfers[WAFER_DDR_TILE_CONFLICT_TRANSFER_COUNT] =
        {
            256U,
            4096U,
};

static void wafer_ddr_tile_cache_range(uint64_t begin, uint32_t bytes,
                                       uint32_t invalidate_only) {
  enum {
    WAFER_DDR_TILE_SUPERVISOR_MODE = 1,
    WAFER_DDR_TILE_MACHINE_MODE = 3,
  };
  uintptr_t mode;
  __asm__ volatile("fence" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
  __asm__ volatile("csrr %0, mxstatus" : "=r"(mode));
  mode = (mode >> 30) & 3U;
  for (uintptr_t address = (uintptr_t)begin; address < (uintptr_t)begin + bytes;
       address += WAFER_DDR_TILE_CACHE_LINE_BYTES) {
    if (mode == WAFER_DDR_TILE_MACHINE_MODE) {
      if (invalidate_only != 0U)
        __asm__ volatile("dcache.ipa %0" : : "r"(address) : "memory");
      else
        __asm__ volatile("dcache.cipa %0" : : "r"(address) : "memory");
    } else if (mode == WAFER_DDR_TILE_SUPERVISOR_MODE) {
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

static uint32_t wafer_ddr_tile_read32(uint32_t offset) {
  return *(const volatile uint32_t *)(uintptr_t)(WAFER_DDR_TILE_PMU_BASE +
                                                 offset);
}

static uint64_t wafer_ddr_tile_read64(uint32_t low_offset) {
  uint32_t low = 0;
  uint32_t high_after = 0;
  for (uint32_t retry = 0; retry < WAFER_DDR_TILE_PMU_STABLE_RETRIES; ++retry) {
    uint32_t high_before = wafer_ddr_tile_read32(low_offset + 4U);
    low = wafer_ddr_tile_read32(low_offset);
    high_after = wafer_ddr_tile_read32(low_offset + 4U);
    if (high_before == high_after)
      break;
  }
  return ((uint64_t)high_after << 32) | low;
}

static WaferDDRTilePMU wafer_ddr_tile_read_pmu(void) {
  WaferDDRTilePMU result;
  result.rdma_instructions = wafer_ddr_tile_read32(GR_PMU_RDMA_INST_NUMS);
  result.wdma_instructions = wafer_ddr_tile_read32(GR_PMU_WDMA_INST_NUMS);
  result.rdma_blocking = wafer_ddr_tile_read32(GR_PMU_RDMA_BLOCKING_TIME);
  result.wdma_blocking = wafer_ddr_tile_read32(GR_PMU_WDMA_BLOCKING_TIME);
  result.rdma_execution = wafer_ddr_tile_read64(GR_PMU_RDMA_EXE_TIME);
  result.wdma_execution = wafer_ddr_tile_read64(GR_PMU_WDMA_EXE_TIME);
  result.fu_execution = wafer_ddr_tile_read64(GR_PMU_FU_EXE_TIME);
  result.statistics_window = wafer_ddr_tile_read64(GR_PMU_STATISTICS_WINDOW);
  return result;
}

static uint64_t wafer_ddr_tile_cycle(void) {
  uint64_t cycle;
  __asm__ volatile("rdcycle %0" : "=r"(cycle));
  return cycle;
}

static void wafer_ddr_tile_rdma(uint64_t source, uint64_t destination,
                                uint32_t bytes) {
  wafer_tx81_rdma_v3(source, destination, bytes, bytes, 0U, 0U, 0U, 1U, 1U, 1U,
                  Fmt_UINT8, 0U);
}

static void wafer_ddr_tile_wdma(uint64_t source, uint64_t destination,
                                uint32_t bytes) {
  wafer_tx81_wdma_v3(source, destination, bytes, bytes, 0U, 0U, 0U, 1U, 1U, 1U,
                  Fmt_UINT8, 0U);
}

static uint64_t wafer_ddr_tile_identity(uint32_t rank, uint32_t allocation,
                                        uint32_t direction,
                                        uint32_t offset_index,
                                        uint32_t cell_sample) {
  return (uint64_t)rank | ((uint64_t)allocation << 8) |
         ((uint64_t)direction << 16) | ((uint64_t)offset_index << 24) |
         ((uint64_t)cell_sample << 32);
}

static void wafer_ddr_tile_write_row(
    volatile uint64_t *row, uint32_t rank, uint32_t allocation,
    uint32_t direction, uint32_t offset_index, uint32_t cell_sample,
    uint64_t input_base, uint64_t output_base, uint64_t ddr_address,
    uint64_t spm_address, uint64_t completion_cycles,
    const WaferDDRTilePMU *before, const WaferDDRTilePMU *after,
    uint32_t archive_offset) {
  row[WAFER_DDR_TILE_ROW_MAGIC_WORD] = WAFER_DDR_TILE_ROW_MAGIC;
  row[WAFER_DDR_TILE_ROW_IDENTITY] = wafer_ddr_tile_identity(
      rank, allocation, direction, offset_index, cell_sample);
  row[WAFER_DDR_TILE_ROW_OFFSET] = wafer_ddr_tile_offsets[offset_index];
  row[WAFER_DDR_TILE_ROW_INPUT_BASE] = input_base;
  row[WAFER_DDR_TILE_ROW_OUTPUT_BASE] = output_base;
  row[WAFER_DDR_TILE_ROW_DDR_ADDRESS] = ddr_address;
  row[WAFER_DDR_TILE_ROW_SPM_ADDRESS] = spm_address;
  row[WAFER_DDR_TILE_ROW_COMPLETION_CYCLES] = completion_cycles;
  row[WAFER_DDR_TILE_ROW_INST_DELTA] =
      direction == WAFER_DDR_TILE_DIRECTION_RDMA
          ? (uint32_t)(after->rdma_instructions - before->rdma_instructions)
          : (uint32_t)(after->wdma_instructions - before->wdma_instructions);
  row[WAFER_DDR_TILE_ROW_BLOCKING_DELTA] =
      direction == WAFER_DDR_TILE_DIRECTION_RDMA
          ? (uint32_t)(after->rdma_blocking - before->rdma_blocking)
          : (uint32_t)(after->wdma_blocking - before->wdma_blocking);
  row[WAFER_DDR_TILE_ROW_ENGINE_EXEC_DELTA] =
      direction == WAFER_DDR_TILE_DIRECTION_RDMA
          ? after->rdma_execution - before->rdma_execution
          : after->wdma_execution - before->wdma_execution;
  row[WAFER_DDR_TILE_ROW_FU_EXEC_DELTA] =
      after->fu_execution - before->fu_execution;
  row[WAFER_DDR_TILE_ROW_WINDOW_DELTA] =
      after->statistics_window - before->statistics_window;
  row[WAFER_DDR_TILE_ROW_PAYLOAD_BYTES] = WAFER_DDR_TILE_PAYLOAD_BYTES;
  row[WAFER_DDR_TILE_ROW_ARCHIVE_OFFSET] = archive_offset;
  row[WAFER_DDR_TILE_ROW_GUARD_WORD] = WAFER_DDR_TILE_ROW_GUARD;
}

static void wafer_ddr_tile_measure_rdma(uint32_t rank, uint32_t allocation,
                                        uint32_t offset_index,
                                        uint32_t cell_sample,
                                        uint64_t input_base,
                                        uint64_t output_base,
                                        volatile uint64_t *row) {
  uint64_t canary_source = input_base + WAFER_DDR_TILE_CANARY_SOURCE_OFFSET;
  uint64_t source = input_base + WAFER_DDR_TILE_SWEEP_BASE +
                    wafer_ddr_tile_offsets[offset_index];
  uint32_t archive_offset =
      WAFER_DDR_TILE_ARCHIVE_BASE +
      (offset_index * WAFER_DDR_TILE_SAMPLES + cell_sample) *
          WAFER_DDR_TILE_SLOT_BYTES;
  wafer_ddr_tile_rdma(canary_source, WAFER_DDR_TILE_SPM_GUARDED,
                      WAFER_DDR_TILE_SLOT_BYTES);
  wafer_tx81_ncc_join(1U);

  WaferDDRTilePMU before = wafer_ddr_tile_read_pmu();
  uint64_t begin = wafer_ddr_tile_cycle();
  wafer_ddr_tile_rdma(source,
                      WAFER_DDR_TILE_SPM_GUARDED + WAFER_DDR_TILE_GUARD_BYTES,
                      WAFER_DDR_TILE_PAYLOAD_BYTES);
  wafer_tx81_ncc_join(1U);
  uint64_t completion_cycles = wafer_ddr_tile_cycle() - begin;
  WaferDDRTilePMU after = wafer_ddr_tile_read_pmu();

  wafer_ddr_tile_wdma(WAFER_DDR_TILE_SPM_GUARDED, output_base + archive_offset,
                      WAFER_DDR_TILE_SLOT_BYTES);
  wafer_tx81_ncc_join(1U);
  wafer_ddr_tile_write_row(
      row, rank, allocation, WAFER_DDR_TILE_DIRECTION_RDMA, offset_index,
      cell_sample, input_base, output_base, source,
      WAFER_DDR_TILE_SPM_GUARDED + WAFER_DDR_TILE_GUARD_BYTES,
      completion_cycles, &before, &after, archive_offset);
}

static void wafer_ddr_tile_measure_wdma(uint32_t rank, uint32_t allocation,
                                        uint32_t offset_index,
                                        uint32_t cell_sample,
                                        uint64_t input_base,
                                        uint64_t output_base,
                                        volatile uint64_t *row) {
  uint64_t canary_source = input_base + WAFER_DDR_TILE_CANARY_SOURCE_OFFSET;
  uint64_t source = input_base + WAFER_DDR_TILE_SWEEP_BASE +
                    wafer_ddr_tile_offsets[offset_index];
  uint64_t destination = output_base + WAFER_DDR_TILE_SWEEP_BASE +
                         wafer_ddr_tile_offsets[offset_index];
  uint32_t archive_offset =
      WAFER_DDR_TILE_ARCHIVE_BASE +
      (WAFER_DDR_TILE_OFFSET_COUNT * WAFER_DDR_TILE_SAMPLES +
       offset_index * WAFER_DDR_TILE_SAMPLES + cell_sample) *
          WAFER_DDR_TILE_SLOT_BYTES;

  wafer_ddr_tile_rdma(canary_source, WAFER_DDR_TILE_SPM_GUARDED,
                      WAFER_DDR_TILE_SLOT_BYTES);
  wafer_ddr_tile_wdma(WAFER_DDR_TILE_SPM_GUARDED,
                      destination - WAFER_DDR_TILE_GUARD_BYTES,
                      WAFER_DDR_TILE_GUARD_BYTES);
  wafer_ddr_tile_wdma(WAFER_DDR_TILE_SPM_GUARDED,
                      destination + WAFER_DDR_TILE_PAYLOAD_BYTES,
                      WAFER_DDR_TILE_GUARD_BYTES);
  wafer_ddr_tile_rdma(source, WAFER_DDR_TILE_SPM_PAYLOAD,
                      WAFER_DDR_TILE_PAYLOAD_BYTES);
  wafer_tx81_ncc_join(1U);

  WaferDDRTilePMU before = wafer_ddr_tile_read_pmu();
  uint64_t begin = wafer_ddr_tile_cycle();
  wafer_ddr_tile_wdma(WAFER_DDR_TILE_SPM_PAYLOAD, destination,
                      WAFER_DDR_TILE_PAYLOAD_BYTES);
  wafer_tx81_ncc_join(1U);
  uint64_t completion_cycles = wafer_ddr_tile_cycle() - begin;
  WaferDDRTilePMU after = wafer_ddr_tile_read_pmu();

  wafer_ddr_tile_rdma(destination - WAFER_DDR_TILE_GUARD_BYTES,
                      WAFER_DDR_TILE_SPM_READBACK, WAFER_DDR_TILE_SLOT_BYTES);
  wafer_ddr_tile_wdma(WAFER_DDR_TILE_SPM_READBACK, output_base + archive_offset,
                      WAFER_DDR_TILE_SLOT_BYTES);
  wafer_tx81_ncc_join(1U);
  wafer_ddr_tile_write_row(row, rank, allocation, WAFER_DDR_TILE_DIRECTION_WDMA,
                           offset_index, cell_sample, input_base, output_base,
                           destination, WAFER_DDR_TILE_SPM_PAYLOAD,
                           completion_cycles, &before, &after, archive_offset);
}

static uint32_t wafer_ddr_tile_conflict_local_ordinal(
    uint32_t base_index, uint32_t offset_index, uint32_t issue_order_index,
    uint32_t schedule_index, uint32_t cell_sample) {
  return (((base_index * WAFER_DDR_TILE_CONFLICT_OFFSET_COUNT + offset_index) *
               WAFER_DDR_TILE_CONFLICT_ORDER_COUNT +
           issue_order_index) *
              WAFER_DDR_TILE_CONFLICT_SCHEDULE_COUNT +
          schedule_index) *
             WAFER_DDR_TILE_CONFLICT_SAMPLES +
         cell_sample;
}

static uint32_t
wafer_ddr_tile_conflict_ordinal(uint32_t allocation, uint32_t base_index,
                                uint32_t offset_index, uint32_t transfer_index,
                                uint32_t issue_order_index,
                                uint32_t schedule_index, uint32_t cell_sample) {
  return (((((allocation * WAFER_DDR_TILE_CONFLICT_BASE_COUNT + base_index) *
                 WAFER_DDR_TILE_CONFLICT_OFFSET_COUNT +
             offset_index) *
                WAFER_DDR_TILE_CONFLICT_TRANSFER_COUNT +
            transfer_index) *
               WAFER_DDR_TILE_CONFLICT_ORDER_COUNT +
           issue_order_index) *
              WAFER_DDR_TILE_CONFLICT_SCHEDULE_COUNT +
          schedule_index) *
             WAFER_DDR_TILE_CONFLICT_SAMPLES +
         cell_sample;
}

static uint64_t wafer_ddr_tile_conflict_identity(
    uint32_t rank, uint32_t allocation, uint32_t base_index,
    uint32_t offset_index, uint32_t transfer_index, uint32_t issue_order_index,
    uint32_t schedule_index, uint32_t cell_sample) {
  return (uint64_t)rank | ((uint64_t)allocation << 8) |
         ((uint64_t)base_index << 16) | ((uint64_t)offset_index << 24) |
         ((uint64_t)transfer_index << 32) |
         ((uint64_t)issue_order_index << 40) |
         ((uint64_t)schedule_index << 48) | ((uint64_t)cell_sample << 56);
}

static uint32_t wafer_ddr_tile_conflict_archive_offset(uint32_t transfer_index,
                                                       uint32_t local_ordinal) {
  uint32_t transfer_bytes = wafer_ddr_tile_conflict_transfers[transfer_index];
  uint32_t guarded_slot_bytes =
      transfer_bytes + 2U * WAFER_DDR_TILE_GUARD_BYTES;
  uint32_t archive_base = transfer_index == 0U
                              ? WAFER_DDR_TILE_CONFLICT_SMALL_ARCHIVE_BASE
                              : WAFER_DDR_TILE_CONFLICT_LARGE_ARCHIVE_BASE;
  return archive_base + local_ordinal * 2U * guarded_slot_bytes;
}

static void wafer_ddr_tile_write_conflict_row(
    volatile uint64_t *row, uint32_t rank, uint32_t allocation,
    uint32_t base_index, uint32_t offset_index, uint32_t transfer_index,
    uint32_t issue_order_index, uint32_t schedule_index, uint32_t cell_sample,
    uint32_t schedule_position,
    uint64_t input_base, uint64_t output_base, uint64_t address_a,
    uint64_t address_b, uint32_t transfer_bytes, uint32_t archive_a,
    uint32_t archive_b, uint64_t completion_cycles,
    const WaferDDRTilePMU *before, const WaferDDRTilePMU *after) {
  row[WAFER_DDR_TILE_CONFLICT_ROW_MAGIC_WORD] =
      WAFER_DDR_TILE_CONFLICT_ROW_MAGIC;
  row[WAFER_DDR_TILE_CONFLICT_ROW_IDENTITY] = wafer_ddr_tile_conflict_identity(
      rank, allocation, base_index, offset_index, transfer_index,
      issue_order_index, schedule_index, cell_sample);
  row[WAFER_DDR_TILE_CONFLICT_ROW_OFFSET] =
      wafer_ddr_tile_conflict_offsets[offset_index];
  row[WAFER_DDR_TILE_CONFLICT_ROW_INPUT_BASE] = input_base;
  row[WAFER_DDR_TILE_CONFLICT_ROW_OUTPUT_BASE] = output_base;
  row[WAFER_DDR_TILE_CONFLICT_ROW_ADDRESS_A] = address_a;
  row[WAFER_DDR_TILE_CONFLICT_ROW_ADDRESS_B] = address_b;
  row[WAFER_DDR_TILE_CONFLICT_ROW_SPM_A] =
      WAFER_DDR_TILE_CONFLICT_SPM_A + WAFER_DDR_TILE_GUARD_BYTES;
  row[WAFER_DDR_TILE_CONFLICT_ROW_SPM_B] =
      WAFER_DDR_TILE_CONFLICT_SPM_B + WAFER_DDR_TILE_GUARD_BYTES;
  row[WAFER_DDR_TILE_CONFLICT_ROW_TRANSFER_BYTES] = transfer_bytes;
  row[WAFER_DDR_TILE_CONFLICT_ROW_ARCHIVE_A] = archive_a;
  row[WAFER_DDR_TILE_CONFLICT_ROW_ARCHIVE_B] = archive_b;
  row[WAFER_DDR_TILE_CONFLICT_ROW_COMPLETION_CYCLES] = completion_cycles;
  row[WAFER_DDR_TILE_CONFLICT_ROW_INST_DELTA] =
      (uint32_t)(after->rdma_instructions - before->rdma_instructions);
  row[WAFER_DDR_TILE_CONFLICT_ROW_BLOCKING_DELTA] =
      (uint32_t)(after->rdma_blocking - before->rdma_blocking);
  row[WAFER_DDR_TILE_CONFLICT_ROW_ENGINE_EXEC_DELTA] =
      after->rdma_execution - before->rdma_execution;
  row[WAFER_DDR_TILE_CONFLICT_ROW_FU_EXEC_DELTA] =
      after->fu_execution - before->fu_execution;
  row[WAFER_DDR_TILE_CONFLICT_ROW_WINDOW_DELTA] =
      after->statistics_window - before->statistics_window;
  row[WAFER_DDR_TILE_CONFLICT_ROW_BASE_OFFSET] =
      wafer_ddr_tile_conflict_bases[base_index];
  row[WAFER_DDR_TILE_CONFLICT_ROW_ALLOCATION] = allocation;
  row[WAFER_DDR_TILE_CONFLICT_ROW_SCHEDULE] = schedule_index + 1U;
  row[WAFER_DDR_TILE_CONFLICT_ROW_ISSUE_ORDER] = issue_order_index + 1U;
  row[WAFER_DDR_TILE_CONFLICT_ROW_SAMPLE] = cell_sample;
  row[WAFER_DDR_TILE_CONFLICT_ROW_GUARD_WORD] =
      WAFER_DDR_TILE_CONFLICT_ROW_GUARD;
  row[WAFER_DDR_TILE_CONFLICT_ROW_SCHEDULE_POSITION] = schedule_position;
}

static void wafer_ddr_tile_measure_conflict_pair(
    uint32_t rank, uint32_t allocation, uint32_t base_index,
    uint32_t offset_index, uint32_t transfer_index, uint32_t issue_order_index,
    uint32_t schedule_index, uint32_t cell_sample, uint32_t schedule_position,
    uint64_t input_base, uint64_t output_base, volatile uint64_t *row) {
  uint32_t transfer_bytes = wafer_ddr_tile_conflict_transfers[transfer_index];
  uint32_t guarded_slot_bytes =
      transfer_bytes + 2U * WAFER_DDR_TILE_GUARD_BYTES;
  uint64_t address_a = input_base + wafer_ddr_tile_conflict_bases[base_index];
  uint64_t address_b = address_a + WAFER_DDR_TILE_CONFLICT_PAIR_GAP +
                       wafer_ddr_tile_conflict_offsets[offset_index];
  uint32_t local_ordinal = wafer_ddr_tile_conflict_local_ordinal(
      base_index, offset_index, issue_order_index, schedule_index, cell_sample);
  uint32_t archive_a =
      wafer_ddr_tile_conflict_archive_offset(transfer_index, local_ordinal);
  uint32_t archive_b = archive_a + guarded_slot_bytes;
  uint64_t canary_source = input_base + WAFER_DDR_TILE_CANARY_SOURCE_OFFSET;

  wafer_ddr_tile_rdma(canary_source, WAFER_DDR_TILE_CONFLICT_SPM_A,
                      guarded_slot_bytes);
  wafer_ddr_tile_rdma(canary_source, WAFER_DDR_TILE_CONFLICT_SPM_B,
                      guarded_slot_bytes);
  wafer_tx81_ncc_join(1U);

  WaferDDRTilePMU before = wafer_ddr_tile_read_pmu();
  uint64_t begin = wafer_ddr_tile_cycle();
  if (issue_order_index == 0U) {
    wafer_ddr_tile_rdma(
        address_a, WAFER_DDR_TILE_CONFLICT_SPM_A + WAFER_DDR_TILE_GUARD_BYTES,
        transfer_bytes);
    if (schedule_index == 0U)
      wafer_tx81_ncc_join(1U);
    wafer_ddr_tile_rdma(
        address_b, WAFER_DDR_TILE_CONFLICT_SPM_B + WAFER_DDR_TILE_GUARD_BYTES,
        transfer_bytes);
  } else {
    wafer_ddr_tile_rdma(
        address_b, WAFER_DDR_TILE_CONFLICT_SPM_B + WAFER_DDR_TILE_GUARD_BYTES,
        transfer_bytes);
    if (schedule_index == 0U)
      wafer_tx81_ncc_join(1U);
    wafer_ddr_tile_rdma(
        address_a, WAFER_DDR_TILE_CONFLICT_SPM_A + WAFER_DDR_TILE_GUARD_BYTES,
        transfer_bytes);
  }
  wafer_tx81_ncc_join(1U);
  uint64_t completion_cycles = wafer_ddr_tile_cycle() - begin;
  WaferDDRTilePMU after = wafer_ddr_tile_read_pmu();

  wafer_ddr_tile_wdma(WAFER_DDR_TILE_CONFLICT_SPM_A, output_base + archive_a,
                      guarded_slot_bytes);
  wafer_ddr_tile_wdma(WAFER_DDR_TILE_CONFLICT_SPM_B, output_base + archive_b,
                      guarded_slot_bytes);
  wafer_tx81_ncc_join(1U);
  wafer_ddr_tile_write_conflict_row(
      row, rank, allocation, base_index, offset_index, transfer_index,
      issue_order_index, schedule_index, cell_sample, schedule_position,
      input_base, output_base, address_a, address_b, transfer_bytes, archive_a,
      archive_b,
      completion_cycles, &before, &after);
}

static uint32_t
wafer_ddr_tile_request_status(uint32_t rank, const volatile uint64_t *request) {
  if (request[WAFER_DDR_TILE_REQ_MAGIC] != WAFER_DDR_TILE_REQUEST_MAGIC ||
      request[WAFER_DDR_TILE_REQ_SCHEMA] != WAFER_DDR_TILE_SCHEMA ||
      request[WAFER_DDR_TILE_REQ_RANK] != rank ||
      request[WAFER_DDR_TILE_REQ_RESOURCE_BYTES] !=
          WAFER_DDR_TILE_RESOURCE_BYTES ||
      request[WAFER_DDR_TILE_REQ_PAYLOAD_BYTES] !=
          WAFER_DDR_TILE_PAYLOAD_BYTES ||
      request[WAFER_DDR_TILE_REQ_ALLOCATIONS] != WAFER_DDR_TILE_ALLOCATIONS ||
      request[WAFER_DDR_TILE_REQ_OFFSET_COUNT] != WAFER_DDR_TILE_OFFSET_COUNT ||
      request[WAFER_DDR_TILE_REQ_GUARD] != WAFER_DDR_TILE_REQUEST_GUARD)
    return WAFER_DDR_TILE_STATUS_BAD_REQUEST;
  uint32_t mode = (uint32_t)request[WAFER_DDR_TILE_REQ_MODE];
  if (mode == WAFER_DDR_TILE_MODE_OFFSET) {
    if (request[WAFER_DDR_TILE_REQ_CONFLICT_SAMPLES] != 0U ||
        request[WAFER_DDR_TILE_REQ_CONFLICT_BASE_COUNT] != 0U ||
        request[WAFER_DDR_TILE_REQ_CONFLICT_OFFSET_COUNT] != 0U ||
        request[WAFER_DDR_TILE_REQ_CONFLICT_TRANSFER_COUNT] != 0U ||
        request[WAFER_DDR_TILE_REQ_CONFLICT_ORDER_COUNT] != 0U ||
        request[WAFER_DDR_TILE_REQ_CONFLICT_SCHEDULE_COUNT] != 0U)
      return WAFER_DDR_TILE_STATUS_BAD_REQUEST;
  } else if (mode == WAFER_DDR_TILE_MODE_CONFLICT_EQUIVALENCE) {
    if (request[WAFER_DDR_TILE_REQ_CONFLICT_SAMPLES] !=
            WAFER_DDR_TILE_CONFLICT_SAMPLES ||
        request[WAFER_DDR_TILE_REQ_CONFLICT_BASE_COUNT] !=
            WAFER_DDR_TILE_CONFLICT_BASE_COUNT ||
        request[WAFER_DDR_TILE_REQ_CONFLICT_OFFSET_COUNT] !=
            WAFER_DDR_TILE_CONFLICT_OFFSET_COUNT ||
        request[WAFER_DDR_TILE_REQ_CONFLICT_TRANSFER_COUNT] !=
            WAFER_DDR_TILE_CONFLICT_TRANSFER_COUNT ||
        request[WAFER_DDR_TILE_REQ_CONFLICT_ORDER_COUNT] !=
            WAFER_DDR_TILE_CONFLICT_ORDER_COUNT ||
        request[WAFER_DDR_TILE_REQ_CONFLICT_SCHEDULE_COUNT] !=
            WAFER_DDR_TILE_CONFLICT_SCHEDULE_COUNT)
      return WAFER_DDR_TILE_STATUS_BAD_REQUEST;
  } else {
    return WAFER_DDR_TILE_STATUS_BAD_REQUEST;
  }
  return WAFER_DDR_TILE_STATUS_OK;
}

static void wafer_ddr_tile_write_header(volatile uint64_t *header,
                                        uint32_t rank, uint32_t sample,
                                        uint32_t status, uint32_t mode,
                                        const uint64_t *inputs,
                                        const uint64_t *outputs,
                                        uint64_t request_guard) {
  for (uint32_t word = 0; word < WAFER_DDR_TILE_HEADER_WORDS; ++word)
    header[word] = 0;
  header[WAFER_DDR_TILE_HDR_MAGIC] = WAFER_DDR_TILE_RECORD_MAGIC;
  header[WAFER_DDR_TILE_HDR_SCHEMA_AND_WORDS] =
      ((uint64_t)WAFER_DDR_TILE_SCHEMA << 32) | WAFER_DDR_TILE_HEADER_WORDS;
  header[WAFER_DDR_TILE_HDR_STATUS] = status;
  header[WAFER_DDR_TILE_HDR_RANK] = rank;
  header[WAFER_DDR_TILE_HDR_SAMPLE] = sample;
  header[WAFER_DDR_TILE_HDR_ROW_COUNT] =
      mode == WAFER_DDR_TILE_MODE_OFFSET ? WAFER_DDR_TILE_ROWS : 0U;
  header[WAFER_DDR_TILE_HDR_PAYLOAD_BYTES] = WAFER_DDR_TILE_PAYLOAD_BYTES;
  header[WAFER_DDR_TILE_HDR_GUARD_BYTES] = WAFER_DDR_TILE_GUARD_BYTES;
  header[WAFER_DDR_TILE_HDR_INPUT0_BASE] = inputs[0];
  header[WAFER_DDR_TILE_HDR_INPUT1_BASE] = inputs[1];
  header[WAFER_DDR_TILE_HDR_OUTPUT0_BASE] = outputs[0];
  header[WAFER_DDR_TILE_HDR_OUTPUT1_BASE] = outputs[1];
  header[WAFER_DDR_TILE_HDR_REQUEST_GUARD] = request_guard;
  header[WAFER_DDR_TILE_HDR_MODE] = mode;
  header[WAFER_DDR_TILE_HDR_CONFLICT_ROW_COUNT] =
      mode == WAFER_DDR_TILE_MODE_CONFLICT_EQUIVALENCE
          ? WAFER_DDR_TILE_CONFLICT_ROWS
          : 0U;
  header[WAFER_DDR_TILE_HDR_CONFLICT_SAMPLES] =
      mode == WAFER_DDR_TILE_MODE_CONFLICT_EQUIVALENCE
          ? WAFER_DDR_TILE_CONFLICT_SAMPLES
          : 0U;
  header[WAFER_DDR_TILE_HDR_CONFLICT_BASE_COUNT] =
      mode == WAFER_DDR_TILE_MODE_CONFLICT_EQUIVALENCE
          ? WAFER_DDR_TILE_CONFLICT_BASE_COUNT
          : 0U;
  header[WAFER_DDR_TILE_HDR_CONFLICT_OFFSET_COUNT] =
      mode == WAFER_DDR_TILE_MODE_CONFLICT_EQUIVALENCE
          ? WAFER_DDR_TILE_CONFLICT_OFFSET_COUNT
          : 0U;
  header[WAFER_DDR_TILE_HDR_CONFLICT_TRANSFER_COUNT] =
      mode == WAFER_DDR_TILE_MODE_CONFLICT_EQUIVALENCE
          ? WAFER_DDR_TILE_CONFLICT_TRANSFER_COUNT
          : 0U;
  header[WAFER_DDR_TILE_HDR_CONFLICT_RANK_ORDER] =
      mode == WAFER_DDR_TILE_MODE_CONFLICT_EQUIVALENCE
          ? ((sample & 1U) != 0U
                 ? WAFER_DDR_TILE_CONFLICT_RANK_ORDER_REVERSE
                 : WAFER_DDR_TILE_CONFLICT_RANK_ORDER_FORWARD)
          : 0U;
  header[WAFER_DDR_TILE_HDR_RECORD_GUARD] = WAFER_DDR_TILE_RECORD_GUARD;
}

__attribute__((visibility("hidden"))) void
wafer_tx81_ddr_tile_offset_probe(uint32_t rank, uint64_t input0,
                                 uint64_t input1, uint64_t output0,
                                 uint64_t output1, uint64_t status_ddr) {
  const uint64_t inputs[WAFER_DDR_TILE_ALLOCATIONS] = {input0, input1};
  const uint64_t outputs[WAFER_DDR_TILE_ALLOCATIONS] = {output0, output1};
  wafer_tx81_direct_dte_begin_after_prepare(status_ddr, WAFER_DDR_TILE_RANKS);
  wafer_ddr_tile_cache_range(input0, WAFER_DDR_TILE_REQUEST_WORDS * 8U, 1U);
  const volatile uint64_t *request =
      (const volatile uint64_t *)(uintptr_t)input0;
  uint32_t status = wafer_ddr_tile_request_status(rank, request);
  uint32_t sample = (uint32_t)request[WAFER_DDR_TILE_REQ_SAMPLE];
  uint32_t mode = (uint32_t)request[WAFER_DDR_TILE_REQ_MODE];
  volatile uint64_t *header = (volatile uint64_t *)(uintptr_t)output0;
  volatile uint64_t *rows = header + WAFER_DDR_TILE_HEADER_WORDS;
  volatile uint64_t *conflict_rows = header + WAFER_DDR_TILE_HEADER_WORDS;

  for (uint32_t rank_position = 0; rank_position < WAFER_DDR_TILE_RANKS;
       ++rank_position) {
    uint32_t active_rank =
        mode == WAFER_DDR_TILE_MODE_CONFLICT_EQUIVALENCE &&
                (sample & 1U) != 0U
            ? WAFER_DDR_TILE_RANKS - 1U - rank_position
            : rank_position;
    hrt_barrier();
    if (rank == active_rank && status == WAFER_DDR_TILE_STATUS_OK) {
      if (mode == WAFER_DDR_TILE_MODE_OFFSET) {
        for (uint32_t allocation = 0; allocation < WAFER_DDR_TILE_ALLOCATIONS;
             ++allocation) {
          for (uint32_t offset_index = 0;
               offset_index < WAFER_DDR_TILE_OFFSET_COUNT; ++offset_index) {
            for (uint32_t cell_sample = 0; cell_sample < WAFER_DDR_TILE_SAMPLES;
                 ++cell_sample) {
              uint32_t allocation_row_base =
                  allocation * WAFER_DDR_TILE_DIRECTIONS *
                  WAFER_DDR_TILE_OFFSET_COUNT * WAFER_DDR_TILE_SAMPLES;
              uint32_t rdma_row = allocation_row_base +
                                  offset_index * WAFER_DDR_TILE_SAMPLES +
                                  cell_sample;
              uint32_t wdma_row =
                  allocation_row_base +
                  WAFER_DDR_TILE_OFFSET_COUNT * WAFER_DDR_TILE_SAMPLES +
                  offset_index * WAFER_DDR_TILE_SAMPLES + cell_sample;
              wafer_ddr_tile_measure_rdma(
                  rank, allocation, offset_index, cell_sample,
                  inputs[allocation], outputs[allocation],
                  rows + rdma_row * WAFER_DDR_TILE_ROW_WORDS);
              wafer_ddr_tile_measure_wdma(
                  rank, allocation, offset_index, cell_sample,
                  inputs[allocation], outputs[allocation],
                  rows + wdma_row * WAFER_DDR_TILE_ROW_WORDS);
            }
          }
        }
      } else if (mode == WAFER_DDR_TILE_MODE_CONFLICT_EQUIVALENCE) {
        for (uint32_t allocation = 0; allocation < WAFER_DDR_TILE_ALLOCATIONS;
             ++allocation) {
          for (uint32_t base_index = 0;
               base_index < WAFER_DDR_TILE_CONFLICT_BASE_COUNT; ++base_index) {
            for (uint32_t offset_index = 0;
                 offset_index < WAFER_DDR_TILE_CONFLICT_OFFSET_COUNT;
                 ++offset_index) {
              for (uint32_t transfer_index = 0;
                   transfer_index < WAFER_DDR_TILE_CONFLICT_TRANSFER_COUNT;
                   ++transfer_index) {
                for (uint32_t issue_order_index = 0;
                     issue_order_index < WAFER_DDR_TILE_CONFLICT_ORDER_COUNT;
                     ++issue_order_index) {
                  for (uint32_t cell_sample = 0;
                       cell_sample < WAFER_DDR_TILE_CONFLICT_SAMPLES;
                       ++cell_sample) {
                    uint32_t reverse_schedule =
                        (cell_sample + sample) & 1U;
                    for (uint32_t schedule_position = 0;
                         schedule_position <
                         WAFER_DDR_TILE_CONFLICT_SCHEDULE_COUNT;
                         ++schedule_position) {
                      uint32_t schedule_index =
                          reverse_schedule != 0U
                              ? WAFER_DDR_TILE_CONFLICT_SCHEDULE_COUNT - 1U -
                                    schedule_position
                              : schedule_position;
                      uint32_t ordinal = wafer_ddr_tile_conflict_ordinal(
                          allocation, base_index, offset_index, transfer_index,
                          issue_order_index, schedule_index, cell_sample);
                      wafer_ddr_tile_measure_conflict_pair(
                          rank, allocation, base_index, offset_index,
                          transfer_index, issue_order_index, schedule_index,
                          cell_sample, schedule_position, inputs[allocation],
                          outputs[allocation],
                          conflict_rows +
                              ordinal * WAFER_DDR_TILE_CONFLICT_ROW_WORDS);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    hrt_barrier();
  }

  wafer_ddr_tile_write_header(header, rank, sample, status, mode, inputs,
                              outputs, request[WAFER_DDR_TILE_REQ_GUARD]);
  uint32_t record_bytes = mode == WAFER_DDR_TILE_MODE_CONFLICT_EQUIVALENCE
                              ? WAFER_DDR_TILE_CONFLICT_RECORD_BYTES
                              : WAFER_DDR_TILE_OFFSET_RECORD_BYTES;
  wafer_ddr_tile_cache_range(output0, record_bytes, 0U);
  wafer_tx81_direct_dte_finish();
}
