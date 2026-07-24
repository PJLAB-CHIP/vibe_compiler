#include "Wafer/ABI/Tx81DirectDTEStatusABI.h"
#include "instr_adapter.h"
#include "instr_adapter_plat.h"
#include "instr_def.h"
#include "wafer_ncc_hazard_relation.h"
#include "wafer_ncc_probe_plan.h"
#include "wafer_tx81_crt.h"

#include <stddef.h>
#include <stdint.h>

#define WAFER_NCC_PROBE_PMU_BASE UINT64_C(0x590000)
#define WAFER_NCC_PROBE_QUEUES 5U
#define WAFER_NCC_PROBE_PMU64_COUNTERS 8U
#define WAFER_NCC_PROBE_STABLE_RETRIES 8U
#define WAFER_NCC_PROBE_WORKER_PMU_STRIDE UINT32_C(0x30)
#define WAFER_NCC_PROBE_NE_LOGICAL_DIM 16U

extern int8_t *get_spm_memory_mapping(uint64_t offset);

typedef union WaferNccProbePacket {
  TsmArithInstr ct;
  TsmNeInstr ne;
  TsmRdmaInstr rdma;
  TsmWdmaInstr wdma;
  TsmDataMoveInstr tdma;
} WaferNccProbePacket;

typedef struct WaferNccProbeInstruction {
  uint32_t queue;
  WaferNccProbePacket packet;
  void *owner;
  uint32_t has_owner;
} WaferNccProbeInstruction;

static const uint32_t wafer_ncc_probe_pmu64_offsets
    [WAFER_NCC_PROBE_PMU64_COUNTERS] = {
        GR_PMU_STATISTICS_WINDOW, GR_PMU_FU_EXE_TIME,   GR_PMU_CT_EXE_TIME,
        GR_PMU_NE_EXE_TIME,      GR_PMU_RDMA_EXE_TIME, GR_PMU_WDMA_EXE_TIME,
        GR_PMU_TDMA_EXE_TIME,    GR_PMU_SCALAR_EXE_TIME,
};

static const uint32_t
    wafer_ncc_probe_instruction_offsets[WAFER_NCC_PROBE_QUEUES] = {
        GR_PMU_CT_INST_NUMS, GR_PMU_NE_INST_NUMS, GR_PMU_RDMA_INST_NUMS,
        GR_PMU_WDMA_INST_NUMS, GR_PMU_TDMA_INST_NUMS,
};

static const uint32_t
    wafer_ncc_probe_blocking_offsets[WAFER_NCC_PROBE_QUEUES] = {
        GR_PMU_CT_BLOCKING_TIME, GR_PMU_NE_BLOCKING_TIME,
        GR_PMU_RDMA_BLOCKING_TIME, GR_PMU_WDMA_BLOCKING_TIME,
        GR_PMU_TDMA_BLOCKING_TIME,
};

static uint32_t wafer_ncc_probe_read_pmu32(uint32_t offset) {
  return *(const volatile uint32_t *)(uintptr_t)(WAFER_NCC_PROBE_PMU_BASE +
                                                  offset);
}

static uint64_t wafer_ncc_probe_read_pmu64(uint32_t low_offset,
                                           uint64_t stability_bit,
                                           uint64_t *stable_mask) {
  uint32_t low = 0;
  uint32_t high_after = 0;
  for (uint32_t retry = 0; retry < WAFER_NCC_PROBE_STABLE_RETRIES; ++retry) {
    uint32_t high_before = wafer_ncc_probe_read_pmu32(low_offset + 4);
    low = wafer_ncc_probe_read_pmu32(low_offset);
    high_after = wafer_ncc_probe_read_pmu32(low_offset + 4);
    if (high_before == high_after) {
      *stable_mask |= stability_bit;
      break;
    }
  }
  return ((uint64_t)high_after << 32) | low;
}

static void wafer_ncc_probe_read_controls(volatile uint64_t *record,
                                           uint32_t base) {
  for (uint32_t worker = 0; worker < WAFER_NCC_PROTOCOL_WORKERS; ++worker)
    record[base + worker] =
        get_ncc_reg(worker, GR_CSR_CONTROL_ADDR);
}

static void wafer_ncc_probe_read_snapshot(volatile uint64_t *record,
                                           uint32_t pmu64_base,
                                           uint32_t instruction_base,
                                           uint32_t blocking_base,
                                           uint32_t stable_index) {
  uint64_t stable_mask = 0;
  for (uint32_t index = 0; index < WAFER_NCC_PROBE_PMU64_COUNTERS; ++index)
    record[pmu64_base + index] = wafer_ncc_probe_read_pmu64(
        wafer_ncc_probe_pmu64_offsets[index], UINT64_C(1) << index,
        &stable_mask);
  for (uint32_t worker = 0; worker < WAFER_NCC_PROTOCOL_WORKERS; ++worker) {
    for (uint32_t queue = 0; queue < WAFER_NCC_PROBE_QUEUES; ++queue) {
      uint32_t index = worker * WAFER_NCC_PROBE_QUEUES + queue;
      uint32_t worker_offset =
          worker * WAFER_NCC_PROBE_WORKER_PMU_STRIDE;
      record[instruction_base + index] = wafer_ncc_probe_read_pmu32(
          wafer_ncc_probe_instruction_offsets[queue] + worker_offset);
      record[blocking_base + index] = wafer_ncc_probe_read_pmu32(
          wafer_ncc_probe_blocking_offsets[queue] + worker_offset);
    }
  }
  record[stable_index] = stable_mask;
}

static void wafer_ncc_probe_publish(volatile uint64_t *record) {
  enum {
    WAFER_TX81_SUPERVISOR_MODE = 1,
    WAFER_TX81_MACHINE_MODE = 3,
  };
  uintptr_t mode;
  uintptr_t begin = (uintptr_t)record;
  __asm__ volatile("fence" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
  __asm__ volatile("csrr %0, mxstatus" : "=r"(mode));
  mode = (mode >> 30) & 3U;
  for (uintptr_t address = begin;
       address <
       begin + WAFER_NCC_PROTOCOL_RECORD_WORDS * sizeof(uint64_t);
       address += WAFER_TX81_DIRECT_DTE_STATUS_V2_CACHE_LINE_BYTES) {
    if (mode == WAFER_TX81_MACHINE_MODE)
      __asm__ volatile("dcache.cipa %0" : : "r"(address) : "memory");
    else if (mode == WAFER_TX81_SUPERVISOR_MODE)
      __asm__ volatile("dcache.civa %0" : : "r"(address) : "memory");
  }
  __asm__ volatile("sync.is" ::: "memory");
  __asm__ volatile("fence" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
}

static void wafer_ncc_probe_invalidate(uint64_t begin, uint32_t bytes) {
  enum {
    WAFER_TX81_SUPERVISOR_MODE = 1,
    WAFER_TX81_MACHINE_MODE = 3,
  };
  uintptr_t mode;
  __asm__ volatile("fence" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
  __asm__ volatile("csrr %0, mxstatus" : "=r"(mode));
  mode = (mode >> 30) & 3U;
  for (uintptr_t address = begin; address < begin + bytes;
       address += WAFER_TX81_DIRECT_DTE_STATUS_V2_CACHE_LINE_BYTES) {
    if (mode == WAFER_TX81_MACHINE_MODE)
      __asm__ volatile("dcache.ipa %0" : : "r"(address) : "memory");
    else if (mode == WAFER_TX81_SUPERVISOR_MODE)
      __asm__ volatile("dcache.iva %0" : : "r"(address) : "memory");
  }
  __asm__ volatile("sync.is" ::: "memory");
  __asm__ volatile("fence" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
}

static volatile uint8_t *wafer_ncc_probe_spm8(uint64_t offset) {
  return (volatile uint8_t *)(void *)get_spm_memory_mapping(offset);
}

static volatile uint16_t *wafer_ncc_probe_spm16(uint64_t offset) {
  return (volatile uint16_t *)(void *)get_spm_memory_mapping(offset);
}

static uint32_t wafer_ncc_probe_tdma_elements(uint32_t bytes,
                                               uint32_t format) {
  switch ((Data_Format)format) {
  case Fmt_INT8:
    return bytes;
  case Fmt_BOOL:
    return bytes * 8U;
  case Fmt_FP16:
  case Fmt_BF16:
    return bytes / 2U;
  default:
    return 0;
  }
}

static void wafer_ncc_probe_fill8(volatile uint8_t *destination, uint8_t value,
                                  uint32_t bytes) {
  for (uint32_t index = 0; index < bytes; ++index)
    destination[index] = value;
}

static void wafer_ncc_probe_fill16(volatile uint16_t *destination,
                                   uint16_t value, uint32_t elements) {
  for (uint32_t index = 0; index < elements; ++index)
    destination[index] = value;
}

static uint64_t wafer_ncc_probe_mismatch8(const volatile uint8_t *actual,
                                          uint8_t expected, uint32_t bytes) {
  uint64_t mismatches = 0;
  for (uint32_t index = 0; index < bytes; ++index)
    mismatches += actual[index] != expected;
  return mismatches;
}

static void wafer_ncc_probe_set_worker(uint32_t *inter_type, uint32_t worker) {
  *inter_type = (*inter_type & ~UINT32_C(0x300)) | ((worker % 3U) << 8);
}

static int wafer_ncc_probe_prepare_ct(WaferNccProbeInstruction *instruction,
                                      uint32_t worker, uint64_t lhs,
                                      uint64_t rhs, uint64_t output,
                                      uint32_t elements) {
  instruction->queue = WAFER_NCC_ENGINE_CT;
  instruction->owner = TsmNewArith();
  instruction->has_owner = instruction->owner != NULL;
  if (!instruction->has_owner)
    return 0;
  TsmArith *arith = (TsmArith *)instruction->owner;
  arith->AddVV(&instruction->packet.ct, lhs, rhs, output, elements,
               RND_NEAREST_EVEN, Fmt_FP16);
  wafer_ncc_probe_set_worker(&instruction->packet.ct.inter_type, worker);
  return 1;
}

static int wafer_ncc_probe_prepare_rdma(WaferNccProbeInstruction *instruction,
                                        uint32_t worker, uint64_t source,
                                        uint64_t destination,
                                        uint32_t bytes) {
  instruction->queue = WAFER_NCC_ENGINE_RDMA;
  instruction->owner = TsmNewRdma();
  instruction->has_owner = instruction->owner != NULL;
  if (!instruction->has_owner)
    return 0;
  TsmRdma *rdma = (TsmRdma *)instruction->owner;
  rdma->AddSrcDst(&instruction->packet.rdma, source, destination, Fmt_UINT8);
  rdma->ConfigStrideIteration(&instruction->packet.rdma, bytes, 0, 1, 0, 1, 0,
                              1);
  wafer_ncc_probe_set_worker(&instruction->packet.rdma.inter_type, worker);
  return 1;
}

static int wafer_ncc_probe_prepare_wdma(WaferNccProbeInstruction *instruction,
                                        uint32_t worker, uint64_t source,
                                        uint64_t destination,
                                        uint32_t bytes) {
  instruction->queue = WAFER_NCC_ENGINE_WDMA;
  instruction->owner = TsmNewWdma();
  instruction->has_owner = instruction->owner != NULL;
  if (!instruction->has_owner)
    return 0;
  TsmWdma *wdma = (TsmWdma *)instruction->owner;
  wdma->AddSrcDst(&instruction->packet.wdma, source, destination, Fmt_UINT8);
  wdma->ConfigStrideIteration(&instruction->packet.wdma, bytes, 0, 1, 0, 1, 0,
                              1);
  wafer_ncc_probe_set_worker(&instruction->packet.wdma.inter_type, worker);
  return 1;
}

static void wafer_ncc_probe_release(WaferNccProbeInstruction *instruction) {
  if (!instruction->has_owner)
    return;
  switch (instruction->queue) {
  case WAFER_NCC_ENGINE_CT:
    TsmDeleteArith((TsmArith *)instruction->owner);
    break;
  case WAFER_NCC_ENGINE_NE:
    TsmDeleteGemm((TsmGemm *)instruction->owner);
    break;
  case WAFER_NCC_ENGINE_RDMA:
    TsmDeleteRdma((TsmRdma *)instruction->owner);
    break;
  case WAFER_NCC_ENGINE_WDMA:
    TsmDeleteWdma((TsmWdma *)instruction->owner);
    break;
  case WAFER_NCC_ENGINE_TDMA:
    TsmDeletePeripheral((TsmPeripheral *)instruction->owner);
    break;
  default:
    break;
  }
  instruction->has_owner = 0;
  instruction->owner = NULL;
}

static void *wafer_ncc_probe_packet(WaferNccProbeInstruction *instruction) {
  switch (instruction->queue) {
  case WAFER_NCC_ENGINE_CT:
    return &instruction->packet.ct;
  case WAFER_NCC_ENGINE_NE:
    return &instruction->packet.ne;
  case WAFER_NCC_ENGINE_RDMA:
    return &instruction->packet.rdma;
  case WAFER_NCC_ENGINE_WDMA:
    return &instruction->packet.wdma;
  case WAFER_NCC_ENGINE_TDMA:
    return &instruction->packet.tdma;
  default:
    return NULL;
  }
}

static uint64_t
wafer_ncc_probe_issue_raw(WaferNccProbeInstruction *instruction) {
  /*
   * Every schema-v2 issue owns a unique packet slot and is dispatched once.
   * Execute that packet directly so NE's execute-time end materialization is
   * not lost in a temporary copy.  Other engines clear their packet after
   * dispatch, so their post-dispatch ranges are read from NCC registers.
   */
  return TsmExecute(wafer_ncc_probe_packet(instruction));
}

static void wafer_ncc_probe_wait_worker(uint32_t worker) {
  (void)TsmWaitfinish_bywork(worker % WAFER_NCC_PROTOCOL_WORKERS);
}

static uint32_t
wafer_ncc_probe_packet_inter_type(const WaferNccProbeInstruction *instruction) {
  switch (instruction->queue) {
  case WAFER_NCC_ENGINE_CT:
    return instruction->packet.ct.inter_type;
  case WAFER_NCC_ENGINE_NE:
    return instruction->packet.ne.inter_type;
  case WAFER_NCC_ENGINE_RDMA:
    return instruction->packet.rdma.inter_type;
  case WAFER_NCC_ENGINE_WDMA:
    return instruction->packet.wdma.inter_type;
  case WAFER_NCC_ENGINE_TDMA:
    return instruction->packet.tdma.inter_type;
  default:
    return 0;
  }
}

#define WAFER_NCC_V2_MAX_TRANSFER_BYTES UINT32_C(65536)
#define WAFER_NCC_V2_SPM_SLOT_BASE UINT64_C(0x10000)
#define WAFER_NCC_V2_SPM_SLOT_STRIDE UINT64_C(0x20000)
#define WAFER_NCC_V2_SPM_READ0_OFFSET UINT64_C(0x100)
#define WAFER_NCC_V2_SPM_READ1_OFFSET UINT64_C(0x5100)
#define WAFER_NCC_V2_SPM_WRITE_OFFSET UINT64_C(0xa100)
#define WAFER_NCC_V2_GUARD_BYTES UINT32_C(256)
#define WAFER_NCC_V2_REPEATED_SLOT_BYTES UINT64_C(16384)
#define WAFER_NCC_V2_DDR_SLOT_STRIDE                                  \
  (WAFER_NCC_V2_REPEATED_SLOT_BYTES + 2U * WAFER_NCC_V2_GUARD_BYTES)
#define WAFER_NCC_V2_OUTPUT_SLOT_BASE UINT64_C(4096)
#define WAFER_NCC_V2_RESOURCE_BYTES                                      \
  (WAFER_NCC_V2_OUTPUT_SLOT_BASE +                                      \
   WAFER_NCC_PROTOCOL_MAX_ISSUES * WAFER_NCC_V2_DDR_SLOT_STRIDE)
#define WAFER_NCC_V2_RECORD_GUARD UINT64_C(0xd87c2a916be4035f)
#define WAFER_NCC_V2_NE_PHYSICAL_BYTES UINT32_C(256)
#define WAFER_NCC_V2_NE_RHS_BYTES UINT32_C(512)
#define WAFER_NCC_V2_NE_RESULT_BYTES UINT32_C(32)
#define WAFER_NCC_V2_NE_LARGE_M UINT32_C(64)
#define WAFER_NCC_V2_NE_LARGE_K UINT32_C(128)
#define WAFER_NCC_V2_NE_LARGE_N UINT32_C(128)
#define WAFER_NCC_V2_NE_LARGE_LHS_BYTES UINT32_C(16384)
#define WAFER_NCC_V2_NE_LARGE_RHS_BYTES UINT32_C(32768)
#define WAFER_NCC_V2_NE_LARGE_RESULT_BYTES UINT32_C(16384)
#define WAFER_NCC_V2_NE_LARGE_READ1_OFFSET UINT64_C(0x4300)
#define WAFER_NCC_V2_NE_LARGE_WRITE_OFFSET UINT64_C(0xc500)
#define WAFER_NCC_V2_HAZARD_SELECTED_OFFSET UINT64_C(0x4000)
#define WAFER_NCC_V2_HAZARD_SECOND_BASELINE_OFFSET UINT64_C(0x6000)
#define WAFER_NCC_V2_HAZARD_SLOT_BEGIN_OFFSET UINT64_C(0x3000)
#define WAFER_NCC_V2_HAZARD_SLOT_END_OFFSET UINT64_C(0x8000)
#define WAFER_NCC_V2_STRIDED_INITIAL_SOURCE_SLOT \
  WAFER_NCC_PROTOCOL_MAX_ISSUES

typedef struct WaferNccV2Context {
  uint64_t payload_ddr;
  uint64_t output_ddr;
  uint64_t completion_marker_address;
  uint32_t completion_marker_expected;
  uint64_t operands[WAFER_NCC_PROTOCOL_MAX_ISSUES][3];
  WaferNccHazardComposition
      hazards[WAFER_NCC_PROTOCOL_MAX_ROUNDS];
  uint32_t hazard_seeded_mask;
  uint32_t configured;
  uint32_t has_hazard;
  uint32_t ordered_producer_consumer;
  uint32_t double_slot_observation;
  uint32_t constructor_captured;
  uint64_t constructor_address;
  uint64_t issued_inter_types[WAFER_NCC_PROTOCOL_MAX_ISSUES];
  WaferNccProbeInstruction
      instructions[WAFER_NCC_PROTOCOL_MAX_ISSUES];
} WaferNccV2Context;

static uint64_t wafer_ncc_v2_spm_slot(uint32_t slot) {
  return WAFER_NCC_V2_SPM_SLOT_BASE +
         (uint64_t)slot * WAFER_NCC_V2_SPM_SLOT_STRIDE;
}

static uint64_t wafer_ncc_v2_read0(uint32_t slot) {
  return wafer_ncc_v2_spm_slot(slot) + WAFER_NCC_V2_SPM_READ0_OFFSET;
}

static uint64_t wafer_ncc_v2_read1(uint32_t slot) {
  return wafer_ncc_v2_spm_slot(slot) + WAFER_NCC_V2_SPM_READ1_OFFSET;
}

static uint64_t wafer_ncc_v2_write(uint32_t slot) {
  return wafer_ncc_v2_spm_slot(slot) + WAFER_NCC_V2_SPM_WRITE_OFFSET;
}

static uint64_t wafer_ncc_v2_payload_address(const WaferNccV2Context *context,
                                             uint32_t slot) {
  return context->payload_ddr +
         (uint64_t)slot * WAFER_NCC_V2_DDR_SLOT_STRIDE +
         WAFER_NCC_V2_GUARD_BYTES;
}

static uint64_t wafer_ncc_v2_output_address(const WaferNccV2Context *context,
                                            uint32_t slot) {
  return context->output_ddr + WAFER_NCC_V2_OUTPUT_SLOT_BASE +
         (uint64_t)slot * WAFER_NCC_V2_DDR_SLOT_STRIDE +
         WAFER_NCC_V2_GUARD_BYTES;
}

static int
wafer_ncc_v2_is_large_ne_lane(const WaferNccProbeLane *lane) {
  return lane->engine == WAFER_NCC_ENGINE_NE &&
         lane->transfer_bytes == WAFER_NCC_V2_NE_LARGE_RESULT_BYTES;
}

static uint32_t
wafer_ncc_v2_ne_lhs_bytes(const WaferNccProbeLane *lane) {
  return wafer_ncc_v2_is_large_ne_lane(lane)
             ? WAFER_NCC_V2_NE_LARGE_LHS_BYTES
             : WAFER_NCC_V2_NE_PHYSICAL_BYTES;
}

static uint32_t
wafer_ncc_v2_ne_rhs_bytes(const WaferNccProbeLane *lane) {
  return wafer_ncc_v2_is_large_ne_lane(lane)
             ? WAFER_NCC_V2_NE_LARGE_RHS_BYTES
             : WAFER_NCC_V2_NE_RHS_BYTES;
}

static uint32_t
wafer_ncc_v2_ne_result_bytes(const WaferNccProbeLane *lane) {
  return wafer_ncc_v2_is_large_ne_lane(lane)
             ? WAFER_NCC_V2_NE_LARGE_RESULT_BYTES
             : WAFER_NCC_V2_NE_RESULT_BYTES;
}

static uint32_t
wafer_ncc_v2_ne_output_span(const WaferNccProbeLane *lane) {
  return wafer_ncc_v2_is_large_ne_lane(lane)
             ? WAFER_NCC_V2_NE_LARGE_RESULT_BYTES
             : WAFER_NCC_V2_NE_PHYSICAL_BYTES;
}

static int wafer_ncc_v2_dma_layout_equal(const WaferNccProbeLane *lhs,
                                         const WaferNccProbeLane *rhs) {
  return lhs->layout_kind == WAFER_NCC_LAYOUT_DMA_STRIDED &&
         rhs->layout_kind == WAFER_NCC_LAYOUT_DMA_STRIDED &&
         lhs->layout_inner_bytes == rhs->layout_inner_bytes &&
         lhs->layout_stride0_bytes == rhs->layout_stride0_bytes &&
         lhs->layout_stride1_bytes == rhs->layout_stride1_bytes &&
         lhs->layout_stride2_bytes == rhs->layout_stride2_bytes &&
         lhs->layout_iteration0 == rhs->layout_iteration0 &&
         lhs->layout_iteration1 == rhs->layout_iteration1 &&
         lhs->layout_iteration2 == rhs->layout_iteration2;
}

static uint32_t
wafer_ncc_v2_dma_envelope_bytes(const WaferNccProbeLane *lane) {
  if (lane->layout_kind != WAFER_NCC_LAYOUT_DMA_STRIDED)
    return lane->transfer_bytes;
  uint64_t last_offset =
      (uint64_t)(lane->layout_iteration0 - 1U) *
          lane->layout_stride0_bytes +
      (uint64_t)(lane->layout_iteration1 - 1U) *
          lane->layout_stride1_bytes +
      (uint64_t)(lane->layout_iteration2 - 1U) *
          lane->layout_stride2_bytes;
  return (uint32_t)(last_offset + lane->layout_inner_bytes);
}

static uint32_t wafer_ncc_v2_dma_chunk_offset(
    const WaferNccProbeLane *lane, uint32_t wanted_chunk) {
  uint32_t chunk = 0;
  for (uint32_t outer = 0; outer < lane->layout_iteration2; ++outer)
    for (uint32_t middle = 0; middle < lane->layout_iteration1; ++middle)
      for (uint32_t inner = 0; inner < lane->layout_iteration0; ++inner) {
        uint32_t offset =
            outer * lane->layout_stride2_bytes +
            middle * lane->layout_stride1_bytes +
            inner * lane->layout_stride0_bytes;
        if (chunk++ == wanted_chunk)
          return offset;
      }
  return UINT32_MAX;
}

static uint32_t wafer_ncc_v2_strided_second_shift(
    const WaferNccProbeRequest *request) {
  const WaferNccProbeLane *lane = &request->lanes[0];
  if (request->range_relation == WAFER_NCC_RANGE_PARTIAL) {
    uint32_t chunks = lane->layout_iteration0 * lane->layout_iteration1 *
                      lane->layout_iteration2;
    return wafer_ncc_v2_dma_chunk_offset(lane, chunks / 2U);
  }
  if (request->range_relation == WAFER_NCC_RANGE_ADJACENT)
    return wafer_ncc_v2_dma_envelope_bytes(lane);
  return 0;
}

static int
wafer_ncc_v2_is_dma_roundtrip(const WaferNccProbeRequest *request) {
  return request->lane_count == 2 && request->rounds == 1 &&
         request->effect_relation == WAFER_NCC_EFFECT_RAW &&
         request->range_relation == WAFER_NCC_RANGE_EXACT &&
         request->schedule == WAFER_NCC_SCHEDULE_SERIAL &&
         request->wait_kind == WAFER_NCC_WAIT_BY_WORKER &&
         request->wait_worker_mask == 1 && request->flags == 0 &&
         request->issue_limit == 0 &&
         request->lanes[0].engine == WAFER_NCC_ENGINE_RDMA &&
         request->lanes[1].engine == WAFER_NCC_ENGINE_WDMA &&
         request->lanes[0].worker == 0 &&
         request->lanes[1].worker == 0 &&
         request->lanes[0].issue_mode == WAFER_NCC_ISSUE_WRAPPER &&
         request->lanes[1].issue_mode == WAFER_NCC_ISSUE_WRAPPER &&
         request->lanes[0].element_format ==
             WAFER_NCC_PROTOCOL_DMA_FORMAT_FP16 &&
         request->lanes[1].element_format ==
             WAFER_NCC_PROTOCOL_DMA_FORMAT_FP16 &&
         ((request->lanes[0].layout_inner_bytes |
           request->lanes[0].layout_stride0_bytes |
           request->lanes[0].layout_stride1_bytes |
           request->lanes[0].layout_stride2_bytes) &
          UINT32_C(1)) == 0 &&
         request->first_operand == WAFER_NCC_OPERAND_WRITE &&
         request->second_operand == WAFER_NCC_OPERAND_READ0 &&
         wafer_ncc_v2_dma_layout_equal(&request->lanes[0],
                                       &request->lanes[1]);
}

static int wafer_ncc_v2_ordered_pair_supported(
    const WaferNccProbeRequest *request) {
  if (request->flags != WAFER_NCC_REQUEST_ORDERED_PRODUCER_CONSUMER ||
      request->lane_count != 2 || request->rounds != 1 ||
      request->effect_relation != WAFER_NCC_EFFECT_RAW ||
      request->range_relation != WAFER_NCC_RANGE_EXACT ||
      request->schedule != WAFER_NCC_SCHEDULE_SERIAL ||
      request->first_operand != WAFER_NCC_OPERAND_WRITE ||
      request->second_operand != WAFER_NCC_OPERAND_READ0)
    return 0;
  const WaferNccProbeLane *first = &request->lanes[0];
  const WaferNccProbeLane *second = &request->lanes[1];
  if (first->element_format != Fmt_FP16 ||
      second->element_format != Fmt_FP16)
    return 0;
  if (first->engine == WAFER_NCC_ENGINE_NE &&
      second->engine == WAFER_NCC_ENGINE_WDMA)
    return first->transfer_bytes == WAFER_NCC_V2_NE_PHYSICAL_BYTES &&
           second->transfer_bytes == WAFER_NCC_V2_NE_RESULT_BYTES;
  if (second->engine == WAFER_NCC_ENGINE_NE)
    return first->engine == WAFER_NCC_ENGINE_TDMA &&
           first->transfer_bytes == WAFER_NCC_V2_NE_PHYSICAL_BYTES &&
           second->transfer_bytes == WAFER_NCC_V2_NE_PHYSICAL_BYTES;
  if (first->transfer_bytes != second->transfer_bytes)
    return 0;
  return (first->engine == WAFER_NCC_ENGINE_RDMA &&
          second->engine == WAFER_NCC_ENGINE_CT) ||
         (first->engine == WAFER_NCC_ENGINE_CT &&
          second->engine == WAFER_NCC_ENGINE_WDMA) ||
         (first->engine == WAFER_NCC_ENGINE_TDMA &&
          second->engine == WAFER_NCC_ENGINE_CT);
}

static uint32_t wafer_ncc_v2_operand_effect(uint32_t operand) {
  switch (operand) {
  case WAFER_NCC_OPERAND_READ0:
    return WAFER_NCC_MEMORY_READ0;
  case WAFER_NCC_OPERAND_READ1:
    return WAFER_NCC_MEMORY_READ1;
  case WAFER_NCC_OPERAND_WRITE:
    return WAFER_NCC_MEMORY_WRITE;
  default:
    return 0;
  }
}

static int wafer_ncc_v2_operand_uses_spm(uint32_t engine,
                                         uint32_t operand) {
  if (operand == WAFER_NCC_OPERAND_READ1)
    return engine == WAFER_NCC_ENGINE_CT || engine == WAFER_NCC_ENGINE_NE;
  if (operand == WAFER_NCC_OPERAND_READ0)
    return engine != WAFER_NCC_ENGINE_RDMA &&
           engine != WAFER_NCC_ENGINE_TDMA;
  if (operand == WAFER_NCC_OPERAND_WRITE)
    return engine != WAFER_NCC_ENGINE_WDMA;
  return 0;
}

static uint64_t wafer_ncc_v2_default_operand(
    const WaferNccV2Context *context, const WaferNccProbeLane *lane,
    uint32_t slot,
    uint32_t operand) {
  uint32_t engine = lane->engine;
  if (engine == WAFER_NCC_ENGINE_RDMA &&
      operand == WAFER_NCC_OPERAND_READ0)
    return wafer_ncc_v2_payload_address(context, slot);
  if (engine == WAFER_NCC_ENGINE_WDMA &&
      operand == WAFER_NCC_OPERAND_WRITE)
    return wafer_ncc_v2_output_address(context, slot);
  if (operand == WAFER_NCC_OPERAND_READ0)
    return wafer_ncc_v2_read0(slot);
  if (operand == WAFER_NCC_OPERAND_READ1)
    if (wafer_ncc_v2_is_large_ne_lane(lane))
      return wafer_ncc_v2_spm_slot(slot) +
             WAFER_NCC_V2_NE_LARGE_READ1_OFFSET;
  if (operand == WAFER_NCC_OPERAND_READ1)
    return wafer_ncc_v2_read1(slot);
  if (wafer_ncc_v2_is_large_ne_lane(lane))
    return wafer_ncc_v2_spm_slot(slot) +
           WAFER_NCC_V2_NE_LARGE_WRITE_OFFSET;
  return wafer_ncc_v2_write(slot);
}

static int
wafer_ncc_v2_hazard_supported(const WaferNccProbeRequest *request) {
  if (request->lane_count != 2 ||
      request->range_relation < WAFER_NCC_RANGE_EXACT ||
      request->range_relation > WAFER_NCC_RANGE_ADJACENT ||
      request->lanes[0].transfer_bytes !=
          request->lanes[1].transfer_bytes ||
      request->lanes[0].transfer_bytes > WAFER_NCC_V2_MAX_TRANSFER_BYTES ||
      request->lanes[0].transfer_bytes % sizeof(uint16_t) != 0 ||
      request->lanes[0].element_format != Fmt_FP16 ||
      request->lanes[1].element_format != Fmt_FP16)
    return 0;

  uint32_t first = request->lanes[0].engine;
  uint32_t second = request->lanes[1].engine;
  switch (request->effect_relation) {
  case WAFER_NCC_EFFECT_RAW:
    return (first == WAFER_NCC_ENGINE_RDMA &&
            second == WAFER_NCC_ENGINE_CT &&
            request->first_operand == WAFER_NCC_OPERAND_WRITE &&
            (request->second_operand == WAFER_NCC_OPERAND_READ0 ||
             request->second_operand == WAFER_NCC_OPERAND_READ1)) ||
           wafer_ncc_v2_is_dma_roundtrip(request);
  case WAFER_NCC_EFFECT_WAR:
    return first == WAFER_NCC_ENGINE_CT &&
           second == WAFER_NCC_ENGINE_TDMA &&
           (request->first_operand == WAFER_NCC_OPERAND_READ0 ||
            request->first_operand == WAFER_NCC_OPERAND_READ1) &&
           request->second_operand == WAFER_NCC_OPERAND_WRITE;
  case WAFER_NCC_EFFECT_WAW:
    return first == WAFER_NCC_ENGINE_RDMA &&
           second == WAFER_NCC_ENGINE_TDMA &&
           request->first_operand == WAFER_NCC_OPERAND_WRITE &&
           request->second_operand == WAFER_NCC_OPERAND_WRITE;
  case WAFER_NCC_EFFECT_RAR:
    return first == WAFER_NCC_ENGINE_CT &&
           second == WAFER_NCC_ENGINE_WDMA &&
           (request->first_operand == WAFER_NCC_OPERAND_READ0 ||
            request->first_operand == WAFER_NCC_OPERAND_READ1) &&
           request->second_operand == WAFER_NCC_OPERAND_READ0;
  default:
    return 0;
  }
}

static uint32_t wafer_ncc_v2_configure_context(
    WaferNccV2Context *context, const WaferNccProbeRequest *request) {
  if (context->configured)
    return 0;
  context->ordered_producer_consumer =
      request->flags == WAFER_NCC_REQUEST_ORDERED_PRODUCER_CONSUMER;
  context->double_slot_observation =
      request->flags == WAFER_NCC_REQUEST_DOUBLE_SLOT_OBSERVATION;
  for (uint32_t lane = 0; lane < request->lane_count; ++lane) {
    for (uint32_t round = 0; round < request->rounds; ++round) {
      uint32_t slot = lane * WAFER_NCC_PROTOCOL_MAX_ROUNDS + round;
      for (uint32_t operand = 0; operand < 3; ++operand)
        context->operands[slot][operand] = wafer_ncc_v2_default_operand(
            context, &request->lanes[lane], slot, operand);
    }
  }

  context->has_hazard =
      request->effect_relation != WAFER_NCC_EFFECT_NONE;
  if (context->double_slot_observation) {
    for (uint32_t round = 0; round < request->rounds; ++round) {
      uint32_t physical_slot = round & 1U;
      uint32_t rdma_slot = round;
      uint32_t ct_slot = WAFER_NCC_PROTOCOL_MAX_ROUNDS + round;
      uint32_t wdma_slot =
          2U * WAFER_NCC_PROTOCOL_MAX_ROUNDS + round;
      uint64_t input = wafer_ncc_v2_read0(physical_slot);
      uint64_t rhs = wafer_ncc_v2_read1(physical_slot);
      uint64_t output = wafer_ncc_v2_write(physical_slot);
      context->operands[rdma_slot][WAFER_NCC_OPERAND_WRITE] = input;
      context->operands[ct_slot][WAFER_NCC_OPERAND_READ0] = input;
      context->operands[ct_slot][WAFER_NCC_OPERAND_READ1] = rhs;
      context->operands[ct_slot][WAFER_NCC_OPERAND_WRITE] = output;
      context->operands[wdma_slot][WAFER_NCC_OPERAND_READ0] = output;
    }
    context->configured = 1;
    return 0;
  }
  if (context->ordered_producer_consumer) {
    if (!wafer_ncc_v2_ordered_pair_supported(request))
      return 1;
    uint32_t first_slot = 0;
    uint32_t second_slot = WAFER_NCC_PROTOCOL_MAX_ROUNDS;
    context->operands[second_slot][WAFER_NCC_OPERAND_READ0] =
        context->operands[first_slot][WAFER_NCC_OPERAND_WRITE];
    context->configured = 1;
    return 0;
  }
  if (wafer_ncc_v2_is_dma_roundtrip(request)) {
    uint64_t shared =
        context->operands[0][WAFER_NCC_OPERAND_WRITE];
    uint32_t envelope =
        wafer_ncc_v2_dma_envelope_bytes(&request->lanes[0]);
    context->operands[WAFER_NCC_PROTOCOL_MAX_ROUNDS]
                     [WAFER_NCC_OPERAND_READ0] = shared;
    WaferNccHazardComposition *composition = &context->hazards[0];
    composition->first_operand = WAFER_NCC_OPERAND_WRITE;
    composition->second_operand = WAFER_NCC_OPERAND_READ0;
    composition->flags = WAFER_NCC_HAZARD_COMPOSITION_ENVELOPE_ONLY;
    composition->first_range =
        (WaferNccHazardRange){shared, shared + envelope};
    composition->second_range = composition->first_range;
    composition->overlap = composition->first_range;
    composition->footprint = composition->first_range;
    composition->canary_before =
        (WaferNccHazardRange){shared - WAFER_NCC_V2_GUARD_BYTES, shared};
    composition->canary_after =
        (WaferNccHazardRange){
            shared + envelope,
            shared + envelope + WAFER_NCC_V2_GUARD_BYTES};
    context->configured = 1;
    return 0;
  }
  if (wafer_ncc_probe_is_strided_dependency(request)) {
    const WaferNccProbeLane *lane = &request->lanes[0];
    uint32_t envelope = wafer_ncc_v2_dma_envelope_bytes(lane);
    uint32_t second_shift = wafer_ncc_v2_strided_second_shift(request);
    if (second_shift == UINT32_MAX)
      return 1;
    uint64_t first_base = wafer_ncc_v2_write(0);
    uint64_t second_base = first_base + second_shift;
    uint32_t second_slot = WAFER_NCC_PROTOCOL_MAX_ROUNDS;
    context->operands[0][request->first_operand] = first_base;
    context->operands[second_slot][request->second_operand] = second_base;

    WaferNccHazardComposition *composition = &context->hazards[0];
    composition->first_operand = request->first_operand;
    composition->second_operand = request->second_operand;
    composition->flags = WAFER_NCC_HAZARD_COMPOSITION_ENVELOPE_ONLY;
    composition->first_range =
        (WaferNccHazardRange){first_base, first_base + envelope};
    composition->second_range =
        (WaferNccHazardRange){second_base, second_base + envelope};
    composition->overlap =
        (WaferNccHazardRange){
            second_base,
            second_base < first_base + envelope
                ? first_base + envelope
                : second_base,
        };
    composition->footprint =
        (WaferNccHazardRange){first_base, second_base + envelope};
    composition->canary_before =
        (WaferNccHazardRange){
            first_base - WAFER_NCC_V2_GUARD_BYTES, first_base};
    composition->canary_after =
        (WaferNccHazardRange){
            second_base + envelope,
            second_base + envelope + WAFER_NCC_V2_GUARD_BYTES};
    context->configured = 1;
    return 0;
  }
  if (!context->has_hazard) {
    context->configured = 1;
    return 0;
  }
  if (!wafer_ncc_v2_hazard_supported(request))
    return 1;

  static const uint64_t unselected_offsets[2][3] = {
      {UINT64_C(0x8000), UINT64_C(0xa000), UINT64_C(0xc000)},
      {UINT64_C(0x10000), UINT64_C(0x12000), UINT64_C(0x14000)},
  };
  uint32_t bytes = request->lanes[0].transfer_bytes;
  for (uint32_t round = 0; round < request->rounds; ++round) {
    uint64_t pair_base = wafer_ncc_v2_spm_slot(round);
    uint32_t slots[2] = {round, WAFER_NCC_PROTOCOL_MAX_ROUNDS + round};
    for (uint32_t lane = 0; lane < 2; ++lane)
      for (uint32_t operand = 0; operand < 3; ++operand)
        if (wafer_ncc_v2_operand_uses_spm(request->lanes[lane].engine,
                                         operand))
          context->operands[slots[lane]][operand] =
              pair_base + unselected_offsets[lane][operand];

    WaferNccHazardLaneMemory lanes[2] = {0};
    uint32_t operands[2] = {request->first_operand,
                            request->second_operand};
    uint64_t baselines[2] = {
        pair_base + WAFER_NCC_V2_HAZARD_SELECTED_OFFSET,
        pair_base + WAFER_NCC_V2_HAZARD_SECOND_BASELINE_OFFSET,
    };
    for (uint32_t lane = 0; lane < 2; ++lane) {
      lanes[lane].effects = wafer_ncc_v2_operand_effect(operands[lane]);
      lanes[lane].operands[operands[lane]].baseline =
          (WaferNccHazardRange){baselines[lane], baselines[lane] + bytes};
    }
    WaferNccHazardSelection selection = {request->first_operand,
                                         request->second_operand};
    uint32_t status = wafer_ncc_hazard_build_composition(
        &lanes[0], &lanes[1], request->effect_relation,
        request->range_relation, selection,
        (WaferNccHazardRange){
            pair_base + WAFER_NCC_V2_HAZARD_SLOT_BEGIN_OFFSET,
            pair_base + WAFER_NCC_V2_HAZARD_SLOT_END_OFFSET,
        },
        WAFER_NCC_V2_GUARD_BYTES, &context->hazards[round]);
    if (status != WAFER_NCC_HAZARD_OK)
      return 1;
    context->operands[slots[0]][request->first_operand] =
        context->hazards[round].first_range.begin;
    context->operands[slots[1]][request->second_operand] =
        context->hazards[round].second_range.begin;
  }
  context->configured = 1;
  return 0;
}

static uint64_t wafer_ncc_v2_operand_address(
    const WaferNccV2Context *context, const WaferNccProbeIssue *issue,
    uint32_t operand) {
  return context->operands[issue->slot][operand];
}

static uint16_t wafer_ncc_v2_positive_integer_f16(uint32_t value) {
  static const uint16_t values[] = {
      UINT16_C(0x0000), UINT16_C(0x3c00), UINT16_C(0x4000),
      UINT16_C(0x4200), UINT16_C(0x4400), UINT16_C(0x4500),
      UINT16_C(0x4600), UINT16_C(0x4700), UINT16_C(0x4800),
      UINT16_C(0x4880), UINT16_C(0x4900), UINT16_C(0x4980),
      UINT16_C(0x4a00), UINT16_C(0x4a80),
  };
  return value < sizeof(values) / sizeof(values[0]) ? values[value] : 0;
}

static uint16_t wafer_ncc_v2_hazard_initial_f16(uint32_t round) {
  return wafer_ncc_v2_positive_integer_f16(2U + round);
}

static uint16_t wafer_ncc_v2_hazard_second_write_f16(uint32_t round) {
  return wafer_ncc_v2_positive_integer_f16(8U + round);
}

static uint8_t wafer_ncc_v2_pattern_byte(uint32_t slot, uint32_t index) {
  return (uint8_t)(((slot + 1U) * 29U + index * 17U) & UINT8_MAX);
}

static int wafer_ncc_v2_dma_compact_index(
    const WaferNccProbeLane *lane, uint64_t base, uint64_t address,
    uint32_t *compact_index) {
  if (address < base)
    return 0;
  uint64_t relative = address - base;
  uint32_t chunk = 0;
  for (uint32_t outer = 0; outer < lane->layout_iteration2; ++outer)
    for (uint32_t middle = 0; middle < lane->layout_iteration1; ++middle)
      for (uint32_t inner = 0; inner < lane->layout_iteration0; ++inner) {
        uint32_t offset =
            outer * lane->layout_stride2_bytes +
            middle * lane->layout_stride1_bytes +
            inner * lane->layout_stride0_bytes;
        if (relative >= offset &&
            relative < (uint64_t)offset + lane->layout_inner_bytes) {
          *compact_index =
              chunk * lane->layout_inner_bytes +
              (uint32_t)(relative - offset);
          return 1;
        }
        ++chunk;
      }
  return 0;
}

static void wafer_ncc_v2_seed_strided_pattern(
    uint64_t base, const WaferNccProbeLane *lane, uint32_t source_slot) {
  volatile uint8_t *destination = wafer_ncc_probe_spm8(base);
  uint32_t compact_cursor = 0;
  for (uint32_t outer = 0; outer < lane->layout_iteration2; ++outer)
    for (uint32_t middle = 0; middle < lane->layout_iteration1; ++middle)
      for (uint32_t inner = 0; inner < lane->layout_iteration0; ++inner) {
        uint32_t offset =
            outer * lane->layout_stride2_bytes +
            middle * lane->layout_stride1_bytes +
            inner * lane->layout_stride0_bytes;
        for (uint32_t byte = 0; byte < lane->layout_inner_bytes; ++byte)
          destination[offset + byte] = wafer_ncc_v2_pattern_byte(
              source_slot, compact_cursor + byte);
        compact_cursor += lane->layout_inner_bytes;
      }
}

static uint8_t wafer_ncc_v2_strided_final_byte(
    const WaferNccV2Context *context,
    const WaferNccProbeRequest *request, uint64_t address) {
  const WaferNccProbeLane *lane = &request->lanes[0];
  const WaferNccHazardComposition *composition = &context->hazards[0];
  uint32_t first_index = 0;
  uint32_t second_index = 0;
  int in_first = wafer_ncc_v2_dma_compact_index(
      lane, composition->first_range.begin, address, &first_index);
  int in_second = wafer_ncc_v2_dma_compact_index(
      lane, composition->second_range.begin, address, &second_index);
  switch (request->effect_relation) {
  case WAFER_NCC_EFFECT_RAW:
    return in_first ? wafer_ncc_v2_pattern_byte(0, first_index)
                    : UINT8_C(0xc3);
  case WAFER_NCC_EFFECT_WAR:
    return in_second
               ? wafer_ncc_v2_pattern_byte(
                     WAFER_NCC_PROTOCOL_MAX_ROUNDS, second_index)
               : UINT8_C(0xc3);
  case WAFER_NCC_EFFECT_WAW:
    if (in_second)
      return wafer_ncc_v2_pattern_byte(
          WAFER_NCC_PROTOCOL_MAX_ROUNDS, second_index);
    return in_first ? wafer_ncc_v2_pattern_byte(0, first_index)
                    : UINT8_C(0xc3);
  case WAFER_NCC_EFFECT_RAR:
    return in_first || in_second
               ? wafer_ncc_v2_pattern_byte(
                     WAFER_NCC_V2_STRIDED_INITIAL_SOURCE_SLOT,
                     in_first ? first_index : second_index)
               : UINT8_C(0xc3);
  default:
    return 0;
  }
}

static uint64_t wafer_ncc_v2_strided_final_mismatches(
    const WaferNccV2Context *context,
    const WaferNccProbeRequest *request) {
  const WaferNccHazardComposition *composition = &context->hazards[0];
  const volatile uint8_t *actual =
      wafer_ncc_probe_spm8(composition->footprint.begin);
  uint64_t mismatches = 0;
  uint32_t bytes = (uint32_t)(composition->footprint.end -
                              composition->footprint.begin);
  for (uint32_t byte = 0; byte < bytes; ++byte)
    mismatches +=
        actual[byte] != wafer_ncc_v2_strided_final_byte(
                            context, request,
                            composition->footprint.begin + byte);
  return mismatches;
}

static uint64_t wafer_ncc_v2_strided_mismatches(
    uint64_t address, const WaferNccProbeLane *lane, uint32_t source_slot) {
  const volatile uint8_t *actual = wafer_ncc_probe_spm8(address);
  uint32_t envelope_cursor = 0;
  uint32_t compact_cursor = 0;
  uint64_t mismatches = 0;
  for (uint32_t outer = 0; outer < lane->layout_iteration2; ++outer) {
    for (uint32_t middle = 0; middle < lane->layout_iteration1; ++middle) {
      for (uint32_t inner = 0; inner < lane->layout_iteration0; ++inner) {
        uint32_t offset =
            outer * lane->layout_stride2_bytes +
            middle * lane->layout_stride1_bytes +
            inner * lane->layout_stride0_bytes;
        mismatches += wafer_ncc_probe_mismatch8(
            actual + envelope_cursor, UINT8_C(0xc3),
            offset - envelope_cursor);
        for (uint32_t byte = 0; byte < lane->layout_inner_bytes; ++byte)
          mismatches +=
              actual[offset + byte] !=
              wafer_ncc_v2_pattern_byte(source_slot,
                                        compact_cursor + byte);
        compact_cursor += lane->layout_inner_bytes;
        envelope_cursor = offset + lane->layout_inner_bytes;
      }
    }
  }
  uint32_t envelope = wafer_ncc_v2_dma_envelope_bytes(lane);
  mismatches += wafer_ncc_probe_mismatch8(
      actual + envelope_cursor, UINT8_C(0xc3),
      envelope - envelope_cursor);
  return mismatches + (compact_cursor != lane->transfer_bytes);
}

static uint8_t wafer_ncc_v2_tdma_byte(uint32_t slot, uint32_t format,
                                      uint32_t index) {
  if ((Data_Format)format == Fmt_INT8)
    return (uint8_t)(UINT8_C(0x31) + slot * 7U);
  if ((Data_Format)format == Fmt_FP16) {
    uint16_t value = wafer_ncc_v2_positive_integer_f16(slot + 1U);
    return (uint8_t)((index & 1U) ? value >> 8 : value);
  }
  if ((Data_Format)format == Fmt_BF16)
    return (index & 1U) ? UINT8_C(0x3f) : UINT8_C(0x80);
  return 0;
}

static void wafer_ncc_v2_seed_region(uint64_t address, uint32_t bytes,
                                     uint8_t value) {
  wafer_ncc_probe_fill8(
      wafer_ncc_probe_spm8(address - WAFER_NCC_V2_GUARD_BYTES),
      UINT8_C(0x6d), bytes + 2U * WAFER_NCC_V2_GUARD_BYTES);
  wafer_ncc_probe_fill8(wafer_ncc_probe_spm8(address), value, bytes);
}

static uint64_t wafer_ncc_v2_guard_mismatches(uint64_t address,
                                               uint32_t bytes) {
  return wafer_ncc_probe_mismatch8(
             wafer_ncc_probe_spm8(address - WAFER_NCC_V2_GUARD_BYTES),
             UINT8_C(0x6d), WAFER_NCC_V2_GUARD_BYTES) +
         wafer_ncc_probe_mismatch8(
             wafer_ncc_probe_spm8(address + bytes), UINT8_C(0x6d),
             WAFER_NCC_V2_GUARD_BYTES);
}

static int wafer_ncc_v2_is_hazard_operand(
    const WaferNccProbeRequest *request, const WaferNccProbeIssue *issue,
    uint32_t operand) {
  if (request->effect_relation == WAFER_NCC_EFFECT_NONE)
    return 0;
  return operand ==
         (issue->lane == 0 ? request->first_operand
                           : request->second_operand);
}

static void wafer_ncc_v2_seed_hazard(
    WaferNccV2Context *context, const WaferNccProbeIssue *issue) {
  uint32_t bit = UINT32_C(1) << issue->round;
  if ((context->hazard_seeded_mask & bit) != 0)
    return;
  const WaferNccHazardComposition *composition =
      &context->hazards[issue->round];
  uint64_t guarded_begin = composition->canary_before.begin;
  uint64_t guarded_end = composition->canary_after.end;
  wafer_ncc_probe_fill8(wafer_ncc_probe_spm8(guarded_begin),
                        UINT8_C(0x6d),
                        (uint32_t)(guarded_end - guarded_begin));
  wafer_ncc_probe_fill16(
      wafer_ncc_probe_spm16(composition->footprint.begin),
      wafer_ncc_v2_hazard_initial_f16(issue->round),
      (uint32_t)((composition->footprint.end -
                  composition->footprint.begin) /
                 sizeof(uint16_t)));
  context->hazard_seeded_mask |= bit;
}

static uint32_t
wafer_ncc_v2_effect_compute(const WaferNccProbeRequest *request,
                            const WaferNccProbeLane *lane) {
  (void)request;
  (void)lane;
  return WAFER_NCC_MEMORY_READ0 | WAFER_NCC_MEMORY_READ1 |
         WAFER_NCC_MEMORY_WRITE;
}

static uint32_t
wafer_ncc_v2_effect_copy(const WaferNccProbeRequest *request,
                         const WaferNccProbeLane *lane) {
  (void)request;
  (void)lane;
  return WAFER_NCC_MEMORY_READ0 | WAFER_NCC_MEMORY_WRITE;
}

static uint32_t
wafer_ncc_v2_effect_fill(const WaferNccProbeRequest *request,
                         const WaferNccProbeLane *lane) {
  (void)request;
  (void)lane;
  return WAFER_NCC_MEMORY_WRITE;
}

static int wafer_ncc_v2_seed(void *opaque,
                             const WaferNccProbeRequest *request,
                             const WaferNccProbeIssue *issue) {
  WaferNccV2Context *context = (WaferNccV2Context *)opaque;
  if (wafer_ncc_v2_configure_context(context, request) != 0)
    return 1;
  uint32_t bytes = issue->lane_spec->transfer_bytes;
  if (bytes > WAFER_NCC_V2_MAX_TRANSFER_BYTES)
    return 1;
  uint64_t read0 = wafer_ncc_v2_operand_address(
      context, issue, WAFER_NCC_OPERAND_READ0);
  uint64_t read1 = wafer_ncc_v2_operand_address(
      context, issue, WAFER_NCC_OPERAND_READ1);
  uint64_t write = wafer_ncc_v2_operand_address(
      context, issue, WAFER_NCC_OPERAND_WRITE);
  if (context->ordered_producer_consumer) {
    uint32_t bit = UINT32_C(1) << issue->round;
    if ((context->hazard_seeded_mask & bit) == 0) {
      const WaferNccProbeLane *producer = &request->lanes[0];
      uint32_t shared_bytes =
          producer->engine == WAFER_NCC_ENGINE_NE
              ? wafer_ncc_v2_ne_output_span(producer)
              : producer->transfer_bytes;
      wafer_ncc_v2_seed_region(
          context->operands[0][WAFER_NCC_OPERAND_WRITE],
          shared_bytes, UINT8_C(0xc3));
      context->hazard_seeded_mask |= bit;
    }
  } else if (wafer_ncc_v2_is_dma_roundtrip(request) ||
             wafer_ncc_probe_is_strided_dependency(request)) {
    uint32_t bit = UINT32_C(1) << issue->round;
    if ((context->hazard_seeded_mask & bit) == 0) {
      wafer_ncc_v2_seed_region(
          context->hazards[issue->round].footprint.begin,
          (uint32_t)(context->hazards[issue->round].footprint.end -
                     context->hazards[issue->round].footprint.begin),
          UINT8_C(0xc3));
      if (wafer_ncc_probe_is_strided_dependency(request) &&
          (request->effect_relation == WAFER_NCC_EFFECT_WAR ||
           request->effect_relation == WAFER_NCC_EFFECT_RAR))
        wafer_ncc_v2_seed_strided_pattern(
            context->hazards[issue->round].first_range.begin,
            &request->lanes[0],
            WAFER_NCC_V2_STRIDED_INITIAL_SOURCE_SLOT);
      context->hazard_seeded_mask |= bit;
    }
  } else if (context->has_hazard)
    wafer_ncc_v2_seed_hazard(context, issue);
  switch (issue->engine) {
  case WAFER_NCC_ENGINE_CT: {
    if (bytes % sizeof(uint16_t) != 0 ||
        issue->lane_spec->element_format != Fmt_FP16)
      return 1;
    if (!wafer_ncc_v2_is_hazard_operand(request, issue,
                                        WAFER_NCC_OPERAND_READ0)) {
      wafer_ncc_v2_seed_region(read0, bytes, UINT8_C(0x73));
      wafer_ncc_probe_fill16(
          wafer_ncc_probe_spm16(read0),
          context->has_hazard
              ? UINT16_C(0x3c00)
              : wafer_ncc_v2_positive_integer_f16(issue->slot + 1U),
          bytes / sizeof(uint16_t));
    }
    if (!wafer_ncc_v2_is_hazard_operand(request, issue,
                                        WAFER_NCC_OPERAND_READ1)) {
      wafer_ncc_v2_seed_region(read1, bytes, UINT8_C(0x74));
      wafer_ncc_probe_fill16(wafer_ncc_probe_spm16(read1),
                             UINT16_C(0x3c00),
                             bytes / sizeof(uint16_t));
    }
    wafer_ncc_v2_seed_region(write, bytes, UINT8_C(0xc3));
    break;
  }
  case WAFER_NCC_ENGINE_NE: {
    if ((bytes != WAFER_NCC_V2_NE_PHYSICAL_BYTES &&
         bytes != WAFER_NCC_V2_NE_LARGE_RESULT_BYTES) ||
        issue->lane_spec->element_format != Fmt_FP16)
      return 1;
    uint32_t lhs_bytes = wafer_ncc_v2_ne_lhs_bytes(issue->lane_spec);
    uint32_t rhs_bytes = wafer_ncc_v2_ne_rhs_bytes(issue->lane_spec);
    uint32_t output_span = wafer_ncc_v2_ne_output_span(issue->lane_spec);
    if (!wafer_ncc_v2_is_hazard_operand(request, issue,
                                        WAFER_NCC_OPERAND_READ0)) {
      wafer_ncc_v2_seed_region(read0, lhs_bytes, 0);
      wafer_ncc_probe_fill16(
          wafer_ncc_probe_spm16(read0),
          wafer_ncc_v2_positive_integer_f16(issue->slot + 1U),
          lhs_bytes / sizeof(uint16_t));
    }
    wafer_ncc_v2_seed_region(read1, rhs_bytes, 0);
    wafer_ncc_v2_seed_region(write, output_span, UINT8_C(0xc3));
    volatile uint16_t *rhs = wafer_ncc_probe_spm16(read1);
    uint32_t logical_dim = wafer_ncc_v2_is_large_ne_lane(issue->lane_spec)
                               ? WAFER_NCC_V2_NE_LARGE_N
                               : WAFER_NCC_PROBE_NE_LOGICAL_DIM;
    for (uint32_t index = 0; index < logical_dim; ++index)
      rhs[index * logical_dim + index] =
          UINT16_C(0x3c00);
    break;
  }
  case WAFER_NCC_ENGINE_RDMA:
    if (issue->lane_spec->element_format !=
        ((context->has_hazard || context->double_slot_observation)
             ? Fmt_FP16
             : Fmt_INT8))
      return 1;
    if (!wafer_ncc_v2_is_hazard_operand(request, issue,
                                        WAFER_NCC_OPERAND_WRITE))
      wafer_ncc_v2_seed_region(write, bytes, UINT8_C(0xc3));
    break;
  case WAFER_NCC_ENGINE_WDMA:
    if (issue->lane_spec->element_format !=
        ((context->has_hazard || context->double_slot_observation)
             ? Fmt_FP16
             : Fmt_INT8))
      return 1;
    if (!wafer_ncc_v2_is_hazard_operand(request, issue,
                                        WAFER_NCC_OPERAND_READ0)) {
      wafer_ncc_v2_seed_region(read0, bytes, UINT8_C(0x73));
      for (uint32_t index = 0; index < bytes; ++index)
        wafer_ncc_probe_spm8(read0)[index] =
            wafer_ncc_v2_pattern_byte(issue->slot, index);
    }
    break;
  case WAFER_NCC_ENGINE_TDMA:
    if (issue->lane_spec->element_format != Fmt_INT8 &&
        issue->lane_spec->element_format != Fmt_FP16 &&
        issue->lane_spec->element_format != Fmt_BF16 &&
        issue->lane_spec->element_format != Fmt_BOOL)
      return 1;
    if (!wafer_ncc_v2_is_hazard_operand(request, issue,
                                        WAFER_NCC_OPERAND_WRITE))
      wafer_ncc_v2_seed_region(write, bytes, UINT8_C(0xc3));
    break;
  default:
    return 1;
  }
  __asm__ volatile("fence iorw, iorw" ::: "memory");
  return 0;
}

static int wafer_ncc_v2_prepare_ne(WaferNccProbeInstruction *instruction,
                                   uint32_t worker, uint64_t lhs, uint64_t rhs,
                                   uint64_t output,
                                   const WaferNccProbeLane *lane) {
  instruction->queue = WAFER_NCC_ENGINE_NE;
  instruction->owner = TsmNewGemm();
  instruction->has_owner = instruction->owner != NULL;
  if (!instruction->has_owner)
    return 0;
  TsmGemm *gemm = (TsmGemm *)instruction->owner;
  gemm->AddInput(&instruction->packet.ne, lhs, rhs, Fmt_FP16);
  if (wafer_ncc_v2_is_large_ne_lane(lane))
    gemm->ConfigMKN(&instruction->packet.ne, WAFER_NCC_V2_NE_LARGE_M,
                    WAFER_NCC_V2_NE_LARGE_K,
                    WAFER_NCC_V2_NE_LARGE_N);
  else
    gemm->ConfigMKN(&instruction->packet.ne, 1,
                    WAFER_NCC_PROBE_NE_LOGICAL_DIM,
                    WAFER_NCC_PROBE_NE_LOGICAL_DIM);
  gemm->ConfigBatch(&instruction->packet.ne, 1, 1);
  gemm->SetTransflag(&instruction->packet.ne, 0, 1);
  gemm->SetPsum(&instruction->packet.ne, 0, 0, Fmt_UNUSED);
  gemm->SetQuant(&instruction->packet.ne, 0, 0, 0, 0);
  gemm->AddBias(&instruction->packet.ne, 0, 0);
  gemm->SetNegativeAxisScale(&instruction->packet.ne, 0, 0);
  gemm->SetPositiveAxisScale(&instruction->packet.ne, 0, 0);
  gemm->DisableRelu(&instruction->packet.ne);
  gemm->DisableLeakyRelu(&instruction->packet.ne);
  gemm->AddOutput(&instruction->packet.ne, output, Fmt_FP16);
  wafer_ncc_probe_set_worker(&instruction->packet.ne.inter_type, worker);
  return 1;
}

static int wafer_ncc_v2_prepare_tdma(
    WaferNccProbeInstruction *instruction,
    const WaferNccProbeRequest *request, const WaferNccProbeIssue *issue,
    uint64_t destination) {
  uint32_t format = issue->lane_spec->element_format;
  uint32_t bytes = issue->lane_spec->transfer_bytes;
  uint32_t inner_bytes =
      issue->lane_spec->layout_kind == WAFER_NCC_LAYOUT_INNER_STRIDED
          ? issue->lane_spec->layout_inner_bytes
          : bytes;
  uint32_t value =
      request->flags == WAFER_NCC_REQUEST_ORDERED_PRODUCER_CONSUMER
          ? wafer_ncc_v2_positive_integer_f16(4U)
      : request->effect_relation != WAFER_NCC_EFFECT_NONE
          ? wafer_ncc_v2_hazard_second_write_f16(issue->round)
      : format == Fmt_INT8
          ? (uint32_t)wafer_ncc_v2_tdma_byte(issue->slot, format, 0)
          : format == Fmt_FP16
                ? wafer_ncc_v2_positive_integer_f16(issue->slot + 1U)
                : UINT32_C(0x3f80);
  St_StrideIteration stride = {
      inner_bytes, bytes / inner_bytes, 0, 1, 0, 1,
  };
  instruction->queue = WAFER_NCC_ENGINE_TDMA;
  instruction->owner = TsmNewPeripheral();
  instruction->has_owner = instruction->owner != NULL;
  if (!instruction->has_owner)
    return 0;
  TsmPeripheral *peripheral = (TsmPeripheral *)instruction->owner;
  peripheral->Memset(&instruction->packet.tdma, destination, value,
                     wafer_ncc_probe_tdma_elements(inner_bytes, format),
                     &stride, (Data_Format)format);
  wafer_ncc_probe_set_worker(&instruction->packet.tdma.inter_type,
                             issue->worker);
  return 1;
}

static int wafer_ncc_v2_prepare(void *opaque,
                                const WaferNccProbeRequest *request,
                                const WaferNccProbeIssue *issue,
                                uint64_t *preparation_flags) {
  WaferNccV2Context *context = (WaferNccV2Context *)opaque;
  WaferNccProbeInstruction *instruction =
      &context->instructions[issue->slot];
  uint32_t bytes = issue->lane_spec->transfer_bytes;
  uint64_t read0 = wafer_ncc_v2_operand_address(
      context, issue, WAFER_NCC_OPERAND_READ0);
  uint64_t read1 = wafer_ncc_v2_operand_address(
      context, issue, WAFER_NCC_OPERAND_READ1);
  uint64_t write = wafer_ncc_v2_operand_address(
      context, issue, WAFER_NCC_OPERAND_WRITE);
  if (issue->lane_spec->issue_mode == WAFER_NCC_ISSUE_WRAPPER)
    return issue->worker == 0 ? 0 : 1;
  int built = 0;
  switch (issue->engine) {
  case WAFER_NCC_ENGINE_CT:
    built = wafer_ncc_probe_prepare_ct(instruction, issue->worker, read0,
                                       read1, write,
                                       bytes / sizeof(uint16_t));
    break;
  case WAFER_NCC_ENGINE_NE:
    built = wafer_ncc_v2_prepare_ne(instruction, issue->worker, read0, read1,
                                    write, issue->lane_spec);
    break;
  case WAFER_NCC_ENGINE_RDMA:
    built = wafer_ncc_probe_prepare_rdma(
        instruction, issue->worker,
        wafer_ncc_v2_payload_address(context, issue->slot), write, bytes);
    break;
  case WAFER_NCC_ENGINE_WDMA:
    built = wafer_ncc_probe_prepare_wdma(
        instruction, issue->worker, read0,
        wafer_ncc_v2_output_address(context, issue->slot), bytes);
    break;
  case WAFER_NCC_ENGINE_TDMA:
    built = wafer_ncc_v2_prepare_tdma(instruction, request, issue, write);
    break;
  default:
    return 1;
  }
  if (!instruction->has_owner)
    return 1;
  *preparation_flags |= WAFER_NCC_ISSUE_PREPARE_BUILDER_ACQUIRED;
  if (!built) {
    wafer_ncc_probe_release(instruction);
    *preparation_flags |= WAFER_NCC_ISSUE_PREPARE_BUILDER_RELEASED;
    return 1;
  }
  *preparation_flags |= WAFER_NCC_ISSUE_PREPARE_PACKET_MATERIALIZED;
  if (request->flags == WAFER_NCC_REQUEST_CONSTRUCTOR_OBSERVATION) {
    context->constructor_address = (uint64_t)(uintptr_t)instruction->owner;
    context->constructor_captured = 1;
  }
  /*
   * TsmNew* owns only the method table used to materialize the packet.  The
   * packet itself lives in the probe context and is the sole TsmExecute input.
   * Release each constructor immediately so a window never retains one heap
   * object per queued packet.
   */
  wafer_ncc_probe_release(instruction);
  *preparation_flags |= WAFER_NCC_ISSUE_PREPARE_BUILDER_RELEASED;
  return 0;
}

static int wafer_ncc_v2_issue(void *opaque,
                              const WaferNccProbeRequest *request,
                              const WaferNccProbeIssue *issue,
                              uint64_t *execute_rc) {
  WaferNccV2Context *context = (WaferNccV2Context *)opaque;
  uint32_t bytes = issue->lane_spec->transfer_bytes;
  uint64_t read0 = wafer_ncc_v2_operand_address(
      context, issue, WAFER_NCC_OPERAND_READ0);
  uint64_t read1 = wafer_ncc_v2_operand_address(
      context, issue, WAFER_NCC_OPERAND_READ1);
  uint64_t write = wafer_ncc_v2_operand_address(
      context, issue, WAFER_NCC_OPERAND_WRITE);
  if (issue->engine == WAFER_NCC_ENGINE_RDMA && bytes != 0) {
    context->completion_marker_address =
        write + wafer_ncc_v2_dma_envelope_bytes(issue->lane_spec) - 1U;
    context->completion_marker_expected =
        context->double_slot_observation
            ? (uint8_t)(wafer_ncc_v2_positive_integer_f16(
                            issue->round + 1U) >>
                        8)
            : wafer_ncc_v2_pattern_byte(issue->slot, bytes - 1U);
  } else if (issue->engine == WAFER_NCC_ENGINE_NE) {
    uint16_t expected =
        wafer_ncc_v2_positive_integer_f16(issue->slot + 1U);
    context->completion_marker_address =
        write + wafer_ncc_v2_ne_result_bytes(issue->lane_spec) - 1U;
    context->completion_marker_expected = (uint8_t)(expected >> 8);
  }
  if (issue->lane_spec->issue_mode == WAFER_NCC_ISSUE_RAW) {
    WaferNccProbeInstruction *instruction =
        &context->instructions[issue->slot];
    context->issued_inter_types[issue->slot] =
        wafer_ncc_probe_packet_inter_type(instruction);
    *execute_rc = wafer_ncc_probe_issue_raw(instruction);
    return 0;
  }

  *execute_rc = UINT64_MAX;
  switch (issue->engine) {
  case WAFER_NCC_ENGINE_CT:
    wafer_tx81_elementwise_add(read0, read1, write,
                               bytes / sizeof(uint16_t), Fmt_FP16);
    break;
  case WAFER_NCC_ENGINE_NE:
    if (wafer_ncc_v2_is_large_ne_lane(issue->lane_spec))
      wafer_tx81_gemm(read0, read1, write, WAFER_NCC_V2_NE_LARGE_M,
                      WAFER_NCC_V2_NE_LARGE_K,
                      WAFER_NCC_V2_NE_LARGE_N, 1, Fmt_FP16);
    else
      wafer_tx81_gemm(read0, read1, write, 1,
                      WAFER_NCC_PROBE_NE_LOGICAL_DIM,
                      WAFER_NCC_PROBE_NE_LOGICAL_DIM, 1, Fmt_FP16);
    break;
  case WAFER_NCC_ENGINE_RDMA:
    if (issue->lane_spec->layout_kind == WAFER_NCC_LAYOUT_DMA_STRIDED)
      wafer_tx81_rdma(
          wafer_ncc_v2_payload_address(context, issue->slot), write, bytes,
          issue->lane_spec->layout_inner_bytes,
          issue->lane_spec->layout_stride0_bytes,
          issue->lane_spec->layout_stride1_bytes,
          issue->lane_spec->layout_stride2_bytes,
          issue->lane_spec->layout_iteration0,
          issue->lane_spec->layout_iteration1,
          issue->lane_spec->layout_iteration2,
          issue->lane_spec->element_format);
    else
      wafer_tx81_rdma(wafer_ncc_v2_payload_address(context, issue->slot), write,
                      bytes, bytes, 0, 0, 0, 1, 1, 1, Fmt_UINT8);
    break;
  case WAFER_NCC_ENGINE_WDMA:
    if (issue->lane_spec->layout_kind == WAFER_NCC_LAYOUT_DMA_STRIDED)
      wafer_tx81_wdma(
          read0, wafer_ncc_v2_output_address(context, issue->slot), bytes,
          issue->lane_spec->layout_inner_bytes,
          issue->lane_spec->layout_stride0_bytes,
          issue->lane_spec->layout_stride1_bytes,
          issue->lane_spec->layout_stride2_bytes,
          issue->lane_spec->layout_iteration0,
          issue->lane_spec->layout_iteration1,
          issue->lane_spec->layout_iteration2,
          issue->lane_spec->element_format);
    else
      wafer_tx81_wdma(read0, wafer_ncc_v2_output_address(context, issue->slot),
                      bytes, bytes, 0, 0, 0, 1, 1, 1, Fmt_UINT8);
    break;
  case WAFER_NCC_ENGINE_TDMA: {
    uint32_t format = issue->lane_spec->element_format;
    uint32_t value =
        request->flags == WAFER_NCC_REQUEST_ORDERED_PRODUCER_CONSUMER
            ? wafer_ncc_v2_positive_integer_f16(4U)
        : request->effect_relation != WAFER_NCC_EFFECT_NONE
            ? wafer_ncc_v2_hazard_second_write_f16(issue->round)
        : format == Fmt_INT8
            ? (uint32_t)wafer_ncc_v2_tdma_byte(issue->slot, format, 0)
            : format == Fmt_BOOL
                ? UINT32_C(1)
            : format == Fmt_FP16
                  ? wafer_ncc_v2_positive_integer_f16(issue->slot + 1U)
                  : UINT32_C(0x3f80);
    wafer_tx81_memset(write, value,
                      wafer_ncc_probe_tdma_elements(bytes, format), format);
    break;
  }
  default:
    return 1;
  }
  return 0;
}

static int wafer_ncc_v2_observe_raw_registers(
    const WaferNccProbeIssue *issue,
    WaferNccProbeObservation *observation) {
  uint32_t worker = issue->worker;
  observation->flags = WAFER_NCC_ISSUE_PACKET_OBSERVED;
  switch (issue->engine) {
  case WAFER_NCC_ENGINE_CT:
    observation->read0_begin = get_ncc_reg(worker, GR_CT_SRC0_ADDR);
    observation->read0_end = get_ncc_reg(worker, GR_CT_SRC0_END_ADDR);
    observation->read1_begin = get_ncc_reg(worker, GR_CT_SRC1_ADDR);
    observation->read1_end = get_ncc_reg(worker, GR_CT_SRC1_END_ADDR);
    observation->write_begin = get_ncc_reg(worker, GR_CT_DST0_ADDR);
    observation->write_end = get_ncc_reg(worker, GR_CT_DST0_END_ADDR);
    observation->flags |= WAFER_NCC_ISSUE_READ0_VALID |
                          WAFER_NCC_ISSUE_READ1_VALID |
                          WAFER_NCC_ISSUE_WRITE_VALID;
    return 0;
  case WAFER_NCC_ENGINE_NE:
    observation->read0_begin = get_ncc_reg(worker, GR_NE_SRC_A_ADDR);
    observation->read0_end = get_ncc_reg(worker, GR_NE_SRCA_END);
    observation->read1_begin = get_ncc_reg(worker, GR_NE_SRC_W_ADDR);
    observation->read1_end = get_ncc_reg(worker, GR_NE_SRCW_END);
    observation->write_begin = get_ncc_reg(worker, GR_NE_OUT_ADDR);
    observation->write_end = get_ncc_reg(worker, GR_NE_OUT_END);
    observation->flags |= WAFER_NCC_ISSUE_READ0_VALID |
                          WAFER_NCC_ISSUE_READ1_VALID |
                          WAFER_NCC_ISSUE_WRITE_VALID;
    return 0;
  case WAFER_NCC_ENGINE_RDMA:
    observation->read0_begin = get_ncc_reg(worker, GR_RD_SRC_ADDR);
    observation->read0_end = get_ncc_reg(worker, GR_RD_SRC_END);
    observation->write_begin = get_ncc_reg(worker, GR_RD_DST_ADDR);
    observation->write_end = get_ncc_reg(worker, GR_RD_DST_END);
    observation->flags |=
        WAFER_NCC_ISSUE_READ0_VALID | WAFER_NCC_ISSUE_WRITE_VALID;
    return 0;
  case WAFER_NCC_ENGINE_WDMA:
    observation->read0_begin = get_ncc_reg(worker, GR_WD_SRC_ADDR);
    observation->read0_end = get_ncc_reg(worker, GR_WD_SRC_END);
    observation->write_begin = get_ncc_reg(worker, GR_WD_DST_ADDR);
    observation->write_end = get_ncc_reg(worker, GR_WD_DST_END);
    observation->flags |=
        WAFER_NCC_ISSUE_READ0_VALID | WAFER_NCC_ISSUE_WRITE_VALID;
    return 0;
  case WAFER_NCC_ENGINE_TDMA:
    observation->write_begin = get_ncc_reg(worker, GR_TD_DST_ADDR);
    observation->write_end = get_ncc_reg(worker, GR_TD_DST_END);
    observation->flags |= WAFER_NCC_ISSUE_WRITE_VALID;
    return 0;
  default:
    return 1;
  }
}

static int wafer_ncc_v2_observe(void *opaque,
                                const WaferNccProbeRequest *request,
                                const WaferNccProbeIssue *issue,
                                WaferNccProbeObservation *observation) {
  (void)request;
  WaferNccV2Context *context = (WaferNccV2Context *)opaque;
  WaferNccProbeInstruction *instruction =
      &context->instructions[issue->slot];
  uint32_t bytes = issue->lane_spec->transfer_bytes;
  uint64_t read0 = wafer_ncc_v2_operand_address(
      context, issue, WAFER_NCC_OPERAND_READ0);
  uint64_t read1 = wafer_ncc_v2_operand_address(
      context, issue, WAFER_NCC_OPERAND_READ1);
  uint64_t write = wafer_ncc_v2_operand_address(
      context, issue, WAFER_NCC_OPERAND_WRITE);
  int raw_issue =
      issue->lane_spec->issue_mode == WAFER_NCC_ISSUE_RAW;
  if (raw_issue &&
      request->flags != WAFER_NCC_REQUEST_TIGHT_DEPTH_PLUS_ONE) {
    observation->inter_type = context->issued_inter_types[issue->slot];
    return wafer_ncc_v2_observe_raw_registers(issue, observation);
  }
  (void)instruction;
  observation->inter_type = raw_issue
                                ? context->issued_inter_types[issue->slot]
                                : ((uint64_t)issue->engine |
                                   ((uint64_t)issue->worker << 8));
  observation->flags = 0;
  switch (issue->engine) {
  case WAFER_NCC_ENGINE_CT:
    observation->read0_begin = read0;
    observation->read0_end = read0 + bytes - 1U;
    observation->read1_begin = read1;
    observation->read1_end = read1 + bytes - 1U;
    observation->write_begin = write;
    observation->write_end =
        write + wafer_ncc_v2_dma_envelope_bytes(issue->lane_spec) - 1U;
    observation->flags |= WAFER_NCC_ISSUE_READ0_VALID |
                          WAFER_NCC_ISSUE_READ1_VALID |
                          WAFER_NCC_ISSUE_WRITE_VALID;
    break;
  case WAFER_NCC_ENGINE_NE:
    observation->read0_begin = read0;
    observation->read0_end =
        read0 + wafer_ncc_v2_ne_lhs_bytes(issue->lane_spec) - 1U;
    observation->read1_begin = read1;
    observation->read1_end =
        read1 + wafer_ncc_v2_ne_rhs_bytes(issue->lane_spec) - 1U;
    observation->write_begin = write;
    observation->write_end =
        write + wafer_ncc_v2_ne_output_span(issue->lane_spec) - 1U;
    observation->flags |= WAFER_NCC_ISSUE_READ0_VALID |
                          WAFER_NCC_ISSUE_READ1_VALID |
                          WAFER_NCC_ISSUE_WRITE_VALID;
    break;
  case WAFER_NCC_ENGINE_RDMA:
    observation->read0_begin =
        wafer_ncc_v2_payload_address(context, issue->slot);
    observation->read0_end =
        observation->read0_begin +
        wafer_ncc_v2_dma_envelope_bytes(issue->lane_spec) - 1U;
    observation->write_begin = write;
    observation->write_end =
        write + wafer_ncc_v2_dma_envelope_bytes(issue->lane_spec) - 1U;
    observation->flags |=
        WAFER_NCC_ISSUE_READ0_VALID | WAFER_NCC_ISSUE_WRITE_VALID;
    break;
  case WAFER_NCC_ENGINE_WDMA:
    observation->read0_begin = read0;
    observation->read0_end =
        read0 + wafer_ncc_v2_dma_envelope_bytes(issue->lane_spec) - 1U;
    observation->write_begin =
        wafer_ncc_v2_output_address(context, issue->slot);
    observation->write_end =
        observation->write_begin +
        wafer_ncc_v2_dma_envelope_bytes(issue->lane_spec) - 1U;
    observation->flags |=
        WAFER_NCC_ISSUE_READ0_VALID | WAFER_NCC_ISSUE_WRITE_VALID;
    break;
  case WAFER_NCC_ENGINE_TDMA:
    observation->write_begin = write;
    observation->write_end = write + bytes - 1U;
    observation->flags |= WAFER_NCC_ISSUE_WRITE_VALID;
    break;
  default:
    return 1;
  }
  return 0;
}

static uint32_t wafer_ncc_v2_hazard_source_value(uint32_t round,
                                                 uint32_t source) {
  switch (source) {
  case WAFER_NCC_HAZARD_EXPECTED_INITIAL:
    return 2U + round;
  case WAFER_NCC_HAZARD_EXPECTED_FIRST_WRITE:
    return 4U + round;
  case WAFER_NCC_HAZARD_EXPECTED_SECOND_WRITE:
    return 8U + round;
  default:
    return 0;
  }
}

static uint32_t wafer_ncc_v2_hazard_source_at(
    const WaferNccHazardExpectedSpan *spans, uint32_t span_count,
    uint64_t address) {
  for (uint32_t index = 0; index < span_count; ++index)
    if (spans[index].range.begin <= address &&
        address < spans[index].range.end)
      return spans[index].source;
  return UINT32_MAX;
}

static uint64_t wafer_ncc_v2_hazard_final_mismatches(
    const WaferNccHazardComposition *composition, uint32_t round) {
  uint64_t mismatches = 0;
  for (uint32_t span = 0; span < composition->final_span_count; ++span) {
    const WaferNccHazardExpectedSpan *expected =
        &composition->final_spans[span];
    uint16_t value = wafer_ncc_v2_positive_integer_f16(
        wafer_ncc_v2_hazard_source_value(round, expected->source));
    const volatile uint16_t *actual =
        wafer_ncc_probe_spm16(expected->range.begin);
    uint32_t elements =
        (uint32_t)((expected->range.end - expected->range.begin) /
                   sizeof(uint16_t));
    for (uint32_t index = 0; index < elements; ++index)
      mismatches += actual[index] != value;
  }
  return mismatches;
}

static uint64_t wafer_ncc_v2_hazard_canary_mismatches(
    const WaferNccHazardComposition *composition) {
  return wafer_ncc_probe_mismatch8(
             wafer_ncc_probe_spm8(composition->canary_before.begin),
             UINT8_C(0x6d),
             (uint32_t)(composition->canary_before.end -
                        composition->canary_before.begin)) +
         wafer_ncc_probe_mismatch8(
             wafer_ncc_probe_spm8(composition->canary_after.begin),
             UINT8_C(0x6d),
             (uint32_t)(composition->canary_after.end -
                        composition->canary_after.begin));
}

static uint64_t wafer_ncc_v2_hazard_ct_mismatches(
    const WaferNccV2Context *context, const WaferNccProbeIssue *issue,
    uint64_t output, uint32_t bytes) {
  const WaferNccHazardComposition *composition =
      &context->hazards[issue->round];
  WaferNccHazardRange selected =
      issue->lane == 0 ? composition->first_range
                       : composition->second_range;
  const volatile uint16_t *actual = wafer_ncc_probe_spm16(output);
  uint64_t mismatches = 0;
  for (uint32_t index = 0; index < bytes / sizeof(uint16_t); ++index) {
    uint64_t address = selected.begin + index * sizeof(uint16_t);
    uint32_t source = WAFER_NCC_HAZARD_EXPECTED_INITIAL;
    if (issue->lane == 1)
      source = wafer_ncc_v2_hazard_source_at(
          composition->second_read_spans,
          composition->second_read_span_count, address);
    uint32_t value = wafer_ncc_v2_hazard_source_value(issue->round, source);
    uint16_t expected = wafer_ncc_v2_positive_integer_f16(value + 1U);
    mismatches += source == UINT32_MAX || actual[index] != expected;
  }
  return mismatches;
}

static int wafer_ncc_v2_hazard_oracle(
    WaferNccV2Context *context, const WaferNccProbeRequest *request,
    const WaferNccProbeIssue *issue, uint64_t *result_mismatches,
    uint64_t *guard_mismatches) {
  uint32_t bytes = issue->lane_spec->transfer_bytes;
  uint64_t read0 = wafer_ncc_v2_operand_address(
      context, issue, WAFER_NCC_OPERAND_READ0);
  uint64_t read1 = wafer_ncc_v2_operand_address(
      context, issue, WAFER_NCC_OPERAND_READ1);
  uint64_t write = wafer_ncc_v2_operand_address(
      context, issue, WAFER_NCC_OPERAND_WRITE);
  if (wafer_ncc_v2_is_dma_roundtrip(request)) {
    if (issue->lane == 0) {
      *result_mismatches += wafer_ncc_v2_strided_mismatches(
          context->hazards[issue->round].first_range.begin,
          issue->lane_spec, issue->round);
      *guard_mismatches += wafer_ncc_v2_hazard_canary_mismatches(
          &context->hazards[issue->round]);
    }
    return 0;
  }
  if (wafer_ncc_probe_is_strided_dependency(request)) {
    if (issue->lane == 0) {
      *result_mismatches +=
          wafer_ncc_v2_strided_final_mismatches(context, request);
      *guard_mismatches += wafer_ncc_v2_hazard_canary_mismatches(
          &context->hazards[issue->round]);
    }
    return 0;
  }
  if (issue->engine == WAFER_NCC_ENGINE_CT)
    *result_mismatches +=
        wafer_ncc_v2_hazard_ct_mismatches(context, issue, write, bytes);

  if (issue->lane == 0) {
    *result_mismatches += wafer_ncc_v2_hazard_final_mismatches(
        &context->hazards[issue->round], issue->round);
    *guard_mismatches += wafer_ncc_v2_hazard_canary_mismatches(
        &context->hazards[issue->round]);
  }

  if (issue->engine == WAFER_NCC_ENGINE_CT) {
    if (!wafer_ncc_v2_is_hazard_operand(request, issue,
                                        WAFER_NCC_OPERAND_READ0))
      *guard_mismatches += wafer_ncc_v2_guard_mismatches(read0, bytes);
    if (!wafer_ncc_v2_is_hazard_operand(request, issue,
                                        WAFER_NCC_OPERAND_READ1))
      *guard_mismatches += wafer_ncc_v2_guard_mismatches(read1, bytes);
    *guard_mismatches += wafer_ncc_v2_guard_mismatches(write, bytes);
  } else if (issue->engine == WAFER_NCC_ENGINE_RDMA) {
    if (!wafer_ncc_v2_is_hazard_operand(request, issue,
                                        WAFER_NCC_OPERAND_WRITE))
      *guard_mismatches += wafer_ncc_v2_guard_mismatches(write, bytes);
  } else if (issue->engine == WAFER_NCC_ENGINE_WDMA) {
    if (!wafer_ncc_v2_is_hazard_operand(request, issue,
                                        WAFER_NCC_OPERAND_READ0))
      *guard_mismatches += wafer_ncc_v2_guard_mismatches(read0, bytes);
  } else if (issue->engine == WAFER_NCC_ENGINE_TDMA) {
    if (!wafer_ncc_v2_is_hazard_operand(request, issue,
                                        WAFER_NCC_OPERAND_WRITE))
      *guard_mismatches += wafer_ncc_v2_guard_mismatches(write, bytes);
  } else {
    return 1;
  }
  return 0;
}

static uint64_t wafer_ncc_v2_uniform_f16_mismatches(
    uint64_t address, uint32_t bytes, uint16_t expected) {
  const volatile uint16_t *actual = wafer_ncc_probe_spm16(address);
  uint64_t mismatches = 0;
  for (uint32_t index = 0; index < bytes / sizeof(uint16_t); ++index)
    mismatches += actual[index] != expected;
  return mismatches;
}

static int wafer_ncc_v2_ordered_oracle(
    WaferNccV2Context *context, const WaferNccProbeRequest *request,
    const WaferNccProbeIssue *issue, uint64_t *result_mismatches,
    uint64_t *guard_mismatches) {
  const WaferNccProbeLane *producer = &request->lanes[0];
  uint64_t read0 = wafer_ncc_v2_operand_address(
      context, issue, WAFER_NCC_OPERAND_READ0);
  uint64_t read1 = wafer_ncc_v2_operand_address(
      context, issue, WAFER_NCC_OPERAND_READ1);
  uint64_t write = wafer_ncc_v2_operand_address(
      context, issue, WAFER_NCC_OPERAND_WRITE);
  uint32_t producer_span =
      producer->engine == WAFER_NCC_ENGINE_NE
          ? wafer_ncc_v2_ne_output_span(producer)
          : producer->transfer_bytes;
  if (issue->lane == 0) {
    uint16_t expected = UINT16_C(0);
    if (issue->engine == WAFER_NCC_ENGINE_RDMA ||
        issue->engine == WAFER_NCC_ENGINE_TDMA)
      expected = wafer_ncc_v2_positive_integer_f16(4U);
    else if (issue->engine == WAFER_NCC_ENGINE_CT)
      expected = wafer_ncc_v2_positive_integer_f16(2U);
    else if (issue->engine == WAFER_NCC_ENGINE_NE)
      expected = wafer_ncc_v2_positive_integer_f16(1U);
    else
      return 1;
    uint32_t result_bytes =
        issue->engine == WAFER_NCC_ENGINE_NE
            ? wafer_ncc_v2_ne_result_bytes(issue->lane_spec)
            : producer_span;
    *result_mismatches +=
        wafer_ncc_v2_uniform_f16_mismatches(write, result_bytes, expected);
    *guard_mismatches +=
        wafer_ncc_v2_guard_mismatches(write, producer_span);
    if (issue->engine == WAFER_NCC_ENGINE_CT) {
      *guard_mismatches +=
          wafer_ncc_v2_guard_mismatches(read0,
                                        issue->lane_spec->transfer_bytes) +
          wafer_ncc_v2_guard_mismatches(read1,
                                        issue->lane_spec->transfer_bytes);
    } else if (issue->engine == WAFER_NCC_ENGINE_NE) {
      *guard_mismatches +=
          wafer_ncc_v2_guard_mismatches(
              read0, wafer_ncc_v2_ne_lhs_bytes(issue->lane_spec)) +
          wafer_ncc_v2_guard_mismatches(
              read1, wafer_ncc_v2_ne_rhs_bytes(issue->lane_spec));
    }
    return 0;
  }

  if (issue->engine == WAFER_NCC_ENGINE_CT) {
    *result_mismatches += wafer_ncc_v2_uniform_f16_mismatches(
        write, issue->lane_spec->transfer_bytes,
        wafer_ncc_v2_positive_integer_f16(5U));
    *guard_mismatches +=
        wafer_ncc_v2_guard_mismatches(read1,
                                      issue->lane_spec->transfer_bytes) +
        wafer_ncc_v2_guard_mismatches(write,
                                      issue->lane_spec->transfer_bytes);
  } else if (issue->engine == WAFER_NCC_ENGINE_NE) {
    *result_mismatches += wafer_ncc_v2_uniform_f16_mismatches(
        write, wafer_ncc_v2_ne_result_bytes(issue->lane_spec),
        wafer_ncc_v2_positive_integer_f16(4U));
    *guard_mismatches +=
        wafer_ncc_v2_guard_mismatches(
            read1, wafer_ncc_v2_ne_rhs_bytes(issue->lane_spec)) +
        wafer_ncc_v2_guard_mismatches(
            write, wafer_ncc_v2_ne_output_span(issue->lane_spec));
  } else if (issue->engine == WAFER_NCC_ENGINE_WDMA) {
    uint16_t expected =
        wafer_ncc_v2_positive_integer_f16(
            producer->engine == WAFER_NCC_ENGINE_NE ? 1U : 2U);
    *result_mismatches += wafer_ncc_v2_uniform_f16_mismatches(
        read0, issue->lane_spec->transfer_bytes, expected);
  } else {
    return 1;
  }
  return 0;
}

static int wafer_ncc_v2_double_slot_oracle(
    WaferNccV2Context *context, const WaferNccProbeRequest *request,
    const WaferNccProbeIssue *issue, uint64_t *result_mismatches,
    uint64_t *guard_mismatches) {
  uint64_t read0 = wafer_ncc_v2_operand_address(
      context, issue, WAFER_NCC_OPERAND_READ0);
  uint64_t read1 = wafer_ncc_v2_operand_address(
      context, issue, WAFER_NCC_OPERAND_READ1);
  uint64_t write = wafer_ncc_v2_operand_address(
      context, issue, WAFER_NCC_OPERAND_WRITE);
  uint32_t latest = request->rounds - 1U;
  if ((latest & 1U) != (issue->round & 1U))
    --latest;
  uint16_t input_expected =
      wafer_ncc_v2_positive_integer_f16(latest + 1U);
  uint16_t output_expected =
      wafer_ncc_v2_positive_integer_f16(latest + 2U);
  if (issue->round == latest) {
    if (issue->engine == WAFER_NCC_ENGINE_RDMA)
      *result_mismatches += wafer_ncc_v2_uniform_f16_mismatches(
          write, issue->lane_spec->transfer_bytes, input_expected);
    else if (issue->engine == WAFER_NCC_ENGINE_CT)
      *result_mismatches += wafer_ncc_v2_uniform_f16_mismatches(
          write, issue->lane_spec->transfer_bytes, output_expected);
    else if (issue->engine == WAFER_NCC_ENGINE_WDMA)
      *result_mismatches += wafer_ncc_v2_uniform_f16_mismatches(
          read0, issue->lane_spec->transfer_bytes, output_expected);
    else
      return 1;
  }
  if (issue->engine == WAFER_NCC_ENGINE_RDMA)
    *guard_mismatches += wafer_ncc_v2_guard_mismatches(
        write, issue->lane_spec->transfer_bytes);
  else if (issue->engine == WAFER_NCC_ENGINE_CT)
    *guard_mismatches +=
        wafer_ncc_v2_guard_mismatches(read1,
                                      issue->lane_spec->transfer_bytes) +
        wafer_ncc_v2_guard_mismatches(write,
                                      issue->lane_spec->transfer_bytes);
  return 0;
}

static int wafer_ncc_v2_oracle(void *opaque,
                               const WaferNccProbeRequest *request,
                               const WaferNccProbeIssue *issue, uint32_t phase,
                               uint64_t *result_mismatches,
                               uint64_t *guard_mismatches) {
  WaferNccV2Context *context = (WaferNccV2Context *)opaque;
  (void)phase;
  uint32_t bytes = issue->lane_spec->transfer_bytes;
  uint64_t read0 = wafer_ncc_v2_operand_address(
      context, issue, WAFER_NCC_OPERAND_READ0);
  uint64_t read1 = wafer_ncc_v2_operand_address(
      context, issue, WAFER_NCC_OPERAND_READ1);
  uint64_t write = wafer_ncc_v2_operand_address(
      context, issue, WAFER_NCC_OPERAND_WRITE);
  *result_mismatches = 0;
  *guard_mismatches = 0;
  if (context->double_slot_observation)
    return wafer_ncc_v2_double_slot_oracle(
        context, request, issue, result_mismatches, guard_mismatches);
  if (context->ordered_producer_consumer)
    return wafer_ncc_v2_ordered_oracle(
        context, request, issue, result_mismatches, guard_mismatches);
  if (context->has_hazard)
    return wafer_ncc_v2_hazard_oracle(
        context, request, issue, result_mismatches, guard_mismatches);
  switch (issue->engine) {
  case WAFER_NCC_ENGINE_CT: {
    uint16_t expected =
        wafer_ncc_v2_positive_integer_f16(issue->slot + 2U);
    const volatile uint16_t *actual = wafer_ncc_probe_spm16(write);
    for (uint32_t index = 0; index < bytes / sizeof(uint16_t); ++index)
      *result_mismatches += actual[index] != expected;
    *guard_mismatches =
        wafer_ncc_v2_guard_mismatches(read0, bytes) +
        wafer_ncc_v2_guard_mismatches(read1, bytes) +
        wafer_ncc_v2_guard_mismatches(write, bytes);
    break;
  }
  case WAFER_NCC_ENGINE_NE: {
    uint16_t expected =
        wafer_ncc_v2_positive_integer_f16(issue->slot + 1U);
    const volatile uint16_t *actual = wafer_ncc_probe_spm16(write);
    uint32_t result_elements =
        wafer_ncc_v2_ne_result_bytes(issue->lane_spec) / sizeof(uint16_t);
    for (uint32_t index = 0; index < result_elements;
         ++index)
      *result_mismatches += actual[index] != expected;
    *guard_mismatches =
        wafer_ncc_v2_guard_mismatches(
            read0, wafer_ncc_v2_ne_lhs_bytes(issue->lane_spec)) +
        wafer_ncc_v2_guard_mismatches(
            read1, wafer_ncc_v2_ne_rhs_bytes(issue->lane_spec)) +
        wafer_ncc_v2_guard_mismatches(
            write, wafer_ncc_v2_ne_output_span(issue->lane_spec));
    break;
  }
  case WAFER_NCC_ENGINE_RDMA: {
    const volatile uint8_t *actual = wafer_ncc_probe_spm8(write);
    for (uint32_t index = 0; index < bytes; ++index)
      *result_mismatches +=
          actual[index] != wafer_ncc_v2_pattern_byte(issue->slot, index);
    *guard_mismatches = wafer_ncc_v2_guard_mismatches(write, bytes);
    break;
  }
  case WAFER_NCC_ENGINE_WDMA: {
    const volatile uint8_t *actual = wafer_ncc_probe_spm8(read0);
    for (uint32_t index = 0; index < bytes; ++index)
      *result_mismatches +=
          actual[index] != wafer_ncc_v2_pattern_byte(issue->slot, index);
    *guard_mismatches = wafer_ncc_v2_guard_mismatches(read0, bytes);
    break;
  }
  case WAFER_NCC_ENGINE_TDMA: {
    const volatile uint8_t *actual = wafer_ncc_probe_spm8(write);
    for (uint32_t index = 0; index < bytes; ++index)
      *result_mismatches +=
          actual[index] != wafer_ncc_v2_tdma_byte(
                               issue->slot,
                               issue->lane_spec->element_format, index);
    *guard_mismatches = wafer_ncc_v2_guard_mismatches(write, bytes);
    break;
  }
  default:
    return 1;
  }
  return 0;
}

static void wafer_ncc_v2_release(void *opaque,
                                 const WaferNccProbeIssue *issue) {
  WaferNccV2Context *context = (WaferNccV2Context *)opaque;
  wafer_ncc_probe_release(&context->instructions[issue->slot]);
}

static int wafer_ncc_v2_snapshot(void *opaque, uint32_t phase,
                                 volatile uint64_t *record) {
  WaferNccV2Context *context = (WaferNccV2Context *)opaque;
  switch (phase) {
  case WAFER_NCC_SNAPSHOT_BEFORE:
    if (context->constructor_captured) {
      record[WAFER_NCC_REC_CONSTRUCTOR_ADDRESS] =
          context->constructor_address;
      record[WAFER_NCC_REC_FLAGS] |=
          WAFER_NCC_RECORD_CONSTRUCTOR_CAPTURED;
    }
    record[WAFER_NCC_REC_PMU_ENABLE] =
        wafer_ncc_probe_read_pmu32(GR_PMU_EN);
    for (uint32_t worker = 0; worker < WAFER_NCC_PROTOCOL_WORKERS; ++worker)
      record[WAFER_NCC_REC_SERIAL_MODE + worker] =
          get_ncc_reg(worker, GR_CSR_SERIAL_MODE_ADDR);
    wafer_ncc_probe_read_controls(record, WAFER_NCC_REC_CONTROL_BEFORE);
    wafer_ncc_probe_read_snapshot(
        record, WAFER_NCC_REC_PMU64_BEFORE,
        WAFER_NCC_REC_INSTRUCTION_BEFORE, WAFER_NCC_REC_BLOCKING_BEFORE,
        WAFER_NCC_REC_STABLE_BEFORE);
    return 0;
  case WAFER_NCC_SNAPSHOT_BOUNDARY:
    wafer_ncc_probe_read_controls(record, WAFER_NCC_REC_CONTROL_BOUNDARY);
    if (context->completion_marker_address != 0) {
      record[WAFER_NCC_REC_COMPLETION_MARKER_EXPECTED] =
          context->completion_marker_expected;
      record[WAFER_NCC_REC_COMPLETION_MARKER_ADDRESS] =
          context->completion_marker_address;
      record[WAFER_NCC_REC_COMPLETION_MARKER_BOUNDARY] =
          *wafer_ncc_probe_spm8(context->completion_marker_address);
    }
    return 0;
  case WAFER_NCC_SNAPSHOT_FINAL:
    wafer_ncc_probe_read_controls(record, WAFER_NCC_REC_CONTROL_FINAL);
    if (context->completion_marker_address != 0)
      record[WAFER_NCC_REC_COMPLETION_MARKER_FINAL] =
          *wafer_ncc_probe_spm8(context->completion_marker_address);
    wafer_ncc_probe_read_snapshot(
        record, WAFER_NCC_REC_PMU64_AFTER, WAFER_NCC_REC_INSTRUCTION_AFTER,
        WAFER_NCC_REC_BLOCKING_AFTER, WAFER_NCC_REC_STABLE_AFTER);
    return 0;
  default:
    return 1;
  }
}

static int wafer_ncc_v2_wait_mask(uint32_t worker_mask) {
  for (uint32_t worker = 0; worker < WAFER_NCC_PROTOCOL_WORKERS; ++worker)
    if ((worker_mask & (UINT32_C(1) << worker)) != 0)
      wafer_ncc_probe_wait_worker(worker);
  return 0;
}

static int wafer_ncc_v2_serial_drain(void *opaque, uint32_t worker_mask) {
  (void)opaque;
  return wafer_ncc_v2_wait_mask(worker_mask);
}

static int wafer_ncc_v2_requested_wait(void *opaque, uint32_t wait_kind,
                                       uint32_t worker_mask) {
  (void)opaque;
  switch (wait_kind) {
  case WAFER_NCC_WAIT_BY_WORKER:
    return wafer_ncc_v2_wait_mask(worker_mask);
  case WAFER_NCC_WAIT_DEFAULT:
    (void)TsmWaitfinish();
    return 0;
  case WAFER_NCC_WAIT_LOCAL_FENCE:
    wafer_tx81_local_fence();
    return 0;
  default:
    return wait_kind == WAFER_NCC_WAIT_NONE ? 0 : 1;
  }
}

static int wafer_ncc_v2_safety_drain(void *opaque, uint32_t worker_mask) {
  (void)opaque;
  return wafer_ncc_v2_wait_mask(worker_mask);
}

static uint64_t wafer_ncc_v2_read_cycle(void *opaque) {
  (void)opaque;
  uint64_t cycle;
  __asm__ volatile("rdcycle %0" : "=r"(cycle));
  return cycle;
}

static uint64_t wafer_ncc_v2_read_worker_control(void *opaque,
                                                 uint32_t worker) {
  (void)opaque;
  return get_ncc_reg(worker, GR_CSR_CONTROL_ADDR);
}

static const WaferNccProbeEngineAdapter wafer_ncc_v2_adapters[] = {
    {WAFER_NCC_ENGINE_CT, 6, UINT32_C(3), wafer_ncc_v2_effect_compute,
     wafer_ncc_v2_seed, wafer_ncc_v2_prepare, wafer_ncc_v2_issue,
     wafer_ncc_v2_observe, wafer_ncc_v2_oracle, wafer_ncc_v2_release},
    {WAFER_NCC_ENGINE_NE, 6, UINT32_C(3), wafer_ncc_v2_effect_compute,
     wafer_ncc_v2_seed, wafer_ncc_v2_prepare, wafer_ncc_v2_issue,
     wafer_ncc_v2_observe, wafer_ncc_v2_oracle, wafer_ncc_v2_release},
    {WAFER_NCC_ENGINE_RDMA, 6, UINT32_C(3), wafer_ncc_v2_effect_copy,
     wafer_ncc_v2_seed, wafer_ncc_v2_prepare, wafer_ncc_v2_issue,
     wafer_ncc_v2_observe, wafer_ncc_v2_oracle, wafer_ncc_v2_release},
    {WAFER_NCC_ENGINE_WDMA, 6, UINT32_C(3), wafer_ncc_v2_effect_copy,
     wafer_ncc_v2_seed, wafer_ncc_v2_prepare, wafer_ncc_v2_issue,
     wafer_ncc_v2_observe, wafer_ncc_v2_oracle, wafer_ncc_v2_release},
    {WAFER_NCC_ENGINE_TDMA, 4, UINT32_C(3), wafer_ncc_v2_effect_fill,
     wafer_ncc_v2_seed, wafer_ncc_v2_prepare, wafer_ncc_v2_issue,
     wafer_ncc_v2_observe, wafer_ncc_v2_oracle, wafer_ncc_v2_release},
};

static const WaferNccProbeExecutionHooks wafer_ncc_v2_hooks = {
    wafer_ncc_v2_snapshot, wafer_ncc_v2_serial_drain,
    wafer_ncc_v2_requested_wait, wafer_ncc_v2_safety_drain,
    wafer_ncc_v2_read_cycle, wafer_ncc_v2_read_worker_control,
};

static uint32_t wafer_ncc_v2_result_bytes(const WaferNccProbeIssue *issue) {
  return issue->engine == WAFER_NCC_ENGINE_NE
             ? wafer_ncc_v2_ne_result_bytes(issue->lane_spec)
             : issue->lane_spec->transfer_bytes;
}

static void wafer_ncc_v2_copy_results(const WaferNccProbeRequest *request,
                                      WaferNccV2Context *context) {
  for (uint32_t lane = 0; lane < request->lane_count; ++lane) {
    for (uint32_t round = 0; round < request->rounds; ++round) {
      uint32_t ordinal = lane * request->rounds + round;
      if (request->issue_limit != 0 && ordinal >= request->issue_limit)
        continue;
      uint32_t slot = lane * WAFER_NCC_PROTOCOL_MAX_ROUNDS + round;
      WaferNccProbeIssue issue = {
          ordinal,
          lane,
          round,
          slot,
          request->lanes[lane].engine,
          request->lanes[lane].worker,
          0,
          &request->lanes[lane],
      };
      if (issue.engine == WAFER_NCC_ENGINE_WDMA)
        continue;
      if (context->double_slot_observation)
        continue;
      uint32_t bytes = wafer_ncc_v2_result_bytes(&issue);
      if (issue.lane_spec->layout_kind == WAFER_NCC_LAYOUT_DMA_STRIDED)
        wafer_tx81_wdma(
            wafer_ncc_v2_operand_address(context, &issue,
                                         WAFER_NCC_OPERAND_WRITE),
            wafer_ncc_v2_output_address(context, slot), bytes,
            issue.lane_spec->layout_inner_bytes,
            issue.lane_spec->layout_stride0_bytes,
            issue.lane_spec->layout_stride1_bytes,
            issue.lane_spec->layout_stride2_bytes,
            issue.lane_spec->layout_iteration0,
            issue.lane_spec->layout_iteration1,
            issue.lane_spec->layout_iteration2,
            issue.lane_spec->element_format);
      else
        wafer_tx81_wdma(
            wafer_ncc_v2_operand_address(context, &issue,
                                         WAFER_NCC_OPERAND_WRITE),
            wafer_ncc_v2_output_address(context, slot),
            bytes, bytes, 0, 0, 0, 1, 1, 1, Fmt_UINT8);
      /*
       * Copyback is a recovery-safe observation step after the final PMU
       * snapshot, not part of the measured issue window.  Drain every entry so
       * a maximal 3-lane x 4-round plan never creates an implicit WDMA backlog
       * beyond the documented queue bound.
       */
      wafer_tx81_local_fence();
    }
  }
}

__attribute__((visibility("hidden"))) void
wafer_tx81_ncc_execution_probe(uint64_t request_ddr, uint64_t payload_ddr,
                               uint64_t output_ddr) {
  wafer_ncc_probe_invalidate(request_ddr, WAFER_NCC_V2_RESOURCE_BYTES);
  wafer_ncc_probe_invalidate(payload_ddr, WAFER_NCC_V2_RESOURCE_BYTES);
  const volatile uint64_t *request_words =
      (const volatile uint64_t *)(uintptr_t)request_ddr;
  volatile uint64_t *record =
      (volatile uint64_t *)(uintptr_t)output_ddr;
  WaferNccProbeRequest request;
  uint32_t status = wafer_ncc_probe_decode_request(request_words, &request);
  if (status != WAFER_NCC_STATUS_OK) {
    for (uint32_t index = 0; index < WAFER_NCC_PROTOCOL_RECORD_WORDS; ++index)
      record[index] = 0;
    record[WAFER_NCC_REC_MAGIC] = WAFER_NCC_PROTOCOL_RECORD_MAGIC;
    record[WAFER_NCC_REC_SCHEMA_AND_WORDS] =
        ((uint64_t)WAFER_NCC_PROTOCOL_SCHEMA << 32) |
        WAFER_NCC_PROTOCOL_RECORD_WORDS;
    record[WAFER_NCC_REC_STATUS] = status;
    wafer_ncc_probe_publish(record);
    return;
  }

  WaferNccV2Context context = {0};
  context.payload_ddr = payload_ddr;
  context.output_ddr = output_ddr;
  status = wafer_ncc_probe_execute_plan(
      &request, wafer_ncc_v2_adapters,
      sizeof(wafer_ncc_v2_adapters) / sizeof(wafer_ncc_v2_adapters[0]),
      &wafer_ncc_v2_hooks, &context, record);
  record[WAFER_NCC_REC_OUTPUT_SLOT_BASE] = WAFER_NCC_V2_OUTPUT_SLOT_BASE;
  record[WAFER_NCC_REC_OUTPUT_SLOT_STRIDE] = WAFER_NCC_V2_DDR_SLOT_STRIDE;
  record[WAFER_NCC_REC_OUTPUT_GUARD_BYTES] = WAFER_NCC_V2_GUARD_BYTES;
  record[WAFER_NCC_REC_RESOURCE_BYTES] = WAFER_NCC_V2_RESOURCE_BYTES;
  record[WAFER_NCC_REC_RECORD_GUARD] = WAFER_NCC_V2_RECORD_GUARD;
  if (status == WAFER_NCC_STATUS_OK &&
      request.command == WAFER_NCC_COMMAND_EXECUTE)
    wafer_ncc_v2_copy_results(&request, &context);
  wafer_ncc_probe_publish(record);
}
