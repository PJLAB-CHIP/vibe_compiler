#include "instr_adapter.h"
#include "instr_adapter_plat.h"
#include "instr_def.h"
#include "wafer_memory_descriptor_calibration_probe_protocol.h"
#include "wafer_spm_cross_tile_conflict_probe_protocol.h"
#include "wafer_tx81_crt.h"

#include <stdint.h>

#define WAFER_MDC_PMU_BASE UINT64_C(0x590000)
#define WAFER_MDC_STABLE_RETRIES 8U
#define WAFER_MDC_ENGINES 5U
#define WAFER_MDC_SPM_GUARD_BYTES 64U
#define WAFER_MDC_OUTPUT_LAYOUT_ALIGNMENT 256U
#define WAFER_MDC_PAYLOAD_SPM_SEED_OFFSET 135168U
#define WAFER_MDC_PAYLOAD_SPM_SCRATCH0_OFFSET 204800U
#define WAFER_MDC_PAYLOAD_SPM_SCRATCH1_OFFSET 212992U
#define WAFER_MDC_WORKERS 3U
#define WAFER_MDC_WORKER_PMU_STRIDE UINT32_C(0x30)
#define WAFER_MDC_PERF_MAX_ISSUES                                  \
  (2U * WAFER_MDC_PERF_MAX_ROUNDS)
#define WAFER_MDC_DRAIN_SPINS UINT32_C(1000000)

extern void hrt_barrier(void);

typedef struct WaferMDCRequest {
  uint32_t case_id;
  uint32_t kind;
  uint32_t engine_a;
  uint32_t engine_b;
  uint32_t schedule;
  uint32_t effect;
  uint32_t relation;
  uint32_t oracle;
  uint32_t format;
  uint32_t src_ddr_offset;
  uint32_t dst_ddr_offset;
  uint64_t spm_a;
  uint64_t spm_b;
  uint32_t inner_bytes;
  uint32_t stride0;
  uint32_t stride1;
  uint32_t stride2;
  uint32_t iteration0;
  uint32_t iteration1;
  uint32_t iteration2;
  uint32_t compact_bytes;
  uint32_t envelope_bytes;
  uint32_t output_bytes;
  uint32_t sample;
  uint32_t expected_instructions[WAFER_MDC_ENGINES];
  uint32_t worker_a;
  uint32_t worker_b;
  uint32_t rounds;
  uint32_t buffer_count;
  uint32_t issue_order;
} WaferMDCRequest;

typedef struct WaferMDCPMU {
  uint32_t instructions[WAFER_MDC_ENGINES];
  uint32_t blocking[WAFER_MDC_ENGINES];
  uint32_t worker_instructions[WAFER_MDC_WORKERS][WAFER_MDC_ENGINES];
  uint32_t worker_blocking[WAFER_MDC_WORKERS][WAFER_MDC_ENGINES];
  uint64_t execution[WAFER_MDC_ENGINES];
  uint64_t full_execution;
  uint64_t stable_mask;
} WaferMDCPMU;

typedef union WaferMDCPacket {
  TsmArithInstr ct;
  TsmNeInstr ne;
  TsmRdmaInstr rdma;
  TsmWdmaInstr wdma;
  TsmDataMoveInstr tdma;
} WaferMDCPacket;

typedef struct WaferMDCPreparedInstruction {
  uint32_t engine;
  WaferMDCPacket packet;
} WaferMDCPreparedInstruction;

typedef struct WaferMDCEngineAddresses {
  uint64_t read0;
  uint64_t read1;
  uint64_t write;
  uint32_t transfer_bytes;
  uint32_t slot_span;
} WaferMDCEngineAddresses;

static const uint32_t wafer_mdc_instruction_offsets[WAFER_MDC_ENGINES] = {
    GR_PMU_CT_INST_NUMS, GR_PMU_NE_INST_NUMS, GR_PMU_RDMA_INST_NUMS,
    GR_PMU_WDMA_INST_NUMS, GR_PMU_TDMA_INST_NUMS,
};
static const uint32_t wafer_mdc_blocking_offsets[WAFER_MDC_ENGINES] = {
    GR_PMU_CT_BLOCKING_TIME, GR_PMU_NE_BLOCKING_TIME,
    GR_PMU_RDMA_BLOCKING_TIME, GR_PMU_WDMA_BLOCKING_TIME,
    GR_PMU_TDMA_BLOCKING_TIME,
};
static const uint32_t wafer_mdc_execution_offsets[WAFER_MDC_ENGINES] = {
    GR_PMU_CT_EXE_TIME, GR_PMU_NE_EXE_TIME, GR_PMU_RDMA_EXE_TIME,
    GR_PMU_WDMA_EXE_TIME, GR_PMU_TDMA_EXE_TIME,
};

static void wafer_mdc_cache_range(uint64_t begin, uint32_t bytes,
                                  uint32_t invalidate_only) {
  enum {
    WAFER_MDC_SUPERVISOR_MODE = 1,
    WAFER_MDC_MACHINE_MODE = 3,
    WAFER_MDC_CACHE_LINE_BYTES = 64,
  };
  uintptr_t mode;
  __asm__ volatile("fence" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
  __asm__ volatile("csrr %0, mxstatus" : "=r"(mode));
  mode = (mode >> 30) & 3U;
  for (uintptr_t address = (uintptr_t)begin;
       address < (uintptr_t)begin + bytes;
       address += WAFER_MDC_CACHE_LINE_BYTES) {
    if (mode == WAFER_MDC_MACHINE_MODE) {
      if (invalidate_only != 0U)
        __asm__ volatile("dcache.ipa %0" : : "r"(address) : "memory");
      else
        __asm__ volatile("dcache.cipa %0" : : "r"(address) : "memory");
    } else if (mode == WAFER_MDC_SUPERVISOR_MODE) {
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

static uint32_t wafer_mdc_read32(uint32_t offset) {
  return *(const volatile uint32_t *)(uintptr_t)(WAFER_MDC_PMU_BASE + offset);
}

static uint64_t wafer_mdc_cycle(void) {
  uint64_t value;
  __asm__ volatile("rdcycle %0" : "=r"(value));
  return value;
}

static uint64_t wafer_mdc_read64(uint32_t low_offset, uint64_t stability_bit,
                                 uint64_t *stable_mask) {
  uint32_t low = 0;
  uint32_t high_after = 0;
  for (uint32_t retry = 0; retry < WAFER_MDC_STABLE_RETRIES; ++retry) {
    uint32_t high_before = wafer_mdc_read32(low_offset + 4U);
    low = wafer_mdc_read32(low_offset);
    high_after = wafer_mdc_read32(low_offset + 4U);
    if (high_before == high_after) {
      *stable_mask |= stability_bit;
      break;
    }
  }
  return ((uint64_t)high_after << 32) | low;
}

static WaferMDCPMU wafer_mdc_read_pmu(void) {
  WaferMDCPMU result = {0};
  result.full_execution =
      wafer_mdc_read64(GR_PMU_FU_EXE_TIME, UINT64_C(1),
                       &result.stable_mask);
  for (uint32_t engine = 0; engine < WAFER_MDC_ENGINES; ++engine) {
    for (uint32_t worker = 0; worker < WAFER_MDC_WORKERS; ++worker) {
      uint32_t worker_offset = worker * WAFER_MDC_WORKER_PMU_STRIDE;
      result.worker_instructions[worker][engine] = wafer_mdc_read32(
          wafer_mdc_instruction_offsets[engine] + worker_offset);
      result.worker_blocking[worker][engine] = wafer_mdc_read32(
          wafer_mdc_blocking_offsets[engine] + worker_offset);
      result.instructions[engine] +=
          result.worker_instructions[worker][engine];
      result.blocking[engine] += result.worker_blocking[worker][engine];
    }
    result.execution[engine] = wafer_mdc_read64(
        wafer_mdc_execution_offsets[engine], UINT64_C(1) << (engine + 1U),
        &result.stable_mask);
  }
  return result;
}

static uint32_t wafer_mdc_checked_descriptor(
    const WaferMDCRequest *request) {
  if (request->inner_bytes == 0U || request->iteration0 == 0U ||
      request->iteration1 == 0U || request->iteration2 == 0U)
    return 0U;
  uint64_t compact = (uint64_t)request->inner_bytes * request->iteration0 *
                     request->iteration1 * request->iteration2;
  uint64_t envelope =
      request->inner_bytes +
      (uint64_t)request->stride0 * (request->iteration0 - 1U) +
      (uint64_t)request->stride1 * (request->iteration1 - 1U) +
      (uint64_t)request->stride2 * (request->iteration2 - 1U);
  return compact == request->compact_bytes &&
         envelope == request->envelope_bytes &&
         compact <= UINT32_MAX && envelope <= UINT32_MAX;
}

static uint32_t wafer_mdc_spm_range(uint64_t begin, uint64_t bytes) {
  return begin >=
             WAFER_MDC_SPM_ALLOCATABLE_BEGIN + WAFER_MDC_SPM_GUARD_BYTES &&
         bytes != 0U && begin <= UINT64_MAX - bytes &&
         begin + bytes + WAFER_MDC_SPM_GUARD_BYTES <=
             WAFER_MDC_SPM_ALLOCATABLE_END;
}

static uint64_t wafer_mdc_max_u64(uint64_t left, uint64_t right) {
  return left > right ? left : right;
}

static uint64_t wafer_mdc_align_output_offset(uint64_t offset) {
  return (offset + WAFER_MDC_OUTPUT_LAYOUT_ALIGNMENT - 1U) /
         WAFER_MDC_OUTPUT_LAYOUT_ALIGNMENT *
         WAFER_MDC_OUTPUT_LAYOUT_ALIGNMENT;
}

static uint64_t
wafer_mdc_output_result_end(const WaferMDCRequest *request) {
  uint64_t end = (uint64_t)WAFER_MDC_RECORD_WORDS * sizeof(uint64_t);
  if (request->kind == WAFER_MDC_KIND_PARALLEL_PAIR)
    return WAFER_MDC_PERF_OUTPUT_GUARD_BASE +
           WAFER_MDC_PERF_OUTPUT_GUARD_LANE_STRIDE +
           2U * WAFER_MDC_SPM_GUARD_BYTES;
  if (request->kind == WAFER_MDC_KIND_DMA)
    return wafer_mdc_max_u64(
        end, (uint64_t)WAFER_MDC_OUTPUT_DATA_OFFSET +
                 request->dst_ddr_offset + request->envelope_bytes);
  if (request->kind == WAFER_MDC_KIND_ENGINE_ACCESS ||
      request->kind == WAFER_MDC_KIND_ENGINE_PAIR) {
    uint32_t lanes =
        request->kind == WAFER_MDC_KIND_ENGINE_ACCESS ? 1U : 2U;
    uint32_t engines[2] = {request->engine_a, request->engine_b};
    uint32_t transfer = request->compact_bytes;
    for (uint32_t lane = 0; lane < lanes; ++lane) {
      uint64_t output_end =
          (uint64_t)WAFER_MDC_OUTPUT_DATA_OFFSET +
          (request->kind == WAFER_MDC_KIND_ENGINE_PAIR
               ? lane * request->compact_bytes
               : 0U) +
          transfer;
      end = wafer_mdc_max_u64(end, output_end);
      if (engines[lane] == WAFER_MDC_ENGINE_WDMA) {
        uint64_t sink =
            lane == 0U ? WAFER_MDC_OUTPUT_SINK0_OFFSET
                       : WAFER_MDC_OUTPUT_SINK1_OFFSET;
        end = wafer_mdc_max_u64(end, sink + transfer);
      }
    }
    return end;
  }
  end = wafer_mdc_max_u64(
      end, (uint64_t)WAFER_MDC_OUTPUT_DATA_OFFSET + request->output_bytes);
  if (request->engine_a == WAFER_MDC_ENGINE_WDMA)
    end = wafer_mdc_max_u64(
        end, (uint64_t)WAFER_MDC_OUTPUT_SINK0_OFFSET + 4096U);
  if (request->engine_b == WAFER_MDC_ENGINE_WDMA)
    end = wafer_mdc_max_u64(
        end, (uint64_t)WAFER_MDC_OUTPUT_SINK1_OFFSET + 4096U);
  return end;
}

static uint32_t
wafer_mdc_spm_dump_lanes(const WaferMDCRequest *request) {
  if (request->kind == WAFER_MDC_KIND_PARALLEL_PAIR)
    return 0U;
  return request->kind == WAFER_MDC_KIND_ENGINE_PAIR ? 2U : 1U;
}

static uint32_t
wafer_mdc_spm_dump_slot_span(const WaferMDCRequest *request) {
  if (request->kind == WAFER_MDC_KIND_DMA)
    return request->compact_bytes;
  if (request->kind == WAFER_MDC_KIND_ENGINE_ACCESS)
    return request->compact_bytes > 512U ? 0x6000U : 0x1000U;
  if (request->kind == WAFER_MDC_KIND_ENGINE_PAIR)
    return WAFER_MDC_SPM_PAIR_SLOT_BYTES;
  return request->output_bytes;
}

static uint64_t
wafer_mdc_first_spm_dump_offset(const WaferMDCRequest *request) {
  return wafer_mdc_align_output_offset(wafer_mdc_output_result_end(request));
}

static uint32_t
wafer_mdc_output_layout_fits(const WaferMDCRequest *request) {
  if (request->kind == WAFER_MDC_KIND_PARALLEL_PAIR)
    return wafer_mdc_output_result_end(request) <=
           WAFER_MDC_RESOURCE_BYTES;
  uint64_t cursor = wafer_mdc_first_spm_dump_offset(request);
  uint64_t dump_bytes =
      (uint64_t)wafer_mdc_spm_dump_slot_span(request) +
      2U * WAFER_MDC_SPM_GUARD_BYTES;
  for (uint32_t lane = 0; lane < wafer_mdc_spm_dump_lanes(request);
       ++lane) {
    if (cursor > WAFER_MDC_RESOURCE_BYTES ||
        dump_bytes > WAFER_MDC_RESOURCE_BYTES - cursor)
      return 0U;
    cursor = wafer_mdc_align_output_offset(cursor + dump_bytes);
  }
  return 1U;
}

static void wafer_mdc_expected_counts(const WaferMDCRequest *request,
                                      uint32_t counts[WAFER_MDC_ENGINES]) {
  for (uint32_t engine = 0; engine < WAFER_MDC_ENGINES; ++engine)
    counts[engine] = 0U;
  if (request->kind == WAFER_MDC_KIND_DMA) {
    counts[WAFER_MDC_ENGINE_RDMA] = 1U;
    counts[WAFER_MDC_ENGINE_WDMA] = 1U;
  } else if (request->kind == WAFER_MDC_KIND_ENGINE_ACCESS) {
    if (request->engine_a < WAFER_MDC_ENGINES)
      counts[request->engine_a] = 1U;
  } else if (request->kind == WAFER_MDC_KIND_ENGINE_PAIR ||
             request->kind == WAFER_MDC_KIND_PARALLEL_PAIR) {
    if (request->engine_a < WAFER_MDC_ENGINES)
      counts[request->engine_a] +=
          request->kind == WAFER_MDC_KIND_PARALLEL_PAIR
              ? request->rounds
              : 1U;
    if (request->engine_b < WAFER_MDC_ENGINES)
      counts[request->engine_b] +=
          request->kind == WAFER_MDC_KIND_PARALLEL_PAIR
              ? request->rounds
              : 1U;
  } else if (request->kind == WAFER_MDC_KIND_ADDRESS_RELATION) {
    if (request->engine_a < WAFER_MDC_ENGINES)
      ++counts[request->engine_a];
    if (request->engine_b < WAFER_MDC_ENGINES)
      ++counts[request->engine_b];
  }
}

static void wafer_mdc_parallel_active_range(
    const WaferMDCRequest *request, uint32_t lane, uint64_t *begin,
    uint64_t *end) {
  uint32_t engine =
      lane == 0U ? request->engine_a : request->engine_b;
  uint64_t base = lane == 0U ? request->spm_a : request->spm_b;
  uint64_t transfer = request->compact_bytes;
  if (engine == WAFER_MDC_ENGINE_CT) {
    *begin = base + WAFER_MDC_PERF_READ0_OFFSET;
    *end = wafer_mdc_max_u64(
        base + WAFER_MDC_PERF_READ1_OFFSET +
            request->buffer_count * transfer,
        base + WAFER_MDC_PERF_WRITE_OFFSET +
            request->rounds * transfer);
  } else if (engine == WAFER_MDC_ENGINE_NE) {
    *begin = base + WAFER_MDC_PERF_READ0_OFFSET;
    *end = wafer_mdc_max_u64(
        base + WAFER_MDC_PERF_READ1_OFFSET + 32768U,
        base + WAFER_MDC_PERF_WRITE_OFFSET +
            request->rounds * transfer);
  } else if (engine == WAFER_MDC_ENGINE_WDMA) {
    *begin = base + WAFER_MDC_PERF_READ0_OFFSET;
    *end = base + WAFER_MDC_PERF_READ0_OFFSET +
           request->buffer_count * transfer;
  } else {
    *begin = base + WAFER_MDC_PERF_WRITE_OFFSET;
    *end = base + WAFER_MDC_PERF_WRITE_OFFSET +
           request->rounds * transfer;
  }
}

static uint32_t wafer_mdc_validate_kind(const WaferMDCRequest *request) {
  if (request->kind > WAFER_MDC_KIND_PARALLEL_PAIR ||
      request->schedule > WAFER_MDC_SCHEDULE_WINDOW ||
      request->oracle > WAFER_MDC_ORACLE_OBSERVATION ||
      request->relation > WAFER_MDC_RELATION_STRIDED ||
      request->effect > WAFER_MDC_EFFECT_RAR)
    return 0U;
  if (!wafer_mdc_checked_descriptor(request))
    return 0U;
  if (request->kind == WAFER_MDC_KIND_DMA) {
    if (request->engine_a != WAFER_MDC_ENGINE_RDMA ||
        request->engine_b != WAFER_MDC_ENGINE_WDMA ||
        request->oracle != WAFER_MDC_ORACLE_EXACT ||
        request->compact_bytes > 65536U ||
        (uint64_t)WAFER_MDC_PAYLOAD_DATA_OFFSET +
                request->src_ddr_offset + request->envelope_bytes >
            WAFER_MDC_RESOURCE_BYTES ||
        (uint64_t)WAFER_MDC_OUTPUT_DATA_OFFSET +
                request->dst_ddr_offset + request->envelope_bytes >
            WAFER_MDC_RESOURCE_BYTES ||
        !wafer_mdc_spm_range(request->spm_a, request->compact_bytes))
      return 0U;
  } else if (request->kind == WAFER_MDC_KIND_ENGINE_ACCESS) {
    if (request->engine_a >= WAFER_MDC_ENGINES ||
        request->engine_b != WAFER_MDC_ENGINE_NONE ||
        request->schedule != WAFER_MDC_SCHEDULE_SERIAL ||
        request->oracle != WAFER_MDC_ORACLE_EXACT ||
        request->spm_a % 256U != 0U)
      return 0U;
  } else if (request->kind == WAFER_MDC_KIND_ENGINE_PAIR) {
    if (request->spm_b < request->spm_a)
      return 0U;
    uint64_t relative_offset = request->spm_b - request->spm_a;
    uint32_t phase = (uint32_t)(request->spm_a % 256U);
    uint32_t general_offset =
        phase == 0U &&
        (relative_offset == 4352U || relative_offset == 8192U ||
         relative_offset == 65536U);
    uint32_t bank_period_offset =
        request->engine_a == WAFER_MDC_ENGINE_CT &&
        request->engine_b == WAFER_MDC_ENGINE_RDMA && phase == 0U &&
        relative_offset >= 4096U && relative_offset <= 6144U &&
        (relative_offset - 4096U) % 256U == 0U;
    uint32_t alignment_phase =
        request->engine_a == WAFER_MDC_ENGINE_CT &&
        request->engine_b == WAFER_MDC_ENGINE_RDMA &&
        relative_offset == 8192U &&
        (phase == 0U || phase == 64U || phase == 128U || phase == 192U);
    uint32_t legacy_pair =
        request->compact_bytes == 256U && request->issue_order == 0U &&
        (general_offset || bank_period_offset || alignment_phase);
    uint32_t conflict_equivalence =
        alignment_phase &&
        (request->compact_bytes == 256U ||
         request->compact_bytes == 512U) &&
        request->issue_order <= 1U;
    if (request->engine_a >= WAFER_MDC_ENGINES ||
        request->engine_b >= WAFER_MDC_ENGINES ||
        request->engine_a >= request->engine_b ||
        request->format != Fmt_UINT8 ||
        request->oracle != WAFER_MDC_ORACLE_EXACT ||
        request->output_bytes != 2U * request->compact_bytes ||
        request->spm_b % 256U != phase ||
        !(legacy_pair || conflict_equivalence) ||
        relative_offset <
            WAFER_MDC_SPM_PAIR_SLOT_BYTES +
                2U * WAFER_MDC_SPM_GUARD_BYTES ||
        !wafer_mdc_spm_range(request->spm_a,
                             WAFER_MDC_SPM_PAIR_SLOT_BYTES) ||
        !wafer_mdc_spm_range(request->spm_b,
                             WAFER_MDC_SPM_PAIR_SLOT_BYTES))
      return 0U;
  } else if (request->kind == WAFER_MDC_KIND_PARALLEL_PAIR) {
    uint32_t transfer = request->compact_bytes;
    uint32_t dependency =
        request->effect == WAFER_MDC_EFFECT_RAW ||
        request->effect == WAFER_MDC_EFFECT_WAR;
    uint32_t expected_transfer =
        request->engine_a == WAFER_MDC_ENGINE_CT ||
                request->engine_b == WAFER_MDC_ENGINE_CT ||
                request->engine_a == WAFER_MDC_ENGINE_NE ||
                request->engine_b == WAFER_MDC_ENGINE_NE
            ? 16384U
            : 65536U;
    if (request->engine_a >= WAFER_MDC_ENGINES ||
        request->engine_b >= WAFER_MDC_ENGINES ||
        request->engine_a == request->engine_b ||
        request->oracle != WAFER_MDC_ORACLE_EXACT ||
        request->worker_a >= WAFER_MDC_WORKERS ||
        request->worker_b >= WAFER_MDC_WORKERS ||
        request->rounds == 0U ||
        request->rounds > WAFER_MDC_PERF_MAX_ROUNDS ||
        request->buffer_count == 0U ||
        request->buffer_count > WAFER_MDC_PERF_MAX_BUFFERS ||
        request->issue_order > 1U ||
        request->output_bytes != 2U * request->rounds * transfer)
      return 0U;
    if (dependency) {
      uint64_t expected_b =
          request->spm_a + WAFER_MDC_PERF_WRITE_OFFSET -
          WAFER_MDC_PERF_READ0_OFFSET;
      if (request->engine_a != WAFER_MDC_ENGINE_RDMA ||
          request->engine_b != WAFER_MDC_ENGINE_WDMA ||
          transfer != 4096U || request->rounds != 1U ||
          request->buffer_count != 1U ||
          request->relation == WAFER_MDC_RELATION_ADJACENT ||
          request->relation == WAFER_MDC_RELATION_STRIDED ||
          request->effect !=
              (request->issue_order == 0U ? WAFER_MDC_EFFECT_RAW
                                          : WAFER_MDC_EFFECT_WAR))
        return 0U;
      if (request->relation == WAFER_MDC_RELATION_DISJOINT)
        expected_b = WAFER_MDC_PERF_DISJOINT_B;
      else if (request->relation == WAFER_MDC_RELATION_PARTIAL)
        expected_b += transfer / 2U;
      if (request->spm_b != expected_b)
        return 0U;
    } else if (request->effect != WAFER_MDC_EFFECT_NONE ||
               request->relation != WAFER_MDC_RELATION_DISJOINT ||
               request->spm_a != WAFER_MDC_PERF_DISJOINT_A ||
               request->spm_b != WAFER_MDC_PERF_DISJOINT_B ||
               transfer != expected_transfer ||
               request->rounds != WAFER_MDC_PERF_MAX_ROUNDS ||
               request->buffer_count != WAFER_MDC_PERF_MAX_BUFFERS ||
               request->issue_order != 0U) {
      return 0U;
    }
    uint64_t active_begin[2];
    uint64_t active_end[2];
    for (uint32_t lane = 0; lane < 2U; ++lane) {
      wafer_mdc_parallel_active_range(
          request, lane, &active_begin[lane], &active_end[lane]);
      if (!wafer_mdc_spm_range(
              active_begin[lane],
              active_end[lane] - active_begin[lane]))
        return 0U;
    }
    if (request->relation == WAFER_MDC_RELATION_DISJOINT &&
        active_end[0] + WAFER_MDC_SPM_GUARD_BYTES >
            active_begin[1] - WAFER_MDC_SPM_GUARD_BYTES)
      return 0U;
  } else {
    if (request->engine_a >= WAFER_MDC_ENGINES ||
        request->engine_b >= WAFER_MDC_ENGINES ||
        request->effect < WAFER_MDC_EFFECT_RAW ||
        request->oracle != WAFER_MDC_ORACLE_OBSERVATION ||
        request->schedule != WAFER_MDC_SCHEDULE_WINDOW ||
        request->output_bytes != 16384U ||
        request->spm_a % 256U != 0U)
      return 0U;
    uint64_t expected_b = request->spm_a;
    if (request->relation == WAFER_MDC_RELATION_PARTIAL)
      expected_b += 1024U;
    else if (request->relation == WAFER_MDC_RELATION_ADJACENT)
      expected_b += 2048U;
    else if (request->relation == WAFER_MDC_RELATION_DISJOINT)
      expected_b += 8192U;
    else if (request->relation == WAFER_MDC_RELATION_STRIDED)
      expected_b += 128U;
    if (request->spm_b != expected_b ||
        !wafer_mdc_spm_range(request->spm_a, request->output_bytes))
      return 0U;
  }
  if (!wafer_mdc_output_layout_fits(request))
    return 0U;
  uint32_t counts[WAFER_MDC_ENGINES];
  wafer_mdc_expected_counts(request, counts);
  for (uint32_t engine = 0; engine < WAFER_MDC_ENGINES; ++engine)
    if (counts[engine] != request->expected_instructions[engine])
      return 0U;
  return 1U;
}

static uint32_t wafer_mdc_decode(const volatile uint64_t *wire,
                                 WaferMDCRequest *request) {
  if (wire[WAFER_MDC_REQ_MAGIC] != WAFER_MDC_REQUEST_MAGIC ||
      wire[WAFER_MDC_REQ_SCHEMA_AND_WORDS] !=
          (((uint64_t)WAFER_MDC_SCHEMA << 32) |
           WAFER_MDC_REQUEST_WORDS) ||
      wire[WAFER_MDC_REQ_RESOURCE_BYTES] != WAFER_MDC_RESOURCE_BYTES ||
      wire[WAFER_MDC_REQ_PAYLOAD_DATA_OFFSET] !=
          WAFER_MDC_PAYLOAD_DATA_OFFSET ||
      wire[WAFER_MDC_REQ_OUTPUT_DATA_OFFSET] !=
          WAFER_MDC_OUTPUT_DATA_OFFSET ||
      wire[WAFER_MDC_REQ_GUARD] != WAFER_MDC_REQUEST_GUARD)
    return WAFER_MDC_STATUS_BAD_REQUEST;
#define WAFER_MDC_COPY32(field, word)                                         \
  request->field = (uint32_t)wire[word]
  WAFER_MDC_COPY32(case_id, WAFER_MDC_REQ_CASE);
  WAFER_MDC_COPY32(kind, WAFER_MDC_REQ_KIND);
  WAFER_MDC_COPY32(engine_a, WAFER_MDC_REQ_ENGINE_A);
  WAFER_MDC_COPY32(engine_b, WAFER_MDC_REQ_ENGINE_B);
  WAFER_MDC_COPY32(schedule, WAFER_MDC_REQ_SCHEDULE);
  WAFER_MDC_COPY32(effect, WAFER_MDC_REQ_EFFECT);
  WAFER_MDC_COPY32(relation, WAFER_MDC_REQ_RELATION);
  WAFER_MDC_COPY32(oracle, WAFER_MDC_REQ_ORACLE);
  WAFER_MDC_COPY32(format, WAFER_MDC_REQ_FORMAT);
  WAFER_MDC_COPY32(src_ddr_offset, WAFER_MDC_REQ_SRC_DDR_OFFSET);
  WAFER_MDC_COPY32(dst_ddr_offset, WAFER_MDC_REQ_DST_DDR_OFFSET);
  request->spm_a = wire[WAFER_MDC_REQ_SPM_A];
  request->spm_b = wire[WAFER_MDC_REQ_SPM_B];
  WAFER_MDC_COPY32(inner_bytes, WAFER_MDC_REQ_INNER_BYTES);
  WAFER_MDC_COPY32(stride0, WAFER_MDC_REQ_STRIDE0);
  WAFER_MDC_COPY32(stride1, WAFER_MDC_REQ_STRIDE1);
  WAFER_MDC_COPY32(stride2, WAFER_MDC_REQ_STRIDE2);
  WAFER_MDC_COPY32(iteration0, WAFER_MDC_REQ_ITERATION0);
  WAFER_MDC_COPY32(iteration1, WAFER_MDC_REQ_ITERATION1);
  WAFER_MDC_COPY32(iteration2, WAFER_MDC_REQ_ITERATION2);
  WAFER_MDC_COPY32(compact_bytes, WAFER_MDC_REQ_COMPACT_BYTES);
  WAFER_MDC_COPY32(envelope_bytes, WAFER_MDC_REQ_ENVELOPE_BYTES);
  WAFER_MDC_COPY32(output_bytes, WAFER_MDC_REQ_OUTPUT_BYTES);
  WAFER_MDC_COPY32(sample, WAFER_MDC_REQ_SAMPLE);
  for (uint32_t engine = 0; engine < WAFER_MDC_ENGINES; ++engine)
    request->expected_instructions[engine] =
        (uint32_t)wire[WAFER_MDC_REQ_CT_INSTRUCTIONS + engine];
  WAFER_MDC_COPY32(worker_a, WAFER_MDC_REQ_WORKER_A);
  WAFER_MDC_COPY32(worker_b, WAFER_MDC_REQ_WORKER_B);
  WAFER_MDC_COPY32(rounds, WAFER_MDC_REQ_ROUNDS);
  WAFER_MDC_COPY32(buffer_count, WAFER_MDC_REQ_BUFFER_COUNT);
  WAFER_MDC_COPY32(issue_order, WAFER_MDC_REQ_ISSUE_ORDER);
#undef WAFER_MDC_COPY32
  return wafer_mdc_validate_kind(request) ? WAFER_MDC_STATUS_OK
                                          : WAFER_MDC_STATUS_BAD_REQUEST;
}

static void wafer_mdc_seed_slot(uint64_t payload_ddr, uint64_t begin,
                                uint32_t bytes) {
  uint32_t span = bytes + 2U * WAFER_MDC_SPM_GUARD_BYTES;
  wafer_tx81_rdma(
      payload_ddr + WAFER_MDC_PAYLOAD_SPM_SEED_OFFSET,
      begin - WAFER_MDC_SPM_GUARD_BYTES, span, span, 0U, 0U, 0U, 1U, 1U, 1U,
      Fmt_UINT8);
}

static uint64_t wafer_mdc_dump_slot(uint64_t output_ddr,
                                    uint64_t output_offset, uint64_t begin,
                                    uint32_t bytes) {
  uint32_t span = bytes + 2U * WAFER_MDC_SPM_GUARD_BYTES;
  wafer_tx81_wdma(begin - WAFER_MDC_SPM_GUARD_BYTES,
                  output_ddr + output_offset, span, span, 0U, 0U, 0U, 1U, 1U,
                  1U, Fmt_UINT8);
  return wafer_mdc_align_output_offset(output_offset + span);
}

static void wafer_mdc_rdma_descriptor(uint64_t source, uint64_t destination,
                                      const WaferMDCRequest *request) {
  wafer_tx81_rdma(
      source, destination, request->compact_bytes, request->inner_bytes,
      request->stride0, request->stride1, request->stride2,
      request->iteration0, request->iteration1, request->iteration2,
      request->format);
}

static void wafer_mdc_wdma_descriptor(uint64_t source, uint64_t destination,
                                      const WaferMDCRequest *request) {
  wafer_tx81_wdma(
      source, destination, request->compact_bytes, request->inner_bytes,
      request->stride0, request->stride1, request->stride2,
      request->iteration0, request->iteration1, request->iteration2,
      request->format);
}

static WaferMDCEngineAddresses
wafer_mdc_engine_addresses(const WaferMDCRequest *request, uint64_t base) {
  WaferMDCEngineAddresses result;
  result.transfer_bytes =
      request->compact_bytes;
  if (result.transfer_bytes > 512U) {
    result.read0 = base + 0x100U;
    result.read1 = base + 0x2100U;
    result.write = base + 0x4100U;
    result.slot_span = 0x6000U;
  } else {
    result.read0 = base + 0x100U;
    result.read1 = base + 0x400U;
    result.write = base + 0x800U;
    result.slot_span =
        request->kind == WAFER_MDC_KIND_ENGINE_PAIR
            ? WAFER_MDC_SPM_PAIR_SLOT_BYTES
            : 0x1000U;
  }
  return result;
}

static uint64_t wafer_mdc_lane_payload(uint64_t payload_ddr,
                                       uint32_t lane) {
  return payload_ddr + WAFER_MDC_PAYLOAD_DATA_OFFSET +
         (uint64_t)lane * 16384U;
}

static uint64_t wafer_mdc_lane_sink(uint64_t output_ddr, uint32_t lane) {
  return output_ddr +
         (lane == 0U ? WAFER_MDC_OUTPUT_SINK0_OFFSET
                     : WAFER_MDC_OUTPUT_SINK1_OFFSET);
}

static uint32_t wafer_mdc_prepare_engine(
    const WaferMDCRequest *request, uint32_t engine, uint32_t lane,
    uint64_t base, uint64_t payload_ddr) {
  WaferMDCEngineAddresses addresses =
      wafer_mdc_engine_addresses(request, base);
  if (!wafer_mdc_spm_range(base, addresses.slot_span))
    return 0U;
  wafer_mdc_seed_slot(payload_ddr, base, addresses.slot_span);
  uint64_t payload = wafer_mdc_lane_payload(payload_ddr, lane);
  if (engine == WAFER_MDC_ENGINE_CT) {
    wafer_tx81_rdma(payload, addresses.read0, addresses.transfer_bytes,
                    addresses.transfer_bytes, 0U, 0U, 0U, 1U, 1U, 1U,
                    Fmt_UINT8);
    wafer_tx81_rdma(payload + 4096U, addresses.read1,
                    addresses.transfer_bytes, addresses.transfer_bytes, 0U,
                    0U, 0U, 1U, 1U, 1U, Fmt_UINT8);
  } else if (engine == WAFER_MDC_ENGINE_NE) {
    wafer_tx81_rdma(payload, addresses.read0, 256U, 256U, 0U, 0U, 0U, 1U, 1U,
                    1U, Fmt_UINT8);
    wafer_tx81_rdma(payload + 4096U, addresses.read1, 512U, 512U, 0U, 0U, 0U,
                    1U, 1U, 1U, Fmt_UINT8);
  } else if (engine == WAFER_MDC_ENGINE_WDMA ||
             engine == WAFER_MDC_ENGINE_TDMA) {
    wafer_tx81_rdma(payload, addresses.read0, addresses.transfer_bytes,
                    addresses.transfer_bytes, 0U, 0U, 0U, 1U, 1U, 1U,
                    Fmt_UINT8);
  }
  return 1U;
}

static void wafer_mdc_issue_engine(const WaferMDCRequest *request,
                                   uint32_t engine, uint32_t lane,
                                   uint64_t base, uint64_t payload_ddr,
                                   uint64_t output_ddr) {
  WaferMDCEngineAddresses addresses =
      wafer_mdc_engine_addresses(request, base);
  uint64_t payload = wafer_mdc_lane_payload(payload_ddr, lane);
  switch (engine) {
  case WAFER_MDC_ENGINE_CT:
    wafer_tx81_elementwise_add(addresses.read0, addresses.read1,
                               addresses.write,
                               addresses.transfer_bytes / 2U, Fmt_FP16);
    break;
  case WAFER_MDC_ENGINE_NE:
    wafer_tx81_gemm(addresses.read0, addresses.read1, addresses.write, 1U, 16U,
                    16U, 1U, Fmt_FP16);
    break;
  case WAFER_MDC_ENGINE_RDMA:
    wafer_tx81_rdma(payload + 8192U, addresses.write,
                    addresses.transfer_bytes, addresses.transfer_bytes, 0U,
                    0U, 0U, 1U, 1U, 1U, Fmt_UINT8);
    break;
  case WAFER_MDC_ENGINE_WDMA:
    wafer_tx81_wdma(addresses.read0, wafer_mdc_lane_sink(output_ddr, lane),
                    addresses.transfer_bytes, addresses.transfer_bytes, 0U,
                    0U, 0U, 1U, 1U, 1U, Fmt_UINT8);
    break;
  case WAFER_MDC_ENGINE_TDMA:
    wafer_tx81_gather_scatter(
        addresses.read0, addresses.write, addresses.transfer_bytes,
        addresses.transfer_bytes, 0U, 0U, 0U, 1U, 1U, 1U, 0U, 0U, 0U, 1U,
        1U, 1U);
    break;
  default:
    break;
  }
}

static uint64_t wafer_mdc_engine_result(const WaferMDCRequest *request,
                                        uint32_t engine, uint64_t base) {
  WaferMDCEngineAddresses addresses =
      wafer_mdc_engine_addresses(request, base);
  return engine == WAFER_MDC_ENGINE_WDMA ? addresses.read0 : addresses.write;
}

static void wafer_mdc_set_packet_worker(uint32_t *inter_type,
                                        uint32_t worker) {
  *inter_type =
      (*inter_type & ~UINT32_C(0x300)) | ((worker % 3U) << 8);
}

static void *wafer_mdc_prepared_packet(
    WaferMDCPreparedInstruction *instruction) {
  switch (instruction->engine) {
  case WAFER_MDC_ENGINE_CT:
    return &instruction->packet.ct;
  case WAFER_MDC_ENGINE_NE:
    return &instruction->packet.ne;
  case WAFER_MDC_ENGINE_RDMA:
    return &instruction->packet.rdma;
  case WAFER_MDC_ENGINE_WDMA:
    return &instruction->packet.wdma;
  case WAFER_MDC_ENGINE_TDMA:
    return &instruction->packet.tdma;
  default:
    return 0;
  }
}

static uint64_t wafer_mdc_perf_payload_address(
    uint64_t payload_ddr, uint32_t lane, uint64_t section,
    uint32_t index, uint32_t transfer) {
  return payload_ddr + WAFER_MDC_PERF_PAYLOAD_BASE +
         lane * WAFER_MDC_PERF_PAYLOAD_LANE_STRIDE + section +
         (uint64_t)index * transfer;
}

static uint64_t wafer_mdc_perf_output_address(
    uint64_t output_ddr, uint64_t section, uint32_t lane,
    uint32_t round, uint32_t transfer) {
  return output_ddr + section +
         lane * WAFER_MDC_PERF_OUTPUT_LANE_STRIDE +
         (uint64_t)round * transfer;
}

static uint32_t wafer_mdc_prepare_parallel_instruction(
    WaferMDCPreparedInstruction *instruction,
    const WaferMDCRequest *request, uint64_t payload_ddr,
    uint64_t output_ddr, uint32_t lane, uint32_t round) {
  uint32_t engine =
      lane == 0U ? request->engine_a : request->engine_b;
  uint32_t worker =
      lane == 0U ? request->worker_a : request->worker_b;
  uint64_t base = lane == 0U ? request->spm_a : request->spm_b;
  uint32_t transfer = request->compact_bytes;
  uint32_t buffer = round % request->buffer_count;
  uint64_t read0 = base + WAFER_MDC_PERF_READ0_OFFSET +
                   (uint64_t)buffer * transfer;
  uint64_t read1 = base + WAFER_MDC_PERF_READ1_OFFSET +
                   (engine == WAFER_MDC_ENGINE_NE
                        ? 0U
                        : (uint64_t)buffer * transfer);
  uint64_t write = base + WAFER_MDC_PERF_WRITE_OFFSET +
                   (uint64_t)round * transfer;
  instruction->engine = engine;
  if (engine == WAFER_MDC_ENGINE_CT) {
    TsmArith *arith = TsmNewArith();
    if (arith == 0)
      return 0U;
    arith->AddVV(&instruction->packet.ct, read0, read1, write,
                 transfer / sizeof(uint16_t), RND_NEAREST_EVEN,
                 Fmt_FP16);
    wafer_mdc_set_packet_worker(
        &instruction->packet.ct.inter_type, worker);
    TsmDeleteArith(arith);
  } else if (engine == WAFER_MDC_ENGINE_NE) {
    TsmGemm *gemm = TsmNewGemm();
    if (gemm == 0)
      return 0U;
    gemm->AddInput(&instruction->packet.ne, read0, read1, Fmt_FP16);
    gemm->ConfigMKN(&instruction->packet.ne, 64U, 128U, 128U);
    gemm->ConfigBatch(&instruction->packet.ne, 1U, 1U);
    gemm->SetTransflag(&instruction->packet.ne, 0U, 1U);
    gemm->SetPsum(&instruction->packet.ne, 0U, 0U, Fmt_UNUSED);
    gemm->SetQuant(&instruction->packet.ne, 0U, 0U, 0U, 0U);
    gemm->AddBias(&instruction->packet.ne, 0U, 0U);
    gemm->SetNegativeAxisScale(&instruction->packet.ne, 0U, 0U);
    gemm->SetPositiveAxisScale(&instruction->packet.ne, 0U, 0U);
    gemm->DisableRelu(&instruction->packet.ne);
    gemm->DisableLeakyRelu(&instruction->packet.ne);
    gemm->AddOutput(&instruction->packet.ne, write, Fmt_FP16);
    wafer_mdc_set_packet_worker(
        &instruction->packet.ne.inter_type, worker);
    TsmDeleteGemm(gemm);
  } else if (engine == WAFER_MDC_ENGINE_RDMA) {
    TsmRdma *rdma = TsmNewRdma();
    if (rdma == 0)
      return 0U;
    rdma->AddSrcDst(
        &instruction->packet.rdma,
        wafer_mdc_perf_payload_address(
            payload_ddr, lane, WAFER_MDC_PERF_PAYLOAD_RDMA_OFFSET,
            round, transfer),
        write, Fmt_UINT8);
    rdma->ConfigStrideIteration(&instruction->packet.rdma, transfer, 0U,
                                1U, 0U, 1U, 0U, 1U);
    wafer_mdc_set_packet_worker(
        &instruction->packet.rdma.inter_type, worker);
    TsmDeleteRdma(rdma);
  } else if (engine == WAFER_MDC_ENGINE_WDMA) {
    TsmWdma *wdma = TsmNewWdma();
    if (wdma == 0)
      return 0U;
    wdma->AddSrcDst(
        &instruction->packet.wdma, read0,
        wafer_mdc_perf_output_address(
            output_ddr, WAFER_MDC_PERF_OUTPUT_SINK_BASE, lane, round,
            transfer),
        Fmt_UINT8);
    wdma->ConfigStrideIteration(&instruction->packet.wdma, transfer, 0U,
                                1U, 0U, 1U, 0U, 1U);
    wafer_mdc_set_packet_worker(
        &instruction->packet.wdma.inter_type, worker);
    TsmDeleteWdma(wdma);
  } else if (engine == WAFER_MDC_ENGINE_TDMA) {
    St_StrideIteration stride = {
        transfer, 1U, 0U, 1U, 0U, 1U,
    };
    TsmPeripheral *peripheral = TsmNewPeripheral();
    if (peripheral == 0)
      return 0U;
    peripheral->Memset(
        &instruction->packet.tdma, write,
        (request->sample + lane + round + 2U) & 0xffU, transfer,
        &stride, Fmt_UINT8);
    wafer_mdc_set_packet_worker(
        &instruction->packet.tdma.inter_type, worker);
    TsmDeletePeripheral(peripheral);
  } else {
    return 0U;
  }
  return 1U;
}

static uint32_t wafer_mdc_drain_worker(uint32_t worker,
                                       uint64_t *final_control);

static uint32_t wafer_mdc_seed_parallel_range(
    uint64_t payload_ddr, uint64_t begin, uint64_t end) {
  uint64_t cursor = begin - WAFER_MDC_SPM_GUARD_BYTES;
  uint64_t limit = end + WAFER_MDC_SPM_GUARD_BYTES;
  uint32_t pending = 0U;
  uint64_t control = 0U;
  while (cursor < limit) {
    uint32_t chunk =
        limit - cursor > 65536U ? 65536U : (uint32_t)(limit - cursor);
    wafer_tx81_rdma(
        payload_ddr + WAFER_MDC_PERF_PAYLOAD_GUARD_SEED_OFFSET,
        cursor, chunk, chunk, 0U, 0U, 0U, 1U, 1U, 1U, Fmt_UINT8);
    cursor += chunk;
    if (++pending == 4U) {
      if (!wafer_mdc_drain_worker(0U, &control))
        return 0U;
      pending = 0U;
    }
  }
  return pending == 0U ||
         wafer_mdc_drain_worker(0U, &control);
}

static void wafer_mdc_load_parallel_inputs(
    const WaferMDCRequest *request, uint64_t payload_ddr,
    uint32_t lane) {
  uint32_t engine =
      lane == 0U ? request->engine_a : request->engine_b;
  uint64_t base = lane == 0U ? request->spm_a : request->spm_b;
  uint32_t transfer = request->compact_bytes;
  if (engine != WAFER_MDC_ENGINE_CT &&
      engine != WAFER_MDC_ENGINE_NE &&
      engine != WAFER_MDC_ENGINE_WDMA)
    return;
  for (uint32_t buffer = 0; buffer < request->buffer_count; ++buffer) {
    wafer_tx81_rdma(
        wafer_mdc_perf_payload_address(
            payload_ddr, lane, WAFER_MDC_PERF_PAYLOAD_READ0_OFFSET,
            buffer, transfer),
        base + WAFER_MDC_PERF_READ0_OFFSET +
            (uint64_t)buffer * transfer,
        transfer, transfer, 0U, 0U, 0U, 1U, 1U, 1U, Fmt_UINT8);
    if (engine == WAFER_MDC_ENGINE_CT ||
        engine == WAFER_MDC_ENGINE_NE) {
      if (engine == WAFER_MDC_ENGINE_NE && buffer != 0U)
        continue;
      uint32_t read1_bytes =
          engine == WAFER_MDC_ENGINE_NE ? 32768U : transfer;
      wafer_tx81_rdma(
          wafer_mdc_perf_payload_address(
              payload_ddr, lane, WAFER_MDC_PERF_PAYLOAD_READ1_OFFSET,
              buffer, transfer),
          base + WAFER_MDC_PERF_READ1_OFFSET +
              (engine == WAFER_MDC_ENGINE_NE
                   ? 0U
                   : (uint64_t)buffer * transfer),
          read1_bytes, read1_bytes, 0U, 0U, 0U, 1U, 1U, 1U,
          Fmt_UINT8);
    }
  }
}

static uint32_t wafer_mdc_drain_worker(uint32_t worker,
                                       uint64_t *final_control) {
  __asm__ volatile("fence iorw, iorw" ::: "memory");
  for (uint32_t spin = 0; spin < WAFER_MDC_DRAIN_SPINS; ++spin) {
    uint32_t control = get_ncc_reg(worker, GR_CSR_CONTROL_ADDR);
    if ((control & UINT32_C(0xff)) == 0U &&
        (control & UINT32_C(0x100)) != 0U) {
      *final_control = control;
      __asm__ volatile("fence iorw, iorw" ::: "memory");
      __asm__ volatile("sync" ::: "memory");
      return 1U;
    }
  }
  *final_control = get_ncc_reg(worker, GR_CSR_CONTROL_ADDR);
  return 0U;
}

static void wafer_mdc_seed_relation(const WaferMDCRequest *request,
                                    uint64_t payload_ddr) {
  wafer_mdc_seed_slot(payload_ddr, request->spm_a, request->output_bytes);
  if (request->relation == WAFER_MDC_RELATION_STRIDED) {
    uint64_t source = request->spm_a + 0x20000U;
    uint64_t destination = request->spm_a + 0x24000U;
    wafer_tx81_rdma(payload_ddr + WAFER_MDC_PAYLOAD_SPM_SCRATCH0_OFFSET,
                    source, 8192U, 8192U, 0U, 0U, 0U, 1U, 1U, 1U,
                    Fmt_UINT8);
    wafer_tx81_rdma(payload_ddr + WAFER_MDC_PAYLOAD_SPM_SCRATCH1_OFFSET,
                    destination, 8192U, 8192U, 0U, 0U, 0U, 1U, 1U, 1U,
                    Fmt_UINT8);
  }
  wafer_tx81_local_fence();
}

static void wafer_mdc_relation_tdma_write(uint64_t source,
                                          uint64_t destination) {
  wafer_tx81_gather_scatter(source, destination, 2048U, 128U, 128U, 0U, 0U,
                            16U, 1U, 1U, 256U, 0U, 0U, 16U, 1U, 1U);
}

static void wafer_mdc_relation_tdma_read(uint64_t source,
                                         uint64_t destination) {
  wafer_tx81_gather_scatter(source, destination, 2048U, 128U, 256U, 0U, 0U,
                            16U, 1U, 1U, 128U, 0U, 0U, 16U, 1U, 1U);
}

static void wafer_mdc_issue_relation(const WaferMDCRequest *request,
                                     uint64_t payload_ddr,
                                     uint64_t output_ddr) {
  if (request->relation == WAFER_MDC_RELATION_STRIDED) {
    uint64_t source_a = request->spm_a + 0x20000U;
    uint64_t source_b = source_a + 4096U;
    uint64_t destination_a = request->spm_a + 0x24000U;
    uint64_t destination_b = destination_a + 4096U;
    switch (request->effect) {
    case WAFER_MDC_EFFECT_RAW:
      wafer_mdc_relation_tdma_write(source_a, request->spm_a);
      wafer_mdc_relation_tdma_read(request->spm_b, destination_b);
      break;
    case WAFER_MDC_EFFECT_WAR:
      wafer_mdc_relation_tdma_read(request->spm_a, destination_a);
      wafer_mdc_relation_tdma_write(source_b, request->spm_b);
      break;
    case WAFER_MDC_EFFECT_WAW:
      wafer_mdc_relation_tdma_write(source_a, request->spm_a);
      wafer_mdc_relation_tdma_write(source_b, request->spm_b);
      break;
    case WAFER_MDC_EFFECT_RAR:
      wafer_mdc_relation_tdma_read(request->spm_a, destination_a);
      wafer_mdc_relation_tdma_read(request->spm_b, destination_b);
      break;
    default:
      break;
    }
    return;
  }
  uint64_t payload_a =
      payload_ddr + WAFER_MDC_PAYLOAD_DATA_OFFSET;
  uint64_t payload_b = payload_a + 16384U;
  uint64_t sink_a = output_ddr + WAFER_MDC_OUTPUT_SINK0_OFFSET;
  uint64_t sink_b = output_ddr + WAFER_MDC_OUTPUT_SINK1_OFFSET;
  switch (request->effect) {
  case WAFER_MDC_EFFECT_RAW:
    wafer_tx81_rdma(payload_a, request->spm_a, 2048U, 2048U, 0U, 0U, 0U, 1U,
                    1U, 1U, Fmt_UINT8);
    wafer_tx81_wdma(request->spm_b, sink_b, 2048U, 2048U, 0U, 0U, 0U, 1U, 1U,
                    1U, Fmt_UINT8);
    break;
  case WAFER_MDC_EFFECT_WAR:
    wafer_tx81_wdma(request->spm_a, sink_a, 2048U, 2048U, 0U, 0U, 0U, 1U, 1U,
                    1U, Fmt_UINT8);
    wafer_tx81_rdma(payload_b, request->spm_b, 2048U, 2048U, 0U, 0U, 0U, 1U,
                    1U, 1U, Fmt_UINT8);
    break;
  case WAFER_MDC_EFFECT_WAW:
    wafer_tx81_rdma(payload_a, request->spm_a, 2048U, 2048U, 0U, 0U, 0U, 1U,
                    1U, 1U, Fmt_UINT8);
    wafer_tx81_rdma(payload_b, request->spm_b, 2048U, 2048U, 0U, 0U, 0U, 1U,
                    1U, 1U, Fmt_UINT8);
    break;
  case WAFER_MDC_EFFECT_RAR:
    wafer_tx81_wdma(request->spm_a, sink_a, 2048U, 2048U, 0U, 0U, 0U, 1U, 1U,
                    1U, Fmt_UINT8);
    wafer_tx81_wdma(request->spm_b, sink_b, 2048U, 2048U, 0U, 0U, 0U, 1U, 1U,
                    1U, Fmt_UINT8);
    break;
  default:
    break;
  }
}

static void wafer_mdc_record_request(volatile uint64_t *record,
                                     const WaferMDCRequest *request) {
  record[WAFER_MDC_REC_CASE] = request->case_id;
  record[WAFER_MDC_REC_KIND] = request->kind;
  record[WAFER_MDC_REC_ENGINE_A] = request->engine_a;
  record[WAFER_MDC_REC_ENGINE_B] = request->engine_b;
  record[WAFER_MDC_REC_SCHEDULE] = request->schedule;
  record[WAFER_MDC_REC_EFFECT] = request->effect;
  record[WAFER_MDC_REC_RELATION] = request->relation;
  record[WAFER_MDC_REC_ORACLE] = request->oracle;
  record[WAFER_MDC_REC_FORMAT] = request->format;
  record[WAFER_MDC_REC_SRC_DDR_OFFSET] = request->src_ddr_offset;
  record[WAFER_MDC_REC_DST_DDR_OFFSET] = request->dst_ddr_offset;
  record[WAFER_MDC_REC_SPM_A] = request->spm_a;
  record[WAFER_MDC_REC_SPM_B] = request->spm_b;
  record[WAFER_MDC_REC_INNER_BYTES] = request->inner_bytes;
  record[WAFER_MDC_REC_STRIDE0] = request->stride0;
  record[WAFER_MDC_REC_STRIDE1] = request->stride1;
  record[WAFER_MDC_REC_STRIDE2] = request->stride2;
  record[WAFER_MDC_REC_ITERATION0] = request->iteration0;
  record[WAFER_MDC_REC_ITERATION1] = request->iteration1;
  record[WAFER_MDC_REC_ITERATION2] = request->iteration2;
  record[WAFER_MDC_REC_COMPACT_BYTES] = request->compact_bytes;
  record[WAFER_MDC_REC_ENVELOPE_BYTES] = request->envelope_bytes;
  record[WAFER_MDC_REC_OUTPUT_BYTES] = request->output_bytes;
  record[WAFER_MDC_REC_SAMPLE] = request->sample;
  record[WAFER_MDC_REC_OUTPUT_DATA_OFFSET] = WAFER_MDC_OUTPUT_DATA_OFFSET;
  record[WAFER_MDC_REC_REQUEST_GUARD] = WAFER_MDC_REQUEST_GUARD;
  record[WAFER_MDC_REC_WORKER_A] = request->worker_a;
  record[WAFER_MDC_REC_WORKER_B] = request->worker_b;
  record[WAFER_MDC_REC_ROUNDS] = request->rounds;
  record[WAFER_MDC_REC_BUFFER_COUNT] = request->buffer_count;
  record[WAFER_MDC_REC_ISSUE_ORDER] = request->issue_order;
}

static void wafer_mdc_record_pmu(volatile uint64_t *record,
                                 const WaferMDCPMU *before,
                                 const WaferMDCPMU *after,
                                 uint64_t plan_cycles) {
  for (uint32_t engine = 0; engine < WAFER_MDC_ENGINES; ++engine) {
    record[WAFER_MDC_REC_CT_INST_DELTA + engine] =
        (uint32_t)(after->instructions[engine] -
                   before->instructions[engine]);
    record[WAFER_MDC_REC_CT_EXEC_DELTA + engine] =
        after->execution[engine] - before->execution[engine];
    record[WAFER_MDC_REC_CT_BLOCKING_DELTA + engine] =
        (uint32_t)(after->blocking[engine] - before->blocking[engine]);
  }
  record[WAFER_MDC_REC_FULL_EXEC_DELTA] =
      after->full_execution - before->full_execution;
  record[WAFER_MDC_REC_PLAN_CYCLES] = plan_cycles;
  record[WAFER_MDC_REC_PMU_ENABLE] = wafer_mdc_read32(GR_PMU_EN);
  record[WAFER_MDC_REC_SERIAL_MODE] =
      get_ncc_reg(0U, GR_CSR_SERIAL_MODE_ADDR);
  record[WAFER_MDC_REC_STABLE_BEFORE] = before->stable_mask;
  record[WAFER_MDC_REC_STABLE_AFTER] = after->stable_mask;
}

static void wafer_mdc_init_record(volatile uint64_t *record,
                                  uint32_t status) {
  for (uint32_t index = 0; index < WAFER_MDC_RECORD_WORDS; ++index)
    record[index] = 0;
  record[WAFER_MDC_REC_MAGIC] = WAFER_MDC_RECORD_MAGIC;
  record[WAFER_MDC_REC_SCHEMA_AND_WORDS] =
      ((uint64_t)WAFER_MDC_SCHEMA << 32) | WAFER_MDC_RECORD_WORDS;
  record[WAFER_MDC_REC_STATUS] = status;
  record[WAFER_MDC_REC_OUTPUT_DATA_OFFSET] = WAFER_MDC_OUTPUT_DATA_OFFSET;
  record[WAFER_MDC_REC_RECORD_GUARD] = WAFER_MDC_RECORD_GUARD;
}

static uint32_t wafer_mdc_execute_dma(const WaferMDCRequest *request,
                                      uint64_t payload_ddr,
                                      uint64_t output_ddr,
                                      volatile uint64_t *record) {
  wafer_mdc_seed_slot(payload_ddr, request->spm_a, request->compact_bytes);
  wafer_tx81_local_fence();
  WaferMDCPMU before = wafer_mdc_read_pmu();
  uint64_t plan_begin = wafer_mdc_cycle();
  wafer_mdc_rdma_descriptor(
      payload_ddr + WAFER_MDC_PAYLOAD_DATA_OFFSET + request->src_ddr_offset,
      request->spm_a, request);
  wafer_tx81_local_fence();
  wafer_mdc_wdma_descriptor(
      request->spm_a,
      output_ddr + WAFER_MDC_OUTPUT_DATA_OFFSET + request->dst_ddr_offset,
      request);
  wafer_tx81_local_fence();
  uint64_t plan_end = wafer_mdc_cycle();
  WaferMDCPMU after = wafer_mdc_read_pmu();
  wafer_mdc_record_pmu(record, &before, &after, plan_end - plan_begin);
  wafer_mdc_dump_slot(output_ddr, wafer_mdc_first_spm_dump_offset(request),
                      request->spm_a, request->compact_bytes);
  wafer_tx81_local_fence();
  record[WAFER_MDC_REC_SPM_GUARD_MISMATCHES] = 0U;
  record[WAFER_MDC_REC_FLAGS] =
      WAFER_MDC_FLAG_PREPARED | WAFER_MDC_FLAG_ISSUED |
      WAFER_MDC_FLAG_COMPLETED | WAFER_MDC_FLAG_READBACK;
  return 1U;
}

static uint32_t wafer_mdc_execute_engines(const WaferMDCRequest *request,
                                          uint64_t payload_ddr,
                                          uint64_t output_ddr,
                                          volatile uint64_t *record) {
  uint32_t lanes =
      request->kind == WAFER_MDC_KIND_ENGINE_ACCESS ? 1U : 2U;
  uint32_t engines[2] = {request->engine_a, request->engine_b};
  uint64_t bases[2] = {request->spm_a, request->spm_b};
  for (uint32_t lane = 0; lane < lanes; ++lane)
    if (!wafer_mdc_prepare_engine(request, engines[lane], lane, bases[lane],
                                  payload_ddr))
      return 0U;
  /* Keep all setup RDMA outside the measured pair window. */
  wafer_tx81_local_fence();
  WaferMDCPMU before = wafer_mdc_read_pmu();
  uint64_t plan_begin = wafer_mdc_cycle();
  uint32_t first =
      request->kind == WAFER_MDC_KIND_ENGINE_PAIR &&
              request->issue_order != 0U
          ? 1U
          : 0U;
  uint32_t second = first ^ 1U;
  wafer_mdc_issue_engine(request, engines[first], first, bases[first],
                         payload_ddr, output_ddr);
  if (lanes == 2U &&
      request->schedule == WAFER_MDC_SCHEDULE_SERIAL)
    wafer_tx81_local_fence();
  if (lanes == 2U)
    wafer_mdc_issue_engine(request, engines[second], second, bases[second],
                           payload_ddr, output_ddr);
  wafer_tx81_local_fence();
  uint64_t plan_end = wafer_mdc_cycle();
  WaferMDCPMU after = wafer_mdc_read_pmu();
  wafer_mdc_record_pmu(record, &before, &after, plan_end - plan_begin);
  uint64_t dump_offset = wafer_mdc_first_spm_dump_offset(request);
  for (uint32_t lane = 0; lane < lanes; ++lane) {
    WaferMDCEngineAddresses addresses =
        wafer_mdc_engine_addresses(request, bases[lane]);
    uint32_t transfer = addresses.transfer_bytes;
    uint32_t output_offset =
        WAFER_MDC_OUTPUT_DATA_OFFSET +
        (request->kind == WAFER_MDC_KIND_ENGINE_PAIR
             ? lane * request->compact_bytes
             : 0U);
    wafer_tx81_wdma(wafer_mdc_engine_result(request, engines[lane],
                                            bases[lane]),
                    output_ddr + output_offset, transfer, transfer, 0U, 0U, 0U,
                    1U, 1U, 1U, Fmt_UINT8);
    dump_offset = wafer_mdc_dump_slot(output_ddr, dump_offset, bases[lane],
                                      addresses.slot_span);
  }
  wafer_tx81_local_fence();
  record[WAFER_MDC_REC_SPM_GUARD_MISMATCHES] = 0U;
  record[WAFER_MDC_REC_FLAGS] =
      WAFER_MDC_FLAG_PREPARED | WAFER_MDC_FLAG_ISSUED |
      WAFER_MDC_FLAG_COMPLETED | WAFER_MDC_FLAG_READBACK;
  return 1U;
}

static uint32_t wafer_mdc_issue_parallel_lane(
    WaferMDCPreparedInstruction instructions[2][WAFER_MDC_PERF_MAX_ROUNDS],
    uint32_t lane, uint32_t rounds) {
  for (uint32_t round = 0; round < rounds; ++round)
    if (TsmExecute(wafer_mdc_prepared_packet(
                       &instructions[lane][round])) != 1)
      return 0U;
  return 1U;
}

static uint32_t wafer_mdc_execute_parallel(
    const WaferMDCRequest *request, uint64_t payload_ddr,
    uint64_t output_ddr, volatile uint64_t *record) {
  uint64_t active_begin[2];
  uint64_t active_end[2];
  for (uint32_t lane = 0; lane < 2U; ++lane)
    wafer_mdc_parallel_active_range(
        request, lane, &active_begin[lane], &active_end[lane]);
  if (request->relation == WAFER_MDC_RELATION_DISJOINT) {
    for (uint32_t lane = 0; lane < 2U; ++lane)
      if (!wafer_mdc_seed_parallel_range(
              payload_ddr, active_begin[lane], active_end[lane]))
        return 0U;
  } else {
    if (!wafer_mdc_seed_parallel_range(
            payload_ddr,
            active_begin[0] < active_begin[1] ? active_begin[0]
                                              : active_begin[1],
            active_end[0] > active_end[1] ? active_end[0]
                                          : active_end[1]))
      return 0U;
  }
  uint64_t setup_control = 0U;
  for (uint32_t lane = 0; lane < 2U; ++lane) {
    wafer_mdc_load_parallel_inputs(request, payload_ddr, lane);
    if (!wafer_mdc_drain_worker(0U, &setup_control))
      return 0U;
  }

  WaferMDCPreparedInstruction
      instructions[2][WAFER_MDC_PERF_MAX_ROUNDS] = {0};
  for (uint32_t lane = 0; lane < 2U; ++lane)
    for (uint32_t round = 0; round < request->rounds; ++round)
      if (!wafer_mdc_prepare_parallel_instruction(
              &instructions[lane][round], request, payload_ddr,
              output_ddr, lane, round))
        return 0U;

  uint32_t first = request->issue_order == 0U ? 0U : 1U;
  uint32_t second = first ^ 1U;
  uint32_t workers[2] = {request->worker_a, request->worker_b};
  uint64_t final_control[2] = {0U, 0U};
  WaferMDCPMU before = wafer_mdc_read_pmu();
  uint64_t plan_begin = wafer_mdc_cycle();
  if (request->schedule == WAFER_MDC_SCHEDULE_SERIAL) {
    if (!wafer_mdc_issue_parallel_lane(
            instructions, first, request->rounds) ||
        !wafer_mdc_drain_worker(
            workers[first], &final_control[first]) ||
        !wafer_mdc_issue_parallel_lane(
            instructions, second, request->rounds) ||
        !wafer_mdc_drain_worker(
            workers[second], &final_control[second]))
      return 0U;
  } else {
    for (uint32_t round = 0; round < request->rounds; ++round) {
      if (TsmExecute(wafer_mdc_prepared_packet(
                         &instructions[first][round])) != 1 ||
          TsmExecute(wafer_mdc_prepared_packet(
                         &instructions[second][round])) != 1)
        return 0U;
    }
    if (!wafer_mdc_drain_worker(
            workers[first], &final_control[first]))
      return 0U;
    if (workers[second] == workers[first])
      final_control[second] = final_control[first];
    else if (!wafer_mdc_drain_worker(
                 workers[second], &final_control[second]))
      return 0U;
  }
  uint64_t plan_end = wafer_mdc_cycle();
  WaferMDCPMU after = wafer_mdc_read_pmu();
  wafer_mdc_record_pmu(
      record, &before, &after, plan_end - plan_begin);
  record[WAFER_MDC_REC_LANE_A_WORKER_INST_DELTA] =
      (uint32_t)(after.worker_instructions[request->worker_a]
                                          [request->engine_a] -
                 before.worker_instructions[request->worker_a]
                                           [request->engine_a]);
  record[WAFER_MDC_REC_LANE_B_WORKER_INST_DELTA] =
      (uint32_t)(after.worker_instructions[request->worker_b]
                                          [request->engine_b] -
                 before.worker_instructions[request->worker_b]
                                           [request->engine_b]);
  record[WAFER_MDC_REC_LANE_A_WORKER_BLOCKING_DELTA] =
      (uint32_t)(after.worker_blocking[request->worker_a]
                                      [request->engine_a] -
                 before.worker_blocking[request->worker_a]
                                       [request->engine_a]);
  record[WAFER_MDC_REC_LANE_B_WORKER_BLOCKING_DELTA] =
      (uint32_t)(after.worker_blocking[request->worker_b]
                                      [request->engine_b] -
                 before.worker_blocking[request->worker_b]
                                       [request->engine_b]);
  record[WAFER_MDC_REC_WORKER_MASK] =
      (UINT64_C(1) << request->worker_a) |
      (UINT64_C(1) << request->worker_b);
  record[WAFER_MDC_REC_CONTROL_FINAL] =
      (final_control[0] & UINT64_C(0xffffffff)) |
      ((final_control[1] & UINT64_C(0xffffffff)) << 32);

  uint32_t engines[2] = {request->engine_a, request->engine_b};
  uint64_t bases[2] = {request->spm_a, request->spm_b};
  uint32_t pending_readbacks = 0U;
  for (uint32_t lane = 0; lane < 2U; ++lane) {
    for (uint32_t round = 0; round < request->rounds; ++round) {
      uint32_t buffer = round % request->buffer_count;
      uint64_t source =
          engines[lane] == WAFER_MDC_ENGINE_WDMA
              ? bases[lane] + WAFER_MDC_PERF_READ0_OFFSET +
                    (uint64_t)buffer * request->compact_bytes
              : bases[lane] + WAFER_MDC_PERF_WRITE_OFFSET +
                    (uint64_t)round * request->compact_bytes;
      wafer_tx81_wdma(
          source,
          wafer_mdc_perf_output_address(
              output_ddr, WAFER_MDC_PERF_OUTPUT_RESULT_BASE, lane,
              round, request->compact_bytes),
          request->compact_bytes, request->compact_bytes, 0U, 0U, 0U,
          1U, 1U, 1U, Fmt_UINT8);
      if (++pending_readbacks == 4U) {
        if (!wafer_mdc_drain_worker(0U, &setup_control))
          return 0U;
        pending_readbacks = 0U;
      }
    }
  }

  uint64_t guard_begin[2] = {active_begin[0], active_begin[1]};
  uint64_t guard_end[2] = {active_end[0], active_end[1]};
  if (request->relation != WAFER_MDC_RELATION_DISJOINT) {
    uint64_t begin =
        active_begin[0] < active_begin[1] ? active_begin[0]
                                          : active_begin[1];
    uint64_t end = active_end[0] > active_end[1] ? active_end[0]
                                                 : active_end[1];
    guard_begin[0] = guard_begin[1] = begin;
    guard_end[0] = guard_end[1] = end;
  }
  for (uint32_t lane = 0; lane < 2U; ++lane) {
    uint64_t guard_output =
        output_ddr + WAFER_MDC_PERF_OUTPUT_GUARD_BASE +
        lane * WAFER_MDC_PERF_OUTPUT_GUARD_LANE_STRIDE;
    wafer_tx81_wdma(
        guard_begin[lane] - WAFER_MDC_SPM_GUARD_BYTES,
        guard_output, WAFER_MDC_SPM_GUARD_BYTES,
        WAFER_MDC_SPM_GUARD_BYTES, 0U, 0U, 0U, 1U, 1U, 1U,
        Fmt_UINT8);
    if (++pending_readbacks == 4U) {
      if (!wafer_mdc_drain_worker(0U, &setup_control))
        return 0U;
      pending_readbacks = 0U;
    }
    wafer_tx81_wdma(
        guard_end[lane], guard_output + WAFER_MDC_SPM_GUARD_BYTES,
        WAFER_MDC_SPM_GUARD_BYTES, WAFER_MDC_SPM_GUARD_BYTES, 0U, 0U,
        0U, 1U, 1U, 1U, Fmt_UINT8);
    if (++pending_readbacks == 4U) {
      if (!wafer_mdc_drain_worker(0U, &setup_control))
        return 0U;
      pending_readbacks = 0U;
    }
  }
  if (pending_readbacks != 0U &&
      !wafer_mdc_drain_worker(0U, &setup_control))
    return 0U;
  record[WAFER_MDC_REC_SPM_GUARD_MISMATCHES] = 0U;
  record[WAFER_MDC_REC_FLAGS] =
      WAFER_MDC_FLAG_PREPARED | WAFER_MDC_FLAG_ISSUED |
      WAFER_MDC_FLAG_COMPLETED | WAFER_MDC_FLAG_READBACK;
  return 1U;
}

static uint32_t wafer_mdc_execute_relation(const WaferMDCRequest *request,
                                           uint64_t payload_ddr,
                                           uint64_t output_ddr,
                                           volatile uint64_t *record) {
  wafer_mdc_seed_relation(request, payload_ddr);
  WaferMDCPMU before = wafer_mdc_read_pmu();
  uint64_t plan_begin = wafer_mdc_cycle();
  wafer_mdc_issue_relation(request, payload_ddr, output_ddr);
  wafer_tx81_local_fence();
  uint64_t plan_end = wafer_mdc_cycle();
  WaferMDCPMU after = wafer_mdc_read_pmu();
  wafer_mdc_record_pmu(record, &before, &after, plan_end - plan_begin);
  wafer_tx81_wdma(request->spm_a,
                  output_ddr + WAFER_MDC_OUTPUT_DATA_OFFSET,
                  request->output_bytes, request->output_bytes, 0U, 0U, 0U, 1U,
                  1U, 1U, Fmt_UINT8);
  wafer_mdc_dump_slot(output_ddr, wafer_mdc_first_spm_dump_offset(request),
                      request->spm_a, request->output_bytes);
  wafer_tx81_local_fence();
  record[WAFER_MDC_REC_SPM_GUARD_MISMATCHES] = 0U;
  record[WAFER_MDC_REC_FLAGS] =
      WAFER_MDC_FLAG_PREPARED | WAFER_MDC_FLAG_ISSUED |
      WAFER_MDC_FLAG_COMPLETED | WAFER_MDC_FLAG_READBACK;
  return 1U;
}

static void wafer_mdc_execute_wire(const volatile uint64_t *wire,
                                   uint64_t payload_ddr,
                                   uint64_t output_ddr) {
  volatile uint64_t *record = (volatile uint64_t *)(uintptr_t)output_ddr;
  WaferMDCRequest request = {0};
  uint32_t status = wafer_mdc_decode(wire, &request);
  wafer_mdc_init_record(record, status);
  if (status == WAFER_MDC_STATUS_OK) {
    wafer_mdc_record_request(record, &request);
    uint32_t executed = 0U;
    if (request.kind == WAFER_MDC_KIND_DMA)
      executed =
          wafer_mdc_execute_dma(&request, payload_ddr, output_ddr, record);
    else if (request.kind == WAFER_MDC_KIND_ADDRESS_RELATION)
      executed = wafer_mdc_execute_relation(&request, payload_ddr, output_ddr,
                                            record);
    else if (request.kind == WAFER_MDC_KIND_PARALLEL_PAIR)
      executed = wafer_mdc_execute_parallel(
          &request, payload_ddr, output_ddr, record);
    else
      executed =
          wafer_mdc_execute_engines(&request, payload_ddr, output_ddr, record);
    if (!executed)
      status = WAFER_MDC_STATUS_EXECUTE_FAILED;
    record[WAFER_MDC_REC_STATUS] = status;
  }
  wafer_mdc_cache_range(output_ddr,
                        WAFER_MDC_RECORD_WORDS * sizeof(uint64_t), 0U);
}

static uint32_t wafer_mdc_ce_same_pair(
    const WaferMDCRequest *serial, const WaferMDCRequest *window,
    const volatile uint64_t *meta) {
  if (serial->kind != WAFER_MDC_KIND_ENGINE_PAIR ||
      window->kind != WAFER_MDC_KIND_ENGINE_PAIR ||
      serial->engine_a != WAFER_MDC_ENGINE_CT ||
      serial->engine_b != WAFER_MDC_ENGINE_RDMA ||
      window->engine_a != serial->engine_a ||
      window->engine_b != serial->engine_b ||
      serial->schedule != WAFER_MDC_SCHEDULE_SERIAL ||
      window->schedule != WAFER_MDC_SCHEDULE_WINDOW ||
      serial->effect != WAFER_MDC_EFFECT_NONE ||
      window->effect != serial->effect ||
      serial->relation != WAFER_MDC_RELATION_DISJOINT ||
      window->relation != serial->relation ||
      serial->oracle != WAFER_MDC_ORACLE_EXACT ||
      window->oracle != serial->oracle ||
      serial->format != Fmt_UINT8 || window->format != serial->format ||
      serial->src_ddr_offset != 0U ||
      window->src_ddr_offset != serial->src_ddr_offset ||
      serial->dst_ddr_offset != 0U ||
      window->dst_ddr_offset != serial->dst_ddr_offset ||
      serial->spm_a != window->spm_a ||
      serial->spm_b != window->spm_b ||
      serial->spm_b - serial->spm_a != 8192U ||
      serial->inner_bytes != window->inner_bytes ||
      serial->stride0 != window->stride0 ||
      serial->stride1 != window->stride1 ||
      serial->stride2 != window->stride2 ||
      serial->iteration0 != window->iteration0 ||
      serial->iteration1 != window->iteration1 ||
      serial->iteration2 != window->iteration2 ||
      serial->compact_bytes != window->compact_bytes ||
      serial->envelope_bytes != window->envelope_bytes ||
      serial->output_bytes != window->output_bytes ||
      serial->sample != window->sample ||
      serial->worker_a != window->worker_a ||
      serial->worker_b != window->worker_b ||
      serial->rounds != window->rounds ||
      serial->buffer_count != window->buffer_count ||
      serial->issue_order != window->issue_order)
    return 0U;
  for (uint32_t engine = 0; engine < WAFER_MDC_ENGINES; ++engine)
    if (serial->expected_instructions[engine] !=
        window->expected_instructions[engine])
      return 0U;
  return meta[WAFER_MDC_CE_REQ_SERIAL_CASE] == serial->case_id &&
         meta[WAFER_MDC_CE_REQ_WINDOW_CASE] == window->case_id &&
         meta[WAFER_MDC_CE_REQ_SPM_A] == serial->spm_a &&
         meta[WAFER_MDC_CE_REQ_SPM_B] == serial->spm_b &&
         meta[WAFER_MDC_CE_REQ_TRANSFER_BYTES] ==
             serial->compact_bytes &&
         meta[WAFER_MDC_CE_REQ_ISSUE_ORDER] ==
             serial->issue_order &&
         meta[WAFER_MDC_CE_REQ_SAMPLE] == serial->sample;
}

static uint32_t wafer_mdc_ce_decode(
    const volatile uint64_t *wire, WaferMDCRequest *serial,
    WaferMDCRequest *window) {
  const volatile uint64_t *meta =
      wire + WAFER_MDC_CE_REQUEST_META_WORD;
  if (meta[WAFER_MDC_CE_REQ_MAGIC] !=
          WAFER_MDC_CE_REQUEST_MAGIC ||
      meta[WAFER_MDC_CE_REQ_SCHEMA_AND_WORDS] !=
          (((uint64_t)WAFER_MDC_CE_SCHEMA << 32) |
           WAFER_MDC_CE_REQUEST_META_WORDS) ||
      meta[WAFER_MDC_CE_REQ_SERIAL_REQUEST_WORD] !=
          WAFER_MDC_CE_SERIAL_REQUEST_WORD ||
      meta[WAFER_MDC_CE_REQ_WINDOW_REQUEST_WORD] !=
          WAFER_MDC_CE_WINDOW_REQUEST_WORD ||
      meta[WAFER_MDC_CE_REQ_RESOURCE_BYTES] !=
          WAFER_MDC_RESOURCE_BYTES ||
      meta[WAFER_MDC_CE_REQ_SERIAL_OUTPUT_OFFSET] !=
          WAFER_MDC_CE_SERIAL_OUTPUT_OFFSET ||
      meta[WAFER_MDC_CE_REQ_WINDOW_OUTPUT_OFFSET] !=
          WAFER_MDC_CE_WINDOW_OUTPUT_OFFSET ||
      meta[WAFER_MDC_CE_REQ_OUTPUT_ROW_BYTES] !=
          WAFER_MDC_CE_OUTPUT_ROW_BYTES ||
      meta[WAFER_MDC_CE_REQ_RECORD_META_WORD] !=
          WAFER_MDC_CE_RECORD_META_WORD ||
      meta[WAFER_MDC_CE_REQ_FIRST_SCHEDULE] >
          WAFER_MDC_SCHEDULE_WINDOW ||
      meta[WAFER_MDC_CE_REQ_SAMPLE] >= WAFER_MDC_CE_SAMPLES ||
      meta[WAFER_MDC_CE_REQ_ROWS] != WAFER_MDC_CE_ROWS ||
      meta[WAFER_MDC_CE_REQ_GUARD] !=
          WAFER_MDC_CE_REQUEST_GUARD)
    return WAFER_MDC_CE_STATUS_BAD_ENVELOPE;
  if (wafer_mdc_decode(
          wire + WAFER_MDC_CE_SERIAL_REQUEST_WORD, serial) !=
          WAFER_MDC_STATUS_OK ||
      wafer_mdc_decode(
          wire + WAFER_MDC_CE_WINDOW_REQUEST_WORD, window) !=
          WAFER_MDC_STATUS_OK ||
      !wafer_mdc_ce_same_pair(serial, window, meta))
    return WAFER_MDC_CE_STATUS_BAD_PAIR;
  return WAFER_MDC_CE_STATUS_OK;
}

static void wafer_mdc_ce_write_meta(
    uint64_t request_ddr, uint64_t payload_ddr, uint64_t output_ddr,
    uint64_t row_output_ddr, uint32_t row, uint32_t status,
    uint32_t ordinal, const WaferMDCRequest *request,
    const volatile uint64_t *request_meta) {
  volatile uint64_t *record =
      (volatile uint64_t *)(uintptr_t)output_ddr +
      WAFER_MDC_CE_RECORD_META_WORD +
      row * WAFER_MDC_CE_RECORD_META_STRIDE_WORDS;
  for (uint32_t word = 0; word < WAFER_MDC_CE_RECORD_META_WORDS;
       ++word)
    record[word] = 0U;
  uint64_t inner_request_ddr =
      request_ddr +
      (row == 0U ? WAFER_MDC_CE_SERIAL_REQUEST_WORD
                 : WAFER_MDC_CE_WINDOW_REQUEST_WORD) *
          sizeof(uint64_t);
  uint64_t lane_a_payload =
      payload_ddr + WAFER_MDC_PAYLOAD_DATA_OFFSET;
  uint64_t lane_b_payload = lane_a_payload + 16384U;
  record[WAFER_MDC_CE_REC_MAGIC] = WAFER_MDC_CE_RECORD_MAGIC;
  record[WAFER_MDC_CE_REC_SCHEMA_AND_WORDS] =
      ((uint64_t)WAFER_MDC_CE_SCHEMA << 32) |
      WAFER_MDC_CE_RECORD_META_WORDS;
  record[WAFER_MDC_CE_REC_STATUS] = status;
  record[WAFER_MDC_CE_REC_COORDINATE] =
      request_meta[WAFER_MDC_CE_REQ_COORDINATE];
  record[WAFER_MDC_CE_REC_INNER_CASE] = request->case_id;
  record[WAFER_MDC_CE_REC_SCHEDULE] = request->schedule;
  record[WAFER_MDC_CE_REC_ISSUE_ORDER] = request->issue_order;
  record[WAFER_MDC_CE_REC_SAMPLE] = request->sample;
  record[WAFER_MDC_CE_REC_EXECUTION_ORDINAL] = ordinal;
  record[WAFER_MDC_CE_REC_FIRST_SCHEDULE] =
      request_meta[WAFER_MDC_CE_REQ_FIRST_SCHEDULE];
  record[WAFER_MDC_CE_REC_REQUEST_DDR] = request_ddr;
  record[WAFER_MDC_CE_REC_INNER_REQUEST_DDR] =
      inner_request_ddr;
  record[WAFER_MDC_CE_REC_PAYLOAD_DDR] = payload_ddr;
  record[WAFER_MDC_CE_REC_OUTPUT_DDR] = output_ddr;
  record[WAFER_MDC_CE_REC_ROW_OUTPUT_DDR] = row_output_ddr;
  record[WAFER_MDC_CE_REC_CT_INPUT0_DDR] = lane_a_payload;
  record[WAFER_MDC_CE_REC_CT_INPUT1_DDR] =
      lane_a_payload + 4096U;
  record[WAFER_MDC_CE_REC_RDMA_INPUT_DDR] =
      lane_b_payload + 8192U;
  record[WAFER_MDC_CE_REC_RESULT_A_DDR] =
      row_output_ddr + WAFER_MDC_OUTPUT_DATA_OFFSET;
  record[WAFER_MDC_CE_REC_RESULT_B_DDR] =
      row_output_ddr + WAFER_MDC_OUTPUT_DATA_OFFSET +
      request->compact_bytes;
  record[WAFER_MDC_CE_REC_SPM_A] = request->spm_a;
  record[WAFER_MDC_CE_REC_SPM_B] = request->spm_b;
  record[WAFER_MDC_CE_REC_WINDOW_FLAGS] =
      WAFER_MDC_CE_WINDOW_FLAGS;
  record[WAFER_MDC_CE_REC_RECORD_GUARD] =
      WAFER_MDC_CE_RECORD_GUARD;
}

static void wafer_mdc_execute_conflict_equivalence(
    const volatile uint64_t *wire, uint64_t request_ddr,
    uint64_t payload_ddr, uint64_t output_ddr) {
  const volatile uint64_t *meta =
      wire + WAFER_MDC_CE_REQUEST_META_WORD;
  WaferMDCRequest requests[WAFER_MDC_CE_ROWS] = {{0}};
  uint32_t status =
      wafer_mdc_ce_decode(wire, &requests[0], &requests[1]);
  uint32_t first =
      status == WAFER_MDC_CE_STATUS_OK &&
              meta[WAFER_MDC_CE_REQ_FIRST_SCHEDULE] ==
                  WAFER_MDC_SCHEDULE_WINDOW
          ? 1U
          : 0U;
  uint32_t order[WAFER_MDC_CE_ROWS] = {first, first ^ 1U};
  for (uint32_t ordinal = 0; ordinal < WAFER_MDC_CE_ROWS;
       ++ordinal) {
    uint32_t row = order[ordinal];
    uint64_t row_output_ddr =
        output_ddr +
        (row == 0U ? WAFER_MDC_CE_SERIAL_OUTPUT_OFFSET
                   : WAFER_MDC_CE_WINDOW_OUTPUT_OFFSET);
    uint32_t row_status = status;
    if (status == WAFER_MDC_CE_STATUS_OK) {
      wafer_mdc_execute_wire(
          wire + (row == 0U
                      ? WAFER_MDC_CE_SERIAL_REQUEST_WORD
                      : WAFER_MDC_CE_WINDOW_REQUEST_WORD),
          payload_ddr, row_output_ddr);
      volatile uint64_t *inner_record =
          (volatile uint64_t *)(uintptr_t)row_output_ddr;
      if (inner_record[WAFER_MDC_REC_STATUS] !=
          WAFER_MDC_STATUS_OK)
        row_status = WAFER_MDC_CE_STATUS_EXECUTE_FAILED;
    } else {
      wafer_mdc_init_record(
          (volatile uint64_t *)(uintptr_t)row_output_ddr,
          WAFER_MDC_STATUS_BAD_REQUEST);
    }
    wafer_mdc_ce_write_meta(
        request_ddr, payload_ddr, output_ddr, row_output_ddr, row,
        row_status, ordinal, &requests[row], meta);
  }
  uint32_t output_bytes =
      (WAFER_MDC_CE_RECORD_META_WORD +
       (WAFER_MDC_CE_ROWS - 1U) *
           WAFER_MDC_CE_RECORD_META_STRIDE_WORDS +
       WAFER_MDC_CE_RECORD_META_WORDS) *
      sizeof(uint64_t);
  wafer_mdc_cache_range(output_ddr, output_bytes, 0U);
}

__attribute__((visibility("hidden"))) void
wafer_tx81_instruction_family_probe(uint64_t request_ddr,
                                    uint64_t payload_ddr,
                                    uint64_t output_ddr) {
  wafer_mdc_cache_range(request_ddr, WAFER_MDC_RESOURCE_BYTES, 1U);
  wafer_mdc_cache_range(payload_ddr, WAFER_MDC_RESOURCE_BYTES, 1U);
  const volatile uint64_t *wire =
      (const volatile uint64_t *)(uintptr_t)request_ddr;
  if (wire[WAFER_MDC_CE_REQUEST_META_WORD +
           WAFER_MDC_CE_REQ_MAGIC] ==
      WAFER_MDC_CE_REQUEST_MAGIC)
    wafer_mdc_execute_conflict_equivalence(
        wire, request_ddr, payload_ddr, output_ddr);
  else
    wafer_mdc_execute_wire(wire, payload_ddr, output_ddr);
}

static uint32_t wafer_spm_ct_same_pair(
    const WaferMDCRequest *serial, const WaferMDCRequest *window,
    const volatile uint64_t *meta) {
  if (serial->kind != WAFER_MDC_KIND_ENGINE_PAIR ||
      window->kind != WAFER_MDC_KIND_ENGINE_PAIR ||
      serial->engine_a != WAFER_MDC_ENGINE_CT ||
      serial->engine_b != WAFER_MDC_ENGINE_RDMA ||
      window->engine_a != serial->engine_a ||
      window->engine_b != serial->engine_b ||
      serial->schedule != WAFER_MDC_SCHEDULE_SERIAL ||
      window->schedule != WAFER_MDC_SCHEDULE_WINDOW ||
      serial->effect != WAFER_MDC_EFFECT_NONE ||
      window->effect != serial->effect ||
      serial->relation != WAFER_MDC_RELATION_DISJOINT ||
      window->relation != serial->relation ||
      serial->oracle != WAFER_MDC_ORACLE_EXACT ||
      window->oracle != serial->oracle ||
      serial->format != Fmt_UINT8 || window->format != serial->format ||
      serial->src_ddr_offset != window->src_ddr_offset ||
      serial->dst_ddr_offset != window->dst_ddr_offset ||
      serial->spm_a != window->spm_a ||
      serial->spm_b != window->spm_b ||
      serial->spm_b - serial->spm_a != 8192U ||
      serial->inner_bytes != window->inner_bytes ||
      serial->stride0 != window->stride0 ||
      serial->stride1 != window->stride1 ||
      serial->stride2 != window->stride2 ||
      serial->iteration0 != window->iteration0 ||
      serial->iteration1 != window->iteration1 ||
      serial->iteration2 != window->iteration2 ||
      serial->compact_bytes != window->compact_bytes ||
      serial->envelope_bytes != window->envelope_bytes ||
      serial->output_bytes != window->output_bytes ||
      serial->sample != window->sample ||
      serial->worker_a != window->worker_a ||
      serial->worker_b != window->worker_b ||
      serial->rounds != window->rounds ||
      serial->buffer_count != window->buffer_count ||
      serial->issue_order != window->issue_order)
    return 0U;
  for (uint32_t engine = 0; engine < WAFER_MDC_ENGINES; ++engine)
    if (serial->expected_instructions[engine] !=
        window->expected_instructions[engine])
      return 0U;
  return meta[WAFER_SPM_CT_REQ_SERIAL_CASE] == serial->case_id &&
         meta[WAFER_SPM_CT_REQ_WINDOW_CASE] == window->case_id &&
         meta[WAFER_SPM_CT_REQ_SPM_A] == serial->spm_a &&
         meta[WAFER_SPM_CT_REQ_SPM_B] == serial->spm_b &&
         meta[WAFER_SPM_CT_REQ_TRANSFER_BYTES] ==
             serial->compact_bytes &&
         meta[WAFER_SPM_CT_REQ_ISSUE_ORDER] ==
             serial->issue_order &&
         meta[WAFER_SPM_CT_REQ_SAMPLE] == serial->sample;
}

static uint32_t wafer_spm_ct_decode(
    uint32_t rank, const volatile uint64_t *wire, WaferMDCRequest *serial,
    WaferMDCRequest *window) {
  const volatile uint64_t *meta =
      wire + WAFER_SPM_CT_REQUEST_META_WORD;
  if (meta[WAFER_SPM_CT_REQ_MAGIC] != WAFER_SPM_CT_REQUEST_MAGIC ||
      meta[WAFER_SPM_CT_REQ_SCHEMA_AND_WORDS] !=
          (((uint64_t)WAFER_SPM_CT_SCHEMA << 32) |
           WAFER_SPM_CT_REQUEST_META_WORDS) ||
      meta[WAFER_SPM_CT_REQ_RANK] != rank ||
      meta[WAFER_SPM_CT_REQ_RANK_COUNT] != WAFER_SPM_CT_RANKS ||
      meta[WAFER_SPM_CT_REQ_SERIAL_REQUEST_WORD] !=
          WAFER_SPM_CT_SERIAL_REQUEST_WORD ||
      meta[WAFER_SPM_CT_REQ_WINDOW_REQUEST_WORD] !=
          WAFER_SPM_CT_WINDOW_REQUEST_WORD ||
      meta[WAFER_SPM_CT_REQ_RESOURCE_BYTES] !=
          WAFER_MDC_RESOURCE_BYTES ||
      meta[WAFER_SPM_CT_REQ_GUARD] != WAFER_SPM_CT_REQUEST_GUARD ||
      meta[WAFER_SPM_CT_REQ_EXECUTION_ROUND] >=
          WAFER_SPM_CT_COUNTERBALANCED_ROUNDS ||
      meta[WAFER_SPM_CT_REQ_RANK_ORDER] !=
          (meta[WAFER_SPM_CT_REQ_EXECUTION_ROUND] & 1U) ||
      meta[WAFER_SPM_CT_REQ_RANK_PHASE] !=
          (meta[WAFER_SPM_CT_REQ_RANK_ORDER] ==
                   WAFER_SPM_CT_RANK_ORDER_FORWARD
               ? rank
               : WAFER_SPM_CT_RANKS - 1U - rank) ||
      meta[WAFER_SPM_CT_REQ_FIRST_SCHEDULE] !=
          ((rank + meta[WAFER_SPM_CT_REQ_EXECUTION_ROUND]) & 1U))
    return WAFER_SPM_CT_STATUS_BAD_ENVELOPE;
  if (wafer_mdc_decode(
          wire + WAFER_SPM_CT_SERIAL_REQUEST_WORD, serial) !=
          WAFER_MDC_STATUS_OK ||
      wafer_mdc_decode(
          wire + WAFER_SPM_CT_WINDOW_REQUEST_WORD, window) !=
          WAFER_MDC_STATUS_OK ||
      !wafer_spm_ct_same_pair(serial, window, meta))
    return WAFER_SPM_CT_STATUS_BAD_PAIR;
  return WAFER_SPM_CT_STATUS_OK;
}

static void wafer_spm_ct_write_meta(
    uint64_t output_ddr, uint32_t status, uint32_t rank,
    uint32_t coordinate, const WaferMDCRequest *request,
    uint32_t schedule, uint64_t request_ddr, uint64_t payload_ddr,
    const volatile uint64_t *request_meta,
    uint32_t execution_ordinal) {
  volatile uint64_t *meta =
      (volatile uint64_t *)(uintptr_t)output_ddr +
      WAFER_SPM_CT_RECORD_META_WORD;
  for (uint32_t word = 0; word < WAFER_SPM_CT_RECORD_META_WORDS; ++word)
    meta[word] = 0U;
  meta[WAFER_SPM_CT_REC_MAGIC] = WAFER_SPM_CT_RECORD_MAGIC;
  meta[WAFER_SPM_CT_REC_SCHEMA_AND_WORDS] =
      ((uint64_t)WAFER_SPM_CT_SCHEMA << 32) |
      WAFER_SPM_CT_RECORD_META_WORDS;
  meta[WAFER_SPM_CT_REC_STATUS] = status;
  meta[WAFER_SPM_CT_REC_RANK] = rank;
  meta[WAFER_SPM_CT_REC_RANK_COUNT] = WAFER_SPM_CT_RANKS;
  meta[WAFER_SPM_CT_REC_COORDINATE] = coordinate;
  meta[WAFER_SPM_CT_REC_INNER_CASE] = request->case_id;
  meta[WAFER_SPM_CT_REC_SCHEDULE] = schedule;
  meta[WAFER_SPM_CT_REC_REQUEST_DDR] = request_ddr;
  meta[WAFER_SPM_CT_REC_PAYLOAD_DDR] = payload_ddr;
  meta[WAFER_SPM_CT_REC_OUTPUT_DDR] = output_ddr;
  meta[WAFER_SPM_CT_REC_SPM_A] = request->spm_a;
  meta[WAFER_SPM_CT_REC_SPM_B] = request->spm_b;
  meta[WAFER_SPM_CT_REC_SAMPLE] = request->sample;
  meta[WAFER_SPM_CT_REC_REQUEST_GUARD] =
      WAFER_SPM_CT_REQUEST_GUARD;
  meta[WAFER_SPM_CT_REC_RECORD_GUARD] = WAFER_SPM_CT_RECORD_GUARD;
  meta[WAFER_SPM_CT_REC_EXECUTION_ROUND] =
      request_meta[WAFER_SPM_CT_REQ_EXECUTION_ROUND];
  meta[WAFER_SPM_CT_REC_RANK_ORDER] =
      request_meta[WAFER_SPM_CT_REQ_RANK_ORDER];
  meta[WAFER_SPM_CT_REC_RANK_PHASE] =
      request_meta[WAFER_SPM_CT_REQ_RANK_PHASE];
  meta[WAFER_SPM_CT_REC_FIRST_SCHEDULE] =
      request_meta[WAFER_SPM_CT_REQ_FIRST_SCHEDULE];
  meta[WAFER_SPM_CT_REC_EXECUTION_ORDINAL] =
      execution_ordinal;
}

__attribute__((visibility("hidden"))) void
wafer_tx81_spm_cross_tile_conflict_probe(
    uint32_t rank, uint64_t request_ddr, uint64_t payload_ddr,
    uint64_t serial_output_ddr, uint64_t window_output_ddr,
    uint64_t status_ddr) {
  wafer_tx81_direct_dte_begin_after_prepare(
      status_ddr, WAFER_SPM_CT_RANKS);
  wafer_mdc_cache_range(request_ddr, WAFER_MDC_RESOURCE_BYTES, 1U);
  wafer_mdc_cache_range(payload_ddr, WAFER_MDC_RESOURCE_BYTES, 1U);
  const volatile uint64_t *wire =
      (const volatile uint64_t *)(uintptr_t)request_ddr;
  const volatile uint64_t *request_meta =
      wire + WAFER_SPM_CT_REQUEST_META_WORD;
  WaferMDCRequest serial = {0};
  WaferMDCRequest window = {0};
  uint32_t status =
      wafer_spm_ct_decode(rank, wire, &serial, &window);
  uint32_t rank_order =
      request_meta[WAFER_SPM_CT_REQ_RANK_ORDER] ==
              WAFER_SPM_CT_RANK_ORDER_REVERSE
          ? WAFER_SPM_CT_RANK_ORDER_REVERSE
          : WAFER_SPM_CT_RANK_ORDER_FORWARD;
  uint32_t first_schedule =
      request_meta[WAFER_SPM_CT_REQ_FIRST_SCHEDULE] ==
              WAFER_MDC_SCHEDULE_WINDOW
          ? WAFER_MDC_SCHEDULE_WINDOW
          : WAFER_MDC_SCHEDULE_SERIAL;

  for (uint32_t phase = 0; phase < WAFER_SPM_CT_RANKS; ++phase) {
    uint32_t active_rank =
        rank_order == WAFER_SPM_CT_RANK_ORDER_FORWARD
            ? phase
            : WAFER_SPM_CT_RANKS - 1U - phase;
    hrt_barrier();
    if (rank == active_rank) {
      if (status == WAFER_SPM_CT_STATUS_OK) {
        if (first_schedule == WAFER_MDC_SCHEDULE_SERIAL) {
          wafer_mdc_execute_wire(
              wire + WAFER_SPM_CT_SERIAL_REQUEST_WORD, payload_ddr,
              serial_output_ddr);
          wafer_mdc_execute_wire(
              wire + WAFER_SPM_CT_WINDOW_REQUEST_WORD, payload_ddr,
              window_output_ddr);
        } else {
          wafer_mdc_execute_wire(
              wire + WAFER_SPM_CT_WINDOW_REQUEST_WORD, payload_ddr,
              window_output_ddr);
          wafer_mdc_execute_wire(
              wire + WAFER_SPM_CT_SERIAL_REQUEST_WORD, payload_ddr,
              serial_output_ddr);
        }
      } else {
        wafer_mdc_init_record(
            (volatile uint64_t *)(uintptr_t)serial_output_ddr,
            WAFER_MDC_STATUS_BAD_REQUEST);
        wafer_mdc_init_record(
            (volatile uint64_t *)(uintptr_t)window_output_ddr,
            WAFER_MDC_STATUS_BAD_REQUEST);
      }
      uint32_t coordinate =
          (uint32_t)request_meta[WAFER_SPM_CT_REQ_COORDINATE];
      wafer_spm_ct_write_meta(
          serial_output_ddr, status, rank, coordinate, &serial,
          WAFER_MDC_SCHEDULE_SERIAL, request_ddr, payload_ddr,
          request_meta,
          first_schedule == WAFER_MDC_SCHEDULE_SERIAL ? 0U : 1U);
      wafer_spm_ct_write_meta(
          window_output_ddr, status, rank, coordinate, &window,
          WAFER_MDC_SCHEDULE_WINDOW, request_ddr, payload_ddr,
          request_meta,
          first_schedule == WAFER_MDC_SCHEDULE_WINDOW ? 0U : 1U);
      wafer_mdc_cache_range(
          serial_output_ddr,
          (WAFER_SPM_CT_RECORD_META_WORD +
           WAFER_SPM_CT_RECORD_META_WORDS) *
              sizeof(uint64_t),
          0U);
      wafer_mdc_cache_range(
          window_output_ddr,
          (WAFER_SPM_CT_RECORD_META_WORD +
           WAFER_SPM_CT_RECORD_META_WORDS) *
              sizeof(uint64_t),
          0U);
    }
    hrt_barrier();
  }
  wafer_tx81_direct_dte_finish();
}
