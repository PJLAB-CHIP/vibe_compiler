#include "Wafer/ABI/Tx81DirectDTEStatusABI.h"
#include "instr_adapter.h"
#include "instr_adapter_plat.h"
#include "instr_def.h"
#include "wafer_tx81_crt.h"
#include "wafer_worker_placement_probe_protocol.h"

#include <stddef.h>
#include <stdint.h>

#define WAFER_WP_PMU_BASE UINT64_C(0x590000)
#define WAFER_WP_PMU_WORKER_STRIDE UINT32_C(0x30)
#define WAFER_WP_STABLE_RETRIES 8U
#define WAFER_WP_TASK_DONE UINT64_C(0x100)
#define WAFER_WP_NE_M 64U
#define WAFER_WP_NE_K 128U
#define WAFER_WP_NE_N 128U
#define WAFER_WP_NE_LHS_BYTES (WAFER_WP_NE_M * WAFER_WP_NE_K * 2U)
#define WAFER_WP_NE_RHS_BYTES (WAFER_WP_NE_K * WAFER_WP_NE_N * 2U)
#define WAFER_WP_F16_ONE UINT16_C(0x3c00)
#define WAFER_WP_F16_TWO UINT16_C(0x4000)
#define WAFER_WP_F16_THREE UINT16_C(0x4200)
#define WAFER_WP_CANARY UINT8_C(0xa5)

extern int8_t *get_spm_memory_mapping(uint64_t offset);

typedef union WaferWPPacket {
  TsmArithInstr ct;
  TsmNeInstr ne;
  TsmRdmaInstr rdma;
} WaferWPPacket;

typedef struct WaferWPInstruction {
  uint32_t engine;
  uint32_t worker;
  uint32_t output_bytes;
  uint32_t slot;
  WaferWPPacket packet;
} WaferWPInstruction;

typedef struct WaferWPRequest {
  uint32_t kind;
  uint32_t engine;
  uint32_t worker_mask;
  uint32_t target_worker;
  uint32_t observer_worker;
  uint32_t sample;
  uint32_t worker_issues[WAFER_WP_WORKERS];
  uint32_t primary_bytes;
  uint32_t sentinel_bytes;
  uint32_t issue_rotation;
  uint32_t join_rotation;
} WaferWPRequest;

typedef struct WaferWPContext {
  WaferWPRequest request;
  WaferWPInstruction instructions[WAFER_WP_MAX_ISSUES];
  uint32_t issue_count;
  uint64_t payload_ddr;
  uint64_t output_ddr;
} WaferWPContext;

static WaferWPContext wafer_wp_context;

static const uint32_t wafer_wp_pmu64_offsets[WAFER_WP_PMU64_COUNTERS] = {
    GR_PMU_STATISTICS_WINDOW, GR_PMU_FU_EXE_TIME,   GR_PMU_CT_EXE_TIME,
    GR_PMU_NE_EXE_TIME,      GR_PMU_RDMA_EXE_TIME, GR_PMU_WDMA_EXE_TIME,
    GR_PMU_TDMA_EXE_TIME,    GR_PMU_SCALAR_EXE_TIME,
};

static const uint32_t wafer_wp_instruction_offsets[WAFER_WP_QUEUES] = {
    GR_PMU_CT_INST_NUMS, GR_PMU_NE_INST_NUMS, GR_PMU_RDMA_INST_NUMS,
    GR_PMU_WDMA_INST_NUMS, GR_PMU_TDMA_INST_NUMS,
};

static const uint32_t wafer_wp_blocking_offsets[WAFER_WP_QUEUES] = {
    GR_PMU_CT_BLOCKING_TIME, GR_PMU_NE_BLOCKING_TIME,
    GR_PMU_RDMA_BLOCKING_TIME, GR_PMU_WDMA_BLOCKING_TIME,
    GR_PMU_TDMA_BLOCKING_TIME,
};

static void wafer_wp_zero(void *pointer, uint32_t bytes) {
  uint8_t *output = (uint8_t *)pointer;
  for (uint32_t index = 0; index < bytes; ++index)
    output[index] = 0;
}

static void wafer_wp_fill8(volatile uint8_t *output, uint8_t value,
                           uint32_t bytes) {
  for (uint32_t index = 0; index < bytes; ++index)
    output[index] = value;
}

static void wafer_wp_fill16(volatile uint16_t *output, uint16_t value,
                            uint32_t elements) {
  for (uint32_t index = 0; index < elements; ++index)
    output[index] = value;
}

static uint64_t wafer_wp_cycle(void) {
  uint64_t cycle;
  __asm__ volatile("rdcycle %0" : "=r"(cycle));
  return cycle;
}

static uint32_t wafer_wp_read_pmu32(uint32_t offset) {
  return *(const volatile uint32_t *)(uintptr_t)(WAFER_WP_PMU_BASE + offset);
}

static uint64_t wafer_wp_read_pmu64(uint32_t offset, uint64_t bit,
                                    uint64_t *stable_mask) {
  uint32_t low = 0;
  uint32_t high_after = 0;
  for (uint32_t retry = 0; retry < WAFER_WP_STABLE_RETRIES; ++retry) {
    uint32_t high_before = wafer_wp_read_pmu32(offset + 4U);
    low = wafer_wp_read_pmu32(offset);
    high_after = wafer_wp_read_pmu32(offset + 4U);
    if (high_before == high_after) {
      *stable_mask |= bit;
      break;
    }
  }
  return ((uint64_t)high_after << 32) | low;
}

static void wafer_wp_snapshot(volatile uint64_t *record, uint32_t pmu64_base,
                              uint32_t instruction_base,
                              uint32_t blocking_base,
                              uint32_t stable_index) {
  uint64_t stable = 0;
  for (uint32_t index = 0; index < WAFER_WP_PMU64_COUNTERS; ++index)
    record[pmu64_base + index] = wafer_wp_read_pmu64(
        wafer_wp_pmu64_offsets[index], UINT64_C(1) << index, &stable);
  for (uint32_t worker = 0; worker < WAFER_WP_WORKERS; ++worker) {
    uint32_t worker_offset = worker * WAFER_WP_PMU_WORKER_STRIDE;
    for (uint32_t queue = 0; queue < WAFER_WP_QUEUES; ++queue) {
      uint32_t index = worker * WAFER_WP_QUEUES + queue;
      record[instruction_base + index] = wafer_wp_read_pmu32(
          wafer_wp_instruction_offsets[queue] + worker_offset);
      record[blocking_base + index] = wafer_wp_read_pmu32(
          wafer_wp_blocking_offsets[queue] + worker_offset);
    }
  }
  record[stable_index] = stable;
}

static void wafer_wp_controls(volatile uint64_t *record, uint32_t base) {
  for (uint32_t worker = 0; worker < WAFER_WP_WORKERS; ++worker)
    record[base + worker] = get_ncc_reg(worker, GR_CSR_CONTROL_ADDR);
}

static void wafer_wp_cache_range(uint64_t begin, uint32_t bytes,
                                 uint32_t invalidate) {
  enum {
    WAFER_WP_SUPERVISOR_MODE = 1,
    WAFER_WP_MACHINE_MODE = 3,
  };
  uintptr_t mode;
  __asm__ volatile("fence" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
  __asm__ volatile("csrr %0, mxstatus" : "=r"(mode));
  mode = (mode >> 30) & 3U;
  for (uintptr_t address = (uintptr_t)begin;
       address < (uintptr_t)begin + bytes;
       address += WAFER_TX81_DIRECT_DTE_STATUS_V2_CACHE_LINE_BYTES) {
    if (mode == WAFER_WP_MACHINE_MODE) {
      if (invalidate)
        __asm__ volatile("dcache.ipa %0" : : "r"(address) : "memory");
      else
        __asm__ volatile("dcache.cipa %0" : : "r"(address) : "memory");
    } else if (mode == WAFER_WP_SUPERVISOR_MODE) {
      if (invalidate)
        __asm__ volatile("dcache.iva %0" : : "r"(address) : "memory");
      else
        __asm__ volatile("dcache.civa %0" : : "r"(address) : "memory");
    }
  }
  __asm__ volatile("sync.is" ::: "memory");
  __asm__ volatile("fence" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
}

static void wafer_wp_publish_record(volatile uint64_t *record) {
  wafer_wp_cache_range((uint64_t)(uintptr_t)record,
                       WAFER_WP_RECORD_WORDS * sizeof(uint64_t), 0U);
}

static int wafer_wp_decode_u32(const volatile uint64_t *words,
                               uint32_t index, uint32_t *value) {
  if (words[index] > UINT32_MAX)
    return 0;
  *value = (uint32_t)words[index];
  return 1;
}

static uint32_t wafer_wp_mask_from_counts(const WaferWPRequest *request) {
  uint32_t mask = 0;
  for (uint32_t worker = 0; worker < WAFER_WP_WORKERS; ++worker)
    if (request->worker_issues[worker] != 0)
      mask |= UINT32_C(1) << worker;
  return mask;
}

static int wafer_wp_placement_counts_are_valid(
    const WaferWPRequest *request) {
  uint32_t nonzero = 0;
  uint32_t sum = 0;
  uint32_t expected = 0;
  for (uint32_t worker = 0; worker < WAFER_WP_WORKERS; ++worker) {
    sum += request->worker_issues[worker];
    nonzero += request->worker_issues[worker] != 0;
  }
  if (nonzero == 1U)
    expected = 6U;
  else if (nonzero == 2U)
    expected = 3U;
  else if (nonzero == 3U)
    expected = 2U;
  if (sum != WAFER_WP_PLACEMENT_ISSUES || expected == 0U)
    return 0;
  for (uint32_t worker = 0; worker < WAFER_WP_WORKERS; ++worker)
    if (request->worker_issues[worker] != 0 &&
        request->worker_issues[worker] != expected)
      return 0;
  return 1;
}

static uint32_t wafer_wp_decode_request(const volatile uint64_t *words,
                                        WaferWPRequest *request) {
  if (words[WAFER_WP_REQ_MAGIC] != WAFER_WP_REQUEST_MAGIC ||
      words[WAFER_WP_REQ_SCHEMA_AND_WORDS] !=
          (((uint64_t)WAFER_WP_SCHEMA << 32) | WAFER_WP_REQUEST_WORDS) ||
      words[WAFER_WP_REQ_GUARD] != WAFER_WP_REQUEST_GUARD)
    return WAFER_WP_STATUS_BAD_REQUEST;
  for (uint32_t index = WAFER_WP_REQ_RESERVED_BASE;
       index < WAFER_WP_REQUEST_WORDS; ++index)
    if (words[index] != 0)
      return WAFER_WP_STATUS_BAD_REQUEST;
  uint32_t *fields[] = {
      &request->kind,
      &request->engine,
      &request->worker_mask,
      &request->target_worker,
      &request->observer_worker,
      &request->sample,
      &request->worker_issues[0],
      &request->worker_issues[1],
      &request->worker_issues[2],
      &request->primary_bytes,
      &request->sentinel_bytes,
      &request->issue_rotation,
      &request->join_rotation,
  };
  const uint32_t indices[] = {
      WAFER_WP_REQ_KIND,
      WAFER_WP_REQ_ENGINE,
      WAFER_WP_REQ_WORKER_MASK,
      WAFER_WP_REQ_TARGET_WORKER,
      WAFER_WP_REQ_OBSERVER_WORKER,
      WAFER_WP_REQ_SAMPLE,
      WAFER_WP_REQ_WORKER0_ISSUES,
      WAFER_WP_REQ_WORKER1_ISSUES,
      WAFER_WP_REQ_WORKER2_ISSUES,
      WAFER_WP_REQ_PRIMARY_BYTES,
      WAFER_WP_REQ_SENTINEL_BYTES,
      WAFER_WP_REQ_ISSUE_ROTATION,
      WAFER_WP_REQ_JOIN_ROTATION,
  };
  for (uint32_t field = 0;
       field < sizeof(fields) / sizeof(fields[0]); ++field)
    if (!wafer_wp_decode_u32(words, indices[field], fields[field]))
      return WAFER_WP_STATUS_BAD_REQUEST;
  if (words[WAFER_WP_REQ_RESOURCE_BYTES] != WAFER_WP_RESOURCE_BYTES ||
      request->kind > WAFER_WP_KIND_CONCURRENT ||
      request->engine > WAFER_WP_ENGINE_RDMA ||
      request->target_worker >= WAFER_WP_WORKERS ||
      request->observer_worker >= WAFER_WP_WORKERS ||
      request->target_worker == request->observer_worker ||
      request->issue_rotation != request->sample % WAFER_WP_WORKERS ||
      request->join_rotation !=
          (request->sample + 1U) % WAFER_WP_WORKERS ||
      request->worker_mask != wafer_wp_mask_from_counts(request))
    return WAFER_WP_STATUS_BAD_REQUEST;
  if (request->kind == WAFER_WP_KIND_PLACEMENT)
    return (request->engine == WAFER_WP_ENGINE_CT ||
            request->engine == WAFER_WP_ENGINE_RDMA) &&
                   request->primary_bytes == WAFER_WP_PLACEMENT_BYTES &&
                   request->sentinel_bytes == 0U &&
                   wafer_wp_placement_counts_are_valid(request)
               ? WAFER_WP_STATUS_OK
               : WAFER_WP_STATUS_BAD_REQUEST;

  if ((request->engine != WAFER_WP_ENGINE_NE &&
       request->engine != WAFER_WP_ENGINE_RDMA) ||
      request->sentinel_bytes != WAFER_WP_SENTINEL_BYTES ||
      request->primary_bytes !=
          (request->engine == WAFER_WP_ENGINE_NE
               ? WAFER_WP_NE_RESULT_BYTES
               : WAFER_WP_RDMA_BACKLOG_BYTES))
    return WAFER_WP_STATUS_BAD_REQUEST;
  uint32_t target = request->worker_issues[request->target_worker];
  uint32_t observer = request->worker_issues[request->observer_worker];
  for (uint32_t worker = 0; worker < WAFER_WP_WORKERS; ++worker)
    if (worker != request->target_worker &&
        worker != request->observer_worker &&
        request->worker_issues[worker] != 0)
      return WAFER_WP_STATUS_BAD_REQUEST;
  if (request->kind == WAFER_WP_KIND_BACKLOG_ONLY)
    return target == WAFER_WP_BACKLOG_ISSUES && observer == 0U
               ? WAFER_WP_STATUS_OK
               : WAFER_WP_STATUS_BAD_REQUEST;
  if (request->kind == WAFER_WP_KIND_SENTINEL_ONLY)
    return target == 0U && observer == 1U ? WAFER_WP_STATUS_OK
                                         : WAFER_WP_STATUS_BAD_REQUEST;
  return target == WAFER_WP_BACKLOG_ISSUES && observer == 1U
             ? WAFER_WP_STATUS_OK
             : WAFER_WP_STATUS_BAD_REQUEST;
}

static volatile uint8_t *wafer_wp_spm8(uint64_t offset) {
  return (volatile uint8_t *)(void *)get_spm_memory_mapping(offset);
}

static volatile uint16_t *wafer_wp_spm16(uint64_t offset) {
  return (volatile uint16_t *)(void *)get_spm_memory_mapping(offset);
}

static uint64_t wafer_wp_slot(uint32_t slot) {
  return WAFER_WP_SPM_SLOT_BASE +
         (uint64_t)slot * WAFER_WP_SPM_SLOT_STRIDE;
}

static uint64_t wafer_wp_read0(uint32_t slot) {
  return wafer_wp_slot(slot) + WAFER_WP_SPM_READ0_OFFSET;
}

static uint64_t wafer_wp_read1(uint32_t slot) {
  return wafer_wp_slot(slot) + WAFER_WP_SPM_READ1_OFFSET;
}

static uint64_t wafer_wp_write(uint32_t slot) {
  return wafer_wp_slot(slot) + WAFER_WP_SPM_WRITE_OFFSET;
}

static uint64_t wafer_wp_payload(const WaferWPContext *context,
                                 uint32_t slot) {
  return context->payload_ddr +
         (uint64_t)slot * WAFER_WP_OUTPUT_SLOT_STRIDE +
         WAFER_WP_GUARD_BYTES;
}

static uint64_t wafer_wp_archive(const WaferWPContext *context,
                                 uint32_t slot) {
  return context->output_ddr + WAFER_WP_OUTPUT_SLOT_BASE +
         (uint64_t)slot * WAFER_WP_OUTPUT_SLOT_STRIDE;
}

static uint8_t wafer_wp_pattern(const WaferWPRequest *request, uint32_t slot,
                                uint32_t index) {
  return (uint8_t)(UINT32_C(0x51) + request->sample * 13U + slot * 29U +
                   index * 17U + (index >> 8) * 7U);
}

static void wafer_wp_set_worker(uint32_t *inter_type, uint32_t worker) {
  *inter_type = (*inter_type & ~UINT32_C(0x300)) | (worker << 8);
}

static uint32_t wafer_wp_packet_inter_type(
    const WaferWPInstruction *instruction) {
  switch (instruction->engine) {
  case WAFER_WP_ENGINE_CT:
    return instruction->packet.ct.inter_type;
  case WAFER_WP_ENGINE_NE:
    return instruction->packet.ne.inter_type;
  case WAFER_WP_ENGINE_RDMA:
    return instruction->packet.rdma.inter_type;
  default:
    return 0U;
  }
}

static void *wafer_wp_packet(WaferWPInstruction *instruction) {
  switch (instruction->engine) {
  case WAFER_WP_ENGINE_CT:
    return &instruction->packet.ct;
  case WAFER_WP_ENGINE_NE:
    return &instruction->packet.ne;
  case WAFER_WP_ENGINE_RDMA:
    return &instruction->packet.rdma;
  default:
    return NULL;
  }
}

static uint32_t wafer_wp_prepare_ct(WaferWPInstruction *instruction) {
  TsmArith *arith = TsmNewArith();
  if (arith == NULL)
    return 0U;
  arith->AddVV(&instruction->packet.ct, wafer_wp_read0(instruction->slot),
               wafer_wp_read1(instruction->slot),
               wafer_wp_write(instruction->slot),
               instruction->output_bytes / sizeof(uint16_t),
               RND_NEAREST_EVEN, Fmt_FP16);
  wafer_wp_set_worker(&instruction->packet.ct.inter_type,
                      instruction->worker);
  TsmDeleteArith(arith);
  return 1U;
}

static uint32_t wafer_wp_prepare_ne(WaferWPInstruction *instruction) {
  TsmGemm *gemm = TsmNewGemm();
  if (gemm == NULL)
    return 0U;
  gemm->AddInput(&instruction->packet.ne,
                 wafer_wp_read0(instruction->slot),
                 wafer_wp_read1(instruction->slot), Fmt_FP16);
  gemm->ConfigMKN(&instruction->packet.ne, WAFER_WP_NE_M, WAFER_WP_NE_K,
                  WAFER_WP_NE_N);
  gemm->ConfigBatch(&instruction->packet.ne, 1, 1);
  gemm->SetTransflag(&instruction->packet.ne, 0, 1);
  gemm->SetPsum(&instruction->packet.ne, 0, 0, Fmt_UNUSED);
  gemm->SetQuant(&instruction->packet.ne, 0, 0, 0, 0);
  gemm->AddBias(&instruction->packet.ne, 0, 0);
  gemm->SetNegativeAxisScale(&instruction->packet.ne, 0, 0);
  gemm->SetPositiveAxisScale(&instruction->packet.ne, 0, 0);
  gemm->DisableRelu(&instruction->packet.ne);
  gemm->DisableLeakyRelu(&instruction->packet.ne);
  gemm->AddOutput(&instruction->packet.ne,
                  wafer_wp_write(instruction->slot), Fmt_FP16);
  wafer_wp_set_worker(&instruction->packet.ne.inter_type,
                      instruction->worker);
  TsmDeleteGemm(gemm);
  return 1U;
}

static uint32_t wafer_wp_prepare_rdma(WaferWPContext *context,
                                      WaferWPInstruction *instruction) {
  TsmRdma *rdma = TsmNewRdma();
  if (rdma == NULL)
    return 0U;
  rdma->AddSrcDst(&instruction->packet.rdma,
                  wafer_wp_payload(context, instruction->slot),
                  wafer_wp_write(instruction->slot), Fmt_UINT8);
  rdma->ConfigStrideIteration(&instruction->packet.rdma,
                              instruction->output_bytes, 0, 1, 0, 1, 0, 1);
  wafer_wp_set_worker(&instruction->packet.rdma.inter_type,
                      instruction->worker);
  TsmDeleteRdma(rdma);
  return 1U;
}

static uint32_t wafer_wp_seed_and_prepare(
    WaferWPContext *context, WaferWPInstruction *instruction) {
  uint64_t write = wafer_wp_write(instruction->slot);
  wafer_wp_fill8(wafer_wp_spm8(write - WAFER_WP_GUARD_BYTES),
                 WAFER_WP_CANARY,
                 instruction->output_bytes + 2U * WAFER_WP_GUARD_BYTES);
  if (instruction->engine == WAFER_WP_ENGINE_CT) {
    wafer_wp_fill16(wafer_wp_spm16(wafer_wp_read0(instruction->slot)),
                    WAFER_WP_F16_ONE,
                    instruction->output_bytes / sizeof(uint16_t));
    wafer_wp_fill16(wafer_wp_spm16(wafer_wp_read1(instruction->slot)),
                    WAFER_WP_F16_TWO,
                    instruction->output_bytes / sizeof(uint16_t));
    return wafer_wp_prepare_ct(instruction);
  }
  if (instruction->engine == WAFER_WP_ENGINE_NE) {
    wafer_wp_fill16(wafer_wp_spm16(wafer_wp_read0(instruction->slot)),
                    WAFER_WP_F16_ONE,
                    WAFER_WP_NE_LHS_BYTES / sizeof(uint16_t));
    wafer_wp_fill16(wafer_wp_spm16(wafer_wp_read1(instruction->slot)), 0,
                    WAFER_WP_NE_RHS_BYTES / sizeof(uint16_t));
    volatile uint16_t *rhs =
        wafer_wp_spm16(wafer_wp_read1(instruction->slot));
    for (uint32_t index = 0; index < WAFER_WP_NE_N; ++index)
      rhs[index * WAFER_WP_NE_N + index] = WAFER_WP_F16_ONE;
    return wafer_wp_prepare_ne(instruction);
  }
  return instruction->engine == WAFER_WP_ENGINE_RDMA
             ? wafer_wp_prepare_rdma(context, instruction)
             : 0U;
}

static uint32_t wafer_wp_append_issue(WaferWPContext *context,
                                      uint32_t worker, uint32_t engine,
                                      uint32_t bytes) {
  if (context->issue_count >= WAFER_WP_MAX_ISSUES)
    return 0U;
  WaferWPInstruction *instruction =
      &context->instructions[context->issue_count];
  instruction->slot = context->issue_count;
  instruction->worker = worker;
  instruction->engine = engine;
  instruction->output_bytes = bytes;
  ++context->issue_count;
  return 1U;
}

static uint32_t wafer_wp_build_issues(WaferWPContext *context) {
  const WaferWPRequest *request = &context->request;
  if (request->kind == WAFER_WP_KIND_PLACEMENT) {
    uint32_t remaining[WAFER_WP_WORKERS] = {
        request->worker_issues[0], request->worker_issues[1],
        request->worker_issues[2]};
    uint32_t total = WAFER_WP_PLACEMENT_ISSUES;
    while (total != 0U) {
      uint32_t progressed = 0U;
      for (uint32_t offset = 0; offset < WAFER_WP_WORKERS; ++offset) {
        uint32_t worker =
            (request->issue_rotation + offset) % WAFER_WP_WORKERS;
        if (remaining[worker] == 0U)
          continue;
        if (!wafer_wp_append_issue(context, worker, request->engine,
                                   request->primary_bytes))
          return 0U;
        --remaining[worker];
        --total;
        progressed = 1U;
      }
      if (!progressed)
        return 0U;
    }
    return 1U;
  }
  if (request->kind != WAFER_WP_KIND_SENTINEL_ONLY)
    for (uint32_t index = 0; index < WAFER_WP_BACKLOG_ISSUES; ++index)
      if (!wafer_wp_append_issue(context, request->target_worker,
                                 request->engine,
                                 request->primary_bytes))
        return 0U;
  if (request->kind != WAFER_WP_KIND_BACKLOG_ONLY &&
      !wafer_wp_append_issue(context, request->observer_worker,
                             WAFER_WP_ENGINE_CT,
                             request->sentinel_bytes))
    return 0U;
  return 1U;
}

static uint64_t wafer_wp_mismatch(const WaferWPContext *context,
                                  const WaferWPInstruction *instruction) {
  const volatile uint8_t *actual =
      wafer_wp_spm8(wafer_wp_write(instruction->slot));
  uint64_t mismatches = 0;
  if (instruction->engine == WAFER_WP_ENGINE_RDMA) {
    for (uint32_t index = 0; index < instruction->output_bytes; ++index)
      mismatches += actual[index] != wafer_wp_pattern(
          &context->request, instruction->slot, index);
    return mismatches;
  }
  uint16_t expected = instruction->engine == WAFER_WP_ENGINE_NE
                          ? WAFER_WP_F16_ONE
                          : WAFER_WP_F16_THREE;
  const volatile uint16_t *actual16 =
      (const volatile uint16_t *)(const void *)actual;
  for (uint32_t index = 0;
       index < instruction->output_bytes / sizeof(uint16_t); ++index)
    mismatches += actual16[index] != expected;
  return mismatches;
}

static uint64_t wafer_wp_guard_mismatch(
    const WaferWPInstruction *instruction) {
  uint64_t mismatches = 0;
  const volatile uint8_t *before = wafer_wp_spm8(
      wafer_wp_write(instruction->slot) - WAFER_WP_GUARD_BYTES);
  const volatile uint8_t *after = wafer_wp_spm8(
      wafer_wp_write(instruction->slot) + instruction->output_bytes);
  for (uint32_t index = 0; index < WAFER_WP_GUARD_BYTES; ++index) {
    mismatches += before[index] != WAFER_WP_CANARY;
    mismatches += after[index] != WAFER_WP_CANARY;
  }
  return mismatches;
}

static void wafer_wp_wait_worker(volatile uint64_t *record, uint32_t worker,
                                 uint32_t cycle_index,
                                 uint32_t duration_index) {
  uint64_t before = wafer_wp_cycle();
  (void)TsmWaitfinish_bywork(worker);
  uint64_t after = wafer_wp_cycle();
  record[cycle_index + worker] = after - before;
  if (duration_index != UINT32_MAX)
    record[duration_index] += after - before;
}

static void wafer_wp_echo_record(volatile uint64_t *record,
                                 const WaferWPRequest *request) {
  record[WAFER_WP_REC_MAGIC] = WAFER_WP_RECORD_MAGIC;
  record[WAFER_WP_REC_SCHEMA_AND_WORDS] =
      ((uint64_t)WAFER_WP_SCHEMA << 32) | WAFER_WP_RECORD_WORDS;
  record[WAFER_WP_REC_KIND] = request->kind;
  record[WAFER_WP_REC_ENGINE] = request->engine;
  record[WAFER_WP_REC_WORKER_MASK] = request->worker_mask;
  record[WAFER_WP_REC_TARGET_WORKER] = request->target_worker;
  record[WAFER_WP_REC_OBSERVER_WORKER] = request->observer_worker;
  record[WAFER_WP_REC_SAMPLE] = request->sample;
  record[WAFER_WP_REC_PRIMARY_BYTES] = request->primary_bytes;
  record[WAFER_WP_REC_SENTINEL_BYTES] = request->sentinel_bytes;
  record[WAFER_WP_REC_REQUEST_GUARD] = WAFER_WP_REQUEST_GUARD;
  record[WAFER_WP_REC_RECORD_GUARD] = WAFER_WP_RECORD_GUARD;
  record[WAFER_WP_REC_RESOURCE_BYTES] = WAFER_WP_RESOURCE_BYTES;
  record[WAFER_WP_REC_OUTPUT_SLOT_BASE] = WAFER_WP_OUTPUT_SLOT_BASE;
  record[WAFER_WP_REC_OUTPUT_SLOT_STRIDE] =
      WAFER_WP_OUTPUT_SLOT_STRIDE;
  record[WAFER_WP_REC_OUTPUT_GUARD_BYTES] = WAFER_WP_GUARD_BYTES;
}

static void wafer_wp_archive_outputs(WaferWPContext *context,
                                     volatile uint64_t *record) {
  for (uint32_t slot = 0; slot < context->issue_count; ++slot) {
    WaferWPInstruction *instruction = &context->instructions[slot];
    uint32_t owned_bytes =
        instruction->output_bytes + 2U * WAFER_WP_GUARD_BYTES;
    uint32_t cursor = 0U;
    while (cursor < owned_bytes) {
      uint32_t remaining = owned_bytes - cursor;
      uint32_t chunk =
          remaining > WAFER_WP_OUTPUT_SLOT_DATA_BYTES
              ? WAFER_WP_OUTPUT_SLOT_DATA_BYTES
              : remaining;
      wafer_tx81_wdma(
          wafer_wp_write(slot) - WAFER_WP_GUARD_BYTES + cursor,
          wafer_wp_archive(context, slot) + cursor, chunk, chunk, 0, 0, 0,
          1, 1, 1, Fmt_UINT8);
      wafer_tx81_local_fence();
      cursor += chunk;
    }
    record[WAFER_WP_REC_OUTPUT_BYTES_BASE + slot] =
        instruction->output_bytes;
    record[WAFER_WP_REC_OUTPUT_OFFSET_BASE + slot] =
        WAFER_WP_OUTPUT_SLOT_BASE +
        (uint64_t)slot * WAFER_WP_OUTPUT_SLOT_STRIDE +
        WAFER_WP_GUARD_BYTES;
  }
  record[WAFER_WP_REC_FLAGS] |= WAFER_WP_RECORD_OUTPUT_ARCHIVED;
}

__attribute__((visibility("hidden"))) void
wafer_tx81_worker_placement_probe(uint64_t request_ddr,
                                  uint64_t payload_ddr,
                                  uint64_t output_ddr) {
  wafer_wp_cache_range(request_ddr, WAFER_WP_REQUEST_WORDS * sizeof(uint64_t),
                       1U);
  wafer_wp_cache_range(payload_ddr, WAFER_WP_RESOURCE_BYTES, 1U);
  const volatile uint64_t *request_words =
      (const volatile uint64_t *)(uintptr_t)request_ddr;
  volatile uint64_t *record =
      (volatile uint64_t *)(uintptr_t)output_ddr;
  for (uint32_t index = 0; index < WAFER_WP_RECORD_WORDS; ++index)
    record[index] = 0;

  WaferWPContext *context = &wafer_wp_context;
  wafer_wp_zero(context, sizeof(*context));
  context->payload_ddr = payload_ddr;
  context->output_ddr = output_ddr;
  uint32_t status =
      wafer_wp_decode_request(request_words, &context->request);
  wafer_wp_echo_record(record, &context->request);
  if (status != WAFER_WP_STATUS_OK) {
    record[WAFER_WP_REC_STATUS] = status;
    wafer_wp_publish_record(record);
    return;
  }
  if (!wafer_wp_build_issues(context)) {
    record[WAFER_WP_REC_STATUS] = WAFER_WP_STATUS_BAD_REQUEST;
    wafer_wp_publish_record(record);
    return;
  }
  record[WAFER_WP_REC_ISSUE_COUNT] = context->issue_count;

  for (uint32_t slot = 0; slot < context->issue_count; ++slot) {
    WaferWPInstruction *instruction = &context->instructions[slot];
    if (!wafer_wp_seed_and_prepare(context, instruction)) {
      record[WAFER_WP_REC_STATUS] = WAFER_WP_STATUS_PREPARE_FAILED;
      wafer_wp_publish_record(record);
      return;
    }
    record[WAFER_WP_REC_ISSUE_WORKER_BASE + slot] =
        instruction->worker;
    record[WAFER_WP_REC_ISSUE_ENGINE_BASE + slot] =
        instruction->engine;
    record[WAFER_WP_REC_ISSUE_BYTES_BASE + slot] =
        instruction->output_bytes;
    record[WAFER_WP_REC_ISSUE_INTER_TYPE_BASE + slot] =
        wafer_wp_packet_inter_type(instruction);
  }
  __asm__ volatile("fence iorw, iorw" ::: "memory");
  record[WAFER_WP_REC_PMU_ENABLE] = wafer_wp_read_pmu32(GR_PMU_EN);
  for (uint32_t worker = 0; worker < WAFER_WP_WORKERS; ++worker)
    record[WAFER_WP_REC_SERIAL_MODE_BASE + worker] =
        get_ncc_reg(worker, GR_CSR_SERIAL_MODE_ADDR);
  wafer_wp_snapshot(record, WAFER_WP_REC_PMU64_BEFORE,
                    WAFER_WP_REC_INSTRUCTION_BEFORE,
                    WAFER_WP_REC_BLOCKING_BEFORE,
                    WAFER_WP_REC_STABLE_BEFORE);
  record[WAFER_WP_REC_FLAGS] |= WAFER_WP_RECORD_BEFORE_CAPTURED;
  record[WAFER_WP_REC_START_CYCLE] = wafer_wp_cycle();
  for (uint32_t slot = 0; slot < context->issue_count; ++slot) {
    uint64_t rc = TsmExecute(wafer_wp_packet(&context->instructions[slot]));
    record[WAFER_WP_REC_ISSUE_RC_BASE + slot] = rc;
    if (rc != 1U) {
      status = WAFER_WP_STATUS_ISSUE_FAILED;
      break;
    }
  }
  record[WAFER_WP_REC_AFTER_ISSUE_CYCLE] = wafer_wp_cycle();
  record[WAFER_WP_REC_FLAGS] |= WAFER_WP_RECORD_ISSUES_SUBMITTED;
  wafer_wp_controls(record, WAFER_WP_REC_CONTROL_AFTER_ISSUE);

  const WaferWPRequest *request = &context->request;
  if (request->kind == WAFER_WP_KIND_PLACEMENT) {
    for (uint32_t offset = 0; offset < WAFER_WP_WORKERS; ++offset) {
      uint32_t worker =
          (request->join_rotation + offset) % WAFER_WP_WORKERS;
      if ((request->worker_mask & (UINT32_C(1) << worker)) != 0U)
        wafer_wp_wait_worker(record, worker, WAFER_WP_REC_JOIN_CYCLE_BASE,
                             UINT32_MAX);
    }
  } else if (request->kind == WAFER_WP_KIND_BACKLOG_ONLY) {
    wafer_wp_wait_worker(record, request->target_worker,
                         WAFER_WP_REC_JOIN_CYCLE_BASE,
                         WAFER_WP_REC_TARGET_WAIT_CYCLES);
  } else {
    wafer_wp_wait_worker(record, request->observer_worker,
                         WAFER_WP_REC_JOIN_CYCLE_BASE,
                         WAFER_WP_REC_OBSERVER_WAIT_CYCLES);
  }
  wafer_wp_controls(record, WAFER_WP_REC_CONTROL_BOUNDARY);
  record[WAFER_WP_REC_OBSERVER_BOUNDARY_CYCLE] = wafer_wp_cycle();
  record[WAFER_WP_REC_OBSERVER_DONE_BOUNDARY] =
      (record[WAFER_WP_REC_CONTROL_BOUNDARY +
              request->observer_worker] &
       WAFER_WP_TASK_DONE) != 0U;
  record[WAFER_WP_REC_TARGET_PENDING_BOUNDARY] =
      (record[WAFER_WP_REC_CONTROL_BOUNDARY + request->target_worker] &
       WAFER_WP_TASK_DONE) == 0U;
  if (request->kind == WAFER_WP_KIND_SENTINEL_ONLY ||
      request->kind == WAFER_WP_KIND_CONCURRENT) {
    uint32_t sentinel_slot = context->issue_count - 1U;
    record[WAFER_WP_REC_BOUNDARY_MISMATCHES] =
        wafer_wp_mismatch(context, &context->instructions[sentinel_slot]) +
        wafer_wp_guard_mismatch(&context->instructions[sentinel_slot]);
  }
  record[WAFER_WP_REC_FLAGS] |= WAFER_WP_RECORD_BOUNDARY_CAPTURED;

  if (request->kind == WAFER_WP_KIND_CONCURRENT)
    wafer_wp_wait_worker(record, request->target_worker,
                         WAFER_WP_REC_JOIN_CYCLE_BASE,
                         WAFER_WP_REC_TARGET_WAIT_CYCLES);
  for (uint32_t worker = 0; worker < WAFER_WP_WORKERS; ++worker)
    if ((request->worker_mask & (UINT32_C(1) << worker)) != 0U &&
        (get_ncc_reg(worker, GR_CSR_CONTROL_ADDR) & WAFER_WP_TASK_DONE) ==
            0U)
      wafer_wp_wait_worker(record, worker, WAFER_WP_REC_JOIN_CYCLE_BASE,
                           UINT32_MAX);
  record[WAFER_WP_REC_FLAGS] |= WAFER_WP_RECORD_SAFETY_DRAINED;
  wafer_wp_controls(record, WAFER_WP_REC_CONTROL_FINAL);
  wafer_wp_snapshot(record, WAFER_WP_REC_PMU64_AFTER,
                    WAFER_WP_REC_INSTRUCTION_AFTER,
                    WAFER_WP_REC_BLOCKING_AFTER,
                    WAFER_WP_REC_STABLE_AFTER);
  record[WAFER_WP_REC_FINAL_CYCLE] = wafer_wp_cycle();
  record[WAFER_WP_REC_FLAGS] |= WAFER_WP_RECORD_FINAL_CAPTURED;

  uint64_t final_mismatches = 0;
  uint64_t final_guard_mismatches = 0;
  for (uint32_t slot = 0; slot < context->issue_count; ++slot) {
    uint64_t mismatches =
        wafer_wp_mismatch(context, &context->instructions[slot]);
    uint64_t guard_mismatches =
        wafer_wp_guard_mismatch(&context->instructions[slot]);
    record[WAFER_WP_REC_SLOT_MISMATCH_BASE + slot] = mismatches;
    record[WAFER_WP_REC_SLOT_GUARD_MISMATCH_BASE + slot] =
        guard_mismatches;
    final_mismatches += mismatches;
    final_guard_mismatches += guard_mismatches;
  }
  record[WAFER_WP_REC_FINAL_MISMATCHES] = final_mismatches;
  record[WAFER_WP_REC_FINAL_GUARD_MISMATCHES] =
      final_guard_mismatches;
  if (status == WAFER_WP_STATUS_OK &&
      (final_mismatches != 0U || final_guard_mismatches != 0U ||
       record[WAFER_WP_REC_BOUNDARY_MISMATCHES] != 0U))
    status = WAFER_WP_STATUS_ORACLE_FAILED;
  for (uint32_t worker = 0; worker < WAFER_WP_WORKERS; ++worker)
    if ((request->worker_mask & (UINT32_C(1) << worker)) != 0U &&
        (record[WAFER_WP_REC_CONTROL_FINAL + worker] &
         WAFER_WP_TASK_DONE) == 0U)
      status = WAFER_WP_STATUS_WAIT_FAILED;

  wafer_wp_archive_outputs(context, record);
  record[WAFER_WP_REC_STATUS] = status;
  wafer_wp_publish_record(record);
}
