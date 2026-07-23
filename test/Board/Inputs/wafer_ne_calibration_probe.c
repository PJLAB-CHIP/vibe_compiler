#include "wafer_ne_calibration_probe_protocol.h"
#include "wafer_tx81_crt.h"

#include "instr_def.h"

#include <stdint.h>

extern int8_t *get_spm_memory_mapping(uint64_t offset);

typedef struct WaferNECCase {
  uint32_t case_id;
  uint32_t dtype;
  uint32_t lhs_orientation;
  uint32_t rhs_orientation;
  uint32_t batch;
  uint32_t m;
  uint32_t k;
  uint32_t n;
  uint32_t lhs_span;
  uint32_t rhs_span;
  uint32_t output_span;
} WaferNECCase;

static void wafer_nec_cache_range(uint64_t begin, uint32_t bytes,
                                  uint32_t invalidate_only) {
  enum {
    WAFER_NEC_SUPERVISOR_MODE = 1,
    WAFER_NEC_MACHINE_MODE = 3,
    WAFER_NEC_CACHE_LINE_BYTES = 64,
  };
  uintptr_t mode;
  __asm__ volatile("fence" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
  __asm__ volatile("csrr %0, mxstatus" : "=r"(mode));
  mode = (mode >> 30) & 3U;
  for (uintptr_t address = (uintptr_t)begin;
       address < (uintptr_t)begin + bytes;
       address += WAFER_NEC_CACHE_LINE_BYTES) {
    if (mode == WAFER_NEC_MACHINE_MODE) {
      if (invalidate_only != 0)
        __asm__ volatile("dcache.ipa %0" : : "r"(address) : "memory");
      else
        __asm__ volatile("dcache.cipa %0" : : "r"(address) : "memory");
    } else if (mode == WAFER_NEC_SUPERVISOR_MODE) {
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

static uint32_t wafer_nec_align_up(uint32_t value, uint32_t alignment) {
  return ((value + alignment - 1U) / alignment) * alignment;
}

static uint32_t wafer_nec_aligned_c(uint32_t channels) {
  uint32_t full_blocks = channels / 64U;
  uint32_t remainder = channels % 64U;
  uint32_t tail = 0;
  if (remainder != 0U) {
    if (remainder <= 32U) {
      if (remainder <= 4U)
        tail = 4U;
      else if (remainder <= 8U)
        tail = 8U;
      else if (remainder <= 16U)
        tail = 16U;
      else
        tail = 32U;
    } else {
      ++full_blocks;
    }
  }
  return full_blocks * 64U + tail;
}

static uint32_t wafer_nec_physical_span(uint32_t batch, uint32_t rows,
                                        uint32_t channels) {
  uint32_t batch_elements =
      wafer_nec_align_up(rows * wafer_nec_aligned_c(channels), 128U);
  return batch * batch_elements * 2U;
}

static uint32_t wafer_nec_geometry(uint32_t batch, uint32_t m, uint32_t k,
                                   uint32_t n) {
  if (batch == 1U && m == 64U && k == 128U && n == 128U)
    return 0U;
  if (batch == 1U && m == 65U && k == 129U && n == 129U)
    return 1U;
  if (batch == 2U && m == 32U && k == 64U && n == 65U)
    return 2U;
  return UINT32_MAX;
}

static uint32_t wafer_nec_decode(const volatile uint64_t *request,
                                 WaferNECCase *selected) {
  if (request[WAFER_NEC_REQ_MAGIC] != WAFER_NEC_REQUEST_MAGIC ||
      request[WAFER_NEC_REQ_SCHEMA_AND_WORDS] !=
          (((uint64_t)WAFER_NEC_SCHEMA << 32) |
           WAFER_NEC_REQUEST_WORDS) ||
      request[WAFER_NEC_REQ_GUARD] != WAFER_NEC_REQUEST_GUARD ||
      request[WAFER_NEC_REQ_RESOURCE_BYTES] != WAFER_NEC_RESOURCE_BYTES ||
      request[WAFER_NEC_REQ_SLOT_BYTES] != WAFER_NEC_SLOT_BYTES ||
      request[WAFER_NEC_REQ_BODY_OFFSET] != WAFER_NEC_BODY_OFFSET)
    return WAFER_NEC_STATUS_BAD_REQUEST;

  uint32_t dtype = (uint32_t)request[WAFER_NEC_REQ_DTYPE];
  uint32_t lhs_orientation =
      (uint32_t)request[WAFER_NEC_REQ_LHS_ORIENTATION];
  uint32_t rhs_orientation =
      (uint32_t)request[WAFER_NEC_REQ_RHS_ORIENTATION];
  uint32_t batch = (uint32_t)request[WAFER_NEC_REQ_BATCH];
  uint32_t m = (uint32_t)request[WAFER_NEC_REQ_M];
  uint32_t k = (uint32_t)request[WAFER_NEC_REQ_K];
  uint32_t n = (uint32_t)request[WAFER_NEC_REQ_N];
  uint32_t geometry = wafer_nec_geometry(batch, m, k, n);
  if (dtype > WAFER_NEC_BF16 || lhs_orientation > 1U ||
      rhs_orientation > 1U || geometry == UINT32_MAX)
    return WAFER_NEC_STATUS_UNSUPPORTED_CASE;

  uint32_t orientation = lhs_orientation * 2U + rhs_orientation;
  uint32_t expected_case =
      WAFER_NEC_CASE_BASE + dtype * 12U + geometry * 4U + orientation;
  uint32_t lhs_rows = lhs_orientation != 0U ? k : m;
  uint32_t lhs_columns = lhs_orientation != 0U ? m : k;
  uint32_t rhs_rows = rhs_orientation != 0U ? n : k;
  uint32_t rhs_columns = rhs_orientation != 0U ? k : n;
  uint32_t lhs_span =
      wafer_nec_physical_span(batch, lhs_rows, lhs_columns);
  uint32_t rhs_span =
      wafer_nec_physical_span(batch, rhs_rows, rhs_columns);
  uint32_t output_span = wafer_nec_physical_span(batch, m, n);
  if (request[WAFER_NEC_REQ_CASE] != expected_case ||
      request[WAFER_NEC_REQ_LHS_SPAN] != lhs_span ||
      request[WAFER_NEC_REQ_RHS_SPAN] != rhs_span ||
      request[WAFER_NEC_REQ_OUTPUT_SPAN] != output_span ||
      WAFER_NEC_BODY_OFFSET + lhs_span > WAFER_NEC_SLOT_BYTES ||
      WAFER_NEC_BODY_OFFSET + rhs_span > WAFER_NEC_SLOT_BYTES ||
      WAFER_NEC_BODY_OFFSET + output_span > WAFER_NEC_SLOT_BYTES)
    return WAFER_NEC_STATUS_BAD_REQUEST;

  selected->case_id = expected_case;
  selected->dtype = dtype;
  selected->lhs_orientation = lhs_orientation;
  selected->rhs_orientation = rhs_orientation;
  selected->batch = batch;
  selected->m = m;
  selected->k = k;
  selected->n = n;
  selected->lhs_span = lhs_span;
  selected->rhs_span = rhs_span;
  selected->output_span = output_span;
  return WAFER_NEC_STATUS_OK;
}

static void wafer_nec_init_record(volatile uint64_t *record,
                                  uint32_t status) {
  for (uint32_t index = 0; index < WAFER_NEC_RECORD_WORDS; ++index)
    record[index] = 0;
  record[WAFER_NEC_REC_MAGIC] = WAFER_NEC_RECORD_MAGIC;
  record[WAFER_NEC_REC_SCHEMA_AND_WORDS] =
      ((uint64_t)WAFER_NEC_SCHEMA << 32) | WAFER_NEC_RECORD_WORDS;
  record[WAFER_NEC_REC_STATUS] = status;
  record[WAFER_NEC_REC_OUTPUT_DDR_OFFSET] =
      WAFER_NEC_OUTPUT_DDR_OFFSET;
  record[WAFER_NEC_REC_SLOT_BYTES] = WAFER_NEC_SLOT_BYTES;
  record[WAFER_NEC_REC_BODY_OFFSET] = WAFER_NEC_BODY_OFFSET;
  record[WAFER_NEC_REC_RECORD_GUARD] = WAFER_NEC_RECORD_GUARD;
}

static uint64_t wafer_nec_output_guard_mismatches(uint32_t output_span) {
  const volatile uint8_t *output =
      (const volatile uint8_t *)(const void *)
          get_spm_memory_mapping(WAFER_NEC_SPM_OUTPUT);
  uint32_t allowed_begin = WAFER_NEC_BODY_OFFSET;
  uint32_t allowed_end = allowed_begin + output_span;
  uint64_t mismatches = 0;
  for (uint32_t index = 0; index < WAFER_NEC_SLOT_BYTES; ++index)
    if ((index < allowed_begin || index >= allowed_end) &&
        output[index] != WAFER_NEC_SLOT_CANARY)
      ++mismatches;
  return mismatches;
}

__attribute__((visibility("hidden"))) void
wafer_tx81_instruction_family_probe(uint64_t request_ddr,
                                    uint64_t payload_ddr,
                                    uint64_t output_ddr) {
  wafer_nec_cache_range(request_ddr, WAFER_NEC_RESOURCE_BYTES, 1);
  wafer_nec_cache_range(payload_ddr, WAFER_NEC_RESOURCE_BYTES, 1);
  const volatile uint64_t *request =
      (const volatile uint64_t *)(uintptr_t)request_ddr;
  volatile uint64_t *record =
      (volatile uint64_t *)(uintptr_t)output_ddr;
  WaferNECCase selected = {0};
  uint32_t status = wafer_nec_decode(request, &selected);
  wafer_nec_init_record(record, status);

  if (status == WAFER_NEC_STATUS_OK) {
    record[WAFER_NEC_REC_CASE] = selected.case_id;
    record[WAFER_NEC_REC_DTYPE] = selected.dtype;
    record[WAFER_NEC_REC_LHS_ORIENTATION] = selected.lhs_orientation;
    record[WAFER_NEC_REC_RHS_ORIENTATION] = selected.rhs_orientation;
    record[WAFER_NEC_REC_BATCH] = selected.batch;
    record[WAFER_NEC_REC_M] = selected.m;
    record[WAFER_NEC_REC_K] = selected.k;
    record[WAFER_NEC_REC_N] = selected.n;
    record[WAFER_NEC_REC_LHS_SPAN] = selected.lhs_span;
    record[WAFER_NEC_REC_RHS_SPAN] = selected.rhs_span;
    record[WAFER_NEC_REC_OUTPUT_SPAN] = selected.output_span;
    record[WAFER_NEC_REC_SAMPLE] = request[WAFER_NEC_REQ_SAMPLE];
    record[WAFER_NEC_REC_REQUEST_GUARD] =
        request[WAFER_NEC_REQ_GUARD];

    wafer_tx81_rdma(payload_ddr, WAFER_NEC_SPM_A,
                    WAFER_NEC_SLOT_BYTES, WAFER_NEC_SLOT_BYTES,
                    0, 0, 0, 1, 1, 1, Fmt_UINT8);
    wafer_tx81_rdma(payload_ddr + WAFER_NEC_SLOT_BYTES,
                    WAFER_NEC_SPM_B, WAFER_NEC_SLOT_BYTES,
                    WAFER_NEC_SLOT_BYTES, 0, 0, 0, 1, 1, 1,
                    Fmt_UINT8);
    wafer_tx81_rdma(payload_ddr + 2U * WAFER_NEC_SLOT_BYTES,
                    WAFER_NEC_SPM_OUTPUT, WAFER_NEC_SLOT_BYTES,
                    WAFER_NEC_SLOT_BYTES, 0, 0, 0, 1, 1, 1,
                    Fmt_UINT8);
    wafer_tx81_local_fence();
    wafer_tx81_gemm_oriented_v2(
        WAFER_NEC_SPM_A + WAFER_NEC_BODY_OFFSET,
        WAFER_NEC_SPM_B + WAFER_NEC_BODY_OFFSET,
        WAFER_NEC_SPM_OUTPUT + WAFER_NEC_BODY_OFFSET,
        selected.m, selected.k, selected.n, selected.batch,
        selected.dtype == WAFER_NEC_F16 ? Fmt_FP16 : Fmt_BF16,
        selected.lhs_orientation, selected.rhs_orientation);
    wafer_tx81_local_fence();
    record[WAFER_NEC_REC_EXECUTE_RESULT] = 1;
    record[WAFER_NEC_REC_OUTPUT_GUARD_MISMATCHES] =
        wafer_nec_output_guard_mismatches(selected.output_span);
    wafer_tx81_wdma(WAFER_NEC_SPM_OUTPUT,
                    output_ddr + WAFER_NEC_OUTPUT_DDR_OFFSET,
                    WAFER_NEC_SLOT_BYTES, WAFER_NEC_SLOT_BYTES,
                    0, 0, 0, 1, 1, 1, Fmt_UINT8);
    wafer_tx81_local_fence();
  }
  wafer_nec_cache_range(output_ddr,
                        WAFER_NEC_RECORD_WORDS * sizeof(uint64_t), 0);
}
