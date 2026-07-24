#include "wafer_ne_calibration_probe_protocol.h"
#include "wafer_tx81_crt.h"

#include "instr_adapter.h"
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
  uint32_t kind;
  uint32_t profile;
  uint32_t option;
  uint32_t aux_span;
  uint32_t disposition;
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

static uint32_t wafer_nec_option_disposition(uint32_t option) {
  return option >= WAFER_NEC_OPTION_LEAKY_RELU
             ? WAFER_NEC_BOARD_OBSERVED
             : WAFER_NEC_BOARD_EXACT;
}

static uint32_t wafer_nec_option_aux_span(uint32_t option,
                                          uint32_t batch,
                                          uint32_t rows,
                                          uint32_t channels) {
  if (option == WAFER_NEC_OPTION_PSUM)
    return wafer_nec_physical_span(batch, rows, channels);
  if (option == WAFER_NEC_OPTION_BIAS ||
      option == WAFER_NEC_OPTION_POSITIVE_AXIS_SCALE ||
      option == WAFER_NEC_OPTION_NEGATIVE_AXIS_SCALE)
    return wafer_nec_physical_span(1U, 1U, channels);
  return 0U;
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
  uint32_t kind = (uint32_t)request[WAFER_NEC_REQ_KIND];
  uint32_t profile = (uint32_t)request[WAFER_NEC_REQ_PROFILE];
  uint32_t option = (uint32_t)request[WAFER_NEC_REQ_OPTION];
  uint32_t disposition =
      (uint32_t)request[WAFER_NEC_REQ_DISPOSITION];
  if (dtype > WAFER_NEC_BF16 || lhs_orientation > 1U ||
      rhs_orientation > 1U || kind > WAFER_NEC_CONV ||
      option > WAFER_NEC_OPTION_NEGATIVE_AXIS_SCALE)
    return WAFER_NEC_STATUS_UNSUPPORTED_CASE;

  uint32_t expected_case = 0U;
  uint32_t expected_disposition = WAFER_NEC_BOARD_EXACT;
  uint32_t lhs_span = 0U;
  uint32_t rhs_span = 0U;
  uint32_t output_span = 0U;
  uint32_t aux_span = 0U;
  if (kind == WAFER_NEC_GEMM) {
    uint32_t geometry = wafer_nec_geometry(batch, m, k, n);
    if (geometry == UINT32_MAX)
      return WAFER_NEC_STATUS_UNSUPPORTED_CASE;
    uint32_t orientation = lhs_orientation * 2U + rhs_orientation;
    if (profile == WAFER_NEC_DENSE &&
        option == WAFER_NEC_OPTION_NONE) {
      expected_case =
          WAFER_NEC_CASE_BASE + dtype * 12U + geometry * 4U + orientation;
    } else if (profile >= WAFER_NEC_BF16_CANCELLATION &&
               profile <= WAFER_NEC_BF16_NAN) {
      if (dtype != WAFER_NEC_BF16 || geometry != 0U ||
          orientation != 0U || option != WAFER_NEC_OPTION_NONE)
        return WAFER_NEC_STATUS_UNSUPPORTED_CASE;
      expected_case =
          WAFER_NEC_SPECIAL_CASE_BASE + profile - 1U;
      if (profile == WAFER_NEC_BF16_SIGNED_ZERO ||
          profile == WAFER_NEC_BF16_SUBNORMAL ||
          profile == WAFER_NEC_BF16_OVERFLOW_INF ||
          profile == WAFER_NEC_BF16_NAN)
        expected_disposition = WAFER_NEC_BOARD_OBSERVED;
    } else if (profile == WAFER_NEC_DENSE &&
               option != WAFER_NEC_OPTION_NONE) {
      if (geometry != 0U || orientation != 0U)
        return WAFER_NEC_STATUS_UNSUPPORTED_CASE;
      expected_case = WAFER_NEC_GEMM_OPTION_CASE_BASE +
                      dtype * 6U + option - 1U;
      expected_disposition =
          wafer_nec_option_disposition(option);
    } else {
      return WAFER_NEC_STATUS_UNSUPPORTED_CASE;
    }
    uint32_t lhs_rows = lhs_orientation != 0U ? k : m;
    uint32_t lhs_columns = lhs_orientation != 0U ? m : k;
    uint32_t rhs_rows = rhs_orientation != 0U ? n : k;
    uint32_t rhs_columns = rhs_orientation != 0U ? k : n;
    lhs_span =
        wafer_nec_physical_span(batch, lhs_rows, lhs_columns);
    rhs_span =
        wafer_nec_physical_span(batch, rhs_rows, rhs_columns);
    output_span = wafer_nec_physical_span(batch, m, n);
    aux_span =
        wafer_nec_option_aux_span(option, batch, m, n);
  } else {
    if (lhs_orientation != 0U || rhs_orientation != 0U ||
        batch != 2U || m != 17U || k != 19U ||
        (profile != WAFER_NEC_CONV_LARGE &&
         profile != WAFER_NEC_CONV_HELDOUT) ||
        (profile == WAFER_NEC_CONV_LARGE && n != 96U) ||
        (profile == WAFER_NEC_CONV_HELDOUT && n != 65U) ||
        (profile == WAFER_NEC_CONV_HELDOUT &&
         option != WAFER_NEC_OPTION_NONE))
      return WAFER_NEC_STATUS_UNSUPPORTED_CASE;
    if (option == WAFER_NEC_OPTION_NONE) {
      expected_case = WAFER_NEC_CONV_CASE_BASE + dtype * 2U +
                      (profile == WAFER_NEC_CONV_HELDOUT);
    } else {
      expected_case = WAFER_NEC_CONV_OPTION_CASE_BASE +
                      dtype * 6U + option - 1U;
      expected_disposition =
          wafer_nec_option_disposition(option);
    }
    lhs_span = wafer_nec_physical_span(2U, 17U * 19U, 65U);
    rhs_span = wafer_nec_physical_span(1U, 3U * 2U * n, 65U);
    output_span = wafer_nec_physical_span(2U, 17U * 10U, n);
    aux_span =
        wafer_nec_option_aux_span(option, 2U, 17U * 10U, n);
  }
  if (request[WAFER_NEC_REQ_CASE] != expected_case ||
      request[WAFER_NEC_REQ_LHS_SPAN] != lhs_span ||
      request[WAFER_NEC_REQ_RHS_SPAN] != rhs_span ||
      request[WAFER_NEC_REQ_OUTPUT_SPAN] != output_span ||
      request[WAFER_NEC_REQ_AUX_SPAN] != aux_span ||
      disposition != expected_disposition ||
      WAFER_NEC_BODY_OFFSET + lhs_span > WAFER_NEC_SLOT_BYTES ||
      WAFER_NEC_BODY_OFFSET + rhs_span > WAFER_NEC_SLOT_BYTES ||
      WAFER_NEC_BODY_OFFSET + output_span > WAFER_NEC_SLOT_BYTES ||
      WAFER_NEC_BODY_OFFSET + aux_span > WAFER_NEC_SLOT_BYTES)
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
  selected->kind = kind;
  selected->profile = profile;
  selected->option = option;
  selected->aux_span = aux_span;
  selected->disposition = disposition;
  return WAFER_NEC_STATUS_OK;
}

static Data_Format wafer_nec_format(uint32_t dtype) {
  return dtype == WAFER_NEC_F16 ? Fmt_FP16 : Fmt_BF16;
}

static void wafer_nec_configure_gemm_option(TsmGemm *gemm,
                                             TsmNeInstr *instruction,
                                             const WaferNECCase *selected) {
  Data_Format format = wafer_nec_format(selected->dtype);
  gemm->SetPsum(instruction,
                selected->option == WAFER_NEC_OPTION_PSUM,
                WAFER_NEC_SPM_AUX + WAFER_NEC_BODY_OFFSET,
                selected->option == WAFER_NEC_OPTION_PSUM
                    ? format
                    : Fmt_UNUSED);
  gemm->AddBias(instruction,
                selected->option == WAFER_NEC_OPTION_BIAS,
                WAFER_NEC_SPM_AUX + WAFER_NEC_BODY_OFFSET);
  gemm->SetNegativeAxisScale(
      instruction,
      selected->option == WAFER_NEC_OPTION_NEGATIVE_AXIS_SCALE,
      WAFER_NEC_SPM_AUX + WAFER_NEC_BODY_OFFSET);
  gemm->SetPositiveAxisScale(
      instruction,
      selected->option == WAFER_NEC_OPTION_POSITIVE_AXIS_SCALE,
      WAFER_NEC_SPM_AUX + WAFER_NEC_BODY_OFFSET);
  if (selected->option == WAFER_NEC_OPTION_RELU)
    gemm->EnableRelu(instruction);
  else
    gemm->DisableRelu(instruction);
  if (selected->option == WAFER_NEC_OPTION_LEAKY_RELU)
    gemm->EnableLeakyRelu(instruction);
  else
    gemm->DisableLeakyRelu(instruction);
}

static uint64_t wafer_nec_issue_gemm(const WaferNECCase *selected) {
  Data_Format format = wafer_nec_format(selected->dtype);
  TsmNeInstr instruction = {0};
  TsmGemm *gemm = TsmNewGemm();
  gemm->AddInput(&instruction,
                 WAFER_NEC_SPM_A + WAFER_NEC_BODY_OFFSET,
                 WAFER_NEC_SPM_B + WAFER_NEC_BODY_OFFSET, format);
  gemm->ConfigMKN(&instruction, selected->m, selected->k, selected->n);
  gemm->ConfigBatch(&instruction, selected->batch, selected->batch);
  gemm->SetTransflag(&instruction, selected->lhs_orientation,
                     !selected->rhs_orientation);
  gemm->SetQuant(&instruction, 0, 0, 0, 0);
  wafer_nec_configure_gemm_option(gemm, &instruction, selected);
  gemm->AddOutput(&instruction,
                  WAFER_NEC_SPM_OUTPUT + WAFER_NEC_BODY_OFFSET,
                  format);
  uint64_t result = TsmExecute(&instruction);
  TsmDeleteGemm(gemm);
  (void)TsmWaitfinish_bywork(0);
  return result;
}

static Data_Shape wafer_nec_shape(uint32_t n, uint32_t h, uint32_t w,
                                  uint32_t c) {
  Data_Shape shape = {(uint16_t)n, (uint16_t)h, (uint16_t)w,
                      (uint16_t)c};
  return shape;
}

static uint64_t wafer_nec_issue_conv(const WaferNECCase *selected) {
  Data_Format format = wafer_nec_format(selected->dtype);
  TsmNeInstr instruction = {0};
  TsmConv *conv = TsmNewConv();
  conv->AddInput(&instruction,
                 WAFER_NEC_SPM_A + WAFER_NEC_BODY_OFFSET,
                 wafer_nec_shape(2U, 17U, 19U, 65U), format);
  conv->AddWeight(&instruction,
                  WAFER_NEC_SPM_B + WAFER_NEC_BODY_OFFSET,
                  wafer_nec_shape(3U, 2U, selected->n, 65U),
                  format);
  conv->AddBias(&instruction,
                selected->option == WAFER_NEC_OPTION_BIAS,
                WAFER_NEC_SPM_AUX + WAFER_NEC_BODY_OFFSET);
  conv->AddOutput(&instruction,
                  WAFER_NEC_SPM_OUTPUT + WAFER_NEC_BODY_OFFSET,
                  wafer_nec_shape(2U, 17U, 10U, selected->n),
                  format);
  conv->SetOpType(&instruction, 0U);
  conv->SetNegativeAxisScale(
      &instruction,
      selected->option == WAFER_NEC_OPTION_NEGATIVE_AXIS_SCALE,
      WAFER_NEC_SPM_AUX + WAFER_NEC_BODY_OFFSET);
  conv->SetPositiveAxisScale(
      &instruction,
      selected->option == WAFER_NEC_OPTION_POSITIVE_AXIS_SCALE,
      WAFER_NEC_SPM_AUX + WAFER_NEC_BODY_OFFSET);
  conv->SetSparse(&instruction, 0U, 0U);
  conv->SetPsum(&instruction,
                selected->option == WAFER_NEC_OPTION_PSUM,
                WAFER_NEC_SPM_AUX + WAFER_NEC_BODY_OFFSET,
                selected->option == WAFER_NEC_OPTION_PSUM
                    ? format
                    : Fmt_UNUSED);
  conv->SetPads(&instruction, 1U, 0U, 2U, 1U);
  conv->SetUnPads(&instruction, 0U, 0U, 0U, 0U);
  conv->SetKernelStrides(&instruction, 3U, 2U, 2U, 1U);
  conv->SetDilations(&instruction, 1U, 1U);
  conv->SetQuant(&instruction, 0, 0, 0, 0);
  if (selected->option == WAFER_NEC_OPTION_RELU)
    conv->EnableRelu(&instruction);
  else
    conv->DisableRelu(&instruction);
  if (selected->option == WAFER_NEC_OPTION_LEAKY_RELU)
    conv->EnableLeakyRelu(&instruction);
  else
    conv->DisableLeakyRelu(&instruction);
  uint64_t result = TsmExecute(&instruction);
  TsmDeleteConv(conv);
  (void)TsmWaitfinish_bywork(0);
  return result;
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
    record[WAFER_NEC_REC_KIND] = selected.kind;
    record[WAFER_NEC_REC_PROFILE] = selected.profile;
    record[WAFER_NEC_REC_OPTION] = selected.option;
    record[WAFER_NEC_REC_AUX_SPAN] = selected.aux_span;
    record[WAFER_NEC_REC_DISPOSITION] = selected.disposition;

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
    wafer_tx81_rdma(payload_ddr + 3U * WAFER_NEC_SLOT_BYTES,
                    WAFER_NEC_SPM_AUX, WAFER_NEC_SLOT_BYTES,
                    WAFER_NEC_SLOT_BYTES, 0, 0, 0, 1, 1, 1,
                    Fmt_UINT8);
    wafer_tx81_local_fence();
    uint64_t execute_result =
        selected.kind == WAFER_NEC_GEMM
            ? wafer_nec_issue_gemm(&selected)
            : wafer_nec_issue_conv(&selected);
    record[WAFER_NEC_REC_EXECUTE_RESULT] = execute_result;
    if (execute_result == 0U) {
      status = WAFER_NEC_STATUS_EXECUTE_FAILED;
    } else {
      record[WAFER_NEC_REC_OUTPUT_GUARD_MISMATCHES] =
          wafer_nec_output_guard_mismatches(selected.output_span);
      wafer_tx81_wdma(WAFER_NEC_SPM_OUTPUT,
                      output_ddr + WAFER_NEC_OUTPUT_DDR_OFFSET,
                      WAFER_NEC_SLOT_BYTES, WAFER_NEC_SLOT_BYTES,
                      0, 0, 0, 1, 1, 1, Fmt_UINT8);
      wafer_tx81_local_fence();
    }
    record[WAFER_NEC_REC_STATUS] = status;
  }
  wafer_nec_cache_range(output_ddr,
                        WAFER_NEC_RECORD_WORDS * sizeof(uint64_t), 0);
}
