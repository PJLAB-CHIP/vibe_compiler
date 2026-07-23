#include "instr_def.h"
#include "wafer_instruction_family_probe_protocol.h"
#include "wafer_tx81_crt.h"

#include <stddef.h>
#include <stdint.h>

extern int8_t *get_spm_memory_mapping(uint64_t offset);

typedef struct WaferIFPDescriptor {
  uint32_t id;
  uint32_t disposition;
  uint32_t family;
  uint32_t dtype;
  uint32_t oracle;
  uint32_t result_bytes;
  uint32_t output_span;
  uint32_t aux_span;
} WaferIFPDescriptor;

static const WaferIFPDescriptor wafer_ifp_catalog[] = {
#define WAFER_IFP_DESCRIPTOR(SYMBOL, ID, SPELLING, DISPOSITION, FAMILY, DTYPE, \
                             ORACLE, REASON, RESULT_BYTES, OUTPUT_SPAN,        \
                             AUX_SPAN)                                         \
  {ID, WAFER_IFP_##DISPOSITION, WAFER_IFP_##FAMILY, WAFER_IFP_##DTYPE,        \
   WAFER_IFP_##ORACLE, RESULT_BYTES, OUTPUT_SPAN, AUX_SPAN},
    WAFER_IFP_CASES(WAFER_IFP_DESCRIPTOR)
#undef WAFER_IFP_DESCRIPTOR
};

static const WaferIFPDescriptor *wafer_ifp_find(uint32_t id) {
  for (size_t index = 0;
       index < sizeof(wafer_ifp_catalog) / sizeof(wafer_ifp_catalog[0]);
       ++index)
    if (wafer_ifp_catalog[index].id == id)
      return &wafer_ifp_catalog[index];
  return NULL;
}

static void wafer_ifp_cache_range(uint64_t begin, uint32_t bytes,
                                  uint32_t invalidate_only) {
  enum {
    WAFER_TX81_SUPERVISOR_MODE = 1,
    WAFER_TX81_MACHINE_MODE = 3,
    WAFER_TX81_CACHE_LINE_BYTES = 64,
  };
  uintptr_t mode;
  __asm__ volatile("fence" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
  __asm__ volatile("csrr %0, mxstatus" : "=r"(mode));
  mode = (mode >> 30) & 3U;
  for (uintptr_t address = (uintptr_t)begin;
       address < (uintptr_t)begin + bytes;
       address += WAFER_TX81_CACHE_LINE_BYTES) {
    if (mode == WAFER_TX81_MACHINE_MODE) {
      if (invalidate_only != 0)
        __asm__ volatile("dcache.ipa %0" : : "r"(address) : "memory");
      else
        __asm__ volatile("dcache.cipa %0" : : "r"(address) : "memory");
    } else if (mode == WAFER_TX81_SUPERVISOR_MODE) {
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

static void wafer_ifp_invalidate(uint64_t begin, uint32_t bytes) {
  wafer_ifp_cache_range(begin, bytes, 1);
}

static void wafer_ifp_publish(volatile uint64_t *record) {
  wafer_ifp_cache_range((uint64_t)(uintptr_t)record,
                        WAFER_IFP_RECORD_WORDS * sizeof(uint64_t), 0);
}

static uint64_t wafer_ifp_body(uint64_t slot) {
  return slot + WAFER_IFP_BODY_OFFSET;
}

static void wafer_ifp_seed(uint64_t payload_ddr) {
  const uint64_t destinations[] = {
      WAFER_IFP_SPM_A,
      WAFER_IFP_SPM_B,
      WAFER_IFP_SPM_OUTPUT,
      WAFER_IFP_SPM_AUX,
  };
  for (uint32_t slot = 0; slot < 4; ++slot)
    wafer_tx81_rdma(payload_ddr + slot * WAFER_IFP_SLOT_BYTES,
                    destinations[slot], WAFER_IFP_SLOT_BYTES,
                    WAFER_IFP_SLOT_BYTES, 0, 0, 0, 1, 1, 1, Fmt_UINT8);
  wafer_tx81_local_fence();
}

static uint64_t wafer_ifp_guard_mismatches(uint64_t slot,
                                           uint32_t allowed_span) {
  const volatile uint8_t *bytes =
      (const volatile uint8_t *)(const void *)get_spm_memory_mapping(slot);
  uint64_t mismatches = 0;
  uint32_t allowed_end = WAFER_IFP_BODY_OFFSET + allowed_span;
  for (uint32_t index = 0; index < WAFER_IFP_SLOT_BYTES; ++index)
    if ((index < WAFER_IFP_BODY_OFFSET || index >= allowed_end) &&
        bytes[index] != WAFER_IFP_SLOT_CANARY)
      ++mismatches;
  return mismatches;
}

static uint32_t wafer_ifp_fp_format(const WaferIFPDescriptor *descriptor) {
  switch (descriptor->dtype) {
  case WAFER_IFP_F16:
  case WAFER_IFP_I8_TO_F16:
  case WAFER_IFP_BF16_TO_F16:
  case WAFER_IFP_F16_TO_BF16:
  case WAFER_IFP_F16_TO_I16:
    return Fmt_FP16;
  case WAFER_IFP_BF16:
  case WAFER_IFP_I8_TO_BF16:
    return Fmt_BF16;
  default:
    return Fmt_UNUSED;
  }
}

static uint64_t wafer_ifp_bit2fp_mismatches(uint64_t auxiliary,
                                            uint32_t format) {
  const volatile uint16_t *actual =
      (const volatile uint16_t *)(const void *)get_spm_memory_mapping(auxiliary);
  uint16_t true_value =
      format == Fmt_FP16 ? WAFER_IFP_BIT2FP_TRUE_F16
                         : WAFER_IFP_BIT2FP_TRUE_BF16;
  uint64_t mismatches = 0;
  for (uint32_t index = 0; index < 128; ++index) {
    uint16_t expected = ((index / 8U) & 1U) == 0U
                            ? true_value
                            : WAFER_IFP_BIT2FP_FALSE;
    mismatches += actual[index] != expected;
  }
  return mismatches;
}

static int wafer_ifp_dispatch(const WaferIFPDescriptor *descriptor,
                              volatile uint64_t *record) {
  const uint64_t input_a = wafer_ifp_body(WAFER_IFP_SPM_A);
  const uint64_t input_b = wafer_ifp_body(WAFER_IFP_SPM_B);
  const uint64_t output = wafer_ifp_body(WAFER_IFP_SPM_OUTPUT);
  const uint64_t auxiliary = wafer_ifp_body(WAFER_IFP_SPM_AUX);
  const uint32_t format = wafer_ifp_fp_format(descriptor);
  const uint32_t elements = 128;
  record[WAFER_IFP_REC_STEP_FLAGS] = WAFER_IFP_STEP_TARGET_ISSUED;

  switch (descriptor->id) {
  case WAFER_IFP_CASE_CT_NEG_F16:
  case WAFER_IFP_CASE_CT_NEG_BF16:
    wafer_tx81_elementwise_neg(input_a, output, elements, format);
    break;
  case WAFER_IFP_CASE_CT_ADD_F16:
  case WAFER_IFP_CASE_CT_ADD_BF16:
    wafer_tx81_elementwise_add(input_a, input_b, output, elements, format);
    break;
  case WAFER_IFP_CASE_CT_SUB_F16:
  case WAFER_IFP_CASE_CT_SUB_BF16:
    wafer_tx81_elementwise_sub(input_a, input_b, output, elements, format);
    break;
  case WAFER_IFP_CASE_CT_MUL_F16:
  case WAFER_IFP_CASE_CT_MUL_BF16:
    wafer_tx81_elementwise_mul(input_a, input_b, output, elements, format);
    break;
  case WAFER_IFP_CASE_CT_MAX_F16:
  case WAFER_IFP_CASE_CT_MAX_BF16:
    wafer_tx81_elementwise_max(input_a, input_b, output, elements, format);
    break;
  case WAFER_IFP_CASE_CT_MIN_F16:
  case WAFER_IFP_CASE_CT_MIN_BF16:
    wafer_tx81_elementwise_min(input_a, input_b, output, elements, format);
    break;
  case WAFER_IFP_CASE_CT_POW2_F16:
  case WAFER_IFP_CASE_CT_POW2_BF16:
    wafer_tx81_elementwise_pow2(input_a, output, elements, format);
    break;
  case WAFER_IFP_CASE_CT_RELU_F16:
  case WAFER_IFP_CASE_CT_RELU_BF16:
    wafer_tx81_elementwise_relu(input_a, output, elements, format);
    break;
  case WAFER_IFP_CASE_CONVERT_I8_F16_ZP0:
    wafer_tx81_convert_int8_fp16(input_a, output, elements, 0, 0);
    break;
  case WAFER_IFP_CASE_CONVERT_I8_BF16_ZP0:
    wafer_tx81_convert_int8_bf16(input_a, output, elements, 0, 0);
    break;
  case WAFER_IFP_CASE_CONVERT_BF16_F16_PLAIN:
    wafer_tx81_convert_bf16_fp16(input_a, output, elements, 0, 0);
    break;
  case WAFER_IFP_CASE_CONVERT_F16_BF16_ROUND:
    wafer_tx81_convert_fp16_bf16(input_a, output, elements, 0, 0);
    break;
  case WAFER_IFP_CASE_CONVERT_F16_I16_ROUND:
    wafer_tx81_convert_fp16_int16(input_a, output, elements, 0, 0);
    break;
  case WAFER_IFP_CASE_REDUCE_SUM_F16:
    wafer_tx81_reduce_sum(input_a, output, 1, 1, 1, 4, 64, Fmt_FP16);
    break;
  case WAFER_IFP_CASE_REDUCE_MAX_F16:
    wafer_tx81_reduce_max(input_a, output, 1, 1, 1, 4, 64, Fmt_FP16);
    break;
  case WAFER_IFP_CASE_REDUCE_MIN_BF16:
    wafer_tx81_reduce_min(input_a, output, 1, 1, 1, 4, 64, Fmt_BF16);
    break;
  case WAFER_IFP_CASE_REDUCE_AVG_BF16:
    wafer_tx81_reduce_avg(input_a, output, 1, 1, 1, 4, 64, Fmt_BF16);
    break;
  case WAFER_IFP_CASE_SELECT_F16:
  case WAFER_IFP_CASE_SELECT_BF16:
    wafer_tx81_bit2fp(input_b, auxiliary, elements, format);
    wafer_tx81_local_fence();
    record[WAFER_IFP_REC_STEP_FLAGS] |= WAFER_IFP_STEP_BIT2FP_COMPLETED;
    record[WAFER_IFP_REC_BIT2FP_MISMATCHES] =
        wafer_ifp_bit2fp_mismatches(auxiliary, format);
    wafer_tx81_mask_move(input_a, (uint32_t)auxiliary, output, elements,
                         format);
    record[WAFER_IFP_REC_STEP_FLAGS] |= WAFER_IFP_STEP_MASK_MOVE_ISSUED;
    break;
  case WAFER_IFP_CASE_GEMM_F16:
    wafer_tx81_gemm(input_a, input_b, output, 1, 16, 16, 1, Fmt_FP16);
    break;
  case WAFER_IFP_CASE_TDMA_PAD_F16:
    wafer_tx81_tdma_pad(input_a, output, 1, 2, 2, 64, 1, 4, 4, 64, 1, 1, 1,
                        1, Fmt_FP16);
    break;
  case WAFER_IFP_CASE_POOL_F16:
    wafer_tx81_pool_max(
        input_a, output, OP_FUNC_CGRATensor_PoolOp_T_T_max, 1, 2, 4, 64, 1, 1,
        2, 64, 0, 0, 0, 0, 2, 2, 2, 2, Fmt_FP16);
    break;
  case WAFER_IFP_CASE_PERIPHERAL_ARGMAX_F16:
    wafer_tx81_peripheral_argmax(
        input_a, output, output + 4,
        OP_FUNC_CGRATensor_PeriOp_V_V_argmax, elements, Fmt_FP16, 0, 0, 0, 0);
    break;
  case WAFER_IFP_CASE_PERIPHERAL_ARGMIN_F16:
    wafer_tx81_peripheral_argmin(
        input_a, output, output + 4,
        OP_FUNC_CGRATensor_PeriOp_V_V_argmin, elements, Fmt_FP16, 0, 0, 0, 0);
    break;
  default:
    return 1;
  }
  wafer_tx81_local_fence();
  record[WAFER_IFP_REC_STEP_FLAGS] |= WAFER_IFP_STEP_FINAL_FENCE_COMPLETED;
  return 0;
}

static uint32_t wafer_ifp_decode(
    const volatile uint64_t *request,
    const WaferIFPDescriptor **descriptor_out) {
  if (request[WAFER_IFP_REQ_MAGIC] != WAFER_IFP_REQUEST_MAGIC ||
      request[WAFER_IFP_REQ_SCHEMA_AND_WORDS] !=
          (((uint64_t)WAFER_IFP_SCHEMA << 32) | WAFER_IFP_REQUEST_WORDS) ||
      request[WAFER_IFP_REQ_GUARD] != WAFER_IFP_REQUEST_GUARD)
    return WAFER_IFP_STATUS_BAD_REQUEST;

  const WaferIFPDescriptor *descriptor =
      wafer_ifp_find((uint32_t)request[WAFER_IFP_REQ_CASE]);
  if (descriptor == NULL)
    return WAFER_IFP_STATUS_UNKNOWN_CASE;
  if (request[WAFER_IFP_REQ_DISPOSITION] != descriptor->disposition ||
      request[WAFER_IFP_REQ_FAMILY] != descriptor->family ||
      request[WAFER_IFP_REQ_DTYPE] != descriptor->dtype ||
      request[WAFER_IFP_REQ_ORACLE] != descriptor->oracle ||
      request[WAFER_IFP_REQ_RESULT_BYTES] != descriptor->result_bytes ||
      request[WAFER_IFP_REQ_OUTPUT_SPAN] != descriptor->output_span ||
      request[WAFER_IFP_REQ_AUX_SPAN] != descriptor->aux_span ||
      request[WAFER_IFP_REQ_RESOURCE_BYTES] != WAFER_IFP_RESOURCE_BYTES ||
      request[WAFER_IFP_REQ_SLOT_BYTES] != WAFER_IFP_SLOT_BYTES ||
      request[WAFER_IFP_REQ_BODY_OFFSET] != WAFER_IFP_BODY_OFFSET)
    return WAFER_IFP_STATUS_BAD_REQUEST;
  *descriptor_out = descriptor;
  return descriptor->disposition == WAFER_IFP_SAFE
             ? WAFER_IFP_STATUS_OK
             : WAFER_IFP_STATUS_DEFERRED_CASE;
}

static void wafer_ifp_init_record(volatile uint64_t *record, uint32_t status) {
  for (uint32_t index = 0; index < WAFER_IFP_RECORD_WORDS; ++index)
    record[index] = 0;
  record[WAFER_IFP_REC_MAGIC] = WAFER_IFP_RECORD_MAGIC;
  record[WAFER_IFP_REC_SCHEMA_AND_WORDS] =
      ((uint64_t)WAFER_IFP_SCHEMA << 32) | WAFER_IFP_RECORD_WORDS;
  record[WAFER_IFP_REC_STATUS] = status;
  record[WAFER_IFP_REC_OUTPUT_DDR_OFFSET] = WAFER_IFP_OUTPUT_DDR_OFFSET;
  record[WAFER_IFP_REC_SLOT_BYTES] = WAFER_IFP_SLOT_BYTES;
  record[WAFER_IFP_REC_BODY_OFFSET] = WAFER_IFP_BODY_OFFSET;
  record[WAFER_IFP_REC_RECORD_GUARD] = WAFER_IFP_RECORD_GUARD;
}

__attribute__((visibility("hidden"))) void
wafer_tx81_instruction_family_probe(uint64_t request_ddr,
                                    uint64_t payload_ddr,
                                    uint64_t output_ddr) {
  wafer_ifp_invalidate(request_ddr, WAFER_IFP_RESOURCE_BYTES);
  wafer_ifp_invalidate(payload_ddr, WAFER_IFP_RESOURCE_BYTES);

  const volatile uint64_t *request =
      (const volatile uint64_t *)(uintptr_t)request_ddr;
  volatile uint64_t *record = (volatile uint64_t *)(uintptr_t)output_ddr;
  const WaferIFPDescriptor *descriptor = NULL;
  uint32_t status = wafer_ifp_decode(request, &descriptor);
  wafer_ifp_init_record(record, status);

  if (descriptor != NULL) {
    record[WAFER_IFP_REC_CASE] = descriptor->id;
    record[WAFER_IFP_REC_DISPOSITION] = descriptor->disposition;
    record[WAFER_IFP_REC_FAMILY] = descriptor->family;
    record[WAFER_IFP_REC_DTYPE] = descriptor->dtype;
    record[WAFER_IFP_REC_ORACLE] = descriptor->oracle;
    record[WAFER_IFP_REC_RESULT_BYTES] = descriptor->result_bytes;
    record[WAFER_IFP_REC_OUTPUT_SPAN] = descriptor->output_span;
    record[WAFER_IFP_REC_AUX_SPAN] = descriptor->aux_span;
    record[WAFER_IFP_REC_SAMPLE] = request[WAFER_IFP_REQ_SAMPLE];
    record[WAFER_IFP_REC_REQUEST_GUARD] =
        request[WAFER_IFP_REQ_GUARD];
  }

  if (status == WAFER_IFP_STATUS_OK) {
    wafer_ifp_seed(payload_ddr);
    if (wafer_ifp_dispatch(descriptor, record) != 0) {
      status = WAFER_IFP_STATUS_DISPATCH_FAILED;
    } else {
      record[WAFER_IFP_REC_OUTPUT_GUARD_MISMATCHES] =
          wafer_ifp_guard_mismatches(WAFER_IFP_SPM_OUTPUT,
                                     descriptor->output_span);
      record[WAFER_IFP_REC_AUX_GUARD_MISMATCHES] =
          wafer_ifp_guard_mismatches(WAFER_IFP_SPM_AUX,
                                     descriptor->aux_span);
      wafer_tx81_wdma(WAFER_IFP_SPM_OUTPUT,
                      output_ddr + WAFER_IFP_OUTPUT_DDR_OFFSET,
                      WAFER_IFP_SLOT_BYTES, WAFER_IFP_SLOT_BYTES, 0, 0, 0, 1,
                      1, 1, Fmt_UINT8);
      wafer_tx81_local_fence();
    }
    record[WAFER_IFP_REC_STATUS] = status;
  }
  wafer_ifp_publish(record);
}
