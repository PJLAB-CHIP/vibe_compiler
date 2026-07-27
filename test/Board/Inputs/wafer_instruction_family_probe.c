#include "instr_def.h"
#include "instr_adapter.h"
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

static void wafer_ifp_wait_worker0_drain(void) {
  /*
   * TsmWaitfinish() observes TASK_DONE only.  This probe seeds a four-command
   * RDMA window, so also require the worker-0 instruction buffer to become
   * empty before its first consumer is issued.
   */
  __asm__ volatile("fence iorw, iorw" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
  __asm__ volatile("sync.is" ::: "memory");
  while (TsmGetCsrIbcounter() != 0U ||
         TsmGetCsrTaskstatus() != 1U) {
  }
  __asm__ volatile("fence iorw, iorw" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
}

static void wafer_ifp_copy_spm_bytes(uint64_t destination, uint64_t source,
                                    uint32_t bytes) {
  volatile uint8_t *destination_mapping =
      (volatile uint8_t *)(void *)get_spm_memory_mapping(destination);
  const volatile uint8_t *source_mapping =
      (const volatile uint8_t *)(const void *)get_spm_memory_mapping(source);
  for (uint32_t index = 0; index < bytes; ++index)
    destination_mapping[index] = source_mapping[index];
  __asm__ volatile("fence iorw, iorw" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
  __asm__ volatile("sync.is" ::: "memory");
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
  /*
   * The four RDMA seeds and the tested instruction use independent engines.
   * Complete this real producer-to-consumer boundary before any instruction
   * reads or overwrites the seeded SPM slots.
   */
  wafer_ifp_wait_worker0_drain();
}

static uint32_t wafer_ifp_fp_format(const WaferIFPDescriptor *descriptor) {
  switch (descriptor->dtype) {
  case WAFER_IFP_F16:
  case WAFER_IFP_I8_TO_F16:
  case WAFER_IFP_BF16_TO_F16:
  case WAFER_IFP_F16_TO_BF16:
  case WAFER_IFP_F16_TO_I16:
    return Fmt_FP16;
  case WAFER_IFP_F32:
    return Fmt_FP32;
  case WAFER_IFP_BF16:
  case WAFER_IFP_I8_TO_BF16:
    return Fmt_BF16;
  default:
    return Fmt_UNUSED;
  }
}

static int
wafer_ifp_dispatch_reduce_capability(const WaferIFPDescriptor *descriptor,
                                     uint64_t input, uint64_t output,
                                     uint32_t format) {
  enum {
    WAFER_IFP_REDUCE_MATRIX_BASE = 144,
    WAFER_IFP_REDUCE_MATRIX_END = 191,
    WAFER_IFP_REDUCE_CX_BASE = 192,
    WAFER_IFP_REDUCE_CX_END = 195,
  };
  uint32_t operation;
  uint32_t dimension;
  uint32_t n;
  uint32_t h;
  uint32_t w;
  uint32_t c;
  if (descriptor->id >= WAFER_IFP_REDUCE_MATRIX_BASE &&
      descriptor->id <= WAFER_IFP_REDUCE_MATRIX_END) {
    const uint32_t offset =
        descriptor->id - WAFER_IFP_REDUCE_MATRIX_BASE;
    static const uint32_t dimensions[] = {0U, 1U, 2U, 4U};
    operation = offset / 12U;
    dimension = dimensions[offset % 4U];
    n = 1U;
    if (dimension == 1U) {
      h = 1U;
      w = 4U;
      c = 64U;
    } else if (dimension == 0U) {
      h = 2U;
      w = 3U;
      c = 65U;
    } else {
      h = 3U;
      w = 2U;
      c = 65U;
    }
  } else if (descriptor->id >= WAFER_IFP_REDUCE_CX_BASE &&
             descriptor->id <= WAFER_IFP_REDUCE_CX_END) {
    operation = descriptor->id - WAFER_IFP_REDUCE_CX_BASE;
    dimension = 0U;
    n = h = 1U;
    w = 4U;
    c = 8U;
  } else {
    return 0;
  }

  switch (operation) {
  case 0:
    wafer_tx81_reduce_sum(input, output, dimension, n, h, w, c, format);
    break;
  case 1:
    wafer_tx81_reduce_avg(input, output, dimension, n, h, w, c, format);
    break;
  case 2:
    wafer_tx81_reduce_max(input, output, dimension, n, h, w, c, format);
    break;
  case 3:
    wafer_tx81_reduce_min(input, output, dimension, n, h, w, c, format);
    break;
  default:
    return 0;
  }
  return 1;
}

static int
wafer_ifp_dispatch_pool_capability(const WaferIFPDescriptor *descriptor,
                                   uint64_t input, uint64_t output,
                                   uint32_t format) {
  enum {
    WAFER_IFP_POOL_SYMMETRIC_BASE = 204,
    WAFER_IFP_POOL_SYMMETRIC_END = 221,
    WAFER_IFP_POOL_ASYMMETRIC_BASE = 222,
    WAFER_IFP_POOL_ASYMMETRIC_END = 227,
    WAFER_IFP_POOL_PADDED_BASE = 228,
    WAFER_IFP_POOL_PADDED_END = 233,
    WAFER_IFP_POOL_TIE_BASE = 234,
    WAFER_IFP_POOL_TIE_END = 235,
  };
  uint32_t operation;
  uint32_t src_h;
  uint32_t src_w;
  uint32_t dst_h;
  uint32_t dst_w;
  uint32_t pad_top;
  uint32_t pad_bottom;
  uint32_t pad_left;
  uint32_t pad_right;
  uint32_t kernel_x;
  uint32_t kernel_y;
  uint32_t stride_x;
  uint32_t stride_y;
  if (descriptor->id >= WAFER_IFP_POOL_SYMMETRIC_BASE &&
      descriptor->id <= WAFER_IFP_POOL_SYMMETRIC_END) {
    operation = (descriptor->id - WAFER_IFP_POOL_SYMMETRIC_BASE) / 3U;
    src_h = 2;
    src_w = 4;
    dst_h = 1;
    dst_w = 2;
    pad_top = pad_bottom = pad_left = pad_right = 0;
    kernel_x = kernel_y = stride_x = stride_y = 2;
  } else if (descriptor->id >= WAFER_IFP_POOL_ASYMMETRIC_BASE &&
             descriptor->id <= WAFER_IFP_POOL_ASYMMETRIC_END) {
    operation = descriptor->id - WAFER_IFP_POOL_ASYMMETRIC_BASE;
    src_h = 3;
    src_w = 5;
    dst_h = dst_w = 2;
    pad_top = pad_bottom = pad_left = pad_right = 0;
    kernel_x = 3;
    kernel_y = 2;
    stride_x = 2;
    stride_y = 1;
  } else if (descriptor->id >= WAFER_IFP_POOL_PADDED_BASE &&
             descriptor->id <= WAFER_IFP_POOL_PADDED_END) {
    operation = descriptor->id - WAFER_IFP_POOL_PADDED_BASE;
    src_h = 2;
    src_w = 3;
    dst_h = dst_w = 2;
    pad_top = 1;
    pad_bottom = 0;
    pad_left = pad_right = 1;
    kernel_x = 3;
    kernel_y = 2;
    stride_x = 2;
    stride_y = 1;
  } else if (descriptor->id >= WAFER_IFP_POOL_TIE_BASE &&
             descriptor->id <= WAFER_IFP_POOL_TIE_END) {
    operation = descriptor->id == WAFER_IFP_POOL_TIE_BASE ? 3U : 5U;
    src_h = 2;
    src_w = 4;
    dst_h = 1;
    dst_w = 2;
    pad_top = pad_bottom = pad_left = pad_right = 0;
    kernel_x = kernel_y = stride_x = stride_y = 2;
  } else {
    return 0;
  }

  const uint32_t value_elements = dst_h * dst_w * 64U;
  const uint32_t value_bytes =
      value_elements * (format == Fmt_FP32 ? 4U : 2U);
  const uint64_t index_output =
      output + ((value_bytes + 255U) & ~UINT32_C(255));
  switch (operation) {
  case 0:
    wafer_tx81_pool_avg(
        input, output, OP_FUNC_CGRATensor_PoolOp_T_T_avg, 1, src_h, src_w, 64,
        1, dst_h, dst_w, 64, pad_top, pad_bottom, pad_left, pad_right,
        kernel_x, kernel_y, stride_x, stride_y, format);
    break;
  case 1:
    wafer_tx81_pool_sum(
        input, output, OP_FUNC_CGRATensor_PoolOp_T_T_sum, 1, src_h, src_w, 64,
        1, dst_h, dst_w, 64, pad_top, pad_bottom, pad_left, pad_right,
        kernel_x, kernel_y, stride_x, stride_y, format);
    break;
  case 2:
    wafer_tx81_pool_max(
        input, output, OP_FUNC_CGRATensor_PoolOp_T_T_max, 1, src_h, src_w, 64,
        1, dst_h, dst_w, 64, pad_top, pad_bottom, pad_left, pad_right,
        kernel_x, kernel_y, stride_x, stride_y, format);
    break;
  case 3:
    wafer_tx81_pool_indexedmax(
        input, output, index_output,
        OP_FUNC_CGRATensor_PoolOp_T_T_indexedmax, 1, src_h, src_w, 64, 1,
        dst_h, dst_w, 64, pad_top, pad_bottom, pad_left, pad_right, kernel_x,
        kernel_y, stride_x, stride_y, format);
    break;
  case 4:
    wafer_tx81_pool_min(
        input, output, OP_FUNC_CGRATensor_PoolOp_T_T_min, 1, src_h, src_w, 64,
        1, dst_h, dst_w, 64, pad_top, pad_bottom, pad_left, pad_right,
        kernel_x, kernel_y, stride_x, stride_y, format);
    break;
  case 5:
    wafer_tx81_pool_indexedmin(
        input, output, index_output,
        OP_FUNC_CGRATensor_PoolOp_T_T_indexedmin, 1, src_h, src_w, 64, 1,
        dst_h, dst_w, 64, pad_top, pad_bottom, pad_left, pad_right, kernel_x,
        kernel_y, stride_x, stride_y, format);
    break;
  default:
    return 0;
  }
  return 1;
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

  if (wafer_ifp_dispatch_reduce_capability(descriptor, input_a, output, format))
    goto wafer_ifp_dispatch_complete;
  if (wafer_ifp_dispatch_pool_capability(descriptor, input_a, output, format))
    goto wafer_ifp_dispatch_complete;

  switch (descriptor->id) {
  case WAFER_IFP_CASE_CT_NEG_F16:
  case WAFER_IFP_CASE_CT_NEG_BF16:
    wafer_tx81_elementwise_neg(input_a, output, elements, format);
    break;
  case WAFER_IFP_CASE_CT_ADD_F16:
  case WAFER_IFP_CASE_CT_ADD_BF16:
    wafer_tx81_elementwise_add(input_a, input_b, output, elements, format);
    break;
  case WAFER_IFP_CASE_CT_ADD_F16_TAIL130:
    wafer_tx81_elementwise_add(input_a, input_b, output, 130, Fmt_FP16);
    break;
  case WAFER_IFP_CASE_CT_ADD_BF16_TAIL130:
    wafer_tx81_elementwise_add(input_a, input_b, output, 130, Fmt_BF16);
    break;
  case WAFER_IFP_CASE_CT_ADD_F32:
    wafer_tx81_elementwise_add(input_a, input_b, output, elements, Fmt_FP32);
    break;
  case WAFER_IFP_CASE_CT_ADD_SPECIAL_F16:
    wafer_tx81_elementwise_add(input_a, input_b, output, elements, Fmt_FP16);
    break;
  case WAFER_IFP_CASE_CT_ADD_SPECIAL_BF16:
    wafer_tx81_elementwise_add(input_a, input_b, output, elements, Fmt_BF16);
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
    wafer_tx81_mask_move(input_a, (uint32_t)auxiliary, output, elements,
                         format);
    record[WAFER_IFP_REC_STEP_FLAGS] |= WAFER_IFP_STEP_MASK_MOVE_ISSUED;
    break;
  case WAFER_IFP_CASE_GEMM_F16:
    wafer_tx81_gemm(input_a, input_b, output, 1, 16, 16, 1, Fmt_FP16);
    break;
  case WAFER_IFP_CASE_GEMM_F16_ACCUM_ROUND:
    wafer_tx81_gemm(input_a, input_b, output, 1, 16, 16, 1, Fmt_FP16);
    break;
  case WAFER_IFP_CASE_GEMM_F16_M4:
    wafer_tx81_gemm(input_a, input_b, output, 4, 16, 16, 1, Fmt_FP16);
    break;
  case WAFER_IFP_CASE_GEMM_F16_BATCH2_M8:
    wafer_tx81_gemm(input_a, input_b, output, 8, 16, 16, 2, Fmt_FP16);
    break;
  case WAFER_IFP_CASE_GEMM_F16_N17:
    wafer_tx81_gemm(input_a, input_b, output, 1, 16, 17, 1, Fmt_FP16);
    break;
  case WAFER_IFP_CASE_GEMM_F16_K17:
    wafer_tx81_gemm(input_a, input_b, output, 1, 17, 16, 1, Fmt_FP16);
    break;
  case WAFER_IFP_CASE_GEMM_F16_N65:
    wafer_tx81_gemm(input_a, input_b, output, 1, 16, 65, 1, Fmt_FP16);
    break;
  case WAFER_IFP_CASE_GEMM_F16_ORIENTED_NT:
    wafer_tx81_gemm_oriented_v2(
        input_a, input_b, output, 4, 16, 8, 1, Fmt_FP16,
        WAFER_IFP_GEMM_ORIENTATION_NORMAL,
        WAFER_IFP_GEMM_ORIENTATION_TRANSPOSE);
    break;
  case WAFER_IFP_CASE_GEMM_BF16_ORIENTED_NT:
    wafer_tx81_gemm_oriented_v2(
        input_a, input_b, output, 4, 16, 8, 1, Fmt_BF16,
        WAFER_IFP_GEMM_ORIENTATION_NORMAL,
        WAFER_IFP_GEMM_ORIENTATION_TRANSPOSE);
    break;
  case WAFER_IFP_CASE_GEMM_F16_ORIENTED_TN:
    wafer_tx81_gemm_oriented_v2(
        input_a, input_b, output, 4, 16, 8, 1, Fmt_FP16,
        WAFER_IFP_GEMM_ORIENTATION_TRANSPOSE,
        WAFER_IFP_GEMM_ORIENTATION_NORMAL);
    break;
  case WAFER_IFP_CASE_GEMM_F16_ORIENTED_TT:
    wafer_tx81_gemm_oriented_v2(
        input_a, input_b, output, 4, 16, 8, 1, Fmt_FP16,
        WAFER_IFP_GEMM_ORIENTATION_TRANSPOSE,
        WAFER_IFP_GEMM_ORIENTATION_TRANSPOSE);
    break;
  case WAFER_IFP_CASE_GEMM_F16_PSUM: {
    TsmNeInstr instruction = {0};
    TsmGemm *gemm = TsmNewGemm();
    gemm->AddInput(&instruction, input_a, input_b, Fmt_FP16);
    gemm->ConfigMKN(&instruction, 1, 16, 16);
    gemm->ConfigBatch(&instruction, 1, 1);
    gemm->SetTransflag(&instruction, 0, 1);
    gemm->SetPsum(&instruction, 1, auxiliary, Fmt_FP16);
    gemm->SetQuant(&instruction, 0, 0, 0, 0);
    gemm->AddBias(&instruction, 0, 0);
    gemm->SetNegativeAxisScale(&instruction, 0, 0);
    gemm->SetPositiveAxisScale(&instruction, 0, 0);
    gemm->DisableRelu(&instruction);
    gemm->DisableLeakyRelu(&instruction);
    gemm->AddOutput(&instruction, output, Fmt_FP16);
    (void)TsmExecute(&instruction);
    TsmDeleteGemm(gemm);
    break;
  }
  case WAFER_IFP_CASE_GEMM_BF16:
  case WAFER_IFP_CASE_GEMM_BF16_ACCUM_ROUND:
    wafer_tx81_gemm(input_a, input_b, output, 1, 16, 16, 1, Fmt_BF16);
    break;
  case WAFER_IFP_CASE_GEMM_BF16_BATCH2_M8:
    wafer_tx81_gemm(input_a, input_b, output, 8, 16, 16, 2, Fmt_BF16);
    break;
  case WAFER_IFP_CASE_GEMM_BF16_K17:
    wafer_tx81_gemm(input_a, input_b, output, 1, 17, 16, 1, Fmt_BF16);
    break;
  case WAFER_IFP_CASE_GEMM_BF16_N65:
    wafer_tx81_gemm(input_a, input_b, output, 1, 16, 65, 1, Fmt_BF16);
    break;
  case WAFER_IFP_CASE_TDMA_PAD_F16:
    wafer_tx81_tdma_pad(input_a, output, 1, 2, 2, 64, 1, 4, 4, 64, 1, 1, 1,
                        1, Fmt_FP16);
    break;
  case WAFER_IFP_CASE_TDMA_IMG2COL_F16:
  case WAFER_IFP_CASE_TDMA_IMG2COL_BF16:
    wafer_tx81_tdma_img2col(input_a, output, 1, 3, 3, 64, 1, 4, 4, 64, 0, 0, 0,
                            0, 2, 2, 1, 1, format);
    break;
  case WAFER_IFP_CASE_CONV_F16:
  case WAFER_IFP_CASE_CONV_BF16:
    wafer_tx81_conv(input_a, input_b, output, 0, 1, 1, 3, 4, 1, 1, 4, 4, 1, 1,
                    2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 1, 1, format);
    break;
  case WAFER_IFP_CASE_POOL_F16:
  case WAFER_IFP_CASE_POOL_BF16:
    wafer_tx81_pool_max(
        input_a, output, OP_FUNC_CGRATensor_PoolOp_T_T_max, 1, 2, 4, 64, 1, 1,
        2, 64, 0, 0, 0, 0, 2, 2, 2, 2, format);
    break;
  case WAFER_IFP_CASE_POOL_AVG_F16:
    wafer_tx81_pool_avg(
        input_a, output, OP_FUNC_CGRATensor_PoolOp_T_T_avg, 1, 2, 4, 64, 1, 1,
        2, 64, 0, 0, 0, 0, 2, 2, 2, 2, Fmt_FP16);
    break;
  case WAFER_IFP_CASE_POOL_SUM_F16:
    wafer_tx81_pool_sum(
        input_a, output, OP_FUNC_CGRATensor_PoolOp_T_T_sum, 1, 2, 4, 64, 1, 1,
        2, 64, 0, 0, 0, 0, 2, 2, 2, 2, Fmt_FP16);
    break;
  case WAFER_IFP_CASE_POOL_MIN_F16:
    wafer_tx81_pool_min(
        input_a, output, OP_FUNC_CGRATensor_PoolOp_T_T_min, 1, 2, 4, 64, 1, 1,
        2, 64, 0, 0, 0, 0, 2, 2, 2, 2, Fmt_FP16);
    break;
  case WAFER_IFP_CASE_POOL_INDEXED_MIN_F16:
    wafer_tx81_pool_indexedmin(
        input_a, output, output + 256U,
        OP_FUNC_CGRATensor_PoolOp_T_T_indexedmin, 1, 2, 4, 64, 1, 1, 2, 64,
        0, 0, 0, 0, 2, 2, 2, 2, Fmt_FP16);
    break;
  case WAFER_IFP_CASE_POOL_INDEXED_MAX_F16_ASYMMETRIC:
    wafer_tx81_pool_indexedmax(
        input_a, output, output + 512U,
        OP_FUNC_CGRATensor_PoolOp_T_T_indexedmax, 1, 3, 5, 64, 1, 2, 2, 64,
        0, 0, 0, 0, 3, 2, 2, 1, Fmt_FP16);
    break;
  case WAFER_IFP_CASE_UNPOOL_F16:
  case WAFER_IFP_CASE_UNPOOL_MASK_BF16:
  case WAFER_IFP_CASE_UNPOOL_MASK_F32:
    wafer_tx81_pool_indexedmax(
        input_a, input_b, auxiliary,
        OP_FUNC_CGRATensor_PoolOp_T_T_indexedmax, 1, 2, 2, 64, 1, 1, 1, 64, 0,
        0, 0, 0, 2, 2, 2, 2, format);
    wafer_tx81_unpool_mask(
        input_b, output, OP_FUNC_CGRATensor_DataMoveOp_T_T_maskunpool,
        (uint32_t)auxiliary, 1, 1, 1, 64, 1, 2, 2, 64, 2, 2, 2, 2, format);
    break;
  case WAFER_IFP_CASE_UNPOOL_INDEX_F16:
  case WAFER_IFP_CASE_UNPOOL_INDEX_BF16_OBSERVED:
  case WAFER_IFP_CASE_UNPOOL_INDEX_F32_OBSERVED:
    wafer_tx81_pool_indexedmax(
        input_a, input_b, auxiliary,
        OP_FUNC_CGRATensor_PoolOp_T_T_indexedmax, 1, 2, 2, 64, 1, 1, 1, 64, 0,
        0, 0, 0, 2, 2, 2, 2, format);
    wafer_tx81_unpool_unpool(
        input_b, output, OP_FUNC_CGRATensor_DataMoveOp_T_T_unpool,
        (uint32_t)auxiliary, 1, 1, 1, 64, 1, 2, 2, 64, 2, 2, 2, 2, format);
    break;
  case WAFER_IFP_CASE_UNPOOL_AVG_F16:
  case WAFER_IFP_CASE_UNPOOL_AVG_BF16:
  case WAFER_IFP_CASE_UNPOOL_AVG_F32:
    wafer_tx81_unpool_avg(
        input_a, output, OP_FUNC_CGRATensor_DataMoveOp_T_T_unpool_avg, 0U, 1,
        1, 1, 64, 1, 2, 2, 64, 2, 2, 2, 2, format);
    break;
  case WAFER_IFP_CASE_UNPOOL_INDEX_F16_ASYMMETRIC_OBSERVED:
    wafer_tx81_pool_indexedmax(
        input_a, input_b, auxiliary,
        OP_FUNC_CGRATensor_PoolOp_T_T_indexedmax, 1, 3, 5, 64, 1, 2, 2, 64, 0,
        0, 0, 0, 3, 2, 2, 1, Fmt_FP16);
    wafer_tx81_unpool_unpool(
        input_b, output, OP_FUNC_CGRATensor_DataMoveOp_T_T_unpool,
        (uint32_t)auxiliary, 1, 2, 2, 64, 1, 3, 5, 64, 3, 2, 2, 1, Fmt_FP16);
    break;
  case WAFER_IFP_CASE_UNPOOL_INDEX_F16_REPEATED_OVERLAP_OBSERVED:
    wafer_tx81_pool_indexedmax(
        input_a, input_b, auxiliary,
        OP_FUNC_CGRATensor_PoolOp_T_T_indexedmax, 1, 3, 5, 64, 1, 2, 2, 64, 0,
        0, 0, 0, 3, 2, 2, 1, Fmt_FP16);
    wafer_ifp_wait_worker0_drain();
    wafer_ifp_copy_spm_bytes(
        input_b, input_a + WAFER_IFP_REPEATED_SENTINEL_OFFSET,
        WAFER_IFP_REPEATED_SENTINEL_BYTES);
    record[WAFER_IFP_REC_STEP_FLAGS] |=
        WAFER_IFP_STEP_REPEATED_OVERLAP_VALUES_STAGED;
    wafer_tx81_unpool_unpool(
        input_b, output, OP_FUNC_CGRATensor_DataMoveOp_T_T_unpool,
        (uint32_t)auxiliary, 1, 2, 2, 64, 1, 3, 5, 64, 3, 2, 2, 1, Fmt_FP16);
    break;
  case WAFER_IFP_CASE_UNPOOL_MASK_F16_ASYMMETRIC:
    wafer_tx81_pool_indexedmax(
        input_a, input_b, auxiliary,
        OP_FUNC_CGRATensor_PoolOp_T_T_indexedmax, 1, 3, 5, 64, 1, 2, 2, 64, 0,
        0, 0, 0, 3, 2, 2, 1, Fmt_FP16);
    wafer_tx81_unpool_mask(
        input_b, output, OP_FUNC_CGRATensor_DataMoveOp_T_T_maskunpool,
        (uint32_t)auxiliary, 1, 2, 2, 64, 1, 3, 5, 64, 3, 2, 2, 1, Fmt_FP16);
    break;
  case WAFER_IFP_CASE_UNPOOL_MASK_F16_REPEATED_OVERLAP_OBSERVED:
    wafer_tx81_pool_indexedmax(
        input_a, input_b, auxiliary,
        OP_FUNC_CGRATensor_PoolOp_T_T_indexedmax, 1, 3, 5, 64, 1, 2, 2, 64, 0,
        0, 0, 0, 3, 2, 2, 1, Fmt_FP16);
    wafer_ifp_wait_worker0_drain();
    wafer_ifp_copy_spm_bytes(
        input_b, input_a + WAFER_IFP_REPEATED_SENTINEL_OFFSET,
        WAFER_IFP_REPEATED_SENTINEL_BYTES);
    record[WAFER_IFP_REC_STEP_FLAGS] |=
        WAFER_IFP_STEP_REPEATED_OVERLAP_VALUES_STAGED;
    wafer_tx81_unpool_mask(
        input_b, output, OP_FUNC_CGRATensor_DataMoveOp_T_T_maskunpool,
        (uint32_t)auxiliary, 1, 2, 2, 64, 1, 3, 5, 64, 3, 2, 2, 1, Fmt_FP16);
    break;
  case WAFER_IFP_CASE_UNPOOL_AVG_F16_ASYMMETRIC_OBSERVED:
    wafer_tx81_unpool_avg(
        input_a, output, OP_FUNC_CGRATensor_DataMoveOp_T_T_unpool_avg, 0U, 1,
        2, 2, 64, 1, 3, 5, 64, 3, 2, 2, 1, Fmt_FP16);
    break;
  case WAFER_IFP_CASE_PERIPHERAL_ARGMAX_F16:
    wafer_tx81_peripheral_argmax(
        input_a, output, output + 4,
        OP_FUNC_CGRATensor_PeriOp_V_V_argmax, elements, Fmt_FP16, 0, 0, 0, 0);
    break;
  case WAFER_IFP_CASE_PERIPHERAL_ARGMIN_F16:
  case WAFER_IFP_CASE_PERIPHERAL_ARGMIN_NEGATIVE_F16_OBSERVED:
    wafer_tx81_peripheral_argmin(
        input_a, output, output + 4,
        OP_FUNC_CGRATensor_PeriOp_V_V_argmin, elements, Fmt_FP16, 0, 0, 0, 0);
    break;
  case WAFER_IFP_CASE_PERIPHERAL_ARGMIN_TIE_F16_OBSERVED:
  case WAFER_IFP_CASE_PERIPHERAL_ARGMIN_NAN_F16_OBSERVED:
    wafer_tx81_peripheral_argmin(
        input_a, output, output + 4,
        OP_FUNC_CGRATensor_PeriOp_V_V_argmin, elements, Fmt_FP16, 0, 0, 0, 0);
    wafer_ifp_wait_worker0_drain();
    wafer_ifp_copy_spm_bytes(auxiliary, input_a, elements * sizeof(uint16_t));
    record[WAFER_IFP_REC_STEP_FLAGS] |=
        WAFER_IFP_STEP_ARGMIN_INPUT_SNAPSHOTTED;
    break;
  case WAFER_IFP_CASE_PERIPHERAL_LUT16_F16:
    wafer_tx81_peripheral_lut16(
        input_a, input_b, output, OP_FUNC_CGRATensor_PeriOp_V_V_lut16,
        elements, Fmt_FP16, elements, 0, 0, 0);
    break;
  case WAFER_IFP_CASE_PERIPHERAL_BILINEAR_F16:
    wafer_tx81_peripheral_bilinear(
        input_a, output, OP_FUNC_CGRATensor_PeriOp_T_T_bilinear, elements,
        Fmt_FP16, 1, 1, 2, 64, 1, 1, 2, 64, 0, 0, 0, 0);
    break;
  case WAFER_IFP_CASE_PERIPHERAL_FACTORIZE_F32_OBSERVED: {
    TsmPeripheralInstr instruction = {0};
    TsmPeripheral *peripheral = TsmNewPeripheral();
    peripheral->Factorize(&instruction, input_a, output, output + 512U,
                          output + 1024U, 32U);
    (void)TsmExecute(&instruction);
    TsmDeletePeripheral(peripheral);
    break;
  }
  case WAFER_IFP_CASE_PERIPHERAL_LUT32_OBSERVED:
    wafer_tx81_peripheral_lut32(
        input_a, input_b, output, OP_FUNC_CGRATensor_PeriOp_V_V_lut32,
        elements, Fmt_FP32, elements, 0, 0, 0);
    break;
  case WAFER_IFP_CASE_PERIPHERAL_RANDGEN_F16_OBSERVED:
    wafer_tx81_peripheral_rand_gen(
        input_a, input_b, output, output + 512U, output + 1024U,
        OP_FUNC_CGRATensor_PeriOp_V_rand_gen, elements, Fmt_FP16, 0, 0, 0,
        0);
    break;
  case WAFER_IFP_CASE_PERIPHERAL_ELEMMASK_F16_OBSERVED:
    wafer_tx81_peripheral_elem_mask(
        input_a, output, OP_FUNC_CGRATensor_PeriOp_V_V_elem_mask, elements,
        Fmt_FP16, 0, UINT32_C(0x00003c00), 50U, RND_STOCHASTIC);
    break;
  default:
    return 1;
  }
wafer_ifp_dispatch_complete:
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
      wafer_tx81_wdma(WAFER_IFP_SPM_OUTPUT,
                      output_ddr + WAFER_IFP_OUTPUT_DDR_OFFSET,
                      WAFER_IFP_SLOT_BYTES, WAFER_IFP_SLOT_BYTES, 0, 0, 0, 1,
                      1, 1, Fmt_UINT8);
      wafer_tx81_wdma(WAFER_IFP_SPM_AUX,
                      output_ddr + WAFER_IFP_AUX_DDR_OFFSET,
                      WAFER_IFP_SLOT_BYTES, WAFER_IFP_SLOT_BYTES, 0, 0, 0, 1,
                      1, 1, Fmt_UINT8);
      wafer_tx81_local_fence();
      record[WAFER_IFP_REC_STEP_FLAGS] |=
          WAFER_IFP_STEP_FINAL_FENCE_COMPLETED;
      if (descriptor->family == WAFER_IFP_CT_SELECT_COMPOSITE)
        record[WAFER_IFP_REC_STEP_FLAGS] |=
            WAFER_IFP_STEP_BIT2FP_COMPLETED;
    }
    record[WAFER_IFP_REC_STATUS] = status;
  }
  wafer_ifp_publish(record);
}
