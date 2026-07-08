#include "wafer_tx81_crt.h"

#ifndef CONFIG_NO_PLATFORM_HOOK_H
#define CONFIG_NO_PLATFORM_HOOK_H 1
#endif
#ifndef USING_RISCV
#define USING_RISCV 1
#endif

#include "instr_adapter.h"

static Data_Format wafer_format(uint32_t format) {
  return (Data_Format)format;
}

static RND_MODE wafer_rounding(uint32_t rounding_mode) {
  return (RND_MODE)rounding_mode;
}

static Data_Shape wafer_shape4(uint32_t n, uint32_t h, uint32_t w,
                               uint32_t c) {
  Data_Shape shape = {(uint16_t)n, (uint16_t)h, (uint16_t)w, (uint16_t)c};
  return shape;
}

static St_StrideIteration wafer_stride_iteration(uint32_t stride0,
                                                 uint32_t stride1,
                                                 uint32_t stride2,
                                                 uint32_t iteration0,
                                                 uint32_t iteration1,
                                                 uint32_t iteration2) {
  St_StrideIteration si = {stride0, iteration0, stride1,
                           iteration1, stride2, iteration2};
  return si;
}

static uint32_t wafer_format_bytes(uint32_t format) {
  switch ((Data_Format)format) {
  case Fmt_INT8:
  case Fmt_UINT8:
  case Fmt_BOOL:
    return 1;
  case Fmt_INT16:
  case Fmt_UINT16:
  case Fmt_FP16:
  case Fmt_BF16:
    return 2;
  case Fmt_INT32:
  case Fmt_UINT32:
  case Fmt_FP32:
  case Fmt_TF32:
    return 4;
  case Fmt_INT64:
  case Fmt_UINT64:
    return 8;
  default:
    return 1;
  }
}

static uint32_t wafer_elem_count_from_bytes(uint32_t bytes, uint32_t format) {
  uint32_t elem_bytes = wafer_format_bytes(format);
  return elem_bytes == 0 ? bytes : bytes / elem_bytes;
}

static uint64_t wafer_shape_elements(Data_Shape shape) {
  return (uint64_t)shape.n * shape.h * shape.w * shape.c;
}

static int32_t wafer_bilinear_scale(uint32_t src, uint32_t dst) {
  if (dst == 0)
    return 0;
  return (int32_t)(((uint64_t)src << 16) / dst);
}

static void wafer_execute_ct(CT_Param *instr) { (void)TsmExecute(instr); }
static void wafer_execute_ne(TsmNeInstr *instr) { (void)TsmExecute(instr); }
static void wafer_execute_rdma(TsmRdmaInstr *instr) { (void)TsmExecute(instr); }
static void wafer_execute_wdma(TsmWdmaInstr *instr) { (void)TsmExecute(instr); }
static void wafer_execute_td(TsmDataMoveInstr *instr) { (void)TsmExecute(instr); }

static void wafer_store_u32(uint64_t addr, uint32_t value) {
  *(volatile uint32_t *)(uintptr_t)addr = value;
}

static void wafer_store_value(uint64_t addr, uint32_t format, uint64_t value) {
  switch ((Data_Format)format) {
  case Fmt_INT8:
  case Fmt_UINT8:
  case Fmt_BOOL:
    *(volatile uint8_t *)(uintptr_t)addr = (uint8_t)value;
    return;
  case Fmt_INT16:
  case Fmt_UINT16:
  case Fmt_FP16:
  case Fmt_BF16:
    *(volatile uint16_t *)(uintptr_t)addr = (uint16_t)value;
    return;
  case Fmt_INT64:
  case Fmt_UINT64:
    *(volatile uint64_t *)(uintptr_t)addr = value;
    return;
  default:
    wafer_store_u32(addr, (uint32_t)value);
    return;
  }
}

void wafer_tx81_rdma(uint64_t src, uint64_t dst, uint32_t byte_count,
                     uint32_t inner_bytes, uint32_t stride0, uint32_t stride1,
                     uint32_t stride2, uint32_t iteration0,
                     uint32_t iteration1, uint32_t iteration2,
                     uint32_t format) {
  (void)byte_count;
  TsmRdmaInstr instr = {0};
  TsmRdma *rdma = TsmNewRdma();
  rdma->AddSrcDst(&instr, src, dst, wafer_format(format));
  rdma->ConfigStrideIteration(
      &instr, wafer_elem_count_from_bytes(inner_bytes, format), stride0,
      iteration0, stride1, iteration1, stride2, iteration2);
  wafer_execute_rdma(&instr);
  TsmDeleteRdma(rdma);
}

void wafer_tx81_wdma(uint64_t src, uint64_t dst, uint32_t byte_count,
                     uint32_t inner_bytes, uint32_t stride0, uint32_t stride1,
                     uint32_t stride2, uint32_t iteration0,
                     uint32_t iteration1, uint32_t iteration2,
                     uint32_t format) {
  (void)byte_count;
  TsmWdmaInstr instr = {0};
  TsmWdma *wdma = TsmNewWdma();
  wdma->AddSrcDst(&instr, src, dst, wafer_format(format));
  wdma->ConfigStrideIteration(
      &instr, wafer_elem_count_from_bytes(inner_bytes, format), stride0,
      iteration0, stride1, iteration1, stride2, iteration2);
  wafer_execute_wdma(&instr);
  TsmDeleteWdma(wdma);
}

void wafer_tx81_gather_scatter(
    uint64_t src, uint64_t dst, uint32_t byte_count, uint32_t inner_bytes,
    uint32_t src_stride0, uint32_t src_stride1, uint32_t src_stride2,
    uint32_t src_iteration0, uint32_t src_iteration1,
    uint32_t src_iteration2, uint32_t dst_stride0, uint32_t dst_stride1,
    uint32_t dst_stride2, uint32_t dst_iteration0,
    uint32_t dst_iteration1, uint32_t dst_iteration2) {
  (void)byte_count;
  TsmDataMoveInstr instr = {0};
  TsmDataMove *move = TsmNewDataMove();
  St_StrideIteration src_si =
      wafer_stride_iteration(src_stride0, src_stride1, src_stride2,
                             src_iteration0, src_iteration1, src_iteration2);
  St_StrideIteration dst_si =
      wafer_stride_iteration(dst_stride0, dst_stride1, dst_stride2,
                             dst_iteration0, dst_iteration1, dst_iteration2);
  move->GatherScatter(&instr, src, dst, inner_bytes, &src_si, &dst_si);
  wafer_execute_td(&instr);
  TsmDeleteDataMove(move);
}

void wafer_tx81_memset(uint64_t dst, uint32_t value, uint32_t elem_count,
                       uint32_t format) {
  TsmDataMoveInstr instr = {0};
  TsmPeripheral *peripheral = TsmNewPeripheral();
  St_StrideIteration si = {0};
  peripheral->Memset(&instr, dst, value, elem_count, &si,
                     wafer_format(format));
  wafer_execute_td(&instr);
  TsmDeletePeripheral(peripheral);
}

void wafer_tx81_bit2fp(uint64_t src, uint64_t dst, uint32_t elem_count,
                       uint32_t format) {
  TsmPeripheralInstr instr = {0};
  TsmPeripheral *peripheral = TsmNewPeripheral();
  peripheral->Bit2Fp(&instr, src, dst, elem_count, wafer_format(format));
  wafer_execute_ct(&instr);
  TsmDeletePeripheral(peripheral);
}

void wafer_tx81_mask_move(uint64_t src, uint64_t mask, uint64_t dst,
                          uint32_t elem_count, uint32_t format) {
  TsmMaskDataMoveInstr instr = {0};
  TsmMaskDataMove *move = TsmNewMaskDataMove();
  move->MaskMove(&instr, src, (uint32_t)mask, dst, elem_count,
                 wafer_format(format));
  wafer_execute_ct(&instr);
  TsmDeleteMaskDataMove(move);
}

#define WAFER_DEFINE_ARITH_UNARY(SYMBOL, METHOD)                              \
  void SYMBOL(uint64_t src, uint64_t dst, uint32_t elem_count,                 \
              uint32_t format) {                                               \
    TsmArithInstr instr = {0};                                                  \
    TsmArith *arith = TsmNewArith();                                            \
    arith->METHOD(&instr, src, dst, elem_count, wafer_format(format));          \
    wafer_execute_ct(&instr);                                                   \
    TsmDeleteArith(arith);                                                      \
  }

#define WAFER_DEFINE_ARITH_BINARY(SYMBOL, METHOD)                             \
  void SYMBOL(uint64_t lhs, uint64_t rhs, uint64_t dst,                        \
              uint32_t elem_count, uint32_t format) {                          \
    TsmArithInstr instr = {0};                                                  \
    TsmArith *arith = TsmNewArith();                                            \
    arith->METHOD(&instr, lhs, rhs, dst, elem_count, RND_NEAREST_EVEN,          \
                  wafer_format(format));                                        \
    wafer_execute_ct(&instr);                                                   \
    TsmDeleteArith(arith);                                                      \
  }

#define WAFER_DEFINE_RELATION(SYMBOL, METHOD, BOOL_METHOD)                    \
  void SYMBOL(uint64_t lhs, uint64_t rhs, uint64_t dst,                        \
              uint32_t elem_count, uint32_t format) {                          \
    TsmRelationInstr instr = {0};                                               \
    TsmRelation *relation = TsmNewRelation();                                   \
    if (format == Fmt_BOOL)                                                     \
      relation->BOOL_METHOD(&instr, lhs, rhs, dst, elem_count,                  \
                            wafer_format(format));                              \
    else                                                                        \
      relation->METHOD(&instr, lhs, rhs, dst, elem_count,                       \
                       wafer_format(format));                                   \
    wafer_execute_ct(&instr);                                                   \
    TsmDeleteRelation(relation);                                                \
  }

#define WAFER_DEFINE_LOGIC_UNARY(SYMBOL, METHOD, BOOL_METHOD)                 \
  void SYMBOL(uint64_t src, uint64_t dst, uint32_t elem_count,                 \
              uint32_t format) {                                               \
    TsmLogicInstr instr = {0};                                                  \
    TsmLogic *logic = TsmNewLogic();                                            \
    if (format == Fmt_BOOL)                                                     \
      logic->BOOL_METHOD(&instr, src, dst, elem_count);                         \
    else                                                                        \
      logic->METHOD(&instr, src, dst, elem_count, wafer_format(format));        \
    wafer_execute_ct(&instr);                                                   \
    TsmDeleteLogic(logic);                                                      \
  }

#define WAFER_DEFINE_LOGIC_BINARY(SYMBOL, METHOD, BOOL_METHOD)                \
  void SYMBOL(uint64_t lhs, uint64_t rhs, uint64_t dst,                        \
              uint32_t elem_count, uint32_t format) {                          \
    TsmLogicInstr instr = {0};                                                  \
    TsmLogic *logic = TsmNewLogic();                                            \
    if (format == Fmt_BOOL)                                                     \
      logic->BOOL_METHOD(&instr, lhs, rhs, dst, elem_count);                    \
    else                                                                        \
      logic->METHOD(&instr, lhs, rhs, dst, elem_count, wafer_format(format));   \
    wafer_execute_ct(&instr);                                                   \
    TsmDeleteLogic(logic);                                                      \
  }

#define WAFER_DEFINE_TRANS(SYMBOL, METHOD)                                     \
  void SYMBOL(uint64_t src, uint64_t dst, uint32_t elem_count,                 \
              uint32_t format) {                                               \
    TsmTranscendentalInstr instr = {0};                                         \
    TsmTranscendental *trans = TsmNewTranscendental();                         \
    trans->METHOD(&instr, src, dst, elem_count, wafer_format(format));          \
    wafer_execute_ct(&instr);                                                   \
    TsmDeleteTranscendental(trans);                                             \
  }

#define WAFER_DEFINE_ACTIVATION(SYMBOL, METHOD)                               \
  void SYMBOL(uint64_t src, uint64_t dst, uint32_t elem_count,                 \
              uint32_t format) {                                               \
    TsmActivationInstr instr = {0};                                             \
    TsmActivation *activation = TsmNewActivation();                             \
    activation->METHOD(&instr, src, dst, elem_count, wafer_format(format));     \
    wafer_execute_ct(&instr);                                                   \
    TsmDeleteActivation(activation);                                            \
  }

WAFER_DEFINE_ARITH_UNARY(wafer_tx81_elementwise_abs, AbsVV)
WAFER_DEFINE_ARITH_UNARY(wafer_tx81_elementwise_recip, RecipVV)
WAFER_DEFINE_ARITH_UNARY(wafer_tx81_elementwise_square, SquareVV)
WAFER_DEFINE_ARITH_UNARY(wafer_tx81_elementwise_sqrt, SqrtVV)
WAFER_DEFINE_ARITH_UNARY(wafer_tx81_elementwise_rsqrt, RsqrtVV)
WAFER_DEFINE_ARITH_UNARY(wafer_tx81_elementwise_neg, NegVV)
WAFER_DEFINE_ARITH_BINARY(wafer_tx81_elementwise_max, MaxVV)
WAFER_DEFINE_ARITH_BINARY(wafer_tx81_elementwise_min, MinVV)
WAFER_DEFINE_ARITH_BINARY(wafer_tx81_elementwise_add, AddVV)
WAFER_DEFINE_ARITH_BINARY(wafer_tx81_elementwise_sub, SubVV)
WAFER_DEFINE_ARITH_BINARY(wafer_tx81_elementwise_mul, MulVV)
WAFER_DEFINE_ARITH_BINARY(wafer_tx81_elementwise_div, DivVV)
WAFER_DEFINE_RELATION(wafer_tx81_elementwise_eq, EqualVV, BoolEqualVV)
WAFER_DEFINE_RELATION(wafer_tx81_elementwise_ne, UnEqualVV, BoolUnEqualVV)
WAFER_DEFINE_RELATION(wafer_tx81_elementwise_ge, GreaterEqualVV,
                      BoolGreaterEqualVV)
WAFER_DEFINE_RELATION(wafer_tx81_elementwise_gt, GreaterVV, BoolGreaterVV)
WAFER_DEFINE_RELATION(wafer_tx81_elementwise_le, LessEqualVV,
                      BoolLessEqualVV)
WAFER_DEFINE_RELATION(wafer_tx81_elementwise_lt, LessThenVV, BoolLessThenVV)
WAFER_DEFINE_LOGIC_UNARY(wafer_tx81_elementwise_logic_not, NotV, BoolNotV)
WAFER_DEFINE_LOGIC_BINARY(wafer_tx81_elementwise_logic_and, AndVV, BoolAndV)
WAFER_DEFINE_LOGIC_BINARY(wafer_tx81_elementwise_logic_or, OrVV, BoolOrV)
WAFER_DEFINE_LOGIC_BINARY(wafer_tx81_elementwise_logic_xor, XorVV, BoolXorV)
WAFER_DEFINE_TRANS(wafer_tx81_elementwise_log2, Log2)
WAFER_DEFINE_TRANS(wafer_tx81_elementwise_ln, Ln)
WAFER_DEFINE_TRANS(wafer_tx81_elementwise_pow2, Pow2)
WAFER_DEFINE_TRANS(wafer_tx81_elementwise_exp, Exp)
WAFER_DEFINE_TRANS(wafer_tx81_elementwise_exp_lp, Explp)
WAFER_DEFINE_TRANS(wafer_tx81_elementwise_sin, Sin)
WAFER_DEFINE_TRANS(wafer_tx81_elementwise_cos, Cos)
WAFER_DEFINE_ACTIVATION(wafer_tx81_elementwise_tanh, Tanh)
WAFER_DEFINE_ACTIVATION(wafer_tx81_elementwise_sigmoid, Sigmoid)
WAFER_DEFINE_ACTIVATION(wafer_tx81_elementwise_relu, Relu)
WAFER_DEFINE_ACTIVATION(wafer_tx81_elementwise_satrelu, Satrelu)
WAFER_DEFINE_ACTIVATION(wafer_tx81_elementwise_leakyrelu, Leakyrelu)
WAFER_DEFINE_ACTIVATION(wafer_tx81_elementwise_softplus, Softplus)

#define WAFER_DEFINE_REDUCE(SYMBOL, METHOD)                                   \
  void SYMBOL(uint64_t src, uint64_t dst, uint32_t dim, uint32_t n,            \
              uint32_t h, uint32_t w, uint32_t c, uint32_t format) {           \
    TsmReduceInstr instr = {0};                                                \
    TsmReduce *reduce = TsmNewReduce();                                        \
    reduce->METHOD(&instr, src, dst, dim, wafer_shape4(n, h, w, c),            \
                   wafer_format(format));                                      \
    wafer_execute_ct(&instr);                                                  \
    TsmDeleteReduce(reduce);                                                   \
  }

WAFER_DEFINE_REDUCE(wafer_tx81_reduce_sum, ReduceSum)
WAFER_DEFINE_REDUCE(wafer_tx81_reduce_max, ReduceMax)
WAFER_DEFINE_REDUCE(wafer_tx81_reduce_min, ReduceMin)
WAFER_DEFINE_REDUCE(wafer_tx81_reduce_avg, ReduceAvg)

#define WAFER_DEFINE_CONVERT_ZP(SYMBOL, METHOD)                               \
  void SYMBOL(uint64_t src, uint64_t dst, uint32_t elem_count,                 \
              uint32_t zero_point, uint32_t rounding_mode) {                   \
    (void)rounding_mode;                                                       \
    TsmConvertInstr instr = {0};                                               \
    TsmConvert *convert = TsmNewConvert();                                     \
    convert->METHOD(&instr, src, zero_point, dst, elem_count);                 \
    wafer_execute_ct(&instr);                                                  \
    TsmDeleteConvert(convert);                                                 \
  }

#define WAFER_DEFINE_CONVERT_ROUND(SYMBOL, METHOD)                            \
  void SYMBOL(uint64_t src, uint64_t dst, uint32_t elem_count,                 \
              uint32_t zero_point, uint32_t rounding_mode) {                   \
    (void)zero_point;                                                          \
    TsmConvertInstr instr = {0};                                               \
    TsmConvert *convert = TsmNewConvert();                                     \
    convert->METHOD(&instr, src, dst, elem_count, wafer_rounding(rounding_mode)); \
    wafer_execute_ct(&instr);                                                  \
    TsmDeleteConvert(convert);                                                 \
  }

#define WAFER_DEFINE_CONVERT_PLAIN(SYMBOL, METHOD)                            \
  void SYMBOL(uint64_t src, uint64_t dst, uint32_t elem_count,                 \
              uint32_t zero_point, uint32_t rounding_mode) {                   \
    (void)zero_point;                                                          \
    (void)rounding_mode;                                                       \
    TsmConvertInstr instr = {0};                                               \
    TsmConvert *convert = TsmNewConvert();                                     \
    convert->METHOD(&instr, src, dst, elem_count);                             \
    wafer_execute_ct(&instr);                                                  \
    TsmDeleteConvert(convert);                                                 \
  }

WAFER_DEFINE_CONVERT_ZP(wafer_tx81_convert_int8_fp16, INT8_FP16)
WAFER_DEFINE_CONVERT_ZP(wafer_tx81_convert_int8_bf16, INT8_BF16)
WAFER_DEFINE_CONVERT_ZP(wafer_tx81_convert_int8_fp32, INT8_FP32)
WAFER_DEFINE_CONVERT_ZP(wafer_tx81_convert_int8_tf32, INT8_TF32)
WAFER_DEFINE_CONVERT_PLAIN(wafer_tx81_convert_int16_fp16, INT16_FP16)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_int16_bf16, INT16_BF16)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_int16_fp32, INT16_FP32)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_int16_tf32, INT16_TF32)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_int32_fp16, INT32_FP16)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_int32_bf16, INT32_BF16)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_int32_fp32, INT32_FP32)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_int32_tf32, INT32_TF32)
WAFER_DEFINE_CONVERT_PLAIN(wafer_tx81_convert_bf16_int8, BF16_INT8)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_bf16_int16, BF16_INT16)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_bf16_int32, BF16_INT32)
WAFER_DEFINE_CONVERT_PLAIN(wafer_tx81_convert_bf16_fp16, BF16_FP16)
WAFER_DEFINE_CONVERT_PLAIN(wafer_tx81_convert_bf16_fp32, BF16_FP32)
WAFER_DEFINE_CONVERT_PLAIN(wafer_tx81_convert_bf16_tf32, BF16_TF32)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_fp16_int8, FP16_INT8)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_fp16_int16, FP16_INT16)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_fp16_int32, FP16_INT32)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_fp16_bf16, FP16_BF16)
WAFER_DEFINE_CONVERT_PLAIN(wafer_tx81_convert_fp16_fp32, FP16_FP32)
WAFER_DEFINE_CONVERT_PLAIN(wafer_tx81_convert_fp16_tf32, FP16_TF32)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_fp32_int8, FP32_INT8)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_fp32_int16, FP32_INT16)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_fp32_int32, FP32_INT32)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_fp32_fp16, FP32_FP16)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_fp32_bf16, FP32_BF16)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_fp32_tf32, FP32_TF32)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_tf32_int8, TF32_INT8)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_tf32_int16, TF32_INT16)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_tf32_int32, TF32_INT32)
WAFER_DEFINE_CONVERT_PLAIN(wafer_tx81_convert_tf32_fp16, TF32_FP16)
WAFER_DEFINE_CONVERT_ROUND(wafer_tx81_convert_tf32_bf16, TF32_BF16)
WAFER_DEFINE_CONVERT_PLAIN(wafer_tx81_convert_tf32_fp32, TF32_FP32)

void wafer_tx81_gemm(uint64_t lhs, uint64_t rhs, uint64_t dst, uint32_t m,
                     uint32_t k, uint32_t n, uint32_t batch_count,
                     uint32_t format) {
  TsmNeInstr instr = {0};
  TsmGemm *gemm = TsmNewGemm();
  gemm->AddInput(&instr, lhs, rhs, wafer_format(format));
  gemm->ConfigMKN(&instr, m, k, n);
  gemm->ConfigBatch(&instr, batch_count, batch_count);
  gemm->SetTransflag(&instr, 0, 0);
  gemm->SetPsum(&instr, 0, 0, Fmt_UNUSED);
  gemm->SetQuant(&instr, 0, 0, 0, 0);
  gemm->AddBias(&instr, 0, 0);
  gemm->SetNegativeAxisScale(&instr, 0, 0);
  gemm->SetPositiveAxisScale(&instr, 0, 0);
  gemm->DisableRelu(&instr);
  gemm->DisableLeakyRelu(&instr);
  gemm->AddOutput(&instr, dst, wafer_format(format));
  wafer_execute_ne(&instr);
  TsmDeleteGemm(gemm);
}

#define WAFER_CONFIGURE_CONV(OP, INSTR)                                       \
  do {                                                                         \
    OP->AddInput(INSTR, input, wafer_shape4(input_n, input_h, input_w, input_c), \
                 wafer_format(format));                                        \
    OP->AddWeight(INSTR, weight,                                               \
                  wafer_shape4(weight_n, weight_h, weight_w, weight_c),        \
                  wafer_format(format));                                       \
    OP->AddBias(INSTR, 0, 0);                                                  \
    OP->AddOutput(INSTR, dst,                                                  \
                  wafer_shape4(output_n, output_h, output_w, output_c),        \
                  wafer_format(format));                                       \
    OP->SetOpType(INSTR, (uint8_t)kind);                                       \
    OP->SetNegativeAxisScale(INSTR, 0, 0);                                     \
    OP->SetPositiveAxisScale(INSTR, 0, 0);                                     \
    OP->SetSparse(INSTR, 0, 0);                                                \
    OP->SetPsum(INSTR, 0, 0, Fmt_UNUSED);                                      \
    OP->SetPads(INSTR, pad_top, pad_bottom, pad_left, pad_right);              \
    OP->SetUnPads(INSTR, unpad_top, unpad_bottom, unpad_left, unpad_right);    \
    OP->SetKernelStrides(INSTR, kernel_x, kernel_y, stride_x, stride_y);       \
    OP->SetDilations(INSTR, dilation0, dilation1);                             \
    OP->SetQuant(INSTR, 0, 0, 0, 0);                                           \
    OP->DisableRelu(INSTR);                                                    \
    OP->DisableLeakyRelu(INSTR);                                               \
  } while (0)

void wafer_tx81_conv(uint64_t input, uint64_t weight, uint64_t dst,
                     uint32_t kind, uint32_t input_n, uint32_t input_h,
                     uint32_t input_w, uint32_t input_c, uint32_t weight_n,
                     uint32_t weight_h, uint32_t weight_w, uint32_t weight_c,
                     uint32_t output_n, uint32_t output_h, uint32_t output_w,
                     uint32_t output_c, uint32_t pad_top, uint32_t pad_bottom,
                     uint32_t pad_left, uint32_t pad_right,
                     uint32_t unpad_top, uint32_t unpad_bottom,
                     uint32_t unpad_left, uint32_t unpad_right,
                     uint32_t kernel_x, uint32_t kernel_y, uint32_t stride_x,
                     uint32_t stride_y, uint32_t dilation0,
                     uint32_t dilation1, uint32_t format) {
  TsmNeInstr instr = {0};
  TsmConv *conv = TsmNewConv();
  WAFER_CONFIGURE_CONV(conv, &instr);
  wafer_execute_ne(&instr);
  TsmDeleteConv(conv);
}

void wafer_tx81_depthwise_conv(
    uint64_t input, uint64_t weight, uint64_t dst, uint32_t kind,
    uint32_t input_n, uint32_t input_h, uint32_t input_w, uint32_t input_c,
    uint32_t weight_n, uint32_t weight_h, uint32_t weight_w,
    uint32_t weight_c, uint32_t output_n, uint32_t output_h,
    uint32_t output_w, uint32_t output_c, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right,
    uint32_t unpad_top, uint32_t unpad_bottom, uint32_t unpad_left,
    uint32_t unpad_right, uint32_t kernel_x, uint32_t kernel_y,
    uint32_t stride_x, uint32_t stride_y, uint32_t dilation0,
    uint32_t dilation1, uint32_t format) {
  TsmNeInstr instr = {0};
  TsmDepthwiseConv *conv = TsmNewDepthwiseConv();
  WAFER_CONFIGURE_CONV(conv, &instr);
  wafer_execute_ne(&instr);
  TsmDeleteDepthwiseConv(conv);
}

void wafer_tx81_backward_conv(
    uint64_t input, uint64_t weight, uint64_t dst, uint32_t kind,
    uint32_t input_n, uint32_t input_h, uint32_t input_w, uint32_t input_c,
    uint32_t weight_n, uint32_t weight_h, uint32_t weight_w,
    uint32_t weight_c, uint32_t output_n, uint32_t output_h,
    uint32_t output_w, uint32_t output_c, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right,
    uint32_t unpad_top, uint32_t unpad_bottom, uint32_t unpad_left,
    uint32_t unpad_right, uint32_t kernel_x, uint32_t kernel_y,
    uint32_t stride_x, uint32_t stride_y, uint32_t dilation0,
    uint32_t dilation1, uint32_t format) {
  TsmNeInstr instr = {0};
  TsmConv *conv = TsmNewConv();
  WAFER_CONFIGURE_CONV(conv, &instr);
  wafer_execute_ne(&instr);
  TsmDeleteConv(conv);
}

#define WAFER_DEFINE_POOL(SYMBOL, METHOD)                                     \
  void SYMBOL(uint64_t input, uint64_t dst, uint32_t kind, uint32_t src_n,     \
              uint32_t src_h, uint32_t src_w, uint32_t src_c,                 \
              uint32_t dst_n, uint32_t dst_h, uint32_t dst_w,                 \
              uint32_t dst_c, uint32_t pad_top, uint32_t pad_bottom,          \
              uint32_t pad_left, uint32_t pad_right, uint32_t kernel_x,        \
              uint32_t kernel_y, uint32_t stride_x, uint32_t stride_y,         \
              uint32_t format) {                                              \
    (void)kind;                                                               \
    (void)dst_n;                                                              \
    (void)dst_h;                                                              \
    (void)dst_w;                                                              \
    (void)dst_c;                                                              \
    TsmPoolInstr instr = {0};                                                 \
    TsmPool *pool = TsmNewPool();                                             \
    pool->METHOD(&instr, input, wafer_shape4(src_n, src_h, src_w, src_c),     \
                 dst, wafer_shape4(pad_top, pad_bottom, pad_left, pad_right), \
                 wafer_shape4(kernel_x, kernel_y, stride_x, stride_y),        \
                 wafer_format(format));                                       \
    wafer_execute_ct(&instr);                                                 \
    TsmDeletePool(pool);                                                      \
  }

#define WAFER_DEFINE_POOL_INDEXED(SYMBOL, METHOD)                             \
  void SYMBOL(uint64_t input, uint64_t value_dst, uint64_t index_dst,          \
              uint32_t kind, uint32_t src_n, uint32_t src_h,                  \
              uint32_t src_w, uint32_t src_c, uint32_t dst_n,                 \
              uint32_t dst_h, uint32_t dst_w, uint32_t dst_c,                 \
              uint32_t pad_top, uint32_t pad_bottom, uint32_t pad_left,        \
              uint32_t pad_right, uint32_t kernel_x, uint32_t kernel_y,        \
              uint32_t stride_x, uint32_t stride_y, uint32_t format) {         \
    (void)kind;                                                               \
    (void)dst_n;                                                              \
    (void)dst_h;                                                              \
    (void)dst_w;                                                              \
    (void)dst_c;                                                              \
    TsmPoolInstr instr = {0};                                                 \
    TsmPool *pool = TsmNewPool();                                             \
    pool->METHOD(&instr, input, wafer_shape4(src_n, src_h, src_w, src_c),     \
                 value_dst, index_dst,                                        \
                 wafer_shape4(pad_top, pad_bottom, pad_left, pad_right),      \
                 wafer_shape4(kernel_x, kernel_y, stride_x, stride_y),        \
                 wafer_format(format));                                       \
    wafer_execute_ct(&instr);                                                 \
    TsmDeletePool(pool);                                                      \
  }

WAFER_DEFINE_POOL(wafer_tx81_pool_avg, AvgPool)
WAFER_DEFINE_POOL(wafer_tx81_pool_sum, SumPool)
WAFER_DEFINE_POOL(wafer_tx81_pool_max, MaxPool)
WAFER_DEFINE_POOL(wafer_tx81_pool_min, MinPool)
WAFER_DEFINE_POOL_INDEXED(wafer_tx81_pool_indexedmax, IndexdMaxPool)
WAFER_DEFINE_POOL_INDEXED(wafer_tx81_pool_indexedmin, IndexdMinPool)

void wafer_tx81_unpool_unpool(uint64_t input, uint64_t dst, uint32_t kind,
                              uint32_t index, uint32_t src_n, uint32_t src_h,
                              uint32_t src_w, uint32_t src_c, uint32_t dst_n,
                              uint32_t dst_h, uint32_t dst_w, uint32_t dst_c,
                              uint32_t kernel_x, uint32_t kernel_y,
                              uint32_t stride_x, uint32_t stride_y,
                              uint32_t format) {
  (void)kind;
  (void)src_n;
  (void)src_h;
  (void)src_w;
  (void)src_c;
  TsmUnPoolInstr instr = {0};
  TsmUnPool *unpool = TsmNewUnPool();
  unpool->Unpool(&instr, input, index, dst,
                 wafer_shape4(dst_n, dst_h, dst_w, dst_c),
                 wafer_shape4(kernel_x, kernel_y, stride_x, stride_y),
                 wafer_format(format));
  wafer_execute_ct(&instr);
  TsmDeleteUnPool(unpool);
}

void wafer_tx81_unpool_avg(uint64_t input, uint64_t dst, uint32_t kind,
                           uint32_t index, uint32_t src_n, uint32_t src_h,
                           uint32_t src_w, uint32_t src_c, uint32_t dst_n,
                           uint32_t dst_h, uint32_t dst_w, uint32_t dst_c,
                           uint32_t kernel_x, uint32_t kernel_y,
                           uint32_t stride_x, uint32_t stride_y,
                           uint32_t format) {
  (void)kind;
  (void)index;
  (void)src_n;
  (void)src_h;
  (void)src_w;
  (void)src_c;
  TsmUnPoolInstr instr = {0};
  TsmUnPool *unpool = TsmNewUnPool();
  unpool->UnpoolAvg(&instr, input, dst,
                    wafer_shape4(dst_n, dst_h, dst_w, dst_c),
                    wafer_shape4(kernel_x, kernel_y, stride_x, stride_y),
                    wafer_format(format));
  wafer_execute_ct(&instr);
  TsmDeleteUnPool(unpool);
}

void wafer_tx81_unpool_mask(uint64_t input, uint64_t dst, uint32_t kind,
                            uint32_t index, uint32_t src_n, uint32_t src_h,
                            uint32_t src_w, uint32_t src_c, uint32_t dst_n,
                            uint32_t dst_h, uint32_t dst_w, uint32_t dst_c,
                            uint32_t kernel_x, uint32_t kernel_y,
                            uint32_t stride_x, uint32_t stride_y,
                            uint32_t format) {
  (void)kind;
  (void)src_n;
  (void)src_h;
  (void)src_w;
  (void)src_c;
  TsmUnPoolInstr instr = {0};
  TsmUnPool *unpool = TsmNewUnPool();
  unpool->UnpoolIdx(&instr, input, index, dst,
                    wafer_shape4(dst_n, dst_h, dst_w, dst_c),
                    wafer_shape4(kernel_x, kernel_y, stride_x, stride_y),
                    wafer_format(format));
  wafer_execute_ct(&instr);
  TsmDeleteUnPool(unpool);
}

void wafer_tx81_tdma_pad(uint64_t src, uint64_t dst, uint32_t src_n,
                         uint32_t src_h, uint32_t src_w, uint32_t src_c,
                         uint32_t dst_n, uint32_t dst_h, uint32_t dst_w,
                         uint32_t dst_c, uint32_t pad_top,
                         uint32_t pad_bottom, uint32_t pad_left,
                         uint32_t pad_right, uint32_t format) {
  TsmDataMoveInstr instr = {0};
  TsmDataMove *move = TsmNewDataMove();
  move->Pad(&instr, src, wafer_shape4(src_n, src_h, src_w, src_c), dst,
            wafer_shape4(dst_n, dst_h, dst_w, dst_c),
            wafer_shape4(pad_top, pad_bottom, pad_left, pad_right),
            wafer_format(format));
  wafer_execute_td(&instr);
  TsmDeleteDataMove(move);
}

void wafer_tx81_tdma_img2col(uint64_t src, uint64_t dst, uint32_t src_n,
                             uint32_t src_h, uint32_t src_w, uint32_t src_c,
                             uint32_t dst_n, uint32_t dst_h, uint32_t dst_w,
                             uint32_t dst_c, uint32_t pad_top,
                             uint32_t pad_bottom, uint32_t pad_left,
                             uint32_t pad_right, uint32_t kernel_x,
                             uint32_t kernel_y, uint32_t stride_x,
                             uint32_t stride_y, uint32_t format) {
  Data_Shape src_shape = wafer_shape4(src_n, src_h, src_w, src_c);
  Data_Shape dst_shape = wafer_shape4(dst_n, dst_h, dst_w, dst_c);
  TsmDataMoveInstr instr = {0};
  TsmDataMove *move = TsmNewDataMove();
  move->Img2col(&instr, src, src_shape, dst, dst_shape,
                wafer_shape_elements(src_shape), wafer_shape_elements(dst_shape),
                wafer_shape4(kernel_x, kernel_y, stride_x, stride_y),
                wafer_shape4(pad_top, pad_bottom, pad_left, pad_right),
                wafer_format(format));
  wafer_execute_td(&instr);
  TsmDeleteDataMove(move);
}

static void wafer_arg_writeback(uint64_t value_dst, uint64_t index_dst,
                                uint32_t format, TsmPeripheralInstr *instr) {
  (void)TsmWaitfinish();
  wafer_store_value(value_dst, format, instr->param.wb_data0);
  wafer_store_u32(index_dst, (uint32_t)instr->param.wb_data1);
}

void wafer_tx81_peripheral_argmax(
    uint64_t src, uint64_t value_dst, uint64_t index_dst, uint32_t kind,
    uint32_t elem_count, uint32_t format, uint32_t lut_elem_count,
    uint32_t scale, uint32_t probability, uint32_t rounding_mode) {
  (void)kind;
  (void)lut_elem_count;
  (void)scale;
  (void)probability;
  (void)rounding_mode;
  TsmPeripheralInstr instr = {0};
  TsmPeripheral *peripheral = TsmNewPeripheral();
  peripheral->ArgMax(&instr, src, elem_count, wafer_format(format));
  wafer_execute_ct(&instr);
  wafer_arg_writeback(value_dst, index_dst, format, &instr);
  TsmDeletePeripheral(peripheral);
}

void wafer_tx81_peripheral_argmin(
    uint64_t src, uint64_t value_dst, uint64_t index_dst, uint32_t kind,
    uint32_t elem_count, uint32_t format, uint32_t lut_elem_count,
    uint32_t scale, uint32_t probability, uint32_t rounding_mode) {
  (void)kind;
  (void)lut_elem_count;
  (void)scale;
  (void)probability;
  (void)rounding_mode;
  TsmPeripheralInstr instr = {0};
  TsmPeripheral *peripheral = TsmNewPeripheral();
  peripheral->ArgMin(&instr, src, elem_count, wafer_format(format));
  wafer_execute_ct(&instr);
  wafer_arg_writeback(value_dst, index_dst, format, &instr);
  TsmDeletePeripheral(peripheral);
}

void wafer_tx81_peripheral_factorize(
    uint64_t src, uint64_t dst0, uint64_t dst1, uint64_t dst2, uint32_t kind,
    uint32_t elem_count, uint32_t format, uint32_t lut_elem_count,
    uint32_t scale, uint32_t probability, uint32_t rounding_mode) {
  (void)kind;
  (void)format;
  (void)lut_elem_count;
  (void)scale;
  (void)probability;
  (void)rounding_mode;
  TsmPeripheralInstr instr = {0};
  TsmPeripheral *peripheral = TsmNewPeripheral();
  peripheral->Factorize(&instr, src, dst0, dst1, dst2, elem_count);
  wafer_execute_ct(&instr);
  TsmDeletePeripheral(peripheral);
}

void wafer_tx81_peripheral_bilinear(
    uint64_t src, uint64_t dst, uint32_t kind, uint32_t elem_count,
    uint32_t format, uint32_t src_n, uint32_t src_h, uint32_t src_w,
    uint32_t src_c, uint32_t dst_n, uint32_t dst_h, uint32_t dst_w,
    uint32_t dst_c, uint32_t lut_elem_count, uint32_t scale,
    uint32_t probability, uint32_t rounding_mode) {
  (void)kind;
  (void)elem_count;
  (void)lut_elem_count;
  (void)scale;
  (void)probability;
  (void)rounding_mode;
  TsmPeripheralInstr instr = {0};
  TsmPeripheral *peripheral = TsmNewPeripheral();
  peripheral->Bilinear(&instr, src, dst, wafer_shape4(src_n, src_h, src_w, src_c),
                       wafer_shape4(dst_n, dst_h, dst_w, dst_c),
                       wafer_bilinear_scale(src_w, dst_w),
                       wafer_bilinear_scale(src_h, dst_h),
                       wafer_format(format));
  wafer_execute_ct(&instr);
  TsmDeletePeripheral(peripheral);
}

void wafer_tx81_peripheral_lut16(
    uint64_t src, uint64_t lut, uint64_t dst, uint32_t kind,
    uint32_t elem_count, uint32_t format, uint32_t lut_elem_count,
    uint32_t scale, uint32_t probability, uint32_t rounding_mode) {
  (void)kind;
  (void)format;
  (void)scale;
  (void)probability;
  (void)rounding_mode;
  TsmPeripheralInstr instr = {0};
  TsmPeripheral *peripheral = TsmNewPeripheral();
  peripheral->Lut16(&instr, src, dst, lut, elem_count, lut_elem_count);
  wafer_execute_ct(&instr);
  TsmDeletePeripheral(peripheral);
}

void wafer_tx81_peripheral_lut32(
    uint64_t src, uint64_t lut, uint64_t dst, uint32_t kind,
    uint32_t elem_count, uint32_t format, uint32_t lut_elem_count,
    uint32_t scale, uint32_t probability, uint32_t rounding_mode) {
  (void)kind;
  (void)format;
  (void)scale;
  (void)probability;
  (void)rounding_mode;
  TsmPeripheralInstr instr = {0};
  TsmPeripheral *peripheral = TsmNewPeripheral();
  peripheral->Lut32(&instr, src, dst, lut, elem_count, lut_elem_count);
  wafer_execute_ct(&instr);
  TsmDeletePeripheral(peripheral);
}

void wafer_tx81_peripheral_rand_gen(
    uint64_t src0, uint64_t src1, uint64_t dst0, uint64_t dst1, uint64_t dst2,
    uint32_t kind, uint32_t elem_count, uint32_t format,
    uint32_t lut_elem_count, uint32_t scale, uint32_t probability,
    uint32_t rounding_mode) {
  (void)kind;
  (void)lut_elem_count;
  (void)scale;
  (void)probability;
  (void)rounding_mode;
  TsmPeripheralInstr instr = {0};
  TsmPeripheral *peripheral = TsmNewPeripheral();
  peripheral->RandGen(&instr, src0, src1, dst0, dst1, dst2, elem_count,
                      wafer_format(format));
  wafer_execute_ct(&instr);
  TsmDeletePeripheral(peripheral);
}

void wafer_tx81_peripheral_elem_mask(
    uint64_t src, uint64_t dst, uint32_t kind, uint32_t elem_count,
    uint32_t format, uint32_t lut_elem_count, uint32_t scale,
    uint32_t probability, uint32_t rounding_mode) {
  (void)kind;
  (void)lut_elem_count;
  TsmPeripheralInstr instr = {0};
  TsmPeripheral *peripheral = TsmNewPeripheral();
  peripheral->ElemMask(&instr, src, scale, dst, elem_count, wafer_format(format),
                       probability, wafer_rounding(rounding_mode));
  wafer_execute_ct(&instr);
  TsmDeletePeripheral(peripheral);
}

void wafer_tx81_local_fence(void) { (void)TsmWaitfinish(); }
