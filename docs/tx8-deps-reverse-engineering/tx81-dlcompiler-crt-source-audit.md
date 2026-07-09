# DLCompiler TX81 CRT Source Audit

本文审计旧工程源码：

```text
/root/dlc_dev/DLCompiler/third_party/wafer/crt/
```

它是旧 `libvr.a` 的源码来源。本文只作为 evidence audit，不是 Wafer 当前 compiler ABI
合同。Wafer 当前 target CRT 边界仍然是 `runtime/wafer_crt/include/wafer_tx81_crt.h` 和
`runtime/wafer_crt/src/wafer_tx81_crt.c` 中的 `wafer_tx81_*` symbols。

## Scope

- 读取 `lib/Tx81/*.c` 全部 105 个 C source。
- 抽取并核对 181 个 C function definition。
- 对照当前 Wafer CRT 的 105 个 production `wafer_tx81_*` symbols。
- 结论只用于确认 Tsm wrapper 调用线索、参数单位、同步/写回风险和不能继承的旧实现模式。

## Top-Level Findings

旧 TX81 CRT 对我们有用，但有明确边界：

- 有用：它给出了 `Tsm*` public wrapper 的实际调用顺序、method 名、shape/stride/format 参数单位，以及
  argmax/argmin writeback、bool relation/logic、convert zero-point/rounding 等细节。
- 不能用：它的 `__*` ABI 是旧 Tx81/Triton op ABI，不是 Wafer compiler target ABI；不能恢复
  `libvr.a`、不能让 target LLVM 调 `__Gemm` / `__AddVV` / `__Rdma`。
- 不能直接继承：旧源码混用了 `g_intrinsic()->*_pointer`、`TsmNew*`、`TsmWaitfinish()` 和
  `ENABLE_SYNCHRONOUS_INTRINSIC` 宏；同步语义不一致，不能作为 Wafer 的 implicit completion 规则。
- 旧 `__Conv` 在 `enLeakyRelu=false` 时默认 `EnableRelu()`，这不符合 Wafer V0 “无 fused activation”
  语义。当前 Wafer CRT 显式 `DisableRelu` / `DisableLeakyRelu` 的方向是对的。
- 旧 Direct DTE `__Send` 固化 4x4 tile ring、SPM sync slot 和 next/prev tile；`__Recv` 是空实现。
  这只能作为反例，不能进入 production closure。

## Current CRT Impact

| area | old-source evidence | impact on current Wafer CRT |
| --- | --- | --- |
| Arithmetic / activation / transcendental | 直接 `TsmArith`、`TsmActivation`、`TsmTranscendental` wrapper | 当前 per-kind `wafer_tx81_elementwise_*` 保持 fixed ABI 即可；old `__*` 名字不能泄漏 |
| Relation / logic | bool and value variants 是不同 wrapper method | 当前用 `format == Fmt_BOOL` 选择 bool method 是合理方向；dest `i1` 不能反向决定 input format |
| Convert | INT8 source uses zero point; FP/INT narrowing uses `RND_MODE`; plain converts无 extra param | 当前 zero-point / rounding / plain 三组 ABI 划分有旧源码支持 |
| RDMA / WDMA | 4D path uses `AddSrcDst` + `ConfigStrideIteration`; generic path在旧 runtime 做 vectorize fallback | 当前 target CRT 应只接收 lowering 已合法化的 3-level descriptor；不要把旧 runtime vectorize loop 放进 CRT |
| GatherScatter / TDMA | stride-iteration descriptor order 有证据；pad/img2col/mirror/rotate/transpose/NCHW-NHWC wrappers 存在 | V0 只把已收敛的 pad/img2col/gather_scatter 放进 production；transform-like TDMA 后续单独扩 target surface |
| GEMM | `AddInput -> ConfigMKN -> AddOutput -> SetPsum -> SetTransflag -> ConfigBatch -> optional features` | 当前禁用 psum/bias/scale/quant/activation 是合理的；未来启用必须扩 IR，不从旧 ABI 默认继承 |
| Conv | shape order is `n,h,w,c`; pads/unpads/strides/dilations are explicit | 当前 fixed fields 合理；旧默认 ReLU 是反例 |
| ArgMax / ArgMin | 必须 wait 后从 `wb_data0/wb_data1` 写回 value/index；旧代码用 `get_spm_memory_mapping` 写 SPM | 当前 `wafer_arg_writeback` 的 wait 语义合理；若 ABI dest 是 SPM offset，直接 pointer store 需要修正为 mapped SPM store |
| MaskMove | 旧源码传 `uint64_t mask`，但 current public header is `uint32_t mask` | 当前 cast to `uint32_t` matches `instr_adapter_plat.h`; ABI 文档要明确这是 mask field/offset，不是普通 64-bit address |
| GELU / MXFP / ReduceMul | composite/software helper，含 SPM mapping、software loop 或多条 wrapper issue | 不能无条件进入 target CRT production；需要 IR 表示 composite scratch、ordering 和 completion |
| Send / Recv | `__Send` 是 ring-specific direct-DTE prototype；`__Recv` TODO | 不能作为 Direct DTE production implementation |
| Stubs / compatibility | `common.c`、`empty.c`、`print.c` 只为 link/runtime compatibility | 不进入 Wafer ABI |

## High-Priority Follow-Up Checks

1. **SPM writeback mapping**：本批已把 `wafer_tx81_peripheral_argmax/argmin` 的
   `value_dst` / `index_dst` 收敛为 SPM offset ABI。CRT 等待 public writeback 完成后，通过
   `get_spm_memory_mapping(offset)` 映射 SPM offset，再写入 `wb_data0` value 和 `wb_data1` index。
2. **MaskMove address width**：`instr_adapter_plat.h` 的 `TsmMaskDataMove::MaskMove` 第三个参数是
   `uint32_t mask`。当前 ABI 传 `uint64_t mask` 再截断，必须明确该字段是 mask offset/field，
   不是 generic 64-bit SPM address。
3. **DMA elem count vs bytes**：旧 `__Rdma4d/__Wdma4d` 直接接 `elem_count`；当前 ABI 接
   `inner_bytes` 再按 `format` 换算 element count。保持这个设计可以，但 verifier/lowering 必须保证
   `inner_bytes % sizeof(format) == 0`。
4. **Composite ops**：GELU、MXFP、reduce_mul、channelnorm 这类旧 helper 不是单条 public wrapper；
   后续若要支持，应该先扩 IR/ABI 表达 scratch、ordering 和 software fallback，而不是复制旧 helper。

## Function Inventory

| source | functions audited | classification |
| --- | --- | --- |
| `abs.c` | `__AbsVV` | direct arith wrapper evidence |
| `argmax.c` | `__ArgMax` | peripheral writeback; requires wait and SPM mapped store |
| `argmin.c` | `__ArgMin` | peripheral writeback; requires wait and SPM mapped store |
| `arith.c` | `__AddVV`, `__SubVV`, `__MulVV`, `__DivVV`, `__AddVS`, `__SubVS`, `__MulVS`, `__DivVS`, `__MaxVV`, `__MinVV` | direct arith wrapper evidence; VS forms not in current production ABI |
| `assert.c` | `__Assert` | runtime assert/log shim; not compiler ABI |
| `atomic_barrier_in.c` | `__AtomicBarrierIn` | board/runtime sync helper; not target CRT production |
| `atomic_barrier_out.c` | `__AtomicBarrierOut` | board/runtime sync helper; not target CRT production |
| `barrier.c` | `__Barrier` | maps to local wait; current `wafer_tx81_local_fence` covers the useful part |
| `bf16_fp16.c` | `__BF16_FP16` | direct convert wrapper evidence |
| `bf16_fp32.c` | `__BF16_FP32` | direct convert wrapper evidence |
| `bf16_int16.c` | `__BF16_INT16` | direct convert wrapper evidence; rounding mode |
| `bf16_int32.c` | `__BF16_INT32` | direct convert wrapper evidence; rounding mode |
| `bf16_int8.c` | `__BF16_INT8` | direct convert wrapper evidence |
| `bf16_tf32.c` | `__BF16_TF32` | direct convert wrapper evidence |
| `bilinear.c` | `__Bilinear` | peripheral bilinear wrapper evidence; scale derived from source/dest shape |
| `bit2fp.c` | `__Bit2Fp` | direct peripheral wrapper evidence |
| `channelnorm.c` | `__ChannelNorm`, `__DechannelNorm` | composite GatherScatter layout materialization; not current production ABI |
| `common.c` | `main`, `get_app_version`, `nvram_get_val` | compatibility symbols for link/runtime; not compiler ABI |
| `concat.c` | `__Concat` | TDMA concat wrapper evidence; not current production target surface |
| `conv.c` | `__Conv` | wrapper sequence evidence; old default ReLU is not reusable |
| `cos.c` | `__Cos` | direct transcendental wrapper evidence |
| `count.c` | `__Count` | peripheral count writeback lacks current Wafer IR representation |
| `empty.c` | `sqrt`, `floor`, `fmin`, `fmax`, `ceil` | assert-only placeholder symbols |
| `exp.c` | `__Exp` | direct transcendental wrapper evidence |
| `explp.c` | `__Explp` | direct transcendental wrapper evidence |
| `fp16_bf16.c` | `__FP16_BF16` | direct convert wrapper evidence; rounding mode |
| `fp16_fp32.c` | `__FP16_FP32` | direct convert wrapper evidence |
| `fp16_int16.c` | `__FP16_INT16` | direct convert wrapper evidence; rounding mode |
| `fp16_int32.c` | `__FP16_INT32` | direct convert wrapper evidence; rounding mode |
| `fp16_int8.c` | `__FP16_INT8` | direct convert wrapper evidence; rounding mode |
| `fp16_tf32.c` | `__FP16_TF32` | direct convert wrapper evidence |
| `fp32_bf16.c` | `__FP32_BF16` | direct convert wrapper evidence; rounding mode |
| `fp32_fp16.c` | `__FP32_FP16` | direct convert wrapper evidence; rounding mode |
| `fp32_int16.c` | `__FP32_INT16` | direct convert wrapper evidence; rounding mode |
| `fp32_int32.c` | `__FP32_INT32` | direct convert wrapper evidence; rounding mode |
| `fp32_int8.c` | `__FP32_INT8` | direct convert wrapper evidence; rounding mode |
| `fp32_tf32.c` | `__FP32_TF32` | direct convert wrapper evidence; rounding mode |
| `gatherscatter.c` | `__GatherScatter` | direct TDMA GatherScatter evidence; stride order useful |
| `gelu_none.c` | `__GeluNone` | composite GELU helper wrapper; not direct target CRT op |
| `gelu_tanh.c` | `__GeluTanh` | composite GELU helper wrapper; needs scratch buffer |
| `gemm.c` | `__Gemm` | NE GEMM wrapper sequence evidence |
| `img2col.c` | `__Img2col` | TDMA img2col wrapper evidence |
| `int16_bf16.c` | `__INT16_BF16` | direct convert wrapper evidence; rounding mode |
| `int16_fp16.c` | `__INT16_FP16` | direct convert wrapper evidence |
| `int16_fp32.c` | `__INT16_FP32` | direct convert wrapper evidence; rounding mode |
| `int16_tf32.c` | `__INT16_TF32` | direct convert wrapper evidence; rounding mode |
| `int32_bf16.c` | `__INT32_BF16` | direct convert wrapper evidence; rounding mode |
| `int32_fp16.c` | `__INT32_FP16` | direct convert wrapper evidence; rounding mode |
| `int32_fp32.c` | `__INT32_FP32` | direct convert wrapper evidence; rounding mode |
| `int32_tf32.c` | `__INT32_TF32` | direct convert wrapper evidence; rounding mode |
| `int8_bf16.c` | `__INT8_BF16` | direct convert wrapper evidence; zero point |
| `int8_fp16.c` | `__INT8_FP16` | direct convert wrapper evidence; zero point |
| `int8_fp32.c` | `__INT8_FP32` | direct convert wrapper evidence; zero point |
| `int8_tf32.c` | `__INT8_TF32` | direct convert wrapper evidence; zero point |
| `leakyrelu.c` | `__Leakyrelu` | direct activation wrapper evidence |
| `ln.c` | `__Ln` | direct transcendental wrapper evidence |
| `log2.c` | `__Log2` | direct transcendental wrapper evidence |
| `logic.c` | `__AndVV`, `__OrVV`, `__XorVV`, `__BoolNotV`, `__BoolAndV`, `__BoolOrV`, `__BoolXorV` | logic value/bool wrapper split evidence |
| `lut16.c` | `__Lut16` | peripheral LUT wrapper evidence |
| `lut32.c` | `__Lut32` | peripheral LUT wrapper evidence |
| `mask_move.c` | `__MaskMove` | mask move evidence; old pointer-like argument conflicts with public `uint32_t mask` field |
| `memcpy.c` | `__Memcpy` | GatherScatter memcpy helper; bool bitpack handling is old ABI-specific |
| `memset.c` | `__Memset` | peripheral memset wrapper evidence; old stride TODO is not reusable |
| `mirror.c` | `__Mirror` | TDMA transform wrapper evidence; not current production target surface |
| `mxfp_bf16.c` | `__FP8E5M2_BF16`, `__FP8E4M3_BF16`, `__FP8E4M3FN_BF16`, `__FP4E2M1_BF16` | software MXFP conversion using mapped SPM loads/stores |
| `mxfp_fp16.c` | `__FP8E5M2_FP16`, `__FP8E4M3_FP16`, `__FP8E4M3FN_FP16`, `__FP4E2M1_FP16` | software MXFP conversion using mapped SPM loads/stores |
| `mxfp_scale_bf16.c` | `__mxfpScaleBF16` | composite MXFP scale using software scale decode plus `TsmArith::MulVS` |
| `mxfp_scale_fp16.c` | `__mxfpScaleFP16` | composite MXFP scale using software scale decode plus `TsmArith::MulVS` |
| `nchw2nhwc.c` | `__Nchw2nhwc` | TDMA layout transform wrapper evidence; not current production target surface |
| `neg.c` | `__NegVV` | direct arith wrapper evidence |
| `nhwc2nchw.c` | `__Nhwc2nchw` | TDMA layout transform wrapper evidence; not current production target surface |
| `op_gelu.c` | `get_ptr_value_by_idx_new`, `get_erf_value`, `get_tanh_value`, `op_gelu_none`, `op_gelu_tanh` | software/composite GELU implementation; needs scratch/order/completion modeling |
| `op_reduce_mul_impl.c` | `op_reduce_mul_impl` | composite reduce-mul implementation; current production reduce excludes mul |
| `pad.c` | `__Pad` | TDMA pad wrapper evidence |
| `pow.c` | `round_to_even2`, `powf` | software math helper; not target CRT ABI |
| `pow2.c` | `__Pow2` | direct transcendental wrapper evidence |
| `print.c` | `__Print` | no-op/print compatibility; not compiler ABI |
| `randgen.c` | `__RandGen` | peripheral random wrapper evidence |
| `rdma.c` | `__Rdma4d`, `__Rdma1d`, `__RdmaVectorize`, `__Rdma` | RDMA wrapper evidence plus old runtime vectorize fallback |
| `recip.c` | `__RecipVV` | direct arith wrapper evidence |
| `recv.c` | `__Recv` | TODO empty implementation; not usable |
| `reduce.c` | `__ReduceSum`, `__ReduceAvg`, `__ReduceMax`, `__ReduceMin`, `__ReduceMul` | native reduce evidence; reduce-mul delegates composite helper |
| `relation.c` | `__BoolEqualVV`, `__BoolUnEqualVV`, `__BoolGreaterEqualVV`, `__BoolGreaterVV`, `__BoolLessEqualVV`, `__BoolLessThenVV`, `__EqualVV`, `__UnEqualVV`, `__GreaterEqualVV`, `__GreaterVV`, `__LessEqualVV`, `__LessThenVV`, `__BoolEqualVS`, `__BoolUnEqualVS`, `__BoolGreaterEqualVS`, `__BoolGreaterVS`, `__BoolLessEqualVS`, `__BoolLessThenVS`, `__EqualVS`, `__UnEqualVS`, `__GreaterEqualVS`, `__GreaterVS`, `__LessEqualVS`, `__LessThenVS` | relation value/bool and VV/VS evidence; current production only uses VV-style symbols |
| `relu.c` | `__Relu` | direct activation wrapper evidence |
| `rotate180.c` | `__Rotate180` | TDMA transform wrapper evidence; not current production target surface |
| `rotate270.c` | `__Rotate270` | TDMA transform wrapper evidence; not current production target surface |
| `rotate90.c` | `__Rotate90` | TDMA transform wrapper evidence; not current production target surface |
| `rsqrt.c` | `__RsqrtVV` | direct arith wrapper evidence |
| `satrelu.c` | `__Satrelu` | direct activation wrapper evidence |
| `send.c` | `getNextNearestTileId`, `getPrevNearestTileId`, `tile_sync_by_spm_single_direction`, `initTileId`, `__Send` | ring-specific Direct DTE prototype; not production ABI |
| `sigmoid.c` | `__Sigmoid` | direct activation wrapper evidence |
| `sin.c` | `__Sin` | direct transcendental wrapper evidence |
| `softplus.c` | `__Softplus` | direct activation wrapper evidence |
| `sqrt.c` | `__SqrtVV` | direct arith wrapper evidence |
| `tanh.c` | `__Tanh` | direct activation wrapper evidence |
| `tensornorm.c` | `__TensorNorm` | TDMA tensor norm wrapper evidence; not current production target surface |
| `tf32_bf16.c` | `__TF32_BF16` | direct convert wrapper evidence; rounding mode |
| `tf32_fp16.c` | `__TF32_FP16` | direct convert wrapper evidence |
| `tf32_fp32.c` | `__TF32_FP32` | direct convert wrapper evidence |
| `tf32_int16.c` | `__TF32_INT16` | direct convert wrapper evidence; rounding mode |
| `tf32_int32.c` | `__TF32_INT32` | direct convert wrapper evidence; rounding mode |
| `tf32_int8.c` | `__TF32_INT8` | direct convert wrapper evidence; rounding mode |
| `transpose.c` | `__Transpose` | TDMA transpose wrapper evidence; not current production target surface |
| `tx81.c` | `is_contiguous`, `next_power_of_two_64`, `get_dtype_size_new`, `get_cx_align_base_new`, `no_reverse_memory_access`, `tx81_memcpy`, `legalizeMemoryOpAttribute`, `get_spm_memory_mapping_wrapper` | runtime planning/helper logic plus SPM offset mapping helper; use as legality/planning/writeback evidence, not target CRT ABI |
| `wdma.c` | `__Wdma4d`, `__Wdma1d`, `__WdmaVectorize`, `__Wdma` | WDMA wrapper evidence plus old runtime vectorize fallback |

## Coverage Against Current Production CRT

Current Wafer CRT intentionally covers a smaller, fixed ABI surface:

- covered from old direct wrapper evidence: RDMA/WDMA, GatherScatter, Memset, Bit2Fp, MaskMove, elementwise
  arithmetic/relation/logic/transcendental/activation, native reduce sum/avg/max/min, convert families,
  GEMM, Conv, Pad, Img2col, ArgMax/ArgMin, Bilinear, LUT16/32, RandGen.
- covered from public TX8 headers rather than old source: Pool/Unpool, depthwise/backward conv variants,
  peripheral factorize and elem-mask.
- deliberately excluded: Direct DTE send/recv, peripheral count, reduce-mul, GELU composite helpers, MXFP
  software conversion/scale, channelnorm/dechannelnorm, concat, mirror, rotate, transpose, NCHW/NHWC,
  tensornorm, print/assert/common/empty compatibility symbols.

This means the old source is useful as a per-wrapper sanity check, but not a reason to expand the production
symbol closure without corresponding IR, verifier, lowering, and device-link tests.
