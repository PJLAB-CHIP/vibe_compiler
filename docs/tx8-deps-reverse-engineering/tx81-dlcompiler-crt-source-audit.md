# DLCompiler TX81 CRT Source Audit

本文审计一份未 vendored 的旧 DLCompiler TX81 CRT source snapshot；它是旧 `libvr.a` 的源码来源，
外部 checkout 位置不是仓库合同。本文只记录旧 source 能直接支持的静态事实，不维护 Wafer compiler
ABI、production membership 或任务状态。

查询当前边界时，IR / ABI 合同读取 `tasks/14-target-llvm-golden-packet.md`，prototype读取
`runtime/wafer_crt/include/wafer_tx81_crt.h`，repo-local实现读取
`runtime/wafer_crt/src/wafer_tx81_crt.c`，静态闭包检查读取
`tools/check_target_crt_symbols.py`的结果，任务状态读取`tasks/progress.md`。

## Scope

- 读取 `lib/Tx81/*.c` 全部 105 个 C source。
- 抽取并核对 181 个 C function definition。
- 将旧 function inventory 与仓库内 public CRT header/source snapshot 做静态交叉检查。
- 结论只用于确认 Tsm wrapper 调用线索、参数单位、同步/写回行为和旧实现的证据限制。

## Top-Level Findings

旧 TX81 CRT 对我们有用，但有明确边界：

- 有用：它给出了 `Tsm*` public wrapper 的实际调用顺序、method 名、shape/stride/format 参数单位，以及
  argmax/argmin writeback、bool relation/logic、convert zero-point/rounding 等细节。
- ABI 边界：它的 `__*` ABI 是旧 Tx81 / Triton op ABI。该 snapshot 只提供 wrapper 证据，不能证明
  Wafer target ABI，也不能证明仓库应恢复 `libvr.a` 或调用 `__Gemm` / `__AddVV` / `__Rdma`。
- 不能直接继承：旧源码混用了 `g_intrinsic()->*_pointer`、`TsmNew*`、`TsmWaitfinish()` 和
  `ENABLE_SYNCHRONOUS_INTRINSIC` 宏；同步语义不一致，不能作为 Wafer 的 implicit completion 规则。
- 旧 `__Conv` 在 `enLeakyRelu=false` 时默认 `EnableRelu()`；这一行为不能证明 Wafer
  fused-activation semantics。
- 旧 Direct DTE `__Send` 固化 4x4 tile ring、SPM sync slot 和 next/prev tile；`__Recv` 是空实现。
  这只能证明旧 prototype 的 topology-specific 行为和 receive-side 实现缺失。

## Static Cross-Reference Observations

| area | old-source evidence | evidence limit |
| --- | --- | --- |
| Arithmetic / activation / transcendental | 存在直接 `TsmArith`、`TsmActivation`、`TsmTranscendental` wrapper 及 per-method 参数 | 旧 `__*` 名字和 prototype 不能证明 Wafer symbol 或 ABI |
| Relation / logic | bool 和 value variants 使用不同 wrapper method；source 依据 format 选择 method | 不能证明 Wafer operand / result type relation 或 verifier rule |
| Convert | INT8 source 使用 zero point；FP / INT narrowing 使用 `RND_MODE`；plain convert 无 extra parameter | 不能证明 Wafer signature 分组或 rounding legality |
| RDMA / WDMA | 4D path 使用 `AddSrcDst` + `ConfigStrideIteration`；generic path 在旧 runtime 做 vectorize fallback | 不能证明 Wafer descriptor rank、byte / element unit 或 fallback ownership |
| GatherScatter / TDMA | stride-iteration descriptor order 有证据；pad / img2col / mirror / rotate / transpose / NCHW-NHWC wrappers 存在 | 不能证明当前 membership 或通用 layout / movement IR 边界 |
| GEMM | issue order 是 `AddInput -> ConfigMKN -> AddOutput -> SetPsum -> SetTransflag -> ConfigBatch -> optional features` | 不能证明 feature profile、legality 或 completion contract |
| Conv | shape order 是 `n,h,w,c`；pads / unpads / strides / dilations 显式传入 | 旧默认 ReLU 行为不能证明 Wafer fused-activation semantics |
| ArgMax / ArgMin | wait 后读取 `wb_data0` / `wb_data1` 并写回 value / index；旧代码使用 `get_spm_memory_mapping` | 不能证明 Wafer destination address class、writeback ABI 或 completion semantics |
| MaskMove | 旧 source 的 helper 参数为 `uint64_t mask`；观察到的 public adapter method 接受 `uint32_t mask` | 不能证明字段语义、address class 或 narrowing legality |
| GELU / MXFP / ReduceMul | composite / software helper 包含 SPM mapping、software loop 或多次 wrapper issue | 不能证明单条 target command、scratch ownership 或 completion contract |
| Send / Recv | `__Send` 是 ring-specific Direct DTE prototype；`__Recv` body effectively absent | 不能证明 endpoint / channel binding、receive behavior 或 Direct DTE completion |
| Stubs / compatibility | `common.c`、`empty.c`、`print.c` 提供 link / runtime compatibility 入口 | 不能证明 target compiler ABI |

## Detailed Static Observations

1. **SPM writeback mapping**：旧 `__ArgMax` / `__ArgMin` 等待 public writeback 完成后，通过
   `get_spm_memory_mapping(offset)` 映射 destination，再写入 `wb_data0` value 和 `wb_data1` index。
2. **MaskMove address width**：`instr_adapter_plat.h`的`TsmMaskDataMove::MaskMove`第三个参数是
   `uint32_t mask`，而旧 helper prototype 暴露 `uint64_t mask`；这只证明两处静态类型存在差异。
3. **DMA element count**：旧 `__Rdma4d` / `__Wdma4d` 直接接收 `elem_count`；该事实不能证明
   Wafer descriptor 使用 element count 还是 byte count。
4. **Composite ops**：GELU、MXFP、reduce_mul、channelnorm这类旧helper不是单条public wrapper；
   本文只记录其 scratch、ordering 和 software-loop 证据，不能据此推导 Wafer IR / ABI。

## Function Inventory

| source | functions audited | classification |
| --- | --- | --- |
| `abs.c` | `__AbsVV` | direct arith wrapper evidence |
| `argmax.c` | `__ArgMax` | peripheral writeback after wait, followed by SPM-mapped value / index stores |
| `argmin.c` | `__ArgMin` | peripheral writeback after wait, followed by SPM-mapped value / index stores |
| `arith.c` | `__AddVV`, `__SubVV`, `__MulVV`, `__DivVV`, `__AddVS`, `__SubVS`, `__MulVS`, `__DivVS`, `__MaxVV`, `__MinVV` | direct VV and VS arithmetic wrapper evidence |
| `assert.c` | `__Assert` | runtime assert / log shim behavior |
| `atomic_barrier_in.c` | `__AtomicBarrierIn` | board / runtime synchronization helper behavior |
| `atomic_barrier_out.c` | `__AtomicBarrierOut` | board / runtime synchronization helper behavior |
| `barrier.c` | `__Barrier` | local wait wrapper behavior |
| `bf16_fp16.c` | `__BF16_FP16` | direct convert wrapper evidence |
| `bf16_fp32.c` | `__BF16_FP32` | direct convert wrapper evidence |
| `bf16_int16.c` | `__BF16_INT16` | direct convert wrapper evidence; rounding mode |
| `bf16_int32.c` | `__BF16_INT32` | direct convert wrapper evidence; rounding mode |
| `bf16_int8.c` | `__BF16_INT8` | direct convert wrapper evidence |
| `bf16_tf32.c` | `__BF16_TF32` | direct convert wrapper evidence |
| `bilinear.c` | `__Bilinear` | peripheral bilinear wrapper evidence; scale derived from source/dest shape |
| `bit2fp.c` | `__Bit2Fp` | direct peripheral wrapper evidence |
| `channelnorm.c` | `__ChannelNorm`, `__DechannelNorm` | composite GatherScatter layout materialization behavior |
| `common.c` | `main`, `get_app_version`, `nvram_get_val` | link / runtime compatibility symbols |
| `concat.c` | `__Concat` | TDMA concat wrapper evidence |
| `conv.c` | `__Conv` | wrapper sequence evidence; old helper enables ReLU by default when leaky ReLU is disabled |
| `cos.c` | `__Cos` | direct transcendental wrapper evidence |
| `count.c` | `__Count` | peripheral count writeback after wait, followed by a mapped destination store |
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
| `gelu_none.c` | `__GeluNone` | composite GELU helper wrapper behavior |
| `gelu_tanh.c` | `__GeluTanh` | composite GELU helper wrapper with scratch-buffer use |
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
| `memcpy.c` | `__Memcpy` | GatherScatter memcpy helper with bool bitpack handling |
| `memset.c` | `__Memset` | peripheral memset wrapper evidence; source contains a stride TODO |
| `mirror.c` | `__Mirror` | TDMA transform wrapper evidence |
| `mxfp_bf16.c` | `__FP8E5M2_BF16`, `__FP8E4M3_BF16`, `__FP8E4M3FN_BF16`, `__FP4E2M1_BF16` | software MXFP conversion using mapped SPM loads/stores |
| `mxfp_fp16.c` | `__FP8E5M2_FP16`, `__FP8E4M3_FP16`, `__FP8E4M3FN_FP16`, `__FP4E2M1_FP16` | software MXFP conversion using mapped SPM loads/stores |
| `mxfp_scale_bf16.c` | `__mxfpScaleBF16` | composite MXFP scale using software scale decode plus `TsmArith::MulVS` |
| `mxfp_scale_fp16.c` | `__mxfpScaleFP16` | composite MXFP scale using software scale decode plus `TsmArith::MulVS` |
| `nchw2nhwc.c` | `__Nchw2nhwc` | TDMA layout transform wrapper evidence |
| `neg.c` | `__NegVV` | direct arith wrapper evidence |
| `nhwc2nchw.c` | `__Nhwc2nchw` | TDMA layout transform wrapper evidence |
| `op_gelu.c` | `get_ptr_value_by_idx_new`, `get_erf_value`, `get_tanh_value`, `op_gelu_none`, `op_gelu_tanh` | software / composite GELU implementation with explicit scratch and issue order |
| `op_reduce_mul_impl.c` | `op_reduce_mul_impl` | composite reduce-mul implementation |
| `pad.c` | `__Pad` | TDMA pad wrapper evidence |
| `pow.c` | `round_to_even2`, `powf` | software math helper behavior |
| `pow2.c` | `__Pow2` | direct transcendental wrapper evidence |
| `print.c` | `__Print` | no-op / print compatibility behavior |
| `randgen.c` | `__RandGen` | peripheral random wrapper evidence |
| `rdma.c` | `__Rdma4d`, `__Rdma1d`, `__RdmaVectorize`, `__Rdma` | RDMA wrapper evidence plus old runtime vectorize fallback |
| `recip.c` | `__RecipVV` | direct arith wrapper evidence |
| `recv.c` | `__Recv` | TODO and empty implementation body |
| `reduce.c` | `__ReduceSum`, `__ReduceAvg`, `__ReduceMax`, `__ReduceMin`, `__ReduceMul` | native reduce evidence; reduce-mul delegates composite helper |
| `relation.c` | `__BoolEqualVV`, `__BoolUnEqualVV`, `__BoolGreaterEqualVV`, `__BoolGreaterVV`, `__BoolLessEqualVV`, `__BoolLessThenVV`, `__EqualVV`, `__UnEqualVV`, `__GreaterEqualVV`, `__GreaterVV`, `__LessEqualVV`, `__LessThenVV`, `__BoolEqualVS`, `__BoolUnEqualVS`, `__BoolGreaterEqualVS`, `__BoolGreaterVS`, `__BoolLessEqualVS`, `__BoolLessThenVS`, `__EqualVS`, `__UnEqualVS`, `__GreaterEqualVS`, `__GreaterVS`, `__LessEqualVS`, `__LessThenVS` | relation value / bool and VV / VS wrapper evidence |
| `relu.c` | `__Relu` | direct activation wrapper evidence |
| `rotate180.c` | `__Rotate180` | TDMA transform wrapper evidence |
| `rotate270.c` | `__Rotate270` | TDMA transform wrapper evidence |
| `rotate90.c` | `__Rotate90` | TDMA transform wrapper evidence |
| `rsqrt.c` | `__RsqrtVV` | direct arith wrapper evidence |
| `satrelu.c` | `__Satrelu` | direct activation wrapper evidence |
| `send.c` | `getNextNearestTileId`, `getPrevNearestTileId`, `tile_sync_by_spm_single_direction`, `initTileId`, `__Send` | ring-specific Direct DTE prototype with fixed topology and sync slots |
| `sigmoid.c` | `__Sigmoid` | direct activation wrapper evidence |
| `sin.c` | `__Sin` | direct transcendental wrapper evidence |
| `softplus.c` | `__Softplus` | direct activation wrapper evidence |
| `sqrt.c` | `__SqrtVV` | direct arith wrapper evidence |
| `tanh.c` | `__Tanh` | direct activation wrapper evidence |
| `tensornorm.c` | `__TensorNorm` | TDMA tensor norm wrapper evidence |
| `tf32_bf16.c` | `__TF32_BF16` | direct convert wrapper evidence; rounding mode |
| `tf32_fp16.c` | `__TF32_FP16` | direct convert wrapper evidence |
| `tf32_fp32.c` | `__TF32_FP32` | direct convert wrapper evidence |
| `tf32_int16.c` | `__TF32_INT16` | direct convert wrapper evidence; rounding mode |
| `tf32_int32.c` | `__TF32_INT32` | direct convert wrapper evidence; rounding mode |
| `tf32_int8.c` | `__TF32_INT8` | direct convert wrapper evidence; rounding mode |
| `transpose.c` | `__Transpose` | TDMA transpose wrapper evidence |
| `tx81.c` | `is_contiguous`, `next_power_of_two_64`, `get_dtype_size_new`, `get_cx_align_base_new`, `no_reverse_memory_access`, `tx81_memcpy`, `legalizeMemoryOpAttribute`, `get_spm_memory_mapping_wrapper` | runtime planning / helper logic and SPM offset mapping behavior |
| `wdma.c` | `__Wdma4d`, `__Wdma1d`, `__WdmaVectorize`, `__Wdma` | WDMA wrapper evidence plus old runtime vectorize fallback |

## Current Boundary Navigation

本审计不维护 covered / excluded list。查询当前事实时：

- IR / ABI 和production closure合同读取`tasks/14-target-llvm-golden-packet.md`；
- prototype读取`runtime/wafer_crt/include/wafer_tx81_crt.h`，实现读取
  `runtime/wafer_crt/src/wafer_tx81_crt.c`；
- symbol、signature 和 object-level 静态闭包读取 `tools/check_target_crt_symbols.py` 的检查结果；
- 队列状态读取 `tasks/progress.md`。

inventory与这些当前边界的差异只表示evidence gap，不授权扩展symbol、IR、ABI或runtime path。
