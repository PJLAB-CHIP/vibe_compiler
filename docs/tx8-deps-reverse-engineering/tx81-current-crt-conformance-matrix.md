# TX81 Current CRT Conformance Matrix

本文把旧 DLCompiler TX81 CRT source audit 转成当前 Wafer CRT 的生产面合规矩阵。
它不是新的 ABI source；当前 ABI 仍以 `tasks/14-target-llvm-golden-packet.md`、
`runtime/wafer_crt/include/wafer_tx81_crt.h` 和 `runtime/wafer_crt/src/wafer_tx81_crt.c`
为准。

## Scope

- Current production surface：`runtime/wafer_crt/include/wafer_tx81_crt.h` 和
  `runtime/wafer_crt/src/wafer_tx81_crt.c` 中的 105 个 `wafer_tx81_*` symbols。
- Evidence source：旧 DLCompiler `third_party/wafer/crt/lib/Tx81` source audit。
- 目标：记录每个 production CRT family 如何借鉴旧 source、只借鉴 public TX8 header，或明确排除
  旧 helper / ABI。

## Pipeline Position

Pipeline position:
- Upstream artifact / IR: memory-planned `wafer.instr.*` with accepted SPM/DDR offset facts.
- Current stage responsibility: lower fixed instruction ABI calls to Wafer-owned target CRT functions and issue public TX8 wrappers.
- Output artifact / IR: target LLVM / device object linked with repo-local Wafer CRT object.
- Downstream consumer: `tools/wafer_device_link.py` and package metadata/runtime launch stages.
- User-level driver / named pipeline: `--wafer-lower-instr-to-target-llvm`, `wafer-lower-groups-to-target-llvm`, and device-link lit tools.
- Explicit non-goals: old `__*` ABI, `libvr.a`, Direct DTE prototype helpers, composite GELU/MXFP/reduce-mul helpers.
- Completion gate: conformance checker, CRT symbol checker, focused lit tests, and full lit regression.

## Matrix

| CRT family | Current status | Old-source evidence used | Required conformance rule |
| --- | --- | --- | --- |
| RDMA / WDMA | direct-wrapper-derived | `__Rdma4d`, `__Wdma4d` | use `AddSrcDst` then `ConfigStrideIteration`; convert `inner_bytes` to element count; do not import old vectorize fallback |
| GatherScatter | direct-wrapper-derived | `__GatherScatter`, `__Memcpy` | preserve byte `inner_bytes` and source/dest stride-iteration order |
| Memset / Bit2Fp / MaskMove | direct-wrapper-derived | `__Memset`, `__Bit2Fp`, `__MaskMove` | use public wrappers; mask is `uint32_t` field/offset, not a generic 64-bit pointer |
| Elementwise arithmetic / relation / logic | direct-wrapper-derived | `arith.c`, `relation.c`, `logic.c`, unary activation/transcendental files | use fixed per-kind symbols; relation/logic bool branch only for `Fmt_BOOL`; no VS/VuV production ABI |
| Convert | direct-wrapper-derived | dtype conversion files | INT8 source uses zero point, FP/INT narrowing uses rounding, plain wrappers ignore extra ABI fields |
| Reduce | direct-wrapper-derived | `__ReduceSum/Avg/Max/Min` | native production reduce is sum/avg/max/min only; reduce-mul remains excluded |
| GEMM | direct-wrapper-derived | `__Gemm` | issue wrapper sequence and explicitly disable psum/bias/scale/quant/fused activation |
| Conv / Depthwise / BackwardConv | mixed: conv old source plus public headers | `__Conv`, public wrapper headers | preserve NHWC/HWOI field order and explicitly disable optional/fused features; old default ReLU is a negative example |
| Pool / Unpool | public-header-derived | no old source file | keep current public wrapper mapping; do not infer old source evidence |
| TDMA pad/img2col | direct-wrapper-derived | `__Pad`, `__Img2col` | use only V0 production TDMA surface; transform-like TDMA wrappers remain excluded |
| Peripheral argmax/argmin | mismatch-fixed-this-batch | `__ArgMax`, `__ArgMin`, SPM mapping helper | wait for writeback registers, map SPM offset destinations, then store value/index |
| Peripheral factorize/elem_mask | public-header-derived | no old source file | keep as public-header-derived only |
| Peripheral bilinear/LUT/rand | direct-wrapper-derived | `__Bilinear`, `__Lut16`, `__Lut32`, `__RandGen` | use public wrappers with fixed ABI fields; bilinear scale is derived from shapes |
| Count / Direct DTE / composite helpers | intentionally-excluded | `__Count`, `__Send`, `__Recv`, GELU/MXFP/reduce_mul/channelnorm/etc. | no production CRT symbol until IR/ABI represents result, scratch, ordering, endpoint/channel, and completion semantics |

## Current Fix Status

- `mismatch-fixed-this-batch`: argmax/argmin writeback destinations are SPM offsets, and CRT maps them before
  Kcore stores.
- `needs-test`: DMA byte/element conversion, MaskMove mask width, fused-feature disablement and convert
  grouping are enforced by the static conformance checker and focused lit test.
- `intentionally-excluded`: old source helpers remain out of production CRT unless instruction IR, verifier,
  lowering, CRT and device-link tests are extended together.
