# TX81 Extended CRT Surface Triage

本文把旧 DLCompiler TX81 CRT source 中未进入当前 Wafer production CRT closure 的能力做一次性分级。
它是 `tasks/14-target-llvm-golden-packet.md` 第 8 节的 evidence matrix，不是新的 ABI source。

当前 production CRT closure 仍然是 `runtime/wafer_crt/include/wafer_tx81_crt.h` 和
`runtime/wafer_crt/src/wafer_tx81_crt.c` 中已验证的 105 个 `wafer_tx81_*` symbols。

## Status Vocabulary

| status | meaning |
| --- | --- |
| `already-covered` | 旧 source 的可用语义已由当前 production symbol、verifier 或 conformance checker 覆盖 |
| `promote-now` | public wrapper 语义清楚，后续可以按同一批次扩 IR / ABI / CRT / tests |
| `needs-composite-ir` | 旧 helper 需要多条 wrapper、software loop、scratch、SPM mapping 或 completion 表达 |
| `needs-layout-ir` | 旧 helper 本质是 layout/materialization/movement，不应先成为 CRT helper |
| `needs-dte-abi` | 旧 helper 需要 endpoint/channel/FSM/tile topology/runtime binding |
| `reject-permanently` | 旧 runtime/link compatibility 或 stub，不属于 compiler ABI |

## Promote-Now Candidates

| old source | old functions | reason | required closure |
| --- | --- | --- | --- |
| `count.c` | `__Count` | public wrapper returns scalar through writeback register, same class as argmax/argmin | add explicit scalar writeback peripheral IR/ABI, mapped SPM writeback CRT, verifier arity, target LLVM call, device-link and negative tests |
| `arith.c` scalar-immediate forms | `__AddVS`, `__SubVS`, `__MulVS`, `__DivVS` | public wrappers are direct CT operations; current production only models VV | add scalar-immediate operand form or dedicated instr kind; do not overload VV ABI |
| `relation.c` scalar-immediate forms | `__Bool*VS`, `__*VS` relation functions | public wrappers are direct CT operations; current production only models VV | add explicit scalar-immediate relation IR and bool/value verifier rule |

`promote-now` does not mean “copy old `__*` functions”。It means the semantic gap is small enough that
the next implementation batch can close IR, ABI, CRT and tests together.

## Needs Composite IR

| old source | old functions | missing representation |
| --- | --- | --- |
| `gelu_none.c`, `gelu_tanh.c`, `op_gelu.c` | `__GeluNone`, `__GeluTanh`, `get_erf_value`, `get_tanh_value`, `op_gelu_none`, `op_gelu_tanh` | approximation mode, dtype conversion policy, scratch buffer, multi-issue order, completion |
| `op_reduce_mul_impl.c`, `reduce.c` | `op_reduce_mul_impl`, `__ReduceMul` | init value, multiply reduction order, numeric policy, temporary storage; old source delegates to composite helper |
| `mxfp_bf16.c`, `mxfp_fp16.c` | `__FP8E5M2_BF16`, `__FP8E4M3_BF16`, `__FP8E4M3FN_BF16`, `__FP4E2M1_BF16`, FP16 variants | packed MXFP dtype/type contract, software SPM loads/stores, completion |
| `mxfp_scale_bf16.c`, `mxfp_scale_fp16.c` | `__mxfpScaleBF16`, `__mxfpScaleFP16` | scale decode, scratch/order, dependency on scalar-immediate/mul wrapper sequence |

These should lower to explicit instruction sequences or a composite instruction region. The CRT must not
become a hidden scheduler or scratch allocator.

## Needs Layout IR

| old source | old functions | required IR boundary |
| --- | --- | --- |
| `channelnorm.c` | `__ChannelNorm`, `__DechannelNorm` | layout/materialization operation or gather/scatter descriptor sequence |
| `concat.c` | `__Concat` | segment movement / concat materialization IR with explicit offsets |
| `transpose.c` | `__Transpose` | permutation IR and descriptor legality; not name-based helper selection |
| `mirror.c` | `__Mirror` | axis-aware materialization IR |
| `rotate90.c`, `rotate180.c`, `rotate270.c` | `__Rotate90`, `__Rotate180`, `__Rotate270` | rotation as explicit permutation/movement plan |
| `nchw2nhwc.c`, `nhwc2nchw.c` | `__Nchw2nhwc`, `__Nhwc2nchw` | layout conversion IR; current compiler should prefer general materialization over old layout helper ABI |
| `tensornorm.c` | `__TensorNorm` | tensor layout normalization IR and legality |

这些 wrapper 可以作为 TDMA/gather-scatter 参数线索，但不能从旧 helper 名字直接进入 target CRT。

## Needs DTE ABI

| old source | old functions | missing ABI |
| --- | --- | --- |
| `send.c` | `getNextNearestTileId`, `getPrevNearestTileId`, `tile_sync_by_spm_single_direction`, `initTileId`, `__Send` | local/remote tile, channel, FSM id, receive buffer, sync slot, topology and completion token |
| `recv.c` | `__Recv` | implementation is effectively absent; receive-side binding still needs IR/runtime contract |
| `atomic_barrier_in.c`, `atomic_barrier_out.c` | `__AtomicBarrierIn`, `__AtomicBarrierOut` | board/runtime barrier semantics and scope; not local CRT fence |

旧 `__Send` 固化 4x4 ring 和 SPM sync slots，只能作为反例和字段线索，不能成为 production DTE ABI。

## Already Covered Or Excluded

| old source / family | status | note |
| --- | --- | --- |
| `barrier.c::__Barrier` | `already-covered` | useful local drain is covered by `wafer_tx81_local_fence` |
| direct VV arith/relation/logic/activation/transcendental/convert/reduce/GEMM/Conv/RDMA/WDMA/GatherScatter/Pad/Img2col/Bilinear/LUT/RandGen | `already-covered` | covered by current production closure and conformance checker |
| Pool/Unpool, factorize, elem_mask | `already-covered` | covered from public TX8 headers rather than old source |
| `common.c` | `reject-permanently` | `main`, `get_app_version`, `nvram_get_val` are old runtime compatibility |
| `empty.c` | `reject-permanently` | math/assert placeholder symbols |
| `assert.c`, `print.c` | `reject-permanently` | diagnostics/runtime glue, not target compiler ABI |
| `pow.c` | `reject-permanently` | software math helper; not a target CRT symbol |

## Implementation Rule

No future task should add an extended `wafer_tx81_*` symbol by only editing the CRT. A promoted family must
land with:

- instruction IR op/kind/result ABI and verifier.
- typed target LLVM call emission.
- repo-local CRT implementation.
- conformance/symbol checker update.
- positive and negative lit tests, including required-symbol device-link gate when a new CRT symbol is added.

If any item cannot be written clearly, the family stays staged instead of entering production.
