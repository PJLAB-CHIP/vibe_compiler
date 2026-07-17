# TX81 CRT Static Evidence Matrix

本文把旧DLCompiler TX81 CRT source audit与repo-local Wafer CRT实现做静态对照。它只记录
source-backed evidence，不拥有production membership、prototype/signature、IR、ABI、lowering或
runtime policy。当前IR / ABI和production closure合同查看`tasks/14-target-conversion-module-publication.md`，
prototype与repo-local实现分别查看`runtime/wafer_crt/include/wafer_tx81_crt.h`和
`runtime/wafer_crt/src/wafer_tx81_crt.c`；闭合状态查看`tools/check_target_crt_symbols.py`与
`tasks/progress.md`。

## Scope

- Evidence source：旧 DLCompiler `lib/Tx81` source snapshot与public TX8 headers。
- Repo-local observation：compiler target lowering以及`runtime/wafer_crt` public header/source中可直接
  读取的legality、ABI、wrapper调用、参数处理、wait/writeback和optional-feature配置。
- 本表不因旧helper存在而授权新symbol，也不定义某个family的production状态。
- 当前symbol checker从target lowering和Wafer enum registry推导出104个production symbol；这是当前
  实现的可重放观察，closure事实源仍是checker，不由本表另建清单。

## Static Matrix

| family | old-source / public-header evidence | repo-local implementation observation |
| --- | --- | --- |
| RDMA / WDMA | `__Rdma4d`, `__Wdma4d` call `AddSrcDst` and `ConfigStrideIteration`; old generic helpers also contain a vectorize fallback | Wafer compiler/CRT用checked conversion把`inner_bytes`转为element count：bitpacked BOOL要求`inner_bytes <= UINT32_MAX / 8`并计算`bytes * 8`，其它format要求可被element byte width整除；不满足时fail closed，不做截断除法或溢出乘法 |
| GatherScatter | `__GatherScatter` and `__Memcpy` expose source/destination stride-iteration ordering | Wafer CRT passes byte `inner_bytes` and separate source/destination descriptors to `GatherScatter` |
| Memset / Bit2Fp / MaskMove | old direct helpers and public wrapper declarations expose these operations; the public MaskMove field is `uint32_t` | Wafer CRT header/source都以`uint32_t mask`调用public wrapper，不含隐藏cast；compiler lowering证明mask来自已规划SPM allocation、view/range不越界且完整physical address range适配`uint32_t`后才发i32参数 |
| Elementwise arithmetic / relation / logic | `arith.c`, `relation.c`, `logic.c` and unary files contain VV, VS, bool and value wrapper variants | Wafer CRT source defines per-kind wrapper calls; relation/logic macros branch on `Fmt_BOOL` |
| Convert | old dtype conversion files separate zero-point, rounding and plain wrapper forms | Wafer CRT source has corresponding zero-point, rounding and plain macro groups |
| Reduce | old source contains sum/avg/max/min direct wrappers and a composite reduce-mul path | Wafer CRT source contains direct sum/avg/max/min wrapper definitions; no reduce-mul definition is observed |
| GEMM | `__Gemm` records input/config/output calls and optional feature setters | Wafer CRT explicitly sets psum, bias, scale, quant and activation fields to disabled values around the GEMM call |
| Conv / Depthwise / BackwardConv | old `__Conv` and public headers expose NHWC/HWOI fields; old `__Conv` may enable ReLU by default | Wafer CRT uses explicit shape/pad/stride fields and explicitly disables optional/fused features |
| Pool / Unpool | no matching old source file was found; public headers expose wrapper entry points | Wafer CRT contains public-wrapper calls for pool and unpool families |
| TDMA pad / img2col | `__Pad` and `__Img2col` expose direct wrapper parameters; other old files contain additional transform helpers | Wafer CRT contains pad and img2col definitions; the other transform helper names are absent |
| Peripheral argmax / argmin | `__ArgMax`, `__ArgMin` and the SPM mapping helper show register writeback followed by mapped SPM stores | Wafer CRT waits, maps value/index destinations and stores the two writeback fields |
| Peripheral factorize | no matching old source file was found; public headers expose the wrapper entry point | repo-local CRT header/source不含factorize symbol；compiler target lowering将该IR kind显式判为`unsupported_target_operation`，不能进入production closure |
| Peripheral elem-mask | no matching old source file was found; public headers expose the wrapper entry point | Wafer CRT contains the corresponding public-wrapper call |
| Peripheral bilinear / LUT / random | `__Bilinear`, `__Lut16`, `__Lut32` and `__RandGen` provide direct wrapper evidence | Wafer CRT contains the corresponding public-wrapper calls; bilinear scale is computed from shapes |
| Count / Direct DTE / composite helpers | old source contains `__Count`, `__Send`, an empty `__Recv`, GELU, MXFP, reduce-mul and layout helpers | these helper names are not observed in the repo-local CRT source snapshot; the old source alone does not establish reusable Wafer IR / ABI or completion semantics |

Direct DTE当前没有physical endpoint/slot和target CRT closure，target conversion在call emission前将
`wafer.instr.dte_*`判为`unsupported_target_transport`。同样，只有arena-relative offset、没有explicit
arena base binding的compiler-managed DDR allocation会判为`unsupported_target_address`；两者都不能由
runtime或CRT补做语义恢复。

## Evidence Limits

- Static source matching can show wrapper call order and field handling; it cannot establish compiler legality,
  instruction coverage, target ABI acceptance or runtime completion.
- Missing old source does not imply missing hardware capability, and an old helper name does not imply a reusable
  Wafer target symbol.
- Production closure contract、header prototype、repo-local implementation和closure result分别以编号设计、
  public header/source、checker与任务队列为准。

## Legacy Checker Compatibility

`tools/check_target_crt_conformance.py`当前仍做presence-only的历史文本检查。下面的token只为保持现有
gate可重放，不是状态、分类协议或ABI source；解除该耦合已由`tasks/progress.md`中的
`supporting-doc-tool-decoupling`单独排期。

- `direct-wrapper-derived`
- `public-header-derived`
- `intentionally-excluded`
- `mismatch-fixed-this-batch`

Pipeline position:
- Upstream artifact / IR: none; this file consumes source snapshots as evidence.
- Current stage responsibility: preserve an auditable static evidence comparison.
- Output artifact / IR: none.
- Downstream consumer: legacy presence-only conformance check; no compiler or runtime consumer.
- User-level driver / named pipeline: none.
- Explicit non-goals: owning production membership, IR, ABI, lowering, package or runtime policy.
- Completion gate: evidence statements remain source-backed and current contracts remain in numbered designs.
