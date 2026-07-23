# TX81 CRT Static Evidence Matrix

本文把旧DLCompiler TX81 CRT source、当前public headers/installed binary反汇编与repo-local Wafer CRT实现做静态对照。它只记录
source/disassembly-backed evidence，不拥有production membership、prototype/signature、IR、ABI、lowering或
runtime policy。当前IR / ABI和production closure合同查看`tasks/14-target-conversion-module-publication.md`，
prototype与repo-local实现分别查看`runtime/wafer_crt/include/wafer_tx81_crt.h`和
`runtime/wafer_crt/src/wafer_tx81_crt.c`；闭合状态查看`tools/check_target_crt_symbols.py`与
`tasks/progress.md`。

## Scope

- Evidence source：旧 DLCompiler `lib/Tx81` source snapshot、public TX8 headers，以及digest-qualified installed Kcore/vendor module反汇编。
- Repo-local observation：compiler target lowering以及`runtime/wafer_crt` public header/source中可直接
  读取的legality、ABI、wrapper调用、参数处理、wait/writeback和optional-feature配置。
- 本表不因旧helper存在而授权新symbol，也不定义某个family的production状态。
- 当前symbol checker从target lowering和Wafer enum registry推导出111个production symbol；这是当前
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
| GEMM | `__Gemm` records input/config/output calls and optional feature setters；current V5.6 Kcore source与SDK example均显示RHS raw hardware transpose bit和semantic transpose相反，semantic NN发`SetTransflag(0, 1)` | Wafer CRT保持public call中的orientation为semantic值，只在packet wrapper边界把RHS bit取反：v1 NN发`(0, 1)`，v2发`(lhs_orientation, !rhs_orientation)`；psum、bias、scale、quant和activation仍显式关闭 |
| Conv / Depthwise / BackwardConv | old `__Conv` and public headers expose NHWC/HWOI fields; old `__Conv` may enable ReLU by default | Wafer CRT uses explicit shape/pad/stride fields and explicitly disables optional/fused features |
| Pool / Unpool | no matching old source file was found; public headers expose wrapper entry points | Wafer CRT contains public-wrapper calls for pool and unpool families |
| TDMA pad / img2col | `__Pad` and `__Img2col` expose direct wrapper parameters; other old files contain additional transform helpers | Wafer CRT contains pad and img2col definitions; the other transform helper names are absent |
| Peripheral argmax / argmin | `__ArgMax`, `__ArgMin` and the SPM mapping helper show register writeback followed by mapped SPM stores | Wafer CRT waits, maps value/index destinations and stores the two writeback fields |
| Peripheral factorize | no matching old source file was found; public headers expose the wrapper entry point | repo-local CRT header/source不含factorize symbol；compiler target lowering将该IR kind显式判为`unsupported_target_operation`，不能进入production closure |
| Peripheral elem-mask | no matching old source file was found; public headers expose the wrapper entry point | Wafer CRT contains the corresponding public-wrapper call |
| Peripheral bilinear / LUT / random | `__Bilinear`, `__Lut16`, `__Lut32` and `__RandGen` provide direct wrapper evidence | Wafer CRT contains the corresponding public-wrapper calls; bilinear scale is computed from shapes |
| Direct DTE | old source contains `__Send` and an empty `__Recv`; current public/Kcore headers and installed firmware additionally expose direct sync、FSM、async send/wait/release、tile-id和peer-SPM helpers | Wafer CRT implements typed begin/begin-after-prepare、send/recv prepare、wait和finish status lifecycle。sender source与receiver FSM使用raw local SPM offset，sender destination使用`get_tile_spm_addr_base(remote,4,4)+offset`；status-v2以64-byte storage/alignment独占cache line，offset 0的`u32`写入cacheable DDR后执行C908 cache clean/invalidate |
| Count / composite helpers | old source contains `__Count`, GELU, MXFP, reduce-mul and layout helpers | these helper names are not observed in the repo-local CRT source snapshot; the old source alone does not establish reusable Wafer IR / ABI or completion semantics |

Direct DTE当前只消费compiler已经all-rank accepted的physical binding、remote receiver offset、token/wait和typed status slot；
target conversion/CRT不重新选择endpoint、FSM或route。cluster prepare先执行`init_tile_id(__get_pid(0),4)`，再执行
`direct_sync_init(16)`；main不重复清ready slots。只有arena-relative offset、没有explicit arena base binding的
compiler-managed DDR allocation仍会判为`unsupported_target_address`，不能由runtime或CRT补做语义恢复。

## Evidence Limits

- Static source/disassembly matching can show wrapper call order and field handling; it cannot by itself establish compiler legality,
  instruction coverage, target ABI acceptance or runtime completion.
- Missing old source does not imply missing hardware capability, and an old helper name does not imply a reusable
  Wafer target symbol.
- Production closure contract、header prototype、repo-local implementation和closure result分别以编号设计、
  public header/source、checker与任务队列为准。

Pipeline position:
- Upstream artifact / IR: none; this file consumes source and binary snapshots as evidence.
- Current stage responsibility: preserve an auditable static evidence comparison.
- Output artifact / IR: none.
- Downstream consumer: human/static evidence audit；conformance tools derive expected facts from code and do not parse this document.
- User-level driver / named pipeline: none.
- Explicit non-goals: owning production membership, IR, ABI, lowering, package or runtime policy.
- Completion gate: evidence statements remain source/disassembly-backed and current contracts remain in numbered designs.
