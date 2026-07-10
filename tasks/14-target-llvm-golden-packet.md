# Wafer Target LLVM Lowering and Golden Packet Design

状态：production target CRT implementation、device-code link gate 和 required-symbol closure 已覆盖当前
105 个 production `wafer_tx81_*` symbols；straight-line target LLVM call emission 已落地。2026-07-10
系统审计确认当前 lowering 会把 nested structured control flow 平铺到单一 LLVM entry block，因此
target LLVM semantic correctness 已重新打开；修复或 fail-closed 前，call-emission 子 gate 不算完成。
扩展 surface 只按第 8 节 staged matrix 推进，不能从旧 CRT source 逐个复制。
范围：
memory-planned target-aligned `wafer.instr.*` 到 target CRT call、structured LLVM dialect / LLVM IR、
compiler-generated `KernelAbiDescriptor`、environment/artifact fingerprints、device module 和 wrapper/register golden
packet 的 lowering。

本轮长期target/ABI/artifact-set边界已收敛；上述状态只描述当前实现证据和缺口，不降低本文合同。

本文取代旧 compiler-facing helper ABI 设计。当前结论是：`wafer.instr.*` 必须对齐目标指令、
TX81 target op 或 public TSM wrapper 的可 lower 粒度；LLVM lowering 生成 **Wafer-owned target
CRT symbol** 调用，例如 `wafer_tx81_gemm`、`wafer_tx81_bit2fp`、`wafer_tx81_mask_move`。TX81/Triton
CRT 中的 `__Gemm`、`__Bit2Fp`、`__MaskMove` 只作为实现证据和参数单位参考，不是 Wafer compiler
长期 ABI 名称。主线不再经过 Wafer 自定义 helper ABI，也不再编译或链接 capture shim。

参考 Triton/TX81 的有效分层：

```text
linalg / mk semantic op
  -> tx81 target op
  -> LLVM call @__Foo
  -> CRT __Foo
  -> TSM wrapper + TsmExecute
```

Wafer 后端采用同类边界：

```text
wafer.tile.* semantic/buffer op
  -> target-aligned wafer.instr.*
  -> LLVM call @wafer_tx81_foo
  -> Wafer CRT implementation
  -> TSM wrapper + TsmExecute or runtime/DTE helper
```

本文依赖：

- `tasks/10-compute-movement.md`
- `tasks/11-instruction-ir.md`
- `tasks/13-communication.md`
- `tasks/15-launch-runtime-package.md`
- `docs/wafer-register-level-instruction-spec.md`
- `docs/tx8-deps-reverse-engineering/tx8-interface-contract.md`
- MLIR Dialect Conversion: https://mlir.llvm.org/docs/DialectConversion/
- MLIR LLVM IR target: https://mlir.llvm.org/docs/TargetLLVMIR/
- Protobuf serialization is not canonical: https://protobuf.dev/programming-guides/serialization-not-canonical/
- NIST FIPS 180-4 SHA-256: https://csrc.nist.gov/pubs/fips/180-4/upd1/final
- ELF note format: https://gabi.xinuos.com/elf.pdf
- llvm-objcopy section injection: https://llvm.org/docs/CommandGuide/llvm-objcopy.html

## 1. 目标和非目标

目标：

- 定义 `wafer.instr.*` 到 Wafer-owned target CRT / LLVM call 的 lowering 边界。
- 使用 dialect conversion 保持 function、block、branch、loop 和 call 语义；unsupported structured
  container 在 mutation 前 fail-closed，不以 recursive walk 平铺 leaf instruction。
- 固定地址单位、SPM/DDR offset 消费、format、shape/stride、wait/completion 和 status 责任。
- 由 instruction verifier、target lowering preflight 和 ABI descriptor generation 共用 geometry legality，
  统一检查 shape/count/stride/iteration/range/narrowing，不能在 LLVM constant 或 CRT cast 时静默截断。
- 要求每个可 lower 的 `wafer.instr.*` 都有明确 public TSM wrapper、Direct DTE helper 或 TX81/CRT
  证据；证据不能直接替代 Wafer-owned symbol 合同。
- 通过 golden packet / wrapper tests 验证 target CRT 参数到 TSM wrapper/register packet 的映射。
- 由 compiler 生成精确 kernel entrypoint ABI descriptor；final ELF、package manifest 和 runtime
  只消费该 descriptor/hash，不从 LLVM 文本、参数名或 module path 猜 ABI。

非目标：

- 不恢复 compiler-facing helper ABI family。
- 不新增 `wafer.abi` dialect。
- 不把 capture shim、C stub 表格或 hand-written package/JSON input 当 production lowering。
- 不重新选择 group、tile shape、layout、SPM/DDR memory plan 或 communication schedule。
- 不让 package exporter 解析 LLVM 文本重建 function ABI，不维护与 compiler function/resource view
  平行的 binding-order 或 geometry 事实源。
- 不把 Tx81 CRT 的所有符号原样提升为 Wafer IR 或 Wafer target ABI；Wafer IR 只表达当前 pipeline
  需要且 verifier 能检查的目标动作。

## 2. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  whole-variant atomic commit后的`wafer.executable.entry/resource/transport`、committed projection，以及对应
  static rank function中的memory-planned target-aligned `wafer.instr.*` IR和accepted SPM/DDR offsets。
- Current stage responsibility:
  从 target-aligned instruction IR 和 accepted facts 派生 Wafer-owned target CRT calls，并继续 lower 到 LLVM
  dialect / LLVM IR。该阶段消费Wafer memory attr、SPM/DDR offset、DTE peer/token、layout/format和typed
  executable bindings；不读取pass-local side table，不发明compiler-facing ABI wrapper。先以shared geometry
  legality 和 full-module conversion target 做无 mutation preflight，再在原 control-flow / block 位置改写
  leaf instruction；标准 SCF/CF/function 容器通过 dialect conversion 保持结构，尚未支持的 region、
  multiblock或call结构化失败。该阶段同时从converted function boundary、typed executable entry/resource/
  transport/projection refs生成`KernelAbiDescriptor`，不解析已打印LLVM IR。
- Output artifact / IR:
  LLVM dialect module、LLVM IR artifact、Wafer target CRT symbol declarations/calls、canonical
  `KernelAbiDescriptor`及其hash、`TargetEnvironmentFingerprint`、`TargetArtifactFingerprint`和debug/golden-packet输入。TX8 relocatable object、
  带mandatory `.note.wafer.abi`的 kcore shared object、module digest 和 symbol report 由 device-code gate
  从这些 compiler-generated artifacts 继续生成。
- Downstream consumer:
  device-code compile/link gate、IR-derived `PackageManifest` assembly、wrapper-facing call contract 和
  board/runtime adapter。
- User-level driver / named pipeline:
  production主线由`wafer-opt --program-pipeline=stablehlo-to-executable`选择的direct owner-aware driver消费committed
  executable；局部pass `--wafer-lower-instr-to-target-llvm`和group named pipeline
  `wafer-lower-groups-to-target-llvm`只作debug/regression索引，不能绕过atomic commit。
- Explicit non-goals:
  不重新做 frontend/SPMD/group/tile/layout/SPM/DDR/communication planning；不把 helper ABI、
  capture shim 或 C stub emission 作为中间层；不从 LLVM 文本、symbol spelling 或参数数量恢复 kernel
  ABI；不把 physical runtime handle 写入 descriptor。
- Completion gate:
  call-emission 子 gate 要求 supported `wafer.instr.*` 生成 verifier-legal LLVM dialect / LLVM IR，并能由
  `mlir-translate` 输出 LLVM IR；SCF/CF/function 的分支、循环、block 和 call 语义必须保持，尚未支持的
  容器在 mutation 前结构化失败。至少用 false branch、不同 loop trip count、nested branch 和 function
  call 证明语义，不只检查 emitted call 数量；geometry 边界值和越界值必须在 shared legality 中分别通过
  和 fail-closed。完整 target CRT / golden boundary 还要求 Wafer-owned CRT symbol 有 typed wrapper 合同
  和 packet golden coverage。device-code gate 只链接 target LLVM object、Wafer CRT object 和 repo-vendored
  TX8 deps，不默认链接 capture shim，并负责 required-symbol closure、ELF descriptor/hash、target
  fingerprint 和 module digest 验证；package manifest引用的 descriptor digest必须与
  `.note.wafer.abi`一致。
```

## 3. Instruction Legality

`wafer.instr.*` 是 target instruction-ish IR，不是新的 semantic buffer IR。进入本 stage 前，每个 op
必须满足：

- operand/result buffer 是 Wafer-tagged memref，SPM/DDR domain 可从 type 和 accepted offset facts 推出。
- op family、dtype/layout、shape/stride、byte count、wait/completion 和 resource effects 能由 verifier
  检查。
- 所有进入固定 `i32` / wrapper field 的 geometry 都通过 shared legality：数值范围、element-size
  整除、descriptor coverage 和 target field width 明确；lowering 不能依赖 integer attr 构造或 C cast截断。
- leaf instruction 所在的 region/block/call 结构属于 target lowering 已证明可保持的 subset；否则在
  生成任何 LLVM function/call 前拒绝，不能用 recursive walk 丢弃容器语义。
- lowering 可以找到 Wafer-owned target CRT symbol 方案或明确记录 unsupported diagnostic。
- coverage 必须来自 `tasks/11-instruction-ir.md` 的 instruction coverage matrix。只有标为
  `V0 production target op` 或 `V0 production target sync` 的 op 是 target LLVM call emission 的 production 输入；
  `V0 composite lowering` 必须已经在 instruction lowering 前展开，`future` / `unsupported`
  不能通过现有泛 op 隐式进入 LLVM lowering。

production conversion的唯一C++ owner是`WaferInstrToTargetLLVM`中的
`wafer::conversion::VerifiedTargetConversionRequest`，不是`WaferTargetArtifacts`或pass-local raw request。该type是
move-only、non-aggregate、private-construction closed union：`EntryCore`或`CloneDependency`。唯一factories固定为：

```text
verifyEntryCoreConversionRequest(
    OwningOpRef<ModuleOp> isolatedInput,
    SmallVector<VerifiedTargetEntryBoundary> nonemptyBoundaries,
    const compiler_identity::VerifiedStaticClosureUnit &closure,
    const compiler::VerifiedTargetCompilationContext &context)
  -> VerifiedTargetConversionRequest<EntryCore>

verifyCloneDependencyConversionRequest(
    OwningOpRef<ModuleOp> isolatedInput,
    const compiler_identity::VerifiedCloneableDependency &dependency,
    const compiler::VerifiedTargetCompilationContext &context)
  -> VerifiedTargetConversionRequest<CloneDependency>

convertVerifiedWaferInstrModuleToTargetLLVM(VerifiedTargetConversionRequest)
  -> ConvertedTargetLLVMModule
```

`OwningOpRef<ModuleOp>`不拥有`MLIRContext`。`VerifiedTargetEntryBoundary`和`VerifiedCloneableDependency`的private storage必须从
committed executable owner继承同一个nonsemantic MLIR-context lifetime share；两个request factory逐项重验该share owner/
generation与`isolatedInput->getContext()`一致并把它保留进request，`ConvertedTargetLLVMModule`继续保留直到owned module销毁。
factory不接受独立`MLIRContext`/lifetime token参数，caller也不能从proof提取或替换它；cross-context unit、owner提前析构或仅move
module不move context share都在conversion mutation前失败。

`ConvertedTargetLLVMModule`是move-only owner且不公开`ModuleOp`/`LLVMFuncOp`/`Operation *`、权威borrowed view或mutator。
它只按值签发owner-bound `ConvertedTargetEntryHandle`和non-authoritative summary；KAD API必须同时接收live converted-module
owner与该owner签发的handle并重验owner/ordinal。实际owned module、sealed boundary与`handle -> LLVMFuncOp`映射只能由
`compiler_identity::detail::KernelAbiBuildAccess`和
`target::detail::ConvertedTargetModuleAccess`在owner lifetime内借用。KAD builder和prelink不能取得可逃逸handle，任何rewrite
都会使owner无效并要求重新conversion，不能在conversion成功后静默修改function再复用boundary/KAD proof。

`EntryCore`要求至少一个由sealed committed executable entry/resource/completion relation构造的
`VerifiedTargetEntryBoundary`，且closure root/entry/slot/context generation全部匹配；KAD随后从该boundary与converted
function共同构造并cross-check，不能反向成为request的先决条件。`CloneDependency`要求零entry
boundary并重放pure private helper/immutable non-address-significant global、无escape/effect/recursive SCC及固定linkage/COMDAT
proof。两种request都先完成structure/geometry/full conversion legality再允许mutation，不能互相转换或由caller填kind。
production不存在接受raw `ModuleOp + symbols/profile/options`的conversion overload；replay test只能通过受限test adapter走同一
verifier。

依赖方向固定为：`WaferTargetLegality -> WaferIR/WaferCompilerIdentity/WaferABI`，
`WaferInstrToTargetLLVM -> WaferTargetLegality/WaferCompilerIdentity/WaferABI`，
`WaferTargetArtifacts -> WaferInstrToTargetLLVM`。conversion library不得依赖`WaferTargetArtifacts`；artifact builder负责形成
closure/core/clone inputs和消费converted result，但不能复制request/boundary/verifier owner。

low-precision与DTE的C command ABI也遵守该依赖方向。唯一layout/prototype owner是runtime-safe
`include/Wafer/ABI/Tx81CommandAbi.h`：header使用fixed-width C types、explicit version/size/alignment和`extern "C"` guards，
repo-local Wafer CRT C source与C++ compiler共同include，不能在`WaferTargetArtifacts`、conversion或runtime复制mirror struct/
prototype。WaferABI另提供immutable/private-construction command values和逐字段validator；ABI version、field policy和CRT
symbol-set进入target artifact fingerprint。

`WaferCompilerIdentity`中的唯一resolver从committed instruction op、matched `QuantStorageAbiProfileV1`、accepted
transport/projection/control-status slots和`VerifiedTargetCompilationContext`构造non-forgeable
`VerifiedTx81CommandView`。`VerifiedTargetEntryBoundary`/`VerifiedTargetConversionRequest`sealed持有complete op-to-command views；
`WaferInstrToTargetLLVM`按typed op occurrence join view并生成共享header声明的fixed call，不能接raw command aggregate、
自由symbol或include/link `WaferTargetArtifacts`。DTE view缺endpoint/channel/FSM/receiver/status/completion任一字段即失败；
low-precision view缺q0/q1/zp/scale/storage/profile或native/composite relation即失败。artifact tasks只消费converted object和共享
ABI/CRT implementation做link/golden，不再给conversion提供`Target/LowPrecisionAbi`或`Target/DTEAbi`头。

当前状态：

| Wafer instr | target evidence / lowering direction | 状态 |
| --- | --- | --- |
| `wafer.instr.rdma` / `wdma` | TSM RDMA/WDMA wrapper；target CRT 接收 DDR/SPM pointer/offset、shape/stride、format | typed LLVM call emitted；repo-local CRT / device-link gate covered |
| `wafer.instr.gather_scatter` | `TsmDataMove::GatherScatter` / TX81 gather-scatter CRT evidence | typed LLVM call emitted；repo-local CRT / device-link gate covered |
| `wafer.instr.fill` | `TsmPeripheral::Memset` evidence | typed LLVM call emitted；repo-local CRT / device-link gate covered |
| `wafer.instr.elementwise` | `#wafer.instr_elementwise_kind` CT arith/relation/activation/transcendental target wrapper families；select 不在该 enum 中 | per-kind typed LLVM call emitted；repo-local CRT / device-link gate covered |
| `wafer.instr.bit2fp` | Triton/TX81 `mk.bit2fp -> tx81.bit2fp -> __Bit2Fp` as evidence; Wafer lowering emits Wafer-owned CRT symbol | typed LLVM call emitted；repo-local CRT / device-link gate covered |
| `wafer.instr.mask_move` | Triton/TX81 `mk.mask_move -> tx81.mask_move -> __MaskMove` as evidence; Wafer CRT explicitly adapts SPM mask address to wrapper `uint32_t` mask field | typed LLVM call emitted；repo-local CRT / device-link gate covered |
| `wafer.instr.reduce` | `#wafer.instr_reduce_kind` target wrapper families plus native `dim` code | per-kind typed LLVM call emitted；repo-local CRT / device-link gate covered |
| `wafer.instr.convert` | `#wafer.instr_convert_kind` opcode 139..174 dtype pair plus zero-point/rounding/plain signature groups；same-format copy must lower through movement, not convert | per-kind typed LLVM call emitted；repo-local CRT / device-link gate covered |
| `wafer.instr.gemm` | `TsmGemm` wrapper / `__Gemm` style CRT evidence; Wafer-owned symbol must not expose Tx81 ABI verbatim | typed LLVM call emitted；repo-local CRT / device-link gate covered |
| `wafer.instr.conv` | `#wafer.instr_conv_kind` NE Conv/Depthwise/BackwardConv target wrapper families；optional/fused operands are not implicit | per-kind-family typed LLVM call emitted；repo-local CRT / device-link gate covered |
| `wafer.instr.pool` / `unpool` | `#wafer.instr_pool_kind` / `#wafer.instr_unpool_kind` CT Pool/UnPool wrapper families; unpool scalar index is wrapper-aligned | per-kind typed LLVM call emitted；repo-local CRT / device-link gate covered |
| `wafer.instr.tdma_data_move` | V0 production only covers pad/img2col wrappers; ordinary copy and transform-like movement use gather/scatter | pad/img2col LLVM call emitted；transform-like kind 结构化失败 |
| `wafer.instr.peripheral` | `#wafer.instr_peripheral_kind` for arg/factorize/bilinear/LUT/rand/elem_mask; count writeback and bitcount remain unsupported | per-kind typed LLVM call emitted；repo-local CRT / device-link gate covered |
| `wafer.instr.dte_send` / `dte_recv` / `dte_wait` | Direct DTE/FSM runtime binding evidence still incomplete | call-emission can produce logical peer/bytes calls；excluded from current production CRT closure until ABI / runtime endpoint binding is fixed |
| `wafer.instr.local_fence` | `TsmWaitfinish` / local drain evidence | typed LLVM call emitted；repo-local CRT / device-link gate covered |

Target LLVM call emission 输出必须调用 Wafer-owned target CRT symbols。`__*` 符号可以出现在 Wafer CRT
内部实现、golden packet test 或 TX81/Triton 对照说明中，但不能作为 compiler IR lowering 的长期
ABI 名称。

### 3.1 Wrapper Signature Groups

这张表约束 target LLVM pass 的 symbol/descriptor 分组。不能用一个“万能 ABI symbol”吞掉同一
family 下参数不同的 kind；若 public wrapper signature 不一致，要么拆 Wafer-owned symbol family，
要么生成显式 descriptor struct，并由 verifier 保证字段完整。

| `wafer.instr.*` group | Wafer-owned target CRT 方向 | public wrapper signature group | 当前 IR 必须提供的字段 / 处理 |
| --- | --- | --- | --- |
| RDMA / WDMA contiguous/strided | `wafer_tx81_rdma*` / `wafer_tx81_wdma*` | direction-specific RDMA/WDMA wrappers | DDR/SPM address、`byte_count`、`inner_bytes`、3-level stride/iteration、format；external DDR base 来自 launch binding，compiler-managed DDR base 来自 accepted DDR offset |
| `gather_scatter` | `wafer_tx81_gather_scatter` | `TsmDataMove::GatherScatter` | SPM source/dest address、inner byte count、source/dest stride-iteration descriptor；copy/layout/slice/broadcast 都必须已展开成这一组 |
| `fill` | `wafer_tx81_memset` | `TsmPeripheral::Memset` | SPM dest address、scalar value、element count/format；不表达 peripheral count/writeback |
| elementwise arith unary | `wafer_tx81_ew_unary` or per-kind symbols | `TsmArith` unary wrapper | source、dest、element count、format；无 rounding |
| elementwise arith VV | `wafer_tx81_ew_arith_vv` or per-kind symbols | `TsmArith::*VV` | two sources、dest、element count、rounding/reserved mode、format；VS/VuV/Loop 不是当前 IR 形态 |
| relation / logic | `wafer_tx81_relation_*` / `wafer_tx81_logic_*` | value/bool variants differ | source arity、dest storage、element count、format；relation 的 `format` 来自 input element type，不能用 `i1` dest format 替代；bitpacked bool/VuV/Loop 需要后续扩 IR，不能由当前 generic elementwise 隐式选择 |
| activation / transcendental | `wafer_tx81_activation_*` / `wafer_tx81_trans_*` | unary wrappers | source、dest、element count、format；kind 决定 wrapper |
| `bit2fp` | `wafer_tx81_bit2fp` | TX81 `__Bit2Fp` evidence | i1/bool source、FP dest、element count、format |
| `mask_move` | `wafer_tx81_mask_move` | TX81 `__MaskMove` evidence; header/CRT mask parameter differs | source、mask、dest、element count、format；Wafer CRT must settle mask address width explicitly |
| `reduce` | `wafer_tx81_reduce` or per-kind reduce symbols | `TsmReduce::{ReduceSum,ReduceAvg,ReduceMax,ReduceMin}` | source、dest、native `dim` code、shape descriptor、format、optional init handling; tile semantic `dimensions` 不进入 target LLVM |
| `convert` zero-point group | `wafer_tx81_convert_zp` or per-kind symbols | INT8->FP wrappers | source、dest、`zero_point`、element count；`rounding_mode` forbidden |
| `convert` rounding group | `wafer_tx81_convert_round` or per-kind symbols | FP/INT wrappers with `RND_MODE` | source、dest、`rounding_mode`、element count；`zero_point` forbidden |
| `convert` plain group | `wafer_tx81_convert_plain` or per-kind symbols | wrappers without extra params | source、dest、element count；`zero_point` / `rounding_mode` forbidden |
| `gemm` | `wafer_tx81_gemm` | `TsmGemm` wrapper sequence | lhs/rhs/dest addresses、M/K/N、batch attrs、format/layout flags；bias/scale/activation/quant must remain disabled unless IR is extended |
| `conv` | `wafer_tx81_conv` / `wafer_tx81_depthwise_conv` / `wafer_tx81_backward_conv` | `TsmConv` / `TsmDepthwiseConv` wrappers | input/weight/dest addresses、rank-4 shape descriptors、pads/unpads/strides/dilations、format; optional/fused operands are not implicit |
| `pool` | `wafer_tx81_pool` / indexed variants | normal pool vs indexed pool dest arity differs | input address、value dest、optional index dest for indexed max/min、source/dest shape、pads、strides、format |
| `unpool` | `wafer_tx81_unpool` / `wafer_tx81_unpool_avg` / mask/indexed variant | `Unpool/UnpoolIdx` take scalar `uint32_t index`; `UnpoolAvg` does not | input/dest addresses、source/dest shape、strides、optional scalar `index` according to kind |
| `tdma_data_move` pad/img2col | `wafer_tx81_tdma_pad` / `wafer_tx81_tdma_img2col` | pad and img2col signatures differ | pad requires `pads`; img2col requires `pads` + `kernel_strides`; both forbid unrelated attrs |
| transform-like TDMA group | no production Wafer symbol in V0 | mirror/transpose/rotate/NCHW-NHWC/TensorNom wrappers exist but are not default lowering surface | instr lowering must materialize these movements before target LLVM/package export; transpose uses `permutation`, mirror/rotate use `axes`; future target enablement requires board/golden coverage and an explicit target-surface update |
| `peripheral` arg/factorize/rand | `wafer_tx81_peripheral_*` | arity differs by kind | verifier fixes input/dest arity; `format` comes from first input; arg index dest must be i32 |
| `peripheral` bilinear/LUT/elem_mask | separate Wafer symbols or descriptor variants | bilinear uses shape attrs; LUT uses `lut_elem_count`; elem_mask uses scale/probability/rounding | fixed ABI includes first-input `format`; required/forbidden attrs already checked by IR/package validator；bilinear scale is derived from source/dest shape |
| `peripheral count` | no production symbol | public wrapper writeback is not represented as normal dest buffer | verifier rejects until IR gains explicit writeback/result semantics |
| Direct DTE | `wafer_tx81_dte_send` / `wafer_tx81_dte_recv` / wait helper | Direct DTE/FSM helper, not `TsmExecute` | current call-emission passes only logical peer and bytes; Q1 excludes these calls from production CRT closure until endpoint/channel binding is expressible |
| local fence | `wafer_tx81_local_fence` | local wait/drain helper | no hidden multi-tile barrier semantics |

不在 V0 production target surface 中的硬件能力不属于本 stage 的默认 lowering surface。concat、maskgather
variants、raw DTE non-unicast、SCALAR/CSR ordinary execution、Peripheral bitcount 和 Conv optional/fused
operand policy 等能力需要先在 instruction IR 中新增或扩展明确 op / kind / verifier，并更新 coverage
matrix；target LLVM call emission 不能用 `wafer.instr.elementwise`、`gather_scatter` 或 ad hoc CRT call
隐式覆盖这些语义。

`wafer.instr.elementwise <select>` 非法。tile semantic select 需要在 instruction lowering 中改写成目标
序列：

```text
false_value -> dest                    // SPM copy / gather_scatter
predicate i1 -> floating mask           // wafer.instr.bit2fp
true_value + mask -> dest               // wafer.instr.mask_move
```

这与 Triton/TX81 的 select lowering 对齐，避免生成不存在的 Wafer 自定义 select helper。

## 4. LLVM Call Shape

LLVM lowering 直接声明/调用 Wafer-owned target CRT symbol。target LLVM pass 在 LLVM dialect 中生成
fixed function declaration 和 typed `llvm.call`；地址统一为 `i64`，descriptor / enum / shape /
stride / iteration / format 字段统一为 `i32`：

```mlir
llvm.func @wafer_tx81_gemm(!llvm.i64, !llvm.i64, !llvm.i64, !llvm.i32,
                           !llvm.i32, !llvm.i32, !llvm.i32, !llvm.i32)
llvm.call @wafer_tx81_gemm(%lhs, %rhs, %dest, %m, %k, %n, %batch, %fmt)
```

这不是 compiler-facing 万能 helper ABI：symbol 仍按 Wafer target action / kind family 拆分，
参数仍从当前 IR、accepted offset facts 和 verifier-checked attrs 派生。`wafer_tx81_crt.h`
是 C ABI 的显式 prototype 表；lowering、repo-local CRT source 和 device link gate 必须保持一致。

示例形态：

```mlir
// before
wafer.instr.bit2fp %pred into %mask
  : memref<8xi1, #wafer.memory<spm, tensor>>
  to memref<8xf32, #wafer.memory<spm, tensor>>

// after target lowering, schematic
llvm.call @wafer_tx81_bit2fp(%pred_addr, %mask_addr, %elem_count, %fmt)
```

```mlir
// before
wafer.instr.mask_move %src, %mask into %dst
  : memref<8xf32, #wafer.memory<spm, tensor>>,
    memref<8xf32, #wafer.memory<spm, tensor>>
  into memref<8xf32, #wafer.memory<spm, tensor>>

// after target lowering, schematic
llvm.call @wafer_tx81_mask_move(%src_addr, %dst_addr, %elem_count, %mask_addr, %fmt)
```

具体 symbol、参数顺序和返回值以 Wafer-owned CRT 头文件/实现为合同，并由 repo-local TX81/public
TSM wrapper 证据验证。TX81 `__*` 名字不是 compiler lowering contract；没有证据的 op 不能 silent
fallback 到 fake ABI 或万能 helper。

### 4.1 Kernel ABI Descriptor

Target CRT call ABI 和 model kernel entrypoint ABI 是两个不同合同。`wafer_tx81_crt.h` 约束 module 内部
call；compiler-generated `KernelAbiDescriptor` 约束 runtime 调用 module entrypoint。descriptor 必须由
`wafer.executable.entry`、converted function boundary和同一 typed resource/launch view直接生成，至少包含：

- kernel ABI schema/version、LLVM symbol和return type。`KernelAbiDescriptor`的WCRE semantic digest是唯一
  ABI identity，不再引入可独立变化或自引用的`EntryAbiId`。executable variant、shape guard、
  rank/stage class和endpoint mapping由executable/package entry graph关联，不进入函数调用ABI本体。
- ordered argument slots；每个slot使用ABI-local稳定`SlotId`并记录role、LLVM scalar/pointer
  representation、address space、access、alignment、shape/capacity relation、alias contract和typed
  `StateSlotVersionRole = none | current | candidate | in_place`。非state slot只能是`none`；其它值必须与
  executable state-group policy及entry binding逐项一致。具体
  `ResourceId`和state version role由`wafer.executable.entry`的
  `SlotId -> (ResourceId, StateSlotVersionRole)` binding提供，不进入descriptor semantic digest；参数
  顺序是descriptor的typed sequence，不另存可独立修改的`binding_order`。
- external IO、immutable weight、persistent state、workspace和 relocatable endpoint/control/status resource
  slots；不存在的 role 不生成占位参数，存在的 slot 必须与 LLVM function parameter 一一对应。
- low-precision launch-visible DDR packed source、scale/zero-point resource、stream staging或cross-entry scratch root也只
  通过普通typed `SlotId`/resource slots表达；descriptor记录slot role、semantic/storage type、shape/capacity/alignment/
  access和alias，不嵌具体业务ResourceId。entry内部可从static function use-def/SPM offset重算的decode destination、
  scale copy和scratch仍由instruction IR/SPM planner拥有，不升级成executable resource/KAD slot。
  `wafer.executable.entry` binding负责关联launch-visible source/staging及stream window，KAD/resource verifier沿typed
  load/use-def证明它们覆盖内部SPM operands。
- pinned transport projection hash或relocatable transport control slots，以及ABI-local
  `CompletionExportId`/typed device completion-error exports。executable entry必须把每个export一一绑定到
  variant completion DAG node；descriptor不复制instruction list或communication schedule。
- `TargetEnvironmentFingerprint` dependency、必要projection/relocation ABI dependency和descriptor canonical hash。

`KernelAbiDescriptor` identity只取决于函数调用合同、typed `SlotId`s、completion exports和必要target ABI /
transport dependency；不以`ExecutableVariantId`或`RankClassId`加盐。因此ABI相同的shape variants或ranks可
共享descriptor/module；若pinned transport进入code，projection hash属于必要target dependency并自然导致
artifact分裂，relocatable artifact则通过typed control slots保持共享。

Descriptor generation 和 LLVM function conversion 必须在同一 compiler transaction 中完成：conversion
失败时不产生 descriptor；descriptor verifier 失败时不输出 LLVM/module artifact。禁止 package tool 用
regex统计 `i64` 参数、用参数名猜 role，或从手写 LLVM IR 补造 ABI。

单function transaction不是发布边界。target stage只能通过compiler-private
`compiler::detail::TargetProgramOutputAccess::beginTargetArtifactBuild(compiler::ProgramOutputTransaction &,
compiler::StagedExecutableToken,
TargetVariantId, TargetArtifactBuildLimits)`取得move-only、non-aggregate `TargetArtifactBuildSession`。session一次绑定
同一transaction的稳定`CanonicalEncodingOwnerToken`、受限canonical encoding session、exact
`VerifiedTargetCompilationContext`、outer budget/cancellation和`VerifiedTargetStagingArea`；该context只能从staged executable
typed commitment move-own的exact `VerifiedTargetCompilationContextRegistry`按`TargetVariantId`解析，原`CompilationRequest`销毁后仍有效。caller不能分别传入或替换
encoding context、staging root、toolchain profile、target context或limits。session/attachment-table generation在每次tool launch、
readback和attach前重验，任一错配把outer transaction标为uncommittable。

`VerifiedTargetStagingArea`只由上述session创建并保有root capability/namespace generation。写入后的object只能以
private-construction `StagedObjectRef`流转；该ref绑定transaction owner、staging generation、object kind、exact size/content
digest和same-object capability identity，但不公开`acquireReadLease/open/read`且不常驻每object FD。只有live
`TargetArtifactBuildSession`的private access可按stage/outer limits签发move-only `TargetBuildReadLease`并从ref取得
same-handle opened lease；opened lease持有reader/FD/buffer预算到销毁，cross-session/generation或提前release失败。
prepare/link/note injection/final verification只接session或其refs，不接raw path/
caller-selected directory；内部tool adapter即使需要path也只能从session临时映射，并在同一handle上stat/read/hash/stat，不能
verify后按path reopen。这样target child staging同outer limits、TOCTOU和rollback合同一致。

一个`TargetVariantId`引用的全部rank-class modules必须先写入staging
artifact set，逐个完成structure-preserving conversion、device link、required-symbol、ELF ABI descriptor/
hash、environment/artifact fingerprints和module digest验证，再原子发布完整`TargetArtifactSet`。任一module失败则该set
整体不可见，PackageManifest不得引用已成功的子集。content-addressed cache可以保留独立verified blob，
但cache存在不等于variant artifact set已accepted。

`TargetArtifactSet` root绑定source `wafer.executable` program semantic digest和`TargetVariantId`；稳定
`TargetArtifactSetId`就是由canonical ordered member map、environment/artifact fingerprints和全部member
digests计算。每个member显式映射
`EntryId + static-function semantic digest -> final module content digest + entry symbol + KernelAbiDescriptor semantic digest + covered
RankClassIds`。相同ABI不能授权替换不同code digest或其它executable的module。set verifier还必须检查每个
`CompletionExportId`与executable entry/DAG引用一一闭合。

Device-code gate 将 descriptor delivery bytes与semantic digest写入 final ELF的必选Wafer ABI note，
并从包含该note的final ELF计算module content digest。`PackageManifest`记录同一descriptor
semantic digest、environment/artifact fingerprints和module digest；loader在module load和launch前逐项
比对。manifest自身不嵌回ELF，因此module digest、descriptor digest和manifest引用之间
没有循环依赖。ELF exported descriptor symbol不是V1 ABI；后续若为provider discovery增加
symbol，它只能指向与note相同的bytes，不能成为第二个descriptor owner。

### 4.2 Canonical Encoding, Digest, and ELF Note Contract

Protobuf delivery bytes、compiler semantic identity和artifact content checksum是三个不同合同。
Protobuf的deterministic serialization可用于同一pinned producer的reproducible output，但不是
canonical serialization，不能直接作为长期semantic ID。Wafer V1固定一个独立、有版本的
`Wafer Canonical Record Encoding`（WCRE）：它只是从已验证typed object生成digest preimage的
deterministic projection，不承载额外语义，不可反向恢复或修改compiler IR。

WCRE V1 byte encoding固定为：

```text
record := 0x31 || u32be(record_type) || u32be(schema_version)
               || u32be(field_count) || field...
field  := u32be(field_number) || u64be(value_size) || value

false  := 0x01
true   := 0x02
u64    := 0x10 || u64be(value)
i64    := 0x11 || u64be(two_complement_bits)
bytes  := 0x20 || u64be(size) || raw_bytes
ascii  := 0x21 || u64be(size) || ASCII_bytes
list   := 0x30 || u64be(count) || (u64be(element_size) || element)...
union  := 0x32 || u32be(discriminant) || u64be(payload_size) || payload
```

编码实现必须对百万级records保持bounded memory。所有record-specific identity factories显式接收validated
move-only/non-aggregate `CanonicalEncodingContext`，由
`createCanonicalEncodingContext(CanonicalEncodingLimits, unique_ptr<CanonicalScratchStore>)`唯一构造；limits至少约束recursive depth、fields/
nested/list/set elements、single/total encoded bytes、inline bytes、set-sort scratch bytes/runs/open FDs和concurrent
encoders，全部positive checked且0不表示unbounded。算法固定为两遍：

context私有拥有scratch、concurrency/FD tokens和collision registry，只公开const limits accessor；没有default/global/
loose limits overload。compiler的outer transaction/request持有一个context并把non-const ref显式传给model/distributed/
executable/static/target/package record factories；runtime loader从deployment validated policy创建独立context。large
`CanonicalRecordBackingRef`自持shared immutable owner，可安全越过context lifetime。

1. `measureCanonicalRecord`先做schema/unknown-field/limit验证，用checked arithmetic得到每个field/value/list element的
   exact encoded size和总size，不输出bytes、不按untrusted count reserve。
2. `streamCanonicalRecord`按measurement向`CanonicalRecordSink`顺序写exact WCRE bytes；sink可同时更新domain-separated
   SHA-256、exact content digest和bounded file/CAS writer，不要求一个contiguous `SmallVector`。

ordered list直接流式写；set/member map仍严格按完整element WCRE bytes lexicographic排序，不能改成digest排序。小set在
inline limit内可共享immutable buffers；大set把每个已测element编码到scratch object，以bounded runs进行deterministic
external merge sort，比较prefix后在collision时流式比较完整bytes。scratch位置、run size、worker count和memory limit
不进入identity；充分limits下必须产生byte-identical stream。超限返回typed limit failure，不截断set或改变order。

public strong ID wrapper只保存domain-specific digest，不为每个ID复制canonical record。需要collision proof或delivery
bytes的factory返回共享immutable `CanonicalRecordBackingRef(exact size, content digest, backing owner)`；small record可
inline、large record可spill/CAS，但ref/locator不进入semantic identity。transaction-local collision registry遇到同typed
semantic digest时流式比较backing exact bytes/verified record，non-equivalent hard fail。任何返回unbounded
`SmallVector<uint8_t>`的public encoder非法；测试-only inline helper必须先证明`size <= maxInlineRecordBytes`。

- record field按field number严格递增；对应Protobuf message时复用同一field number。
- top-level `record_type`固定为：`1 target_environment`、`2 topology_snapshot`、
  `3 execution_mesh`、`4 projection_set`、`5 executable_semantic`、`6 static_function`、
  `7 kernel_abi`、`8 target_artifact`、`9 target_artifact_set`、`10 package_manifest`、
  `11 model_interface`、`12 distributed_program`、`13 model_boundary_resource_id`、
  `14 model_internal_resource_id`、`15 model_dimension_id`、`16 executable_resource_id`、
  `17 target_variant_id`、`18 executable_variant_id`、`19 entry_id`、
  `20 quant_storage_abi_profile`、`21 state_migration_plan`。
  `0`只用于由父field schema已唯一定位的nested record；其它type id未经本文更新不得发布。
- optional缺失时不编码；但所有identity-critical field必须先由semantic verifier证明存在。
- list若表达序列则保留语义顺序；若表达set/member map则先按元素WCRE bytes升序。
  不允许map、native float/NaN normalization、locale-dependent text或不定长整数进入V1 identity record；
  registered float type/attr必须编码float semantics enum和exact fixed-width bit-pattern bytes。
- identity string必须是ASCII；人类名称、路径、diagnostic和任意Unicode text不进入identity。
- 对当前schema version不识别的Protobuf unknown field在semantic digest前结构化失败，
  不得被忽略或以serializer顺序混入hash。

record 11 `model_interface`只编码model-local resource/dimension keys及relations以避免self-digest；record 13/14
`model_boundary_resource_id`/`model_internal_resource_id`必须编码完整`ModelInterfaceSemanticId` owner和tagged local
key，record 15 `model_dimension_id`必须编码同owner、typed public ResourceId和dimension ordinal。public factory不提供
无owner resource/dimension digest入口。不同model-interface records即使local keys完全相同也必须产生不同typed IDs；
model semantic/content变化导致owner及scoped IDs变化是V1明确的版本隔离，不得由schema adapter“稳定化”。

WCRE field number/value kind/order semantics的唯一registry由repo-owned schema持有：

- `schema/wafer/semantic_identity.proto`定义IR-owned的model interface、distributed program、target
  environment、topology、mesh、projection、committed executable、static function、target artifact和set
  identity projection，以及versioned nested `QuantizationDescriptor`、`StorageEncodingDescriptor`和verified
  quant/storage ABI profile registry entry；这些message只由verified IR/
  target build profile构造，不是可导入的第二份program/executable格式。
- `schema/wafer/kernel_abi.proto`定义ELF note交付和KAD identity fields。
- `schema/wafer/target_artifact_set.proto`定义locator-free `VerifiedTargetArtifactSetRecord`和独立versioned
  `TargetArtifactSetDeliveryRoot`。前者只import/reuse `semantic_identity.proto`中的set identity、typed set/member
  keys和验证evidence，可被package schema安全引用；后者才增加final blob digest/size和build-output delivery locator。
  delivery-only message/field没有WCRE options且不能嵌入任何semantic record。locator不进入set semantic digest，
  package bundle也不能复用这个build locator。
- `schema/wafer/package_manifest.proto`定义package wire message及其identity projection。
- `schema/wafer/state_migration.proto`定义old/new model-interface和canonical migration components；每个component包含
  nonempty old/new state-group sets、各组complete typed member sets和exact descriptor/slot/shape/storage relation。
  `full_copy`/`page_cow_snapshot`要求group/member bijection，并用canonical `StateMigrationScopeSelectorV1` closed union和显式
  all-and-only `StateMigrationScopePairV1`保存scope bijection；`trusted_transform`才允许每个KAD input/output slot显式引用selector的
  bounded N:M reshard/split/merge。完整relation和所有required `(TargetVariantId, TargetArtifactSetId,
  TargetArtifactMemberKey)` refs进入record-21 identity；它不编码runtime durable scope key/provider/domain，不能复用package diagnostic
  aliases/resource names、靠两侧排序ordinal zip或隐式相似恢复mapping。

`StateSlotVersionRole` enum及KAD slot field由`kernel_abi.proto`唯一编号，`package_manifest.proto`必须import并复用
该enum，不能复制numeric table。executable/package state-group records引用scoped `StateConsistencyGroupId`并包含
axis-covered group realization、member refs和slot-version relation；streamed resource records使用typed
`ResourceRealizationRecordKey`/`StreamWindowId`。这些identity projection由`semantic_identity.proto`登记，package wire只
交付同一typed事实，不能另造state/streaming side table。

每个participating field必须在schema中声明WCRE value kind以及`ordered`或`set`语义；nested message使用完整
WCRE nested record，enum按显式非负numeric value编码为`u64`。field number一经发布不得复用，删除字段必须
`reserved`，改变value kind/order semantics或identity inclusion必须提升message schema version。不得通过
C++ struct declaration order、ODS attr dictionary order、Protobuf reflection iteration order或字段名自动分配
field number。canonicalizer若遇到schema未登记的identity-relevant IR field、type、attr或message field必须
失败；只有在owner schema中显式标记为non-identity diagnostic/locator的字段才可排除。

实现分层也属于合同：runtime-safe canonical core只接收generated identity/artifact-set/package/KAD messages，按
schema reflection/options生成WCRE。每个public factory直接返回对应record-specific digest wrapper；generic raw
digest最多作为implementation-private `detail::SemanticDigestValue`存在，并必须在同一factory内立即封装，不能进入
public header、cross-stage API或generic caller-selected domain。canonical core不依赖MLIR，也不认识任一IR op。
compiler-only adapters分别从verified `wafer.model.*`、`wafer.distributed.*`、target environment/topology/mesh、
projection、committed executable和static function构造对应generated identity message。每类IR owner必须有一个
typed builder和round-trip-independent verifier；其它pass只能消费builder返回的message/typed digest，不能直接调用raw
record builder、自留field number或对打印IR再hash。Package/Runtime只链接runtime-safe canonical core和artifact
loader，不因identity验证拖入MLIR、conversion或compiler pipeline。

低精度描述符也服从这一分层。compiler IR只用registered MLIR Quant/float8 types及Wafer-owned
block-scale/storage attrs/interfaces表达语义；它不公开另一套同名C++ aggregate。`semantic_identity.proto`是
可交付nested descriptor field/value的唯一schema owner，runtime-safe `WaferABI`提供private-construction、
immutable `abi::QuantizationDescriptor`和`abi::StorageEncodingDescriptor` verified values。
`WaferCompilerIdentity`中的唯一adapter从verified IR types/attrs投影并可反验这些values；KAD、target artifact set、
package都import/reuse同一generated messages和C++ values。Runtime只验证、比较并绑定，不链接WaferIR/MLIR、
不解释公式或重新packing。任何WaferIR header中复制字段的同名descriptor struct都非法。

`TargetArtifactSet` delivery bytes由pinned C++ Protobuf producer确定性输出，并另有exact-byte blob digest；它们不是
set semantic identity。runtime-safe loader必须先校验delivery blob digest，再parse、拒绝unknown/current-version不支持
字段、运行唯一set semantic verifier并重算`TargetArtifactSetId`，最后才暴露immutable verified member map。package
assembly只能消费该verified set，不能读取compiler IR、手写index或用locator/build order重建coverage。

target builder只把`TargetArtifactSetVerifiedRecord`、delivery root/index和module blobs attach到当前outer
`ProgramOutputTransaction`的private staging，不执行独立root rename/publish。最终统一delivery root或
`ProgramDeliveryCommitRecord`验证全部requested targets/package后，才能构造nonsemantic
`TrustedTargetArtifactSetDeliveryRef`：持有root capability、validated relative index locator、expected delivery exact
size/content digest和expected `TargetArtifactSetId`。该ref只能由包含standalone target-root commitment的已验证program
delivery commit record创建，不能由target-set root内部字段、PackageManifest或bundle index自证。

production metadata loader唯一形态是
`loadAndVerifyTargetArtifactSetMetadata(TrustedTargetArtifactSetDeliveryRef,
ArtifactMetadataVerificationSession &) -> LoadedTargetArtifactSetMetadata`；没有raw path/root/digest/size、独立
`ArtifactAdmissionLimits`/`CanonicalEncodingContext`或default/unbounded session overload。该runtime-neutral session由
`WaferArtifact`拥有并按`tasks/15`的typed metadata child ledger与共享`HostVerificationRegistry` physical parent共同限制
reader/worker/verified-bytes/open-FD；loader只验证
delivery/index、locator-free records和lazy source facts，不打开实际ELF module，也不反向依赖Runtime。compiler build/readback不调用该
production loader；它只能从同一`TargetArtifactBuildSession`的owner-backed staging/module backing调用底层record/ELF verifier。
diagnostic inspect若保留，消费独立diagnostic capability并返回不可转换的inspection result。

metadata loader在ref capability下beneath/no-follow open，先比external expected bytes，再运行delivery/record semantic验证。
unreferenced staging/orphan root、整套internally-consistent root replacement和wrong expected set ID都不能返回production
metadata。diagnostic accept-any reader若保留，返回类型不能转成trusted ref/verified set，也不能被package/runtime消费。
返回的metadata及其module/source views move-own immutable backing/root capability和metadata provenance token，不借用metadata
session或parse lease；session栈销毁后仍可pure preflight/bind，但provenance不能授权open。pure runtime preflight先从
metadata与verified environment选择exact domains；随后只能通过`tasks/15`的one-way all-or-none bind，把一个或多个metadata sets、
exact-domain service context和同owner `RuntimeArtifactVerificationSession` move入`RuntimeBoundTargetArtifactSets`。后续actual
module/ELF/KAD/profile验证必须借该bound session；cross-session、已move或stale binding在object open/provider load前失败。
metadata对象只持`UnboundTargetModuleSourceDescriptor`，它没有任何acquire/open/read能力；bind成功后才生成
`BoundTargetModuleSource`。
`BoundTargetModuleSource`没有public `acquire/open/read`；只有`RuntimeArtifactVerificationSession`的private access可签发并
move入同owner的`ArtifactVerificationReadLease`，该lease原子占用active-reader与open-FD预算，返回的
`OpenedTargetModuleLease`持有它直到same-handle backing/FD销毁。worker与simultaneously-verified-byte预算由另一move-only
`ArtifactVerificationWorkLease`在hash/ELF/KAD验证期间持有；read/work lease都不能由caller构造、拆分、复制、跨session
复用或在operation尚未结束时提前归还。compiler staging source使用`TargetArtifactBuildSession`自己的build-budget lease，
不能伪造runtime lease；diagnostic source/lease类型与production不互转。

locator-free record semantic verifier还必须生成owner-backed、non-aggregate `VerifiedTargetArtifactModuleView`。每个view绑定
parent `TargetArtifactSetVerifiedRecord` backing/generation、恰好一个ModuleEvidence以及parent中all-and-only引用该module
content digest的member joins；caller不能传member数组、删entry或把另一个record的module evidence拼入。standalone metadata
loader从上述trusted ref返回verified metadata/module views以及同root capability派生的
`UnboundTargetModuleSourceDescriptor`。package loader不调用standalone metadata loader：它直接对manifest内嵌的locator-free verified record运行同一semantic verifier形成module views，physical
bytes只从package root/index派生的`BoundBlobSource`取得。PackageManifest/bundle不包含原target delivery root或locator，因此
不能为package path构造`TrustedTargetArtifactSetDeliveryRef`。

production KAD、target-set record、ELF note/profile和package-import parser的byte input统一使用runtime-safe WaferABI拥有的
private-construction `abi::ImmutableByteBackingRef`，而不是跨调用保存`ArrayRef<uint8_t>`、`StringRef`或caller buffer pointer。
该ref共享持有immutable owner，绑定checked size、exact content digest和受限`readAt`/`slice`；slice仍返回owner-backed ref，
不能泄露超过当前admission budget的连续raw view。capability/file来源必须从同一opened lease完成stat/read/hash/stat后创建，
in-memory来源必须先受limits约束并move-own backing。parser/result proof保留所需backing lifetime，任何owner/digest/size mismatch
失败。仅unit/debug adapter可从inline bytes创建ref，并必须受`ArtifactAdmissionLimits.maxInlineRecordBytes`限制；该adapter
不能接受unbounded bytes、生成trusted delivery ref或绕过production source factory。

所有runtime-safe KAD、target-artifact-set、ELF note/profile reader必须显式接收validated
`ArtifactAdmissionLimits`。它是caller/deployment提供的nonidentity配置，所有字段都是positive checked integer，
0不表示unbounded；至少限制exact delivery/KAD bytes、Protobuf recursion、ASCII string/locator bytes、set members、
unique blobs、KAD descriptors、slots、completion exports、ELF program/section headers、ABI notes、quant/storage
profile records、单module验证bytes、同时验证的module bytes以及nested/total records。parser在读取length/count后先
checked累加并与limits比较，再reserve、map、open或构造对象；不能先按wire count分配vector，不能先映射整个
untrusted module再做上限检查。有效上限是deployment limits与可信运行环境/provider可报告硬上限的逐项minimum。
失败返回typed `artifact_admission_limit_exceeded`或`size_overflow`和stable owner/limit kind，且不返回partial
verified proof、不publish set、不保留opened module handle。该配置只决定当前deployment是否接纳同一artifact，
不得截断member/note/slot或改变semantic identity。PackageManifest自身另由`PackageParseLimits`约束；两者不能用
彼此的默认值形成绕过。

runtime-safe `WaferABI`也是跨compiler/package/runtime structural ID value type的唯一C++ owner。至少定义并复用：

- tagged `ResourceId = ModelBoundaryResourceId | ModelInternalResourceId | ExecutableResourceId`；每个alternative是
  自己domain的typed digest，oneof discriminant参与package identity，不能退化成untyped 32 bytes。
- non-interchangeable digest wrappers：`TargetEnvironmentFingerprint`、`TopologySnapshotId`、`ExecutionMeshId`、
  `ProjectionSetId`、`ExecutableSemanticDigest`、`StaticFunctionDigest`、`KernelAbiSemanticDigest`、
  `TargetArtifactFingerprint`、`TargetArtifactSetId`、`PackageManifestId`、`ModelInterfaceSemanticId`、
  `DistributedProgramSemanticId`、`QuantStorageAbiProfileId`、`StateMigrationPlanId`、`TargetVariantId`、`ExecutableVariantId`和`EntryId`。它们不能通过generic
  `SemanticDigest`参数互传；只有显式debug byte view可比较raw bytes。
- runtime-safe low-precision values：private-construction、immutable `QuantizationDescriptor`、
  `StorageEncodingDescriptor`和`QuantStorageAbiProfileV1`；只由generated schema verifier或compiler adapter创建。
- owner-backed parser bytes：private-construction、immutable `ImmutableByteBackingRef`；KAD/set/package proof不保存caller raw
  byte view。
- scoped composites：`ModelEntrypointId`、`InvocationPolicyFieldId`、`ExecutionInstanceId`、`RankClassId`、`CompletionNodeId`、`ActivationPredicateId`、`ResourceRealizationRecordKey`、
  `StreamWindowId`；scoped integers：
  `ModelProgramMemberId`、`StateConsistencyGroupId`、`ComponentId`、`DdrArenaId`、`IterationDomainId`、`SlotId`、
  `CompletionExportId`、`TransportActionId`、`TransportBindingMemberId`。

这些type使用private/verified constructors、逐字段canonical comparison/hash和唯一generated-message conversion；
compiler adapter、package verifier和runtime不得各自定义同名raw byte/string/int alias。runtime-only
`InvocationId/EntryInstanceId/ScopeInstanceId/ResourceVersionId/StateGroupVersionId/StateNamespaceId`由
`Wafer/Runtime/RuntimeIdentity.h`的WaferRuntime value types定义且没有package proto conversion。schema message只承载编号owner允许序列化的IDs；不能因为公共C++ type存在就把
runtime-only identity写入manifest。

任何以`Verified`、`Accepted`或`Identity`命名并被下游当作proof token的C++ result type都必须immutable、non-aggregate，
使用private constructor和const accessors，只能由对应parse/verify/build factory创建；至少覆盖
`StaticFunctionIdentity`、`VerifiedKernelAbiDescriptor`、`TargetArtifactFingerprint`、verified ELF/module/member/set
和loaded package/environment/profile types。unit compile checks要求`std::is_aggregate_v`和公共直接构造为false。
若某个wire/message object仍是public aggregate，consumer必须在同一调用中重新运行唯一verifier，不能凭type name
跳过验证。

semantic digest使用FIPS 180-4 SHA-256，preimage固定为：

```text
ASCII("WAFER\0") || u16be(1) || u16be(domain_size) || domain_ascii
                 || u64be(record_size) || WCRE_record
```

V1 domain不可复用：

| identity | domain |
| --- | --- |
| target environment | `wafer.target-environment.v1` |
| topology snapshot | `wafer.topology-snapshot.v1` |
| execution mesh | `wafer.execution-mesh.v1` |
| projection set | `wafer.projection-set.v1` |
| committed executable | `wafer.executable-semantic.v1` |
| committed static function | `wafer.static-function.v1` |
| Kernel ABI descriptor | `wafer.kernel-abi.v1` |
| target artifact fingerprint | `wafer.target-artifact.v1` |
| target artifact set | `wafer.target-artifact-set.v1` |
| package manifest semantic identity | `wafer.package-manifest.v1` |
| model interface | `wafer.model-interface.v1` |
| distributed program | `wafer.distributed-program.v1` |
| model boundary resource ID | `wafer.model-boundary-resource.v1` |
| model internal resource ID | `wafer.model-internal-resource.v1` |
| model dimension ID | `wafer.model-dimension.v1` |
| compiler-created executable resource ID | `wafer.executable-resource.v1` |
| target variant ID | `wafer.target-variant.v1` |
| executable variant ID | `wafer.executable-variant.v1` |
| executable entry ID | `wafer.executable-entry.v1` |
| quant/storage ABI profile | `wafer.quant-storage-abi-profile.v1` |
| state migration plan | `wafer.state-migration-plan.v1` |

digest在typed object/Protobuf中用algorithm enum + 32 raw bytes表示；`sha256:<lowercase-hex>`只是
debug/CLI spelling。Final ELF、weight/payload和exact Protobuf package blob的content digest则为
`SHA-256(raw bytes)`，不加semantic domain；它们与semantic ID是独立字段，不能混用。

model-interface digest从`wafer.model.*`的typed symbols、ports/resources/constraints/alias和payload content
identity构造；payload locator/path不进入identity。distributed-program digest从model-interface digest、
components、canonical instances/classes、typed edges/resource shards和globally coherent variants构造；payload
content/shard identity进入，payload locator不进入。committed executable digest再从这两个source digests、
target requirements、resources、static entries/ranks/classes、transport/projection refs、entry graph和completion
DAG构造。每层只编码其owner fields，不重新扫描上游printed IR或container path。

static-function digest使用WCRE `static_function` record和下列compiler-owned structural encoder。计算前必须先完成
full typed call/global closure resolution并取得non-forgeable `ResolvedClosureSymbolIdentity`；不能对isolated function
用source symbol spelling直接hash：

1. 输入必须是isolated、verified static function，且属于已通过traversal/layout/memory/transport/geometry gates的
   commit-ready candidate clone或committed executable；先拒绝residual group/template/candidate-only body op、
   unregistered dialect、unresolved symbol、dynamic target geometry和schema未登记type/attr，不运行会改变语义的
   canonicalization。candidate中计算的digest只能供同一transaction的projection/commit verifier使用，在atomic
   commit前不得发布到artifact cache、ELF、package或外部索引；commit后对同一function重算必须一致。
2. function record编码function type、visibility、argument/result attrs和body；`sym_name`不进入body identity，
   entry symbol由KAD/member map单独绑定。symbol reference必须经closure proof分类：public executable entry编码typed
   `EntryId`；private function/global编码tasks/14 module-partition算法自底向上得到的structural node digest/canonical ref；
   reviewed external ABI symbol才编码versioned external-symbol identity及stable ASCII ABI spelling。private source path/name
   永不编码。
3. region按op region ordinal编码，block按region storage order分配从0开始的ordinal；block argument和op result
   按结构顺序分配value ordinal。operand、successor和CFG edge只编码这些ordinal，不编码SSA print name或
   pointer identity。
4. 每个op nested record固定编码registered op name、ordered operand value refs、ordered result types、按attr
   name ASCII bytes排序的registered attrs、ordered successors和ordered regions。op name、operand/result/
   successor/region数量都显式编码，不能通过缺省值恢复。
5. builtin integer/float/index/tensor/memref/function type按kind、width/signedness或float semantics、shape、
   element type、layout和memory space递归编码；Wafer type/attr由同一schema registry逐field编码。symbol ref、
   dense elements、integer/float attr使用exact bit pattern；unsupported opaque/resource attr失败。
6. 只排除`Location`、SSA print name以及schema显式列出的diagnostic/pass-trace attr。任何未分类attr默认失败，
   不允许为获得稳定hash静默忽略。private callee/global body变化会改变structural ref并进而改变entry
   `StaticFunctionDigest`；仅rename/reorder private declarations不改变。external ref由artifact fingerprint中的toolchain/
   CRT/allowlist version继续绑定，不能只凭entry digest授权替换module。

encoder version是`static_function` record的mandatory field，也进入`TargetArtifactFingerprint`；MLIR/toolchain
revision同样进入fingerprint，不假设不同compiler revision可以共享cache。打印MLIR文本、MLIR bytecode raw
bytes、Operation pointer/hash或文件名不是semantic digest输入。上述算法允许等价IR因保守字段产生不同digest
而降低cache hit，但禁止不同语义因ignored unknown field得到同一digest。

V1 ELF必选note合同为：

- section name是`.note.wafer.abi`，section type必须是`SHT_NOTE`，flags必须包含`SHF_ALLOC`且不得包含
  `SHF_WRITE`/`SHF_EXECINSTR`，`sh_addralign = 4`。
- 每个module内每个唯一`KernelAbiDescriptor` semantic digest恰好一条note，按32-byte raw value升序；多个entry
  引用相同descriptor时通过member `EntryId -> KAD digest` relation共享该note。encoder先按digest去重，same digest的
  canonical record和delivery bytes必须byte-equal，否则按collision失败；重复输出note仍失败。`namesz = 6`，name bytes是
  `WAFER\0`，`type = 0x57414249`。ELF note header按ELF `EI_DATA`编码并按4-byte padding。
- note descriptor不依赖target endianness：

```text
ASCII("WABI")
|| u16be(note_format_version = 1)
|| u16be(hash_algorithm = 1)          # SHA-256
|| u32be(canonical_encoding_version = 1)
|| u32be(kernel_abi_schema_version)
|| u32be(descriptor_delivery_size)
|| 32-byte KernelAbiDescriptor semantic digest
|| descriptor_delivery_bytes
```

`descriptor_delivery_bytes`是generated Protobuf `KernelAbiDescriptor`的pinned C++ producer输出；
loader parse后用同一C++ semantic verifier重构WCRE record并比对digest，不对raw Protobuf
bytes声称canonical。当前LLVM 20工具链的实现入口必须组合
`--add-section .note.wafer.abi=<file>`、`--set-section-type .note.wafer.abi=7`、
`--set-section-flags .note.wafer.abi=alloc,readonly,contents`和
`--set-section-alignment .note.wafer.abi=4`；单独add-section只产生`SHT_PROGBITS`，不满足合同。
device-code gate必须再解析final ELF验证note type/flags/name/type-code/
alignment/descriptor，不把objcopy成功当作ABI证明。module content digest在note注入和
final ELF verifier之后计算。

`TargetArtifactFingerprint` WCRE record必须覆盖environment digest、target triple、ISA/
extensions、MABI、KAD schema/canonical-encoder version、Wafer CRT/symbol-set digest、该closure/module实际引用的
canonical `usedQuantStorageProfiles` (`QuantStorageAbiProfileId`) set、
toolchain/device-link profile、`ModulePartitionPolicyV1`以及pinned projection digest或relocatable slot schema。

`toolchain/device-link profile`不是revision string。versioned `VerifiedTargetToolchainProfileV1`由reviewed registry构造，
至少绑定compiler/linker/objcopy binary exact content digests、target triple/ISA/MABI、sysroot与CRT/object digests、
canonical ordered compile/link/objcopy arguments、允许影响codegen的environment key/value以及profile version。binary path
只作locator；loader必须在运行前复核同一content digest，未登记environment被清除或拒绝。public emission API只接收
non-forgeable verified profile并由它生成immutable arguments；caller不能传`clangPath`、`fixedArguments`或追加flags。
完整profile identity进入artifact fingerprint，任一binary/arg/env变化必须改变fingerprint或在spawn前失败。
global target build profile只拥有profile registry/schema version；每个instruction/resource/KAD引用一个verified
profile ID，artifact builder从closure实际refs重算duplicate-free used set。plain module显式使用empty set，一个module
可以使用多个profiles。每个closure unit的prelink key保留其actual used-profile set，保证codegen输入可重放；但不同
profile sets本身不构成module partition boundary。只要environment/toolchain/projection/ABI等module-wide compatibility
fields一致，deterministic packer可合并units，并对packed module的所有unit/KAD/resource refs求canonical duplicate-free
union，随后才计算该module的`TargetArtifactFingerprint`。兼容INT8/MXFP/plain units因此不会仅因set不同爆炸成多个
modules；不兼容profile field必须由明确module-wide ABI/capability verifier拒绝，而不是按字符串bucket。unused registry
capability/profile不得被caller塞入unit/module set，也不能因全局存在而强迫所有modules共享一个singular profile。

artifact build service另接收validated nonidentity `TargetArtifactBuildLimits`，至少限制input graph nodes/edges、
closure units、entries、modules、KAD descriptors/notes、single/total prelink bytes、single/total final module bytes、
staging disk、simultaneous compiler/linker/objcopy workers、open FDs和in-memory buffers/bytes，以及per-process/total
wall-clock timeout、CPU time、RSS/address-space、process count、stdout/stderr bytes和cancel/kill/reap deadline。所有字段为positive checked
integer，0不表示unbounded。builder先从完整committed target coverage与固定`ModulePartitionPolicyV1`用checked
arithmetic做纯preflight；通过前不创建artifact-set root、staging output或tool subprocess。limits只决定当前build
service是否接纳，不得改变closure connectivity、canonical unit sort/packing、拆module阈值或选择entry prefix；同一
合法input在不同足够大的build limits下必须产生相同modules/set identity。超限或实际计数偏离preflight回滚全部
transaction，返回typed limit/owner diagnostic且零published root。

所有tool invocations通过`VerifiedTargetToolchainInvocation`进入sandboxed process group，清理未登记environment，设置
OS/provider resource limits并使用bounded stdout/stderr sinks；不能先capture unlimited output再截断。caller cancellation、
timeout、CPU/RSS/output/process limit或子进程异常必须终止整个process group，在kill/reap deadline内回收并删除partial
object/note/module staging；reap失败把outer transaction标为uncommittable并隔离orphan。充分limits不进入artifact
fingerprint、不改变partition/arguments/output bytes；limits只决定当前build service能否完成exact invocation。

`TargetArtifactSetId`只在所有final module content digest可用后计算，其WCRE member list按
`EntryId + static-function semantic digest + final module content digest`排序，并包含source executable digest、
`TargetVariantId`、environment/artifact fingerprints、module digest、entry symbol、KAD digest、
covered `RankClassId` 和completion exports。不存在独立可变的`ModuleId`：final content digest就是package/artifact
边界的module identity。link前的module grouping只使用transaction-local、从sorted typed entry/function relations
构造的`ModuleGroupKey`，不得序列化、进入identity或由path/build order生成。任一member失败都不产生set root。

一个`TargetArtifactSet`的coverage owner是`(source ExecutableSemanticDigest, TargetVariantId)`，不是单个
`ExecutableVariantId`。builder必须收集该executable内所有引用同一TargetVariantId的committed shape variants，
对每个variant的EntryId/RankClassId/projection/completion coverage闭合；caller不能选择其中一个shape variant。
相同EntryId/static function/KAD/projection dependency可以共享member/module，不同pinned projection或其它artifact
fingerprint输入必须形成不同member/closure unit，set root包含每个member自己的fingerprint。semantic verifier按每个
ExecutableVariantId重放coverage，既拒绝漏variant/class，也拒绝把不同fingerprint的code错误合并。

target build profile必须包含typed `ModulePartitionPolicyV1`，至少固定positive
`max_entries_per_module`、`max_prelink_object_bytes`、`max_kad_note_bytes`和policy version；KAD note bytes按unit/module
引用的unique descriptor digest set计算，不按entry重复。policy还固定dependency-classification version、
`clone_proven_pure_private_v1`和每module clone/dedup bytes上限；完整policy进入
`TargetArtifactFingerprint`。artifact builder始终消费一个committed target variant的全部entries，caller不能传
任意entry subset或预分组。deterministic partition算法固定为：

1. 从typed static function refs构造完整call/global dependency graph并对每个node/edge分类；未解析ref立即失败。
   `entry_to_entry`、mutable global、address-taken/address-significant global/function、indirect/unknown external effect和
   shared state/control dependency是non-clonable connectivity。private helper只有在无address escape/indirect call、
   `MemoryEffectOpInterface`及reviewed external registry递归证明pure、只读取immutable non-address-significant globals、
   无recursive SCC时才是`cloneable_pure`；private global只有immutable、initializer acyclic、无地址身份/escape时才可
   clone/dedup。unknown proof一律non-clonable或结构化失败，不能因名字像helper而放宽。V1拒绝recursive private call、
   cyclic initializer和无法分类的indirect call。
2. 对acyclic private dependency graph自底向上计算transaction-local `ClosureStructuralDigest`：node payload编码
   function/global type、linkage、body/initializer structural semantics并排除symbol/location，private refs替换为被引用node
   digest；closure按typed node/edge records排序。digest collision对应non-equivalent node hard fail。prelink前把private
   symbols规范化为digest-derived collision-checked internal labels，entry symbol只由typed EntryId/member relation生成，
   因而source rename和module declaration permutation不影响object bytes。
3. non-clonable connectivity形成不可拆closure units；`cloneable_pure` dependencies记录为按structural digest排序的
   clone set但不把多个entry units连在一起。每个entry恰好属于一个unit，non-clonable dependency必须闭合。unit codegen
   key包含sorted `(EntryId, static-function digest)`、non-clonable `ClosureStructuralDigest`、canonical clone dependency
   digests、unique KAD digests、actual used-profile set和module-wide environment/toolchain/projection/ABI compatibility key；
   不包含尚未形成的final module artifact fingerprint。
4. 每个unit core及每个unique cloneable dependency独立完成transactional conversion/prelink object，使用digest-derived
   collision-checked private labels，并测量exact `.text + .rodata + required target data` bytes。同一digest+record只生成
   一个transaction-local clone object；collision失败。单个non-clonable core或单个clone object超过policy limit失败，
   不能靠估算/path拆；共享pure helper不会仅因被千个entries引用形成巨型unit。
   为允许core与clone dependency分开`.o`，clone symbols只在prelink阶段使用versioned
   `hidden_linkonce_odr_comdat_v1`：digest-derived ASCII symbol和COMDAT signature都来自完整structural digest，selection
   kind固定为`any`，但builder在link前要求同signature的canonical record和object section content byte-equal；same digest/
   unequal definition hard fail，不能让linker任取。core refs使用同一hidden symbol。每个final module只提供一个definition；
   final link/normalization后这些symbols必须hidden或local且不在dynamic/public exports，只有typed entry symbols可导出。
   跨final-moduleprivate undefined ref非法。linkage/COMDAT/localization policy version及exact tool args进入
   ModulePartitionPolicy/toolchain identity和artifact fingerprint。
5. 以`(sorted entry keys, core ContentDigest, canonical clone-object digests, unique KAD set, actual profile set,
   module compatibility key)`形成`UnitPackingKey`并lexicographic排序，随后deterministic sequential packing。只有
   module compatibility key相同的units可合并；加入下一个unit时，entry/core+deduplicated-clone bytes、unique KAD note
   bytes及其它policy limits都用checked arithmetic计算，超限结束当前module。同一packed module中结构digest相同的
   pure helper/immutable global object只链接一次。相同输入/profile/toolchain得到相同member lists，enumeration/
   worker order不影响。
6. pack完成后从全部unit实际profile/KAD/resource refs求canonical unions，计算该module唯一
   `TargetArtifactFingerprint`，再final link unit cores、deduplicated clone objects、Wafer CRT和显式external registry；
   任何跨module non-clonable private undefined ref失败。transaction-local `ModuleGroupKey`是sorted UnitPackingKeys和
   final fingerprint的typed tuple，只用于关联link结果；发布后module identity只用final content digest。

改变purity/classification/clone/dedup或其它partition algorithm必须提升`ModulePartitionPolicy` version并进入artifact
fingerprint；不能在implementation启发式改变V1边界。set verifier证明entry exactly-once coverage、non-clonable closure、
clone proof/dedup、profile union、limits和canonical packing。
tests必须覆盖相同EntryId/static wrapper ref但private helper body或global initializer不同的两个closure，要求prelink/
member分离；仅rename private symbols或permutation declarations必须得到相同closure digest、packing和final bytes。
recursive/cyclic dependency、digest collision和canonical label collision均在link前失败。
另有两个EntryId引用byte-equal同一KAD digest的positive case：module只含一条note，entry/member mapping仍完整；same
digest但descriptor record/delivery bytes不同必须collision失败。
规模gate还包含数千entry共享同一proved-pure helper/immutable table：它们可按limits拆module且每module只链接一份clone；
把helper加入effect、address-take或mutable global edge后必须形成non-clonable connectivity，oversize时fail-closed而不是
错误clone。兼容INT8/MXFP units可进入同module并得到profile union；改变worker/limits之外的非semantic执行参数不改变
充分预算下的canonical result。

### 4.3 Shared Geometry Legality

所有会进入 LLVM `i32`、CRT `uint32_t`、TX81 `uint16_t` shape field、address offset或 byte descriptor 的
geometry 使用一个共享 legality contract。instruction verifier、target lowering preflight、
`KernelAbiDescriptor` verifier和 golden tests必须复用同一字段范围与单位定义，至少检查：

- static/dynamic dimension和 variant bound满足 target field width，乘积与 byte-size计算不溢出。
- byte count、inner bytes、element bytes、stride/iteration和 descriptor coverage一致且需要时整除。
- SPM/DDR address、offset、alignment和 end address在 accepted allocation内。
- rank/stage launch geometry、transport buffer capacity和 kernel slot shape/capacity relation一致。
- affine quant scale/zero-point count、axis/group relation、q0/q1与zp bit range、accumulator bound和explicit
  saturation/requantization一致；packed data/scale slot capacity与`StorageEncodingDescriptor` exact bytes一致。
- block-scaled decode的element/block count、block/tail divisibility、packed source/scale/scratch/destination span、
  FP encoding和NaN/Inf/subnormal/overflow policy与matched `QuantStorageAbiProfileV1`一致。

超范围 geometry 只能在 target legality或 variant guard处被拒绝，不能通过 LLVM integer constant
截断、C cast、default format或 runtime best-effort继续执行。

target-independent structural `InstructionGeometry`属于`WaferIR`；它不链接compiler identity或conversion。
target-specific limits、allocation/root verification和narrowing属于独立`WaferTargetLegality`，该库只向下依赖
`WaferIR/WaferCompilerIdentity/WaferABI`，不得依赖Whole或conversion。allocation/root事实通过non-aggregate
`VerifiedPhysicalAllocationView`进入同一target-legality core。只有两个compiler-private
adapter：pre-commit adapter从同一whole-variant transaction的current generation、fresh internal
`ExecutableResourceView`、accepted memory/transport/projection和exact`VerifiedTargetCompilationContext`构造；post-commit
adapter从sealed executable的`VerifiedTargetConversionRequest`构造。两者都绑定op/root/view/operand ordinal、access、
half-open range、allocation capacity/alignment和owner generation，不能由raw ranges/callback/public aggregate创建。

pre-commit `verifyCandidateTargetPreflight`只检查完整candidate的supported structure、external symbol availability、全部
instruction geometry/range/narrowing和entry boundary feasibility，返回transformation-local proof。它不运行LLVM conversion、
不生成KAD/prelink bytes/staged object且不能传给target artifact builder；whole commit内部必须fresh重跑而不接caller proof。
commit后artifact builder从committed adapter再次重放相同core并完成full conversion/KAD，任何差异或新失败使整个target set
失败，不能引用candidate result。limits/worker变化不改变合法结果，任一rewrite使view/proof失效。

## 5. Address and Resource Derivation

Target lowering 必须在进入 LLVM dialect 前把 Wafer memory / endpoint / resource 事实消解成标量参数：

```text
SPM byte address / offset:
  accepted wafer.spm.offset on root allocation
  + static view byte offset derived from memref view and Wafer physical layout
  + instruction-local byte offset

external DDR byte address:
  runtime launch binding base
  + static memref subview byte offset
  + descriptor-local byte offset

compiler-managed DDR byte address:
  runtime workspace/resident base
  + accepted wafer.ddr.offset
  + static view byte offset
  + descriptor-local byte offset

DTE peer / endpoint:
  logical peer rank in wafer.instr.dte_* op
  -> tasks/13 accepted physical transport
  -> pinned physical endpoint/channel/FSM/receiver buffer
     or relocatable endpoint/control/status ABI slots
```

external DDR function argument 不允许被解释成编译期绝对地址；它只代表 runtime launch binding base。
`#wafer.ddr_offset` 只适用于 compiler-managed / resident / workspace DDR allocation。

### 5.1 Environment, Artifact Fingerprints, and Module Identity

两个fingerprint不能混用：

- `TargetEnvironmentFingerprint`由`tasks/04`的target capability/ABI/arena declarations产生，用于
  `TargetVariantId` compatibility；它不包含deployment topology、availability或projection。
- `TargetArtifactFingerprint`是module/artifact cache identity，覆盖`TargetEnvironmentFingerprint`、target
  triple、ISA/extensions、MABI、kernel ABI schema、Wafer CRT/symbol-set、canonical
  `usedQuantStorageProfiles` ID set、toolchain/
  device-link profile和codegen dependencies。pinned code把`ProjectionSetId`/projection digest及必要
  topology assumption纳入artifact fingerprint；relocatable code只纳入relocation slot ABI/schema，不把运行时
  选择的member伪装成target environment。

Runtime先按environment fingerprint过滤target axis，再独立选择committed projection；artifact fingerprint
只验证所选module与该组合兼容，不参与shape guard或rank-class选择。

Final link 之后计算 module digest，并验证 ELF machine、ISA/MABI、exports、undefined-symbol policy和 ABI
mandatory descriptor note。production verifier不接public aggregate expected fields。compiler侧唯一factory从
`ModuleGroupKey`、完整`PreparedTargetEntry`/non-clonable core/clone dependency proofs、exact
`VerifiedTargetCompilationContext`和同一`TargetArtifactBuildSession`构造non-aggregate
`VerifiedTargetElfContract`；loader侧唯一factory
`buildLoadedTargetElfContract(const VerifiedTargetArtifactModuleView &)`只消费上述exact owner-backed module view，不接
caller member array、delivery ref或compiler/runtime context参数。两者都绑定exact entry set、KAD/profile union、
environment/artifact fingerprints和owner generation，不能由caller逐字段拼装。

compiler verifier只从`StagedObjectRef`取得expected bytes；standalone loader从`BoundTargetModuleSource`、package loader从
`BoundBlobSource`取得bytes，并调用同一runtime-safe ELF verification core；
不存在接受自由`ExpectedElfContent`、raw path或digest/size aggregate的production overload。验证在同一opened handle上完成
stat/read/full digest/stat、ELF/note/member join，成功才返回owner-backed verified module proof。实际runtime environment/
provider compatibility由RuntimeSession在target/projection选择后把该proof与`VerifiedRuntimeEnvironmentSnapshot`重新join；
set loader不能提前替代部署期检查。environment/artifact fingerprints、descriptor digest和module digest都是object/package
stage的typed输出；module path只用于受限内部定位，不能承担identity或compatibility语义。

## 6. Golden Packet Boundary

Golden packet / wrapper tests 验证的是 target CRT implementation 对 public TSM wrapper/register 的映射，
不是上层 compiler IR 的第二语义源。

允许的验证输入：

- target CRT symbol 的 unit test / capture output。
- `wafer.instr.*` lowering 生成的 LLVM call 参数 dump。
- public TSM wrapper/register packet field 对照。

不允许的验证输入：

- capture shim。
- compiler-facing helper ABI。
- 手写 C stub issue table 作为 production lowering 证明。

## 7. Wafer CRT 全量实现设计

本节定义 `wafer_tx81_*` 的 repo-local Wafer CRT 实现方案。这里的“全量”只指
`LowerInstrToTargetLLVM.cpp` 当前可能 emit 的 Wafer-owned target CRT symbol 全量，不指硬件所有
opcode、所有 TSM wrapper 或未来未进入 `wafer.instr.*` coverage matrix 的能力。

### 7.1 Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  target LLVM call-emission 生成的 LLVM IR artifact和 `KernelAbiDescriptor`，调用 `wafer_tx81_*`
  Wafer-owned target CRT symbols；
  typed executable resources/entry bindings、accepted offsets和committed projection已在LLVM call参数中
  materialize；CRT本层不恢复resource semantics。
- Current stage responsibility:
  在 repo-local Wafer CRT 中定义 Q1 确认的 production `wafer_tx81_*` symbols；每个 symbol 直接
  创建 public `Tsm*Instr` 所需对象，调用 public TSM wrapper，再执行 `TsmExecute` 或等价 public
  helper。ABI-incomplete Direct DTE symbols 不属于当前 Q2-Q3 production closure。该层不调用
  TX81/Triton `__*` ABI，不恢复 capture shim，不写空实现。CRT implementation、wrapper/golden
  coverage 和 device-code required-symbol gate 必须作为同一个 closure 验证，不能拆成可单独报 done
  的最小单元。
- Output artifact / IR:
  Wafer CRT RISC-V object / archive、带mandatory `.note.wafer.abi`的 kcore shared object、
  required-symbol/ELF ABI report、environment/artifact fingerprints、descriptor semantic digest和module digest。module path只是
  artifact locator，不是 package identity或 ABI source。
- Downstream consumer:
  device-code compile/link symbol-closure gate、IR-derived `PackageManifest`、runtime adapter / board gate。
- User-level driver / named pipeline:
  production由`stablehlo-to-executable` direct driver的target-artifact stage执行；
  `wafer-lower-groups-to-target-llvm`、`--wafer-lower-instr-to-target-llvm`和
  `tools/wafer_device_link.py`是其内部构件/分阶段debug入口。
- Explicit non-goals:
  不把 `__Gemm`、`__AddVV` 等 TX81/Triton CRT symbol 改成 Wafer compiler ABI；不通过
  `--allow-shlib-undefined` 放过 Wafer-owned symbol；不新增 helper ABI dialect；不在 CRT 内重新做
  layout/search/SPM/DDR planning；不从 linked symbol、LLVM文本或 module path反推 kernel argument role。
- Completion gate:
  Q1 production closure 中的所有 `wafer_tx81_*` symbols 都有 repo-local definition、typed signature
  和 wrapper/golden 覆盖，或被 verifier / lowering 明确拒绝并移出 production closure；positive
  device-code gate 不包含 ABI-incomplete Direct DTE calls；final kcore `.so` 中不得残留 undefined
  production `wafer_tx81_*`；ELF descriptor semantic digest与compiler descriptor一致，environment/artifact fingerprints和module
  digest可由 package直接消费；lit/ctest 覆盖成功闭合、缺失 symbol和 descriptor mismatch失败路径。
```

### 7.2 文件和所有权

Wafer CRT 是 compiler target boundary，不放进 opaque third-party archive 里作为二进制补丁。
源代码放在 repo-local runtime 目录，由 device link gate 显式编译：

| 文件 | 职责 |
| --- | --- |
| `runtime/wafer_crt/include/wafer_tx81_crt.h` | 唯一的 `wafer_tx81_*` C ABI prototype 定义；compiler lowering 和 CRT implementation 共同引用 |
| `runtime/wafer_crt/src/wafer_tx81_crt.c` | 定义 Q1 production closure 中的 `wafer_tx81_*` symbols；直接调用 `instr_adapter_plat.h` / `instr_adapter.h` public wrapper |
| `lib/Wafer/Transforms/Target/LowerInstrToTargetLLVM.cpp` | 使用同一 symbol/signature registry 生成 fixed LLVM function declaration；不再长期依赖 `void (...)` vararg |
| `tools/wafer_device_link.py` | 编译 Wafer CRT source/object，链接 target LLVM object、Wafer CRT object、repo-vendored TX8 deps，并做 required-symbol 检查 |
| `test/Tools/` | 覆盖 `.ll -> .o -> kcore .so`、Wafer CRT object link、undefined `wafer_tx81_*` failure |

旧 `libvr.a` archive 已从当前 dependency tree 删除；device link 不链接 `-lvr`，Wafer-owned symbol
不从旧 TX81/Triton CRT 的 `__*` 名字继承 ABI，也不通过 alias/wrapper 调用 `__*` 完成主线发射。
需要用到的底层能力应通过 public TSM wrapper 或 Direct DTE helper 调用。

### 7.3 ABI 规则

`wafer_tx81_*` 使用固定 C prototype；target LLVM pass 必须生成 fixed function type，不允许回退到
`void (...)` vararg declaration。ABI 规则：

- 所有地址参数用 `uint64_t` byte address。
- 所有 element count、byte count、format、kind、dim、shape、stride、iteration 用 `uint32_t`。
- `format` 参数是 `Data_Format` 的整数编码。
- `stride` 参数是 byte stride；`iteration` 是 logical loop count，不是硬件字段中的 `iteration - 1`。
- rank-4 shape 在 ABI 中统一按 `n, h, w, c` 展开；weight shape、pad、unpad、stride、dilation 也按固定
  field 顺序展开，不传裸 pointer 到 compiler-owned temporary array。
- ordinary compute/move symbol 只 issue，不默认 wait；显式 drain 只在 `wafer_tx81_local_fence`、
  DTE wait 或后续 runtime-visible completion boundary 中发生。例外是 public wrapper 只提供
  writeback register 结果、没有普通 dest buffer 的 `peripheral_argmax/argmin`：CRT 必须等待本地完成后
  把 writeback value/index 写入 ABI dest。argmax/argmin value/index ABI destinations are SPM offsets；
  CRT maps them with `get_spm_memory_mapping` before writing，不能把它们当 host/mapped pointer 传入。
- 普通compute/move function返回`void`。Direct DTE是异步provider边界，按7.6固定返回`int32_t`
  immediate call status并写typed status record；不得把该例外扩散成其它instruction family的隐式错误通道。
  静态错误仍属于verifier / target lowering / device link gate；CRT内部不通过silent return表示unsupported。

### 7.4 Q1 symbol surface audit and coverage closure

Q1 审计结果以 `LowerInstrToTargetLLVM.cpp` 的 symbol 构造规则、`WaferAttrs.td` 的 enum spelling 和
instruction verifier 的合法输入为事实源：

- 固定 symbol 由 `makeTargetSymbol("<base>")` 构造。
- per-kind symbol 由 `makeTargetSymbol("<base>", kind)` 构造，suffix 使用 ODS enum spelling。
- `wafer.instr.tdma_data_move` 只有 `pad` / `img2col` 进入 target LLVM；mirror、transpose、
  rotate、NCHW/NHWC 和 tensor_nom 在 target LLVM lowering 中结构化失败。
- `wafer.instr.peripheral <count>` 在 verifier 中被拒绝，因为 count writeback 还没有 instruction IR
  表示；因此 `wafer_tx81_peripheral_count` 不是当前 production closure。
- `wafer.instr.dte_send` / `dte_recv` / `dte_wait` 当前 call-emission 能生成
  `wafer_tx81_dte_*`，但只传 logical peer / bytes / token count，缺 endpoint、channel、
  remote FSM 和 receive-buffer binding。Q1 将 Direct DTE 从当前 Q2-Q3 production CRT closure 排除；
  后续必须先扩 IR / ABI / lowering，再把 DTE 加回 production closure。不能用空 CRT stub 或
  success return 让 Q2-Q3 required-symbol gate 通过。

以下 symbol 是当前 Q2-Q3 要实现并链接闭合的 production CRT closure。实现完成前，任何列在这里的
symbol 都不能在 final `.so` 中保持 undefined。

全面指令覆盖 gate：

- coverage 以 `tasks/11-instruction-ir.md` 的 production coverage matrix 和本节 production symbol
  closure 为全集，不能只覆盖当前 fixture 或某个代表性 workload 调到的子集。
- 每个 production symbol 都必须在 signature registry、`wafer_tx81_crt.h`、LLVM lowering declaration、
  repo-local CRT implementation 和 required-symbol scan 中可追踪。
- wrapper/golden test 可以按 family 复用 fixture，但必须枚举该 family 的全部 production kind /
  signature variant，并校验参数单位、shape/stride、format、wait/completion 责任。
- 如果 public wrapper / ABI evidence 不能支撑某个 symbol，不能用空 CRT 实现补齐；必须先收紧
  verifier / lowering，将该 symbol 从 production closure 移除或标为结构化 unsupported。

基础 memory / TDMA / sync：

```text
wafer_tx81_rdma
wafer_tx81_wdma
wafer_tx81_gather_scatter
wafer_tx81_memset
wafer_tx81_bit2fp
wafer_tx81_mask_move
wafer_tx81_gemm
wafer_tx81_tdma_pad
wafer_tx81_tdma_img2col
wafer_tx81_local_fence
```

Elementwise symbols：

```text
wafer_tx81_elementwise_abs
wafer_tx81_elementwise_recip
wafer_tx81_elementwise_square
wafer_tx81_elementwise_sqrt
wafer_tx81_elementwise_rsqrt
wafer_tx81_elementwise_neg
wafer_tx81_elementwise_max
wafer_tx81_elementwise_min
wafer_tx81_elementwise_add
wafer_tx81_elementwise_sub
wafer_tx81_elementwise_mul
wafer_tx81_elementwise_div
wafer_tx81_elementwise_eq
wafer_tx81_elementwise_ne
wafer_tx81_elementwise_ge
wafer_tx81_elementwise_gt
wafer_tx81_elementwise_le
wafer_tx81_elementwise_lt
wafer_tx81_elementwise_logic_not
wafer_tx81_elementwise_logic_and
wafer_tx81_elementwise_logic_or
wafer_tx81_elementwise_logic_xor
wafer_tx81_elementwise_log2
wafer_tx81_elementwise_ln
wafer_tx81_elementwise_pow2
wafer_tx81_elementwise_exp
wafer_tx81_elementwise_exp_lp
wafer_tx81_elementwise_sin
wafer_tx81_elementwise_cos
wafer_tx81_elementwise_tanh
wafer_tx81_elementwise_sigmoid
wafer_tx81_elementwise_relu
wafer_tx81_elementwise_satrelu
wafer_tx81_elementwise_leakyrelu
wafer_tx81_elementwise_softplus
```

Reduce symbols：

```text
wafer_tx81_reduce_sum
wafer_tx81_reduce_max
wafer_tx81_reduce_min
wafer_tx81_reduce_avg
```

Convert symbols：

```text
wafer_tx81_convert_int8_fp16
wafer_tx81_convert_int8_bf16
wafer_tx81_convert_int8_fp32
wafer_tx81_convert_int8_tf32
wafer_tx81_convert_int16_fp16
wafer_tx81_convert_int16_bf16
wafer_tx81_convert_int16_fp32
wafer_tx81_convert_int16_tf32
wafer_tx81_convert_int32_fp16
wafer_tx81_convert_int32_bf16
wafer_tx81_convert_int32_fp32
wafer_tx81_convert_int32_tf32
wafer_tx81_convert_bf16_int8
wafer_tx81_convert_bf16_int16
wafer_tx81_convert_bf16_int32
wafer_tx81_convert_bf16_fp16
wafer_tx81_convert_bf16_fp32
wafer_tx81_convert_bf16_tf32
wafer_tx81_convert_fp16_int8
wafer_tx81_convert_fp16_int16
wafer_tx81_convert_fp16_int32
wafer_tx81_convert_fp16_bf16
wafer_tx81_convert_fp16_fp32
wafer_tx81_convert_fp16_tf32
wafer_tx81_convert_fp32_int8
wafer_tx81_convert_fp32_int16
wafer_tx81_convert_fp32_int32
wafer_tx81_convert_fp32_fp16
wafer_tx81_convert_fp32_bf16
wafer_tx81_convert_fp32_tf32
wafer_tx81_convert_tf32_int8
wafer_tx81_convert_tf32_int16
wafer_tx81_convert_tf32_int32
wafer_tx81_convert_tf32_fp16
wafer_tx81_convert_tf32_bf16
wafer_tx81_convert_tf32_fp32
```

Conv / Pool / UnPool / Peripheral symbols：

```text
wafer_tx81_conv
wafer_tx81_depthwise_conv
wafer_tx81_backward_conv
wafer_tx81_pool_avg
wafer_tx81_pool_sum
wafer_tx81_pool_max
wafer_tx81_pool_indexedmax
wafer_tx81_pool_min
wafer_tx81_pool_indexedmin
wafer_tx81_unpool_unpool
wafer_tx81_unpool_avg
wafer_tx81_unpool_mask
wafer_tx81_peripheral_argmax
wafer_tx81_peripheral_argmin
wafer_tx81_peripheral_factorize
wafer_tx81_peripheral_bilinear
wafer_tx81_peripheral_lut16
wafer_tx81_peripheral_lut32
wafer_tx81_peripheral_rand_gen
wafer_tx81_peripheral_elem_mask
```

### 7.5 Wrapper mapping

每个 Wafer CRT symbol 直接使用 public wrapper。`__*` 只允许作为文档证据或 golden comparison，不允许
在 implementation 中作为主线调用目标。

| Symbol family | Wafer CRT implementation rule |
| --- | --- |
| `rdma` / `wdma` | 创建 `TsmRdmaInstr` / `TsmWdmaInstr`；调用 `TsmRdma/TsmWdma::AddSrcDst` 和 `ConfigStrideIteration`；`inner_bytes` 转成 wrapper `elem_count` 前必须使用 `format` 对应 element bytes 校验整除 |
| `gather_scatter` | 创建 `TsmDataMoveInstr` 和两个 `St_StrideIteration[3]`；调用 `TsmDataMove::GatherScatter`；`inner_bytes`、stride 都保持 byte unit |
| `memset` | 创建 peripheral packet；调用 `TsmPeripheral::Memset`；shape/stride 在 compiler 已经规整成 contiguous element count |
| arith unary/binary | 创建 `TsmArithInstr`；`abs/recip/square/sqrt/rsqrt/neg` 调 unary wrapper；`max/min/add/sub/mul/div` 调 `*VV` wrapper；当前 compiler 不生成 scalar-immediate VS/VuV ABI |
| relation | 创建 `TsmRelationInstr`；`eq/ne/ge/gt/le/lt` 调 value/bool relation wrapper；wrapper `format` 来自 input dtype，bool storage 由 destination dtype 决定 |
| logic | 创建 `TsmLogicInstr`；`logic_not/and/or/xor` 调 value 或 bool wrapper；若当前 IR 无法区分 bitpacked bool 与 value bool，必须先固定 dtype/format rule |
| transcendental / activation | 创建 `TsmArithInstr` 或 `TsmActivationInstr`；`log2/ln/pow2/exp/exp_lp/sin/cos` 调 `TsmTranscendental`，`tanh/sigmoid/relu/satrelu/leakyrelu/softplus` 调 `TsmActivation` |
| `reduce_*` | 创建 `TsmReduceInstr` 和 `Data_Shape{n,h,w,c}`；按 kind 调 `ReduceSum/ReduceMax/ReduceMin/ReduceAvg`；`dim` 使用 native `0:C, 1:W, 2:H, 3:N, 4:HW, 5:HWC` |
| `convert_*` | 创建 `TsmConvertInstr`；按 dtype pair 调具体 `TsmConvert` wrapper；zero-point group 必须消费 `zero_point`，rounding group 必须消费 `rounding_mode`，plain group 禁止使用额外参数 |
| `bit2fp` | 使用 public bit-to-float / mask conversion wrapper；source bool storage 和 destination FP format 必须由 ABI 参数固定 |
| `mask_move` | 使用 `TsmMaskDataMove::MaskMove`；Wafer ABI 传 SPM mask address，CRT 将其按当前 TX8 SPM address width 适配到 public wrapper 的 `uint32_t mask` 参数 |
| `gemm` | 创建 `TsmNeInstr`；调用 `TsmGemm::AddInput`、`ConfigMKN`、`ConfigBatch`、`SetTransflag`、`AddOutput`；bias/scale/activation/quant/psum disabled，除非 IR 后续显式扩展 |
| `conv` / `depthwise_conv` / `backward_conv` | 创建 `TsmNeInstr`；调用 `TsmConv` 或 `TsmDepthwiseConv` wrapper；只启用基础 NHWC/HWOI shape、pads/unpads/strides/dilations；bias/scale/sparse/quant/fused activation disabled |
| `pool_*` | 创建 `TsmPoolInstr`；按 kind 调 `AvgPool/SumPool/MaxPool/MinPool/IndexdMaxPool/IndexdMinPool`；indexed variant 必须有 value dest 和 index dest |
| `unpool_*` | 创建 `TsmUnPoolInstr`；按 kind 调 `Unpool/UnpoolAvg/UnpoolIdx`；scalar `index` 由 ABI 显式传入，不能用名字或 side table 恢复 |
| `tdma_pad` / `tdma_img2col` | 创建 `TsmDataMoveInstr`；调用 `Pad` / `Img2col`；source/dest shape、pad、kernel stride 由 fixed ABI field 提供 |
| `peripheral_*` | 创建 peripheral packet；按 kind 调 public wrapper；count/arg/factorize/bilinear/LUT/rand/elem_mask 的输入输出 arity、format 和 kind-specific attrs 由 fixed ABI 和 verifier 共同保证；argmax/argmin 需要等待 public writeback 后，把 `value_dst` / `index_dst` 作为 SPM offset 经 `get_spm_memory_mapping` 映射后写入 |
| `local_fence` | 只调用 `TsmWaitfinish` 或等价 local drain helper；不携带 multi-tile barrier 语义 |

当前 call-emission 仍可能生成但不属于 Q2-Q3 production closure 的 ABI-incomplete symbols：

```text
wafer_tx81_dte_send
wafer_tx81_dte_recv
wafer_tx81_dte_wait
```

### 7.5.1 Low-Precision CRT ABI

plain `wafer_tx81_gemm`保持quant disabled。typed low-precision extension使用两个独立Wafer-owned symbols，不能
增加varargs、复用convert ABI或直接导出legacy `__FP8*`：

```c
int32_t wafer_tx81_quantized_gemm(uint64_t command_addr);
int32_t wafer_tx81_mxfp_decode(uint64_t command_addr);
```

`WaferTx81QuantizedGemmCommandV1`固定160 bytes、8-byte alignment；所有reserved/flags在V1必须为0：

| offset | field | type / rule |
| --- | --- | --- |
| 0/2/4 | `version/size/flags` | `uint16_t/uint16_t/uint32_t`；必须`1/160/0` |
| 8/16/24 | `lhs_addr/rhs_addr/dst_addr` | `uint64_t`；SPM addresses，8-byte aligned ABI fields |
| 32/40 | `scale_positive_addr/scale_negative_addr` | `uint64_t`；disabled mode必须0 |
| 48/52/56 | `m/k/n` | `uint32_t`；shared geometry证明target ranges |
| 60/64 | `left_batch/right_batch` | `uint32_t`；positive target ranges |
| 68/72 | `input_format/output_format` | `uint32_t`；matched capability的closed target formats |
| 76 | `transpose_bits` | `uint32_t`；bit0=left、bit1=right，其它0 |
| 80/84 | `q0_second_shift/q1_first_shift` | `uint32_t`；各`0..31`；raw fields分别写`quant_q0`和`quant_q1` |
| 88/92 | `zero_point_left_raw/zero_point_right_raw` | `uint32_t`；raw TX81 field `0..255`；compiler按matched capability从mathematical zp checked编码 |
| 96 | `scale_mode` | `uint32_t`；`0 none, 1 positive, 2 negative, 3 both` |
| 100/104 | `rounding_mode/saturation_mode` | `uint32_t`；必须等于matched profile固定implicit policy常量，不是runtime选择 |
| 108 | `reserved0` | `uint32_t = 0` |
| 112/120/128 | `lhs_span/rhs_span/dst_span` | `uint64_t`；accepted SPM range bytes |
| 136/144 | `scale_positive_span/scale_negative_span` | `uint64_t`；与enabled scale table exact coverage一致 |
| 152/156 | `profile_version/reserved1` | `uint32_t`；`QuantStorageAbiProfileV1` version / zero |

CRT创建`TsmNeInstr`，按command调用`TsmGemm::AddInput/ConfigMKN/ConfigBatch/SetTransflag/AddOutput`，只在
scale mode启用时调用对应scale wrapper，并用`SetQuant(q0,q1,zp_left_raw,zp_right_raw)`。首个signed-i8 profile只
允许mathematical zp `[0,127]`到同值raw field；negative/128..255/two's-complement interpretation没有独立capability
与golden/board证据时必须拒绝，不能C cast。V1 reference/golden固定数学顺序：先用left/right zero point形成
9-bit operands和i32 dot accumulator，再以`q1_first_shift`右移/clip到i16，乘matched scale后以
`q0_second_shift`右移/clip到int9，最后按fixed output zero point 0得到INT8 result。i32是internal accumulator；
native FP16 output在独立capability/golden存在前非法，需要显式INT8->FP16 dequant/convert。首个planned native profile
固定`scale_mode=none`、scale addresses/spans为0；只有disassembly/packet+board numeric证明positive/negative scale的
exact formula、axis indexing和table dtype后，新profile才能启用这些fields。否则标准affine scale走显式composite。
rounding/saturation fields只校验profile常量；改变它们但生成同一packet必须被CRT/verifier拒绝。V1明确禁止bias、activation、
sparse和implicit psum。shared geometry/KAD verifier必须先证明descriptor语义可由这些exact fields表达；command ABI
不能反过来缩窄上层generic affine descriptor。

`WaferTx81MxfpDecodeCommandV1`固定128 bytes、8-byte alignment，表达packed FP8 + block scale到BF16/FP16的
explicit composite：

| offset | field | type / rule |
| --- | --- | --- |
| 0/2/4 | `version/size/flags` | `uint16_t/uint16_t/uint32_t`；必须`1/128/0` |
| 8/16/24/32 | `packed_addr/scale_addr/scratch_addr/dst_addr` | `uint64_t`；accepted SPM ranges |
| 40/48/56/64 | corresponding `*_span` | `uint64_t`；exact packed/scale/scratch/destination bytes |
| 72/76/80 | `element_count/block_count/block_size` | `uint32_t`；V1 block size必须32，tail按policy |
| 84/88/92 | `source_encoding/scale_encoding/dst_format` | `uint32_t` closed enums；TX81 V1 dst仅BF16/FP16 profile |
| 96/100/104/108 | `tail/nan/inf/subnormal_policy` | `uint32_t` closed enums，逐项匹配source type/profile |
| 112/116 | `overflow_policy/rounding_mode` | `uint32_t` closed enums |
| 120/124 | `profile_version/reserved0` | `uint32_t`；matched profile version / zero |

CRT实现只能使用repo-audited software SPM load/store和public arithmetic wrapper组成decode/scale，不把旧helper ABI作为
callee合同。return `0`表示command已合法issue，`-1`表示version/size/reserved错误，`-2`表示field/range错误，`-3`
表示profile/encoding不支持；nonzero必须进入entry status/error export并禁止consumer issue。return 0不自动证明
CGRA/local completion：compiler必须让decode completion/local drain支配GEMM consumer和scratch reuse。

两个command struct都必须有C `_Static_assert`和C++ `static_assert`覆盖sizeof/alignment/每个offset，target lowering、
CRT conformance、golden packet和device link共用同一header。只有对应instruction实际出现时symbols才进入required set；
profile/struct ABI version、CRT object digest和symbol set共同进入`TargetArtifactFingerprint`。

### 7.6 DTE ABI

Direct DTE不能通过`TsmExecute`发射。repo-vendored
`interface/op_fw_sim_if/peripheral/include/direct_dte_and_fsm.h`表明`DirectDTESendInfo`至少需要
`src_addr`、`dst_addr`、`length`、`remote_fsm_id`、`mode`、`dst_tile`、`tile_this`、三个
stride/iteration pair和provider-owned`dte_node`；`include/components/oplib_tx81/riscv/riscv/include/dte/kuiper_dte.h`
与`fsm/kuiper_streamfsm.h`进一步给出exact DTE block、FSM/stream和nonblocking status接口。因此logical
`buffer, peer, bytes`既不是production ABI，也不能通过CRT内猜测补全。

V1使用一个versioned typed command record，而不是巨型vararg、vendor struct pointer或opaque sidecar。它由
committed accepted transport/projection、entry `TargetEntrySlot`和`RelocationSchema`机械materialize；conversion后
KAD builder记录并验证同一control/status slots，而不是反向成为conversion输入。field order、offset、size和alignment是
`runtime/wafer_crt/include/wafer_tx81_crt.h`的长期C ABI：

| offset | field | C type | V1 contract |
| ---: | --- | --- | --- |
| 0 | `abi_version` | `uint32_t` | 固定为1 |
| 4 | `record_bytes` | `uint32_t` | 固定为160 |
| 8 | `action_id` | `uint64_t` | entry-local `TransportActionId`，不得由名字生成 |
| 16/24 | `src_addr` / `dst_addr` | `uint64_t` | byte address；recv的`src_addr`为0 |
| 32/40 | `context_addr` / `status_addr` | `uint64_t` | 分别8-byte aligned、指向本action的typed control/status resource |
| 48 | `timeout_cycles` | `uint64_t` | 正数；从issue开始的有限target-cycle budget |
| 56/60/64 | `length` / `packet_bytes` / `packet_count` | `uint32_t` | 正数且经accepted transport verifier证明capacity/segmentation关系 |
| 68/72 | `direction` / `receiver_memory_kind` | `uint32_t` | direction: send=1, recv=2；memory: spm=1, ddr=2 |
| 76/80 | `local_fsm_id` / `remote_fsm_id` | `uint32_t` | 来自完整binding member；必须在environment允许集合中 |
| 84/88 | `local_stream_id` / `remote_stream_id` | `uint32_t` | accepted packet/stream assignment，不由CRT临时分配 |
| 92/96 | `dte_block_id` / `dte_channel` | `uint32_t` | exact accepted block/channel且映射一致；block 0等reserved值由environment拒绝 |
| 100/104/108 | `mode` / `tile_this` / `dst_tile` | `uint32_t` | exact Direct DTE mode和physical endpoint；不是logical rank |
| 112..135 | `stride0,iteration0,...,stride2,iteration2` | six `uint32_t` | byte stride/logical count；contiguous维也显式编码 |
| 136 | `flags` | `uint32_t` | V1固定0；未知bit失败 |
| 140 | `reserved0` | `uint32_t` | 固定0 |
| 144/152 | `reserved1` / `reserved2` | two `uint64_t` | 固定0 |

`WaferTx81DteCommandV1`总长160、alignment 8。它不是新的transport owner：pinned projection把完整member
字段固化到device command storage；relocatable projection只能从同一record的finite allowed members通过
`RelocationSchema`填入已登记字段。任何caller-provided command在target lowering transaction外都非法。

每个action另有128-byte、alignment-8的`WaferTx81DteContextV1 { uint64_t opaque[16]; }`。其唯一ABI语义是
保存一次issue到wait/release之间的provider state；vendor `dte_node`或FSM handle只允许封装在这里，不进入
KAD、package identity、runtime handle或compiler IR。header中的compile-time assertion必须证明repo-local私有
implementation context不超过128 bytes。status record固定为：

```c
typedef struct {
  uint32_t abi_version;      /* 1 */
  uint32_t outcome;
  int32_t provider_code;
  uint32_t flags;            /* V1: 0 */
  uint64_t action_id;
  uint64_t observed_cycles;
} WaferTx81DteStatusV1;      /* size 32, alignment 8 */
```

`outcome`的ABI numeric value固定为`success=0`、`timeout=1`、`transport_error=2`、
`peer_failure=3`、`pending=UINT32_MAX`。Direct DTE CRT只能写`pending/success/timeout/transport_error`：
`peer_failure`没有repo-local device helper evidence，必须由RuntimeSession completion DAG在观察其它rank/stage
failure后写入或合成到同一typed completion surface。禁止把peer timeout冒充local transport error，或让device
CRT凭remote name/status猜peer failure。

固定prototype只有以下三个；参数是typed command record的byte address，要求8-byte alignment：

```c
int32_t wafer_tx81_dte_recv(uint64_t command_addr);
int32_t wafer_tx81_dte_send(uint64_t command_addr);
int32_t wafer_tx81_dte_wait(uint64_t command_addr);
```

immediate return固定为`0=call accepted/status updated`、`-1=invalid command/version/range`、
`-2=provider issue/wait/release failure`、`-3=target capability unsupported`；原始provider return保存在
`provider_code`。成功的send/recv先把status写成pending；wait按direction等待同一context，使用
`timeout_cycles`和nonblocking DTE/FSM status source形成有限等待，写terminal status后释放provider state。
`dte_recv`先配置exact local FSM/stream和receiver storage；`dte_send`只使用exact accepted block/channel、remote
FSM/stream、remote address和endpoint，不能调用会从未约束pool重新选择block/channel的helper；`dte_wait`不等同
local NCC drain。若target revision没有可验证的exact block selection、nonblocking recv status或finite-cycle
wait capability，environment必须把production inline Direct DTE标为unsupported，artifact gate结构化失败，不能
退回unbounded helper、动态resource search或success stub。

instruction lowering为每个send/recv token保留同一`TransportActionId`和command/context/status relation；一个
variadic`wafer.instr.dte_wait`按token producer顺序发出逐action wait并把每个terminal status连接到typed
`CompletionExportId`。pinned常量和relocatable slot都必须来自committed projection。缺字段、half-concrete member、
未绑定status、无法区分terminal outcome或未消费wait的action不得进入device-code symbol-closure gate。

### 7.7 Link 和 required-symbol gate

`tools/wafer_device_link.py` 的执行顺序：

```text
compiler-generated target LLVM IR
  + compiler-generated KernelAbiDescriptor / environment and artifact fingerprints
  -> LLVM clang++ .ll -> target object
  -> TX8 GCC compile runtime/wafer_crt/src/wafer_tx81_crt.c -> wafer CRT object
  -> repo-vendored GCC link target object + wafer CRT object + TX8 deps -> kcore .so
  -> embed/verify mandatory .note.wafer.abi descriptor note
  -> readelf/nm required-symbol and target ABI scan
  -> module digest
```

required-symbol scan 的规则：

- final `.so` 中任何 undefined `wafer_tx81_*` 都是失败。
- 其它 undefined symbol 只能来自按 target/runtime ABI version列出的显式 allowlist；
  `--allow-shlib-undefined` 本身不是合法性证明。
- 测试必须覆盖一个 intentionally missing `wafer_tx81_missing`，证明链接器返回成功也会被 gate 拒绝。
- 成功测试必须从 compiler-generated LLVM IR 或 `wafer-lower-groups-to-target-llvm` output 进入 device link，
  不能只用手写 package/JSON input。
- final ELF的descriptor bytes/hash、environment/artifact fingerprints必须与compiler输出一致；module digest在descriptor
  写入后计算，并作为 package assembly的输入。

### 7.8 Verification

CRT 全量实现的验证分四层：

| 层 | 验证 |
| --- | --- |
| Header/signature | checked symbol registry 确认 Q2-Q3 production closure 中 105 个 `wafer_tx81_*` 都有 `wafer_tx81_crt.h` prototype、repo-local CRT implementation reference 和 lowering family marker |
| CRT object | TX8 GCC 编译 `runtime/wafer_crt/src/wafer_tx81_crt.c`，`nm --defined-only` 确认 object 定义 105 个 production `wafer_tx81_*`，且不定义 Direct DTE symbols |
| Device link | `.ll -> .o -> kcore .so` 实际执行，final `.so` 无 undefined `wafer_tx81_*`；negative test 证明 `wafer_tx81_missing` 会被 required-symbol gate 拒绝 |
| Kernel ABI / ELF identity | compiler-generated `KernelAbiDescriptor` 与 LLVM function type一一对应；`.note.wafer.abi`和package引用使用同一descriptor semantic digest，environment/artifact fingerprints匹配，module digest覆盖final ELF |
| Pipeline integration | `wafer-lower-groups-to-target-llvm` output和 descriptor能进入 device link；HF program-chain target LLVM integration 是下一层 gate，不用手写 package/JSON 代替 |

Q2-Q3 target CRT implementation / device-code required-symbol closure 通过只证明 symbol surface 闭合。
`tasks/progress.md` 还必须优先关闭 target LLVM structured-control correctness、instruction geometry/range
和 exact package entrypoint ABI，才能恢复 Q4 typed `PackageManifest` assembly。只实现 header、只生成
object、只覆盖一个代表性 instruction family、或只靠 `--allow-shlib-undefined` 得到 `.so` 都不算完成。

## 8. Extended Target CRT Surface Staging

本节只定义旧 TX81 CRT source 中未进入当前 production closure 的能力如何分级、何时可以进入
Wafer target CRT。它不直接扩大 Q2-Q3 的 105 个 production `wafer_tx81_*` symbols，也不把旧
`__*` ABI 作为兼容目标。

### 8.1 Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  旧 TX81 CRT source audit、当前 `wafer.instr.*` production coverage matrix、target LLVM call ABI
  和 repo-local Wafer CRT conformance gate。
- Current stage responsibility:
  把旧 source 中未进入 production CRT 的函数按 IR / ABI 缺口分级，决定哪些可以扩展 instruction
  IR 后进入 production，哪些必须先成为 composite/layout/DTE IR，哪些永久不属于 compiler ABI。
- Output artifact / IR:
  extended surface staging matrix、后续 instruction IR / target LLVM / CRT implementation 任务边界，
  以及每类能力的 completion gate。
- Downstream consumer:
  `tasks/11-instruction-ir.md` 的 op/kind/verifier 扩展、`LowerInstrToTargetLLVM.cpp` typed call
  emission、repo-local Wafer CRT implementation、device-link required-symbol gate 和 package/runtime gate。
- User-level driver / named pipeline:
  仍由`stablehlo-to-executable` direct production driver消费；局部
  `wafer-lower-groups-to-target-llvm`和`--wafer-lower-instr-to-target-llvm`只重放stage，extended surface
  不能引入要求用户手动调用旧CRT helper的长期流程。
- Explicit non-goals:
  不恢复 `libvr.a`，不暴露 `__Count` / `__Gelu*` / `__Send` 等旧 ABI 名称，不在 CRT 内隐藏
  scheduler、scratch allocator、layout planner 或 DTE endpoint binder。
- Completion gate:
  每个 promoted family 必须同时更新 instruction IR/verifier、target LLVM typed ABI、CRT symbol
  implementation、conformance checker、device-link required-symbol gate 和 positive/negative lit；
  只添加 CRT 函数、只添加 header prototype、或只复制旧 helper 都不算完成。
```

### 8.2 Staging Status

| status | meaning | production entry rule |
| --- | --- | --- |
| `already-covered` | 旧 source 的有用语义已经被当前 production `wafer_tx81_*` 或 verifier 规则覆盖 | 不新增 symbol；只保留 checker / doc gate |
| `promote-now` | public wrapper 语义清楚，缺口主要是显式 IR kind/result ABI 和 verifier | 下一实现批次可以扩 `wafer.instr.*`、target LLVM ABI、CRT 和 tests 一起推进 |
| `needs-composite-ir` | 旧 helper 依赖多条 wrapper issue、software loop、scratch buffer、SPM mapping 或显式 completion | 先设计 composite instruction / region / lowering sequence；不能先塞进 CRT helper |
| `designed-composite` | typed semantic/storage/instruction/command ABI与completion合同已固定，但production implementation/golden尚未闭合 | 严格按已编号设计实现；在全部gate前保持target-illegal |
| `needs-layout-ir` | 旧 helper 本质是 layout/materialization/movement 规划 | 先用 layout/materialization IR 和 gather/scatter descriptor 表达；CRT 只接收已合法化 movement |
| `needs-dte-abi` | 旧 helper 依赖 endpoint、channel、remote FSM、tile topology 或 sync slot binding | 先扩 DTE instruction ABI 和 runtime binding；不能用空 send/recv 函数补符号 |
| `reject-permanently` | link/runtime compatibility、assert/print/math stub 或旧工程 glue | 不进入 Wafer IR、target LLVM ABI 或 production CRT |

### 8.3 Extended Surface Matrix

| old source family | status | required design before production |
| --- | --- | --- |
| `count.c::__Count` | `promote-now` | 增加 explicit scalar writeback peripheral IR/ABI；value destination 是 SPM offset，CRT wait 后 mapping 写回，和 argmax/argmin 共用 writeback policy |
| arith/relation scalar-immediate `*VS` forms | `promote-now` | 增加显式 scalar-immediate instruction kind 或 operand form；verifier 区分 VV/VS，不从 operand 名字或 old ABI 恢复 |
| GELU helpers `__GeluNone` / `__GeluTanh` / `op_gelu_*` | `needs-composite-ir` | 表达 approximation mode、dtype conversion policy、scratch buffer、wrapper issue order 和 completion；优先 lower 成已有 convert/arith/activation sequence |
| `reduce_mul` | `needs-composite-ir` | 若无 native public reduce-mul wrapper 证据，必须表达初始化、multiply reduction order、temporary storage 和 numeric policy |
| GEMM `SetQuant` / scale surface | `promote-now` | 使用`wafer.instr.quantized_gemm`和`WaferTx81QuantizedGemmCommandV1`；只开放capability证明的INT8 subset，不修改plain GEMM ABI |
| MXFP convert / scale helpers | `designed-composite` | 按`wafer.instr.mxfp_decode`和`WaferTx81MxfpDecodeCommandV1`实现software SPM decode、scale、scratch和local completion；不导出旧helper ABI |
| channelnorm/dechannelnorm、concat、transpose、mirror、rotate、NCHW/NHWC、tensornorm | `needs-layout-ir` | 先在 layout/materialization 层表达 permutation/axis/padding/segment movement；target CRT 只接收已合法化 gather/scatter 或专门 TDMA symbol |
| Direct DTE `send` / `recv` / atomic barrier helpers | `needs-dte-abi` | 表达 local/remote tile、channel、FSM id、receive buffer、sync slot 和 completion token；旧 4x4 ring prototype 不能成为 ABI |
| `common.c`、`empty.c`、`assert.c`、`print.c`、software `powf` glue | `reject-permanently` | 只属于旧 runtime/link compatibility；Wafer 需要这类能力时应走 toolchain/runtime 标准库或明确 diagnostics |

### 8.4 Implementation Ordering

后续不能按“发现一个旧函数就补一个 CRT symbol”的方式推进。实现批次必须按 surface family 闭环：

1. **Writeback scalar batch**：`count` 和未来同类 writeback peripheral result。完成条件是 IR/verifier、
   target LLVM ABI、mapped SPM writeback CRT、checker、device-link 和 negative arity tests 同时更新。
2. **Scalar-immediate wrapper batch**：arith/relation `VS` 等显式 scalar-immediate forms。完成条件是
   instruction IR 能表达 scalar operand，target LLVM fixed ABI 不复用 VV symbol，不引入旧 `__*` 名称。
3. **Layout/materialization batch**：transpose/mirror/rotate/NCHW-NHWC/channelnorm/concat 等先统一走
   layout/materialization IR 和 gather/scatter descriptor legality；只有 public TDMA wrapper 比
   gather/scatter 更能表达且 verifier 能检查时，才新增 target CRT symbol。
4. **Low-precision batch**：先实现native affine INT8 capability/instruction/command/golden，再实现已设计的MXFP
   explicit decode composite及scratch/order/completion；GELU/reduce-mul仍必须先有各自composite IR。CRT不做隐藏planner。
5. **Direct DTE batch**：补 endpoint/channel/FSM/runtime binding 后再恢复 production `wafer_tx81_dte_*`
   closure。

每个批次完成后都要更新本节、`tasks/11-instruction-ir.md`、conformance matrix、symbol checker 和
device-link tests；不能只更新其中一层。

## 9. 当前缺口

- Wafer-owned target CRT implementation、typed wrapper contract、CRT object definition check 和
  device-code required-symbol closure 已覆盖 Q2-Q3 production closure 中的 105 个 symbols。
- 更细的 register-packet field golden capture 可作为后续硬化项；不能反向替代当前 fixed ABI /
  repo-local CRT / required-symbol gate，也不能把 opaque `libvr.a` alias 恢复为主线。
- Device-code compile/link helper 已能对已有 / compiler-generated LLVM IR 执行 `.ll -> .o -> kcore .so`
  和 object metadata normalization；production `wafer_tx81_*` 不能以未解释 undefined symbol 形式残留，
  也不能经过 capture shim。
- Structured control conversion、shared geometry legality和 compiler-generated `KernelAbiDescriptor`尚未
  完成；在 false branch/loop/call语义、边界范围和 descriptor-to-ELF一致性 gate通过前，straight-line
  call emission不能升级为完整 target artifact合同。
- `TargetEnvironmentFingerprint`、`TargetArtifactFingerprint`、ELF mandatory `.note.wafer.abi`和final module digest必须成为device-code稳定输出；
  package auto-export不能继续解析 LLVM文本、参数数量或 module path恢复 ABI。
- Extended target CRT surface 已按 `already-covered`、`promote-now`、`needs-composite-ir`、
  `needs-layout-ir`、`needs-dte-abi` 和 `reject-permanently` 分级；后续不能只补 CRT 函数，必须按
  surface family 闭环 IR、ABI、CRT、checker 和 device-link gate。
- package assembly后续只消费committed executable和原子发布的complete `TargetArtifactSet`；恢复
  compiler-generated package gate要等device-code compile/link gate与这些artifact衔接完成。
