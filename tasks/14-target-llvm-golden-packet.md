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
  带 ABI descriptor note/export 的 kcore shared object、module digest 和 symbol report 由 device-code gate
  从这些 compiler-generated artifacts 继续生成。
- Downstream consumer:
  device-code compile/link gate、IR-derived `PackageManifest` assembly、wrapper-facing call contract 和
  board/runtime adapter。
- User-level driver / named pipeline:
  production主线由`wafer-opt --program-pipeline=stablehlo-to-executable`或等价driver消费committed
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
  fingerprint 和 module digest 验证；package manifest引用的 descriptor hash必须与 ELF note/export一致。
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

- kernel ABI schema/version、stable entry ABI id、LLVM symbol和return type。executable variant、shape guard、
  rank/stage class和endpoint mapping由executable/package entry graph关联，不进入函数调用ABI本体。
- ordered argument slots；每个slot使用ABI-local稳定`SlotId`并记录role、LLVM scalar/pointer
  representation、address space、access、alignment、shape/capacity relation和alias contract。具体
  `ResourceId`由`wafer.executable.entry`的`SlotId -> ResourceId` binding提供，不进入descriptor hash；参数
  顺序是descriptor的typed sequence，不另存可独立修改的`binding_order`。
- external IO、immutable weight、persistent state、workspace和 relocatable endpoint/control/status resource
  slots；不存在的 role 不生成占位参数，存在的 slot 必须与 LLVM function parameter 一一对应。
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

单function transaction不是发布边界。一个`TargetVariantId`引用的全部rank-class modules必须先写入staging
artifact set，逐个完成structure-preserving conversion、device link、required-symbol、ELF ABI descriptor/
hash、environment/artifact fingerprints和module digest验证，再原子发布完整`TargetArtifactSet`。任一module失败则该set
整体不可见，PackageManifest不得引用已成功的子集。content-addressed cache可以保留独立verified blob，
但cache存在不等于variant artifact set已accepted。

`TargetArtifactSet` root绑定source `wafer.executable` program semantic digest和`TargetVariantId`；稳定
`TargetArtifactSetId`/root digest由canonical ordered member map、environment/artifact fingerprints和全部member
digests计算。每个member显式映射
`EntryId + static-function semantic digest -> module id + entry symbol + KernelAbiDescriptor hash + covered
RankClassIds`。相同ABI不能授权替换不同code digest或其它executable的module。set verifier还必须检查每个
`CompletionExportId`与executable entry/DAG引用一一闭合。

Device-code gate 将 canonical descriptor/hash写入 final ELF 的 Wafer ABI note和/或只读 exported
descriptor symbol，并从包含该 descriptor 的 final ELF 计算 module digest。`PackageManifest` 记录同一
descriptor hash、environment/artifact fingerprints和module digest；loader在module load和launch前逐项比对。
manifest自身不嵌回 ELF，因此 module digest、descriptor hash和 manifest引用之间没有循环依赖。

### 4.2 Shared Geometry Legality

所有会进入 LLVM `i32`、CRT `uint32_t`、TX81 `uint16_t` shape field、address offset或 byte descriptor 的
geometry 使用一个共享 legality contract。instruction verifier、target lowering preflight、
`KernelAbiDescriptor` verifier和 golden tests必须复用同一字段范围与单位定义，至少检查：

- static/dynamic dimension和 variant bound满足 target field width，乘积与 byte-size计算不溢出。
- byte count、inner bytes、element bytes、stride/iteration和 descriptor coverage一致且需要时整除。
- SPM/DDR address、offset、alignment和 end address在 accepted allocation内。
- rank/stage launch geometry、transport buffer capacity和 kernel slot shape/capacity relation一致。

超范围 geometry 只能在 target legality或 variant guard处被拒绝，不能通过 LLVM integer constant
截断、C cast、default format或 runtime best-effort继续执行。

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
  triple、ISA/extensions、MABI、kernel ABI schema、Wafer CRT/symbol-set、quant/storage ABI、toolchain/
  device-link profile和codegen dependencies。pinned code把`ProjectionSetId`/projection digest及必要
  topology assumption纳入artifact fingerprint；relocatable code只纳入relocation slot ABI/schema，不把运行时
  选择的member伪装成target environment。

Runtime先按environment fingerprint过滤target axis，再独立选择committed projection；artifact fingerprint
只验证所选module与该组合兼容，不参与shape guard或rank-class选择。

Final link 之后计算 module digest，并验证 ELF machine、ISA/MABI、exports、undefined-symbol policy和 ABI
descriptor note/export。environment/artifact fingerprints、descriptor hash和module digest都是object/package stage的
typed输出；module path只用于定位文件，不能承担 identity或 compatibility语义。

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
  Wafer CRT RISC-V object / archive、带 ABI descriptor note/export 的 kcore shared object、
  required-symbol/ELF ABI report、environment/artifact fingerprints、descriptor hash和module digest。module path只是
  artifact locator，不是 package identity或 ABI source。
- Downstream consumer:
  device-code compile/link symbol-closure gate、IR-derived `PackageManifest`、runtime adapter / board gate。
- User-level driver / named pipeline:
  production由`stablehlo-to-executable`或等价driver的target-artifact stage执行；
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
  production `wafer_tx81_*`；ELF descriptor hash与compiler descriptor一致，environment/artifact fingerprints和module
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
- 每个 function 返回 `void`。错误检查属于 verifier / target lowering / device link gate；CRT 内部不通过
  silent return 表示 unsupported。

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

### 7.6 DTE ABI

Direct DTE 不能通过 `TsmExecute` 发射。`direct_dte_send_async` 需要 `DirectDTESendInfo`，字段包括
`src_addr`、`dst_addr`、`length`、`remote_fsm_id`、`mode`、`dst_tile`、`tile_this`、
`stride_iterations[3]` 和 `dte_node`。因此当前 call-emission 只传
`buffer, peer, bytes` 的 shape 不足以作为完整 production DTE ABI。

`wafer_tx81_dte_send`、`wafer_tx81_dte_recv` 和 `wafer_tx81_dte_wait` 不进入当前 Q2-Q3 production
CRT closure。要重新纳入 production closure，必须先改 ABI / lowering：

- `wafer.instr.dte_*` lowering 必须消费 `tasks/13` 的 accepted physical transport，而不是重新从 logical
  peer 猜测 endpoint。pinned 模式 materialize `tile_this`、`dst_tile`、channel/FSM和 receiver buffer；
  relocatable 模式通过 `KernelAbiDescriptor` 暴露 endpoint/control/status slots。
- `wafer_tx81_dte_recv` 负责初始化或更新 local FSM monitor，使 peer send 有明确 remote destination。
- `wafer_tx81_dte_send` 负责 attach/config/send Direct DTE node，不允许假设 remote dst address 等于 local source。
- `wafer_tx81_dte_wait` 只等待由 send/recv 建立的 DTE/FSM completion，不做 local NCC drain，并把
  success、timeout、transport error和 peer failure映射到 descriptor声明的 status/error surface。

如果这些字段尚未能从 IR/runtime binding materialize，DTE lowering 不能继续作为 production target path
进入 device-code symbol-closure gate；必须先扩 ABI 或扩 IR，而不是在 CRT 中写空成功函数。

### 7.7 Link 和 required-symbol gate

`tools/wafer_device_link.py` 的执行顺序：

```text
compiler-generated target LLVM IR
  + compiler-generated KernelAbiDescriptor / environment and artifact fingerprints
  -> LLVM clang++ .ll -> target object
  -> TX8 GCC compile runtime/wafer_crt/src/wafer_tx81_crt.c -> wafer CRT object
  -> repo-vendored GCC link target object + wafer CRT object + TX8 deps -> kcore .so
  -> embed/verify ABI descriptor note/export
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
| Kernel ABI / ELF identity | compiler-generated `KernelAbiDescriptor` 与 LLVM function type一一对应；ELF note/export和package引用使用同一descriptor hash，environment/artifact fingerprints匹配，module digest覆盖final ELF |
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
  仍由`stablehlo-to-executable`或等价production driver消费；局部
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
| MXFP convert / scale helpers | `needs-composite-ir` | 先引入 dtype/type 或 explicit packed-format contract；表达 software SPM loads/stores、scale decode、scratch 和 completion |
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
4. **Composite compute batch**：GELU/MXFP/reduce-mul 必须先有 composite IR 或 explicit lowering sequence，
   以及 scratch/order/completion gate；CRT 不做隐藏 planner。
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
- `TargetEnvironmentFingerprint`、`TargetArtifactFingerprint`、ELF ABI descriptor note/export和final module digest必须成为device-code稳定输出；
  package auto-export不能继续解析 LLVM文本、参数数量或 module path恢复 ABI。
- Extended target CRT surface 已按 `already-covered`、`promote-now`、`needs-composite-ir`、
  `needs-layout-ir`、`needs-dte-abi` 和 `reject-permanently` 分级；后续不能只补 CRT 函数，必须按
  surface family 闭环 IR、ABI、CRT、checker 和 device-link gate。
- package assembly后续只消费committed executable和原子发布的complete `TargetArtifactSet`；恢复
  compiler-generated package gate要等device-code compile/link gate与这些artifact衔接完成。
