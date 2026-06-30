# Wafer Target LLVM Lowering and Golden Packet Design

状态：设计草案；范围：memory-planned target-aligned `wafer.instr.*` 到 target CRT call、LLVM dialect /
LLVM IR 和 wrapper/register golden packet 的 lowering。

本文取代旧 compiler-facing helper ABI 设计。当前结论是：`wafer.instr.*` 必须对齐目标指令、
TX81 target op 或 public TSM wrapper 的可 lower 粒度；LLVM lowering 直接生成 target CRT symbol
调用，例如 `__Gemm`、`__Bit2Fp`、`__MaskMove` 这类由 TX81/CRT 证据支持的符号。主线不再经过
Wafer 自定义 helper ABI，也不再编译或链接 capture shim。

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
  -> LLVM call @__Foo
  -> TX81 CRT / Wafer CRT implementation
  -> TSM wrapper + TsmExecute or runtime/DTE helper
```

本文依赖：

- `tasks/10-compute-movement.md`
- `tasks/11-instruction-ir.md`
- `tasks/13-communication.md`
- `tasks/15-launch-runtime-package.md`
- `docs/wafer-register-level-instruction-spec.md`
- `docs/tx8-deps-reverse-engineering/tx8-interface-contract.md`

## 1. 目标和非目标

目标：

- 定义 `wafer.instr.*` 到 target CRT / LLVM call 的 lowering 边界。
- 固定地址单位、SPM/DDR offset 消费、format、shape/stride、wait/completion 和 status 责任。
- 要求每个可 lower 的 `wafer.instr.*` 都有明确 TX81/TSM wrapper 或 target CRT symbol 证据。
- 通过 golden packet / wrapper tests 验证 target CRT 参数到 TSM wrapper/register packet 的映射。

非目标：

- 不恢复 compiler-facing helper ABI family。
- 不新增 `wafer.abi` dialect。
- 不把 capture shim、C stub 表格或 package metadata input 当 production lowering。
- 不重新选择 group、tile shape、layout、SPM/DDR memory plan 或 communication schedule。
- 不把 Tx81 CRT 的所有符号原样提升为 Wafer IR；Wafer IR 只表达当前 pipeline 需要且 verifier
  能检查的目标动作。

## 2. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  memory-planned `wafer.tile.region` / `wafer.instr.*` IR、accepted SPM/DDR offset facts、
  topology/execution-mesh contract、program parameter shard metadata/resource view、薄 launch/block binding
  和 communication/sync lowering。
- Current stage responsibility:
  从 target-aligned instruction IR 和 accepted facts 派生 target CRT calls，并继续 lower 到 LLVM
  dialect / LLVM IR。该阶段消费 Wafer memory attr、SPM/DDR offset、DTE peer/token、layout/format
  和 resource view；不读取 pass-local side table，不发明 compiler-facing ABI wrapper。
- Output artifact / IR:
  LLVM dialect module、LLVM IR artifact、target CRT symbol declarations/calls、debug/golden-packet
  输入。TX8 relocatable object 和 kcore shared object 由 device-code gate 从 LLVM IR 继续生成。
- Downstream consumer:
  device-code compile/link gate、IR-derived package metadata、wrapper-facing call contract 和
  board/runtime adapter。
- User-level driver / named pipeline:
  当前稳定主线仍是 `wafer-lower-groups-to-ddr-memory-planned-instr`。target LLVM lowering 完成后，
  再引入语义命名的 program pipeline；不恢复旧 ABI/LLVM pipeline 名称。
- Explicit non-goals:
  不重新做 frontend/SPMD/group/tile/layout/SPM/DDR/communication planning；不把 helper ABI、
  capture shim 或 C stub emission 作为中间层。
- Completion gate:
  supported `wafer.instr.*` 生成 verifier-legal LLVM dialect / LLVM IR，并能由 `mlir-translate`
  输出 LLVM IR；unsupported target op 结构化失败。device-code gate 只链接 target LLVM object 和
  TX8/CRT 依赖，不默认链接 capture shim。
```

## 3. Instruction Legality

`wafer.instr.*` 是 target instruction-ish IR，不是新的 semantic buffer IR。进入本 stage 前，每个 op
必须满足：

- operand/result buffer 是 Wafer-tagged memref，SPM/DDR domain 可从 type 和 accepted offset facts 推出。
- op family、dtype/layout、shape/stride、byte count、wait/completion 和 resource effects 能由 verifier
  检查。
- lowering 可以找到 target CRT symbol 或明确记录 unsupported diagnostic。

当前状态：

| Wafer instr | target evidence / lowering direction | 状态 |
| --- | --- | --- |
| `wafer.instr.rdma` / `wdma` | TSM RDMA/WDMA wrapper；target CRT 应接收 DDR/SPM pointer/offset、shape/stride、format | pending target LLVM |
| `wafer.instr.gather_scatter` | `TsmDataMove::GatherScatter` / TX81 gather-scatter CRT evidence | pending target LLVM |
| `wafer.instr.fill` | `TsmPeripheral::Memset` evidence | pending target LLVM |
| `wafer.instr.elementwise` | CT arith/relation/activation/transcendental wrapper families；仅保留非-select kind | pending target LLVM / further split audit |
| `wafer.instr.bit2fp` | Triton/TX81 `mk.bit2fp -> tx81.bit2fp -> __Bit2Fp` | IR added, LLVM pending |
| `wafer.instr.mask_move` | Triton/TX81 `mk.mask_move -> tx81.mask_move -> __MaskMove` | IR added, LLVM pending |
| `wafer.instr.reduce` | `TsmReduce` wrapper families | pending target LLVM |
| `wafer.instr.convert` | `TsmConvert` wrapper families；same-format copy may lower through movement | pending target LLVM |
| `wafer.instr.gemm` | `TsmGemm` wrapper / `__Gemm` style CRT evidence | pending target LLVM |
| `wafer.instr.dte_send` / `dte_recv` / `dte_wait` | Direct DTE/FSM runtime binding evidence still incomplete | partial IR done, production lowering pending |
| `wafer.instr.local_fence` | `TsmWaitfinish` / local drain evidence | pending target LLVM |

`wafer.instr.elementwise <select>` 非法。tile semantic select 需要在 instruction lowering 中改写成目标
序列：

```text
false_value -> dest                    // SPM copy / gather_scatter
predicate i1 -> floating mask           // wafer.instr.bit2fp
true_value + mask -> dest               // wafer.instr.mask_move
```

这与 Triton/TX81 的 select lowering 对齐，避免生成不存在的 Wafer 自定义 select helper。

## 4. LLVM Call Shape

LLVM lowering 直接声明/调用 target CRT symbol。示例形态：

```mlir
// before
wafer.instr.bit2fp %pred into %mask
  : memref<8xi1, #wafer.memory<spm, tensor>>
  to memref<8xf32, #wafer.memory<spm, tensor>>

// after target lowering, schematic
llvm.call @__Bit2Fp(%pred_addr, %mask_addr, %elem_count, %fmt)
```

```mlir
// before
wafer.instr.mask_move %src, %mask into %dst
  : memref<8xf32, #wafer.memory<spm, tensor>>,
    memref<8xf32, #wafer.memory<spm, tensor>>
  into memref<8xf32, #wafer.memory<spm, tensor>>

// after target lowering, schematic
llvm.call @__MaskMove(%src_addr, %dst_addr, %elem_count, %mask_addr, %fmt)
```

具体 symbol、参数顺序和返回值以 repo-local TX81/Wafer CRT 证据为准；没有证据的 op 不能 silent
fallback 到 fake ABI。

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
  -> wafer.execution.mesh rank-domain endpoint view
  -> wafer.target.topology physical endpoint / runtime DTE channel
```

external DDR function argument 不允许被解释成编译期绝对地址；它只代表 runtime launch binding base。
`#wafer.ddr_offset` 只适用于 compiler-managed / resident / workspace DDR allocation。

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

## 7. 当前缺口

- `wafer.instr.* -> LLVM call @__*` pass 尚未实现。
- target CRT symbol coverage 需要逐个 op 对齐 repo-local TX81/Wafer CRT 和硬件文档。
- Direct DTE production lowering 还缺 runtime endpoint / DTE channel binding。
- package auto-export 当前只能消费已有 LLVM IR；恢复 compiler-generated package gate 要等 target LLVM
  lowering 完成。
