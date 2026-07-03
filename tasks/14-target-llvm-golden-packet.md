# Wafer Target LLVM Lowering and Golden Packet Design

状态：设计草案；范围：memory-planned target-aligned `wafer.instr.*` 到 target CRT call、LLVM dialect /
LLVM IR 和 wrapper/register golden packet 的 lowering。

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

## 1. 目标和非目标

目标：

- 定义 `wafer.instr.*` 到 Wafer-owned target CRT / LLVM call 的 lowering 边界。
- 固定地址单位、SPM/DDR offset 消费、format、shape/stride、wait/completion 和 status 责任。
- 要求每个可 lower 的 `wafer.instr.*` 都有明确 public TSM wrapper、Direct DTE helper 或 TX81/CRT
  证据；证据不能直接替代 Wafer-owned symbol 合同。
- 通过 golden packet / wrapper tests 验证 target CRT 参数到 TSM wrapper/register packet 的映射。

非目标：

- 不恢复 compiler-facing helper ABI family。
- 不新增 `wafer.abi` dialect。
- 不把 capture shim、C stub 表格或 package metadata input 当 production lowering。
- 不重新选择 group、tile shape、layout、SPM/DDR memory plan 或 communication schedule。
- 不把 Tx81 CRT 的所有符号原样提升为 Wafer IR 或 Wafer target ABI；Wafer IR 只表达当前 pipeline
  需要且 verifier 能检查的目标动作。

## 2. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  memory-planned `wafer.tile.region` / `wafer.instr.*` IR、accepted SPM/DDR offset facts、
  topology/execution-mesh contract、program parameter shard metadata/resource view、薄 launch/block binding
  和 communication/sync lowering。
- Current stage responsibility:
  从 target-aligned instruction IR 和 accepted facts 派生 Wafer-owned target CRT calls，并继续 lower 到 LLVM
  dialect / LLVM IR。该阶段消费 Wafer memory attr、SPM/DDR offset、DTE peer/token、layout/format
  和 resource view；不读取 pass-local side table，不发明 compiler-facing ABI wrapper。
- Output artifact / IR:
  LLVM dialect module、LLVM IR artifact、Wafer target CRT symbol declarations/calls、debug/golden-packet
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
- lowering 可以找到 Wafer-owned target CRT symbol 方案或明确记录 unsupported diagnostic。
- coverage 必须来自 `tasks/11-instruction-ir.md` 的 instruction coverage matrix。只有标为
  `V0 native instr` 或 `V0 native sync` 的 op 是 target LLVM lowering 的 production 输入；
  `V0 composite lowering` 必须已经在 instruction lowering 前展开，`future` / `unsupported`
  不能通过现有泛 op 隐式进入 LLVM lowering。

当前状态：

| Wafer instr | target evidence / lowering direction | 状态 |
| --- | --- | --- |
| `wafer.instr.rdma` / `wdma` | TSM RDMA/WDMA wrapper；target CRT 应接收 DDR/SPM pointer/offset、shape/stride、format | pending target LLVM |
| `wafer.instr.gather_scatter` | `TsmDataMove::GatherScatter` / TX81 gather-scatter CRT evidence | pending target LLVM |
| `wafer.instr.fill` | `TsmPeripheral::Memset` evidence | pending target LLVM |
| `wafer.instr.elementwise` | `#wafer.instr_elementwise_kind` CT arith/relation/activation/transcendental target wrapper families；select 不在该 enum 中 | pending target LLVM |
| `wafer.instr.bit2fp` | Triton/TX81 `mk.bit2fp -> tx81.bit2fp -> __Bit2Fp` as evidence; Wafer lowering emits Wafer-owned CRT symbol | IR added, LLVM pending |
| `wafer.instr.mask_move` | Triton/TX81 `mk.mask_move -> tx81.mask_move -> __MaskMove` as evidence; header/CRT mask parameter mismatch must be resolved in Wafer CRT | IR added, LLVM pending |
| `wafer.instr.reduce` | `#wafer.instr_reduce_kind` target wrapper families plus native `dim` code | pending target LLVM |
| `wafer.instr.convert` | `#wafer.instr_convert_kind` opcode 139..174 dtype pair plus zero-point/rounding/plain signature groups；same-format copy must lower through movement, not convert | pending target LLVM |
| `wafer.instr.gemm` | `TsmGemm` wrapper / `__Gemm` style CRT evidence; Wafer-owned symbol must not expose Tx81 ABI verbatim | pending target LLVM |
| `wafer.instr.conv` | `#wafer.instr_conv_kind` NE Conv/Depthwise/BackwardConv target wrapper families；optional/fused operands are not implicit | IR added, LLVM pending |
| `wafer.instr.pool` / `unpool` | `#wafer.instr_pool_kind` / `#wafer.instr_unpool_kind` CT Pool/UnPool wrapper families; unpool scalar index is wrapper-aligned | IR added, LLVM pending |
| `wafer.instr.tdma_data_move` | V0 production only covers pad/img2col wrappers; ordinary copy and transpose-like movement use gather/scatter | IR added, LLVM pending |
| `wafer.instr.peripheral` | `#wafer.instr_peripheral_kind` for arg/factorize/bilinear/LUT/rand/elem_mask; count writeback and bitcount remain unsupported | IR added, LLVM pending |
| `wafer.instr.dte_send` / `dte_recv` / `dte_wait` | Direct DTE/FSM runtime binding evidence still incomplete | partial IR done, production lowering pending |
| `wafer.instr.local_fence` | `TsmWaitfinish` / local drain evidence | pending target LLVM |

Target LLVM lowering 输出必须调用 Wafer-owned target CRT symbols。`__*` 符号可以出现在 Wafer CRT
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
| relation / logic | `wafer_tx81_relation_*` / `wafer_tx81_logic_*` | value/bool variants differ | source arity、dest storage、element count、format；bitpacked bool/VuV/Loop 需要后续扩 IR，不能由当前 generic elementwise 隐式选择 |
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
| transpose-like TDMA transform group | no production Wafer symbol in V0 | mirror/transpose/rotate/NCHW-NHWC/TensorNom wrappers exist but are not default lowering surface | lower via `gather_scatter` or fail; future native enablement requires board/golden coverage and an explicit verifier update |
| `peripheral` arg/factorize/rand | `wafer_tx81_peripheral_*` | arity differs by kind | verifier fixes input/dest arity; arg index dest must be i32 |
| `peripheral` bilinear/LUT/elem_mask | separate Wafer symbols or descriptor variants | bilinear uses shape attrs; LUT uses `lut_elem_count`; elem_mask uses scale/probability/rounding | required/forbidden attrs already checked by IR/package validator |
| `peripheral count` | no production symbol | public wrapper writeback is not represented as normal dest buffer | verifier rejects until IR gains explicit writeback/result semantics |
| Direct DTE | `wafer_tx81_dte_send` / `wafer_tx81_dte_recv` / wait helper | Direct DTE/FSM helper, not `TsmExecute` | logical peer and bytes from IR; endpoint/channel binding from execution mesh/topology/runtime |
| local fence | `wafer_tx81_local_fence` | local wait/drain helper | no hidden multi-tile barrier semantics |

不在 V0 native coverage 中的硬件能力不属于本 stage 的默认 lowering surface。concat、maskgather
variants、raw DTE non-unicast、SCALAR/CSR ordinary execution、Peripheral bitcount 和 Conv optional/fused
operand policy 等能力需要先在 instruction IR 中新增或扩展明确 op / kind / verifier，并更新 coverage
matrix；target LLVM lowering 不能用 `wafer.instr.elementwise`、`gather_scatter` 或 ad hoc CRT call
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

LLVM lowering 直接声明/调用 Wafer-owned target CRT symbol。示例形态：

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

- `wafer.instr.* -> LLVM call @wafer_tx81_*` pass 尚未实现。
- Wafer-owned target CRT symbol coverage 需要逐个 op 对齐 repo-local TX81/public TSM wrapper 和硬件文档。
- Direct DTE production lowering 还缺 runtime endpoint / DTE channel binding。
- package auto-export 当前只能消费已有 LLVM IR；恢复 compiler-generated package gate 要等 target LLVM
  lowering 完成。
