# Wafer Target LLVM Lowering and Golden Packet Design

状态：实现中；target LLVM call-emission pass 已落地，Wafer CRT implementation、device-code link gate 和
wrapper/register golden packet 仍待完成。范围：memory-planned target-aligned `wafer.instr.*` 到
target CRT call、LLVM dialect / LLVM IR 和 wrapper/register golden packet 的 lowering。

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
  和 resource view；不读取 pass-local side table，不发明 compiler-facing ABI wrapper。当前已落地的是
  call emission：生成 `wafer_tx81_*` declarations/calls 并保证 unsupported target op 结构化失败；
  这不证明对应 repo-local Wafer CRT symbol 已定义或 packet mapping 已 golden。
- Output artifact / IR:
  LLVM dialect module、LLVM IR artifact、Wafer target CRT symbol declarations/calls、debug/golden-packet
  输入。TX8 relocatable object 和 kcore shared object 由 device-code gate 从 LLVM IR 继续生成。
- Downstream consumer:
  device-code compile/link gate、IR-derived package metadata、wrapper-facing call contract 和
  board/runtime adapter。
- User-level driver / named pipeline:
  局部 pass 入口是 `--wafer-lower-instr-to-target-llvm`；group 边界 named pipeline 是
  `wafer-lower-groups-to-target-llvm`。真实 HF program pipeline 升级到 target LLVM 前，后续
  device-code/package gate 只能消费 compiler-generated target LLVM artifact，不恢复旧 ABI/LLVM pipeline 名称。
- Explicit non-goals:
  不重新做 frontend/SPMD/group/tile/layout/SPM/DDR/communication planning；不把 helper ABI、
  capture shim 或 C stub emission 作为中间层。
- Completion gate:
  call-emission 子 gate 要求 supported `wafer.instr.*` 生成 verifier-legal LLVM dialect / LLVM IR，并能由
  `mlir-translate` 输出 LLVM IR；unsupported target op 结构化失败。完整 target CRT / golden boundary
  还要求 Wafer-owned CRT symbol 有 typed wrapper 合同和 packet golden coverage。device-code gate 只链接
  target LLVM object 和 TX8/CRT 依赖，不默认链接 capture shim，并负责 required-symbol closure。
```

## 3. Instruction Legality

`wafer.instr.*` 是 target instruction-ish IR，不是新的 semantic buffer IR。进入本 stage 前，每个 op
必须满足：

- operand/result buffer 是 Wafer-tagged memref，SPM/DDR domain 可从 type 和 accepted offset facts 推出。
- op family、dtype/layout、shape/stride、byte count、wait/completion 和 resource effects 能由 verifier
  检查。
- lowering 可以找到 Wafer-owned target CRT symbol 方案或明确记录 unsupported diagnostic。
- coverage 必须来自 `tasks/11-instruction-ir.md` 的 instruction coverage matrix。只有标为
  `V0 production target op` 或 `V0 production target sync` 的 op 是 target LLVM call emission 的 production 输入；
  `V0 composite lowering` 必须已经在 instruction lowering 前展开，`future` / `unsupported`
  不能通过现有泛 op 隐式进入 LLVM lowering。

当前状态：

| Wafer instr | target evidence / lowering direction | 状态 |
| --- | --- | --- |
| `wafer.instr.rdma` / `wdma` | TSM RDMA/WDMA wrapper；target CRT 接收 DDR/SPM pointer/offset、shape/stride、format | LLVM call emitted；CRT/golden packet pending |
| `wafer.instr.gather_scatter` | `TsmDataMove::GatherScatter` / TX81 gather-scatter CRT evidence | LLVM call emitted；CRT/golden packet pending |
| `wafer.instr.fill` | `TsmPeripheral::Memset` evidence | LLVM call emitted；CRT/golden packet pending |
| `wafer.instr.elementwise` | `#wafer.instr_elementwise_kind` CT arith/relation/activation/transcendental target wrapper families；select 不在该 enum 中 | per-kind LLVM call emitted；CRT/golden packet pending |
| `wafer.instr.bit2fp` | Triton/TX81 `mk.bit2fp -> tx81.bit2fp -> __Bit2Fp` as evidence; Wafer lowering emits Wafer-owned CRT symbol | LLVM call emitted；CRT/golden packet pending |
| `wafer.instr.mask_move` | Triton/TX81 `mk.mask_move -> tx81.mask_move -> __MaskMove` as evidence; header/CRT mask parameter mismatch must be resolved in Wafer CRT | LLVM call emitted；CRT/golden packet pending |
| `wafer.instr.reduce` | `#wafer.instr_reduce_kind` target wrapper families plus native `dim` code | per-kind LLVM call emitted；CRT/golden packet pending |
| `wafer.instr.convert` | `#wafer.instr_convert_kind` opcode 139..174 dtype pair plus zero-point/rounding/plain signature groups；same-format copy must lower through movement, not convert | per-kind LLVM call emitted；CRT/golden packet pending |
| `wafer.instr.gemm` | `TsmGemm` wrapper / `__Gemm` style CRT evidence; Wafer-owned symbol must not expose Tx81 ABI verbatim | LLVM call emitted；CRT/golden packet pending |
| `wafer.instr.conv` | `#wafer.instr_conv_kind` NE Conv/Depthwise/BackwardConv target wrapper families；optional/fused operands are not implicit | per-kind-family LLVM call emitted；CRT/golden packet pending |
| `wafer.instr.pool` / `unpool` | `#wafer.instr_pool_kind` / `#wafer.instr_unpool_kind` CT Pool/UnPool wrapper families; unpool scalar index is wrapper-aligned | per-kind LLVM call emitted；CRT/golden packet pending |
| `wafer.instr.tdma_data_move` | V0 production only covers pad/img2col wrappers; ordinary copy and transform-like movement use gather/scatter | pad/img2col LLVM call emitted；transform-like kind 结构化失败 |
| `wafer.instr.peripheral` | `#wafer.instr_peripheral_kind` for arg/factorize/bilinear/LUT/rand/elem_mask; count writeback and bitcount remain unsupported | per-kind LLVM call emitted；CRT/golden packet pending |
| `wafer.instr.dte_send` / `dte_recv` / `dte_wait` | Direct DTE/FSM runtime binding evidence still incomplete | LLVM call emitted for logical peer/bytes；runtime endpoint/channel binding pending |
| `wafer.instr.local_fence` | `TsmWaitfinish` / local drain evidence | LLVM call emitted；CRT/golden packet pending |

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
| transform-like TDMA group | no production Wafer symbol in V0 | mirror/transpose/rotate/NCHW-NHWC/TensorNom wrappers exist but are not default lowering surface | instr lowering must materialize these movements before target LLVM/package export; transpose uses `permutation`, mirror/rotate use `axes`; future target enablement requires board/golden coverage and an explicit target-surface update |
| `peripheral` arg/factorize/rand | `wafer_tx81_peripheral_*` | arity differs by kind | verifier fixes input/dest arity; arg index dest must be i32 |
| `peripheral` bilinear/LUT/elem_mask | separate Wafer symbols or descriptor variants | bilinear uses shape attrs; LUT uses `lut_elem_count`; elem_mask uses scale/probability/rounding | required/forbidden attrs already checked by IR/package validator |
| `peripheral count` | no production symbol | public wrapper writeback is not represented as normal dest buffer | verifier rejects until IR gains explicit writeback/result semantics |
| Direct DTE | `wafer_tx81_dte_send` / `wafer_tx81_dte_recv` / wait helper | Direct DTE/FSM helper, not `TsmExecute` | logical peer and bytes from IR; endpoint/channel binding from execution mesh/topology/runtime |
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

LLVM lowering 直接声明/调用 Wafer-owned target CRT symbol。当前 pass 在 LLVM dialect 中生成 vararg
声明和 `llvm.call`：

```mlir
llvm.func @wafer_tx81_gemm(...)
llvm.call @wafer_tx81_gemm(%lhs, %rhs, %dest, %m, %k, %n, %batch, %fmt)
    vararg(!llvm.func<void (...)>) : (...) -> ()
```

这里的 vararg 是当前 compiler-side field stream 的 LLVM 表达形态，用于避免在 IR lowering 层把
rank、optional field 和 kind-specific descriptor 过早固化成 C prototype。它不是 compiler-facing
万能 helper ABI：symbol 仍按 Wafer target action / kind family 拆分，参数仍从当前 IR、accepted offset
facts 和 verifier-checked attrs 派生。后续 Wafer CRT 头文件、implementation 和 golden packet test
可以把这些 field stream 固定成 typed wrapper 或 descriptor struct，但不能反向恢复旧 helper ABI。

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

## 7. Wafer CRT 全量实现设计

本节定义 `wafer_tx81_*` 的 repo-local Wafer CRT 实现方案。这里的“全量”只指
`LowerInstrToTargetLLVM.cpp` 当前可能 emit 的 Wafer-owned target CRT symbol 全量，不指硬件所有
opcode、所有 TSM wrapper 或未来未进入 `wafer.instr.*` coverage matrix 的能力。

### 7.1 Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  target LLVM call-emission 生成的 LLVM IR artifact，调用 `wafer_tx81_*` Wafer-owned target CRT symbols；
  accepted SPM/DDR offset facts、topology/execution-mesh、program parameter shard metadata/resource view
  已在 LLVM call 参数中 materialize。
- Current stage responsibility:
  在 repo-local Wafer CRT 中定义所有 compiler-emitted `wafer_tx81_*` symbols；每个 symbol 直接
  创建 public `Tsm*Instr` / Direct DTE helper 所需对象，调用 public TSM wrapper 或 Direct DTE/FSM
  helper，再执行 `TsmExecute` 或对应 DTE API。该层不调用 TX81/Triton `__*` ABI，不恢复 capture shim，
  不写空实现。
- Output artifact / IR:
  Wafer CRT RISC-V object / archive、kcore shared object、required-symbol report、可供 package
  metadata auto-export 记录的 module path。
- Downstream consumer:
  device-code compile/link symbol-closure gate、IR-derived package metadata、runtime adapter / board gate。
- User-level driver / named pipeline:
  `wafer-lower-groups-to-target-llvm` 或 `--wafer-lower-instr-to-target-llvm` 产出 LLVM IR；
  `tools/wafer_device_link.py` 编译 LLVM IR 和 Wafer CRT source/object，并链接 repo-vendored TX8 deps。
- Explicit non-goals:
  不把 `__Gemm`、`__AddVV` 等 TX81/Triton CRT symbol 改成 Wafer compiler ABI；不通过
  `--allow-shlib-undefined` 放过 Wafer-owned symbol；不新增 helper ABI dialect；不在 CRT 内重新做
  layout/search/SPM/DDR planning。
- Completion gate:
  当前 compiler 可能 emit 的所有 `wafer_tx81_*` symbols 都有 repo-local definition；final kcore `.so`
  中不得残留 undefined `wafer_tx81_*`；lit/ctest 覆盖成功闭合和缺失 symbol 失败两条路径。
```

### 7.2 文件和所有权

Wafer CRT 是 compiler target boundary，不放进 `third_party/wafer_crt/lib/libvr.a` 里作为 opaque
二进制补丁。源代码放在 repo-local runtime 目录，由 device link gate 显式编译：

| 文件 | 职责 |
| --- | --- |
| `runtime/wafer_crt/include/wafer_tx81_crt.h` | 唯一的 `wafer_tx81_*` C ABI prototype 定义；compiler lowering 和 CRT implementation 共同引用 |
| `runtime/wafer_crt/src/wafer_tx81_crt.c` | 定义所有 compiler-emitted `wafer_tx81_*` symbols；直接调用 `instr_adapter_plat.h` / `instr_adapter.h` / Direct DTE helper |
| `lib/Wafer/Transforms/Target/LowerInstrToTargetLLVM.cpp` | 使用同一 symbol/signature registry 生成 fixed LLVM function declaration；不再长期依赖 `void (...)` vararg |
| `tools/wafer_device_link.py` | 编译 Wafer CRT source/object，链接 target LLVM object、Wafer CRT object、repo-vendored TX8 deps，并做 required-symbol 检查 |
| `test/Tools/` | 覆盖 `.ll -> .o -> kcore .so`、Wafer CRT object link、undefined `wafer_tx81_*` failure |

`third_party/wafer_crt/lib/libvr.a` 仍可作为已有 TX8/Triton CRT evidence archive 或底层依赖参与链接，
但 Wafer-owned symbol 不从这个 archive 的 `__*` 名字继承 ABI，也不通过 alias/wrapper 调用 `__*`
完成主线发射。需要用到的底层能力应通过 public TSM wrapper 或 Direct DTE helper 调用。

### 7.3 ABI 规则

`wafer_tx81_*` 使用固定 C prototype。当前 target LLVM pass 生成 vararg declaration 只是 call-emission
bring-up 形态；symbol-closure gate 必须把它替换为 fixed function type。ABI 规则：

- 所有地址参数用 `uint64_t` byte address。
- 所有 element count、byte count、format、kind、dim、shape、stride、iteration 用 `uint32_t`。
- `format` 参数是 `Data_Format` 的整数编码。
- `stride` 参数是 byte stride；`iteration` 是 logical loop count，不是硬件字段中的 `iteration - 1`。
- rank-4 shape 在 ABI 中统一按 `n, h, w, c` 展开；weight shape、pad、unpad、stride、dilation 也按固定
  field 顺序展开，不传裸 pointer 到 compiler-owned temporary array。
- ordinary compute/move symbol 只 issue，不默认 wait；显式 drain 只在 `wafer_tx81_local_fence`、
  DTE wait 或后续 runtime-visible completion boundary 中发生。
- 每个 function 返回 `void`。错误检查属于 verifier / target lowering / device link gate；CRT 内部不通过
  silent return 表示 unsupported。

### 7.4 必须闭合的 symbol set

以下 symbol 是 current compiler-emitted set。实现完成前，任何列在这里的 symbol 都不能在 final `.so`
中保持 undefined。

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
wafer_tx81_dte_send
wafer_tx81_dte_recv
wafer_tx81_dte_wait
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
wafer_tx81_peripheral_count
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
| relation | 创建 `TsmRelationInstr`；`eq/ne/ge/gt/le/lt` 调 value/bool relation wrapper；bool storage 选择必须由 destination dtype/format 显式决定 |
| logic | 创建 `TsmLogicInstr`；`logic_not/and/or/xor` 调 value 或 bool wrapper；若当前 IR 无法区分 bitpacked bool 与 value bool，必须先固定 dtype/format rule |
| transcendental / activation | 创建 `TsmArithInstr` 或 `TsmActivationInstr`；`log2/ln/pow2/exp/exp_lp/sin/cos` 调 `TsmTranscendental`，`tanh/sigmoid/relu/satrelu/leakyrelu/softplus` 调 `TsmActivation` |
| `reduce_*` | 创建 `TsmReduceInstr` 和 `Data_Shape{n,h,w,c}`；按 kind 调 `ReduceSum/ReduceMax/ReduceMin/ReduceAvg`；`dim` 使用 native `0:C, 1:W, 2:H, 3:N, 4:HW, 5:HWC` |
| `convert_*` | 创建 `TsmConvertInstr`；按 dtype pair 调具体 `TsmConvert` wrapper；zero-point group 必须消费 `zero_point`，rounding group 必须消费 `rounding_mode`，plain group 禁止使用额外参数 |
| `bit2fp` | 使用 public bit-to-float / mask conversion wrapper；source bool storage 和 destination FP format 必须由 ABI 参数固定 |
| `mask_move` | 使用 `TsmMaskDataMove::MaskMove` 或等价 public wrapper；当前 `TsmMaskDataMove` mask 参数是 `uint32_t`，若 Wafer IR 提供的是 mask address，必须在 ABI 中明确 mask scalar/address policy，不能沿用歧义 |
| `gemm` | 创建 `TsmNeInstr`；调用 `TsmGemm::AddInput`、`ConfigMKN`、`ConfigBatch`、`SetTransflag`、`AddOutput`；bias/scale/activation/quant/psum disabled，除非 IR 后续显式扩展 |
| `conv` / `depthwise_conv` / `backward_conv` | 创建 `TsmNeInstr`；调用 `TsmConv` 或 `TsmDepthwiseConv` wrapper；只启用基础 NHWC/HWOI shape、pads/unpads/strides/dilations；bias/scale/sparse/quant/fused activation disabled |
| `pool_*` | 创建 `TsmPoolInstr`；按 kind 调 `AvgPool/SumPool/MaxPool/MinPool/IndexdMaxPool/IndexdMinPool`；indexed variant 必须有 value dest 和 index dest |
| `unpool_*` | 创建 `TsmUnPoolInstr`；按 kind 调 `Unpool/UnpoolAvg/UnpoolIdx`；scalar `index` 由 ABI 显式传入，不能用名字或 side table 恢复 |
| `tdma_pad` / `tdma_img2col` | 创建 `TsmDataMoveInstr`；调用 `Pad` / `Img2col`；source/dest shape、pad、kernel stride 由 fixed ABI field 提供 |
| `peripheral_*` | 创建 peripheral packet；按 kind 调 public wrapper；count/arg/factorize/bilinear/LUT/rand/elem_mask 的输入输出 arity 由 fixed ABI 和 verifier 共同保证 |
| `local_fence` | 只调用 `TsmWaitfinish` 或等价 local drain helper；不携带 multi-tile barrier 语义 |

### 7.6 DTE ABI

Direct DTE 不能通过 `TsmExecute` 发射。`direct_dte_send_async` 需要 `DirectDTESendInfo`，字段包括
`src_addr`、`dst_addr`、`length`、`remote_fsm_id`、`mode`、`dst_tile`、`tile_this`、
`stride_iterations[3]` 和 `dte_node`。因此当前 call-emission 只传
`buffer, peer, bytes` 的 shape 不足以作为完整 production DTE ABI。

全量 symbol closure 仍必须定义 `wafer_tx81_dte_send`、`wafer_tx81_dte_recv` 和
`wafer_tx81_dte_wait`，但实现前要先改 ABI / lowering：

- `wafer.instr.dte_*` lowering 必须从 topology/execution-mesh/runtime binding 派生 `tile_this`、
  `dst_tile`、`remote_fsm_id`、mode 和远端 receive buffer / FSM monitor binding。
- `wafer_tx81_dte_recv` 负责初始化或更新 local FSM monitor，使 peer send 有明确 remote destination。
- `wafer_tx81_dte_send` 负责 attach/config/send Direct DTE node，不允许假设 remote dst address 等于 local source。
- `wafer_tx81_dte_wait` 只等待由 send/recv 建立的 DTE/FSM completion，不做 local NCC drain。

如果这些字段尚未能从 IR/runtime binding materialize，DTE lowering 不能继续作为 production target path
进入 device-code symbol-closure gate；必须先扩 ABI 或扩 IR，而不是在 CRT 中写空成功函数。

### 7.7 Link 和 required-symbol gate

`tools/wafer_device_link.py` 的执行顺序：

```text
compiler-generated target LLVM IR
  -> LLVM clang++ .ll -> target object
  -> TX8 GCC compile runtime/wafer_crt/src/wafer_tx81_crt.c -> wafer CRT object
  -> repo-vendored GCC link target object + wafer CRT object + TX8 deps -> kcore .so
  -> readelf/nm required-symbol scan
```

required-symbol scan 的规则：

- final `.so` 中任何 undefined `wafer_tx81_*` 都是失败。
- `--allow-shlib-undefined` 只允许 runtime/loader 解析的非 Wafer-owned symbol。
- 测试必须覆盖一个 intentionally missing `wafer_tx81_missing`，证明链接器返回成功也会被 gate 拒绝。
- 成功测试必须从 compiler-generated LLVM IR 或 `wafer-lower-groups-to-target-llvm` output 进入 device link，
  不能只用手写 package metadata input。

### 7.8 Verification

CRT 全量实现的验证分四层：

| 层 | 验证 |
| --- | --- |
| Header/signature | generated or checked symbol registry 确认 `LowerInstrToTargetLLVM.cpp` emitted signature 与 `wafer_tx81_crt.h` 一致 |
| CRT unit/golden | 每个 wrapper family 至少有 packet field capture 或 public wrapper call trace；Conv/Pool/Peripheral 需要 kind-specific golden |
| Device link | `.ll -> .o -> kcore .so` 实际执行，final `.so` 无 undefined `wafer_tx81_*` |
| Pipeline integration | `wafer-lower-groups-to-target-llvm` output 能进入 device link；HF program-chain target LLVM integration 是下一层 gate，不用手写 package metadata 代替 |

完成后，`tasks/progress.md` 中 Q1/Q2/Q3 才能从 target CRT surface / implementation /
device-code required-symbol gate 推进到 Q4 package metadata auto-export。只实现 header、只生成 object、
或只靠 `--allow-shlib-undefined` 得到 `.so` 都不算完成。

## 8. 当前缺口

- Wafer-owned target CRT implementation / typed wrapper contract 仍需逐个 op 对齐 repo-local
  TX81/public TSM wrapper 和硬件文档；当前 LLVM lowering 只生成 Wafer-owned symbol declarations/calls。
- Golden packet tests 仍待补，用来验证 Wafer CRT 参数到 public TSM wrapper/register packet 的映射。
- Device-code compile/link helper 已能对已有 / compiler-generated LLVM IR 执行 `.ll -> .o -> kcore .so`
  和 object metadata normalization；任务队列中的 Q1/Q2/Q3 仍需补 symbol surface audit、
  repo-local Wafer CRT implementation / golden coverage 和 required-symbol closure，不能让
  `wafer_tx81_*` 以未解释 undefined symbol 形式残留，也不能经过 capture shim。
- Direct DTE production lowering 还缺 runtime endpoint / DTE channel binding。
- package auto-export 当前只能消费已有 LLVM IR；恢复 compiler-generated package gate 要等 device-code
  compile/link gate 和 resource metadata export 衔接完成。
