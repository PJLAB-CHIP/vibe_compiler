# Wafer Instruction IR Design

状态：设计草案；范围：instruction-level Wafer hardware invocation IR、memref-backed buffer contract 和
instruction legalization。

本文定义 instruction-level Wafer IR。核心结论：

- `wafer.instr.*` 是当前 compiler pipeline 需要的 target-aligned instruction subset，不是完整
  硬件 ISA、Tsm wrapper 或 opcode 全量镜像。每个进入该层的 op 必须能被 verifier 解释，并且在
  target LLVM lowering 中要么 lower 到明确 target CRT / DTE helper，要么结构化失败。
- 只新增 `wafer.instr.*` 硬件相关调用级 op，包括 CT/NE/RDMA/WDMA/TDMA 和 Direct DTE 的 V0 子集。
- `wafer.instr.*` 不再直接复用 tile 层 `Compute*Kind`。tile 层的
  `#wafer.elementwise_kind` / `#wafer.reduce_kind` 表示 target-abstract compute semantics；
  instruction 层使用 `#wafer.instr_elementwise_kind`、`#wafer.instr_reduce_kind` 和
  `#wafer.instr_convert_kind` 表示已经选到硬件 wrapper / opcode family 的 target kind。
- instruction-level buffer value 统一使用 MLIR `memref`，不再把 `!wafer.storage` 作为长期 IR
  合同。
- Wafer 的 SPM / DDR address domain 和 physical layout marker 放在 memref memory-space attr 中，
  例如 `memref<2x65xf16, #wafer.memory<spm, cx>>`。
- `Cx/NCx` 只是 Wafer physical layout marker；`C0`、storage bytes、range-end、bool bitpack 和
  256B padding 必须由统一 Wafer layout calculator 从 memref type 推导，不写进 IR 字段。
- `Cx/NCx` 不使用 MLIR memref layout slot，也不实现为 `MemRefLayoutAttrInterface`。MLIR memref
  layout slot 仍只用于 MLIR 能按 affine / strided 语义解释的普通 layout。
- 不引入 `wafer.physical_view`、`!wafer.physical_memref`、side descriptor value、SPM offset、DDR
  DDR planning result、raw packet 或 target CRT call。

`wafer.instr` 的作用是把 target-abstract tile-region op 变成可执行硬件动作或硬件通信调用，并让下游能从
memref SSA、Wafer memory attr、op operands、attrs、MemoryEffects 和显式 fence 直接推导 endpoint/resource
输入。它不是另一层 buffer IR。

实现边界：

- 仓库代码当前已落地 `wafer.instr.local_fence`，以及
  `wafer.instr.rdma`、`wafer.instr.wdma`、`wafer.instr.gather_scatter`、`wafer.instr.fill`、
  `wafer.instr.elementwise`、`wafer.instr.bit2fp`、`wafer.instr.mask_move`、
  `wafer.instr.reduce`、`wafer.instr.convert`、`wafer.instr.gemm`、`wafer.instr.conv`、
  `wafer.instr.pool`、`wafer.instr.unpool`、`wafer.instr.tdma_data_move`、
  `wafer.instr.peripheral`
  和 `wafer.instr.dte_send` / `dte_recv` / `dte_wait`
  的 ODS、verifier、MemoryEffects、`WaferInstructionOpInterface` 和 lit/unit 覆盖。
  `wafer.instr.elementwise` 当前承载有 CT elementwise wrapper 证据的 unary/binary arithmetic、
  relation、logic、activation 和 transcendental target kind；`select` 不存在于
  `#wafer.instr_elementwise_kind`。instruction lowering 会把 floating select 改写成 false-copy
  `wafer.instr.gather_scatter` + `wafer.instr.bit2fp` + `wafer.instr.mask_move`。
  `wafer.instr.convert` 使用 opcode-aligned `#wafer.instr_convert_kind<...>`，覆盖硬件
  convert opcode 139..174 的 dtype pair；INT8->FP kind 携带 `zero_point`，需要 rounding 的 kind
  携带 `rounding_mode`，plain wrapper kind 不允许带这两类 attr。它不是通过 `src_dtype` /
  `dst_dtype` 表达任意转换。
- `wafer.instr.*` op 只读写 Wafer-tagged memref，不产生 buffer result，不携带 SPM offset、
  worker id、raw packet field 或 compiler-facing ABI 字段。
- Direct DTE instruction ops 已替代旧 tile-level p2p prototype，并在 SPM memory planning 前暴露
  buffer lifetime、peer、byte count 和 async token。all-gather 的 strided gather slot 通过
  `wafer.instr.gather_scatter` 与连续 communication buffer 互相 materialize；DTE op 本身只收发
  连续 SPM buffer。
- 当前实现已支持 target-abstract tile-region op 到这些 instruction op 的
  DialectConversion；静态 `extract_slice`、`insert_slice`、`broadcast` 和 `transpose`
  通过统一 logical-to-physical offset calculator 生成 logical movement segments，并尽量打包成
  三层 stride/iteration `wafer.instr.gather_scatter` descriptor。
- tile-region lowering 已产出 memref-backed `wafer.tile.region`，candidate materialization 已为
  explicit static boundary slice 和 candidate output tile offsets/sizes 接入 DDR `memref.subview` producer；
  instruction lowering 必须基于该 unplaced Wafer-tagged memref graph 做转换，不能再引入
  storage/buffer IR 层。
- RDMA/WDMA lowering 可以消费 DDR `memref.subview` / strided memref view，但不会从
  group tiling plan 自己生成这些 view。closed-loop planner 后续产生的 candidate tile 仍必须先由
  candidate/accepted materialization 显式变成 DDR subview。
- tile-region 已支持 `scf.if` / `scf.for` 作为 tile-region 内 structured control-flow。instruction lowering 必须递归
  legalize 这些 region body 内的 executable target-abstract op，并保留 `scf` container；是否选择
  硬件 branch/loop、predication 或 unroll 不是 instruction-level IR 的当前职责。

本文依赖：

- `tasks/07-tile-region.md`
- `tasks/10-compute-movement.md`
- `tasks/09-spm-memory-planning.md`
- `docs/wafer-hardware-instruction-set-and-programming-model.md`
- `docs/wafer-register-level-instruction-spec.md`

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  `wafer.tile.region` IR，内部包含带 Wafer memory attr 的 memref values、
  `memref.alloc` / `memref.subview` / verifier-legal metadata view、
  `wafer.tile.load` / `wafer.tile.store`、
  `wafer.tile.materialize_layout`、`wafer.tile.fill/gemm/elementwise/reduce`、
  `wafer.tile.copy/extract_slice/insert_slice/transpose/broadcast`、
  `wafer.tile.*` buffer-level collective、tile-region 内 `scf.if` / `scf.for`
  structured control-flow 和 `wafer.instr.local_fence`。
- Current stage responsibility:
  只做 Wafer instruction legalization / selection：把可执行的 target-abstract op 改写成
  `wafer.instr.*`，把 tile-level `Compute*Kind` 选择成 instr-level target kind，并保留 memref SSA
  graph。对 `scf.if` / `scf.for` 只递归转换其 region body，
  不改变 control-flow 结构。对 accepted `wafer.tile.*` collective，生成 explicit
  `wafer.instr.dte_send` / `dte_recv` / `dte_wait` p2p schedule。instruction op 通过
  interface 显式暴露 instruction family、memref read/write、descriptor attrs 和 effect。
- Output artifact / IR:
  同一个 `wafer.tile.region` execution scope 内的 instruction-level IR：
  memref values with `#wafer.memory<space, layout>` + `wafer.instr.*` +
  `wafer.instr.local_fence`，或结构化 legalization failure reason。
- Downstream consumer:
  SPM memory planning、DDR memory planning、closed-loop candidate driver、
  target instruction LLVM lowering、package metadata 和 runtime adapter。
- User-level driver / named pipeline:
  主线由 closed-loop planner 调用；局部 bring-up / candidate evaluation 入口是
  `wafer-lower-tile-region-to-instr` 和 `wafer-lower-groups-to-instr` named pipeline。
  candidate evaluation 调用 instruction lowering 时，tiled DDR load/store operand 必须已经由 candidate 或 accepted
  materialization 表达成 tile view；如果仍是 whole-boundary memref，instruction lowering 只能生成 whole-boundary
  descriptor。
  `--wafer-convert-tile-region-to-instr` 只作为 lit/debug pass 入口。
- Explicit non-goals:
  不新增第二套 storage/buffer IR，不决定 group boundary、tile shape、layout assignment、SPM offset、
  DDR planning result、raw register packet field、DTE/FSM resource id、Tsm wrapper call、target CRT
  symbol 或 launch ABI。SCALAR 仍是 reserved/stub；CSR helper/sync 若进入主线，必须作为明确
  instruction/sync family 另行定义，不能混入 CT/NE/RDMA/WDMA/TDMA 或 DTE op。
- Completion gate:
  对 tile-region 已支持的 load/store、静态可证明 layout materialize、fill、GEMM、
  elementwise/relation、reduce、copy 和 metadata view 生成 verifier-legal instruction-level IR
  或标准 memref view，并覆盖 nested `scf.if` / `scf.for` body 递归转换。unsupported hardware
  instruction form，包括当前无法证明的 slice/broadcast/transpose descriptor，必须结构化失败，
  不能让 SPM memory planning 从 target-abstract op 猜 demand。
```

## 2. Wafer MemRef Contract

Wafer instruction-level IR 的 buffer value 都是 `memref`：

```mlir
memref<128x64xf16, #wafer.memory<spm, tensor>>
memref<128x64xf16, #wafer.memory<spm, cx>>
memref<2x3x5x65xf16, #wafer.memory<spm, ncx>>
memref<128x64xf16, #wafer.memory<ddr, tensor>>
```

`#wafer.memory<space, layout>` 是放在 memref memory-space slot 的 Wafer target attr：

| field | values | meaning |
| --- | --- | --- |
| `space` | `spm`, `ddr` | addressable storage domain |
| `layout` | `tensor`, `ntensor`, `cx`, `ncx` | Wafer physical layout family marker |

memref shape 仍是 logical shape，element type 仍是 logical element type。对
`#wafer.memory<spm, cx>` 和 `#wafer.memory<spm, ncx>` 来说，SPM footprint 不等于
`product(shape) * sizeof(element)`；必须调用统一解析入口：

```text
computeWaferPhysicalTensorInfo(memrefType)
```

该入口从 memref type 推导：

- address space。
- physical layout family。
- dtype storage size 和 bool bitpack policy。
- `Cx/NCx` 的 block、`C0` tail/fold、aligned C。
- 256B line/layout padding 和 layout footprint。
- begin/end range、wrapper layout enum、instruction operand legality。

所有 verifier、SPM memory planning、DDR memory planning 和 instruction lowering 都必须使用这一个
入口。禁止每个 pass 自己写 `if layout == cx` 的局部解析。

### 2.1 Why Not MemRef Layout Slot

MLIR memref layout slot 表达的是 MLIR 可解释的 affine / strided address layout。Wafer
`Cx/NCx` 的 full-block 物理顺序是 channel-block major，且 retained tail 是 compact `C0`
span：

```text
full C blocks:
  Cx  : [CxBlock][outer][lane]
  NCx : [N][CxBlock][HW][lane]
tail block:
  compact C0 span after full blocks inside each Cx/NCx batch
```

因此 `aligned_C` 不能解释成 logical outer/HW row 的 dense stride。对 full block 中的
`c = cb * B + lane`：

```text
Cx  offset = cb * outer * B + outer_idx * B + lane
NCx offset = n * batch_num + cb * hw * B + hw_idx * B + lane
```

这不是普通 `strided<[aligned_C, 1]>`，也不是一个单一 affine map 能完整表达的 layout。因此
`Cx/NCx` 不放进 memref layout slot，不实现为 `MemRefLayoutAttrInterface`。普通 compact tensor、
metadata-only reshape 或标准 strided view 可以继续使用 MLIR memref layout / view 机制，但
Wafer `Cx/NCx` physical interpretation 只由 `#wafer.memory<..., cx/ncx>` marker 和
`computeWaferPhysicalTensorInfo` 解释。

`wafer.tile.reshape` 先表达 StableHLO / tensor 层的 logical linear-order reindex：source 和
result 的同一个 canonical linear element number 对齐，但 result multi-index 按新 shape
重新解释，因此它不是“无语义 no-op”。R3.2d lowering 再判断这个 logical reindex 能否由当前
physical storage alias 表达。compact `tensor/ntensor` 的物理字节序已经等于 canonical linear
order，所以可用 `memref.reinterpret_cast`；`Cx/NCx` 或其它 physical interpretation 改变时，必须按
同一 linear element number 分别计算 source/result physical byte offset，必要时 materialize
成 `wafer.instr.gather_scatter`。

R3.2d 不能把 `Cx/NCx` layout marker 本身当作 movement trigger。对 reshape 来说，是否需要
instruction movement 取决于同一 canonical linear element number 在 source/result 中的
physical byte offset 映射是否变化，以及 result 是否需要新的 materialized physical footprint；
不是取决于 op 名字、layout marker 或 `physicalBytes` 是否相等。`computeWaferPhysicalTensorInfo` 必须按硬件文档的 `get_CxC0` /
`common_tensor_info_generate_i64` 口径实现 INT8/UINT8 block 128、其它 dtype block 64、tail
retain/fold、tail align 和 256B bank padding。若该 helper 不能给出真实 mapping，R3.2d lowering
不能用局部 `ceil(C/64)` 近似来证明 reshape identity 或 descriptor 合法性。

### 2.2 Generic MemRef Op Boundary

带 Wafer memory attr 的 memref 是标准 SSA buffer value，但不是任意 generic memref op 都能在
instruction-level IR 中解释它：

- 允许 `memref.alloc` / ownership-preserving aliases 表达 allocation、lifetime 和 value identity。
- 允许 `memref.dim` 读取 logical shape。
- 对 `#wafer.memory<spm, tensor/ntensor>` 和 `#wafer.memory<ddr, tensor/ntensor>`，可在 verifier
  证明 metadata view 与 logical layout 一致时使用 `memref.cast` / `memref.reinterpret_cast` /
  `memref.subview`。对 DDR 边界，`memref.subview` / strided memref layout 是 tile load/store 的
  显式 slice/stride fact，R3.2d lowering 必须消费它生成 RDMA/WDMA descriptor。
- 对 `#wafer.memory<spm, cx/ncx>`，不能用 generic `memref.load/store/copy/subview` 伪装硬件
  physical indexing；真实 layout conversion、slice movement 和 copy 必须通过
  `wafer.tile.materialize_layout` 或 `wafer.instr.gather_scatter` 等 Wafer op 表达。
- 在 Wafer resource projection / realization 前，不能让 generic memref-to-LLVM lowering 按 dense memref
  footprint 处理 `#wafer.memory<spm, cx/ncx>`。

## 3. IR Boundary

R3.2d 前：

```text
memref values with #wafer.memory<space, layout>
memref.alloc / verifier-legal metadata views
wafer.tile.load / wafer.tile.store
wafer.tile.materialize_layout
wafer.tile.fill/gemm/elementwise/reduce
wafer.tile.copy/extract_slice/insert_slice/transpose/broadcast
scf.if / scf.for
wafer.instr.local_fence
```

R3.2d 后：

```text
memref values with #wafer.memory<space, layout>
memref.alloc / verifier-legal metadata views
wafer.instr.rdma / wafer.instr.wdma
wafer.instr.gather_scatter
wafer.instr.{fill, elementwise, bit2fp, mask_move, reduce, convert}
wafer.instr.gemm / wafer.instr.conv
wafer.instr.pool / wafer.instr.unpool
wafer.instr.tdma_data_move
wafer.instr.peripheral
scf.if / scf.for
wafer.instr.local_fence
```

R3.2d 不做 memref type conversion。它只把 executable target-abstract op 改写成 instruction op，
并复用同一批 memref values。physical base address、SPM offset、end address、bank/color、
worker register window、runtime pointer 和 packet word 都不属于 R3.2d。
R3.2d 后的 instruction IR 不允许 `#wafer.elementwise_kind` / `#wafer.reduce_kind`
这类 tile-level semantic attr 出现在 `wafer.instr.*` op 上；这些语义必须在 lowering 时选择成
instr-level target kind。

## 4. Instruction Ops And Families

`wafer.instr.*` 的 op mnemonic 表达指令语义，不把 CT/NE/TDMA/DTE 这类硬件 family 做成
额外 namespace。Instruction family 由 `WaferInstructionOpInterface` 派生；固定 family 的 op 不打印
冗余 attr。只有当同一个 instruction op 在相同 operand/result/attr contract 下确实能合法选择
多个 instruction family 时，才允许引入显式 `instruction_family` attr，并由 verifier 保证取值和 op contract
一致。

| instruction family | V0 op | 来源 | 说明 |
| --- | --- | --- | --- |
| RDMA | `wafer.instr.rdma` | `wafer.tile.load` | DDR memref -> SPM memref |
| WDMA | `wafer.instr.wdma` | `wafer.tile.store` | SPM memref -> DDR memref |
| TDMA | `wafer.instr.gather_scatter` | `wafer.tile.materialize_layout`、tile movement ops | byte-counted SPM movement；contiguous copy 是 descriptor 特例 |
| CT | `wafer.instr.fill` | `wafer.tile.fill` | scalar/immediate fill |
| CT | `wafer.instr.elementwise` | `wafer.tile.elementwise` | `#wafer.instr_elementwise_kind` target kind；不含 select |
| CT | `wafer.instr.bit2fp` | tile semantic select lowering | i1 mask -> floating mask target peripheral op |
| TDMA | `wafer.instr.mask_move` | tile semantic select lowering | masked SPM data movement target op |
| CT | `wafer.instr.reduce` | `wafer.tile.reduce` | `#wafer.instr_reduce_kind` target kind + native reduce `dim` code |
| CT | `wafer.instr.convert` | future convert lowering | `#wafer.instr_convert_kind` opcode-aligned dtype pair + kind-specific wrapper attrs |
| NE | `wafer.instr.gemm` | `wafer.tile.gemm` | tile-local GEMM / batched GEMM |
| NE | `wafer.instr.conv` | future conv lowering / imported target op | basic Conv / Depthwise / BackwardConv packet fields |
| CT | `wafer.instr.pool` / `wafer.instr.unpool` | future pool/unpool lowering / imported target op | pool indexed-output arity and unpool scalar-index descriptor |
| TDMA | `wafer.instr.tdma_data_move` | future structured data-move lowering / imported target op | V0 production target 只允许 pad / img2col；mirror / transpose / rotate / NCHW-NHWC / TensorNom 这类 transform movement 如果以 imported target op 进入 instr IR，必须由 instr lowering 在 target LLVM / package export 前 materialize 成 gather_scatter，或在后续板端验证后再开启 target path |
| CT | `wafer.instr.peripheral` | future peripheral lowering / imported target op | arg / factorize / bilinear / LUT / random / element-mask target kind; count remains rejected until writeback is represented |
| DTE | `wafer.instr.dte_send` / `dte_recv` / `dte_wait` | accepted `wafer.tile.*` collective p2p schedule | fixed-size unicast Direct DTE invocation over unplaced SPM memrefs |

### 4.1 Instruction Coverage Matrix

`wafer.instr` coverage 按 compiler IR 合同分层，而不是按硬件 opcode 数量分层。target LLVM lowering
只能把 **V0 production target surface** 当作必须支持的 production lowering 输入；其它类别不能隐式进入现有
泛 op 或 lowering fallback。

| 硬件 / wrapper 能力 | 当前 `wafer.instr` 表示 | coverage tier | 处理规则 |
| --- | --- | --- | --- |
| RDMA / WDMA contiguous 和三层 stride descriptor | `wafer.instr.rdma` / `wafer.instr.wdma` | V0 production target op | target LLVM lowering 必须生成 target CRT call；descriptor 保持 byte-level `inner_bytes`、stride 和 iteration |
| TDMA `TsmDataMove::GatherScatter` | `wafer.instr.gather_scatter` | V0 production target op | layout materialization、SPM copy 和可静态证明的 slice/transpose/broadcast movement 都展开为一条或多条 gather/scatter；无法压成 V0 descriptor 时结构化失败 |
| `TsmPeripheral::Memset` / scalar fill | `wafer.instr.fill` | V0 production target op | 仅表达填充 destination SPM memref；不把 peripheral writeback/count 语义塞进 fill |
| CT arithmetic / relation / logic / activation / selected transcendental | `wafer.instr.elementwise` + `#wafer.instr_elementwise_kind` | V0 production target op | 覆盖当前 enum 中的 target kind；broadcast、scalar immediate、bitpacked bool loop/VuV 和 rounding mode 是 target lowering 内部选择或后续扩展，不改变该 IR 合同 |
| semantic select | 无单条 select op | V0 composite lowering | 必须展开为 false-copy `gather_scatter` + `bit2fp` + `mask_move`；`wafer.instr.elementwise <select>` 非法 |
| CT reduce `sum/avg/max/min` | `wafer.instr.reduce` + `#wafer.instr_reduce_kind` + target `dim` code | V0 production target op | tile 层 reduce `dimensions` 在 instruction lowering 中 materialize 为 native `TsmReduce` dim code：`0:C`、`1:W`、`2:H`、`3:N`、`4:HW`、`5:HWC`；aligned physical layout、rank 和 dtype 合法性由 verifier / target lowering 检查 |
| CT convert opcode 139..174 | `wafer.instr.convert` + `#wafer.instr_convert_kind<src_dst>` + kind-specific attrs | V0 production target op | dtype pair 由 kind 唯一决定；INT8->FP 要求 `zero_point`，rounding wrapper 要求 `rounding_mode`，plain wrapper 不允许额外转换参数；same-format copy 必须走 movement，不允许伪造成 convert |
| NE GEMM | `wafer.instr.gemm` | V0 production target op | 只表达 GEMM / batched GEMM 主路径参数；bias、scale、quant、fused activation 和复杂 psum policy 不能被隐式打开 |
| Direct DTE fixed-size unicast | `wafer.instr.dte_send` / `dte_recv` / `dte_wait` | V0 production target op；LLVM call emitted；production binding pending | IR 表达 logical peer、bytes 和 async token；target LLVM lowering 生成 Wafer CRT send/recv/wait call，runtime endpoint/channel binding 仍由后续 device/runtime gate 固定 |
| local NCC drain / visibility fence | `wafer.instr.local_fence` | V0 production target sync；LLVM call emitted | target LLVM lowering 映射到 local wait/drain Wafer CRT call；不是 multi-tile barrier |
| SPM memcpy helper / copy | 无单独 copy op | V0 composite lowering | copy 是 `gather_scatter` 的 descriptor 特例；不引入 `wafer.instr.copy` |
| ChannelNorm / DechannelNorm / Tensor-Normalization | 无单条 op | V0 composite lowering | 作为 layout materialization algorithm 展开为 gather/scatter 序列；native TensorNom opcode 133 不作为 V0 主路径 |
| tile collectives `all_gather/all_reduce/reduce_scatter/all_to_all/collective_permute` | 无 collective instr op | V0 composite lowering | 先在 tile collective / schedule 层选 ring、tree 或 direct p2p，再 lower 成 DTE send/recv/wait + local movement/compute |
| `TsmConv` / `TsmDepthwiseConv` / backward conv | `wafer.instr.conv` + `#wafer.instr_conv_kind` | V0 production target op；LLVM call emitted | IR 覆盖基础 Conv/Depthwise/BackwardConv packet fields；bias、scale、sparse、INT8 quant、fused activation 和 psum policy 仍需后续扩展；CRT/golden packet 仍待补 |
| `TsmPool` / `TsmUnPool` | `wafer.instr.pool` / `wafer.instr.unpool` | V0 production target op；LLVM call emitted | 明确 NHWC descriptor、pad/kernel/stride 和 indexed output arity；不能复用 reduce 或 movement op 表达；CRT/golden packet 仍待补 |
| TDMA pad / img2col | `wafer.instr.tdma_data_move` + `#wafer.instr_data_move_kind` | V0 production target op；LLVM call emitted | 单条 TDMA data-move op 表达 wrapper-level pad/img2col；普通 copy/layout segment 仍优先使用 `gather_scatter`；transform-like kind 到 target LLVM 必须结构化失败 |
| TDMA mirror / transpose / rotate / NCHW-NHWC / TensorNom | 无 production target op；enum 保留用于 imported/pre-lowering IR | V0 composite lowering or future target extension | V0 由 compiler lowering 展开为 `gather_scatter` 或结构化失败。`transpose` 使用 `permutation`，`mirror` 使用一个或多个 `axes`，`rotate90/180/270` 使用有序二元 `axes` 表示旋转平面，NCHW/NHWC 使用固定 4D layout permutation，`tensor_nom` 使用 logical-linear 到 physical-layout materialization。package / target export 如果仍看到这些 kind，说明 pipeline 漏了 materialization，应拒绝 |
| TDMA concat / maskgather variants | 无单独 op | future target extension or composite | `concat` 属于 CT packet，maskgather variants 需要 bool/index operand policy；未定义前不能复用 `tdma_data_move` |
| bitpacked bool loop / VuV / scalar immediate variants | 当前 `elementwise` 只覆盖 opcode family kind，不单独建模 variant | future extension | 需要区分 value bool、bitpacked bool、VuV/VuVLoop、scalar immediate 和 output storage；不能靠 `elementwise` 名称吞掉所有 variant |
| Peripheral count/argmax/argmin/factorize/bilinear/lut/rand/elem_mask | `wafer.instr.peripheral` + `#wafer.instr_peripheral_kind` | V0 production target op；LLVM call emitted | IR 明确 kind、input/output arity 和 elem_count；bitcount opcode 176 仍因缺 public wrapper 不纳入；CRT/golden packet 仍待补 |
| raw DTE non-unicast / stream / mailbox | 无 | future communication ABI | 需要独立 communication ABI 和板端验证；V0 collective 不直接生成 raw non-unicast DTE packet |
| SCALAR / CSR ordinary execution | 无 | not compiler instr IR in V0 | `TsmExecute` 普通 dispatch 不覆盖 SCALAR/CSR；CSR wait/sync 只能通过明确 sync/runtime ABI 进入 |

V0 不定义 `wafer.instr.copy`。公开 SPM memcpy helper 本身也是
`TsmDataMove::GatherScatter` 样例；把 copy 单独做成 instruction op 会把 helper 名字提升为 IR
语义。

`ChannelNorm/DechannelNorm` 也不是 V0 单条 instruction op。它们是 layout materialization algorithm；
R3.2d 要么展开成一条或多条 `wafer.instr.gather_scatter`，要么结构化失败。对于
`C > block` 且存在 retained `C0` tail 的 `Cx/NCx`，full C blocks 和 compact tail block 的
inner width / stride 不同，lowering 通常需要至少两段 GatherScatter：一段搬 full blocks，一段搬
tail `C0`。如果 full-block 段和 tail 段都无法分别表示为 V0 三层 stride/iteration descriptor，
R3.2d 必须失败。

`TsmExecute` 普通 dispatch path 只覆盖 CT/NE/RDMA/WDMA/TDMA。DTE 不走这条 dispatch path，
但仍属于 `wafer.instr.*` 的硬件通信调用层；后续 target LLVM lowering 负责把 `wafer.instr.dte_*`
映射到 Direct DTE/FSM helper、runtime-compatible wrapper 或 raw-DTE ABI。SCALAR 当前 reserved/stub；
CSR/sync helper 不在 V0 ordinary compute path 中。

## 5. Operand And Result Model

instruction op 直接读写 memref，但 **instruction op 本身不产生 buffer result**。
凡是 target-abstract op 原来返回 buffer 的地方，R3.2d 先确保存在 destination `memref.alloc`
或 verifier-legal destination memref，再生成写入该 memref 的 instruction op，并用 destination
memref 替换原 op result 的 uses。

这个模型避免把指令 issue 和 buffer identity 混在一起：

- source memref 是 instruction operand。
- destination memref 也是 instruction operand。
- result/temp/psum/staging buffer 由 `memref.alloc` 或 accepted alias/view 创建。
- metadata-only reshape 由 verifier-legal memref view 表达；physical layout conversion 必须是
  explicit movement。
- SPM offset、range、bank span 由 R3.2f 写入；后续 runtime/target-codegen 阶段必须从这些 facts
  和当前 memref use-def/view relation 派生 address/range 参数，不能复制成独立 placed/access
  descriptor 中间协议。
- RDMA/WDMA 的 DDR side 使用 `memref<..., #wafer.memory<ddr, layout>>`；DDR memory planning stage 负责
  external allocation contract、default arena resource、compiler-managed/resident requirement、planned DDR
  ranges 和 constant residency。

R3.2d 只 materialize **unplaced logical descriptor facts**：byte count、stride/iteration、op kind、
tile reduce dimensions 到 native reduce `dim` code、GEMM M/K/N、batched GEMM `batch_count` 和 batch/m/n/k dimension attrs、
elementwise kind 等。这些字段能从当前 IR type、attrs 和
source op verifier 重算。

## 6. Common Instruction Interface

所有 `wafer.instr.*` op 必须实现同一个窄 interface：

```text
WaferInstructionOpInterface {
  getInstructionFamily() -> InstrFamily
  collectInstructionEffects(...) -> memref read/write + hardware invocation effect
  verifyInstructionContract()
}
```

`InstrFamily` 是 Wafer enum/interface fact，V0 至少包含：

| enum | hardware invocation family |
| --- | --- |
| `ct` | CT / CGRA queue |
| `ne` | NE queue |
| `rdma` | RDMA queue |
| `wdma` | WDMA queue |
| `tdma` | TDMA queue |
| `dte` | Direct DTE / FSM communication invocation |

interface 返回的是 op-local facts，不返回 planner side table，也不复制全局 schedule。对
`rdma`、`wdma`、`gather_scatter`、`fill`、`elementwise`、`reduce`、`convert`、`gemm` 和
`dte_*` 这类固定 family 的 V0 op，`getInstructionFamily()` 由 op class 静态派生，不要求 IR
打印 `instruction_family` attr。后续若出现同一个 op contract 下可选择多个 family 的 instruction
op，才在该 op 上增加显式 attr，并把合法取值纳入 verifier。

resource effects 至少要表达：

| family | buffer effects | invocation effect |
| --- | --- | --- |
| RDMA | DDR memref read + SPM memref write | Movement/RDMA issue |
| WDMA | SPM memref read + DDR memref write | Movement/WDMA issue |
| TDMA | SPM memref read + SPM memref write | Movement/TDMA issue |
| CT | SPM memref read/write as operand contract requires | Compute/CT issue |
| NE | SPM memref read + SPM memref write | Compute/NE issue |
| DTE | send reads SPM source, recv writes SPM destination, wait consumes async token | Communication/DTE issue or wait |

R3.2d 不建模 worker id。`TsmExecute` 的 worker bits、register window 和 packet field 属于
committed instruction 后的 target LLVM lowering。

## 7. ODS-Level Op Contracts

本节是实现时的 ODS 合同。assembly format 可以按 MLIR 可读性微调，但 operand/result/attr
语义不能变。

### 7.1 DMA Descriptor Attributes

RDMA、WDMA 和 TDMA 共同使用 fixed-rank descriptor attrs。字段是 logical descriptor，不是
physical packet：

| attr | type | meaning |
| --- | --- | --- |
| `byte_count` | `I64Attr` | 该 instruction 的 total logical payload bytes，用于 resource effect |
| `inner_bytes` | `I64Attr` | 最内层 contiguous byte count |
| `src_strides` | `DenseI64ArrayAttr` | source byte strides，长度为 3 |
| `src_iterations` | `DenseI64ArrayAttr` | source logical iterations，长度为 3，值为正数 |
| `dst_strides` | `DenseI64ArrayAttr` | destination byte strides，长度为 3 |
| `dst_iterations` | `DenseI64ArrayAttr` | destination logical iterations，长度为 3，值为正数 |

contiguous movement 使用 `inner_bytes == byte_count`，stride 全 0，iteration 全 1。byte stride
必须已经从 element stride 转换完成。硬件 RDMA/WDMA 和 TDMA 都是“最内层连续搬运 +
三层 byte stride/logical iteration”的 descriptor 模型；RDMA/WDMA packetization 时最内层字段会按
dtype element count 写入，R3.2d IR 仍用 byte-level `inner_bytes` / `byte_count` 作为统一
resource/legality 合同。descriptor 表达不了的 dynamic stride、超过 3 层的静态 stride 或不规则
非连续访问，R3.2d 必须结构化失败，不能生成名字上合法但下游无法 packetize 的 instruction op。

### 7.2 RDMA / WDMA

```text
wafer.instr.rdma source to dest attr-dict : type(source) to type(dest)
wafer.instr.wdma source to dest attr-dict : type(source) to type(dest)
```

| op | operands | result | required attrs |
| --- | --- | --- | --- |
| `wafer.instr.rdma` | `source: MemRef<#wafer.memory<ddr, *>>`, `dest: MemRef<#wafer.memory<spm, *>>` | none | `byte_count`, `inner_bytes`, `src_strides`, `src_iterations` |
| `wafer.instr.wdma` | `source: MemRef<#wafer.memory<spm, *>>`, `dest: MemRef<#wafer.memory<ddr, *>>` | none | `byte_count`, `inner_bytes`, `dst_strides`, `dst_iterations` |

R3.2c 的 `wafer.tile.load/store` SPM 侧必须是 compact tensor layout；如果 consumer 需要 `Cx/NCx`，
必须通过 `wafer.tile.materialize_layout` 再 lower 到 TDMA，而不是让 RDMA/WDMA 隐式承担 layout
conversion。DDR 侧可以是 compact boundary，也可以是 `memref.subview` / strided memref view；
R3.2d 从 DDR memref layout 中恢复静态 element stride，转成 byte stride/iteration descriptor。
R3.2d 不负责把 whole-boundary DDR memref 按 tile shape 切成 subview；该事实必须由 R3.2e
explicit static boundary slice producer、planner candidate evaluation 或 accepted materialization 通过 IR view
显式提供。
动态 view、负 stride、bit-packed element、超过三层 stride/iteration 或不能静态证明 descriptor 的
情况必须 structured failure，不能从 memref 名字或 shape 猜测。

### 7.3 GatherScatter

```text
wafer.instr.gather_scatter source to dest attr-dict
    : type(source) to type(dest)
```

| op | operands | result | required attrs |
| --- | --- | --- | --- |
| `wafer.instr.gather_scatter` | `source: MemRef<#wafer.memory<spm, *>>`, `dest: MemRef<#wafer.memory<spm, *>>` | none | `byte_count`, `inner_bytes`, optional `src_offset` / `dst_offset`, `src_strides`, `src_iterations`, `dst_strides`, `dst_iterations` |

V0 只定义这一条 TDMA-backed movement op。copy、layout materialization、static slice movement、broadcast
和 transpose 都要么映射成一条或多条 gather_scatter，要么失败。`wafer.instr.copy` 不作为
单独 IR op；contiguous copy 是 gather_scatter descriptor 特例。`src_offset` / `dst_offset`
是 operand buffer 内的字节偏移，用于表达同一 buffer 内的分段 movement；它们不是
`wafer.spm.offset` / `wafer.ddr.offset` 这类 accepted base offset fact。

### 7.4 Fill / Elementwise / Reduce / Convert

```text
wafer.instr.fill dest, value attr-dict : type(dest), type(value)
wafer.instr.elementwise #wafer.instr_elementwise_kind<kind> inputs into dest attr-dict
    : type(inputs) into type(dest)
wafer.instr.reduce #wafer.instr_reduce_kind<kind> input into dest (, init)? attr-dict
    : type(input) into type(dest)
wafer.instr.convert #wafer.instr_convert_kind<src_dst> source into dest attr-dict
    : type(source) to type(dest)
```

| op | operands | result | required attrs |
| --- | --- | --- | --- |
| `wafer.instr.fill` | `dest: SPM memref`, `value: scalar` | none | none |
| `wafer.instr.elementwise` | `inputs: Variadic<SPM memref>`, `dest: SPM memref` | none | `kind: #wafer.instr_elementwise_kind`; optional `indexing_maps` copied from target-abstract op |
| `wafer.instr.reduce` | `input: SPM memref`, `dest: SPM memref`, optional scalar `init` | none | `kind: #wafer.instr_reduce_kind`, `dim` target reduce code |
| `wafer.instr.convert` | `source: SPM memref`, `dest: SPM memref` | none | `kind: #wafer.instr_convert_kind`; required `zero_point` for INT8->FP kinds, required `rounding_mode` for rounding wrapper kinds, no extra attrs for plain kinds |

`wafer.instr.convert` 作为 instruction op 定义，因为 hardware convert 当前属于 CT instruction family；
其 `kind` 直接对应 convert wrapper / opcode pair，例如 `fp32_int32`，verifier 从 kind 推导
source/dest element type 并检查 memref type。INT8->FP wrapper 组需要 `zero_point`；FP/INT
之间需要 rounding 的 wrapper 组需要 `rounding_mode`，取值范围由 target rounding mode 编码约束；
plain wrapper 组不允许携带这两个 attr。当前没有 `wafer.tile.convert` source op，因此 V0 定义
ODS/verifier 和 package metadata intake，但不声称存在 tile convert lowering pattern。

`#wafer.elementwise_kind` / `#wafer.reduce_kind` 只允许出现在 tile-level target-abstract op。
instruction lowering 必须显式执行：

```text
#wafer.elementwise_kind<add> -> #wafer.instr_elementwise_kind<add>
#wafer.reduce_kind<sum>      -> #wafer.instr_reduce_kind<sum>
semantic select              -> gather_scatter + bit2fp + mask_move
```

这样 `select` 和未来其它无法直接对应目标 wrapper 的 semantic op 不会靠 verifier 黑名单混入
instruction IR。

### 7.5 GEMM

```text
wafer.instr.gemm lhs, rhs into dest attr-dict
    : type(lhs), type(rhs) into type(dest)
```

| op | operands | result | required attrs |
| --- | --- | --- | --- |
| `wafer.instr.gemm` | `lhs: SPM memref`, `rhs: SPM memref`, `dest: SPM memref` | none | `m`, `k`, `n`; optional batched GEMM attrs copied from `wafer.tile.gemm` |

V0 要求三个 SPM memref operand 都使用 aligned layout marker：rank <= 2 使用
`#wafer.memory<spm, cx>`，rank > 2 使用 `#wafer.memory<spm, ncx>`。Fused bias、activation、
quant、psum accumulation policy 和 sparse / INT8 variants 不属于 R3.2d V0。

### 7.6 Conv / Pool / UnPool

```text
wafer.instr.conv #wafer.instr_conv_kind<kind> input, weight into dest attr-dict
    : type(input), type(weight) into type(dest)
wafer.instr.pool #wafer.instr_pool_kind<kind> input into dests attr-dict
    : type(input) into type(dests)
wafer.instr.unpool #wafer.instr_unpool_kind<kind> input into dest attr-dict
    : type(input) into type(dest)
```

| op | operands | result | required attrs |
| --- | --- | --- | --- |
| `wafer.instr.conv` | `input: aligned SPM memref`, `weight: aligned SPM memref`, `dest: aligned SPM memref` | none | `kind: #wafer.instr_conv_kind`, `input_shape`, `weight_shape`, `output_shape`, `pads`, `unpads`, `kernel_strides`, `dilations` |
| `wafer.instr.pool` | `input: aligned SPM memref`, `dests: Variadic<aligned SPM memref>` | none | `kind: #wafer.instr_pool_kind`, `source_shape`, `dest_shape`, `pads`, `kernel_strides` |
| `wafer.instr.unpool` | `input: aligned SPM memref`, `dest: aligned SPM memref` | none | `kind: #wafer.instr_unpool_kind`, `source_shape`, `dest_shape`, `kernel_strides`; scalar `index` attr required for `unpool` / `mask`, forbidden for `avg` |

`wafer.instr.conv` 覆盖基础 Conv / Depthwise / BackwardConv packet family。V0 IR 只表达
feature、weight、dest 和基础 geometry descriptor；bias、scale、activation、sparse、INT8 quant
和 psum policy 不能作为隐式 default 藏在 target lowering 里，后续需要扩 operand / attr 后才合法。

`wafer.instr.pool` / `wafer.instr.unpool` 覆盖 CT Pool/UnPool wrapper family。普通 pool kind 只有
一个 value dest；`indexedmax` / `indexedmin` 必须有 value dest 和 i32 index dest。UnPool public
wrapper 的 indexed/mask variants 使用 scalar `uint32_t index`，不是 index SPM buffer；`avg` variant
没有 index 参数。shape、pad、stride attr 都是 wrapper-level descriptor 字段，不是 tile-level
semantic layout 描述；source lowering 需要先把 feature layout materialize 到对应 aligned SPM layout。

### 7.7 Structured TDMA DataMove / Peripheral

```text
wafer.instr.tdma_data_move #wafer.instr_data_move_kind<kind> source into dest attr-dict
    : type(source) to type(dest)
wafer.instr.peripheral #wafer.instr_peripheral_kind<kind> inputs into dests attr-dict
    : type(inputs) into type(dests)
```

| op | operands | result | required attrs |
| --- | --- | --- | --- |
| `wafer.instr.tdma_data_move` | `source: SPM memref`, `dest: SPM memref` | none | `kind: #wafer.instr_data_move_kind`, `source_shape`, `dest_shape`; pre-lowering transform attrs are `permutation` for transpose and `axes` for mirror/rotate; V0 production target allows only `pad` with `pads` and `img2col` with `pads` + `kernel_strides` |
| `wafer.instr.peripheral` | `inputs: Variadic<SPM memref>`, `dests: Variadic<SPM memref>` | none | `kind: #wafer.instr_peripheral_kind`, `elem_count`; kind-specific attrs for bilinear/LUT/elem_mask |

`wafer.instr.tdma_data_move` 的 V0 production surface 只表达 wrapper-level `pad` / `img2col`。
mirror、transpose、rotate、NCHW/NHWC 和 TensorNom 这类 transform-like DataMove kind 虽然有
public wrapper/header 证据，但 V0 不把它们作为 production target surface；普通 copy、layout segment
movement、static slice / broadcast / transpose / mirror / rotate 的可证明 byte movement 由 compiler
lowering 展开成 `wafer.instr.gather_scatter`，无法表达时由 lowering 结构化失败。`mirror`
的 `axes` 是被翻转的 logical axis 集合；`rotate90/180/270` 的 `axes = [a, b]` 是有序旋转平面，
`rotate90` 表示在该平面上的 clockwise quarter turn，`rotate270` 表示反向 quarter turn，
`rotate180` 保持 shape 不交换但翻转两个轴。

`wafer.instr.peripheral` 覆盖 argmax、argmin、factorize、bilinear、lut16、lut32、rand_gen 和
elem_mask 的 public wrapper 形态。kind 决定 input/dest arity；argmax/argmin 的第一个 dest 是
value，第二个 dest 必须是 i32 index。`bilinear` 要求 `source_shape` / `dest_shape`；
`lut16` / `lut32` 要求 `lut_elem_count`；`elem_mask` 要求 `scale`、`probability` 和
`rounding_mode`。`count` public wrapper 的 writeback 不表现为普通 dest buffer，当前 instruction IR
不表示它；opcode 176 `bitcount` 只有 enum 证据、没有 public wrapper 证据，不进入当前 IR kind。

## 8. Lowering Rules

R3.2d 应实现为 MLIR DialectConversion：

- illegal：`wafer.tile.load`、`wafer.tile.store`、`wafer.tile.materialize_layout`、
  `wafer.tile.fill/gemm/elementwise/reduce` 和 tile movement ops。
- legal：`memref.alloc`、standard memref view ops、`wafer.instr.*`、
  `wafer.instr.local_fence`、`wafer.tile.region` container、`scf.if` / `scf.for` container
  和必要 scalar/support op。
- no type conversion for Wafer tagged memref values。
- conversion failure 必须结构化返回给 planner；rejected instruction IR 不进入 committed 主线 IR。

V0 mapping：

| target-abstract op | instruction-level lowering |
| --- | --- |
| `wafer.tile.load` | ensure / create destination `memref<..., #wafer.memory<spm, tensor>>`; read source DDR memref strided layout, emit contiguous or three-level strided `wafer.instr.rdma`; replace original result with dest memref |
| `wafer.tile.store` | read destination DDR memref strided layout, emit contiguous or three-level strided `wafer.instr.wdma`; erase store |
| `wafer.tile.materialize_layout` | ensure / create destination memref with requested marker; compare source/result logical element to physical byte mapping through the unified physical layout calculator; coalesce adjacent byte segments, pack regular segments into up to three stride/iteration levels, and emit one or more `wafer.instr.gather_scatter`; do not require source/result physical byte counts to match; structured failure only when static logical movement cannot be represented by V0 descriptors |
| `wafer.tile.fill` | emit `wafer.instr.fill` writing the existing dest memref |
| `wafer.tile.gemm` | ensure / create destination aligned SPM memref; emit `wafer.instr.gemm`; replace result with dest memref |
| `wafer.tile.elementwise` | ensure / create destination SPM memref; map non-select `#wafer.elementwise_kind` to `#wafer.instr_elementwise_kind` and emit `wafer.instr.elementwise`; semantic select lowers to false-copy `gather_scatter` + `bit2fp` + `mask_move`; replace result with dest memref |
| `wafer.tile.reduce` | ensure / create destination SPM memref; map `#wafer.reduce_kind` to `#wafer.instr_reduce_kind`, map tile semantic `dimensions` to native reduce `dim` code, emit `wafer.instr.reduce`, and replace result with dest memref |
| `wafer.tile.copy` | ensure / create destination SPM memref; emit one gather_scatter; replace result with dest memref |
| `wafer.tile.extract_slice` | create destination SPM memref; enumerate the static slice result logical domain, map each result index through offsets/sizes/strides back to the source logical index, compute physical byte offsets with the unified Wafer layout calculator, coalesce adjacent byte segments, pack regular segments into up to three stride/iteration levels, emit one or more `wafer.instr.gather_scatter`, and replace result with dest memref |
| `wafer.tile.insert_slice` | create destination SPM memref; first copy the original destination payload into the result via logical-to-physical segments, then enumerate source logical indices and overlay them into the statically described destination slice via packed `wafer.instr.gather_scatter` descriptors; replace result with the new memref |
| `wafer.tile.broadcast` | create destination SPM memref; enumerate the static result domain, map source dims through `dimensions`, compute source/result physical byte offsets, coalesce adjacent segments, pack regular segments into up to three stride/iteration levels, emit one or more `wafer.instr.gather_scatter`, and replace result with dest memref |
| `wafer.tile.transpose` | create destination SPM memref; enumerate the static result domain, invert `permutation` to source logical indices, compute source/result physical byte offsets, coalesce adjacent segments, pack regular segments into up to three stride/iteration levels, emit one or more `wafer.instr.gather_scatter`, and replace result with dest memref |
| `wafer.tile.reshape` | identity replacement when types are identical; otherwise preserve source/result canonical linear element order and reinterpret result multi-indices through the new shape; compact `tensor/ntensor` reshape lowers to a verifier-legal standard memref view because compact physical bytes already follow that linear order; `Cx/NCx` reshape first compares same-linear-element source/result physical byte offsets with the unified physical layout calculator, materializes a destination memref and emits packed `wafer.instr.gather_scatter` descriptors only when the physical mapping or required footprint changes; structured failure only when the static reshape movement plan cannot be represented by V0 descriptors |
| `scf.if` / `scf.for` | preserve the structured control-flow op; recursively legalize executable target-abstract ops in each nested region; keep scalar and memref yields explicit |
| `wafer.tile.all_gather` | V0 requires matching `tensor/ntensor` SPM layouts and infers the unique gather axis from compact local/gather buffer shapes. Schedule policy `auto` maps to `ring` by default. `ring` copies the local chunk into the local gather slot, fences, then forwards slot views around the logical ring. `direct` copies the local chunk into the local slot, fences once, then sends that local slot directly to every other logical rank while receiving each peer chunk into that peer's result slot. |
| `wafer.tile.all_reduce` | V0 requires matching `tensor` SPM buffers and sum/max/min reduce kind. Schedule policy `auto` maps to `ring` by default. `ring` creates an accumulator and forward staging buffer, fences local copies, then forwards the most recently received partial around the logical ring and accumulates with `wafer.instr.elementwise`. `tree` materializes a binomial-tree reduce to group-local root 0, fences accumulator writes before any DTE read of accumulator, then broadcasts the final accumulator down the reverse tree. |
| `wafer.tile.reduce_scatter` | V0 requires a full `tensor` SPM input whose scatter `axis` size is `group_size * result_axis_size`, plus matching local-slot recv/result buffers. Schedule policy `auto` maps to `direct` by default. `direct` creates a local accumulator from the current rank slot, then emits phase-ordered fixed-size `wafer.instr.dte_send` / `dte_recv` / `dte_wait` steps where phase `d` sends input slot `(local_rank + d) mod group_size` to that slot owner and receives this rank's local-slot contribution from `(local_rank - d) mod group_size`, accumulating each received contribution with `wafer.instr.elementwise`. |

R3.2d.4 已覆盖 static movement descriptor splitting / packing：

- `wafer.tile.extract_slice`、`wafer.tile.insert_slice`、`wafer.tile.broadcast`、`wafer.tile.transpose`
  都复用统一 logical-to-physical calculator，覆盖 compact `tensor/ntensor` 与 `Cx/NCx`。它们先从
  op 语义恢复 source/result logical index relation，再计算两端 physical byte offset；不能用 generic
  memref load/store/copy 或名字匹配绕过 movement 语义。
- V0 当前只 materialize 静态、byte-addressable、可按 buffer-local offset 表示的 descriptor 序列。
  lowering 先 coalesce 相邻 byte 段，再贪心识别可由三层 source/dest stride/iteration 同时描述的
  规则 segment block；不能被单个 descriptor 表达的剩余段继续拆成后续 `wafer.instr.gather_scatter`。
  dynamic shape、bit-packed element、超过三层或 helper 无法证明真实 physical offset 的情况仍
  structured failure。

R3.2d V0 communication coverage：

- `wafer.tile.all_gather` 已能在 `rank_group`、group-local `local_rank`、`group_size`、`bytes`
  和静态 compact `tensor/ntensor` SPM buffer shape 均可验证时 materialize fixed-size unicast schedule。
  pass option `all-gather-schedule=auto|ring|direct` 只选择 rewrite policy；accepted result 仍是
  explicit `wafer.instr.dte_*` body，不保存 schedule attr。`auto` 默认 `ring`。
- `wafer.tile.all_reduce` 已能在 `rank_group`、group-local `local_rank`、`group_size`、`bytes`、
  `tensor` SPM buffer type 和 sum/max/min reduce kind 均可验证时 materialize fixed-size unicast schedule。
  pass option `all-reduce-schedule=auto|ring|tree` 只选择 rewrite policy；`auto` 默认 `ring`。
  `tree` 使用 group-local root 0 的 binomial reduce + reverse broadcast。reduction 不藏进 DTE side
  effect；每个 reduce step 都先 wait DTE token，再用 `wafer.instr.elementwise` 做本地累计。
- `wafer.tile.reduce_scatter` 使用 full input + local slot result 表示。tile-region lowering 不再预先把
  input 截成当前 rank 的 slot；instruction lowering 从 full input 的 per-target slot `memref.subview`
  直接派生 p2p send source，并在 wait 后显式累计 recv contribution。该 V0 是 phase-ordered
  all-to-owner unicast schedule。pass option `reduce-scatter-schedule=auto|direct` 只选择 rewrite policy；
  `auto` 默认 `direct`，不保存全局 plan attr。

R3.2d may generate multiple instruction ops for a single target-abstract movement op, but it must not write a
global schedule attr. The instruction sequence is the region body itself.
Nested `scf` regions are part of that body: R3.2d rewrites their executable contents under MLIR region
scoping rules, but it does not lower them to hardware branch/loop instructions.

## 9. Failure Contract

R3.2d failure is a legalization result, not an IR artifact. A rejected legalization attempt may carry
diagnostics to the closed-loop planner or debug pass, but rejected instruction IR is discarded.

必须结构化失败的情况：

- non-ranked or dynamic-shaped memref where V0 needs static byte/stride computation.
- unsupported Wafer memory attr, address space or physical layout marker for an instruction family.
- unsupported dtype, including relation/elementwise/convert pairs not mapped to CT V0.
- movement descriptor cannot be represented with buffer-local offsets, `inner_bytes` and three
  stride/iteration levels.
- `Cx/NCx` materialization with retained `C0` tail cannot be split into separately representable
  full-block and tail GatherScatter descriptors.
- static slice/insert/broadcast/transpose whose logical index relation or physical byte offsets cannot be
  converted into one or more `gather_scatter` descriptors.
- unsupported control-flow op, multi-block region, or nested region whose executable body cannot be fully
  legalized under the same instruction conversion rules.
- NE GEMM dimension attrs cannot be derived from operand/result types and optional batch attrs.
- tile reduce `dimensions` cannot map to supported native reduce `dim` code.
- any source op that would require raw DTE resource ids, CSR/SCALAR, raw packet fields, SPM offset,
  DDR planning result or
  runtime ABI call to be legal.

Diagnostics should mention the source op and the missing legality fact, for example:
`tile.reduce_scatter lowering requires tensor SPM buffers` or
`tile.broadcast lowering requires static positive iteration shape`.

## 10. Verifier Contract

R3.2d verifier checks only instruction legality:

- ODS type constraints enforce memref/tensor/scalar operand classes.
- all Wafer tagged memref types used by instruction ops are ranked and static for V0.
- memory attr matches instruction family:
  RDMA reads `#wafer.memory<ddr, *>` and writes `#wafer.memory<spm, *>`；
  WDMA reads `#wafer.memory<spm, *>` and writes `#wafer.memory<ddr, *>`；
  TDMA/CT/NE read/write tile-local SPM memrefs。
- RDMA/WDMA/TDMA descriptor attrs have fixed array length, positive iteration values, non-negative byte
  strides and positive byte counts.
- descriptor byte counts are consistent with compact tensor payload size or statically described
  slice/broadcast/transpose domain as applicable. Cx/NCx padding span is derived from
  `computeWaferPhysicalTensorInfo(memrefType)` and target policy, not copied into `byte_count`.
- NE GEMM and CT reduce require supported aligned layout marker, dtype and rank.
- relation/elementwise bool storage uses logical `i1`; physical byte size remains derived, not stored.
- `wafer.instr.elementwise` / `wafer.instr.reduce` use instr-level target kind attrs only；generic
  `#wafer.elementwise_kind` / `#wafer.reduce_kind` on instruction ops is verifier-illegal.
- `wafer.instr.convert` uses `#wafer.instr_convert_kind` only；source/dest dtype is derived from the
  convert kind and checked against memref element types. It does not accept free-form `src_dtype` /
  `dst_dtype` attrs as instruction semantics. Kind-specific `zero_point` / `rounding_mode` attrs are
  required or forbidden according to the public wrapper signature group.
- `wafer.instr.conv` requires aligned SPM input/weight/dest memrefs, matching element types, rank-4
  positive shape descriptors, rank-4 non-negative pads/unpads, rank-4 positive kernel strides and rank-2
  positive dilations.
- `wafer.instr.pool` / `wafer.instr.unpool` require aligned SPM operands, matching value element type,
  rank-4 positive source/dest shape descriptors and positive stride descriptors. Indexed pool index dest
  must use i32 element type. Unpool `unpool` / `mask` require scalar uint32 `index`; `avg` forbids it.
- `wafer.instr.tdma_data_move` requires SPM source/dest memrefs with matching element type, rank-4
  positive source/dest descriptors, and kind-specific descriptor attrs. V0 production target only
  accepts `pad` with `pads` and `img2col` with `pads` + `kernel_strides`. Transform-like kinds
  `mirror/transpose/rotate*/nchw2nhwc/nhwc2nchw/tensor_nom` are verifier-legal only as explicit
  pre-lowering/imported IR and must be materialized to `gather_scatter` before target LLVM or package
  export. `transpose` requires `permutation`; `mirror` requires non-empty unique `axes`; rotate kinds
  require exactly two unique `axes`; `axes` entries must be within the source/dest logical tensor rank,
  and mirror/rotate operands must be rank-compatible. Unrelated attrs are rejected per kind. These kinds are not
  production target lowering input unless a future target path is explicitly enabled.
- `wafer.instr.peripheral` verifies kind-specific input/dest arity and positive `elem_count`; arg
  peripheral ops require a same-dtype value dest plus an i32 index dest. `bilinear` requires
  source/dest shape attrs; LUT kinds require `lut_elem_count`; `elem_mask` requires `scale`,
  `probability` and `rounding_mode`; `count` is rejected until its writeback semantics are represented.
- no SPM offset/end/bank attrs before SPM offset assignment.
- no raw DTE resource id, raw DTE register field, CSR helper or SCALAR ordinary instruction op before
  the corresponding instruction/sync family is defined and verified. Direct DTE p2p must use
  `wafer.instr.dte_send` / `dte_recv` / `dte_wait`, not ad hoc tile p2p ops or side tables.

Instruction lowering does **not** verify physical address range, SPM bank conflicts, DDR default arena capacity,
runtime symbol, packet bit layout or worker register window. Those checks belong to SPM/DDR offset assignment,
target LLVM lowering, package metadata and runtime adapter.
DDR offset assignment must accept or reject the explicit DDR views, descriptors and compiler-managed DDR `memref.alloc`
already present in this IR, and must materialize accepted DDR offset facts before target LLVM lowering,
package metadata and runtime adapter consume them through resource view analysis.

## 11. Example

```mlir
wafer.tile.region ... {
  %a_ddr = ... : memref<128x64xf16, #wafer.memory<ddr, tensor>>
  %b_ddr = ... : memref<64x128xf16, #wafer.memory<ddr, tensor>>
  %c_ddr = ... : memref<128x128xf16, #wafer.memory<ddr, tensor>>

  %a_tensor = memref.alloc() {alignment = 256}
      : memref<128x64xf16, #wafer.memory<spm, tensor>>
  %b_tensor = memref.alloc() {alignment = 256}
      : memref<64x128xf16, #wafer.memory<spm, tensor>>
  %a_cx = memref.alloc() {alignment = 256}
      : memref<128x64xf16, #wafer.memory<spm, cx>>
  %b_cx = memref.alloc() {alignment = 256}
      : memref<64x128xf16, #wafer.memory<spm, cx>>
  %c_cx = memref.alloc() {alignment = 256}
      : memref<128x128xf16, #wafer.memory<spm, cx>>
  %c_tensor = memref.alloc() {alignment = 256}
      : memref<128x128xf16, #wafer.memory<spm, tensor>>

  wafer.instr.rdma %a_ddr to %a_tensor
      {byte_count = 16384 : i64, inner_bytes = 16384 : i64,
       src_strides = array<i64: 0, 0, 0>, src_iterations = array<i64: 1, 1, 1>}
      : memref<128x64xf16, #wafer.memory<ddr, tensor>>
     to memref<128x64xf16, #wafer.memory<spm, tensor>>

  wafer.instr.rdma %b_ddr to %b_tensor
      {byte_count = 16384 : i64, inner_bytes = 16384 : i64,
       src_strides = array<i64: 0, 0, 0>, src_iterations = array<i64: 1, 1, 1>}
      : memref<64x128xf16, #wafer.memory<ddr, tensor>>
     to memref<64x128xf16, #wafer.memory<spm, tensor>>

  wafer.instr.gather_scatter %a_tensor to %a_cx
      {byte_count = 16384 : i64, inner_bytes = 128 : i64,
       src_strides = array<i64: 128, 0, 0>, src_iterations = array<i64: 128, 1, 1>,
       dst_strides = array<i64: 256, 0, 0>, dst_iterations = array<i64: 128, 1, 1>}
      : memref<128x64xf16, #wafer.memory<spm, tensor>>
     to memref<128x64xf16, #wafer.memory<spm, cx>>

  wafer.instr.gather_scatter %b_tensor to %b_cx
      {byte_count = 16384 : i64, inner_bytes = 128 : i64,
       src_strides = array<i64: 128, 0, 0>, src_iterations = array<i64: 128, 1, 1>,
       dst_strides = array<i64: 256, 0, 0>, dst_iterations = array<i64: 128, 1, 1>}
      : memref<64x128xf16, #wafer.memory<spm, tensor>>
     to memref<64x128xf16, #wafer.memory<spm, cx>>

  wafer.instr.gemm %a_cx, %b_cx into %c_cx
      {m = 128 : i64, k = 64 : i64, n = 128 : i64}
      : memref<128x64xf16, #wafer.memory<spm, cx>>,
        memref<64x128xf16, #wafer.memory<spm, cx>>
    into memref<128x128xf16, #wafer.memory<spm, cx>>

  wafer.instr.gather_scatter %c_cx to %c_tensor
      {byte_count = 32768 : i64, inner_bytes = 256 : i64,
       src_strides = array<i64: 256, 0, 0>, src_iterations = array<i64: 128, 1, 1>,
       dst_strides = array<i64: 256, 0, 0>, dst_iterations = array<i64: 128, 1, 1>}
      : memref<128x128xf16, #wafer.memory<spm, cx>>
     to memref<128x128xf16, #wafer.memory<spm, tensor>>

  wafer.instr.wdma %c_tensor to %c_ddr
      {byte_count = 32768 : i64, inner_bytes = 32768 : i64,
       dst_strides = array<i64: 0, 0, 0>, dst_iterations = array<i64: 1, 1, 1>}
      : memref<128x128xf16, #wafer.memory<spm, tensor>>
     to memref<128x128xf16, #wafer.memory<ddr, tensor>>

  wafer.instr.local_fence
}
```

这是形态示例，不固定 parser/printer，也不固定 planner 对 layout materialization 的具体选择。
例子中 stride 数值只说明 descriptor 字段位置，不作为 Cx padding 或 hardware packet 的规范值；
真实 padded size、Cx/NCx 对齐、bool bitpack、descriptor stride 和 SPM offset 分别由
`computeWaferPhysicalTensorInfo`、SPM memory planning 和 later realization 处理。

## 12. Implementation Work

R3.2c 已完成的前置：

1. `#wafer.memory<space, layout>` target memory attr。
2. `computeWaferPhysicalTensorInfo(memrefType)` 和
   `computeWaferPhysicalElementByteOffset(memrefType, indices)`，统一计算 layout marker、Cx/C0、
   footprint、range-end、bool bitpack、wrapper layout enum 和 logical-to-physical offset。
3. 旧 `#wafer.memory_space` / `#wafer.mem_layout` / `!wafer.storage` / `wafer.tile.alloc`
   合同已从主线 IR 定义和测试中删除。

R3.2d.1 已完成：

1. 增加 `InstrFamily` enum、`WaferInstructionOpInterface` 和 instruction effect helper。
2. 增加 `wafer.instr.rdma`、`wafer.instr.wdma`、
   `wafer.instr.gather_scatter`、`wafer.instr.fill`、
   `wafer.instr.elementwise`、`wafer.instr.reduce`、
   `wafer.instr.convert`、`wafer.instr.gemm`、`wafer.instr.conv`、
   `wafer.instr.pool`、`wafer.instr.unpool`、`wafer.instr.tdma_data_move`
   和 `wafer.instr.peripheral` ODS。
3. 为每个 op 实现 verifier、MemoryEffects、instruction interface 和 positive/negative lit tests。
4. `wafer.instr.elementwise`、`wafer.instr.reduce` 和 `wafer.instr.convert` 已切到
   instr-specific target kind attrs；旧 tile-level `Compute*Kind` attr 不能再作为 instruction op
   attr。`wafer.instr.convert` 的 kind 与硬件 convert opcode pair 对齐，verifier 从 kind 检查
   source/dest dtype，并按 public wrapper signature group 检查 `zero_point` / `rounding_mode`。
   `wafer.instr.reduce` 使用 native reduce `dim` code；tile reduce semantic `dimensions` 只在
   tile 层保留，并在 lowering 中 materialize 成 `dim`。
5. `wafer.instr.unpool` 已改成 wrapper-aligned scalar `index` attr；`mask` / `unpool` 需要
   uint32 `index`，`avg` 禁止携带。`wafer.instr.tdma_data_move` verifier 按 kind 检查字段组合：
   `pad` / `img2col` 是 V0 production target input，transform-like kind 只允许作为 pre-lowering/imported
   instr IR；`transpose`、`mirror`、`rotate*`、NCHW/NHWC 和 TensorNom 由
   `--wafer-convert-tile-region-to-instr` 在 target/package 边界前 materialize；
   `wafer.instr.peripheral` verifier 按
   kind-specific wrapper signature 检查 required/forbidden attrs；`count` peripheral 因 writeback
   未被当前 IR 表示而结构化拒绝。

R3.2d.2 已完成：

6. 实现 `--wafer-convert-tile-region-to-instr` DialectConversion，并提供
   `WaferTileRegionToInstr` conversion library API。
7. conversion 递归处理 `scf.if` / `scf.for` region body，并保留 scalar / memref yield 关系。
8. conversion tests 覆盖 load/store、layout materialize、fill、GEMM、elementwise、reduce、
   copy、metadata view lowered to standard memref view、`Cx/NCx` block-major reshape、
   tail-only metadata reshape、nested `scf.if`、tile communication structured failure，以及
   padding layout materialization structured failure。

R3.2d.3 已完成：

9. 增加 `wafer-lower-tile-region-to-instr` 和 `wafer-lower-groups-to-instr` named pipeline，
   复用同一 `WaferTileRegionToInstr` conversion implementation。它们是 bring-up / explicit-view
   lowering 入口；R3.2e 已支持当前 IR 中 explicit static boundary slice 到 DDR tile view，也提供
   candidate output tile offsets/sizes 的 evaluation materialization。R3.2d 仍只消费 DDR view，不负责
   搜索 traversal / tile shape。
10. pipeline tests 覆盖多 group、structured `scf.if` / `scf.for`、tile communication
   structured failure，以及转换后不能残留 executable target-abstract op 的 pipeline-level gate。

R3.2d.4 已完成：

11. `wafer.tile.extract_slice`、`wafer.tile.insert_slice`、`wafer.tile.broadcast` 和
   `wafer.tile.transpose` 的静态 movement lowering 已接入同一个 DialectConversion，按 op 语义枚举
   logical iteration domain，调用统一 physical layout calculator 计算 source/dest byte offset，并
   coalesce 相邻 byte 段后打包成三层 stride/iteration `wafer.instr.gather_scatter` descriptor。
12. conversion tests 覆盖 compact tensor movement，以及 `Cx:[CxBlock][outer][lane]` /
    `NCx:[N][CxBlock][HW][lane]` block-major physical offset 的切片 descriptor 和 reshape/layout
    materialization descriptor packing。

后续仍需：

13. convert lowering 等 `wafer.tile.convert`
   或等价 source op 出现后再接入 completion gate。
14. dynamic DDR view、超过三层的 RDMA/WDMA descriptor packing、以及非 tensor-layout DDR 边界等
    source op 出现后再接入；当前不能靠名字或 shape 猜测出缺失的 strided boundary facts。
