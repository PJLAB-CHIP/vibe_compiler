# Wafer Instruction IR Design

状态：2026-07-20同步MLIR-native physical-dataflow candidate边界；当前合同覆盖instruction-level hardware invocation IR、
memref buffer和shared physical geometry/ABI legality。shared verifier 已闭合当前支持子集的静态 DMA descriptor payload/range/
element-width relation、fill/elementwise/reduce/convert/GEMM element/shape relation、ordinary conv、pool/
unpool、TDMA pad/img2col和peripheral kind-specific capacity，并在target字段写入前检查ABI narrowing。
尚无精确算子关系或physical binding的shape/profile必须fail closed；当前depthwise/backward conv等未证明
family、peripheral factorize和没有explicit arena base binding的compiler-managed DDR allocation均为production
target-illegal。Direct DTE fixed-size unicast只在Q16.T all-rank acceptance已提交physical endpoint、receiver offset、
FSM/completion和status ABI后进入production target；缺任一binding仍fail closed。`mask_move`的compiler/CRT ABI
已统一为显式`uint32_t mask`，不允许wrapper内部隐藏narrowing。
Q22 review确认现有target lowering不消费reduce init，也不消费elementwise `indexing_maps`。Q0.L终态要求二者只存在于
tile-level IR，并在tile→instruction阶段完成movement/materialization、显式reduce分解或拒绝；terminal instruction op不再
携带这些字段。该缺口不回滚Q17 artifact publication状态，但CModel不得补偿。
当前ODS对compact RDMA/WDMA继续省略offset pair，对mapped transfer则要求两端root-relative offset同时显式存在（包括0）；
plain GEMM继续表示implicit normal/normal，oriented GEMM必须显式携带两个closed orientation attrs并选择v2 profile/ABI。
Q32.V已闭合mapped transfer、physical-footprint fill与versioned oriented GEMM的source/Tile/Instr/TargetCall/formal/SystemC纵向；
Q32.M/S才负责让这些能力进入共同candidate owner，planner不能复制或猜测字段。实现状态以`tasks/progress.md`为准。

本文定义 instruction-level Wafer IR。核心结论：

- `wafer.instr.*` 是当前 compiler pipeline 需要的 target-aligned instruction subset，不是完整
  硬件 ISA、Tsm wrapper 或 opcode 全量镜像。每个进入该层的 op 必须能被 verifier 解释，并且在
  target LLVM call emission 中要么 lower 到明确 target CRT / DTE helper call，要么结构化失败。
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
- committed instruction artifact 不是单个 `wafer.tile.region`、task、tile 或 representative sample，
  而是每个 logical rank 一份覆盖完整 static traversal 的 structured instruction program；这些 rank
  programs 只能作为完整 variant set 原子提交。

`wafer.instr` 的作用是把完整 rank traversal 中的 target-abstract tile-region scopes 变成可执行硬件动作或硬件通信调用，并让下游能从
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
  IR 外的调度计划自行恢复这些 view。closed-loop scheduler 产生的 candidate tile 必须先由
  candidate/accepted materialization 显式变成 DDR subview。
- instruction lowering只消费candidate rewrite已经物化到payload IR的typed implementation字段、operands、views和memory/
  layout types，以及同一compile request的`TargetProfileId`/immutable target facts；不消费implementation/route proposal或其它
  side plan。compute lowering验证显式selected字段；boundary lowering从两端typed views/encoding和08 transfer proof导出
  descriptor cover。不能在本层重新选择implementation/route，也不能让target/CModel从layout、shape或旧trace补猜选择。
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
  whole-variant evaluation clone 中每个 static rank entry 的完整 structured traversal；其中局部
  `wafer.tile.region` scopes 包含带 Wafer memory attr 的 memref values、
  `memref.alloc` / `memref.subview` / verifier-legal metadata view、
  `wafer.tile.load` / `wafer.tile.store`、
  `wafer.tile.materialize_layout`、`wafer.tile.fill/gemm/elementwise/reduce`、
  `wafer.tile.copy/extract_slice/insert_slice/transpose/broadcast`、
  `wafer.tile.*` buffer-level collective、tile-region 内 `scf.if` / `scf.for`
  structured control-flow 和 `wafer.instr.local_fence`。
- Current stage responsibility:
  只做 Wafer instruction legalization / selection：递归覆盖所有 rank entries 的完整 traversal，验证candidate rewrite已
  物化到typed payload IR的implementation字段、operands、views和memory/layout types；boundary lowering从这些IR事实与
  显式target profile fresh导出唯一direct descriptor cover，不读取family对象或side plan；
  把每个可执行 target-abstract op 改写成
  `wafer.instr.*`，把 tile-level `Compute*Kind` 选择成 instr-level target kind，并保留 memref SSA
  graph。对 `scf.if` / `scf.for` 递归转换其 region body并保留control-flow结构；每个`scf.for`
  body在backedge前materialize显式local fence，使每次动态迭代的local issue独立完成。该producer fence不把
  loop body创建并跨backedge携带的allocation/task变成单实例；后续memory planning仍须要求multi-instance表示或
  fail closed。对 accepted
  `wafer.tile.*` collective，生成 explicit
  `wafer.instr.dte_send` / `dte_recv` / `dte_wait` p2p schedule。instruction op 通过
  interface 显式暴露current v1 instruction family、memref read/write、descriptor attrs、issue effect和completion relation；
  current GS/staged movement还暴露IR中已经物化的local offset/count与valid-lane mode。Q32.V已增加
  GEMM orientation、mapped DMA双端root-relative offset和physical-fill domain；这些字段只来自当前source/Tile/Instr IR
  和exact physical proof，不来自planner历史。
- Output artifact / IR:
  candidate clone 中每个 static rank 一份完整 instruction-level structured program：
  control flow + tile-local `wafer.tile.region` scopes + memref values with
  `#wafer.memory<space, layout>` + `wafer.instr.*` + explicit token/wait/fence；或结构化
  legalization failure reason。单个 task/traversal fragment/tile 不是可提交 artifact。
- Downstream consumer:
  whole-rank SPM planning、whole-variant DDR planning、event/physical-transport/all-rank transport/
  target-entry verification 和
  closed-loop whole-variant candidate driver；atomic commit 后才由 target LLVM、package 和 runtime 消费。
- User-level driver / named pipeline:
  Q16以后由同一
  `wafer-compile --input-program-dir ... --output-program-dir ... --execution-ranks={1|16} --target-profile=<registered-id>`
  内的closed-loop
  candidate loop调用；当前baseline为closed v1。Q32.V oriented request必须显式选择其typed profile/ABI revision，且不与v1隐式转换。
  当前Q15只产出verified structured tensor program directory，不执行instruction lowering；
  `wafer-opt`只处理显式IR，局部bring-up / candidate evaluation入口是
  `wafer-lower-tile-region-to-instr` named pipeline。
  candidate evaluation 调用 instruction lowering 时，tiled DDR load/store operand 必须已经由 candidate 或 accepted
  materialization 表达成 tile view；如果仍是 whole-boundary memref，instruction lowering 只能生成 whole-boundary
  descriptor。
  `--wafer-convert-tile-region-to-instr` 只作为 lit/debug pass 入口。这些局部/direct入口都不是用户
  stop-stage，也不能把
  `DirectFullShape`、单 task 或单 tile-region 结果直接送入 committed target/package flow；用户级
  completion 必须经过 whole-variant candidate-selection/commit pipeline。
  instruction lowering与`wafer-plan-spm-memory`的focused tests继续覆盖pre-existing
  identity-preserving recurrence安全正例，以及loop body fresh allocation作为recurrence result的失败反例。
- Explicit non-goals:
  不新增第二套 storage/buffer IR，不决定 scheduling boundary、tile shape、implementation/layout/transfer/residency proposal、SPM offset、
  DDR planning result、raw register packet field、DTE/FSM resource id、Tsm wrapper call、target CRT
  symbol 或 launch ABI。SCALAR 仍是 reserved/stub；CSR helper/sync 若进入主线，必须作为明确
  instruction/sync family 另行定义，不能混入 CT/NE/RDMA/WDMA/TDMA 或 DTE op。
  本层也不按 task/rank 部分提交，不允许 `DirectFullShape` 或 representative tile 绕过完整 gates，
  不把 hardware `busytable` 解释为 completion event，也不依据presumed rank equivalence省略或合并
  显式rank records。
- Completion gate:
  对每个 static rank entry 的完整 traversal 中已支持的 load/store、静态可证明 layout materialize、fill、GEMM、
  elementwise/relation、reduce、copy 和 metadata view 生成 verifier-legal instruction-level IR
  或标准 memref view，并覆盖 nested `scf.if` / `scf.for` body 递归转换。unsupported hardware
  instruction form，包括当前无法证明的 slice/broadcast/transpose descriptor，必须结构化失败，
  不能让 SPM memory planning 从 target-abstract op 猜 demand。每个 issue 必须由 explicit async
  token/wait 或 local fence 完成；generic async task identity、Direct DTE completion和local engine pending set
  分别验证，每条 exit path terminal drain 后均为空；variant-set gate
  还要证明所有 transport 匹配和 shared physical geometry/range/narrowing contract。production elementwise必须
  不携带map且same-shape；reduce init必须在tile→instruction阶段显式分解或拒绝，terminal instruction op不携带init。
  target-profile×engine×format×selected-fields必须由tasks/14 typed conversion/ABI合同准入。Q32.V启用mapped DMA时，
  必须通过descriptor-cover和两端range closure；若启用oriented GEMM，必须匹配versioned ABI capability。任一 rank/task/traversal scope
  失败都丢弃整个 clone。
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
| TDMA | `wafer.instr.fill` | `wafer.tile.fill` | `TsmPeripheral::Memset`使用`TsmDataMoveInstr`并最终发往TDMA queue的scalar/immediate fill |
| CT | `wafer.instr.elementwise` | `wafer.tile.elementwise` | `#wafer.instr_elementwise_kind` target kind；不含 select |
| CT | `wafer.instr.bit2fp` | tile semantic select lowering | i1 mask -> floating mask target peripheral op |
| CT | `wafer.instr.mask_move` | tile semantic select lowering | `TsmMaskDataMove::MaskMove`使用CT packet并最终发往CT/CGRA queue的masked SPM data movement target op |
| TDMA/CT composite | `wafer.instr.fill` + `gather_scatter` + `wafer.instr.elementwise` | `wafer.tile.reduce` | Q0.L init-first canonical-order correctness baseline；native `wafer.instr.reduce`只有compiler-owned full-domain等价证明后才可替换 |
| CT | `wafer.instr.convert` | future convert lowering | `#wafer.instr_convert_kind` opcode-aligned dtype pair + kind-specific wrapper attrs |
| NE | `wafer.instr.gemm` | `wafer.tile.gemm` | tile-local GEMM / batched GEMM |
| NE | `wafer.instr.conv` | future conv lowering / imported target op | basic Conv / Depthwise / BackwardConv packet fields |
| CT | `wafer.instr.pool` / `wafer.instr.unpool` | future pool/unpool lowering / imported target op | pool indexed-output arity and unpool scalar-index descriptor |
| TDMA | `wafer.instr.tdma_data_move` | future structured data-move lowering / imported target op | V0 production target 只允许 pad / img2col；mirror / transpose / rotate / NCHW-NHWC / TensorNom 这类 transform movement 如果以 imported target op 进入 instr IR，必须由 instr lowering 在 target LLVM / package export 前 materialize 成 gather_scatter，或在后续板端验证后再开启 target path |
| CT | `wafer.instr.peripheral` | future peripheral lowering / imported target op | arg / factorize / bilinear / LUT / random / element-mask target kind；factorize保留IR kind但当前target-illegal；count typed writeback合同见7.7，live implementation在完成前仍拒绝 |
| DTE | `wafer.instr.dte_send` / `dte_recv` / `dte_wait` | accepted `wafer.tile.*` collective p2p schedule | fixed-size unicast Direct DTE invocation over unplaced SPM memrefs |

### 4.1 Invalid-Lane Execution Domain

instruction op不保存planner的`InvalidLaneState`，但必须让该state可从最终命令重建：

- logical/segmented mode用一条或多条typed descriptor明确每段local bit/byte offset、element/byte count和valid domain；
- full-physical mode只有在registered family的operand predicate/result transfer function成立时合法，实际M/K/N或element count
  必须唯一决定hardware read/write domain；
- mask mode必须有显式mask operand/effect和target capability，不能只写proof attr；
- fill只有在typed fill domain覆盖相应lanes，且qualified fill semantics能把scalar唯一映射到canonical raw element时才建立
  `KnownSplat(typed raw value)`；不能假设scalar bits原样复制，NaN、signed zero和format conversion均按profile验证。只写valid
  segments的RDMA/WDMA/GS不改变未覆盖padding，未先fill时仍为Unknown。

Q32.V若确有多个TargetCall可观察mode，可增加封闭typed
`#wafer.valid_lane_mode<logical_segments|full_physical|masked>`核对selected contract，但它不能替代offset/count/mask。
若现有TargetCall无法从实际字段唯一恢复execution domain，就必须新增versioned ABI字段或拒绝该capability；CModel只消费
最终TargetCall sequence。当前v1 CT elementwise只接受Tensor/no-invalid-lane row；Cx/NCx CT必须先完成segmented或
full-physical TargetCall/SystemC纵向。

### 4.2 Instruction Coverage Matrix

`wafer.instr` coverage 按 compiler IR 合同分层，而不是按硬件 opcode 数量分层。target LLVM call emission
只能把 **V0 production target surface** 当作必须支持的 production lowering 输入；其它类别不能隐式进入现有
泛 op 或 lowering fallback。

这里的 `V0 production target op` 只说明 instruction IR / verifier / target LLVM call-emission 层必须识别
该 op，并生成 Wafer-owned `wafer_tx81_*` call 或结构化失败。它不说明 repo-local Wafer CRT wrapper
已经定义该 symbol，也不说明 packet/register provenance、device-code required-symbol gate 或板端执行已经通过；
CRT、device link、required-symbol和module publication属于`tasks/14`，manifest/package/runtime boundary属于`tasks/15`，
packet/register与板端证据另由`tasks/16` gate。

| 硬件 / wrapper 能力 | 当前 `wafer.instr` 表示 | coverage tier | 处理规则 |
| --- | --- | --- | --- |
| RDMA / WDMA contiguous 和三层 stride descriptor | `wafer.instr.rdma` / `wafer.instr.wdma` | V0 production target op | target LLVM call emission 必须生成 target CRT call；descriptor 保持 byte-level `inner_bytes`、stride 和 iteration |
| TDMA `TsmDataMove::GatherScatter` | `wafer.instr.gather_scatter` | V0 production target op | layout materialization、SPM copy 和可静态证明的 slice/transpose/broadcast movement 都展开为一条或多条 gather/scatter；无法压成 V0 descriptor 时结构化失败 |
| `TsmPeripheral::Memset` / scalar fill | `wafer.instr.fill` | V0 production target op | attr缺省保持v1 Tensor logical-valid count；显式`physical_footprint`从Cx/NCx/BOOL physical encoding checked派生count并覆盖padding/tail/unused bits，其它组合fail closed |
| CT arithmetic / relation / logic / activation / selected transcendental | `wafer.instr.elementwise` + `#wafer.instr_elementwise_kind` | V0 production target op | 覆盖当前 enum 中的 target kind；tile-level map必须先materialize为movement/同形状operand并strip，terminal op不携带`indexing_maps`；scalar immediate、bitpacked bool loop/VuV和rounding mode需显式target variant后才能开放 |
| semantic select | 无单条 select op | V0 composite lowering | 必须展开为 false-copy `gather_scatter` + `bit2fp` + `mask_move`；`wafer.instr.elementwise <select>` 非法 |
| CT reduce `sum/avg/max/min` | `wafer.instr.reduce` + `#wafer.instr_reduce_kind` + target `dim` code | target-native leaf；Q0.L source-reduce correctness baseline不直接生成 | terminal op不携带init operand/attr；只有compiler-owned target policy对完整value domain证明native identity/order/special-value等价时才可替换有序composite，不得依赖Q22 candidate profile |
| CT convert opcode 139..174 | `wafer.instr.convert` + `#wafer.instr_convert_kind<src_dst>` + kind-specific attrs | V0 production target op | dtype pair 由 kind 唯一决定；INT8->FP 要求 `zero_point`，rounding wrapper 要求 `rounding_mode`，plain wrapper 不允许额外转换参数；same-format copy 必须走 movement，不允许伪造成 convert |
| NE GEMM | `wafer.instr.gemm` | V0 production target op | 只表达 GEMM / batched GEMM 主路径参数；当前单一format及同element-type合同不表达product/accumulator/FMA/rounding，program-selectable行为必须先扩IR/CRT ABI，target-fixed行为必须按revision/tuple唯一映射；bias、scale、quant、fused activation和复杂psum policy不能隐式打开 |
| NE affine INT8 GEMM | `wafer.instr.quantized_gemm` | typed production extension；未完成capability/CRT/golden前target-illegal | exact M/K/N/batch/format、q0/q1、left/right zero point、typed scale operands/mode和matched capability；不复用plain GEMM flag |
| MXFP/FP8 packed decode | `wafer.instr.mxfp_decode` | explicit-composite production extension；未完成scratch/completion/CRT gate前target-illegal | packed source + block scale + destination + scratch；decode到BF16/FP16，不能冒充CT convert或native FP8 GEMM |
| Direct DTE fixed-size unicast | `wafer.instr.dte_send` / `dte_recv` / `dte_wait` | V0 production target op with committed Q16.T binding | IR表达entry-local logical peer、bytes和async token；post-memory all-rank acceptance提交physical endpoint、remote receiver offset、FSM/completion和status ABI后，target conversion生成opaque event/ready/send/wait/release CRT calls。缺binding或不一致仍以`unsupported_target_transport`拒绝；RuntimeSession不能补做endpoint/channel planning |
| local NCC drain / visibility fence | `wafer.instr.local_fence` | V0 production target sync；LLVM call emitted | target LLVM call emission 映射到 local wait/drain Wafer CRT call；不是 multi-tile barrier |
| SPM memcpy helper / copy | 无单独 copy op | V0 composite lowering | copy 是 `gather_scatter` 的 descriptor 特例；不引入 `wafer.instr.copy` |
| ChannelNorm / DechannelNorm / Tensor-Normalization | 无单条 op | V0 composite lowering | 作为 layout materialization algorithm 展开为 gather/scatter 序列；native TensorNom opcode 133 不作为 V0 主路径 |
| tile collectives `all_gather/all_reduce/reduce_scatter/all_to_all/collective_permute` | 无 collective instr op | V0 composite lowering | 先在 tile collective / schedule 层选 ring、tree 或 direct p2p，再 lower 成 DTE send/recv/wait + local movement/compute |
| ordinary `TsmConv` | `wafer.instr.conv` + `#wafer.instr_conv_kind<conv>` | V0 production target op；LLVM call emitted | input/weight/output attrs必须与memref shape一致，并证明batch/channel/kernel/stride/dilation/pad/unpad输出关系；bias、scale、sparse、INT8 quant、fused activation和psum policy仍需后续扩展 |
| `TsmDepthwiseConv` / backward conv | `#wafer.instr_conv_kind`枚举保留，但当前无合法production实例 | target-illegal pending exact profile | 当前shared verifier只有ordinary conv精确关系；不能因wrapper symbol存在就发call。恢复前必须分别定义channel/group、weight和output relation及negative gate |
| `TsmPool` / `TsmUnPool` | `wafer.instr.pool` / `wafer.instr.unpool` | V0 production target op；LLVM call emitted | source/dest attrs必须匹配memref，并证明NHWC batch/channel、pad/kernel/stride输出关系；indexed output arity/index dtype精确检查，不能复用reduce或movement op表达 |
| TDMA pad / img2col | `wafer.instr.tdma_data_move` + `#wafer.instr_data_move_kind` | V0 production target op；LLVM call emitted | pad/img2col分别证明source/dest shape、pad、kernel/stride关系；普通copy/layout segment仍优先使用`gather_scatter`；transform-like kind到target LLVM必须结构化失败 |
| TDMA mirror / transpose / rotate / NCHW-NHWC / TensorNom | 无 production target op；enum 保留用于 imported/pre-lowering IR | V0 composite lowering or future target extension | V0 由 compiler lowering 展开为 `gather_scatter` 或结构化失败。`transpose` 使用 `permutation`，`mirror` 使用一个或多个 `axes`，`rotate90/180/270` 使用有序二元 `axes` 表示旋转平面，NCHW/NHWC 使用固定 4D layout permutation，`tensor_nom` 使用 logical-linear 到 physical-layout materialization。package / target export 如果仍看到这些 kind，说明 pipeline 漏了 materialization，应拒绝 |
| TDMA concat / maskgather variants | 无单独 op | future target extension or composite | `concat` 属于 CT packet，maskgather variants 需要 bool/index operand policy；未定义前不能复用 `tdma_data_move` |
| bitpacked bool loop / VuV / scalar immediate variants | 当前 `elementwise` 只覆盖 opcode family kind，不单独建模 variant | future extension | 需要区分 value bool、bitpacked bool、VuV/VuVLoop、scalar immediate 和 output storage；不能靠 `elementwise` 名称吞掉所有 variant |
| Peripheral argmax/argmin/bilinear/lut/rand/elem_mask | `wafer.instr.peripheral` + `#wafer.instr_peripheral_kind` | V0 production target op；LLVM call emitted | kind决定input/output arity；`elem_count`必须与primary input和所有kind-specific buffers/shape capacity一致，LUT table另与`lut_elem_count`一致。Count mechanical/numeric纵向属于later Q3.6，current target保持拒绝；bitcount仍不纳入production |
| Peripheral factorize | `wafer.instr.peripheral` + `#wafer.instr_peripheral_kind<factorize>` | IR kind保留；production target-illegal | 当前没有精确factorize semantic profile；target conversion以`unsupported_target_operation`拒绝，repo-local CRT header/source不得保留对应symbol |
| raw DTE non-unicast / stream / mailbox | 无 | future communication ABI | 需要独立 communication ABI 和板端验证；V0 collective 不直接生成 raw non-unicast DTE packet |
| SCALAR / CSR ordinary execution | 无 | not compiler instr IR in V0 | `TsmExecute` 普通 dispatch 不覆盖 SCALAR/CSR；CSR wait/sync 只能通过明确 sync/runtime ABI 进入 |

semantic select中的mask不是地址或隐式pointer-width整数。compiler-emitted
`wafer_tx81_mask_move(uint64_t src, uint32_t mask, uint64_t dst, uint32_t elem_count, uint32_t format)`
与CRT header/source使用同一显式`uint32_t`签名；lowering必须在call前证明mask physical address/range适配该
字段：mask必须静态根植于带accepted offset的SPM allocation，view范围不得越过root allocation，完整
physical address range必须适配`uint32_t`，然后才生成i32参数。CRT不得再通过`uint64_t`形参加内部cast
隐藏narrowing。

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
但仍属于 `wafer.instr.*` 的硬件通信调用层。当前fixed-size unicast只在Q16.T committed physical binding、
remote receiver offset、FSM/completion、status ABI与CRT合同闭合后lower到Direct DTE/FSM helper；缺binding、
不一致或超出accepted profile时明确拒绝，不能直接回退到raw-DTE ABI。SCALAR 当前 reserved/stub；
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
- SPM offset、range、bank span由R3.2f写入；后续target-codegen必须从这些facts、committed executable
  bindings和当前memref use-def/view relation派生address/range参数，不能复制成独立placed/access descriptor
  中间协议。后续runtime只实例化verified manifest声明的resource/ABI slots，不读取instruction memref或SPM plan。
- RDMA/WDMA 的 DDR side 使用 `memref<..., #wafer.memory<ddr, layout>>`；DDR memory planning stage 负责
  external allocation contract、declared arena/placement-domain resource、compiler-managed/resident requirement、planned DDR
  ranges 和 constant residency。

R3.2d 只 materialize **unplaced logical descriptor facts**：byte count、stride/iteration、op kind、
tile reduce的canonical reduction tuple/slice relation（以及只有native proof存在时才使用的target `dim` code）、GEMM M/K/N、
batched GEMM `batch_count` 和 batch/m/n/k dimension attrs、elementwise kind 等。这些字段能从当前 IR type、attrs 和
source op verifier 重算。

## 6. Common Instruction Contract

Q32.M已把instruction/resource consumer迁到typed op、value-associated standard effects和custom
`SideEffects::Resource`，并删除`verifyInstructionContract`及Wafer resource-effect interface/record的重复合同。
下述内容是当前实现合同。

instruction结构合同由typed op、ODS/op verifier和标准effect组成：

```text
WaferInstructionOpInterface {
  getInstructionFamily() -> InstrFamily
}

MemoryEffectOpInterface
  -> 对actual SSA value或MLIR SideEffects::Resource的conservative
     Read/Write/Allocate/Free projection
```

`WaferInstructionOpInterface`只保留有多个generic consumer使用的family标记；结构验证直接由op verifier拥有，
不再通过`verifyInstructionContract`重复调用同一verifier。SPM/DDR/Compute/Movement/Communication/Sync
resource继续使用MLIR custom `SideEffects::Resource`，但不再复制成`WaferResourceEffect` record。
buffer role从operand/result和typed op semantics取得，bytes/footprint从type、encoding和descriptor fields重算，
issue/wait/fence completion从SSA token和显式op推导。

`InstrFamily` 是 Wafer enum/interface fact，V0 至少包含：

| enum | hardware invocation family |
| --- | --- |
| `ct` | CT / CGRA queue |
| `ne` | NE queue |
| `rdma` | RDMA queue |
| `wdma` | WDMA queue |
| `tdma` | TDMA queue |
| `dte` | Direct DTE / FSM communication invocation |

family marker只返回当前op可验证的instruction family，不返回planner side table，也不复制全局schedule。
MemoryEffectOpInterface直接关联actual buffer value或custom SideEffects::Resource；它不携带第二份role/index/bytes
record。真正completion由op semantics、SSA token、wait/fence及path-covering verifier共同证明。对
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

所有issue、wait、fence、compute/movement/communication instruction和observable write均为non-speculatable。generic
transformation只能对没有standard MemoryEffect/custom SideEffects::Resource effect且不会触发UB的structural op使用
`Pure`；effect不替代下述completion proof。

completion 是 instruction program 的显式数据流合同。generic `async.call`返回的`!async.token`/
`!async.value`携带所访问buffer root，但task identity是另一项事实；只有path-covering `async.await`，或direct
`async.create_group`经`async.add_to_group`后由`async.await_all`，才能完成对应task。group alias、loop body动态task
加入captured group、SelectLike合并不同task identity和非identity-preserving `scf.for`必须fail closed；
`scf.if` result wait只完成origin确实局限在该branch path的task，不能取消分支前已发起而未被选择的task。

instruction-level IR中的`func.call`不是隐式资源边界。memory planning只把defined private、无副作用/嵌套call、
且storage-shaped result完整解析到formal的helper当作alias SSA；type-erased tensor/memref result仍保留caller root。
其它可能触及SPM/DDR的direct/indirect/external/async call必须有未来显式arena/resource summary，否则下游按动态
execution scope fail closed。静态memory-space type、callee名字或“未解析到alias”不能替代该summary。

DTE issue返回`!async.token`并由匹配`wafer.instr.dte_wait`消费；可能异步的local compute/movement issue要么
返回具有已定义terminal的token，要么进入`wafer.instr.local_fence`明确收口的pending effect set。generic async
completion proof不替代SPM owner的DTE origin/exact-wait proof；SPM当前保守拒绝所有loop-carried async token。
`busytable` 只能作为 target capability /
legality / cost input，不能替代 token、wait/fence、effects 或 terminal drain。每条当前
`wafer.tile.region` exit path 都必须证明没有未消费 DTE token、pending local compute/movement
issue及其 SPM read/write、generic async task，也没有未完成 recv。跨tile-region的SPM buffer、alias和event
必须由显式SSA/control-flow表达并纳入whole-rank plan；SPM planner仍拒绝nested tile-region scope，且不会为
各region独立分配物理arena。

R3.2d 不建模 worker id。`TsmExecute` 的 worker bits、register window 和 packet field 属于
committed instruction 后的 target LLVM call emission。

## 7. ODS-Level Op Contracts

本节是实现时的 ODS 合同。assembly format 可以按 MLIR 可读性微调，但 operand/result/attr
语义不能变。

### 7.1 DMA Descriptor Attributes

RDMA、WDMA 和 TDMA 共同使用 fixed-rank descriptor attrs。字段是 logical descriptor，不是
physical packet：

| attr | type | meaning |
| --- | --- | --- |
| `byte_count` | `I64Attr` | 该 instruction / descriptor实际搬运的payload bytes；cost/resource analysis从该typed字段读取，split cover中不是整个logical tensor或整个cover的bytes |
| `inner_bytes` | `I64Attr` | 最内层 contiguous byte count |
| `src_strides` | `DenseI64ArrayAttr` | source byte strides，长度为 3 |
| `src_iterations` | `DenseI64ArrayAttr` | source logical iterations，长度为 3，值为正数 |
| `dst_strides` | `DenseI64ArrayAttr` | destination byte strides，长度为 3 |
| `dst_iterations` | `DenseI64ArrayAttr` | destination logical iterations，长度为 3，值为正数 |
| `src_offset` | `I64Attr` | allocation root内的非负source byte offset；GatherScatter可选携带；mapped RDMA/WDMA与`dst_offset`成对显式携带（包括0） |
| `dst_offset` | `I64Attr` | allocation root内的非负destination byte offset；GatherScatter可选携带；mapped RDMA/WDMA与`src_offset`成对显式携带（包括0） |

contiguous movement 使用 `inner_bytes == byte_count`，stride 全 0，iteration 全 1。byte stride
必须已经从 element stride 转换完成。硬件 RDMA/WDMA 和 TDMA 都是“最内层连续搬运 +
三层 byte stride/logical iteration”的 descriptor 模型；RDMA/WDMA packetization 时最内层字段会按
dtype element count 写入，R3.2d IR 仍用 byte-level `inner_bytes` / `byte_count` 作为统一
resource/legality 合同。bitpacked BOOL按`inner_bytes * 8`恢复logical element count，并要求乘法结果
适配`uint32_t`；其它format要求`inner_bytes`被target element byte width整除。descriptor 表达不了的
dynamic stride、超过 3 层的静态 stride 或不规则
非连续访问，R3.2d 必须结构化失败，不能生成名字上合法但下游无法 packetize 的 instruction op。
compact RDMA/WDMA不携带`src_offset`/`dst_offset`，从operand root与accepted allocation offset形成地址；mapped
RDMA/WDMA则必须成对携带offset。directional offset不是allocation placement fact，而是descriptor相对allocation root的
access fact；它由typed view和exact transfer proof物化并验证，不能由lowering从planner历史补猜。

### 7.2 RDMA / WDMA

```text
wafer.instr.rdma source to dest attr-dict : type(source) to type(dest)
wafer.instr.wdma source to dest attr-dict : type(source) to type(dest)
```

| op | operands | result | required attrs |
| --- | --- | --- | --- |
| `wafer.instr.rdma` | `source: MemRef<#wafer.memory<ddr, *>>`, `dest: MemRef<#wafer.memory<spm, *>>` | none | current v1：`byte_count`, `inner_bytes`, `src_strides`, `src_iterations` |
| `wafer.instr.wdma` | `source: MemRef<#wafer.memory<spm, *>>`, `dest: MemRef<#wafer.memory<ddr, *>>` | none | current v1：`byte_count`, `inner_bytes`, `dst_strides`, `dst_iterations` |

当前selected tile lowering的`wafer.tile.load/store`仍要求SPM侧为compact Tensor；consumer需要`Cx/NCx`时使用显式
`wafer.tile.materialize_layout`/GatherScatter。这是已实现保守路径，不是终态instruction限制。当前
`StorageLoadOp`还是source→result形态；Q32.R先迁移为explicit source/destination、无result，再由本节按
destination-style lowering消费。

#### Q32.V：Mapped DMA

direct mapped RDMA只在DDR logical view到目标SPM physical byte order的复合映射可被一个或多个
“strided DDR source + sequential SPM destination segment”descriptor精确覆盖时成立；每段显式`src_offset`选择DDR root内
piece，`dst_offset`选择同一SPM root内写入位置。WDMA严格反向，`src_offset`选择sequential SPM segment，`dst_offset`选择
strided DDR destination piece。
RDMA不得带destination strides，WDMA不得带source strides；需要两侧任意strided映射时必须显式选择SPM
GatherScatter/TDMA staging，不能把mapped transfer解释成通用layout-conversion engine。DDR侧可以是compact boundary，
也可以是`memref.subview`/strided memref view；R3.2d从DDR memref layout中恢复静态element stride并转换成byte
stride/iteration descriptor。
R3.2d 不负责把 whole-boundary DDR memref 按 tile shape 切成 subview；该事实必须由 R3.2e
explicit static boundary slice producer、planner candidate evaluation 或 accepted materialization 通过 IR view
显式提供。
每个mapped transfer必须由统一logical-to-physical calculator证明all-and-only coverage、tail、payload、local offset和
两端range；accepted结果是一条或多条显式RDMA/WDMA op，不保存transfer sidecar。动态 view、负 stride、
bit-packed element、超过三层 stride/iteration 或不能静态证明 descriptor 的
情况必须 structured failure，不能从 memref 名字或 shape 猜测。

一个direct cover拆成多条RDMA/WDMA时，每条op的`byte_count`只等于该条descriptor实际搬运的bytes，并独立满足
`byte_count == inner_bytes * product(iterations)`；cover总bytes只能由这些显式op checked求和后与logical valid-domain
payload核对，不能把whole-tensor payload或physical footprint重复写入每条descriptor。padding fill是独立命令，不计入DMA
`byte_count`。

`dst_offset`/`src_offset`在target lowering中分别checked加到各自source/destination allocation root/base，
形成DDR与SPM两端最终`uint64` address；SPM侧local offset不扩RDMA/WDMA CRT descriptor签名。两者仍必须
留在Instr IR直到address derivation，不能提前折入`wafer.spm.offset`或memref allocation fact。

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
wafer.instr.reduce #wafer.instr_reduce_kind<kind> input into dest attr-dict
    : type(input) into type(dest)
wafer.instr.convert #wafer.instr_convert_kind<src_dst> source into dest attr-dict
    : type(source) to type(dest)
```

| op | operands | result | required attrs |
| --- | --- | --- | --- |
| `wafer.instr.fill` | `dest: SPM memref`, `value: scalar` | none | attr缺省为`logical_valid`且只接受Tensor；显式`physical_footprint`按physical elements计数，BOOL按physical bytes×8计数，scalar按原始storage bits写入32-bit ABI字段 |
| `wafer.instr.elementwise` | `inputs: Variadic<SPM memref>`, `dest: SPM memref` | none | `kind: #wafer.instr_elementwise_kind`; operands/dest same-shape；`indexing_maps`不属于terminal op合同 |
| `wafer.instr.reduce` | `input: SPM memref`, `dest: SPM memref` | none | `kind: #wafer.instr_reduce_kind`, `dim` target reduce code；init operand/`init_value`不属于terminal op合同 |
| `wafer.instr.convert` | `source: SPM memref`, `dest: SPM memref` | none | `kind: #wafer.instr_convert_kind`; required `zero_point` for INT8->FP kinds, required `rounding_mode` for rounding wrapper kinds, no extra attrs for plain kinds |

`wafer.instr.fill`的current v1 Tensor行为从logical element count checked派生`elem_count`，ODS不携带domain attr。

Q32.V已增加typed `#wafer.fill_domain<logical_valid|physical_footprint>`，其中`physical_footprint`表示从dest view base
连续覆盖`computeWaferPhysicalTensorInfo`给出的完整physical bytes：非BOOL要求
footprint可整除format element bytes并取商，BOOL按`bytes * 8` checked得到bit count；count、range和target field必须可表示。
它是建立Cx/NCx padding与bitpacked unused bits `KnownSplat`的唯一full-fill路径；若底层Memset不能按该count完整写入则row非法。
TargetCall仍消费明确`elem_count`，无需读取planner state；formal/SystemC语义必须证明scalar到canonical raw element的
映射，无法唯一确定的NaN/-0/conversion tuple不得建立KnownSplat。当前Cx/NCx/BOOL physical-fill尚不是v1 capability；
该domain已闭合Tile/Instr/TargetCall/model正负纵向；未命中typed约束的组合仍fail closed。

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

当前target LLVM elementwise emission只按kind、operand/dest地址、element count和单一format选择unary/binary vector
wrapper，不消费`indexing_maps`。因此tile→instruction必须先把所有map展开为`gather_scatter`/其它movement和same-shape
operands，再生成无map的terminal `wafer.instr.elementwise`；identity map也strip，避免重复事实。ODS/verifier拒绝任何残留
`indexing_maps` attr，target conversion只做defensive check，CModel不得读取该attr补做broadcast。

当前target LLVM reduce emission不传init，所以tile→instruction先验证tile-level SSA `init`与`init_value`互斥且类型一致。
Q0.L source-reduce correctness baseline把可表示的init写入result-shaped accumulator，按canonical lexicographic reduction
tuple依次materialize同shape slice，并以与source combiner精确对应的map-free elementwise op在两块accumulator间ping-pong；
correctness-first基线在fill、每个slice movement、每次elementwise更新和final movement后均形成显式completion；只有当前
IR可验证的同engine顺序或dependency relation才允许合并。不能被typed fill表示的dynamic init、没有exact elementwise
mapping的combiner（当前包括avg）或超过checked expansion budget的case在effect前拒绝。tile→instruction先形成只在本次
lowering存活的完整expansion，checked统计实际拆分后的engine command和completion op；每个accepted rank的独立上限为4096。
这个static reduce terminal-op预算不复用tasks/06同为4096的candidate materialization counter，也不是target/workload语义。
该序列不使用native reduce，因此不会把
`native_reduce(xs) op init`误当成source-order `(((init op x0) op x1)...)`。

terminal `wafer.instr.reduce`的ODS移除optional init operand，verifier拒绝`init_value`等残留attr。它只表示无init字段的
target-native leaf；production source path只有在未来compiler-owned target policy对完整value domain证明identity、combiner、
order、rounding和special-value等价后才可用它优化上述composite。Q0.L不能引用尚未板端校准的Q22
`NumericSemanticsProfile`、host默认或有限corpus证明等价。保留在Instr IR但不进入CRT call的init不是合法production语义，
CModel不得补偿。

### 7.5 GEMM

```text
wafer.instr.gemm lhs, rhs into dest attr-dict
    : type(lhs), type(rhs) into type(dest)
```

| op | operands | result | required attrs |
| --- | --- | --- | --- |
| `wafer.instr.gemm` | `lhs: SPM memref`, `rhs: SPM memref`, `dest: SPM memref` | none | current v1：`m`, `k`, `n`及optional batched GEMM attrs；implicit normal/normal |

current v1 ODS没有orientation attr，只接受canonical normal/normal relation；target call只携带
`batch_count/M/K/N/format`。stored shapes、M/K/N、batch和physical footprint仍必须精确匹配。

#### Q32.V：Oriented GEMM

orientation扩展用与tile层共用的封闭typed enum `#wafer.gemm_orientation<normal|transpose>`，不使用raw packet bool，也不为
NN/NT/TN/TT建立四个op。对每个batch slice，normal lhs stored shape为`[M,K]`、transpose lhs为`[K,M]`；
normal rhs为`[K,N]`、transpose rhs为`[N,K]`；destination始终为`[M,N]`。plain form的lhs/rhs/dest都必须恰好rank 2、
使用`#wafer.memory<spm, cx>`且不得携带batched attrs；batched form都必须恰好rank 3、只有一个canonical leading batch
dimension、使用`#wafer.memory<spm, ncx>`，并按相同orientation relation携带完整batched dimension attrs，
`batch_count`必须等于该batch维。stored shape、orientation、M/K/N和physical footprint必须同时验证，不能通过flag
把错误bytes重新解释成另一个operand。
rank 1和rank >= 4均结构化拒绝。target call不携带自由rank/layout字段；若把多个batch维的
rank-4 tensor直接flatten，调用边界会丢失NCx per-batch bank boundary，因此不能由CModel猜测或静默压平。上游若未来
需要多batch维，必须先以显式reshape/layout movement canonicalize为单batch维并重新验证physical bytes。

当canonical rank-3 form的`batch_count=1`时，`NCx[1,M,C]`与plain `Cx[M,C]`的footprint及全部logical-element offset
相同；回归覆盖两个完整channel block与`C0` tail。因此当前target call在`batch_count=1`处擦除source rank不会产生
byte-order歧义。Fused bias、activation、quant、psum accumulation policy 和 sparse / INT8 variants 不属于 R3.2d V0。

目标Instr IR在选择versioned oriented ABI时必须显式携带两个orientation字段且不得依赖default；旧v1 lowering仍只能接受
normal/normal。orientation进入tasks/14的typed target-call/ABI capability和tasks/17的
`NumericCommandKey`/qualification identity。typed ABI、compiler emission和SystemC qualification闭合后可进入model-only
profile；真实board provider还必须命中对应environment的board-supported allowlist，二者不能混称。

plain GEMM还只要求lhs/rhs/dest element type相同，target CRT call只传一个format；IR没有product、accumulator、
逐MAC rounding、FMA或reduction-order字段。若这些行为是program-selectable，必须先扩typed tile/instruction op及CRT ABI；
若它们是target revision固定行为，则target revision和完整command tuple必须在execution capability中唯一映射到一个
`NumericSemanticsProfile`。未校准的f16/bf16 narrow/wide、TF32和integer候选不能由lowering/CModel按dtype猜测。

generic online reduction和non-GEMM FMA contraction因此不属于current Q32 instruction contract。Q32.N必须先增加明确source
predicate、selected state/fused op、对应Instr/TargetCall/必要ABI和SystemC数值纵向；target固定GEMM FMA profile不能被source
rewrite当作通用contract许可。

floating rank-local algebraic reassociation/reduction-tree rewrite也不通过Instr attr恢复：Q32.N先要求production source无损
携带standard permission，再把选择物化为显式SSA DAG/SCF；instruction lowering只消费该actual DAG，不读取隐藏order或
“已重结合”标志。StableHLO collective的ordered tree是另一条语义：只要实际左右子树使中序遍历保持`rank_group`，
它就是logical collective允许的实现次序，不依赖该rank-local fast-math permission。

#### 7.5.1 Fixed Cx/NCx Encoding Absorption

current target没有独立`vector_width`、packing mode/factor或packing ABI字段。Cx/NCx packing由`TargetProfileId`、dtype、
typed encoding、shape/tail唯一决定。Q32.M若在actual clone中删除前置Tensor↔Cx/NCx `materialize_layout`/GS，
`wafer.instr.gemm`直接消费同一Cx/NCx memref；Instr和TargetCall仍只携带既有format/shape字段，不记录“已吸收”标志。
verifier必须用shared physical geometry核对block、C0 tail、padding、valid lane和footprint，不能从缺失movement反推packing。

#### 7.5.2 Low-Precision Instructions

```text
wafer.instr.quantized_gemm lhs, rhs (, scale_p, scale_n)? into dest attr-dict
wafer.instr.mxfp_decode packed, scale, scratch into dest attr-dict
```

`wafer.instr.quantized_gemm`只由verified `wafer.tile.quantized_gemm`和matched native
`LowPrecisionComputeCapabilityV1`产生。operands/effects显式覆盖lhs/rhs/dest及enabled scale buffers；attrs固定M/K/N、
left/right batch、transpose、input/output target format、q0/q1、left/right zero point、closed scale mode和typed
`QuantStorageAbiProfileId` ref。q0/q1和zero point必须在target证明范围内，accumulator/result/saturation relation必须与
上游descriptor一致。首个signed-i8 profile只允许mathematical zero point `[0,127]`并checked转换为同值raw field；
negative zp、128..255或two's-complement reinterpretation必须由另一个有golden/board证据的capability显式开放，不能
static_cast。首个native profile的destination是INT8且output zero point固定为0；i32仅为internal accumulator，f16
结果必须由后续显式dequant/convert op产生。bias、activation、sparse、implicit psum、其它output zero point或
capability未声明的granularity非法。
V1 command中的rounding/saturation是profile固定implicit hardware policy的冗余防错编码，不是caller-selectable
packet field；CRT只接受与profile常量完全相等的值。首个planned native profile还要求`scale_mode=none`和两个scale
operands absent；axis scale在exact formula/indexing/table dtype证据形成新profile前target-illegal。
`wafer.instr.convert`的single-source zero-point不是该op的替代品。

`wafer.instr.mxfp_decode`显式记录registered FP8 encoding、packed/scale/destination storage descriptor、element count、
block shape/count、tail和NaN/Inf/subnormal/overflow policy；operands是packed source、E8M0或profile允许的typed scale、
BF16/FP16 destination和exact scratch memref。它的MemoryEffects必须包含source/scale/scratch read、scratch/destination
write和composite issue/local-completion；decode completion支配任何destination consumer，scratch/destination在该
completion前不可复用。TX81首发只允许profile证明的32-value block/software decode+scale组合；其它block/encoding
结构化失败。

两种op都不得保存wrapper symbol或旧`__*` helper名。target LLVM只按typed profile选择Wafer-owned fixed ABI；在
profile、geometry、SPM range、TargetCall/CRT conformance、device-link任一gate完成前，这些op可以用于
parser/verifier negative/plan测试，但必须在production target legality中失败。

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

`#wafer.instr_conv_kind`保留Conv / Depthwise / BackwardConv packet family枚举，但当前production verifier
只接受ordinary `conv`：input/weight/output attrs必须逐项匹配memref shape，并证明batch/channel、kernel、
stride、dilation、pad/unpad与output的精确关系。Depthwise/BackwardConv尚无各自channel/group、weight和
output relation，必须以`unsupported_target_geometry`失败；bias、scale、activation、sparse、INT8 quant
和psum policy也不能作为隐式default藏在target lowering里。

`wafer.instr.pool` / `wafer.instr.unpool` 覆盖CT Pool/UnPool wrapper family，并证明source/dest attrs与
memref shape一致以及batch/channel、kernel/stride/pad的精确输出关系。普通pool kind只有
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

`wafer.instr.tdma_data_move` 的V0 production surface只表达wrapper-level `pad` / `img2col`；两者的
source/dest attrs必须匹配memref shape，并分别证明pad或kernel/stride/pad的输出关系。
mirror、transpose、rotate、NCHW/NHWC 和 TensorNom 这类 transform-like DataMove kind 虽然有
public wrapper/header 证据，但 V0 不把它们作为 production target surface；普通 copy、layout segment
movement、static slice / broadcast / transpose / mirror / rotate 的可证明 byte movement 由 compiler
lowering 展开成 `wafer.instr.gather_scatter`，无法表达时由 lowering 结构化失败。`mirror`
的 `axes` 是被翻转的 logical axis 集合；`rotate90/180/270` 的 `axes = [a, b]` 是有序旋转平面，
`rotate90` 表示在该平面上的 clockwise quarter turn，`rotate270` 表示反向 quarter turn，
`rotate180` 保持 shape 不交换但翻转两个轴。

`wafer.instr.peripheral` IR kind覆盖argmax、argmin、factorize、bilinear、lut16、lut32、rand_gen、elem_mask和当前
fail-closed的count enum。kind决定input/dest arity；`elem_count`必须等于primary input的
静态element count，并按kind证明dest/secondary input capacity；argmax/argmin的scalar value/index dest
各只有一个element，LUT table capacity必须等于`lut_elem_count`。argmax/argmin 的第一个 dest 是
value，第二个 dest 必须是 i32 index。`bilinear` 要求 `source_shape` / `dest_shape`；
`lut16` / `lut32` 要求 `lut_elem_count`；`elem_mask` 要求 `scale`、`probability` 和
`rounding_mode`。opcode 176 `bitcount`只有enum证据、没有public wrapper证据，不进入当前IR kind。
factorize虽有IR enum与kind-specific verifier，但缺少精确production semantic profile；target conversion
必须拒绝它，repo-local CRT header/source不提供对应prototype/definition。

#### Later Q3.6：Count writeback typed closure

```text
Pipeline position:
- Upstream artifact / IR: future target-profile revision下的verified instruction module中显式存在的
  `wafer.instr.peripheral<count>`及其typed SPM source/destination；该机械入口不表示source/provider已经具备选择Count的资格。
- Current stage responsibility: verify Count arity、source format/element-count、destination memref shape/layout及Instr层
  read/write/synchronous-completion effect；本stage不创建TargetCall或解释CRT ABI。
- Output artifact / IR: verified `wafer.instr.peripheral<count>`及其typed operands/attrs/effects，仍处于instruction IR。
- Downstream consumer: 14的target conversion/decoder形成typed TargetCall transaction；只有真实runtime/model consumer
  需要时，14-17才同批扩展publication、package readback和execution admission。
- User-level driver / named pipeline: Q3.6最终复用existing wafer-compile source-to-bundle pipeline且不增加Count-only入口；
  在此之前只允许compiler-owned complete instruction/TargetCall transaction test seam。production source admission要求
  source IR明确表达predicate、独立golden和实际model kernel，机械合同覆盖不绕过该gate；Q3.6 positive机械输入来自
  compiler-owned complete instruction/TargetCall transaction test seam，不冒充source-produced vertical。
- Explicit non-goals: no host scalar return, raw register IR, guessed Count predicate, model fallback or board claim.
- Completion gate: later Q3.6的11/14-17 mechanical gate及synchronous-writeback effect原子通过；source/model admission
  另要求明确predicate、closed format语义、独立golden和实际kernel。该gate独立于Q32/Q32.V，也不改变current Instr
  completion状态。
```

Count继续复用通用`wafer.instr.peripheral`，不新增专用op或SSA scalar result。终态arity为`1 input + 1 dest`：dest必须是
static compact `memref<1xi32, #wafer.memory<spm, tensor>>`，其中bits按独立的raw unsigned-u32 writeback contract保存；它不是
`LogicalFormat::U32`，也不建立`CT x U32` input/output encoding row。`elem_count`等于source logical element count并属于closed
boundary class `positive_u32 = [1, UINT32_MAX]`；Count禁止`source_shape`、`dest_shape`、`lut_elem_count`、`scale`、
`probability`和`rounding_mode`。

首版source也必须是static `#wafer.memory<spm, tensor>` compact-contiguous view；其logical element product恰等于
`elem_count`，physical span由14 format registry checked计算为`elem_count * storage_bytes(format)`并从source起址连续覆盖。
Cx/NCx、strided/holey subview、padding-bearing view、dynamic shape/offset/stride或span overflow全部在Instr/target effect前拒绝；
contiguous subview只有其root-relative byte range可exact证明时合法。source与4-byte destination range必须proven-disjoint，exact/
partial overlap和unknown alias一律pre-effect reject；当前机械证据不授权依赖“硬件可能已先读完”的alias优化。source alignment按
format storage width、destination按4-byte要求进入09/14 fresh address gate。

本层只证明single-element compact i32 SPM destination及其4-byte physical footprint；最终offset存在时还必须满足4-byte
alignment。14唯一拥有`writeback_raw_u32`的target byte-order/store合同并在address lowering后验证alignment/range，17只消费
该typed target contract和地址绑定，不能从u64地址反推本层memref shape、dtype或layout。

later Q3.6的synchronous writeback必须由`wafer.instr.peripheral<count>`自身的typed effect/completion合同和exact
TargetCall语义表达，不能按op名恢复，也不需要再造versioned completion-property registry。若wrapper在返回前等待local
compute/movement并写入destination，Instr effect、path verifier和target conversion直接验证这一事实；consumer、alias和
lifetime据此排序。普通terminal local fence可以保守收口其它issue，但不能替代Count op自身的同步合同。

当前资料只证明opcode/wrapper和raw low-u32 writeback，不能证明Count predicate、特殊值或format语义。因此current
source interface不得产生Count candidate，target/model/board全部pre-effect拒绝。Q3.6只有取得明确source semantics、
typed TargetCall、独立golden和实际model consumer后才开放对应普通capability row；不预先冻结completion wire ordinal、
execution digest、qualification record或package schema。ArgMax/ArgMin已有wait-before-store实现只能作为同步effect的
代码证据，不能外推Count numeric语义。

## 8. Lowering Rules

R3.2d 应实现为 MLIR DialectConversion：

- illegal：`wafer.tile.load`、`wafer.tile.store`、`wafer.tile.materialize_layout`、
  `wafer.tile.fill/gemm/elementwise/reduce` 和 tile movement ops。
- legal：`memref.alloc`、standard memref view ops、`wafer.instr.*`、
  `wafer.instr.local_fence`、`wafer.tile.region` container、`scf.if` / `scf.for` container
  和必要 scalar/support op。
- no type conversion for Wafer tagged memref values。
- conversion failure 必须结构化返回给 planner；rejected instruction IR 不进入 committed 主线 IR。

当前V0与终态扩展的mapping边界：

下表collective三行记录instruction-level materialization合同，不构成instruction层的算法选择合同。Q32.M已删除
public pass schedule option/parser；production candidate owner从同一tile-region parent建立All-Gather Direct/Ring、
Reduce-Scatter Direct/Ring和All-Reduce Ring/Tree完整clone并分别执行instruction/SPM/DDR/verifier/cost gate。
IR-local conversion入口使用显式typed options重放单个参数点，不保存selector或algorithm attr。Q36把这些普通C++参数
改为从current topology/mesh派生的peer/rank order及Tree root/parent/left-right children；最终Instr IR仍只包含实际
peer edge、message、local work和completion。

| target-abstract op | instruction-level lowering |
| --- | --- |
| `wafer.tile.load` | Q32.R后consume destination-style source/dest；current capability只处理compact Tensor。Q32.V mapped extension从两端typed views/encoding、TargetProfileId和08 exact transfer proof导出direct cover，并发射显式`src_offset`/`dst_offset`的RDMA；若consumer要求known padding，先fill完整destination再发valid segments；staged alternative必须已显式物化为Tensor+GS payload IR |
| `wafer.tile.store` | consume destination-style source/dest；current capability从compact Tensor发射WDMA。Q32.V mapped extension从两端typed views/encoding、TargetProfileId和08 exact transfer proof导出direct cover，并发射显式`src_offset`/`dst_offset`的WDMA；staged alternative必须已显式物化在payload IR，随后用local fence闭合source lifetime |
| `wafer.tile.materialize_layout` | ensure / create destination memref with requested marker; compare source/result logical element to physical byte mapping through the unified physical layout calculator; coalesce adjacent byte segments, pack regular segments into up to three stride/iteration levels, and emit one or more `wafer.instr.gather_scatter`; do not require source/result physical byte counts to match; structured failure only when static logical movement cannot be represented by V0 descriptors |
| `wafer.tile.fill` | current v1只对Tensor logical-valid domain生成无domain attr的`wafer.instr.fill`；padding/physical-footprint初始化已由Q32.V增加typed Instr/TargetCall字段并闭合count/raw-value纵向 |
| `wafer.tile.gemm` | ensure/create selected aligned SPM physical versions；current v1只在normal/normal relation成立时生成无orientation字段的`wafer.instr.gemm`。若operand/result已是合法Cx/NCx，直接消费该encoding且不插入packing字段；本lowering不判断历史上是否删除过layout/GS，tasks/06 Q32.S/G集成证据从winner readback证明absorption。Q32.V把tile-level typed orientation无损写入versioned Instr op；不能从shape或op名恢复flag |
| `wafer.tile.elementwise` | materialize every input indexing map into explicit movement/same-shape operands; strip even identity maps; ensure/create destination; map non-select kind and emit map-free `wafer.instr.elementwise`; semantic select lowers to false-copy `gather_scatter` + `bit2fp` + `mask_move`; reject if a map is unrepresentable |
| `wafer.tile.reduce` | verify init operand/attr mutual exclusion/type; fill result-shaped accumulator; enumerate reduction tuples in canonical lexicographic order; materialize each non-reduced slice to result shape and map an exact combiner to map-free elementwise ping-pong accumulators with explicit completion; move final accumulator to destination; reject dynamic-init/combiner/budget cases not representable by the typed baseline; do not emit native reduce without a compiler-owned full-domain equivalence proof |
| `wafer.tile.copy` | ensure / create destination SPM memref; emit one gather_scatter; replace result with dest memref |
| `wafer.tile.extract_slice` | create destination SPM memref; enumerate the static slice result logical domain, map each result index through offsets/sizes/strides back to the source logical index, compute physical byte offsets with the unified Wafer layout calculator, coalesce adjacent byte segments, pack regular segments into up to three stride/iteration levels, emit one or more `wafer.instr.gather_scatter`, and replace result with dest memref |
| `wafer.tile.insert_slice` | create destination SPM memref; first copy the original destination payload into the result via logical-to-physical segments, then enumerate source logical indices and overlay them into the statically described destination slice via packed `wafer.instr.gather_scatter` descriptors; replace result with the new memref |
| `wafer.tile.broadcast` | create destination SPM memref; enumerate the static result domain, map source dims through `dimensions`, compute source/result physical byte offsets, coalesce adjacent segments, pack regular segments into up to three stride/iteration levels, emit one or more `wafer.instr.gather_scatter`, and replace result with dest memref |
| `wafer.tile.transpose` | create destination SPM memref; enumerate the static result domain, invert `permutation` to source logical indices, compute source/result physical byte offsets, coalesce adjacent segments, pack regular segments into up to three stride/iteration levels, emit one or more `wafer.instr.gather_scatter`, and replace result with dest memref |
| `wafer.tile.reshape` | identity replacement when types are identical; otherwise preserve source/result canonical linear element order and reinterpret result multi-indices through the new shape; compact `tensor/ntensor` reshape lowers to a verifier-legal standard memref view because compact physical bytes already follow that linear order; `Cx/NCx` reshape first compares same-linear-element source/result physical byte offsets with the unified physical layout calculator, materializes a destination memref and emits packed `wafer.instr.gather_scatter` descriptors only when the physical mapping or required footprint changes; structured failure only when the static reshape movement plan cannot be represented by V0 descriptors |
| `scf.if` / `scf.for` | preserve the structured control-flow op; recursively legalize executable target-abstract ops in each nested region; keep scalar and memref yields explicit |
| `wafer.tile.all_gather` | requires matching `tensor/ntensor` SPM layouts and infers the unique gather axis from compact local/gather buffer shapes. `ring` copies the local chunk into the local gather slot, fences, then forwards one slot per round around the topology-derived rank order. `direct` uses the deterministic cyclic order of semantic group indices to send the local slot to every other logical rank while receiving each peer chunk into its result slot；it does not invoke topology Ring search. |
| `wafer.tile.all_reduce` | requires matching `tensor` SPM buffers and sum/max/min reduce kind. `ring` is standard chunked reduce-scatter plus all-gather: each round communicates only one nonzero typed chunk and reduces only that chunk; the final result is assembled from all reduced chunks. `tree` uses a topology-derived ordered binary tree whose inorder traversal equals `rank_group`: each node combines left-subtree partial, local operand and right-subtree partial in that order, then broadcasts the final accumulator down the reverse tree. This ordered Tree is valid for floating collectives without fast-math. The current cyclic Ring is limited to integer element types until production IR carries permission for its leaf permutation. Both forms materialize all local reduction and DTE-read/consumer fences explicitly. |
| `wafer.tile.reduce_scatter` | requires a full `tensor` SPM input whose scatter `axis` size is `group_size * result_axis_size`, plus matching local-slot recv/result buffers. `direct` sends each destination-owned input slot to its owner and accumulates received contributions in `rank_group` order for the local slot. `ring` performs one typed chunk transfer/reduction per round along the selected rank order. The current Ring is integer-only for the same leaf-order reason. Non-contiguous slots require explicit pack/unpack; an unrepresentable extra candidate fails without weakening the direct baseline. |

旧all-reduce ring曾每轮发送full buffer；它只能作为Q36修复前的实现事实，不能继续满足本表的Ring合同或用于
算法优劣证明。equal-split all-to-all和collective-permute虽然在更早的structured→tile-region rewrite直接产生
Instr DTE op，也遵守相同completion要求：remote insert/recv、local copy/fill与后续consumer之间必须有显式
wait/fence，不能依赖block order或terminal region偶然收口。

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

R3.2d.5 static movement plan的host构造复杂度不属于IR协议，但必须保持可扩展且与统一physical mapping等价：

- 静态logical domain按canonical lexicographic次序遍历；实现可以用一次校验后的odometer增量维护multi-index，不能要求
  每个element重新用除法/取模从linear index反线性化。source/result index relation和segment顺序不变。
- lowering复用tasks/08拥有的`WaferStaticPhysicalOffsetCalculator`，在构造时缓存从memref type和
  `computeWaferPhysicalTensorInfo`派生的shape stride、Cx/NCx full/tail block常量，用于同一conversion内的hot loop；
  `computeWaferPhysicalElementByteOffset`仍是规范physical mapping，focused differential必须覆盖Tensor/NTensor、Cx/NCx、
  C0 tail和越界拒绝。缓存只是可重算analysis，不进入IR、attr、package或全局side table。
- per-element index scratch由当前lowering invocation拥有并复用；不能让buffer名、地址或workload shape成为fast-path语义。
  任一overflow、dynamic/invalid shape或无法证明的layout继续structured failure，不允许为了性能跳过range/verifier检查。
- 性能优化完成证明必须比较优化前后packed source/dest descriptor和最终target command，而不仅是wall time；7B scale gate还要
  保持all-rank package、transaction/SystemC delta、numeric counters和完整PyTorch differential。

R3.2d V0 communication coverage：

- `wafer.tile.all_gather` 已能在 `rank_group`、group-local `local_rank`、`group_size`、`bytes`
  和静态 compact `tensor/ntensor` SPM buffer shape 均可验证时 materialize fixed-size unicast schedule。
  typed conversion参数可物化ring或direct；accepted result始终是explicit `wafer.instr.dte_*` body，
  不保存schedule attr。production candidate owner独立建立两种actual clone。
- `wafer.tile.all_reduce` 已能在 `rank_group`、group-local `local_rank`、`group_size`、`bytes`、
  `tensor` SPM buffer type 和 sum/max/min reduce kind 均可验证时 materialize fixed-size unicast schedule。
  typed conversion参数可物化ring或tree；production candidate owner独立建立两种actual clone。
  `tree`由current topology/placement和`rank_group`通过interval DP派生minimum-total-shortest-hop ordered binary
  tree；其中序遍历严格等于`rank_group`，root不固定，left/local/right reduction后沿reverse tree broadcast。
  reduction不藏进DTE side effect；每个reduce step都先wait DTE token，再用`wafer.instr.elementwise`做本地累计。
  ordered Tree可用于floating collective；current Ring只对integer element type生成，浮点Ring等待显式numeric permission。
- `wafer.tile.reduce_scatter` 使用 full input + local slot result 表示。tile-region lowering 不再预先把
  input 截成当前 rank 的 slot；instruction lowering 从 full input 的 per-target slot `memref.subview`
  直接派生 p2p send source，并在 wait 后显式累计 recv contribution。Direct是phase-ordered
  all-to-owner baseline；Ring按topology-derived cycle执行`group_size - 1`轮typed chunk归约。两者均不保存
  全局plan attr；current浮点输入只保留按`rank_group`次序累计的Direct。

R3.2d may generate multiple instruction ops for a single target-abstract movement op, but it must not write a
global schedule attr. The instruction sequence is the region body itself.
Nested `scf` regions are part of that body: R3.2d rewrites their executable contents under MLIR region
scoping rules, but it does not lower them to hardware branch/loop instructions.

## 9. Failure Contract

R3.2d failure is a legalization result, not an IR artifact. A rejected legalization attempt may carry
diagnostics to the closed-loop planner or debug pass, but rejected instruction IR is discarded。任一 rank、
task、traversal scope 或后续 whole-rank gate 失败时，complete variant clone 整体丢弃；不能提交已
legalize 的其它 instruction fragments。

必须结构化失败的情况：

- non-ranked or dynamic-shaped memref where V0 needs static byte/stride computation.
- unsupported Wafer memory attr, address space or physical layout marker for an instruction family.
- unsupported dtype, including relation/elementwise/convert pairs not mapped to CT V0.
- movement descriptor cannot be represented with buffer-local offsets, `inner_bytes` and three
  stride/iteration levels.
- current compact DMA或GatherScatter的typed view/descriptor/range无法证明；Q32.V mapped DMA direct cover需要
  非顺序SPM侧、两侧strided、coverage有hole/overlap、local offset/range溢出，或descriptor序列不能all-and-only覆盖logical relation。
- `Cx/NCx` materialization with retained `C0` tail cannot be split into separately representable
  full-block and tail GatherScatter descriptors.
- static slice/insert/broadcast/transpose whose logical index relation or physical byte offsets cannot be
  converted into one or more `gather_scatter` descriptors.
- unsupported control-flow op, multi-block region, or nested region whose executable body cannot be fully
  legalized under the same instruction conversion rules.
- current NE GEMM dimension/batch attrs不能精确匹配stored operand/result types；Q32.V oriented row的两个typed
  orientation attrs还必须匹配stored shapes并命中selected target profile/Kernel Runtime ABI tuple。
- tile reduce dimensions无法形成static canonical tuple/slice movement、combiner没有exact elementwise mapping、init不被typed
  fill表示或checked expansion budget超限；native optimization另在`dimensions`不能映射target `dim`时拒绝。
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
  strides and positive byte counts；`byte_count == inner_bytes * product(iterations)`，且RDMA/WDMA的
  bitpacked BOOL `inner_bytes`必须不超过`UINT32_MAX / 8`，按`bytes * 8`恢复logical element count；
  非BOOL `inner_bytes`必须能被target data-format element byte width整除。两条路径都必须checked，不能在
  CRT中溢出乘法或截断除法。
- 每条descriptor的`byte_count`只等于该instruction实际搬运bytes并独立满足payload等式；split cover按checked sum与
  compact tensor或statically described slice/broadcast/transpose的logical valid-domain payload做all-and-only核对。
  Cx/NCx padding span由`computeWaferPhysicalTensorInfo(memrefType)`和target policy推导，不复制进任一DMA `byte_count`。
- logical shape 到 physical footprint、view/root/descriptor range、offset arithmetic 和 target field
  narrowing 都通过同一个 shared physical geometry/range/narrowing verifier；instruction、SPM/DDR
  planning、target/package lowering 复用该 verifier。每次 narrowing 都必须证明源值在目标字段范围内；
  silent i64-to-i32 或 size-to-packet-field truncation 非法。
- RDMA只允许DDR source stride/iteration，WDMA只允许DDR destination stride/iteration。current v1从compact operand root和
  accepted allocation offset计算SPM sequential range与DDR strided range，不要求ODS中不存在的directional offset字段。
  Q32.V mapped DMA要求`src_offset`/`dst_offset`显式存在（包括0）、分别相对各自root，并证明descriptor序列对logical
  relation的all-and-only coverage；任何版本都不存在双侧任意stride的隐式合法化。
- logical element type或`Data_Format` enum存在不证明任一engine可编码该dtype。production target verifier必须查询
  target-profile×instruction-family×format `TargetFormatEncodingRecord`并引用tasks/08 layout profile；当前typed convert kind可选择TF32 wrapper与
  RDMA/WDMA/fill/elementwise/reduce/GEMM通用format encoder缺TF32是两个独立legality row。UINT/64-bit direct DMA等
  未有engine encoding证据的row在shared registry闭合前target-illegal。
- NE GEMM and CT reduce require supported aligned layout marker, dtype and rank. current v1 plain GEMM按implicit
  normal/normal relation匹配stored shape与M/K/N/batch；Q32.V oriented tuple只有在typed orientation字段、versioned
  target ABI和capability row同时匹配时合法。每个command tuple还必须唯一映射numeric semantics profile；terminal CT reduce has no init operand/
  attr，任何残留字段target-illegal；Q0.L source reduce必须更早lower为有序fill/movement/elementwise composite或拒绝，native
  reduce只有compiler-owned full-domain equivalence proof后才可进入production。
- fixed Cx/NCx absorption不增加Instr字段；verifier只从current Instr/operands证明existing encoding/profile/shape/tail完全决定
  packing，并核对geometry、valid/padding lane、range、alias、effect和completion。历史movement是否消失及删除前后等价性由
  tasks/06 Q32.S/G和tasks/16集成gate证明，不由Instr verifier反推。
- `quantized_gemm`要求exact matched low-precision capability/profile、signed INT8 storage、legal q/zp/scale mode、
  scale operand range和accumulator/saturation proof；plain GEMM不能携带这些fields。
- `mxfp_decode`要求packed/scale/destination/scratch types与block/element/tail policy一致，packed capacity和scale
  count exact，decode/local-completion支配consumer与scratch reuse；不能使用native FP8 format假设。
- relation/elementwise bool storage uses logical `i1`; physical byte size remains derived, not stored.
- `wafer.instr.elementwise` entering target conversion carries no indexing-map attr and requires same-shape operands；all
  permutation/broadcast/identity maps must already have been materialized/stripped. `wafer.instr.elementwise` /
  `wafer.instr.reduce` use instr-level target kind attrs only；generic
  `#wafer.elementwise_kind` / `#wafer.reduce_kind` on instruction ops is verifier-illegal.
- `wafer.instr.convert` uses `#wafer.instr_convert_kind` only；source/dest dtype is derived from the
  convert kind and checked against memref element types. It does not accept free-form `src_dtype` /
  `dst_dtype` attrs as instruction semantics. Kind-specific `zero_point` / `rounding_mode` attrs are
  required or forbidden according to the public wrapper signature group.
- `wafer.instr.conv` requires aligned SPM input/weight/dest memrefs, matching element types, attrs that
  exactly match the three memref shapes, and an exact operator relation. Current production accepts only
  ordinary conv；depthwise/backward conv remain `unsupported_target_geometry` until their distinct
  channel/group and output equations are defined.
- `wafer.instr.pool` / `wafer.instr.unpool` require aligned SPM operands, matching value element type,
  source/dest attrs equal to memref shapes, and exact batch/channel/spatial output equations. Indexed pool
  index dest must use i32 element type. Unpool `unpool` / `mask` require scalar uint32 `index`; `avg` forbids it.
- `wafer.instr.tdma_data_move` requires SPM source/dest memrefs with matching element type, rank-4
  positive source/dest descriptors equal to memref shapes, and kind-specific descriptor attrs/equations.
  V0 production target only accepts `pad` with exact pad output relation and `img2col` with exact
  kernel/stride/pad output relation. Transform-like kinds
  `mirror/transpose/rotate*/nchw2nhwc/nhwc2nchw/tensor_nom` are verifier-legal only as explicit
  pre-lowering/imported IR and must be materialized to `gather_scatter` before target LLVM or package
  export. `transpose` requires `permutation`; `mirror` requires non-empty unique `axes`; rotate kinds
  require exactly two unique `axes`; `axes` entries must be within the source/dest logical tensor rank,
  and mirror/rotate operands must be rank-compatible. Unrelated attrs are rejected per kind. These kinds are not
  production target lowering input unless a future target path is explicitly enabled.
- `wafer.instr.peripheral` verifies kind-specific input/dest arity, uint32 `elem_count`, primary input
  exact element count and every kind-specific secondary/dest capacity；arg
  peripheral ops require a same-dtype value dest plus an i32 index dest. `bilinear` requires
  source/dest shape attrs; LUT kinds require `lut_elem_count`; `elem_mask` requires `scale`,
  `probability` and `rounding_mode`. Count的终态只接受一个source和单元素i32 SPM dest，禁止其它kind attrs并要求匹配14的
  versioned ABI/profile；当前实现仍拒绝，直到typed dest、TargetCall/CRT写回及capability preflight同批落地。
  `factorize` may pass the instruction verifier but remains target-illegal until an exact production semantic
  profile exists.
- no SPM offset/end/bank attrs before SPM offset assignment.
- no raw DTE resource id, raw DTE register field, CSR helper or SCALAR ordinary instruction op before
  the corresponding instruction/sync family is defined and verified. Direct DTE p2p must use
  `wafer.instr.dte_send` / `dte_recv` / `dte_wait`, not ad hoc tile p2p ops or side tables；在physical
  endpoint/slot binding和target CRT support闭合前，这三类op在production target conversion中必须整体拒绝；
  Q16.T已对当前single-card fixed-size unicast profile闭合该binding，超出profile的模式继续拒绝。
- 每个 issue 都有可验证 completion relation；每个isolated `wafer.tile.region`的exit在显式terminal
  drain/wait/fence后pending-event set为空，variant gate覆盖rank中的all-and-only regions。`busytable` state
  不能作为completion proof。
- async handle的root provenance与task identity分别验证；handle alias或path union不能冒充完成了未被terminal
  wait覆盖的task，unsupported flow和missing terminal分别稳定失败。

variant-set verifier 还检查 committed instruction programs 覆盖所有 static rank entries 和完整 traversal，
不含 logical/scheduled `wafer.group`，只使用一套 final layout/SPM/DDR facts，并匹配所有跨 rank
transport send/recv/token relation。无法从 local IR 推导的 endpoint/channel/FSM facts 只在 late target
binding boundary 显式 materialize，然后作为 variant-set relation 验证，不能从名字或隐藏 side table 推断。
module-level executable/variant symbol 可以引用 per-rank `func.func` entry symbols，但 function body 是
instruction/control-flow 的唯一 code owner；symbol 或 package metadata 不能复制完整 instruction sequence。
任何上游rank-equivalence提示都不能替代验证。Q16必须在每个完整entry的instruction、layout/SPM/DDR、
event、transport和target binding均通过后，才由atomic commit构造all-and-only typed C++ rank records；
representative rank或byte-identical module不能替代未验证entry。

Instruction op-local lowering 不分配 physical address range、不解决 SPM bank conflict、不选择 DDR arena
placement，也不绑定 runtime symbol、packet bit 或 worker window。这些值属于 SPM/DDR planning 和 late
target binding；但所有 consumer 都必须在 whole-variant commit 前调用同一个 shared physical
geometry/range/narrowing verifier，target LLVM/package 不能成为首次发现 overflow 或 silent narrowing 的阶段。
DDR offset assignment必须接受或拒绝当前IR中的explicit DDR views/descriptors/compiler-managed
`memref.alloc`。Q16 commit前从accepted IR的use-def/type/effect/offset直接校验resource/entry/completion facts，
并materialize typed C++ rank record；target只派生address/range，package/runtime不得从instruction IR重新恢复resource语义。
`wafer.ddr.offset`是arena-relative fact，不是absolute device address；当前没有explicit arena base binding的
compiler-managed DDR `memref.alloc`在target conversion中必须以`unsupported_target_address`拒绝，不能把offset
直接常量化成地址。

当前shared physical geometry事实由Wafer IR中的`computeWaferPhysicalTensorInfo`及instruction verifier拥有；
target lowering直接复用这些facts并在clone preflight中补target field-width/range checks。仓库没有独立
identity/ABI/target-legality library或conversion-request对象，不能把讨论中的分层写成现状。若后续真实consumer要求
拆库，依赖仍须保持IR geometry → target legality → lowering单向。

shared target verifier不以post-commit conversion request作为唯一allocation输入。candidate preflight必须由
whole-variant transaction内部从当前clone、accepted offsets/transport binding和exact target context重算
root/view/slot/capacity relation；caller不能传raw range map、resolver callback或旁路resource view。
candidate preflight只形成transformation-local validation result，不能产出LLVM/ABI/artifact；commit函数内部重新
构造并验证，不接受caller proof。post-commit target conversion必须再次从committed
facts重跑同一geometry core，不能复用candidate analysis或把commit前result当artifact authority。

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

这是当前V0保守路径的形态示例，不固定parser/printer，也不固定planner对physical-dataflow realization的选择。
例中的GEMM省略orientation表示v1 normal/normal。versioned oriented ABI必须显式打印两个typed orientation；
mapped transfer也必须用带local offset的typed RDMA/WDMA表达，不能由lowering猜测。
例子中 stride 数值只说明 descriptor 字段位置，不作为 Cx padding 或 hardware packet 的规范值；
真实 padded size、Cx/NCx 对齐、bool bitpack、descriptor stride 和 SPM offset 分别由
`computeWaferPhysicalTensorInfo`、SPM memory planning 和 later realization 处理。

## 12. Implementation Work

本节“已完成”只记录 op-local ODS、conversion 和局部 pipeline coverage，不是 committed executable
completion proof。完整traversal、whole-rank SPM、whole-variant DDR、terminal event closure、transport/target binding、
all-rank typed bundle和whole-variant atomic commit仍必须按第1、9、10节合同统一验收。

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
   `wafer.instr.reduce`现为不接受init operand/attr的target-native leaf；source tile reduce按init-first canonical-order
   fill/movement/map-free elementwise composite展开，不直接生成该leaf。未来只有compiler-owned full-domain等价证明才能
   选择native reduce，不能由Q22 candidate或CModel policy反向授权。
5. `wafer.instr.unpool` 已改成 wrapper-aligned scalar `index` attr；`mask` / `unpool` 需要
   uint32 `index`，`avg` 禁止携带。`wafer.instr.tdma_data_move` verifier 按 kind 检查字段组合：
   `pad` / `img2col` 是 V0 production target input，transform-like kind 只允许作为 pre-lowering/imported
   instr IR；`transpose`、`mirror`、`rotate*`、NCHW/NHWC 和 TensorNom 由
   `--wafer-convert-tile-region-to-instr` 在 target/package 边界前 materialize；
   `wafer.instr.peripheral` verifier 按
   kind-specific wrapper signature 检查 required/forbidden attrs；当前live code仍因Count writeback未实现而结构化拒绝，
   Q3.6终态合同固定为7.7的single-element i32 SPM dest，不再留下result/ABI设计分支。

R3.2d.2 已完成：

6. 实现 `--wafer-convert-tile-region-to-instr` DialectConversion，并提供
   `WaferTileRegionToInstr` conversion library API。
7. conversion 递归处理 `scf.if` / `scf.for` region body，并保留 scalar / memref yield 关系。
8. conversion tests 覆盖 load/store、layout materialize、fill、GEMM、elementwise、reduce、
   copy、metadata view lowered to standard memref view、`Cx/NCx` block-major reshape、
   tail-only metadata reshape、nested `scf.if`、tile communication structured failure，以及
   padding layout materialization structured failure。
9. 每个 lowered WDMA 后立即生成 `wafer.instr.local_fence`，使当前 tile 的 SPM read lifetime
   在 store completion 后结束；每个`scf.for` body terminator前补一个backedge local fence，使loop body
   的RDMA/compute/movement不能跨动态迭代悬空；每个 single-block `wafer.tile.region` terminator 前还补一个
   terminal local fence，覆盖无 store 的路径。SPM planner 不把 region exit 当隐式完成：它按控制流分别跟踪
   local compute/movement 的全部 SPM read/write 和 Direct DTE token，只有 path-matching local fence
   或 exact DTE wait 才能 drain。可能零次执行的loop并非整体非法：pre-loop pending state不能只由body fence
   完成，body issue必须在backedge前完成；SPM则保守拒绝所有loop-carried async token。

R3.2d.3 当前边界：

10. `wafer-lower-tile-region-to-instr`复用`WaferTileRegionToInstr` conversion implementation，
    只作为bring-up / explicit-view lowering入口；它消费candidate已物化的DDR tile view，不负责搜索
    traversal / tile shape。
11. pipeline tests覆盖多region、structured `scf.if` / `scf.for`、tile communication
    structured failure，以及转换后不能残留executable target-abstract op的pipeline-level gate；
    focused memory-planning tests证明携带pre-existing root的安全loop在backedge fence后通过，而loop body创建
    fresh allocation并作为recurrence result携带时以`unsupported_lifetime_alias`拒绝。

R3.2d.4 已完成：

12. `wafer.tile.extract_slice`、`wafer.tile.insert_slice`、`wafer.tile.broadcast` 和
   `wafer.tile.transpose` 的静态 movement lowering 已接入同一个 DialectConversion，按 op 语义枚举
   logical iteration domain，调用统一 physical layout calculator 计算 source/dest byte offset，并
   coalesce 相邻 byte 段后打包成三层 stride/iteration `wafer.instr.gather_scatter` descriptor。
13. conversion tests 覆盖 compact tensor movement，以及 `Cx:[CxBlock][outer][lane]` /
    `NCx:[N][CxBlock][HW][lane]` block-major physical offset 的切片 descriptor 和 reshape/layout
    materialization descriptor packing。

后续仍需：

14. convert lowering 等 `wafer.tile.convert`
   或等价 source op 出现后再接入 completion gate。
15. dynamic DDR view、超过三层的 RDMA/WDMA descriptor packing、以及非 tensor-layout DDR 边界等
    source op 出现后再接入；当前不能靠名字或 shape 猜测出缺失的 strided boundary facts。
16. 按7.5.2实现`wafer.instr.quantized_gemm` ODS/interface/verifier/effects、tile lowering和native capability gate；
    未完成target CRT/golden/device-link前保持production target-illegal。
17. 实现`wafer.instr.mxfp_decode`及packed/scale/scratch/completion合同，lower到显式software decode + scale
    composite；不得把legacy helper或普通convert当完成证明。
