# Wafer Tile Region Design

日期：2026-05-25

状态：设计草案；2026-05-25 独立边界收口；2026-05-27 对齐 tensor collective 到 `wafer.tile.*` communication materialization；
2026-06-03 R3.2c 对齐 MLIR DialectConversion full conversion；2026-06-04 对齐
instruction-level Wafer IR 先于 SPM placement，并补正式 `group -> tile_region` conversion pass；
2026-06-05 对齐 memref-backed buffer contract，`Cx/NCx` 改为 Wafer memory attr marker

本文定义 `wafer.tile.region` 作为 `wafer.group` lowering 之后的 tile-local execution boundary。
它组织 tile-local storage、movement、layout materialization、target-abstract compute、communication
和 sync/effect op。它不重新做 group formation、traversal selection、root tile search 或 runtime
launch/package 组织。

SPMD 后的 StableHLO collective 在进入本文之前应已经规整成 Wafer LinalgExt-style tensor collective
并参与 group/tiling。`wafer.tile.region` 是这些 tiled tensor collective 第一次拥有 SPM storage、
placement-derived endpoint 和 communication staging demand 的层级；`wafer.tile.*` communication ops 不应在这之前
作为 group 输入出现。

本文依赖：

- `tasks/2026-05-12-wafer-group-design.md`
- `tasks/2026-05-21-wafer-layout-materialization-design.md`
- `tasks/2026-05-21-wafer-spm-bufferization-design.md`
- `tasks/2026-05-25-wafer-ddr-resource-allocation-design.md`
- `tasks/2026-05-25-wafer-compute-dialect-design.md`
- `tasks/2026-05-25-wafer-communication-dialect-design.md`
- `tasks/2026-06-05-wafer-instruction-ir-design.md`

当前实现状态口径：

- 仓库代码已有 `wafer.tile.region`、旧 `!wafer.storage`、`wafer.tile.alloc/load/store`、
  target-abstract compute/layout/move/view/comm op 和 group-to-tile-region conversion 原型。
- 2026-06-05 之后的主线合同已经改为 memref-backed buffer value：
  `memref<..., #wafer.memory<space, layout>>`。因此旧 `!wafer.storage` / `wafer.tile.alloc`
  只能视为待迁移原型，不能作为 R3.2c done 口径。
- 本文下面的 R3.2c coverage 表描述迁移后的目标合同；若与当前 ODS / test fixture 冲突，
  以 `tasks/progress.md` 的“当前 IR 状态”和本节口径为准。

## 1. 目标和非目标

目标：

- 给 group 后的 tiled program 一个稳定 region boundary。
- 把 tensor tile value materialize 成 tile-local memref、descriptor 或 placed instruction-level value。
- 在同一个 region 内表达 load/store、layout materialization、compute、communication、sync 和
  wait/drain ordering。
- 为 layout planning、SPM allocation 和 DDR/resource planning 提供可重算 IR 结构。
- 为 lower-level Wafer ops、C ABI 和 launch outline 提供清楚的输入。

非目标：

- 不决定哪些 op 可以 group 到一起。
- 不保存 planner 搜索过程、失败候选、cost model trace 或 shadow schedule。
- 不把 SPM offset、DDR buffer object address、DTE resource 或 C ABI call 提前塞进 tensor/group 层。
- 不替代 `wafer.launch`。`tile_region` 是 device-side execution scope，`launch` 是 host/device
  invocation boundary。

## 2. Stage Position

```text
wafer.group
  -> group-to-tile-region lowered `wafer.tile.region` IR
  -> instruction-level wafer.instr.* IR over unplaced Wafer-tagged memref values
  -> SPM placement on the same instruction-level IR
  -> DDR / resource legality on placed instruction IR
  -> accepted / rejected / split decision
  -> committed wafer.tile.region
  -> materialized placed instruction-level IR / placed memref / access descriptor
  -> codegen emission to C ABI / packet / launch
```

本文区分两种生命周期：

- `wafer.tile.region` IR：R3.2c 通过同一套 MLIR DialectConversion builder
  构造。局部 dump / planner 可以在 transformation-local scratch IR 中观察 tile-region IR；
  `--wafer-convert-group-to-tile-region` 可以把当前模块中的 supported logical group 重写成
  tile-region IR，用作 legality/debug/后续 pass bring-up。它仍不是 accepted plan，
  也不是 SPM allocator 的直接输入；rejected tile-region IR 不进入主线 accepted IR。
- committed `wafer.tile.region`：R3.3 只把 R3.2g 已接受的 plan 写入主 IR。后续 R3.4/R3.5
  只 materialize accepted layout/SPM/DDR/instruction facts，不重新决定 group 是否可行。

### 2.1 R3.2c Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  R3.1 verifier-legal tensor-level logical `wafer.group` op、R3.2a
  `GroupTilingDemand` analysis result 和 R3.2b `GroupLayoutPlan` analysis result。
- Current stage responsibility:
  通过 MLIR DialectConversion 构造 `wafer.tile.region` IR；
  `wafer.group` 是 illegal root，conversion pattern 产出完整 legal `wafer.tile.region`，并用 full
  conversion 保证 converted region 内不残留 logical group / linalg / tensor allocation op；
  把 logical group boundary 映射成 `wafer.tile.load` / `wafer.tile.store`，把 selected
  layout 和 materialization cut 映射成带 `#wafer.memory<space, layout>` 的 memref value
  与 `wafer.tile.materialize_layout`，
  把可验证的 structured compute / tensor collective 映射成 target-abstract
  `wafer.tile.*` compute / `wafer.tile.*` communication / sync/effect op。
- Output artifact / IR:
  verifier-legal `wafer.tile.region` IR 或结构化 failure reason。局部 conversion
  pass 可以把 supported group 重写成 tile-region IR；planner scratch entry 返回 tile-region IR 用于
  dump、verification 和后续 planning analysis。rejected tile-region IR 不写入主线 accepted IR。
- Downstream consumer:
  R3.2d Wafer instruction legalization / selection、R3.2e SPM placement、R3.2f DDR/resource planning +
  compute/movement legality analysis、R3.2g closed-loop planner。
- User-level driver / named pipeline:
  主线仍由 `wafer-opt --program-pipeline=stablehlo-spmd-to-group` 产生 logical group；
  R3.2c 的局部验证入口是 `wafer-opt --wafer-convert-group-to-tile-region` 和
  `wafer-opt --wafer-dump-group-to-tile-region`。
- Explicit non-goals:
  不做 SPM offset allocation、不做 DDR pool/range/bandwidth planning、不 accept/reject/split
  group、不把 tile-region IR 当成 R3.3 accepted materialization、不 lower 到 packet/ABI/LLVM。
- Completion gate:
  FileCheck、conversion pass、dump pass 和主线 pipeline 覆盖 R2.4/R3.1 已能产出的 Wafer V0 硬件可承载 local
  compute/movement/view family：load/store boundary、layout materialization、SPM abstract
  allocation、fill、GEMM、elementwise/relation、native reduce、passthrough broadcast/transpose/copy、
  tensor slice movement 和 static reshape view。硬件 V0 无承载或当前 IR 缺 placement/local-rank /
  runtime ABI 事实时才允许结构化 failure。
```

### 2.2 R3.2c Target Coverage Matrix

本表是 R3.2c memref-backed migration 的完成表，不是“当前代码已全部支持”的声明。R3.2c 只应该转换
当前 IR 可以验证的 op；缺少 IR 事实或目标 op 的情况必须结构化失败，不能用名字、case 或 side
table 补协议。

| source IR / op family | R3.2c 目标处理 | 覆盖状态 | 正确 conversion 做法 / 后续要求 |
| --- | --- | --- | --- |
| `wafer.group` / `wafer.group.yield` boundary | DialectConversion 中由 `OpConversionPattern<wafer.group>` 构造 `wafer.tile.region`，对 `ins + outs` 生成 `wafer.tile.load` / `wafer.tile.store` / `wafer.tile.yield`；同一 builder 支持 scratch dump 和正式 `--wafer-convert-group-to-tile-region` pass | supported for tile-region IR | `ConversionTarget` 将 `wafer.group` / `wafer.group.yield` 标为 illegal；debug dump 只调用同一 conversion builder，不能自己承担 lowering 逻辑。 |
| ranked tensor boundary values | 生成 logical-shape memref，memory attr 默认 `#wafer.memory<spm, tensor>`，记录 compact tensor layout version | supported for ranked tensor | 保持类型/verifier 驱动；后续 instruction-level IR 再决定 concrete instruction，SPM placement 再决定 offset/window。 |
| scalar boundary values | 作为 tile-region block scalar SSA value 传入，供 fill/reduce init 等 scalar operand 使用 | supported for scalar | 只支持 float / integer / index scalar；不生成 storage，不作为长期 side channel。 |
| `arith.constant` tensor | clone constant 后 `wafer.tile.load` 到 tensor-layout storage | partial | 只适合 tensor constant；scalar constant 只应在 compute body 或显式 init 语义中消费。需要区分 constant residency / DDR / immediate policy。 |
| `arith.constant` scalar | clone scalar constant，并作为 `wafer.tile.fill`、`wafer.tile.reduce init_value` 或 elementwise body 推导输入 | supported for scalar constants | scalar 语义通过 SSA value 或 typed attr 进入目标 op；不靠名字或原 op 残留。 |
| `tensor.empty` | 生成 `memref.alloc`，memref type 携带 logical shape、element type 和 `#wafer.memory<spm, layout>` marker | supported as abstract allocation demand | `memref.alloc` 不分配物理 offset/window；R3.2d 只把 target-abstract op 合法化到读写该 memref 的 instruction op，R3.2e 才做真实 SPM placement。不能把 empty 偷映射成 output alias。 |
| `tensor.extract` scalar | clone 到 tile-region 内，供动态 scalar init / scalar value 使用 | supported for scalar extract | 只作为 scalar SSA 支持 op；不表示 tile compute。 |
| `linalg.fill` | 生成显式 `wafer.tile.fill`，写入 existing storage；fill result 映射为该 initialized buffer | supported for scalar fill | `wafer.tile.fill` 暴露 target-abstract write relation；具体是否 lower 成 CT fill、memset 或 immediate pattern 由 R3.2d instruction selection 决定。 |
| `linalg.matmul` | lhs/rhs materialize 到 `cx`，生成 `wafer.tile.gemm`，结果记录为 `cx` | supported for simple `linalg.matmul` | pattern 应检查 rank、dtype、accumulator/result relation、layout requirement；batch matmul / generic contraction 另列，不应混成 matmul 特判。 |
| `linalg.generic` simple elementwise / relation | 单 result、单 `linalg.yield`，typed scalar mapper 识别 add/sub/mul/div/min/max/neg/exp/sqrt/rsqrt/tanh 和 cmp eq/ne/lt/le/gt/ge；生成 `wafer.tile.elementwise` 并带原 `indexing_maps` | supported for simple CT family | 不靠 op name string 恢复语义；复杂 region、多 result、select/mask 和 convert 仍需要明确 kind / op contract。 |
| `linalg.reduce` / reduction-like generic | reduction iterator + scalar combiner lower 到 `wafer.tile.reduce`；input/result materialize 到 aligned `cx`/`ncx`；constant init 用 `init_value`，dynamic init 用 scalar operand | supported for native sum/max/min | `avg` 是 Wafer reduce kind，但当前 R2.4 fixture 尚未产出可直接识别的 avg combiner；`mul` 不伪装 native。 |
| passthrough `linalg.generic` for broadcast / transpose / copy | 根据 projected-permutation indexing map lower 到 `wafer.tile.broadcast`、`wafer.tile.transpose` 或 `wafer.tile.copy` | supported for static projected maps | 这是 movement，不是 elementwise compute；result map 必须是 identity，动态/非 projected map 结构化失败。 |
| `tensor.extract_slice` / `tensor.insert_slice` | static offsets/sizes/strides lower 到 `wafer.tile.extract_slice` / `wafer.tile.insert_slice` | supported for static slices，包括 MLIR 合法的 rank-reduced slice | move op verifier 检查 full slice shape、可选 rank reduction、element type、layout/memory space 和 slice range；dynamic slice metadata 需要先扩 IR。 |
| `tensor.expand_shape` / `tensor.collapse_shape` | static element-count-preserving reshape lower 到 `wafer.tile.reshape` | supported for static shape-only reshape | `wafer.tile.reshape` 不移动数据；layout/materialization 不兼容时必须使用 explicit movement，不用 reshape 逃避 physical layout。 |
| `wafer.tensor.*` | R3.2a/R3.2b 可收集 demand/layout；R3.2c 当前失败为缺 placement/local-rank facts | explicitly deferred | 只有 placement/local-rank/buffer facts 进入可验证 IR 后，才能 pattern 化到 `wafer.tile.*` communication ops / `wafer.instr.local_drain` 和后续 sync boundary；不能写死 local rank 或 ring schedule。 |
| unknown op inside group | 结构化失败 | unsupported | conversion target 应把 `wafer.group` 设为 illegal；unsupported body op 应导致 conversion failure，而不是留下半转换 group。 |

R3.2c 迁移完成后，supported 子集必须由同一 conversion builder 覆盖，并把 unsupported / deferred
子集的 failure gate 固定下来。扩展 coverage 时每新增一类 op 都要同时补：IR 语义、verifier、
conversion pattern、negative test 和下游 instruction/memref demand 来源。

`wafer.tile.region` 可以跨这些 lowering 子阶段保留为 region container。当前长期边界是：buffer
value 使用带 Wafer memory attr 的 `memref`，lower-level descriptor 只在 placement / emission
能验证其语义时出现。因此不能简单说 tile region 内“永远不允许 memref”或“永远不允许 lower-level
op”。正确边界是：

- abstract tile-region 阶段只允许 verifier 可解释的 Wafer-tagged `memref.alloc`、metadata view 和
  Wafer movement/compute op；不允许 generic `memref.load/store/copy` 作为语义逃逸。
- placement / realization 后允许 verifier 可解释的 placed `memref` / descriptor / lower-level Wafer op。
- LLVM call、runtime call、C ABI call 不属于 `tile_region` 主体，应在 launch / ABI lowering 后
  出现。

## 3. Op Contract

`wafer.tile.region` 的概念合同：

```text
wafer.tile.region (...) -> (...) {
  ^bb0(%tile_id, %block_id, %region_args...):
    ...
    wafer.yield ...
}
```

必须表达：

- region argument / result 与 group outputs 或 launch boundary 的 SSA 关系。
- per-tile identity，例如 logical rank、block id、physical tile coordinate 或 placement-derived
  descriptor。
- external input/output、constant source、inter-group value 的 load/store boundary。
- tile-local storage ownership、memory space、layout 和 effect。
- async issue 与对应 wait/drain/barrier。

不应表达：

- raw register packet field。
- runtime buffer object handle。
- group planner 的 rejected group plan。
- case-specific K tile、psum lifetime 或 epilogue placement 作为固定 protocol。

## 4. Tile Buffer and Memory Space

tile-region 内部的长期 buffer value 是 MLIR `memref`。Wafer target-specific memory attr 放在
memref memory-space slot 中：

```mlir
memref<64x256xf16, #wafer.memory<spm, tensor>>
memref<64x256xf16, #wafer.memory<spm, cx>>
memref<64x256xf16, #wafer.memory<ddr, tensor>>
```

该 attr 同时携带两类 target facts：

- address space：`spm` 或 `ddr`。
- physical layout marker：`tensor`、`ntensor`、`cx`、`ncx`。

memref shape / element type 表达 logical shape / dtype。`Cx/NCx` 的 `C0`、storage bytes、
range-end、bool bitpack 和 256B padding 不写入 type 字段，统一由
`computeWaferPhysicalTensorInfo(memrefType)` 从 logical shape、element type 和 Wafer memory attr 推导。

SPM buffer 由 SPM allocator 放置；DDR buffer/descriptor 由 DDR planner 和 launch/runtime 负责
ownership。二者使用同一套 memory-space 语义，不在不同文档发明不同含义。

Wafer `Cx/NCx` marker 不占用 MLIR memref layout slot。MLIR memref layout slot 只用于 MLIR 能
按 affine / strided 语义解释的普通 layout；Wafer `Cx/C0` 是 target physical layout marker，
由 Wafer verifier、SPM allocator 和 instruction lowering 解释。

在 placement / realization 前：

- `memref.alloc` 表达 tile-local allocation identity 和 lifetime，不表达 physical offset。
- `memref.dim` 可用于读取 logical shape。
- generic `memref.load/store/copy` 不能用于 Wafer-tagged SPM buffer。
- metadata-only view 只有在 verifier 能证明 Wafer layout marker 仍然合法时才允许；真实 physical
  layout conversion 必须通过 `wafer.tile.materialize_layout` / TDMA movement 表达。

## 5. Core Ops Inside Tile Region

V0 需要以下 op family：

| family | 作用 | 主要 verifier |
| --- | --- | --- |
| `wafer.tile.load` | 从 `#wafer.memory<ddr, tensor>` / external / constant source 读入 tile-local memref | source range、dtype、layout、stride、effect |
| `wafer.tile.store` | 写回 external output / inter-group DDR value | destination range、layout、visibility、effect |
| `wafer.tile.materialize_layout` | 显式 layout conversion | source/result layout relation、可消除冗余转换 |
| `wafer.tile.*` compute ops | target-abstract compute | operand/result layout、instruction family legality、scratch/psum demand |
| `wafer.tile.*` communication ops | tile 间或 collective movement；由 tiled tensor collective + SPM buffer + placement materialize | endpoint、token、fixed byte count、buffer lifetime |
| `wafer.instr.local_drain` 和后续 sync boundary | local drain、comm wait、barrier | async op completion、effect ordering |

这些 op 的具体算法分别归 layout、SPM、DDR、compute、communication 文档。`tile_region` 只负责
把它们放在一个可验证 execution scope 里。

## 6. Lowering Responsibilities

实现上可以分多步，但每一步只改写当前 IR：

1. `wafer.group` lowering：R3.2c 把 logical group + R3.2a/R3.2b facts 转成
   transformation-local `wafer.tile.region` IR。
2. target-abstract op selection：在 tile-region IR 内把 tile-level linalg/tensor compute 绑定到
   `wafer.tile.*` compute / movement op；把 tiled tensor collective 在可表达的 placement / buffer /
   communication demand 下 materialize 为 `wafer.tile.*` communication 或 explicit p2p schedule proposal。
3. layout assignment：为 op 约束选择 physical layout marker，在 cut edge 插入
   `wafer.tile.materialize_layout`。
4. Wafer instruction legalization / selection：R3.2d 把 target-abstract executable op 合法化并
   选择成 instruction-level `wafer.instr.*`，复用现有 Wafer-tagged memref SSA graph。
   instruction-level IR 需要列出 issue family、read/write/issue effects、descriptor attrs、
   temp/psum/staging memref values、alias/view 关系和 reject reason。
5. SPM placement：R3.2e 只消费 instruction-level IR with unplaced Wafer-tagged memref values，
   在同一 IR 上填入 offset/end/bank span 和 lifetime/reuse；不能直接从 target-abstract op 猜
   memref demand。
6. DDR/resource planning：R3.2f 消费 placed instruction-level IR、SPM facts 和 DDR boundary，做 capacity /
   bandwidth / range legality。
7. closed-loop decision：R3.2g 接受、拒绝或要求 split / retry；rejected tile-region IR 丢弃。
8. committed `wafer.tile.region` materialization：R3.3 只把 accepted plan 写入主 IR。
9. placement realization：把 accepted unplaced memref 降到 placed memref / address descriptor。
10. lower-level op lowering：转成 wrapper-friendly Wafer ops，最后进入 C ABI / launch。

未接受的候选 plan 不能落入 IR 后等待下游修复。合法性失败应反馈给 group/layout/resource
planner 重新选择 tile shape、internal split、layout 或 group boundary。

当前 ODS 已有 `wafer.tile.region` 内 movement、layout、compute、comm 和 sync op 的基础
layout/materialization/resource interface 查询入口，但这些接口和 verifier 仍基于旧 storage 原型；
R3.2c 需要迁移到 memref-backed contract。第 4 步 instruction legalization / selection 需要把
target-abstract op 先降到 instruction-level IR；第 5-7 步的完整 SPM/DDR resource planning 和
closed-loop decision 仍未完成。

当前实现状态：

- `--wafer-convert-group-to-tile-region` 是正式 MLIR conversion pass，使用 `Passes.td` 声明和
  DialectConversion legality target，在 supported 子集上重写当前模块；当前输出仍使用旧
  `!wafer.storage` / `wafer.tile.alloc` 原型，R3.2c active work 是迁到
  `memref<..., #wafer.memory<space, layout>>`。`--wafer-dump-group-to-tile-region`
  是同一 builder 的只读 dump 入口，并显式 preserve analyses。
- 旧 `--wafer-materialize-single-tile` explicit unit/debug pass 已删除。后续 tile_region materialization
  必须由 R3 消费真实 frontend/SPMD program chain 和 group contract 后恢复。
- `--wafer-materialize-multi-tile-no-comm` 已删除。旧实现按 placement rank 数 clone whole-tensor
  tile_region，既没有 per-rank shard slice，也没有 output merge/writeback contract，不能作为
  multi-tile materialization 证据。

P4.3/P4.4 需要重新建立 per-tile logical rank、block id、physical coordinate、local shard slice 和
launch args / identity lowering 的 IR contract。当前没有 multi-tile no-comm 主线 materialization。

## 7. Verifier

`wafer.tile.region` verifier 至少检查：

- region argument/result 和 terminator 类型匹配。
- region 中没有无法解释的 side table dependency 或名字匹配语义。
- external load/store boundary 都有明确 memory space、shape、dtype、layout 和 effect。
- async producer 的 source/destination 在 wait/drain 前不能被非法复用。
- `wafer.tile.materialize_layout` 的输入输出 layout relation 合法；同 layout 冗余转换应由 verifier
  拒绝，上游应避免生成这种 no-op conversion。`wafer.tile.reshape` 这类无副作用 view op 可由
  canonicalization 删除同类型 no-op。
- `#wafer.memory<spm, *>` memref 在 placement realization 前必须经过 SPM allocation；
  `#wafer.memory<ddr, *>` memref 必须有 DDR resource plan / allocation policy。
- lower-level op 出现时，其 operand 已经是 placed memref 或 verifier 可解释 descriptor。

Verifier 不检查 group 是否应该形成；那是 `wafer.group` 和 planner 的职责。
`wafer.group` / `wafer.tile.region` 这类 region op 的 boundary invariants 放在普通 `verify()`，
body / terminator / nested region 关系放在 `verifyRegions()`；父 region op 只解释自己 body 的直接 op，
不递归解释 tensor collective 等子 op 的内部 region。

## 8. 与 Case 的关系

设计文档可以用典型 case 展示：

```text
load A/B tile -> matmul -> optional epilogue -> store outputs
```

但 case 中的 K tile、psum lifetime、epilogue 位置、double-buffer depth、output order 都只是
planner 候选结果，不是 `wafer.tile.region` contract。多输出、不同 output domain、hidden
dimension、layout cut 和 memory split 都由 op interface、SSA use-def、effect 和 verifier 处理，
不能通过 case 名字固定。
