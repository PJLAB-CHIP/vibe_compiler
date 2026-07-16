# Wafer Tile Region Design

状态：2026-07-16按已完成Q29终态边界同步。本文定义memref-backed `wafer.tile.region`在rank-local tiled
task/dataflow program中的职责；structured tensor program由scheduler和
`WaferTensorProgramToTileRegion` materializer进入本层。实现状态以`tasks/progress.md`为准。

`wafer.tile.region`是完整rank traversal中的结构化task/traversal fragment。它组织Wafer-tagged memref、
movement、layout materialization、target-abstract compute、communication和sync/effect op，但**不是**一组
独立SPM arena、DDR切边、candidate、executable或提交单元。跨region的resident SPM value和event必须作为
显式SSA operand/result传递；region边界本身不得自动生成DDR store/reload。

当前rank-local scheduler从structured iterator、indexing map和SSA dataflow生成完整task traversal，
再按需要使用一个或多个tile region组织局部执行。region boundary不隐式materialize DDR，
跨region residency只由显式SPM SSA value和event表达。

SPMD后的StableHLO collective在进入本文之前应已经规整成`wafer.linalg_ext.collective.*`。tile-dataflow
materialization在本层把它们连接到SPM buffer、rank membership和completion；`wafer.tile.*` collective和
`wafer.instr.dte_*`不应提前污染tensor-level structured source。

本文依赖：

- `tasks/06-group.md`
- `tasks/08-layout-materialization.md`
- `tasks/09-spm-memory-planning.md`
- `tasks/12-ddr-memory-planning.md`
- `tasks/10-compute-movement.md`
- `tasks/13-communication.md`
- `tasks/11-instruction-ir.md`

当前structured materializer的typed lowering能力：

- 仓库代码已有 `wafer.tile.region`、`wafer.tile.load/store`、target-abstract
  compute/layout/move/view/comm op 和tensor-program-to-tile-region conversion。
- R3.2c已把tile-local buffer value迁移为memref-backed contract：
  `memref<..., #wafer.memory<space, layout>>`；external tensor boundary先materialize为
  `memref<..., #wafer.memory<ddr, tensor>>`，tile-local value使用
  `memref<..., #wafer.memory<spm, layout>>`。`tensor.empty` 作为 writable program output 时降为 DDR
  `memref.alloc`；tile-local temporary 降为 SPM `memref.alloc`。旧 `!wafer.storage`、
  `wafer.tile.alloc`、`#wafer.memory_space` 和 `#wafer.mem_layout` 已从主线 IR 定义、verifier
  和测试中删除。
- `wafer-schedule-tensor-program` IR-local pipeline和production scheduler共用typed materializer；
  function-boundary One-Shot Bufferize把tensor boundary转成`#wafer.memory<ddr, tensor>` memref boundary。
- 本文下面的coverage表记录当前typed mapping和negative gate，不定义另一条compatibility pipeline。

## 1. 目标和非目标

目标：

- 为rank-local tiled task/dataflow program提供可验证的structured execution fragment。
- 把tensor tile value materialize成Wafer-tagged memref、typed movement/compute/collective和instruction operand。
- 在region内及显式region边界表达load/store、layout materialization、compute、communication、sync和
  fence/wait ordering。
- 让layout、whole-rank SPM lifetime、显式DDR spill和instruction planning都能从accepted IR重算。
- 为lower-level Wafer ops、target CRT和launch outline提供清楚的输入。

非目标：

- 不形成或解释group边界，也不把“融合”编码成opaque复合op；resident edge由SPM SSA dataflow表达。
- 不独立决定whole-rank candidate；它只materialize scheduler选定proposal中的typed execution fragment。
- 不保存 planner 搜索过程、失败候选、cost model trace 或 shadow schedule。
- 不把 representative first/tail tile、单个 `wafer.tile.region` 或单个 group 的通过结果当成
  完整 traversal/variant completion proof。
- 不把SPM offset、DDR runtime allocation address、DTE resource或target CRT call提前塞进tensor source层。
- 不替代 runtime launch metadata。`tile_region` 是 device-side execution scope，runtime package /
  launch metadata 是 host/device invocation boundary。

## 2. Stage Position

```text
verified static rank structured tensor-program clone
  -> bounded scope/traversal/reduction proposal
  -> per-scope full traversal materialization; one or more `wafer.tile.region` scopes organize task fragments
  -> buffer-level communication collective materialization over unplaced SPM memrefs
  -> p2p Direct DTE instruction schedule over unplaced SPM memrefs
  -> candidate DDR tile-view materialization for planner candidate evaluation
  -> instruction-level wafer.instr.* IR over unplaced Wafer-tagged memref values
  -> commit all selected task fragments into one complete rank candidate clone
  -> one post-commit canonicalization of static one-trip wrappers and equivalent full views
  -> derive bounded spill baseline + deterministic maximal full-buffer-resident rank alternatives
  -> whole-entry SPM memory planning on complete rank programs
  -> whole-entry/variant-set DDR memory planning on memory-planned instruction IR
  -> event completion and physical geometry/range/narrowing gates
  -> bounded rank frontier
  -> per-alternative function-boundary One-Shot Bufferization + SPM/DDR replanning + fresh recost
  -> filter failed alternatives; fail rank only when the finalized frontier is empty
  -> physical transport acceptance + all-rank transport verification + target-entry ABI preflight
  -> closed-loop candidate driver for whole-variant retry/split/commit decision
  -> atomic commit of complete static rank instruction programs; no tensor/Linalg/wafer.group remains
  -> target instruction LLVM call emission from committed IR + accepted offsets + committed transport binding
  -> device-code symbol closure / package / runtime adapter
```

本文区分两种生命周期：

- transformation-local candidate IR：tile-dataflow materializer在whole-rank clone中构造完整traversal。
  只允许当前structured scheduler的IR-local dump/verification；已退役入口不能成为显式IR replay/debug，
  也不能让单region、单task或full-shape initial candidate绕过whole-variant selector。
- committed static rank program：`wafer.tile.region`只作为完整traversal内的局部scope存在。
  committed materialization只把candidate-selection已选中、且整个static rank variant set已通过gates
  的complete candidate clone原子写入主IR，同时消解所有tensor/Linalg/`wafer.group`。clone在提交前已经包含
  stage-accepted transport binding和all-rank verification；它们与 static rank entries 一起原子提交。后续 target
  LLVM call emission只从committed IR、accepted offsets和经当前IR验证的entry/resource/completion facts派生
  address/range/descriptor；后续typed C++ bundle承接这些事实，package不重新决定candidate是否可行，也不复制
  placed/access descriptor中间协议。当前rank-local selected-candidate、tile-region、instruction、whole-function
  SPM和whole-variant coordinator的本地路径已闭合。TP16 7B compile-only及rank-count=1/16结构证据由
  `tasks/06-group.md`和`tasks/16-verification-plan.md`记录；双配置full gate和组织检查也已闭合，Q29和Q28的
  完成状态只看`tasks/progress.md`。

production driver在进入本文前为每个logical rank建立isolated module clone，并把rank作为显式typed C++调用参数传给
tile-dataflow/tile-region lowering。当前没有executable dialect、candidate rank op或代表rank归并协议；rank identity不能从
SymbolRef、文件名、pass默认值或代表rank恢复。

### 2.1 Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  transformation-local complete static rank candidate clone，其中包含由tasks/06从verified structured tensor
  program直接构造的rank-local tiled task/dataflow proposal、static traversal、typed boundary slice、
  当前scope-prefix residency和deterministic order、显式rank；不接收grouped artifact。generic per-edge
  residency、task-order/layout-cut搜索尚未实现，也不是本stage自行恢复的隐式输入。
- Current stage responsibility:
  为clone中每个rank entry物化all-and-only完整traversal，用一个或多个memref-backed
  `wafer.tile.region`组织typed task fragment；把external/parameter/final-output和candidate显式spill边
  materialize成`#wafer.memory<ddr, tensor>` subview及`wafer.tile.load/store`，把resident edge连接成跨task/
  跨region的SPM memref SSA和event；把selected layout和materialization cut映射成带
  `#wafer.memory<space, layout>`的memref value与`wafer.tile.materialize_layout`，
  把可验证的 structured compute / tensor collective 映射成 target-abstract
  `wafer.tile.*` compute / `wafer.tile.*` communication / sync/effect op；
  full conversion保证materialized candidate中不残留tensor/Linalg/`wafer.group`，并把函数external tensor
  signature/return转成`#wafer.memory<ddr, tensor>` memref boundary，并保证spill WDMA/RDMA、region boundary和static
  full view足以由下游直接验证full-buffer handoff。typed op mapper由
  `WaferTensorProgramToTileRegion`单一拥有，不存在另一套builder/API决定production traversal或DDR cut。
- Output artifact / IR:
  对完整rank traversal有覆盖的verifier-legal memref-backed tile task/dataflow candidate IR，或结构化failure reason。
  resident/spill、movement、layout、collective、buffer version和completion均在IR中显式；
  任一rank/task/traversal失败时整个candidate clone都不写入accepted IR。不存在group conversion输出、
  group-based debug artifact或第二条pipeline。
- Downstream consumer:
  candidate DDR tile-view materialization和instruction lowering；全部selected task fragments进入完整rank clone后，
  whole-variant coordinator只运行一次canonicalization，再从同一canonical committed rank派生spill baseline与
  deterministic maximal full-buffer-resident两个有界alternatives，交给SPM/DDR offset assignment、event/transport
  verification、cost和whole-variant candidate-selection。rank frontier进入Q16后，每个alternative独立完成
  function-boundary bufferization与SPM/DDR replanning；失败alternative被过滤，survivor从最终instruction IR重新
  计算cost，只有frontier为空才使该rank失败。
- User-level driver / named pipeline:
  production由同一
  `wafer-compile --input-program-dir ... --output-program-dir ... --execution-ranks={1|16} --target-profile=wafer-tx81-single-card-kernel-v1`
  在全部显式
  per-rank clone的whole-variant candidate flow内调用本stage。上游是verified structured tensor program；
  `wafer-opt` 通过`wafer-schedule-tensor-program`只提供同一structured task materializer的IR-local
  debug/test，不能独立形成production artifact或用户stop-stage。
- Explicit non-goals:
  不做SPM offset allocation、不做DDR offset/range/resource acceptance、不独立select/reject whole-rank
  candidate、不把tile-region IR
  当成 whole executable 或独立 committed materialization、不 lower 到 packet/target LLVM；不允许
  full-shape initial candidate绕过正常 gates，也不允许 representative tile/rank 成为提交依据；本stage不得基于
  presumed rank equivalence跳过任何显式rank clone；不枚举generic per-edge resident/spill subset，也不在某个
  producer result的任一consumer不安全时做partial handoff改写。
- Completion gate:
  local tests覆盖structured source到typed tile-region family：external/spill DDR memref load/store boundary、
  cross-task/cross-region resident SPM edge、layout materialization、DDR/SPM
  `memref.alloc`、tile-local allocation、fill、GEMM、elementwise/relation、带显式constant init的source reduce、passthrough
  broadcast/transpose/copy、tensor slice movement、static reshape view、top-level
  `wafer.linalg_ext.collective.*` 到 `wafer.tile.*` collective materialization、tile-region 内
  `scf.if` / `scf.for` 递归 lowering 和 function-boundary One-Shot bufferization。硬件 V0 无承载或当前
  IR缺runtime ABI/nested collective事实时才允许结构化failure。integrated completion还要求从真实
  structured program出发，在同一candidate clone中覆盖所有rank/task的完整traversal，后续全entry layout/SPM/DDR、instruction、
  event、transport和target gates全部通过，并以一次atomic commit消解所有tensor/Linalg；已退役surface保持
  无注册、无consumer；局部pass、
  单tile dump或representative tile/rank通过不构成completion。即使两个rank最终module byte-identical，
  Q16 bundle也必须保留两个已分别验证的rank records，不能据此省略entry。
```

### 2.2 Structured Materialization Coverage Matrix

本表记录structured task materializer已有的typed scalar/op mapping和negative gate。它不是
production artifact boundary；materializer只转换当前IR能验证的task，缺少IR事实或目标op时必须
结构化失败，不能用名字、case或side table补协议。

| source IR / op family | R3.2c 目标处理 | 覆盖状态 | 正确 conversion 做法 / 后续要求 |
| --- | --- | --- | --- |
| ranked tensor boundary values | 生成 logical-shape DDR memref，memory attr 默认 `#wafer.memory<ddr, tensor>`；full tensor use 产生 compact SPM tensor-layout version；external boundary 上的 static `tensor.extract_slice` / direct output `tensor.insert_slice` storeback 和 candidate tile offsets/sizes evaluation lowering 由 candidate tile-view materialization 转成 DDR `memref.subview` tile view | supported for ranked tensor and static candidate slice | 保持类型/verifier 驱动；后续 instruction-level IR 再决定 concrete instruction，SPM offset assignment 再决定 offset/window；函数边界由 named pipeline 的 One-Shot Bufferize 转成 DDR memref。candidate tile-view materialization 只构造 candidate evaluation tile views，不选择最终 plan；closed-loop traversal / tile-shape search 归 candidate-selection。 |
| scalar boundary values | 作为 tile-region block scalar SSA value 传入，供 fill/reduce init 等 scalar operand 使用 | supported for scalar | 只支持 float / integer / index scalar；不生成 storage，不作为长期 side channel。 |
| `arith.constant` tensor | splat常量不进入tile-region boundary，由exact typed scalar + `wafer.tile.fill`在SPM局部物化；static slice只传播splat属性并直接按result tile shape fill，不先物化full tensor或伪造bit-packed movement；只有仍被显式存储语义消费的非splat tensor才可作为read-only DDR source | supported for splat; partial for stored non-splat | region operand/block argument不得为已证明的splat重复保留一份`memref.global`/DDR事实；能在local-compute normalization折叠成标量的非splat view链应先被清理。target不为没有显式consumer/storage合同的global增加fallback。 |
| `arith.constant` scalar | clone scalar constant，并作为 `wafer.tile.fill`、`wafer.tile.reduce init_value` 或 elementwise body 推导输入 | supported for scalar constants | scalar 语义通过 SSA value 或 typed attr 进入目标 op；不靠名字或原 op 残留。 |
| `tensor.empty` | external task-program output的`tensor.empty`生成`#wafer.memory<ddr, tensor>` boundary value；tile-local temporary生成`#wafer.memory<spm, tensor>` `memref.alloc`。materializer从source result/use relation识别前者 | supported as abstract allocation demand | `memref.alloc`不分配物理offset/window；DDR boundary/explicit spill requirement的planned range归DDR offset assignment；accepted IR的use-def、effect和offset验证后写入typed C++ rank record，package不再从raw IR恢复；SPM offset/window归whole-rank SPM assignment。不能把arbitrary empty偷映射成output alias。 |
| `tensor.extract` scalar | 从已 materialized DDR boundary memref 生成 `memref.load`，供动态 scalar init / scalar value 使用 | supported for boundary scalar extract | 只作为 scalar SSA 支持 op；不表示 tile compute；tile-local tensor element read 不能用 generic memref.load 伪装。 |
| `linalg.fill` | canonical scalar payload生成fresh SPM result和显式`wafer.tile.fill`，不原地改写DPS init；仅当结果只作为overwrite-only GEMM的已证明identity init时可只传播metadata | supported for exact scalar fill | tensor SSA旧init必须保持不变；named payload必须精确yield fill value。具体CT fill/memset/immediate选择归instruction lowering。 |
| `linalg.matmul` | exact multiply-accumulate payload、一个DPS init且init可证明为`+0`时，lhs/rhs materialize到`cx`并生成overwrite-only `wafer.tile.gemm` | supported for exact simple `linalg.matmul` | 非零、`-0.0`、未知init、额外payload、错误wiring、fastmath/overflow flag均不能被当前GEMM合同静默丢弃；batch matmul执行同一identity/payload gate。 |
| `linalg.generic` simple elementwise / relation / select | 单result、单`linalg.yield`，typed scalar DAG mapper按SSA wiring逐op生成`wafer.tile.elementwise` | supported for exact simple CT family and basic select | 当前kind不能区分的unsigned div/min/max、unsigned integer relation、`maxnum/minnum`和不匹配target NaN语义的float predicate必须fail closed；不能只看yielded op class。 |
| `linalg.reduce` / reduction-like generic | reduction iterator + scalar combiner lower到`wafer.tile.reduce`；input/result materialize到aligned `cx`/`ncx`；constant init用typed `init_value`或直接constant SSA operand保存 | supported for sum/max/min source semantics with constant init | tile verifier与tile→instruction baseline均拒绝dynamic init；accepted op按canonical lexicographic order展开为init-first fill/movement/map-free elementwise composite，不直接生成native reduce。`avg`尚无可直接识别的source combiner，`mul`不伪装已支持kind。 |
| passthrough `linalg.generic` for broadcast / transpose / copy | exact body-only passthrough yield根据permutation-only indexing map lower到broadcast/transpose/copy | supported for static permutation maps | result map必须identity；payload存在任何额外op/effect时不能整op替换。 |
| `tensor.extract_slice` / `tensor.insert_slice` | static slice lower到tile movement；external extract形成DDR subview/load；direct-yield insert仅在旧dest除该insert及可证明unread DPS-init链外无其它use时early store | supported for static slices，包括合法rank reduction | 否则必须先load旧dest并生成fresh functional insert result，统一在finish store，避免early store污染旧tensor SSA value。dynamic metadata结构化失败。 |
| `tensor.expand_shape` / `tensor.collapse_shape` | static element-count-preserving reshape lower 到 `wafer.tile.reshape` | supported for static shape-only reshape | `wafer.tile.reshape` 表达 canonical linear element order 保持不变、result multi-index 按新 shape 重新解释的 logical reindex；tile-region 层 op 本身无 write effect，但下游若当前 physical layout 不能 alias 该 logical reindex，必须 materialize 成 explicit movement，不能用 reshape 逃避 physical layout。 |
| `scf.if` | 保留为tile-region内structured control-flow；condition使用scalar SSA，then/else body递归lower，tensor result/yield value以SPM memref result穿过`scf.if` | supported for single-block `scf.if` with supported nested ops | 分支内局部value不泄漏；只有`scf.yield` result重新进入父scope。外部scalar必须作为rank/task-program显式input或在enclosing structured scope内定义，并由structured scope的`IsolatedFromAbove`合同直接保证。本层不展开分支或选择硬件branch指令。 |
| `scf.for` | 保留为 tile-region 内 structured loop；lb/ub/step 使用 scalar SSA，iter_args 中的 tensor value 转为 SPM memref loop-carried value，body 递归 lower，`scf.yield` 传回 SPM memref/scalar | supported for single-block `scf.for` with supported nested ops | loop-carried tensor 只表达 tile-local buffer dataflow，不做 unroll、trip-count planning、SPM offset planning 或 hardware loop/branch instruction selection。并行 loop / while / execute_region 不在本阶段放开。 |
| `wafer.linalg_ext.collective.all_gather` / `reduce_scatter` / `all_reduce` | rank-specialized task materialization根据`logical-rank` context、`rank_group`、SPM buffer shape和combiner region生成`wafer.tile.all_gather` / `wafer.tile.reduce_scatter` / `wafer.tile.all_reduce`；当前入口是structured task materializer | supported for top-level single-result V0 collectives | `logical-rank`只用于计算`rank_group`内的group-local `local_rank`，输出IR显式保存`rank_group`、`local_rank`、`group_size`和`bytes`；不选择p2p schedule，不写endpoint或DTE packet。`reduce_scatter` materialization保留full input SPM buffer，并让`wafer.tile.reduce_scatter`显式携带scatter `axis`；recv/result buffer是当前rank的local slot shape。 |
| `wafer.linalg_ext.collective.collective_permute` | rank-specialized materialization 根据 source/target pair 和当前 logical rank 生成直接 DTE send/recv/wait；非本 rank result 由 numeric zero fill 表达，自发自收用 local copy | supported for top-level single-result V0 permute | 只覆盖 shape-preserving single input/output permute；peer 仍是 logical rank，physical endpoint / DTE resource 留给后续边界。 |
| `wafer.linalg_ext.collective.all_to_all` | rank-specialized materialization 要求 `split_count == rank_group.size()`，把 split slot extract 成连续 SPM comm buffer，按 rank order direct DTE send/recv，再把 recv slot insert 到 concat result slot | supported for top-level single-result V0 all-to-all | 当前不引入 `wafer.tile.all_to_all`，也不保存 algorithm attr；只覆盖静态 shape、单输入/单输出、single rank-group row 可选中的 V0 direct p2p path。 |
| `wafer.linalg_ext.collective.segmented_all_to_all` | materialize为`wafer.tile.segmented_all_to_all`，保留SSA counts/displacements、static capacities、count-exchange和data-phase token | long-term contract fixed；implementation pending | V0可先选择`padded_fixed_capacity` policy；不得把ragged route伪装成equal-split all-to-all或按当前token count增删rank artifact。完整verifier/lowering见`tasks/13-communication.md`。 |
| nested collective | 当前无 nested control-flow materialization | explicitly deferred | 需要先补可验证的 nested buffer slice / peer / token / schedule 表达；不能把 unsupported collective 静默降成名字约定或 pass-local side table。 |
| unknown op inside structured task source | 结构化失败 | unsupported | production materializer不得留下半转换task candidate；unsupported body op导致candidate conversion failure，source clone保持不变。 |

表中的 `rank_group` 只表示collective membership，`logical-rank` / `local_rank`只表示该schedule的显式参与者
身份；它们不允许承担bundle identity、去重或省略rank entry的语义。

表中typed mapping与unsupported/deferred failure gate由structured materializer单一拥有。扩展coverage时
每新增一类op都要同时补：IR语义、verifier、conversion pattern、negative test和下游instruction/memref demand来源。

`wafer.tile.region` 可以跨这些 lowering 子阶段保留为 region container。当前长期边界是：buffer
value 使用带 Wafer memory attr 的 `memref`，lower-level descriptor 只在 resource projection / emission
能验证其语义时出现。因此不能简单说 tile region 内“永远不允许 memref”或“永远不允许 lower-level
op”。正确边界是：

- abstract tile-region 阶段只允许 verifier 可解释的 Wafer-tagged `memref.alloc`、metadata view 和
  Wafer movement/compute op；不允许 generic `memref.load/store/copy` 作为语义逃逸。
- resource projection / realization 后允许 verifier 可解释的 placed `memref` / descriptor / lower-level Wafer op。
- LLVM call、runtime call、target CRT call 不属于 `tile_region` 主体，应在 target LLVM / launch lowering 后
  出现。

## 3. Op Contract

`wafer.tile.region` 的概念合同：

```text
wafer.tile.region (...) -> (...) {
  ^bb0(%traversal_indices..., %region_args...):
    ...
    wafer.tile.yield ...
}
```

必须表达：

- region argument/result与enclosing rank task/data/event edge或static-rank function boundary的SSA关系。
- 当前 traversal iteration 的逻辑 indices；static execution identity 由 enclosing executable rank/entry
  mapping 提供，不在 region 内复制。block id、physical coordinate 和 endpoint descriptor 只属于
  accepted transport binding。
- external input/output、constant source和candidate显式spill edge的load/store boundary；跨task/region resident
  value保持SPM SSA，不因scope边界自动写回。
- tile-local storage ownership、memory space、layout 和 effect。
- async issue 与对应 fence/wait/barrier。

不应表达：

- raw register packet field。
- runtime allocation handle。
- tile/residency/layout planner的rejected candidate或search trace。
- case-specific K tile、psum lifetime 或 epilogue scheduling 作为固定 protocol。

full-buffer resident finalization只接受由当前IR封闭证明的完整handoff：producer output必须对应
compiler-owned、single-use、`Tensor` layout的DDR `memref.alloc`；producer必须以static full WDMA从region-owned、
exact SPM Tensor buffer写出，且WDMA后只允许local fence/yield。跨producer result到sibling operand的external
view chain当前只接受等physical storage的static collapse/expand/cast；producer region内WDMA destination和consumer
region内RDMA source的local chain还可接受zero-offset、unit-stride、full-size static `memref.subview`。每个terminal
transfer必须是complete RDMA。验证在改写前覆盖全部uses和全部consumers；external full subview或任一consumer不安全时，
整个producer-result edge保持spill，不做partial promotion。未来扩大external view proof是性能follow-up，不是correctness
fallback。

安全改写后，producer `wafer.tile.region`显式返回SPM result，每个sibling consumer以显式SPM operand接收该值，
consumer内重建的static full view仍是SPM view，原complete RDMA被local SPM gather替换。该合同不允许
跨region隐式捕获、side table binding或名字匹配，也不把deterministic maximal promotion误写成generic per-edge frontier。
whole-rank lifetime analysis必须沿yield值、region result和sibling operand传播同一个allocation root；否则会错误缩短
lifetime并低估SPM high-water。当前7B rank-0结构证据中的2,725,568-byte high-water实际包含这种跨region result
handoff。tiled producer若只暴露多个partial tile而没有一个完整resident root，则保持显式spill；当前gate/up output和
projection-to-collective input属于该边界。

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

在 resource projection / realization 前：

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
| `wafer.tile.store` | 写回external output或candidate显式spill DDR value | destination range、layout、visibility、effect |
| `wafer.tile.materialize_layout` | 显式 layout conversion | source/result layout relation、可消除冗余转换 |
| `wafer.tile.*` compute ops | target-abstract compute | operand/result layout、instruction family legality、workspace/psum demand |
| `wafer.tile.*` collective ops | buffer-level collective semantic；由 tiled tensor collective + SPM buffer + endpoint view materialize | rank group、local rank、fixed byte count、buffer lifetime |
| `wafer.instr.dte_*` ops | Direct DTE p2p hardware invocation schedule；由 accepted tile collective algorithm materialize | peer、token、fixed byte count、communication lifetime |
| `wafer.instr.local_fence` 和后续 sync boundary | local fence、comm wait、barrier | async op completion、effect ordering |

这些 op 的具体算法分别归 layout、SPM、DDR、compute、communication 文档。`tile_region` 只负责
把它们放在一个可验证 execution scope 里。

## 6. Lowering Responsibilities

实现上可以分多步，但每一步只读取当前IR并直接改写candidate clone：

1. task traversal proposal：tasks/06 scheduler从rank-local structured source构造tile coordinates、受数值gate
   约束的reduction split和scope-prefix partition；当前V0保持deterministic source/DAG order，并让同scope
   dataflow edge直接成为resident SPM SSA。本层不从group数量或顺序恢复这些决策，也不伪造尚未实现的generic
   per-edge residency、task-order或layout-cut候选。
2. target-abstract task materialization：在完整rank clone中把每个structured source task映射成
   `wafer.tile.*` compute/movement/collective，并用一个或多个`wafer.tile.region`组织traversal fragment。
   task间SPM memref和event显式跨region传递。typed op mapper由该materializer单一拥有，
   不引入隐式boundary/default DDR round-trip。
3. layout assignment：在完整rank-entry lifetime/resource context中选择candidate physical layout marker，并在
   cut edge插入`wafer.tile.materialize_layout`；committed memref type/explicit movement是唯一owner。
4. DDR boundary和spill materialization：只为external input、parameter/constant source、final output及candidate
   显式spill edge生成`memref.subview`和`wafer.tile.load/store`。static `tensor.extract_slice`、
   `tensor.insert_slice`及indexing-map image提供真实tile offsets/sizes；这一步不重新选择candidate。
5. Wafer instruction legalization/selection：把target-abstract executable task降成`wafer.instr.*`；accepted
   collective algorithm降成`wafer.instr.dte_send/recv/wait`。instruction IR显式列出instruction family、
   read/write/effects、descriptor attrs、temp/psum/staging memref、alias/view和completion。
6. post-task-commit canonicalization与bounded storage alternatives：全部selected task fragments已写入完整rank clone后
   只运行一次canonicalization，消除static one-trip traversal wrapper和等价full view；从同一canonical committed rank
   派生保留原WDMA/RDMA的spill baseline，以及按稳定IR顺序提升全部安全full-buffer handoff的deterministic maximal
   resident alternative，不枚举promotion subset。
7. whole-rank SPM offset assignment：分别消费两个完整rank alternatives的instruction task graph、structured control flow
   和event lifetime，
   为跨task/regionresident buffer分配offset/end/bank span并做lifetime reuse；单region peak只可作lower bound，
   不能独立提交arena。
8. whole-variant DDR offset assignment：消费external boundary、显式spill和descriptor demand，验证view/root range、
   arena capacity/largest-contiguous、shared bandwidth、alignment、overlap和completion；不得把task/region边界
   自动解释成DDR resource。
9. bounded candidate evaluation：对每个complete clone运行layout/instruction/SPM/DDR/event/transport/ABI gates；
   first/tail shape可作cheap prefilter，但accepted candidate必须materialize完整traversal。失败clone整体丢弃。
10. rank frontier finalization：每个scheduler alternative独立执行function-boundary bufferization、whole-rank SPM/DDR
    replanning和fresh cost recomputation；只过滤失败alternative，不因一个失败项丢弃仍有survivor的rank。
11. physical transport/all-rank verification：在accepted ranges和exact topology上完成cross-rank message matching、
   transport binding、resource/status closure和target-entry ABI preflight。
12. whole-variant atomic commit：只在complete passing candidates之间选择，原子替换全部rank programs；committed
    entries覆盖完整traversal，消解tensor/Linalg/`wafer.group`并与accepted transport binding一起提交。
13. target instruction LLVM call emission：从committed instruction IR、typed rank record、accepted offsets和
    transport binding生成wrapper-friendly LLVM/target CRT/packet builder输入；不新增placed memref/access
    descriptor旁路协议。

任何会引入communication staging buffer、send/recv token lifetime、reuse fence或local-fence requirement的lowering
必须发生在SPM assignment之前，使planner能从IR/effect/lifetime看到真实需求。rejected candidate不得落入主IR等待
下游修复；当前失败反馈到tile/reduction size、scope partition或candidate selection；未来加入generic
resident/spill、task order或layout cut时仍沿同一反馈边界，不能变成新的隐式materialization boundary。spill与
resident alternatives分别运行SPM、DDR、verifier/completion和cost重算：resident任一resource/verifier gate失败或
有效cost不优于有效spill时保留spill；spill无效而resident有效时可选resident；二者都无效才拒绝基础rank candidate。
任一alternative的改写和offset都不得泄漏到另一alternative。

static one-trip `scf.for`在candidate traversal materialization和selection期间仍表达multiplicity、coordinate、resource和
cost语义，不能因为trip count为一就提前折叠。第6步canonicalization只能发生在全部selected task fragments提交到完整
rank clone之后，并且两个storage alternatives必须从同一个canonical rank派生。

`scf.if` / `scf.for` 在structured materializer中是tile-region内的结构化control-flow container，不是
instruction-level branch/loop。materializer只负责把region内tensor dataflow递归降成Wafer-tagged
SPM memref dataflow，并让 `scf.yield` / loop-carried value 显式携带 SPM memref 或 scalar SSA。
instruction lowering 只递归 legalize control-flow body 内的 executable target-abstract op，并保留 `scf` container。
SPM memory planning 后续应直接消费这些 region/control-flow lifetime；硬件 branch/loop、predication 或
展开只能在 committed instruction / codegen 边界具备足够 lifetime 和 effect 信息后决定。`scf.while`、
`scf.forall`、`scf.execute_region` 和 CFG branch 需要额外 memory-effect / concurrency / multi-block
合同，当前必须结构化失败。

当前ODS已有`wafer.tile.region`内movement、layout、compute、comm和sync op的基础
layout/materialization/resource interface 查询入口；这些接口和 verifier 已基于
`memref<..., #wafer.memory<space, layout>>` 合同。第 4 步必须让 candidate boundary slice 的 DDR
operand 表达真实 tile view；第 5 步 instruction legalization / selection 再把 target-abstract op
降到 instruction-level IR；第 6-8 步分别完成 SPM planned offset、DDR memory planning
和closed-loop candidate driver，不能互相推迟协议补全。该段描述当前op/interface资产和已接入的
whole-rank task-dataflow production path。

当前materialization边界：

- `wafer-schedule-tensor-program`使用`Passes.td`声明，在whole-rank candidate clone上调用
  `WaferTensorProgramToTileRegion` typed API。production driver把当前rank作为typed C++ API参数显式传入；
  IR-local pass中的`logical-rank`只用于测试显式IR，不是production identity通道。
- explicit static boundary slice producer已接入：boundary `tensor.extract_slice`生成DDR
  `memref.subview` + tile load，direct output `tensor.insert_slice` storeback生成DDR `memref.subview` + tile store；
  instruction lowering消费这些view并生成RDMA/WDMA三层stride/iteration descriptor。slice proposal
  直接由linalg indexing maps和candidate tile offsets/sizes派生，不经过独立dump artifact。
- 已退役的single-tile、multi-tile-no-comm和调度容器conversion/debug pass仍保持不可用；
  negative tombstone只防止旧CLI回归，不形成compatibility pipeline。

当前主线已用compact `scf.for`加static tail表达multi-tile traversal；这与已删除的旧
multi-tile-no-comm debug pass无关。后续仍需建立per-tile block id、physical coordinate、local shard slice和
launch args/identity lowering的IR contract；logical rank继续来自typed rank-local driver context。

## 7. Verifier

`wafer.tile.region` verifier 至少检查：

- region argument/result 和 terminator 类型匹配。
- region 中没有无法解释的 side table dependency 或名字匹配语义。
- external load/store boundary 都有明确 memory space、shape、dtype、layout 和 effect。
- async producer 的 source/destination 在 fence/wait 前不能被非法复用。
- region 是完整 traversal 中的局部 scope；它的 iteration offsets/sizes、boundary subview 和 yield/writeback
  必须与外层 structured control flow 一致。单独的 first/tail representative region 不能满足 coverage gate。
- `wafer.tile.materialize_layout` 的输入输出 layout relation 合法；同 layout 冗余转换应由 verifier
  拒绝，上游应避免生成这种 no-op conversion。`wafer.tile.reshape` 这类无副作用 view op 可由
  上述post-task-commit canonicalization删除同类型no-op。
- `#wafer.memory<spm, *>` memref在executable commit/target-codegen前必须经过whole-entry SPM allocation；
  compiler-managed `#wafer.memory<ddr, *>` alloc 必须有 DDR memory planning 接受的
  `wafer.ddr.offset` fact；external DDR boundary value 的 descriptor/view/root validation 由当前
  instruction-level IR 重算。
- Q16 commit前必须从当前candidate IR、program shard declarations、accepted offsets和accepted transport binding重算并验证
  resource/entry/completion facts，再materialize为typed C++ `RankExecutable` record；post-commit target只派生
  lower-level address/range，package只序列化typed owners。不能要求
  tile-region 主线额外携带 placed memref 或 access descriptor 旁路事实。

whole-variant verifier另行检查committed program不残留tensor/Linalg/`wafer.group`，所有static rank entries和
source outputs都有all-and-only完整coverage，最终layout和全entry SPM/DDR facts唯一，所有async issue均有
显式 completion/terminal drain，跨 rank transport 成对，且 shared physical geometry/range/narrowing
合同通过。任一失败都使整个 candidate 不可提交。

tile-region verifier不决定whole-rank candidate是否最优；当前traversal/reduction/scope-prefix policy由tasks/06
scheduler提出并由完整gates验证，generic per-edge residency/task-order/layout-cut仍是后续性能维度。
`wafer.tile.region`这类region op的boundary invariants放在普通`verify()`，body/
terminator/nested region关系放在`verifyRegions()`；父region op只解释自己body的直接op，不递归解释tensor
collective等子op的内部region。structured scope verifier与materializer negative tests直接证明typed mapping和
`IsolatedFromAbove`覆盖。

## 8. 与 Case 的关系

设计文档可以用典型 case 展示：

```text
load A/B tile -> matmul -> optional epilogue -> store outputs
```

但 case 中的 K tile、psum lifetime、epilogue 位置、double-buffer depth、output order 都只是
planner 候选结果，不是 `wafer.tile.region` contract。多输出、不同 output domain、hidden
dimension、layout cut 和 memory split 都由 op interface、SSA use-def、effect 和 verifier 处理，
不能通过 case 名字固定。
