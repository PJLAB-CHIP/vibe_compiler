# Wafer Logical Group 与 Candidate 设计

状态：2026-07-12按当前实现重基线。本文拥有tensor-level`wafer.group`、tiling-demand analysis、candidate
generation/materialization/legality/ranking/commit和complete-traversal合同；Q16才把这些能力重放到全部static
rank clones并形成ExecutableBundle。实现状态看`tasks/progress.md`。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  Q15中post-SPMD StableHLO经过local normalization得到的`linalg`/`tensor`/`scf`/`arith`/`math` tensor IR，
  以及`wafer.linalg_ext.collective.*`logical collective。
- Current stage responsibility:
  先形成dependency-preserving logical `wafer.group`；在Q16每rank isolated clone内，从当前IR重算tiling
  demand，生成bounded candidates，完整materialize traversal，运行layout/SPM/DDR/instruction/geometry/
  completion legality，只提交一个完整passing candidate。
- Output artifact / IR:
  Q15输出verified logical-group program；Q16输出每rank不再含`wafer.group`的accepted instruction/memory/
  completion program，并由typed `RankExecutable[]`组成atomic `ExecutableBundle`。
- Downstream consumer:
  Q17 target conversion/device link；Q19 reference executor。二者只消费accepted rank executable，不读取
  rejected candidate、group search trace或debug dump。
- User-level driver / named pipeline:
  production只经`wafer-compile`。`wafer-opt`、registered group/instruction pipelines和dump passes只处理显式IR；
  其中固定logical-rank=0的named pipelines只用于IR-local replay，不能证明all-rank bundle。
- Explicit non-goals:
  `wafer.group`不保存physical layout/address、SPM/DDR offset、DTE route、target CRT、manifest、rank class、
  planner trace或cost breakdown；不要求用户选择tile-search pass、stage或stop point。
- Completion gate:
  logical group由真实Q15输出形成并通过verifier；Q16对rank-count=1/16建立all-and-only isolated clones，
  每rankaccepted result覆盖完整traversal并通过全部legality gates；任一group/rank失败不形成partial bundle。
```

## 2. 核心边界

`wafer.group`表达tile-local residency与fusion planning boundary，不是新的数学op，也不是committed executable：

```text
local tensor graph
  -> logical group boundary
  -> candidate-specific complete tiled materialization
  -> accepted tile/instruction/memory program
```

logical group说明哪些tensor SSA producers/consumers和destination-style outputs可以在同一候选中共同规划。tile
shape、reduction split、layout assignment、buffer offsets、engine、event和physical endpoint只在能解释它们的下游
candidate IR中出现。

candidate可以被cheap prefilter提前拒绝，但任何accepted artifact都必须重放完整output traversal和reduction
chunks。单个representative tile、first/last tile、shape-only dump或某个group通过都不是commit proof。

Q15的transaction到verified grouped program即结束。Q16为每个logical rank重新clone该program并运行candidate
链；rank clones彼此隔离，全部通过后才构造ExecutableBundle。当前不存在`wafer.executable.*` dialect、代表rank
或rank-class dedup协议，也不需要它们才能实现correctness-first static bundle。

## 3. `wafer.group` IR 合同

当前ODS定义：

```mlir
%results = wafer.group
    ins(%inputs : ...)
    outs(%destinations : ...)
    ({
      ^bb0(%input_args: ..., %out_args: ...):
        ... tensor-level body ...
        wafer.group.yield %values : ...
    }) : ...
```

op具备以下结构事实：

- `IsolatedFromAbove`，所有外部依赖通过explicit `inputs`/`outs`和region block arguments进入；
- 单block region和`wafer.group.yield`terminator；
- outputs/results为tensor且类型、数量与yield逐项一致；
- destination-style ties由body内DPS op和explicit outs表达；
- recursive memory effects用于阻止把未知side effect误当pure tensor fusion；
- `WaferTilingInterface`提供后续需求查询，而不是把tile plan写成attrs。

group不携带root kind、fusion reason、tile size、temporary list、rejected reason或cost。body已经表达执行数据流，
因此也不另存shadow schedule。

### 3.1 Formation

当前formation在同一block内按SSA use-def保守扩张：

1. root/hero是tensor-levelDPS `linalg.*`（`linalg.fill`不单独作为root）或
   `wafer.linalg_ext.collective.*`；
2. producer只有在其results的所有uses都能留在selection内时才吸收；
3. consumer只有在不会制造selection外live use冲突时才吸收；
4. group results是selection中仍有外部uses的tensor values；outs沿DPS tied-init关系追到group外destination；
5. 只吸收可静态证明且仅供body使用的support ops，例如`arith.constant`、`tensor.empty`、静态
   extract/insert-slice、expand/collapse-shape；
6. body按原block order clone，所有external values显式remap到block arguments。

当前故意不跨block，不吸收raw StableHLO、memref/runtime/target op或未知side effect。multi-use producer、不同
output domain和更宽shape-view只有在SSA/type/indexing/verifier可以证明时才能放宽，不能靠op/value名字。

## 4. Tiling-Demand Analysis

`GroupTilingDemand`是从当前group局部派生、可失效、可重算的analysis，不进入IR或artifact。它记录：

- boundary input/output/result values和对应slice loop dimensions；
- 每个body op属于support、Linalg、LinalgExt collective或failure；
- Linalg iterator types、indexing-map投影、operand/result slice；
- reduction result与reduction dimensions；
- collective interface提供的rank-group、axis和tile relation；
- unsupported/dynamic-unprovable relation的structured failure reason。

普通Linalg必须从`LinalgOp`structured semantics恢复需求，collective必须通过MLIR`TilingInterface`与Wafer
collective interface；不得匹配`matmul+bias+relu`等固定序列。debug dump只是同一analysis的文本视图，不修改
IR，也不能替代candidate materialization。

## 5. Candidate 责任拆分

candidate pipeline按五个稳定职责组织：

1. **Generation**：从static result shape、indexing maps和reduction dimensions生成bounded candidate specs；
2. **Materialization**：只在isolated module clone中构造candidate traversal、tile region和后续IR；
3. **Legality**：运行layout、SPM、DDR、instruction geometry/range、event/completion和target preflight；
4. **Ranking**：只比较已经完整通过legality的candidates；
5. **Commit**：重新重放complete materialization proof，然后用accepted clone替换当前rank module。

generation的search frontier与ranking score属于analysis。当前`WaferTargetPolicy`/`TileSearchPolicy`是compiler内部
policy；`maxSearchCandidates`、beam width、preferred tile sizes及debug pass options不进入public
`CompilationRequest`、program metadata或ExecutableBundle identity。若Q16需要在typed driver内选择effort，只能作为
compiler-private orchestration/default，不得把pass pipeline暴露给用户。

`DirectFullShape`是普通candidate，不是fallback或绕过legality的平行pipeline。若它或tiled candidates均不合法，
当前correctness-first行为是整个rank/request失败；搜索不完备不能被误报为workload语义非法。

## 6. Complete Traversal

accepted candidate必须覆盖每个static result element all-and-only一次，并包含所有reduction contribution：

- output tile domain按checked ceil-div/product枚举，包含非整除tail；
- reduction split显式物化全部chunks，并由exact combiner把reduced value与accumulator连接；
- multi-output当前只接受equal static result shape、彼此独立、直接yielded的single-result Linalg roots；
- producer chain、不同output domain或无法由SSA证明的output relation对tiled path fail closed；
- conservative full-group candidate仍走同一legality和commit proof。

当前实现把output tiles/reduction chunks静态展开，并设置4096个materialization实例预算。checked overflow或
超限在commit前失败。4096只是当前unrolled实现的编译资源保护，不是硬件容量、IR语义、workload限制或16-rank
topology事实；长期用compact loop表示后可替换该实现边界。

representative materialization可以用于cheap SPM/geometry筛选和estimate，但`CandidateArtifactSource`必须表明它
不是complete artifact。selector在接受和commit时都调用complete-traversal API，防止first-tile success泄漏为
partial output。

## 7. 下游 Legality 闭环

candidate materialization依次建立或验证：

```text
logical group
  -> output/operand tile views
  -> wafer.tile.region + target-abstract compute/movement/collective
  -> wafer.instr.*
  -> accepted SPM offsets and lifetime/completion
  -> accepted DDR views/offsets
  -> physical geometry/range/ABI legality
```

group planner必须接受下游失败并换candidate或fail request，但不能把低层事实写回logical group attrs：

- layout materialization会改变physical bytes和temporary demand；
- SPM需要覆盖padding、workspace/psum、communication staging及completion之前的live range；
- DDR需要区分external view与compiler-managed allocation，并验证range/alignment/capacity；
- async issue的全部read/write resources必须活到可信completion；
- physical geometry和ABI narrowing在target conversion前闭合；
- Direct DTE在peer/slot/CRT合同尚未闭合时target-illegal。

任一late failure丢弃candidate clone。Q16外层再把同一规则提升到all-rank transaction：某rank成功不能授权保留其
artifact，直到全部rank通过。

## 8. Rank 与 Bundle 边界

Q16的rank domain只来自validated `ExecutionConfig`/execution mesh：

```text
for logicalRank in [0, rankCount):
  clone verified grouped program
  lower/select/plan with explicit logicalRank
  verify no logical groups remain
  verify instruction/memory/completion artifact
  create RankExecutable
verify all-and-only rank coverage
create ExecutableBundle atomically
```

每个rank clone必须显式传入logical rank；pipeline builder中固定rank 0的registered named pipelines只是debug
replay。禁止默认rank 0、从parameter shard filename推rank、只编代表rank或复用可变module跨rank。

`RankExecutable`至少需要携带rank identity、accepted module和Q17/Q19直接消费的resource/completion facts；具体typed
字段由Q16实现与consumer共同收敛。bundle不复制instruction schedule到另一份JSON，也不引入rank class。跨卡、MPMD
和hybrid specialization等真实consumer出现后再扩展。

## 9. 验证

局部IR coverage：

- group parser/printer/verifier、inputs/outs/results/yield type relation；
- formation的producer/consumer、multi-use boundary、support op和forbidden op；
- tiling-demand对matmul、elementwise、broadcast、reduction、multi-group和logical collective；
- unsupported op/indexing/dynamic relation负例；
- candidate generation/ranking与DirectFullShape同gate；
- one/multi/tail output tiles、reduction chunks、multi-output支持面；
- 4096预算与count overflow fail closed；
- late SPM/DDR/geometry/completion failure不修改source module。

Q16 integrated coverage：

- 直接消费Q15重新读取验证过的grouped program；
- rank-count=1与16全部创建isolated clones，无missing/duplicate/default rank；
- rank-specific collective/shard/entry facts可区分；
- 任一rank/candidate late failure无accepted bundle或partial final output；
- 全部rank通过后bundle readback仍无`wafer.group`且各rank artifact verifier-legal。

局部FileCheck、dump pass、手写group或rank-0 named pipeline只补覆盖，不能完成Q16。
