# Layout Movement Elimination 实施计划

状态：设计已收敛，待实现。任务状态以`tasks/progress.md`中的`layout-movement-elimination`为准。

本任务在现有physical-dataflow candidate owner内联合处理逻辑view、physical encoding、compute implementation、
fanout版本共享和真实materialization位置。目标不是减少IR中所有view数量，而是减少最终程序真实执行的
SPM/DDR/NoC搬运、descriptor、等待和额外live storage。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  post-SPMD、静态shape、verifier-legal的structured tensor program，以及candidate owner创建的
  isolated complete-rank actual clone；op语义、indexing maps、SSA、effect和typed encoding均已显式。
- Current stage responsibility:
  从当前IR重算IndexRelation、physical access、alias/lifetime和invalid-lane事实；传播并组合view relation；
  联合选择compute implementation、Tensor/NTensor/Cx/NCx、fanout共享版本和必要materialization；每个方案
  立即物化为actual clone并重跑现有exact gates。
- Output artifact / IR:
  至多四个自包含optimized actual clones及独立baseline。PBQP projection、assignment、relation proof和
  cost hint在clone进入现有frontier前销毁；最终只有现有candidate owner提交的typed winner IR进入bundle。
- Downstream consumer:
  TileRegion到Instr conversion、SPM/DDR planning、completion/resource/transport gate、target publication、
  package、SystemC和板端执行。
- User-level driver / named pipeline:
  现有wafer-compile source-to-bundle production pipeline；wafer-opt仅作局部replay/test。
- Explicit non-goals:
  不新增layout dialect、VirtualTensor、Linear Layout、shadow graph/plan、layout demand或physical-access接口、
  `TargetProfileId`查询参数、conversion matrix、solver result attr、外部ILP/CP-SAT或第二个decision owner；
  不支持无法由当前IR精确证明的dynamic-shape关系。
- Completion gate:
  targeted source的非连续view与重复layout movement在selected IR中真实消失；baseline始终保留；
  host、完整package/fresh no-card和FP16/BF16真实板端正确性、guard及matched性能门禁通过。
```

## 已收敛设计

### 1. 复用现有事实源

- `IndexRelation`只表示logical index关系，继续支持composition、identity、domain/image、functional、
  injective/bijective及piecewise证明。
- `WaferPhysicalEncodingAttrInterface`继续独占footprint、alignment、valid/padding、mapping和segment事实；
  查询签名保持不变，不接收target profile或`TargetProfileId`。
- `PhysicalAccessRelation`、`TransferRealizability`和`MovementSupport`继续分别拥有physical组合、route/view
  可实现性和descriptor构造；typed op verifier继续拥有算子合法性。
- `InvalidLaneState`保持由当前IR派生的`NoInvalidLanes`、`KnownSplat(value)`、`Unknown`，不进入attr或side table。

### 2. Relation-guided view motion

- 在pure SSA区域沿def-use双向运行固定点，按现有`IndexRelation`组合reshape、transpose、slice、broadcast、
  concat和其它structured indexing relation；只用existing exact `isEquivalentTo`在相同domain内去重，不用文本、op名或
  relation side hash恢复语义。
- 固定点有独立hard cap：每个`(origin value, current value)`最多保留
  `IndexRelationLimits.maxDisjuncts`个exact canonical alternatives，每条def-use edge最多接受同样数量的新relation，
  全scope最多处理`maxDisjuncts * (value count + use count)`个worklist item并使用现有32-variable/8-disjunct
  relation budget；checked overflow或任一上限耗尽时停止该connected component的优化并保留baseline。
- 不实际把effectful op交换位置。对可穿越算子构造重索引后的actual op candidate：逐位置算子组合
  indexing maps；GEMM只使用现有orientation与batch/M/N/K语义；reduce同时验证dimension/result mapping、init、
  combiner、valid-lane/neutral、effect/completion和既有numeric contract，floating只是不要求保持leaf order。
- concat始终保留每个input root各自的`staticConcatPiece`，并证明pieces互斥且完整覆盖destination；多root concat不能成为
  单一metadata alias，但这些pieces可以合并为写同一actual destination的compound movement。
- relation组合为identity时删除跨多个非连续op的view；单root physical map等价时形成metadata view；否则把每个
  source root的canonical composed relation交给现有movement realization，在assignment选中的cut只物化一次。
- unknown effect/alias、无法对齐的fanin、非不变loop-carried relation或证明预算耗尽均形成边界并保留baseline。

### 3. Layout、implementation与fanout联合分配

- 不扩展`TargetImplementationCandidate`，也不新增layout-demand/access-constraint接口。existing source interface仍只枚举
  implementation kind；每个kind先在disposable clone中调用现有materializer，再从materialized typed op/subgraph的external
  DPS tie、indexing maps、operand/result memref和当前可构造的Tensor/NTensor/Cx/NCx attr生成port tuple probe。
- probe按stable port/encoding ordinal增量组合，先用`IndexRelation`和`PhysicalAccessRelation`剪掉不兼容tuple，再用现有
  op builders重建typed op/subgraph并运行每个concrete op verifier与同一lowerability preflight；只有成功的actual probe进入PBQP domain。
  每source op最多保留32个accepted states，超过时先保留每个implementation kind的stable baseline tuple，再按local
  materialization-byte lower bound、footprint、kind/port ordinal保留nondominated states；若每kind baseline本身超过32，
  该component不进入PBQP并保留existing candidate path。probe clone随后销毁，不能把
  compatibility table或verifier结果保存为跨pass事实源。
- invocation-local候选模型包含两类变量：op变量的状态只是上述已probe通过的`implementation kind + complete port tuple`
  ordinal；value变量的状态是一个primary encoding和至多一个共享secondary encoding。它们只引用当前IR anchor与稳定
  ordinal，不复制shape、relation、descriptor或target事实。
- producer-result factor要求value primary与actual producer result encoding一致，或由Exact metadata relation等价；public ABI
  boundary root固定为compact Tensor，内部producer/result仍可选择Cx/NCx并通过显式或mapped boundary movement连接。
  选择secondary时，其一次共享materialization bytes/command/footprint只记在value unary cost；没有secondary时不收费。
- consumer-operand factor在port demand能由primary/secondary exact满足时为zero；否则只在existing movement realization可表达时
  记录该consumer私有materialization cost。多个consumer需要同一第三种encoding时会分别收费，因此solver可以选择共享secondary，
  但不会凭side state假设转换已存在。
- fanout的secondary在producer后只创建一次，所有兼容consumer复用同一真实SSA version；不按consumer edge
  重复收费，也不枚举`2^fanout`子集。已有share-vs-recompute clone在本算法外形成，本算法分别求解。
- 兼容physical traversal或metadata view的边成本为零；真实转换复用现有movement bytes和可得的command
  lower bound；typed verifier明确拒绝才为不可选。资源、lifetime、completion和最终性能只由actual clone gate判断。
- proposal objective依次使用可证明movement bytes、unknown-command count、known command lower bound、额外physical
  footprint和稳定ordinal；
  最终winner仍只由现有`InstructionProgramCost`、Pareto和target static policy决定。
- 不触及collective的component保持rank-local。任何连接all-rank collective port的component提升到现有whole-variant
  coordinator：按semantic collective ordinal建立一个跨rank共享变量，其domain是各rank probe-accepted encoding/mapping
  states的exact交集，PBQP一次提出并原子物化完整rank tuple。Top-4上限作用于coordinated tuple，不对per-rank shortlist做
  Cartesian product；任一rank失败丢弃整个tuple。

### 4. Production PBQP

这里选择PBQP是因为domain probe后，operation implementation/port tuple与value primary/secondary都已成为有限label，
真实耦合可以分解为unary和producer/consumer pair factor；低度图可精确消元，高度核心再受统一预算约束。e-graph适合生成
等价relation/rewrite但不负责physical version、fanout shared cost和actual resource gate；tree-DP依赖低treewidth；ILP/CP-SAT
虽可作离线oracle，却会引入外部依赖、不可控求解延迟和第二套legality模型。因此PBQP只负责有界提案，最终合法性与选择仍由
actual clone和现有owner承担。未来若出现不能安全分解的higher-order constraint，应在outer actual-clone枚举中保持有界，不能
把它偷偷压成pair cost。

- 生产只实现确定性PBQP：connected-component拆分、dominance pruning、degree-0/1/2精确消元、高阶核心的
  有界确定性分支及单变量/双变量局部改进；只有state A的unary及对每个neighbor-state的pair cost都不高于state B，
  且至少一项更低时才能删除B。不实现生产穷举器，不依赖外部solver。
- 只有exact relation/physical proof能产生zero-cost edge；unsupported/resource-exhausted proof不能授权zero-copy，但仍可选择
  已有explicit materialization edge。movement bytes必须known；command lower bound未知时保留可行性并在objective中增加
  `unknown-command`计数，不能把Unknown当0或infinite。
- PBQP matrix使用checked `ProposalCost{movementBytes, unknownCommandCount, knownCommandLowerBound,
  extraFootprintBytes}`，逐字段相加并按该顺序lexicographic比较，不使用任意加权标量；只有concrete verifier或
  existing movement realizability明确拒绝的state/edge是infinite。
  stable state vector只用于完整assignment同cost时的tie-break，不参与cost算术。
- invocation-wide `solverWorkBudget`初值为1,048,576；dominance scan、degree-0/1/2消元、pair-matrix更新、
  reconstruction、branch/local evaluation、Top-4子问题重解和heap candidate评分中，每计算一次candidate
  `ProposalCost`加法或比较都checked消费一个work unit。所有component与Top-4子问题共享该counter；耗尽后停止产生
  新proposal，只保留已经完整求得的unique assignments，任何未完成assignment丢弃，baseline不受影响。每个dominance/
  reduction更新先checked计算完整step upper bound，只有预算足够才原子提交，不能保留半张matrix；Top-4使用固定sift顺序的
  deterministic binary heap，使heap比较次数和budget消费不依赖library实现。
- 高阶核心按incident known-movement impact降序、stable SSA ordinal打破平局；state按partial objective和state ordinal展开。
  invocation-wide `searchExpansionBudget`初值为4096；每个branch child或完整one/two-variable local neighbor计一次，且其
  cost计算仍消费`solverWorkBudget`。search budget耗尽后只在剩余solver work内按相同顺序greedy补全current incumbent；
  solver work不足则丢弃该未完成proposal，baseline不受影响。
- 单次component solve只返回一个complete assignment。Top-4使用带继承约束域的deterministic prefix-partition：heap node保存
  该子问题的完整forced/forbidden state constraints及其solution；pop `(constraints, vector)`后，对每个stable variable position
  建立“继承constraints、此前vector prefix固定、当前位置禁止vector当前state”的子问题。各子域按first-differing position
  互斥并覆盖parent domain除当前vector外的全部assignment；求解后按proposal objective放入按canonical constraint set和完整
  state vector去重的min-heap。重复pop/partition直到四个unique vectors或共享solver/search budget耗尽。该方法只有在所有
  已需子问题都exact闭合且预算未耗尽时称为k-best；有界核心或预算触发时只称ordered proposals，不宣称最优。
- component assignment按`(movement bytes, unknown-command count, command lower bound, extra footprint,
  complete state-ordinal vector)`排序；identity就是stable variable顺序下的完整state-ordinal vector，不使用模糊的
  “layout-diverse”判定。
- 多个independent components各自的有序Top-4用deterministic min-heap做k-best sum：从全零component-index vector开始，
  每次只递增一个component index、去重index vector并最多pop四次，因此不展开完整Cartesian product。
- 最多四个global optimized assignments各尝试物化一次；某个assignment失败不回填第五名。另保留独立baseline，并继续受
  现有96次optimized rank evaluation、256 general rank frontier和whole-variant hard cap约束。
- 每个assignment重新clone current IR、fresh重建relation/proof并立即创建typed view/movement/compute；验证失败
  只丢弃该clone。assignment和PBQP projection在clone进入worklist前销毁，下游不读取solver结果。

### 5. Current target纵向

以下是Q46针对当前TX81 target的终态合同，不表示pre-Q46 compiler已经实现这些候选。

- same-order elementwise、relation、logic、select、bitpacked和compatible convert均可在Tensor/NTensor/Cx/NCx上
  形成候选；判定依据是typed verifier和physical traversal compatibility，不是op-family Tensor白名单。
- pre-Q46 `MemoryAttr` implementation尚拒绝bitpacked Cx/NCx；Q46在同一个existing
  `WaferPhysicalEncodingAttrInterface` implementation内补齐`i1`的block/tail/bit offset，不增加接口或第二事实源。
  dtype转换遇到不同block geometry时，只有composed physical access证明兼容才直接执行，否则保留一次显式materialization。
- pointwise可以遍历完整physical footprint；invalid output lane可保持`Unknown`。会混合lane的reduce/GEMM等仍须
  满足现有neutral、mask或valid-domain合同。
- Reduce保持归约维度映射正确；浮点归约默认允许重排，沿用现有数值验证。native与composite均形成actual
  candidates，composite内部步骤继续参与layout分配，不结构性强制Tensor。
- Tree AllReduce在全部参与rank具有相同encoding与完整physical mapping时直接处理Cx/NCx physical footprint，
  包括tail/padding；AllReduce/ReduceScatter Ring仅在每个chunk都有exact、互斥physical-range证明时生成对应候选。
- public input/output ABI继续保持compact Tensor。full-buffer resident handoff在descriptor展开前转交producer已有
  SPM version；不兼容fanout保留DDR spill sibling，late transfer elimination继续作cleanup。

## 实施 Checkpoints

1. 补齐bitpacked Cx/NCx physical mapping与`InvalidLaneState` transfer tests，保持encoding interface签名不变。
2. 在hardware-supported row上放宽现有compute/instr concrete verifier的固定Tensor限制，并将lowering中的固定Tensor
   优先选择改成consumer typed demand；不增加access-constraint接口，metadata reshape传播全部兼容已有版本。
3. 实现relation固定点、重索引候选和compound relation realization；先闭合chain、diamond、fanin/fanout正负例。
4. 实现disposable actual-op domain probe、deterministic PBQP projection/Top-4、rank-local与collective coordinated
   actual-clone materialization，并接入现有candidate owner。
5. 接入pointwise/convert、reduce、Tree/Ring collective和pre-descriptor resident handoff纵向。
6. 重跑完整exact gates，生成FP16/BF16 package并推进到board-ready；真实板端matched验证通过后才能完成。

## Verification Contract

- relation：非连续inverse view、非identity多view合并、reshape/transpose/slice/broadcast/concat、effect/alias/control-flow barrier；
- physical：Tensor/NTensor/Cx/NCx full/tail、bitpacked relation/select、unknown padding、neutral reduce、跨dtype block mismatch；
- solver：domain probe不扩接口、fixed-point/probe/solver-work/search-expansion hard cap、deterministic component Top-4与k-best merge、共享secondary
  只物化一次、双版本cap、失败不回填、baseline保留、无solver residue；
- vertical：GEMM到pointwise/convert/reduce链、Tree AllReduce full footprint、exact Ring chunk正负例、跨tile-region resident handoff；
- production：同源baseline/winner均经source-to-package、fresh no-card、output/guard验证；板端使用FP16/BF16，
  targeted winner必须减少final IR movement bytes/commands且matched性能不劣于baseline。

只完成文档、接口声明、局部FileCheck、proposal生成或单个case的IR op数下降均不算实现完成。
