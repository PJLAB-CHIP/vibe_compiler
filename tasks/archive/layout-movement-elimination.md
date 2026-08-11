# Layout Movement Elimination 实施计划

状态：无卡阶段已闭合，等待真实板端matched验证。任务状态以`tasks/progress.md`中的
`layout-movement-elimination`为准。

本任务在现有physical-dataflow candidate owner内联合处理逻辑view、physical encoding、compute implementation、
fanout版本共享和真实materialization位置。目标不是减少IR中所有view数量，而是减少最终程序真实执行的
SPM/DDR/NoC搬运、descriptor、等待和额外live storage。

Q49集成说明：本文闭合的IndexRelation、PBQP projection和movement realization保留为proposal/action mechanics；
layout proposal不再先形成局部winner，而是与region partition、tile/loop、resident/spill/recompute、materialization和
communication进入06的同一query-local structural frontier；只有统一预算准入的有界代表才物化actual clone，exact
失败按稳定顺序从未物化frontier补位。每个complete static rank entry可包含一个或多个non-nested
`wafer.tile.region` SPM residency domains；layout/traversal边界不机械切region，region boundary本身也不自动产生DDR
movement或join。若candidate选择跨region cut，所有跨界data必须通过显式DDR store、可信completion和matching load表达；
SPM root/alias不得跨界，boundary只完成仍访问被释放roots的work。本文原有固定Top-K只记录Q46
当时的机制gate，不是Q49分层搜索或IR语义上限。
本文后续未显式标注Q49的Top-K、rank/whole frontier和candidate输出均是Q46历史完成记录，不是当前production协议。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  post-SPMD、静态shape、verifier-legal的structured tensor program，以及candidate owner创建的
  isolated complete-rank actual clone；op语义、indexing maps、SSA、effect和typed encoding均已显式。
- Current stage responsibility:
  从当前IR重算IndexRelation、physical access、alias/lifetime和invalid-lane事实；传播并组合view relation；
  对现有candidate recipe先物化真实compute implementation，再从该typed clone联合选择
  Tensor/NTensor/Cx/NCx、fanout共享版本和必要materialization；这些layout mechanics由06与region merge/split、tile、
  residency和communication共同物化到actual clone，不自行拥有partition policy。每个方案立即重跑现有exact gates；
  任何layout改写都使旧root/lifetime/coexistence与SPM allocation结论失效。
- Output artifact / IR:
  每个implementation recipe至多四个自包含layout optimized actual clones及独立baseline，并继续受现有rank/whole
  frontier hard cap约束。PBQP projection、assignment、relation proof和cost hint在clone进入现有frontier前销毁；
  最终只有现有candidate owner提交的typed winner IR进入bundle。
- Downstream consumer:
  complete-rank Tile/Dataflow到Instr conversion、从final actual IR的roots/lifetime/control-flow coexistence派生的
  SPM fixed allocation problems、whole-variant DDR planning、complete-rank completion/resource/transport gate、target publication、
  package、SystemC和板端执行。
- User-level driver / named pipeline:
  现有wafer-compile source-to-bundle production pipeline；wafer-opt仅作局部replay/test。
- Explicit non-goals:
  不新增layout dialect、VirtualTensor、Linear Layout、shadow graph/plan、layout demand或physical-access接口、
  target identity查询参数、conversion matrix、solver result attr、外部ILP/CP-SAT或第二个decision owner；
  不支持无法由当前IR精确证明的dynamic-shape关系；多root concat继续由现有exact concat-piece与显式
  `insert_slice` compound movement表达，不把它伪装成单root alias，也不在本任务把必要concat copy算作冗余layout movement。
- Completion gate:
  targeted source的非连续view与重复layout movement在selected IR中真实消失；baseline始终保留；
  host、完整package/fresh no-card和FP16/BF16真实板端正确性、guard及matched性能门禁通过。
```

## 已收敛设计

### 1. 复用现有事实源

- `IndexRelation`只表示logical index关系，继续支持composition、identity、domain/image、functional、
  injective/bijective及piecewise证明。
- `WaferPhysicalEncodingAttrInterface`继续独占footprint、alignment、valid/padding、mapping和segment事实；
  查询签名保持不变，不接收target identity。
- `MemLayout`是compiler IR、Kernel ABI、numeric model与qualification共享的唯一layout family枚举；
  `MemoryAttr`只组合`MemorySpace + MemLayout`并派生geometry，不在model、codec或candidate projection中复制
  一套同值layout marker。
- `PhysicalAccessRelation`、`TransferRealizability`和`MovementSupport`继续分别拥有physical组合、route/view
  可实现性和descriptor构造；typed op verifier继续拥有算子合法性。
- `InvalidLaneState`保持由当前IR派生的`NoInvalidLanes`、`KnownSplat(value)`、`Unknown`，不进入attr或side table。

### 2. Relation-guided view motion

- 在同一block的direct def-use上运行有界固定点，按现有`IndexRelation`组合`reshape`、`transpose`、
  `extract_slice`、`broadcast`和`copy`的exact unary relation；只用exact `isEquivalentTo`判断组合后是否为identity，
  不保存relation side state，也不按文本或op名恢复tensor角色。
- 全scope最多处理`8 * (value count + use count)`个候选；每次改写都重新从current IR构造relation。checked overflow、
  relation budget耗尽、dynamic shape、不同block或不能精确组合时停止该路径并保留baseline。
- pointwise clone仍位于原compute位置，不交换effectful op。pre-view source到compute之间如存在可能写该source的effect，
  该pre-view形成barrier；其它root上的exact unary movement可以留在diamond中，不会因粗粒度全局resource effect误挡。
- relation组合为identity时删除真实pre/post movement并以typed pointwise clone承接原destination shape；post relation自身
  为identity时可以只删除安全的post copy而保留有mutation barrier的pre snapshot。clone verifier失败时整次proposal丢弃。
- 多root concat继续由`staticConcatPiece`和现有`MoveInsertSliceOp`序列显式表达互斥piece写入；它不是单root unary alias，
  因此不进入本固定点。unknown effect/alias、无法对齐的fanin和control-flow边界均fail closed。

### 3. Layout、implementation与fanout联合分配

- 不扩展`TargetImplementationCandidate`，也不新增layout-demand/access-constraint接口。existing source interface仍只枚举
  implementation kind；scheduler以稳定recipe先物化该kind的isolated complete-rank actual clone，再从该clone重建layout PBQP；
  Q49终态由06联合选择region partition、tile/loop、layout/version、residency、materialization和communication；
  本计划原有PBQP只作为proposal reducer，不独立决定region partition，也不把不同traversal/tile shape机械变成region。
  default implementation和每个non-default implementation分别交叉至多四个layout proposal，仍受12个semantic recipe、
  96次optimized rank evaluation及whole-variant hard cap约束。
- PBQP node直接保存可由typed verifier与`provePhysicalTraversal`接受的`MemLayout`值，不再包装第二个layout enum、
  `LayoutState`、port tuple或value-state对象。当前target domain最多为`Tensor/NTensor/Cx/NCx`四态；op/result shape、dtype、
  memory space和implementation语义始终从actual IR读取。
- producer/consumer SSA相邻node之间建立pair cost；相同或compatible physical traversal为zero，否则只在现有
  materialization可表达时记physical bytes、command lower bound和extra footprint。public ABI及图外consumer的当前typed
  layout形成unary requirement，不另存layout demand。
- fanout的secondary在producer后只创建一次，所有兼容consumer复用同一真实SSA version；不按consumer edge
  重复物化，也不枚举`2^fanout`子集。materializer的每个root最多保留当前primary加一个secondary版本；已有
  share-vs-recompute clone在本算法外形成，本算法分别求解。
- 兼容physical traversal或metadata view的边成本为零；真实转换复用现有movement bytes和可得的command
  lower bound；typed verifier明确拒绝才为不可选。资源、lifetime、completion和最终性能只由actual clone gate判断。
- proposal objective依次使用可证明movement bytes、unknown-command count、known command lower bound、额外physical
  footprint和稳定ordinal；
  最终winner仍只由fresh final `InstructionProgramCost`、Pareto保留和当前校准hardware cost model决定。
- 不触及collective的候选保持rank-local。连接AllReduce的recipe使用现有stable recipe/generation identity进入
  whole-variant coordinator；各rank必须选择同一recipe ordinal，Direct-DTE transport activation再要求reduction
  send/recv具有完全相同的physical `MemRefType`。任一rank materialization、typed verifier、whole-rank resource或
  cross-rank physical payload检查失败，整个tuple丢弃，不保存一份跨ranklayout side table。

### 4. Production PBQP

这里选择PBQP是因为actual implementation已经物化，剩余layout choice是每个typed op的至多四个`MemLayout` label，
耦合可以分解为图外boundary unary cost和producer/consumer pair cost；低度图可精确消元，高度核心再受统一预算约束。
PBQP只负责有界提案，不复制implementation、value、shape或legality；最终合法性与选择仍由actual clone和现有owner承担。

- 生产只实现确定性PBQP：dominance pruning、degree-0/1/2精确消元、高阶核心的
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
  `ProposalCost`加法或比较都checked消费一个work unit。整个graph与Top-4子问题共享该counter；耗尽后停止产生
  新proposal，只保留已经完整求得的unique assignments，任何未完成assignment丢弃，baseline不受影响。每个子问题都在
  disposable graph copy求解，失败不会留下半张matrix；Top-4 heap的push/pop也先在disposable copy完成固定sift，budget
  足够时才commit，因此比较次数和失败边界不依赖library实现。
- 高阶核心按active degree降序、stable SSA ordinal打破平局；state按stable layout ordinal展开。
  invocation-wide `searchExpansionBudget`初值为4096；每个branch child或完整one/two-variable local neighbor计一次，且其
  cost计算仍消费`solverWorkBudget`。search或solver work不足时只保留已经完整求得的incumbent，未完成proposal丢弃，
  baseline不受影响。
- 单次受约束graph solve只返回一个complete assignment。Top-4使用带继承约束域的deterministic prefix-partition：heap node保存
  该子问题的完整forced/forbidden state constraints及其solution；pop `(constraints, vector)`后，对每个stable variable position
  建立“继承constraints、此前vector prefix固定、当前位置禁止vector当前state”的子问题。各子域按first-differing position
  互斥并覆盖parent domain除当前vector外的全部assignment；求解后按proposal objective放入按canonical constraint set和完整
  state vector去重的min-heap。重复pop/partition直到四个unique vectors或共享solver/search budget耗尽。该方法只有在所有
  已需子问题都exact闭合且预算未耗尽时称为k-best；有界核心或预算触发时只称ordered proposals，不宣称最优。
- graph assignment按`(movement bytes, unknown-command count, command lower bound, extra footprint,
  complete state-ordinal vector)`排序；identity就是stable variable顺序下的完整state-ordinal vector，不使用模糊的
  “layout-diverse”判定。
- disconnected node/component仍在同一PBQP graph中由degree-0/1/2消元精确处理，不额外复制component graph、assignment
  或第二层k-best heap。
- 每个implementation recipe最多四个optimized assignments各尝试物化一次；某个assignment失败不回填第五名。另保留独立baseline，并继续受
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
- public input/output ABI继续保持compact Tensor。full-buffer resident handoff只有在pre-Instr owner已把
  producer/consumer和兼容tile schedule物化进同一selected SPM residency domain内的可连接traversal时，才可在descriptor展开前转交
  producer已有SPM version；不兼容fanout保留显式DDR materialization，region内selective spill保持显式。late transfer
  elimination不独立创建region；region merge/split由06基于actual cost共同选择，跨界data必须是DDR，不能发明SPM alias。

## 实施 Checkpoints

1. `[closed]`补齐bitpacked Cx/NCx physical mapping与transfer/codec/numeric/model tests，保持encoding interface签名不变。
2. `[closed]`在hardware-supported row上放宽compute/instr concrete verifier与target preflight的固定Tensor限制；
   dtype-changing traversal要求source覆盖完整destination physical traversal且valid element ordinal一致。
3. `[closed]`实现exact unary relation固定点、chain/diamond/broadcast/rank-reduced slice、fanin/fanout和source-mutation
   barrier；多root concat继续保留必要的显式compound movement，不误报zero-copy。
4. `[closed]`实现deterministic PBQP/Top-4、transactional actual-clone materialization、失败无残留，并接入现有
   candidate owner；implementation/layout通过actual recipe交叉，不复制一套port/value state。
5. `[closed]`接入pointwise/convert、composite reduce、Tree AllReduce full physical footprint、Ring padded-layout拒绝、
   fanout共享secondary及whole-rank exact physical payload gate。
6. `[board-ready]`同源FP16 `GEMM -> square -> GEMM` baseline/winner已生成完整package并通过fresh no-card；
   winner保持非layout target call inventory不变，将final `gather_scatter`从7降到4。真实板端matched验证尚未执行。

## Verification Contract

- relation：非连续inverse view、reshape/transpose/slice/broadcast、concat piece显式movement保留、source mutation与
  control-flow barrier；
- physical：Tensor/NTensor/Cx/NCx full/tail、bitpacked relation/select、unknown padding、neutral reduce、跨dtype block mismatch；
- solver：typed layout legality不扩接口、fixed-point/solver-work/search-expansion hard cap、deterministic Top-4、共享secondary
  只物化一次、双版本cap、失败不回填、baseline保留、无solver residue，以及implementation/layout actual recipe联合候选；
- vertical：GEMM到pointwise/convert/reduce链、Tree AllReduce full footprint、exact Ring chunk正负例，以及同一
  residency region内不同traversal/tile shape之间的resident handoff与selective DDR materialization；同一source还覆盖
  single-region resident与multi-region explicit materialization actual candidates。stage/layout/movement边界不自动切region或
  生成completion；selected region boundary的data只走DDR且无SPM memref/root/alias，且只完成仍访问被释放roots的pending work；
  SPM planner从final actual IR派生一个或多个fixed allocation problems，problem/query数量只作work diagnostic；
- production：同源baseline/winner均经source-to-package、fresh no-card、output/guard验证；板端使用FP16/BF16，
  targeted winner必须减少final IR movement bytes/commands且matched性能不劣于baseline。

只完成文档、接口声明、局部FileCheck、proposal生成或单个case的IR op数下降均不算实现完成。

当前无卡证据：host unit与compiler integration gate通过，lit为209/209；`layout-movement-chain`两包通过schema-v7
package/no-card，baseline与winner的GEMM、multiply、RDMA、WDMA和join call count完全一致，winner只减少3个
`wafer_tx81_gather_scatter`。当前CMake均为`WAFER_ENABLE_BOARD_TEST_EXECUTION=OFF`，因此没有硬件correctness、guard或
matched性能结论，任务状态只能是`board-ready`而不是`done`。
