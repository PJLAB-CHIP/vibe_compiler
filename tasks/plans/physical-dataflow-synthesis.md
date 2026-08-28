# Physical Dataflow Search 当前实施计划

状态：Q52 `search-scalability`正在执行，Q53 `production-host-readiness`等待Q52。稳定设计只看
`tasks/06-physical-dataflow-synthesis.md`及其直接下游编号文档；动态状态只看`tasks/progress.md`。
Q49–Q51和Q52已完成checkpoint的详细证据在`tasks/archive/physical-dataflow-synthesis-working-history.md`，
不再复制到current plan。

## Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  current产品入口经05号attention semantic recognition产生并验证的card-local TensorProgram；同一只读source artifact分别进入
  none和search独立事务。Structured logical e-graph normalization尚未执行时由本计划的直接前置work item闭合。
- Current stage responsibility:
  保留已完成的verifier清理、baseline direct materializer和search execution construction donor；删除在actual IR之前发明
  physical value、movement、storage、event、execution structure和schedule的shadow search chain。Search只保存显式choice，
  先在policy分叉前对post-attention pure Tensor/Linalg component完成bounded access-relation e-graph normalization；随后在结构choice后
  立即物化actual TileRegion IR，之后compact temporal tile-and-fuse保持attention opaque；独立selected-attention lowering再把它
  一次性改写为actual Linalg/Tensor/SCF。Layout/bufferization、movement、execution structure、Instr/order/completion和memory均读
  各自current IR。
- Output IR / files:
  none和search各自产生policy-complete、verifier-valid Instr IR，随后经同一actual SPM/DDR/transport/target leaf
  形成CardExecutable与ExecutablePackage。
- Downstream consumer:
  target conversion、device link、package emission和runtime launch。
- User-level driver / named pipeline:
  wafer-compile的typed `none`与`search`入口；focused测试使用注册named pipeline或同一compiler API。
- Explicit non-goals:
  不新建future-output IR、shadow candidate schema、expected-inventory builder、plan/actual parity verifier或兼容双路径；
  不以footprint/shape/推算值决定SPM legality；不猜join/wait；不修改数值语义；本计划不运行真实设备。
- Completion criteria:
  下表的20个checkpoint按线性顺序通过；current production不再有future SSA/buffer/event/schedule的跨stage owner；
  同一current FP16 LLaMA block的none和search分别在15分钟Release门限内生成package并通过strict readback/no-card，
  且两者实际进入Instr、MiniMalloc、DDR和target。
```

## 不可变边界

### Current IR是唯一事实源

```text
current IR
  -> typed choice
  -> candidate-owned transformation
  -> verifier
  -> fresh analysis
  -> next direct consumer
```

- Search key只保存spatial/region/temporal及当前transformation尚未消费的显式choice。
- Operation、SSA、buffer、alias、movement、lifetime、event、order和completion必须先存在于candidate-owned IR。
- Analysis只从current IR和显式target configuration重算；IR mutation后默认失效。
- Cost estimate只能排序choice，不代签actual op/buffer/Instr inventory、legality或memory结论。
- Failed candidate owner整体销毁；Accepted owner不rematerialize、不重建offset。

### Baseline与search是两套实现

| 边界 | Baseline / `none` | Search / `search` |
| --- | --- | --- |
| controller | `compileCardBaseline` | `runUnifiedSearch` |
| construction input | current TensorProgram + fixed baseline rules；无search choice/domain/state | current TensorProgram + bounded spatial/region/temporal choice frontier |
| actual IR owner | baseline direct materializer | search structural materializer |
| downstream | 只消费baseline current IR | 只消费当前search candidate IR |
| feedback | 仅actual SPM capacity rejection生成smaller temporal choice | typed actual result返回frontier/controller |
| failure | 不启动search | 不fallback baseline |

两条policy可共享source analysis、IndexRelation、single-op rewrite/conversion和actual memory/target leaf，但不共享
controller、Card/Tile materializer、candidate owner、fallback或accepted result。

### SPM与completion

- SPM legality只由current Instr的actual allocation、alias/effect/completion/lifetime和唯一MiniMalloc决定。
- BodyEmitter只报告本次actual buffer relation，不推断owner或持有跨scope relation。
- Completion只从current Instr的effect/token/control-flow/lifetime和已证hardware/ABI事实fresh构造。
- Unknown保持typed unknown；不插入固定worker join、结构化join、全worker drain或issue后立即await。

### 测试规模

Production-style positive默认rank至少为3、主要迭代维至少1024；partition、tiling、loop和memory链成对覆盖
1024及1025/1031，实际经过多Tile、多block/wave、remainder和tail。Tiny只用于独立oracle、最小负例或
scalar/zero-rank，并必须有真实规模对应项。

## 已完成输出的当前用途

Q52前4项的详细证据已归档，current实施只消费下列输出：

| 已完成项 | 保留输出 | 不代签 |
| --- | --- | --- |
| expansion evidence | stage inventory、双materializer和IR膨胀根因证据 | 新materialization boundary正确 |
| verifier cleanup | local op verifier、container/stage check和verify-each边界 | 任何shadow plan/plan replay的正确性或其代签current IR的合法性 |
| search execution materializer | typed single-op builders、FA/FD算术、Card/Tile assembly与覆盖oracle；第11项迁出有效资产后删除旧wrapper | 旧group/materializer API、future value/movement/storage/schedule或继续可用的search route |
| baseline root materializer | fixed spatial/root isolation、actual feedback分类与覆盖oracle；第11项迁出有效资产后删除`CanonicalBaselinePlan`和旧controller | 继续可用的none route、search路径或Q52最终验收 |

## Q52线性实施顺序

任一项失败留在本项修复，不跳过、不fallback旧路径或另一policy。每项都重复完整工作流程。

下表按**施工依赖**排列。第6--10项已经闭合actual leaf、Instr completion、execution structure、movement和logical graph
normalization；它们继续作为后续stage的已验证输入/下游，
不在当前队列重复施工。第11项先删除错误的旧none/search production入口和future/shadow事实owner；在新链完整前两种policy均明确
不可用且互不fallback，compiler仍保持可构建。随后按artifact producer/consumer顺序闭合新链：先物化Spatial/Region actual IR，再由
compact temporal tile-and-fuse形成ordinary current loop/use graph并保持attention semantic op opaque；selected-attention lowering生成其
actual Linalg/Tensor/SCF，至此才得到最终current SSA/use graph并建立完整layout domain、PBQP assignment、exact view和bufferization。
此前已经完成的output DPS、copy closure、PBQP solver和duplicate/unused materialization cleanup保留为
`current-ir-layout-bufferization`的已有实现资产，但不能代签该work item整体完成。

Baseline gate只在上述新actual-IR producers闭合后运行；随后`search-current-ir-integration`只注册新链，不再承担旧代码cutover或
retirement。最终search运行顺序固定为：

```text
post-attention Tensor/Linalg
  -> structured-logical-egraph-normalization
  -> spatial-region-current-ir-materialization
  -> compact-temporal-tile-and-fuse
  -> selected-attention-lowering
  -> current-ir-layout-bufferization
  -> current-ir-movement-boundary-closure
  -> current-tile-execution-structure
  -> current-instr-schedule-completion
  -> actual-leaf-current-ir-contract
```

第11项之后不注册任何临时policy pipeline，也不把未被真实上游和直接consumer共同读取的新stage标成完成。`none`在第17项、`search`在
第18项接通前分别只返回明确typed unavailable；不得调用旧materializer、cross-policy fallback或发布package。Baseline和search仍是
两条policy-owned路径。Logical e-graph normalization在policy分叉前只运行一次；compact tile-and-fuse只替换shared
ordinary transformation mechanics且不展开attention；selected-attention lowering是05号语义的唯一candidate-owned lowering，不增加第三个
materializer或policy路径。第18项只把这些已经唯一存在的atomic stages接入search controller，不恢复旧complete/shadow路径。

### Current upstream/downstream职责迁移表

| 现有行为/代码 | 最终semantic owner | 迁移动作与时点 | 直接验收 |
| --- | --- | --- | --- |
| old `compileCardBaseline`与`runUnifiedSearch`→complete/shadow materializer production routes | 无长期owner | 第11项移除两种policy product caller、旧wrapper、future fact type/CMake/test；保留的纯choice算法、exact relation和oracle移到直接owner。新链完成前none/search均typed unavailable、互不fallback且不发布 | active source/CMake/current-doc中旧policy materializer caller和future fact owner为0；真实rank≥3、1024输入调用两种policy时analysis/candidate/leaf/package次数均为0；build与direct stage APIs不依赖旧routes |
| post-Linalg reshape/transpose/broadcast/concat及pure structured compute graph | `structured-logical-egraph-normalization`，设计owner 05 | attention semantic recognition后对剩余ordinary pure component执行一次bounded access-relation e-graph；C++只导入current原始节点并提供request-local relation service，egg dynamic rules创建新e-node；attention保持opaque，candidate和后续stage不再调用e-graph | planning输入无可严格支配的transform graph；至少一个真实phase-order case由两条以上rules连续闭合；on/off exact语义与downstream representability一致；budgeted unchanged可复核 |
| memory入口内的function-boundary bufferization | `current-ir-layout-bufferization` | actual leaf只让缺失boundary typed失败；layout stage在movement前接通唯一production transformation | layout→movement→execution-structure→Instr→leaf链中bufferization恰一次；leaf调用次数为0 |
| memory入口内的NCC join rebuild | 第7项`current-instr-schedule-completion` | 第6项只从leaf删除repair；第7项在final worker/order后从current Instr fresh构造 | 删除或扰动completion时第6项typed失败；第7项输出直接通过第6项且leaf不新增join/wait |
| bufferization/layout产生的普通allocation及其IR位置 | `current-ir-layout-bufferization` | layout stage在创建transaction中确定dominance、alias和effect；不把placement另存成plan | actual allocation/view/copy all-and-only，1024/1025/1031 lifetime与MiniMalloc witness一致 |
| software-pipeline rotating allocation/slot SSA | 第8项`current-tile-execution-structure` | 第8项随prefix/steady/tail一次物化；第6项只读 | 1031 tail与multi-wave检查root、slot、reuse obligation和actual offsets |
| `sinkStaticSPMAllocationsToFirstUse` | 无长期owner | 第6项先做caller/test inventory；若fresh production不依赖则删除helper和专用fixture期望；若依赖则停止并把过早创建定位到第8或10项直接producer，禁止把helper移入allocator或新增通用pass | leaf call graph中helper为0；相同current IR的合法性只由实际MiniMalloc决定；任一真实依赖必须有producer修复后才能签发对应owner |
| `buildPrepareInstrForMemoryPlanningPipeline`组合wrapper | 无长期owner | 第6项leaf停止调用后做zero-caller检查并删除wrapper；其bufferization/completion kernel分别留给第10/7项 | source/CMake/test/current-doc中组合wrapper caller为0，单项owner测试仍保留 |
| actual SPM/DDR high-water/headroom | 第6项`actual-leaf-current-ir-contract` | 只从accepted MiniMalloc/DDR offsets重算并作为diagnostic/resource witness | 与accepted offsets/range精确一致；不参与SPM admission、retile或winner猜测 |
| function-boundary bufferization的冗余DDR→DDR publication copy与Instr copy-only TileRegion fallback | `current-ir-layout-bufferization`的output DPS正确性 | output在bufferization前绑定唯一actual destination；只消除因DPS缺失而复制同一logical result的copy；必要copy保留exact SSA/alias/effect witness并交给typed movement；删除Instr临时建Region/staging的fallback | 冗余publication copy为0；必要copy all-and-only；movement closure后未分类`memref.copy`为0；每个observable result唯一publication；Instr创建copy-only TileRegion次数为0 |
| 手写`first / steady / tail`递归loop、`LocalUseDelivery`和无current-IR控制的producer fusion worklist/cache | `compact-temporal-tile-and-fuse` | 第11项先删除旧delivery/placement/nested-temporal owner；第13项再用pinned SCF tiling/fusion生成canonical loop nest，只读exact tile-relation query仅消除唯一派生参数，attention保持opaque | 第11项后active旧type/caller为0；第13项loop内/外producer只由actual loop/SSA表达，自由search域不被query剪枝 |
| attention future action/value/materialization inventory及其nested invocation replay | `selected-attention-lowering`，设计owner 05 | 第11项删除future inventory与旧replay；有效FA/FD算术、coupled semantic relation和oracle作为源码/测试资产保留。第14项直接从current attention op与显式choice生成actual Linalg/Tensor/SCF | 第11项后future action/value/physical-version/materialization owner为0；第14项每个attention occurrence只lower一次且layout入口attention为0 |

该表分配的是最终职责，不授权临时复制实现。Actual leaf移除隐藏行为后若baseline暴露上游输入缺口，失败保持在产生该输入的
completion、execution-structure或layout owner；不得把行为放回leaf来维持旧测试。

### 本轮新增整改的施工顺序

1. `structured-logical-egraph-normalization`消费attention semantic recognition之后的ordinary pure Tensor/Linalg component并且全局只运行
   一次。C++ importer只导入current原始`Input/Access/Concat/Compute`节点；request-local `IndexRelation`服务为egg dynamic Applier
   提供relation composition、operand-map composition、concat axis factoring和受限parallel-result reindex，禁止预构造最终candidate或
   candidate recipe。首批闭合Access identity/composition、generic compute data-operand absorption、Concat single/flatten/common-access
   extraction及bijective parallel-result reindex；attention/collective/SCF/call/effect保持barrier。Budget exhaustion保持原verified graph，
   不产生legality或candidate feedback；attention展开及后续stage不调用e-graph。
2. `legacy-shadow-materializer-retirement`先删除错误的旧none/search product routes、complete-candidate/baseline wrappers以及future
   representation/movement/storage/execution/schedule、delivery/placement/nested-temporal和attention action/value/materialization owner。
   纯spatial/connected-partition/free-temporal/PBQP/relation算法与FA/FD算术/oracle只有迁到直接owner后才保留。删除后compiler保持可构建，
   `none`/`search`均返回typed unavailable，actual candidate、cross-policy fallback和package publication为0；不添加compatibility wrapper。
3. `spatial-region-current-ir-materialization`消费selected SpatialPlan/Assignment、ExactDemand/RootWork及connected membership/explicit replica，
   创建candidate-owned CardModule、TileModules和non-nested structural TileRegions。Ordinary spatial pieces立即成为actual operation/SSA；
   attention保持opaque，free temporal与reduction/attention contribution choice尚未消费。无layout、movement、buffer或completion。
4. `compact-temporal-tile-and-fuse`在第3项actual Region上决定fusion。Temporal domain保留全部自由tile/loop-order choice；一次调用内的
   exact tile-relation query只有在total single-valued proof下才消除派生参数，unsupported/indeterminate保持独立producer与Region candidate。
   使用pinned MLIR SCF tiling/fusion替换parallel/reduction手写三段递归builder；fusion control直接检查current producer/use、
   indexing relation、use count、effect和region boundary。
   Attention保持semantic op，只允许接口支持的output/parallel tiling及外部exact edge。先生成canonical bounded loop，再按内到外peel
   ragged remainder、promote单次tail并运行bounded canonicalization/CSE/DCE；删除front peel、按wave复制body和无delivery通用fusion
   worklist/cache。Candidate owner由controller至多建立一次，本stage不再clone Card/Tile owner。
5. `selected-attention-lowering`在第4项输出上保持独立stage。它直接读取current attention op、固定FA/FD以及尚未消费的K1/K2、FD
   contribution/merge choice，一次性rewrite成actual QK、scale/mask、Maximum/Sum/Accumulator、PV、combine/finalize、tensor slice与
   canonical SCF state；不调用generic tile-and-fuse，不构造future action/value/materialization ID，不clone整个candidate owner。
   失败按typed candidate failure返回，不换algorithm或builder；成功后attention op为0，输出直接进入layout。它只复用第11项保留的
   有效算术/relation/oracle，不恢复已删除replay。
6. `current-ir-layout-bufferization`消费最终tile/fuse与selected-attention current SSA，补齐value/use layout domain、op tuple constraints、PBQP
   assignment/apply及IndexRelation+PhysicalLayoutRelation exact view；保留已经完成的output DPS、necessary copy closure、PBQP solver和
   duplicate/unused cleanup。只有不兼容edge创建layout materialization，随后bufferization恰一次。
7. `current-ir-downstream-orchestration`显式重接movement/boundary、execution structure、TileRegion→Instr、worker/order/completion和actual leaf，
   不恢复旧complete facade或跨stage plans。
8. `baseline-current-ir-integration`与`search-current-ir-integration`分别重启none/search独立controller；最后依次完成scale inventory和
   同源LLaMA验收。

上述graph/spatial/tile/attention/layout项不改变SPM admission：每个candidate仍先形成上述actual IR，再由唯一MiniMalloc返回accepted offset或typed actual
capacity rejection。减少DDR交互是current movement transformation和actual cost的目标，不得用DDR/SPM bytes估算代替合法性。

| 顺序 | Work item | 状态 | 单一责任与输出 | 精确完成条件 | 本项固定执行流程 |
| ---: | --- | --- | --- | --- | --- |
| 1 | `expansion-evidence-and-contract` | `done` | 冻结同源两条policy的stage inventory、双路径和膨胀根因 | instrumentation on/off结果一致，违规caller和删除边界可静态复核 | 读AGENTS/progress→读06/Q52与本项矩阵→调研stage instrumentation→查官方及pinned MLIR→改诊断/tests→fresh验证→重读设计并按MLIR复审→更新并提交 |
| 2 | `verifier-contract-cleanup` | `done` | 将operation、container和materialized-stage检查收回正确owner | local verifier不跨scope，必要负例仍在直接stage失败，无总体parity verifier | 读AGENTS/progress→读19/Q52与本项矩阵→调研LLVM/MLIR verifier实践→查pinned源码→改代码/tests→fresh verify-each/build→重读设计/MLIR复审→更新并提交 |
| 3 | `search-execution-materializer` | `done` | search generic/attention使用一个execution-owned construction donor | canonical/noncanonical不选另一builder，producer compute不随fanout重复 | 读AGENTS/progress→读06/07/10与本项矩阵→调研MLIR/IREE/Triton materialization→查pinned API→改代码/tests→fresh Card/Instr验证→重读设计/MLIR复审→更新并提交 |
| 4 | `baseline-root-materializer` | `done` | baseline独立实root构造region并运行actual feedback | 每compute region恰一root，baseline call graph不进入search owner | 读AGENTS/progress→读06/07/10与本项矩阵→读DDR/completion事实→调研baseline materialization→查pinned API→改代码/tests→fresh Card/Instr/SPM验证→重读设计/MLIR复审→更新并提交 |
| 5 | `current-ir-contract-reset` | `done` | 重写active设计和施工顺序，冻结shadow owner删除清单 | AGENTS、01、05–19、Q52 plan/progress和memory只保留一套actual-IR合同；archive只作历史 | 读AGENTS/progress→读01/05–19/Q52与本项矩阵→调研VPlan/GlobalISel/Transform/Bufferization取舍→查pinned文档/源码→改current docs→文本检查→重读设计/MLIR复审→更新并提交 |
| 6 | `actual-leaf-current-ir-contract` | `done` | 从现有Card→Executable入口抽出completion-closed canonical Instr→actual SPM/DDR/transport/target的唯一typed leaf；baseline原入口立即调用该leaf；执行上表leaf removal、legacy wrapper和generic allocation-sink inventory，不重新实现上游能力、allocator或target | leaf不运行function-boundary bufferization、join/wait rebuild、allocation sinking或其它lifetime/completion mutation，不读取policy/preparation/shadow plan；组合wrapper zero-caller后删除；MiniMalloc实际运行；typed failure和Accepted owner保持；fresh baseline source实际进入该leaf；若真实case依赖sink，按上表停止并归因到execution-structure或layout直接producer而不在leaf repair | 读AGENTS/progress→读09/11–14及本项矩阵→读memory/resource硬件事实→调研actual allocation/liveness与typed allocator feedback→查官方及pinned MLIR→拆分唯一stage boundary/tests→fresh baseline Instr/SPM/DDR/target验证→重读设计/MLIR复审→更新并提交 |
| 7 | `current-instr-schedule-completion` | `done` | 唯一拥有TileRegion-to-Instr、current dependence、worker/order和fresh completion；消费第8项给出的function-boundary-bufferized、execution-structure-closed physical TileRegion，输出直接进入第6项 | 本stage及baseline调用链不运行bufferization，不消费EventId→op反查或ClosedSchedulePlan；EventGraph不跨mutation；relation在conversion后仍属于current IR；join/wait有actual effect/token/lifetime和hardware witness；第6项不再修改completion；fresh baseline source经第7→6项成功 | 读AGENTS/progress→读06/09–11/13及本项矩阵→读NCC/DTE硬件/ABI事实→调研LLVM MachineScheduler/MLIR async→查pinned API→改唯一baseline stage/tests→fresh baseline TileRegion→Instr→第6项验证→重读设计/硬件/MLIR复审→更新并提交 |
| 8 | `current-tile-execution-structure` | `done` | 唯一拥有software-pipeline execution structure及其rotating allocation roots/slot SSA；在movement-closed physical TileRegion上立即物化serialized或selected prefix/steady/tail与chunk control，输出直接进入第7项 | 本stage不消费ExecutionStructurePlan/BufferPlan跨stage事实，不创建completion或offset；Serialized不改IR；pipelined case的loop、allocation root、slot、reuse obligation和movement/compute occurrence all-and-only；fresh actual source经第8→7→6项成功 | 读AGENTS/progress→读06–11及本项矩阵→读worker/completion/memory硬件事实→调研MLIR/LLVM software pipelining与rotating-buffer实现→查pinned SCF/Rewriter API→改current-IR transform/tests→fresh actual source pipeline→第7→6项验证→重读设计/硬件/MLIR复审→更新并提交 |
| 9 | `current-ir-movement-boundary-closure` | `done` | 先冻结layout-resolved endpoint→movement-closed physical TileRegion合同，再在baseline唯一路径中把movement从materializer内部选择/修补拆成current producer/use/endpoint/relation transformation，闭合staging、boundary和effect；最后运行唯一current-IR exact full-transfer cleanup | 本stage及baseline调用链不消费future version或donor scan；compatible local edge零DDR；movement不创建compute或重选layout；只删除full/same-map/alias-effect-lifetime安全的transfer，partial/permuted/layout-changing保留；fresh baseline source经第9→8→7→6项成功 | 读AGENTS/progress→读06–10/13及本项矩阵→读transport硬件事实→调研MLIR data movement/fanout和copy coalescing→查pinned API→迁移现有eliminator proof/tests并接唯一baseline caller→fresh baseline movement/cleanup→第8→7→6项验证→重读设计/硬件/MLIR复审→更新并提交 |
| 10 | `structured-logical-egraph-normalization` | `done` | 05号normalization唯一拥有attention recognition之后ordinary pure Tensor/Linalg graph的一次性bounded access-relation e-graph pass；通过hermetic pinned `egg` Rust staticlib和typed C ABI实现core。C++只导入current原始`Input/Access/Concat/Compute`节点、提供同步request-local `IndexRelation` relation service并物化唯一extraction；egg固定dynamic rules在rebuild后连续创建RHS e-node。实现Access identity/composition、generic Compute data-operand absorption、Concat single/flatten/common-access extraction、bijective parallel-result reindex、异构all-users fanout propagation和同一pass内严格结构下降定点；attention不创建e-node且保持opaque，candidate及后续stage不调用e-graph | 不自研/fallback第二个e-graph，不保留`build...Alternatives`、imported-equivalence-only union或candidate recipe；C ABI callback/status/allocator/panic和Cargo/CMake offline合同闭合；不修改scalar/combiner/dtype、computeId multiset或downstream支持范围；不越attention/collective/SCF/call/effect barrier；deterministic relation/e-node/match/rebuild/extraction budget耗尽保持原IR；输出由现有StructuredDAG/exact-demand直接消费；reshape/broadcast/concat与elementwise/reduction/contraction连续链及异构fanout有1024/1025/1031 focused exact断言；直接rewrite current IR且不clone Module/Func/DAG；第二次运行byte-equivalent | 读AGENTS/progress→读05/06/19及本项矩阵→调研`egg` dynamic rewrite、TENSAT、Glenside和MLIR map/reshape transforms→逐项确认pinned MLIR已有能力与直接下游accepted maps→先更新设计并删除scoped/candidate旧合同→重做typed language/e-class facts/relation-service C ABI/dynamic rules/extractor/materializer→fresh 1024/1025/1031 graph、真实PyTorch/HF component和StructuredDAG/exact-demand witness→重读设计/完整diff/MLIR复审→更新并提交 |
| 11 | `legacy-shadow-materializer-retirement` | `pending` | 先移除`none`/`search`旧CardExecutable product route、complete-candidate/materializer/preparation facade和所有代表future representation/movement/storage/execution/schedule、delivery/nested-temporal及attention action/value的production owner；保留并迁出Spatial/ExactDemand/Region partition/Temporal free-domain/PBQP/FA-FD算术、current transforms和actual leaf等有直接新owner的算法资产 | repo保持build/link；`none`和`search`在CardExecutable边界明确`operation_not_supported`且actual analysis/candidate/baseline fallback/MiniMalloc/package次数均为0；`CompleteCandidatePlan`、`CanonicalBaselinePlan`、post-temporal PlanningState、`LocalUseDelivery`、future attention inventory、旧group/materializer API、`compileCardModuleToExecutable`隐藏facade及only-purpose CMake/test/current-doc全仓为0；direct normalization、retained算法oracle和`compileCanonicalInstructionTilesToExecutable`仍通过 | 读AGENTS/progress→读01/05–19/Q52删除清单与本项矩阵→逐definition/caller/CMake/test分类delete/migrate/retain→调研成熟compiler破坏性pipeline cutover与typed unavailable边界→查pinned LLVM/MLIR ownership/API→先迁有效算法/oracle再同批删除旧route/schema/facade→fresh build、none/search unavailable、零fallback/actualization/package、direct retained-assets与actual-leaf验证→重读设计/完整diff/MLIR复审→更新并提交 |
| 12 | `spatial-region-current-ir-materialization` | `pending` | 消费current TensorProgram、selected `SpatialPlan`/closed `SpatialAssignment`、ExactDemand/RootWork和只含connected membership/explicit replica的Region choice，创建candidate-owned CardModule、all-and-only TileModules及non-nested structural TileRegions；ordinary spatial pieces立即成为actual ops/SSA，attention保持opaque，尚未消费的reduction/attention contribution choice继续作为显式参数交给直接consumer | partition intervals、Tile embedding、reduction group/merge owner和Region membership all-and-only；1024/1025/1031、rank 3–6、多Tile、balanced/uniform、chain/diamond/fanout/reduction及FA/FD spatial constraints覆盖；无temporal loop、layout、movement、buffer、completion或future execution ID；output verifier-valid且直接被第13项消费；Spatial proposal on/off不改raw domain | 读AGENTS/progress→读04–07/10/19及本项矩阵→调研MLIR structured partition/materialization与成熟compiler spatial mapping→查pinned Tiling/DPS/IRMapping API→保留SpatialDomain/ExactDemand/connected-partition算法并实现唯一current-IR structural transform→fresh 1024/1025/1031 exact coverage、Tile/Region/SSA owner、attention opaque与第13项direct-input witness→重读设计/完整diff/MLIR复审→更新并提交 |
| 13 | `compact-temporal-tile-and-fuse` | `pending` | 消费第12项actual structural Card/TileRegion和free temporal choice；使用pinned MLIR SCF tile-and-fuse实际改写current SSA，attention只允许接口明确支持的output/parallel tiling及无需推测内部use/replica的外部exact edge | exact total single-valued relation才消除派生参数，non-unique/unsupported/indeterminate不缩减raw domain或Region candidate；一个traversal一个canonical `scf.for` nest；1024无remainder clone，1025/1031只产生必要main/remainder且不peel first，`r`个ragged axes最多`2^r`；未融合producer loop外一次，multi-use默认不clone，只有explicit replica实际复制；每个attention occurrence保持op kind/algorithm/type且只由outer tile/tail/replica解释 | 读AGENTS/progress→读05–07/10–11/19及本项矩阵→调研MLIR SCF/Linalg tiling、producer fusion、reduction tiling、loop peeling和成熟compiler fusion control→查pinned API→实现request-local exact query、standard tile/fuse、late remainder与窄exact adapters→fresh 1024/1025/1031 structural output、producer occurrence、attention opaque和第14项direct-input witness→重读设计/完整diff/MLIR复审→更新并提交 |
| 14 | `selected-attention-lowering` | `pending` | 消费第13项candidate-owned structural IR中保持opaque的current attention op，以及固定FA/FD和尚未消费的K1/K2、FD contribution/merge choice；05号唯一transformation直接生成actual Linalg/Tensor/SCF、coupled state、slice与canonical loops，并只对这些new loops复用late remainder specialization | 每个attention occurrence只lower一次；FA一个K2 owner的online recurrence，FD的每个selected contribution、coupled merge/finalize和cross-region tensor boundary all-and-only；1024无attention remainder，1025/1031 static main/tail exact且无first；layout入口attention为0；不重跑e-graph/generic tile-and-fuse，不clone Card/Tile owner，不恢复第11项已删future inventory/replay | 读AGENTS/progress→读05–07/10/19及本项矩阵→调研FlashAttention/FlashDecoding、MLIR coupled reduction/attention lowering和成熟compiler semantic-op decomposition→查pinned attention/Tiling/SCF API→从第11项保留的算术/relation/oracle实现current-op direct rewrite与late remainder→fresh 1024/1025/1031 FA/FD actual Linalg/SCF、failure atomicity及第15项layout-input witness→重读设计/完整diff/MLIR复审→更新并提交 |
| 15 | `current-ir-layout-bufferization` | `pending` | 消费第14项最终current SSA/use graph，唯一拥有完整value/use layout domain、op tuple constraints、query-local exact PBQP assignment/apply、IndexRelation+PhysicalLayoutRelation exact view、function-boundary/region-local bufferization和observable output DPS | solver与flat oracle一致；layout-polymorphic op可传播assignment并只为不兼容edge创建materialization；exact view零allocation/copy；soft projection unknown时整组禁用；bufferization恰一次；冗余publication copy为0、必要copy有witness并typed；same-layout/unused为0、shared conversion一个SSA；输出是第16项直接消费的layout-resolved current IR | 读AGENTS/progress→读05–11/19及本项矩阵→调研PBQP、resource-aware projection、One-Shot Bufferization和MLIR layout/view实现→查pinned interfaces/API→补value/use domain、tuple factor、assignment apply与physical-map view并复审已有DPS/copy实现→fresh solver oracle、layout/copy inventory及第16项direct-input witness→重读设计/完整diff/MLIR复审→更新并提交 |
| 16 | `current-ir-downstream-orchestration` | `pending` | 在不建立complete materializer的前提下，把第15项layout-resolved owner依次交给current movement/boundary、execution-structure immediate apply、TileRegion→Instr、worker/order/fresh completion和唯一`compileCanonicalInstructionTilesToExecutable` actual leaf；每个atomic stage仍由原owner实现，orchestration只固定current-IR调用与typed handoff | 任何缺失的direct current movement/execution/completion API先在其原owner修复，不恢复Movement/Buffer/SchedulePlan；local/DDR/peer、Serialized/pipelined、tail/rotating slot、Instr、join/wait硬件witness与actual MiniMalloc全部到达；`compileCardModuleToExecutable`、`CompleteCandidatePreparation`为0；输入choice不枚举、失败不repair/fallback、Accepted owner不重建 | 读AGENTS/progress→读06–14/19及本项矩阵→读movement/completion/memory硬件事实→调研MLIR staged lowering与LLVM pass-pipeline ownership→查pinned conversion/SCF API→逐stage接唯一current transform并删除隐藏facade→fresh 1024/1025/1031 movement→execution→Instr→completion→leaf与typed failure验证→重读设计/完整diff/硬件/MLIR复审→更新并提交 |
| 17 | `baseline-current-ir-integration` | `pending` | 重新启用`none`：baseline controller只产生固定spatial/Region/free-temporal及后续显式choice，依次调用第12–16项atomic current-IR stages；actual capacity rejection创建新的完整attempt，其它typed状态停止 | baseline不调用search state/domain/materializer，不构造`CanonicalBaselinePlan`或shadow reclose；每region一semantic root；e-graph→spatial/region→tile/fuse→attention→layout→movement→execution→Instr/completion→leaf均到达；fresh current FP16 LLaMA none≤15分钟、package唯一、strict readback/no-card通过 | 读AGENTS/progress→读05–16及本项矩阵→确认current baseline/source/package边界→读相关硬件/ABI事实→调研deterministic baseline current-IR controller→查pinned API→实现独立fixed-choice controller并重启none route→fresh focused矩阵和none package/no-card→重读设计/MLIR复审→更新并提交 |
| 18 | `search-current-ir-integration` | `pending` | 重新启用`search`：保留Spatial/Region/free-Temporal complete lazy controller，structural choice选中后立即由第12–16项actualize；layout/movement/execution/order等后续raw choices从各自current IR生成并立即apply；PBQP只作首proposal，Accepted current Instr actual result进入controller比较 | 不存在旧cutover/fallback/complete plan；e-graph只在policy分叉前一次；proposal开关不改raw domains；pre-structural state无future value/buffer/event；每个complete point一次actual leaf；controller从Accepted current Instr比较resource-aware objective并保留同一owner；baseline路径不变 | 读AGENTS/progress→读05–17/Q52及本项矩阵→读NE/CT、overlap和target-profile事实→调研current-IR search transaction、resource-aware cost与nested raw-domain traversal→查pinned API→把保留的controller/domain接到第12–16项typed actualizer→fresh search全链、raw-domain on/off、engine-cost反例、actual feedback及baseline隔离验证→重读设计/完整diff/MLIR复审→更新并提交 |
| 19 | `scale-regression-and-inventory` | `pending` | 在新actual-IR pipeline上profile并仅保留有证据的e-graph、PBQP、memo、priority、DP和LNS，补齐logical transform、spatial/Region actualization、fusion、attention、copy、layout conversion、transfer elimination与Instr只读汇总 | 完整e-node/e-class/match/budget、logical transform before/after、Tile/TileRegion/current SSA edge/producer-occurrence/attention actual decomposition/copy/layout-conversion/buffer/movement/execution-structure/Instr/target inventory；e-graph on/off保持exact语义和downstream reachability；proposal/PBQP on/off保持raw domains/accepted set；instrumentation on/off等价；actual MiniMalloc到达 | 读AGENTS/progress→读05/06/Q52及本项矩阵→调研search/equality-saturation scalability和PBQP proposal算法→查pinned MLIR/LLVM→改instrumentation/tests→fresh真实规模验证→重读设计/MLIR复审→更新并提交 |
| 20 | `llama-baseline-search-acceptance` | `pending` | 同一current FP16 LLaMA source顺序运行独立none和search事务 | 每次Release≤15分钟；各自package strict readback/no-card；两条路径互不调用；e-graph实际到达且无budget-dependent nondeterminism；均进入Instr/MiniMalloc/DDR/target；冗余DDR copy和Instr copy-only Region为0，必要copy已typed；search没有静态body倍增 | 读AGENTS/progress→重读05/06/Q52/Q53及本项矩阵→确认current source/tool和runtime/ABI边界→fresh顺序运行→逐项核对设计与MLIR/runtime规范→更新状态并提交 |

### Checkpoint 10实现与连续组合覆盖证据

- 唯一production调用位于`wafer-lower-stablehlo-to-linalg`的attention recognition之后；formal compiler driver与`wafer-opt` named
  pipeline调用同一个func pass。仓库没有scoped、candidate或post-tiling入口，也没有C++ fallback e-graph。
- Core使用pinned `egg 0.11.0` revision `fb6167957beb5dd7c784121459e08ebd1ccb1a00`、committed Cargo lock和offline
  directory source。Typed C ABI覆盖checked record、callback四态、Rust panic、Rust-owned alloc/free以及连续/并行determinism。
- 1024/1025/1031矩阵覆盖identity/composition、Compute input absorption、Concat flatten/common access、elementwise/reduction/contraction
  result reindex、15-use fanout、gap、budget unchanged、call和attention barrier；reduction与batch contraction结果直接进入pinned partial
  reduction tiler，不修改后端接受范围。
- 同一current FP16 HF Llama block以batch 2分别fresh capture sequence 1024和1025。两次正式pipeline均`verify-each`通过且无budget
  exhaustion；normalization均为431→374个op、实际提交删除19个Access、9个e-graph component改变，并执行3次all-users Access
  propagation。为闭合fanout变化，单次pass invocation内部达到actual Access/Concat严格下降定点；累计ABI work为1245→583 records、
  20208→9424 bytes，relation query 424、e-node 475、match 440。当前Release host profile两者均约142 ms/32.8 MiB；所有`arith`/`math`
  scalar op逐类计数不变。这些数值只校准work budget，不进入compiler选择；删除数只在actual rewrite提交后累计，unchanged request不计入。
- Materialization在首次mutation前preflight，随后按extracted拓扑统一插在旧root之前并直接rewrite current IR；不clone Module、Func或
  DAG。一次定位中发现的跨recipe insertion-point dominance bug已由同一fresh HF case覆盖，scalar region与compute occurrence不变。
- Formal `wafer-compile`产品入口使用同一个pipeline生成accepted 16-Tile package，strict manifest检查和`wafer-run --no-card`通过；该
  package gate只证明本stage接入未破坏正式driver，不代签第13/16项真实规模LLaMA package验收。

连续组合覆盖按能力而非预设“全消”记录如下：

| 分组 | 组合与shape | before→after op | work与结果 |
| --- | --- | ---: | --- |
| Access/单Compute | reshape/broadcast/common-access concat分别接elementwise、reduction、contraction；1024/1025/1031，含concat tail | 127→56 | 实际提交删除29个Access；e-node 158、relation query 253；8个changed component均multi-rule，0 exhaustion |
| 多Compute/分支 | 4-Compute交替Access、elementwise→reduction、contraction→elementwise、fanout diamond、broadcast→contraction、异构fanout、暂时init use及observable/unsupported-contraction barrier | 201→130 | 实际提交删除32个Access；e-node 187、relation query 311；4次all-users propagation；aligned/ragged elementwise+reduction及ragged elementwise+reduction+contraction共享同一source且不复制Compute，暂时use在同次pass闭合，两个真实barrier保持，0 exhaustion |
| 深链尺度 | 4/8/16/32个Compute，每层交替transpose，rank-3×1025 | 30→18 / 58→34 / 114→66 / 226→130 | e-node 41/89/185/377，relation query 100/236/508/1052，match 106/258/562/1170；Compute/scalar分别保持4/8/16/32，0 exhaustion |

组合测试发现并修复的实现缺陷包括：不可物化的reshape/projected可选composition不再把其它合法路径变成WorkLimit；Rust relation service
增加request-local typed memo，16层链从8193次query和budget exhaustion降到508次并成功；Elementwise result reindex增加显式operand
Access的第二RHS，使`transposed contraction→elementwise→transpose`连续闭合为现有`linalg.batch_matmul + elementwise`；共享projected
Access在全部elementwise/reduction/contraction data uses均可表示时原子传播，canonical contraction直接成为对应matmul variant；e-graph删除
暂时旁支后暴露的新边界在同一次pass内达到定点，第二次运行byte-equivalent。任一observable use或unsupported contraction signature仍使
整组保持。统计只在current IR实际提交后记录删除量，不用unchanged extraction的空output字段推算。

## Checkpoint 11–14 确定实现边界

### Legacy shadow materializer retirement

```text
Pipeline position:
- Upstream IR / input:
  第10项完成后的active source/CMake/test；`none`仍经`CanonicalBaselinePlan`，`search`仍经Spatial→Region→Temporal后调用
  `materializeSearchStructuralCandidate`，两者最终进入隐藏layout/bufferization/completion的`compileCardModuleToExecutable` facade。
- Current stage responsibility:
  先删除两条错误product route和所有future/shadow事实owner；把仍正确的纯choice算法、relation、actual rewrite mechanics和oracle迁到
  直接owner。此项不实现替代candidate pipeline。
- Output IR / files:
  repo保持build/link；normalization、direct stage APIs/tests和`compileCanonicalInstructionTilesToExecutable` actual leaf仍可用；
  `none`/`search` CardExecutable与package暂不可用。
- Downstream consumer:
  第12项只从保留的TensorProgram/Spatial/ExactDemand/Region partition资产构造new actual structural IR，不读取被删schema。
- User-level driver / named pipeline:
  `wafer-compile --optimization-policy=none|search`在CardExecutable边界返回`operation_not_supported`；不调用另一个policy、actualizer、
  MiniMalloc或package writer。
- Explicit non-goals:
  不维持临时production continuity，不增加compatibility wrapper/V2，不把旧source移出CMake后继续当隐式协议，不顺带实现新pipeline。
- Completion criteria:
  删除/保留矩阵、routing负例、build/link、direct retained-assets与actual-leaf矩阵全部通过；active docs只描述new target chain。
```

删除前的active调用链冻结为：

```text
none:
  CardExecutable policy route
    -> compileCardBaseline
    -> CanonicalBaselinePlan
       [Spatial, ExactDemand, RootWork, Region, Temporal,
        Representation, Movement, Serialized, Storage, Schedule,
        AttentionWork, PreparedAttention]
    -> materializeBaselineCandidate
       -> generic StructuredNodeShardGroup or expandSelectedAttentionAlgorithm
    -> compileCardModuleToExecutable

search:
  CardExecutable policy route
    -> PlanningProblem / PlanningSession / runUnifiedSearch
    -> Spatial -> Region -> Temporal
    -> evaluateCurrentStructuralCandidate
    -> materializeSearchStructuralCandidate
       ordinary: Region/Temporal -> StructuredNodeShardGroup
       attention: Representation -> Movement -> Serialized -> Storage
                  -> Schedule -> AttentionWork -> prepared replay
    -> compileCardModuleToExecutable

shared hidden facade:
  CardModule split
    -> current layout cleanup
    -> function-boundary bufferization
    -> selected preparation
    -> TileRegion-to-Instr
    -> transfer cleanup
    -> completion rebuild
    -> compileCanonicalInstructionTilesToExecutable
```

Baseline actual capacity rejection还会改变Temporal并重新关闭后续全部shadow components；search attention actualization虽然前台只枚举到
Temporal，内部仍重建Representation到Schedule。这两条事实是第11项同时断开none/search而非只删一个search wrapper的原因。

当前真实调用链按owner拆除：

| 现有层 | 第11项处理 | 保留或直接replacement |
| --- | --- | --- |
| `CardExecutable.cpp`的none/search product branches | 移除`compileCardBaseline`、PlanningProblem/Session/`runUnifiedSearch`调用；两种policy均明确unavailable | normalization与OptimizationConfig parsing保留；第17/18项分别重启 |
| `CanonicalBaselinePlan`及reclose | 删除完整schema、builder、caller与only-purpose tests | canonical Spatial/ExactDemand/RootWork算法保留；第17项用局部fixed choices重建controller |
| `CompleteCandidatePlan`、`materializeCardCandidate`、`materializeSearchCardCandidate` | 删除；它们已无production caller | 无replacement；actual owner取代complete plan |
| `CompleteCandidatePreparation`与`evaluateCompleteCandidate` | 删除；不得保留test-only replay API | 第16项直接串联atomic current stages |
| post-temporal `RepresentationState`、`MovementState`、`InitialBufferState`、`ExecutionStructureState`、`BufferState`、`ScheduledState`及continuation/memo | 从PlanningState/Session删除 | Spatial/Region/Temporal controller core保留；后续choices从各自current IR生成 |
| `UnifiedSearch`与`ActualResultController`对`FullFeasibility`/old materializer的直接依赖 | 删除直接actualizer调用和旧result include；product route断开 | 纯lazy frontier、work accounting、typed accepted-result ownership与cost comparison在接收caller-owned actualizer/result的窄API后保留给第18项 |
| `RegionPlan` delivery/placement、nested invocation和`StructuredNodeShardGroup` construction directives | 删除错误字段与旧group materializer API | connected partition/explicit replica算法保留；第12项定义new current-IR input |
| Representation/Movement/Storage/Schedule cross-stage plans | 删除production state、canonical replay和future resource records | PBQP solver、exact transfer proof、current execution materializer、Instr completion mechanics分别保留在直接owner |
| `AttentionWorkDescription`、canonical projection、prepared decomposition及action/value/scratch occurrence mapping | 删除 | attention op interfaces、`AttentionLinalgOps`、FA/FD算术与独立reference/oracle保留给第14项 |
| `compileCardModuleToExecutable`隐藏facade | 删除；不能继续藏layout、bufferization、Tile→Instr或completion mutation | `compileCanonicalInstructionTilesToExecutable`保留；第16项显式重接每个stage |
| `StructuredMaterializationRelations` | 保留 | 只记录本次actual rewrite已经创建的operation/buffer，并由candidate owner随IR epoch更新 |

删除后只允许重建下面一条artifact链：

```text
verified TensorProgram
  -> policy-owned Spatial / Region / free-Temporal choices
  -> [12] candidate-owned actual Card/TileRegion
  -> [13] compact temporal tile-and-fuse
  -> [14] selected-attention lowering
  -> [15] current layout query/apply + bufferization + exact views
  -> [16] movement/boundary
          -> execution-structure immediate apply
          -> TileRegion-to-Instr
          -> worker/order/fresh completion
          -> compileCanonicalInstructionTilesToExecutable
```

第17项baseline只产生fixed choices；actual capacity rejection创建new complete attempt。第18项search保留complete lazy
Spatial/Region/free-Temporal traversal，并在每层current IR上生成后续raw choice、立即apply和fresh analyze。两者共享atomic mechanics与actual
leaf，不共享controller、candidate owner、fallback或winner。

保留源码不等于保留旧API：例如Region/Temporal、attention和execution structure文件中的有效算法必须先迁到不依赖future schema的
直接helper或测试资产，然后原旧type/caller删除。无法与旧schema分离且没有direct consumer的fixture不迁移。

删除后的routing矩阵使用rank至少3且主要维度为1024/1025：两种policy都在CardExecutable分析或candidate创建前返回同一明确类别；
source IR byte-identical，baseline/search materializer、actual leaf、ProgramData adoption和package publication计数均为0。Direct
`compileCanonicalInstructionTilesToExecutable`仍以completion-closed Instr正负例运行，证明删除的是错误上游而非actual leaf。

### Spatial and Region current-IR materialization

```text
Pipeline position:
- Upstream IR / input:
  verified current TensorProgram；selected `SpatialPlan`关闭为exact `SpatialAssignment`，ExactDemand/RootWork已经从同一IR epoch重算；
  Region choice只含connected membership和explicit replica。
- Current stage responsibility:
  创建candidate-owned CardModule、all-and-only TileModules和non-nested structural TileRegions；ordinary spatial pieces立即成为actual
  operations/SSA。Attention保持opaque，尚未消费的K1/K2 contribution/merge和free temporal choice继续作为显式参数。
- Output IR / files:
  verifier-valid structural Card/TileRegion IR；actual Tile、iteration intervals、operation、SSA和Region membership已经存在；无temporal loop、
  layout、movement、buffer、worker、completion或offset。
- Downstream consumer:
  第13项compact temporal tile-and-fuse。
- User-level driver / named pipeline:
  focused direct API测试；第17/18项分别由baseline/search controller提供choice并调用同一atomic structural transformation。
- Explicit non-goals:
  不选择或执行fusion，不展开attention，不创建DDR/peer carrier，不预测SPM，不把SpatialAssignment或Region choice保存到下一IR stage。
- Completion criteria:
  Spatial/Region覆盖矩阵、failure atomicity、candidate ownership、verifier和第13项direct-input witness全部通过。
```

Spatial保留下列明确choice：

```text
SpatialPlan
  iterator partition scheme/parameter
  logical-shard-to-Tile embedding
  reduction merge Tile
      ↓ exact close
SpatialAssignment
  ordered nonempty iterator intervals
  physical Tile for each logical shard
  reduction group and merge Tile
```

`SpatialPlanDomain`继续惰性覆盖partition scheme、injective embedding和merge placement；graph-coherent/topology proposal只重排普通domain
member，不形成shortlist。Close只产生query-local exact intervals，不签发operand demand、layout、movement、buffer、schedule、cost或SPM结论。
ExactDemand/RootWork随后从current TensorProgram验证producer/consumer coverage。

Region choice对每个相关producer只含`external`、`local-once`和`explicit-replica`。Materializer不决定producer最终位于loop内或loop外，
也不建立stored/direct/nested delivery。Ordinary operation按selected spatial interval直接创建到对应Tile/Region；reduction merge与attention
K2 contribution尚需第13/14项消费时保持显式choice，不创建代表未来partial/merge op的ID。

FA/FD algorithm已经是attention op事实。Spatial只选择output/parallel pieces、K2 spatial partition、Tile embedding和FD merge Tile：FA的
K2 contribution count必须为1，FD必须至少两个nonempty、无重叠且完整覆盖。第12项创建opaque attention occurrence与目标Region；第14项才
在这些actual Regions中生成QK/PV/state/merge。

Current `TileRegionOp`允许body只含terminator，因此FD可在第12项创建actual target Region shells：opaque attention root留在selected final-owner
Region，其它contribution/merge target Regions暂不含可执行compute。Shell本身只表达已经选择的Region membership，不代表未来operation、SSA
或buffer；它必须由同一explicit attention contribution/merge choice逐一解释，只能顺序通过第13项并由第14项填入actual work，禁止进入layout。
第12项stage check检查这一直接handoff，不在operation verifier中遍历全Card重建future inventory。

该transformation创建的是最终candidate subtree，不是scratch clone。Caller需要试行alternative时只在controller边界建立一次最近的
`IsolatedFromAbove` owner；所有外部operand通过`IRMapping`映到new owner，failure擦除整个subtree，success由同一owner进入第13项。

### Compact temporal tile-and-fuse

```text
Pipeline position:
- Upstream IR / input:
  第12项提交并verify的candidate-owned structural Card/TileRegion、保持opaque的attention semantic op，以及已经选定的free temporal choice。
- Current stage responsibility:
  从current SSA生成ordinary canonical SCF temporal loops并执行producer fusion；只消除由exact relation唯一决定的冗余参数；
  late specialize ragged remainder。Attention只作为接口可见的opaque structured op参与外部edge和output/parallel tiling。
- Output IR / files:
  candidate-owned structural Card/TileRegion IR；ordinary producer的loop内/外位置已由actual SSA表达，attention op仍存在且内部未展开。
- Downstream consumer:
  第14项selected-attention lowering；无attention的candidate直接进入第15项current-IR layout/bufferization。
- User-level driver / named pipeline:
  focused direct API测试；第17/18项分别由baseline/search candidate owner调用同一个atomic loop/fusion transformation。
- Explicit non-goals:
  不展开或选择FA/FD，不创建layout/buffer/movement/completion，不决定SPM legality或winner，不调用Transform dialect或全图e-graph。
- Completion criteria:
  下述domain、actual rewrite、tail、multi-use和typed failure矩阵全部通过，输出直接被第14项或第15项消费。
```

本stage不重新枚举Region。它只读取第12项actual TileRegion membership和current SSA；`local-once` producer是否进入consumer loop由
standard fusion实际结果决定，`explicit-replica`必须已经作为actual producer存在。Stored/direct、nested placement、rewire和storage均无输入字段。

Temporal domain只保存root/output tile、无法唯一推导的producer tile、reduction自由轴和dependence-legal loop order。每个自由轴的
合法值仍在lazy raw domain中，不预先构造Cartesian product。一次materialization调用内的只读exact tile-relation query概念输入为：

```text
current producer/result
current consumer/operand
selected free tile parameters
current indexing maps + IndexRelation/ExactIndexSet
```

它不调用可能修改IR的`TilingInterface` builder，不读取名称/shape策略，不保存derived offset/extent到`TemporalPlan`，不构造future op/SSA、
buffer、occurrence或footprint。结果必须区分：

- `Exact`：只有total single-valued proof才返回，并允许把对应producer参数作为派生量；
- `Unsupported`：当前relation/interface未实现，保留独立producer与Region candidate；
- `Indeterminate`：无法精确证明，处理同上，不能变成rejection/no-good；
- `BrokenContract`：current verifier-valid合同矛盾，终止candidate并报告compiler error。

若relation存在多个合法producer tile，相关参数继续是自由变量并完整枚举；选定producer/consumer参数后只有exact pair才允许fusion。
因此该query只减少重复或不一致的参数坐标，不减少supported actual-IR候选。Proposal、cost、priority和actual capacity feedback
只能改变访问顺序或构造下一完整candidate，不能据此剪掉其它free choice；SPM结论仍只来自下游actual MiniMalloc。

Actual rewrite使用pinned `scf::tileConsumerAndFuseProducersUsingSCF`及其standard recursive producer worklist，由Wafer callback检查同Region、
current use-def、exact relation、dominance、effect、`TilingInterface`和无隐式replica。Contraction/reduction不是统一barrier；result tile
all-and-only且无重叠、standard interface与loop-carried accumulator均可表达时允许，partial-reduction只保留已有窄adapter。Collective、
cross-region、overlap、effectful或unknown relation保持未融合current producer。

Multi-use固定为：同一consumer的相同exact request由standard fusion后scoped CSE合成一个producer tile；多个consumer默认共享一个
loop外producer；只有`explicit-replica`先实际创建额外producer后才允许分别融合。不得用自定义
`MaterializedCoupledProducerTile` cache、operation ordinal或future delivery恢复共享。

Standard tiling先形成一个canonical bounded loop nest。随后对ragged loops从内到外调用pinned
`scf::peelForLoopAndSimplifyBounds`，用`scf::ForOp::promoteIfSingleIteration`提升单次tail loop，并在candidate owner内运行bounded canonicalization、
`eliminateCommonSubExpressions`和DCE。Aligned axis不产生remainder；`r`个ragged tiled axes最多形成`2^r`个static main/tail组合，
不生成first/prologue。非`TilingInterface`的stream/copy只复用机械main-loop+static-tail helper，不能进入structured fusion。

Controller为一次attempt至多新建或clone一次最近的`IsolatedFromAbove` owner；如果上游已交付candidate-owned IR，本stage直接rewrite。
Atomic tile/fuse不创建第二个scratch Card/Tile owner。Standard tiling产生的tiled op是保留的actual result，不是candidate transaction clone；
失败销毁owner，成功owner直接下传且不重放。

Attention在本stage产生的每个actual occurrence必须保持同一op kind、fixed algorithm、type、result和opaque内部语义；新增occurrence
只能由selected outer tile、necessary main/tail variant或explicit replica逐一解释。它当前`TilingInterface`只在K1/K2保持完整时
支持output/parallel tile；exact query只能读取公开operand/result relation。它不得创建或检查QK、PV、Maximum/Sum/Accumulator、
contribution或merge，也不得为了fusion先调用selected-attention lowering。

### Selected attention lowering

```text
Pipeline position:
- Upstream IR / input:
  第13项输出的candidate-owned structural Card/TileRegion；current attention op仍存在，FA/FD已经固定，K1/K2 block及FD
  contribution/merge是尚未消费的显式choice。
- Current stage responsibility:
  在candidate CardModule这一最近共同owner上，把每个current attention op一次性改写为selected FA/FD的actual
  Linalg/Tensor/SCF、slice、coupled state和control flow；FD可在多个TileModule/TileRegion内创建contribution与selected merge；
  最后只对本次新建的ragged loops复用第13项late remainder specialization utility。
- Output IR / files:
  verifier-valid structural Card/TileRegion；attention op为零，所有scratch/state/contribution/merge均由current SSA/region/control flow表达。
- Downstream consumer:
  第15项current-IR layout/view/bufferization。
- User-level driver / named pipeline:
  focused direct API测试；第17/18项分别由baseline/search candidate owner调用05号定义的同一个atomic transformation。
- Explicit non-goals:
  不重新选择FA/FD，不重跑e-graph或generic tile-and-fuse，不选择layout/movement/worker/completion，不创建Instr/join/wait，
  不预测SPM，不新增attention op或通用Wafer fusion interface，不直接创建wafer.tile attention。
- Completion criteria:
  FA/FD真实规模矩阵、typed failure/atomicity和唯一new API通过；不接收或恢复第11项已删除future inventory；输出由第15项直接消费。
```

FA为每个selected output piece生成一个K2 spatial owner及其canonical temporal recurrence，Maximum、Sum和Accumulator是同一`scf.for`
的actual loop-carried SSA；QK、scale/mask、block reduction、PV、state combine和finalize均为current Linalg/Tensor/SCF。FD为每个selected
K2 contribution生成同一局部recurrence，并在selected merge owner创建actual coupled combine/finalize；跨Tile/Region state先作为current
structural tensor boundary，随后由layout和movement闭合，不提前创建route或buffer。Attention new loops遵守同一static tail合同：1024
无remainder，1025/1031只生成必要main/tail并且不peel first。

Lowering只读current attention op、其standard/Wafer semantic interfaces和显式choice；不消费`AttentionWorkDescription`式future
action/value/physical-version ID、`SelectedAttentionDecomposition` replay或nested invocation class。短生命期preflight只验证当前op、maps、
choice与symbol closure；首次mutation后所有create/replace/erase通过同一rewriter。Failure销毁candidate owner，不换algorithm、builder或
opaque fallback；success擦除attention op并把同一owner交给layout。

## Q52逐项覆盖矩阵

### 1. Expansion evidence

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| instrumentation off/on | generic chain/fanout及同一LLaMA source；rank 3–6，1024/1025/1031 | report sink失败不改compiler result | source identity、stage reachability、op/relation inventory和failure类别一致 | 后续checkpoint的冻结输入 |

### 2. Verifier cleanup

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| local op/container/stage | Card/Tile/collective/TileRegion正负例；1024/1025 actual Card | malformed local IR由op verifier拒绝，跨op错误由stage check拒绝 | verifier不读parent sibling或重放未来IR | named/driver `verify-each`与actual memory/ABI gate |

### 3. Search execution materializer donor

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| generic/attention execution | single/multi-root/replica/coupled、fanout；rank 3–6，1024/1025/1031 | selected structural recipe不可表达时typed failure | 每execution all-and-only一次，fanout不重复compute | 第12项Spatial/Region、第13项compact、第14项attention与第18项search integration的算法资产 |

### 4. Baseline root materializer

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| root isolation与actual feedback | chain/fanout/matmul/attention；rank 3–6，1024/1025/1031，16 Tile | 只capacity rejection推进，其它typed状态停止 | 每region一root；candidate CardModule数等于actual planner数；accepted不重建 | baseline Instr/MiniMalloc/package |

### 5. Current-IR contract reset

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| active docs/code call graph | 01、05–19、Q52 plan/progress、memory；baseline/search两条路径 | 一个旧plan能力尚无actual owner时保留为待迁移，不伪装已删 | 每个字段分类为choice/current fact；active文档无正向shadow合同；删除清单有new owner/test | 第6–18项consumer-first施工、early retirement与new integration的唯一边界 |

### 6. Actual leaf current-IR contract

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| completion-closed canonical unplaced Instr、actual allocation/alias/effect/lifetime | baseline focused current source；aligned/ragged；rank 3–6，1024/1025/1031；16 Tile | capacity、ResourceExhausted、timeout、unsupported、compiler error保持区分 | leaf不改变function boundary或join/wait；MiniMalloc/offset/conflict/owner只来自current IR；读取policy/preparation/shadow state次数为0；Accepted owner不重建 | 第11项删除后direct leaf仍可用；第16项orchestration、第17项none与第18项search消费同一typed outcome |
| leaf隐藏行为与legacy wrapper retirement | 已闭合Instr正例，以及tensor function boundary、missing completion、preexisting offset负例；rank 3–6，1024/1025/1031 | 输入合同缺失按typed failure停止，不bufferize、不补join、不移动allocation | leaf内bufferization/completion/sink调用均为0；组合prepare wrapper caller为0；SPM/DDR high-water逐一等于accepted range end的重算结果 | 第11项删除隐藏facade后，第16项从各atomic owner直接接入leaf |

### 7. Current Instr schedule and completion

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| baseline function-boundary-bufferized、execution-structure-closed TileRegion中的compute/DDR/DTE/control flow | straight-line、loop/tail、cross-worker、observable terminal；rank 3–6，1024/1025/1031 | missing hardware/ABI语义为typed unknown；token/effect malformed为IR failure | 本stage不运行bufferization；TileRegion-to-Instr只转换一次且relation保持current；event graph fresh重算，order后失效；minimum/latest join/wait有直接hardware/effect/lifetime witness；第6项不再重建completion | fresh TensorProgram经baseline materializer和第8项输出current Instr，直接运行第6项leaf |

### 8. Current Tile execution structure

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| movement-closed physical TileRegion上的Serialized与selected pipeline | straight-line、nested loop、multi-wave、prefix/steady/tail、observable publication；rank 3–6，1024/1025/1031 | recurrence、effect、slot reuse或required completion obligation无法由current SSA/effect/token表达时typed unknown/unsupported，不退回预测buffer | Serialized IR byte-equivalent；pipeline的chunk occurrence all-and-only；prefix/steady/tail cover完整；rotating roots、slot selection、loop-carried SSA和reuse obligation显式；不创建join/offset | fresh actual TensorProgram经同一transformation的Serialized与pipelined choice后顺序通过第7→6项 |

### 9. Current-IR movement and boundary closure

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| baseline current producer/use与layout-resolved endpoint；local/DDR/peer/relay/gather | equal/partial window、fanout/fanin、reduction、attention；rank 3–6，1024/1025/1031 | alias/effect/route不明时typed unknown/unsupported，不猜近似carrier或改layout | compatible local edge零DDR；compute数不随fragment增长；staging/boundary、payload、token和effect all-and-only；physical TileRegion无未闭合tensor boundary；baseline不消费旧movement plan | fresh TensorProgram经baseline materializer和本stage输出physical TileRegion，顺序通过第8→7→6项 |
| layout-resolved→physical form transition | direct region chain、cross-region DDR、cross-Tile peer/collective；1024/1025/1031 | malformed bridge、missing endpoint或wrong-form residue由直接stage typed拒绝 | 不使用phase attr；所有logical tensor boundary恰好闭合一次；SPM memref不成为region I/O | physical-form stage verifier与第8项execution-structure consumer |
| current full/partial transfer cleanup | full equivalent、same-map view/donation与partial/permuted/layout-changing；straight-line及supported/unsupported loop | relation、alias、effect、lifetime或control flow不能证明时保留transfer，不猜等价 | full/same-map且安全时删除；partial、permuted、真正layout-changing、unknown ownership保留；cleanup只有一个production caller | 第8项actual movement/slot inventory与第7→6项下游 |

### 10. Structured logical e-graph normalization

| 输入等价类 | shape/结构 | typed/optimization failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| multi-rule ordinary graph | rank 3–6；1024/1025/1031；Access/Concat/Compute长链、chain/diamond | relation unknown或任一work limit保持原component | common-access extraction→composition→identity elimination由不同egg rules连续创建；Importer equivalence/final candidate为0；computeId multiset不变 | StructuredDAG、exact-demand与spatial domain直接消费canonical graph |
| continuous mixed Access/Compute chain | inverse reshape、transpose、broadcast、common-access concat分别位于elementwise/reduction/contraction前后；4个以上Compute交替Access长链、compute→reduction、contraction→elementwise及fanout diamond；elementwise/reduction/contraction异构fanout、暂时DPS-init use和observable barrier；1024/1025/1031 | 只有可证明且下游可表示的连续子链rewrite；general reshape、missing loop-bound map、concat gap/overlap和unsupported contraction signature保留；mixed reshape/projected可选composition不得把其它合法路径变成work limit；异构fanout任一use不合法时整组保持，observable use持续阻挡 | focused case逐op检查Access消除、最终maps、rule-kind组合、compute/scalar/iterator/init all-and-only及二次运行byte-equivalent；elementwise可用显式operand Access的第二RHS向single-use producer传播result reindex，但结构不下降时不得提取；multi-use projected Access对全部pure single-result Linalg data uses完成preflight后一次性传播，混合consumer逐项检查map、iterator、payload、init与共享source SSA；暂时use删除后在同一pass invocation以actual Access/Concat严格下降闭合 | StructuredDAG单root、partial-reduction tiler及既有named contraction lowering；fanout删除共享Access时所有consumer仍共享同一source SSA且不复制Compute |
| generic Compute operand absorption | elementwise、matmul/batch-matmul、generic contraction/reduce；1/2/15 uses；aligned/ragged | non-total/single-valued、DPS init、map不可表示或downstream unsupported时不rewrite | iteration domain/order、scalar/combiner、init/result、compute occurrence不变；broadcast进入reduction时原loop multiplicity保留 | 现有tiling/Linalg-to-Tile/reduction consumer不改后端即可消费 |
| restricted result reindex | elementwise/contraction/reduction parallel result transpose/reassociation；1024/1025/1031 | non-bijective、projection、slice、涉及reduction axis或map/init不同步时不rewrite | exact iteration bijection；所有maps同步；reduction domain/order和compute occurrence不变 | StructuredDAG、reduction tiling和直接conversion可消费 |
| recovered canonical concat | ordered N-ary/nested segments；common Access；1024/1025 tail | overlap/gap/partial coverage/dynamic/axis投影/intermediate external use不恢复或不提取 | exact prefixes与pieces all-and-only cover；common relation和axis remap exact；extract回标准Tensor IR | exact-demand piece propagation与consumer maps |
| attention/collective/effect barrier | ordinary graph邻接attention、collective、SCF/call/effect | rule不跨barrier；malformed输入由原verifier失败 | attention数量/type/result/attrs/algorithm/region不变；无attention e-node或scoped/candidate调用 | physical planning看到相同semantic roots |
| pinned `egg` relation-service C ABI | hermetic offline build；empty/single/dense；连续/并行请求；malformed record/callback/status/forced panic | configure/build缺依赖失败；typed Unsupported/WorkLimit/InternalError不发布partial rewrite | Cargo pin/许可固定；importer只导原始节点；dynamic Applier创建RHS；无MLIR对象跨ABI；handle同步不逃逸；alloc/free all-and-only | named/driver同一pass和adapter |
| e-graph budget/extraction/instrumentation | tiny independent oracle与fresh HF/LLaMA ordinary component | deterministic relation/e-node/match/rebuild/extraction limit不是legality failure | 相同budget产生相同IR/diagnostic；至少一个fresh真实component由两条以上rules有效改变；记录relation/e-node/e-class/match/iteration/wall/RSS | 第19项scale inventory与第20项完整pipeline reachability |

### 11. Legacy shadow materializer retirement

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| product policy routing retirement | `none`/`search`；generic与attention；rank 3–6，1024/1025；compile timing on/off | 两种policy均在CardExecutable边界返回`operation_not_supported`；frontend/normalization错误保持原分类 | analyze-card-program、baseline/search materializer、candidate actualization、actual leaf、ProgramData adoption、package writer调用均为0；source IR byte-identical；不得出现另一policy/fallback | 第12–16项direct APIs不依赖product route；第17/18项分别重启 |
| complete/shadow source/CMake retirement | baseline/search、ordinary/attention、active library与unit/lit | 有效算法或negative oracle尚无direct owner时先迁移；无法分离且只验证旧replay的fixture删除 | `CanonicalBaselinePlan`、`CompleteCandidatePlan/Preparation`、post-temporal PlanningState/continuation、future representation/movement/storage/schedule、delivery/nested temporal、attention inventory、old group/materializer API与`compileCardModuleToExecutable` facade全仓为0 | fresh build/link；retained Spatial/ExactDemand/PBQP/FA-FD/current-transform tests通过 |
| controller core decoupling | tiny exhaustive Spatial→Region→Temporal frontier与fake typed actual result | product actualizer不存在不构造candidate；fake result malformed为controller contract failure | UnifiedSearch/ActualResultController不include FullFeasibility/old plan，不调用IR materializer；frontier/order/work/accepted-owner tests保持 | 第18项只注入第12–16项actualizer，不恢复旧wrapper |
| retained actual leaf与current mechanics | completion-closed Instr、PBQP flat oracle、transfer cleanup、execution structure direct transform；1024/1025/1031 | direct input不满足本stage合同保持各自typed failure | `compileCanonicalInstructionTilesToExecutable`仍actual运行；retained helper无旧header/include依赖；source removal不靠stale object | 第12 Spatial资产、第14 attention资产、第15–16 current stages直接消费 |

### 12. Spatial and Region current-IR materialization

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| SpatialPlan close与actual Tile work | balanced/uniform partition；rank 3–6，1024/1025/1031；1/16 Tile；multi-axis embedding | partition/embedding/merge malformed=`BrokenContract`；source semantics不支持=`Unsupported` | exact ordered nonempty intervals all-and-only覆盖；Tile embedding injective；proposal on/off raw domain相同；actual TileModule/op/SSA逐shard存在且无future operation ID | 第13项直接读取actual TileRegion与free temporal choice |
| connected Region membership | chain/diamond/fanout/fanin、single/multi-root、reduction；external/local-once/explicit-replica | effectful replica、缺consumer或跨Tile local membership typed unsupported；mutation failure销毁owner | connected grouping完整；local-once只actual一次；explicit replica逐一实际存在；无stored/direct/nested/rewire/representation字段；Region non-nested | 第13项从current SSA决定fusion，不读取RegionPlan delivery |
| reduction及attention spatial constraints | contraction/reduction merge；FA单K2 owner；FD至少两个K2 contributions；rank 4–6，1024/1025/1031 | reduction coverage、FA/FD K2 constraint或merge Tile不exact时typed unsupported | reduction groups/contribution intervals/merge Tile all-and-only；attention保持opaque；FD target Region shells逐choice存在且只有terminator，不创建QK/PV/state/merge operation ID | 第13项保持attention opaque；第14项填充shell并消费显式K1/K2 contribution/merge choice |
| transaction ownership与stage verifier | external operands、multi-function symbol closure、failure injection | preflight失败不mutation；post-mutation failure擦除完整candidate subtree | 每attempt至多一个new/clone `IsolatedFromAbove` owner；无scratch Module/Func；output无temporal/layout/movement/buffer/completion | 第13项直接verify并rewrite同一owner |

### 13. Compact temporal tile and fuse

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| free/derived temporal domain | 第12项actual Region；rank 3–6的1024/1025/1031真实矩阵；另用注明原因的tiny有界oracle；identity/bijective/projected/non-unique/unsupported relation；全部dependence-legal loop orders | unsupported/indeterminate保持独立producer；broken contract终止candidate | 只有total single-valued参数从free domain移除；non-unique producer raw domain不变；query on/off canonical actual-IR集合相同；proposal顺序不改域 | 第14项收到相同attention choice；无attention case由第15项消费 |
| canonical SCF temporal loop | parallel/reduction/mixed iterator；rank 3–6，1024/1025/1031；aligned、单轴ragged、双轴ragged | pinned interface不能表达selected tile或static remainder不能闭合时typed unsupported，不回退旧builder | 1024一个shared body且无remainder；1025/1031只含必要main/remainder，不peel first；双ragged轴最多四种static组合，一般不超过`2^r`；loop/body数量不随trip count增长；reduction accumulator与loop-carried state exact | 第14项保持outer loops；第15项直接消费final loop/use graph |
| current-IR producer fusion | same-region SSA、single/multi-use、chain/diamond/fanout、reduction/contraction、reshape/insert、collective | result-tile relation、dominance、effect或无隐式replica条件不能证明时保持未融合current producer | 未融合producer loop外一次；相同request scoped CSE后一个producer tile；多consumer默认不clone；只有actual explicit replica分别融合；除explicit replica外iteration tiles无重叠 | 第15项layout use-binding与第19项producer-occurrence inventory |
| attention opaque boundary | FA/FD；batch/head/seqlen/head-dim rank 4及multi-axis rank 5–6；1024/1025/1031 | K1/K2 partial请求、内部relation query或需要内部use/replica推测的外部edge为typed unsupported | 每个attention occurrence保持op kind/algorithm/type/result与opaque语义；新增occurrence逐一由full-K1/K2 output/parallel tile、tail或actual replica解释；QK/PV/state occurrence为0 | 第14项直接读取current attention op和剩余choice |

### 14. Selected attention lowering

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| FlashAttention selected lowering | FP16/BF16；batch/head/M/K1/K2/N rank 4–6；K2为1024及1025/1031，多K2 block和output piece | current op/map/choice无法形成完整coupled recurrence时typed unsupported；mutation失败销毁candidate | 一个K2 spatial owner；QK、scale/mask、Maximum/Sum/Accumulator、PV、combine/finalize all-and-only；三个state为同一actual loop-carried SSA；无完整score/probability tensor；1024无remainder，1025/1031 static main/tail exact且无first；attention和Instr/join/wait均为0 | 第15项为每个actual operand/state/scratch建立current layout domain |
| FlashDecoding selected lowering | FP16/BF16；至少两个K2 contributions；batch/head/query/KV/head-dim rank 4–6；1024/1025/1031与ragged partition | contribution coverage、merge owner或coupled component relation不exact时typed unsupported，不退回FA | contributions all-and-only覆盖K2且不重叠；每个局部recurrence一次；Maximum/Sum/Accumulator同一merge/finalize owner；cross-Region state只作为current structural tensor boundary；attention和Instr/join/wait均为0 | 第15/16项依次闭合state endpoint、movement与actual leaf |
| opaque-to-actual stage boundary | attention邻接ordinary producer/consumer、single/multi-use、multi-root、FD target Region shells、call/collective barrier | selected lowering前compact失败保持其typed状态；lowering失败不运行layout | 第13项前后attention opaque；第14项对每个actual occurrence一次并填充all-and-only target shells；不重跑e-graph/generic fusion，不clone Card/Tile owner，不更改algorithm；external SSA rewiring保持；layout入口无attention residual或未填充attention shell | 第15项stage verifier接受actual Linalg/Tensor/SCF并拒绝任何residual |
| deleted-inventory non-recurrence | generic/FA/FD与第11项保留的算术/relation/oracle | new lowering需要future action/value/materialization ID即为contract failure | active source不恢复`AttentionWorkDescription`、prepared replay、nested invocation或operation ordinal mapping；coupled semantic query只返回current maps/grouping | fresh build、05号interface/oracle及第15项direct consumer |

### 15. Current-IR layout and bufferization

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| final tile/fuse current SSA的function boundary与primary/shared/alias/use-local需求 | 1/2/15 uses；Tensor/NTensor/Cx/NCx；rank 3–6，1024/1025/1031 | relation或bufferization不能证明时保留explicit materialization或typed unsupported，不建立alias | function boundary与region-local bufferization只运行一次；shared layout一个SSA definition；alias同storage；allocation dominance/effect正确；无unused conversion；每个logical boundary有current endpoint；不消费PhysicalVersion | 第16项直接消费layout-resolved endpoint |
| value/use layout domain与op tuple factor | elementwise/convert flexible chain、GEMM/reduce fixed tuple、chain/fanout | tuple unsupported=`NoSolution`；budget/overflow=`Indeterminate`；malformed=`BrokenContract` | 每个current value/use all-and-only一个变量/binding；assignment直接改actual types/uses；只为不兼容edge创建materialization | movement transformation和layout-resolved stage verifier |
| exact logical/physical view | reshape/transpose/broadcast/concat residual；same/different physical map | physical-map、valid/padding、alias或write safety不exact时保留materialization | `Psource(R(i)) == Pdest(i)`时同storage view且零copy/allocation；否则actual movement显式 | movement count与第19项layout-conversion inventory |
| observable output DPS与copy regression | single/multi-result、loop-carried result、chain/fanout；rank 3–6，1024/1025/1031；16 Tile | output无法绑定唯一destination或copy必要性缺少SSA/alias/effect witness时stage失败 | 每个result直接写唯一destination；冗余DDR publication copy为0；必要copy all-and-only且movement后typed；Instr不创建copy-only Region | 第16项movement→leaf与第17项baseline integration |
| exact PBQP solver与soft projection | flat oracle R0/R1/R2/residual；NE/Vector/movement/control trade-off；真实rank 3–6 | relevant work/rate/multiplicity unknown时整组soft term禁用；hard infinity/overflow/无解分类保持 | optimal cost/tie与oracle一致；unique conversion descriptor只计一次；projected term与apply后fresh actual analysis一致；不得退回flat instruction cost | 第18项final objective、第19项projected-vs-actual报告 |
| 已实现cleanup回归 | same-layout、unused、1/2/15共享use、intervening alias write、necessary copy | alias/effect不exact时保持独立conversion/copy | same-layout/unused为0；shared一个SSA；source write阻止复用；necessary copy不误删 | current movement/instruction inventory |

### 16. Current-IR downstream orchestration

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| layout-resolved→physical movement | local、DDR、peer/relay/collective、partial overlap；rank 3–6，1024/1025/1031 | endpoint/relation/route/effect unknown保持typed unsupported，不猜carrier或改layout | actual movement/staging/token/effect all-and-only；compatible local edge零DDR；necessary copy typed；physical TileRegion无logical tensor boundary | execution-structure direct transform读取同一current owner |
| execution structure immediate apply | Serialized、multi-wave pipeline、prefix/steady/tail、rotating slots；1024/1025/1031 | recurrence/slot/reuse obligation无法从current SSA/effect表达时typed unsupported | Serialized byte-equivalent；pipelined chunk、loop、root、slot SSA、reuse obligation all-and-only；不创建join/offset | TileRegion→Instr直接消费rewritten current IR |
| Instr/order/completion | straight-line、loop/tail、cross-worker、DTE token、observable terminal | hardware/ABI证据unknown、token/effect malformed和order overflow分类保持 | TileRegion→Instr一次；worker/order applied后fresh构造minimum/latest join/wait；无证steady-state join为0；DTE wait不与NCC join混用 | completion-closed Instr直接进入actual leaf |
| actual leaf与facade retirement | accepted、SPM capacity、ResourceExhausted、unsupported、compiler failure；16 Tile | typed status原样返回controller，不repair/retile/fallback | `compileCanonicalInstructionTilesToExecutable`恰一次；actual MiniMalloc/DDR/transport/target到达；`compileCardModuleToExecutable`和`CompleteCandidatePreparation` caller为0；Accepted owner不重建 | 第17 baseline与第18 search调用同一atomic stage sequence |

### 17. Baseline current-IR integration

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| current TensorProgram + fixed baseline rules及第12–16项atomic mechanics | chain/fanout/matmul/attention；rank 3–6，1024/1025/1031；16 Tile；current FP16 LLaMA block | e-graph budgeted unchanged不是失败；PBQP非Optimal按typed状态停止；只有actual capacity rejection构造new smaller-temporal attempt；其它typed状态停止 | baseline调用search state/domain/materializer次数为0；无`CanonicalBaselinePlan`/shadow reclose；global e-graph一次；每attempt一candidate owner；spatial/region→tile/fuse→attention→layout→movement→execution→Instr/completion→leaf均到达；DDR fallback为0；fresh none package唯一 | 第18项search保持baseline隔离；第20项同源两policy验收 |

### 18. Search current-IR integration

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| structural及downstream current choices | chain/fanout/reduction/attention；rank 3–6，1024/1025/1031；16 Tile | choice preflight failure、post-mutation failure、actual typed outcome保持区分 | Spatial/Region/free-Temporal完整lazy traversal；每个structural tuple由第12–16项actualize；layout/movement/execution/order raw choices从各自current IR生成并立即apply；pre-structural state无future value/buffer/event；Accepted owner不重建 | 第19项完整stage inventory与第20项search acceptance |
| PBQP first proposal与raw layout enumeration | tiny complete layout oracle及真实chain/fanout/attention；Tensor/NTensor/Cx/NCx；1024/1025/1031 | PBQP `Indeterminate`不形成rejection/no-good；`BrokenContract`为compiler error | e-graph不进入layout key；PBQP on/off raw legal set相同；Optimal assignment恰为首proposal且只apply一次；solver result不进candidate key/IR；继续其它raw layout | actual movement/memory gate及第19项PBQP work/quality report |
| Accepted actual objective与winner | 相同instruction count但NE/Vector work不同、相同compute work但control或movement不同、mixed engine schedule；rank 3–6，1024/1025/1031 | relevant work/rate/schedule unknown、排序依赖未证NE/CT overlap或overflow时objective typed incomparable | final current Instr逐Tile采集NE/Vector logical ops；两种throughput、movement与instruction-control分别计价；flat instruction反例选中resource-aware winner；未比较其它accepted candidate不得称best | retained actual winner、coverage和第19项per-term inventory |
| early-retirement non-recurrence与baseline隔离 | generic/attention、none/search、旧fixture | new integration需要旧schema时按contract failure停止 | 第11项删除的type/builder/facade/CMake/current-doc仍为0；search不fallback baseline；baseline调用链不变；每complete point actual leaf一次 | fresh build、named/driver、baseline隔离、actual memory/target与第20项验收 |

### 19. Scale regression and inventory

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| mixed DAG/HF/LLaMA | rank 3–6，1024/1025/1031，16 Tile | stage未到达、e-graph budgeted unchanged与typed compiler/resource failure分开 | e-graph component/e-node/e-class/match/iteration/extraction work/wall/RSS、logical transform before/after、TileRegion/structured-execution/op分布、current SSA edge/producer occurrence、attention actual decomposition、copy、layout conversion、actual buffer/movement/eliminated transfer/execution-structure、final Instr total/per-Tile/per-kind、NE/Vector logical work和makespan；非explicit-replica compute overlap为0 | actual MiniMalloc、package strict readback |
| temporal refinement结构稳定性 | full tile、一次及多次actual capacity feedback；parallel/reduction root；1024/1025/1031 | actual rejection、unsupported和instrumentation failure分类保持 | 相同Region choice下refinement只改变自由tile/loop bounds/remainder；Region数不因copy fallback增加；static body/producer/materialize-layout数量不随wave trip count或无关root倍增 | 第18项actual candidates及第20项search acceptance |
| PBQP与其它search optimization | tiny exhaustive oracle与真实规模profile | timeout/resource/unknown-rate/overflow不伪装exact rejection或comparable winner | exhaustive PBQP on/off保持raw legal、actual accepted set和winner；budgeted run允许访问顺序和best-found变化，但必须保持coverage分类并记录solver work、proposal命中、PBQP-projected-vs-actual per-term delta、conversion/movement下降；其它safe optimization/instrumentation on/off保持result | retained actual winner一次publication |

### 20. LLaMA acceptance

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| same-source two-policy acceptance | current FP16 LLaMA block；两个独立process、ProgramData和output directory | timeout/OOM/skip/fallback/未进actual planner均失败 | 每次≤15分钟；source identity相同；IR/result不共享；package唯一；冗余DDR→DDR publication copy为0；必要copy在movement closure后typed且Instr无copy-only Region；search temporal feedback不产生静态body倍增 | strict loader、host reference、no-card |

## Q52旧路径删除清单

下列对象在第11项统一退出active production/source contract。可复用的算法或negative test先迁到直接actual-IR owner，随后在同一
work item删除旧owner；第12–18项不得重新引入：

- `CardExecutable.cpp`到`compileCardBaseline`/PlanningSession/`runUnifiedSearch`的旧policy routes，以及`CanonicalBaselinePlan`、
  `CompleteCandidatePlan`、`CompleteCandidatePreparation`、`materializeCardCandidate`/`materializeSearchCardCandidate`和
  `compileCardModuleToExecutable`隐藏facade；
- PlanningState/Session中Temporal之后的Representation、Movement、InitialBuffer、ExecutionStructure、Buffer与Scheduled state、
  continuation、memo和domain traversal；Spatial/Region/free-Temporal controller core只在与这些type解耦后保留；
- `RepresentationState`、`PhysicalVersionId/Plan`、旧`RepresentationDomain`和production `PhysicalVersionBuilder`；PBQP的exact
  R0/R1/R2/residual算法只在迁移为`current-ir-layout-bufferization`的current-IR query、修复全局tie并由baseline/search真实调用后保留，
  旧representation carrier/API删除；
- 以future physical value为source/destination的`MovementState/MovementPlan`和canonical movement replay；
- `InitialBufferState`、`BufferState`、pre-IR `StorageObjectId`、`StorageDomain`、`StructureSpecificStorageDomain`和
  canonical storage/lifetime replay；
- `ExecutionStructureState/Plan`作为cross-stage candidate state的路径；第8项只保留从current physical TileRegion枚举并立即应用的
  query-local choice，以及物化后的actual loop/rotating roots/slot SSA；
- `ScheduledState`、`ClosedSchedulePlan`、`EventId -> operation`物化后反查和completion parity；
- 携带上述shadow facts的`CompleteCandidateKey/Plan`、`FullFeasibility` domain rebuild和`CompleteCandidatePreparation`；
- function-boundary冗余DDR→DDR publication copy在Instr conversion中临时创建TileRegion、SPM staging和RDMA/WDMA的fallback；
  `current-ir-layout-bufferization`必须修复产生该copy的output DPS/bufferization实现并删除fallback；有witness的必要copy仍由typed movement表达；
- `LocalUseDelivery`、`DirectNestedValue`、`StoredRegionValue`、`ReconstructedRegionValue`、`rewireDirectSSA`及依赖它们重放
  future fusion/storage结构的Region choice；手写parallel/reduction `first / steady / tail`递归笛卡尔展开，以及不读取current
  SSA legality而递归融合任意`TilingInterface` producer的通用worklist/cache；`compact-temporal-tile-and-fuse`只保留pinned SCF
  tile-and-fuse和确有直接consumer的窄exact reshape/insert adapter；
- attention的future action/value/physical-version/materialization ID、`SelectedAttentionDecomposition` replay、nested invocation class及
  把这些record作为actual QK/PV/state/merge owner的caller；05号只保留从current attention op重算的iterator/map/coupled-group语义查询，
  selected-attention lowering直接rewrite同一candidate owner；
- donor movement扫描/替换/erase、edge-owned compute重建、first-use layout恢复、无producer action surface和only-purpose fixtures；
  现有test-only/Instr-only full-transfer eliminator的proof与正负测试迁到第9项唯一current-IR production owner后删除旧入口；
- active设计、memory、CMake和tests中把这些对象当作current事实源或supported production contract的内容。

只在一次rewrite内使用的typed choice、`IRMapping`、SSA lookup、exact tile-relation结果或query-local event/lifetime graph不在该删除
范围，但必须在IR mutation后失效，不能成为下一stage的事实源。Exact tile-relation结果只能减少已证明唯一的参数自由度，不能保存
future tile occurrence或签发SPM/resource结论。

## Search scalability

只有`search-current-ir-integration`完成后才重新解释profile和优化search工作：

- PBQP只优先一个由`current-ir-layout-bufferization`的current-IR domain证明合法的layout assignment；solver budget、soft-cost可用性和proposal开关不得改变
  raw layout域或exhaustive actual accepted set，不恢复Top-k layout截断；budgeted run若因访问顺序改变best-found，必须保持
  `BudgetedFeasible`/partial coverage并报告差异；
- memo只保存从immutable source IR或current candidate IR重算的pure typed query result，不保存IR owner、operation pointer、offset或actual result；
- component DP只在current IR可证明separator关闭时使用，无证明时回到base traversal；
- priority、memo和safe bound只改变choice访问顺序或重复工作；fixed beam/Top-k/LNS-only等有损策略只能返回
  `BudgetedFeasible`；
- LNS只在已有actual Accepted incumbent后启动，每个repair alternative仍必须物化为actual candidate并运行普通gate；
- 对accepted candidate记录TileRegion、buffer、movement、Instr、MiniMalloc/DDR/target工作和publication数，不记录预测IR inventory。

## Q53 Production Host Readiness

Q53只消费Q52签发的current none与search路径，不改变search算法、deadline或candidate。一个case只导出一次
immutable source artifact；两种policy分别重新parse/import，并在独立process、work directory、ProgramData owner和
output directory中完成编译。Source identity只证明输入一致，不授权共享Module、analysis、IR、CardExecutable或package。

### Q53 work item

| Work item | 直接输入 | 完成输出 | 固定执行流程 |
| --- | --- | --- | --- |
| `production-host-readiness` | Q52、Q60产品入口、Q55 current interface、Q56 board-ready package/runtime | fresh source/IR/package/oracle/runner/no-card矩阵和可直接串行上板的case；Q53状态到`board-ready` | 读AGENTS/progress→读02/06/14–16及本项矩阵→读hardware/runtime/ABI事实→调研host qualification与board-ready组织→查官方及pinned API→改runner/tests→fresh host/no-card验证→重读设计并按MLIR/runtime复审→更新并提交 |

### Q53覆盖矩阵

| 输入等价类 | shape/dtype/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| generic structured | chain/diamond/fanout/reduction/mixed movement；rank 3–6，1024/1025/1031；FP16/BF16 | source/IR/resource/package/runtime分类保持 | exact coverage、owner、tail、actual movement/buffer/completion all-and-only | current package strict loader/no-card |
| attention | HF prefill、functional two-step decode；1024/1025/1031；FP16/BF16 | unsupported/resource与compiler failure区分 | fixed FA/FD语义、actual physical IR和step continuation同policy独立 | host oracle、package/no-card和board case binding |
| representative model | current LLaMA block；FP16 mandatory | timeout/OOM/skip/fallback不计通过 | none和search分别fresh生成package；无cross-policy state | strict readback、CPU reference、no-card |
| board-case preparation | representative communication、attention/decode和LLaMA的两种policy | 缺package/input/oracle/guard/deadline即非board-ready | package、payload、all outputs、guard、continuation、timeout和固定串行顺序完整 | 后续board runner无需修改source或临时补oracle |

Q53完成要求registered case实际执行且非skip/unsupported；每个case使用本轮source和package；strict loader验证
canonical manifest/module/program-data和all-and-only 16 Tile entries；no-card在provider side effect前关闭resource、binding、
transport和completion；host oracle与guard通过。真实板测只能由Q53之后的显式任务执行。

## 收尾

- 状态只更新`tasks/progress.md`，本计划不维护第二份owner状态。
- 每个checkpoint只提交本项相关代码、测试和文档；旧能力删除前必须有new actual-IR owner和direct test。
- fresh验证必须确认关键lit/CTest实际执行；文档-only改动至少运行链接、旧术语、职责和格式检查。
- 临时profile数字、workload路径、build目录和单case偶发现象不进入稳定设计或memory。
