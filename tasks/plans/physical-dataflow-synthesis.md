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
  立即物化actual TileRegion IR，之后temporal tile-and-fuse、layout/bufferization、movement、execution structure、
  Instr/order/completion和memory均读各自current IR。
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
  下表的16个checkpoint按线性顺序通过；current production不再有future SSA/buffer/event/schedule的跨stage owner；
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
| search execution materializer | generic/attention的唯一execution construction donor | 物理值、movement、storage或schedule跨stage合同 |
| baseline root materializer | baseline-owned single-root region与actual feedback loop | search路径或Q52最终验收 |

## Q52线性实施顺序

任一项失败留在本项修复，不跳过、不fallback旧路径或另一policy。每项都重复完整工作流程。

下表按**施工依赖**排列。第6--9项已经从最终actual leaf向上闭合直接consumer；它们继续作为后续stage的已验证下游，
不在当前队列重复施工。新发现的logical graph和layout缺口必须按artifact producer/consumer顺序闭合：先在05号边界形成
post-attention canonical Tensor/Linalg graph，再由compact temporal tile-and-fuse产生最终current SSA/use graph，随后才能建立完整
layout domain、PBQP assignment、exact view和bufferization。此前已经完成的output DPS、copy closure、PBQP solver和duplicate/unused
materialization cleanup保留为`current-ir-layout-bufferization`的已有实现资产，但不能代签该work item整体完成。

Baseline gate只在上述三项闭合后运行；search structural producer、production cutover和旧shadow删除仍保持一次原子迁移。最终search
运行顺序固定为：

```text
post-attention Tensor/Linalg
  -> structured-logical-egraph-normalization
  -> structural choice and actual structural Card/TileRegion
  -> compact-temporal-tile-and-fuse
  -> current-ir-layout-bufferization
  -> current-ir-movement-boundary-closure
  -> current-tile-execution-structure
  -> current-instr-schedule-completion
  -> actual-leaf-current-ir-contract
```

Search cutover以前不为同一policy注册第二条compiler pipeline，也不把未被真实上游和直接consumer共同读取的新stage标成完成。
Baseline和search仍是两条policy-owned路径；已闭合downstream stages先替换baseline内部对应片段，search旧链只保留到原子cutover，
不能作为这些新stage的完成证据。Logical e-graph normalization在policy分叉前只运行一次；compact tile-and-fuse只替换shared
transformation mechanics，不增加第三个materializer或policy路径。Structural producer、search production cutover和旧complete/shadow
路径删除是一个原子接口迁移，不能拆成并存的search production checkpoints。

### Current upstream/downstream职责迁移表

| 现有行为/代码 | 最终semantic owner | 迁移动作与时点 | 直接验收 |
| --- | --- | --- | --- |
| post-Linalg reshape/transpose/broadcast/concat及pure structured compute graph | `structured-logical-egraph-normalization`，设计owner 05 | attention semantic recognition后先运行pinned MLIR folds，再对剩余pure component执行bounded access-relation e-graph；candidate attention展开调用同一scoped kernel | planning输入无可严格支配的transform graph；on/off exact语义与downstream representability一致，不要求raw choice identity；budgeted unchanged可复核 |
| memory入口内的function-boundary bufferization | `current-ir-layout-bufferization` | actual leaf只让缺失boundary typed失败；layout stage在movement前接通唯一production transformation | layout→movement→execution-structure→Instr→leaf链中bufferization恰一次；leaf调用次数为0 |
| memory入口内的NCC join rebuild | 第7项`current-instr-schedule-completion` | 第6项只从leaf删除repair；第7项在final worker/order后从current Instr fresh构造 | 删除或扰动completion时第6项typed失败；第7项输出直接通过第6项且leaf不新增join/wait |
| bufferization/layout产生的普通allocation及其IR位置 | `current-ir-layout-bufferization` | layout stage在创建transaction中确定dominance、alias和effect；不把placement另存成plan | actual allocation/view/copy all-and-only，1024/1025/1031 lifetime与MiniMalloc witness一致 |
| software-pipeline rotating allocation/slot SSA | 第8项`current-tile-execution-structure` | 第8项随prefix/steady/tail一次物化；第6项只读 | 1031 tail与multi-wave检查root、slot、reuse obligation和actual offsets |
| `sinkStaticSPMAllocationsToFirstUse` | 无长期owner | 第6项先做caller/test inventory；若fresh production不依赖则删除helper和专用fixture期望；若依赖则停止并把过早创建定位到第8或10项直接producer，禁止把helper移入allocator或新增通用pass | leaf call graph中helper为0；相同current IR的合法性只由实际MiniMalloc决定；任一真实依赖必须有producer修复后才能签发对应owner |
| `buildPrepareInstrForMemoryPlanningPipeline`组合wrapper | 无长期owner | 第6项leaf停止调用后做zero-caller检查并删除wrapper；其bufferization/completion kernel分别留给第10/7项 | source/CMake/test/current-doc中组合wrapper caller为0，单项owner测试仍保留 |
| actual SPM/DDR high-water/headroom | 第6项`actual-leaf-current-ir-contract` | 只从accepted MiniMalloc/DDR offsets重算并作为diagnostic/resource witness | 与accepted offsets/range精确一致；不参与SPM admission、retile或winner猜测 |
| function-boundary bufferization的冗余DDR→DDR publication copy与Instr copy-only TileRegion fallback | `current-ir-layout-bufferization`的output DPS正确性 | output在bufferization前绑定唯一actual destination；只消除因DPS缺失而复制同一logical result的copy；必要copy保留exact SSA/alias/effect witness并交给typed movement；删除Instr临时建Region/staging的fallback | 冗余publication copy为0；必要copy all-and-only；movement closure后未分类`memref.copy`为0；每个observable result唯一publication；Instr创建copy-only TileRegion次数为0 |
| 手写`first / steady / tail`递归loop、`LocalUseDelivery`和无current-IR控制的producer fusion worklist/cache | `compact-temporal-tile-and-fuse` | 使用pinned SCF tiling/fusion生成一个canonical loop nest；fusion直接读取current SSA并只peel必要remainder；窄exact reshape/insert adapter保留 | nested/independent producer只由actual loop/SSA表达；loop body不随wave数复制；旧delivery、通用loop/fusion owner、caller、CMake/test/current-doc为0 |

该表分配的是最终职责，不授权临时复制实现。Actual leaf移除隐藏行为后若baseline暴露上游输入缺口，失败保持在产生该输入的
completion、execution-structure或layout owner；不得把行为放回leaf来维持旧测试。

### 本轮新增整改的施工顺序

1. `structured-logical-egraph-normalization`先在attention semantic recognition之后运行pinned MLIR deterministic folds，再以
   bounded access-relation e-graph收敛剩余pure Tensor/Linalg component；首批覆盖static reshape、transpose、broadcast、canonical
   concat、pure elementwise及受限contraction/reduction anchor。Budget exhaustion保持原verified graph，不产生legality结论。
2. `compact-temporal-tile-and-fuse`删除`LocalUseDelivery`、`DirectNestedValue`、`StoredRegionValue`、
   `ReconstructedRegionValue`和`rewireDirectSSA`等future materialization协议。Region choice只保留group membership；先生成actual
   operations/current SSA，再决定fusion。使用pinned MLIR SCF tiling/fusion API替换parallel/reduction手写三段递归builder；fusion control直接
   检查current producer/use、indexing relation、use count、effect和region boundary。Explicit recompute/replica先创建actual op。
   先生成一个bounded-tail canonical loop nest；只有static Tile/Instr直接要求时才peel最后一个partial iteration并运行
   canonicalization。删除front peel、按wave复制body和无delivery通用fusion worklist/cache。
3. `current-ir-layout-bufferization`消费最终tile/fuse current SSA，补齐value/use layout domain、op tuple constraints、PBQP
   assignment/apply及IndexRelation+PhysicalLayoutRelation exact view；保留已经完成的output DPS、necessary copy closure、PBQP solver和
   duplicate/unused cleanup。只有不兼容edge创建layout materialization，随后bufferization恰一次。
4. `baseline-current-ir-integration-gate`重签完整none链；`search-current-ir-cutover-and-shadow-retirement`再原子切换search并删除旧
   shadow/complete owner；最后依次完成scale inventory和同源LLaMA none/search验收。

上述graph/tile/layout项不改变SPM admission：每个candidate仍先形成上述actual IR，再由唯一MiniMalloc返回accepted offset或typed actual
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
| 10 | `structured-logical-egraph-normalization` | `pending` | 05号normalization唯一拥有post-attention pure Tensor/Linalg graph的bounded access-relation e-graph；先复用pinned MLIR folds，再处理剩余static reshape/transpose/broadcast/canonical concat、pure elementwise及受限contraction/reduction access graph；candidate attention展开调用同一scoped kernel | 不修改scalar/combiner/dtype；不复制compute occurrence；attention/collective/SCF/effect等barrier不越界；deterministic work budget耗尽保持原verified IR；global/scoped输出均由直接StructuredDAG或tile-and-fuse消费；on/off以exact proof保持语义、downstream representability和determinism，不要求raw choice identity | 读AGENTS/progress→读05/06/19及本项矩阵→调研egg、TENSAT、Glenside和成熟tensor graph optimizer并比较full EqSat/有界access e-graph/定向rewrite→查官方及pinned MLIR Linalg/Tensor patterns与API→实现query-local language/e-class analysis/rewrite/extractor/instrumentation/tests→fresh真实规模graph→planning witness→重读设计/完整diff/MLIR复审→更新并提交 |
| 11 | `compact-temporal-tile-and-fuse` | `pending` | 消费第10项canonical Tensor/Linalg graph，将baseline/search共享的temporal loop与producer-fusion mechanics收敛到pinned MLIR SCF tile-and-fuse；Region choice只保留membership，materializer先创建actual operations/current SSA，fusion再直接读取该IR；static Instr要求只由late remainder peeling满足 | 一个traversal一个canonical `scf.for` nest；1024无remainder clone，1025/1031只产生必要main/remainder且不peel first；未融合producer在consumer loop外只物化一次，融合producer只存在于对应loop；multi-use/reduction/contraction/collective/cross-region无exact证明时保持barrier；除actual explicit recompute外iteration domain无重叠；`LocalUseDelivery`、`DirectNestedValue`、`StoredRegionValue`、`ReconstructedRegionValue`、`rewireDirectSSA`、旧三段builder和通用fusion worklist/cache零production残留 | 读AGENTS/progress→读05–07/10–11/19及本项矩阵→调研MLIR SCF/Linalg tiling、producer fusion、reduction tiling、loop peeling和成熟compiler fusion control→查pinned API→先固化current膨胀/重算回归→迁移current-IR loop/fusion/remainder及窄exact adapters并删除旧owner→fresh 1024/1025/1031 structural output和producer-occurrence witness→重读设计/完整diff/MLIR复审→更新并提交 |
| 12 | `current-ir-layout-bufferization` | `pending` | 消费第11项最终current SSA/use graph，唯一拥有完整value/use layout domain、op tuple constraints、query-local exact PBQP assignment/apply、IndexRelation+PhysicalLayoutRelation exact view、function-boundary/region-local bufferization和observable output DPS；保留已实现solver、copy closure和安全cleanup | solver与flat oracle一致；layout-polymorphic op可传播assignment并只为不兼容edge创建materialization；exact view零allocation/copy；soft projection unknown时整组禁用；bufferization恰一次；冗余publication copy为0、必要copy有witness并typed；same-layout/unused为0、shared conversion一个SSA；fresh输出顺序通过movement→execution-structure→Instr→leaf | 读AGENTS/progress→读05–11/19及本项矩阵→调研PBQP、resource-aware projection、One-Shot Bufferization和MLIR layout/view实现→查pinned interfaces/API→补value/use domain、tuple factor、assignment apply与physical-map view并复审已有DPS/copy实现→fresh solver oracle、layout/copy inventory及baseline链→重读设计/完整diff/MLIR复审→更新并提交 |
| 13 | `baseline-current-ir-integration-gate` | `pending` | 对第10--12项与已闭合movement→leaf的baseline唯一路径运行完整none门禁；不首次接线、不重做root构造 | baseline不调用search state/domain/materializer；每region一semantic root；e-graph→tile/fuse→layout→movement→execution-structure→Instr→leaf和actual MiniMalloc均到达；冗余DDR copy为0、必要copy已typed；fresh current FP16 LLaMA none在15分钟内生成package并strict readback/no-card | 读AGENTS/progress→读05–16及本项矩阵→确认current baseline/source/package边界→读相关硬件/ABI事实→调研deterministic baseline集成验证→查pinned API→补完整gate/tests→fresh focused矩阵和none package/no-card→重读设计/MLIR复审→更新并提交 |
| 14 | `search-current-ir-cutover-and-shadow-retirement` | `pending` | spatial/region/temporal choice用第11项唯一current-IR mechanics物化candidate-owned structural Card/TileRegion；search使用第12项同一PBQP assignment作为首proposal并保留完整raw layout alternatives；movement、execution structure和Instr choice在current IR上立即变换并进入leaf；同批重接controller、切换production并删除旧shadow/complete路径 | e-graph不进入candidate key且global只运行一次；PBQP on/off不改变raw域；controller从Accepted current Instr比较resource-aware objective；pre-structural state无future value/buffer/event；Accepted owner不重建；旧type/builder/domain/state/caller/CMake/test/current-doc零production残留；baseline路径不变 | 读AGENTS/progress→读05–17/Q52删除清单与本项矩阵→读NE/CT、overlap和target-profile事实→调研resource-aware cost、current-IR search transaction和原子API migration→查pinned API→实现producer/current-IR枚举/PBQP proposal/final objective/controller接线并同批删除→fresh search全链、engine-cost反例、PBQP on/off、e-graph exact语义/downstream reachability及baseline隔离验证→重读设计/完整diff/MLIR复审→更新并提交 |
| 15 | `scale-regression-and-inventory` | `pending` | 在新actual-IR pipeline上profile并仅保留有证据的e-graph、PBQP、memo、priority、DP和LNS，补齐logical transform、actual fusion、producer occurrence、copy、layout conversion、transfer elimination与Instr只读汇总 | 完整e-node/e-class/match/budget、logical transform before/after、Tile/TileRegion/current SSA edge/producer-occurrence/copy/layout-conversion/buffer/movement/execution-structure/Instr/target inventory；e-graph on/off保持exact语义和downstream reachability但允许canonical choice set变化；PBQP on/off保持raw layout域/accepted set；instrumentation on/off等价；actual MiniMalloc到达 | 读AGENTS/progress→读05/06/Q52及本项矩阵→调研search/equality-saturation scalability和PBQP proposal算法→查pinned MLIR/LLVM→改instrumentation/tests→fresh真实规模验证→重读设计/MLIR复审→更新并提交 |
| 16 | `llama-baseline-search-acceptance` | `pending` | 同一current FP16 LLaMA source顺序运行独立none和search事务 | 每次Release≤15分钟；各自package strict readback/no-card；两条路径互不调用；e-graph实际到达且无budget-dependent nondeterminism；均进入Instr/MiniMalloc/DDR/target；冗余DDR copy和Instr copy-only Region为0，必要copy已typed；search没有静态body倍增 | 读AGENTS/progress→重读05/06/Q52/Q53及本项矩阵→确认current source/tool和runtime/ABI边界→fresh顺序运行→逐项核对设计与MLIR/runtime规范→更新状态并提交 |

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
| generic/attention execution | single/nested/replica/coupled、fanout；rank 3–6，1024/1025/1031 | selected structural recipe不可表达时typed failure | 每execution all-and-only一次，fanout不重复compute | 第11项compact mechanics与第14项search structural cutover |

### 4. Baseline root materializer

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| root isolation与actual feedback | chain/fanout/matmul/attention；rank 3–6，1024/1025/1031，16 Tile | 只capacity rejection推进，其它typed状态停止 | 每region一root；candidate CardModule数等于actual planner数；accepted不重建 | baseline Instr/MiniMalloc/package |

### 5. Current-IR contract reset

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| active docs/code call graph | 01、05–19、Q52 plan/progress、memory；baseline/search两条路径 | 一个旧plan能力尚无actual owner时保留为待迁移，不伪装已删 | 每个字段分类为choice/current fact；active文档无正向shadow合同；删除清单有new owner/test | 第6–13项consumer-first施工与原子cutover的唯一边界 |

### 6. Actual leaf current-IR contract

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| completion-closed canonical unplaced Instr、actual allocation/alias/effect/lifetime | baseline focused current source；aligned/ragged；rank 3–6，1024/1025/1031；16 Tile | capacity、ResourceExhausted、timeout、unsupported、compiler error保持区分 | leaf不改变function boundary或join/wait；MiniMalloc/offset/conflict/owner只来自current IR；读取policy/preparation/shadow state次数为0；Accepted owner不重建 | baseline现有Card→Executable入口直接调用该leaf；第13项完整none与第14项search controller消费同一typed outcome |
| leaf隐藏行为与legacy wrapper retirement | 已闭合Instr正例，以及tensor function boundary、missing completion、preexisting offset负例；rank 3–6，1024/1025/1031 | 输入合同缺失按typed failure停止，不bufferize、不补join、不移动allocation | leaf内bufferization/completion/sink调用均为0；组合prepare wrapper caller为0；SPM/DDR high-water逐一等于accepted range end的重算结果 | completion/execution-structure/tile-fuse/layout各自owner测试和第13项baseline完整门禁 |

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
| post-attention ordinary Tensor/Linalg graph | rank 3–6；1024/1025/1031；chain/diamond/fanout | unsupported/dynamic relation或budget exhaustion保持原verified component | global normalization只运行一次；before/after exact result relation、dtype和scalar region一致；transform减少且compute occurrence不增 | StructuredDAG、exact-demand与spatial domain消费canonical graph |
| reshape/transpose/broadcast/elementwise | adjacent/non-adjacent；1/2/15 uses；projected maps | unknown DPS init、effect、SCF/call/collective/attention barrier不rewrite | standard folds先行；e-graph只处理剩余phase-order conflict；共享access一个SSA；不越barrier | candidate root/materializer和tile/fuse读取rewrite后current uses |
| contraction/reduction anchors | rank-3/4 matmul、batch matmul、generic contraction/reduce；parallel-axis transform | unsupported orientation、K role混合、reduction iterator重排或map不可lower时保持原IR | iteration domain、M/N/K/batch role、ordered reduction iterators、scalar/combiner/init不变 | candidate Linalg-to-Tile与reduction tiling可消费 |
| recovered canonical concat | ordered N-ary/nested segments；1024/1025 tail；transpose/broadcast/pointwise | overlap/gap/partial coverage/dynamic/intermediate external use不恢复Concat | exact prefixes与pieces all-and-only cover；不枚举input排列/子集；extract回标准Tensor IR | exact-demand piece propagation与consumer maps |
| e-graph budget/extraction/instrumentation | tiny congruence oracle与真实HF/LLaMA dense component | deterministic work limit或resource exhaustion不是legality failure | 相同budget产生相同IR/diagnostic；无strict dominance保持原IR；on/off exact语义和downstream reachability一致但不要求raw choice identity；记录e-node/e-class/match/iteration/wall/RSS | 第15项scale inventory与第16项acceptance |

### 11. Compact temporal tile and fuse

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| canonical SCF temporal loop | parallel/reduction/mixed iterator；rank 3–6，1024/1025/1031；aligned/ragged | pinned interface不能表达selected tile或static remainder不能闭合时typed unsupported，不回退旧builder | 1024一个shared body且无remainder；1025/1031只含必要main/remainder，不peel first；loop/body数量不随trip count增长；reduction accumulator与loop-carried state exact | 第12项layout domain直接消费最终current loop/use graph |
| current-IR producer fusion | same-region SSA、single/multi-use、chain/diamond/fanout、reduction/contraction、reshape/insert、collective | result-tile relation、dominance、effect或无重算条件不能证明时保持未融合current producer，不伪造nested plan | 未融合producer在consumer loop外每execution一次；融合producer只位于对应loop；除actual explicit recompute外source iteration tiles无重叠；multi-use/reduction/contraction/collective无证明时不融合 | layout use-binding与exact producer-occurrence inventory |
| source/CMake/caller/test/current-doc retirement | baseline/search、generic/attention、旧only-purpose fixtures | 窄adapter仍无standard replacement时保留并写清输入输出；通用旧owner不得兼容并存 | `LocalUseDelivery`及其枚举值、`rewireDirectSSA`、`first/steady/tail`递归builder、通用fusion worklist/cache和旧production caller为0；只保留exact reshape/insert adapter及standard SCF owner | fresh build、verify-each与第12项直接consumer |

### 12. Current-IR layout and bufferization

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| final tile/fuse current SSA的function boundary与primary/shared/alias/use-local需求 | 1/2/15 uses；Tensor/NTensor/Cx/NCx；rank 3–6，1024/1025/1031 | relation或bufferization不能证明时保留explicit materialization或typed unsupported，不建立alias | function boundary与region-local bufferization只运行一次；shared layout一个SSA definition；alias同storage；allocation dominance/effect正确；无unused conversion；每个logical boundary有current endpoint；不消费PhysicalVersion | fresh layout-resolved endpoint顺序通过movement→execution-structure→Instr→leaf |
| value/use layout domain与op tuple factor | elementwise/convert flexible chain、GEMM/reduce fixed tuple、chain/fanout | tuple unsupported=`NoSolution`；budget/overflow=`Indeterminate`；malformed=`BrokenContract` | 每个current value/use all-and-only一个变量/binding；assignment直接改actual types/uses；只为不兼容edge创建materialization | movement transformation和layout-resolved stage verifier |
| exact logical/physical view | reshape/transpose/broadcast/concat residual；same/different physical map | physical-map、valid/padding、alias或write safety不exact时保留materialization | `Psource(R(i)) == Pdest(i)`时同storage view且零copy/allocation；否则actual movement显式 | movement count与第15项layout-conversion inventory |
| observable output DPS与copy regression | single/multi-result、loop-carried result、chain/fanout；rank 3–6，1024/1025/1031；16 Tile | output无法绑定唯一destination或copy必要性缺少SSA/alias/effect witness时stage失败 | 每个result直接写唯一destination；冗余DDR publication copy为0；必要copy all-and-only且movement后typed；Instr不创建copy-only Region | movement→leaf与第13项baseline gate |
| exact PBQP solver与soft projection | flat oracle R0/R1/R2/residual；NE/Vector/movement/control trade-off；真实rank 3–6 | relevant work/rate/multiplicity unknown时整组soft term禁用；hard infinity/overflow/无解分类保持 | optimal cost/tie与oracle一致；unique conversion descriptor只计一次；projected term与apply后fresh actual analysis一致；不得退回flat instruction cost | 第14项final objective、第15项projected-vs-actual报告 |
| 已实现cleanup回归 | same-layout、unused、1/2/15共享use、intervening alias write、necessary copy | alias/effect不exact时保持独立conversion/copy | same-layout/unused为0；shared一个SSA；source write阻止复用；necessary copy不误删 | current movement/instruction inventory |

### 13. Baseline current-IR integration gate

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| current TensorProgram + fixed baseline rules及共享e-graph/PBQP/layout/temporal mechanics | chain/fanout/matmul/attention；rank 3–6，1024/1025/1031；16 Tile；current FP16 LLaMA block | e-graph budgeted unchanged不是失败；PBQP非Optimal按typed状态停止；只有actual capacity rejection构造smaller temporal attempt；其它typed状态停止 | baseline调用search state/domain/materializer次数为0；global e-graph一次；每attempt一次PBQP；每region一semantic root；e-graph→tile/fuse→layout→movement→execution-structure→Instr→leaf均到达；DDR fallback为0；fresh none package唯一 | 第14项cutover的baseline隔离回归与第16项同源两policy验收 |

### 14. Search current-IR cutover and shadow retirement

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| spatial/region/temporal choice及其current-IR layout/movement/structure/schedule alternatives | chain/fanout/reduction/attention；rank 3–6，1024/1025/1031；16 Tile | rewrite前不可表达为typed failure；mutation后失败销毁candidate owner；tile/fuse→leaf typed outcome原样返回controller | actual TileRegion/loop/SSA all-and-only；每层choice选中后立即变成current IR；pre-structural state无future value/buffer/event；Accepted owner不重建 | tile/fuse→layout→movement→execution-structure→Instr→leaf正式search调用及第15项inventory |
| shared PBQP first proposal与raw layout enumeration | tiny complete layout oracle及真实chain/fanout/attention；Tensor/NTensor/Cx/NCx；1024/1025/1031 | PBQP `Indeterminate`不形成rejection/no-good；`BrokenContract`为compiler error | e-graph不进入layout key；PBQP on/off的raw legal set相同；Optimal assignment恰为首个proposal且只apply一次；solver result不进入candidate key/IR；search可继续其它raw layout | actual movement/memory gate及第15项PBQP work/quality report |
| Accepted actual objective与winner | 相同instruction count但NE/Vector work不同、相同compute work但control或movement不同、mixed engine schedule；rank 3–6，1024/1025/1031 | relevant work/rate/schedule unknown、排序依赖未证NE/CT overlap或overflow时objective typed incomparable，coverage不得称`ComparableBest` | final current Instr逐Tile采集NE/Vector logical ops；两种throughput、movement与instruction-control分别计价；同一profile/enabled terms比较；flat instruction反例选中resource-aware winner；`FirstAccepted`只能是checkpoint，未比较其它accepted candidate时不得称best | retained actual winner、coverage和第15项per-term inventory |
| source/CMake/caller/test/current-doc inventory | generic/attention、none/search、旧fixture | 仍有独有能力没有current-IR owner和direct test时停止cutover，不保留compatibility path | 旧production type/builder/domain/state/caller/CMake/current-doc为0；controller不依赖CompleteCandidateKey；search不fallback旧path/baseline；baseline调用链不变 | fresh build、named/driver、baseline隔离、actual memory/target和第16项验收 |

### 15. Scale regression and inventory

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| mixed DAG/HF/LLaMA | rank 3–6，1024/1025/1031，16 Tile | stage未到达、e-graph budgeted unchanged与typed compiler/resource failure分开 | e-graph component/e-node/e-class/match/iteration/extraction work/wall/RSS、logical transform before/after、TileRegion/structured-execution/op分布、current SSA edge/producer occurrence、copy、layout conversion、actual buffer/movement/eliminated transfer/execution-structure、final Instr total/per-Tile/per-kind、NE/Vector logical work和makespan；非explicit-recompute compute overlap为0 | actual MiniMalloc、package strict readback |
| temporal refinement结构稳定性 | full tile、一次及多次actual capacity feedback；parallel/reduction root；1024/1025/1031 | actual rejection、unsupported和instrumentation failure分类保持 | 相同RegionPlan下refinement只改变loop bounds/tile/remainder；Region数不因copy fallback增加；static body/producer/materialize-layout数量不随wave trip count或无关root倍增 | 第14项actual candidates及第16项search acceptance |
| PBQP与其它search optimization | tiny exhaustive oracle与真实规模profile | timeout/resource/unknown-rate/overflow不伪装exact rejection或comparable winner | exhaustive PBQP on/off保持raw legal、actual accepted set和winner；budgeted run允许访问顺序和best-found变化，但必须保持coverage分类并记录solver work、proposal命中、PBQP-projected-vs-actual per-term delta、conversion/movement下降；其它safe optimization/instrumentation on/off保持result | retained actual winner一次publication |

### 16. LLaMA acceptance

| 输入等价类 | shape/结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| same-source two-policy acceptance | current FP16 LLaMA block；两个独立process、ProgramData和output directory | timeout/OOM/skip/fallback/未进actual planner均失败 | 每次≤15分钟；source identity相同；IR/result不共享；package唯一；冗余DDR→DDR publication copy为0；必要copy在movement closure后typed且Instr无copy-only Region；search temporal feedback不产生静态body倍增 | strict loader、host reference、no-card |

## Q52旧路径删除清单

下列对象不能在终态继续作为production cross-stage protocol。可复用的算法或negative test先迁到actual-IR owner，
然后在同一work item删除旧owner：

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
- donor movement扫描/替换/erase、edge-owned compute重建、first-use layout恢复、无producer action surface和only-purpose fixtures；
  现有test-only/Instr-only full-transfer eliminator的proof与正负测试迁到第9项唯一current-IR production owner后删除旧入口；
- active设计、memory、CMake和tests中把这些对象当作current事实源或supported production contract的内容。

只在一次rewrite内使用的typed choice、`IRMapping`、SSA lookup或query-local event/lifetime graph不在该删除范围，但必须在
IR mutation后失效，不能成为下一stage的事实源。

## Search scalability

只有`search-current-ir-cutover-and-shadow-retirement`完成后才重新解释profile和优化search工作：

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
