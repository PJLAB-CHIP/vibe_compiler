# Physical Dataflow Synthesis 实施计划

状态：Q50.0无策略CardExecutable编译/准入边界、Q54 MLIR infrastructure、Q59 compiler entry transaction、Q50.A
production demand boundary、Q49.P deterministic baseline、Q51.Core search control与Q50.S structured semantic
alternative均已闭合；当前从Q50.B开始让各physical axis逐项接入同一个owner，最后闭合Q51。
算法、IR和长期pipeline contract仍只由
`tasks/06-physical-dataflow-synthesis.md` 拥有；本文件只规定施工依赖、现有代码处置和独立 checkpoint。

本计划的核心约束是：**机制可以独立交付，选择不能独立发生**。每个 Q50 子项只提供合法域、transition、actual
materializer、exact failure 和测试，不保留局部 winner、局部 shortlist、局部 beam 或跨阶段默认选择。Q51 是唯一
physical-dataflow winner owner；Q52只在该正确性基础上改善10–30分钟预算内的anytime质量和搜索吞吐。

## Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  GSPMD 已完成 card-level partition 且 target-independent canonicalization 完成后的 card-local structured
  TensorProgram；current SSA、structured semantics、IndexRelation、effect、type、shape 和 dtype 均可验证，尚未绑定
  Tile、temporal schedule 或 storage action。
- Current stage responsibility:
  typed `none`由Q49.P deterministic feasibility controller从原始TensorProgram完成canonical placement、single-root
  TileRegion、temporal/SPM、representation/movement、single-buffer、order/completion功能合法化；typed `search`先由Q50.S
  semantic-alternative builder从typed SSA物化actual TensorProgram roots，再由唯一query-local physical-dataflow search owner
  选择root并联合展开spatial partition/placement、TileRegion、coupled traversal、temporal tiling、layout/representation、
  explicit movement、buffer、ready/order/worker、NoC/DDR/compute resource timeline和条件式stage pipeline。只有resolved
  assignment物化后的actual IR可以跨越共同compile/verification seam。
- Output IR / files:
  selected wafer.card.module 及其中按 target tile_id 区分的 wafer.tile.module；随后投影为各 Tile actual module。
- Downstream consumer:
  TileRegion-to-Instr conversion、fresh completion reconstruction、fixed-capacity SPM/DDR planning、
  communication/resource/ABI verification、target conversion、package writing 和 runtime launch。
- User-level driver / named pipeline:
  wafer-compile source-to-package pipeline；public optimization policy 只保留 typed `search` 与 `none`。
- Explicit non-goals:
  不重做跨 card GSPMD；不建立第二 search owner、shadow plan、late repair selector 或 workload/name shortcut；
  不把 query-local candidate set、solver state、estimated allocation 或 board 结果写入 IR；不由本计划拥有 Q48 semantic
  superoptimization。
- Completion gate:
  Q50.0先建立共同CardExecutable compile/verification seam；Q54按19号合同收口MLIR infrastructure；Q58/Q56/Q59先闭合
  program data ownership、package data与compile commit；Q50.A修复production exact-demand boundary；Q49.P基于这些
  current事实闭合baseline功能合法化及policy、结构、一次性actual owner和materialization隔离；Q51.Core复用其accepted executable作为初始incumbent，
  建立只管理frontier/evaluation/result的共同control kernel，同批让public `search`只进入新Core并删除旧search控制链；此时没有
  production mechanism，`search`直接返回accepted incumbent。Q50.S与Q50.B–Q50.K逐轴交付新mechanism和独立domain oracle，
  同批删除对应旧实现；Q50.F只加入不构造IR的scoped feasibility analysis；Q51通过full production flat exhaustive runner、complete
  CardExecutable gates和有效fusion闭合新链；
  Q52 在真实 workload 上形成可复现的 10/30 分钟 anytime 质量与吞吐结论；Q60建立产品frontend后，Q53从该入口生成 fresh package、oracle、runner
  并通过 no-card 达到 board-ready，真实 matched 板端 A/B 后才 done。
```

Semantic algorithm alternative若改变structured DAG，必须先物化成一个真实TensorProgram root再进入同一Q51 search。
例如typed SSA先证明普通Attention或functional decode资格，internal builder再物化ordinary、online或partition/merge DAG；
算法族及改变拓扑的window/split参数是共同root transition，不是public独立selector，也不是SPM分配结论。physical placement、
进一步temporal tile、layout和buffer仍由同一后续联合搜索选择。

## 全程不变量

### 唯一选择 owner

Q51 search invocation 是唯一允许比较完整 candidate 并更新 incumbent 的对象。任一 mechanism API 只能返回：

1. 从 current IR 和已关闭的 partial assignment 派生的有限合法域；
2. 一个或多个可回到共同 candidate set 的 typed transition；
3. 对 isolated clone 的 actual transformation；
4. 作用域明确的 accepted facts、`deferred(required coordinates)`、proven exact rejection、indeterminate failure 或 no-good。

mechanism 不得返回“本轴最佳值”，不得按估算删除其它轴仍可能使之变优的候选，不得在失败时自行 retile、改 layout、
spill、减 buffer 或改 schedule。estimate 只能排序或构成已证明的 lower bound；最终 legality 和 cost 来自 actual IR。
`ResourceExhausted`、solver timeout或internal failure属于indeterminate，只消耗work并保留state；只有proven infeasible或确定
unsupported才能成为exact rejection并形成causal no-good。

### Compile seam

共同搜索与 mutable IR 之间只保留两个方向的 seam：

```text
current immutable structured IR
  -> query-local analysis/domain/transition
  -> selected partial or complete assignment
  -> complete assignment的一次性actual materialization与Q50.0 evaluation
  -> accepted actual owner | exact rejection | indeterminate failure
  -> common candidate set
```

- query-local state 只保存不能从 current IR 和已选坐标重算的 typed assignments；ready/live、exact
  demand、lifetime、resource calendar、SPM high-water、lower bound 和 makespan 都绑定一次immutable IR borrow并按typed
  assignment重算，是query-local analysis cache，不进入state identity，也不序列化为output或计划attr；
- actual transformation不修改原source，也不在scratch IR内修复candidate；accepted owner继续下传，rejected owner销毁；
- analysis cache只在一个immutable borrow内存活；Q50.A `IREpoch`只拒绝跨borrow trial，不进入semantic cache key，也不
  代替nested structural snapshot。borrow内的key必须包含target facts和会影响结论的全部typed assignments；改变traversal、
  tile、layout、movement、buffer或schedule后，旧calendar、lifetime、SPM/legality结果全部失效；
- 任意时刻最多一个 live actual clone；host 可并行计算 immutable analysis，但 candidate set insertion 和 tie-break 使用稳定 key；
- regular mapping、reuse signature和coarse resource estimate只能给普通typed transitions排序；不能clone-per-mapping，不能把
  reuse/cost annotation写进候选IR，也不能用function name、JSON或opaque solver payload跨越compile seam；
- 每个complete candidate只通过一次完整CardModule splitting、TileRegion-to-Instr、fresh completion、SPM/DDR、resource、ABI、
  verification和final recost；accepted executable本身进入incumbent，成为winner后不重新物化或编译。

### Search result 等级

- `optimal-certified`：finite domain 已完整覆盖，所有未展开状态均被 exact infeasibility、equivalence、admissible bound 或
  dominance 排除，incumbent 与全局 lower bound 相等；
- `feasible-with-bound`：未展开 completion 仍被完整 exact candidate set（包括可惰性展开的 parent）表示，且
  admissible lower bound 有效；在证明最优前因 work/time budget 中止也可报告有效 gap；
- `budgeted-feasible`：fixed-width/diverse candidate set、LNS-only 或其它启发式已永久丢弃或未表示某些合法
  completion，只证明返回 actual executable 合法，不得宣称全局 bound 或最优。仅有 hard
  work/time budget 不自动降级，等级取决于未展开域是否仍被 exact candidate set 和 bound 完整代表。

所有等级都必须保留已经通过Q50.0 exact verification的Q49.P `none` baseline作为合法incumbent；“没有比baseline更好的
candidate”不等于搜索失败。`none`对声明支持的正常上游输入必须先完成自己的deterministic feasibility legalization；若最终
exact gate仍失败，必须是最小canonical fallback已被typed proof排除、输入确实超出支持域，或明确的compiler/internal failure，
不能把没有进入performance search当作失败，也不能伪造fallback。

## 施工 checkpoint

| 顺序 | Checkpoint | 施工责任 | 完成后才能开始 |
| --- | --- | --- | --- |
| 0 | Q50.0 CardExecutable compilation boundary | 抽出无策略的actual compile/verification seam，不允许lowering修候选 | Q54 |
| 1 | Q54 MLIR infrastructure conformance | typed IR/interface、scoped pass/analysis、named pipeline与rewrite transaction收口 | Q49.P、Q50.A |
| 2 | Q50.A placement-demand repair | 对完整logical shard trial形成typed exact-demand/coverage proof；reduction、broadcast、window/stride、multi-piece、init与tensor transform relation均不被carrier/layout/route反写 | Q49.P；Core完成后供Q50.B消费 |
| 3 | Q49.P baseline functional closure与policy isolation | canonical `none`从正常上游IR完成确定性placement/tiling/SPM合法化且不消费search对象；一root一region；每个closed coordinate只构造一个CardModule并由Q50.0消费，accepted owner直接下传 | Q51.Core、baseline性能复核 |
| 4 | Q51.Core | 从零建立typed assignment与transition apply、deterministic frontier、baseline incumbent、ledger/budget、evaluation/result evidence及finite state-graph model；public `search`同批改接新Core并删除旧search控制链；不含新真实轴或actual materializer | Q50.S、Q50.B |
| 5 | Q50.S structured semantic alternatives | typed proof与actual TensorProgram roots接入共同owner | Q51 closure |
| 6 | Q50.B spatial partition + placement | 完整spatial domain接入共同owner | Q50.C |
| 7 | Q50.C maximal single-root TileRegion | 单root local work的完整actual region materialization | Q50.D |
| 8 | Q50.D coupled traversal / region fusion | 多op boundary、coupled traversal和合法cut进入同一state | Q50.E |
| 9 | Q50.E complete temporal tiling | 全iterator finite breakpoint domain | Q50.F |
| 10 | Q50.F scoped feasibility analysis | immutable facts上的lower bound、deferred坐标和causal taxonomy进入共同candidate set；不构造IR | Q50.G |
| 11 | Q50.G layout / representation | layout、encoding、version与conversion transition | Q50.H |
| 12 | Q50.H explicit movement | local、NoC、DDR、collective、spill/recompute action | Q50.I |
| 13 | Q50.I rotating buffers | rotating slots与multi-buffer lifetime | Q50.J |
| 14 | Q50.J event/resource schedule | ready/order/worker/completion/resource calendars | Q50.K |
| 15 | Q50.K conditional stage pipeline | 在已证明条件下组合跨op wave pipeline | Q51 closure |
| 16 | Q51 closure | 全轴联合正确性、small oracle、完整新source-to-package链、旧实现零残留和exact-domain closure；public routing已由Core完成 | Q52 |
| 17 | Q52 profile-driven anytime / LNS | 10–30分钟内高质量actual winner与可解释trade-off | Q53 |
| 18 | Q53 production | representative workload package/no-card/board A/B | Q48后续工作 |

Q51.Core是Q51新链的首个施工checkpoint，不等于Q51已完成。它直接替换public control owner，因此不建立第二个driver；在
真实轴尚未接入时，`search`只返回Q49.P accepted incumbent。Q50.S与Q50.B–Q50.K必须只在该Core上逐轴交付并同批删除对应旧
实现；每轴独立reference enumerator、全部轴的production flat exhaustive runner、actual fusion、exact-domain证明和完整
new-search source-to-package链满足后，Q51才能整体完成。

### Checkpoint完成边界

Q50.S与Q50.B–Q50.K的顺序是**mechanism/builder 实现可用性**，不是把某个 search 轴提前选定或冻结。
每个Q50 checkpoint只能用显式typed test assignment补齐尚未施工的轴，并证明本轴的domain coverage、transition、
materializer、verifier、exact rejection/deferred和无局部winner。这些test assignment不是production default，新路径不得调用
旧owner暗中补全其它坐标。

“某个局部更贵的choice在后续轴闭合后成为global winner”、fusion/buffering/pipeline协同、complete
CardExecutable cost与全轴actual winner都由Q51 closure验收，不得用旧selector提前签发。表中
“完成后才能开始”只表示下游mechanism可依赖上游typed contract；已选assignment变化后仍必须失效并重新展开所有
受影响轴。

## 现有代码分类与处置

现有文件不能因“结构乱”被批量删除。进入每个 checkpoint 前先把相关实现归到下列类别，并只处理当前轴。

| 类别 | 当前代表实现 | 处置 |
| --- | --- | --- |
| 稳定 downstream trunk | CardModule/TileModule IR、CardModule-to-Tile conversion、TileRegion-to-Instr、Tile memory planning、card resource/runtime-launch verification、retained target output、package/runtime | 保留；所有 policy 复用同一路径；现有`TileMemoryPlanning`/`CardExecutableLowering`仅作实现定位，后者仍需按实际职责收敛名称 |
| 可复用 core facts | `StructuredDAGAnalysis`、target topology、Q50.A immutable-borrow/exact-demand合同、Q50.0 move-only accepted result | 只消费能脱离旧candidate/search owner独立调用的current IR事实和accepted result；work reservation按新ledger重新实现。`TileExecutionCandidate`、schedule-state、metrics、stable ordinal、feedback history、evaluator和proposal order不进入Core |
| 可提取 mechanism 素材 | `BidirectionalTiling`、`CompleteTraversal`、placement option、movement/collective lowering、selected-buffer materialization、ready-order/worker/completion verifier | Q50.S与Q50.B–Q50.K先定义终态typed query/transition/apply合同，再迁入仍正确的局部算法、proof、verifier和negative case；不迁移旧API、调用顺序或winner行为 |
| Q51.Core同批删除的active search monolith | 当前executable-synthesis中的`deriveShortlist`、coordinate sweep、candidate family、mixed evaluator/materializer、allocation/buffer feedback、beam、accepted cohort、schedule-plan selector和winner rematerialization | 不建立adapter或proposal bridge；Core建立新control owner、改接public caller并删除整条旧控制/选择调用闭包及其专属统计和测试 |
| Q51.Core同批删除的旧bounded/rank search | `RankCandidateSearch`、旧structured candidate generation/evaluation/selection及rank-era candidate set | 不保留路径或文件；若其中有新合同仍需要的独立proof、verifier或test witness，先移入对应稳定owner，其余随Core删除，不恢复固定beam/cap或rank-local winner |
| compatibility lowering | `StructuredDAGEdgeStrategyPlan`、dense rectangle fragment、现有 local/peer lowering | 在新 representation/movement IR 可完整消费 exact demand 前保留；Q50.G/H 逐项替换，不提前删除 |
| late selector/fixup | layout/movement optimization、ready-order、worker placement、buffer/allocation feedback 中会重新做选择的部分 | 先改成 verifier/materializer 或 typed transition mechanism，再按轴删除选择责任 |

独立pass、analysis、lowering、proof或negative witness若仍属于新合同，必须由新owner承接并形成actual-IR witness和fresh test；
旧search控制链本身没有行为承接或输出对照要求，随Q51.Core直接删除。仍依附于downstream/baseline的旧轴内selector或fixup，
在对应Q50机制落地时同批删除选择责任，不延后汇总。

### 新链分轴施工

Q50.S与Q50.B–K分别直接实现新assignment上的semantic、spatial、region/fusion、temporal/feasibility、representation/movement、buffer和
event/resource职责；不存在“先让旧owner返回新类型”的中间合同。Q51.Core先删除旧control branch，使后续public `search`和
每批测试只能经过新Core。每个Q50变更把本轴mechanism接入新assignment时，同批删除仍依附于baseline/downstream文件中的旧
selector、repair、stats和专属测试；确有独立价值的局部算法、proof或test witness移动到新owner后按新合同验证，其余不迁移。

## Q50.0：CardExecutable Compilation Boundary

先从当前综合大流程抽出唯一、无策略的CardExecutable编译/准入函数：输入已经选择且物化的CardModule，依次执行
Tile module splitting、TileRegion-to-Instr、fresh completion reconstruction、fixed-capacity SPM/DDR planning、
communication/resource/runtime-launch verification，输出accepted CardExecutable、proven exact rejection或indeterminate failure。
target ABI preparation/lowering/translation属于下游真正保留的target output，不在candidate/CardExecutable准入时提前执行。

返回taxonomy必须完整区分`accepted CardExecutable`、`proven exact rejection`与`indeterminate failure`；allocator
`ResourceExhausted`、timeout或内部错误属于最后一类，不能伪装成candidate非法。

这个边界不得枚举候选、修改选择、在lowering中retile/spill/rebuffer，也不得把失败降级成performance Unknown。Q49.P与Q51
必须共用它；定向测试要证明同一CardModule得到相同accepted digest或相同rejection，且没有第二条兼容编译路径。

实现结论：actual compile/verification seam只接收owned、已选择的CardModule和typed buffering assignment，返回
accepted、proven exact rejection或indeterminate三态结果；baseline与search materialization都调用该入口。Tile
memory planning保留SPM failure kind，只有capacity overflow与unsupported lifetime等可验证失败进入exact rejection，未分类
allocator/internal failure保持indeterminate，调用方不得据此裁剪候选或启动repair。

## Q49.P：Deterministic Baseline功能闭环、Policy与结构隔离

Q49.P不是对既有baseline做性能润色，也不是把`none`缩成fixed-assignment verifier；它要从current正常上游输入同时补齐
功能合法化、policy、结构、probe、causal diagnosis和materialization
边界。改写后必须从未选placement/temporal/layout/buffer的正常上游IR产生可执行结果，并用本软件产物重新证明result/digest与
no-card，不得把历史package当成新调用链的证明。

2026-08-16 current实现已经具备三个可复用事实：`none`在shortlist/candidate family前提前返回；
TileRegion SPM capacity evaluation已经在scratch-owned真实region上运行；accepted路径只调用一次Q50.0完整
CardExecutable compilation。它仍不满足Q49.P：baseline construction继续消费search-oriented placement domain/evaluator、
`TileExecutionCandidate`及其stable ordinal/proposal mechanics；同Tile的多个独立structured root可由共同group materialization
进入同一TileRegion；temporal refinement前后会重复完整CardModule materialization，tile-scoped路径仍形成其它Tile no-work
wrapper；`RequiresFunctionScope`被计数后跳过；capacity attribution会按typed DAG edge加相同type/shape扩展歧义producer，并把
同region其余root一并纳入refinement；baseline还写入candidate proposal/fusion/layout/buffer类统计。
`actual_fused_edges=0`和“没有进入search loop”都不能证明这些耦合已经消失。

2026-08-17 follow-up review又确认了四个仍属于Q49.P、不能转交Q52的结构问题：

1. 所谓policy-free baseline仍调用`deriveStructuredDAGNodePlacementOptions`，先展开每个node的全部合法
   iterator-axis × connected-rectangle placement options，再用递归constraint solve只取一个canonical assignment；这是把旧
   search domain换了入口名，baseline必须改为从typed axis/topology facts直接推导当前canonical coordinate，只在typed exact
   rejection后推进下一个必要coordinate，不得materialize完整Q50.B域。
2. controller曾在同一coordinate上依次materialize root shard、Tile entry和完整CardModule；前两份actual IR通过后被销毁，
   最终链又重建相同语义。Q50.0本身已经返回带current relation attribution的typed SPM rejection，因此baseline必须让每个
   closed coordinate只构造一次完整CardModule并由Q50.0直接消费；accepted owner继续下传，exact SPM rejection销毁该owner并
   驱动下一次确定性temporal refinement，不存在probe后重建。
3. Q50.0已经返回accepted executable后，baseline仍构造`StaticSchedulePlan`并运行duration estimation，结果既不影响baseline
   output也不进入Q51 incumbent；该shadow plan可让合法executable因旧cost失败而失败，必须从baseline路径删除。
4. common compile seam在每次trial/compile中无条件把全部Tile dataflow IR打印成字符串，即使production caller最终丢弃trace；
   IR inspection必须是显式请求且只在最终accepted output上执行，不能成为baseline或candidate admission的固定成本。

同一review还确认baseline代码、100字段的search statistics和trace-bearing synthesis result仍共同定义在旧search
monolith/header中。Q49.P完成前必须把baseline的窄assignment、typed outcome和必要work ledger移到不依赖旧search对象的current
owner；Q51.Core随后删除旧controller时不能再次迁移或适配baseline。

### Current实现向新baseline合同的融合方案（含`2f2e8de0`代码复核）

2026-08-17对`2f2e8de0`及current HEAD的Q49.P生产调用链逐项复核；该提交之后没有Q49.P生产源码修正。这里不建立
“DS修复线”和“新baseline线”两套工作：current代码只作为迁移输入，每项能力必须原位吸收进上文定义的single-coordinate、
single-root deterministic baseline，或从baseline调用闭包删除。融合取舍如下：

| current资产 | 融合到新baseline的方式 | 终态owner |
|---|---|---|
| typed region/function SPM probe与`RequiresFunctionScope` routing | 保留其中真实TileRegion-to-Instr、memory-planning pipeline、SPM checker和typed evidence转换能力；baseline删除probe materialization与scope escalation控制，直接消费Q50.0对actual CardModule的同一结论。未来search若能从immutable facts完成纯query则只返回typed facts；一旦构造actual IR，该owner必须继续进入完整evaluation，不能probe后重建。 | P1/P5的一次性actual evaluation；Q50.F不得恢复discard-and-rebuild。 |
| `StorageRootMemo` | 原位保留为同IR epoch的query-local派生缓存；不进入assignment、identity或跨mutation cache。 | `StructuredBufferRelations`局部query，P2。 |
| 无Card/Tile shell的per-Tile scoped入口 | `lowerRootShards`/`lowerTileEntries`只为discarded probe服务，退出baseline及current公共API。保留一次read-only source/target与root/support关系分析；actual CardModule apply按resolved root和Tile shard直接构造最终region，不为取得verdict生成另一份临时IR。 | P3/P7共享分析与一次性actual materialization；未来Q50.C复用apply。 |
| post-hoc `splitStructuredRootBoundaries`和DDR spill/reload实现 | 保留其中typed DDR boundary、SSA/resource移动和transaction proof，迁入按root直接构造region的materializer；baseline终态不先group多root再靠repair split。current splitter只作能力donor，完成后退出baseline调用链。 | P4 singleton-region apply与跨root canonical DDR carrier。 |
| lazy exact-demand option-pair query | Q50.A exact single-pair query保留；option domain、compatibility map、propagation和recursive solve整体从baseline删除。未来Q50.B在new Core上用自己的惰性domain重建，不继承旧helper。 | P6 single-coordinate legality；Q50.B future search mechanism。 |
| SPM failure converter与raw certificate | 合并成一条先完整复制raw demand集合、再用complete current relations补owner的typed conversion；有无relations只影响attribution，不影响certificate内容。 | P1/P5/P8唯一actual gate evidence边界。 |
| reduction temporal materialization与浮点reassociation policy | 接入baseline的通用legal-breakpoint/workset推导，用于确定性capacity fallback；不把旧search proposal/test带入baseline。reduction spatial factor、partial ownership和merge仍只归Q50.B。 | Q49.P functional fallback与Q50.E共享机制；Q50.B spatial。 |
| search statistics、`allocationFeedbackTransitions`、shadow schedule和默认IR trace | 不迁移；baseline窄ledger、move-only accepted result与optional inspection另立current owner后删除这些依赖。 | P6/P7；旧search对象随后由Q51.Core删除。 |

因此融合后的控制流是：先对每个root直接构造唯一canonical spatial coordinate与完整temporal vector；这个coordinate可以
覆盖多个甚至全部Tiles。source session只建立immutable root/support/relation事实，随后为该coordinate一次性构造actual
CardModule并交给Q50.0。Q50.0 accepted owner直接成为baseline结果；其typed exact SPM rejection才沿direct witness推进下一
temporal coordinate。memo、DDR boundary和temporal materialization不得由旧option/CSP controller、discarded probe、
whole-Tile trial或search result/statistics拥有。

融合时必须修复的current阻塞项如下；“单测能通过”或diagnostic显示零不能替代这些调用链和current-IR合同：

| 等级 | current代码事实 | 影响与收口位置 |
|---|---|---|
| correctness / policy | `derivePolicyFreeBaseline`仍调用`deriveStructuredDAGNodePlacementOptions`，随后由`deriveCanonicalBaselinePlacements`建立每node option domain、反复propagate并递归`solve`；lazy pair query只延迟Presburger调用，没有消除axis×rectangle域或CSP。最终diagnostic却固定打印`placement_enumeration=0`。 | baseline仍是search，且可观察ledger与真实work矛盾；LLaMA慢case的全option/CSP与generic Presburger双重热路径仍在。由P6删除调用闭包并改成真实窄ledger。 |
| functional coordinate | `getNodeSpatialAxes`在没有映射到result的parallel iterator时返回空，`deriveNodePlacementDomain`随即不给任何placement；因此纯reduction/标量结果即使合法地选择all-factor=1和一个canonical Tile，也会在temporal fallback之前失败。 | Q49.P的窄assignment必须显式表达该root专用的unpartitioned单参与Tile coordinate，不能伪造shard axis；这是无parallel轴时的退化，不是baseline全局只用一个Tile。reduction spatial factor>1、partial ownership和merge仍由Q50.B实现。 |
| correctness / witness | `remapStructuredBufferRelations`会省略没有mapping的entry，但`TileRegionEvaluationScope::remapAfterBodySwap`只检查剩余value是否live，不比较应保留relation数；final `planTileMemory`又在memory-planning preparation后调用`retainCurrentStructuredBufferRelations`静默删除被rewrite的relation。function probe则对同类stale relation直接AnalysisFailure，且现有function-scope测试传入的是空relations。 | probe与final尚未证明消费相同的owner/witness集合；missing relation既可能被静默忽略，也可能只让probe失败。P1必须让conversion、body swap、bufferization和canonicalization使用同一显式remap/completeness合同，并用非空result/operand/output relations比较probe/final evidence。 |
| correctness / split relation | structured-root split把other-root consumer的SPM operand改写成新`reloadResult`，但`splitRegionAfterPrefix`只retarget wrapper argument/result、spill和shared-DDR relation；原`operandBuffers`仍可指向已经移入prefix且仍然live的旧SPM value。`checkStructuredBufferRelationsCurrent`因此会通过，却不能证明relation仍描述suffix consumer实际使用的buffer。 | direct causal attribution可能缺失或指向错误root。P4必须在split事务中显式retarget/rebuild consumer relation，并验证relation语义而不只验证SSA liveness。 |
| correctness / attribution | capacity loop在demand没有direct relation时仍以“region恰有一个root”为由写入`operandDemandNode`，违反P5的no-witness-is-indeterminate合同；同时`sawAttributedDemand`只在写入fallback前更新，故全部demand都走fallback时反而仍报“no demand evidence”，mixed direct/fallback时却接受推测值。function-scope probe覆盖整个Tile Func时，同一fallback还可能把其它region的unmatched allocation归到发起escalation的root。 | 当前行为既不完全fail closed，也不稳定地接受同一种证据。P5删除root猜测；每个accepted causal coordinate必须来自同次planner certificate与current relation，缺失即typed indeterminate。 |
| correctness / structure | splitter和controller只拒绝`resultRoots.size() > 1`，zero-root compute region会通过“exactly one root”合同；split循环还有与IR/target无关的固定256轮上限，退出时没有独立postcondition证明每个region恰一root。现有`NoneProbesIndependentStructuredOwnersInOneActualRegion`只检查没有multi-root diagnostic，不读取实际region数、root cardinality或DDR boundary。 | P4尚未闭合。改为以“剩余excess roots”单调下降的worklist终止，最终逐region验证`rootCount == 1`；测试直接检查actual IR、DDR store/reload和relation归属。 |
| functional evidence | `planTileMemory`无`materializationRelations`分支只复制SPM failure scalar，漏掉`largestDemands`、`capacityConflictDemands`和`individuallyOversizedDemands`；对应fresh unit已失败。 | raw planner certificate在合法入口丢失，P8按同一converter先完整复制evidence，再可选补structured owner。 |
| scope / work | controller第一次trial先构造完整CardModule；overflow后per-Tile helper仍重新clone/prepare完整TensorProgram并物化该Tile的全部roots，fit后再回到完整CardModule。每个root在每个参与Tile上的shard trial并没有从一开始走per-root materializer。 | P3只完成“无card-shaped wrapper”，P7的最窄trial和完整CardModule一次仍未完成；per-Tile preparation可复用immutable validated facts，但probe输入必须投影到当前root、该参与Tile shard及必要closure。 |
| output / ownership | accepted Q50.0 result后仍构造`StaticSchedulePlan`和duration estimate且不消费estimate；compile seam无条件`captureTileIR`，result/tests把`tileDataflowIRTrace`当普通baseline合同；temporal refinement还写入search bag中的`allocationFeedbackTransitions`。 | 合法executable仍会受shadow cost失败影响并承担默认IR打印，baseline也未脱离旧search result/statistics owner。由P6–P7删除。 |

现有测试还有三类假阳性必须同步清理：baseline测试把固定字符串`placement_enumeration=0`当隔离证明，却没有检查transitive
call graph或真实work；root结构测试只看diagnostic缺失；function-scope probe测试用空relations，无法覆盖relation replacement、
completeness和evidence一致性。该次2026-08-17 review先用有界小图定位这些缺口；它当时不运行重型LLaMA的决定不是current
完成合同。下方fresh模型复核已经证明小图未覆盖multi-producer tensor input materialization，因此current Q49.P在定向证明闭合后还
必须执行一轮fresh FP16 LLaMA `optimization-none` source-to-package/no-card。

```text
Pipeline position:
- Upstream IR / input:
  verified card-local TensorProgram、available Tile、immutable target facts及可从current SSA/structured semantics派生的policy-free
  relation机制；尚未创建search state/candidate，也没有预先计算的placement-specific demand或selected spatial、TileRegion
  grouping、temporal、layout、route、buffer choice。Q50.A exact demand由controller对每次closed spatial trial现场查询。
- Current stage responsibility:
  由独立deterministic feasibility controller从正常上游IR自行构造canonical spatial assignment、每root独立TileRegion、显式
  DDR boundary、target-required representation/movement、single-buffer assignment、order/completion和完整temporal vector；
  每个closed coordinate只构造一个actual CardModule并立即交给Q50.0；只有Q50.0的direct typed SPM witness允许沿有限canonical
  temporal fallback前进，第一个accepted executable直接保留并下传。该
  合法化不评分或比较性能，但必须覆盖声明支持的baseline域，不能因最大初始tile/placement
  不合法或没有search candidate而停止。
- Output IR / files:
  accepted coordinate的完整baseline CardModule经Q50.0消费后形成CardExecutable；Q59 transaction随后提交
  verified ExecutablePackage。rejected coordinate只留下复制出的typed witness，不留下IR owner或shadow plan；普通编译不生成printed-IR
  snapshot，显式inspection只在最终accepted output上按请求采集。
- Downstream consumer:
  `none`直接发布accepted executable/package；Q51.Core复用同一个accepted executable及其actual cost/digest作为初始
  incumbent，不用search carrier重新构造baseline。
- User-level driver / named pipeline:
  wafer-compile source-to-package pipeline的typed `none`；typed `search`只调用该controller取得初始incumbent，之后才进入
  candidate-selection owner。
- Explicit non-goals:
  不增加第二套Card/Tile/Instr lowering或SPM planner；不在baseline比较placement质量或搜索group/fusion/multi-buffer/可选
  route；不把lower bound或局部分析冒充card-level transport/resource/ABI证明；不改变Q51的性能候选域。这里的非目标不排除
  baseline为走通程序而确定canonical placement、temporal tile、representation/movement、buffer、order和completion。
- Done criteria:
  baseline不依赖search state/candidate、完整placement-option domain/ranking evaluator、proposal order/group materializer，也不
  调用domain propagation、recursive CSP/backtracking或其它option-list assignment solver；canonical placement从typed
  structured/topology facts直接推导，始终只有一个live coordinate，不能先展开全部connected rectangle/axis options再取第一个；每个
  baseline TileRegion恰有一个structured compute root和由exact demand证明必要的non-root tensor transforms；跨root shaped dependency显式DDR；
  exact rejection携带来自同一actual owner的direct typed causal witness；每个closed coordinate的完整CardModule materialization与
  CardExecutable compilation严格一一对应，accepted owner不重建；同一coordinate的全部Tile root execution domain由一次
  grouped exact-demand query处理，relation按operation/result/operand只建立一次；structured producer形成停止边界，final region
  只一次性物化typed recipe证明需要的operation和carrier endpoint，16个独立Tile构造bounded并发并按Tile ID稳定归并；accepted后不构造`StaticSchedulePlan`、duration estimate或其它不被output消费的shadow result，
  production调用不打印/保存Tile IR trace；初始完整tile超SPM的正例会重新推导完整workset并确定性缩到合法tile后通过package/no-card，
  浮点reduction轴自由重结合后每个demand都可缩到最小合法vector，故「最小合法tile超限」的浮点capacity负例在当前lowering下不可构造，
  typed capacity/unsupported terminal保留为fail-closed防御出口；baseline使用独立窄work ledger，不持有或清零旧search统计来
  证明隔离；fresh
  source-to-package、oracle/no-card、digest、计数和结构正负测试通过。
```

`none`不得构造Q51 state、candidate set、candidate family或全图performance Cartesian组合，也不得通过调用search-oriented
domain/ranking evaluator后只取第一个结果来伪装canonical construction。它只复用typed iterator/topology事实、单coordinate
legality/materialization机制和Q50.A logical demand/coverage query，不复用“生成全部合法placement options”的domain API；对baseline
合同内的mandatory coordinates执行有限、完整、确定且不被beam/cap/time budget截断的
feasibility resolution；按semantic全序逐个关闭coordinate，每个coordinate只在关闭后构造actual CardModule并由Q50.0准入，
第一个accepted executable即完成。不能复制第二套placement语义；trial只是typed coordinate，不提前构造actual IR。每个closed coordinate
只构造一份完整CardModule并由Q50.0消费；accepted owner继续下传，typed exact SPM rejection销毁该owner并推进下一个
temporal breakpoint。不得在同一coordinate上先构造root/Tile probe再重建CardModule。

controller交给共同materializer的是窄immutable resolved baseline assignment。它是上述feasibility resolution的输出而非入口
前置条件，只含per-root placement、显式singleton region boundary、完整temporal vector和已经确定的canonical
representation/movement/buffer/order/completion事实，不含evaluation/score、stable ordinal、transition/failure history或
controller flags。materializer只apply这些已选事实，不得根据同Tile root集合自行group，也不得补search default。

### 功能合法化与SPM收缩

baseline以每个root的完整local iterator extent作为第一个temporal trial，并从structured iterator、indexing map、tail、target
vector/alignment、source numeric/reassociation、最小合法粒度形成有限breakpoint lattice；浮点reduction iterator与parallel
iterator共享同一breakpoint lattice（自由重结合，typed comparator验收，不消费fast-math flag），整数no-wrap/overflow语义
保持barrier。每个trial必须按
当前tile重新推导全部operand slice、stride/dilation halo、result/init/accumulator、temporary、materializing copy、movement
staging、alignment/bank和actual lifetime，并由消费actual CardModule的Q50.0判断；不能只缩output shape、沿用上一trial的
workset/lifetime或先建立一份最终不会下传的probe IR。

proven SPM overflow只允许controller沿direct typed witness对应的合法维度进入semantic全序中的下一组更小breakpoint；多轴
shape必须一直覆盖到target允许的最小合法vector。第一个fit即停止，不为性能比较其它fit。若最小vector以及canonical
placement/representation/movement/single-buffer fallback均被exact证明不可行，返回typed capacity/unsupported；indeterminate是
compiler/internal failure，不得冒充输入unsupported。对声明支持且存在baseline-domain completion的输入，`none`必须形成完整
CardExecutable/package，缺少performance search不能成为失败原因。

```text
TensorProgram
-> deterministic baseline feasibility resolution
-> resolved placement / singleton regions / temporal / representation / movement / buffer / order
-> deterministic CardModule construction
-> Tile module splitting
-> TileRegion-to-Instr conversion
-> Tile memory planning
-> card resource and runtime-launch verification
-> retained target ABI / LLVM output
-> ExecutablePackage writing
```

`OptimizationConfig::none()` 使用独立 deterministic feasibility controller：最大合法非空 Tile participation；每个structured compute
root独立TileRegion；root间显式DDR boundary；零fusion；buffer count为1。participant count、Tile group、iterator/factor和独立
root顺序按typed structured/relation/topology facts的完整semantic key决定，不使用pointer、walk ordinal、`stableOrdinal`或
search proposal order。controller沿不截断的有限canonical fallback逐个形成closed coordinate；每个coordinate先完成grouped
exact-demand与carrier coverage验证，再只构造一个actual CardModule交给Q50.0，第一个accepted executable即完成。它不计算score、
不维护incumbent/candidate family，也不保留用于质量比较的备选方案。

participant group按root定义，只包含该root的非空执行Tile；accepted full CardModule仍必须拥有target要求的all-and-only完整
Tile domain。未参与某个root的Tile只在最终完整CardModule中按IR合同存在；baseline不为局部结论创建额外no-work
Tile/Func wrapper。

同一Tile可以按上述顺序承载多个region，但一个baseline TileRegion最多有一个structured compute root；shape/index/view、
target-local materialization等没有独立structured DAG identity的op才是non-root support closure。显式Fill、Reduce或DPS init
producer只要映回另一个structured DAG node就仍算第二root；一个root lower成多个compute/instruction op则仍算同一root。
root cardinality由同次materialization relation证明。多个独立root共用region即使edge action全为RegionCut，也会共同占用
SPM/lifetime/lowering scope，属于search的region-grouping选择，baseline不得使用。

Q50.A对reduction、broadcast、affine window及stride/dilation、strided slice/view和multi-piece set给出`satisfied`后，baseline
必须由policy-free canonical correctness carrier把同一exact demand有限分解并形成all-and-only DDR/必要peer movement；dense-only
fragment或单descriptor表达失败不能改判logical placement，只说明该canonical carrier尚未闭合。Q49.P负责走通这一条确定性
correctness carrier，Q50.G/H随后扩展可搜索的representation/movement完整选择域；二者都消费同一Q50.A proof，不重建或压缩
logical demand。

Q49.P直接复用Q50.0的TileRegion-to-Instr、lifetime和SPM capacity边界。capacity witness可以是冲突集合，但其中all-and-only
actual allocation/lifetime owner必须经同一份actual CardModule的materialization relation关联到root result、operand demand和
temporal assignment；unsupported witness必须命名无法表达的typed lifetime/call relation。不按type/shape、位置字符串、region内
“可能相关”的其它root、pointer identity或diagnostic字符串猜测。只有Q50.0返回的proven exact SPM failure允许baseline
controller沿确定性fallback lattice前进；其它exact rejection不可由baseline私自repair，indeterminate立即传播。

`none`与`search`共享immutable structured/relation/target facts、policy-free single-root TileRegion materializer和
actual Card/Tile/Instr、completion、SPM/DDR、verification、package机制；不共享search state/candidate/evaluator、group
boundary、proposal order、score或feedback。Q49.P先产出并准入baseline，Q51只把accepted executable/cost作为incumbent，
不得通过search representation重建同一baseline。Q49.P施工时把resolved baseline assignment的single-root apply落在稳定
PhysicalDataflow/Conversion边界；Q50.B/C随后扩展完整search domain与single-root mechanism。Q50.F只做pure analysis；
partial assignment不构造IR，complete assignment的actual CardModule一经构造就由Q50.0一次消费。
共享的是已关闭coordinate的
mechanism，不是生成或选择coordinate的控制：任何option-domain construction、constraint propagation、recursive CSP、backtracking、
candidate/evaluator或winner协议都不得进入`none`的transitive call graph。baseline默认值不限制Q51域。

历史official HF prefill、functional decode、LLaMA block 的 source、oracle、package 和 no-card runner只证明旧入口mechanics；
它们不能单独把Q49.P标为`done`。Q49.P迭代期经Q59 compile transaction使用fresh有界小图、overfull-to-fit、五类relation及轻量
source-to-package/no-card输入定位correctness/work缺口；算法、结构和ledger闭合后只执行一轮fresh FP16 LLaMA
`optimization-none` source-to-package/no-card作为真实baseline门禁。该运行不得进入search，也不承担性能比较。Q51完整new-search
链闭合前不执行LLaMA `search`或反复执行重型baseline；Q52才执行LLaMA search的bounded scalability profile，Q53签发正式
模型package/oracle/no-card与board-ready证据。
定向结构测试必须覆盖：同Tile多个独立root形成多个region；显式structured producer不能伪装成support closure且一个root的
lowered multi-op不会误判成多root；call或unsupported lifetime提升到最近合法scope；equal-shape fanin只按direct witness
refinement；search state/candidate、search-oriented domain/ranking
evaluator和grouping调用计数为零，baseline work不写入candidate proposal统计。它不把历史耗时
写成长期阈值，也不得借“统一入口”让`none`再进入search feedback。

定向功能测试还必须从没有selected assignment的正常TensorProgram进入：至少覆盖初始完整tile因operand/halo/temporary/
alignment/lifetime真实占用而溢出、经过多个合法breakpoint后fit并完成source-to-package/no-card；覆盖multi-axis、tail和最小
合法粒度。浮点reduction轴自由重结合后每个demand都可沿其breakpoint lattice缩到最小合法vector；没有parallel result轴的
纯reduction必须先形成all-factor=1、单参与Tile的unpartitioned canonical coordinate，再验证reduction temporal fallback；这只是
该root没有parallel轴时的退化，不是baseline全局Tile数。不能把current placement domain表达不了当成source unsupported。
原「最小合法tile超限」反例既未进入该合法coordinate，也不能作为
capacity terminal的当前证明；typed capacity/unsupported terminal仍保留为fail-closed防御出口。测试同时断言每次trial重新计算
workset/lifetime、没有beam/cap/budget截断fallback，且这些trial不进入candidate统计。

## Q50.A：Placement-Demand Boundary Repair

“给定placement”只表示调用者正在检查的一次closed logical shard trial，不表示baseline或search入口已经拥有selected
assignment。Q49.P按canonical feasibility顺序产生trial，Q50.B在共同state中产生spatial transition；Q50.A只回答当前trial的
logical dependency demand是否被精确表达和覆盖，不生成placement、不选择winner，也不物化physical carrier。

```text
Pipeline position:
- Upstream IR / input:
  verified card-local structured TensorProgram、current SSA/structured DAG、从typed op/interface/indexing semantics派生的
  dependency relation，以及producer/consumer当前trial的完整logical iteration/result/ownership domain、reduction/replication
  role和IR epoch；尚未选择layout、encoding、movement、route、buffer或schedule。
- Current stage responsibility:
  对每条data/init dependency及其中间pure tensor transform relation形成或组合exact IndexRelation；将consumer当前完整logical
  execution domain取image得到all-and-only producer demand，与producer shard ownership求交并形成typed coverage proof；
  区分satisfied、proven logical infeasible、unsupported semantic relation和indeterminate/compiler failure。
- Output IR / files:
  不修改IR、不产生文件；输出仅在当前IR epoch有效的query-local typed result，包含dependency/relation identity、consumer
  logical domain、producer logical demand、每个producer ownership intersection、dependency/reduction/init role及coverage witness。
- Downstream consumer:
  Q49.P deterministic feasibility controller与Q50.B spatial transition只消费logical outcome；Q49.P canonical correctness
  carrier及Q50.G/H representation/movement materializer在相应坐标关闭后消费`satisfied` exact demand；Q50.0仍是完整
  CardExecutable准入边界。
- User-level driver / named pipeline:
  wafer-compile source-to-package pipeline中的typed `none`与`search`共同physical-dataflow synthesis路径；不提供独立public pass、
  selector或磁盘格式。
- Explicit non-goals:
  不枚举、排序或选择placement；不决定TileRegion/fusion、temporal tile、layout/encoding、local/remote/DDR/peer action、route、
  bytes、fragment、buffer、send/recv、schedule或cost；不把physical carrier能力当logical legality；不保存跨IR mutation的side table。
- Done criteria:
  policy-free placement-assignment与typed outcome API脱离candidate/schedule/evaluator；production baseline和spatial transition只在
  `proven logical infeasible`时删除当前trial，unsupported/indeterminate不形成placement no-good；reduction、broadcast、
  affine window及stride/dilation、strided slice/view、multi-piece、multi-result、DPS init producer与pure tensor input-chain relation
  的exact image/coverage正负测试通过；carrier失败不改变logical结果；cache key/invalidation、production接线和fresh
  source-to-package witness闭合。
```

### Logical domain、dependency与ownership合同

Q50.A输入的是完整logical domain，不得再从`shardDimension + participant count`恢复balanced一维矩形。producer和consumer
trial必须显式包含all-iterator spatial factors、parallel/reduction role、remainder/tail、logical shard-to-Tile binding，以及
unique partition、explicit replication或partial-reduction contribution等ownership语义。Q50.B负责生成并验证这些placement
assignment；Q50.A在本checkpoint先用显式typed test assignment闭合query合同，使后续multi-axis、非整除、非矩形、非对称和
非连通placement不被旧单轴接口截断。

每条structured dependency按current SSA use、producer result、consumer operand、DPS operand role和support op semantics形成
typed descriptor：

- 普通DPS/data input对consumer已选完整iteration domain应用operand indexing relation；
- reduction保留当前partial contribution domain、combiner/numeric约束与merge role，不能只从result shard推回完整K域，也不能
  把多个partial owner当作可互换replica；
- 显式structured Fill或其它DPS init producer仍是独立DAG root，其init/update dependency必须形成exact demand，不能因consumer
  typed lowering稍后会处理init就从计划中消失；
- 没有独立structured root的view/reshape/slice/pad等pure tensor input graph按typed semantics组合relation；多operand support graph
  对每个data-carrying predecessor分别保留dependency，不能假设只有unary chain，也不能把support名字当语义；
- multi-result、fanout和fanin按实际producer result/consumer operand分别建relation，不假设`result(0)`、单init或equal shape。

对每个consumer logical shard，query以其完整execution domain求relation image，再与每个producer ownership domain求交。
这些intersection的typed union必须精确覆盖producer demand；未覆盖set是`proven logical infeasible`的direct witness。显式
replica允许多个eligible owner时，Q50.A保留等价ownership relation而不选择source；partial reduction owner则全部保留为必需
contribution并交给已选merge语义。malformed assignment、过期IR epoch或缺失上游role是compiler contract failure，不得作为
某个placement的普通no-good。

以下不是case特化，而是本边界必须支持的通用relation类别：

- **reduction**：parallel与reduction iterator共同定义consumer contribution domain，exact demand覆盖所有必要input/init片段，
  partial结果与merge义务不丢失；
- **broadcast**：many-to-one/投影relation的image保持唯一source logical set，重复consumer使用不膨胀为伪造ownership需求；
- **affine window**：convolution、pooling或supported reduce-window的offset、stride、dilation、kernel和显式pad/fill relation共同
  推导halo与valid source pieces；边界fill与真实producer demand分别保留；
- **stride与view/slice**：非unit stride、permutation、collapse/expand和static slice通过relation组合形成精确strided set；
- **multi-piece**：Presburger union、window/pad分段、tail或组合relation产生的有限多piece set原样保留，禁止先取bounding box或
  dense rectangle再声称exact。

上述current声明支持的类别不能以`unsupported`作为Q50.A完成后的常规出口。真正超出typed structured/IndexRelation合同的
dynamic、non-affine或未定义numeric semantics可以返回`unsupported semantic relation`，但它作用于对应语义/alternative，
不是换一个physical placement即可消除的失败。

### Typed outcome与physical隔离

query结果必须是named typed outcome，而不是`FailureOr + string`再压成`bool legal`：

1. `satisfied`：携带exact relation image、ownership intersections和空uncovered witness；
2. `proven logical infeasible`：只表示well-formed trial在logical partition/relation/ownership上存在可证明矛盾，并携带direct
   uncovered或role witness；Q49.P/Q50.B只有收到该结果才能跳过当前trial；
3. `unsupported semantic relation`：当前typed relation能力无法表达verified source semantics；不得缓存或改写为placement非法；
4. `indeterminate/compiler failure`：Presburger预算/内部错误、过期epoch、协议缺失或无法分类的失败；调用者停止对应合法化路径
   并保留compiler failure，不尝试其它placement掩盖问题。

`satisfied`结果不包含layout、encoding、bytes、dense rectangle、local/remote action、route、buffer、send/recv、fusion或
resource schedule。representation/movement关闭后，lowering才可把exact set分解为physical pieces并证明其union all-and-only
覆盖logical demand；某个dense/strided descriptor或route表达失败只拒绝该physical assignment。Q49.P为baseline选择的
canonical DDR/必要peer carrier若还不能有限表达一个已支持的exact set，属于baseline correctness carrier缺口，不是Q50.A
relation失败；Q50.G/H扩展完整performance alternatives时仍复用同一proof。

### API、cache与current迁移

policy-free logical placement assignment必须位于structured/relation边界，不能因Q50.A读取它就include candidate schedule或携带
evaluation、stable ordinal、route、residency、calendar和candidate统计。query只读current IR和immutable typed assignment，
不apply mutation；关系或domain需要跨mutation时建立新immutable borrow并重算，不把operation pointer、diagnostic字符串或旁路
side table当稳定语义键。

relation cache绑定一次immutable borrow，并以nested structural snapshot保护borrow内mutation；`IREpoch`token只拒绝跨borrow
trial，不进入semantic key。cache还必须观察typed edge/relation identity、完整consumer execution domain、producer ownership
domains、partition/reduction/replication role及会改变relation image的全部assignment。只有证明某些字段在当前relation下
extensionally等价时才能使用投影key；当前仅按两端shard dimension和participant count缓存`bool`不能成为终态合同。

current query和production adapter已经消费窄policy-free placement assignment、typed dependency descriptor与four-state outcome；
DPS init、pure tensor input graph和multi-result ownership均进入同一all-and-only exact relation/coverage边界。baseline producer value carrier直接消费
query给出的per-destination demand与ownership intersections，并把empty destination解释为无physical action，不再从balanced producer
rectangle正向猜测support image。direct edge的canonical dense carrier及其它layout、route、residency和resource calendar仍是
Q50.G/H后续扩展的physical alternatives；其失败不得回写Q50.A cache或删除spatial trial。

### Gate

- 独立exact-demand suite直接测试query，不以edge-strategy/materializer成功代签；覆盖multi-axis remainder、reduction input/init与
  partial merge、broadcast、affine window+stride+dilation+pad、strided slice/view、multi-piece union、multi-result、fanout/fanin、
  explicit init root及多operand pure tensor input graph；
- metamorphic gate对同一logical trial替换dense/strided/multi-piece carrier能力或让route失败，Q50.A outcome和exact set
  extensionally不变；ownership hole只产生typed `proven logical infeasible`，unsupported与indeterminate不会进入
  legality `bool`或no-good cache；
- cache gate跨immutable borrow、改变nested structural snapshot或任一观察到的domain/role都会miss/invalidate；
  extensionally等价placement重算得到相同stable logical proof，结果不依赖Tile/pointer/hash遍历顺序；
- Q49.P定向调用链不出现candidate/schedule evaluator或edge strategy selector，只对proven logical failure推进canonical
  placement trial；上述五类relation至少各有一个normal TensorProgram进入baseline canonical carrier并继续到完整Q50.0 gate；
- Q50.B explicit test assignment与后续production domain调用同一query；Q50.G/H只消费`satisfied` output，不从physical fragments
  反推另一份logical demand。旧skip-init/support与carrier-rejects-placement测试必须替换为current合同的正负证据。

### 实现检查点与follow-up review结论（2026-08-17）

- policy-free合同位于`ExactDemand.h`：`LogicalNodeTrial`同时携带完整iteration domain、按唯一Tile绑定的
  exact execution shard，以及按producer result绑定的ownership；role区分unique partition、explicit replication
  和partial-reduction contribution。`IREpoch`是由共享immutable token表达的borrow identity，不使用进程级
  mutable generation，也不冒充IR mutation detector。
- production adapter先按placement选定的spatial iterator直接在iteration space构造all-and-only balanced shard，
  再分别通过每个result的indexing map求exact image并分类ownership。这样multi-result不会把各result的preimage
  并集成重叠执行域；无法由unique partition或完整replication表达的partial overlap会fail closed。production
  placement domain不再排除multi-result node，定向case证明multi-result node可同时作为producer和consumer进入
  `StructuredDAGPlacementEvaluator`并得到完整carrier。
- query入口验证execution shard的space、非空性、唯一Tile、两两不交及对完整iteration domain的all-and-only
  覆盖；随后按Tile id稳定顺序分别求relation image、ownership intersection和uncovered witness。carrier组装要求
  `perDestination`的Tile集合与consumer placement完全相等，遗漏或重复destination均为physical carrier合同失败，
  不回写logical verdict。
- per-edge relation memo的真实失效边界由借入function的nested `OperationFingerPrint`保护。fingerprint观察operation
  identity/nesting、attributes/properties、block arguments、location、operands、successors和result types；保持op数量
  不变的nested attribute原位mutation也使旧query返回`IndeterminateFailure`。`IREpoch`只检查trial与query属于同一
  borrow，不承担第二份mutation状态。
- `IndexRelation`的projected-rectangle fast path同时保留source和destination bounds；矩形越界时回退generic
  Presburger结果。composition不再未经证明传播projection pattern，只有对完整bounded relation做等价证明的builder
  才恢复fast path，避免中间domain clipping被丢失。follow-up stack capture说明variable/disjunct structural limit不能单独约束
  generic `isEqual/isSubsetOf/subtract`的wall-time；logical proof和four-state outcome仍由Q50.A拥有，supported rectangle在baseline
  热路径上的closed-form witness保留、generic solver调用前的结构预算检查和work ledger由Q49.P P6闭合，不重新打开physical carrier或placement选择。
- typed payload在成功与失败路径都按semantic Tile id稳定排序，补齐consumer domain、producer result、dependency
  role、per-destination intersection和merge obligation。partial-reduction定向case证明倒序caller input仍保留全部
  contribution owners与merge义务；overlap failure witness同样不依赖caller枚举顺序。
- baseline support-chain carrier不再为每个balanced producer shard正向重建单矩形image，而是复用同一borrow上的
  per-destination exact-demand结果；strided view按destination形成有限fragment，insert-slice overwrite造成的empty
  destination不生成physical action，global multi-piece set由其余destination fragments all-and-only覆盖。
- normal TensorProgram production gate分别覆盖reduction、broadcast、affine convolution window、stride-2
  `extract_slice` view和middle-overwrite multi-piece relation；五个case都进入canonical baseline carrier，完成一次
  CardModule compilation并形成accepted 16-Tile CardExecutable。fresh source-to-package入口也实际执行并通过。
- fresh验证覆盖relation/epoch/query及carrier/multi-result production 62/62、上述五类production gate 5/5、Q50.0
  baseline/稳定性/reduction 3/3、source-to-package 1/1和主树完整构建。`ThreeStageChainCanUseThreeDistinctTileGroups`
  仍有约163秒的既有placement枚举成本；它归Q52，不改变Q50.A logical correctness结论。

Q50.A logical completion gate已闭合并在`tasks/progress.md`标为`done`。Q49.P P6补的是single-coordinate调用方的work closure，
不改变本节typed outcome；Q50.G/H继续扩展完整physical carrier alternatives，这些后续性能/representation能力不重建或改判
本节的logical proof。

## Q49.P 施工步骤（2026-08-17 建立）

Q49.P 的 contract 见上节；本节只拆解施工步骤、依赖 checkpoint 和每个步骤的验证。步骤按依赖顺序编号，
P1–P2 是两个既有失败的根因修复（gate case），P3–P7 是契约隔离收口，P8 是 fresh 端到端验证。

本轮融合前的已知gate基线（2026-08-17，`build/q55-current-fresh`）：
`NoneJointlyRefinesExplicitProducerStageAndConsumerDemand` 已复现：2048 breakpoint 上全部 region probe 返回
`requires-function-scope`（`UnsupportedLifetime`），controller 计数后跳过视同 fit，随后唯一完整 gate 以
`spm-allocation` 失败。CROSS 挂起按 `memory/bugs.md` 记录复现（`validateSelectedTileLayouts` 每 op×每 relation
从零递归 `collectStorageRoots`，无 memo）。current实现已补scope routing和memo，但上节代码复核列出的relation completeness、
single-root structure、policy isolation与work/output问题仍未闭合，不能把这两个局部修正解释为P1–P8完成。

### P1 actual gate与temporal refinement的一次性所有权

- 目标：同一closed coordinate只拥有一份actual CardModule；Q50.0直接消费并返回accepted、proven exact rejection或
  indeterminate。不存在region probe、function probe和final gate三份平行IR。
- 实现：baseline删除`lowerRootShards`、`lowerTileEntries`和`evaluateTileRegion/FunctionSPMCapacity`调用；Q50.0返回的完整
  `TileMemoryPlanningFailure`经current materialization relations形成direct typed causal witness。accepted executable原样返回；
  exact SPM capacity rejection销毁本coordinate owner并执行一次单调temporal refinement；其它exact rejection和indeterminate
  不触发repair。
- relation一致性由唯一actual链闭合：CardModule split、TileRegion-to-Instr、memory preparation和SPM planning每次mutation后
  都检查current relation；不得静默删除缺失witness，也不得把另一个scope或上一coordinate的SSA带入证书。
- 验证：work ledger证明`cardModuleMaterializations == cardModuleCompilationInvocations == controllerIterations`、accepted
  rematerialization为零；overfull case至少有一次typed exact rejection并最终accepted，普通fit case两者均为一。

### P2 CROSS 挂起修复（storage-root memo）

- 目标：`validateSelectedTileLayouts` 链路在 CROSS（16 destination × 16 owner peer fragment）上不出现
  O(ops×strategies×walk) 分钟级放大。
- 实现：`StructuredBufferRelations` 增加 query-local `StorageRootMemo`（`DenseMap<Value, DenseSet<Value>>`，
  同一 IR epoch 内只读），`shareStructuredBufferStorage`/`collectStructuredNodesUsedByOperation`/
  `operationUsesStructuredNode` 增加带 memo 的重载；`validateSelectedTileLayouts` 每 Tile root 构造一个 memo
  并贯穿调用链。current DS代码可原位吸收；旧无memo入口只保留给确实一次性的调用点，memo不得跨IR mutation、clone或trial。
- 验证：CROSS lit case 在 wall-time 上限内通过（目标秒级）；`WaferUnitTests --gtest_filter=...` 相关
  layout/fusion 测试不回归；fresh 主树构建通过。

### P3 全Tile exact-demand分析与最终single-root物化

- 目标：从全部structured root的Tile execution domain一次性推导operand demand；tensor transform relation按
  operation/result/operand建立一次，structured producer形成typed停止边界。16 Tile只apply各自非空demand、boundary、layout和
  target facts，不按root/Tile clone或遍历完整TensorProgram。
- 实现：source session拥有同一immutable IR epoch内的`TileRootDemand`、`ConsumerInputDemand`、`ProducerValueRequirement`和
  `ConsumerInputReconstruction`；以`(value, Tile)`合并exact set并按反向SSA拓扑传播。所有boundary carrier coverage在mutation前验证；
  actual CardModule materializer用`IRMapping`把recipe选中的operation一次性构造成final single-root region，不缓存materialized IR。
  16个独立Tile apply使用bounded executor并发，结果按Tile identity稳定归并。
- Q51沿用同一query/apply分层：与assignment无关的typed relation对象可在immutable search session共享；candidate-dependent
  execution/ownership demand必须观察全部相关坐标并可失效重算。partial state不构造IR，complete candidate只构造一份actual
  CardModule，accepted executable直接进入incumbent，不按Tile/candidate缓存actual clone或为winner再次构造。
- `lowerRootShards`、`lowerTileEntries`、`BaselineRegionSource`、support rebuild及只为discarded probe/closure存在的carrier、API和
  test退出current合同；仍需的single-root construction与relation proof先迁入唯一CardModule materializer再删除旧入口。
- 验证：work count证明relation construction不随Tile重复、每个非空`(value, Tile)`只处理一次、per-root完整TensorProgram clone为
  零；相同relation的16 Tile共享分析，不同tail/offset/communication demand仍产生各自正确IR；multi-producer
  `insert_slice`、CROSS/CHAIN/GEMM和最终fresh FP16 LLaMA baseline通过。

### P4 一 root 一 region 结构合同

- 目标：每个 baseline TileRegion 恰有一个 structured compute root 和exact operand demand证明必要的non-root tensor transform；
  同 Tile 多 root 形成多个顺序 region；跨 root shaped dependency 显式 DDR。
- 实现：resolved baseline assignment显式携带singleton root boundary；materializer按一个root及其`ConsumerInputReconstruction`证明需要的
  non-root tensor transforms直接创建一个TileRegion，不能先把同Tile root group进共同region再做late repair。跨root shaped dependency在边界构造时使用current
  canonical carrier形成显式DDR store/reload；从DS splitter迁移SSA/resource移动能力时，必须在同一transaction同步retarget
  consumer operand relation到actual reload。current固定256轮的post-hoc splitter和所有late boundary construction退出baseline
  调用链；generic split kernel只能服务其它已经显式选择的IR transformation，不能被baseline apply调用。唯一actual materialization逐region验证`rootCount == 1`，
  zero-root compute region也拒绝；同root lower成多compute op仍按node ID归一。
- 验证：新增结构单测并直接检查actual IR（同 Tile 多独立 root → 多 region；每region恰一root；跨root有显式DDR
  store/reload且consumer relation指向reload；显式structured producer不伪装成support closure；单 root multi-op不误判多root；
  missing relation/zero-root fail closed）；额外断言baseline transitive call graph不含post-hoc multi-root splitter；现有五类
  production gate与CROSS/CHAIN/GEMM通过。

### P5 direct-witness-only capacity attribution

- 目标：capacity refinement 的 causal coordinate 只来自 typed conflict certificate 和 materializationRelations，
  不做同 shape/type 的歧义 producer 扩展，不把同 region 其余 root 并入 refinement。
- 实现：删除 `synthesizeDeterministicBaseline` 中 equal-shape producer 扩展（`StructuredDAGEdgeStrategyPlan`
  段落的 shape 匹配）与 region-wide structuredNodes 组合；demand 无 typed node witness 时返回
  indeterminate（typed failure），不得按shape/type、唯一root或function-scope region集合猜测。P4的one-root-per-region只缩小
  probe scope，不能替代allocation→current buffer→structured owner的typed relation。删除current“无relation时写入
  `operandDemandNode`”fallback，并分别覆盖all-unattributed与mixed direct/unattributed，二者都不得因局部bool更新顺序产生不同结论。
- 验证：equal-shape fanin定向测试证明只按direct witness选择refinement目标；无direct witness为indeterminate；mixed certificate
  不接受推测owner；`NoneJointlyRefines...`仍通过。

### P6 baseline 不消费 search domain/state/candidate/evaluator/statistics

- 目标：baseline controller 只消费typed iterator/topology事实、单coordinate legality、Q50.A exact-demand query 和 canonical
  carrier；不构造完整per-node placement-option集合、`StructuredDAGPlacementSearchDomain`、
  `StructuredDAGPlacementEvaluator`、stable ordinal、
  proposal/transition 统计或 selected evaluation metrics；交给 materializer 的是窄 resolved baseline
  assignment（placement、singleton region boundary、完整 temporal vector、canonical representation/
  movement/buffer/order/completion），不含 score/ordinal/transition history。
- 实现：
  1. 删除baseline对`deriveStructuredDAGNodePlacementOptions`及其“全部iterator-axis × connected-rectangle”结果的调用；
     删除`deriveCanonicalBaselinePlacements`中的domain propagation与recursive CSP/backtracking。从current node structured axes、
     verified topology和semantic tie-break直接产生唯一current maximum-participation coordinate；每条edge只查询这一对已关闭的
     producer/consumer shard。只有direct typed exact rejection才执行预定义、单调且不回溯的functional legalization transition，
     同时销毁旧coordinate；不得保存alternative、回退栈或no-good。Q50.B未来完整domain仍由自己的惰性mechanism与reference
     enumerator拥有。没有可spatial partition的parallel result轴时直接构造typed unpartitioned coordinate：iterator factors全1、
     一个canonical Tile、完整result domain且没有伪造`shardDimension`；这不是reduction spatial split。
  2. `deriveBaseline`/`deriveDeterministicBaseline` 改为直接构造 `TileMapping`（nodePlacements +
     outputPlacements + canonical edge strategies/layouts + 完整 temporal vector + materializationMode +
     bufferCount=1）的窄 assignment 类型，不再经 `TileExecutionCandidate`（stableOrdinal、evaluation、
     transition、feedbackRootOrdinal）与 `StructuredDAGPlacementEvaluator`；legality由每edge当前single pair的Q50.A query及全
     assignment demand plan/canonical carrier闭合，不建立option compatibility table。per-root/output placement用typed sum明确区分
     `unpartitioned`与`partitioned(axis, factors)`；current强制unsigned `shardDimension`的carrier不能用0/sentinel伪造纯reduction
     scalar output，producer、materializer、verifier和consumer在同一current合同中一起切换。
  3. 对single-coordinate exact-demand热路径建立独立work closure。2026-08-17 bounded stack capture确认当前LLaMA诊断停在
     `getExactStaticRectangularImage`的generic `PresburgerSet::isEqual`，继而进入`isSubsetOf/subtract`；变量/分段数上限不能保证该
     等价证明有界。支持的projected/permuted/static-rectangle indexing semantics必须由`IndexRelation` builder/composition保留或
     直接构造closed-form rectangular-image witness，使baseline不进入generic equality recovery；generic fallback在调用前按完整
     relation完整结构复杂度在任何昂贵solver调用前检查并计入query-local ledger，超限返回`ResourceExhausted`/indeterminate，绝不能当作
     logical rejection或触发下一个coordinate。用轻量、同relation结构的定向case覆盖该路径，不靠重型LLaMA重现。
  4. baseline 路径不再接收`CardExecutableSearchStatistics`这个search bag，也不再写candidate proposal/fusion/layout/buffer
     统计和selected evaluation metrics；必要的probe/materialization/work计数进入baseline专属窄ledger，
     diagnostics以baseline专用行报告真实work；删除current固定`placement_enumeration=0`的自证字符串，测试只断言窄ledger、
     compile-work ledger和旧statistics type不在调用闭包。temporal fallback是functional legalization step，不再写
     `allocationFeedbackTransitions`或其它search feedback字段。
- 验证：grep/调用计数证明 baseline 调用链不含完整placement-option/search domain/evaluator及旧statistics type；
  `deriveStructuredDAGNodePlacementOptions`、`deriveCanonicalBaselinePlacements`、`TileExecutionCandidate`和
  `CardExecutableSearchStatistics`不在baseline transitive call graph；ledger证明placement domain size=0、recursive
  solve/backtrack=0、exact pair query数只随actual DAG edge和deterministic legalization step增长，且supported rectangular-image
  case不进入generic Presburger equality recovery；CROSS/CHAIN/GEMM、显式`none`的overfull-to-fit与五类production gate通过，
  不执行或改造旧search-named回归；
  主树、board runtime、SystemC 三棵树 fresh 构建通过。

### P7 baseline materialization/output seam清理

- typed coordinate query不materialize IR。每个closed coordinate构造一次完整CardModule并立即调用一次Q50.0；两者owner单向
  传递且work count一一对应。current `root shard → Tile entry → CardModule`和`full card → scoped Tile → full card`链路全部退出。
  source/target validation与immutable semantic index每个baseline session只建立一次。
- Q50.0 accepted result直接成为baseline semantic result；删除baseline的`buildAcceptedStructuredDAGSchedulePlan`、duration
  estimation和`StaticSchedulePlan`/theoretical-cost include。Q51需要的同cohort actual cost从accepted current Instr/resource
  facts按自己的typed cost边界取得，不由baseline预建shadow schedule。
- compile API把IR inspection改为显式optional sink/请求；普通source-to-package、`none`及候选evaluation均不调用IR printer。
  显式dump只对最终accepted Tile modules生成一次，不进入executable/package语义或acceptance gate。
- baseline implementation和typed result从旧search monolith/shared header中分离；result只拥有move-only accepted executable、
  必要actual facts和窄ledger，不携旧search statistics、printed trace、candidate relation或failure string state machine。
- transaction/failure-injection与baseline定向测试显式使用最短`none`路径；不运行默认旧search、旧winner对照或异常长integration。
- 验证：work count证明source validation/index=1、每个controller iteration的CardModule/Q50.0各一次、accepted
  rematerialization=0、普通IR print=0、schedule-plan build=0；显式IR dump只来自accepted owner；baseline不可能因未消费的
  cost/trace失败。

### P8 fresh 端到端验证与收口

2026-08-17复核把剩余缺口定性如下：

- `TileMemoryPlanningTest.ReportsSPMFailureForOwnedTileModule`失败不是上游SPM planner没有evidence。
  `PlanSPMMemory`已填充`largestDemands`、`capacityConflictDemands`和`individuallyOversizedDemands`；
  `planTileMemory`在没有`materializationRelations`的分支只复制scalar summary，漏传这三个集合。Q49.P须让有无relations两条路径
  都先保留完整typed evidence，再由relations可选补充structured owner attribution；定向测试同时覆盖两个入口。
- 重型LLaMA block的`none`曾运行2h21m仍未完成。bounded stack capture显示它在
  `deriveCanonicalBaselinePlacements → StructuredDAGExactDemandQuery::imageDemand →
  IndexRelation::getExactStaticRectangularImage → PresburgerSet::isEqual/isSubsetOf/subtract`消耗CPU。这里同时暴露了完整
  placement-option/recursive CSP和generic Presburger rectangle recovery两层无界工作，按P6分别拆除；该运行只作根因证据，
  不能作为通过证据。对应算法、结构与work ledger闭合后，Q49.P仍须运行一轮fresh、bounded FP16 LLaMA
  `optimization-none` source-to-package/no-card；不得在施工中反复用重模型代替定向定位。
- current spatial placement只从parallel iterator生成axis，reduction factor保持1且没有partial-result merge；这是Q50.B的明确
  implementation gap；同时current placement carrier在没有parallel result轴时连该root的factor=1/单参与Tile coordinate也不能表达，后者是
  Q49.P功能baseline必须修复的前置缺口。相反，fresh `SearchProposesAndMaterializesMultipleReductionIteratorAxes`和
  `CardBaselineCompilationTest.CarriesReductionDemandThroughTheCompleteExecutableGate`已经证明多reduction轴temporal materialization与baseline complete
  gate存在；Q50.E待闭合的是完整breakpoint/wave-loop domain，不是从零补一个reduction temporal split。
- current融合代码复核发现的relation completeness、consumer reload retarget、zero-root、direct-witness fallback和空raw evidence问题，
  分别由P1/P4/P5和本步骤的定向gate闭合；不得仅修局部bool或放宽diagnostic后继续沿whole-Tile旧controller运行。
- 旧`MultiOutputFanoutCanPlaceBranchesOnDifferentDestinationGroups`单case超过3h、RSS约4.3GB的现象属于待删除的旧spatial/search
  枚举链。Q51.Core前不再运行该长case；Q50.B以tiny reference enumerator重建域，Q51 closure只验证new chain。

P8完成顺序：

1. 闭合P1–P7及上述SPM evidence copy，更新`tasks/progress.md`、相关设计和稳定memory；
2. fresh运行direct unit、定向lit、overfull-to-fit、五类relation和轻量source-to-package/no-card；显式`none`，不进入旧search或
   paired optimization；这些门禁闭合后只运行一轮fresh FP16 LLaMA `optimization-none` source-to-package/no-card；
3. ledger证明placement option/domain/recursive solve/backtrack为零，single-coordinate exact query有界，完整CardModule
   materialization与CardExecutable compilation各一次，accepted后schedule-plan/默认IR print为零；
4. 检查主树、board runtime与SystemC/model三棵树构建；确认相关lit/CTest实际执行而非unsupported；
5. 提交本批改动（作者规范见`AGENTS.md`）。

此前fresh证据（2026-08-17）的默认lit 216/216、三棵树构建、Tools/Runtime lit 33/38（5 unsupported、0 failed）及显式
`none` f16 reference/AddModel source-to-package/no-card可复用为未改边界的背景，但不代签新baseline。原“受影响unit 70/70”
已被current `TileMemoryPlanningTest.ReportsSPMFailureForOwnedTileModule` fresh失败覆盖，不能继续写成current通过结论；P1–P7施工后
必须按上述新relation/structure/work gate重跑有界集合。旧LLaMA capture或未完成compile不进入本项证据；新算法闭合后的fresh
FP16 LLaMA baseline no-card是Q49.P最终功能门禁。Q51完整new-search链闭合后，Q52才首次运行重型LLaMA `search`显式profile；
Q53再生成正式模型package/oracle/no-card并进入board-ready。

### 2026-08-18 施工前合并代码复核（后续模型复核再次打开）

本次复核以合并后的 current HEAD 为对象，同时检查 baseline 新路径、DS 引入的 probe/relation/split 代码及其
直接消费者，不把 DS commit 孤立成第二条实现线。复核只运行有界 `none`/relation 定向验证，没有执行 search，
也没有执行重型 LLaMA block。结论是：function-scope escalation、raw demand evidence copy、storage-root memo、
accepted-result move、shadow schedule 删除和显式 IR trace 等资产可以保留，但原“P1/P3/P5/P6/P7 已闭合”的状态
在该复核时点尚不成立，因此当时将Q49.P保持为`doing`；后续小图修复曾形成一份完成记录，但fresh FP16 LLaMA又证明
materialization算法仍未闭合，下面将该记录明确标为已撤回并给出current替代合同。

阻塞项按 correctness 优先级如下：

1. **`IndexRelation` 新 fast path 不能作为 exact proof。** `functionalByConstruction`只证明 affine map
   single-valued，不证明 bounded relation 对完整 destination domain total；domain/source restriction 后仍保留旧
   projected-rectangle pattern，也会绕过 clipping。complete-reduction shortcut 没有证明 iteration domain 覆盖完整
   source，all-zero affine fallback 还可能把常量坐标扩大成完整 source。相关 shortcut 会使 Q50.A demand legality
   接受不完整或越界关系，必须先恢复数学正确性、删除成功路径的 `IMGFALLBACK-DEBUG`，并补 clipped-totality、
   restricted-pattern、incomplete-reduction-coverage 负例。
2. **P3 的 root-scoped probe 实际仍包含 sibling root。** `externalBoundaryCarrier`被传入 materializer 后没有
   consumer；producer/consumer 两侧都会进入 root strategy，boundary supply 只重接 operand，不能阻止 sibling
   compute materialize。fresh IR dump 中 producer-root 与 consumer-root 两个 probe 均有两个
   `wafer.tile.elementwise`；现有测试只断言 compute op `>= 1`，与“无 sibling”注释不一致。
3. **P7 仍是 full-card → per-root probe → full-card。** baseline 初次 materialize 完整 CardModule，冲突后才进入
   scoped probe，fit 后重置 scope 再 materialize 完整 CardModule；现有测试还固定
   `baselineCardModuleMaterializations == 2`。每个 root/Tile probe 又重新 clone/prepare 整个 TensorProgram。
   这才是当前 LLaMA 慢 case 的本质 work amplification；最终 16 个 Tile 的 lowering 已经由 bounded executor
   并发执行，不能用最终 Tile 并发掩盖 controller 前面的重复全图工作。
4. **P5 的 witness attribution 仍会沿任意 downstream op 扩散。** 未识别 user 会把全部 DPS init/result
   继续压入 worklist，一份 allocation 可被归给无关 descendant/fanout root；mixed attributed/unattributed evidence
   也没有 fail closed。同时 probe result 的 `SPMDemandEvidence` 暴露 scratch IR 的 `mlir::Value`，scratch 销毁后
   handle 悬空。跨 probe 边界只能返回稳定 node/type/bytes/relation witness，不能返回 scratch SSA。
5. **P4 仍依赖 post-hoc splitter。** baseline 仍调用 `splitStructuredRootBoundaries`；one-root verifier 只做局部
   buffer 等值匹配，不能证明 storage-root/alias 后的 structured root all-and-only，zero-root legacy branch仍可成功。
   splitter 的 SSA/resource/reload retarget 能力可以迁移，但不能把 late repair 记为 construction-time singleton-root
   合同已完成。
6. **P6 只移除了 option/CSP 枚举，没有闭合整个 baseline seam。** `none` 当前确实直接构造一个 canonical
   coordinate，不能再描述成候选枚举或 search；但 maximum-participation coordinate 被 exact reject 后会直接失败，
   尚无确定、单调的 spatial legalization。typed `unpartitioned` 只是未被 consumer/hash/order/resource signature读取的
   bool，所谓 scalar test 实际是 `tensor<1x1>`，rank-0 和无 parallel result轴仍没有端到端合同。product compile
   入口也仍构造/打印旧 search statistics bag。
7. **reduction temporal 与 spatial 必须分开表述。** current materializer和 baseline complete gate已经支持
   reduction iterator 的 temporal tile；baseline capacity fallback是根据 exact conflict witness 逐步缩小有限 breakpoint
   的确定性贪心 legalization，不是候选枚举。尚未实现的是 reduction 轴的 spatial partition及 partial-result merge，
   归 Q50.B；Q50.E补的是完整 temporal breakpoint/wave-loop domain。另一方面，constant indexing-map 的新放宽过宽：
   只检查“dim或constant”，没有证明 constant 为0、对应 extent为1、其余 dim不重复，upstream verifier 可接受的
   非零constant map会被错误当作 complete reduction slice，必须收紧或实现真正的 exact general semantics。

修复顺序固定为：先恢复 relation shortcut 的 exactness并补反例；再建立 session级 immutable preparation/index 和
真正的 root/Tile 窄请求；随后把 singleton-root boundary前移到 construction并退出 post-hoc splitter；再收紧 direct
witness与稳定 evidence 类型；之后闭合 end-to-end unpartitioned coordinate及 deterministic spatial legalization；最后删除
baseline product closure中的旧 search statistics/header，并用 P1–P8 原 done criteria重新验收。局部测试绿色、最终 16 Tile
并发或单个 gate通过均不构成 Q49.P 完成证明。

本轮 fresh evidence：main、board-runtime和SystemC/model三棵现有build tree均成功增量构建；显式非search unit
71/71、baseline CHAIN/CROSS/GEMM/no-card lit 1/1、relation定向unit 20/20通过。绿色结果没有覆盖上述反例：
`WAFER_DUMP_NARROW=1` 的narrow-root定向case虽返回pass，两个probe dump仍各含两个structured compute op；relation
成功case会无条件打印`IMGFALLBACK-DEBUG`；pinned MLIR verifier也接受非零constant output map的合法上游IR，证明
current lowering不能假设所有constant天然等价于extent-one zero slice。后续修复须把这些反例变成negative/structural gate。

### 2026-08-18 Q49.P 旧完成记录（已撤回）

以下内容记录当时合入P1–P8后观察到的局部能力，不再是current完成结论。随后fresh FP16 LLaMA baseline证明
multi-producer tensor input operand仍会被过宽SSA closure和post-hoc rebuild错误物化，因此这些证据只能说明旧小图门禁通过。
其中一次CardModule/Q50.0 owner、非search controller、unpartitioned coordinate、relation exactness、temporal reduction和
bounded Tile executor等独立能力继续保留；root closure/rebuild及其work证据不再属于目标设计。当时记录如下：

- `none`直接从typed iterator/topology事实构造一个live canonical coordinate；exact rejection只推进预定义、单调、无分支且
  不回溯的spatial/temporal functional legalization。baseline public API/result/ledger由
  `CardBaselineCompilation`拥有，不接收旧search statistics、candidate/domain/evaluator、score、shadow schedule或
  默认IR trace；普通compile的IR print为零，显式caller-owned trace只在accepted result后生成。
- rank-0或无parallel result轴使用typed unpartitioned coordinate：一个canonical参与Tile、完整result domain、无伪造
  `shardDimension`；一般ranked workload仍按canonical maximum-participation placement使用最多16个active Tiles，完整Card ABI
  始终包含16个Tile entry。最终per-Tile lowering由bounded executor并发；这不把baseline描述成single-Tile，也不改变
  真实板端launch串行约束。
- 一个immutable `TileMaterializationSourceSession`每次baseline只prepare一次；每个legalization coordinate建立一个mapping-local
  session。typed coordinate query不物化IR；每个coordinate只形成一次完整CardModule并立即由一次Q50.0 CardExecutable compile
  消费。accepted owner直接下传，exact SPM rejection才销毁owner并推进下一coordinate。
- IndependentDDRStages从observable shard和actual selected edge endpoint推导真实参与root；独立component直接构造并按node顺序
  拼接，selected RegionCut/Peer carrier在物化transaction中形成DDR store/reload边界。baseline调用链已删除post-hoc
  `splitStructuredRootBoundaries`；每个compute TileRegion以current-SSA relation的region ownership验证恰一structured root，zero-root
  和multi-root均fail closed。
- exact relation fast path补齐bounded totality、restriction失效、complete-reduction source coverage、rank-0 Presburger矩形和
  constant-map exact条件；generic rectangle recovery在任何昂贵solver工作前检查分段、constraint、local variable及绝对系数，
  超限返回typed indeterminate。capacity attribution只返回稳定node/type/bytes witness，scratch SSA不逃逸；raw SPM certificate在
  有无relation时均完整保留。
- deterministic temporal fallback是贪心functional legalization，不是候选枚举；current已支持reduction iterator temporal tiling。
  reduction轴spatial partition及partial-result merge仍归Q50.B，完整temporal breakpoint/wave-loop搜索域仍归Q50.E，二者均不反写
  Q49.P为未完成。

合入代码review另发现并修复了四个会破坏上述合同的实现问题：source-only destination traversal曾先在SPM拼完整spatial shard再
复制到DDR；source-only clone漏传structured node mapping；RegionCut effect closure漏追resident-fragment DDR读对应的writer，能
形成跨region读先于写；per-component edge过滤会删除remote incoming PeerFragments，且independent consumer sealed result未进入
共同materialization cache，导致CROSS重新融合或重算sibling root。修复后carrier、current relation和query cache均以actual
producer/consumer endpoint及当前IR epoch为准。

当时的局部验收证据：

1. fresh定向unit 106/106通过，覆盖exact relation、16-shard interval coverage、CardModule actual construction、one-root-per-region、
   root闭包公共分析、Q50.0一一对应、SPM overfull-to-fit refinement、memory planning和current relation；
2. `wafer-compile-card-baseline.test` 1/1通过：CHAIN/CROSS/GEMM三个FP16图均完成source-to-package和no-card，CROSS actual Instr
   含DTE send/recv/wait；
3. `build/q55-current-fresh`、`build/wafer-board-check`和`build/q54-fresh-model`三棵树均以`-j$(nproc)`增量构建通过，compiler/model
   public link smoke 2/2通过；
4. ledger证明每个coordinate的CardModule与Q50.0一一对应、accepted不重物化；fit branch图只做2次公共closure分析/4个operation，
   而不是16 Tile各自重扫完整TensorProgram；16 Tile actual entry通过共享线程池并发且按Tile ID稳定归并；
5. 当时没有运行search case或重型LLaMA block；把重型LLaMA完全推迟到Q52的决定已经由后续fresh baseline反例推翻。

fresh FP16 LLaMA `optimization-none`在约33秒内到达Q49 materialization并失败，diagnostic为consumer input dependency存在未组装的
structured producer。定向诊断确认同一consumer operand由两个structured producer经`extract_slice`和`insert_slice`汇合：
其中一个producer对该destination的exact demand为空，另一个producer有非空需求；Q50.A已经给出正确empty事实，但
`buildBaselineRegionSource`仍从result/edge endpoint无条件回溯operand closure，`rebuildPeerSupportInput`随后又从consumer operand
递归复制support graph，于是把empty producer重新拉入。该反例撤销Q49.P的`done`状态，也证明“公共closure操作数较少”不能作为
需求正确性或模型通用性的证据。

### 2026-08-18 current exact-demand materialization算法

本节替换上方旧记录中的root closure、support rebuild和LLaMA门禁；前文Q49.P Pipeline Contract的其它边界继续有效。

#### 输入、结果与不变量

算法只读同一immutable TensorProgram epoch、Q50.A logical shard trial、structured node/result identity、每个node在每个Tile上的
exact execution domain、每个producer result的exact ownership和已经关闭的temporal coordinate。查询产生三类短生命周期typed
结果：

- `TileRootDemand`：一个structured root、semantic Tile及其exact execution domain；
- `ConsumerInputDemand`：该root一个tensor input实际读取的exact index set；
- `ProducerValueRequirement`：传播到structured producer result后形成的停止边界，包含producer node/result、destination Tile和非空exact
  required domain。

pure tensor transform另外产生`ConsumerInputReconstruction`：记录current operation/result、output demand、各tensor input的exact read
demand和同一语义对应的reconstruction动作。上述对象不持有materialized IR，不进入IR attr、mapping identity、candidate、磁盘格式
或跨epoch cache；apply前若borrow失效即返回indeterminate。`TileMapping.edgeStrategies`只表达已经证明非空的physical carrier，
不增加empty flag或nullable旁路。

必须始终满足：structured producer是传播停止边界；每个final TileRegion恰有一个structured root；每个operand reconstruction的
initialized coverage与`ConsumerInputDemand`相等；每个`ProducerValueRequirement`由local/peer/DDR fragment all-and-only覆盖；同一
`(structured node, Tile)`只执行一次，同一physical fragment只发射一次。

#### 全Tile反向传播

先为closed coordinate中全部nonempty `(root, Tile)`建立seed，而不是为16个Tile分别walk TensorProgram：

```text
for each scheduled structured node N:
  for each Tile T with nonempty execution domain E[N,T]:
    for each tensor input K of N:
      D = exactAccessRelation(N, K).image(E[N,T])
      addDemand(N.operand[K], T, D)
```

`addDemand`以`(SSA value, semantic Tile)`为key累积Presburger set union，只传播相对已处理集合的新差集。当前single-block
functional TensorProgram按反向SSA拓扑序即可一次完成；同一operation/result/operand的`IndexRelation`建立一次并对所有Tile demand
复用。若未来region/control-flow不能提供等价typed relation和稳定拓扑，则保持unsupported，不能退回递归closure。

```text
propagate(V, T, D):
  D := D - processed[V,T]
  if D is empty: return

  if V is a function argument or constant:
    record directly available input(V,T,D); return

  if V is tensor.empty and D is nonempty:
    return typed unsupported uninitialized read

  if V is a structured producer result:
    record ProducerValueRequirement(V,T,D); return

  if V = tensor.extract_slice X:
    DX := extractRelation.preimage(D)
    record recipe(V,D,{X:DX}); addDemand(X,T,DX); return

  if V = tensor.insert_slice Source into Destination over image W:
    DS := inverseInsert(D intersect W)
    DD := D − W
    record recipe(V,D,{Source:DS, Destination:DD})
    addDemand(Source,T,DS); addDemand(Destination,T,DD); return

  if V is an admitted pure unary view/tensor transform:
    DI := exactRelation.preimage(D)
    record recipe(V,D,{Input:DI}); addDemand(Input,T,DI); return

  return typed unsupported or indeterminate
```

generic multi-input compute不属于support：具有structured semantics的operation必须成为structured node；effectful、alias不明、
只能近似传播或没有reconstruction证明的operation在mutation前fail closed。nonrectangular demand保留exact Presburger set或有限
disjoint rectangle union，不先扩大成bounding rectangle。

#### 局部语义证明与组合证明

对任意value `V`和需求`D`，apply必须建立`M(V,D)|D == V|D`，且`D`外没有consumer读取。每个admitted
`ConsumerInputReconstruction`必须证明：若所有operand materialization分别在recipe给出的read demand上等于源operand，则reconstruction
在output demand上等于源operation。query和apply只消费这一份recipe，不各自维护matcher。

`tensor.insert_slice`的证明直接来自overwrite语义。设source覆盖result区域`W`，任意需求`D`唯一分为
`D ∩ W`和`D − W`；source demand是前者的inverse image，destination demand是后者。destination demand为空时，
apply可以用未初始化tensor作为只承载source insertion的容器，但coverage仍只有`D intersect W`，任何其它读取都会被拒绝；这不是
empty strategy或模型特判。`extract_slice`和一元view由exact preimage得到同样的局部定理。

function argument/constant与已验证carrier构成归纳基，structured producer result形成停止边界，每个consumer input reconstruction构成归纳步；
因此沿acyclic SSA结构归纳可得每个root operand在其read domain上与源程序相等，再由structured op indexing semantics推出root在
execution domain上相等。temporal waves按稳定顺序all-and-only覆盖execution domain，reduction accumulator显式跨wave携带；
没有partial-result merge时reduction spatial factor仍保持1。

#### Boundary覆盖与一次性apply

对每个`ProducerValueRequirement(producer result P, destination T, required D)`，按producer ownership求：

```text
fragment[S] = D intersect ownership[P,S]
```

所有非空fragment必须两两不重叠且disjoint union等于`D`。本地fragment绑定baseline的compiler-owned DDR stage，远端fragment绑定
peer send/receive/wait；empty intersection没有physical action。coverage缺口、重叠、route/descriptor无法表达和内部失败保持各自
typed outcome，不能改判Q50.A logical result。

全部query和coverage验证成功后才创建最终CardModule。每个Tile按structured DAG稳定拓扑顺序直接构造single-root regions；
`IRMapping`只把已经选定的source argument、constant、tensor transform和root映射到最终region，每个operation只作为最终IR复制
一次。apply按recipe重建operand：boundary leaf使用已绑定fragment，view/extract使用exact slice，insert按两路coverage组合；完成
后检查coverage等于`ConsumerInputDemand`再物化root。source-only send只加入producer endpoint，不拉入remote consumer。禁止创建
`BaselineRegionSource` scratch module、无条件operand closure、post-hoc region splitter、RegionCut/Peer support rebuild、replay或
失败后补边。

whole-shard `ProducerValueRequirement`定义persistent carrier coverage；root temporal loop内以当前wave execution domain重新求同一recipe的
wave-local slice，只把当前wave demand带入SPM，不先组装完整spatial shard。interior和tail使用同一relation与loop IV表达，不按wave
复制support graph。

#### 实现迁移和完成门禁

1. 将Q50.A current query原位扩展为按consumer operand分组的exact-demand结果；现有per-edge consumer全部迁移，不能并存第二套
   baseline relation恢复逻辑。
2. `TileMaterializationSession::create`在borrowed source上完成全Tile query和carrier coverage验证；`lowerCardModule`只apply已验证
   typed结果。partial/search query不构造IR，complete coordinate才进入同一apply。
3. common CardModule/TileRegion materializer直接构造final single-root region；baseline与后续search复用该policy-free apply，不能
   保留baseline-only修补分支。
4. 同批删除`buildBaselineRegionSource`、两套support rebuild、empty/destination补丁、post-hoc root closure/splitter及只覆盖这些
   路径的statistics/tests；旧SPM capacity probe API不恢复。
5. 每种admitted consumer input reconstruction用tiny static shape穷举需求子集，比较full evaluation与partial reconstruction；组合测试覆盖
   chain、diamond、fanout、multi-producer `insert_slice`、Peer/RegionCut混合、multi-result、empty branch、nonrectangular pieces、
   reduction temporal wave和unsupported atomic failure。
6. 显式测试计数证明relation construction按semantic support edge计数、每个非空`(value, Tile)`只处理一次、16 Tile没有完整DAG
   重复walk、CardModule/Q50.0一一对应且accepted不重物化。这些计数只在`--compile-timing`或测试传入sink时建立，普通编译不创建
   统计对象、不打印日志。定向unit/lit与轻量FP16 source-to-package/no-card通过后，只运行一轮fresh FP16 LLaMA
   `optimization-none` source-to-package/no-card；不得进入search。该门禁已于2026-08-19通过。

#### 2026-08-19 current实现与剩余门禁

current实现已经迁入按职责拆分的`Compiler/Baseline`、`Compiler/Planning`和TensorProgram→TileRegion conversion文件：

- 一个baseline invocation只建立一次只读source session；`StructuredNodeUseIndex`一次索引structured use，`StorageRootMemo`压缩
  storage-root查询，mapping session消费按consumer operand分组的exact demand；
- final materializer在structured producer处停止，只重建typed recipe要求的argument/constant/view/extract/insert与carrier；
  multi-producer fan-in、exact-empty branch和Peer/RegionCut混合不再通过递归support closure恢复；
- `buildBaselineRegionSource`、root/function SPM probe、post-hoc support rebuild、winner rematerialization与默认Tile IR/statistics已经退出
  baseline主路径；每个closed coordinate只产生一个CardModule并由Q50.0消费一次；
- CardModule fan-out时，大型Tile body直接move进唯一Tile output；只有每个output确实需要的小型card-shared declaration按
  `IRMapping`复制。16个Tile entry与后续独立Tile stage使用bounded并发并按Tile ID稳定归并；
- active测试已覆盖single-root、multi-producer、reduction temporal accumulator、Peer/RegionCut混合、Card→Tile move-only owner、
  source-to-package与no-card。已经删除的旧TileRegion候选测试不复活；其中仍属current合同的proof已迁入CardModule/current gate。

首次fresh FP16 LLaMA `optimization-none`编译在约163秒内生成current package，CardExecutable阶段约99秒、peak RSS约4.48GiB；
第一次完整CTest随后因PyTorch runner仍读取已退役的`package/functions/forward.mlir`而失败。current package有意只保存
manifest/modules/program data，因此runner现改为从source program读取`functions/forward.meta`、从package读取manifest port，并已通过
定向runner unit、对该fresh package的payload准备和`wafer-run --no-card`。修复后official CTest从fresh source重新执行完整链，
于181.70秒通过1/1；Q49.P据此收口为`done`。

这次显式timing还记录到16个Tile合计执行约22,968次TileRegion→Instr conversion。结果已经有界且正确，但相似Tile body仍有大量
重复lowering；该问题归Q52按semantic shape、mapping/tail/endpoint事实拆分可共享immutable analysis与Tile-local apply。Q51 search
只能复用这些可重算facts，不能缓存actual IR、clone完整DAG或重放materialized result；Q51完整链闭合前不运行LLaMA search。

## Q51.Core：Search Control Kernel

### 2026-08-17 review 结论

原节把Q51终态assignment schema、Q50.F旧式局部actual gate、Q51全轴oracle和Q52 production策略都提前算进Core，和current施工
顺序不一致，也会诱导实现直接复用正在由Q49.P拆除的`TileExecutionCandidate`。Core现在只拥有**搜索控制**：typed child
state的原子接纳、deterministic frontier、stable dedup、incumbent、预算与已用work计数、typed evaluation outcome与result
evidence。它不生成任何physical choice，不预声明尚未施工的轴，也不签发真实domain completeness。

current public `search`不是可保留的proposal provider，而是一条需要整体替换的错误实现链：

1. `deriveShortlist`同时构造baseline、placement coordinate descent、temporal/layout/edge/buffer sweep、witness seed和cheap排序；
2. `TileExecutionCandidate`同时携带assignment、derived metrics、stable ordinal、allocator history和feedback控制；
3. placement evaluator提前混入canonical carrier、movement/residency和resource schedule，materializer又承担fusion/layout/buffer选择验证；
4. complete compile失败后在同一loop里生成allocator/buffer siblings、beam closure、priority reorder和candidate-local caches；
5. accepted cohort清空actual executable，以`StaticSchedulePlan`和旧tie-break选winner，再完整rematerialize一次；
6. public diagnostics、statistics和大量测试锁定上述proposal数、ordinal、feedback与旧winner行为，并反复执行异常长旧链。
7. registered board “source contract”逐文件读取源码并检查旧`RankCandidateSearch/Evaluation/Selection` symbol marker，optimization
   comparison catalog和pending hardware inventory又强制保留`none/search` paired driver；旧source即使不进active CMake也因此不能删除。
8. PyTorch/board runner默认选择旧`search`并解析`card-executable-selection actual_fused_edges` stderr；package commit等test-only
   failure injection也被硬绑为只有`search` policy可用，使与搜索无关的回归重复进入旧长链。
9. common candidate boundary把failure原因降成字符串`gate`，controller再比较这些字符串决定cache、feedback和排序；
   `CardExecutableSearchStatistics`以约百个字段把baseline、proposal、feedback、winner和rematerialization耦成一个协议。

这些责任不通过adapter迁入Core，旧行为不参与任何新链判定。新链只复用能脱离旧owner独立调用的IR facts、Q50.A exact demand、
Q50.0 complete compilation和downstream lowering/verifier；其它算法只有在对应Q50机制按新typed合同重新证明后才选择性迁入。

| Search责任 | current实现事实 | 新owner与处置 |
| --- | --- | --- |
| public policy routing | `OptimizationConfig::search`直接进入单体synthesis search branch | Q51.Core改接new-search session并删除旧branch；无production mechanism时返回Q49.P incumbent |
| functional fallback | search内部重新derive baseline并把它编码成candidate seed | 只接收Q49.P已accepted move-only executable作为session incumbent；不进入candidate state |
| state identity | `TileExecutionCandidate`混合mapping、metrics、ordinal和feedback history | Q51.Core空typed aggregate；各Q50只加入本轴不可重算assignment，derived facts留analysis |
| spatial/domain generation | placement domain、evaluator、coordinate descent和hand-written witness seed耦合 | Q50.B按all-iterator/topology/exact-demand合同重写惰性typed domain；Q50.A proof独立复用 |
| temporal/region/layout/movement/buffer choices | `deriveShortlist`内多轮coordinate sweep与固定action/buffer集合 | Q50.C–I分别拥有domain/query/transition/apply；不存在共同旧generator或默认补值 |
| schedule/resource ordering | candidate schedule在placement evaluation和cheap ranking中提前生成，accepted后又建selector plan | Q50.J/K从新assignment重算event/resource analysis并物化actual order；Q51只比较完整actual cost |
| actual materialization | `materializeCandidate`混合CardModule构造、fusion/layout/buffer检查与complete compile | 各Q50 apply只物化已选事实；complete assignment经唯一materializer进入Q50.0，Q50.0保持无策略 |
| late failure handling | allocator/buffer failure生成siblings、lookahead、beam closure并重排shortlist | Q50.F及complete gate只返回deferred/exact rejection/indeterminate；Core回到typed parent，不做late repair |
| incumbent/winner | accepted cohort丢弃actual modules，以schedule plan/stable ordinal选winner后再rematerialize | Q51 session持有move-only accepted incumbent，以同cohort actual cost和semantic tie-break原子替换；不重物化winner |
| failure routing | materializer写字符串`failureGate`，controller按字符串决定prune/cache/feedback/order | 新Core只消费closed typed evaluation outcome和typed causal evidence；diagnostic label只打印，不参与控制流 |
| inspection | 每个complete compile在acceptance前无条件打印全部Tile IR并穿过synthesis result | Q49.P先把common seam改为显式winner-only inspection；Core/candidate evaluation不携printed IR |
| policy/cost aggregate | `WaferTargetPolicy`混合memory facts与Quick/Default/Deep、candidate/beam cap；`TargetScheduleCostPolicy`混合hard memory、profile reference和未校准prior | hard target facts回各target/memory owner；Core只接收session-level typed comparable cost，不消费aggregate policy；Q52只在fresh profile后建立独立estimate prior |
| telemetry/tests | proposal、feedback、ordinal、selected metrics及旧长链golden；源码marker CTest、paired optimization catalog、stderr parser和search-only fault injection保活旧实现 | 新global ledger、per-axis oracle和new source-to-package tests；删除旧source-contract/catalog/inventory binding、旧数值/digest/耗时/执行路径断言，事务测试走最短current路径 |

```text
Pipeline position:
- Upstream IR / input:
  一个immutable TensorProgram borrow、transaction-owned ProgramData view及target/cost-cohort facts；Q49.P完成后通过Q50.0完整准入的move-only baseline
  executable、actual resource cost和stable actual digest。Core入口没有semantic-alternative、spatial、region、temporal、
  layout、movement、buffer或schedule assignment，也不重新构造baseline。
- Current stage responsibility:
  建立唯一replacement search session的control plane：管理typed partial assignments及其ephemeral transitions、对已接纳
  transitions不丢state的deterministic frontier、canonical dedup、global work/time budget、typed evaluation routing、
  session-level incumbent和coverage/lower-bound evidence。Core只接受mechanism已经形成的typed transition/evaluation result，
  不枚举或物化某个轴。
- Output IR / files:
  不产生新IR层或文件。输出query-local control result：原样持有Q49.P baseline incumbent，并携带frontier、work、coverage和
  unresolved-work evidence。Core checkpoint没有production physical candidate，因此public `search`直接返回该incumbent；后续
  Q50机制形成complete assignment后，结果才可持有经Q50.0 accepted的actual executable。candidate不写package、不commit目录；
  Q59 outer transaction只发布最终winner一次。
- Downstream consumer:
  Q50.S与Q50.B–Q50.K逐项扩展current typed assignment和transition；Q50.F接入scoped feasibility analysis；Q51 closure接入
  全部真实轴后的complete actual materialization/Q50.0 gate parity、全轴oracle、exact-domain证明和source-to-package chain。
- User-level driver / named pipeline:
  不新增pass、CLI、optimization policy或磁盘sidecar。Q51.Core同批把既有`search` policy路由到new-search session并删除旧
  branch；任一时刻只有一个public owner。Core阶段该入口只发布Q49.P accepted incumbent，后续Q50机制原位扩展同一session。
- Explicit non-goals:
  不实现Q50.S/B–K任一choice domain；不在Core或Q50.F建立materialize-and-discard局部gate；不预设best-first priority、
  default budget、no-good/dominance/DP/LNS算法；不创建generic provider registry、`any`/字符串tag/opaque payload或future-axis
  placeholder；不以mock domain声明production physical domain完整。
- Done criteria:
  finite typed fixture由独立state-graph model证明Core遍历与reference reachable-state set一致；budget stop不会吞掉仍由exact
  frontier表示的state；rejected/indeterminate child都不影响合法siblings；baseline不进入
  candidate key或candidate统计且不会重编译；stable winner与ledger不依赖hash iteration、pointer、proposal ordinal或
  evaluation完成顺序。public `search`只进入新Core并在空production domain时返回Q49.P incumbent；Core实现、链接和测试均不含
  `TileExecutionCandidate`、旧generator/evaluator/feedback/selector、字符串gate控制或旧长耗时integration；registered CTest/
  board inventory不再读取旧source marker、不要求`none/search` paired package或解析旧selection stderr；与搜索无关的transaction/
  failure-injection test不再被强制走`search`；Q50.S/Q50.B可在不修改control semantics的前提下加入第一批真实typed轴。
```

### 四类query-local事实

```text
Immutable session input:
  immutable IR borrow + target facts + cost cohort + currently implemented mechanism set

Candidate assignment:
  only named typed choices implemented and consumed by current Q50 mechanisms

Session control:
  deterministic non-dropping frontier + incumbent + work/budget ledger + coverage/bound evidence

Derived cache:
  current-IR analysis + exact demand + lifetime/calendar/SPM/cost facts
```

- `IREpoch`沿用Q50.A的current合同：它只验证query与trial属于同一immutable borrow，不进入semantic key，也不替代nested IR
  structural snapshot或MLIR analysis invalidation。Core不得把token地址、`Operation *`、walk ordinal或printed IR当stable key。
- Core建立的current assignment aggregate起初不含未来轴。每个Q50 checkpoint同批加入本轴named typed field、transition、
  canonical encoding、query/apply和失效关系；Wafer-owned接口原位演进，不保留编号schema、旧wrapper或两套state。
- mechanism query只读source、target和parent assignment；transition apply先完整验证，再原子构造child。child不保存transition
  history、proposal priority、feedback root、failure history或derived metric。缺少尚未施工的轴表示partial assignment不可
  materialize，不允许从baseline default或旧selector暗中补齐。
- mechanism set、priority、incumbent和ledger是session facts，不属于任一candidate。future-boundary equivalence只有在本轴
  mechanism证明被投影choice不再影响任何future consumer时才可合并；Core不提供按aggregate bytes/makespan自动合并state的
  generic shortcut。mechanism顺序由唯一current driver静态组合，不提供runtime registry、dynamic plugin list或user option。

### Baseline incumbent 与新链candidate

Q51 driver先运行Q49.P并把已经accepted的move-only executable直接交给Core。Core从该executable已有的actual resource facts按
本次冻结的comparison cohort取得cost；同一term对baseline和后续candidate同时启用或删除，不能candidate-local缺项。baseline：

- 不是root search state，不进入frontier/dedup/no-good，也不把canonical placement/temporal等functional choice变成seed；
- 不计入generated/expanded/actual-probed/search-accepted统计，baseline时间与Q50.0 work单独记账；
- 没有更优complete candidate、其它candidate exact reject或保持unresolved时，仍原样成为结果，不再次materialize或compile；
- accepted candidate只有在完整actual cost更优，或cost相同且完整semantic tie-break更小时，才以move替换incumbent；比较不依赖
  queue ordinal、hash insertion或并行完成顺序。

Core checkpoint用typed fixture evaluator证明控制算法，并以public source-to-package定向测试证明`search`经新Core直接发布
Q49.P incumbent；production mechanism set暂为空。它不从旧path取得complete assignment，也不为了验证actual comparison而重放
旧materializer。Q50.S/B–K加入的partial state在全部required coordinates闭合前不能调用Q50.0，也不存在“用baseline或旧selector
补齐未施工轴”的complete candidate。第一条production candidate必须由新typed mechanisms构造，并在其所属Q50
source-to-Q50.0测试中通过actual IR与Q50.0。

`ProgramDataHandoff`始终由外层compiler transaction唯一拥有，Core和每个actual executable只消费stable program
identity/range，不复制owner或为candidate materialize package。落选executable按RAII销毁；最终winner选定后，Q59才把同一
handoff随唯一结果继续交给target/package transaction。baseline work不进入candidate统计，但driver保留从Q49.P开始的总wall/
work账本和time-to-baseline，Core不能重置计时起点掩盖baseline成本。

### Transition、evaluation 与budget

- exact expansion对一个parent返回当前mechanism的全部immediate typed children，或返回仍留在frontier中的typed semantic
  continuation；opaque iterator/cursor不能成为遗漏siblings的旁路。先访问的child无论accepted、rejected或indeterminate，
  其它siblings都保持可达。
- `deferred(required coordinates)`列出typed prerequisite，只有相关assignment改变或闭合后才重新query；不得poll同一state。
  proven exact rejection只删除被其typed causal witness覆盖的state。Core本身不从diagnostic string合成no-good或扩大scope。
- resource/solver exhaustion及Q50.0 indeterminate outcome形成unresolved work：不删除state、不更新incumbent，也不能签发
  optimality。Core自身的malformed transition、stale borrow或broken invariant是fatal compiler error，不得用baseline掩盖。
- Q51.Core只路由fixture evaluator outcome；第一项能形成complete assignment的Q50 mechanism接入时再路由production Q50.0
  outcome。Core不构造actual IR。Q50.F只从immutable facts提供可重算的lower-bound/deferred/causal analysis；complete
  assignment一次性物化CardModule并进入Q50.0。
- ledger分别记录generated、deduplicated、expanded、deferred、exact-rejected、unresolved、full-compiled、accepted
  及每个exact gate work；执行work前先reserve确定的credit。wall deadline只是安全中断，不能改变已完成state的stable排序。
- budget中止时Core只报告typed evidence，不自行声称`feasible-with-bound`：Q51 closure只有在exact frontier仍完整且有typed
  admissible bound时才映射为该等级；显式丢弃/未表示state后只能是`budgeted-feasible`。Q52再根据profile确定默认budget和
  有损policy。

### Oracle分层

1. **Core state-graph model**：test-only typed fixture直接给出有限reference graph和accepted cost，不调用production mechanism；比较
   reachable state key、frontier exhaustion、winner、ledger和budget coverage。覆盖empty domain、diamond convergence、同state
   不同transition order、exact rejection、deferred、indeterminate、fatal invariant和budget stop。
2. **Per-axis reference enumerator**：从Q50.S开始，每个mechanism用tiny真实semantics独立枚举本轴合法typed choice，并与
   production mechanism逐parent比较stable key；这不是Core gate。
3. **Full production flat exhaustive runner**：Q51 closure在全部真实轴闭合后才使用actual materializer、cost和Q50.0 gates，
   比较candidate set、pruning、winner与actual digest。Core fixture不能替代这一层。

### Current code处置与完成条件

- 可直接复用Q50.0 move-only accepted result、Q49.P resolved baseline handoff、Q50.A immutable-borrow合同，以及
  `StructuredDAGAnalysis`中仍为current IR可重算的事实；existing work-ledger reservation思路可迁移，但固定cap和rank合同不得带入。
- 当前`TileExecutionCandidate`混合assignment、`TileExecutionMetrics`和`TileExecutionTransition`；后者还含stable ordinal、
  feedback root、allocator history和beam控制。Core不拆分、适配或构造该组合类型；新assignment从空aggregate开始，由每个Q50
  mechanism加入本轴named typed field。`StructuredDAGScheduleState`中的ready/live算法只有在Q50.J按新event assignment合同
  重建后才能迁入，旧candidate schedule不进入identity或Core API。
- `CardExecutableSearchStatistics`中只服务旧proposal、feedback、shortlist、selected ordinal和winner rematerialization的字段
  随Core旧branch同批删除；Q49.P已在前一任务脱离该shared bag，Core使用新的global ledger，derived winner metrics从accepted
  executable重算。不得保留同字段的新struct或为旧diagnostic提供compat adapter。
- 删除旧`RankCandidateSearch/Evaluation/Selection`、coordinated/bounded driver及其未注册test/source island；仍被Q50未来合同需要的
  独有mechanic按对应Q50 owner逐项迁移，其余直接删除。同步删除`wafer-compiler-optimization-source-contract`、optimization
  comparison cases/driver、pending hardware inventory中的paired qualification row和只验证旧symbol/source文本的tests；不把它们
  改名移植为新Core gate。
- `OptimizationConfig::search`的public spelling可以保留，但默认/显式调用都只能路由new Core；现有14处显式旧search长链CTest、
  PyTorch board默认值/`actual_fused_edges` stderr assertion及search-only test-control限制逐项改为current最小路径或基于actual
  IR/package的测试。旧winner、统计、耗时与日志格式无回归义务。
- Core state-graph tests全部通过；同一fixture按不同insertion/hash/analysis completion顺序得到相同state set、incumbent和ledger；
  public source-to-package `search`只进入新Core并返回同源incumbent；link closure不引用旧generator/evaluator/feedback/selector，
  也不要求并行actual evaluation。
- 本checkpoint独立提交但不改变Q51整体`queued`状态；它完成后Q50.S与Q50.B开始加入真实typed axes。per-axis reference
  enumerator、full production flat exhaustive runner、exact full-domain coverage、有效fusion、new source-to-package
  chain和旧实现零残留仍是Q51 closure的完成条件；public routing已由Core完成。

### 2026-08-19 完成结果

public `search`已经切到窄的`CardExecutableSearch`入口；在尚无production mechanism的Core checkpoint，它只接收并move返回
Q49.P已经通过Q50.0的baseline，不重新构造TensorProgram、CardModule或CardExecutable。`SearchControl`只定义query-local
typed evaluation、按semantic key排序的frontier/dedup、显式work budget、未决state和incumbent替换；这些计数不打印、不持久化，
当前空domain的public编译路径不创建统计或日志对象。

能力迁移先于删除：current-SSA structured DAG facts迁入`StructuredDAGAnalysis`；baseline canonical parallel-axis与observable
placement closure迁入`Compiler/Baseline`并以multi-producer、exact demand和malformed coordinate测试覆盖。随后删除旧rank/coordinated、
placement enumeration、candidate schedule与accepted schedule-plan source/test island；Q50.B不会复活该枚举器，而从typed spatial
assignment和tiny reference oracle重建完整域。读取旧源码marker的注册测试、paired `none/search` driver/catalog、hardware inventory
binding、旧selection stderr和14个旧长链search CTest同步退出；只保留一个小型FP16 public `search` source-to-package/no-card路由门禁。

fresh验证包括Core/placement/exact-demand相关unit、全部724个C++ unit、source organization、PyTorch/board host contract及轻量
`wafer-compile-search-routing` lit。Q51完整production链尚未闭合，因此没有运行LLaMA `search`；该Core checkpoint之后由
Q50.S和Q50.B–K分别迁移独有mechanism、建立真实axis oracle并删除对应旧local owner。

## Q50.S：Structured Semantic Alternatives

```text
Pipeline position:
- Upstream IR / input:
  一个immutable、card-local、single-function TensorProgram；函数体仍是target-independent tensor/Linalg SSA，尚未出现TileRegion、
  memory space、movement、buffer、worker或physical schedule。
- Current stage responsibility:
  从current SSA的Linalg indexing map、iterator、DPS relation、scalar region、use-def、view relation和function result可观测性证明
  可用的attention算法族；惰性给出typed graph-alternative参数域。只有调用者请求一个具体点时，才clone最近的isolated Module，
  在该clone中重新证明并物化ordinary、online recurrence或split-K/V partition/merge TensorProgram root。
- Output IR / files:
  query只返回不拥有IR的typed finite domain；materialization返回一个move-only verifier-legal TensorProgram Module。函数签名、
  result arity和functional K/V cache append保持不变，不产生文件、TileRegion或physical事实。
- Downstream consumer:
  Q51 partial assignment把该typed root choice作为第一轴；Q50.B–Q50.K继续为同一root补齐physical coordinates。只有完整assignment
  才构造CardModule并调用Q50.0，Q50.S自身不lower、不估价、不选择winner。
- User-level driver / named pipeline:
  只由既有public `search` session内部使用，不新增pass、CLI、registry或独立selector。其余physical轴尚未闭合时，public入口只
  query当前domain并返回accepted baseline，不枚举partial roots、不clone，也不创建默认统计或日志。
- Explicit non-goals:
  不把普通output/reduction temporal tiling、SPM fit、Tile count、layout、movement或schedule塞进graph choice；不按op/name、
  Q length、参数位置或单个模型恢复语义；不缓存、replay或重物化actual root。
- Done criteria:
  tiny functional decode的production domain与独立reference枚举完全一致且顺序稳定；original、online和split-K/V点分别构造
  actual verifier-legal root，source保持byte-identical；precomputed-score near-miss只保留original；query clone/work为零，
  每次actual materialization最多拥有一个isolated clone；旧attention local pass/selector/source和source-marker test全部退出；
  current build、Q51/Q50.S unit、轻量public search source-to-package/no-card及source organization通过。
```

### Typed domain与算法边界

对已证明的attention reduction extent `R`，domain不预建point vector，而由typed successor常数工作地产生：

- `Original`恰好一个；
- `OnlineAttention(block)`覆盖`1 <= block <= R`；
- 只有functional decode证明成立时，`SplitKeyValueAttention(block, partitions)`覆盖
  `2 <= partitions <= R`且`1 <= block <= ceil(R / partitions)`。

`block`在这里决定online recurrence图中每个K/V片段及loop-carried `(maximum, sum, unnormalized-output)` 状态，因此是graph
parameter；已经形成该图之后，structured op的普通temporal tile和wave order仍归Q50.E。`partitions`决定独立partial
log-sum-exp/output sibling及显式merge拓扑。SPM capacity、Tile数和某个常见block常数都不缩小合法域，只能在后续physical
assignment与cost中发挥作用。

资格证明要求score producer是可tile的current SSA producer，并且score结果只流向已证明的row-max与shift路径；预计算score、
多重observable attention、非functional cache update或不一致的K/V append relation均不广告对应算法。物化直接建立纯tensor/
Linalg/SCF图，保留函数类型和returned cache values；它不建立output tile循环，不把graph block混同Q50.E temporal choice。

### 2026-08-19 完成结果

active实现已经按职责归入`Compiler/Search`：ordinary attention proof、functional decode proof、tensor graph op construction、typed
domain/materialization和public search routing彼此分离。producer slice materialization只通过窄的`ProducerTileFusion`内部接口复用
current TilingInterface机制；`TensorProgramScope`也从单体`Internal.h`拆出。旧`AttentionSemantics`、
`MaterializeFlashAttention`和`MaterializeFlashDecoding` source已在独有SSA proof及online/split actual-root能力迁入并受测后删除，
没有恢复旧pass、provider、cost prior或字符串key。

Q50.J清理dormant source时发现旧implementation interface仍独有exact pointwise `1/x` reciprocal sibling，已按能力先迁移再删除补入
current `CardComputeImplementationDomain`：每个structured node默认保留Natural，只有all-parallel、static、identity-output且每个input
`IndexRelation` exact、scalar body恰有一个`arith.divf 1.0, x`时才增加Reciprocal。query不clone；selected choice随node-shard group进入
同一次CardModule actual apply，Natural生成Div，Reciprocal生成Recip，apply再次验证exact `1/x`。旧target-capability bool、外部interface
materializer和dormant source/test已删除；该typed implementation assignment由Q51与其它轴联合选择。

fresh验证覆盖12个Q51 Core/Q50.S定向unit、全部729个C++ unit、轻量FP16 public `search` source-to-16-Tile package/no-card、
source organization和主构建。
本checkpoint没有运行重型LLaMA `search`，也没有把Q50.S partial domain提前枚举进普通编译路径；重型search仍等Q50.B–Q50.K与
Q51完整链闭合后再运行。

## Q50.B：Spatial Partition + Physical Placement

```text
Pipeline position:
- Upstream IR / input:
  Q50.S选定或original的immutable TensorProgram root、current StructuredDAG、verified available Tile集合和Q50.A exact-demand query。
- Current stage responsibility:
  为每个structured node惰性产生覆盖全部iterator的block factor vector、canonical logical-coordinate顺序、到distinct physical
  Tile的有序embedding，以及spatial reduction的显式merge Tile；把完整per-node assignment闭合为Q50.A logical trial。
- Output IR / files:
  不产生新IR或文件。输出typed `CardSpatialPlacementAssignment`与`satisfied/proven logical infeasible/unsupported/indeterminate`
  evaluation；satisfied携带all-and-only execution shards、result ownership、partial contribution和merge owner。
- Downstream consumer:
  Q50.C按选定per-Tile shard物化single-root TileRegion；Q50.D–K继续补region/fusion、temporal、representation、movement、buffer和
  schedule。只有完整assignment才进入Q50.0。
- User-level driver / named pipeline:
  只由public `search` session内部静态组合，不新增pass、CLI、provider或独立placement selector。Q50.C–K未闭合时public路径只
  query compact domain，不枚举Cartesian product、不clone并直接保留accepted baseline。
- Explicit non-goals:
  不物化TileRegion/CardModule，不选择temporal tile、layout、route、buffer或winner；不把connected/rectangle/all-16、result axis、
  participant count或常见factor当legality；不把Q50.A unsupported/indeterminate改写成placement rejection。
- Done criteria:
  single node、chain、independent branch、diamond、multi-axis remainder、scalar、reduction merge和partial redistribution的production
  domain与独立reference集合一致；任意有序available-Tile subset可达；numeric/PartialReduction interface不支持时reduction factor
  只保留1；完整trial经Q50.A typed evaluation，旧single-axis placement字段零残留，query/production路径无clone与默认统计。
```

### 机制

- 从structured iterator semantics枚举每个op覆盖所有iterator的完整spatial factor vector与block partition
  relation；它必须联合表达multi-iterator factors、non-divisible remainder/tail、parallel/reduction iterator role和
  必要的partial-reduction merge owner，result axis不能替代iterator axis；
- 对每个partition惰性枚举verified topology上全部verifier-legal participant subsets、logical-mesh embedding和physical
  placement。compact rectangle、natural mesh、connected set和all-available-Tile只是排序seed；除非verifier能从硬件
  topology证明connectivity是legality，否则必须保留disconnected/asymmetric placement；available Tiles均可参与；
- 在完整域中优先生成ordered factorized regular mapping：把logical iterator的一个或多个spatial factors映射到typed physical
  topology dimensions。只有axis/factor nesting改变extensional partition relation、logical-mesh embedding或physical placement
  时才由对应typed relation/embedding区别；等价factorization的生成顺序canonicalize，不进入state identity。未映射local
  extent只传给Q50.E，不能在这里隐式固定temporal tile或wave-loop order；
- 不同 op 可使用不同 Tile sets，独立 branches 可并行，dependent nodes 可 co-locate、partial overlap 或 disjoint；
- consumer placement关闭时直接以完整logical domain调用Q50.A exact demand/ownership coverage；只有typed `proven logical
  infeasible`删除该transition，unsupported/indeterminate向owning scope传播。physical representation限制不得反写成logical
  demand不合法；
- 只有topology、已选placement和当前resource assignments同时对称时才能omit symmetric state；coverage lower bound
  和admissible placement bound可在共同candidate set内剪枝；mechanism不保留
  “最佳 placement”。

### Actual witness 与 gate

independent tiny reference enumerator与mechanism生成域在single op、chain、independent branch、diamond/fanin、
multi-axis remainder、reduction merge和partial redistribution上的stable key集合完全一致；Q50.B的actual witness是Q50.A
消费的closed logical trial，selected CardModule/TileModule ownership由紧随其后的Q50.C物化。局部gate通过后删除旧single-axis
placement field和rank==Tile假设，但保留
可复用topology/domain/cost mechanics。“非最大参与、非相同Tile group或暂时更贵的placement成为global actual
winner”是Q51 closure的跨轴gate。

同一factor count中，确实改变partition relation、logical-mesh embedding或physical placement的不同axis/factor nesting必须产生
不同typed witness；extensionally等价的生成路径必须canonicalize。regular/full-occupancy proposal与非矩形、partial、
asymmetric、disconnected合法补集均可达；关闭或改变proposal排序后，tiny exhaustive domain和winner不变。

### 2026-08-19 完成结果

current placement identity只含完整iterator factor vector、row-major logical coordinate到physical Tile的有序embedding，以及
reduction merge Tile；旧`spatialPartition(iteratorDimension,resultDimension)`双事实源已删除。每个factor维度覆盖`1..extent`且
product不超过available Tile数，physical sequence惰性覆盖全部distinct ordered subsets，因此非最大、非连续、非对称和不同branch
placement均可达。factor生成只有一个canonical iterator顺序；其它factor nesting若表达同一坐标到Tile映射不会产生重复state。

spatial reduction只有同时具备`PartialReductionOpInterface`和current scalar combiner numeric proof时才广告factor>1；proof已从
conversion私有实现迁到`Analysis/Structured/ReductionSemantics`，query与actual partial materializer共用。closed trial把每个partial
owner标为`PartialReductionContribution`并携显式merge Tile，Q50.A保留全部必要intersection和merge obligation。7个Q50.B unit加
27个exact-demand及5个partial materialization unit通过；public search只构造Q50.S/Q50.B compact domains，未枚举、clone或运行
重型LLaMA search。fresh全量C++ unit 736/736、轻量public search source-to-package/no-card和source/IR organization通过。

## Q50.C：Maximal Single-Root TileRegion

### Pipeline contract

Pipeline position:
- Upstream IR / input:
  Q50.S选定或original的immutable单函数TensorProgram、current `StructuredDAG`、Q50.B已经满足Q50.A的
  `LogicalShardTrial`，以及其中每个node在每个Tile上的exact rectangular iterator shard。调用方已经选择singleton
  structured root boundary；本层不得自行产生或选择root group。
- Current stage responsibility:
  对每个非空node shard直接物化一个actual `wafer.tile.region`。该region只包含一个structured DAG node的local
  iteration work和从其operand SSA反向可达、但在其它structured node处截断的pure tensor transform；function input、常量和
  其它structured node result成为显式region DDR boundary。一次apply只构造将被下游继续消费的actual IR，不先构造probe、
  不保存可重放body，也不在accepted后重建。
- Output IR / files:
  一个拥有完整available-Tile domain的`wafer.card.module`。每个`wafer.tile.module`包含该Tile选中node shards的
  private single-root region functions；函数的shaped arguments/results是当前checkpoint的typed DDR boundary obligation。
  `StructuredMaterializationRelations`把每个actual compute映回唯一structured node。该中间CardModule是Q50.D–K继续原位
  变换的actual candidate ownership，不是可交给Q50.0的完整CardExecutable输入。
- Downstream consumer:
  Q50.D在actual regions上物化或拒绝coupled traversal/fusion；Q50.E–K继续加入temporal、representation、movement、buffer、
  event/resource order和conditional pipeline。只有上述选择全部物化、boundary obligation被Q50.H具体movement关闭后，完整
  CardModule才进入Q50.0。
- User-level driver / named pipeline:
  由public `search` session内部静态组合；Q50.C本身不新增CLI、pass、独立provider或用户可手拼pipeline。Q50.C–K未闭合时
  public `search`仍不枚举完整Cartesian product、不运行重型LLaMA search。Q49.P与Q50.C复用同一个policy-free exact
  iterator-tile与TileRegion body-emission mechanics；baseline继续由自己的canonical assignment controller提供boundary和
  movement，不经search state或Q50.C中间容器重建。
- Explicit non-goals:
  不选择multi-root group、temporal tile、layout/encoding、local/peer/DDR action、route、buffer、schedule、cost或winner；不把
  boundary argument当成已经选择的movement；不把函数名、operation ordinal、buffer或result axis当root identity；不缓存或复制
  已物化的actual IR，不对相似Tile replay body。
- Done criteria:
  single-result、multi-result、DPS init、unpartitioned reduction、spatial partial-reduction contribution/merge ownership和effect
  boundary正负例通过；每个非空selected shard恰有一个actual outer TileRegion且relation只映回请求的node；其它structured
  producer不能进入该region；所有available Tiles存在且空Tile保持合法；baseline与Q50.C共用的iterator-tile primitive受同组
  roundtrip/atomic-failure测试；不存在旧complete-rank/rank-tile私有入口仍独占本合同所需能力。

### 机制

对给定structured root、spatial shard和尚未细分的local iterator domain，每个Tile物化一个覆盖该root全部local work和
exact operand demand所需non-root tensor transforms的maximal single-root TileRegion。显式producer只要映回另一个structured DAG node，就以boundary
input/DDR obligation留在region外，除非后续Q50.D显式选择multi-root coupled group。这里maximal表示“对已选region boundary
不漏work、不把同一root local work任意拆成多个相互不知情region”，不是强迫使用最大tile、最大fusion group或单一结果驱动入口。

region materializer必须由SSA、structured interface、indexing maps、DPS init/effect和observable roots驱动；不得从buffer、
op、result名恢复traversal。多个result、init operand、non-root producer-to-consumer tensor chain和reduction均要得到all-and-only local work；
root cardinality由materialization relation映回structured DAG node，不能按lowered compute op数量或“support”标签猜测。

### Gate

- single-result、multi-result、DPS init、reduction 和 side-effect boundary 正负测试通过；
- actual Tile IR 对每个 selected node shard 都有唯一、完整、可 verifier 的 single-root region witness；spatial reduction的
  selected merge Tile另有一个同node single-root merge witness，不能把merge伪装成某个contribution或movement标签；
- Q49.P完成后建立的resolved baseline assignment apply与Q51显式test assignment复用该policy-free single-root materializer；
  materializer只消费调用方已给定的singleton root boundary，不接收或自行推导search candidate/grouping；
- 不在本 checkpoint 做 multi-op fusion、temporal winner、layout 或 buffer 选择。

### 2026-08-19 完成结果

Q50.B的closed `LogicalShardTrial`先按node/Tile恢复exact rectangular iterator shards，再由一次actual apply建立完整available-Tile
CardModule。每个shard函数直接创建在最终`TileModule`，其最小backward closure通过`IRMapping`一次性构造将成为actual body的pure tensor operation；
其它structured node result和source input成为typed DDR function boundary。function-level conversion直接消费最终owner，不为取得
module anchor构造synthetic module，也不存在probe、accepted replay、actual IR cache或默认统计。

`StructuredIterationTile`现在是baseline configured temporal leaf与search single-root apply共同使用的TilingInterface primitive；
multi-result elementwise generic lowering按每个typed result/yield建立实际buffer relation。spatial reduction contribution通过
`PartialReductionOpInterface`使用neutral accumulator输出partial tensor，完整iteration-space partial assembly和merge只在Q50.B选定的
merge Tile上物化，原始DPS init因此只消费一次。Card assembly、root function construction和partial merge分别位于独立source，
没有回到单体conversion文件。

fresh测试覆盖multi-axis remainder、空Tile、zero-dimensional iterator、multi-result、DPS-init structured producer、两个structured
producer经`tensor.insert_slice`汇合、unpartitioned reduction、multi-axis spatial reduction contribution/merge和effect atomic failure。
fresh host unit 745/745（普通722与dependency conformance 23分开执行）、相关Q49.P baseline/CardModule conversion/Q50.C定向64/64、
source/IR/dependency organization和轻量public search routing通过；本checkpoint按约束未运行重型LLaMA search。旧
complete-rank/coupled traversal仍只作为Q50.D能力donor，未重新注册为production owner。

## Q50.D：Maximal Coupled Traversal

### Pipeline contract

Pipeline position:
- Upstream IR / input:
  Q50.S selected/original immutable TensorProgram、current StructuredDAG、Q50.B satisfied logical shard trial，以及Q50.C已经证明可
  singleton物化的每个node/Tile iterator shard。Q51只提供一个explicit group partition；本层不得自行选择winner。
- Current stage responsibility:
  在同一Tile上惰性枚举所有合法connected node-shard partitions，并对调用方已选partition一次性构造actual regions。singleton
  group复用Q50.C；multi-node group以group sink为驱动递归tile/fuse内部producer，跨group structured producer仍是DDR function
  boundary。共享producer的同一exact tile/version在同一block复用，不能为fanout重复物化或经DDR round-trip。
- Output IR / files:
  typed `CoupledRegionAssignment`只包含稳定Tile与node group partition；actual apply输出完整available-Tile CardModule，每个group
  对应一个outer TileRegion，`StructuredMaterializationRelations`中的emitted node集合与group成员exact相等。group identity、source
  pointers、scores和derived traversal cache不写入IR或磁盘。
- Downstream consumer:
  Q50.E在已选actual group traversal上展开完整temporal vector/wave-loop order；Q50.F–K继续处理feasibility、representation、
  movement、buffer和schedule。只有complete assignment才进入Q50.0。
- User-level driver / named pipeline:
  只由public `search` session静态组合，不新增pass、CLI、单独fusion provider或旧rank selector；Q50.D–K未闭合时public search仍
  不枚举完整domain或运行重型LLaMA search。
- Explicit non-goals:
  不用per-edge `fused` bool、action recipe、group attr、symbol名或operation ordinal表示selected fusion；不选择temporal、layout、
  movement、buffer、cost或winner；不把只有same Tile但exact demand依赖remote owner的edge广告为coupled；不在Q50.C actual IR上
  事后拼region或clone/replay body。
- Done criteria:
  per-Tile typed partition domain与independent restricted-growth reference在chain、fanin、fanout、diamond和disconnected图上集合相同；
  nonlocal demand、partial reduction、effect或不支持的tensor relation保持cut；singleton、maximal和中间cut均能actual物化；每个
  multi-node group恰一region、emitted node集合exact，内部producer→consumer SSA不出现DDR store/load，fanout相同producer tile只
  有一个actual version；旧CompleteTraversal的仍需mechanics与测试迁入后删除旧入口/selector。

### 机制

Q51 可以对一个 SSA-connected multi-op group 选择 coupled traversal。对一个**已经选择的 group boundary**，materializer 必须
consumer-driven 地递归遍历全部 required producer closure，复用共享 producer/value，正确处理 fanout、fanin、diamond、
conversion、最后 consumer 和 observable boundary。

“maximal coupled traversal”不等于“最大 group 永远胜出”。机制必须同时产生合法 boundary cuts、独立 traversal 和
不同 coupled groups；最大 closure 只是其中一个候选。group、共同 iterator relation、region-local retention choice和
boundary-demand obligation作为typed assignments进入Q51 state；具体lifetime由它们与后续buffer/order重算，boundary
movement由Q50.H显式选择，不能退化成单条edge的`fused=true`或`CoupledFusion` recipe。

### Gate

- chain、two-input fanin、fanout、diamond 和不兼容 iterator boundary 均有正负测试；
- actual fused witness 是共同 TileRegion/recursive traversal，中间 SSA value 不经无意义 DDR round-trip；
- 删除 fused-edge bool/local recipe winner，但保留 CompleteTraversal、BidirectionalTiling 和 verifier 中可复用部分。

maximal与较小cut都能在本checkpoint物化；“maximal group因SPM/parallelism不优而由较小cut获胜”放在
Q51 closure，不在temporal/layout/buffer/schedule尚未接入时使用旧owner证明。

### 2026-08-19 完成结果

current `CoupledRegionDomain`只从Q50.B satisfied trial、StructuredDAG和typed producer-to-consumer relation构造。一个edge只有在
consumer destination demand非空、全部ownership intersection都由同一physical Tile拥有、tensor chain受current reconstruction
contract支持且两个endpoint都不是spatial partial reduction时才连入fusable graph；其它情况保持显式cut。每个Tile先拆fusable
connected components，再把assignment自身解码成canonical restricted-growth labels逐点推进，不保存partition列表。完全不相干的
8-node图因此直接得到singleton终点，不扫描Bell数量的跨component非法partition；多Tile independent components形成普通Cartesian
product。

actual apply不消费edge bool/action：caller给出的node group partition直接从immutable source SSA构造最终CardModule。singleton走
Q50.C，multi-node group按一个或多个sink materialize exact iterator tile，并在同一function/TileRegion中递归tile/fuse全部内部producer。
共享producer tile cache只活在该function和current IR epoch，fanout/diamond的相同producer slice只生成一个actual version；跨group
structured producer仍是DDR function boundary。producer同时observable并供下游使用时，等值attr/SSA-constant `OpFoldResult`按
folded integer比较并复用同一actual tile，不会生成第二个producer。测试证明chain maximal与中间cut分别产生1/2个region和1/2个output boundary store，
direct fanin、两个producer经`tensor.insert_slice`汇合、fanout和diamond均为一个actual region，emitted-node集合与group exact相等。

旧`CompleteTraversal.cpp`及其undefined rank/connection action owner已删除；`CoupledFusion`不再是current edge identity或test default，
低层temporal donor按实际动作改名`RecursiveProducerTiling`，public search/group domain不消费它。fresh host unit 752/752；随后
observable-cache修正相关Q50/Baseline/CardModule conversion 71/71，source/IR/dependency organization和轻量public search routing通过；本checkpoint未运行
重型LLaMA search。

## Q50.E：Complete Temporal Tiling

### Pipeline contract

Pipeline position:
- Upstream IR / input:
  immutable structured TensorProgram、Q50.B selected spatial assignment/trial和Q50.D explicit node-group partition；每个node已有static
  local iterator box，actual group boundary已经固定，但尚未出现temporal wave loops。
- Current stage responsibility:
  为每个scheduled node产生覆盖全部iterator的positive tile-size vector，并为实际产生多wave的iterator产生canonical loop nesting
  permutation；domain惰性覆盖全部有限整数size与语义不同order。调用方选定完整assignment后，Q50.C/D group materializer使用共同
  TilingInterface leaf mechanics物化compact prologue/steady/tail traversal。
- Output IR / files:
  query输出typed temporal assignment，不写IR；apply直接更新当前actual CardModule中的group traversal并只保留真实loops/tails。
  assignment不携score、capacity verdict、failure history、materialized body或默认layout/buffer/movement。
- Downstream consumer:
  Q50.F按selected spatial/group/temporal scope计算pure lower bound/deferred facts；Q50.G–K继续物化representation、movement、buffer和
  schedule，complete candidate最终只进入一次Q50.0。
- User-level driver / named pipeline:
  只由public search session静态组合；baseline deterministic fallback与search domain复用同一local-extent、breakpoint和actual leaf
  owner，但baseline只沿预定义单调顺序推进，不进入domain enumeration。
- Explicit non-goals:
  不选择layout、movement、retention、buffer、worker、schedule、cost或winner；不从allocator failure私下`tile/2`；不保存fixed seed、
  maximum-fit或feedback history；不运行packer/probe，不clone/replay actual IR。
- Done criteria:
  parallel、single/multi-reduction、remainder、scalar及不同local extents的lazy domain与independent reference完全一致；inactive one-wave
  iterator不制造等价order state，active iterator permutation完整；baseline与search共享breakpoint/workset函数；selected assignment
  在singleton/coupled actual region中形成compact loops和finite tails；旧fixed temporal/rank feedback owner在能力迁移后删除。

### 机制

每个 scheduled structured op 的 temporal assignment 是覆盖全部 iterator 的完整向量，加上Q50.D已选traversal内部有限、
语义可区分的wave-loop nesting/order。每个 spatial mapping 从完整per-Tile local extent开始，惰性枚举所有会改变实际
hardware work或legality的有限breakpoint与order；需要
layout、buffer或implementation facts的breakpoint是对这些typed assignment参数化的mechanism查询，不得用未选default：

- `ceilDiv` wave 数、tail 和 divisor；
- native issued geometry、padding 和 physical work；
- DDR/SPM/NoC transaction、descriptor 和 layout footprint；
- implementation geometry、reduction split 和 buffer-count feasibility；
- immutable analysis证明的must-coexist lower-bound breakpoint，以及complete assignment经Q50.0产生的exact capacity breakpoint。

只有能证明生成相同actual traversal、reuse、lifetime、tail且整数wrap/overflow语义的reduction order一致的permutation才能canonicalize；浮点reduction reassociation是supported numeric variant，不因order差异阻止canonicalize。Q50.E
不在内部选择load/receive hoist或retention winner；Q50.H在已选wave-loop order下生成这些movement alternatives。

容量失败可在其它轴固定的 branch 内产生二分子状态，再补入区间内所有非二次幂 breakpoint；二分只用于发现 breakpoint，
不能把最大可放下 tile 固化为全局事实。改变 traversal、layout、buffer、placement 或 Q/other iterator tile 后必须重新判断。
seed 只影响顺序，不限制域；late allocator 不得在 accepted candidate 上私下 `tile/2`。

Q49.P的functional fallback与Q50.E必须复用同一policy-free legal-breakpoint和workset推导机制，不能各维护一套tiling事实。
Q49.P只沿其中预定义的deterministic feasibility全序从完整local extent走到第一个fit，保证`none`功能闭环；Q50.E在Q51 state中
展开所有语义可区分的vector、nesting和order，并受其它坐标参数化。baseline选中的“第一个fit”不是全局最大可放下事实，也不
剪掉Q50.E的其它temporal completion。

### Gate

- parallel、reduction、多 reduction axis、tail/padding 和不同 shape breakpoint 有 complete-domain property test；
- independent tiny reference enumerator与mechanism生成域的stable key集合相同；涉及尚未施工轴的property
  test必须传入显式typed test assignment；
- Q49.P canonical fallback与Q50.E mechanism对相同closed coordinates产生相同legal breakpoint、workset和exact fit结论；
  baseline路径只是完整domain中的确定性功能路径，不是第二套generator或合法域限制；
- reduction 的 numeric-order legality 由 current policy/IR 证明；
- 至少一个tiny case证明相同tile vector的不同wave-loop order会形成不同actual nesting且两种选择均可达；由layout、movement、
  buffer和resource assignment才决定的reuse/SPM/global-winner影响留在Q51跨轴gate，不在本轴伪造默认坐标；
- 删除 fixed temporal seed domain 和 allocation-feedback candidate family 的选择责任。

“小tile使fusion/buffering成为winner”及任何需要actual layout/buffer/resource cost的比较统一放在Q51 closure。

### 2026-08-19 完成结果

`TemporalNodeDomain`从Q50.B placement的per-node local iterator extents出发，直接在assignment上推进全部positive size vector；
每个vector只为实际多wave的iterator枚举全排列，one-wave iterator不进入identity。2x3 tiny domain由不调用production
successor的双重循环与`next_permutation`独立得到同一8个stable points，spatial remainder仍使用Q49.P与search共享的
`deriveLocalIteratorExtents`，没有第二套shape恢复。

selected `CardTemporalAssignment`随Q50.D group partition一起转换成node-id绑定的conversion request；Card assembly验证同一node
跨Tile assignment一致、rank/positive size/order合法，再只对actual shard上真正多wave的维度建loop。`TemporalWaveLoop`成为
baseline output traversal与new singleton/coupled/partial apply共同的prologue/steady/tail owner；leaf继续只用
`StructuredIterationTile`和producer fusion。完整spatial contribution在wave leaf中组装partial tensor，原始DPS init仍只由最终
merge region消费。实际测试覆盖5x7 parallel tail、2x10 reduction accumulator、两reduction轴同vector的两种nesting、scalar、
multi-result、coupled chain及spatial partial-reduction contributions。

review同时发现并修复两类边界错误：`StructuredOpTemporalTile`新增order后，Card preparation、candidate clone与selected-edge clone
最初只复制size vector，现均原位复制order；configured internal producer此前固定按result/reduction自然序，现分别按显式
parallel/reduction相对顺序构造。若一个内部producer选择reduction-before-parallel而当前coupled consumer traversal要求相反外层，
该group assignment明确失败并保留Q50.D cut sibling；不能静默重排，也不为兼容而组装、销毁再重建一份actual producer。

实现没有引入allocator反馈、fixed seed、maximum-fit cache、probe、candidate clone/replay或默认统计。Q49.P的确定性capacity路径仍
只沿共同合法域中的单调functional顺序前进，search域不受其第一个fit限制；重型LLaMA search继续等待Q51整条链闭合。

## Q50.F：Scoped Feasibility Analysis

### Pipeline contract

Pipeline position:
- Upstream IR / input:
  immutable current TensorProgram/CardProgramAnalysis、Q50.B exact logical shard trial、Q50.D group assignment、Q50.E temporal assignment、
  显式target memory facts，以及partial-state owner明确列出的尚缺physical coordinates；没有actual CardModule。
- Current stage responsibility:
  只从上述current IR和typed assignments计算单buffer必占logical payload下界、non-binding lowering residency estimate、
  `deferred(required coordinates)`、exact lower-bound rejection或indeterminate；exact witness绑定node、Tile、group、spatial shard与
  temporal vector/order。
- Output IR / files:
  不写IR或文件；返回typed `ScopedFeasibilityResult`。query结束后只留下value facts，不留下operation pointer cache、scratch module或
  materialized candidate。
- Downstream consumer:
  Q51 partial-state transition只对`Deferred`展开明确缺失的Q50.G–J坐标，只对`ExactRejection`使用同一causal key的no-good；
  complete assignment仍只由Q50.0签发actual accepted/rejected/indeterminate。
- User-level driver / named pipeline:
  public `search` session在Q51 closure后调用；`none`不进入candidate analysis，继续只走Q49.P→Q50.0。
- Explicit non-goals:
  不选择layout、movement、buffer、route、worker或schedule；不clone/lower IR，不调用packer，不从estimate签发legality，不修改
  parent state，不生成retile/spill/rebuffer sibling。
- Done criteria:
  pure query不改source；缺坐标、unsupported footprint与single-buffer exact overflow可区分；estimate overfull不能reject；exact
  rejection绑定完整causal key且更小temporal sibling仍可达；active search owner中没有allocator repair/feedback beam。

### 机制

本checkpoint只建立从immutable IR、target facts和显式partial assignment可重算的scoped analysis，不物化actual candidate IR。
cheap footprint只能返回proven must-coexist lower bound、non-binding ranking estimate或
`deferred(required coordinates)`；只有第一种超过capacity才能拒绝状态。估算可放下不能证明actual packing可行，未证明
coexistence的估算超限也不能exact reject。近似或未经boundary-faithful proof的solver/constraint结果只可排序；exact局部
solver只对其准确建模的纯query子问题返回proof。返回：

```text
proven scoped lower-bound facts
| deferred(required coordinates)
| exact rejection(reason, causal assignment key, conflict witness)
```

analysis不选择下一tile、不修改state，也不clone、lower、spill、retile或rebuffer IR。坐标未闭合时只能返回
`deferred`，不得用default layout/carrier/buffer补齐，也不得拒绝spatial/region/temporal parent。Q51对
`deferred`只会继续生成required-coordinate transitions；对exact rejection才可在同一parent上使用causal no-good。
只有结论依赖的全部轴都包含在typed key且IR epoch未变化时才能memo；不缓存materialized IR。packer的真实结论只来自complete
candidate的一次Q50.0；`ResourceExhausted`或internal failure返回indeterminate并保留parent/siblings。

### Gate

- cheap lower-bound rejection、deferred missing-coordinate与indeterminate分别可诊断；actual accepted、actual packing failure和
  unsupported lowering只由complete candidate的一次Q50.0报告；
- estimated-fit/actual-fail与estimated-overfull-but-unproven-coexistence反例证明estimate不签发legality；
- exact failure回共同parent并保留不同temporal/traversal以及required-coordinate siblings；这里只证明candidate set继续，
  不在本checkpoint选出跨轴winner；
- source IR无变化且analysis不创建clone；任意时刻最多一个complete candidate的live actual owner；
- Q49.P与Q51都只让complete actual CardModule进入Q50.0；Q50.F只增加纯query lower bound、deferred坐标和causal taxonomy，
  baseline不进入candidate set；
- 删除 allocator repair 和 feedback beam，保留 packer、conflict witness 与 exact validator。

### 2026-08-19 完成结果

current `ScopedFeasibility`直接接收program epoch、Q50.B trial、Q50.D/Q50.E domain与assignment、显式`TargetMemoryPolicy`和调用方
列出的missing coordinates。它不建立默认memory policy或physical assignment。对每个selected node/Tile shard，query从actual
temporal leaf vector和structured indexing map计算每个operand/result单独logical payload，取其中最大值作为“至少一个buffer必须
容纳”的boundary-faithful下界；不同value、node或group的bytes不相加，因为layout alias、movement和lifetime尚未选择。
现有lowering residency estimate仍可返回给后续排序，但明确标为non-binding。

返回类型区分`LowerBound`、`Deferred`、`ExactRejection`与`Indeterminate`，并用typed reason区分missing coordinates、unsupported
footprint和minimum-footprint overflow。exact rejection witness包含node、Tile、完整group、spatial offsets/sizes、temporal vector/order
及capacity；同一full-tile overflow后更小temporal sibling重新query为deferred，证明结果没有写回parent或形成全局maximum-fit。
complex element footprint在没有physical facts时返回indeterminate，不用zero/default代替。

unit直接比较query前后source文本，证明没有IR mutation/clone；另有`estimate > capacity > proven minimum`反例保持deferred，
single FP16 buffer最小payload超限才exact reject。Q51 Core既有测试继续证明exact rejection不删除accepted sibling、deferred与
indeterminate留在unresolved set。active Planning/Search中不存在allocator repair、feedback beam或packer调用；actual packing与
unsupported lowering仍只属于Q50.0 complete-candidate gate。

## Q50.G：Layout and Physical Representation

### Pipeline contract

Pipeline position:
- Upstream IR / input:
  immutable selected TensorProgram、Q50.B exact node/Tile shards、Q50.D group partition、Q50.E temporal leaf shapes及current Wafer
  physical encoding interface；没有movement、buffer或schedule assignment。
- Current stage responsibility:
  为每个node/Tile的实际shaped operand use和result value建立primary physical-version domain；只枚举encoding interface对该leaf
  shape可完整解释的layout family。selected apply在最终TileRegion构造时物化primary version，target compute额外需要的layout形成
  显式derived `wafer.tile.materialize_layout`，不改变primary identity。
- Output IR / files:
  query输出typed per-use/result `CardPhysicalRepresentationAssignment`；apply输出带exact MemoryAttr encoding和显式layout conversion的
  current CardModule。assignment不携movement、lifetime、score、proposal ordinal或actual IR handle。
- Downstream consumer:
  Q50.H以producer primary version、consumer operand version和exact demand生成movement/conversion组合；Q50.F footprint、Q50.I lifetime及
  Q50.J calendar在representation变化后重算。Q50.0只验证complete selected IR。
- User-level driver / named pipeline:
  Q51 closure后的public `search` session；baseline仍由Q49.P current deterministic representation构造，不进入本域枚举。
- Explicit non-goals:
  不选择route、spill、recompute、retention、buffer、worker或winner；不运行PBQP/top-k proposal，不按local conversion bytes冻结layout，
  不把layout写回logical exact-demand或edge action。
- Done criteria:
  domain与independent value-layout Cartesian reference一致；unsupported encoding无候选；selected operand/result conversion、output
  writeback、fanout primary sharing及malformed assignment atomic failure有actual证据；旧edge layout字段、fake conversion action、
  late default selector和dormant PBQP implementation/test删除。

### 机制

对每个 scheduled value 枚举 producer/consumer/implementation 可接受的 `MemLayout`、physical encoding 和 storage
representation。representation 不一致时产生显式 local conversion、transfer conversion 或 spill/reload transition；conversion
的typed representation/version/padding与boundary obligation进入state，后续movement和buffer由Q50.H/I显式选择，
lifetime从最终region、movement、buffer和order assignment重算，不作为重复state字段。

fanout 的同一 physical version 必须可被多个 consumer 共享；只有确需不同 representation 时才创建派生版本。layout mechanism
返回全部合法候选和 materializer，不按局部 conversion bytes 选 winner。

### Gate

actual IR有不同layout、conversion和fanout version-sharing witness；negative tests覆盖unsupported encoding、遗漏
conversion、错误alias/version，且不同layout assignment会使相关SPM analysis/probe cache失效。通过后删除
layout independent selector和late default assignment；lowering只验证并消费selected typed representation。layout使相同
temporal tile的SPM legality或global winner改变是Q51 closure gate。

### 2026-08-19 完成结果

`CardPhysicalRepresentationDomain`按stable `(Tile,node,operand|result,index)`顺序建立惰性Cartesian domain。operand是实际payload
读取use；未读取DPS init不伪造version。每项shape来自Q50.B shard与Q50.E最大actual leaf经structured indexing map的精确像，
`PhysicalLayoutRelation::create`逐项验证current `MemoryAttr` encoding的footprint、padding、alignment与logical-to-physical relation。
current encoding合同没有独立于`MemLayout`的第二个可选schema，因此domain直接覆盖`Tensor/NTensor/Cx/NCx`中全部可解释layout；
vector-element等unsupported encoding使domain fail closed。

selected assignment随Q50.D group request进入final function conversion。Body emitter在每个structured op前临时暴露该consumer
operand的selected version，target compute若需要不同layout便从它物化derived version；转换后恢复producer primary map，避免一个fanout
consumer覆盖另一个consumer的source identity。structured result转换完成后只保留selected primary version供后续use，natural compute
output仅作为其定义链。function output仍显式转换到外部Tensor DDR layout。Cx result正例产生Tensor→Cx primary与Cx→Tensor output
writeback；fanout正例中一个Cx producer primary被两个consumer共同引用且relation只记录一个producer version。

旧`SpatialEdgeStrategy::{hasLayoutAssignment,producerLayout,consumerLayout}`、不消费layout字段的`LocalPhysicalConversion` action和
shape-driven `assignPhysicalLayouts`已删除；logical edge plan恢复只携带exact demand。dormant 1587行
`LayoutMovementOptimization.cpp`的PBQP、fixed proposal cap、local cost winner、clone/apply owner及921行专属测试同批删除，
`CompleteRankMaterialization`中的proposal ordinal入口也退出。malformed assignment在构造CardModule前拒绝且source不变。

fresh Q50.G/Q50.C–E/legacy conversion定向unit 69/69、source/IR organization和主构建通过。layout影响SPM legality与global winner的
跨轴比较仍归Q51；本checkpoint不运行LLaMA search。

## Q50.H：Explicit Data Movement

### Pipeline contract

Pipeline position:
- Upstream IR / input:
  immutable selected TensorProgram、Q50.A exact per-destination demand/ownership、Q50.B placement、Q50.D group boundary、Q50.E temporal
  order、Q50.G source/destination primary physical versions及current card topology；尚未选择buffer slots或event schedule。
- Current stage responsibility:
  生成每个exact data use的retained、same-region refetch、DDR、recompute、peer fragment、multicast及partial-reduction gather完整有限域；
  每个peer选择携带logical/physical payload和simple route。selected apply直接生成layout conversion、load/store、send/recv/await及
  intermediate relay TileRegions。
- Output IR / files:
  query输出typed movement assignment与可重算reuse facts；apply输出current CardModule中的explicit movement/event-producing ops，
  不写sidecar、route attr bag、cost或winner。route通过逐hop ops存在于IR，不靠C++ assignment跨stage保活。
- Downstream consumer:
  Q50.I从actual waves/movement completion推导slot domain，Q50.J从send/recv/await、compute与resource effects构造calendar；complete
  candidate最终由Q50.0做message matching、completion、SPM/DDR和runtime gate。
- User-level driver / named pipeline:
  Q51 closure后的public `search` session；baseline继续使用其确定性DDR/peer functional carrier，不进入本域枚举。
- Explicit non-goals:
  不选择buffer count、worker、issue order、pipeline或winner；不按bytes/hops冻结最大broadcast，不缓存actual IR，不保留旧rank/provider
  proposal或implicit cross-region SPM residency。
- Done criteria:
  local/refetch/DDR/recompute、multi-piece partial overlap、all-simple routes、layout-changing peer、unicast/partial/maximal multicast、fanout及
  partial-reduction gather均有actual witness；coverage/route/message/completion负例受typed validation；reuse facts不进入identity；旧
  complete-rank/global-relation/NoC provider与未注册tests在能力迁移后删除。

### 机制

从Q50.A exact logical demand、Q50.G representation、current placement/topology，以及已选retention/release与boundary
obligation枚举；movement assignment形成后再重算lifetime，并使相关Q50.F/J analysis失效：

- query-local relation-derived reuse analysis从exact demand、selected placement、TileRegion/traversal、Q50.E wave-loop order与
  representation推导spatial-demand equivalence/invariance classes、temporal-wave invariance classes和exact payload/coverage；
  它可失效、可重算，不压成aggregate bool attr，也不输出movement winner；

- 已选**同一 TileRegion**内的retained-value reuse、region-local recompute或解除retention；retained reuse不是
  独立SPM-residency轴，它只是region/traversal assignment及最终lifetime/allocation联合证明的结果；
- 跨TileRegion或retention解除后的same-Tile refetch/recompute与显式boundary movement；不允许不同TileRegion共享
  隐式SPM驻留；
- explicit SPM movement、card-shared DDR spill/reload；
- partial peer transfer、multicast、gather/reduction 及已定义 collective；
- representation conversion 与 movement 的合法组合。

reuse analysis只改善候选顺序与构造效率；direct/refetch/unicast、partial或多维multicast/broadcast、same-region
load/receive外提与retain/release等全部合法siblings仍由本mechanism生成。最大broadcast不得提前删除在NoC contention或后续
schedule下更好的partial broadcast/unicast。

每种 action 物化明确 participant、logical/physical payload、route/link、destination version、join/completion obligation 和
resource work。route域必须有finite normal form：默认只枚举verifier-legal cycle-free/simple physical path和有限collective
decomposition；若某个typed transport semantics确实需要重访link，必须由其显式有限bound扩展，不允许任意环路。

route或collective子求解器只能在future-boundary-faithful时返回exact Pareto alternatives或no-good：boundary
必须包含payload/coverage、physical version、join/completion、lifetime/staging、directed-link/resource-order和会影响
后续calendar的保留信息。只按local bytes/latency/hops得到的Pareto集只能作proposal排序，不能剪掉其它
route，也不能向Q51返回局部route winner。

### Gate

local、partial overlap、disjoint、multicast/collective、spill/recompute和fanout共享均有actual witness；至少一个case中
maximal broadcast因NoC contention败给partial broadcast或unicast；coverage、payload、
participant、route、join、premature completion、非finite route和跨region隐式retention有负例。通过后逐项替换
`StructuredDAGEdgeStrategyPlan`的选择责任和late movement repair；compatibility carrier仅在仍被actual lowering消费时
保留。carrier尚未可构造时返回`deferred(required coordinates)`；在已赋值representation/movement下表达失败只能
拒绝该physical action assignment，不能拒绝Q50.A logical demand或Q50.B spatial placement。“局部movement较贵但允许
后续fusion/overlap而成为global winner”是Q51 closure gate。

### 2026-08-19 完成结果

`CardDataMovementDomain`从Q50.A satisfied demand与producer ownership直接建立stable `(edge,destination Tile)`项；每项保留Q50.G
consumer version和按owner区分的source/transport layout、logical bytes、encoding-owned physical bytes与rectangle。local exact demand在
同一Q50.D group中只有retained；同region解除retention形成显式SPM→DDR→SPM refetch；group cut提供DDR或pure producer recompute。
remote/mixed ownership提供Peer，每个remote fragment的route由`SimpleRoute`在assignment上惰性推进全部available-Tile simple path，
不预存path vector或只保留shortest route。

相同edge/payload的peer destinations再惰性枚举全部set partitions：singleton保持unicast，任意size≥2 group形成partial/maximal
multicast。只有selected routes的union形成单parent rooted tree时该point合法；apply在source、intermediate relay和可同时消费/forward的
destination上各收发一次共同payload，因此maximal multicast actual send数少于同route unicast。不同source/destination layout先显式
materialize transport version；local+remote overlap保留local subview load并只接收missing fragments。每个hop使用独立round，所有
send/recv后立即`async.await`，cycle、重复parent、payload/coverage/route不一致均被assignment membership拒绝。

spatial partial reduction另有typed gather项：每个contribution result/source Tile与merge input relation在actual conversion时显式记录，
Peer gather用独立communication identity把remote partial送到merge Tile并保留local contribution。施工时由此发现Q50.C旧partial
contribution和merge只append output argument却未写回destination；现三个partial路径统一用full-result destination binding，fresh
测试直接证明4个contribution store加1个merge output store，peer gather再删除被替代的remote DDR边界。

query-local `DataReuseFact`从current consumer indexing map给出temporal-invariant iterators，并按exact producer set列出spatially
equivalent destinations；它只供proposal/构造复用，不删除unicast/refetch/recompute siblings。fanout primary version继续由Q50.G共享。

旧`GlobalTileRelation`、duplicate execution/collective topology、complete-rank/rank-tile residency、old collective lowering、
coordinated/NoC/intermediate/partial providers及其未注册tests均在对应relation、route、multicast、relay、partial gather和negative proof
进入active owner后删除，source checker allowlist同步收缩。fresh Q50.C–H/legacy conversion定向unit 64/64、lit 216/216、source/IR
organization和主构建通过；没有运行LLaMA search。contention下broadcast/route/global winner仍由Q51共同轴证明。

## Q50.I：Buffer and Rotating Slots

### Pipeline contract

Pipeline position:
- Upstream IR / input:
  immutable selected TensorProgram、Q50.B exact node/Tile shards、Q50.D coupled groups、Q50.E temporal leaf vector/order、Q50.G
  physical representation、Q50.H exact movement assignment及调用方显式提供的target SPM capacity；尚未选择event/resource schedule。
- Current stage responsibility:
  按coupled-group内的exact logical edge子集建立serialized或rotating-slot候选，有限slot上界只由actual steady-wave count与单slot
  exact physical payload的capacity下界推导；selected apply把每个独立scope变成actual allocation family、SSA rotation及SCF
  prologue/steady/epilogue，并从current Instr关系证明producer、movement endpoint、consumer和最后use位于同一static loop。
- Output IR / files:
  query输出typed `CardBufferingAssignment`，其中每个scope只含Tile、group nodes、被pipeline的edge集合和slot count；apply输出
  current Instr IR中的独立allocation、loop-carried pointer rotation与显式release。query结果不含IR pointer、location、score、
  lifetime cache或默认统计。
- Downstream consumer:
  Q50.J从actual Instr control/effect和selected buffering重建ready/resource calendar；Q50.0 final gate在slot物化后重新执行SPM allocation、
  message/completion和ABI验证。
- User-level driver / named pipeline:
  Q51 closure后的public `search` session；baseline保持serialized且不枚举buffer域。
- Explicit non-goals:
  不选择worker、issue order、route、winner或overlap收益，不clone/lower partial state，不把slot count写回logical edge carrier，不用3作为
  永久上限，不把materializer失败升级成parent spatial/temporal rejection。
- Done criteria:
  lazy domain覆盖single/double/triple及wave/capacity允许的更多slot；selected scope可转换为exact per-Tile request；actual
  double/triple/multi-slot、tail、Direct-DTE issue/wait、alias与共同循环正负例通过；旧edge `bufferCount`、无edge扫描入口、
  local buffer winner和固定`[2,3]`上限删除。

### 机制

buffer count 是 op-wave/edge pipeline 的共同候选维度。domain 至少表达 single/double/triple buffering；若 current IR 与硬件
允许更多 slot，则以并发 wave、lifetime 和 capacity 推导有限上界，不能把 3 作为未证明的永久 cap。候选只携带可独立物化的
coupled-group scope、exact logical-edge子集和slot count；rotation、producer release、最后async consumer及
prologue/steady/epilogue不是另一份shadow recipe，而是selected apply从actual allocation、SSA use、effect、wait和static loop直接构造并验证。
aligned footprint和actual lifetime由current immutable IR borrow与region/representation/movement/order assignment重算。

buffer mechanism 只生成 serialized/overlap-capable actual alternatives；是否真正 overlap 由 Q50.J/K resource schedule 决定。
无法证明 buffer 独立或无 alias 时不生成对应 transition。

Q54的current-IR gate已经暴露一个必须由本项处理的结构前置：只有producer、consumer及其message endpoint确实位于同一个
static `scf.for`时，现有rotating-slot materializer才有准确的iteration、release和reuse边界。若producer在逐row循环内、consumer
在循环外消费完整assembled result，则“逻辑edge相关”不等于“存在可pipeline的共同循环”；此时必须拒绝buffer候选。Q50.I应先
把共同wave/stage loop物化到actual IR，再在该loop上生成slot rotation，不能恢复Location/provenance或上游op关系作为循环证据。

### Gate

single、double、triple、tail reuse、capacity failure、alias hazard、issue/wait和premature slot release有显式typed
assignment与actual Instr正负测试；buffer recipe改变会进入state key，并使SPM/resource analysis cache失效。删除
buffer feedback family/local buffer winner，保留selected-buffer materializer与verifier。不同buffer与schedule/stage组合是否
实现重叠及其global winner只在Q51 closure证明。正例必须检查所有exact logical-edge endpoint共享actual static loop；
循环外consumer负例必须稳定报告“no static loop containing every exact logical-edge endpoint”。

### 2026-08-19 完成结果

`CardBufferingDomain`对每个selected coupled group保留serialized identity，并惰性枚举可pipeline的retained edge非空子集与全部slot
multiplicity。slot上界不再固定为3：temporal owner给出compact traversal中真正存在的steady full-wave trip count，Q50.G selected
producer-result/consumer-operand leaf layout经`PhysicalLayoutRelation`给出单slot必须容纳的physical footprint下界，调用方显式SPM
capacity再给出安全有限上界。不同edge可有不同上界，domain按当前count只枚举仍可达edge；tail wave不被错误计入steady loop。

single-slot assignment不生成request、不运行materializer。multi-slot assignment经`BufferingApply`转换成按Tile排序的独立
`SelectedBufferingScope`；同一Tile上的不同coupled groups顺序原位物化，绝不为了寻找共同anchor把两个scope合并或clone module。
每个scope的request只绑定current DAG producer/consumer和actual retained dataflow。Q50.0在TileRegion-to-Instr和memory-preparation后，
由current buffer relation检查每个endpoint属于同一static `scf.for`，再构造独立allocation family、loop-carried pointer rotation、
SCF prologue/steady/epilogue和owner-block release；随后fresh SPM allocation验证全部slot真正可共存。

旧`SpatialEdgeStrategy::bufferCount`和baseline赋值已删除，logical movement carrier不再拥有buffering；无logical-edge request、扫描任意
loop并自行挑选candidate的overload删除，`[2,3]`硬上限删除。actual tests覆盖2/3/4 slots、wave上限、capacity上限、tail、
loop-external cross-stage write hazard、Direct-DTE issue/direct wait/final release、不同static loops稳定拒绝，以及同一Tile两个独立scope
顺序物化。query/apply均无默认统计、partial-state clone或local overlap winner；是否真正获益仍由Q50.J/K与Q51共同选择证明。
fresh Q50.C–I/baseline/executable定向unit 87/87、lit 216/216、source/IR organization和主构建通过；按计划未运行重型LLaMA search。

## Q50.J：Ready/Order/Worker/Resource Schedule

### Pipeline contract

Pipeline position:
- Upstream IR / input:
  one complete selected physical candidate after Q50.H movement and Q50.I slot materialization, represented as owned canonical unplaced Instr modules
  for all available Tiles；Q63 completion interface/analysis与current target topology是immutable facts。SPM/DDR offset、DTE binding和stage
  pipeline尚未物化。
- Current stage responsibility:
  对每个current Instr block建立SSA/effect/alias/token/completion hard dependency DAG，惰性枚举全部topological ready order与typed NCC worker
  assignments；从一个assignment重建query-local pending-event/resource calendar，并把selected order/worker原位应用到同一owned complete
  candidate，随后fresh重建participant joins。
- Output IR / files:
  query输出只在当前unchanged Instr epoch内有效的typed `CardInstructionScheduleAssignment`；apply输出actual operation order、worker attrs及
  completion joins。event/resource calendar是可销毁analysis result，不写IR attr、sidecar、score或默认统计。
- Downstream consumer:
  Q50.K在改变event structure后重新建立本域；Q51 complete-candidate inner closure枚举并选择schedule后调用Q50.0剩余SPM/DDR、transport、
  resource和ABI gate；Q52才使用calendar work/contension estimate排序。
- User-level driver / named pipeline:
  Q51 closure后的public `search` session；baseline保持其canonical worker0/source order，不枚举本域。
- Explicit non-goals:
  不clone module、不按priority选local winner、不枚举时间戳/idle、不用nominal bandwidth签发legality，不跨IR epoch保留operation handle，
  不恢复static capability/profitability registry。
- Done criteria:
  tiny independent reference逐点匹配order×worker域；selected apply、mutation失效、Direct-DTE issue/wait、multi-worker join、alias/effect、
  shared DDR及directed peer-link资源正负例闭合；旧ready-order/worker selector与`TargetSchedulingCapabilityRegistry`整岛删除。

### 机制

共同state只保存ready choice、resource order、worker、issue/wait和completion等typed assignments。ready/running/completed
op-wave、pending data/completion、per-Tile compute/SPM、每条directed NoC link、card-shared DDR和observable obligations
由current immutable IR borrow与这些assignment重建query-local event/resource analysis；calendar与makespan是可失效cache，不是state字段。
实际transition必须发布data-ready/completion，不能在最后用`max(compute, movement)`猜overlap。

schedule域使用finite normal form：枚举有限ready event、worker、dependency/resource order和从已选issue/wait形成的有限
release/completion boundary；canonical schedule builder从这些选择推导开始时间和calendar。如果需要有意延迟，必须
用明确的resource order或某个已存在release/completion boundary表达，不枚举任意时间戳或无界idle。

不同 ready waves 可在 disjoint resources 上并行；fanin/fanout、effect、alias、worker join 和 slot reuse 必须等待对应 event。
局部 schedule DP/solver 只在 boundary 完整的固定子问题上返回 alternatives、bound 或 no-good，不能持有全图 schedule 或 winner。
Q50.J mechanism必须可重入：任何region、temporal、movement、buffer或stage assignment变化都会清除旧event/resource
analysis，并从新typed assignments重新生成schedule alternatives。

Query-local hierarchical resource projection复用existing target facts，把Instr/movement assignment投影到compute unit/worker、
相关local SPM resource、directed NoC links与DDR channel/engine。资源集合相交只产生contention estimate/排序信号，不相交可
优先形成并行proposal；是否可重叠仍由actual event/resource semantics判定。Nominal bandwidth split和解析关键路径只作
estimate，不能替代actual event calendar、completion、legality或admissible bound。

current `TargetSchedulingCapabilityRegistry`以稀疏静态row同时表达pair/group legality和profitability，缺row又返回`Unknown`；它不能
成为Q50.J的新事实源。hard issue/completion/resource相容性必须来自typed target operation facts、Instr interface/effect及Q63拆层后的
completion adapter；profitability只属于Q52 profile/estimate。Q50.J接入新event calendar时删除该registry、row table和只验证
registry closure的专属tests，不增加第二份capability matrix。

### 2026-08-19 完成结果

`CardInstructionScheduleDomain`借用一个complete canonical unplaced Card Instr epoch，按Tile和block建立有限event windows。window只含
typed Instr event；derived NCC join不进入候选identity，synchronous-writeback、region/call和有effect的非Instr operation形成明确边界。
hard DAG从transitive SSA、view-root buffer RAW/WAR/WAW、Direct-DTE token/wait及synchronous completion构造。每个window使用惰性的
lexicographic topological-successor算法枚举全部ready orders，不预存permutation，也不按movement-first priority选winner；每个typed
NCC issue独立枚举closed ODS worker域，Card assignment是普通order×worker Cartesian product。

domain保存current operation/attr/operand/result-type snapshot；插入、删除、重排、worker/attr或operand/type mutation都会使assignment失效。
apply消费完整owned Tile module集合，确认同一IR epoch后原位移动operation、设置worker，删除旧derived joins并从current
SSA/effect/range/completion fresh重建minimum participant joins；失败时owned modules整体销毁，不clone source或暴露部分mutation。

`CardInstructionResourceAnalysis`从selected actual IR重算engine、NCC worker、Tile SPM、card-shared DDR和directed peer-link resource uses；
Direct-DTE/NCC issue到exact wait/join之间出现disjoint engine work时记录structural overlap witness。shared DDR/link只作为Q52 contention输入，
不签发legality/profitability，不生成时间戳或shadow calendar。

tiny independent case逐点得到`2! × 3² = 18`个order/worker assignments；fanout得到`2! × 3³ = 54`，same-buffer WAW只保留
`3² = 9`。actual tests另覆盖epoch invalidation、selected in-place order/worker/join、Direct-DTE issue→independent compute→wait、
shared DDR和双端同一directed link。旧greedy ReadyOrder、clone-based WorkerPlacement、dormant implementation model及其专属tests删除；
`TargetSchedulingCapabilityRegistry`、静态row/profitability、IR query adapter和专属tests整岛删除。无默认统计；Q50.K改变event structure后
必须新建domain。
fresh Q50.S/H–J/Q63/baseline/lifetime/cost/lowering定向unit 156/156、lit 216/216、target public link、source/IR organization及主构建通过；
未运行重型LLaMA search。feature-on SystemC transport target已注册，但当前build未启用该feature，未计入通过数。

### Gate

independent branches、fanin/fanout、shared DDR、NoC link contention、multi-worker join、effect ordering和completion hazard有
正负测试；actual Instr顺序、worker和completion可验证，finite normal form的tiny域与independent reference
enumerator一致。删除ready-order、worker和overlap的独立selector及静态scheduling capability registry；query-local calendar在对应IR epoch/assignment
失效或winner materialize后销毁，不成为shadow schedule。

## Q50.K：Conditional Stage Pipeline

### Pipeline contract

Pipeline position:
- Upstream IR / input:
  one owned prepared canonical Instr module、Q50.I selected independent buffering scopes/current buffer relations及尚未物化的Q50.J schedule；
  input无SPM/DDR offsets或DTE binding。
- Current stage responsibility:
  empty scope保持serialized identity；每个selected multi-slot scope在其exact common static loop上构造dependency stages、slot rotation及SCF
  prologue/steady/epilogue，返回actual stage/slot facts。任何stage mutation使旧Q50.J domain失效，调用方必须从新IR重建order/worker/resource域。
- Output IR / files:
  move-only `StagePipelineMaterialization`拥有同一mutated Instr module、current relations及per-scope stage/slot facts；不产生pipeline attr、
  side plan、clone、文件、winner或默认统计。
- Downstream consumer:
  Q50.J从新event structure重建schedule assignment；随后Tile memory planning分配全部rotating slots，Q50.0继续transport/resource/ABI gate。
- User-level driver / named pipeline:
  Q51 complete-candidate inner closure；baseline和single-buffer candidate走empty-scope serialized identity。
- Explicit non-goals:
  不把每条dependent edge默认pipeline，不自行选buffer count/order/worker，不从estimated overlap接受candidate，不恢复whole-Module
  fixed-slot clone API。
- Done criteria:
  serialized identity、2+ actual stages、selected multiplicity、prologue/steady/epilogue、Direct-DTE issue/wait、alias/external-write/tail负例和
  Q50.J epoch invalidation/re-entry受测；旧FixedSlotPipeline source/header/test零残留。

### 机制

operator stage pipeline不是每条dependent edge的默认模式。Q50.K builder先从以下typed facts生成有限的
pipeline event-structure transition：

1. producer/consumer 有可对应的多个 temporal waves；
2. Q50.H explicit movement 能为 consumer wave 产生独立 data-ready event；
3. Q50.I slot/lifetime 允许 producer 前进而不覆盖未消费数据；
4. effect、completion、observable和tail obligations在该event structure下均可保持。

该transition会使旧schedule analysis失效，随后可重入调用Q50.J mechanism，为serialized、local coupled traversal和
cross-Tile staged pipeline分别生成schedule alternatives并从新assignment证明compute、SPM、NoC或DDR overlap。
不得用serialized candidate已选calendar决定pipeline是否存在，也不得在Q50.K内冻结schedule winner。同一parent必须
保留上述三类alternatives。pipeline materializer形成真实prologue/steady/epilogue issue/wait/compute sequence，
不以估算overlap或`pipeline=true` attr代替。

### Gate

actual serialized、pipeline event structure、可重入schedule、resource-conflicting pipeline rejection和tail hazard witness齐全；
通过后移除multi-stage placement recipe和late overlap repair。“某case由pipeline获胜，另一case因buffer/
movement/parallelism成本由serialized或fused region获胜”放在Q51 closure。

### 2026-08-19 完成结果

Q50.I的`SelectedBufferingScope`同时是stage cut的exact edge集合，不再复制另一份pipeline recipe。低层materializer现返回actual
`stageCount`、`maximumSlotCount`和allocation count；Q50.K `materializeStagePipelines`消费owned prepared Instr及current relations，
按scope顺序原位构造stage pipeline并要求每项`stageCount >= 2`且actual slot multiplicity等于selected count。empty scopes直接move返回
原module，是serialized/single-buffer identity。

Tile memory planning改为调用该唯一composition owner，之后才为全部slot分配SPM。测试证明selected 3-slot scope产生2+ stages及额外
prologue/epilogue Instr，旧Q50.J domain在mutation后拒绝，基于新IR重建的schedule domain有效；两个独立scope继续由Q50.I memory test
证明顺序物化。Direct-DTE issue/direct wait/release、external-write alias、trip-count和tail hazard继续由同一active selected-buffer tests
覆盖，不复制旧大套件。

旧`FixedSlotPipeline`整Module clone、无logical-edge loop扫描、local stage candidate、public `SoftwarePipelining` header及2118行专属test
均在能力迁入Q50.I/J/K后删除，source checker dormant表归零。Q50.K不宣称任一pipeline获益；serialized/pipeline/fused的global winner
只由Q51完整physical candidate比较。
fresh Q50.I–K/Q50.S/baseline定向unit 36/36、lit 216/216、source/IR organization及主构建通过；未运行重型LLaMA search。

## Q51 Closure：全轴联合正确性

### 完整共同状态

Q51 closure时一个partial state只携带已选typed assignments：

```text
semantic DAG root
per-op spatial partition relation and Tile placement
single-op/coupled TileRegion boundaries and traversal choices
all-iterator temporal tile vectors and finite wave-loop nesting/order within selected traversals
value layout/encoding/physical-version choices
movement/collective/spill/recompute actions
buffer recipes and rotating-slot choices
stage/event/resource order, worker and completion choices
```

`StructuredDAGAnalysis`、exact demand、ready/running/completed classes、live physical versions、retained-value lifetime、live SPM roots、
pending data/completion events、resource calendars、lower bound和current makespan都不是state字段。它们是从current immutable IR
borrow、target facts与上述assignments重算的query-local analysis cache；任一依赖assignment变化都必须精确失效。

任何 exact failure 都回到生成它的共同 parent，允许修改任一因果轴；不存在 late allocator、layout、movement、buffer、route、
worker 或 schedule repair selector。

### Two-layer small exhaustive oracle

在2–4 Tile tiny topology和每轴2–3个真实breakpoint上，independent reference domain enumerator不调用production
domain builder，从verifier semantics平铺枚举semantic root × spatial × TileRegion/coupled boundary × temporal ×
layout × movement × buffer × stage/order/resource的全部合法typed combinations。它先与mechanism生成域逐轴、
逐parent比较stable key集合，独立证明domain coverage。

随后production-mechanism flat exhaustive runner使用同一mechanism、materializer、cost比较和exact gates展开该域，
但禁用排序剪枝；Q51 exact candidate set再与它比较：

- single op、chain、independent branch、diamond/fanout、fanin/reduction、partial redistribution；
- 每轴至少一个 winner-changing 对立 case；
- 至少包含：非最大/非矩形spatial choice获胜、小tile + fusion、maximal group败给较小cut、layout
  conversion + multi-buffer、局部更贵movement + overlap、暂时较差placement + stage pipeline；
- exact pruning 逐项开启后 accepted optimum、cost 和 stable actual IR digest 不变；
- 先展开child无论accepted还是rejected，其它合法siblings均仍可达；
- serial/parallel analysis与actual cost比较结果一致。

### Final gate

- generic mixed DAG 产生 accepted CardExecutable winner；
- ordinary attention/decode 等已物化 semantic roots 复用同一 physical search，不存在算法专用 selector；
- complete candidate 通过 TileRegion-to-Instr、fresh completion、fixed SPM/DDR、communication/resource、ABI、verification 和
  final recost；
- actual rejection 可继续 candidate set，`search` 无更优 accepted candidate 时返回同源 Q49.P accepted baseline；
- selected IR 至少有一个有效 multi-op fusion，中间 actual value resident 且没有无意义 DDR round-trip；
- public policy只剩`search|none`，且`search`只进入new-search session→Q50 mechanisms→Q50.F analysis→complete materialization→
  Q50.0→incumbent/winner链；current `deriveShortlist`、candidate/evaluator/feedback/accepted cohort、schedule-plan selector、winner
  rematerialization、旧bounded/rank owner、各轴local winner、performance `Unknown`和shadow plan均已删除；
- public routing在Q51.Core已完成，各轴旧实现已随对应Q50变更删除；
- 旧proposal数、stable ordinal、feedback统计、winner digest和异常长旧integration没有新链对照要求；new path只由独立oracle、
  actual IR、Q50.0结果与source-to-package契约验收；
- Q51 独立提交并标完成后，Q52 才开始改变 search policy 的吞吐与预算行为。

Q51在small-oracle mode完整展开有限域。真实workload从正确性闭合时就支持明确的global work/time中断，
但Q51只交付机制，不在fresh profile之前固定production默认10/30分钟截止或有损策略。中断时返回结果
必须actual accepted：若未展开completion仍由exact continuation与admissible bound完整表示，则标
`feasible-with-bound`；若某些合法completion已被丢弃或从未表示，则标`budgeted-feasible`。不得借
correctness mode在真实负载上无限运行。Q52基于fresh profile选择production budget并改善同一预算下的
time-to-first和incumbent质量。

## Q52：Profile-Driven Anytime Search and LNS

Q52开始前必须先拆掉`TargetScheduleCostPolicy`/`WaferTargetPolicy`这类跨owner aggregate：exact accepted Instr只产生可重算的
resource/work metrics；hard SPM/DDR地址、容量和alignment留在target memory owner；package/profile的reference rate留在其报告
owner；只有fresh profile证明可用的bandwidth/startup/hop/issue prior才形成独立search estimate输入。Q52不保留
`TileSearchEffort::{Quick,Default,Deep}`、固定candidate/beam cap或默认构造全局policy；新的budget/priority均从本任务的profile和
coverage等级显式产生，不能影响legality或Q51 exact oracle。

### 目标与时间合同

Q52 的生产目标不是在 10–30 分钟内证明所有真实 workload 全局最优，而是在有限预算内尽快得到合法 actual executable，
持续改善 incumbent，并在能够保留 exact candidate set 时报告可信 lower-bound gap。

- **10 分钟以内**：可接受，但仍记录 work、RSS、time-to-first-actual 和 incumbent 曲线；
- **超过 10 分钟**：不立刻停止，必须形成热点、重复状态和质量停滞归因；
- **最长先观察到 30 分钟**：收集完整 profile；若仍无法承受，才根据已测证据启用明确的 bounded/heuristic policy；
- 到预算返回的 executable 必须 actual accepted 且不劣于同源 Q49.P accepted baseline；若永久丢过合法状态，结果标
  `budgeted-feasible`，不能沿用 exact gap 或暗示最优。

预算是 policy 参数和回归证据，不是合法域定义。不得为了满足时间线提前固定 tile、fusion depth、layout、buffer count、
split、placement group 或候选数。

### Fresh profiling

Q51完整new-search链提交并通过其small-oracle/new source-to-package gate后，本节才首次允许执行重型LLaMA `search`与质量
profile；Q49.P的单次fresh FP16 LLaMA `optimization-none`只证明baseline功能/materialization，不提供本节证据。profile必须是
显式选择的bounded批次，不进入普通unit/lit/CTest或每次功能改动回归。在 generic mixed DAG、HF prefill、functional
decode 和 LLaMA representative block 上记录：

- 各轴 transition generated/rejected/deduplicated/expanded 数和 candidate set width/peak live states；
- canonical key 重复率、separator width、dominance/no-good 命中和失败作用域；
- time-to-baseline、time-to-first-better-actual，以及 1/3/10/30 分钟 incumbent actual cost/digest；
- lower bound、incumbent、gap 随时间变化；一旦启发式丢状态则停止宣称全局 gap；
- immutable closure/relation analysis、complete CardModule materialization、TileRegion-to-Instr、SPM/DDR packing、schedule、cost、verification 的调用数与
  wall/CPU；
- RSS、global work units、fusion groups、resident bytes、DDR/NoC movement、buffer 和 resource overlap；
- 每类 actual rejection，以及 candidate 是否因同一失败被重复完整 materialize。
- ordered-factorized spatial、relation-derived reuse等proposal family各自的generated/accepted/probed/full-compiled数量、
  time-to-first贡献与单位actual compile成本；
- analytic estimate rank相对final accepted recost的误差、resource-term prediction error，以及`best-found@k`、winner
  recall@k与regret@k；`k`只是一组profile横轴，不预设生产常量。

### 先做保持完备的优化

| Profile 事实 | 首选方法 | 最优性条件 |
| --- | --- | --- |
| 相同 future boundary 重复 | canonical memoization / subgraph DP | key 包含全部 future-visible facts |
| chain/tree 或稳定小 separator | boundary-compatible Pareto DP | separator 完整；alternatives 回共同 owner |
| 大量重复 actual failure | scoped no-good cache | key 只排除同一因果 assignment |
| incumbent 与 admissible lower bound 分离明显 | branch-and-bound / strict dominance | bound admissible；dominance future-compatible |
| 固定局部 order/route/packing 组合昂贵 | local exact DP/solver | solver 模型与边界 faithful，不作全局 winner |
| actual lowering 重复 | immutable analysis、causal no-good与直接保留accepted executable | analysis key完整；不缓存materialized IR，不重建winner |

这些方法在条件满足时不改变 small oracle optimum。solver 的 `OPTIMAL`/bound 只属于被准确建模的局部子问题；`FEASIBLE`、
timeout 或 resource exhaustion 只能提供 proposal 或当前子问题未决，不能升级成全局 rejection。

### 推荐 anytime 架构

```text
Constructive lane:
  deterministic fusion-oriented best-first candidate set
  + ordered-factorized regular spatial proposals
  + relation-derived reuse / movement proposals
  + baseline incumbent
  + canonical memo / scoped no-good / proven dominance

Improvement lane:
  incumbent-driven coupling-aware LNS
  + local exact DP/solver repair when boundary-faithful
  + profile-justified diverse candidate set when needed
  -> complete materialization/Q50.0 gate
  -> only update the same global incumbent
```

Constructive lane首先保证尽快得到合法actual incumbent；small-oracle mode要求公平展开全部未证明无用的状态，
并可给出`optimal-certified`。真实负载若保留完整exact continuation和admissible lower bound，单纯因budget中断可报告
`feasible-with-bound`；启用LNS-only、fixed/diverse candidate set或其它会永久跳过合法状态的Improvement policy后，
才只报告`budgeted-feasible`。production不默认要求保留exact candidate set，但必须按实际保留情况标注等级。

LNS 的 destroy/repair 必须按耦合关系选 neighborhood，而不是再次逐坐标下降：

- 一个 coupled group 的 placement、temporal、layout、region boundary/retained values、buffer 和 order 一起释放；
- fanout producer 与全部 consumer edges 一起释放；
- 共享 DDR/NoC link 或 critical-path resource window 的 events 一起释放；
- stage pipeline 的 producer/consumer waves、slots、movement 和 worker 一起释放。

repair 使用相同 mechanism、materializer和exact gates；不存在“修成能跑即可”的隐藏 compatibility selector。不同 neighborhood
size、选择策略和 repair budget 必须由 profile 与 small oracle regret 曲线决定。

### 何时允许 diverse candidate set 或其它 trade-off

只有从actual profile确认candidate set/RSS不可承受，才启用deterministic diverse candidate set或其它有损策略：

- 普通 fixed-width beam 会永久丢掉暂时估算较差、但靠后续 fusion/layout/buffer 协同成为最优的状态；结果只能
  `budgeted-feasible`；
- 若采用可回溯 beam/iterative widening，只有所有被延迟状态最终得到公平展开时才能恢复 complete 声明；
- candidate set 必须按semantic root、critical structural class和baseline coverage保留多样性，但coverage规则不等于质量保证；
- fixed top-k只能在`best-found@k`、winner recall/regret和actual compile成本实测后启用；论文或外部系统的`k`不能直接成为
  本项目默认值，任何永久丢弃合法completion的top-k结果都标`budgeted-feasible`；
- 模拟退火或遗传算法只有在LNS/candidate set profile仍显示明显local basin且actual gate budget允许时才研究，并只作为
  proposal mechanism，不作为默认生产 owner。

### 会丢好解的高风险剪枝

以下规则未经 small oracle 和证明不得进入 exact lane：

- 按单 op、单 edge、单 layout 或单 route 的局部 winner 冻结轴；
- 只保留当前 estimated makespan 最小的 placement/tile；
- 在 layout/buffer/fusion 未固定时缓存“最大可放下 tile”；
- 用 aggregate makespan、bytes 或 peak SPM 合并 live versions/calendar 不同的状态；
- 在 actual packing 失败后对 accepted clone 原地 retile/spill；
- 把solver restricted-model bound当作global physical-dataflow bound；
- 用 workload/op/shape 名称、固定 `{2,4,8,16}`、`128` seed 或历史 winner 缩域；
- 关闭暂时不改善 incumbent、但需要两步以上协同变换才能进入的 basin。

### Q52 gate

- small exhaustive oracle 上 exact mode 的 optimum/digest 与 Q51 相同；
- 每种启发式分别报告 small-oracle regret、真实 workload incumbent 曲线、work、wall 和 RSS；
- 10/30 分钟 profile 可复现，代表 workload 返回 actual accepted 且不劣于 baseline；
- 输出明确区分 `optimal-certified`、`feasible-with-bound`、`budgeted-feasible`；
- 至少证明一个 LNS neighborhood 能找到当前逐坐标/窄 beam 会遗漏的协同 winner；
- 不把实际 profile 得出的排序 prior 重新写成合法域限制；
- Q52 独立提交后标完成，Q53 才签发 production 证据。

## Q53：Production Board Readiness

### Host / property / integration

- Card/Tile verifier、coverage、message matching、SPM ownership、completion 和 ABI；
- 每个 semantic root 的资格与 actual DAG materialization；
- spatial、TileRegion、coupled traversal、temporal、layout、movement、buffer、schedule 和 stage pipeline 的对立候选；
- ready-set concurrency、branch/fanin/fanout、resource hazard、actual rejection cleanup、determinism 和 small oracle；
- generic GEMM、elementwise、reduction、conv/mixed DAG 的 distinct MPMD Tile modules 与 package readback。

### Source / package / no-card

- 全部case由Q60产品adapter或pre-exported portable StableHLO入口产生，并进入同一Q59 compile transaction；
- official HF prefill FP16/BF16；
- functional two-step KV-cache decode FP16/BF16；
- Llama-2 7B representative block FP16/BF16；
- source 保持原始 HF/PyTorch 语义，不添加 mask、`-inf`、shape 或 decode 特判；
- 每个 case 提供 source、case-owned CPU oracle、deterministic runner、current package 和 fresh no-card；
- source、metadata、payload 和 expected 的 dtype/shape 一致；host build/test 按 `nproc` 并行。

### Fusion effectiveness gate

每个要求融合的代表 case 同时证明：

1. selected actual IR 存在至少一个由共同 iteration 与 retained-value dataflow 物化的 multi-op TileRegion/traversal；
2. 中间 value 的 actual lifetime 和 offset 证明其驻留 SPM；
3. 不存在对应中间 value 的无意义 DDR spill/reload round-trip；
4. fused 与同源 `none` baseline 数值一致；
5. actual work 和 matched board A/B 没有因过小 tile、额外 movement 或并行度损失而退化。

只统计 fused-edge 数、检查 group 字段或观察 pass 成功均不满足 gate。

### Board-ready 与 done

- 无板阶段完整生成全部 current package、oracle 和 runner，并逐 case fresh no-card 后才标 Q53 `board-ready`；
- 真实板端单进程串行执行，不读取、回放或重新判定历史输出，不自动 retry/reset；
- Llama 及至少一个 prefill/decode 代表执行同源 matched `none`/`search` A/B，并校验 exact output/guard；
- 获得可重复实际性能改善后 Q53 才标 `done`，同时解除 Q48 的前置阻塞。

## 提交与收尾

1. Q58、Q56、Q50.0、Q54、Q59与Q50.A已达到各自当前门禁；随后Q49.P、Q51.Core、Q50.S、
   Q50.B–Q50.K、Q51 closure、Q52、Q60与Q53分别形成独立可评审提交；不得把全部迁移积累成一个dirty diff。
2. Q51.Core同批建立新control、改接public `search`并删除旧控制链；此后Q50只施工和验证新链，每个轴落地时同步删除对应旧
   mechanism/selector/repair/test，不运行旧search，也不维护行为对照。Q51只闭合完整new source-to-package结果和旧实现零残留。
3. 状态转换以 `tasks/progress.md` 为准；本计划不单独维护第二份动态状态表。
4. 每项提交前运行 fresh 定向 build/test；端到端或主线 gate 还需确认 relevant lit/CTest 实际执行而非 skip/unsupported。
5. 提交使用 `Codex <codex@openai.com>` 并附 `Co-authored-by: hehesnail <shashen008he@gmail.com>`。
6. 稳定 bug 模式进入 `memory/bugs.md`，可复用 build/debug workflow 进入 `memory/general_dev.md`；临时 profile 数字、
   workload 路径、未校准 prior 和单 case winner 不进入长期设计。
