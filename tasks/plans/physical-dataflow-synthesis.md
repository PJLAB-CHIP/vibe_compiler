# Physical Dataflow Synthesis 实施计划

状态：Q50.0无策略CardExecutable编译/准入边界、Q54 MLIR infrastructure、Q59 compiler entry transaction与Q50.A
production demand boundary均已闭合。当前由Q49.P在current source-to-package路径上闭合baseline functional legalization及
policy/structure/probe/materialization隔离，之后建立
Q51.Core的typed assignment、deterministic frontier、incumbent和evaluation/result control seam；随后让
Q50.S与Q50.B–Q50.K逐轴接入同一个owner，最后闭合Q51。
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
  current事实闭合baseline功能合法化及policy、结构、probe和materialization隔离；Q51.Core复用其accepted executable作为初始incumbent，
  建立只管理frontier/evaluation/result的共同control kernel，同批让public `search`只进入新Core并删除旧search控制链；此时没有
  production mechanism，`search`直接返回accepted incumbent。Q50.S与Q50.B–Q50.K逐轴交付新mechanism和独立domain oracle，
  同批删除对应旧实现；Q50.F才加入通用actual-region probe；Q51通过full production flat exhaustive runner、complete
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
  -> isolated actual clone/materialization/probe
  -> accepted actual facts | deferred(required coordinates) | scoped exact rejection | indeterminate failure
  -> common candidate set
```

- query-local state 只保存不能从 current IR 和已选坐标重算的 typed assignments；ready/live、exact
  demand、lifetime、resource calendar、SPM high-water、lower bound 和 makespan 都绑定一次immutable IR borrow并按typed
  assignment重算，是query-local analysis cache，不进入state identity，也不序列化为output或计划attr；
- actual probe 不修改原 source，也不在 clone 内修复 candidate；
- analysis/probe cache只在一个immutable borrow内存活；Q50.A `IREpoch`只拒绝跨borrow trial，不进入semantic cache key，也不
  代替nested structural snapshot。borrow内的key必须包含target facts和会影响结论的全部typed assignments；改变traversal、
  tile、layout、movement、buffer或schedule后，旧calendar、lifetime、SPM/legality结果全部失效；
- 任意时刻最多一个 live actual clone；host 可并行计算 immutable analysis，但 candidate set insertion 和 tie-break 使用稳定 key；
- regular mapping、reuse signature和coarse resource estimate只能给普通typed transitions排序；不能clone-per-mapping，不能把
  reuse/cost annotation写进候选IR，也不能用function name、JSON或opaque solver payload跨越compile seam；
- 最终 winner 仍重新通过完整 CardModule splitting、TileRegion-to-Instr、fresh completion、SPM/DDR、resource、ABI、
  verification 和 final recost，局部 probe 不替代complete CardExecutable gate。

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
| 2 | Q50.A placement-demand repair | 对完整logical shard trial形成typed exact-demand/coverage proof；reduction、broadcast、window/stride、multi-piece、init与support relation均不被carrier/layout/route反写 | Q49.P；Core完成后供Q50.B消费 |
| 3 | Q49.P baseline functional closure与policy isolation | canonical `none`从正常上游IR完成确定性placement/tiling/SPM合法化且不消费search对象；一root一region；最窄exact probe后完整CardModule与CardExecutable各形成一次 | Q51.Core、baseline性能复核 |
| 4 | Q51.Core | 从零建立typed assignment与transition apply、deterministic frontier、baseline incumbent、ledger/budget、evaluation/result evidence及finite state-graph model；public `search`同批改接新Core并删除旧search控制链；不含新真实轴或actual probe | Q50.S、Q50.B |
| 5 | Q50.S structured semantic alternatives | typed proof与actual TensorProgram roots接入共同owner | Q51 closure |
| 6 | Q50.B spatial partition + placement | 完整spatial domain接入共同owner | Q50.C |
| 7 | Q50.C maximal single-root TileRegion | 单root local work的完整actual region materialization | Q50.D |
| 8 | Q50.D coupled traversal / region fusion | 多op boundary、coupled traversal和合法cut进入同一state | Q50.E |
| 9 | Q50.E complete temporal tiling | 全iterator finite breakpoint domain | Q50.F |
| 10 | Q50.F actual region probe | actual lowering/lifetime/SPM反馈回共同candidate set | Q50.G |
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
| 稳定 downstream trunk | CardModule/TileModule IR、CardModule-to-Tile conversion、TileRegion-to-Instr、Tile memory planning、card resource/target verification、package/runtime | 保留；所有 policy 复用同一路径；现有`TileMemoryPlanning`/`CardExecutableLowering`仅作实现定位，后者仍需按实际职责收敛名称 |
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

Q50.S与Q50.B–K分别直接实现新assignment上的semantic、spatial、region/fusion、temporal/probe、representation/movement、buffer和
event/resource职责；不存在“先让旧owner返回新类型”的中间合同。Q51.Core先删除旧control branch，使后续public `search`和
每批测试只能经过新Core。每个Q50变更把本轴mechanism接入新assignment时，同批删除仍依附于baseline/downstream文件中的旧
selector、repair、stats和专属测试；确有独立价值的局部算法、proof或test witness移动到新owner后按新合同验证，其余不迁移。

## Q50.0：CardExecutable Compilation Boundary

先从当前综合大流程抽出唯一、无策略的CardExecutable编译/准入函数：输入已经选择且物化的CardModule，依次执行
Tile module splitting、TileRegion-to-Instr、fresh completion reconstruction、fixed-capacity SPM/DDR planning、
communication/resource/ABI verification，输出accepted CardExecutable、proven exact rejection或indeterminate failure。

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
2. controller先完整materialize CardModule才能发现capacity conflict，refinement后再做Tile scoped probe，probe通过后又重新完整
   materialize CardModule。声明“最终完整CardModule一次”不能只统计Q50.0 compile；所有feasibility trial必须先走single-root/
   single-Tile最窄materializer，全部fit后才形成唯一完整CardModule。
3. Q50.0已经返回accepted executable后，baseline仍构造`StaticSchedulePlan`并运行duration estimation，结果既不影响baseline
   output也不进入Q51 incumbent；该shadow plan可让合法executable因旧cost失败而失败，必须从baseline路径删除。
4. common compile seam在每次trial/compile中无条件把全部Tile dataflow IR打印成字符串，即使production caller最终丢弃trace；
   IR inspection必须是显式请求且只在最终accepted output上执行，不能成为baseline或candidate admission的固定成本。

同一review还确认baseline代码、100字段的search statistics和trace-bearing synthesis result仍共同定义在旧search
monolith/header中。Q49.P完成前必须把baseline的窄assignment、typed outcome和必要work ledger移到不依赖旧search对象的current
owner；Q51.Core随后删除旧controller时不能再次迁移或适配baseline。

```text
Pipeline position:
- Upstream IR / input:
  verified card-local TensorProgram、available Tile、immutable target facts及可从current SSA/structured semantics派生的policy-free
  relation机制；尚未创建search state/candidate，也没有预先计算的placement-specific demand或selected spatial、TileRegion
  grouping、temporal、layout、route、buffer choice。Q50.A exact demand由controller对每次closed spatial trial现场查询。
- Current stage responsibility:
  由独立deterministic feasibility controller从正常上游IR自行构造canonical spatial assignment、每root独立TileRegion、显式
  DDR boundary、target-required representation/movement、single-buffer assignment、order/completion和完整temporal vector；
  使用最窄合法scope的actual lowering/lifetime/SPM probe，只沿direct causal witness遍历有限canonical fallback，直到得到
  第一个所有required scoped exact probe均为fit的完整canonical completion；只有随后一次Q50.0完整gate才能标为accepted。该
  合法化不评分或比较性能，但必须覆盖声明支持的baseline域，不能因最大初始tile/placement
  不合法或没有search candidate而停止。
- Output IR / files:
  一次性物化的完整baseline CardModule，以及经Q50.0一次完整准入后得到的accepted CardExecutable；Q59 transaction随后提交
  verified ExecutablePackage。局部probe clone和query-local witness不成为output或shadow plan；普通编译不生成printed-IR
  snapshot，显式inspection只在最终accepted output上按请求采集。
- Downstream consumer:
  `none`直接发布accepted executable/package；Q51.Core复用同一个accepted executable及其actual cost/digest作为初始
  incumbent，不用search carrier重新构造baseline。
- User-level driver / named pipeline:
  wafer-compile source-to-package pipeline的typed `none`；typed `search`只调用该controller取得初始incumbent，之后才进入
  candidate-selection owner。
- Explicit non-goals:
  不增加第二套Card/Tile/Instr lowering或SPM planner；不在baseline比较placement质量或搜索group/fusion/multi-buffer/可选
  route；不把一个region的局部fit冒充card-level transport/resource/ABI证明；不改变Q51的性能候选域。这里的非目标不排除
  baseline为走通程序而确定canonical placement、temporal tile、representation/movement、buffer、order和completion。
- Done criteria:
  baseline不依赖search state/candidate、完整placement-option domain/ranking evaluator、proposal order/group materializer，也不
  调用domain propagation、recursive CSP/backtracking或其它option-list assignment solver；canonical placement从typed
  structured/topology facts直接推导，始终只有一个live coordinate，不能先展开全部connected rectangle/axis options再取第一个；每个
  baseline TileRegion恰有一个structured compute root和必要non-root support closure；跨root shaped dependency显式DDR；
  region-local和最近合法isolated-ancestor probe均闭合，exact rejection携带direct typed causal witness；完整CardModule materialization与
  CardExecutable compilation各一次；accepted后不构造`StaticSchedulePlan`、duration estimate或其它不被output消费的shadow result，
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
feasibility resolution，取得semantic全序中第一个required scoped probe全部fit的canonical completion，再交给一次Q50.0完整
gate准入；不能复制第二套placement语义。query-local
trial/fit是功能合法化状态，不是search candidate。它可以在single-root范围内反复探测temporal breakpoint，全部root接受后只
形成一次完整CardModule并通过Q50.0完整编译一次。

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
staging、alignment/bank和actual lifetime，再由同一scoped exact probe判断fit；不能只缩output shape或沿用上一trial的
workset/lifetime。

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
-> card resource and target verification
-> ExecutablePackage writing
```

`OptimizationConfig::none()` 使用独立 deterministic feasibility controller：最大合法非空 Tile participation；每个structured compute
root独立TileRegion；root间显式DDR boundary；零fusion；buffer count为1。participant count、Tile group、iterator/factor和独立
root顺序按typed structured/relation/topology facts的完整semantic key决定，不使用pointer、walk ordinal、`stableOrdinal`或
search proposal order。controller沿不截断的有限canonical fallback跳过illegal option并取得第一个scoped-exact-feasible
completion，再由Q50.0签发accepted；它不计算score、不维护incumbent/candidate family，也不保留用于质量比较的备选方案。

participant group按root定义，只包含该root的非空执行Tile；accepted full CardModule仍必须拥有target要求的all-and-only完整
Tile domain。未参与某个root的Tile只在最终完整CardModule中按IR合同存在，scoped probe不得为保持card形状创建no-work
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

Q49.P建立比Q50.F更窄的policy-free scoped TileRegion-to-Instr/lifetime/SPM capacity seam。每次调用的输入是controller当前
trial已物化的single-root TileRegion、完整temporal tile vector以及该trial的spatial、DDR-boundary、representation和
single-buffer assignment；这些closed coordinates只约束一次exact query，不表示baseline入口已经选好完整assignment。
region-local lowering若需要call/symbol closure或function lifetime，只提升到最近合法`IsolatedFromAbove`
ancestor并在那里完成同一次probe。`RequiresFunctionScope`是scope escalation请求，不是fit或可忽略结果；无法在准确scope形成
结论时返回indeterminate，不扩大局部结论。

probe返回fit、带direct typed witness的proven infeasible/unsupported，或indeterminate。capacity witness可以是冲突集合，
但其中all-and-only actual allocation/lifetime owner必须经同次materialization relation关联到当前single root的result/
operand demand和temporal assignment；unsupported witness必须命名无法表达的typed lifetime/call relation及已尝试的最窄合法
scope。不按type/shape、位置字符串、region内“可能相关”的其它root、pointer identity或diagnostic字符串猜测。只有
proven failure允许baseline controller沿确定性fallback lattice前进；probe本身只判断当前closed-coordinate trial，不生成下一
tile、不选择分支、不repair IR，也不创建Q51 candidate。

`none`与`search`共享immutable structured/relation/target facts、policy-free single-root TileRegion materializer、scoped probe和
actual Card/Tile/Instr、completion、SPM/DDR、verification、package机制；不共享search state/candidate/evaluator、group
boundary、proposal order、score或feedback。Q49.P先产出并准入baseline，Q51只把accepted executable/cost作为incumbent，
不得通过search representation重建同一baseline。Q49.P施工时先把resolved baseline assignment的single-root apply与每次
closed-coordinate feasibility probe落在稳定PhysicalDataflow/Conversion边界；Q50.B/C/F随后扩展完整search domain、
single-root mechanism和deferred
probe taxonomy时必须复用同一实现，不能再建baseline-private或search-private materializer/probe。共享的是已关闭coordinate的
mechanism，不是生成或选择coordinate的控制：任何option-domain construction、constraint propagation、recursive CSP、backtracking、
candidate/evaluator或winner协议都不得进入`none`的transitive call graph。baseline默认值不限制Q51域。

历史official HF prefill、functional decode、LLaMA block 的 source、oracle、package 和 no-card runner只证明旧入口mechanics，
未执行真实板端，故不得标`done`。Q49.P的gate经Q59 current compile transaction使用fresh有界小图、overfull-to-fit、五类relation
定向case及轻量source-to-package/no-card输入，证明CardModule、accepted CardExecutable与package digest稳定、oracle/no-card通过，
并以fresh阶段计时和work count确认完整CardModule materialization与CardExecutable compilation各一次，scoped probe不构造
card-shaped/no-work-Tile wrapper。Q51完整new-search链闭合前，Q49.P、Q50各checkpoint和Q51.Core都不执行重型LLaMA block，
`none`与`search`均如此；也不把既有未完成运行变成重跑门禁或current证据。首轮重型LLaMA执行归Q52显式scalability profile，
Q53才签发正式package/oracle/no-card与board-ready证据。
定向结构测试必须覆盖：同Tile多个独立root形成多个region；显式structured producer不能伪装成support closure且一个root的
lowered multi-op不会误判成多root；call或unsupported lifetime提升到最近合法scope；equal-shape fanin只按direct witness
refinement；search state/candidate、search-oriented domain/ranking
evaluator和grouping调用计数为零，baseline work不写入candidate proposal统计。它不把历史耗时
写成长期阈值，也不得借“统一入口”让`none`再进入search feedback。

定向功能测试还必须从没有selected assignment的正常TensorProgram进入：至少覆盖初始完整tile因operand/halo/temporary/
alignment/lifetime真实占用而溢出、经过多个合法breakpoint后fit并完成source-to-package/no-card；覆盖multi-axis、tail和最小
合法粒度。浮点reduction轴自由重结合后每个demand都可沿其breakpoint lattice缩到最小合法vector，「最小合法tile超限」的
浮点capacity negative case在当前lowering下不可构造（原反例程序还落在placement domain之外：纯reduction标量输出没有
parallel轴，placement domain本就不表达）；typed capacity/unsupported terminal保留为fail-closed防御出口。测试同时断言每次trial重新计算
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
  对每条data/init dependency及其中间pure support relation形成或组合exact IndexRelation；将consumer当前完整logical
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
  affine window及stride/dilation、strided slice/view、multi-piece、multi-result、DPS init producer与pure support-chain relation
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
- 没有独立structured root的view/reshape/slice/pad等pure support graph按typed semantics组合relation；多operand support graph
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
DPS init、pure support graph和multi-result ownership均进入同一all-and-only exact relation/coverage边界。baseline support carrier直接消费
query给出的per-destination demand与ownership intersections，并把empty destination解释为无physical action，不再从balanced producer
rectangle正向猜测support image。direct edge的canonical dense carrier及其它layout、route、residency和resource calendar仍是
Q50.G/H后续扩展的physical alternatives；其失败不得回写Q50.A cache或删除spatial trial。

### Gate

- 独立exact-demand suite直接测试query，不以edge-strategy/materializer成功代签；覆盖multi-axis remainder、reduction input/init与
  partial merge、broadcast、affine window+stride+dilation+pad、strided slice/view、multi-piece union、multi-result、fanout/fanin、
  explicit init root及多operand pure support graph；
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
  热路径上的closed-form witness保留、generic preflight和work ledger由Q49.P P6闭合，不重新打开physical carrier或placement选择。
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

现状基线（2026-08-17 HEAD，`build/q55-current-fresh`）：
`NoneJointlyRefinesExplicitProducerStageAndConsumerDemand` 已复现：2048 breakpoint 上全部 region probe 返回
`requires-function-scope`（`UnsupportedLifetime`），controller 计数后跳过视同 fit，随后唯一完整 gate 以
`spm-allocation` 失败。CROSS 挂起按 `memory/bugs.md` 记录复现（`validateSelectedTileLayouts` 每 op×每 relation
从零递归 `collectStorageRoots`，无 memo）。

### P1 RequiresFunctionScope escalation（probe/final 一致性）

- 目标：region-scoped probe 返回 `RequiresFunctionScope` 时不是 fit，也不可跳过；controller 提升到最近合法
  `IsolatedFromAbove` ancestor（该 Tile 的 FuncOp）做函数级 probe，并以其 typed verdict 作为该 region 的结论。
- 实现：
  1. 新增函数级 capacity probe `evaluateTileFunctionSPMCapacity`（`TileRegionSPMCapacityEvaluation.h/.cpp`）：
     输入该 Tile FuncOp（clone 到 scratch，block args 替换 inputs，同 `TileRegionEvaluationScope` 模式），
     运行与最终 `planTileMemory` 相同的 `instr-memory-planning-preparation` named pipeline 和
     `assign-spm-offsets` capacity 检查，返回 Fits / proven CapacityExceeded（含 `SPMMemoryPlanningFailure`
     evidence）/ unsupported lifetime / indeterminate。命名、失败分类与 `TileRegionSPMCapacityEvaluation` 对齐。
     该实现与 Q50.0 最终 planning 共用同一 pipeline/checker，probe 与 final 消费同一 demand 集合。
  2. `evaluateTileRegionSPMCapacity` 的 `RequiresFunctionScope` 不再由 controller 计数跳过：controller 对该
     region 的 owner FuncOp 调用函数级 probe，`Fits` 记为 fit，`CapacityExceeded` 进入现有 causal refinement
     路径（evidence 经同一 `makeTileSPMCapacityFailure`/materializationRelations 转换），其余为
     indeterminate 并中止 baseline（typed failure），不得跳过。
  3. 诊断保留 region-scope 与 function-scope 两层 outcome；`baselineRegionSPMChecksRequiringFunctionScope`
     只作计数，新增 `baselineFunctionScopedSPMCapacityChecks`。
- 验证：`NoneJointlyRefines...` 由新路径 fresh 通过（一次 refinement 后所有 probe fit、唯一完整 gate accepted、
  `cardModuleCompilationInvocations==1`）；原有 `baselineCardModuleMaterializations==2` 类断言按新契约更新。

### P2 CROSS 挂起修复（storage-root memo）

- 目标：`validateSelectedTileLayouts` 链路在 CROSS（16 destination × 16 owner peer fragment）上不出现
  O(ops×strategies×walk) 分钟级放大。
- 实现：`StructuredBufferRelations` 增加 query-local `StorageRootMemo`（`DenseMap<Value, DenseSet<Value>>`，
  同一 IR epoch 内只读），`shareStructuredBufferStorage`/`collectStructuredNodesUsedByOperation`/
  `operationUsesStructuredNode` 增加带 memo 的重载；`validateSelectedTileLayouts` 每 Tile root 构造一个 memo
  并贯穿调用链。旧无 memo 入口保留给一次性调用点。
- 验证：CROSS lit case 在 wall-time 上限内通过（目标秒级）；`WaferUnitTests --gtest_filter=...` 相关
  layout/fusion 测试不回归；fresh 主树构建通过。

### P3 scoped probe 不再构造 card-shaped/no-work-Tile wrapper

- 目标：tile-scoped probe 只物化被探 Tile 的 FuncOp，不创建其它 Tile 的 no-work entry，也不创建
  CardModule/TileModule shell。
- 实现：`WaferTensorProgramToCardModule` 增加只物化单 Tile FuncOp 的入口（复用 `lowerTensorProgramToCardModuleImpl`
  的 per-Tile lowering，输出 minimal module + 该 Tile 的 relations）；controller 的 scoped probe 改走该入口，
  region 收集从该 FuncOp walk，删除“跳过非探 Tile module”逻辑。完整 CardModule 路径不变，
  Tile domain 完整性检查仍只作用于完整 materialization。
- 验证：定向单测断言 scoped probe 输出无 no-work TileModule；`baselineCardModuleMaterializations` 与
  `baselineScopedCardModuleMaterializations` 语义分离；CROSS/CHAIN/GEMM 及现有 baseline 单测通过。

### P4 一 root 一 region 结构合同

- 目标：每个 baseline TileRegion 恰有一个 structured compute root 和必要 non-root support closure；
  同 Tile 多 root 形成多个顺序 region；跨 root shaped dependency 显式 DDR。
- 实现：完整 CardModule materialization 后做结构验证：每 region 经 materializationRelations 收集 distinct
  structured root（现有 `collectStructuredNodesUsedByOperation` + 显式 structured DAG 节点集合），
  多 root 即 typed failure 并命名两个 root；同 root lower 出多 compute op 仍算一 root（按 node id 去重）。
  若当前 `IndependentDDRStages` materialization 在已有 case 上不满足，则按 root 切分 stage 使每 region 一 root。
- 验证：新增结构单测（同 Tile 多独立 root → 多 region；显式 structured producer 不伪装成 support closure；
  单 root multi-op 不误判多 root）；现有五类 production gate 与 CROSS/CHAIN/GEMM 通过。

### P5 direct-witness-only capacity attribution

- 目标：capacity refinement 的 causal coordinate 只来自 typed conflict certificate 和 materializationRelations，
  不做同 shape/type 的歧义 producer 扩展，不把同 region 其余 root 并入 refinement。
- 实现：删除 `synthesizeDeterministicBaseline` 中 equal-shape producer 扩展（`StructuredDAGEdgeStrategyPlan`
  段落的 shape 匹配）与 region-wide structuredNodes 组合；demand 无 typed node witness 时返回
  indeterminate（typed failure），不得按 shape/type 猜测。P4 的 one-root-per-region 保证 region 内唯一
  refinement root 即当前 single root。
- 验证：equal-shape fanin 定向测试证明只按 direct witness 选择 refinement 目标；`NoneJointlyRefines...` 仍通过。

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
     enumerator拥有。
  2. `deriveBaseline`/`deriveDeterministicBaseline` 改为直接构造 `TileMapping`（nodePlacements +
     outputPlacements + canonical edge strategies/layouts + 完整 temporal vector + materializationMode +
     bufferCount=1）的窄 assignment 类型，不再经 `TileExecutionCandidate`（stableOrdinal、evaluation、
     transition、feedbackRootOrdinal）与 `StructuredDAGPlacementEvaluator`；legality由每edge当前single pair的Q50.A query及全
     assignment demand plan/canonical carrier闭合，不建立option compatibility table。
  3. 对single-coordinate exact-demand热路径建立独立work closure。2026-08-17 bounded stack capture确认当前LLaMA诊断停在
     `getExactStaticRectangularImage`的generic `PresburgerSet::isEqual`，继而进入`isSubsetOf/subtract`；变量/分段数上限不能保证该
     等价证明有界。支持的projected/permuted/static-rectangle indexing semantics必须由`IndexRelation` builder/composition保留或
     直接构造closed-form rectangular-image witness，使baseline不进入generic equality recovery；generic fallback在调用前按完整
     relation complexity做fail-closed preflight并计入query-local ledger，超限返回`ResourceExhausted`/indeterminate，绝不能当作
     logical rejection或触发下一个coordinate。用轻量、同relation结构的定向case覆盖该路径，不靠重型LLaMA重现。
  4. baseline 路径不再接收`CardExecutableSynthesisStatistics`这个search bag，也不再写candidate proposal/fusion/layout/buffer
     统计和selected evaluation metrics；必要的probe/materialization/work计数进入baseline专属窄ledger，
     diagnostics 以 baseline 专用行报告（`card-executable-selection` 行移除 stable ordinal/evaluation 字段或
     由 baseline 专用行替代），测试只断言窄ledger和旧statistics type不在调用闭包。
- 验证：grep/调用计数证明 baseline 调用链不含完整placement-option/search domain/evaluator及旧statistics type；
  `deriveStructuredDAGNodePlacementOptions`、`deriveCanonicalBaselinePlacements`、`TileExecutionCandidate`和
  `CardExecutableSynthesisStatistics`不在baseline transitive call graph；ledger证明placement domain size=0、recursive
  solve/backtrack=0、exact pair query数只随actual DAG edge和deterministic legalization step增长，且supported rectangular-image
  case不进入generic Presburger equality recovery；CROSS/CHAIN/GEMM、显式`none`的overfull-to-fit与五类production gate通过，
  不执行或改造旧search-named回归；
  主树、board runtime、SystemC 三棵树 fresh 构建通过。

### P7 baseline materialization/output seam清理

- 所有capacity feasibility trial只materialize当前single-root/single-Tile最窄scope；不得先建完整CardModule再决定需要probe。
  全部required trial fit后只构造一次完整CardModule，并只调用一次Q50.0 complete compile。
- Q50.0 accepted result直接成为baseline semantic result；删除baseline的`buildAcceptedStructuredDAGSchedulePlan`、duration
  estimation和`StaticSchedulePlan`/theoretical-cost include。Q51需要的同cohort actual cost从accepted current Instr/resource
  facts按自己的typed cost边界取得，不由baseline预建shadow schedule。
- compile API把IR inspection改为显式optional sink/请求；普通source-to-package、`none`及候选evaluation均不调用IR printer。
  显式dump只对最终accepted Tile modules生成一次，不进入executable/package语义或acceptance gate。
- baseline implementation和typed result从旧search monolith/shared header中分离；result只拥有move-only accepted executable、
  必要actual facts和窄ledger，不携旧search statistics、printed trace、candidate relation或failure string state machine。
- transaction/failure-injection与baseline定向测试显式使用最短`none`路径；不运行默认旧search、旧winner对照或异常长integration。
- 验证：work count证明完整CardModule=1、Q50.0 compile=1、IR print=0、schedule-plan build=0；显式IR dump单独证明winner后
  每Tile恰好一次；accepted baseline不可能因未消费的cost/trace失败。

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
  不再继续、重跑或作为Q49.P完成门禁。
- current spatial placement只从parallel iterator生成axis，reduction factor保持1且没有partial-result merge；这是Q50.B的明确
  implementation gap。相反，fresh `SearchProposesAndMaterializesMultipleReductionIteratorAxes`和
  `NoneCarriesReductionDemandThroughTheCompleteExecutableGate`已经证明多reduction轴temporal materialization与baseline complete
  gate存在；Q50.E待闭合的是完整breakpoint/wave-loop domain，不是从零补一个reduction temporal split。
- 旧`MultiOutputFanoutCanPlaceBranchesOnDifferentDestinationGroups`单case超过3h、RSS约4.3GB的现象属于待删除的旧spatial/search
  枚举链。Q51.Core前不再运行该长case；Q50.B以tiny reference enumerator重建域，Q51 closure只验证new chain。

P8完成顺序：

1. 闭合P1–P7及上述SPM evidence copy，更新`tasks/progress.md`、相关设计和稳定memory；
2. fresh运行direct unit、定向lit、overfull-to-fit、五类relation和轻量source-to-package/no-card；显式`none`，不进入旧search、
   paired optimization或重型LLaMA；
3. ledger证明placement option/domain/recursive solve/backtrack为零，single-coordinate exact query有界，完整CardModule
   materialization与CardExecutable compilation各一次，accepted后schedule-plan/默认IR print为零；
4. 检查主树、board runtime与SystemC/model三棵树构建；确认相关lit/CTest实际执行而非unsupported；
5. 提交本批改动（作者规范见`AGENTS.md`）。

已有fresh证据（2026-08-17）可复用但不代签剩余门禁：受影响unit 70/70、默认lit gate 216/216、三棵树fresh构建通过；
Tools/Runtime lit 33/38（5 unsupported、0 failed）；显式`none`的f16 reference/AddModel已走通source-to-package/no-card。
LLaMA capture或旧未完成compile不进入本项证据。Q51完整new-search链闭合后，Q52才首次运行重型LLaMA显式profile；Q53再生成
正式package/oracle/no-card并进入board-ready。

## Q51.Core：Search Control Kernel

### 2026-08-17 review 结论

原节把Q51终态assignment schema、Q50.F actual probe、Q51全轴oracle和Q52 production策略都提前算进Core，和current施工
顺序不一致，也会诱导实现直接复用正在由Q49.P拆除的`TileExecutionCandidate`。Core现在只拥有**搜索控制**：typed child
state的原子接纳、deterministic frontier、stable dedup、incumbent、budget/work ledger、typed evaluation outcome与result
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
   `CardExecutableSynthesisStatistics`以约百个字段把baseline、proposal、feedback、winner和rematerialization耦成一个协议。

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
  一个immutable TensorProgram borrow、transaction-owned ProgramData view及target/cost-cohort facts；Q49.P已经通过Q50.0完整准入的move-only baseline
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
  Q50.S与Q50.B–Q50.K逐项扩展current typed assignment和transition；Q50.F接入affected-region probe；Q51 closure接入
  全部真实轴后的complete actual materialization/Q50.0 gate parity、全轴oracle、exact-domain证明和source-to-package chain。
- User-level driver / named pipeline:
  不新增pass、CLI、optimization policy或磁盘sidecar。Q51.Core同批把既有`search` policy路由到new-search session并删除旧
  branch；任一时刻只有一个public owner。Core阶段该入口只发布Q49.P accepted incumbent，后续Q50机制原位扩展同一session。
- Explicit non-goals:
  不实现Q50.S/B–K任一choice domain；不把Q49.P scoped capacity probe升级成通用search probe；不预设best-first priority、
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
  outcome。Core不拥有scoped probe。Q49.P的single-root capacity seam仍服务baseline；完整causal coordinates关闭后，Q50.F
  复用其policy-free lowering/packing mechanics并加入new-search deferred、memo和causal scope合同。
- ledger分别记录generated、deduplicated、expanded、deferred、exact-rejected、unresolved、actual-probed、full-compiled、accepted
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
- `CardExecutableSynthesisStatistics`中只服务旧proposal、feedback、shortlist、selected ordinal和winner rematerialization的字段
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

## Q50.S：Structured Semantic Alternatives

Q50.S只从typed SSA、structured semantics、effect和loop-carried dataflow证明可用算法族，并在isolated clone中物化真实
TensorProgram alternative。普通DAG、online recurrence和partition/local-reduction/merge DAG共享同一builder contract；
near-miss必须保持原图。

若K/V window或split count改变recurrence/partition拓扑，它就是root transition参数，每一点都先成为actual TensorProgram；
若只是已有recurrence上的普通temporal tile，则归Q50.E。`128`等常数只能排序合法参数点，不能定义合法域；SPM capacity、
Tile count或KV length都不能提前替代该选择。Q50.S不暴露public Flash selector/pass，也不按名字、Q length或shape恢复语义。
所有actual roots都交给Q51同一physical-dataflow candidate set，不能在这里选winner。

## Q50.B：Spatial Partition + Physical Placement

### 机制

- 从structured iterator semantics枚举每个op覆盖所有iterator的完整spatial factor vector与block partition
  relation；它必须联合表达multi-iterator factors、non-divisible remainder/tail、parallel/reduction iterator role和
  必要的partial-reduction merge，result axis不能替代iterator axis；
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
multi-axis remainder、reduction merge和partial redistribution上的stable key集合完全一致；selected CardModule/TileModule
直接表达Tile ownership。局部gate通过后删除spatial coordinate-descent owner和rank==Tile假设，但保留
可复用topology/domain/cost mechanics。“非最大参与、非相同Tile group或暂时更贵的placement成为global actual
winner”是Q51 closure的跨轴gate。

同一factor count中，确实改变partition relation、logical-mesh embedding或physical placement的不同axis/factor nesting必须产生
不同typed witness；extensionally等价的生成路径必须canonicalize。regular/full-occupancy proposal与非矩形、partial、
asymmetric、disconnected合法补集均可达；关闭或改变proposal排序后，tiny exhaustive domain和winner不变。

## Q50.C：Maximal Single-Root TileRegion

### 机制

对给定structured root、spatial shard和尚未细分的local iterator domain，每个Tile物化一个覆盖该root全部local work和
non-root support closure的maximal single-root TileRegion。显式producer只要映回另一个structured DAG node，就以boundary
input/DDR obligation留在region外，除非后续Q50.D显式选择multi-root coupled group。这里maximal表示“对已选region boundary
不漏work、不把同一root local work任意拆成多个相互不知情region”，不是强迫使用最大tile、最大fusion group或单一结果驱动入口。

region materializer必须由SSA、structured interface、indexing maps、DPS init/effect和observable roots驱动；不得从buffer、
op、result名恢复traversal。多个result、init operand、non-root support chain和reduction均要得到all-and-only local work；
root cardinality由materialization relation映回structured DAG node，不能按lowered compute op数量或“support”标签猜测。

### Gate

- single-result、multi-result、DPS init、reduction 和 side-effect boundary 正负测试通过；
- actual Tile IR 对每个 selected Tile 都有唯一、完整、可 verifier 的 single-root region witness；
- Q49.P已建立的resolved baseline assignment apply与Q51显式test assignment复用该policy-free single-root materializer；
  materializer只消费调用方已给定的singleton root boundary，不接收或自行推导search candidate/grouping；
- 不在本 checkpoint 做 multi-op fusion、temporal winner、layout 或 buffer 选择。

## Q50.D：Maximal Coupled Traversal

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

## Q50.E：Complete Temporal Tiling

### 机制

每个 scheduled structured op 的 temporal assignment 是覆盖全部 iterator 的完整向量，加上Q50.D已选traversal内部有限、
语义可区分的wave-loop nesting/order。每个 spatial mapping 从完整per-Tile local extent开始，惰性枚举所有会改变实际
hardware work或legality的有限breakpoint与order；需要
layout、buffer或implementation facts的breakpoint是对这些typed assignment参数化的mechanism查询，不得用未选default：

- `ceilDiv` wave 数、tail 和 divisor；
- native issued geometry、padding 和 physical work；
- DDR/SPM/NoC transaction、descriptor 和 layout footprint；
- implementation geometry、reduction split 和 buffer-count feasibility；
- actual region probe 产生的 scoped infeasible/feasible boundary。

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
- 至少一个tiny case证明相同tile vector的不同wave-loop order会改变reuse/SPM或global winner，且两种选择均可达；
- 删除 fixed temporal seed domain 和 allocation-feedback candidate family 的选择责任。

“小tile使fusion/buffering成为winner”及任何需要actual layout/buffer/resource cost的比较统一放在Q51 closure。

## Q50.F：Actual Region Probe

### 机制

cheap footprint只能返回proven must-coexist lower bound、non-binding ranking estimate或
`deferred(required coordinates)`；只有第一种超过capacity才能拒绝状态。估算可放下不能证明actual packing可行，未证明
coexistence的估算超限也不能exact reject。近似或未经boundary-faithful proof的solver/constraint结果只可排序；exact局部
solver可对准确建模的子问题返回proof/proposal，但selected offset/choice仍须物化并通过typed gate。Q51可对受影响TileRegion
请求isolated actual region probe；
该probe复用Q54形成的region-local conversion/lifetime/packing seam，不建立第二套local wrapper或pipeline。
probe先从lowering/packing contract求出结论依赖的causal coordinates。只有traversal、temporal以及实际会影响该
region的representation、movement/staging、buffer/slot、order/completion等坐标全部显式赋值后，才实际物化并执行
lowering、fresh lifetime/completion和fixed-capacity SPM packing，返回：

```text
accepted scoped actual facts
| deferred(required coordinates)
| exact rejection(reason, causal assignment key, conflict witness)
```

probe不选择下一tile，不修改state，不在clone内spill/retile/rebuffer。坐标未闭合时只能返回
`deferred`，不构造带default layout/carrier/buffer的clone，也不得拒绝spatial/region/temporal parent。Q51对
`deferred`只会继续生成required-coordinate transitions；对exact rejection才可在同一parent上使用causal no-good。
只有结论依赖的全部轴都包含在cache key中时才能memo；只有proven exact rejection才能形成no-good。packer
`ResourceExhausted`或internal failure必须返回indeterminate并保留parent/siblings。

### Gate

- cheap lower-bound rejection、deferred missing-coordinate、actual accepted、actual packing failure和已赋值representation下的
  unsupported lowering分别可诊断；
- estimated-fit/actual-fail与estimated-overfull-but-unproven-coexistence反例证明estimate不签发legality；
- exact failure回共同parent并保留不同temporal/traversal以及required-coordinate siblings；这里只证明candidate set继续，
  不在本checkpoint选出跨轴winner；
- source IR 无变化、probe clone 全销毁、任意时刻最多一个 live actual；
- Q49.P deterministic feasibility probe与Q51 closed-coordinate probe复用同一lowering/lifetime/packing implementation；
  Q50.F只增加deferred坐标和
  common-state反馈taxonomy，baseline不进入candidate set；
- 删除 allocator repair 和 feedback beam，保留 packer、conflict witness 与 exact validator。

## Q50.G：Layout and Physical Representation

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

## Q50.H：Explicit Data Movement

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

## Q50.I：Buffer and Rotating Slots

### 机制

buffer count 是 op-wave/edge pipeline 的共同候选维度。domain 至少表达 single/double/triple buffering；若 current IR 与硬件
允许更多 slot，则以并发 wave、lifetime 和 capacity 推导有限上界，不能把 3 作为未证明的永久 cap。每个候选显式携带 slot
binding scope、rotation、producer release、最后async consumer和prologue/steady/epilogue phase等typed choices；aligned footprint和actual lifetime
由current immutable IR borrow与region/representation/movement/order assignment重算。

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

## Q50.J：Ready/Order/Worker/Resource Schedule

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

### Gate

independent branches、fanin/fanout、shared DDR、NoC link contention、multi-worker join、effect ordering和completion hazard有
正负测试；actual Instr顺序、worker和completion可验证，finite normal form的tiny域与independent reference
enumerator一致。删除ready-order、worker和overlap的独立selector及静态scheduling capability registry；query-local calendar在对应IR epoch/assignment
失效或winner materialize后销毁，不成为shadow schedule。

## Q50.K：Conditional Stage Pipeline

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
- public policy只剩`search|none`，且`search`只进入new-search session→Q50 mechanisms→Q50.F probe→complete materialization→
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

Q51完整new-search链提交并通过其small-oracle/new source-to-package gate后，本节才首次允许执行重型LLaMA block；执行必须是
显式选择的bounded profile批次，不进入普通unit/lit/CTest或每次功能改动回归。在 generic mixed DAG、HF prefill、functional
decode 和 LLaMA representative block 上记录：

- 各轴 transition generated/rejected/deduplicated/expanded 数和 candidate set width/peak live states；
- canonical key 重复率、separator width、dominance/no-good 命中和失败作用域；
- time-to-baseline、time-to-first-better-actual，以及 1/3/10/30 分钟 incumbent actual cost/digest；
- lower bound、incumbent、gap 随时间变化；一旦启发式丢状态则停止宣称全局 gap；
- clone、region probe、complete CardModule materialization、TileRegion-to-Instr、SPM/DDR packing、schedule、cost、verification 的调用数与
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
| actual probe/lowering 重复 | immutable analysis 与 accepted result memo | IR epoch、typed inputs 和 target facts 完整 |

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
  -> actual probe/full gate
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
