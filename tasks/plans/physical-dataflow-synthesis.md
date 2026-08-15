# Physical Dataflow Synthesis 实施计划

状态：Q49 deterministic `none` baseline 已于 2026-08-11 达到 `board-ready`，但其控制流隔离和编译耗时不属于已签发的
正确性证据。Q50.0无策略CardExecutable编译/准入边界与Q54 MLIR infrastructure已闭合；当前先由Q56/Q58/Q59收口
package data、program data backing和compiler entry transaction，再以该current source-to-package路径隔离Q49.P baseline控制流，
并修复Q50.A production demand boundary、建立Q51.Core共同状态、transition和actual-probe seam；随后让
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
  Q50.S semantic-alternative builder先从typed SSA物化actual TensorProgram roots；唯一query-local physical-dataflow search
  owner只在这些roots中选择一个，并联合展开spatial partition/placement、
  TileRegion、coupled traversal、
  temporal tiling、layout/representation、explicit movement、buffer、ready/order/worker、NoC/DDR/compute resource
  timeline 和条件式 stage pipeline；只有 actual IR 可以跨越 candidate-materialization seam。
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
  Q50.0先建立共同CardExecutable compile/verification seam；Q54按19号合同收口MLIR infrastructure；Q56/Q58/Q59先闭合
  package data、large payload backing与compile commit；Q49.P基于该current入口隔离
  baseline控制流；Q50.A修复production exact-demand boundary；Q51.Core建立共同search kernel；Q50.S与Q50.B–Q50.K逐轴交付
  mechanism 且移除对应旧 owner；Q51 通过two-layer small exhaustive oracle、complete CardExecutable gates 和有效 fusion 闭合；
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
  demand、lifetime、resource calendar、SPM high-water、lower bound 和 makespan 都是按 IR epoch 与 typed
  assignment 重算的 query-local analysis cache，不进入 state identity，也不序列化为 output 或计划 attr；
- actual probe 不修改原 source，也不在 clone 内修复 candidate；
- analysis/probe cache key 必须包含 IR epoch、target facts 和会影响结论的全部 typed assignments。改变
  traversal、tile、layout、movement、buffer 或 schedule 后，旧 calendar、lifetime、SPM/legality 结果全部失效；
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

所有等级都必须保留已经通过Q50.0 exact verification的Q49 `none` baseline作为合法incumbent；“没有比baseline更好的
candidate”不等于搜索失败。若baseline本身无法通过exact gate，必须报告明确失败，不能伪造fallback。

## 施工 checkpoint

| 顺序 | Checkpoint | 施工责任 | 完成后才能开始 |
| --- | --- | --- | --- |
| 0 | Q50.0 CardExecutable compilation boundary | 抽出无策略的actual compile/verification seam，不允许lowering修候选 | Q54 |
| 1 | Q54 MLIR infrastructure conformance | typed IR/interface、scoped pass/analysis、named pipeline与rewrite transaction收口 | Q49.P、Q50.A |
| 2 | Q49.P baseline control-flow isolation | `none`不构造search对象；scoped probe后只完整编译一次 | baseline性能复核 |
| 3 | Q50.A placement-demand repair | `IndexRelation.image()` exact demand不被carrier/layout/route反写 | Q51.Core |
| 4 | Q51.Core | 共同state、transition、candidate set、incumbent、ledger、budget与oracle harness | Q50.S、Q50.B |
| 5 | Q50.S structured semantic alternatives | typed proof与actual TensorProgram roots接入共同owner | Q51 closure |
| 6 | Q50.B spatial partition + placement | 完整spatial domain接入共同owner | Q50.C |
| 7 | Q50.C maximal single-op TileRegion | 单op local work的完整actual region materialization | Q50.D |
| 8 | Q50.D coupled traversal / region fusion | 多op boundary、coupled traversal和合法cut进入同一state | Q50.E |
| 9 | Q50.E complete temporal tiling | 全iterator finite breakpoint domain | Q50.F |
| 10 | Q50.F actual region probe | actual lowering/lifetime/SPM反馈回共同candidate set | Q50.G |
| 11 | Q50.G layout / representation | layout、encoding、version与conversion transition | Q50.H |
| 12 | Q50.H explicit movement | local、NoC、DDR、collective、spill/recompute action | Q50.I |
| 13 | Q50.I rotating buffers | rotating slots与multi-buffer lifetime | Q50.J |
| 14 | Q50.J event/resource schedule | ready/order/worker/completion/resource calendars | Q50.K |
| 15 | Q50.K conditional stage pipeline | 在已证明条件下组合跨op wave pipeline | Q51 closure |
| 16 | Q51 closure | 全轴联合正确性、small oracle、旧owner清理和policy cutover | Q52 |
| 17 | Q52 profile-driven anytime / LNS | 10–30分钟内高质量actual winner与可解释trade-off | Q53 |
| 18 | Q53 production | representative workload package/no-card/board A/B | Q48后续工作 |

Q51.Core是Q51的提前施工checkpoint，不等于Q51已完成。Q50.S与Q50.B–Q50.K必须在core上逐轴交付；Q51只有在全部
轴组合、small exhaustive oracle、actual fusion 和旧 owner 删除门禁都满足后才能整体完成。

### Checkpoint gate 分层

Q50.S与Q50.B–Q50.K的顺序是**mechanism/builder 实现可用性**，不是把某个 search 轴提前选定或冻结。
每个Q50 checkpoint只能用显式typed test assignment补齐尚未施工的轴，并证明本轴的domain coverage、transition、
materializer、verifier、exact rejection/deferred和无局部winner。这些test assignment不是production default，新路径不得调用
旧owner暗中补全其它坐标。

“某个局部更贵的choice在后续轴闭合后成为global winner”、fusion/buffering/pipeline协同、complete
CardExecutable cost与全轴actual winner都是Q51 closure gate，不得用尚未迁移的旧selector提前签发。表中
“完成后才能开始”只表示下游mechanism可依赖上游typed contract；已选assignment变化后仍必须失效并重新展开所有
受影响轴。

## 现有代码分类与处置

现有文件不能因“结构乱”被批量删除。进入每个 checkpoint 前先把相关实现归到下列类别，并只处理当前轴。

| 类别 | 当前代表实现 | 处置 |
| --- | --- | --- |
| 稳定 downstream trunk | CardModule/TileModule IR、CardModule-to-Tile conversion、TileRegion-to-Instr、Tile memory planning、card resource/target verification、package/runtime | 保留；所有 policy 复用同一路径；现有`TileMemoryPlanning`/`CardExecutableLowering`仅作实现定位，后者仍需按实际职责收敛名称 |
| 可复用 core | `StructuredDAGAnalysis`、现有schedule-state/candidate类、edge-demand plan与placement domain/cost mechanics | 行为测试保护后纳入Q51.Core；旧类名只是实现定位，不升级成长期架构对象，也不让其自行选择winner |
| 可复用 mechanism | `BidirectionalTiling`、`CompleteTraversal`、`TileMaterialization`、`CandidateMaterialization`、movement/collective lowering、selected-buffer materialization、ready-order/worker/completion verifier | 按Q50.S与Q50.B–Q50.K接到transition/materializer seam；保留符合新合同的算法与verifier |
| 过渡 monolith | 当前executable-synthesis实现中的coordinate sweep、candidate family、allocation/buffer feedback、shortlist和mixed materialization | public entry暂保留，先抽出Q50.0编译边界，再按轴抽出；最终只保留Q49 baseline controller、Q51 driver和共同materialization seam |
| 旧 bounded/rank search | `RankCandidateSearch`、旧structured candidate generation/evaluation/selection及rank-era candidate set | 只作为 transformation、typed proof、diagnostic 和测试来源；不得恢复固定 beam/cap 或 rank-local winner |
| compatibility lowering | `StructuredDAGEdgeStrategyPlan`、dense rectangle fragment、现有 local/peer lowering | 在新 representation/movement IR 可完整消费 exact demand 前保留；Q50.G/H 逐项替换，不提前删除 |
| late selector/fixup | layout/movement optimization、ready-order、worker placement、buffer/allocation feedback 中会重新做选择的部分 | 先改成 verifier/materializer 或 typed transition mechanism，再按轴删除选择责任 |

tracked 删除必须同时给出：current replacement、actual-IR witness、fresh replacement test。缺任一项都不删除。不得按目录
恢复或删除，也不得因为旧文件包含错误 owner 就丢掉其中仍未迁移的 pass、analysis、lowering 或 negative test。

### 逐轴退役旧 owner

| 轴 | 当前需要退役的选择责任 | 删除门禁 |
| --- | --- | --- |
| spatial | monolith coordinate descent、rank==Tile 映射、placement-local shortlist | Q50.B 局部domain/materializer gate通过且新路径只由Q51选typed placement；global winner留Q51 closure |
| TileRegion / fusion | `CoupledFusion` edge action、fused-edge bool、rank-local traversal winner | Q50.C/D 能物化独立和 coupled traversal，fanout/diamond 正负测试通过 |
| temporal / SPM feedback | 固定 seed domain、accepted candidate 上的 allocator retile、allocation-feedback beam | Q50.E/F 局部breakpoint coverage、deferred/exact probe和无repair gate通过；跨轴SPM winner留Q51 closure |
| layout / movement | layout selector、edge strategy local winner、late route/spill repair | Q50.G/H 产出 actual representation/movement IR 并由 Q51 选择 |
| buffer | buffer feedback family、固定默认 buffer winner | Q50.I 的 slot/rotation/release/hazard typed choices 接入共同 state，lifetime保持为可重算analysis |
| order / worker / overlap | ready-order、worker、completion 或 overlap 的独立 selector | Q50.J/K 的可重入event/resource transition与actual verifier局部闭合；pipeline winner留Q51 closure |
| 全局旧 owner | bounded rank candidate set、rank candidate selector、shadow schedule | Q51 small oracle、full actual gate、public `search|none` cutover 全部通过 |

任何轴在 replacement 完成前都允许旧代码继续存在，但只能由当前旧路径消费；不得同时让新旧 owner 对同一 candidate
各选一次。迁移当批需要用定向测试证明新路径仅消费显式typed assignment，再删除本轴旧选择入口；跨轴
协同与production cutover仍必须等Q51 closure，不以局部fixture winner代替。

## Q50.0：CardExecutable Compilation Boundary

先从当前综合大流程抽出唯一、无策略的CardExecutable编译/准入函数：输入已经选择且物化的CardModule，依次执行
Tile module splitting、TileRegion-to-Instr、fresh completion reconstruction、fixed-capacity SPM/DDR planning、
communication/resource/ABI verification，输出accepted CardExecutable、proven exact rejection或indeterminate failure。

返回taxonomy必须完整区分`accepted CardExecutable`、`proven exact rejection`与`indeterminate failure`；allocator
`ResourceExhausted`、timeout或内部错误属于最后一类，不能伪装成candidate非法。

这个边界不得枚举候选、修改选择、在lowering中retile/spill/rebuffer，也不得把失败降级成performance Unknown。Q49与Q51
必须共用它；定向测试要证明同一CardModule得到相同accepted digest或相同rejection，且没有第二条兼容编译路径。

实现结论：actual compile/verification seam只接收owned、已选择的CardModule和typed buffering assignment，返回
accepted、proven exact rejection或indeterminate三态结果；baseline与search materialization都调用该入口。Tile
memory planning保留SPM failure kind，只有capacity overflow与unsupported lifetime等可验证失败进入exact rejection，未分类
allocator/internal failure保持indeterminate，调用方不得据此裁剪候选或启动repair。

## Q49 / Q49.P：Deterministic Baseline与控制流隔离

Q49已签发的正确性合同和历史board-ready状态不重定义；Q49.P只处理该证据未覆盖的控制流和耗时问题。
但控制流改写后必须用本软件产物重新证明result/digest与no-card不变，不得把历史package当成新调用链的证明。
`none`不得构造Q51 state、candidate set、candidate family或全图Cartesian组合；它可以在单op范围内探测temporal
breakpoint，全部op接受后只通过Q50.0完整编译一次。

```text
TensorProgram
-> deterministic CardModule construction
-> Tile module splitting
-> TileRegion-to-Instr conversion
-> Tile memory planning
-> card resource and target verification
-> ExecutablePackage writing
```

`OptimizationConfig::none()` 使用独立 deterministic controller：最大合法 Tile participation；每个 Linalg op 独立
TileRegion；op 间显式 DDR boundary；零 fusion；buffer count 为 1。它不创建 search candidate set、candidate family、Pareto、
feedback beam 或 optional edge action。实际 SPM 失败只沿同一 baseline 状态按有限 temporal breakpoint 前进，不产生分支，
不启用 fusion/layout/buffer/route 搜索。

Q49.P先抽出比Q50.F更窄的policy-free `local-fit` seam。它的输入是单个已物化op/TileRegion、当前完整temporal
tile vector以及baseline已固定的spatial、DDR-boundary、representation和single-buffer assignment；它只运行该独立
region所需的lowering、fresh lifetime和fixed-capacity SPM packing，返回fit、带conflict witness的proven infeasible/unsupported，
或indeterminate。只有第二类的proven failure才允许baseline controller前进到下一确定性breakpoint；indeterminate必须
终止并报告，不能冒充“当前tile放不下”。
`local-fit`直接以真实TileRegion或能提供必要call/symbol closure的最近`IsolatedFromAbove` ancestor为scope；不得为每个region
构造synthetic Module/Func并运行完整Tile memory planning。region-local lowering/lifetime/packing与必要的Func/Module
summary边界遵循19号合同；需要全局事实却无法形成exact summary时返回indeterminate，不扩大局部结论。
`local-fit`不生成下一tile，baseline controller才按唯一确定性breakpoint顺序前进。它的安全性仅来自“每op独立
TileRegion + op边界DDR + 无fusion”；该合同改变后必须返回完整CardExecutable gate，不得复用局部成功。

Q49与`search`只共享actual Card/Tile/Instr、completion、SPM/DDR、verification和package seam；baseline默认值不限制
Q51 域。历史official HF prefill、functional decode、Llama block 的 source、oracle、package 和 no-card runner只证明
旧入口mechanics，未执行真实板端，故不得标`done`。Q49.P的gate必须经Q59 current compile transaction用fresh
prefill/decode/Llama输入证明
CardModule、accepted CardExecutable与package digest稳定，fresh no-card与oracle通过，并以fresh阶段计时确认不再为
同一baseline反复materialize/compile whole graph。它不把历史耗时写成长期阈值，也不得借“统一入口”让
`none`再进入search feedback。

## Q50.A：Placement-Demand Boundary Repair

输入是 current structured producer/consumer SSA edge 和一组给定 placement。query-local placement-demand analysis从
structured indexing semantics导出`IndexRelation`，以relation image求每个consumer shard需要的all-and-only producer
logical set，并验证producer shard ownership coverage。当前`StructuredDAGEdgeDemandPlanner`只是待改造的实现索引。

该query-local analysis result只包含consumer logical domain、producer logical demand和producer ownership；不包含
layout、encoding、bytes、local/remote action、route、buffer、send/recv或fusion。当前`StructuredDAGEdgeDemandPlan`与dense fragment carrier由显式
compatibility lowering 消费该 output；它不是长期 search owner。

Q50.A的analysis和reduction、broadcast/window、stride、multi-piece与ownership coverage正负测试已经闭合，但production
placement query当前仍会立即降成dense rectangle、layout fragments和route，因此任务不能标done。修复后，placement
transition只消费上述layout-independent demand；compatibility carrier只在representation/movement坐标关闭后由显式lowering
消费。carrier表达失败不得反写为spatial placement非法。

## Q51.Core：先建立共同搜索骨架

### 责任

Q51.Core 在增加下一项 capability 前建立唯一、可测试的 search kernel：

```text
immutable structured IR + IR epoch
  + semantic DAG root assignment
  + spatial / region / traversal / temporal tile and wave-loop order assignments
  + representation / movement / buffer assignments
  + order / worker / completion / stage assignments
  + enabled mechanism transitions
  + incumbent / global work ledger

derived query-local analysis caches:
  StructuredDAGAnalysis, exact demand, ready/live set, lifetime,
  resource calendar, SPM high-water, lower bound and makespan
```

partial state 的 canonical key 必须只由 current IR identity/epoch 和已选 typed assignments 构成，不依赖
pointer/hash iteration、op 名、历史 ordinal 或 analysis cache。future-boundary equivalence可以投影不再可观测的旧assignment，
但必须保留将来会影响physical version、layout、region boundary、buffer/slot、order/worker/completion和resource
reservation的所有typed choices。ready/live、retained-value lifetime、calendar、resource availability与makespan从这些
assignment重算；它们可用于future-compatible dominance，不能反向成为第二份state事实。

### 初始算法

- 以 Q49 actual baseline 作为第一个 incumbent；
- production使用确定性best-first constructive search：先形成baseline，优先构造大TileRegion、coupled traversal、较大高效
  temporal tile、ordered factorized regular mapping与relation-derived reuse proposal；这些只是稳定排序seed，不是生成条件；
- exact lane每次展开parent都必须把其余合法region cut、tile、layout、movement、buffer和schedule sibling以可惰性
  continuation保留在共同candidate set，不论先展开的child是accepted还是rejected。typed rejection只能产生与其
  causal assignment精确对应的no-good或调整排序，不能作为生成sibling的触发器；
- 完整合法域由mechanism惰性可生成，不把固定candidate count、depth、beam width或wall timeout写成legality；真实负载从
  第一天支持中断于global work/time budget；如果exact continuation与bound仍完整则可返回
  `feasible-with-bound`，只有丢弃或未表示合法completion时才是`budgeted-feasible`；Q52再基于fresh profile确定
  production默认budget和有损策略；
- clone 前只允许 typed legality、topology symmetry、exact equivalence、已证明 lower bound、作用域正确的 no-good 和严格
  future-compatible dominance；
- estimates 只决定展开顺序；admissible lower bound 才能 branch-and-bound；
- linear/tree 或小 separator 只允许 boundary-compatible DP，返回 alternatives 给共同 candidate set，不返回局部 winner；
- global ledger 分别计数 generated、rejected、deduplicated、expanded、actual-probed、accepted 与各 exact gate work。

### Core oracle harness

oracle必须分两层，不能用同一mechanism生成的域反过来证明它自己完整：

1. **independent tiny reference domain enumerator**：直接从tiny structured semantics、verifier legality、tiny topology和target
   facts平铺枚举有限typed choices，不调用production mechanism的domain builder；它与mechanism生成域比较stable
   key集合，发现漏项、多项或重复。
2. **production-mechanism flat exhaustive runner**：使用真实mechanism、materializer、cost比较和exact gates，但禁用排序
   剪枝，对mechanism合法域全展开；再与Q51 candidate set/memo/no-good/dominance逐项对比optimum与actual digest。

两层都通过tiny topology、tiny static extent和真实有限breakpoint得到小空间，而不是通过隐藏cap缩小
production域。第一批至少覆盖single op、two-op chain、independent branches和diamond；每增加一个Q50轴，
都向independent enumerator和flat exhaustive runner加入完整tiny域与至少一组对立choice。

### Gate

- `none` 不构造 Q51 state，`search` 只存在一个 driver；
- 空 mechanism 集、单轴 mock mechanism 和 exact-rejection 回退都能稳定运行；
- serial/parallel immutable analysis与actual cost比较得到相同state/winner digest；
- baseline incumbent、lower-bound、ledger 和 actual-probe 生命周期有 unit test；
- core不含fixed seed、预设beam/candidate cap、workload分支或任一实际轴的局部winner；
- 本checkpoint独立提交，但Q51状态仍为未闭合；随后Q50.S与Q50.B可以在同一core上按依赖施工。

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
- consumer placement 关闭时直接调用 Q50.A exact demand/ownership coverage；physical representation 限制不得反写成
  logical demand 不合法；
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

## Q50.C：Maximal Single-Op TileRegion

### 机制

对给定 op、spatial shard 和尚未细分的 local iterator domain，每个 Tile 物化一个覆盖该 op 所有必需 producer/support
closure 的 maximal single-op TileRegion。这里 maximal 表示“对已选 region boundary 不漏 work、不把同一 local work 任意拆成
多个相互不知情 region”，不是强迫使用最大 tile、最大 fusion group 或单一结果驱动入口。

region materializer 必须由 SSA、structured interface、indexing maps、DPS init/effect 和 observable roots 驱动；不得从 buffer、
op、result 名恢复 traversal。多个 result、init operand、support chain 和 reduction 均要得到 all-and-only local work。

### Gate

- single-result、multi-result、DPS init、reduction 和 side-effect boundary 正负测试通过；
- actual Tile IR 对每个 selected Tile 都有唯一、完整、可 verifier 的 single-op region witness；
- baseline 的 per-op independent TileRegion 复用同一 materializer；
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
- implementation geometry、reduction order 和 buffer-count feasibility；
- actual region probe 产生的 scoped infeasible/feasible boundary。

只有能证明生成相同actual traversal、reuse、lifetime、tail和numeric reduction order的permutation才能canonicalize。Q50.E
不在内部选择load/receive hoist或retention winner；Q50.H在已选wave-loop order下生成这些movement alternatives。

容量失败可在其它轴固定的 branch 内产生二分子状态，再补入区间内所有非二次幂 breakpoint；二分只用于发现 breakpoint，
不能把最大可放下 tile 固化为全局事实。改变 traversal、layout、buffer、placement 或 Q/other iterator tile 后必须重新判断。
seed 只影响顺序，不限制域；late allocator 不得在 accepted candidate 上私下 `tile/2`。

### Gate

- parallel、reduction、多 reduction axis、tail/padding 和不同 shape breakpoint 有 complete-domain property test；
- independent tiny reference enumerator与mechanism生成域的stable key集合相同；涉及尚未施工轴的property
  test必须传入显式typed test assignment；
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
- Q49 baseline probe 与 Q51 probe 共用 downstream mechanics，但 baseline 不进入 candidate set；
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
由current IR epoch与region/representation/movement/order assignment重算。

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
由current IR epoch与这些assignment重建query-local event/resource analysis；calendar与makespan是可失效cache，不是state字段。
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

### Gate

independent branches、fanin/fanout、shared DDR、NoC link contention、multi-worker join、effect ordering和completion hazard有
正负测试；actual Instr顺序、worker和completion可验证，finite normal form的tiny域与independent reference
enumerator一致。删除ready-order、worker和overlap的独立selector；query-local calendar在对应IR epoch/assignment
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
pending data/completion events、resource calendars、lower bound和current makespan都不是state字段。它们是从current IR epoch、
target facts与上述assignments重算的query-local analysis cache；任一依赖assignment变化都必须精确失效。

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
- actual rejection 可继续 candidate set，`search` 无更优 accepted candidate 时返回同源 Q49 baseline；
- selected IR 至少有一个有效 multi-op fusion，中间 actual value resident 且没有无意义 DDR round-trip；
- public policy 只剩 `search|none`；旧 bounded/rank owner、各轴 local winner、performance `Unknown` 和 shadow plan 均已删除；
- Q51 独立提交并标完成后，Q52 才开始改变 search policy 的吞吐与预算行为。

Q51在small-oracle mode完整展开有限域。真实workload从正确性闭合时就支持明确的global work/time中断，
但Q51只交付机制，不在fresh profile之前固定production默认10/30分钟截止或有损策略。中断时返回结果
必须actual accepted：若未展开completion仍由exact continuation与admissible bound完整表示，则标
`feasible-with-bound`；若某些合法completion已被丢弃或从未表示，则标`budgeted-feasible`。不得借
correctness mode在真实负载上无限运行。Q52基于fresh profile选择production budget并改善同一预算下的
time-to-first和incumbent质量。

## Q52：Profile-Driven Anytime Search and LNS

### 目标与时间合同

Q52 的生产目标不是在 10–30 分钟内证明所有真实 workload 全局最优，而是在有限预算内尽快得到合法 actual executable，
持续改善 incumbent，并在能够保留 exact candidate set 时报告可信 lower-bound gap。

- **10 分钟以内**：可接受，但仍记录 work、RSS、time-to-first-actual 和 incumbent 曲线；
- **超过 10 分钟**：不立刻停止，必须形成热点、重复状态和质量停滞归因；
- **最长先观察到 30 分钟**：收集完整 profile；若仍无法承受，才根据已测证据启用明确的 bounded/heuristic policy；
- 到预算返回的 executable 必须 actual accepted 且不劣于同源 Q49 baseline；若永久丢过合法状态，结果标
  `budgeted-feasible`，不能沿用 exact gap 或暗示最优。

预算是 policy 参数和回归证据，不是合法域定义。不得为了满足时间线提前固定 tile、fusion depth、layout、buffer count、
split、placement group 或候选数。

### Fresh profiling

在 generic mixed DAG、HF prefill、functional decode 和 Llama representative block 上记录：

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

1. Q56、Q58、Q59由各自计划先行闭合；Q50.0、Q54、Q49.P、Q50.A、Q51.Core、Q50.S、Q50.B–Q50.K、Q51 closure、
   Q52、Q60与Q53分别形成独立可评审提交；不得把全部迁移积累成一个dirty diff。
2. 每个 checkpoint 开始前记录将替换的旧 owner 调用链；提交前证明新调用链唯一，并只删除当前轴满足三项门禁的旧入口。
3. 状态转换以 `tasks/progress.md` 为准；本计划不单独维护第二份动态状态表。
4. 每项提交前运行 fresh 定向 build/test；端到端或主线 gate 还需确认 relevant lit/CTest 实际执行而非 skip/unsupported。
5. 提交使用 `Codex <codex@openai.com>` 并附 `Co-authored-by: hehesnail <shashen008he@gmail.com>`。
6. 稳定 bug 模式进入 `memory/bugs.md`，可复用 build/debug workflow 进入 `memory/general_dev.md`；临时 profile 数字、
   workload 路径、未校准 prior 和单 case winner 不进入长期设计。
