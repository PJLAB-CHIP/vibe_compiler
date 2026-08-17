# Physical Dataflow Synthesis 实施计划

状态：Q49 deterministic `none` baseline 已于 2026-08-11 达到 `board-ready`，但其search-policy隔离、结构不变量、scoped
probe闭合和整图物化成本不属于已签发的正确性证据。Q50.0无策略CardExecutable编译/准入边界与Q54 MLIR infrastructure
已闭合；Q59 compiler entry transaction在代码review后重开，Q49.P继续等待其重新完成。当前先修复Q50.A production
demand boundary，再由Q49.P在current source-to-package路径上闭合baseline functional legalization及
policy/structure/probe/materialization隔离，之后建立
Q51.Core共同状态、transition和actual-probe seam；随后让
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
  current事实闭合baseline功能合法化及policy、结构、probe和materialization隔离；Q51.Core复用其accepted executable作为初始incumbent
  并建立共同search kernel；Q50.S与Q50.B–Q50.K逐轴交付
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
candidate”不等于搜索失败。`none`对声明支持的正常上游输入必须先完成自己的deterministic feasibility legalization；若最终
exact gate仍失败，必须是最小canonical fallback已被typed proof排除、输入确实超出支持域，或明确的compiler/internal failure，
不能把没有进入performance search当作失败，也不能伪造fallback。

## 施工 checkpoint

| 顺序 | Checkpoint | 施工责任 | 完成后才能开始 |
| --- | --- | --- | --- |
| 0 | Q50.0 CardExecutable compilation boundary | 抽出无策略的actual compile/verification seam，不允许lowering修候选 | Q54 |
| 1 | Q54 MLIR infrastructure conformance | typed IR/interface、scoped pass/analysis、named pipeline与rewrite transaction收口 | Q49.P、Q50.A |
| 2 | Q50.A placement-demand repair | 对完整logical shard trial形成typed exact-demand/coverage proof；reduction、broadcast、window/stride、multi-piece、init与support relation均不被carrier/layout/route反写 | Q49.P、Q51.Core |
| 3 | Q49.P baseline functional closure与policy isolation | canonical `none`从正常上游IR完成确定性placement/tiling/SPM合法化且不消费search对象；一root一region；最窄exact probe后完整CardModule与CardExecutable各形成一次 | Q51.Core、baseline性能复核 |
| 4 | Q51.Core | 共同state、transition、candidate set、incumbent、ledger、budget与oracle harness | Q50.S、Q50.B |
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
| 可复用 core | `StructuredDAGAnalysis`、现有schedule-state/candidate类、edge-demand plan与placement domain/cost mechanics | immutable structured/relation事实及policy-free logical placement-assignment窄类型供baseline构造trial并由exact-demand query读取；该窄类型和query必须从candidate/schedule依赖中抽离。schedule-state/candidate/evaluator/proposal-order只纳入Q51.Core，Q49.P必须先断开对它们的依赖。旧类名只是实现定位，不升级成长期架构对象，也不让其自行选择winner |
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

## Q49 / Q49.P：Deterministic Baseline功能闭环、Policy与结构隔离

Q49已签发的数值正确性、完整Tile domain、零实际fusion和历史board-ready状态不重定义；这些证据没有证明baseline与
search-policy解耦，也没有证明每个TileRegion只有一个structured compute root。Q49.P因此不是对旧baseline做性能润色，也
不是把`none`缩成fixed-assignment verifier；它要同时补齐功能合法化、policy、结构、probe、causal diagnosis和materialization
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
  verified ExecutablePackage。局部probe clone和query-local witness不成为output或shadow plan。
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
  baseline不依赖search state/candidate、search-oriented domain/ranking evaluator、proposal order/group materializer；每个
  baseline TileRegion恰有一个structured compute root和必要non-root support closure；跨root shaped dependency显式DDR；
  region-local和最近合法isolated-ancestor probe均闭合，exact rejection携带direct typed causal witness；完整CardModule materialization与
  CardExecutable compilation各一次；初始完整tile超SPM的正例会重新推导完整workset并确定性缩到合法tile后通过package/no-card，
  最小合法tile超限的反例返回typed capacity failure；baseline diagnostics/statistics不再冒充candidate proposal；fresh
  source-to-package、oracle/no-card、digest、计数和结构正负测试通过。
```

`none`不得构造Q51 state、candidate set、candidate family或全图performance Cartesian组合，也不得通过调用search-oriented
domain/ranking evaluator后只取第一个结果来伪装canonical construction。但它必须复用policy-free placement-domain机制和
Q50.A logical demand/coverage query，对baseline合同内的mandatory coordinates执行有限、完整、确定且不被beam/cap/time budget截断的
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
vector/alignment、source numeric/reassociation、reduction order和最小合法粒度形成有限breakpoint lattice。每个trial必须按
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

Q49与`search`共享immutable structured/relation/target facts、policy-free single-root TileRegion materializer、scoped probe和
actual Card/Tile/Instr、completion、SPM/DDR、verification、package机制；不共享search state/candidate/evaluator、group
boundary、proposal order、score或feedback。Q49.P先产出并准入baseline，Q51只把accepted executable/cost作为incumbent，
不得通过search representation重建同一baseline。Q49.P施工时先把resolved baseline assignment的single-root apply与每次
closed-coordinate feasibility probe落在稳定PhysicalDataflow/Conversion边界；Q50.B/C/F随后扩展完整search domain、
single-root mechanism和deferred
probe taxonomy时必须复用同一实现，不能再建baseline-private或search-private materializer/probe。baseline默认值不限制Q51域。
历史official HF prefill、functional decode、Llama block 的 source、oracle、package 和 no-card runner只证明
旧入口mechanics，未执行真实板端，故不得标`done`。Q49.P的gate必须经Q59 current compile transaction用fresh
prefill/decode/Llama输入证明
CardModule、accepted CardExecutable与package digest稳定，fresh no-card与oracle通过，并以fresh阶段计时和work count确认
完整CardModule materialization与CardExecutable compilation各一次，scoped probe不构造card-shaped/no-work-Tile wrapper。
定向结构测试必须覆盖：同Tile多个独立root形成多个region；显式structured producer不能伪装成support closure且一个root的
lowered multi-op不会误判成多root；call或unsupported lifetime提升到最近合法scope；equal-shape fanin只按direct witness
refinement；search state/candidate、search-oriented domain/ranking
evaluator和grouping调用计数为零，baseline work不写入candidate proposal统计。它不把历史耗时
写成长期阈值，也不得借“统一入口”让`none`再进入search feedback。

定向功能测试还必须从没有selected assignment的正常TensorProgram进入：至少覆盖初始完整tile因operand/halo/temporary/
alignment/lifetime真实占用而溢出、经过多个合法breakpoint后fit并完成source-to-package/no-card；覆盖multi-axis、tail和最小
合法粒度；negative case只有遍历到最小合法vector后才返回direct typed capacity failure。测试同时断言每次trial重新计算
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
不apply mutation；关系或domain需要跨mutation时由新IR epoch重算，不把operation pointer、diagnostic字符串或旁路side table
当稳定语义键。

relation cache至少按IR epoch与typed edge/relation identity失效；demand/coverage cache还必须观察完整consumer execution
domain、producer ownership domains、partition/reduction/replication role及会改变relation image的全部assignment。只有证明某些
字段在当前relation下extensionally等价时才能使用投影key；当前仅按两端shard dimension和participant count缓存`bool`不能成为
终态合同。

当前`StructuredDAGEdgeDemandPlanner`、`StructuredDAGEdgeStrategyPlanner`、placement evaluator与相关测试只是迁移索引：
实现仍经candidate schedule取得placement type，只接受direct static single-result/single-init Linalg与balanced一维矩形，跳过
DPS init和非direct support dependency，并把relation、carrier、route/resource失败混入同一个placement rejection。施工顺序是：

1. 抽出窄policy-free placement assignment、typed dependency descriptor与typed outcome；
2. 让exact relation/image/coverage query直接支持上述完整relation类别，并把DPS init/support graph纳入all-and-only DAG依赖；
3. 将production placement transition改为只消费logical outcome，删除`bool legal`/diagnostic-string控制流；
4. 把dense fragment、layout、route、residency和resource calendar移到显式compatibility/representation/movement lowering；
5. compatibility code保留到Q50.G/H能力迁移完成，但其失败不得回写Q50.A cache或删除spatial trial。

### Gate

- 独立exact-demand suite直接测试query，不以edge-strategy/materializer成功代签；覆盖multi-axis remainder、reduction input/init与
  partial merge、broadcast、affine window+stride+dilation+pad、strided slice/view、multi-piece union、multi-result、fanout/fanin、
  explicit init root及多operand pure support graph；
- metamorphic gate对同一logical trial替换dense/strided/multi-piece carrier能力或让route失败，Q50.A outcome和exact set
  extensionally不变；ownership hole只产生typed `proven logical infeasible`，unsupported与indeterminate不会进入
  legality `bool`或no-good cache；
- cache gate改变任一观察到的domain/role或IR epoch都会miss/invalidate；extensionally等价placement重算得到相同stable logical
  proof，结果不依赖Tile/pointer/hash遍历顺序；
- Q49.P定向调用链不出现candidate/schedule evaluator或edge strategy selector，只对proven logical failure推进canonical
  placement trial；上述五类relation至少各有一个normal TensorProgram进入baseline canonical carrier并继续到完整Q50.0 gate；
- Q50.B explicit test assignment与后续production domain调用同一query；Q50.G/H只消费`satisfied` output，不从physical fragments
  反推另一份logical demand。旧skip-init/support与carrier-rejects-placement测试必须替换为current合同的正负证据。

### 实现检查点与review结论（2026-08-17）

- 新policy-free类型位于`include/Wafer/Analysis/PhysicalDataflow/ExactDemand.h`：`IREpoch`（进程级IR世代
  token，synthesis入口捕获一次、全链query共享）、`LogicalShardTrial`（完整consumer iteration domain +
  per-Tile result-space ownership + UniquePartition/ExplicitReplication/PartialReductionContribution
  角色）、`DemandDependency`（DataInput/InitInput + support chain）、四态`ExactDemandStatus`与
  `ExactDemandResult`（per-owner intersection、uncovered witness、dependencyKind、mergeObligation）。
- typed query位于`lib/Wafer/Compiler/StructuredDAGExactDemandQuery.{h,cpp}`：对每条edge构造consumer
  iteration空间→producer result空间的exact `IndexRelation`；support chain逐op compose
  （expand/collapse→`staticReshape`、extract_slice→`staticSlice`、insert_slice source→
  `staticInsertSlice`、insert_slice dest→identity∩补集、pad→`staticInsertSlice`、transpose→置换map、
  cast→identity），`IndexRelation::staticInsertSlice`为新增builder；coverage为typed union与uncovered
  witness，UniquePartition重叠为可证partition矛盾。多operand support op对每个data-carrying
  predecessor各建一条依赖；同一op多路径指向同一producer为UnsupportedSemanticRelation。状态映射：
  Exact→继续，Unsupported→UnsupportedSemanticRelation，SoundBound/Invalid/ResourceExhausted→
  IndeterminateFailure。query只保留per-edge placement-independent relation memo（实例绑定epoch +
  函数指针防御），不建demand/coverage bool缓存；热路径memo由调用方按完整placement内容或
  域决定字段投影键控，值始终typed；该投影是否足以表达per-consumer-shard demand仍需按下述review阻塞项修正。
- production接线：evaluator `getEdgeEntry`每edge存typed verdict，carrier只作建模输入、其失败置
  `edgeCarrierComplete=false`不删placement；`EdgeTransitionLegalityCache`用投影键但只存typed
  status；baseline EdgeCompatibility对全部edge（data/init/support）由query判定，仅
  ProvenLogicalInfeasible删option pair，Unsupported/Indeterminate写
  `statistics.demandAbortStatus/demandAbortDetail`并终止baseline；seed filter query-gated（只
  ProvenLogicalInfeasible丢seed），carrier best-effort；search worker loop按typed verdict区分reject
  与abort；`materializeCandidate`入口新增`edge-carrier-materialization` gate（exact
  physical-assignment rejection，进preBufferFailureCache），
  `StructuredDAGPlacementCandidate/TileExecutionCandidate`携带`edgeCarrierComplete`标记。
- 生产适配器`buildLogicalShardTrial`/`buildEdgeShardTrial`/`buildBalancedOwnership`从当前balanced单轴
  placement构造trial；当前adapter实际写入node-wide loop domain，并未形成per-consumer-shard execution domain，
  因此只能视为过渡实现。枚举入口`enumerateStructuredDAGPlacements`（当前仅单测路径）的
  final-accept loop同样由typed query判定，
  carrier失败计入`edgeCarrierIncompleteTransitions`。
- 当前`StructuredDAGExactDemandQueryTest.cpp`实际包含17个query case；它们覆盖direct/permuted/strided
  relation、node-wide reduction/broadcast、init、insert/pad support、coverage witness、
  unsupported/indeterminate分类和carrier metamorphic等局部mechanism，但没有证明multi-result、
  per-consumer-shard demand、partial-reduction/replication query、affine window+stride+dilation、
  multi-axis remainder或真实IR mutation invalidation。
- 其它限制（移交Q50.B/Q52/Q50.G/H）：枚举测试的prefix状态空间与`extendState`逐option pairwise拓扑距离成本是
  既有搜索域形状，未在本任务内改动——`DiamondFanin...`/`ReductionDataTransition...`两case在HEAD即>30min
  （状态数100→10,000→1M逐节点膨胀），已与`MultiOutputFanout...`一并登记为既有病理长跑case；
  multi-piece等非dense demand在materialization gate被物理拒绝，Q50.G/H迁移carrier时补上表达与成本罚项。

本次review确认Q50.A尚未满足本节completion gate，阻塞项如下：

1. **production尚未对每个consumer logical shard求demand。** Trial builder将整个Linalg loop domain写入
   `completeIterationDomain`，query只对该node-wide domain求一次image且不读取consumer Tile binding。
   因此consumer shard/Tile变化不会产生相应的per-destination exact demand、ownership intersections和
   witness；逐destination Tile demand仍由`StructuredDAGEdgeDemandPlan`的direct-data-only路径重新从
   balanced result shard求得，不能代签统一logical boundary。
2. **multi-result不可表达且production adapter显式拒绝。** `LogicalTileBinding`没有producer-result
   identity，无法为同一node的不同result表达各自shape/domain/ownership；两个trial builder均固定读取
   `result(0)`并在`getNumResults() != 1`时失败。完成时ownership必须与具体producer result绑定，valid
   multi-result TensorProgram必须经production adapter/query进入下游。
3. **`IREpoch`没有形成真实per-IR mutation invalidation。** 当前进程级全局generation没有production
   mutation owner推进；query只比较trial与自身捕获的epoch及FuncOp指针。原位mutation后old query + old
   trial仍可使用旧relation cache。应由真实IR snapshot/lifetime表达失效边界，不依赖进程级mutable
   singleton作为编译语义。
4. **typed result与确定性合同未闭合。** `ExactDemandResult::consumerIterationDomain`在成功路径没有填充；
   `ownershipIntersections`沿caller binding顺序输出，多overlap时witness也取决于首个遍历pair。完成时应
   补齐consumer domain、dependency/result identity和role payload，并以完整semantic tie-break保证结果
   与Tile、pointer、hash及caller enumeration顺序无关。

以上缺口修复后，需要用对应定向正负case和正常TensorProgram→Q50.0路径重新签发，才能将Q50.A恢复为
`done`；Q52长跑性能与Q50.G/H physical carrier扩展不用于掩盖这些logical boundary缺口。

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
- implementation geometry、reduction order 和 buffer-count feasibility；
- actual region probe 产生的 scoped infeasible/feasible boundary。

只有能证明生成相同actual traversal、reuse、lifetime、tail和numeric reduction order的permutation才能canonicalize。Q50.E
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

1. Q58、Q56、Q50.0与Q54已达到各自当前门禁；Q59按其计划先闭合review阻塞项。随后Q50.A、Q49.P、Q51.Core、Q50.S、
   Q50.B–Q50.K、Q51 closure、Q52、Q60与Q53分别形成独立可评审提交；不得把全部迁移积累成一个dirty diff。
2. 每个 checkpoint 开始前记录将替换的旧 owner 调用链；提交前证明新调用链唯一，并只删除当前轴满足三项门禁的旧入口。
3. 状态转换以 `tasks/progress.md` 为准；本计划不单独维护第二份动态状态表。
4. 每项提交前运行 fresh 定向 build/test；端到端或主线 gate 还需确认 relevant lit/CTest 实际执行而非 skip/unsupported。
5. 提交使用 `Codex <codex@openai.com>` 并附 `Co-authored-by: hehesnail <shashen008he@gmail.com>`。
6. 稳定 bug 模式进入 `memory/bugs.md`，可复用 build/debug workflow 进入 `memory/general_dev.md`；临时 profile 数字、
   workload 路径、未校准 prior 和单 case winner 不进入长期设计。
