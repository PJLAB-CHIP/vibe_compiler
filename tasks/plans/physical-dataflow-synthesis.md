# Physical Dataflow Planning 与 Selected Execution 实施计划

状态：当前线性计划第1--17项已经闭合；Q50.0/Q50.A、Q49.P、Q50.B--K的domain/selected construction/actual admission、
Q51.Core controller及Q51 resumable unified traversal已经连接。下一项是Q50.S `attention-production-closure`，随后按Q52和Q53的
顺序施工。此前关于
baseline incumbent、同一complete-candidate probe/rebuild与winner rematerialization、统一全轴search、scalability/LNS及model-scale search质量的完成声明均不再是
current证据。
算法、IR和长期pipeline contract仍只由
`tasks/06-physical-dataflow-synthesis.md` 拥有；本文件只规定施工依赖、现有代码处置和独立 checkpoint。

本计划的核心约束是：**机制可以独立交付，选择不能独立发生**。每个 Q50 子项只提供合法域、transition、actual
materializer、exact failure 和测试，不保留局部 winner、局部 shortlist、局部 beam 或跨阶段默认选择。Q51 是唯一
physical-dataflow winner owner；Q52只在该正确性基础上改善10–30分钟预算内的anytime质量和搜索吞吐。

## Q51 起实施重审计

本轮审计不以commit标题、新type存在、独立domain测试或package能生成作为能力迁移证明。每项必须同时核对：原设计责任、被删
donor中的独有算法/proof/diagnostic/test witness、current owner、production consumer、query/apply边界和端到端证据。旧实现不整体
恢复；仍需要的能力必须迁入current typed边界并由production调用，之后才能确认旧文件删除成立。

以下硬事实是search-control-foundation切换前的重建审计快照；后文“当前已实现子集与重新打开门禁”及施工checkpoint才是更新后的
current完成判断。旧fresh数字只说明当时实现子集通过，不能覆盖本轮production call-graph复核：

- Q51.Core提交删除了`CardExecutableSynthesis`、rank search/evaluation/selection、graph placement、schedule、cost和attention等
  多个独有算法owner；新`CardExecutableSearch`当时只是baseline passthrough，通用`runSearchControl`从创建到退役始终只有unit
  consumer，没有进入production search。
- current `UnifiedPhysicalDataflowAssignment`只有spatial、coupled、temporal、implementation、representation、movement和buffering；
  Q50.F feasibility、Q50.J instruction schedule和Q50.K stage pipeline不在联合assignment。
- 已删除无production caller且会在IR外推测资源结论的plan-side feasibility/footprint路径。Q50.J只在Tile lowering后构造domain并直接应用first assignment；Q50.K
  由nonempty buffering scope自动触发，不是search choice。
- Q50.G删除了deterministic PBQP/Top-4 layout solver及其fixed-point、movement cost、shared secondary、transactional apply和16项测试；
  删除前`CompleteRankMaterialization`的三条production candidate路径都直接调用该solver，且已有同源package/no-card证据；它不是
  “dormant”。current实现只为每个value独立枚举合法layout的Cartesian product，无跨value约束、solver或cost。
- Q50.H同批删除大量NoC topology/intermediate/partial-dataflow算法和alias/lifetime/slice proof测试；current domain/apply覆盖基本
  DDR/recompute/peer route/multicast/reduction gather，但尚无逐项能力承接证明。
- Q50.K以一个stage-materialization wrapper和2项直接测试替代旧fixed-slot pipeline及40项DTE/NCC/alias/periodic/hazard测试；
  `SelectedBufferMaterialization`另有5项测试，不能证明其余能力已迁移。
- Q51 closure只有一个first-assignment joint test和一个public complete-candidate test；Q52之后也只有constructive与fusion两个
  complete proposal。current代码没有LNS、partial frontier、no-good、admissible lower bound或`feasible-with-bound`生产路径。
- production search默认只evaluation一个complete candidate。该点固定选择original TensorProgram、singleton coupled groups、首个
  legallayout、retained-or-DDR movement和single buffer；semantic alternative、fusion proposal、peer/recompute、多buffer、schedule
  alternative和stage pipeline不能共同构成默认winner。
- CLI在未显式提供`--optimization-policy`时也构造`OptimizationConfig::search(1)`；因此这条只编译first proposal的路径还是隐式
  默认，而不是仅供实验的显式autotuning模式。
- current normal candidate evaluation已经具备“完整候选构造CardModule、lower到Instr、运行SPM/DDR planning并取得actual cost”的必要
  形态，但旧实现没有把buffer relation totality、typed rejection、候选事务销毁和accepted owner复用收敛成同一合同。该路径必须
  原位改成Q50.F actual admission；每个完整候选至多actualize一次，rejected/loser transaction销毁，accepted winner不得重建。

初步任务判定如下；“部分可保留”只表示已有mechanism可能继续复用，不表示任务完成：

| 任务 | current事实 | 初步判定 |
| --- | --- | --- |
| Q51.Core | baseline passthrough加未被production调用的通用control模板；旧控制/代价/placement/schedule owner已删除 | 未实现 |
| Q50.S | attention/decode proof与actual alternative materializer有实质代码；旧11项能力测试缩为5项，且production顺序/预算让alternative饥饿 | mechanism部分可保留，算法与集成重审 |
| Q50.B | per-node all-iterator legality domain和exact-demand trial存在；graph-level compact topology、independent-component、chain DP/general-DAG bounded search未迁 | legality domain可保留，placement算法缺失 |
| Q50.C | single-root apply与多结果/reduction/support/effect测试较完整；后续真实模型仍暴露multi-producer/output closure缺口 | apply部分可保留，完成结论撤回 |
| Q50.D | connected partition domain和coupled apply存在；production proposal只有singleton或greedy merge，旧connection coupling算法未承接 | domain/apply部分可保留，选择算法缺失 |
| Q50.E | 全整数tile size与loop permutation可枚举；旧capacity-guided proposal已删除，尚无factorized breakpoint和联合质量算法 | domain/apply部分可保留，scalability/选择缺失 |
| Q50.F | 已删除无可靠资源事实的pure feasibility/footprint路径；actual admission seam已存在但关系完整性与controller反馈尚未闭合 | 重建为complete-candidate actual admission |
| Q50.G | per-value layout Cartesian domain替代PBQP/Top-4及movement-elimination算法 | 核心算法丢失 |
| Q50.H | exact fragment/basic route/multicast/reduction apply存在；大量NoC/alias/lifetime/partial-dataflow能力无承接证明 | 部分迁移，重大缺口 |
| Q50.I | slot-count domain与rotating allocation mechanism存在；first choice固定single buffer，无联合选择算法 | mechanism部分可保留，选择缺失 |
| Q50.J | 审计时只有hard dependency/order/worker domain且lowering只取first；当前已由post-K EventGraph、closed schedule和selected materialization替代 | domain闭合；production接入由后续full-feasibility负责 |
| Q50.K | wrapper按已选buffer scope自动物化stage，没有独立domain或search transition | 任务合同未实现，proof大幅丢失 |
| Q51 | dependent Cartesian iterator直接materialize每个complete point；缺F/J/K、真实Core、完整typed rejection和actual-result owner合同 | 完成结论无效 |
| Q52 | 两个heuristic proposal、evaluation count cap和六项raw metric字典序；无文档所述anytime/LNS | 完成结论无效 |
| Q53 | source/oracle/runner及`none`证据可单独复用；旧search可能返回baseline且winner trace错配 | search证据无效，禁止据此board-ready |

第一批donor对照已经确认下列直接删除关系。测试数量只是审计入口，不单独证明能力优劣；关键是旧test witness表达的语义是否已有
current owner和production consumer：

| 责任 | 退役donor与独有能力 | current替代与缺口 |
| --- | --- | --- |
| Q51 control / quality | `CardExecutableSynthesis`约7556行，rank search/evaluation/selection约5600行，`StructuredCandidateCost`约1419行，另有placement/schedule plan；覆盖connection DP/beam、work reservation、actual action backfill、cost margin、unknown metric和duration selection | Core提交只有12行baseline passthrough和未被production调用的202行模板；Q51/Q52后来没有恢复这些算法 |
| Q50.S | old structured implementation/attention tests 11项，包含scalar seeds、official HF、compute prior、output view/destination | current alternative tests 5项；actual builder较完整，但能力parity与production可达性未证明 |
| Q50.B | `StructuredDAGPlacementEnumeration`约2218行及15项placement tests，覆盖compact rectangle、independent components、long-DAG nondominated states、fanout与three-stage groups | current 7项domain tests；保留合法域，缺graph-level构造与质量算法 |
| Q50.G | `LayoutMovementOptimization`约1587行，PBQP reduction、dominance、bounded exact core、Top-4、movement/footprint cost、exact unary fixed point、shared secondary和transactional apply；16项tests | current `PhysicalRepresentation`约224行和3项tests，只做独立value layout Cartesian枚举 |
| Q50.H | NoC communication/intermediate/partial-dataflow、complete-rank/topology及相关tests同批净删除约1.34万行；36项直接transport tests覆盖owner rotation、tree/ring partial spill elision、alias overwrite、dealloc/use-before-reload、slice overlap/tail/alignment等 | current movement domain/apply约1480行和6项direct tests，另有若干CardModule tests；基础机制存在，但没有逐项承接上述proof |
| Q50.J | `ReadyOrder`、`WorkerPlacement`及target scheduling analysis共约1300行；22项direct tests覆盖DTE issue/wait window、RAW/WAR/WAW/view alias、completion barrier和worker components | 已迁移到EventGraph、ScheduleDomain、ScheduleMaterialization及Q63/Q50.0 verifier；旧InstructionSchedule source/test已删除 |
| Q50.K | `FixedSlotPipeline`约1902行、40项tests，覆盖2/3-stage、odd tail、periodic DTE、NCC backedge、endpoint reuse、alias/external root和atomic rejection | current `StagePipeline`约67行、2项tests，加`SelectedBufferMaterialization`5项tests；大部分proof没有等价承接记录 |

current production调用链也已核对：

```text
wafer-compile search
  -> [committed实现先完整compile Q49.P baseline；工作树中的删除尚属审计期修正]
  -> CardExecutableSearch
     -> TensorProgramAlternativeDomain（original first）
     -> UnifiedPhysicalDataflowDomain
        assignment = spatial + coupled + temporal + implementation
                   + representation + movement + buffering
        -> materialize complete CardModule
        -> build selected buffering scopes
     -> Q50.0 complete CardModule-to-CardExecutable
        -> stage pipeline：buffering scope非空即自动materialize
        -> instruction schedule：构造domain后直接apply first assignment
        -> SPM/DDR planning与final cost
     -> 与已有结果比较后销毁loser
```

这条链没有Q50.F调用，没有Q50.J/K assignment，也没有plan-level partial frontier。默认一次evaluation只执行constructive proposal；
其余轴的first值分别是original root、singleton coupled group、Tensor-first layout、retained-or-DDR movement和single buffer。因此轻量
source-to-package成功只能证明一个固定构造点能lower，不能证明Q50.S attention能力迁移、Q50.B–K联合搜索或质量改善。

现有验证证据的有效范围同时收缩：

- Q51.Core的finite-state tests只执行未接入production的模板，不能证明public search control。
- Q50.S现有测试只证明部分attention/decode graph mechanics，Q50.B–K多数测试分别证明domain membership、enumeration或直接apply；
  没有共同测试证明旧算法能力完整迁移，也没有证明所有physical axes被
  同一个production state消费。
- Q51 closure提交只有一个first-joint-assignment actual gate和一个public complete-candidate test；它没有覆盖F/J/K、第二个合法点、
  partial frontier、winner选择、winner reuse或唯一publication。
- public search lit只检查一个candidate可生成16-Tile package并通过no-card；旧版本允许candidate失败后返回baseline，当前审计期测试
  即使证明不调用baseline，也仍只覆盖fixed first proposal。
- Q52 model profile记录的是完整baseline加一个actual candidate的混合事务，不能证明search-only吞吐、LNS、质量曲线或相对baseline改善。
- Q53 runner目前只核对16份TileDataflow/Instr/LLVM dump存在、非空且层级名称合理；没有验证multi-op fusion、中间SPM lifetime、
  DDR round-trip消除、movement选择、schedule/pipeline alternative或winner相对`none`的cost差异。旧search TileDataflow trace还可能与
  最终executable不属于同一candidate。
- 当前没有Q53 matched真实板端A/B结果；no-card只证明package/runtime接口和host oracle链，不证明性能。

审计闭合前：不提交Q51+完成状态，不运行重型LLaMA search，不以旧profile或旧package签发当前结论，也不继续删除donor或测试资产。
下一步为每项建立“旧能力→current owner→production caller→测试”的完整对照，并据此决定迁移、重写或有依据地淘汰。

### 重建顺序

1. **安全路由**：在unified-search-closure完成前，默认产品编译不得隐式进入旧first-proposal路径；`none`保持独立，显式
   `search`只进入new planning owner，未完成时typed failure。
2. **一次性work item**：施工身份只使用`tasks/progress.md`列出的semantic work item；Q50.*是设计owner，不进入队列反复切状态。
3. **Artifact顺序**：每个work item必须有唯一producer、直接consumer和done witness；当前线性顺序以本计划checkpoint表及
   `tasks/progress.md`为共同事实源。
4. **Policy隔离**：deterministic-baseline-closure与search work items不互相消费；二者只共享policy-free schema/query/materializer。
5. **Actual admission与唯一发布**：partial state展开不构造IR；每个complete candidate进入一次独立actual admission。controller只比较
   actual accepted结果并发布一个winner，rejected/loser transaction销毁，winner不重新物化或重新编译。
6. **验证终点**：attention-production-closure之后才能进入search-scalability；production-host-readiness只完成Q53 board-ready，
   不执行真实设备qualification。

## Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  GSPMD 已完成 card-level partition 且 target-independent canonicalization 完成后的 card-local structured
  TensorProgram；current SSA、structured semantics、IndexRelation、effect、type、shape 和 dtype 均可验证，尚未绑定
  Tile、temporal schedule 或 storage action。
- Current stage responsibility:
  typed `none`由Q49.P deterministic feasibility controller从原始TensorProgram完成canonical placement、single-root
  TileRegion、temporal/SPM、representation/movement、single-buffer、order/completion功能合法化；05/Q50.S在两条policy分叉前从
  typed SSA产生fixed attention semantic roots；typed `search`由唯一query-local physical-dataflow search owner从B spatial axis开始联合展开
  spatial partition/placement、TileRegion、coupled traversal、temporal tiling、layout/representation、
  explicit movement、buffer、ready/order/worker、NoC/DDR/compute resource timeline和条件式stage pipeline。partial assignment只存在于
  typed planning state；每个complete assignment必须在独立transaction中物化actual IR并跨越共同compile/verification seam，actual
  SPM/DDR/transport结果返回controller。只有一个accepted actual owner最终进入package发布，且不得重建。
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
- Done criteria:
  Q50.0先建立共同CardExecutable compile/verification seam；Q54按19号合同收口MLIR infrastructure；Q58/Q56/Q59先闭合
  program data ownership、package data与compile commit。当前20个execution work items按`tasks/progress.md`唯一顺序逐项闭合：
  已建立的semantic/spatial/demand与canonical plan artifacts继续作为输入；先重新闭合deterministic baseline的hardware-evidence-driven
  completion normal form，随后建立完整physical domains、search controller、
  unified complete-candidate actual evaluation、唯一winner发布、attention production evidence和measured scalability。最后production-host-readiness从Q60产品入口生成
  fresh package、oracle、runner并通过no-card达到board-ready；真实matched板端A/B不在当前目标内。
```

Q50.S只把完整attention归一为一个带fixed FA/FD mode的semantic op，不返回graph assignment或algorithm domain。Q50.B/E分别选择
K2 spatial partition与temporal block，A提供coupled partial/merge；每个complete attention candidate在自己的actual transaction中物化
selected Linalg/Tensor/SCF并转换为wafer.tile。physical placement、temporal tile、layout、buffer和schedule仍由同一后续联合搜索选择；
只有最终accepted winner的actual owner进入发布。

## 全程不变量

### 测试shape默认规则

所有checkpoint中的IR、analysis、planning、rewrite、conversion、materialization和lowering正例默认必须使用真实workload
规模：rank至少为3，至少一个主要迭代维度不小于1024；涉及partition、tiling、loop或resource lifetime时成对覆盖
`1024`等整除长度与`1025`、`1031`等非整除长度，实际经过均匀块、多Tile、多block/wave、remainder和tail。只解析、查询或改写static IR的case不会按
logical element分配内存，不得为了测试书写方便退化成个位数shape。小shape只用于有界逐点穷举oracle、最小verifier负例、
scalar/zero-rank合同或单一故障定位，并在case中说明原因；同一机制仍须有真实规模覆盖矩阵。每个checkpoint按相关语义
等价类覆盖单轴/多轴、fan-in/fan-out、broadcast、reduction、view/slice等路径，并断言exact coverage、无重叠、owner、
demand、merge、tail和直接下游结果；任意单个case成功都不构成完成证明。本规则是测试覆盖要求，不进入IR legality、
shape matcher、candidate domain或cost policy。

### 唯一选择 owner

Q51 search invocation 是唯一允许比较完整 candidate 并更新 incumbent 的对象。任一 mechanism API 只能返回：

1. 从 current IR 和已关闭的 partial assignment 派生的有限合法域；
2. 一个或多个可回到共同 candidate set 的 typed transition；
3. complete-candidate construction使用的prepare/emitter primitive；
4. 作用域明确的accepted facts、`deferred(required coordinates)`、proven exact rejection、unsupported或indeterminate result。

mechanism 不得返回“本轴最佳值”，不得按估算删除其它轴仍可能使之变优的候选，不得在失败时自行 retile、改 layout、
spill、减 buffer 或改 schedule。estimate 只能排序或构成已证明的 lower bound；最终 legality 和 cost 来自 actual IR。
`ResourceExhausted`、solver timeout属于Indeterminate，不形成rejection/no-good；controller按coverage/work policy报告unresolved或停止，
不得以retry为由重新物化同一candidate。只有producer-supplied extension-closed exact proof才能形成causal `ForbiddenAssignment`。
Unsupported不形成no-good，internal failure是compiler bug并停止compile。

### Complete-candidate actual admission与发布边界

partial planning与mutable IR分开；资源合法性只在complete candidate actual admission中产生：

```text
current immutable structured IR
  -> query-local analysis/domain/transition
  -> complete assignment
  -> isolated candidate materialization
  -> Q50.0 actual lowering / SPM / DDR / transport / verification
  -> Accepted(actual owner + actual cost) | typed rejection/failure
  -> controller比较accepted results
  -> publish exactly one retained winner
```

- query-local state 只保存不能从 current IR 和已选坐标重算的 typed assignments；ready/live、exact
  demand、structural lifetime/resource calendar、performance lower bound 和 makespan estimate都绑定一次immutable IR borrow并按typed
  assignment重算，是query-local analysis cache，不进入state identity，也不序列化为output或计划attr；
- actual transformation不修改原source，也不在candidate IR内repair；每个complete assignment只进入该边界一次。带完整actual witness的
  exact rejection返回controller处理其它typed sibling；无owner demand、unsupported、resource exhaustion、timeout或compiler failure不得
  伪装成容量拒绝；
- analysis cache只在一个immutable IR borrow或显式planning session内存活；Q50.A使用MLIR analysis invalidation和session close，
  不保留manual epoch、fingerprint或nested snapshot协议。query key必须包含target facts和会影响结论的全部typed assignments；
  改变traversal、Tile、layout、movement、buffer或schedule后，旧calendar、lifetime、SPM/legality结果全部失效；
- controller明确拥有每个live actual transaction；实现可以限制同时live数量或只保留incumbent，但不得因销毁loser而重建winner。
  host可并行计算immutable analysis，candidate result合并与tie-break使用稳定semantic key；
- regular mapping、reuse signature和coarse resource estimate只能给普通typed transitions排序；不能clone-per-mapping，不能把
  reuse/cost annotation写进候选IR，也不能用function name、JSON或opaque solver payload跨越compile seam；
- 每个complete candidate各执行一次完整CardModule splitting、TileRegion-to-Instr、fresh completion、SPM/DDR、resource、ABI和
  verification；partial state不执行这些阶段。最终只发布一个accepted owner，winner不重新物化或编译。

### Search result 等级

- `objective-optimal`：finite domain已完整覆盖或所有未展开状态被exact infeasibility/equivalence/admissible bound排除，已证明current
  planning objective optimum及tie key；
- `feasible-with-bound`：未展开 completion 仍被完整 exact candidate set（包括可惰性展开的 parent）表示，且
  admissible lower bound 有效；在证明最优前因 work/time budget 中止也可报告有效 gap；
- `feasible-unranked`：至少一个complete candidate已通过actual admission，但objective Unknown或存在incomparable accepted results；只作functional选择；
- `budgeted-feasible`：fixed-width/diverse candidate set、LNS-only 或其它启发式已永久丢弃或未表示某些合法
  completion，或者remaining state没有admissible bound；不得宣称global bound或最优。

所有search等级只描述本次search session仍表示的partial states、complete candidates和actual results，不接收Q49.P `none`结果。预算内没有任何actual accepted search candidate就是typed
search failure，不能隐式运行baseline或伪造fallback。`none`对声明支持的正常上游输入必须
独立完成自己的deterministic feasibility legalization；若最终exact gate仍失败，必须是最小canonical路径已被typed proof排除、
输入确实超出支持域，或明确的compiler/internal failure。

## 施工 checkpoint

| 顺序 | Work item | 状态 | 设计owner | 单一输出责任 | 逐项执行门禁 | 下一work item |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `deterministic-baseline-closure` | `done` | Q49.P | 保留actual candidate→SPM→typed feedback闭环；删除per-block/per-element/structural join与unproved immediate await，fresh none满足最小completion及动态work gate | 读AGENTS/progress→读编号设计与本项覆盖矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→实现代码/测试→fresh验证→按设计与LLVM/MLIR规范复审实现、diff和下游witness→更新状态并提交 | region-execution-domain |
| 2 | `spatial-domain` | `done` | Q50.B | complete spatial successors、reference enumerator及proposal | 读AGENTS/progress→读编号设计与本项覆盖矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→实现代码/测试→fresh验证→按设计与LLVM/MLIR规范复审实现、diff和下游witness→更新状态并提交 | search-control-foundation |
| 3 | `search-control-foundation` | `done` | Q51.Core | SpatialState frontier/continuation及public search routing | 读AGENTS/progress→读编号设计与本项覆盖矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→实现代码/测试→fresh验证→按设计与LLVM/MLIR规范复审实现、diff和下游witness→更新状态并提交 | root-work-domain |
| 4 | `root-work-domain` | `done` | Q50.C | full root/merge work domain、Core consumer及complete-candidate emitter输入 | 读AGENTS/progress→读编号设计与本项覆盖矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→实现代码/测试→fresh验证→按设计与LLVM/MLIR规范复审实现、diff和下游witness→更新状态并提交 | region-execution-domain |
| 5 | `region-execution-domain` | `done` | Q50.D | 完整region/execution/use-binding域、selected RegionPlan直接构造、nested/replica/coupled verifier及actual downstream witness；region builder不选择worker/participant/completion | 读AGENTS/progress→读编号设计与本项覆盖矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→实现代码/测试→fresh验证→按设计与LLVM/MLIR规范复审实现、diff和下游witness→更新状态并提交 | temporal-domain |
| 6 | `temporal-domain` | `done` | Q50.E | complete temporal sizes/orders/tails、top-level/nested/coupled actual loop construction、verifier及Core consumer | 读AGENTS/progress→读编号设计与本项覆盖矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→实现代码/测试→fresh验证→按设计与LLVM/MLIR规范复审实现、diff和下游witness→更新状态并提交 | partial-feasibility |
| 7 | `partial-feasibility` | `done` | Q50.F | 复核A–E结构完整性/missing coordinates及5/6新schema，资源合法性保持unknown | 读AGENTS/progress→读编号设计与本项覆盖矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→实现代码/测试→fresh验证→按设计与LLVM/MLIR规范复审实现、diff和下游witness→更新状态并提交 | layout-domain |
| 8 | `layout-domain` | `done` | Q50.G | operation/interface constraint graph、PBQP精确消元+residual solver、production tuple/alias facts及selected physical-version construction/verifier | 读AGENTS/progress→读编号设计与本项覆盖矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→实现代码/测试→fresh验证→按设计与LLVM/MLIR规范复审实现、diff和下游witness→更新状态并提交 | movement-domain |
| 9 | `movement-domain` | `done` | Q50.H | local/DDR/direct/relay/fanout/gather current完整域、external every-root reuse、exact multi-piece payload proof、token-only selected construction/verifier及actual surgery donor retirement；不在issue后立即await；raw collective因无current typed primitive而无state | 读AGENTS/progress→读编号设计与本项覆盖矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→实现代码/测试→fresh验证→按设计与LLVM/MLIR规范复审实现、diff和下游witness→更新状态并提交 | storage-domain |
| 10 | `storage-domain` | `done` | Q50.I | production alias/reuse/1..U requirements、peer-relay exact-piece ownership、selected object/multi-axis rotation construction及actual definition/use/completion/release lifetime verifier；零SPM估算或capacity控制流 | 读AGENTS/progress→读编号设计与本项覆盖矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→实现代码/测试→fresh验证→按设计与LLVM/MLIR规范复审实现、diff和下游witness→更新状态并提交 | event-resource-foundation |
| 11 | `event-resource-foundation` | `done` | Q50.J | 完整EventGraph/resource/recurrence/completion facts、Q63/effect接入及fixed-K后同一builder重建J的typed seam；missing contract保持typed unknown而非默认Synchronous | 读AGENTS/progress→读编号设计与本项覆盖矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→实现代码/测试→fresh验证→按设计与LLVM/MLIR规范复审实现、diff和下游witness→更新状态并提交 | execution-structure-domain |
| 12 | `execution-structure-domain` | `done` | Q50.K | sound Serialized/Pipelined eligibility与完整有限域、selected phase/loop construction、structure verifier及donor retirement | 读AGENTS/progress→读编号设计与本项覆盖矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→实现代码/测试→fresh验证→按设计与LLVM/MLIR规范复审实现、diff和下游witness→更新状态并提交 | structure-specific-storage |
| 13 | `structure-specific-storage` | `done` | Q50.I | fixed-K occurrence/slot/lifetime重闭、actual rotating-slot construction，并触发post-K EventGraph重建 | 读AGENTS/progress→读编号设计与本项覆盖矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→实现代码/测试→fresh验证→按设计与LLVM/MLIR规范复审实现、diff和下游witness→更新状态并提交 | schedule-domain |
| 14 | `schedule-domain` | `done` | Q50.J | post-K EventGraph、slot/FSM lifetime、worker/resource/completion完整域及minimum-participant/latest-unavoidable selected wait/join emission/verifier | 读AGENTS/progress→读编号设计与本项覆盖矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→实现代码/测试→fresh验证→按设计与LLVM/MLIR规范复审实现、diff和下游witness→更新状态并提交 | full-feasibility |
| 15 | `full-feasibility` | `done` | Q50.F | 全字段complete-candidate materialization、actual SPM/DDR/transport/target gate、plan/actual join-wait parity与动态work gate、typed rejection及Core反馈 | 读AGENTS/progress→读编号设计与本项覆盖矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→实现代码/测试→fresh验证→按设计与LLVM/MLIR规范复审实现、diff和下游witness→更新状态并提交 | search-control-closure |
| 16 | `search-control-closure` | `done` | Q51.Core | all-axis CompleteCandidateKey、actual-result admission、cost/bound、causal no-good、coverage及independent controller oracle | 读AGENTS/progress→读编号设计与本项覆盖矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→实现代码/测试→fresh验证→按设计与LLVM/MLIR规范复审实现、diff和下游witness→更新状态并提交 | unified-search-closure |
| 17 | `unified-search-closure` | `done` | Q51 | parent-by-parent/full-plan oracle、可恢复完整遍历、每complete candidate一次actual evaluation及唯一winner发布 | 读AGENTS/progress→读编号设计与本项覆盖矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→实现代码/测试→fresh验证→按设计与LLVM/MLIR规范复审实现、diff和下游witness→更新状态并提交 | attention-production-closure |
| 18 | `attention-production-closure` | `doing` | Q50.S | donor retirement及prefill/decode none/search package/no-card；attention algorithm层零固定worker、零per-K2-block completion | 读AGENTS/progress→读编号设计与本项覆盖矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→实现代码/测试→fresh验证→按设计与LLVM/MLIR规范复审实现、diff和下游witness→更新状态并提交 | search-scalability |
| 19 | `search-scalability` | `queued` | Q52 | measured memo/DP/bound/LNS及有限预算LLaMA actual evaluation | 读AGENTS/progress→读编号设计与本项覆盖矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→实现代码/测试→fresh验证→按设计与LLVM/MLIR规范复审实现、diff和下游witness→更新状态并提交 | production-host-readiness |
| 20 | `production-host-readiness` | `queued` | Q53 | fresh source/IR/package/oracle/runner/no-card矩阵 | 读AGENTS/progress→读编号设计与本项覆盖矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→实现代码/测试→fresh验证→按设计与LLVM/MLIR规范复审实现、diff和下游witness→更新状态并提交 | Q53 board-ready |

表中读取硬件/ABI事实源是算法调研之前的独立正确性门禁：涉及completion、resource、memory hierarchy或target行为时，必须先读
对应`docs/`、current lowering和CRT/runtime实现，将结论标为`supported`、`board-observed`、`unknown`或`excluded`；不能用通用
同步经验、operation类别或结构边界填补unknown。表中“调研”指向仓库外已有知识与成熟实践：优先阅读原始论文、算法资料、官方设计文档和相关compiler的真实实现，比较候选算法的
完整性、最坏复杂度、正确性/近似边界、工程代价及与Wafer IR边界的适配。调研产出必须在对应owner小节记录采用方案、未采用方案和
原因，然后才查询官方LLVM/MLIR文档及仓库pinned源码确认可用API。仅阅读本仓或pinned源码、找到一个可调用helper，不构成算法调研，
也不能开始编码。

### 当前闭合边界

第1--14项已经按线性依赖闭合。current `CompleteCandidatePlan`携带Region、Temporal、Representation、Movement、Buffer和Execution
Structure；selected Region/representation直接构造Card IR，movement builder替换current donor，post-K storage与J schedule在同一owned
candidate上落地。第15项负责把这些stage通过Q50.0的两个typed preparation边界接到唯一actual gate。attention-specific target completion
contract及prefill/decode产品证据仍由队列第18项拥有；Q50.F只接收已经形成完整`ScheduledState`的输入，不在actual admission里猜缺失的
attention worker或completion。

剩余第16--20项依次关闭all-axis actual-result controller、完整可恢复遍历与winner handoff、attention production、measured scalability和
Q53 host board-ready。SPM合法性在全部后续工作中仍只接受actual current IR上的`PlanSPMMemory`/MiniMalloc结果；不引入footprint、bytes
比例或预测lifetime准入路径。

work item是唯一调度身份，Q50.*只表示设计owner。一个owner可以拥有多个work item，但每个work item只出现一次、只签发一个typed
artifact或production gate；owner整体完成由`tasks/progress.md`的owner map汇总，不进入施工队列反复切状态。

第2--4项当前已满足的foundational artifacts继续作为输入；Q49.P中与同步无关的exact demand、actual SPM反馈和policy隔离实现可原位
复用，但必须重新通过本节completion gate。历史状态、case ledger和fresh数字由
`tasks/archive/completed-task-index.md`与Git记录。若复核发现某个current work item没有满足自己的production完成条件，就在
`tasks/progress.md`中把同一semantic work item重新打开并重放直接后继；不得旁挂`*-closure`影子任务、复制动态状态表或以历史`done`
阻止修复。真正已被后继吸收且不在current queue中的历史任务仍不恢复。

每个current work item转为`doing`且写代码前，必须先完成上述算法/实现调研及取舍记录，并在自己的owner小节列出本项覆盖矩阵：
1024级整除/非整除、适用的单轴/多轴与结构语义类别、positive/typed failure、exact输出字段和直接下游witness；不适用项写明理由。
实现后逐行绑定fresh结果，并在复读owner设计与LLVM/MLIR工程合同后才能更新状态。全局shape规则、测试总数或一个success case都不能代签。

### 2026-08-23 Completion、join与wait专项审计

#### 硬件与ABI事实

本轮只采用current硬件/ABI证据，不用通用异步编程经验补规则：

- TX81 same-worker RAW/WAR/WAW/RAR的24个case、72个sample证明busytable与issue order能落实已存在的有界地址依赖；
  wait-each相对wait-once稳定增加plan cycles。普通same-worker链、无条件loop backedge和跨迭代slot reuse不能逐edge drain。
- `wafer.instr.ncc_join`经target lowering成为`wafer_tx81_ncc_join(participant_mask)`；CRT在fence/sync之间对每个mask participant
  调用`TsmWaitfinish_bywork`。它是阻塞的worker-domain drain，不是无成本IR marker。
- Direct DTE是独立completion domain。current target lowering发射prepare后显式发射issue；CRT issue路径执行peer-ready同步、
  endpoint attach和真实`send_async`，wait路径负责completion与release。当前receiver资源只有4个FSM；structured whole-program
  verification必须证明live-range四着色和wait graph无环。
- 因此NCC join只允许出现在current effect/range/lifetime证明的cross-worker、NCC→Kcore/Direct-DTE/host、actual reuse/release或
  observable terminal cut，并只完成真实pending participant；DTE wait必须位于recv first read、send/relay last release或实际FSM/slot
  reuse之前。缺少证据时是unknown并typed defer/reject，不能插一个“保守”drain，也不能把所有token立即await。

#### 2026-08-23审计偏差与处置

| 位置 | 审计时行为 | 为什么不符合合同 | 当前处置与后续owner |
| --- | --- | --- | --- |
| `SelectedAttentionDecomposition.cpp`、`TensorControlFlowLowering.cpp` | 每个K2 block state update创建带worker0 participant的`TensorCompletionOp`，随后直接lower为`SyncNCCJoinOp` | algorithm decomposition在worker/order/lifetime尚未选择时固定schedule；join动态数随K2 block数增长并清空本可跨block保留的NCC窗口 | Q49.P已删除该op和lowering；Q50.S只保留SSA running max/sum/accumulator，后续Q50.J从selected actual effects放置completion |
| `TensorControlFlowLowering.cpp` external/cache copy | copy tile固定为各维1；每个leaf分配1-element SPM，发射load、store、worker0 join并dealloc | 对rank-4 decode cache prefix，DMA/allocation/join按logical element数线性增长；这不是保守正确性，而是把结构化copy退化为逐元素阻塞程序 | Q49.P已改为selected output temporal tile的exact SCF main/tail traversal并由actual SPM planner验证；E/H/I/J继续拥有完整search选择、movement、lifetime与completion |
| `BodyEmitter.cpp` peer endpoint | streamed和non-streamed send/recv刚产生token就立即`async.await`；`PeerLowering`再lower为`InstrDTEWaitOp` | H在J选择order、first read、last release和FSM allocation前把通信串行化，合法overlap window消失 | Q49.P已使non-streamed token延迟到first read、last release、resource reuse或terminal；streamed scratch只在actual store/dealloc前wait。H/I/J继续扩完整selected域 |
| `WaferTileRegionToInstr.cpp` required-join reconstruction | 每个TileRegion terminator闭合仍访问region-local roots的worker；managed materialization WDMA后和reload前固定闭合same worker | region ownership、WDMA和store/reload类别本身都不是hardware completion event；current规则把结构边界当同步proof | Q49.P已删除region/materialization特殊规则并建立ordered-pending lifetime；Q50.J仍负责selected schedule的完整latest-unavoidable placement |
| `EventGraph.cpp` | execution contract缺失时默认`CompletionProtocol::Synchronous`；DDR/no-hop movement也默认Synchronous | missing typed fact被静默解释为同步完成，掩盖真实RDMA/WDMA async domain和Q63接线缺口 | `event-resource-foundation`必须从Q63和H typed action取得completion；缺失为typed malformed/deferred，不得default Synchronous |
| `ScheduleDomain.cpp` | 每个completion obligation固定放在自己的completion event | 没有枚举或推导latest-unavoidable legal boundary，schedule axis实际未关闭 | `schedule-domain`在fixed K/I EventGraph上枚举有限EventBoundaryId，满足hard/lifetime/resource edge后选择最晚最弱合法点并生成actual IR |
| canonical schedule/attention projection | canonical prefix把全部node绑定worker0，`CanonicalAttentionWorkProjection`也只接受该canonical coordinate | 对canonical `none`这是合法检查，但它证明该helper不能直接充当noncanonical selected-search consumer；worker0不能成为attention语义 | canonical helper继续只服务canonical producer；selected attention construction从worker-independent work facts与closed J plan分别取值，不复用canonical worker0检查缩小J合法域 |
| target/cost/controller | 每个join都会发射真实target call；ExecutionCost已统计total/steady/nonterminal join与participant waits，但public search首个Accepted即停止 | cost计数本身不能消除已硬编码同步，也不能在没有候选比较时选择更好plan | 先由H/I/J构造合法最小completion，再由Q51比较actual cost；`full-feasibility`检查plan/actual join-wait parity和动态work |
| donor/inactive路径 | old `DataMovementApply`含immediate await和同步surgery；selected-buffer donor曾把storage与同步混装 | 当前没有production caller不等于可作为后续迁移模板；直接接回会重新引入同类错误 | H/I/K已迁移typed payload、slot/lifetime与phase witnesses并删除两套actual donor；wait/join placement统一交给J |

Q63保持`done`：pure target completion protocol、MLIR operation interface和current-IR pending analysis的分层是正确的。本轮不新增
第二completion interface，也不让Q63选择worker/order；Q49.P已清除审计发现的production绕行，剩余缺口是Q50.J尚未把这些facts扩展成
完整selected schedule domain。

#### 测试审计与新门禁

审计时的fresh定向结果揭示“测试通过但合同错误”：attention decomposition测试通过时仍明确要求每个K2 block一个
`TensorCompletionOp`；movement builder测试正确要求matching send/recv token且零immediate await；TileRegion-to-Instr lit同时包含正确的
unconditional same-worker loop零join，以及要求每个region/materialization产生join的旧期望。Q49.P已经随实现修正这些错误期望；这条
教训继续作为后续门禁，不能把与合同相反的绿灯当完成证明。

每个受影响work item的覆盖矩阵增加以下exact gate：

1. FA/FD aligned/ragged实际candidate在没有typed cross-domain cut时，`steadyStateNCCJoinCount == 0`且
   `nonTerminalNCCJoinCount == 0`；K2 block数增加不增加join数。真实terminal或cross-worker正例仍保留minimum participant join。
2. rank>=3、1024/1025/1031 external/cache copy检查DMA、allocation、join的static site和dynamic execution count；DMA执行数必须
   等于selected temporal tile count而不是logical element count，allocation site保持有界，steady/nonterminal join为0；actual SPM
   offsets与tail coverage仍由MiniMalloc和IR witness证明。
3. peer direct/relay/fanout/gather分别检查token-only issue window、recv-before-first-read、send/relay-before-last-release、最多4个overlap
   receiver live ranges、全卡无环wait graph和matching dynamic occurrence；即时wait只在这些hard facts迫使时出现。
4. required-join正负例分别覆盖unconditional/conditional same-worker loop、cross-worker RAW/WAR/WAW、NCC→Kcore、NCC→DTE、
   host-observed writeback、TileRegion sibling residency、managed store/reload与terminal return；断言位置、participant和pending-set变化，
   不只断言“有join”。
5. `full-feasibility`与Q53 accepted-IR inspection从actual Instr重算total/steady/nonterminal join、participant wait和DTE wait，并与
   `ClosedSchedulePlan`逐项parity；没有plan owner的同步是compiler bug，计划要求的同步缺失则typed reject。

work item责任据此固定：Q49.P已经关闭current production regression；D/E不得写同步；H只产生movement/token；I及post-K I拥有
lifetime/reuse；J foundation取得完整typed completion facts，J closure拥有全部worker/order/wait/join placement；F检查actual gate；
Q51按actual result比较；Q50.S删除attention per-block completion。后续任何本项开始编码前都要重新阅读本节、11/13号合同和对应
硬件/ABI事实源。

Q49.P的实现不建立baseline-only completion算法。它给唯一policy-free completion constructor传入canonical actual
order/worker、Q63 effects、H token和current lifetime，取得确定性的最小completion；后续J closure向同一constructor传selected
order/worker/boundary。Q49.P只关闭这条shared primitive及`none`调用点，不签发J domain、worker alternatives或search schedule完成。

### Artifact producer / consumer 依赖审计

施工顺序以typed artifact DAG为准，不以任务编号或章节顺序为准：

| Artifact / checkpoint | 唯一producer | 必须消费 | 首个真实consumer | 失效 / re-entry |
| --- | --- | --- | --- | --- |
| `SpatialPlan/SpatialAssignment` schema、close与validator | spatial-plan-schema | typed spatial fields、structured iterators、topology | canonical-spatial-assignment、spatial-domain | schema原位演进；不携canonical选择或FA/FD field |
| fixed attention semantic facts | attention-normalization | normalized TensorProgram、attention op/interface、FA/FD classifier | attention-spatial-integration、exact-demand-boundary、attention-demand-integration | source normalization改变后重建；不进入candidate state |
| attention spatial constraints | attention-spatial-integration | spatial schema、fixed attention facts、K1/K2 roles | canonical-spatial-assignment、spatial-domain | semantic mode或iterator relation改变后重建；不进入candidate state |
| canonical closed `SpatialAssignment` | canonical-spatial-assignment | spatial-plan-schema、attention spatial constraints、topology | exact-demand-boundary、canonical chain | source/topology改变后重建 |
| attention-ready `ExactDemandProof` | exact-demand-boundary + attention-demand-integration | closed assignment、current structured IR、coupled attention facts | canonical-root-work、spatial-domain、layout/movement/feasibility | normalized semantic root或spatial改变后重算 |
| canonical plan components | canonical-root-work至canonical-schedule | closed assignment、attention-ready demand、baseline semantics | attention-work-projection、attention-selected-decomposition、deterministic-baseline-closure | canonical coordinate改变后重建；不签发资源合法性 |
| `AttentionWorkDescription` | attention-work-projection | fixed semantic facts及canonical plan prefix | attention-selected-decomposition、domain extensions | observed plan choice改变后重算；不预测physical allocation |
| prepared selected attention decomposition | attention-selected-decomposition | complete canonical plan与work IDs | baseline/search complete-candidate materialization | 只属于当前candidate transaction；不签发SPM合法性 |
| full spatial successors | spatial-domain | fixed semantic facts、spatial schema、exact-demand query | search-control-foundation、root-work-domain | spatial改变使全部下游失效 |
| `RootRegionWork` alternatives | root-work-domain | spatial assignment、exact demand、canonical-root-work | region-execution-domain、selected leaf emitter | spatial/root改变后派生重算 |
| `RegionPlan` / execution/use bindings及selected construction | region-execution-domain | root work、exact relation、canonical-region-plan | temporal-domain、layout/movement selected builders、Core Region state、full-feasibility | region改变使全部下游失效；actual subtree只属于当前candidate transaction |
| `TemporalPlan` alternatives及selected loop construction | temporal-domain | execution scopes、canonical-temporal-plan、selected Region construction | partial-feasibility、layout-domain、Core Temporal state、full-feasibility | temporal改变使全部下游失效；nested/coupled scope按parent relation重建 |
| partial structural readiness | partial-feasibility | exact demand至temporal prefix | layout-domain、Core transition | missing coordinate显式；资源合法性保持unknown |
| `RepresentationPlan` alternatives | layout-domain | partial structural readiness、canonical-representation-plan | movement-domain、Core Representation state | representation改变使全部下游失效 |
| `MovementPlan` / communication actions | movement-domain | boundaries、physical versions、topology、canonical-movement-plan | storage-domain、event/full feasibility、Core Movement state | movement改变使storage/schedule/feasibility失效 |
| Serialized identity + initial storage alternatives | serialized-execution + storage-domain | temporal/movement occurrences、versions | event-resource-foundation、execution-structure-domain | structure choice使initial storage/event facts失效 |
| pre-K foundation `EventGraph`与rebuild builder | event-resource-foundation | fixed semantic facts、initial storage、Q63/effect、target resources | execution-structure-domain、structure-specific-storage | structure改变后丢弃pre-K实例；同一builder等待post-K BufferPlan重建 |
| `ExecutionStructurePlan` | execution-structure-domain | EventGraph、temporal/movement/storage eligibility | structure-specific-storage | 必须re-enter storage，禁止直达schedule |
| structure-specific `BufferPlan` | structure-specific-storage | fixed structure occurrences/live distance | post-K EventGraph rebuild、schedule-domain | structure sibling/stage/distance改变后重闭 |
| post-K `EventGraph` | event-resource-foundation同一builder，由structure-specific-storage re-entry调用 | fixed ExecutionStructurePlan、post-K BufferPlan、Q63/effect/target facts | schedule-domain | structure/storage generation改变后重建；不得复用pre-K graph |
| `ClosedSchedulePlan` | schedule-domain | fixed structure/storage、post-K event/resource facts | full-feasibility、Core Schedule state | structure/storage/worker/order改变后重闭 |
| actual candidate result | full-feasibility | complete physical assignment具体化后的Card/Tile/Instr与current relations | search-control-closure | Accepted保留offset/IR；typed rejection反馈controller；indeterminate停止 |
| all-axis `CompleteCandidateKey`与search controller | search-control-foundation + search-control-closure | 全parent state、真实domain APIs及actual candidate result | explicit public `search`、unified-search-closure | exact rejection默认只命中同一完整key；coverage typed；无默认cost cohort，不消费资源估算 |
| retained actual winner | unified-search-closure | accepted actual candidate | package/target downstream | 直接保留，不重新materialize或重新规划offset |

审查规则：一行没有producer、没有真实consumer、使用尚未产生的artifact，或绕过“失效/re-entry”列，顺序即不成立。test-only fixture
可以提供显式closed assignment，但不能充当production producer或让任务提前标完成。

### Checkpoint完成边界

Q50.S是共同graph normalization与selected-decomposition设计owner，不是search轴；其工作分别由attention-normalization、
attention-demand-integration、attention-work-projection、attention-selected-decomposition和attention-production-closure一次性work items
承担。其它Q50 owners同样通过owner map汇总，不再把一个owner写成跨阶段调度任务。每个domain work item只能用显式typed test assignment补齐
尚未施工的轴，并证明本轴的domain coverage、transition、
materializer、verifier、exact rejection/deferred和无局部winner。这些test assignment不是production default，新路径不得调用
旧owner暗中补全其它坐标。

work item只有在query/domain、production consumer、direct oracle及其输出合同同批闭合后才`done`。Design owner不是下游前置；
下游只依赖表中具体work item产物，因此不存在owner-done与production-consumer互相等待的循环。unified-search-closure不消费
deterministic baseline output，attention-production-closure也只汇总两条独立证据。

“某个局部更贵的choice在后续轴闭合后成为global winner”、fusion/buffering/pipeline协同、complete
CardExecutable cost与全轴actual winner都由unified-search-closure验收，不得用旧selector提前签发。表中“下一work item”固定当前唯一
施工顺序，不自动声明policy或artifact依赖；真实producer/consumer只看artifact表和`tasks/progress.md`直接输入。已选assignment变化后仍必须
失效并重新展开所有受影响轴。

## 现有代码分类与处置

现有文件不能因“结构乱”被批量删除。进入每个 checkpoint 前先把相关实现归到下列类别，并只处理当前轴。

| 类别 | 当前代表实现 | 处置 |
| --- | --- | --- |
| 稳定 downstream trunk | CardModule/TileModule IR、CardModule-to-Tile conversion、TileRegion-to-Instr、Tile memory planning、card resource/runtime-launch verification、retained target output、package/runtime | 保留；所有 policy 复用同一路径；现有`TileMemoryPlanning`/`CardExecutableLowering`仅作实现定位，后者仍需按实际职责收敛名称 |
| 可复用 core facts | `StructuredDAGAnalysis`、target topology、Q50.A immutable-borrow/exact-demand合同、Q50.0 move-only accepted result | 只消费能脱离旧candidate/search owner独立调用的current IR事实和accepted result；work reservation按invocation-local accounting重新实现。`TileExecutionCandidate`、schedule-state、metrics、stable ordinal、feedback history、evaluator和proposal order不进入Core |
| 可提取 mechanism 素材 | attention/decode graph proof与recurrence、`BidirectionalTiling`、`CompleteTraversal`、placement option、movement/collective lowering、selected-buffer materialization、ready-order/worker/completion verifier | Q50.S graph vertical与Q50.B–Q50.K先定义终态typed query/transition/apply合同，再迁入仍正确的局部算法、proof、verifier和negative case；不迁移旧API、调用顺序或winner行为 |
| 待退役active search monolith | 当前executable-synthesis中的`deriveShortlist`、coordinate sweep、candidate family、mixed evaluator/materializer、allocation/buffer feedback、beam、accepted cohort、schedule-plan selector和winner rematerialization | 不建立adapter；attention-demand-integration、spatial-domain与search-control-foundation就位后切public controller并删除旧控制路径；其中独有算法/proof/test仍按后续owner work item迁移，不能随controller一起误删 |
| 旧bounded/rank search donor | `RankCandidateSearch`、旧structured candidate generation/evaluation/selection及rank-era candidate set | 不保留旧接口或clone架构；先逐项迁移placement、cost、schedule、coverage和test witness，只有明确淘汰的fixed cap/rank-local winner可直接退出 |
| compatibility lowering | `StructuredDAGEdgeStrategyPlan`、dense rectangle fragment、现有 local/peer lowering | 在新 representation/movement IR 可完整消费 exact demand 前保留；Q50.G/H 逐项替换，不提前删除 |
| late selector/fixup | layout/movement optimization、ready-order、worker placement、buffer/allocation feedback 中会重新做选择的部分 | 先改成 verifier/materializer 或 typed transition mechanism，再按轴删除选择责任 |

独立pass、analysis、lowering、proof或negative witness若仍属于新合同，必须由新owner承接并形成actual-IR witness和fresh test；
旧search输出格式没有兼容义务，但控制链中的算法、proof和test witness必须完成能力对照；全部current axis与Core闭合后才删除旧控制。
仍依附于downstream/baseline的旧轴内selector或fixup，在对应Q50机制完成能力迁移时删除选择责任。

### 新链分轴施工

attention、spatial、region/fusion、temporal/feasibility、representation/movement、buffer和event/resource仍由各自Q owner设计，
但施工只通过上表唯一work item交付。每个domain work item必须同批建立Core state/transition、invalidation、production caller和direct
oracle，不能先堆成孤立mechanism再统一接线。每个旧selector/repair/test只有在替代work item取得current owner、Core consumer和
direct witness后才能删除。

## Q50.0：CardExecutable Compilation Boundary

先从当前综合大流程抽出唯一、无策略的CardExecutable编译/准入函数：输入已经选择且物化的CardModule，依次执行
Tile module splitting、TileRegion-to-Instr、fresh completion reconstruction、fixed-capacity SPM/DDR planning、
communication/resource/runtime-launch verification，输出accepted CardExecutable、proven exact rejection或indeterminate failure。
target ABI preparation/lowering/translation属于下游真正保留的target output，不在candidate/CardExecutable准入时提前执行。

返回taxonomy必须完整区分`accepted CardExecutable`、`proven exact rejection`与`indeterminate failure`；allocator
`ResourceExhausted`、timeout或内部错误属于最后一类，不能伪装成candidate非法。

这个边界不得枚举候选、修改选择、在lowering中retile/spill/rebuffer，也不得把失败降级成performance Unknown。Q49.P和Q51对
各自的每个complete candidate调用一次；同一candidate不得probe后重建。定向测试要证明同一CardModule得到相同accepted digest或
相同typed rejection，且没有第二条兼容编译路径。带完整current owner relation的actual SPM capacity rejection可以驱动外层
controller生成下一个candidate；其它失败不得伪装成该反馈。

实现结论：actual compile/verification seam只接收owned、complete CardModule、current relation和一个invocation-local
`CardExecutablePreparation`。Card split后可对current Tile dataflow执行selected movement/representation核对，TileRegion→Instr后可执行
selected storage、K phase及J order/worker/completion；canonical preparation为空操作。两个callback都不能保留IR pointer，失败由外层销毁
整份candidate。随后baseline与search共同调用同一Tile lowering、MiniMalloc、DDR、transport和target gate，返回Accepted、
ProvenExactRejection、Unsupported、Indeterminate或CompilerFailure。Tile memory planning保留SPM failure kind和actual demand owner
evidence；production caller只有在evidence完整时才能把capacity rejection作为当前candidate反馈，不能从Location、shape、bytes比例或
diagnostic字符串猜因果。

## Q49.P：Deterministic Baseline功能闭环、Policy与结构隔离

Q49.P不是对既有baseline做性能润色，也不是把`none`缩成fixed-assignment verifier；它要从current正常上游输入同时补齐
功能合法化、policy、结构、probe、causal diagnosis和materialization
边界。改写后必须从未选placement/temporal/layout/buffer的正常上游IR产生可执行结果，并用本软件产物重新证明result/digest与
no-card，不得把历史package当成新调用链的证明。

### Current实现收敛要求

current实现只保留一条完整候选路径：immutable source analysis→canonical assignment→caller-owned materialization transaction→
Q50.0 actual admission。局部root/Tile/function capacity probe、placement-option/CSP baseline、post-hoc root split repair、
Location/shape归因、plan-side footprint、shadow schedule、default IR trace和search statistics owner全部退出baseline调用闭包。
candidate transaction统一管理result/operand/output/movement/scratch relations；任一actual SPM demand缺owner即contract failure。
每个candidate CardModule/Q50.0各一次，rejected transaction销毁，Accepted owner直接下传。

```text
Pipeline position:
- Upstream IR / input:
  verified card-local TensorProgram、available Tile、immutable target facts及可从current SSA/structured semantics派生的policy-free
  relation机制；尚未创建search state/candidate，也没有预先计算的placement-specific demand或selected spatial、TileRegion
  grouping、temporal、layout、route、buffer choice。Q50.A exact demand由controller对每次closed spatial trial现场查询。
- Current stage responsibility:
  由独立deterministic controller从正常上游IR构造canonical spatial assignment、每root独立TileRegion、显式DDR boundary、
  representation/movement、single-buffer assignment、order/completion和temporal candidate；每个candidate只具体化一次并调用Q50.0。
  只有actual SPM capacity rejection且每个demand都有current owner relation时才生成下一temporal candidate；accepted executable直接
  保留并下传。该合法化不评分或比较性能，但必须覆盖声明支持的baseline域。
- Output IR / files:
  accepted baseline CardModule经Q50.0形成带actual offsets的CardExecutable；Q59 transaction随后提交verified ExecutablePackage。
  rejected candidate的IR与relations一同销毁，只保留typed actual witness；普通编译不生成printed-IR
  snapshot，显式inspection只在最终accepted output上按请求采集。
- Downstream consumer:
  `none`直接发布accepted executable/package；search不消费该结果，matched A/B由外层分别启动独立编译事务。
- User-level driver / named pipeline:
  wafer-compile source-to-package pipeline的typed `none`；typed `search`不调用该controller。
- Explicit non-goals:
  不增加第二套Card/Tile/Instr lowering或SPM planner；不在baseline比较placement质量或搜索group/fusion/multi-buffer/可选
  route；不把lower bound或局部分析冒充card-level transport/resource/ABI证明；不改变Q51的性能候选域。这里的非目标不排除
  baseline为走通程序而确定canonical placement、temporal tile、representation/movement、buffer、order和completion。
- Done criteria:
  baseline不依赖search state/candidate、完整placement-option domain/ranking evaluator、proposal order/group materializer，也不
  调用domain propagation、recursive CSP/backtracking或其它option-list assignment solver；canonical placement从typed
  structured/topology facts直接推导，始终只有一个live coordinate，不能先展开全部connected rectangle/axis options再取第一个；每个
  baseline TileRegion恰有一个structured compute root和由exact demand证明必要的non-root tensor transforms；跨root shaped dependency显式DDR；
  exact rejection只来自actual Instr allocation/lifetime和MiniMalloc conflict demand，并通过current relation映回root；每个candidate的
  CardModule materialization与CardExecutable compilation各一次，accepted owner不重建；同一coordinate的全部Tile root execution domain由一次
  grouped exact-demand query处理，relation按operation/result/operand只建立一次；structured producer形成停止边界，final region
  只一次性物化typed recipe证明需要的operation和carrier endpoint，16个独立Tile构造bounded并发并按Tile ID稳定归并；accepted后不构造`StaticSchedulePlan`、duration estimate或其它不被output消费的shadow result，
  production调用不打印/保存Tile IR trace；初始完整tile超SPM的正例由actual SPM rejection驱动下一candidate并最终通过package/no-card，
  reduction轴可沿typed tiling contract缩到最小合法vector；原「最小合法tile超限」fixture未进入完整合法coordinate，
  typed capacity/unsupported terminal保留为fail-closed防御出口；baseline使用独立窄work accounting，不持有或清零search统计来
  证明隔离；fresh
  source-to-package、oracle/no-card、digest、计数和结构正负测试通过。
```

`none`不得构造Q51 state、candidate set、candidate family或全图performance Cartesian组合，也不得通过调用search-oriented
domain/ranking evaluator后只取第一个结果来伪装canonical construction。它只复用typed iterator/topology事实、单coordinate
legality/materialization机制和Q50.A logical demand/coverage query，不复用“生成全部合法placement options”的domain API；对baseline
合同内的mandatory coordinates执行有限、完整、确定且不被beam/cap/time budget截断的actual candidate序列。每个coordinate只具体化
一次完整CardModule并由Q50.0消费；actual capacity rejection销毁该candidate并驱动下一个coordinate，accepted owner继续下传。
不得为同一coordinate另建probe、footprint、synthetic demand或第二份CardModule。

### deterministic-baseline-closure同步与copy实现调研

本项采用两条成熟IR实践，但不复制其它项目的dialect或runtime：

| 事实源 | 可复用原则 | Wafer采用边界 |
| --- | --- | --- |
| [MLIR Async dialect](https://mlir.llvm.org/docs/Dialects/AsyncDialect/) | async operation用SSA token显式表达完成；consumer依赖必须显式，不能从结构或共享状态隐式推断 | DTE send/recv只产生token，wait消费exact token；NCC仍使用自己的typed pending-worker域，二者不互相完成 |
| [IREE Stream timepoint propagation](https://iree.dev/reference/mlir-passes/Stream/#iree-stream-propagate-timepoints)与[resource lifetime](https://iree.dev/reference/mlir-dialects/Stream/) | timepoint可穿过call/control flow以避免过早host wait；resource在对应timepoint前保持live，释放/复用发生在completion之后 | wait放在receiver first read、sender/relay last release、实际FSM/resource reuse或terminal之前；不在issue、loop或region边界默认等待。Wafer仍由TX81的1 sender、4 receiver FSM及Q63 participant事实限制 |

实现采用一个actual-IR派生算法，不增加baseline-only completion协议：

1. attention selected decomposition只返回running maximum/sum/accumulator的普通SSA values；删除只为携带worker0 schedule而存在的
   `wafer.tensor.completion` op、lowering和测试。worker/participant不再出现在algorithm层。
2. external output copy的tile来自当前`SpatialOutputShard.temporalTileSizes`，即candidate已经选定并由output owner消费的temporal
   coordinate。materializer把`outputIndex → tile sizes`作为query-local typed input传给TileRegion construction；BodyEmitter只验证
   rank、正值和`tile <= copy extent`并按该tile构造exact main/tail SCF waves。缺失或冲突plan为contract failure；禁止fallback到全1
   tile、shape heuristic、target capacity或footprint estimate。每个wave仍进入actual MiniMalloc，capacity rejection只能回到外层controller。
3. required NCC completion从current Instr effects/ranges/control flow重建。same-worker issue只保持order；region/yield、WDMA、managed
   store/reload、alloc/free不构成completion。cross-worker alias、NCC→DTE/Kcore/call和observable function terminal仍在最晚必要位置完成
   actual pending participant。
4. non-streamed peer token保持pending：recv在对应tensor第一次SSA读取前wait；send在同Tile唯一sender resource复用或terminal前wait；
   receiver只在第五个并存token前完成最早pending token。streamed scratch的recv wait紧邻store是first read，send wait紧邻dealloc是
   last release，属于实际lifetime cut而不是默认issue-after-wait规则。所有剩余token在Tile entry terminal前exact闭合。

本项覆盖矩阵如下；static site、dynamic execution和direct downstream witness必须同时断言：

| 输入等价类 | 代表输入 | 结构路径 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- | --- |
| FA/FD state recurrence | rank-4/5 FP16或BF16，K2为1024及1025/1031，single/multi block | selected decomposition→Tile→Instr→completion rebuild | action/value mapping缺失、cross-domain contract缺失 | algorithm IR零`tensor.completion`和固定worker；无cross-domain cut时steady/nonterminal join为0，K2 block增长不增加join | actual cost、SPM plan和terminal join participant从final Instr重算 |
| external cache/output copy | rank-4 `[1,32,1024/1025,128]`及prefix tail | selected output temporal tile→exact SCF main/tail→RDMA/WDMA→MiniMalloc | missing/duplicate output tile、rank mismatch、nonpositive/oversized tile | scratch tile等于selected output temporal tile（tail按exact size）；DMA dynamic count等于selected tile count而非element count；每element exact覆盖一次；steady/nonterminal join为0 | split Tile module、Instr work count、actual SPM offsets和output relation完整 |
| NCC same/cross-worker | rank-3 1024/1025/1031，same-worker region/store/reload、cross-worker RAW/WAR/WAW | current effects/ranges→pending fixed point→join reconstruction | unknown effect、unsupported CFG、invalid worker | same-worker结构边界零join；cross-worker只在冲突前完成真实participant；return闭合剩余pending | target lowering只为保留join生成真实`ncc_join` call |
| Direct DTE token lifetime | rank-3 1024/1025 direct、streamed、two-hop relay及五个overlap receive | token issue→first read/last release/FSM reuse→wait→binding | missing token、第五个无法释放的receiver、cyclic wait graph | non-streamed至少一个合法issue window；wait exact一次；最多4个live recv与1个live sender；streamed adjacency有明确first-read/last-release witness | PeerLowering、DirectDTETransport binding及CardExecutable verifier通过 |
| full `none` vertical | aligned/ragged prefill、decode cache和ordinary multi-root | source→canonical candidate→Q50.0→target/package→no-card | typed unsupported/indeterminate保持原分类 | candidate/Q50.0各一次、无search、join/wait/DMA work满足上述门禁、16 Tiles all-and-only | fresh package、CPU oracle和no-card消费同一accepted owner |

### deterministic-baseline-closure实现结果

- `wafer.tensor.completion`及其ODS、printer/parser、lowering和测试已经删除；attention online state只通过普通SSA value传递。算法层不再
  写worker或participant。
- complete static `tensor.insert_slice` assembly由一个只读query检查exact disjoint/full coverage；candidate output materializer与
  BodyEmitter消费同一结果。Card mapping直接传入`SpatialOutputShard.temporalTileSizes`；single-root入口通过当前`linalg::LinalgOp`
  result indexing map把已选iterator tile精确投影到output tile。两条入口都不从source shape、SPM容量或估算值补tile。
- TileRegion→Instr不再把region terminator、WDMA/reload类别或same-worker loop backedge当作join理由。Lifetime analysis在resolved
  same-worker NCC链中用后继issue界定旧地址lifetime；任何实际物理复用会在后续同worker issue时由busytable按地址排序。该缩短仍保留
  worker-domain obligation，不同worker、Direct DTE、Kcore/call或其它observer必须先有覆盖participant join；仅丢弃旧pending状态属于
  contract bug。observable terminal继续完成真实pending participant。
- non-streamed Direct DTE保留SSA token窗口；recv在对应logical value第一次读取前wait，send在唯一sender复用或terminal前wait，
  第五个receiver issue前释放最早pending receiver。1个sender slot与4个receiver FSM由共享target hard fact定义；streamed scratch只在
  actual store/dealloc边界等待。

2026-08-23 fresh证据：增量构建通过；受影响unit为154/154，current tracked unit suite为904/904；Dialect/Pipelines/Transforms lit为
198/198；IR/source organization、自检和target/runtime/compiler/model public-link 4/4通过。FP16 attention prefill、two-step KV-cache
decode及LLaMA-2 7B block均从source重新生成package并通过CPU reference与no-card，分别为14.95秒、309.34秒和697.34秒。LLaMA本轮
运行中观测RSS约10.6 GiB，功能与dynamic join/copy work门禁已通过，但该wall/RSS样本不是性能完成证据；Q52必须在完整search链上重新profile并
解释或消除该热点，Q53不得复用本轮package。没有执行真实板端测试。

controller交给共同materializer的是窄immutable resolved baseline assignment。它是上述feasibility resolution的输出而非入口
前置条件，只含per-root placement、显式singleton region boundary、完整temporal vector和已经确定的canonical
representation/movement/buffer/order/completion事实，不含evaluation/score、stable ordinal、transition/failure history或
controller flags。materializer只apply这些已选事实，不得根据同Tile root集合自行group，也不得补search default。

### 功能合法化与SPM收缩

baseline以每个root的完整local iterator extent作为第一个temporal candidate；parallel与reduction iterator的后继只来自同一typed
tiling contract。每个candidate实际生成operand slice、stride/dilation halo、result/init/accumulator、temporary、materializing copy、
movement staging、layout、completion和lifetime，再运行唯一SPM planner；不得从上一个candidate沿用resource结论。

actual SPM overflow只允许controller沿每个conflict demand的current owner relation对应root生成下一合法temporal candidate；多轴
shape必须一直覆盖到最小合法粒度。第一个actual accepted candidate即停止，不为性能比较其它fit。若最小candidate仍被actual
planner证明不可行，返回typed capacity rejection；ResourceExhausted、Indeterminate、Unsupported和CompilerBug保持各自分类并停止，
不得互相冒充。对声明支持且存在baseline-domain completion的输入，`none`必须形成完整
CardExecutable/package，缺少performance search不能成为失败原因。

```text
TensorProgram
-> deterministic baseline candidate construction
-> candidate CardModule / TileRegion materialization
-> CardModule-to-Tile module splitting
-> TileRegion-to-Instr conversion
-> fresh completion and actual SPM / DDR / transport / ABI admission
-> typed feedback or Accepted CardExecutable
-> retained winner target ABI / LLVM output
-> ExecutablePackage writing once
```

`OptimizationConfig::none()` 使用独立 deterministic feasibility controller：最大合法非空 Tile participation；每个structured compute
root独立TileRegion；root间显式DDR boundary；零fusion；buffer count为1。participant count、Tile group、iterator/factor和独立
root顺序按typed structured/relation/topology facts的完整semantic key决定，不使用pointer、walk ordinal、`stableOrdinal`或
search proposal order。controller沿不截断的有限canonical successor逐个形成candidate；每个candidate完成grouped exact-demand和
carrier coverage后只构造一次actual CardModule交给Q50.0，actual SPM rejection才允许生成下一candidate。它不计算score、
不维护incumbent/candidate family，也不保留用于质量比较的备选方案。

### TileRegion-to-Instr有序reduction的结构化表示

Q49.P的actual gate不得因host端逐tuple展开而使Instr IR、编译时间和RSS随reduction extent产生无意义膨胀。TileRegion-to-Instr按
current `MemRefType`和physical encoding的exact piece bounds/period划分tuple domain；每个run用首点、相邻点和末点的exact
`IndexRelation`证明descriptor数量、byte count、inner bytes、stride、iteration、destination字段不变且source byte offset符合checked
affine序列，然后把原顺序编码成常量有界`scf.for`。循环携带两个实际accumulator buffer并逐次交换，body仍是同一gather/scatter后接
同一elementwise accumulation；不能改变tuple lexicographic order、reduce kind、init、dtype或target instruction。不能从encoding
合同证明该run时fail closed；多reduction轴的有界路径保持逐点静态表示并由独立oracle核对，不作为capacity或numeric选择。

动态byte offset作为`wafer.instr.gather_scatter`的SSA offset输入，static descriptor字段继续由同一个exact planner产生；op-local
verifier只检查表示自身，function/target stage使用常量循环范围和ValueBounds证明offset全域非负且每次descriptor均在actual source
buffer物理范围内。它不创建allocation、不参与SPM admission、不作为candidate选择或numeric policy；SPM lifetime、completion、
schedule cost和target lowering直接消费该`scf.for`。request-local descriptor cache只复用total、bounded、projected的相同typed query，
不保存Operation/Value、失败结果或跨IR epoch状态。

本项覆盖矩阵：rank-3/rank-4 FP16/BF16 reduction及attention actual路径使用主要维度`1024`和`1025/1031`；覆盖Cx/NCx block内、block边界、
tail、单轴可压缩和多轴/非仿射保留静态序列；正例断言loop-carried accumulator、exact descriptor序列、fresh completion、actual
MiniMalloc和target LLVM均可消费，负例断言越界dynamic offset、动态/不闭合循环范围和不等价descriptor不能通过stage verifier。
小shape只用于逐tuple静态序列与loop展开oracle对照，不能代替上述真实规模纵向。

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

Q49.P直接消费Q50.0从actual allocation、lifetime、completion和MiniMalloc产生的typed result。capacity witness的每个demand必须通过
current materialization relation关联到root result、operand、scratch、movement或output；不按type/shape、位置字符串、region内唯一root、
pointer identity或diagnostic字符串猜测。只有完整归因的actual capacity rejection允许controller前进；resource exhaustion或
indeterminate立即传播。

`none`与`search`共享immutable structured/relation/target facts、policy-free single-root TileRegion materializer和
actual Card/Tile/Instr、completion、SPM/DDR、verification、package机制；不共享search state/candidate/evaluator、group
boundary、proposal order、score、winner或结果owner。Q49.P只服务`none`；Q51从TensorProgram独立建立search candidate，不接收或
重建baseline。Q49.P施工时把resolved baseline assignment的single-root apply落在稳定
PhysicalDataflow/Conversion边界；Q50.B/C随后扩展完整search domain与single-root mechanism。Q50.F不在IR外预测资源合法性；
baseline与search的完整candidate都由同一actual Q50.0 gate决定SPM/DDR/transport结果。
共享的是已关闭coordinate的
mechanism，不是生成或选择coordinate的控制：任何option-domain construction、constraint propagation、recursive CSP、backtracking、
candidate/evaluator或winner协议都不得进入`none`的transitive call graph。baseline默认值不限制Q51域。

历史official HF prefill、functional decode、LLaMA block 的 source、oracle、package 和 no-card runner只证明旧入口mechanics；
它们不能单独把Q49.P标为`done`。Q49.P迭代期经Q59 compile transaction使用fresh有界小图、overfull-to-fit、五类relation及轻量
source-to-package/no-card输入定位correctness/work缺口；算法、结构和work evidence闭合后只执行一轮fresh FP16 LLaMA
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
合法粒度。reduction轴可沿其typed breakpoint lattice缩到最小合法vector；没有parallel result轴的
纯reduction必须先形成all-factor=1、单参与Tile的unpartitioned canonical coordinate，再验证reduction temporal fallback；这只是
该root没有parallel轴时的退化，不是baseline全局Tile数。不能把current placement domain表达不了当成source unsupported。
原「最小合法tile超限」反例既未进入该合法coordinate，也不能作为
capacity terminal的当前证明；typed capacity/unsupported terminal仍保留为fail-closed防御出口。测试同时断言每次trial重新计算
workset/lifetime、没有beam/cap/budget截断fallback，且这些trial不进入candidate统计。

### Q49.P actual-feedback与completion闭合结果

以下记录汇总仍在current production路径中的actual-feedback机制。`none`从normal TensorProgram建立一个deterministic coordinate，完整candidate
各自只物化一次CardModule并调用一次Q50.0；actual SPM capacity rejection只有在每个allocation/conflict demand都能沿current
result、operand、scratch、movement或output relation回到typed structured owner时才推进下一temporal coordinate。rejected owner立即
销毁，accepted CardExecutable直接保留并进入target/package；baseline调用闭包不构造search state、candidate family、score、winner或
plan-side capacity/footprint结果。SPM只使用MiniMalloc，没有allocator fallback。

TileRegion emission现在由caller-owned recorder收集完整buffer relation；selected DDR stage、output insert、Tile-to-Instr scratch和
support copy都保留current typed owner。跨Tile独立carrier使用常量有界`scf.for`表达实际prologue/steady/tail movement，不在host端生成
fragment Cartesian product。attention temporal operand按block绝对坐标从原operand取得resident slice，512→256等refinement因此实际缩小
SPM allocation，不重建完整operand window。Structured relation只为actual structured payload可达的support results建立事实。

Tile-to-Instr有序reduction按encoding提供的exact physical piece bounds和tile periods构造descriptor runs；每个run只验证首点、相邻点和
末点，使用SSA byte offset与loop-carried accumulator保持原tuple顺序。dynamic offset由function/target stage用constant bounds证明全域
在实际buffer范围内。request-local descriptor/traversal cache只有相同typed semantic key第三次出现后才保留plan，失败不缓存；1025级
work-count证明四次相同exact layout query只执行三次descriptor planning，唯一/低复用relation随pattern销毁。

本轮fresh覆盖如下：

| 覆盖 | 输入与结果 |
| --- | --- |
| rank-4 aligned | FP16，`batch=2, head=2, Q=1024, KV=1024, Dq=128, Dv=64`；actual SPM rejection→refinement→16-Tile accepted，fresh通过 |
| rank-4 ragged | FP16，`batch=1, head=2, Q=1025, KV=1031, Dq=64, Dv=128`并带mask；覆盖remainder/tail和actual MiniMalloc，fresh通过 |
| FA/FD纵向 | direct aligned/ragged prefill与functional decode均从selected decomposition走完整CardExecutable gate；decode cache output不形成whole-tensor SPM residency |
| ordinary relation矩阵 | broadcast、reduction、affine window、strided、multi-piece、multi-producer、scalar/zero-rank、wave-bounded carrier及finite temporal traversal均通过complete gate |
| lowering/allocator | dynamic gather offset正负例、fresh NCC completion、current owner relation、MiniMalloc/packing和target lowering通过；算法层Tensor completion op已删除 |
| production | fresh FP16 LLaMA 2 7B block以public `optimization-none`生成完整package并通过no-card；16 Tiles all-and-only，未进入search |

完整增量构建和全部configured lit通过；lit中本项相关case均实际执行。Tensor层scheduled attention recurrence仍由后续
`region-execution-domain`/`temporal-domain`统一设计，Q49.P不提前建立第二套region recurrence或dynamic-slice exact-demand合同。
2026-08-23专项修正已删除per-K2-block completion、逐元素copy join、non-streamed peer immediate await及结构化
region/materialization join；具体实现和fresh数字以前述“deterministic-baseline-closure实现结果”为准。Q49.P当前为`done`，后续D/E/H/I/J
仍须在各自完整domain中建立selected construction，不能把baseline canonical placement当作search完成证据。

### 已完成输入：Exact-Demand Boundary

`exact-demand-boundary`与`attention-demand-integration`已经退出current施工队列。current消费者只依赖其稳定输出：从immutable
TensorProgram、closed `SpatialAssignment`和typed structured/index relation派生all-and-only operand demand、final owner、
reduction/coupled contribution与merge requirements。该边界不签发SPM/DDR合法性，不读取candidate actual IR，不从名字、shape或
diagnostic恢复语义。历史施工、donor迁移和fresh数字由archive与Git保留；本轮若fresh验证暴露回归，回到原owner修复并重放直接后继。

### Current exact-demand 与 actual SPM feedback legalization

本节替换上方旧记录中的root closure、support rebuild和LLaMA门禁；前文Q49.P Pipeline Contract的其它边界继续有效。

#### 输入、结果与不变量

算法只读同一immutable TensorProgram、Q50.B `SpatialAssignment`、Q50.A `ExactDemandProof`、structured node/result identity和
已经关闭的temporal coordinate；execution domain、final ownership和operand demand不再由Q49.P复制或恢复。查询产生三类短生命周期typed
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

#### Actual SPM feedback legalization

exact demand和carrier coverage关闭的是语义输入，不签发SPM合法性。baseline对每个temporal coordinate只执行一次完整实际链：

```text
compileDeterministicBaseline(source):
  coordinate = initialCanonicalCoordinate(source)
  while true:
    candidate = materializeCardModule(source, coordinate)
    result = compileCardModuleToExecutable(candidate)
    switch result:
      Accepted(actualOffsets):
        return result
      ExactSPMCapacityRejection(actualDemands, currentRelations):
        roots = requireEveryDemandHasCurrentOwner(actualDemands,
                                                  currentRelations)
        coordinate = nextDeterministicTemporalCandidate(coordinate, roots)
      ResourceExhausted | Unsupported | Indeterminate | CompilerBug:
        return typed failure
```

没有actual CardModule→Tile→Instr allocation、completion和lifetime就没有SPM结论。每个失败candidate与其relations一起销毁；同一
coordinate不另建probe或第二份IR。next candidate只由实际rejection触发，仍须重新经过完整SPM planning，不能假设缩小后一定合法。

#### 实现迁移和完成门禁

1. 将Q50.A current query原位扩展为按consumer operand分组的exact-demand结果；现有per-edge consumer全部迁移，不能并存第二套
   baseline relation恢复逻辑。
2. borrowed source上的planning session完成全Tile exact-demand与carrier coverage；每个完整coordinate只具体化一次并运行actual
   Q50.0。只有完整归因的SPM capacity rejection生成下一coordinate。
3. common CardModule/TileRegion materializer直接构造final single-root region；baseline与search分别调用该policy-free apply，不能
   保留baseline-only修补分支。
4. 同批删除`buildBaselineRegionSource`、两套support rebuild、empty/destination补丁、post-hoc root closure/splitter及只覆盖这些
   路径的statistics/tests；旧SPM capacity probe API不恢复。
5. 每种admitted consumer input reconstruction用明确标注的有界static shape穷举需求子集，比较full evaluation与partial reconstruction；
   同一reconstruction另有真实规模整除/非整除正例。组合测试覆盖chain、diamond、fanout、multi-producer `insert_slice`、
   Peer/RegionCut混合、multi-result、empty branch、nonrectangular pieces、
   reduction temporal wave和unsupported atomic failure。
6. 显式测试计数证明relation construction按semantic support edge计数、每个非空`(value, Tile)`只处理一次、每个candidate的
   CardModule/Q50.0各一次、actual rejection数等于temporal transition数，accepted candidate不重建。这些计数只在显式测试sink建立，普通编译不创建
   统计对象、不打印日志。定向unit/lit与轻量source-to-package/no-card通过后，只运行一轮fresh FP16 LLaMA
   `optimization-none` source-to-package/no-card；不得进入search。

#### 2026-08-19 current实现能力与重新打开的门禁

current实现已经迁入按职责拆分的`Compiler/Baseline`、`Compiler/Planning`和TensorProgram→TileRegion conversion文件：

- 一个baseline invocation只建立一次只读source session；`StructuredNodeUseIndex`一次索引structured use，`StorageRootMemo`压缩
  storage-root查询，mapping session消费按consumer operand分组的exact demand；
- final materializer在structured producer处停止，只重建typed recipe要求的argument/constant/view/extract/insert与carrier；
  multi-producer fan-in、exact-empty branch和Peer/RegionCut混合不再通过递归support closure恢复；
- `buildBaselineRegionSource`、root/function局部probe、post-hoc support rebuild、同coordinate rematerialization与默认Tile IR/statistics
  已经退出baseline主路径；每个coordinate的一份CardModule由Q50.0直接消费，rejection成为controller的actual feedback；
- CardModule fan-out时，大型Tile body直接move进唯一Tile output；只有每个output确实需要的小型card-shared declaration按
  `IRMapping`复制。16个Tile entry与后续独立Tile stage使用bounded并发并按Tile ID稳定归并；
- active测试已覆盖single-root、multi-producer、reduction temporal accumulator、Peer/RegionCut混合、Card→Tile move-only owner、
  source-to-package与no-card。已经删除的旧TileRegion候选测试不复活；其中仍属current合同的proof已迁入CardModule/current gate。

首次fresh FP16 LLaMA `optimization-none`编译在约163秒内生成current package，CardExecutable阶段约99秒、peak RSS约4.48GiB；
第一次完整CTest随后因PyTorch runner仍读取已退役的`package/functions/forward.mlir`而失败。current package有意只保存
manifest/modules/program data，因此runner现改为从source program读取`functions/forward.meta`、从package读取manifest port，并已通过
定向runner unit、对该fresh package的payload准备和`wafer-run --no-card`。修复后official CTest从fresh source重新执行完整链，
于181.70秒通过1/1；它证明旧baseline功能链可用，但不能证明actual-feedback legalization，因此不再支持Q49.P的
`done`状态。

这次显式timing还记录到16个Tile合计执行约22,968次TileRegion→Instr conversion。结果已经有界且正确，但相似Tile body仍有大量
重复lowering；每个不同coordinate因actual SPM feedback必须独立具体化，不能跨candidate复用IR；accepted candidate内部16 Tile的等价pure query共享才归Q52。
Q51 search只能复用这些可重算facts，不能缓存actual IR、clone完整DAG或重放materialized result；Q49.P重新通过前不运行LLaMA search。

## Q51.Core：Search Control Kernel

### Current control boundary

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
| public policy routing | `OptimizationConfig::search`直接进入单体synthesis search branch | Q51.Core改接new-search session并删除旧branch；没有accepted search candidate时typed failure，不转入Q49.P |
| functional fallback | search内部重新derive baseline并把它编码成candidate seed | 删除该fallback；Q49.P只由`none`调用，search candidate必须由B–K完整physical assignment产生 |
| state identity | `TileExecutionCandidate`混合mapping、metrics、ordinal和feedback history | Core foundation从`SpatialState`开始；Q50.S fixed facts只在immutable problem中，各Q50 physical提交原位扩current variant/successor，derived facts留analysis，不预留nullable字段 |
| spatial/domain generation | placement domain、evaluator、coordinate descent和hand-written witness seed耦合 | Q50.B按all-iterator/topology/exact-demand合同重写惰性typed domain；Q50.A proof独立复用 |
| temporal/region/layout/movement/buffer choices | `deriveShortlist`内多轮coordinate sweep与固定action/buffer集合 | Q50.C–I分别拥有domain/query/transition/apply；不存在共同旧generator或默认补值 |
| schedule/resource ordering | candidate schedule在placement evaluation和cheap ranking中提前生成，accepted后又建selector plan | Q50.J/K把event/resource facts与choice纳入同一planning assignment；Q51比较plan cost/bound，不在lowering后另建selector |
| actual materialization | `materializeCandidate`混合CardModule构造、fusion/layout/buffer检查与complete compile | 每个complete assignment进入Q50.F actual admission一次；materializer只消费closed assignment，不在actual IR中补选项 |
| late failure handling | allocator/buffer failure生成siblings、lookahead、beam closure并重排shortlist | 只有typed actual rejection返回Core；allocator/lowering不得生成siblings或repair，Core按domain继续其它complete assignments |
| winner | accepted cohort丢弃actual modules，以schedule plan/stable ordinal选winner后再rematerialize | Q51 session保留move-only accepted incumbent；更差结果销毁，最终winner原样发布，不rematerialize |
| failure routing | materializer写字符串`failureGate`，controller按字符串决定prune/cache/feedback/order | 新Core只消费closed typed evaluation outcome和typed causal evidence；diagnostic label只打印，不参与控制流 |
| inspection | 每个complete compile在acceptance前无条件打印全部Tile IR并穿过synthesis result | candidate evaluation不携printed IR；显式inspection只读取最终accepted winner且不参与选择 |
| policy/cost aggregate | `WaferTargetPolicy`混合memory facts与Quick/Default/Deep、candidate/beam cap；`TargetScheduleCostPolicy`混合hard memory、profile reference和未校准prior | hard target facts回各target/memory owner；Core只接收session-level typed comparable cost，不消费aggregate policy；Q52只在fresh profile后建立独立estimate prior |
| telemetry/tests | proposal、feedback、ordinal、selected metrics及旧长链golden；源码marker CTest、paired optimization catalog、stderr parser和search-only fault injection保活旧实现 | invocation-local work accounting、per-axis oracle和new source-to-package tests；删除旧source-contract/catalog/inventory binding、旧数值/digest/耗时/执行路径断言，事务测试走最短current路径 |

```text
Pipeline position:
- Upstream IR / input:
  一个immutable TensorProgram borrow、transaction-owned ProgramData view及target/cost-cohort facts。Core入口没有baseline executable，
  Q50.S attention algorithm已经固定在normalized op上；入口没有预填的spatial、region、temporal、layout、movement、buffer或schedule assignment。
- Current stage responsibility:
  建立唯一replacement search session的control plane：管理typed partial assignments及其ephemeral transitions、对已接纳
  transitions不丢state的deterministic frontier、canonical dedup、global work/time budget、typed planning result routing、
  search-local accepted-result incumbent和coverage/lower-bound evidence。foundation只组合B/A真实transitions；后续Q50提交逐轴扩展current
  variants。Core只接受mechanism已经形成的typed transition/query result，不枚举或物化某个轴。
- Output IR / files:
  不产生新IR层或文件。foundation在缺axis时返回typed `IncompletePlanningDomain`及已验证frontier/work，不构造actual candidate；
  所有axis闭合后，Q51 closure把每个complete assignment交给Q50.F actual admission并最终输出move-only accepted winner、coverage和
  unresolved-work evidence。
- Downstream consumer:
  Q50.S提供immutable semantic facts和planning description，Q50.B–Q50.K逐项扩展current typed assignment和transition；Q50.F在
  complete assignment后接入actual materialization/Q50.0；Q51 closure接入全轴oracle、actual-result comparison和source-to-package chain。
- User-level driver / named pipeline:
  不新增pass、CLI、optimization policy或磁盘sidecar。spatial-domain与exact-demand-boundary完成后，
  search-control-foundation把explicit public `search`切到new owner并删除旧controller；
  incomplete结果明确失败且绝不fallback。Q51 closure只解除最后missing-axis gate并接common actual admission与唯一发布，不再次切controller。
- Explicit non-goals:
  不实现Q50.B–K任一choice domain，也不为Q50.S建立algorithm domain；Core只静态调用已经闭合的真实axis APIs。partial state不
  materialize；complete state必须走Q50.F actual gate。不预设best-first priority、
  default budget、no-good/dominance/DP/LNS算法；不创建generic provider registry、`any`/字符串tag/opaque payload或future-axis
  placeholder；不以mock domain声明production physical domain完整。
- Done criteria:
  foundation用真实B/A有界穷举states与独立state-graph model证明frontier/continuation，不使用mock-only provider；缺Region轴稳定返回
  `IncompletePlanningDomain`且CardModule/Q50.0为零。每个后续axis提交证明新增variant有producer/consumer且旧state不能越过missing axis；
  control closure再证明全轴reachable-state set、budget stop不会吞掉仍由exact
  frontier表示的state；rejected/indeterminate child都不影响合法siblings；baseline不进入search driver、candidate key或统计；
  stable winner与work accounting不依赖hash iteration、pointer、proposal ordinal或evaluation完成顺序。Core实现、链接和测试均不含
  `TileExecutionCandidate`、旧generator/evaluator/feedback/selector、字符串gate控制或旧长耗时integration；registered CTest/
  board inventory不再读取旧source marker、不要求`none/search` paired package或解析旧selection stderr；与搜索无关的transaction/
  failure-injection test不再被强制走`search`；Q50.S可以增加fixed semantic facts，Q50.B可以加入第一批真实typed axis，均不修改control semantics。
```

### Work Item `search-control-foundation`覆盖矩阵

本项只建立真实spatial prefix的control plane并切public routing；Region及后续坐标不存在时必须typed incomplete，不能调用旧complete
assignment、CardModule materializer或Q50.0。实现和收尾逐项检查：

| 等价类 | 输入 | exact断言 | 直接witness |
| --- | --- | --- | --- |
| 真实aligned/ragged spatial prefix | rank-3/4 FP16/BF16，1024与1025/1031，16 Tiles | production `SpatialPlanDomain` proposal与canonical cursor形成validated `SpatialState`；Q50.A satisfied；source byte-identical | first frontier state返回`RequiredPlanningCoordinate::Region`，candidate materialization/Q50.0/workspace offset计数均为0 |
| tiny完整continuation | 2--4 extent、2--4 Tiles，包含Balanced/Uniform、embedding和reduction merge | session逐次resume至exhausted得到的stable state key集合与独立`SpatialPlanReference`完全相等；proposal/canonical重复只入队一次 | continuation不保存IR/domain object/ordinal，frontier按`SpatialPlan`完整semantic key确定性出队 |
| graph与scalar | scalar、chain、independent branches、diamond/fanin/fanout、multi-result | zero-rank singleton和program Cartesian states不丢失；相同state由proposal/canonical两条路径到达时dedup | missing Region对每个pop出的state一致，不从baseline/default字段补后缀 |
| typed outcome | invalid state、unsupported demand、indeterminate demand、domain successor contract failure | unsupported resolved choice不删未访问siblings；indeterminate保留同一choice以便resume；broken contract停止session | failure分类来自typed domain/Q50.A outcome，不解析diagnostic string |
| public policy routing | 默认/显式`none`与显式`search`，timing on/off | 默认和显式none仍走Q49.P；search只进入new session并稳定失败为incomplete Region；两条policy互不调用 | search diagnostics含required coordinate与spatial work，且无baseline、CardModule/Q50.0、package或winner输出 |
| interface retirement | CLI/API/CMake/test/catalog/source检查 | `OptimizationConfig`只含closed Search/None；candidate-evaluation count option/API、旧CardExecutableSearch controller/test/source和paired search no-card catalog零残留 | source-organization、public link、driver option及lit正负例fresh通过 |

本项不运行search model/package/no-card成功路径，因为缺Region时成功发布本身就是合同错误；真实规模只验证analysis/state/routing，不构造
actual candidate。后续每个axis在同一state/continuation合同上原位扩variant，直到`full-feasibility`才允许首个actual admission。

### Work Item `search-control-foundation`闭合结果

- 新owner位于`Planning/PhysicalDataflow/Search`：`PhysicalDataflowPlanningProblem`只借用immutable `CardProgramAnalysis`并拥有current
  `SpatialPlanDomain`；`SpatialState`只保存validated `SpatialPlan`；`PhysicalDataflowPlanningSession`拥有proposal keys、canonical cursor、
  paused indeterminate choice和按完整state key排序的frontier。candidate state不含parent、pointer、proposal ordinal、demand proof或统计。
- continuation先发checked proposal，再沿Q50.B complete canonical successor推进；resolved proposal key只用于proposal/canonical dedup，
  不保存全部canonical domain。Q50.A satisfied才入队；Unsupported消费当前choice但保留下一sibling，Indeterminate保留同一choice，
  compiler contract failure停止。
- public policy switch已经原位改接new session。默认和显式`none`只调用Q49.P；显式`search`取得首个真实spatial state后返回typed
  `IncompletePlanningDomain(required=Region)`，diagnostic明确`candidate_actualizations=0`，不构造CardModule、不调用Q50.0、不写package，
  也不调用baseline。
- `OptimizationConfig`只保留closed Search/None identity；默认改为none。旧candidate evaluation count API/CLI/help、
  `CardExecutableSearch`/`SearchWork` source和5项old complete-candidate tests删除；test-only commit/transaction fault controls按真实outer owner
  可在none路径运行。PyTorch public runner仍接受`search|none`，但current source/no-card catalog只注册none，不再用paired search entry伪造资格。
- `UnifiedPhysicalDataflowDomain`及各D--I donor mechanism暂时保留为后续owner的能力来源，但已不在public search call tree；new Core headers/source
  不include Baseline或CodeGen，partial expansion不持有或clone IR。
- old `CardExecutableSearch`中仍正确的“一个完整assignment只经一次materialize→Q50.0”能力继续由
  `UnifiedPhysicalDataflowTest`和共享`compileCardModuleToExecutable` seam直接见证；public evaluation-count cap、TensorProgram alternative
  clone、六项raw metric局部winner和summary/trace controller均不满足终态合同，没有作为能力迁入，随旧controller删除。

fresh证据：`PlanningSessionTest` 6/6、`SearchRoutingTest` 2/2，覆盖rank-3 aligned FP16、rank-4 ragged BF16、scalar、chain、tiny
reference exhaustion、proposal/canonical convergence、stable frontier、跨parse pointer-independent key、Unsupported、Indeterminate和invalid state；
普通host unit 823/823；四项定向Tools lit 4/4；完整configured lit 262 passed、4 configured unsupported；source organization通过；configured
PyTorch board/catalog Python CTest 1/1。该结果只签发foundation和public incomplete routing，不签发Q51 complete search或package成功。

### 四类query-local事实

```text
Immutable session input:
  immutable IR borrow + target facts + cost cohort + currently implemented mechanism set

Candidate assignment:
  only named typed choices implemented and consumed by current Q50 mechanisms

Session control:
  deterministic non-dropping frontier + search-local accepted-result incumbent + work/budget accounting + coverage/bound evidence

Derived cache:
  session-owned source facts + exact demand + structural calendar + cost estimate/bound facts
```

- source-only facts由与MLIR analysis wrapper共用的policy-free builder建立；Core直接拥有typed result，不持有`Analysis *`，也不在
  pass外构造`AnalysisManager`。session lifetime绑定同一immutable source borrow；Core不得把token地址、`Operation *`、walk ordinal
  或printed IR当stable key。
- Core建立的current assignment aggregate起初不含未来轴。每个Q50 checkpoint同批加入本轴named typed field、transition、
  canonical encoding、query/apply和失效关系；Wafer-owned接口原位演进，不保留编号schema、旧wrapper或两套state。
- mechanism query只读source、target和parent assignment；transition apply先完整验证，再原子构造child。child不保存transition
  history、proposal priority、feedback root、failure history或derived metric。缺少尚未施工的轴表示partial assignment不可
  materialize，不允许从baseline default或旧selector暗中补齐。
- mechanism set、priority、search-local incumbent和work accounting是session facts，不属于任一candidate。future-boundary equivalence只有在本轴
  mechanism证明被投影choice不再影响任何future consumer时才可合并；Core不提供按aggregate bytes/makespan自动合并state的
  generic shortcut。mechanism顺序由唯一current driver静态组合，不提供runtime registry、dynamic plugin list或user option。

### Search-local actual result与winner

Q51 driver直接从Q50.S-normalized immutable TensorProgram建立search session，不先运行Q49.P。Q50.B–K加入的partial state在全部required
coordinates闭合前不能形成complete candidate，也不存在“用baseline或旧selector补齐未施工轴”的路径。每个complete assignment由new
typed mechanisms构造，随后立即进入Q50.F actual admission。第一份Accepted建立search-local move-only incumbent；后续Accepted只在
typed actual-cost comparison为Better，或Equivalent且完整semantic tie-break更小时替换。ExactRejection按typed owner evidence反馈；
Indeterminate/Unsupported/CompilerBug按各自合同处理。比较不依赖queue ordinal、hash insertion或并行完成顺序。预算结束仍没有
Accepted candidate时返回typed failure；baseline只由独立`none`调用，matched A/B由外层测试分别启动两个编译事务。

`ProgramDataHandoff`始终由外层compiler transaction唯一拥有，search只消费stable program identity/range，不为candidate复制owner或
materialize package。winner确定后，Q59把同一handoff随retained actual result继续交给target/package transaction。search计时从入口
开始并覆盖analysis、state expansion、全部complete-candidate actualization与winner selection，不混入另一次`none`事务的work。

### Transition、evaluation 与budget

- exact expansion对一个parent返回当前mechanism的全部immediate typed children，或返回仍留在frontier中的typed semantic
  continuation；opaque iterator/cursor不能成为遗漏siblings的旁路。先访问的child无论accepted、rejected或indeterminate，
  其它siblings都保持可达。
- `deferred(required coordinates)`列出typed prerequisite，只有相关assignment改变或闭合后才重新query；不得poll同一state。
  proven exact rejection只删除被其typed causal witness覆盖的state。Core本身不从diagnostic string合成no-good或扩大scope。
- resource/solver exhaustion形成unresolved work：不删除state、不更新winner，也不能签发optimality。Core自身的malformed transition、
  stale borrow或broken invariant是fatal compiler error，不得用baseline掩盖。
- Q51.Core foundation在缺axis期间只路由partial query outcome；Q51 closure接入Q50.F后，complete state调用actual admission。
  各轴query仍可提供可重算的structural rejection、deferred、cost estimate和已证明bound，但不能签发SPM legality。
- optional work counts记录generated、complete-actualized、actual-accepted、exact-rejected、indeterminate、solver/query work和winner
  updates；wall deadline只是安全中断，不能改变已完成result的stable排序。actual compilation必须计入统一work与profile。
- budget中止时Core只报告typed evidence，不自行声称`feasible-with-bound`：Q51 closure只有在exact frontier仍完整且有typed
  admissible bound时才映射为该等级；显式丢弃/未表示state后只能是`budgeted-feasible`。Q52再根据profile确定默认budget和
  有损policy。

### Oracle分层

1. **Core state-graph model**：test-only typed fixture直接给出有限reference graph和accepted cost，不调用production mechanism；比较
   reachable state key、frontier exhaustion、winner、work counts和budget coverage。覆盖empty domain、diamond convergence、同state
   不同transition order、exact rejection、deferred、indeterminate、fatal invariant和budget stop。
2. **Per-axis reference enumerator**：从Q50.B开始，每个physical mechanism用tiny真实semantics独立枚举本轴合法typed choice，并与
   production mechanism逐parent比较stable key；Q50.S另以graph matcher、FA/FD classifier和algorithm reference验证fixed facts，
   不生成axis leaves。这不是Core gate。
3. **Independent full flat exhaustive oracle**：Q51 closure在全部真实轴闭合后于tiny domain独立枚举complete assignments，并逐点调用
   同一actual materializer和Q50.0，比较production reachable domain、typed results、winner与actual digest。oracle枚举器不进入production，
   但complete-candidate actual gate与production完全相同；Core fixture不能替代这一层。

### Current code处置与完成条件

- 可直接复用Q50.A immutable-borrow合同及`StructuredDAGAnalysis`中仍为current IR可重算的事实；Q50.0只属于complete-candidate actual admission，Q49.P
  baseline不进入Core。existing work accounting思路可迁移，但固定cap和rank合同不得带入。
- 当前`TileExecutionCandidate`混合assignment、`TileExecutionMetrics`和`TileExecutionTransition`；后者还含stable ordinal、
  feedback root、allocator history和beam控制。Core不拆分、适配或构造该组合类型；foundation直接以B真实plan构造closed
  `SpatialState`，不存在RootAlgorithm variant或空aggregate。foundation不定义未来nullable fields；后续axis同批原位
  扩current sum type。`StructuredDAGScheduleState`中的ready/live算法只有在Q50.J按新event assignment合同
  重建后才能迁入，旧candidate schedule不进入identity或Core API。
- `CardExecutableSearchStatistics`中只服务旧proposal、feedback、shortlist、selected ordinal和winner rematerialization的字段
  随Core旧branch同批删除；Core使用invocation-local work accounting，derived winner metrics从plan query重算。不得保留同字段的新struct
  或为旧diagnostic提供compat adapter。
- 旧`RankCandidateSearch/Evaluation/Selection`、coordinated/bounded driver及tests先建立donor能力对照；仍需的placement、cost、schedule、
  no-good、coverage和test witness按对应owner迁移，只有接口/clone/旧output可直接退出。不得用“未注册”“旧symbol”推断算法无用。
- `OptimizationConfig::search`的public spelling保留，并在search-control-foundation切换到new owner；现有14处显式旧search长链CTest、
  PyTorch board默认值/`actual_fused_edges` stderr assertion及search-only test-control限制逐项改为current最小路径或基于actual
  IR/package的测试。旧winner、统计、耗时与日志格式无回归义务。
- search-control-foundation、逐domain和search-control-closure tests分别通过；不同insertion/hash/analysis completion顺序得到相同reachable prefix set和work；
  missing axis返回typed incomplete，完整域才产生actual candidate与winner；link closure不引用旧controller。parallel actual evaluation只有
  在owner隔离、共享目录和资源限制明确且结果按semantic key稳定合并时才允许。
- search-control-foundation在spatial-domain/demand work items后独立提交并切public routing；后续每个domain work item同批扩Core。per-axis reference
  enumerator、independent flat exhaustive oracle、exact planning-domain coverage、有效fusion、actual candidate gate、single-winner
  source-to-package chain和donor能力迁移仍是Q51 closure的完成条件；Q51 closure不再切public controller。

## Q50.S：Graph-Level Attention Algorithm Normalization

Q50.S不再是Q51的semantic-algorithm search axis。它实现05号稳定合同：在policy分叉前把已证明的完整Q/K/V
attention归一为一个`wafer.linalg_ext.attention` op，并确定`flash_attention`或`flash_decoding`。`none`与`search`
消费同一normalized TensorProgram；Q51从Q50.B spatial state开始，不保存或枚举Materialized/Online、FA/FD、block或partition
algorithm points。

### Pipeline contract

```text
Pipeline position:
- Upstream IR / input:
  GSPMD和official StableHLO-to-Linalg normalization后的immutable card-local TensorProgram；Q/K/V、scale、optional mask、
  softmax与value contraction仍由current structured SSA、indexing maps、DPS、types和effects表达，尚未绑定Tile。
- Current stage responsibility:
  一次性证明完整attention语义及functional KV-cache decode relation，创建一个self-contained
  wafer.linalg_ext.attention并设置closed algorithm attr；实现standard tiling/effect/shape contracts及planning只读coupled-state合同；
  selected decomposition再提供K2 partial mechanics；提供query-local AttentionWorkDescription和selected Linalg decomposition。
- Output IR / files:
  同一normalized TensorProgram，matched root由wafer.linalg_ext.attention表达；FA/FD是op上的current graph fact。
  不产生algorithm domain、candidate graph、CardModule、physical plan、IR sidecar或文件。
- Downstream consumer:
  Q50.B/A从op及coupled-state合同派生spatial assignment、exact operand demand和per-output merge requirements；
  C--K/F/J消费AttentionWorkDescription投影到既有root work、versions、movement、storage、events和resources；
  Q51唯一winner在Card subtree中展开selected Linalg/Tensor/SCF并确定性转换到wafer.tile。
- User-level driver / named pipeline:
  wafer-compile的none/search共同normalization；wafer-opt只复用同一leaf pipeline做parser/verifier/rewrite测试。
- Explicit non-goals:
  不比较materialized softmax与online recurrence，不选择Tile、K2 partition count、temporal block、layout、movement、slot、
  worker、schedule或winner；不创建第二attention op、wafer.tile/instr attention或runtime字段；不按name/shape/参数位置识别。
- Done criteria:
  op/matcher/classifier/interface/work-description/selected-decomposition及donor能力迁移闭合；planning无IR；B/A能够只读消费
  FA/FD合同；每个complete candidate只展开一次并在Q50.0前无attention和executable Linalg残留；final winner不重建，none/search各有
  source-to-package/no-card witness。
```

### S-1：一个op与一个graph owner

current IR只新增：

```text
wafer.linalg_ext.attention
  inputs: query, key, value, scale, optional mask
  destination/result: output
  inherent fields: algorithm, indexing_maps
```

`algorithm`是closed `flash_attention | flash_decoding`。op表示最终attention result；online
`(maximum, sum, accumulator)`只是interface-defined partial state，不作为第二个attention op或跨stage bag公开。
Q/K/V/mask/output iterator roles由05定义的map关系推导；ODS字段、parser/printer、verifier和generated accessors是唯一IR事实源。

op实现：

- `DestinationStyleOpInterface`；
- `TilingInterface`；
- `MemoryEffectOpInterface`；
- `ReifyRankedShapedTypeOpInterface`；
- 一个窄`WaferCoupledReductionOpInterface`。

custom coupled interface只返回K2 reduction iterators、Maximum/Sum/Accumulator component maps/types、initialization、coupled merge和final
owner；它不include Planning类型，也不返回Tile、layout、buffer或event。pinned PartialReduction interface的generic driver要求
partial init与source DPS results同构，因此一个final-result attention op不实现该接口；K2 partial由winner内selected Linalg/SCF builder
物化。当前configured pinned MLIR没有`IndexingMapOpInterface`的header/TableGen定义；Linalg consumer使用pinned
`mlir::linalg::LinalgOp`，attention consumer使用同一op的typed ODS map/static-range accessors，不新增Wafer indexing interface。

### S-2：Graph proof与algorithm分类

normalizer在official Linalg conversion后建立一次function-local SSA/indexing proof session，从observable value contraction反向证明QK、scale、
optional score mask、softmax和PV。它的production输入必须来自Q60产品入口的fresh post-Linalg IR。常见PyTorch/HF差异只按05定义的
有限结构族处理：named/generic contraction、transparent transpose/reshape/cast、scale位置、additive或compare/select mask、
softmax SSA等价形式及exact KV prefix append；不为模型、rank或参数位置注册pattern。precomputed scores、普通softmax、缺Q/K/V
relation或effectful conflict保持原Linalg DAG。

```text
normalizeAttention(function):
  facts = buildStructuredValueFacts(function)
  matches = prove all QK -> scale/mask -> softmax -> PV roots
  validate replacement/effect ownership conflicts before mutation
  for match in stable semantic order:
    algorithm = classifyAttentionAlgorithm(match, facts)
    create one wafer.linalg_ext.attention
    replace only the proven final result
  erase only dead memory-effect-free matched operations
  verify function
```

classification只使用current tensor state/dataflow：

```text
if K and V are exact prefix-appends,
   updated K/V are the attention operands,
   both updated states are returned by the function,
   and K2 admits at least two nonempty pieces:
  flash_decoding
else:
  flash_attention
```

decode state仍是ordinary function input/output；op和runtime不拥有KV-cache policy。无法证明decode得到FA，不按Q长度、模型名或argument
位置猜测。FD后续physical closure失败是typed compile failure，不静默改回FA。

normalization先收集全部proof再用同一个`IRRewriter` create/replace/erase；不clone Module/Func/DAG。near-miss无mutation；true conflict或
postverify failure终止。创建op的pass声明全部dependent dialects。

输入覆盖采用两层证据：从fresh PyTorch/HF export最小化得到的typed IR覆盖用于matcher unit；Q60产品入口每轮重新导出的portable
StableHLO与post-Linalg IR用于integration。前者便于定位canonical rule，后者防止exporter/GSPMD/conversion变化使fixture脱离真实输入。
只有手写理想matmul/softmax fixture不能关闭S-2。

### S-3：FA与FD算法mechanics

以下`scores(S)`已经包含current scale和optional mask。共同partial state为：

```text
m(S) = max(scores(S))
p(S) = exp(scores(S) - m(S))
l(S) = sum(p(S))
a(S) = sum(p(S) * V(S))
state(S) = (m(S), l(S), a(S))

combine(left, right):
  m = max(m_left, m_right)
  left_scale  = exp(m_left - m)
  right_scale = exp(m_right - m)
  l = left_scale * l_left + right_scale * l_right
  a = left_scale * a_left + right_scale * a_right
  return (m, l, a)

finalize(state): output = a / l
```

FA要求B不空间切K2；E在一个owner内对K2 blocks构造multi-result loop-carried state。每个block只产生当前QK score、mask/scale、
max/exp/sum、PV和state update；完整score/probability tensor不得进入actual plan/resource description。

FD要求B把K2空间切成至少两个contributions。每个contribution内部运行同一个FA recurrence；A按output piece证明all-and-only K2
fiber和一个coupled merge requirement，H/J把全部state components搬到selected merge owner并在ready后执行同一combine，最后只finalize
一次。partition count、merge Tile、transfer/combine tree、temporal block和worker仍分别归B/H/E/J。

K1是QK contraction reduction，K2是online normalization/value reduction；二者可以各自temporal tiling，但只有K2参与FA/FD coupled
state和FD spatial contribution。implementation不得用一个generic reduction flag混淆两类iterator。

### S-4：Pure planning description与A--K接入

#### attention-demand-integration contract

```text
Pipeline position:
- Upstream IR / input:
  verifier-valid且algorithm已固定的wafer.linalg_ext.attention、closed SpatialAssignment、function-local structured relation facts，
  以及同一op的WaferCoupledReductionOpInterface描述；输入尚未选择temporal block、layout、movement、storage或schedule。
- Current stage responsibility:
  按attention indexing maps为每个destination shard派生Q/K/V/optional mask的exact demand；FA保持普通final owner且不产生K2
  spatial merge；FD按每个output-domain piece证明all-and-only K2 contributions，产生一个包含Maximum/Sum/Accumulator的coupled
  ReductionMergeRequirement，并只把selected merge Tile暴露为final output owner。
- Output IR / files:
  attention-ready ExactDemandProof中的DependencyDemand、FinalResultOwner和ReductionMergeRequirement；不修改IR、不写attr或文件，
  不创建attention decomposition、CardModule或physical movement action。
- Downstream consumer:
  canonical-root-work、spatial-domain、representation/movement/feasibility owner通过同一proof读取operand slice、coupled state、
  contribution和final owner；后续attention-work-projection再把这些事实投影为action/value/resource对象。
- User-level driver / named pipeline:
  无独立pass、pipeline或CLI；none与search planning session静态调用同一exact-demand query。
- Explicit non-goals:
  不选择K2 partition、merge Tile、transfer/combine tree、temporal block或winner；不逐component发布独立merge，不物化online
  recurrence，也不增加attention-specific demand side table。
- Done criteria:
  FA/FD、mask/no-mask、single/multi-K2、整除/非整除和multi-axis/all-16-Tile矩阵均返回exact proof；FD每组K2 fiber无hole、
  overlap或重复，三个component的map/type/domain与source interface一致，partial不是final owner；mode-violating或缺merge的
  assignment得到typed compiler-contract failure，StructuredDemandView可直接消费proof且query前后source IR不变。
```

本work item的覆盖矩阵如下。表中shape只用于确保production路径真正经过多Tile、remainder和coupled merge，不进入IR legality、
algorithm分类或workload协议；小shape仅允许给单一故障负例，且不能替代对应真实规模正例。

| 覆盖类 | 代表输入 | 必须断言的exact结果 | 直接下游witness |
| --- | --- | --- | --- |
| FA整除、无mask | rank-5，M/K2为1024，B/M/N沿16 Tile多轴划分，K2保持一个interval | Q/K/V per-destination demand，16个普通final owners，零coupled merge，source IR byte-identical | `StructuredDemandView`接受且报告该root无spatial reduction |
| FA非整除、broadcast mask | M=1025、K2=1031，parallel轴含remainder，K2仍不空间切 | Q与mask projection/broadcast exact，owner boxes覆盖output且无hole/overlap，零coupled merge | 同一view与后续root lookup得到稳定owner顺序 |
| FD整除、single K2 | rank-5，M/K2为1024，B/M与K2共同使用全部16 Tile | 每个output group恰收齐K2 contributions；Q slice跨contribution相同，K/V/mask按K2切分；Maximum/Sum/Accumulator同组，只有merge Tile是final owner | view接受proof并报告spatial reduction；proof可被后续C/G/H/F按group读取 |
| FD非整除、single K2 | M=1025、K2=1031，K2产生不均匀pieces | contribution intervals连续覆盖0..1031且大小含tail；component domain、type、init/merge/finalization和final owner逐字段精确 | 同一assignment改变shard枚举顺序后proof保持semantic order |
| FD非整除、multi-K2 | rank-6，K2为33x31且至少一个其它主要维度>=1024，两个K2轴共同分片 | contribution数等于两个K2 interval count之积；二维fiber all-and-only覆盖，row与accumulator component domains分别正确 | downstream view按一个coupled group消费，不拆成三个独立reduction |
| typed failure | 在上述真实规模assignment上删除merge group，或让FA切K2/FD不切K2 | `InvalidSpatialAssignment`准确归因到reduction group/mode；不返回unsupported、普通candidate rejection或部分proof | session仍可查询未损坏assignment，source IR不变 |

#### attention-spatial-integration contract

```text
Pipeline position:
- Upstream IR / input:
  verifier-valid `wafer.linalg_ext.attention`、其generated algorithm/indexing-map accessors、
  `AttentionIterationRoles`，以及spatial-plan-schema定义的iterator extents与`IteratorPartition`。
- Current stage responsibility:
  从同一个attention op派生query-local `AttentionSpatialConstraints`：K1 iterator IDs、K2 iterator IDs，以及K2必须保持
  一个logical interval（FA）或必须形成多个logical intervals（FD）。用一个typed predicate同时约束后续canonical constructor与
  full spatial domain；K1和其它iterator不增加attention-specific限制。
- Output IR / files:
  不修改IR、不写attr或文件；只返回可重算的typed constraint value与typed violation。
- Downstream consumer:
  `canonical-spatial-assignment`构造满足约束的deterministic plan；`spatial-domain`只枚举满足同一约束的全部plans。
- User-level driver / named pipeline:
  无独立pass、pipeline或CLI；由baseline/search planning problem builder静态调用同一query。
- Explicit non-goals:
  不选择具体partition scheme、Tile subset/embedding、merge placement、temporal block或winner；不推导operand demand或
  reduction groups；不把algorithm/K1/K2复制进`SpatialPlan`。
- Done criteria:
  rank-independent role与constraint query独立受测；对K2所有axes的interval-count product，FA只接受1，FD只接受大于1；
  malformed iterator domain与algorithm constraint violation可区分；BalancedParts/UniformExtent、multi-K2、tail及physical
  embedding变化不产生第二套mode判断。
```

```text
AttentionSpatialConstraints
  queryKeyReductionIterators: unsigned[]
  keyValueReductionIterators: unsigned[]
  keyValuePartition: SingleInterval | MultipleIntervals

checkAttentionSpatialConstraints(constraints, iteratorExtents, partitions)
  -> None
   | IteratorDomainMismatch
   | FlashAttentionPartitionsKeyValue
   | FlashDecodingLeavesKeyValueUnpartitioned
```

`keyValuePartition`只比较K2 axes形成的logical interval-count product；它不等于总participant count。FA仍可沿B/M/N或在通用
partial mechanics闭合后沿K1做spatial work；FD至少一个K2 axis形成多个intervals。available Tile不足、embedding或merge不合法由B
schema/canonical/domain的既有validator处理，不在本constraint中伪装成algorithm不支持。

planning不通过decompose/lower actual IR取得cost或resource。由attention owner提供：

```text
AttentionWorkDescription
  root/output-piece/contribution/merge IDs
  action IDs: QK, ScaleMask, RowMax, Exponential, RowSum,
              PV, StateUpdate, StateMerge, Finalize
  value IDs: exact operand slices, score/probability scratch,
             block/running/partial states and final output
  indexing/occurrence relations
  logical work and selected-decomposition relations
```

该对象是从current op及B/E prefix重算的query-local result，不进入Q51 state、IR attr或文件。stable IDs由semantic root、output piece、
contribution和closed action kind形成，不含pointer、walk/order、Tile ordinal或printed string。

接入现有owner，而不是新增attention-specific plan layers：

| Owner | 施工内容 |
| --- | --- |
| B | mode约束K2 spatial domain；其它iterator、Tile subset/embedding和per-output merge placement沿B current domain |
| A | Q/K/V/mask exact demand、final owner、FD coupled contributions/init/final rule；使用同一ReductionAlgebra结果 |
| C/D | root/contribution/merge work与producer/consumer use binding；内部actions保持一个semantic root |
| E | K1/K2/parallel scopes、tail、multi-result running state和nested classes |
| G | internal logical values进入既有RegionValueVersionId/PhysicalVersionId及operation tuple constraints |
| H | operand fanout/stage、FD state payload和typed CombineRelation；不隐藏local arithmetic |
| I | state/scratch/staging storage、lifetime、slot family及ReuseAfterCompletion requirements |
| J/K | compute/movement/combine events、completion、serialized/pipelined structure及K后I/J重闭 |
| F | complete candidate materialization后从actual state/scratch/message/event/field和current relations运行SPM/DDR/transport admission |

`none`与`search`读取同一mode约束。B canonical producer对FA保持K2单一logical interval；对FD构造其canonical合法K2 partition和
stable embedding/merge owner。search枚举全部合法K2 partitions/embeddings/per-output merge placements；每个complete point由actual gate
判定。baseline不能把FD改回FA，
search也不能把FA当成FD空间切分。

Q50.B/A完成attention read-only接入后，Q51.Core的首个真实state是`SpatialState`，不是`RootAlgorithmState`。Q51 problem借用固定
normalized semantic roots；`PhysicalDataflowPlan`不复制algorithm assignment，FA/FD从source op读取并进入observed dependency key。

#### attention-work-projection contract

`attention-work-projection`只把fixed attention semantics投影到已经闭合的canonical C--K对象，不创建新的physical选择层。每个
root execution形成一个scope；FD每个coupled merge另有一个merge scope。scope内stable actions为QK contraction、scale/mask、row
maximum、exponential、row sum、value contraction、state update，以及按mode存在的state merge/finalize。action IDs由scope和closed kind
组成，不含pointer、ordinal或打印字符串。

typed values区分per-block score/scaled-score/probability、block Maximum/Sum/Accumulator、running Maximum/Sum/Accumulator和final output。
score/probability只是当前execution block的work/value description，不绑定G physical version/storage，也不表示完整attention score tensor；
FD contribution/merge running components与final output必须逐项绑定A/G/I已经存在的`PhysicalVersionId`、`StorageObjectId`。FA running state
保持execution-local scratch，final output绑定现有version。实际scratch allocation、alias、lifetime和coexistence只由complete candidate
materialization后的current IR与buffer relations解释；work description不携带容量倍数、simultaneity upper bound或opaque bytes。

```text
Pipeline position:
- Upstream IR / input:
  verifier-valid fixed-mode wafer.linalg_ext.attention、canonical RootRegionWork，以及G representation、H movement、I storage和J schedule
  coordinates；全部root/contribution/merge/execution/value/action已有typed identity，尚未selected-decompose或构造CardModule。
- Current stage responsibility:
  从op maps/roles与每个execution exact domain派生attention action/value/resource DAG；把operand fragments、FD components、remote gather
  staging、final output和parent schedule node逐项投影到既有C--K IDs并验证双射。
- Output IR / files:
  query-local CanonicalAttentionWorkDescription集合；不修改IR、不写attr或文件，不进入candidate state。
- Downstream consumer:
  attention-selected-decomposition为当前complete candidate读取同一description并准备Linalg/Tensor/SCF builder；后续domain work items
  重算对应projection，actual admission从物化后的allocation/use/effect和current buffer relations运行SPM planning。
- User-level driver / named pipeline:
  无独立pass、pipeline或CLI；none/search canonical prefix静态调用同一query。
- Explicit non-goals:
  不重新识别graph或选择FA/FD，不选择K2 partition/merge/movement/storage/schedule，不物化score/probability tensor、online loop、
  CardModule或wafer.tile，不增加attention-specific search axis、side table或numeric policy。
- Done criteria:
  FA/FD、mask/no-mask、1024/1025/1031、single/multi-K2、local/remote merge、multi-root及输入顺序扰动均得到stable all-and-only
  action/value/resource/projection；component maps/types/domains与source/A/G一致，每个remote component匹配H/I gather staging，所有parent
  execution/action存在于J；missing/duplicate/mismatched C--K fact为typed failure，query不修改source且不签发资源合法性。
```

| 覆盖类 | 代表输入 | 必须断言的AttentionWorkDescription | selected-decomposition / actual witness |
| --- | --- | --- | --- |
| FA aligned/no-mask | rank-5、M/K2=1024、all-16 Tile | 每root execution有QK→scale→max/exp/sum/PV→state update→finalize；Q/K/V fragments精确，零merge/gather projection | block/running state和final output mapping完整；actual scratch allocation全部有owner relation，score/probability无长期physical binding |
| FA ragged/mask | M=1025、K2=1031、broadcast mask | mask fragments纳入对应scope，score/row/output domains保留tail；仍无FD merge action | selected builder可按maps构block scratch，不从shape/name恢复mask |
| FD aligned/ragged | K2=1024及1031、local+remote contributions | contribution scopes无finalize；每group一个merge scope含state merge→finalize，Maximum/Sum/Accumulator逐component绑定version/storage | remote component各匹配一个gather+staging，local component无伪movement；actual planner从真实state/scratch lifetime判定SPM |
| FD multi-K2 | rank-6、K2=33x31且其它主维>=1024 | 每二维K2 contribution scope独立，group/merge和action/value IDs稳定；score work domain含全部K2 axes | selected decomposition消费同一Cartesian pieces，不压成一维split count |
| multi-root/determinism | 两个独立attention roots，反转rootWorks/resources/actions/bindings输入 | root/scope/action/value/projection逐semantic ID相同，无pointer/walk顺序 | 下游可分别准备两个roots，不共享scratch/state |
| typed failure | 删除/复制component version/storage/gather staging/schedule node，或篡改component type/domain | `BrokenAttentionWorkProjection`准确分类且无partial description | 修正输入可重新query，source byte-identical、CardModule/Q50.0计数为零 |

实现闭合：`AttentionWorkDescription`按root/contribution/merge scope提供stable action/value IDs、exact work/value domains、operand maps、
mandatory block/running state groups及C--K projections。FA block/running scratch无physical binding，FD cross-execution running components和
final output逐项绑定version/storage，remote components逐项绑定gather staging和schedule node；没有IR mutation、完整score tensor或
attention search axis。fresh定向5/5、完整`WaferUnitTests` 796/796、default configured lit 224/224、compiler public link、完整configured
build及IR/source organization通过。

### S-5：Complete candidate内selected Linalg decomposition

selected decomposition是Q51 Card subtree transaction中的一次actual construction，不是Q50.S planning transformation：

```text
prepareAttentionWork(op, complete B--K plan):
  validate all action/value/version/storage/event/resource IDs
  return PreparedAttentionWork                    // no IR mutation

emitAttentionLinalg(prepared, new Card subtree):
  create selected tensor slices and compact SCF loops/tails
  create selected linalg matmul/generic/reduce for all actions
  return action/value -> actual SSA mapping

convertSelectedAttentionCompute(mapping, prepared physical bindings):
  deterministically create existing wafer.tile.gemm/reduce/elementwise
  bind G physical versions, I storage and J event sites
```

H/I/J/K movement、allocation、peer、wait/join/release由各自prepared builder直接创建wafer.tile/memref/SCF对象；Linalg decomposition不
选择这些事实。Linalg中间态只存在于winner的新subtree并被同一commit消费，不形成用户stop stage、candidate cache或second pipeline。

进入Q50.0前card-scoped verifier要求：attention op与executable Linalg均为零，all-and-only action/value/materialization IDs匹配；
Q50.0随后从actual allocations和relations建立唯一资源结论。失败擦除整个candidate Card subtree。

禁止新增`wafer.tile.attention`、`wafer.instr.attention`、attention TargetCall、package algorithm字段或runtime cache branch。

#### attention-selected-decomposition contract

本work item由pure prepare和caller-owned selected-subtree emitter组成。prepare只验证`AttentionWorkDescription`内部所有value/action/
scratch引用完整一致；不签发资源合法性、不改IR。emitter由caller用semantic
root选择新subtree中的一个`wafer.linalg_ext.attention`，只消费description已有scope、exact operand pieces、state group和action顺序。

每个root scope生成一个block-sized QK contraction、scale/mask、row max、shift/exp、row sum、PV和state update。FA scope随后finalize并把
output slice插回DPS destination；FD contribution只发布Maximum/Sum/Accumulator SSA，merge scope按S-3 combine逐component合并后只finalize
一次。multi-K2保持Cartesian operand/score shapes。score/probability最大shape等于description的scope block，禁止创建完整global score。

emitter返回`AttentionActionId -> actual Operation[]`和`AttentionValueId -> actual Value`的current-subtree映射；pointer只在该mutation epoch
内使用。所有创建/replace通过`RewriterBase`；prepare之后的emit failure要求C outer guard丢弃新subtree，不在原source rollback或换plan。
这些actual ops仅为existing Linalg/Tensor/可选SCF，现有structured-to-tile converter是直接consumer；本work item建立typed mapping seam，
`deterministic-baseline-closure`负责把它接入new Card subtree并给出actual wafer.tile/Q50.0 witness，不能在本项尚无Card owner时虚报纵向。

```text
Pipeline position:
- Upstream IR / input:
  selected canonical AttentionWorkDescription及caller-owned新subtree中的matching normalized attention op；B--K choices已
  固定且只存在一个winner transaction。
- Current stage responsibility:
  pure prepare验证action/value/scratch内部coverage；candidate emitter按exact scope构造online Linalg/Tensor/SCF、替换attention op并返回
  typed actual mapping。
- Output IR / files:
  caller-owned selected subtree中的standard Linalg/Tensor/SCF及SelectedAttentionRootMaterialization；不修改source、不写attr/sidecar/file。
- Downstream consumer:
  deterministic-baseline-closure和full-feasibility把当前complete candidate mapping交给existing structured-to-wafer.tile construction，再由Q50.0从
  actual buffers、completion和lifetime运行SPM planning。
- User-level driver / named pipeline:
  无独立pass/CLI；none或search的complete-candidate transaction静态调用。
- Explicit non-goals:
  不重新匹配graph、选择FA/FD/block/partition/merge，不跨candidate保留或clone whole source Module，不做movement/storage/schedule或
  fallback，不新增attention tile/instr/target/runtime op。
- Done criteria:
  FA/FD、mask/no-mask、1024/1025/1031、multi-K2、local/remote merge和multi-root prepare均all-and-only映射；selected clone verifier通过且
  attention为零、无global score；work-description mismatch在mutation前失败，emit failure只污染caller-owned disposable subtree；tile mapping
  seam有直接consumer，actual Card/wafer.tile witness明确留给下一work item。
```

| 覆盖类 | 代表输入 | 必须断言的selected IR/mapping | 下一stage witness |
| --- | --- | --- | --- |
| FA aligned/no-mask | rank-5、M/K2=1024、all-16 scopes | 每scope八类action和十类value all-and-only映射；score小于global score，final slices覆盖output，attention op为零 | mapped Generic/Fill/Tensor ops进入structured-to-tile converter |
| FA ragged/mask | 1025/1031、broadcast mask | operand exact pieces、tail score/output shapes及mask map正确；source byte-identical | tail不靠shape/name恢复，Card builder按mapping绑定physical values |
| FD aligned/ragged | K2 1024/1031、local+remote contributions | contribution无finalize、每merge一个StateMerge+Finalize，三component SSA与description一一对应 | H/I/J按已有IDs接movement/staging/order，不由emitter重选 |
| FD multi-K2 | rank-6、33x31 K2且query=1025 | score/value contraction保留二维K2 Cartesian shape，merge输入数等于全部contribution states | tile conversion不接收一维split-count旁路 |
| prepare/determinism | complete assignment/work description、两个roots/输入顺序扰动 | action set必须exact相等，prepared roots按semantic ID；无pointer identity | none/search的每个candidate独立prepare，无共享actual owner |
| failure atomicity | missing work action、mode/op mismatch、malformed piece或emit中途failure | prepare mutation前typed failure；emit failure后source不变且candidate subtree由caller整体丢弃 | 不fallback另一个algorithm/plan，不调用Q50.0 |

实现闭合：pure prepare验证complete work-description action coverage；selected emitter按description exact pieces构造block-sized QK、scale/mask、row
max/exp/sum、PV、state update/merge/finalize，并返回all-and-only action/value actual映射。FA/FD的1024/1025/1031、rank-6 multi-K2和
两个roots均在caller-owned clone中替换为verifier-legal Linalg/Tensor IR且attention为零；中途failure只污染disposable clone。旧Search
`AttentionTensorOps`已迁为policy-free `PhysicalDataflow/AttentionLinalgOps`，donor与new builder共用一份实现。fresh定向7/7、完整
`WaferUnitTests` 811/811、default configured lit 224/224、compiler public link、完整configured build及IR/source organization通过。

actual Card/wafer.tile/Q50.0 witness尚未由本项虚报：后续`deterministic-baseline-closure`和`full-feasibility`必须把mapping交给
existing structured-to-tile consumer，并从actual allocation/use/effect和current relation运行资源gate；该纵向失败不能回到本emitter
换algorithm或修改当前candidate。

已完成的`canonical-plan-coverage-closure`不再作为current checkpoint重复展开；Q49.P与后续domain只消费已闭合的
canonical B--K/attention typed artifacts。若本轮actual candidate验证暴露这些输入的缺陷，按原owner修复，不能用SPM估算、
test例外或重新引入已完成work item绕过。

### S-6：Current/donor能力迁移

| Current / donor能力 | 终态owner | 必须迁移的witness | 退役条件 |
| --- | --- | --- | --- |
| `AttentionAnalysis` softmax/value-contraction proof | 05 graph normalizer + attention verifier | named/generic QK/PV、mask/scale、views、multiple roots、near-miss/effect conflict | 不再返回长期graph shadow或single-observable-root gate |
| `DecodeAttentionAnalysis` cache append proof | 05 algorithm classifier + Q53 explicit-state witness | prefix append、updated K/V consumption、returned state、argument/name perturbation | 不再控制search domain或runtime policy |
| `TensorProgramAlternativeDomain` Original/Online/Split Cartesian | Q50.S normalized op/classifier；block/partition归E/B | no whole-program clone、no algorithm search state、both policies same input | old kind/parameter successor/materialization API零残留 |
| `AttentionAlternative` online recurrence | coupled interface + selected Linalg decomposition | block/tail、state update/finalize、arbitrary output piece | 不创建whole-function alternative或hidden full scores |
| old FlashAttention donor | FA selected decomposition + C/E/D producer coupling | QK/PV tile、mask、tail、producer exact slice | current caller与actual Tile witness闭合 |
| old FlashDecoding donor | FD B/A/H/J integration | K2 partition/tail、all partial states、merge/output、two-step decode | 无decode-only physical search gate或split algorithm op |
| old Search `AttentionTensorOps::cloneLinalg` | policy-free `AttentionLinalgOps` + selected Linalg builder / local mapping | exact regions/maps/SSA in one winner | old Search helper path已删除；Module/Func/DAG alternative clone和replay cache仍由production closure退役 |

source是否active或已删除不证明能力迁移。每行必须同时有current production caller、direct test和downstream actual witness，之后才能删除
旧source/test。Q50.S completion不依赖保留旧symbol或compat wrapper。

### S-7：施工checkpoints与Gate

Q50.S通过六个独立work items交付，不作为一个跨阶段task调度：

1. **attention-normalization**：ODS op/interface、attention proof、FA/FD classifier、DCE和共同normalization；
2. **attention-spatial-integration**：K1/K2 role及FA/FD canonical/full spatial constraints；
3. **attention-demand-integration**：Q/K/V/mask demand及coupled contribution/merge；
4. **attention-work-projection**：stable action/value IDs、C–K/F typed projections及resource descriptor；
5. **attention-selected-decomposition**：prepare、Linalg/Tensor/SCF emitter、structured-to-tile conversion及rollback；
6. **attention-production-closure**：确认前五项逐项迁移donor后删除旧alternatives/clone/gates，并完成prefill/decode的两policy证据。

当前attention planning在post-K EventGraph前没有selected execution的typed target completion contract，因此还不能形成完整ScheduledState。
第6项必须从selected decomposition的实际Linalg action/effect与Q63接口构造该contract，再进入通用J/F；缺失时保持typed failure，禁止默认
worker、Synchronous completion、per-K2 join或结构性drain。该接线完成前，Q50.F不能用普通candidate的通过结果代签attention actual witness。

本work item施工前冻结覆盖矩阵。所有positive使用FP16/BF16、rank>=3、sequence 1024与1025/1031配对；shape仅用于覆盖均匀/tail，
不参与FA/FD识别、algorithm选择或search legality。

| 输入等价类 | 代表输入 | production路径 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- | --- |
| FA prefill | rank-4 B/H/M/K，1024 no-mask与1025/1031 additive mask | 同一attention op分别进入none/search selected decomposition→Card→Q50.0→package | matcher/prepare/materialize任一失败不fallback | attention op在actual Tile前归零、candidate/Q50.0各一次、两policy package均16 Tile | fresh no-card读取current manifest/target modules |
| FD decode | rank-4/5，K2 1024及1025/1031，local+remote contributions | contribution/state merge/finalize IDs经B--J进入两policyactual | component/gather/staging/schedule缺失typed失败 | all contribution和唯一merge/finalize actual mapping，search不调用none | package/no-card及returned state inventory完整 |
| multi-K2/tail | rank-6 33×31且query=1025 | 二维Cartesian K2和tail沿same decomposition/temporal carrier | 一维split-count旁路禁止 | 无global score，tail coverage无gap/overlap，source不变 | Q52可在同semantic op上优化physical plans |
| policy isolation | 同一source各开独立transaction运行none/search，输入顺序扰动 | 共同normalization/materializer/Q50.0，controller独立 | search失败不返回none package | diagnostics/call tree无互调；各自commit一次；search coverage typed | matched tests只在外层比较package facts |
| package/no-card | prefill/decode各policy fresh output | source→portable frontend→package→`wafer-run --no-card` | no-card unsupported只允许配置性feature，不跳case | manifest、16 modules、dtype/source/program descriptors一致，no-card实际执行 | Q53直接复用case/oracle/runner进入board-ready |
| donor retirement | AttentionAnalysis/Alternative/TensorProgramAlternative/old clone callers | current owner+caller+test逐项对照后删除 | 仍有独有proof/test则停止删除 | old algorithm frontier、whole-module alternative clone、decode-only gate零production caller | source organization和call-tree gate |

完成验证至少包括：

- custom/generic op roundtrip、map/shape/effect/verifier正负例及dependent dialect；
- matcher覆盖fresh PyTorch-derived named/generic QK/PV、scale位置、additive/select mask、view/cast/shared inputs/multiple roots/
  extra use/precomputed score near-miss；手写fixture只作补充；
- classifier只由functional state relation决定，symbol/argument/model名字扰动不改变结果；
- 有界独立reference覆盖FA temporal blocks和FD partitions/merge trees；同一算法另以rank至少为3、sequence至少1024的
  整除/非整除真实规模shape对覆盖均匀块、tail、多个block、partition和output pieces；
- coupled query与selected partial decomposition的component maps/types/init/final owner一致；
- planning source byte-identical、Module/Func/CardModule/Q50.0计数为零；
- B/A/C/E/F direct queries证明无逐component独立merge、无hidden state/scratch和init/final重复；
- winner selected Linalg只构造一次，随后只剩existing wafer.tile compute/movement；failure injection保持source/parent一致；
- none/search各plan=1、commit=1、Q50.0=1，无algorithm frontier、baseline/search互调或fallback；
- FP16/BF16 official prefill与functional two-step decode从Q60产品入口形成package/fresh no-card；Q53之前不声称board correctness或性能。

Q50.S只有owner map列出的六个work items、donor矩阵和production gate全部闭合才可完成。op注册、matcher unit、partial interface或单个Linalg dump
都不能代签完整任务。

## Q50.B：Spatial Partition + Physical Placement

```text
Pipeline position:
- Upstream IR / input:
  spatial-plan-schema只定义typed schema、structural close和validator；canonical-spatial-assignment读取attention-normalization后的immutable
  TensorProgram/current StructuredDAG、attention-spatial-integration给出的iterator/mode constraints及topology。spatial-domain再调用attention-ready
  exact-demand query，不接收per-root algorithm assignment。三个work items都尚未选择TileRegion、temporal、layout或movement。
- Current stage responsibility:
  spatial-plan-schema定义`SpatialPlan`、`SpatialAssignment`及structural close/validation；canonical-spatial-assignment只实现
  canonical constructor，不枚举domain；spatial-domain为每个structured node惰性产生覆盖全部iterator的typed partition scheme、canonical logical coordinate mesh、到
  distinct available Tiles的embedding，以及每个spatial-reduction output group的merge placement；把compact plan按需关闭为
  all-and-only exact execution shards并调用Q50.A派生demand/final availability。FA attention的K2 logical interval-count product必须
  为1；FD必须大于1；K1和其它iterator仍按各自standard/coupled contracts处理。
- Output IR / files:
  不产生新IR或文件。Q51 state只保存typed `SpatialPlan`；query返回ephemeral `SpatialAssignment`与Q50.A
  `ExactDemandProof`/typed failure。assignment携带exact execution domains和per-output merge groups，不携带derived ownership、
  movement、cost或统计。
- Downstream consumer:
  Q50.C按选定per-Tile shard物化single-root TileRegion；Q50.D–K继续补region/fusion、temporal、representation、movement、buffer和
  schedule。只有完整assignment才进入Q50.0。
- User-level driver / named pipeline:
  canonical-spatial-assignment由exact-demand-boundary和后续baseline work item消费；spatial-domain只由public `search` session静态组合。不新增pass、
  CLI、provider或独立placement selector；search不调用baseline constructor。
- Explicit non-goals:
  不物化TileRegion/CardModule，不选择temporal tile、layout、route、buffer或winner；不把connected/rectangle/all-16、result axis、
  participant count或常见factor当legality；不从cross-op demand制造placement no-good，不把Q50.A unsupported/resource/compiler error
  改写成另一个spatial point。
- Done criteria:
  spatial-plan-schema的typed roundtrip、close/validation独立受测且不选择canonical plan或携带FA/FD field；attention-spatial-integration后
  canonical-spatial-assignment对全部roots产生deterministic assignment，exact-demand-boundary不再include旧trial；
  attention-demand-integration后spatial-domain才进入production。随后
  single node、chain、independent branch、diamond、multi-axis remainder、scalar、multi-parallel/multi-reduction merge和partial
  redistribution的production domain与独立reference集合一致；任意verifier-legal participant subset/embedding及per-group merge
  placement可达或由完整topology automorphism proof canonicalize；partial-reduction interface不能表达时对应reduction
  partition不进入domain；完整assignment经Q50.A typed proof，旧single-axis/node-wide-merge/trial字段零残留，query无clone与默认统计。
```

### Work Item `spatial-domain`覆盖矩阵

本项只签发完整`SpatialPlan` successors、typed close/evaluation、独立reference set和proposal order；不提前实现Core、C--K或actual
candidate。实现前后的逐行门禁为：

| 等价类 | 真实规模输入 | exact断言 | 直接下游witness |
| --- | --- | --- | --- |
| 单root aligned/ragged | rank-3/4 FP16/BF16，parallel extent含1024及1025/1031 | BalancedParts与UniformExtent产生的全部非空interval序列按extensional equality只保留一个；每个shard all-and-only覆盖、无重叠、tail精确 | close后的`SpatialAssignment`由Q50.A得到typed demand/final owners，Q50.C canonical root work逐Tile读取 |
| multi-axis与physical embedding | 两个以上parallel轴、16个available Tiles及non-contiguous Tile subset | cell product不超过16的全部scheme组合；每个cell到任意distinct Tile的injective embedding可达，扰动Tile输入顺序不改stable set | `RootRegionWork`的logical shard ID不随Tile置换改变，physical Tile绑定按embedding改变 |
| scalar/zero-rank | rank-zero tensor与无iteratorstructured root | empty Cartesian product恰一个cell；每个available singleton Tile raw state可达 | exact demand与root work均保留empty coordinate，不伪造spatial axis |
| ordinary reduction | rank-3 1024/1025，parallel+一个或多个reduction轴 | 只有完整`PartialReductionOpInterface`允许reduction scheme；每个parallel coordinate形成独立`ReductionGroupId`，每组全部merge Tile choices可达 | Q50.A contributions/init/final owner与group、merge Tile逐项一致 |
| FA/FD | rank-5/6，1024与1025/1031，mask/no-mask、single/multi-K2 | FA所有K2 interval-count product恒1；FD恒大于1且二维fiber全覆盖；mode predicate与canonical owner共用 | attention-ready exact-demand proof保持Q/K/V/mask和coupled component group完整 |
| graph结构 | 1024/1025 chain、independent branches、diamond/fanin/fanout、multi-result | program successor等于per-node raw domain Cartesian product；global topology equivalence只能整plan作用，不能per-node删状态 | search-control-foundation可直接以一个closed `SpatialPlan`建立首个真实state |
| tiny独立oracle | extent和Tile count均2--4，覆盖两scheme、embedding、merge | test-only nested-loop reference不include/call production successor；逐plan stable key集合完全相等 | proposal首点属于exact set，关闭/反转proposal family不改变reference集合 |
| typed failure | unsupported reduction mechanics、FA切K2、FD不切K2、duplicate/unavailable embedding、missing/duplicate merge、demand unsupported/resource/compiler failure | domain structural unsupported与plan compiler error区分；Q50.A typed outcome原样传播且不删除raw siblings、不修改source | 同session随后关闭合法sibling成功；无IR clone、无capacity/route/layout side effect |

本项不把1024级shape带入tiny穷举；真实规模矩阵证明production query/close/downstream，tiny矩阵只证明有限域完整性。任一success case、
proposal首点或旧factor count都不能代签上述集合相等与per-group merge覆盖。

### Work Item `spatial-domain`闭合结果

本项已经把完整raw spatial机制切到`Planning/PhysicalDataflow/SpatialDomain`，并把active search consumer原位改为只保存
`SpatialPlan`。当前闭合边界如下：

- `SpatialRootDomainFacts`是canonical constructor与完整domain共用的唯一typed source-semantics query；Linalg、attention、
  all-result-mapped partitionable parallel iterator、per-result coordinate map、partial/coupled reduction和FA/FD约束不再分别推导。
- per-node successor严格按partition scheme、injective embedding、per-group merge Tile推进，program successor是逐root lazy
  Cartesian product；不预建point vector，不调用IR builder，不用exact demand、topology compactness或下游failure删除raw sibling。
- `BalancedParts`和`UniformExtent`按exact ordered intervals做extensional dedup；zero-iterator定义一个cell；普通partial reduction按
  result-mapped parallel coordinate和result group建组，attention同时覆盖rank-5单K2与rank-6多K2。
- current proposal family只包含经`contains`复核的maximum-balanced compact point、首个uniform-tail边界点、independent-component
  disjoint point和raw first point。它不声称局部最优，也不形成shortlist；同component overlap、非compact subset、任意Tile permutation
  与任意merge Tile仍由raw successor可达。
- `UnifiedPhysicalDataflowDomain`已经直接消费`SpatialPlanDomain`，旧`SpatialPlacementAssignment`、
  `CardSpatialPlacementDomain`、source、CMake registration、test和test fixture reader全部删除；baseline只复用source facts和structural
  close，不调用full domain或proposal。

当前target topology尚未暴露link capacity、endpoint color或fixed program endpoint等完整colored-graph facts，因此本项不建立
不完整的automorphism quotient：identity等价隐含成立，raw embedding一个不删。communication factor、verified nontrivial
automorphism、QAP/recursive-bisection/DP、admissible transport bound和local improvement由`search-scalability`在Core及H/J坐标闭合后
实现和测量；它们继续消费同一raw domain，不能反写本项legality。这是owner顺序，不是可选的第二条实现路径。

fresh验证结果：`SpatialDomainTest` 14/14，覆盖FP16/BF16的1024/1025、KV=1031、scalar、ordinary/multi-result/multi-axis reduction、tiny
scheme/embedding Cartesian reference、tiny/真实diamond、independent components、FA/FD及masked multi-K2；受影响consumer与完整普通
host unit为816/816；隔离的rank-4 aligned/ragged actual-memory case分别为1/1（749.995s）与1/1（232.009s）；configured lit共266项，
262 passed、4项按配置unsupported。`UnifiedPhysicalDataflowTest`继续证明current plan可被完整candidate materialization与Q50.0消费；
Q50.C actual TileRegion的全域materialization仍由后续`root-work-domain`拥有，不在本项提前施工。

### B-1 专项调研与spatial表示边界

本子任务核对了current `SpatialPlacement`、旧`StructuredDAGPlacementEnumeration`、Q50.A/S/D接口，以及OpenXLA Shardy和
XLA tile assignment。成熟表示给出的共同结构是：先用逻辑mesh轴说明**怎么切**，再用device-id order说明每个逻辑坐标**放到哪里**；
非整除dimension允许uneven shards，partial/unreduced状态显式表示，不能从device count或result shape猜测。

这些原则在本项目中的具体落点为：

- `IteratorPartition`只定义一个structured iterator如何形成有序intervals，不含Tile；`embedding`只把完整logical cell coordinate
  映到available Tile，不反向决定partition。
- `LogicalShardId`由`(SemanticRootKey, canonical logical cell coordinate)`派生，和physical Tile无关；交换两个Tile只改变embedding，
  不改变logical demand identity。
- spatial reduction的partial/unreduced事实由iterator role与semantic op interface产生，merge placement按output-domain group显式选择；
  它不是普通replication，也不能压成node-wide merge Tile。
- Shardy属于global tensor/SPMD层；这里已经是card-local structured iterator→Tile planning。直接把`sdy.sharding`复制进本层会混淆
  tensor dimension与operation iterator、还会把query-local candidate写回IR，因此只借鉴表示分层，不引入第二套sharding dialect。

current与donor的偏差审计如下：

| 实现事实 | 保留能力 | 必须替换 |
| --- | --- | --- |
| current per-node factor vector覆盖全部iterators，ordered distinct Tile序列覆盖任意injective embedding | all-iterator coordinate、任意available subset/order及lazy successor | factor只表示balanced-part count；assignment、logical scheme和physical embedding混在一个type |
| current factor product限制为available Tile数且支持non-divisible quotient/remainder | bounded participant count和balanced tail | 没有uniform-extent+tail scheme，也没有extensionally equal scheme canonicalization |
| current只有一个`reductionMergeTile` | 显式merge placement概念 | node-wide字段不能表示M/N/K mixed下多个output groups或coupled multi-result |
| current Card domain直接做所有node domain的Cartesian product | exact per-node domain可作为reference building block | 没有graph factorization；下游未关闭时也无法安全按local estimate删state |
| current `evaluate`构造`LogicalShardTrial`并用cross-edge demand过滤 | actual assignment必须能交给A closure | A已经不再把redistribution当placement rejection；trial/epoch/status链全部删除 |
| old donor按connected/rectangle topology groups、single result axis生成options | topology locality可作proposal，旧tests保留不同group/remap/branch覆盖 | connected rectangle、single axis、最大参与数和first proposal不能成为legal domain |
| old donor维护partial-state compute/transition/topology metrics及nondominated live-boundary | graph-level proposal素材 | 未关闭G–J时这些estimate不能证明future dominance，不能删除assignment |

#### 精确的current mechanism范围

本阶段承诺的是**regular Cartesian block partition**，不是所有可能的point partition。对extent `L`，每个iterator的完整scheme family为：

```text
BalancedParts(p), 1 <= p <= L
  // p个连续非空interval，长度为floor(L/p)或ceil(L/p)

UniformExtent(s), 1 <= s <= L
  // 连续长度s的full intervals，加至多一个nonempty tail
```

只有cell count不超过available Tile数的组合进入domain。两种scheme若产生完全相同的有序interval sequence，只保留
`BalancedParts`优先、参数更小、enum更小的canonical representation；这是extensional equality，不是heuristic pruning。
`BalancedParts(1)`是有iterator时的唯一unpartitioned key；zero-iterator/scalar node的空Cartesian product定义一个singleton cell，
仍可映到任一available Tile。future cyclic/block-cyclic、一个Tile拥有多个noncontiguous cells或任意point sets必须新增
独立closed scheme、A relation constructor和C emitter，不能暗改这两个enum的语义。

设node有`d`个iterators、scheme domains `S_i`。一个node的exact domain是：

```text
for partitions in CartesianProduct(S_0 ... S_d-1):
  cells = CartesianProduct(partitions.intervals)
  if cells.size > availableTiles.size: continue
  for embedding in all injective maps(cells -> availableTiles):
    groups = deriveReductionGroupIds(node, cells)
    for mergePlacement in all maps(groups -> availableTiles):
      yield NodeSpatialPlan(partitions, embedding, mergePlacement)
```

`deriveReductionGroupIds`只读selected semantic algorithm、iterator roles、result maps和coupled-result interface；group identity为
`(SemanticRootKey, coupled result group, parallel cell coordinate)`，不含Tile、pointer或printed set。没有spatial reduction时groups为空。
merge Tile可为contributor、下游co-located Tile或其它available Tile；route/cost归H/Q51，不在B-1缩小集合。

完整program domain是所有node exact domains的product，但`SpatialPlan`只保存compact schemes、embedding和merge placements；
`closeSpatialPlan`才构造ephemeral exact intervals/cells、验证all-and-only覆盖和injectivity，并交给A派生final owners/demand。这样candidate
identity不复制Presburger sets，A也不从participant count反推逻辑分区。

closure结果区分：valid plan得到`SpatialAssignment + ExactDemandProof`；source op缺必要partition/tile interface是unsupported；scheme
展开超过统一work limit是indeterminate；重复node/iterator、非法embedding、遗漏merge group或A contract failure是compiler error。
cross-op movement、route、SPM和schedule不属于这些结果。domain/query不clone、不写IR、不创建默认统计，也不引入其它设计线。

本子任务的表示验证为：对小extent直接枚举全部Balanced/Uniform interval sequences、injective embeddings及per-group merge maps，与
typed plan reference逐字段比较；覆盖non-divisible tail、scalar/zero-rank、multi-axis、multi-result、M/N/K mixed、任意Tile subset/order
和invalid duplicate embedding。另用metamorphic test交换两个物理Tile，证明logical shard IDs与A exact sets不变而embedding改变。

### Typed spatial plan与closed assignment

Q51 candidate state保存的唯一spatial choice是compact `SpatialPlan`：

```text
SpatialPlan
  nodes: NodeSpatialPlan[]                 // SemanticRootKey稳定顺序

NodeSpatialPlan
  root: SemanticRootKey
  axes: IteratorPartition[]                // 每个iterator恰一项
  embedding: TileId[]                      // canonical mesh coordinate -> Tile
  reductionMerges: MergePlacement[]

IteratorPartition
  iterator: unsigned
  scheme: BalancedParts | UniformExtent
  parameter: positive integer

MergePlacement
  group: ReductionGroupId
  tile: TileId
```

`BalancedParts(p)`把extent按quotient/remainder分成`p`个连续非空interval；`UniformExtent(s)`以固定`s`形成完整tiles加一个tail。
两种scheme生成extensionally相同intervals时只保留一个canonical key。多iterator partitions的Cartesian cells按iterator维度顺序
lexicographic编号；`embedding[cell]`是该cell的physical Tile，current regular mapping要求injective，因此factor product等于
participant count。physical Tile序列不是“选择前几个Tile”：任意available subset及任意logical-coordinate embedding都属于
domain，compact rectangle/natural mesh/full occupancy只是proposal顺序。

该current mechanism明确支持regular Cartesian block partitions；它不把“所有数学上可能的任意point partition”伪装成已经
实现。若后续要支持cyclic/block-cyclic或一个Tile拥有非矩形cell union，必须新增closed `IteratorPartition` scheme、Q50.A exact
set constructor与Q50.C materializer，并加入tiny reference domain；不能用同一个factor字段暗中改变语义。physical participant
subset可以是non-rectangular、disconnected或asymmetric，这与logical block是否矩形是两件事。

空间reduction的merge不是node-wide optional Tile。对给定axes和semantic algorithm，Q50.B从op result maps/
coupled-reduction interface派生stable `ReductionGroupId`：它由SemanticRootKey、coupled result group和parallel logical coordinate组成，不含
pointer、printed set或Tile id。每个group的merge Tile可从所有available Tiles选择，包括contributor、下游co-located或独立Tile；
route/cost稍后决定。M/N按`P_M/P_N`切、K按`P_K`切时有`P_M*P_N`个merge groups，每组有`P_K`个contributions。

Q50.B按需关闭plan，而不把derived Presburger sets保存在Q51 state：

```text
closeSpatialPlan(structuredIR, plan):
  for nodePlan in plan.nodes:
    verify every iterator has one supported partition
    intervals = build exact intervals for every axis
    cells = CartesianProduct(intervals)
    verify cells all-and-only partition complete iteration domain
    verify embedding is injective and uses available Tiles
    shards = bind cells to embedding Tiles
    groups = deriveReductionGroups(node, shards, nodePlan.reductionMerges)
  assignment = SpatialAssignment(shards, groups)
  demand = deriveExactDemand(structuredIR, assignment)
  return (assignment, demand)
```

`SpatialAssignment`只活在本次query/apply borrow中，是Q50.A所需的closed typed input；`SpatialPlan`才是candidate identity。final
result ownership、dependency demand和reduction contributions只由Q50.A返回，Q50.B不复制。observable function output的physical
owner同样从actual result availability与Q50.H selected store推导，不再用单独`ObservablePlacement`从result axis猜。

partition legality来自current op interfaces：普通parallel iterator只要result tile relation无未授权overlap即可切；reduction
iterator只有source-owned partial-reduction/coupled algebra interface完整时可空间切。TilingInterface说明如何
materialize选定tile，不负责profitability；Q50.B query只读iteration/indexing/result-tile facts，不调用其IR-building methods。

### Canonical spatial constructor

`canonical-spatial-assignment`是baseline所需的单值构造器，不是完整spatial domain的首点接口。它按下面规则直接构造一份
`SpatialPlan`并调用同一个structural close得到`SpatialAssignment`：

1. 从每个函数结果按SSA use-def反向到全部structured roots。`SemanticRootKey`以函数结果index为anchor，并记录沿途每条
   producer-result/consumer-operand relation；一个root有多条observable path时取typed path字典序最小值。没有observable path、
   key重复或跨越effectful/unsupported relation时fail closed。block ordinal、op名字、location和pointer不进入key。
2. 普通Linalg root通过pinned `mlir::linalg::LinalgOp`读取indexing maps和static loop ranges；attention通过自身typed ODS
   map/static-range accessors读取同类事实。`TilingInterface`提供iterator kinds，`DestinationStyleOpInterface`定位result maps。
   只有映射到result的parallel iterator进入canonical spatial
   participation；普通reduction iterator保持一个interval。FA的K2保持一个interval；FD的K2 interval-count product必须大于1。
3. 在available Tile count内，以bounded participant-count dynamic program直接求唯一factor vector：先最大化非空logical cell
   数；相同cell数时按structured iterator顺序选择字典序较大的interval-count vector。该过程只保存每个participant count的一个
   prefix，不建立option list、domain、proposal、score、backtracking或cross-op legality query。每个axis使用
   `BalancedParts(intervalCount)`；scalar/没有result-mapped parallel axis的普通root得到singleton cell。
4. available Tiles按typed `TileId`排序，canonical embedding把logical cells依次映到前`cellCount`个Tile。FD coupled reduction按
   result-mapped parallel coordinate形成一个stable group，每组选择该coordinate下字典序第一个contributor Tile作为merge Tile；
   final owners和partial contribution仍由Q50.A派生。
5. attention constraint query和spatial structural validator必须同时接受结果；任何失败都是source/interface或compiler contract
   failure，不在本work item内换另一个placement。返回值不含layout、movement、resource、cost或IR mutation。

定向验证覆盖：semantic root key不受无关op插入及operation pointer变化影响；chain、diamond、multiple outputs和support fanin不碰撞；
multi-axis/non-divisible/scalar取得确定的最大参与数；Tile输入顺序扰动不改变结果；FA保持K2单interval；FD形成K2多interval、
per-output-coordinate merge group及closed exact shards。测试同时断言source IR在query前后byte-identical。

### B-2 专项调研：lazy successor、global symmetry与proposal算法

Shardy propagation把op-specific dimension关系先抽成factor rule，再在graph上迭代传播，而不是为每条edge保存一个按participant count
压缩的legality cache；XLA tile assignment则把logical tile coordinate与ordered device ids一起验证。对照old donor后，本阶段采用
“完整raw domain + 独立proposal solver”的双入口：successor是legality唯一事实源，factor graph只排序和给sound lower bound。

current `nextFactorVector/nextDistinctTileSequence`可迁移为mixed-radix与k-permutation primitive；旧
`EdgeTransitionLegalityCache`把full domains压成`partitioned/axis/participant count`会让不同tail、multi-axis和embedding碰撞，必须删除。
old partial-state compute/transition/topology metrics可作为proposal features，但不能再形成live-boundary Pareto deletion。

Q50.B把每个`NodeSpatialPlan`定义为后续factor graph的typed variable，但本项本身只拥有raw domain和无剪枝proposal。
structured dependency与reduction group在Q52才成为constraint/cost factor。per-node exact domain由三个嵌套successor组成，
均不预建point vector：

```text
nextNodeSpatialPlan(current):
  if next merge-group Tile exists:
    return change that MergePlacement
  reset merges
  if next injective embedding exists:
    return change embedding
  reset embedding to first canonical embedding
  if next iterator-partition combination exists:
    derive mesh cells and reduction groups
    return first embedding + first merge placements
  return end
```

iterator partitions按iterator稳定顺序做mixed-radix successor，并在中间product超过available Tile count前停止；scheme产生的
interval vector先canonicalize，extensionally相同的balanced/uniform points只出现一次。embedding使用lexicographic injective
mapping successor；merge placements再按`ReductionGroupId`、Tile id稳定推进。任何successor都只修改caller-owned typed value，
不clone IR、不构造CardModule、不保存全域vector。

raw Tile排列只在确实可观察时保留。`TopologyAutomorphismAnalysis`从actual available endpoints、directed adjacency/link属性、
capacity、Tile-specific immutable target facts和fixed program endpoints构造颜色图，使用partition refinement/backtracking枚举并逐一
验证automorphism。若work limit耗尽，结果只含已经证明的automorphisms，至少始终含identity；未证明的permutation绝不参与dedup。

automorphism必须**同时作用于整份program plan**：所有node embeddings及所有merge Tiles都应用同一个Tile permutation。不能给每个
node各自选择orbit最小embedding，因为那会改变producer/consumer co-location、component overlap和route endpoints。完整plan的
canonical key为：

```text
canonicalSpatialKey(plan, automorphisms):
  best = serialize(plan)
  for permutation in automorphisms:
    best = min(best, serialize(applyToAllEmbeddingsAndMerges(plan,
                                                             permutation)))
  return best
```

Q52引入该能力时只在完整`SpatialPlan`上做这个quotient，per-node exact successor仍覆盖raw embeddings。future若要在partial plan上提前
canonicalize，只能使用保持已固定semantic prefix和external colors不变的automorphism stabilizer，并用tiny orbit oracle证明；否则宁可
多枚举。program inputs、fixed endpoints或unavailable Tiles会通过颜色自然打破对称。

#### NoC-aware communication graph与经典mapping proposals

Alpa的auto-sharding把每个op的partition strategy作为one-hot变量，把compute/collective放在node cost、resharding放在edge cost，并用
ILP联合选择；GSPMD/Shardy把conflicting shardings显式变成reshard/collective，而不是假设same device count就local。经典process/NoC
mapping把application communication matrix与processor distance matrix组成Quadratic Assignment Problem；高质量实现常用multilevel
graph partition、recursive bisection和swap/local search生成mapping。NCCL则先读取actual topology，再分别搜索ring、tree、split-tree
等collective graph。这些方法共同说明：placement必须看完整communication graph和actual topology，但placement、route family与event
schedule仍是三层不同决策。

Q52从A的complete proof为一个已关闭的spatial plan建立query-local communication-demand graph：

```text
CommunicationDemandGraph
  vertices:
    (semantic root, logical execution cell)
    reduction merge group
    fixed program/DDR endpoint when one really exists
  edges/hyperedges:
    producer-version -> consumer-cell exact fragment
    one producer-version -> equivalent destination cells (multicast opportunity)
    contribution cells -> merge group (distinct reduction information)
  weight:
    exact logical domain and, only when target encoding facts prove it,
    minimum bytes across every future representation
```

unknown minimum bytes使用0作为admissible bound，同时可用logical element count作proposal hint；hint不能剪枝。fanout是hyperedge而非把
同一payload机械重复计费，reduction contributions则是不同信息，必须分别到达merge。precomputed scores、multi-result、tail和
nonrectangular demand都沿A exact domain进入，不按op或tensor名字分类。

对一个complete placement，Q52可在H/J所需target facts已闭合后计算下列route-independent lower bounds：

```text
dilationLB:
  maximum over required fragments of
    minimum shortest-path latency from any eligible source to destination

hopWorkLB:
  sum shortest-path hop-bytes for fragments that carry distinct information
  multicast hyperedge uses at least payload * farthest-destination distance,
    not the sum of duplicated unicasts

cutCongestionLB:
  for every verified topology cut,
    mandatory distinct bytes crossing the cut / aggregate cut bandwidth
  take the maximum cut value

communicationTimeLB = max(dilationLB, cutCongestionLB)
```

directed links、asymmetric bandwidth和unavailable endpoints来自current topology；无证明的nominal bandwidth或假设route不进入bound。
这些bounds仍不选择path，也不声称多个transfers可重叠。H稍后在同一placement上枚举unicast/k-shortest、multicast arborescence、
reduction tree/ring/split-tree及broadcast-based many-to-many reshard；J才按actual links/events判断contention和overlap。

Q52的高质量placement proposals使用四个互补构造器，输出都只是Q50.B exact domain中的完整points：

1. **Topology-shaped seeds**：row/column/serpentine及最小内部hop的subsets，适合regular mesh但不限制domain。
2. **Communication graph recursive bisection**：按weighted demand cut递归切logical work graph，同时按actual NoC minimum-capacity cut递归
   切Tile graph，再匹配两棵partition trees；适合branch/cluster和非规则topology。
3. **QAP/one-hot solve**：变量`x[cell,Tile]`满足每个cell恰一Tile、同一node cells injective；目标用pairwise demand hint乘topology
   distance，加merge-location unary/pair factors。对bounded small proposal set可exact branch-and-bound/ILP；预算耗尽只返回feasible
   proposal与lower bound，不证明其它mapping不可达。
4. **Deterministic improvement**：对seed做Tile swap、cell move和merge-Tile 1-median/1-center调整；每次按完整semantic key打破平局，
   只保留经domain membership检查的points。它相当于Kernighan–Lin/局部QAP refinement，不形成no-good。

chain/tree factor DP可以把上述per-node proposal sets与pair lower bounds联合；general DAG先用recursive-bisection/QAP seeds，再由Q51
best-first比较。一个构造器失败或返回空不影响其它构造器，更不影响raw successor。对16 Tile，topology cut和small-QAP oracle可完整
枚举；production仍有显式work limit，不能把“硬件只有16 Tile”变成无界`K_n` product的理由。

边界示意：两个producer/consumer plan拥有相同Tile数量但embedding不同，Q52的QAP/hop/cut facts会给出不同proposal顺序；H可能为同一
placement选择不同unicast或multicast trees；J又可能因同时发生的其它traffic改变winner。任何一层都不能把自己的局部最优写成上一层
legality。

完整Q52 proposal层按下列顺序扩展本项的基础seeds，但proposal不是legality：

1. unpartitioned/single-Tile、最大parallelism、balanced与uniform-tail的边界点；
2. topology内部hop work较小的compact subsets/embeddings；
3. producer/consumer exact-demand owner交集较大的co-located或partial-overlap placement；
4. independent observable components优先使用disjoint Tile subsets；
5. reduction merge优先在contributor或直接consumer participant上，但其它available Tile siblings仍可达。

Q52为chain和一般DAG产生较好的complete spatial proposals时，复用一个不物化IR的factor-graph lower-bound query：

```text
unary(nodePlan):
  compute lower bound from logical work / participant count
  + mandatory reduction-merge logical work

pair(edge, producerPlan, consumerPlan):
  return a route-independent lower bound derived from endpoint partitions,
         relation rank/extent and topology
  return zero when a positive bound is not proven

proposeSpatialPlans(graph):
  build finite high-priority option iterators per node (not a legality cap)
  if dependency factor graph is a chain/tree:
    run exact dynamic programming on the supplied proposal options
  else if min-fill tree decomposition width is within proposal work limit:
    run variable-elimination DP
  else:
    run deterministic best-first partial assignment with admissible bound
  yield complete proposals in nondecreasing bound order
  finally leave Q51 exact coordinate successor available for every sibling
```

partial proposal factor不得调用另一套per-edge exact-demand planner。只有complete `SpatialPlan`关闭为`SpatialAssignment`后，才调用A的
唯一operand-level query一次并用完整proof细化proposal cost；DP中的pair factor只可读A的func-scoped relation analysis和两个endpoint
plans，给出保证不高估future redistribution的lower bound，无法证明就为0。它不返回boundary recipe、owner或placement rejection。

chain DP的recurrence为`DP[n,o]=unary(n,o)+min_p(DP[pred,p]+pair(pred,n,p,o))`；branch/fanin在tree decomposition
bag中联合计算所有incident pair factors。independent components可先分别求proposal frontiers，再用Tile-overlap lower bound做
deterministic product merge；不能假设它们永远可并行，因为physical Tile是共享resource。

这些Q52 bounds只能包含当前已知且单调的logical work、mandatory merge与最小transport；topology compactness、local-edge count、
component overlap等不能证明admissible的量只作tie-break/proposal priority。Q50.B不再像旧实现那样在region/fusion/layout/buffer/
schedule未赋值时用live-boundary estimated metrics删除spatial states。只有以下两类在本层可删除：

- exact spatial plans的extensional identity完全相同；
- 经verified topology automorphism和全部immutable colored facts证明等价。

full-coordinate dominance、no-good和bound pruning归Q50.F/Q51，在相应下游坐标已关闭后证明。旧nondominated/live-boundary算法的
数据结构与测试意图可迁入proposal/Pareto验证或Q50.F，但旧删除结论本身不能因source曾存在而照搬。

设node `n`有`K_n`个spatial points，logical mesh有`C`个cells，available Tiles为`T`，merge groups为`G`，verified topology
automorphism数为`A_t`。一个partition的raw
embedding数为`P(T,C)=T!/(T-C)!`，merge placements为`T^G`；完整program product为`ΠK_n`，所以禁止预建。successor的state
space为`O(rank+C+G)`，单步amortized work同阶；full-plan canonical key为`O(A_t * total plan size)`。chain DP在显式proposal options上为`O(NK²)`，treewidth `w`的variable
elimination为`O(NK^(w+1))`；超出work limit只停止proposal算法并返回coverage说明，不改变exact domain。Q52以后可根据profile
优化proposal/memo，不能反写合法域。

NoC mapping本质是QAP，exact solve最坏指数级；本文不承诺多项式。对`T=16`可用`2^(T-1)`cut subsets建立test oracle或在
work limit内预计算actual topology cut facts，production也可用directed min-cut下界替代全cut枚举。recursive bisection一次seed的
graph work近似`O((V+E) log V)`加partitioner work；一次完整Tile-swap sweep为`O(T²)`个delta evaluations。它们都只影响proposal
质量，work耗尽不触碰raw domain。

### 机制

- exact domain由`IteratorPartition × injective embedding × per-group merge placement`的三个lazy successors组成；
- structured iterator/result/reduction语义决定scheme eligibility，Q50.A只派生demand和availability，不作cross-edge placement filter；
- current compact、uniform-tail与independent-component disjoint placements是基础proposal siblings；Q52再加入co-located、
  partial-overlap和graph-driven siblings。全部verified-legal subsets/embeddings始终由exact successor可达；
- topology symmetry只通过verified automorphism quotient，启发式相似或resource renaming不能删state；
- chain/tree/general-DAG算法只产生proposal order和admissible lower bound，不保留local winner，不混入movement、schedule或
  fixed-capacity结论。

### B-3 专项调研：独立oracle、donor迁移与actual gate

Shardy verifier逐层检查tensor rank与dimension-sharding数量、mesh轴唯一/不重叠、device-id顺序和shape约束；XLA
`HloSharding::Validate`同样区分tuple/leaf、tile rank、device range、device uniqueness和device count，并在错误中附带具体shape/sharding
上下文。对应到本项目，`SpatialPlan` verifier必须只验证自身closed structural contract，Q50.A验证derived logical proof，Q50.C验证
actual work；不能让任何一层用下一层成功倒推自己正确。

三个oracle彼此独立并按其直接consumer顺序施工：

1. `SpatialPlanReference`用普通nested loops直接枚举小域，不include production domain header，也不调用production successor、
   canonicalizer或proposal helper。
2. `TopologyMappingReference`由`search-scalability`在H/J target facts闭合后施工，对小communication graph穷举embedding、merge
   placement、topology automorphism和route tree，验证Q52 bounds/proposals；它只在test中调用actual route enumeration。
3. `SpatialMaterializationReference`由`root-work-domain`施工，逐iteration point计算期望`(LogicalShardId, TileId)`和merge group，
   再读Q50.C actual TileRegion/operation relation比较all-and-only coverage；不以IR文本中出现多少Tile名字代签。

current及donor test的逐项迁移如下：

| 旧witness | 新owner与改写 |
| --- | --- |
| current multi-axis factor/embedding Cartesian equality | 加入Balanced/Uniform extensional dedup、tail、per-group merge后与`SpatialPlanReference`比较 |
| duplicate/out-of-domain embedding negative | plan verifier分别覆盖duplicate logical cell、duplicate Tile within node、unavailable Tile、wrong arity和merge group缺失 |
| current remainder/reduction embedding | 拆成parallel tail domain与partial-interface structural domain，不再混入其它判断 |
| old unsupported-reduction-combiner test | 删除该方向；只验证partial-reduction interface能否完整表示selected result group |
| scalar empty-factor/every singleton Tile | zero-iterator singleton-cell定义；任一available Tile raw state可达，global automorphism只在整plan canonicalize |
| current chain/branch/diamond Card Cartesian reference | Q50.B升级为program-level raw product；Q52加入global automorphism oracle，不能对node独立quotient |
| current partial owner/one merge Tile | 改为M/N/K mixed的多个`ReductionGroupId`、每组全部merge Tile choices及A final-owner proof |
| donor multi-result producer | per-result/coupled-result groups由interface声明，所有results进入A/C实际witness |
| donor bounded partial/disjoint、same-group remap、three distinct groups | 保留为exact siblings；proposal顺序可不同，domain不能因local transition class删除 |
| donor full-mesh rectangle/compact-first | rectangle/serpentine/compact成为topology-shaped seeds；任意subset/embedding仍由raw successor覆盖 |
| donor independent branches use all Tiles | recursive-bisection/disjoint proposal正例，同时保留共享Tile sibling给Q51比较resource/schedule |
| donor long-DAG nondominated states | Q52改为129-node proposal/work test；Q50.B successor本身不保留旧live-boundary pruning结论 |
| donor fanout/diamond/temporal-residency | fanout hyperedge、fanin pair factors与spatial/temporal轴独立性；B proposal不能固定E |
| donor reduction transition exact fragments | logical fragments归A，placement/merge归B，actual route归H；一个混合test拆到三个owners |
| donor deterministic order | 扰动DenseMap insertion、Tile input order、proposal family顺序和并行completion，raw/canonical keys保持确定 |
| donor unsupported/indeterminate demand abort | A typed outcome停止compile但不改B raw domain；不再缓存成edge transition legality |
| donor carrier failure does not reject placement | 保留并扩展layout/route/SPM/schedule failure metamorphic isolation |

实现切换时所有producer/consumer同批改接：Q51 state只保存`SpatialPlan`；A只接closed `SpatialAssignment`；C/D/G/H/F/I/J不再接
`LogicalShardTrial`或`SpatialPlacementAssignment`。Q49.P baseline不调用B domain。旧current/donor source中的factor/Tile/node-wide merge
结构可以作为算法素材，但不能保留compat wrapper或双reader。

independent有界穷举reference enumerator不调用production domain builder；在extent与Tile count均为2–4的小域中直接枚举：

- 每个iterator的BalancedParts/UniformExtent exact intervals并按extensional equality去重；
- factor product不超过available Tiles的全部multi-axis combinations；
- 全部injective logical-coordinate→Tile embeddings；
- 每个reduction group到全部available Tiles的merge placements；
- 由typed reduction interface允许/禁止的axis集合。

production successor与reference stable key集合必须逐node、逐program完全一致。shard domains再和逐点iteration oracle比较
all-and-only coverage；Q50.A结果验证final owners和每个merge group的contribution count、init与output domain。覆盖single op、
chain、independent branches、diamond/fanin/fanout、multi-result、scalar、multi-axis remainder、M/N/K mixed、两个reduction axes、
same-group remap、partial overlap、disjoint和three-stage distinct groups。

Q52 proposal验证与Q50.B domain验证分开：在同一小域上，chain DP与treewidth DP的首个proposal cost等于flat factor-graph minimum；general
DAG best-first的bound单调且不会让exact successor中的siblings不可达。关闭compact/component/merge proposals或反转它们的顺序，
domain集合与Q51有界穷举oracle optimum不变。verified topology automorphism前后的canonical set相同；增加unavailable/fixed-color
Tile只打破真实symmetry。

Q52/H的NoC专项oracle覆盖line、ring、2-D mesh、directed/asymmetric link、broken link和heterogeneous endpoint colors，以及chain、star
multicast、all-to-all、diamond和reduction gather communication graphs。有界穷举reference枚举全部embeddings和route trees，检查：
exact QAP/DP模式达到reference placement cost；heuristic模式只返回domain member；`dilation/hopWork/cutCongestion`从不超过actual
route optimum；multicast payload不会按destination错误重复成lower bound；改变NCCL式ring/tree proposal family只影响H的顺序，不改变
B domain或canonical key。

metamorphic tests让Q50.A relation unsupported/resource failure、Q50.G layout unavailable、Q50.H route unavailable和Q50.J schedule
failure分别发生，证明它们不会改写Q50.B domain；invalid `SpatialPlan`只返回compiler contract error，不能靠另一个
Tile placement掩盖。默认调用不构造statistics；显式test observer只证明16-Tile large-shape successor不预建排列/Cartesian
product、无IR clone/materialization且单步work受state size约束。Q51完整链前不运行重型LLaMA search。

Q50.B的actual downstream witness由Q50.C提供：complete assignment关闭后，每个exact shard和merge group必须各映到all-and-only actual
TileRegion work。局部gate通过后删除旧single-axis/node-wide-merge/trial/observable-placement字段。“非最大参与、非compact
physical subset或局部lower-bound较差的placement成为global winner”留给Q51跨轴gate。

本项完成门禁要求active source/test中旧`SpatialPlacementAssignment/CardSpatialPlacementDomain`、`iteratorFactors`、
`reductionMergeTile`、`getMaximumParticipantAssignment`作为winner、`EdgeTransitionLegalityCache`、`ObservablePlacement`和旧
live-boundary state零残留；CMake只注册新owner。fresh实现验证运行B reference/property、A closure、current production consumer、
named/production parity和FP16/BF16真实规模host gate；C actual、NoC bounds和完整source-to-package分别由后续直接owner施工。
测试必须实际执行而非unsupported；Q51完整链前不运行LLaMA search。

### Current/donor能力迁移

| Current / donor能力 | 终态owner | 保留或修正的witness | 退役条件 |
| --- | --- | --- | --- |
| current完整iterator factor与有序distinct-Tile successor | Q50.B lazy partition/embedding successors | all factors/tails/subsets/embeddings reference equality | 迁入两scheme和per-group merge后删除旧`iteratorFactors/tiles` API；nontrivial automorphism归Q52 |
| current maximum-participant constructive choice | Q50.B proposal ordering | proposal合法且关闭后domain不变 | 不再作为baseline fallback或local winner |
| old physical rectangle/compactness catalog | Q50.B compact proposal generator | full mesh包含all rectangles、compact 2x2优先 | exact domain不再只含connected rectangles；hop work不作legality |
| old `getNodeSpatialAxes` result0/parallel-axis恢复 | all-iterator interface query | multi-result、reduction、scalar与projected-result正负例 | result-axis/single-axis字段零残留 |
| old independent-component placement | Q50.B proposal + Q51 resource-aware combination | disjoint branch proposal、overlap sibling仍可达 | 不用component shortcut删除共享Tile plans |
| old chain/general-DAG partial states与live-boundary Pareto | Q52 factor-graph proposal；full pruning归Q50.F/Q51 | long chain、diamond、three-stage proposal和tiny optimum | 删除基于未关闭movement/schedule metrics的state pruning |
| old exact-demand transition cache | Q50.A func analysis/query memo | same assignment同proof、failure typed传播 | Q50.B不缓存legality bool、不按Tile relabel近似demand |
| old observable result axis/support-chain mapping | Q50.A final availability + Q50.H selected output store | multi-output/view链actual output coverage | 无`ObservablePlacement`或名字/axis恢复 |
| old edge carrier、residency、peer movement、schedule/resource signature | Q50.H、Q50.I/J与Q51 full state | 对应donor tests随各owner迁移 | Q50.B source/test不再include movement/schedule类型 |
| old/current reduction one-merge-Tile | Q50.B per-group merge + Q50.A requirement | M/N/K mixed、多reduction轴、coupled multi-result | node-wide optional merge字段和partial-as-owner tests零残留 |

## Q50.C：Maximal Single-Root TileRegion

### Pipeline contract

Pipeline position:
- Upstream IR / input:
  Q50.S归一后的immutable单函数TensorProgram与fixed semantic root facts、current StructuredDAG、Q50.B关闭得到的
  `SpatialAssignment`及Q50.A `ExactDemandProof`。输入只描述每个node/Tile的exact iterator work、operand reconstruction、
  final boundaries和reduction requirements；尚未选择region group、temporal、representation或movement。
- Current stage responsibility:
  派生每个`(structured root, Tile)`的`RootRegionWork`：all-and-only local iterator cells、同root contribution/merge work、由
  Q50.A operand reconstruction指定的pure tensor support DAG，以及在其它structured result/program input/constant处的typed
  boundaries。提供一个singleton root emitter；production只在Q50.D形成complete group assignment后由candidate materializer调用，不先建singleton
  CardModule再改成fusion。
- Output IR / files:
  planning query不产生IR或文件，只返回ephemeral `RootRegionWork`。independent singleton actualization或complete-candidate emitter
  产生一个`wafer.tile.region`，其actual operations通过typed materialization relation映回唯一semantic root；boundary只是logical/
  selected-value接口，不默认等于DDR movement。
- Downstream consumer:
  Q50.D把singleton work作为group partition的叶子并选择coupled groups；Q50.E–K为最终groups补齐temporal、representation、
  movement、buffer、event/resource和pipeline。candidate materializer直接从当前closed assignment构造一次完整CardModule，Q50.H
  关闭boundary movement后才进入Q50.0。
- User-level driver / named pipeline:
  由public `search` session内部静态组合；Q50.C本身不新增CLI、pass、独立provider或用户可手拼pipeline。Q50.C–K未闭合时
  public `search`仍不枚举完整Cartesian product、不运行重型LLaMA search。Q49.P与Q50.C复用同一个policy-free exact
  iterator-tile与TileRegion body-emission mechanics；baseline继续由自己的canonical assignment controller提供boundary和
  movement，不经search state或Q50.C中间容器重建。
- Explicit non-goals:
  不选择multi-root group、temporal tile、layout/encoding、local/peer/DDR action、route、buffer、schedule、cost或winner；不把
  boundary当成已经选择的DDR；不从函数名、operation ordinal、buffer或result axis恢复root；不缓存/replay body，不为Q50.D先
  物化可丢弃singleton IR。
- Done criteria:
  single/multi-result、DPS init、program input/constant、multi-producer/multi-path support DAG、rank-zero、softmax-weighted-sum、unpartitioned
  reduction、spatial contribution/per-output merge和effect boundary正负例通过；singleton actualization的每个`(root,Tile)`最多一个
  outer region且emission relation只映回该root，其它structured producer均为typed boundary；Q49.P与complete-candidate emitter复用同一leaf
  primitive；production planning/materialization计数证明零singleton prebuild、零Module/DAG clone/replay。

### Q50.C work-item分界

Q50.C由两个线性work items完成，不能在`canonical-root-work`中提前混入完整search domain：

- `canonical-root-work`只定义并派生每个canonical `(root, Tile)`的`RootRegionWork`，同时提供一个消费该work的caller-owned
  singleton leaf prepare/emit primitive。它直接验证work与A proof逐字段一致，并用test-only caller构造actual single-root
  `wafer.tile.region`；不枚举root alternatives、不建立Core state、不选择multi-root group，也不提交完整CardModule。
- `root-work-domain`在search-control-foundation之后补齐全部root/contribution/merge work successors、Core transition和selected
  group emitter，并在outer winner transaction中取代旧Module-return/closure walker。旧`SingleRootRegion`控制入口、whole-Module
  clone及donor退役属于后一个work item，不能由canonical leaf存在提前代签。

当前`canonical-root-work`的覆盖矩阵如下。shape只用于覆盖真实执行规模，不进入work identity、legality或materializer分支；
rank-zero和单一故障负例可使用小shape，但同一support/merge机制必须另有真实规模正例。

| 覆盖类 | 代表输入 | `RootRegionWork` exact断言 | leaf/downstream witness |
| --- | --- | --- | --- |
| aligned single-root | rank>=3、主要维度1024、all-16 Tile | 每个nonempty `(root,Tile)`恰有一个work；execution pieces与assignment逐字段相同，program input/constant boundaries和result pieces all-and-only | test-only singleton leaf每个work最多一个outer TileRegion，emission只映回该root |
| ragged multi-axis | rank>=3、主要维度1025/1031、两个以上partition axes | tail offsets/sizes、result domains和stable work order精确；Tile/shard输入顺序扰动不改变semantic work | actual tile/result slice覆盖无hole/overlap，source IR不变 |
| multi-result与DPS init | 1024级multi-result root及显式structured init producer | 每个result piece独立保留；init producer是typed structured boundary，不被support closure吞入 | leaf返回all-and-only selected root results，init root无emission relation |
| shared multi-producer support | 1025级insert/reshape/slice/pad或same-producer multi-path | 同一support result只有一个`SupportValueId`，required domain取exact union；每个data-carrying input和exact-empty branch均保留自己的boundary/use | support op只emit一次，两个structured producers都停在boundary |
| ordinary reduction | rank>=3、整除/非整除spatial contribution与多个per-output merge group | contribution由source Tile work持有，merge由merge Tile work持有；partial不是result owner，多个group同Tile仍合并为一个root work | singleton contribution/merge leaf分别消费同一A requirement；init exactly once由typed relation检查 |
| attention FD coupled merge | rank-5/6、K2为1024及1031或33x31 | Maximum/Sum/Accumulator仍属于一个merge work，component domains和final owner原样引用attention-ready proof | 本项只证明leaf输入闭合；selected attention action emission仍由attention-selected-decomposition完成 |
| bounded exceptions/failure | rank-zero contract；effectful support、missing boundary或work/proof mismatch单故障 | rank-zero形成唯一empty-vector piece；错误返回typed unsupported或compiler contract failure，不产生partial work | failure前后source generic text相同，无临时Func/Module/TileRegion残留 |

### Work Item `root-work-domain`覆盖矩阵

本项把既有per-site `RootRegionWorkAnalysis`收口成完整root×Tile successor并接入SpatialState；RootRegionWork仍是derived value，不新增
candidate choice或state field。施工门禁如下：

| 等价类 | 输入 | exact断言 | 直接witness |
| --- | --- | --- | --- |
| aligned/ragged完整域 | rank-3/4 FP16/BF16，1024及1025/1031，16 Tiles | successor按`(SemanticRootKey, TileId)`稳定顺序覆盖all-and-only nonempty execution/contribution/merge work；NoRoot被跳过但不是failure | Core只在完整root work域全部validated后入队SpatialState，仍返回missing Region且actualization=0 |
| support与boundary | 1025级multi-producer insert、same-producer multi-path、slice/reshape/pad、constant/capture、DPS init | support result去重、exact domain union、structured/program/constant boundary及exact-empty use逐项保留 | test caller用work准备singleton leaf；其它structured roots没有emission relation |
| multi-result/reduction/attention | multi-result 1024、ordinary multi-axis reduction 1025/1031、rank-5/6 FD | per-result final piece、每Tilecontribution、每group merge和Maximum/Sum/Accumulator coupled tuple完整；merge-only Tile仍是domain member | 多group同Tile只出现一个root work ID；leaf preparation不把partial当final owner |
| scalar/graph | rank-zero、chain、fanin/fanout/diamond、independent outputs | empty-vector execution合法；roots/tiles输入顺序扰动不改变work ID和successor顺序 | tiny/real collection与逐site reference相等，source byte-identical |
| typed failure | unsupported support、piece limit、proof/assignment mismatch、invalid successor cursor | Unsupported、Indeterminate、CompilerBug分类保持；当前spatial sibling continuation不被误删 | Core work counts区分query/validated work；不解析diagnostic string，不产生partial work或IR |
| owner迁移 | baseline canonical plan、Core、old `SingleRootRegion` wrapper/CMake/tests | baseline与search消费同一domain/collector；public search call tree无whole-Module root wrapper | old wrapper删除；test-only direct singleton actualization继续覆盖TileRegion/relations，production Region选择前singleton prebuild=0 |

本项不构造outer winner CardModule transaction：D尚未选择region partition，H--K也没有boundary/movement/event闭合。所谓emitter witness是
caller-owned selected singleton leaf primitive及test-only direct lowering；完整candidate emitter仍按后续work items原位扩展，同一SpatialState
不能因本项通过而越过missing Region。

### Work Item `root-work-domain`闭合结果

- 新`RootWorkDomain`借用同一immutable DAG、closed `SpatialAssignment`和`ExactDemandProof`，按
  `(SemanticRootKey, TileId)`字典序lazy跳过`NoRootRegionWork`并返回每个nonempty execution/contribution/merge work。successor携带
  domain-created `RootWorkCursor`，consumer不能从raw site伪造cursor，也不为验证cursor重复执行昂贵work query；domain不保存point vector。
- site outcome与domain successor分别typed区分NoWork/End、Unsupported、Indeterminate和CompilerBug；`collectRootWorks`只为需要完整
  derived set的直接consumer物化values。baseline canonical plan已经改用该collector，删除自己的root×Tile walker。
- Core对每个Q50.A-satisfied spatial choice完整走完root-work successor后才入队`SpatialState`，记录root-work step/validated counts；
  Unsupported消费当前choice但保留spatial sibling，Indeterminate保留同一spatial choice，broken contract停止。state identity仍只有
  `SpatialPlan`，RootRegionWork/proof不进入state。
- leaf合同原位演进为`PreparedRootWorkLeaf{optional execution, merges}`：merge-only Tile不再返回“空leaf”，多group同Tile显式保留全部
  merge placements。test-only singleton actualizer逐merge证明至少一个贡献shard承接；Region未选择前production仍不预建singleton IR。
- old `Planning/Search/SingleRootRegion` whole-Module wrapper和CMake owner删除；Conversion中仍被D--K donor消费的selected group lowering
  保留到对应owner迁移，不以本项虚报完整outer transaction已完成。

fresh证据：`RootWorkDomainTest` 3/3、`RootWorkOutcomeTest` 1/1，覆盖1024/1025、Tile顺序扰动、remote merge-only、multi-group、typed
piece-limit和cursor；既有`RootRegionWorkAnalysisTest` 9/9、`SingleRootTileRegionTest` 16/16、`PlanningSessionTest` 6/6、
`CanonicalBaselinePlanTest` 3/3及`SearchRoutingTest` 2/2共同通过；普通host unit 827/827；完整configured lit 262 passed、4 configured
unsupported；source organization通过。public search仍稳定停在missing Region且`candidate_actualizations=0`，本项不签发complete search。

### C-1 专项调研与leaf boundary

MLIR `getBackwardSlice`沿transitive defs返回topological operation set并允许filter frontier，适合结构性slice；它不知道每条use读取的
exact domain、structured-result stop boundary、exact-empty path或partial merge。MLIR tile-and-fuse通过TilingInterface从consumer
slice求producer tile，是actual transformation mechanism；其公开合同也说明multiple uses当前可能多次tile/clone同一producer。因此
Q50.C不能把通用backward slice或greedy tile-and-fuse当成correctness owner，只能把它们作为A proof已经决定边界后的actual builder。

Q50.A迁移已经让single/coupled/current materializers消费`SpatialAssignment + ExactDemandProof`，删除`LogicalShardTrial`、第二次
support-chain walk和concrete-op `TypeSwitch`；`ConsumerInputReconstruction`按同一`TensorIndexingOpInterface` description验证并clone
selected steps。仍待Q50.C收口的是`RootRegionEmitter`扫描完整function、root union/closure、implicit DDR stage、peer event与region order等
跨层职责；这些动作分别属于C、D、H、J和outer transaction，不能继续由一个root emitter长期拥有。

`RootRegionWork`是纯derived value，不是Q51坐标：

```text
RootRegionWork
  root: SemanticRootKey
  tile: TileId
  executionPieces: ExactIteratorPiece[]
  contributionWork: ReductionContribution[]
  mergeWork: ReductionMergeRequirement[]
  operands: RootOperandWork[]
  supportValues: ReconstructedValueWork[]
  boundaries: RootBoundaryRequirement[]
  resultPieces: RootResultWork[]

ReconstructedValueWork
  value: SupportValueId                 // current op result semantic id
  requiredDomain: ExactIndexSet         // union of all root-local uses
  inputs: SupportInputRequirement[]

RootBoundaryRequirement
  source: StructuredResultVersion | ProgramInput | Constant
  requiredDomain: ExactIndexSet         // may be exact-empty
  consumerUses: RootUseId[]
```

`SupportValueId`按source operation semantic id与result number定义；同一个support result经多个root operands或多条path使用时先union
required domains，再只建立一个work node。不能把`(op,result,demand)`作为identity，否则不同demand会把同一SSA definition复制多次。
每个input requirement仍保留自己的exact relation/domain，故union不会丢失multi-operand insert/concat provenance。

work派生算法为：

```text
deriveRootRegionWork(root, tile, assignment, demandProof):
  executionPieces = assignment.cells(root, tile)
  contributionWork = demandProof.contributionsProducedBy(root, tile)
  mergeWork = demandProof.mergeGroupsOwnedBy(root, tile)
  if all three are empty: return NoRootWork

  for each payload-read root operand in operand order:
    key = DemandKey(root, operand, logicalShardMappedTo(tile))
    reconstruction = demandProof.reconstruction(key)
    add its structured/program/constant boundaries, including Exact(empty)
    for step in reconstruction.topologicalSteps:
      node = supportValues[SupportValueId(step.result)]
      node.requiredDomain union= step.outputDemand
      union each typed input requirement by operand and exact domain

  topologically sort supportValues by SSA def-use with semantic-id tie-break
  verify every nonempty root use has one reconstructed value or boundary
  verify each support node is pure and owns no structured semantic root
  verify every region capture is a listed boundary
  return RootRegionWork
```

`maximal single-root`只表示同一`(root,Tile)`的selected execution/contribution/merge及其all-and-only support不被任意拆散；它不向另一个
structured root穿透，也不把forward users、peer endpoints、DDR stores或event order拉进来。merge-only Tile仍可有该root的work；完全
empty Tile没有region。

同一个support value若服务同一D-selected group内多个roots，D group emitter可按`SupportValueId + exact domain`合并各root works；
跨group sharing必须由H形成显式boundary/version或由D增加明确的consumer-local replica，C不能借全function buffer map暗中共享。这样fanout既不在每个use
重复构造，也不会跨region产生不可见SSA依赖。

leaf emitter的窄接口为：

```text
prepareRootRegion(work, algorithm, temporal, boundaryBindings,
                  representations): PreparedRootRegion
emitRootRegion(prepared, TileRegionOp region, RewriterBase &rewriter):
  bind listed boundaries to region-owned SSA values
  emit each support value once in topological order
  emit selected semantic root/contribution/merge work once
  return result SSA values and root materialization relations
```

这里`boundaryBindings`只表示已经由G/H选定并由outer builder提供的SSA值，不选择local/peer/DDR。emitter不扫描parent function、不创建
TileModule/CardModule、不决定region peers/order，也不修改自己的root op。standard ViewLike/Subset/Tiling builders优先；只有标准
interface实际需要复制一个selected op region时才使用局部`IRMapping`，不能generic-clone整条support chain。

正确性由A proof逐step归纳：所有boundary在required domain上已绑定正确值；每个support transfer对其output demand精确重建；拓扑顺序
保证inputs先定义；最终root在executionPieces上得到all-and-only operands。merge/contribution另由A requirement覆盖。因此证明与
root op数量、rank、Tile数和workload无关。构造relation graph由A完成一次；C work derivation复杂度为当前root/Tile非空
reconstruction nodes/edges与merge groups线性，不walk完整DAG，也不会因为16 Tile固定重复完整分析。

本子任务直接测试work而非actual IR：single/multi-result、program input/constant、DPS init、multi-producer insert、same-producer
diamond、exact-empty、shared support、rank-zero、merge-only Tile和effect boundary；断言structured producer恰为boundary、support value
只出现一次、无peer/DDR/event字段，并将work中的每个domain与A proof逐字段比较。

### 机制

对给定structured root和Tile，把Q50.B分配给该Tile的所有exact cells、该Tile负责的同root merge groups以及Q50.A给出的
per-operand reconstruction合并成一个`RootRegionWork`。这里maximal只表示“同一root/Tile的selected work不被任意拆成多个
互不知情regions，shared support step在该region内只构造一次”；不是最大Tile、最大fusion或全DAG closure。

closure只从Q50.A唯一operand proof构造，禁止再递归walk所有operands形成第二种边界语义：

```text
deriveRootRegionWork(root, tile, spatialAssignment, demandProof):
  work.iterationDomains = all execution cells bound to (root, tile)
  work.mergeRequirements = all same-root groups merged on tile
  for each tensor operand of root:
    recipe = demandProof.reconstruction(root, operand, tile)
    union recipe steps by source op result, accumulating exact demanded domains
    add structured results and program inputs as typed boundaries
    record constants/scalar captures from SSA
  topologically order support steps by source block order and SSA edges
  verify every root operand and region capture is reconstructed exactly once
  verify no support step owns another structured root or effect
  return work
```

同一producer通过`insert_slice`等multi-operand op的多条path到达时，work包含一个shared support node和每个data-carrying operand的
exact boundary；不能因“有两个producer”递归穿过其中一个，也不能把support op本身升级为semantic root。显式DPS init producer
同样在boundary停止。program input、constant、rank-zero scalar和nested region capture按typed SSA role处理。

singleton emitter消费最终已选的temporal/physical boundary values后，通过root的TilingInterface或semantic op tiled interface构造
actual iterator work；support steps按`TensorIndexingOpInterface`/standard view/subset semantics重建。一个source structured op可能
lower成多个Linalg/Tile ops，它们仍全部映回同一root id；反之region中出现第二个root id就是verifier error。空间reduction的
contribution与同root merge可在同一Tile region内按SSA依赖排序；多个output groups选择同一merge Tile时共享一个root region，
不为每组创建孤立region。

### C-2 专项调研与complete-candidate CardModule transaction

MLIR operation pass只能修改其current operation内部的IR，不能从leaf pass增删parent block中的siblings；可嵌套调度的anchor又必须是
`IsolatedFromAbove`。MLIR rewrite要求一次pattern在首次mutation前完成match，并经`RewriterBase`执行create/replace/erase；dialect
conversion的通用rollback需要延迟erase/replace和额外bookkeeping，官方也建议能预验证时使用no-rollback。LLVM的`scope_exit`
只提供“作用域退出执行清理、成功后release”的窄RAII语义。对应到本仓不能把whole-Module clone包装成所谓事务：parent
`builtin.module` transformation才拥有新增`wafer.card.module`和删除旧TensorProgram的权限，新建CardModule subtree是唯一rollback
scope。

current IR已经给出恰当结构边界：`CardModuleOp`和`TileModuleOp`都是`IsolatedFromAbove + SymbolTable`，Card verifier从同一个parent
Module读取target topology并要求all-and-only available TileModules；因此另造outer Module、复制module attrs/topology、复制全部直接
symbol都没有语义必要。current `CandidateMaterialization`和`DependentDataflow`先clone完整Module，current
`SingleRootTileRegion`又创建新Module并运行`cloneModuleTopology`，这些是旧的隔离手段，不是终态apply API。

事务明确拆成只读prepare和一次emit/commit：

```text
prepareCardModuleCommit(sourceModule, completePlan):
  require one unchanged source IR epoch, fixed semantic facts and one complete B--K assignment
  validate every selected algorithm, placement, group, traversal,
           representation, movement, buffer, event and execution structure
  derive all RootRegionWork, RegionGroupWork and physical boundary bindings
  close every Tile entry result, cross-Tile endpoint and program output
  compute the exact card-shared and Tile-private symbol declaration closure
  require all available TileIds exactly once and establish stable TileId order
  prove target IR needs no source-function SSA or source-private symbol
  prove every source symbol selected for deletion has no surviving external use
  return PreparedCardModule

emitPreparedCardModule(sourceModule, prepared, rewriter):
  create CardModuleOp directly in sourceModule's body
  cleanup = scope_exit(rewriter.eraseOp(cardModule))
  create every available TileModule in stable TileId order, including empty Tiles
  create only the declarations in prepared.symbolClosure
  for each selected region group in (TileId, selected event order):
    call C singleton emitter or D coupled emitter exactly once
  create H--K selected movement, buffers, events and Tile entries exactly once
  verify each TileRegion, TileModule, CardModule and the complete parent Module
       while the source TensorProgram is still present
  erase the already-proven-dead source TensorProgram/planning operations
  cleanup.release()
  return the committed CardModuleOp
```

`PreparedCardModule`是本次module epoch内的move-only typed value：它保存closed plans、stable semantic ids、declaration descriptors和
必要的短生命周期non-owning source references，不拥有、clone或缓存IR。prepare期间零mutation；emit只新增一个CardModule subtree。
所有可能失败的builder、interface、symbol closure和verifier调用都发生在source erase之前。target不得引用source function内部SSA，且
所有需要保留的symbol已经在Card/Tile symbol table内建立；所以最后的source erase只是经预证的parent rewrite，不再包含可能失败的
步骤。失败时RAII只擦除新CardModule，原Module、source function、topology和其它siblings逐字节保持不变；它不依赖dialect-conversion
backtracking，也不返回partial Tile set或让planner换winner重试。

topology始终留在原parent Module，由Card verifier直接读取；不会再有`cloneModuleTopology`。card-shared声明按typed symbol-use closure
建立，Tile-private声明按对应Tile closure建立，禁止“clone所有direct declarations”。若一个immutable declaration可以在完整预验证后
唯一move，move只能放在最终无失败commit段；通常直接由typed descriptor创建selected declaration更清楚。普通SSA不能跨
`IsolatedFromAbove`边界，program input、structured boundary、movement buffer和result必须成为TileRegion/Tile entry的显式argument、
owned value或symbol reference。

`IRMapping`只允许服务一次actual selected construction，例如给一个tiled op的region参数建立新映射；不能扩大到Module、Func、完整
DAG、其它Tile body或loser candidate。同一个support value由C-1的`SupportValueId`在其owner region内只emit一次；跨Tile的相似work
通过共享immutable relation analysis和prepared descriptors避免重复query，但各Tile的actual operations仍因不同SSA owner、offset、route
或schedule而各自构造，不能缓存body后replay。

Tile emit只有在Q52证明source read、context/diagnostic、symbol allocation和detached ownership均thread-safe后才可并发；并发任务只构造
互不引用的detached Tile-owned payload，最后按TileId稳定合并。C-2不以并发为完成条件。默认路径不采集per-root统计；显式observer
可以记录work count/timing，但不进入合法性、控制流或返回类型。

failure分类为：source op/interface本身不能表达candidate work是typed unsupported；prepared work缺boundary、symbol closure或与complete
assignment矛盾是compiler contract error；emit/postverify失败是candidate materialization bug。SPM、DDR、transport和target resource
rejection只在C-2生成actual IR后由Q50.F/Q50.0返回；C emitter自身不得repair、fallback或选择下一candidate。

### Gate

- `deriveRootRegionWork`直接测试single/multi-result、DPS init、program input/constant、region capture、view/reshape/slice/pad、
  multi-producer insert/concat、same-producer multi-path、rank-zero、softmax-weighted-sum、unpartitioned reduction、multi-axis contribution和
  per-output merge；不以actual emitter成功代签query；
- multi-producer case逐项检查两个structured result boundaries、每条exact required domain、一个shared insert/reconstruction DAG和
  consumer root emission；不能只检查“producer node没有被标成emitted”；
- test-only singleton actualization对每个`(root,Tile)`最多一个outer TileRegion，all emitted compute映回同一root；其它structured
  producers不进入region。多个merge groups同Tile时一个region收齐all-and-only contributions且原init exactly once；
- source/custom/generic roundtrip、verify-each、TilingInterface failure、effectful support、missing boundary和postverify injection覆盖
  atomic failure；失败前后source digest相同且无CardModule/temporary symbol；
- atomicity按阶段注入failure：只读prepare、首个Tile中途、后续Tile中途、symbol closure、Card verify各有witness；每个case都比较
  source完整generic-form文本、operation数量和symbol use，断言没有残留CardModule、TileModule、declaration或orphan use；
- current single-root tests逐项迁移：exact-shard/empty-Tile检查所有available TileModule与all-and-only regions；capture检查显式region
  arguments且无跨isolated SSA；multi-producer support-chain检查两个structured boundaries、shared support一次和consumer root；
  multi-result/rank-zero由C actual witness承担，temporal/reduction mechanics分别由E与A/C owner承担但仍从同一outer transaction调用；
- production counters证明partial planning与Q50.D group selection期间singleton actualization为零，每个complete candidate完整CardModule只构造一次；无
  Module/Func/DAG clone、body cache/replay或default statistics；
- Q49.P canonical assignment与Q51 winner复用同一root-work/emit primitive；两者policy controller独立。轻量FP16/BF16
  source-to-package证明下游直接消费，Q51完整链前不运行重型LLaMA search。

### Current/donor能力迁移

| Current / donor能力 | 终态owner | 保留或修正的witness | 退役条件 |
| --- | --- | --- | --- |
| current `collectRootClosure` recursive operand walk | Q50.A operand proof + Q50.C `deriveRootRegionWork` | init/multi-producer/capture/effect boundaries | 唯一proof消费就位后删除第二walker和“support”分类 |
| current `buildRegionFunction`/`IRMapping` one-root construction | Q50.C selected leaf emitter | actual single-root relation、multi-result、rank-zero | 不再先建private fragment Func再move/replay；最小actual op mapping保留 |
| current `composeTileEntry` implicit DDR destination与pending fallback | Q50.H explicit boundaries + Q50.J selected event order + outer materializer | cross-root/output writeback、dependency cycle正负例 | 无missing-value临时DDR、无`pending.begin()`隐式顺序 |
| current `cloneModuleTopology`/all-direct-symbol clone/new outer Module | parent-Module winner transaction + typed symbol closure | all available Tiles、card/Tile declaration lookup、source-preserving failure | old `OwningOpRef<ModuleOp>` result、topology copy和all-symbol copy零残留 |
| current `StructuredIterationTile`与temporal/reduction helpers | shared root tiled implementation owned byC/E | arbitrary offsets/sizes/tails、multi-result、reduction | baseline/search同一primitive且不选择tile |
| current node-wide partial contribution/merge region | Q50.A per-group requirement + Q50.C same-root work | M/N/K mixed、多groups同merge Tile、init once | node-wide merge optional/partial-as-owner零残留 |
| current `materializeCardSingleRootRegions(LogicalShardTrial)` | C work query + outer transaction中的singleton emitter | exact assignment、empty Tiles、atomic failure | old trial/epoch/module-return API删除；test也不通过第二个Module取得隔离 |
| current `CandidateSupport.cpp`中的root capability、tile geometry、loop-tile与output-boundary杂糅 | Q50.S fixed semantic description、Q50.C root work、Q50.E temporal geometry及outer entry builder | 对应interface、tail、reduction与boundary tests逐项迁移 | 文件与API不再使用`CandidateSupport`/`candidate tile`/`dataflow synthesis`泛名；每项进入实际owner后删除旧杂糅文件 |
| current `CandidateMaterialization`/`DependentDataflow` whole-Module clone | A/C/H各自query与complete-candidate emitter | selected edges、operand binding、failure classification | 本路径不再有candidate clone API；其它owner的clone必须在其迁移任务单独清点，不能由C虚报全仓清零 |
| old complete-rank/coupled traversal builders | Q50.D donor | connection/fusion能力与tests逐项迁移 | 不因C singleton完成提前删除或混入C |

## Q50.D：Region Formation and Producer Execution

### Pipeline contract

Pipeline position:
- Upstream IR / input:
  Q50.S normalized immutable TensorProgram与fixed semantic facts、current StructuredDAG、Q50.B closed spatial assignment、Q50.A
  exact demand/reduction proof，以及Q50.C为每个`(root,Tile)`派生的singleton `RootRegionWork`；尚未选择temporal/layout/movement。
- Current stage responsibility:
  在每个Tile上惰性枚举由local dependency graph连接的root-work partitions；为每份mandatory root work及显式额外producer work
  选择top-level或consumer-nested execution instance，并让每个exact consumer-use fragment绑定region-local stored value、direct nested
  value或cross-region boundary。query只形成typed region/execution plan；complete-candidate outer materializer一次构造actual regions。
- Output IR / files:
  不产生IR或文件。Q51 state保存typed `RegionPlan`：root partition、execution instances和use bindings；derived nested relations、
  operand reconstruction、lifetime和cost不存state。candidate actual IR中每个group对应一个outer TileRegion，内部SSA/loops直接证明
  selected execution与delivery，不写group/score/cache attr。
- Downstream consumer:
  Q50.E在已选actual group traversal上展开完整temporal vector/wave-loop order；Q50.F–K继续处理feasibility、representation、
  movement、buffer和schedule。只有complete assignment才进入Q50.0。
- User-level driver / named pipeline:
  只由public `search` session静态组合，不新增pass、CLI、单独fusion provider或旧rank selector；Q50.D–K未闭合时public search仍
  不枚举完整domain或运行重型LLaMA search。
- Explicit non-goals:
  不用per-edge `fused` bool、把retention/fusion/recompute揉在一起的action recipe、group attr、symbol名或operation ordinal表示selected
  execution；不选择temporal、layout、
  cross-group movement、buffer、cost或winner；不把same Tile自动当coupled，不在Q50.C actual IR上事后拼region或clone/replay body。
- Done criteria:
  per-Tile partition、execution-instance与use-binding domain和independent reference在chain、fanin、fanout、diamond、multiple sinks、
  disconnected图上集合相同；independent stored、consumer-nested、explicit replica和cross-group boundary均可达，temporal-dependent
  nesting明确deferred；singleton、maximal和intermediate groups的independent/production actualization各恰一region，mandatory work exact覆盖，
  replica显式可数，ordinary root的direct nested SSA无中间DDR，stored value有独立producer traversal，fanout共享version只生成一次；
  merge与attention execution identity分别被Q50.E和Q50.S直接消费，不在D伪造低层action；旧CompleteTraversal/ProducerTileFusion能力
  按E/G/H/S的直接consumer逐项迁移后退役。

### Q50.D work-item分界

Q50.D按两个work items交付：

- `canonical-region-plan`只把canonical `RootRegionWork`关闭为deterministic singleton `RegionPlan`。每个nonempty work独占一个group；
  每个execution piece和merge requirement各有一个top-level required execution；每个boundary use按source/consumer/eligible owner形成
  external `DemandFragmentId`。本项没有multi-root group、nested execution、replica、local stored/direct binding或search successor。
- `region-execution-domain`在`root-work-domain`之后枚举完整connected partitions、execution placements、sharing/replica和local use
  bindings，接入Core并提供selected group emitter。旧`CoupledRegionDomain`、movement反向决定region行为及whole-Module apply只在后一个
  work item迁移/退役，不能因singleton plan存在提前报完成。

当前`canonical-region-plan`覆盖矩阵如下。真实shape只用于覆盖完整fragment和tail，不进入group/execution/binding identity；
tiny graph只用于独立组合oracle或单一故障负例。

| 覆盖类 | 代表输入 | canonical plan exact断言 | 直接下游witness |
| --- | --- | --- | --- |
| aligned/ragged single root | rank>=3、主要维度1024/1025、all-16 Tile | 每个nonempty work恰有一个singleton group和一个required execution；tail不改变group identity/order | canonical-temporal-plan能逐execution读取完整iterator scope；planning前后IR不变 |
| chain与跨root boundary | 1024级producer→consumer，same/different Tile映射 | 两个roots永不在本项自动合组；每个nonempty structured boundary use按eligible owner形成all-and-only external fragments | H后续可从fragment直接派生movement，D不写DDR/peer选择 |
| fanin/fanout/diamond与multi-path | 1025/1031级multi-root DAG | 一个boundary source用于多个operands/uses时fragment identity不按node pair折叠；input/root/shard顺序扰动后plan逐字段相同 | region-execution-domain以同一fragments枚举sharing/split siblings |
| multi-result与DPS init | rank>=3 multi-result producer/consumer | producer result、consumer operand、owner shard/group均进入fragment identity；init boundary不被当普通data edge丢失 | 后续representation/movement能按result/use区分版本 |
| ordinary spatial reduction | 整除/非整除、多output merge group且多个group可同Tile | execution instance覆盖每个contribution shard一次；每个merge group另有一个merge execution，partial不签发final version | canonical-temporal只给execution scope建temporal plan，merge保持typed merge scope |
| FD coupled state | rank-5/6、K2为1024/1031或multi-K2 | Maximum/Sum/Accumulator仍由一个merge execution引用同一requirement，不能拆成三个region executions | attention-work-projection后续从同一group/merge ID投影actions/resources |
| invariant与empty work | program input/constant/scalar capture、merge-only Tile、empty Tile、rank-zero | invariant boundary形成无owner external fragment；merge-only work仍有group/merge execution；empty Tile无group；rank-zero execution ID合法 | leaf/temporal query不依赖shape rank或虚构shard |
| typed failure | duplicate work ID、boundary use缺required domain/owner、execution/merge不属于work | 返回compiler-contract failure且不产生partial plan，不压成unsupported或空domain | 修正后的同一输入可重新query，source IR byte-identical |

### Work Item `region-execution-domain`覆盖矩阵

本项同时签发`RegionPlan`/RegionState exact domain和供完整candidate transaction调用的selected Region/Execution construction与verifier。
partial query仍然零IR；actual builder只在完整suffix已关闭的caller-owned subtree运行，不自行补Temporal/G--K default。实现和收尾逐项检查：

| 等价类 | 输入 | exact断言 | 直接witness |
| --- | --- | --- | --- |
| connected partition | 2--6 root的chain、fanin、fanout、diamond、multiple sinks与disconnected graph | 每Tile connected partition无重复；不同potential components不合组；root/work输入顺序扰动不改变plan集合 | independent flat RGS×use-choice oracle与production集合/count一致 |
| use supply正交选择 | same-Tile producer/consumer | External、StoredRequired、DirectRequired、StoredReplica、DirectReplica全部可达；required execution不能同时stored和nested到不兼容consumer | `RegionPlan`显式local/external binding、required/replica execution和TopLevel/Nested placement |
| mixed local/remote与replica | producer/consumer同Tile、不同Tile、同operand多owner pieces | required local只在same group；pure producer的stored/direct replica可跨group/Tile；effectful producer不产生direct/replica | boundary sibling始终保留；replica数量和consumer group归属逐项可数 |
| fanout sharing | 一个producer供多个operands/consumers | 同一个required execution可由多个stored bindings共享；split plan显式产生多个`ReplicaExecutionId`；不同nested consumer不能误共享required execution | shared/split plans均为domain member，canonical key不依赖first-use/materialization order |
| real-scale graph | rank-3/4 FP16/BF16，1024及1025/1031 | RootWork fragments、tail、multi-result/result number与RegionPlan identity完整；query前后source byte-identical | Core形成`RegionState`且partial actualization=0；test-only closed suffix与full-feasibility分别构造selected singleton/intermediate/maximal regions |
| reduction/attention identity | ordinary contribution/per-group merge、FD coupled state | merge execution可stored/direct但不能伪造root replica；Maximum/Sum/Accumulator保持一个merge identity | D逐ID签发给E/S；partial merge的parent-dependent traversal由E构造，FA/FD action expansion由attention-production-closure构造，不在D伪造actual node |
| selected construction | stored required、direct nested、stored/direct replica、fanout shared/split，1024/1025 | 每个selected execution和use binding all-and-only materialized；direct nested SSA、stored traversal、replica count与region boundary精确 | verifier从actual region/SSA/relations重建同一RegionPlan；后续G/H builder按typed IDs消费 |
| typed failure | duplicate/missing group work、extra binding/replica、非法placement、domain cursor/plan错配、prepare/emit中途失败 | query `contains` fail closed；actual失败由candidate subtree guard回滚，不删除其它Region siblings | 修正plan可重新验证；source不变，失败subtree无残留 |
| donor/owner | old `CoupledRegionDomain`与cross-layer apply | public/Core和actual materializer只消费current RegionPlan/selected builder | nested/replica/coupled能力及negative tests逐项迁移后删除旧domain/apply，不保留第二consumer |

本项query输出typed selected group/execution/use descriptors；同一work item还提供outer complete-candidate transaction调用的selected builder和
verifier。它不在partial state提前生成CardModule，真正outer emit仍只在Temporal、representation、movement、storage、event和schedule
全部关闭后执行一次。

### Work Item `region-execution-domain`当前已实现子集与重新打开门禁

- `RegionPlan`原位扩成required execution placement、explicit replica execution、local/external use binding与Stored/Direct delivery的current
  schema；candidate identity只含typed root/work/fragment/execution IDs，不含exact sets、pointer、proposal history或movement choice。
- 新`RegionDomain`从canonical RootRegionWork leaf descriptors建立per-Tile potential components；restricted-growth successor只枚举connected
  partitions。固定partition后对每个structured fragment完整枚举External、StoredRequired、DirectRequired、StoredReplica、DirectReplica；
  required local只在same group，pure root replica可跨group/Tile，effectful root不产生direct/replica。
- required execution可由多个stored uses共享；不兼容nested consumers不能共用同一required execution，split sibling通过独立
  `ReplicaExecutionId`显式表达。selected local bindings必须使multi-root group连通；extra/missing binding、replica或placement使`contains`
  fail closed。domain/cursor不预建plan vector，root-work输入顺序扰动不改变successor集合。
- Core增加`RegionContinuation`与session-local `RegionDomain` cache；cursor不进入`RegionState`。同一SpatialState的全部Region siblings可
  resume至exhausted，public search取得first validated RegionState后稳定返回missing Temporal，仍不构造IR或actual candidate。
- old `CoupledRegionDomain`及cross-layer apply已退出public/state，但因仍独有selected temporal/layout/movement actual lowering能力而按
  donor规则暂留；后续E/G/H emitter逐项迁入后再删除，不能为追求零文件提前丢能力。

fresh证据：`RegionDomainTest` 5/5，覆盖same/cross Tile、五类supply、fanout shared/split replica、chain/fanin/fanout/diamond/disconnected
independent oracle及rank-4 BF16 1031；`PlanningSessionTest` 6/6、`CanonicalRegionPlanTest` 5/5、`CanonicalTemporalPlanTest` 4/4、
`CoupledRegionTest` 6/6、`DataMovementTest` 6/6和`SingleRootTileRegionTest` 16/16共同通过；普通host unit 832/832；完整configured lit
262 passed、4 configured unsupported；source organization通过。public search diagnostic为missing Temporal且`candidate_actualizations=0`；
本项不签发Temporal或完整candidate actualization。

完成状态复核后，本段证据只签发RegionDomain与Core transition，不能签发整个work item。current complete-candidate carrier不含
RegionPlan，full-feasibility只接受canonical singleton region；selected stored/direct、nested、replica和coupled group没有通过同一
candidate transaction直接构造并验证。`region-execution-domain`因此重新打开，必须在本项内迁入selected construction、actual verifier和
真实规模下游witness，不能旁挂第二个closure任务或继续让旧CoupledRegion donor代签。

### D-1 专项调研：region、execution instance与use binding边界

MLIR structured fusion把consumer tile的operand slice经indexing relation反推到producer iteration tile，再在consumer loop中
rematerialize该producer；它不是一个“edge已融合”布尔事实。官方`fuse_into_containing_op`实现还明确记录：同一producer在containing op
内有多个uses时当前可能被tile/clone多次，复用仍是待解决项。因此TilingInterface/SCF tile-and-fuse只能作为selected execution的
builder，不能替planning决定duplicate/share。MLIR One-Shot Bufferize也先在完整SSA上分析alias/equivalence再统一rewrite，进一步说明
stored value与direct nested value要先成为typed plan，而不能等buffer lowering按当前use临时决定。XLA把fusion限制在独立fusion
pipeline，IREE也把dispatch formation作为明确phase；二者都没有把fusion作为任意movement action的一个枚举标签。

Q50.A迁移已经删除逐edge `traceProducerToConsumerChain`；current coupled materializer用共享`StructuredDAGAnalysis`的typed edge
adjacency判断group sinks，input reconstruction读取同一proof。仍待D收口的是`CoupledRegionDomain`只保存group及
`fusableEdges/forbiddenInternalEdges`二值矩阵、`CardDataMovementDomain`以Retained/Refetch/Recompute反向决定group内部行为，以及
`materializeCardCoupledRegions`同时接temporal、representation、implementation和movement assignment并返回整个cloned Module；
`ProducerTileFusion`仍递归处理slice并可能隐式assemble full producer。需要迁移的是exact slice composition、DPS init rebasing、
insert-slice window、fanout reuse和observable result mechanics，不是这套跨层API和默认递归策略。

`Retained / Coupled / Recompute`也不是同一维度的三个互斥action：前两者混合描述execution nesting与value delivery，recompute描述
额外执行次数。终态把region membership、execution instance和use binding拆成三个正交对象：

```text
RegionPlan
  groups: RegionGroupPlan[]

RegionGroupPlan
  tile: TileId
  mandatoryRoots: RootRegionWorkId[]
  executions: ExecutionInstancePlan[]
  localBindings: LocalUseBinding[]

ExecutionInstancePlan
  root: SemanticRootKey
  workSource: RequiredPiece(RootWorkPieceId)
            | ReplicaFor(DemandFragmentId)
  placement: TopLevel
           | NestedUnder(consumerExecution, DemandFragmentId)

LocalUseBinding
  use: DemandFragmentId
  producer: RegionValueVersionId
  delivery: StoredRegionValue | DirectNestedValue
```

`RegionValueVersionId`是typed sum：`ExecutionResultVersionId`、region内唯一support version或由多个uses共同引用的
`BoundaryVersionId`。后者只声明这些uses需要同一个logical boundary value；实际由一次DDR/NoC/load还是其它H choice实现仍归H。
`DirectNestedValue`只能引用execution result，不能把boundary伪装成nested compute。

`DemandFragmentId`不是旧node-pair edge：它由Q50.A的`DemandKey=(consumer, operand, destination shard)`加producer result、owner
Tile和exact ownership intersection稳定形成。这样一个consumer operand同时含local与remote pieces、同一producer result用于consumer
多个operands、insert/concat多path和multi-result时都不会被一条edge覆盖掉。exact set仍从A proof重算，不复制进candidate identity。

跨group且未被consumer-local replica关闭的fragment不出现在`localBindings`，由plan结构直接派生Q50.H boundary。group内每个nonempty
fragment必须恰有一个binding：

- independent stored：`RequiredPiece + TopLevel + StoredRegionValue`。producer完成独立traversal，semantic result version在region内
  保留到consumer use；是否最终占用一个或多个physical slots由G/I决定；
- consumer-driven：required piece或explicit replica选择`NestedUnder + DirectNestedValue`。consumer exact demand驱动producer work，
  actual value经direct SSA供给，不先形成DDR/SPM round-trip；
- explicit replica：`ReplicaFor(use)`显式增加pure producer execution。它可以top-level后供多个compatible uses共享，也可以nested于
  一个consumer；原root的mandatory work、observable result和其它uses仍独立闭合，不能用“找不到value就clone”隐式产生；
- cross-region boundary：没有local binding。H只能为该明确boundary选择local refetch、DDR或NoC等movement；H不再新增compute
  replica，也不改变D的execution count。

same Tile只说明local binding可能存在，不证明nesting或retention。same region下independent stored是普通candidate；同一root的required
pieces可以由top-level与nested instances共同实现，但必须满足mandatory-coverage verifier：全部required work的union exact、遗漏为错，
任何重叠必须归属显式`ReplicaFor`而不能伪装成required coverage。program output或effect-observable work始终保留mandatory coverage；
replica不能替它签字。partial contribution、merge和final result按Q50.A per-output requirement分别形成work/version，不再一律cut或用
node-wide merge flag。

`NestedExecutionRelation`是func-scoped/query-local analysis，从Q50.A relation DAG、producer result/consumer operand relation、
TilingInterface只读tile semantics、DPS init、speculatability与MemoryEffectOpInterface派生：

```text
NestedExecutionRelation
  demandFragment
  consumerDomain
  requiredProducerResultDomain
  producerWorkPreimage
  coverage: required piece | explicit replica
  status: Exact
        | RequiresTemporal(TraversalInstanceId[], IteratorId[])
        | Unsupported(reason)
```

planning不调用IR-building TilingInterface方法。`Exact`表示当前spatial/root boundary下已证明producer work与result tile all-and-only；
`RequiresTemporal`精确列出Q50.E必须赋值的traversal/iterator坐标，D state可保留该nested proposal但完整candidate在E关闭前不能accepted；
`Unsupported`只移除该nested choice，不移除independent stored、replica或boundary sibling。layout、physical retention和SPM feasibility
分别到G/I/F判断。

consumer-local replica的operand work仍由Q50.A唯一relation engine派生：

```text
deriveReplicaWork(fragment, producerWorkPreimage, exactDemandAnalysis):
  require producer is speculatable and has no non-local effects
  create ReplicaWorkId from semantic producer/result/use identity
  query exactDemandAnalysis.deriveOperands(producer, producerWorkPreimage)
  stop at selected structured boundaries or recursively selected replicas
  reject any cycle or effect boundary
  return typed replica work and its boundary fragments
```

这不是第二份operand walker；它复用A的normal-form relation graph与multi-seed propagation，且每次沿structured DAG严格逆拓扑，故
终止。相同replica只有在`(root, exact work, nesting parent, result coverage, effect scope)`完全一致时才能共享。temporal和representation
尚未关闭，不能塞进D的cache key：E/G若不能满足共享要求就exact-reject该组合，另一个显式split-replica plan仍在D domain中。

通用正确性条件是：每个mandatory root piece被required instances all-and-only覆盖；每个nonempty local use恰绑定一个定义先于use的
result version；每个boundary fragment恰留给H；每个replica显式计数并由pure/effect proof授权；nested relation逐点等于A relation
preimage；所有observable result都来自mandatory coverage。该证明只依赖SSA、indexing/effect interface和exact sets，与root类型、rank、
shape、Tile数或workload名称无关。

每个group必须在“至少一种local value/effect relation可行”的无向root graph中connected；relation既包括producer-consumer fragment，
也包括可共享的support/boundary version及不可跨越的effect order。若两组roots之间没有任何这类relation，拆成两个TileRegions可保持
相同H boundaries和J order，同时不会扩大lifetime/capacity；`wafer.tile.region`是scheduling/lifetime scope而不是独立device launch，
因此这种拆分是downstream-independent dominance。该证明不禁止独立components使用同一Tile或由Q50.J并发/串行调度。

fanin/fanout由`DemandFragmentId`而不是node pair表达。多个uses只有绑定同一`ExecutionResultVersionId`时才共享；不同work/nesting
scope必须是不同instances，不能让first consumer的actual cache决定其它uses。derived version requirement可重算，不进入IR attr；
representation差异由G为同一semantic version选择primary/derived physical versions，不反向制造D replica。

### D-2 专项调研：exact successor、proposal与complete-candidate emitter

connected region partition本身就是graph-restricted set partition。Graph Coalition Structure Generation的结果说明一般图上即使group
value只是edge-weight sum也为NP-complete；完整partition枚举在complete graph上至少有Bell number个输出。其DP可在典型additive
objective下做到`O(3^n)`求最优值，但不能把完整合法域变成多项式。SystemML的operator-fusion工作同样把DAG fusion作为exact
cost-based plan问题；XLA GPU current priority fusion则按estimated benefit使用priority queue，并明确把producer duplication计入cost。
这些实践对应本仓的结论是：exact successor必须lazy且承认指数规模；DP、priority fusion和最大group只能生成proposal，不能成为
legality filter或局部winner。

本项实现采用下列一手资料的共同边界：

| 资料 | 可复用事实 | Wafer采用/不采用 |
| --- | --- | --- |
| [MLIR `fuse_into_containing_op`](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.td) | transform从consumer实际slice反推producer tile；同一producer有多个uses时当前实现可能重复tile/clone | 只复用selected use的tile-and-fuse mechanics；share/split、replica count和允许融合的`DemandFragmentId`必须先由RegionPlan决定 |
| [MLIR One-Shot Bufferize](https://mlir.llvm.org/docs/Bufferization/) | 先在完整tensor SSA上分析alias/equivalence与RaW conflict，再统一rewrite | stored/direct语义先进入typed plan；不允许BodyEmitter根据当前users临时选择alias或allocation |
| [SystemML operator-fusion planning](https://www.vldb.org/pvldb/vol11/p1755-boehm.pdf) | 明确分离valid candidate exploration、cost-based selection与code generation，DAG plan空间为指数级 | exact domain、proposal排序和selected construction使用三个边界；不把cost或materializer失败当legality filter |
| [Coalition Structure Generation on Graphs](https://arxiv.org/abs/1410.6516) | 可行coalition是connected induced subgraph，完整connected partition仍需tree search/DP | 用canonical-parent reverse search逐个产生connected subsets/partitions，不扫描全部Bell partition再过滤 |
| [XLA GPU PriorityFusion](https://github.com/openxla/xla/blob/main/xla/backends/gpu/transforms/priority_fusion.h) | priority为estimated unfused/fused收益差，并显式计入producer duplication | 只作为可关闭proposal队列；不采用greedy结果作为domain、local winner或correctness repair |

#### Duplicate-free connected partition successor

每个Tile先构造`PotentialRegionConnectionGraph`：vertex是mandatory `RootRegionWorkId`；某个nonempty `DemandFragmentId`存在local
stored/direct-nested supply、两个roots可引用同一support/boundary version，或effect relation要求共同scope时才连无向edge。先拆
connected components；不同components分开求plan，因为把它们塞进同一region既不关闭boundary、也不产生local value/effect relation。

一个component内不扫描全部bitmask。对固定最小anchor，connected subset使用canonical parent形成reverse-search tree：

```text
parent(S, anchor):
  removable = {v in S - {anchor} | induced(S - {v}) is connected}
  return S - {largest semantic-id vertex in removable}

enumerateConnectedSets(anchor, allowed):
  visit S = {anchor}
  for each v adjacent to S, in semantic-id order:
    child = S union {v}
    if child subset allowed and parent(child, anchor) == S:
      recursively visit child

enumerateConnectedPartitions(remaining):
  if remaining is empty: yield empty partition
  anchor = smallest root in remaining
  for G in enumerateConnectedSets(anchor, remaining):
    for suffix in enumerateConnectedPartitions(remaining - G):
      yield canonical [G] + suffix
```

每个非singleton connected set至少有一个非anchor removable vertex，largest-removable parent唯一，所以每个connected subset和每个
connected partition恰访问一次。最终还要以selected local bindings、shared region versions和effect relations重验group connected；
potential graph只是安全superset，若一组把所有潜在relations都externalize则不会因潜在edge被错误保留。

#### Execution/use successor

固定partition后不做mixed-radix全积，而在canonical partial plan上每次解决第一个未决对象：

```text
successors(prefix):
  if an unresolved DemandFragment exists:
    emit typed children for every supported choice:
      ExternalBoundary
      StoredFromRequiredExecution
      DirectFromNestedRequiredExecution
      StoredFromExplicitReplica
      DirectFromNestedReplica
      StoredFromSharedBoundaryOrSupportVersion
    return

  if compatible uses have unresolved sharing:
    choose the anchor use and enumerate every compatible version group
    return

  derive exact CoverageAtoms for each mandatory RootWorkPiece
  if an unresolved atom exists:
    assign it to TopLevelRequired or one eligible NestedRequired instance
    return

  canonicalize instances, validate coverage/bindings/connectivity, yield plan
```

同一producer piece的所有required-nested preimages把mandatory set划成有限Boolean atoms；构造只用A的normal-form
intersection/difference/union并在任何昂贵set operation前检查work bound。每个atom恰分给一个required execution；同时执行同一atom的
其它instances标成explicit replicas。多个atoms若有相同`(root, purpose, placement, version-group)`就union成一个instance，exact-empty
instance删除。这样top-level remainder、部分nested coverage、overlapping fanout和explicit duplicate都有唯一normal form。

version sharing也用anchor partition而非first-use cache：只有D-1定义的semantic/effect/nesting requirement兼容的uses才可进入同一
version group；完整枚举包含shared和split两种plan。`RequiresTemporal` relation保留具体obligations，coverage closure标为deferred；
Q50.E赋值后重算同一atoms并exact accept/reject。work bound耗尽返回typed indeterminate，不能当作没有successor或exact rejection。

每次transition至少关闭一个finite use、sharing group或coverage atom；replica operand closure严格沿structured DAG逆拓扑，所以算法
终止。canonical key只含semantic IDs和typed choices，不含pointer、hash order、proposal ordinal或iterator stack。若component有`n`
个roots、`u`个use fragments、`a`个coverage atoms，partition输出最坏为`B_n`，use choices有指数因子，compatible version partitions
又可达Bell规模，coverage最多有`(1+c)^a`种owner assignment；这是问题本身的完整输出规模。lazy successor的额外内存只与
`n + u + a + recursion depth`和当前A relation normal form成正比，不预建全部plans。

#### Proposals与lower bounds

exact domain之外提供三类可关闭的proposal，均调用同一successor并返回普通domain member：

1. singleton+boundary、maximal connected+stored、consumer-chain nested、fanout shared、split replicas和replica-vs-boundary作为结构seed；
2. 小component用connected-subset DP产生k-best group proposals：

   ```text
   best(S):
     anchor = min(S)
     min over connected G containing anchor:
       groupLowerBound(G) + boundaryLowerBound(G, S-G) + best(S-G)
   ```

   `groupLowerBound`只计mandatory work、unavoidable replica work和已知effect precedence；`boundaryLowerBound`只用A logical bytes与
   topology最小possible transport，下游unknown项取0。该additive DP最坏`O(3^n)`，只排序partition proposals；
3. 大component使用XLA-style deterministic benefit queue生成一个或少量seed，benefit显式扣除duplicate work；随后用相同sound
   lower bound的best-first/branch-and-bound继续探索。queue/DP预算耗尽只结束proposal generation，不缩小exact domain。

禁止beam cap、固定top-k domain截断、greedy merge repair和per-Tile local winner。Q51可在全局budget下选择停止，但必须报告未穷尽
coverage，不能把proposal exhaustion伪装成“无合法plan”。

#### Complete-candidate group emitter

outer C transaction先把完整D--K assignment准备成`PreparedRegionGroup`；D不返回Module、不重查其它轴，也不自行选order：

```text
emitPreparedRegionGroup(region, prepared, rewriter):
  create region arguments/results from prepared boundary bindings
  for execution in Q50.J selected event/topological order:
    if execution is TopLevel:
      emit its required or replica work once
    if execution owns Nested children:
      install a resolver limited to those exact DemandFragmentIds
      emit each child through its selected consumer traversal
    publish every ExecutionResultVersionId exactly once
  bind every LocalUseBinding to its selected version
  verify mandatory coverage, replica count, dominance and boundary closure
  return region results and typed materialization relations
```

SCF/TilingInterface tile-and-fuse的control callback只接受plan中列出的`DemandFragmentId`；其它generated slice保留为prepared boundary或
明确失败，绝不递归融合“当前能看到的一切”。sharing直接查prepared `ExecutionResultVersionId`，不以operation pointer、block、
type/offset或first materialized tile恢复identity。DPS init rebasing、reshape/slice composition、windowed insert和observable result
复用迁移后的typed builders；无法表达selected relation时返回commit bug/typed unsupported，不隐式assemble full producer、改为stored、
切region或重选plan。atomicity仍由C的新CardModule subtree guard承担。

#### 本轮实现结果与下游边界

- production successor改为canonical-parent connected-subset reverse search，再递归组成connected partitions；它不再枚举全部Bell
  partition后过滤。2--6 root的chain、fanin、fanout、diamond、multiple sinks与disconnected graph继续由独立flat oracle核对集合与
  cardinality。singleton/maximal stored/direct及singleton stored/direct replica只作为checked proposals；Core先发proposal，再恢复同一
  raw cursor并跳过重复，proposal开关或顺序不改变exact set。
- `CompleteCandidatePlan`现在携带selected `RegionPlan`。ordinary structured root的candidate transaction先clone最近的source Module，
  为每个`ReplicaExecutionId`创建独立candidate-local structured node并精确rewire对应consumer operand，再把required/replica execution、
  shard与Temporal scope转换为common group materializer的typed descriptor。失败只销毁candidate clone，source保持不变。
- common group materializer按selected top-level execution先物化并缓存exact producer tile；consumer-nested execution沿实际consumer
  slice构造direct SSA。fanout split replica具有不同candidate-local node identity；shared required execution只物化一次。Q50.0在
  function-boundary bufferization之后保留同一current-IR relation epoch，再执行TileRegion-to-Instr与实际SPM规划。
- actual verifier从current `TileRegion` parent、operation emission与result-buffer relation重建每个`(Tile,node-set)`，逐组比较selected
  RegionPlan和candidate-local execution relation；不读取group attr、symbol名、operation ordinal、shape或打印文本。
- 本项只关闭normalized semantic-root层的Region/execution/use-binding与ordinary structured group construction。由Q50.S把一个attention
  semantic root展开成多action current IR后，action/execution关系及FA/FD production actualization仍由
  `attention-production-closure`关闭；partial-reduction merge的跨consumer nested traversal由Q50.E在完整parent-dependent temporal
  scope中关闭。D已经保留merge execution与attention coupled identity，不能在本项用假node、固定worker或单wave特例代替这些直接
  下游合同。

本轮fresh验证：增量完整build通过；`SelectedRegionMaterializationTest` 4/4覆盖rank-3 FP16的1024/1025 stored/direct required、
stored/direct replica、fanout split与atomic failure，并全部进入一次Q50.0 actual admission；`RegionDomainTest` 5/5、
`PlanningSessionTest` 6/6及donor `DataMovementTest`定向回归通过。排除尚未闭合且未纳入本项提交的future
`AttentionProductionClosureTest`后，host unit 908/908通过；configured lit、source/IR organization和4个public-link smoke全部通过。
future attention case当前在decode search的merge execution缺少EventGraph temporal coverage处失败，这一失败属于E/J/S后续门禁，未被
改写成D成功证据。

### Gate

- independent reference在每Tile 2--6 roots上平铺所有set partitions、过滤selected-binding connected groups，再枚举use supply、compatible
  sharing partition和coverage-atom owner；与production canonical plan集合逐项相同，覆盖chain、fanin、fanout、diamond、multiple
  sinks、disconnected components、mixed local/remote pieces、同producer多operands、multi-result、partial/merge和effect boundaries；
- reverse-search connected sets与bitmask+connectivity有界穷举oracle集合一致、无重复；complete graph观察Bell输出数，disconnected graph
  不组合components；输入root/use顺序、DenseMap insertion和指针扰动不改变canonical plan顺序；
- `NestedExecutionRelation`逐点oracle比较consumer domain、producer result和work preimage；temporal-dependent cases只产生精确
  obligations，并在Q50.E assignments下分别accepted/rejected，不被提前cut；A work limit返回indeterminate而非空domain；
- proposal DP在有界穷举graph与flat additive-bound optimum一致；关闭、反转或随机化seed顺序不改变exact set或Q51
  有界穷举optimum；默认
  statistics/logging关闭，无beam、fixed top-k、local winner或repair；
- independent/production candidate actual IR分别证明independent stored有两个traversals且无中间DDR，required nested只有consumer内producer tile和
  direct SSA，explicit replica的额外producer work可数，external boundary恰由H selected movement关闭；
- fanout shared plan只产生一个semantic version，split plan产生明确多个instances；不同work/nesting scope不错误共享。fanin/
  multi-producer support只消费A reconstruction，diamond mandatory coverage和emitted roots exact；
- maximal、中间cut、all-singleton和internal-edge externalization都能actualize；较小cut或replica方案能否成为global winner归Q51跨轴
  gate。D实现期不运行重型LLaMA search。

### Current/donor能力迁移

| Current / donor能力 | 终态owner | 保留或修正的witness | 退役条件 |
| --- | --- | --- | --- |
| current restricted-growth partition successor | Q50.D connected-set reverse search + partial-plan successor | chain/fanin/fanout/diamond/reference equality、disconnected不扫跨component Bell | execution/use/coverage坐标就位后删除旧labels/fusable矩阵 |
| current greedy `getFusionOrientedAssignment` | Q50.D DP/benefit-queue proposal | maximal、中间cut、fanout-share和replica-vs-boundary proposal、order metamorphic | 不再作repair/local winner，关闭proposal不改变exact set |
| current local-only exact-demand fusable test | Q50.A fragments + D `NestedExecutionRelation`/stored/boundary domains | mixed local/remote、empty、partial、effect、deferred正负例 | Q50.D不重跑per-edge query或一律forbid partial/nonlocal |
| current coupled apply同时消费temporal/layout/implementation/movement并造Module | outer Q50.C winner transaction + D group emitter | complete typed assignment、atomic actual IR | D API只query/emit selected group，无local defaults或Module return |
| old connection topology与consumer-driven producer demand | D use fragments + nested relation | held-out transpose/reshape、fanin/fanout fragment/version count | operation ordinal、tile-size/action recipe字段删除 |
| old Coupled/SeparatedResident/five-state choices | D execution placement + use binding；H只拥有physical boundary movement | independent stored、required nested、explicit replica、external boundary actual witnesses | D/H不再共享recompute或retention enum，DDR/CrossRegion/SelectiveSpill只在H |
| old shared separated-version compatibility | D semantic `ExecutionResultVersionId`；physical derived versions归G | mixed fanout shared/split、different work/nesting tests | string signature、representation-in-D与tile-size proxy零残留 |
| old CompleteTraversal/BidirectionalTiling | C/D/E selected TilingInterface mechanics | multiple roots/sinks、exact tail、DPS init、observable producer | undefined rank/provider/pass入口删除，能力有current caller/test |
| current ProducerTileFusion slice bubbling、DPS-init rebasing、tile memo | D selected-use emitter + A tensor relation | unit/nonunit reshape、DPS init、windowed insert、shared/split versions | implicit full-producer assembly、recursive fuse-all和pointer-based cache删除 |

## Q50.E：Complete Temporal Tiling

### Pipeline contract

Pipeline position:
- Upstream IR / input:
  Q50.S normalized immutable TensorProgram/fixed semantic facts、Q50.B closed spatial assignment、Q50.A demand和Q50.D `RegionPlan`；
  D已经确定mandatory/replica execution instances、top-level/nested placement和use bindings，但尚未出现temporal wave loops。
- Current stage responsibility:
  为每个top-level work piece和每种nested invocation class产生覆盖全部iterator的positive tile-size vector及实际多wave iterator的
  loop nesting order；nested producer的requested work由parent traversal与exact relation派生，但其内部iterator仍拥有明确temporal
  variables。domain以integer intervals/constraints惰性表示全部合法sizes/orders，breakpoints只控制proposal和bound。complete-candidate emitter
  用共同TilingInterface mechanics构造compact loops/tails。
- Output IR / files:
  不写IR或文件。Q51 complete state保存typed `TemporalPlan`；partial state可保存尚未定点的integer intervals。derived wave work、
  producer preimages、exact tile domain和cost按需重算。candidate actual IR只保留真实loops/tails，不写score、capacity history或recipe attr。
- Downstream consumer:
  Q50.F按selected spatial/group/temporal scope计算pure lower bound/deferred facts；Q50.G–K继续物化representation、movement、buffer和
  schedule，complete candidate最终只进入一次Q50.0。
- User-level driver / named pipeline:
  只由public search session静态组合；baseline deterministic fallback与search domain复用同一local-extent、breakpoint和actual leaf
  owner，但baseline只沿预定义单调顺序推进，不进入domain enumeration。
- Explicit non-goals:
  不选择layout、movement、retention、buffer、worker、schedule、cost或winner；不从allocator failure私下`tile/2`；不保存fixed seed、
  maximum-fit或feedback history；不把target-preferred sizes或breakpoints当legality；不运行packer/probe，不clone/replay actual IR。
- Done criteria:
  per-Tile required/replica execution、nested producer、parallel、single/multi-reduction、multi-piece、remainder和scalar的lazy domain与
  independent reference一致；每个可tile iterator的`1..extent`size可达，inactive iterator不制造order state，允许的active order完整；
  baseline复用同一domain query但只走deterministic greedy；complete candidate在singleton/group actual region形成compact loops/tails；旧
  per-node/global-max Cartesian、fixed seed和allocation-feedback owner在能力迁移后删除。

### Q50.E work-item分界

Q50.E由两个work items完成：

- `canonical-temporal-plan`只为canonical `RegionPlan`中的每个required root execution建立一个top-level traversal scope，tile-size
  vector等于该execution piece自己的完整local extents，因此是single wave且`waveLoopOrder`为空。merge execution没有root iterator，
  不伪造temporal scope。该点不读取target preference、capacity或allocator结果。
- `temporal-domain`在`region-execution-domain`后补齐每个scope的`1..L_i`interval successors、active-order全部linear extensions、
  nested main/tail/halo invocation classes、proposal和Core consumer，并迁移/退役旧node-wide domain。canonical full-extent点只是
  domain member和baseline初始坐标，不是default winner或legality shortcut。

当前`canonical-temporal-plan`覆盖矩阵如下。shape用于覆盖per-Tile exact extents与remainder，不进入scope identity或tile-size分支；
rank-zero是明确的结构合同例外。

| 覆盖类 | 代表输入 | canonical TemporalPlan exact断言 | 直接下游witness |
| --- | --- | --- | --- |
| aligned/ragged all-Tile | rank>=3、主要维度1024/1025、all-16 Tile | 每个required root execution恰有一个scope；size vector逐轴等于自己的local interval size，1025 remainder Tiles不共享ceil maximum；order为空 | canonical-representation-plan按scope读取真实local shape；planning前后IR不变 |
| multi-axis与input-order扰动 | 两个以上spatial axes、不同Tile/root work顺序 | canonical scope ID由required execution与top-level piece组成，逐字段stable sort；同descriptor可重算但不合并不同Tile assignment | temporal-domain在每个scope独立展开siblings |
| ordinary reduction/contribution | single/multi-reduction、整除/非整除spatial contribution | root execution保持完整local reduction extents；spatial contribution不新增temporal owner；merge execution无scope | F/E后续能区分local sequential waves与A/B spatial merge |
| FD coupled state | rank-5/6、K2为1024/1031或multi-K2 | 每个FD contribution root execution有一个full-local K2 scope；Maximum/Sum/Accumulator merge仍无三个伪scope | attention-work-projection从同一execution/merge identity派生actions |
| rank-zero与merge-only | rank-zero root、只有merge的Tile、empty Tile | rank-zero产生唯一empty size/order scope；merge-only与empty Tile均不产生scope，但前者仍由RegionPlan保留execution | serialized/storage/schedule阶段不会用假iterator表示merge |
| typed failure | duplicate/missing execution、scope引用错误work/shard、非正local extent | compiler-contract failure且无partial plan，不fallback到node-wide extent或size=1 | 修正输入可重新query，source IR byte-identical |

`temporal-domain`施工前覆盖矩阵如下。小extent只用于逐点穷举reference；production witness仍使用rank至少为3、主要维度至少为1024的
static workload，并成对覆盖整除与非整除边界。

| 覆盖类 | 代表输入 | exact断言 | Core / 直接下游witness |
| --- | --- | --- | --- |
| 完整size域 | 独立oracle的`2x3`、production rank-3 `2x1024x128`与`2x1025x128` | 每个`Tileable`轴恰有`1..L`且无重复；`FullExtentOnly`轴只有`L`；full-local canonical点是domain member | `RegionState -> TemporalState`保存同一typed plan，不写IR、不读target/SPM |
| per-Tile remainder与assignment独立性 | 同一root在两个Tile上的local extent分别为3和2；另有16-Tile 1025 ragged输入 | 两个scope分别有3和2个一维点，joint plan为6种；输入顺序与descriptor memo不改变plan key集合 | temporal改变只失效本scope及其child，不把不同Tile压成一个node-wide vector |
| active-order | 两个以上active轴、无precedence/chain/diamond DAG；single/multi-reduction作为普通iterator facts | Kahn successor与独立flat permutation+edge filter逐key相同；one-wave轴不进order；cycle typed失败 | query与后续loop builder消费同一order，无late second legality gate |
| exact wave/tail | rank-3的1024整除、1025/1031非整除，多轴与nonzero offset piece | 每轴wave两两不交且all-and-only覆盖原interval；末wave使用`min(size, remaining)`；Cartesian piece不稠密化 | prepared scope可直接交给唯一temporal loop builder，不需要post-hoc retile |
| top-level execution类别 | required root、top-level pure replica、rank-zero、merge-only | required/replica使用不同typed execution ID；rank-zero唯一empty plan；merge-only无scope | canonical consumers显式只接收top-level required scope，不能从ordinal或operation pointer恢复 |
| nested invocation合同 | direct required/replica的main/tail/halo typed classes、shared与split execution version | class identity来自parent、exact requested/producer rectangles和relation class；同class共享一个plan，split version保留独立plan；缺失composed region-use relation typed unsupported | child scope依赖parent concrete plan；parent变化重建classes，不把full producer或bounding box当fallback |
| selected loop construction | top-level/nested/coupled scopes，rank-3/6的1024及1025/1031 | actual loops逐scope使用selected size/order，main/tail/halo coverage all-and-only；replica独立可数，coupled intermediate为tile-sized | Temporal verifier从SCF/SSA/result-carried state重建plan；G/I/J直接消费actual scope/occurrence关系 |
| typed failure与资源隔离 | duplicate scope/execution、missing work/shard、rank不一致、nonpositive/dynamic extent、precedence cycle、非法plan/order | 无partial `TemporalPlan`；unsupported/indeterminate/compiler bug可区分；source IR byte-identical | 改变target capacity、allocator或SPM结果不改变temporal domain及successor顺序 |

### E-1 专项调研：temporal scope、完整size域与order合同

MLIR `TilingInterface`明确把“生成loop结构”与“operation如何生成一个iteration tile”分离，并说明tile-and-fuse只提供mechanism、
不判断profitability；`SCFTilingOptions`把tile sizes和loop interchange作为显式caller choice。`generateResultTileValue`还明确提示fusion
可能造成recomputation，责任在caller。对应本仓，E必须完整描述size/order，D负责execution/replica，actual builder不得按target
preferred size或当前slice临时选择。pinned MLIR的SCF tile-and-fuse也要求caller保证给定sizes下fusion有效，不能把一次builder失败
当成temporal domain查询。

current代码已经有可迁移的机制：`TemporalWaveLoop`生成prologue、steady loop和exact tail；`TemporalNodeDomain`也枚举positive sizes及
active-dimension permutations。但current identity是node-wide，`deriveLocalIteratorExtents`对空间remainder取所有Tiles共同的ceil
maximum，Card apply又要求同一node跨Tile使用同一vector；因此extent为5、分2个Tiles时把3同时当成两个Tile的domain，丢失Tile-local
`3`与`2`的独立选择。更严重的是query接受全部active permutations，actual `TemporalRegionTraversal`再通过另一套reduction check拒绝
其中一部分，形成query/apply合同错位。两者都必须随本任务原位替换。

D关闭后先从每个`ExecutionInstancePlan`的exact work normal form派生temporal scopes：

```text
TemporalScope
  id: TraversalScopeId
  execution: ExecutionInstanceId
  invocationClass: TopLevelWorkPieceId | NestedInvocationClassId
  iterationOffsets: ExactIntegerVector       // derived, not candidate state
  iterationExtents: PositiveIntegerVector    // derived, not candidate state
  iteratorCapabilities: Tileable | FullExtentOnly

TemporalPlan
  scopes: TemporalScopePlan[]

TemporalScopePlan
  scope: TraversalScopeId
  iteratorTileSizes: PositiveInteger[]
  waveLoopOrder: IteratorId[]
```

top-level execution的每个maximal rectangular normal-form work piece形成一个scope；multi-piece work不被bounding box稠密化。nested
execution在parent temporal point关闭后，经D `NestedExecutionRelation`得到实际producer work；把absolute parent-wave offset正规化为
class-local zero后，extent/relation/effect scope相同的translation-equivalent invocations合并为一个`NestedInvocationClassId`。main、
各tail及halo/boundary的不同extent/relation保留不同classes，不按wave ordinal逐个复制state；absolute offset由actual parent leaf SSA提供。
child scope依赖parent scope，parent变化使全部derived child classes与plans失效重建。

与旧设计不同，nested producer不是“完全没有vector”：parent决定它被请求的result/work domain，但producer内部仍可能有reduction、
window或其它迭代维，也允许把requested work继续分成更小temporal waves。每个derived invocation class因此使用自己的完整vector，
child loops实际位于parent leaf内部。多个consumers共享同一D execution version时必须共享一个兼容E plan；不兼容时该组合exact-reject，
由D的split-version sibling表达多个instances，materializer不能临时复制。

candidate只保存`TraversalScopeId + sizes + order`，不复制offsets/extents、wave work、producer preimage或actual loops。ID由Tile、region、
execution version和canonical exact invocation class组成，不含Operation pointer、materialized loop、hash或wave ordinal。

对一个nonempty scope的local extent vector `L`，temporal legality为：

- standard/semantic tiling interface声明`Tileable`的iterator，所有`1 <= t_i <= L_i`均是合法点；每wave size为
  `min(t_i, L_i - offset)`，不要求整除，不因native geometry、alignment、transaction、SPM或preferred size删点；
- interface只支持完整遍历的iterator只有singleton `{L_i}`。该capability必须来自typed interface/current op semantics，不能通过
  shape、名称或actual materialization probe恢复；
- exact-empty work在C/D已消失，不生成scope；rank-zero iteration domain生成唯一空vector/空order plan；静态physical planning要求
  scope extents在此边界已关闭，未关闭的dynamic extent返回typed deferred/unsupported而不是伪造有限上界；
- parallel和同Tile sequential reduction iterator都可使用任意positive size。reduction waves通过一个loop-carried result state顺序更新，
  DPS init在首wave之前恰消费一次；空间partial contribution仍只改变A/B owner/merge关系，不缩小本地temporal size域；
- nested relation、multi-result和result tile必须在每个concrete point重算exact coverage；一个组合不兼容只拒绝该D+E组合，不增加
  fallback、隐式stored execution或full-producer assembly。

`waveLoopOrder`只排列实际`ceilDiv(L_i,t_i) > 1`的active iterators；one-wave dimensions不进入identity。每个scope从SSA
loop-carried state、control/effect relation、parent/child nesting和op interface明确的structural restriction建立
`TemporalPrecedenceDAG`，domain是它的全部linear extensions。target preference不产生precedence；既有op dtype/运算语义原样消费，
本阶段不增加额外order gate。parent loops天然位于child loops之外，这种cross-scope nesting不重复写入child `waveLoopOrder`。

不同Tile上的同一semantic root始终是不同variables，remainder Tile直接使用自己的exact extents。extensionally相同的
`TemporalDomainDescriptor=(interface capability, exact extents, precedence DAG, parent relation class)`可在同一immutable session共享
query/enumerator memo，从而避免16 Tiles重复分析；它只复用**域描述和计算**，不强迫各Tile选择同一点。只有Q51对完整program、topology
automorphism及所有下游choices给出global symmetry proof时才能quotient candidate states，E本身不按“看起来一样”合并assignment。

Q50.E不把所有integer points预建为Cartesian vectors。每个`Tileable` variable初始为closed interval`[1,L_i]`，
`FullExtentOnly`为singleton；Q51 partial state逐个定点或split interval，complete state才含concrete vector。target issued geometry、
wave/tail、transaction和A/D relation shape changes只能产生proposal breakpoints；interval内其它integer siblings始终可达，除非
Q51对完整依赖坐标给出sound structural equivalence或performance-objective bound proof。SPM capacity或任何footprint/working-set估算
不得产生temporal breakpoint、priority或pruning。

### E-2 专项调研：interval successor、order枚举、proposal与actual loops

Halide autoscheduler和Ansor都把fusion/compute placement等结构选择与concrete tile annotations分层，并用tree/beam、cost model或
evolutionary search处理巨大schedule space；它们也明确承认只探索其定义的schedule子集。该经验适合本仓的proposal排序和hierarchical
state，却不能证明Wafer完整temporal合同。E的exact successor因此覆盖全部positive integer points，学习/beam/divisor/native points
只能作为可关闭的proposal provider。

本项采用的一手资料边界如下：

| 资料 | 可复用事实 | Wafer采用/不采用 |
| --- | --- | --- |
| [MLIR SCF `TileUsingInterface.cpp`](https://mlir.llvm.org/doxygen/TileUsingInterface_8cpp_source.html) | caller显式提供tile sizes/interchange，builder通过`TilingInterface`生成loop与tiled body；producer fusion是独立步骤 | 复用TilingInterface/SCF construction mechanics；size/order合法域、nested class与share/replica仍由D/E typed plan决定 |
| [Halide learned autoscheduler](https://halide-lang.org/papers/halide_autoscheduler_2019.pdf) | hierarchical loop nest同时表示compute/store placement与nested tiling，beam只探索候选子集 | 只借鉴层次state与proposal组织；不采用beam作为exact domain或legality gate |
| [Ansor](https://www.usenix.org/system/files/osdi20-zheng.pdf) | 从hierarchical representation采样完整program，再用evolutionary search/cost model排序 | proposal可按完整program排序；不让sampling、learned cost或hardware-preferred点删除positive integer siblings |

仓库pinned MLIR确认当前可用的是`TilingInterface::getTiledImplementation`、
`getIterationDomainTileFromOperandTile`与SCF tile/fuse实现；current nested emitter只调用这些实际存在的接口，不依赖较新upstream API。

#### Dependency-aware interval successor

先为全部top-level work pieces建立scope；nested classes只有在parent size/order关闭后才派生。partial state每次解决stable
scope-dependency order中的第一个未决对象：

```text
temporalSuccessors(prefix):
  if a parent plan just became concrete:
    derive and canonicalize all nested invocation classes
    add their scope descriptors in stable semantic order

  if an unresolved size interval I=[lo,hi] exists:
    p = highest-priority unused proposal in I
    if no p: p = lo + (hi-lo)/2
    yield singleton [p,p] first
    if p < hi: yield [p+1,hi]
    if lo < p: yield [lo,p-1]
    return

  if a concrete scope has an incomplete wave order:
    available = unchosen active iterators whose predecessors are chosen
    for iterator in stable available order:
      yield prefix with iterator appended
    return

  yield complete TemporalPlan
```

三个interval children两两不交且union恰为parent，proposal直接成为first singleton；没有proposal时midpoint同样严格缩小。interval state
不是legal candidate，只是Q51尚未关闭的typed coordinate；F/Q51只能对整段给sound lower bound，不能抽一个代表点代签整段。size全部
singleton后active set才固定，避免one-wave维度产生重复order states。

order successor是Kahn式linear-extension枚举：每一步只选当前零入度iterator。任何合法linear extension的每个prefix都有唯一选择路径，
因此全部且只枚举`TemporalPrecedenceDAG`的linear extensions；DAG有cycle是typed contract error，不是“使用source order”fallback。

nested class派生不逐wave建立state：

```text
deriveNestedInvocationClasses(parentScopePlan, nestedExecution):
  derive finite parent main/tail work classes from exact wave partition
  map each class through NestedExecutionRelation
  normalize producer work to exact rectangular pieces
  normalize absolute offsets to class-local coordinates
  union translation-equivalent (extent, relation, effect-scope) classes
  return stable classes and child scope dependencies
```

halo、stride、boundary clipping或parent tails形成不同exact classes；仅wave ordinal不同而work/relation相同的invocations复用一个class。
normal-form work limit在mapping/union前检查，耗尽返回indeterminate，不把child省略。D execution nesting无cycle，故递归scope discovery
终止。

完整性按scope DAG归纳：top-level每个size point/order由interval+linear-extension successor唯一到达；固定parent plan决定唯一finite child
class set，每个child又完整枚举；因此所有可表达hierarchical temporal plans都可达。对一维`[o,o+L)`和size`t`，waves为
`[o+k*t, o+min((k+1)*t,L))`，两两不交且union为原区间；Cartesian product覆盖一个rectangular piece，多piece scopes按A/D exact
normal form保持all-and-only union。loop-carried result state让每个source iteration执行一次、DPS init只在整个scope第一wave前进入；
nested relation的exact image/preimage再把child coverage归纳到parent use。证明不依赖workload名、shape常数或target preference。

#### Proposal algorithms

proposal point可来自wave-count/tail等价类、divisors、native issued geometry、alignment/transaction边界和A/D relation shape changes；
provider只返回普通integer points及可解释priority，不返回allowed-set。实际SPM rejection由外层controller产生下一candidate，不进入E的
proposal provider。
移除任一provider后midpoint successor仍到达全部points。

跨scope proposal使用dependency factor graph：parent/nested relation和shared D version形成compatibility factors，known issued work形成
lower-bound factors。chain/tree在已产生的proposal points上做k-best DP，一般低treewidth图可做variable elimination，其余由Q51
best-first interval search。若每scope当前有`K`个proposal points，chain为`O(NK^2)`、treewidth`w`为`O(NK^(w+1))`；这里`K`只是
本轮proposal work量，不是domain cap。预算耗尽停止proposal生成并报告coverage，exact interval children仍存在；无fixed top-k、beam
legality、per-scope winner或默认统计。

#### Baseline deterministic greedy

baseline复用相同`TemporalScope`、positive-size successor、exact-tail builder和domain-descriptor memo，但不构造E/Q51 frontier：

```text
chooseBaselineTemporalPlan(scopes):
  start every scope at full exact extent and canonical structural order
  while actual Q50.0 returns an attributed SPM capacity rejection:
    obtain only causally implicated roots from actual demand relations
    choose the deterministic next smaller temporal candidate
    rebuild and actualize that candidate once
  return first actual accepted candidate
  if all implicated axes are one: return typed capacity failure
```

这是`none`的单调actual-feedback policy：不回溯、不保存alternative、不调用search。每个不同candidate都有且仅有一份实际CardModule；
rejection后销毁该transaction，accepted candidate原样保留。Indeterminate或ResourceExhausted直接失败，不生成下一candidate。
任何wave-class relief、bytes比例或footprint都不参与选择或合法性。

#### Complete-candidate actual emission

complete winner先为每个scope建立`PreparedTemporalScope`，核对concrete sizes/order、parent class和D use/version relation；D group
emitter随后通过`RewriterBase`调用唯一loop builder：

```text
emitTemporalScope(prepared, resultState, rewriter, emitLeaf):
  carry resultState through active loops in selected waveLoopOrder
  for each dimension emit compact prologue / steady scf.for / exact tail
  at each Cartesian leaf call selected C/D execution builder once
  resolve nested execution with the prepared child invocation class
  return the carried result state
```

multi-result state一起loop-carry；DPS init在scope entry绑定一次，reduction/update waves读取前一state。top-level multi-piece scopes按stable
piece order各调用一次；nested child loops位于parent leaf且只使用D selected-use resolver。current `TemporalWaveLoop`的compact
prologue/steady/tail mechanics可迁移，current recursive fuse-all和operation-pointer temporal lookup不能保留。emit失败由C outer
transaction回滚，不修改tile size/order、不post-hoc retile、不返回planner尝试另一point。

若scope `s`的rank为`r_s`、extent为`L_si`，concrete size vectors为`product_i L_si`，每point的orders为precedence DAG linear
extensions；nested class数量再按actual exact relation相乘，组合爆炸不可消除。partial-state额外空间与当前scope DAG、intervals和order
prefix线性，query memo按extensional descriptor共享；性能优化不能改变leaf集合。

### Gate

- independent有界穷举reference对每个scope平铺`1..L_i`Cartesian sizes并枚举precedence DAG全部topological orders；与interval/order
  successor leaves逐key相同。覆盖parallel、single/multi-reduction、rank-zero、multi-piece、tail和nested child internal iterators；
- spatial extent 5分成Tile-local 3/2两个scope：分别含3与2个一维points且joint assignment有6种，不再以shared maximum强迫相同vector；
  相同descriptor memo只构造一次，关闭memo或改变Tile输入顺序不改变per-Tile plan集合；
- nested main/tail/halo classes与逐parent-wave oracle extensionally相同但class数只随distinct relation classes增长；parent point变化正确
  invalidate child，work limit为indeterminate；shared D version要求一个compatible child plan，split versions保留不同plans；
- size/coverage property逐点验证一维partition、rectangular Cartesian、多piece union、multi-result state和init exactly once；空间partial
  contribution的local waves与A per-output merge结合正确，不另设temporal reduction owner；
- Kahn successor与flat permutation+precedence filter集合一致、无duplicate；current query-accept/apply-reject order差异零残留；
- proposal DP在tiny chain/tree与flat proposal-point minimum一致；删除全部providers仍由midpoint分支到达每个integer，provider/order
  扰动不改变exact leaves或Q51有界穷举optimum；默认logging/statistics关闭；
- Q49.P与E共享scope derivation、legal next-size和actual loop builder；overfull→fit、多轴/tail/all-ones failure稳定，baseline路径
  无search frontier，每个failed candidate实际执行一次Q50.0且accepted candidate不重建；
- independent/production candidate actual IR检查selected nesting、compact main/tail、nested producer child loops、multi-reduction state和无post-hoc retile；
  每个candidate一份IR，无同coordinate probe/replay；
- source gate删除`TemporalNodeAssignment/Domain`、node-wide `StructuredOpTemporalTile`、shared-ceil extent、fixed seed、preferred-size
  legality、allocation feedback和operation-pointer lookup；旧witness全部迁入new scope owner后才删source/tests。

### Current/donor能力迁移

| Current / donor能力 | 终态owner | 保留或修正的witness | 退役条件 |
| --- | --- | --- | --- |
| current every-positive-size/permutation successor | E per-scope interval leaves/Kahn order successor | 2x3、multi-axis、per-Tile remainder与nested-child reference equality | 删除node-wide domain且不预建Cartesian vectors |
| current one-wave order canonicalization | E active iterator key | scalar/one-wave无重复、active order完整 | 同一逻辑保留在new type |
| actual SPM feedback | Q49.P controller + Q50.0 | full→smaller candidate、aligned/ragged tail、typed demand owner | 无owner demand失败；无estimate/alternate allocator |
| current baseline `setCardBaselineTemporalTiles` | shared per-scope query + Q49.P controller | full initial candidate、per-Tile remainder、output correctness | 不读取target capacity或footprint |
| current TemporalWave/Region/PartialReduction traversal | C/D/E prepared actual emitter | parallel/reduction/multi-axis/tail/init/nested child | apply不更新既有CardModule、不选tile、不递归fuse-all |
| current actual-only reduction order check | E structural order + same emitter verifier | query/apply parity、multi-reduction all structural orders | 删除独立reassociation/numeric-policy legality helper；E只保留source arithmetic op与dtype |
| old rank/connection tile sizes、preferred points | E proposal facts | independent/nested execution intent由D/E表达 | action recipe、rank provider、local cap删除 |
| old allocation feedback bags/binary endpoint | Q49.P/Q51 actual-result controller | current relation、actual conflict demand、siblings preserved | 无feedback history、accepted retile或shortlist |

### `temporal-domain`当前已实现子集与重新打开门禁

- `TemporalScopePlan`原位使用`TraversalScopeId`，required与replica execution、top-level piece与nested invocation class不再共用一个
  node-wide identity。plan只保存scope、concrete sizes和active order；offset/extent、precedence、wave rectangles和exact relation仍由
  immutable domain facts重算。
- `TemporalDomain`不预建size Cartesian vectors：complete successor覆盖每个`Tileable`轴的`1..L`和`FullExtentOnly` singleton，Kahn式
  successor只枚举precedence DAG的linear extensions。`splitTemporalSizeInterval`把proposal或midpoint分成互斥singleton/above/below，
  三者union恰为parent；proposal不成为allowed-set。
- top-level required/replica scopes直接使用各自exact local intervals。direct nested scope在parent concrete point后，经consumer operand
  `IndexRelation`、selected fragment owner domain和producer result preimage生成main/tail/halo producer rectangles；parent plan变化重建
  descendants。explicit reconstruction没有D composed relation时typed unsupported，relation work limit为indeterminate，均不使用full-producer
  或bounding-box fallback。
- `TemporalState`与`TemporalContinuation`已经接入production planning session；public search取得first complete temporal point后返回
  missing `partial-feasibility`，`candidate_actualizations=0`。Q49.P baseline的full-local初始point复用同一domain，不进入search frontier。
- old `TemporalNodeDomain`仍只被未迁完的旧Unified/Coupled/representation/movement donor调用，不进入public search或new state；按能力迁移
  规则留到G/H及selected emitter接管其实际lowering witnesses后删除，不能提前丢弃donor能力。

fresh证据：`TemporalDomainTest` 11/11，覆盖2x3 independent oracle、1025 interval完整性、precedence diamond、per-Tile 3/2 joint 6点、
rank-zero、rank-6 coupled vector、1024/1025/1031 exact tails、top-level replica、1025 parent-dependent convolution main/tail/halo和work-limit；
`PlanningSessionTest` 6/6并穷尽tiny temporal 4点；temporal/Core/baseline直接集合24/24（含shared nested use与actual-feedback refined plan
domain membership）及balanced actual CardExecutable通过；ordinary host unit 842/842（6个独立heavy baseline cases不在本项重复）；core lit 227/227，
Tools/Runtime lit 35 passed、4 configured unsupported；compiler
public link、source organization和diff检查通过。下一项只签发A–E结构ready/missing，不引入任何资源估算。

“小tile使fusion/buffering成为winner”及任何需要actual layout/buffer/resource cost的比较统一放在Q51 closure。

本轮closure把上述重新打开门禁收口为同一条actual链：

- nested invocation不再按absolute wave ordinal复制state。domain把absolute offsets正规化为class-local zero，以
  `(execution,parent,use relation,requested extent,producer extent)`合并translation-equivalent steady waves；main、tail、halo的不同
  extent/relation仍为不同class。absolute offset只来自actual parent leaf的SSA slice。
- selected group descriptor分别携带top-level与nested temporal class。nested producer resolver按candidate-local producer/parent node、
  result、operand及actual requested extent选择唯一class，再把该class的size/order交给同一个TilingInterface/SCF traversal；required与
  replica使用同一实现，缺失/重复class在创建IR前失败。
- actual verifier从current outer TileRegion、emission/result relation及`scf.for` bounds/step/nesting核对selected top-level/nested
  compact loops；修改保留的query-local descriptor会被拒绝。1024/1025双轴required与replica case均生成parent/child steady loops、ragged
  tail并各自通过一次Q50.0 actual SPM gate。
- actual SPM capacity feedback helper移到TemporalDomain owner，输入只含actual causal roots；它只产生更小的top-level domain prefix，
  随后由同一domain重建nested classes。baseline与search共享该机制，不读取bytes、footprint、capacity estimate或packing proxy。
- 删除了独立`ReductionSemantics` reassociation/numeric-policy helper及temporal/partial-reduction调用。E只选择structural iterator
  size/order并保留source arithmetic op与dtype；typed Tile reduce没有等价target kind时仍由目标表示边界明确拒绝，不反向缩小E domain。
- function-boundary bufferization在TileRegion listener前先把attribution relation重绑到唯一storage root，cleanup后只删除已消失的dead
  SSA entry；随后TileRegion-to-Instr仍由唯一listener维护current relation。

本轮fresh验证：完整build通过；`TemporalDomainTest` 11/11、`SelectedRegionMaterializationTest` 5/5、
`BidirectionalTilingTest` 5/5及temporal/baseline/search/full-feasibility定向回归通过。排除尚未闭合且不属于本项提交的future
`AttentionProductionClosureTest`后，host unit 909/909通过；configured lit、source/IR organization与4个public-link smoke全部通过。
future attention case仍在EventGraph错误要求merge execution具有temporal scope处失败；merge按E合同没有假iterator scope，该问题保留给
event-resource/attention production owner，未被本项伪装为Temporal成功。

## Q50.F Owner Contract：Structural Readiness and Actual Resource Admission

Q50.F不在IR外预测SPM、DDR或transport合法性。它分成两个线性work item：`partial-feasibility`只检查当前A–E prefix是否结构完整、
哪些坐标尚未产生；`full-feasibility`在B–K、I、J全部关闭后具体化完整candidate并调用唯一Q50.0 actual gate。

### Partial structural readiness

```text
Pipeline position:
- Upstream IR / input:
  immutable TensorProgram、closed B–E assignments及其typed producer facts；G–K/I/J可能尚未产生。
- Current stage responsibility:
  验证closed A--E坐标的identity、rank、scope、dependency和producer/consumer完整性；只返回
  `ReadyForNextCoordinate(Representation)`或`CompilerBug`。更早的missing coordinate及其Unsupported/Indeterminate由对应typed
  state/continuation owner分类，本query不重复解释。
- Output IR / files:
  query-local structural result；不写IR，不计算bytes，不产生resource rejection。
- Downstream consumer:
  layout-domain及后续Core transitions。
- User-level driver / named pipeline:
  public search planning session内部调用；baseline不需要该partial query。
- Explicit non-goals:
  不计算footprint、lower/upper bound、residency estimate、synthetic demand、packing或capacity；不prune任何资源候选。
- Done criteria:
  earlier missing coordinate不会调用本query，malformed closed coordinate返回CompilerBug；任意target memory capacity变化不改变
  partial structural result。
```

partial result没有resource lower-bound、footprint rejection或non-binding estimate字段。尚无actual candidate时资源状态就是unknown，
不能把逻辑tensor payload、单buffer大小或任何保守值转换成rejection。

本项采用的一手实现边界如下：

| 资料 | 可复用事实 | Wafer采用/不采用 |
| --- | --- | --- |
| [XLA `HloPassPipeline`](https://github.com/openxla/xla/blob/main/xla/hlo/pass/hlo_pass_pipeline.cc) | invariant checker在pass边界运行且必须自身不改图；失败保留原pass上下文 | readiness是pure invariant query，重复调用不修改IR/domain；不复制XLA hash作为Wafer语义identity |
| [MLIR Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/) | full conversion与partial conversion对legal/illegal/unknown operation有明确不同合同 | closed prefix只验证自己已经拥有的typed coordinate；不把尚未进入本stage的坐标用unknown-op式规则默认为合法 |
| [LLVM analysis invalidation](https://llvm.org/docs/NewPassManager.html#implementing-analysis-invalidation) | analysis默认随IR mutation失效，只有明确preserved才复用 | readiness只借用immutable TemporalDomain/Plan；不持久化epoch、resource cache或manual preservation side table |

`partial-feasibility`施工前覆盖矩阵如下。该query只消费closed typed prefix/domain membership；shape用于覆盖真实prefix结构，不进入
readiness分支或资源判断。

| 覆盖类 | 代表输入 | exact断言 | Core / 下游witness |
| --- | --- | --- | --- |
| closed A--E prefix | rank>=3、主要维度1024/1025、all-16 Tile及nested main/tail/halo | validated `TemporalState`返回`ReadyForNextCoordinate(Representation)`；重复query结果相同，source IR byte-identical | public search从missing partial-feasibility推进为missing representation，actualization仍为0 |
| earlier closed prefix | valid `SpatialState`、valid `RegionState` | typed state自身分别要求Region与Temporal；partial readiness尚未调用且不补default suffix | 对应continuation仍由其axis owner展开，readiness query count保持0 |
| rank-zero与merge-only | rank-zero execution scope、merge execution无scope | empty temporal vector是valid closed coordinate；merge-only不因无iterator被误判missing execution | layout-domain可继续消费typed logical versions，不收到假resource结论 |
| domain/plan mismatch | 来自不同temporal domain的plan、删除或复制scope | `CompilerBug`且detail定位当前closed coordinate；不降级成MissingCoordinate | source和domain不变，修正plan后可重新query |
| prior typed unsupported | region/temporal relation本身unsupported或indeterminate | 仍由对应axis transition分类并保留siblings；partial readiness不得重新分类、吞掉或转成resource rejection | Core work counts区分axis failure，readiness query count不增加 |
| resource隔离 | 相同A--E prefix配任意不同target memory capacity/allocator环境 | readiness API没有bytes、footprint、capacity、packing、lower/upper bound字段；结果严格相同 | source scan及API type gate；只有full-feasibility可调用actual Q50.0 |

### `partial-feasibility`当前已实现子集与重新验证门禁

- `StructuralReadinessResult`只包含`ReadyForNextCoordinate`/`CompilerBug`、下一坐标和detail；没有
  resource bytes、bound、packing problem、capacity witness或rejection字段。query只验证current `TemporalPlan`是否属于对应typed domain，
  重复调用不修改IR或domain；RegionPlan已由产生`TemporalState`的前一typed transition验证，不在这里重复。
- planning coordinate identity移到独立`PlanningCoordinate` owner；Q50 structural query只依赖plan/domain schema，不反向include或拥有Q51
  state/frontier。`SpatialState`与`RegionState`继续由各自axis owner报告missing Region/Temporal及其typed failure，不为readiness增加
  永远不可达的`MissingCoordinate`/`Unsupported` result variant。
- production session在first validated `TemporalState`上复用同一cached `TemporalDomain`执行一次readiness query，成功后public search报告
  missing Representation；prior spatial/region/temporal unsupported或indeterminate仍由原axis transition分类，readiness count不增加。
- query/API/source没有resource estimator或target-memory输入；partial state仍保持`candidate_actualizations=0`。只有后续完整assignment能进入
  actual candidate admission，readiness不能签发SPM/DDR/transport结论。

本轮fresh证据：`StructuralReadinessTest` 3/3覆盖rank-3 1025、重复query、duplicate scope、rank-zero、merge-only以及translation-normalized
nested + top-level replica closed prefix；来自另一个nested domain的plan返回CompilerBug。`PlanningSessionTest` 6/6与
`SearchRoutingTest` 2/2证明closed state顺序、production readiness count=1、missing Representation、source不变及actualization=0；direct
集合11/11通过。header/source scan确认result/API没有bytes、footprint、capacity、packing、allocator或target-memory字段；本项不创建IR、
不调用Q50.0，也不保留永远不可达的Missing/Unsupported result variant。Region/Temporal schema重新验证门禁已经闭合。

### Actual resource admission boundary

完整candidate的资源结果来自现有Q50.0：CardModule拆分、TileRegion→Instr、fresh completion、actual allocation/lifetime、
`PlanSPMMemory`/MiniMalloc、DDR planning、transport和target verification。每个SPM demand必须通过current materialization relation映回
operand、result、scratch、movement或output owner；无owner demand是contract failure。

typed结果只包含：

- `Accepted`：保留同一次actual candidate、offset和下游结果；
- `ExactSPMCapacityRejection`：携带actual allocation conflict/oversized demands及完整owner relation，可反馈controller；
- 其它exact target rejection：只在对应actual verifier提供完整witness时成立；
- `ResourceExhausted`、unsupported、timeout和内部失败：保持Indeterminate/Unsupported/CompilerBug，不生成no-good。

baseline和search调用同一actual gate。任何estimate API、plan-side packing problem、fast allocator、fallback allocator或parity proof都不是
合法性入口；相关源码、测试和文档必须不存在。

## Q50.G：Layout and Physical Representation

### Pipeline contract

Pipeline position:
- Upstream IR / input:
  immutable normalized TensorProgram/algorithms、Q50.A exact value requirements、Q50.B/D/E spatial/region/traversal plans、partial
  structural readiness及current Wafer physical encoding/access interfaces；没有selected movement、buffer或schedule。
- Current stage responsibility:
  为每个logical value/version选择唯一primary representation，为actual use requirements选择primary或可共享derived versions；只枚举
  encoding/access interface对exact logical domain可解释的layouts。建立operation tuple、SSA/version、view/alias、external boundary、
  reduction和future movement的constraint/cost factor graph；deterministic solver产生proposals/bounds，exact domain仍由typed
  variables完整表示。complete-candidate emitter一次物化selected primary/derived versions。
- Output IR / files:
  query不写IR，Q51 state保存typed `RepresentationPlan`；solver reduction records/cost matrices保持query-local。candidate actual
  CardModule用MemoryAttr/typed layout ops和SSA uses自包含表示selected versions，不携solver residue、proposal ordinal或score。
- Downstream consumer:
  Q50.H以producer primary version、consumer operand version和exact demand生成movement/conversion组合；Q50.I lifetime及Q50.J calendar
  在representation变化后重算。Q51联合其它轴形成complete candidates；Q50.0从每个candidate actual IR判定SPM/DDR合法性。
- User-level driver / named pipeline:
  Q51 planning session；`none`仍由Q49.P current deterministic representation构造，不进入本域枚举。
- Explicit non-goals:
  不选择route、spill、recompute、retention、buffer、worker或全局winner；layout solver不得clone/apply IR或用固定Top-k永久截断全局合法域，
  不把layout写回logical exact-demand或edge action，不让每个operand use各自覆盖producer primary identity。
- Done criteria:
  domain与independent value-version/use-binding reference一致；solver对chain/diamond/fanout/view/reduction/collective与movement lower-cost
  小图达到reference optimum并报告coverage；unsupported encoding无state；winner primary/derived conversion、output writeback、fanout
  secondary sharing及atomic failure有candidate actual证据。旧PBQP donor的constraint、cost、reduction、shared-secondary和tests逐项迁入后，
  fixed Top-4/local winner/clone apply保持删除。

### Q50.G work-item分界

Q50.G由两个work items完成：

- `canonical-representation-plan`只为canonical B--E已经显式产生的shaped logical versions建立一对一primary physical version，
  使用current `MemLayout::Tensor` correctness encoding。boundary fragment、support result、execution result、ordinary partial和coupled
  component各有typed `RegionValueVersionId`；exact domain/type只进入query-local resource description，不复制进plan identity。已经由
  bounded construction证明为finite disjoint rectangles的其它exact normal form在此规范化为语义等价的`BoxUnion`，使G/H/I/F消费
  同一可物化resource；无法完成该证明时返回typed unsupported，不用bounding box。exact-empty与scalar不伪造physical version。
  本项没有derived conversion、alias version、shared secondary或layout solver。
- `layout-domain`在partial-feasibility后扩同一current合同，加入全部legal primary/derived encodings、conversion/alias producer、use
  bindings、constraint solver、Core consumer和selected apply，并迁移/退役旧fixed-slot/current apply。Tensor canonical点只是独立
  correctness carrier，不是preferred layout、operation tuple default或global winner。

当前`canonical-representation-plan`覆盖矩阵如下。shape用于证明exact logical versions和physical byte-span描述覆盖真实规模，
不进入encoding选择；小shape只用于单一合同负例。

| 覆盖类 | 代表输入 | canonical RepresentationPlan exact断言 | resource/downstream witness |
| --- | --- | --- | --- |
| aligned/ragged all-Tile | rank>=3、主要维度1024/1025、all-16 Tile | 每个nonempty boundary fragment、support value和execution final result各有且仅有一个primary Tensor version；输入顺序扰动后ID/plan稳定 | resource description逐version保留exact domain、element type和Tensor encoding，canonical-movement直接消费 |
| chain/fanin/fanout/diamond | 1025级multi-root/multi-path | 同一producer的不同consumer fragments是不同boundary logical versions；同一support result只一个version；不按node pair、first use或layout slot合并 | H可逐fragment生成movement，同时同support version在region内只物化一次 |
| multi-result/DPS init/view | rank>=3 multi-result、slice/reshape/pad及1025级finite GeneralPresburger分片 | result index、support semantic ID和boundary source/use/owner均进入typed version identity；scalar init/capture无physical version；可证明的finite disjoint pieces规范化为等价BoxUnion | resource exact domain与source type逐项对应，不从operand位置恢复；不能矩形化时typed unsupported而非bounding box |
| ordinary reduction | 整除/非整除contribution及多个merge group | 每个partial result、merge final result分别一个version；partial不是final primary的alias | H/I按group/version区分payload、exact byte span和lifetime；actual gate独立规划offset |
| FD coupled state | rank-5/6、single/multi-K2 | 每个contribution及merge的Maximum/Sum/Accumulator各自有component version，但同一component不按use重复；final output另有result version | component resource type/domain与attention-ready requirement逐字段一致，不隐藏scratch/state |
| empty/scalar/rank-zero | exact-empty boundary、scalar scale/capture、rank-zero tensor | empty/scalar不建version；nonempty rank-zero tensor仍建Tensor version | movement对empty无action，rank-zero resource保持0-rank exact set |
| typed failure | duplicate logical/version ID、missing work/execution/fragment、dynamic/non-shaped descriptor或plan/resource不一致 | compiler-contract/typed unsupported且无partial plan，不默选其它layout | 修正输入可重新query，source IR byte-identical |

`layout-domain`施工前覆盖矩阵如下。小图只用于独立assignment/solver oracle；query与actual witness使用rank至少为3、主要维度
1024/1025/1031的static tensors，并覆盖整除与非整除physical spans。

| 覆盖类 | 代表输入 | exact断言 | Core / apply witness |
| --- | --- | --- | --- |
| primary encoding完整域 | rank-3 1024/1025、rank-6 coupled component、rank-zero | 每个logical value恰一个primary；domain逐exact box枚举current encoding interface接受的全部Tensor/NTensor/Cx/NCx states，无preferred/default剪枝 | `TemporalState -> RepresentationState`；query零IR，public search推进到Movement |
| use binding与derived version | chain/fanout/diamond，两个compatible uses与两个incompatible uses | use可绑定primary或显式layout-conversion path；shared anchor只产生一个derived definition，per-use anchor产生不同typed IDs；每use恰一binding | `PhysicalVersionBuilder`按dependency order bind一次并直接wire planned ID，不按first use查找source |
| dependent source/use constraints | tiny fanout shared/per-use图及input-order扰动 | primary与use option依赖完整展开；complete successors与independent flat assignment oracle逐key一致；duplicate/missing source/use fail closed | 无fixed Top-k或local winner；Core continuation穷尽tiny 112点 |
| view/alias与conversion链 | primary→derived、derived→derived及无exact relation的alias request | conversion path逐step显式source/target encoding与anchor并天然acyclic；没有exact view relation时无alias state，不在apply猜 | actual IR中每个planned conversion ID恰一个`tile.materialize_layout`；preflight失败零残留 |
| reductions/attention/boundary | ordinary partial/merge、FD Maximum/Sum/Accumulator、external input/output | logical component/result identity与canonical resource一一对应；layout choice不合并components、不把partial当final、不隐式load boundary | H继续从selected physical version读取，不由G生成DDR/peer action |
| typed failure与资源隔离 | duplicate/missing version/use、wrong source path/layout/domain、cycle、contradictory tuple、unsupported element/encoding | 无partial plan；query不clone/materialize、不计算SPM/DDR capacity或winner；source IR byte-identical | 修正assignment可重试；complete-candidate transaction之外不留下IR |

### 机制

representation identity基于Q50.A/C/D/E产生的`RegionValueVersionId`与exact domain，不基于`(node,operand)`位置：required/replica
execution result、stored/shared boundary、reduction contribution/merge result和program boundary是不同logical versions；同一logical
version的fanout uses共享identity。

```text
RepresentationPlan
  values: LogicalRepresentation[]
  physicalVersions: PhysicalVersionPlan[]
  uses: PhysicalUseBinding[]

LogicalRepresentation
  value: RegionValueVersionId
  primary: PhysicalVersionId

PhysicalVersionPlan
  id: PhysicalVersionId
  logicalValue: RegionValueVersionId
  encoding: PhysicalEncoding
  exactDomainClass: ExactDomainClassId
  scope: RegionGroupId | TraversalScopeId
  producer: ExecutionOutput(execution, result)
          | BoundaryInput(boundary)
          | AliasView(sourceVersion, exactViewRelation)
          | LayoutConversion(sourceVersion, anchor)

PhysicalUseBinding
  use: RegionUseId
  version: PhysicalVersionId
```

每个`LayoutConversion` version最多物化一次并可服务所有exact domain/encoding/scope-compatible uses；部分overlap或不同scope不能仅因
layout相同强行共享。conversion source和anchor是plan字段，emitter不得“找一个现有layout”自行选链。`PhysicalEncoding`直接使用current
closed MemLayout/encoding attr及interface facts，不再包装第二个layout enum。

constraint graph有两类variables和typed factors：

- `ValueVersionVariable` domain是`(primary physical version, conversion/alias version subset/grouping)`；
- `UseBindingVariable` domain是该use可见的physical versions；
- operation factor验证一个selected implementation在完整operand/result encoding tuple上的Tiling/compute legality；
- SSA factor要求producer result与primary一致、use binding实际存在；
- view/alias factor证明metadata alias、same-storage或必须materialize；
- boundary factor固定/限制program input/output、collective和reduction merge encodings；
- H尚未选route/action时只加入route-independent conversion/movement lower bound，不冻结movement。

hard-invalid tuple成本为infinity；unary/pairwise cost是checked typed vector，至少分开logical conversion bytes、minimum commands、extra
physical bytes、compute implementation penalty和knowledge。未知term不能当0参与winner比较：它对bound取sound lower value并
标记knowledge，完整cost由H–J补齐。solver只生成proposal/bound，Q51 exact state仍能逐variable访问全部legal assignments。

### 约束求解算法

这里采用有限constraint graph，PBQP是其中pairwise部分的求解形式，不把LLVM register allocator的启发式结论直接当成全局最优。
LLVM pinned实现的R0/R1/R2分别删除degree 0/1/2节点、把被删节点的最小代价传给相邻节点，并按reduction stack反向恢复选择；它在
高阶core上允许启发式断边，这适合register allocation，但不能为本任务证明完整搜索。Q50.G只复用前三种等价变换和反向恢复结构；
高阶core必须由可证明的branch-and-bound处理，或者明确返回预算未完成，不能伪装成最优。

本项采用的一手资料与实现边界如下：

| 资料 | 可复用事实 | Wafer采用/不采用 |
| --- | --- | --- |
| [SSA-Based Register Allocation with PBQP](https://beza1e1.tuxen.de/pdfs/buchwald11cc.pdf) | RE/R1/R2把问题缩小且保持最优值；degree>=3的RN是质量启发式 | 采用degree-0/1/2 exact reduction及反向恢复；不采用RN、spill-cost ratio或early local choice |
| [LLVM `ReductionRules.h`](https://www.llvm.org/doxygen/ReductionRules_8h_source.html) | pinned实现按neighbor state把R1最小值累加到unary，R2把被删变量的条件最小值累加到neighbor matrix | 逐式复核current R1/R2矩阵方向与choice record；不用LLVM mutable graph metadata作为Wafer state |
| [LLVM `RegAllocPBQP.h`](https://llvm.org/doxygen/RegAllocPBQP_8h_source.html) | LLVM先耗尽R0/R1/R2，再对剩余节点使用conservatively-allocatable/spill启发式 | Wafer在R0/R1/R2后对residual core做稳定exact enumeration；work limit返回Indeterminate，不把启发式结果称Optimal/NoSolution |

仓库pinned LLVM源码与上述公式一致；current solver使用checked nonnegative cost、显式infinity、stable state ID tie-break和最终原问题
复验。proposal work limit只限制proposal生成，不进入RepresentationDomain legality或raw successor。

#### 图的精确构造

所有变量和状态先按`RegionValueVersionId`、`RegionUseId`、encoding及region中的稳定语义顺序编号，不用operation地址、walk偶然顺序
或hash遍历顺序。为了避免把“哪些use共享secondary version”预先枚举成集合分割，query内部把上面的
`ValueVersionVariable`正规化为三类有限变量；它们仍共同投影成上面的`PhysicalVersionPlan`与`PhysicalUseBinding`：

```text
PrimaryVersion(value)                    -> one present PhysicalVersionId
PhysicalVersion(value, encoding, anchor) -> absent
                                           | ExecutionOutput
                                           | BoundaryInput
                                           | AliasView(sourceVersion)
                                           | LayoutConversion(sourceVersion)
UseBinding(use)                           -> one dominating PhysicalVersionId
```

`anchor`只来自actual IR中可证明合法的conversion位置：value定义之后的producer region入口、单个use入口，以及一组use的最近公共支配
region入口。纯、无alias破坏且encoding relation exact时，任意服务同一组use的conversion都可移动到它们最近的合法公共支配点而不改变
结果；因此这些canonical anchors覆盖共享与不共享两种选择。若effect、alias、region crossing或dominance证明不成立，只建立per-use
anchor，不能猜测共享。version producer state显式选择source；conversion/alias dependency必须形成DAG并最终到达ExecutionOutput或
BoundaryInput。一个present `LayoutConversion`收取一次conversion成本；任意多个compatible `UseBinding`可引用它，binding factor要求
该version为present、anchor支配use、exact domain一致且lifetime scope合法。这样fanout secondary sharing和derived-to-derived chain由
assignment直接表达，不靠apply阶段看到多个use后临时合并或选择任意source。

operation对完整operand/result encoding tuple的约束通常是高元factor。求解器按以下规则正规化：

- unary和binary factor直接成为PBQP cost vector/matrix；
- 若高元factor的合法tuple表在预先计算的work bound内，则建立一个auxiliary tuple variable；它的每个state就是一个合法tuple，并用
  projection-equality matrix连接原变量。这一变换在assignment与cost之间是一一对应；
- 若tuple乘积超过bound，则保留按interface惰性求值的高元factor，由residual search直接查询。超过资源限制只会得到
  `Indeterminate`或已验证的feasible proposal，不会把未展开tuple判成illegal，也不会选一个default tuple。

#### 有限域传播与共享版本收敛

图构造后运行support propagation，直到没有domain变化：对每个factor和每个参与变量状态，检查当前其它domain中是否至少有一个
finite-cost支持tuple；没有支持才删除该状态。删除状态后只把相邻factor重新入队。`PhysicalVersion=absent`会删除指向它的binding和
producer-source choice，而仅剩某个binding时会反向要求对应version为present；primary/version producer、operation tuple、view/alias
与boundary constraints以
同一规则传播，不另写“layout恢复”循环。

```text
propagate(graph):
  queue = all factors in stable order
  while queue not empty:
    factor = pop_front(queue)
    for variable in factor.variables:
      for state in variable.domain:
        if no finite supporting tuple in factor under current domains:
          remove state
          if variable.domain is empty: return NoSolution
          enqueue variable's other factors in stable order
  return ReducedDomains
```

传播只删除已由exact factor证明没有支持的状态。每次迭代至少删除一个有限状态，故最多删除初始状态总数次后结束；用factor tuple的
support counter可把一次完整收敛控制在与实际检查tuple数量及失效support数量成正比的范围。cost dominance不参与legality传播。
只有在一次proposal求解内部，某状态的known unary cost及所有incident factor row均不差、至少一项严格更好、且future H–J项已证明
相同，才可暂不探索被支配状态；该结论不能写成Q51 no-good，未知cost会使两状态不可比较，完整typed domain仍保留该状态。

#### 单个最优proposal与预算结果

传播完成后先按connected component分解。每个component依次执行下列算法：

```text
solveComponent(component, constraints, workLimit):
  domains = propagate(component restricted by constraints)
  if a domain is empty: return NoSolution

  while a degree-0/1/2 variable exists:
    eliminate it exactly with R0/R1/R2
    record the minimizing state for every remaining-neighbor state tuple

  search the residual core:
    choose the smallest remaining domain; break ties by larger factor degree,
      then stable semantic id
    visit states by admissible local lower bound, then stable state id
    after each choice run incremental propagate
    lowerBound = sum(min remaining unary/factor cost independently)
    prune only when lowerBound cannot beat the current feasible assignment

  if every residual branch was closed, reverse elimination records
    and return OptimalForRepresentationBound
  if workLimit ended after finding a checked feasible assignment,
    return FeasibleWithLowerBound(assignment, lowerBound)
  return Indeterminate(lowerBound)
```

R1计算`neighborState`固定时被删变量的最小值；R2为两个neighbor的每个state pair计算被删变量的最小值并累加成新matrix，随后按
record反向恢复，所以两者不丢assignment。residual lower bound把每个尚未决定的factor独立取最小值，可能偏松但不会高估；只有完整
穷尽后才报告`OptimalForRepresentationBound`。`NoSolution`只来自exact空domain或完整穷尽，资源耗尽、unsupported interface、cost
overflow和内部合同错误分别返回`Indeterminate`、`Unsupported`、`Indeterminate`和`CompilerError`。已找到的assignment在返回前必须
重新逐factor验证；未通过就是`CompilerError`，不能进入Q51。

cost比较由Q51传入的immutable typed comparison policy完成，checked arithmetic禁止overflow。Q50.G最优只表示对当前
representation lower-bound cost最优，不表示跨movement/buffering/schedule的global winner；H–J补齐cost后仍由Q51比较完整candidate。
求解过程中只维护必要的work counter和cancellation检查，不默认输出graph dump、统计、proposal日志或timing。

#### 惰性后继而非固定Top-k

Q51需要另一个representation proposal时，使用domain partition生成下一个，不设置固定Top-4，也不重复物化IR。设当前解按稳定变量
顺序为`x[0..n)`；对每个位置`i`建立一个互不重叠的子问题：固定`x[0..i)`，并从第`i`个domain排除`x[i]`。各子问题只保存domain
限制并调用上面的pure solver，结果按`(cost lower bound, complete semantic assignment)`进入稳定priority queue：

```text
nextProposal():
  if queue is empty: solve unrestricted domains and push result
  best = pop the entry with the smallest lower bound
  if best is not exact: resume that subproblem until exact or cancellation
  if cancellation stops it: return only an unranked feasible suggestion/Indeterminate
  for i in stable variable order:
    child = best.constraints
    child.fixPrefix(best.assignment, i)
    child.exclude(i, best.assignment[i])
    if child domain is nonempty: solve child and push its result
  return best.checkedAssignment
```

这些children对当前解之外的assignment形成不交且完备的partition；只有queue中最小lower bound对应的子问题已经exact solved时，才把
它报告为严格的下一最优。预算中止时只返回已验证但不带排名承诺的proposal及其`FeasibleWithLowerBound`状态，Q51不能据此宣布其它
layout不可达。Q51也可绕过proposal顺序，直接对
任一typed variable/state建立child，因此solver永远不是完整域的唯一入口。

#### 通用性、复杂度与本子任务验证

- 通用性来自SSA value/use identity、operation/interface tuple legality、dominance、alias/effect与exact index relation；算法没有LLaMA、
  op名字、固定shape、固定四种layout数量、单结果或链式DAG分支。unsupported op/layout由interface事实fail closed。
- auxiliary tuple变换、R0/R1/R2、反向恢复和domain partition都有assignment保持证明；shared derived version由presence/binding约束一次
  计费，apply不再猜测fanout关系。
- 设变量最大domain为`K`、residual core变量数为`c`。R1最坏`O(K^2)`，R2最坏`O(K^3)`；高元tuple显式化成本是参与domain
  大小之积且必须先过bound；residual exact search最坏`O(K^c)`。PBQP本身是NP-hard，文档和API不得承诺多项式时间。
- 有界穷举oracle枚举所有assignment，与solver比较chain、cycle、diamond、multi-result operation factor、view/alias和shared-secondary图的
  optimum/NoSolution；另测tuple显式化与lazy factor等价、R2 back-substitution、每一个lazy successor无重复且集合完备、预算耗尽分类、
  stable-id确定性，以及未知H–J cost不会触发dominance删除。上述只验证query；actual representation apply与下游消费留给下一子任务。

### G-3 专项调研：independent oracle与complete-candidate physical-version apply

OpenXLA Layout Assignment先在logical graph上传播完整layout constraints，冲突或graph endpoints再显式插入copy；physical layout成为HLO
shape的一部分，下游直接读取，不由每个consumer临时挑选。MLIR One-Shot Bufferize同样先基于完整SSA做analysis，再统一rewrite并用
explicit materialization/copy表达out-of-place结果。两者共同支持本仓G的apply边界：`RepresentationPlan`先关闭，complete-candidate construction
按plan建立一次physical version graph；`getOrMaterialize`式“缺什么就现场找/造什么”不能作为compiler contract。

current apply混装的具体问题必须逐项退出：`BufferVersions`按source `Value`固定保存四个layout slots，`lookupAny`按硬编码顺序选择source；
operand conversion前覆盖`buffers[operand]`、convert后再恢复；result conversion把selected layout变成新的唯一primary；找不到source时
还会隐式load DDR、clone allocation或materialize constant。这样同一fanout的first consumer决定cache、G偷偷执行H movement、source
SSA和physical version identity混为一体，也无法证明plan中“present一次”的derived version在IR中恰出现一次。

outer C preparation在任何IR mutation前把完整G/H计划关闭为`PreparedPhysicalVersions`：

```text
preparePhysicalVersions(regionPlan, temporalPlan,
                        representationPlan, movementPlan):
  require every RegionValueVersionId has exactly one primary PhysicalVersionId
  require every PhysicalVersionId is unique and belongs to one exact scope
  validate each ExecutionOutput against selected operation encoding tuple
  validate each BoundaryInput against the exact H endpoint/version
  validate each AliasView with ViewLike/Bufferizable and exact index relation
  validate each LayoutConversion source, target encoding, exact domain,
           dominance anchor, effect scope and PhysicalLayoutRelation
  require conversion/version dependency graph acyclic
  require every RegionUseId binds one dominating, exact-covering version
  require every program output has an explicit external writeback version
  return PreparedPhysicalVersions in dependency/anchor order
```

`PhysicalLayoutRelation`继续是encoding-owned verifier：对每个planned memref type检查logical→physical bit-offset relation、all-and-only
piece coverage、exact bit span、alignment、padding和element spans。它不选encoding、不比较cost，也不靠generic Presburger试错；unsupported
encoding在query domain中没有state，不能等apply再换Tensor layout。

actual construction不单独运行一个layout pass；它是D/E/H emitters共同调用的窄version builder：

```text
PhysicalVersionBuilder
  bindBoundary(versionId, SSAValue)
  bindExecutionOutput(versionId, SSAValue)
  emitAliasView(versionId, sourceSSA, exactViewRelation, rewriter)
  emitLayoutConversion(versionId, sourceSSA, targetType, rewriter)
  lookup(versionId) -> FailureOr<SSAValue>

emitPreparedPhysicalVersions(prepared, builder, rewriter):
  for version in prepared dependency/anchor order:
    require every source version already bound
    execute exactly the producer named by PhysicalVersionPlan
    bind versionId once; duplicate bind is a compiler-contract error
  for use in stable RegionUseId order:
    wire the exact planned version directly to the selected consumer operand
  verify all-and-only planned versions/uses and output writebacks were emitted
```

execution output通常由selected compute emitter直接绑定；alias view不产生copy；layout conversion只生成一个显式
`wafer.tile.materialize_layout`及其typed memory effects。boundary load/send/recv仍由H producer绑定，G不得创建它。map的key是
`PhysicalVersionId`，不是source SSA、layout enum或operation pointer；primary只是`LogicalRepresentation.primary`指向的普通version，
不会因consumer use临时覆盖。fanout consumers绑定同一ID自然共享一次conversion，split versions则明确产生多个ops。

prepare已验证所有可能失败的encoding/relation/dominance/closure事实；emit仍逐op fail closed并由C CardModule subtree guard整体回滚。
失败不切换source encoding、不插default copy、不回solver取下一proposal。普通apply不采集version counts/timing；显式instrumentation
可观察但不参与semantics。

#### Oracle、actual witnesses与迁移门禁

- independent有界穷举reference平铺每个logical value的primary encoding、canonical conversion presence/source/anchor、compatible use
  groupings和bindings，逐factor过滤；与G exact successor canonical plan集合一致。solver在chain/cycle/diamond/fanout/view/multi-result/
  boundary小图的proposal cost/NoSolution与flat oracle相同，预算结果不删domain；
- `PhysicalLayoutRelation`只在明确标注的2--4维有界oracle shape逐logical point比较独立codec，覆盖padding、alignment、multi-piece；
  同一encoding另有rank>=3、主要维度>=1024的整除/非整除production正例，覆盖strided/tail路径；rank-zero与unsupported type
  分别保留为明确的边界/负例。query domain与prepare使用同一capability事实；
- actual IR覆盖：producer直接产primary、primary→derived、derived→derived chain、alias-only view、两个fanout uses共享一个conversion、
  incompatible uses产生两个planned versions、D replica/stored version、reduction contribution/merge、boundary input与program output
  writeback；每个plan ID恰一definition、每个use恰一binding；
- negative injection覆盖missing/duplicate version、cycle、wrong domain/encoding、non-dominating anchor、alias relation mismatch、missing H
  boundary和output writeback；失败前后source一致且C subtree无残留；generic/custom roundtrip、verify-each和Q50.0 downstream verifier执行；
- planning/solver counters证明partial query零IR mutation/materialization。每个complete candidate中version/conversion count只由assignment决定，关闭observer后IR
  不变；不运行重型LLaMA search。

迁移表：

| Current / donor能力 | 终态owner | 保留或修正的witness | 退役条件 |
| --- | --- | --- | --- |
| current `CardPhysicalRepresentationDomain`按`(Tile,node,operand/result)`Cartesian | G value/version/use exact domain | per-Tile、fanout、multi-result、boundary reference equality | old role/index assignment与node-wide temporal/trial inputs零残留 |
| current `BufferVersions`/`lookupAny`/`getOrMaterialize` | `PhysicalVersionBuilder` keyed by `PhysicalVersionId` | primary/derived/alias/shared/split actual witnesses | fixed four-slot cache、arbitrary source choice和first-use materialization删除 |
| current operand map save/overwrite/restore | prepared `PhysicalUseBinding` direct wiring | mixed-layout fanout不覆盖producer primary | `buffers[operand]=primary`临时状态机零残留 |
| current hidden DDR load/allocation clone in layout lookup | H boundary producer + C symbol/SSA closure | boundary load与layout conversion分别可数 | G transitive call graph无StorageLoad/peer/alloc clone决策 |
| current `StructuredNodePhysicalRepresentation` grouped bynode | `PreparedPhysicalVersions` | multi-root/replica/version scope exact mapping | operand/result nullable vectors与“nodes disagree”协议删除 |
| current `LayoutMaterializeOp`与`PhysicalLayoutRelation` | G actual op与encoding verifier | exact layout copy、padding/alignment/point oracle | 保留typed op/interface，不保留selector/cache side effects |
| old PBQP constraint/reduction/shared-secondary tests | G-1/G-2 current factor graph/solver + G-3 apply | oracle optimum、shared version once、budget classification | 所有独有tests迁入active owners后旧source/tests才可删除 |

source organization gate要求query/domain/solver、physical-layout analysis、winner preparation和actual version builder为独立libraries/files，
H/I只include窄typed plan/relations；删除current whole-Module apply、compat wrappers、default layout selector、fixed Top-k、clone/replay和
operation-pointer version cache后才算G闭合。

### Gate

G-1/G-2 query/solver gate与G-3 oracle/apply/migration gate全部通过；不同RepresentationPlan使F/I/H/J按observed typed dependencies
失效重算，不存在SPM probe cache。lowering只验证并消费selected physical versions。layout使相同temporal point的SPM legality或global
winner改变是Q51 closure gate，不能用G局部proposal代签。

### `layout-domain`当前已实现子集与重新打开门禁

- `RepresentationPlan`原位收敛成一套`logicalValues + physicalVersions + uses`合同；`PhysicalVersionId`以logical value和typed derivation
  path表达primary、layout conversion及identity alias，step显式保存source/target encoding、shared/per-use anchor和alias source logical value。
  canonical producer及movement/storage/attention consumers同批切换，没有parallel schema或旧field reader。
- `RepresentationDomain`逐logical exact box调用`PhysicalLayoutRelation`建立Tensor/NTensor/Cx/NCx legal states，惰性完整枚举primary与use
  bindings。use可绑定primary、shared conversion、per-use conversion或已有exact identity proof的alias；tuple constraints只保留显式legal
  encoding tuples。successor不设Top-k、不比较局部winner，并与independent primary/use/anchor flat oracle逐key一致。
- shared compatible fanout uses引用同一derived ID并只产生一个definition；per-use anchors产生不同IDs。conversion path是source-prefix DAG，
  identity alias显式引用另一个logical primary；domain不为domain/type不等的alias建立state。input/resource/use顺序扰动不改变first plan或集合。
- `prepareRepresentationPlan`在mutation前验证all-and-only versions、source path和encoding；`PhysicalVersionBuilder`只按ID bind/lookup，
  primary→derived与derived→derived各生成一个`wafer.tile.materialize_layout`，identity alias零copy，duplicate/wrong-layout preflight零IR残留。
  boundary/movement producer仍由H绑定，G builder不隐式load、clone allocation或选择transport。
- `RepresentationState`与continuation/cache已经接入production session；readiness之后first canonical Tensor point进入new domain，public search
  报告missing Movement且actualization仍为0。tiny one-Tile production continuation穷尽112个siblings。旧node/operand role domain与
  `BufferVersions/lookupAny`只作为旧Unified及尚未迁完H/I emitter donor保留，不进入new state；后续owner迁完实际movement/storage witnesses后删除。

fresh证据：`RepresentationDomainTest` 4/4覆盖1024/1025、rank-6 1031、rank-zero、196点flat oracle、shared/per-use conversions、
identity alias、matching tuple constraints、input-order与typed failures；`PhysicalVersionBuilderTest` 3/3覆盖shared、derived chain、identity alias、
duplicate/wrong-layout atomic failure；`PlanningSessionTest` 6/6、`SearchRoutingTest` 2/2及`CanonicalRepresentationPlanTest` 5/5共同证明
current schema、Core 112 siblings、missing Movement、source不变和zero actualization，direct集合20/20；ordinary host unit 851/851
（6个独立heavy baseline cases不在本项重复）；core lit 227/227，Tools/Runtime lit 35 passed、4 configured unsupported；compiler public
link、source organization和diff检查通过。

完成状态复核确认这些证据只覆盖Cartesian successor、手工tuple constraint和test-only builder。production没有构造operation/interface
tuple或alias constraints，没有本节规定的support propagation、PBQP R0/R1/R2、residual branch-and-bound和proposal lower bound；
`PhysicalVersionBuilder`没有production caller，full-feasibility拒绝derived representation。`layout-domain`重新打开，必须完整实现并接入上述
solver、constraint producer、selected construction/verifier与actual downstream witness。

本轮closure在同一current schema上补齐：

- `RepresentationPBQPSolver`使用checked nonnegative cost与显式infinity，先应用exact R0/R1/R2并保存逐neighbor-state choice record，
  再对degree>=3 residual core按domain size/degree/semantic ID稳定穷举；最终在原factor graph复验assignment。path、cycle与clique同flat
  oracle的cost/lexicographic optimum一致；无解、work-limit Indeterminate与broken graph保持不同typed状态，没有RN或fixed Top-k。
- production factor graph为每个primary value、每个use option及每个n-ary legal tuple auxiliary建立变量；value/use与alias-source equality
  是binary hard factors，高元tuple通过projection-equality factors保持一一对应。PBQP只先发一个普通domain member；Core随后恢复raw
  cursor并跳过重复，因此proposal关闭或预算耗尽不改变exact leaf集合。
- representation inventory不再要求singleton Region或top-level required-only Temporal。required/nested/replica execution、local use与external
  boundary均进入同一logical/version/use inventory；replica result使用`RegionExecutionId`保持candidate-local identity，merge execution继续
  没有假temporal scope。
- `CompleteCandidatePlan`携带selected `RepresentationPlan`并在actual mutation前经current domain与
  `prepareRepresentationPlan`复验。selected group emitter逐candidate-local node解析operand use version与result primary layout；shared anchor
  让同一derived layout跨compatible fanout uses保留一次，per-use anchor分别物化。1024/1025 fanout实测selected NTensor conversion分别为
  1与2，二者都进入一次Q50.0；required/replica stored/direct既有actual矩阵同时改为显式Tensor representation输入。
- full-feasibility不再把derived representation预先标成Unsupported；它从selected plan重建primary/full exact resource coordinate后继续
  movement/storage compatibility。当前没有Wafer-specific operation layout interface的structured op通过显式use conversion表达可达tuple；
  已提供的typed tuple/identity-alias facts进入同一factor graph，不从operation名、shape或target猜额外hard tuple。
- solver、domain、prepare和actual builder均不读取SPM bytes、footprint、capacity、packing或actual offset。布局改变后的实际SPM合法性仍只
  能由后续完整candidate的Q50.0签发。

本轮fresh验证：`RepresentationPBQPSolverTest` 3/3覆盖R0/R1/R2 path、degree>=3 cycle/clique residual、flat optimum、stable tie、
NoSolution、Indeterminate与broken graph；`RepresentationDomainTest` 4/4继续覆盖完整raw primary/use/alias/tuple域，并验证PBQP proposal
是普通member、预算不影响raw set。`SelectedRegionMaterializationTest` 6/6新增1024/1025 fanout shared/per-use NTensor actual witness，
selected conversion定义分别为1/2且均通过Q50.0；required/replica、stored/direct、nested main/tail既有矩阵全部使用显式Tensor
RepresentationPlan。完整build与排除future attention production case后的host unit 914/914通过；configured lit、source/IR organization及
4个public-link smoke通过。future attention failure仍属于EventGraph merge-scope合同，不作为G证据。

## Q50.H：Explicit Data Movement

### Pipeline contract

Pipeline position:
- Upstream IR / input:
  immutable selected TensorProgram、Q50.A exact per-destination demand/ownership与reduction merge requirements、Q50.B placement、Q50.D
  local use bindings与explicit boundaries、Q50.E temporal scopes、Q50.G source/destination physical versions及current card transport facts；
  尚未选择buffer slots、event schedule或stage pipeline。
- Current stage responsibility:
  为每个D/A boundary生成exact DDR stage、target-routed peer transfer、explicit Tile relay graph及已获target capability支持的collective
  movement完整有限域；形成all-and-only logical/physical payload、endpoint、relay、completion obligation和topology/resource facts。
  candidate apply才生成load/store、send/recv/token及relay work；wait由J在fixed storage/structure上放置。compute replica和layout versions
  分别已由D/G决定。
- Output IR / files:
  query不写IR，输出typed movement assignment及可重算reuse/cost/resource facts；candidate apply输出current CardModule中的explicit
  movement/event-producing ops，不写sidecar或route attr bag。explicit relay graph在commit后由逐transfer ops表达；只有target真的
  暴露programmable route时，routing才是typed assignment并由target IR合同消费。
- Downstream consumer:
  Q50.I从planned waves/movement completion推导slot domain，Q50.J从planned send/recv/completion、compute与resource effects构造有限
  worker/resource/control/completion schedule；
  每个complete candidate最终由Q50.0做message matching、completion、SPM/DDR和runtime gate。
- User-level driver / named pipeline:
  Q51 planning session；`none`继续使用其确定性DDR/peer functional carrier，不进入本域枚举。
- Explicit non-goals:
  不选择compute replica、layout conversion、buffer count、worker、issue order、pipeline或winner；不把software relay path称作raw NoC
  route，不按bytes/hops冻结最大broadcast，不缓存actual IR，不保留旧rank/provider proposal或implicit cross-region SPM residency。
- Done criteria:
  same-region external boundary、same-Tile cross-region DDR、cross-Tile DDR/direct peer/relay、multi-piece partial overlap、unicast/partial/
  maximal multicast、fanout、gather及supported collective均有candidate actual witness；coverage/endpoint/relay/message/completion负例受typed
  validation；opaque target routing不伪造per-link load。旧complete-rank/global-relation/NoC provider中的topology、owner rotation、
  tree/ring、alias/lifetime/slice proof和tests逐项迁入current query/apply后，旧owner才能退役。

### Q50.H work-item分界

Q50.H由两个work items完成：

- `canonical-movement-plan`只关闭deterministic correctness carrier：每个nonempty program/constant shaped boundary使用external load；
  每个structured cross-region fragment使用Tensor→Tensor DDR transfer；ordinary/FD contribution若source Tile不是merge Tile则逐result/
  component使用DDR gather，同Tile同group不伪造movement；final root result沿pure non-structured SSA path可达function return时显式DDR
  publication。若一个execution result piece已经产生、但被后续pure tensor reconstruction完全覆盖且因此没有transfer/publication，plan用
  typed `ResultDiscardPlan`显式关闭该version disposition；discard不是movement action、没有payload resource，也不能代替缺失的carrier。
  plan引用G physical versions和D execution/fragment IDs，不选择route、relay、reuse、buffer或order。
- `movement-domain`在layout-domain后扩同一current合同，加入DDR/direct peer/relay/multicast/collective/reuse/occurrence完整域、proof、
  Core consumer和selected apply，并迁移/退役旧edge/action与post-hoc surgery。canonical DDR点是baseline correctness sibling，不是
  preferred transport或global winner。

当前`canonical-movement-plan`覆盖矩阵如下。shape用于覆盖exact payload与tail，不进入movement kind判断；tiny只用于单一故障负例。

| 覆盖类 | 代表输入 | canonical MovementPlan exact断言 | resource/downstream witness |
| --- | --- | --- | --- |
| program/constant input | rank>=3、1024/1025、all-16 Tile | 每个nonempty shaped external fragment恰有一个load到G destination version；scalar/exact-empty无action | payload resource逐action保留exact domain/type/source-destination Tile |
| chain/fanin/fanout/diamond | 1025级same/cross-Tile owners、multiple sinks | 每个structured `DemandFragmentId`恰有一个DDR transfer，source/destination physical version存在且domain与G fragment resource相同；共享source不按first use覆盖 | I/J可从action IDs推导stage lifetime与store-before-load依赖 |
| multi-result/view/multi-piece | rank>=3 1024/1025、result index不同、slice/reshape/pad及insert overwrite path | fragments不按node pair/bytes合并；publication沿pure support path覆盖terminal result；被证明完全覆盖的producer piece各有一个typed discard，仍缺carrier的observable result fail closed | output/stage/discard保持typed source/result identity；I只给explicit discard建立definition-local lifetime，不靠名字或空use猜测 |
| ordinary reduction | 整除/非整除、多group同merge Tile | 每个remote partial result一个DDR gather；merge-Tile local contribution无action；group/result身份完整 | merge execution可枚举all-and-only remote inputs，partial不变final owner |
| FD coupled state | rank-5/6、single/multi-K2 | 每个remote Maximum/Sum/Accumulator各一个gather，三component共享group但不合并成opaque payload；local contribution零movement | component type/domain与G resource及A requirement逐字段一致 |
| rank-zero/empty/merge-only | nonempty rank-zero tensor、exact-empty fragment、merge-only Tile | rank-zero可有0-rank load/transfer；empty无action；merge-only只接remote gathers | storage plan不为empty创建slot，也不因无root scope漏merge payload |
| typed failure | missing/duplicate physical version、fragment/resource domain不一致、owner/execution不存在、effectful output path | unsupported或compiler-contract failure且无partial plan，不fallback到peer/另一个owner | 修正输入可重新query，source IR byte-identical |

`movement-domain`施工前覆盖矩阵如下。小endpoint graph只用于transfer-graph独立oracle；production payload使用rank至少为3、主要维度
1024/1025/1031的exact resource，并区分target-routed endpoint transfer与compiler-emitted relay。

| 覆盖类 | 代表输入 | exact断言 | Core / apply witness |
| --- | --- | --- | --- |
| DDR与same-region边界 | program/constant input、same-Tile cross-region、rank-zero与exact-empty | external load/publication保持显式DDR boundary；same-Tile boundary只有DDR sibling；empty无action | `RepresentationState -> MovementState`，query零IR，public search推进到Storage |
| cross-Tile target-routed peer | rank-3 1024/1025、non-identity endpoints、opaque target routing | 每个cross-Tile payload保留DDR与direct peer siblings；direct只记录source/destination/message/version，不写raw path/link attr | selected emitter生成matching peer send/recv/wait obligations；opaque routing无per-link结论 |
| software relay与fanout boundaries | tiny 3--5 Tile endpoint set、多个独立fanout boundaries | 每个boundary relay chain逐hop显式source/destination且acyclic；one/multi-relay simple paths逐key匹配flat permutation oracle | relay Tile有typed recv→forward chain，不冒充hardware route；跨boundary shared multicast仍由保留donor承接后续迁移 |
| reduction与coupled components | ordinary remote partial、FD Maximum/Sum/Accumulator、local contribution | remote payload逐result/component完整枚举DDR/direct/relay；local contribution无movement；component不合并opaque payload | merge completion obligation逐payload存在，I/J可直接消费 |
| representation/version一致性 | primary/shared/per-use conversion、identity alias、mismatched layouts | H只引用G selected source/destination IDs；不插layout conversion、不按first use换source；unsupported transfer encoding无state | G builder与H emitter definitions all-and-only，boundary load仍由H owner绑定 |
| typed failure与资源隔离 | missing/duplicate action/version、coverage hole/overlap、unavailable endpoint、relay cycle/dead node、message mismatch | ExactRejection/Unsupported/Indeterminate/CompilerBug分类稳定；query不物化、不算SPM capacity、不选winner | failure零source mutation；修正plan可重试，actual gate后置 |

### H-1 专项调研：transport topology、movement boundary与payload合同

MLIR async合同要求所有依赖显式进入token/value，且“可并发”不保证实际并发；因此send/recv token、consumer visibility和buffer release
必须分别表达，不能由route顺序暗示。经典Dally--Seitz NoC理论用channel-dependency/virtual-channel证明router-level deadlock freedom；
NCCL graph search和TACCL则从actual topology选择ring/tree/communication sketch。这些工作都先区分**network route**与**collective/
forwarding algorithm**。本仓current `SimpleRoute`却枚举Tile邻接simple paths，再在每个中间Tile物化recv→send；它实际是software relay
path，不是可编程router path，命名和cost语义都必须纠正。

仓库硬件/IR事实进一步限定边界：current target为4x4 Tile grid，NoC有四方向独立收/发带宽；`wafer.tile.peer_*`和Direct DTE只携
任意available physical peer、bytes、message与token，明确不携raw route。Kcore DTE目的地址直接编码remote Tile，现有public accepted
subset是fixed-size unicast；raw broadcast/scatter寄存器和窄board case有证据，但current typed Tile/Instr primitive、target helper/ABI与
完整participant/completion合同尚未闭合，不能作为production capability。`TargetTopology::getCanonicalOnCardPath`自身也声明只是modeling
fact，不是target routing claim。因此current direct peer
transfer是end-to-end target-routed action；只有显式在中间Tile接收、保存并再次发送时才存在compiler-selected relay graph。

2026-08-23按current事实源重新分类如下；分类控制H的typed state，而不是只作说明：

| 分类 | current事实 | H中的处理 |
| --- | --- | --- |
| `supported` | `wafer.tile.peer_send/peer_recv`与Instr Direct-DTE fixed-size unicast、physical endpoint、message、token及独立DTE completion domain | 生成`TargetRoutedPeer`或由多条endpoint transfer组成的software relay/fanout；H只发token，J决定wait |
| `board-observed` | raw DTE broadcast/scatter的2/4/8/15 fanout、单目标source-gather、四源fanin，以及16-Tile Direct/Ring/ordered-tree collective历史correctness | 只作为未来typed capability或普通unicast action-DAG proposal的证据；不能直接产生raw collective op |
| `unknown` | Direct-DTE内部physical route、per-link contention/device phase、任意fanout/ragged scatter、8/15源fanin、subgroup barrier及未列participant/dtype/shape | 不建route/link/schedule事实，不按“保守同步”补wait/join，也不据此删candidate |
| `excluded` | current Tile/Instr/target ABI没有typed raw broadcast/scatter/source-gather collective primitive；cross-card transport也不在本任务 | 即使底层寄存器或历史board case存在，H仍不生成`QualifiedTargetCollective`；先扩typed IR/ABI与完整completion gate后才能开放 |

因此“qualified collective”不是一个布尔开关。current transport facts根本不声明这种capability，domain也没有对应state；只有participant、
payload mapping、alignment、status和completion等closed typed descriptor及其直接IR consumer一起加入时才扩同一schema。不得把
board-observed raw mode提升成current production support。

#### 两层topology facts

```text
OnCardTransportFacts
  endpoints: available TileId[]
  endpointTransfers: DirectPeerCapability[]
  physicalNoC: DirectedPhysicalLink[]
  targetRouting: OpaqueTargetRouting
               | DeterministicTargetRoute(query)
               | ProgrammableRouteDomain(query)
  relayCapability: receive / local retain / forward constraints
  acceptedPrimitives: fixed unicast + explicitly qualified capabilities
  targetLimits: payload/alignment/descriptor/resource facts

DirectedPhysicalLink
  id: DirectedLinkId
  source, destination: TileId
  bandwidth/resourceClass: immutable target fact
```

current production从Tile grid与unavailable endpoints构造双向directed neighbor links，并从target facts读取统一方向带宽；
`card_interconnect`是card-level C2C种类，不能拿来推导Tile torus。DirectPeerCapability覆盖同卡任意available endpoint pair，physicalNoC
只提供minimum-hop/cut下界；target routing为opaque时H/J不得声称某条exact link被占用。若未来target暴露asymmetric/broken link、VC、
programmable route或hardware multicast，先扩同一typed target事实/current IR合同；不能从4x4编号、日志或测试fixture猜。unit oracle可
直接注入line/ring/directed/asymmetric graph验证算法，不冒充current hardware能力。

#### Boundary与movement typed objects

D已经决定哪些uses由local stored/direct nested value满足、哪些是explicit boundary；compute replica不再是movement sibling。H按
`MovementBoundaryId`而不是旧node-pair edge建立：

```text
MovementPlan
  externalLoads: ExternalLoadPlan[]
  boundaryTransfers: DDRBoundaryTransferPlan[]
  reductionGathers: ReductionGatherPlan[]
  publications/discards
  peerGraphs: PeerTransferGraphPlan[]

PeerTransferGraphPlan
  actions: sorted all-and-only MovementActionId[]
  kind: TargetRoutedPeer | SoftwareRelay | SoftwareFanout | ExternalLoadFanout
  hops: MovementHop[]
  ddrRoot: optional ExternalLoadId

MovementResourceDescription  // query-local, not plan identity
  action: MovementActionId
  exactDomain: ExactIndexSet
  elementType
  source/destination Tile

Prepared payload piece
  graph actions + payloadSlice
  exact logical offsets/sizes
  source/destination/relay current-IR buffers
  checked physicalBytes
```

未被任何`peerGraphs`成员引用的action使用其显式DDR carrier；peer graph只保存一次，不在member action上复制。qualified target
collective只有在未来closed typed primitive存在时才作为另一种top-level movement owner加入，不能预留一个空enum值。

同一region内已local-bind的value没有H entry；同region但D选择external boundary、same-Tile sibling regions及program I/O用显式DDR
stage。cross-Tile boundary至少保留DDR sibling，并可选择direct peer或由多个direct transfers组成的relay graph。reduction gather只消费
A的contribution/merge requirement并移动typed payload；local combine仍是compute IR，H不能把算术藏进“reduce route”。

payload从A exact ownership intersection与G physical versions构造：每个logical piece先与source ownership相交，再经
`PhysicalLayoutRelation`/transfer realizability得到source/destination spans。所有payload logical domains的union必须恰等于required
coverage；非显式replication不得重叠，exact-empty不生成action。multi-piece/stride/padding保留分段，不能densify bounding box或按
`logicalBytes==physicalBytes`判断等价。transport encoding/version由完整G/H plan显式给出，不由apply插layout conversion。

`PeerTransferGraph`是compiler-selected communication algorithm，commit后由各Tile自己的send/recv/relay buffers和tokens自包含表示，
planning object销毁；Tile IR仍不保存route/tree attr。一个`TargetRouted` transfer没有programmable physical path，cost只使用endpoint
latency、minimum-hop/cut lower bound及unknown link contention。explicit relay edge若选邻接endpoints可确定至少一条physical link，若
选远端endpoints仍按target-routed处理。只有`ProgrammableRouteDomain`存在时才枚举router paths并用target提供的channel-dependency/VC
合同证明deadlock freedom；H不能自行套XY/DOR改变硬件行为。

#### 失败分类与scope

- boundary coverage hole/illegal overlap、unavailable endpoint、relay cycle、source/destination span mismatch、超target fixed payload limit
  是该MovementPlan的ExactRejection，witness绑定boundary、G versions、payload和topology facts；
- target没有所需primitive/encoding/relay contract是Unsupported；route/collective enumeration work耗尽是Indeterminate；缺少尚未赋值的
  G/H typed coordinate是Deferred；malformed upstream plan是compiler bug；
- link contention、descriptor concurrency、buffer capacity和issue order分别defer给F/I/J，不因H局部估价拒绝；
- 任一H rejection只删除同一physical action assignment，不回写A logical demand、B placement、D execution或G representation。

H query只读immutable plans/topology，不修改IR、构造relay TileRegion或运行transport verifier；默认无route dump、stats或timing。
production direct tests先覆盖mesh/unavailable/opaque-routing、boundary type/coverage、target-routed direct与explicit relay身份、DDR sibling及
五类failure；classic route/reuse算法、candidate apply和donor迁移分别由H-2/H-3闭合。

### H-2 专项调研：reuse classes、relay/collective exact domain与proposals

Steiner tree在一般graph上NP-hard；Dreyfus--Wagner对`k`个terminals的exact DP含`3^k`项。Kou--Markowsky--Berman通过terminal
metric closure的MST产生polynomial approximation。NCCL从topology搜索ring/tree graph，TACCL则用communication sketch约束MILP，并把
path、link order和chunk contiguity分阶段求解；TACCL也明确说明其routing阶段只给忽略contention/order的时间下界。这些方法适合产生
高质量H proposals，但没有任何一个允许“最短/最大broadcast”替代完整Wafer domain。

专项调研不能只记算法名字。下列primary implementation/paper给出的可迁移结论和Wafer边界分别是：

| 工作 | 实际方法 | Wafer采用 | 不能照搬 |
| --- | --- | --- | --- |
| [NCCL topology graph search](https://github.com/NVIDIA/nccl/blob/master/src/graph/search.cc)与[double binary tree](https://github.com/NVIDIA/nccl/blob/master/src/graph/trees.cc) | 在带宽受限topology path上递归搜索ring/tree与多个channel，并按硬件能力选择algorithm/protocol | ring、balanced/binomial/double-tree、多个chunk stream作为proposal family；endpoint与资源压力必须显式 | NCCL知道其transport path和GPU/NIC协议；不能把它的path、channel数或阈值当成TX81事实 |
| [MPICH recursive doubling](https://github.com/pmodels/mpich/blob/main/src/mpi/coll/allreduce/allreduce_intra_recursive_doubling.c)与[reduce-scatter + allgather](https://github.com/pmodels/mpich/blob/main/src/mpi/coll/allreduce/allreduce_intra_reduce_scatter_allgather.c) | 递归交换、递归halving/doubling及非2次幂participant的pre/post pairing | 形成普通chunk transfer/combine DAG proposal，并覆盖任意participant数 | MPI rank、datatype临时buffer和固定阈值不是本仓IR合同 |
| [SCCL](https://arxiv.org/pdf/2008.08708) | 用chunk的pre/post location、step/round及带宽关系定义有限collective synthesis问题，SMT搜索latency/bandwidth Pareto点 | 采用pre/post payload-state语义和tiny exact oracle；用topology automorphism做有证明的对称约简 | 不把SMT、同步step或`k`上界放进默认production legality，也不宣称全体算法最优 |
| [TACCL](https://www.usenix.org/system/files/nsdi23-shah.pdf) | 用logical-topology sketch缩域，并分离routing、per-link ordering和chunk contiguity | H只选payload传播图，J选order，G/I证明contiguity/buffer；每层保持typed依赖 | 不接受用户sidecar sketch，不把忽略contention的routing lower bound说成可执行schedule |
| [Blink](https://pages.cs.wisc.edu/~shivaram/publications/blink-mlsys2020.pdf)与[bandwidth-optimal pipeline schedules](https://arxiv.org/abs/2305.18461) | 以带权arborescence packing、bottleneck cut和chunk pipeline提高异构topology利用率 | cut lower bound、edge-disjoint/tree-packing seeds及多树chunk striping；只在route/capacity可证明时给exact link结论 | current target routing opaque，不能从physical mesh直接声称tree packing达到link optimum |
| [Bruck all-to-all](https://authors.library.caltech.edu/records/ypzfe-0bb45)与[Rabenseifner reduction](https://fs.hlrs.de/projects/rabenseifner/publ/myreduce_iccs2004_2long.pdf) | 在message startup、payload bytes、round数间取不同trade-off | Bruck、pairwise、ring、recursive exchange都作为同一domain的checked proposal | 算法适用条件必须由selected physical spans、local pack/combine和target primitive证明，不能按collective名字放行 |
| [Dally--Seitz](https://authors.library.caltech.edu/records/fd0yr-br438) | 用channel-dependency graph刻画router-level deadlock | 仅在target暴露programmable/deterministic route及VC/channel事实时验证raw route | current Direct DTE内部route不透明；software send/recv wait graph由Q50.0 actual whole-card verifier证明，不能套XY/DOR猜硬件 |

因此H不采用“算法枚举值 + 固定参数表”。经典方案是快速找到好点的构造器，完整语义由下面一个统一的payload-state问题拥有。

#### Exact reuse identity

query先按exact facts建立可重算`TransferReuseClass`：

```text
TransferReuseClass
  logicalValue: RegionValueVersionId
  logicalDomain: ExactIndexSet
  sourcePhysicalSpans: PhysicalSpan[]
  transportEncoding: PhysicalEncoding
  destinationRequirements: (MovementBoundaryId, PhysicalVersionId)[]
  temporalOccurrences: ExactOccurrenceClass[]
```

只有logical value/domain、source bytes、transport encoding都相同的destinations才能共享一个payload；相同byte count或shape不够。
temporal occurrence从E scope/order与index relation证明`PerUse`、`PerInvocationClass`或`OncePerRegion`，每种都是普通H choice；一次接收后
跨uses保留还必须由I lifetime/slots和J completion/order关闭。reuse class本身不进candidate identity，selected destination grouping、
payload/version和occurrence choice进入`MovementPlan`。关闭reuse analysis或改变其发现顺序不改变unicast/DDR/exact tree domain。

#### Destination partitions、source owners与relay arborescences

一个reuse class的destination set先用anchor recursion惰性枚举全部set partitions；每个block是一份可共享transfer，故all-singleton、
partial和maximal multicast都可达。每个block再枚举A proof允许的全部source physical versions/owners；replicated owner selection是H
choice，不按nearest owner或Tile id冻结。

对固定root、terminal set和current endpoint-transfer graph，software broadcast tree的exact successor直接枚举canonical parent maps：

```text
enumerateOutArborescences(root, terminals, availableRelays):
  lazily choose an active subset of optional relay Tiles
  nodes = {root} union terminals union activeRelays
  for node in stable (nodes - root):
    choose exactly one parent with a legal DirectPeerCapability
    prune a partial parent map as soon as it closes a directed cycle
  require every node reaches root
  require every active relay reaches at least one terminal descendant
  canonicalize by the complete (node -> parent) semantic map
```

每个rooted arborescence有唯一node set和parent map，所以只出现一次；destinations可以同时forward。star就是direct unicast set，chain、
balanced tree、partial relay及使用non-destination Steiner Tiles都在同一domain。gather使用方向相反的in-arborescence，sink为merge/
publisher Tile。单payload graph必须acyclic；ring collective的物理participant cycle通过显式round/payload-state DAG展开，不与relay cycle
混淆。

current DirectPeerCapability是available endpoints complete digraph，因此exact domain允许far endpoint edge；以physical mesh adjacency
限制parent只是一类proposal。若未来target只允许某些direct pairs，domain自然读typed capability。DDRStage始终作为独立sibling，除非
target合同明确没有card DDR path。

#### 统一的Tile通信问题、完整域与correctness state

只有A/D产生typed multi-participant requirement时才建立collective-shaped问题，但H核心不按collective名字分支。每个boundary/reuse class先
变成同一个有限问题：

```text
TileCommunicationProblem
  participants: TileId[]
  chunks: PayloadChunk[]
  initialStates: (TileId, PayloadStateId)[]
  requiredStates: (TileId, PayloadStateId)[]
  endpointTransfers: DirectPeerCapability[]
  legalCombines: CombineRelation[]
  targetFacts: endpoint / descriptor / alignment / topology facts

PayloadState
  chunk: PayloadChunkId
  logicalDomain: ExactIndexSet
  physicalSpans: PhysicalSpan[]
  origins: exact OriginSet
  combineWitness: upstream-approved tree/order when required

TileCommunicationPlan
  actions: TransferAction | CombineAction | QualifiedCollectiveAction
  dependencies: ActionId -> ActionId[]
  terminalBindings: RequiredStateId -> PayloadStateId
```

`PayloadChunk`来自exact logical domain与G-selected physical spans。外层`ChunkPartitionDomain`惰性覆盖target transfer granularity上全部连续
partition，包括不等长tail；chunk数量上界只能来自payload原子数、descriptor/message字段或其它typed target limit，不能是经验常数。
多个chunk只有在同source、同destination且physical spans已证明可由一个selected transport version连续发送时才能coalesce；否则Bruck式
聚合必须有G提供的显式pack/unpack version，H不能隐藏local copy或改变layout。

非combining payload的`OriginSet`为singleton。需要local combine时，`CombineRelation`由上游semantic op/interface给出可接受的input
partition与结果witness；H只允许disjoint origin sets合并且结果取exact union，不自行发明或放宽运算规则。一个transfer复制同一
`PayloadState`，一个combine严格扩大origin coverage。terminal必须逐`RequiredStateId`证明chunk/domain/origin及destination exact相等。

完整successor按AND/OR proof DAG工作，而不是先选Ring/Tree标签：

```text
proveHave(tile, requiredState, active):
  if initialStates contains (tile, requiredState):
    return Initial
  if (tile, requiredState) is in active:
    reject this derivation cycle

  alternatives = []
  for source in stable participants where source != tile:
    if endpointTransfers allows source -> tile:
      alternatives += Transfer(
          proveHave(source, requiredState, active + current), source, tile)

  for each legal disjoint partition inputs -> requiredState:
    alternatives += Combine(
        [proveHave(tile, input, active + current) for input in inputs])

  canonicalize equivalent proof DAGs by typed states and action edges
  return alternatives

enumeratePlans(problem):
  choose the stable first unsatisfied terminal
  expand proveHave for that terminal
  hash-cons identical subproofs so multiple terminals may share forwarding work
  yield only plans whose every action contributes to a terminal proof
```

这个域有限且不漏掉有用计划：chunk/state/tile集合有限；transfer必须让一个新Tile获得尚未拥有的state，combine必须严格扩大有限
`OriginSet`，active-set禁止依赖环，因此每条derivation终止。任意正确计划中不贡献terminal的新action可删除；同一Tile重复取得完全相同
state只保留第一次，不影响其后多个send；剩余irredundant plan可按action依赖拓扑序重排成上述proof DAG。故完整性是相对于
`ChunkPartitionDomain + endpointTransfers + legalCombines + qualified target primitives`的明确有限合同，不声称覆盖硬件合同之外的任意
packet程序。

经典方案都只是该域中的普通结构：

- broadcast/fanout是共享subproof形成的out-arborescence；gather/fanin是反向in-arborescence；
- all-gather是每个origin chunk在全部participants上的terminal集合；all-to-all是per-origin、per-destination的不同terminal集合；
- reduce-scatter让不同chunk的full-origin state终止于不同owner，all-reduce再为每个full-origin chunk追加all-gather terminals；
- ring是按participant cycle展开后的chunk-state DAG，recursive doubling/halving是distance classes形成的exchange DAG；
- tree/double-tree/multi-tree是不同chunks共享或分摊arborescence；二维row/column方案是先完成一维subgoals再并发完成另一维subgoals；
- participant graph可以含cycle，但每个具体chunk/round的action dependency必须acyclic。硬件route deadlock与软件event deadlock分别由
  target route verifier和J负责，不能因“Ring看起来有环”误拒绝，也不能因proof DAG无环就宣称router deadlock-free。

ring order是participant Hamiltonian cycle的canonical permutation：固定最小semantic Tile为首，只有完整topology automorphism证明时才
quotient反向/旋转。tree local-combine不代表fabric计算；combine始终是explicit local work。

#### Proposals、lower bounds与work control

proposals全部由exact constructor验证为domain member：

1. DDR、direct star、canonical source owner、nearest-owner star；
2. physical adjacency graph的BFS/shortest-path tree、Kou metric-closure Steiner、degree-constrained balanced tree；terminal很小时可在proposal
   budget内运行Dreyfus--Wagner，它不是默认legality check；
3. binomial、balanced、double-tree及edge-disjoint/tree-packing chunk stripes；current opaque routing下它们只是geometry-aware seeds，
   只有future exact route/link-service facts才能声称link-disjoint或bandwidth-optimal；
4. topology-derived Hamiltonian/serpentine ring及反向ring，ring reduce-scatter+all-gather和pipelined ring；存在unavailable Tile时必须从
   induced participant graph重新构造，不能硬编码16-Tile ring；
5. recursive doubling、recursive halving+doubling及非2次幂pre/post pairing；这些构造器读participant集合和typed spans，不读rank名字；
6. 规则或可证明factorized participant subgraph上的row-then-column/column-then-row broadcast、gather、all-gather和all-to-all，另含
   row/column panel fanout proposal；M/N/K或operator名字不是触发条件，触发事实是需求图与physical embedding；
7. Bruck-style aggregate exchange与round-robin/pairwise all-to-all；只有selected contiguous bundle或显式pack/unpack work完整时接纳；
8. TACCL-style in-process typed sketch（allowed endpoint edges、proven symmetry、chunk class）驱动bounded suggestion，以及cut-aware source
   rotation、reuse-aware partial multicast；没有外部JSON/sidecar或default MILP/SMT。

proposal bank不会先按payload大小硬选一个family，也不产生local winner。message count、injected bytes、relay copies、critical dependency
depth、endpoint issue pressure、明确staging actions和可证明topology bounds交给I/J/Q51作结构与performance排序；staging的实际SPM allocation、
lifetime和capacity只在complete candidate actual gate中判定。同一个workload的不同boundary可选择不同family。

对opaque target routing只使用sound bounds：

```text
dilationLB = bytes * max shortestHop(root, terminal)
treeWorkLB = bytes * ceil(metricMST(root + terminals) / 2)
messageLB = pairwise-decomposition receiver count * target setup lower bound
cutLB = sum over a proven edge-disjoint cut family of mandatory crossing bytes
```

`MST/2`来自任一Steiner tree加倍可形成覆盖terminals的tour；它不高估optimum。broadcast shared edge不能按destination重复计bytes；gather
的distinct origins则按cut上必须跨越的独立information累加。deterministic target route存在时可加入exact directed-link work；opaque
routing时per-link load/contention保持Unknown，由J不能据此局部prune。

另外对每个directed physical cut `C=(S,V-S)`按required origin/terminal relation计算必须跨cut的信息量；以该方向link capacity之和得到
route-independent cut lower bound。它能让B/Q51较早识别把高复用producer放在需求几何中心附近的优质placement，但只作为admissible
bound/proposal排序，不能让H反向修改B。若target只给uniform point bandwidth而没有可验证sustained/capacity合同，则只保留crossing bytes
和hop count，不合成伪时间。

exact successor不做local Pareto删除。proposal solver只有在保留payload/coverage、source/destination versions、relay/staging、origin
state、completion dependencies及对未来I/J有影响的resource facts时才能返回bound；预算耗尽返回unranked checked suggestion或
Indeterminate，destination partitions/parent-map/ring siblings仍由Q51可达。默认路径不运行Dreyfus--Wagner/MILP、打印topology或记录
statistics。

设reuse class有`d`个destinations、可选relay `r`个、origin数`o`和chunk partition数`c`：destination partitions最坏为`B_d`；一个active
node set的parent assignments最坏为`(|nodes|-1)^(|nodes|-1)`再经arborescence过滤，relay subsets另有`2^r`；origin bipartitions、chunk
compositions和ring orders分别呈指数或阶乘增长。精确域只能lazy/output-sensitive，不能承诺多项式。BFS/row-column/recursive/ring seeds
分别为线性或`O(P log P)`量级；Kou为polynomial，Dreyfus--Wagner、Held--Karp ring、SMT/MILP和tree packing只在tiny oracle或显式
bounded proposal work中使用。query-local memo按extensional payload/topology descriptor共享，不强迫不同boundary选择同一graph。

完整4x4 mesh的几何对称只在automorphism同时保持available mask、endpoint capability、placement、payload需求、target resources时用于
quotient。16个Tile可共享domain/proposal计算和canonical representative，但每个Tile的assignment、buffer、event仍独立；有hole、非对称
需求或不同resource facts时automorphism自然缩小或消失。并行生成independent proposal不按完成先后影响顺序，所有可观察顺序使用完整
semantic key。

### H-2 Gate

- 有界穷举oracle对2--6 endpoints平铺destination set partitions、source owners、relay subsets和parent maps；与production exact plans逐key
  相同且无duplicate，覆盖star/chain/balanced/Steiner destination-relay/asymmetric direct capability；
- line、2-D mesh、ring、unavailable endpoints、directed/asymmetric link fixtures中，每个tree/ring edge均有typed endpoint capability；
  current opaque target routing从不输出exact internal link sequence；
- reuse property覆盖same bytes/different domain、same domain/different physical spans、fanout shared/split、temporal invariant/non-invariant、
  local+remote partial overlap；关闭reuse只改proposal顺序不改domain；
- broadcast逐destination恰一origin，gather/tree combine逐origin恰一次，ring每round/chunk状态与independent finite-state oracle一致；cycle、
  duplicate/lost origin、relay dead-end和coverage overlap/gap拒绝；
- shortest-path/Kou/Dreyfus有界proposals都是exact members；Dreyfus与brute minimum tree一致，`dilation/MST/2/cut`从不高于brute optimum；
  关闭/反转proposals不改变exact set或Q51有界穷举optimum；
- independent finite-state oracle对2--4 Tiles平铺所有单调transfer/combine proof，逐plan比较initial/terminal state、origin cover和action DAG；
  production successor与oracle无缺失、无duplicate，classic family只是其中的命中样例；
- ring、binomial/double-tree、recursive exchange、非2次幂pairing、row/column、Bruck/pairwise分别有plan-level witness；这些witness改变proposal
  顺序但不改变统一successor定义，ragged chunk与non-contiguous bundle必须显式通过或拒绝；
- 同一problem经过合法topology automorphism重标号后plan/cost key同构；增加unavailable Tile只删除真正使用该endpoint的actions，不能
  恢复hardcoded 16-Tile order；opaque-routing fixture没有exact link schedule，deterministic-route fixture才验证link load/CDG；
- work-limit不把未完成tree/ring search变成NoSolution；无default MILP/heavy Steiner solver/logging/statistics。H-2只验证query，actual IR、
  lifetime/alias/slice proof和donor删除由H-3闭合。

### H-3 专项调研：complete-candidate movement IR、effect/lifetime proof与donor迁移

MLIR async token只表示producing operation完成；consumer readiness、source lifetime和resource effects仍须通过显式token/use/effect关系
连接。MemoryEffectOpInterface的Resource适合区分SPM、DDR、communication等资源，但官方也明确说它不负责address/size级alias；exact
slice overlap必须由alias/range analysis补充。因此H actual不能只数send/recv或立即插await，而要同时保留typed payload spans、SSA roots、
token dependencies和range lifetime proof。

current `applySelectedDataMovement`是在已建好的DDR CardModule上做late surgery：按node/shape/layout寻找buffer和store/load，删除或替换
它们，必要时`replaceAllUsesWith`、新建alloc/dealloc、擦layout op，再为relay临时建立private function/TileRegion；每个send/recv后立即
await，message identity用edge/Tile/fragment arithmetic拼出。它既让H改变G versions和J schedule，也依赖baseline IR恰好有可替换
load/store，必须整体退出。

#### Prepare与direct construction

C outer transaction在mutation前把G--K完整winner关闭成：

```text
PreparedMovement
  ddrStages: PreparedDDRStage[]
  transfers: PreparedPeerTransfer[]
  relays: PreparedRelayWork[]
  localCombines: PreparedCollectiveCombine[]
  messages: PreparedMessageIdentity[]
  coverageProofs: PreparedPayloadCoverage[]
  bufferBindings: BufferPlan-provided PhysicalVersionId -> BufferId
  eventBindings: EventSchedule-provided issue/wait/use/release points
```

prepare逐项验证H payload、G physical versions、I buffer ownership和J event dependencies；为sorted `(CardId, TransferId, PayloadId,
OccurrenceId)`分配checked dense communication/round/slice字段，稳定identity不来自operation ordinal、pointer或symbol名。Card-shared DDR
stage及relay-only region进入C的symbol/region closure；all available TileModules仍一次创建，H不能另造Module/Func取得anchor。

actual由职责分离的builders协作：

```text
emitDDRStage(preparedStage, physicalVersions, eventBuilder, rewriter):
  bind source/destination exact views
  emit destination-style StorageStore/StorageLoad at J-selected points
  publish resulting PhysicalVersionId and completion events

emitPeerTransfer(preparedTransfer, physicalVersions,
                 bufferBindings, eventBuilder, rewriter):
  bind source/destination/relay exact views
  create matching CommPeerSendOp and CommPeerRecvOp
  register their tokens under TransferCompletionId
  let J eventBuilder place waits before first read/forward and before release

emitRelayWork(preparedRelay, ...):
  receive into the planned relay buffer
  consume its recv-completion before local combine or child sends
  keep relay buffer live through every child-send completion
```

H从空target subtree直接创建selected movement，不先创建DDR sibling再删除。layout/transport conversions只由G
`PhysicalVersionBuilder`按planned IDs创建；allocation/rotation只由I；wait位置、issue order和local combine order只由J。H emitter只返回
created op/token与plan ID的typed relations，不能更新其它stage map或在失败时选另一个action。

same-region externalization和same-Tile cross-region stage都使用explicit DDR value；SPM SSA不跨TileRegion。cross-Tile direct/relay每个
Tile拥有独立SPM root/view，任何destination兼任relay时仍先recv-complete再forward。tree/ring只是transfer/token graph；local reduction/
combine由selected compute op显式存在，fabric不隐藏算术。qualified hardware collective若未来加入，仍必须产生等价typed completion与
payload relations，不能绕过CardExecutable verification。

#### Actual movement verification

一个card-scoped `MovementMaterializationVerifier`在所有Tile payload发出后一次检查，避免每leaf重复walk CardModule：

1. plan relation total/unique：每个DDR stage、transfer、payload、relay version、message和completion恰有actual owner；IR无额外H op；
2. logical/physical cover：从actual subview/layout relation重算每个source/destination span，union all-and-only且origin multiplicity正确；
3. alias/range：source/relay/destination roots及overlap使用ViewLike、Bufferizable、current range analysis证明，allocation-root相同不能代签
   exact subview；
4. lifetime/effect：source定义后到send completion之间无overlapping write/free，destination在recv wait前无read，relay在最后child wait
   前不overwrite/free，DDR store completion precedes matching load；
5. message/event：send/recv方向、physical peers、bytes、message key和dynamic occurrence一一匹配；token由selected wait恰消费，wait不
   完成NCC worker；
6. publication：required output writer读取closed final version，范围compact/exact，writer后无覆盖；partial/ring origin state无缺失、重复、
   overlap或gap。

这是selected-movement stage verifier，不是第二个planner；Q50.0随后仍在memory-planned Instr/CardExecutable上提交Direct-DTE endpoint、
remote offset、FSM/status/resource binding。两者消费不同IR层且共享typed message/payload definitions。

任一prepare/emit/verify失败由C新CardModule subtree guard整体擦除，source保持不变；candidate materialization bug不返回H domain换plan。
只有Q50.0给出带完整actual witness的typed rejection时controller才继续其它candidate。default path
不打印IR、payload table或统计。

#### Donor capability migration matrix

| Current / donor能力 | 终态owner | 必须迁移的witness | 退役条件 |
| --- | --- | --- | --- |
| current `CardDataMovementDomain(edge,destination)`与`DataMovementKind` | H boundary/payload/transfer graph；D local/replica | DDR/direct/relay/mixed fragments及domain oracle | Retained/Refetch/Recompute、edge/node role和trial/epoch inputs零残留 |
| current `SimpleRoute` | H explicit relay parent-map successor | chain/tree/Steiner relay、cycle/dead-end negatives | route命名与“adjacent path=hardware route”假设删除 |
| current `applySelectedDataMovement` post-hoc store/load surgery | C direct construction + H narrow emitters | same-region DDR、peer、relay、gather/ring actual | find-by-node/shape、replaceAllUses、erase baseline load/store和late alloc/dealloc删除 |
| current per-op immediate `async.await` | J selected event order + H transfer tokens | recv-before-read、send-before-release、relay forward、overlap-capable case | H不自行放wait或串行化所有movement |
| current apply-time layout conversion | G physical-version producer DAG | different transport/destination encodings、shared conversion once | H不创建/erase LayoutMaterializeOp或选择source layout |
| current message arithmetic fromedge/Tile/fragment | prepared semantic message allocation | stable input-order/hash/parallel completion、ID exhaustion | operation ordinal/pointer/symbol/hand-coded multiplier零残留 |
| donor boundary direct/receive-forward fanout与owner rotation | H source/group/tree domain + proposal | replicated/partitioned boundaries、owner diversity、read-only query | opaque provider point/complete-rank clone API删除 |
| donor intermediate spill-cut/output publisher proof | D/H boundary + actual verifier | non-equivalent value、RMW init、required publisher、output writer | post-Instr opportunity matcher和spill-pattern rewrite删除 |
| donor alias overwrite/dealloc/use-before-reload/nested use tests | H actual alias/lifetime verifier | 每个negative保持atomic rejection | 不因旧test未注册或source删除而丢失 |
| donor tree/ring partial proof | A origin/merge requirement + H transfer state + J events | combiner input relation、dead round、origin multiplicity、tail/gap/overlap/misalignment、writer覆盖 | old all-rank matcher/provider删除；不增加额外语义gate |
| current peer/reduction apply tests | H query+actual owners | multi-owner fragments、local+remote、partial/max multicast、gather | 旧module-return fixture改接C transaction和typed version IDs |

#### H-3 Gate

- actual count/relation tests覆盖same-region external DDR、same-Tile cross-region DDR、far direct peer、adjacent relay chain、fanout star/
  partial/max tree、local+remote multi-piece、gather tree、ring rounds和output publication；每个plan ID all-and-only出现；
- tokens/effects测试至少有一个合法overlap plan不被H立即await串行化，并由J-selected waits证明recv visibility与send/relay lifetime；
- donor negative matrix全部迁移：alias overwrite、early dealloc、use-before-recv/reload、nested/unproven alias、allocation-root-only slice、
  missing/duplicate message、dead round、origin duplicate、ring tail/gap/overlap/misalignment、noncompact/late-overwritten writer；
- failure injection覆盖prepare首/末payload、mid relay、message allocation、emit后actual verifier；source文本/ops/symbols不变且Card subtree
  无残留；custom/generic roundtrip、verify-each、Tile→Instr和CardExecutable message/range/completion gates执行；
- source/call-tree gate删除old kinds/types/SimpleRoute/apply surgery、dormant NoC provider与未迁移tests；query/prepare/emitter/verifier各自独立
  file/library，H不include solver/driver或G/I/J internals；
- Q49.P deterministic carrier与search winner复用同一payload/direct emitter但policy controllers独立；轻量source→CardModule→Q50.0→
  package/no-card通过。H施工不运行重型LLaMA search。

H只有上述query、actual和迁移gate全部闭合才完成；旧“current basic apply已存在”或若干send/recv计数不能代签alias/lifetime/slice proof。

### `movement-domain`实现与闭合证据

- `MovementPlan.peerGraphs`现在让每个peer payload graph只保存一次sorted all-and-only action set与endpoint edges；未被graph引用的action
  是DDR sibling，不在每个member上复制同一graph。external reuse group另用typed `ddrRoot`指定唯一seed load，逐一枚举组内每个destination
  作root。一个endpoint edge始终表示target-routed transfer；`SoftwareRelay`/`SoftwareFanout`/`ExternalLoadFanout`只描述compiler显式
  接收再转发的action graph，不保存raw
  route、link load、cost、operation ordinal或手写message arithmetic。
- `MovementDomain`先按logical value、exact Presburger domain、source physical version、dtype和transport encoding建立reuse class；随后用
  restricted-growth sequence惰性枚举“任意DDR子集 + 其余destination的全部set partitions”。每个peer block再枚举任意active relay subset
  与全部legal parent maps，只保留单root、每node恰一parent、全node可达且每relay贡献terminal的rooted arborescence。这个统一域同时包含
  direct、chain、partial/max fanout和Steiner relay；它不按bytes/hops预删合法点。
- current `MovementTransportFacts`只保存available endpoints、opaque directed endpoint-transfer capability和DDR availability。
  current TX81 builder产生all-pairs opaque endpoint capability；directed/asymmetric fixture可注入更小能力图。physical mesh没有被冒充成
  router route。current facts不提供无typed descriptor的collective字段，domain因而不可能生成空壳state。
- query先给maximal reuse arborescence与all-singleton endpoint graphs作为checked proposals，再进入同一个raw successor；proposal关闭或
  去重不改变raw集合。三个destination的独立组合oracle验证`sum C(3,k)*Bell(k)=15`个DDR/partition形态，maximal block按Cayley公式有16个
  rooted arborescences，production逐key无duplicate；program-input fanout另证明每个destination都可作唯一DDR root，未分组load仍各自DDR。
- remote ordinary/coupled gather复用同一per-payload DDR/direct/relay graph；local contribution仍无action。没有上游typed
  multi-participant state/combine requirement时不凭collective名字构造ring/all-reduce；将来出现该requirement时，ring/tree只生成普通
  transfer/combine DAG proposal，local combine仍是显式compute。
- `prepareSelectedPeerGraphs`对整份selected plan做mutation-free all-and-only preflight并按semantic action set去重；actual SPM memref
  逐endpoint验证physical bytes/layout、单root、acyclic reachability和terminal后，按sorted action set分配candidate-local dense message
  identity；external group先生成唯一typed DDR root load，再由`emitPreparedSelectedPeerGraphs`生成matching `tile.peer_send/recv` token。
  `verifyTokenOnlySelectedPeerGraphs`逐graph/root/hop/message/token验证，并把任何H层immediate `async.await`作为错误。
  caller拥有candidate transaction，失败不换movement、不修layout、不改buffer。
- legacy `DataMovementApply` post-hoc store/load/layout/alloc surgery及其immediate-await实现已经从source和CMake删除；legacy joint-domain
  materializer遇到old Peer/Refetch assignment时fail closed。current H query/emitter不include old `SimpleRoute`、node/edge movement kind或
  shape/name lookup。旧query-only coupled/buffering cluster仍由后续I/J/K迁移其独有测试后统一退役，不能重新取得actual owner身份。

fresh定向证据覆盖rank-3 `1024/1025/1031`：single payload的DDR/direct/全部relay permutation、directed-only relay、layout mismatch、
same-Tile DDR、ordinary reduction gather、external every-root fanout、split/partial/max fanout、15个partition形态、16个maximal arborescence、
input-order稳定，以及current domain中不存在raw collective state；selected actual覆盖aligned/ragged external DDR-root fanout、ordinary
fanout、1024/1025 two-piece payload的all-and-only message slices和1031 reduction relay的matching token、零immediate await，以及missing
binding、layout mismatch、cycle和人工immediate-await负例。Core proposal/raw去重、search到Storage的直接consumer和完整fresh回归在本项
收尾结果中记录：本项及ordinary host unit `915/915`通过（另一个尚未归本项的attention production closure不参与），configured lit
`225/225`、4个public link smoke、source/IR organization和diff检查通过。

## Q50.I：Buffer and Rotating Slots

### Pipeline contract

Pipeline position:
- Upstream IR / input:
  immutable selected TensorProgram、Q50.D execution/value bindings、Q50.E temporal scopes/occurrences、Q50.G physical-version graph、Q50.H
  exact DDR/peer/relay movement plan、Q50.F foundation facts及immutable target SPM/resource limits；尚未选择event/resource schedule或
  stage-pipeline execution structure。
- Current stage responsibility:
  为每个physical version选择storage object/alias binding，并为具有同一exact recurrence coordinate的versions/transfers建立single或
  rotating slot-family domain；推导必须由J满足的def/use/completion/release ordering requirements。query只产出typed `BufferPlan`；
  candidate apply才建立allocations、views、loop-carried slot rotation和release。
- Output IR / files:
  query不写IR，输出typed storage/slot bindings及schedule requirements；derived structural lifetime/interference facts可重算。candidate actual
  IR用memref SSA roots/views、SCF iter args和dealloc/release自包含表示，不写buffer recipe attr。query结果不含IR pointer、Location、
  shadow timestamps或默认统计。
- Downstream consumer:
  Q50.J foundation从D--I计划建立event/resource DAG；Q50.K选择serialized/pipelined execution structure。K若改变occurrence/lifetime结构，
  对应I/J choices失效并由Q51重算；J closure最终关闭wait/order/release。每个complete candidate进入一次Q50.0
  SPM/message/completion gate。
- User-level driver / named pipeline:
  Q51 planning session；`none`保持serialized且不枚举buffer域。
- Explicit non-goals:
  不选择worker、issue order、stage pipeline、route、winner或overlap收益，不clone/lower partial state，不把slot count写回logical
  boundary/version，不用3作为永久上限，不从schedule estimate或materializer失败推断parent spatial/temporal rejection。
- Done criteria:
  exact domain覆盖fresh/reused/in-place storage、single/double/triple及finite upper bound内更多slots、不同rotation coordinates和required
  orderings；independent oracle一致。actual double/triple/multi-slot、tail、Direct-DTE issue/wait、alias和common-occurrence正负例通过；
  旧edge `bufferCount`、actual-loop discovery、local winner和fixed `[2,3]`上限删除；K/J invalidation与联合proposal有production consumer。

### Q50.I work-item分界

Q50.I由三个work items闭合：

- `canonical-storage-plan`为canonical G/H及Serialized point建立correctness-first single-slot storage。每个nonempty physical version各有一个
  fresh object；external load和DDR destination直接定义对应version object，remote reduction gather另有一个destination staging object，
  publication只读取source object。H中显式`ResultDiscardPlan`对应的execution result仍保留实际output storage，其definition与唯一local
  disposition均绑定同一execution；除此之外任何缺失carrier/use仍为typed failure。derived lifetime只记录typed execution/action
  definition与uses，不保存order、timestamp或offset。
- `storage-domain`在movement-domain后扩展同一current合同，枚举fresh/alias/reuse、exact view、`1..U` slot family/rotation、order
  requirements、Core consumer及selected construction；canonical fresh/single点只是一个合法成员。
- `structure-specific-storage`在K选择后从新的occurrence/live distance重闭I；Serialized收敛回single slot，Pipelined重新证明multiplicity、
  rotation和reuse。它复用storage-domain算法，不建立第二套buffer schema。

当前work item不引入`SlotFamilyId`或值恒为1的multiplicity字段：一个`StorageObjectPlan`就是一个独立single slot。后续只有出现真实
multi-occurrence consumer时才扩slot-family合同。`StorageObjectId`在canonical点由`PhysicalVersionId`或`ReductionGatherId`派生，不能使用
ordinal、pointer、operation name或输入顺序。

```text
Pipeline position:
- Upstream IR / input:
  canonical RepresentationPlan/resource descriptions、MovementPlan/resource descriptions及SerializedExecutionPlan；全部physical version、
  root/merge execution和movement action已有stable typed identity，尚无event/order、alias/reuse、rotation、offset或actual IR。
- Current stage responsibility:
  为每个physical version建立一个fresh single-slot object，为每个remote reduction gather建立一个destination staging object；验证所有
  producer/consumer execution、movement action和result discard disposition闭合，并派生每个object的typed definition/use lifetime。
- Output IR / files:
  query-local BufferPlan及StorageResourceDescription/StorageLifetimeDescription；不修改IR、不写attr或文件，不保存timestamp/offset。
- Downstream consumer:
  canonical-schedule从definition/use关系建立source-order、worker0依赖；attention-work-projection和canonical feasibility读取object
  resource/domain。后续storage-domain原位扩current schema，structure-specific-storage在固定K后重闭。
- User-level driver / named pipeline:
  无独立pass、pipeline或CLI；none与search canonical planning prefix静态调用同一constructor。
- Explicit non-goals:
  不选择alias/reuse/in-place、slot multiplicity/rotation、event/order/worker、SPM/DDR offset、route或winner；不扫描actual loop，不调用
  allocator、SelectedBufferMaterialization或StagePipeline，不把external program output伪造成Tile-local object。
- Done criteria:
  1024/1025 all-16、diamond/fanout、ordinary/FD local与remote merge、rank-zero、exact-empty/scalar和输入顺序扰动均得到all-and-only
  objects/bindings/lifetimes；每个remote gather恰有一个staging、local contribution没有staging；每个explicit discard恰有一个
  definition-local lifetime且不会生成schedule edge；missing/duplicate version/action/discard/execution、carried-and-discarded冲突、resource
  mismatch和unclosed lifetime为typed failure；source IR不变，canonical-schedule可直接消费，fresh build/unit与organization通过。
```

| 覆盖类 | 代表输入 | 必须断言的BufferPlan/lifetime | 直接下游witness |
| --- | --- | --- | --- |
| external load/publication | rank-3、1024/1025、all-16 Tile | 每个boundary/result physical version一个fresh object；load定义boundary，execution读取；execution定义result，publication读取；无output staging | schedule建立load→execution→publication，resource保留exact tail/type/encoding |
| structured diamond/fanout | 1025级multi-root/multi-sink | source result object由producer execution定义并被每个DDR action读取；每个destination version object由对应DDR action定义并只交给destination execution | shared source lifetime覆盖全部transfers，destination lifetime不按node pair或first use折叠 |
| ordinary reduction | rank>=3、1024/1025，local+remote contributions | 每个partial version一个object；remote gather读取source并定义唯一staging，merge读取staging；merge-Tile local partial直接由merge读取且无staging | schedule可生成producer→gather→merge及local producer→merge两类依赖 |
| FD coupled state | rank-5、1024及1025/1031 K2 | Maximum/Sum/Accumulator每个physical version独立；remote component各有staging，local component直接进同一merge；merge component object在merge内定义/消费 | attention projection按typed component/group/execution读取，不合并opaque payload |
| support/multi-result/rank-zero/discard | pad/support、multiple result、1024/1025 insert overwrite、nonempty rank-zero、exact-empty/scalar | support/result各自一object并绑定正确execution；explicitly discarded result的definition/use为同一execution；rank-zero保留0-rank；empty/scalar不造object | schedule不为discard造伪action/edge；actual materialization只为真实nonempty object创建allocation并记录owner |
| typed failure | duplicate/missing representation resource、unknown serialized execution、action/version binding mismatch、duplicate/missing movement resource、duplicate discard、carried-and-discarded冲突、未定义boundary或无合法disposition object | `BrokenStoragePlan`准确区分version/action/discard/execution/resource/lifetime合同错误；不返回partial plan | 修正输入可重新query，source IR byte-identical |

`storage-domain`施工前覆盖矩阵如下。小version/occurrence graph只用于独立coloring/rotation oracle；production resource使用rank至少为3、
主要维度1024/1025/1031，并且slot upper bound只来自typed occurrence/selector表示范围。

| 覆盖类 | 代表输入 | exact断言 | Core / construction witness |
| --- | --- | --- | --- |
| fresh与identity alias | primary、layout conversion、G exact identity alias、rank-zero | 每个nonempty physical version恰一binding；fresh各自object，identity alias强制同一object/SSA且不造allocation | `MovementState -> InitialBufferState`，query零IR，public search推进到EventResource |
| proven reuse与order requirement | 2--6 same-Tile versions、proven-disjoint与schedule-dependent pairs | fresh sibling始终存在；只有exact compatible/proven pair产生reuse；schedule-dependent reuse携typed MustPrecede requirement，不用current order判定 | J消费hard edge，I不选worker/order；unknown overlap无reuse state |
| slot family `1..U` | single/multi-axis occurrence、trip upper 4/5/1025、rank-zero/one-trip | multiplicity每个`1..U`可达，rotation coordinate/linearization typed且tail同mod mapping；改变SPM capacity/bytes不改变域 | builder创建exact multiplicity objects并返回slot bindings，不搜索actual loop |
| peer/relay/gather occurrence | direct、two-hop relay、remote ordinary/FD component、same-region DDR | common occurrence才形成family；relay/source/destination/last-use关系不完整则只有single fresh sibling | H tokens与I slot identity交给J completion/release，不immediate await |
| input order与K invalidation | reverse objects/bindings/requirements，Serialized与future Pipelined structure | plan集合逐key相同；current pre-K domain不把future overlap写死，K改变occurrence后必须重建I | Core cache key包含observed MovementState；后续K state不复用pre-K family |
| typed failure与资源隔离 | duplicate/missing binding/object、MustSeparate alias、incompatible type/range、reuse/order cycle、invalid U/rotation | 无partial plan；query不扫描loop、不物化、不按buffer bytes/SPM capacity裁剪、不选winner | preflight failure零IR，修正plan可重试；actual offset仍归Q50.0 |

实现闭合：canonical `BufferPlan`为每个physical version建立一个独立fresh object，并只为remote reduction gather增加一个typed staging
object；`StorageLifetimeDescription`仅保存execution/action definition与uses。后续coverage closure补齐了insert overwrite：H显式签发
`ResultDiscardPlan`，I只给该version建立definition-local self-use，missing publication/transfer仍返回typed failure；duplicate discard与
carried-and-discarded冲突也fail closed。没有alias/reuse、multiplicity字段、slot family、event/order、worker、offset、actual loop scan或
IR mutation。原fresh定向6/6证据由`canonical-plan-coverage-closure`的1024/1025 multi-piece/multi-producer全链及完整fresh gate更新。

### I-1 专项调研：storage binding、lifetime requirements与slot-family domain

MLIR One-Shot Bufferize先分析完整SSA alias/read-write conflict，再统一rewrite；它不会在每个use处临时决定buffer。XLA HeapSimulator则在
已有logical schedule后用明确start/end live intervals分配chunks。rotating-register/modulo-scheduling工作进一步说明slot multiplicity、
iteration overlap和release schedule必须联合。对应本仓，I位于J/K之前时只能保存storage/rotation choices与必要ordering constraints，
不能虚构时间戳或把current source order当最终lifetime。

采用方案与边界如下：

| primary source | 可迁移结论 | Wafer采用 | 明确不采用 |
| --- | --- | --- | --- |
| [MLIR One-Shot Bufferize](https://mlir.llvm.org/docs/Bufferization/)与[ownership-based deallocation](https://mlir.llvm.org/docs/OwnershipBasedBufferDeallocation/) | 先用完整SSA use-def、alias/equivalence和read/write conflict做analysis，再rewrite；ownership/release是独立SSA问题 | identity alias只接受G的exact alias；其它reuse先形成typed requirement，selected construction与release verifier分开 | 不在每个use临时决定in-place，不把unknown alias当可复用，不照搬通用runtime retain/dealloc ABI |
| [OpenXLA HeapSimulator](https://github.com/openxla/xla/blob/main/xla/service/heap_simulator/heap_simulator.h) | 明确schedule/live interval之后才分配offset，最终统一返回assignment | I只选object/slot identity；actual offsets仍由final current-IR MiniMalloc产生 | 不把plan-side lifetime或heap size当SPM legality，不复制XLA schedule owner |
| [Rau iterative modulo scheduling](https://doi.org/10.1145/192724.192731)及modulo variable expansion | 跨迭代live distance决定需要的独立名字/slot，rotation与pipeline schedule联合 | pre-K只从exact E occurrence枚举`1..U`与axis permutation；K后按actual stage distance重闭 | 不把double buffer设默认，不由slot数宣称overlap或收益，不在I里选择stage/order |
| SSA interference/coloring | must-alias、must-separate与可排序reuse分别建约束；fresh color始终可用 | compatible且无同site forced overlap的pair产生`ReuseAfterCompletion` sibling，J决定能否满足 | 不用当前源码顺序证明lifetime disjoint，不运行SPM packing或以bytes裁剪reuse |

硬件事实同样只控制可表达边界：

| 分类 | current事实 | I中的处理 |
| --- | --- | --- |
| `supported` | SPM typed allocation/view、actual range/effect/completion、3 MiB fixed arena与MiniMalloc final placement | selected object/slot全部成为真实memref allocation；offset/capacity只在Q50.0判定 |
| `board-observed` | single slot及double-slot偶/奇iteration correctness；Direct-DTE source/destination在matching completion后可复用 | double slot是普通domain member和回归case，不升级成默认multiplicity或收益 |
| `unknown` | 任意3/4/5+ slot硬件收益、queue resident数量、SPM bank/port penalty及未物化pipeline的live distance | 不进入legality、upper bound、proposal裁剪或cost；只有actual SSA与completion可关闭lifetime |
| `excluded` | plan-side footprint、buffer-count倍数、shape公式、预测lifetime和allocator retry | 不出现在requirement derivation、domain或feedback控制流 |

current `CardBufferingDomain`按`(group nodes, retained edge subset, slotCount)`建域，旧实现还用单edge bytes与SPM capacity裁剪
slot count；这条容量猜测必须删除。真正的common loop、alias、endpoint和release拖到1,900行`SelectedBufferMaterialization`在actual Instr中搜索。materializer又
move-own整个Module、clone schedule operations并返回新的module/relations。这证明`bufferCount`不是足够的plan，也违反winner direct
construction。

#### Typed plan

```text
BufferPlan
  storageObjects: StorageObjectPlan[]
  versionBindings: PhysicalVersionStorageBinding[]
  slotFamilies: SlotFamilyPlan[]
  orderRequirements: BufferOrderRequirement[]

StorageObjectPlan
  id: StorageObjectId
  tile: TileId
  scope: RegionGroupId | TraversalScopeId
  physicalType: derived from PhysicalVersionPlan
  kind: Fresh | ExactAliasOf(StorageObjectId) | ReusableObject

PhysicalVersionStorageBinding
  version: PhysicalVersionId
  object: StorageObjectId
  viewRelation: ExactPhysicalViewRelation

SlotFamilyId
  objects: StorageObjectId[]

SlotFamilyPlan
  id: SlotFamilyId
  occurrence: OccurrenceRelationId
  multiplicity: PositiveInteger
  rotation: SlotIndexExpr(iteration coordinates) mod multiplicity

BufferOrderRequirement
  define/issue/use/completion/release EventIds
  relation: MustPrecede | SameOccurrence | ReuseAfterCompletion
```

候选只保存typed IDs、binding、multiplicity和rotation choice；physical type、exact byte span、live relation和event set从G/H/E/current target facts
重算。`Fresh`/`ReusableObject`表示allocation identity，不是actual offset；Q50.0 memory planning仍分配offset。MustAlias/Equivalent只能绑定
同一object，MustSeparate不能；MayAlias/Unknown不因省内存被强行合并。exact subview可共享root但保留range relation，allocation-root相同
不等价于同一slice。

`OccurrenceRelation`从E wave scope和H payload occurrence建立，不从actual `scf.for`搜索：

```text
OccurrenceRelation
  iterationSpace: exact finite iteration domain
  recurrenceCoordinates: IteratorId[]
  producerOccurrences: affine/exact map from iteration -> EventId
  transferOccurrences: affine/exact map from iteration -> EventId
  consumerOccurrences: affine/exact map from iteration -> EventId
  tailClasses: exact finite classes
```

只有producer、所有movement endpoints、consumer/forwarder和last use能映到同一relation，且每次iteration coverage all-and-only时，才广告
`multiplicity>1`。循环外assembled consumer、不同nested scopes、non-affine/unknown occurrence或effect crossing只有single-slot/fresh
siblings；不能等actual apply再说`NoExactLoop`。

#### Lifetime与reuse constraints

query建立partial-order lifetime requirements而不是timestamps：definition/recv completion早于read，send source活到send completion，
relay活到所有child completion，D stored version活到last use，slot reuse必须晚于上一occupant的最后completion/read。两个objects：

- forced-overlap + MustSeparate：不能绑定同一object/slot；
- proven-disjoint：可直接reuse；
- 可由某个J ordering变成disjoint：reuse choice携精确`BufferOrderRequirement`，在J关闭前为Deferred；
- unknown alias/lifetime：不生成reuse choice，不靠乐观假设。

I不选择这些order，只把它们作为J hard edges；若hard edges形成cycle，该I assignment exact-reject，fresh/separate sibling保留。

#### Finite multiplicity与exact successor

single slot永远表示serialized storage（若一个operation本身同时需要两个MustSeparate versions，它们仍是两个single objects）。对一个
可rotation occurrence：

```text
tripUpper = number of dynamic occurrences in the exact finite recurrence
targetUpper = largest multiplicity representable by loop/selector/resource types
U = min(tripUpper, targetUpper)
```

`U`只由actual execution structure可产生的finite occurrence数和target selector表示范围界定，不读取SPM capacity、buffer bytes或任何
packing estimate。domain含每个`1..U`和每个interface-valid recurrence-coordinate/linearization；tail按同一modulo mapping进入安全slot并由
J release relation关闭。每个complete multiplicity choice物化出真实allocations后，再由Q50.F actual gate判定是否装得下。

exact successor在stable semantic partial plan上每次解决第一个未绑定version或family：

```text
bufferSuccessors(prefix):
  if an unbound PhysicalVersion exists:
    yield Fresh object
    yield every exact-alias required object
    yield every proven-disjoint or order-constrained compatible reuse object
    return
  if an unresolved recurrence family exists:
    for rotation coordinate/linearization in exact domain:
      for multiplicity in [1,U]: yield child
    return
  validate object/view coverage and order-requirement acyclicity
  yield BufferPlan
```

storage binding等价于带alias/interval/order constraints的graph coloring/packing，最坏指数级；rotation choices再乘`sum U`。query lazy且
只保存当前prefix。proposal可用fresh-all/serialized、DPS in-place、greedy interval coloring、double buffer、minimum recurrence-distance
slots；每个proposal经exact constructor验证，不能成为local winner或删域。SPM capacity rejection只由外层controller消费，不进入I proposal。

K选择新的stage pipeline/iteration overlap后，如果`OccurrenceRelation`、event set或reuse distance改变，affected I family及J facts必须
失效，Q51按依赖重新展开I→J；不能让K把旧multiplicity解释成另一条pipeline。反过来I multiplicity/order requirements约束K/J，但不
预先保证overlap收益。

#### I-1 Gate

- 2--6 version有界穷举oracle平铺fresh/alias/reuse objects、slot multiplicity/rotation choices并过滤MustAlias/MustSeparate/lifetime/order cycle；
  与production exact plans逐key一致、无duplicate；
- occurrence property覆盖single/multi-dimensional waves、main/tail、nested child、peer send/recv/relay、same-region DDR和loop-external
  consumer；只有all-and-only common recurrence广告multi-slot；
- upper-bound test覆盖trip与target representability分别成为minimum、rank-zero/one-trip只有1、4/5及更大slot可达；改变SPM capacity或
  physical byte span不改变I domain，actual admission另行接受/拒绝complete candidates；
- alias/range覆盖DPS in-place、fresh out-of-place、exact subview、partial overlap、MayAlias、early overwrite/free与completion-constrained
  reuse；schedule-dependent choice产生精确J requirement而非当前order verdict；
- K structure mutation使affected I/J失效，unrelated component保留；proposal provider关闭/反转不改变exact set或Q51有界穷举optimum；
- query source byte-identical、零actual loop scan/materialization/clone/default statistics。candidate apply、lifetime verifier和donor迁移由I-2。

### I-2 专项调研：construction-time slots、lifetime verifier与migration

MLIR ownership-based deallocation把buffer ownership作为SSA/lattice问题，并在bufferization之后单独插dealloc；XLA heap assignment也只在
明确live intervals上放置chunks。current `SelectedBufferMaterialization`反过来在Instr IR里寻找“像pipeline的loop”，clone/move整个loop
body、split waits、重建NCC joins、改upper bound，再remap relations。它同时承担K execution structure、J scheduling/completion、I storage
和Q50.0 memory preparation，必须由construction-time协作取代，而不是继续拆helper保留同一算法。

| 调研对象 | 本项采用 | 本项不采用 |
| --- | --- | --- |
| [LLVM MachinePipeliner / modulo-variable expansion](https://llvm.org/doxygen/classllvm_1_1MachinePipeliner.html) | overlap live range需要`m`个独立physical names；selected schedule先于renaming，renaming不改变dependence | 不迁移register move、MachineInstr clone、target rotating-register hook或MVE heuristic；Wafer slots是typed memref roots且multiplicity已由I plan选择 |
| [Rau iterative modulo scheduling](https://doi.org/10.1145/192724.192731) | recurrence distance与resource/renaming约束分层；变量扩展只实现已选择schedule | 不让I重选stage/II，也不以slot不足触发局部reschedule |
| [MLIR ownership-based buffer deallocation](https://mlir.llvm.org/docs/OwnershipBasedBufferDeallocation/) | allocation owner、alias retain与deallocation是SSA/control-flow合同；unknown region/effect fail closed | 不运行通用deallocation pass猜selected release，也不插runtime alias check替代I/J exact relation |
| [MLIR Bufferization](https://mlir.llvm.org/docs/Bufferization/) | analysis与rewrite分离，actual mutation前关闭alias/equivalence事实 | 不把tensor in-place heuristic当slot-family选择；G/I已经给出唯一physical object/binding |

#### Prepared buffers与direct construction

完整G--K plan在C mutation前关闭为：

```text
PreparedBuffers
  allocations: PreparedStorageObject[]
  views: PreparedVersionView[]
  slotFamilies: PreparedSlotFamily[]
  occurrenceBindings: (PhysicalVersionId, OccurrenceClassId) -> SlotUse
  loopBindings: SlotFamilyId -> selected E/K LoopStructureId
  releaseBindings: StorageObjectId -> J EventId
```

prepare验证每个physical version恰有storage/view binding、MustAlias/MustSeparate与range关系、slot types/alignments一致、multiplicity/rotation
匹配selected occurrence、全部I order requirements已进入J DAG、release event存在且F full-coordinate capacity已关闭。任何missing/duplicate/
stale relation在零mutation时失败。

actual没有独立“buffering pass”。C的Tile/region builder先创建storage objects，G/H/D emitters按occurrence索取slot；E/K创建selected loop
时由I提供额外iter args和yield permutation，J在prepared event points放wait/release：

```text
BufferObjectBuilder::createObject(object, rewriter):
  create one typed memref allocation for a non-rotating object
  or create exactly multiplicity independent allocations for a slot family
  bind every PhysicalVersionId view through its exact relation

SlotRotationBuilder::extendLoop(loopPlan, slotFamily, rewriter):
  create all slot roots before the selected steady loop
  normalize the selected loop IV to E/K's exact occurrence coordinate
  select coordinate mod multiplicity through typed arith/select SSA
  bind operations to that selected memref; leave release to J's exact boundary

BufferObjectBuilder::release(object, selectedEvent, rewriter):
  emit one deallocation/release after all uses and async completions
```

`multiplicity=1`不生成selector scaffolding。每个slot是独立allocation，不把一个大allocation切片伪装成独立lifetime。current K只对
innermost recurrence axis形成steady pipeline，因此rotation唯一为`{recurrenceAxis}`；outer active axes在进入inner pipeline时重新使用同一
family，不产生另一组等价rotation states。selector coordinate由`prefixCount + (iv-lower)/step`精确构造，不能从loop位置、op ordinal或
shape猜。standard `SelectLike` SSA保留all-and-only slot origins，actual lifetime/SPM analysis可重建alias union。main/tail/prologue/
steady/epilogue由E/K拥有，I只绑定slot和backedge reuse。

这套construction发生在TileRegion→Instr之前；H peer tokens和J event plan已在target-abstract层可表达，不需要等Instr后扫描。Q50.0随后
从actual memref roots/lifetimes分配SPM offsets，不能再调用selected-buffer rewrite或返回另一个owned Module。

#### Buffer lifetime verification

per-Tile/region `BufferMaterializationVerifier`复用current alias/range/lifetime analyses，并消费typed relation：

```text
PhysicalVersionMaterialization
  PhysicalVersionId -> (SSA view, StorageObjectId, optional SlotFamilyId)
EventMaterialization
  EventId -> actual operation/token/wait
```

一次检查：

1. every plan object/version/view/slot/release all-and-only materialized；每个slot family allocation数量恰为multiplicity且types/ranges相同；
2. MustAlias共享同一root+exact view，MustSeparate在forced overlap内使用不同storage/range；partial overlap按range而非root判定；
3. definition/load/recv completion支配first read，所有write/read/effect覆盖计划occurrence；
4. send/relay source活到matching completion，receive destination在wait前不可读，最后consumer/completion早于release；
5. 同一slot的相邻dynamic occurrences之间存在`ReuseAfterCompletion` path，包含loop backedge与tail；slot permutation是selected
   `SlotIndexExpr`，没有use-before-def、overwrite或premature dealloc；
6. independent regions/families不被强制合并；跨Tile/TileRegion普通SSA仍非法。

verifier不计算profitability、不改order、不插wait/dealloc，也不重复Card-wide packing；Q50.0拥有actual high-water/offset/capacity。

#### Complexity、failure与migration

prepare/emit work与`physical versions + views + sum(slot multiplicities) + event relations`线性；alias/range verifier使用现有query memo，
最坏与actual uses/conflict pairs成正比，不运行packing solver。任何prepare/emit/verify失败由C subtree guard回滚，是winner contract bug或
typed unsupported，不把slot count减半、不切换single、不卡回search。默认无timing/count日志。

| Current / donor能力 | 终态owner | 必须迁移的witness | 退役条件 |
| --- | --- | --- | --- |
| current `BufferingChoice(groupNodes,pipelinedEdges,slotCount)` | I `BufferPlan` storage/version/family bindings | 1..U、alias/reuse/rotation oracle | node/edge carrier与subset+count identity零残留 |
| 已删除的`SelectedBufferRequest/Message/Scope` | prepared physical-version/event relations | local、peer send/recv、relay、independent scopes | optional node IDs/message scan API已删除 |
| `buildSelectedBufferingScopes` | I prepare | per-Tile stable scope/ID mapping | expectedTileIds adapter和movement-domain backlookup删除 |
| 已删除的1900-line `SelectedBufferMaterialization` loop discovery/clone | E/K loop builder + I construction-time slots | 2/3/4/5 slots、tail、different scopes | `NoExactLoop` search、operation clone/move、OwningOpRef result已删除 |
| materializer stage inference/SCF pipelining | K execution structure | prologue/steady/epilogue、cross-engine stages | I不调用`scf::pipelineForLoop`或改loop bounds |
| wait splitting/operation rotation | J event schedule | Direct-DTE issue/wait/overlap | I不split waits或reorder operations |
| required NCC join rebuild | Q63/J completion closure | pending-worker/join negatives | I不include NCC completion internals |
| relation retarget/remap | typed PhysicalVersion/Event materialization | current relation、failure atomicity | pointer remap和node-based StructuredMaterializationRelations退出I |
| current actual tests | I query/prepare/builder/verifier owners | multiplicity>trip、external consumer、cross-stage write、send wait/release、different loops | 每项迁移后才删旧fixture |

#### I-2 Gate

- actual IR覆盖fresh/identity-alias/reuse object、single/2/3/4/5 slots、outer-axis repetition、1024 aligned及1025/1031 tail、typed peer-relay
  object和同Tile独立families；slot definitions/count、normalized coordinate、selector chain与plan逐ID一致；
- negative覆盖multiplicity>selected-axis occurrences、wrong rotation axis、missing/duplicate allocation、selector tamper、cross-stage write无I proof、
  async completion后release缺失及early dealloc；
- K/J generation test证明新iteration class重prepare I，old BufferPlan/SlotLifetime稳定失败；unrelated families保持stable；post-K EventGraph只从
  新BufferState重建；
- actual matrix把selector交给K phase builder并在epilogue后dealloc，再由唯一MiniMalloc路径规划all slots；capacity只作为actual结果，
  不回灌domain。source/Card subtree在prepare failure时零mutation；
- source/call-tree gate删除old request/materializer/clone、late relation remap、NCC/schedule混装；query、object builder、selector和verifier保持
  独立职责。complete-candidate总接入等待J提供同generation event order/completion，不在I中提前拼接；
- baseline single-slot继续无selector，search Pipelined才消费post-K multiplicity；本项不运行重型LLaMA或板端。

### `storage-domain`实现与闭合证据

- `BufferPlan`继续由一个schema拥有fresh/identity-alias/reuse binding、slot family和order requirement；`PeerRelayStorageId`与
  `PeerTransferSiteId`把H graph/payload slice/relay Tile及实际hop lifetime带进同一合同，不按node、shape或message ordinal恢复。
  canonical storage为每个internal relay和每个exact payload piece建立一个typed object；terminal/root继续绑定G physical version或gather
  staging，不复制graph事实。
- production `deriveStorageRequirements`直接消费current representation、movement、canonical lifetime、selected TemporalPlan和同一
  TemporalDomain descriptor。identity alias保持must-alias；compatible且没有共同actual semantic site的version pair生成
  `ReuseAfterCompletion` sibling，fresh始终保留。unknown/forced-overlap pair不产生reuse。
- occurrence按descriptor extent与selected tile size逐axis checked ceil-div得到；只有一个明确common scope且总occurrence大于1时形成family。
  `U=min(exact occurrence product, uint32 selector range)`，active-axis全部permutation可达；rotation work limit耗尽为typed
  `Indeterminate`。改变resource bytes或SPM capacity不改变family集合。
- `StorageDomain`验证reuse target/object totality、禁止self/duplicate reuse、强制identity alias、检查order cycle，并证明family `U`不超过
  exact occurrence product。binding改变时family object ID随selected object重闭；relay object没有version binding也不会被误删。
- `prepareStoragePlan`与`StorageObjectBuilder`按selected multiplicity创建exact独立allocations，提供typed object/version modulo lookup；
  `verifyEmittedStorageObjects`检查all-and-only count/type/binding，`verifySelectedStorageLifetimes`在actual current block上检查definition、
  use、matching async completion与dealloc顺序。两者不插wait、不移动operation、不计算offset。
- PlanningSession现在为derived physical versions和peer relay storage生产真实requirements并进入同一`MovementState -> InitialBufferState`
  transition；missing/unsupported/indeterminate保持typed分类。legacy `BufferingApply`/expected-Tile scope adapter已经删除；old query-only
  `Buffering`只等K/J独有oracle迁移后统一退役，不能再取得actual owner。

fresh evidence的定向矩阵覆盖rank-3 `1024/1025/1031`、input reversal、bytes/capacity metamorphic independence、
all `1..U`与multi-axis rotation、2/3/4/5 actual allocations、identity alias/reuse cycle、peer relay multi-piece object、Direct-DTE
send→await→release lifetime以及missing/duplicate/early-release/invalid-U/rotation-work负例。ordinary host unit `921/921`通过（独立
attention production closure仍不归本项），configured lit `225/225`、4个public link smoke、source/IR organization和diff检查通过。

### I-closure：固定K后的structure-specific storage重闭

storage-domain只给event-resource-foundation和execution-structure-domain eligibility提供基于Serialized occurrences的storage/slot
possibilities；它不是最终`BufferPlan`。structure选择会改变prologue/steady/epilogue occurrence、跨迭代live distance和reuse boundary，
因此必须销毁initial storage/event facts并进入structure-specific-storage：

```text
recloseBuffers(prefixThroughMovement, initialBufferFacts, structure):
  occurrences = deriveOccurrences(E, H, structure)
  requirements = deriveStorageAndCompletionRequirements(occurrences)
  if structure is Serialized:
    yield canonical single-slot bindings
  else:
    enumerate the same exact I storage/slot domain under requirements
    retain every multiplicity/rotation satisfying live distance and reuse
  return BufferState {structureGeneration, BufferPlan}
```

Serialized不保留没有consumer的extra slots；Pipelined也不继承initial multiplicity，必须按selected stage/distance重新证明`1..U`、rotation、
MustSeparate和ReuseAfterCompletion。`BufferState`携带typed structure generation，J closure只接受同generation；K sibling或stage/distance变化
必须产生新的I continuation。这个closure复用I-1 successor、I-2 prepare/verifier和F core resource schema，不建立第二buffer算法。

完成门禁：K的每个Serialized/Pipelined有界穷举plan与独立occurrence/slot reference集合一致；故意复用pre-K plan稳定失败；I重闭后J EventGraph
中的storage/reuse edges逐ID变化，unrelated scopes保持相同；query零IR，candidate construction只消费post-K plan。只有该checkpoint通过，
Q50.J schedule closure才能开始。

本work item在实现前冻结下面的覆盖矩阵。slot upper/lower bounds只来自fixed K的exact occurrence和ready→release stage distance；
1024/1025/1031只验证整除、tail和多维recurrence共享同一算法，任何shape/bytes/SPM capacity都不进入admission。

| 输入等价类 | 代表输入 | fixed-K storage路径 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- | --- |
| Serialized重闭 | rank-3 1024/1025、single/multi-component、rank-zero | 清除pre-K extra families，保留fixed storage binding并规范成single occurrence | stale/missing structure scope | post-K BufferPlan无rotation/multiplicity旁路，generation来自parent state | schedule-domain只能接收新的BufferState |
| Pipelined live distance | rank>=3 1024及1025/1031，2/3+ stages | 按每个physical object的BufferReady/BufferRelease stage推导minimum multiplicity，枚举`min..U` | missing ready/release、release早于ready | all-and-only object families、live distance、lower/upper bound和plan multiplicity | J按family occurrence/multiplicity建立slot backedge |
| 多维outer loop与tail | axis occurrences如`[2,8,1]`、`[2,9,1]`，recurrence axis为1 | 唯一rotation为`{1}`；outer axis重复inner family，tail使用同一global coordinate modulo relation | recurrence iterator越界、occurrence overflow、非selected axis rotation | 1024 upper=8、1025/1031 upper=9；无axis permutation duplicate，tail仍映到合法slot | selected construction直接从normalized steady IV生成selector |
| alias/reuse/gather staging | identity alias、两个versions复用一object、remote gather staging | family identity基于selected physical StorageObjectId，semantic lifetimes可多对一但allocation family唯一 | unknown selected object、conflicting shared family | alias/reuse object只产生一个family；gather staging不因非version origin漏失 | storage object builder按object multiplicity创建exact slots |
| 独立scopes与有界oracle | 2--6 objects、trip 2--7、两个independent components | family option做Cartesian product，unrelated Serialized scope不变 | successor work-limit为Indeterminate，malformed family为compiler bug | production plans与独立`multiplicity × rotation` reference逐key相等 | schedule-domain可逐state重建，不依赖proposal |
| Core re-entry | public rank-3 1024/1025 search prefix | ExecutionStructureState→post-K BufferState→missing schedule | pre-K BufferPlan故意提交到post-K domain稳定失败 | work count、state identity、required coordinate和source byte identity精确 | `schedule-domain`消费BufferState及同generation derived lifetime facts |

实现闭合：`StructureSpecificStorageDomain`保存完整K generation。Serialized清除pre-K families；Pipelined逐semantic lifetime计算
ready→release stage distance，再对selected physical `StorageObjectId`取maximum。minimum multiplicity只由该distance与current唯一
launch distance 1导出；upper bound只取`recurrence.axisOccurrences[recurrenceAxis]`和`uint32`表示范围。1024/128与1025/128因此分别为
8和9，不再乘outer axes。current K只pipeline innermost axis，所以rotation唯一为`{recurrenceAxis}`，已删除无actual loop对应的axis
permutation枚举及其work-limit状态。

`SlotLifetimeRequirement`现在携同一`PipelineIterationClass`，J可复验prefix/steady/tail generation。`StorageObjectBuilder`按selected
multiplicity创建exact独立SPM allocations；`buildSteadyOccurrenceCoordinate`从static `(iv-lower)/step + prefixCount`生成global coordinate，
dynamic selector用standard arith modulo/compare/select形成all-and-only slot origin union。selector verifier逐operation检查modulus、slot顺序、
chain result和无部分mutation；wrong axis在创建任何IR前失败。1025/1031 tail使用最后一个global coordinate进入同一family。

actual integration把每个slot root作为I-owned external storage proof交给K prepare；K只在proof对应同generation BufferPlan family、recurrence和
rotation axis时允许cross-stage write。selected loop完成后所有slot release保留在software-pipeline epilogue之后，actual MiniMalloc从
current SelectLike origins/lifetime/dealloc重算并分配offset。I没有调用SPM估算、没有根据capacity改multiplicity，也没有插wait/join。
PlanningSession继续以`ExecutionStructureState -> BufferState`形成generation，并用同一event builder重建post-K EventGraph；ScheduleDomain
只接收该BufferState及typed slot lifetimes。

旧`SelectedBufferMaterialization`、node/message request schema、loop discovery/clone、wait splitting、NCC rebuild、Tile memory-planning hook、
source/test/CMake已经全部删除。remaining legacy `Buffering`只属于待随J/unified donor一起退役的旧query facade，不再有actual owner。

本轮fresh验证：structure-specific storage、actual selector/K/MiniMalloc及直接J/Core下游42/42；排除future attention production fixture后
普通host unit 922/922；default configured lit 225/225；4个public-header/link smoke、完整configured build、IR/source organization及
diff检查全部通过。shape、bytes和3 MiB test arena只用于实际覆盖与MiniMalloc调用，没有进入domain或selector选择。

## Q50.J：Ready/Order/Worker/Resource Schedule

### Pipeline contract

Pipeline position:
- Upstream IR / input:
  immutable TensorProgram、fixed semantic facts与closed B--I typed plans，以及从这些facts/plans、Q63 completion和immutable target topology/resource facts可重算的
  compute/movement/buffer events；没有actual Tile/Instr module、worker assignment或execution-structure pipeline。
- Current stage responsibility:
  foundation先建立plan-level event nodes、hard dependencies、async completion、exact buffer/effect constraints、resource uses及
  disjunctive resource-order choices；不选择order/worker。Q50.K改变execution structure后重建foundation；J closure再枚举order/worker/
  completion并生成query-local calendar。每个complete candidate materialization按selected schedule创建actual order/worker/waits/joins。
- Output IR / files:
  foundation/query不写IR或文件；`EventGraphAnalysis`和calendar可销毁、可重算，不进candidate state。closure state只保存typed
  schedule choices。candidate actual IR用op order、worker attrs、async waits和completion joins自包含表示；对应由plan IDs映射，不保存pointer。
- Downstream consumer:
  Q50.K在planning event structure变化后重建本域；Q51把每个complete schedule/pipeline assignment交给Q50.F actual gate，使用actual
  SPM/DDR/transport/resource/ABI结果和cost比较accepted candidates；Q52使用calendar work/contention estimate排序partial work。
- User-level driver / named pipeline:
  Q51 closure后的public `search` session；baseline保持其canonical worker0/source order，不枚举本域。
- Explicit non-goals:
  foundation不选order/worker/pipeline/winner，不clone/materialize module、不枚举时间戳/idle、不用nominal bandwidth签发legality，不保存
  operation handle或shadow calendar，不恢复static capability/profitability registry。
- Done criteria:
  有界穷举independent reference逐点匹配order×worker域；planning→candidate apply correspondence、Direct-DTE issue/wait、multi-worker join、
  alias/effect、DDR estimate、opaque endpoint与known directed-link资源正负例闭合；production search不构造Instr后取first。旧ready-order/worker算法与测试
  完成能力迁移后才能删除其owner。

### Q50.J work-item分界

Q50.J由三个work items闭合：

- `canonical-schedule`只关闭canonical single-slot/Serialized point。它直接复用I的typed `execution/action` definition/use sites作为
  `ScheduleNodeId`，为每个object加入definition→use hard edge，按stable semantic ID做Kahn topological order，并把全部高层nodes绑定
  `NCCWorker::Worker0`。同一execution内部的define/use不造self edge。该点是保守全序，不拆async issue/completion，不建立resource lane、
  timestamp或wait placement。
- `event-resource-foundation`在完整storage-domain后把semantic action展开为issue/completion/wait/release等`EventId`，建立hard DAG、
  resource facts、completion obligations、exact successors及Core consumer；不选择最终schedule。
- `schedule-domain`在K→I重闭后枚举worker/resource/control order和completion boundary，形成最终`ClosedSchedulePlan`及selected apply
  所需的完整typed合同。actual apply必须等full-feasibility和unified-search-closure拥有的winner transaction，不能在本项提前生成IR。
  本域不能把canonical全序当作唯一domain或用worker0兜底其它失败。

canonical checkpoint的`CanonicalSchedulePrefix`只保存`order`和`workerBindings`；definition/use dependencies作为可重算的
`CanonicalScheduleCoordinate.dependencies`返回。`ScheduleNodeId`不是未来`EventId`：一个movement action在event foundation可展开成多个
endpoint/issue/completion nodes，当前ID不能被后续误当成已关闭completion。

```text
Pipeline position:
- Upstream IR / input:
  canonical BufferPlan/resource/lifetime descriptions及SerializedExecutionPlan；每个storage object已有typed definition/use，尚无event、
  completion、resource instance、pipeline structure或actual IR。
- Current stage responsibility:
  验证object/lifetime和execution/action node coverage，建立definition→use dependency DAG，生成stable topological source-semantic order并为
  每个node绑定worker0。
- Output IR / files:
  query-local CanonicalScheduleCoordinate，其中CanonicalSchedulePrefix只含node order/worker bindings，dependencies为derived proof；不修改IR或
  写文件。
- Downstream consumer:
  attention-work-projection和canonical feasibility消费closed canonical prefix；deterministic baseline后续把同一order投影到selected
  construction。event-resource-foundation从B--I/K重新建立更细event graph，不把本项当完整search domain。
- User-level driver / named pipeline:
  无独立pass、pipeline或CLI；none与search canonical planning prefix静态调用同一constructor。
- Explicit non-goals:
  不拆async issue/completion，不放wait/join/release，不枚举worker/resource/order、timestamp/idle或pipeline stage，不读取actual op/block/
  pointer，不调用InstructionSchedule或修改worker attr。
- Done criteria:
  1024/1025 all-16、multi-root/diamond、ordinary/FD local+remote merge、support/rank-zero及输入顺序扰动均形成all-and-only stable nodes、
  dependencies、order和worker0 bindings；duplicate/missing object/lifetime/execution、unknown site及cycle得到typed failure；source IR不变，
  attention projection可直接消费，fresh build/unit与organization通过。
```

| 覆盖类 | 代表输入 | 必须断言的canonical schedule | 直接下游witness |
| --- | --- | --- | --- |
| load/execute/publish | rank-3、1024/1025、all-16 Tile | 每个Tile恰有load→root execution→publication，所有node出现一次且worker0；ragged顺序不依赖局部extent大小 | attention/F按相同execution/action IDs读取closed prefix |
| chain/fanout/diamond | 1025级multi-root/multi-sink | producer execution早于全部DDR uses，每个DDR早于对应consumer；shared source产生多dependency但node不复制 | 后续event foundation可把每个DDR action拆endpoint/completion而不恢复producer |
| ordinary/FD reduction | rank>=3、1024/1025/1031，local+remote components | root→remote gather→merge、local root→merge、merge→publication全部成立；三个component不增加execution node，只增加各自object evidence | attention projection读取同一merge execution，future completion在gather action内细化 |
| support/rank-zero/merge-only | pad/support、0-rank root、only-merge Tile | 同execution define/use不造self cycle；rank-zero/merge-only node仍各一次且worker0 | canonical feasibility不因无extent或无movement漏execution |
| determinism | 反转serialized、objects、lifetimes和uses输入 | nodes/dependencies/order/worker逐字段相同，tie-break只用semantic ID | production none与test query共享一个canonical point |
| typed failure | empty/duplicate storage object或lifetime、missing resource/lifetime、unknown execution site、dependency cycle | `BrokenSchedulePlan`准确分类且不返回partial order | 修正输入可重新query，source IR byte-identical |

实现闭合：canonical `CanonicalSchedulePrefix`复用storage的execution/action sites，保存stable Kahn total order和node-level worker0 default；
`ScheduleDependency`逐object保留definition→use evidence。同execution self-use不造edge，没有EventId、async completion、resource lane、
timestamp、wait/join、pipeline或actual IR。fresh定向6/6、完整`WaferUnitTests` 791/791、default configured lit 224/224、compiler
public link、完整configured build及IR/source organization通过。

### event-resource-foundation-1 专项调研：event DAG、completion与resource-effect boundary

LLVM MachineScheduler先构造ScheduleDAG，再由独立strategy选ready node并可选择是否追踪register pressure；CIRCT scheduling也把problem
components/input constraints和solution properties/verification分开。MLIR async则要求所有依赖通过token/value显式表达，MemoryEffect
resource只描述资源类别，精确range仍由alias analysis负责。对应本仓，foundation必须是“待调度问题”的pure typed analysis，而不是
从已排好序的Instr block反推可移动windows。

| 调研对象 | 采用的工程规则 | 未采用的部分及原因 |
| --- | --- | --- |
| [LLVM MachineScheduler](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/MachineScheduler.cpp) | DAG/problem construction与ready-node selection分离；event identity及hard edge先于schedule policy稳定 | 不迁移MachineInstr pointer、register-pressure启发式或target hook；本stage尚无actual Instr，也不拥有profitability |
| [CIRCT scheduling infrastructure](https://circt.llvm.org/docs/Scheduling/) | input constraints、selected solution与独立verification分层；共享资源只产生待解约束 | 不引入另一套schedule dialect或operator registry；current typed plan已经拥有semantic ID与resource key |
| [MLIR Async dialect](https://mlir.llvm.org/docs/Dialects/AsyncDialect/) | issue/completion由显式token/value关系表达，等待只能消费对应completion | 不把region/block boundary解释成隐式await，也不创建无法被当前lowering消费的async side table |
| [MLIR side-effect/speculation model](https://mlir.llvm.org/docs/Rationale/SideEffectsAndSpeculation/) | 只从typed effect/interface签发可重排事实；unknown/effectful execution fail closed | `MemoryEffectOpInterface`只描述effect，不能单独证明exact alias range，因此不按同root名字或source顺序补RAW/WAR/WAW |

本项读取的current硬件、runtime和ABI事实分级如下；这里只把`supported`事实放入hard event/resource合同：

| 事实 | 等级 | foundation中的处理 |
| --- | --- | --- |
| ordinary NCC issue按typed worker有序，participant join只完成mask中的pending worker | `supported`，由Q63 operation interface、analysis与target protocol共同拥有 | execution/DDR issue使用closed worker domain；completion只保留typed participant obligation，不提前插join |
| Direct DTE send/recv产生独立token，wait/finish只完成该token | `supported`，见13号文档及current CRT/target lowering | 每个selected payload piece和physical hop建立独立`DirectDTE` issue/completion；不与NCC completion互换 |
| current Direct DTE内部NoC route | `unknown`/opaque | 只记录endpoint DTE exact use与NoC transfer estimate；没有显式target route就不创建link contention |
| deterministic target route输入给出的逐link路径 | `supported`仅限显式typed输入 | `ExactMovementRoute`绑定一个selected graph hop；逐link usage只附着到该hop的每个exact payload event。当前schema没有capacity/VC字段，因此mode为`CapacityUnits`，不生成exclusive order |
| raw broadcast/scatter或其它collective硬件执行形态 | `board-observed`但current compiler primitive未闭合 | `excluded`于本域；H没有selected typed action时J不恢复collective或同步协议 |

已删除的`CardInstructionScheduleDomain`曾借actual Module/Block/Operation pointers并用snapshot attrs/operands/types模拟epoch；它按source
block order给同root memref建立RAW/WAR/WAW，把任何synchronous op两侧全序化，并把任意peer endpoint伪装成exact
`DirectedPeerLink`。已迁移的是topological enumeration、closed worker enum、Q63 completion和resource classification witness；actual-
IR-first domain、snapshot和隐式source-order规则没有保留。

#### Planned event model

```text
EventGraphAnalysis
  events: PlannedEvent[]
  hardDependencies: EventDependency[]
  orderChoices: DisjunctiveResourceOrder[]
  completionObligations: CompletionObligation[]
  resourceUses: PlannedResourceUse[]
  components: EventComponent[]

PlannedEvent
  id: EventId
  scope: CardId / TileId / RegionGroupId / TraversalScopeId / OccurrenceClassId
  kind: ComputeIssue | MovementIssue | Completion | Wait | LocalCombine
      | BufferReady | BufferRelease | ObservableWrite
  operation: typed semantic/B--I plan action identity
  workerDomain: optional closed NCCWorker set

EventDependency
  predecessor, successor: EventId
  reason: SSAValue | NestedExecution | LoopRecurrence | EffectOrder
        | TransferReady | BufferLifetime | Completion | Publication

DisjunctiveResourceOrder
  events: EventId[]
  resource: ResourceKey
  allowedOrders: typed finite precedence alternatives

CompletionObligation
  issue: EventId
  completion: EventId
  participants/tokens: typed Q63 or H completion identity
```

`EventId`从semantic action ID与exact occurrence class形成，不含operation ordinal、pointer、symbol、Location或未来start time。一个async
action至少有issue和completion两个nodes；只有typed target/Q63/H contract明确声明synchronous的action才可用zero-distance
issue→completion表示，missing contract或DDR/no-hop movement不得默认成Synchronous。Direct-DTE completion只来自H token/wait contract，
NCC issue/join只来自Q63 typed completion，二者绝不互相完成。

Q50.S semantic owner和每个B--I mechanism通过静态typed visitor贡献event descriptors；没有runtime provider registry、字符串operator type或一份event attr IR。
query-local graph在candidate state变化后销毁；每个complete candidate materialization后，其actual IR是该transaction内唯一事实源，
因此不构成长生命周期shadow schedule或跨candidate cache。
H的每个`TileCommunicationPlan` action和dependency直接产生对应issue/completion/local-combine event与hard edge；J不再从
`Ring/Tree`标签恢复round，也不重新选择chunk graph。对current opaque Direct DTE只建source/destination endpoint与message依赖，只有
target提供exact route时才加入per-link resource uses。

#### Hard dependencies与order choices

foundation依次加入：

- G physical-version definition→D/H consumer use，D nested parent/child和E scope/loop recurrence；
- H source-ready→send、receive-preparation/issue/completion→read/forward、relay/gather origin state及DDR store→load；
- I define/use/completion/release、slot backedge和`BufferOrderRequirement`；
- current TensorProgram的effect/control/observable ordering及typed call/region boundary；
- Q63 issue→participant join或synchronous writeback。

必然方向才成为hard edge。两个MustSeparate ranges共享同一exclusive resource但semantic order可任选时，建立
`DisjunctiveResourceOrder`，由J closure枚举`A→B`或`B→A`；nominal source order不代签。MayAlias/unknown range不能用root equality强行排序，
而是Deferred/Unsupported；exact subview non-overlap可并行。

foundation构图后做stable Kahn cycle check；hard cycle是完整fixed-semantic+B--I combination的ExactRejection并返回最小semantic cycle witness。它不
选择一个resource order“修复”cycle，也不删除其它I/H/D siblings。

#### Resource facts

```text
ResourceKey =
  TileEngine(TileId)                         // estimate only; no FU identity
  | NCCWorker(TileId, NCCWorker)
  | DirectDTESender(TileId)                 // one exact sender slot
  | SPMRange(StorageObjectId, ExactPhysicalRange)
  | CardDDRResource(CardId)                 // estimate only; no proven channel
  | DirectedNoCLink(DirectedLinkId)          // exact usage, unknown capacity
  | OpaqueNoCTransfer(SourceTile, DestTile)  // estimate only
  | DTEReceiverFSM(TileId)                   // four exact interchangeable lanes

PlannedResourceUse
  event: EventId
  resource: ResourceKey
  mode: Read | Write | Exclusive | CapacityUnits
  interval: IssueToCompletion | Instantaneous | Until(EventId)
  knowledge: Exact | LowerBound | Estimate
```

target operation/completion interfaces给worker/completion facts，G/I exact storage ranges给SPM。没有typed FU identity的粗粒度
`TileEngine(TileId)`和H DDR plan都只给estimate；current hardware资料没有证明它们是capacity-1 hard resource，因此不得进入order。只有H target routing是
deterministic/programmable且route closed时才创建`DirectedNoCLink`；current schema只证明link usage，不证明capacity/VC，所以仍不进入hard
contention。opaque target-routed peer只产生endpoint DTE exact resource、NoC hop/cut estimate。共享resource本身不自动产生hard edge：capacity/exclusive constraints进入order
choices或J closure，bandwidth只形成estimate。

#### Analysis scope、failure与invalidation

event graph的最窄完整container是Card plan，因为peer matching和DDR可跨Tiles；构造先按Tile/region分component，再加cross-Tile edges，
无连接components可独立query。source-only relation/effect builder可以有MLIR AnalysisManager薄wrapper；driver planning直接拥有同一
builder的typed result。assignment-dependent event graph保持query-local，不伪装成MLIR analysis或跨session cache。
extensionally相同component可在immutable session memo descriptor，但assignment仍独立。

缺plan field或尚未接入的completion contract返回F-style scoped Deferred；source/target明确没有受支持event/resource contract为Unsupported；构图work limit为Indeterminate；malformed
ID/duplicate action是compiler bug。fixed semantic fact、B--I或K任一observed choice改变时，相关component graph、resource facts和后续J assignment全部失效；
不比较operation snapshots或manual epoch。

#### event-resource-foundation-1 Gate

本work item在实现前冻结下面的覆盖矩阵。1024/1025/1031只用于证明同一通用event构造覆盖整除、ragged和tail，
不进入EventId、dependency或resource合同；2--7 event小图只用于有界独立oracle。

| 输入等价类 | 代表输入 | 结构路径 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- | --- |
| execution与普通storage lifetime | rank-3，主维1024/1025，至少两个Tile | definition/ready→compute issue/completion→use/release，独立branch、chain、fanin/fanout | duplicate/missing execution、object、resource、lifetime | all-and-only EventId、每条data/lifetime edge reason、SPM range；粗粒度Tile engine只有estimate且无hard order | execution-structure-domain直接读取同一graph的components和hard DAG |
| DDR边界与publication | rank-3，1024/1031，multi-piece transfer | external load、DDR stage、observable write；CardDDR只贡献estimate | action/resource不一致、缺completion boundary | issue/completion、ready/publication依赖、CardDDR `Estimate/CapacityUnits`和observable event逐项相等；无CardDDR hard order | K不能通过source order恢复movement边界 |
| peer direct/relay/gather | rank>=3，1025/1031，local与remote contribution | Direct DTE endpoint、每hop relay、gather local combine | malformed hop、endpoint不连续、unknown action | 每hopissue/completion链、source/destination DTE exact use；opaque route没有DirectedNoCLink | 后续schedule域在相同events上选择order/worker/wait |
| alias、reuse与slot family | rank-3，1024/1025，fresh/identity alias/reuse、multiplicity>1 | alias共享object；ReuseAfterCompletion变成completion/release→later-ready hard edge | alias source缺失、reuse端点缺失、order cycle | object ready/release覆盖、slot backedge及order reason，不按shape猜lifetime | K改变occurrence后丢弃本graph并先重进I |
| Q63 typed completion与resource facts | rank-3，1024/1025，worker-capable compute与participant join descriptor | issue→participant completion；NCC completion、coarse engine estimate与exact DTE资源保持分层 | missing contract为Deferred；empty worker domain、invalid participant分别typed失败 | obligation participants、worker domain和resource knowledge逐字段一致；无implicit Synchronous或coarse-engine exclusive | J closure只枚举foundation给出的closed typed domains |
| determinism与有界oracle | 2--7 event tiny oracle，加rank-3 1025输入顺序扰动 | stable Kahn/component划分、exact successor | hard cycle、work limit、unsupported resource contract、compiler bug分别分类 | graph全字段、最小cycle witness、输入反转结果一致，source IR byte-identical | Core cache只复用同一state的可重算结果 |

- plan-level unit覆盖independent branches、chain/fanin/fanout、nested execution、multi-piece DDR、peer direct/relay/gather、slot rotation、
  release和observable output；每个selected action产生all-and-only events/completion；
- hard-edge oracle逐reason比较SSA/effect/range/token/Q63关系；可交换WAW/共享resource形成两个order choices而非source-order edge，exact
  non-overlap无冲突，MayAlias不猜；
- opaque target routing没有DirectedNoCLink exact use；deterministic/asymmetric route fixture才有正确directed links；DDR/DTE/NCC/SPM
  resource scope与capacity modes正确；
- hard cycle返回最小causal witness且只拒绝相同fixed-semantic+B--I combination；missing/unsupported/work-limit/compiler-bug分类独立；
- semantic/B--I/K mutation精确invalidate affected components，hash/input/parallel discovery不改变stable graph；source byte-identical、零Instr/module
  construction、pointer snapshot、default statistics。order/worker算法由schedule-domain。

实现闭合：同一个`buildEventGraph`同时服务pre-K `InitialBufferState`和post-K `BufferState`。后者以完整K identity与重闭后的
`BufferPlan`独立缓存，structure或storage sibling变化必然重建，`ScheduleDomain`只消费post-K graph。candidate state仍不保存graph、
operation pointer或calendar。

- `MovementEventAction`用typed `Logical/DDRLoad/DDRStore/PeerSend/PeerReceive` phase、exact payload-piece编号和selected physical hop形成stable
  ID；multi-piece DDR显式形成store→load，external fanout形成一次root DDR load→所有root sends，relay形成parent receive completion→child
  send issue。send/receive issue保持独立，两侧completion同时依赖matching两侧issue。
- `PeerTransferSiteId`把I的relay definition/use lifetime直接接到同一per-piece/hop event；resident source、relay和terminal storage没有按
  action名或shape补owner。`CanonicalStoragePlan`不再同时登记一份粗粒度peer source use。
- ordinary DDR phase使用all-worker closed domain与`NCCParticipant` obligation，participant在J选择worker后确定；peer hop只使用
  `DirectDTE`。logical movement completion不伪装成硬件completion，缺少execution contract返回`Deferred`，unknown/effectful source
  execution返回`Unsupported`，`CompletionProtocol::Unknown`不能进入ScheduleDomain。
- current production contract producer从selected root的typed op/effect读取事实：memory-effect-free Linalg execution按current
  structured-to-Tile-to-Instr合同产生worker-selectable ordered NCC issue；纯ViewLike execution明确没有async issue；其它operation不猜测。
  final actual Instr仍由Q63 analysis重算completion并在full-feasibility做plan/actual parity。
- exact route现在绑定`(peer graph actions, physical transfer hop)`，不会把一条logical action的route错误套到整棵relay/fanout graph；
  route缺失时只有opaque estimate，不进入hard resource order。
- unit直接覆盖rank-3 1024/1025/1031 execution、DDR、direct/relay、external fanout、exact route、storage lifetime、missing/unsupported
  contract、unknown protocol与pre/post-K rebuild。worker/resource sequence、control linearization和latest-unavoidable completion placement仍由
  `schedule-domain`拥有；本项没有放置wait/join。

本轮fresh验证：相关9个suite 42/42、排除尚未归本项的future attention production fixture后普通host unit 923/923、default configured
lit 225/225、4个public-header/link smoke、完整configured build、IR/source organization及diff检查全部通过。

### schedule-domain 专项调研：ready/resource successor、list proposals与bounds

CIRCT把scheduling problem的input/solution constraints分别`check`/`verify`，并为acyclic、cyclic、shared-resource和modulo problem使用
不同模型；LLVM MachineScheduler也让ScheduleDAG与pick-node strategy分离。本仓不引入SSP dialect或字符串operator library，但复用该
分层：foundation提供一个只读`ScheduleDomainKernel`，K前后都从当前EventGraph重建；candidate只保存typed choices。

#### Canonical schedule choices

```text
SchedulePlan
  workerBindings: EventId -> NCCWorker
  resourceBindings: EventId -> ResourceInstanceId
  resourceSequences: ResourceInstanceId -> EventId[]
  controlOrders: ControlScopeId -> EventId[]
  completionPlacements: CompletionObligationId -> EventBoundaryId
```

foundation阶段可生成partial `SchedulePlan`供K判断，但不把它当最终winner。worker domain来自event的closed typed set；interchangeable
resource instances使用canonical lane symmetry：第一次使用必须选最小未用lane，后续才能开启下一lane，避免仅换lane编号的duplicate。

同一个capacity-1/exclusive resource上的pairwise order不逐bool保存，而是枚举该resource users在hard dependencies约束下的完整
`resourceSequence`，只把相邻events作为precedence加入graph；同一transitive order因此只有一个identity。capacity>1先选择canonical
resource instance，再分别枚举per-instance sequences。SPM exact non-overlap和read-only sharing不进入sequence；opaque NoC transfer无exact
link resource。

每个Tile/region control scope的actual block需要一个linear order，使用ready-set successor：

```text
scheduleSuccessors(prefix):
  if unresolved worker/resource binding exists:
    yield each canonical typed binding
    return
  if unresolved resource sequence exists:
    yield each Kahn-ready resource user
    return
  if unresolved control order exists:
    ready = unscheduled events whose hard/resource predecessors are scheduled
    yield prefix + each event in stable EventId order
    return
  if unresolved completion placement exists:
    yield every existing boundary after issue and before all dependent uses/releases
    return
  validate all constraints and yield SchedulePlan
```

跨Tiles只有dependency，不建立无意义的全Card total order；independent components分别枚举再由Q51组合。completion位置只取已有event
boundaries，故finite；不枚举任意timestamp/idle。若真正需要延迟，必须由resource sequence、dependency或completion boundary表达。

Kahn prefix对每个linear extension有唯一path；resource sequence/canonical lane同理。因此exact domain完整、无编号对称duplicate。最坏
control orders为阶乘、worker choices为`W^N`、resource bindings/sequences指数级，kernel只能lazy。

#### Query-local calendar与sound bounds

给定一个complete或partial plan，calendar builder只在query中求earliest feasible boundaries；它不进入state：

```text
criticalPathLB = longest hard-dependency path using known latency lower bounds
resourceWorkLB(resource) = ceil(sum known work / exact resource capacity)
recurrenceLB = max cycle-distance bound                         // K后才存在
transportLB = H dilation/cut/tree-work lower bounds
makespanLB = max(all known lower bounds)
```

unknown latency/work项以0参与数学下界但显式标`Unknown`，不能与known complete cost混成winner；nominal bandwidth产生estimate，不进入
legality或admissible bound。resource capacity/latency来自typed target operation facts；无exact route时没有link-work resource bound。
calendar必须发布每个data-ready/completion/release，不能最后用`max(compute,movement)`猜overlap。

#### List/DP proposals

deterministic list proposal从ready set按`critical-path remaining desc, scarce exact resources desc, EventId`选event，并为worker/resource使用
least-loaded canonical instance；每一步都经exact kernel验证。chain/tree component可在有限worker/resource proposals上DP，一般component
用best-first prefixes。它们只给ordinary domain member和lower bound，不做local winner、beam legality或fixed top-k；work-limit结束
proposal，exact successor仍可调用。

`TargetSchedulingCapabilityRegistry`稀疏row把legality和profitability混合、缺row为Unknown，不进入新设计。hard facts来自typed target
operation/resource/Q63 contracts；profile profitability归Q52。registry、row table、adapter和只验证row closure的tests在donor能力迁移后
整岛删除，不建第二份matrix。

#### Foundation result/failure

- exact empty worker/resource domain、hard/resource-order cycle、无completion boundary是当前fixed-semantic+B--I/K combination的ExactRejection；
- missing K structure或未赋值I/J coordinate为Deferred；unsupported target event/resource contract为Unsupported；work-limit为
  Indeterminate；malformed graph/plan为compiler bug；
- no-good key只含实际参与witness的event/resource/upstream choices，不删其它worker/order/K siblings。

#### schedule-domain Gate

- independent reference对2--7 events平铺worker/resource instances、resource sequences、control topological orders和completion
  boundaries；与kernel plan集合一致、无lane/transitive duplicate；
- donor counts在对应plan graph保持：两个independent worker-capable events为`2! * 3^2 = 18`，producer+两个fanout consumers为
  `2! * 3^3 = 54`；same-range ordered writes只保留semantic合法方向；
- Direct-DTE issue→independent compute→wait、fanin/fanout、relay、slot reuse、multi-worker join、DDR estimate及known/opaque NoC resources
  的ready sets/requirements正确；
- critical-path/resource/transport bounds与tiny exhaustive schedule optimum比较从不高估；unknown fact不变known，list/DP proposal始终为
  exact member；关闭/反转proposal不改变domain；
- K graph变化重建kernel/calendar；input/hash/parallel discovery不改stable plans。query无IR/pointer/timestamp state、default stats；
  candidate apply与完整schedule closure留给K之后的J-closure。

## Q50.K：Conditional Stage Pipeline

### Pipeline contract

Pipeline position:
- Upstream IR / input:
  immutable TensorProgram、closed D--I plans、I `OccurrenceRelation`/slot constraints及J foundation EventGraph/resource facts；没有actual
  Tile/Instr IR、final J schedule、SPM/DDR offsets或DTE binding。
- Current stage responsibility:
  每个recurrence scope保留`Serialized` identity；只有structural dependence、occurrence、completion、slot和trip facts允许时，惰性生成
  ordered stage partitions与inter-iteration distance组成的`Pipelined` execution structures。choice改变EventGraph/occurrences后使I/J facts
  失效并重算；planning不改IR。
- Output IR / files:
  query不写IR或文件，输出typed `ExecutionStructurePlan`；stage expansion、prologue/steady/epilogue和cyclic event graph可重算。winner
  construction才由E/K loop builder创建SCF phases并由I/J插slot/completion；不产生pipeline attr、side plan、clone或默认统计。
- Downstream consumer:
  Q50.J从planned event structure重建schedule alternatives；Q51为每个complete assignment物化selected pipeline并由Tile memory
  planning分配全部rotating slots，Q50.0验证transport/resource/ABI。
- User-level driver / named pipeline:
  Q51 planning session；`none`不进入本域，search的single-buffer plan使用serialized identity。
- Explicit non-goals:
  不把每条dependency或nonempty multi-slot默认pipeline，不自行选buffer count/final order/worker，不从estimated overlap接受candidate，
  不恢复whole-Module fixed-slot clone API或在actual IR上调用pipeline transform寻找结构。
- Done criteria:
  serialized identity、2+ actual stages、selected multiplicity、prologue/steady/epilogue、Direct-DTE issue/wait、alias/external-write/tail负例和
  Q50.J invalidation/re-entry受测；production assignment与candidate apply对应。旧FixedSlotPipeline source/header/test的独有能力全部映射并
  迁移后，旧接口才能保持退役。

### Q50.K work-item分界

Q50.K由两个work items闭合：

- `serialized-execution`只建立canonical B--H point使用的Serialized execution coverage。输入是canonical `RegionPlan`和
  `TemporalPlan`；输出复用既有`ExecutionInstanceId`，把region中每个required root/merge execution各列一次。root execution必须与
  一个temporal scope一一对应；merge execution执行一次但不伪造temporal scope。ordinary/FD contribution及Maximum/Sum/Accumulator
  component仍属于其root execution和merge execution，不新增独立execution。该plan不含recurrence、stage、launch distance、slot、
  event、worker、resource或actual IR。
- `execution-structure-domain`在event-resource-foundation和storage-domain之后建立完整Serialized/Pipelined结构域、Core consumer及
  K→I→J re-entry起点。此时才引入由E recurrence及connected J event component定义的`PipelineScopeId`；不得把当前
  `ExecutionInstanceId`列表事后解释成pipeline scope。selected construction必须等structure-specific-storage和schedule-domain关闭
  同一个K choice后，才由unified-search-closure接入完整candidate materialization。

当前work item的pipeline contract为：

```text
Pipeline position:
- Upstream IR / input:
  canonical RegionPlan及Canonical TemporalPlan；root/merge execution identity已由D关闭，root local extent/tail已由E关闭，尚无I/J/K
  recurrence、storage、event、stage或schedule事实。
- Current stage responsibility:
  验证region execution与temporal root scopes的一致性，按ExecutionInstanceId稳定排序，为每个required root/merge execution签发且只
  签发一个Serialized execution entry。
- Output IR / files:
  query-local SerializedExecutionPlan；不修改IR、不写attr或文件，不新增另一套execution/scope identity。
- Downstream consumer:
  canonical-storage-plan用该列表建立single-occurrence lifetime；canonical-schedule随后只对这些executions/actions建立source-order/
  worker0点。后续execution-structure-domain吸收该canonical member并在固定K后触发I/J重闭。
- User-level driver / named pipeline:
  无独立pass、pipeline或CLI；none与search的canonical planning prefix静态调用同一constructor。
- Explicit non-goals:
  不识别recurrence，不生成Pipelined choice、stage、prologue/steady/epilogue、slot、event、worker或resource，不调用旧StagePipeline/
  SelectedBufferMaterialization，不把contribution/component拆成execution。
- Done criteria:
  aligned/ragged all-16、multi-root、ordinary reduction、FD coupled merge、merge-only、rank-zero和输入顺序扰动均得到all-and-only stable
  execution集合；duplicate region execution、missing/duplicate root temporal scope及unexpected merge scope得到typed contract failure；
  source IR不变，canonical-storage-plan可直接消费，fresh unit/build与IR/source organization通过。
```

| 覆盖类 | 代表输入 | 必须断言的Serialized结果 | 直接下游witness |
| --- | --- | --- | --- |
| aligned/ragged root executions | rank-3、1024/1025、all-16 Tile | 16个root execution各一次，集合与region executions及temporal scopes精确相等；输入顺序扰动后逐ID相同 | canonical storage逐execution建立single occurrence，不共享ragged Tile identity |
| multi-root/fanout | 1025级多个root/region groups | 每个root/Tile execution保留独立ID，stable semantic sort不按输入walk顺序 | 后续storage/schedule可按同一execution ID连接各自versions/actions |
| ordinary reduction | rank>=3、1024/1025 contributions和merge | contribution仍属于root execution；每个merge group只有一个merge execution且无temporal scope | partial versions与gather在merge execution处收敛，不产生component伪execution |
| FD coupled state | rank-5、1024及1025/1031 K2 | 每个contribution root execution一次，每个coupled group merge execution一次；三个components不增加entry | canonical storage从G/H component versions/actions引用同一root/merge IDs |
| rank-zero/merge-only | rank-zero root及只有merge work的Tile | rank-zero root有一个entry并匹配empty temporal scope；merge-only execution保留且没有伪scope | single-occurrence storage不因empty size或无root scope漏掉合法execution |
| typed failure | empty/duplicate region execution、删除/复制root scope、为merge加入scope | `EmptyExecutionSet`、`DuplicateExecution`、`MissingTemporalScope`、`DuplicateTemporalScope`或`UnexpectedTemporalScope`；不返回partial plan | 修正输入可重新query，source IR不变 |

实现闭合：`SerializedExecutionPlan`直接保存stable `ExecutionInstanceId`集合，canonical constructor只验证region/temporal双射并排序；
没有新增pipeline scope、stage、slot、event、worker、IR mutation或旧StagePipeline调用。fresh定向5/5、完整`WaferUnitTests` 779/779、
default configured lit 224/224、compiler public-header/link smoke、完整configured build及IR/source organization通过。

### K-1 专项调研：serialized/pipelined execution-structure domain

LLVM MachinePipeliner和iterative modulo scheduling把loop-carried dependence distance、resource model和initiation interval作为显式约束；
CIRCT也为`CyclicProblem`/`ModuloProblem`单独建模跨迭代dependence，而不是在普通acyclic order上加一个pipeline flag。pinned MLIR
`scf::pipelineForLoop`只机械生成prologue/kernel/epilogue，caller必须先提供合法schedule。对应本仓，K是execution structure axis，J才是
结构内的resource schedule，I是storage/rotation；current StagePipeline对Instr Module调用SelectedBufferMaterialization的做法没有独立K
domain。

| 调研对象 | 本项采用 | 本项不采用 |
| --- | --- | --- |
| [Swing Modulo Scheduling](https://upcommons.upc.edu/bitstreams/a90677c0-9f85-4a0b-8573-a170b09f0555/download)与[LLVM MachinePipeliner](https://llvm.org/doxygen/classllvm_1_1MachinePipeliner.html) | recurrence dependence distance、stage与target lowering capability必须在选择前显式；lifetime/slot pressure由独立owner验证 | 不迁移MachineInstr、DFA itinerary、register-pressure heuristic或target-specific II；K不拥有cycle cost和winner |
| [CIRCT cyclic/modulo scheduling](https://circt.llvm.org/docs/Scheduling/) | problem input先`check`，selected solution再`verify`；distance、resource和solution property分别建模 | 不引入SSP/LoopSchedule dialect或另一个operator library；Wafer保持typed EventId/plan并把resource order交给J |
| [MLIR SCF software pipelining](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/SCF/Transforms/LoopPipelining.cpp) | 只在caller-owned candidate subtree上消费已经验证的`(operation, stage/order)`；使用其prologue/kernel/epilogue和cross-stage SSA mechanics | 不在source/shared IR试探schedule，不把transform success当legality；失败可能已改IR，因此API必须消费owned transaction |
| [MLIR `scf.for` semantics](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scffor-scf-forop) | static positive trip、iter_args/yield和single-block结构是actual builder的直接合同 | 不把shape、loop名字或block ordinal恢复成selected recurrence |

#### Typed structure

```text
ExecutionStructurePlan
  scopes: ExecutionStructureChoice[]

ExecutionStructureChoice =
  Serialized {
    scope: PipelineScopeId
  }
  | Pipelined {
    scope: PipelineScopeId
    recurrence: OccurrenceRelationId
    iterationClass: { recurrenceAxis, prefixCount, steadyTripCount, tailCount }
    launchDistance: 1
    eventStages: (EventId -> StageId)[]
    lowering: SCFDistanceOne | FiniteUnrolled
  }

PipelineDependence
  source, destination: EventId
  iterationDistance: NonNegativeInteger
  completionKind: DataReady | AsyncCompletion | BufferReuse | Effect
```

`PipelineScopeId`是一个E/I recurrence及其connected J event component，不是edge/node或actual loop pointer。`StageId`从0开始、无空洞；
同stage events由J closure排序。current IR没有表示大于1的logical launch stride，因此唯一可构造值是`launchDistance=1`；不得枚举一个
无法lower的整数再在commit时缩回1。`iterationClass`直接对应E的compact traversal：第一个full wave已经在steady loop前物化，整除case
没有tail，非整除case有一个exact tail，K只改写中间static `scf.for`。例如1024/128为`1+7+0`，1025/128为`1+7+1`，二者都不能把
8/9误当steady-loop trip count。outer active axes重复同一个innermost-axis pipeline，不把Cartesian product展平成另一种loop语义。

#### Eligibility

每个scope无条件有一个Serialized plan。Pipelined只在以下全部成立时进入domain：

1. E提供static finite innermost active recurrence axis、exact prefix/steady/tail class且steady trip足以容纳selected stages；
2. J foundation的每个kernel event可映到该recurrence，所有hard dependencies可标成明确的distance 0/1；observable publication只允许作为
   epilogue cut，event component无unknown effect/call/observer escape；
3. H movement为每个consumer occurrence提供独立data-ready/completion identity，Direct-DTE wait不被NCC join替代；
4. I存在与candidate live distance兼容的slot family/rotation，MustSeparate ranges独立，上一slot在reuse前有completion requirement；
5. synchronous writeback、host/Kcore observer、region/output publication和tail要么形成pipeline cut，要么由typed stage/epilogue relation完整
   表达；
6. target operation/resource contracts允许这些events进入cyclic execution；`NCCSynchronousWriteback`和unknown completion只保留
   Serialized。Direct-DTE issue/completion跨stage时只能使用有明确code-size bound的`FiniteUnrolled`，保证token不成为loop-carried value；
   unknown profitability不影响legality。

“buffer count>1”只满足第4项的一部分。producer/consumer在不同occurrence relations、loop-external assembled consumer、one-trip、unknown
alias/effect或unmatched completion时只保留Serialized，不实际造loop再失败。

#### Complete finite domain

对eligible scope，先按EventId稳定顺序惰性枚举event的ordered set partitions：每个event选择`0..S-1`，stage IDs canonical无洞；
distance 0要求source stage不晚于destination，distance 1允许由下一iteration消费，current不接受更大distance。`S`范围为
`2..min(eventCount, steadyTripCount, targetRepresentableStages)`，`launchDistance`固定为1。每个完整stage plan根据Direct-DTE跨stage
事实选择`SCFDistanceOne`或bounded `FiniteUnrolled`，再交I revalidation和J closure验证storage/resource约束。

```text
structureSuccessors(prefix):
  yield Serialized once
  if scope is structurally eligible:
    choose next event's canonical StageId
    when stage map closes, derive its one supported lowering
    derive cyclic EventGraph and invalidate affected I/J facts
    yield only plans whose structural/dependence checks close
```

ordered partitions最坏为ordered Bell/Fubini number；只能lazy。stage ID表示有序phase，交换两个nonempty
stage会改变执行结构，不能当成label symmetry删除；canonicalization只拒绝空洞stage。不同independent scopes分别选择，不为取得一个
“card pipeline”合并loops。

Serialized与该scope未使用的extra rotating slots extensionally冗余；canonical Serialized要求相关I family回到multiplicity 1，Q51按
K→I invalidation重建。Pipelined计划产生structure-specific occurrence/live distance，I重新选择/验证slot plan，再由J closure调度；若
I/J不满足，只拒绝该K combination，Serialized和其它stage partitions保留。该迭代沿“固定K后重算I/J”闭合，不在K内部贪心改slot/order。

#### Proposals、bounds与failure

proposals包括Serialized、source-order two-stage、load/transfer→compute→store三阶段、critical-path balanced partition、Direct-DTE
issue/independent compute/wait结构，以及donor的drain-minimizing split。每个proposal经exact constructor；estimated overlap只排序。
K可使用J foundation的recurrence/resource lower bound筛选proposal priority，但不局部选winner或删domain。

explicit Pipelined assignment的unmapped event、invalid distance/stage、hard cyclic contradiction是ExactRejection；missing I/J revalidation为
Deferred；target不支持cyclic action为Unsupported；enumeration work-limit为Indeterminate；malformed plan为compiler bug。Serialized不因
另一个Pipelined失败而消失。

#### K-1 Gate

本work item在实现前冻结下面的覆盖矩阵。production structural eligibility只读E的exact occurrence facts和J foundation graph；initial I
definition/use/release已体现在graph中，但structure-specific slot/live-distance compatibility必须由下一项在fixed K后重闭；
1024/1025/1031是相同structure算法的coverage输入，不成为pipeline识别条件。2--6 event只用于完整finite reference oracle。

| 输入等价类 | 代表输入 | structure路径 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- | --- |
| Serialized identity | rank-3 1024/1025、single/multi-root、merge-only | 每个J connected component恰有一个Serialized choice；无recurrence的component仍保留 | missing component/event、duplicate scope | all-and-only scope/event coverage，plan不含stage/slot/order/worker | structure-specific-storage把相关family规范回single occurrence |
| 单recurrence Pipelined | rank>=3，1024整除与1025/1031 tail，steady trip>=2 | innermost active axis的2至bounded stage无空洞ordered partitions；launch distance唯一为1 | unmapped recurrence/event、invalid stage/distance | 1024=`prefix1+steady7+tail0`、1025/1031=`1+7+1`，stage map全覆盖、distance 0/1成立 | I按同一recurrence axis/stage distance重建occurrence/live range |
| independent components | 两个rank-3 1025 roots，共享/不共享Card resource | scope choice做Cartesian product，不合并成card-global pipeline | component overlap | exact plan数等于独立scope option数乘积，输入反转identity一致 | J foundation/shared resource在fixed K后重新连接components |
| 不eligible结构 | 无steady loop/rank-zero、multi-recurrence component、unknown或synchronous-writeback completion/effect、loop-external cut | 只保留Serialized，不通过先造IR再失败发现 | unsupported cyclic action保持typed且不删除Serialized | 关闭/反转proposal不改变domain；multi-slot本身不产生Pipelined | 后续I/J不接收伪pipeline requirement |
| 有界oracle与确定性 | 2--6 events、steady trip 2--7，依赖chain/fanout/independent | independent surjection/distance枚举与production逐plan相等 | successor work-limit为Indeterminate，malformed input为compiler bug | 无stage空洞、无duplicate、stable successor、source IR byte-identical | Core exact successor可继续，不依赖proposal |
| Core invalidation/re-entry | public rank-3 1024/1025 search prefix | J foundation→K first Serialized/Pipelined state；固定K后required coordinate必须是structure-specific storage | 旧EventGraph或pre-K BufferPlan不得被当成closed I/J | work count、state identity和missing coordinate精确；零IR/clone/materialization | `structure-specific-storage`从ExecutionStructureState重闭I，禁止直达schedule |

- independent有界穷举oracle对2--6 events平铺ordered stage partitions并检查distance 0/1 dependences，与production domain逐key一致、
  无stage-label duplicate；Serialized恰一；
- eligibility覆盖single/multi-dimensional waves、nested scopes、one-trip、loop-external consumer、DDR/peer/relay、Direct-DTE token、NCC
  completion、synchronous observer、alias/effect和tail；multi-slot alone不广告pipeline；
- structure choice产生正确cyclic EventGraph、occurrence/live distance并精确invalidate I/J；Serialized重置相关extra slots，unrelated scope不变；
- proposal关闭/反转不改变domain或Q51有界穷举optimum；work-limit不变NoSolution；source byte-identical、零actual IR/clone/default stats；
- actual phase construction、bounded DTE finite-unroll和donor mechanics由K-2。

实现闭合：`ExecutionStructurePlan`现在同时保存per-axis `OccurrenceRelationId`、E-derived `PipelineIterationClass`、distance 0/1
`PipelineDependence`、typed completion obligations、唯一可构造的launch distance 1及`SCFDistanceOne/FiniteUnrolled` lowering。production
descriptor从EventGraph、current TilingInterface/root work和TemporalPlan重建；普通parallel innermost axis无loop-carried edge，reduction axis显式
加入completion(i)→issue(i+1) distance-one edge。component含多个execution recurrence、没有steady loop、unknown或
`NCCSynchronousWriteback`时只产生Serialized，不通过实际改写试错。

domain只对E已经物化的innermost steady `scf.for`枚举ordered stage partitions；prefix与tail保持原位。1024/128与1025/128分别得到
`1+7+0`和`1+7+1`，两者steady trip均为7。Direct-DTE issue/completion跨stage时只有在`events×steadyTrip`不超过显式code-size limit时
产生`FiniteUnrolled`，否则该stage assignment不进入current domain；没有估算resource或SPM合法性控制流。

### K-2 专项调研：phase construction、bounded transport与donor migration

pinned MLIR `pipelineForLoop`文档明确声明：它只机械生成prologue/kernel/epilogue，不决定schedule，并假设caller schedule合法；current
实现还只支持loop-carried distance 0/1、single-block body，且某些失败发生在已修改IR之后。它适合作为selected winner中最小新建loop的
mechanical builder，不适合candidate query/validation，也不能在原source或shared IR上尝试多个schedule。

#### Lowering capability与prepare

K domain只广告有current actual builder的结构：

```text
ExecutionStructureLowering =
  SCFDistanceOne                 // launchDistance=1, dependence distance 0/1
  | FiniteUnrolled               // bounded static steady loop, no token backedge
```

一般stage plan若不属于二者返回Unsupported，而不是等commit时缩成distance-one。完整winner在C mutation前形成：

```text
PreparedExecutionStructure
  scope/loop: PipelineScopeId -> LoopStructureId
  trip/main/tail classes: exact E facts
  eventStageOrder: J-closed (EventId, StageId, in-stage order)[]
  dependenceDistances: verified K facts
  slots: I-closed family/rotation bindings
  completions: H/Q63/J issue-wait-join-release bindings
  lowering: ExecutionStructureLowering
```

prepare模拟pinned SCF schedule checks：每个body event恰有stage、stage无洞、operand/event依赖满足unrolled-cycle relation、loop-carried values
distance在capability内、trip count足够或tail strategy显式、nested region/predication支持明确。Finite-unroll/paired DTE还验证全部participant有同构
occurrence、payload/message/slot selector和completion；不从actual op count猜。

#### Complete-candidate construction

Serialized直接调用E loop builder并按J order发出events；Pipelined有两种窄路径：

```text
emitSCFPipeline(prepared, rewriter):
  build one selected unpipelined scf.for inside the new CardModule subtree
  let I add the prepared slot roots, iter args and yield rotation
  emit D/G/H actions plus J-selected waits/joins in the unpipelined body
  attach every actual operation to its EventId through local typed mapping
  call scf::pipelineForLoop with the prepared (operation, stage/order) schedule
  let I/J place final releases after the resulting epilogue/completion

emitFiniteUnrolledPipeline(prepared, rewriter):
  use the same verified SCF schedule to form prologue/kernel/epilogue
  fully unroll the bounded steady kernel before commit
  leave E's exact residual tail after the software-pipeline epilogue
  bind each Direct-DTE occurrence to its planned slot/message selector
  preserve participant-isomorphic send/recv/wait structure and leave no token block argument
```

SCF builder在新subtree内对selected loop做的mechanical op duplication会成为final prologue/kernel/epilogue；它不是source/Module clone、loser
materialization或replay cache。`modifiedIR` failure也只污染C guard即将擦除的新subtree。可维护性允许后续把它替为direct phase builder，
但不能并存第二套semantics。

I slot allocations在loop外一次创建，slot iter args/yield按selected rotation；J stage内order和resource sequences直接决定kernel order。
Direct-DTE issue token在同一dynamic occurrence由exact wait消费，不跨backedge悬空；NCC participant join只完成Q63 workers。synchronous
writeback、publication和loop-external observer成为prologue/epilogue cut或Serialized scope，不被移入steady kernel。

#### Structure verification

`ExecutionStructureVerifier`从actual SCF/SSA/token/effect重建并逐plan比较：

- every original occurrence的semantic work exactly once，prologue/steady/epilogue iteration coverage无gap/overlap，tail exact；
- event stage/order、cross-stage SSA values、iteration distances及loop-carried results匹配；
- slot family、rotation、alias/range和reuse-after-completion满足I plan；
- DTE sender/receiver finite occurrence shapes、message occurrence、wait/release和relay/gather state一致；
- NCC pending participants、minimum joins、synchronous islands和observable output在scope exit前闭合；
- Serialized scope没有pipeline-only extra work/slots，多个independent scopes不被合并。

随后I lifetime、H movement、J schedule/Q63 completion及Q50.0 memory/transport verifier各在自身IR层fresh执行；K verifier不复制这些完整
分析，只比较structure relations。

#### Donor migration

| Current / donor能力 | 终态owner | 必须迁移的witness | 退役条件 |
| --- | --- | --- | --- |
| 已删除的`StagePipeline` wrapper | K prepare/constructor | Serialized、selected stages、J invalidation | 已删除OwningOpRef module chain与scope-index protocol；CardExecutable不再接收post-hoc buffering scopes |
| pinned `scf::pipelineForLoop` mechanics | K `SCFDistanceOne` builder | prologue/kernel/epilogue、cross-stage SSA、odd/even trip | 只消费prepared schedule，不在source/loser上试探 |
| donor static fixed-slot pipeline | K+I+J construction | 1/2/3/4/odd/even trip、stage shortage、slot permutation、tail | whole-Module clone/provider/public header删除 |
| donor periodic Direct-DTE specialization | K finite-unrolled builder + H/I/J | paired occurrence、selector、residual tail、message/wait | post-hoc all-rank mutation和ordinal matching删除 |
| donor NCC/DTE completion placement | Q63/J closure | same-worker order、NCC→DTE、DTE→consumer/reuse、terminal join | K不重建completion或解析target calls |
| 已删除的selected-buffer stage inference donor | K-1 domain | cross-engine stage/eligibility negatives | production caller、StagePipeline和actual scan均已删除 |
| 已删除的selected-buffer donor tests | K/I/J focused owners | 每项mechanics与negative逐表迁移 | K stage/token与I slot/lifetime witnesses已迁移；J只消费typed completion，不保留旧wait split/rebuild算法 |

#### Complexity、failure与K-2 Gate

SCF construction work/code size为`O(events * stages + slot multiplicity)`，finite-unrolled builder为
`O(events * steadyTripCount + tail)`；phase/trip数均来自finite plan且受work/code-size bound，超限时对应Pipelined plan不进入current
domain，不部分emit后改Serialized。commit失败由C guard回滚，
不返回planner重试。

本轮K-2覆盖矩阵在实现前冻结如下；小loop只用于逐plan机械oracle，同一builder另用rank>=3的1024/1025/1031 typed fixture证明真实prefix/steady/tail：

| 输入等价类 | actual结构 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- |
| SCF distance-one | rank-3 1024/1025/1031、2/3+ stage、distance 0/1 SSA | missing/duplicate event-op binding、stage/SSA contradiction | source loop全部top-level op恰映射一次；software prologue/kernel/epilogue动态coverage恰等于steady trip；E prefix/tail不被复制 | I可在同一loop外创建slot roots，J可按EventId order映射clone |
| Direct-DTE跨stage | static bounded steady trip、issue→wait、aligned/ragged tail | token缺wait、work/code-size超限 | 使用`FiniteUnrolled`；result中无async token loop-carried argument，每个dynamic issue有同occurrence completion | H/J可绑定message/wait，Q50.0 transport verifier读取actual tokens |
| external storage/effect | read-only external、loop-local allocation、caller提供的exact rotating-family proof；另有external write/unknown effect负例 | 无exact family proof的跨stage write为Unsupported | 不按root名字或memref type猜独立；prepare在第一次IR修改前结束全部检查 | I下一项提供actual slot-family proof，不由K分配或猜alias |
| multiple scopes与rollback | 两个independent loops，第二个prepare/apply failure | stale pointer、scope overlap、pinned transform partial failure | input plan/binding先全量prepare；apply消费owned module，失败后无partial module返回；成功结果`mlir::verify`及structure verifier通过 | full-feasibility可在candidate transaction中调用一次，不重放loser |

- actual K tests覆盖Serialized、2/3 stages、distance 0/1 SSA、1024/1025/1031 main/tail、两个independent scopes及cross-stage async token；
  `FiniteUnrolled`结果没有token block argument，phase relation逐event/iteration证明dynamic coverage无gap/overlap。
- negative覆盖missing/duplicate event-op binding、stage/SSA contradiction、trip不一致、external cross-stage write无I proof、stale relation、
  synchronous observer和finite-unroll work limit；prepare在首次修改前检查全部scope，apply消费owned module，pinned failure不返回partial IR。
- external storage正例必须由下一项I的actual rotating-family proof接入；wait/join、resource/control order由J接入。K builder已经把二者设为
  必需typed输入，不建立slot、completion或resource fallback。
- source/call-tree gate已删除old `StagePipeline`、CardExecutable selected-buffer scope参数、post-hoc memory-planning stage insertion及无实际
  producer的rotating-slot statistic；structure-specific-storage随后已删除remaining `SelectedBufferMaterialization` source/test/CMake。

本轮fresh验证：K/domain/materialization及直接上下游59/59；排除future attention production fixture后普通host unit 928/928；default
configured lit 225/225；4个public-header/link smoke、完整configured build、IR/source organization及diff检查全部通过。production
complete-candidate接入仍按既定artifact DAG等待structure-specific-storage与schedule-domain提供同generation BufferPlan/event order；
full-feasibility在它们闭合前继续拒绝Pipelined，而不是偷偷使用旧actual scan。

K不选择global winner。“某case pipeline获胜、另一case因movement/buffer/parallelism选择Serialized或其它region”由Q51 closure证明。

## Q50.J Schedule Closure

### schedule-domain调研：有限偏序、资源实例与completion lifetime

固定K后，J只关闭Wafer IR实际表达的有限选择：worker binding、capacity-1 resource sequence、per-scope control order、receiver FSM lane和
completion boundary。RCPSP、modulo scheduling和machine scheduling都说明“依赖/资源问题”与“选择策略”应分离；它们不构成在Wafer
IR中增加cycle timestamp或idle变量的理由。

| 调研对象 | 采用的工程规则 | 未采用的部分及原因 |
| --- | --- | --- |
| [LLVM MachineScheduler](https://github.com/llvm/llvm-project/blob/main/llvm/lib/CodeGen/MachineScheduler.cpp) | 先建立稳定DAG和resource facts，再由独立策略选择ready node；最终顺序可以在owned IR上统一提交 | 不迁移MachineInstr pointer、register-pressure heuristic或target hook；J的identity是`EventId` |
| [CIRCT scheduling infrastructure](https://circt.llvm.org/docs/Scheduling/) | problem、solution和verification分层；resource constraint必须由solution显式满足 | 不引入第二套schedule dialect、calendar或字符串operator registry |
| [MLIR Async dialect](https://mlir.llvm.org/docs/Dialects/AsyncDialect/) | async completion由exact SSA token表示，wait只能消费对应token | NCC仍使用Q63 participant域；NCC join与DTE wait不能互相清除 |
| [IREE Stream timepoints和resource lifetime](https://iree.dev/reference/mlir-passes/Stream/#iree-stream-propagate-timepoints) | completion可以推迟到first dependent、resource reuse或release之前，不能在issue后默认等待 | Wafer直接保存离散`EventBoundaryId`，不复制IREE dialect或runtime timepoint |

硬件事实只从current target/ABI文档进入hard constraint：每Tile有1个Direct-DTE sender slot和4个receiver FSM；sender programming与remote
receive preparation可以独立发生，但两侧completion都要求matching send和receive已经issue；NCC same-worker普通链由busy-table保持地址
顺序，不能因此插join；Direct-DTE route在没有显式`ExactMovementRoute`时保持opaque。上述事实分别由
`TargetDirectDTEResourceLimits`、Q63和`verifyDirectDTETransportSchedule`的current实现交叉确认。未证实的route latency、barrier、join或wait
语义保持unknown，不进入domain。

#### Fixed-K rebuild与closed plan

Q51固定一个K choice后，按K-1约定先重建structure-specific occurrences，重新关闭I BufferPlan，再重建J EventGraph。只有三者同一
dependency generation时才进入closure：

```text
ClosedSchedulePlan
  structure: selected ExecutionStructureChoice
  buffers: revalidated BufferPlan identity
  workers: EventId -> NCCWorker
  resourceInstances: EventId -> ResourceInstanceId
  resourceSequences: ResourceInstanceId -> EventId[]
  controlOrders: ControlScopeId -> EventId[]
  completionBoundaries: CompletionObligationId -> EventBoundaryId
```

plan不保存start/end cycle、pending sets、makespan或calendar。participant mask由typed obligation、selected worker及same-worker
handoff关系确定；DTE token group在actual EventId binding上重建。相同boundary的completion canonical grouping不形成额外search axis。

#### 有限exact successor

```text
buildScheduleDomain(postKEventGraph, fixedStructure, fixedBuffers):
  validate one fixed K/I generation and all typed completion/resource intervals
  reject a hard cycle or a mandatory receiver-live-range graph that needs >4 colors
  enumerate worker Cartesian product
  enumerate each capacity-1 resource's topological sequence
  add release(previous) -> issue(next), not issue(previous) -> issue(next)
  enumerate each (pipeline scope, Tile/Card) control linear extension
  color receiver intervals canonically with lanes 0..3
  derive the latest boundary before the first hard dependent or exact resource conflict
  return only plans that pass ScheduleDomain::contains
```

successor顺序和tie-break只使用typed semantic ID；输入容器反转不改变plan集合。current唯一明确的capacity-1 transport resource是
Direct-DTE sender slot；粗粒度Tile engine、CardDDR和只有path没有capacity的known link都不生成串行边。receiver pool是4-color interval assignment。`CapacityUnits`不能被误降为单laneexclusive resource；不同receiver lane
不构成completion cut。opaque NoC estimate既不生成resource sequence，也不影响合法集合。
`SPMRangeResource`只表示同一selected storage object内的exact access/alias range，用于completion observer检查；definition/use/reuse顺序来自
SSA与I lifetime hard edges，不再把多个reads和write压成一个resource全序search axis。它不是allocation offset、footprint或capacity proof，
也不推断两个不同allocation object在actual SPM中是否冲突。后者只能由complete candidate的MiniMalloc结果决定。

每个Direct-DTE hop有独立send/receive issue及各自completion。两个issue之间没有硬件同步边；logical source-ready约束root send，relay只在
parent receive completion后issue child send；send和receive completion都依赖matching两侧issue。该关系与actual transport verifier一致，
不会靠“recv必须先执行”或“issue后立即wait”规避deadlock。

completion placement只沿同一control order、同一selected K stage向后移动。遇到stage边界、typed local combine、actual exact-range/resource observer、FSM
lane reuse、相关buffer release或observable terminal即停止；纯结构marker不是同步理由。J不能通过移动wait/join改变已经选定的K structure。same-worker且下一actual
overlapping-SPM consumer使用同一worker时，中间NCC placement的mask为0；其它NCC
placement使用minimum selected worker bit。`NCCSynchronousWriteback`使用selected worker但不生成join；Direct-DTE始终保留自己的token
completion，不能被NCC mask替代。

#### Why no timestamp axis

Wafer当前IR只表达order、resource lifetime、token和wait/join，不表达cycle start time。固定resource/control sequence和completion
boundary后，额外idle既不产生新的IR，也没有current target事实可验证。因此J不枚举timestamp、II、calendar或idle；若未来target增加可验证的
timed contract，应建立独立lowering层表示，不能把性能估计塞进本合法域。

#### Complexity与策略边界

最坏复杂度是worker Cartesian product与resource/control linear extensions的乘积，属于指数/阶乘级。J只提供bounded lazy successor；
`maxSuccessorSteps`耗尽返回`Indeterminate`，不报告optimal。proposal、bound、memo和LNS分别由`search-control-closure`与
`search-scalability`拥有；关闭或更换这些策略不得改变J的plan集合。

#### Failure/no-good

hard DAG cycle、capacity-1 sequence无可行linear extension和mandatory receiver interval超过4路属于`ExactRejection`。stale K/I generation、
unbound completion/resource interval和重复issue属于`BrokenContract`；successor work limit属于`Indeterminate`。J不生成no-good；Q51只能从这些
typed witness构造causal no-good，不能把一个worker/resource order的失败扩大到其它K/I/H sibling。

#### J-closure-1 Gate

本work item在实现前冻结下面的覆盖矩阵。1024/1025/1031只证明同一event/resource closure可消费整除、tail和多Tile输入；
2--7 event仅用于完整worker×resource×control×completion reference oracle。schedule legality不读取cycle estimate、SPM footprint或actual op order。

| 输入等价类 | 代表输入 | schedule路径 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- | --- |
| worker/control domain | rank-3 1024/1025，2个independent worker-capable events及producer→fanout | closed worker set做Cartesian product；per-control scope枚举全部hard-ready linear extensions | empty worker domain、unknown event | 2-event `2!×3^2=18`、fanout `2!×3^3=54`且无duplicate | full-feasibility按EventId绑定selected worker，不从Instr默认恢复 |
| shared resource sequence | rank>=3 1025/1031，多个Direct-DTE sends及SPM read/read/write | one-slot sender users枚举所有与hard DAG一致的resource sequence，并加入release→next-issue；SPM继续使用SSA/lifetime edge | resource/hard cycle | sender sequence all-and-only；read-read不造冲突或全序；TileEngine/CardDDR/SPM/known-link-without-capacity/opaque-NoC均无额外hard sequence | actual emitter按sender order和storage hard edges创建operations |
| multi-scope/cross-Tile | 2个Tile independent scopes、cross-Tile direct/relay；后续vertical覆盖all-16 | control按`(PipelineScopeId, Tile/Card)`分别成序，cross dependency保留在global DAG，不造card total order或CardDDR串行边 | local sequence与hard/resource edge合成cycle | 两Tile各2 events得到`2!×2!=4`且没有跨Tile全序；input反转plan集合相同 | selected construction逐Tile提交，cross-Tile token relation交给Q50.0 whole-card verifier |
| completion与publication | Direct-DTE、NCC worker、same/cross-worker、grouped waits、buffer reuse和observable terminal | boundary下沉到first hard dependent/exact conflict之前的latest位置；participant从typed obligation/selected worker导出 | duplicate/unmatched issue、terminal pending | every issue恰一placement；same-worker handoff mask0；DTE/NCC域不互相清除；两个DTE token可在同boundary合并成一个wait | full-feasibility按同一plan/actual mapping生成minimum wait/join并做Q63 parity |
| fixed K/I generation | Serialized与Pipelined、multiplicity/rotation/lifetime requirement | ClosedSchedulePlan携exact structure+BufferPlan，domain/state双重校验generation | stale pre-K buffer或K sibling | plan identity逐字段相等，slot lifetime lower/upper与selected family一致 | ScheduledState是full-feasibility唯一输入 |
| exact successor与work limit | 2--7 event bounded oracle、输入顺序扰动 | worker→control→resource的finite lazy successor覆盖每个leaf | work-limit Indeterminate、malformed problem contract failure | 与独立permutation oracle逐plan相等；无timestamp/idle/calendar/default stats | Q51直接调用exact successor，不依赖proposal |

#### Domain实现结果

`ClosedSchedulePlan`保存exact `ExecutionStructurePlan`和post-K `BufferPlan` generation、EventId worker/resource bindings、capacity-1 resource
sequences、per-pipeline Tile/Card control orders及all-and-only completion placements。`ScheduleDomain::contains`从原始problem重新验证每个字段；
plan不携带actual operation pointer、timestamp、cost或隐藏epoch。

EventGraph将peer endpoint拆成`PeerSend`与`PeerReceive`。source sender使用1-slot `DirectDTESenderResource`；destination receive使用
4-lane `DTEReceiverFSMResource`。receiver interval按selected control order进行canonical exact coloring；四个强制重叠interval使用lanes
0/1/2/3，第五个强制重叠得到`ExactRejection`，可排序的五个interval复用lane。capacity-1 async resource sequence写为
`completion(previous) -> issue(next)`，不会把issue-to-completion占用错误缩成issue order。

worker/control oracle覆盖`2!×3^2=18`和fanout `2!×3^3=54`；2--7 event逐项与permutation reference比较。两Tile独立control
得到`2!×2!=4`，没有card total order。rank-3 1024/1025/1031 EventGraph和actual Instr fixture覆盖aligned/ragged payload、direct/relay、
same/cross-worker、NCC+DTE separate completion、receiver 4/5 lane及grouped waits。calendar/bound/proposal继续由后续work item拥有。

### selected schedule物化与actual verification

旧`InstructionSchedule`同时从actual operation pointer推断domain、保存snapshot epoch并修改IR，导致问题构造与selected application有两套
owner。现行实现删除这套source/test/API。J domain只由post-K typed facts建立；selected application接收caller显式提供的
`EventId -> operations`关系，在owned candidate transaction中统一重排、设置worker并发射completion。它不从operation name、ordinal、
打印文本或block初始顺序恢复事件身份。

这里采用LLVM/MLIR常见的“先只读prepare，再对owned IR提交变换”，而不是给D/G/H/I/K各复制一套callback emitter。后者会让同一个
operation construction出现第二条路径。所有可能的stale binding、SSA、worker/completion和token检查先于首次mutation；提交失败不返回
partial module。

#### Prepared schedule

```text
PreparedScheduleMaterialization
  plan: ClosedSchedulePlan
  modules: borrowed Tile ModuleOp[] for one immediate epoch
  scopes: ControlOrder + owning Block + all-and-only ScheduleEventIRBinding[]

ScheduleEventIRBinding
  event: EventId
  tile: TileId
  operations: exact top-level operation group; empty only for a structural marker
```

prepare先调用`ScheduleDomain::contains`验证generation和plan。每个control EventId、actual issue operation、worker binding和completion
placement必须all-and-only；每个DTE issue token必须尚无consumer且有唯一`DirectDTE` placement；每个NCC issue必须有selected worker和
`NCCParticipant` placement。输入不得含旧`SyncNCCJoinOp`或`InstrDTEWaitOp`。同一control scope必须落在一个actual block，异步issue与其
boundary也必须同block；不满足时返回typed failure而不修改IR。

#### Owned IR materialization

```text
materializeSchedule(ownedModules, prepared):
  move each exact event operation group into its selected per-scope order
  set the selected worker through WaferNCCIssueOpInterface
  group CompletionPlacement by EventBoundaryId
  for each boundary:
    emit at most one SyncNCCJoinOp for the union of nonzero selected masks
    emit at most one InstrDTEWaitOp for the exact issue tokens in that group
  verify MLIR, actual order/worker/boundary/token parity and Q63 terminal state
  return the owned modules or no module
```

different Tile scopes do not acquire a synthetic card order。cross-Tile readiness仍由explicit send/receive message和actual whole-card transport
relation表达。same-worker handoff的mask0不生成空join；cross-worker只加入minimum selected participant。DTE wait不清NCC pending，NCC
join不清DTE token。两个独立completion落在同一latest boundary时可形成一条join和一条wait；两个DTE token落在同一boundary时形成一个
two-token wait。

buffer release不是J新建的旁路operation；它已经是EventGraph中的typed event及I lifetime边界。selected control/resource order保证release
位于对应completion之后，后续actual SPM/transport verifier检查真实allocation/token lifetime。J不得在失败后retile、spill、换slot或插
global barrier。

#### Verifier ownership

J的plan verifier和actual materialization verifier分别检查：

1. fixed K/I generation、hard/resource/control linear extension、slot-family generation和receiver interval coloring；
2. all-and-only EventId operation groups、actual per-control order、selected NCC worker及completion operation所在boundary；
3. every Direct-DTE issue token恰由其selected group中的一个wait消费，group没有多余token、join或wait；
4. Q63从current module重算pending participants，任何function return前不得有pending NCC work；
5. `mlir::verify`通过，actual IR中不存在未归属于selected group的derived join/wait。

整卡Direct-DTE matching、dynamic occurrence、send/receive preparation、global wait graph、actual receiver conflict coloring、range/address和binding已经
由Q50.0的`verifyDirectDTETransportSchedule`/`bindDirectDTETransport`唯一实现。J不复制该算法，也不声称plan-level EventGraph能替代actual
whole-card proof。`full-feasibility`必须在同一owned candidate上运行该gate，并核对selected J completion/FSM与actual结果；失败保持原typed
分类，不能通过插global barrier修复。

#### Complexity、atomicity与migration

prepare/materialization与event operations、control order和completion groups线性；domain enumeration复杂度已在前述finite successor中说明。
failure消费并销毁当前owned candidate，不换order、worker或K plan；默认不生成schedule dump、calendar或stats。

| Current / donor能力 | 终态owner | 必须迁移的witness | 退役条件 |
| --- | --- | --- | --- |
| 已删除的`CardInstructionScheduleDomain` actual op/block pointer+snapshot | J `EventGraph`/`ScheduleDomain` | 18/54 domains、alias/effect、generation | pointer/window/snapshot/manual epoch已删除 |
| 已删除的`applyInstructionSchedule`隐式domain/apply | `prepareScheduleMaterialization` + owned transaction | selected order/worker/min joins/atomicity | TileMemoryPlanning flag与旧apply API已删除 |
| `InstructionWindowOrder/WorkerChoice` | EventId control/resource/worker bindings | multi-scope/per-Tile exact mapping | operation pointer identity删除 |
| 已删除的`analyzeInstructionResources` | J plan resource facts + Q50.0 actual verifier | engine/worker/SPM/DDR/DTE/known link/overlap | opaque route不再伪装exact link，root-only range推断已删除 |
| Q63 `place/rebuildRequiredNCCJoins` canonical completion | Q63 current-IR analysis；J selected completion groups | same-worker、cross-worker、NCC→DTE、terminal participants | J不调用它作为selected-plan fallback；最终none/search共同入口由attention-production-closure收敛 |
| old ReadyOrder/WorkerPlacement/registry donors | J exact successor | topological/worker alternatives和target contracts | greedy/local winner/clone/static row整岛删除 |
| 已删除的InstructionSchedule tests | J domain/materialization tests | 每项direct witness迁移 | old Instr-first fixture不代签production pipeline |

#### J-closure-2 Gate

- domain tests覆盖18/54、2--7 reference enumeration、capacity-1 release-before-next-issue、两Tile独立control、fixed K/I stale generation、
  4/5 receiver FSM和work-limit typed failure；
- completion/materialization覆盖rank-3 1025/1031 same-worker mask0、cross-worker minimum join、NCC+DTE same latest boundary、two-token grouped
  wait、prepare无mutation失败及verifier tamper；EventGraph另覆盖1024/1025 direct与1031 two-hop relay；
- source gate删除Instr-first domain/apply/snapshot、TileMemoryPlanning schedule flag、`ControlResource`和endpoint合并表示；没有footprint、SPM
  estimate、default join/wait、timestamp、calendar或第二个actual transport verifier；
- `full-feasibility`是直接下游集成owner：它必须把selected schedule materialization接到actual candidate，再运行Q50.0 SPM/DDR/transport/
  target gates。该vertical gate不能由本项unit提前代签。

本项完成后，winner Instr IR是唯一actual schedule事实；`ClosedSchedulePlan`只随当前candidate transaction存在并由下游核对，不写sidecar。

本轮fresh验证：EventGraph/ScheduleDomain/ScheduleMaterialization直接24/24，其中ScheduleMaterialization 6/6；排除用户尚未归入
本项的future attention production fixture后普通host unit 927/927；default configured lit 225/225；4个public-header/link smoke、完整configured
build、IR/source organization及diff检查通过。full-feasibility vertical尚未由本项代签，按线性计划作为下一work item执行。

## Work Item `full-feasibility`：Complete-Candidate Actual Admission（Q50.F）

`full-feasibility`只在一个B–K→I→J assignment已经完整时执行。它不是pure proof builder，而是完整candidate actualization与Q50.0
调用边界；搜索合法集合等于这些actual结果的集合。

```text
Pipeline position:
- Upstream IR / input:
  immutable verified TensorProgram和一个complete physical assignment，包含spatial、region、temporal、layout、movement、storage、
  execution structure、worker/order/completion选择；尚无accepted offsets。
- Current stage responsibility:
  在独立candidate transaction中构造CardModule，降低全部TileRegion到Instr，重建completion，运行actual SPM/DDR/transport/
  target gates，并把typed结果连同current owner relations返回controller。
- Output IR / files:
  Accepted candidate保留其actual CardExecutable/offset；rejected candidate只返回typed witness后销毁IR；不写sidecar。
- Downstream consumer:
  search-control-closure消费Accepted、ExactRejection、Unsupported和Indeterminate；accepted candidate可直接成为winner。
- User-level driver / named pipeline:
  public search driver调用与baseline相同的CardModule-to-executable pipeline。
- Explicit non-goals:
  不构造plan-side resource problem，不估算footprint/lifetime，不运行替代allocator，不根据failure修改candidate内部选择；不在本stage
  为尚未形成完整ScheduledState的attention execution猜worker或completion contract，该接线由attention-production-closure拥有。
- Done criteria:
  每个完整assignment只actualize一次；MiniMalloc是唯一SPM allocator；每个actual demand有current owner；accepted结果不重建；
  exact rejection可重放到相同candidate得到同一typed witness；solver exhaustion不改变合法集合。
```

### Actual evaluation algorithm

```text
evaluateCompleteCandidate(source, assignment):
  candidate = materializeCardModule(source, assignment)
  relations = requireCompleteCurrentBufferRelations(candidate)
  result = compileCardModuleToExecutable(candidate, relations)
  switch result:
    Accepted:
      retain candidate, offsets and actual cost
    ExactSPMCapacityRejection:
      require every conflict/oversized demand has an owner
      return typed feedback
    OtherExactRejection:
      return verifier-owned typed witness
    ResourceExhausted | Unsupported | Indeterminate | CompilerBug:
      return typed non-rejection failure
```

Q49.P对actual SPM rejection生成确定性的下一temporal candidate；Q51对完整domain sibling继续枚举。二者都不得使用rejection的bytes
做线性缩放来预测fit，也不得把一个candidate的allocation/lifetime结论复用到另一个candidate。

### complete-candidate materialization调研与实现边界

| 调研对象 | 采用的工程规则 | 未采用的部分及原因 |
| --- | --- | --- |
| [MLIR Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/) | conversion内部可以延迟replace/erase并回滚pattern；跨stage调用仍须由外层owned IR控制失败边界 | 不把pattern rollback当Card/16-Tile transaction，也不在source/parent上试运行完整candidate |
| [IREE compiler phases](https://iree.dev/developers/general/developer-tips/)与[IREECodegen translation info](https://iree.dev/reference/mlir-dialects/IREECodegen/) | selected configuration在稳定executable/codegen边界进入同一正式pipeline；每个phase有明确input/output | 不复制IREE dialect、runtime或pipeline attr；Wafer的selected facts仍使用自己的typed plan和actual IR |
| [TVM MetaSchedule Builder/Runner](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html) | design-space candidate与真实build/run result分离；只有已构建candidate才产生actual result | 不引入JSON database、跨编译结构hash复用或hardware measurement作为legality；Q50.F只做本次host actual compilation |

本项采用一个typed staged handoff，而不是让Planning复制Q50.0：

```text
PreparedCompleteCandidate
  selected B--K/I/J plans and generation identity
  exact execution -> structured-node relation
  prepared representation/movement/storage/structure/schedule pieces
  all-and-only expected event/version/object/action coverage

materializeCompleteCandidate(source, prepared):
  build one owned CardModule with selected region/temporal/representation
  split once into the selected all-Tile domain
  apply selected movement to current Tile dataflow IR
  lower TileRegion -> Instr once while preserving caller-owned typed relations
  apply selected storage/structure and worker/control/completion to current Instr IR
  run actual SPM -> DDR -> Direct-DTE -> target/resource gates once
  return retained Accepted owner or typed failure; destroy every rejected IR
```

Q50.0提供同一个staged compiler API供none/search调用；selected preparation是完整命名对象，不是多个bool或nullable output。没有selected
assignment时，none构造canonical preparation后调用同一stage implementation，不能另走隐式join/placement pipeline。Planning只贡献typed
prepare/materializer，CodeGen继续拥有Tile splitting、Tile→Instr、SPM/DDR/transport/target lowering。任何stage失败都销毁本次owned candidate，
不修改source、不选择sibling、不调用baseline fallback。

selected relation只在当前candidate epoch内存在：`PhysicalVersionId -> Value`、`StorageObjectId -> slots`、movement action/hop -> tokens及
`EventId -> operations`由实际builder报告；不从operation name、ordinal、Location、shape或打印文本恢复。accepted Instr IR通过SSA、type、
effect、worker attr、token和operation order自包含；plan/relation不写sidecar或package。

### Coverage and gate

本work item施工前冻结下面矩阵；每个positive都从完整`ScheduledState`进入一次独立candidate transaction。shape只提供真实规模覆盖，
任何plan-side bytes/footprint/预测lifetime都不进入结果分类。

| 覆盖类 | 代表输入 | 必须断言 |
| --- | --- | --- |
| actual overfull→fit | 已形成完整ScheduledState的rank>=3 ordinary 1024/1025/1031；FA/FD的ScheduledState与产品证据由第18项闭合 | 每次transition前均有actual capacity rejection；accepted candidate有actual offsets；无estimate调用 |
| relation completeness | Tensor/Cx/NCx operand、result、scratch、movement、output buffers | 每个conflict demand至少一个typed owner；无owner立即contract failure |
| MiniMalloc status | feasible、proven clique/oversized、zero budget/resource exhaustion、invalid result | 只有feasible和proven rejection改变candidate状态；无first-fit/best-fit fallback |
| determinism | 相同candidate重复actualization、不同hash/insertion顺序 | accepted offsets或typed rejection一致 |
| atomicity | Tile lowering、relation remap、completion、SPM/DDR/transport任一失败 | candidate transaction整体销毁；source与其它candidate不变 |
| selected B--I construction | 1024/1025/1031 singleton/coupled/nested/replica、primary/derived layout、DDR/direct/relay/fanout/gather、fresh/alias/reuse/2--5 slots | actual region/version/action/object all-and-only等于plan；每个buffer relation有current owner；不接受“验证plan后发canonical IR” |
| selected K/J construction | Serialized、SCF distance-one、finite-unrolled DTE；same/cross-worker、sync writeback、grouped DTE waits、4-FSM reuse | actual phase/event order、worker、wait/join、slot selector与ClosedSchedulePlan逐项一致；Q63和whole-card transport fresh通过；没有canonical completion残留 |
| Q50.0 stage parity | none canonical preparation与search selected preparation，named leaf pipeline与正式driver | 两policy调用同一split/Tile→Instr/SPM/DDR/transport/target implementation；每candidate各stage一次，关闭selected preparation不改变canonical actual digest |

完成证明必须包含真实规模aligned/ragged full pipeline和actual allocator结果；局部packing unit、shape dump或plan object不能代签。

实现闭合：Card construction已经从baseline owner移到policy-free `PhysicalDataflow`，`CompleteCandidatePlan`携带Region、Temporal、
Representation、Movement、Buffer和ExecutionStructure。Q50.0的同一正式入口在Card split后交出current Tile dataflow view，在
TileRegion→Instr后交出current Instr view；`CompleteCandidatePreparation`在前者重建selected direct/relay/fanout/external-load movement，
在后者构造alias/reuse、rotating slots、selected K phase及J worker/order/wait/join。operation-result relation显式携`resultIndex`；event owner
在storage合并前从current relation捕获，不能靠alias后的共同root恢复。新SPM allocation只在共同`wafer.tile.region`内创建，跨region共享
保持typed Unsupported。

每个stage完成后Q50.0同时运行IR verifier和current-relation检查，再进入唯一MiniMalloc/DDR/transport/target gate。Accepted保留原
move-only executable；actual SPM capacity rejection逐conflict/oversized demand验证structured-node/output→SemanticRoot owner并返回stable
causal roots；其它Unsupported/Indeterminate/CompilerFailure保持原分类。direct coverage包含1024/1025 alias/reuse→MiniMalloc、1031
Pipelined/rotation→selected K→MiniMalloc、1024/1025 external-load fanout donor replacement；full vertical包含1024/1025 accepted、ragged
overfull owner witness、重复actualization determinism、non-first schedule和software relay/fanout→whole-card transport。selected
Region/derived representation、2--5 slot、sync writeback、grouped wait和4-FSM的各owner direct witness与本项vertical共同复核，source在每次
transaction前后保持不变。

attention-specific work在形成完整`ScheduledState`前仍缺target completion contract；该输入缺口由队列第18项按Q50.S设计闭合，不能在
Q50.F用默认worker、Synchronous或结构性join补齐。Q50.F对任何已经完整的ScheduledState不再退回canonical/baseline产物，也不保留第二条
actual compile路径。

## Work Item `search-control-closure`：Actual Result Controller（Q51.Core）

```text
Pipeline position:
- Upstream IR / input:
  complete ScheduledState semantic key和full-feasibility typed actual result；Accepted结果拥有唯一actual executable/cost，尚无public winner。
- Current stage responsibility:
  预留一次actualization、拒绝duplicate key；只让Accepted进入同cohort comparison；记录exact-complete rejection，汇总Unsupported/
  Indeterminate，按objective knowledge发布coverage和move-only incumbent。
- Output IR / files:
  controller不写IR/文件；返回一个retained accepted candidate或typed no-result，以及coverage/counters。loser/rejected executable由RAII销毁。
- Downstream consumer:
  unified-search-closure提供真实ScheduledState traversal并消费retained result；public driver只发布该winner，不重建。
- User-level driver / named pipeline:
  只属于public `search` controller；`none`不构造或调用本controller。
- Explicit non-goals:
  本项不补全轴continuation、不设默认cost rate、不按SPM high-water/bytes lexicographic选winner、不从owner roots推广prefix no-good、
  不读取diagnostic字符串或actual IR pointer作为semantic key。
- Done criteria:
  reserve/evaluate exactly-once；Accepted/Exact/Unsupported/Indeterminate/CompilerBug转移正确；Known同cohort比较与Unknown/Incomparable coverage正确；
  exact-complete cache on/off不改变accepted set；strict bound pruning只在Known bound > Known incumbent时成立。
```

### Controller算法调研与选择

| 调研对象 | 采用的规则 | 未采用的部分及原因 |
| --- | --- | --- |
| [Land--Doig branch-and-bound](https://jmvidal.cse.sc.edu/library/land60a.pdf) | 只有已证明下界不能改善incumbent时才剪枝；Wafer存在semantic tie-break，因此只允许`bound > incumbent`，相等仍访问 | 本项不构造松弛问题或预测bound；Q52只有提供typed admissibility proof后才能调用seam |
| [Chu--Stuckey inter-instance nogood learning](https://people.eng.unimelb.edu.au/pstuckey/interprob/interprob.pdf) | learned nogood必须由能解释推理/冲突的assignment literals支持 | actual SPM causal roots不是prefix assignment explanation，不能把一个完整candidate rejection推广到相同root、shape或任一parent prefix |
| [OR-Tools CP-SAT statuses](https://developers.google.com/optimization/cp/cp_solver) | 区分optimal、feasible、infeasible、model-invalid和unknown；budget/unknown不能报告infeasible | 不引入CP-SAT model或solver；只采用typed coverage语义 |
| [TVM MetaSchedule candidate/result接口](https://tvm.apache.org/docs/reference/api/python/meta_schedule.html) | candidate generation与Builder/Runner result分离，result按candidate一一回传给search strategy | 不引入measurement database、结构hash、warm start或runtime measurement；Wafer legality只消费本次Q50.F actual result |
| [Halide autoscheduler tree search](https://halide-lang.org/papers/halide_autoscheduler_2019.pdf) | 证明search policy可以与schedule representation/actual result分层 | beam截断只适合quality policy，不能作为Q51.Core合法覆盖或NoFeasible证明；留给Q52 measured policy |

采用的controller状态机如下：

```text
CompleteCandidateKey =
  Spatial + Region + Temporal + Representation + Movement +
  InitialBuffer + ExecutionStructure + PostKBuffer + ClosedSchedule

reserve(key):
  require key.schedule generation == key.(structure, postKBuffer)
  reject completed/in-flight duplicate
  consume exactly one actualization credit and mark in-flight

record(key, typed actual result):
  require exactly one matching in-flight reservation
  Accepted       -> compare only Known objectives from the same cohort
  ExactRejection -> store exact full-key nogood and typed witness
  Unsupported    -> count only; do not learn
  Indeterminate  -> count and make coverage incomplete
  CompilerBug    -> poison controller and release every retained loser

finish(frontier status):
  require no in-flight candidate
  move exactly one incumbent, if any
  distinguish comparable-best / feasible-unranked / feasible-partial /
              no-feasible / incomplete-no-candidate / failed
```

`CompleteCandidateKey`是唯一semantic tie-break和exact-cache key；不包含pointer、ordinal、diagnostic、actual offset或cost。Q52可以提供
同cohort的typed admissible lower bound，但不能绕过key、result taxonomy或coverage状态机。

| 输入等价类 | 代表输入 | controller路径 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- | --- |
| exactly-once ownership | rank-3 1024/1025 accepted state及duplicate key | reserve→actual result→move-only incumbent；同key第二次reserve拒绝 | unreserved/duplicate finish为compiler bug | evaluations、actualizations、accepted计数逐项一致；winner只move一次 | unified-search可直接take retained executable |
| actual result分类 | Accepted、SPM ExactRejection、其它exact、Unsupported、Indeterminate、CompilerBug | 只有Accepted比较；exact只记完整plan；unsupported/indeterminate只降coverage | compiler bug终止且无winner替换 | forbidden size、incumbent、coverage和各类count all-and-only | public result不会把unsupported当no-solution/fallback |
| explicit cost cohort | actual instruction/DDR/NoC metrics Known/Unknown/overflow | 只有显式rates且所有required metrics Known时产生Known ticks；否则Unknown | rate zero/overflow/metric unavailable | Better/Worse/Equivalent/Incomparable及semantic tie确定 | Q52可替换cohort facts而不改domain |
| admissible bound seam | Known/Unknown complete-plan lower bound和Known/Unknown incumbent | strict `bound > incumbent`才允许prune；equal继续 | cohort mismatch/unknown不prune | 关闭bound只增加work，不改变selected accepted key | unified frontier复用同一predicate |
| exact-complete feedback | 相同schedule但upstream axis不同的siblings、重复SPM witness、owner roots不同 | cache key为完整CompleteCandidateKey；只做exact equality/duplicate | incomplete axis或stale K/I/J generation在key factory拒绝 | rejected point命中自身、不命中任一sibling；删除cache不改accepted set | item17 traversal保留parent continuation |
| deterministic finish | 2--7 synthetic accepted results、输入反转、mixed comparable/incomparable | Known objective优先；Equivalent/Incomparable用完整semantic key确定commit但coverage区分 | 无accepted、allowance exhausted | winner key/order-independent，coverage为ComparableBest/FeasibleUnranked/NoFeasible | public search一次发布并准确报告coverage |

实现闭合：`CompleteCandidateKey`逐值携带Spatial、Region、Temporal、Representation、Movement、initial Buffer、ExecutionStructure、
post-K Buffer和ClosedSchedule；factory复核schedule generation及post-K只能改变slot families。`UnifiedSearchRunner`已经从真实
`ScheduledState`构造该key，controller reservation、completed set、exact cache、semantic tie和retained winner全部只使用这一类型；相同
ClosedSchedulePlan但其它axis不同的candidate不会互相命中。

actualization credit在启动前预留；duplicate、exhausted和closed reservation分别计数且不启动candidate。只有
`FullFeasibilityStatus::Accepted`、typed compilation status和实际executable三者一致的结果进入incumbent；ExactRejection必须携
`ProvenExactRejection`，SPM capacity还必须有causal owner。Unsupported/Indeterminate不写cache，CompilerBug poison controller。
accepted executable在replace/finish中保持move-only，loser由RAII销毁。exact cache具有显式Enabled/Disabled policy，两种模式的accepted
集合、winner和coverage一致。

`SearchCostCohort`必须由caller显式给出四个positive target tick rates；actual instruction/DDR/min-hop metrics任一Unknown或checked arithmetic
overflow即得到无value的`UnknownSearchObjective`。同cohort Known objective才比较Better/Worse/Equivalent；Unknown或cohort不同为
Incomparable，commit只用CompleteCandidateKey确定，但coverage标`FeasibleUnranked`。`SearchLowerBound`显式绑定完整key；strict bound
predicate仅在Known同cohort且`bound > incumbent`时为true，相等、Unknown或cohort mismatch都不prune。finish输入使用typed
`SearchFrontierStatus`并区分ComparableBest、FeasibleUnranked、FeasiblePartial、NoFeasible、IncompleteNoCandidate和Failed。Q52在有
admissibility proof前不得构造Known bound。

direct coverage以1024/1025 key逐axis检查值身份和stale generation，2--7 bounded independent oracle在反向输入上独立计算objective/key winner；
另覆盖Known/Unknown/overflow、strict bound、相同schedule的full-key sibling、SPM/其它exact/Unsupported/Indeterminate、cache on/off、malformed
Accepted、allowance及全部count。真实1024/1025 `FullFeasibilityTest`已走key→reserve→record→finish→take retained executable；current
`UnifiedSearchTest`和public `SearchRoutingTest`直接消费同一controller。全轴continuation的可恢复遍历仍由下一项
`unified-search-closure`拥有；本项没有用controller oracle代签frontier coverage。fresh direct controller 7/7、controller/
full-feasibility/unified/public routing integration 17/17、排除用户未提交attention fixture的host unit 935/935、core lit 225/225、
public link 4/4及source/IR organization均通过。

## Work Item `unified-search-closure`：Typed Traversal and Winner Handoff（Q51）

本work item施工前冻结覆盖矩阵。production traversal必须调用每个current domain自己的exact cursor，不从plan vector、variant index或
actual IR恢复“下一个”选择；bounded只暂停continuation并降低coverage，不把未访问集合说成empty。

| 输入等价类 | 代表输入 | traversal路径 | typed failure | 精确断言 | 直接下游witness |
| --- | --- | --- | --- | --- | --- |
| complete dependency chain | rank-3 1024/1025 ordinary single-root | Spatial→Region→Temporal→Representation→Movement→InitialBuffer→K→post-K I→J→actual | 任一axis compiler bug终止 | 每个child由parent domain contains；ScheduledState actualize一次 | public search直接take retained executable |
| sibling continuation | 2--3 choices/axis tiny oracle、schedule 2--7 leaves | child返回后parent cursor继续，K sibling各自重建I/J | Unsupported/exact只丢当前complete point | visited semantic keys与flat reference相等、无duplicate | controller cache不删除合法sibling |
| actual feedback | first accepted、actual SPM exact rejection、Unsupported schedule | Accepted进入controller；exact记录complete key；unsupported继续 | Indeterminate暂停并保留coverage | actualization count等于Granted keys；source byte-identical | result coverage准确而无baseline fallback |
| work allowance | step前0/1/N credits、candidate边界耗尽 | 每个successor/actualization前reserve；耗尽不修改cursor | exhausted不是NoSolution | resume后visited前缀稳定；有winner为FeasiblePartial，无winnerIncomplete | Q52可替换measured policy而不改语义 |
| deterministic handoff | proposal/input/Tile顺序扰动、无cost cohort | frontier按semantic key，controller按objective/semantic tie | incomparable不宣称best | winner key与输入顺序无关；只存在一个move-only executable | driver不重建Q50.0/offset |
| public routing | explicit search与independent none，同一FP16 1024/1025 input | search只调用unified driver；none只调用baseline | search失败不返回none产物 | call-tree/diagnostic、candidate_actualizations、package path分离 | attention-production-closure复用同一路由 |

### Traversal调研与实现选择

| 调研对象 | 采用的规则 | 未采用的部分及原因 |
| --- | --- | --- |
| [Gecode search restoration](https://www.gecode.org/doc-latest/MPG.pdf) | search必须保存足以回到parent并访问下一alternative的path/choice状态 | 不clone或recompute candidate IR；Wafer各axis已经有value-owned continuation/cursor，直接保存在显式stack中 |
| [OR-Tools all-solutions与stop callback](https://developers.google.com/optimization/cp/cp_tasks) | “停止在N个解”与“完整枚举”是不同状态，stop后不能报告exhausted/infeasible | 不引入CP solver；采用typed Paused/FrontierExhausted/AcceptedCheckpoint/Indeterminate/CompilerBug |
| [TVM MetaSchedule SearchStrategy](https://tvm.apache.org/docs/reference/api/doxygen/classtvm_1_1s__tir_1_1meta__schedule_1_1SearchStrategyNode.html) | candidate generation与runner result notification分开，search strategy持有跨batch状态 | 不使用measurement batch/database；每个Wafer complete key仍同步调用一次Q50.F actual gate，result立即交给Q51.Core |
| recursive DFS | canonical child顺序和实现简单 | C++调用栈销毁continuation，cutoff后只能从root重放且可能重复actualization，因此退役 |
| best-first/beam | 可改善time-to-first | Q51 correctness先使用显式deterministic DFS stack证明完整和resume；priority/memo/LNS由Q52基于profile加入，beam不能签发exhausted coverage |

采用的session状态机：

```text
frontier stack = [SpatialFrame]
resume(credits):
  before each axis successor or actualization, consume one credit
  top continuation emits at most one child and remains below that child
  child pushes the next-axis continuation
  exhausted/unsupported parent pops; indeterminate preserves terminal coverage
  ScheduledState -> CompleteCandidateKey -> reserve -> Q50.F once -> record
  exact rejection records only full key; canonical parent continuation stays
  budget exhaustion leaves the complete stack untouched and returns Paused

finish():
  FrontierExhausted -> controller.finish(Exhausted)
  Paused/AcceptedCheckpoint/Indeterminate -> controller.finish(Incomplete)
  move winner once; destroy stack and all loser state
```

test-only trace只有显式传入时才复制prefix values和actual key/status；普通compile不构造plan ledger。独立reference composer使用第二个
PlanningSession及自己的递归nested loops，逐parent调用同一已由各Q50 direct oracle证明的successors，但不调用UnifiedSearchSession、
ActualResultController或production stack。两者比较每层prefix set和complete key set；actual truth再由每个reference complete state调用同一
Q50.F一次得到。

实现闭合：`UnifiedSearchSession`以显式variant stack持有Spatial root及Region→Schedule continuations；一次`resume(credits)`在每个axis
successor或complete actualization前消费credit，budget为零不修改stack。parent continuation留在child下方，child exhaust后继续同一cursor；
不再依赖C++递归栈或从root重放。`UnifiedSearchResumeStatus`区分Paused、FrontierExhausted、AcceptedCheckpoint、Indeterminate、CompilerBug和
Finished，`finish`只移交一次controller winner。持久session复制frontend verification value和ExecutionConfig，避免借用调用表达式临时值；
TensorProgram、diagnostics和ProgramData仍由外层transaction持有。

每个ScheduledState先生成CompleteCandidateKey并reserve，随后只调用一次Q50.F。exact rejection不再按causal root调用temporal refinement；
canonical parent continuation自然访问其它temporal siblings。fresh-session direct actualization会从state逐层重建Region/Temporal/
Representation/Movement/Storage/K/post-K I/J domains，不依赖之前遍历偶然填充的cache。Temporal nested choice返回typed Unsupported时携带可继续的
cursor，Indeterminate仍保留当前point而不跳过。

public `search`使用typed `SearchTerminationPolicy::FirstAccepted`作为明确anytime checkpoint，因此无显式cost cohort时继续报告
`FeasiblePartial`，不声称best/optimal；同一resumable mechanism也支持Exhaustive。test-only trace未传入时不保存prefix/key ledger。
独立recursive composer逐parent组合真实Q50 continuations，production在每个单credit cutoff得到相同prefix和前两个complete keys；Oracle B对
fresh parse的每个state各调用一次Q50.F，typed result与production一致，Oracle actualization前后重跑的key/status不变，winner handoff一次。

本项已删除被current Q50 owner替代的legacy Buffering、CoupledRegion、DataMovement/SimpleRoute、PhysicalRepresentation、TemporalTiling、
UnifiedPhysicalDataflow和numeric ComputeImplementation search donors及对应tests；它们的domain/apply witness分别由Storage/StructureSpecific,
Region/SelectedRegion, Movement/MovementTransferBuilder, Representation/PBQP/PhysicalVersion, Temporal及UnifiedSearch suites承接。attention
alternative/clone donor仍有第18项独有迁移证据，明确留给紧接的attention-production-closure，不在本项提前删除。显式reciprocal
lowering仍由Conversion direct test覆盖1024/1025正例和非`1/x`负例，但不再存在compute implementation search axis。

fresh `UnifiedSearchTest` 5/5包含1024 first-accepted one-shot、每credit resume、两次parse determinism及parent-by-parent/Oracle-B前两个
complete keys；controller/session/temporal/public routing定向30/30通过。排除用户未提交attention fixture的host unit 918/918、core lit
225/225、public link 4/4及source/IR organization通过；public 1025 search source→package→no-card也使用本轮产物实际执行。

当前production checkpoint只发布FeasiblePartial；measured priority、memo/DP/LNS和有界质量由`search-scalability`拥有。
Region/layout/movement/storage/pipeline/schedule selected constructors已经接入full-feasibility；当前Unsupported只保留给各owner明确排除的
IR/target语义和跨TileRegion storage sharing，不能fallback canonical产物。attention在形成ScheduledState前缺失的typed completion接线由
attention-production-closure拥有；Q52不能把该缺口或FeasiblePartial改写成全domain optimal。

## Q51 Planning and Search

### Pipeline contract

Pipeline position:
- Upstream IR / input:
  immutable normalized card-local TensorProgram、显式semantic root ops、current target/topology/resource facts，以及已经分别闭合的
  Q50.S fixed semantic/query/decomposition合同与Q50.A--K typed physical domain/query/materializer contracts；没有baseline executable、
  CardModule、Instr或accepted offsets。
- Current stage responsibility:
  建立一次search-owned planning problem，在typed partial states上惰性展开真实Q50 choices。partial state只做结构、语义、cost与
  可证明bound query；一旦assignment complete，就进入Q50.F actual admission，构造candidate CardModule并运行Q50.0实际
  SPM/DDR/transport gate。controller消费actual Accepted或typed rejection/failure，比较accepted actual results并确定唯一winner。
- Output IR / files:
  partial state不写IR/文件/sidecar。每个complete candidate拥有独立disposable transaction；Accepted结果保留move-only
  CardExecutable、offset和actual cost，rejected/loser transaction销毁。session结束只返回一个retained winner或typed failure。
- Downstream consumer:
  Q59 target/package transaction直接消费retained accepted winner；不重新运行candidate materialization、Q50.0或offset planning。
- User-level driver / named pipeline:
  `wafer-compile`的public `search` policy；`none`进入独立baseline controller。两者复用同一actual candidate materializer和named
  lowering subpipelines，但policy controller绝不互调。
- Explicit non-goals:
  不把search写成operation pass，不在partial frontier或memo保存IR/offset，不执行baseline取得初值或fallback，不用runtime provider
  registry/opaque candidate bag，不让proposal或局部mechanism选择winner，不用footprint/容量估算替代actual admission，不把optional
  report/statistics变成默认编译路径。
- Done criteria:
  source在全部candidate transactions期间byte-identical；partial state只含typed assignments；每个complete candidate actualize/Q50.0
  各一次，actual rejection只在owner relation完整时反馈；rejected/loser IR全部销毁，只发布一个accepted winner且不重建。
  baseline/search transitive call graph独立，有界穷举oracle证明partial state、complete candidate、actual result与winner coverage。

### Q51-1 专项调研：driver边界、PlanningState与ownership

MLIR pass infrastructure要求operation pass只处理其anchor及nested IR，pass failure会停止pipeline；它适合对一个candidate IR实例做analysis/
transformation，不适合持有全局frontier、比较多个results或拥有外部work allowance。MLIR的named/nested pipeline承载每个complete
candidate的actual leaf transformations。LLVM MachineScheduler则把ScheduleDAG、ready state和`MachineSchedStrategy`分开：问题/合法依赖
不因选择策略改变。Q51采用同一分层，但planning对象是card-wide typed plans，controller是compiler driver library，不是假装成一个大pass。

owner按lifetime固定：compiler driver创建并关闭planning session，拥有frontier、budget、actual result comparison、winner handoff和output transaction；
session拥有immutable source borrow、typed problem、target/cost cohort及`ObservedDependencyKey` memo；MLIR pass/analysis只处理当前
candidate IR epoch。source analysis、session memo和candidate IR不共享analysis对象、pointer identity或offset；candidate transaction结束即
销毁自己的analysis，session memo只保存可从immutable source和typed partial assignment重算的非IR结果。

current `runCardExecutableSearch`已经具有每个complete point生成CardModule、运行Q50.0并把accepted executable放进incumbent的必要骨架；
需要替换的是semantic alternative clone、坐标不完整、raw evaluation cap、relation缺口和失败分类。终态保留complete-candidate actual
evaluation，改由typed work allowance计数，并保证每个candidate一次、loser销毁、accepted winner不重建。

#### Immutable planning problem

```text
PhysicalDataflowPlanningProblem
  source: borrowed verified TensorProgram lifetime-bound to outer transaction
  program: StructuredDAG + SemanticRootKey/value-use keys + fixed attention facts + exact output/effect facts
  target: immutable topology/memory/operation/transport/resource facts
  mechanisms: statically composed Q50 query functions

PhysicalDataflowPlan
  spatial: SpatialPlan
  regions: RegionPlan
  temporal: TemporalPlan
  representations: RepresentationPlan
  movement: MovementPlan / TileCommunicationPlan actions
  buffers: final structure-specific BufferPlan
  executionStructure: ExecutionStructurePlan
  schedule: ClosedSchedulePlan
```

`mechanisms`是编译期直接调用关系，不是runtime registry、字符串factory或用户selector。planning problem在outer source lifetime内
借用source operation，并通过与pass analysis wrapper共用的policy-free builder建立session-owned typed facts；它不持有
`Analysis *`或在pass外构造`AnalysisManager`。任何state/key/result只使用`SemanticRootKey`及其它typed semantic keys。
ProgramDataHandoff、output directory和target/package owner不进入planning problem。

#### Closed typed state variants

partial state不用一个含十几个nullable fields的bag，也不保存parent transition。frontier element是下列closed variant之一；每个variant
只存在于其前缀合同已经验证时：

```text
PlanningState =
  SpatialState
  | RegionState
  | TemporalState
  | RepresentationState
  | MovementState
  | InitialBufferState
  | ExecutionStructureState
  | BufferState                 // K-fixed后重建
  | ScheduleState

SpatialState            = immutable problem + SpatialPlan
RegionState             = prior + RegionPlan
TemporalState           = prior + TemporalPlan
RepresentationState     = prior + RepresentationPlan
MovementState           = prior + MovementPlan
InitialBufferState      = prior + pre-K BufferPlan and event-resource-foundation eligibility
ExecutionStructureState = prior + ExecutionStructurePlan
BufferState             = structure-specific occurrences + reclosed BufferPlan
ScheduleState           = prior + ClosedSchedulePlan
```

partial-feasibility在A--E之后只检查结构坐标完整性；J foundation是从`InitialBufferState`重算的derived problem，不进入state。
`ScheduleState`形成complete `PhysicalDataflowPlan`后立即交给Q50.F actual admission。严格prefix顺序不删除组合：前一轴的所有siblings
仍在frontier，后轴失败回到共同owner继续展开其它prefix；K改变occurrences后明确经过新的`BufferState`而不是复用pre-K plan。

每个axis plan是immutable value object，可由多个states共享`shared_ptr<const PlanT>`以减少复制；共享的是该轴的语义值，不是parent链、
transition history或mutable cache。equality、canonical order和dedup逐typed contents比较，pointer/hash只作局部加速且必须以完整相等检查
消除collision。state不保存proposal来源、ordinal、failure history、score、lower bound、work count或analysis snapshot。

#### Ownership与lifetime

| Owner | 独占/借用内容 | 明确不拥有 |
| --- | --- | --- |
| outer compilation transaction | source `OwningOpRef`、ProgramDataHandoff、target/package/output transaction | frontier、candidate executable |
| `PhysicalDataflowPlanningProblem` | immutable analyses/facts；在outer lifetime内borrow source | IR clone、winner、offset |
| `PhysicalDataflowPlanningSession` | frontier、typed state memo、actual rejection cache、work allowance、accepted-result incumbent | baseline result、package |
| `PlanningState` | immutable typed axis plans及variant tag | parent/history、derived analysis、IR pointer |
| complete `PhysicalDataflowPlan` | 当前candidate的closed assignment；只用于本次actualization | solver witness/offset、source ownership |
| candidate transaction | 新Card subtree、plan-ID→actual Value/Event/current buffer relation短生命周期映射 | 其它candidate、fallback selector |
| Q50.0 | move-only CardModule→accepted CardExecutable或typed actual rejection | planning frontier、repair path |
| retained winner | accepted CardExecutable、actual offsets/cost和semantic tie key | rejected/loser IR、重建recipe |

session销毁时frontier/memo/rejection cache全部释放。每个candidate transaction不修改source；失败或落选时由RAII owner整体销毁，accepted
incumbent在被更优结果替换时同样销毁。最终winner直接移交outer transaction，Q50.0和offset planning不得重跑；ProgramDataHandoff只在
最终发布时移动，search不能为candidate复制或提前移动它。

#### Policy isolation与driver routing

```text
normalized TensorProgram
  +-- optimization=none
  |     -> canonical candidate -> actual admission -> typed feedback loop
  |
  +-- optimization=search
        -> partial frontier -> complete candidates -> actual admission/comparison

one retained accepted CardExecutable
  -> target/package publication transaction once
```

两个controller只共享immutable analyses、complete assignment schema、candidate materializer、actual admission和downstream publication。
baseline assignment/result不进入search session，search不
调用baseline取得incumbent，也没有失败fallback。外层matched test可以用两个独立source transaction比较结果，但比较不进入任何controller。

#### 文件边界与current migration

```text
lib/Wafer/Planning/PhysicalDataflow/Search/
  PlanningProblem.*       // immutable input/facts
  PlanningState.*         // closed variants/final plan schema
  PlanningSession.*       // frontier ownership/work/result handoff

lib/Wafer/Planning/Baseline/
  ...                     // independent canonical controller

lib/Wafer/Conversion/WaferTensorProgramToTileRegion/
  ...                     // policy-free complete-candidate construction
```

search result原位演进为move-only accepted CardExecutable + actual cost/semantic key/coverage的typed结果，不改成plan-only output；
`CardExecutableSearchSummary`/IR trace移到显式instrumentation/report owner。旧
`UnifiedPhysicalDataflowAssignment/Domain`不作为新state壳复用：其Cartesian successor、materialize、buffer-scope build拆回对应Q50 owner，
完成donor mapping后删除。`TensorProgramAlternativeDomain`及clone materialization由Q50.S common graph normalization、fixed
algorithm fact和single-winner selected decomposition替代，不进入PlanningState。
`SearchWorkBudget.maximumEvaluations`退出semantic/API；后续Q51-5定义planning work allowance，Q52才依据profile给production policy。

### Q51-1 Gate

- type/unit检查每个variant只能从合法前缀构造，K后必须重闭I/J；invalid cross-generation plan在建state时作为compiler bug失败；
- ownership test覆盖source/session/complete assignment/candidate transaction/Q50.0/retained winner/publication的move顺序、early failure和析构；
  无同candidate重复actualization、winner重建、ProgramData提前move或source use；
- state key在input order/hash/parallel discovery变化下稳定，pointer collision不影响equality；复制/共享plan不携parent或mutable cache；
- call-tree/static source gate证明search partial expansion不含baseline、CardModule/Q50.0/materialize/clone，complete state只经injected
  evaluator进入共同actual path；baseline不含search session/state；
- production driver两种policy分别产生retained accepted result后汇合到publication；named leaf pipeline与candidate evaluator调用同一builders，
  未手工拼第二链；
- source files按problem/state/session/controller/materializer分离，public headers自包含且CMake无Planning↔CodeGen循环；普通调用无summary、trace、
  statistics或IR dump对象。

### Q51-2 专项调研：typed transition、dependency expansion与完整性

CSP的maintaining-consistency方法在每次赋值后传播已知约束，但仍靠系统backtracking覆盖其它值；AND/OR search把独立子问题显式成AND
branches，可能比纯Cartesian OR tree小得多；OR-Tools的fixed search与all-solutions模式也明确区分“选择顺序”和“是否枚举全部解”。对应
本仓，Q50 mechanism负责一个轴的exact successor和local propagation，Q51只组合真实轴；proposal可以改变先后，不能替代canonical
enumeration。A*/best-first priority留给Q51-5，本节只定义不会漏state的transition kernel。

#### Static dependency order

真实依赖在编译期闭合，不用provider registry动态求拓扑：

```text
Spatial
  -> Region
  -> Temporal
  -> Representation
  -> Movement
  -> InitialBuffer
  -> ExecutionStructure       // consumes J foundation derived from prefix
  -> Buffer                   // rebuild structure-specific occurrences
  -> Schedule                 // rebuild J graph and close schedule
  -> ActualAdmission            // complete assignment only
```

每个箭头表示child domain的全部inputs已存在，不表示前轴local winner。A--E之后的partial-feasibility只检查结构坐标完整性；完整
Schedule assignment才进入ActualAdmission。`RequiredCoordinate`必须指向当前variant以后、且依赖已满足的真实coordinate。要求已关闭的
coordinate或越过未满足dependency是内部合同错误。若未来加入新axis，必须同批扩closed state variant、dependency table、successor、
structural requirement和tiny oracle，不能注册一个opaque optional provider。

#### 每个axis同一query合同

Q50 exact domain对Q51暴露窄而具体的四个动作；C++用overload/concept或各axis typed free functions静态组合，不用virtual registry：

```text
getProposals(parent, allowance) -> checked Choice[] | Indeterminate
getFirstChoice(parent)          -> Choice | EmptyDomainWitness | failure
getNextChoice(parent, current)  -> Choice | Exhausted | failure
contains(parent, choice)        -> bool / typed validation failure

makeChild(parent, choice) -> next closed PlanningState variant
```

`Choice`就是对应Q50 plan type，不是tag+payload。`getFirst/getNext`定义canonical complete enumeration；proposal必须先经`contains`并产生普通
domain member。`EmptyDomainWitness`只有在domain完整构造且证明无choice时才是exact rejection；work limit、analysis precision或proposal
停止都不能返回Empty/Exhausted。proposal allowance耗尽只结束该批proposal，随后仍进入canonical enumeration；只有exact successor自身
所需analysis未决时才暂停continuation。

#### Continuation属于session control

惰性枚举不能把iterator stack、proposal ordinal或“下一个child”塞进candidate state，也不能每次resume从头扫完整domain。session拥有
closed typed continuation：

```text
AxisContinuation =
  SpatialContinuation
  | RegionContinuation
  | TemporalContinuation
  | RepresentationContinuation
  | MovementContinuation
  | InitialBufferContinuation
  | ExecutionStructureContinuation
  | BufferContinuation
  | ScheduleContinuation

ConcreteContinuation
  parent: immutable typed prefix value
  canonicalCursor: NotStarted | LastChoice(Choice) | Exhausted
  emittedProposalKeys: session-local exact semantic keys
```

continuation持有parent的axis plan refs和last concrete `Choice`，不持有operation pointer、domain object、generator closure或materialized IR。
derived domain可在resume时由parent重建，或由session-local memo按完整parent key复用；memo不进state。continuation销毁不会改变任何child，
也不是commit所需的replay recipe。

一次resume最多发出一个新child，保证global controller能在不同parents间公平交错：

```text
resume(continuation, allowance):
  while allowance permits one successor step:
    if an unused checked proposal exists:
      choice = stable-first proposal
    else if cursor is NotStarted:
      choice = getFirstChoice(parent)
    else:
      choice = getNextChoice(parent, cursor.lastChoice)

    if choice is EmptyDomainWitness:
      return ExactRejection(parent, witness)
    if choice is Exhausted:
      mark continuation exhausted; return NoChild
    if choice is Indeterminate/work-stop:
      preserve cursor; return Paused

    cursor = LastChoice(choice) only for canonical enumeration
    if semantic choice key was already emitted:
      continue
    child = makeChild(parent, choice)
    result = run applicable F foundation checks(child)
    return classifyAndEmit(child, result, updated continuation)
```

proposal emission不移动canonical cursor；canonical enumeration遇到同key只去重后继续。改变、关闭或反转proposal providers不改变最终child
set。并行proposal/query返回先按完整semantic key归并，再选stable-first；完成线程顺序不可观察。

#### Typed transition outcomes

```text
ExpansionOutcome =
  ChildReady(PlanningState, AxisContinuation)
  | ChildDeferred(PlanningState, NonEmptyVector<RequiredCoordinate>, continuation)
  | ExactRejection(FeasibilityWitness, continuation)
  | Unsupported(UnsupportedFeature, continuation)
  | Indeterminate(IndeterminateCause, preserved continuation)
  | ParentExhausted
  | CompilerBug(diagnostic)
```

`ChildDeferred`若requirements正是后续axis coordinates，child按dependency继续展开；它不是blocked error。ExactRejection只丢当前child的
causal assignment，parent continuation立即保留并可发下一个sibling。Unsupported记录本prefix在current compiler无completion但不形成
no-good；immutable capability下无需反复重试。Indeterminate保留state/continuation并由controller以后补work或最终报告coverage。只有
`ParentExhausted`证明该parent所有canonical choices已经访问。

`ScheduleState`是complete assignment边界：controller为它预留actualization work并调用Q50.F。Accepted actual result成为incumbent
contender；ExactRejection关闭当前complete point并保留Schedule parent continuation；Indeterminate按typed原因保留或终止当前coverage。
actual admission result不伪装成普通axis `ExpansionOutcome`。

#### 完整性、无重复与终止证明

设每个axis在固定parent `p`下的canonical successor是有限无重复集合`D_i(p)`，且Q50对应oracle已证明all-and-only：

1. Q50.S-normalized semantic roots是immutable base problem；Spatial canonical successor给出全部合法首层physical plans；
2. 归纳假设第`i`个variant prefixes完整。每个parent continuation最终访问`D_{i+1}(p)`的每个choice一次，`makeChild`是一一typed
   extension；因此所有合法`i+1` prefixes均被访问；
3. exact causal rejection删除的只是verifier/feasibility已证明无合法completion的child；proposal dedup只删同key duplicate，均不破坏归纳；
4. K后使用新的occurrence generation和Buffer/Schedule successors，所以不存在旧I/J state被错误延伸；
5. 每个domain有限、axis数有限，unbounded-work exact mode最终终止；bounded invocation只暂停并保留continuations，不把未访问集合说成空。

完整组合最坏仍是`Π_p |D_i(p)|`的指数规模；AND/OR component decomposition只能在boundary facts证明独立时减少重复，不改变这个语义
集合。Q51-2先允许axis内部使用其已证明component successor；跨axes memo/DP与scalability归Q52。单次resume的额外memory是parent plan+
one choice，session总memory与live frontier/continuations成正比，不预建Cartesian product。

#### Current successor迁移

旧`UnifiedPhysicalDataflowDomain::getNextAssignment`在一个函数里按buffer→movement→representation→implementation→temporal→coupled
混合进位，并在每次上游变化后重建整个suffix first points；它遗漏fixed semantic facts、J/K/F closure，也让proposal和canonical cursor混在complete
assignment循环中。终态删除该global mixed-radix owner：各Q50 `getFirst/getNext/contains`中经oracle证明的局部算法迁入新plan types，Q51
typed continuations逐axis调用；`getConstructiveAssignment/getFusionOrientedAssignment`拆回B/D/E proposals。

### Q51-2 Gate

- 使用真实Q50有界穷举domains逐axis比较canonical child keys与independent nested-loop reference，不写只靠mock provider通过的Core test；
- 2--3 choices/axis的mixed有界穷举problem在任意resume cut、work allowance和frontier交错下最终集合与flat product一致，无duplicate；
- proposal关闭、反转、并行延迟及重复proposal不改变集合；canonical cursor在proposal后仍覆盖该point的后继；
- 在每个axis注入ExactRejection、Deferred、Unsupported、Indeterminate和empty-domain，验证parent sibling可达、no-good强度和coverage分类；
- K Serialized/Pipelined siblings分别重建I/J，故意混用generation稳定失败；full F Indeterminate暂停后可继续同一complete state；
- source byte-identical，work暂停/恢复无IR、clone、replay、default stats；旧global successor/proposals完成donor mapping后source/call-tree零残留。

### Q51-3 专项调研：typed work、admissible bound与comparison cohort

LLVM `InstructionCost`区分valid/invalid，并特别提醒invalid不是Unknown或“极大cost”；OpenXLA latency-hiding scheduler把profiling table和
analytic model组合，但每个估计都依赖明确的platform/model inputs。对应本仓，exact work、可证明lower bound和performance estimate必须
是三种不同知识，不能继续把unknown字段写`value=0`，也不能把一个默认常数表和saturating arithmetic包装成hard cost。

current search从accepted actual IR读取六个metrics，只有全Known才按`DDR read, DDR write, NoC bytes, instruction, SPM high-water,
DDR high-water`做lexicographic replacement。这个顺序没有共同时间/目标含义，且SPM high-water是capacity事实；它既不是Q51最终目标，也
不能证明搜索质量。current theoretical duration又使用隐式default rates、canonical shortest-path route和saturating add。可迁移的是final
IR work collectors、per-field knowledge和cohort-wide term enablement，不是旧winner rule。

#### 三层typed cost result

```text
CostKnowledge<T> = Known(T) | Unknown(CostUnknownReason)

PlanWorkMetrics
  computeWork: per target operation/resource exact counts
  ddr: bytes / transactions / issue counts
  localMovement: bytes / descriptors / issue counts
  communication: endpoint bytes/messages, waits, relay/local-combine work
  topology: minimum-hop link-bytes, cut crossing, exact link work when qualified
  control: instructions, events, joins, participant waits
  structure: critical dependency depth, recurrence/fill/steady/epilogue facts

PlanLowerBound
  objective: Known(EstimatedDurationLowerBound) | Unknown(reason)
  components: typed bound components with derivation/dependencies

PlanEstimate
  objective: Known(EstimatedDuration) | Unknown(reason)
  components: typed resource estimates
  assumptions: CostAssumption[]

PlanCostAnalysis
  work: PlanWorkMetrics
  lowerBound: PlanLowerBound
  estimate: PlanEstimate
  observed: exact axis plans + immutable TargetCostFacts actually read
```

`CostKnowledge`是closed sum，Unknown没有可读value。unsupported collector、缺target fact、dynamic multiplicity、opaque route和arithmetic
overflow各有typed reason；严重程度不靠enum max合并，组合函数按所需term返回all-and-only causes。exact zero只来自静态不可达/empty work，
不能由Unknown降级得到。

`PlanWorkMetrics`是审计事实，不等于duration；每个字段注明unit、aggregation和knowledge。aggregate与per-Tile maximum分别保留，不能把不同
Tiles的最大值拼成虚构critical Tile。SPM/DDR offset high-water只从Q50.F/Q50.0 actual result用于capacity/headroom diagnostic，不进入
partial bound或winner objective。
current opaque Direct DTE保留endpoint work、message、minimum-hop/cut bounds；canonical shortest path只可产生明确标记的estimate，绝不
产生exact directed-link load。

#### Comparison cohort与objective

一次session建立一个immutable `CostComparisonCohort`：

```text
CostComparisonCohort
  targetFacts: exact rates/latencies/table identities available this session
  enabledTerms: only terms whose parameter + collector contract apply uniformly
  objective: MinimizeEstimatedCardCompletion
  unit: target cost tick
```

cohort只从current target/profile facts构造，不从第一个candidate决定term，也不在candidate间增删。一个term若不能对所有合法plans以同一
语义求值，就从本cohort estimate中关闭；它的exact work仍保留。关闭term不是按零计入，而是整个objective可能成为Unknown，除非其它
enabled terms已形成完整目标。不同cohort的estimate绝不比较；cohort equality比较typed facts contents，不依赖pointer、profile名或版本号。

target rate与work转换使用checked integer/rational helper并明确rounding direction：estimate按同一规则round，lower bound只能向不高估方向
round。任何add/multiply/divide overflow或zero rate返回Unknown；禁止saturating到`UINT64_MAX`后用于priority、dominance或pruning。

#### Bound构造

每个Q50 mechanism可贡献两类东西：exact work和对**当前cohort objective**有证明的bound。没有proof metadata的估计永远不是bound。组合
遵循J event/resource DAG：hard sequential chain相加，可真正并行且resource-disjoint的branches取maximum，同一compute/transport
resource的mandatory issued work除以exact service parallelism形成performance lower bound，K recurrence用RecMII，H使用endpoint/min-hop/cut
bounds；这里不包含SPM bytes、packing或capacity。不同bound若可能重复计同一work，
只能取maximum或保留分量，不能相加。

```text
derivePlanBound(state, cohort):
  collect current mandatory work and every unassigned axis minimum
  map each component through a proven monotone target term
  combine by hard dependency/resource proof
  if any objective-critical component has no safe derivation:
    return Unknown(reasons)
  return Known(lower bound)
```

partial-state lower bound只有在“任意completion的objective都不小于它”时才admissible。proposal point的预测、average bandwidth、modeled route、
future overlap猜测、SPM estimate和learned priority均不能剪枝。Q51有界穷举oracle对每个partial prefix枚举全部completions并验证
`bound <= minimum completion objective`；找不到Known objective时只检查component bounds。

#### Complete-plan comparison

```text
compare(a, b, cohort) = Better | Worse | Equivalent | Incomparable

if either estimate objective is Unknown or cohorts differ:
  Incomparable
else if a.objective < b.objective:
  Better
else if a.objective > b.objective:
  Worse
else:
  Equivalent
```

只有Q50.F actual Accepted结果能进入accepted-result comparison。第一个结果建立functional incumbent；后续只有`Better`才替换。
`Equivalent`使用完整semantic plan key作确定性tie-break，但不改变objective；`Incomparable`必须保留必要summary并按明确规则降低coverage，
不能按semantic key宣称“更优”。若搜索结束仍有多个incomparable plans，driver可按semantic key选择一个用于single commit，但结果标记为
`feasible-unranked`，不声称best/optimal/gap。Q51-5定义controller与coverage的最终状态机。

exact work vector只有在所有future objective映射已证明monotone时才能支持safe dominance；普通“所有raw bytes少一点”不能越过compute/
communication trade-off。lower bound仅在与一个Known incumbent objective同cohort且`bound >= incumbent`时可prune；estimate不能prune，
也不能形成no-good。

#### Estimate与actual cost边界

fixed semantic description与各B--K plan descriptors提供partial ordering/bound所需的work estimate；complete candidate的current Instr
collector计算actual work/cost。controller只用同一cohort的actual accepted cost更新incumbent；estimate与actual差异用于profile和改进
ordering，不是compiler bug，也不能覆盖actual值、签发SPM legality或把package measurement回灌当前compile。actual action/resource IDs仍须
all-and-only对应complete assignment；该结构对应失败才是emitter/lowering bug。

Q9/profile仍只观察最终package；未来Q52可用fresh profile更新下一次compile的immutable target estimate facts，但不能在一次session中途
改变cohort，否则已有priority/bounds失效。

#### Current迁移与文件边界

| Current | 终态owner | 改造/删除 |
| --- | --- | --- |
| `ScheduleCostMetric{value, knowledge, reason}` | closed `CostKnowledge<T>` + typed work field | Unknown无value；severity merge和zero sentinel删除 |
| accepted-IR six-field lexicographic winner | `CostComparisonCohort` + estimated completion objective | SPM/DDR high-water退出objective；actual candidate loop删除 |
| implicit `getScheduleEstimatePolicy()` constants | immutable target estimate facts | 无默认policy singleton；缺参数成为Unknown或cohort-disabled |
| `StaticDurationTermMask` candidate cohort logic | typed cohort builder | term只uniform enable，不用bitmask绕过missing field |
| saturating duration arithmetic | checked cost arithmetic | overflow为Unknown，不参与prune/replace |
| modeled canonical NoC route | H/J typed route knowledge | opaque时只是estimate，exact route才有link work |
| old `StaticSchedulePlan` shadow schedule | J `ClosedSchedulePlan` | duration直接消费J plan；不从actual IR另造shadow structure |
| final Instr work collectors/package profile fields | actual parity与report owner | 保留raw facts；不拥有search winner |

实现文件分为`PlanWorkAnalysis.*`、`PlanCostModel.*`、`PlanCostComparison.*`；actual Instr collectors继续留在
`Analysis/ScheduleCost/`，共享typed field definitions和checked arithmetic，不让Planning include package/profile writer。

### Q51-3 Gate

- 每个work field覆盖zero/Known/Unknown causes/overflow，unknown consumer无法读取value；input/Tile order和parallel aggregation确定；
- 有界穷举complete plans逐项与actual Instr work一致，故意hidden alloc/message/wait/instruction使parity失败；opaque/exact route fixtures分层正确；
- 每个Known partial bound与flat completion minimum比较不高估；关闭任一bound只减少pruning，不改domain/winner；
- estimate table/analytic term缺失、cohort不同、overflow和dynamic multiplicity产生Incomparable，不按0/极大值或semantic key替换；
- equal objective只按semantic key稳定tie；compute-vs-DDR、message-vs-byte、tree-vs-ring、fusion-vs-buffer trade-off不被raw lexicographic
  dominance误删；
- actual accepted cost由controller按同一cohort比较；estimate只用于priority/bound且不能覆盖actual值。普通compile无cost dump、statistics或
  profile读取，source和partial states不变；
- source/call-tree gate删除six-field winner、implicit policy、saturating search cost和shadow schedule，保留的actual collectors有明确consumer。

### Q51-4 专项调研：causal forbidden assignments与soundness

CSP no-good recording缓存“某个变量子集的赋值组合不可能出现在任何解中”；CDCL learned clause则必须从implication reasons推导，不能在看到
一次失败后自行删掉看似次要的decisions。Wafer当前没有SAT式完整implication graph，因此Q51初版不做resolution、backjump clause
generalization或automatic core minimization；只记录Q50/F verifier直接给出的extension-closed causal conjunction。API使用更直白的
`ForbiddenAssignment`，文档说明它对应标准no-good。

#### Typed representation

```text
CoordinateBinding =
  SpatialBinding(SpatialCoordinateId, SpatialChoice)
  | RegionBinding(RegionCoordinateId, RegionChoice)
  | TemporalBinding(TraversalScopeId, TemporalChoice)
  | RepresentationBinding(PhysicalVersionCoordinateId, RepresentationChoice)
  | MovementBinding(MovementBoundaryId, MovementChoice)
  | BufferBinding(BufferCoordinateId, BufferChoice, occurrenceGeneration)
  | ExecutionStructureBinding(RecurrenceScopeId, StructureChoice)
  | ScheduleBinding(ScheduleCoordinateId, ScheduleChoice, occurrenceGeneration)

ForbiddenAssignment
  bindings: NonEmptyVector<CoordinateBinding>
  proof: ExactRejectionProof
  scope: ExtensionClosed | ExactCompletePlan
```

每个binding使用该axis自己的strong type和semantic scope ID，不是axis字符串、variant index、pointer、printed plan或generic bytes。bindings按
`coordinate kind + semantic ID + typed value`规范排序；同一coordinate重复/冲突、空bindings或跨generation I/J组合是compiler bug。
cache由一个`PhysicalDataflowPlanningProblem`独占，source/target facts在整个session immutable，因此不跨session保存problem fingerprint。

`ExtensionClosed`只用于structural verifier已经证明“任意保留这些bindings的future assignments均失败”的情况；它可命中partial state。
actual SPM rejection默认只证明当前complete candidate失败，因此必须生成包含全部bindings的`ExactCompletePlan`，不能按owner root、bytes或
last changed axis自行删字段。若其它actual verifier也只能证明
一个完整point失败，就生成含该complete plan全部bindings的`ExactCompletePlan`，不能自行删字段。proof本身是closed typed sum，例如
exact demand hole、empty exact domain、actual SPM capacity rejection、layout constraint contradiction、message cover、resource/deadlock cycle
或target field range；diagnostic text只用于展示。

#### 唯一合法的产生路径

```text
record(result):
  switch result:
    ExactRejection with producer-supplied CompleteCandidateKey and actual witness:
      validate every binding occurs in result.observed dependencies
      validate proof says ExtensionClosed or ExactCompletePlan
      normalize and insert ForbiddenAssignment
    Accepted:
      compare and retain actual result
    Deferred / Unsupported / Indeterminate:
      insert nothing
    compiler bug / incomplete owner relation:
      stop compilation; insert nothing
```

Q50 actual witness producer必须同时给出all-and-only complete candidate key、current owner relation和typed rejection；cache不从“largest demand owner”、node邻接、group
members、last changed axis、error location或diagnostic字符串猜cause。cost bound/estimate、local Pareto、proposal failure、solver timeout和
resource exhaustion永远不能产生ForbiddenAssignment。Q50.0对每个complete candidate运行；actual capacity rejection至少关闭该
`ExactCompletePlan`，只有verifier直接提供extension-closed typed proof时才能关闭更宽prefix。owner attribution可以帮助baseline选择下一
temporal axis或search安排相关sibling，但不能自动扩大rejection范围。

#### Matching、subsumption与continuation

```text
matches(state, forbidden):
  for binding in forbidden.bindings:
    if state has not assigned binding.coordinate: return false
    if state choice at coordinate != binding.choice: return false
  return true
```

PlanningState prefix只追加依赖正确的choices，所以一旦包含全部extension-closed bindings，任意completion仍包含它们，整state可安全prune。
未赋值不能按default/missing匹配。K重建I/J时旧generation fields已从new variant移除，因此旧binding不会误命中。

parent continuation先生成child，再查cache；命中只丢child，continuation保留并发下一个sibling。若新forbidden bindings是已有项的superset，
新项冗余；若新项是subset，可删除被它subsumed的supersets。subset关系必须逐typed binding equality证明，不按witness kind、string或hash。
删除/evict任意cache entry只损失性能，不改变domain；Q51 correctness mode可保留全部direct entries，Q52再依据profile设计bounded index。

index先按binding数量和stable first binding分桶，再做完整subset/match；hash collision必须比较typed contents。topology automorphism只在同时
变换所有bindings且proof invariant已验证时生成对称项；Q51初版不需要该优化，宁可保留raw semantic assignment。

#### Soundness证明

对任一`ForbiddenAssignment N`，producer proof给出`N -> contradiction`。若state `s` matches N，则`s`的任意completion `c`保留N的全部
bindings，因此`c -> contradiction`；剪掉`s`不删除合法complete plan。subsumption中若`N1.bindings ⊆ N2.bindings`，任何匹配N2的state也
匹配N1，所以删除N2安全。其它result不满足该蕴含，禁止记录。

cache本身不是完备性所需：关闭cache后Q51-2 canonical continuations仍访问全部points。exact有界穷举mode分别以cache off/on平铺，complete
plan set、best comparable objective和selected semantic key必须相同；只有visited work可减少。

#### Current迁移

current `CardExecutableSearch`在每个actual candidate失败后只增`exactRejected`并继续，且旧
`isProvenExactTileMemoryPlanningFailure`曾从不完整relation推测受影响node。终态保留complete-candidate actual loop，但删除推测归因：
Q50.0返回actual demand及完整current owner relation，Q51先记录精确complete-point rejection，再按typed domain继续siblings。任何更宽
no-good必须由对应verifier直接证明，不能由容量大小或owner集合猜出。

实现文件为`ForbiddenAssignment.*`和`ExactRejectionProof.*`；它们只依赖PlanningState/Q50 typed witness schema，不include CodeGen、diagnostic
parser、MLIR operation或cost model。

### Q51-4 Gate

- 每类Q50/F exact witness有direct binding/proof test；删除任一真正causal binding产生至少一个可行sibling反例，证明不能再generalize；
- missing coordinate、different scope、different generation、different target session均不匹配；完整相等、subset/subsumption和hash collision正确；
- Deferred、Unsupported、Indeterminate、proposal exhaustion、cost worse/unknown、actual parity/commit failure逐项证明cache size不变；
- 在每个axis sibling注入failure，命中后parent continuation仍发出全部其它children；cache off/on有界穷举accepted set/objective/winner相同；
- input/hash/parallel insertion顺序不改变normalized entries或selected result；cache无pointer/string/ordinal、cross-session persistence、default log/
  stats或automatic core minimizer；
- source/call-tree gate保留actual rejection→controller的唯一typed路径并删除broad node attribution；SPM rejection默认只关闭完整candidate，
  更宽prefix必须有verifier直接给出的extension-closed proof。

### Q51-5 专项调研：deterministic best-first、branch-and-bound与coverage

A*的最优性依赖admissible heuristic与明确path-cost合同；AND/OR anytime研究还指出，深度优先分解和尽快改善incumbent之间存在真实取舍。
Wafer的axis prefix不是普通edge-cost path，因此不冒充标准A*。controller采用deterministic best-first改善time-to-first，用Q51-3独立证明的
complete-plan objective/lower bound做branch-and-bound和coverage；priority estimate本身没有剪枝权限。

#### Planning与candidate actualization使用同一work allowance

```text
PlanningWorkKind =
  AxisSuccessorStep
  | RelationPropagationStep
  | BoundDerivationStep
  | ScheduleSearchNode
  | CompleteCandidateActualization

PlanningWorkAllowance
  remainingCredits: uint64
  reserve(kind, count) -> Granted | Exhausted
```

一credit是对应kernel文档化的一个deterministic primitive step，用于可重放中断，不声称各kind wall-time相等。query在开始一个不可再分的step前
reserve；失败则不修改cursor/cache并返回Indeterminate/Paused。local solver接收从同一allowance预留的node budget并回报实际消费。并行
task在提交前按semantic key顺序reserve，未获额度的不启动，因此线程完成顺序不改变visited set。

这个allowance同时限制partial successor/query work和complete-candidate actualization；开始candidate transaction前必须按semantic key
预留一个`CompleteCandidateActualization`，Q50.0内部MiniMalloc仍使用其owner-defined typed work budget并原样传播
`ResourceExhausted`。旧`maximumEvaluations`字段和`--search-max-candidate-evaluations`类CLI退出，不建立第二份candidate cap。
Q51定义机制与test allowance；Q52依据fresh profile决定production默认work/time policy。取消/outer deadline只在step或candidate边界
停止，结果coverage按保留的continuations、已完成actual results和bounds计算，不能把wall timeout说成NoSolution。

#### Frontier item与priority

```text
FrontierItem = PlanningState | AxisContinuation | PausedCompleteState

FrontierPriority
  lowerBoundClass/value       // Known ascending, then Unknown
  estimateClass/value         // Known ascending, then Unknown
  prefixDepth                 // deeper first only after the two cost fields
  semanticItemKey             // total deterministic tie-break
```

priority是session-derived query，不进state/key。Known lower bound优先有助于尽早闭合global bound，estimate在bound相同/unknown时提高
time-to-first，depth只促成complete plan；无一项改变domain。Q52可用profile替换priority policy，但Q51 exact mode无论priority如何都保留
全部items。

每个parent continuation与其child分别入queue；resume只产生一个child后把updated continuation重新入queue。相同PlanningState semantic key
只保留一个frontier owner，但它的所有incoming continuations已各自继续推进，不靠state parent/history恢复siblings。

#### Controller

```text
searchPhysicalDataflow(problem, allowance):
  frontier = {SpatialContinuation(problem)}
  forbidden = {}
  incumbent = none

  while frontier not empty:
    if allowance cannot reserve next controller step:
      return finishBudgeted(frontier, incumbent)

    item = pop minimum FrontierPriority

    if item is a state and forbidden.matches(item):
      continue

    analysis = derive work/bound/estimate for item
    if hasKnownIncumbentObjective(incumbent) and
       analysis.lowerBound is Known and
       analysis.lowerBound > incumbentObjective:
      continue                         // strict > preserves equal-cost tie keys

    if item is AxisContinuation:
      outcome = resume(item, allowance)
      record only producer-supplied ExactRejection
      requeue preserved/updated continuation as required
      enqueue ready/deferred child according to Q51-2
      continue

    if item is ScheduleState:
      reserve CompleteCandidateActualization
      result = evaluateCompleteCandidate(source, item.completeAssignment)
      if Accepted:
        compare actual cost and retain/update move-only incumbent
      else if ExactRejection:
        record exact-complete rejection or verifier-proved extension-closed witness
      else classify Unsupported/Indeterminate/CompilerBug exactly
      continue

    enqueue continuation for the state's next static axis

  return finishExhausted(incumbent, unsupported, indeterminate)
```

strict `bound > incumbent`只删除不可能改善actual accepted objective的state；`bound == incumbent`仍展开，以发现同objective但更小semantic plan key。
若未来为tie key证明minimum completion key，可扩为lexicographic bound；当前不猜。Incomparable full plans都保留；Known objective的
`Better/Equivalent`按Q51-3更新同一comparable incumbent。

#### Coverage与typed result

```text
PlanningCoverage =
  ObjectiveOptimal
  | FeasibleWithBound {lowerBound, incumbent, gap}
  | FeasibleUnranked
  | BudgetedFeasible

SearchResult =
  SelectedActual {move-only CardExecutable, actual cost, semantic key, coverage}
  | ExactNoPlan {proof summary}
  | UnsupportedPlanSpace {features}
  | IndeterminatePlanSpace {causes}
  | WorkExhaustedWithoutPlan
  | CompilerBug
```

- `ObjectiveOptimal`：frontier已exhaust/pruned，所有potentially better states有Known admissible bound，actual accepted objectives同cohort可比；若存在
  equal-objective states，也已闭合semantic tie；它只证明current planning objective optimum，不外推真实板端最优；
- `FeasibleWithBound`：预算停止但所有未展开continuations仍保留，global frontier lower bound Known且incumbent actual objective可比；gap按同一
  unit/cohort给出；
- `FeasibleUnranked`：已有actual Accepted candidate，但存在incomparable accepted result或objective Unknown；semantic-min accepted
  owner可用于functional发布，但不声称best/gap；
- `BudgetedFeasible`：已有actual Accepted candidate，但至少一个remaining state缺admissible bound，或Q52 future heuristic永久丢过state；
- `ExactNoPlan`：frontier完全exhaust，所有branches都由exact proofs关闭，且没有Unsupported/Indeterminate；
- Unsupported/Indeterminate会阻止ExactNoPlan；预算内没有actual Accepted candidate必须返回明确failure，不能运行/返回baseline。

若frontier所有Known bounds严格大于incumbent且不存在Unknown/incomparable/tie-pending item，可提前得到ObjectiveOptimal；否则只在exhaust后
判定。coverage、work counts和frontier reason只在显式report sink请求时详细输出；核心`SearchResult`只保留用户/下游需要的typed状态。

#### Determinism、终止与复杂度

finite domains + Q51-2 continuations保证unbounded exact run终止。priority queue push/pop为`O(log F)`，每item cost query另计对应Q50 work；
最坏仍访问指数个states；controller metadata为`O(F + continuations + accepted summaries + forbidden assignments)`，live candidate IR受
明确owner上限约束且只有retained incumbent跨evaluation存活。branch-and-bound/no-good关闭
只减少work。相同source/target/allowance在不同hash seed/线程数下reserve序列、visited semantic keys、coverage和selected plan相同。

real workload不承诺Q51阶段全局穷尽；机制允许bounded result。Q52才根据profile引入memo/DP/LNS和可能有损policy，并必须相应降低coverage。

#### Current迁移

`SearchWorkBudget{maximumEvaluations}`只按candidate数截断，不能表达partial work与actual solver状态。终态由统一
`PlanningWorkAllowance`、可选`PlanningWorkReport`和上述`PlanningCoverage`原位替换，同时保留typed actualized/accepted/rejected
计数；不留compat wrapper或旧CLI。`winnerUpdates/proposalDetail/lastDetail`不进semantic result，显式report由typed events收集，普通
compile不构造字符串。

实现文件分为`PlanningController.*`、`PlanningWork.*`和`PlanningCoverage.*`，priority policy不塞进state/session owner文件。

### Q51-5 Gate

- real Q50有界穷举domain的controller与flat oracle比较complete set、best objective、tie key和ExactNoPlan；不是mock-only graph；
- 在每个work cutoff运行并验证：有plan时coverage精确、无plan时WorkExhausted、continuations无丢失；resume到exhaust与一次unbounded结果相同；
- adversarial estimate顺序、Unknown bound、equal-cost smaller key、incomparable plans证明estimate不剪枝、strict bound/tie正确；
- branch-and-bound/no-good分别关闭后objective/winner不变；每个pruned prefix由flat completion minimum验证确实不能改善；
- parallel query延迟/hash seed/priority provider顺序变化不改reserve/visited/winner；overflow和cancellation保持typed coverage；
- work counters分别证明partial-state CardModule/Q50.0为零、每个complete candidate各一次；旧maximum-evaluation API/CLI/coverage/
  summary strings source与help零残留；
- 普通compile不创建report/stats/log，显式report完整但不影响priority、allowance或result。

### Q51-6 专项调研：complete-candidate事务、lowering与verification

MLIR pattern transaction只覆盖单个op的in-place modification；DialectConversion rollback还会延迟部分erase/replace且有显著bookkeeping成本，
并不等价于整个compiler stage transaction。Transform dialect也要求：precondition failure应发生在mutation前，mutation后违反postcondition是
irrecoverable。Wafer complete candidate跨CardModule、16个TileModule、buffer/movement/event和多个lowering stage，正确边界是“先pure
prepare，后只在一棵candidate Card subtree内构造，失败或落选整棵删除”，而不是让每个builder各自clone/rollback。

#### Prepare不写IR

complete assignment进入actualization前生成：

```text
PreparedPhysicalDataflow
  source/program/target identity checks
  card/tile domain and RootRegionWork
  prepared execution instances and exact temporal occurrences
  prepared physical-version producer/use graph
  prepared communication actions/messages
  prepared storage objects/slots/rotation
  prepared execution structure
  prepared event order/worker/resource/completion/release
  expected structural identity/coverage/work relations
```

它由C/G--K各自的prepare kernel贡献typed pieces，随后做一次card-scoped totality/dependency check。prepare只borrow source和complete assignment，
不创建Module/Func/TileRegion，不调用pass，不分配actual offset，不建立IR trace。缺plan ID、跨generation关系、unsupported lowering capability、
message/field range或结构关系不完整在这里返回typed failure；它不计算SPM footprint、packing problem或capacity结论。

#### 一个Card subtree transaction

```text
evaluateCompletePhysicalDataflowCandidate(parentModule, source, assignment):
  prepared = preparePhysicalDataflow(source, assignment)
  if failed: return typed failure with parent/source unchanged

  CardSubtreeTransaction transaction(parentModule)
  card = transaction.createCardModule(prepared.cardIdentity)
  create all-and-only available TileModules exactly once

  emitter = PhysicalDataflowEmitter(card, prepared)
  emit root regions/execution/temporal control
  emit physical versions and storage objects
  emit selected communication actions
  emit structure-specific loops/phases
  emit events in PreparedSchedule order, including waits/joins/releases

  verify materialization relations once at Card scope
  run named TileRegion-to-Instr subpipeline on candidate Tile roots
  executable = run Q50.0 CardModule-to-CardExecutable once
  if Accepted: transaction.retain(); return move-only actual candidate
  if ExactRejection: capture typed current relations; destroy subtree; return feedback
  otherwise: destroy subtree; return typed failure
```

`CardSubtreeTransaction`只拥有本次新建Card op及其descendants；不snapshot/clone parent或source，不接管其它现有ops。candidate期间
source TensorProgram不被erase/rewire；logical source cleanup若需要，由拥有parent的外层conversion在最终winner发布后执行。失败或落选
删除的只是candidate subtree，不影响source/ProgramData。

#### Construction-time协作而非post-hoc passes

一个scope-local `PhysicalDataflowEmitter`组合窄builders：

- C root/execution builder只创建selected root/support/nested work；
- G `PhysicalVersionBuilder`按`PhysicalVersionId`创建/绑定一次；
- I storage/slot builder在first use前创建independent allocs和rotation expression；
- H movement builder创建DDR/peer/relay/bundle action及tokens；
- K builder创建Serialized、SCF pipeline或qualified periodic structure；
- J `ScheduleEmitter`拥有insertion/order/worker/wait/join/release points并调用上述typed callbacks。

builders通过prepared IDs交换values/events，不写共享string map、lookup-any cache或save/restore operand map。创建后的actual op/value mapping只在
transaction lifetime内存在；accepted IR用SSA、regions、types、effects、attrs和op order自包含，不保存plan、score或schedule side table。

不为每个builder再建一个全Card verifier。一个`PhysicalDataflowMaterializationVerifier`在所有Tile emit完成后线性检查plan-ID totality、
coverage、version producer/use、storage/action/event relation及无extra actual op；dialect op verifier负责局部形状/type/effect，Q50.0在Instr层
负责memory/transport/resource/ABI。三层范围不同，避免每leaf重复walk同一module。

#### 允许与禁止的复制

- 禁止clone/replay source TensorProgram、同一complete candidate重复物化、probe Module/Func和winner rematerialization；
- candidate SCF loop pipelining可在**本次新subtree内**由pinned `scf::pipelineForLoop`机械生成prologue/steady/epilogue，使用`IRMapping`；
- Q50.0可以按all-and-only Tile domain将candidate CardModule拆分/移动为每Tile唯一output，并使用既有bounded host parallelism；
  每Tile/variant只lower/translate一次，不把16 Tiles变成16次logical analysis/search；
- independent有界穷举oracle与production都逐complete assignment调用同一actual gate；任何一方都不保存loser作为下一candidate模板。

#### Failure classification

prepare前置条件失败若源/target确实无current capability，返回typed Unsupported；malformed assignment、generation mismatch、emitter
extra/missing relation或无owner SPM demand是CompilerBug。带完整actual witness的ExactRejection返回controller；pass/external tool/environment
错误按其owner返回typed failure。allocator/lowering不得自己修改candidate、换next point或调用baseline fallback。

ProgramDataHandoff仍由outer transaction持有；candidate evaluation只borrow stable identity/range，最终winner发布后才move入package
transaction。失败、unsupported、accepted loser都不能移动handoff或写package/output directory。

#### Named pipeline parity

search driver负责partial expansion、调用上面的candidate facade及比较actual results。TileRegion→Instr、memory planning、Direct-DTE binding、resource verification分别用
current named subpipeline/leaf pass builder；`wafer-opt`测试对一个已由test fixture构造的selected CardModule调用同一builders。不会注册
“search pass”或另一条direct mutation pipeline。

#### Current迁移

| Current | 终态 |
| --- | --- |
| `UnifiedPhysicalDataflowDomain::materialize/buildBufferingScopes` | C/G--K prepare/emitter；domain不include apply |
| `materializeTensorProgramAlternative` root clone | Q50.S common normalization + per-complete-candidate selected attention Linalg decomposition |
| per-assignment `compileCardModuleToExecutable` | Q50.F complete-candidate actual admission；每个assignment一次 |
| `CardExecutableSearchResult{executable, IR trace}` | move-only accepted actual result + coverage；trace仅显式final-winner instrumentation |
| DataMovement/Buffering旧apply与InstructionSchedule隐式domain/apply | H/I/K typed construction + J explicit EventId owned materialization |
| candidate failure loop and accepted executable incumbent | typed actual-result loop；complete-point rejection精确反馈，accepted incumbent不重建 |

实现文件为`PhysicalDataflowPreparation.*`、`PhysicalDataflowEmitter.*`、`CardSubtreeTransaction.*`和
`PhysicalDataflowMaterializationVerifier.*`；Q50.0仍在CodeGen/Executable owner。文件不以Q51/阶段编号命名。

### Q51-6 Gate

- work counters证明partial state CardModule/Q50.0为零、每个complete candidate Card subtree/Q50.0各一次；最终只剩一个retained winner，
  同candidate重复actualization和winner重建在ownership层不可表达；
- mixed DAG分别选择non-default spatial/region/temporal/layout/communication/slot/structure/schedule，actual IR按plan IDs all-and-only对应；
- failure injection覆盖prepare首末、每类builder、mid-Tile、materialization verifier、TileRegion→Instr、resource parity和Q50.0；source/parent除新
  subtree外byte/operation不变，失败后新subtree为零；
- current buffer relation totality与Q51-3 actual work/cost检查通过；hidden或无owner allocation、message、wait、worker或lowering
  temporary稳定报CompilerBug，不被猜归因；
- custom/generic roundtrip、verify-each、named/production leaf-pipeline parity、CardModule→Instr→CardExecutable通过；
- call-tree/source gate证明partial-state expansion不含materialize/Q50.0/clone，complete-candidate path只调用共同actual gate且不含baseline、
  repair或同candidate重建；旧post-hoc apply和第二candidate compile路径零残留；
- 每个16-Tile complete candidate的每Tile只emit/lower一次，最终winner target translate一次；independent downstream host jobs使用bounded parallelism且结果顺序稳定；普通
  compile无IR trace/stats/log对象。

### Q51-7 专项调研：public `none`/`search` routing与policy isolation

LLVM PassBuilder按optimization level构造不同顶层pipeline，同时复用同一pass/analysis infrastructure；MLIR named pipeline registration也把
稳定pipeline builder与driver option解析分开。Wafer同样只在driver policy switch处分叉，不能把`none`实现成“search取第一个”，也不能让
`search`先跑baseline拿一个executable。两者共享的是事实、final plan schema和lowering，不是controller状态或结果。

#### 唯一public分叉与共同汇合

```text
source/import/normalization
  -> analyzePhysicalDataflowProblem          // immutable policy-free facts
  -> switch OptimizationPolicy
       None:
         repeat canonical candidate -> actual admission
         only after actual SPM rejection: deterministic temporal successor
       Search:
         expand typed partial states
         each complete assignment -> actual admission
         compare accepted actual results
  -> one retained accepted CardExecutable
  -> one target/package publication transaction
```

`OptimizationPolicy`是closed `None | Search`；current `OptimizationConfig::search(evaluations)`及
`getMaximumSearchCandidateEvaluations()`删除。Q52的production allowance/priority是compiler-owned search policy facts，不重新塞进public
optimization identity；test可从internal API注入allowance，不增加长期user candidate-count selector。

common `analyzePhysicalDataflowProblem`只返回StructuredDAG、target/topology、exact relation与semantic IDs等immutable facts，不创建baseline
coordinate、proposal、frontier、score或winner。baseline/search各自从相同source transaction borrow facts，但controller objects和result
ownership独立。

#### 允许共享的API必须以complete assignment与actual admission为边界

| Shared | 输入/输出 | 禁止携带 |
| --- | --- | --- |
| policy-free analyses | current IR/target facts -> typed facts | policy、proposal order、winner |
| Q50 domain kernels | typed parent -> choices/witness | baseline fallback、actual IR |
| `PhysicalDataflowPlan` schema | complete typed assignments | origin policy、score、coverage、history |
| prepare/emitter/verifier | complete assignment -> candidate Card subtree | frontier、next-choice callback、capacity estimate |
| Q50.0/named lowering | complete candidate CardModule -> actual result | candidate choice、policy switch、repair |

baseline可以调用某个Q50 **single-coordinate query/validator**验证自己已经直接构造的canonical choice；它不能调用返回domain/options、
continuation、proposal或“选一个assignment”的search API。search可以使用同一canonical choice作为普通domain member/proposal，但不能调用
`buildCanonicalPhysicalDataflowPlan`或接收其plan/result。

#### Include/CMake/call graph

```text
Planning/PhysicalDataflow/Common  <- analyses + plan schema
Planning/Baseline                -> Common + narrow Q50 queries
Planning/PhysicalDataflow/Search -> Common + all Q50 domains + narrow actual-result/evaluator schema
Conversion/...                   -> Common plan schema, no policy controller
CodeGen/Executable               -> common candidate actual admission/Q50.0
Compiler driver                  -> policy facade + injected candidate evaluator + final publication
```

Baseline library不得include/link Search headers/library；Search不得include/link Baseline或CodeGen implementation。compiler driver向两者注入
窄`CompleteCandidateEvaluator`，其实现调用共同CodeGen actual admission；Search只看typed result schema。Conversion/Transforms不得反向依赖
任一controller。静态source/CMake gate与link-only tests同时检查，避免
通过forward declaration或utility header偷渡调用。

#### Failure绝不fallback

- None canonical analysis/legalization/plan failure直接返回baseline typed failure；不创建search session；
- Search的ExactNoPlan、Unsupported、Indeterminate、WorkExhaustedWithoutPlan直接返回search typed failure；不运行baseline；
- 带完整actual witness的SPM capacity rejection只回到当前policy controller；其它actualization failure按typed contract停止或记录
  unresolved，不调用另一个policy；
- target/package/environment failure发生在共同downstream，仍不改变policy或plan；
- diagnostics必须保留实际失败owner，禁止将search failure改写成“baseline accepted”或反之。

#### CLI与matched A/B

production CLI只保留`--optimization-policy=none|search`。删除`--search-max-candidate-evaluations`、candidate count、mechanism selector及其help/
parser/test。未给policy时使用current documented default，但default解析后仍得到同一closed enum，不走“try search then fallback”。
`--compile-timing`只安装optional instrumentation sink；它不能改变policy、allowance、priority或call graph。

matched comparison属于test/report harness：

```text
compile(sourceFresh1, policy=None,   outputA, ProgramDataA)
compile(sourceFresh2, policy=Search, outputB, ProgramDataB)
compare accepted outputs / actual IR / work / board results outside compiler
```

两次从同一source artifact独立parse/import，不能把baseline CardExecutable、CardModule、plan、offset、IR trace、ProgramDataHandoff或analysis cache
传给search；output transaction与temporary directory也独立。共享source identity只用于证明输入同源。

#### Current迁移

current `compileTensorProgramModuleToCardExecutable`在policy branches内部各自产生accepted executable，然后才共同包装；search branch还读
`OptimizationConfig` candidate budget。终态保留policy-specific controller，把complete candidate统一交给同一个injected actual evaluator，
并把唯一retained winner交给共同publication transaction。测试用failure injection不能再要求
“只有search可用”的无关耦合；每个injection按其真实outer transaction capability决定。

### Q51-7 Gate

- static include/CMake/transitive call graph证明Baseline↛Search、Search↛Baseline/CodeGen、Conversion↛controllers；policy facade恰一个switch；
- spies/work counts分别证明None search-session=0、Search baseline-controller=0；两者`candidateActualizations == Q50.0 invocations`，
  rejected/loser owners全部销毁，published result恰为1且winner无重建；
- baseline/search各类plan failure、work exhaustion和共同commit/target/package failure都不调用另一policy；diagnostic owner准确；
- public CLI正负例只接受`none|search`，旧candidate-evaluation option/help/API/test零残留；default policy没有fallback trace；
- explicit timing on/off得到相同plan key/IR/package，普通输出无compile stats；
- matched A/B使用两份fresh source/ProgramData/output transaction，故意尝试跨policy传plan/executable在type/API层不可表达；
- 两条policy选同一显式plan时，common emitter/named pipeline产生等价actual IR；不同plan差异来自typed assignments，不来自branch-specific lowering。

### Q51-8 专项调研：全轴oracle、production parity与总迁移门禁

MLIR/LLVM把内部API unit、IR regression、diagnostic和whole-pipeline integration分层；Alive2的translation-validation实践则说明，与其只
相信transform实现，不如对本次具体input/output关系做独立检查。Wafer不能直接套Alive2语义，但应采用同一原则：Q50逐轴reference证明
domain，Q51 independent composer证明全轴组合，independent runner验证每个complete assignment的actual result；production使用同一
complete-candidate actual gate并只发布winner。

#### Oracle A：完全独立的plan enumerator

`ReferencePhysicalDataflowPlans`位于test-only library，不include production PlanningState、continuation、controller、proposal、no-good、
bound或materializer。它组合每个Q50 checkpoint已经要求的plain reference enumerator：

```text
read fixed semantic roots and attention FA/FD facts
enumerate spatial partitions/embeddings/merge owners
for each enumerate region partitions/execution replicas/use bindings
for each enumerate temporal integer points/orders/tails
for each enumerate physical-version producer/use bindings
for each enumerate communication payload-state/action DAGs
for each enumerate pre-K buffer + structure + post-K buffer
for each enumerate worker/resource/control/completion schedules
derive reference work/objective with test-local checked arithmetic
emit canonical semantic plan tuple
```

有界穷举范围为2--4 Tiles、每axis 2--3真实choices及有限event/buffer/chunk domains；它们只限制oracle test size。reference types可最终转换为
`PhysicalDataflowPlan`做commit，但enumeration/filter/key计算不调用production successor/contains/canonicalizer。逐parent比较，而不只比较
最终count，以定位第一个漏域/重复stage。

#### Oracle B：independent complete-candidate actualization

第二层独立枚举complete assignments并调用与production相同的actual gate，用实际结果定义资源合法集合：

```text
for each reference complete plan:
  parse/import a fresh source transaction
  actualize exactly that plan directly
  run Q50.0 once
  record Accepted / ExactRejection / Unsupported / Indeterminate
  require complete current buffer relations for every SPM demand
  derive ActualPlanKey and actual cost from typed IR relations, not printed names/pointers
```

每个plan使用fresh parse而不是clone前一个actual result；ProgramData和output owner独立。reference enumerator只在test library中；
actual evaluator与production完全相同。oracle得到的actual accepted set、cost和semantic tie key用于核对production controller，不能用
shape/bytes公式预先标注feasible/rejected。

#### 全轴case矩阵

| Axis | reference/actual witness |
| --- | --- |
| fixed attention algorithm | FA prefill、FD decode、多个independent roots；algorithm不因physical proposal order改变 |
| spatial | non-max factors、non-rectangular embedding、merge owner、hole/asymmetric topology |
| region/execution | split/maximal group、nested/direct stored、explicit replica、multi-producer fanin |
| temporal | all-positive interval leaves、order precedence、ragged tail、per-Tile/nested differences |
| representation | primary/alias/conversion/shared secondary versions、different transport encoding |
| communication | DDR/direct/relay、partial/max fanout、gather、ring/tree/double-tree/recursive/row-column/Bruck-pairwise |
| buffering | Fresh/Alias/Reusable、1..U independent slots、completion-dependent reuse |
| execution structure | Serialized、SCF pipeline、qualified periodic、K→I-post-K→J generation reset |
| schedule | ready orders、worker/resource binding、wait/join/release、known/opaque NoC resources |
| feasibility/cost | actual MiniMalloc result、typed certificate、Unknown/incomparable、actual accepted cost |

另有至少三个联合反例：局部最小movement因schedule失去overlap而败；最大fusion因buffer/lifetime败给region cut；最短hop placement因
representation/relay/worker critical path败给另一placement。case只实例化通用typed关系，不把operator/shape名写进算法。

#### Production controller parity

同一tiny problem依次运行：

1. reference flat enumerator得到raw complete-assignment set；Oracle B逐项actualize，得到actual accepted/rejected set和reference winner/
   incomparable set；
2. production canonical continuations，关闭proposal、no-good和bound pruning，比较每variant/parent/final keys；
3. 逐项开启proposal、ForbiddenAssignment、branch-and-bound、parallel query，exhausted actual accepted set与winner不变；
4. 在每个work cutoff停止并resume，最终结果与unbounded run一致，intermediate coverage准确；
5. production public search对每个visited complete plan actualize一次，typed result与Oracle B一致；最终retained winner的ActualPlanKey、
   actual cost和semantic tie key与reference winner一致，publication一次且不重建。

priority变化可改变visited prefix但不能改变exhausted exact result；budgeted run只比较其coverage所允许的claim。Unknown objective时reference与
production都保留incomparable set并得到`FeasibleUnranked/BudgetedFeasible`，不能强行比较。

#### Complete-candidate一次性actualization与唯一发布证明

test observer是显式注入sink，记录semantic work而不改变控制流：

```text
partialStateCardModules = 0
completeCandidates = N
candidateCardModules = N
candidateQ50Invocations = N
candidateTileLowerings = N * available Tile count
retainedWinners = 1
winnerRematerializations = 0
packageTransactions = 1
```

observer关闭时普通对象不创建；source静态call tree证明partial expansion不调用CodeGen、complete state只经injected evaluator进入一条
Q50.0路径，防止计数漏埋点。IR dump只可查看accepted winner，不能出现candidate index/plan list。

#### Current/donor总迁移矩阵

| Current Search files/capability | New owner | 删除前必须存在的direct witness |
| --- | --- | --- |
| `Attention*`, `DecodeAttentionAnalysis`, `TensorProgramAlternative*` | Q50.S one attention op/classifier/work description/selected decomposition | Q/K/V detection、FA/FD、block/partition ownership、state merge、no root clone/algorithm axis |
| `ComputeImplementation*` | 10号deterministic lowering；真正semantic alternative留给Q48 | Natural/reciprocal exact applicability逐项判定，不借Q50.S恢复generic algorithm registry |
| `SpatialPlacement*` | Q50.B raw factor/subset/embedding/merge；Q52 automorphism与graph proposal | Q50.B raw/reference equality；Q52 topology equivalence、proposal quality与work witness |
| `SingleRootRegion*` | Q50.C | exact RootRegionWork、multi-producer support cut、one subtree construction |
| `CoupledRegion*` | Q50.D | connected partitions、execution/use/replica、nested relation |
| `TemporalTiling*` | Q50.E | complete vectors/orders/tails/nested scopes |
| 已删除的plan-side feasibility/footprint路径 | Q50.F actual admission | complete candidate→Q50.0、typed feedback、relation completeness |
| `PhysicalRepresentation*` | Q50.G | constraint solver proposals、version DAG、actual builder |
| query-only `SimpleRoute`、`DataMovement*`（`DataMovementApply*`已删除） | Q50.H；共享legacy query调用随I/J/K/Q51统一退役 | current payload partitions/arborescences、token-only emitter和迁移后的direct/relay/fanout/gather witness；legacy query不再取得actual owner |
| query-only `Buffering*`（`BufferingApply*`已删除） | Q50.I/Q50.K | current storage/slot/lifetime plans与K-specific overlap oracle；legacy query不再产生actual scope |
| 已删除的`InstructionSchedule*` | Q50.J/Q63 | EventGraph/resource domain、closed schedule、selected wait/join materialization及current-IR verifier |
| 已删除的`StagePipeline*` | Q50.K selected constructor | Serialized/SCF/finite-unrolled structures及K→I-post-K→J reclosure；旧scope-index actual scan不恢复 |
| `UnifiedPhysicalDataflow*` | Q51-1/2 state+continuations | full reference equality；mixed-radix/materialize facade均无剩余能力 |
| `CardExecutableSearch*`, `SearchWork*` | Q51-3--7 cost/controller/work/coverage/routing | per-complete actual gate、budget/coverage、unique publication、policy isolation |
| historical rank/candidate/frontier/NoC providers | corresponding Q50 owner或明确淘汰 | archive donor matrix逐项有current source+test；“未注册”不算迁移 |

迁移按owner分批提交：先加入new owner和tests，再改public caller，最后删除old file/header/CMake/test。不能一次删整目录后以新Core能编译代签。
最终`lib/Wafer/Planning/Search`和`unittests/Planning/Search`旧布局清空；新代码分别位于
`Planning/PhysicalDataflow/{Search,Feasibility,...}`及对应tests，baseline保持独立目录，不留compat include或forwarder。

### Q51-8 / Q51 Final Gate

- 每个Q50 axis direct oracle先通过，再运行Q51 parent-by-parent/full-plan oracle；没有mock-only Core或只看final count；
- independent actualization对全部tiny reference assignments运行同一Q50.0，形成actual accepted/rejected truth set；production result、work、
  owner relation与ActualPlanKey逐项一致；
- proposal/no-good/bound/parallel逐项开关不改exhausted actual accepted set、best objective/incomparable set和winner；任意cutoff/resume正确；
- public search的每个complete candidate actualize/Q50.0一次，partial state零IR；rejected/loser销毁，retained winner不重建并完成
  CardExecutable→package/no-card轻量链；无baseline fallback；
- public none/search call graph、CLI和fresh matched transaction满足Q51-7；普通输出无stats/trace，显式instrumentation不改结果；
- custom/generic roundtrip、verifier正负例、analysis invalidation、failure atomicity、named/production parity和`verify-each`覆盖实际leaf stages；
- donor矩阵每行具备new owner、production caller和direct test后，旧source/header/CMake/tests/CLI/help/symbol全部删除；source organization与public
  link检查通过；
- Q51 implementation独立提交并在上述fresh gates通过后才标done。重型LLaMA search不属于Q51 correctness gate；Q52在完整new path上首次做
  bounded scalability/profile，不能拿旧search运行时间代签。

Q51完成只证明current typed planning domain、objective/coverage和single-winner compilation正确；真实workload的time-to-first、RSS和
best-found quality由Q52，端到端/board readiness由Q53。

## Q52：Profile-Driven Anytime Search and LNS

### Pipeline contract

Pipeline position:
- Upstream IR / input:
  Q51已闭合的typed continuations、complete-candidate actual admission、actual-result comparison与coverage；显式profiling invocation可附加
  `PlanningProfileSink`，source/target facts仍immutable。
- Current stage responsibility:
  先测量state growth/repeated work/time-to-first/RSS，再在有完整future-boundary proof的位置加入pure memo、component DP、stronger
  admissible bound和profile-driven priority；必要时再加入明确有损的anytime/LNS policy。不得改变Q50合法域或Q51 exact oracle。
- Output IR / files:
  search仍只输出一个retained accepted actual result或typed failure；ordinary compile不写profile。显式profile生成独立报告，winner只发布
  一次且不重建。
- Downstream consumer:
  Q59 target/package publication、Q53 production/no-card/board qualification。
- User-level driver / named pipeline:
  public `search` policy内部的compiler-owned planning policy；不增加mechanism、beam、candidate-count或LNS CLI selector。
- Explicit non-goals:
  不缓存IR/offset/solver stack，不跨invocation复用未验证state，不从workload/op/shape名选策略，不用memo hit或learned estimate签发legality，
  不让profile instrumentation进入默认路径。
- Done criteria:
  safe optimizations on/off与Q51有界穷举complete assignment、actual result、objective和winner一致；representative heavy workload在
  bounded policy下获得actual Accepted winner并只发布一次；profile说明每项优化对应的实测hotspot和收益，coverage准确反映是否丢state。

### Q52-1 专项调研：profile、future-boundary memo与component DP

Halide learned autoscheduler和Ansor都用hierarchical/sketch search加cost model解决巨大schedule space，但beam/evolution只覆盖其定义的子集；
它们可迁移的是“先profile热点、分层生成proposal”，不能成为Wafer exact-domain证明。AND/OR search说明separator/treewidth可指数减少重复，
MLIR AnalysisManager则给出IR analysis按preservation失效的成熟边界。Q52因此先做无损memo/DP，任何有损search留给Q52-2。

#### 只在完整Q51链上profile

Q51 new path通过全轴oracle和轻量source-to-package后，才运行显式bounded profile。记录：

- 每axis/variant generated、deduplicated、exact-rejected、deferred、unsupported、indeterminate、continuation resume及peak frontier；
- relation/demand/domain/J/cost query的call count、work credits、wall/CPU和cacheable input重复率；
- complete-candidate actualization、每Tile lowering、actual SPM/DDR problems、MiniMalloc状态、current relation检查及Q50.0的次数/wall/RSS；
- proposal family到first actual Accepted candidate的贡献、actual rejection分类和time-to-first-accepted；
- known/unknown frontier bound、comparable/incomparable accepted results、incumbent actual objective/coverage随work和wall变化；
- peak RSS按state/continuation/memo/forbidden、live candidate transaction和retained winner分类；
- partial-state CardModule/Instr/Q50.0为零；`completeCandidates == candidateCardModules == Q50.0 invocations`，published winner为一；
- 与fresh独立`none`编译的actual work/运行证据只在外层报告比较，不进入search session。

profile sink只在明确请求的profiling run构造，普通unit/lit/compile无这些counters/string/report。重型LLaMA search首次出现在这里，并使用
current new path；旧search数据和历史日志不能回放。

#### 三类memo owner

1. **IR-derived facts**由policy-free builder定义；pass内可由MLIR AnalysisManager按operation scope缓存，planning driver在session中
   直接拥有同一builder的typed result，不保留`Analysis *`或另建epoch/fingerprint cache。
2. **assignment query memo**只在一个planning session内缓存pure typed result：domain descriptor、exact demand projection、J event graph、
   work/bound/estimate等。actual candidate IR、SPM/DDR problem/result、offset和owner relation不得进入memo或跨candidate复用。
   每类query声明`ObservedDependencyKey`，key包含它实际读取的axis choices和target facts。
3. **future-boundary DP memo**只在能证明两个prefix拥有相同completion set和downstream contribution时共享suffix result。

所有memo value不含Operation pointer、IR、offset、mutable solver、diagnostic string或statistics；source borrow由planning session
lifetime约束，pass analysis引用只在对应pass内存在。hash只作lookup，hit后比较完整typed key。evict/cache-off只增加work，不改变result；
并行miss使用single-flight或按semantic key确定性合并，不能让先
完成的thread成为semantic owner。

#### FutureBoundaryKey

```text
FutureBoundaryKey
  closed axis choices that downstream queries actually observe
  open SSA/value-version domains crossing the component boundary
  exact ownership/use/communication obligations
  Tile occupancy and target resource identities
  buffer lifetime/slot and event/completion obligations crossing boundary
  known route/link facts or opaque-fabric coupling class
  observable output/effect obligations
```

两个state只有在以下性质有proof时才可共享suffix：相同legal successor keys；exact structural work/downstream objective contribution相同
或可由key中保存的boundary summary组合。共享suffix只避免partial query，任何complete assignment仍独立走actual admission。key不能只用shape、bytes、layout、Tile count、estimated makespan或
plan hash。增加/删除一个fanout destination、relay child、slot backedge、worker join、hole Tile或output writer都必须改变key。

每个memo实现带test-only dependency perturbation：逐个修改所有可能输入，若result能变化但key不变就是contract bug。无法证明最小key时先用
完整typed parent prefix；profile再证明哪些字段可安全投影，不能为了hit rate先做窄key。

#### Component/separator DP

从current typed constraint graph构造components，而不是按function/op名字切：vertices为axis coordinates、physical versions、communication
actions、storage objects和events；constraint/resource edges来自A/J及各domain的typed facts。移除separator后components必须条件独立。separator至少含：

- 跨component value/domain/version/use bindings和observable effects；
- shared Tile placement、card DDR、DTE/FSM endpoint、qualified NoC link/channel；
- opaque NoC fabric若cost model把transfers全局耦合，则整个coupling class留在separator，不能伪装per-link independent；
- buffer alias/lifetime、K recurrence、J cross-component dependency/completion/resource order；
- 所有downstream cost/bound需要的boundary work summaries。

```text
solveComponents(separatorAssignment):
  for component in stable semantic order:
    enumerate exact local assignments conditioned on separator
    retain all future-distinct entries
    remove an entry only by proven same-boundary dominance
  combine component entries by deterministic DP convolution
  validate combined assignment through normal structural/J queries
  leave canonical global continuations available for every non-emitted sibling
```

DP entry保存typed boundary assignment、exact work、admissible bound和构造complete plan所需的local axis plans；不保存IR或local winner。
Pareto删除只允许相同`FutureBoundaryKey`且所有future objective terms monotone；Unknown/incomparable保留。没有fixed Top-k。chain/treewidth `w`
在每separator domain size `K`下典型为`O(N*K^(w+1))`；separator过宽或global resource无法factor时直接回Q51 best-first，不运行重solver。

#### 16-Tile重复工作的处理

完整4x4 mesh上，verified colored-topology automorphism和extensionally equal scope descriptors可以让16个Tiles共享relation/domain/resource
**query result**。每个Tile的semantic IDs、plan choices、buffer/events仍独立；memo hit只重绑定typed IDs，不强迫同assignment/offset。
unavailable Tile、非对称fanout、不同local extent、relay/worker/resource facts会拆开equivalence class。prepare/emission/Q50.0对每个
complete candidate的每Tile仍各一次，不试图缓存或复制actual module。

#### Stronger safe bounds

component DP可把Q50.B factor lower bound、H cut/endpoint bound、I minimum slots及J critical/resource/recurrence work按proved
separator组合成Q51-3 admissible bound。任何restricted solver的`OPTIMAL`只对其local conditioned problem有效；要提升为global bound必须加上
其它components的admissible minima和separator work。timeout/FEASIBLE/resource exhaustion只给proposal/Indeterminate。

#### Q52-1 Gate

- 先有profile再启用每类memo/DP；report把hotspot、key cardinality、hit rate、work/wall/RSS变化与selected coverage关联；
- cache off/on、eviction、parallel single-flight和不同hash seed保持Q51有界穷举exact set/objective/winner/coverage；
- dependency perturbation逐字段证明key完整；无pointer/digest/manual epoch、IR/offset/solver/result string cache；
- chain/tree/component DP与flat oracle逐separator assignment比较local/global sets与objective，Unknown trade-off不被Pareto删除；
- opaque/known NoC、DDR estimate、exact DTE、cross-Tile message、slot/recurrence/join分别证明正确进入separator；错误拆分有negative；
- symmetric 16-Tile case显著减少query次数，hole/asymmetric case停止错误sharing；assignment与winner actual Tile差异仍保留；
- partial-state materialization保持零；每个complete candidate actualize一次，loser销毁、winner不重建且只发布一次；ordinary compile无
  profile sink/stats/log，source不变；
- 文件分为`PlanningProfile.*`、`PlanningMemo.*`、`ComponentPlanning.*`，不回到controller/state大文件或引入全局cache。

### Q52-2 专项调研：bounded anytime policy、coupling-aware LNS与LLaMA gate

Shaw/Pisinger--Ropke LNS的核心是destroy一组相关variables，再用constraint solver repair大邻域；OR-Tools current LNS实现也从active
constraints、intervals、precedences和connected components构造neighborhood，而不是随机改一个变量。Wafer采用这个结构，但repair必须走
同一Q50 typed transitions；partial repair不物化IR，生成的每个complete assignment必须走同一Q50.F actual gate。

#### Immutable planning policy

```text
PlanningPolicy
  workAllowance
  wallDeadline
  frontierPriority
  memo/component settings justified by Q52-1 profile
  improvementNeighborhoods: optional deterministic schedule
  stateRetention: PreserveAll | ExplicitlyBudgeted
```

policy在session开始时冻结；不从本轮临时wall timing、thread completion或live board measurement中途改变。work allowance仍是Q51-5
deterministic primary stop，wall deadline只作外层安全中断。production default的具体credits/deadline/memo limits必须由本任务fresh profile选择、
在实现提交中记录依据并由回归test读取current constant；设计文档不预先拍一个candidate count或beam width。

它是compiler-owned policy，不进入IR/package/plan identity，也不新增public CLI。internal tests可以注入小allowance/neighborhood schedule；
普通用户仍只选`search|none`。

#### Constructive lane始终先取得actual Accepted candidate

Q51 best-first continuations是constructive lane，session开始即运行。memo/DP/proposals只重排或共享pure partial work。LNS只有在至少一个
complete assignment通过actual admission后才启用，不能消耗全部budget导致没有functional result。若constructive lane在production
allowance内没有Accepted candidate，返回`WorkExhaustedWithoutPlan`；不能让baseline或final publication补救。

work allocation使用profile-derived deterministic rounds：每轮先保证constructive continuation进展，再允许一个improvement neighborhood；
具体比例由Q52实施profile决定。任何时刻只有一个retained actual incumbent跨evaluation存活；每个complete candidate仍各执行一次
CardModule/Q50.0。

#### Typed neighborhood

```text
PlanningNeighborhood
  incumbentPlan: immutable reference
  releasedBindings: NonEmptyVector<CoordinateBinding>
  fixedBoundary: FutureBoundaryKey + complement bindings
  dependencyClosure: every downstream binding invalidated by released choices
  repairAllowance: reserved PlanningWorkAllowance
```

destroy从Q52-1 typed factor/resource graph选connected closure；释放一个上游choice必须同时释放所有依赖的downstream plans，不能保留stale
G/H/I/J/K。至少提供下列generic families：

- region neighborhood：相关Spatial/Region/Temporal/Representation/Buffer/Schedule choices；
- fanout/fanin neighborhood：producer owner、全部destination use bindings、communication tree/ring、staging/slots/events；
- resource-window neighborhood：共享DDR、DTE endpoint、known NoC link或opaque-fabric class内的movement/buffer/structure/schedule；
- pipeline neighborhood：stage cut两侧execution groups、chunk/temporal、movement、slots、worker/order/completion；
- output/effect neighborhood：所有影响同一observable writer/ordering的producers和publication actions。

family发现依据semantic graph/IDs/effects/resources，不用GEMM/attention/LLaMA/shape名。neighborhood size与调度顺序只能由profile和tiny regret
选择；没有固定“改一个axis”fallback。

#### Repair

```text
repair(neighborhood):
  validate fixedBoundary and dependency closure
  create the earliest PlanningState variant whose released axis is unassigned
  enumerate released coordinates with the same Q51 typed continuations
  keep complement bindings as ordinary constraints
  run normal structural/J/cost/bound and ForbiddenAssignment checks
  actualize every resulting complete assignment through Q50.F
  return typed actual results or Indeterminate on work exhaustion
```

local DP/solver的Feasible solution只是checked proposal，必须转换为普通complete assignment并通过global actual admission；restricted local
`OPTIMAL`不等于global optimal。timeout/resource exhaustion不产生rejection。repair的Accepted result进入Q51同一actual-result comparison，
不建立LNS-local winner，也不绕过统一work allowance或publication owner。

#### Coverage effect

- LNS作为额外proposal且constructive exact continuations全部保留时，不改变Q51 coverage；
- iterative widening/延迟queue只有最终仍可恢复所有states时可保留exact/bound claim；
- fixed beam、top-k、LNS-only、evict-without-continuation或其它永久丢state的policy必须设置`stateRetention=ExplicitlyBudgeted`，结果最多
  `BudgetedFeasible`；即使碰巧找到reference optimum也不能升级；
- learned/empirical priority只排序，不能成为bound/no-good/legality；
- 如果RSS profile没有证明必须丢state，production不启用有损retention。外部论文或系统的固定k/beam width不能成为默认值。

Q52优先交付PreserveAll memo/DP + constructive/LNS proposal lane；只有fresh heavy profile仍无法在可接受资源内返回plan时，才在同一任务中
选择一个明确有损policy并写清coverage/quality evidence。

#### Heavy workload与时间门禁

Q52 implementation完成后执行显式、非默认的fresh campaign：generic mixed DAG、HF prefill、functional decode和代表LLaMA block。每个case：

1. 独立运行一次current `none`，确认baseline仍完成并记录wall/RSS/actual work；
2. 独立运行`search`，在profile checkpoints记录time-to-first-accepted、incumbent actual objective/coverage、partial work、actual candidate
   次数和RSS；
3. 到selected production deadline必须返回search自己的retained Accepted winner，随后一次package publication和no-card通过；
4. 未返回、同candidate重复actualization、winner重建、publication超过一次或coverage虚报均使Q52未完成，不能用延长到无限、旧日志或
   baseline结果代替；
5. compare只在外层报告actual IR/work和quality，不回灌当前session。

profile可在1/3/10分钟等checkpoint观察并在一次专项campaign中延长到30分钟定位停滞，但这些是诊断上限，不是production默认或普通CI
timeout。最终实现必须根据曲线拐点选择并记录一个有限production policy；代表LLaMA在该policy下实际跑通才过gate。重型case不加入普通
unit/lit/CTest，每次功能小改也不自动重跑。

#### High-risk rules仍禁止

- 按单op/edge/layout/route local winner冻结轴；
- 用shape/bytes/estimated makespan相同合并future boundary不同的states；
- 只保留最大Tile、最大fusion、最短route或一个classic communication family；
- actual packing失败后在candidate IR内原地retile/spill/rebuffer，或在没有typed rejection/owner relation时推进controller；
- 缺estimate按零、solver timeout当NoSolution、restricted bound当global bound；
- 把16-Tile同构query复用升级为16 Tiles相同assignment或actual IR复制。

#### Files

`PlanningPolicy.*`只持immutable production settings；`PlanningNeighborhood.*`负责typed destroy closure；`PlanningRepair.*`复用Q51
transitions；`PlanningProfile.*`仍是optional sink。controller只调用这些接口，不把LNS塞回一个大函数。

### Q52 Final Gate

- Q52-1 safe memo/DP各自有profile依据，on/off保持Q51 exact oracle set/objective/winner/coverage；
- 每个neighborhood dependency closure与flat reference repair set相同；至少一个联合陷阱由LNS找到而single-coordinate proposal漏掉；
- PreserveAll lane不降coverage；故意启用有损retention稳定只报BudgetedFeasible，不能因找到optimum升级；
- every cutoff/cancellation/solver exhaustion保持typed result，预算内无Accepted candidate不fallback baseline；partial state零IR、每个
  complete candidate actualize/Q50.0一次、winner不重建且publication一次；
- heavy campaign的四类workload均用current fresh source；`none`与`search`独立，代表LLaMA在最终有限production policy内返回actual winner并完成
  package/no-card，wall/RSS/time-to-first曲线保存为本轮证据而非compiler输入；
- public CLI/help无candidate/beam/LNS/budget selector；ordinary compile无profile/stats/log，profile sink不改变plan/result；
- priority/memo/component/LNS/repair文件独立，source/call-tree只有一条current actual-candidate evaluator，无IR cache、clone/replay、第二
  evaluator或fallback；
- Q52独立提交并fresh验证后才标done；Q53只消费其current production policy，不重新调search算法。

## Q53：Production Board Readiness

### Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  Q60生成或current portable入口提供的只读source program；Q49.P产生的canonical `none`
  `PhysicalDataflowPlan`，或Q51/Q52 production policy产生的`search` plan；Q50.0、target、package和runtime
  current合同。
- Current stage responsibility:
  不再改变planner或lowering语义；从同一source bytes分别启动全新的`none`和`search`编译事务，证明各自只提交
  一个selected plan，并逐层验证accepted Tile dataflow、Instr、Target LLVM、ExecutablePackage和no-card
  invocation。把正确性、实际IR有效性和编译规模证据分开。
- Output IR / files:
  production输出仍只有current ExecutablePackage；显式测试运行可另写一次性IR inspection目录和测试结果，
  它们不属于compiler/package接口，也不被后续编译读取。
- Downstream consumer:
  Q53真实板端runner；普通用户仍只消费ExecutablePackage。
- User-level driver / named pipeline:
  标准`wafer-compile --optimization-policy=none|search`和`wafer-run --no-card|--board`；测试不手工拼pass。
- Explicit non-goals:
  不在Q53发明新candidate、cost、fallback或repair；不序列化shadow plan；不从名字恢复语义；不把历史输出、
  host timing、理论work或no-card当作实卡结果；不让IR dump、统计或heavy model进入默认编译路径。
- Done criteria:
  Q53-1的source/actual-IR/package/no-card矩阵由current构建fresh通过且相关测试实际执行；Q53-2完成全部
  board-ready准备并在真实设备上取得本节要求的correctness与matched performance证据。
```

### Q53-1 专项调研：成熟compiler怎样组合端到端证据

| 实践 | 成熟边界 | 本仓采用 | 不照搬的部分 |
| --- | --- | --- | --- |
| [MLIR Testing Guide](https://mlir.llvm.org/getting_started/TestingGuide/) | lit/FileCheck检查IR变换，integration test执行lowered program，unit test检查库合同 | Q50/Q51保留IR与oracle测试，Q53另走完整产品入口 | 不让一个FileCheck或一次pass成功代替source→package |
| [LLVM Testing Infrastructure](https://llvm.org/docs/TestingGuide.html)与[test-suite](https://llvm.org/docs/TestSuiteGuide.html) | regression、unit与whole-program分层；whole program编译、执行并对照reference output，compile time与exec time是不同metric | case-owned reference、完整编译产物和性能A/B各自出证据 | 不用短程序wall time或单次样本宣称性能改善 |
| [IREE testing guide](https://iree.dev/developers/general/testing-guide/) | compiler lit与core E2E分开；host编译standalone artifact，host/device runner执行断言 | host生成current package，no-card验证完整runtime plan，board runner只消费同一package | 不另建只为测试存在的lowering或device专用source语义 |
| [OpenXLA tooling](https://openxla.org/xla/tools) | `run_hlo_module`把backend编译执行结果同reference实现比较；SPMD使用独立multihost runner | Wafer的CPU oracle位于case，`none`/`search`只改变compiler policy | reference结果不参与search选择，也不把其它backend的调度当Wafer期望值 |

因此Q53必须同时回答四个不同问题：source语义是否保持、selected actual IR是否真的实现所选结构、package/runtime合同是否
闭合、真实板端是否正确且更快。任何一层只能为自己签字；Q50/Q51的局部证明不会被Q53重复实现，Q53也不能用最终输出相等
掩盖错误的额外搬运、错误的inactive Tile或多次commit。

### Q53-1 source与policy隔离算法

一个case只导出一次确定的只读source artifact；`none`与`search`各自在独立进程、work directory、output directory和
ProgramData ownership中重新parse/import该artifact。同一source identity用于证明输入一致，不共享Module、analysis、plan、
CardExecutable、target owner或staging directory。

```text
runHostVertical(case, sourceDType):
  source = case.export(seed, sourceDType)
  require current source verifier accepts source
  expected = case.computeReference(source-owned inputs)

  for policy in [None, Search]:
    run = createFreshRunDirectory(case, sourceDType, policy)
    package = invokeWaferCompileFreshProcess(
        source, policy, run.output,
        inspection = ExplicitTestOnly)
    require compile process selected exactly one plan and published one package
    inspectAcceptedIR(run.inspection, package, case.expectedFeatures)
    requireCurrentPackage(package, source, expected)
    requireNoCardInvocation(package, case.runtimePayload(expected))
    retain only test result needed for outer comparison

  compareHostBoundaries(None.result, Search.result)
```

这里的“selected exactly one”由Q51只保留一个accepted winner、禁止winner重建的ownership/call-graph gate和本次唯一published product共同证明；Q53不新增
production counter。显式inspection只截取accepted winner已经拥有的IR，普通compile不创建字符串trace、统计对象或dump目录。
测试结束不把inspection结果作为下一次编译输入。

两步decode在每个policy内部形成两次普通source→package事务。no-card阶段可以用case oracle的step-1 result构造step-2 payload，
它只证明两份package和runtime binding完整；只有Q53-2把step-1真实capture作为step-2输入后，才证明设备上的functional state
threading。`none`的step-1结果、package或source transaction绝不传给`search`。

### Q53-1 case矩阵

所有production vertical使用FP16和BF16两个case variant。shape、seed和模型配置只是测试数据；compiler合同只观察current
source IR、typed boundary、target/topology和Q50/Q51对象，不匹配case名称。

| Case | 通用语义覆盖 | 必须出现的host证据 | 执行分层 |
| --- | --- | --- | --- |
| representative heterogeneous structured DAG | rank>=3、主要维度>=1024且成对覆盖整除/非整除shape的matmul、elementwise、branch、fanin/fanout、reduction与multi-result | distinct active/inactive Tile interfaces、至少两个semantic roots、exact result coverage与stable package | ordinary source/no-card gate；适合快速失败定位但不缩小logical shape |
| convolution mixed DAG | convolution→branch→join→reduction，producer有多个consumer | multi-producer root boundary、region/value version、join依赖与最终两项输出 | ordinary或显式integration；不承担heavy timing |
| actual-SPM-rejected cross-Tile fanout/reduction | single-Tile complete candidate经actual SPM planning拒绝，另一个多Tile candidate被actual gate接受；同时有one-to-many和many-to-one payload | 两个candidate各自的actual result、完整demand owner relation、endpoint transfers/combine、send/receive/wait配对和合法inactive Tiles；不要求某个classic algorithm名称 | representative source/no-card；专门覆盖actual admission、H/J与NoC topology合同 |
| temporal-tail and rotating-storage chain | 1024级整除/非整除iteration domain、producer/consumer wave与可重复storage | exact full+tail coverage、slot/lifetime/order和必要completion；未选pipeline时不得伪造pipeline | representative source/no-card；与Q50.E/I/J/K有界oracle互补 |
| official HF attention prefill | causal Q/K/V attention长sequence、reduction与大intermediate | normalized op为`flash_attention`；K2 temporal state、block-sized score/probability scratch、无K2 spatial partial、无attention/Linalg residual，以及完整package | explicit model campaign，非普通unit/lit |
| functional two-step KV-cache decode | 显式past key/value输入与present key/value输出，第二步依赖第一步状态 | normalized op为`flash_decoding`；至少两个K2 contributions、all-and-only coupled-state transfer/merge、一个final output owner、两个独立package/state ports及两次no-card invocation；board阶段再验证actual continuation | explicit model campaign |
| representative Llama-2 7B block | attention、MLP、residual与大parameter集合组成的真实block | Q52 finite production policy内`search`自己的actual Accepted winner、无重建、一次publication、完整IR/package/no-card | explicit heavy campaign；不注册为普通回归默认集合 |

`capacity-forced`只描述case为何能覆盖跨Tile通信，不成为编译规则。其shape由测试中current target capacity推导或固定为已验证
反例；换target时case可以换shape，但H的payload-state、endpoint legality和J的event proof不变。classic Ring、tree、double-tree、
recursive exchange、row/column、Bruck等只负责给Q50.H建议通信图；Q53只验证winner的普通transfer/combine actions和终态payload，
不根据proposal family名称放行。

### Q53-1 accepted IR阶段矩阵

inspection必须parse IR并查询typed op/type/attr/interface；文件名和字符串计数只可帮助报错，不能决定pass/fail。每个Tile的
`card_id`、`tile_id`和`launch_slot`从IR/module metadata读取，再与package对应entry比较，不从`tile_00003`一类文件名恢复。

| 边界 | 正向证明 | 关键负向检查 | 主要owner |
| --- | --- | --- | --- |
| source | current verifier接受唯一entry、metadata、parameters和distributed boundary；case oracle从同一inputs产生 | source没有compiler marker、case-name分支或额外target hint | Q60/frontend |
| accepted Tile dataflow | all-and-only available Tile interfaces；每个required semantic root及piece由一个selected execution覆盖；Region、nested work、physical versions、movement actions和events引用closed typed IDs | 无未选root、重复piece、dangling use、临时plan attr、source op残留或跨Tile裸SSA引用；inactive Tile只能有canonical typed boundary | C、D、E、G、H、I、K、J materialization verifier |
| Instr | TileRegion source分类全部消失；compute、RDMA/WDMA、DTE、NCC、alloc、loop和wait均由typed verifier接受；SPM/DDR offset与worker/completion闭合 | 无unknown-source-as-legal、未placed storage、missing peer、pending async work、resource collision或post-hoc repair | Q50.0/Instr verifier |
| Target LLVM | 每个Tile interface只lower/translate一次；entry、typed target-call metadata、argument row和physical identity roundtrip | 无额外export、undefined target call、metadata缺失或按module ordinal恢复Tile | target codegen/readback |
| ExecutablePackage | strict current loader验证manifest、modules、program-data、digest和exact field set；entries覆盖all-and-only physical domain | 无source tree、未引用module、重复/缺失entry、部分staging output或额外旧格式文件 | Q56/Q59 package owner |
| no-card invocation | current runtime建立完整memory、argument、module、phase、transport和completion plan，并明确`board_execution: false` | 在全部validation闭合前无provider/device effect；不把no-card成功写成设备correctness | runtime/package owner |

Q53不再给每一层添加一个重复module walk。局部dialect verifier、Q51 card-scoped materialization verifier、Q50.0 resource verifier、
target readback和runtime strict loader各自保持原责任；Q53的inspection只检查跨层关系和下述“是否有效”，不再实现一套legality。

### Q53-1 actual IR effectiveness

Q53不能只证明“合法”。test-only inspector从accepted Tile dataflow与Instr建立最小的typed关系：semantic output piece→
`PhysicalVersionId`→storage object→movement action→consumer use，并用Q50.E的occurrence和J的event order解释实际执行。
它不保存或重建planner state。

```text
inspectAcceptedExecution(tileDataflow, instr, expectedFeatures):
  require both stages cover the same physical Tile identities
  derive actual producer/use/version relation from SSA and typed IDs
  derive actual storage/lifetime relation from memory spaces, offsets and uses
  derive actual transfer/combine/completion relation from typed actions/events
  prove every observable output has one complete producer chain

  for feature in expectedFeatures:
    apply the feature's semantic predicate to those derived relations
  reject any unexplained spill, transfer, wait, allocation or execution
```

具体predicate如下：

- **region retention**：两个或更多structured computations共享合法iteration relation并位于同一actual TileRegion/traversal；被保留的
  producer version直接到达consumer use。不存在该version的WDMA到DDR后又RDMA回同Tile/同lifetime的往返；不是数一个`fused`字段。
- **temporal coverage**：top-level与nested execution occurrence并集精确覆盖logical iteration domain，交集只包含语义允许的
  reduction/combine；末尾piece的offset/size来自exact tail，而非越界padding或漏算。
- **physical representation**：每个use绑定到compatible actual version；需要转换时有唯一producer action和storage，不能通过
  `lookupAny`或同shape替代SSA identity。
- **cross-Tile communication**：每个required payload state都有从initial state到terminal binding的transfer/combine proof；actual
  send/receive的endpoint、chunk/span和completion一一匹配。current Direct DTE内部route不透明，因此不检查虚构的per-link/channel
  schedule，只核对physical endpoint legality、minimum-hop/cut事实的sound使用及software relay/event graph。
- **storage**：SPM/DDR对象的size/alignment/offset覆盖实际span；相交lifetime不重叠地址；exact alias和rotating slots满足I的
  occurrence relation；无从plan之外冒出的hidden spill/staging object。
- **execution order**：所有data/effect/resource边在actual order或wait/join中被满足；DTE/NCC completion早于相应reuse、publication
  和entry return。只有K选择Pipelined时才要求prologue/steady/epilogue或qualified periodic结构，Serialized不得被判失败。
- **work parity**：从actual Instr重算的exact compute/movement/communication execution及memory high-water与selected plan的Q51
  commit parity一致；objective estimate只用于解释，不能替代actual work，也不在Q53重新选winner。

“不存在无意义spill”只对同一`PhysicalVersionId`、同一required use path和没有外部可观察/容量必要性的往返成立；合法的跨region
publication、fanout staging、layout conversion或capacity-required spill不能被粗暴禁止。独立反例必须覆盖同shape不同version、合法
external publication、relay staging和opaque NoC，防止effectiveness checker过拟合。

### Q53-1 package与no-card gate

每个`case × dtype × policy × decode-step`都必须从本轮source生成自己的package并运行current strict loader/no-card。authoritative
schema与binding检查调用package/runtime共享实现；Python runner读取manifest只用于准备case payload和做额外all-and-only断言，不能
维护兼容reader或在loader拒绝后继续。

必须证明：

1. package只有current canonical files；module和program-data的size/digest经read-backed owner验证；
2. manifest entry与accepted IR/Target LLVM逐项对应`(card_id, tile_id, launch_slot)`，覆盖且只覆盖current 16-Tile domain；
3. `TileEntryArgument`、program tensor、target tensor与external input/output all-and-only闭合；package-owned payload只写一次，
   external ports没有package bytes；
4. 每个module/export/phase/transport/completion均能形成runtime invocation；inactive Tile仍有合法entry，不伪造工作；
5. no-card在任何provider effect前完成验证，并明确没有board execution；缺resource、错误payload extent、坏digest、重复Tile和
   stale target identity由既有negative suites fail closed；
6. compile/package/no-card任一失败只终止当前policy的独立run，不调用另一个policy，也不读取另一个run的output。

### Q53-1 分层执行与完成门禁

- ordinary host gate运行真实规模representative structured、communication、tail/storage矩阵；unit、lit、catalog和no-card按`nproc`可用并发度执行；
- HF prefill、functional decode和Llama block属于显式model campaign。Q52已经选定的finite production search policy在这里原样消费，
  Q53不延长deadline来掩盖search停滞；
- 每个registered test必须检查实际执行，configured dependency缺失导致的unsupported/skipped不计入通过数；
- compile timing、planning profile和IR inspection只在相应显式run开启；关闭它们后plan key、IR和package identity不变，普通编译不
  构造报告；
- correctness run不比较容易受缓存/并发影响的wall time。host compile规模只在显式campaign记录，并与board performance分开；
- 不读取历史raw、历史package或旧日志。本轮代码/runner变化后，证据只来自本轮current build和fresh invocation；
- Q53-1完成要求：两种policy的全部representative case和model source/no-card矩阵成功，实际IR predicates、package与runtime检查全部闭合，
  Q51的complete-candidate一次性actualization、winner无重建和唯一publication证明仍由current路径执行。它只完成无板阶段，不单独把Q53标成`done`。

### Q53-2 专项调研：设备qualification、通信correctness与matched benchmark

[IREE benchmark/profiling](https://iree.dev/developers/performance/benchmarking/)把整体benchmark与细粒度device profile分开；
[LLVM benchmarking tips](https://llvm.org/docs/Benchmarking.html)要求高分辨率计时、重复运行并控制噪声；
[Google Benchmark random interleaving](https://google.github.io/benchmark/random_interleaving.html)说明交错repetition能降低系统状态漂移；
[NCCL tests](https://github.com/NVIDIA/nccl-tests)则在同一collective workload上分别做warmup、重复计时和correctness check。
这些实践共同支持以下边界：Q53主门禁测完整card invocation并逐次验结果；细粒度DTE/NCC profile只在需要归因时显式开启；
`none`与`search`交错而不共享编译结果；通信吞吐不能脱离正确payload和completion单独签字。

#### Board-ready先于设备访问

无板阶段为每个要上板的`case × policy × step`准备完整对象：

```text
BoardCaseRun
  immutable source identity
  current ExecutablePackage
  policy: None | Search
  input payload bindings
  all output oracle bindings and guard regions
  optional continuation edges between ordinary invocations
  completion deadline

MatchedBoardCase
  None BoardCaseRun
  Search BoardCaseRun
  fixed pair count and deterministic order
```

这些只是test-runner内存对象，不是compiler IR、package schema或磁盘sidecar。每个package必须已由Q53-1的fresh product compile和
no-card产生；runner在无设备环境中已经能闭合port、payload、output、continuation和timeout。缺任一项、使用历史package、依赖
上板后临时补oracle或需要修改source的case都不是`board-ready`。

Board-ready矩阵包括全部representative case、HF prefill、functional two-step decode和representative Llama block的`none`/`search` package；
真实Q53性能门禁只执行代表性子集，避免把完整无板矩阵机械变成昂贵板测：

- 一个真实规模multi-Tile communication case先证明current package/runtime/Direct-DTE路径和完整16-Tile launch可用；
- HF prefill或functional decode至少选一个做matched A/B；若选decode，两个step构成一个不可拆的case sequence；
- representative Llama block必须做matched A/B；两种policy都必须实际执行，不能只运行search。

#### 一个qualified session，case级串行，Tile级按计划并发

当前runtime已经提供`QualifiedBoardRuntimeSession`。Q53 runner必须在一个进程中直接复用它，而不是每次iteration重新启动
`wafer-run`并重复device count、selection和inventory：

```text
runBoardCampaign(cases, driver, qualification):
  first = first case's first complete correctness invocation
  result, session = executeBoardInvocationAndStartSession(
      first.package, first.request, driver)
  checkResult(first, result)              // qualification and a real case together完成

  for invocation in remaining fixed sequence:
    require session.isUsable()
    result = executeBoardInvocationInSession(
        invocation.package, invocation.request, session)
    checkResult(invocation, result)

  finish without reset, power or extra provider probe
```

第一次完整case同时取得session capability，不额外跑heartbeat、空kernel或无关calibration。后续request必须携带同一device和qualification；
session验证它们一致，但不再调用device enumeration/selection/info。runtime library digest、runtime version、device name、PCI identity、
exact available Tile domain和target capability在本次重启会话只确认一次。driver或device被判poisoned后session不可再用。

“板测串行”指**case/invocation在host侧一个接一个执行**，禁止两个CTest、两个进程或两个case同时占设备。它不表示16个Tile
串行执行：每个launch phase由一次`submitKernelPhase`提交all-and-only 16个`BoardTileLaunch`；provider可以使用多个command queues，
设备并发由selected J/K event/resource plan、Tile-local control和DTE completion决定。`launch_slot`是稳定binding/submission order，
不是Tile之间的串行依赖。inactive Tile可以只有canonical entry；runner不得为了“16并发”伪造工作，也不得把16个Tile拆成16个
独立host launch。

#### 每次invocation的correctness与lifecycle

每个第一次运行、未计时settling run和计时sample都必须完整检查，而不是只检查最后一次：

1. result覆盖all-and-only package entries，`(card_id,tile_id,launch_slot,module,entry)`与package一致；
2. validation、resource allocation、H2D、module load、entry resolve、launch、completion、D2H和cleanup按current lifecycle闭合；
3. 所有case outputs通过case-owned oracle，guard区域保持；没有未声明capture或丢失output；
4. 每个phase在同一absolute deadline内terminal，entry return满足local drain，card-level required completion全部观察到；
5. functional decode的step-2输入只来自**同一policy、同一case instance**的step-1 actual capture；两个policy不交换状态；
6. 每次结束后submission state正常release，下一次才可开始；不存在跨sample遗留DTE/NCC work或复用旧output buffer内容。

timeout、provider error、missing completion、output/guard失败或session poison会立即终止当前campaign。runner不自动retry，不发reset/power，
也不继续另一个policy来“收集更多数据”。此前样本可作为本轮raw诊断，但本次campaign整体不产生通过结论。

#### NoC与经典communication plan的板端证明范围

Q50.H已经把Ring、tree、double-tree、recursive exchange、row/column、Bruck等降为普通transfer/combine proposal。Q53板端不读取
family标签，而验证三件事：

- package声明的Direct-DTE/DDR transport与live provider capability一致；缺能力时在launch前拒绝，不换transport、不回baseline；
- cross-Tile case的每个required terminal payload得到正确结果，实际send/receive/wait/combine已在Q53-1与Instr/package对应；板端
  completion证明该software action graph没有暴露的hang或未完成依赖；
- overall card latency是通信plan在compute、storage和schedule共同作用下的结果。只有current target显式提供且已qualification的
  route/counter/timestamp才能作为补充归因；opaque DTE内部路径仍不宣称per-link bandwidth、contention或某classic family最优。

representative communication case可以同时报告payload bytes、actual endpoint action count和card latency，但这些只是该case的证据。不能用一个
collective microbenchmark替代HF/Llama整体A/B，也不能用较少message或理论cut bound替代真实改善。

#### Matched A/B执行算法

两份package在上板前已独立compile/no-card。进入同一qualified session后，每份package先运行一次不计入性能结论的完整settling
invocation；它仍必须通过上述correctness/lifecycle。这既消除首次module/runtime状态，又不是失败retry。之后运行固定的paired序列：

```text
runMatchedCase(matched, session):
  runAndCheck(matched.None)                // settling, unmeasured
  runAndCheck(matched.Search)              // settling, unmeasured

  samples = []
  for pair in [0 .. matched.pairCount):
    order = [None, Search] if pair is even else [Search, None]
    pairSamples = []
    for policy in order:
      request high-resolution completion observation
      result = runAndCheck(matched[policy])
      pairSamples[policy] = extractTiming(result)
    samples.append(pairSamples)

  return classifyMatchedResult(samples)
```

current initial qualification contract固定attention representative为5 pairs、Llama block为3 pairs；它们在运行前确定，不能看到结果后
增加样本。所有sample保留，不删outlier。若后续current workload持续时间变化，可以在case合同中重新审查这一固定值，但不能把自适应
repeat变成“直到search赢”。pair的首个policy交替，降低温度、runtime和device状态单向漂移；不使用并发A/B。

primary metric是`BoardRuntimeInvocationResult::launchToCompletionNanoseconds`，即紧邻provider submission到完整card completion；
source export、compiler、package load、allocation、H2D、D2H和output comparison不混入该metric。`hostSubmitNanoseconds`、qualified
`deviceExecutionNanoseconds`和actual IR work只用于解释差异，不能与primary metric相加或替换。compile time由Q52/Q53-1另报。

对第`i`个pair定义：

```text
delta_i = none_launch_to_completion_i - search_launch_to_completion_i
resolution_i = max(none_observation_resolution_i,
                   search_observation_resolution_i)
```

`Improved`要求：所有sample correctness通过；至少过半pairs满足`delta_i > resolution_i`；并且`median(delta_i)`大于
`median(resolution_i)`。`Regressed`为对称负向条件；其它情况是`Inconclusive`。同时报告每个raw pair、median ratio和观察分辨率，
不只报告一个百分比。`Inconclusive`可以保留board correctness结论，但Q53仍停在`board-ready`；本轮不自动追加样本或重启设备。
新的比较必须由明确安排的fresh session产生。

这是一项有限的production qualification gate，不声称长期统计分布或所有模型最优。若要持续性能监控，另由benchmark系统保留
多次session结果；不能把该历史数据库重新作为Q53当前输入。

#### Runner与文件责任

当前`wafer_board_pytorch_test.py`同时做source export、compile、manifest/binding、no-card、单policy board loop和decode continuation，
实施时必须按真实责任拆开：

- `wafer_pytorch_board_cases.py`只保留source construction、inputs和case-owned oracle/continuation definition；
- source/package runner只负责Q53-1的export、两份独立compile、inspection、payload preparation和no-card；不拥有device；
- board case runner只把已验证package、raw bindings、oracle和continuation转成固定invocation sequence；不调用compiler；
- test-only matched executor用`QualifiedBoardRuntimeSession`串行执行sequence和形成typed sample result；不解析source、plan或case name；
- current package loader、`BoardRuntimeInvocationRequest`、board file binding/output publication和TX provider仍由Runtime库唯一实现；若
  `WaferRunBoardIO`需要同时被`wafer-run`和test executor复用，应迁成窄Runtime support library，不复制一份Python schema/ABI validator。

matched executor不是production compiler路径，也不为普通`wafer-run`增加A/B、repeat或统计默认。它只在显式hardware campaign构造；
结果文件是本轮测试产物，不进入package、IR或下一次compile。

#### Q53-2 Gate与状态

`board-ready`要求：

- Q53-1全部current source/IR/package/no-card门禁通过，且registered test没有以unsupported/skipped冒充执行；
- 上述board case、oracle、guard、continuation、fixed order、deadline和single-session runner完整；
- 每个待执行package已经fresh生成且与runner binding一致；不需要设备上临时修改任何文件；
- static/runtime tests证明qualification只发生一次，case串行而每phase提交complete Tile domain，timeout/poison后零后续provider call。

`done`要求在一轮current、单进程、无异常的qualified device campaign中：

- small multi-Tile communication case correctness/lifecycle通过；
- Llama `none`与`search`全部settling和3个pairs通过case oracle，matched分类为`Improved`；
- prefill或functional decode代表的两种policy全部settling和5个pairs通过；decode还必须逐pair使用同policy actual continuation，matched
  分类为`Improved`；
- 所有证据来自本轮package和本轮输出；无retry/reset/power、无历史raw输入、无policy fallback；
- Q53才从`board-ready`改为`done`并解除Q48前置阻塞。若correctness通过但任一matched结果`Inconclusive`或`Regressed`，明确保留
  `board-ready`，不能用Q51 cost、Q52 profile或更少actual work代签。

## 提交与收尾

1. Q58、Q56、Q50.0、Q54、Q59与Q60等既有前置保持；当前严格按`tasks/progress.md`列出的20个唯一work items形成独立可评审提交；
   不得倒序，也不得用Q owner内部阶段建立第二份调度。
2. 每个domain work item完成donor迁移与direct query/test-only apply时，同批扩search controller consumer及feasibility dependencies。
   deterministic-baseline-closure不进入search；full-feasibility之后search-control-closure闭合actual-result controller，
   unified-search-closure闭合完整候选actual evaluation与唯一winner发布。旧mechanism/selector/repair/test只有在替代work item的owner、
   consumer和witness就位后才能删除。
3. 状态转换以 `tasks/progress.md` 为准；本计划不单独维护第二份动态状态表。
4. 每项提交前运行 fresh 定向 build/test；端到端或主线 gate 还需确认 relevant lit/CTest 实际执行而非 skip/unsupported。
5. 提交使用 `Codex <codex@openai.com>` 并附 `Co-authored-by: hehesnail <shashen008he@gmail.com>`。
6. 稳定 bug 模式进入 `memory/bugs.md`，可复用 build/debug workflow 进入 `memory/general_dev.md`；临时 profile 数字、
   workload 路径、未校准 prior 和单 case winner 不进入长期设计。
