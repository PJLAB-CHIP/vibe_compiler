# Physical Dataflow Planning 与 Selected Execution 实施计划

状态：Q50.0无策略CardExecutable编译/准入边界仍是已闭合基础；Q50.A production demand boundary与Q49.P deterministic
baseline已经重新打开，前者重建exact ownership/demand，后者删除per-coordinate actual compile并改用Q50.F plan proof。
Q51.Core、Q50.S attention vertical、Q50.B–K、Q51、Q52与Q53的search部分全部重新进入审计和施工状态。此前关于
baseline incumbent、complete-candidate反复物化、统一全轴search、scalability/LNS及model-scale search质量的完成声明均不再是
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

已确认的硬事实：

- Q51.Core提交删除了`CardExecutableSynthesis`、rank search/evaluation/selection、graph placement、schedule、cost和attention等
  多个独有算法owner；新`CardExecutableSearch`当时只是baseline passthrough，通用`runSearchControl`从创建到退役始终只有unit
  consumer，没有进入production search。
- current `UnifiedPhysicalDataflowAssignment`只有spatial、coupled、temporal、implementation、representation、movement和buffering；
  Q50.F feasibility、Q50.J instruction schedule和Q50.K stage pipeline不在联合assignment。
- Q50.F的`analyzeScopedFeasibility`没有production caller。Q50.J只在Tile lowering后构造domain并直接应用first assignment；Q50.K
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
- current normal candidate evaluation会构造CardModule、lower到Instr、做SPM/DDR planning并计算cost；增加evaluation上限会重复
  complete materialization/lowering后销毁loser。该行为不是计划级search与一次性commit边界。

初步任务判定如下；“部分可保留”只表示已有mechanism可能继续复用，不表示任务完成：

| 任务 | current事实 | 初步判定 |
| --- | --- | --- |
| Q51.Core | baseline passthrough加未被production调用的通用control模板；旧控制/代价/placement/schedule owner已删除 | 未实现 |
| Q50.S | attention/decode proof与actual alternative materializer有实质代码；旧11项能力测试缩为5项，且production顺序/预算让alternative饥饿 | mechanism部分可保留，算法与集成重审 |
| Q50.B | per-node all-iterator legality domain和exact-demand trial存在；graph-level compact topology、independent-component、chain DP/general-DAG bounded search未迁 | legality domain可保留，placement算法缺失 |
| Q50.C | single-root apply与多结果/reduction/support/effect测试较完整；后续真实模型仍暴露multi-producer/output closure缺口 | apply部分可保留，完成结论撤回 |
| Q50.D | connected partition domain和coupled apply存在；production proposal只有singleton或greedy merge，旧connection coupling算法未承接 | domain/apply部分可保留，选择算法缺失 |
| Q50.E | 全整数tile size与loop permutation可枚举，capacity proposal复用per-node贪心；无factorized breakpoint和联合质量算法 | domain/apply部分可保留，scalability/选择缺失 |
| Q50.F | pure analysis和unit存在 | 未接入production |
| Q50.G | per-value layout Cartesian domain替代PBQP/Top-4及movement-elimination算法 | 核心算法丢失 |
| Q50.H | exact fragment/basic route/multicast/reduction apply存在；大量NoC/alias/lifetime/partial-dataflow能力无承接证明 | 部分迁移，重大缺口 |
| Q50.I | slot-count domain与rotating allocation mechanism存在；first choice固定single buffer，无联合选择算法 | mechanism部分可保留，选择缺失 |
| Q50.J | hard dependency/order/worker domain存在，但不属于unified assignment，lowering只取first | 未接入search，旧能力需逐项复核 |
| Q50.K | wrapper按已选buffer scope自动物化stage，没有独立domain或search transition | 任务合同未实现，proof大幅丢失 |
| Q51 | dependent Cartesian iterator直接materialize每个complete point；缺F/J/K、真实Core和plan-level winner | 完成结论无效 |
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
| Q50.J | `ReadyOrder`、`WorkerPlacement`及target scheduling analysis共约1300行；22项direct tests覆盖DTE issue/wait window、RAW/WAR/WAW/view alias、completion barrier和worker components | current `InstructionSchedule`约681行和6项tests；domain有实质内容，但production只取first assignment |
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
  partial frontier、winner选择或一次性commit。
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
5. **Winner-only commit**：planning阶段零IR，只有unified-search-closure选出的winner或baseline关闭的唯一plan进入一次Q50.0。
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
  explicit movement、buffer、ready/order/worker、NoC/DDR/compute resource timeline和条件式stage pipeline。只有resolved
  winner assignment才能物化actual IR并跨越共同compile/verification seam；planning search不为比较候选构造CardModule或Instr。
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
  program data ownership、package data与compile commit。当前37个execution work items按`tasks/progress.md`唯一顺序逐项闭合：
  先建立semantic/spatial/demand与canonical plan artifacts，再闭合deterministic baseline；随后建立完整physical domains、search controller、
  unified single-winner commit、attention production evidence和measured scalability。最后production-host-readiness从Q60产品入口生成
  fresh package、oracle、runner并通过no-card达到board-ready；真实matched板端A/B不在当前目标内。
```

Q50.S只把完整attention归一为一个带fixed FA/FD mode的semantic op，不返回graph assignment或algorithm domain。Q50.B/E分别选择
K2 spatial partition与temporal block，A提供coupled partial/merge；只有最终winner在新Card subtree中物化selected Linalg/Tensor/SCF，
再转换为wafer.tile。physical placement、temporal tile、layout、buffer和schedule仍由同一后续联合搜索选择。

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
3. winner construction使用的prepare/emitter primitive；
4. 作用域明确的accepted facts、`deferred(required coordinates)`、proven exact rejection、unsupported或indeterminate result。

mechanism 不得返回“本轴最佳值”，不得按估算删除其它轴仍可能使之变优的候选，不得在失败时自行 retile、改 layout、
spill、减 buffer 或改 schedule。estimate 只能排序或构成已证明的 lower bound；最终 legality 和 cost 来自 actual IR。
`ResourceExhausted`、solver timeout属于indeterminate，只消耗work并保留state；只有producer-supplied extension-closed exact proof才能形成
causal `ForbiddenAssignment`。Unsupported不形成no-good，internal failure是compiler bug并停止compile。

### Compile seam

共同搜索与 mutable IR 之间只保留一次commit seam：

```text
current immutable structured IR
  -> query-local analysis/domain/transition
  -> planning frontier与cost/bound
  -> selected complete winner assignment
  -> winner的一次性actual materialization与Q50.0 verification/lowering
  -> accepted actual owner | typed compiler/unsupported failure
```

- query-local state 只保存不能从 current IR 和已选坐标重算的 typed assignments；ready/live、exact
  demand、lifetime、resource calendar、SPM high-water、lower bound 和 makespan 都绑定一次immutable IR borrow并按typed
  assignment重算，是query-local analysis cache，不进入state identity，也不序列化为output或计划attr；
- actual transformation不修改原source，也不在scratch IR内修复candidate；只有winner进入该边界。Q50.0失败说明planning
  legality或lowering合同缺口，不能作为正常控制流再物化下一个candidate；
- analysis cache只在一个immutable borrow内存活；Q50.A `IREpoch`只拒绝跨borrow trial，不进入semantic cache key，也不
  代替nested structural snapshot。borrow内的key必须包含target facts和会影响结论的全部typed assignments；改变traversal、
  tile、layout、movement、buffer或schedule后，旧calendar、lifetime、SPM/legality结果全部失效；
- 任意时刻最多一个 live actual owner；host 可并行计算 immutable analysis，但 candidate set insertion 和 tie-break 使用稳定 key；
- regular mapping、reuse signature和coarse resource estimate只能给普通typed transitions排序；不能clone-per-mapping，不能把
  reuse/cost annotation写进候选IR，也不能用function name、JSON或opaque solver payload跨越compile seam；
- 整个search invocation只有selected winner通过一次完整CardModule splitting、TileRegion-to-Instr、fresh completion、SPM/DDR、
  resource、ABI和verification；不能用“每个candidate各通过一次”解释一次性要求。winner不重新物化或编译。

### Search result 等级

- `objective-optimal`：finite domain已完整覆盖或所有未展开状态被exact infeasibility/equivalence/admissible bound排除，已证明current
  planning objective optimum及tie key；
- `feasible-with-bound`：未展开 completion 仍被完整 exact candidate set（包括可惰性展开的 parent）表示，且
  admissible lower bound 有效；在证明最优前因 work/time budget 中止也可报告有效 gap；
- `feasible-unranked`：至少一个full-proof plan可提交，但objective Unknown或存在incomparable full plans；只作functional选择；
- `budgeted-feasible`：fixed-width/diverse candidate set、LNS-only 或其它启发式已永久丢弃或未表示某些合法
  completion，或者remaining state没有admissible bound；不得宣称global bound或最优。

所有search等级只描述本次planning session仍表示的plan set，不接收Q49.P `none`结果。预算内没有任何full-proof search plan就是typed
search failure，不能隐式运行baseline或伪造fallback。`none`对声明支持的正常上游输入必须
独立完成自己的deterministic feasibility legalization；若最终exact gate仍失败，必须是最小canonical路径已被typed proof排除、
输入确实超出支持域，或明确的compiler/internal failure。

## 施工 checkpoint

| 顺序 | Work item | 设计owner | 单一输出责任 | 下一work item |
| --- | --- | --- | --- | --- |
| 1 | `spatial-plan-schema` | Q50.B | Spatial plan/assignment schema、close和validator | attention-normalization |
| 2 | `attention-normalization` | Q50.S | attention op/interface、graph proof、FA/FD classifier及共同normalization | attention-spatial-integration |
| 3 | `attention-spatial-integration` | Q50.S | K1/K2 role及FA/FD canonical/full spatial constraints | canonical-spatial-assignment |
| 4 | `canonical-spatial-assignment` | Q50.B | all normalized roots的deterministic closed assignment | foundational-coverage-matrix |
| 5 | `foundational-coverage-matrix` | 16（Q50.B/Q50.S） | 前四项已提交边界的真实规模整除/非整除及结构语义覆盖矩阵 | exact-demand-boundary |
| 6 | `exact-demand-boundary` | Q50.A | operand demand、final owner与per-output reduction requirement | attention-demand-integration |
| 7 | `attention-demand-integration` | Q50.S | Q/K/V/mask demand及coupled contribution/merge | canonical-root-work |
| 8 | `canonical-root-work` | Q50.C | canonical RootRegionWork与single-root leaf primitive | canonical-region-plan |
| 9 | `canonical-region-plan` | Q50.D | singleton RegionPlan、execution instance及use binding | canonical-temporal-plan |
| 10 | `canonical-temporal-plan` | Q50.E | canonical TemporalPlan、tail及loop order | canonical-representation-plan |
| 11 | `canonical-representation-plan` | Q50.G | canonical RepresentationPlan与resource description | canonical-movement-plan |
| 12 | `canonical-movement-plan` | Q50.H | canonical local/DDR/peer correctness carrier | serialized-execution |
| 13 | `serialized-execution` | Q50.K | unique Serialized execution identity | canonical-storage-plan |
| 14 | `canonical-storage-plan` | Q50.I | single-slot BufferPlan及lifetime facts | canonical-schedule |
| 15 | `canonical-schedule` | Q50.J | source-order/worker0 ClosedSchedulePlan | attention-work-projection |
| 16 | `attention-work-projection` | Q50.S | AttentionWorkDescription及C–K/F resource projections | canonical-feasibility-proof |
| 17 | `canonical-feasibility-proof` | Q50.F | canonical problem/parity及FullFeasibilityProof | attention-selected-decomposition |
| 18 | `attention-selected-decomposition` | Q50.S | winner-only selected Linalg/Tensor/SCF→wafer.tile builder | deterministic-baseline-closure |
| 19 | `deterministic-baseline-closure` | Q49.P | pure legalization、一次commit/Q50.0及fresh none纵向 | spatial-domain |
| 20 | `spatial-domain` | Q50.B | complete spatial successors、reference enumerator及proposal | search-control-foundation |
| 21 | `search-control-foundation` | Q51.Core | SpatialState frontier/continuation及public search routing | root-work-domain |
| 22 | `root-work-domain` | Q50.C | full root/merge work domain、Core consumer及selected emitter | region-execution-domain |
| 23 | `region-execution-domain` | Q50.D | region/execution/use-binding domain及Core consumer | temporal-domain |
| 24 | `temporal-domain` | Q50.E | complete temporal sizes/orders/tails及Core consumer | partial-feasibility |
| 25 | `partial-feasibility` | Q50.F | A–E minimum/interference、Deferred及causal query | layout-domain |
| 26 | `layout-domain` | Q50.G | representation constraint solver、Core consumer及apply | movement-domain |
| 27 | `movement-domain` | Q50.H | local/DDR/DTE/relay/collective domain、proof及Core consumer | storage-domain |
| 28 | `storage-domain` | Q50.I | fresh/alias/reuse与1..U slot domain及Core consumer | event-resource-foundation |
| 29 | `event-resource-foundation` | Q50.J | EventGraph、resource/recurrence facts及Core consumer | execution-structure-domain |
| 30 | `execution-structure-domain` | Q50.K | Serialized/Pipelined structure domain及Core consumer | structure-specific-storage |
| 31 | `structure-specific-storage` | Q50.I | fixed-K occurrence、slot multiplicity、rotation及lifetime closure | schedule-domain |
| 32 | `schedule-domain` | Q50.J | fixed-K/I order、worker、resource与completion domain | full-feasibility |
| 33 | `full-feasibility` | Q50.F | full resource proof、oracle及Core admission input | search-control-closure |
| 34 | `search-control-closure` | Q51.Core | full-plan admission、cost/bound、causal rejection、coverage及controller oracle | unified-search-closure |
| 35 | `unified-search-closure` | Q51 | bounded exhaustive oracle、single winner及一次production commit | attention-production-closure |
| 36 | `attention-production-closure` | Q50.S | donor retirement及prefill/decode none/search package/no-card | search-scalability |
| 37 | `search-scalability` | Q52 | measured memo/DP/bound/LNS及有限预算LLaMA一次commit | production-host-readiness |
| 38 | `production-host-readiness` | Q53 | fresh source/IR/package/oracle/runner/no-card矩阵 | Q53 board-ready |

work item是唯一调度身份，Q50.*只表示设计owner。一个owner可以拥有多个work item，但每个work item只出现一次、只签发一个typed
artifact或production gate；owner整体完成由`tasks/progress.md`的owner map汇总，不进入施工队列反复切状态。

`foundational-coverage-matrix`不修改前四项已经提交的IR/API边界，也不伪造新的compiler artifact；它补齐这些边界此前由
个位数shape留下的证据缺口。至少包含rank不小于3且主要维度不小于1024的整除/非整除shape对，覆盖attention普通/多K2、
spatial单轴/多轴、all-16 Tile、fan-in/fan-out、reduction及exact shard coverage，并检查source IR不变、coverage无hole/overlap、
stable identity、mode constraints、owners和merge groups。该项fresh完成并提交后，`exact-demand-boundary`才恢复`doing`。

```text
Pipeline position:
- Upstream IR / input:
  已提交的normalized attention IR、SpatialPlan/SpatialAssignment schema、attention spatial constraints与canonical closed assignment。
- Current stage responsibility:
  为上述边界补齐真实规模的整除/非整除和结构语义覆盖矩阵；发现实现缺陷时只在原owner中修复并复验。
- Output IR / files:
  owner原有unit/lit测试中的representative fixtures、exact assertions及fresh测试结果；不产生新的production IR或sidecar。
- Downstream consumer:
  exact-demand-boundary及后续canonical/search work items的可信输入边界。
- User-level driver / named pipeline:
  WaferUnitTests、相关configured lit与原production normalization pipeline。
- Explicit non-goals:
  不改变前四项的IR/API语义，不新增shape特判、test-only production入口、numeric policy或第二套fixture协议。
- Done criteria:
  四个owner各自的覆盖矩阵逐项实际执行；大规模整除/非整除、rank>=3、all-16 Tile及相关结构类别的exact断言fresh通过，
  完整unit和IR/source organization检查通过，设计复读确认没有用小oracle或单个success代签。
```

从`exact-demand-boundary`开始，每个work item在转为`doing`且写代码前，必须在自己的设计owner小节和Gate中列出与本项语义
直接相关的覆盖矩阵：整除/非整除、单轴/多轴、结构语义类别、正负/typed failure、需要断言的exact输出以及直接下游witness。
不能只引用本计划的全局shape规则，也不能只写“相关unit通过”。如果某一维不适用，要在该Gate中说明理由；完成复读时逐项
核对矩阵实际执行，单个case成功不能关闭work item。

`attention-normalization`先形成normalized attention roots，`attention-spatial-integration`再向spatial owner提供K1/K2与FA/FD
constraints，`canonical-spatial-assignment`才签发assignment；其已提交边界先经`foundational-coverage-matrix`补齐证据，随后
`exact-demand-boundary`与`attention-demand-integration`关闭demand和
coupled merge。任何实现不得把尚未归一化的
matmul/softmax外形交给physical owner猜测，也不得把FA/FD做成spatial schema字段。

canonical work items各自产生一个下游真实消费的typed artifact；`deterministic-baseline-closure`只在这些artifact齐备后编排
baseline。它排在`spatial-domain`与search work items之前只是当前施工顺序，不建立policy依赖；search planning不接收baseline
plan、executable、controller或statistics。`attention-production-closure`只汇合两条policy的纵向证据，不让policy互相调用。

Q53达到`board-ready`后不是再排一条假线性尾巴：Q57 resident execution、Q61 whole-program scale和Q48 semantic
superoptimization是三个独立分支，分别服从自己的前置；Q53真实板端matched gate继续串行占用设备并决定Q53何时`done`。

### Artifact producer / consumer 依赖审计

施工顺序以typed artifact DAG为准，不以任务编号或章节顺序为准：

| Artifact / checkpoint | 唯一producer | 必须消费 | 首个真实consumer | 失效 / re-entry |
| --- | --- | --- | --- | --- |
| `SpatialPlan/SpatialAssignment` schema、close与validator | spatial-plan-schema | typed spatial fields、structured iterators、topology | canonical-spatial-assignment、spatial-domain | schema原位演进；不携canonical选择或FA/FD field |
| fixed attention semantic facts | attention-normalization | normalized TensorProgram、attention op/interface、FA/FD classifier | attention-spatial-integration、exact-demand-boundary、attention-demand-integration | source normalization改变后重建；不进入candidate state |
| attention spatial constraints | attention-spatial-integration | spatial schema、fixed attention facts、K1/K2 roles | canonical-spatial-assignment、spatial-domain | semantic mode或iterator relation改变后重建；不进入candidate state |
| canonical closed `SpatialAssignment` | canonical-spatial-assignment | spatial-plan-schema、attention spatial constraints、topology | exact-demand-boundary、canonical chain | source/topology改变后重建 |
| attention-ready `ExactDemandProof` | exact-demand-boundary + attention-demand-integration | closed assignment、current structured IR、coupled attention facts | canonical-root-work、spatial-domain、layout/movement/feasibility | normalized semantic root或spatial改变后重算 |
| canonical plan components | canonical-root-work至canonical-schedule | closed assignment、attention-ready demand、baseline semantics | attention-work-projection、canonical-feasibility-proof、deterministic-baseline-closure | canonical coordinate改变后重建；无IR |
| `AttentionWorkDescription` / resource projections | attention-work-projection | fixed semantic facts及canonical plan prefix | canonical-feasibility-proof、attention-selected-decomposition、domain extensions | observed plan choice改变后重算；不进入state |
| baseline `FullFeasibilityProof` | canonical-feasibility-proof | canonical closed plan及attention resource descriptions | attention-selected-decomposition、deterministic-baseline-closure | ExactRejection推进coordinate；actual failure不re-entry |
| prepared selected attention decomposition | attention-selected-decomposition | complete canonical plan、work/resource IDs、F proof | baseline winner commit、unified search commit | 只属于selected transaction；不进入planning state/cache |
| full spatial successors | spatial-domain | fixed semantic facts、spatial schema、exact-demand query | search-control-foundation、root-work-domain | spatial改变使全部下游失效 |
| `RootRegionWork` alternatives | root-work-domain | spatial assignment、exact demand、canonical-root-work | region-execution-domain、selected leaf emitter | spatial/root改变后派生重算 |
| `RegionPlan` / execution/use bindings | region-execution-domain | root work、exact relation、canonical-region-plan | temporal-domain、Core Region state | region改变使全部下游失效 |
| `TemporalPlan` alternatives | temporal-domain | execution scopes、canonical-temporal-plan | partial-feasibility、layout-domain、Core Temporal state | temporal改变使全部下游失效 |
| partial feasibility facts | partial-feasibility | exact demand至temporal prefix | layout-domain、Core transition | observed component改变后失效；不写state |
| `RepresentationPlan` alternatives | layout-domain | partial proof、canonical-representation-plan | movement-domain、Core Representation state | representation改变使全部下游失效 |
| `MovementPlan` / communication actions | movement-domain | boundaries、physical versions、topology、canonical-movement-plan | storage-domain、event/full feasibility、Core Movement state | movement改变使storage/schedule/feasibility失效 |
| Serialized identity + initial storage alternatives | serialized-execution + storage-domain | temporal/movement occurrences、versions | event-resource-foundation、execution-structure-domain | structure choice使initial storage/event facts失效 |
| foundation `EventGraph` | event-resource-foundation | fixed semantic facts、storage prefix、Q63、target resources | execution-structure-domain | structure改变后丢弃 |
| `ExecutionStructurePlan` | execution-structure-domain | EventGraph、temporal/movement/storage eligibility | structure-specific-storage | 必须re-enter storage，禁止直达schedule |
| structure-specific `BufferPlan` | structure-specific-storage | fixed structure occurrences/live distance | schedule-domain | structure sibling/stage/distance改变后重闭 |
| `ClosedSchedulePlan` | schedule-domain | fixed structure/storage、event/resource facts | full-feasibility、Core Schedule state | structure/storage/worker/order改变后重闭 |
| full search feasibility proof | full-feasibility | fixed semantic facts与complete physical plan | search-control-closure | 只derived；不存offset |
| search controller | search-control-foundation + search-control-closure | 已实现的真实domain APIs及full proof | explicit public `search`、unified-search-closure | missing axis只在foundation期间typed incomplete |
| selected actual Card subtree | unified-search-closure | complete plan、attention decomposition及prepared builders | Q50.0 | failure终止compile，不返回planning |

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

这个边界不得枚举候选、修改选择、在lowering中retile/spill/rebuffer，也不得把失败降级成performance Unknown。Q49.P与Q51
只在各自最终plan确定后调用它一次；定向测试要证明同一CardModule得到相同accepted digest或相同rejection，且没有第二条兼容
编译路径。production full proof后再出现resource rejection是planning/actual parity bug，不能驱动baseline fallback或search重选。

实现结论：actual compile/verification seam只接收owned、已选择的CardModule和typed buffering assignment，返回
accepted、proven exact rejection或indeterminate三态诊断；baseline与search的**唯一selected materialization**调用该入口。Tile
memory planning保留SPM failure kind，供direct tests与parity诊断精确分类；production caller不得把任何Q50.0 failure当作正常
candidate pruning或functional legalization控制流。

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
   output也不被任何下游消费；该shadow plan可让合法executable因旧cost失败而失败，必须从baseline路径删除。
4. common compile seam在每次trial/compile中无条件把全部Tile dataflow IR打印成字符串，即使production caller最终丢弃trace；
   IR inspection必须是显式请求且只在最终accepted output上执行，不能成为baseline或candidate admission的固定成本。

同一review还确认baseline代码、100字段的search statistics和trace-bearing synthesis result仍共同定义在旧search
monolith/header中。Q49.P完成前必须把baseline的窄assignment、typed outcome和必要work counts移到不依赖旧search对象的current
owner；Q51.Core随后删除旧controller时不能再次迁移或适配baseline。

### Current实现向新baseline合同的融合方案（含`2f2e8de0`代码复核）

2026-08-17对`2f2e8de0`及current HEAD的Q49.P生产调用链逐项复核；该提交之后没有Q49.P生产源码修正。这里不建立
“DS修复线”和“新baseline线”两套工作：current代码只作为迁移输入，每项能力必须原位吸收进上文定义的single-coordinate、
single-root deterministic baseline，或从baseline调用闭包删除。融合取舍如下：

| current资产 | 融合到新baseline的方式 | 终态owner |
|---|---|---|
| typed region/function SPM probe与`RequiresFunctionScope` routing | 只迁移其中storage/lifetime/resource problem构造和typed certificate能力到Q50.F；删除probe materialization、scope escalation与Q50.0-per-trial控制。actual Q50.0只验证最终plan parity，不再提供baseline下一步。 | Q50.F pure feasibility + Q50.0 winner-only parity；不得恢复discard-and-rebuild。 |
| `StorageRootMemo` | 原位保留为同IR epoch的query-local派生缓存；不进入assignment、identity或跨mutation cache。 | `StructuredBufferRelations`局部query，P2。 |
| 无Card/Tile shell的per-Tile scoped入口 | `lowerRootShards`/`lowerTileEntries`只为discarded probe服务，退出baseline及current公共API。保留一次read-only source/target与root/support关系分析；actual CardModule apply按resolved root和Tile shard直接构造最终region，不为取得verdict生成另一份临时IR。 | P3/P7共享分析与一次性actual materialization；未来Q50.C复用apply。 |
| post-hoc `splitStructuredRootBoundaries`和DDR spill/reload实现 | 保留其中typed DDR boundary、SSA/resource移动和transaction proof，迁入按root直接构造region的materializer；baseline终态不先group多root再靠repair split。current splitter只作能力donor，完成后退出baseline调用链。 | P4 singleton-region apply与跨root canonical DDR carrier。 |
| lazy exact-demand option-pair query | Q50.A exact single-pair query保留；option domain、compatibility map、propagation和recursive solve整体从baseline删除。未来Q50.B在new Core上用自己的惰性domain重建，不继承旧helper。 | P6 single-coordinate legality；Q50.B future search mechanism。 |
| SPM failure converter与raw certificate | 将raw demand/interference事实迁成Q50.F plan-level typed certificate；actual memory planner只保留独立重算与parity诊断，有无relations不改变certificate事实。 | Q50.F exact rejection；Q50.0 actual parity。 |
| reduction temporal materialization | 接入baseline的通用legal-breakpoint/workset推导，用于确定性plan-level capacity legalization；不把旧search proposal/test带入baseline。reduction spatial factor、partial ownership和merge仍只归Q50.B。 | Q49.P canonical plan builder与Q50.E共享机制；Q50.F判断resource；Q50.B spatial。 |
| search statistics、`allocationFeedbackTransitions`、shadow schedule和默认IR trace | 不迁移；baseline optional work counts、move-only accepted result与optional inspection另立current owner后删除这些依赖。 | P6/P7；旧search对象随后由Q51.Core删除。 |

因此融合后的控制流是：先对每个root直接构造唯一canonical spatial coordinate与完整temporal vector；这个coordinate可以
覆盖多个甚至全部Tiles。source session只建立immutable root/support/relation事实，Q50.F从当前coordinate及canonical下游事实构造
typed resource problem。ExactRejection才沿direct witness推进下一temporal coordinate；FullFeasibilityProof关闭最终plan。随后只为
该plan构造一次actual CardModule并交给Q50.0，accepted owner直接成为baseline结果；Q50.0 rejection是parity/implementation bug，
不是下一coordinate。memo、DDR boundary和temporal materialization不得由旧option/CSP controller、discarded probe、
whole-Tile trial或search result/statistics拥有。

融合时必须修复的current阻塞项如下；“单测能通过”或diagnostic显示零不能替代这些调用链和current-IR合同：

| 等级 | current代码事实 | 影响与收口位置 |
|---|---|---|
| correctness / policy | `derivePolicyFreeBaseline`仍调用`deriveStructuredDAGNodePlacementOptions`，随后由`deriveCanonicalBaselinePlacements`建立每node option domain、反复propagate并递归`solve`；lazy pair query只延迟Presburger调用，没有消除axis×rectangle域或CSP。最终diagnostic却固定打印`placement_enumeration=0`。 | baseline仍是search，且可观察work counts与真实work矛盾；LLaMA慢case的全option/CSP与generic Presburger双重热路径仍在。由P6删除调用闭包并改成真实窄work counts。 |
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
  每个closed coordinate只调用Q50.A/F及必要的pure typed query；只有Q50.F的direct typed exact witness允许沿有限canonical
  temporal fallback前进，第一个FullFeasibilityProof关闭最终plan。该plan只构造一个actual CardModule并调用一次Q50.0，accepted
  executable直接保留并下传。该合法化不评分或比较性能，但必须覆盖声明支持的baseline域，不能因最大初始tile/placement
  不合法或没有search candidate而停止。
- Output IR / files:
  final plan的完整baseline CardModule经Q50.0消费后形成CardExecutable；Q59 transaction随后提交
  verified ExecutablePackage。rejected coordinate没有IR，只留下query-local typed witness；普通编译不生成printed-IR
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
  exact rejection携带来自同一typed resource problem的direct causal witness；planning CardModule/Q50.0均为零，final plan的
  CardModule materialization与CardExecutable compilation各一次且accepted owner不重建；同一coordinate的全部Tile root execution domain由一次
  grouped exact-demand query处理，relation按operation/result/operand只建立一次；structured producer形成停止边界，final region
  只一次性物化typed recipe证明需要的operation和carrier endpoint，16个独立Tile构造bounded并发并按Tile ID稳定归并；accepted后不构造`StaticSchedulePlan`、duration estimate或其它不被output消费的shadow result，
  production调用不打印/保存Tile IR trace；初始完整tile超SPM的正例由plan query重新推导完整workset并确定性缩到合法tile后通过package/no-card，
  reduction轴可沿typed tiling contract缩到最小合法vector；原「最小合法tile超限」fixture未进入完整合法coordinate，
  typed capacity/unsupported terminal保留为fail-closed防御出口；baseline使用独立窄work accounting，不持有或清零search统计来
  证明隔离；fresh
  source-to-package、oracle/no-card、digest、计数和结构正负测试通过。
```

`none`不得构造Q51 state、candidate set、candidate family或全图performance Cartesian组合，也不得通过调用search-oriented
domain/ranking evaluator后只取第一个结果来伪装canonical construction。它只复用typed iterator/topology事实、单coordinate
legality/materialization机制和Q50.A logical demand/coverage query，不复用“生成全部合法placement options”的domain API；对baseline
合同内的mandatory coordinates执行有限、完整、确定且不被beam/cap/time budget截断的
feasibility resolution；按semantic全序逐个关闭coordinate并只运行pure typed queries，
第一个FullFeasibilityProof即完成planning。不能复制第二套placement语义；trial只是typed coordinate，不构造actual IR。closed
final plan只构造一份完整CardModule并由Q50.0消费；accepted owner继续下传，Q50.0 failure终止compile。不得在任一coordinate上
构造root/Tile probe或discarded CardModule。

controller交给共同materializer的是窄immutable resolved baseline assignment。它是上述feasibility resolution的输出而非入口
前置条件，只含per-root placement、显式singleton region boundary、完整temporal vector和已经确定的canonical
representation/movement/buffer/order/completion事实，不含evaluation/score、stable ordinal、transition/failure history或
controller flags。materializer只apply这些已选事实，不得根据同Tile root集合自行group，也不得补search default。

### 功能合法化与SPM收缩

baseline以每个root的完整local iterator extent作为第一个temporal trial，并从structured iterator、indexing map、tail、target
vector/alignment和最小合法粒度形成有限breakpoint lattice；parallel与reduction iterator都由同一typed tiling contract给出可用
尺寸。每个trial必须按
当前tile重新推导全部operand slice、stride/dilation halo、result/init/accumulator、temporary、materializing copy、movement
staging、alignment/bank和plan-level lifetime，并由Q50.F的typed resource problem判断；不能只缩output shape、沿用上一trial的
workset/lifetime，也不能为判断trial建立IR。

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
exact-demand、carrier coverage与Q50.F resource验证，不构造CardModule。第一个FullFeasibilityProof关闭plan后才构造一次actual
CardModule交给Q50.0。它不计算score、
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

Q49.P直接复用Q50.F从A--K typed facts构造的storage/lifetime/resource problems。capacity witness可以是冲突集合，但其中
all-and-only storage/lifetime owner必须由semantic IDs关联到root result、operand demand和temporal assignment；unsupported witness
必须命名无法表达的typed relation。不按type/shape、位置字符串、region内“可能相关”的其它root、pointer identity或diagnostic
字符串猜测。只有Q50.F返回的proven exact rejection允许baseline controller沿确定性fallback lattice前进；heuristic no-fit、
resource exhaustion或indeterminate立即传播。最终Q50.0只重建actual problem并验证parity，不向controller返回下一步。

`none`与`search`共享immutable structured/relation/target facts、policy-free single-root TileRegion materializer和
actual Card/Tile/Instr、completion、SPM/DDR、verification、package机制；不共享search state/candidate/evaluator、group
boundary、proposal order、score、winner或结果owner。Q49.P只服务`none`；Q51从TensorProgram独立建立search candidate，不接收或
重建baseline。Q49.P施工时把resolved baseline assignment的single-root apply落在稳定
PhysicalDataflow/Conversion边界；Q50.B/C随后扩展完整search domain与single-root mechanism。Q50.F只做pure analysis；
所有baseline coordinate都不构造IR，只有通过full proof的complete assignment构造actual CardModule并由Q50.0一次消费。
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

## Q50.A：Placement-Demand Boundary Repair

“给定placement”只表示调用者交来一份closed `SpatialAssignment`，不表示baseline或search入口已经拥有selected winner。
spatial-plan-schema提供representation/close，attention-normalization固定semantic roots，canonical-spatial-assignment再构造唯一
canonical value；spatial-domain生成search transitions。exact-demand-boundary只从immutable normalized structured IR和该assignment
派生exact dependency demand、final result availability及reduction merge requirements，不生成placement、不选择winner，也不
物化physical carrier。

```text
Pipeline position:
- Upstream IR / input:
  attention-normalization归一后的verified card-local structured TensorProgram、current SSA/structured DAG，以及
  canonical-spatial-assignment关闭的
  `SpatialAssignment`；assignment只含per-node exact execution shards、Tile embedding和已由typed reduction semantics允许的
  per-output-piece merge placement，尚未选择layout、encoding、movement、route、buffer或schedule。
- Current stage responsibility:
  一次构造operand-level pure tensor SSA relation graph；从consumer execution shard反向传播exact demand，在structured result、
  program input和constant处形成boundary；派生merge后的final result owners和per-output-piece reduction requirements，并区分
  satisfied、unsupported semantics、indeterminate resource exhaustion和compiler contract error。
- Output IR / files:
  不修改IR、不产生文件；输出仅在当前immutable planning session有效的`ExactDemandProof`，包含per-operand/per-destination
  boundary demand、final-owner intersections、operand reconstruction和reduction merge requirements。
- Downstream consumer:
  deterministic-baseline-closure消费同一proof；spatial-domain把proof作为assignment的derived data/cost input而非普通
  placement rejection；Q50.G/H representation/movement与Q50.J schedule消费final-owner和merge requirements；Q50.0仍是完整
  CardExecutable准入边界。
- User-level driver / named pipeline:
  wafer-compile source-to-package pipeline中的typed `none`与`search`共同physical-dataflow planning路径；不提供独立public pass、
  selector或磁盘格式。
- Explicit non-goals:
  不枚举、排序或选择placement；不决定TileRegion/fusion、temporal tile、layout/encoding、local/remote/DDR/peer action、route、
  bytes、fragment、buffer、send/recv、schedule或cost；不把physical carrier能力当logical legality；不保存跨IR mutation的side table。
- Done criteria:
  `SpatialAssignment`、`ExactDemandProof`、per-output-piece merge和typed failure API替换旧trial/role/epoch/merge-bool接口；唯一
  operand-level反向传播覆盖reduction、broadcast、affine window及stride/dilation、strided slice/view、multi-piece、multi-result、
  DPS init、program input和multi-operand pure tensor graph；production supported relation不进入无界generic Presburger proof；
  baseline、physical consumers、analysis invalidation、fresh source-to-package witness和donor能力矩阵闭合。ordinary exact-demand
  边界先完成后，attention-demand-integration沿同一typed extension seam补齐Q/K/V/mask demand及coupled contribution/merge；
  两项共同形成attention-ready ExactDemandProof，不建立第二套query或side table。
```

### A-1 专项调研与现状审计

本子任务分别核对了MLIR Linalg/Presburger合同、OpenXLA indexing analysis、current query/consumer/tests以及该实现形成过程中的
donor提交。调研结论不是照搬其它编译器的数据结构，而是用来判断本项目哪些能力是真正的structured semantics，哪些只是旧控制链
留下的实现偶然性：

- MLIR Linalg把“iteration-space子集读写哪个data子集”明确交给indexing maps和structured interfaces；因此Q50.A从完整consumer
  iteration domain经typed map求operand demand是正确层级，从operand编号、shape相等或op名字恢复需求不是可接受替代。
- OpenXLA的grouped output-to-input indexing以use-before-def顺序遍历producer graph，为同一producer保留一组组合后的indexing maps，
  每步compose、simplify并显式保留unknown/known-empty区别。这验证了“一次operand-rooted反向worklist、在多路径处union、在semantic
  producer处停止”的结构；它没有为每条edge先做一遍chain walk、再做第二遍operand reconstruction。
- MLIR Presburger relation能够表达union/intersection/subtraction和存在量化，但接口可表达不等于任意大关系操作都适合作为production
  hot path。current rectangular fast path及finite-piece constructors属于应保留能力；无法在操作前给出work上界的generic equality、
  subset或subtraction只能作明确有界的穷举oracle或返回typed resource exhaustion。
- 两个成熟实现都把unsupported indexing作为明确结果，不用默认identity、bounding box或空集合继续；known-empty则是有效的精确结果。
  因此current exact-empty branch测试必须迁移，`unsupported`与empty不得合并。

current实现的真实结构和偏差如下；“保留”表示能力需迁入终态owner，不表示保留现有API或控制流：

| Current事实 | 审计结论 | 终态处理 |
| --- | --- | --- |
| `IndexRelation`已有affine/permutation、reduction/broadcast、reshape、slice、insert/pad、strided和有限rectangle-piece能力 | 核心语义能力，已有direct/permuted/reduction/broadcast/strided/pad/decode witnesses | 保留constructor及extensional tests，改为construction-aware normal form；generic Presburger只作bounded fallback/oracle |
| `query(edge, trial)`逐edge解析producer-to-consumer chain、求relation image、ownership coverage和per-destination facts | relation与coverage能力需保留；edge是输出切片，不应是主遍历单位 | 并入唯一operand-rooted worklist；从同一传播结果投影dependency facts，不再维护edge-local chain/cache |
| `queryOperand`先调用每条incoming edge的`query`，随后又从consumer operand递归穿过support graph并比较是否到达各boundary | 对同一SSA关系执行两遍工作，而且per-edge resolver把同producer多路径标为ambiguous，和operand walk的多路径union能力冲突 | 删除第一遍edge chain walk；一个worklist同时形成reconstruction、structured boundaries、program input/constant及exact-empty结果 |
| `LogicalShardTrial`同时复制execution shards、result ownership、partial role、merge Tile及borrow epoch | execution与派生ownership形成重复事实源；node-wide merge不能表达M/N/K mixed及multi-result group | 输入收敛为`SpatialAssignment`；Q50.A派生final availability与per-output-piece merge requirements |
| partial contribution存进普通result ownership并由`mergeObligation + reductionMergeTile`补充解释 | partial value并不是可供下游选择的完整result owner | contribution只进入`ReductionMergeRequirement`；merge完成后再生成唯一/replicated final owners |
| 每次`query`/`queryOperand`比较整个function `OperationFingerPrint`，同时用手工`IREpoch`绑定trial | 每次whole-function walk是重复开销，两套失效协议也不属于MLIR analysis lifetime | 迁为func-scoped analysis/session invalidation；query-local pointer只在该IR epoch内使用，不进入稳定key |
| rectangular path在16 Tile大矩形上有closed-form coverage，非矩形继续调用generic union/subtract/equality | fast path是重要scalability donor；generic path缺少统一的pre-operation work bound | 保留矩形/有限piece证明，所有可能扩张的构造先估算piece/constraint work；超限返回resource exhaustion，不启动无界证明 |
| `ScopedCompileTimingSpan`散布在query内部 | optional timing有调试价值，但不是语义、cache或默认输出 | instrumentation由外层显式启用；关闭时不产生日志/持久统计，算法只保留资源控制必需的work counter |

donor测试资产不能按旧结论整体删除。需要迁移的能力至少包括：direct与permuted edge、one-to-many reduction、broadcast、显式DPS init、
strided support chain、pad、insert overwrite exact-empty、multi-operand support graph、multi-result per-result ownership、replication、partial
contributions、per-destination remainder、16-Tile rectangle coverage、deterministic proof及carrier-failure metamorphic isolation。需要改写而非
原样保留的旧witness包括：ownership hole/overlap作为普通placement rejection、ambiguous single-chain unsupported、borrow-token与
fingerprint测试、node-wide merge，以及init/support edge“不产生spatial action”替代完整logical requirement的断言。

上述审计确认当前约2600行query并非都应删除：relation constructors、typed exact/empty/unknown区分、multi-producer reconstruction和
coverage witnesses是能力donor；`LogicalShardTrial`聚合、两遍传播、manual invalidation、node-wide merge及无界generic fallback才是要
替换的实现结构。后续A-2到A-4分别闭合算法、analysis/failure API和迁移验证，不能用本审计本身把Q50.A标为完成。

### 重审后的唯一数据合同

Q50.A 不再把一次检查输入称为含糊的 `trial`，也不接收一份同时复制 execution、result ownership 和 reduction
状态的聚合结构。唯一输入是 Q50.B 产生的一份只读 `SpatialAssignment`；Q50.A 从该 assignment 和 current
structured IR 派生其余事实：

这里要求的producer是canonical-spatial-assignment，不是spatial-plan-schema或spatial-domain：schema先定义plan/assignment及
structural close，attention-normalization固定roots后canonical work item再签发assignment；exact-demand-boundary随后实现query并由
attention-demand-integration关闭coupled relation。只有这些输入完成后，spatial-domain才枚举所有plans并逐个调用query。实现不得为先施工而
临时保留`LogicalShardTrial` adapter，也不得自行构造默认assignment。

```text
SpatialAssignment
  nodes: NodeExecutionPartition[]

NodeExecutionPartition
  root: SemanticRootKey
  shards: ExecutionShard[]
  reductionGroups: ReductionGroupPlacement[]

ExecutionShard
  shard: LogicalShardId
  tile: TileId
  iterationDomain: ExactIndexSet

ReductionGroupPlacement
  results: ReductionResultSlice[]
  mergeTile: TileId
```

`iterationDomain` 是该 shard 在 op **完整 iterator space** 中的集合，parallel 与 reduction iterator 均在其中；
assignment 必须已经证明各 shard 两两不交且并集等于完整 iteration domain。`ReductionGroupPlacement` 只在
Q50.B 已选择空间 reduction 时存在。它按 result 的 output-domain piece 分组，因此 M/N 与 K 同时切分时可以有多个
merge group；禁止再用一个 node-wide `reductionMergeTile` 表示所有输出。

Q50.A 派生并返回下面的 named typed result：

```text
ExactDemandProof
  dependencyDemands: DependencyDemand[]
  reductionMerges: ReductionMergeRequirement[]

DependencyDemand
  source: StructuredResult | ProgramInput | Constant
  destination: ConsumerOperand
  perDestination: DestinationDemand[]

DestinationDemand
  destinationShard: LogicalShardId
  requiredSourceDomain: ExactIndexSet
  eligibleFinalOwners: OwnerIntersection[]
  operandReconstruction: OperandReconstruction

ReductionMergeRequirement
  results: ReductionResultSlice[]
  contributions: ReductionContribution[]
  mergeTile: TileId
  initialization: ReductionInitialization
  algebra: ReductionAlgebra
```

这里有两种不同且不能混用的事实：

- `eligibleFinalOwners`只描述**完成 merge 后**可向下游提供 producer result 的 owner。unique partition 要求这些
  owner 的domain两两不交；explicit replication保留全部等价owner供Q50.H选择source。下游dependency永远不能把
  pre-merge partial value误当作完整result replica。
- `ReductionMergeRequirement`描述同一output-domain piece上全部必需partial contribution、merge位置、原init只应用
  一次还是每个partial使用identity，以及所用merge operation。它是node内部必须完成的逻辑义务，随后由
  Q50.H选择movement、Q50.J选择event/order并由winner materializer显式产生merge。

因此 current `LogicalTileBinding::role`、result binding 中的
`PartialReductionContribution`、node-wide `reductionMergeTile`和`ExactDemandResult::mergeObligation`不能作为终态接口。
实施时迁移为上面的final-result availability与per-output-piece reduction requirement；旧字段及其所有producer/consumer
同批删除，不保留wrapper或双合同。

### A-2 专项调研与唯一反向传播算法

pinned MLIR的`TilingInterface`把iteration domain、operand/result tile到iteration tile的映射和actual tiled implementation分开，并
明确这些接口只提供变换mechanism、不判断profitability。current Linalg external model对operand/result tile的逆映射仍要求
projected permutation；broadcast、window、reduction和一般affine access不能靠这个便利API反推。Q50.A因此使用
`getIterationDomain`、DPS/indexing-map语义及窄exact transfer interface构造**正向的iteration-to-value relation**，再对已选
iteration set求image；不能把便利API失败误报为source semantic unsupported。

MLIR DataFlowSolver的稳定模式是“state改变才把dependents放回worklist”；OpenXLA grouped indexing则按use-before-def顺序组合
producer maps并在同一producer处合并多条路径。Q50.A的pure tensor closure已经由SSA与effect检查证明为有限DAG，因此不需要通用
CFG fixed point，也不应把assignment-specific state塞进长期MLIR analysis：先一次构造typed relation DAG，然后以所有consumer
operand/destination为seed做一次逆拓扑传播即可。

主查询不再按consumer、operand、destination Tile分别walk support graph。它先建立所有seed，再在一个query-local state中保留
`DemandKey -> ExactIndexSet`：

```text
DemandKey = (consumer node, consumer operand, destination logical shard)

ValueDemand
  key: DemandKey
  state: Absent | Exact(empty-or-nonempty set)

ExactDemandProof
  dependencyDemands
  programInputDemands
  constantRequirements
  operandReconstructions
  reductionMerges
```

`Absent`表示从该consumer seed到当前value根本没有SSA路径；`Exact(empty)`表示路径存在、但该路径对所选destination确实不读任何
元素。二者不可合并，否则insert overwrite/pad boundary等exact-empty proof又会退化成“没有发现路径”。

```text
deriveExactDemand(structuredIR, spatialAssignment):
  verifyExecutionPartitions(spatialAssignment)
  relationDAG = getOrCreateFunctionRelationAnalysis(structuredIR)
  finalOwners, reductionMerges = deriveResultAvailability(spatialAssignment)

  state = map<Value, map<DemandKey, ExactIndexSet>>()
  for consumer in stable structured-node order:
    for each payload-read tensor operand in operand-number order:
      relation = relationDAG.iterationToOperand(consumer, operand)
      for destination shard in stable logical-shard order:
        key = (consumer, operand, destination.shard)
        seed = exactImage(relation, destination.iterationDomain)
        state[operand][key] = Exact(seed)       // seed may be empty

  for value in relationDAG.reverseTopologicalOrder():
    entries = state[value]
    if entries is empty: continue
    if value is a structured result:
      recordBoundary(value, entries, finalOwners[value])
      continue
    if value is a program input:
      recordProgramInputBoundary(value, entries)
      continue
    if value is a constant:
      recordConstantRequirement(value, entries)
      continue

    transfer = relationDAG.transferFor(value.definingResult)
    require transfer is pure and exact
    for data-carrying operand in transfer.stableOperandOrder:
      relation = transfer.resultToOperand(operand)
      for each distinct exact-set class in entries:
        mapped = exactImage(relation, class.set)
        for key in class.keys:
          unionPresent(state[operand][key], mapped)
          appendReconstructionStep(key, value, operand, mapped)

  for every recorded structured boundary:
    intersections = intersectWithFinalOwners(boundary.set, finalOwners)
    proveAllAndOnlyCoverage(boundary.set, intersections)
  proveEveryReductionGroup(reductionMerges)
  return buildStableProof(state, boundaries, reconstruction, reductionMerges)
```

`relationDAG`只包含current function中能由standard interface或Wafer exact transfer interface解释的value/result/operand关系；
structured op result、program input和constant是停止边界。建图时即拒绝跨effect、非法region crossing和cycle，不在传播时靠名字猜测。
一个multi-result support op按具体result选择transfer；一个multi-operand op为所有data-carrying operands建边。同一producer通过多条
路径到达时，`unionPresent`对同一个`DemandKey`取exact union，因此没有“ambiguous chain”分支。

所有destination同时传播不表示把它们的demand混在一起：`DemandKey`一直保留consumer/operand/shard provenance，只有normal form
extensionally相同的set临时归为一个exact-set class以共享一次image。regular 16-Tile partition通常可用同一relation对一组boxes做
bulk affine/interval arithmetic；不同offset、tail或support path仍分别产生正确集合。最坏情况下每个destination都不同，集合运算仍
需线性执行，这是问题本身的信息量；被消除的是16次IR walk、16次relation derivation、per-edge第二遍传播和相同set的重复image。

逆拓扑顺序保证一个value在所有下游路径的contribution都union后才向其operands传播；不需要反复扫描到fixed point。若future
TensorProgram允许RegionBranch/call/loop-carried tensor，必须由相应standard interface把它显式展开成有限relation graph，或另立
interprocedural/循环analysis边界；本算法不能静默把有环IR当成当前单block DAG。

正确性按reverse-topological induction证明。seed是consumer iteration shard经operand indexing relation的精确image；假设某value
state包含从所有已处理consumer paths到该value的all-and-only demand，则每条exact result-to-operand relation的image给出对应operand
的all-and-only demand，而union保持所有路径且不重复改变集合语义。到达boundary时，state因此正好是所有consumer paths的并集；
再与已证明的final owners求交即可得到完整dependency proof。`Exact(empty)`由exact image直接产生并沿拓扑保留，故empty proof也满足
同一归纳。

终止性来自三个有限量：relation DAG的value/edge数有限，reverse-topological pass每条edge访问一次，每个value上的`DemandKey`集合
由有限seed给定且只做一次归并。normal-form piece expansion在执行前受work limit约束；超限返回typed resource exhaustion，而不是
重新入队或降为无界generic solver。设relation graph为`V/E`、seed数为`L`、edge上不同exact-set class数为`U_e`、每个normal form
最多`P` pieces，则结构工作为`O(V+E)`，set work为`O(sum_e U_e * image(P))`，owner coverage为
`O(L * O * intersect(P))`。最坏`U_e=L`，但不再额外乘整棵IR扫描或per-edge chain walk；tensor元素数量不进入复杂度。

输入和输出均不含layout、encoding、bytes、route、buffer、event、worker、cost、candidate编号或统计。immutable planning session
可按完整`SpatialAssignment` semantic key memoize `ExactDemandProof`，使同一spatial state扩展其它坐标时不重复计算；memo和proof都不
跨session、pass或首次IR mutation。winner直接消费该session中已经验证的proof，不在commit前以“复核”为名再跑一次同样query。
首次mutation前销毁/关闭query view；之后不能再访问relation analysis或任何带source pointer的proof。

### Exact relation表示与有界算法

pinned MLIR 的`PresburgerRelation`适合作为精确整数集合语义，不自带本项目需要的编译work上界：union可以保留重叠
disjunct，relation composition会形成两侧disjunct的笛卡尔积，`isEqual`、`isSubsetOf`和`subtract`可能进入integer
simplex/lexicographic optimization。因而“操作完成后再检查variable/disjunct数量”不能防止一次调用已经耗时失控。

Q50.A继续以MLIR Presburger relation/set为语义核心，但`IndexRelation`和`ExactIndexSet`必须保留**构造时已证明的normal
form**，并在执行可能扩张的操作前估算work：

```text
RelationForm
  AffineMap              // projected permutation、broadcast、window affine map
  BoundedAffineImage     // bounded iterator box经affine map所得集合
  RowMajorReshape        // collapse/expand的row-major interval correspondence
  Piecewise              // pad/insert/slice/tail的有限互斥pieces
  GeneralPresburger      // 仅作有界穷举oracle或显式unsupported/indeterminate出口

SetForm
  BoxUnion
  StridedBoxUnion
  BoundedAffineImageUnion
  RowMajorIntervalUnion
  GeneralPresburger
```

normal form不是第二份语义或跨pass side table；它与Presburger对象同属一个query-local typed value，并说明该对象是通过哪个
exact constructor得到的。构造、composition、image、union和intersection只有在能证明新normal form时才传播它；否则降为
`GeneralPresburger`。production `none`与`search`对声明支持的source semantics必须始终留在前四类，不允许降级后无界调用
generic equality来碰运气。`GeneralPresburger`只用于有界independent oracle核对前四类结果，或在source真正超出current
typed relation语言时返回`unsupported`；若是piece/work增长超过统一compiler resource limit，则返回typed
`indeterminate(resource exhaustion)`，不能删placement。

每个可能扩张的操作在调用MLIR API前执行以下检查：

```text
compose(lhs, rhs):
  require lhs.rangeSpace == rhs.domainSpace
  predictedPieces = lhs.pieces * rhs.pieces
  reject-as-indeterminate if predictedPieces exceeds relationWorkLimit
  compose construction forms directly when a closed rule exists
  materialize the exact Presburger relation only after that check

image(relation, set):
  dispatch by (RelationForm, SetForm)
  projected/broadcast/window: affine arithmetic or bounded affine image
  slice/view: preserve strides; never replace with a bounding box
  reshape: map row-major intervals, splitting only at dimension boundaries
  piecewise: image each mutually exclusive piece, canonicalize and merge equal pieces
  reject-as-indeterminate before a predicted piece-product exceeds the limit

intersect/union:
  use interval/lattice arithmetic for compatible normal forms
  preserve a deterministic disjoint normal form
  never call generic subtraction/equality merely to rediscover constructor facts
```

`relationWorkLimit`是baseline与search共用的compiler safety limit，不是search candidate budget，也不改变合法域；达到上限表示
compiler本轮无法完成精确证明。默认编译路径不采集或打印统计。只有显式timing/profile选项启用时，query才通过可空observer
报告relation composition、piece count和work count；observer不进入返回值、cache key或控制流。

Linalg consumer优先从仓库pinned `mlir::linalg::LinalgOp`的`getIndexingMapsArray()`和
`getIteratorTypesArray()`读取indexing maps与iterator types，并使用`DestinationStyleOpInterface`和`TilingInterface`
读取operand、DPS与tile relation。pinned MLIR的`TilingInterface`描述tile materialization机制和result/operand tile映射，但明确不判断
profitability，也不能为所有pure tensor op返回任意exact set relation；因此它不能被当作Q50.A的完整relation接口。对标准
`ViewLike`/`Subset`等接口仍无法表达的multi-operand tensor transform，新增的唯一Wafer扩展必须是窄
`TensorIndexingOpInterface`：按具体result列出transform kind、data-carrying operands、source/destination role及static
offset/stride参数；analysis从这份source-owned description构造有界exact relation。`tensor.expand_shape`、`collapse_shape`、
`extract_slice`、`insert_slice`、`pad`和`cast`通过external model实现；query和winner reconstruction都消费同一typed
description，不再各自维护concrete-op `TypeSwitch`、operation-name matcher或第二次support-chain walk。

relation graph构造本身为`O(V+E)`次interface查询，加上每条edge一次normal-form relation construction；assignment-specific复杂度
使用上节`L/U_e/P/O`表达。所有piece乘积在执行前受统一work limit约束；算法不逐tensor元素枚举，也不随16个Tile重复构造relation
graph或扫描整棵function。只有真正不同的per-destination exact sets保留线性工作，避免用不正确的“所有Tile相同”假设换性能。

### Result availability与空间reduction算法

Q50.A只分析Q50.B已经构造完整的spatial assignment。若每个node的execution shards all-and-only覆盖原iteration
domain，则普通parallel partition无论producer与consumer使用相同、部分重叠还是完全不同的Tile集合，都不会因cross-op
demand而变成非法：producer最终会产生完整logical result，Q50.H总能在local、peer或DDR等后续physical alternatives中选择
一种显式redistribution。Q50.A负责精确给出redistribution所需集合，不用“ownership hole”替Q50.B或Q50.H拒绝坐标。

对每个structured result，availability按下面算法派生：

```text
deriveResultAvailability(node, executionShards, reductionPlacement):
  for result in node.results:
    outputImage[shard] = image(node.iterationToResult[result],
                               shard.iterationDomain)

  if no reduction iterator is spatially partitioned:
    prove outputImage pieces are disjoint and cover the complete result
    return one final owner (shard.tile, outputImage[shard]) per nonempty image

  algebra = queryReductionAlgebra(node)
  require algebra.allowsSpatialPartition and an actual typed partial builder
  group result images by their exact output-domain key
  require images from different groups are disjoint
  for each group:
    prove its contribution iteration domains partition the complete
          reduction fiber for that output domain
    require exactly one selected merge Tile from reductionPlacement
    emit ReductionMergeRequirement(group.results, all contributions,
                                   merge Tile, identity/init rule, algebra)
    expose only (merge Tile, group output domains) as final result owners
```

对current Linalg block partition，result indexing map把reduction iterator投影掉，因此同时切M/N与K时，具有相同parallel
coordinate而不同K coordinate的shards产生相同output image，并自然形成一个merge group；不同M/N coordinates的images
互不相交。算法不假设只有一个parallel轴、一个reduction轴或一个merge：`P_M * P_N`个output groups可各有自己的merge
Tile，每组包含`P_K0 * P_K1 * ...`个必要contributions。若future structured op的result images只部分重叠而不是equal-or-disjoint，
其op interface必须给出可验证的piece partition；Q50.A不会通过generic set subtraction自行猜出分组。

`queryReductionAlgebra`返回typed `ReductionAlgebra`，至少说明result components是independent还是coupled、partial identity、
原DPS init的唯一应用位置、实际merge operation由谁物化，以及planning所需的partial component maps/positions。仓库pinned
`PartialReductionOpInterface`只提供identity tensor、partial tile和merge三项IR-building mechanics；其pinned通用driver要求partial
init与source DPS result同构。普通Linalg由该standard interface和现有external model补充只读algebra；只有一个final result但拥有
Maximum/Sum/Accumulator三个internal states的attention，由Q50.S的`WaferCoupledReductionOpInterface`提供component maps、group和
init/final owner，并由selected Linalg/SCF builder提供actual partial mechanics。Q50.B只在read-only algebra与actual typed builder
共同完整表达selected result group时生成reduction factor大于1的assignment；
不能只支持single-result/single-init后把该限制伪装成通用reduction合同。

independent multi-result reduction可以形成多个result groups；coupled state必须由source op的同一个typed algebra和同一次merge
共同声明，不能逐result调用scalar matcher后假设独立。每个partial使用interface给出的identity，原DPS init在final merge中exactly
once消费；若某种algebra要求first-partial seeded init或严格顺序，则该规则显式进入`ReductionInitialization`，不由materializer
根据Tile编号猜测。

这里与Q50.E的temporal reduction要严格分开：同一Tile上保持原iterator顺序、用一个loop-carried accumulator依次执行K tiles，
不产生partial owners；Q50.E负责证明该遍历与tail。跨Tile并发K partition才使用本节的partial/merge
算法。两者都可以存在于一个完整candidate，但不能互相代签合法性。

通用正确性证明由四个不变量组成：

1. `execution partition`：每个原iteration point恰属于一个spatial shard；
2. `result availability`：未空间切reduction时每个result point恰有一个final owner或一组显式等价replicas；
3. `reduction completion`：空间切reduction时每个原iteration point恰贡献一次，每个output-domain group收齐完整reduction
   fiber，原init按typed rule恰消费一次，且merge后只暴露一个final owner；
4. `dependency exactness`：consumer operand seed relation与每个pure SSA transfer relation均exact，反向传播对多路径取union，
   因而每个structured/program boundary的required domain是all-and-only demand。

这四项按relation composition和SSA DAG拓扑归纳，不依赖模型名、shape常量、参数顺序、Tile数量或op名称。remainder只改变
`ExactIndexSet`边界；多parallel轴、多reduction轴、multi-result、fanin/fanout和multi-operand support graph不改变算法。

结果分类也随职责收窄：

- `satisfied`返回上述proof；
- source op/interface无法表达exact dependency relation时返回`unsupported semantics`，该结论与physical placement无关；
- normal-form或piece work超过统一limit时返回`indeterminate resource exhaustion`；
- Q50.B交来不覆盖iteration domain、缺merge group或重复contribution的assignment是`compiler contract
  error`，不能当普通candidate rejection继续搜索；
- physical descriptor、route、SPM或schedule失败属于Q50.G–K，不进入Q50.A结果。

因此终态Q50.A不再用`proven logical infeasible`作为普通Q50.B transition filter。旧ownership-hole测试保留为
`SpatialAssignment`/analysis contract负例，而不是“换一个placement也许成功”的search no-good；production Q50.B域必须从源头
不生成这种状态。

### Logical domain、dependency与ownership合同

Q50.A输入的是完整logical execution partition，不得再从`shardDimension + participant count`恢复balanced一维矩形。
`SpatialAssignment`显式包含all-iterator execution domains、parallel/reduction role、remainder/tail、logical shard-to-Tile
embedding和per-output-piece merge placement；final ownership与partial contribution都由Q50.A派生，不能同时作为一份可相互
矛盾的输入。Q50.B负责生成并验证assignment；Q50.A以同一API消费baseline canonical assignment、test assignment和后续
search assignment，使multi-axis、非整除、非矩形、非对称和非连通placement不被旧单轴接口截断。

每条structured dependency按current SSA use、producer result、consumer operand、DPS operand role和support op semantics形成
typed descriptor：

- 普通DPS/data input对consumer已选完整iteration domain应用operand indexing relation；
- reduction保留当前partial contribution domain与merge role，不能只从result shard推回完整K域，也不能
  把多个partial owner当作可互换replica；
- 显式structured Fill或其它DPS init producer仍是独立DAG root，其init/update dependency必须形成exact demand，不能因consumer
  typed lowering稍后会处理init就从计划中消失；
- 没有独立structured root的view/reshape/slice/pad等pure tensor input graph按typed semantics组合relation；多operand support graph
  对每个data-carrying predecessor分别保留dependency，不能假设只有unary chain，也不能把support名字当语义；
- multi-result、fanout和fanin按实际producer result/consumer operand分别建relation，不假设`result(0)`、单init或equal shape。

对每个consumer logical shard，query以其完整execution domain求relation image，再与producer的**final result owners**求交。
显式replica允许多个eligible owner时，Q50.A保留等价owners而不选择source；partial contributions只存在于独立
`ReductionMergeRequirement`中，不能出现在下游dependency owner集合。完整execution partition必然产生完整result；因此
cross-op demand不以ownership hole删除placement。malformed assignment、缺失merge group或缺失上游typed role是compiler contract
failure，不是某个placement的普通no-good。

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
dynamic、non-affine或缺少typed relation semantics可以返回`unsupported semantic relation`，但它作用于对应语义/alternative，
不是换一个physical placement即可消除的失败。

### A-3 专项调研：typed outcome、analysis lifetime与physical隔离

MLIR pass infrastructure把analysis定义为“由当前operation构造、只读、惰性缓存”的普通class；所有analysis默认在一个可能修改IR的
pass后失效，只有明确`markAnalysesPreserved`才继续存在。pinned `AnalysisMap`按analysis type缓存并在pass结束时依preserved set删除，
不会在同一个pass的每次rewriter mutation后自动刷新。因此本项目不能把删除`OperationFingerPrint`理解为“query对象可任意跨mutation
继续用”，而是必须同时收紧调用结构：

```text
pass-local consumer:
  relation = getAnalysis<StructuredRelationAnalysis>()
  result = runPolicyFreeQuery(relation.typedFacts(), localInputs)
  return before any IR mutation that invalidates relation

driver-owned planning entry:
  relation = buildStructuredRelationFacts(source)
  session = DemandPlanningSession(move(relation), immutable target facts)
  ... baseline or search performs pure queries without a PassManager analysis handle ...
  winner = closePlanningSession(session)     // session becomes unusable
  materializeWinner(winner)                  // first and only IR mutation phase
```

`StructuredRelationFacts`由policy-free builder从包含全部structured/support SSA的最窄current operation建立；现有表示下是对应
`func.func`，若后续有真正`IsolatedFromAbove`的TensorProgram container才可进一步下沉。它只包含从current IR和typed interface
semantics可重算的relation DAG、`SemanticRootKey`/value paths和normal-form constructors；构造失败保存在closed typed result中。
它不包含`SpatialAssignment`、target配置、candidate、cost、合法性结果或winner。

pass consumer可通过`StructuredRelationAnalysis`薄wrapper让AnalysisManager缓存同一builder结果，并按mutation明确
preserve/invalidate。driver-owned `DemandPlanningSession`直接拥有builder结果和query-local memo，不持有non-owning analysis引用，
也不在pass外自行构造`AnalysisManager`。两条入口共享builder和typed schema，不共享cache；任何IR mutation都会关闭session，
不能用`IREpoch`、fingerprint、revision integer或singleton/global cache建立第二套失效协议。

Q50.A public query返回一个named sum type，不返回`bool`、`FailureOr + string`或可被误解为candidate rejection的四态enum：

```text
ExactDemandOutcome =
    ExactDemandProof
  | UnsupportedDemandSemantics
  | DemandWorkLimitReached
  | InvalidSpatialAssignment
  | BrokenDemandContract

DemandFailureSite
  node: optional StructuredNodeId
  result: optional result number
  operand: optional operand number
  shard: optional LogicalShardId
  operation: RelationOperationKind
```

- `UnsupportedDemandSemantics`表示verified source op/interface没有current exact relation或reduction algebra；它可被产品入口报告为
  unsupported，但与某个placement无关。
- `DemandWorkLimitReached`携带发生在哪类normal-form操作、pre-operation predicted work和limit；这是indeterminate resource failure，
  不能换placement、生成no-good或使用approximate result继续。
- `InvalidSpatialAssignment`携带partition hole/overlap、缺merge group、重复Tile/shard、incomplete reduction fiber等closed violation；
  production Q49.P/Q50.B产生它就是compiler bug，test-only显式assignment则获得可定位的contract diagnostic。
- `BrokenDemandContract`表示verified IR与standard/Wafer interface互相矛盾、analysis graph有非法cycle或内部exact proof未通过；它是
  compiler error。

每个failure payload以closed reason enum和`DemandFailureSite`驱动控制流；字符串和`Location`只在最终diagnostic格式化时附加，不能
被caller解析。LLVM的recoverable-error实践要求文件/环境边界用`Error/Expected`，但本query是compiler内部有限语义分支，named sum
type比为每个candidate分配`llvm::Error`更直接；到compiler driver/package边界时再把上述类型映射为唯一产品错误。

query没有普通`ExactRejection`：完整spatial domain应由Q50.B构造正确，Q50.A不会因physical carrier缺失删除其中一个assignment。
`isCacheableLegalityConclusion`、`ProvenLogicalInfeasible`及“unsupported/indeterminate再试下一个placement”的控制链全部退出；session
memo可以缓存success和与source semantics无关的unsupported，但resource exhaustion按相同work policy重算也只会得到同类结果，不能
转成legality cache。

确定性合同覆盖结果本身而非仅遍历：proof按consumer semantic id、operand、logical shard、source kind/result和owner shard排序；
normal form使用canonical integer tuples；failure site使用上述semantic fields。operation/value地址只可作为analysis内部临时lookup，
不得进入排序、diagnostic identity、memo key或持久结果。相同IR与extensionally equal assignment在不同Tile输入顺序、DenseMap seed及
parallel task完成顺序下必须产生逐字段相同proof。

本子任务的验证必须包含两个真正的analysis-lifetime tests：read-only nested pass显式preserve后第二个consumer观察同一analysis
construction；中间mutating pass不preserve后第二个consumer观察fresh construction。另有compile-time API tests证明首次mutation后
`DemandPlanningSession`不能再query、所有failure kind由typed visitor穷尽处理、diagnostic文本变化不影响分支，以及关闭默认
instrumentation时重复query不walk/fingerprint整棵func。

上面的五个variant在对外分类时仍是四类：proof success、unsupported semantics、indeterminate resource exhaustion和compiler
contract error；后者在内部拆成`InvalidSpatialAssignment`与`BrokenDemandContract`，只是为了把上游assignment错误和IR/interface
错误定位到不同owner，不形成第五种恢复策略。

Q50.A没有“某条cross-op demand不覆盖，所以换一个placement”的正常分支。若Q50.B未来引入确实会改变logical semantics的
spatial alternative，它必须在该alternative自己的typed domain中给出exact rejection，而不能重新把physical failure塞回
Q50.A。

`satisfied`结果不包含layout、encoding、bytes、dense rectangle、local/remote action、route、buffer、send/recv、fusion或
resource schedule。representation/movement关闭后，lowering才可把exact set分解为physical pieces并证明其union all-and-only
覆盖logical demand；某个dense/strided descriptor或route表达失败只拒绝该physical assignment。Q49.P为baseline选择的
canonical DDR/必要peer carrier若还不能有限表达一个已支持的exact set，属于baseline correctness carrier缺口，不是Q50.A
relation失败；Q50.G/H扩展完整performance alternatives时仍复用同一proof。

### API、cache与current迁移

policy-free `SpatialAssignment`位于structured/relation边界，不能因Q50.A读取它就include candidate schedule或携带evaluation、
stable ordinal、route、residency、calendar和candidate统计。query只读current IR和immutable assignment，不apply mutation；
IR mutation后由pass manager失效function analysis，后续pass如需关系便在新planning session重算，不把operation pointer、diagnostic
字符串、手工epoch或fingerprint旁路表当稳定语义键。同一个pass一旦进入mutation phase则禁止再次query，不能等待pass结束时的自动
invalidation替调用者兜底。

func-scoped relation analysis只缓存current SSA、standard/op-interface relation和normal-form constructor facts。assignment相关cache
观察consumer execution domains、result availability和reduction placement的完整semantic key；只有具备extensional equivalence
proof才能投影key。cache lifetime不跨IR mutation，不保存legality bool，不按shard dimension、participant count、pointer或Tile
enumeration顺序近似identity。

baseline consumer-input reconstruction、Q50.B derived demand及Q50.G/H/J全部消费同一operand-level proof；empty destination解释为
无physical action。direct edge、init、support graph、program input和multi-result不再分别维护planner。layout、route、residency和
resource calendar仍是Q50.G–J后续physical choices，其失败不得回写Q50.A cache或删除spatial assignment。

### Gate

- 独立exact-demand suite直接测试operand-level query，不以edge strategy或materializer成功代签；覆盖multi-axis remainder、
  multi-reduction-axis input/init、M/N/K mixed partition的多个merge groups、independent/coupled multi-result、broadcast、affine
  window+stride+dilation+pad、strided slice/view、multi-piece union、fanout/fanin、program input、constant、explicit init root及
  多operand/multi-path pure tensor graph；
- relation constructor与normal-form算法可在2–4维有界小域上和独立generic Presburger oracle逐点比较，并明确标注这是穷举
  oracle；production path另用rank至少为3、主要维度至少1024的整除/非整除shape对和16-Tile assignment，证明均匀块与tail
  都不调用generic equality/subtraction、不逐元素枚举且在声明work bound内完成；
- metamorphic gate对同一assignment替换dense/strided/multi-piece physical carrier能力或让route失败，Q50.A proof
  extensionally不变；invalid assignment产生compiler-contract error，unsupported与resource exhaustion不会进入legality bool或
  no-good cache；
- analysis invalidation gate通过MLIR analysis manager证明IR mutation后旧analysis不再可取；同一immutable session内重复query
  不walk/fingerprint整棵function。extensionally等价assignment重算得到相同stable proof，结果不依赖Tile/pointer/hash或并行
  完成顺序；
- Q49.P调用链不出现candidate/schedule evaluator或edge strategy selector；reduction、broadcast、affine window、strided view和
  multi-piece至少各有一个normal FP16/BF16 TensorProgram进入baseline canonical carrier并继续到完整Q50.0 gate；不运行重型
  LLaMA search；
- Q50.B explicit assignment与production domain调用同一query；Q50.G/H/J只消费`satisfied` proof，不从physical fragments反推
  另一份logical demand。旧skip-init/support、ambiguous-chain、node-wide merge、partial-as-final-owner和
  carrier-rejects-placement测试必须替换为current合同的正负证据。

### Current/donor能力迁移与施工顺序

#### A-4 专项调研、调用链替换与验证闭环

MLIR testing guide把用户可见IR/输出验证优先放在lit/FileCheck，把malformed IR放在`verify-diagnostics`，只有内部C++数据结构和analysis
API才使用unit test；PassManager还能在每个transformation后运行verifier。由此本任务不能用一个C++ query unit或最终
`mlir::verify(module)`覆盖整条链：analysis、typed API、每个下游IR边界、named/production pipeline及最终package分别需要对应证据。
pass statistics/timing也由显式instrumentation启用，不应为了证明算法路径把默认日志嵌入production query。

本轮逐symbol扫描确认旧合同不是只存在于Q50.A文件：`LogicalShardTrial`被baseline、single/coupled region、representation、movement、
buffering、feasibility和大量fixtures直接持有；`IREpoch`还进入`CardProgramAnalysis`、edge strategy与placement；旧status进入baseline
placement、search filtering以及CardModule mapping validation。迁移必须是完整边界替换，不能先写一个新query后让这些consumer继续
通过adapter读取旧trial。

producer替换表：

| Current producer | 终态producer | 协调替换要求 |
| --- | --- | --- |
| baseline placement closure逐edge调用`buildEdgeShardTrial` | deterministic-baseline-closure调用canonical-spatial-assignment取得一份closed `SpatialAssignment` | producer只在attention-normalization后的roots上工作；不再为每条edge复制endpoint trial或用demand status循环过滤；canonical assignment无效即compiler contract error |
| search `SpatialPlacementDomain`产出per-node factors/Tile list/node-wide merge Tile，再由`buildLogicalShardTrial`派生 | Q50.B关闭compact `SpatialPlan`为完整`SpatialAssignment` | all-iterator shards、logical shard identity、embedding和per-output merge group一次形成；A不从factor重建placement |
| test fixtures手工mint `IREpoch`并拼`LogicalNodeTrial` | narrow `SpatialAssignmentBuilder` test utility | builder调用与production同一structural verifier；可以构造显式invalid值测试typed failure，但不能成为第二套domain |
| `CardProgramAnalysis::epoch`及edge strategy默认epoch | 无 | analysis lifetime由pass manager/current operation决定；program analysis只保留current IR派生的DAG/topology facts |

consumer替换表：

| Current consumer | 终态消费 | 必须删除的重复工作 |
| --- | --- | --- |
| `CardBaselineConsumerInputs`调用`queryOperand`，`CardBaselineEdgeCarriers`又调用per-edge query | Q49.P一次消费`ExactDemandProof`，按nonempty boundary生成canonical carrier | 第二次edge query、support reachability比较、按physical结果反推logical requirement |
| CardModule `MappingValidation`接收`ConsumerInputDemand`和`SpatialEdgeStrategy` | winner preparation接收proof投影出的`RootRegionWork`及Q50.H movement plan | `traceProducerToConsumerChain`、direct/unary-only假设、carrier set与logical set的generic equality、init/support skip规则 |
| `StructuredDAGEdgeDemandPlan/StrategyPlan`组装per-edge demand和physical action | Q50.H从`DependencyDemand`建立movement choices | direct-only planner、edge-local ownership、logical/physical字段混装和`ProvenLogicalInfeasible`过滤 |
| Q50.C/D single/coupled region从trial重查shard/partial/merge | `RootRegionWork`/`RegionPlan`消费assignment和proof | `findNodeTrial`、partial role推断、node-wide merge字段和第二份boundary walk |
| Q50.G representation与Q50.H movement从`(Tile,node,operand/result)`及trial恢复shape/owner | G按logical value version选encoding，H按proof boundary/final owner选movement | operand位置覆盖producer identity、partial-as-final-owner及shape/Tile命名恢复 |
| Q50.F/I/J从trial、representation、movement各自重建footprint/lifetime/order前提 | 各stage消费A proof中对应logical demand/reduction requirement及自身上游typed plan | caller-supplied missing facts、重复module walk和把resource failure回写A legality |

donor witness逐项处置：

| 现有witness组 | 新witness及判断 |
| --- | --- |
| direct、permuted map、reduction、broadcast、DPS init | 全部改接唯一`deriveExactDemand`；同时检查operand/destination/source result和exact set，不只看status |
| strided support、pad、decode insert+reshape、multi-operand support、exact-empty insert | 迁入relation-analysis + labeled reverse-topological query；加入同producer多路径union、program input、constant和multi-result support op |
| multi-result ownership、explicit replication、partial contributions | 改为per-result final availability和per-output reduction groups；加入M/N/K mixed、two reduction axes、coupled multi-result及init exactly-once |
| ownership hole/unique overlap返回`ProvenLogicalInfeasible` | 改为`InvalidSpatialAssignment` contract negative；production Q49.P/Q50.B domain test证明不会生成这些值 |
| ambiguous producer chain unsupported | 删除旧预期；同producer多路径应exact union。真正缺interface/effect crossing才是`UnsupportedDemandSemantics` |
| cross-borrow epoch与in-place fingerprint mutation | 替换为真实AnalysisManager preserve/invalidate tests及compile-time session-close API test |
| Presburger exhaustion、16 large rectangles、balanced remainder | 保留并扩展为pre-operation work-limit分类、normal-form observer和有界generic oracle；默认instrumentation关闭 |
| carrier failure metamorphic、equivalent placement determinism、fanout | 保留；再加入hash insertion、Tile input order和parallel completion扰动，proof逐字段一致 |
| edge strategy partial overlap/different shard axes/transpose/reduction/strided/broadcast | logical部分迁到A query tests，physical fragment/action部分迁到H；旧混合fixture不能继续作为一个test owner |
| old init/support “no spatial action” | 改为A始终保留logical requirement，H/baseline依据exact-empty/local/DDR事实决定是否生成physical action |

验证按由内到外的六层执行；任何一层失败都不能用更外层偶然成功覆盖：

1. **Relation unit/property**：每个normal-form constructor、composition、image、union/intersection在2–4维明确有界的oracle域与independent generic
   Presburger逐点oracle等价；unsupported和work limit有typed negative。
2. **Analysis/query unit**：labeled multi-seed逆拓扑、exact-empty、多路径、program boundary、multi-result/reduction groups与stable order；
   reference实现独立逐`DemandKey`枚举有界图，和共享算法比较完整proof而非只比aggregate demand。
3. **Pass lifetime/API**：read-only preserve复用、mutation后重建、session close后不可query；generic/custom IR parser roundtrip和
   `verify-diagnostics`覆盖新增interface/verifier负例。
4. **Direct consumers**：Q49.P、Q50.B、C、G、H、F、I、J分别有一个test证明实际读取新field；mock/observer断言旧edge query、
   chain trace、trial adapter和physical-to-logical equality调用次数为零。
5. **Pipeline**：`wafer-opt` named pipeline与production driver对相同input得到相同selected logical proof/IR；`verify-each`开启，
   source op分类fail closed，所有相关lit实际supported并执行。
6. **纵向gate**：轻量FP16/BF16 reduction、broadcast、window、strided view、multi-piece、explicit init和multi-producer各自走
   TensorProgram→selected CardModule→Q50.0→package/no-card。Q50.A实现期不跑search LLaMA；Q49.P完整闭环后只按其合同跑一次baseline
   LLaMA，不能拿旧日志或重复重跑代签A。

source清理门禁是active tree中下列旧identity零残留（archive文字不计）：`LogicalShardTrial`、`LogicalNodeTrial`、`LogicalTileBinding`、
`TileRole`、`IREpoch`、per-edge `StructuredDAGExactDemandQuery::query`、`buildEdgeShardTrial`、`buildLogicalShardTrial`、
`ProvenLogicalInfeasible`、`isCacheableLegalityConclusion`、node-wide `reductionMergeTile`/`mergeObligation`和direct-only edge demand plan。
零symbol只能证明旧接口退出，仍需上面donor witness→current production caller→test三列全部有证据，才能删除旧source/tests并把Q50.A
重新标为done。

| Current / donor能力 | 终态owner | 必须保留或改正的witness | 退役条件 |
| --- | --- | --- | --- |
| `IndexRelation`的affine、broadcast、slice、reshape、insert/pad和finite-piece exact constructors | Q50.A relation analysis | constructor与有界Presburger oracle extensional equality；真实规模shape不走generic proof | normal-form API和全部直接consumer迁移后删除旧optional pattern/late fallback |
| `StructuredDAGExactDemandQuery::query` per-edge relation/image | Q50.A operand-level `deriveExactDemand` | direct/permuted/reduction/broadcast/init/multi-result/per-destination tests改接唯一operand query | 不再有per-edge chain resolver、edge-local demand cache或第二份coverage结果 |
| `queryOperand` multi-producer reconstruction | Q50.A operand-level SSA worklist | insert/pad/reshape、多operand、同producer多路径、exact-empty branch、program input/constant | materializer只消费`OperandReconstruction`，不再TypeSwitch或按“是否到达”比较 |
| `LogicalShardTrial`、`LogicalNodeTrial`、`TileRole`、`IREpoch` | Q50.B `SpatialAssignment` + Q50.A derived proof | multi-axis/tail、replication、analysis invalidation和stable ordering | 所有baseline/search/G–J caller同批迁移；零wrapper、零fingerprint/epoch协议 |
| node-wide `reductionMergeTile`、`mergeObligation`、partial contribution result owner | Q50.A `ReductionMergeRequirement` + Q50.B per-group placement | M/N/K mixed、two reduction axes、multi-result/coupled、init exactly-once、每组final owner | Q50.C/H/J materializer与cost/schedule消费新要求后删除旧字段和对应测试 |
| `PartialReductionOpInterface`的source-owned partial mechanics | Q50.A per-output contribution requirement与winner materializer | multi-reduction-axis、identity/init exactly-once、interface failure atomicity | Q50.A不增加numeric legality/reassociation gate；planning不调用materialization，winner不为改destination clone op |
| baseline `CardBaselineConsumerInputs`/`EdgeCarriers`的exact boundary消费 | Q49.P canonical assignment consumer | 五类normal relation、empty destination、support multi-producer、完整Q50.0/source-to-package | baseline无direct-only `StructuredDAGEdgeDemandPlan`、无support rebuild或physical→logical反推 |
| old direct-only `StructuredDAGEdgeDemandPlan`和更早balanced single-axis demand planner | 无；能力并入上述owners | direct SSA、stable per-Tile demand和relation reuse保留为新API测试 | current callers与独有test witness均迁移后删除source/header/test，不以未进CMake判废 |

实现按下面顺序原位替换，不建立新旧双reader或compatibility wrapper：

1. 在analysis library内先补`TensorIndexingOpInterface` external models、typed partial-reduction mechanics及construction-aware
   `IndexRelation`；现有API在这一提交内只增加可复用能力，不切换pipeline。
2. 一个协调提交同时替换`LogicalShardTrial`/`TileRole`/`IREpoch`与所有baseline、Q50.B及G–J direct consumers，建立
   func-scoped relation analysis、`SpatialAssignment`和`ExactDemandProof`；编译树中间不能同时存在两套current合同。
3. Q49.P canonical consumer、Q50.B domain和Q50.C/H/J winner materializer分别消费同一proof；删除direct-only edge planner、
   second chain walker、OperationFingerPrint、generic production fallback和旧reduction fields。
4. 迁移上表所有unit/property/integration witnesses，运行fresh定向build/test、source-to-package和Q50.0 gate；只做设计阶段时
   不执行这些实现验证，也不把旧通过数字继续当作新设计证据。

## Q49.P Current exact-demand 与 plan-only resource legalization

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

#### Plan-only resource legalization

上述exact demand、carrier、temporal occurrence和canonical G--K事实全部闭合后，baseline为当前coordinate调用Q50.F：

```text
chooseCanonicalBaselinePlan(source, target):
  coordinate = initialCanonicalCoordinate(source, target)
  while true:
    facts = deriveExactDemandAndCanonicalDownstreamFacts(coordinate)
    result = queryFullCoordinateFeasibility(facts)
    switch result:
      FullFeasibilityProof:
        return closePhysicalDataflowPlan(coordinate, facts, result)
      ExactRejection(witness):
        coordinate = nextMonotoneLegalization(coordinate, witness)
      Unsupported | Indeterminate | CompilerBug:
        return typed failure
```

循环内CardModule、TileRegion、Instr和Q50.0调用次数都必须为零；旧coordinate被值语义替换，不留alternative、IR或solver state。
只有返回的closed plan进入一次Card subtree transaction与一次Q50.0。actual normalized resource problem与full proof不一致、或
actual packing失败，均为compiler bug并终止；不得回到循环。这样baseline仍是确定性first-fit合法化，而不是search，也不再把
lowering当资源query。

#### 实现迁移和完成门禁

1. 将Q50.A current query原位扩展为按consumer operand分组的exact-demand结果；现有per-edge consumer全部迁移，不能并存第二套
   baseline relation恢复逻辑。
2. borrowed source上的planning session完成全Tile query、carrier coverage与Q50.F resource验证；`lowerCardModule`只apply最终
   FullFeasibilityProof plan。任何rejected/partial coordinate均不构造IR。
3. common CardModule/TileRegion materializer直接构造final single-root region；baseline与search分别调用该policy-free apply，不能
   保留baseline-only修补分支。
4. 同批删除`buildBaselineRegionSource`、两套support rebuild、empty/destination补丁、post-hoc root closure/splitter及只覆盖这些
   路径的statistics/tests；旧SPM capacity probe API不恢复。
5. 每种admitted consumer input reconstruction用明确标注的有界static shape穷举需求子集，比较full evaluation与partial reconstruction；
   同一reconstruction另有真实规模整除/非整除正例。组合测试覆盖chain、diamond、fanout、multi-producer `insert_slice`、
   Peer/RegionCut混合、multi-result、empty branch、nonrectangular pieces、
   reduction temporal wave和unsupported atomic failure。
6. 显式测试计数证明relation construction按semantic support edge计数、每个非空`(value, Tile)`只处理一次、16 Tile没有完整DAG
   重复walk、planning CardModule/Q50.0均为零、selected CardModule/Q50.0各一次。这些计数只在显式测试sink建立，普通编译不创建
   统计对象、不打印日志。定向unit/lit与轻量source-to-package/no-card通过后，只运行一轮fresh FP16 LLaMA
   `optimization-none` source-to-package/no-card；不得进入search。

#### 2026-08-19 current实现能力与重新打开的门禁

current实现已经迁入按职责拆分的`Compiler/Baseline`、`Compiler/Planning`和TensorProgram→TileRegion conversion文件：

- 一个baseline invocation只建立一次只读source session；`StructuredNodeUseIndex`一次索引structured use，`StorageRootMemo`压缩
  storage-root查询，mapping session消费按consumer operand分组的exact demand；
- final materializer在structured producer处停止，只重建typed recipe要求的argument/constant/view/extract/insert与carrier；
  multi-producer fan-in、exact-empty branch和Peer/RegionCut混合不再通过递归support closure恢复；
- `buildBaselineRegionSource`、root/function SPM probe、post-hoc support rebuild、winner rematerialization与默认Tile IR/statistics已经退出
  baseline主路径；但每个closed coordinate仍产生一个CardModule并由Q50.0消费一次，这是本节plan-only迁移必须删除的剩余控制流；
- CardModule fan-out时，大型Tile body直接move进唯一Tile output；只有每个output确实需要的小型card-shared declaration按
  `IRMapping`复制。16个Tile entry与后续独立Tile stage使用bounded并发并按Tile ID稳定归并；
- active测试已覆盖single-root、multi-producer、reduction temporal accumulator、Peer/RegionCut混合、Card→Tile move-only owner、
  source-to-package与no-card。已经删除的旧TileRegion候选测试不复活；其中仍属current合同的proof已迁入CardModule/current gate。

首次fresh FP16 LLaMA `optimization-none`编译在约163秒内生成current package，CardExecutable阶段约99秒、peak RSS约4.48GiB；
第一次完整CTest随后因PyTorch runner仍读取已退役的`package/functions/forward.mlir`而失败。current package有意只保存
manifest/modules/program data，因此runner现改为从source program读取`functions/forward.meta`、从package读取manifest port，并已通过
定向runner unit、对该fresh package的payload准备和`wafer-run --no-card`。修复后official CTest从fresh source重新执行完整链，
于181.70秒通过1/1；它证明旧baseline功能链可用，但不能证明plan-only legalization或winner-only Q50.0，因此不再支持Q49.P的
`done`状态。

这次显式timing还记录到16个Tile合计执行约22,968次TileRegion→Instr conversion。结果已经有界且正确，但相似Tile body仍有大量
重复lowering；其中**coordinate之间**的重复由Q49.P plan-only迁移直接消除，最终winner内部16 Tile的等价pure query共享才归Q52。
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
| actual materialization | `materializeCandidate`混合CardModule构造、fusion/layout/buffer检查与complete compile | planning阶段不物化；预算结束只将selected winner assignment交给唯一materializer和Q50.0 |
| late failure handling | allocator/buffer failure生成siblings、lookahead、beam closure并重排shortlist | Q50.F及各轴pure query在planning中返回typed结果；winner commit失败是合同缺口，不回到Core做late repair |
| winner | accepted cohort丢弃actual modules，以schedule plan/stable ordinal选winner后再rematerialize | Q51 session持有search-local winner assignment；预算结束只物化一次，不存在accepted executable cohort |
| failure routing | materializer写字符串`failureGate`，controller按字符串决定prune/cache/feedback/order | 新Core只消费closed typed evaluation outcome和typed causal evidence；diagnostic label只打印，不参与控制流 |
| inspection | 每个complete compile在acceptance前无条件打印全部Tile IR并穿过synthesis result | Q49.P先把common seam改为显式winner-only inspection；Core/candidate evaluation不携printed IR |
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
  search-local full-plan incumbent和coverage/lower-bound evidence。foundation只组合B/A真实transitions；后续Q50提交逐轴扩展current
  variants。Core只接受mechanism已经形成的typed transition/query result，不枚举或物化某个轴。
- Output IR / files:
  不产生新IR层或文件。foundation在缺axis时返回typed `IncompletePlanningDomain`及已验证frontier/work，不构造winner；所有axis闭合后
  输出move-only winner plan、coverage和unresolved-work evidence。Q51 closure才将winner交给一次性materialization/Q50.0。
- Downstream consumer:
  Q50.S提供immutable semantic facts和planning description，Q50.B–Q50.K逐项扩展current typed assignment和transition；Q50.F接入scoped feasibility analysis；Q51 closure接入
  全部真实轴后的winner一次性materialization/Q50.0、全轴oracle、planning-domain证明和source-to-package chain。
- User-level driver / named pipeline:
  不新增pass、CLI、optimization policy或磁盘sidecar。spatial-domain与exact-demand-boundary完成后，
  search-control-foundation把explicit public `search`切到new owner并删除旧controller；
  incomplete结果明确失败且绝不fallback。Q51 single-winner closure只解除最后missing-axis gate并接common commit，不再次切controller。
- Explicit non-goals:
  不实现Q50.B–K任一choice domain，也不为Q50.S建立algorithm domain；Core只静态调用已经闭合的真实axis APIs。不在Core或Q50.F建立materialize-and-discard gate；不预设best-first priority、
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

### 四类query-local事实

```text
Immutable session input:
  immutable IR borrow + target facts + cost cohort + currently implemented mechanism set

Candidate assignment:
  only named typed choices implemented and consumed by current Q50 mechanisms

Session control:
  deterministic non-dropping frontier + search-local winner + work/budget accounting + coverage/bound evidence

Derived cache:
  session-owned source facts + exact demand + lifetime/calendar/SPM/cost facts
```

- source-only facts由与MLIR analysis wrapper共用的policy-free builder建立；Core直接拥有typed result，不持有`Analysis *`，也不在
  pass外构造`AnalysisManager`。session lifetime绑定同一immutable source borrow；Core不得把token地址、`Operation *`、walk ordinal
  或printed IR当stable key。
- Core建立的current assignment aggregate起初不含未来轴。每个Q50 checkpoint同批加入本轴named typed field、transition、
  canonical encoding、query/apply和失效关系；Wafer-owned接口原位演进，不保留编号schema、旧wrapper或两套state。
- mechanism query只读source、target和parent assignment；transition apply先完整验证，再原子构造child。child不保存transition
  history、proposal priority、feedback root、failure history或derived metric。缺少尚未施工的轴表示partial assignment不可
  materialize，不允许从baseline default或旧selector暗中补齐。
- mechanism set、priority、search-local winner和work accounting是session facts，不属于任一candidate。future-boundary equivalence只有在本轴
  mechanism证明被投影choice不再影响任何future consumer时才可合并；Core不提供按aggregate bytes/makespan自动合并state的
  generic shortcut。mechanism顺序由唯一current driver静态组合，不提供runtime registry、dynamic plugin list或user option。

### Search-local winner 与candidate

Q51 driver直接从Q50.S-normalized immutable TensorProgram建立search session，不先运行Q49.P。Q50.B–K加入的partial state在全部required
coordinates闭合前不能形成full plan，也不存在“用baseline或旧selector补齐未施工轴”的complete candidate。第一条production
full-proof plan必须由new typed mechanisms构造并通过Q50.F/J等planning query，随后按Q51-3同cohort cost建立search-local plan
incumbent；后续full-proof plans只在typed comparison为Better，或Equivalent且完整semantic tie-break更小时替换。整个planning阶段
CardModule、Instr和Q50.0调用均为零，比较不依赖queue ordinal、hash insertion或并行完成顺序。预算结束仍没有full-proof plan时
返回typed failure；selected plan随后只commit一次。baseline只由独立`none`调用，matched A/B也由外层测试分别启动两个编译事务。

`ProgramDataHandoff`始终由外层compiler transaction唯一拥有，planning Core只消费stable program identity/range，不复制owner或
materialize package。winner选定并commit后，Q59才把同一handoff随唯一actual result继续交给target/package transaction。search计时
从search入口开始并覆盖analysis、state expansion与winner selection；winner materialization/Q50.0作为一次独立下游stage记录，不混入
另一次`none`事务的work。

### Transition、evaluation 与budget

- exact expansion对一个parent返回当前mechanism的全部immediate typed children，或返回仍留在frontier中的typed semantic
  continuation；opaque iterator/cursor不能成为遗漏siblings的旁路。先访问的child无论accepted、rejected或indeterminate，
  其它siblings都保持可达。
- `deferred(required coordinates)`列出typed prerequisite，只有相关assignment改变或闭合后才重新query；不得poll同一state。
  proven exact rejection只删除被其typed causal witness覆盖的state。Core本身不从diagnostic string合成no-good或扩大scope。
- resource/solver exhaustion形成unresolved work：不删除state、不更新winner，也不能签发optimality。Core自身的malformed transition、
  stale borrow或broken invariant是fatal compiler error，不得用baseline掩盖。
- Q51.Core只路由planning query outcome，不构造actual IR或调用Q50.0。Q50.F与各轴query从immutable facts提供可重算的
  lower-bound/deferred/causal/cost analysis；Q51 closure在预算结束后才物化winner并进入一次Q50.0。
- optional work counts记录generated、evaluated、exact-rejected、indeterminate、solver/query work和winner updates；wall deadline只是
  安全中断，不能改变已完成state的stable排序。actual compilation count不属于planning budget。
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
3. **Test-only full flat exhaustive oracle**：Q51 closure在全部真实轴闭合后可于tiny domain逐点使用actual materializer和Q50.0，
   比较planning domain、cost/pruning、winner与actual digest；该runner不进入production compile。Core fixture不能替代这一层。

### Current code处置与完成条件

- 可直接复用Q50.A immutable-borrow合同及`StructuredDAGAnalysis`中仍为current IR可重算的事实；Q50.0只属于winner commit，Q49.P
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
  missing axis返回typed incomplete，完整域才产生winner，link closure不引用旧controller，也不执行parallel actual evaluation。
- search-control-foundation在spatial-domain/demand work items后独立提交并切public routing；后续每个domain work item同批扩Core。per-axis reference
  enumerator、test-only flat exhaustive oracle、exact planning-domain coverage、有效fusion、single-winner source-to-package
  chain和donor能力迁移仍是Q51 closure的完成条件；Q51 closure只接winner commit，不再切public controller。

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
  FA/FD合同；winner只展开一次并在Q50.0前无attention和executable Linalg残留；none/search各有source-to-package/no-card witness。
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
  logical work and mandatory simultaneous-state groups
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
| F | every described state/scratch/message/event/field进入full resource problem；winner hidden resource为compiler bug |

`none`与`search`读取同一mode约束。B canonical producer对FA保持K2单一logical interval；对FD从B domain中最小合法非平凡K2 partition和
stable embedding/merge owner开始，只有F exact causal rejection要求更多contributors时才沿canonical B successor单调增加，
取得第一个full-proof plan。search枚举全部合法K2 partitions/embeddings/per-output merge placements。baseline不能把FD改回FA，
search也不能把FA当成FD空间切分。

Q50.B/A完成attention read-only接入后，Q51.Core的首个真实state是`SpatialState`，不是`RootAlgorithmState`。Q51 problem借用固定
normalized semantic roots；`PhysicalDataflowPlan`不复制algorithm assignment，FA/FD从source op读取并进入observed dependency key。

### S-5：Winner内selected Linalg decomposition

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

进入Q50.0前card-scoped verifier要求：attention op与executable Linalg均为零；all-and-only action/value/materialization IDs匹配；F
plan/actual resource problem相同。失败擦除整个新Card subtree，不回frontier换plan或algorithm。

禁止新增`wafer.tile.attention`、`wafer.instr.attention`、attention TargetCall、package algorithm字段或runtime cache branch。

### S-6：Current/donor能力迁移

| Current / donor能力 | 终态owner | 必须迁移的witness | 退役条件 |
| --- | --- | --- | --- |
| `AttentionAnalysis` softmax/value-contraction proof | 05 graph normalizer + attention verifier | named/generic QK/PV、mask/scale、views、multiple roots、near-miss/effect conflict | 不再返回长期graph shadow或single-observable-root gate |
| `DecodeAttentionAnalysis` cache append proof | 05 algorithm classifier + Q53 explicit-state witness | prefix append、updated K/V consumption、returned state、argument/name perturbation | 不再控制search domain或runtime policy |
| `TensorProgramAlternativeDomain` Original/Online/Split Cartesian | Q50.S normalized op/classifier；block/partition归E/B | no whole-program clone、no algorithm search state、both policies same input | old kind/parameter successor/materialization API零残留 |
| `AttentionAlternative` online recurrence | coupled interface + selected Linalg decomposition | block/tail、state update/finalize、arbitrary output piece | 不创建whole-function alternative或hidden full scores |
| old FlashAttention donor | FA selected decomposition + C/E/D producer coupling | QK/PV tile、mask、tail、producer exact slice | current caller与actual Tile witness闭合 |
| old FlashDecoding donor | FD B/A/H/J integration | K2 partition/tail、all partial states、merge/output、two-step decode | 无decode-only physical search gate或split algorithm op |
| `AttentionTensorOps::cloneLinalg` | selected Linalg builders / local IRMapping | exact regions/maps/SSA in one winner | Module/Func/DAG clone和replay cache为零 |

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

Q50.B把spatial问题建成factor graph，但不在本层选择winner：每个`NodeSpatialPlan`是variable，每条structured dependency和
reduction group是constraint/cost factor。per-node exact domain由三个嵌套successor组成，均不预建point vector：

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

初始实现只在完整`SpatialPlan`上做这个quotient，per-node exact successor仍覆盖raw embeddings。future若要在partial plan上提前
canonicalize，只能使用保持已固定semantic prefix和external colors不变的automorphism stabilizer，并用tiny orbit oracle证明；否则宁可
多枚举。program inputs、fixed endpoints或unavailable Tiles会通过颜色自然打破对称。

#### NoC-aware communication graph与经典mapping proposals

Alpa的auto-sharding把每个op的partition strategy作为one-hot变量，把compute/collective放在node cost、resharding放在edge cost，并用
ILP联合选择；GSPMD/Shardy把conflicting shardings显式变成reshard/collective，而不是假设same device count就local。经典process/NoC
mapping把application communication matrix与processor distance matrix组成Quadratic Assignment Problem；高质量实现常用multilevel
graph partition、recursive bisection和swap/local search生成mapping。NCCL则先读取actual topology，再分别搜索ring、tree、split-tree
等collective graph。这些方法共同说明：placement必须看完整communication graph和actual topology，但placement、route family与event
schedule仍是三层不同决策。

Q50.B从A的complete proof为一个已关闭或部分关闭的spatial plan建立query-local communication-demand graph：

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

对一个complete placement，B可计算下列route-independent lower bounds：

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

高质量placement proposals使用四个互补构造器，输出都只是exact domain中的完整points：

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

边界示意：两个producer/consumer plan拥有相同Tile数量但embedding不同，B的QAP/hop/cut facts会给出不同proposal顺序；H可能为同一
placement选择不同unicast或multicast trees；J又可能因同时发生的其它traffic改变winner。任何一层都不能把自己的局部最优写成上一层
legality。

domain提供下列proposal顺序，但proposal不是legality：

1. unpartitioned/single-Tile、最大parallelism、balanced与uniform-tail的边界点；
2. topology内部hop work较小的compact subsets/embeddings；
3. producer/consumer exact-demand owner交集较大的co-located或partial-overlap placement；
4. independent observable components优先使用disjoint Tile subsets；
5. reduction merge优先在contributor或直接consumer participant上，但其它available Tile siblings仍可达。

为chain和一般DAG产生较好的complete spatial proposals时，复用一个不物化IR的factor-graph lower-bound query：

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

这些bounds只能包含当前已知且单调的logical work、mandatory merge与最小transport；topology compactness、local-edge count、
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
- compact、co-located、partial-overlap、disjoint与independent-component placements都是proposal siblings；全部verified-legal
  subsets/embeddings仍由exact successor可达；
- topology symmetry只通过verified automorphism quotient，启发式相似或resource renaming不能删state；
- chain/tree/general-DAG算法只产生proposal order和admissible lower bound，不保留local winner，不混入movement、schedule或
  fixed-capacity结论。

### B-3 专项调研：独立oracle、donor迁移与actual gate

Shardy verifier逐层检查tensor rank与dimension-sharding数量、mesh轴唯一/不重叠、device-id顺序和shape约束；XLA
`HloSharding::Validate`同样区分tuple/leaf、tile rank、device range、device uniqueness和device count，并在错误中附带具体shape/sharding
上下文。对应到本项目，`SpatialPlan` verifier必须只验证自身closed structural contract，Q50.A验证derived logical proof，Q50.C验证
actual work；不能让任何一层用下一层成功倒推自己正确。

三个oracle彼此独立：

1. `SpatialPlanReference`用普通nested loops直接枚举小域，不include production domain header，也不调用production successor、
   canonicalizer或proposal helper。
2. `TopologyMappingReference`对小communication graph穷举embedding、merge placement、topology automorphism和route tree，验证B的
   canonical set与bounds/proposals；它只在test中调用actual route enumeration。
3. `SpatialMaterializationReference`逐iteration point计算期望`(LogicalShardId, TileId)`和merge group，再读Q50.C actual
   TileRegion/operation relation比较all-and-only coverage；不以IR文本中出现多少Tile名字代签。

current及donor test的逐项迁移如下：

| 旧witness | 新owner与改写 |
| --- | --- |
| current multi-axis factor/embedding Cartesian equality | 加入Balanced/Uniform extensional dedup、tail、per-group merge后与`SpatialPlanReference`比较 |
| duplicate/out-of-domain embedding negative | plan verifier分别覆盖duplicate logical cell、duplicate Tile within node、unavailable Tile、wrong arity和merge group缺失 |
| current remainder/reduction embedding | 拆成parallel tail domain与partial-interface structural domain，不再混入其它判断 |
| old unsupported-reduction-combiner test | 删除该方向；只验证partial-reduction interface能否完整表示selected result group |
| scalar empty-factor/every singleton Tile | zero-iterator singleton-cell定义；任一available Tile raw state可达，global automorphism只在整plan canonicalize |
| current chain/branch/diamond Card Cartesian reference | 升级为program-levelraw product + global automorphism oracle，不能对node独立quotient |
| current partial owner/one merge Tile | 改为M/N/K mixed的多个`ReductionGroupId`、每组全部merge Tile choices及A final-owner proof |
| donor multi-result producer | per-result/coupled-result groups由interface声明，所有results进入A/C实际witness |
| donor bounded partial/disjoint、same-group remap、three distinct groups | 保留为exact siblings；proposal顺序可不同，domain不能因local transition class删除 |
| donor full-mesh rectangle/compact-first | rectangle/serpentine/compact成为topology-shaped seeds；任意subset/embedding仍由raw successor覆盖 |
| donor independent branches use all Tiles | recursive-bisection/disjoint proposal正例，同时保留共享Tile sibling给Q51比较resource/schedule |
| donor long-DAG nondominated states | 改为129-node lazy/proposal work test；不保留旧live-boundary pruning结论 |
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

proposal验证与domain验证分开：在同一小域上，chain DP与treewidth DP的首个proposal cost等于flat factor-graph minimum；general
DAG best-first的bound单调且不会让exact successor中的siblings不可达。关闭compact/component/merge proposals或反转它们的顺序，
domain集合与Q51有界穷举oracle optimum不变。verified topology automorphism前后的canonical set相同；增加unavailable/fixed-color
Tile只打破真实symmetry。

NoC专项oracle覆盖line、ring、2-D mesh、directed/asymmetric link、broken link和heterogeneous endpoint colors，以及chain、star
multicast、all-to-all、diamond和reduction gather communication graphs。有界穷举reference枚举全部embeddings和route trees，检查：
exact QAP/DP模式达到reference placement cost；heuristic模式只返回domain member；`dilation/hopWork/cutCongestion`从不超过actual
route optimum；multicast payload不会按destination错误重复成lower bound；改变NCCL式ring/tree proposal family只影响H的顺序，不改变
B domain或canonical key。

metamorphic tests让Q50.A relation unsupported/resource failure、Q50.G layout unavailable、Q50.H route unavailable和Q50.J schedule
failure分别发生，证明它们不会改写Q50.B domain；invalid `SpatialPlan`只返回compiler contract error，不能靠另一个
Tile placement掩盖。默认调用不构造statistics；显式test observer只证明16-Tile large-shape successor不预建排列/Cartesian
product、无IR clone/materialization且单步work受state size约束。Q51完整链前不运行重型LLaMA search。

Q50.B的actual downstream witness由Q50.C提供：selected plan关闭后，每个exact shard和merge group必须各映到all-and-only actual
TileRegion work。局部gate通过后删除旧single-axis/node-wide-merge/trial/observable-placement字段。“非最大参与、非compact
physical subset或局部lower-bound较差的placement成为global winner”留给Q51跨轴gate。

完成门禁还要求active source/test中旧`SpatialPlacementAssignment/CardSpatialPlacementDomain`、`iteratorFactors`、
`reductionMergeTile`、`getMaximumParticipantAssignment`作为winner、`EdgeTransitionLegalityCache`、`ObservablePlacement`和旧
live-boundary state零残留；CMake只注册新owner。fresh实现验证依次运行B reference/property、A closure、C actual、NoC bounds、
named/production parity和轻量FP16/BF16 source-to-package/no-card；测试必须实际执行而非unsupported。设计复核阶段不运行这些构建，
也不运行LLaMA search。

### Current/donor能力迁移

| Current / donor能力 | 终态owner | 保留或修正的witness | 退役条件 |
| --- | --- | --- | --- |
| current完整iterator factor与有序distinct-Tile successor | Q50.B lazy partition/embedding successors | all factors/tails/subsets/embeddings reference equality | 迁入两scheme、automorphism key和per-group merge后删除旧`iteratorFactors/tiles` API |
| current maximum-participant constructive choice | Q50.B proposal ordering | proposal合法且关闭后domain不变 | 不再作为baseline fallback或local winner |
| old physical rectangle/compactness catalog | Q50.B compact proposal generator | full mesh包含all rectangles、compact 2x2优先 | exact domain不再只含connected rectangles；hop work不作legality |
| old `getNodeSpatialAxes` result0/parallel-axis恢复 | all-iterator interface query | multi-result、reduction、scalar与projected-result正负例 | result-axis/single-axis字段零残留 |
| old independent-component placement | Q50.B proposal + Q51 resource-aware combination | disjoint branch proposal、overlap sibling仍可达 | 不用component shortcut删除共享Tile plans |
| old chain/general-DAG partial states与live-boundary Pareto | Q50.B factor-graph proposal；full pruning归Q50.F/Q51 | long chain、diamond、three-stage proposal和tiny optimum | 删除基于未关闭movement/schedule metrics的state pruning |
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
  boundaries。提供一个singleton root emitter；production只在Q50.D选定最终group后由winner materializer调用，不先建singleton
  CardModule再改成fusion。
- Output IR / files:
  planning query不产生IR或文件，只返回ephemeral `RootRegionWork`。test-only singleton actualization或selected winner emitter
  产生一个`wafer.tile.region`，其actual operations通过typed materialization relation映回唯一semantic root；boundary只是logical/
  selected-value接口，不默认等于DDR movement。
- Downstream consumer:
  Q50.D把singleton work作为group partition的叶子并选择coupled groups；Q50.E–K为最终groups补齐temporal、representation、
  movement、buffer、event/resource和pipeline。winner materializer直接从这些closed assignments构造一次完整CardModule，Q50.H
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
  outer region且emission relation只映回该root，其它structured producer均为typed boundary；Q49.P与winner emitter复用同一leaf
  primitive；production planning/materialization计数证明零singleton prebuild、零Module/DAG clone/replay。

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

### C-2 专项调研与winner-only CardModule transaction

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

failure分类为：source op/interface本身不能表达selected work是typed unsupported；prepared work缺boundary、symbol closure或与complete
plan矛盾是compiler contract error；emit/postverify失败是winner commit bug。SPM、route、contention或schedule rejection必须在
Q50.F/G--K关闭plan前返回，不能在C-2失败后触发repair、fallback或另一次materialization。

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
- production counters证明planning与Q50.D group selection期间singleton actualization为零，winner完整CardModule只构造一次；无
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
| current `CandidateMaterialization`/`DependentDataflow` whole-Module clone | A/C/H各自query与winner-only emitter | selected edges、operand binding、failure classification | 本路径不再有candidate clone API；其它owner的clone必须在其迁移任务单独清点，不能由C虚报全仓清零 |
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
  value或cross-region boundary。query只形成typed region/execution plan；winner outer materializer一次构造actual regions。
- Output IR / files:
  不产生IR或文件。Q51 state保存typed `RegionPlan`：root partition、execution instances和use bindings；derived nested relations、
  operand reconstruction、lifetime和cost不存state。winner actual IR中每个group对应一个outer TileRegion，内部SSA/loops直接证明
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
  nesting明确deferred；singleton、maximal和intermediate groups的test-only/winner actualization各恰一region，mandatory work exact覆盖，
  replica显式可数，direct nested SSA无中间DDR，stored value有独立producer traversal，fanout共享version只生成一次；旧
  CompleteTraversal/ProducerTileFusion能力逐项迁移后退役。

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

### D-2 专项调研：exact successor、proposal与winner emitter

connected region partition本身就是graph-restricted set partition。Graph Coalition Structure Generation的结果说明一般图上即使group
value只是edge-weight sum也为NP-complete；完整partition枚举在complete graph上至少有Bell number个输出。其DP可在典型additive
objective下做到`O(3^n)`求最优值，但不能把完整合法域变成多项式。SystemML的operator-fusion工作同样把DAG fusion作为exact
cost-based plan问题；XLA GPU current priority fusion则按estimated benefit使用priority queue，并明确把producer duplication计入cost。
这些实践对应本仓的结论是：exact successor必须lazy且承认指数规模；DP、priority fusion和最大group只能生成proposal，不能成为
legality filter或局部winner。

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

#### Winner-only group emitter

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
- test-only/winner actual IR分别证明independent stored有两个traversals且无中间DDR，required nested只有consumer内producer tile和
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
  variables。domain以integer intervals/constraints惰性表示全部合法sizes/orders，breakpoints只控制proposal和bound。winner emitter
  用共同TilingInterface mechanics构造compact loops/tails。
- Output IR / files:
  不写IR或文件。Q51 complete state保存typed `TemporalPlan`；partial state可保存尚未定点的integer intervals。derived wave work、
  producer preimages、footprint和cost按需重算。winner actual IR只保留真实loops/tails，不写score、capacity history或recipe attr。
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
  baseline复用同一domain query但只走deterministic greedy；selected plan在singleton/group actual region形成compact loops/tails；旧
  per-node/global-max Cartesian、fixed seed和allocation-feedback owner在能力迁移后删除。

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
execution在parent temporal point关闭后，经D `NestedExecutionRelation`得到实际producer work；offset/extent/relation extensionally相同的
invocations合并为一个`NestedInvocationClassId`，main、各tail及halo/boundary差异保留不同classes，不按wave ordinal逐个复制state。
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
`FullExtentOnly`为singleton；Q51 partial state逐个定点或split interval，complete state才含concrete vector。target geometry、wave/tail、
transaction、layout footprint和feasibility lower bound只能产生proposal breakpoints；interval内其它integer siblings始终可达，除非
Q50.F/Q51对完整依赖坐标给出sound equivalence或bound proof。

### E-2 专项调研：interval successor、order枚举、proposal与actual loops

Halide autoscheduler和Ansor都把fusion/compute placement等结构选择与concrete tile annotations分层，并用tree/beam、cost model或
evolutionary search处理巨大schedule space；它们也明确承认只探索其定义的schedule子集。该经验适合本仓的proposal排序和hierarchical
state，却不能证明Wafer完整temporal合同。E的exact successor因此覆盖全部positive integer points，学习/beam/divisor/native points
只能作为可关闭的proposal provider。

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
  union extensionally identical (offset, extent, relation, effect-scope) classes
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

proposal point可来自wave-count/tail等价类、divisors、native issued geometry、alignment/transaction边界、A/D relation shape changes、
representation/footprint lower bound和已知buffer feasibility；provider只返回普通integer points及可解释priority，不返回allowed-set。
移除任一provider后midpoint successor仍到达全部points。

跨scope proposal使用dependency factor graph：parent/nested relation和shared D version形成compatibility factors，known work/footprint形成
lower-bound factors。chain/tree在已产生的proposal points上做k-best DP，一般低treewidth图可做variable elimination，其余由Q51
best-first interval search。若每scope当前有`K`个proposal points，chain为`O(NK^2)`、treewidth`w`为`O(NK^(w+1))`；这里`K`只是
本轮proposal work量，不是domain cap。预算耗尽停止proposal生成并报告coverage，exact interval children仍存在；无fixed top-k、beam
legality、per-scope winner或默认统计。

#### Baseline deterministic greedy

baseline复用相同`TemporalScope`、positive-size successor、exact-tail builder和domain-descriptor memo，但不构造E/Q51 frontier：

```text
chooseBaselineTemporalPlan(scopes):
  start every scope at full exact extent and canonical structural order
  while the policy-free feasibility query returns a typed exact rejection:
    obtain only causally implicated scopes from the rejection witness
    for every refinable axis derive its next smaller legal wave-count point
    rank by guaranteed capacity relief / added waves, then semantic tie-break
    update exactly one plan coordinate
  return first fully closed plan
  if all implicated axes are one: return typed capacity failure
```

这是`none`的单调first-fit贪心：不回溯、不保存alternative、不调用search，也不把first-fit反写成E合法域。若某轴没有可证明的wave-class
relief，按stable axis逐integer下降仍严格趋向1并终止。循环只处理typed plans和Q50.F最终可重算的feasibility；deferred或indeterminate
直接作为typed baseline failure，不用actual IR探测。first-fit plan关闭后才进入C的一次CardModule commit和Q50.0，commit失败不回到
greedy；因此没有失败coordinate materialization、clone或replay。extensionally相同Tile scopes共享query计算，但各自的causal
assignment可不同。current Q49.P若仍靠Q50.0 rejection推进，必须在F closed-plan core接管canonical feasibility时立即迁到本终态，
不能等待完整search-domain F closure。

#### Winner-only actual emission

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
- Q49.P greedy与E共享scope derivation、legal next-size和actual loop builder；overfull→fit、多轴/tail/all-ones failure稳定，baseline路径
  无search domain/frontier、无failed-coordinate IR，first-fit只commit一次且不剪search；
- test-only/winner actual IR检查selected nesting、compact main/tail、nested producer child loops、multi-reduction state和无post-hoc retile；
  planning零IR materialization、clone/replay；
- source gate删除`TemporalNodeAssignment/Domain`、node-wide `StructuredOpTemporalTile`、shared-ceil extent、fixed seed、preferred-size
  legality、allocation feedback和operation-pointer lookup；旧witness全部迁入new scope owner后才删source/tests。

### Current/donor能力迁移

| Current / donor能力 | 终态owner | 保留或修正的witness | 退役条件 |
| --- | --- | --- | --- |
| current every-positive-size/permutation successor | E per-scope interval leaves/Kahn order successor | 2x3、multi-axis、per-Tile remainder与nested-child reference equality | 删除node-wide domain且不预建Cartesian vectors |
| current one-wave order canonicalization | E active iterator key | scalar/one-wave无重复、active order完整 | 同一逻辑保留在new type |
| current `getCapacityGuidedAssignment`/TemporalTileShape | baseline greedy + E proposal provider | proposal在domain、overfull-to-fit | 不作为search default/winner或hard legality |
| current baseline `setCardBaselineTemporalTiles` | shared per-scope query + Q49.P controller | deterministic first-fit、per-Tile remainder、output correctness | node/output axis恢复、shared-ceil extent与第二套vector删除 |
| current `StructuredOperationTileFootprint` | Q50.F foundation，E只消费bound | affine window/physical estimate正负例 | estimate不在E签发capacity legality |
| current TemporalWave/Region/PartialReduction traversal | C/D/E prepared actual emitter | parallel/reduction/multi-axis/tail/init/nested child | apply不更新既有CardModule、不选tile、不递归fuse-all |
| current actual-only reduction order check | E structural precedence query + same emitter verifier | query/apply parity、multi-reduction all legal structural orders | `ReductionSemantics`或其它late second legality gate不在E调用链 |
| old rank/connection tile sizes、preferred points | E proposal facts | independent/nested execution intent由D/E表达 | action recipe、rank provider、local cap删除 |
| old allocator feedback/binary endpoint/temporal depth | Q49.P monotonic controller或Q50.F/Q51 bound | causal capacity witness、siblings preserved | feedback history/accepted retile/shortlist零残留 |

“小tile使fusion/buffering成为winner”及任何需要actual layout/buffer/resource cost的比较统一放在Q51 closure。

## Q50.F：Scoped Feasibility Analysis

### Pipeline contract

Pipeline position:
- Upstream IR / input:
  immutable normalized TensorProgram、显式target facts及typed plan components。最早的closed-plan core消费Q49.P从现有baseline迁出的
  canonical spatial/singleton-region/temporal/representation/movement/single-slot/Serialized/order facts；partial foundation消费
  Q50.S fixed semantic facts与B/D/E prefix；full closure消费fixed semantic facts与B–K→I→J完整search plan。三种入口都没有actual CardModule或offset。
- Current stage responsibility:
  先建立所有入口共用的resource problem schema、checked arithmetic、validated placement/status与plan/actual parity；closed canonical
  plan可立即获得FullFeasibilityProof或typed rejection以服务baseline。随后foundation从current IR/state识别**当前Core合同中已经存在**的尚缺坐标，计算individual value、DPS/alias/effect、D execution/version和E temporal scope能
  证明的mandatory-live resource lower bounds及non-binding estimates；G--K闭合后从完整typed plan构造SPM/DDR packing、DTE/FSM、
  message/descriptor、worker/event及target field-limit问题，并只凭verified witness或完整proof关闭full-coordinate feasibility。返回
  consistent、deferred、exact rejection、unsupported或indeterminate，exact witness绑定所有实际读取的坐标。
- Output IR / files:
  不写IR或文件；返回typed `ScopedFeasibilityResult`。full-coordinate consistent结果含无offset的`FullFeasibilityProof`；query结束后
  不留下operation pointer cache、scratch module、solver state或materialized candidate，candidate assignment也不保存offset/witness。
- Downstream consumer:
  Q49.P在closed-plan core就位后立即用同一query推进plan-only temporal greedy并只commit一次；它不等待或调用完整search domains。
  Q51 partial-state transition只对`Deferred`展开明确缺失的Q50.G–K坐标，只对`ExactRejection`使用同一causal key；complete planning
  assignment由Q51 cost/bound比较，只有最终winner进入Q50.0。
- User-level driver / named pipeline:
  public `search` session和Q49.P deterministic controller静态调用同一个policy-free query；`none`不调用search domain/controller，
  只把F关闭后的first-fit plan交给一次C/Q50.0 commit。
- Explicit non-goals:
  不选择layout、movement、buffer、route、worker或schedule；不clone/lower IR。foundation不调用packer，full closure只对全部坐标已关闭的
  contender运行pure resource solver并验证witness，不写actual offset、不从estimate签发legality、不修改parent state，不生成
  retile/spill/rebuffer sibling，也不把heuristic no-fit或“未发现矛盾”称为完整feasible。
- Done criteria:
  closed-plan core先以baseline canonical case证明pure query、validated witness/status和actual problem parity，planning CardModule/Q50.0为零；
  partial/full层再证明自动缺坐标、unsupported、resource exhaustion、single-value与mandatory-coexistence exact overflow可区分；
  estimate overfull不能reject；完整plan只有validated placement/resource witness才获得full proof，proven infeasible才reject，solver耗尽/
  heuristic no-fit保持indeterminate；exact rejection绑定完整causal key且不同temporal/region及required-coordinate siblings仍可达；无
  manual epoch、clone/probe、allocator repair或feedback beam。

施工期不得让F为尚未实现的G/H/I/K/J预声明`RequiredCoordinate` enum/nullable field。E后的首个F-partial只返回A–E能证明的
Consistent/Exact/Unsupported/Indeterminate facts；Core自身因next axis尚未接入返回`IncompletePlanningDomain`。G随后交付时同批增加
RepresentationState及F对该真实coordinate的Deferred/observed dependency，H/I/K/J依次同理。最终文档可以展示完整
`RequiredCoordinate` sum，但current代码每次只包含已有producer和consumer。

### F-core：closed-plan resource problem与baseline纵向

canonical-feasibility-proof不是缩小版search，也不预埋尚未实现的axis field。前序semantic/spatial/demand work items提供normalized input；
canonical plan work items把baseline必需事实迁到最终policy-free component types，attention-work-projection再补resource facts。
deterministic-baseline-closure不生产这些对象，只在canonical-feasibility-proof和attention-selected-decomposition之后消费。每个component同时提供一个窄的resource description overload：

```text
describeResources(SpatialPlan, ExactDemandProof)
describeResources(RegionPlan, TemporalPlan)
describeResources(RepresentationPlan, MovementPlan)
describeResources(BufferPlan, ExecutionStructurePlan, ClosedSchedulePlan)
```

baseline与search共同读取Q50.S-normalized TensorProgram。root execution component直接引用current semantic root；attention algorithm
从op读取，不存在空algorithm field、`SemanticRootExecution(algorithm)` sibling或nullable algorithm占位。两条policy仍由各自controller
构造不同physical plan，不共享plan或executable。

这些overload只描述该component确定拥有的storage、lifetime、message、event、worker和field requirements，不选择其它component。
F-core规范化并组合它们，运行与F full closure相同的problem validator、fast witness与exact solver status映射；Q50.0从selected actual IR
调用同一normalizer做parity和offset assignment。Q50.S新增fixed semantic action/resource description时扩同一problem schema；
后续Q50.B–K加入新合法choice时扩对应component domain及resource description，
不创建第二个baseline problem schema。

施工顺序避免循环依赖：先定义component type和canonical constructor，再由attention-work-projection/各owner定义resource description，
之后canonical-feasibility-proof关闭plan，attention-selected-decomposition准备builder，最后baseline controller消费。canonical proof
不能调用full-feasibility oracle，也不能调用baseline controller。完成门禁为：overfull-to-fit每个
coordinate零IR、FullFeasibilityProof后CardModule/Q50.0各一次、actual normalized problems逐semantic ID相等、Q50.0 failure不返回first-fit循环。
F整个任务仍保持`queued`，直到partial foundation、K→I→J后的F full closure和独立oracle全部完成；F-core通过不冒充Q50.F完成。

### F-foundation-1 专项调研：结果分类、依赖坐标与scope

MLIR One-Shot Bufferize采用analysis→rewrite两阶段，并通过interface区分exact alias/equivalence与unknown；unknown不能被当成must-alias或
no-alias。成熟constraint solver也严格区分proven infeasible与因time/memory limit停止的unknown，model invalid则是独立的输入合同错误。
MLIR Transform又区分尚未mutation的recoverable precondition failure与mutation后的irrecoverable failure。这些实践共同说明：F不能用
一个`bool fits`、`LowerBound`状态或diagnostic字符串承载控制流，也不能把“缺选择”“没有实现”“没算完”和“已证明冲突”合并。

current `ScopedFeasibility`仍由caller传`unresolvedCoordinates`，只要caller漏一项就会把partial state误标`LowerBound`；结果是
kind+reason+多个optional fields组成的隐式状态机，unsupported footprint又被归入Indeterminate。它还按node/trial和node-wide temporal
vector建立key，而不是D/E的execution/version/scope。终态直接替换为closed typed sum：

```text
ScopedFeasibilityResult =
  Consistent {
    facts: FeasibilityFacts
    observed: FeasibilityDependencyKey
  }
  | Deferred {
    facts: FeasibilityFacts
    required: NonEmptyVector<RequiredCoordinate>
  }
  | ExactRejection {
    witness: FeasibilityWitness
  }
  | Unsupported {
    feature: UnsupportedFeasibilityFeature
  }
  | Indeterminate {
    cause: FeasibilityIndeterminateCause
    partialFacts: FeasibilityFacts
  }

FeasibilityFacts
  lowerBounds: ResourceLowerBound[]
  estimates: NonBindingEstimate[]

RequiredCoordinate =
  RepresentationFor(ValueVersionId)
  | MovementFor(BoundaryId)
  | BufferingFor(ValueVersionId | MovementId | TraversalScopeId)
  | ExecutionStructureFor(RegionGroupId)
  | EventScheduleFor(TileId | RegionGroupId)

FeasibilityDependencyKey
  semantics: SemanticRootKey and fixed root facts actually read
  spatial: selected shard/owner/merge choices actually read
  regions: selected execution/version/use-binding choices actually read
  temporal: selected TraversalScopeId points/orders actually read
  physical: assigned G--K choices actually read, if present
  target: exact immutable resource facts actually read
```

`Consistent`只表示本scope所有**当前可执行的检查**无矛盾且没有这些检查所需的缺失坐标；它从不命名`Feasible`，不授权winner
commit。`Deferred`只表示一个或多个合法planning coordinates尚未赋值；required是scope精确的typed objects，不是enum+nullable scope。
`Unsupported`表示source/interface/target mechanism没有current分析合同；`Indeterminate`只表示work limit、analysis precision或外部
资源使本轮无法下结论。malformed state、impossible ID或verifier-valid输入下的内部不一致经`FailureOr`/diagnostic作为compiler bug
失败，不伪装成Indeterminate。除ExactRejection外的结果不能形成no-good。

missing-coordinate discovery是F静态代码的一部分，不是caller list、provider registry或optional bag：

```text
collectRequirements(scope, state, checks):
  for each check in stable semantic order:
    inspect the exact typed inputs that check needs
    if a value has no RepresentationPlan: add RepresentationFor(value)
    if a boundary has no MovementPlan: add MovementFor(boundary)
    if a live version/transfer has no BufferPlan: add BufferingFor(owner)
    if region execution structure is unresolved: add ExecutionStructureFor(region)
    if resource/effect order is unresolved: add EventScheduleFor(scope)
  deduplicate by typed identity and stable-sort
```

foundation不会机械要求全部G--K：例如一个已证明单value minimum超过capacity可在representation缺失时直接reject；反之某项lower bound
确实依赖layout时只要求该`ValueVersionId`的representation，不把整个Card标missing。后续每个F closure扩展只在同一visitor中增加其
实际消费的typed plan，不新增平行missing API。

结果组合顺序同样明确：先验证state合同；再收集能独立形成的exact witness；一个closed witness即使其它检查deferred也可返回
ExactRejection。没有rejection时，只对当前坐标下必经的检查报告Unsupported/Indeterminate；被缺失coordinate阻挡的检查进入
Deferred，不预判其未来路径。最后required为空才返回Consistent。所有result及witness按semantic ID稳定排序，不依赖DenseMap、pointer
或并行完成顺序。

scope使用typed sum而非模糊“node”字段：foundation最窄检查单位为`TraversalScopeId`或`RegionGroupId`，full closure再加入Tile/Card
resource scope。func-scoped MLIR analyses只提供current SSA、IndexRelation、DPS/Bufferizable/ViewLike/effect事实；assignment-local
query组合plan。跨scope结论没有闭合时返回精确required coordinate。cache只观察结论实际读取的typed choices和immutable target facts，
并由planning session按`ObservedDependencyKey`失效；source-only pass cache才依赖AnalysisManager的preserve/invalidate。两者都不使用
manual epoch/fingerprint。默认路径不记录统计或日志；显式instrumentation
可观察query/work count但不进入result或控制流。

### F-foundation-2 专项调研：minimum storage与interference certificates

LLVM register allocation先从live intervals建立interference：有edge的values不能占同一register；clique给出必要register-pressure下界，
但一般graph里clique bound不等于可color/可allocate。MLIR BufferizableOpInterface同样明确区分equivalent、unknown alias和read/write
effects。仓库current MiniMalloc adapter已有一项值得迁移的能力：在进入solver前从exact conflict graph构造deterministic greedy
edge-clique cover，任一over-capacity clique都可独立证明infeasible。应提取的是这个pure certificate principle，不是把MiniMalloc、
packing search或其默认百万级node budget搬进planning。

foundation按`TraversalScopeId`/`RegionGroupId`建立logical minimum storage objects：

```text
StorageDemand
  id: StorageDemandId
  owner: RegionValueVersionId | TraversalStateId | BoundaryId
  exactDomain: ExactIndexSet
  minimumFootprint: Proven(bytes, minimumAlignment)
                  | Requires(RepresentationFor(value))
                  | Unsupported(feature)
  lifetimeRequirement: MandatoryLiveSegment[]

MandatoryInterferenceEdge
  lhs, rhs: StorageObjectId
  proof: MustSeparateProof + MandatoryOverlapProof
  dependencies: FeasibilityDependencyKey

CapacityCertificate
  scope: FeasibilityScopeId
  objects: NonEmptyVector<StorageObjectId>
  minimumBytes: uint64
  pairProofs: MandatoryInterferenceEdge[]
  capacityBytes: uint64
```

`minimumFootprint`不是“默认dense layout”：exact element count、existing element storage和immutable target facts先给出所有legal
representations共同的minimum；若target representation capability不能证明共同下界，就只返回scope精确的Representation requirement。
padding、alignment和bank fragmentation只有在所有legal representations都强制时才进入minimum，否则留在estimate或G/F closure。
checked arithmetic可直接证明“已经超过finite capacity”时保留exact above-capacity witness；普通overflow不能用saturation冒充具体bytes。

alias处理只使用标准/typed强结论：

- MustAlias/Equivalent values union为一个minimum storage object，bytes取成员maximum；
- MustSeparate且存在mandatory overlap时形成interference edge；
- MayAlias/Unknown既不union、也不为reject相加，而是保留partial fact并在更强检查需要时要求Buffering/Representation坐标；
- DPS input/result是否可in-place由DestinationStyle/Bufferizable contract决定，不能按operand/result位置猜测。

`MandatoryOverlapProof`只描述任何future G--K choice都无法消除的重叠，例如同一selected leaf/interface明确要求同时驻留的distinct
objects、loop-carried state与当前step不可alias inputs，或D stored version跨越一个必经use且另一object在该use同时必需。是否能stream
inputs、逐个merge contributions、调整event order或复用slot若尚未选择，就不建edge，返回相应required coordinate。root类型或某个
case的state tuple不硬编码进F；semantic op若确有mandatory simultaneous state，由其typed interface报告普通storage group。

默认算法刻意不求maximum weighted clique：

```text
analyzeFoundationStorage(scope, state):
  demands = deriveStorageDemands(A/C/D/E and immutable target facts)
  objects = unionOnlyMustAlias(demands)
  check every proven single-object minimum against capacity
  edges, explicitGroups = deriveMandatoryInterference(objects, state)
  check every explicit simultaneous group

  for each connected component in stable order:
    for each uncovered edge (u,v) within the polynomial work allowance:
      clique = {u,v}
      candidates = commonNeighbors(u,v)
      greedily add candidates by (minimumBytes desc, StorageObjectId)
          only when adjacent to every clique member
      record clique lower bound and stop immediately if it exceeds capacity
      mark its covered edges

  facts.storageLowerBound = maximum found certificate weight
  facts.certificateCoverage = CompleteForExplicitGroups | PartialCliqueSearch
  return combineResult(facts, automaticallyDerivedRequirements)
```

任一输出clique内所有objects两两不能共享固定address range，因此其ranges必须pairwise disjoint，`sum(minimumBytes)`是capacity必要下界；
忽略alignment只会减小该值。只要和大于capacity就能ExactRejection，不要求clique为maximum。相反，greedy没找到over-capacity clique
绝不证明packing可行；`Consistent`只携partial lower-bound coverage。默认work allowance随当前scope的`V+E`受控，耗尽只停止这个
**可选strengthening**并保留Partial facts，不使普通compile失败或启动更重solver。完整Bron--Kerbosch/branch-and-bound只用于2--10
object有界穷举oracle或显式诊断，不进入默认编译路径、返回类型或semantic control。

该默认搜索使用bitset common-neighbor intersection时，建图`O(V+E)`，处理`q`个seed edges最坏`O(q*V^2/wordSize)`且`q`受scope work
allowance限制；内存`O(V^2/wordSize + E)`。single object和explicit groups为线性。full maximum-clique oracle仍为指数级，只验证
certificate coverage而不为production提供结论。

non-binding estimate可复用current affine tile footprint，但必须携带`knowledge/reason/dependencies`并与lowerBounds分栏；它只供proposal
排序。alignment、bank/offset fragmentation、selected lifetime和actual packing尚未闭合，foundation Consistent不保证winner可pack。

D `NestedExecutionRelation`在E concrete scopes下的exact compatibility是另一项cheap foundation check：若selected nested instance无
exact producer work/result tile，witness只绑定该D execution/use与E parent/child temporal points；不能把整个D group、其它E siblings
或boundary alternative设为no-good。

analysis不选择下一tile、不修改state，也不clone/lower/spill/retile/rebuffer IR。Q51只消费typed requirements和exact witness；
source-only pass analysis由AnalysisManager失效，assignment/target相关memo由planning session按observed typed choices失效；两种
cache都不保存materialized IR、diagnostic字符串、optional solver state或估算bool。

### Gate

- unit覆盖single oversize、explicit pair/triple simultaneous group、greedy weighted clique、MustAlias max-once、MustSeparate、MayAlias/
  Unknown不相加、D top-level/nested/replica versions、stored/shared boundary、loop-carried state和merge streaming deferred；certificate逐
  object列出minimum bytes、pair proof、scope与dependency key；
- 每个production certificate由independent有界穷举address packer确认infeasible；反向不要求foundation发现全部infeasible
  problems。random 2--10 object graphs中每个reported clique都真是clique且weight正确，maximum-clique oracle只比较lower-bound strength；
- work allowance 0/边界/耗尽仅把coverage标Partial，不产生rejection或compile failure；显式single/group certificate仍可零搜索reject，
  default call graph不含Bron--Kerbosch、MiniMalloc或packing solver；
- estimate-overfull但无mandatory interference保持Deferred/Consistent；foundation Consistent而tiny actual pack失败证明它不签发feasible；
  representation共同minimum未知时只要求精确ValueVersionId，不使用dense default；
- required coordinates由partial state自动推导，删除任一G--K coordinate产生精确scope项；public API无caller list；Unsupported、
  Indeterminate和Partial lower bound不进入no-good；
- nested+temporal incompatibility只拒绝完整D+E causal combination，更小tile、top-level execution、split replica或region boundary siblings
  保留；
- input/root/use/hash顺序变化得到相同facts/certificate；IR mutation使analysis失效；source byte-identical，planning无clone/probe/
  packer/actual owner/default statistics；Q49.P与Q51各自只把selected plan送一次Q50.0。

### Current/donor能力迁移

| Current / donor能力 | 终态owner | 保留或修正的witness | 退役条件 |
| --- | --- | --- | --- |
| current max single operand/result logical bytes | F `StorageDemand`/size-one certificate | single-object overflow与smaller temporal sibling | 从node/trial迁为execution/version/scope demand，不套dense default |
| current affine physical residency estimate | F NonBindingEstimate | overfull estimate不能reject、known/unknown reason | 不混入lower-bound field或bool |
| current `LowerBound` kind、reason+optional状态机 | F-foundation-1 closed typed result | 五种结果、compiler-bug与result precedence tests | old enum/reason/optional protocol零残留 |
| current caller-supplied missing coordinate list | F automatic dependency inspection | 每类G–K缺项与scope stable ordering | public API删除`unresolvedCoordinates`/`FeasibilityCoordinate` |
| old region/function `SPMCapacityEvaluation` clones/probes | 不恢复；pure bound归F，final exact packing归Q50.0 | Fits/overflow/unsupported/failure分类与atomicity意图 | 无probe API、private Module、clone-remapped relation或phase label控制流 |
| old StructuredBufferRelations probe/final remap | actual IR relations由C/H与memory planning拥有 | stale/foreign relation负例 | planning query不remap clone pointers |
| MiniMalloc individual oversize与greedy edge-clique certificate | 提取pure `CapacityCertificate` helper；actual solve仍09/Q50.0 | size-one/weighted clique、nonzero arena与deterministic owners | helper不include/callMiniMalloc，不分配offset，不把solver budget带进F |
| old allocator feedback/lookahead/temporal probe | E baseline greedy或F/Q51 bound | siblings preserved、causal key | feedback depth/streak/shortlist/accepted retile零残留 |

source清理门禁还包括old `ScopedFeasibilityAssignmentKey`的node/group/raw-vector identity、`getNodeLowerBound` max-only实现、manual epoch
checks及probe terminology零残留；current tests先逐项迁入new typed API和certificate oracle，不能因旧source未再被production调用就删除。

## Q50.G：Layout and Physical Representation

### Pipeline contract

Pipeline position:
- Upstream IR / input:
  immutable normalized TensorProgram/algorithms、Q50.A exact value requirements、Q50.B/D/E spatial/region/traversal plans、Q50.F
  foundation facts及current Wafer physical encoding/access interfaces；没有selected movement、buffer或schedule。
- Current stage responsibility:
  为每个logical value/version选择唯一primary representation，为actual use requirements选择primary或可共享derived versions；只枚举
  encoding/access interface对exact logical domain可解释的layouts。建立operation tuple、SSA/version、view/alias、external boundary、
  reduction和future movement的constraint/cost factor graph；deterministic solver产生proposals/bounds，exact domain仍由typed
  variables完整表示。winner emitter一次物化selected primary/derived versions。
- Output IR / files:
  query不写IR，Q51 state保存typed `RepresentationPlan`；solver reduction records/cost matrices保持query-local。winner actual
  CardModule用MemoryAttr/typed layout ops和SSA uses自包含表示selected versions，不携solver residue、proposal ordinal或score。
- Downstream consumer:
  Q50.H以producer primary version、consumer operand version和exact demand生成movement/conversion组合；Q50.F footprint、Q50.I lifetime及
  Q50.J calendar在representation变化后重算。Q51联合其它轴选择winner，Q50.0只验证winner actual IR。
- User-level driver / named pipeline:
  Q51 planning session；`none`仍由Q49.P current deterministic representation构造，不进入本域枚举。
- Explicit non-goals:
  不选择route、spill、recompute、retention、buffer、worker或全局winner；layout solver不得clone/apply IR或用固定Top-k永久截断全局合法域，
  不把layout写回logical exact-demand或edge action，不让每个operand use各自覆盖producer primary identity。
- Done criteria:
  domain与independent value-version/use-binding reference一致；solver对chain/diamond/fanout/view/reduction/collective与movement lower-cost
  小图达到reference optimum并报告coverage；unsupported encoding无state；winner primary/derived conversion、output writeback、fanout
  secondary sharing及atomic failure有actual证据。旧PBQP donor的constraint、cost、reduction、shared-secondary和tests逐项迁入后，
  fixed Top-4/local winner/clone apply保持删除。

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
physical footprint、compute implementation penalty和knowledge。未知term不能当0参与winner比较：它对bound取sound lower value并
标记knowledge，完整cost由H–J补齐。solver只生成proposal/bound，Q51 exact state仍能逐variable访问全部legal assignments。

### 约束求解算法

这里采用有限constraint graph，PBQP是其中pairwise部分的求解形式，不把LLVM register allocator的启发式结论直接当成全局最优。
LLVM pinned实现的R0/R1/R2分别删除degree 0/1/2节点、把被删节点的最小代价传给相邻节点，并按reduction stack反向恢复选择；它在
高阶core上允许启发式断边，这适合register allocation，但不能为本任务证明完整搜索。Q50.G只复用前三种等价变换和反向恢复结构；
高阶core必须由可证明的branch-and-bound处理，或者明确返回预算未完成，不能伪装成最优。

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

### G-3 专项调研：independent oracle与winner-only physical-version apply

OpenXLA Layout Assignment先在logical graph上传播完整layout constraints，冲突或graph endpoints再显式插入copy；physical layout成为HLO
shape的一部分，下游直接读取，不由每个consumer临时挑选。MLIR One-Shot Bufferize同样先基于完整SSA做analysis，再统一rewrite并用
explicit materialization/copy表达out-of-place结果。两者共同支持本仓G的apply边界：`RepresentationPlan`先关闭，winner construction
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
piece coverage、footprint、alignment、padding和element spans。它不选encoding、不比较cost，也不靠generic Presburger试错；unsupported
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
- planning/solver counters证明零IR mutation/materialization。production winner中version/conversion count只由plan决定，关闭observer后IR
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
  winner apply才生成load/store、send/recv/token/wait及relay work；compute replica和layout versions分别已由D/G决定。
- Output IR / files:
  query不写IR，输出typed movement assignment及可重算reuse/cost/resource facts；winner apply输出current CardModule中的explicit
  movement/event-producing ops，不写sidecar或route attr bag。explicit relay graph在commit后由逐transfer ops表达；只有target真的
  暴露programmable route时，routing才是typed assignment并由target IR合同消费。
- Downstream consumer:
  Q50.I从planned waves/movement completion推导slot domain，Q50.J从planned send/recv/await、compute与resource effects构造calendar；
  selected winner最终由Q50.0做message matching、completion、SPM/DDR和runtime gate。
- User-level driver / named pipeline:
  Q51 planning session；`none`继续使用其确定性DDR/peer functional carrier，不进入本域枚举。
- Explicit non-goals:
  不选择compute replica、layout conversion、buffer count、worker、issue order、pipeline或winner；不把software relay path称作raw NoC
  route，不按bytes/hops冻结最大broadcast，不缓存actual IR，不保留旧rank/provider proposal或implicit cross-region SPM residency。
- Done criteria:
  same-region external boundary、same-Tile cross-region DDR、cross-Tile DDR/direct peer/relay、multi-piece partial overlap、unicast/partial/
  maximal multicast、fanout、gather及supported collective均有winner actual witness；coverage/endpoint/relay/message/completion负例受typed
  validation；opaque target routing不伪造per-link load。旧complete-rank/global-relation/NoC provider中的topology、owner rotation、
  tree/ring、alias/lifetime/slice proof和tests逐项迁入current query/apply后，旧owner才能退役。

### H-1 专项调研：transport topology、movement boundary与payload合同

MLIR async合同要求所有依赖显式进入token/value，且“可并发”不保证实际并发；因此send/recv token、consumer visibility和buffer release
必须分别表达，不能由route顺序暗示。经典Dally--Seitz NoC理论用channel-dependency/virtual-channel证明router-level deadlock freedom；
NCCL graph search和TACCL则从actual topology选择ring/tree/communication sketch。这些工作都先区分**network route**与**collective/
forwarding algorithm**。本仓current `SimpleRoute`却枚举Tile邻接simple paths，再在每个中间Tile物化recv→send；它实际是software relay
path，不是可编程router path，命名和cost语义都必须纠正。

仓库硬件/IR事实进一步限定边界：current target为4x4 Tile grid，NoC有四方向独立收/发带宽；`wafer.tile.peer_*`和Direct DTE只携
任意available physical peer、bytes、message与token，明确不携raw route。Kcore DTE目的地址直接编码remote Tile，现有public accepted
subset是fixed-size unicast；raw broadcast/scatter寄存器有证据，但public-helper/板端completion尚未闭合，不能作为production
capability。`TargetTopology::getCanonicalOnCardPath`自身也声明只是modeling fact，不是target routing claim。因此current direct peer
transfer是end-to-end target-routed action；只有显式在中间Tile接收、保存并再次发送时才存在compiler-selected relay graph。

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
  boundaries: BoundaryMovementPlan[]
  collectives: CollectiveMovementPlan[]

BoundaryMovementPlan
  boundary: MovementBoundaryId
  requiredCoverage: ExactIndexSet
  destinationVersion: PhysicalVersionId
  payloads: PayloadPlan[]
  realization: DDRStagePlan | PeerTransferGraph | QualifiedTargetCollective

PayloadPlan
  id: PayloadId
  logicalValue: RegionValueVersionId
  logicalDomain: ExactIndexSet
  sourceVersion: PhysicalVersionId
  destinationVersion: PhysicalVersionId
  sourcePhysicalSpans: PhysicalSpan[]
  destinationPhysicalSpans: PhysicalSpan[]
  physicalBytes: uint64

PeerTransferGraph
  transfers: DirectPeerTransfer[]
  relayVersions: RelayPhysicalVersion[]
  completionObligations: TransferCompletionId[]

DirectPeerTransfer
  source, destination: TileId
  payload: PayloadId
  routing: TargetRouted | QualifiedProgrammableRoute
```

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
五类failure；classic route/reuse算法、winner apply和donor迁移分别由H-2/H-3闭合。

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
| [Dally--Seitz](https://authors.library.caltech.edu/records/fd0yr-br438) | 用channel-dependency graph刻画router-level deadlock | 仅在target暴露programmable/deterministic route及VC/channel事实时验证raw route | current Direct DTE的内部route不透明；software send/recv死锁由J的event/wait graph证明，不能套XY/DOR猜硬件 |

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
   只有future exact route/capacity facts才能声称link-disjoint或bandwidth-optimal；
4. topology-derived Hamiltonian/serpentine ring及反向ring，ring reduce-scatter+all-gather和pipelined ring；存在unavailable Tile时必须从
   induced participant graph重新构造，不能硬编码16-Tile ring；
5. recursive doubling、recursive halving+doubling及非2次幂pre/post pairing；这些构造器读participant集合和typed spans，不读rank名字；
6. 规则或可证明factorized participant subgraph上的row-then-column/column-then-row broadcast、gather、all-gather和all-to-all，另含
   row/column panel fanout proposal；M/N/K或operator名字不是触发条件，触发事实是需求图与physical embedding；
7. Bruck-style aggregate exchange与round-robin/pairwise all-to-all；只有selected contiguous bundle或显式pack/unpack work完整时接纳；
8. TACCL-style in-process typed sketch（allowed endpoint edges、proven symmetry、chunk class）驱动bounded suggestion，以及cut-aware source
   rotation、reuse-aware partial multicast；没有外部JSON/sidecar或default MILP/SMT。

proposal bank不会先按payload大小硬选一个family，也不产生local winner。message count、injected bytes、relay copies、critical dependency
depth、endpoint issue pressure、SPM staging和可证明topology bounds一起交给J/F/Q51；同一个workload的不同boundary可选择不同family。

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

### H-3 专项调研：winner-only movement IR、effect/lifetime proof与donor迁移

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

任一prepare/emit/verify失败由C新CardModule subtree guard整体擦除，source保持不变；winner commit bug不返回H domain换plan。default path
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
  winner apply才建立allocations、views、loop-carried slot rotation和release。
- Output IR / files:
  query不写IR，输出typed storage/slot bindings及schedule requirements；derived lifetime/interference/footprint facts可重算。winner actual
  IR用memref SSA roots/views、SCF iter args和dealloc/release自包含表示，不写buffer recipe attr。query结果不含IR pointer、Location、
  shadow timestamps或默认统计。
- Downstream consumer:
  Q50.J foundation从D--I计划建立event/resource DAG；Q50.K选择serialized/pipelined execution structure。K若改变occurrence/lifetime结构，
  对应I/J choices失效并由Q51重算；J closure最终关闭wait/order/release。winner进入一次Q50.0 SPM/message/completion gate。
- User-level driver / named pipeline:
  Q51 planning session；`none`保持serialized且不枚举buffer域。
- Explicit non-goals:
  不选择worker、issue order、stage pipeline、route、winner或overlap收益，不clone/lower partial state，不把slot count写回logical
  boundary/version，不用3作为永久上限，不从schedule estimate或materializer失败推断parent spatial/temporal rejection。
- Done criteria:
  exact domain覆盖fresh/reused/in-place storage、single/double/triple及finite upper bound内更多slots、不同rotation coordinates和required
  orderings；independent oracle一致。actual double/triple/multi-slot、tail、Direct-DTE issue/wait、alias和common-occurrence正负例通过；
  旧edge `bufferCount`、actual-loop discovery、local winner和fixed `[2,3]`上限删除；K/J invalidation与联合proposal有production consumer。

### I-1 专项调研：storage binding、lifetime requirements与slot-family domain

MLIR One-Shot Bufferize先分析完整SSA alias/read-write conflict，再统一rewrite；它不会在每个use处临时决定buffer。XLA HeapSimulator则在
已有logical schedule后用明确start/end live intervals分配chunks。rotating-register/modulo-scheduling工作进一步说明slot multiplicity、
iteration overlap和release schedule必须联合。对应本仓，I位于J/K之前时只能保存storage/rotation choices与必要ordering constraints，
不能虚构时间戳或把current source order当最终lifetime。

current `CardBufferingDomain`按`(group nodes, retained edge subset, slotCount)`建域，只取所有nodes的最大steady trip count和单edge
footprint；真正的common loop、alias、endpoint和release拖到1,900行`SelectedBufferMaterialization`在actual Instr中搜索。materializer又
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
  slotFamily: optional SlotFamilyId

SlotFamilyPlan
  id: SlotFamilyId
  occurrence: OccurrenceRelationId
  members: PhysicalVersionId[]
  multiplicity: PositiveInteger
  rotation: SlotIndexExpr(iteration coordinates) mod multiplicity

BufferOrderRequirement
  define/issue/use/completion/release EventIds
  relation: MustPrecede | SameOccurrence | ReuseAfterCompletion
```

候选只保存typed IDs、binding、multiplicity和rotation choice；physical type、footprint、live relation和event set从G/H/E/current target facts
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
capacityUpper = floor(tile SPM capacity / proven minimum single-slot bytes)
targetUpper = largest multiplicity representable by loop/selector/resource types
U = min(tripUpper, capacityUpper, targetUpper)
```

其它mandatory buffers尚未扣除只会让`capacityUpper`偏大，仍是safe finite upper bound；full feasibility由F closure。footprint/extent
算不出则Deferred/Unsupported，不用1或3默认。domain含每个`1..U`和每个interface-valid recurrence-coordinate/linearization；tail按同一
modulo mapping进入安全slot并由J release proof关闭。

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
slots和capacity-aware multi-slot；每个proposal经exact constructor验证，不能成为local winner或删域。

K选择新的stage pipeline/iteration overlap后，如果`OccurrenceRelation`、event set或reuse distance改变，affected I family及J facts必须
失效，Q51按依赖重新展开I→J；不能让K把旧multiplicity解释成另一条pipeline。反过来I multiplicity/order requirements约束K/J，但不
预先保证overlap收益。

#### I-1 Gate

- 2--6 version有界穷举oracle平铺fresh/alias/reuse objects、slot multiplicity/rotation choices并过滤MustAlias/MustSeparate/lifetime/order cycle；
  与production exact plans逐key一致、无duplicate；
- occurrence property覆盖single/multi-dimensional waves、main/tail、nested child、peer send/recv/relay、same-region DDR和loop-external
  consumer；只有all-and-only common recurrence广告multi-slot；
- upper-bound test覆盖trip/capacity/target分别成为minimum、rank-zero/one-trip只有1、4/5及更大slot可达、unknown footprint不默选；
- alias/range覆盖DPS in-place、fresh out-of-place、exact subview、partial overlap、MayAlias、early overwrite/free与completion-constrained
  reuse；schedule-dependent choice产生精确J requirement而非当前order verdict；
- K structure mutation使affected I/J失效，unrelated component保留；proposal provider关闭/反转不改变exact set或Q51有界穷举optimum；
- query source byte-identical、零actual loop scan/materialization/clone/default statistics。winner apply、lifetime verifier和donor迁移由I-2。

### I-2 专项调研：construction-time slots、lifetime verifier与migration

MLIR ownership-based deallocation把buffer ownership作为SSA/lattice问题，并在bufferization之后单独插dealloc；XLA heap assignment也只在
明确live intervals上放置chunks。current `SelectedBufferMaterialization`反过来在Instr IR里寻找“像pipeline的loop”，clone/move整个loop
body、split waits、重建NCC joins、改upper bound，再remap relations。它同时承担K execution structure、J scheduling/completion、I storage
和Q50.0 memory preparation，必须由construction-time协作取代，而不是继续拆helper保留同一算法。

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
  append all slot roots as loop-carried iter args
  bind this occurrence to SlotIndexExpr-selected region argument
  yield the selected static permutation for the next occurrence

BufferObjectBuilder::release(object, selectedEvent, rewriter):
  emit one deallocation/release after all uses and async completions
```

`multiplicity=1`不生成rotation scaffolding。每个slot是独立allocation，不把一个大allocation切片伪装成独立lifetime。普通`+1 mod m`
rotation可用`[slot1,...,slot0]`SCF iter args；其它selected affine recurrence由prepared static permutation/selector表达，不能在lowering按
loop position猜。main/tail/prologue/steady/epilogue由E/K loop structure拥有，I只绑定slot和backedge reuse。

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
| `SelectedBufferRequest/Message/Scope` | prepared physical-version/event relations | local、peer send/recv、relay、independent scopes | optional node IDs/message scan API删除 |
| `buildSelectedBufferingScopes` | I prepare | per-Tile stable scope/ID mapping | expectedTileIds adapter和movement-domain backlookup删除 |
| 1900-line `SelectedBufferMaterialization` loop discovery/clone | E/K loop builder + I construction-time slots | 2/3/4/5 slots、tail、different scopes | `NoExactLoop` search、operation clone/move、OwningOpRef result删除 |
| materializer stage inference/SCF pipelining | K execution structure | prologue/steady/epilogue、cross-engine stages | I不调用`scf::pipelineForLoop`或改loop bounds |
| wait splitting/operation rotation | J event schedule | Direct-DTE issue/wait/overlap | I不split waits或reorder operations |
| required NCC join rebuild | Q63/J completion closure | pending-worker/join negatives | I不include NCC completion internals |
| relation retarget/remap | typed PhysicalVersion/Event materialization | current relation、failure atomicity | pointer remap和node-based StructuredMaterializationRelations退出I |
| current actual tests | I query/prepare/builder/verifier owners | multiplicity>trip、external consumer、cross-stage write、send wait/release、different loops | 每项迁移后才删旧fixture |

#### I-2 Gate

- actual IR覆盖fresh/reused/in-place storage、exact subviews、single/2/3/4/5 slots、multi-dimensional rotation、main/tail、nested scope、
  peer send/recv/relay和同Tile多个独立families；slot definitions/counts/iter args/yields与plan逐ID一致；
- negative覆盖multiplicity>occurrences、wrong rotation、missing/duplicate allocation、MustAlias/MustSeparate mismatch、partial overlap、
  cross-loop/external consumer、use-before-load/recv、cross-stage overwrite、send-before-wait release和early/double dealloc；
- K/J mutation/invalidation test证明新structure/order重prepare I，old plan不可apply；unrelated families保持stable；
- failure injection覆盖prepare首/末object、mid family、loop extension、release及postverify，source/Card subtree atomic；custom/generic roundtrip、
  verify-each、Tile→Instr、SPM planning、Direct-DTE和Q50.0 gates通过；
- source/call-tree gate删除old domain/request/materializer/clone、late relation remap、NCC/schedule混装；query、prepare、object builder、rotation
  builder和verifier独立files/libraries；
- baseline single-slot和search winner复用同一builder但policy独立；轻量source→CardModule→Q50.0→package/no-card通过，不运行重型LLaMA
  search。

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
中的storage/reuse edges逐ID变化，unrelated scopes保持相同；query零IR，winner construction只消费post-K plan。只有该checkpoint通过，
Q50.J schedule closure才能开始。

## Q50.J：Ready/Order/Worker/Resource Schedule

### Pipeline contract

Pipeline position:
- Upstream IR / input:
  immutable TensorProgram、fixed semantic facts与closed B--I typed plans，以及从这些facts/plans、Q63 completion和immutable target topology/resource facts可重算的
  compute/movement/buffer events；没有actual Tile/Instr module、worker assignment或execution-structure pipeline。
- Current stage responsibility:
  foundation先建立plan-level event nodes、hard dependencies、async completion、exact buffer/effect constraints、resource uses及
  disjunctive resource-order choices；不选择order/worker。Q50.K改变execution structure后重建foundation；J closure再枚举order/worker/
  completion并生成query-local calendar。只有winner commit才创建actual order/worker/waits/joins。
- Output IR / files:
  foundation/query不写IR或文件；`EventGraphAnalysis`和calendar可销毁、可重算，不进candidate state。closure state只保存typed
  schedule choices。winner actual IR用op order、worker attrs、async waits和completion joins自包含表示；对应由plan IDs映射，不保存pointer。
- Downstream consumer:
  Q50.K在planning event structure变化后重建本域；Q51联合比较schedule/pipeline plans并选择winner，随后一次Q50.0验证SPM/DDR、
  transport、resource和ABI；Q52使用calendar work/contention estimate排序。
- User-level driver / named pipeline:
  Q51 closure后的public `search` session；baseline保持其canonical worker0/source order，不枚举本域。
- Explicit non-goals:
  foundation不选order/worker/pipeline/winner，不clone/materialize module、不枚举时间戳/idle、不用nominal bandwidth签发legality，不保存
  operation handle或shadow calendar，不恢复static capability/profitability registry。
- Done criteria:
  有界穷举independent reference逐点匹配order×worker域；planning→winner apply correspondence、Direct-DTE issue/wait、multi-worker join、
  alias/effect、shared DDR、opaque endpoint与known directed-link资源正负例闭合；production search不构造Instr后取first。旧ready-order/worker算法与测试
  完成能力迁移后才能删除其owner。

### event-resource-foundation-1 专项调研：event DAG、completion与resource-effect boundary

LLVM MachineScheduler先构造ScheduleDAG，再由独立strategy选ready node并可选择是否追踪register pressure；CIRCT scheduling也把problem
components/input constraints和solution properties/verification分开。MLIR async则要求所有依赖通过token/value显式表达，MemoryEffect
resource只描述资源类别，精确range仍由alias analysis负责。对应本仓，foundation必须是“待调度问题”的pure typed analysis，而不是
从已排好序的Instr block反推可移动windows。

current `CardInstructionScheduleDomain`借actual Module/Block/Operation pointers，snapshot attrs/operands/types模拟epoch；它按source block
order给同root memref建立RAW/WAR/WAW，把任何synchronous op两侧全序化，并把任意peer endpoint伪装成exact `DirectedPeerLink`。apply又
move-own Tile modules、移动ops、改worker并重建joins。可迁移的是topological enumeration、closed worker enum、Q63 completion和resource
classification测试，不是actual-IR-first API。

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
action至少有issue和completion两个nodes；synchronous action可由zero-distance issue→completion表示。Direct-DTE completion只来自H token/
wait contract，NCC issue/join只来自Q63 typed completion，二者绝不互相完成。

Q50.S semantic owner和每个B--I mechanism通过静态typed visitor贡献event descriptors；没有runtime provider registry、字符串operator type或一份event attr IR。
query-local graph在candidate state变化后销毁，winner commit后actual IR成为唯一事实源，因此不构成长生命周期shadow schedule。
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
  TileEngine(TileId, TargetOperationResource)
  | NCCWorker(TileId, NCCWorker)
  | TileDTEEngine(TileId)
  | SPMRange(StorageObjectId, ExactPhysicalRange)
  | CardDDRResource(CardId, channelClass)
  | DirectedNoCLink(DirectedLinkId)          // only with exact target route
  | OpaqueNoCTransfer(SourceTile, DestTile)  // estimate only
  | ControlResource(scope)

PlannedResourceUse
  event: EventId
  resource: ResourceKey
  mode: Read | Write | Exclusive | CapacityUnits
  interval: IssueToCompletion | Instantaneous | Until(EventId)
  knowledge: Exact | LowerBound | Estimate
```

target operation/completion interfaces给engine/worker/DTE facts，G/I exact storage ranges给SPM，H DDR plan给DDR。只有H target routing是
deterministic/programmable且route closed时才创建`DirectedNoCLink`；current opaque target-routed peer只产生endpoint DTE exact resource、
NoC hop/cut estimate，不能进入hard link contention。共享resource本身不自动产生hard edge：capacity/exclusive constraints进入order
choices或J closure，bandwidth只形成estimate。

#### Analysis scope、failure与invalidation

event graph的最窄完整container是Card plan，因为peer matching和DDR可跨Tiles；构造先按Tile/region分component，再加cross-Tile edges，
无连接components可独立query。source-only relation/effect builder可以有MLIR AnalysisManager薄wrapper；driver planning直接拥有同一
builder的typed result。assignment-dependent event graph保持query-local，不伪装成MLIR analysis或跨session cache。
extensionally相同component可在immutable session memo descriptor，但assignment仍独立。

缺plan field返回F-style scoped Deferred；source/target action无event/resource contract为Unsupported；构图work limit为Indeterminate；malformed
ID/duplicate action是compiler bug。fixed semantic fact、B--I或K任一observed choice改变时，相关component graph、resource facts和后续J assignment全部失效；
不比较operation snapshots或manual epoch。

#### event-resource-foundation-1 Gate

- plan-level unit覆盖independent branches、chain/fanin/fanout、nested execution、multi-piece DDR、peer direct/relay/gather、slot rotation、
  release和observable output；每个selected action产生all-and-only events/completion；
- hard-edge oracle逐reason比较SSA/effect/range/token/Q63关系；可交换WAW/共享resource形成两个order choices而非source-order edge，exact
  non-overlap无冲突，MayAlias不猜；
- opaque target routing没有DirectedNoCLink exact use；deterministic/asymmetric route fixture才有正确directed links；DDR/DTE/NCC/SPM
  resource scope与capacity modes正确；
- hard cycle返回最小causal witness且只拒绝相同fixed-semantic+B--I combination；missing/unsupported/work-limit/compiler-bug分类独立；
- semantic/B--I/K mutation精确invalidate affected components，hash/input/parallel discovery不改变stable graph；source byte-identical、零Instr/module
  construction、pointer snapshot、default statistics。order/worker算法由event-resource-foundation-2。

### event-resource-foundation-2 专项调研：ready/resource successor、list proposals与bounds

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

#### event-resource-foundation-2 Gate

- independent reference对2--7 events平铺worker/resource instances、resource sequences、control topological orders和completion
  boundaries；与kernel plan集合一致、无lane/transitive duplicate；
- donor counts在对应plan graph保持：两个independent worker-capable events为`2! * 3^2 = 18`，producer+两个fanout consumers为
  `2! * 3^3 = 54`；same-range ordered writes只保留semantic合法方向；
- Direct-DTE issue→independent compute→wait、fanin/fanout、relay、slot reuse、multi-worker join、shared DDR及known/opaque NoC resources
  的ready sets/requirements正确；
- critical-path/resource/transport bounds与tiny exhaustive schedule optimum比较从不高估；unknown fact不变known，list/DP proposal始终为
  exact member；关闭/反转proposal不改变domain；
- K graph变化重建kernel/calendar；input/hash/parallel discovery不改stable plans。query无IR/pointer/timestamp state、default stats；
  winner apply与完整schedule closure留给K之后的J-closure。

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
  Q50.J从planned event structure重建schedule alternatives；Q51选择完整winner，随后一次commit materialize pipeline并由Tile memory
  planning分配全部rotating slots，Q50.0验证transport/resource/ABI。
- User-level driver / named pipeline:
  Q51 planning session；`none`不进入本域，search的single-buffer plan使用serialized identity。
- Explicit non-goals:
  不把每条dependency或nonempty multi-slot默认pipeline，不自行选buffer count/final order/worker，不从estimated overlap接受candidate，
  不恢复whole-Module fixed-slot clone API或在actual IR上调用pipeline transform寻找结构。
- Done criteria:
  serialized identity、2+ actual stages、selected multiplicity、prologue/steady/epilogue、Direct-DTE issue/wait、alias/external-write/tail负例和
  Q50.J invalidation/re-entry受测；production assignment与winner apply对应。旧FixedSlotPipeline source/header/test的独有能力全部映射并
  迁移后，旧接口才能保持退役。

### K-1 专项调研：serialized/pipelined execution-structure domain

LLVM MachinePipeliner和iterative modulo scheduling把loop-carried dependence distance、resource model和initiation interval作为显式约束；
CIRCT也为`CyclicProblem`/`ModuloProblem`单独建模跨迭代dependence，而不是在普通acyclic order上加一个pipeline flag。pinned MLIR
`scf::pipelineForLoop`只机械生成prologue/kernel/epilogue，caller必须先提供合法schedule。对应本仓，K是execution structure axis，J才是
结构内的resource schedule，I是storage/rotation；current StagePipeline对Instr Module调用SelectedBufferMaterialization的做法没有独立K
domain。

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
    launchDistance: PositiveInteger
    eventStages: (EventId -> StageId)[]
  }

PipelineDependence
  source, destination: EventId
  iterationDistance: NonNegativeInteger
  completionKind: DataReady | AsyncCompletion | BufferReuse | Effect
```

`PipelineScopeId`是一个E/I recurrence及其connected J event component，不是edge/node或actual loop pointer。`StageId`从0开始、无空洞；
同stage events由J closure排序。`launchDistance`表示相邻启动的logical iteration距离，不是cycle timestamp/target II；J closure在typed
latency/resource facts上求可行calendar/II。prologue/steady/epilogue由trip domain、stage map和launch distance唯一派生，不进入state。

#### Eligibility

每个scope无条件有一个Serialized plan。Pipelined只在以下全部成立时进入domain：

1. E提供static finite、至少两个可启动occurrences，main/tail/nested classes及iteration mapping exact；
2. J foundation的每个event可映到该recurrence，所有hard dependencies可标成明确iteration distance，event component无unknown effect/
   call/observer escape；
3. H movement为每个consumer occurrence提供独立data-ready/completion identity，Direct-DTE wait不被NCC join替代；
4. I存在与candidate live distance兼容的slot family/rotation，MustSeparate ranges独立，上一slot在reuse前有completion requirement；
5. synchronous writeback、host/Kcore observer、region/output publication和tail要么形成pipeline cut，要么由typed stage/epilogue relation完整
   表达；
6. target operation/resource contracts允许这些events进入cyclic execution；unknown profitability不影响legality。

“buffer count>1”只满足第4项的一部分。producer/consumer在不同occurrence relations、loop-external assembled consumer、one-trip、unknown
alias/effect或unmatched completion时只保留Serialized，不实际造loop再失败。

#### Complete finite domain

对eligible scope，先按EventId稳定顺序惰性枚举event的ordered set partitions：每个event选择`0..S-1`，stage IDs canonical无洞，且
intra-iteration hard dependency不允许destination处于source之前的非法stage。`S`范围为`2..min(eventCount, tripUpper,
targetRepresentableStages)`；每个legal launchDistance为`1..tripUpper`，但不能使steady occurrence集合为空。完整stage plan再产生
`PipelineDependence` cyclic graph，交I revalidation和J closure验证distance/resource constraints。

```text
structureSuccessors(prefix):
  yield Serialized once
  if scope is structurally eligible:
    choose next event's canonical StageId
    when stage map closes, choose each legal launchDistance
    derive cyclic EventGraph and invalidate affected I/J facts
    yield only plans whose structural/dependence checks close
```

ordered partitions最坏为ordered Bell/Fubini number，乘launch-distance points；只能lazy。两个stage labels仅重编号时由无洞/first-use
canonicalization去重。不同independent scopes分别选择，不为取得一个“card pipeline”合并loops。

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

- independent有界穷举oracle对2--6 events平铺ordered stage partitions/launch distances并检查dependences，与production domain逐key一致、
  无stage-label duplicate；Serialized恰一；
- eligibility覆盖single/multi-dimensional waves、nested scopes、one-trip、loop-external consumer、DDR/peer/relay、Direct-DTE token、NCC
  completion、synchronous observer、alias/effect和tail；multi-slot alone不广告pipeline；
- structure choice产生正确cyclic EventGraph、occurrence/live distance并精确invalidate I/J；Serialized重置相关extra slots，unrelated scope不变；
- proposal关闭/反转不改变domain或Q51有界穷举optimum；work-limit不变NoSolution；source byte-identical、零actual IR/clone/default stats；
- actual phase construction、periodic DTE和donor mechanics由K-2。

### K-2 专项调研：phase construction、periodic transport与donor migration

pinned MLIR `pipelineForLoop`文档明确声明：它只机械生成prologue/kernel/epilogue，不决定schedule，并假设caller schedule合法；current
实现还只支持loop-carried distance 0/1、single-block body，且某些失败发生在已修改IR之后。它适合作为selected winner中最小新建loop的
mechanical builder，不适合candidate query/validation，也不能在原source或shared IR上尝试多个schedule。

#### Lowering capability与prepare

K domain只广告有current actual builder的结构：

```text
ExecutionStructureLowering =
  SCFDistanceOne                 // launchDistance=1, dependence distance 0/1
  | PeriodicUnrolled(period)     // static bounded donor mechanism
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
distance在capability内、trip count足够或tail strategy显式、nested region/predication支持明确。Period/paired DTE还验证全部participant有同构
occurrence、payload/message/slot selector和completion；不从actual op count猜。

#### Winner construction

Serialized直接调用E loop builder并按J order发出events；Pipelined有两种窄路径：

```text
emitSCFPipeline(prepared, rewriter):
  build one selected unpipelined scf.for inside the new CardModule subtree
  let I add the prepared slot roots, iter args and yield rotation
  emit D/G/H actions plus J-selected waits/joins in the unpipelined body
  attach every actual operation to its EventId through local typed mapping
  call scf::pipelineForLoop with the prepared (operation, stage/order) schedule
  let I/J place final releases after the resulting epilogue/completion

emitPeriodicPipeline(prepared, rewriter):
  construct the finite modulo-unrolled main period directly
  construct exact residual tail from E classes
  bind each Direct-DTE occurrence to its planned slot/message selector
  preserve participant-isomorphic send/recv/wait structure
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
- DTE sender/receiver periodic shapes、message occurrence、wait/release和relay/gather state一致；
- NCC pending participants、minimum joins、synchronous islands和observable output在scope exit前闭合；
- Serialized scope没有pipeline-only extra work/slots，多个independent scopes不被合并。

随后I lifetime、H movement、J schedule/Q63 completion及Q50.0 memory/transport verifier各在自身IR层fresh执行；K verifier不复制这些完整
分析，只比较structure relations。

#### Donor migration

| Current / donor能力 | 终态owner | 必须迁移的witness | 退役条件 |
| --- | --- | --- | --- |
| current `StagePipeline` wrapper | K prepare/constructor | Serialized、selected stages/slots、J invalidation | OwningOpRef module chain与scope-index protocol删除 |
| pinned `scf::pipelineForLoop` mechanics | K `SCFDistanceOne` builder | prologue/kernel/epilogue、cross-stage SSA、odd/even trip | 只消费prepared schedule，不在source/loser上试探 |
| donor static fixed-slot pipeline | K+I+J construction | 1/2/3/4/odd/even trip、stage shortage、slot permutation、tail | whole-Module clone/provider/public header删除 |
| donor periodic Direct-DTE specialization | K periodic builder + H/I/J | paired occurrence、selector、residual tail、message/wait | post-hoc all-rank mutation和ordinal matching删除 |
| donor NCC/DTE completion placement | Q63/J closure | same-worker order、NCC→DTE、DTE→consumer/reuse、terminal join | K不重建completion或解析target calls |
| current selected-buffer stage inference | K-1 domain | cross-engine stage/eligibility negatives | actual scan、local stage winner和failure-driven repair删除 |
| current StagePipeline/SelectedBuffer tests | K/I/J focused owners | 每项mechanics与negative逐表迁移 | 不因旧source删除或测试未注册而丢失 |

#### Complexity、failure与K-2 Gate

SCF construction work/code size为`O(events * stages + slot multiplicity)`，periodic builder为`O(events * period + tail)`；period/phase数均来自
finite plan且受work/code-size bound，超限在planning为Indeterminate/Unsupported，不部分emit后改Serialized。commit失败由C guard回滚，
不返回planner重试。

- actual覆盖Serialized、2/3+ stages、2/3/4/5 slots、main/odd/even/short/tail、nested scope、DDR→compute→store、Direct-DTE
  issue/independent compute/wait、relay/gather和periodic paired Tiles；
- negative覆盖stage/dependence mismatch、trip不足、dynamic/unsupported region、alias/external-write、slot reuse、DTE token/backedge、paired
  period不一致、tail/message/participant mismatch和synchronous observer；
- failure injection在prologue/kernel/epilogue/periodic tail/postverify各点保持source/C atomic；`verify-each`、I/H/J/Q63、Tile→Instr、SPM/
  DDR/Direct-DTE及Q50.0 gates通过；
- source/call-tree gate删除old StagePipeline/FixedSlot/selected-buffer coupling、whole-module clone、post-hoc repair和default stats；K domain、
  prepare、SCF builder、periodic builder、verifier独立files/libraries；
- baseline使用Serialized同一construction，search winner才可能Pipelined；轻量source→package/no-card通过，不运行重型LLaMA search。

K不选择global winner。“某case pipeline获胜、另一case因movement/buffer/parallelism选择Serialized或其它region”由Q51 closure证明。

## Q50.J Schedule Closure

### J-closure-1 专项调研：full resource-constrained domain、algorithm与bounds

固定K后问题成为resource-constrained scheduling：RCPSP一般为strongly NP-hard；modulo scheduling还要满足inter-iteration distance和resource
congruence constraints。经典critical-path/resource-work/RecMII/ResMII都是lower bounds，不是feasible schedule。CIRCT
`ModuloProblem`也把dependence distance、II和resource utilization分别验证。Wafer实际IR不编码cycle timestamp或arbitrary idle，因此
完整搜索对象仍是worker/resource binding、resource/control order和completion boundary；calendar/II只作为这些choices的派生分析。

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

plan不保存start/end cycle、pending sets、makespan或calendar。participant mask、DTE token groups和release sets从event prefix在selected
boundary重算；这避免相同join/wait用不同冗余mask形成duplicate state。

#### Exact closure search

```text
solveSchedule(eventGraph, fixedStructure, fixedBuffers, constraints):
  propagate worker/resource domains and mandatory precedences
  reject an exact empty domain or hard/resource cycle
  decompose components only when no cross-component dependency/resource exists

  choose the unresolved variable with smallest domain:
    worker/resource binding, resource sequence prefix,
    control ready choice, then completion boundary
  visit states by sound lower bound, then semantic ID
  after each choice incrementally update ready sets, pending completions,
    slot reuse and resource capacities

  at a complete leaf:
    derive canonical earliest calendar/steady recurrence summary
    verify every dependency, capacity, completion and publication
    return checked ClosedSchedulePlan plus query-local facts
```

exact successor from event-resource-foundation-2 remains public toQ51, so solver/proposal不是唯一domain入口。component分解必须保留shared DDR、known exact
NoC links、cross-Tile messages、card completion和publication edges；opaque NoC只有estimate，不能制造或删除component hard edge。

resource capacity-1由selected sequence证明；capacity-k由canonical instance assignment和per-instance sequence证明。fully pipelined
resource的outstanding/issue limit从event interval结构计数。K cyclic scope用`PipelineDependence.iterationDistance`验证same/different
iteration precedence和slot reuse；SCF distance-one/periodic capability各用自己的finite recurrence verifier，不把estimated cycle latency
变成legality。

completion placement逐boundary更新：pending NCC workers、DTE tokens、buffer readers/writers和observable obligations。same-worker ordered
NCC issue可仅靠issue order，跨completion-domain consumer必须join；DTE recv/read、send/reuse只能由matching token wait；entry/region
terminal只完成其typed obligations。任何未清pending observer为exact leaf rejection。

#### Why no timestamp axis

固定precedence、resource sequences/bindings和completion boundaries后，canonical earliest calendar把每个event放到所有preds/resources ready的
最早位置。对nonpreemptive regular objective（makespan、high-water upper envelope），无新增dependency/resource release的额外idle不会
改善feasibility或objective；若delay有意义，它必对应另一个resource order或已有completion boundary，已经在domain中。因此不枚举
unbounded timestamps。actual Wafer IR也只表达order/token/wait，不表达cycle start time。

#### Bounds、proposals与knowledge

lower bound取以下known项maximum：acyclic critical path、per-resource total work/capacity、K recurrence distance、I outstanding slots、H
dilation/tree/cut和mandatory completion latency。对cyclic resource可计算structural ResMII/RecMII-like bound供排序；current target没有
exact latency时标Unknown并只保留structural work bound。checked arithmetic/knowledge随每项传播，Unknown不参与strict winner比较。

list schedule、drain-latest、movement-early、compute-early、critical-path、resource-balanced和pipeline-aware modulo order都只是proposal；
每个返回exact leaf。branch-and-bound只在lower bound与同knowledge incumbent可比时prune；work-limit返回checked feasible suggestion+
lower bound或Indeterminate，不报告Optimal/NoSolution。

完整穷尽后才能报告`OptimalForScheduleBound`，它也只针对固定semantic+B--K representation/movement/buffer/structure和current cost knowledge，global
winner仍由Q51。最坏复杂度含topological orders、worker/resource assignments、completion placements的乘积，为指数/阶乘级；incremental
ready/pending/resource updates与实际visited states成正比，默认不运行MILP或输出calendar dump。

#### Failure/no-good

ExactRejection witness可以是empty domain、hard/resource/deadlock cycle、capacity violation、unmatched completion、slot reuse或publication
failure，并列出all-and-only causal semantic/B--K/J choices。Unsupported/Indeterminate/compiler bug沿F taxonomy传播。No-good只覆盖witness key，
不能删另一个worker/resource order、K Serialized、不同slot或movement sibling。

#### J-closure-1 Gate

- 2--7 event exhaustive oracle比较完整worker/resource/control/completion leaves、feasible set和fixed-knowledge optimum；component decomposition
  前后集合/optimum一致；
- serialized与SCF/periodic pipeline覆盖independent/fanin/fanout、multi-worker、DDR、target-routed peer/relay/gather、known directed link、
  rotating slots、tail和observable output；
- pending-state oracle逐prefix核对NCC participants、DTE tokens、buffer last-use/release和terminal obligations；deadlock/resource cycle给最小
  causal witness；
- critical/resource/recurrence/transport bounds从不高于tiny optimum；Unknown不变Known，list/modulo proposals为exact leaves，关闭proposals
  不改domain；
- no timestamp/idle state、actual IR、clone或default stats；work-limit分类和stable determinism受测。winner apply、wait/join materialization
  与migration由J-closure-2。

### J-closure-2 专项调研：construction-time order/completion与deadlock verification

current `applyInstructionSchedule`在已生成Tile Instr modules中move operations、设置worker、删除所有旧joins并调用
`rebuildRequiredNCCJoins`；domain又用operation snapshot判断“epoch”。终态已经有ClosedSchedulePlan和C winner transaction，不需要这份
post-hoc owner。Q63 current-IR analysis仍是final verification资产：它从actual structured control/calls重算per-op pending worker mask；
placement algorithm则迁成plan-level completion builder。

#### Prepared schedule

```text
PreparedSchedule
  controlScopes: PreparedControlOrder[]
  eventBindings: EventId -> EventEmissionSite
  workerBindings: EventId -> NCCWorker
  resourceBindings/sequences: closed J choices
  dteWaitGroups: EventBoundaryId -> TransferCompletionId[]
  nccJoinGroups: EventBoundaryId -> derived participant workers/issues
  releaseGroups: EventBoundaryId -> StorageObjectId[]
  publication/terminal obligations: exact closed set
```

prepare重建一次fixed-K EventGraph并要求与ClosedSchedulePlan identity、I/H/K dependencies完全相同；每个EventId、completion、resource和
control position total/unique。DTE wait group只含该boundary已issued且仍pending的tokens；NCC participant set从pending issues导出，不由
caller传mask；empty/redundant joins canonicalize away。所有可能的cycle/resource/lifetime检查先于mutation。

#### Direct event emission

每个D/G/H/I/K builder不再自己挑insertion point，而向一个scope-local `ScheduleEmitter`注册EventId与typed callback：

```text
emitControlScope(preparedOrder, emitter, rewriter):
  for event in prepared control order:
    invoke exactly that event's typed builder at current insertion point
    set selected worker when creating worker-capable target op
    bind actual operation/token/value to EventId
    at a completion boundary:
      emit one DTE wait for its exact token group
      emit one NCC participant join for its derived nonempty worker set
      emit planned buffer releases after both completion domains close
```

不同Tiles独立构造，cross-Tile messages通过prepared TransferId/EventId关联；不需要一个global insertion order。K SCF pipeline先在
unpipelined selected loop中按J event order建立ops/waits/joins，再由mechanical phase builder复制；periodic builder直接按相同event map
构造。TileRegion→Instr lowering继承worker/completion relation并生成current Instr op；不从op name/ordinal恢复。

same-worker issue order只由selected control/resource sequence表达，不插逐edge join。NCC→DTE/Kcore/call/return或synchronous observer在
prepared boundary加入minimum participants；DTE wait不清NCC pending，NCC join不清DTE tokens。recv destination在wait后才publish
PhysicalVersion，send/relay buffer在wait前不能release。

#### Actual schedule/deadlock verifier

Card-scoped verifier从actual plan relations、SSA、MemoryEffects、Q63 completion与H messages重建：

1. all-and-only EventIds materialized，per-control actual order与worker/resource binding完全匹配；
2. hard/resource/K cyclic dependencies、exact range RAW/WAR/WAW和I slot reuse均满足；
3. every DTE issue token由exact selected wait一次消费，NCC pending masks与join participants逐点匹配，synchronous writeback正确清域；
4. build global wait-for graph：event/resource/buffer/message preparation/issue/completion为nodes，hold/wait/ready为edges；acyclic scopes必须
   无cycle，periodic scopes在一个verified finite period加backedge上无unbroken wait cycle；
5. receiver preparation precedes matching send requirement，relay/gather forward waits正确，resource instance/outstanding capacities不超；
6. 每个region/function/entry terminal无未闭合observable、buffer、NCC或DTE obligation；opaque NoC routing不伪造link check。

deadlock witness返回最小stable cycle及held resources/messages；不能靠插global barrier修复。Q50.0随后在memory-planned Instr上执行更低层
Direct-DTE binding/range/status和SPM/DDR gates，二者不重复owner。

#### Complexity、atomicity与migration

prepare/emission与events+dependencies+completion groups线性；range conflict和wait-for graph使用memo后与actual accesses/edges成正比；
periodic verifier只展开selected finite period，不展开full trip。失败由C subtree guard回滚，不换order/worker/K plan；默认无schedule dump、
calendar或stats。

| Current / donor能力 | 终态owner | 必须迁移的witness | 退役条件 |
| --- | --- | --- | --- |
| `CardInstructionScheduleDomain` actual op/block pointer+snapshot | J plan-level kernel/ClosedSchedulePlan | 18/54 domains、alias/effect、invalidation | pointer/window/snapshot/manual epoch零残留 |
| `applyInstructionSchedule` move ops/set worker/rebuild joins | ScheduleEmitter + completion builder | selected order/worker/min joins/atomicity | owned Tile modules post-hoc apply删除 |
| `InstructionWindowOrder/WorkerChoice` | EventId control/resource/worker bindings | multi-scope/per-Tile exact mapping | operation pointer identity删除 |
| current `analyzeInstructionResources` | J plan resource facts + actual verifier | engine/worker/SPM/DDR/DTE/known link/overlap | opaque route不再记DirectedPeerLink，root-only SPM conflict删除 |
| `place/rebuildRequiredNCCJoins` selection logic | plan-level nccJoinGroups | same-worker、cross-worker、NCC→DTE、terminal participants | final Q63 analysis保留，post-hoc placement API退出production |
| old ReadyOrder/WorkerPlacement/registry donors | J exact kernel/proposals | topological/worker alternatives和target contracts | greedy/local winner/clone/static row整岛删除 |
| current InstructionSchedule tests | J foundation/closure query+actual owners | 每项direct witness迁移 | old Instr-first fixture不代签production pipeline |

#### J-closure-2 Gate

- actual order/worker/resource tests覆盖18/54有界穷举case、fanin/fanout、exact-range alias、multi-Tile DDR、peer direct/relay/gather、slots、Serialized/
  SCF/periodic pipeline及synchronous observer；每个EventId/worker/completion all-and-only；
- completion tests覆盖same-worker无join、minimum cross-worker join、NCC→DTE、DTE recv→consumer、send/relay→release、grouped DTE waits、
  region/function/entry terminal；Q63 fresh pending states与plan一致；
- deadlock negatives覆盖send/recv preparation cycle、resource hold-and-wait、cross-Tile relay cycle、slot backedge和periodic completion cycle，
  witness稳定且没有global-barrier repair；
- failure injection覆盖prepare首/末scope、worker emit、DTE wait、NCC join、release和postverify，source/C subtree atomic；custom/generic
  roundtrip、verify-each、K/I/H/Q63、Tile→Instr、Q50.0 Direct-DTE/SPM/DDR gates通过；
- source/call-tree gate删除Instr-first domain/apply/snapshot、post-hoc join placement、false link/resource analysis、registry/clone/default stats；
  event analysis、domain kernel、solver、prepare/emitter、actual verifier分文件/library；
- baseline canonical order/worker0与search复用emitter但policy独立；轻量source→package/no-card通过，不运行重型LLaMA search。

J只有foundation、closure、actual和donor gate全部闭合才完成；query-local calendar在result交付后销毁，winner IR为唯一schedule事实源。

## Q50.F Full-Coordinate Feasibility Closure

### F-closure-1 专项调研：完整资源问题、witness与安全证明

F foundation故意不能回答完整可执行性；G--K闭合后也不能继续用“maximum live bytes”“first-fit失败”或一次lowering试跑替代证明。
专项调研得到的可迁移边界是：

- [MLIR One-Shot Bufferization](https://mlir.llvm.org/docs/Bufferization/)先分析SSA alias/equivalence再rewrite，unknown relation不会被
  猜成in-place；F同样必须只消费typed alias/lifetime结论，不能从未来offset反推上层关系；
- [OpenXLA BufferAssignment](https://github.com/openxla/xla/blob/main/xla/service/buffer_assignment.h)与
  [HeapSimulator](https://github.com/openxla/xla/blob/main/xla/service/heap_simulator/heap_simulator.h)在schedule/live range已知后才做
  buffer assignment，并把heuristic packing与实际allocation结果分开；可迁移的是“closed execution facts -> pure allocation problem ->
  validated assignment”，不是把XLA heuristic当不可行证明；
- [OR-Tools CP-SAT status contract](https://developers.google.com/optimization/cp/cp_solver)明确区分`FEASIBLE`、`INFEASIBLE`、
  `MODEL_INVALID`和因time/memory/custom limit停止的`UNKNOWN`；F必须保持同样的信息强度；
- current `StaticPackingProblem`/MiniMalloc adapter已经有validated placement、ProvenInfeasible、ResourceExhausted、HeuristicNoFit和
  invalid-result分层。它应原位演进为typed semantic IDs和plan/actual共同问题边界，而不是再建一套search allocator。

#### Pipeline handoff

full query只接受同一dependency generation内已经关闭的fixed semantic facts与B--K assignment：D的execution/value bindings、E exact occurrences、G physical
versions、H `TileCommunicationPlan` action DAG、I storage/slot bindings、K execution structure及J `ClosedSchedulePlan`。任一required
coordinate缺失时仍返回foundation定义的`Deferred`，不构造“剩余部分”的packing problem。

输出仍使用同一个`ScopedFeasibilityResult`。完整成功不是一个新的平行enum，而是：

```text
Consistent {
  facts: FeasibilityFacts {
    coverage: FullCoordinate(FullFeasibilityProof)
  }
  observed: FeasibilityDependencyKey
}

FullFeasibilityProof
  dependencyKey: all-and-only semantic/B--K/target facts read
  resourceProblems: ResourceProblemProof[]
  coverage: EveryPlannedResourceClosed

ResourceProblemProof
  identity: typed scope + semantic object/resource IDs + target arena/resource
  method: ValidatedPlacement | ExactSolverProof | DirectCapacityProof
```

`FullFeasibilityProof`不携actual offset、solver decision stack、operation pointer、digest或日志。validated placement只在当前query栈中作为
存在性witness，验证后销毁；candidate assignment不保存proof或offset。Q51可以session-local按完整typed problem contents memo同一query
结果，但memo不是IR事实源，任一observed choice变化即失效。

#### 一套current资源问题

现有`StaticPackingProblem`原位演进，不新增第二套编号API：`stableOrdinal`/裸`demandIndex`从public合同退出，由owner-private adapter将
`StorageObjectId`映射到solver index。full query建立：

```text
FullCoordinateResourceProblem
  memoryArenas: ArenaPackingProblem[]
  occupancies: ResourceOccupancyProblem[]
  representability: FieldLimitProblem[]
  dependencies: FeasibilityDependencyKey

ArenaPackingProblem
  scope: TileSPM(TileId) | CardDDR(CardId)
  arena: exact [begin, end), reserved/fixed ranges
  objects: (StorageObjectId, bytes, alignment)[]
  mustAlias: exact equivalence classes
  conflicts: (StorageObjectId, StorageObjectId, LifetimeProof)[]

ResourceOccupancyProblem
  resource: DTE/FSM/NCC worker/descriptor/qualified NoC channel/resource ID
  capacity: exact target units/instances
  uses: (EventId, issue, completion, units, selected instance/order)[]

FieldLimitProblem
  owner: message/round/slice/descriptor/instruction/control object
  valueOrCount: exact checked integer
  acceptedRange: immutable target range
```

问题构造只使用各plan对象本身能解释的事实：

1. G给每个`PhysicalVersionId`的exact footprint/alignment及alias producer；I把它们绑定到`StorageObjectId`和slot family；E/K给有限
   occurrence/period，J给definition、first/last use、completion、release和resource sequence，由此建立per-Tile SPM objects/conflicts；
2. H的DDR stages、program boundary和output publication建立card-shared DDR objects/lifetimes；external/fixed roots及reserved ranges显式
   进入arena，不用symbol或buffer名识别；
3. H的每个transfer/relay/bundle产生exact source、destination、relay staging和message/descriptor demands；J-selected issue/completion及
   I slots决定outstanding intervals，target facts给每Tile DTE/FSM/descriptor capacity；
4. J/K已经选定worker、resource instance、control/resource sequences和periodic backedge；F只验证这些选择形成的occupancy，不重新排程；
5. current opaque NoC routing没有per-link occupancy problem，只检查endpoint、message、relay buffer及route-independent hard fields；只有
   target提供deterministic/programmable route与channel capacity时才加入qualified link/channel problem；
6. target instruction、message identity、offset/address、descriptor或structured-control字段若有hard representability limit，使用checked
   exact count/range；bandwidth、hop、estimated duration和profitability不属于feasibility。

每种selected plan action必须提供一个窄typed resource description，F与winner emitter共同消费；emitter不得产生未描述的hidden allocation、
message、event或resource use。actual IR verifier仍从SSA/effect/control独立重建同一normalized problem，防止共同helper的错误自证。
若某个lowering必然引入但plan层无法exact描述的temporary，当前mechanism返回精确`Unsupported`并扩合同，不能把它留给winner commit碰运气。

#### Query算法与status映射

```text
queryFullCoordinateFeasibility(scope, state, workAllowance):
  validate typed state and dependency generation
  foundation = analyzeFoundation(scope, state)
  if foundation is ExactRejection/Unsupported/Indeterminate:
    return foundation
  required = collectRequirements(scope, state, all full checks)
  if required is nonempty:
    return Deferred(foundation.facts, required)

  problem = buildFullCoordinateResourceProblem(state)
  validate problem structure before invoking any solver
  proofs = []

  for subproblem in stable semantic order:
    if direct size/clique/count/occupancy proof exceeds capacity:
      return ExactRejection(exact causal witness)

    if subproblem is arena packing:
      placement = deterministicFastPlacement(subproblem)
      if placement exists and validatePlacement(subproblem, placement):
        proofs += ValidatedPlacement(subproblem.identity)
        continue
      if fast result is malformed:
        return compiler bug
      if no exact work was reserved from workAllowance:
        return Indeterminate(ResourceWorkExhausted, partial proofs)

      exact = solvePacking(subproblem, workAllowance)
      if exact has a placement and validatePlacement(...):
        proofs += ExactSolverProof(subproblem.identity)
      else if exact proved infeasible:
        return ExactRejection(exact solver/certificate witness)
      else if exact stopped or heuristic found no fit:
        return Indeterminate(ResourceWorkExhausted, partial proofs)
      else:
        return compiler bug
    else:
      validate exact occupancy/range/count and append DirectCapacityProof

  return Consistent(FullCoordinateProof(proofs, observed dependencies))
```

fast placement是**充分witness生成器**，不是必要性判断；`HeuristicNoFit`永远不变成rejection。exact solver的
`ResourceExhausted`/timeout保持Indeterminate；`ProvenInfeasible`只有solver完整穷尽或返回可独立验证的capacity certificate时才成为
ExactRejection。builder已经验证过的问题若被solver报invalid、返回缺失/重复/冲突placement或产生arithmetic inconsistency，是compiler
bug。只有source/target本身的typed field明确超出可表示范围时才是普通ExactRejection，不能把host整数overflow笼统归给用户。

`workAllowance`来自已有compile/search invocation的统一work accounting，只控制本次pure proof effort，不进入candidate identity、IR、CLI
或长期cache。Q51只对完整且可能建立/改善incumbent的contender调用full query；partial nodes继续只跑foundation。默认先执行线性
direct checks与deterministic placement，只有完整contender的fast witness失败才进入exact search；不在每个transition运行MiniMalloc、
maximum clique、SMT或MILP。

#### Soundness、通用性与Q50.0 parity

soundness分三层：

1. **Plan-resource correspondence**：对fixed semantic description及每个B--K plan type证明resource description与winner emitter一一对应；所有planned storage、
   message、event、slot、worker和target-limited action恰出现一次，emitter无hidden resource；
2. **Problem-result soundness**：validated placement给每个object一个in-range、aligned offset且所有conflict ranges不相交，所以是存在性证明；
   direct capacity/count witness逐exact interval/range验证；ProvenInfeasible只来自必要certificate或complete solver proof；
3. **Plan-actual parity**：Q50.0从新Card subtree的actual Instr/SSA/lifetime/effect重新建立normalized problems并逐typed semantic ID与
   plan问题比较，然后重新运行current allocator分配actual offsets。相同完整plan在F有full proof而Q50.0问题不同或不能pack，是
   planning/emitter/lowering compiler bug；不能返回Q51换candidate。

offset witness不进入state会使winner在Q50.0重算一次packing，但只对唯一winner发生；它避免offset成为跨mutation shadow plan，也让actual
IR保持唯一地址事实源。F和Q50.0共享problem schema、validation与solver status映射，不共享plan-specific builder，也不保留两套current
packing API。

通用性来自typed resource correspondence，而非workload表。不同shape、Tile数、hole topology、ragged temporal tail、multi-producer、
direct/relay/ring/row-column通信、single/multi-slot、Serialized/SCF/periodic execution都只改变objects、events、conflicts和target facts。
新增mechanism只有实现resource description、emitter correspondence和actual parity后才可进入full proof；没有case/name fallback。

#### Scope、并行与复杂度

问题先按真实physical arena/resource分解：每个Tile SPM、per-Tile DTE/FSM可独立求解，card DDR、cross-Tile message identity及known shared
link/channel保持Card scope。独立subproblems可使用compiler已有bounded host worker pool并发，aggregate按semantic key归并；不能每Tile
另开无界solver线程，也不能让并行完成顺序决定result。经verified automorphism得到的extensionally相同问题可共享pure result，但每Tile
仍重新绑定自己的semantic IDs，不能强迫offset或assignment相同。

problem build与direct checks为`O(objects + conflicts + events)`，resource occupancy按selected sequence sweep为`O(events log events)`；
deterministic packing最坏quadratic。exact static packing一般为NP-hard并可能指数增长，所以只对closed contender按work allowance运行。
solver停止只降低knowledge，不损害合法域。默认不创建problem dump、statistics、timing或work report；显式instrumentation可观察work但不参与
status和选择。

Q49.P对自己的唯一closed canonical plan调用同一full query，不建立search frontier；其canonical single-buffer/serialized结构应优先通过
fast validated placement。若得到ProvenInfeasible才执行预定义单调functional legalization；Indeterminate不能伪装成overflow后缩tile，
也不能先提交Q50.0探测。Q51则保留Indeterminate state并可在后续获得work allowance时重试同一pure problem。

F-closure-1只收敛问题、算法和proof边界；independent oracle、false-rejection反例、plan/actual parity矩阵以及current/donor退役门禁由
F-closure-2闭合。

### F-closure-2 专项调研：independent oracle、false-rejection与迁移门禁

solver cross-check必须区分“检查一个可行model”和“相信一个不可行结论”。SMT-COMP的model-validation/proof tracks分别检查solver给出的
model和不可行proof；OR-Tools也把无solution但未证明infeasible的停止状态定义为Unknown。本仓现有MiniMalloc测试已经用独立tiny
exhaustive address enumerator核对seeded arbitrary conflict graphs，这是正确起点，但它尚未覆盖semantic/B--K plan problem builder、occupancy、
plan/actual parity及旧probe能力迁移。

#### 三个独立reference oracle

test-only oracle不能调用production F builder、packing solver或status mapper：

```text
ReferenceAddressOracle(problem):
  validate arena/object/conflict fields with test-local checks
  visit objects by semantic StorageObjectId
  enumerate every absolutely aligned offset in [begin, end - bytes]
  reject a partial placement only on an already placed conflict overlap
  return one witness, or Infeasible only after the finite tree is exhausted

ReferenceOccupancyOracle(problem):
  walk the selected EventId/resource sequence and finite periodic backedge
  update per-instance outstanding units at issue/completion boundaries
  independently check capacity, release and terminal zero-pending state

ReferenceFieldOracle(problem):
  perform arbitrary-precision/test-local range and count comparison
  return exact owner IDs whose values are outside the target interval
```

address oracle限制在0--6 objects、small arena和small alignments；occupancy oracle限制在2--8 events/一个finite period。限制只属于测试规模，
不是production domain。每个seed记录在测试源码中，失败直接打印完整typed problem；普通编译不生成oracle、problem dump或随机状态。

结果性质必须双向检查：

- production返回`FullFeasibilityProof`时，所有placement先过production validator；tiny问题还必须由reference oracle确认存在witness；
- production返回ExactRejection时，direct certificate由test独立重算，有界solver-only rejection必须由reference tree确认Infeasible；
- `HeuristicNoFit`、`ResourceExhausted`、zero work allowance及故障注入均只能得到Indeterminate，即使reference恰好知道答案；
- malformed production problem、invalid solver placement、duplicate/missing object或host arithmetic inconsistency走compiler-bug路径，不能进入
  no-good；
- 增大arena/容量不能把Feasible变Infeasible，删除conflict不能把Feasible变Infeasible，增加conflict不能把Infeasible变Feasible；增大
  work allowance只能把Indeterminate收敛为同一个Feasible或Infeasible结论，不能翻转已证明结论。

#### 必须保留的false-rejection反例

| 反例 | 容易出现的错误 | 正确结论/owner |
| --- | --- | --- |
| current four-demand first-fit counterexample | `HeuristicNoFit`当overflow | fast query Indeterminate；reserved exact work找到validated placement |
| nonzero arena base + alignment | 先算relative offset再整体加base | absolute alignment验证；feasible与range overflow分开 |
| disconnected conflict components | 对每component使用独立arena或错误累加 | 同一arena可跨互不冲突components复用range，reference oracle一致 |
| zero-byte object与aligned positive object | zero-height solver encoding误判 | zero object占空range但仍检查aligned representability，不阻塞positive reuse |
| multi-segment/cyclic lifetime conflict graph | 把lifetime压成单interval或只看peak sum | exact conflict edges决定packing；cycle graph可合法交错 |
| MustAlias、MayAlias与mutually-exclusive branch | alias成员相加、unknown建edge或分支全相加 | MustAlias取一次maximum；其余只按typed exact relation/conflict处理 |
| communication source/recv/relay buffers | send issue即释放、recv issue即可读或relay转发后立即复用 | lifetime分别延至exact send/recv/child completion，由H/I/J IDs证明 |
| multi-slot与periodic backedge | 只展开一个iteration或把slots当大buffer slices | 每slot独立object；finite period+backedge oracle证明reuse |
| 16 Tiles上同构但不同semantic owners | memo结果强迫相同offset/assignment | 只共享existence result，按每Tile IDs重新绑定；各SPM arena可独立复用同一数值offset |
| opaque NoC direct transfers | canonical shortest path被当hard link contention | 不建立per-link occupancy；只验证endpoint与route-independent hard limits |

这些是通用resource形态；table中的数字和拓扑只界定tiny test，不成为workload/shape协议。

#### Plan/actual normalized-problem parity矩阵

每个case先从immutable source和一份closed typed assignment建立plan problems；test-only commit一次winner，随后从新Card subtree的actual
SSA/effect/control重建actual problems。比较对象是按semantic ID排序的arena、objects、size/alignment、MustAlias classes、conflicts、
fixed ranges、occupancies和field limits，不比较solver offset或operation pointer。

| 输入机制 | 必须出现的plan facts | actual parity/negative |
| --- | --- | --- |
| D/E top-level、nested、replica、tail | 每个execution result/state的exact occurrence与lifetime | missing/extra execution object、tail occurrence错配失败 |
| G primary/alias/conversion/secondary versions | one producer per physical version、exact footprint/alignment | hidden conversion alloc、lookup-any source或alias overwrite失败 |
| H DDR/direct/fanout/relay/gather/ring/row-column | source/destination/relay staging、bundle、messages、completion | hidden DDR sibling、immediate release、missing relay/message或opaque fake link失败 |
| I Fresh/Alias/Reusable与1..U slots | one storage object/alias class、独立slot objects、backedge lifetime | large-buffer fake slots、unproven reuse或early release失败 |
| J worker/resource/control/completion | selected instance/order、outstanding interval、wait/join/release | post-hoc reordering/join、missing completion或capacity excess失败 |
| K Serialized/SCF/periodic | structure-specific occurrences、finite phase/period与recurrence | reuse old I/J generation、missing prologue/epilogue或periodic edge失败 |
| target field limits | exact message/descriptor/instruction/control counts/ranges | truncation、saturation、unchecked narrowing和ID collision失败 |

每个mechanism的resource descriptor/emitter pair另有direct unit；parity integration不试图平铺全Cartesian product，完整组合覆盖归Q51有界flat
oracle。failure injection在problem build首/末object、parallel subproblem、solver result validation和actual comparison处执行；source不变，
尚未commit时无IR，commit失败由C subtree guard全量回滚。

#### Current/donor capability migration

| Current / donor | 终态owner | 必须迁移的能力与test witness | 删除/改造门禁 |
| --- | --- | --- | --- |
| current `ScopedFeasibilityKind::LowerBound`、reason+optional fields | F closed typed sum + coverage | partial lower bound、unsupported、missing、single overflow与precedence | old enum/reason/optional state machine零残留 |
| caller `FeasibilityCoordinate[]`与`ScopedFeasibilityAssignmentKey` | automatic requirements + observed dependency key | 删除每个G--K field得到exact scoped requirement；sibling causal keys | caller list、node/group/raw vector key、manual epoch零残留 |
| current max-one-value `getNodeLowerBound` | foundation StorageDemand/certificates | operand/result、alias group、mandatory coexistence与nested scope | node-wide max-only implementation退出 |
| `StaticPackingDemand.stableOrdinal`/raw indices | `StorageObjectId` public schema + private solver index map | input permutation、parallel build、failure owner稳定 | ordinal不再是semantic/canonical key |
| hidden `kBasePackingSearchNodes`/default MiniMalloc-first policy | caller-owned unified work allowance，fast validated witness first | first-fit反例、zero allowance、exact resolve与determinism | planning call graph无默认百万级solver；无新CLI knob |
| MiniMalloc adapter | current owner-private exact packing backend | arbitrary graph、gaps、alignment、nonzero base、zero objects、typed statuses | third-party type不越界；NoFit/exhaustion不变rejection |
| clique/individual overflow evidence | foundation/full direct certificate helper | deterministic object IDs、pair proofs、capacity bytes | certificate不依赖MiniMalloc encoding或offset |
| `checkTileRegionSPMCapacity` | actual problem builder/validator；plan query归F | structured lifetime、DTE completion、unsupported control | 无production probe caller后删除public region query；不恢复scratch use |
| `planSPMMemoryModule`/`planDDRMemoryModule` | Q50.0 actual build→solve→apply | complete actual scope、absolute alignment、DDR contiguous/range、atomic offsets | 保留winner-only apply；analysis/build可独立测，不被baseline/search probe调用 |
| old `SPMCapacityEvaluation` region/function clone wrappers | 不恢复；F plan proof + Q50.0 actual parity | fits/overflow、unsupported lifetime、full-scope demand、foreign/missing relation、failure atomicity | 无clone/remap/phase-label/diagnostic control；每项witness已有current owner |
| `StructuredMaterializationRelations` raw Value attribution | C/G--K typed plan IDs；actual transaction-local ID→Value relation | all-and-only current relation、foreign/stale negative | planning result不含Value；actual mapping不跨mutation/cache |
| `TileMemoryPlanning`中的selected-buffer/stage/schedule post-hoc calls | I/K/J construction-time emitters，Q50.0只做actual memory/transport gate | slot/order/worker/completion parity | memory planning不选择或重建上游choices |
| baseline通过Q50.0 failure推进temporal | F full plan proof + causal monotonic legalization | overfull-to-fit、Indeterminate不缩tile、one commit | no probe/actual feedback/accepted rebuild |

旧probe的“关系在clone后仍current”不原样保留，因为clone协议本身退役；它真正保护的能力被拆成plan semantic ID totality和actual
transaction-local relation totality。旧conversion rollback归C/winner transaction，旧unsupported lifetime归F Unsupported及actual lifetime
negative，不能因删除原test文件而丢失。

#### 文件与library边界

实现不得重新塞回一个大`ScopedFeasibility.cpp`：

```text
lib/Wafer/Planning/PhysicalDataflow/Feasibility/
  Requirements.*          // automatic required coordinates/dependency key
  StorageCertificates.*   // foundation demands/interference/certificates
  ResourceProblem.*       // full typed arena/occupancy/field problems
  FullFeasibility.*       // status composition and work allowance

lib/Wafer/Transforms/MemoryPlanning/
  StaticMemoryPacking.*   // one current pure schema/validator/policy
  MiniMallocPacking.*     // owner-private backend adapter only
  LifetimeAnalysis.*      // actual SSA/control/effect lifetime facts

lib/Wafer/CodeGen/Executable/
  TileMemoryPlanning.*    // winner actual orchestration; no search choice
```

public headers只暴露typed query/result及current packing schema；problem builders、solver adapters、actual relation maps和failure injection留在各
owner private headers。CMake依赖方向为Planning→pure MemoryPlanning schema，CodeGen→Planning result schema/Transforms actual kernels；
MemoryPlanning不得include search controller/materializer，避免循环依赖。instrumentation sink可由测试/显式compile timing注入，普通调用不
创建统计、dump或日志对象。

### F-closure-2 Gate

- reference address/occupancy/field oracles覆盖上述范围；seeded small problems、monotonic metamorphic和input permutation fresh通过，运行时间
  受small bounds约束，不把大solver test放进普通全量路径；
- result-strength gate证明FullProof→validated witness、ExactRejection→verified certificate/oracle infeasible、Indeterminate永不进入no-good；
  first-fit counterexample、solver exhaustion及invalid-result failure injection全部命中；
- plan/actual parity矩阵逐mechanism比较normalized problems；一个mixed 2--4 Tile winner完成F proof→一次C commit→一次Q50.0 offsets/
  transport/resource gate，source byte-identical且无loser IR；
- baseline direct gate证明一条canonical plan只做plan query和一次winner commit；exact causal overflow才走预定义monotonic transition，
  Indeterminate/Unsupported不调用search或actual probe；
- source/call-tree gate删除old enum/caller list/raw assignment key/default solver budget/public region capacity query，以及旧clone/probe/remap/
  phase-diagnostic控制链；旧unique witnesses逐项在新owner找到直接test；
- query/build/certificate/full solver、actual problem extraction、offset apply和orchestration分文件；没有default stats/logging、packing state cache、
  pointer/ordinal key、IR clone或repair；
- 定向unit、custom/generic/verify-each、named/production parity、source-to-CardModule→Q50.0→package/no-card轻量纵向fresh执行。F本身不运行
  重型search workload，完整模型时间/质量门禁在Q51/Q52链闭合后进行。

F只有foundation、full proof、independent oracle、plan/actual parity和donor迁移全部闭合后才完成；“MiniMalloc unit通过”或“actual
allocator能找到offset”都不能代签planning合同。

## Q51 Planning and Search

### Pipeline contract

Pipeline position:
- Upstream IR / input:
  immutable normalized card-local TensorProgram、显式semantic root ops、current target/topology/resource facts，以及已经分别闭合的
  Q50.S fixed semantic/query/decomposition合同与Q50.A--K typed physical domain/query/materializer contracts；没有baseline executable、
  CardModule、Instr或accepted offsets。
- Current stage responsibility:
  建立一次search-owned planning problem，在typed partial states上惰性展开真实Q50 choices、消费F/J legality与cost/bound，选择一个
  complete `PhysicalDataflowPlan`。planning不修改IR；选择结束后把唯一plan移交winner commit。
- Output IR / files:
  planning阶段只返回move-only selected plan或typed failure，不写IR/文件/sidecar。commit阶段一次构造CardModule并进入Q50.0；accepted
  CardExecutable仍由Q50.0拥有。
- Downstream consumer:
  C/G--K construction-time emitters、TileRegion→Instr、actual SPM/DDR/transport/resource verification、target/package transaction。
- User-level driver / named pipeline:
  `wafer-compile`的public `search` policy；`none`进入独立baseline controller。winner后两者复用同一materializer和named lowering
  subpipelines，但policy controller绝不互调。
- Explicit non-goals:
  不把search写成operation pass，不在frontier保存IR/offset/calendar/metrics，不执行baseline取得初值或fallback，不用runtime provider
  registry/opaque candidate bag，不让proposal或局部mechanism选择winner，不把optional report/statistics变成默认编译路径。
- Done criteria:
  planning source byte-identical且actual materialization count为零；state只含typed assignments；budget结束产生一个full-feasibility proof并只commit
  一次；commit失败是typed unsupported/compiler bug且不返回frontier。baseline/search transitive call graph独立，有界穷举oracle证明state与
  transition coverage。

### Q51-1 专项调研：driver边界、PlanningState与ownership

MLIR pass infrastructure要求operation pass只处理其anchor及nested IR，pass failure会停止pipeline；它适合对一个IR实例做analysis/
transformation，不适合持有全局frontier、在多个不物化choices间选winner或拥有外部work allowance。MLIR的named/nested pipeline仍应
承载winner后的leaf transformations。LLVM MachineScheduler则把ScheduleDAG、ready state和`MachineSchedStrategy`分开：问题/合法依赖
不因选择策略改变。Q51采用同一分层，但planning对象是card-wide typed plans，controller是compiler driver library，不是假装成一个大pass。

owner按lifetime固定：compiler driver创建并关闭planning session，拥有frontier、budget、winner handoff和output transaction；
session拥有immutable source borrow、typed problem、target/cost cohort及`ObservedDependencyKey` memo；MLIR pass/analysis只处理当前
actual IR epoch；winner后的named subpipeline只做selected IR transformation。source analysis、session memo和winner IR三者不共享
cache或引用，materialization开始即关闭前两者。

current `runCardExecutableSearch`正相反：它物化semantic alternative clone，构造`UnifiedPhysicalDataflowAssignment`，每个complete point都
生成CardModule、运行Q50.0并把accepted executable放进incumbent；`SearchWorkBudget.maximumEvaluations`直接计完整编译次数。该owner必须
整体替换，不能在循环外挪一个clone后继续沿用。

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

F foundation在各variant之间作为pure gate；J foundation是从`InitialBufferState`重算的derived problem，不进入state。`ScheduleState`通过F full
query后才转换成move-only `PhysicalDataflowPlan`，proof仍是derived result而不存进plan。严格prefix顺序不删除组合：前一轴的所有siblings
仍在frontier，后轴失败回到共同owner继续展开其它prefix；K改变occurrences后明确经过新的`BufferState`而不是复用pre-K plan。

每个axis plan是immutable value object，可由多个states共享`shared_ptr<const PlanT>`以减少复制；共享的是该轴的语义值，不是parent链、
transition history或mutable cache。equality、canonical order和dedup逐typed contents比较，pointer/hash只作局部加速且必须以完整相等检查
消除collision。state不保存proposal来源、ordinal、failure history、score、lower bound、work count或analysis snapshot。

#### Ownership与lifetime

| Owner | 独占/借用内容 | 明确不拥有 |
| --- | --- | --- |
| outer compilation transaction | source `OwningOpRef`、ProgramDataHandoff、target/package/output transaction | frontier、candidate executable |
| `PhysicalDataflowPlanningProblem` | immutable analyses/facts；在outer lifetime内borrow source | IR clone、winner、offset |
| `PhysicalDataflowPlanningSession` | frontier、typed state memo、causal rejection cache、work allowance、best full-proof plan | baseline result、CardModule、package |
| `PlanningState` | immutable typed axis plans及variant tag | parent/history、derived analysis、IR pointer |
| selected `PhysicalDataflowPlan` | move-only complete assignment；从session移交commit | solver witness/offset、source ownership |
| winner commit transaction | 唯一新Card subtree、plan-ID→actual Value/Event短生命周期映射 | loser state、fallback selector |
| Q50.0 | move-only CardModule→accepted CardExecutable及actual resource facts | planning frontier、repair path |

session销毁时frontier/memo/failure cache全部释放；selected plan消费后不能再次commit。commit前source从未mutation；commit只在outer module父级
建立一个新Card subtree，失败由C RAII guard擦除该subtree。Q50.0 accepted后ProgramDataHandoff才由outer transaction继续下传，search
不能为candidate复制或提前移动它。

#### Policy isolation与driver routing

```text
normalized TensorProgram
  +-- optimization=none
  |     -> buildCanonicalPhysicalDataflowPlan(...)
  |
  +-- optimization=search
        -> planPhysicalDataflow(...)

selected PhysicalDataflowPlan
  -> materializePhysicalDataflowPlan once
  -> CardModule-to-CardExecutable named lowering/verification
  -> target/package transaction
```

两个controller只共享immutable analyses、final plan schema、materializer和downstream pipelines。baseline plan不进入search session，search不
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
  ...                     // policy-free winner construction
```

`CardExecutableSearchResult`删除并由`PhysicalDataflowPlan`结果替代；`CardExecutableSearchSummary`/IR trace移到显式instrumentation/report owner。旧
`UnifiedPhysicalDataflowAssignment/Domain`不作为新state壳复用：其Cartesian successor、materialize、buffer-scope build拆回对应Q50 owner，
完成donor mapping后删除。`TensorProgramAlternativeDomain`及clone materialization由Q50.S common graph normalization、fixed
algorithm fact和single-winner selected decomposition替代，不进入PlanningState。
`SearchWorkBudget.maximumEvaluations`退出semantic/API；后续Q51-5定义planning work allowance，Q52才依据profile给production policy。

### Q51-1 Gate

- type/unit检查每个variant只能从合法前缀构造，K后必须重闭I/J；invalid cross-generation plan在建state时作为compiler bug失败；
- ownership test覆盖source/session/selected plan/commit/Q50.0 move顺序、early failure和析构；无double commit、ProgramData提前move或source use；
- state key在input order/hash/parallel discovery变化下稳定，pointer collision不影响equality；复制/共享plan不携parent或mutable cache；
- call-tree/static source gate证明search transitive callees不含baseline、CardModule/Q50.0/materialize/clone，baseline也不含search session/state；
- production driver两种policy分别产生final plan后汇合；named leaf pipeline与production调用同一builders，未手工拼第二链；
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
  -> FullFeasibility
```

每个箭头表示child domain的全部inputs已存在，不表示前轴local winner。F foundation在箭头前后均可运行cheap checks；其
`RequiredCoordinate`必须指向当前variant以后、且依赖已满足的真实coordinate。要求已关闭的coordinate或越过未满足dependency是内部合同
错误。若未来加入新axis，必须同批扩closed state variant、dependency table、successor、F requirement和tiny oracle，不能注册一个opaque
optional provider。

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

full feasibility只在`ScheduleState`调用。FullProof使state成为selected-plan contender；ExactRejection回到Schedule parent continuation，
但不重新物化任何东西；Indeterminate保留complete state供后续proof work。winner commit不属于ExpansionOutcome。

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
Tiles的最大值拼成虚构critical Tile。SPM/DDR offset high-water只在F/Q50.0作capacity/headroom和diagnostic，不进入winner objective。
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
遵循J event/resource DAG：hard sequential chain相加，可真正并行且resource-disjoint的branches取maximum，同一resource的mandatory work
除以exact capacity形成service lower bound，K recurrence用RecMII，H使用endpoint/min-hop/cut bounds。不同bound若可能重复计同一work，
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

只有F full proof的complete plan能进入full-plan set。第一个plan建立functional incumbent；后续只有`Better`才替换。`Equivalent`使用完整
semantic plan key作确定性tie-break，但不改变objective；`Incomparable`必须保留在session的complete nondominated set或明确降低coverage，
不能按semantic key宣称“更优”。若搜索结束仍有多个incomparable plans，driver可按semantic key选择一个用于single commit，但结果标记为
`feasible-unranked`，不声称best/optimal/gap。Q51-5定义controller与coverage的最终状态机。

exact work vector只有在所有future objective映射已证明monotone时才能支持safe dominance；普通“所有raw bytes少一点”不能越过compute/
communication trade-off。lower bound仅在与一个Known incumbent objective同cohort且`bound >= incumbent`时可prune；estimate不能prune，
也不能形成no-good。

#### Planning/actual cost parity

fixed semantic description与各B--K plan descriptors提供plan work collector；winner commit后current Instr collector独立重算exact work。按typed action/resource IDs比较
all-and-only counts、bytes、messages、waits和qualified topology work；mismatch是emitter/lowering bug。actual estimate可用同一cohort重算
作为validation/report，但不会再次选择winner、返回frontier或把package measurement回灌当前compile。

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
- actual recost只验证/report，不进入controller；普通compile无cost dump、statistics或profile读取，source和planning states不变；
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

`ExtensionClosed`表示proof producer已证明：任意保留这些bindings的future assignments均失败；它可命中partial state。若verifier只能证明
一个完整point失败，就生成含该complete plan全部bindings的`ExactCompletePlan`，不能自行删字段。proof本身是closed typed sum，例如
exact demand hole、empty exact domain、capacity certificate、layout constraint contradiction、message cover、resource/deadlock cycle或target
field range；diagnostic text只用于展示。

#### 唯一合法的产生路径

```text
record(result):
  switch result:
    ExactRejection with producer-supplied CausalBindingSet:
      validate every binding occurs in result.observed dependencies
      validate proof says ExtensionClosed or ExactCompletePlan
      normalize and insert ForbiddenAssignment
    Consistent / Deferred / Unsupported / Indeterminate:
      insert nothing
    compiler bug / winner commit failure / actual parity mismatch:
      stop compilation; insert nothing
```

F/Q50 witness producer必须同时给出all-and-only observed dependency key与causal bindings；cache不从“largest demand owner”、node邻接、group
members、last changed axis、error location或diagnostic字符串猜cause。cost bound/estimate、local Pareto、proposal failure、solver timeout和
resource exhaustion永远不能产生ForbiddenAssignment。Q50.0只验证唯一winner，故其actual rejection也不回写cache；若planning full proof
后commit失败，按Q51-1是contract bug/unsupported，不是另一个learning机会。

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

current `CardExecutableSearch`在每个actual candidate失败后只增`exactRejected`并继续，没有真正causal cache；`isProvenExactTileMemoryPlanningFailure`
又从actual relation attribution推测哪些node受影响。终态删除actual candidate loop及其failure-to-search path。SPM/DDR capacity learning由F
plan-level `CapacityCertificate`产生typed bindings；Q50.0的actual failure classification只服务唯一commit diagnostic/bug boundary。

实现文件为`ForbiddenAssignment.*`和`ExactRejectionProof.*`；它们只依赖PlanningState/Q50 typed witness schema，不include CodeGen、diagnostic
parser、MLIR operation或cost model。

### Q51-4 Gate

- 每类Q50/F exact witness有direct binding/proof test；删除任一真正causal binding产生至少一个可行sibling反例，证明不能再generalize；
- missing coordinate、different scope、different generation、different target session均不匹配；完整相等、subset/subsumption和hash collision正确；
- Deferred、Unsupported、Indeterminate、proposal exhaustion、cost worse/unknown、actual parity/commit failure逐项证明cache size不变；
- 在每个axis sibling注入failure，命中后parent continuation仍发出全部其它children；cache off/on有界穷举accepted set/objective/winner相同；
- input/hash/parallel insertion顺序不改变normalized entries或selected result；cache无pointer/string/ordinal、cross-session persistence、default log/
  stats或automatic core minimizer；
- source/call-tree gate删除actual rejection→search learning和broad node attribution；remaining Q50.0 classification没有planning consumer。

### Q51-5 专项调研：deterministic best-first、branch-and-bound与coverage

A*的最优性依赖admissible heuristic与明确path-cost合同；AND/OR anytime研究还指出，深度优先分解和尽快改善incumbent之间存在真实取舍。
Wafer的axis prefix不是普通edge-cost path，因此不冒充标准A*。controller采用deterministic best-first改善time-to-first，用Q51-3独立证明的
complete-plan objective/lower bound做branch-and-bound和coverage；priority estimate本身没有剪枝权限。

#### Planning work allowance不是candidate编译次数

```text
PlanningWorkKind =
  AxisSuccessorStep
  | RelationPropagationStep
  | BoundDerivationStep
  | ScheduleSearchNode
  | PackingSearchNode

PlanningWorkAllowance
  remainingCredits: uint64
  reserve(kind, count) -> Granted | Exhausted
```

一credit是对应kernel文档化的一个deterministic primitive step，用于可重放中断，不声称各kind wall-time相等。query在开始一个不可再分的step前
reserve；失败则不修改cursor/cache并返回Indeterminate/Paused。local solver接收从同一allowance预留的node budget并回报实际消费。并行
task在提交前按semantic key顺序reserve，未获额度的不启动，因此线程完成顺序不改变visited set。

这个allowance只限制planning successor/query/solver work：CardModule materialization与Q50.0计数恒为零，旧
`maximumEvaluations`语义和`--search-max-candidate-evaluations`类CLI全部退出。Q51定义机制与test allowance；Q52依据fresh profile决定
production默认work/time policy。取消/outer deadline只在step边界停止，结果coverage按保留的continuations和bounds计算，不能把wall timeout
说成NoSolution。

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
planPhysicalDataflow(problem, allowance):
  frontier = {SpatialContinuation(problem)}
  forbidden = {}
  fullPlans = {}

  while frontier not empty:
    if allowance cannot reserve next controller step:
      return finishBudgeted(frontier, fullPlans)

    item = pop minimum FrontierPriority

    if item is a state and forbidden.matches(item):
      continue

    analysis = derive work/bound/estimate for item
    if hasKnownIncumbentObjective(fullPlans) and
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
      feasibility = queryFullCoordinateFeasibility(item, allowance)
      if FullProof:
        cost = analyzePlanCost(item)
        insert/update complete nondominated plan set
      else classify exactly as Q51-2
      continue

    enqueue continuation for the state's next static axis

  return finishExhausted(fullPlans, unsupported, indeterminate)
```

strict `bound > incumbent`只删除不可能改善objective的state；`bound == incumbent`仍展开，以发现同objective但更小semantic plan key。
若未来为tie key证明minimum completion key，可扩为lexicographic bound；当前不猜。Incomparable full plans都保留；Known objective的
`Better/Equivalent`按Q51-3更新同一comparable incumbent。

#### Coverage与typed result

```text
PlanningCoverage =
  ObjectiveOptimal
  | FeasibleWithBound {lowerBound, incumbent, gap}
  | FeasibleUnranked
  | BudgetedFeasible

PlanningResult =
  SelectedPlan {move-only PhysicalDataflowPlan, coverage}
  | ExactNoPlan {proof summary}
  | UnsupportedPlanSpace {features}
  | IndeterminatePlanSpace {causes}
  | WorkExhaustedWithoutPlan
  | CompilerBug
```

- `ObjectiveOptimal`：frontier已exhaust/pruned，所有potentially better states有Known admissible bound，complete estimates同cohort可比；若存在
  equal-objective states，也已闭合semantic tie；它只证明current planning objective optimum，不外推真实板端最优；
- `FeasibleWithBound`：预算停止但所有未展开continuations仍保留，global frontier lower bound Known且full-plan objective可比；gap按同一
  unit/cohort给出；
- `FeasibleUnranked`：已有FullProof plan，但存在incomparable complete plan或objective Unknown；semantic-min plan可用于functional single
  commit，但不声称best/gap；
- `BudgetedFeasible`：已有FullProof plan，但至少一个remaining state缺admissible bound，或Q52 future heuristic永久丢过state；
- `ExactNoPlan`：frontier完全exhaust，所有branches都由exact proofs关闭，且没有Unsupported/Indeterminate；
- Unsupported/Indeterminate会阻止ExactNoPlan；预算内没有FullProof plan必须返回明确failure，不能运行/返回baseline。

若frontier所有Known bounds严格大于incumbent且不存在Unknown/incomparable/tie-pending item，可提前得到ObjectiveOptimal；否则只在exhaust后
判定。coverage、work counts和frontier reason只在显式report sink请求时详细输出；核心`PlanningResult`只保留用户/下游需要的typed状态。

#### Determinism、终止与复杂度

finite domains + Q51-2 continuations保证unbounded exact run终止。priority queue push/pop为`O(log F)`，每item cost query另计对应Q50 work；
最坏仍访问指数个states、memory为`O(F + continuations + complete incomparable plans + forbidden assignments)`。branch-and-bound/no-good关闭
只减少work。相同source/target/allowance在不同hash seed/线程数下reserve序列、visited semantic keys、coverage和selected plan相同。

real workload不承诺Q51阶段全局穷尽；机制允许bounded result。Q52才根据profile引入memo/DP/LNS和可能有损policy，并必须相应降低coverage。

#### Current迁移

`SearchWorkBudget{maximumEvaluations}`、`SearchWorkCounts{generated,evaluated,accepted...}`和
`CardExecutableSearchCoverage`均围绕actual compile loop。终态由`PlanningWorkAllowance`、可选`PlanningWorkReport`和上述
`PlanningCoverage`原位替换；不留compat wrapper或旧CLI。`winnerUpdates/proposalDetail/lastDetail`不进semantic result，显式report由typed
events收集，普通compile不构造字符串。

实现文件分为`PlanningController.*`、`PlanningWork.*`和`PlanningCoverage.*`，priority policy不塞进state/session owner文件。

### Q51-5 Gate

- real Q50有界穷举domain的controller与flat oracle比较complete set、best objective、tie key和ExactNoPlan；不是mock-only graph；
- 在每个work cutoff运行并验证：有plan时coverage精确、无plan时WorkExhausted、continuations无丢失；resume到exhaust与一次unbounded结果相同；
- adversarial estimate顺序、Unknown bound、equal-cost smaller key、incomparable plans证明estimate不剪枝、strict bound/tie正确；
- branch-and-bound/no-good分别关闭后objective/winner不变；每个pruned prefix由flat completion minimum验证确实不能改善；
- parallel query延迟/hash seed/priority provider顺序变化不改reserve/visited/winner；overflow和cancellation保持typed coverage；
- work counters证明planning CardModule/Q50.0/clone为零；旧maximum-evaluation API/CLI/coverage/summary strings source与help零残留；
- 普通compile不创建report/stats/log，显式report完整但不影响priority、allowance或result。

### Q51-6 专项调研：winner一次构造、lowering与verification

MLIR pattern transaction只覆盖单个op的in-place modification；DialectConversion rollback还会延迟部分erase/replace且有显著bookkeeping成本，
并不等价于整个compiler stage transaction。Transform dialect也要求：precondition failure应发生在mutation前，mutation后违反postcondition是
irrecoverable。Wafer winner跨CardModule、16个TileModule、buffer/movement/event和多个lowering stage，正确边界是“先pure prepare，后只在
一棵新Card subtree内构造，失败整棵删除”，而不是让每个builder各自clone/rollback。

#### Prepare不写IR

selected plan进入commit前被move消费，并一次生成：

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
  expected normalized resource/work problems
```

它由C/G--K各自的prepare kernel贡献typed pieces，随后做一次card-scoped totality/dependency check。prepare只borrow source和selected plan，
不创建Module/Func/TileRegion，不调用pass，不分配actual offset，不建立IR trace。缺plan ID、跨generation关系、unsupported lowering capability、
message/field range或resource mismatch在这里返回typed failure；F full proof与prepare observed dependencies必须一致。

#### 一个Card subtree transaction

```text
compileSelectedPhysicalDataflowPlan(parentModule, source, selectedPlan):
  prepared = preparePhysicalDataflow(source, move(selectedPlan))
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
  run named TileRegion-to-Instr subpipeline on selected Tile roots
  compare actual normalized resource/work problems with prepared expectations
  executable = run Q50.0 CardModule-to-CardExecutable once
  if any step failed: transaction destructor erases the complete new subtree
  else: transaction.commit(); return executable
```

`CardSubtreeTransaction`只拥有本次新建Card op及其descendants；不snapshot/clone parent或source，不接管其它现有ops。commit前source
TensorProgram不被erase/rewire；logical source cleanup若需要，由拥有parent的外层conversion在accepted后执行。失败删除内容可恢复，且明确
告诉用户删除的是未提交新subtree，不影响source/ProgramData。

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

- 禁止source TensorProgram、complete candidate、loser CardModule、probe Module/Func和winner rematerialization；
- selected SCF loop pipelining可在**新subtree内**由pinned `scf::pipelineForLoop`机械生成prologue/steady/epilogue，使用`IRMapping`，这是
  one winner transform，不是candidate isolation；
- Q50.0/target output可以按all-and-only Tile domain将selected CardModule拆分/移动为每Tile唯一输出，并使用既有bounded host parallelism；
  每Tile/variant只lower/translate一次，不把16 Tiles变成16次logical analysis/search；
- test-only有界穷举oracle可从独立fresh source逐plan actualize来核对domain，但不调用production search entry，也不保存为下一candidate模板。

#### Failure classification

prepare前置条件失败若源/target确实无current capability，返回typed Unsupported；malformed selected plan、F/prepare generation mismatch、
emitter extra/missing relation、actual resource/work parity mismatch及full-feasibility proof后packing失败是CompilerBug。pass/external tool/environment错误按其
owner返回typed failure。任何failure都会结束该compile；不存在“exact rejection后回frontier”“换next winner”或baseline fallback。

ProgramDataHandoff仍由outer transaction持有；只有accepted Q50.0 result确定后才move入CardExecutable。失败/unsupported不能discard/adopt
candidate data或写package/output directory。

#### Named pipeline parity

search driver只负责plan选择和调用上面的commit facade。TileRegion→Instr、memory planning、Direct-DTE binding、resource verification分别用
current named subpipeline/leaf pass builder；`wafer-opt`测试对一个已由test fixture构造的selected CardModule调用同一builders。不会注册
“search pass”或另一条direct mutation pipeline。

#### Current迁移

| Current | 终态 |
| --- | --- |
| `UnifiedPhysicalDataflowDomain::materialize/buildBufferingScopes` | C/G--K prepare/emitter；domain不include apply |
| `materializeTensorProgramAlternative` root clone | Q50.S common normalization + winner selected attention Linalg decomposition |
| per-assignment `compileCardModuleToExecutable` | single post-planning Q50.0 call |
| `CardExecutableSearchResult{executable, IR trace}` | `PlanningResult`→one commit→CardExecutable；trace仅显式outer instrumentation |
| DataMovement/Buffering/InstructionSchedule post-hoc apply | H/I/J/K construction-time builders |
| candidate failure loop and accepted executable incumbent | plan-level F/cost incumbent；commit failure terminal |

实现文件为`PhysicalDataflowPreparation.*`、`PhysicalDataflowEmitter.*`、`CardSubtreeTransaction.*`和
`PhysicalDataflowMaterializationVerifier.*`；Q50.0仍在CodeGen/Executable owner。文件不以Q51/阶段编号命名。

### Q51-6 Gate

- planning work counters始终CardModule=0/Q50.0=0；selected result消费后Card subtree=1、Q50.0=1，重复commit在type/ownership层不可表达；
- mixed DAG分别选择non-default spatial/region/temporal/layout/communication/slot/structure/schedule，actual IR按plan IDs all-and-only对应；
- failure injection覆盖prepare首末、每类builder、mid-Tile、materialization verifier、TileRegion→Instr、resource parity和Q50.0；source/parent除新
  subtree外byte/operation不变，失败后新subtree为零；
- F plan/actual resource parity与Q51-3 work parity通过；hidden allocation/message/wait/worker或lowering temporary稳定报CompilerBug且不重选；
- custom/generic roundtrip、verify-each、named/production leaf-pipeline parity、CardModule→Instr→CardExecutable通过；
- call-tree/source gate证明production search planning不含materialize/Q50.0/clone，commit不含frontier/next assignment/baseline；旧post-hoc
  apply和candidate compile loop零残留；
- 16-Tile selected winner每Tile只emit/lower/translate一次，independent downstream host jobs使用bounded parallelism且结果顺序稳定；普通
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
         buildCanonicalPhysicalDataflowPlan(facts)
       Search:
         planPhysicalDataflow(facts, targetCostFacts, planningAllowance)
  -> one PhysicalDataflowPlan
  -> compileSelectedPhysicalDataflowPlan once
  -> target/package transaction
```

`OptimizationPolicy`是closed `None | Search`；current `OptimizationConfig::search(evaluations)`及
`getMaximumSearchCandidateEvaluations()`删除。Q52的production allowance/priority是compiler-owned search policy facts，不重新塞进public
optimization identity；test可从internal API注入allowance，不增加长期user candidate-count selector。

common `analyzePhysicalDataflowProblem`只返回StructuredDAG、target/topology、exact relation与semantic IDs等immutable facts，不创建baseline
coordinate、proposal、frontier、score或winner。baseline/search各自从相同source transaction borrow facts，但controller objects和result
ownership独立。

#### 允许共享的API必须以closed plan为边界

| Shared | 输入/输出 | 禁止携带 |
| --- | --- | --- |
| policy-free analyses | current IR/target facts -> typed facts | policy、proposal order、winner |
| Q50 domain kernels | typed parent -> choices/witness | baseline fallback、actual IR |
| `PhysicalDataflowPlan` schema | complete typed assignments | origin policy、score、coverage、history |
| prepare/emitter/verifier | closed plan -> new Card subtree | frontier、next-choice callback |
| Q50.0/named lowering | selected CardModule -> CardExecutable | candidate loop、policy switch |

baseline可以调用某个Q50 **single-coordinate query/validator**验证自己已经直接构造的canonical choice；它不能调用返回domain/options、
continuation、proposal或“选一个assignment”的search API。search可以使用同一canonical choice作为普通domain member/proposal，但不能调用
`buildCanonicalPhysicalDataflowPlan`或接收其plan/result。

#### Include/CMake/call graph

```text
Planning/PhysicalDataflow/Common  <- analyses + plan schema
Planning/Baseline                -> Common + narrow Q50 queries
Planning/PhysicalDataflow/Search -> Common + all Q50 domains
Conversion/...                   -> Common plan schema, no policy controller
CodeGen/Executable               -> policy facade + common commit/Q50.0
```

Baseline library不得include/link Search headers/library；Search不得include/link Baseline或CodeGen executable compilation。policy facade可以依赖
两者，但只拿move-only plan results。Conversion/Transforms不得反向依赖任一controller。静态source/CMake gate与link-only tests同时检查，避免
通过forward declaration或utility header偷渡调用。

#### Failure绝不fallback

- None canonical analysis/legalization/plan failure直接返回baseline typed failure；不创建search session；
- Search的ExactNoPlan、Unsupported、Indeterminate、WorkExhaustedWithoutPlan直接返回search typed failure；不运行baseline；
- 两条policy的selected plan commit失败都结束compile，不调用另一个controller或重新选择；
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
`OptimizationConfig` candidate budget。终态把branch上移到**plan producer**，共同commit只有一份。测试用failure injection不能再要求
“只有search可用”的无关耦合；每个injection按其真实outer transaction capability决定。

### Q51-7 Gate

- static include/CMake/transitive call graph证明Baseline↛Search、Search↛Baseline/CodeGen、Conversion↛controllers；policy facade恰一个switch；
- spies/work counts分别证明None search-session=0、Search baseline-controller=0，两者plan=1/commit=1/Q50.0=1；
- baseline/search各类plan failure、work exhaustion和共同commit/target/package failure都不调用另一policy；diagnostic owner准确；
- public CLI正负例只接受`none|search`，旧candidate-evaluation option/help/API/test零残留；default policy没有fallback trace；
- explicit timing on/off得到相同plan key/IR/package，普通输出无compile stats；
- matched A/B使用两份fresh source/ProgramData/output transaction，故意尝试跨policy传plan/executable在type/API层不可表达；
- 两条policy选同一显式plan时，common emitter/named pipeline产生等价actual IR；不同plan差异来自typed assignments，不来自branch-specific lowering。

### Q51-8 专项调研：全轴oracle、production parity与总迁移门禁

MLIR/LLVM把内部API unit、IR regression、diagnostic和whole-pipeline integration分层；Alive2的translation-validation实践则说明，与其只
相信transform实现，不如对本次具体input/output关系做独立检查。Wafer不能直接套Alive2语义，但应采用同一原则：Q50逐轴reference证明
domain，Q51 independent composer证明全轴组合，test-only actualization验证plan→IR；production仍只物化winner。

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
apply independent F address/occupancy/field oracles
derive reference work/objective with test-local checked arithmetic
emit canonical semantic plan tuple
```

有界穷举范围为2--4 Tiles、每axis 2--3真实choices及有限event/buffer/chunk domains；它们只限制oracle test size。reference types可最终转换为
`PhysicalDataflowPlan`做commit，但enumeration/filter/key计算不调用production successor/contains/canonicalizer。逐parent比较，而不只比较
最终count，以定位第一个漏域/重复stage。

#### Oracle B：test-only selected-plan actualization

第二层只验证plan/commit合同，不进入production search：

```text
for each reference-feasible bounded-exhaustive plan:
  parse/import a fresh source transaction
  commit exactly that plan directly
  run Q50.0 once
  require accepted CardExecutable
  compare plan/actual resource problems and work metrics
  derive ActualPlanKey from typed IR relations, not printed names/pointers

for selected reference-rejected plans:
  require prepare/F direct typed rejection before mutation, or the exact
  negative boundary owned by its stage
```

每个plan使用fresh parse而不是clone前一个actual result；ProgramData和output owner独立。这个runner可以为tiny proof物化多个plans，因为它只在
test binary中直接调用commit facade；production `planPhysicalDataflow`及其transitive callees永远不可达它。

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
| feasibility/cost | first-fit false negative、exact certificate、Unknown/incomparable、admissible bound |

另有至少三个联合反例：局部最小movement因schedule失去overlap而败；最大fusion因buffer/lifetime败给region cut；最短hop placement因
representation/relay/worker critical path败给另一placement。case只实例化通用typed关系，不把operator/shape名写进算法。

#### Production controller parity

同一tiny problem依次运行：

1. reference flat enumerator，得到raw/feasible plan sets和reference best/incomparable set；
2. production canonical continuations，关闭proposal、no-good和bound pruning，比较每variant/parent/final keys；
3. 逐项开启proposal、ForbiddenAssignment、branch-and-bound、parallel query，feasible set与selected plan不变；
4. 在每个work cutoff停止并resume，最终结果与unbounded run一致，intermediate coverage准确；
5. production public search选择plan后只commit一次，ActualPlanKey与selected plan一致。

priority变化可改变visited prefix但不能改变exhausted exact result；budgeted run只比较其coverage所允许的claim。Unknown objective时reference与
production都保留incomparable set并得到`FeasibleUnranked/BudgetedFeasible`，不能强行比较。

#### 零loser物化证明

test observer是显式注入sink，记录semantic work而不改变控制流：

```text
planningCardModules = 0
planningTileRegions = 0
planningInstrLowerings = 0
planningQ50Invocations = 0
selectedPlanCommits = 1
selectedQ50Invocations = 1
selectedTileLowerings = available Tile count, each exactly once
packageTransactions = 1
```

observer关闭时普通对象不创建；source静态call tree还必须证明planning library不link commit/CodeGen symbols，防止计数漏埋点。IR dump只可查看
accepted winner，不能出现candidate index/plan list。

#### Current/donor总迁移矩阵

| Current Search files/capability | New owner | 删除前必须存在的direct witness |
| --- | --- | --- |
| `Attention*`, `DecodeAttentionAnalysis`, `TensorProgramAlternative*` | Q50.S one attention op/classifier/work description/selected decomposition | Q/K/V detection、FA/FD、block/partition ownership、state merge、no root clone/algorithm axis |
| `ComputeImplementation*` | 10号deterministic lowering；真正semantic alternative留给Q48 | Natural/reciprocal exact applicability逐项判定，不借Q50.S恢复generic algorithm registry |
| `SpatialPlacement*` | Q50.B | factor/subset/embedding/merge、automorphism、proposal/reference |
| `SingleRootRegion*` | Q50.C | exact RootRegionWork、multi-producer support cut、one subtree construction |
| `CoupledRegion*` | Q50.D | connected partitions、execution/use/replica、nested relation |
| `TemporalTiling*` | Q50.E | complete vectors/orders/tails/nested scopes |
| `ScopedFeasibility*` | Q50.F | closed results、certificates/full proof/oracle/parity |
| `PhysicalRepresentation*` | Q50.G | constraint solver proposals、version DAG、actual builder |
| `SimpleRoute`, `DataMovement*`, `DataMovementApply*` | Q50.H | generic payload-state plans、classic topology proposals、winner emitter |
| `Buffering*`, `BufferingApply*` | Q50.I | storage/slot/lifetime plans、construction-time builder |
| `InstructionSchedule*` | Q50.J/Q63 | event/resource domain、closed schedule、wait/join emitter/verifier |
| `StagePipeline*` | Q50.K | Serialized/SCF/periodic structures及K→I-post-K→J reclosure |
| `UnifiedPhysicalDataflow*` | Q51-1/2 state+continuations | full reference equality；mixed-radix/materialize facade均无剩余能力 |
| `CardExecutableSearch*`, `SearchWork*` | Q51-3--7 cost/controller/work/coverage/routing | zero-loser work、budget/coverage、single commit、policy isolation |
| historical rank/candidate/frontier/NoC providers | corresponding Q50 owner或明确淘汰 | archive donor matrix逐项有current source+test；“未注册”不算迁移 |

迁移按owner分批提交：先加入new owner和tests，再改public caller，最后删除old file/header/CMake/test。不能一次删整目录后以新Core能编译代签。
最终`lib/Wafer/Planning/Search`和`unittests/Planning/Search`旧布局清空；新代码分别位于
`Planning/PhysicalDataflow/{Search,Feasibility,...}`及对应tests，baseline保持独立目录，不留compat include或forwarder。

### Q51-8 / Q51 Final Gate

- 每个Q50 axis direct oracle先通过，再运行Q51 parent-by-parent/full-plan oracle；没有mock-only Core或只看final count；
- test-only actualization对全部tiny reference-feasible plans接受，并完成F resource/Q51 work/ActualPlanKey parity；negative plans命中准确owner；
- proposal/no-good/bound/parallel逐项开关不改exact feasible set、best objective/incomparable set和selected plan；任意cutoff/resume正确；
- public search source→one plan→one CardModule/Q50.0→CardExecutable→package/no-card轻量链通过，planning零IR、winner一次commit；commit failure
  terminal且无baseline/next-plan fallback；
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
  Q51已闭合的IR-free planning session、typed continuations、cost/bound/coverage和single-winner commit；显式profiling invocation可附加
  `PlanningProfileSink`，source/target facts仍immutable。
- Current stage responsibility:
  先测量state growth/repeated work/time-to-first/RSS，再在有完整future-boundary proof的位置加入pure memo、component DP、stronger
  admissible bound和profile-driven priority；必要时再加入明确有损的anytime/LNS policy。不得改变Q50合法域或Q51 exact oracle。
- Output IR / files:
  planning仍只输出`PlanningResult`和一个selected plan；ordinary compile不写profile。显式profile生成独立报告，winner仍只commit一次。
- Downstream consumer:
  Q51-6 common commit、Q53 production/no-card/board qualification。
- User-level driver / named pipeline:
  public `search` policy内部的compiler-owned planning policy；不增加mechanism、beam、candidate-count或LNS CLI selector。
- Explicit non-goals:
  不缓存IR/offset/solver stack，不跨invocation复用未验证state，不从workload/op/shape名选策略，不用memo hit或learned estimate签发legality，
  不让profile instrumentation进入默认路径。
- Done criteria:
  safe optimizations on/off与Q51有界穷举exact set/objective/winner一致；representative heavy workload在bounded policy下获得full-proof plan且只commit
  一次；profile说明每项优化对应的实测hotspot和收益，coverage准确反映是否丢state。

### Q52-1 专项调研：profile、future-boundary memo与component DP

Halide learned autoscheduler和Ansor都用hierarchical/sketch search加cost model解决巨大schedule space，但beam/evolution只覆盖其定义的子集；
它们可迁移的是“先profile热点、分层生成proposal”，不能成为Wafer exact-domain证明。AND/OR search说明separator/treewidth可指数减少重复，
MLIR AnalysisManager则给出IR analysis按preservation失效的成熟边界。Q52因此先做无损memo/DP，任何有损search留给Q52-2。

#### 只在完整Q51链上profile

Q51 new path通过全轴oracle和轻量source-to-package后，才运行显式bounded profile。记录：

- 每axis/variant generated、deduplicated、exact-rejected、deferred、unsupported、indeterminate、continuation resume及peak frontier；
- relation/demand/domain/F/J/cost query的call count、work credits、wall/CPU和cacheable input重复率；
- per-Tile/per-scope resource problem descriptor数量、extensionally equal数量，以及16-Tile automorphism/同构失效原因；
- proposal family到first full-proof plan的贡献，full-feasibility fast witness/exact solver nodes和time-to-first-full-proof；
- known/unknown frontier bound、comparable/incomparable full plans、incumbent objective/coverage随work和wall变化；
- peak RSS按state/continuation/plan object/memo/forbidden/full-plan set分类；
- planning CardModule/Instr/Q50.0始终为零，selected commit/Q50.0各一次并单独计时；
- 与fresh独立`none`编译的actual work/运行证据只在外层报告比较，不进入search session。

profile sink只在明确请求的profiling run构造，普通unit/lit/compile无这些counters/string/report。重型LLaMA search首次出现在这里，并使用
current new path；旧search数据和历史日志不能回放。

#### 三类memo owner

1. **IR-derived facts**由policy-free builder定义；pass内可由MLIR AnalysisManager按operation scope缓存，planning driver在session中
   直接拥有同一builder的typed result，不保留`Analysis *`或另建epoch/fingerprint cache。
2. **assignment query memo**只在一个planning session内缓存pure typed result：domain descriptor、exact demand projection、F problem/result、
   J event graph、work/bound/estimate等。每类query声明`ObservedDependencyKey`，key包含它实际读取的axis choices和target facts。
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

两个state只有在以下性质有proof时才可共享suffix：相同legal successor keys；任一对应completion的F outcome相同；exact work/downstream
objective contribution相同或可由key中保存的boundary summary组合。key不能只用shape、bytes、layout、Tile count、estimated makespan或
plan hash。增加/删除一个fanout destination、relay child、slot backedge、worker join、hole Tile或output writer都必须改变key。

每个memo实现带test-only dependency perturbation：逐个修改所有可能输入，若result能变化但key不变就是contract bug。无法证明最小key时先用
完整typed parent prefix；profile再证明哪些字段可安全投影，不能为了hit rate先做窄key。

#### Component/separator DP

从current typed constraint graph构造components，而不是按function/op名字切：vertices为axis coordinates、physical versions、communication
actions、storage objects和events；constraint/resource edges来自A/F/J等唯一facts。移除separator后components必须条件独立。separator至少含：

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
  validate combined plan through normal F/J queries
  leave canonical global continuations available for every non-emitted sibling
```

DP entry保存typed boundary assignment、exact work、admissible bound和构造complete plan所需的local axis plans；不保存IR或local winner。
Pareto删除只允许相同`FutureBoundaryKey`且所有future objective terms monotone；Unknown/incomparable保留。没有fixed Top-k。chain/treewidth `w`
在每separator domain size `K`下典型为`O(N*K^(w+1))`；separator过宽或global resource无法factor时直接回Q51 best-first，不运行重solver。

#### 16-Tile重复工作的处理

完整4x4 mesh上，verified colored-topology automorphism和extensionally equal scope descriptors可以让16个Tiles共享relation/domain/resource
**query result**。每个Tile的semantic IDs、plan choices、buffer/events仍独立；memo hit只重绑定typed IDs，不强迫同assignment/offset。
unavailable Tile、非对称fanout、不同local extent、relay/worker/resource facts会拆开equivalence class。prepare/emission/Q50.0对winner每Tile仍
各一次，不试图缓存或复制actual module。

#### Stronger safe bounds

component DP可把Q50.B factor lower bound、H cut/endpoint bound、I minimum slots、J critical/resource/recurrence及F capacity facts按proved
separator组合成Q51-3 admissible bound。任何restricted solver的`OPTIMAL`只对其local conditioned problem有效；要提升为global bound必须加上
其它components的admissible minima和separator work。timeout/FEASIBLE/resource exhaustion只给proposal/Indeterminate。

#### Q52-1 Gate

- 先有profile再启用每类memo/DP；report把hotspot、key cardinality、hit rate、work/wall/RSS变化与selected coverage关联；
- cache off/on、eviction、parallel single-flight和不同hash seed保持Q51有界穷举exact set/objective/winner/coverage；
- dependency perturbation逐字段证明key完整；无pointer/digest/manual epoch、IR/offset/solver/result string cache；
- chain/tree/component DP与flat oracle逐separator assignment比较local/global sets与objective，Unknown trade-off不被Pareto删除；
- opaque/known NoC、shared DDR/DTE、cross-Tile message、slot/recurrence/join分别证明正确进入separator；错误拆分有negative；
- symmetric 16-Tile case显著减少query次数，hole/asymmetric case停止错误sharing；assignment与winner actual Tile差异仍保留；
- planning materialization保持零、winner一次commit；ordinary compile无profile sink/stats/log，source/IR不变；
- 文件分为`PlanningProfile.*`、`PlanningMemo.*`、`ComponentPlanning.*`，不回到controller/state大文件或引入全局cache。

### Q52-2 专项调研：bounded anytime policy、coupling-aware LNS与LLaMA gate

Shaw/Pisinger--Ropke LNS的核心是destroy一组相关variables，再用constraint solver repair大邻域；OR-Tools current LNS实现也从active
constraints、intervals、precedences和connected components构造neighborhood，而不是随机改一个变量。Wafer采用这个结构，但repair必须走
同一Q50 typed transitions/F gate，且不在线materialize任何plan。

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

#### Constructive lane始终先拿full-proof plan

Q51 best-first continuations是constructive lane，session开始即运行。memo/DP/proposals只重排或共享pure work。LNS只有在至少一个
`PhysicalDataflowPlan`通过F full proof后才启用，不能消耗全部budget导致没有functional plan。若constructive lane在production allowance内
没有full-proof plan，返回`WorkExhaustedWithoutPlan`；不能让LNS、baseline或winner commit补救。

work allocation使用profile-derived deterministic rounds：每轮先保证constructive continuation进展，再允许一个improvement neighborhood；
具体比例由Q52实施profile决定。任何时刻只有plan-level incumbent，CardModule/Q50.0仍为零。

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
  run normal F/J/cost/bound and ForbiddenAssignment checks
  return checked complete-plan proposals or Indeterminate on work exhaustion
```

local DP/solver的Feasible solution只是checked proposal，必须转换为普通plan并通过global F full query；restricted local `OPTIMAL`不等于global
optimal。timeout/resource exhaustion不产生rejection。repair result进入Q51同一full-plan set并按同一cohort比较，不建立LNS-local winner，
也不commit。

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
2. 独立运行`search`，在profile checkpoints记录time-to-first-full-proof、incumbent objective/coverage、planning work/RSS；
3. 到selected production deadline必须返回search自己的full-proof plan，随后一次commit、package和no-card通过；
4. 未返回、commit超过一次、planning出现IR或coverage虚报均使Q52未完成，不能用延长到无限、旧日志或baseline结果代替；
5. compare只在外层报告actual IR/work和quality，不回灌当前session。

profile可在1/3/10分钟等checkpoint观察并在一次专项campaign中延长到30分钟定位停滞，但这些是诊断上限，不是production默认或普通CI
timeout。最终实现必须根据曲线拐点选择并记录一个有限production policy；代表LLaMA在该policy下实际跑通才过gate。重型case不加入普通
unit/lit/CTest，每次功能小改也不自动重跑。

#### High-risk rules仍禁止

- 按单op/edge/layout/route local winner冻结轴；
- 用shape/bytes/estimated makespan相同合并future boundary不同的states；
- 只保留最大Tile、最大fusion、最短route或一个classic communication family；
- actual packing/commit失败后原地retile/spill/rebuffer或回planner；
- 缺estimate按零、solver timeout当NoSolution、restricted bound当global bound；
- 把16-Tile同构query复用升级为16 Tiles相同assignment或actual IR复制。

#### Files

`PlanningPolicy.*`只持immutable production settings；`PlanningNeighborhood.*`负责typed destroy closure；`PlanningRepair.*`复用Q51
transitions；`PlanningProfile.*`仍是optional sink。controller只调用这些接口，不把LNS塞回一个大函数。

### Q52 Final Gate

- Q52-1 safe memo/DP各自有profile依据，on/off保持Q51 exact oracle set/objective/winner/coverage；
- 每个neighborhood dependency closure与flat reference repair set相同；至少一个联合陷阱由LNS找到而single-coordinate proposal漏掉；
- PreserveAll lane不降coverage；故意启用有损retention稳定只报BudgetedFeasible，不能因找到optimum升级；
- every cutoff/cancellation/solver exhaustion保持typed result，预算内无plan不fallback baseline；planning零IR、selected commit/Q50.0各一次；
- heavy campaign的四类workload均用current fresh source；`none`与`search`独立，代表LLaMA在最终有限production policy内返回plan并完成
  package/no-card，wall/RSS/time-to-first曲线保存为本轮证据而非compiler输入；
- public CLI/help无candidate/beam/LNS/budget selector；ordinary compile无profile/stats/log，profile sink不改变plan/result；
- priority/memo/component/LNS/repair文件独立，source/call-tree无old actual-candidate evaluator、IR cache、clone/replay或fallback；
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

这里的“selected exactly one”由Q51的一次commit ownership/call-graph gate和本次唯一published product共同证明；Q53不新增
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
| capacity-forced cross-Tile fanout/reduction | source规模使单Tile exact capacity proof失败，合法plan必须使用多个Tiles；同时有one-to-many和many-to-one payload | actual endpoint transfers/combine proof、send/receive/wait配对、合法inactive Tiles；不要求某个classic algorithm名称 | representative source/no-card；专门覆盖H/J与NoC topology合同 |
| temporal-tail and rotating-storage chain | 1024级整除/非整除iteration domain、producer/consumer wave与可重复storage | exact full+tail coverage、slot/lifetime/order和必要completion；未选pipeline时不得伪造pipeline | representative source/no-card；与Q50.E/I/J/K有界oracle互补 |
| official HF attention prefill | causal Q/K/V attention长sequence、reduction与大intermediate | normalized op为`flash_attention`；K2 temporal state、block-sized score/probability scratch、无K2 spatial partial、无attention/Linalg residual，以及完整package | explicit model campaign，非普通unit/lit |
| functional two-step KV-cache decode | 显式past key/value输入与present key/value输出，第二步依赖第一步状态 | normalized op为`flash_decoding`；至少两个K2 contributions、all-and-only coupled-state transfer/merge、一个final output owner、两个独立package/state ports及两次no-card invocation；board阶段再验证actual continuation | explicit model campaign |
| representative Llama-2 7B block | attention、MLP、residual与大parameter集合组成的真实block | Q52 finite production policy内`search`自己的full-proof plan、一次commit、完整IR/package/no-card | explicit heavy campaign；不注册为普通回归默认集合 |

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
  Q51一次commit证明仍由current路径执行。它只完成无板阶段，不单独把Q53标成`done`。

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

1. Q58、Q56、Q50.0、Q54、Q59与Q60等既有前置保持；当前严格按`tasks/progress.md`列出的37个唯一work items形成独立可评审提交；
   不得倒序，也不得用Q owner内部阶段建立第二份调度。
2. 每个domain work item完成donor迁移与direct query/test-only apply时，同批扩search controller consumer及feasibility dependencies。
   deterministic-baseline-closure不进入search；full-feasibility之后search-control-closure只闭合controller，unified-search-closure只接
   single-winner commit。旧mechanism/selector/repair/test只有在替代work item的owner、consumer和witness就位后才能删除。
3. 状态转换以 `tasks/progress.md` 为准；本计划不单独维护第二份动态状态表。
4. 每项提交前运行 fresh 定向 build/test；端到端或主线 gate 还需确认 relevant lit/CTest 实际执行而非 skip/unsupported。
5. 提交使用 `Codex <codex@openai.com>` 并附 `Co-authored-by: hehesnail <shashen008he@gmail.com>`。
6. 稳定 bug 模式进入 `memory/bugs.md`，可复用 build/debug workflow 进入 `memory/general_dev.md`；临时 profile 数字、
   workload 路径、未校准 prior 和单 case winner 不进入长期设计。
