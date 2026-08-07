# Whole-Rank Tile Dataflow Synthesis 实施计划

状态：2026-08-07 C0–C6 compiler cutover、旧owner删除和host/no-card gate均已闭合，任务达到`board-ready`。
真实板端matched Llama baseline/winner A/B与functional decode representative尚未执行，因此不是`done`。任务状态以
`tasks/progress.md`中的`whole-rank-tile-dataflow-synthesis`为准。

本计划只拆解 `tasks/06-physical-dataflow-synthesis.md` 的施工顺序、迁移删除面和验证 checkpoint；算法、IR 和
pipeline contract以该设计文档为唯一owner；exact workload、收益和board门禁只由`tasks/16-verification-contract.md`
的对应gate拥有。Q32/Q38–Q41/Q46/Q47已有的relation、layout、actual-clone、worker、NoC、overlap、计时、packing和
current ABI能力全部复用，本任务不再发明替代层。

Checkpoint严格按`C0 → C1 → C2 → C3 → C4 → C5 → C6`推进；C3与C5可拆成多个可评审子提交，但不能越过前一checkpoint的
本阶段gate。C1只允许将唯一conservative baseline经新complete-rank decision point送入一条baseline-only finalization seam，
用现有downstream mechanics完成等价package/no-card；它不建frontier、worker/order siblings或exact-failure feedback。C2只运行当前
structured IR层能解释的verifier、coverage和safe bound，不执行Tile→Instr、SPM/DDR packing或executable finalization；它只闭合
candidate generation接口，不单独宣称搜索语义完成。C3再用受global ledger管理的通用executable-finalization service取代C1 baseline-only
seam，并把exact success/failure反馈给仍存活的同一frontier；C2+C3共同闭合搜索循环。

这些编号只是实施检查点，不进入pass、pipeline、IR、diagnostic、artifact或代码类型名。它们在完整终态中的职责关系是：

| Checkpoint | 只闭合什么 | 明确不做什么 |
| --- | --- | --- |
| C0 | fresh问题基线、work/cost计数和同源IR证据 | 不改搜索语义 |
| C1 | complete-rank decision point、SPM-residency region IR/verifier、新路径conservative baseline全链等价 | 不建通用finalization frontier或选优化winner |
| C2 | 持久all-rank coordinator、global ledger、query-local structural frontier与有界actual-clone admission | 不让每个domain point都clone，不做Tile→Instr或exact packing，也不单独宣称搜索闭环 |
| C3 | 作为coordinator的executable-finalization service执行fresh completion和allocation-domain SPM/DDR/transport/ABI exact gates，并反馈typed result | 不拥有frontier、预算、repair或winner |
| C4 | 用当前校准hardware cost model从admitted-executable frontier选winner并原子提交 | 不生成候选、不重新packing |
| C5 | 在同一C2→C4闭环中加入collective residency、interface-driven native/partial reduction与隔离的implementation alternatives；Attention只作其中一个真实验证族 | 不建立算子专用pipeline、exporter形态特判或第二decision owner |
| C6 | 启用已验证的完整winner policy，删除C1起已零consumer的旧per-task/独立selector实现 | 不保留隐藏兼容分支 |

### 完成状态与 fresh evidence（2026-08-07）

- C0–C6均已闭合：production只保留complete-rank/all-rank coordinator与global ledger；structural proposal先剪枝、再按统一预算
  actual-clone；implementation/communication provider只贡献typed opaque point；所有admitted candidates经过相同completion、SPM、DDR、
  transport、ABI与final recost。hardware-cost selector、原子commit、`production|none` public policy和旧per-task/rank-frontier/
  attempt-plan/late-NoC owner删除均已完成。
- 8个logical workloads生成10个fresh packages并逐case串行通过no-card：异构非Attention rank-1 FP16 `8.18 s`、异构非Attention
  TP16 BF16 `24.29 s`、official HF prefill rank-1 FP16 `25.91 s`、BF16 `25.36 s`、functional decode rank-1 FP16
  `332.91 s`与BF16 `331.60 s`（两项各含两个连续cache-state packages）、official HF Llama-2 7B block TP16 FP16
  `327.83 s`与BF16 `320.30 s`。这些是同轮CTest wall observations，不是架构timeout或性能阈值。
- official HF prefill和functional decode均由production实际选择合法implementation sibling；decode第二个package消费第一个package
  返回的K/V state。两个Llama blocks只验证多元workload、TP16和完整source-to-package/no-card链，不要求implementation alternative获胜。
- fresh host证据包括默认C++ unit `832/832`、独立NoC compiler-integration `33/33`、lit `211/211`且无
  skip/unsupported、BoardIO `39/39`、PyTorch board common/cases `2/2`、capture contract `21/21`；campaign catalog与
  source-organization静态合同也通过。
- 无卡阶段已达到`board-ready`。真实板端matched Llama baseline/winner A/B与functional decode representative未执行；这些外部门禁
  通过后才能将Q49标为`done`。Q49已解除Q48的依赖，但本计划不启动Q48。

## 施工边界

```text
post-SPMD all-rank transaction of complete-rank structured IR
  -> interface-derived per-site / per-connection domains
  -> one coordinated query-local structural frontier and deterministic work budget
  -> bounded actual-clone admission for joint region-partition / traversal-fusion / tile-loop / physical-dataflow plans
  -> finalization all-rank Tile candidate
  -> per-rank complete-entry Instr variants with worker/fixed-slot/ready-order
  -> erase and fresh-rebuild completion once on each finalized rank-entry Instr variant
  -> derive fixed SPM allocation problems from all roots / lifetimes / coexistence and validate all-and-only coverage
  -> derive DDR placement problems from explicit arenas/domains, then post-memory Direct-DTE / ABI exact evaluation
  -> atomic all-rank ExecutableBundle
```

施工期间始终满足：

- domain point和structural proposal只在一次coordinator invocation内短暂存在，可携带typed决定和current-IR-derived通用facts，
  供DP/Pareto beam、equivalence、dominance和预算准入使用；它们不能跨pass、序列化、发布或进入late exact gate。只有统一
  actual-clone budget准入的代表才物化到complete-rank IR并销毁proposal；accepted语义仍只存在于actual current IR，不引入
  persistent、serialized或cross-stage component/fusion graph、layout plan、repair sidecar或新schema。
- 新算法只按 SSA、structured interfaces、`IndexRelation`、effects、typed numeric/target capability分派；禁止模型名、
  参数名、固定shape和字符串op-name matcher。
- `tile.region`物化一个SPM residency domain；static rank entry可包含一个或多个non-nested regions。region op语义固定，
  region partition则与tile、traversal、residency/materialization和communication联合搜索。跨region data显式走DDR，SPM root/alias
  不跨界；region不是独立artifact、launch、全局completion或lowering提交单元。仍访问region-owned root的pending work必须在
  `wafer.tile.yield`内侧由exact-participant join闭合后才能释放root，无关pending work保持live并跨region传播。
- 搜索必须保留并比较single-region fused-small-tile、multi-region separated-large-tile、same-region separated traversal和
  selective spill等actual candidates；遍历拆分、per-value spill与region cut相关但互不等价，region数、spill数或tile大小
  都不能单独作为目标。
- allocator、conversion、completion和all-rank verifier仍是独立owner；只有06 decision owner生成alternative和选winner。all-rank
  coordination、全局预算与frontier在C2就是candidate construction的前置不变量，不是C4事后补上的正确性步骤。
- DDR按all-rank aggregate bytes/executions计量并审计max-rank issue；GS按max-rank tile-local work计量并审计aggregate；
  completion按max-rank steady/nonterminal/total participant waits及critical-path位置计量，join op count只作次级统计。
  direct dominance还必须覆盖compute/recompute、tile utilization、NoC、Instr、descriptor/resource、SPM movement、critical-path和all-rank coupling；
  final selector中任一`Unknown`都保持不可比较，相同reason/disposition也不表示相等。`ProvenBenefit`只来自conservative bound分离；
  同explicit sequential schedule下Known primitive no-regression与同一control resource的20% changed-work tradeoff只能形成
  `EstimatedBenefit`，audit summary只作guard。
- baseline永远通过同一生产pipeline和late exact gates，不调用被退役的兼容路径。
- 每个checkpoint必须包含真实source到actual rewrite再到该阶段自己拥有的verifier/gate；除C1 baseline-only等价验证
  seam外，通用executable-finalization service从C3起才要求。
  空框架、手工pass或局部fixture不算闭合。

## C0：Fresh Baseline 与可审计 Work Counters

目标：冻结实现前事实，建立能同时解释 DDR、GS、NCC/join、Instr 和 compile-search work 的同源计数。

实施：

1. 扩展现有`InstructionProgramCost`/`estimateStats`事实源和invocation-local diagnostics，不另造第二套count逻辑；独立dump parser
   只作test oracle。计数明确区分static site、static-trip loop-expanded execution work、conditional path lower/upper bound、
   symbolic/Unknown和Q9 runtime-measured count，不写manifest、package、schema或长期IR attr。
2. 分别记录每rank与all-rank aggregate：
   - RDMA/WDMA calls和bytes；
   - GS/pack/unpack calls和bytes；
   - NCC collective/peer engine work、Direct-DTE work、`NCCJoin`；
   - compute/recompute、各engine issue、dependency depth和`max(per-rank critical-path lower bound)`；
   - SPM high-water、DDR arena high-water、buffer/descriptor pressure。
3. 记录candidate expanded states、actual clones、finalization Tile→Instr lowerings、SPM/DDR pack calls、peak RSS和Release wall time。
   wall/RSS复用Q41的compile-transaction timer与process peak-RSS事实源：同一Release build、source/payload/options、`nproc`
   并发度和host身份下先做1次不计入的warm-up，再以3个独立process测量；冻结wall中位数、peak RSS最大值和每次search-work counters。
4. 对同一source、payload、launch、ABI和fresh build生成pre-Q49 optimized production baseline与`none`保守参照；两者只允许
   optimization policy不同。C0从当前pipeline dump post-SPMD structured IR与final Instr IR；selected complete-rank
   Tile/Dataflow IR必须在C1建立真实decision point后由同一driver直接dump。当前per-task pipeline在selection前已经破坏性lower
   Tile candidate，C0不得为补齐中间dump而保留/replay影子clone或建立sidecar。dump名和诊断不含任务号、case名或临时目录约定。
5. Fresh重放16的Q49 gate所定义的当前Llama primary，生成TP16 FP16完整package/no-card；只做本任务定向gate，不重跑无关catalog或历史板测。

Gate：

- fixed static loops的expanded work可从dump独立精确复算；conditional/dynamic control只报告bounds、symbolic或Unknown；
- serial/parallel compile得到相同work counts、winner digest和package；
- 当前output、guard、ABI与package contract不变；
- baseline记录至少包含本轮fresh RDMA/WDMA/GS/join/bytes、SPM high-water、compile work/RSS/wall；
- C1建立complete-rank decision point后，selected Tile/Dataflow dump能与同一次compile的post-SPMD和final Instr直接关联；
- 未满足以上条件，不得用旧日志或理论机会量签后续收益。

不算完成：只统计静态call site、只报全卡Instr总数、用aggregate/平均值隐藏逐rank与rank-maxima，或把未校准work加成伪cycle。

### C0 fresh evidence（2026-08-05）

同一Release build、同一PyTorch-exported FP16 Llama block source/payload、TP16 kernel launch与current ABI下，production和
`none`均从source生成完整16-rank package并fresh no-card。production三次独立正常并发compile的transaction wall为
`90269 / 93817 / 90493 ms`，中位数`90493 ms`；peak RSS为`2952700 / 2969012 / 2951464 KiB`，最大值
`2969012 KiB`。三次compiler work均严格相同：expanded states `37488`、finalization clones `48784`、finalization
Tile→Instr lowerings `49040`、SPM planning `60084`、DDR planning `46992`。

固定单CPU affinity的完整同源compile得到相同五项work、winner Instr与递归package bytes；其transaction wall为
`1352964 ms`、peak RSS为`814604 KiB`。正常并发三份package、单CPU package、manifest与rank-0 Instr digest分别逐字节一致；
manifest digest为`bead668201b76b7728813ed25a657f542ffda9093d03e2092c99b7451b16facb`，rank-0 Instr digest为
`445dd648b1d3f287e1ab9c1c8195778a0f36293cdfd85d4759f2870be3302d9f`。并发执行只改变吞吐和RSS，不改变search work或artifact。

final Instr同源counter如下；call/site与loop-expanded execution均保留在diagnostic中，表中列出最直接回答当前问题的exact work：

| all-rank exact work | production | `none` | production相对`none` |
| --- | ---: | ---: | ---: |
| Instr | 10324 | 15120 | -31.7% |
| GS calls | 3280 | 4064 | -19.3% |
| GS bytes | 885675520 | 904680960 | -2.1% |
| Direct-DTE send / receive / wait | 60 / 60 / 120 | 960 / 960 / 960 | DTE总work -91.7% |
| `NCCJoin` | 870 | 1792 | -51.5% |
| nonterminal `NCCJoin` | 854 | 1776 | -51.9% |
| DDR read / write bytes | 888076384 / 430446080 | 888076384 / 430446080 | 0% |
| SPM movement bytes | 2204197984 | 2223203424 | -0.9% |
| compiler-owned SPM / DDR buffers | 3824 / 832 | 4112 / 832 | SPM -7.0%，DDR 0% |
| summed SPM / DDR high-water bytes | 46850048 / 95428608 | 46850048 / 95428608 | 0% |

production的rank-0为`137 RDMA / 86 WDMA / 205 GS / 54 join`，rank-15为相同movement与`54 join`；各维度
rank-maxima为`658 Instr / 205 GS / 56 join / 24 DTE`，不是伪造的单一critical rank。该C0事实快照中的旧
dependency-depth analysis不能解释final Instr structured control，因而报告`Unknown(unsupported-control-flow)`，没有当作零
或伪造cycle；后续C4施工已对固定trip-count loop和常量条件建立current-IR structural analyzer，只有动态或无法证明的控制流
继续保留typed `Unknown`。

这组事实说明旧独立优化确实大量压低了Direct-DTE与约一半join，但没有减少任何DDR traffic/high-water，GS bytes也只下降
2.1%；因此后续收益不能再靠late局部cleanup，必须由C1开始把decision point移到complete-rank pre-Instr structured IR。
当前driver只能直接保存post-SPMD structured IR与final Instr；selected complete-rank Tile/Dataflow证据按上面的边界在C1建立，
没有为C0引入shadow clone、replay或sidecar。

## C1：Complete-Rank Decision Point

状态：已完成。production在旧per-scope lowering之前直接持有complete-rank structured clone，只由request shard 0
物化唯一conservative baseline；同一actual Tile clone在Tile→Instr前保存selected IR，再经现有late gates形成rank-count
1/16完整package并fresh no-card。旧per-scope实现与NoC独立扩展只保留testing consumer，统一在C6物理删除。
`tile.region` shaped data边界已收紧为DDR-only、top-level non-nested SPM residency domain；complete-rank graph、不同
traversal domain、rank-changing view、structured concat、显式two-region DDR cut及same-region selective spill均有fresh
source/IR gate，SPM root/alias escape和unsupported indexed payload fail closed。

目标：在任何不可逆 Tile→Instr、SPM/DDR placement或final completion之前，建立完整rank的structured decision scope。

实施：

1. 调整production orchestration，使scheduler先持有每个rank的完整post-SPMD structured clone；旧task/scope只能作为
   source IR中的结构，不再各自形成standalone function和lowered candidate。bring-up在同一complete-rank clone内直接生成
   conservative actual Tile candidate，再经baseline-only finalization seam只执行一次complete-rank Tile→Instr和现有late gates；
   这条seam只证明新decision point的等价package/no-card，不建立通用frontier、worker/order siblings或exact-failure repair。
   不得保留per-scope selection、standalone module、placement或join，也不得按旧scope一对一生成region。
2. 从current SSA、`TilingInterface`、`IndexRelation`、effects、control flow和numeric contract派生transient connection
   frontier；对每条connection计算tile propagation、numeric、storage、completion和rank-coupling legality。
3. 冻结有限producer集合和严格拓扑progress measure；每次propagation只能消费尚未覆盖的domain/edge，避免single-tile
   fusion worklist重入或同一view链反复展开。
4. 含collective的actual clones不能rank-local误剪；coordinator可从current typed IR即时派生transient hash预筛，但每个tuple
   必须逐项fresh重证，hash不进入state、IR、schema或正确性判断；不用task/layout ordinal拼card-wide tuple。
5. 在winner仍选conservative baseline的bring-up阶段，保持selected IR和package等价，先证明新的decision point没有语义漂移。
6. C1建立`tile.region`的SPM-residency合同和multi-region legality，但conservative baseline只需物化由current physical edges
   决定的确定性合法partition，不能为了checkpoint硬造一个region。旧scope、component、loop、tile shape或collective都不自动
   产生sibling region；selected cut的全部crossing data必须显式store/completion/load，resident SPM root/alias不能跨界。
   same-region可以包含多个SCF traversals，selective spill只结束目标root；opaque clobber无法typed解释时继续fail closed。
7. 当前逐scope `clone→lower→import` helper不是可交付中间态；C1完成前production consumer已切到直接拥有
   complete-rank current IR的新路径，旧per-scope decision/materialization path的production consumer为零。新路径不得调用旧
   full-buffer handoff promotion把DDR边界改写成跨sibling-region SPM SSA；旧实现代码在C6统一物理删除。
8. 同一checkpoint收紧`tile.region` ODS/verifier和lifetime/planner输入：region data operand/result只允许DDR，SPM
   value作为boundary operand/result及SPM root/alias escape一律非法；region内部SPM roots合法并由SSA/effect验证。
   scalar/event/control按typed合同通过。rank-entry允许一个或多个top-level non-nested
   regions；每个region exit只完成仍访问其SPM roots的pending work，必要时在`wafer.tile.yield`内侧插exact-participant join，
   不drain无关DTE/NCC/generic work；entry final drain才闭合observable completion。

Gate：

- chain、diamond、fanin/fanout、shared input、multi-root、structured control flow、unknown/observable effect的component与
  edge-legality正负例通过；
- 旧source scope、shape变化、layout不兼容、collective和单个intermediate store/reload不会自动切region；selected
  merge/split必须由完整residency/materialization action产生，external input仍按storage semantics显式load；
- final `tile.region` inputs/results为variadic，chain、diamond、fanin/fanout和multi-root不受人为边数上限；
- C1 conservative baseline的每个current static rank entry含一个或多个top-level non-nested regions，所有region data
  operand/result的SPM数量为零；当前宽松ODS/verifier及跨region SPM escape路径在C1内收紧；
- C1 source gate至少包含一个跨两个旧scope但可merge为single residency region的multi-op entry、一个显式two-region DDR cut，
  以及long-lived root `A`保持SPM resident而高压力root `B`在同region selective spill/reload的case。`B`的DDR edge不结束`A` lifetime；
- source-observable ordering只阻断对应schedule/completion action；unsupported numeric reorder、不可表示control flow及其它
  未知legality也只阻断其对应维度；analysis/component停止本身不生成region、DDR或join，只有selected完整partition/
  materialization action可以物化region cut；
- production candidate construction不再调用单task独立Tile→Instr/SPM/DDR再import的路径；旧死代码统一在C6删除；
- rank-count 1/16 baseline经完整late gates和fresh no-card。

不算完成：只构建C++ component graph而未改变production边界，或暂时保留两条decision pipeline等待以后切换。

## C2：Coordinated Global Frontier 与 Joint Traversal/Tiling Search

目标：先建立all-rank coordinator、唯一deterministic work budget和coordinated global frontier，再从consumer tile反推
producer slice/iteration，联合选择region partition、traversal fusion/materialization、tile shape/loop order、layout/version、
resident/spill/recompute、movement和communication；
本checkpoint不产生任何rank/component/layer winner。

实施：

1. 从output/observable root的finite tile domain出发，用DPS、indexing maps和`TilingInterface`计算operand tiles，沿
   `IndexRelation`穿过view、slice、broadcast、permutation和reshape。
2. 每条connection原子枚举region、schedule与storage维度及其完整tile/physical参数：
   - **same-region coupled traversal**：选择consumer/root tile并经`TilingInterface`、DPS与index relation反推producer demand；
     producer tile不是独立变量，只有派生relation和materialization均exact时才耦合；
   - **same-region separated traversals**：同一residency domain中物化两套独立loop nests，各自选择tile/loop/layout并重算root lifetime；
   - **cross-region cut**：两侧独立选择tile/loop/layout，切口全部crossing values显式store/completion/load，SPM root/alias不跨界；
   - 每种same-region schedule再选择resident SSA、local movement、selective spill/streaming或recompute。separated traversal、
     per-value spill和region cut互不自动推出。
   枚举时先形成只含current-IR引用、typed决定、lower bound与版本化通用estimate的transient structural proposal；它们在同一次
   coordinator invocation内进入DP/Pareto beam、equivalence、dominance与coverage剪枝，但不能跨pass、发布或进入late exact gate。
   只有统一actual-clone budget准入的frontier代表才在complete-rank Tile clone中同时物化region partition、loop/tile、lifetime和
   physical action，随后销毁proposal。不能先选局部tile winner再补storage，也不能把proposal当作跨stage schedule plan。
3. 对pure producer生成direct SSA tile、shared version、consumer-local recompute和loop-invariant materialize alternatives；
   fanout/fanin在同一live frontier联合处理。shared input的load/layout placement随consumer loop construction决定，
   不在Instr层对effectful load做事后LICM。
4. 对每个可耦合connection必须显式比较single-region fused-small-tile与multi-region separated-large-tile；后者的收益包括两侧独立retile、tile
   utilization提升、loop-expanded work下降，以及缩短并发SPM-root lifetime/live bound，即使原方案可以pack也不能漏掉。
   对working set中单个高压力root另保留same-region selective-spill sibling，无关root继续resident；另保留same-region
   separated traversal以证明schedule split不等于region cut。不按图连通性、旧scope或单个DDR op机械partition，也不把所有
   traversal误绑成统一mega-tile。
5. layout PBQP只作为同一frontier/analysis epoch的proposal reducer。它只能删除已证legality-infeasible的项，或者在
   future live interface、physical versions、SPM lifetime/capacity disposition，以及06定义的完整selection vector（DDR
   all-rank aggregate/max-rank issue、GS/local max-rank/aggregate、completion/wait/critical path、NoC/link/endpoint、
   SPM movement、compute/recompute、tile utilization、Instr、descriptor/resource、all-rank coupling）和所有`Unknown`
   disposition连同current-IR derivation上完全等价的proposal；这只作pre-clone coverage merge，不把same-reason Unknown变成final
   cost相等。不能用component-local conversion cost选winner。proposal的准入、物化与销毁按第2条执行，
   不形成artifact product或`physicalLayoutProposalOrdinal`。
6. 对未处理IR可观察的live-frontier facts完全相同的chain/tree structural state做frontier DP；一般DAG使用deterministic
   Pareto beam。两者都发生在actual-clone admission之前，transient hash只可预筛。direct dominance要求DDR、GS、completion、compute/recompute、tile utilization、NoC、
   Instr、SPM movement、descriptor/resource、critical-path和all-rank coupling全部selection-sensitive维度不差且至少一项更好；任一`Unknown`
   disposition不同即不可比较，不能仅按某一movement、region数或插入顺序剪枝。
7. coordinator在任何generation前建立唯一tuple-level deterministic work ledger，为conservative baseline的mandatory C3 gates
   预留credits，并保留有限repair reserve。每个非baseline schedule action只有在可按deterministic upper bound预留其C3
   Tile→Instr、worker/order/completion、SPM、DDR、transport和ABI evaluation credits后才准入；generation不能消耗这些预留。
   component、rank、layout、worker或collective不拥有可相乘的独立cap。允许从actual clone即时派生不可发布的rank/component
   factor summary，携带region boundary与collective/peer interface、additive/max-reduction cost contribution和`Unknown` disposition，
   只供同一coordinator的factorized DP/beam使用；不得形成local winner/commit或`N^R` tuple。具体cap由C0
   work counters标定，是production policy，不是IR语义或shape特化常量。
8. collective/peer及其shared parameters从生成开始就以all-rank coordinated state扩展；rank factor summary可以逐rank计算和
   Pareto压缩，但不能独立选winner或丢失shared interface。任何能影响message matching的组合都由coordinator逐项fresh重证，
   DDR aggregate、GS/completion rank-maximum和NoC contribution按明确reduction operator组合。
9. 第一条production vertical使用通用indexing semantics把transpose/view relation吸收到oriented contraction actual clone，
   再对effect-proven read-only、原始逻辑shape的weight执行exact composed Tensor DDR mapped transfer到Cx；不得把它描述成
   identity DMA，也不得假设任意transpose可直接RDMA。
10. 在相同通用vertical中证明loop-invariant LHS load/layout位于N tile loop外，且不依赖QKV、gate/up、weight名或4096/11008。

Gate：

- chain、diamond、shared-input contraction、fanout/fanin、tail、multi-root和不同scalar body的`linalg.generic` held-out通过；
- full-weight runtime transpose的RDMA→GS→WDMA链在适用oriented-GEMM actual candidate/finalization shortlist中消失；
- LHS RDMA/GS static-trip loop-expanded work不随N-loop trip count线性重复；
- 同一source在structural frontier保留single-region fused-small-tile、multi-region separated-large-tile、same-region separated
  traversal和selective-spill代表，并由统一预算实际物化至少一组对立actual candidates；resident direct edge没有中间WDMA/RDMA，cross-region
  edge有完整store/completion/load，internal spill只结束对应root；
- C2不提交winner、不运行Tile→Instr/SPM/DDR executable-admission gates，也不单独声称frontier已在capacity意义闭合；C3只消费C2
  actual-clone admission产生的coordinated Tile clones并把typed exact结果反馈同一owner；失败后从仍在同次invocation内的未物化
  structural frontier确定性补位，不重新枚举domain或建立第二个repair selector；
- 全局budget耗尽、serial/parallel generation和不同host worker数产生相同coordinated frontier digest；baseline及已准入finalization
  candidate的C3 reservations仍完整，不存在per-rank finalization cap或rank/component artifact Cartesian product；
- heuristic冻结后才运行held-out graph，失败不得添加shape/op-name特例。

不算完成：只用late transfer elimination删一两个GS、先独立选traversal/tile/layout再笛卡尔积，或只对手写
`wafer.tile.*` fixture产生预期IR。

## C3：Executable Finalization、Exact Admission 与 Fresh Completion

目标：C3作为C2 coordinator调用的唯一executable-finalization service，只对已准入的coordinated all-rank Tile variants执行昂贵
lowering和exact gates；它不拥有frontier、budget或winner。在最终worker/order上fresh重建completion，从complete current IR
派生SPM/DDR fixed allocation problems并all-and-only覆盖roots/domains，再原子执行transport/ABI exact evaluation。
本checkpoint产出admitted all-rank executable frontier，不提交production winner。

实施：

1. candidate evaluation只接受C2 structural frontier中的complete all-rank Tile recipe；frontier在一次invocation内最多64项，
   不为每项分配独立finalization allowance。mandatory baseline canonical seed外置于A/B轮转，但其actual materialization
   attempt和successful exact action仍计全局16/8。其余action由coordinator按stable A-first在A（下一个new-Tile canonical seed）
   和B（推进已有exact-seeded cursor）之间轮转，live cursor最多8个。每次已开始action原子生成匹配的all-rank siblings，执行
   Tile→Instr、worker/order、completion和exact gates；成功或materialization/exact failure都消耗一次attempt并换lane，setup前
   failure不计attempt。任何时刻只保留1个action clone，不建立per-rank、per-Tile、per-provider frontier/quota或`W^R`组合。
   qualification/characterization可请求明确typed family/point，但不绕过同一invocation-wide budget与exact gates。
   pre-Instr search不保存worker/NCC shadow schedule。
2. completion的canonical input不含旧`wafer.instr.ncc_join`：先删除clone中全部旧join，再从current worker order、
   typed issue/effect、event token、alias/range、reuse和observer fresh rebuild。每个join必须有hazard、protocol或
   observable-barrier witness；DTE wait与group barrier不能由join替代。region结构exit不是全局drain；只对仍访问region-owned
   roots的exact participant mask在`wafer.tile.yield`内侧插join，无关pending frontier继续传播。
3. 在每个finalized rank-entry Instr variant中，先完成function-boundary bufferization和fresh completion，再按current SPM
   roots、control-flow coexistence、lifetime/conflict形成all-and-only fixed SPM allocation problems，执行3 MiB
   MiniMalloc placement与physical alias/range验证。已证明不重叠的regions/roots可复用地址，可能重叠的regions必须联合满足容量；
   problem/query数量只是work counter，不与entry或region数量绑定。allocator只返回三态、validated placement和high-water/headroom
   diagnostic；不插join、不修改order、不做capacity bisection/quality probe、不生成repair。
4. 只有all-and-only ranks均已形成finalized rank-entry Instr variant后，才对这个complete all-rank variant原子构造并验证
   current explicit DDR arenas/placement domains，随后执行post-memory Direct-DTE/message-resource、target ABI与final recost。
   per-rank default arena是current IR可能形成的domain，不是固定solver次数；此前逐rank structural/lower-bound检查只能cheap reject，
   不能产生rank-local commit。
5. packing或其它exact gate失败时，C3只把typed failure交回仍然存活的C2 coordinator；由coordinator在剩余global repair
   credits内从仍持有的无offset actual Tile parent生成有限repartition/merge/split、retile、traversal separation/fusion、selective spill、share→recompute、
   layout/route、loop order或buffering sibling，重新进入同一global frontier；
   C3不生成alternative，也不得原地修复failed clone。C2先闭合generation接口；C3接入后production由同一个C2 owner保持
   frontier与budget存活直到executable finalization停止条件满足，只有C2+C3联合gate才证明搜索闭环。新neighbor从unplaced parent
   物化并fresh派生自己的allocation problems；已经写入offset的failed Instr variant绝不原地改写或复用placement。DDR-clean
   region split的一个bounded representative优先按static physical SPM bytes平衡cut两侧，byte事实不完整或溢出时才回退到Tile
   dataflow op count；该prior不代替exact completion、packing和cost。
6. final cost全部从通过exact gates的current Instr/DDR IR fresh重算：
   - DDR：all-rank aggregate bytes/executions为主，max-rank issue pressure为审计；
   - GS/local movement：max-rank tile-local bytes/executions为主，aggregate为审计；
   - completion：max-rank steady/nonterminal/total participant waits与critical-path placement为主，join op count次级；
   - NoC/link/endpoint、SPM movement、compute/recompute、tile utilization、Instr、descriptor/resource pressure分别保留；
     SPM high-water只作capacity/headroom。
   删除无量纲`ExternalMovementFirst`捷径和历史SPM0/RAM_ACC `128 GB/s` flat-duration；SPM1 exact movement改用
   `256 B * 1 GHz = 256 GB/s/tile`单bank nominal prior形成max-rank resource envelope，并携带versioned assumption；
   它不是lower bound或proof，`Unknown`与known仍不可直接支配。
7. all-rank collective/peer alternatives以typed参数共同物化，tuple从current IR逐项fresh重证。C3只消费C2的
   coordinated frontier，不再拥有第二个rank frontier、tuple builder或generation budget；任何rank/region/component不得
   先挑局部winner再做artifact Cartesian product。
8. 每个admitted executable重验residency合同：一个或多个non-nested regions，SPM root/alias不跨region，cross-region data有
   显式store/completion/load，internal spill只结束目标root；region boundary没有全局自动join，root-release exact join只位于
   `wafer.tile.yield`内侧，entry final drain只有fresh obligations。跨region的scalar block argument/result沿matching input/yield回溯
   外围SSA，不能把原本静态的loop bound误判为dynamic control。

预算合同（由C2唯一owner管理）：

- 只有一个coordinated-state/work ledger，不再设per-rank、per-component、per-layout或per-worker cap；
- structural frontier invocation-wide最多64项；finalization all-rank Tile candidates与worker/order/completion扩展
  全invocation共享总计最多16次actual materialization attempt和8个successful exact action，不形成`64 * 16`、
  `per-rank cap ^ rank-count`或任何per-Tile/provider-local乘积；
- reserved baseline canonical seed外置于A/B轮转且完整exact gate只执行一次，但其attempt/success计全局16/8；
  已准入action的reservation不被generation挪用，未使用credits按stable规则释放；
- A/B按stable A-first轮转；setup前failure不计attempt，已开始action的成功或materialization/exact failure都消耗
  attempt并换lane；live cursor最多8个，peak action clone为1；
- intermediate transitions只运行relation/type/effect/footprint lower bound和局部verifier；
- full Tile→Instr、SPM和DDR gate只对finalization candidates执行；
- budget按deterministic work units，不按wall-clock timeout；耗尽时保留baseline，并停止没有reservation的新neighbor。

Gate：

- SPM不足通过同一frontier产生retile/traversal separation/selective-spill/recompute sibling，不触发all-or-nothing full-buffer回退；
- fanout可选择共享version、有界K个consumer-compatible versions、分支local conversion或局部spill；K是search policy，不是IR语义上限；
- 三个consumer且存在多个互不兼容physical demands的held-out graph通过，不把初始primary+secondary覆盖写成通用上限；
- 每个survivor都是actual clone，factor/search对象销毁后verifier和lowering结论不变；
- packing failure不留下partial offsets，不使allocator修改partition/tile/layout/order；每个final executable candidate的SPM/DDR
  roots/domains由fresh fixed problems all-and-only覆盖，problem/query数量进入统一budget但不是IR语义；
- single-region fused-small-tile、multi-region separated-large-tile、same-region separated traversal和selective-spill均走相同
  exact gate；final verifier拒绝region boundary上的全部SPM data和隐式movement/join；
- serial/parallel exact frontier确定一致；
- final cost不再包含SPM0/RAM_ACC 1024-bit接口外推的per-tile SPM flat duration；SPM1只用单个2048-bit bank在1 GHz下的
  `256 GB/s/tile`作为保守point prior，和外部/compute service取max而非相加，避免同一DMA双计；
- 每个fresh `NCCJoin`都有witness，same-worker安全ordered chain不join，aggregate join数不掩盖max-rank steady waits；
- 不出现“region/task数 × recipe数”的full lowering笛卡尔积，expanded states、finalization lowering次数和RSS保持显式bounded；

不算完成：C2/C3各自挑局部winner、每个loop/component独立lower/pack/join、用high-water反复probe排序，或用外部solver结果作为
production语义输入。

## C4：Hardware-Cost Selection、Scalability Proof 与原子提交

目标：不新增候选、不重做packing/DDR exact evaluation，只从C3 admitted all-rank executable frontier按当前校准的hardware cost model
选择唯一winner并原子提交；同时证明C2/C3已内建的全局预算和all-rank coordination在model scale下可界且确定。

实施：

1. C4只消费C3已完exact gate的actual all-rank variants及其fresh final-IR cost；不从source或Tile parent重建候选，
   不再调用Tile→Instr、MiniMalloc或DDR planner。commit前可从accepted offsets/current IR重跑无搜索的range、alias、message、ABI和
   all-and-only-rank validator，不重新求解或移动offset/join。
2. Pareto保留与终态winner选择分开：direct dominance只有在DDR、GS、completion、compute/recompute、tile utilization、
   NoC、Instr、SPM movement、descriptor/resource、critical-path与all-rank coupling全部为`Known`、可比较且不差，并至少一项严格更好时
   成立；SPM high-water只作hard capacity/headroom。任一`Unknown`都不可比较，即使reason/disposition相同也不视作相等。终态只允许
   当前校准的hardware cost model对hard-legal states排序。`ProvenBenefit`只在candidate conservative upper bound严格低于baseline
   conservative lower bound时成立。若两边Known的零overlap-window证明同一explicit sequential schedule，且逐rank compute engine、
   DDR、SPM、NoC与intrinsic NCC drain primitive均no-regression，则可在同一control resource内用point priors核算instruction、DTE
   waited event和NCC participant wait的changed work；清除20% margin后只签发`EstimatedBenefit`。duration/audit aggregate与max summary
   只作guard，不能创造benefit；stable semantic order只能在hardware-cost comparison tuple完全相等时作最后确定性tie-break，
   不得从不可比Pareto states中任意选winner。
3. parallel evaluation只并行独立actual variants；frontier insertion、hardware-cost comparison、最后tie-break和package不由完成
   时序决定。finalization Tile→Instr、MiniMalloc和DDR planning次数继续由C2唯一work budget/counter审计。
4. 用C0的37488 expanded states、48784 finalization clones、49040 Tile→Instr、60084 SPM plans、46992 DDR plans及
   wall/RSS作为迁移基线；证明昂贵gate次数随finalization all-rank candidates和其明确worker/order扩展增长，
   而不是随region/task/rank-local shortlist相乘。
5. winner commit只移交C3 actual clone中已验证的all-and-only rank programs、offsets、bindings和current-IR-derived
   records；C2 proposal、search object、failed clone、rejected offset和单独rank result均不得成为commit输入。

Gate：

- global cap耗尽、serial/parallel执行和不同host worker数产生相同admitted-executable frontier、winner/package；
- expanded state、actual clone、Tile→Instr、MiniMalloc、DDR plan和peak live clone均有硬上界及fresh counter；
- 不出现`traversals × tiles × layouts × spill modes × workers` artifact Cartesian product；
- all-rank collective tuple不会因任一rank的局部DDR/GS/join较差被提前误剪；
- `Unknown` path至少保留一个typed conservative representative直到能exact化或结构化拒绝；
- unique winner只能来自C3 admitted all-rank executable frontier，任何rank/component局部best均无提交入口；
- C4的MiniMalloc和DDR planner调用计数为零；commit前validator不产生新offset、join或candidate；
- production build/runtime/link surface没有ILP/CP-SAT依赖、离线小图oracle或调用；
- late failure销毁完整clone，不恢复旧保守join pipeline或per-component winner。

不算完成：把每层beam相乘后只限制最终tuple、在C4补做all-rank正确性或重跑packing、靠wall-clock timeout
决定winner、用semantic order代替hardware cost model，或用rank平均值掩盖max-rank瓶颈。

## C5：Interface-Driven Reduction 与 Implementation Alternatives

目标：让collective、native/partial reduction与任何会改变计算结构的implementation alternative进入同一个通用whole-rank
structural frontier。普通structured op只使用现有typed interfaces；Attention只是验证该扩展边界的一个真实算法族，不成为
coordinator、SPM、lifetime、cost或frontend中的特殊路径。

实施分成四个顺序边界：

1. **C5a collective residency + native/partial**：logical collective保持typed rank/group/payload semantics；Direct/Ring/Tree仅作为
   executable conversion的typed参数逐点生成actual Instr sibling。compute tile可跨collective连接，但真实NCC/DTE wait、participant
   completion和all-rank resource matching不被融合删除，collective也不自动产生DDR store/reload。普通reduction的domain、tile
   materialization与partial能力分别由`TilingInterface`、indexing/iterator semantics和`PartialReductionOpInterface`提供；
   coordinator不识别具体op种类。
2. **C5b implementation-alternative extension**：production pipeline向通用coordinator注入一个typed extension set。每个provider只可
   从current SSA证明适用性、暴露query-local输入输出与iteration/reduction domain、生成合法算法参数并materialize等价structured
   IR；它不能提供自定义winner、修改通用frontier/comparator或调用专用late gate。普通实现不经过provider，直接使用
   `TilingInterface`。provider产生的domain point与普通tile/layout/connection proposal共同进入pre-materialization DP/beam；只有
   统一actual-clone budget准入后才调用materializer，随后由actual current IR重证全部事实。provider可给exact point附带typed optional
   peak-live-byte/compute-work estimate；common coordinator先保留`parallelism=serial|partitioned × residency`粗coverage，再按资源
   headroom稳定排序。estimate缺失不构成失败，coverage/order都不解释opaque identity或绕过common exact gates。
3. **C5c Attention implementation provider**：query-local analysis只从current structured SSA的type、indexing map、iterator、scalar
   body、SSA use-def、view relation与function boundary推导可等价实现的算法domain。PyTorch/Hugging Face当前导出的计算、常量、
   operands、control flow和数值原样作为输入；frontend、fixture与provider均不得注入、规范化或重建模型语义。RoPE cos/sin在
   model侧按原实现precompute并作为普通值进入图，mask的finite值或`-inf`原样lower；compiler不注入或重算二者。无法证明等价时不
   生成alternative并保留baseline，不按HF版本、名字、固定rank、shape或export序列猜测。

   普通attention语义可贡献FlashAttention-2 implementation；只有current SSA同时证明past/new K/V的functional update、attention
   读取updated K/V以及updated K/V作为调用边界结果继续线程化时，才可贡献FlashDecoding。`S_q=1`或某个KV shape都不能替代该
   数据流证明。FlashAttention-2以Q/output工作单元顺序遍历KV block并维护普通SSA online state；FlashDecoding建立真实split-KV
   并行维、split-indexed partial state和独立final merge。KV block与split只是provider暴露的合法参数域，target先验可排序但不能
   硬选；全部参数继续由通用structural facts、DP/beam、actual-clone admission、exact gates和final recost决定。两个materializer
   均只产生普通structured SSA，不引入专用Instr、ABI、SPM或cost路径。

   Pipeline position:
   - Upstream artifact / IR: post-SPMD complete-rank structured tensor IR及其current SSA语义；算法domain尚未物化。
   - Current stage responsibility: Attention provider只证明、描述并物化一个query-local implementation alternative；通用
     coordinator负责tile/connection/region组合、pre-clone剪枝、actual-clone准入和唯一winner选择。
   - Output artifact / IR: actual-clone准入后生成的等价structured SSA；算法参数随materialization销毁，所选执行结构只由IR表达。
   - Downstream consumer: 与所有其它actual candidates相同的TensorProgram→Tile、Tile→Instr、completion、memory、target与
     executable recost流程。
   - User-level driver / named pipeline: production named pipeline自动注册provider；standalone pass只用于typed IR测试。
   - Explicit non-goals: 不修改或补全PyTorch model，不识别模型/函数/operand名字，不拥有candidate选择，不引入长期sidecar；
     runtime-owned cache resource、page allocator和跨request服务仍不是本stage职责。
   - Completion gate: provider正负例和numeric oracle通过；真实official HF source中prefill与functional decode分别生成并由通用
     cost/legality实际选择合法sibling，decode的current input/output状态关系在final IR中保持。
4. **C5d diverse workload matrix**：先用普通`TilingInterface`异构图证明新增op无需修改coordinator，再验证official HF prefill、
   functional decode和较大Llama-2-7B风格完整block。仓库只保留必要的通用输入/输出layout adapter，不复制模型数学、不打compiler
   marker、不按shape分支。所有case按16的FP16/BF16、rank-count 1/16合同完成source→package→oracle→fresh no-card。

Gate：

- 新普通`TilingInterface` op无需修改coordinator；新增implementation provider无需修改common search、SPM、lifetime或cost；
- common coordinator、resource analyses和final selector中不存在workload/算法名字分支；
- native、partial与Attention alternatives分别有typed positive/negative和numeric oracle，unsupported alternative fail closed；
- collective本身不机械切region；resident、retile/cross-region和streaming siblings由逐connection通用搜索比较；
- structural proposal数可以远大于actual clone数，clone数受统一预算约束，exact失败能从未物化frontier确定性补位；
- 真实HF prefill和functional decode分别至少有一个sibling由production cost/legality实际选中，而不是测试强制指定；
- 非decode图、仅`S_q=1`或仅存在KV形状不得通过decode provider；
- 异构非Attention图、较大Llama block及FP16/BF16 rank-count 1/16 package/model/no-card矩阵通过；
- held-out失败只能暴露interface/capability/resource问题，不能追加shape、名字或模型matcher。

不算完成：普通op绕过`TilingInterface`进入专用contributor、算法类型进入通用traversal enum、每个domain point都actual-clone、
为通过某个case硬编码tile/split/resource proxy、修改frontend model、或只用synthetic case和公式宣称implementation已支持。

## C6：Optimizing Policy Cutover 与旧路径删除

目标：C1已保证默认driver只有一条complete-rank decision/materialization路径；本checkpoint将`production`从
conservative-only切到已通过C2–C5全部gate的winner policy，并删除C1起已零consumer的旧代码、开关和测试入口。

必须迁移或删除：

- standalone task function clone、per-task Tile→Instr→SPM/DDR evaluation和task candidate commit；
- task `func.return` final join；
- `taskAlternativeOrdinal`、`spmWorkingSetMultiplicity`和per-task layout ordinal；
- spill/spill-ready/resident/resident-ready等artifact-kind Cartesian machinery；
- all-or-nothing full-buffer residency decision owner；
- late NoC-resident tuple decision owner；typed peer/collective mechanics迁入06/13 action library；
- 历史NoC profitability中的per-tile SPM flat-rate duration及其配置/测试期望；SPM bytes保留为exact work，
  不用未经校准的单一rate计时；
- 被统一owner吸收的scope/layout/residency/NoC独立public optimization flags；
- 任何隐藏compatibility pipeline、手工pass拼接或名字/shape matcher。

实现审计至少点名`CandidateCommit.cpp`、`FullBufferHandoff.cpp`、`RankCandidateFrontier.h`的artifact enum、
`ScheduleTensorProgramInternal.h`的旧ordinal/multiplicity/layout fields、`NoCResidentDataflow.cpp`的decision部分和
`OptimizationConfig.h`旧独立flags。Q40的DTE-overlap/fixed-slot consumer迁到executable Instr sibling；Q41的timing、bounded executor、
RSS和diagnostics继续复用，配置/CLI/help/tests同步收口为public `production`/`none`两种policy，不保留scope/residency/NoC等独立开关。
qualification选择仍可用compiler-private typed seam，但不是public optimization axis。

保留但降为单一职责：

- redundant transfer、GS coalescing、view folding和local LICM只作policy-free generic canonicalization；
- worker placement、fixed-slot、PBQP和NoC lowering保留为typed mechanics/action library；
- SPM/DDR allocator继续作为hard exact gate；SPM allocator按09从final current IR的rank-local 3 MiB SPM window、roots、
  lifetime/control-flow coexistence和conflict派生fixed allocation problems，all-and-only覆盖后用MiniMalloc放置；
  high-water只报headroom，bank phase不改变hard feasible set，
  allocator不拥有candidate partition/dataflow选择；
- `tile.region`继续表达SPM residency domain和SSA containment；current static rank entry允许一个或多个non-nested regions，
  partition由06联合选择。task、per-loop或单个spill不能机械建region，cross-region data必须显式materialize，SPM root/alias不跨界。

Gate：

- `wafer-compile`是唯一production named pipeline，production/none A/B都走同一owner和late gates；
- 旧symbols、config fields、diagnostics和tests的consumer静态搜索为空；仅archive可保留历史文字；
- current typed capability、conservative baseline和negative failure均由新路径覆盖；
- Release model-scale compile按C0同一Q41测量协议复跑；三次search-work counters均满足显式上界，wall与peak RSS按相同host、
  source、并发和build条件报告并处于可解释的工程合理范围。wall/RSS不设任意固定百分比或单case 60秒硬门槛；明显回退必须归因，
  clone/finalization/packing次数必须受确定性上界约束，不能以减少重复次数、改变并发/workload、放宽容差或删除exact gate换取通过；
- compiler host/unit/lit/integration按`nproc`最大安全并发通过，unsupported/skipped单独核对；
- Llama与attention矩阵生成完整package并通过fresh no-card，达到`board-ready`。
- 板端matched baseline/winner均由cutover后同一fresh build和新pipeline重新生成；不得复用C0旧package。

不算完成：只关闭feature flag、留下死代码/旧test seam，或只让新路径在`wafer-opt`手工pass中工作。

## Workload 收益与验证门禁

所有exact workload、收益阈值、counter口径和board矩阵只引用`tasks/16-verification-contract.md`的Q49 gate；本计划不复制第二份
数字或shape事实源。比较基线是C0 fresh pre-Q49 optimized production，不是`none`；typed capability若证明某阈值不可达，先修改06/16并
说明证据，不能在implementation plan中静默放宽或追加case特化。

## 验证层次

### Unit / property

- edge-legality vector、component traversal、termination和complete coverage；
- IndexRelation pull/push、tail union、fanout/shared-input relation；
- current-IR frontier-fact equivalence、beam dominance、baseline retention和budget exhaustion；structural frontier不超过64，
  baseline外置但计全局16/8，A/B stable A-first且已开始action无论成败均消耗attempt并换lane，setup前failure不计attempt，
  live cursor不超过8且peak action clone为1；
- actual-clone atomicity、PBQP proposal销毁、packing sibling和无partial offset；
- completion hazard witness、erase+fresh rebuild、fixed-problem all-root coverage与physical-alias rejection；
- native/partial/online numeric和unsupported cases。

### IR / integration

- oriented GEMM + effect-proven read-only weight的exact composed mapped transfer，无runtime full transpose；
- LHS load/layout不随N-loop重复；
- fused producer/consumer可进入single-region耦合traversal且resident edge无WDMA/RDMA；multi-region separated-large-tile、
  same-region separated traversal和selective-spill反例均有actual IR与exact gate；
- current static rank entry覆盖一个或多个top-level non-nested regions；region boundary没有SPM memref/root/alias，两个旧scope、
  两个loop、layout变化、collective和单个spill都不会自动制造sibling region；selected split有完整DDR store/completion/load；
- collective有正确wait/completion但无自动DDR；
- same-worker ordered chain不join，cross-worker/range reuse/observer必须join；
- chain、diamond、fanout/fanin、multi-root、reduction、control flow和effect barrier；
- FP16/BF16、整tile和tail；两个不同scalar body的generic op证明无case matcher。

### Source / package / model

- rank-count 1和TP16当前Llama full block；
- read-only cached-attention与long prefill主/held-out矩阵；
- CPU/PyTorch oracle、完整output/guard、SystemC/CModel、typed ABI、package和fresh no-card；
- 按rank分别报告static site、loop-expanded/bounded work、Q9 runtime-measured count和aggregate shared-resource demand。

### Board

无卡阶段必须准备完整case、oracle、runner、package并fresh no-card，状态最多为`board-ready`。真实板端按仓库规则单进程、
逐case、bounded timeout执行16的Q49 board矩阵。设备异常后停止，不自动retry/reset/power；长prefill由host/model承担泛化gate，
不把完整prefill板测设为唯一外部阻塞。

## 提交与收尾

每个checkpoint单独形成可评审提交，且同步：

- `tasks/progress.md`只更新当前row状态；
- 06及直接受影响的07–13、16、18 pipeline contract；
- 新的稳定bug pattern或workflow才写`memory/`，不写临时case状态；
- fresh验证命令、实际执行/unsupported清单和未完成风险进入提交说明或task evidence，不复制成第二份设计。

C0–C6、全部host/package/no-card gate和旧路径删除完成后只达到`board-ready`；真实板端matched correctness/performance通过后，
才可将任务标为`done`。
