# NoC-Resident Tile Dataflow 实施计划

状态：Q39为`done`。共同compiler owner已经从
correspondence一致的complete-rank actual
tuples闭合typed input/parameter boundary的owner-only DDR load与显式peer fan-out/forward、可证明
intermediate spill/reload cut、replicated required output的round-2 publication，以及tree/ring
partial-reduction周围的NoC-resident cut。result-driven、operand-driven和
`PartialReductionOpInterface`路径已经进入同一bounded candidate owner；四个IR-derived materializer覆盖
input/parameter、intermediate、partial/reduction和output五类role，并在同一个discardable clone中按依赖
顺序累计，不维护role enum或shadow plan。tree验证contribution、combiner和publisher；ring从final message
tuple验证每个slice的reduce-scatter/all-gather传播、origin multiplicity和完整覆盖。无法从current IR证明的
range、control或message occurrence仍原子拒绝。whole-program message wait graph、Q38 fixed-slot/typed worker
realization、large contraction、compound workload及NoC×fixed-slot×nonzero-worker同候选的非板端纵向均已
闭合。当前重新打开的host边界是complete-rank NoC跨资源profitability：合法candidate不能再因严格减少
DDR bytes自动成为normal winner。该门禁闭合后，configured board matched performance promotion与同源
winner fresh correctness仍是外部门禁。

本计划的目标不是实现一个GEMM专用融合，也不是把collective拆小后重新排序。目标是让compiler能从当前
structured IR和完整rank domain中，通用地选择：

- 哪个logical rank从DDR取得一个输入、权重或其它boundary tile；
- 哪些rank在SPM中生产、消费、转发或归约这个tile；
- tile在producer、NoC、consumer之间何时保持resident，何时必须materialize movement；
- compute tile、transport segment和software-pipeline work quantum如何解耦；
- 最终哪些rank必须按既有program boundary完成DDR writeback。

GEMM、K-sharded GEMM、GEMM+Softmax/LayerNorm和attention只作为不同dataflow形态的验证case。候选发现、
IR协议、compiler owner和verifier不得按operator名、shape、rank数、模型或fixture匹配。

## Pipeline Contract

### A. Complete-tuple resident composition（当前实现与host资格边界）

```text
Pipeline position:
- Upstream artifact / IR:
  rank-local scheduling/finalization产生并导入bundle-owner MLIRContext的actual rank frontiers；每个slot携带
  semantic stable ordinal、physical artifact kind、reserved-baseline、buffering和worker-placement metadata，
  module可能因metadata prefilter而为空。frontend verifier同时提供完整logical-rank domain及distributed
  input/parameter的typed global/local ProgramRankSlice，ExecutionConfig提供rank count与TargetProfileId。
- Current stage responsibility:
  先调用与whole-variant coordinator相同的attempt-plan逻辑，按稳定顺序恢复correspondence-valid的完整
  actual tuples；reserved baseline固定为第一个seed，总seed上限为8。canonical generation parent必须是
  materialized current Instr中的`Single/plan-0 + Unplaced/plan-0` complete tuple。compiler从每个该类
  all-rank tuple的SSA、typed effects和可证明buffer ranges原子派生一个
  `DisjointComponents/nonzero-plan` sibling；worker选择直接写入actual issue attrs，并从改写后的current IR
  fresh重建minimum participant joins。已有nonzero worker assignment只作为独立actual seed保留，不被原地
  重写。worker sibling形成后，`Single`与`DisjointComponents`两类realization才分别派生保留原worker
  assignment的bounded `StaticFixedSlot/nonzero-plan` sibling。null、缺rank、mixed correspondence和仅有
  metadata而无actual worker assignment的tuple均跳过。
  common owner不保存固定tile-role enum，而是在每个fanout strategy的同一discardable clone中按依赖顺序
  累计四类IR-derived materializer：
  (1) 上游complete traversal已通过`PartialReductionOpInterface`产生local partial/merge并接入typed
  collective；post-import partial materializer只从current Instr message、partial provenance、combiner和
  publisher证明冗余spill/reload可删除。tree按rank
  contribution multiplicity验证ordered message、单一typed combiner、compact publisher和output
  final writer；ring按slice range验证reduce-scatter merge/forward、all-gather传播、origin multiplicity和
  final exact cover。range overlap、gap、duplicate、错位view或无法解释的message tuple保持fail closed；
  (2) 从frontend required output boundary、current produced-value equivalence、exact all-rank WDMA
  descriptor和final-writer relation恢复replicated output publication；保留每个required WDMA，只保留一个
  producer closure并用PeerDataflow round 2把produced value送入其它publisher的fresh SPM destination；
  descriptor不完整、publisher缺失/重复、produced value不等价或存在后续writer时整组不改写；
  (3) 从actual producer store、consumer reload、SSA/view occurrence、effect和lifetime证明一个
  intermediate spill cut，owner复用producer root、peer直接接收进consumer root并删除该组额外WDMA/RDMA；
  (4) 从typed ProgramRankSlice、function boundary、tile-region/view链和compact actual RDMA重建global
  input/parameter tile，生成Direct或ReceiveForward owner-load peer traffic。
  新peer receive preparation只在自己的structured block内前移到该block所有既有/新增transport issue之前；
  materialized peer op随后lower到Instr。SPM planning及periodic Direct-DTE site specialization之后，从current
  complete-rank Instr IR重建accepted direct-call closure、call/region/loop occurrence、message endpoint和
  issue/wait依赖，whole-program wait graph有环、message无匹配、静态site需要不一致binding或occurrence无法
  证明时整代fail closed；该只读gate不写binding或保留analysis side table。resident materializer输出的
  canonical/unplaced current Instr tuple先原子派生typed worker sibling，再从保留各自worker assignment的
  siblings派生bounded static fixed-slot realization。已有nonzero worker或actual fixed-slot seed保持自身
  realization，不原地重写。每个新sibling都fresh重跑completion、SPM/DDR、Direct-DTE、whole-variant
  resource和target capability gate；全部generation/rank domain先验证，再向每个frontier原子追加。
- Output artifact / IR:
  existing frontiers加上fresh stable ordinal、artifact kind为Resident的complete-rank actual siblings。
  每个materialized strategy是一个semantic generation；canonical/typed-worker realization及其可选
  StaticFixedSlot siblings共享stable ordinal，由worker-placement和buffering metadata建立correspondence，
  而actual attrs/joins/SCF仍是语义事实。module中的RDMA/WDMA、peer movement、token/wait、compute/reduce、
  worker assignment、loop和required writeback是唯一语义事实；没有role enum、owner map、channel表、
  stage list或shadow schedule。
- Downstream consumer:
  现有whole-variant selector消费完整frontiers；winner继续经过whole-variant DDR planning、Direct-DTE
  matching/binding、all-rank message/range/resource、TargetCall/LLVM、device link、package和model/runtime
  gates。该stage不发布独立中间artifact。
- User-level driver / named pipeline:
  wafer-compile source-to-bundle production pipeline，在frontier import之后、whole-variant selection之前
  由ExecutableBundle owner调用；没有NoC专用public mode或用户手工pass pipeline。
- Explicit non-goals:
  resident role materializer本身不重新选择compute implementation、tile owner或collective算法，也不删除或
  重分配required output store。worker placement是同一candidate owner中的独立post-Instr派生维度：它只从
  canonical/unplaced current IR构造atomic sibling，不原地重写已有nonzero assignment；result/operand/partial
  interface traversal在其上游complete-traversal candidate stage完成并由同一named pipeline消费。output
  publication当前只接受所有rank要求同一个replicated exact full-buffer值的情形；partitioned output、
  range-changing view和需要重新聚合/计算的output仍保持baseline。
  partial不为普通local reduction凭空发明跨rank collective，也不改变source要求的reduction order；
  dynamic/conditional control、zero-trip transport、递归/间接/未调用
  DTE helper及DTE-bearing multi-block function等无法证明的dynamic occurrence继续fail closed。不按GEMM、
  attention、MoE、shape或名字匹配，不创建detached tile
  graph/channel/owner/stage协议，不在上层IR写DTE FSM、physical endpoint、route或packet。partitioned
  boundary的unique shard已经是每shard一次DDR load的最小coverage；没有重复global tile consumer时不生成
  peer traffic，也不声称降低总DDR。
- Completion gate:
  typed boundary/global-tile relation、exact compact descriptor、intermediate producer/consumer
  provenance/lifetime、Direct/ReceiveForward、null/mixed tuple rejection、
  seed cap/determinism、all-rank failure atomicity、fixed-slot、typed worker+DTE、per-block receive
  preparation、跨block可进展schedule和真实wait cycle拒绝均由host tests覆盖。helper定义顺序不参与
  message occurrence，调用path/次数错位及unused helper由negative覆盖。source-to-package/model/no-card
  纵向分别证明replicated input
  16次RDMA降为1次并出现15份peer traffic、intermediate cut不再产生额外DDR。output unit/production
  coverage证明一个producer通过15份PeerDataflow round-2 traffic供给16个required WDMA，descriptor缺口
  原子拒绝。tree与ring partial source纵向都经过target model、ELF、package和no-card；strict ring资格从
  current IR证明DDR movement只包含entry-boundary read和returned-output write，不读取artifact标签。
  compound纵向只证明replicated boundary、partial、GEMM/add/mul downstream compute和required output
  coverage，不代签intermediate或round-2 output materializer；这两类由独立正负例拥有。large contraction、
  五类role和NoC/fixed-slot/worker actual composition均有独立host证据；configured board仍是external gate。
```

### B. Generic tile-role foundation（当前实现）

```text
Pipeline position:
- Upstream artifact / IR:
  与A相同的complete-rank actual tuples和typed global/local boundary；另外从current accepted IR读取
  TilingInterface、PartialReductionOpInterface、IndexRelation、SSA use-def、view/subset、
  MemoryEffectOpInterface、numeric permissions、execution mesh及actual producer/consumer placement。
- Current stage responsibility:
  把A的exact all-rank ownership/materialization扩展到五类interface-derived role：
  input、parameter、intermediate、partial/reduction和output。result-driven pull、operand-driven push和
  partial-reduction只能由标准interface及current IR进入；有界枚举owner、peer fan-out/forward、显式
  local merge、resident/spill cut和required writeback，对每个参数点原子修改complete-rank actual tuple。
- Output artifact / IR:
  覆盖完整静态traversal的complete-rank actual clones；owner-only boundary movement、resident SSA、
  peer send/recv、local compute/reduce、token/wait、SCF和required output publication全部显式。
- Downstream consumer:
  与A相同，并复用C的generic asynchronous realization。
- User-level driver / named pipeline:
  与A相同的wafer-compile共同candidate owner；不增加operator-specific driver或manual pipeline。
- Explicit non-goals:
  不假设router multicast/in-network reduction，不创建NoC专用tensor dialect、运行时task graph或
  serialized schedule；不以论文中的GPU block/CTA、channel API、cycle或带宽参数作为target事实；第一版
  不支持dynamic/ragged rank tile、跨卡route或运行时token routing。
- Completion gate:
  input/parameter、intermediate、partial/reduction和output role各有interface-derived正负例；至少两个
  structured compute family、一个multi-operator compound source、一个large contraction和一个复杂
  dataflow case产生fully accepted qualification candidate，分别通过完整host/no-card/model/package；
  final IR证明DDR transaction变化、显式NoC traffic、all-and-only output coverage与完整completion。只有
  complete-rank profitability门禁闭合的candidate才可成为normal winner，之后仍需fresh board qualification。
```

### C. Instruction-level asynchronous realization

```text
Pipeline position:
- Upstream artifact / IR:
  complete-rank instruction clone；structured loops、fixed-size SPM buffers/views、typed NCC/Kcore work以及
  peer send/recv已经lower为Direct-DTE issue token和exact wait语义，但offset/binding可以尚未提交。
- Current stage responsibility:
  先从current complete-rank IR重建typed message及entry call-expanded structured execution trace：
  receive issue是prepare，V3 send issue依赖matching receive prepare，exact wait依赖matching send，并以
  rank内执行顺序形成whole-program wait graph；只有无环且每个static site的所有call occurrence共享同一
  physical peer binding时才进入后续realization。该analysis每次从IR重算，不进入attr、stage plan或side
  table。随后从current SSA/effect/range/token构造dependency DAG；把owner load、peer transport、local compute/
  reduce和writeback分配到合法stage与fixed slots；用pinned scf::pipelineForLoop只机械生成
  prologue/steady/epilogue；把exact DTE wait放在最晚的真实consumer/reuse cut，把participant join限制在
  NCC跨domain观察或terminal；每次rewrite后fresh重建completion、lifetime和resource事实。
- Output artifact / IR:
  实际multi-buffer complete-rank instruction program。loop-carried values、allocation roots/views、
  DTE tokens/waits、NCC participants和SCF control flow完整表达执行窗口；不存在stage plan attr或隐藏signal。
- Downstream consumer:
  SPM/DDR fixed-capacity placement、all-rank Direct-DTE matching/binding、whole-variant selection、
  TargetCall/LLVM、package、functional model和board runtime。
- User-level driver / named pipeline:
  现有wafer-compile production pipeline中的generic software-pipeline stage。
- Explicit non-goals:
  本software-pipeline stage不重新选择tile owner、compute implementation、collective算法或physical layout；
  独立post-Instr worker-placement stage已经把选择物化为actual attrs和minimum joins，本stage只在派生
  fixed-slot时保留该assignment。不从paper公式、queue depth、Q9 profile或live card决定window；不让NCC join
  完成DTE，也不让DTE wait完成NCC。当前call-expanded proof
  只接受direct、acyclic、single-block function closure及常量正步长structured loop；无法静态证明的
  control occurrence不通过猜测或函数名放行。
- Completion gate:
  NoC-resident candidate的single/odd/even iteration、tail、one/two-slot capacity、source/destination early reuse、
  DTE->NCC、NCC->DTE/Kcore、same/cross-worker和terminal均有exact正负例；steady state没有可避免的
  waitfinish，DTE wait只消费matching event；prologue/nested或sibling steady/epilogue的可进展跨block
  schedule通过，互等cycle、message/call occurrence错位与unprovable control拒绝且不遗留binding；
  同源baseline与winner均通过完整late gates和fresh board correctness，性能promotion只使用Q9对最终产物的
  matched observation。
```

### D. Complete-rank NoC profitability admission

```text
Pipeline position:
- Upstream artifact / IR:
  已通过completion、SPM/DDR placement、Direct-DTE matching、whole-card resource和target capability gate的
  disposable complete-rank actual Instr candidate，以及同一source/config中已经通过全部late gate的reserved
  baseline；ExecutionConfig提供exact TargetProfileId和typed execution topology。
- Current stage responsibility:
  只从两份current final Instr fresh重算DDR read/write、per-rank CT/NE work、Direct-DTE message/bytes、
  source/destination endpoint pressure、minimum-hop total link-byte、minimum-hop message demand、wait/join、
  SPM movement和dependency事实。上述可由final Instr与typed topology完整计数的
  bytes/messages/ops/endpoints/minimum-hop work保持
  exact `Known`；`Known`不表示硬件采用某条physical route，也不表示对应duration已经测量。物理route和
  arbiter仍未知，point model只按typed topology构造modeled deterministic shortest path，并显式记录
  `EstimatedRoute` assumption。
  analysis再消费compiler-shipped versioned point parameters形成nominal makespan，并把真正有证据的
  peak floor / conservative bound分别保留。16个tile的DDR traffic只除以一次整卡共享带宽。没有qualified
  multi-buffer时，DDR、NoC和compute按dependency-order sequential phases计费；只有current IR的fixed-slot/
  multi-buffer结构与target capability共同通过资格后，才使用steady-state resource maximum。normal production
  允许exact work加point model在20% margin下签发`EstimatedBenefit`；只有candidate conservative upper在同一
  margin下严格小于baseline lower才是`ProvenBenefit`。只有必要工作量仍为dynamic/unsupported或算术无法
  表示时才`Indeterminate`，缺少保守bound本身不再把nominal estimate降为Unknown。
- Output artifact / IR:
  不产生新IR、attr、side table或shadow schedule；只产生invocation-local的typed duration interval、decision
  和reason，供whole-variant selector决定是否把已存在candidate送入normal target gate/frontier。
- Downstream consumer:
  production whole-variant selector；放行winner继续进入TargetCall/LLVM、device link、package、model和board
  runtime，未放行candidate仍可由既有compiler-private qualification mode消费。
- User-level driver / named pipeline:
  wafer-compile source-to-bundle production pipeline中的complete-rank candidate-selection；没有NoC专用
  public CLI、workload profile或live-card反馈。
- Explicit non-goals:
  不按GEMM、shape、case名或artifact kind恢复收益；不把200 GB/s峰值、150 GB/s nominal、128 GB/s
  directional reference混成同一置信级别；不把modeled deterministic shortest path冒充physical route，
  不猜SPM bank、worker fairness、queue occupancy或未校准group overlap。DTE startup、endpoint service、
  hop、SPM和control的point prior只签发`EstimatedBenefit`，不能冒充板端测量或`ProvenBenefit`。
- Completion gate:
  host tests覆盖整卡共享DDR、read/write共同占用、4x4 mesh endpoint/link work、消息计数、Unknown传播、
  route-independent lower bound、`EstimatedRoute`与endpoint hotspot不混淆、小payload normal baseline、
  大DDR-bound `EstimatedBenefit`、synthetic完整bounds的`ProvenBenefit`、sequential/fixed-slot两种schedule、
  非NoC优化不回归，以及final IR变化后fresh重算。板端promotion仍要求同source/profile/shape、平衡顺序、
  重复Primary device elapsed、完整correctness/status/terminal和final-artifact attestation形成versioned row。
```

## 1. 已确认事实与设计决议

### 1.1 当前实现的真实边界

当前production基础纵向位于rank frontier完成finalization/import之后、whole-variant selection之前。它不从
fixture重造一条结构化schedule，而是闭合已有actual candidate的complete-tuple composition：

- 用whole-variant attempt plan从各rank frontier恢复correspondence一致的actual tuples；baseline优先，
  只保留有界数量的materialized current Instr seeds。canonical `Single + Unplaced` complete tuple先原子
  派生actual typed `DisjointComponents` worker sibling，再从保留各自worker assignment的siblings派生
  `StaticFixedSlot`；已有nonzero/fixed-slot actual seed保持原样且不被原地重写；
- common层只负责complete-rank clone、lowering、receive preparation ordering、SPM/fixed-slot和late gate；
  partial、output、intermediate和input/parameter由独立materializer callback从current IR发现并在同一clone
  累计，不维护固定role enum；
- 从frontend typed distributed input/parameter、function argument、tile-region/block argument、
  `memref.subview`/`memref.cast`、structured occurrence与actual compact RDMA重建rank-local tile和global tile；
- 对replicated boundary验证所有rank slice等价覆盖同一global tile；对partitioned boundary验证dense、
  disjoint且完整覆盖global shape；
- 当且仅当两个以上actual loads覆盖同一global tile时，从实际有coverage的load holder中确定owner，
  保留owner RDMA，并生成Direct或ReceiveForward显式peer movement；普通partitioned unique shard不改写；
- 对intermediate要求producer store与consumer reload的global tile、structured occurrence、定义版本、
  effect和lifetime全部可证明；owner直接把producer root交给本地consumer，peer接收进各自consumer root，
  删除该组spill/reload和consumer allocation，从actual IR形成zero-extra-DDR cut；
- partial路径从standard interface产生真实local partial/merge，并把已有typed tree/ring collective接入
  resident proof。tree按contribution multiplicity、combiner、compact publisher和final writer验证；
  ring从final `(source,destination,phase,round,slice)` tuple验证reduce-scatter/all-gather、origin
  multiplicity及final exact cover；任一range gap、overlap、duplicate、错位view或publisher覆盖均保守拒绝；
- peer op立即lower到Instr；typed worker sibling从该canonical/unplaced current IR的SSA、effects和ranges派生，
  actual attrs与fresh minimum joins是唯一placement事实。worker-preserving fixed-slot siblings随后分别重跑
  completion、SPM/DDR、Direct-DTE、target capability和whole-variant resource gate，最后以fresh stable
  ordinal原子追加完整rank domain。

跨消息顺序已由whole-program hard gate拥有。新peer receive preparation及其可安全移动的static allocation
只在各自structured block内前移到该block任何既有或新增DTE issue之前，consumer wait仍留在真实消费点；
是否能组合多个block、seed已有collective及新peer traffic，不再由block identity决定。peer lowering、SPM
planning与fixed-slot periodic site specialization之后，gate从current complete-rank Instr IR展开direct
accepted call closure，以typed source/destination/message identity、call/TileRegion/常量loop occurrence及
rank内issue/wait顺序建依赖图：V3 send issue等待peer receive prepare，exact wait等待matching send。
message/call occurrence不等、一个static site需要不同binding、未调用helper、无法证明的control或图中存在cycle
都会整代原子拒绝；成功时只证明transport可进展，不保存graph，也不移动wait或发明pipeline stage。纯
elementwise add/multiply source已经通过同一generic candidate owner：replicated input候选由16次DDR read
收敛为1次owner read和15份peer traffic；当前4 KiB实例因message startup/route成本无法清除production margin，
normal production保留baseline，winner/baseline均通过target model，最终ELF不含GEMM调用。这证明op无关的
candidate formation与profitability gate分离，不把“小case也启动NoC”当功能证明。

required output store仍保持原样，不改变host-visible ABI。当前output publication只在所有rank的frontend
boundary、produced-value SSA/effect equivalence、exact full-buffer WDMA和final-writer关系全部一致时，
保留一个producer closure并显式转发到其它required publisher；它不重分配publisher、不减少required
writeback，也不开放partitioned/range-changing output。当前“partial resident”只表示已证明的explicit
collective周围DDR cut可被删除；普通local reduction仍不会凭空获得跨rank collective。四类materializer
已经能在同一candidate累计，result/operand/partial interface recipes也由上游bounded traversal进入同一
frontier。compound source只拥有其final IR实际出现的boundary+partial证据，不能据此宣称每个role都在该
source中触发。

### 1.2 通用抽象是双向tile traversal

一个logical tile在不同SSA edge上可以同时是某个op的result、另一个op的operand、一个rank的boundary
slice、另一个rank的receive buffer或最终output slice。设计不为这些角色建立固定enum；角色由current
use-def和boundary relation重算。

通用transformation使用三条标准路径：

1. **Result-driven pull**：从result tile调用`generateResultTileValue`或
   `getIterationDomainTileFromResultTile`，向producer反推所需tile。现有complete traversal属于这条路径。
2. **Operand-driven push**：从已resident或刚到达的operand tile调用
   `getIterationDomainTileFromOperandTile`和`getTiledImplementationFromOperandTile`，materialize恰好消费
   该tile的consumer iteration tile。它覆盖input fan-out、AllGather-like prologue和producer到多个consumer。
3. **Partial reduction**：只有实现`PartialReductionOpInterface`且numeric contract允许时，使用
   `generateInitialTensorForPartialReduction`、`tileToPartialReduction`和`mergeReductions`生成真实partial
   SSA与merge；不再用matmul/generic op-name matcher恢复reduction。

`TilingInterface`只提供mechanism，不提供profitability。owner、residency、transport和work quantum仍由06的
bounded actual-clone candidate owner选择，并由final IR exact facts比较。当前complete traversal已有
result-driven、operand-driven和partial-reduction recipes；post-Instr materializer再从actual boundary、
producer/consumer cut、typed tree/ring collective和produced-value relation恢复五类role。output保留
all-rank required WDMA，partial保留source numeric order。partitioned/range-changing output、无法精确证明的
view/range和普通local reduction自动跨rank化仍不支持，不能从相近shape或aggregate bytes推断。

### 1.3 三种粒度必须解耦

- **Compute tile**：一个structured op implementation实际消费/产生的iteration tile。
- **Data tile**：沿一条SSA edge由exact IndexRelation确定的logical operand/result region。
- **Transport segment**：一次peer message覆盖的一个或多个相邻data tile physical segments。
- **Work quantum**：software pipeline一次推进的SCF iteration或有界iteration group。

四者可以相同，但协议不要求相同。transport可以聚合多个连续data tile以减少message，或在合法physical
segments上拆分一个data tile；compute tile不能为了通信方便被无条件切小；work quantum也不能成为tensor
shape或communication identity。

这吸收TileLink中compute/communication tile解耦的有效部分，但Wafer不复制其channel/mapping side table：
accepted形态只保留实际subview、peer op、SCF和token。

## 2. Tile ownership 与驻留

### 2.1 Global logical tile relation

all-rank synthesis需要判断两个rank看到的tile是否代表同一global logical region。一次transformation内从以下
事实组合：

- verified `ProgramRankSlice`的global offsets/sizes/strides和replicated/partitioned distribution；
- current rank specialization；
- standard tensor/memref slice、view、reshape和structured indexing map；
- 08的`IndexRelation`、physical encoding valid domain和transfer realizability；
- explicit logical collective的rank group、axis和result mapping。

组合结果是可失效、可重算的analysis value，不写入rank-local IR。candidate materialize后，whole-variant
acceptance必须重新用同一upstream program boundary和actual send/recv/subview验证global cover；不能只因
bytes相等或message成对就认定语义正确。

### 2.2 合法owner集合

对每个data tile，合法producer/DDR-owner来自可证明事实：

- replicated input/parameter boundary tile：所有持有同一global region且actual clone中确有matching load的
  rank可作为当前owner候选；当前实现按rank-major current-IR walk建立exact global-tile/typed-occurrence
  equivalence class，以program member ordinal和class discovery ordinal从有coverage的rank集合确定性选取owner；
- partitioned input/parameter boundary tile：只有verified rank slice覆盖该region且actual clone中确有
  matching load的rank可直接从DDR取得。每个unique shard只有一个load时已经达到“一次DDR load/unique
  shard”的最小coverage，不生成peer traffic，也不能把global tensor的partitioned读取宣称为总DDR下降；
- intermediate tile：只有实际materialized producer result所在rank可作为owner；当前实现只在producer
  store、consumer reload、same-version use-def/effect/lifetime和global tile全部可证明时建立peer residency，
  无法证明则保留原spill；
- recomputable tile：只有op可speculate、effect/numeric/recompute gate允许且actual clone已物化计算时才增加
  producer；
- partial tile：owner由实际partial-reduction iteration、combiner SSA和typed collective message决定；
  当前tree/ring路径不重新选择算法，只有exact range、contribution/origin multiplicity、merge expression、
  publisher和final-writer全部闭合时才删除spill/reload；
- output tile：required publisher由frontend boundary、SSA result relation和host ABI共同约束；当前rewrite
  保留existing store；replicated exact full-buffer值可由一个producer经round-2 peer movement供给其它
  required rank，仍由原publisher all-and-only writeback。

owner choice不猜DDR controller、bank或physical route。replicated输入可用tile coordinate、typed occurrence
ordinal与execution mesh的确定性affine mapping把不同tile分散到多个owner，以利用多rank load和NoC aggregate
traffic；这种mapping必须从current IR和verified relation重算，不能从tensor名、地址或进程内semantic hash选择。
LLVM hash只可作为同一次analysis中的候选预筛，不能决定equivalence-class顺序、communication ID、physical owner
或package bytes；最终分组仍须由exact relation/SSA/effect证明。

### 2.3 驻留和转发

一个tile只有在以下条件全部成立时才能跨edge保持SPM/NoC resident：

- logical relation、dtype、valid/padding domain和physical segments可证明；
- producer completion与consumer availability由same-domain issue order、typed participant join或exact DTE
  event表达；
- source/destination root在最后一个transport/consumer完成前保持live；
- fan-out中的每个reader和overwrite/reuse之间没有RAW/WAR/WAW冲突；
- SPM fixed-capacity、alignment、descriptor和event gate通过；
- host、Kcore cache或DDR publication等external observer没有被跳过。

receiver可在一个segment到达后立即消费、local reduce并向下一peer转发；无需等待整个logical tensor。但每个
forward edge都必须有独立buffer range和completion，不能以“collective尚未结束”隐藏。

## 3. Materialized IR 边界

### 3.1 最小新增buffer-level peer语义

现有`wafer.tile.all_*`表示logical collective的buffer-level handoff；现有`wafer.instr.dte_*`已经包含
instruction-level transport和token。二者都不能准确表达“physical dataflow选择后，一个普通boundary或
intermediate SPM tile在两个logical rank之间移动”这一中间语义。

当前已用最小target-abstract pair表达该边界：

- `wafer.tile.peer_send`：读取一个SPM buffer range，携带logical peer、fixed bytes和typed communication
  identity；
- `wafer.tile.peer_recv`：写入一个SPM destination range，携带反向peer、相同identity和bytes。

它们不携带physical endpoint、FSM、route、algorithm、stage、slot、cost或owner kind。op verifier检查SPM
memory space、static byte cover、logical peer domain和effect；whole-rank conversion把它们lower成
`wafer.instr.dte_send/recv/wait`，whole-variant acceptance再做一一匹配和physical binding。

当前input、intermediate和output fan-out由多个explicit send或receive-then-forward表达；output使用独立
PeerDataflow round-2 identity把同一个produced value送到所有replicated required publishers，并保留每个
publisher的WDMA。partial路径复用已有typed collective展开的recv/local NCC reduce/forward，而不把
collective改写成peer pair。TX81 raw broadcast/scatter/source-gather/fan-in或未来router collective只有在
独立typed target capability闭合后，才可作为相同upper IR的lowering alternative；不得新增
`noc.reduce`并假设fabric计算。

### 3.2 Accepted traversal形态

一个accepted candidate可以包含：

```text
scf.for tile/work quantum
  owner rank:
    DDR subview -> tile.load -> SPM slot
    peer_send SPM slot
  non-owner rank:
    peer_recv -> SPM slot
  any consumer rank:
    tile compute / local reduce / peer forward
    optional resident handoff to next structured op
  required output owner:
    tile.store -> original DDR boundary slice
```

这只是总体形态示意，不是固定stage数或IR模板。当前A从actual boundary RDMA、intermediate spill/reload、
exact replicated output publication和已有typed collective partial cut识别机会，并在一个complete-rank
clone中累计。它保留每个required output store；对replicated output只替换peer rank的冗余producer
closure，并确保每个publisher在writeback前取得同一正确tile。若boundary是partitioned output，当前保持
baseline且只允许对应slice owner写回。任何优化都不能静默改变package reconstruction或host ABI。

### 3.3 无shadow schedule

允许一次rewrite调用使用短生命周期C++ proposal枚举：

- tile seed direction；
- legal owner choice；
- fan-out/forward tree edge；
- bounded tile/segment/work-quantum参数；
- resident/spill choice。

proposal一旦选中必须立即改写完整rank tuple；成功后只保存actual clones，失败则丢弃全部tuple。不得在IR
attr、bundle、package、cache或diagnostic schema中保留第二份owner map、tile graph或stage list。下游所有
legality/cost从修改后的IRfresh重算。

## 4. Generic candidate space

### 4.1 Ingress / operand tile

当前production stage从typed distributed input和parameter boundary的actual RDMA出发，物化：

- baseline：每个consumer rank独立DDR load；
- owner-load + direct fan-out；
- owner-load + receive/forward chain；
- 基于同一upstream complete tuple的canonical/unplaced与atomic typed-worker siblings，以及随后保留各自
  worker assignment派生的合法static fixed-slot siblings。

complete traversal先通过operand tile和`TilingInterface`主动推进真实consumer iteration；进入post-Instr
resident stage后，actual RDMA及其destination use-def已经证明该rank确实消费该tile，ProgramRankSlice和
view链负责把local tile映射为global tile并证明owner coverage。只有同一global tile的actual load group大小
至少为2才形成fan-out；normal partitioned unique shards保持各自一次DDR load。complete tuple seeds、
fan-out strategy、worker siblings和fixed-slot neighbors分别有hard cap，不枚举rank subset或无actual
consumer的receiver。

### 4.2 Intermediate tile

producer result到consumer operand的exact relation允许：

- same-rank same-root resident handoff；
- cross-rank peer transfer；
- receive后直接供多个local consumer；
- transfer与recompute的bounded alternatives；
- compound operation中跨compute engine的pipeline。

任一observable store、unsupported relation、effect barrier、numeric change或capacity conflict都切断resident
edge并保留baseline。当前实现已对actual producer store/consumer reload cut建立共享global-tile relation：
owner本地复用producer root，peer直接接收进consumer root，并在所有rank删除matching WDMA/RDMA、spill
allocation和reload allocation。pure overwrite允许无boundary定义；read-modify-write必须证明相同prior
state。alias write/free、early observer、不同structured occurrence、dynamic/unrepresentable view或任一rank
缺口都会整代回退。operand-driven consumer traversal已经进入candidate recipe；bounded recompute及无法由
现有SSA cut表达的更复杂producer chain仍是明确非目标，不能靠名字或side table补语义。

### 4.3 Partial / reduction tile

partial reduction有两种正交选择：

- 计算分解：如何把reduction iteration分成有界work quanta；
- 通信聚合：在哪个rank、以什么explicit peer/local-reduce顺序合并partial。

Stream-K的可迁移原则是按总work而非仅按output tile分解，并把额外partial seam限制为与资源宽度相关的有界
数量；不能照搬其GPU CTA/fixup协议。当前candidate通过`PartialReductionOpInterface`和numeric permission
materialize真实local partial/merge，再消费typed tree/ring collective。tree proof检查rank contribution、
ordered merge、compact publisher和output final writer；ring proof检查每个slice的source range、
reduce-scatter merge/forward、all-gather propagation、origin multiplicity和final exact cover。错位subview、
重复/缺失贡献、range overlap/gap、noncompact publisher和publisher后覆盖均原子拒绝。普通local reduction
不会仅因实现了interface就被自动升级为跨rank collective；memory-bound shape若增加NoC/NCC work而不减少DDR，
也不因“overlap更多”自动胜出。

### 4.4 Egress / output tile

最终output choice包括：

- 原rank直接writeback；
- partial在NoC中聚合到boundary owner后一次writeback；
- producer rank先转发给required output rank，再由后者按原ABI writeback；
- intermediate output只被下游op消费时删除DDR store/reload cut。

host-visible output coverage、rank slice和publication保持hard legality；输出少写或重复写均不是性能
tradeoff，而是candidate rejection。当前实现对replicated exact full-buffer output保留16个required WDMA，
只保留一个等价producer closure，并用PeerDataflow round 2把值送入其它publisher的fresh SPM destination。
partitioned、range-changing或需要重新聚合/计算的output继续保留baseline。input fan-out winner本身不能
代签output publication；两类role分别从final IR重证。

## 5. Scheduling 与 engine parallelism

NoC-resident synthesis只创造可并行的真实work和buffer关系，不自行猜cycle stage。generic instruction
software-pipeline stage在lowered complete-rank Instr上统一处理：

- WDMA/RDMA、CT/NE/TDMA、Direct DTE和Kcore/NCC worker的dependency；
- issue-to-completion lifetime；
- fixed slots和loop-carried rotation；
- prologue、steady state、epilogue；
- minimum-strength、latest-unavoidable DTE wait和NCC participant join。

典型steady state可以是：

```text
iteration i:
  DTE transfers/forwards data tile i-1; NCC performs any required local merge
  matrix/vector engine computes tile i
  RDMA prepares owner input tile i+1
```

这不是必须实现的三stage模板。实际stage由DAG和target capabilities决定。V3 Direct-DTE production路径为
`prepare -> explicit issue -> exact wait/release`；V1/V2只保留wait内auto-issue兼容，不取得独立transport
overlap legality。任何profile下，只有一个buffer导致reuse冲突或wait无法推迟到真实consumer cut时，都不能
宣称overlap winner。

当前NoC owner先从materialized current Instr的canonical `Single + Unplaced` complete tuple派生一个
all-rank atomic typed-worker sibling。dependency lanes只从SSA、typed value-associated effects、exact或保守
static buffer ranges和stable issue order建立；Unknown conflict保持同lane。派生clone直接写actual worker attrs，
删除旧compiler-generated joins并从current IR fresh建立minimum participant joins；source及已有nonzero worker
candidate都不被原地重写。随后才从canonical和typed-worker siblings分别尝试保留worker assignment的generic
static fixed-slot neighbors；只有structural loop、stage/slot、target scheduling capability以及fresh
completion/SPM/DDR/Direct-DTE/resource gate全部通过时才追加。whole-variant coordinator对V3 worker+DTE组合
查询Direct-DTE overlap capability并通过worker qualification late gates。组合资格在同一个pre-target tuple上
同时检查boundary-only DDR、实际Direct-DTE、fixed-slot和多个含非零值的typed worker；各自独立passing或
metadata拼接均不算组合。Q38 generic prologue/steady/epilogue直接消费该actual IR；production profitability
仍只使用静态target policy，不读取live profile。

每个structured block内，新receive preparation先于该block既有和新增transport issue，matching wait仍位于
真实consumer/reuse cut。跨block与已有collective的组合由complete-rank call-expanded wait graph统一判定：
prologue、nested/sibling steady occurrence及epilogue只要结构对应且图无环即可开放，真实互等或ambiguous call
occurrence拒绝。该闭环没有替代C后续根据SSA/effect/lifetime移动wait、旋转buffer并生成generic pipeline。

## 6. Cost、selection 与DDR瓶颈

candidate必须从final actual IR收集至少：

- DDR read/write bytes、transactions和full-shape/intermediate materialization数；
- SPM movement bytes、allocation high-water、fixed-slot bytes和lifetime；
- Direct-DTE bytes/messages/waits、source/destination rank、maximum endpoint pressure、
  minimum-hop total link-byte、directed link count和idealized peak-link lower bound；
- NCC/Kcore logical work、partial merge/recompute work；
- issue、exact DTE wait、participant join、terminal drain和pipeline fill/drain；
- output coverage和immutable payload identity。

exact resource Pareto继续负责无tradeoff的支配关系；出现“DDR下降、candidate引入或保留NoC依赖”的
跨资源tradeoff时，必须再过
paired duration gate，不能让resource class字典序直接选winner。成本合同先区分事实层和估计层：

- final Instr与typed topology能完整计数的DDR/SPM bytes、compute ops、DTE bytes/messages、source/destination
  endpoint pressure和minimum-hop work是exact `Known`；
- physical route、arbiter和实际hot-link并不因此Known。point model使用modeled deterministic shortest path
  暴露cut/hotspot，并将该项显式标成`EstimatedRoute` assumption；在这个假设下算出的link load是模型结果，
  不是硬件路由事实；
- nominal duration是exact work乘versioned target point parameters的静态分析结果；conservative lower/upper
  只由真正的硬件/校准bound形成。`EstimatedBenefit`与`ProvenBenefit`因此是不同结论。

当前versioned target参数分为不同证据强度：

- `200 GB/s`是整卡DDR峰值，用于DDR duration乐观下界；
- `150 GB/s`是当前16-tile共享整卡的nominal operating point，不是每tile独享速率，也不冒充
  guaranteed bound；
- `128 GB/s`是单方向NoC link payload reference；point model另以`128 GB/s`作为DTE endpoint service policy
  prior。二者数值相同但语义分开，均不是fabric aggregate、collective latency或sustained lower bound；
- Direct-DTE每message的`α = 10 us`与maximum modeled route每hop `1 ns`是当前profile中惩罚小消息和
  route fill的point policy prior，用来阻止小payload仅凭DDR byte下降胜出；它们不是板端测量，未来必须由
  同profile matched board calibration替换；
- SPM point prior为每tile `128 GB/s`，来源是1024-bit interface在1 GHz model quantum下的一beat估计；
  instruction、DTE event release和NCC participant wait的control prior均为`1 ns`。这些值不是SPM sustained
  bandwidth或wait latency bound；
- CT/NE documented peak logical throughput作为compute point reference，不冒充sustained lower bound。

对candidate `P`分别形成resource duration interval。整卡DDR项为：

```text
DDR lower  = total_card_bytes / 200 GB/s
DDR nominal = total_card_bytes / 150 GB/s
DDR upper  = Unknown，直到profile有重复样本支持的sustained lower bandwidth
```

NoC乐观lower使用route-independent
`minimum-hop total link-byte / directed link count`形成idealized peak-link floor。nominal在modeled
deterministic shortest path下取peak directed-link serialization、maximum transmit endpoint与maximum receive
endpoint的最大值，每个endpoint都显式加入message `α`，再加入route fill hop prior。compute nominal按每rank
engine work计算后跨rank取最大；SPM用maximum per-rank movement和per-tile point prior；control按maximum
rank-local issue/event/wait计数。所有work仍来自current final IR，policy prior不回写IR。

没有qualified multi-buffer时，nominal makespan为：

```text
max(DDR nominal + compute nominal + NoC nominal, SPM nominal) + control nominal
```

只有current IR具有完整fixed-slot/multi-buffer recurrence、exact wait/reuse cut，且target capability gate也
通过时，才使用steady-state resource envelope：

```text
max(DDR nominal, compute nominal, NoC nominal, SPM nominal) + control nominal
```

这对应实际prologue/steady/epilogue结构，而不是由“存在DTE”或queue depth自动授予overlap credit。真正的
  upper以exact minimum-hop message demand乘route dilation和hop upper覆盖串行message-hop；仍要求
  DDR/NoC/endpoint/message startup/route dilation/hop fill/compute/SPM/control的保守参数，缺失时
upper保持Unknown，但不影响point estimate自身可用。

production先用同一参数实例做paired point comparison：

```text
candidate.nominal * 1.20 < baseline.nominal
```

该式成立时签发`EstimatedBenefit`并允许normal production选择；不成立则`Reject`。若同时具备真实
conservative bounds且`candidate.upper * 1.20 < baseline.lower`，结论升级为`ProvenBenefit`。必要work
仍为dynamic/unsupported或overflow时才`Indeterminate`；缺少physical route观测、startup measurement或upper
bound只会保留显式assumption/Unknown upper，不再让全部候选退回baseline。qualification路径始终保留合法
candidate，板端matched promotion负责校准prior并验证实际收益，不向compiler提供live feedback。

具体原则：

1. 额外partial或intermediate落DDR的candidate不能以Unknown overlap击败zero-extra-DDR resident candidate；
2. DDR减少而candidate引入或保留NoC依赖，以及SPM增加、compute/reduce增加，都是显式tradeoff，保留frontier；
3. physical route未知时，lower只使用topology可证明的minimum-hop work；nominal可使用显式
   `EstimatedRoute`的modeled deterministic shortest path，但不能把其peak link load伪装成actual hot-link、
   aggregate guaranteed bandwidth或duration upper bound；
4. Q9只profile最终baseline/winner package，不能把live card measurement回灌compiler ranking；
5. overlap只由current-IR multi-buffer/fixed-slot事实和target capability共同开放；单个pair、DTE async
   correctness或仅存在多个engine不能代签pipeline group；
6. `EstimatedBenefit`是静态production选择，不是板端promotion证据。真正校准和Q39 completion仍要求同源、
   同ABI、完整correctness后的matched end-to-end observation，不能只看某个engine active counter。

NoC aggregate bandwidth在本设计中体现为：一个global tile避免多次DDR取得后，可以由多个rank并行执行
owner load、peer transfer、local compute和forward；nominal模型用typed topology的directed link count和
modeled deterministic shortest path、endpoint maxima区分mesh并行与单点注入热点。它可以在显式assumption
和margin下签发静态`EstimatedBenefit`，但不是物理route证明或`ProvenBenefit`。

成本比较必须按unique global tile group计数，而不是按boundary的`replicated/partitioned`标签推断收益。
replicated case只有actual重复load被删除时才有DDR read reduction；partitioned case通常每个shard只有一个
load，total DDR bytes与baseline相同，此时不能用“owner load”名义虚构DDR收益。若未来某个partitioned
boundary确有多个rank消费exact same shard，收益也只能来自被actual IR删除的重复load，并同时显式计入新增
NoC bytes/messages。

## 7. 通用case

以下case检验同一generic合同的不同dataflow形态，不是operator注册表。当前已覆盖replicated
input/parameter、producer-store/consumer-reload intermediate、tree/ring partial、replicated output、
纯elementwise add/mul非GEMM纵向、large contraction、compound GEMM/add/mul以及
NoC/fixed-slot/typed-worker actual composition。compound只证明
其final IR实际出现的boundary+partial+downstream compute+required output coverage；intermediate和output
publication由独立角色测试拥有。SUMMA row/column panel、dynamic/ragged tile和未实现target op仍是未来空间，
不进入Q39当前协议。

### 7.1 Large GEMM / SUMMA-like dataflow

以2D rank mesh为例，可以让A panel的合法owner沿row fan-out、B panel的合法owner沿column fan-out，各rank
保持C tile accumulator resident并对K panels双缓冲。每个A/B panel只由覆盖其global slice的owner从DDR取得，
最终C按original output boundary writeback。

示例中的A/B/C、M/N/K和row/column只是structured indexing map与mesh mapping的一种实例；协议事实来自
operand/result tile relation、owner coverage、SPM lifetime和peer IR，不来自`matmul`名字。卷积、batched
contraction或其它实现相同interface的op可走同一机制。

### 7.2 K-sharded producer + reduction

现有K-sharded GEMM每rank产生same-shapedpartial并执行terminal all-reduce。新候选可把output tile细化为
有界work quantum：local partial完成后立即send，receiver wait exact event、local reduce并forward，同时下一
tile继续compute。若保持原floating reduction order所需的顺序无法证明，就只做tile间pipeline，不改变partial
merge顺序。

这是partial/reduction路径的一个case，不为Q35或GEMM注册独立pass。

### 7.3 Compound operation

GEMM→distributed Softmax、GEMM→LayerNorm或attention包含input fan-out、intermediate resident、row
max/sum reduction、multicast-like redistribution和第二个compute consumer。它们验证operand-driven和
multi-operator能力；只有各op interface、numeric/effect和target implementation闭合的子集进入production。

Softmax、LayerNorm和attention名字不进入generic opportunity discovery。尚未有target implementation的op
只能用于structured/host-negative或保持baseline，不能借NoC dataflow绕过lowering legality。

## 8. 论文调研与可迁移结论

| 工作 | 可迁移结论 | 不直接照搬 |
| --- | --- | --- |
| Stream-K | work-centric decomposition应按总work quantum和尾部不均衡建模；额外partial seam、fixup和memory work必须进入同一cost | GPU CTA、workspace/fixup协议、论文中的绝对性能数和特定GEMM matcher |
| TileLink | compute/communication tile可解耦；tile ready/wait、push/pull、resource binding和dependency signal共同决定何时能overlap | 通用signal/channel side table、Triton/NVSHMEM地址模型和论文平台参数 |
| Flux | AllGather-like input通信是compute prologue依赖，ReduceScatter-like output通信是epilogue依赖；过细拆kernel可能损失compute效率 | GEMM prologue/epilogue专用kernel和GPU remote pointer |
| Lightweight Collective-Capable NoC | 通信按`α + nβ`、congestion/dilation和endpoint压力建模；input multicast、reduction及qualified double buffering可减少external-memory压力，steady state取compute/communication resource maximum | router multicast/reduction、Direct Compute Access及论文中的cycle/bandwidth绝对参数不是TX81事实 |
| FlatAttention | 多tile聚合SPM可放大reuse；input load+row/column multicast、local compute、reduce和async多engine可形成完整dataflow | attention专用group形状、softmax公式和假设的hardware collective |
| COMET | compound op必须显式计collective、memory hierarchy位置、operation dependency、ramp-up/down和resource contention | 独立YAML mapping tree和长期collective plan |
| TileFlow / LoopTree | 跨operator tiling、retention、recompute和resource binding要联合考虑；intermediate不应默认落DDR | 另建tree IR或离线mapping artifact |
| TENET / DISTAL | relation可统一表达data assignment、compute placement和machine mapping；data与compute distribution应可独立探索 | 新relation DSL、运行时task graph或把analysis序列化 |
| FEATHER / FlooNoC | layout/dataflow switching和wide multi-stream NoC说明input layout、stream并发与on-chip reorder同样重要 | 未证明的TX81 router、bank、link width或reorder硬件 |

原始资料：

- [Stream-K](https://arxiv.org/abs/2301.03598)
- [TileLink](https://arxiv.org/abs/2503.20313)
- [Flux](https://arxiv.org/abs/2406.06858)
- [A Lightweight High-Throughput Collective-Capable NoC for Large-Scale ML Accelerators](https://arxiv.org/abs/2603.26438)
- [FlatAttention](https://arxiv.org/abs/2604.02110)
- [COMET](https://arxiv.org/abs/2509.00599)
- [TileFlow](https://sizezheng.github.io/files/micro23-101.pdf)
- [LoopTree](https://arxiv.org/abs/2409.13625)
- [TENET](https://arxiv.org/abs/2105.01892)
- [DISTAL](https://arxiv.org/abs/2203.08069)
- [FEATHER](https://arxiv.org/abs/2405.13170)
- [FlooNoC](https://arxiv.org/abs/2409.17606)
- [MLIR TilingInterface](https://mlir.llvm.org/doxygen/TilingInterface_8h_source.html)
- [MLIR SCF dialect](https://mlir.llvm.org/docs/Dialects/SCFDialect/)

## 9. 实施边界

### 9.1 当前实现与host资格

当前实现与host资格包括：

1. **Shared global-tile relation**：candidate owner从frontend `ProgramRankSlice`、function/tile-region
   boundary、view/cast/subview、SCF occurrence和ValueBounds可证明表达恢复global tile；无法证明的dynamic
   relation、private helper call-site composition和overflow均fail closed。
2. **Complete-tuple seed closure**：复用whole-variant attempt plan，保留canonical baseline优先与稳定
   optimized顺序；总seed上限8。canonical `Single + Unplaced` current-Instr tuple先形成all-rank atomic
   `DisjointComponents` sibling，随后两类worker realization各自形成worker-preserving `StaticFixedSlot`
   sibling；已有nonzero/fixed-slot actual seed保持原样。缺失/null/mixed/伪metadata tuple不参与。
3. **Cumulative IR-derived materializers**：common owner不保存role enum。input/parameter materializer要求
   exact compact boundary descriptor并生成owner-load Direct/ReceiveForward；intermediate materializer证明
   same-version producer/consumer cut并删除所有rank的matching spill/reload；partial materializer验证tree
   contribution/combiner/publisher和ring slice-precise reduce-scatter/all-gather provenance；output
   materializer保留all-rank required WDMA并用round 2分发等价produced value。四者按
   partial→output→intermediate→boundary顺序在同一discardable clone累计；任一机会不存在不阻断其它role，
   任一rewrite或late gate失败则整组丢弃。
4. **Buffer-level peer IR与atomic rewrite**：target-abstract peer send/recv拥有typed peer、fixed bytes、
   communication identity、memory effects和verifier，并lower到真实DTE issue/token/wait。每个complete-rank
   generation先通过所有rank验证，failure不改变任何frontier。
5. **Transport-order safety**：每个block的新receive preparation先于该block全部既有/新增DTE issue，
   matching wait仍在consumer cut；lowering、SPM planning和fixed-slot specialization之后从current IR
   call-expand execution occurrence并构造whole-program wait graph。跨block无环组合可追加，cycle、无匹配、
   call/loop path错位、unused helper或无法证明的control整代拒绝，失败不写physical binding。
6. **Actual physical composition**：worker placement从canonical/unplaced current Instr的SSA/effects/ranges
   原子派生，直接物化actual attrs并fresh重建minimum joins；已有nonzero assignment不原地重写。fixed-slot
   只从保留worker assignment的siblings继续派生；每个结果与Direct-DTE共同经过fresh completion、SPM/DDR、
   whole-variant resource和target capability gates。同候选qualification还从current IR检查boundary-only
   DDR、每rank实际DTE和多个含非零值的typed worker；旧frontier slot与baseline保持不变。
7. **Vertical evidence**：replicated-input source-to-package路径证明16个相同input loads变为1个DDR read和
   15份显式peer traffic；intermediate host tests证明owner local reuse和所有参与rank零额外intermediate
   WDMA/RDMA；output host tests证明round-2 all-and-only publication及descriptor/final-writer负例；tree/ring
   partial source纵向证明matching spill/reload消失、slice/round/message精确并通过model/package/no-card。
   pure elementwise纵向独立覆盖无GEMM的replicated-input fan-out candidate、add/mul compute、required output
   以及小payload profitability回退；large contraction覆盖大GEMM estimated winner，compound qualification覆盖
   boundary+partial+GEMM/add/mul+required output；它不
   代签intermediate/output publication。NoC×fixed-slot×nonzero-worker由同一actual candidate的
   target-model、ELF、package、attestation和no-card qualification纵向闭合。

### 9.2 外部门禁与明确扩展边界

Q39现为`done`。静态模型以exact final-IR work、versioned point parameters、`EstimatedRoute`和current-IR
schedule qualification闭合normal production profitability；configured board已完成同源baseline/candidate
重复exact correctness与winner profile。高wait和未闭合overlap转交Q40，编译搜索耗时转交Q41。Q9只观察final
package，不给compiler live feedback。SUMMA专用panel mapping、dynamic/ragged rank tile、跨卡routing、runtime token
routing、partitioned/range-changing output publication、普通local reduction自动发明collective，以及无法由
current SSA/effect/range证明的recompute或producer chain均是明确非目标；若要推进，先建立新的tracking item，
不能重新写成Q39未闭合的隐含任务。

## 10. 验证矩阵

| 维度 | 当前实现与host资格 | Q39总体completion |
| --- | --- | --- |
| Interface/relation | result/operand双向TilingInterface、PartialReductionOpInterface、typed input/parameter ProgramRankSlice、same-version intermediate及produced-output relation均由current IR派生 | dynamic/ragged、无法证明的view/range和普通local reduction跨rank化保持拒绝 |
| Complete tuples | baseline-first；canonical Single/Unplaced current Instr先原子派生typed DisjointComponents sibling，再派生worker-preserving StaticFixedSlot；已有nonzero/fixed-slot不原地重写；四materializer同clone累计；null/mixed/伪metadata跳过；failure atomic | 五类tile role、worker/fixed-slot和structured occurrence保持complete-rank correspondence |
| IR | owner RDMA、intermediate zero-DDR、tree/ring partial、round-2 output、local merge、bytes/peer/message、DTE token/wait、actual worker/loop及required store全部显式 | 不保留role/owner/channel/stage side protocol |
| Lifetime/scheduling | Direct/ReceiveForward、SPM/fixed-slot/worker late gates；receive-prep-before-issue；call-expanded whole-program wait graph；Q38 odd/even/tail和same/cross-worker流水 | Unknown alias/control/message occurrence原子拒绝 |
| Cost | DDR/SPM/NoC message+endpoint/compute/join从final IR exact重算；partitioned unique shard不报DDR下降；200 peak lower、150 DDR nominal、128 link/endpoint point reference和10 us message `α`分级；modeled deterministic shortest path标`EstimatedRoute`；无qualified multi-buffer按sequential phases，qualified fixed-slot按steady-state maximum；20% margin区分`EstimatedBenefit`/`ProvenBenefit` | dynamic/unsupported work才`Indeterminate`；configured-board matched calibration与winner correctness尚未闭合 |
| Genericity | 四materializer覆盖五类role，不含op/shape/name matcher；纯elementwise非GEMM和GEMM两个compute family均进入同一candidate/model，small elementwise回退而large contraction清除margin，compound也覆盖 | case只验证协议，不成为matcher |
| Vertical | input、纯elementwise非GEMM小payload回退、tree/ring partial、large contraction positive、compound qualification、strict boundary-only ring及独立fixed-slot/worker纵向均闭合；intermediate/output/parameter各有独立原子正负例；NoC×fixed-slot×nonzero-worker同候选的target model、ELF、package、attestation和no-card fresh纵向已闭合 | 已完成 |
| Board | K-sharded `4096³`同源baseline/winner完成6次16-rank exact output；winner profile有效且保持16-rank exact | 已完成；wait/overlap转交Q40，compile-time search转交Q41 |

## 11. 完成定义

Q39完成边界：

1. production opportunity discovery不含operator、workload、shape、rank或名字matcher；
2. input/parameter、intermediate、partial/reduction和output tile均由result/operand/partial标准interface进入
   同一个bounded complete-tuple actual-clone owner；
3. peer movement、resident buffers、compute/reduce、writeback和completion全部存在于accepted IR，而不是
   从seed metadata或case角色推断；
4. 至少两个compute family和一个compound source产生fully accepted qualification candidate；exact work加
   versioned point model清除20% margin时可产生normal `EstimatedBenefit`，完整bounds只能把结论升级为
   `ProvenBenefit`；
5. 一个large contraction和一个复杂dataflow case证明DDR transaction下降、NoC traffic显式、SPM合法，
   并进入matched performance qualification；
6. 真实DTE issue/exact wait和generic software pipeline直接消费这些candidate，NoC/fixed-slot/
   nonzero-worker需要的组合关系由actual IR及typed completion证明；
7. baseline与candidate通过相同host/no-card/model/package gate；matched board observation与winner profile
   形成可审计证据并通过fresh correctness；wait/overlap性能由Q40继续；
8. 文档、queue、memory与代码一致，operator-specific原型和旁路协议清理并提交。

只有论文分析、cost model、局部pass、手写peer IR、GEMM microcase、raw DTE probe、结构上出现send/recv，或
“理论上DDR更少”、nominal crossover或单个engine counter都不算完成。五类role、tree/ring、同clone累计、
generic traversal和multi-engine composition
分别由自己的current-IR gate拥有；compound不能代签未在其final message traffic中出现的intermediate或output
publication。Q39已完成；Q40继续wait/overlap，Q41独立处理compile-time search scaling。
