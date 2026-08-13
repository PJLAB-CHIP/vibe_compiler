# Whole-Card Multi-Tile 综合实施计划

状态：2026-08-11 完成独立deterministic baseline controller及最终fresh验证，Q49达到`board-ready`；
2026-08-13 将过宽的Q50 capability migration拆为Q50.A–Q50.K十一个独立队列项，并完成Q50.A的
layout-independent exact edge demand artifact及canonical compatibility lowering；Q50.B保持`next`。算法、IR与
pipeline contract仍只由`tasks/06-physical-dataflow-synthesis.md`拥有；本文件是唯一实施计划，只描述施工顺序、
现有代码处置和独立completion checkpoint；workload/board gate由`tasks/16-verification-contract.md`拥有。

拆分的是交付闭环，不是搜索维度。Q50.A–Q50.K分别提供可被共同搜索消费的一种candidate mechanism、actual
transformation和verifier；这些队列项都不选择局部winner。Q51仍是唯一whole-DAG选择owner，不得重新引入layout、
fusion、communication、buffering或worker独立selector。

## Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  GSPMD完成card级分区且target-independent canonicalization完成后的card-local structured TensorProgram；
  current SSA、structured semantics、IndexRelation、effect、type、shape和dtype均可验证，尚未绑定physical Tile。
- Current stage responsibility:
  先把合法算法实现改写为真实structured DAG，再由唯一whole-card owner联合选择spatial mapping、temporal tile、
  multi-op fusion、layout/storage、SPM residency、buffering、NoC/DDR/compute资源时序、worker/order和completion；
  只把shortlist物化为actual IR，并由late exact gates决定最终合法性。
- Output artifact / IR:
  selected wafer.card.program及其中按physical tile_id区分的wafer.tile.program；随后投影为各Tile actual module。
- Downstream consumer:
  TileRegion-to-Instr conversion、fresh completion reconstruction、fixed-capacity SPM/DDR planning、
  transport/resource/ABI verification、target conversion、package publication和runtime launch。
- User-level driver / named pipeline:
  wafer-compile source-to-package pipeline；public optimization policy只保留`search`与`none`。
- Explicit non-goals:
  不重做跨card GSPMD；不建立第二搜索owner、shadow plan或late repair selector；不从名字、shape或workload恢复语义；
  不把board结果写回IR；本计划不包含Q48 semantic superoptimization。
- Completion gate:
  Q49闭合baseline；Q50.A–Q50.K分别闭合一种候选机制；Q51、Q52依次闭合联合搜索正确性和实际负载scalability；Q53为通用DAG、HF prefill、
  functional decode和Llama block生成FP16/BF16 fresh package、oracle、runner并通过no-card后达到board-ready；
  真实matched板端A/B获得可重复改善后才done。
```

## 队列拆分

| 队列项 | 唯一职责 | 产出 | 独立完成门禁 |
| --- | --- | --- | --- |
| Q49 | 收口当前实现并固定同路径baseline | 可编译的current pipeline和accepted baseline package | baseline fresh build/package/no-card通过 |
| Q50.A | 精确edge relation与logical demand correctness | layout-independent exact demand plan | relation image、ownership coverage与target lowering严格分层 |
| Q50.B | Attention/decode语义证明 | typed SSA proof及合法算法集合 | 普通图、functional decode和near-miss通过 |
| Q50.C | online recurrence DAG变换 | isolated clone中的真实online structured DAG | recurrence与observable结果验证通过 |
| Q50.D | partitioned-K/V DAG变换 | 真实partition/local-reduction/merge DAG | coverage、merge和near-miss测试通过 |
| Q50.E | physical layout与representation | layout/encoding候选、actual movement及fanout共享 | current actual-IR witness与正负测试通过 |
| Q50.F | value residency与recompute | resident/refetch/spill/recompute actual IR | lifetime/capacity/alias gate通过 |
| Q50.G | peer与collective通信 | peer/collective participant、payload、transport和join IR | coverage/completion正负测试通过 |
| Q50.H | ready-order scheduling | dependency/effect/resource合法的per-Tile Instr顺序 | branch/fanin/fanout/effect测试通过 |
| Q50.I | multi-buffer流水机制 | fixed-slot、rotating buffer与Direct-DTE issue/wait IR | slot lifetime/capacity/hazard测试通过 |
| Q50.J | worker与completion | typed worker placement及跨worker join/completion IR | multi-worker actual witness与负例通过 |
| Q50.K | 真正fusion traversal | 共同TileRegion/consumer-driven recursive traversal | fusion与独立traversal均可实际物化 |
| Q51 | 闭合唯一联合搜索及search-policy cutover | accepted whole-DAG winner actual IR | 小图最优oracle、exact gate和有效融合通过 |
| Q52 | 按实际负载优化搜索规模和时间 | 有质量守护的memo/DP/pruning/solver或启发式实现 | 代表负载在受管时间内完成且不劣于baseline |
| Q53 | 形成current production board-ready证据 | fresh package、oracle、runner、no-card和融合有效性证据 | 无卡阶段board-ready；真实板测后done |

同一时刻只执行一个队列项。后项可以提前记录需求，但不得在前项未闭合时把其实现、测试或完成声明混入当前提交。

## 当前工作树与已有代码处置

Q49开始时先确认当前checkout不存在并发writer；存在重叠修改时先停止并明确变更owner。随后只针对
whole-card相关diff建立文件与能力归属，不把当前大范围dirty worktree当作可整体回滚或整体提交的单元。

现有实现按以下规则处理：

1. `WholeCardExecutableSynthesis`、CardProgram/TileProgram IR、MPMD projection、TileRegion-to-Instr、
   `PhysicalTileFinalization`、SPM/DDR planning、whole-card admission、package和runtime是current trunk，保留并修通；
   不创建第二条baseline或第二套lowering。
2. 当前新增的whole-DAG state、relation、recursive tile-and-fuse、NoC和packing代码先保留；是否重构只由对应任务的
   fresh test和职责边界决定，不因文件较大直接重写。
3. tracked删除分为“旧owner外壳”和“仍承载未迁移能力”两类。前者最终可以删除；后者在current替代实现及测试
   闭合前必须保留或从HEAD选择性恢复。禁止批量checkout恢复，也禁止继续按目录批量删除。
4. 旧实现只作为算法、actual transformation、typed verifier和测试来源；不得恢复为search selector或兼容旁路。
5. 每个删除项必须能指向current实现位置、actual-IR witness和实际执行的替代测试；缺任一项即不满足删除门禁。

## Q49：Current Baseline Stabilization

### 责任边界

Q49只回答“当前已有实现能否稳定地产生一个真实合法whole-card executable”。它不证明`search`策略的搜索质量，
不迁移全部旧能力，也不做model-scale搜索优化。

### 实施

1. 固定当前入口和真实链路：

   ```text
   lowerTensorProgramToCardProgram
   -> projectCardProgramToPhysicalTileModules
   -> convertTileRegionToInstrModule
   -> finalizePhysicalTileModule
   -> admitWholeCardExecutable
   -> package
   ```

2. `OptimizationConfig::none()`由独立的deterministic baseline controller拥有：它不创建candidate frontier、winner、
   Pareto排序、feedback beam或edge-action transition。baseline与`search`只共享CardProgram materialization、physical
   Tile projection、Instr lowering、completion、SPM/DDR和admission，不允许`none`走legacy compatibility或复制第二套
   lowering。
3. baseline固定使用最大合法physical Tile参与集合；每个structured op独立保留完整iterator temporal tile向量，所有
   Linalg op间边界统一显式落入compiler-owned DDR，`actual_fused_edges`必须为零，buffer count固定为1。同一physical
   Tile拥有producer/consumer shard时直接使用DDR RegionCut；两个确定性spatial shard集合不一致时，只按typed indexing
   relation物化完成该依赖所必需的exact peer fragments，并在consumer端DDR assembly后开始独立op stage。baseline不搜索
   edge action、route、layout、residency、recompute或overlap；这些可选维度只由Q50.A–Q50.K提供机制并由Q51联合选择，baseline默认值不成为
   `search`搜索空间限制。
4. baseline从current spatial mapping产生的完整per-Tile local extent开始。actual fixed-capacity SPM packing失败后，
   allocator返回带source lineage的冲突demand；controller只按有限breakpoint缩小该Linalg op的parallel/reduction
   iterator tile，并使其DDR load/store只物化当前temporal window，然后重新执行完整exact gate，直到单一baseline状态
   actual合法。一个反馈轮次可以合成同一allocator conflict set中互不冲突的per-op缩减，但不得分支为候选、改变edge
   action、启用fusion/peer/layout/buffering，allocator也不得在accepted executable上私下repair。
5. 对普通多op chain、spatially sharded compute和cross-Tile dataflow运行同一source-to-package链路；验证
   CardProgram coverage、physical Tile projection、fresh completion、fixed-capacity SPM/DDR和package readback。
6. 将`WholeCardExecutableSynthesis`内部职责过大的部分只做行为保持型抽取；每次抽取前后复用同一baseline test，
   不借代码整理改变候选语义。
7. official HF attention prefill、两步functional KV-cache decode和Llama block使用原始source DAG运行FP16 `none`
   路径；不要求先识别或
   materialize Flash算法。对应search-policy融合与model-scale搜索时间不属于Q49合法baseline结论。

### Gate

- fresh host build和定向unit/integration测试通过，相关测试没有unsupported或skip；
- 三类代表图均产生完整accepted package并fresh no-card；
- SPM legality来自materialized actual IR的lifetime、alignment、buffer和fixed-capacity packing；
- `none`结果稳定可复现，并可在Q51 `search`没有accepted winner时作为合法fallback；
- official HF prefill和functional KV-cache decode均有同源eager oracle、payload、完整package、no-card和可串行执行的
  board runner；无板阶段标`board-ready`，真实板端运行后才可标`done`；
- 当前whole-card相关文件均已归属到Q49、Q50.A–Q50.K或Q51–Q53之一，没有未解释的继续删除项；
- Q49独立提交且真实板端baseline通过后标`done`；Q50.A可在Q49达到`board-ready`后进入`doing`。

长期回归owner为`test/Tools/wafer-compile-whole-card-baseline.test`：它从FP16 StableHLO source分别构造普通
multi-op chain、16-Tile GEMM和transpose前后均为observable的16-Tile dataflow，检查none零融合、本地依赖零peer、
跨轴依赖只有exact required peer fragments、consumer端DDR stage、16-Tile coverage、完整schema-v8 package及fresh
no-card。Q50.A–Q50.K提供可选peer/retained/recompute/layout机制，Q51负责联合选优和route；Q49只保留当前spatial assignment要求的
确定性correctness communication，不把它当作搜索坐标。

### 2026-08-11失效checkpoint与重新开放原因

- 先前通用baseline lit、prefill和decode的fresh结果只能证明当时输入可通过，不能证明baseline controller边界正确。
- official HF FP16 prefill原始DAG `none` package/no-card fresh通过（25.60秒）；actual dump包含16份Tile dataflow、
  16份Instr和16份target LLVM，空间extent 64继续细分为4×1024 temporal slice，并具有fresh SPM/DDR offsets。
- official HF FP16 functional KV-cache decode原始DAG `none`两步package/no-card fresh通过（369.97秒）。
- Llama FP16 `none`只有一个初始proposal，却在首个SPM失败后进入公共allocation-feedback frontier，30分钟内累计
  数百次CardProgram物化并尝试`CoupledFusion` edge transition，最终被bounded deadline终止；因此先前
  `board-ready`结论撤销，Q49回到`doing`。
- Q49不运行search-policy workload或fusion/peer gate；`search`机制、融合、通信、质量和scalability证据只由Q50.A–Q50.K及Q51–Q53签发，
  不进入baseline状态判断。Q49重新达到board-ready前，prefill、decode、Llama必须在最终controller上fresh重跑。
- board CTest已为prefill和decode的`none` baseline准备同源FP16、固定seed、独立work-dir、repeat=3、单一resource lock的
  串行runner；但runner准备不能替代最终controller上的三类fresh package/no-card，因此当时Q49仍为`doing`。

### 2026-08-11最终board-ready checkpoint

- 最终增量构建完成`WaferUnitTests`与`wafer-compile`；定向运行whole-card synthesis、TensorProgram到CardProgram和
  长DDR RegionCut provenance共51个unit均通过，configured build-tree baseline lit也实际执行并通过，无skip或
  unsupported。
- official HF FP16 prefill原始DAG以`none`生成完整16-Tile package并fresh no-card通过，总CTest 31.46秒；唯一状态
  `spm_probe_attempts=2`、`buffer_count=1`、`actual_fused_edges=0`，selected executable rematerialization为零。
- functional KV-cache decode两个原始DAG step均以`none`生成完整16-Tile package并fresh no-card通过，总CTest
  337.78秒；每步均只有一个accepted baseline，实际SPM反馈只沿同一temporal状态推进，selected executable
  rematerialization均为零。
- Llama-2-7B block FP16原始DAG以`none`生成完整16-Tile package并fresh no-card通过，总CTest 1490.64秒
  （24.84分钟）；共24次SPM probe，其中22次在因果Tile失败后提前停止其余330个Tile exact work，最终
  `buffer_count=1`、`actual_fused_edges=0`、selected executable rematerialization为零。相对移除重复winner
  rematerialization前的32.37分钟结果，完整闭环减少约7.5分钟。
- baseline的跨轴peer只服务typed indexing relation要求的correctness fragment；它不恢复可选通信搜索或fusion。
  三类workload的source、eager oracle、payload、package与no-card runner完整，Q49因此标`board-ready`；本轮没有
  启动真实板端，故不能标`done`。融合有效性仍是Q51/Q53的独立门禁，Q49零融合是baseline合同而非融合结论。

## Q50.A–Q50.K：Capability Mechanism Migration

### 共同边界

原Q50不再作为一个有状态、需一次性完成的队列项；它只保留为Q50.A–Q50.K的任务族名称。每个子项只迁移一种
compiler mechanism，独立同步设计、实现、actual-IR witness、正负测试和提交，完成后即可标`done`。后项只能消费
前项的current artifact，不能为赶进度恢复旧selector、side table或compatibility path。

每项均遵守同一施工顺序：先识别旧实现中应保留的算法、transformation、verifier和测试，再剥离旧独立owner；随后
在current IR上提供query-local analysis、candidate provider或actual materializer，并把结果直接写入isolated clone的
structured/Card/Tile/Instr IR。只有current正负测试和actual witness等价或更强后，旧入口才满足删除门禁。

Q50.A–Q50.K都不选择局部winner，也不对Q51搜索域设置固定tile、layout、fusion、buffer、worker或算法上限。
Q51必须同时消费这些机制并作唯一whole-DAG选择。

### Q50.A：Exact Edge Relation and Logical Demand Correctness

- 输入：已给定spatial placement的current structured producer/consumer SSA edge及其indexing semantics。placement只给出
  每个node的shard维、participant和physical Tile ownership；它不附带layout、encoding、buffer或movement action。
- 职责：只从structured semantics导出query-local `IndexRelation`，以relation image求consumer shard需要的all-and-only
  producer logical demand，并证明producer shard ownership覆盖该集合。
- 输出：`WholeDAGEdgeDemandPlan`。每个destination Tile记录consumer logical domain、exact producer demand及各producer
  Tile拥有的logical domain；不记录`MemLayout`、physical offset/bytes、local/remote action、route、buffer、send/recv或fusion。
- 下游边界：当前CardProgram仍只接受稠密矩形fragment，因此通过显式compatibility lowering把exact demand与ownership求交，
  再派生layout、`LocalShardResidency`/peer fragment和bytes。stride/multi-piece在logical stage合法；旧carrier不能表达时只在该
  lowering失败。Q50.E迁移layout/representation，Q50.F迁移residency，Q50.G迁移communication，Q50.K迁移真正fusion。
- Gate：reduction、broadcast/window、stride及ownership coverage均有正负测试；placement transition直接消费demand planner，
  不经过strategy/layout；canonical compatibility lowering和现有actual Card/Tile IR纵向测试继续证明dense local/peer路径，
  baseline保持逐op DDR boundary和零fusion。
- 非目标：不选spatial、temporal、layout、representation、residency、route、buffer或fusion，不处理placement frontier性能。

### Q50.B：Attention and Decode Semantic Proof

- 输入：target-independent canonicalization后的typed structured SSA DAG。
- 职责：证明普通Attention或functional K/V-cache decode语义并导出合法算法集合；不改图。
- 输出：query-local semantic proof及允许进入后续materializer的普通、online、partitioned-K/V候选种类。
- Gate：普通Attention、functional decode和near-miss正负测试通过；不依赖op/tensor名、shape、参数位置、
  `Q length == 1`或外部mode。

### Q50.C：Online-Recurrence DAG Materialization

- 输入：Q50.B证明允许online实现的semantic candidate。
- 职责：在isolated clone中实际构造online max/sum/output recurrence structured DAG。
- 输出：可直接进入physical search的真实DAG，而不是算法名或未物化参数向量。
- Gate：recurrence、mask/effect和observable result verifier与正负测试闭合；`keyValueTileSize`只表示K/V reduction
  window，SPM最终是否合法留给actual packing，`128`等值只可作为遍历prior。

### Q50.D：Partitioned-K/V DAG Materialization

- 输入：Q50.B证明为functional decode且允许split-K/V的semantic candidate。
- 职责：在isolated clone中实际构造partition、local reduction和merge DAG。
- 输出：带真实partition/merge SSA dataflow的structured DAG。
- Gate：partition coverage、merge数学语义、effect/observable result和near-miss测试通过；`splitCount`仅为候选参数，
  不等于KV长度、SPM容量或allocation结果。

### Q50.E：Physical Layout and Representation Mechanisms

- 输入：current structured value、typed movement relation和target representation facts。
- 职责：迁移layout assignment、physical encoding legality、movement transformation和fanout版本共享。
- 输出：可由Q51组合的layout/encoding候选及其actual movement IR。
- Gate：每种current候选均有actual-IR witness、fanout共享正例、encoding/movement负例；不恢复layout独立selector。

### Q50.F：Value Residency and Recompute Mechanisms

- 输入：已给定placement/layout的value及current lifetime、capacity和alias facts。
- 职责：迁移resident、refetch、spill和recompute的actual transformation及verifier。
- 输出：显式local SPM或DDR value lifetime与storage action IR。
- Gate：actual witness及capacity、alias、premature release负例通过；不选择通信route、buffer数或fusion partition。

### Q50.G：Peer and Collective Communication Mechanisms

- 输入：Q50.A exact demand、已给定placement/layout/residency和typed topology。
- 职责：迁移peer与collective movement，包括all-gather、all-reduce、reduce-scatter的participant、payload、transport和join。
- 输出：actual send/recv/collective及completion IR。
- Gate：local、peer与collective正例，以及coverage、mismatch、missing join和premature completion负例通过；不选route优劣。

### Q50.H：Ready-Order Scheduling Mechanisms

- 输入：已物化Tile-level work的SSA dependency、effect和resource约束。
- 职责：形成合法per-Tile ready-order instruction sequence。
- 输出：具有显式依赖顺序的actual Tile/Instr IR。
- Gate：独立分支、fanin/fanout和effect hazard正负测试通过；不决定buffer数、worker placement或whole-DAG winner。

### Q50.I：Multi-Buffer Pipeline Mechanisms

- 输入：Q50.H合法ordered instruction work及current lifetime/resource facts。
- 职责：迁移fixed-slot prologue/steady/epilogue、rotating allocation、multi-buffer和Direct-DTE issue/wait。
- 输出：具有可验证slot lifetime和exact range hazard的serialized与overlapped Instr候选。
- Gate：actual Instr witness、capacity、alias、issue/wait和slot-reuse正负测试通过；不选择全图pipeline方案。

### Q50.J：Worker and Completion Mechanisms

- 输入：ordered或pipelined Instr work及typed worker/completion facts。
- 职责：迁移NCC worker placement、participant join和跨worker completion。
- 输出：可由Q51组合的actual multi-worker Instr候选。
- Gate：非零worker actual witness及missing join、premature reuse/completion负例通过；不恢复worker独立selector。

### Q50.K：Fusion Traversal Materialization

- 输入：structured producer/consumer edge、Q50.A exact relation及已给定tile/residency候选。
- 职责：迁移complete traversal、boundary movement和真正multi-op recursive tile-and-fuse。
- 输出：实际共同TileRegion或consumer-driven recursive traversal，以及可验证的中间值驻留。
- Gate：融合和独立traversal均可物化；融合正例不存在对应中间值的无意义DDR round-trip；单边action、group字段或
  `fused edge > 0`统计不能代替actual witness。Q50.K不选最终fusion partition。

Q50.A–Q50.K全部独立提交并标`done`后，Q51才可进入`doing`。若某个旧删除项仍无法指向上述某一current实现、
actual witness和替代测试，则对应子项不能完成；但它不再阻塞无关子项形成独立提交。

## Q51：Unified Whole-DAG Search Correctness

### 责任边界

Q51建立唯一选择owner并先证明搜索正确。它不以model-scale编译时间为首要目标；Q52在Q51的正确性oracle和
fallback之上优化性能，不改变合法集合与exact gate。

### 共同状态

一次query-local partial state共同携带：

```text
materialized semantic DAG variant
ready/running/completed op-wave classes
per-op spatial work domain and physical Tile placement
all-iterator temporal tile vector and traversal order
multi-op fusion partition and value residency
value MemLayout / physical encoding / storage action
buffer count, rotating slot and live SPM roots
peer/collective/DDR movement and pending completion
per-Tile compute/SPM, per-link NoC and card-shared DDR calendars
worker/order, observable obligations and current makespan
```

这些信息只能存在于一次candidate-selection invocation，允许失效和重算；winner的执行事实必须进入actual IR，
search state随后销毁。

### 搜索过程

1. Q50产生的每个materialized semantic DAG成为一个root。
2. 从current structured iterators、reduction semantics、IndexRelation、implementation geometry和topology惰性生成
   spatial、temporal、fusion、layout、storage、buffer与resource transition。
3. transition按依赖顺序补充partial assignment，但没有维度被永久冻结；后续约束失败必须回到共同parent，允许修改
   placement、tile、fusion、layout、buffer或schedule，不能调用late repair selector。
4. clone前只允许使用typed semantics、coverage、effect/alias、topology symmetry、已证明SPM lower bound和数值
   dominance剪枝。seed只改变顺序，不能删除合法候选。
5. shortlist在isolated whole-card clone中actual materialize，依次通过TileRegion-to-Instr、fresh completion、
   fixed-capacity SPM/DDR packing、communication/resource、ABI/admission和final recost。
6. exact failure形成作用域明确的constraint/no-good并返回共同frontier；不得在clone或allocator内修改原candidate。
7. accepted候选使用同一enabled cost terms与actual work排序；性能参数缺失只对整个cohort删除对应term，不产生
   performance `Unknown`或candidate-local零值。

### 联合域要求

- Spatial：搜索不同op使用不同Tile集合、intra-op shard、独立branch并行、partial co-location以及redistribution；
  available Tile均可进入空间，不能把GSPMD logical partition直接当physical Tile。
- Temporal：每个spatial mapping从完整per-Tile local extent开始；容量失败产生二分区间，并补tail/divisor、native
  geometry、layout padding、transaction、wave和buffer-count breakpoint。改变其它轴后重新判断容量，不能复用全局
  “最大可放下tile”。
- Fusion：搜索对象是多op group及其共同iteration/residency，而不是独立边上的布尔标记；fanout、共享producer、
  conversion、live range和最后consumer必须共同计算。maximal local fusion与cross-Tile operator pipeline同时存在。
- Layout/SPM：layout和physical encoding直接进入状态；SPM估算只用于排序或证明下界，最终合法性只来自actual packing。
- Buffer/overlap：single/double/triple buffer、slot rotation、prologue/steady/epilogue、DTE issue/wait与compute、NoC、DDR
  resource calendar共同选择；无独立buffering或overlap后处理。
- Communication/worker：local reuse、partial NoC、multicast、gather/reduction、spill/reload、worker/order和completion均由
  同一state选择并物化。

### Correctness oracle与search-policy cutover

1. 为有限小图实现不裁剪的完整枚举oracle，覆盖single-op、chain、independent branch、diamond/fanout、fanin/reduction、
   partial redistribution和不同tile shape。
2. `search`在这些小图上必须返回与oracle相同的最优accepted candidate和stable digest；serial/parallel
   evaluation结果一致。
3. 每个状态维度至少有一组对立候选测试，证明改变该维度会改变actual IR、legality或cost，而不是未消费字段。
4. `search`与`none`共享相同source、CardProgram projection和late exact pipeline；`search`无winner时只能返回Q49
   baseline，不能进入legacy path。
5. 旧rank==Tile、rank-local winner、late NoC/layout/buffering selector和performance Unknown只能在Q50 parity与本节gate
   均闭合后清理。

### Gate

- bounded小图完整枚举与`search`结果一致；
- generic mixed DAG产生合法whole-card actual winner；
- packing failure不原地retile/spill，actual failure可回到共同frontier继续搜索；
- ordinary Attention、functional decode的真实DAG均进入同一物理搜索；
- spatial、temporal、layout、multi-op fusion、SPM、buffer、NoC/DDR/compute overlap均存在实际参与选择的对立候选；
- selected IR至少包含一个有效多op fusion group，中间值实际resident且没有无意义DDR round-trip；
- public policy只剩`search|none`，不存在second selector或shadow plan；
- Q51独立提交后标`done`，Q52才进入`doing`。

## Q52：Workload-Driven Search Scalability

### 责任边界

Q52只在Q51正确搜索之上减少重复工作和组合爆炸。不得先写固定tile、fusion depth、buffer count、候选数量或
shape/op/name shortcut，再用编译变快证明其合理。

### Fresh profiling

在generic mixed DAG、HF prefill、functional decode和Llama representative load上记录：

- 各类transition产生、拒绝、去重和进入frontier的状态数；
- canonical state重复率、frontier宽度和peak live state；
- semantic、coverage、SPM lower-bound、actual packing、completion、ABI等失败原因；
- IR clone、materialization、Tile-to-Instr、SPM/DDR packing、cost和admission耗时；
- wall、RSS、work units、winner cost/digest、fusion group和movement变化。

先从结果判断热点和重复来源，再选择算法：

| 已证实热点 | 可采用方法 |
| --- | --- |
| 相同boundary/state反复出现 | canonical key、memoization、subgraph DP |
| 大量重复exact failure | 作用域明确的constraint/no-good cache |
| lower bound与winner差距足够 | dominance和branch-and-bound |
| 局部资源时序组合复杂 | 局部exact solver或DP，结果回到共同owner |
| DAG存在稳定separator | boundary-compatible composition/DP |
| 全局空间仍无法承受 | best-first/beam、遗传算法或模拟退火，并显式标记trade-off |

一般DAG默认使用complete lazy generation与best-first/Pareto ordering；任何bounded beam、候选cap或随机启发式都只能
在本任务的实际profiling后启用。启发式模式必须保留Q49合法fallback，并在Q51小图oracle上持续测量最优性损失。

### 时间与质量门禁

- 不设置60秒等预设停止线；10分钟以内可接受；超过10分钟必须形成热点归因并继续分析；
- 可运行到30分钟以收集完整状态和重复证据；超过30分钟后才基于已测热点选择明确trade-off；
- 优化前后在Q51小图上的最优winner不变；
- 代表负载的`search`结果不得劣于同源`none` baseline；
- 优化不能把合法tile域缩成seed集合，也不能把融合、layout、buffer或overlap维度变成固定默认值；
- compile work、wall、RSS、winner digest和质量差异可复现；
- Q52独立提交后标`done`，Q53才进入`doing`。

## Q53：Production Board Readiness

### Unit / property

- Card/Tile verifier、coverage、message matching和SPM ownership；
- semantic alternative eligibility和actual DAG materialization；
- ready-set concurrency、event transition、branch/fanin、wave pipeline和resource hazard；
- finite tile breakpoint、actual SPM feedback、cost term enable/disable、determinism和小图最优oracle。

### IR / integration

- generic GEMM、elementwise、reduction、conv/mixed DAG；
- distinct MPMD Tile programs、same-region不同tile shape、multi-op local fusion、operator pipeline与partial redistribution；
- multi-buffer issue/wait、worker/order、actual failure cleanup、fresh completion、SPM/DDR all-and-only roots和ABI/package readback。

### Source / package / no-card

- official HF prefill FP16/BF16；
- functional two-step KV-cache decode FP16/BF16；
- Llama-2 7B representative block FP16/BF16；
- source保持原始HF/PyTorch语义，不添加mask、`-inf`、shape或decode特判；
- 每个case提供source、case-owned CPU oracle、deterministic runner、current production package和fresh no-card；
- source、metadata、payload和expected的dtype/shape一致；host build/test按`nproc`并行。

### Fusion effectiveness gate

每个要求融合的代表case同时证明：

1. selected actual IR存在至少一个由共同iteration/residency物化的多op fusion group；
2. 融合中间值的actual lifetime和offset证明其驻留SPM；
3. 不存在对应中间值的无意义DDR spill/reload round-trip；
4. fused与同源fusion-disabled baseline数值一致；
5. theoretical/actual work或后续matched board A/B没有因小tile、额外movement或并行度损失而退化。

只统计`fused edge > 0`、只检查group字段或只观察pass成功均不满足本门禁。

### Board-ready与done

- 无板阶段完整生成全部current package、oracle和runner，并逐case fresh no-card后才把Q53标`board-ready`；
- 真实板端单进程串行执行，不读取、回放或重新判定历史输出，不自动retry/reset；
- Llama及至少一个prefill/decode代表执行同源matched `none`/`search` A/B并校验exact output/guard；
- 获得可重复实际性能改善后Q53才标`done`，同时解除Q48的前置阻塞。

## 提交与收尾

1. Q49、Q50.A–Q50.K、Q51、Q52、Q53各自使用独立可评审提交；不得等到最后把全部dirty diff一次提交。
2. 每项提交前同步`tasks/progress.md`对应row；只有当前项gate全部fresh通过才转换状态并启动下一项。
3. 提交使用`Codex <codex@openai.com>`并附`Co-authored-by: hehesnail <shashen008he@gmail.com>`。
4. 终态同步01/03/04/06-09/11-13/16/18和受影响memory；稳定的新bug模式进入`memory/bugs.md`，可复用build/debug
   workflow进入`memory/general_dev.md`。动态测试数字、临时路径和未校准推测不进入长期设计。
