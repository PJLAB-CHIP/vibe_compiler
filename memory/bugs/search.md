# Bug模式：搜索、候选与成本

按主题记录可复用根因；入口见[bug索引](../bugs.md)。规则冲突时以当前编号设计与[AGENTS](../../AGENTS.md)为准。

## Optimization policy被接收但没有进入search owner

- 现象：`search`与`none`在所有workload上产生相同IR/package，新增搜索代码从未影响winner。
- 根因：driver解析了optimization policy，却在executable search边界丢弃或绕过它；测试反而把相同结果锁成合同。
- 修复模式：`search`与`none`是独立controller并拥有各自的actual IR materializer和candidate/attempt owner。Baseline
  从current TensorProgram和固定规则直接构造IR，不创建search structural choice state；search才消费explicit choice。两者只共享
  policy-free source facts、single-op rewrite/conversion和无policy SPM/DDR/target leaf，互不调用或fallback。
- 防复发：test-only call/work witness分别证明None的search-session为零、Search的baseline-controller为零；pre-structural
  frontier不构造IR，structural choice闭合后只产生一份candidate owner，winner不重建且publication一次。普通compile不创建统计对象。

## 多层C++ shadow plan不能代替future IR

- 现象：search在actual IR之前依次建立future value、movement、storage、event、execution structure和schedule record，
  最后由materializer重放成MLIR。一旦不对应，便继续增加ID mapping、expected inventory和plan/actual parity verifier。
- 根因：把“避免物化候选IR”误当成架构目标，混淆了transformation choice与只能由current IR表达或派生的
  operation、SSA、buffer、alias、lifetime、effect、order和completion事实。每个局部schema都能解决一个接线问题，
  但合起来形成与MLIR并行的第二编译器。
- 修复模式：search只保存尚未被消费的显式choice。Spatial/region/temporal choice闭合后立即在candidate-owned
  `IsolatedFromAbove` transaction中生成actual TileRegion IR。之后每个choice都作用于current IR，经verifier后使旧analysis失效；
  layout/bufferization、movement、execution structure、Instr scheduling/completion和memory planning只读当前stage IR；memory leaf只接受
  completion-closed Instr，不重建join/wait。Rejected owner销毁，Accepted owner不重建。
- 防复发：每个新plan字段先分类为“choice”或“物化后IR事实”。后者不得进入C++ cross-stage state。搜索源码和active docs
  必须保持future operation/value/buffer/event/schedule owner、rebuild/parity verifier和winner replay为零；每个stage用actual IR数量、
  SSA/effect/lifetime witness和直接下游验证，不用plan inventory代签。

## 在structural choice闭合前物化或重建candidate会导致时间与RSS失控

- 现象：model-scale图对未闭合structural choice就clone/lower，或者同一candidate IR先probe、再actual memory/target gate、winner再重建；大量actual owners同时
  存活，编译时间和RSS失控。
- 根因：没有明确pre-structural choice frontier与candidate-owned actual IR的materialization boundary，也没有move-only accepted owner。
- 修复模式：pre-structural state只运行typed choice/query；spatial/region/temporal闭合后物化一份candidate-owned TileRegion IR。
  后续choice作用于current IR；rejected/loser owner立即销毁，session只保留必要summary和一个retained incumbent，winner原样发布。
- 防复发：显式计数structural materialization、后stage transformation、actual gate和winner publication，断言同一choice/IR epoch不重建。
  不能用低candidate count掩盖winner rematerialization或16 Tile重复整图分析。

## Late exact failure不能触发隐藏repair

- 现象：SPM packing或ABI failure后，late pass自行缩tile、spill、改worker或切communication，selected IR与search cost不一致。
- 根因：allocator/finalizer被赋予了搜索职责，产生第二winner owner。
- 修复模式：actual gate只消费当前candidate IR并返回typed Accepted/rejection/failure；allocator/finalizer自身不得缩tile、spill、
  改worker或切communication。带完整witness的rejection可由外层当前policy controller消费，其它状态不得伪装成rejection。
- 防复发：failure injection锁定packing/ABI/target失败不在candidate IR内repair、不调用另一policy；repo scan禁止late
  retile/spill/replan selector和allocator fallback。

## 性能未知项不能阻塞或污染比较

- 现象：大量candidate因某个硬件参数未校准而不可比较，或把缺失参数按零/无穷大后选择出现偏置。
- 根因：把hard legality与performance knowledge混为一个状态，并允许candidate-local缺项。
- 修复模式：cohort开始时统一确定enabled terms；有实值用实值，否则用明确理论值，完全未知项从全部candidate删除。
- 防复发：逐项移除optional rate，排序仍确定；同一term不能只对部分candidate启用；hard capacity仍独立fail closed。

## Algorithm shortcut会缩窄通用search

- 现象：为某个Attention/decode或shape case加入名字/形状matcher、独立public selector/pass、opaque provider key或第二winner
  分支，异构DAG无法复用；另一种错误是完全禁止typed算法变换，使online/partition-merge DAG永远不可能出现。
- 根因：没有区分“由typed SSA证明并物化actual TensorProgram alternative”和“按模型名进入第二实现协议”。
- 修复模式：算法资格只从structured semantics、indexing relation、type、SSA/effect、loop-carried dataflow和numeric policy证明；
  每个算法/拓扑参数点先成为verifier-legal actual TensorProgram，再由同一physical-dataflow owner选择。名字、shape或独立
  selector不进入协议，physical tiling/layout/buffer也不能提前替代算法DAG选择。
- 防复发：源码检查禁止framework/model/function/value-name和固定shape matcher；generic mixed DAG与真实HF/Llama走同一public
  pipeline，并以正负witness证明typed semantic builder既会命中合法图，也会保持near-miss原图。

## Whole-program candidate budget不能耗在单edge fusion siblings上

- 现象：RegionDomain有1360→112 Region的graph-coherent proposal，但4个whole-program actualization slots被singleton和三个单edge
  siblings用完，最终winner只有1359个Region；测试仍因“非singleton”而错误通过。
- 根因：把partition lattice的breadth-first merge distance当成quality progress。大图在level-1已有大量siblings，而一次actual candidate需要完整
  Temporal、layout、movement、MiniMalloc和target leaf，有限budget不可能靠逐edge枚举达到有效融合深度。
- 修复模式：proposal只构造一条dynamic maximum-gain graph-coherent merge序列，按本次新增local exact payload/bindings排序并在merge后更新
  incident gain；固定whole-program slots只物化该序列的singleton、均匀merge-distance prefixes和coherent endpoint，不限制group root数，也不
  枚举partition lattice。所有snapshot仍经actual IR和MiniMalloc判定，capacity failure不跨prefix剪枝。
- 防复发：真实规模chain/fanout断言中间prefix同时合并多个Tile和多个semantic edges，proposal allowance不会饿死coherent endpoint；LLaMA完成
  不能再以Region减少1或“存在local binding”签发，必须由整图Region收缩、actual DDR store/load与Instr减少以及final objective共同证明。

## Source node与selected Instr不是一对一关系

- 现象：合法的dead init消除或producer/consumer融合后，final Instr没有某个source node的独立operation，cost plan把整个
  exact-verified candidate误判为source relation丢失。
- 根因：把query-local source identity当成一源节点一最终指令的持久映射，忽略合法elimination和fusion。
- 修复模式：observable terminal source relation保持强制；内部pure node只有在全部observable successor路径已被下游覆盖时才允许空
  phase；融合多个source node的operation必须由DAG证明唯一downstream consumer并只计一次，foreign/incomparable relation仍拒绝。
- 防复发：内部节点消除、dependent fusion、terminal relation丢失和ambiguous fusion正负例成对覆盖，且phase raw work与whole accepted
  IR保持精确守恒。

## Temporal search必须覆盖内部reduction iterator

- 现象：observable output tile很小，但上游projection或downstream contraction仍把完整K维weight计入单Tile SPM；继续缩output
  tile也无法降低驻留，decode/Llama等DAG在exact packing统一失败。
- 根因：candidate只记录function result的temporal shape，没有为每个structured op的iterator domain选择temporal tile，因而
  reduction K维被排除在搜索空间外。
- 修复模式：为每个current structured op记录覆盖全部parallel/reduction iterator的完整tile向量并生成有限breakpoint；
  reduction iterator用typed accumulator、prologue/steady `scf.for`/tail递归物化，叶子统一调用TilingInterface，再沿每个
  operand indexing map反推exact window。不得把某一个reduction轴另存成旁路chunk合同。
- 防复发：小output/大K matmul与多级projection正例证明full weight不resident；检查accumulator、chunk coverage、tail、actual loop
  cost multiplicity和source arithmetic op/dtype保持不变，不按模型名、shape或operand位置特判，也不引入numeric policy。

## Baseline显式op边界不能无条件作用于search coupled region

- 现象：为了让`none`逐op独立tiling而给每个consumer补DDR seal/reload后，search已选的coupled region也被同一逻辑
  强制切断，导致full weight reload、SPM溢出或最终零融合。
- 根因：candidate materialization没有区分independent baseline boundary与search selected actual group，把baseline策略写成了
  所有policy共享的结构改写。
- 修复模式：显式consumer boundary只在independent baseline生效；search从source SSA按selected node group直接构造共同region，
  以actual in-region use-def和emitted-node relation证明，不以edge action/proposal代签。

## Typed API存在不能证明算法已经迁移

- 现象：新实现增加Assignment、Domain、typed ID和direct apply后，旧placement/layout/movement/schedule owner及semantic tests被删除；
  轻量case仍可由first point通过，但cross-value constraint、PBQP reduction、movement elimination、NoC/alias proof或pipeline能力消失。
- 根因：把“旧local winner和clone owner必须退出”扩大成“其中算法无需迁移”，完成评审只检查新type、枚举或package成功，
  没有建立donor capability到current query、production caller和test witness的逐项对应。
- 修复模式：退役前列出旧owner的candidate construction、solver、proof、diagnostic和tests，逐项标成迁入current typed
  query/transition、由current IR明确淘汰或仍待处理。Layout等solver只返回typed proposal/bound，不clone/apply IR也不拥有winner；
  selected assignment只在policy-specific materializer中apply一次。
- 防复发：每项能力同时核对definition、direct consumer、production call graph、negative和actual downstream witness。
  “有domain”“测试绿”“package生成”“source未进CMake”都不能单独证明迁移完成；旧接口可以删除，未迁算法不能被改名为dormant。

## Baseline既不能复用search，也不能退化成fixed-assignment validator

- 现象：一种实现只在search loop前提前返回且报告零fusion，却仍复用search domain、group materializer和first-choice solver；
  另一种实现为避免耦合，只接受预选physical assignment并验证一次，正常上游IR遇到SPM超限时无人继续功能合法化。
- 根因：把“禁止性能候选选择”与“禁止确定性功能合法化”混淆，也把零fusion当成controller、region和resource owner隔离证明。
- 修复模式：baseline从未绑定physical choice的TensorProgram进入，始终只有一个live deterministic candidate。它直接构造
  per-root region、deterministic representation/movement/buffer/order/completion；同Tile多root为多个顺序region，跨root shaped
  dependency显式materialize。每个candidate actualize一次，只有带current owner relation的actual SPM capacity rejection可推进
  预定义、单调、不分支且不回溯的smaller temporal successor。Accepted owner直接move返回，不执行winner protocol、不重建，
  也不运行无consumer的schedule/duration或默认trace。
- 防复发：call graph证明baseline不include/call search domain、preparation或materializer；测试同时检查一root一region、跨root
  carrier、initial-overfull-to-fit、minimum仍超限typed failure、candidate/actual gate一一对应、accepted rematerialization和默认trace为0。

## Baseline不搜索通信方案不等于所有依赖零peer

- 现象：为落实逐op DDR baseline而把所有peer fragment禁止后，带reshape/transpose和多种spatial轴的Llama DAG在16-Tile
  placement CSP中立即无解；若继续宣称可由共享内部DDR替代，又与current ABI中workspace为Tile-scoped的事实冲突。
- 根因：把“baseline不枚举可选edge action/route”误写成“任何跨Tile correctness communication都不存在”。不同op的确定性
  spatial shard集合可能不一致，同一Tile的private workspace也不能冒充card-shared中间buffer。
- 修复模式：本地依赖使用compiler-owned DDR RegionCut；跨轴依赖只按typed indexing relation物化唯一required peer fragments，
  destination先在compiler-owned DDR assembly，再进入独立consumer stage。可选peer、explicit replica、layout/movement choice和route优化
  仍只由统一search拥有；choice被选中后必须立即成为current operation/SSA/movement，不保存retain/recompute旁路状态。
- 防复发：baseline回归同时覆盖本地multi-op chain的零peer和observable transpose的exact send/recv/wait；两者都必须16-Tile、
  buffer=1、actual fusion为零并通过package/no-card。

## 大型RegionCut链不能按edge重复扫描完整候选

- 现象：106-node/120-edge独立DDR baseline中，单Tile的`split-ddr-stages`约耗时66秒，16-Tile初次物化和每轮SPM反馈被该
  transformation反复放大。
- 根因：每个cut都重新遍历完整module找marker allocation，每个候选allocation又扫描整个region分类prefix/suffix uses，split还
  clone大型body；verifier对长DDR provenance链重复递归。
- 修复模式：一次遍历建立marker到cut及直接spill allocation索引，一次region walk分类全部eligible DDR allocation；按cut移动
  operation而不是clone整段body；verifier使用query-local provenance memo。不得把索引或memo写入IR。
- 防复发：长RegionCut chain测试验证最终IR和provenance，并在模型规模compile timing中检查单Tile split不再随edge数乘法增长。

## 16 Tile不能把root公共分析和整图clone机械重复16次

- 现象：structured roots较多时，旧baseline按root shard、Tile entry、完整TileModule set三轮构造；仅root阶段就接近
  `root数 × 16 Tile`次TensorProgram conversion。即使16个worker并发，CPU work、RSS和诊断仍是同一工作被放大16倍。
- 根因：把Tile差异（offset/tail/peer endpoint）和root不变量（support relation、consumer access和structured identity）放在同一个
  per-Tile materializer里；materializer从output/edge endpoint无条件回溯SSA closure并clone scratch function，而不是从全部root
  execution domain一次性求exact operand demand。并发只缩短wall time，没有消除重复分析或过宽物化。
- 修复模式：每个candidate在immutable source上同时seed全部`(root, Tile)`，以`(value, Tile)`合并exact domain并按反向SSA拓扑传播；
  relation按operation/result/operand建立一次，structured producer立即形成boundary demand。carrier coverage验证后，final region只
  对当前candidate一次性物化typed demand recipe要求的operation和endpoint，不建立公共SSA closure或materialized-IR cache。16个Tile实际构造可
  bounded并发并按Tile ID稳定归并，但并发不是work消重机制。
- 防复发：显式test work counts检查relation construction、非空value/Tile demand、physical fragment和candidate Tile entry；每个
  candidate TileModule set/actual memory/target gate各一次，winner不重建。测试必须包含16 Tile demand不同的fanin/fanout，证明不是16次完整DAG walk。
  Pre-structural frontier只共享immutable analysis，不能按candidate/Tile缓存actual IR。

## Per-strategy whole-root walk会把materialization验证放大成高阶工作

- 现象：chain和GEMM很快，但16-Tile transpose/fanout在materialization verification中长期占满CPU；采样栈持续位于
  root membership、buffer relation和storage-root递归，说明是高阶重复工作而非单点死循环。
- 根因：对每个strategy、operation和buffer relation从零遍历完整Tile root并重建visited set，fragment数量、Tile数和IR
  膨胀相乘。非默认重型gate长期未执行又掩盖了回归。
- 修复模式：先用stage timing和中断采样定位首次重复walk，再把同一immutable IR epoch的storage-root或relation查询改为
  request-local memo/反向索引；memo value必须有稳定owner，不能返回会因map/vector扩容悬空的引用。更优先的是删除上游
  重复materialization，不能仅用cache掩盖错误construction。
- 防复发：真实规模fanout/transpose记录relation construction、非空fragment、root walk和candidate次数；受影响非默认gate
  必须实际执行并带合理timeout。

## root/component裁剪必须按本Tile拥有的edge endpoint保留策略

- 现象：多输出CROSS或两个独立consumer在同一Tile时，per-component materializer删除remote incoming PeerFragments，ordinary
  output traversal回退为本地融合remote producer；另一路中独立consumer已经seal到DDR，却在outgoing peer查询时被重新计算。
- 根因：edge过滤错误要求producer、consumer两个structured node都属于当前Tile component；remote producer按定义不在当前Tile
  root集合。独立stage又只在已有global materialization cache entry时更新value，cache miss时没有插入sealed result。
- 修复模式：策略只要当前component实际拥有producer endpoint或consumer endpoint之一就保留；独立consumer stage完成后对共同
  `materialized` cache执行insert-or-update，子窗口从sealed DDR view派生。source-only destination traversal还必须传递当前clone的
  structured node mapping，否则current structural materializer无法形成result-buffer relation。
- 防复发：用多输出transpose→consumer的16-Tile source-to-package/no-card gate，同时由内部postcondition拒绝zero-root/multi-root；
  单纯的小型single-output fixture不足以覆盖remote endpoint裁剪和outgoing cache复用。

## Stage pipeline不能再拥有一套whole-Module candidate/clone入口

- 现象：旧fixed-slot实现扫描任意loop、clone完整Module、自行推导slot/stage并返回一个local candidate；buffer count、logical edge、
  ready order和pipeline identity分散，后续只能靠ordinal/clone对应关系拼回search。
- 根因：slot lifetime与stage event structure没有以同一selected edge scope为边界，rollback又被误写成每个mechanism各clone一次。
- 修复模式：结构choice后立即构造candidate-owned TileRegion IR。Pipelined alternative在最近`IsolatedFromAbove`
  scope上物化actual SCF phases、movement、rotating buffers和SSA slot relation，然后从该IR重算event、lifetime、completion和
  MiniMalloc demand。不先建initial storage/EventGraph/execution-structure/storage/schedule链，也不等winner后再重放结构。
- 防复发：旧public API和whole-Module clone可以删除；旧source/test中的periodic DTE、NCC backedge、endpoint reuse、alias/external root、
  odd tail和atomic failure能力必须逐项迁入current owner并受测后才能删除。stage还必须成为search typed transition，而不是由nonempty
  buffering scope自动触发。2个wrapper test和少量selected-buffer test不能代签旧40项semantic witness。

## Candidate admission与target ABI preparation必须共用boundary verifier

- 现象：search保留的`DeviceExecutable`已经通过memory/resource gate，最终target codegen才发现某个Tile entry的program output binding落在
  zero-result function上，整次compile在已有合法incumbent后失败。
- 根因：candidate admission只检查binding数量一致，target ABI preparation另行检查argument/result与explicit DDR binding的exact coverage；
  两个边界使用了不同合法性实现。结构不完整候选还被当成可由其它Temporal size修复，重复运行相同失败。
- 修复模式：抽取唯一只读program/DDR function-boundary verifier，在`TileExecutable`构造前与target ABI preparation共同调用。该failure携带
  typed `StructuralChoiceInvariant` scope，search只拒绝当前structural choice，不解析diagnostic、不重试其Temporal domain，也不影响已接受的
  sibling incumbent。Rejected candidate验证不向成功compile发射error diagnostic。
- 防复发：单op多Spatial choice覆盖合法incumbent加不完整siblings，断言最终package仍成功、每个invariant sibling只actualize一次；直接lowering
  负例必须在`deviceExecutablesProduced`增加前返回`ProgramResourceBindings`，final target ABI保留同一verifier的独立负例。

## Baseline不能无条件构造search schedule domain

- 现象：把schedule planning canonical query/apply直接放入共同`planTileMemory`后，baseline scalar case从约2秒退化到45秒；大block会承担O(n²)
  dependency DAG构造，即使用户选择`none`。
- 根因：共享exact gate与共享search policy混淆；baseline的canonical source order/worker0已经确定，不需要枚举或建立schedule domain。
- 修复模式：Tile memory planning不拥有schedule选择，也不接受“是否apply search schedule”的布尔开关。Baseline和
  search各自先产生current Instr；baseline在该IR上应用deterministic order/worker，search在该IR上运行自己的scheduler。
  两者都从应用后的current Instr fresh构造completion，然后共享SPM/DDR、transport和最终verification。
- 防复发：baseline定向wall-time与work count必须检查search scheduler调用为零；任何新search axis接入共同lowering时都只能
  提供当前transformation的显式choice，不能在无actual IR的路径中构造future domain后再取first。

## Search constructive proposal不能静默退回exact域第一点

- 现象：LLaMA的显式search一直只报告同一个90,177,536-byte SPM overflow；看似actual-feedback temporal无效，实际proposal在进入
  actual gate前失败，controller随后用budget 1评估了“全部node单Tile、full temporal”的exact first point。
- 根因：constructive各node独立取最大spatial factor后没有先闭合整卡exact demand；随后coupled domain按connected component追加group，
  node id交错时产生未排序assignment并被自己的`contains`拒绝。proposal failure又没有独立诊断字段。
- 修复模式：在公共participant ceiling上由大到小重建完整Card spatial assignment，每点先过exact demand；temporal successor只在前一
  complete candidate得到actual SPM rejection后建立。coupled first/repair assignment在可观察边界按Tile/node semantic key排序。
  显式profile summary分别返回proposal与actual failure；预算同时计partial work和actual evaluation。
- 防复发：测试同时覆盖interleaved disconnected components、large transposed weight和完整LLaMA；exact enumeration去重必须比较完整
  assignment，不能只比较spatial前缀。启发式proposal只能改变访问顺序，不能删除exact sibling或在失败时冒充已评估candidate。

## Actual feedback必须绑定完整显式choice和current source epoch

- 现象：两个candidate的最后一个choice相同，但spatial/region/temporal/layout/movement choice不同；若controller只按
  最后choice reserve/cache，第一个actual rejection会把合法sibling当duplicate或forbidden。
- 根因：用未物化的最后plan object代替完整transformation choice identity和current source epoch；actual conflict root又被误当成
  可推广到prefix的no-good explanation。
- 修复模式：reservation和exact full-point cache使用完整显式choice key与immutable source identity；actual IR由该choice通过
  唯一materializer生成一次。SPM owner/conflict只随该exact rejection保存，没有证明时不做prefix subsumption。
- 防复发：用最后choice相同、上游choice不同的1025 sibling证明rejection只命中自身；逐choice identity、cache on/off
  和反序独立winner oracle同时执行。禁止按root、shape、bytes、最后一轴或parent pointer扩大no-good。

## Resumable search不能借用临时config或依赖遍历顺序填充cache

- 现象：one-shot search正常，但把同一遍历切成每次一个credit后，persistent session在actual gate读取到损坏的program boundary；
  另一个fresh session从完整choice生成candidate时暴露隐藏cache前置。
- 根因：resumable owner保存了调用表达式产生的`FrontendProgramVerificationResult`和`ExecutionConfig`引用，resume时临时值已经析构；
  actual evaluator又假定cache必然按固定顺序预热，显式choice和source本身不足以独立物化candidate。
- 修复模式：persistent session复制小型immutable config values，只借用明确由outer transaction持有的TensorProgram、diagnostics和
  ProgramData。任一完整choice均能通过唯一materializer独立产生candidate-owned current IR；cache只是request-local pure-query memo，
  不是隐藏前置。显式continuation stack在cutoff后原位resume，不重放或重建winner IR。
- 防复发：同一1024 source分别one-shot和每credit resume到同一first-accepted choice；fresh parse可直接物化前两个
  complete choices，且在pure-query cache on/off下IR/status一致。不得用延长timeout、保活临时对象或先跑warmup修补cache依赖。

## Driver-owned private IR不应为同一planning kernel反复构造临时PassManager

- 现象：大图baseline在多轮actual SPM rejection后首次进入DDR planning，固定在`PassManager`析构时报
  `double free or corruption`；GDB调用栈落在per-Tile DDR pass adapter，SPM rejection和candidate IR本身均已完成。
- 根因：compiler driver已经拥有独立、可丢弃的per-Tile `ModuleOp` transaction，却为每个Tile重新创建只包装一个DDR planning
  kernel的临时`PassManager`。这既没有提供跨pass analysis复用，也把pinned pass adapter的额外所有权/析构周期放进每个candidate的
  热路径。
- 修复模式：注册的named pipeline继续使用AnalysisManager-aware pass adapter；拥有private Module transaction的driver直接调用同一个
  typed query/apply kernel。kernel仍执行相同actual lifetime、capacity、range和offset算法，失败时由caller丢弃当前Module，不建立第二套
  planner或绕过verifier。
- 防复发：direct kernel与registered pass adapter对1024/1025级rank-3输入比较exact offsets和失败类别；真实16-Tile candidate证明
  DDR、target和package均实际到达。不得用禁用MLIR multithreading、固定单worker或猜测其它analysis线程安全来掩盖析构问题。

## 可选的不可物化relation不能把其它合法e-graph路径变成work limit

- 现象：简单inverse reshape、transpose和Compute测试均通过，但把它们串成`reshape→transpose→Compute→transpose`后，e-node和match
  很少，component仍以budget exhaustion保持原图；把iteration上限从8提高到32完全无效。
- 根因：Access composition无条件对`projected map ∘ general row-major reshape`调用通用Presburger compose。这个可选RHS通常没有单一
  Tensor/Linalg materialization form，却先触发relation solver的`ResourceExhausted`；callback把它正确翻译成WorkLimit后，整个request按
  合同销毁，掩盖了相邻expand/collapse先闭合identity再通过congruence暴露transpose的合法路径。
- 修复模式：relation service先按materialization class分派。identity、reshape∘reshape和projected∘projected走各自exact构造；没有当前
  materialization form的reshape/projected mixed pair直接返回typed Unsupported，不启动通用solver。真正执行的query达到work limit仍保持
  整个component不变，不能降级成Unsupported。
- 防复发：同时保留1024/1025/1031的reshape/broadcast/concat与elementwise/reduction/contraction连续链，以及4个以上Compute交替Access
  深链；记录budget、e-node、relation query、match和before/after op。发现低work图exhaustion时先按callback种类归因，不能直接提高budget。

## Fanout等价变换必须进入同一个multi-root e-graph request

- 现象：共享Access同时服务多个Compute或observable root时，single-root e-graph无法看到完整fanout；在egg外补all-users C++ rewrite后，
  一次pass内的两个owner会观察到不同use集合，出现第二次运行才闭合、producer复制或phase-order差异。
- 根因：把multi-use边界当成独立pattern问题，而不是同一pure component的多root等价提取问题；C++旁路和egg分别拥有部分等价规则。
- 修复模式：一次request导入ordered observable roots和共享SSA DAG，egg在同一e-graph中创建全部等价式，提取时按统一e-class choice
  hash-cons为一个共享输出DAG，全部roots一次preflight和原子替换。C++ importer/materializer只解释current MLIR，不再直接改写等价图。
- 防复发：1024/1025/1031矩阵覆盖2/15 roots、异构fanout、暂时DPS-init use、observable barrier和第二次运行byte-equivalent；检查旧
  all-users rewrite caller为零、input/output producer node各一次。不能用交替运行两个rewrite owner、重复产品pipeline或whole-graph clone
  掩盖phase order。

## E-graph commit scaffold必须在同一次request内产生和清除

- 现象：第一次normalization只在function return前增加identity `linalg.generic`，第二次运行才继续消除transpose/concat；pass不再
  byte-idempotent，StableHLO直返constant/concat也残留无意义compute。多输入Concat还可能只消除最后一个piece的Access。
- 根因：output commit scaffold在component extraction结束后才创建；同时component connectivity沿raw `insert_slice` operands/users，
  没把已验证的完整insert chain视为一个N-ary Concat semantic producer。
- 修复模式：临时DPS output closure在component收集前创建，同一调用结束前精确移除或由extraction消费，并从logical transform统计中
  排除。Validated Concat的component edge和internal-user判断使用其N-ary inputs与完整implementation chain，最终仍物化标准Tensor IR。
- 防复发：一次/两次pass输出逐byte一致；constant、nested concat、transpose→concat→elementwise/reduction长链同时覆盖；断言所有
  piece的Access一起消除、scaffold attr和identity wrapper输出为0，不能通过放宽FileCheck掩盖第二次运行才闭合。

## Planning identity不能通过映射表延长成current IR identity

- 现象：Spatial materialization返回`RegionExecutionId -> TileRegion`，后续Temporal stage发现一个Region含多个work，又准备增加
  `execution -> operation`映射；FD同时依赖empty Region shell和同一ID在后续回填attention state。映射越补越完整，但current IR仍不能独立说明
  哪个operation、state和merge真实存在。
- 根因：pre-materialization choice在actual rewrite后没有被消费，而是被当成跨stage operation identity。空shell和missing output进一步迫使
  下游按plan重建future work，形成一套与SSA/parent关系并行的事实源。
- 修复模式：产生actual IR的transformation在返回前消费全部planning identity；ordinary contribution/merge/output以及attention state
  contribution/merge直接物化。下一stage从live operation/interface建立query-local choice并立即apply；clone只使用该次`IRMapping`。
  Physical位置由typed parent op表达，参与者由SSA表达。
- 防复发：新增任何`plan ID -> operation/value/Tile`关系前，先检查producer能否直接物化缺失事实、consumer能否从current IR重算。
  如果答案是可以，删除mapping；如果确实需要跨stage保存，必须先有用户同意的authoritative typed IR，而不是扩充C++ side table。

## PBQP全局tie-break必须先按connected component分解

- 现象：16-Tile FA/FD layout factor graph的数值最优解很快得到，但为每个value variable固定字典序state并重求整个问题，单测从亚秒增长到
  约49秒并耗尽默认work budget；各Tile component实际上互不连接。
- 根因：R0/R1/R2只减少单次solve的图，却让semantic tie probe反复遍历所有disconnected component；one-state auxiliary hub还因degree大于2
  留在residual core。
- 修复模式：先按factor edge把问题确定性分解为connected components，共享一个checked work budget；每个component独立求numeric optimum与
  semantic-variable tie，再按原variable index组合assignment/cost。任意度数的一状态变量直接把incident edge cost传播到neighbor unary，
  auxiliary变量只确定性重建，不扩大外部semantic tie前缀。
- 防复发：flat oracle覆盖disconnected cost/assignment，独立高degree fixed hub验证任意度消元，auxiliary-prefix测试验证重复求解确定；
  16-Tile FA/FD纵向记录solver work和wall。不能用Top-k、跳过exact solve或提高timeout掩盖重复全图工作。

## E-graph barrier fallback必须形成互不重叠的拓扑request

- 现象：一个119-op component有早期barrier root和最终root，单一late anchor不能支配早期use。旧fallback为最终root重新纳入113个operation，
  既与早期slice重叠，又在LLaMA上耗尽e-node/match budget，导致weight transpose未吸收到matmul。
- 根因：fallback只按ancestor集合判断shared producer，忽略已分配slice和不在当前slice的ancestor user；一次DFS顺序不能保证downstream-closed。
- 修复模式：非法单锚点component按source-order fanout/root建立fallback roots；从后向前只纳入全部semantic users已在当前slice的operation，
  已签发operation不进入后续request；这些互不重叠request在同一pass内按拓扑顺序各执行一次。Output closure被原样剥离且没有保留任何
  Access/Concat/node变化时，结果仍报告`Unchanged`，不靠重复运行e-graph追认完成。
- 防复发：真实规模early `extract_slice` barrier与later transpose→contraction同图测试必须无budget exhaustion、保留barrier、消除transpose且
  第二次运行byte-equivalent；长链与multi-root共享DAG原有覆盖继续通过。

## 首个容量证据足以拒绝，但不足以高效指导多区域搜索

- 根因：唯一allocator遇到单独超大allocation或第一个超容量clique就返回，使同一actual conflict graph中已经存在的其它冲突
  只能在下次完整物化后逐个暴露。仅加预算、深化修复链或调整尺寸排序不能消除这种串行反馈。
- 修复模式：在原edge-clique cover遍历中收集已证明的超容量子集。按实际size降序继续收集不相交的超容量前缀，
  防止单个巨大allocation遮住同一clique中的其它冲突；不足的余项不算证据，不另行枚举clique或猜测未来allocation。
  返回actual allocation的确定性去重并集，并从其存活owner/input map派生反馈；并集bytes不是同时存活峰值。
- 防复发：固定问题穷举oracle与输入排列保持合法集合不变；rank3、1024/1025/1031的独立Region、超大allocation及无关allocation
  混合输入检查exact反馈和live owner，修正后的actual IR通过同一SPM gate。搜索另检验深层修复、原兄弟和普通探索均能获得服务。

## Dynamic e-graph callback memo不等于规则展开复用

- 根因：Searcher按e-class匹配，dynamic applier内部再遍历root/child的等价节点组合；relation callback即使已经memo，
  相同读取状态的每轮调用仍会复制和枚举全部组合。
- 修复边界：只在同一request/规则内比较完整typed read set，包括实际读取节点、canonical child及analysis facts。
  缓存调用前状态；自身新建等价式、child变化、union/rebuild或facts变化都必须重新执行。不得仅用节点数或hash判断相同。
- 防复发：同时检查真实skip、child先于parent的phase-order闭合、自身新等价式继续组合、独立root保留和并行request确定性。
  对相同预算的真实输入比较最终IR及原match/node/merge计数；新rule读取更深事实时同步扩展失效合同。

## 容量反馈的输入需求不能只沿单一Source追踪

- 根因：`tensor.insert_slice`同时有Source和Destination。只接受一个Source的查询会丢掉完整reader集合，
  使实际输入窗口的容量失败无法关联到其temporal参数；直接沿全部SSA operand追踪又会误计已覆盖的旧值。
- 修复边界：在不可变structural IR上携带exact需求区域反向查询，Source取与写入窗口的交集，Destination取差集。
  多层覆盖、slice和reshape共同传递需求，空分支停止；未覆盖的未知producer继续阻止不完整归因。
  这只是输入访问到搜索参数的关联，actual Instr的buffer来源及SPM合法性仍由独立证据决定。
- 防复发：检查部分/完整/重叠覆盖、不同输入与重复输入、未知分支保留/覆盖，并以真实RDMA/GS容量certificate验证all-and-only坐标。

## Region successor不能指数枚举必拒绝的external binding

- 根因：同一Region内的producer/consumer每个use仍枚举external取值，而`buildPlan`最终必拒绝；几十个view use足以让下一候选查询停滞。
- 修复边界：由当前partition和既有local-use合同构造有限取值域，同组保留required-local/replica，异组保留external/replica；最终验证不变。
- 防复发：有界穷举oracle核对合法集合与顺序；真实规模32个use、多Tile及尾部检查exact coverage、绑定和实际构造尝试数。
