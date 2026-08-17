# Card 内 Physical Dataflow 综合

状态：2026-08-17 按 current `TensorProgram -> CardModule -> TileRegion -> Instr -> CardExecutable`
主线收敛。本文是 card 内 spatial mapping、TileRegion formation、temporal tiling、融合、physical
representation、movement、buffering、instruction scheduling 与候选选择的唯一设计 owner。动态状态和施工顺序只看
`tasks/progress.md` 与 `tasks/plans/physical-dataflow-synthesis.md`。

旧 whole-rank、coordinated selector 和当前单体 synthesis 实现只提供可审计的历史机制或迁移素材，不构成本文合同。
现有实现能直接适配就复用，需要改变边界就改造，只有算法或 verifier 有价值就提取后替换，没有价值或无法适配则删除；
不为了保留旧文件而扭曲终态架构，也不在 replacement 完成前无证据删除仍需迁移的能力。

## 1. 核心结论

编译器优化对象是一个 card-local structured DAG 的完整物理数据流，而不是单个 op、单条 edge、预先切好的 16 个
logical rank 或所有 op 共用的 tile shape。一个候选必须共同决定：

1. structured iterator 如何切成 logical shards；
2. logical shards 如何放到 Tiles，reduction partial/merge 如何分布；
3. 每个 Tile 上哪些计算共享一个 `wafer.tile.region`；
4. region 内哪些 producer/consumer 形成 coupled traversal，哪些采用独立 traversal；
5. 每个 traversal 的完整 temporal tile vector、loop order 与 tail；
6. physical layout、local/peer/collective/DDR movement；
7. buffer recipe、ready order、worker、completion 与资源 overlap；
8. 是否形成跨 Tile stage pipeline，以及其 chunk、Tile groups、movement 和 multi-buffer；
9. observable output/effect 全部完成时的整卡 cost。

统一搜索只统一上述决策的 owner、回溯和最终选择，不把各机制实现集中到一个文件或一层目录。analysis 靠近其可解释的
IR，transformation 靠近被修改的 IR，conversion、memory planning 和 verification 保持独立。候选一旦物化，以当前 IR
为事实；搜索状态不能成为下游 shadow plan。

首选构造是：

```text
健康的 spatial mapping
  + 每个 Tile 上尽可能大的 TileRegion
  + region 内尽可能大的 coupled-traversal closure
  + 尽量大的高效 temporal tile
```

真实 SPM、fanout、重复计算、通信或并行性出现问题时，再按因果缩 temporal tile、解除 coupled edge、切 region、改变
layout/buffer、回退 spatial，或构造带 multi-buffer 的 stage pipeline。这里的“尽可能大”是候选生成和排序方向，
不是 legality shortcut；最终 winner 只由完整实际 IR、fixed-capacity planning 和统一 cost 比较决定。

## 2. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  GSPMD完成card级分区、target-independent normalization完成后的card-local TensorProgram；若存在算法级等价
  alternative，每个alternative已经在isolated clone中物化为真实structured DAG。SSA、structured iterator、
  indexing relation、effect、type、shape和dtype均可验证，尚未绑定Tile。
- Current stage responsibility:
  `none`从正常TensorProgram沿canonical feasibility fallback确定可执行的iteration partition、Tile placement、single-root
  TileRegion、temporal tile、representation/movement、single buffering、order和completion；`search`在同一可回溯选择过程中
  联合决定iteration partition、Tile placement、reduction distribution、TileRegion、traversal connection、temporal tile、
  physical representation、movement、buffering、stage pipeline、order、worker和completion。两者的完整选择都进入正常
  Card/TileRegion/Instr lowering和exact resource verification。
- Output IR / files:
  被选中的CardModule及其all-and-only Tile modules；其中TileRegion、loop、movement、Instr、SPM/DDR
  allocation、completion和physical endpoint均是实际IR或accepted resource事实。通过全部gate后形成CardExecutable。
- Downstream consumer:
  target conversion、device link、ExecutablePackage emission与runtime launch。
- User-level driver / named pipeline:
  wafer-compile source-to-package pipeline；public optimization policy只有`search`和`none`，不增加逐机制selector。
- Explicit non-goals:
  不重做跨card GSPMD；不要求真实大workload全局最优证明；不从op/workload/文件/shape名字恢复语义；不建立
  cycle-exact simulator；不让lowering、allocator、communication或completion私自repair候选；不发布search sidecar。
- Completion gate:
  `none`从未预选physical assignment的输入产生accepted CardExecutable，包含初始tile超SPM后缩至合法breakpoint的正例；
  small DAG由独立reference domain enumerator证明search合法域完整，再由production-mechanism flat exhaustive runner证明
  candidate set、剪枝和winner一致；真实workload在约定预算内返回通过完整编译与verification的
  best-known CardExecutable；selected IR证明spatial all-and-only coverage与有效region fusion。至少一个代表case还要证明
  coupled traversal、tile-sized intermediate和无无意义DDR往返；selected multi-buffer/stage-pipeline仅在对应候选获选时证明。
```

`optimization=none`是完整的功能基线：它从同一正常上游TensorProgram自行完成目标执行所必需的确定性placement、
single-root region、temporal tiling、canonical representation/movement、single buffering、order与completion合法化，形成一个
可直接发布的CardExecutable，而不是要求调用方先提供已经合法的fixed assignment。`optimization=search`再由本文唯一
candidate-selection owner管理多个性能选择。两者可以共享由current IR导出的immutable typed facts、single-root region
materializer、scoped exact probe以及后续lowering、memory planning、verification和package emission，但不能共享会携带
候选集合、group boundary、proposal order、score、backtracking或repair语义的search state/carrier/evaluator。baseline完成
唯一确定性可执行结构后才进入共同CardModule materialization和CardExecutable compilation；“提前跳出search loop”本身不构成
policy解耦，去掉search依赖也不能删掉使程序走通所必需的feasibility legalization。

## 3. 稳定 IR 与 output 边界

### 3.1 `wafer.card.module`

`wafer.card.module` 是一个 `card_id` 的完整 MPMD verifier 范围，拥有 card-local observable inputs/outputs、
shared DDR boundary、all-and-only Tile modules，以及跨 Tile message coverage/completion 验证范围。它不保存
candidate set、score、候选列表、route side table 或 target calibration。

### 3.2 `wafer.tile.module`

`wafer.tile.module` 绑定唯一 physical `tile_id`。不同 Tile 可以拥有不同 op、loop、temporal shape、worker 和执行长度。
实际顺序、并发和依赖由 body 中 loop、SSA、send/recv/wait 与 event 表达；不另存全局 schedule attr。SPM root 或
alias 不能跨 `tile.module` SSA 传递。

### 3.3 `wafer.tile.region`

`wafer.tile.region` 是一个 Tile 上的 SPM ownership/lifetime domain，不是硬件 Tile，也不是单个 loop 或
“fusion group”标签。一个 region 可以包含：

- consumer-driven coupled traversal；
- 多个不同 temporal shape 的独立 traversal；
- local physical conversion；
- selective spill 某个 root，而其它 roots 继续存活；
- region 内 movement、communication、sync/effect ordering。

SPM root 和 shaped alias 不能跨 TileRegion。不同 TileRegion 间 shaped data 必须显式 materialize 为 DDR
store/completion/load；不同 Tile 间则由 source SPM、NoC/DTE send、destination SPM staging、recv/wait 表达，
不能共享同一个 SPM root。

上述是IR的一般合法能力，不是`optimization=none`的默认分组策略。`none`中每个TileRegion最多拥有一个独立
structured compute root及其不可分割的non-root support closure；root cardinality按materialization relation映回current
structured DAG node计算，不按lower后的compute op数量计算。同一Tile可以按确定顺序拥有多个这样的TileRegion。将多个独立
compute root放入同一TileRegion即使没有coupled producer/consumer edge，也会共享SPM预算、lifetime/lowering scope和失败
归因，属于需要由`search`显式选择的region grouping，不能因后端legal或`actual_fused_edges == 0`而泄漏到baseline。

### 3.4 `CardExecutable` 与 `ExecutablePackage`

`CardExecutable` 是内存中已经通过 all-and-only Tile coverage、Instr、SPM/DDR、transport、resource、completion 和 ABI
verification 的执行对象。`ExecutablePackage` 是 target lowering、link 和 emission 后交给 runtime 的磁盘发布物。旧 C++
当前实现以 `CardExecutable` 表示完整的 Tile executable 集合，不再维护另一套容器层次。

## 4. Query-local problem 与 search state

### 4.1 问题定义

一次 candidate-selection invocation 可读取：

- current TensorProgram alternative 与稳定 DAG node/edge identity；
- physical topology 与 available Tiles；
- structured iterator、indexing relation、effect 与 numeric policy；
- 各层合法选择枚举和实际 materialization 能力；
- target geometry、capacity 和 cost 参数。

它不保存 winner，不保存 accepted offset，也不把算法名或 opaque parameter bag 传入 physical legality。

### 4.2 Search session、incumbent 与 candidate assignment

一次 search invocation 内的事实分为四类，不能再合并成一个 candidate bag：

1. **immutable session input**：borrowed TensorProgram root、transaction-owned ProgramData view、target facts、cost comparison cohort，以及本次可用的
   typed mechanism；mechanism顺序由唯一current driver静态组合，不是runtime registry、user option或candidate字段；
   `IREpoch`只证明query与trial属于同一immutable borrow，并保护query/cache生命周期，不进入
   candidate semantic key，也不充当IR mutation counter；
2. **session control**：candidate frontier、已经通过完整gate的move-only incumbent、global work/budget ledger和coverage/
   lower-bound evidence；这些事实不属于任一candidate；
3. **candidate assignment**：只保存当前已经实现且不能从current IR与其它选择重算的typed choices；
4. **derived query-local facts**：exact demand、ready/live set、lifetime、resource calendar、SPM high-water、cost estimate和
   lower bound，按IR borrow、target facts和相关assignment重算或失效。

Q49.P accepted baseline只作为session-level incumbent和fallback，不进入candidate frontier，不被重编码为search assignment，
也不把其canonical functional choices当作未施工轴的默认值。mechanism availability、proposal priority、work ledger、diagnostic
statistics和incumbent同样不进入candidate identity。

`ProgramDataHandoff`继续由外层compiler transaction唯一拥有，不复制进baseline或candidate。Q50.0 accepted result只引用已验证
program identity/range；落选actual executable析构不改变handoff，最终winner确定后才由Q59同一outer transaction把handoff随
唯一发布结果向下游移动。search不得为每个candidate写package或提交目录。

Q51终态的candidate assignment包含下列typed choices；Q51.Core不预声明尚未由对应Q50 mechanism实现、验证和消费的字段。
每个Q50 checkpoint在同一current aggregate中加入本轴的named typed field、ephemeral transition、canonical encoding和精确
失效关系，不能使用`any`、字符串tag、opaque payload、通用provider registry或placeholder optional field提前占位：

```text
TensorProgram alternative
StageId  -> members / Tile group
NodeId   -> iteration partition / physical placement / reduction role
RegionId -> members / traversal roots
EdgeId   -> coupled / independent / recompute / local-or-remote movement
LoopId   -> complete temporal tile vector / finite wave-loop nesting and order within selected traversal
ValueId  -> selected physical representation
TileId   -> buffer recipe / instruction order / worker / completion choice
```

typed transition只用于从parent原子构造child；child保存apply后的assignment，不保存transition history、proposal ordinal、
allocator feedback history或repair path。状态不保存 raw `Operation *`、estimated SPM、汇总 bytes、fragments、resource
calendar、makespan、actual offsets 或repair history。候选物化后以 IR 为唯一事实源。

### 4.3 失效依赖

不同机制不互相调用或维护局部 winner；跨层耦合只通过明确依赖失效表达：

```text
spatial改变
  -> local DAG / TileRegion / exact edge demand及所有下游选择失效

region、traversal connection或temporal改变
  -> lifetime / representation / movement / buffer / Instr / SPM结论失效

representation或movement改变
  -> buffer / order / resource / SPM结论失效

buffer、order、worker或completion改变
  -> lifetime / resource calendar / SPM packing失效
```

统一 selection owner只能清除并重新展开受影响的 choices，不能让 spatial analysis 内嵌 communication selector，也不能让
candidate compilation看到失败后自行换方案。

## 5. Spatial mapping

Spatial mapping 精确由三部分组成，而不是“选几个 Tile”。

### 5.1 Iteration-space partition

对 structured op 的迭代域 `D = (i0, i1, ..., in)`，选择 parallel/reduction iterator 的 spatial factors，并为每个
logical shard形成精确 domain。以 GEMM 为例：

```text
C[M,N] += A[M,K] * B[K,N]
P_M * P_N * P_K <= available Tile count
```

`P_M/P_N`产生output-parallel shards；`P_K`产生partial reductions。非整除维度使用精确 balanced interval或typed
relation表达tail，不能要求整除、丢元素或产生非法重叠。必须证明：

- parallel shards all-and-only覆盖原迭代域；
- 除selected recompute外没有重复执行；
- reduction partitions完整且存在显式合法merge；
- floating-point reduction order满足current numeric policy。

初期可先完整支持multi-axis block partition；未来若支持cyclic/block-cyclic，应扩展typed relation和lowering，不增加
shape/name matcher。

### 5.2 Tile placement

logical shard coordinate 与 physical `tile_id` 是不同对象。Placement决定 logical mesh 到available topology的embedding、
participant subset、partial/merge owner和独立branch的Tile groups。它考虑producer/consumer同Tile对齐、拓扑距离、负载平衡
和NoC热点，但只确定端点，不选择route、layout、buffer或transport implementation。

紧凑矩形、自然mesh、使用全部16 Tiles可以作为高优先级seed；它们不是legality条件。搜索域必须惰性包含合法的participant
count、多轴mesh、reduction partition、remainder和非对称placement。只有硬件拓扑和当前资源真正对称的状态才能
canonicalize。

高优先级proposal可以先做ordered factorized regular mapping：把一个或多个logical iterator factor映射到typed physical
topology dimensions。只有axis/factor nesting实际改变extensional partition relation、logical-mesh embedding或physical
placement时才形成不同typed choice；等价factorization的生成顺序必须canonicalize。它只是完整placement域的构造顺序，
不能把连续矩形、二维mesh、full occupancy、偶数participant或某一种factor order提升成合法性；未映射的local extent继续交给
temporal选择，不能由spatial proposal暗中固定tile或loop order。

### 5.3 Exact producer/consumer demand

不同op可以选择不同mapping。exact-demand query消费的是一次trial中已经显式形成的完整logical shard assignment：consumer
all-iterator execution domain、result/update domain、producer ownership domains、logical shard-to-Tile binding，以及
parallel/reduction、unique/replicated/partial-contribution role。它不能从result axis、单个`shardDimension`、participant count
或balanced一维矩形反推这些事实。对每个consumer logical shard：

```text
consumer complete logical iteration/contribution domain
  -> IndexRelation.image
  -> exact producer logical demand
  -> 与每个producer Tile ownership求交
  -> exact coverage / uncovered witness
```

dependency descriptor从current SSA use-def、producer result、consumer operand、DPS operand role、structured indexing
semantics和pure support graph派生，不从op名、shape或后续lowering猜测：

- 普通data input应用consumer已选完整iteration domain到operand indexing relation；
- reduction保留partial contribution domain、combiner/numeric约束与merge role；只看result shard会丢掉reduction iterator，
  也会把必须全部参与的partial owners误写成可互换replica；
- 显式structured Fill或其它DPS init producer是独立root，其init/update dependency必须有exact demand；consumer lowering以后
  负责init不能成为省略edge的理由；
- view、reshape、slice、pad等没有独立structured root的pure support graph组合typed relation；多operand support graph对每个
  data-carrying predecessor分别保留dependency，不假设unary chain；
- multi-result、fanout和fanin按实际producer result与consumer operand分别建relation，不固定`result(0)`、单init或equal shape。

relation image与producer ownership intersection都保留为exact logical set。intersection的union必须覆盖all-and-only demand；
非空uncovered set才形成当前trial的proven logical infeasibility witness。显式replication产生多个eligible owners时query保留
等价ownership而不选择source；partial reduction owner是不同必要contribution，不能去重。logical query不选择local/remote、
DDR/peer或reduction transport action。

该边界必须通用支持以下relation，而不是把当前fixture或physical carrier限制升级成协议：

- reduction的parallel/reduction partition、input/init demand、partial contribution和merge role；
- broadcast/projection的many-to-one image，重复consumer使用不膨胀唯一source set；
- convolution、pooling和supported reduce-window等affine window的kernel、offset、stride、dilation、halo及显式pad/fill pieces；
- non-unit-stride slice/view、permutation、collapse/expand等可组合static relation；
- Presburger union、tail、pad/window分段和组合relation形成的有限multi-piece set。

这些声明支持的类别不能因为不是dense rectangle而返回unsupported。真正超出typed structured/IndexRelation合同的dynamic、
non-affine或未定义numeric semantics可以形成semantic unsupported，但该结论作用于语义/alternative，不是换一个physical
placement即可消除的no-good。

query返回named typed outcome：

1. `satisfied`携带relation identity、consumer domain、producer demand、ownership intersections、dependency/reduction/init role
   和空uncovered proof；
2. `proven logical infeasible`只表示well-formed trial存在精确partition/relation/ownership矛盾，并携带direct witness；
3. `unsupported semantic relation`表示current typed relation能力不能解释verified source semantics，不得缓存成placement非法；
4. `indeterminate/compiler failure`覆盖Presburger预算/内部错误、IR epoch失效和上游协议缺失，调用者不得靠尝试其它placement
   掩盖。

baseline feasibility和spatial search只有收到第2类结果才能删除当前trial。`FailureOr + diagnostic string`压成`bool legal`会把
unsupported/internal failure误当候选no-good，禁止作为跨stage协议。malformed assignment和过期IR epoch同样是compiler
contract failure，不是普通logical rejection。

`satisfied`结果不包含layout、encoding、bytes、dense fragment、local/remote action、route、buffer、send/recv、fusion或
resource schedule。Representation/movement materialization稍后才对exact demand与ownership求交并将其有限分解为physical
pieces；dense rectangle、descriptor、route或transport暂时表达不了，只能拒绝对应physical assignment，不能反写删除logical
placement。deterministic baseline必须为已声明支持的exact set提供一条canonical correctness carrier；它与完整search
representation/movement域都消费同一proof，不另建压缩版demand。

analysis result只在current immutable IR borrow有效；`IREpoch`token只拒绝跨borrow trial，不进入semantic cache key或替代
nested structural snapshot。relation cache观察typed edge/relation identity和borrow内结构，demand/coverage cache还要观察完整
consumer domain、producer ownership、partition/reduction/replication role和所有会改变image的assignment。只有具备extensional
equivalence proof才能投影cache key；按shard dimension和participant count缓存一个legality bool不满足合同。

当前 production placement evaluation仍过早调用physical edge strategy planning；因此“exact demand analysis已存在”不等于
这一边界已在主线闭合。current query还受candidate schedule placement type、direct static single-result/single-init Linalg、
balanced一维shard、skip-init/support和字符串失败分类约束；这些是待替换实现事实，不得缩小上述终态能力。

### 5.4 Spatial与融合机会

Spatial前可以由typed SSA、structured semantics、indexing relation、effect与numeric policy证明哪些依赖边理论上支持
coupled traversal，但真正的TileRegion只能在placement后形成：只有同一Tile上重叠的producer/consumer shard
才能共享SPM domain。部分Tile集合重叠时只对local intersection形成region/coupled候选，其余需求必须显式movement。

更多Tiles不一定更快。Search cost应比较parallelism、local extent、redistribution、reduction merge和fusion机会。
“关键compute维度低于64”当前只可作为target排序seed/penalty示例，不能成为协议常量或legality；它只适用于映射到
compute geometry的相关维度，最终阈值由target facts和实测校准。

## 6. TileRegion、融合与 temporal tiling

### 6.1 两级融合

本文所说的主要融合目标是 region fusion：多个原始 op 进入同一 TileRegion，producer中间值不发生DDR往返。同一region
内可以有两种关系：

```text
independent traversal + SPM retention
  producer先完成自己的loop，完整local shard留在SPM，再由consumer loop读取

coupled traversal
  consumer temporal tile通过exact relation直接驱动所需producer slice
```

前者已经消除DDR，但可能把完整local shard materialize到SPM；后者只保留当前tile/halo/partial result，通常显著缩短
lifetime并降低SPM压力。因此候选优先级是：

1. 同一最大TileRegion中的最大coupled closure；
2. 必要时解除个别coupled edge，但保持同region independent traversal；
3. region仍不可取时才cut，并显式DDR materialization。

不存在“不同TileRegion但同一个shaped value继续SPM驻留”的选择。`SPM residency`不是独立的上层domain，而是最终
TileRegion、Instr order、buffer slots、lifetime和accepted allocation共同证明的事实。

### 6.2 TileRegion formation

Placement后，对每个Tile构造local DAG。从该Tile的observable/local outputs反向遍历producer，沿语义可融合且
local demand非空的关系形成最大连通region envelope。Region formation必须处理：

- boundary inputs/outputs和effect顺序；
- fanout的所有consumer，而不是只给单条edge写`fused=true`；
- reduction partial/merge和numeric policy；
- local/remote demand的精确拆分；
- unrelated components不能仅为“region大”而获得虚假收益。

Region membership、traversal roots、coupled edges和temporal shape都是query-local选择；最终必须物化成真实
`wafer.tile.region`、loop和SSA，而不是group attr或side table。

### 6.3 Coupled closure

从region traversal root的一个consumer tile出发，使用`IndexRelation.image`沿producer链递归求精确slice。只有effect、alias、
coverage、reduction和numeric legality都可证明时，producer才进入coupled closure。Fanout必须明确选择multi-output
traversal、保留一个region-local materialized version、selected recompute或解除coupling；不能遗漏任一consumer。

有效coupled traversal必须在实际IR中证明：producer位于consumer temporal traversal内、需求slice all-and-only覆盖、
intermediate为tile-sized而非完整local shard，且不存在中间DDR round-trip。

### 6.4 Complete temporal tile vector 与 wave-loop order

Temporal state覆盖structured op/traversal的全部iterator，并在Q50.D已选traversal内选择有限、语义可区分的wave-loop
nesting/order：

```text
[parallel tiles..., reduction tiles..., batch/head/channel/window tiles...]
+ wave-loop nesting / order within selected traversal
```

它不是单一block size，也不另设与主向量脱节的reduction字段。候选从当前spatial local extent和target-efficient shapes开始，
系统产生会改变wave、tail、native issued work、padding、transaction、halo、buffer feasibility或SPM footprint的breakpoints。
若其它choices固定且footprint对某一维单调，可用二分定位capacity边界；layout、buffer、fusion或spatial改变后结论必须失效。
只有能证明生成相同actual traversal、exact demand、lifetime、tail和numeric order的排列才能canonicalize；默认loop order不能
删除会改变reuse、SPM或最终resource schedule的合法组合。

Fusion不能在temporal之前选出独立winner，temporal也不能先固定后再附加fusion。完整局部选择至少是：

```text
(spatial shard,
 TileRegion membership,
 traversal roots,
 coupled edges,
 complete temporal tile vector,
 physical representation,
 buffer recipe)
```

Actual SPM失败时由selection owner生成新选择；allocator不得在已选IR上反复`tile_size / 2`并冒充原winner。小tile只要
legal仍保留在domain，但会因低compute利用率、更多waves/messages/instructions受到cost惩罚。

## 7. Physical representation、movement 与 instruction pipeline

### 7.1 Representation 与 movement

每个scheduled physical value选择consumer和lowering可接受的`MemLayout`/encoding。Representation不一致时，候选必须显式
选择local conversion、peer/collective transfer中的转换、DDR spill/reload或其它已定义movement，并把retention/release与
boundary obligation编码为typed assignment；lifetime和completion再从实际region、movement、buffer、order与IR epoch重算。
Lowering只能消费并验证selected choice，不能提供隐藏canonical winner。

Logical demand先于physical fragment。Movement materialization必须对reduction、broadcast、window/stride与multi-piece在内的
exact demand和producer ownership求交，保留init/contribution/replication role，证明local和remote fragments all-and-only覆盖、
必要contribution不丢失且非显式replica不重叠，再生成实际staging、send/recv/wait或DDR store/load。任何densify、bounding-box
或descriptor分段都必须回证union等于原exact set。

Movement proposal可以消费一个query-local、可失效的relation-derived reuse analysis：它从Q50.A exact demand、selected
placement、TileRegion/traversal、wave-loop order和representation推导spatial-demand equivalence/invariance classes、
temporal-wave invariance classes与exact payload/coverage。结果不压成几个boolean attr，不写回候选IR，也不选择broadcast、
load/receive placement、retention、route或buffer winner；Q50.H据此完整生成direct/refetch/unicast、partial或多维
multicast/broadcast、合法的same-region load/receive外提与retain/release等普通typed alternatives。最大broadcast只能优先，
不能提前删除在NoC contention或后续schedule下更好的partial broadcast/unicast。

### 7.2 Buffer recipe

搜索对象不是孤立`bufferCount`，而是能实际物化的recipe：

```text
loop/wave identity
slot multiplicity and rotation
producer/transport/consumer ownership
prologue / steady / epilogue
issue / wait
last consumer and reuse completion
```

Single buffer可以表达合法但串行的执行；稳定compute/movement或stage overlap通常需要double/multi-buffer。所有slot必须是
独立allocation，actual range/lifetime验证通过后才能宣称overlap。

### 7.3 Ready order、worker 与 completion

Ready order、worker和completion只在selected Tile/Instr结构上物化。Query-local event/calendar可以用于排序、局部
feasibility和cost，但不能成为持久shadow schedule。Fresh completion必须从最终Instr control/effect关系重建；任何旧analysis
结果在IR变化后失效。

## 8. Stage pipeline

Stage pipeline是可表达的一等候选，但不是无条件展开的第一主轴，也不等同于多batch。单次invocation可沿temporal output、
token、head、channel、spatial或合法partial-reduction chunk形成流水：

```text
t0: stage A(chunk0)
t1: stage A(chunk1) | stage B(chunk0)
t2: stage A(chunk2) | stage B(chunk1) | stage C(chunk0)
```

一个有效候选必须联合包含：

```text
stage partition
+ each stage Tile group
+ intra-stage spatial mapping and maximal fusion
+ exact stream chunk relation
+ source/destination staging
+ NoC/DDR movement
+ multi-buffer recipe
+ issue/wait, ready order and completion
```

只有stage cut或只有`bufferCount=2`没有执行语义。合法性要求chunk coverage、producer-before-send、matching recv/wait、
consumer readiness、slot reuse safety、reduction merge和无死锁全部由实际IR证明。

候选排序前可用streamability analysis检查：consumer chunk是否只依赖有限producer slice、chunk数能否摊薄fill/drain、
stage是否使用可并行资源、buffer recipe能否物化、stage是否严重失衡。证明不满足可删除该具体pipeline选择；没有profile
收益只应降低优先级，不应把硬件可表达的结构永久宣布为unsupported。

Decode batch=1可能没有足够token waves，通常优先head/channel、K/V partition、GEMM spatial和DTE/compute overlap；不能为
“支持pipeline”强制产生大量无意义stage cuts。Stage pipeline的proposal优化在主spatial/region/temporal路径和multi-buffer
机制闭合后，由实际workload profile驱动。

## 9. SPM、DDR 与 resource verification

Search-time working set可以形成lower bound和排序信息：

```text
live inputs + tile-sized intermediates or retained local shards + outputs
+ representation conversion + communication staging + rotating slots + padding
```

只有已证明必然同时live的集合超过capacity时才能安全早拒绝。Region-local probe使用实际TileRegion lowering和SPM packing，
可快速拒绝固定causal choices；它成功不能替代整卡verification，相关spatial/region/temporal/layout/buffer/order变化后必须失效。
Symbolic footprint必须区分proven must-coexist lower bound、non-binding ranking estimate和缺少坐标时的
`deferred(required coordinates)`：只有第一种超过capacity才能exact reject；估算可放下不能证明packing可行，未证明
coexistence的估算超限也不能拒绝候选。近似或未经typed boundary-faithful proof的外部solver/序列化constraint结果只能排序；
exact局部solver可对准确建模的子问题返回proof/proposal，但selected offset/choice仍须物化并通过typed gate复验。

完整候选完成TileRegion->Instr、selected order/worker和fresh completion后，SPM/DDR planner才从actual roots、control flow、
lifetime和coexistence求validated offsets。Planner区分`Feasible`、`ProvenInfeasible`与`ResourceExhausted`，调用边界还必须
区分internal failure；只有`ProvenInfeasible`或确定unsupported才能拒绝对应causal assignment并形成no-good，资源耗尽或
内部失败是indeterminate，不得删除合法状态。Planner不改变tile、region、traversal、spill、buffer、order、worker或completion。

完整CardExecutable verification还必须原子验证message matching、transport range、per-link/card-shared resource、ABI和observable
completion。任一失败丢弃clone，不允许提交部分Tile或复用失败clone中的offset。

## 10. Cost 与分层候选编译

### 10.1 Cost边界

Hard legality/capacity与performance estimate严格分离。Cost model复用current work collectors和target参数；缺失参数按同一
comparison cohort统一删除该term，不能candidate-local按零或无穷。Estimate只用于排序、diversity和best-known选择；除非
某lower bound或dominance逐项证明安全，否则不能删除合法候选。

至少保留per-Tile issued work、critical path、DDR/SPM/NoC bytes、directed-link load、message/descriptor/instruction/wait、
Tile idle与pipeline prologue/II/epilogue。SPM high-water只用于capacity/headroom与诊断，不作为Pareto目标；working-set
带来的性能差异通过movement、spill、buffer、waves与issued work计价。Overlap只有actual buffer/resource独立性可证明时
才计入，不对整程序无条件相加或取`max`。

分层resource projection复用现有target facts，把actual或partial typed assignment依次投影到compute unit/worker、相关local
SPM resource、directed NoC links与DDR channel/engine。资源集合相交只形成contention estimate/排序信号，不相交可优先提出
并行proposal；是否能重叠及其cost仍由actual event/resource semantics判定。Nominal bandwidth split、解析关键路径或其它近似
模型只能用于ordering；没有actual buffer、issue/wait、event与resource independence时，不能据此签发completion、overlap、
legality或admissible bound。

### 10.2 分层编译成本

搜索按成本层次推进，但不把这些层次建成新的长期output：

| 层次 | 责任 | 能否成为winner |
| --- | --- | --- |
| relation/semantic gate | exact coverage、effect、numeric、topology symmetry | 否 |
| analytic ranking | resource lower bound与performance estimate | 否 |
| affected TileRegion probe | 实际local lowering/lifetime/SPM packing | 否，仅形成scoped failure或cache |
| complete candidate compilation | CardModule→TileRegion→Instr→SPM/DDR→transport/resource/ABI | 是 |

完整candidate compilation只能成功形成accepted CardExecutable、proven exact rejection或indeterminate failure；它不能修候选、补默认layout、自动换DDR、
解除fusion或生成新的搜索选择。

## 11. Search algorithm 与质量合同

### 11.1 唯一 owner与两种策略

只有`PhysicalDataflowSearch`语义层可以管理candidate frontier、回溯、budget、incumbent和最终choice。它通过所属IR层的普通
analysis取得合法choices，通过transformation/conversion物化selected choices；不存在统一`Providers`框架或局部selector。
search kernel本身不生成spatial、region、temporal、layout、movement、buffer或schedule choice，也不预先定义这些轴的
placeholder schema。每个mechanism进行pure domain query并返回named typed transition；kernel只验证/apply transition、维护
deterministic frontier和stable dedup、路由typed evaluation outcome、记账并更新同一个incumbent。

Q51.Core只先闭合上述control kernel。它不拥有Q49.P的single-root capacity probe；Q50.F在完整causal coordinates闭合后加入
通用affected-region probe。Core也不靠mock domain签发真实physical domain完整性：它只用独立finite state-graph model检查
frontier mechanics；每个Q50轴再加入本轴的independent reference enumerator，全部真实轴的production flat exhaustive runner
与actual digest/winner comparison由Q51 closure签发。Core checkpoint即原子接管public `search`的frontier、actual comparison和
winner update；尚未迁移的current bounded generation只可作为显式lossy proposal source进入同一Core，不再是第二owner。

同一choice domain支持：

- small-DAG exact strategy：独立tiny reference domain enumerator不调用production domain builder，证明合法typed域完整；
  production-mechanism flat exhaustive runner再用真实materializer/cost/exact gates完整展开，作为candidate set与剪枝oracle；
- real-workload anytime strategy：先建立actual baseline incumbent，在预算内持续产生和编译更好候选。

### 11.2 Anytime构造

真实负载不展开平铺笛卡尔积。Q51 closure建立的non-lossy基本过程是：

1. 直接接收Q49.P已经完整准入的deterministic baseline及同comparison cohort actual cost，立即获得合法incumbent，不重新
   materialize或compile baseline；
2. 生成producer/consumer aligned、compute-balanced、topology-local、reduction-parallel及ordered factorized regular mapping等
   spatial seeds；
3. 对每个seed构造最大TileRegion、最大coupled closure和高效大temporal tile，并用relation-derived reuse signature优先
   direct/local、multicast/broadcast与temporal retention alternatives；
4. exact cheap gates后按cost与结构diversity排序；
5. promising state做affected-region probe，少量完整state做CardExecutable compilation；
6. actual accepted cost按完整semantic tie-break更新incumbent；只有proven exact failure及future-compatible causal scope才能
   形成no-good，resource exhaustion或未分类evaluation保留为unresolved，不能删除state或签发最优性；
7. budget耗尽返回best actual accepted candidate及frontier coverage、valid lower bound、unresolved work和ledger evidence。

Q51.Core不固定production best-first priority、默认budget、dominance/no-good算法或LNS。Q51只使用不会永久丢state的稳定
ordering和exact continuation闭合正确性；Q52依据fresh profile再加入或调整candidate priority、memo、safe dominance、
branch-and-bound和large-neighborhood search。Large neighborhood必须联合修改耦合choices，例如`spatial+region+coupled+temporal`，或
`stage cut+两侧Tile groups+chunk+movement+buffer+order`；当前单坐标改进只能暂时作为warm-start，不能继续拥有coverage、
family closure或最终winner。

### 11.3 剪枝和trade-off

可以预先使用而不损失合法解的优化包括exact legality、canonical memo、真实topology symmetry、typed causal no-good、
已证明safe dominance和完全等价的partial-compilation cache。

时间上限、bounded diverse candidate set、beam、LNS或只完整编译部分高分候选可能漏掉全局最优。它们只能作为Q52实测后的显式
trade-off；完整合法domain仍需lazy可生成，但一次production invocation不承诺访问全部状态。结果分为：finite域完整覆盖且
全局bound闭合的`optimal-certified`；未展开completion仍由完整exact continuation与admissible bound表示的
`feasible-with-bound`；已经永久丢弃或未表示合法completion的`budgeted-feasible`。单纯预算中止不自动降级，只有最后一类
不能声称global bound或最优。DP或局部solver只处理边界和语义完整的typed子问题，不能复制第二套全局relation/lifetime/resource模型。

Q52代表负载以10分钟作为热点与首轮质量检查点，允许继续到30分钟；这两个数是当前验证计划，不进入IR、pass或output
协议。确定性work credits、actual compilation数、wall、RSS和incumbent曲线同时记录。

### 11.4 Deterministic baseline

`none`在performance search之外构造单一方案：原始DAG、每个structured compute root独立使用最大合法Tile participation、
每root独立TileRegion、root boundary compiler-owned DDR、buffer=1、零fusion和零可选edge action。同一Tile允许按完整
semantic tie-break承载多个独立TileRegion，但一个baseline TileRegion只能有一个structured compute root。shape/index/view、
target-local materialization等没有独立structured DAG identity且不能独立调度的op才属于non-root support closure；显式
Fill、Reduce或DPS init producer只要是另一个structured DAG node就仍是第二root，不能借support名义并入。root cardinality按
typed materialization relation计算，因此一个root lower成多个compute/instruction op仍只算一个root。每个root从其
local spatial extent开始确定性缩减完整temporal tile vector直到actual SPM可放下；relation要求的必要peer fragments只用于
correctness，不搜索route或communication方案。

这里的baseline是功能闭环的最低实现，不是“只验证预先选好方案”的通道。它接收未绑定Tile、TileRegion、temporal tile、
representation或buffer choice的正常上游IR，并拥有产生可执行结果所必需的全部canonical决定。对于声明支持的
structured semantics和target，只要baseline合法域内存在可执行completion，`none`必须结束于accepted CardExecutable；没有
性能search、最大初始tile放不下或第一个placement option不合法都不是失败理由。它不承诺最优、不尝试fusion/multi-buffer/
可选route，也不把baseline域扩大成Q51性能域。“声明支持”由current typed op/interface、verifier、canonical target lowering和
representation合同判定，不由workload名字、shape特判或当前fallback是否碰巧成功反推；语义超出该合同、最小合法执行仍超出
真实资源以及compiler/internal failure必须分别报告。

per-root participant group只描述该root的非空执行Tile；最终CardModule仍覆盖target要求的all-and-only完整Tile domain。
未参与某个root的Tile只在最终完整CardModule中按current IR合同存在，不得为了局部probe构造card-shaped no-work wrapper。

baseline controller直接从typed structured semantics、exact relation和target facts构造这一条canonical路径。最大非空
participant count、physical Tile group、iterator/factor choice和独立root顺序均使用完整semantic tie-break，不使用pointer、
walk ordinal、`stableOrdinal`或search proposal order。它以有限、确定且不会被beam/cap/time budget截断的feasibility
resolution遍历baseline合同内的合法fallback，返回固定全序中第一个所有required scoped exact probe均为fit的canonical
completion；只有随后一次完整Q50.0 gate才能将其标为accepted。query-local trial和fit结果
只是合法化过程，不是Q51 search candidate：controller不评分、不维护incumbent/candidate family、不保留用于比较的备选方案，
也不得创建或调用search candidate/state、search-oriented domain/ranking evaluator、grouping materializer和feedback repair。
它复用policy-free placement-domain机制和上述logical demand/coverage query，不能复制第二套placement语义，也不能调用
physical edge strategy/evaluator来决定logical trial是否合法。

controller产出的窄immutable selected assignment是feasibility resolution的**输出**，不是baseline入口的前置条件；baseline与
search可以在这个policy-free已选事实或actual IR边界汇合，而不是共享candidate wrapper。该结果只含materialization所需的
per-root placement、显式singleton region boundary、完整temporal vector以及已经由baseline确定的canonical
representation/movement/buffer/order/completion事实，不含score、derived metrics、stable ordinal、transition/failure history或
controller flags；materializer不得自行补grouping/default choice。

logical query对reduction、broadcast、affine window、stride/view和multi-piece返回`satisfied`后，baseline的canonical
correctness carrier必须把exact set有限分解为all-and-only DDR/必要peer movement并继续完整编译；dense-only fragment或单条
descriptor表达失败是baseline carrier的实现缺口，不是logical placement failure。完整search可在之后枚举其它representation、
route、retention或collective，但不能因此改变Q50.A logical outcome。

temporal feasibility从root的完整local iterator extent开始，在iterator/indexing semantics、tail、target vector/alignment、
source numeric/reassociation、reduction order及最小合法粒度共同定义的有限breakpoint lattice上按semantic全序推进。每次trial
都重新推导全部operand slice、stride/dilation halo、result/init/accumulator、temporary、materializing copy、movement staging、
alignment/bank和实际lifetime，不得只按output tensor字节数缩放估算。actual SPM overflow时只沿direct typed witness影响的
合法维度进入下一组更小breakpoint，重新执行同一exact probe；
序列必须覆盖到target允许的最小合法temporal vector。第一个fit立即成为该root的resolved assignment，不为了性能继续比较
其它fit；trial数不得写入candidate proposal/fusion/layout/buffer统计。

为避免重复整图materialization，baseline在单root TileRegion边界做actual scoped fit。probe先消费真实TileRegion；lowering若
需要call/symbol closure，则只提升到最近合法`IsolatedFromAbove` ancestor并在该scope得到最终typed结果，不得忽略
requires-ancestor-scope，也不得为单region构造synthetic Module/Func或带其它no-work Tile的整卡wrapper。结果只能是fit、带
direct typed witness的proven exact rejection/unsupported，或indeterminate。capacity witness可以包含一个冲突集合，但必须把
all-and-only allocation/lifetime owner直接关联到当前single root的result/operand demand和temporal assignment；unsupported
witness必须命名无法表达的typed lifetime/call relation及已尝试的最窄合法scope。两者都不能按相同type/shape猜测producer、
解析diagnostic字符串或同时缩减多个歧义match。只有proven exact rejection允许controller推进确定性fallback；
indeterminate必须终止并作为compiler/internal failure报告，不能伪装成输入unsupported或“none没有方案”。

全部root获得resolved baseline assignment后，baseline只物化一次完整CardModule，并只进行一次complete CardExecutable
compilation。局部成功的安全性建立在“一root一TileRegion、跨root shaped dependency显式DDR、无fusion、sequential
single-buffer execution”结构不变量上；改变任一项后局部成功不能继续当作完整证明。只有target允许的最小合法temporal
vector连同其它canonical fallback都被exact证明不可行时，才能返回typed capacity/unsupported；声明支持的输入在完整gate才
暴露可由更早baseline legality query发现的失败，属于compiler contract缺口，不是正常`none`结果。没有search candidate绝不能
成为失败原因。
`actual_fused_edges == 0`只是必要结果，不能替代对每个baseline TileRegion structured-root cardinality、跨root DDR
store/completion/load和search-policy调用闭包的结构检查。

## 12. Existing mechanics处置与代码边界

### 12.1 处置分类

1. **直接复用**：`StructuredDAGAnalysis`、`IndexRelation` exact demand、physical access/transfer proof、Card/Tile/Instr
   lowering、SPM/DDR planning、transport/resource/ABI verification。
2. **改造接口后复用**：placement domain、event/resource mechanics、selected-buffer materialization、physical movement
   materialization、当前完整候选正常编译链。
3. **提取机制后替换owner**：旧layout PBQP的合法域/materialization、ready-order、software pipeline、worker/completion、
   NoC/collective、structured algorithm transformations。
4. **replacement稳定后删除**：当前大candidate bag、coordinate-descent winner owner、materialization failure后的原地repair、
   hidden layout/route/action winner、forced-fusion selector和旧coordinated selector。
5. **无适配价值则删除**：只服务rank==Tile、固定proposal cap、旧qualification组合或重复事实源的代码和测试。

旧机制文件存在不等于production能力；若未进入current build，必须按新接口逐项恢复、编译和验证。反过来，旧文件曾经工作也
不构成永久保留理由。

### 12.2 源码层次

不按硬件范围或抽象角色新增横向目录层；目录必须对应稳定的 IR、analysis、lowering 或验证职责。稳定职责分布为：

- TensorProgram analysis：fusibility、iteration partition、streamability；
- PhysicalDataflow analysis/transforms：exact demand、TileRegion formation、coupled traversal、representation、movement；
- Instr analysis/transforms：buffer lifetime、ready order、worker、completion；
- Conversion：TensorProgram到CardModule、TileRegion到Instr；
- Compiler search：typed assignment、deterministic frontier、dependency invalidation、session evidence与exact/anytime control；
  profile-driven priority、scalability优化和LNS由Q52加入；
- CardExecutable compilation/verification：串接既有lowering和exact gates，不实现choice generation。

顶层编译入口只编排`none`或`search`并发布结果；不能继续容纳具体spatial/fusion/layout/buffer算法。Q51.Core先原子抽出唯一
public frontier、incumbent和winner control；尚未迁移的current生成能力只能作为显式lossy typed proposal回到同一Core，内部
beam/cap/shortlist必须报告coverage loss，不能伪装成exact domain、比较actual cost、更新winner或repair accepted IR。每迁移一个choice轴，新旧生成路径复用只消费显式typed
assignment的同一mechanism并删除重复mechanics；新路径迁移同等能力并通过actual witness后，旧字段、hash分支和repair逻辑
必须删除，不能长期保留两套事实源。

## 13. 当前差距与任务闭环

Q50.0、Q54、Q59与Q50.A已经闭合无策略complete-candidate compilation、MLIR scope/pipeline、compiler transaction和
policy-free exact logical demand边界。Q50.A current API已经脱离candidate schedule/evaluator，以immutable borrow、完整logical
trial、typed relation/role和四态outcome服务baseline及未来spatial mechanism；reduction、broadcast、window/stride、multi-piece、
multi-result、init/support relation与production carrier gate已经受测，不再属于Q51.Core待修接口。

Q49.P仍在闭合baseline functional legalization、single-root structure、scope escalation、direct causal witness和search-policy
隔离；其动态缺口只看`tasks/progress.md`与实施计划。主线仍有下列设计差距：

- baseline仍复用search-oriented candidate/domain evaluator、stable ordinal/proposal ordering、candidate统计和group
  materialization，同Tile的多个独立structured root可能进入同一TileRegion；零fused-edge统计掩盖了
  SPM/lifetime/lowering scope耦合；
- temporal refinement前后仍重复完整CardModule materialization，scoped路径仍会形成包含no-work Tile的card-shaped
  wrapper；requires-ancestor-scope没有在最近合法scope完成probe；
- SPM failure attribution仍可能从DAG edge和相同type/shape反推受影响producer并扩大refinement，缺少从实际
  lifetime/packing到当前single root及其temporal assignment的direct typed causal witness；
- placement production transition仍过早消费physical edge strategy；Q50.B尚未从all-iterator semantics生成完整multi-axis、
  remainder、reduction/merge和非矩形physical placement域；
- candidate、shortlist、repair与完整编译混在单体synthesis文件；
- spatial domain仍主要是单output axis与连通矩形Tile group；
- region/temporal/fusion/layout/buffer/communication尚未由一个轻量state联合回溯；
- 多个旧机制文件未进入current production build；
- stage pipeline只有历史机制，未接入current common search。

任务按以下output闭环推进，具体状态以`tasks/progress.md`为准：

1. 提取无repair的CardExecutable compilation/verification边界；
2. 修复placement与layout-independent exact demand边界，以完整logical shard domain、typed dependency role和四态outcome给
   baseline/search提供不携带representation/movement policy的proof；闭合reduction、broadcast、window/stride、multi-piece、
   DPS init与support relation，删除carrier失败和字符串失败对spatial legality的反写；
3. 以Q49.P从current输入闭合baseline功能合法化、结构、policy、probe、causal witness和materialization解耦；
4. 在accepted baseline可直接作为incumbent后建立只管理frontier/evaluation/result的search control core，以独立finite
   state-graph model证明kernel不会漏state、吞sibling或把indeterminate改成rejection，并原子接管public winner control；
5. 依次闭合structured alternatives、spatial partition/placement、TileRegion/temporal/fusion、representation/movement、
   Instr pipeline机制，每项边实现边接入common state，并加入本轴独立reference domain oracle；
6. 完成所有维度的production flat exhaustive runner、search closure和实际fusion gate；
7. 从第一版保留条件式stage-pipeline transition；Q52只在profile后优化其proposal顺序，并加入或加强memo、typed no-good与LNS；
8. fresh workload/package/no-card/board closure。

## 14. Completion Gate

### IR / verifier

- distinct Tile modules可以包含不同op、loop、shape和长度；
- duplicate/unavailable Tile、coverage hole/overlap、illegal reduction merge、cross-Tile/region SPM alias失败；
- same-region independent traversal、coupled traversal、region cut、selective spill、recompute和communication staging可验证；
- selected spatial mapping的iteration coverage、result ownership和edge demand all-and-only闭合。
- exact-demand query按完整logical domain支持reduction input/init/partial contribution、broadcast、affine window+stride/dilation、
  strided view/slice、multi-piece、multi-result和support relation；physical carrier变化不改变logical outcome。

### Baseline与search correctness

- tiny chain、diamond、fanout、reduction、mixed compute/movement和stage+buffer case先由独立reference enumerator证明domain
  coverage，再由production flat exhaustive runner证明candidate set/剪枝/winner；
- 构造必须联合改变spatial/region/temporal等多个choice才改善的陷阱，anytime策略能找到actual accepted改进；
- estimate、memo、no-good或dominance逐项开启不改变small-oracle winner；
- `none`和`search`共用完整candidate compilation/verification；
- baseline与spatial mechanism共用policy-free typed exact-demand query，只有`proven logical infeasible`删除trial；unsupported和
  indeterminate不会进入legality bool/no-good cache；跨immutable borrow、nested structural snapshot或任一观察到的domain/role
  变化使cache失效；
- `none`不构造或调用search state/candidate、search-oriented domain/ranking evaluator、proposal ordering/group materializer
  或candidate统计；每个baseline TileRegion恰有一个structured compute root，跨root shaped dependency均有显式DDR边界，
  同Tile多root表现为多个顺序TileRegion；
- `none`从未选择placement/temporal/layout/buffer的正常上游IR自行形成resolved baseline assignment；至少一个初始完整
  temporal tile因真实operand/halo/temporary/lifetime footprint超出SPM的case必须沿合法breakpoint缩小后通过完整
  CardExecutable、package与no-card，且multi-axis/tail/alignment变化会重新推导完整workset；
- deterministic feasibility resolution覆盖到target最小合法temporal vector，不设会丢失fallback的beam/cap/budget；最小
  vector仍超限的negative case返回direct typed capacity witness，indeterminate按compiler failure而非unsupported处理；
- baseline scoped probe覆盖region-local与最近合法isolated-ancestor两种scope，exact SPM rejection携带直接causal witness；
  fresh计数证明完整CardModule materialization和CardExecutable compilation各一次；
- 结果明确区分`optimal-certified`、`feasible-with-bound`与`budgeted-feasible`，并报告work、wall、RSS和incumbent，
  不伪造bound或最优性。

### Fusion与pipeline有效性

- 多个原始op实际进入同一TileRegion，中间edge无无意义DDR store/load；
- 至少一个selected case物化真实coupled traversal，intermediate为tile-sized而非完整local shard；
- fanout、reduction和observable coverage完整；
- selected stage pipeline具有足够chunk、实际Tile groups、movement、multi-buffer slots、issue/wait和reuse completion；
- 不要求每个decode都使用stage pipeline，无收益case保持maximal local fusion。

### End to end

- generic mixed DAG、official HF prefill、functional KV-cache decode和Llama block以FP16/BF16 fresh生成完整
  ExecutablePackage并通过no-card；
- current package逐case证明selected layout/physical payload、all-and-only movement、Direct-DTE issue/wait与compute overlap
  witness；source、fixed seed、dtype、framework eager oracle和原dtype/shape comparator由case owner固定，不进入产品frontend；
- Q52在10/30分钟检查点解释主要热点、重复工作和质量曲线；
- board-ready后真实设备只串行执行current matched cases；Llama及至少一个prefill/decode代表相对同源baseline获得可重复
  改善后才能标记done。

## 15. 参考算法原则

本设计吸收但不复制外部实现中的consumer-driven exact demand propagation、factorized spatial mapping、coarse-to-fine
schedule construction、large-neighborhood search、boundary-compatible DP和event/resource scheduling。任何论文或仓库的
shape、operator集合、solver依赖、beam width或GPU block常数都不成为本项目协议。

对[TileLoom](https://arxiv.org/abs/2512.22168)只选择性吸收四类机制：ordered factorized regular mapping作为高优先级proposal
family、从exact relation推导的spatial/temporal reuse analysis、基于现有target facts的分层resource estimate，以及Q52对
ranking/top-k的实测校准方法。它们只影响query-local analysis、proposal order与profile-driven policy；Q51的完整合法域、
typed assignments、唯一winner owner和actual exact gates保持不变。

明确不采用：前端预先固定block/tile shape；把连续矩形、full occupancy、二维映射或偶数participant当作完整域；greedy
pre-fusion或只保留最大broadcast；clone-per-mapping、函数名/attr、aggregate reuse bool或JSON/opaque sidecar状态；approximate
footprint代替actual lifetime/packing；无actual slots/events就假设double buffering或`max(compute,movement)` overlap；固定top-k、
论文硬件常数或外部solver/model bound进入exact pruning；把已经算法化的Flash/Decode输入误当成physical planner自动发现算法。
