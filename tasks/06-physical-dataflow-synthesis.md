# Card 内 Physical Dataflow 综合

状态：2026-08-13 按 current `TensorProgram -> CardProgram -> TileRegion -> Instr -> CardExecutable`
主线重写。本文是 card 内 spatial mapping、TileRegion formation、temporal tiling、融合、physical
representation、movement、buffering、instruction scheduling 与候选选择的唯一设计 owner。动态状态和施工顺序只看
`tasks/progress.md` 与 `tasks/plans/physical-dataflow-synthesis.md`。

旧 whole-rank、coordinated selector 和当前单体 synthesis 实现只提供可审计的历史机制或迁移素材，不构成本文合同。
现有实现能直接适配就复用，需要改变边界就改造，只有算法或 verifier 有价值就提取后替换，没有价值或无法适配则删除；
不为了保留旧文件而扭曲终态架构，也不在 replacement 完成前无证据删除仍需迁移的能力。

## 1. 核心结论

编译器优化对象是一个 card-local structured DAG 的完整物理数据流，而不是单个 op、单条 edge、预先切好的 16 个
logical rank 或所有 op 共用的 tile shape。一个候选必须共同决定：

1. structured iterator 如何切成 logical shards；
2. logical shards 如何放到 physical Tiles，reduction partial/merge 如何分布；
3. 每个 physical Tile 上哪些计算共享一个 `wafer.tile.region`；
4. region 内哪些 producer/consumer 形成 coupled traversal，哪些采用独立 traversal；
5. 每个 traversal 的完整 temporal tile vector、loop order 与 tail；
6. physical layout、local/peer/collective/DDR movement；
7. buffer recipe、ready order、worker、completion 与资源 overlap；
8. 是否形成跨 Tile stage pipeline，以及其 chunk、Tile groups、movement 和 multi-buffer；
9. observable output/effect 全部完成时的整卡 cost。

统一搜索只统一上述决策的 owner、回溯和最终选择，不把各机制实现集中到一个文件或一层目录。analysis 靠近其可解释的
IR，transformation 靠近被修改的 IR，conversion、memory planning 和 admission 保持独立。候选一旦物化，以当前 IR
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
- Upstream artifact / IR:
  GSPMD完成card级分区、target-independent normalization完成后的card-local TensorProgram；若存在算法级等价
  alternative，每个alternative已经在isolated clone中物化为真实structured DAG。SSA、structured iterator、
  indexing relation、effect、type、shape和dtype均可验证，尚未绑定physical Tile。
- Current stage responsibility:
  在同一可回溯选择过程中决定iteration partition、physical Tile placement、reduction distribution、TileRegion、
  traversal connection、temporal tile、physical representation、movement、buffering、stage pipeline、order、worker和
  completion；完整候选进入正常Card/TileRegion/Instr lowering和exact resource admission。
- Output artifact / IR:
  被选中的CardProgram及其all-and-only physical Tile programs；其中TileRegion、loop、movement、Instr、SPM/DDR
  allocation、completion和physical endpoint均是实际IR或accepted resource事实。通过全部gate后形成CardExecutable。
- Downstream consumer:
  target conversion、device link、ExecutablePackage emission与runtime launch。
- User-level driver / named pipeline:
  wafer-compile source-to-package pipeline；public optimization policy只有`search`和`none`，不增加逐机制selector。
- Explicit non-goals:
  不重做跨card GSPMD；不要求真实大workload全局最优证明；不从op/workload/文件/shape名字恢复语义；不建立
  cycle-exact simulator；不让lowering、allocator、communication或completion私自repair候选；不发布search sidecar。
- Completion gate:
  small DAG由独立reference domain enumerator证明合法域完整，再由production-mechanism flat exhaustive runner证明
  frontier、剪枝和winner一致；真实workload在约定预算内返回通过完整编译与admission的
  best-known CardExecutable；selected IR证明spatial all-and-only coverage与有效region fusion。至少一个代表case还要证明
  coupled traversal、tile-sized intermediate和无无意义DDR往返；selected multi-buffer/stage-pipeline仅在对应候选获选时证明。
```

`optimization=none`由确定性 baseline construction产生一个选择，不进入性能搜索；`optimization=search`由本文唯一
candidate-selection owner管理多个选择。两者从 CardProgram materialization 起进入同一 lowering、memory planning、
admission 和 package emission 路径。

## 3. 稳定 IR 与 artifact 边界

### 3.1 `wafer.card.program`

`wafer.card.program` 是一个 `card_id` 的完整 MPMD verifier 范围，拥有 card-local observable inputs/outputs、
shared DDR boundary、all-and-only physical Tile programs，以及跨 Tile message coverage/completion 验证范围。它不保存
frontier、score、候选列表、route side table 或 target calibration。

### 3.2 `wafer.tile.program`

`wafer.tile.program` 绑定唯一 physical `tile_id`。不同 Tile 可以拥有不同 op、loop、temporal shape、worker 和执行长度。
实际顺序、并发和依赖由 body 中 loop、SSA、send/recv/wait 与 event 表达；不另存全局 schedule attr。SPM root 或
alias 不能跨 `tile.program` SSA 传递。

### 3.3 `wafer.tile.region`

`wafer.tile.region` 是一个 physical Tile 上的 SPM ownership/lifetime domain，不是硬件 Tile，也不是单个 loop 或
“fusion group”标签。一个 region 可以包含：

- consumer-driven coupled traversal；
- 多个不同 temporal shape 的独立 traversal；
- local physical conversion；
- selective spill 某个 root，而其它 roots 继续存活；
- region 内 movement、communication、sync/effect ordering。

SPM root 和 shaped alias 不能跨 TileRegion。不同 TileRegion 间 shaped data 必须显式 materialize 为 DDR
store/completion/load；不同 physical Tile 间则由 source SPM、NoC/DTE send、destination SPM staging、recv/wait 表达，
不能共享同一个 SPM root。

### 3.4 `CardExecutable` 与 `ExecutablePackage`

`CardExecutable` 是内存中已经通过 all-and-only Tile coverage、Instr、SPM/DDR、transport、resource、completion 和 ABI
admission 的执行对象。`ExecutablePackage` 是 target lowering、link 和 emission 后交给 runtime 的磁盘发布物。旧 C++
实现中的 `*Bundle` 名称只可作为迁移索引；它不再定义主线 artifact 层次，新设计不继续扩散该词。

## 4. Query-local problem 与 search state

### 4.1 问题定义

一次 candidate-selection invocation 可读取：

- current TensorProgram alternative 与稳定 DAG node/edge identity；
- physical topology 与 available Tiles；
- structured iterator、indexing relation、effect 与 numeric policy；
- 各层合法选择枚举和实际 materialization 能力；
- target geometry、capacity 和 cost 参数。

它不保存 winner，不保存 accepted offset，也不把算法名或 opaque parameter bag 传入 physical legality。

### 4.2 Search state

状态只保存不可从当前选择重算的 typed assignments：

```text
TensorProgram alternative
StageId  -> members / physical Tile group
NodeId   -> iteration partition / physical placement / reduction role
RegionId -> members / traversal roots
EdgeId   -> coupled / independent / recompute / local-or-remote movement
LoopId   -> complete temporal tile vector / finite wave-loop nesting and order within selected traversal
ValueId  -> selected physical representation
TileId   -> buffer recipe / instruction order / worker / completion choice
```

状态不保存 raw `Operation *`、estimated SPM、汇总 bytes、fragments、resource calendar、makespan、actual offsets 或
repair history。exact edge demand、lifetime、calendar、cost 和 SPM high-water 都是 query-local、可失效、可重算的
analysis cache；候选物化后以 IR 为唯一事实源。

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

### 5.2 Physical Tile placement

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

不同 op 可以选择不同 mapping。对每个 consumer Tile：

```text
consumer local iteration domain
  -> IndexRelation.image
  -> exact producer logical demand
  -> 与每个producer Tile ownership求交
  -> local / remote / reduction demand
```

需求与 ownership all-and-only覆盖失败才是 logical placement failure。Layout、dense rectangle、fragment carrier、route 或
transport暂时表达不了，只能拒绝对应的physical representation/movement choice，不能反写删除logical placement。

当前 production placement evaluation仍过早调用physical edge strategy planning；因此“exact demand analysis已存在”不等于
这一边界已在主线闭合，任务状态必须据实际调用链重新开放。

### 5.4 Spatial与融合机会

Spatial前可以由typed SSA、structured semantics、indexing relation、effect与numeric policy证明哪些依赖边理论上支持
coupled traversal，但真正的TileRegion只能在placement后形成：只有同一physical Tile上重叠的producer/consumer shard
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

Placement后，对每个physical Tile构造local DAG。从该Tile的observable/local outputs反向遍历producer，沿语义可融合且
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

Logical demand先于physical fragment。Movement materialization必须对exact demand与producer ownership求交，证明local和remote
fragments all-and-only覆盖、互不重叠，再生成实际staging、send/recv/wait或DDR store/load。

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

Ready order、worker和completion只在selected physical Tile/Instr结构上物化。Query-local event/calendar可以用于排序、局部
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
+ each stage physical Tile group
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

## 9. SPM、DDR 与 resource admission

Search-time working set可以形成lower bound和排序信息：

```text
live inputs + tile-sized intermediates or retained local shards + outputs
+ representation conversion + communication staging + rotating slots + padding
```

只有已证明必然同时live的集合超过capacity时才能安全早拒绝。Region-local probe使用实际TileRegion lowering和SPM packing，
可快速拒绝固定causal choices；它成功不能替代整卡admission，相关spatial/region/temporal/layout/buffer/order变化后必须失效。
Symbolic footprint必须区分proven must-coexist lower bound、non-binding ranking estimate和缺少坐标时的
`deferred(required coordinates)`：只有第一种超过capacity才能exact reject；估算可放下不能证明packing可行，未证明
coexistence的估算超限也不能拒绝候选。近似或未经typed boundary-faithful proof的外部solver/序列化constraint结果只能排序；
exact局部solver可对准确建模的子问题返回proof/proposal，但selected offset/choice仍须物化并通过typed gate复验。

完整候选完成TileRegion->Instr、selected order/worker和fresh completion后，SPM/DDR planner才从actual roots、control flow、
lifetime和coexistence求validated offsets。Planner区分`Feasible`、`ProvenInfeasible`与`ResourceExhausted`，调用边界还必须
区分internal failure；只有`ProvenInfeasible`或确定unsupported才能拒绝对应causal assignment并形成no-good，资源耗尽或
内部失败是indeterminate，不得删除合法状态。Planner不改变tile、region、traversal、spill、buffer、order、worker或completion。

完整CardExecutable admission还必须原子验证message matching、transport range、per-link/card-shared resource、ABI和observable
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

搜索按成本层次推进，但不把这些层次建成新的长期artifact：

| 层次 | 责任 | 能否成为winner |
| --- | --- | --- |
| relation/semantic gate | exact coverage、effect、numeric、topology symmetry | 否 |
| analytic ranking | resource lower bound与performance estimate | 否 |
| affected TileRegion probe | 实际local lowering/lifetime/SPM packing | 否，仅形成scoped failure或cache |
| complete candidate compilation | CardProgram→TileRegion→Instr→SPM/DDR→transport/resource/ABI | 是 |

完整candidate compilation只能成功形成accepted CardExecutable、proven exact rejection或indeterminate failure；它不能修候选、补默认layout、自动换DDR、
解除fusion或生成新的搜索选择。

## 11. Search algorithm 与质量合同

### 11.1 唯一 owner与两种策略

只有`PhysicalDataflowSearch`语义层可以管理frontier、回溯、budget、incumbent和最终choice。它通过所属IR层的普通analysis
取得合法choices，通过transformation/conversion物化selected choices；不存在统一`Providers`框架或局部selector。

同一choice domain支持：

- small-DAG exact strategy：独立tiny reference domain enumerator不调用production domain builder，证明合法typed域完整；
  production-mechanism flat exhaustive runner再用真实materializer/cost/exact gates完整展开，作为frontier与剪枝oracle；
- real-workload anytime strategy：先建立actual baseline incumbent，在预算内持续产生和编译更好候选。

### 11.2 Anytime构造

真实负载不展开平铺笛卡尔积。推荐过程是：

1. 编译deterministic baseline，立即获得合法incumbent；
2. 生成producer/consumer aligned、compute-balanced、topology-local、reduction-parallel及ordered factorized regular mapping等
   spatial seeds；
3. 对每个seed构造最大TileRegion、最大coupled closure和高效大temporal tile，并用relation-derived reuse signature优先
   direct/local、multicast/broadcast与temporal retention alternatives；
4. exact cheap gates后按cost与结构diversity排序；
5. promising state做affected-region probe，少量完整state做CardExecutable compilation；
6. actual accepted cost更新incumbent；只有proven exact failure才对其causal choices形成no-good，resource exhaustion或internal
   failure保留为indeterminate；
7. 使用large-neighborhood search联合重选一个fused region、fanout、SPM热点、NoC/DDR热点或stage边界；
8. budget耗尽返回best actual accepted candidate。

Large neighborhood必须联合修改耦合choices，例如`spatial+region+coupled+temporal`，或
`stage cut+两侧Tile groups+chunk+movement+buffer+order`；当前单坐标改进只能暂时作为warm-start，不能继续拥有coverage、
family closure或最终winner。

### 11.3 剪枝和trade-off

可以预先使用而不损失合法解的优化包括exact legality、canonical memo、真实topology symmetry、typed causal no-good、
已证明safe dominance和完全等价的partial-compilation cache。

时间上限、bounded diverse frontier、beam、LNS或只完整编译部分高分候选可能漏掉全局最优。它们只能作为Q52实测后的显式
trade-off；完整合法domain仍需lazy可生成，但一次production invocation不承诺访问全部状态。结果分为：finite域完整覆盖且
全局bound闭合的`optimal-certified`；未展开completion仍由完整exact continuation与admissible bound表示的
`feasible-with-bound`；已经永久丢弃或未表示合法completion的`budgeted-feasible`。单纯预算中止不自动降级，只有最后一类
不能声称global bound或最优。DP或局部solver只处理边界和语义完整的typed子问题，不能复制第二套全局relation/lifetime/resource模型。

Q52代表负载以10分钟作为热点与首轮质量检查点，允许继续到30分钟；这两个数是当前验证计划，不进入IR、pass或artifact
协议。确定性work credits、actual compilation数、wall、RSS和incumbent曲线同时记录。

### 11.4 Deterministic baseline

`none`在performance search之外构造单一方案：原始DAG、每个Linalg op独立使用最大合法16-Tile spatial participation、
每op独立TileRegion、op boundary compiler-owned DDR、buffer=1、零fusion和零可选edge action。每个op从其local spatial
extent开始确定性缩减完整temporal tile vector直到actual SPM可放下；relation要求的必要peer fragments只用于correctness，
不搜索route或communication方案。

为了避免当前重复整图materialization，baseline可在独立op/TileRegion边界做可缓存的actual local fit，全部op获得合法
tiling后只进行一次complete CardExecutable compilation。这个优化建立在op boundary DDR和无fusion的独立性上；改变该合同后
局部成功不能继续当作完整证明。最小temporal endpoint仍失败时baseline必须明确失败，不伪造winner。

## 12. Existing mechanics处置与代码边界

### 12.1 处置分类

1. **直接复用**：`CardDAGAnalysis`、`IndexRelation` exact demand、physical access/transfer proof、Card/Tile/Instr
   lowering、SPM/DDR planning、transport/resource/ABI admission。
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

不新增横向`WholeCard/`、`Providers/`、`Evaluation/`、`Facade`、`Projection`或`ActualEvaluator`层。稳定职责分布为：

- TensorProgram analysis：fusibility、iteration partition、streamability；
- PhysicalDataflow analysis/transforms：exact demand、TileRegion formation、coupled traversal、representation、movement；
- Instr analysis/transforms：buffer lifetime、ready order、worker、completion；
- Conversion：TensorProgram到CardProgram、TileRegion到Instr；
- Compiler search：轻量state、dependency invalidation、exhaustive/anytime/LNS strategy和cost ordering；
- CardExecutable compilation/admission：串接既有lowering和exact gates，不实现choice generation。

顶层编译入口只编排`none`或`search`并发布结果；不能继续容纳具体spatial/fusion/layout/buffer算法。每迁移一个choice轴，
同批删除旧owner中的对应字段、hash分支和repair逻辑，不能长期保留两套事实源。

## 13. 当前差距与任务闭环

当前代码已经有baseline正确性证据、exact logical demand analysis和完整Card/Tile/Instr/SPM/DDR/admission机制，但尚未满足本
设计：

- baseline仍因重复完整materialization与late SPM failure在Llama上耗时过长；
- placement production transition仍过早消费physical edge strategy；
- candidate、shortlist、repair与完整编译混在单体synthesis文件；
- spatial domain仍主要是单output axis与连通矩形Tile group；
- region/temporal/fusion/layout/buffer/communication尚未由一个轻量state联合回溯；
- 多个旧机制文件未进入current production build；
- stage pipeline只有历史机制，未接入current common search。

任务按以下artifact闭环推进，具体状态以`tasks/progress.md`为准：

1. 提取无repair的CardExecutable compilation/admission边界；
2. 保留Q49 correctness证据并以Q49.P单列baseline控制流/性能解耦；
3. 修复placement与layout-independent exact demand边界；
4. 建立可提前运行的search core与两层small exhaustive oracle；
5. 依次闭合structured alternatives、spatial partition/placement、TileRegion/temporal/fusion、representation/movement、
   Instr pipeline机制，每项边实现边接入common state；
6. 完成所有维度的search closure和实际fusion gate；
7. 从第一版保留条件式stage-pipeline transition；Q52只在profile后优化其proposal顺序，并加入或加强memo、typed no-good与LNS；
8. fresh workload/package/no-card/board closure。

## 14. Completion Gate

### IR / verifier

- distinct Tile programs可以包含不同op、loop、shape和长度；
- duplicate/unavailable Tile、coverage hole/overlap、illegal reduction merge、cross-Tile/region SPM alias失败；
- same-region independent traversal、coupled traversal、region cut、selective spill、recompute和communication staging可验证；
- selected spatial mapping的iteration coverage、result ownership和edge demand all-and-only闭合。

### Search correctness

- tiny chain、diamond、fanout、reduction、mixed compute/movement和stage+buffer case先由独立reference enumerator证明domain
  coverage，再由production flat exhaustive runner证明frontier/剪枝/winner；
- 构造必须联合改变spatial/region/temporal等多个choice才改善的陷阱，anytime策略能找到actual accepted改进；
- estimate、memo、no-good或dominance逐项开启不改变small-oracle winner；
- `none`和`search`共用完整candidate compilation/admission；
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
