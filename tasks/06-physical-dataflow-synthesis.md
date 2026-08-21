# Card 内 Physical Dataflow Planning 与执行构造

本文是`TensorProgram -> CardModule -> TileRegion -> Instr -> CardExecutable`主线内card-level spatial mapping、
TileRegion formation、temporal tiling、融合、physical representation、movement、buffering、instruction scheduling与候选选择的
唯一设计owner。动态状态、施工顺序和donor迁移矩阵只看`tasks/progress.md`与
`tasks/plans/physical-dataflow-synthesis.md`。

现有或历史physical-dataflow实现只提供可审计的mechanism、proof和test donor，不构成本文合同。能力必须先迁入current owner并
获得production consumer与direct witness，之后才能删除旧实现；旧commit、profile、独立domain test或源码删除都不能代签完成。

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
IR，transformation 靠近被修改的 IR，conversion、memory planning 和 verification 保持独立。selected winner一旦物化，以当前 IR
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
不是 legality shortcut；最终 winner由完整typed plan、legality、cost/bound比较决定，随后actual IR与fixed-capacity lowering只验证并
实现该winner，不反向参与普通候选选择。

## 2. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  GSPMD完成card级分区、05/Q50.S target-independent normalization完成后的card-local TensorProgram；已证明的完整Q/K/V
  attention由一个`wafer.linalg_ext.attention` op表达，`flash_attention`或`flash_decoding`已经是固定graph fact，
  不是本stage候选。block与partition仍由temporal/spatial轴选择，只有winner在Card subtree transaction中展开selected
  Linalg/Tensor/SCF并转换为wafer.tile work。SSA、structured iterator、indexing relation、effect、type、shape和dtype均可验证，
  尚未绑定Tile。
- Current stage responsibility:
  `none`由独立canonical controller从正常TensorProgram构造一个live plan，并只用Q50的pure typed query沿预定义单调
  legalization顺序关闭iteration partition、Tile placement、single-root TileRegion、temporal tile、canonical
  representation/movement、single buffering、order和completion；`search`由独立planning session联合枚举完整性能域。两者只在
  closed `PhysicalDataflowPlan` schema与共同winner commit处汇合；每次invocation只有各自最终plan进入一次
  Card/TileRegion/Instr lowering和actual resource verification。
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
- Done criteria:
  `none`从未预选physical assignment的输入产生accepted CardExecutable，包含初始tile超SPM后缩至合法breakpoint的正例；
  small DAG由独立reference domain composer证明search合法域完整，再由test-only fresh-plan actualization证明
  plan set、剪枝、winner与actual IR一致；真实workload在约定预算内返回full-proof plan并只对winner完成编译与verification的
  best-known CardExecutable；selected IR证明spatial all-and-only coverage与有效region fusion。至少一个代表case还要证明
  coupled traversal、tile-sized intermediate和无无意义DDR往返；selected multi-buffer/stage-pipeline仅在对应候选获选时证明。
```

`optimization=none`是完整的功能基线：它从同一正常上游TensorProgram自行完成目标执行所必需的确定性placement、
single-root region、temporal tiling、canonical representation/movement、single buffering、order与completion合法化，形成一个
可直接发布的CardExecutable，而不是要求调用方先提供已经合法的fixed assignment。`optimization=search`再由本文唯一
candidate-selection owner管理多个性能选择。两者可以共享由current IR导出的immutable typed facts、single-root region
materializer、grouped exact-demand query以及后续lowering、memory planning、verification和package emission，但不能共享会携带
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
structured compute root及其exact operand demand证明必要的non-root support operations；root cardinality按materialization relation映回current
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

- current normalized TensorProgram、fixed semantic root facts、`SemanticRootKey`与typed SSA use identity；
- physical topology 与 available Tiles；
- structured iterator、indexing relation与effect；
- 各层合法选择枚举和实际 materialization 能力；
- target geometry、capacity 和 cost 参数。

它不保存 winner或accepted offset。attention FA/FD通过current op的closed attr进入B/A legality与derived work；
不另传算法字符串、opaque parameter bag或search coordinate。

### 4.2 Semantic root identity

`SemanticRootKey`是一次immutable TensorProgram borrow内的query-local semantic key。它取root到程序可观察边界的
canonical typed SSA路径：路径以entry function result index等ABI有序边界为锚，逐段记录producer result number、consumer
operand number及跨越的typed support relation；存在多条路径时取这些字段字典序最小的一条，其余路径仍参与demand/coverage，
但不复制进identity。显式effect root必须由当前IR中的typed effect/control boundary及其SSA或region relation提供同等路径；
无法从当前IR得到这种路径的operation不是可规划root，应在normalization/verifier处拒绝或先扩IR，而不是用名字或位置补身份。

该key不包含block/operation ordinal、walk顺序、symbol spelling、打印文本、Tile编号、地址或hash迭代顺序。实现可以在同一
immutable borrow内用`Operation *`查找key，并为solver建立owner-private dense index；两者都不能进入candidate equality、
canonical ordering、diagnostic、IR、磁盘格式或跨mutation cache。IR mutation会使整张root-key表失效；clone内对应关系使用
`IRMapping`，materialization后以新IR自身的SSA/typed identity为准。

### 4.3 Search session、winner 与 candidate assignment

一次 search invocation 内的事实分为四类，不能再合并成一个 candidate bag：

1. **immutable session input**：borrowed normalized TensorProgram root、transaction-owned ProgramData view、target facts、cost comparison cohort，以及本次可用的
   typed mechanism；mechanism顺序由唯一current driver静态组合，不是runtime registry、user option或candidate字段；
   source-only relation/semantic facts由policy-free builder建立为session-owned typed problem；同一builder的pass consumer可使用
   operation-anchored MLIR analysis wrapper，但planning session不持有`Analysis *`，也不使用manual epoch或fingerprint；
2. **session control**：candidate frontier、已经通过完整gate的move-only search-local winner、global work/budget accounting和coverage/
   lower-bound evidence；这些事实不属于任一candidate；
3. **candidate assignment**：只保存当前已经实现且不能从current IR与其它选择重算的typed choices；
4. **derived query-local facts**：exact demand、ready/live set、lifetime、resource calendar、SPM high-water、cost estimate和
   lower bound，按IR borrow、target facts和相关assignment重算，并以声明all-and-only实际读取字段的
   `ObservedDependencyKey`在本session内失效。

compiler driver拥有policy routing、planning session、frontier/budget和唯一winner handoff；它不执行leaf rewrite。pass与named
subpipeline只在winner commit后对actual Card/Tile IR完成既定IR→IR变换，不持有candidate或选择winner。winner materialization
开始前关闭source borrow及全部planning memo；新Card subtree上的analysis一律从新IR重算，不能复用source `AnalysisManager`结果。

`ObservedDependencyKey`不是通用candidate bag。每类memoized query定义自己的named typed key，只包含该query实际读取的
`SemanticRootKey`、target fields和assignment fields；cache hit比较完整typed value，hash只作lookup。key缺字段是合同错误，
多放字段只会降低复用，不能改变query结果。

Q49.P accepted baseline不进入search invocation、candidate frontier或winner，不被重编码为search assignment，也不把其canonical
functional choices当作未施工轴的默认值。mechanism availability、proposal priority、work accounting、diagnostic statistics和
search-local winner同样不进入candidate identity。预算内没有accepted search candidate时返回typed failure；matched baseline只由
外层以独立`none`编译事务产生。

`ProgramDataHandoff`继续由外层compiler transaction唯一拥有，不复制进baseline或planning state。planning阶段不存在落选
actual executable；只有最终plan调用一次Q50.0并引用已验证program identity/range，accepted后才由Q59同一outer transaction把
handoff随唯一发布结果向下游移动。search不得为任何未选plan构造CardModule、写package或提交目录。

Q51终态的candidate assignment包含下列typed choices；Q51.Core不预声明尚未由对应Q50 mechanism实现、验证和消费的字段。
每个Q50 checkpoint在同一current aggregate中加入本轴的named typed field、ephemeral transition、canonical encoding和精确
失效关系，不能使用`any`、字符串tag、opaque payload、通用provider registry或placeholder optional field提前占位：

```text
StageId  -> members / Tile group
SemanticRootKey -> iteration partition / physical placement / reduction role
RegionId -> mandatory root-work members
ExecutionInstanceId -> required coverage or explicit replica / top-level or consumer-nested placement
UseFragmentId -> stored region value / direct nested value / cross-region boundary
MovementBoundaryId -> local refetch / DDR / NoC / collective realization
LoopId   -> complete temporal tile vector / finite wave-loop nesting and order within selected traversal
ValueId  -> selected physical representation
TileId   -> buffer recipe / instruction order / worker / completion choice
```

typed transition只用于从parent原子构造child；child保存apply后的assignment，不保存transition history、proposal ordinal、
allocator feedback history或repair path。状态不保存 raw `Operation *`、estimated SPM、汇总 bytes、fragments、resource
calendar、makespan、actual offsets 或repair history。winner commit后以 IR 为唯一事实源。

### 4.4 失效依赖

不同机制不互相调用或维护局部 winner；跨层耦合只通过明确依赖失效表达：

```text
normalized semantic root或FA/FD fact改变
  -> complete physical planning problem失效并从SpatialState重建

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

对 structured op 的迭代域 `D = (i0, i1, ..., in)`，为每个parallel/reduction iterator选择typed regular block partition，
并为每个logical shard形成精确domain。current schemes包含balanced part count和uniform extent+tail；两者extensionally相同
时canonicalize。以GEMM为例：

```text
C[M,N] += A[M,K] * B[K,N]
P_M * P_N * P_K <= available Tile count
```

`P_M/P_N`产生output-parallel shards；`P_K`产生partial reductions。非整除维度使用exact intervals/relation表达tail，不能
要求整除、丢元素或产生非法重叠。M/N与K同时切时，每个parallel output coordinate形成独立merge group，而不是整个node共用
一个merge Tile。必须证明：

- parallel shards all-and-only覆盖原迭代域；
- 除selected recompute外没有重复执行；
- reduction partitions完整且存在显式合法merge。

初期可先完整支持multi-axis block partition；未来若支持cyclic/block-cyclic，应扩展typed relation和lowering，不增加
shape/name matcher。

### 5.2 Tile placement

logical shard coordinate 与 physical `tile_id` 是不同对象。`SpatialPlan`只保存partition schemes、logical mesh到available
topology的injective embedding和per-reduction-group merge Tile；exact execution domains由query关闭成ephemeral
`SpatialAssignment`，不写入candidate state。Placement决定participant subset、merge endpoints和独立branch的Tile groups；它考虑producer/consumer同Tile对齐、拓扑距离、负载平衡
和NoC热点，但只确定端点，不选择route、layout、buffer或transport implementation。

紧凑矩形、自然mesh、使用全部16 Tiles可以作为高优先级seed；它们不是legality条件。搜索域必须惰性包含合法的participant
count、多轴mesh、reduction partition、remainder和非对称placement。只有actual topology adjacency和全部immutable Tile facts
上的verified automorphism才能canonicalize；分析只找到identity时保留raw states，不按4x4外观或Tile编号猜symmetry。

高优先级proposal可以先做ordered factorized regular mapping：把一个或多个logical iterator factor映射到typed physical
topology dimensions。只有axis/factor nesting实际改变extensional partition relation、logical-mesh embedding或physical
placement时才形成不同typed choice；等价factorization的生成顺序必须canonicalize。它只是完整placement域的构造顺序，
不能把连续矩形、二维mesh、full occupancy、偶数participant或某一种factor order提升成合法性；未映射的local extent继续交给
temporal选择，不能由spatial proposal暗中固定tile或loop order。chain/tree/general-DAG DP只产生proposal顺序和admissible
lower bound；未关闭fusion/layout/movement/buffer/schedule时不得用spatial-local Pareto metrics删state。

### 5.3 Exact producer/consumer demand

不同op可以选择不同mapping。exact-demand analysis消费closed `SpatialAssignment`：每个node的all-iterator execution
shards、logical shard-to-Tile embedding，以及空间reduction时按output-domain piece给出的merge placement。final result
ownership、partial contributions和dependency demand均从assignment与current structured IR派生，不能作为另一份可矛盾输入，也
不能从result axis、单个`shardDimension`、participant count或balanced一维矩形反推。

analysis先在最近的`func.func`上从SSA、仓库pinned `mlir::linalg::LinalgOp`的
`getIndexingMapsArray()`/`getIteratorTypesArray()`、DPS/Tiling/reduction interface和pure tensor transform semantics
建立可失效、可重算的relation graph。主查询以consumer operand为单位，对每个destination shard执行：

```text
consumer exact iteration domain
  -> IndexRelation.image
  -> operand exact demand
  -> 沿pure tensor SSA对所有data-carrying operands反向传播
  -> structured result / program input / constant boundaries
  -> 与merge后的final result owners求交
```

同一producer经多条path或同一support op的多个operand到达时取exact union，不判为ambiguous；multi-result、fanout、fanin、
DPS init、insert/pad empty branch和显式structured init root按实际SSA/result/operand role保留。query返回每个boundary的
required source domain、每个destination的eligible final owners和同一份`OperandReconstruction`；materializer不得再走第二遍
support matcher。

普通parallel partition产生互不相交并覆盖完整result的final owners。显式replication保留全部等价owners供movement选择source。
空间reduction则把具有相同output-domain piece、不同reduction coordinates的shards组成一组：每组必须收齐完整reduction
fiber、具有一个显式merge Tile和typed algebra/init rule，merge后只有该Tile是下游final owner。M/N与K同时切分时可以有多个
output groups，禁止用一个node-wide merge Tile或把partial contributions冒充完整result replicas。空间partial/merge需要可由
winner materializer调用的typed partial-building mechanics：普通reduction使用`PartialReductionOpInterface`，coupled attention使用
source-owned coupled description与selected Linalg/SCF builder；同Tile保持原顺序的temporal reduction tiling不使用partial owners，
由temporal层单独证明。

coupled attention的`ReductionMergeRequirement`保留source interface给出的component kind、indexing map、element type、merge和
finalization rule，并为每个output group及每个contribution分别给出Maximum/Sum/Accumulator的exact domain。三个components属于
同一requirement；contribution不伪造source op result，只有merge后的attention result进入`FinalResultOwner`。

一个all-and-only覆盖原iteration domain的合法assignment最终产生完整logical results，所以不同producer/consumer Tile sets不会
因为cross-op demand产生placement no-good；exact demand只决定后续redistribution。assignment不覆盖或merge group不完整是
compiler contract error。physical descriptor、route、SPM或schedule失败属于后续physical coordinate，不能反写
本层logical relation。

该边界必须通用支持以下relation，而不是把当前fixture或physical carrier限制升级成协议：

- reduction的parallel/reduction partition、input/init demand、partial contribution和merge role；
- broadcast/projection的many-to-one image，重复consumer使用不膨胀唯一source set；
- convolution、pooling和supported reduce-window等affine window的kernel、offset、stride、dilation、halo及显式pad/fill pieces；
- non-unit-stride slice/view、permutation、collapse/expand等可组合static relation；
- Presburger union、tail、pad/window分段和组合relation形成的有限multi-piece set。

这些声明支持的类别不能因为不是dense rectangle而返回unsupported。`IndexRelation`以MLIR Presburger relation/set为精确语义，
同时保留构造时证明的affine、strided、row-major和finite-piece normal form；composition/image/intersection在执行前估算piece
work。production支持路径不允许降为generic `isEqual/isSubsetOf/subtract`碰运气，因为pinned Presburger的disjunct product和
integer solver work不能由操作后的variable/disjunct计数约束。超过统一compiler relation-work limit返回typed indeterminate；
真正dynamic/non-affine或未定义typed semantics返回unsupported；两者都不是placement no-good。

query结果是named typed variant：`satisfied(ExactDemandProof)`、`unsupported semantics`、`indeterminate resource exhaustion`
或`compiler contract error`。`FailureOr + diagnostic string`不能压成legality bool。重复消费的relation graph使用标准MLIR
analysis invalidation；assignment相关结果保持query-local。不用手工`IREpoch`或每次query fingerprint整棵function。

`satisfied`结果不包含layout、encoding、bytes、dense fragment、local/remote action、route、buffer、send/recv、fusion或
resource schedule。Representation/movement materialization稍后才对exact demand与ownership求交并将其有限分解为physical
pieces；dense rectangle、descriptor、route或transport暂时表达不了，只能拒绝对应physical assignment，不能反写删除logical
placement。deterministic baseline必须为已声明支持的exact set提供一条canonical correctness carrier；它与完整search
representation/movement域都消费同一proof，不另建压缩版demand。

analysis result只在current immutable planning session有效；stable key由structured result/consumer operand identity、完整
execution domains和reduction placement构成，只有extensional proof允许投影。结果不依赖pointer/hash/Tile枚举或并行结束顺序，
默认编译路径不采集或打印relation统计。

### 5.4 Spatial与融合机会

Spatial前可以由typed SSA、structured semantics、indexing relation与effect证明哪些依赖边理论上支持
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

explicit recompute
  为一个consumer/version显式再次执行pure producer work，其它uses仍由自己的selected version满足
```

前者已经消除DDR，但可能把完整local shard materialize到SPM；后者只保留当前tile/halo/partial result，通常显著缩短
lifetime并降低SPM压力。因此候选优先级是：

1. 同一最大TileRegion中的最大coupled closure；
2. 必要时解除个别coupled edge，但保持同region independent traversal；
3. region仍不可取时才cut，并显式DDR materialization。

不存在“不同TileRegion但同一个shaped value继续SPM驻留”的选择。`SPM residency`不是独立的上层domain，而是最终
TileRegion、Instr order、buffer slots、lifetime和accepted allocation共同证明的事实。

### 6.2 TileRegion formation

Placement后先从Q50.A唯一operand reconstruction为每个`(semantic root, Tile)`派生singleton `RootRegionWork`：selected
iterator cells、同root contribution/merge、pure tensor support DAG及在其它structured result/program input/constant处的typed
boundaries。同一root/Tile的work不任意拆分，shared support只出现一次；不再次递归全operand恢复另一份closure。

随后region grouping在每个Tile的local DAG上形成connected root partitions；完全无依赖的components合并不会减少movement且
只会扩大lifetime/capacity约束，因此由严格dominance保持分离。region membership、producer execution instance和consumer-use
binding是三个不同对象：mandatory或显式replica producer work分别选择top-level或consumer-nested execution，每个nonempty exact
use fragment再绑定stored region value、direct nested value或cross-region boundary；same Tile/same region本身不代签fusion。
Region formation必须处理：

- boundary inputs/outputs和effect顺序；
- fanout的所有consumer和operand/ownership fragments，而不是只给node pair写`fused=true`；
- reduction partial/merge；
- local/remote demand的精确拆分；
- unrelated components不能仅为“region大”而获得虚假收益。

Region membership、execution instances、use bindings和temporal shape都是query-local选择；production不先物化singleton
CardModule再融合。只有完整winner直接构造一次真实`wafer.tile.region`、loop和SSA；test-only可单独actualize singleton leaf。
outer transaction在winner-owned source Module中建立可整体擦除的CardModule subtree，验证成功后才消费旧TensorProgram，
不clone Module/Func/DAG或replay其它Tile body。

### 6.3 Coupled closure

从region内一个consumer use fragment出发，使用Q50.A exact demand和result/operand relation求producer result domain与
iteration preimage。只有effect、coverage和structured semantics都可证明时，producer execution才允许nested于consumer；依赖具体
temporal vector/order的结论返回typed requirement给Q50.E。显式recompute不是movement action，而是D拥有的额外producer
execution instance；其operands继续通过Q50.A同一relation engine派生。Fanout必须逐use fragment绑定同一个兼容version或显式不同
replicas/boundaries；不能遗漏任一consumer，也不能让first materialized tile决定其它uses。

有效nested actual IR必须证明producer位于consumer temporal traversal内、需求slice all-and-only覆盖、intermediate为tile-sized
而非完整local shard，且不存在中间DDR round-trip；stored binding必须有独立producer traversal和region-local SSA/buffer version；
replica必须有显式额外producer work。RegionPlan/edge bool不能代替这些witness。

### 6.4 Complete temporal tile vector 与 wave-loop order

Temporal state锚定Q50.D产生的`ExecutionInstancePlan`而不是node-wide default：top-level exact work pieces以及由parent wave relation
形成的nested invocation classes各自有完整vector；parent决定nested producer被请求的work，但其内部reduction/window等iterators仍可
继续temporal切分。不同Tile实例可选不同vector，extensionally相同的scope只共享domain query，不强迫assignment相同。每个state覆盖
该scope全部iterator，并选择语义合法的wave-loop nesting/order：

```text
[parallel tiles..., reduction tiles..., batch/head/channel/window tiles...]
+ wave-loop nesting / order within selected traversal
```

它不是单一block size，也不另设与主向量脱节的reduction字段。对可tile iterator，`1..local extent`每个positive size都属于
exact-tail语义域；target-efficient、wave-count、alignment、transaction、halo或footprint breakpoints只用于proposal/interval
split，不能缩小合法域。Q51以integer intervals惰性branch到concrete points，避免预建全整数Cartesian product。
只有能证明生成相同actual traversal、exact demand、lifetime和tail的排列才能canonicalize；合法order从SSA loop-carried state、
effect/control和nested-execution precedence DAG的全部topological sorts惰性生成，默认loop order不能删除会改变reuse、SPM或最终resource
schedule的合法组合。

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

通信方案不以`Ring/Tree/Direct`枚举为核心。每个explicit boundary从exact chunks的初始Tile、目标Tile和可用local combine关系建立
有限`TileCommunicationProblem`，候选是transfer/combine action DAG；每个terminal payload必须有all-and-only origin/domain证明。
Ring、binomial/double-tree、recursive exchange、二维row/column、Bruck/pairwise和tree-packing只是构造该通用DAG的
topology-aware proposals。关闭或改变proposal顺序不能改变有界穷举oracle覆盖的exact domain；真实规模负载允许在统一budget内返回best-known plan。

片内topology事实分三层：physical directed NoC adjacency、available endpoints间的Direct DTE capability，以及compiler选择的
software relay/action graph。current Direct DTE只公开end-to-end target-routed unicast，内部route不透明；因此physical mesh可以产生
shortest-hop、cut crossing和placement/proposal信息，却不能生成exact per-link contention或channel order。只有target未来显式提供
deterministic/programmable route与channel facts时，resource schedule才消费exact directed-link use和router deadlock proof。
software relay、chunk readiness、send/recv、buffer release和local combine的死锁由actual action/event wait graph独立证明。

规则4x4 mesh可以共享经topology automorphism证明等价的domain/proposal计算，但不同Tile的work、buffer和event仍分别选择和物化；
unavailable Tile、非对称需求或不同resource facts会缩小automorphism，不能继续套固定16-Tile ring或相同assignment。

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

只有已证明必然同时live的集合超过capacity时才能安全早拒绝。pure foundation从A–E facts建立minimum storage demands与
must-coexist graph；single value或任一已证明weighted conflict clique超过capacity才返回causal exact rejection。找不到maximum
clique、representation/lifetime坐标缺失或work耗尽分别返回sound lower bound、Deferred或Indeterminate，不clone/lower/probe
TileRegion，也不调用packer。Symbolic footprint必须区分proven lower bound与non-binding estimate：估算可放下不能证明packing
可行，未证明coexistence的估算超限也不能拒绝候选。

G--K全部关闭后，full feasibility从typed physical versions、communication actions、buffers、execution structure和event/resource
schedule构造与actual memory planner同形的SPM/DDR packing及DTE/FSM/descriptor/field-limit问题。它先尝试可验证的确定性placement，
只有完整contender的fast witness失败且调用方已从统一work allowance预留资源时才运行exact packer；validated placement证明存在，
`ProvenInfeasible`证明该完整causal assignment不可行，`HeuristicNoFit`、`ResourceExhausted`或timeout均为Indeterminate。partial state不
调用packer，full proof也不把offset写进candidate。

winner一次性物化后，SPM/DDR planner从actual roots、control flow、lifetime和coexistence重建同一normalized problems并求真实offset；
actual问题与planning问题不同、或full proof存在却不能pack，属于planning/emitter/lowering合同错误，不能返回search换candidate。
Planner不改变tile、region、traversal、spill、buffer、order、worker或completion。

完整CardExecutable verification还必须原子验证message matching、transport range、known-link/card-shared resource、ABI和observable
completion。任一失败由outer CardModule transaction整体回滚，不允许提交部分Tile、返回planner重选winner或复用失败IR中的offset。

## 10. Cost 与分层候选编译

### 10.1 Cost边界

Hard legality/capacity与performance estimate严格分离。Cost model复用current work collectors和target参数；缺失参数按同一
comparison cohort统一删除该term，不能candidate-local按零或无穷。Estimate只用于排序、diversity和best-known选择；除非
某lower bound或dominance逐项证明安全，否则不能删除合法候选。

至少保留per-Tile issued work、critical path、DDR/SPM/NoC bytes、message/descriptor/instruction/wait、
Tile idle与pipeline prologue/II/epilogue。SPM high-water只用于capacity/headroom与诊断，不作为Pareto目标；working-set
带来的性能差异通过movement、spill、buffer、waves与issued work计价。Overlap只有actual buffer/resource独立性可证明时
才计入，不对整程序无条件相加或取`max`。

NoC内部route不透明时只保留endpoint work、minimum-hop/link-byte与cut lower bound，逐link load为Unknown；target提供exact route
后才把directed-link load加入同一cost/resource tuple。Unknown不能按零处理，也不能据此声称两个transfers无contention。

分层resource projection复用现有target facts，把actual或partial typed assignment依次投影到compute unit/worker、相关local
SPM resource、DTE endpoint、已证明的directed NoC links与DDR channel/engine。资源集合相交只形成contention estimate/排序信号，不相交可优先提出
并行proposal；是否能重叠及其cost仍由actual event/resource semantics判定。Nominal bandwidth split、解析关键路径或其它近似
模型只能用于ordering；没有actual buffer、issue/wait、event与resource independence时，不能据此签发completion、overlap、
legality或admissible bound。

### 10.2 分层编译成本

搜索按成本层次推进，但不把这些层次建成新的长期output：

| 层次 | 责任 | 能否成为winner |
| --- | --- | --- |
| relation/semantic gate | exact coverage、effect、topology symmetry | 否 |
| analytic ranking | resource lower bound与performance estimate | 否 |
| planning legality/resource query | 从current IR、target facts和完整typed assignment计算layout/movement/lifetime/resource约束、proof与bound，不构造IR | 否 |
| selected winner commit | winner assignment一次性CardModule→TileRegion→Instr→SPM/DDR→transport/resource/ABI | 是 |

production search不能把winner选择建立在反复complete candidate compilation上。selected winner commit只能形成accepted
CardExecutable或typed compiler/unsupported failure；它不能修候选、补默认layout、自动换DDR、解除fusion、返回search继续尝试下一个点，
也不能生成新的搜索选择。若只有lowering后才能判断的legality长期成为正常candidate rejection，说明planning IR/query合同尚未闭合。

## 11. Search algorithm 与质量合同

### 11.1 唯一 owner与两种策略

只有`PhysicalDataflowSearch`语义层可以管理candidate frontier、回溯、budget、incumbent和最终choice。它通过所属IR层的普通
analysis取得合法choices，通过transformation/conversion物化selected choices；不存在统一`Providers`框架或局部selector。
search kernel本身不生成spatial、region、temporal、layout、movement、buffer或schedule choice，也不预先定义这些轴的
placeholder schema。每个mechanism进行pure domain/query并返回named typed transition、legality、cost或bound；kernel只维护
deterministic frontier、stable dedup、typed outcome和search-local best plan。planning阶段不apply mutable IR。

先收敛上述control contract，但不实现只靠mock domain运行的空Core。Q50.S先交付两条policy共用的attention semantic IR、
algorithm normalization和read-only planning description；Q50.B–K再按依赖建立真实typed domain/query/apply，Q50.F/J分别经过
foundation与full closure；每轴加入independent reference enumerator并迁移donor能力。`spatial-domain`与`exact-demand-boundary`形成
首批真实domain后，`search-control-foundation`让explicit public `search`进入new owner；缺后续axis时返回typed incomplete，不调用baseline或commit。
此后每个domain work item同批扩state/transition与production consumer，`full-feasibility`后由`search-control-closure`闭合control/coverage，随后Q51以test-only flat
exhaustive oracle、actual digest/winner correspondence和single-winner source-to-package闭合。旧control branch在new owner接管时切除，
不能保留第二controller，也不能让foundation伪装完整search。

终态调用关系固定为：

```text
CompilationOptions::search
  -> new PhysicalDataflowSearch session
  -> consume Q50.S-normalized fixed semantic roots
  -> Q50.B–Q50.K typed domain/query/transition
  -> Q50.F and per-axis planning legality/cost/bound queries
  -> search-local winner assignment
  -> winner一次性actual CardModule materialization
  -> Q50.0 CardExecutable compilation / verification
  -> one accepted CardExecutable winner
  -> Q59 target/package transaction
```

旧search与Q49.P baseline都不处于这张search调用图；search没有隐式fallback。

同一choice domain支持：

- bounded-DAG exact strategy：独立有界穷举reference domain enumerator不调用production domain builder，证明合法typed域完整；
  同一机制另由真实规模整除/非整除矩阵证明production路径；
  test-only flat exhaustive runner可以逐点调用真实materializer/exact gate作为oracle，但不得成为production编译路径；
- real-workload anytime strategy：尽快建立首个合法search plan，在预算内持续改善typed winner assignment，预算结束只commit一次。

### 11.2 Anytime构造

真实负载不展开平铺笛卡尔积。Q51 closure建立的non-lossy基本过程是：

1. 从immutable TensorProgram和target facts建立search session，不调用Q49.P或接收baseline executable；
2. 生成producer/consumer aligned、compute-balanced、topology-local、reduction-parallel及ordered factorized regular mapping等
   spatial seeds；
3. 对每个seed构造最大TileRegion、最大coupled closure和高效大temporal tile，并用relation-derived reuse signature优先
   direct/local、multicast/broadcast与temporal retention alternatives；
4. exact cheap gates后按cost与结构diversity排序；
5. scoped planning query计算必要legality、resource bound和cost，不构造CardModule/Instr；
6. 首个完整合法plan建立search-local winner，后续plan按同一cost cohort与完整semantic tie-break更新winner；只有pure query给出的
   proven exact failure及future-compatible causal scope才能形成no-good，resource exhaustion或未分类结果保留为unresolved；
7. budget耗尽选定best search plan，随后只materialize/compile一次；没有完整合法plan则typed failure。

Q51.Core固定一个确定、不会丢state的typed-continuation best-first机制、direct causal forbidden assignments、admissible
branch-and-bound和coverage合同，但不固定Q52 production默认allowance或有损策略。Q52依据fresh profile调整priority、memo、safe
dominance、component DP和large-neighborhood search。Large neighborhood必须联合修改耦合choices，例如`spatial+region+coupled+temporal`，或
`stage cut+两侧Tile groups+chunk+movement+buffer+order`；当前单坐标改进只能暂时作为warm-start，不能继续拥有coverage、
family closure或最终winner。

### 11.3 剪枝和trade-off

可以预先使用而不损失合法解的优化包括exact legality、canonical memo、真实topology symmetry、typed causal forbidden assignment、
已证明safe dominance和完全等价的pure-analysis cache。

时间上限、bounded diverse candidate set、beam、LNS或只完整编译部分高分候选可能漏掉全局最优。它们只能作为Q52实测后的显式
trade-off；完整合法domain仍需lazy可生成，但一次production invocation不承诺访问全部状态。结果分为：finite域完整覆盖且
planning objective/global bound闭合的`objective-optimal`；未展开completion仍由完整exact continuation与admissible bound表示的
`feasible-with-bound`；objective Unknown/incomparable的`feasible-unranked`；缺bound或永久丢弃合法completion的
`budgeted-feasible`。DP或局部solver只处理边界和语义完整的typed子问题，不能复制第二套全局relation/lifetime/resource模型。

Q52代表负载以10分钟作为热点与首轮质量检查点，允许继续到30分钟；这两个数是当前验证计划，不进入IR、pass或output
协议。确定性work credits、actual compilation数、wall、RSS和incumbent曲线同时记录。

### 11.4 Deterministic baseline

`none`在performance search之外构造单一方案：原始DAG、每个structured compute root独立使用最大合法Tile participation、
每root独立TileRegion、root boundary compiler-owned DDR、buffer=1、零fusion和零可选edge action。同一Tile允许按完整
semantic tie-break承载多个独立TileRegion，但一个baseline TileRegion只能有一个structured compute root。shape/index/view、
target-local materialization等没有独立structured DAG identity、不能独立调度且被exact operand demand选中的op才属于non-root support；显式
Fill、Reduce或DPS init producer只要是另一个structured DAG node就仍是第二root，不能借support名义并入。root cardinality按
typed materialization relation计算，因此一个root lower成多个compute/instruction op仍只算一个root。每个root从其
local spatial extent开始确定性缩减完整temporal tile vector，逐coordinate重建Q50.F的typed storage/interference/resource problem，
直到取得经验证的plan-level full proof；relation要求的必要peer fragments只用于correctness，不搜索route或communication方案。

“最大合法Tile participation”不要求虚构一个spatial axis：root没有映射到result的parallel iterator时，baseline构造
all-factor=1、单参与Tile、完整result domain的typed unpartitioned coordinate；这只是该root没有parallel轴时的退化，不是
baseline全局只用一个Tile。纯reduction由同一temporal breakpoint机制缩减；
reduction spatial factor大于1所需的partial ownership和显式merge仍由Q50.B引入，不能因该search机制尚未实现而拒绝功能baseline。

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
resolution遍历baseline合同内的合法fallback；每个closed coordinate只运行Q50.A/F及其它必要的pure typed query，不创建IR。
`FullFeasibilityProof`关闭当前plan，带direct typed witness的exact rejection才推进到下一个canonical coordinate。query-local trial和fit结果
只是合法化过程，不是Q51 search candidate：controller不评分、不维护incumbent/candidate family、不保留用于比较的备选方案，
也不得创建或调用search candidate/state、search-oriented domain/ranking evaluator、grouping materializer和feedback repair。
它只复用typed iterator/topology事实、single-coordinate legality/materialization机制和上述logical demand/coverage query。
任一接收option列表/domain并通过propagation、recursive CSP、backtracking或其它assignment solve返回一个结果的helper，即使输出
确定、只取第一个或命名为`policy-free`，仍属于search，禁止出现在`none`的transitive call graph。baseline始终只有一个live
coordinate：直接构造当前canonical spatial placement，对每条edge只查询这一对已关闭的producer/consumer shard；direct typed
rejection只能触发预定义、单调、不分支且不回溯的functional legalization transition，旧coordinate随即销毁，不保留alternative、
no-good、score或winner比较。不得先生成每个node全部iterator-axis × connected-rectangle placement options，也不能调用physical
edge strategy/evaluator来决定logical trial是否合法。Q50.B未来完整placement域由自己的惰性mechanism拥有，baseline不复用整域
generator。

controller产出的窄immutable selected assignment是feasibility resolution的**输出**，不是baseline入口的前置条件；baseline与
search可以在这个policy-free已选事实或actual IR边界汇合，而不是共享candidate wrapper。该结果只含materialization所需的
per-root placement、显式singleton region boundary、完整temporal vector以及已经由baseline确定的canonical
representation/movement/buffer/order/completion事实，不含score、derived metrics、stable ordinal、transition/failure history或
controller flags；materializer不得自行补grouping/default choice。

logical query对reduction、broadcast、affine window、stride/view和multi-piece返回`satisfied`后，baseline的canonical
correctness carrier必须把exact set有限分解为all-and-only DDR/必要peer movement并继续完整编译；dense-only fragment或单条
descriptor表达失败是baseline carrier的实现缺口，不是logical placement failure。完整search可在之后枚举其它representation、
route、retention或collective，但不能因此改变Q50.A logical outcome。

CardModule构造前先从同一closed coordinate的全部Tile root execution domain执行一次multi-destination反向需求传播。worklist项是
current SSA value、semantic Tile和exact integer domain；同一support operation的relation只建立一次，再对各Tile domain应用。
ordinary unary/view support通过`IndexRelation`求exact preimage，`tensor.insert_slice`按被插入区域把需求严格分解到source与
destination；到structured producer result立即停止并形成typed boundary obligation，不继续回溯producer body。exact-empty分支
没有physical action，不能伪造成edge strategy、空fragment或缺失carrier。所有非空obligation必须在任何IR mutation前由已选
local/peer/DDR carrier all-and-only覆盖；缺口、重叠、无法exact传播分别保持typed rejection或indeterminate。验证通过后按稳定
拓扑序一次性构造各Tile最终root、必要support op和endpoint；send-only endpoint不引入remote consumer。禁止先复制result/edge
endpoint的递归SSA闭包，再通过support clone、rebuild、replay或失败后补边修正范围。

上述传播的正确性合同不是“在代表模型上能跑通”，而是对每个current value `V`和exact需求集合`D`证明：物化值
`M(V,D)`在`D`上的投影与源程序`V`相等，`D`外没有consumer读取。每个被接纳的pure tensor support operation必须由同一个
短生命周期typed transfer recipe同时给出output demand、各tensor operand的exact read demand和reconstruction动作，并满足：
只要每个operand在其read demand上与源值相等，reconstruction就在output demand上与源operation相等。query与apply共同消费
该recipe；不能让analysis和materializer各自维护op matcher。recipe只在同一immutable IR epoch内存活，不进入IR、candidate
identity或跨stage cache。

`tensor.insert_slice`的transfer由overwrite语义唯一确定：设source写入result的区域为`W`，则对任意需求`D`，source demand为
`inverse(D ∩ W)`，destination demand为`D − W`；两者经reconstruction形成`D`的all-and-only覆盖。destination demand为空时
可以使用未初始化tensor作为仅承载source insertion的容器，但必须由该集合等式证明，且最终coverage verifier拒绝任何对容器
其它区域的读取。`extract_slice`及一元view/reshape类operation由exact `IndexRelation` preimage给出transfer；effectful、alias
不明、多输入语义未证明或只能近似传播的operation在mutation前返回typed unsupported/indeterminate，不能退回全operand闭包。

整图结论由acyclic SSA上的结构归纳得到：function argument/constant和已验证carrier是归纳基；structured producer result是
停止边界；每个support recipe提供局部归纳步。structured root在Tile上的execution domain通过operand indexing relation产生
read demand，所有operand在这些domain上相等即可推出root在该execution domain上相等。temporal wave必须按稳定顺序all-and-only
覆盖root execution domain并显式携带reduction accumulator；没有partial-result merge时reduction spatial factor仍只能为1。
任一新support semantics只有同时提供上述transfer/reconstruction证明和定向law test后才能进入支持集合。

多Tile通用性由同一传播参数化，而不是复制16份分析：一次查询同时seed全部`(root, Tile, execution domain)`，以
`(SSA value, semantic Tile)`为key累积domain union，只传播新增差集；relation对象按operation/result/operand建立一次。当前
single-block functional TensorProgram可按反向SSA拓扑序一次处理，未来region/control-flow若不能提供等价typed transfer则保持
unsupported。复杂度和work counts按非空value/Tile demand、relation application及carrier fragment计数，不能用线程并发掩盖
16次完整DAG walk。

temporal feasibility从root的完整local iterator extent开始，在iterator/indexing semantics、tail、target vector/alignment及
最小合法粒度共同定义的有限breakpoint lattice上按semantic全序推进；parallel与reduction iterator都由同一typed tiling contract
给出可用尺寸。每次trial
都重新推导全部operand slice、stride/dilation halo、result/init/accumulator、temporary、materializing copy、movement staging、
alignment/bank和实际lifetime，不得只按output tensor字节数缩放估算。plan-level exact SPM rejection时只沿direct typed witness影响的
合法维度进入下一组更小breakpoint，重新构造同一typed resource problem；
序列必须覆盖到target允许的最小合法temporal vector。第一个fit立即成为该root的resolved assignment，不为了性能继续比较
其它fit；trial数不得写入candidate proposal/fusion/layout/buffer统计。

capacity结论来自Q50.F按当前closed coordinate构造的minimum-storage/interference与full-coordinate resource problem，不构造
root/Tile/function scratch IR，也不调用Q50.0。capacity witness可以包含一个冲突集合，但必须把all-and-only storage/lifetime owner
直接关联到当前single root的result/operand demand和temporal assignment；unsupported witness必须命名无法表达的typed relation。
两者都不能按相同type/shape猜测producer、解析diagnostic字符串或同时缩减多个歧义match。只有完整proof产生的exact rejection允许
controller推进确定性fallback；heuristic no-fit、solver耗尽与indeterminate必须终止，不能伪装成输入unsupported或“none没有方案”。

全部root及baseline真实使用的canonical G--K component facts获得resolved plan并通过Q50.F closed-plan core proof后，baseline只物化一次完整CardModule，并只进行一次
complete CardExecutable compilation。Q50.0从actual IR重建normalized resource problems并分配offset；与plan proof不一致或此时
出现capacity failure属于compiler bug，不能继续fallback。局部成功的安全性建立在“一root一TileRegion、跨root shaped dependency显式DDR、无fusion、sequential
single-buffer execution”结构不变量上；改变任一项后局部成功不能继续当作完整证明。只有target允许的最小合法temporal
vector连同其它canonical fallback都被exact证明不可行时，才能返回typed capacity/unsupported；声明支持的输入在完整gate才
暴露可由更早baseline legality query发现的失败，属于compiler contract缺口，不是正常`none`结果。没有search candidate绝不能
成为失败原因。
accepted CardExecutable就是baseline semantic result；不得随后构造未被output消费的`StaticSchedulePlan`/duration estimate，
也不得让普通compile无条件打印或保存Tile IR snapshot。显式IR inspection只在最终accepted output上按请求执行一次，不参与
admission或package语义；search不消费该baseline inspection。
`actual_fused_edges == 0`只是必要结果，不能替代对每个baseline TileRegion structured-root cardinality、跨root DDR
store/completion/load和search-policy调用闭包的结构检查。

## 12. Existing mechanics处置与代码边界

### 12.1 处置分类

1. **直接复用的独立边界**：Q50.0 move-only CardExecutable compilation result、Q50.A `IndexRelation` exact demand、
   `StructuredDAGAnalysis`与target topology facts，以及Card/Tile/Instr lowering、SPM/DDR packing、transport/resource/ABI verifier。
   这些对象必须能在不include或构造旧candidate/search owner的前提下单独调用。
2. **仅作算法与proof素材**：attention/decode graph proof与recurrence、placement option推导、event/resource scheduling、edge carrier、
   selected-buffer、movement、layout、NoC/collective和software-pipeline实现。Q50.S graph normalization及B–K终态typed mechanism
   重新定义输入输出后，可迁入仍正确的局部算法、
   verifier和negative case；不迁移旧API、状态布局、调用顺序或winner行为。
3. **Q51.Core同批删除的旧search owner**：`TileExecutionCandidate`/metrics/transition bag、`deriveShortlist`、coordinate descent、witness
   shortlist、candidate-local schedule/evaluator、allocator/buffer feedback、beam/budget closure、equivalence cache、accepted cohort、
   `StaticSchedulePlan` selector、stable ordinal tie-break、winner rematerialization、字符串failure-gate状态机及其统计/diagnostic合同。
   同批删除读取旧Rank/coordinated source marker的CTest、paired `none/search` optimization catalog/driver和旧长链test registration；
   它们不是新Core的过渡输入或回归基准。
4. **非current源码**：未进入active build的rank/coordinated/algorithm/NoC旧实现不恢复；具体Q50任务若需要其中独有proof、算法或
   test witness，则在同一变更中移入新owner并删除对应旧文件，不以“曾经工作”形成兼容义务。
5. **旧测试资产**：只迁移仍能独立表达新IR/mechanism合同的source、oracle和negative witness；绑定旧proposal数、stable ordinal、
   shortlist、feedback、winner digest或异常长旧`search`执行的测试直接删除，不进入新链测试集合，也不形成对照基准。

### 12.2 源码层次

不按硬件范围或抽象角色新增横向目录层；目录必须对应稳定的 IR、analysis、lowering 或验证职责。稳定职责分布为：

- TensorProgram analysis：fusibility、iteration partition、streamability；
- PhysicalDataflow analysis/transforms：exact demand、TileRegion formation、coupled traversal、representation、movement；
- Instr analysis/transforms：buffer lifetime、ready order、worker、completion；
- Conversion：TensorProgram到CardModule、TileRegion到Instr；
- Compiler search：typed assignment、deterministic frontier、dependency invalidation、session evidence与exact/anytime control；
  profile-driven priority、scalability优化和LNS由Q52加入；
- CardExecutable compilation/verification：串接既有lowering和exact gates，不实现choice generation。

顶层编译入口只编排`none`或新`search`并发布结果；不能继续容纳具体spatial/fusion/layout/buffer算法。Q51.Core直接替换public
`search`的control owner并删除旧candidate/generator/evaluator/feedback/selector/rematerialization调用闭包及其专属测试；在
production mechanism尚未闭合时，新Core返回typed search failure。每个Q50 checkpoint只在新链中验证本轴typed
domain/query/apply、actual-IR witness和下游Q50.0定向准入，同时删除本轴不再使用的旧机制文件和测试；不重放旧monolith，
不保持旧candidate set、winner、统计、digest或耗时。

## 13. 当前差距与任务闭环

Q50.0、Q54与Q59已经闭合无策略selected-executable compilation、MLIR scope/pipeline和compiler transaction。Q50.A旧API已有
immutable borrow、typed relation/role及部分reduction/broadcast/window/multi-piece witness，但本轮重审已因execution/ownership重复、
partial-result owner、node-wide merge及manual epoch等问题重新打开。A的终态输入`SpatialAssignment`由Q50.B representation foundation先
定义、关闭并由baseline canonical producer实际产生；A不能临时保留旧trial作为输入。动态状态只看`tasks/progress.md`。

Q49.P已经保留consumer-operand exact demand、structured-producer截断、single-root final construction、typed carrier和search-policy
隔离等能力；但它仍用每个closed temporal coordinate的一次CardModule/Q50.0来查询capacity，因此重新打开。current旧probe、
post-hoc support rebuild和default statistics不恢复；剩余迁移先把canonical component facts迁入最终plan types，并让Q50.F closed-plan
core接管所有coordinate，只让最终plan进入一次actual commit。它不等待完整search domains。
主线仍有下列设计差距：

- baseline的coordinate loop仍执行actual TileRegion→Instr与memory planning；这些工作必须改成Q50.F typed problem/proof，planning IR为零；
- `spatial-plan-schema`只定义schema、structural close与validator；`attention-normalization`先产生normalized semantic roots和fixed FA/FD fact，
  `attention-spatial-integration`关闭K1/K2与spatial constraints后`canonical-spatial-assignment`才签发assignment；
  `exact-demand-boundary`与`attention-demand-integration`随后关闭demand/final owner/
  coupled merge；
- placement production transition仍过早消费physical edge strategy；Q50.B尚未从all-iterator semantics生成完整multi-axis、
  remainder、reduction/merge和非矩形physical placement域；
- current旧search仍含candidate、shortlist、repair与完整编译耦合；只能在Q50能力逐项迁移并由Q51 planning取代后删除；
- spatial domain仍主要是单output axis与连通矩形Tile group；
- region/temporal/fusion/layout/buffer/communication尚未由一个轻量state联合回溯；
- 多个旧机制文件未进入current production build；
- stage pipeline只有历史机制，未接入current common search。

任务按以下output闭环推进，具体状态以`tasks/progress.md`为准：

1. 保持已经闭合的无repair CardExecutable compilation/verification边界；
2. 按`tasks/progress.md`依次建立semantic/spatial/demand与canonical plan work items，再由`deterministic-baseline-closure`完成`none`；
3. 依次建立全部domain work items、`search-control-foundation`与`search-control-closure`，全程planning零IR且不消费baseline output；
4. 以`unified-search-closure`关闭single winner，再经`attention-production-closure`和`search-scalability`取得production search证据；
5. `production-host-readiness`从Q60产品入口完成fresh workload/package/no-card/runner矩阵，使Q53达到`board-ready`。

## 14. Verification and Done Criteria

physical-dataflow的IR、analysis、planning、materialization和lowering正例默认使用rank至少为3、至少一个主要迭代维度
不小于1024的shape；空间和时间划分必须成对覆盖`1024`等整除长度与`1025`、`1031`等非整除长度，确保测试实际形成
均匀块、多Tile、多个block/wave、remainder和tail。矩阵还要覆盖单轴/多轴及相关fan-in/fan-out、broadcast、reduction、
view/slice语义，并逐项断言exact coverage、无重叠、owner、demand、merge、tail和下游表示。个位数shape只允许用于可穷举
的独立oracle、最小负例或单一故障定位；它和任意单个成功case都不能证明axis、exact-demand、resource、schedule或
selected lowering完成。shape只是case输入，不进入合法域、candidate选择或workload协议。

### IR / verifier

- distinct Tile modules可以包含不同op、loop、shape和长度；
- duplicate/unavailable Tile、coverage hole/overlap、illegal reduction merge、cross-Tile/region SPM alias失败；
- same-region independent traversal、coupled traversal、region cut、selective spill、recompute和communication staging可验证；
- selected spatial mapping的iteration coverage、result ownership和edge demand all-and-only闭合。
- exact-demand query按完整logical domain支持reduction input/init/partial contribution、broadcast、affine window+stride/dilation、
  strided view/slice、multi-piece、multi-result和support relation；physical carrier变化不改变logical outcome。

### Baseline与search correctness

- chain、diamond、fanout、reduction、mixed compute/movement和stage+buffer先由真实规模IR case证明production路径；另用明确标注
  的小domain独立reference enumerator和test-only flat exhaustive oracle证明planning set、剪枝和winner，不能让oracle规模
  替代production shape覆盖；
- 构造必须联合改变spatial/region/temporal等多个choice才改善的陷阱，anytime策略能找到更优plan且production只commit一次；
- estimate、memo、no-good或dominance逐项开启不改变有界穷举oracle的winner；同一机制的真实规模矩阵另行证明production路径；
- `none`和`search`只共享policy-free analysis、typed plan schema、materializer和selected-result compilation/verification，互不调用；
- Q51.Core起旧接口不进入current控制流，但旧source/test中的独有算法、proof和witness必须先迁移；每个checkpoint同时验证typed
  mechanism、production consumer和donor capability，Q51比较new planning与独立oracle、winner actual IR和一次Q50.0结果；
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
- baseline plan query覆盖region/Tile/Card resource scopes，exact SPM rejection携带direct causal witness；无region/function clone probe，
  fresh计数证明完整CardModule materialization和CardExecutable compilation各一次；
- 结果明确区分`objective-optimal`、`feasible-with-bound`、`feasible-unranked`与`budgeted-feasible`；work/wall/RSS/incumbent只在显式
  instrumentation中报告，不伪造bound或最优性。

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

上述workload按任务阶段执行，不是每个checkpoint的共同回归集。Q49.P迭代期使用有界direct unit、定向lit和轻量
source-to-package/no-card；算法、结构与work evidence闭合后必须只运行一轮fresh FP16 LLaMA `optimization-none`
source-to-package/no-card，证明baseline在真实multi-producer support DAG上完成且没有进入search。该功能门禁不承担search质量
profile，也不能由历史输出代签。Q51完整new-search链闭合前不得运行LLaMA `search`或反复执行重型baseline；Q52才对LLaMA
search执行显式、bounded scalability profile，Q53再签发正式模型package/oracle/no-card与board-ready矩阵。轻量case和LLaMA
均不能由模型名特判产生，必须经过同一typed relation、carrier和materialization合同。

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
