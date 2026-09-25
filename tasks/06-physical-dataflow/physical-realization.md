# 06号设计：Current IR上的物理实现

本章属于[06号设计](../06-physical-dataflow-synthesis.md)，保留原章节编号；任务状态只看[progress](../progress.md)。

## 6. Current IR 上的 physical realization

### 6.1 Layout、view 与 bufferization

Layout assignment针对current SSA value/use和consumer interface进行。`IndexRelation`证明logical element mapping，
`PhysicalLayoutRelation`解釆current memref encoding的logical-index-to-physical-offset映射。两者只是可失效analysis，
不创建future buffer identity。

一个use需要不同layout时，rewrite直接创建actual layout materialization SSA result；多个use共享时直接共享该SSA。
Exact metadata view绑定同一storage，不创建copy/allocation。未被actual use消费的layout materialization不得生成。

Tensor层的in-place/out-of-place选择使用DPS、SSA use-def和`BufferizableOpInterface`。进入memref/Instr前，allocation、
destination mutation、view/alias与materializing copy必须已经是actual IR语义。不维护跨stage physical version或storage object。

Redundant full-buffer transfer normalization只在current IR上使用exact logical relation、physical map、SSA root、effect和
use/lifetime证明删除；partial、permuted、layout-changing或alias-unknown transfer保留。

Structured-to-Tile的parallel表达式lowering消费已bufferize的Linalg/current indexing maps，输出既有Tile elementwise、
convert或movement，直接下游为boundary movement及Tile→Instr。输入map中的常量零若对应实际extent=1的memref轴，
可以先用标准rank-reducing `memref.subview`删除该恒定轴，再按剩余projected permutation处理；offset、stride、memory space及
原storage保持，不能为删除unit轴先制造整块copy。结果坐标仍由DPS output map确定，算术body与dtype不变。
此规则属于`none/search`共用lowering，不按attention、模型或固定rank触发；非unit轴、非零常量和无法证明的map维持typed拒绝。
API依据为[MemRef subview](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefsubview-memrefsubviewop)的rank reduction，
具体结果type使用pinned `SubViewOp::inferRankReducedResultType`保留source strides。完成条件为1024/1025/1031、
不同unit位置、置换/广播、strided view的精确alias与访问证明及actual Instr下游；不扩展SPM或硬件layout能力。

复合parallel scalar body的每个中间表达式，以其actual operand indexing maps所依赖的输出坐标并集生成局部结果域；
只依赖feature的转换与算术保持feature尺寸，scalar常量保持scalar尺寸，传播result-to-loop map供下一条表达式消费。
此规则消费已选layout/current buffer与原scalar SSA，不改算术、dtype或重排，不以shape估算判断SPM。最终yield才物化完整结果坐标。
BN融合的1024/1025/1031正例须经过actual Tile/Instr/SPM，断言channel中间量未扩成完整激活；不把generic数量下降独自当作内存收益。

#### 6.1.1 Layout assignment 与 exact PBQP

Layout合法域直接从current structural TileRegion的SSA value/use、consumer interface、exact `IndexRelation`和可验证encoding构造。
Baseline与search都调用同一个query-local PBQP layout optimizer；它不是search state，也不共享两条policy的candidate owner。
Baseline与search对每个实际layout-input求解一次无外部附加约束的完整assignment；outer search不再逐value/use重新约束并枚举布局。
PBQP保留完整合法域与canonical feasible合同。FirstUse和可选LoopInvariant从同一个未修改的实际输入及同一assignment分别clone、apply、bufferize，
再经过完整下游；它们是copy placement选择。单个PBQP解不保证覆盖所有SPM可行布局或全局最快结果。
PBQP在当前IR上按实际 materialization 的 physical bytes（含 padding）与一次 materialization unit
进行 query-local 排序；最终search winner仍由物化后的其它choice和actual objective决定。该排序不能替代实际 MiniMalloc。

C3不因某value邻接view就把整个buffer-equivalent group机械降为`compactOnly`。One-Shot必然alias的DPS init/result和reshape/cast
source/result先合并为一个PBQP value group；它们不是两个可独立选择的buffer变量。该group枚举完整layout交集，但每个state必须由canonical
logical `IndexRelation`与两端`PhysicalLayoutRelation`现场证明physical element mapping、footprint、alignment、padding及write injectivity
一致，才可作为同一buffer的zero-copy state。Fixed-layout consumer需要不兼容layout时沿既有activation创建actual shared
materialization；缺少base-offset/range/alias/effect proof的slice/insert继续只允许standard view layout并fail closed。证明随IR mutation失效，
不进入PBQP之后的side table。

PBQP factor graph只在一次query内存在：value/use是当前SSA的局部变量，op tuple constraint通过auxiliary factor表达；hard factor以
显式infinity拒绝不支持的layout tuple、alias或use binding。优化目标是本次assignment实际创建的layout materialization
physical bytes（含padding）加一次 materialization unit：

```text
layout_cost =
    Σ actual materialization (physical_bytes + 1)
```

每个最终会创建一个actual `bufferization.alloc_tensor` layout copy的选择计其 physical footprint 加1；same-layout、exact metadata view、alias和inactive
activation计0。同一dominance/effect cohort中的shared conversion只计一次，不能按use重复计价；不同cohort或不同target layout分别计价。
该目标不读取NE/Vector throughput、descriptor、instruction、DDR/NoC、SPM duration或其它硬件性能信息。
等bytes/unit cost的assignment使用完整stable semantic tie-break。Hard infinity只表示已证明illegal；finite objective累加溢出
返回`Indeterminate`，不能转成infinity或`NoSolution`。PBQP的`Optimal`只表示在当前合法layout域内bytes/unit cost最小，不表示
最终硬件性能最优。

Query-local PBQP可以删除没有 live consumer 的group state：若某layout既不是该group任一live fixed-compute result的publication layout，也不是
任一current fixed use要求的layout，选择它不会被实际 current use 消费；有可用relevant state时删除该state不改变可行assignment集合。若该group
没有任何live compute/use target，则所有state目标相同，只保留原domain中的第一个canonical state。该约简不修改current IR或原始合法性
证明；无use result不产生publication cost，因为apply也不会为它创建actual materialization。

Solver必须区分`Optimal`、`Feasible`、`NoSolution`、`Indeterminate`和`BrokenContract`。Layout transformation先从同一current
value group、use domain、op tuple和conversion activation构造一个完整canonical feasible assignment；普通value选择domain中的
canonical state，fixed compute use选择其typed required state，不一致处选择actual materialization activation。该assignment必须先通过
PBQP自身的unary/factor检查，再作为exact solver的incumbent。Factor graph先按stable variable index分解connected components；一状态
变量可在任意degree精确传播，随后R0/R1/R2与residual core均受同一checked work budget约束；全assignment tie-break必须与独立flat
oracle一致。Exact search及同分选择完成时返回`Optimal`；预算耗尽时保留搜索过程中已验证的最佳完整assignment并返回`Feasible`，
包含残余搜索、独立分量及同分选择的中断。不能以初始解覆盖已找到的低成本解；同分选择中断保留已证明的主成本下界。两种成功状态使用同一
assignment类型和唯一apply实现，不建立第二条layout lowering。`NoSolution`与已验证incumbent并存是`BrokenContract`；没有合法canonical
assignment的source在mutation前按typed unsupported停止，不能猜测layout或把问题推给下游。
Assignment选中后在各自candidate owner上创建actual view/alias/allocation/layout materialization。
Search只在未修改的layout-input owner存活期间保留query与assignment以生成placement兄弟；该owner关闭时一并销毁，
已物化的下游不读取solver对象。

第17、18项的每个实际layout-input必须恰调用一次PBQP并得到`Optimal`或`Feasible`，且两者都携带完整、factor-valid并已apply的
assignment；记录status、variables、factors、solver work、wall以及apply后的actual materialization数。`Feasible`只表示本次没有完成
完整最优合同；仅当lowerBound等于cost时主成本已证最优，同分选择仍可能未完成。`Indeterminate`只允许在没有合法incumbent时返回，并阻止规定产品case完成；不能通过提高
timeout、放宽work budget或下游layout repair掩盖。性能工作继续优化exact factor formulation、connected-component reduction或有证明的
dominated-state约简，但不影响编译正确性所需的canonical assignment。

Current实现以buffer-equivalent SSA value group、每个实际consumer use和op layout tuple为query-local变量。DPS result/destination、
SCF iter-arg/yield/result以及已证明的alias view只共享同一value-group变量；不能用source structured node、operation ordinal或
bufferization后的反查恢复对应。多operand tuple用一个只枚举该op当前interface明确支持tuple的auxiliary variable编码，auxiliary
state通过binary infinity factor约束各value/use，不能把不支持的tuple变成finite penalty。

同一source的多个read-only use可以共享一个actual conversion，但PBQP不能按use重复计价。每个可共享的dominance/effect cohort和
目标layout使用一个三态activation variable：`inactive`、`source-is-target`、`materialized`。Source-layout factor只允许与当前
primary layout一致的第二态；use factor要求选择该layout的use对应第二或第三态；只有第三态承担一次conversion cost。不同block、
存在intervening alias write/free或dominance不能覆盖全部use时建立不同cohort。Apply必须与activation一一对应创建一个SSA
materialization；same-layout、inactive和没有use的activation不创建operation。

Target descriptor query不进入layout PBQP。第16项可以把该query抽为shared只读analysis，服务actual lowering、inventory和最终candidate
cost/winner比较，但不能改变第15项的layout合法域或bytes/unit objective。PBQP apply后，下游只从new current IR fresh计算
descriptor、engine work和movement；不保存descriptor plan或future Instr inventory，也不把这些性能信息反向写入layout assignment。

Current shared query位于Tile-to-Instr request-local lowering support，由layout movement与mapped elementwise/broadcast共同调用；它只接收
current memref type、projected relation和可选typed subview offset。规则性Tensor↔Cx/NCx cover直接生成有限descriptor，general relation仍走
`PhysicalAccessRelation`；二者均产生actual Instr并由同一inventory计数。第16项orchestration依次执行structured-to-Tile、boundary
movement、execution structure、standalone fanout、per-Tile Instr/cleanup/fresh completion和唯一actual leaf，不保存query结果跨stage。

#### 6.1.2 Output DPS 与一次bufferization

每个`StructuredOutputRelation`在bufferization前从其current TileRegion yield证明actual output piece。Canonical full-tensor
`insert_slice(piece, tensor.empty)`只是一种可消除的structural wrapper：layout transformation把piece作为TileRegion actual endpoint，
并在所属entry function增加对应program output的DDR memref destination及exact static subview；随后使用
`bufferization.materialize_in_destination`把piece绑定到该subview。Offset/size来自current insert/extract relation，不来自Spatial plan
或output名称。无法证明唯一piece、完整subview range或destination ownership时在首次mutation前返回typed unsupported。

同一种canonical wrapper也不能跨same-Tile TileRegion边界变成真实storage。若producer只把
`insert_slice(piece, tensor.empty)`结果交给same-Tile consumer，并且每个consumer block argument都只由offset、size和stride逐项相同的
static `tensor.extract_slice`读取，layout transformation在PBQP和bufferization前同时把producer result、consumer operand和block
argument收窄为`piece`，删除成对的insert/extract wrapper。若存在observable full result、未匹配的use、不同rectangle、非unit stride或真实
assembly语义，则保留current full tensor；不得按shape、operation名称或预期SPM收益猜测收窄。Cross-Tile canonical piece仍由同一stage的
actual boundary-source rewrite形成compact endpoint。这样bufferization只为actual compact value分配storage，不为结构占位壳创建full-shape
SPM allocation、store或reload。

同一module只运行一次function-boundary加region-local One-Shot Bufferization。`func.func` tensor boundary转换为compact DDR memref；
`wafer.tile.region`保持显式tensor boundary，内部通过标准`bufferization.to_memref/to_tensor`连接已经选定layout的actual memref
endpoint。除TileRegion boundary及这些标准bridge外，Linalg/Tensor/SCF必须全部bufferized；unknown executable tensor op不是允许的
partial boundary。Bufferization产生的SPM→DDR output copy是下一movement stage的typed input；同一DDR logical result从临时buffer
再次发布到designated output的DDR→DDR copy为合同错误。因真实old-value read、alias conflict或out-of-place语义产生的copy保留其
SSA/effect witness，不能按copy数量一律删除。

### 6.2 Movement

StructuredToTile之后，物理copy的位置继续消费同一`LayoutMaterializationPlacement` choice。
`FirstUse`保留原位置；`LoopInvariant`只将实际materializing movement外提到可证明的static正trip-count循环之前。
输入是已确定allocation/view语义的Tile memref IR，输出仍为同一SSA及owner，直接下游为boundary movement、Instr、completion与SPM。
新建结果的movement通过标准value-associated Allocate/Write说明独占存储；Wafer custom resource effects说明资源占用，
地址clobber由value-associated memory effects决定，与Instr completion的既有区分一致。
外提要求全部operand在循环外、源的全部可能alias无Write/Free、结果及其views只在循环内读取；未知effect、逃逸、
loop-carried源、zero-trip及条件执行均保持原位置。每次mutation后重建只读alias查询，不保存跨IR的证明。
Search默认使用FirstUse，在确实改变位置时生成LoopInvariant备选；两者各自进入唯一actual SPM规划，none使用FirstUse。
测试覆盖rank3/4、1024/1025/1031、多次循环、view clobber、只读复用、逃逸及zero-trip，检查动态copy次数、owner与直接Instr消费。
该变换不改变算术、dtype、访问映射或descriptor语义，也不以估算footprint决定外提合法性。

Movement choice以current producer value、consumer operand、exact demanded domain和physical layout为输入，选择local view/copy、
DDR store/load、Direct DTE、software relay或已定义collective。选择由唯一movement transformation立即创建actual typed ops、
staging buffer、token和effect。

多个独立communication component仍受每个Tile上actual TileRegion顺序约束。Movement preflight从component实际涉及的source/destination
Region建立precedence graph；同一Tile同一block中的先后顺序直接形成有向边，不同entry block没有顺序证明时同时保留两种可能顺序。若该图
有环，Direct DTE不存在一个与所有Tile current Region顺序一致的component phase order：preflight只在该actual cycle内选择总payload bytes
最小的一个component形成typed shared-DDR boundary，移除该component后fresh重算，直到剩余图无环。bytes相同时按stable component order
tie-break。该选择发生在任何movement mutation之前；不在Direct DTE verifier失败后fallback，不插wait打断环，也不建立旁路phase plan。
无环component及单向fanout继续使用其actual topology ring/tree/sparse realization。

Function/TileRegion observable result在bufferization前通过DPS/out-parameter绑定唯一actual destination。Bufferization可以因
actual alias conflict、保留旧值、out-of-place语义或明确layout/memory-space materialization产生必要copy；这些copy必须由current
SSA、alias、effect和exact relation证明，并在movement closure时成为typed movement。若DDR→DDR `memref.copy`的唯一作用只是把
同一logical result从bufferization temporary发布到designated output，而且正确DPS绑定即可消除，则它是冗余publication copy，
必须在产生点修复为0。Movement closure后未分类`memref.copy`为0，Instr conversion不得用SPM staging、RDMA/WDMA或copy-only
TileRegion掩盖错误。

Movement不从shape、value名或future version ID恢复source/destination，也不先创建donor movement再替换。不同realization
使用同一transformation实现；每个alternative作用于自己的candidate transaction。跨region或跨Tile的每个非空domain
必须all-and-only覆盖，且每个movement op必须有current SSA owner和effect。

同一个current source endpoint向多个Tile提供完全相同的payload时，movement把这些actual endpoint relations视为一个纯复制
fanout。单destination仍直接传输；多destination从current `TargetTopology`和available participant Tiles构造确定性的
topology-aware spreading tree：每轮每个已经持有payload的Tile至多向一个尚未持有payload的Tile发送，候选先均衡已用sender轮次，
再按最短hop和physical Tile ID稳定选择。全部group共用从current boundary relations和同Tile Region执行顺序得到的确定性拓扑序，
relay parent必须早于child；原关系图已有环或没有满足该序的传播edge时typed failure，不能让各group独立选树后再靠wait修环。
每条tree edge在同一次transformation中立即成为actual receive staging、send/recv token和relay use；
relay只转发已经收到的同一typed buffer，不创建future buffer，也不改变payload或算术。不同payload、不同window/layout、同Tile
不同Region residency以及typed reduction/fanin不得错误合组。只有current IR同时证明complete contribution matrix或full-buffer
fanin/fanout以及closed `add/max/min` combine use-def时，search才可在独立candidate中物化Ring ReduceScatter或
ReduceScatter+AllGather AllReduce；每轮combine必须成为actual `wafer.tile.elementwise`，DTE不暗含算术。其余reduction/contraction
保持现行merge owner和evaluation structure。

多个payload group只有在current endpoints属于同一个communication phase时才能组成component；participant集合、shape或dtype相同
不足以合组。Phase connectivity由实际TileRegion source/destination role确定：共享source、共享destination或两个single-edge group互为
source/destination才直接合组；一个Region先接收fanin、经actual compute再产生fanout时，两段是有SSA依赖的连续phase，不能仅因共享该
Region而合并。这样AllReduce的central fanin/fanout和连续exchange不会被错误地合成一个同时发生的round序列。

Temporal tiling完成后、attention decomposition和layout之前，search可显式选择communication closure；none不运行可选Region合并。
只读availability按current relation证明complete participant-pair coverage，不要求各destination收到同一个source result或相同数值。
对每个participating Tile，closure从actual Region顺序和body def-use计算最后一个local producer与
第一个remote consumer。只有严格存在`last producer < first consumer`的共同cut时，才合并该exchange涉及的TileRegion并立即retarget
live relations；合并还必须在最近parent block中保持现有SSA dominance和effect顺序。任一Tile无法满足这些条件时，整个component保持
原current IR，不进行部分合并，也不把顺序不同的阶段冒充all-gather。Search必须保留未合并owner，合并只发生在自己的candidate transaction；
两者分别经过layout、movement、completion、actual memory/target和成本比较。候选资格不决定winner。

Movement在layout/bufferization后从live endpoints fresh重建component，并物化ordinary peer transfer。全部TileRegion转为Instr、但fresh
completion尚未生成时，card-scoped transformation从actual send/recv及其buffer/view range做exact physical-range coalescing。只有同一
communication phase、source/destination Tile、encoding与root相同，而且source和destination物理区间分别构成无gap、无overlap的连续
union时，多个message才能共享一个actual transfer；consumer继续通过current subview读取各自piece。不能用logical bounding box、padding
传输或新建pack copy伪造连续性。coalescing只减少message/IR数量，不改变relation cover、alias、effect或consumer lifetime。

选择peer实现的complete exchange在已确认的native multi-destination合同内形成每source一次broadcast/scatter；其它AllGather默认使用
topology-aware Ring，search还可把post-layout owner克隆并物化recursive doubling作为actual movement candidate。Recursive candidate
创建aggregate SPM allocation和slot subview；local producer allocation能exact donation时直接改写到own slot，否则显式seed copy；remote
consumer改接对应slot。当前native合同只接受每destination `256B`、fanout `2/4/8/15`：broadcast复制同一physical range，scatter按
destination list把连续等长source segments一一分发。其它payload、fanout、ragged segment、dynamic binding或alias保持ordinary
unicast/ring，不从raw register字段外推能力。能够在一个actual cut上发issue的其余稀疏exchange使用sender容量1、receiver容量4的
capacity-constrained maximum matching分轮；相同最大edge coverage下按minimum-hop和stable relation identity选择。每轮在同一次
transformation中直接形成actual receive prepare、send、SSA token和control-flow order；临时component/matching choice随调用销毁。
若bidirectional causal component不存在共同cut，单sender slot下不能把它伪装成同轮peer exchange；baseline在mutation前选择一个exact
shared-DDR store/load boundary。它是从current Region因果顺序得到的显式movement realization，不是transport verifier失败后的fallback。
Complete exchange和round-safe只证明peer候选可用，不禁止合法DDR候选。Search可对current source/destination Region和同Tile顺序组成
无环图的边界显式物化shared DDR；入口load/出口store的当前实现不能直接套在已合并的双向exchange上。合并前的actual owner提供独立
DDR候选，不用future split或推测同步补齐合法性。

Ring和recursive doubling各自在自己的candidate transaction中进入fresh completion、actual MiniMalloc和target/cost；trial budget不足时
只物化Ring。Qualified native、non-power-of-two participant、mixed/non-contiguous payload或没有共同cut时不创建recursive downstream leaf。
Recursive candidate失败不触发movement内部fallback，也不修改Ring owner。

不存在`RoundOp`、round side plan或winner replay。无法形成exact payload、topology ring/matching或显式causal boundary时返回typed
unsupported。传播树、ring、matching和DDR realization均由同一个movement transformation一次性物化；下游只读取actual IR。

传播树的parent、child和round只是在一次movement调用内立即消费的typed choice。Region rank既约束relay legality，也先保证当前
frontier能够最大传播而不延长必要round；同一rank frontier内再按sender负载、minimum hop和physical Tile ID排序。不能用Tile ID代替
Region order。shortest-hop query只作performance ordering，
不能成为route、completion或transport legality事实。调用返回前必须全部物化，临时容器随调用销毁；
不得把edge/action/message/buffer/event清单交给后续stage，也不得在winner上重放。后续只从actual peer ops、SSA token、buffer
effect和control flow重算completion与memory。无法从current topology连接participant、无法证明payload完全一致或物化后stage
verifier失败时，当前candidate返回typed failure，不退回flat direct fanout或DDR donor。

Movement形成后运行一次current-IR exact cleanup。只有full payload、same storage、same physical map且alias/effect/lifetime安全时
才删除transfer；partial、permuted、真正layout-changing、unknown ownership或不受支持的control flow全部保留。Cleanup与layout
creation共用`PhysicalLayoutRelation`/`TransferRealizability` proof，不保留Tile与Instr两套production eliminator。

### 6.3 Execution structure 与 rotating storage

Execution-structure choice只能从movement-closed physical TileRegion中的actual loop、compute、movement、SSA、effect和token重算。
Serialized choice不修改IR；software-pipelined choice由唯一current-IR transformation立即创建prefix/steady/tail、chunk control、
actual stage occurrence、rotating allocation roots、slot selection和loop-carried SSA。它不使用future event/buffer ID、预测lifetime或
SPM footprint，也不把cross-stage execution plan或buffer multiplicity传给下游。

无法证明recurrence、effect、slot reuse、external observation，或无法用current SSA/effect/token表达下游必须闭合的completion
obligation时返回typed unknown/unsupported；不在本stage
插join、分配offset、spill或退回另一structure。每个alternative作用于自己的candidate owner，成功后旧analysis失效并fresh重算。

### 6.4 TileRegion-to-Instr

物理`TileRegion`允许空输入及空结果；canonical文本分别为`()`及`-> ()`，必须可打印、解析并再次验证。
该格式同样用于独立Instr module的checkpoint，不建立额外reader。

Conversion按actual typed Tile op使用DialectConversion/RewritePattern生成canonical Instr。它不重新选择layout、movement、buffer、
execution structure、worker或completion，也不从上游plan恢复这些事实。输出Instr在每个Tile上显式保留actual loop/slot relation、
compute/movement issue、memref use-def、effect、token和control flow。

其中 movement descriptor 的循环层级由对应 Instr ABI 直接约束：RDMA/WDMA 使用最多三层静态 endpoint stride/iteration，
没有动态 offset SSA；GatherScatter 在 descriptor 结构相同且 source/destination offset 通过 checked affine recurrence 可证明时，
由一个 current SCF loop 携带动态 offset，不能把不可表达的端点或非 affine 序列强行合并。`tile.reduce` 先尝试单个或串联多个
合法 `InstrReduceOp`，只有 native signature 不可表达时才使用 G/S + accumulator fallback；movement descriptor 的循环不能代替
带数据依赖的 reduction recurrence。

### 6.5 Worker、order 与 completion

Event/dependence graph只能作为从current Instr的operation、SSA、effect、range、token和control flow重算的query-local analysis。
它可以为scheduler枚举worker/resource/order choice，但不成为candidate identity或跨mutation事实源。

一个order choice应用后，actual block order、worker attr和token relation成为new current IR，旧graph失效。Completion owner随后从
该IR和已证hardware/runtime/ABI合同fresh构造minimum-strength、latest-unavoidable join/wait。不从TileRegion boundary、
loop backedge、movement类别或“保守”经验猜测completion。

### 6.5.1 运行时索引gather的需求与物化边界

输入为05号保留的标准`tensor.gather`及其实际source/indices SSA，输出为selected candidate中的局部gather结果、
索引读取、动态source view和连续行/片段搬运，直接交给既有layout/bufferization、Instr/completion及memory/target leaf。
生产与named入口使用相同op interface和变换；不新增独立embedding driver或runtime分块器。

输出迭代域和indices需求是静态可精确分块的；表的具体行由运行时indices决定，不能伪造affine行集合，也不能把整个source
类型当成SPM物化需求。表必须有实际SSA依赖、可访问的DDR binding及完整生命周期。只读外部表的可访问性和局部输出空间
分别验证；没有来源/完成证明的输入返回typed unsupported，不按名称或“唯一root”恢复owner。

每个Tile的输出块是一个连续allocation，各行是它的真实subview；allocation不得放入逐token循环。索引块只搬运一次并在
连续维上复用，表保留DDR。单/双slot是显式execution choice，选择后先创建真实buffer、SSA和循环；只有唯一SPM planner成功
生成并验证offset才接纳，不能按`块大小×buffer数`估算筛掉候选。块大小由通用输出域、下游布局和actual反馈选择，不写模型参数。

循环中的行/片段发令可保留，禁止逐元素DMA和没有typed crossing的逐行join。NCC→Kcore索引读取、cross-worker及实际复用
仍由最终completion owner从current IR生成所需等待；必须分别覆盖循环入口、回边和零次执行，不能在分析迭代中删除首次执行所需join。
输出块优先由直接下游消费；是否需要跨Region写回或额外layout物化，以actual依赖为准。

效率收口仍在同一current-IR边界完成：已有region/temporal融合必须能消费gather的标准接口；peel后的source、indices与result
类型从实际operand重新推导，不能将已知尾块误作动态输入。整数cast/clamp保留原scalar operation顺序，优先由已选融合中的局部
producer供给索引读取；SDK mapping只在对应可读区间取得可见性后复用，不能跨越实际DDR写入或completion/acquire。
连续reshape的分片协调及局部assembly由既有relation与实际切片处理，不通过扩大全局carrier或估算SPM筛选代替。
直接消费者仍是bufferization、boundary movement及唯一memory/target leaf；验证同时检查正式search产物、mapping动态执行次数、
局部allocation大小、通信payload和全部输出，避免仅用单独gather lowering的内存结果作结论。

本轮不实现索引排序/去重、热点表cache或推测的硬件indexed DMA。构包/heap、issue、DMA和等待开销分开测量；
现有SDK只给出基址和stride，不能把批量循环写成已存在的descriptor-list能力。完成条件及1024/1025/1031、dtype、重复索引、
4/16 Tile、tail和source到直接下游覆盖矩阵由05号3.5及当前板测矩阵共同约束。

### 6.6 SPM、DDR 与 target acceptance

SPM legality只由completion-closed current Instr IR中的actual allocation、layout、SSA alias、effect和lifetime经唯一
`PlanSPMMemory`/MiniMalloc生成并验证offset后确立。不使用footprint estimate、buffer数量、shape公式、synthetic demand或
predicted lifetime决定admission、pruning、retile或fallback。

Actual memory/target leaf不得运行function-boundary bufferization、重建join/wait或修改worker/order。若输入仍含Tile op、未闭合
tensor boundary、缺失completion或preexisting offset，按typed contract failure停止；memory leaf不是completion repair pass。

DDR planning、transport/resource verification和target lowering同样读取已经物化和通过verifier的current IR。任一stage修改
allocation、alias、movement、order或completion后，memory problem、offset和cost全部失效并fresh重算。

Actual gate只返回typed `Accepted`、actual capacity rejection、`Unsupported`、`ResourceExhausted`、timeout或compiler error。
Allocator不返回retile、spill、layout、route或completion repair recipe。

通信表示/顺序构造由13号构造式调度闭合后才进入actual memory/target leaf。Search可以选择transport或collective偏好，
不能把独立bitmask组合当成合法通信proposal；完整proposal必须拥有同一actual IR及可推进的DDR/DTE顺序。
构造分支工作有界并单独记录，沿用原search预算，不通过重命名stage隐藏失败候选或无界回溯。SPM容量仍只由actual allocator决定。

### 6.6.1 独立评分组的并行

- Upstream IR / input：accepted current Instr modules与固定cohort。
- Current stage responsibility：从实际DTE peer和DDRBinding建立保守的独立组，只在不共享完成时间表项的组之间并行。
- Output IR / files：相同SearchObjective及独立的组/worker计数，IR保持只读。
- Downstream consumer：原search controller及其候选比较。
- User-level driver / named pipeline：生产none/search继续调用同一cost求值；使用现有MLIR线程池，不增加public并行开关。
- Explicit non-goals：不修改Region lowering的现行Tile级并行粒度，不改变PBQP预算、候选访问顺序或cost公式，不并发启动依赖评分反馈的候选。
- Completion criteria：serial/parallel的精确ps、coarse、overflow与逻辑work一致；真实输入到package/no-card通过，记录wall/RSS和实际worker数量。

评分的独立组是当前只读调用内的保守分量：同一物理Tile身份、同一root、DTE peer或函数参数中的相同DDRBinding resource会连在一起。
这些键覆盖当前估计器所有跨Tile完成时间读写，额外合并只减少并行度；不推测未来通信。每组独占CompletionTimes及estimator，
每轮并行处理各组、组内按原Tile顺序更新，轮末仍按全局Tile顺序合并结果并检查changed。保留32轮全局上限、逻辑work、
coarse/overflow与失败语义；不改成读取上一轮快照的另一种迭代算法。不同MLIR context或单worker时保持一个顺序组。
共享context及线程池上限由BoundedParallel提供；全部worker结束后才合并结果，线程完成顺序不参与选择或诊断优先级。

| 输入等价类 | 必须保持的结果 | 并行证据 |
| --- | --- | --- |
| rank3、1024/1025/1031，16个独立Instr程序和32/33次循环 | 精确ps与逻辑work一致 | 实际16个独立评分组 |
| 两条独立DDR链、DTE交互及别名状态 | 组内原依赖传播、跨组结果一致 | serial/parallel评分及原cost回归 |
| 原始模型默认预算 | 候选状态、评分及最终结构一致 | fresh package/no-card、wall/RSS；整网相连时允许只有一个组 |

### 6.7 能力 owner

下列能力只保留一个最终owner。Memory/target leaf发现输入缺口时只返回typed failure，不接管上游能力：

| 能力 | 最终owner与输出 | 不允许出现的位置 |
| --- | --- | --- |
| pure structured logical graph normalization | 05号bounded access-relation e-graph；输出verified canonical Tensor/Linalg graph，不发布e-class | spatial/region search、PBQP、movement、Instr lowering |
| temporal tiling、producer fusion和本次新建slice/view local cleanup | 5.2/5.3 structural transformation；输出final current loop/use graph | e-graph extractor、layout PBQP、memory planner |
| attention structural materialization | 05/07号唯一transformation；消费graph attention与closed Spatial/Region choice，输出actual online state contributions、merge/finalize及endpoints | temporal stage、layout或winner阶段通过ID映射补建attention work |
| tiled online-attention decomposition | 05号确定性pattern；输入已完成parallel/K2 tiling的current online-attention，输出actual Linalg/Tensor/SCF且两种attention op为零 | tile选择、Spatial placement、layout PBQP或Instr lowering |
| function-boundary与region-local bufferization、view/alias、materializing allocation | 6.1 layout/bufferization transformation；输出layout-resolved、function-boundary-bufferized current IR | actual memory/target leaf、SPM/DDR planner |
| 普通layout/bufferization allocation的创建位置 | 创建该allocation的6.1 transformation；allocation在current IR中的dominance/effect位置就是memory input事实 | MiniMalloc前的generic first-use sinking或lifetime改写 |
| software pipeline/rotating allocation root、slot selection和loop-carried SSA | 6.3 execution-structure transformation；输出actual loop与allocation roots | lowering旁路buffer plan、SPM planner按queue depth补建 |
| TileRegion-to-Instr、worker/order和minimum completion | 6.4/6.5 current-Instr transformation；输出completion-closed canonical Instr | bufferization、movement、execution-structure或memory planner |
| SPM lifetime/demand、MiniMalloc offset和accepted high-water/headroom | 6.6 actual SPM leaf，从上述current Instr fresh重算 | candidate proposal、footprint estimate或上游shape规则 |
| DDR offset/high-water、transport/resource与target acceptance | 6.6 actual leaf依次调用12、13、14定义的唯一kernel | layout/search shadow state或runtime重新planning |

`sinkStaticSPMAllocationsToFirstUse`不构成一项长期compiler能力：它若只服务memory leaf的synthetic fixture，应连同fixture期望删除；
若fresh production case证明allocation确实创建过早，则分别修正6.1或6.3的直接producer，不能把该helper迁成新的通用pass。
同理，组合式“bufferize + rebuild completion + memory planning”pipeline在leaf停止调用且caller inventory为零后删除；其中bufferization
和completion能力分别由6.1与6.5保留，不随wrapper删除。
