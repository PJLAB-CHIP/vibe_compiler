# Wafer Tile-Dataflow Scheduling 与 Candidate Selection

状态：2026-07-16按已完成Q29终态设计重写。文件名暂时保留为`06-group.md`，只为避免当前编号导航和历史引用
断裂；本文的长期owner是rank-local tiled task/dataflow scheduling、candidate generation/ranking/commit和
complete-traversal合同，**不再把`wafer.group`定义为执行融合、DDR切边、SPM residency或提交单元**。
实现状态只看`tasks/progress.md`。

当前production artifact是verified rank-local structured tensor program。scheduler直接从该artifact构造
rank-local candidate clone；历史调度op、逐容器candidate、conversion/dump和named pipeline已从live实现删除。
production、过渡artifact和IR-local debug共用同一条structured source-to-bundle语义主线。

## 1. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  Shardy/XLA SPMD之后、按logical rank静态specialize的Linalg/Tensor/SCF/Arith/Math tensor program，
  以及wafer.linalg_ext.collective.* logical collective；production request同时
  提供validated ExecutionConfig和typed
  TargetProfileId。
- Current stage responsibility:
  从structured iterator、indexing maps、SSA use-def、shape/dtype、collective interface和effect构造每个rank的
  tiled task/dataflow template；当前V0枚举有限的traversal tile、受数值合法性约束的reduction tile和
  shared-input scope-prefix/保守partition policy；在whole-rank/whole-variant clone上显式物化DDR subview、SPM memref、typed
  movement/layout/compute/collective task及event dependency并完成instruction legalization。rank内所有selected task
  candidates写入同一个
  transformation-local完整clone后才运行一次canonicalization，消除static one-trip traversal wrapper和等价full view；
  从该canonical committed-rank clone派生spill baseline与deterministic maximal full-buffer-resident两个有界
  whole-rank alternatives，分别运行SPM、DDR、completion、transport、geometry、verifier、cost和target ABI gates，
  只在complete passing candidates之间排序并原子提交一个variant。Q16形成all-rank bundle之前，再让每个rank
  frontier alternative独立经过function-boundary bufferization和physical-memory replanning；失败alternative被过滤，
  surviving alternatives从最终instruction IR重新计算cost，只有没有alternative存活时该rank才失败。
- Output artifact / IR:
  每个rank覆盖完整static traversal的accepted target-abstract tiled task/dataflow program，随后形成不含
  tensor/Linalg/wafer.group残留的instruction/memory/completion program；全部rank共同组成atomic、
  profile-bearing ExecutableBundle。accepted IR中的memref SSA、movement op、event和offset fact是唯一执行事实。
- Downstream consumer:
  tasks/07 tile execution、tasks/08 layout、tasks/09和12 memory planning、tasks/10和11 instruction lowering、
  tasks/13 transport、tasks/14 target preflight/LLVM、Q17 target artifact和Q22 target CModel。下游不读取
  rejected candidate、search trace、历史group边界或debug dump。
- User-level driver / named pipeline:
  production只经wafer-compile的同一source-to-bundle事务；已退役surface不保留第二条
  compatibility replay或用户stop-stage。
- Explicit non-goals:
  本stage不定义framework数学语义、SPMD partition、runtime launch/package字段、raw packet/CRT ABI或
  cycle-accurate timing；不引入opaque task payload、长期side table、名字匹配、shadow schedule、代表rank、
  rank-class dedup或任意dynamic graph scheduler。当前V0不枚举generic per-edge resident/spill、task-order、
  layout-cut或double-buffer policy；deterministic maximal full-buffer-resident alternative也不构成任意edge subset
  frontier。这些是保持同一IR/gate合同的延期性能扩展，不是正确性fallback。首版不承诺
  dynamic-shape、online-softmax、动态multi-instance loop或未经event证明的ping-pong。
- Completion gate:
  从真实rank-count=1/16 structured programs构造all-and-only task traversal；producer/consumer短边可通过同一
  SPM buffer跨task传递且不会被tile-region container边界强制写回DDR；layout、DMA、collective、event和buffer
  lifetime均可从accepted IR重算。7B compile gate证明Q/K/V/O/gate/up/down的完整weight-layout DDR临时值消失、
  gate/up共享activation不随每个N tile重复RDMA、collective结果在wait后直接进入residual；全部rank通过
  SPM/DDR/transport/ABI和complete-coverage gate，任一candidate/rank失败不形成partial bundle。Q28继续拥有
  7B完整CModel数值差分和性能观察，不由Q29文档完成替代。
```

## 2. 核心结论和稳定术语

终态pipeline是：

```text
rank-local structured tensor IR
  -> recomputable tiling/dataflow analysis
  -> bounded candidate clone
  -> explicit tiled task/dataflow IR
  -> instruction + exact SPM/DDR/event/transport planning
  -> all-rank atomic commit
```

本文使用以下术语：

- **source op**：上游Linalg、tensor view或logical collective；仍拥有数学/structured语义。
- **tile task instance**：source op在一个parallel tile coordinate和可选reduction step上的target-abstract执行实例。
  task不是通用opaque wrapper，而是typed movement、layout、compute或collective op。
- **data edge**：producer写入的typed buffer version到consumer operand的SSA relation。
- **event edge**：provider completion、multi-input join、buffer reuse、communication wait或terminal drain关系。
- **rank task program**：一个logical rank entry内的完整static task DAG及其structured control flow；它是SPM
  lifetime和candidate scheduling scope。
- **candidate**：一次完整的transformation-local proposal。当前V0由traversal/reduction tile和scope discovery
  policy组成；未来增加task order、layout cut、generic per-edge residency或buffering policy时仍必须形成相同的
  complete clone并经过同一组exact gates。

“融合”不是长期IR对象。两个数学op可以保持两个task，同时通过同一SPM buffer和event直接相连；只有显式
`store -> DDR value -> load`才表示spill。这样既保留可验证的op语义，也让硬件真正关心的storage、movement和
completion成为IR事实。

## 3. Rank-Local Task/Dataflow IR 合同

### 3.1 Scope

rank-specialized `func.func`是完整task graph和memory-lifetime scope。`wafer.tile.region`可以继续作为一个或多个
tile task、static traversal fragment或phase的结构化容器，但不再拥有per-group storage arena，也不允许因region
边界自动生成DDR writeback/reload。跨region的SPM memref和event必须作为显式operand/result传递。

Q29应优先复用现有：

- Linalg/Tensor/SCF及其interfaces；
- `memref.subview`和带`#wafer.memory<space, layout>`的memref；
- `wafer.tile.load/store/materialize_layout`、typed tile compute/collective、`wafer.instr.*`；
- `MemoryEffectOpInterface`、现有async token、fence/wait和structured control flow。

只有现有op/interface无法表达provider completion或buffer version时，才增加最小typed event operand/result或
interface；不得新增一个携带任意payload、task kind字符串或隐藏schedule的通用task op。

### 3.2 Task node

每个task node至少具有：

- 明确的typed input/output memref或scalar SSA operand；
- 当前static tile offsets/sizes或可由enclosing structured loop重算的coordinate；
- 对应memory/layout/compute/collective语义和read/write effect；
- 对异步provider、reuse和communication必要的dependency/completion event；
- lower到具体instruction family所需且可验证的shape、dtype、layout和geometry。

task不携带source名字、model role、group id、candidate id、planner score或runtime handle。一个task是否与前后task
共享SPM，只由buffer SSA和lifetime决定。

### 3.3 Buffer version和event

同一physical SPM range可以在不同lifetime复用，但每次provider写入和consumer读取必须有可验证的版本顺序：

```text
load/layout/compute provider issue
  -> provider completion event
  -> consumers
  -> join(last consumer completion)
  -> next writer / range reuse
```

block order只表达程序顺序，不能代替异步完成。DMA、DTE和其它异步family必须产生或关联可信completion；local
fence只完成其明确覆盖的engine/resource。rank entry返回前必须完成外部write、communication和其它可观察effect。

当前static traversal用compact `scf.for`和有限static tail class表达，event、buffer SSA和lifetime仍从
完整accepted traversal重算，不按动态instance eager展开。未实现的ping-pong必须先引入两个明确slot和
loop-carried ready/free event；动态parity alias、loop外captured mutable buffer或无法区分multi-instance
lifetime的表示继续fail closed。

## 4. 从Linalg Indexing Maps生成Task DAG

task/dataflow lowering只依赖structured语义，不匹配模型、op或value名字。

1. 从Linalg iterator types、static loop ranges、DPS inputs/inits/results和scalar region得到iteration domain、
   reduction relation及exact payload。
2. candidate为parallel iterator和允许的reduction iterator选择有限tile sizes。一个task instance由
   `(source op, parallel tile coordinate, reduction step)`唯一确定。
3. 对每个task iteration box，把box通过各operand indexing map求image：
   - projected permutation形成规则rectangular slice和`memref.subview`；
   - 缺失iterator表示broadcast/reused source slice；
   - permutation形成layout relation；
   - static reshape reassociation形成可证明的coordinate/view relation；
   - 无法表示为规则slice时，只有typed gather/scatter task可以承接，否则candidate失败或继续retile。
4. producer result tile与consumer operand image相等时直接连接同一SPM buffer；contained relation使用显式view；
   layout不同但storage可转换时插入`wafer.tile.materialize_layout`；只有placement/capacity/cost选择spill时才插入
   DDR store/load。
5. tail tile使用实际static sizes，coverage verifier证明每个result element all-and-only一次。

Matmul的maps `(m,k),(k,n),(m,n)`产生A `[mT,kT]`、B `[kT,nT]`和C `[mT,nT]`tile。K split不是
独立partial result：init/fill、每个K step和accumulator event形成有序链，final K completion之后consumer才可执行。

Reduction例如RMSNorm `(row,k)->(row)`或softmax row reduction先产生input chunk和accumulator task，再在final
reduce event之后执行rsqrt、broadcast、exp或normalize。不同full tensor shape不再是切断dataflow的理由。任何改变
source combiner顺序、NaN/rounding或identity语义的split都必须由numeric legality证明，否则保持source顺序或拒绝。

`linalg.generic` scalar body先按tasks/05和10的typed mapper验证。多个generic可以保持多个task并共享SPM buffer；
消除DDR不要求把多个scalar DAG伪装成一个新复合op。

### 4.1 Nested Region Capture

task materialization必须递归检查source op的nested regions，不能只枚举top-level operands。nested body使用、但定义在
当前task fragment外的value形成显式data edge；进入`IsolatedFromAbove`的`wafer.tile.region`时必须改写成region
operand/block argument。scalar constant可以按exact typed attr在region内clone，rank/function input及其它producer
必须保留原SSA/type relation，不能通过名字、默认值或隐式capture恢复。

当前V0已验证的nested-capture正例限于当前typed scalar mapper能解释的body：例如函数`f32`输入被简单elementwise
`linalg.generic` body捕获时，它必须先成为structured scheduling scope的显式input，再成为
`wafer.tile.region` operand/block argument，并由body中的fill/elementwise SSA直接消费。这不表示任意nested
tensor access、control flow或index-dependent body已经支持。

当前structured scheduler支持的`tensor.extract_slice`子集是：offset、size和stride全部static，producer是当前scope可见的
ranked tensor，且slice result的全部uses位于selected structured scope内。函数边界tensor上的该子集直接lower为typed DDR
`memref.subview`和RDMA；已materialized SPM value上的fully-static slice可lower为typed local movement。dynamic metadata、
scope外fanout、effect不明或无法证明view relation时保持显式task/data edge或结构化失败，不能为追求“融合”改变SSA语义，
也不得在function-boundary bufferization后遗留target LLVM无法解释的generic `memref.copy`。

structured scheduler/materializer回归必须同时锁定上述scalar capture的显式input/SSA、external fully-static slice的DDR
subview/RDMA，以及普通index-dependent generic的稳定负例。当前generic scalar mapper不接受`linalg.index`；除非命中
现有exact canonical two-way concat specialized matcher，或未来其它独立且完整验证的specialized pattern，否则必须以
`unsupported linalg.generic body op linalg.index`结构化失败，不能为通过capture测试扩大该能力。

## 5. 空间切分与时间切分

### 5.1 Space partition

空间切分在本stage之前由topology、execution mesh和Shardy/XLA SPMD确定。每个rank task program只消费自己的local
tensor shard和显式replicated values。collective通过`rank_group`、`local_rank`、shape/bytes和combiner连接多个rank
program；Cx/NCx是rank内physical layout，不是新的mesh partition。

scheduler不得从parameter filename、SymbolRef、task order或默认0恢复rank。全部rank分别lower和验证，即使两个
accepted modules byte-identical也保留两个typed rank records。

### 5.2 Temporal tiles

时间切分只描述同一rank上task instance的执行顺序：

- parallel-axis tile决定output traversal；
- reduction-axis tile决定ordered accumulator steps；
- producer/consumer可以采用不同tile domain，但必须用indexing-map image建立显式slice/repack relation；
- invariant operand可以跨多个time tiles常驻SPM，也可以按cost显式重复load；
- 当前V0保持deterministic source/DAG order；未来task-order候选也只能是DAG合法topological order的有限集合，
  不能枚举任意permutation。

tile越小通常降低single-instance SPM peak，却可能增加invariant operand重复load、DMA descriptor、issue和sync数量。
candidate必须从materialized task graph统计这些成本，不能只用tile volume或单group峰值排序。

## 6. Storage、DMA、Layout、Collective和Completion

### 6.1 SPM和DDR edge

SPM allocation scope是完整rank task program。allocator消费instruction-level buffer SSA、alias、effect和event lifetime，
可以让不同task/region的非重叠buffer复用offset。per-region planning可以作cheap lower bound，但不能提交独立arena或
阻止跨task SPM传值。

storage选择有且只有两种可观察IR结果：

- **resident edge**：producer和consumer使用同一SPM memref/version，不存在中间WDMA/RDMA；
- **spill edge**：producer显式store到typed DDR value，consumer显式load对应view。

external input、parameter、constant source和最终output仍拥有DDR/ABI边界；这不授权每个task intermediate落DDR。

同一个canonical committed rank只产生两个有界whole-rank storage alternatives，而不是为每条edge建立搜索bit：

1. **spill baseline**：保留原有typed DDR WDMA/RDMA边界；
2. **deterministic maximal full-buffer-resident**：按稳定IR顺序一次提升所有已完成完整闭包证明的full-buffer
   handoff，不枚举其任意子集。

一条full-buffer handoff只有在以下事实同时由当前IR证明时才可原子改写：producer output对应compiler-owned、
single-use的DDR `memref.alloc`，memory layout为`Tensor`；producer以static full WDMA从region-owned、exact
SPM Tensor buffer写出，且WDMA后只剩local fence/yield；跨producer result到sibling region operand的external view
chain当前只接受等physical storage的static `memref.collapse_shape`、`memref.expand_shape`和`memref.cast`。producer
region内通向WDMA destination、consumer region内通向RDMA source的local view chain还可接受zero-offset、unit-stride、
full-size static `memref.subview`。每个terminal transfer都必须是complete RDMA。external `memref.subview`即使覆盖
full storage也暂不提升；任一use或任一consumer不满足合同，整条producer-result edge保持spill，不允许只改安全consumer。

提升成功后，producer `wafer.tile.region`显式yield SPM result，全部sibling consumers显式接收SPM operand；
consumer所需static full view在本地以SPM view重建，complete RDMA替换成local SPM gather。该表示不引入
隐式跨region arena，也不从buffer或op名字恢复binding。

### 6.2 DMA

target-abstract DMA task明确DDR subview、SPM buffer、byte range、strides、effect和completion。instruction lowering再
生成RDMA/WDMA descriptor。prefetch和double buffer只能通过两个明确buffer slot及event依赖表达。DMA setup数、
strided efficiency和共享bandwidth属于candidate cost；descriptor/range/alignment属于hard legality。

### 6.3 Layout

buffer的accepted layout只存在于memref type和显式layout movement。layout propagation优先让producer或DMA直接产生
consumer需要的layout；需要转换时在SPM中插入typed materialization。single-use weight transpose不得先生成完整DDR
transposed tensor再由GEMM重读：scheduler应按GEMM所需weight tile从原始DDR view加载并在SPM转成Cx/NCx。

若一个prepacked layout有跨invocation、multi-consumer或immutable resident cache的真实owner，才可比较持久化收益；
当前compiler-managed、single-invocation、single-consumer临时layout没有该合同。

### 6.4 Collective

abstract collective task消费/产生SPM buffer并携带rank group、local rank、fixed shape/bytes、combiner和completion。
tasks/13把accepted collective lower为Direct DTE send/recv/wait task graph，并做all-rank message matching。

collective是phase/event barrier，不是DDR barrier。producer完成后可直接把SPM result交给collective；collective wait后
residual/epilogue可以消费同一buffer。只有chunk algorithm、message order、staging lifetime和all-rank transport都通过
时，才允许跨collective做更细时间流水。

## 7. Bounded Candidate Search

### 7.1 Candidate variables

当前V0只改变以下有限决策：

1. 每个tile-domain class的parallel tile size；
2. 一个共同static reduction axis上、通过第7.2节数值gate的reduction tile size；
3. scope discovery policy：shared-input peer prefix `0/1/2/all`、terminal full-only recovery和保守
   same-shape partition。它们是六个独立policy，恢复项不与peer prefix做Cartesian product。

同scope内可证明的短dataflow edge直接成为resident SPM SSA，因此scope-prefix是当前V0的粗粒度residency选择。
generic per-edge resident/spill frontier、reuse-aware task order、layout cut和single/double buffer没有进入当前搜索；
以后实现时也不得枚举source op的任意partition、所有topological orders、所有buffer subsets或无界tile-size
Cartesian product。

### 7.2 Tile-domain class

indexing-map relation相同、consumer slice约束相容的tasks共享tile-domain variable。size menu从static extent、target
geometry、divisors、preferred target sizes和SPM capacity threshold派生。每个pressure axis优先保留：

- full extent；
- first legal size；
- 刚好容纳一个高收益resident set的threshold size；
- 必要时再小一级。

这些是compiler-private search policy，不进入public request、bundle identity、pass/pipeline名称或IR attrs。当前
candidate search不能把搜索不完备误报成workload语义非法。正数
`maxSearchCandidates`是无条件hard cap：无论是否已经找到passing candidate、全部candidate是否失败、是否并行评估，
实际访问数都不能超过该值；并行batch只能消费剩余预算。

candidate materializer对直接yield root使用一份共享的结构能力分类：Linalg root是`Tiled`，可以进入有限tile
refinement；tensor collective root是`FullTraversalOnly`，只访问full-shape candidate；其它root是`Unsupported`并在
搜索前拒绝。该分类只描述traversal materialization能力，不替代完整instruction/SPM/DDR/transport legality。
四个normal scope policies仍优先保留完整dataflow；独立的bounded recovery policy使用
`maxSharedInputPeers=0`精准切开终端`FullTraversalOnly`root。若首个normal policy已在包含Tiled producer的
`FullTraversalOnly`yield scope上确定性失败，recovery立即前移；其余peer prefix只会增加独立root，不能改变该
direct dataflow closure的traversal capability。没有观察到该结构性失败时，recovery仍位于normal policies之后。
这只是原六个policy的短路顺序，不新增alternative、不与peer prefix做Cartesian product；recovery仍接受exact gate
和whole-variant选择。collective-only scope保持完整，collective作为内部producer并由最终Linalg root yield的scope
也不切；该恢复不开放loop collective、跨region residency或新的IR外计划通道。

当前`reductionSplitSizes`只对具有一个共同static reduction axis的roots生成有限菜单，并从full extent逐级refine到
下一个较小候选，不打开多轴或无界Cartesian split。能枚举split不等于允许改变数学求值：selector和直接candidate
materialization共用同一个current-IR numeric-legality gate，后者防止绕过搜索器。

- 普通未拆分reduction保持既有source-order合同，不因缺少fast-math事实而被拒绝。
- 浮点`linalg.generic`只有在exact single-combiner已经由structured body证明，且该combiner显式携带标准
  `fastmath<reassoc,nnan,ninf,nsz>`时，才可生成独立neutral-init partials再按升序combine。
- named浮点`linalg.matmul`当前没有可表达上述重结合许可的typed source-IR fact，因此保持完整K；若完整K无法通过
  geometry/capacity，结果是明确的no-candidate，而不是偷偷生成partial-GEMM tree。
- integer split只覆盖当前exact matcher能证明的modular add（无overflow flags）和signed min/max；unsigned min/max、
  overflow-qualified add、`maxnumf/minnumf`或额外payload仍拒绝。

materialization按reduction logical index升序产生chunks，并用前一chunk的SSA accumulator连接下一步；lowering会为后续
chunk形成neutral-init partial及显式combine。该顺序保证调度可复现，但本身不构成浮点等价证明；允许partial和最终combine
的依据仍只能来自上述source-IR gate。显式传入等于full extent的单chunk不会产生combine，因此不需要重结合许可。

### 7.3 当前scope-prefix residency与后续per-edge frontier

短single-use edge、layout-to-only-consumer和post-collective epilogue在同一scope内通过exact gate后保持SPM；
external result强制写DDR。当前scope discovery把共享external DPS input的peer按
`shared input bytes / result bytes`稳定排序，只枚举`0/1/2/all`个peer的prefix；producer closure随后由SSA
dataflow吸收。这是scope级融合/residency policy，不是每条edge独立的resident/spill bit。

跨time-tile复用、fanout或长lifetime value的generic per-edge frontier尚未实现。未来可按
`avoided DDR bytes / persistent SPM bytes`及reuse distance生成有限prefix，但必须保持显式SPM SSA或成对DDR
store/load，并接受whole-rank lifetime；不能把该扩展写成当前Q29正确性缺口或`2^N`搜索承诺。

第6.1节的deterministic maximal full-buffer-resident alternative是selected task candidates全部写入完整rank clone后
进行的一次有界finalization，不是candidate variable：它只比较“保留全部spill”与“按稳定IR顺序提升全部可证明
full-buffer handoff”两个whole-rank alternatives，不产生per-edge frontier，也不回溯枚举promotion subset。

### 7.4 Task order

当前V0使用deterministic source/DAG order，不枚举order variant。后续只有能降低明确reuse distance的少量
topological变体才值得加入，例如让共享activation的Q/K/V consumers相邻或让同coordinate gate/up task配对；
仍不得枚举所有topological permutations，也不得跨reduction/collective completion重排。

### 7.5 Cost

hard legality和ranking必须分开。只有通过complete task traversal、instruction、SPM、DDR、completion、transport、
geometry和ABI gates的candidate才能比较：

```text
removed_boundary_bytes
  = eliminated WDMA + eliminated consumer RDMA
net_ddr_bytes
  = explicit candidate RDMA/WDMA bytes
    + repeated invariant loads caused by smaller tiles
    + explicit spills

estimated_time
  = compute_cost + spm_movement_cost + issue/setup_cost + communication_cost
    + ddr_cost
```

`estimated_time`只是粗排序值；未校准compute class可能让两个storage alternative的标量时间同时饱和。spill baseline
和deterministic maximal resident之间因此还比较从完整当前IR重算的exact count vector：NPU/vector各compute class、
DDR read/write、SPM movement、NoC transmit/receive、instruction和event。只有两边每个dimension都known、resident
在所有dimension都不更差且至少一个严格更小时，才成立strict execution-cost dominance。任一unknown dimension或
任一tradeoff都不能走该规则；它不会把未校准计数升级成board time，也不放宽hard legality。

只有IR/event能证明overlap时才用`max(compute, movement)`，否则保守相加。GEMM按N切`q`片且A不resident时，
A traffic是`q*M*K*element_bytes`；按M切会重复B；按K切且partial C不能留SPM时会新增partial-result spill。

`docs/wafer-hardware-instruction-set-and-programming-model.md`给出的单tile 3 MiB SPM和单卡200 GB/s DDR带宽可以
作为粗resource/cost输入，但200 GB/s是16 tile共享峰值，不是每rank独占。未经board PMU校准的带宽、DMA setup、
stride efficiency和overlap只用于候选排序，不升级为cycle-accurate声明；Q9继续拥有校准。

### 7.6 Evaluation and commit

cheap analysis可以用tile bytes、liveness lower bound和first/tail shape提前拒绝。accepted candidate必须在clone中
以compact `scf.for`表达主output traversal、显式表达static tail/reduction chunk/collective，并覆盖完整rank
traversal，再运行：

```text
typed task/dataflow materialization
  -> layout + DDR tile views
  -> wafer.instr.*
  -> commit all selected task candidates into one complete rank clone
  -> one post-commit canonicalization
  -> derive spill baseline + deterministic maximal full-buffer-resident alternatives
  -> whole-rank SPM lifetime/offset
  -> whole-variant DDR/range/bandwidth
  -> completion + all-rank transport
  -> target geometry/ABI preflight
```

static one-trip traversal wrapper在candidate traversal materialization和task selection期间仍承载multiplicity、coordinate、
resource与cost语义，不得提前canonicalize；只有全部selected task candidates已经进入完整rank clone后，才运行上述唯一一次
canonicalization，并让两个storage alternatives从同一canonical IR派生。fork之前的materialization、instruction或verifier
失败会丢弃基础candidate clone；fork之后spill与resident分别独立运行whole-rank SPM、whole-variant DDR、verifier/
completion和cost重算。resident的SPM、DDR或verifier失败，或其有效cost不优于有效spill时，保留spill；spill无效而
resident有效时可选择resident；二者都无效才拒绝该rank alternative，任何一侧的offset或改写都不得污染另一侧。
当两边scalar estimate相等或饱和时，第7.5节的strict execution-cost dominance仍可选中complete count vector
严格占优的resident；出现unknown或任一dimension变差时保留spill。

rank-local scheduler产出bounded frontier后，function-boundary bufferization和physical-memory replanning仍可能改变
movement与issue数量。Q16因此逐alternative独立finalize：只过滤该later gate失败的alternative，每个survivor都从
最终instruction IR fresh recost；单个alternative失败不能拒绝仍有合法survivor的rank，只有finalized frontier为空
才拒绝该rank。

每个rank frontier由上述六个policy产生至多六个distinct完整alternative；
whole-variant coordinator先按rank-local cost做最多64个组合的best-first访问，再至多尝试六个all-rank共同
discovery-order组合和一个policy-tail组合，已访问组合会去重，因此总计至多71次exact gate。winning clone最后
重新验证并一次性替换全部rank programs。representative tile、单task、单rank或某个局部memory plan都不能提交。

## 8. 7B Block Mapping Case

本节只展示标准Llama-2 7B单block、batch 1、sequence 16、TP16、FP16 storage如何流经通用合同；shape和tile
数字不是协议。

- space partition：H=4096保持activation local/replicated；32 heads在TP16后每rank 2 heads，Q/K/V local
  N=256；I=11008后每rank MLP width 688。
- RMSNorm：`[16,4096]` input、square/reduce、rsqrt/broadcast和scale形成两phase DAG；candidate比较保留f16
  input、保留f32 value或二次convert，不因reduction shape变化强制DDR。
- Q/K/V：三条`[16,4096] x [4096,256]` GEMM共享activation。N=128时每支两个time tiles；原始
  `[256,4096]` weight按`[128,4096]`加载后在SPM local transpose，Q/K tile继续进入RoPE，V进入view。
- attention：QK batch GEMM、完整row mask/max/exp/sum/normalize和PV batch GEMM按event分phase。当前sequence
  16的score工作集很小，首版保持完整row，不把该case写成通用online-softmax算法。
- O：local attention result进入`[16,256] x [256,4096]` GEMM；当前tiled O output先spill/reload为all-reduce
  complete input，collective wait后的result再以SPM直接进入residual add。
- MLP：gate/up共享一次normalized activation；每个N tile分别加载原始weight并local layout。当前accepted实现仍会
  先把tiled gate/up output写到DDR，再由SiLU/gate链读取，因为tiled producer尚未materialize一个完整resident root；
  SiLU/gate后的完整result会提升到SPM并进入down GEMM；当前tiled down output同样先spill/reload为all-reduce
  complete input，collective wait后的result直接进入residual。该边界是bounded residency性能缺口，不是数值或
  traversal缺口。

该case约有30个逻辑GEMM time-task instances：Q/K/V 6、QK/PV 2、O 2、gate/up 16、down 4。主output
traversal使用compact `scf.for`并显式覆盖static tail；ordered reduction chunk和terminal op仍受4096个host
materialization资源上限保护。该上限不是硬件或workload限制，也不能通过缩小case或静默漏实例规避。

Q29的compile evidence至少观察：

- Q/K/V/O/gate/up/down共7条single-use full weight-layout DDR中间边消失；按当前rank0 IR，这些边约占
  50,593,792 bytes无谓movement，但数字只作case基线；
- gate/up的`[16,4096]`activation不再每个N tile各自RDMA；
- 两个collective result在wait后直接进入residual，不产生中间WDMA/RDMA；
- long residual是否跨第二RMSNorm/MLP驻留由SPM/cost选择；若down tile峰值不足，缩N带来的重复load和issue
  增量必须出现在cost breakdown；
- accepted SPM offsets、DMA bytes、event closure和all-rank messages都从最终IR重算。

2026-07-16标准shape的rank-0 production结构重放选择了26条full-buffer SPM handoff。与同一rank-0 instruction IR在
该选择前相比，显式DDR movement从8,798,792 bytes降到5,100,424 bytes；selected IR包含4,892,168 RDMA bytes和
208,256 WDMA bytes。final structured rank含36个`wafer.tile.region`；反汇编16个current-binary modules后，每rank的
target call inventory都同为362条gather、30条RDMA、10条WDMA、9条GEMM和220条local fence；rank-0在SPM-root
residency落地前的历史对照分别是341、94、59、13和277，证明当前package实际选择了resident candidate，而不是只在
analysis中报告收益。
whole-rank SPM planner沿`wafer.tile.yield -> tile-region result -> sibling operand`传播allocation root并延长lifetime；
最终high-water为2,725,568 bytes，占3,014,656-byte可用window的90.411%。

该证据也明确当前优化边界：gate/up tiled outputs仍spill并由SiLU/gate链读取，tiled projection result在成为collective
input前仍spill；这两类producer都尚未暴露满足full-buffer closure的完整resident root。相对地，post-SiLU value已resident
进入down GEMM，两个collective result都在wait后直接被residual消费，没有中间DDR round-trip。剩余spill不会被文档伪装成
“已融合”；generic tiled-producer residency是后续性能扩展。external full `memref.subview`也按上述supported subset
安全保留spill；未来扩大该view proof是性能follow-up，不是Q29 correctness blocker。

最终current-binary TP16 production driver重放用520.346秒wall、12,543.822秒user和15.246秒system time发布
schema-v3 package；manifest含`rank_count=16`，以及各16个modules、entries和completions与288个typed resources。
16个module均为328,456-byte RISC-V ELF64 DYN，SHA-256逐项readback且因rank-specific DTE metadata保持digest互异；
历史pre-SPM-root module为332,552 bytes。`wafer-run --no-card`逐一对同一package的entry 0..15、Direct DTE status ABI和host-watchdog
requirement做了exact preflight，16项全部通过且均报告`board_execution: false`。这只是compiler、resource、transport、
ABI、package和no-card结构证据；本轮未运行7B target
CModel，也未把output与PyTorch `expected.npy`比较，完整数值gate仍由Q28拥有。

## 9. `wafer.group`退役

Q29不保留group compatibility layer。退役后的终态遵循：

1. scheduler直接消费production verified structured tensor program，从Linalg indexing maps、SSA use-def、types、
   effects和logical collective构造rank task DAG；不先形成group，也不从历史group顺序/root名字恢复语义。
2. nested capture、explicit boundary和complete traversal的source result/type/use relation由structured materializer
   回归直接验证，不发布对照artifact或长期API。
3. 旧op/yield ODS及C++ API、formation/analysis、逐容器candidate、conversion/dump、named pipeline/
   CLI option和只服务旧入口的fixture已删除。
4. source/IR组织检查继续禁止旧目录、注册和consumer回归；negative tombstone不是可执行compatibility路径。
   `tasks/06-group.md`文件名只因编号导航暂不改，不代表IR/API继续存在。

当前Q28工作树中的真实7B corpus、numeric/backend和nested-capture修复必须保留。Q29先替换execution scheduling
边界，Q28随后从同一source/config重新运行完整CModel差分；不得用旧53-group accepted IR冒充新task-dataflow gate。

## 10. Verification and Completion

### 10.1 IR/local coverage

- indexing-map image：matmul、batch matmul、elementwise、broadcast、transpose、reshape、reduce和tail；
- producer/consumer跨不同shape、fanout、contained slice、layout mismatch和unsupported affine map；
- resident edge无DDR movement，spill edge有成对typed store/load；
- provider/consumer/reuse event、DMA/DTE completion、terminal drain和negative missing-wait；
- rank-scope SPM lifetime允许跨tile-region共享且拒绝overlap/alias/control-flow不明确的case；
- bounded traversal/reduction/scope-prefix policies、无条件search budget和deterministic tie-break；
- rejected candidate clone不修改source/accepted module。

### 10.2 Integrated coverage

- rank-count=1和16均从真实structured program进入同一task scheduler；
- all-and-only result coverage、tail、ordered reduction和all-rank message matching；
- Q20/Q21 existing numerical vertical不回退，Q29 7B compile-only structural gate满足第8节观察项；
- 任一rank的SPM、DDR、event、transport或ABI late failure无partial ExecutableBundle/target artifact；
- committed program不依赖group boundary、rejected trace、名字约定或旁路plan。

### 10.3 Q29 complete definition

Q29只有在以下条件同时满足时才能标记完成：

1. production source-to-bundle路径选择rank-local task/dataflow candidate，而不是逐group独立candidate；
2. accepted IR显式表达跨task SPM edge、spill、DMA、layout、collective和event，whole-rank SPM planning实际消费它；
3. 第8节7B compile evidence和rank-count=1/16 integrated gates有fresh结果；
4. 旧group ODS/API/formation/pass/dump/named pipeline/CLI和consumer清零；仓库不保留compatibility/debug执行路径；
5. 双配置build、lit/unit/CTest、unsupported清单和source/dependency/IR/CRT检查通过；
6. 受影响编号设计、progress、计划和可复用memory同步，相关改动提交。

局部FileCheck、shape dump、某个task/group/rank通过或仅减少IR中的group数量都不构成完成。

2026-07-16 fresh completion gate已满足：target-model配置的lit为221 pass、2个明确feature-inverse unsupported，
base/numeric/bulk/SystemC unit分别235/235、48/48、18/18、5/5，CTest 22/22；development配置的lit为
220 pass、3个明确feature unsupported，base unit 235/235，CTest 12/12。dependency、109项CRT symbol、
CRT conformance、IR/source organization与diff检查均通过。详细unsupported清单、CRT计数和纵向transaction证据由
`tasks/16-verification-plan.md`及归档实施记录拥有；Q29据此完成，Q28继续7B完整CModel/PyTorch数值差分。
