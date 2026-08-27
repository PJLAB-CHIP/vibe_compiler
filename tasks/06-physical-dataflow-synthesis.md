# Card 内 Physical Dataflow Search 与执行构造

本文是`TensorProgram -> CardModule -> TileRegion -> Instr -> CardExecutable`主线中card-level physical dataflow的
唯一设计owner。当前任务状态和施工顺序只看`tasks/progress.md`与
`tasks/plans/physical-dataflow-synthesis.md`。历史plan和archive只作审计背景，不定义current pipeline。

## 1. 核心规则

Physical-dataflow search可以选择明确的transformation参数，但current IR是下游事实的唯一来源。

可以在物化前保存的是选择：

- structured iteration的spatial partition和Tile placement；
- TileRegion membership、top-level/nested execution和recompute/replica choice；
- temporal tile vector、wave-loop order和tail方案；
- 针对current value/use的layout、movement、worker或order choice。

必须先进入candidate-owned current IR才能存在的是事实：

- operation、SSA value、block、loop和control flow；
- buffer、allocation、view/alias、copy、scratch和lifetime；
- layout conversion、DDR/peer/collective movement、token和effect；
- Instr issue、resource binding、execution order、completion和actual offset。

一个choice一旦影响上述事实，必须由唯一transformation物化并通过verifier，然后才能被下游消费。
不为未来SSA、buffer、movement、storage或schedule建立多层C++ shadow plan，也不用plan/actual parity verifier
把shadow object追认为IR事实。

唯一允许的candidate流程是：

```text
verified current IR
  -> typed transformation choice
  -> candidate-owned actual rewrite
  -> verifier
  -> fresh analysis on the rewritten IR
  -> next direct consumer
```

搜索可以克隆最近的`IsolatedFromAbove` candidate owner试行alternative。失败的transaction整体擦除；
Accepted owner原样交给下游和最终publication，不重建IR或offset。

## 2. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  GSPMD完成card级分区、05号normalization完成后的verifier-valid card-local TensorProgram。
  SSA、structured iterator、indexing map/relation、region、effect、type、shape和dtype已完整；尚未绑定Tile。
- Current stage responsibility:
  none由baseline-owned materializer从current TensorProgram和固定规则直接构造actual Card/TileRegion IR，不创建search choice/domain/state；
  search才枚举spatial/region/temporal transformation choice并交给search-owned structural materializer。两条policy随后各自只在
  current IR上依次完成layout/view/bufferization、movement、execution structure、TileRegion-to-Instr、worker/order/completion，
  再以completion-closed Instr进入共同actual leaf。
- Output IR / files:
  policy-complete、verifier-valid的CardModule/TileModule/TileRegion/Instr IR，以及由同一accepted owner形成的
  CardExecutable和ExecutablePackage。
- Downstream consumer:
  target conversion、device link、package emission和runtime launch。
- User-level driver / named pipeline:
  wafer-compile的typed `none`与`search`产品入口；局部测试使用注册named pipeline或同一compiler API。
- Explicit non-goals:
  不重做跨card GSPMD；不从名称、shape或workload恢复语义；不新建future-output IR或shadow candidate schema；
  不让lowering、allocator、communication或completion在失败后repair候选；不修改数值语义。
- Completion criteria:
  none从current source直接形成独立baseline attempts，search从explicit structural choices形成独立candidates；每条policy的
  transformation只保留一个实现和一个事实源，不通过mode-switched complete materializer共享；每个进入actual gate的candidate
  只物化一次；唯一MiniMalloc运行于同一current Instr IR；
  current主线不再含未物化physical value、storage object、event或schedule的跨stage协议。
```

## 3. 稳定 IR 边界

### 3.1 `wafer.card.module`

`wafer.card.module`是一个`card_id`的完整MPMD verifier范围，拥有card-local observable inputs/outputs、
shared DDR boundary、all-and-only Tile modules以及跨Tile communication/completion检查范围。它不保存
candidate set、score、rejected alternative或side table。

### 3.2 `wafer.tile.module`

`wafer.tile.module`绑定唯一physical `tile_id`。不同Tile可以有不同op、loop、temporal shape、worker和执行长度。
实际顺序、并发与依赖由body中的control flow、SSA、effect、token和Instr表达。SPM root或alias不跨
TileModule传递。

### 3.3 `wafer.tile.region`

`wafer.tile.region`是一个Tile上的selected execution/local-storage scope，不是硬件Tile、单个loop或标签。07定义同一op的
structural、layout-resolved和physical form；前两者不签发SPM residency或capacity结论。Physical form才是SPM
ownership/lifetime domain，并可以包含：

- consumer-driven coupled traversal；
- 多个独立traversal及其不同temporal shape；
- local view/layout conversion和movement；
- explicit scratch、accumulator、staging和effect ordering。

SPM root和shaped alias不跨TileRegion。Structural/layout-resolved form以tensor boundary保存尚未physical闭合的logical edge；
physical form中的跨region shaped data必须由actual DDR store/completion/load或其它已定义的boundary IR表达；跨Tile data由
actual peer/collective send、recv、token/wait和destination staging表达。
TileRegion boundary本身不是completion boundary。

### 3.4 Instr、CardExecutable 与 package

TileRegion-to-Instr转换产生candidate的current target-abstract instructions。Instr层显式表达engine issue、operand/result memref、
effect、token和control flow；worker/order/completion必须在该IR上物化后才能进入memory planning。

`CardExecutable`是已通过Instr、SPM/DDR、transport、resource、completion和ABI verification的唯一内存owner。
`ExecutablePackage`只序列化accepted executable与runtime必需数据，不序列化search状态或调度副本。

## 4. Search 输入、选择与candidate ownership

### 4.1 Immutable input

一次policy invocation可读取：

- current normalized TensorProgram、SSA use-def、standard interfaces和typed effect；
- structured iterator、indexing map以及从current IR派生的`IndexRelation`；
- available Tiles、topology和显式target configuration；
- 各transformation的有限typed choice domain。

Analysis只保存可从current IR和显式target configuration重算的事实。IR mutation后相关analysis和所有指向旧
operation/value的lookup立即失效。

### 4.2 Candidate key

Search key只包含未被current IR表达的显式choice，例如partition factor、Tile embedding、region grouping和temporal
tile vector。它不包含推算bytes、future SSA identity、buffer identity、event identity、lifetime、completion placement、
actual offset或materializer遍历顺序。

当一组choice被物化，actual IR代替它成为该candidate的事实源。后续layout、movement或schedule alternative
在最近的candidate owner上试行，并以新current IR进入下一stage；不把之前的choice展开为future IR schema。

### 4.3 Candidate transaction

每个candidate owner明确持有：

- 本次新建或clone的最近`IsolatedFromAbove` Card/Tile scope；
- current IR epoch内的SSA、region、buffer relation和effect；
- 可重算的analysis和本次rewrite使用的短生命期临时数据。

失败后不在candidate内retile、spill、换layout、换route或加同步。Controller销毁该owner，根据typed outcome决定
是否生成下一组choice。Accepted owner不经rematerialization进入publication。

## 5. Structural choice 与早期 TileRegion 物化

### 5.1 Spatial partition 与 placement

Structured op的iteration domain使用exact intervals/relation切成logical work pieces。Parallel pieces必须all-and-only覆盖原域；
reduction partition必须保留complete contribution set和explicit merge owner。非整除extent必须产生tail，不能通过
shape假设丢弃。

Placement选择logical work piece到available physical Tile的injective/typed mapping。它不包含buffer、route、completion或预测
SPM footprint。Producer/consumer需求从current structured IR的indexing relation精确传播；unsupported、resource exhausted、
invalid和exact-empty保持不同typed result。

### 5.2 TileRegion formation

对每个Tile的local structured DAG，region choice决定哪些root work进入同一TileRegion，以及producer是top-level、
consumer-nested还是explicit recompute/replica。同region只选择共同local-storage scope，不证明SPM residency；只有actual producer work位于consumer
traversal内、中间值由direct SSA使用且无独立DDR往返时才称为coupled traversal。

同region的local use必须保留所选delivery直到本次structural rewrite完成：`StoredRegionValue`要求producer在consumer
traversal外实际物化一次并由region-local SSA/view复用，`ReconstructedRegionValue`保留current SSA support chain，
`DirectNestedValue`才允许把producer tile物化到consumer traversal内。Materializer不得把这三者压成同一个布尔rewire，
也不得在后续greedy fusion中把stored edge改成consumer-local recompute。该delivery只在一次candidate transaction内控制
transformation，成功后由actual loop、SSA和use-def取代，不成为跨stage plan。

Region choice必须覆盖fanout的每个use、reduction partial/merge、effect order和observable output。完全无依赖的
components不为扩大region而合并。SPM residency不是独立上层choice，只能由最终actual allocation/lifetime/offset证明。

### 5.3 Temporal tiling

每个actual traversal选择complete temporal tile vector和loop order，不允许单一标量`tile_size`代替多轴语义。所有wave和
tail必须all-and-only覆盖local work domain；reduction accumulator和loop-carried state必须在物化的SSA/control flow中表达。

Temporal materialization以一个canonical SCF loop nest承载同一traversal。完整块和remainder共享同一个current loop body；
offset与bounded tile size由loop IV和exact upper bound计算。不得在结构层递归生成`first / steady / tail`的多维笛卡尔积，
也不得按wave trip count复制compute closure。当前Instr只接受static shaped buffer时，先形成上述canonical loop，再在直接
需要static shape的边界只peel最后一个partial iteration并canonicalize其bound；不peel first iteration，不让tail
specialization提前复制无关producer closure。

实现使用pinned MLIR的`TilingInterface`、SCF tiling和producer-fusion API作为loop/fusion的唯一mechanics owner。
Fusion control只接受本次transaction中明确选择的`DirectNestedValue`或explicit recompute/replica edge；stored、reconstructed、
cross-region、collective以及无法证明不会重算的multi-use/reduction/contraction producer保持barrier。Pinned接口暂时不能表达的
exact reshape或`tensor.insert_slice` window只保留窄的current-SSA adapter，不能保留另一套通用loop/fusion engine。

Spatial、region和temporal choice闭合后，唯一structural materializer立即生成candidate-owned CardModule/TileModule/TileRegion、
Linalg/Tensor/SCF或typed Tile work，并返回actual SSA。下游不消费未物化execution/value ID。

### 5.4 Attention

Normalized TensorProgram中的`wafer.linalg_ext.attention`已将`flash_attention`或`flash_decoding`固定为graph fact。Search只选择
Q/K/V的spatial partition、K/V block、temporal traversal、region fusion和movement。Structural materializer在candidate transaction中
展开actual QK contraction、scale/mask、online max/sum、PV contraction、state update/merge和final divide。展开结果使用
普通Linalg/Tensor/SCF与同一Tile/Instr lowering，不创建attention专用shadow graph。

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

#### 6.1.1 Layout domain 与 PBQP proposal

Layout合法域直接从current structural TileRegion的SSA value/use、consumer interface、exact `IndexRelation`和可验证encoding构造。
Baseline与search都调用同一个query-local PBQP layout optimizer；它不是search state，也不共享两条policy的candidate owner。
Baseline对每个actual attempt求解并应用一次确定性assignment，不枚举layout frontier；search把同一assignment作为首个proposal，
随后仍可遍历完整raw layout域。PBQP不是合法性owner，也不决定最终search winner。

PBQP factor graph只在一次query内存在：value/use是当前SSA的局部变量，op tuple constraint通过auxiliary factor表达；hard factor以
显式infinity拒绝不支持的layout tuple、alias或use binding。Soft cost必须是final actual objective在layout stage的精确投影，不能用
统一的instruction权重代替不同engine的工作：

```text
layout_cost_ps =
    time(exact layout-dependent NE FP16/BF16 logical ops, NE throughput)
  + time(exact layout-dependent Vector FP16/BF16 logical ops, Vector FP16/BF16 throughput)
  + time(exact layout-dependent Vector F32 logical ops, Vector F32 throughput)
  + time(exact layout-conversion SPM movement bytes, SPM service rate)
  + exact layout-dependent instruction executions * instruction_issue_ps
```

这里的Vector work最终由CT family执行；instruction term只表示发射/控制开销，不能替代NE或Vector执行时间。一个shared
conversion的actual descriptor及其dynamic execution只计一次，per-use conversion分别计；same-layout、exact metadata view和alias计0。
Layout conversion同时产生的Vector或SPM movement work必须进入各自term，不能只计descriptor。DDR/NoC work只有在current endpoint和
choice-local transformation contract能够精确投影时才进入PBQP；否则留给第9项物化后的actual movement和最终candidate objective。
SPM footprint/capacity仍只由MiniMalloc判断，任何duration term都不参与legality。

吞吐率和每instruction发射开销来自baseline/search共享的显式target performance profile，并携带其证据边界；它们不是IR语义或
legality事实。某项layout-dependent work、static execution multiplicity或对应rate不能从current IR、current interface和显式profile
精确取得时，该soft term对整个solve禁用，不能把单个unknown按0、统一`instruction_tick`、极大值或任意权重参与。没有可比较soft
term时，PBQP只在hard-feasible assignment间使用全局semantic tie-break，并明确不宣称performance optimal。Hard infinity只表示已证明
illegal；finite cost的加法、乘法和work-to-time换算使用checked arithmetic，任一影响比较的overflow返回`Indeterminate`，不能转成
infinity或`NoSolution`。

Solver必须区分`Optimal`、`NoSolution`、`Indeterminate`和`BrokenContract`，R0/R1/R2与residual core均受同一work budget约束；
全assignment tie-break必须与独立flat oracle一致。Baseline只接受`Optimal`结果；`NoSolution`、`Indeterminate`和`BrokenContract`
分别成为typed unsupported、resource failure和compiler error，不fallback canonical layout。Search的PBQP `Indeterminate`只表示首个proposal
不可用，不能删除raw合法域或形成no-good。Assignment选中后立即在各自candidate owner上创建actual
view/alias/allocation/layout materialization，随后销毁factor graph和assignment；下游不读取solver对象。

### 6.2 Movement

Movement choice以current producer value、consumer operand、exact demanded domain和physical layout为输入，选择local view/copy、
DDR store/load、Direct DTE、software relay或已定义collective。选择由唯一movement transformation立即创建actual typed ops、
staging buffer、token和effect。

Function/TileRegion observable result在bufferization前绑定actual destination。Compiler-created DDR→DDR `memref.copy`不是
正常output协议：同一result若可直接写最终DDR destination就使用DPS/out-parameter关系；region-local producer先在SPM形成后
发布时使用actual SPM→DDR store。真正的semantic DDR→DDR copy若不能消除，必须在本stage显式物化其Tile owner、有限SPM
staging、RDMA/WDMA、effect和completion，不能留到Instr conversion临时为每条copy创建TileRegion。

Movement不从shape、value名或future version ID恢复source/destination，也不先创建donor movement再替换。不同realization
使用同一transformation实现；每个alternative作用于自己的candidate transaction。跨region或跨Tile的每个非空domain
必须all-and-only覆盖，且每个movement op必须有current SSA owner和effect。

Movement形成后运行一次current-IR exact cleanup。只有full payload、same storage、same physical map且alias/effect/lifetime安全时
才删除transfer；partial、permuted、真正layout-changing、unknown ownership或不受支持的control flow全部保留。Cleanup与layout
creation共用`PhysicalLayoutRelation`/`TransferRealizability` proof，不保留Tile与Instr两套production eliminator。

### 6.3 Execution structure 与 rotating storage

Execution-structure choice只能从movement-closed physical TileRegion中的actual loop、compute、movement、SSA、effect和token重算。
Serialized choice不修改IR；software-pipelined choice由唯一current-IR transformation立即创建prefix/steady/tail、chunk control、
actual stage occurrence、rotating allocation roots、slot selection和loop-carried SSA。它不使用future event/buffer ID、预测lifetime或
SPM footprint，也不把`ExecutionStructurePlan`或buffer multiplicity传给下游。

无法证明recurrence、effect、slot reuse、external observation，或无法用current SSA/effect/token表达下游必须闭合的completion
obligation时返回typed unknown/unsupported；不在本stage
插join、分配offset、spill或退回另一structure。每个alternative作用于自己的candidate owner，成功后旧analysis失效并fresh重算。

### 6.4 TileRegion-to-Instr

Conversion按actual typed Tile op使用DialectConversion/RewritePattern生成canonical Instr。它不重新选择layout、movement、buffer、
execution structure、worker或completion，也不从上游plan恢复这些事实。输出Instr在每个Tile上显式保留actual loop/slot relation、
compute/movement issue、memref use-def、effect、token和control flow。

### 6.5 Worker、order 与 completion

Event/dependence graph只能作为从current Instr的operation、SSA、effect、range、token和control flow重算的query-local analysis。
它可以为scheduler枚举worker/resource/order choice，但不成为candidate identity或跨mutation事实源。

一个order choice应用后，actual block order、worker attr和token relation成为new current IR，旧graph失效。Completion owner随后从
该IR和已证hardware/runtime/ABI合同fresh构造minimum-strength、latest-unavoidable join/wait。不从TileRegion boundary、
loop backedge、movement类别或“保守”经验猜测completion。

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

### 6.7 能力 owner

下列能力只保留一个最终owner。Memory/target leaf发现输入缺口时只返回typed failure，不接管上游能力：

| 能力 | 最终owner与输出 | 不允许出现的位置 |
| --- | --- | --- |
| function-boundary与region-local bufferization、view/alias、materializing allocation | 6.1 layout/bufferization transformation；输出layout-resolved、function-boundary-bufferized current IR | actual memory/target leaf、SPM/DDR planner |
| 普通layout/bufferization allocation的创建位置 | 创建该allocation的6.1 transformation；allocation在current IR中的dominance/effect位置就是memory input事实 | MiniMalloc前的generic first-use sinking或lifetime改写 |
| software pipeline/rotating allocation root、slot selection和loop-carried SSA | 6.3 execution-structure transformation；输出actual loop与allocation roots | BodyEmitter旁路buffer plan、SPM planner按queue depth补建 |
| TileRegion-to-Instr、worker/order和minimum completion | 6.4/6.5 current-Instr transformation；输出completion-closed canonical Instr | bufferization、movement、execution-structure或memory planner |
| SPM lifetime/demand、MiniMalloc offset和accepted high-water/headroom | 6.6 actual SPM leaf，从上述current Instr fresh重算 | candidate proposal、footprint estimate或上游shape规则 |
| DDR offset/high-water、transport/resource与target acceptance | 6.6 actual leaf依次调用12、13、14定义的唯一kernel | layout/search shadow state或runtime重新planning |

`sinkStaticSPMAllocationsToFirstUse`不构成一项长期compiler能力：它若只服务memory leaf的synthetic fixture，应连同fixture期望删除；
若fresh production case证明allocation确实创建过早，则分别修正6.1或6.3的直接producer，不能把该helper迁成新的通用pass。
同理，组合式“bufferize + rebuild completion + memory planning”pipeline在leaf停止调用且caller inventory为零后删除；其中bufferization
和completion能力分别由6.1与6.5保留，不随wrapper删除。

## 7. `none` 与 `search`

### 7.1 独立 owner

`none`和`search`是两个独立compiler transaction：

| 边界 | `none` | `search` |
| --- | --- | --- |
| controller | deterministic baseline owner | bounded search owner |
| construction input | current TensorProgram + baseline fixed rules；无search choice state | current TensorProgram + explicit spatial/region/temporal choice |
| actual IR | baseline-owned direct materializer | search-owned structural materializer |
| downstream stages | 只消费baseline current IR | 只消费当前search candidate IR |
| feedback | 仅actual SPM capacity rejection生成确定smaller temporal successor | typed actual outcome返回frontier/controller |
| accepted result | 第一个通过全部actual gate的candidate | 预算内的retained best-known actual owner |

两者可共享policy-free source analysis、IndexRelation、single-op rewrite/conversion和actual memory/target leaf，但不共享complete
candidate schema、Card/Tile materializer、controller、fallback或accepted owner。

### 7.2 Baseline

Baseline的functional contract是：每个compute TileRegion恰有一个semantic root，跨root shaped dependency显式经DDR或已定义
peer boundary，单buffer、deterministic order和完整observable output。这些事实必须存在于baseline actual IR，不是
baseline plan的声明。

Baseline不构造Spatial/Region/Temporal search state、frontier、domain或complete-choice key。Baseline materializer在一次调用内从
current structured semantics和exact demand直接使用固定placement、single-root region和temporal规则产生actual IR；这些局部参数不跨stage
成为shared schema。

Baseline从full local temporal extent开始。只有actual MiniMalloc返回带current owner/conflict demand的capacity rejection时，
controller才按稳定semantic顺序选择受影响axis的下一个更小temporal choice。它不评分或保留whole-candidate alternative，
不调用search domain，不用预测bytes选tile size；第6.1.1节的policy-free PBQP只是对每个actual attempt执行一次的local layout
optimization，不形成baseline frontier或fallback。

### 7.3 Search

Search frontier保存显式choice key、work/budget accounting和move-only accepted incumbent。合法域由typed transformation capability
与current-IR verifier定义，不由workload名、shape特例或materializer fallback定义。

DP、memo、priority、dominance和LNS可以改变choice访问顺序和搜索工作，但不得用推算的IR/buffer/instruction inventory
代替actual result。一个choice只在物化为current IR并通过actual gate后才能成为accepted candidate。

### 7.4 Cost 与feedback

Source-IR-derived lower bound只能用于frontier ordering，必须标明不是actual cost。Candidate comparison使用物化后的TileRegion、Instr、
movement、completion和memory/target数据。推算结果不进入legality、SPM feedback或exact no-good。

最终winner objective从每个Accepted owner的final current Instr和fresh schedule/cost analysis计算，不使用
`aggregateInstructionCount * instruction_tick`作为compute cost。比较合同至少分别保留：

- 每Tile的NE FP16/BF16 logical work及target-profile NE throughput；
- 每Tile的Vector/CT FP16/BF16和F32 logical work及各自throughput；
- instruction issue/control、wait、DDR、NoC和显式SPM movement的独立work与service term；
- current control flow、effect、token和已物化execution structure决定的有限schedule/makespan。

NE与Vector的service time分别计算；instruction数量只额外计发射/控制开销。一条NE GEMM与一条Vector instruction即使instruction数相同，
也不能因此得到相同compute cost。当前硬件事实证明CT、NE是不同engine/completion domain，并有CT/NE与movement engine overlap的
profile内观测；尚无证据证明NE与CT彼此如何重叠。Current IR明确依赖或completion顺序的work按该顺序累加；没有依赖的NE/CT work
不能擅自按`max`重叠，也不能把强制串行的诊断上界冒充可比较的actual makespan。若候选排序取决于这项unknown，objective保持
incomparable。只有后续硬件文档和matched profile明确证明的并发关系才能增加对应schedule resource组合。

同一次winner比较的所有candidate必须使用同一target profile和同一组enabled terms。某个实际出现的NE/Vector work、所需rate、
schedule multiplicity或算术结果为unknown/unsupported/overflow时，该objective保持typed incomparable，controller只能报告
`FeasibleUnranked`或其它准确coverage；不得退回统一instruction cost，也不得把semantic tie-break伪装成cost winner。PBQP只使用上述
objective的choice-local精确投影来安排proposal，最终仍以物化后的actual objective比较。

本修改复用final Instr的现有work collector和duration analysis，删除search controller中的flat instruction objective；不新增operation、
attribute、Wafer-specific interface或legality verifier。Cost和duration仍是mutation后失效、可从current IR fresh重算的analysis结果。

Actual capacity rejection默认只对产生该current IR的完整choice有效。只有从actual owner/conflict witness可证明的有限条件
才能作为causal feedback；unknown、unsupported、timeout和compiler error不得改写为capacity rejection。

## 8. Ownership、analysis 与实现边界

- Compiler driver拥有policy routing、frontier/budget、candidate transaction和唯一winner handoff；不实现leaf rewrite。
- Analysis只读current IR和显式target configuration；mutation后默认失效，不把operation pointer或物化前identity传给下游。
- Transformation通过`PatternRewriter`/`IRMapping`或明确owner API修改candidate。每个transformation只有一个production实现。
- Conversion只读已经完整表达源stage语义的actual ops/types/effects，不补choice或repair。
- Event graph、lifetime、buffer demand和cost是可重算analysis result，不进入IR、candidate key或跨mutation cache。
- 如果下游需要一项无法从current IR重算的信息，先修改源IR表示，不增加side plan。

源码稳定职责为：

- TensorProgram analysis：structured semantics、exact demand、spatial/region/temporal choice domain；
- TensorProgram/Card/Tile transforms：structural materialization、layout/view/bufferization和movement；
- TileRegion-to-Instr conversion：deterministic target-abstract lowering；
- Instr analysis/transforms：worker/order、completion、lifetime和memory problem derivation；
- actual memory/transport/target leaf：offset、range、resource、ABI和CardExecutable acceptance；
- compiler controller：choice exploration、typed feedback、budget与winner ownership。

## 9. 实现迁移

Current迁移必须遵守：

1. 先为一个stage建立唯一actual-IR producer和直接下游test，再在同一work item删除旧shadow owner。
2. 不保留V2、mode switch、compatibility wrapper、fallback或baseline/search共享complete materializer。
3. 删除旧source前，将其独有的relation、algorithm和negative test迁到new owner；只检查旧plan字段或parity的fixture不迁移。
4. 旧archive、profile、package和generated output不参与current correctness或完成结论。
5. 新路径切换后对旧type、builder、domain、state、materializer、verifier、CMake、test和doc做零残留检查。

## 10. Verification and Done Criteria

### 10.1 覆盖矩阵

每个非小修work item使用rank至少为3、至少一个主要迭代维不小于1024的static shape。Spatial/temporal切分
成对覆盖`1024`整除与`1025`/`1031`非整除，并实际经过多Tile、多block/wave、remainder和tail。矩阵还需
覆盖chain、diamond、fanout/fanin、broadcast、reduction、view/slice、layout-compatible/incompatible、attention prefill/decode。

每个case必须断言当前stage承诺的exact coverage、owner、SSA use、alias/copy、movement、tail、effect、completion或
下游可消费结果。小shape只用于穷举oracle或最小负例，不代签production。

### 10.2 Current-IR 证据

- instrumentation on/off产生同一IR、candidate result和package；
- 每个candidate的actual Card/TileRegion/Instr owner只物化一次，winner不重建；
- 不存在代表future operation/value/buffer/event/schedule的跨stage状态或为其服务的parity verifier；
- layout/view测试检查actual SSA alias和copy数，movement测试检查actual typed ops/effects；
- schedule/completion测试从current Instr构造并检查位置、participant、token、动态次数和lifetime witness；
- SPM测试检查actual allocation、owner relation、conflict demand和offset，不检查预测footprint。

### 10.3 融合、IR膨胀与Instr汇总

启用compile timing时只从current choice和actual IR输出有界汇总，不参与candidate selection或legality：

- physical Tile数、TileRegion数和structured execution instance总数；
- 每TileRegion的structured execution数的minimum/average/maximum和singleton region数；
- 每TileRegion的actual nested operation数的minimum/average/maximum；
- region-local use、cross-region external use和actual DDR/peer movement数；
- local delivery按stored/reconstructed/direct-nested分类的edge数，以及每类实际producer materialization数；
- function-boundary bufferization产生和movement closure后残留的`memref.copy`按memory-space pair分类；进入Instr conversion的
  compiler-created DDR→DDR copy必须为0；
- accepted final Wafer Instr总数、per-Tile minimum/average/maximum和per-kind exact count；
- accepted final NE/Vector logical work、各自启用的throughput/service time、instruction-control term和最终makespan。

该汇总不逐region打印日志，不把structured execution数与raw operation/Instr数混为一个指标，也不构造
expected inventory。

### 10.4 End to end

- baseline和search分别从同一current FP16/BF16 source形成policy-complete Instr、actual memory plan、CardExecutable和package；
- 两条policy使用独立process、IR owner、ProgramData handoff和output directory，不互调或共享result；
- 两者均实际经过MiniMalloc、DDR、transport、target、strict package readback和no-card；
- timeout、OOM、skip、fallback或未进入actual planner不是通过；
- board-ready与真实设备证据分层，host/package/no-card不得称为board correctness或performance。
