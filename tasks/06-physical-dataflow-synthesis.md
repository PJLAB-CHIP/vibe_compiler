# Card 内 Physical Dataflow Search 与执行构造

本文是`TensorProgram -> TileModule set -> TileRegion -> Instr -> DeviceExecutable`主线中card-level physical dataflow的
唯一设计owner。当前任务状态和施工顺序只看`tasks/progress.md`与
`tasks/plans/physical-dataflow-synthesis.md`。历史plan和archive只作审计背景，不定义current pipeline。

## 1. 核心规则

Physical-dataflow search可以选择明确的transformation参数，但current IR是下游事实的唯一来源。

可以在物化前保存的是选择：

- structured iteration的spatial partition和Tile placement；
- TileRegion membership和显式replica choice；operation最终位于loop内或loop外不是choice；
- 尚未被exact relation唯一决定的temporal tile vector和dependence-legal loop order；
- attention fixed algorithm下的output/K2 spatial partition、contribution Tile和merge Tile；
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

跨Tile shared-DDR的Region DAG并不提供实际执行同步。13号记录了现有release/acquire完成缺口；
其协议及actual downstream验证闭合前，搜索物化和主机编译通过不能证明该候选板端正确。

## 2. Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  GSPMD完成card级分区、05号attention semantic recognition及bounded access-relation e-graph normalization完成后的
  verifier-valid card-local TensorProgram。SSA、structured iterator、canonical indexing relation、region、effect、type、shape和
  dtype已完整；尚未绑定Tile，且不携带e-class、rewrite history或提取side table。
- Current stage responsibility:
  none由baseline-owned materializer从current TensorProgram和固定规则直接构造actual TileModule/TileRegion IR，不创建search choice/domain/state；
  search才枚举Spatial/Region transformation choice并交给search-owned structural materializer。Materializer把selected graph attention
  直接变成每个actual Tile上的三结果online-attention、state endpoints和merge/finalize。两条policy随后从各自candidate current IR建立并
  立即应用temporal tile-and-fuse；online-attention的K2使用三个DPS state的stateful Tiling，之后确定性分解为Linalg/Tensor/SCF，再依次完成layout/view/bufferization、movement、execution structure、
  TileRegion-to-Instr、worker/order/completion，再以completion-closed Instr进入共同actual leaf。
- Output IR / files:
  policy-complete、verifier-valid的TileModule/TileRegion/Instr IR，以及由同一accepted owner形成的
  DeviceExecutable和ExecutablePackage。
- Downstream consumer:
  target conversion、device link、package emission和runtime launch。
- User-level driver / named pipeline:
  wafer-compile的typed `none`与`search`产品入口；局部测试使用注册named pipeline或同一compiler API。
- Explicit non-goals:
  不重做跨card GSPMD；不从名称、shape或workload恢复语义；不新建future-output IR或shadow candidate schema；
  不重新运行全图equality exploration，不让e-graph选择Tile、fusion、layout、movement或winner；不让lowering、allocator、
  communication或completion在失败后repair候选；不修改数值语义。
- Completion criteria:
  none从current source直接形成独立baseline attempts，search从explicit structural choices形成独立candidates；每条policy的
  transformation只保留一个实现和一个事实源，不通过mode-switched complete materializer共享；每个进入actual gate的candidate
  只物化一次；唯一MiniMalloc运行于同一current Instr IR；
  current主线不再含未物化physical value、storage object、event或schedule的跨stage协议。
```

## 3. 稳定 IR 边界

### 3.1 `builtin.module`

`builtin.module`是selected candidate的共同transaction、symbol和module-stage verification范围。它保留target
topology、logical mesh、shared DDR declarations，以及top-level
`wafer.tile.module(card_id=..., tile_id=...)`集合；自身不携带`card_id`，physical identity只存在于typed Tile modules和
topology中。它不保存candidate set、score、rejected alternative或side table。

### 3.2 `wafer.tile.module`

`wafer.tile.module`绑定唯一physical `(card_id, tile_id)`。不同Tile可以有不同op、loop、temporal shape、worker和执行长度。
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

### 3.4 Instr、DeviceExecutable 与 package

TileRegion-to-Instr转换产生candidate的current target-abstract instructions。Instr层显式表达engine issue、operand/result memref、
effect、token和control flow；worker/order/completion必须在该IR上物化后才能进入memory planning。

`DeviceExecutable`是已通过Instr、SPM/DDR、transport、resource、completion和ABI verification的唯一内存owner。
`ExecutablePackage`只序列化accepted executable与runtime必需数据，不序列化search状态或调度副本。

## 4. Search 输入、选择与candidate ownership

### 4.1 Immutable input

一次policy invocation可读取：

- 05号bounded access-relation e-graph normalization已经提交并verify的current TensorProgram、SSA use-def、standard interfaces和
  typed effect；physical planning不读取或重建e-graph；
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

- 本次新建或clone的最近`IsolatedFromAbove` candidate builtin module及其all-and-only TileModule set；
- current IR epoch内的SSA、region、structural boundary/buffer relation和effect；
- 可重算的analysis和本次rewrite使用的短生命期临时数据。

Cross-Region same-Tile edge直接由SSA连接，不保存relation。Cross-Tile structural relation只连接已经存在的source TileRegion result与
destination TileRegion input，不复制`DemandFragmentId`或Tile/Region identity；两端parent chain和current producer/consumer已经给出owner与
payload。它是candidate transaction的current-epoch relation，不是planning state。Temporal tile/fuse、
online-attention decomposition和layout/bufferization必须随IR replacement同步retarget，movement all-and-only消费后清空。
Relation不得携带future route、layout、buffer、storage、event、completion或offset，也不得进入search key、analysis cache、
Instr或package。

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

Search的constructive parallel proposal还应覆盖输入复用方向。对current Linalg中静态、projected-permutation的实际读取operand，
从operand element bytes和indexing map未使用轴的partition factor计算跨piece重复的逻辑访问量；在同等最大participant数下，
先访问该量较小的完整partition，使用完整semantic key打破平局。未知map或dtype保留原proposal排序，不按零开销解释。
原constructive方案、其它proposal和完整raw lazy domain均保留；该指标不声称已经产生DDR load、DTE、allocation或lifetime，
只决定explicit choice的访问顺序。每个候选仍经actual materialization、verifier、fresh analysis和唯一SPM/target/cost leaf；
不得把source访问量当成容量门禁或final winner，也不得因此强制通信。

本项直接下游为`SpatialState` exact demand与structural materializer，两条policy中只修改search的proposal顺序，none不消费它。
对照[Halide autoscheduler](https://halide-lang.org/papers/autoscheduler2019.html)的输入复用特征与
[Ansor](https://tvm.apache.org/2021/03/03/intro-auto-scheduler.html)分层候选搜索；这里使用确定且有界的源码访问指标，
不引入learned model、设备autotuning或第二套cost objective。完成条件是原合法域不变、M/N及置换结构有不同完整候选进入actual比较，
并由actual Instr和匹配profile验证热点改善。覆盖rank3/4、1024/1025/1031、4/16 Tile，窄M/窄N、等extent、置换map及未知map边界。

输入复用proposal还产生沿SSA双向协调的partition。参考[Shardy的数据流传播](https://openxla.org/shardy/propagation)的双向factor传播；本实现使用已有IndexRelation表达实际访问，而不增加按GEMM/conv名称分支的规则表。

传播冲突按显式优先级处理：domain/算法硬约束最高，其次是已有输出并行度，再次是producer/consumer对齐；可选归约切分作为较低优先的显式候选。
协调不能减少当前节点的并行分片数。新增并行切分与独立的可选归约seed冲突时，先保留并行切分，再按轴的semantic顺序撤去可选归约切分；
强制归约轴不撤去。仍不能满足Tile数量或domain约束时，放弃该协调提案。原seed与其它合法raw候选保留，最终优劣由actual IR成本决定。

独立轴证明直接检查完整IndexRelation的约束，限定系数工作量，只通过单位系数等式消去局部变量；轴仅出现在覆盖完整范围的自身约束中时才证明独立。
有限的整数反例可证明不独立；其它情况返回Unknown，不调用通用关系相等、集合差或整数采样求解。预算耗尽保持typed ResourceExhausted。传播的矩形像恢复使用同一工作预算：单位等式消元后，只对有整数精确性证明的上下界对做Fourier–Motzkin投影，
再以区间传播求必要边界，并逐约束证明整个矩形均满足它们。完整余数范围可以投影，带holes的范围不能冒充矩形。
参考[MLIR整数投影合同](https://mlir.llvm.org/doxygen/classmlir_1_1presburger_1_1IntegerRelation.html)及pinned实现；
直接调用通用投影不提供本查询所需的工作上限与整数精确保证。不能证明时保留seed，不进入集合差或整数极值求解。
该证明保留domain bounds、reshape约束及producer reduction fiber，不依赖translated rectangular tile map；未知结果不能清除seed轴。
覆盖rank3+、1024/1025/1031、归约producer到contraction、reshape、受限domain、优先级冲突及强制归约；检查分片覆盖、merge/raw域和actual下游，不扩张搜索预算。
与Shardy的固定点求解不同，此处只生成有界候选：从首个完整seed分别执行前向优先和反向优先的两次遍历，原seed、其余proposal及raw域保留。

每条边组合consumer迭代域、实际operand/support链与producer结果的逆关系。每个source shard必须映射为exact rectangle；
完全相同的矩形表示复用需求，合并为一个logical piece并确定性选择一个source Tile；部分重叠、非矩形或不能由当前
BalancedParts/UniformExtent精确表示的边界不产生协调候选。新target partition必须完整覆盖迭代域，并通过domain `contains`/`close`。
不再因consumer有归约轴或producer读取tensor而跳过。未被结果映射保留的归约轴由关系逆像恢复完整范围；若映射要求切归约轴，
从新axes重新推导全部reduction group。仍存在的group保留seed merge placement，新group选择其首个contribution Tile作为显式merge choice。
多个consumer反推同一个producer时，只有完整partition及placement一致才应用。前向仅根据实际读取operand尝试，按静态bytes与semantic ordinal排序。
目标轴只有经完整逆关系的轴不变性证明后才保留其独立seed切分，不能把单个full-extent image当成独立性证明。
前向以实际读取的data input协调，DPS init沿consumer需求反向协调，不反过来用初始化的seed覆盖计算选择。
没有新边界的whole-target需求保留seed；未改变或未被domain接纳的提案不能阻止继续尝试其它读取operand。
这些选择不声明复制执行、movement、同步或SPM合法性；所有候选仍走actual materialization与fresh分析。

普通partial reduction的DPS init只在merge处消费一次。ExactDemand与RootUseId使用`DemandDestination`明确区分compute shard和
reduction group；init的需求来自该merge完整contribution迭代域的精确像，目标Tile为已选择的merge owner。RootWork、RegionPlan、
replica与materializer消费同一typed use，group依赖排序因而包含init producer到merge的SSA边。partial贡献仍使用接口生成identity；
不向每个贡献广播原init，不在merge物化时按shape或位置补找source，也不偷偷重算init producer。
下游elementwise lowering通过pinned Linalg `getOpOperandsMatchingBBargs`读取body参数与operand的对应，不能假定每个body都有DPS init参数。
Selected tile产生的静态unit轴可在索引表达式内折为0（如`m+w`且`extent(w)=1`）；重建的projected map必须逐轴匹配当前operand/result shape，
不能因output map为identity而绕过input map检查或发出verifier-invalid Tile op。
Tile reduce选择native指令时同时验证Instr的rank≤4合同；更高rank的已选Tile IR走既有ordered reduction构造，不能发出非法Instr后再失败。
普通卷积通过Linalg convolution dimensions统一推导；1D按已知unit高度补成2D native输入/weight/result视图，stride/dilation在该轴为1，
输出再恢复原rank。该规则依赖索引接口和actual shape，不按named/generic op名称区分，也不改算术。
Sparse peer round matching只纳入同一Tile当前block中最早的待发送source Region；后续Region的payload不得因relation编号较小而被安排先接收，
否则source的实际release wait可能与receiver ready形成环。这只约束已有round选择，不插入额外join/wait，不改变transport或使用全局drain。
稀疏merge owner的actual输出也必须闭合14号既有统一entry结果合同：BoundaryMovement从当前输出destination推导完整result port类型，
每Tile每output index物化一个实际DDR root；同Tile多个piece共享，未写该结果的Tile不增加写回。Current-relations检查endpoint唯一性，
layout preflight从实际insert_slice检查同Tile同output的piece类型一致且不重叠，不能以Tile唯一性拒绝多个合法merge piece。不能到ABI层按shape猜测缺失输出。

#### Spatial admission、typed结果与relation协调的合同

```text
Pipeline position:
- Upstream IR / input:
  verifier-valid card-local TensorProgram、current StructuredDAG、target可用Tile及显式relation work limits。
- Current stage responsibility:
  保留空间域入口的typed拒绝；从current indexing relation生成双向完整partition候选，重建归约分组及merge placement。
- Output IR / files:
  typed planning admission结果、SpatialPlan候选、SpatialAssignment及ExactDemandOutcome；需求consumer区分shard/merge，无新IR或切分scheme。
- Downstream consumer:
  none/search driver的admission处理；PlanningSession、RootWorkDomain、RegionDomain及spatial materializer。
- User-level driver / named pipeline:
  none/search共享空间域语义；双向协调扩展search proposal，none仍使用canonical coordinate。
- Explicit non-goals:
  不推断未知op索引语义；不让collective退化为单Tile；不恢复任意切点scheme；不修改算术、同步、memory规划或原始IR。
- Completion criteria:
  正式入口将缺索引语义的root报告为unsupported；已建立domain后的unsupported demand仅关闭该空间choice。
  GEMM/conv两侧及generic等价结构产生可验证的双向候选，归约、multi-use与tail闭合；actual下游及fresh no-card验证通过。
```

域构建失败和candidate失败处于不同边界。缺少迭代空间或result indexing语义时，不能伪造单Tile域，也不能声称已经进入候选搜索：
`PhysicalDataflowPlanningProblem::create`保留builder的typed failure，search入口据此返回unsupported或compiler failure。
none的同类输入保持unsupported。Canonical coordinate只闭合结构选择，不将demand失败压成字符串提前退出；none/search均按
ExactDemandOutcome分类处理unsupported、分析预算耗尽及broken contract。域已建立后，`StructuredRelationFacts`保留root facts的typed失败，`PlanningSession`按
unsupported / indeterminate / broken contract处理当前choice。这不承诺换一个空间坐标就能支持缺少语义的op。

非投影result map但具有完整静态Linalg索引语义的root保留`BalancedParts(1)`，禁用该root的parallel/reduction切分；其operand/result
relation仍精确分析，普通物化负责创建完整计算。动态extent和未知索引语义不扩大admission。

仿射关系不保证保持当前两种切分形式。例如1025均分三份后沿`1024-m`反转，目标区间长度为341/342/342；关系精确但schema不可表达。
这是proposal的表达范围限制，不是“没有可构造输入”。本项以真实关系回归锁定此边界，不新增scheme。

覆盖矩阵：

| 输入等价类 | 整除/非整除、结构分支 | exact输出 / typed失败 | 直接下游witness |
| --- | --- | --- | --- |
| 非投影result map回归 | rank≥3，1024/1025 | 单cell、exact demand与原始IR不变 | actual TileRegion |
| tensor.pack root | rank≥3，1024/1025静态输入 | planning admission与none/search均unsupported；无伪造assignment | 正式driver分类 |
| relation构造预算不足 | rank17、静态主维1025，超默认变量预算 | none/search均indeterminate，不误判unsupported | 正式driver分类 |
| 已有domain但operand relation不支持 | 静态Linalg，合法result map | 当前choice unsupported，非compiler bug；后继仍可访问 | PlanningSession |
| producer → GEMM/conv及generic等价 | 1024/1025/1031，4/16 Tile，parallel/reduction分支 | 精确Cartesian coverage、重建merge group/owner、完整contributions | RootWork与actual spatial region、Instr/SPM |
| partial init来自实际producer | 1024/1025/1031，merge owner在contribution Tile或独立Tile | init只归属merge；60组actual region/temporal；较大partial保留typed capacity冲突证据，同一输入经正式search成功 | Instr、SPM及accepted executable |
| GEMM/conv → pointwise及反向producer需求 | 1024/1025/1031，投影/实际view链、重复需求 | full reduction fiber、重复矩形合并、稳定placement | exact demand与actual region |
| 多consumer一致/冲突 | 相同及置换访问 | 一致才协调；冲突保留seed；两种顺序均保留原seed | source不变、domain合法 |
| halo部分重叠、反向仿射、work limit | 1024/1025，矩形不可表示/unknown | 不伪造partition，原seed保留 | 独立区间期望及relation结果 |

### 5.2 TileRegion formation

对每个Tile的local structured DAG，region choice决定哪些root work进入同一TileRegion。一个producer相对当前Region只有三种
结构选择：位于Region外、在Region内实际存在一次、或明确允许为selected consumer实际复制。它不选择top-level/nested、stored/direct、
spill或future delivery。同region只选择共同local-storage scope，不证明SPM residency；只有actual producer work位于consumer
traversal内、中间值由direct SSA使用且无独立DDR往返时才称为coupled traversal。

Region choice只决定哪些actual operations进入同一TileRegion，不预先指定future producer delivery、nested placement或storage。
05号access-relation e-graph已经在policy分叉前对每个ordinary pure component及其ordered roots完成一次multi-root共享DAG extraction；
multi-use Access propagation不存在egg外rewrite。本stage不读取或重建e-graph。Structural materializer生成
all-and-only TileModules、non-nested TileRegions、actual spatial pieces、local SSA以及cross-boundary actual endpoint relations。
进入Spatial analysis前，policy controller对仍由reshape/concat等support chain返回的static shaped结果运行一次
`closeStructuredProgramOutputs`，形成direct DPS/Tiling output producer。它是verifier-visible current IR legalization，不是e-graph rule、
output名称约定或旁路映射；analysis和materializer直接消费该op，source TensorProgram随后整体由candidate transaction替代。
普通reduction contribution/merge必须已经是actual IR。Graph attention在这里被破坏性转换：FA每个output piece形成一个
`online_attention`；FD每个selected K2 contribution形成一个local `online_attention`，selected merge Tile形成actual state
merge/finalize和三个state endpoints。Candidate中不保留graph attention或empty shell。
其直接consumer是同一transaction中的SCF tile-and-fuse transformation，后者依据
current operation、use-def、indexing relation、effect和region boundary立即决定并执行fusion。Producer留在consumer loop外或
进入loop内只能是rewrite后的actual IR结果，不能由旁路 delivery 记录、布尔 rewiring 或其它非 IR 状态声明。

Static function input只被一个exact requested rectangle消费时，structural materializer在entry tensor boundary创建该rectangle的
`tensor.extract_slice`，并把compact slice作为TileRegion input。Full source argument仍是DDR程序边界，但不能先生成full-shape SPM carrier再在
Region内subview；否则即使compute只消费compact tile，actual allocator仍会看到完整input allocation。多个不同rectangle分别形成typed
boundary slice，相同source/offset/size在同一Region复用；offset/size只来自current exact demand。

Region choice必须覆盖fanout的每个use、reduction partial/merge、effect order和observable output。完全无依赖的
components不为扩大region而合并。Spatial choice中的`mergeTile`只在本次materialization中决定merge op所属的TileModule；物化后
merge位置由parent TileModule、参与者由actual SSA operands表达，choice立即失效。SPM residency不是独立上层choice，只能由最终actual
allocation/lifetime/offset证明。

RegionPlan的跨group dependency先形成一个全局确定性拓扑序，各Tile只发射该序在本Tile上的投影，不能再次按Tile局部排序而改变
actual execution order。local binding不能把producer tile result直接塞给consumer；必须用同一`IRMapping`将actual producer代入
current pure tensor support chain，保持reshape/slice/insert等typed indexing语义。Spatial materializer新增的cross-Tile
external/coupled/partial entry argument使用stage-local `wafer.cross_tile_boundary_input`标记其actual structural boundary身份；
movement消费对应actual relation并删除该argument，该attr不能越过physical movement closure，也不表示future buffer、route或movement。

#### 精确需求内的片段拼接

- Upstream IR / input：selected execution的`RootOperandWork`、exact operand demand、local/external fragment domains及当前tensor endpoints。
- Current stage responsibility：在spatial structural materialization内，把所选consumer需要的多个片段直接拼入其精确矩形；
  source slice仍使用producer坐标，destination slice减去需求原点。显式`tensor.insert_slice` support链使用同一局部需求机制。
- Output IR / files：紧凑`tensor.empty`、source `extract_slice`和相对坐标的`insert_slice`，交给同一个TilingInterface adapter。
  仅为调用原op tiler保留的full-type包装在本次物化内由实际matching slice替换并清理，不跨stage充当存储事实。
- Downstream consumer：temporal、layout/One-Shot Bufferize、boundary movement、Instr/completion及唯一SPM规划。
- User-level driver / named pipeline：正式`wafer-compile --optimization-policy=search`的spatial materializer。
- Explicit non-goals：不扩大或近似operand demand，不改变fragment ownership、Region choice、producer的observable输出或DDR/DTE选择；
  不引入rolling buffer、数值重排或另一条graph rewrite路径。
- Completion criteria：1024/1025/1031、多Tile、named/generic window及多片段fan-in的精确范围、相对offset、coverage和actual下游通过；
  非单矩形需求继续使用有限pieces语义，不能以bounding box冒充精确需求。

算法采用[MLIR DPS与subset bufferization](https://mlir.llvm.org/docs/Bufferization/)的显式destination构造：
先给出正确局部destination，再让bufferization决定alias/copy。对照pinned Tensor `foldExtractAfterInsertSlice`，该fold只消除
紧邻且offset/size/stride相同的insert/extract，不能消除多片段拼接后的较大extract；因此在已选spatial demand的物化owner直接构造，
不增加e-graph外的普通图等价探索，也不依赖canonicalizer消除完整allocation。
后续temporal concat查询的exact结论只证明片段值；static concat loop specialization还须在实际生成的slice上证明
offset grid、静态size及步长满足生成合同。重叠window或其它无法生成static pieces的slice保持读取已有紧凑assembly，
普通TilingInterface切分继续有效；不能把可选concat融合不支持升级成整个temporal变换的compiler failure。
同一次fragment assembly中，相邻片段若来自同一个actual SSA endpoint、相同source rectangle及相同destination rectangle，
只创建一次extract/insert。该规则利用tensor值不可变和相邻同址覆盖恒等式，不修改ExactDemand集合、owner或归约算术；
不同endpoint、不同rectangle或中间有其它写入时保持原顺序。覆盖多contribution共用DPS init的重复矩形，
并由named/generic contraction/conv、1024/1025/1031、多Tile与actual Instr/SPM回归消费。

### 5.3 Temporal tiling


本层的通用性按静态 current IR 的 interface、结果需求和 DPS 使用关系定义，不按 GEMM、卷积或模型名称定义。
独立 traversal、由 consumer 决定的输出范围、producer 内部尚可选择的归约分块是三种不同事实。

#### 融合中的自由归约轴与初始化

- Upstream IR / input：verified、pure tensor 的 current TileRegion；`TilingInterface`、DPS、实际 indexing maps、SSA uses，以及当前 temporal choice。
- Current stage responsibility：物化输出遍历及其 producer；保留未由输出需求确定的内部归约 tile size/order；按实际 slice 物化局部初始化。
- Output IR / files：普通 SCF、Linalg/Tensor 和仍待 decomposition 的 coupled-state op；所有 loop、state、init 与 main/tail 均已实际存在。
- Downstream consumer：既有 decomposition、layout/bufferization、Tile/Instr、completion 和唯一 actual SPM gate。
- User-level driver / named pipeline：none/search 共用 `buildTemporalDomain` / `applyTemporalTiling`；不增加第二条产品入口。
- Explicit non-goals：不支持 dynamic shape，不修改数值 operation/dtype/order，不改变通信、allocator、搜索预算或代价策略，不由估算容量决定是否合法。
- Completion criteria：普通 contraction、卷积、归约和逐元素 producer/consumer 的初始化及内部归约由同一实现处理；多结果状态整体物化，attention 不拥有另一套循环生成逻辑；整除和尾部均通过 actual 下游，失败分类保持 typed。

Joint domain 只能移除由 consumer 输出需求唯一确定的参数。一个 producer 的 output tile 要求完整 reduction fiber，
并不意味着 reduction fiber 内只能使用 full-extent 的计算块。由结果需求确定的 parallel 轴在 fused producer descriptor 中固定；
仍可分块的 reduction 轴保留完整 `1..L` 参数及合法顺序。该 descriptor 的 role 表达它在已物化的 producer tile 内生成循环，
不会再次在 region 顶层生成完整 producer。Independent domain 保持原有全部参数。

Direct、view、broadcast/window 与共享 producer 使用同一个 result-tile materializer。该 materializer 先经 pinned Tensor/SCF helper
生成 actual producer tile，再用该 producer 的显式剩余 choice 生成内部 SCF recurrence；实际嵌套 slice 随后按原有 subset 规则组合。
不按 clone 顺序、operation 名或打印文本恢复 producer 对应，不重新猜测 tile size。
内部归约生成新的输入 slice 后，继续沿已证明的 current producer/view 关系物化上游 tile，直到直接输入；
source proof 只活在本次调用，source 或 view 链依赖被修改或删除时由 rewriter listener 失效。对嵌套 slice 先用 pinned subset 规则组合，
不能先物化完整归约输入再仅切其 SPM subview。需要共同归约循环的 all-use group 唯一拥有其 consumer roots；这些 roots 不同时被另一条边消去，
以免共享 producer 的共同循环丢失或同一个 op 被两条物化路径处理。纯 parallel consumer 仍可共同派生到下游遍历。

当多个 live results 由同一个 parallel consumer 消费时，按全部 result indexing maps、DPS init read 和 selected traversal 证明共同输出域。
一次 output tile 只物化一次完整 producer，并一起改接所有结果。共享 state 轴、额外 observable use、无法证明的映射或不兼容顺序保留原遍历。
该实现以 `TilingInterface` 和 DPS 为循环/状态机制；已知 dialect 的只读 map adapter 只解释当前 op 的语义，不拥有另一条变换路径。

初始化由 actual slice demand 驱动。Scalar fill 的 main/tail 或多个消费者切片可以分别物化同值同 dtype 的 tile fill，
不依赖 fill 只有一个 use；原 full fill 只有在全部 uses 已改接后才能删除。`tensor.empty` 的切片对应局部 empty。
有实际旧值读取的非 uniform 初始化保持原 SSA 读取；不能将它替换成 zero/empty。初始化不拥有无消费者依据的独立搜索轴。

算法对照：MLIR [structured tiling/fusion](https://mlir.llvm.org/docs/Tutorials/transform/Ch0/)区分 result tile 与 loop transformation；
IREE [lowering configs](https://iree.dev/developers/lowering-config/)分别保留输出分块和归约分块。
本实现使用 pinned `SCF/Transforms/TileUsingInterface.h` 的 `tileUsingSCF`、`tileAndFuseProducerOfSlice` 及
Tensor `replaceExtractSliceWithTiledProducer`；不引入 IREE IR/config 或 newer upstream API。
Pinned `SwapExtractSliceWithFillPatterns.cpp` 的单 use 前置只能覆盖单个 slice，不能代签 main/tail 的全部初始化使用关系。

本项覆盖矩阵及本轮结果由 `tasks/plans/board-performance-optimization.md` 的通用 temporal 修复节维护。

每个actual traversal选择complete temporal tile vector和loop order，不允许单一标量`tile_size`代替多轴语义。第12项输出后，
规划阶段的 region/root 记录和预物化 temporal 状态均已在本边界消费，不能作为第13项的 operation identity。
Baseline和search分别从自己candidate内的live `TilingInterface` operation建立query-local domain；choice选中后
立即rewrite同一owner并销毁domain。需要试行alternative时，controller clone最近的`IsolatedFromAbove` owner并用该次clone的
`IRMapping`取得对应operation，不按名称、walk order或ordinal恢复。

Query-local descriptor和choice严格分开：

```text
descriptor（从live current operation/interface重算）
  exact iteration ranges
  per-iterator tiling capability
  dependence precedence

choice（只活到本次apply结束）
  iterator tile-size vector
  active-wave loop order
```

Operation handle只是拥有该current IR的同步调用期间有效的引用，不进入identity、key、cache或下一stage。一个TileRegion可以包含多个独立
traversal，每个current root分别建立scope；merge-only Linalg/state combine没有可tile的source interface时不伪造scope。Local extent必须来自
该operation的actual iteration domain，不能使用同一source node跨Tile的ceil maximum或bounding box。

TemporalDomain按一个未修改的current TileRegion建立并借用其中的live operation handle；descriptor、cursor和choice都必须在第一次apply或其它
IR mutation前销毁。它不提供稳定排序key，也不进入PlanningSession memo、UnifiedSearch prefix或ActualResultController key。Structural controller
只保留Spatial/Region choice；materialization后才在candidate owner上建立temporal domain并立即消费。一个structural key下只有在全部current-IR
inner choices已经闭合时，才允许把actual rejection提升为该key的exact rejection；单个temporal candidate失败不能剪掉整个structural choice。

对local extent`L_i`，typed `Tileable`轴的raw size域完整包含`1..L_i`，不要求整除，也不按native geometry、preferred size、SPM容量或
估算bytes删点；typed `FullExtentOnly`轴只有`{L_i}`。只有`tileSize_i < L_i`的active轴进入loop-order choice，order必须覆盖precedence
DAG的全部linear extensions。当前ordinary `TilingInterface` scope默认使用其完整iterator域；若某op只能full-extent遍历，必须由current
typed interface/capability明确给出，不能从名称或shape恢复。Temporal successor按current scope、size vector和order确定性惰性遍历；
interval proposal只改变先访问哪个size，不改变raw set。

Coupled reduction的parallel轴只有出现在全部state component indexing maps中，才能用现有serial SCF tiler分块；
任一component省略该坐标时，该轴是`FullExtentOnly`。否则同一state会按parallel tile数重复更新，
即使SPM placement合法也不保持数值语义。规则只消费`WaferCoupledReductionOpInterface`的current maps，
两种policy共享同一capability；actual TilingInterface还须拒绝违反此限制的直接调用。独立state复制/merge不属于本合同。

`online_attention`的parallel/output/K2轴使用同一个`TilingInterface`，K2 tile以三个DPS result携带actual
Accumulator/Maximum/Sum。FA在唯一spatial owner内形成K2 recurrence；FD的每个local contribution在自己的exact K2 interval内形成相同
recurrence。K1在该层保持full extent，decomposition后成为QK Linalg contraction的普通reduction codegen问题。第14项不再选择K1/K2、
contribution或merge Tile。

Temporal tiling已经产生actual `tensor.extract_slice`后，若其source是仅删除unit维度的collapse，
使用pinned Tensor的rank-reducing-slice simplification并组合连续slice，使外部tensor只按current tile实际需要的范围读取。
该规则在同一TileRegion的temporal canonicalization中运行，要求actual slice user；没有tile demand的普通reshape仍由05号
e-graph负责。不根据缩小后的shape预判SPM合法性，后续仍实际bufferize、lower和规划。
Boundary movement把SPM carrier的subview改接DDR输入或紧凑allocation时，必须保留原result shape所表达的rank reduction，
只从新的source layout重新推导offset/stride；不能调用无result type的builder重新引入已删除的unit维度。
直接覆盖从rank4输入、1024/1025/1031 temporal main/tail到rank3实际`tile.load`：source/destination shape相同，读取范围至多一个tile。
API依据为[MemRef subview合同](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefsubview-memrefsubviewop)，
以pinned `SubViewOp::inferRankReducedResultType`及`IndependenceTransforms.cpp`调用方式确认。

Boundary movement已经物化actual DDR destination与terminal `tile.store`后，对只用于收集已完成tile的SPM输出carrier做直接写回：
从actual allocation及其subview、原样转发的`scf.for` iter_arg/result证明完整use closure；嵌套循环的result须沿实际init/yield递归证明为同一个buffer。除terminal store外，只允许tile copy写入，
以及可证明source/destination为同一view的冗余copy；存在读取、算术更新、DTE使用、非原样loop yield、未知alias或其它escape时不改写。
DDR destination必须是支配全部tile写入的Region参数，在当前Region中仅由该terminal store使用，且已有actual allocation；
同一Region对该allocation存在其它alias使用时保留原IR。证明完成后，将原tile copy改为同位置的
`tile.store`到对应DDR subview，loop只携带DDR destination，删除完整SPM carrier与最终整块store；不改tile compute、数值顺序或消息。
该变换输入为完整bufferized Tile region，输出为显式per-tile store及DDR alias，直接下游为Tile→Instr、completion、SPM/DDR规划与target验证。
它不按shape/估算容量触发，不在allocator中spill，不创建future buffer，也不改变带read/merge需求的SPM state。

算法依据为[MLIR DPS与bufferization](https://mlir.llvm.org/docs/Bufferization/)的destination reuse/subset写入规则，
以pinned Tensor `InsertSliceOpInterface::bufferize`的destination subview与copy语义确认；当前阶段DDR destination已经存在，故只改写
已确定alias/effect的低层movement，不重新做Tensor in-place选择。覆盖rank3/4、1024/1025/1031、超过SPM的4096整除/4097尾部，
精确检查写回offset/size、无完整SPM carrier，并实际推进Instr/completion/SPM；带读取、变化loop yield或未知alias的负例必须保持原IR。

对于已物化的多结果 state producer及其唯一parallel Linalg consumer，temporal apply可以构造共同的输出遍历。
输入是current `TilingInterface` producer、其全部SSA results、scalar-fill DPS初值及本轮两者的tile/loop choice；不产生另一种state IR。
必须证明全部live state结果只供该consumer使用，consumer不读取DPS旧输出，各state indexing map组合一致，所有active输出轴都在
每个state中出现且允许parallel tiling。两者选择的共同轴tile size和parallel次序必须一致，原选择必须把parallel循环置于reduction之前。
无法证明、额外state consumer、不同tile grid、state共享轴或带reduction的consumer均保留原路径。

物化先建立consumer的输出tile循环，然后在该循环中只创建一次完整producer和同值同dtype的tile初值，再按原选择构造归约recurrence，
最后消费本tile全部state并写入输出tile。不得按不同result分别重算producer；不得改变原归约顺序、算术op或中间dtype。
多头、sequence和head dimension只由current maps/shapes决定，不使用模型名或固定长度。该变换位于actual temporal choice之后、
attention decomposition/layout之前；输出为普通SCF、online state及Linalg，直接下游仍为既有decomposition、bufferization、movement和actual SPM gate。
它不替代普通pure producer的e-graph探索，也不猜测容量、插入spill或选择DDR/通信路径。

算法对照[FlashAttention-2](https://arxiv.org/abs/2307.08691)的输出block内完成online归约及归一化，实际复用pinned MLIR
`SCF/Transforms/TileUsingInterface.h`的`tileUsingSCF`和Tensor tile producer helper。这里只改变已证明独立的输出遍历和state lifetime，
不采用论文中的数值或硬件专用改写。完成条件为完整state allocation消失、每输出tile只有一次三结果producer、K/V动态次数与顺序不变、
exact output main/tail及actual Instr/completion/SPM成功，产品4096×32-head fresh no-card和实卡PyTorch闭合。
覆盖1024/1025/1031、4096/4097、FP16/BF16；额外state use、DPS读取、不同grid/次序和共享state轴的负例保留原IR。

Producer tile是否由consumer tile唯一决定，只能由一次transformation调用内的只读exact tile-relation query判断。该query只读取current
producer/result、current consumer/operand、已经选定的自由tile参数、indexing map与`IndexRelation`/`ExactIndexSet`，不调用会修改IR的
tiling builder，也不把offset、extent、operation或SSA保存到跨stage plan。只有精确需求及其可物化表示已经确定时才能把对应producer参数
作为派生量移出domain。这里的确定性指tile需求，不要求输出点到输入点的关系single-valued；归约的完整fiber是一对多关系，
fiber内部仍保留自由切分参数。存在多个合法取值时继续完整枚举，无法证明或当前接口不支持时保留独立producer traversal和Region candidate。
查询结果区分exact、unsupported、indeterminate和broken contract；unsupported/indeterminate只关闭本次fusion机会，不签发resource结论，
broken contract终止该candidate。

当前exact-derived边界要求producer/consumer位于同一Region和block、producer pure、edge不是DPS destination，并由composed
`IndexRelation`证明实际tile demand。全部terminal uses由同一query收集，零长度、共同或分叉view路径与多use都使用同一需求比较；
受支持的rectangle/有限pieces均可形成Joint。未捕获use、effect、DPS destination、cross-Region或relation失败保持Independent。
复用与window overlap使用下述同一需求门禁。多结果producer不进入ordinary逐result fusion；符合完整state合同的唯一consumer由同一整体物化规则共用输出遍历。

Direct edge不是完整边界。Spatial exact-demand已经通过`WaferTensorIndexingOpInterface`和`IndexRelation`解释static pure
`tensor.cast`、`extract_slice`、`insert_slice`、`expand_shape`、`collapse_shape`和`pad`；该current-op relation构造必须抽为
Analysis/Linalg中的一个共享只读typed builder，Spatial demand与Temporal fusion调用同一实现。TemporalDomain沿same-Region pure support
chain逐段组合result-to-operand relation，并为每个all-use connected component保留independent与joint两类typed transformation choice。
Independent从全部current candidate建立原始scope；joint只移除由current consumer完整决定的参数，并保留 fused producer 的自由归约轴。两类choice各自有独立首项和完整lazy
successor，不把relation proof写进原始per-op size/order域。只有组合结果exact，且完整selected choice使consumer tile形成一个parametric
dense rectangle或work-bounded、互斥、static-shape exact pieces时，producer才可成为derived traversal；joint还要求全部current uses具有
相同iteration domain、tile vector、loop order和exact producer demand。
Relation unknown、unsupported、work limit、effect、DPS destination、未捕获use或不可表示reshape只关闭joint/fusion choice，independent
producer及原raw temporal size/order域仍存在，不把bounding box、完整shape或预测buffer当作tile。

Independent choice沿用pinned单root SCF mechanics。Joint choice建立一个common SCF loop nest，将每个root的DPS结果作为loop-carried
value，按root source order调用其`TilingInterface`形成actual tiles；共享producer的等价actual slices只调用一次
`replaceExtractSliceWithTiledProducer`并由所有root直接使用。General reshape先在actual consumer slice上计算exact image，再使用pinned
`replaceExtractSliceWithTiledProducer`、reshape/subset helper和producer `TilingInterface`向上穿透support chain，并在loop内重建局部view。
非线性reassociation维度full extent时形成一个parametric rectangle；selected tile使main至多一次且tail可静态化时，
`getExactStaticRectangularImagePieces`生成有限、互斥source rectangles，inverse relation逐项恢复consumer-local insertion rectangle。
每个piece通过producer `TilingInterface`实际物化，不枚举element或wave；其它tile size只关闭joint，independent保持原reshape和raw domain。
成功后删除dead完整producer/view链；失败由candidate transaction处理，不恢复shadow recipe。`tensor.pad`、`pack`和`unpack`已经有pinned
`TilingInterface`，在static pure tensor合同下可作为explicit traversal或derived producer；`insert_slice`/concat只对requested tile与source
segments的有限exact交集做tile-local assembly。Static segment边界落在一个完整tile内部时，materialization按segment边界形成互斥的
full-interior与boundary cases；每个case的extract/insert size必须是由static interval和canonical loop grid算出的常量，offset可以继续使用
current loop IV。Case数量随segment边界而不是loop trip count增长，不能用`arith.min/max/sub`结果作为shaped op的dynamic size，也不能把
static concat降成dynamic tensor。Constant Pad的非零padding轴必须在derived consumer scope保持full extent，避免把static
source变成无法被直接下游消费的dynamic padded tile；其局部Pad及pinned mechanics产生的constant `tensor.generate`在本stage确定性降为
tile-local Linalg fill/insert。Pack/UnPack在main/tail type收紧后使用pinned simplify pattern降为local reshape；fused producer的
`tensor.empty` destination折成tile-local empty，不能保留完整intermediate allocation。`linalg.fill`继续作为DPS destination初始化，不增加
独立search axis。collective、nonconstant Pad与dynamic shape不在本项范围。

Temporal materialization以一个canonical SCF loop nest承载同一traversal。完整块和remainder先共享同一个current loop body；
offset与bounded tile size由loop IV和exact upper bound计算。不得在结构层递归生成`first / steady / tail`的多维笛卡尔积，
也不得按wave trip count复制compute closure。当前Instr只接受static shaped buffer时，先形成上述canonical loop，再在直接
需要static shape的边界按内到外peel每个ragged loop的最后一个partial iteration，promote单次tail loop并canonicalize其bound；
不peel first iteration，不让tail specialization提前复制无关producer closure。若有`r`个非整除tiled axes，静态main/tail
组合最多为`2^r`，不能恢复三段式`3^r`展开。

Apply把full-extent size转成zero tile size，因此domain的第一个full-local choice保持IR byte-identical。其它choice以完整interchange permutation
调用pinned SCF tile-and-fuse，先用loop result替换原current op，再从内到外peel ragged last iteration；这样relation listener继续追踪最终SSA。
Peel后运行bounded Region-local Linalg tiling canonicalization，并只剥离static slice到更dynamic type的冗余`tensor.cast`，按actual DPS init
重建当前Linalg/online-attention type。该refinement不改变indexing map、payload或算术语义；其目的是让main/tail的`128`与`1/7`等actual static
shape直接被第14/15项读取。Domain不生成wave列表，也不按trip count展开body。

实现使用pinned MLIR的`TilingInterface`、SCF tiling和producer-fusion API作为loop/fusion的唯一mechanics owner。
Fusion control直接读取current SSA：producer必须位于同一actual Region、tile relation exact、effect允许移动，且不会因multi-use、
reduction/contraction或consumer tile overlap引入未选择的重算；cross-region和collective保持barrier。Explicit replica若作为
search alternative，必须先在controller拥有的candidate IR中实际创建producer operation，再进入同一fusion transformation，不能恢复
future delivery plan。相同consumer中的相同exact request由standard fusion后scoped CSE合并；多个consumer的independent choice共享一个
loop外producer，joint choice只在all-use exact条件下把producer tile放入共同loop，二者都不复制producer；只有显式replica choice才允许
实际复制。Exact但重叠且没有actual shared halo的不同request、unknown relation和effectful producer不融合。Reduction/contraction
不作为统一barrier：标准interface与all-and-only、无重叠result tile relation均可证明时参与fusion；其余保持current producer独立。
Tile-and-fuse后只运行有界local canonicalization、CSE和DCE清理本次新建的slice/view恒等式，不重新运行全图e-graph；
pinned接口暂时不能表达的exact reshape或`tensor.insert_slice` window只保留窄的current-SSA adapter。

#### IndexRelation需求与复用分析

- Upstream IR / input：同一current TileRegion内pure tensor SSA边、typed iterator/result/operand maps及support relation；已选静态tile size与loop order。
- Current stage responsibility：由共享只读关系构造和tile需求查询证明exact image、需求不变性、跨tile互斥和all-use一致性；生成能力与SSA/state legality分别验证。
- Output IR / files：查询只产生本次调用的数学关系与typed结论；apply通过既有TilingInterface/SCF生成actual tensor slices、局部compute及共同循环。
- Downstream consumer：temporal choice/apply，随后为既有decomposition、layout/bufferization、Instr/completion与actual SPM gate。
- User-level driver / named pipeline：none/search共用现有temporal入口。
- Explicit non-goals：dynamic shape、隐式重算、rolling halo storage、数值重排、搜索预算和allocator策略；不另建fusion IR或跨epoch analysis cache。
- Completion criteria：ordinary producer的全部uses由一个入口遍历和证明，direct为零长度view链，shared为多个terminal uses；旧direct/view/shared发现与准入入口、独立group schema及consumer map白名单删除；生成端拒绝与关系失败可区分；真实规模main/tail推进actual下游。

一次ordinary fusion query只接受一个current producer result，沿全部实际uses遍历透明typed support链，返回完整terminal-use集合及每条
consumer迭代域到producer结果的组合IndexRelation。任一未捕获use、effect、DPS destination或跨Region边关闭整个group；不得逐consumer部分提交。
相同consumer多次使用、多个consumer共享同一view、分叉view链及直接/view混合均使用同一遍历和关系比较。共享request须有可物化的共同
iteration grid，并在selected tile/order上证明相同producer需求；没有共同grid或需求不同返回typed不适用，不按map语法提前分流。
若terminal consumer自身已派生到下游遍历，须继续通过其current result/operand关系将需求组合到实际selected root，比较完整需求与grid；
不能把“没有独立consumer choice”解释成任意兼容。共同循环生成位置还须支配原有consumer result的全部使用；较早的observable use不能被越过。

生成描述仅表达既有helper能够执行的结果切片、参数化矩形或有限reshape pieces。是否采用共同循环及在哪一层共享，统一由group的
完整uses和selected需求决定。常规单use可以复用SCF producer-fusion mechanics，多个use可以复用共同循环和slice去重，view helper只负责
重建局部表示；这些helper不重新进行各自的融合准入。General reshape的当前piece生成范围与Pack/UnPack/Pad的TilingInterface限制保持
显式typed生成合同。非透明insert/concat assembly及多结果coupled state保留自身的语义物化合同，不伪装成ordinary unary relation。
Pack输入的关系由source坐标到outer迭代坐标的floor-div关系取逆得到，保留inner tile与outer permutation及源边界；一对多需求仍由同一关系协议消费。
仅增删unit轴的reshape在IndexRelation primitive中规范化为精确投影，identity composition只在完整中间box吻合时消去；
生成局部view时按已证明的轴关系对齐actual tile与consumer slice的形状精度，不能丢失归约主块/尾块的已知维度。

逻辑关系沿current SSA将consumer迭代域映射到producer结果，再由producer结果关系反推完整迭代fiber。
Linalg maps与typed tensor-support description是输入语义；IndexRelation负责组合与证明，不从op名称选择融合规则。
Selected tile family使用`x = q * B + r`和静态domain边界表达，B为本次已选常量，q为tile编号，r为tile内坐标。
需求查询证明整个合法grid及main/tail，不逐element或wave枚举。简单关系使用库内构造证明；其它关系使用有界Presburger查询，
超预算返回typed indeterminate，不以bounding box或只检查首块/尾块代替完整证明。

Broadcast与window是需求的性质，不是互斥的op分类。改变某个active tile坐标而需求保持相同，则该轴可共享同一producer tile；
只有全部需求相关active轴在selected order中构成prefix，且current SSA/effect/state允许，才把producer放在prefix之后、首个不变循环之前。
移除已证明不变的tile坐标后，不同request必须互斥；all-use共享还要求公共遍历上的exact需求一致。
Halo overlap仍保持Independent，让完整current producer作为实际共享值；不把需求重叠自动转成复制或新storage。

Relation exact不代签TilingInterface支持。Pinned Linalg result-tile helper仍要求projected result map；consumer的实际slice构造采用
`map(offsets)`与`map(sizes-1)+1`，temporal须确认它们与已证明的精确operand bounds一致。该限制属于局部计算生成合同，
不能当作分析语言的限制，也不能仅删除入口检查后让已放行candidate在helper中失败。普通result tile、view重建和有限piece assembly
复用同一producer materializer；复杂关系的生成能力按明确支持形式及其直接下游测试签发。
仅增删unit轴的局部view可以携带canonical loop内暂时动态的bounded size；main/tail特化及Linalg type refinement后，
从actual静态source shape与reassociation同步收紧Expand/Collapse result，避免静态source与动态reshape result不满足verifier合同。
对照MLIR [structured tiling/fusion](https://mlir.llvm.org/docs/Tutorials/transform/Ch1/)与
[slicing-based Affine fusion](https://mlir.llvm.org/docs/Passes/#-affine-loop-fusion-fuse-affine-loop-nests)：前者保留op tiling机制，
后者将计算切片与重复计算选择区分。本项只收敛关系分析，保持原有显式replica与actual IR合同。
具体API以pinned `TilingInterface.td`、`Linalg/Transforms/TilingInterfaceImpl.cpp`和Presburger relation实现/测试为准。

Candidate transaction的owner只由controller建立一次：已有candidate-owned IR时本stage直接rewrite，不再clone TileModule owner；只有试行
existing isolated owner上的alternative且caller仍需保留原IR时，controller才clone最近的`IsolatedFromAbove` scope。Standard tiling创建的
tiled producer是最终actual IR，不是scratch owner clone。Baseline独占自己的IR，不进入search clone或frontier。

Spatial和Region choice闭合后，唯一structural materializer立即生成candidate-owned TileModule/TileRegion、actual ordinary spatial
Linalg/Tensor/SCF，以及attention的online state contribution/merge/finalize和current boundary relation。Temporal domain随后只从这些
live operations建立并立即作用于该owner；下游不消费未物化execution/value ID，也不从Spatial plan重建TileModule/TileRegion。

### 5.4 Attention

Normalized TensorProgram中的`wafer.linalg_ext.attention`已将`flash_attention`或`flash_decoding`固定为graph fact。Spatial search只选择
output/K2 partition、Tile embedding、Region membership和FD merge Tile。Structural materialization直接消费这些选择：

- FA为每个output piece在唯一K2 owner中创建三结果`wafer.linalg_ext.online_attention`及finalize；
- FD为每个selected K2 interval在其TileRegion中创建一个local `online_attention`，在selected merge TileModule中创建actual coupled
  merge/finalize；本地state直接接SSA，remote state通过三个actual tensor endpoints进入merge Region；
- merge op不保存Tile ID；parent TileModule给出位置，SSA operands给出参与者。Spatial choice在成功物化后销毁。

第13项在这些current ops上执行ordinary/parallel `TilingInterface` tile-and-fuse，并对online-attention K2调用pinned
stateful `TilingInterface` SCF tiler。它不查看内部QK/PV，也不预构造score tensor。第14项随后只把已经tiled的
`online_attention`确定性分解成QK contraction、scale/mask、Maximum/Sum/Accumulator update、PV和tensor slices；不重新选择tile、
contribution或merge owner，不接收future inventory，也不clone整个candidate owner。进入layout时两种attention op都必须为零。

Decomposition使用一个module-level preflight/apply kernel：score map只包含current B/M/K2 coordinates，QK reduction K1，row max/sum和PV
reduction K2；scale、mask、`math.exp`与三个DPS state按05号固定dataflow形成actual Linalg/Tensor/arith/math。它不新增loop或finalize，
FA/FD共享同一实现；既有SCF loop、FD endpoint和selected merge只由current parent/SSA保留。Named pipeline与controller adapter复用该kernel。

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

#### 6.1.1 Layout assignment 与 exact PBQP

Layout合法域直接从current structural TileRegion的SSA value/use、consumer interface、exact `IndexRelation`和可验证encoding构造。
Baseline与search都调用同一个query-local PBQP layout optimizer；它不是search state，也不共享两条policy的candidate owner。
Baseline每个actual attempt求解并应用一次确定性assignment；search允许对同一实际IR约束合法value/use layout后，
使用同一PBQP求解器产生备选。每个备选在独立actual owner上apply并bufferize，再经完整下游比较；不枚举虚构buffer。PBQP在当前IR上按实际 materialization 的 physical bytes（含 padding）与一次 materialization unit
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
等materialization数的assignment使用完整stable semantic tie-break。Hard infinity只表示已证明illegal；finite materialization count溢出
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
oracle一致。Exact search完成时返回`Optimal`；预算耗尽但incumbent仍合法时返回携带完整assignment的`Feasible`。两种成功状态使用同一
assignment类型和唯一apply实现，不建立第二条layout lowering。`NoSolution`与已验证incumbent并存是`BrokenContract`；没有合法canonical
assignment的source在mutation前按typed unsupported停止，不能猜测layout或把问题推给下游。
Assignment选中后立即在各自candidate owner上创建actual
view/alias/allocation/layout materialization，随后销毁factor graph和assignment；下游不读取solver对象。

第17、18项的每个accepted product attempt必须恰调用一次PBQP并得到`Optimal`或`Feasible`，且两者都携带完整、factor-valid并已apply的
assignment；记录status、variables、factors、solver work、wall以及apply后的actual materialization数。`Feasible`只表示本次没有完成
最优性证明，不得称为materialization-minimal。`Indeterminate`只允许在没有合法incumbent时返回，并阻止规定产品case完成；不能通过提高
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

## 7. `none` 与 `search`

Actual leaf在fan-out完成后，Tile-to-Instr、局部transfer cleanup和NCC completion各自只修改独立Tile module及其owner relations，
使用已有bounded Tile executor并行执行。每个阶段结束后按Tile顺序收集typed失败、统计和IR；跨Tile DTE/DDR completion仍在共同边界执行。
此并行化不改变候选顺序、预算或数值语义。覆盖1024/1025/1031、16 Tile和1/4/16 worker，串行与并行的final Instr及实际SPM offset相同；typed失败按Tile顺序收集。
生产搜索不借用外部工具的进程期限截断trials；用户显式主机进程期限只作为取消边界，不报告为完整预算比较。

### 7.1 独立 owner

`none`和`search`是两个独立compiler transaction：

| 边界 | `none` | `search` |
| --- | --- | --- |
| controller | deterministic baseline owner | bounded search owner |
| construction input | current TensorProgram + baseline fixed Spatial/Region rules；无search choice state | current TensorProgram + explicit Spatial/Region choice；temporal domain只在actual structural IR上建立 |
| actual IR | baseline-owned direct materializer | search-owned structural materializer |
| downstream stages | 只消费baseline current IR | 只消费当前search candidate IR |
| feedback | 仅actual SPM capacity rejection生成确定smaller temporal successor | typed actual outcome返回frontier/controller |
| accepted result | 第一个通过全部actual gate的candidate | 预算内的retained best-known actual owner |

两者可共享policy-free source analysis、IndexRelation、single-op rewrite/conversion、structural transformation实现和actual
memory/target leaf，但每次分别拥有自己的materializer invocation、candidate owner、controller、fallback和accepted result；不存在
共享complete candidate schema或一条policy调用另一条policy的路径。

### Contraction累加精度边界

- Upstream IR / input：已完成attention识别与结构规范化的TensorProgram；普通FP16/BF16乘加contraction及原DPS init。
- Current stage responsibility：在任何Spatial/Temporal K切分之前，把累加状态显式物化为F32，保留低精度乘法输入和原逻辑输出dtype。
- Output IR / files：标准mixed-precision Linalg、F32 init/result，以及原contraction输出处的一次显式truncation；不新增数值op/profile。
- Downstream consumer：同一Spatial/Region/Temporal domain及materializer；partial、merge、SPM allocation和DDR/DTE以actual F32 SSA为准。
- User-level driver / named pipeline：普通none/search入口及`wafer-promote-contraction-accumulation`调用同一transform。
- Explicit non-goals：不删除K choice，不改变attention算法，不重排任意elementwise/reduction，不扩大外部张量dtype；不隐式开启psum alias。
- Completion criteria：named/generic、置换、非零init及多use，F16/BF16×1024/1025/1031实际分块与tail，下游mixed-format target及原PyTorch容差通过。

这属于显式数值合法化，不是ordinary e-graph语言中的同dtype代数等价式；e-graph仍独占原有关系等价探索。
只对精确的低精度乘加payload适用，任意附加算术、未知combiner或不同输入dtype均保持原语义。已有F32 accumulator不再改写。
标准Linalg允许输入提升到输出/累加类型，参见[MLIR Linalg](https://mlir.llvm.org/docs/Dialects/Linalg/)；
宽partial与最终epilogue分离可参考[CUTLASS split-K](https://github.com/NVIDIA/cutlass/blob/main/examples/06_splitK_gemm/splitk_gemm.cu)，
本仓仍保持已有K遍历和显式merge顺序，不据此引入新的归约树。

上述cast是Tensor层的数值边界，不要求独立硬件convert。实际layout/bufferization后的Structured→Tile lowering把F32 init/state
作为GEMM的显式psum读取。Boundary movement关闭Region桥接、实际搬运可见后，统一执行最终输出融合；沿完整输出的copy/layout链，在无其它观察者且可删除的写入只触及私有storage时，
把最终cast合并到GEMM result dtype。若cast读取静态K循环的最终状态，只剥离实际最后一次迭代，先前各块仍写F32；
显式tail本身即最后一块。所有判断读取当前SSA、effect和view关系，未知write/alias、多use和外部可见写入保留原转换。
跨Tile merge与非GEMM计算不据此消除；完成后重建actual owner relations，直接交给原Tile→Instr/completion/SPM路径。

独立Region本身不是保留该转换的理由。对actual私有DDR allocation，沿TileRegion operand/block argument的精确关系，
证明唯一完整store、唯一完整load及没有其它观察者后，可把最终输出format沿这条实际传输链传播：
最后GEMM输出原dtype，DDR allocation、Region argument、store/load和consumer SPM allocation同步使用该dtype，
删除末端convert。中间K state不变，Region和DDR/DTE选择不变；actual effect/lifetime/owner由后续同一路径重新验证。
外部/逃逸buffer、额外F32观察、多个writer或非完整view保持原语义；不能只按shape相等猜测唯一producer。
此处采用当前memref use/effect闭包，遵循[MLIR Bufferization](https://mlir.llvm.org/docs/Bufferization/)的别名与观察者边界；
不重做bufferization，不发明未来传输或沿未物化Region推测地址。

### 7.2 Baseline

Baseline的functional contract是：每个compute TileRegion恰有一个semantic root，跨root shaped dependency显式经DDR或已定义
peer boundary，单buffer、deterministic order和完整observable output。Baseline不因识别到collective而额外合并Region；
已有基本peer/causal DDR物化规则只消费固定Region中的actual需求，不读取测试期望。这些事实必须存在于baseline actual IR，不是
baseline plan的声明。

Baseline不构造Spatial/Region/Temporal search state、frontier、domain或complete-choice key。Baseline materializer在一次调用内从
current structured semantics和exact demand直接使用固定placement与single-root region产生structural IR，再从其中live operation取得full-local
temporal choice并立即apply；这些局部参数不跨stage成为shared schema。

Baseline从full local temporal extent开始。只有actual MiniMalloc返回带current owner/conflict demand的capacity rejection时，
controller才按稳定semantic顺序选择受影响axis的下一个更小temporal choice。它不评分或保留whole-candidate alternative，
不调用search domain，不用预测bytes选tile size；第6.1.1节的policy-free PBQP只是对每个actual attempt执行一次的local layout
optimization，不形成baseline frontier或fallback。

### 7.3 Search

Search frontier保存显式choice key、work/budget accounting和move-only accepted incumbent。合法域由typed transformation capability
与current-IR verifier定义，不由workload名、shape特例或materializer fallback定义。

Region proposal的目标是在有界actualization内提供不同Region数量且boundary placement经过改善的完整partition，而不是沿一条merge path增加更多
prefix。每个Tile component继续直接使用current `RootRegionWork`和`allowsRequiredLocal` relation；cannot-link、group connectivity、use-binding
totality和contracted dependency DAG只由`RegionDomain`现有合法性检查。不得增加持久Graph/Hypergraph、MergeForest、PartitionPlan或其它
RegionPlan平行表示。

Builder先用maximum-gain feasible matching生成不同Region数量的seed，再在固定Region数量下执行bounded FM-style refinement。一次move只把一个
boundary root移到相邻group，且移动前后source/destination connected、binding totality、cannot-link和quotient acyclicity全部成立。每个root在一轮
refinement中最多移动一次；可以经过结构metric暂时不改善的move，但只发布本轮best-prefix对应的完整RegionPlan。Move priority依次比较
known exact localized bytes、local binding数和unknown localized binding数；unknown bytes不按零处理。所有query labels、candidate move和temporary score
在当前调用结束时销毁，下游从不读取它们。

Structural metric分别保留Region数、external/local binding数、known exact localized-use bytes和unknown binding数，不用任意权重压成legality。
本项不对publication closure打分；仍在group外的relation继续作为external binding，直到actual materialization和movement根据current IR决定其实现。
Structural metric只决定proposal访问顺序；不同结构、replica、传播与复用seed均保留入口，不能把局部metric支配当作
最终候选的cost支配。Raw lazy successor集合不因proposal/refinement而缩小；不声称有限预算证明全局最优。

Search由可恢复的候选会话拥有实际IR checkpoint与当前epoch的domain/cursor。一次轮转最多执行一个actual候选；
失败尝试也计费。候选会话暂停时保留原owner和未访问后继，继续时不重复物化已完成的前缀。
Controller接收每个实际leaf的typed结果；leaf更新与结构域关闭分开，未穷尽的容量失败不能成为结构no-good。

容量反馈在allocator返回精确证书、allocation与current owner仍存活时同步读取；只返回本次选择的参数坐标，
不保留失败IR的裸句柄。Region body由同次`IRMapping`关联到未变的上层choice，body-preserving变换继续使用同一block；
通信Region合并等销毁该边界的变换丢弃关联。实际冲突allocation通过current storage-root/owner relation关联到body，
只有该body的Temporal domain存在唯一scope时才据此缩小其可tile维度。多scope或无关联保持unknown并访问普通后继，
不能按allocation大小、诊断位置、遍历序号或所有Region统一缩小来补归因。观察回调不修改IR、不参与SPM合法判定。
成功候选的同一actual executable交给全局incumbent，后续搜索不得按choice重建winner。

Public limits只有`width`和`trials`，默认8/42。`width`限制同时保留的可扩展分支数，不限制累计访问的结构数；
`trials`限制实际候选尝试总数，包括在物化/下游失败的尝试。CLI仍为`--search-width/--search-trials`，C++仍为
`OptimizationConfig::search(SearchLimits)`。不另设每结构Temporal上限，也不在首个可行Temporal后停止。
同一source、target和width的更大trials延续同一确定性序列，已找到的最佳actual objective始终保留。

调度轮转探索、容量修正、候选改进三类工作，每类每轮最多一个实际尝试，空类跳过。探索先给不同结构入口，
再遍历参数；容量修正优先最早开始且仍有实际证据的修正链，配额用完只暂停；改进对保留候选提出成组邻域。
Spatial seed之后交错访问axis tuple的规范placement witness与完整raw placement cursor；二者按完整typed choice去重。
轴投影复用同一domain successor和closure，不改变raw集合。Region/layout/movement各保留实际checkpoint并逐leaf轮转，
某个完整布局的所有后继不会阻止另一个Region closure获得首次尝试。
尚无可行leaf的分支同样必须获得续跑：新结构与已保留的Explore分支交错，不能每次容量反馈无法定位就只启动新结构。
Joint/Independent各自保留初始参数和合法extent区间的几何中点参数（向上取2的幂），作为普通数值入口；
多结果producer和唯一consumer另从actual result/input projected-permutation maps提出协调参数：共同迭代维度使用一致tile，
仅部分结果携带的广播维度保留完整长度，使已有共同输出遍历可被选择。原参数仍保留；相邻几何尺度只作普通候选，
不预测SPM合法性、不修改数值顺序。此query输出typed TemporalChoice，直接交给唯一Temporal物化和fresh下游检查。
它们在任何actual容量结果之前生成，不读取SPM大小、footprint或预测lifetime，也不用于合法性剪枝。
每个参数都实际物化并经过唯一SPM门禁，原raw successor继续保留。

通信transport参数按同一boundary preflight的current component选择。每个component用其中一条现存SSA boundary edge
作query-local anchor；clone只经同次`IRMapping`重映射，物化前重算component并拒绝缺失/重复anchor。
默认DDR/Peer坐标保留，混合transport使用惰性二进制tuple而非只枚举单component修改；现有collective算法参数与该tuple共同物化。
选择Peer仍须满足实际cut、effect与completion合同；只按成功产物的实际DDR/DTE流量计费。覆盖两次连续exchange的四种
transport组合、4/16 Tile和1024/1025/1031，核对消息数、共享DDR范围、双完成域及actual SPM，不按测试名选路。

Layout placement把已选转换在首次use处物化，或在输入Tensor SSA不随循环变化时移到最内必要循环之前。
只跨越静态非空`scf.for`，不跨conditional、未知effect或mixed memref操作；仍在同一TileRegion内。
两种placement使用同一PBQP解、同一One-Shot Bufferization及完整实际内存门禁，延长的实际lifetime由下游重新分析。
搜索保留两种结果比较，不能用估算buffer大小决定是否允许提升；不重排算术、不增加shadow buffer。
Tensor不可变及重新bufferize的依据见[MLIR Bufferization](https://mlir.llvm.org/docs/Bufferization/)。
候选池优先保留不同结构，同类可执行分支按实际估时排序；全局最佳executable独立持有。池满且没有可替换的
已评估分支时继续推进现有分支，保留结构生成cursor，不销毁未完成的容量修正链。
下一项工作类别与待执行容量修正的保留状态分别报告。内部轮转到proposal或已有前缀时，尚有actual反馈修正排队的
session仍不可被新结构替换；修正队列排空后才恢复通常的结构多样性与估时替换规则。
结构分类只决定顺序；去重要求typed choice相同或实际IR等价证明，流量计数相同不能作为等价证明。

显式replica必须携带所选required producer execution的完整输入fragment；这些输入从同一canonical demand导出，
不能只复制计算root再由materializer猜上游值。RegionDomain同时检查新增输入的producer→consumer依赖；materializer
在同Region使用已生成的required SSA结果，在其它Region建立显式endpoint与必要movement，不隐式递归复制producer链。
replica输入与mandatory输入分别验证all-and-only，多个replica读取同一片段允许共享实际Region input。
覆盖leaf与多层producer、DPS init来自structured root、same/cross Tile及局部/最大融合，rank3+、1024/1025/1031；
下游必须包含actual TileRegion、完整boundary relation与Instr/SPM，不以raw proposal存在代替可执行性。

组合邻域覆盖Spatial与producer/use/通信、Region融合与Temporal/驻留、layout与转换位置/共享、tile与双缓冲流水。
Region proposal的全合并标签若因不连通等结构约束不可构造，使用同一合法合并序列的最终分组作为融合入口；
不能只保留该序列的中间样本而丢失最终合法融合方案。该入口与baseline、replica先于合并数量采样，raw domain不变。
Temporal普通参数入口同时保留全extent、每scope仅缩小最大可切轴、全可切轴几何中点及协调状态的尺度邻域。
只缩小一轴的入口保留其它轴的复用与指令粒度，避免所有维度一起缩小产生大量小指令；按extent和轴序确定顺序，
不使用SPM估算或算子/模型名。所有入口仍经typed domain、actual transformation、verifier和同一actual leaf。
同一参数入口先访问已有的基础DDR/Peer transport备选，再访问其共享输入/流水组合，不能将组合插到尚未访问的另一transport之前。
结构session内部同样轮转参数proposal、actual容量修正与已物化前缀的后续layout/movement，向外层报告下一项实际工作类别。
不能因旧参数仍有capacity repair就阻止已可行参数继续比较，也不能先耗尽全部参数seed才恢复已有前缀。
上游choice改变时从存活的实际祖先checkpoint产生新candidate并重建下游analysis。各rewrite分别verify，完整组合
随后经completion、唯一SPM规划和target gate；单项不改善不能直接排除组合。流水估时消费actual Instr依赖及worker，
详细信息不足时仍给标记质量的标量粗估；不从aggregate假造schedule。

Accepted objective按同一profile的标量estimated duration比较；storage与semantic key用于稳定tie-break。
性能估计不能参与capacity admission、hard pruning或同步合法性。Limits只管理工作量，none不接受search limits。

方法比较采用[Ansor](https://www.usenix.org/conference/osdi20/presentation/zheng)的结构/参数分层、
[Halide GPU autoscheduler](https://aekul.github.io/gpu_autoscheduler/)的结构多样性保留；当前使用确定性小预算队列，
不引入学习模型或以历史schedule trace重放actual owner。具体实施与本轮矩阵在统一板测计划。

Candidate形成`TileExecutable`前必须调用target ABI preparation共用的exact program/DDR function-boundary verifier；argument/result binding不完整的
Spatial/Region candidate在controller admission前返回typed failure，不能先作为Accepted winner保留、再由最终target codegen首次发现错误。

DP、memo、priority、dominance和LNS可以改变choice访问顺序和搜索工作，但不得用推算的IR/buffer/instruction inventory
代替actual result。一个choice只在物化为current IR并通过actual gate后才能成为accepted candidate。

### 7.4 Cost 与feedback

#### 搜索空间与失败审计

本轮系统性审计只消费current source、显式Spatial/Region/Temporal/movement choice、actual Instr及typed gate结果。
输出为已有compile counters、失败diagnostic与本轮workload/预算报告，由16号验证合同和统一board-testing消费；
不在审计期间修改候选排序、budget分配、SPM admission或同步语义。
Shared-DDR/Direct-DTE成环失败应附一个完整actual cycle，逐边注明Tile顺序、recv-ready、token completion或
DDR publish/acquire来源，并标识实际Tile和operation。只在失败路径恢复witness，不新增持久依赖图或按文本控制流程。
coverage包括无环、双向ready等待、混合DDR/DTE环和单执行region；真实规模产品失败及现有completion负例验证
同一typed结果与边序列。搜索proposal数量由现有计时session记录，与实际访问数区分；日志不足不能声称空间已穷尽。

Source-IR-derived lower bound只能用于frontier ordering，必须标明不是actual cost。Candidate comparison使用物化后的TileRegion、Instr、
movement、completion和memory/target数据。推算结果不进入legality、SPM feedback或exact no-good。

最终winner objective消费每个Accepted owner的final current Instr及fresh `InstructionProgramAggregateCost`。
`Analysis/Instr/ScheduleCostAnalysis`拥有原始工作量；`Analysis/Instr/CostModel`唯一拥有参数、耗时公式和比较。
输入为逐Tile实际工作量、整卡DDR/NoC统计及不可变`SearchCostCohort`；输出为ps标量估时、资源分项及估计质量，
直接消费者为`ActualResultController`、`SearchCurrentIR`和`UnifiedSearch`。生产入口仍为
`wafer-compile --optimization-policy=search`，不修改IR、同步、SPM合法集合或winner owner。

同一Tile的NE FP16/BF16、CT FP16/BF16/F32、显式SPM movement分别按work/rate计时，
加上DTE endpoint payload、首条/后续sender生命周期及instruction/NCC控制，先逐Tile求和再取最大。
整卡共享DDR按总read+write bytes/rate计一次；NoC link与endpoint承载同一payload，
只补`max(0, peak-link-time - maximum-endpoint-time)`，另计hop估计。
只有aggregate或无法解释当前执行结构时，采用Tile内串行服务、Tile间并行的显式近似。生产比较还可传入仍存活的accepted
Instr modules：CostModel只读actual worker/family、block顺序、SSA/view/slot、typed effect与NCC participant，按资源服务时间
和根级读写依赖估计最早完成时间。同一engine保持顺序；不同engine只有无依赖且未被实际completion隔开时才估算重叠。
根级范围合并会高估依赖，不改变IR。两槽流水直接解释current select/loop-carried buffer；固定循环在携带buffer两轮复现且
没有变化的index carry时，取前五轮，以最后两轮服务增量外推其余完整两轮，并单独解释奇数余轮。其它循环在统一work上限内
继续解释，超限采用有限粗估；不能只平移时钟却丢失后续操作读取的index结果。这是有界性能近似，不是周期模拟、hardware保证或hard pruning bound。
估计器有固定分析工作上限；动态控制、未支持的完成域/alias或超限时回到上述有限串行估计，不给出不可比。
整卡DDR总服务仍为资源下限，与估计的最长Tile路径取最大，不能与已经包含的DDR服务重复求和。所有公式与近似仅在CostModel；
controller只传递current IR并比较返回标量。IR修改后旧估计失效，模型不保存跨stage的buffer、schedule或completion事实。
NE与CT仍各用自己的吞吐率，instruction数量只计控制开销。Storage仅在时间相等时按固定tuple打破平局。

DTE无send时startup为0，有n条时为`first + (n-1) × steady`；使用已有首条13us、后续1.5us先验。
NCC使用已有每call 0.14us和每participant 0.045us，先同Tile组合再取最大。
Sender生命周期已包含send wait，不重复叠加完整wait样本；额外wait项仅为既有微小控制估计。
吞吐和hop先验保持显式profile身份，不把少量板测样本当硬件保证。此次不新增专项板测。

当详细work缺失或格式没有校准时，使用实际instruction执行次数（依次取exact、有限upper、static sites）
和统一每instruction服务先验形成粗估，并标记估计质量；这只是排序近似，不宣称动态次数已知。
算术溢出使用饱和值并标记粗估，不能绕回零。正常同cohort合法候选始终有可比较的数值；
缺cohort或混用profile是调用合同错误，不作为DDR/DTE胜负结论。估计不得充当hard pruning bound。

方法比较：[OpenXLA性能模型](https://github.com/openxla/xla/blob/main/xla/service/gpu/model/gpu_performance_model_base.cc)
按资源服务时间和目标overlap假设构造标量；本仓使用有界current-IR服务估计及串行粗估，不移植GPU常数。
[实现驱动的collective模型](https://arxiv.org/abs/2004.11062)按实际实现计消息和工作量，
本仓同样不从collective名称套轮数。[MLIR analysis管理](https://mlir.llvm.org/docs/PassManagement/#analysis-management)
要求IR mutation后重新统计，本项不生成或重放旁路schedule。

Non-goals：不新增IR/attribute、改变数值语义、推断SPM合法性、插join/wait或强制通信。
完成条件：唯一CostModel实现上述标量、三个consumer接入、下列主机矩阵和canonical build/no-op通过。
板测进度和后续模型验收只写统一板测计划。

| 覆盖输入 | exact要求 | 直接下游witness |
| --- | --- | --- |
| DDR减少/DTE增加与反向；NE/CT交换 | 同cohort均能排序，DDR和DTE各有胜出条件 | Analysis公式unit与actual candidate/no-card |
| 不同Tile的资源峰值 | 先同Tile相加再取max，不合成不存在的Tile | Analysis-only算术oracle |
| 0/1/15/32 DTE消息，combined/split NCC | 原分项ps结果保持，不重复计payload或sender wait | Analysis与Driver回归 |
| 未校准work/缺metric、溢出及缺/异cohort | 粗估标识、饱和不回绕；profile合同错误保持typed | Analysis公式和controller接入 |
| rank≥3、1024/1025/1031 actual候选 | actual SPM/target先验证，保留同一winner owner | 现有主机IR与source→package/no-card |

Actual capacity rejection默认只对产生该current IR的完整choice有效。只有从actual owner/conflict witness可证明的有限条件
才能作为causal feedback；unknown、unsupported、timeout和compiler error不得改写为capacity rejection。

同一个Temporal choice下，Region和movement候选的汇总状态与已执行leaf的容量反馈分别保留。某个leaf已经得到带actual
conflict demand的SPM容量拒绝时，即使其它alternative尚未穷尽或不支持，controller仍可据此提出更小的Temporal choice；
汇总结果继续保持indeterminate/unsupported，不得把该反馈提升为共同owner不合法或用来剪枝。反馈只携带已执行leaf的typed
failure快照，不保存被销毁IR的operation/value指针；新choice必须重新物化并通过同一actual leaf。

Temporal choice下的Region/layout/movement/execution alternatives逐个惰性物化，分别扣除全局trials。
预算耗尽保留未完成域的typed状态；同一actual checkpoint的后继恢复不能重复扣除已执行的leaf。

| 本项覆盖 | exact要求 | 直接下游witness |
| --- | --- | --- |
| 暂停/恢复、14/42/126、width=1/8 | 已执行序列前缀一致；width只限制保留；最佳actual owner不重建 | Driver/controller与实际source编译 |
| Joint/Independent、Region/replica、不同placement | 不同结构先获得入口；raw domain仍可达；同流量不去重 | rank≥3、1024/1025/1031多Tile与tail |
| capacity、unsupported、compiler error | 冲突相关scope才作因果修正；暂停不成为结构no-good | actual Instr→SPM反馈→新candidate |
| Layout/通信/复用/流水及组合 | 分别物化、verify、fresh memory/target；局部坏但组合好的oracle | 实际owner、访问、completion及标量cost |
| 全workload与模型 | 原PyTorch容差；decode实际KV接续；估时与实卡时间分别报告 | fresh package/no-card与串行board |

## 8. Ownership、analysis 与实现边界

- Compiler driver拥有policy routing、frontier/budget、candidate transaction和唯一winner handoff；不实现leaf rewrite。
- Analysis只读current IR和显式target configuration；mutation后默认失效，不把operation pointer或物化前identity传给下游。
- Transformation通过`PatternRewriter`/`IRMapping`或明确owner API修改candidate。每个transformation只有一个production实现。
- Conversion只读已经完整表达源stage语义的actual ops/types/effects，不补choice或repair。
- Event graph、lifetime、buffer demand和cost是可重算analysis result，不进入IR、candidate key或跨mutation cache。
- 如果下游需要一项无法从current IR重算的信息，先修改源IR表示，不增加side plan。

源码稳定职责为：

- TensorProgram analysis：structured semantics、exact demand和Spatial/Region choice domain；
- current-candidate planning：从live operation/interfaces建立query-local Temporal等search choice；layout备选由query-local exact PBQP产生并在各自actual clone中apply，query均在mutation后失效；
- TensorProgram/TileModule/TileRegion transforms：structural materialization、selected temporal tile-and-fuse apply、online-attention decomposition、
  layout/view/bufferization和movement；
- TileRegion-to-Instr conversion：deterministic target-abstract lowering；
- Instr analysis/transforms：worker/order、completion、lifetime和memory problem derivation；
- actual memory/transport/target leaf：offset、range、resource、ABI和DeviceExecutable acceptance；
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

- post-attention ordinary logical normalization只在policy分叉前运行一次；graph attention在该pass中保持opaque，candidate attention转换及
  后续stage不再调用e-graph；e-graph budget exhaustion保持对应current component不变且不进入candidate key或legality；
- structural materialization对每个FA owner或FD K2 contribution创建all-and-only一个三结果online-attention；selected merge Tile由parent
  TileModule证明，参与 state 由 SSA 证明，不存在 empty shell、规划句柄到 operation 的映射或 `merge ID -> TileId`；
- 第13项只从live current operations建立temporal domain；online-attention的parallel轴由`TilingInterface`处理、K2由
  三个DPS state处理K2。第14项只分解已tiled op；layout入口graph/online attention均为零；
- temporal domain只在exact total single-valued proof下删除派生参数；non-unique、unsupported和indeterminate case保留原自由维度或
  独立producer，Region candidate不因fusion无法证明而消失；
- Spatial与Temporal共用同一static tensor indexing relation builder；single-use dense-offset/unit-reshape、general reshape、all-use direct/view、
  broadcast hoist和互斥window成功case均证明原完整producer及第15项对应完整intermediate allocation/copy为零；overlap/unsupported choice
  保持actual独立buffer并由后续MiniMalloc判断，不转换成SPM估算结论；
- instrumentation on/off产生同一IR、candidate result和package；
- 每个candidate的actual TileModule/TileRegion/Instr owner只物化一次，winner不重建；
- 不存在代表future operation/value/buffer/event/schedule的跨stage状态或为其服务的parity verifier；
- layout/view测试检查actual SSA alias和copy数，movement测试检查actual typed ops/effects；
- schedule/completion测试从current Instr构造并检查位置、participant、token、动态次数和lifetime witness；
- SPM测试检查actual allocation、owner relation、conflict demand和offset，不检查预测footprint。

### 10.3 融合、IR膨胀与Instr汇总

启用compile timing时只从current choice和actual IR输出有界汇总，不参与candidate selection或legality：

- global logical normalization的component、input op、relation query、e-node/e-class、match、iteration、extraction work、wall、RSS及
  reshape/transpose/broadcast/concat消除数；budget exhaustion单独计数且输出graph保持原样；
- physical Tile数、TileRegion数和structured execution instance总数；
- 每TileRegion的structured execution数的minimum/average/maximum和singleton region数；
- 每TileRegion的actual nested operation数的minimum/average/maximum；
- region-local use、cross-region external use和actual DDR/peer movement数；
- current SSA local edge、loop外/loop内producer occurrence、fusion barrier和explicit replica数；
- 每个traversal的自由与exact-derived temporal axis数、main/remainder静态variant数；`r`个ragged tiled axes不超过`2^r`且first peel为零；
- graph attention→online-attention转换数、per-Tile K2 contribution、local temporal K2 block/tail、three-state endpoint、selected merge
  parent及decomposition后的QK/PV/state/merge actual occurrence；不输出预测action inventory；
- function-boundary bufferization产生的copy按必要性证据和memory-space pair分类；冗余DDR→DDR publication copy为0，进入
  movement和Instr conversion的未分类`memref.copy`为0；
- accepted final Wafer Instr总数、per-Tile minimum/average/maximum和per-kind exact count；
- accepted final NE/Vector logical work、各自启用的throughput/service time、instruction-control term和最终makespan。

该汇总不逐region打印日志，不把structured execution数与raw operation/Instr数混为一个指标，也不构造
expected inventory。

`--compile-timing`下的current实现使用固定、bounded的`compile-counter`类别输出该汇总：`structured-egraph`记录一次全局logical
normalization work；`search`记录frontier/current actualization与actual-capacity refinement；`layout`和`movement`汇总所有实际运行的
candidate work；`accepted-physical-ir`与`accepted-instr`只记录controller最终保留的同一actual owner。后两类分别给出TileModule/
TileRegion、每Region nested/dataflow op的min/sum/max，以及final Instr的per-Tile min/sum/max、engine/transport/completion kind、logical work、
movement bytes和accepted high-water。字段集合不随图规模增长；unknown或counter overflow必须显式标记，不能打印为可信零值。
Instrumentation关闭时不创建counter，打开/关闭产生byte-identical package。

### 10.4 End to end

- baseline和search分别从同一current FP16/BF16 source形成policy-complete Instr、actual memory plan、DeviceExecutable和package；
- 两条policy使用独立process、IR owner、ProgramData handoff和output directory，不互调或共享result；
- 两者均实际经过MiniMalloc、DDR、transport、target、strict package readback和no-card；
- timeout、OOM、skip、fallback或未进入actual planner不是通过；
- board-ready与真实设备证据分层，host/package/no-card不得称为board correctness或performance。
