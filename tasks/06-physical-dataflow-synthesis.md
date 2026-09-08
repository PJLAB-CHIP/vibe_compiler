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

### 5.3 Temporal tiling

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

Producer tile是否由consumer tile唯一决定，只能由一次transformation调用内的只读exact tile-relation query判断。该query只读取current
producer/result、current consumer/operand、已经选定的自由tile参数、indexing map与`IndexRelation`/`ExactIndexSet`，不调用会修改IR的
tiling builder，也不把offset、extent、operation或SSA保存到跨stage plan。只有证明为total single-valued relation时才能把producer参数
作为派生量移出domain；存在多个合法取值时继续完整枚举，无法证明或当前接口不支持时保留独立producer traversal和Region candidate。
查询结果区分exact、unsupported、indeterminate和broken contract；unsupported/indeterminate只关闭本次fusion机会，不签发resource结论，
broken contract终止该candidate。

当前exact-derived边界要求producer/consumer位于同一Region和block、producer pure、edge不是DPS destination，并由current indexing map或
composed `IndexRelation`证明实际tile demand。Single-use direct/projected chain、general reshape rectangle/有限pieces以及all-use compatible
direct/view chain均可形成Joint；未捕获use、effect、DPS destination、cross-Region或relation失败保持Independent。Broadcast和window遵循下述
额外门禁。`online_attention`始终是独立root，不作为ordinary producer被复制进finalize或其它consumer traversal。

Direct edge不是完整边界。Spatial exact-demand已经通过`WaferTensorIndexingOpInterface`和`IndexRelation`解释static pure
`tensor.cast`、`extract_slice`、`insert_slice`、`expand_shape`、`collapse_shape`和`pad`；该current-op relation构造必须抽为
Analysis/Linalg中的一个共享只读typed builder，Spatial demand与Temporal fusion调用同一实现。TemporalDomain沿same-Region pure support
chain逐段组合result-to-operand relation，并为每个all-use connected component保留independent与joint两类typed transformation choice。
Independent从全部current candidate建立原始scope；joint只移除由current consumer完整决定的producer。两类choice各自有独立首项和完整lazy
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

Broadcast Joint不要求producer与consumer具有相同rank。对projected-permutation result/operand maps，query建立producer-result维度到consumer
iterator的exact映射；consumer中未参与该operand的active轴是broadcast-only轴。只有全部producer-dependent active轴在selected loop order中
构成prefix时，producer result tile才在该prefix之后、首个broadcast-only loop之前物化，并由内层consumer tiles直接共享。反向order或任一
all-use映射不一致只关闭Joint，不改变Independent raw domain。

Window/conv Joint从selected consumer iteration rectangle和operand affine map计算exact operand rectangle。当前闭包只接受symbol-free、
separable、非负线性表达式，且每个source coordinate的离散image经系数覆盖证明为连续区间。每个active consumer轴必须映射到唯一source
coordinate，并满足相邻tile的物理offset shift不小于该coordinate demand span，证明不同iteration的producer demand互斥后才逐tile融合。
Halo overlap时保持Independent，让完整current producer作为实际共享值；不引入预测halo buffer、rolling cache或隐式重算choice。

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

#### 6.1.1 Layout assignment 与 exact PBQP

Layout合法域直接从current structural TileRegion的SSA value/use、consumer interface、exact `IndexRelation`和可验证encoding构造。
Baseline与search都调用同一个query-local PBQP layout optimizer；它不是search state，也不共享两条policy的candidate owner。
Baseline与search对每个actual attempt都只求解并应用一次确定性`Optimal` assignment；layout不是search axis，不建立layout frontier或
raw layout枚举。PBQP在当前IR上按实际 materialization 的 physical bytes（含 padding）与一次 materialization unit
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
Structural Pareto只淘汰相同Region数量下被支配的plan，不跨Region数量删除diversity；它只决定最多6个initial proposal的
访问顺序，P0和graph-coherent endpoint不可饥饿。当initial proposal
还剩两个slots时，proposal owner只接受controller incumbent已有的RegionPlan作为现有choice，并将至多2个邻域RegionPlan排在剩余深seed之前；不读取或复制incumbent的actual
buffer/layout/movement/lifetime事实。

Search最多actualize 8个structural candidates，并共享42次Temporal actual-attempt credits。每个proposal都是普通`RegionPlan`并独立进入
`current IR → actual transformation → verifier → fresh analysis`。Actual capacity rejection只描述该complete Region/Temporal tuple，不泛化到
其它partition；credit exhaustion报告bounded partial。Winner只由final actual objective决定，Accepted owner直接保留，不按query labels重建。
改变proposal/refinement或关闭它们不得改变raw lazy successor集合。General DAG不声明全局最优，quality由tiny fixed-region-count独立穷举oracle
量化optimality gap并要求已知greedy trap严格改善，再由真实规模cut gain和final actual objective共同约束。

第一个Accepted actual objective作为本次search的no-regression reference。普通Better/Worse/Equivalent仍完全由actual service terms与
storage/residency facts的Pareto比较
决定；两个known objective互相incomparable时，只有二者都相对reference为Better或Equivalent，才以现有StructuralCandidateKey中的RegionPlan
group数选择delivery owner，数量相同再用complete key。该选择仍报告`FeasibleUnranked`，不宣称runtime优劣；任何相对reference存在actual term回退的
candidate都不能凭Region更少覆盖safe incumbent。Unknown objective不进入此规则。

Public search work limit只有`width`和`trials`。`width`是可访问的structural choices总数，`trials`是全局actual
compilation次数；默认分别为8和42。正式CLI使用`--search-width`与`--search-trials`，public C++ API使用
`OptimizationConfig::search(SearchLimits)`。Initial/refinement slots和单candidate Temporal上限是内部调度，不对外暴露。
Search limit不是shape、legality、cost或wall-time policy；none不接受它，缺省参数保持现有结果。Effective limits在
compiler diagnostic和compile counters中记录，相同source、target、policy和limits保持确定性。

Candidate形成`TileExecutable`前必须调用target ABI preparation共用的exact program/DDR function-boundary verifier；argument/result binding不完整的
Spatial/Region candidate在controller admission前返回typed failure，不能先作为Accepted winner保留、再由最终target codegen首次发现错误。

DP、memo、priority、dominance和LNS可以改变choice访问顺序和搜索工作，但不得用推算的IR/buffer/instruction inventory
代替actual result。一个choice只在物化为current IR并通过actual gate后才能成为accepted candidate。

### 7.4 Cost 与feedback

Source-IR-derived lower bound只能用于frontier ordering，必须标明不是actual cost。Candidate comparison使用物化后的TileRegion、Instr、
movement、completion和memory/target数据。推算结果不进入legality、SPM feedback或exact no-good。

最终winner objective从每个Accepted owner的final current Instr和fresh schedule/cost analysis计算，不使用
`aggregateInstructionCount * instruction_tick`作为compute cost。比较合同至少分别保留：

- 每Tile的NE FP16/BF16 logical work及target-profile NE throughput；
- 每Tile的Vector/CT FP16/BF16和F32 logical work及各自throughput；
- instruction issue/control、DTE endpoint bytes、DTE message startup、minimum-hop message demand、wait、DDR、NoC和显式SPM movement的独立work与service term；
- current control flow、effect、token和已物化execution structure决定的有限schedule/makespan。

NE与Vector的service time分别计算；instruction数量只额外计发射/控制开销。一条NE GEMM与一条Vector instruction即使instruction数相同，
也不能因此得到相同compute cost。当前硬件事实证明CT、NE是不同engine/completion domain，并有CT/NE与movement engine overlap的
profile内观测；尚无证据证明NE与CT彼此如何重叠。Current IR明确依赖或completion顺序的work按该顺序累加；没有依赖的NE/CT work
不能擅自按`max`重叠，也不能把强制串行的诊断上界冒充可比较的actual makespan。若候选排序取决于这项unknown，objective保持
incomparable。只有后续硬件文档和matched profile明确证明的并发关系才能增加对应schedule resource组合。

同一次winner比较的所有candidate必须使用同一target profile和同一组enabled terms。某个实际出现的NE/Vector work、所需rate、
schedule multiplicity或算术结果为unknown/unsupported/overflow时，该objective保持typed incomparable，controller只能报告
`FeasibleUnranked`或其它准确coverage；不得退回统一instruction cost，也不得把semantic tie-break伪装成cost winner。Layout PBQP不读取
该objective，也不参与frontier ordering；controller只对layout已经唯一确定的candidate继续枚举其它choice，并以物化后的actual objective比较。

当前实现的DTE endpoint、message startup和minimum-hop terms直接消费final Instr cost中的actual transmit bytes、message count和hop-demand；
它们不能退化为固定instruction数量，也不能使用target-independent guessed route。rate必须来自同一target-profile cohort；未校准时保持
`Unknown`/`Incomparable`，不能用默认零值继续排序。

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

- TensorProgram analysis：structured semantics、exact demand和Spatial/Region choice domain；
- current-candidate planning：从live operation/interfaces建立query-local Temporal等search choice；layout由一次query-local exact PBQP唯一确定并
  立即apply，二者均在mutation后失效；
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
