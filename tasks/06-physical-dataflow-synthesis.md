# Card 内 Physical Dataflow Search 与执行构造

本文是`TensorProgram -> TileModule set -> TileRegion -> Instr -> DeviceExecutable`主线中card-level physical dataflow的
唯一设计owner。当前任务状态和施工顺序只看[progress](progress.md)，算法与覆盖见下方专题。
历史证据见[主机资格记录](archive/physical-dataflow-host-readiness.md)与[搜索组织记录](archive/physical-search-organization.md)。
历史plan和archive只作审计背景，不定义current pipeline。

## 1. 核心规则

Physical-dataflow search可以选择明确的transformation参数，但current IR是下游事实的唯一来源。

可以在物化前保存的是选择：

- structured iteration的spatial partition和Tile placement；
- TileRegion membership和显式replica choice；operation最终位于loop内或loop外不是choice；
- 尚未被exact relation唯一决定的temporal tile vector和dependence-legal loop order；
- attention fixed algorithm下的output/K2 spatial partition、contribution Tile和merge Tile；
- 对当前Tensor assembly实际读取组保留共享或按已选窗口局部物化的显式选择；不包含任意驻留范围或缓存层级；
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

跨Tile shared-DDR的Region DAG并不提供实际执行同步。候选须按13号合同物化并验证current Instr中的
`ddr_publish`/`ddr_acquire`及对应资源关系；搜索物化和主机编译通过本身不代签板端正确性。

## 2. Pipeline Contract

Attention composite迁移按[05号4.5节](05-local-compute-normalization.md#45-composite结构化causal与混合精度attention)
补齐本stage的actual spatial/temporal物化：保留绝对query/key位置与有效KV域，跳过完全不可见块的读取、计算和state update，
边界块交给唯一decomposition。两种切分都保持当前tile中间shape、coupled F32 state及exact coverage。
对实际KV循环，若position与IV有可证明的固定偏移，query/end在该循环内不变，且不可见迭代只会原样传递所有state，
将upper bound收紧为原上界、有效KV终点和causal query终点的交集（扣除key基址）。不改变step、起点或tail；
有额外effect/state更新或无法证明关系时保留精确visibility分支，不根据mask名称猜测可跳区间。
当前arith的index min/max通过共同ValueBounds external model提供与operand的大小关系；driver与wafer-opt注册同一模型。
模板模式分析同时消费实际循环步长及包围分支的整数约束，不因有界上界不是常量而退化为全部偏移模式。
这些改动不改变search预算/计费，不增加DTE专项或shadow plan；全部结构变换仍先物化、verify再分析，
SPM合法性只由下述共同actual leaf决定。实施及逐case实卡门槛见[统一板测计划](archive/board-workload-matrix.md#attention导出展开与实卡验收)。

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
  立即应用temporal tile-and-fuse；online-attention的K2使用三个DPS state的stateful Tiling，之后确定性分解为Linalg/Tensor/SCF，
  在实际Tensor子集的共享选择与局部物化闭合后，再依次完成layout/view/bufferization、movement、execution structure、
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

循环累加状态的layout由其实际更新/消费算子约束，初值作为可转换入口，不因初始化来源的存储格式把整个回边钉死。
SCF region argument、yield及result使用一致的状态layout；转换选择须计入当前静态循环执行次数。
该规则同时适用于分块contraction/reduction和attention状态，scalar状态按实际native tuple选择，不统一硬编码NCX。
入口与最终观察边界的转换、buffer alias及SPM合法性分别由原有阶段验证，不能隐式合并psum/output存储。

### 3.5 访问复用统一分析边界

统一概念为访问复用（`AccessReuse`）；第一版限定为可证明内容不变的读取。
本节定义访问复用边界；关系推导、pipeline接入和覆盖矩阵见[访问复用专题](06-physical-dataflow/access-reuse.md)。

- Upstream IR / input：已选spatial/temporal/layout并完成BoundaryMovement的candidate current Tile IR，显式load/subview/SCF、
  typed source/resource identity、effect/alias及现有buffer owner关系。
- Current stage responsibility：`AccessReuseAnalysis`统一解释同源读取的Tile位置、循环域、精确IndexRelation与内容不变性；
  search先用这些current事实与同cohort成本参数做轻量净收益筛选，只为预期净收益明显的机会生成复用候选，
  `materializeAccessReuse`在独占候选中落实。分析不选择cache或peer owner，
  不记录未来buffer、offset、lifetime或completion。
- Output IR / files：可失效的只读分析结果，以及实际物化的现有allocation/view/copy、Tile load/peer与SCF；
  新buffer全部具备current owner，无缓存专用IR和跨stage旁路协议。
- Downstream consumer：原execution structure、Instr、communication/completion、唯一SPM/DDR及actual cost；
  每次相关IR mutation后分析失效，旧anchor不能按名称或遍历序号恢复。
- User-level driver / named pipeline：现有search的physical movement候选入口；production与显式资格测试调用同一typed变换API，
  不新增负责自动搜索的pass。旧跨Tile输入共享分析迁入同一owner，不与另一套时间复用pass独立决策。
- Explicit non-goals：缓存替换、任意层级、动态/间接访问、多轴滑动、近似窗口、GEMM模板、强制空间划分或Ring、
  新layout/同步算法及预测SPM准入；保持原算术与dtype。
- Completion criteria：保留原peer能力；基础驻留、单轴固定步长滑动、相邻scope最多两级及时间/空间组合进入同一候选链；
  低收益机会不产生额外clone/物化/完整评分或trial；联合只包含通过收益门槛的机会。方案覆盖矩阵逐项有actual下游witness，
  并分别证明通过门槛的GEMM组合可表达、可行、可搜索及实际收益。

缓存范围从选定scope的当前读集合精确推导，不另设自由尺寸搜索。scope选择是物化参数，不是SPM合法性结论。
访问窗口、重复流量及同cohort参数用于性能收益筛选与排序；扣除新增DTE传输/启动及SPM复制，不仅比较单块大小。
筛选是search的启发式取舍，不能用缓存footprint或预测lifetime过滤SPM合法性，也不能猜测join/wait或伪造actual inventory。
实际容量仍只能由完整候选的allocation、layout、alias、effects、completion与lifetime规划结果判断；收益门槛见访问复用专题第5.1节。

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

内部候选检查入口可在一次正常搜索的实际求值边界中止并移交当前结果，用于验证落选方案。
输入是本次调用的只读观察回调；回调返回继续搜索或检查当前结果，正式产品调用不设置回调。
观察前仍执行原变换、verifier、唯一SPM/target及cost；移交的是仍存活的actual owner，不重建、不clone或修改评分。
输出沿原DeviceExecutable→target→package路径发布。未命中或命中非accepted结果均不能发布另一候选替代。
测试入口以本次调用的零起始求值序号选择观察点；该序号只用于诊断定位，不是跨运行IR身份或可重放的语义协议。
该入口不属于设备autotuner，不改变正常搜索的候选顺序、计费或winner。

覆盖要求：真实规模整除/尾块source到package/no-card；观察前的搜索前缀与普通调用一致；
直接移交同一owner；非accepted及预算内未命中不发布；生产CLI拒绝内部选项。
板端性能比较使用同一新输入/reference及原精度合同，记录分块差异，不能把不同分块包装成纯transport单变量实验。

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

## 专题章节

以下章节与本文共同构成06号设计，保留原节号；算法、typed failure和覆盖要求由对应章节完整定义。

| 章节 | 职责 |
| --- | --- |
| [第5节：结构选择](06-physical-dataflow/structural-choice.md) | Spatial/Region、需求关系、Temporal tile/fuse及Tensor物化 |
| [第6节：物理实现](06-physical-dataflow/physical-realization.md) | layout、movement、通信、execution structure与actual leaf |
| [第7节：搜索与成本](06-physical-dataflow/search.md) | none/search、proposal、实际容量反馈、估时、standard/deep预算及验收 |
| [访问复用](06-physical-dataflow/access-reuse.md) | 当前读取关系、Peer/Resident/Sliding/Two-level、收益筛选及覆盖 |

## 8. Ownership、analysis 与实现边界

- Compiler driver拥有policy routing、frontier/budget、candidate transaction和唯一winner handoff；不实现leaf rewrite。
- Analysis只读current IR和显式target configuration；mutation后默认失效，不把operation pointer或物化前identity传给下游。
- Transformation通过`PatternRewriter`/`IRMapping`或明确owner API修改candidate。每个transformation只有一个production实现。
- Conversion只读已经完整表达源stage语义的actual ops/types/effects，不补choice或repair。
- Event graph、lifetime、buffer demand和cost是可重算analysis result，不进入IR、candidate key或跨mutation cache。
- 如果下游需要一项无法从current IR重算的信息，先修改源IR表示，不增加side plan。

源码稳定职责为：

- TensorProgram analysis：structured semantics、exact demand和Spatial/Region choice domain；
- current-candidate planning：从live operation/interfaces建立query-local Temporal等search choice；每个actual layout-input的完整assignment由单次query-local PBQP产生，在placement的actual clone中apply；query只在其immutable owner上有效；
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
  后续stage不再调用e-graph；搜索扩展budget结束仍按05号合同提取已证明等价式，关系/提取证据不足才保持原component；budget状态不进入candidate key或legality；
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
  reshape/transpose/broadcast/concat消除数；budget exhaustion与是否提交有效改写分别计数；
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
