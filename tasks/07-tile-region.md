# Wafer Selected Tile-Dataflow IR、SPM Residency Region 与原子物化

状态：本文定义 selected tile/dataflow IR 的 MLIR-native 边界。Q32/Q46已闭合现有typed materialization、
relation和physical-version机制；Q49负责把production调用域收敛为complete-rank、pre-Instr联合综合，并把
`wafer.tile.region`收敛为显式SPM residency domain。
实现状态只看`tasks/progress.md`；历史 task-dataflow scheduling 证据只作背景，不是当前合同。

source structured op 的数学语义始终存在于当前 operation、region、SSA、type、attribute 和标准
MLIR interfaces 中。tasks/06与本文不是“先选出全局plan，再统一import”的两个管线阶段：fuse/tile/
physical-dataflow proposal先用current IR、typed legality、relation和lower bound进入query-local structural frontier，由DP/Pareto
剪枝；只有统一actual-clone budget准入的有界代表才调用本文rewrite library，在isolated complete-rank
candidate clone中用`PatternRewriter`和`DialectConversion`同时物化implementation、tile、physical encoding、
residency、movement、share/recompute、hoist、numeric rewrite和traversal order。物化后的决定只存在actual IR中，
transient proposal随即销毁；actual失败时按稳定顺序从未物化frontier补位，下游只读取current IR。

`wafer.tile.region`物化一个显式的 **SPM residency domain**：region body中的一个或多个structured traversal共享
同一组可同时驻留、可由SSA直接连接并由同一liveness关系解释的tile、temporary、accumulator和staging。它不要求
global tensor整体进入3 MiB SPM，只要求当前traversal位置的live working set可由下游planner合法放置。region组织
Wafer-tagged SPM memref、view、compute、SPM内movement、peer/collective movement、显式DDR streaming或selective
spill/reload、event和structured control flow。数据input/result是variadic DDR value/view，fan-in、fan-out和边数没有
IR上限；typed scalar/control/event可按各自verifier穿过边界，但不得隐式携带SPM alias。region op不隐式执行搬运，
SPM value/root/alias不得成为region I/O。

一个完整static rank entry可以包含一个或多个non-nested `wafer.tile.region`。region partition、每个region的traversal/
tile shape、跨region materialization、region内resident/local movement、selective spill/recompute和communication共同进入
candidate搜索；region数量既不是目标，也不能代替DDR、GS、NCC、completion或tile-utilization成本。两个traversal之间若
有SPM-resident SSA edge，就属于同一residency domain；若选择切成不同region，所有跨界数据必须通过显式DDR
store、可信completion和matching load表达，不能把SPM root、alias或仍访问该root的pending work传给下一region。

selective spill只结束目标SPM root，matching reload建立新root；只要其它root仍跨过该点resident，就不形成完整region cut。
相反，当候选为了独立retile、缩短共同lifetime或其它真实成本而结束切口上的全部SPM residency并显式materialize跨界值时，
可以形成新的sibling region。region boundary本身不是device-wide completion、launch或join边界；它只要求所有仍访问被释放
SPM roots的work已经按typed effect/event证明完成。未建模的opaque SPM clobber仍必须fail closed，不能靠新建region伪装合法。

本文依赖：

- `tasks/05-local-compute-normalization.md`：rank-local structured tensor normal form。
- `tasks/06-physical-dataflow-synthesis.md`：联合选择、clone 生命周期、精确排序和 all-rank atomic commit。
- `tasks/08-physical-realization.md`：physical encoding、view、footprint 和 movement legality。
- `tasks/09-spm-memory-planning.md`、`tasks/12-ddr-memory-planning.md`：从actual region/control-flow/liveness派生的
  allocation domain、whole-variant lifetime、capacity和offset gate。
- `tasks/10-compute-movement.md`：source implementation interface 与 selected compute/movement contract。
- `tasks/11-instruction-ir.md`：complete-rank instruction IR 与 target legality。

## 1. 职责和非目标

本层负责：

- 在 isolated complete-rank clone 上应用当前structural frontier中通过统一actual-clone budget准入的action所选定的
  tiling、fusion、producer propagation 或等价
  indexing rewrite；该action与本次mutation同寿命，不先形成可跨pass保留的完整plan。
- 物化share-vs-recompute与static loop-invariant hoist：share保持同一producer/physical version的多use；recompute只克隆
  pure/speculatable producer并形成consumer-local SSA；hoist把真实op移到loop外并让body捕获dominant SSA value。
- 把floating numeric-validated与integer exact/modular-proof-backed reassociation、显式reduction tree、split和已支持
  algebraic variant物化为真实op DAG、SCF和loop-carried state；不保存numeric-choice attr。
- 按 selected tile domain生成 all-and-only traversal、static tail和合法 reduction sequence。
- 对production capability分类与`TilingInterface`共同证明可切的terminal logical collective，从result tile反向物化
  operand/out tile并融合producer；当前只启用单输入、单输出、shape-preserving `all_reduce`，其每个dynamic loop
  instance只处理当前tile，完整result由显式insert/writeback拼接。其它collective在各自gate闭合前仍是full traversal。
- NoC-resident扩展同时允许从已resident或peer到达的operand tile通过`TilingInterface`正向物化consumer tile，并只通过
  `PartialReductionOpInterface`物化partial/merge；input/parameter、intermediate、partial和output角色均由current
  boundary与SSA relation派生，不进入固定枚举或operator matcher。
- 通过`WaferTargetImplementationOpInterface::materializeSelectedImplementation`创建typed
  `wafer.tile.*` compute。
- 物化 Wafer-tagged memref、standard/typed view、resident SSA edge、显式 movement、spill/reload、
  temporary、accumulator 和 staging。
- 按selected region partition为完整rank entry物化一个或多个non-nested SPM residency regions；每个region只包含
  其共同驻留的selected tile schedules和physical dataflow，跨region值用显式materialization连接。partition决定
  物化进actual IR结构，不另存region-plan attr或side table。
- 用 event SSA、typed wait/dependency、MemoryEffects和structured control flow表达issue、event completion、reuse及仍待下游
  完成的observable obligations；本层不物化compiler-derived participant join或terminal drain。
- 让 conversion legality、op/interface verifier 和 fresh analyses 能从 current clone 独立重建全部事实。
- 任一 rewrite、conversion、coverage 或 verifier 失败时丢弃整个 clone，不污染 source 或其它 clone。

本层不负责：

- 不生成或排序 implementation、tile、encoding、residency、movement 或 task-order 选择。
- 不因 materialization/lowering 失败改选另一实现，不插入临时 fallback，不切分新的搜索分支。
- 不保存选择历史、评分、失败轨迹、影子调度图或其它 IR 外长期语义。
- 不分配 SPM/DDR physical offset；本层只提供下游从region、control flow、SSA和effects重算lifetime/allocation domain
  所需的事实。
- 不让单个traversal、component或旧task module独立执行Tile→Instr、SPM/DDR planning或completion后再拼接rank。
- 不把task、source scope、op名、旧loop边界、layout名、GS或collective机械映射成region boundary；region cut只能来自
  联合搜索选中的residency/materialization方案并由actual IR证明。region数量不单独计奖惩，真实DDR、GS、completion、
  tile utilization和materialization工作分别计价。
- 不因SPM bank phase/conflict选择region partition或DDR spill；09只允许allocator在hard-valid placement中把可重算
  bank phase作为soft preference。当前实现状态见Q49。
- 不 lower raw packet、CRT、LLVM、runtime handle 或 package 字段。
- 不通过 op/value/parameter 名、固定 shape、参数顺序或 workload topology 恢复语义。
- 不把单个 region、representative tile、单个 rank 或局部 FileCheck 当成完整完成证据。

## 2. Pipeline Contract

    Pipeline position:
    - Upstream artifact / IR:
      verifier-legal complete-rank structured tensor IR，以及同一次 transformation 内通过便宜预筛、将立即物化的
      WaferTargetImplementationOpInterface candidate、每个traversal的tile domain/loop order、physical operand/result
      encoding、region partition、residency、movement、
      share/recompute、hoist、current numeric variant和execution order。所有选择都引用current op/value并可在mutation前重新验证；
      它们不跨pass发布。当前输入不得含无法由typed effect解释的opaque SPM clobber；跨rank NoC-resident candidate还接收共同verified post-SPMD snapshot、typed
      global/local rank slice和同一transaction中的完整rank clone tuple。
    - Current stage responsibility:
      clone完整rank module；用PatternRewriter应用selected structured rewrites、producer clone/共享、loop hoist和显式numeric DAG，
      并更新真实use-def；
      用source implementation hook、op builders和DialectConversion创建表达SPM residency domain的typed region、view、
      compute、movement、event和SSA relation；按selected partition在每个complete static rank entry中创建一个或多个
      non-nested regions，并在各region内部生成complete traversals及跨region显式materialization；nested region、SPM data
      跨界或没有真实materialization/residency含义的结构边界都拒绝；每次mutation后丢弃旧
      IndexRelation、alias、effect、liveness和resource observations并从current clone重算；
      最后运行conversion legality和tile/dataflow verifier。
    - Output artifact / IR:
      transaction-local、verifier-legal、覆盖完整rank traversal的selected tile/dataflow IR；整个static rank entry
      包含一个或多个non-nested `wafer.tile.region`作为candidate-selected SPM residency domains。region所有data I/O均为DDR，
      SPM value/root/alias不能跨界；每个boundary前只需完成仍访问被释放SPM roots的work，entry terminal继续负责全部
      observable pending work。若物化失败，在无任何published mutation的情况下返回failure。
      成功IR只含typed operation/region/type/
      attribute、memref/view、compute、movement、event和SSA；不依赖任何外部解释对象。
    - Downstream consumer:
      target-abstract legality，并按executable-finalization typed collective/peer algorithm参数逐点执行complete-rank instruction lowering；
      materialized canonical/unplaced Instr随后派生typed worker/fixed-slot/ready-order siblings，进入fresh completion
      reconstruction、liveness-derived SPM allocation、whole-variant DDR、post-memory communication/transport/resource、ABI gates以及
      all-rank atomic commit。completion按真实root reuse、observer、region boundary和entry terminal分别验证；任一内部
      traversal/component不得脱离complete candidate单独提交这些不可逆stage。
    - User-level driver / named pipeline:
      wafer-compile source-to-bundle production pipeline。wafer-opt只可对同一op/interface/
      conversion做局部parser、verifier和rewrite测试，不形成第二条compile pipeline。
    - Explicit non-goals:
      不在本层重新搜索或重排，不创建独立执行计划，不发布tile/dataflow中间artifact，
      不让runtime选择physical realization，不声明板端性能或timing。
    - Completion gate:
      contraction、pointwise、ordered reduction、view、broadcast/slice、fanout/fanin、多root、
      structured control flow和当前已启用的terminal all-reduce tiled traversal均有真实source正负例；每个成功case发生可观察的
      MLIR mutation并覆盖完整traversal；失败保持source不变；输出被instruction、SPM/DDR、
      completion、transport和ABI gate直接消费；同一source覆盖single-region fused-small-tile、multi-region
      separated-large-tile和region内selective-spill actual forms，并验证region data I/O仅为DDR、SPM root不跨界、
      nested/artificial region拒绝、boundary不自动生成join；rank-count 1/16与冻结7B纵向重放通过。

## 3. IR 生命周期与唯一事实源

    rank-local structured tensor IR
      -> isolated complete-rank clone
      -> selected PatternRewriter mutations
      -> selected structured-to-tile DialectConversion
      -> verifier-legal tile/dataflow IR
      -> per-parameter tile-to-instruction DialectConversion
      -> worker/fixed-slot/ready-order siblings
      -> erase and fresh-rebuild dependency-driven completion
      -> derive fixed SPM allocation problems from complete-rank roots / lifetimes / coexistence
      -> validate all-and-only root coverage and fixed-capacity placement
      -> whole-variant DDR then post-memory communication/transport/resource/ABI gates
      -> atomic commit of all ranks

物化后的长期事实只有：

- structured loop、tile offsets/sizes、block arguments、yield和SSA use-def；
- shared producer/version的普通multi-use、recomputed producer的独立SSA clone、loop外hoisted op与body capture，以及显式
  combiner tree/state tuple；不存在share/recompute/hoist/tree选择attr；
- selected `wafer.tile.*` op form及其typed implementation/numeric fields；
- `memref<..., #wafer.memory<space, encoding>>`、allocation root和typed view；
- explicit compute、movement、temporary、accumulator、staging和spill/reload；
- MemoryEffectOpInterface、必要的MLIR custom SideEffects::Resource、async token、typed wait/dependency和可重建的
  observable/pending-effect obligations；terminal drain尚未物化；
- 下游规划接受后写入的physical offsets和transport/ABI-owned fields。

如果某个selected decision不能从这些对象解释、验证或lower，必须先扩op、type、attribute或interface。
不能把它留在C++对象中让后续stage按位置、名字或历史恢复。任何rewrite改变operation、region、SSA、
type、indexing map、view或effect后，基于旧IR得到的analysis全部失效。

## 4. `wafer.tile.region` 合同

概念形式：

    wafer.tile.region (%ddr_inputs..., %typed_control...) -> (%ddr_outputs..., %typed_control_results...) {
      ^bb0(%ddr_args..., %control_args...):
        ... jointly resident traversals, views, compute, movement and events ...
        wafer.tile.yield ...
    }

region必须表达：

- 一个candidate-selected SPM residency domain；其中global tensor按tile遍历，只有current live working set需要同时驻留；
- variadic region argument/result与enclosing rank DDR value/view的SSA relation；DDR root既可来自function external，
  也可来自compiler-managed materialization，边数不是硬件端口、descriptor或DMA数量上限；
- region内部一个或多个structured traversal domains、各自独立的tile coordinate、tile shape和loop order；
- region-owned SPM allocation roots的logical shape/dtype、memory space和selected physical encoding；
- DDR/constant input到region-owned SPM version的显式load，以及SPM output到DDR view的显式store；
- intermediate producer-consumer edge的resident SPM SSA relation、显式SPM-to-SPM movement、recompute，或显式DDR
  store/completion/reload；selective spill只结束目标root，不结束其它resident root；
- compute/movement dependency、内部reuse及释放region-owned root所需的typed wait/join和pending obligation；region边界
  不自动完成与这些root无关的device work，entry terminal另行验证observable completion；
- every output tile all-and-only一次的traversal relation。

region不得表达：

- fusion identity、候选编号、评分、搜索终止原因或调试统计；
- raw packet field、runtime allocation handle或package locator；
- case-specific模型角色或依赖文件名的operand role；
- SPM value/root/alias作为region operand/result、隐式capture或跨region control state；region所有data I/O必须是DDR；
- 仍会访问region-owned SPM root的async device work跨region传播；与SPM无关的typed event/control是否可穿过边界由其
  自身interface和下游completion verifier决定；
- 隐式DDR round-trip、隐式barrier、私有physical SPM arena，或按task/loop/tile shape建立的执行容器。

region不拥有私有physical arena，也不声明同一或不同region的roots具有共同lifetime。每个root的出生、access、completion
和结束从完整rank current IR的SSA/effect/control flow重算；planner再按真实coexistence/conflict关系形成一个或多个fixed
allocation problems。已证明不重叠的regions/roots可以复用完整3 MiB地址范围，可能重叠的regions必须联合满足容量。
每个root必须被all-and-only一个accepted placement覆盖；solver调用次数只作实现与预算诊断，不是IR语义。allocator不得把
placement结果反写成新的partition选择。

### 4.1 Residency Partition 与 Boundary Contract

每个complete static rank entry包含一个或多个non-nested `wafer.tile.region`。tasks/06把partition与tile shape/loop order、
layout/version、resident/spill/recompute、buffering、movement和communication联合搜索；本文只把selected partition物化并验证。

每条producer-consumer connection选择以下一种或一组真实physical action：

1. **same-region resident/local**：兼容的SPM SSA、view或local movement；两侧可以使用耦合traversal，也可以在同一region内
   使用独立loop nests，只要root lifetime与完整working set合法；
2. **recompute**：在consumer traversal中重新物化pure/speculatable producer；
3. **selective spill/streaming**：显式DDR store、可信completion和matching load只结束目标root；其它root继续resident时
   仍属于同一region；
4. **region cut**：切口上的全部跨界data都显式materialize到DDR，前一region不再有root/alias或仍访问这些root的work
   跨界，后一region建立新的SPM roots并可独立选择tile/traversal/layout；若两个regions在control flow上可能并发，
   下游仍按真实liveness联合规划其SPM地址，不能因结构分开就各自占满3 MiB；
5. **independent roots**：按实际control flow与lifetime进入所属region的allocation problem，不因图上无edge就自动切region。

partition本身不是收益。single-region small-tile、multi-region large-tile、same-region separated traversal、selective spill和
recompute必须按actual DDR、GS、NCC、completion、compute、tile utilization、SPM demand及critical path比较。capacity失败可由
decision owner从无offset parent生成retile、repartition、spill或recompute sibling；allocator不能自行merge/split region。

region boundary必须满足：

- regions non-nested，并按current structured control flow覆盖all-and-only selected traversals；
- data argument/result全部是DDR value/view；跨界值的store/completion/load在actual IR中显式存在，SPM root/alias不能跨界；
- 所有仍访问前一region SPM roots的work在root释放前已有typed completion证明；边界不自动插入join，也不要求drain与这些
  roots无关的participant或device work；
- task、source scope、普通loop、layout conversion、collective或单个spill都不能单独充当cut witness；
- 没有materialization、residency终止或external ownership含义的空壳边界必须由canonicalization移除或由verifier拒绝；
- 未知或untyped opaque clobber直接结构化拒绝，不能通过插入region使其“合法”。

## 5. Buffer、View 与 Physical Version

tile/dataflow中的physical buffer使用MLIR `memref`：

    memref<64x256xf16, #wafer.memory<spm, tensor>>
    memref<64x256xf16, #wafer.memory<spm, cx>>
    memref<64x256xf16, #wafer.memory<ddr, tensor>>

memref shape和element type表达logical shape/dtype；Wafer memory attribute表达address space和selected
physical encoding。block、tail、padding、physical footprint和logical index到physical offset的关系由
tasks/08统一calculator从type和target facts推导，不在多个attrs中重复保存。

一个logical value可以因fanout拥有多个physical versions。每个version必须由显式allocation root、
view或compute/movement result定义，并通过SSA连接到consumer。禁止从logical value到多个buffer建立
旁路绑定。

view遵循以下规则：

- physical-isomorphic relation优先使用standard memref view/subview/reinterpret_cast或typed Wafer view。
- view的rank、offset、size、stride和alias relation必须能从current operands/types验证。
- reshape只有在logical element order和physical alias均保持时是view，否则物化真实movement。
- slice、broadcast、permutation和concat不能靠type change丢失relation；不能作为view时必须显式移动。
- `memref.alloc`只表达allocation identity和lifetime，不表达physical offset。

padding不属于logical domain。valid-lane安全只能由current IR中的typed execution mode、分段
offset/count、mask、physical-footprint fill以及producer/movement effects证明。只写logical segment而未先初始化
padding时，padding保持unknown；不能把analysis结论复制成opaque attr绕过证明。

generic `memref.load/store/copy`不得成为Wafer-tagged SPM compute/movement的语义逃逸。对target可观察的
movement必须使用本文和tasks/10定义的typed op。

## 6. Selected Compute、Movement 与 Event

### 6.1 Compute

每个selected compute op：

- 具有typed input/output/attrs，并由concrete op class、ODS verifier、适用的DPS/Tiling/
  MemoryEffect interfaces和conversion legality解释；
- 携带lowering必须区分的implementation kind和parameters；
- 由op kind、region、typed fields和SSA relation完整表达numeric semantics；
- 显式携带temporary、accumulator或loop-carried state；
- 实现`MemoryEffectOpInterface`和必要的MLIR custom `SideEffects::Resource` effects；
- 能在不回看source op或选择过程的情况下验证并lower。

contraction、pointwise/convert和reduce的具体合同由tasks/10拥有。本文只负责把已选合同落入IR，不为
每种dtype、shape或workload创建新op。

### 6.2 Movement

每个selected movement op通过concrete typed op、operands/results、logical relation、physical direction、
MemoryEffect和async token完整解释当前movement；typed conversion pattern直接分派op，lowering时不重新选择其它形态。

每条physical edge在IR中只能采用一种显式形态：

| 形态 | IR合同 |
| --- | --- |
| resident | producer与consumer共享同一memref SSA value，无中间movement |
| physical-isomorphic view | standard/typed view，alias与logical relation可验证 |
| compute-consumed relation | relation由selected compute op的typed operand contract表达，无隐藏movement |
| boundary load/store | DDR view（function external、跨region materialization或region内selective spill/streaming）与SPM view之间的destination-style identity-coordinate movement；region data I/O只能是DDR，看到region边界本身不得合成movement |
| peer movement | `wafer.tile.peer_send`读取source SPM range，`wafer.tile.peer_recv`写入destination SPM range；logical peer、fixed bytes和communication identity显式，physical endpoint/route/FSM留给instruction acceptance |
| local movement | `wafer.tile.materialize_layout`或其它typed movement产生新physical version |
| staged movement | temp allocation、每段movement和completion全部显式 |
| spill/reload | DDR store、可信completion和matching load全部显式；store结束被spill value的SPM root，reload建立新root。其它root仍resident时它是region内selective spill；切口全部roots结束时可作为selected region cut的一部分 |
| constant load/fill | ConstantLike source或typed immutable resource relation与destination encoding显式 |

boundary movement的概念形式：

    wafer.tile.load  %ddr_view into %spm_view
    wafer.tile.store %spm_view into %ddr_view

它们是destination-style op，无隐式allocation和result；logical coordinate relation固定为identity。

Q32.R后`StorageLoadOp`已满足该合同：materializer先创建SPM allocation/view，再发explicit
source/destination load；所有builder、conversion和tests均不再保留旧result入口。tile-to-instruction lowering对
已有destination发射RDMA并删除load，allocation identity继续由memref SSA拥有。
slice、permutation、reshape或concat先成为可验证view，不能化为同shape identity pieces时使用显式local或
staged movement。op不携带relation副本、descriptor list或lowering-time选择字段。

peer movement不是logical collective wrapper，也不隐含allocation或completion。send/recv两端的SPM root/view必须显式；
tile-to-instruction conversion创建matching Direct-DTE issue token和exact wait，whole-variant acceptance再验证反向peer、
message、bytes、range与binding。第一版fanout使用多个explicit send或receive-then-forward，reduction使用explicit recv、
local compute和forward；没有typed target capability时不假设router multicast或in-network reduction。

### 6.3 Event 与 Effects

具有typed async-event语义的issue必须产生token并由matching typed wait消费；普通NCC-capable Tile op通过SSA、
MemoryEffects、range和structured order留下pending obligation，不在本层借用local fence或`NCCJoin`收口。依赖关系固定为：

    issue
      -> typed event wait（仅对显式event domain）/ pending effect obligation
      -> all consumers
      -> next writer or allocation reuse

block order、loop iteration、traversal结束、spill点和旧task顺序都不自动证明completion。source/destination、temporary和
staging的lifetime必须保守覆盖到显式event wait或下游重建的participant completion。post-worker completion owner从
完整entry的actual effects/ranges重建completion：普通loop/traversal和region boundary都不得自动生成`NCCJoin`；只有root
释放或复用、跨worker/engine consumer、真实protocol/observer和entry terminal需要相应completion。region exit只验证没有仍访问
本region SPM roots的pending work，最终all-and-only external write、communication和其它observable effects在entry terminal闭合。

## 7. 原子 Materialization Algorithm

物化是确定性transformation，不是第二个optimizer：

1. **建立clone**：clone完整rank module，确认所有selected source op/value仍属于current IR。
2. **重验选择输入**：重新检查source op interface、type/indexing、tile domain、physical operands/results和
   target facts；失配在mutation前失败。
3. **应用structured rewrite**：用`PatternRewriter`执行已选tiling、fusion、producer propagation或等价
   relation rewrite；pattern只能按MLIR rewrite contract更新真实use-def。
4. **物化complete traversal**：生成compact `scf.for`、static tail、branch和合法ordered reduction step，
   并验证每个logical output all-and-only一次。
5. **物化residency partition与physical SSA graph**：按selected action对完整static rank entry创建一个或多个non-nested
   regions；在各region内部物化tile schedules、traversal domains/tile shapes、layout/version、residency、recompute、movement
   和selective spill/streaming，并创建region-owned allocation roots、views、resident edges、temporary、accumulator和staging。
   跨region值显式store/completion/load。partition的唯一长期表示就是actual region/SSA/movement结构，不生成shadow plan；
   task/source scope或单个spill不得被机械恢复为region boundary。
6. **物化selected compute**：调用`WaferTargetImplementationOpInterface::materializeSelectedImplementation`，创建typed
   `wafer.tile.*` compute；hook失败时丢弃clone，不换另一implementation。
7. **物化movement和events**：在完整rank clone中创建boundary/local/staged movement、spill/reload、tokens和typed
   dependency；region/task return不物化terminal drain。所有新value立即接入SSA。
8. **运行一次complete-rank structured-to-tile conversion**：用`ConversionTarget`、`TypeConverter`和rewrite patterns消除
   本层声明illegal的source forms；成功后不能残留需要下游猜测的op。
9. **fresh重算与局部canonicalization**：每次mutation后丢弃旧`IndexRelation`、alias、effect、liveness、
   completion和resource结果。canonicalization只能删除语义、storage和effect均等价的no-op。
10. **验证clone**：运行op/interface verifier、complete traversal、view/alias、effect/completion和conversion
    legality。任一失败丢弃整个clone。

成功结果仍是transaction-local clone。只有后续instruction、SPM/DDR、all-rank communication、transport和
ABI全部通过，coordinator才能提交包含全部logical ranks的结果。materializer永远不直接修改accepted module、
`ExecutableBundle`或package。

## 8. Structured Semantic Coverage

扩展性按structured semantics组织：

| semantic family | 从current MLIR读取的事实 | 物化结果 |
| --- | --- | --- |
| contraction | iterator types、indexing maps、DPS init、combiner、type和native numeric semantics/permissions | typed GEMM/batch/accumulator chain |
| pointwise/relation/select/convert | elementwise iterators、scalar region、dtype和valid domain | typed compute或明确composite |
| reduction | reduction iterators、init、combiner、axis/result mapping和typed numeric contract | composite或native op；支持的浮点类型默认允许现有数值合同下的重排，integer保持exact/modular gate |
| share-vs-recompute | SSA use-def、exact dependent region、effect/speculation和cost choice | shared multi-use或consumer-local producer SSA clone |
| loop-invariant hoist | LoopLike、dominance、invariant operands、effect/completion | loop外真实op和body捕获的dominant SSA value |
| numeric reassociation/tree/distribution | current integer或floating scalar op family、integer overflow/wrap语义和combiner/dataflow | 显式SSA combiner tree或等价rewritten op DAG；float不要求额外permission，integer no-wrap保持barrier，无隐藏order attr |
| view/reshape/permutation | type、view/subset semantics、index relation和alias proof | metadata view或explicit movement |
| broadcast/slice/concat | indexing relation、static domain和piece coverage | view、movement或structured failure |
| constant tensor | ConstantLike value、logical slice和selected destination encoding | typed fill/load |
| structured control flow | block arguments、yield、loop-carried values和effects | 保留`scf`并显式传递memref/event |
| communication | rank-local operands、typed peer/group facts、bytes和completion | explicit movement/event body；无未展开占位 |

通用测试至少覆盖chain、diamond、fanout/fanin、shared-input contraction、multi-root、residual、collective、
多个dtype、整tile、非整除tail、f16/bf16无标注正例以及integer modular/no-wrap正负例。新增source op优先通过现有Linalg、
DPS、Tiling、ViewLike和effect interfaces进入这些family；只有新数学语义不能稳定表达时才扩IR。

每个positive必须从真实source进入production materializer并发生非零IR mutation。negative至少覆盖wrong
type/rank/index relation、dynamic unsupported case、rewrite guard failure、conversion illegal op、missing completion、
all-clones-fail和late-rank failure。手写`wafer.tile.*` fixture只补parser/verifier覆盖。

## 9. Control Flow 与 Complete Traversal

`scf.if`、`scf.for`和必要CFG可以包含tile/dataflow。loop-carried tensor物化为显式physical memref
version；ready/free token和buffer slot一起loop-carried。没有可验证iteration-to-slot、parity和completion
relation时，dynamic ping-pong fail closed。

complete traversal可以用compact structured loop表达，不要求静态展开每个tile，但必须证明：

- iteration domain与source output domain exact对应；
- result-driven pull或operand-driven push所覆盖的consumer iteration、boundary tile和peer segment all-and-only；
- interior和tail不重叠且union完整；
- branch/yield和multi-result的每条路径类型及effect闭合；
- reduction order符合source numeric contract；
- fanout的每个consumer读取同一版本或显式派生版本；
- 任一reuse发生在所有相关completion之后。

representative tile、shape-only dump或subview数量不能替代coverage proof。

## 10. Verifier、Failure 与 Atomicity

tile/dataflow verifier至少检查：

- region arguments/results、terminator、enclosing traversal和SSA dominance一致；
- selected compute/movement的operand/result、rank、shape、dtype和typed parameters合法；
- 每个physical version的memory space、encoding、allocation root、view和valid domain可重算；
- metadata view保持physical storage同构，不能用reshape逃避真实movement；
- 每条region内intermediate relation是共享SPM version、显式lowerable SPM movement、recompute或完整DDR
  store/completion/reload之一；internal DDR只结束对应root，不能截断其它live root；
- 完整static rank entry包含一个或多个non-nested regions；每个region覆盖非空selected traversal，nested region、
  materialization-free artificial boundary以及无法由typed effect解释的SPM clobber直接拒绝；
- region data argument/result全部是DDR value/view；不得传递SPM memref/root/alias，也不得让仍关联这些
  roots的pending event/control跨界；与SPM无关的typed event/control只按自身interface与completion合同验证；
- standard MemoryEffect直接关联实际SSA value或custom resource，bytes/footprint可从typed IR重算；
- async producer在completion前不能被读取、覆盖或复用；
- 每个region exit没有仍访问region-owned SPM roots的pending work；普通traversal、loop、spill点或region结构本身不产生join，
  entry terminal保留并闭合all-and-only observable completion所需的control-flow、event、range和effect obligations；
- traversal、tail和reduction顺序符合source contract；
- 输出不依赖任何IR外语义对象，不含opaque implementation payload或模型名matcher。

本层成功后仍必须重放：

- tile-to-instruction `DialectConversion`和instruction legality；
- descriptor cover、checked arithmetic和address range；
- 完整rank的SPM roots/lifetime/coexistence、fixed-capacity placement与all-root coverage；
- whole-variant DDR lifetime/capacity/offset；
- all-rank communication、completion、transport和ABI；
- target emission、readback和atomic bundle publication。

失败规则：

- pattern返回failure前不得修改IR；若后续步骤失败，整个isolated clone直接丢弃。
- unsupported source/implementation、无法表示的relation或illegal conversion不产生partial tile IR。
- 单个clone失败只淘汰该clone；所有clone失败才使该rank失败。
- 任一rank失败或complete variant的later gate失败，不发布partial rank/module/artifact/package。
- 失败不得触发本层临时改变implementation、encoding、residency、movement或execution order。

## 11. Q32.V Typed Target Extensions 与其它 Later 能力

Q32.V已排期闭合并由同一candidate owner消费：

- 非identity、strided或多piece的mapped direct boundary movement；
- physical-footprint fill及其valid/padding/bitpacked domain；
- baseline以外的typed contraction operand orientation和对应target command form。

relation-guided Cx/NCx physical-version absorption属于physical-version assignment而不是新target capability：Q32现有
concrete verifier已接受GEMM/batched GEMM和native reduce的Cx/NCx形态；Q46在相同verifier边界补齐hardware-supported、
physical-traversal-compatible CT relation/select/logic/convert/bitpacked。若08的exact physical-map/valid-lane proof允许，前置
`materialize_layout`/GS movement在本clone中消失；packing identity仍只存在于encoding，不增加vector-width或packing
side attr。

下列能力仍有独立前置，不能被当前机制存在误报为已支持：

- immutable prepacked resource publication；
- dynamic shape、复杂mask和target-specific composite。通用producer-consumer tile composition属于06当前合同；只有
  需要新增target-specific composite instruction、复杂dynamic mask或尚无typed numeric semantics的实现才是later；
- 需要新runtime/ABI/SystemC consumer的movement或completion形态。
- Q39已经闭合NoC-resident result/operand/partial traversal、peer movement和Direct-DTE mechanics；Q49把其decision owner并入06的
  whole-rank frontier，本文只物化selected peer/resident Tile IR，13继续拥有typed lowering与all-rank acceptance。pre-Q49 late
  NoC tuple path只作历史资格证据，不再是终态独立pipeline。
- online/streamed reduction只有在typed running state、combine公式、numeric policy、tail与lowering闭合后才进入同一
  reduction candidate domain；未闭合时保留native/partial baseline。没有typed fused semantics的non-GEMM FMA contraction及
  尚未闭合的其它algebraic contraction保持unsupported；

只有target instruction、ABI和执行consumer具备typed合同后，才能启用其中一项。每项扩展必须同批增加
source interface candidate、selected op fields、PatternRewriter/DialectConversion materialization、verifier、
instruction lowering、effects/completion以及真实source正负测试。缺少任一纵向时保持unsupported；Q32.V三项是明确
checkpoint，不得因删除provider协议而消失；其它later能力也不得被Q32/Q32.V completion假装支持。

## 12. 通用案例

以下只展示关系，不固定shape或workload。source概念上是一个contraction及其pointwise consumer：

    %mm = linalg.matmul ins(%a, %b) outs(%init)
    %act = linalg.generic ... ins(%mm) outs(%out)

planner在transformation内选择tile、target implementation、Cx encoding和resident edge。materializer在clone中
概念上生成：

    wafer.tile.region (...) {
      scf.for %m = ... {
        scf.for %n = ... {
          %a_spm = memref.alloc()
          %b_spm = memref.alloc()
          wafer.tile.load %a_view into %a_spm
          wafer.tile.load %b_view into %b_spm

          %mm_spm = wafer.tile.gemm %a_spm, %b_spm
          %act_spm = wafer.tile.elementwise %mm_spm

          wafer.tile.store %act_spm into %out_view
          ... explicit async wait/effect dependency when required ...
        }
      }
      wafer.tile.yield
    }

真实op必须携带完整types、typed implementation/numeric fields、effects和tokens；上面省略这些字段只为讲解。
resident edge由`%mm_spm`的SSA use-def直接表达，没有中间spill。若layout、capacity、completion或instruction
gate拒绝该clone，clone整体丢弃；materializer不就地换实现。

示例中的shape、tile、Cx和具体compute kind都是选择结果，不是协议。相同合同适用于一般structured graph。
同一source graph至少要能形成三类可比较的actual candidate：一个region内融合producer/consumer并使用较小tile；
把producer/consumer物化成两个regions，以显式DDR store/completion/load换取两侧独立的较大tile/traversal；或仍在一个
region内只spill某个root、让其它root继续驻留。也要保留“独立loop nests但resident edge仍在同一region”的合法形态，
证明traversal coupling、region partition和storage action是相关但不等价的选择。最终region数量只描述winner的residency
结构，不能作为收益依据；offset reuse只由完整rank的真实lifetime/coexistence决定。

## 13. Completion Gate

本文边界完成必须同时满足：

1. production materialization只通过source op interface、`PatternRewriter`、builders和
   `DialectConversion`修改isolated clone；没有op-name/shape/workload matcher。
2. success IR只用typed region/view/compute/movement/event/SSA表达selected事实，选择临时对象销毁后
   verifier和lowering结论不变。
3. every successful case产生真实MLIR mutation；no-match/failure保持source byte-identical。
4. every mutation使旧IndexRelation、alias、effect、liveness、completion和resource analysis失效并fresh重算。
5. complete traversal覆盖chain、diamond、fanout/fanin、multi-root、reduction、view、control flow和communication。
   同一source必须覆盖single-region fused-small-tile、multi-region separated-large-tile、same-region separated traversal和
   selective-spill actual forms；internal spill只结束对应root，region inputs/results的variadic DDR fan-in/fan-out无人为上限。
   必须验证一个或多个non-nested regions、显式cross-region materialization、无SPM alias跨界及无per-region自动join；nested、
   artificial boundary和opaque clobber输入是negative。region partition必须作为06联合搜索变量进入actual candidates。
6. every compute/movement通过op verifier、适用的standard interfaces和MemoryEffect/custom resource effects；
   every async issue都有SSA或typed fence completion。
7. selected tile IR经tile-to-instruction conversion、SPM/DDR、all-rank communication、transport和ABI gate直接消费。
8. 任一clone、rank或later exact gate失败都不产生partial accepted IR、bundle、module、artifact或package。
9. rank-count 1/16和冻结7B source-to-package-to-SystemC/PyTorch fresh数值纵向实际执行；局部fixture不算完成。
10. Q32.V mapped DMA、physical fill和oriented GEMM通过typed Tile/Instr/TargetCall/ABI/SystemC纵向后由同一
    materializer消费；其它未实现target能力结构化拒绝。
11. Q32 existing share/recompute、hoist、Cx/NCx GEMM absorption及每个current numeric variant分别有production actual-IR
    形态和negative；Q46 relation-guided absorption另按独立gate验收。floating reduction reorder/tree与integer exact/modular
    均复用Q32.N的numeric validation；online reduction、non-GEMM FMA及超出current integer-domain子集的
    distribution/factorization只有完整typed semantic/numeric/lowering纵向闭合后才能准入，不能靠伪装attr。

## 14. 规划中的 Actual Clone Handoff

`semantic-superoptimization`不改变本文selected TileDataflow IR的语义。source operator propagation和target instruction
synthesis都必须先在isolated module中形成真实、verifier-clean MLIR clone，再进入本文既有的atomic materialization/
complete traversal；不得把`InstructionSketch`、rewrite rule、solver AST、proof certificate或implementation descriptor
物化为TileRegion op/attr。

- source clone仍经06的同一structural frontier准入action驱动，以本文typed view/compute/movement/event/SSA合同物化；
  proof通过不等于selected，也不允许建立可重放的选择清单。
- target synthesis消费complete verified Tile program和baseline complete-rank tile-to-Instr conversion，只输出disposable actual Instr
  clone；每个clone重新执行本文coverage、effect/completion及下游memory/ABI gates，失败不修改selected complete-rank variant。
- Q46 actual-op probe在Q48中迁移为读取actual typed clones/current IR facts后，旧implementation materializer与
  selected/forced字段全部删除；本文不接收替代side table或新的候选IR。

示例的reassociation、GEMM/layout fusion或DMA序列只用于证明同一handoff可工作，不形成case-specific materializer。
