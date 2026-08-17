# Wafer Selected Tile-Dataflow IR、SPM Residency Region 与原子物化

状态：2026-08-08按CardModule / Tile MPMD更新。本文定义selected Tile-dataflow IR的MLIR-native边界；
`wafer.tile.region`表达一个Tile内的显式SPM residency domain。现有typed materialization、
relation和physical-version机制可被新scheduler复用，但旧complete-rank/rank==Tile调用域只作历史背景。
Q49.P、Q50、Q51–Q53各自是否完成只看`tasks/progress.md`。

source structured op 的数学语义始终存在于当前 operation、region、SSA、type、attribute 和标准
MLIR interfaces 中。tasks/06与本文不是“先发布全局plan，再由本文import”的两个管线阶段：上游一次
query-local选择只作为即将发生的rewrite参数，在isolated complete CardModule candidate中由
`PatternRewriter`和`DialectConversion`立即物化为Q50.S已物化、由Q51选中的TensorProgram semantic root所对应的typed compute、
temporal traversal、physical encoding、movement和执行结构。numeric reassociation、reduction tree、share/recompute、hoist
及其它数学等价variant必须已经存在于该TensorProgram root；本层不再生成另一份semantic alternative。物化后的决定只存在actual IR中，
transient参数随即销毁；本文不建立candidate set、局部winner、可重放action列表或跨pass side plan。

`wafer.tile.region`物化一个显式的、严格属于 **单个Tile** 的 **SPM residency domain**：region body中的一个或多个structured traversal共享
同一组可同时驻留、可由SSA直接连接并由同一liveness关系解释的tile、temporary、accumulator和staging。它不要求
global tensor整体进入3 MiB SPM，只要求当前traversal位置的live working set可由下游planner合法放置。region组织
Wafer-tagged SPM memref、view、compute、SPM内movement、peer/collective movement、显式DDR streaming或selective
spill/reload、event和structured control flow。数据input/result是variadic DDR value/view，fan-in、fan-out和边数没有
IR上限；typed scalar/control/event可按各自verifier穿过边界，但不得隐式携带SPM alias。region op不隐式执行搬运，
SPM value/root/alias不得成为region I/O。

每个`wafer.tile.module`可以包含一个或多个non-nested `wafer.tile.region`。region partition、每个region的traversal/
temporal tile shape、跨region materialization、retain/recompute/selective spill、region cut/root release boundary、local movement、
buffer/slot、event order和communication共同进入candidate搜索；root lifetime、coexistence与resource calendar只从这些typed
选择物化出的current IR派生。region数量既不是目标，也不能代替DDR、GS、NCC、completion或tile-utilization成本。两个traversal之间若
有SPM-resident SSA edge，就属于同一residency domain；若选择切成不同region，所有跨界数据必须通过显式DDR
store、可信completion和matching load表达，不能把SPM root、alias或仍访问该root的pending work传给下一region。

selective spill只结束目标SPM root，matching reload建立新root；只要其它root仍跨过该点resident，就不形成完整region cut。
相反，当候选为了独立retile、缩短共同lifetime或其它真实成本而结束切口上的全部SPM residency并显式materialize跨界值时，
可以形成新的sibling region。region boundary本身不是device-wide completion、launch或join边界；它只要求所有仍访问被释放
SPM roots的work已经按typed effect/event证明完成。未建模的opaque SPM clobber仍必须fail closed，不能靠新建region伪装合法。

这里必须区分三个概念：把多个traversal放入同一个region（有时口语称“region fusion”）只表示它们处于同一
Tile SPM residency domain；它不等于op fusion，也不等于coupled traversal。同一region内可以存在相互独立的
loop nests和显式local staging。coupled traversal则要求producer work真实嵌入consumer traversal，由actual SSA直接把
producer tile交给consumer，且不存在独立producer traversal或中间DDR materialization；因此coupled traversal必然在同一
region内，但同一region并不推出coupled traversal或op fusion。

上层可以选择retain/recompute/spill/cut/release boundary，但不得用独立`resident=true`标签或预先计算的lifetime interval
宣告residency；选择必须立即物化成TileRegion、allocation root、SSA、movement、buffer/slot和event-order结构，lifetime随后从current IR重算。
只有actual clone中已经存在明确的SPM
allocation roots、SSA/view/effect、访问顺序和completion，并且没有该edge的DDR round-trip，才能声称实际residency；
physical offset还必须等09对最终Instr clone完成fresh lifetime与packing后才成立。所谓多stage流水也不能由一个pipeline
flag或估算计划代表：每个chunk/temporal iteration、movement issue/wait、physical buffer或rotating slot、执行顺序和
completion都必须最终出现在actual Tile/Instr IR中并通过09–13的late gates。

本文依赖：

- `tasks/05-local-compute-normalization.md`：card-partition-local structured tensor normal form。
- `tasks/06-physical-dataflow-synthesis.md`：physical-dataflow selection、clone生命周期、精确排序和CardExecutable原子形成。
- `tasks/08-physical-realization.md`：physical encoding、view、footprint 和 movement legality。
- `tasks/09-spm-memory-planning.md`、`tasks/12-ddr-memory-planning.md`：从actual region/control-flow/liveness派生的
  allocation domain、complete CardModule candidate lifetime、capacity和offset gate。
- `tasks/10-compute-movement.md`：typed source semantics与selected compute/movement materialization contract。
- `tasks/11-instruction-ir.md`：per-Tile instruction IR与CardExecutable target legality。

## 1. 职责和非目标

本层负责：

- 在isolated complete CardModule candidate上消费Q50.S已物化、由Q51选中的TensorProgram semantic root，并应用调用方本次选择的
  tiling、coupled traversal与等价indexing rewrite；选择参数与本次mutation同寿命，不先形成可跨pass保留的完整plan。
- 保留该TensorProgram root中已经显式存在的share/recompute、hoist、numeric reassociation、reduction tree、split和
  algebraic variant；本层只为其构造physical traversal，不重新选择或生成这些semantic alternatives。
- 按 selected tile domain生成 all-and-only traversal、static tail和合法 reduction sequence。
- 对production capability分类与`TilingInterface`共同证明可切的terminal logical collective，从result tile反向物化
  operand/out tile并融合producer；当前只启用单输入、单输出、shape-preserving `all_reduce`，其每个dynamic loop
  instance只处理当前tile，完整result由显式insert/writeback拼接。其它collective在各自gate闭合前仍是full traversal。
- NoC-resident扩展同时允许从已resident或peer到达的operand tile通过`TilingInterface`正向物化consumer tile，并只通过
  `PartialReductionOpInterface`物化partial/merge；input/parameter、intermediate、partial和output角色均由current
  boundary与SSA relation派生，不进入固定枚举或operator matcher。
- 通过current structured op class、region、indexing maps、DPS/Tiling semantics和direct typed builders创建
  `wafer.tile.*` compute；不建立target-implementation菜单、external-model registry、selector或长期candidate object。
- 物化 Wafer-tagged memref、standard/typed view、resident SSA edge、显式 movement、spill/reload、
  temporary、accumulator 和 staging。
- 对selected stage pipeline物化实际chunk/temporal loop、每段movement、独立或rotating physical buffer、slot reuse
  relation、程序顺序和typed event；最终Instr issue order与completion由11在actual clone上闭合，不能保留pipeline sketch。
- 按selected spatial placement和region partition为每个physical `tile.module`物化一个或多个non-nested SPM residency regions；每个region只包含
  其共同驻留的selected tile schedules和physical dataflow，跨region值用显式materialization连接。partition决定
  物化进actual IR结构，不另存region-plan attr或side table。
- 用 event SSA、typed wait/dependency、MemoryEffects和structured control flow表达issue、event completion、reuse及仍待下游
  完成的observable obligations；本层不物化compiler-derived participant join或terminal drain。
- 让 conversion legality、op/interface verifier 和 fresh analyses 能从 current clone 独立重建全部事实。
- 任一 rewrite、conversion、coverage 或 verifier 失败时丢弃整个 clone，不污染 source 或其它 clone。

本层不负责：

- 不生成或排序local alternate lowering；也不生成或排序tile、encoding、TileRegion、retain/recompute/spill/cut/release boundary、
  movement、buffer/slot或event/task-order选择；lifetime不是proposal，而是下游从actual IR重算的analysis。
- 不因 materialization/lowering 失败改选另一实现，不插入临时 fallback，不切分新的搜索分支。
- 不保存选择历史、评分、失败轨迹、影子调度图或其它 IR 外长期语义。
- 不分配 SPM/DDR physical offset；本层只提供下游从region、control flow、SSA和effects重算lifetime/allocation domain
  所需的事实。
- 不让单个traversal、component或旧task module独立执行Tile→Instr、SPM/DDR planning或completion后再拼接CardModule。
- 不把task、source scope、op名、旧loop边界、layout名、GS或collective机械映射成region boundary；region cut只能来自
  physical-dataflow selection选中的TileRegion、cut/release boundary与materialization结构并由actual IR证明。region数量不单独计奖惩，真实DDR、GS、completion、
  tile utilization和materialization工作分别计价。
- 不因SPM bank phase/conflict选择region partition或DDR spill；09只允许allocator在hard-valid placement中把可重算
  bank phase作为soft preference。当前实现状态见`tasks/progress.md`的physical-dataflow队列。
- 不 lower raw packet、CRT、LLVM、runtime handle 或 package 字段。
- 不通过 op/value/parameter 名、固定 shape、参数顺序或 workload topology 恢复语义。
- 不把单个region、representative temporal tile、单个Tile或局部FileCheck当成完整完成证据。

## 2. Pipeline Contract

    Pipeline position:
    - Upstream IR / input:
      verifier-legal card-local structured tensor IR，以及同一次transformation内通过便宜预筛、将立即物化的
      physical-dataflow wave placement、每个traversal的temporal tile domain/loop order、physical operand/result
      encoding、region partition、retain/recompute/spill/cut/release boundary、movement、buffer/slot和event order，
      以及Q50.S已物化、由Q51选中的TensorProgram semantic root。semantic root中的share/recompute、hoist和
      current numeric variant只作为current IR语义被消费，不在本层重新生成。所有physical选择都引用current op/value并可在mutation前重新验证；
      它们不跨pass发布。当前输入不得含无法由typed effect解释的opaque SPM clobber；跨Tile NoC-resident
      candidate还接收target topology、Tile placement、exact demanded domain与同一transaction中的complete CardModule clone。
      对外函数保持functional input/result ABI；若rewrite library需要destination-style traversal，CardModule owner只在
      private clone中追加并随后消费typed scheduling destinations，不从source参数恢复output角色。
    - Current stage responsibility:
      clone完整CardModule；用PatternRewriter把Q50.S已物化、由Q51选中的TensorProgram root中的现有structured semantics物化为selected traversal，
      并更新真实use-def；不得在这里新建numeric reassociation、reduction tree、share/recompute或hoist alternative；
      从current concrete structured semantics经direct op builders和DialectConversion创建表达SPM residency domain的typed region、view、
      compute、movement、event和SSA relation；按selected placement/partition在每个Tile module中创建一个或多个
      non-nested regions，并在各region内部生成complete traversals及跨region显式materialization；nested region、SPM data
      跨界或没有真实materialization/residency含义的结构边界都拒绝；每次mutation后丢弃旧
      IndexRelation、alias、effect、liveness和resource observations并从current clone重算；
      最后运行conversion legality和tile/dataflow verifier。
    - Output IR / files:
      transaction-local、verifier-legal、覆盖完整card DAG demanded domain的selected tile/dataflow IR；每个Tile module
      包含一个或多个non-nested `wafer.tile.region`作为单Tile SPM residency domains。region之间的所有
      shaped data I/O均为DDR value/view，
      SPM value/root/alias不能跨界；每个boundary前只需完成仍访问被释放SPM roots的work，entry terminal继续负责全部
      observable pending work。若物化失败，在无任何IR mutation的情况下返回failure。
      成功IR只含typed operation/region/type/
      attribute、memref/view、compute、movement、event和SSA；不依赖任何外部解释对象。
    - Downstream consumer:
      target-abstract legality，并对selected CardModule执行per-Tile instruction lowering；
      materialized canonical/unplaced Instr只接受physical-dataflow selection已经选择并物化的worker/slot/order事实，随后进入fresh completion
      reconstruction、liveness-derived per-Tile SPM allocation、CardModule DDR、post-memory communication/transport/resource、ABI gates以及
      CardExecutable verification。completion按真实root reuse、observer、region boundary和tile-module terminal分别验证；任一内部
      traversal/component不得脱离complete candidate单独提交这些不可逆stage。
    - User-level driver / named pipeline:
      wafer-compile source-to-package `search|none` pipeline。wafer-opt只可对同一op/interface/
      conversion做局部parser、verifier和rewrite测试，不形成第二条compile pipeline。
    - Explicit non-goals:
      不在本层重新搜索或重排，不创建独立执行计划，不发布tile/dataflow中间output，
      不让runtime选择physical realization，不声明板端性能或timing。
    - Completion gate:
      contraction、ordinary 2-D affine-window convolution、显式静态padding、pointwise、ordered reduction、view、
      broadcast/slice、fanout/fanin、多root、
      structured control flow和当前已启用的terminal all-reduce tiled traversal均有真实source正负例；每个成功case发生可观察的
      MLIR mutation并覆盖完整traversal；失败保持source不变；输出被instruction、SPM/DDR、
      completion、transport和ABI gate直接消费；同一source覆盖single-region fused-small-tile、multi-region
      separated-large-tile和region内selective-spill actual forms，并验证region data I/O仅为DDR、SPM root不跨界、
      nested/artificial region拒绝、boundary不自动生成join；generic DAG、HF prefill/decode与Llama block以
      `num_partitions=1`形成distinct Tile modules并完成fresh package/no-card。真实板端A/B改善另由16闭合。

## 3. IR 生命周期与唯一事实源

    card-partition-local structured tensor IR
      -> isolated complete CardModule candidate
      -> selected PatternRewriter mutations
      -> selected structured-to-tile DialectConversion
      -> verifier-legal tile/dataflow IR
      -> per-Tile tile-to-instruction DialectConversion
      -> Q50.J derives selected worker/slot/order actual Instr IR
      -> erase and fresh-rebuild dependency-driven completion
      -> derive per-Tile fixed SPM allocation problems from all tile.module roots / lifetimes / coexistence
      -> validate all-and-only root coverage and fixed-capacity placement
      -> CardModule DDR then CardExecutable communication/transport/resource/ABI verification
      -> atomic CardExecutable formation from all Tile modules

物化后的长期事实只有：

- structured loop、tile offsets/sizes、block arguments、yield和SSA use-def；
- shared producer/version的普通multi-use、recomputed producer的独立SSA clone、loop外hoisted op与body capture，以及显式
  combiner tree/state tuple；不存在share/recompute/hoist/tree选择attr；
- selected `wafer.tile.*` op form及其typed numeric/lowering fields；
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

- 一个单Tile SPM residency domain；其中global tensor按tile遍历，只有actual current live working set需要同时驻留；
- variadic region argument/result与enclosing Tile module DDR value/view的SSA relation；DDR root既可来自function external，
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
- SPM value/root/alias作为region operand/result、隐式capture或跨region control state；任何跨region shaped value都必须
  先显式store到DDR并在下一region显式load，region shaped data I/O只能是DDR；
- 仍会访问region-owned SPM root的async device work跨region传播；与SPM无关的typed event/control是否可穿过边界由其
  自身interface和下游completion verifier决定；
- 隐式DDR round-trip、隐式barrier、私有physical SPM arena，或按task/loop/tile shape建立的执行容器。

region不拥有私有physical arena，也不声明同一或不同region的roots具有共同lifetime。每个root的出生、access、completion
和结束从对应Tile current IR的SSA/effect/control flow重算；planner再按真实coexistence/conflict关系形成一个或多个fixed
allocation problems。已证明不重叠的regions/roots可以复用完整3 MiB地址范围，可能重叠的regions必须联合满足容量。
每个root必须被all-and-only一个accepted placement覆盖；solver调用次数只作实现与预算诊断，不是IR语义。allocator不得把
placement结果反写成新的partition选择。

### 4.1 Residency Partition 与 Boundary Contract

每个有assigned work的physical `tile.module`包含一个或多个non-nested `wafer.tile.region`。tasks/06把region partition与temporal tile shape/loop order、
layout/version、TileRegion、retain/recompute/selective spill、cut/release boundary、buffer/slot、event order、movement和communication联合搜索；
本文只把selected结构物化并验证，root lifetime/coexistence从物化后的SSA/effect/control/event关系派生。

每条producer-consumer connection选择以下一种或一组真实physical action：

1. **same-region resident/local**：兼容的SPM SSA、view或local movement；两侧可以使用coupled traversal，也可以在同一region内
   使用独立loop nests，只要actual root lifetime与完整working set合法。前者是op/traversal融合，后者只是共享residency
   domain，不能因region相同而把后者统计为fusion；
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
- 携带lowering必须解释的typed operation kind和parameters；
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
tile-to-instruction conversion创建matching Direct-DTE issue token和exact wait，CardExecutable verification再验证反向peer、
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

1. **建立clone**：clone完整`wafer.card.module`，确认所有selected source op/value仍属于current IR。
2. **重验选择输入**：重新检查source op class/standard interfaces、type/indexing、tile domain、physical operands/results和
   target facts；失配在mutation前失败。
3. **应用structured rewrite**：用`PatternRewriter`执行已选tiling、fusion、producer propagation或等价
   relation rewrite；pattern只能按MLIR rewrite contract更新真实use-def。
4. **物化complete traversal**：生成compact `scf.for`、static tail、branch和合法ordered reduction step，
   并验证每个logical output all-and-only一次。
5. **物化spatial placement、TileRegion partition与physical SSA graph**：按selected structure为all-and-only可用
   Tile创建distinct `wafer.tile.module`，并在每个program内创建一个或多个non-nested regions；在各region内部物化
   op waves、traversal domains/temporal tile shapes、layout/version、retain/recompute、movement、selective spill/streaming、
   root release boundary、buffer/slot和event order，并创建region-owned allocation roots、views、resident edges、temporary、accumulator和staging；
   root lifetime/coexistence不作为输入物化，而由这些current IR事实fresh派生。
   跨region值显式store/completion/load。partition的唯一长期表示就是actual region/SSA/movement结构，不生成shadow plan；
   task/source scope或单个spill不得被机械恢复为region boundary。
6. **物化selected compute**：依据current structured op class、region、indexing maps和DPS/Tiling semantics调用direct typed
   builder创建`wafer.tile.*` compute；物化失败时丢弃clone，不切换隐藏实现。generic只有标准Linalg iterator、affine
   indexing maps和scalar payload共同证明exact GEMM或ordinary static 2-D convolution时才归一到对应typed compute，
   其余走generic baseline。convolution window的stride/dilation从current affine maps推导；上游`tensor.pad`的low/high和
   fill value从该op本身精确物化，不能从输出shape反推或默认成零。
7. **物化movement和events**：在complete CardModule clone中创建boundary/local/staged movement、跨Tile NoC send/recv、spill/reload、tokens和typed
   dependency；若选择多stage流水，同时创建actual chunk循环/切片、每stage movement、buffer/slot relation和执行顺序；
   region/task return不物化terminal drain。所有新value立即接入SSA，不能用pipeline attr替代这些结构。
8. **运行一次CardModule structured-to-tile conversion**：用`ConversionTarget`、`TypeConverter`和rewrite patterns消除
   本层声明illegal的source forms；成功后不能残留需要下游猜测的op。
9. **fresh重算与局部canonicalization**：每次mutation后丢弃旧`IndexRelation`、alias、effect、liveness、
   completion和resource结果。canonicalization只能删除语义、storage和effect均等价的no-op。
10. **验证clone**：运行op/interface verifier、complete traversal、view/alias、effect/completion和conversion
    legality。任一失败丢弃整个clone。

成功结果仍是transaction-local clone。只有后续per-Tile instruction、SPM/DDR、CardExecutable communication、transport和
ABI全部通过，coordinator才能提交包含all-and-only Tiles的结果。materializer永远不直接修改accepted module、
`CardExecutable`或`ExecutablePackage`。

## 8. Structured Semantic Coverage

扩展性按structured semantics组织：

| semantic family | 从current MLIR读取的事实 | 物化结果 |
| --- | --- | --- |
| contraction | iterator types、indexing maps、DPS init、combiner、type和native numeric semantics/permissions | typed GEMM/batch/accumulator chain |
| affine-window convolution | Linalg convolution dimensions、iterator types、symbol-free affine window maps、DPS init、exact multiply-accumulate payload，以及显式`tensor.pad`的static low/high/value | canonical NHWC/XYOI typed convolution；必要permutation显式为movement，padding显式为fill与insert-slice |
| pointwise/relation/select/convert | elementwise iterators、scalar region、dtype和valid domain | typed compute或明确composite |
| reduction | reduction iterators、init、combiner、axis/result mapping和typed numeric contract | composite或native op；支持的浮点类型默认允许现有数值合同下的重排，integer保持exact/modular gate |
| share-vs-recompute | SSA use-def、exact dependent region、effect/speculation和cost choice | shared multi-use或consumer-local producer SSA clone |
| loop-invariant hoist | LoopLike、dominance、invariant operands、effect/completion | loop外真实op和body捕获的dominant SSA value |
| Q50.S已物化的numeric reassociation/tree/distribution | current TensorProgram root中的integer或floating scalar op family、integer overflow/wrap语义和显式combiner/dataflow | 本层保留并tile现有SSA combiner tree或rewritten op DAG；不生成新的数学等价variant，integer no-wrap保持barrier，无隐藏order attr |
| view/reshape/permutation | type、view/subset semantics、index relation和alias proof | metadata view或explicit movement |
| broadcast/slice/concat | indexing relation、static domain和piece coverage | view、movement或structured failure |
| constant tensor | ConstantLike value、logical slice和selected destination encoding | typed fill/load |
| structured control flow | block arguments、yield、loop-carried values和effects | 保留`scf`并显式传递memref/event |
| communication | Tile operands、typed peer/group facts、bytes和completion | explicit NoC movement/event body；无未展开占位 |

通用测试至少覆盖chain、diamond、fanout/fanin、shared-input contraction、multi-root、residual、collective、
多个dtype、整tile、非整除tail、f16/bf16无标注正例以及integer modular/no-wrap正负例。新增source op优先通过现有Linalg、
DPS、Tiling、ViewLike和effect interfaces进入这些family；只有新数学语义不能稳定表达时才扩IR。

每个positive必须从真实source进入production materializer并发生非零IR mutation。negative至少覆盖wrong
type/rank/index relation、dynamic unsupported case、rewrite guard failure、conversion illegal op、missing completion、
all-clones-fail和late CardExecutable verification failure。手写`wafer.tile.*` fixture只补parser/verifier覆盖。

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
- 每个physical `tile.module`包含零个或多个按其assigned work决定的non-nested regions；有work的program中每个region覆盖非空selected traversal，nested region、
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
- 每个Tile的SPM roots/lifetime/coexistence、fixed-capacity placement与all-root coverage；
- CardModule DDR lifetime/capacity/offset；
- cross-Tile communication、completion、transport和ABI；
- target emission、readback和atomic CardExecutable/ExecutablePackage writing。

失败规则：

- pattern返回failure前不得修改IR；若后续步骤失败，整个isolated clone直接丢弃。
- unsupported source semantics、无法表示的relation或illegal conversion不产生partial tile IR。
- 单个clone失败只淘汰该clone；所有clone失败才使physical-dataflow selection失败。
- 任一Tile module失败或complete candidate的later gate失败，不发布partial Tile/module/output/package。
- 失败不得触发本层临时改写source语义、encoding、residency、movement或execution order。

## 11. Q32.V Typed Target Extensions 与其它 Later 能力

Q32.V已排期闭合并由同一physical-dataflow selection消费：

- 非identity、strided或多piece的mapped direct boundary movement；
- physical-footprint fill及其valid/padding/bitpacked domain；
- baseline以外的typed contraction operand orientation和对应target command form。

relation-guided Cx/NCx physical-version absorption属于physical-version assignment而不是新target capability：现有
concrete verifier已接受GEMM/batched GEMM和native reduce的Cx/NCx形态，也已覆盖hardware-supported、
physical-traversal-compatible CT relation/select/logic/convert/bitpacked；Q50.G/H把这些机制接入current representation/movement
candidate，Q51只选择完整assignment。若08的exact physical-map/valid-lane proof允许，前置
`materialize_layout`/GS movement在本clone中消失；packing identity仍只存在于encoding，不增加vector-width或packing
side attr。

下列能力仍有独立前置，不能被当前机制存在误报为已支持：

- immutable prepacked resource writing；
- dynamic shape、复杂mask和target-specific composite。通用producer-consumer tile composition属于06当前合同；只有
  需要新增target-specific composite instruction、复杂dynamic mask或尚无typed numeric semantics的实现才是later；
- 需要新runtime/ABI/SystemC consumer的movement或completion形态。
- Q39已经闭合NoC-resident result/operand/partial traversal、peer movement和Direct-DTE mechanics；Q50迁移其机制，Q51把
  decision owner并入06的physical-dataflow candidate set，本文只物化selected peer/resident Tile IR，13继续拥有typed lowering与
  CardExecutable verification。pre-current-cutover late
  NoC tuple path只作历史资格证据，不再是终态独立pipeline。
- online/streamed reduction只有在typed running state、combine公式、numeric policy、tail与lowering闭合后才进入同一
  reduction candidate domain；未闭合时保留native/partial baseline。没有typed fused semantics的non-GEMM FMA contraction及
  尚未闭合的其它algebraic contraction保持unsupported；

只有target instruction、ABI和执行consumer具备typed合同后，才能启用其中一项。每项扩展必须同批增加
source semantic recognition、direct typed builder、selected op fields、PatternRewriter/DialectConversion materialization、verifier、
instruction lowering、effects/completion以及真实source正负测试。缺少任一纵向时保持unsupported；Q32.V三项是明确
checkpoint，不得因删除provider协议而消失；其它later能力也不得被Q32/Q32.V completion假装支持。

## 12. 通用案例

以下只展示关系，不固定shape或workload。source概念上是一个contraction及其pointwise consumer：

    %mm = linalg.matmul ins(%a, %b) outs(%init)
    %act = linalg.generic ... ins(%mm) outs(%out)

physical-dataflow selection在transformation内选择tile、Cx encoding、TileRegion、retain edge与root release boundary；
current structured op随后确定性lower，root lifetime从生成的SSA/effect/control/event关系派生。materializer在clone中
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

真实op必须携带完整types、typed numeric/lowering fields、effects和tokens；上面省略这些字段只为讲解。
retained edge由`%mm_spm`的SSA use-def直接表达，没有中间spill；actual residency仍由late lifetime/offset gate证明。若layout、capacity、completion或instruction
gate拒绝该clone，clone整体丢弃；materializer不就地换实现。

示例中的shape、tile、Cx和具体compute kind都是选择结果，不是协议。相同合同适用于一般structured graph。
同一source graph至少要能形成三类可比较的actual candidate：一个region内融合producer/consumer并使用较小tile；
把producer/consumer物化成两个regions，以显式DDR store/completion/load换取两侧独立的较大tile/traversal；或仍在一个
region内只spill某个root、让其它root继续驻留。也要保留“独立loop nests但retained edge仍在同一region”的合法形态，
证明traversal coupling、region partition和storage action是相关但不等价的选择。最终region数量只描述winner的residency
结构，不能作为收益依据；offset reuse只由对应Tile的真实lifetime/coexistence决定。

## 13. Completion Gate

本文边界完成必须同时满足：

1. production materialization只通过current source op class/standard interfaces、`PatternRewriter`、builders和
   `DialectConversion`修改isolated clone；没有op-name/shape/workload matcher。
2. success IR只用typed region/view/compute/movement/event/SSA表达selected事实，选择临时对象销毁后
   verifier和lowering结论不变。
3. every successful case产生真实MLIR mutation；no-match/failure保持source byte-identical。
4. every mutation使旧IndexRelation、alias、effect、liveness、completion和resource analysis失效并fresh重算。
5. complete traversal覆盖chain、diamond、fanout/fanin、multi-root、reduction、ordinary 2-D affine-window convolution、
   explicit static padding、view、control flow和communication。
   同一source必须覆盖single-region fused-small-tile、multi-region separated-large-tile、same-region separated traversal和
   selective-spill actual forms；internal spill只结束对应root，region inputs/results的variadic DDR fan-in/fan-out无人为上限。
   必须验证一个或多个non-nested regions、显式cross-region materialization、无SPM alias跨界及无per-region自动join；nested、
   artificial boundary和opaque clobber输入是negative。region partition必须作为06联合搜索变量进入actual candidates。
6. every compute/movement通过op verifier、适用的standard interfaces和MemoryEffect/custom resource effects；
   every async issue都有SSA或typed fence completion。
7. selected Tile IR经per-Tile tile-to-instruction conversion、SPM/DDR、cross-Tile communication、transport和ABI gate直接消费。
8. 任一clone、Tile module或later exact gate失败都不产生partial accepted IR、CardExecutable、module、output或package。
9. generic DAG、HF prefill/decode与Llama block以card-level `num_partitions=1`完成source-to-package-to-no-card fresh纵向；
   package必须含all-and-only topology-available Tile launch entries，且允许per-Tile op/loop/shape不同。局部fixture不算完成。
10. Q32.V mapped DMA、physical fill和oriented GEMM通过typed Tile/Instr/TargetCall/ABI/SystemC纵向后由同一
    materializer消费；其它未实现target能力结构化拒绝。
11. Q50.S为Q32 existing share/recompute、hoist及每个current numeric variant分别产生production TensorProgram actual-IR
    roots和negative；Q51只选择其中一个root并展开physical dataflow，本层证明selected root可被physical tiling消费，
    不重新生成alternative。Cx/NCx GEMM与CT relation-guided absorption由Q50.G/H物化，并随Q51 selected assignment按同一
    physical gate验收。floating reduction reorder/tree与integer exact/modular
    均复用Q32.N的numeric validation；online reduction、non-GEMM FMA及超出current integer-domain子集的
    distribution/factorization只有完整typed semantic/numeric/lowering纵向闭合后才能准入，不能靠伪装attr。

## 14. Structured Alternative Boundary

`semantic-superoptimization`不改变本文selected TileRegion IR的语义。Q50.S先让所有语义alternative在physical mapping前形成
真实、verifier-clean actual TensorProgram roots，Q51只选择一个root并展开同一physical-dataflow search；不得把instruction sketch、rewrite rule、
solver AST、proof certificate或implementation descriptor物化为TileRegion op/attr。

- actual TensorProgram alternative与baseline进入06的同一spatial/temporal/fusion/communication search；proof通过不等于
  selected，也不允许建立可重放的选择清单；
- Q48是未来对Q50.S同一semantic-alternative builder seam的扩展，不在CardModule或Instr形成后启动第二个candidate selector；target-specific Instr canonicalization仍由11/14的
  deterministic lowering owner负责，不改变Q51 winner；
- 本文不接收额外selector、selected/forced字段、side table或新的候选IR。

示例的reassociation、GEMM/layout fusion或DMA序列只用于证明同一boundary可工作，不形成case-specific materializer。
