# Wafer Compiler Bug Patterns

本文件记录可复用的现象、根因、修复和防复发模式。一次性case状态、历史输出、临时workaround和已经退役接口不在
这里保存；具体任务证据归编号设计、progress或原始测试记录。

## Card partition与Tile被混成一个domain

- 现象：GSPMD的partition count直接决定单卡Tile程序数量，spatial mapping、不同op并行和cross-Tile communication搜索消失。
- 根因：把card-level global tensor partition误当成card内执行映射，并复用同一个整数/ordinal贯穿frontend到runtime。
- 修复模式：`num_partitions`只属于card domain；card-local structured DAG进入physical-dataflow selection，后者显式生成
  all-and-only 16个Tile modules。
- 防复发：single-card `num_partitions=1`的source必须产生16个Tile interfaces；测试同时检查card count、Tile count和
  nontrivial spatial mapping，不能只检查module数量。

## `tile_id`与`launch_slot`不可互相推导

- 现象：identity mapping时所有测试通过，改变Tile枚举顺序后module body、resource或diagnostic绑定到错误Tile。
- 根因：producer或consumer使用vector ordinal、pid、entry index代替typed physical relation。
- 修复模式：从topology到package/runtime/model始终转发 `(card_id,tile_id,launch_slot)`；launch slot只负责dense canonical order。
- 防复发：target modules、linked ELF、JIT、no-card和provider验证都包含非恒等mapping正例及duplicate/missing负例。

## Optimization policy被接收但没有进入search owner

- 现象：`search`与`none`在所有workload上产生相同IR/package，新增搜索代码从未影响winner。
- 根因：driver解析了optimization policy，却在executable search边界丢弃或绕过它；测试反而把相同结果锁成合同。
- 修复模式：`search`与`none`是独立controller并拥有各自的actual IR materializer和candidate/attempt owner。Baseline
  从current TensorProgram和固定规则直接构造IR，不创建search structural choice state；search才消费explicit choice。两者只共享
  policy-free source facts、single-op rewrite/conversion和无policy SPM/DDR/target leaf，互不调用或fallback。
- 防复发：test-only call/work witness分别证明None的search-session为零、Search的baseline-controller为零；pre-structural
  frontier不构造IR，structural choice闭合后只产生一份candidate owner，winner不重建且publication一次。普通compile不创建统计对象。

## 多层C++ shadow plan不能代替future IR

- 现象：search在actual IR之前依次建立future value、movement、storage、event、execution structure和schedule record，
  最后由materializer重放成MLIR。一旦不对应，便继续增加ID mapping、expected inventory和plan/actual parity verifier。
- 根因：把“避免物化候选IR”误当成架构目标，混淆了transformation choice与只能由current IR表达或派生的
  operation、SSA、buffer、alias、lifetime、effect、order和completion事实。每个局部schema都能解决一个接线问题，
  但合起来形成与MLIR并行的第二编译器。
- 修复模式：search只保存尚未被消费的显式choice。Spatial/region/temporal choice闭合后立即在candidate-owned
  `IsolatedFromAbove` transaction中生成actual TileRegion IR。之后每个choice都作用于current IR，经verifier后使旧analysis失效；
  layout/bufferization、movement、execution structure、Instr scheduling/completion和memory planning只读当前stage IR；memory leaf只接受
  completion-closed Instr，不重建join/wait。Rejected owner销毁，Accepted owner不重建。
- 防复发：每个新plan字段先分类为“choice”或“物化后IR事实”。后者不得进入C++ cross-stage state。搜索源码和active docs
  必须保持future operation/value/buffer/event/schedule owner、rebuild/parity verifier和winner replay为零；每个stage用actual IR数量、
  SSA/effect/lifetime witness和直接下游验证，不用plan inventory代签。

## 在structural choice闭合前物化或重建candidate会导致时间与RSS失控

- 现象：model-scale图对未闭合structural choice就clone/lower，或者同一candidate IR先probe、再actual memory/target gate、winner再重建；大量actual owners同时
  存活，编译时间和RSS失控。
- 根因：没有明确pre-structural choice frontier与candidate-owned actual IR的materialization boundary，也没有move-only accepted owner。
- 修复模式：pre-structural state只运行typed choice/query；spatial/region/temporal闭合后物化一份candidate-owned TileRegion IR。
  后续choice作用于current IR；rejected/loser owner立即销毁，session只保留必要summary和一个retained incumbent，winner原样发布。
- 防复发：显式计数structural materialization、后stage transformation、actual gate和winner publication，断言同一choice/IR epoch不重建。
  不能用低candidate count掩盖winner rematerialization或16 Tile重复整图分析。

## Late exact failure不能触发隐藏repair

- 现象：SPM packing或ABI failure后，late pass自行缩tile、spill、改worker或切communication，selected IR与search cost不一致。
- 根因：allocator/finalizer被赋予了搜索职责，产生第二winner owner。
- 修复模式：actual gate只消费当前candidate IR并返回typed Accepted/rejection/failure；allocator/finalizer自身不得缩tile、spill、
  改worker或切communication。带完整witness的rejection可由外层当前policy controller消费，其它状态不得伪装成rejection。
- 防复发：failure injection锁定packing/ABI/target失败不在candidate IR内repair、不调用另一policy；repo scan禁止late
  retile/spill/replan selector和allocator fallback。

## 任务顺序不能让query输入或mechanism consumer凭空出现

- 现象：设计把exact-demand analysis排在closed spatial assignment producer之前；或者把search controller foundation拖到全部physical axes之后，使中间交付的
  search-only domains没有production consumer。线性任务名看似无环，实际API producer/consumer断裂。
- 根因：按任务编号或“先mechanism、后统一接线”排期，没有分别列出representation foundation、query、full domain、Core consumer和
  post-choice invalidation。
- 修复模式：先交付choice representation、structural validation和真实producer，再实现query与full domain。Spatial顺序为
  `spatial foundation → exact demand → full spatial domain`。Spatial/region/temporal choice闭合后立即生成actual TileRegion IR；后续
  layout/movement/buffer/schedule能力必须以该current IR为输入同批交付producer和consumer，不再串接future-plan artifact DAG。
- 防复发：每个checkpoint表列出输入typed object、唯一producer、输出、首个production consumer和invalidates/re-entry；对该artifact DAG
  做拓扑检查。没有producer的input、没有consumer的mechanism、或被invalidated后仍直达下游的edge都使计划未收敛。

## 性能未知项不能阻塞或污染比较

- 现象：大量candidate因某个硬件参数未校准而不可比较，或把缺失参数按零/无穷大后选择出现偏置。
- 根因：把hard legality与performance knowledge混为一个状态，并允许candidate-local缺项。
- 修复模式：cohort开始时统一确定enabled terms；有实值用实值，否则用明确理论值，完全未知项从全部candidate删除。
- 防复发：逐项移除optional rate，排序仍确定；同一term不能只对部分candidate启用；hard capacity仍独立fail closed。

## Algorithm shortcut会缩窄通用search

- 现象：为某个Attention/decode或shape case加入名字/形状matcher、独立public selector/pass、opaque provider key或第二winner
  分支，异构DAG无法复用；另一种错误是完全禁止typed算法变换，使online/partition-merge DAG永远不可能出现。
- 根因：没有区分“由typed SSA证明并物化actual TensorProgram alternative”和“按模型名进入第二实现协议”。
- 修复模式：算法资格只从structured semantics、indexing relation、type、SSA/effect、loop-carried dataflow和numeric policy证明；
  每个算法/拓扑参数点先成为verifier-legal actual TensorProgram，再由同一physical-dataflow owner选择。名字、shape或独立
  selector不进入协议，physical tiling/layout/buffer也不能提前替代算法DAG选择。
- 防复发：源码检查禁止framework/model/function/value-name和固定shape matcher；generic mixed DAG与真实HF/Llama走同一public
  pipeline，并以正负witness证明typed semantic builder既会命中合法图，也会保持near-miss原图。

## Frontend不得为了compiler命中改写模型语义

- 现象：fixture手写mask、替换RoPE、预先计算state或重排参数后优化case通过，但真实framework export失败或数值不同。
- 根因：把case构造器当成compiler semantic adapter，形成两个source事实源。
- 修复模式：operation、constant、mask、RoPE、scalar flow、dtype、control flow和function boundary原样进入compiler；允许的
  preprocessing必须是模型本身的普通值计算。
- 防复发：source fidelity gate比较exported graph与官方API语义；mask中的finite值/`-inf`不由compiler注入或删改。

## 高层关系被过早lower后不能靠名字恢复

- 现象：static concatenate或structured contraction在需要spatial/state analysis前被降成scalar index loops，后续只能靠
  tensor name、shape或operand位置猜测关系。
- 根因：normalization顺序错误，或analysis没有沿exact reshape/slice的SSA传播。
- 修复模式：通用static concat先规范成fresh tensor上的exact `insert_slice`链；exact reshape/collapse/expand在relation analysis中
  透明传播；multi-input contraction使用匹配的structured semantics。
- 防复发：generic concat、reshape、multi-leading-dimension contraction均有正负例，不添加模型专用cache marker。

## Spatial、temporal、fusion不能分阶段各自定案

- 现象：先固定Tile分配再选temporal tile/fusion，或先尽量融合再事后安排NoC，导致SPM放不下、Tile空闲或通信爆炸。
- 根因：将互相决定resource和critical path的变量交给独立selector。
- 修复模式：spatial/region/temporal作为联合structural choice，闭合后立即生成actual Card/TileRegion IR。Communication、buffering和
  overlap choice作用于current candidate IR并生成new verified IR；只有通过actual gate的owner参与winner比较。
- 防复发：测试同时保留maximal local residency与cross-Tile operator pipeline、large-tile cut与small-tile overlap等对立候选。

## 修改temporal size时必须同时关闭active order

- 现象：full-local plan的order为空；actual SPM反馈缩小某个size后仍保留空order。旧emitter把空order解释成隐式source order，所以case能跑，
  但同一plan不属于complete temporal domain，query与apply合同分裂。
- 根因：baseline refinement只改size字段，没有通过baseline-owned deterministic temporal validator重新关闭
  由`size < extent`产生的active iterators。
- 修复模式：先在临时size vector上计算precedence DAG的stable first linear extension，成功后在同一baseline materializer
  调用内原子构造size/order对应的actual loop IR；不构造或调用search temporal domain。
- 防复发：actual-feedback unit必须断言refined actual loop的active iterators、order和tail完整，且baseline search-domain
  call count为零；不能以lowering对空order的隐式解释代签。

## Tile region不要求所有op使用相同tile shape

- 现象：producer/consumer tile size不同时被迫切region并写DDR，或者错误要求一个region共享统一tile vector。
- 根因：把SPM residency domain误当成hardware Tile或统一iteration domain。
- 修复模式：`tile.region`只拥有一个Tile内的SPM lifetime boundary；body可以包含不同tile shape的coupled或独立traversal。
- 防复发：same-region different temporal tiles与selective spill正例；cross-Tile SPM SSA alias负例。

## Completion必须从final actual IR重建

- 现象：rewrite改变worker/order/communication后沿用旧join/wait，过早reuse或entry返回；也可能保留多余等待损害性能。
- 根因：completion被当成持久plan而不是current effects/tokens/control flow的派生语义。
- 修复模式：Instr verification先清除旧required joins，再从current actual Instr fresh构造loop backedge、branch merge、entry return、
  engine join和Direct-DTE exact wait。
- 防复发：selected actual Instr执行fresh completion，直接检查current worker/effect/token/lifetime与join/wait位置；不与future event plan做parity。
- memory planner中的第二次completion rebuild会掩盖上游stage未闭合，并让单独调用leaf与production结果不同。Function-boundary
  bufferization必须先于TileRegion relation listener；TileRegion-to-Instr和任何selected order mutation完成后，由current-Instr owner执行
  唯一fresh rebuild，再把IR交给只验证completion的memory/target leaf。Named pipeline和compiler driver必须调用同一fresh kernel。

## `ReturnAfterLocalDrain`不是card-scoped barrier

- 现象：每个Tile local return合法，却在其它Tile或transport尚未完成时发布output；或在每个entry尾插入全卡等待造成死锁。
- 根因：混淆entry-local drain、cross-Tile message completion和card-scoped invocation success。
- 修复模式：Tile entry只保证本地发起的observable/reuse/status work已收敛；CardExecutable/runtime owner另行等待16个entries与全部transport obligations。
- 防复发：一个Tile提前返回、另一个仍有Direct-DTE/event的正例；缺失global obligations和多余cycle分别失败。

## SPM packing只能求解actual fixed problem

- 现象：partial planner、footprint模型或allocator按iteration volume、operand byte sum和buffer数量提前判断fit；或者capacity失败后
  在planner/emitter内自行retile、spill、换layout或rebuffer。
- 根因：没有先形成包含actual allocation、layout、alias/effect、completion、lifetime和alignment的current Instr problem，
  把性能估算、候选选择和fixed-capacity allocation混成一个owner。
- 修复模式：pre-actual-memory stage的SPM状态保持unknown；每个candidate先实际构造Card/Tile/Instr及current owner relation，
  唯一PlanSPMMemory/MiniMalloc只返回validated offsets或带actual conflict demand的typed capacity rejection。外层controller决定
  是否构造下一candidate，planner和emitter不repair。
- 防复发：用iteration-volume、operand-size和actual lifetime给出相反预测的GEMM/affine-window case，证明前两者不改变合法集合；
  aligned/ragged actual candidate覆盖alias、async use、tail、alignment、overlap clique和offset。
## 资源耗尽不能伪装成exact infeasible

- 现象：allocator或局部solver达到work budget、timeout或内部错误后，search把该结果缓存成capacity failure/no-good，合法的
  spatial、temporal、layout或buffer siblings从candidate domain永久消失。
- 根因：调用边界只有success/failure布尔值，没有区分`ProvenInfeasible`与`ResourceExhausted`/internal failure。
- 修复模式：mechanism统一返回accepted、deferred、proven exact rejection或indeterminate；只有证明无解或确定unsupported
  才能形成causal no-good，资源耗尽、timeout和内部失败只消耗work并保留parent与siblings。
- 防复发：定向测试让同一assignment分别触发证明无解、预算耗尽和内部错误，断言只有第一类缩域；结果等级按exact candidate set
  是否仍完整报告，不能仅因budget中止一律声称有界或一律降级。

## Logical elements与physical bytes不可混用

- 现象：bitpacked/blocked/padded tensor的SPM/DDR range、movement cost或`TileEntryArgument`尺寸按element count计算，出现越界或错误收益。
- 根因：layout enum与physical codec没有成为size/address唯一事实源。
- 修复模式：数学work使用logical elements；allocation、address、ABI、movement与transport使用checked physical footprint。
- 防复发：bitpacked、padding、non-unit stride、alignment和overflow正负例同时覆盖，byte size相等不推断layout。

## Cross-Tile communication不能从编号或名字恢复

- 现象：在identity topology上route正确，改变physical mapping后send/recv、collective tree或status resource错误。
- 根因：使用Tile编号算术、symbol/name、source partition或container order推断endpoint和message correspondence。
- 修复模式：IR显式保存physical source/destination、message identity、domain、encoding与bytes；topology analysis只读current typed topology。
- 防复发：non-identity topology、partial overlap mapping、fanout/fanin、mismatched payload和missing recv负例。

## Program tensor、target representation和device memory不能共用一个identity

- 现象：每个Tile重复保存或上传同一parameter，或者package file offset被直接当成`txMalloc` handle；相反，多个Tile的
  workspace/status又因role相同而错误共享。
- 根因：一个`ResourceId`同时承担logical ProgramTensor、selected target representation、package bytes和provider allocation identity，
  consumer只能从name、shape、role或重复argument猜sharing。
- 修复模式：`ProgramTensor`、`TargetTensor`、program-data file range和runtime `BoardDeviceMemory base + checked offset`分层；
  sharing只由多个`TileEntryArgument`显式引用同一TargetTensor表达。每Tileworkspace/status在invocation memory中取得独立range。
- 防复发：同一TargetTensor只materialize和上传一次；不同TargetTensor即使digest相同也不合并；workspace/status ranges按Tile
  non-overlap；package ID与provider allocation handle互换必须失败。

## Host H2D与编译生成的DDR↔SPM搬运不能混成一层

- 现象：runtime按parameter或Tile反复allocate/copy，甚至尝试在launch时重新规划layout和片上搬运，导致完整模型启动成本和地址合同失控。
- 根因：套用GPU buffer API，把host→device初始化、global DDR storage和device执行期间的local memory movement视为同一动作。
- 修复模式：compiler用`MemLayout`、physical bytes和accepted offsets决定TargetTensor及RDMA/WDMA；runtime只取得一块program-data
  `BoardDeviceMemory`和一块invocation `BoardDeviceMemory`，完成必要H2D，并把`base + checked offset`传给Tile entry。
- 防复发：fake provider检查non-empty program data一次allocation/整体H2D、empty program data零provider call、invocation一次allocation且无per-Tile allocator call；target测试检查
  workspace仍为`workspaceBase + wafer.ddr.offset`，device RDMA/WDMA继续消费该DDR地址。

## Tile entry argument不是runtime pointer-row slot

- 现象：target entry argument被命名为`KernelABISlot`，后续设计误以为它拥有runtime pointer-row storage，
  或把argument identity与pointer-row child range混成一个对象。
- 根因：用runtime carrier中的“slot”给compiler target entry argument命名。
- 修复模式：稳定语义名为`TileEntryArgument`；它记录ordinal、closed kind、target descriptor、bytes/alignment和access，
  current kernel pointer row只是runtime按该schema实现的地址表，不能反向拥有argument语义。
- 防复发：kernel wrapper从同一个argument schema生成并readback；实现改名同批替换全部producer/consumer，
  `txLoadGraph`/`txLaunchModel`退出current产品接口，不保留旧symbol、枚举值或alias。

## 持久化接口更新不能保留兼容reader

- 现象：serializer写current fields，parser仍接受退役field/alias并填默认值，导致runtime得到无法验证的physical identity或completion。
- 根因：把wire升级当成渐进迁移，而当前项目没有必须兼容的外部consumer。
- 修复模式：每类serialized file只有一个current schema identity和exact field contract；删旧reader、translator、wrapper、fixture
  和CLI，旧值在产生外部effect前失败。
- 防复发：current canonical roundtrip与退役field/identity拒绝成对测试；跨模块enum/key/identity只有一个owner。

## Aggregate module不能吞掉Tile interfaces

- 现象：Grid/Cluster低层合成一个module后，只保留一个entry/interface，或dispatcher用pid直接当launch slot选择body。
- 根因：把module topology等同execution domain，并默认 `pid == launch_slot == tile_id`。
- 修复模式：aggregate只合并code payload；target modules/package仍保存16个explicit Tile interfaces与typed mapping，dispatch读取verified
  launch-slot relation。
- 防复发：不同Tile body、共享module、non-identity tile/launch-slot mapping与高ordinal `TileEntryArgument`的集成测试。

## Internal JIT bridge不是runtime ABI

- 现象：host target-call dispatcher symbol被写入package contract、public enum或外部tool，后续JIT实现无法演进。
- 根因：把实现桥接点误当成稳定IR或文件格式边界。
- 修复模式：public合同只到owner-backed target module set、typed descriptors和transaction sink；dispatcher保持internal且不序列化。
- 防复发：package/export allowlist不要求internal host symbol；文档和public header不暴露其调用约定。

## Target LLVM不能被不同consumer重复lower

- 现象：target code generation、model和host frontend各自从accepted IR重新lower，target annotations、Tile entry arguments或call ordinals漂移。
- 根因：没有owner-backed same-invocation target LLVM boundary。
- 修复模式：accepted Tile只翻译一次，target module连同LLVMContext move-own；下游共享不可变owner set。
- 防复发：测试统计单次translation，并让package/TargetCall/SystemC消费同一owner；metadata逐field readback。

## TargetCall语义不能从symbol解析

- 现象：新增或重命名CRT symbol后profiler/model把transaction归错family，或者参数位置发生静默漂移。
- 根因：多个consumer复制symbol table或substring matcher。
- 修复模式：closed `TargetCallDescriptor` registry统一symbol、signature、semantic、issue domain和decoder。
- 防复发：每个descriptor用位置互异sentinel roundtrip；unknown/width/enum/format错误拒绝；source conformance消费完整受控source set。

## Runtime必须在首个side effect前完成validation

- 现象：发现binding、module export或transport capability错误时已经分配内存/加载module，cleanup与错误归因复杂。
- 根因：semantic verification分散在provider调用过程中。
- 修复模式：no-card/runtime validation先闭合manifest、capability、program data、ports、modules、phases、entries、memory plan和
  16-Tile invocation plan，再允许allocation。
- 防复发：每类invalid input断言provider call count为零；no-card与board共享同一plan builder。

## Partial submission必须poison session

- 现象：provider只接受部分Tile或返回不可信状态后runtime继续提交/cleanup/retry，可能复用未知设备状态。
- 根因：错误处理假设submit失败等价于“没有side effect”。
- 修复模式：unknown或non-empty accepted subset、timeout和不可信completion使context sticky poisoned；poison后不再调用provider。
- 防复发：failure injection覆盖每个stage、accepted subset与cleanup；无自动retry/reset/power，session move/invalid状态严格验证。

## Card-shared output必须在card-scoped invocation完成后一次读取

- 现象：某个Tile完成就D2H shared output，读到其它Tile尚未写完的区域；或同一output port被多次copyback覆盖。
- 根因：把Tile-local completion和card-scoped output readback混淆。
- 修复模式：所有phases、16个entries和transport status验证后，按unique output port的planned range一次D2H；随后原子构造result。
- 防复发：不同Tile写disjoint slices的共享output测试，提前D2H和duplicate copyback失败。

## Target model不能为每个TileEntryArgument复制card data

- 现象：functional model按`(launch_slot, argument_ordinal)`建立16份input/output/parameter bytes；跨Tile写入彼此不可见，
  model可能错误通过board上会失败的程序。
- 根因：model把entry argument引用位置当成memory identity，并把所有argument address强制为互不重叠。
- 修复模式：model从显式ProgramTensor/TargetTensor/port与`TileEntryArgument`relation建立private memory；card data共享一个base，
  workspace/status由Tile独占；unique output port只发布一次。model-private memory不定义package/provider allocation identity。
- 防复发：两Tile通过不同arguments读写同一card output必须互相可见；Tile workspace必须隔离；同一TargetTensor使用不同base、
  不同TargetTensor复用重叠base和按name猜alias都必须fail closed。

## SystemC身份不能来自OS thread或调用顺序

- 现象：并发调度顺序改变后transaction被记到错误Tile，private memory alias或completion关系随host scheduling漂移。
- 根因：用thread-local默认值、注册顺序或symbol恢复physical context。
- 修复模式：JIT bridge显式绑定card/tile/launch slot；每个transaction携带typed identity和Tile-local issue ordinal。
- 防复发：并发顺序扰动与non-identity mapping下result/transaction digest仍确定；缺失binding在执行前失败。

## Functional model不能签发硬件性能结论

- 现象：SystemC event count、host JIT时间或theoretical lower bound被写成board latency/throughput改善。
- 根因：混淆functional ordering、host overhead、理论模型和真实设备measurement。
- 修复模式：model只发布functional/numeric结果；board performance使用same-source matched A/B和真实device observations。
- 防复发：报告模板明确evidence level；没有board samples时不出现hardware speedup结论。

## 历史board输出不能重新签发当前结论

- 现象：代码或gate改变后读取旧raw/log并更新summary，得到“fresh通过”但没有新device launch。
- 根因：把审计记录当成可重放test input。
- 修复模式：历史raw只读归档；current结论仅来自本轮build、启动和输出。普通case不重复共享qualification。
- 防复发：runner不接受历史result路径作为输入；board task记录build/package digest、current payload和本轮输出时间。

## 过时测试不能阻止删除旧接口

- 现象：production代码保留deprecated enum、schema field、wrapper或selector，只因为旧fixture仍编译或golden期待它。
- 根因：把test视为需求owner，而不是current design contract的验证者。
- 修复模式：先确认current pipeline contract；删除旧producer/consumer/API/CMake/test，必要的通用negative迁移到新owner。
- 防复发：repo-wide residual scan覆盖header/source/build/test/docs/memory；不允许empty stub、compatibility alias或只为旧case存在的target。

## 不得从函数参数位置或同型关系猜测output boundary

- 现象：functional tensor program的最后一个真实input与result同型时，被CardModule lowering当作trailing output参数删除；
  final Tile entry参数减少，但frontend resource binding仍完整，16个Tile统一在TargetABI exact-boundary gate失败。
- 根因：把structured op内部的destination-style语义错误提升成source function ABI，并用
  `numArguments - numResults`恢复角色。
- 修复模式：source函数只按已验证functional arguments/results消费；需要可写destination时，在private scheduling clone
  中显式追加result destinations，记录source argument count，物化后只删除这个精确区间。
- 防复发：覆盖单input同型result、同型尾部普通input、multiple results和no-work Tile；检查final entry arguments/results
  与frontend typed bindings精确双射。

## CeilDiv breakpoint搜索必须证明严格前进

- 现象：temporal tile已经很小时，循环用`ceilDiv(extent, waves + 1)`求下一候选；相邻wave count可能仍映射到同一tile
  size，导致cheap candidate derivation无限循环，单个普通source compile看起来卡死。
- 根因：把wave count变化误当成tile-size equivalence class变化，没有为循环variant证明严格单调。
- 修复模式：从当前`ceilDiv(extent, tile)`等价类直接计算下一类的最大tile，再canonicalize到真实breakpoint；每次迭代断言
  `1 <= next < current`，乘法和footprint同时使用saturating arithmetic。
- 防复发：覆盖extent能让多个相邻wave count落入同一tile的形状；测试不仅检查最终值，还检查有限步数和严格下降。

## Source node与selected Instr不是一对一关系

- 现象：合法的dead init消除或producer/consumer融合后，final Instr没有某个source node的独立operation，cost plan把整个
  exact-verified candidate误判为source relation丢失。
- 根因：把query-local source identity当成一源节点一最终指令的持久映射，忽略合法elimination和fusion。
- 修复模式：observable terminal source relation保持强制；内部pure node只有在全部observable successor路径已被下游覆盖时才允许空
  phase；融合多个source node的operation必须由DAG证明唯一downstream consumer并只计一次，foreign/incomparable relation仍拒绝。
- 防复发：内部节点消除、dependent fusion、terminal relation丢失和ambiguous fusion正负例成对覆盖，且phase raw work与whole accepted
  IR保持精确守恒。

## Logical raw-exact不能用不同padding模板的整块digest代签

- 现象：formal与backend的每个logical element逐bit相同，但qualification又比较从零模板打包的formal storage和backend
  storage整块digest；backend只改了不可观察padding，freeze却报raw-exact digest不一致。
- 根因：把logical tensor comparator与physical storage repeatability混成一个判据。
- 修复模式：logical raw-exact逐元素比较；需要对比formal/backend storage digest时，把formal logical values覆盖到backend最终
  storage同一padding模板。backend完整storage digest仍独立冻结，用于同一backend执行的repeatability。
- 防复发：选择带physical padding的形状，分别验证logical bit差异必须失败、仅padding差异不改变tensor数值结论、backend
  storage漂移仍由repeatability validation捕获。

## Temporal search必须覆盖内部reduction iterator

- 现象：observable output tile很小，但上游projection或downstream contraction仍把完整K维weight计入单Tile SPM；继续缩output
  tile也无法降低驻留，decode/Llama等DAG在exact packing统一失败。
- 根因：candidate只记录function result的temporal shape，没有为每个structured op的iterator domain选择temporal tile，因而
  reduction K维被排除在搜索空间外。
- 修复模式：为每个current structured op记录覆盖全部parallel/reduction iterator的完整tile向量并生成有限breakpoint；
  reduction iterator用typed accumulator、prologue/steady `scf.for`/tail递归物化，叶子统一调用TilingInterface，再沿每个
  operand indexing map反推exact window。不得把某一个reduction轴另存成旁路chunk合同。
- 防复发：小output/大K matmul与多级projection正例证明full weight不resident；检查accumulator、chunk coverage、tail、actual loop
  cost multiplicity和source arithmetic op/dtype保持不变，不按模型名、shape或operand位置特判，也不引入numeric policy。

## Temporal reduction变换不建立数值策略

- 现象（历史）：reduction分块曾引入`fastmath`、源顺序、整数overflow flag和typed comparator等额外gate，使同一个IR结构变换
  被误写成numeric policy判断，并让domain与actual emitter接受不同的order。
- 收敛结论：删除独立的reassociation/numeric legality helper及全部temporal调用。temporal domain只决定iterator size/order，
  materializer保留source combiner operation与dtype并构造对应loop-carried state；本层不判断数值可交换性、不选择comparator，也不
  建立numeric search axis。
- 防复发：reduction temporal测试断言all-and-only iterator coverage、main/tail、loop order、DPS init和source arithmetic op/dtype仍在；
  不得以`fastmath`、combiner类别、overflow flag或外部comparator决定temporal candidate是否存在。纯reduction标量输出没有parallel轴时仍必须先构造all-factor=1、单参与Tile的typed
  unpartitioned functional coordinate；这是该root的无parallel轴退化，不是baseline全局Tile数。current placement domain表达不了是baseline implementation gap，不能用来跳过temporal
  split或把source判unsupported。已知「最小tile超SPM」反例必须由最小complete candidate的actual SPM rejection证明；没有该结果时
  保持unknown，不用无关placement失败冒充负例。

## Affine-window convolution不能退化成projected-permutation generic

- 现象：PyTorch/HLO normalization把named convolution正规化为`tensor.pad`加带`o*stride+k*dilation`索引的
  `linalg.generic`；只接受projected-permutation maps的lowering会在合法source上报unsupported indexing maps。
- 根因：把ordinary structured convolution误当成某个named op或shape特例，没有使用Linalg iterator、affine indexing map和
  scalar payload恢复当前数学语义；同时试图从output shape猜padding会丢失before/after与fill value。
- 修复模式：先用标准Linalg convolution-dimension inference验证window family，再从symbol-free maps精确提取permutation、
  stride和dilation并验证multiply-accumulate payload；显式`tensor.pad`只从current static low/high和原始
  position-independent value物化fill/insert-slice，map-only form仅在geometry证明implicit pad/unpad为零时准入。
- 防复发：正例覆盖named-op被generalize后的非方形/permuted affine window与非零显式padding；负例覆盖错误iterator、map、
  payload、dynamic/position-dependent padding和shape relation，并验证source→typed Tile conv→Instr conv纵向。

## Fragment carrier不能反向缩窄exact edge relation

- 现象：producer和consumer使用相同Tile集合，且两边都记录`shardDimension = 0`时，edge planner直接选择local fusion；
  transpose却把consumer result维0映射到producer维1，本应发生的peer transfer消失；之后为修正它加入的
  functional/projected-permutation限制又把合法的disjoint reduction误判为无exact relation。
- 根因：先把各node自己的result-dimension编号当成跨op公共坐标，随后又把当前稠密矩形fragment carrier的表达限制反向写成
  logical relation legality；edge planner和materializer还各自重建一份indexing-map判断。
- 修复模式：从producer/consumer共享structured iteration domain只导出一次query-local `IndexRelation`，对consumer shard用
  relation image求all-and-only producer demand；一对多relation保持合法。矩形、strided或多片传输只在后续target realizability
  分层判断。同Tile独立traversal使用`LocalShardResidency`；search的coupling identity是actual connected node group和共同
  TileRegion，旧per-edge fusion recipe不再作为selection state。`RecursiveProducerTiling`只保留为已选region内部的低层temporal donor。
- 防复发：same-numbered transpose必须产生resident加peer fragments并下沉到DTE recv/send/wait；同Tile和disjoint reduction均覆盖
  exact demand，window覆盖一对多稠密像集，stride覆盖“不可用bounding box冒充exact fragment”，baseline继续断言零fusion。

## Direct-DTE局部endpoint顺序会造成FSM溢出或全卡wait环

- 现象：先发射一个Tile的全部recv再wait会超过有限receiver FSM；改成每个局部recv后立即wait后，多个Tile按各自SSA顺序
  又可能形成跨Tile循环等待。
- 根因：communication emitter自行用局部issue顺序和immediate await修补resource/deadlock，却没有统一的actual token lifetime、
  receiver FSM interval和card-scoped wait graph owner。把“立即等”写成协议虽然限制live recv为1，也会无条件丢失异步窗口。
- 修复模式：从explicit communication identity、round、payload slice、endpoint kind和peer建立稳定message/event关系；movement只发射
  matching token，storage/lifetime给出source/destination/relay release，schedule选择wait boundary并关闭plan-level 4-lane interval；actual memory/target gate在同一actual
  whole-card candidate上唯一验证dynamic matching、receiver冲突和wait graph无环。matching send issue与receive preparation保持独立，只有
  completion同时依赖两者。hard constraints确实要求时可以立即wait，但不能把它设为所有endpoint默认值。
- 防复发：多源fanin测试同时检查最大live recv不超过4、send/recv/wait dynamic exact配对、first-read/last-release、全卡无环和
  Direct-DTE binding；另有至少一个token-only issue window证明没有被emitter立即串行化。source-to-package transpose及attention baseline
  必须产生fresh no-card package，不能只检查Tile IR文本。

## Baseline显式op边界不能无条件作用于search coupled region

- 现象：为了让`none`逐op独立tiling而给每个consumer补DDR seal/reload后，search已选的coupled region也被同一逻辑
  强制切断，导致full weight reload、SPM溢出或最终零融合。
- 根因：candidate materialization没有区分independent baseline boundary与search selected actual group，把baseline策略写成了
  所有policy共享的结构改写。
- 修复模式：显式consumer boundary只在independent baseline生效；search从source SSA按selected node group直接构造共同region，
  以actual in-region use-def和emitted-node relation证明，不以edge action/proposal代签。

## Temporal order不能在remap或内部producer traversal中丢失

- 现象：domain中相同tile vector的两个`waveLoopOrder` point都存在，但经过Card preparation或selected-edge clone后actual IR仍按
  自然轴序；另一个缺陷会把DPS accumulator slice插到`scf.yield`之后，直到TileRegion lowering读取terminator才崩溃。
- 根因：新增typed field后只更新了producer，三个current-IR remap仍用旧的二字段aggregate构造；内部producer的parallel与
  reduction递归各自固定自然序。leaf materializer还沿用被TilingInterface改变后的builder insertion point创建accumulator。
- 修复模式：所有current-IR remap原位复制完整typed value；root与producer分别验证并消费active order，不能支持的coupled跨类
  nesting明确返回需要cut的失败。DPS accumulator与所有动态offset计算先把insertion point固定到tiled op之前，output insert固定在
  compute/fusion之后；prologue/steady/tail统一由一个loop owner构造。
- 防复发：测试必须比较同vector不同order的actual nesting，覆盖internal producer、multi-reduction、tail、partial contribution和
  clone/remap后的order；不能只比较domain key或loop数量。任何block conversion前先验证terminator与use-def，而不是等下游assert。

## Typed API存在不能证明算法已经迁移

- 现象：新实现增加Assignment、Domain、typed ID和direct apply后，旧placement/layout/movement/schedule owner及semantic tests被删除；
  轻量case仍可由first point通过，但cross-value constraint、PBQP reduction、movement elimination、NoC/alias proof或pipeline能力消失。
- 根因：把“旧local winner和clone owner必须退出”扩大成“其中算法无需迁移”，完成评审只检查新type、枚举或package成功，
  没有建立donor capability到current query、production caller和test witness的逐项对应。
- 修复模式：退役前列出旧owner的candidate construction、solver、proof、diagnostic和tests，逐项标成迁入current typed
  query/transition、由current IR明确淘汰或仍待处理。Layout等solver只返回typed proposal/bound，不clone/apply IR也不拥有winner；
  selected assignment只在policy-specific materializer中apply一次。
- 防复发：每项能力同时核对definition、direct consumer、production call graph、negative和actual downstream witness。
  “有domain”“测试绿”“package生成”“source未进CMake”都不能单独证明迁移完成；旧接口可以删除，未迁算法不能被改名为dormant。
## Partial reduction计算结果必须显式写回destination

- 现象：spatial partial contribution和最终merge的compute均存在，但TileRegion直接yield原output argument，partial SSA成为死值；只数
  region/emitted node的测试仍会通过。
- 根因：partial路径修改function result type后只调用`appendTileOutputDestinations`，误以为追加argument会自动建立store；complete
  root路径原本另有insert-slice，partial路径没有对应binding。
- 修复模式：所有full-result partial/merge路径在append后统一把returned tensor以全shape insert绑定到destination，再进入
  TensorProgram→TileRegion；movement relation从actual contribution writeback和merge input load建立，不从node名或参数序号猜。
- 防复发：partial gate必须直接检查每个contribution与merge output的actual `wafer.tile.store`，peer gather还要证明remote store/load被
  matched send/recv/await替代；region count、compute count和emission relation不能单独作为功能证明。
- 防复发：none source-to-package检查中间DDR boundary和single-root cardinality；search定向测试检查maximal/中间cut产生不同actual
  region、内部无DDR round-trip且fanout shared producer只有一个actual version。

## Baseline既不能复用search，也不能退化成fixed-assignment validator

- 现象：一种实现只在search loop前提前返回且报告零fusion，却仍复用search domain、group materializer和first-choice solver；
  另一种实现为避免耦合，只接受预选physical assignment并验证一次，正常上游IR遇到SPM超限时无人继续功能合法化。
- 根因：把“禁止性能候选选择”与“禁止确定性功能合法化”混淆，也把零fusion当成controller、region和resource owner隔离证明。
- 修复模式：baseline从未绑定physical choice的TensorProgram进入，始终只有一个live deterministic candidate。它直接构造
  per-root region、deterministic representation/movement/buffer/order/completion；同Tile多root为多个顺序region，跨root shaped
  dependency显式materialize。每个candidate actualize一次，只有带current owner relation的actual SPM capacity rejection可推进
  预定义、单调、不分支且不回溯的smaller temporal successor。Accepted owner直接move返回，不执行winner protocol、不重建，
  也不运行无consumer的schedule/duration或默认trace。
- 防复发：call graph证明baseline不include/call search domain、preparation或materializer；测试同时检查一root一region、跨root
  carrier、initial-overfull-to-fit、minimum仍超限typed failure、candidate/actual gate一一对应、accepted rematerialization和默认trace为0。
## Exact矩形恢复不能把generic Presburger等价证明放在baseline热路径

- 现象：即使单独观察一个logical shard pair query，CPU仍可长时间停在`getExactStaticRectangularImage`后的
  `PresburgerSet::isEqual/isSubsetOf/subtract`；变量数和disjunct数没有超现有limit，因此静态budget检查没有阻止该路径。
- 根因：supported projected/permuted/static-rectangle indexing semantics在relation composition后丢失closed-form witness，矩形
  recovery只好先求generic image、构造bounding rectangle，再调用通用集合等价证明。结构规模上限不能约束Presburger算法实际
  work，外层placement option-pair/CSP又会乘法放大同一查询。
- 修复模式：builder/composition在typed proof成立时保留或直接重建closed-form rectangular-image witness，single-coordinate
  baseline优先消费该witness；generic recovery在solver调用前检查constraint/local/coefficients等完整结构复杂度并记录
  query-local work accounting。超限是`ResourceExhausted`/indeterminate，不能当logical infeasible、不能推进fallback。
- 防复发：用轻量synthetic relation复现相同composition形态，断言supported路径generic equality调用数为零且pair-query数只随
  actual DAG edge和deterministic legalization step增长；重型模型只归后续显式scalability profile，不作为功能bug的常规复现器。

## Baseline不搜索通信方案不等于所有依赖零peer

- 现象：为落实逐op DDR baseline而把所有peer fragment禁止后，带reshape/transpose和多种spatial轴的Llama DAG在16-Tile
  placement CSP中立即无解；若继续宣称可由共享内部DDR替代，又与current ABI中workspace为Tile-scoped的事实冲突。
- 根因：把“baseline不枚举可选edge action/route”误写成“任何跨Tile correctness communication都不存在”。不同op的确定性
  spatial shard集合可能不一致，同一Tile的private workspace也不能冒充card-shared中间buffer。
- 修复模式：本地依赖使用compiler-owned DDR RegionCut；跨轴依赖只按typed indexing relation物化唯一required peer fragments，
  destination先在compiler-owned DDR assembly，再进入独立consumer stage。可选peer/retained/recompute/layout动作和route优化仍只由
  统一search拥有。
- 防复发：baseline回归同时覆盖本地multi-op chain的零peer和observable transpose的exact send/recv/wait；两者都必须16-Tile、
  buffer=1、actual fusion为零并通过package/no-card。

## 大型RegionCut链不能按edge重复扫描完整候选

- 现象：106-node/120-edge独立DDR baseline中，单Tile的`split-ddr-stages`约耗时66秒，16-Tile初次物化和每轮SPM反馈被该
  transformation反复放大。
- 根因：每个cut都重新遍历完整module找marker allocation，每个候选allocation又扫描整个region分类prefix/suffix uses，split还
  clone大型body；verifier对长DDR provenance链重复递归。
- 修复模式：一次遍历建立marker到cut及直接spill allocation索引，一次region walk分类全部eligible DDR allocation；按cut移动
  operation而不是clone整段body；verifier使用query-local provenance memo。不得把索引或memo写入IR。
- 防复发：长RegionCut chain测试验证最终IR和provenance，并在模型规模compile timing中检查单Tile split不再随edge数乘法增长。

## 16 Tile不能把root公共分析和整图clone机械重复16次

- 现象：structured roots较多时，旧baseline按root shard、Tile entry、完整CardModule三轮构造；仅root阶段就接近
  `root数 × 16 Tile`次TensorProgram conversion。即使16个worker并发，CPU work、RSS和诊断仍是同一工作被放大16倍。
- 根因：把Tile差异（offset/tail/peer endpoint）和root不变量（support relation、consumer access和structured identity）放在同一个
  per-Tile materializer里；materializer从output/edge endpoint无条件回溯SSA closure并clone scratch function，而不是从全部root
  execution domain一次性求exact operand demand。并发只缩短wall time，没有消除重复分析或过宽物化。
- 修复模式：每个candidate在immutable source上同时seed全部`(root, Tile)`，以`(value, Tile)`合并exact domain并按反向SSA拓扑传播；
  relation按operation/result/operand建立一次，structured producer立即形成boundary demand。carrier coverage验证后，final region只
  对当前candidate一次性物化typed demand recipe要求的operation和endpoint，不建立公共SSA closure或materialized-IR cache。16个Tile实际构造可
  bounded并发并按Tile ID稳定归并，但并发不是work消重机制。
- 防复发：显式test work counts检查relation construction、非空value/Tile demand、physical fragment和candidate Tile entry；每个
  candidate CardModule/actual memory/target gate各一次，winner不重建。测试必须包含16 Tile demand不同的fanin/fanout，证明不是16次完整DAG walk。
  Pre-structural frontier只共享immutable analysis，不能按candidate/Tile缓存actual IR。

## exact-empty producer不能被support graph重建重新拉入

- 现象：fresh FP16 LLaMA baseline中，一个consumer operand由两个structured producer经`extract_slice`/`insert_slice`组合；exact-demand analysis
  已证明其中一个producer对当前destination的demand为空，physical strategy没有为它生成fragment，但consumer侧递归重建support
  graph时仍遇到该producer并报“unassembled structured producer”。
- 根因：logical exact demand、root scope选择和support reconstruction由三条独立路径恢复。空需求只在per-edge carrier阶段被丢弃，
  无条件SSA closure和post-hoc rebuild不知道该证明，遂把本应停止的structured producer再次拉入。这不是缺少一个empty布尔字段，
  而是query/apply没有共享同一typed demand decomposition。
- 修复模式：从consumer root execution domain求operand demand；每个support operation返回同时供query/apply消费的typed transfer
  recipe。到structured producer停止并形成非空boundary demand；`insert_slice`按写入区域`W`把任意需求`D`精确分成
  `inverse(D ∩ W)`与`D − W`。全部boundary按ownership all-and-only绑定后才一次性构造final single-root region。
- 防复发：穷举tiny shape的insert overwrite需求子集，并覆盖multi-producer、empty branch、Peer/RegionCut混合与模型级baseline。
  删除无条件closure、support clone/rebuild/replay；unsupported relation在mutation前typed fail closed，不能退回复制全部operand。

## 读取源码marker的测试会把退役实现伪装成合同

- 现象：source或test没有进入active CMake/lit/CTest执行图，但一个已注册测试逐文件读取其文本并断言旧symbol marker存在，整体仍显示green。
- 根因：把“仓库里还有某段源码”当成“能力已被编译并经过行为验证”，source inventory又只检查已知子集，形成互相放行的假闭环。
- 修复模式：组织检查以filesystem、CMake source、unit/lit/CTest registration和明确的current-task dormant owner做双向集合闭合；
  能力测试只验证编译后的接口和行为。退役实现独有能力先逐项列出proof、materializer、diagnostic和negative witness，迁入active
  owner并受测，再删除源码和marker断言；“未注册”只能证明当前没有执行，不能证明源码没有独有能力。attention normalization曾发现未注册的
  attention alternative仍独有current-SSA资格证明和online/split actual-root构造，正确顺序是迁入`Planning/Search`后再删除旧源。
- 防复发：新增源码或测试时，checker fixture分别覆盖unregistered source、unregistered test、stale registration和无owner dormant
  四类negative；禁止用源码文本marker作为build/behavior contract。

## logical placement legality不能由physical carrier或未分类失败代签

- 现象：exact `IndexRelation.image()`与ownership coverage query已经存在，但placement evaluator随即把结果降成dense
  fragment、layout、route和resource calendar；任一后续表达失败都被压成`bool legal=false`。同时DPS init和经过pure support
  graph的dependency被以“稍后lowering会处理”为由省略，使显式init producer无法形成跨root demand。
- 根因：logical relation/ownership proof与physical representation/movement选择共用candidate/schedule类型和失败通道；query又从
  shard dimension与participant count恢复balanced一维矩形，导致当前实现限制反向定义上游合法域。`FailureOr + string`无法
  区分proven logical contradiction、semantic unsupported、资源/内部indeterminate和physical carrier failure。
- 修复模式：以完整logical iteration/result/ownership domain、typed data/init/support relation、reduction/replication role和
  IR epoch作为policy-free query输入；exact set与ownership intersection原样保留，返回`satisfied`、带direct witness的
  `proven logical infeasible`、`unsupported semantic relation`或`indeterminate/compiler failure`。只有proven logical failure
  可以删除placement trial；layout/descriptor/route失败只拒绝对应physical assignment。显式init root保留dependency，非root
  support graph组合typed relation；reduction、broadcast、window/stride和multi-piece都不能先densify。
- 防复发：logical-demand单测与edge-strategy/materializer测试分离，并增加carrier metamorphic test；改变dense/strided/
  multi-piece carrier或route可用性时logical outcome必须不变。cache观察IR epoch和完整semantic assignment，禁止把字符串失败
  压成legality bool，也禁止analysis header反向依赖candidate schedule carrier。任务状态必须核对production调用链，不能只凭
  analysis单测标done。

## 长度一的injective embedding successor也必须先处理已完成suffix

- 现象：zero-rank/scalar spatial node从第一个Tile推进到第二个Tile时，`SmallVector`越界并触发assert；rank大于零的常用case未暴露。
- 根因：k-permutation successor更新最后一个logical cell后，仍进入“填充后缀”循环并写`embedding[size()]`，遗漏了
  `position + 1 == embedding.size()`的终止分支。
- 修复模式：更新当前位置后先判断suffix是否已经完整；完整就直接返回success，只有确有后缀时才按未使用Tile填充。
- 防复发：scalar/zero-iterator domain必须枚举每个available singleton Tile并正常到达end；tiny reference同时覆盖长度1和多cell
  embedding，禁止只测首点或rank大于零的常见向量。

## 编译边界不能把typed allocator failure压成一个布尔值

- 现象：CardModule编译入口只看到“SPM allocation failed”，会把unsupported lifetime误归为内部失败，或反过来把未分类的
  allocator failure误当作candidate非法并从搜索域删除。
- 根因：Tile memory planning跨边界时丢失了`SPMMemoryPlanningFailureKind`，上层只能从诊断文本或capacity布尔量猜taxonomy。
- 修复模式：memory-planning failure保留typed SPM failure kind；capacity overflow与unsupported lifetime作为可验证exact rejection，
  resource exhaustion、未分类allocator/internal failure保持indeterminate。组装结果时先复制primary gate/detail，再move failure
  容器；不能依赖函数实参求值顺序同时引用元素和转移其owner。
- 防复发：无策略CardExecutable seam直接测试同一CardModule的可重复exact rejection，并单测不完整/内部调用保持indeterminate；
  caller遇到indeterminate必须终止当前编译，不能生成no-good或repair candidate。

## 跨region替换后保留旧Value relation会造成悬空引用

- 现象：region cut、suffix region重建或function output-destination argument删除后，current-IR relation仍保存旧`Value`；后续
  verifier/capacity feedback解引用时可能SIGSEGV，或者把合法候选误判为relation缺失。
- 根因：IR rewrite完成了SSA替换，但编译器侧typed relation没有沿同一`IRMapping`/replacement map重绑；仅检查pointer非空
  不能证明handle仍属于current IR。
- 修复模式：所有已知replacement在mutation transaction内显式retarget；cleanup结束用opaque `Value` live-set删除dead relation，
  不解引用可能失效的handle，也不推断新关系。required relation缺失仍按typed failure fail closed。
- 防复发：测试覆盖region replacement、function argument erase、dead cleanup与current-relation verifier；ASan/普通构建都不得
  依赖地址仍可读的偶然性。

## Canonicalization删除dead buffer后旧relation不等于编译失败

- 现象：bufferization/canonicalization合法删除无user的temporary buffer，但capacity gate在cleanup后仍要求其旧relation存在，
  将可接受candidate标成indeterminate。
- 根因：query evidence的lifetime跨越了会删除IR的cleanup，却没有在cleanup结束时按current IR收缩。
- 修复模式：cleanup完成后先以live `Value`集合retain current relations，再执行required-witness assertion；只删除dead evidence，
  不把dead entry重定向到同类型buffer。
- 防复发：构造dead relation负例，断言retain后current检查通过；真实required relation被删时仍必须失败。

## Function-boundary bufferization必须早于TileRegion relation listener

- 现象：selected multi-root Region在Tensor/TileRegion阶段通过verifier，但进入actual memory/target gate后报structured buffer relation不属于current IR；
  单root canonical路径可能因bufferization形状恰好稳定而掩盖问题。
- 根因：TileRegion-to-Instr先用listener把relation重绑到Instr SSA，随后function-boundary One-Shot Bufferize又改写function参数和
  boundary SSA；后一个pass不使用该listener，已重绑的relation再次失效。
- 修复模式：在movement和execution structure前，由layout stage一次完成function-boundary与region-local bufferization，并把relation
  重绑到唯一current storage root；cleanup结束显式删除只对应已消失dead SSA的attribution entry并检查其余relation仍属于current module。
  之后
  TileRegion-to-Instr的所有replacement只由同一个listener跟踪。Current-Instr stage随后完成worker/order和completion；
  memory/target leaf在已有materialization relations的production路径不得再次运行bufferization或重建completion，缺失时直接拒绝。
- 防复发：真实规模selected multi-root stored/direct与replica候选必须走完整CardModule→Instr→actual SPM gate；只验证TileRegion
  或单root路径不能签发relation epoch正确性。

## PeerFragments的代表source不能代替逐fragment ownership

- 现象：receive-only Tile被layout gate要求物化producer layout，导致已合法actualized的布局候选被错误拒绝。
- 根因：gate读取strategy级`sourceTile`，但PeerFragments真正的producer分布在每个fragment的source endpoint；representative字段
  不是ownership proof。
- 修复模式：PeerFragments逐fragment检查当前Tile是否materialize producer；非fragment策略才使用strategy级source。
- 防复发：覆盖receive-only Tile、multi-source fragments和非fragment策略，并在diagnostic中输出expected layout、structured node与Tile。

## Repo内同步接口不应各自递增版本

- 现象：frontend nested metadata、profile plan/site、dependency record、workload和内部算法名称分别携带版本，修改同一语义时
  需要跨多层同步数值并保留旧分支。
- 根因：把源码revision内同步演进的内部表示误当成独立兼容边界，用版本号代替exact field contract和集中parser检查。
- 修复模式：先列出真实producer、consumer、存储和部署生命周期；独立文件或ABI保留一个current identity与集中检查入口，
  其余表示原位修改并同批更新所有调用方，旧输入在唯一边界入口fail closed。
- 防复发：新增版本前必须写明独立producer/consumer、支持周期和兼容测试；内部field、算法、hash domain、model和helper
  metadata不得使用`vN`名称或双reader。

## Board probe helper签名变化必须覆盖所有runner

- 现象：共享package helper改为返回`module_path, resource_ids, slots_per_tile`后，常用raw probe已迁移，但NE tail和engine pipeline
  等未注册runner仍按两个返回值解包，直到重新接入no-card才失败。
- 根因：Board source/catalog存在，但未作为current CTest执行；helper调用方清单与CMake执行入口没有一起更新。
- 修复模式：接口变化先用`rg`枚举全部调用方并同批迁移；每个可直接生成current package的raw probe至少注册一个代表性
  `current-interface` no-card CTest，catalog只拥有case语义，CMake只拥有执行入口。
- 防复发：fresh CTest精确运行`-L current-interface`，并验证inventory列出的host/no-card/Board CTest都在CMake中真实注册；
  不用未注册的CTest形状字符串代替执行证据。

## 旧Board executor删除前必须迁移观测合同

- 现象：旧executor因manifest、CLI或SPMD carrier退役而不能运行，直接删文件会同时丢掉model-scale source、full-output oracle、
  status/guard、重复完成或profile采集要求。
- 根因：把“执行接口已退役”误当成“校准问题已无价值”，没有按source、lowering、package、runtime和board边界拆解。
- 修复模式：能经current global lowering生成package的case迁入current no-card/Board CTest；不能生成的case保留current source、
  deterministic oracle和完整待执行要求，并在catalog/inventory绑定具体blocker。只有这些内容已有current owner后才删除旧executor。
- 防复发：逐个核对删除文件的source、shape/dtype、payload、numeric oracle、structural checks、status、timeout、cleanup和profile要求；
  inventory测试必须能从每个保留case解析到真实source/catalog与CMake入口或明确blocker。

## StringRef视图必须绑定在SmallString最后一次修改之后

- 现象：payload resolver持有指向`SmallString tensorProgram`的StringRef，构造时该path尚未append
  "tensor-program"子目录；后续append触发缓冲重分配，resolver仍指向旧缓冲，stat出transactionRoot而非
  tensor-program目录，constants存在性检查误报MissingPayload且找不到新物化的文件。
- 根因：可修改的SmallString在视图（StringRef成员）建立之后继续append；SmallString内联缓冲与堆缓冲的
  重分配时机不透明，旧缓冲内容残留使错误表现为"路径少一段"而非崩溃。
- 修复模式：指向SmallString/String的StringRef成员必须在最后一次修改之后构造；需要跨阶段复用的路径
  用std::string拥有并只取视图，或用值传递的std::string成员。
- 防复发：新代码里任何`StringRef member`绑定本地SmallString时，先确认绑定后该SmallString不再被append/
  resize；评审时对"resolver持有路径引用"这类长生命周期视图重点检查。

## Move-only结果不能引用先于结果销毁的staging文件

- 现象：compiler成功返回`CardExecutable`，但返回后target consumer首次读取parameter就报文件不存在；同时byte-identical
  helper shard虽未进入range identity，重复文件仍随结果存活，规模账本却显示open次数恒定。
- 根因：handoff只保存transaction root下的path，外层scope cleanup在public compile返回时先删除root；source每次range/digest
  再按path打开，真实open/read没有进入只统计establishment的账本；未adopt candidate也没有明确的最后consumer边界。
- 修复模式：让结果类型自己RAII拥有稳定parent下的唯一目录和move-safe read handle；owned bytes成为header/extent/digest最终事实源；
  最低层强制read window并记录actual open/window/bytes/max；最后一次verification后adopt真实source并销毁全部剩余candidate。
- 防复发：直接测试move后删除staging仍能读取且析构清理文件；Card边界断言candidate归零；原地改写fixture明确超过pinned mmap
  threshold；大range断言window数量/bytes、最大window与零新增open；source-to-package规模账本分开验证identity计数和实际I/O work。

## C++17 std::variant比较要求所有alternative同时定义==和!=

- 现象：为6个`TileEntryArgument` payload struct定义friend `operator==`后，`std::variant`比较、`TileEntryArgument`
  aggregate和manifest序列化仍编译失败，报"no match for operator!=（no known conversion）"。
- 根因：C++17的`std::variant operator==/!=`在libstdc++实现中要求每个alternative类型同时有`==`和`!=`；只写`==`并依赖
  C++20的rewritten candidate在C++17不存在。
- 修复模式：closed union的每个payload struct成对定义friend `operator==`和`operator!=`；同类新字段加入时同步两个运算符。
- 防复发：新增进入`std::variant`的payload类型时，先写编译级小测试确认比较完整；不能假设`==`隐含`!=`。

## llvm::ArrayRef模板推导不接受隐式转换

- 现象：`llvm::ArrayRef<int64_t>`与`std::vector<int64_t>`直接`==`比较编译失败，报template argument deduction
  substitution失败。
- 根因：模板实参推导发生在重载决议前，`std::vector`→`ArrayRef`的隐式构造函数不参与推导。
- 修复模式：任一侧显式构造`llvm::ArrayRef<int64_t>(vector)`后再比较；长期语义边界改用typed容器或循环比较。
- 防复发：ArrayRef与STL容器混用时，接口签名优先显式ArrayRef参数，调用侧避免依赖推导转换。

## 结构体持有悬空视图：unique_ptr容器移入owner而不是引用字段

- 现象：`PackageAssembly`先以局部`std::vector<std::unique_ptr<TargetTensorJoin>>`构建`placement`裸指针视图，
  函数返回后manifest打印出垃圾值（diagnostic显示的bytes/offset随机）。
- 根因：视图指针绑定在栈上容器，容器析构后指针悬空；aggregate没有成员所有权。
- 修复模式：把`unique_ptr`容器本身移入`PackageAssembly`作为成员，视图字段只指向成员容器元素；destructor顺序由成员
  声明顺序保证。
- 防复发：任何返回结构体只要含"视图指针"字段，先确认被视图对象本身由同一结构体拥有；禁止栈容器+outlived view模式。

## 非默认gate的lit期望静默过时

- 现象：Tools lit（不在默认CTest路径）一次出现8个失败，现象各异：`spm_planning_invocations`16→24、
  timing表stage改名、`emitOpError`输出丢失`[0]`、wafer-opt pipeline要求explicit mesh shape、XLA helper
  INVALID_ARGUMENT、`buildCardExecutable`多出`ProgramDataHandoff&`形参后SystemC树编译失败、以及
  `StructuredDAGPlacementEnumerationTest.MultiOutputFanout...`单测100% CPU死循环。
- 根因：这些测试不属于默认lit/ctest路径，repeated physical-dataflow refactors多次refactor后无人刷新期望；真实行为变化与测试更新在不同commit，
  且部分期望（如`emitOpError`带operand、旧package source copy）对应的是已经退役的表示。
- 修复模式：逐项确认行为变化commit与测试最后touch的先后（`git merge-base --is-ancestor`），结合pinned LLVM/MLIR源码
  判断哪个是current合同；当前修改只修自己造成的失败和本批已触及文件的陈旧期望，其余登记为独立后续任务。
- 防复发：改production行为时搜索所有test目录（含非默认gate），同步更新期望；"该目录不在CI"不能作为让测试红的理由；
  未知枚举case先用bounded小范围filter定位；已经定性为旧实现状态爆炸的case直接从当前批次排除，不再用长时间单跑确认。

## 全量单测中的单个枚举case可能耗尽CPU

- 现象：聚合测试长时间无输出且一个或多个进程持续100% CPU；单独列举case后可定位到placement/search枚举fixture。
- 根因：test domain没有明确状态规模，逐节点extension、pairwise topology query或Cartesian state使工作指数增长；stdout缓冲又让
  整批看起来像hang。旧case未进入默认gate时更容易长期失察。
- 修复模式：先用test listing、filter和timeout定位具体case，再以更小同构fixture、独立reference enumerator或显式状态上界保留
  semantic witness；已退役枚举实现和only-purpose tests随owner删除，不反复延长timeout。
- 防复发：新增枚举测试声明domain size或预期state数，真实规模production coverage与tiny exhaustive oracle分开；
  聚合suite运行前确认filter实际匹配并报告被排除的已知重型case。
## Public signature变更必须重编所有active配置

- 现象：主configured build编译通过，另一个feature-on、model、runtime或board配置仍用旧参数调用同一public API。
- 根因：多个build tree共享source；只增量构建当前主树不会触达其它配置独有的translation units。
- 修复模式：修改public header、函数签名或跨library返回类型后，从current CMake/test ownership列出全部active consumer配置，
  分别重新configure或fresh build受影响targets，并检查public header/link smoke。
- 防复发：完成证据记录实际重编的配置和consumer，不写死历史build目录；一个tree通过不能代签其它feature组合。
## StringRef::slice第二个参数是exclusive end不是length

- 现象：program-data canonical verification的零padding检查在负例上静默通过——96字节文件中[80,96)非零，
  检查却返回"全零"；该检查写成`content.slice(offset + checked, chunk)`。
- 根因：本仓库pinned LLVM的`llvm::StringRef::slice(Start, End)`第二参数是exclusive end并在内部clamp；
  按length传入时`End < Start`被clamp成空区间，`find_first_not_of`对空串恒为npos，检查退化为恒真。
- 修复模式：slice调用写成`slice(start, start + length)`；涉及区间的验证一律配能直接触发原缺口的负例测试，
  不能只靠正例通过。
- 防复发：新写StringRef区间逻辑时对照pinned header确认slice/substr参数语义；review零值/空区间退化路径。

## Blocked layout的logical row不是bounded physical window

- 现象：window packer按logical C row取值，用该row首尾physical offset构造一个contiguous span；Cx`{2,128}`第一行覆盖
  `[0,192)`，第二行却从64开始，writer报window overlap。单outer-row小fixture会掩盖该问题，且超大C row还绕过byte budget。
- 根因：Cx/NCx是block-major physical order；logical row-major traversal在多个outer element、full block、tail和bank padding之间
  不保持physical offset单调或连续。把logical边界当physical边界属于表示层混淆。
- 修复模式：bounded encoder遍历disjoint contiguous physical-element windows，再从shared physical geometry反算每个位置的
  optional logical index；padding没有owner。byte与element budget均为硬上限，BOOL窗口保持byte alignment；caller只对当前窗口的
  logical indices排序并读取连续source runs。
- 防复发：window输出必须逐字节等于full codec，并覆盖多outer-row Cx/NCx、full block、tail、bank padding和bitpacked BOOL；
  测试同时断言每个window不超过budget、offset连续且logical value all-and-only covered。

## Full-transfer cleanup必须在mutation前验证replacement consumer类型

- 现象：exact byte/map proof允许NCx source替代Tensor destination，但destination的后继是`memref.reinterpret_cast`等standard view；
  cleanup改写operand后才由verifier发现source/result memory attr不一致，production candidate已经被破坏。
- 根因：只验证transfer与storage lifetime，没有验证每个actual consumer是否允许replacement type改变；test-only compute/load consumer
  恰好宽松，未覆盖standard view链。
- 修复模式：收集全部待替换use并在第一次mutation前分类。Replacement type不变可正常替换；type改变只允许合同明确接受该Wafer
  memref的Instr或普通memref load/store，view、select、region/call boundary及其它typed relation consumer保留transfer。成功replacement
  同步retarget caller-owned current relations，cleanup后fresh verify。
- 防复发：rank-changing NCx→Tensor view链必须保留，cross-encoding direct compute/load正例仍可删除；production baseline与独立kernel
  正负矩阵同时执行，不能只看eliminator fixture。

## 同storage width不能代签dtype数值转换

- 现象：F16 1.0转换到BF16时直接把`0x3c00`塞进BF16，结果仍是`0x3c00`而非`0x3f80`；因为两者都是16 bit，
  size/count/codec roundtrip均可能通过。
- 根因：代码只比较storage width/category，并调用目标format的raw-value构造器清padding；该操作验证encoding宽度但不执行数值语义。
- 修复模式：source/target format不同就解析current target conversion route并调用formal numeric conversion；rounding参数显式选择
  deterministic nearest-even，缺route、未实现route或缺zero-point等语义参数时fail closed。identity只能是同format的raw copy。
- 防复发：转换测试必须选择同宽但不同encoding的已知值并断言exact target bits；review中看到`RawLogicalValue{target, source.bits}`或
  `makeRawLogicalValue(target, source.bits, ...)`应默认视为bitcast，除非接口明确命名并验证bitcast语义。

## Strict manifest不能只相信自洽的bytes字段和文件digest

- 现象：manifest可把F32 Tensor`shape={4}`写成`bytes=1`，只要offset、total bytes、文件大小和digest一起修改，旧verifier就接受；
  尾随全零字节和`modules/`下额外空目录也能穿过closure。
- 根因：verifier只检查字段之间自洽，没有从dtype/layout/shape重算physical storage，也把“零padding”扩展到最后一个range之后；
  文件closure只枚举regular payload，忽略目录拓扑。
- 修复模式：strict verifier经同一`NumericTensorKey`/physical codec重算TargetTensor及external port bytes并核对logical/target
  element count；program-data必须精确结束于最后range，尾随字节无论内容都拒绝；目录closure由declared file paths推导全部必要祖先，
  其它目录/文件/symlink一律拒绝。
- 防复发：负例要协同更新size/digest使输入保持表面自洽，分别覆盖wrong codec bytes、logical/target count mismatch、zero/nonzero
  trailing bytes和额外空目录；只篡改digest的测试不能证明semantic verifier有效。

## CMake多OUTPUT custom command不能靠第二个同OUTPUT命令补文件

- 现象：resources自定义命令把linker script和SPMD helper symlink都列进第一个`add_custom_command`的OUTPUT，再为helper单独
  建一个同OUTPUT的命令；ninja只生成第一条rule，helper资源目录存在但symlink从不创建，编译在target阶段才以
  `device link failed`暴露。
- 根因：同一OUTPUT出现两次时CMake只保留第一条custom command，不报错也不合并；目标级`add_custom_target`依赖的是
  OUTPUT名，无法区分。
- 修复模式：一个custom command的OUTPUT必须是它实际产出的全部文件；不同产物分属不同command，各自带真实DEPENDS；
  reconfigure后直接`ninja -t targets`核对产物rule存在。
- 防复发：新增build-tree/install资源时，先检查目标输出文件是否真的生成（`ls`/`test -f`），再测消费它的编译路径。

## pinned LLVM版本的API事实

- `llvm::errc`没有`state_not_recoverable`：内部不变量错误用`llvm::errc::operation_not_permitted`；
  `llvm::sys::path::append`最多接受path+3个组件，超过必须分两步append；`llvm::sys::path::join`不存在。
  编译期宏路径烘焙会在install后失效，外部工具/资源一律运行时发现。
- `PresburgerSet`没有默认构造函数（只有space/move构造）：带`PresburgerSet`成员的结构体用
  `std::optional<PresburgerSet>`字段（`IndexSetResult`同款），默认构造的set语义用"absent"表达，
  不用裸成员+聚合初始化碰运气。
- tensor方言接口归属：`tensor.insert_slice`原生实现`DestinationStyleOpInterface`但**不**声明
  `TilingInterface`；`tensor.pad/pack/unpack`的`TilingInterface`走external model，未注册该model的
  context里`isa<TilingInterface>(pad)`直接fatal（"promised by dialect but never implemented"）。
  判定DAG节点（DPS && Tiling）时必须DPS在前短路，不能让Tiling先查；生产与测试都要注册
  `mlir::tensor::registerTilingInterfaceExternalModels`（声明在`TensorTilingInterfaceImpl.h`，需要显式include）。
- pinned的`linalg.generic`汇编要求显式`} -> tensor<...>`结果类型；省略时op解析为零结果，
  报"cannot name an operation with no results"（该报错指被命名的op无结果）。
- `tensor::ExtractSliceOp::getMixedOffsets/getMixedStrides`、`InsertSliceOp::getMixedOffsets`、
  `PadOp::getMixedLowPad`返回`SmallVector<OpFoldResult>`（动态值不保证常量），常量用
  `mlir::getConstantIntValue`解析后走`staticSlice`/`staticInsertSlice`。

## Move-only path和digest不等于文件资源ownership

- 现象：compiler返回的package对象被称为move-only owner，但字段只有root path、typed manifest和digest；strict readback临时打开的
  module/program-data buffer在返回前已经销毁。对象仍可存活时，磁盘成员却能被删除或替换，后续consumer只能重新按path打开。
- 根因：把不可复制的value identity误当成resource control。`move-only`只约束C++对象复制，digest只证明某次读取的内容；二者都不延长
  file descriptor、mapping或immutable storage的lifetime。
- 修复模式：若类型合同声明拥有文件内容，就用RAII直接持有all-and-only不可变snapshot；只持有fd或file-backed只读mmap仍会
  观察同inode原地改写，不能冒充content ownership。明确descriptor关闭和snapshot析构顺序；
  只需要瞬时验证时则把类型命名和API收窄为verified manifest/snapshot，不得称为package owner。
- 防复发：owner测试必须在返回后覆盖同inode、删除/替换原path，并继续从owner读取已签发内容；还要检查descriptor不泄漏。
  只断言move trait、root字符串和digest相等不能证明ownership。

## 局部capacity probe与complete-candidate gate会产生不同scope和witness

- 现象：baseline先用root/region/function级scratch IR判断SPM，再重新构造完整CardModule；局部路径可能报告fit或要求扩大scope，
  最终actual memory/target gate却在函数级packing失败。relation remap还可能在两次构造间省略或保留不同owner，使同一allocation得到不同归因。
- 根因：把“调用相同pass/checker”误当成消费同一actual IR和同一current relation certificate。只要第一次IR被销毁、第二次重建，
  operation lifetime、buffer relation、region boundary和packing scope就已经是两个事实源；继续增加scope escalation只会扩张平行链。
- 修复模式：删除baseline的root/Tile/function局部capacity probe。每个进入memory/target gate的candidate只产生一份完整CardModule；actual
  planner demand通过candidate transaction的result/operand/output/movement/scratch relations归因，缺owner就是contract failure，不能按
  region恰有一个root猜owner。
- 防复发：显式test work counts证明每个完整物理choice/IR epoch至多生成一份actual-gate CardModule，且
  `actualGateCandidates == actualGateInvocations`、accepted rematerialization为零；测试扰动
  每类relation并检查所有actual SPM demands有owner，不以Location、空relation或diagnostic字符串证明一致性。

## 发布点之后不能再运行会翻转事务结果的validation

- 现象：staging目录先rename到最终路径，随后installed-root readback失败并返回错误；cleanup只覆盖staging，最终目标仍可见。
  profile用两个顺序rename时还会短暂或在进程退出后永久暴露ordinary-only状态。
- 根因：把单次rename的原子性扩张成整个多产品事务的原子性，并假设post-commit readback“只会因compiler bug失败”。任何真实I/O、
  并发mutation和内部不变量错误都仍是可达错误边，两个各自原子的rename也不是共同原子commit。
- 修复模式：所有可失败validation和identity binding在唯一visibility point之前完成，全部共同产品位于一个可单次发布的owner/root；
  若协议确需installed-path动作，显式建模committing/committed状态和可恢复操作，并证明每个错误出口不泄漏本轮目标。
- 防复发：failure injection必须覆盖最后一次发布前后、installed readback和进程/第二产品边界；同时断言返回status、ordinary/profile
  可见性和staging残留，不能只测第二次rename正常返回失败时的best-effort rollback。

## Per-strategy whole-root walk会把materialization验证放大成高阶工作

- 现象：chain和GEMM很快，但16-Tile transpose/fanout在materialization verification中长期占满CPU；采样栈持续位于
  root membership、buffer relation和storage-root递归，说明是高阶重复工作而非单点死循环。
- 根因：对每个strategy、operation和buffer relation从零遍历完整Tile root并重建visited set，fragment数量、Tile数和IR
  膨胀相乘。非默认重型gate长期未执行又掩盖了回归。
- 修复模式：先用stage timing和中断采样定位首次重复walk，再把同一immutable IR epoch的storage-root或relation查询改为
  request-local memo/反向索引；memo value必须有稳定owner，不能返回会因map/vector扩容悬空的引用。更优先的是删除上游
  重复materialization，不能仅用cache掩盖错误construction。
- 防复发：真实规模fanout/transpose记录relation construction、非空fragment、root walk和candidate次数；受影响非默认gate
  必须实际执行并带合理timeout。
## operation count不能作为IR mutation snapshot

- 现象：analysis/query缓存借入一个FuncOp后只记录顶层operation数量；原位修改nested op的attribute、operand或result type时，
  operation数量和root指针均不变，旧relation cache仍被接受。
- 根因：结构元素数量不是IR语义identity；MLIR rewrite可以在不增删operation的情况下改变legality、index relation和lowering。
- 修复模式：需要在不可控mutation边界外fail closed时，使用pinned MLIR的nested `OperationFingerPrint`观察operation identity/
  nesting、attributes/properties、blocks、operands、successors和result types；borrow token只表达共同lifetime，不再维护第二份
  process-global generation。若owner能控制全部mutation，优先由明确的analysis invalidation边界重建query。
- 防复发：invalidation测试必须修改nested semantic attribute、operand或type且保持operation count不变；只测试插入/删除op不能
  证明snapshot完整。

## projected affine fast path必须保留relation两侧和中间domain bounds

- 现象：projected-permutation relation的矩形image/preimage fast path只检查destination bounds，source较小时返回越界矩形；两个
  各自有界的projection compose后直接传播pattern，还会绕过intermediate domain clipping。
- 根因：projection pattern只描述坐标等式，不包含bounded Presburger relation的source、destination和composition中间域；把pattern
  当成完整relation会扩大exact set。
- 修复模式：fast path同时保存并检查source/destination shape，任何边界可能裁剪时回退generic Presburger证明；composition默认丢弃
  pattern，只有builder对完整bounded relation完成等价证明后才能恢复。测试同时比较越界image、反向preimage和bounded compose。
- 防复发：每个关系fast path都必须说明它保留了哪些domain constraints；不能只用in-bounds identity/permutation正例证明exactness。

## support-chain carrier不能从balanced producer rectangle正向重建exact demand

- 现象：logical query已精确支持strided view和`insert_slice` overwrite，但baseline carrier仍逐个正向映射balanced producer
  shard并要求每个image都是非空单矩形。stride未选中的source shard或overwrite掉的destination因此被误报为relation失败；
  multi-piece relation也会被迫构造成bounding rectangle并随即被carrier拒绝。
- 根因：physical compatibility路径重复恢复了一份logical relation，并把“该edge对某个destination无贡献”和“当前dense
  descriptor不能表达”混成placement legality。它既绕过query的per-destination exact set，也丢失empty set的合法语义。
- 修复模式：在同一immutable IR borrow上复用typed exact-demand结果；physical carrier只消费per-destination producer demand和
  ownership intersections。empty demand显式表示无physical action；非空集合按可证明all-and-only的有限fragment分解，无法表达时
  只拒绝该physical assignment，不改写logical verdict。
- 防复发：production gate至少包含一个strided support view和一个overwrite产生empty destinations的multi-piece relation，并证明
  它们进入完整CardModule/CardExecutable gate；只测whole-edge query或只用连续concat piece不能覆盖这类重复恢复缺陷。

## Pipeline合同切换必须同步非默认测试

- 现象：默认suite全绿，但Tools/Runtime等非默认测试仍使用旧partition domain、缺失的新required option、过期diagnostic、
  旧manifest field或target不支持的dtype；后来集中运行时出现多个看似无关失败。
- 根因：接口迁移只更新了默认注册面，没有从CMake/lit/CTest事实枚举全部consumer；旧fixture成功条件还混入已退役
  search、schema或target语义。
- 修复模式：改变SPMD、mesh、source、package或dtype边界时，同批查找所有CLI、field、diagnostic和fixture consumer，
  按current contract更新或删除only-for-old-path测试。target明确不支持的组合由verifier拒绝，普通纵向使用current默认dtype。
- 防复发：source organization gate同时比较filesystem、CMake和test registration；收尾明确列出实际执行的default与non-default
  suites及skip/unsupported。外部helper产生异常前先验证输入是否满足current contract。
## 给未进入active target的源码插桩不会改变二进制

- 现象：修改或插桩源码后构建显示无工作，二进制中没有新symbol/string，运行行为也不变。
- 根因：文件只在`LLVM_OPTIONAL_SOURCES`、历史目录或未被目标消费的source列表中，不属于active build graph；同名能力已由
  其它current实现拥有。
- 修复模式：修改前从current CMake target和构建系统query确认source→object→library/executable闭包；再阅读真正active定义和直接consumer。
- 防复发：source organization checker同时核对filesystem、CMake与tests。"no work to do"且symbol不在binary时先检查build registration，
  不继续向dormant source追加补丁；但未注册也不能直接证明其中没有待迁移能力。


## Block splice必须维持terminator、SSA与relation有效性

- 现象：合并entry或region时先删除terminator，再使用`without_terminator()`移动operations，最后一个真实operation被误当作
  terminator排除并随source block销毁；relation随后指向悬空或已复用的SSA对象。
- 根因：MLIR block helper的前置条件在mutation中途被破坏，同时只移动IR，没有同步current relation和isolated region argument。
- 修复模式：在每一步保持block结构合法：先完成RAUW和typed relation retarget，再移动明确的operation range，最后处理terminator
  与source owner。跨`IsolatedFromAbove`边界的value必须变成region operand/block argument，不能直接捕获entry-level SSA。
- 防复发：测试覆盖last-op movement、空/单op block、region argument、multiple relation append和conversion replacement；
  mutation后立即验证block/region与current relation，不从打印文本或地址恢复对应。
## 证明型 relation fast path 的元数据不能强于关系本身

- **single-valued 不等于 total**：affine map按构造是函数，只证明同一输入至多一个输出；bounded source会裁剪其
  destination domain。跳过access relation的domain equality或range containment前，metadata必须分别证明完整domain coverage、
  range coverage和所需exactness，不能用一个`functionalByConstruction`布尔量同时代替。
- **restriction/composition会使旧矩形证明失效**：对destination/source求交后，projected-rectangle pattern只有在重新证明
  restricted relation等价时才能保留；否则exact image/preimage必须回退到受限关系本身。测试要查询被裁掉区域，而不只检查
  in-bounds identity正例。
- **complete reduction必须证明source coverage**：constant destination map只表示所有iteration落到同一destination，不能证明
  iteration-to-source覆盖整个source。shortcut还要检查constant坐标值、对应extent、维度唯一性和完整source image；用非零
  constant、duplicate dim和只访问source一条slice的反例防复发。
- **成功路径不得遗留debug stderr**：fast path/fallback命中计数进入显式诊断计数或测试hook，不能用无条件`llvm::errs()`作为
  长期观测；全绿测试仍打印debug不是完成状态。

## scratch IR 的 SSA handle 不能逃出 transformation lifetime

- isolated transformation返回后scratch module/func/region随RAII scope销毁；结果结构若保存其中的`mlir::Value`，即使当前caller暂时只读
  同结构中的bytes/type字段，public contract也已经包含悬空handle。
- 边界只返回可独立存活的typed evidence：稳定structured node identity、复制后的type/shape/bytes、relation role及必要的
  semantic coordinate。需要在scratch内追踪SSA时只在scope内消费并转换，不能把地址或`Value`留给controller。
- 新增semantic flag或“narrow”入口时，测试必须检查flag有真实consumer以及actual IR的all-and-only identity/cardinality；
  `>= 1`、diagnostic缺失或仅证明target存在，都无法证明sibling没有被物化。

## DDR stage 不能先在 SPM 拼完整 spatial shard再复制到 DDR

- 现象：temporal tile已从大wave持续缩小，actual Tile entry仍保留两个完整`memref<256x4096xf16>` SPM allocation；
  `NoneJointlyRefinesExplicitProducerStageAndConsumerDemand`长期不收敛，缩temporal coordinate对峰值容量基本无效。
- 根因：independent consumer stage先用`getOrMaterializeSource`把完整spatial shard组装进SPM，再创建第二个完整SPM destination，
  最后才整体copy到DDR。controller缩的是leaf workset，而物化器在leaf之外重建了与temporal tile无关的full-shard residency。
- 修复模式：先分配最终DDR stage destination，把它作为wave loop carry，逐leaf调用
  `materializeCandidateRootTileIntoDestination`直接写DDR；完成后seal为read-only并缓存exact slice。SPM只保留当前leaf/staging，
  不再出现full-shard assembly。
- 防复发：overfull-to-fit case检查full candidate的actual allocations/lifetimes与typed rejection owner，再检查smaller candidate被
  actual gate接受；compiler work同时计deterministic successor和candidate CardModule/actual memory/target gate，每个candidate恰一次。

## 扩展 FuncOp 参数必须同步 argument attrs

- 现象：小型unit里函数签名扩展通过，但正常source-to-package的frontend函数带`arg_attrs`时，追加一个scheduling destination参数后
  verifier报告“argument attribute array ... got 3, expected 4”。
- 根因：代码分别调用`setFunctionType`和entry block `addArguments`，绕过了FuncOp对signature、block argument和argument attr
  数组的一体化维护。
- 修复模式：使用pinned MLIR的`FuncOp::insertArgument`追加typed boundary参数及空`DictionaryAttr`，由op API原子更新三者。
- 防复发：任何函数签名扩展除无attr unit外必须经过一个带frontend argument metadata的真实source-to-package gate；本轮
  CHAIN/CROSS/GEMM定向lit即覆盖该路径。

## 跨RegionCut的显式DDR读写也必须进入effect closure

- 现象：peer assembly region从source-only DDR载入resident fragment，但生产该DDR的compute/store落在后续region；actual IR形成
  read-before-write，后续region同时包含producer与consumer两个structured root。
- 根因：`splitAtRegionCut`只沿SSA依赖和SPM初始化load闭包移动prefix；compiler-owned DDR的store/load通过memory effect关联，
  不存在把writer拉入prefix的SSA边。
- 修复模式：从prefix内所有memref读取收集view root，在同一TileRegion内找到写入相同root的`wafer.tile.store`，把writer及其
  backward compute closure加入prefix并迭代到effect fixed point；随后再按当前IR relation验证每个compute region恰一root。
- 防复发：resident+peer混合fanin必须检查actual region顺序、send/recv/wait和一root一region，不能只检查通信数量或最终verifier。

## root/component裁剪必须按本Tile拥有的edge endpoint保留策略

- 现象：多输出CROSS或两个独立consumer在同一Tile时，per-component materializer删除remote incoming PeerFragments，ordinary
  output traversal回退为本地融合remote producer；另一路中独立consumer已经seal到DDR，却在outgoing peer查询时被重新计算。
- 根因：edge过滤错误要求producer、consumer两个structured node都属于当前Tile component；remote producer按定义不在当前Tile
  root集合。独立stage又只在已有global materialization cache entry时更新value，cache miss时没有插入sealed result。
- 修复模式：策略只要当前component实际拥有producer endpoint或consumer endpoint之一就保留；独立consumer stage完成后对共同
  `materialized` cache执行insert-or-update，子窗口从sealed DDR view派生。source-only destination traversal还必须传递当前clone的
  structured node mapping，否则BodyEmitter无法形成result-buffer relation。
- 防复发：用多输出transpose→consumer的16-Tile source-to-package/no-card gate，同时由内部postcondition拒绝zero-root/multi-root；
  单纯的小型single-output fixture不足以覆盖remote endpoint裁剪和outgoing cache复用。

## current package不是compiler IR归档

- 现象：FP16 LLaMA `optimization-none`已经完成编译并生成current package，PyTorch no-card runner却继续读取
  `package/functions/forward.mlir`，因此在真正runtime payload准备前报文件不存在。
- 根因：runner混淆了两个边界：source program拥有frontend MLIR和`functions/forward.meta`，executable package只拥有manifest、
  target modules和program data。旧测试把曾经存在的package内IR dump当成runtime合同，迫使普通编译保留无consumer的调试产物。
- 修复模式：boundary input locator和shape/dtype元数据从source program读取；runtime port、module和resource只从current package
  manifest读取。删除依赖package内structured IR的validator和case字段，不把IR dump重新塞回package。
- 防复发：普通source-to-package测试断言package没有`functions`目录；runner unit分别传入source与package并检查各自消费边界。
  compiler IR、计时和work统计只能由显式diagnostic选项或caller-owned sink请求，不能成为默认编译路径或package成员。

## Spatial reduction不能在每个contribution重复消费DPS init

- 现象：reduction iterator被spatial partition后，每个Tile都物化“partial + 原始init”的merged result；后续再把Tile结果相加或
  reduce会把同一个任意init累计participant次，zero-init小测试会掩盖错误。
- 根因：把`PartialReductionOpInterface::mergeReductions`的单Tile便利路径误当成跨Tile contribution表示，没有区分neutral
  partial accumulator和最终一次性DPS init语义。
- 修复模式：contribution region只返回`generateInitialTensorForPartialReduction`产生的neutral accumulator上的partial tensor；
  selected merge Tile按完整iteration rectangle无重叠assembly全部partial，再调用一次`mergeReductions`，其interface root只在这一步
  消费原始DPS init。
- 防复发：测试同时partition parallel和reduction iterator，merge Tile与至少一个contribution Tile重合，并检查region分布为
  “每个contribution一个 + merge一个”；numeric legality仍与spatial domain共享combiner proof，不能用常见zero fill作为协议前提。

## OpFoldResult表示不同不能阻止同一actual producer tile复用

- 现象：fanout或observable producer同时作为下游输入时，offset/size/stride都打印为同一常量，function-local cache仍miss，实际
  TileRegion重复物化producer和boundary load。
- 根因：一侧coordinate是`IntegerAttr`，另一侧是等值`arith.constant` SSA；直接用`OpFoldResult::operator==`或`llvm::equal`
  比较的是表示identity，不是current foldable integer value。
- 修复模式：先接受同一attr/SSA identity；否则两侧都用`getConstantIntValue`解析并比较整数。任一动态侧无法证明相等时保持不同，
  不能按打印文本或shape猜测复用。
- 防复发：coupled fanout、diamond和“producer既observable又被consumer使用”都检查同一source/result/window只有一个actual emitted
  version；cache仍限制同block、type、完整offset/size/stride和definition-before-use，不跨IR epoch。

## Rotating buffer不能由logical edge字段或无约束loop扫描拥有

- 现象：logical movement carrier长期携带`bufferCount`，actual materializer又扫描Tile module中所有`scf.for`自行挑一个看似可用的loop，
  并把合法multiplicity写死为2或3；同一Tile存在两个独立region时还会被错误要求共享一个loop。
- 根因：候选identity、actual loop proof和memory planning三个边界混在一个入口。logical edge尚不知道temporal steady loop、physical
  leaf footprint、alias、最后consumer或release，因此既不能拥有slot winner，也不能证明多buffer可物化。
- 修复模式：先物化actual TileRegion loop、endpoint SSA和buffer roots。Multi-slot只是针对current loop/root的一次
  transformation choice；rewrite在同一region内创建actual rotating allocations、slot selection、phase和release relation，然后使旧
  alias/lifetime analysis失效并立即运行fresh verifier和MiniMalloc。不在actual IR之前创建buffer domain、slot family或预测capacity。
- 防复发：删除logical carrier字段和无edge overload；测试必须同时覆盖2/3/4+、tail/capacity、external-write alias hazard、Direct-DTE
  issue/wait、不同loop拒绝与同Tile多scope。query不得clone/lower，统计只在caller显式请求时启用，overlap winner只能由完整search选择。

## NCC completion不能把target ABI、concrete op分类和跨op analysis放在一个IR helper

- 现象：IR public interface header直接include TX81 NCC ABI，一个free concrete-op switch同时特判join、ArgMax/ArgMin和普通issue，
  pending-worker control-flow分析也住在IR实现文件；target/runtime/model与lifetime/scheduling看似共享合同，实际形成跨层事实源。
- 根因：把单op语义、target command协议和current-IR派生关系都叫“synchronization contract”，没有让operation interface和analysis
  lifetime决定owner；新增op只能继续往central switch加case，硬件worker常量也反向进入IR。
- 修复模式：target worker/mask/kind放pure Target/Core协议；IR worker count从closed ODS enum推导，ordinary issue与特殊completion分别由
  typed op interface暴露；adapter只组合interface。跨if/for/TileRegion/direct call的pending before/after由Analysis/Scheduling按current
  Module重算，递归/indirect/unsupported CFG fail closed。
- 防复发：IR/Analysis header不得include TX81 NCC ABI，runtime/model不得include IR completion；新增NCC op必须有interface正例，
  mutation后重建analysis，并用源码搜索保证旧free classifier/switch零残留。Model→Compiler的其它宽link必须按其真实invocation/numeric
  owner拆除，不能为completion复用保留。

## Ready order和worker不能由clone selector或静态capability row决定

- 现象：旧ready-order用固定engine priority直接产出一个顺序，worker placement克隆整个Module后只产出一个lane映射，target registry再用
  稀疏pair/group row把hard legality与旧profile profitability混在一起；缺row返回Unknown，合法域与实验数据共同决定候选是否存在。
- 根因：order、worker、completion和resource analysis各有独立owner/winner，且通过clone隔离而不是typed assignment+owned apply表达事务。
- 修复模式：TileRegion-to-Instr后从current Instr operation、SSA、effect、range、token和control flow构造一次性
  dependence/resource graph。Scheduler枚举并应用order/worker choice到这份IR，mutation后旧graph失效；completion owner
  随后从new current IR fresh生成join/wait。不从spatial、storage或future event plan构造schedule。
- 防复发：tiny DAG与独立reference比较current Instr上的合法order/worker选择，mutation必须使旧graph失效；源码中不得
  恢复priority selector、whole-Module worker clone、capability/profitability row、future event ID或默认calendar日志。

## Stage pipeline不能再拥有一套whole-Module candidate/clone入口

- 现象：旧fixed-slot实现扫描任意loop、clone完整Module、自行推导slot/stage并返回一个local candidate；buffer count、logical edge、
  ready order和pipeline identity分散，后续只能靠ordinal/clone对应关系拼回search。
- 根因：slot lifetime与stage event structure没有以同一selected edge scope为边界，rollback又被误写成每个mechanism各clone一次。
- 修复模式：结构choice后立即构造candidate-owned TileRegion IR。Pipelined alternative在最近`IsolatedFromAbove`
  scope上物化actual SCF phases、movement、rotating buffers和SSA slot relation，然后从该IR重算event、lifetime、completion和
  MiniMalloc demand。不先建initial storage/EventGraph/execution-structure/storage/schedule链，也不等winner后再重放结构。
- 防复发：旧public API和whole-Module clone可以删除；旧source/test中的periodic DTE、NCC backedge、endpoint reuse、alias/external root、
  odd tail和atomic failure能力必须逐项迁入current owner并受测后才能删除。stage还必须成为search typed transition，而不是由nonempty
  buffering scope自动触发。2个wrapper test和少量selected-buffer test不能代签旧40项semantic witness。

## Per-root TileRegion不等于可执行Tile entry

- 现象：root-to-movement stages的CardModule包含正确private root/relay functions和TileRegions，局部tests全绿；首次由search送入actual memory/target gate时却因每Tile没有
  唯一public entry失败，临时用`func.call`组合又被DDR planner的跨call scope合同拒绝。
- 根因：把“每个root已物化”误当成“每Tile program已闭合”，root function的source boundary/result对应关系没有作为同次construction
  typed结果保留，导致后续只能猜名字/顺序或遗漏entry。
- 修复模式：root construction返回source-argument/structured-node-result keys；Card assembly按keys拓扑选择ready stages，把single-block
  stage body直接move进唯一public entry并用current SSA传递local intermediates。无work Tile构造同signature empty entry；不引入call、
  replay、ordinal或跨passside table。
- 防复发：root-work construction independent actual oracle不仅数TileRegion，还必须检查每Tile恰一public entry；production search让每个complete candidate
  进入一次actual memory/target gate并通过DDR/call-closure/program-resource gate，最终只保留winner。private functions存在或局部verifier通过不能代签
  executable boundary。

## Baseline不能无条件构造search schedule domain

- 现象：把schedule planning canonical query/apply直接放入共同`planTileMemory`后，baseline scalar case从约2秒退化到45秒；大block会承担O(n²)
  dependency DAG构造，即使用户选择`none`。
- 根因：共享exact gate与共享search policy混淆；baseline的canonical source order/worker0已经确定，不需要枚举或建立schedule domain。
- 修复模式：Tile memory planning不拥有schedule选择，也不接受“是否apply search schedule”的布尔开关。Baseline和
  search各自先产生current Instr；baseline在该IR上应用deterministic order/worker，search在该IR上运行自己的scheduler。
  两者都从应用后的current Instr fresh构造completion，然后共享SPM/DDR、transport和最终verification。
- 防复发：baseline定向wall-time与work count必须检查search scheduler调用为零；任何新search axis接入共同lowering时都只能
  提供当前transformation的显式choice，不能在无actual IR的路径中构造future domain后再取first。

## Search constructive proposal不能静默退回exact域第一点

- 现象：LLaMA的显式search一直只报告同一个90,177,536-byte SPM overflow；看似actual-feedback temporal无效，实际proposal在进入
  actual gate前失败，controller随后用budget 1评估了“全部node单Tile、full temporal”的exact first point。
- 根因：constructive各node独立取最大spatial factor后没有先闭合整卡exact demand；随后coupled domain按connected component追加group，
  node id交错时产生未排序assignment并被自己的`contains`拒绝。proposal failure又没有独立诊断字段。
- 修复模式：在公共participant ceiling上由大到小重建完整Card spatial assignment，每点先过exact demand；temporal successor只在前一
  complete candidate得到actual SPM rejection后建立。coupled first/repair assignment在可观察边界按Tile/node semantic key排序。
  显式profile summary分别返回proposal与actual failure；预算同时计partial work和actual evaluation。
- 防复发：测试同时覆盖interleaved disconnected components、large transposed weight和完整LLaMA；exact enumeration去重必须比较完整
  assignment，不能只比较spatial前缀。启发式proposal只能改变访问顺序，不能删除exact sibling或在失败时冒充已评估candidate。

## Temporal accumulator read-before-write要用SSA/DPS证明

- 现象：转置权重接matmul的actual-feedback temporal candidate在dynamic `tensor.insert_slice`处被拒绝，diagnostic显示同一destination另有
  `tensor.extract_slice -> linalg.matmul`使用。
- 根因：旧检查把任意destination read都视为in-place hazard，没有区分“先提取当前accumulator slice作为产生本次insert source的DPS init”
  与真正的并行观察者；退回full wave后又把11008x4096权重错误提升为整块SPM。
- 修复模式：只有当extract的全部用户都是DPS init，且insert source的SSA backward closure包含这些owner时，才允许复用private
  destination；其它observer仍fail closed。structured region capture也必须通过MLIR region utility纳入root closure，不能只遍历显式operands。
- 防复发：large transpose→matmul actual test必须通过完整search gate并保持weight window化；large logical stage检查full boundary为DDR、
  SPM只含selected shard/wave。不得按op名、shape或workload放宽alias检查。

## Typed tensor-transform链不能退回无界generic Presburger image

- 现象：functional KV-cache decode在edge 55（`matmul -> expand -> insert_slice -> collapse -> batch_matmul`）的
  `image-complete-demand`单次超过323秒；complete edge改快后，grouped consumer-input和carrier decomposition仍分别在同一链重复卡住。
- 根因：per-edge、per-destination、grouped reconstruction和carrier虽然已有同一typed transform contract，却在不同入口把组合relation或
  primitive relation重新交给generic Presburger image/rectangle recovery；static insert/extract的piece/remainder事实被丢失。
- 修复模式：矩形consumer domain逐段通过consumer indexing map和每个typed support relation；reshape保留row-major pieces，unit-stride
  insert按交集/坐标平移或矩形差的至多2×rank slabs计算，extract直接平移，最后只union明确矩形。per-edge demand、grouped recipe和carrier
  decomposition复用这一条算法；empty image显式构造typed empty set。
- 防复发：用16-Tile decode-shaped insert/expand/collapse→batch_matmul regression同时调用edge和grouped query并检查16 destination；完整
  FP16 decode search必须在bounded profile内完成package/no-card。不得用shape阈值、wall timeout或失败后fallback generic relation掩盖缺失的
  primitive rectangle contract。

## Program source IR authority不能受include顺序或双reader影响

- 现象：source directory同时保留text和generic bytecode时，compiler只读text、测试把bytecode当辅助；不同producer可以提交互相矛盾的IR。
  同批profile header中的未限定`LaunchSlotId`还曾随include顺序解析成target或package type，fresh full build才暴露ODR/API错位。
- 根因：外部format authority、内部TensorProgram text和package typed ID没有按owner显式限定，增量构建掩盖了header重编译事实。
- 修复模式：外部source只读StableHLO portable artifact，retired成员fail closed；post-SPMD text使用不同internal API。跨namespace schema字段
  include其owner并解析成唯一强类型，public header fresh全构建。advisory verifier和compiler复用同一ingestion但不共享verified-path state。
- 防复发：installed product adapter→verifier→compiler→no-card、feature-off absence、portable corrupt/retired/mismatch负例和public full rebuild同批执行；
  不能因某个增量target链接成功就跳过完整header consumer构建。

## 上游shape覆盖不能代签下游语义分支

- 现象：canonical普通load/publication已经有rank-3的1024/1025纵向case，attention demand也有FD整除/非整除case，但coupled
  representation/movement仍只有1025/1031；ordinary reduction gather又只用rank-2的单个1025 fixture。suite全绿仍无法证明这些
  下游分支在aligned/ragged和真实rank下都消费正确的domain/type/owner。
- 根因：把“同一shape在别的stage出现过”或“同一action kind在普通value上受测”当成当前artifact的覆盖。测试没有按
  `artifact × semantic branch × shape class × exact assertion × downstream witness`建立ledger，最终只剩case数量而没有合同对应。
- 修复模式：每个work item在写代码前列本地覆盖矩阵；规则建立前的输出通过独立coverage closure逐项核对。共享fixture只复用输入
  构造，不共享完成结论；coupled component、ordinary partial、tail和local-skip等分支分别用1024与1025/1031配对，并断言本层typed ID、
  exact domain、owner/action以及直接consumer。
- 防复发：完成评审逐行检查设计矩阵是否绑定实际case；上游case、普通分支、完整suite通过或单个纵向success都不能代签本项分支。
  小shape只保留给明确的bounded oracle、rank-zero或单一故障负例，且同机制仍有真实规模正例。

## 大shape会暴露construction proof丢失和ordered lowering的IR规模问题

- 现象：rank-3/4的1024/1025/1031 attention或ordered reduction在host端生成成千上万份constant affine map、movement descriptor和
  gather/elementwise op；个位数case一直把它掩盖成普通慢测试。无界缓存会进一步钉住只出现一次的insert-slice plan，使RSS提前上涨。
- 根因：ordered reduction先逐tuple建立`IndexRelation`再压缩输出IR；`PhysicalAccessRelation`又在只需要point/span query时提前组合
  Presburger物理关系。其它repeated layout/elementwise query没有request-local semantic cache，或把所有unique key都长期保留。
- 修复模式：非零in-bounds projected constant map记录total/bounded construction proof；physical bit/ordinal relation只在真正做等价查询时
  lazy组合。有序reduction直接消费encoding提供的exact physical piece bounds和tile period，每个run只规划首点、相邻点和末点，使用
  常量有界`scf.for`、loop-carried accumulator及`wafer.instr.gather_scatter` SSA byte offset表达原tuple顺序。target stage用ValueBounds
  证明dynamic offset全域在actual buffer内。request-local cache以完整type/map/engine key校验，并只在第三次观察到相同hash后保留成功
  plan；失败和低复用plan不缓存。rank-zero超大ordered route仍保留4096-tuple compiler work limit，不能伪装成workload legality。
- 防复发：1024 aligned与1025/1031 ragged正例检查physical block/tail run数、loop-carried state、dynamic offset target lowering和actual
  MiniMalloc；negative覆盖无界/越界offset和静态/SSA双表示。显式work-count证明四次相同exact query只实际规划三次。不得恢复逐tuple
  affine-map枚举、eager physical composition、全收缓存，也不得把该编译表示问题扩写成numeric policy或reassociation合同。

## Sequential loop backedge不能复用alternative-branch merge

- 现象：一个常量多trip `scf.for`内只有worker0的无条件fill/elementwise序列，completion却在每次迭代尾插入join；函数return前反而没有
  独立join。conditional issue case看似正常，导致同一算法在unconditional stream上静默串行化。
- 根因：fixed-point把loop entry state与第一轮body state送进用于`scf.if`的alternative merge。该helper看到entry尚无worker、body已有
  worker，便把issue标成path-optional；但guaranteed loop backedge是顺序组合，不是二选一控制流。
- 修复模式：第一轮`bodyState`已经等于entry后执行一次body；下一轮直接以该state重新处理body直到固定点。same-worker issue在实际
  发生的每条path上都由busytable按issue order推进；未发生issue的path没有需要完成的工作，因此branch ambiguity也不构成backedge
  join理由。cross-worker conflict仍在下一issue前完成，剩余pending在loop外observable return处join。
- 防复发：分别覆盖cross-worker backedge、unconditional same-worker、conditional same-worker和dynamic potentially-empty loop；检查join
  的participant与内外位置，不只检查“存在某个join”。

## 结构边界和逐元素展开不能替代硬件completion事实

- 现象：attention每个K2 block更新后固定join worker0；external/cache copy按全1 tile为每个element执行load、store、join和dealloc；
  TileRegion exit及managed WDMA store/reload又自动join；peer send/recv则在issue后立即await。小shape和只检查最终结果的测试全部通过，
  真实shape下join/DMA数量却随block或element数线性增长，异步窗口被静默清空。
- 根因：algorithm decomposition、BodyEmitter和lowering把operation类别、region/materialization结构及“保守同步”当成硬件completion
  proof，在worker/order/storage/lifetime尚未关闭前选择participant和insertion point。测试只断言存在completion或程序成功，没有检查
  dynamic work、位置和直接lifetime witness。只把任意后继same-worker issue当成“worker已完成”同样错误：busytable可以在后续
  同worker物理复用时按实际地址排序，但不能让更晚的不同worker、Kcore或DTE在没有join时复用仍在飞行的地址。
- 修复模式：先读current硬件校准、target lowering和CRT/runtime，把结论区分为supported、board-observed、unknown、excluded。
  上层只保留SSA/effect/token/lifetime；bufferization物化actual allocation/view和reuse，movement发actual token，execution-structure
  transformation物化pipeline/rotating slot。TileRegion-to-Instr后从
  current operation/effect/range/token/control flow重建一次性dependence/resource graph，应用worker/order后再fresh生成minimum-participant、
  latest-unavoidable join/wait。Missing contract保持typed unknown，不能默认
  Synchronous或插全worker drain。same-worker普通链只保持issue order；resolved后继可缩短地址lifetime，但必须另存worker-domain
  obligation，不同worker/DTE/Kcore observer在exact join前一律失败。output copy消费selected temporal tile并形成有界SCF main/tail
  traversal，不能退回全1 tile。
- 防复发：1024/1025/1031级FA/FD与copy case检查steady/nonterminal join、participant wait、DMA/allocation和DTE wait的static site及
  dynamic count；无typed cross-domain cut时前两类join为0，计数不随logical element或K2 block线性增长。正例同时覆盖cross-worker、
  NCC→Kcore/DTE/host、terminal join、DTE first-read/last-release、4-FSM和无环wait graph。测试通过但期望per-block/per-element/
  structural completion时，测试合同本身必须修正，不能作为回归依据。

## Current buffer relation引用不能指向可扩容容器元素

- 现象：TileRegion emission记录result/operand/output/scratch relation后继续追加同类relation，早先保存的vector元素指针失效；后续
  Tile-to-Instr replacement listener随机把allocation归到错误owner，或者把actual SPM demand报告成无owner。
- 根因：把`SmallVector`/`std::vector`元素地址当作跨rewrite回调的稳定identity；容器扩容、erase和conversion replacement都会使该地址
  失效。operation/value地址只能在当前IR epoch做局部lookup，也不能替代relation identity。
- 修复模式：listener只保存`{relation kind, index, result number}`等typed reference，每次访问由caller-owned
  `StructuredMaterializationRelations`解析current元素。Tile emission、selected DDR stage、output insert和Tile-to-Instr scratch都由父
  transaction追加relation；BodyEmitter/lowering只报告实际buffer事实，不拥有全局关系表。full conversion后清除已经失效的source
  operation emission，保留已重接到current storage roots的relation。
- 防复发：测试在多次append和多跳replacement后核对每个relation仍指向current IR；support copy→typed store、output source、scratch及
  missing/duplicate/unknown semantic root group分别有正负例。每个actual SPM allocation没有typed owner时必须compiler-contract failure，
  不能按shape、唯一root、Location或buffer名补猜。
- temporal/control-flow materialization克隆wave-local allocation时，clone必须在创建点通过caller-owned recorder记录当前selected stage的
  显式node集合。不能在SPM rejection后沿普通Instr operand/result依赖扩散owner：compute dependency不是alias或ownership，扩散会把
  无关program output编号带入同一demand并使actual feedback失真。Tile-to-Instr新建scratch同样只从source operation的显式emission或
  buffer relation取得owner；手写fixture必须提供该relation，不能要求listener从后续store反推。

## Bufferization不能越过Tile隔离边界或丢失actual execution identity

- 现象：rotating slot在`func`入口创建allocation后被`wafer.tile.region`内的`arith.select`引用，verifier报告isolated region捕获；把两个
  result绑定到同一reuse object后，再按storage root匹配compute event会让两个instruction同时看见两个structured owner。
- 根因：bufferization rewrite把函数作用域当成allocation owner，没有遵守实际SPM planner要求的TileRegion scope；
  同时企图在alias/reuse合并SSA root后恢复未物化event identity，把buffer identity误当成execution identity。
- 修复模式：alias/reuse和rotating allocation都在所有相关root共同且唯一的`wafer.tile.region`中创建；跨TileRegion
  共享返回typed Unsupported。Rewrite使用current operation/SSA relation和listener更新直接owner；不创建EventId或在合并后
  反查execution。Scheduler在TileRegion-to-Instr后从actual operations/effects重建依赖。
- 防复发：用rank-3 1024/1025 alias/reuse和1031 pipelined rotation分别贯通actual TileRegion→Instr→MiniMalloc；检查
  allocation数量、SSA owner、verifier和current-Instr dependency。不得把allocation提升到`func`、按合并后root猜execution，
  或因synthetic fixture能verify就绕过actual TileRegion scope。
- rotation selector必须使用归一化iteration coordinate`(iv-lower)/step`再取模，不能直接对raw induction variable取模；非单位step会
  否则长期选择同一slot。创建的每个slot是同一TileRegion中的actual allocation root，caller-owned relation同步扩展；memory stage
  只从select/root union和fresh completion重算lifetime，不读取slot-family plan。

## Actual feedback必须绑定完整显式choice和current source epoch

- 现象：两个candidate的最后一个choice相同，但spatial/region/temporal/layout/movement choice不同；若controller只按
  最后choice reserve/cache，第一个actual rejection会把合法sibling当duplicate或forbidden。
- 根因：用未物化的最后plan object代替完整transformation choice identity和current source epoch；actual conflict root又被误当成
  可推广到prefix的no-good explanation。
- 修复模式：reservation和exact full-point cache使用完整显式choice key与immutable source identity；actual IR由该choice通过
  唯一materializer生成一次。SPM owner/conflict只随该exact rejection保存，没有证明时不做prefix subsumption。
- 防复发：用最后choice相同、上游choice不同的1025 sibling证明rejection只命中自身；逐choice identity、cache on/off
  和反序独立winner oracle同时执行。禁止按root、shape、bytes、最后一轴或parent pointer扩大no-good。

## Resumable search不能借用临时config或依赖遍历顺序填充cache

- 现象：one-shot search正常，但把同一遍历切成每次一个credit后，persistent session在actual gate读取到损坏的program boundary；
  另一个fresh session从完整choice生成candidate时暴露隐藏cache前置。
- 根因：resumable owner保存了调用表达式产生的`FrontendProgramVerificationResult`和`ExecutionConfig`引用，resume时临时值已经析构；
  actual evaluator又假定cache必然按固定顺序预热，显式choice和source本身不足以独立物化candidate。
- 修复模式：persistent session复制小型immutable config values，只借用明确由outer transaction持有的TensorProgram、diagnostics和
  ProgramData。任一完整choice均能通过唯一materializer独立产生candidate-owned current IR；cache只是request-local pure-query memo，
  不是隐藏前置。显式continuation stack在cutoff后原位resume，不重放或重建winner IR。
- 防复发：同一1024 source分别one-shot和每credit resume到同一first-accepted choice；fresh parse可直接物化前两个
  complete choices，且在pure-query cache on/off下IR/status一致。不得用延长timeout、保活临时对象或先跑warmup修补cache依赖。

## Exact set的normal form不能代替物理可表示性证明

- 现象：multi-producer `insert_slice` reconstruction产生语义有限且精确的`GeneralPresburger` domain；后续layout/movement
  transformation只看到空box metadata并报告resource mismatch。
- 根因：把exact set的construction form误当成可物化性结论，或在多个下游复制domain metadata后比较副本，而不是让
  current-IR transformation消费同一exact relation。
- 修复模式：针对current value/use的layout或movement rewrite在一次调用内对非empty exact set做bounded direct-disjunct recovery；
  只有每个disjunct都是static rectangle且pieces两两disjoint时规范化为语义等价`BoxUnion`。若整体能被bounded exact
  equality证明为一个dense rectangle，可退化成一个piece；否则typed unsupported，绝不使用bounding box或跨stage
  resource description。
- 防复发：同时保留rank-3、主维1024/1025的multi-producer全链和一个1025级GeneralPresburger direct case；后者检查Presburger equality、
  exact piece offsets/sizes及下游可消费。不能靠`getBoxes().empty()`判语义不支持，也不能在每个consumer重复generic equality。

## Definition-only result不能靠伪造use或discard plan闭合

- 现象：`insert_slice`覆盖producer输出的一部分后，canonical spatial仍执行对应shard并需要output storage；该result没有transfer/publication，
  storage把它当成漏use拒绝。若简单允许所有empty-use，又会放过真正漏掉的observable publication。
- 根因：在actual SSA之前预先列出result、carrier和storage，遇到无consumer的future result后只能再发明discard和self-use让计划自洽。
- 修复模式：structural materializer只创建current exact demand和observable/effect closure要求的execution。Pure result物化后没有actual use时，
  由局部DCE/canonicalization删除其compute和allocation；effectful op保留actual effect；observable result必须有actual publication use。不创建
  discard ID、fake self-use或storage edge。
- 防复发：rank-3 1024/1025 multi-piece overwrite检查actual IR无orphan pure compute/allocation；独立负例删除observable publication
  仍必须失败。不得用名字、shape或伪造use判断discard。

## Canonical coordinate不能保留另一套IR materializer

- 现象：同一个complete physical plan在canonical coordinate走旧edge-driven materializer，非canonical Region/representation走新的
  execution-driven materializer。真实16-Tile block中1,696个selected execution形成40,960个compute emission；随后slice、layout、allocation、
  global和movement一起膨胀。两条IR都能通过局部verifier。
- 根因：增量迁移只为新coordinate接入新实现，并用“是否等于canonical/default plan”保留旧apply路径；后续重命名或移动到共同入口时没有退役
  donor。旧路径让edge carrier按每个destination fragment调用producer tiler，execution relation在IR生成后才补记；verifier再把actual
  emission收集成node ID集合，丢失了重复次数和execution identity。
- 修复模式：每个policy的materializer直接按自己的plan构造IR。compute创建helper只对materializer可见；movement/layout API只接收已经存在的
  SSA result/view，类型和include边界上拿不到producer op、tiling callback或compute builder。canonical只是search domain中的普通coordinate，
  不能选择另一套builder。不增加第二份expected plan或execution-count verifier。
- 防复发：迁移operation、representation、region或pipeline实现时，矩阵必须列出旧能力、新owner、唯一production caller和donor删除证据；
  同一controller/domain内的canonical/noncanonical、generic/attention coordinate使用同一actual construction。该规则不要求baseline与search
  共享plan、preparation或actual IR；两者只共享设计明确允许的无策略leaf。禁止用同一domain内的plan equality、default attribute、fixture kind
  或feature presence选择第二套IR transformation；禁止post-hoc追加本应驱动construction的relation；`set`只能验证集合语义，不能验证
  all-and-only occurrence。逐stage inventory必须把logical execution、physical compute、view、movement和target-call分别计数。

## Driver-owned private IR不应为同一planning kernel反复构造临时PassManager

- 现象：大图baseline在多轮actual SPM rejection后首次进入DDR planning，固定在`PassManager`析构时报
  `double free or corruption`；GDB调用栈落在per-Tile DDR pass adapter，SPM rejection和candidate IR本身均已完成。
- 根因：compiler driver已经拥有独立、可丢弃的per-Tile `ModuleOp` transaction，却为每个Tile重新创建只包装一个DDR planning
  kernel的临时`PassManager`。这既没有提供跨pass analysis复用，也把pinned pass adapter的额外所有权/析构周期放进每个candidate的
  热路径。
- 修复模式：注册的named pipeline继续使用AnalysisManager-aware pass adapter；拥有private Module transaction的driver直接调用同一个
  typed query/apply kernel。kernel仍执行相同actual lifetime、capacity、range和offset算法，失败时由caller丢弃当前Module，不建立第二套
  planner或绕过verifier。
- 防复发：direct kernel与registered pass adapter对1024/1025级rank-3输入比较exact offsets和失败类别；真实16-Tile candidate证明
  DDR、target和package均实际到达。不得用禁用MLIR multithreading、固定单worker或猜测其它analysis线程安全来掩盖析构问题。

## Memref SSA identity不是memory version

- 现象：一个Tile entry先把结果写入Card DDR function argument并返回更新值，后续stage却再次读取原argument；One-Shot
  Bufferization为保存Tensor SSA所要求的旧值生成整buffer DDR→DDR copy。类似地，两个相同source/type的layout materialization仅凭
  dominance合并时，若中间存在对source alias的写入，后一个consumer会错误读取写入前的转换结果。
- 根因：把同一个memref SSA value误当成“内存内容始终相同”的version。Memref SSA只固定引用，不会为memory mutation产生新SSA
  definition；Tensor destination/result和显式stage result才形成可见值的current chain。
- 修复模式：mutable function/resource destination由每个writer返回current Tensor/SSA result，后续reader和writer顺序消费该result；
  block argument只作为初始值。复用layout/materialization时除same source/type和dominance外，还要用alias/mod-ref证明两次materialization
  之间没有source或其alias的write/free；不同block、未知effect或不确定alias保留独立materialization。
- 防复发：rank-3 1024/1025 producer→shared Card DDR→consumer case在bufferization前断言current result链、之后断言冗余copy为0；
  1/2/15-use layout case覆盖只读共享，并用intervening alias write反例证明不复用。不能用CSE、copy lowering或copy-only TileRegion掩盖
  错误的current-value串接。
