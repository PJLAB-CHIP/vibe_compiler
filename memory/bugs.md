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
- 根因：driver解析了optimization policy，却在executable synthesis边界丢弃或绕过它；测试反而把相同结果锁成合同。
- 修复模式：`search`调用唯一physical-dataflow selection，`none`只materialize conservative baseline；两者从同一source进入同一
  actual/exact pipeline。
- 防复发：测试验证两种policy都真正进入synthesis统计，并允许结果不同；不能要求它们永远产生相同digest。

## 全候选actual clone导致编译时间与RSS失控

- 现象：model-scale图在candidate笛卡尔积上重复clone、Tile→Instr、completion和packing，编译时间远超单算子合理范围。
- 根因：把actual materialization当作candidate enumeration，或给不同Tile/mechanism各建局部shortlist再组合。
- 修复模式：先用query-local typed legality、topology symmetry、exact equivalence、admissible lower bound与future-compatible
  dominance安全剪枝；其余状态惰性保留，只有需要exact结论时才按需materialize，且peak live actual clone为1。有损shortlist/
  beam/cap只能在实际profile后作为显式budgeted trade-off。
- 防复发：统计generated/rejected/deduplicated/expanded/actual-probed/exact-failure/peak-live；scale gate同时看deterministic work、wall和RSS，
  不设任意固定秒数替代复杂度分析。

## Late exact failure不能触发隐藏repair

- 现象：SPM packing或ABI failure后，late pass自行缩tile、spill、改worker或切communication，selected IR与search cost不一致。
- 根因：allocator/finalizer被赋予了搜索职责，产生第二winner owner。
- 修复模式：late stage只返回validated result或candidate failure；failure回到同一physical-dataflow search选择其它已表示候选。
- 防复发：negative test锁定packing/ABI failure不改写candidate；repo scan禁止late retile/spill/replan selector。

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
- 修复模式：同一complete CardModule candidate共同表达Tile集合/work domain、temporal tile、TileRegion/融合、communication、buffering和overlap。
- 防复发：测试同时保留maximal local residency与cross-Tile operator pipeline、large-tile cut与small-tile overlap等对立候选。

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
- 防复发：每个actual candidate都执行fresh completion；missing/wrong worker/event/participant/reuse分别有负例。

## `ReturnAfterLocalDrain`不是card-scoped barrier

- 现象：每个Tile local return合法，却在其它Tile或transport尚未完成时发布output；或在每个entry尾插入全卡等待造成死锁。
- 根因：混淆entry-local drain、cross-Tile message completion和card-scoped invocation success。
- 修复模式：Tile entry只保证本地发起的observable/reuse/status work已收敛；CardExecutable/runtime owner另行等待16个entries与全部transport obligations。
- 防复发：一个Tile提前返回、另一个仍有Direct-DTE/event的正例；缺失global obligations和多余cycle分别失败。

## SPM packing只能求解final fixed problem

- 现象：allocator在capacity失败时自行缩tile或spill，或只按sum(bytes)估计而忽略lifetime/conflict/alignment。
- 根因：把allocation quality与schedule search混合，且没有从final roots/effects形成exact conflict problem。
- 修复模式：search-time用safe lower/upper bound剪枝；final actual IR形成fixed roots、lifetimes、alignment和conflict，MiniMalloc只返回offsets或失败。
- 防复发：验证overlap clique、alias、loop-carried lifetime、async use、padding与high-water；packing失败不产生IR mutation。

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
  cost multiplicity和numeric regrouping legality，不按模型名、shape或operand位置特判。

## Iterator域体积不能冒充operand tile驻留

- 现象：GEMM的cheap model按`M*N*K*元素字节和`估算SPM，把三个二维operand误当成三个三维buffer；合法大tile被过早剪掉，
  搜索倾向大量小wave，优化后的card-scoped执行反而没有性能收益。
- 根因：只看到iteration domain，没有用每个operand自己的indexing map求实际window；同时用一个整体alignment掩盖了
  per-buffer allocation事实。
- 修复模式：对每个Linalg operand从当前iterator tile和symbol-free affine indexing map求常量包围盒，按真实element width和
  per-buffer alignment累计理论驻留；symbolic或无法证明的项直接不计，不能引入`unknown`哨兵、猜测倍率或阻塞排序。最终
  legality仍由actual clone的fresh SPM packing拥有。
- 防复发：用一个operand footprints能放入SPM、但iteration-volume模型必然超限的GEMM检查none baseline不产生虚假K wave；
  affine-window conv同时覆盖stride/dilation map。

## 浮点reduction split不再以源顺序或fastmath为gate

- 现象（历史）：矩形分块两个reduction轴后，chunk坐标被提到in-tile坐标之前，浮点`addf`求和顺序改变；旧实现把没有
  fastmath的source当成非法候选，还要求前置reduction轴unit-tiled才能split后轴。
- 收敛结论：上述顺序/fastmath gate与01 numeric policy冲突，已删除。f16/bf16/f32的reassociation、tree、distribution/
  factorization、reduction/GEMM split是supported numeric transformation，不消费任何fast-math flag作语义开关，验收统一
  归typed comparator。`verifyReductionSplitNumericLegality`对`addf`直接合法；`preservesSequentialReductionOrder`事实只保留给
  整数combiner分支——整数no-wrap/overflow语义是唯一剩余barrier。
- 防复发：新增reduction split/树合法化时只按整数overflow语义设barrier，不得以源combiner顺序、`fastmath` attr或
  lexicographic前轴条件拒绝浮点split；负例构造注意纯reduction标量输出没有parallel轴，placement domain本就不表达
  （见`tasks/plans/physical-dataflow-synthesis.md` Q49.P gate），且浮点demand都可缩到最小合法vector，「最小tile超SPM」的
  浮点capacity反例当前lowering下不可构造。

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
  分层判断。同Tile独立traversal使用`LocalShardResidency`，只有实际consumer-driven递归traversal使用`CoupledFusion`。
- 防复发：same-numbered transpose必须产生resident加peer fragments并下沉到DTE recv/send/wait；同Tile和disjoint reduction均覆盖
  exact demand，window覆盖一对多稠密像集，stride覆盖“不可用bounding box冒充exact fragment”，baseline继续断言零fusion。

## Direct-DTE局部endpoint顺序会造成FSM溢出或全卡wait环

- 现象：先发射一个Tile的全部recv再wait会超过有限receiver FSM；改成每个局部recv后立即wait后，多个Tile按各自SSA顺序
  又可能形成跨Tile循环等待。
- 根因：通信endpoint只有局部发射顺序，没有所有参与Tile共同遵守的全局message顺序；FSM lifetime和全卡无环性被分别修补。
- 修复模式：从explicit communication identity、round、payload slice、endpoint kind和peer构造稳定全序；每个Tile过滤与自己
  无关的message后仍保持该全序，每个endpoint发射后立即exact wait。这样receiver实时占用为1，且所有依赖边按同一方向推进。
- 防复发：多源fanin测试同时检查最大live recv为1、send/recv/wait exact配对和Direct-DTE binding；source-to-package transpose及
  attention baseline必须产生fresh no-card package，不能只检查Tile IR文本。

## Baseline显式op边界不能无条件作用于search-policy融合

- 现象：为了让`none`逐op独立tiling而给每个consumer补DDR seal/reload后，`search`的`CoupledFusion`候选也被同一逻辑
  强制切断，导致full weight reload、SPM溢出或最终零融合。
- 根因：candidate materialization没有按typed edge action区分`IndependentEdgeBaseline`与`search`的
  `CoupledFusion`/local shard residency，把baseline策略写成了所有policy共享的结构改写。
- 修复模式：显式consumer boundary只在independent baseline action生效；search-policy fusion仍由output traversal物化，并以
  actual in-region SPM use-def witness计数，不以proposal标记代签。
- 防复发：none source-to-package测试断言`actual_fused_edges=0`和中间DDR movement；search-policy三阶段/layout-buffering测试断言
  actual fusion candidate被接受且selected actual fused edges非零。

## 零actual fusion不能证明baseline已与search policy和region资源解耦

- 现象：`none`在candidate family前提前返回且`actual_fused_edges=0`，但仍复用search candidate/domain evaluator、proposal
  ordering和group materializer；多个独立structured root可能进入同一TileRegion，共享SPM预算、lifetime/lowering scope和失败归因。
- 根因：把“没有执行search loop”和“没有coupled edge”当成完整解耦证明，只约束edge action，没有约束baseline调用闭包和
  TileRegion structured-root cardinality。后端允许multi-root region只说明IR合法，不代表它适合作为canonical baseline。
- 修复模式：baseline controller直接从typed structured/relation/target facts构造唯一方案；每个TileRegion只拥有一个structured
  compute root及必要non-root support closure，root cardinality由materialization relation证明；同一Tile上的其它root进入独立
  顺序region，跨root shaped dependency显式DDR。只与search
  共享single-coordinate的policy-free materialization、scoped probe和最终lowering/verification，不共享state/candidate/grouping/
  ordering，也不调用option-domain、propagation、recursive CSP/backtracking或“只取第一个”的assignment solver。
- 防复发：除零actual fusion和DDR movement外，测试还要检查每个baseline region的structured-root数、同Tile multi-root的region
  数、search-policy调用计数、完整CardModule/CardExecutable各一次；equal-shape fanin与function-scope case验证SPM失败只沿直接
  typed causal witness refinement，不能猜测或跳过scope。

## 去search耦合不能把baseline退化成fixed-assignment validator

- 现象：设计为了禁止`none`复用candidate/evaluator，把baseline写成只apply已选placement/temporal/buffer并执行一次scoped
  probe；正常上游IR的完整tile超出SPM时，反而没有owner继续缩tile并产出可执行结果。
- 根因：混淆了“禁止性能候选选择”和“禁止确定性功能合法化”，把resolved assignment误当成baseline输入；只设计了单次
  closed-coordinate query，没有定义谁遍历合法breakpoint、何时终止以及支持域内的完成保证。
- 修复模式：`none`从未绑定物理选择的正常IR进入，由controller按semantic全序维护一个current coordinate；每次只构造当前
  coordinate、重新推导operand/halo/result/temporary/movement/alignment/bank/lifetime并运行exact scoped probe，再按typed
  rejection推进下一项必要coordinate，第一个fit形成resolved assignment。不得预先生成完整placement/temporal option domain；
  trial是query-local feasibility状态，不进入candidate、score、incumbent或proposal统计；任一transition必须预定义、单调、
  不分支且不回溯，旧coordinate立即销毁，避免把deterministic search换名为functional fallback。
- 防复发：至少一个初始完整tile超SPM而较小合法tile可放下的source-to-package/no-card正例，以及最小合法tile仍超限的typed
  negative；声明支持且baseline域存在completion时必须得到accepted executable，indeterminate必须作为compiler failure，不能
  用“未进入search”或“没有fixed assignment”解释失败。

## Exact矩形恢复不能把generic Presburger等价证明放在baseline热路径

- 现象：即使单独观察一个logical shard pair query，CPU仍可长时间停在`getExactStaticRectangularImage`后的
  `PresburgerSet::isEqual/isSubsetOf/subtract`；变量数和disjunct数没有超现有limit，因此静态budget检查没有阻止该路径。
- 根因：supported projected/permuted/static-rectangle indexing semantics在relation composition后丢失closed-form witness，矩形
  recovery只好先求generic image、构造bounding rectangle，再调用通用集合等价证明。结构规模上限不能约束Presburger算法实际
  work，外层placement option-pair/CSP又会乘法放大同一查询。
- 修复模式：builder/composition在typed proof成立时保留或直接重建closed-form rectangular-image witness，single-coordinate
  baseline优先消费该witness；generic recovery调用前使用覆盖constraint/local/coefficients等复杂度的fail-closed preflight并记录
  query-local ledger。超限是`ResourceExhausted`/indeterminate，不能当logical infeasible、不能推进fallback。
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

## 单状态baseline不得套用多候选winner协议或accepted后旁路工作

- 现象：唯一exact-verified Llama baseline在selection后又完整执行一次CardModule、16-Tile Instr、SPM、DDR和resource verification，额外消耗
  数分钟；accepted后还构造并丢弃schedule/duration结果，普通compile无条件生成Tile IR trace，但这些都不改变产物语义。
- 根因：为search cohort控制峰值内存而清空每个accepted candidate Tile module的策略，无条件复用到了只有一个semantic state的
  `none` controller；同时把未消费分析和调试输出误放在producer主路径，而不是由真实consumer显式请求。
- 修复模式：`none`保留已经通过全部exact verification且已清除query-local source relation的唯一executable，直接move返回；
  未被输出合同消费的schedule/duration工作删除，可选trace在请求方惰性生成。未来候选策略如何保存或重建winner由其自身任务决定，
  不能反向规定baseline控制流。
- 防复发：none定向测试断言完整CardModule和CardExecutable各一次、selected executable rematerialization为零、默认trace为零；
  模型规模检查work count和结果返回前的stage计数，不运行旧search或历史winner行为作对照。

## 读取源码marker的测试会把退役实现伪装成合同

- 现象：source或test没有进入active CMake/lit/CTest执行图，但一个已注册测试逐文件读取其文本并断言旧symbol marker存在，整体仍显示green。
- 根因：把“仓库里还有某段源码”当成“能力已被编译并经过行为验证”，source inventory又只检查已知子集，形成互相放行的假闭环。
- 修复模式：组织检查以filesystem、CMake source、unit/lit/CTest registration和明确的current-task dormant owner做双向集合闭合；
  能力测试只验证编译后的接口和行为。退役实现独有能力先迁入active owner并受测，再删除源码和marker断言。
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

- 现象：Q58 TensorPayloadResolver持有指向`SmallString tensorProgram`的StringRef，构造时该path尚未append
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
- 根因：这些测试不属于默认lit/ctest路径，Q49-Q55多次refactor后无人刷新期望；真实行为变化与测试更新在不同commit，
  且部分期望（如`emitOpError`带operand、旧package source copy）对应的是已经退役的表示。
- 修复模式：逐项确认行为变化commit与测试最后touch的先后（`git merge-base --is-ancestor`），结合pinned LLVM/MLIR源码
  判断哪个是current合同；Q56只修自己造成的失败和本批已触及文件的陈旧期望，其余登记为独立后续任务。
- 防复发：改production行为时搜索所有test目录（含非默认gate），同步更新期望；"该目录不在CI"不能作为让测试红的理由；
  未知枚举case先用bounded小范围filter定位；已经定性为旧实现状态爆炸的case直接从当前批次排除，不再用长时间单跑确认。

## 全量单测二进制中的单个case可能100% CPU死循环

- 现象：`WaferUnitTests`全量跑在`StructuredDAGPlacementEnumerationTest.MultiOutputFanoutCanPlaceBranchesOnDifferentDestinationGroups`
  卡死（两个遗留进程各烧CPU 1.5-2小时）；gtest stdout块缓冲下无输出，看起来像整批挂起。
- 根因：placement枚举在该case上状态空间爆炸，100% CPU自旋；测试文件与枚举实现均在Q56改动范围外（最后一次touch是
  前置refactor commit）。
- 修复模式：已有bounded诊断已经定位到旧placement枚举后，不再反复单跑长case；普通全量验证显式排除它。Q51.Core删除旧
  search/test owner，Q50.B以带状态规模上界的tiny reference enumerator重建spatial domain，Q52只profile完整new chain。
- 防复发：全量gtest运行前排除已知旧爆炸suite；新增枚举测试必须自带状态规模上界或明确标注预期case数，超界诊断使用更小的
  同构fixture而不是继续烧原始长case。
- 2026-08-16补充（Q50.A验证）：同一suite还有两个同类病理长跑case——`DiamondFaninRetainsAnExpressibleTwoOperandBoundary`
  与`ReductionDataTransitionKeepsExactFragmentsAndLocalAlternatives`，1800s内不完成（状态数100→10,000→1M逐节点膨胀，
  每状态`extendState`含120次pairwise拓扑最短路径查询约260µs，与Q50.A改动无关——相同状态数、相同legality结果已在
  HEAD代码路径上逐项核对）。全量filter需一并排除这三个case；`ThreeStageChain...`与`CompletePlacementOrder...`
  可正常完成。旧case随Q51.Core删除；需要保留的domain witness迁到Q50.B tiny oracle，Q52不以旧实现耗时作profile基准。

## 跨tree调用点没有随signature变更同步编译

- 现象：Q58给`buildCardExecutable`加`ProgramDataHandoff&`形参后，q55主树编译通过，但SystemC树
  （`build/q54-fresh-model`）的三个integration test调用点仍传7参，该树在此前从未重编过这些TU。
- 根因：多个configured build tree共享同一source；增量验证只跑主树时其它树的编译错误不可见。
- 修复模式：改公共API形参后，所有仍被任务引用的configured tree都要fresh build；调用点按现有模式补齐
  default-constructed handoff。
- 防复发：公共API变更的验证清单里显式列出所有active build tree；提交证据注明哪些tree实际重编过。

## StringRef::slice第二个参数是exclusive end不是length

- 现象：Q56 program-data canonical verification的零padding检查在负例上静默通过——96字节文件中[80,96)非零，
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

## Baseline probe-fit与final SPM planning可能分歧（Q49.P，已修复 2026-08-17）

- 现象：`CardExecutableSynthesisTest.NoneJointlyRefinesExplicitProducerStageAndConsumerDemand`（transpose+
  fill→matmul 4096规模、none policy）在2026-08-16的HEAD（22eb9931）即失败：一次temporal refinement
  （4096→2048）后全卡TileRegion静态SPM probe全部fit，唯一完整CardExecutable编译仍以gate=spm-allocation
  失败。
- 根因：2048 breakpoint上所有region probe返回`UnsupportedLifetime`（`requires-function-scope`），
  controller把它当fit计数后跳过；唯一完整gate的函数级planning才暴露真实overflow。
- 修复：Q49.P新增函数级probe `evaluateTileFunctionSPMCapacity`（clone Tile FuncOp后跑与最终gate相同的
  `instr-memory-planning-preparation`+`assign-spm-offsets`序列）；`RequiresFunctionScope`是scope
  escalation请求：提升到最近合法IsolatedFromAbove ancestor（该Tile的FuncOp），其typed verdict作为该
  region的结论；无法在准确scope得出结论时indeterminate中止，不再跳过。probe与final由此消费同一demand
  集合。
- 防复发：probe若克隆IR，evidence attribution必须经clone-side relations（`StructuredBufferReplacementListener`
  + remap），且attribution的witness收集要沿store/load/view链找transfer端点（DDR wave两端不通过SSA
  别名与wave buffer相连，直接storage-root匹配会miss）。

## 发布点之后不能再运行会翻转事务结果的validation

- 现象：staging目录先rename到最终路径，随后installed-root readback失败并返回错误；cleanup只覆盖staging，最终目标仍可见。
  profile用两个顺序rename时还会短暂或在进程退出后永久暴露ordinary-only状态。
- 根因：把单次rename的原子性扩张成整个多产品事务的原子性，并假设post-commit readback“只会因compiler bug失败”。任何真实I/O、
  并发mutation和内部不变量错误都仍是可达错误边，两个各自原子的rename也不是共同原子commit。
- 修复模式：所有可失败validation和identity binding在唯一visibility point之前完成，全部共同产品位于一个可单次发布的owner/root；
  若协议确需installed-path动作，显式建模committing/committed状态和可恢复操作，并证明每个错误出口不泄漏本轮目标。
- 防复发：failure injection必须覆盖最后一次发布前后、installed readback和进程/第二产品边界；同时断言返回status、ordinary/profile
  可见性和staging残留，不能只测第二次rename正常返回失败时的best-effort rollback。

## wafer-compile-card-baseline CROSS case在baseline materialization验证中长时间挂起（已修复 2026-08-17）

- 现象：`test/Tools/wafer-compile-card-baseline.test`的CROSS（16-Tile transpose support chain，256个peer
  fragment）在`source-to-tensor-program`后无输出、CPU 100%数分钟以上（90s/280s timeout均杀不掉自然结束）；
  CHAIN与GEMM case正常（~1-4s）。直接gdb启动+对inferior发SIGINT采样：主线程在
  `synthesizeDeterministicBaseline → validateSelectedTileLayouts → rootContainsNodeLayout →
  operationUsesStructuredNode → collectStructuredNodesUsedByOperation → shareStructuredBufferStorage →
  collectStorageRoots`区域（StructuredBufferRelations.cpp），两次采样位置不同说明在推进而非死循环。
- 根因：`rootContainsNodeLayout`对每个strategy在整棵tile root上walk，每个op×每个buffer relation都从零
  递归`collectStorageRoots`（fresh visited set，无memo）；CROSS的16 destination × 16 owner peer fragment
  使指令模块膨胀，O(ops×strategies×walk)放大到分钟级。该验证链是旧refactor遗留
  （`git log -S validateSelectedTileLayouts` → ab8bf482/318bba75，早于Q50.A）。
- 证据：`git checkout 22b3fd12`（Q50.A前）与HEAD（92abdf29）均复现同一挂起；Q50.A review-gap批次与
  Gap 1 per-destination改动无关。Tools lit不在默认lit/ctest路径（见“非默认gate的lit期望静默过时”条），
  该测试长期无人执行，属Q49.P baseline functional closure域。
- 修复：`StructuredBufferRelations`新增query-local `StorageRootMemo`（每value一个heap-owned
  `DenseSet`，同一IR epoch内共享），`validateSelectedTileLayouts`等每root一个memo贯穿调用链，把
  per-op×per-relation的storage-root walk摊销为O(1)。注意：memo内层set必须heap-owned（`unique_ptr`），
  否则外层map rehash会悬空已返回的引用。CROSS由分钟级降到~34s整套lit。
- 修复模式：排查这类“编译挂起”时先在同一二进制内用阶段探针二分，再用gdb从启动开始跑inferior并向其
  进程发SIGINT采样栈（attach被ptrace禁止，但gdb启动inferior可行）。
- 防复发：Tools目录的lit/ctest在声称gate通过前要单独执行并带wall-time上限；materialization验证链的
  每op递归walk需要memo化或按relation反向索引，避免O(N²)回归。

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

## SPMD card-level 合同切换后未跟上的 stale 测试（2026-08-17 修复）

- 现象：Tools/Runtime lit 4 个失败，全部在 Q49.P 之前即存在（2026-08-11 `76f68e29` 把 SPMD 层从 16-rank 改成
  card-level 后测试未同步）：
  1. `wafer-compile-structured-tensor-program.test`：用 `CPU_NUM_DEVICES=16` + `--emit-sharded-program` 采 16-device
     sharding 程序喂给 `num_partitions=1` 的 card-level helper，撞 pinned XLA partitioner 未初始化内存 bug（`%pad`
     垃圾 shape 且每次运行值不同，非确定性）；03 合同已废止"partition 数 = Tile 数"旧语义。
  2. `wafer-compile-stablehlo-sharding-propagation.test`：`wafer-materialize-execution-mesh` 的 shape 在 76f68e29
     改为必须显式传（旧 `--default-tile-count=16` fallback 已删），测试 pipeline 没传 → `logical mesh shape must be
     explicit`。
  3. `wafer-run.test`：manifest 读取失败消息已改（`failed to open package manifest member`），FileCheck 期望串过期。
  4. `wafer-compile-spmd-partition.test`：reference capture 是 f32，target 明确拒绝 f32 GEMM
     （`unsupported_target_instr: GEMM does not support f32`），连带 search 在 selected-buffer-materialization 报
     indeterminate；且 manifest `inputs[].role_index` 语义已改（input 域内从 0 编号，不再沿用 parameter 的连续编号）。
- 修复模式：reference capture 模块和输入按 dtype 政策改 f16（`wafer_pytorch_xla_capture.py` 的
  `_make_reference_matmul_module`/`emit_reference_stablehlo_program`），contract mock 同步补 `float16`；
  source-to-package 测试切 `--optimization-policy=none`，只验证当前baseline source-to-package合同，不附带运行或对照旧search；
  mesh pipeline 补 `{shape="1"}`；manifest 断言同步 current 合同。
- 防复发：target 合同里明确不支持的组合（f32 GEMM）先写 verifier 拒绝，再把功能纵向/qualification 默认 dtype 保持在
  f16/bf16（见仓库板测 dtype 规则）；改 SPMD/partition 边界合同时，同批 grep 所有消费旧 CLI/字段/消息的 lit 测试。
  外部 helper 出现非确定性垃圾 shape 时先查"喂给 helper 的输入是否符合当前合同"，不要先怀疑 helper 二进制。

## 给 LLVM_OPTIONAL_SOURCES 里的退役源码插桩不产生任何效果

- 现象：往 `lib/Wafer/Compiler/LowerRankInstrModules.cpp` 加调试打印后 `cmake --build` 报 "ninja: no work to do"，
  二进制里 grep 不到新字符串，运行输出也没有。
- 根因：该文件在 CMakeLists 的 `LLVM_OPTIONAL_SOURCES` 列表而非 `add_mlir_library` 主源列表——它是退役源码，不属于
  active build；同名门禁逻辑已由 `CardExecutableLowering.cpp`/`CompileCardExecutableLLVMModules.cpp` 承接。插桩前没核对
  对象文件是否存在。
- 防复发：改代码前先确认文件在 active build 里（`ninja -C build/… -t query lib/libWaferCompiler.a | grep <文件名>` 或
  `ar t lib/libWaferCompiler.a`）；"ninja: no work to do" + 符号不在二进制 = 源码不在构建图，立即改查 active 实现。
