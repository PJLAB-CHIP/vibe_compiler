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

## Tile entry argument不是kernel-only slot

- 现象：同一descriptor同时被kernel pointer row和model BootParam消费，却命名为`KernelABISlot`，后续设计误以为它等同
  runtime pointer-table storage或只适用于kernel launch。
- 根因：用某一个wrapper的承载形式给跨target consumer的entry argument命名。
- 修复模式：稳定语义名为`TileEntryArgument`；它记录ordinal、closed kind、target descriptor、bytes/alignment和access，
  pointer row或BootParam只是同一entry合同的不同provider lowering。不按vendor函数名分裂package/runtime架构，
  也不把当前adapter缺失写成vendor能力上限。
- 防复发：kernel/model两条wrapper都从同一个argument schema生成并readback；实现改名同批替换全部producer/consumer，
  不保留旧symbol或alias。

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

## 多reduction轴分块必须服从源浮点顺序语义

- 现象：矩形分块两个reduction轴后，chunk坐标被提到in-tile坐标之前，浮点`addf`求和顺序改变；没有fastmath的source也被
  当成合法候选。
- 根因：numeric gate只识别combiner种类，没有比较materialized traversal顺序，也没有读取源`reassoc`语义。
- 修复模式：compact traversal明确判断是否保持源lexicographic reduction顺序；前置reduction轴未unit-tiled时，后轴split会
  重排，只有源`arith.addf`带`fastmath<reassoc>`才允许。候选生成也用同一规则逐adjacent breakpoint推进，避免none先生成
  必然被exact gate拒绝的baseline。
- 防复发：两轴非整除case先验证无`reassoc`原子拒绝，再验证源显式授权后的nested offsets、tail coverage和numeric accumulator；
  单K GEMM仍必须证明顺序保持且可split。

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

## 单状态baseline不得套用多候选winner重物化

- 现象：唯一exact-verified Llama baseline在selection后又完整执行一次CardModule、16-Tile Instr、SPM、DDR和resource verification，额外消耗
  数分钟，但产物语义没有变化。
- 根因：为search cohort控制峰值内存而清空每个accepted candidate Tile module的策略，无条件复用到了只有一个semantic state的
  `none` controller。
- 修复模式：search继续只保留comparison summary并重物化winner；`none`保留已经通过全部exact verification且已清除query-local source relation
  的唯一executable，selection直接move返回。统计必须明确baseline rematerialization为零。
- 防复发：none与search unit分别断言0次和1次selected executable rematerialization；模型规模timing检查结果返回前不再出现
  第二轮16-Tile exact pipeline。

## 分配器carrier失败不得反向删除logical placement

- 现象：logical demand analysis已经用`IndexRelation.image()`正确求出consumer需要的producer集合，但placement evaluation
  随即把它降成dense rectangle/layout fragments/route；当前carrier表达失败被当成spatial placement非法，导致搜索域仍被悄悄缩窄。
- 根因：layout-independent demand与physical representation/movement materialization没有形成调用边界，analysis“存在”被误报为
  production合同已闭合；memory/edge planner又同时承担候选生成和准入。
- 修复模式：placement transition只消费logical demand与ownership coverage；layout、fragment、route和transport只在对应
  physical坐标关闭后物化。任何carrier失败只拒绝包含这些坐标的candidate，并把typed rejection交回唯一search owner；
  lowering、SPM/DDR planner和communication verification都不能修候选或直接操作candidate search。
- 防复发：用同一logical placement构造至少两个representation/movement alternatives，其中一个carrier失败、另一个actual
  accepted；断言spatial state仍可回溯并找到accepted winner。任务状态必须核对production调用链，不能只凭analysis单测标done。

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
