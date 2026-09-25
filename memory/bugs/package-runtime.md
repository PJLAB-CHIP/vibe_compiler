# Bug模式：Package、runtime与执行模型

按主题记录可复用根因；入口见[bug索引](../bugs.md)。规则冲突时以当前编号设计与[AGENTS](../../AGENTS.md)为准。

## Runtime身份迁移必须同时覆盖profiler报告reader

- 现象：普通package/no-card与设备采集都完成，最终报告生成却拒绝output validation字段。
- 根因：C++已按external output PortId写evidence，Python reader/schema和手写fixture仍使用旧resource scope/role；
  fixture与reader互相验证通过，未覆盖真实producer输出。
- 修复模式：reader/schema/fixture统一消费current port，重复身份只按port判定，旧字段明确拒绝，不保留双reader。
- 防复发：身份迁移沿manifest、C++ evidence、schema、reader和真实report生成闭合；手写fixture不能代签producer/consumer边界。

## 板端probe的manifest通过不等于实际ELF和初始内存有效

- 现象：no-card通过，但SDK在entry-resolve找不到kernel；或计算结果正确、guard整片不匹配。
- 根因：替换probe ELF后只更新digest，遗漏current manifest要求的export；同时把新分配的output当作已有canary来源，
  从未建立初始值。Host allocator残留数据会进入SPM并掩盖guard来源错误。
- 修复模式：probe发布前从实际ELF的动态符号表核对manifest export；fixture入口与current runtime合同同步。
  Probe自己初始化完整output与未触碰区；Kcore写入供DMA读取的cacheable DDR须走已验证的cache维护合同。
- 防复发：编译current fixture并加入旧export的真实object负例；no-card准备本轮输入/reference，上板同时检查完整数值、
  guard、status和正常cleanup。不要把mock profile通过、package digest或历史allocator内容当作设备资格。

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

- 现象：target entry argument曾使用含义不清的旧名称，后续设计误以为它拥有runtime pointer-row storage，
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

## Logical raw-exact不能用不同padding模板的整块digest代签

- 现象：formal与backend的每个logical element逐bit相同，但qualification又比较从零模板打包的formal storage和backend
  storage整块digest；backend只改了不可观察padding，freeze却报raw-exact digest不一致。
- 根因：把logical tensor comparator与physical storage repeatability混成一个判据。
- 修复模式：logical raw-exact逐元素比较；需要对比formal/backend storage digest时，把formal logical values覆盖到backend最终
  storage同一padding模板。backend完整storage digest仍独立冻结，用于同一backend执行的repeatability。
- 防复发：选择带physical padding的形状，分别验证logical bit差异必须失败、仅padding差异不改变tensor数值结论、backend
  storage漂移仍由repeatability validation捕获。

## Repo内同步接口不应各自递增版本

- 现象：frontend nested metadata、profile plan/site、dependency record、workload和内部算法名称分别携带版本，修改同一语义时
  需要跨多层同步数值并保留旧分支。
- 根因：把源码revision内同步演进的内部表示误当成独立兼容边界，用版本号代替exact field contract和集中parser检查。
- 修复模式：先列出真实producer、consumer、存储和部署生命周期；独立文件或ABI保留一个current identity与集中检查入口，
  其余表示原位修改并同批更新所有调用方，旧输入在唯一边界入口fail closed。
- 防复发：新增版本前必须写明独立producer/consumer、支持周期和兼容测试；内部field、算法、hash domain、model和helper
  metadata不得使用`vN`名称或双reader。

## 旧Board executor删除前必须迁移观测合同

- 现象：旧executor因manifest、CLI或SPMD carrier退役而不能运行，直接删文件会同时丢掉model-scale source、full-output oracle、
  status/guard、重复完成或profile采集要求。
- 根因：把“执行接口已退役”误当成“校准问题已无价值”，没有按source、lowering、package、runtime和board边界拆解。
- 修复模式：能经current global lowering生成package的case迁入current no-card/Board CTest；不能生成的case保留current source、
  deterministic oracle和完整待执行要求，并在catalog/inventory绑定具体blocker。只有这些内容已有current owner后才删除旧executor。
- 防复发：逐个核对删除文件的source、shape/dtype、payload、numeric oracle、structural checks、status、timeout、cleanup和profile要求；
  inventory测试必须能从每个保留case解析到真实source/catalog与CMake入口或明确blocker。

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

## 发布点之后不能再运行会翻转事务结果的validation

- 现象：staging目录先rename到最终路径，随后installed-root readback失败并返回错误；cleanup只覆盖staging，最终目标仍可见。
  profile用两个顺序rename时还会短暂或在进程退出后永久暴露ordinary-only状态。
- 根因：把单次rename的原子性扩张成整个多产品事务的原子性，并假设post-commit readback“只会因compiler bug失败”。任何真实I/O、
  并发mutation和内部不变量错误都仍是可达错误边，两个各自原子的rename也不是共同原子commit。
- 修复模式：所有可失败validation和identity binding在唯一visibility point之前完成，全部共同产品位于一个可单次发布的owner/root；
  若协议确需installed-path动作，显式建模committing/committed状态和可恢复操作，并证明每个错误出口不泄漏本轮目标。
- 防复发：failure injection必须覆盖最后一次发布前后、installed readback和进程/第二产品边界；同时断言返回status、ordinary/profile
  可见性和staging残留，不能只测第二次rename正常返回失败时的best-effort rollback。

## Per-root TileRegion不等于可执行Tile entry

- 现象：root-to-movement stages的TileModule set包含正确private root/relay functions和TileRegions，局部tests全绿；首次由search送入actual memory/target gate时却因每Tile没有
  唯一public entry失败，临时用`func.call`组合又被DDR planner的跨call scope合同拒绝。
- 根因：把“每个root已物化”误当成“每Tile program已闭合”，root function的source boundary/result对应关系没有作为同次construction
  typed结果保留，导致后续只能猜名字/顺序或遗漏entry。
- 修复模式：root construction返回source-argument/structured-node-result keys；Card assembly按keys拓扑选择ready stages，把single-block
  stage body直接move进唯一public entry并用current SSA传递local intermediates。无work Tile构造同signature empty entry；不引入call、
  replay、ordinal或跨passside table。
- 防复发：root-work construction independent actual oracle不仅数TileRegion，还必须检查每Tile恰一public entry；production search让每个complete candidate
  进入一次actual memory/target gate并通过DDR/call-closure/program-resource gate，最终只保留winner。private functions存在或局部verifier通过不能代签
  executable boundary。

## TileRow参数的可见性不由顶层launch packet递归保证

- 根因：TileRowPointerTable的顶层packet只有DDR row地址；固件invalidate packet后，wrapper仍可能从Kcore cache读取旧row内容。
  参数行是Host→Kcore域交接，tensor DMA和DDR publication不能代替它。
- 修复模式：wrapper读出选中row地址后、任何slot load前，以ABI中的实际slot数invalidate整个row并执行fence/sync；
  TileMajor参数直接位于packet内，不增加间接row操作。使用invalidate而非写回旧cache内容。
- 防复发：检查row地址load→exact byte-range acquire→slot loads的LLVM顺序；连续运行不同整除/尾长的完整PyTorch产品矩阵，
  保持同一设备会话并覆盖参数行和allocation地址复用。

## 设备完成期限不能覆盖主机profile报告

- Profile入口包含Primary、Count、Trace设备执行及后续Python报告。将设备watchdog加退出余量作为整个进程期限，会在设备正常完成后误杀报告，
  还可能留下报告子进程；主机报告超时不能据此判定板卡异常。
- 每次launch保留runtime completion watchdog；主机报告遵守自身处理合同，不套用设备派生期限。主机后处理失败与真实设备timeout分别记录，
  只有后者按板端规则停止批次。测试检查ordinary/profile都传设备期限，而profile不设设备派生的总进程期限。

## 全卡共享资源身份不要求每个 Tile 都有入口参数

- 根因：为保持各Tile参数行同形，把共享payload及通知storage绑定到所有entry，无关Tile仅标记access=none；
  这些无用槽继续进入LLVM/manifest/runtime，放大编译工作及package记录数。
- 修复模式：在payload/notification物化处按实际writer/readers创建binding；ResourceId表示共享身份，ordinal只表示本entry位置。
  Target校验同一ResourceId的存储描述，wrapper和runtime按实际行长计算prefix或indirect row范围。
- 防复发：真实规模稀疏fanout检查无关Tile零binding、参与者exact coverage；不同长度行检查offset与invalidate字节数；
  shared descriptor冲突、额外或缺失ready binding拒绝。Package仅为资源分配一次，板测以完整PyTorch输出确认地址链。

## 不同Tile的不可变输入集合不能按整张binding表比较

- 根因：Target ABI已经允许各entry消费不同literal并删除未使用常量，Package writer仍要求各Tile的完整
  ProgramResourceBinding列表相等，合法主块/尾块在target lowering和link成功后才被拒绝。
- 修复：全卡相等检查只覆盖caller-visible input/output；不可变数据按各entry的binding解析ProgramTensorId，
  复用现有ProgramDataHandoff、descriptor与payload join，不补造常量槽或按其它Tile的ordinal归因。
- 防复发：不同entry常量集合、同源多representation和完整physical bytes共同覆盖；外部端口global shape冲突仍拒绝。
  真实rank4、长度1026的双dtype decode经source/package/no-card与全量实卡通过，不能只测试aggregate LLVM接受不同参数行。

## 地址范围证明必须保留整数位宽与比较语义

- 根因：只分析循环index及数学整数加减，会丢失integer/index cast之前的实际clamp，或错误穿过trunc/wrap。
  未知load不能提供具体值，但其整数类型有完整值域；后续实际clamp可独立证明地址范围。
- 修复模式：固定宽度SSA用pinned `InferIntRangeInterface`传播signed/unsigned区间，未知leaf保留完整类型范围；
  用局部迭代postorder复用共享SSA，checked index循环路径继续单独处理。区间仅证明地址安全，不作精确元素需求或SPM合法性。
- 防复发：穷举小位宽位型核对cast、wrap和夹界，配合真实规模view检查最终LLVM字节地址及越界拒绝。
  Unsigned比较`ugt(x, 0)`允许负数位型，不能据此将signed地址范围收紧为正数；只在两端已非负时复用signed区间比较。

## Loader外部符号必须来自实际固件导出

- SDK头文件中的日志函数不一定存在于当前固件RTMSymTab；将声明直接加入allowlist会让主机链接成功、设备加载失败。
- 按匹配身份的固件导出表核对allowlist和最终ELF全部undefined symbols；实际链接负例应在原子发布前失败并保留旧输出。
  日志函数存在与系统日志能采集到其输出是两个检查，不能互相代签。

## 厂商SDK函数表由模块生命周期持有

- 当前SDK的`TsmNew*`会分配函数表，逐指令创建/释放引入重复Kcore工作。CRT可通过`g_intrinsic()`借用厂商模块表，
  packet仍保持调用局部，不能缓存地址或改变completion。
- 接入前核对实际安装固件对`module_init`/`module_cleanup`的调用；device linker必须动态导出这两个真实hook。
  `--exclude-libs,ALL`可能将hook隐藏，仅在链接命令里写export选项不够；检查最终ELF的dynamic definitions。
- 输入重定义hook、缺失hook或未导出loader符号都必须在原子发布前失败；普通、Count、Trace和extra object入口共同覆盖。
