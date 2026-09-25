# Bug模式：构建、验证与文档维护

按主题记录可复用根因；入口见[bug索引](../bugs.md)。规则冲突时以当前编号设计与[AGENTS](../../AGENTS.md)为准。

## 归档完成历史不能带走未完成work item合同

- 现象：current plan仍列出pending任务和覆盖矩阵，但算法展开、输入等价类、typed failure、精确断言及直接下游witness只在archive；
  后续实现不得不读历史。只补回任务行和矩阵后，choice domain、handoff及query/apply顺序仍可能缺失。
- 根因：按整份文件的完成属性或篇幅归档，没有逐段区分有效规范与已结束施工；以任务名、标题或矩阵仍在误判合同完整。
- 修复模式：逐item核对输入、显式choice/raw domain、derived facts、actual apply、失败与transaction、输出和direct witness；
  稳定语义归编号设计，实施与逐项门禁归current plan。只归档旧调用链、删除账本、动态数字和结束记录，不恢复shadow schema。
- 防复发：建立旧current规范段落到新current owner的映射，逐段对照。新任务不读archive也应能正确实现；
  任何有效规范只在archive出现即停止迁移，不能用总行数、链接可达或恢复矩阵代签完整性。

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

## 读取源码marker的测试会把退役实现伪装成合同

- 现象：source或test没有进入active CMake/lit/CTest执行图，但一个已注册测试逐文件读取其文本并断言旧symbol marker存在，整体仍显示green。
- 根因：把“仓库里还有某段源码”当成“能力已被编译并经过行为验证”，source inventory又只检查已知子集，形成互相放行的假闭环。
- 修复模式：组织检查以filesystem、CMake source、unit/lit/CTest registration和明确的current-task dormant owner做双向集合闭合；
  能力测试只验证编译后的接口和行为。退役实现独有能力先逐项列出proof、materializer、diagnostic和negative witness，迁入active
  owner并受测，再删除源码和marker断言；“未注册”只能证明当前没有执行，不能证明源码没有独有能力。attention normalization曾发现未注册的
  attention alternative仍独有current-SSA资格证明和online/split actual-root构造，正确顺序是迁入`Planning/Search`后再删除旧源。
- 防复发：新增源码或测试时，checker fixture分别覆盖unregistered source、unregistered test、stale registration和无owner dormant
  四类negative；禁止用源码文本marker作为build/behavior contract。

## Board probe helper签名变化必须覆盖所有runner

- 现象：共享package helper改为返回`module_path, resource_ids, slots_per_tile`后，常用raw probe已迁移，但NE tail和engine pipeline
  等未注册runner仍按两个返回值解包，直到重新接入no-card才失败。
- 根因：Board source/catalog存在，但未作为current CTest执行；helper调用方清单与CMake执行入口没有一起更新。
- 修复模式：接口变化先用`rg`枚举全部调用方并同批迁移；每个可直接生成current package的raw probe至少注册一个代表性
  `current-interface` no-card CTest，catalog只拥有case语义，CMake只拥有执行入口。
- 防复发：fresh CTest精确运行`-L current-interface`，并验证inventory列出的host/no-card/Board CTest都在CMake中真实注册；
  不用未注册的CTest形状字符串代替执行证据。

## 非默认gate的lit期望静默过时

- 现象：较少运行的工具、model或可选入口测试仍断言旧计数、diagnostic、pipeline参数或接口签名。
- 根因：行为和测试分开更新，局部验证没有覆盖全部注册consumer；历史退役表示继续留在fixture中。
- 修复模式：核对行为变化与测试更新的Git先后，结合pinned API确认current合同；按实际注册路径运行受影响case，
  无关遗留失败明确登记。重型枚举用listing、filter和timeout定位，再修复其owner或迁移到有界oracle。
- 防复发：同步搜索所有测试目录；skip、未注册、未构建或排除的case不能算通过，也不能以“目录不在CI”保留陈旧期望。

## 全量单测中的单个枚举case可能耗尽CPU

- 现象：聚合测试长时间无输出且一个或多个进程持续100% CPU；单独列举case后可定位到placement/search枚举fixture。
- 根因：test domain没有明确状态规模，逐节点extension、pairwise topology query或Cartesian state使工作指数增长；stdout缓冲又让
  整批看起来像hang。旧case未进入默认gate时更容易长期失察。
- 修复模式：先用test listing、filter和timeout定位具体case，再以更小同构fixture、独立reference enumerator或显式状态上界保留
  semantic witness；已退役枚举实现和only-purpose tests随owner删除，不反复延长timeout。
- 防复发：新增枚举测试声明domain size或预期state数，真实规模production coverage与tiny exhaustive oracle分开；
  聚合suite运行前确认filter实际匹配并报告被排除的已知重型case。

## Public signature变更必须覆盖全部active consumer

- 现象：局部target编译通过，model、runtime、工具或其它active consumer仍使用旧public API。
- 根因：只构建当前修改的target，没有按CMake和注册入口检查全部直接调用者。
- 修复模式：从current CMake/test ownership列出active consumers，同步修改签名和调用；在唯一canonical `build/`内完成
  受影响target、public header/link检查以及无target的完整增量构建。只有配置输入变化时才重新configure。
- 防复发：记录实际构建与执行的consumer，第二次构建应为Ninja no-op；未启用的外部board SDK资格保持明确限制，
  不创建第二主工程，也不把局部target通过写成全部调用者通过。

## CMake多OUTPUT custom command不能靠第二个同OUTPUT命令补文件

- 现象：resources自定义命令把linker script和SPMD helper symlink都列进第一个`add_custom_command`的OUTPUT，再为helper单独
  建一个同OUTPUT的命令；ninja只生成第一条rule，helper资源目录存在但symlink从不创建，编译在target阶段才以
  `device link failed`暴露。
- 根因：同一OUTPUT出现两次时CMake只保留第一条custom command，不报错也不合并；目标级`add_custom_target`依赖的是
  OUTPUT名，无法区分。
- 修复模式：一个custom command的OUTPUT必须是它实际产出的全部文件；不同产物分属不同command，各自带真实DEPENDS；
  reconfigure后直接`ninja -t targets`核对产物rule存在。
- 防复发：新增build-tree/install资源时，先检查目标输出文件是否真的生成（`ls`/`test -f`），再测消费它的编译路径。

## 局部capacity probe与complete-candidate gate会产生不同scope和witness

- 现象：baseline先用root/region/function级scratch IR判断SPM，再重新构造完整TileModule set；局部路径可能报告fit或要求扩大scope，
  最终actual memory/target gate却在函数级packing失败。relation remap还可能在两次构造间省略或保留不同owner，使同一allocation得到不同归因。
- 根因：把“调用相同pass/checker”误当成消费同一actual IR和同一current relation certificate。只要第一次IR被销毁、第二次重建，
  operation lifetime、buffer relation、region boundary和packing scope就已经是两个事实源；继续增加scope escalation只会扩张平行链。
- 修复模式：删除baseline的root/Tile/function局部capacity probe。每个进入memory/target gate的candidate只产生一份完整TileModule set；actual
  planner demand通过candidate transaction的result/operand/output/movement/scratch relations归因，缺owner就是contract failure，不能按
  region恰有一个root猜owner。
- 防复发：显式test work counts证明每个完整物理choice/IR epoch至多生成一份actual-gate TileModule set，且
  `actualGateCandidates == actualGateInvocations`、accepted rematerialization为零；测试扰动
  每类relation并检查所有actual SPM demands有owner，不以Location、空relation或diagnostic字符串证明一致性。

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

## current package不是compiler IR归档

- 现象：FP16 LLaMA `optimization-none`已经完成编译并生成current package，PyTorch no-card runner却继续读取
  `package/functions/forward.mlir`，因此在真正runtime payload准备前报文件不存在。
- 根因：runner混淆了两个边界：source program拥有frontend MLIR和`functions/forward.meta`，executable package只拥有manifest、
  target modules和program data。旧测试把曾经存在的package内IR dump当成runtime合同，迫使普通编译保留无consumer的调试产物。
- 修复模式：boundary input locator和shape/dtype元数据从source program读取；runtime port、module和resource只从current package
  manifest读取。删除依赖package内structured IR的validator和case字段，不把IR dump重新塞回package。
- 防复发：普通source-to-package测试断言package没有`functions`目录；runner unit分别传入source与package并检查各自消费边界。
  compiler IR、计时和work统计只能由显式diagnostic选项或caller-owned sink请求，不能成为默认编译路径或package成员。

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

## Wafer与pinned LLVM必须使用一致的assertion ABI

- 现象：Release Wafer在析构包含`llvm::Statistic`的pass时出现内存破坏；相同源码的局部逻辑和Debug运行正常，调用栈落在与本次变换无关的
  pass storage。
- 根因：pinned LLVM以assertions enabled构建，而Wafer Release单独定义`NDEBUG`。LLVM headers中的assertion-sensitive class layout与已链接
  library不一致，形成跨库C++ ABI mismatch。
- 修复模式：加载pinned LLVM/MLIR CMake package后包含`HandleLLVMOptions`，让Wafer target继承同一assertion compile flags；不能在单个
  source或测试上局部增删`NDEBUG`修补症状。
- 防复发：fresh Release compile command必须与pinned LLVM assertion配置一致，并运行至少一个创建/销毁pass statistics的linked smoke。
  遇到跨pass随机析构损坏时先比较LLVM package flags和consumer flags，不猜线程、allocator或IR ownership。

## IREE current PartialReduction实现不能直接复制到较旧pinned MLIR

- 现象：按IREE current online-attention设计准备实现`PartialReductionOpInterface`时，仓库pinned接口没有
  `getPartialResultTilePosition`；其SCF driver按完整iteration rank索引每个partial result，无法表示Accumulator的output map与Maximum/Sum的
  row map这三种不同rank。
- 根因：只核对了op/interface名称，没有比较pinned TableGen methods和driver如何计算result offsets/sizes；把upstream新接口能力误认为本仓已有。
- 修复模式：保留`attention -> 三结果online_attention`的IR分层，但在当前pinned版本用stateful `TilingInterface`切K2：每个tile消费并
  返回三个DPS state，`scf::tileUsingSCF`直接形成serial loop-carried recurrence；FD spatial merge由actual SSA/Linalg显式物化。
- 防复发：采用外部compiler实现前同时核对op traits、interface TableGen和实际driver；文档不能只写“使用standard interface”。若升级pinned
  LLVM，必须整体切换producer/consumer/tests并删除旧driver，不能维护版本分支。

## Aggregate slot替换必须沿current view链更新composed layout

- 现象：recursive-doubling把原始SPM allocation donation给带非零offset的aggregate slot后，嵌套`memref.subview`仍保留旧的zero-based
  result layout；16-Tile LLaMA FP16/BF16 search在movement gate报`mismatch of result layout`，而none路径不触发该替换。
- 根因：`replaceAllUsesWith`只替换SSA source，不会重算view result type；movement preflight还持有这些SubViewOp，直接删除并重建会造成悬空operation指针或double erase。
- 修复模式：movement transaction完成所有root替换后，按current module中的实际SubViewOp source type逐项归一化同rank result的composed
  offset/stride；保留operation identity，避免破坏仍被transaction消费的endpoint指针。替换外部工具和CMake环境也必须保持typed/可验证边界。
- 防复发：recursive-doubling 4/16-Tile、1024/1025/1031矩阵在movement后立即`mlir::verify`，并检查带非零aggregate offset的嵌套subview；LLaMA
  FP16/BF16 search fresh no-card必须完成package/readback，而不是只看到candidate不再crash。
