# Tile职责分拆与通用循环流水实施方案

本计划归`board-testing`，状态和直接前置只由[progress](../progress.md)拥有。
职责分拆、通用循环流水和指定BF16产品验收已完成，本文保存当时的实施合同与验收证据。
整体板测任务及后续性能优化仍由[性能计划](../plans/board-workload-matrix.md#已有改动的收尾与接续)接续；
本记录不代表全部attention优化完成，也不把流水可用等同于性能改善。
稳定语义由[10号第9节](../10-compute-movement.md#9-循环流水与存储优化的职责边界)拥有，源码组织遵循18号，
作用域及变换规则遵循19号，完成与内存分别遵循11/09号；本计划不建立第二套IR或同步协议。

## 1. 输入、产出与范围

- 输入：分拆前`ExecutionStructure.cpp/.h`及全部直接调用者，candidate-owned physical Tile IR、现有registered tests；
  通用流水验收另含普通分块计算与当前BF16 causal attention的fresh source/package。
- 职责：先按实际功能拆源码和API，再在同一流水实现上扩展动态域、条件、访问依赖和缓冲复用。
- 输出：职责明确的transform实现、唯一driver调用链、显式SCF/memref/Tile流水、对应测试及验证记录。
- 直接下游：原Tile→Instr、fresh worker/order/completion、actual SPM/target/package与runner。
- 用户入口：保留现有none/search产品入口；选择器、focused测试与production消费同一实现。
- 非目标：本项不扩大head/tile搜索，不引入新的数值处理、原生转置或跨Tile协议流水；也不顺带重构其它编译器模块。
- 完成条件：先通过拆分的行为保持门禁，再通过通用流水矩阵及目标实卡；两部分分别提交和说明验证边界。
  只完成源码拆分不能标记通用流水或整个attention优化完成。

此前已修改的tail、输出融合和causal prefix草稿先分别补齐功能验证并提交，累加器方向结合输出融合结果完成取舍。
职责拆分以这些已完成改动组成的版本为对照基线，记录代表IR和测试；不在拆分提交中混入新优化。

## 2. 已核实的问题及归属

| 当前事实 | 原因/影响 | 归属 |
| --- | --- | --- |
| ExecutionStructure同时包含load选择、schedule检查、SCF变换、轮转allocation和多类copy优化 | 名称掩盖职责；变换间顺序和失效边界不直观 | 源码/API组织，第一步处理 |
| `findLoadPipelines`要求静态trip count≥2，body存在region就拒绝 | 动态上界和条件循环没有进入流水候选；不等于硬件不支持 | load资格与循环控制，第二步处理 |
| `materializeRotatingAllocations`按静态次数选择最后槽 | 不能直接用于动态次数或空循环后的observable结果 | 轮转buffer物化 |
| 已物化pipeline的operation被copy优化排除 | 当前排除保护已有schedule引用；单删排除条件会造成失效，而完全跳过又妨碍优化组合 | 第二步调整共同输入和变换顺序 |
| `kernelDynamicTripCount`实际保存静态数字 | 动态次数需要IR SSA与显式unknown统计，0不能承担两种意义 | 循环结果/统计合同 |
| causal loop中的full/boundary分支 | 算法位置关系尚未充分暴露成普通循环 | 上游attention visibility，不进入通用流水matcher |

迁移前的实现在driver内直接调用，不存在名为ExecutionStructure的独立注册MLIR pass。
现有规则总体按current SSA、maps、effect和alias判断；“已有通用规则”与“覆盖完整”是两件事。

## 3. 第一步：按职责拆源码和调用链

下列新增Tile实现文件留在现有`WaferTileTransforms` component，driver仍归原Driver component。
保持现有调用顺序、适用条件和输出，
不在这一步改变选中候选或pipeline stage。API随职责迁移，driver只负责编排。

| 目标文件/现有owner | 从现文件迁入的内容 | 输入 → 输出；直接消费者 |
| --- | --- | --- |
| `LoopPipelining.cpp` | schedule preflight、SSA/effect依赖验证、SCF pipeline/finite-unroll、phase结果检查 | live loop＋显式stage → 同层实际循环；load流水入口/driver |
| `LoadPipelining.cpp` | `findLoadPipelines`、load地址依赖收集、只读use检查、提前读取的stage选择与组合调用 | physical loads/consumer → 当前IR上的选择及统一变换结果；none/search |
| `RotatingBuffers.cpp` | allocation preflight、槽创建、轮转view/select、最后槽及owner更新 | actual root＋loop＋显式multiplicity → actual allocations/SSA；循环变换 |
| `StorageOptimization.cpp` | private scalar读取、GEMM/elementwise写回转交、private publication、last-use复用、loop destination确定 | actual use/alias/effect → 改写后的Tile IR；driver下游 |
| 现有`StorageInitialization.cpp` | 保留已有初始化内容与覆盖证明 | actual private storage → 删除无观察者的初始化；存储优化 |
| `MovementFusion.cpp` | 从拆分前已验收的transpose＋layout融合实现迁入，保持行为 | 两段实际movement → 一段等价movement；Tile→Instr |
| 现有`PhysicalMovementPlacement.cpp` | 保持搬运位置调整的职责 | actual movement/loop invariance → 移动后的operation；driver |
| 现有`CurrentIRExecutablePipeline.cpp` | 组合上述变换并保留typed失败传播 | 同一candidate owner → 后续fanout/Instr；production driver |

不按每个小helper或每种算子单独拆文件。Schedule legality与机械变换先同归LoopPipelining；一次性load证明留在
LoadPipelining私有实现，不仅为了拆文件就新建AnalysisManager cache或公共analysis library。
跨library稳定API放对应public header，单component工作结构和helper留在`lib/`。
旧万能入口及旧header的直接消费者迁完后移除，不保留新旧实现或转发wrapper；原typed failure分类、有限展开预算保持。

迁移顺序：先独立storage与rotation，再独立load选择，最后收口loop变换和driver调用。
每次只搬一个完整职责，更新CMake显式source和直接include。测试按相同职责迁移，保留原exact/negative断言，
不能用“新文件能编译”代替旧能力验证。原文件及consumer都清理完成才签分拆完成。

**第一步验收**：

- 既有循环、rotation、storage/alias、初始化及直接下游测试实际执行；typed失败类别不变。
- 固定的代表输入比较拆分前后Tile/Instr/target IR，忽略的字段仅限事先列明的非语义路径/计时；
  不归一化掉operation顺序、descriptor、offset、owner或同步差异。差异须解释并修复，不用宽松比较掩盖。
- 唯一CMake owner、public header自包含、旧入口残留、source organization和`git diff --check`通过。
- canonical完整增量构建及随后Ninja no-op；纯源码分拆不额外上板刷性能数据。

### 职责迁移对照与本轮检查

| 原能力 | 当前实现和正式调用者 | 对应测试 |
| --- | --- | --- |
| schedule preflight、SCF/有限展开、结果验证 | `prepareLoopPipelines`、`pipelineLoops`、`verifyPipelinedModule`；driver与LoadPipelining共用 | `LoopPipeliningTest`，4项 |
| load选择和组合物化 | `LoadPipelining.cpp`；none/search经同一driver入口 | `LoadPipeliningTest`及Driver capacity feedback |
| actual轮转槽/最后槽/typed owner | `RotatingBuffers.cpp`；load与显式槽调用者 | `RotatingBuffersTest`，真实MiniMalloc |
| scalar、GEMM/逐元素写回、publication、初始化及loop destination | `preservePrivateScalarBroadcasts`、`optimizeStorage`；driver调用 | `StorageOptimizationTest`，14项原机制矩阵 |
| transpose＋layout合成 | `fuseTransposeLayoutMovements`；driver调用 | `MovementFusionTest`，dtype/整除/尾块及额外观察者 |

旧header和万能入口已删除，不保留wrapper；原21项测试迁到五个文件，存储测试直接调用存储API。
纯迁移顺序保持 `pipeline → scalar → movement fusion → GEMM/publication/elementwise/reuse/initialization/destination → verify`；
已绑定流水对象的排除集合来自本次actual结果。第4节的共同输入调整尚未在此步实施。
两个只读/结果辅助函数留在同component的private header，不新建library、IR或analysis cache。
23个迁移leaf和6个入口函数体经token对照相同；两处祖先查找等价改用`Block::findAncestorOpInBlock`，原scope判断保持。
本轮新产品的16 Tile Dataflow、16 Instr与16 LLVM文件同分拆前逐字节一致，未忽略任何字段；纯文件迁移不新增板测。
五个public header分别独立编译、source organization及38项相关lit通过；canonical全量增量及no-op通过。
完整Driver、Compiler public-link smoke及修正后的完整Transforms均通过。首次迁移测试遗留一次`OwningOpRef`自move，
已删除该测试语句，生产代码未受影响；失败首次记录保留，不能将其CTest汇总冒充全通过。
本步证据见[分拆验收](../../docs/data/board-performance/attention-execution-split-20260922.json)。

## 4. 第二步：同一实现扩展通用流水

### 4.1 先让存储优化与流水组合

完成纯拆分后，再把storage优化、movement融合/placement和必要destination确定放到serial/pipeline分叉前。
从变换后的IR重建owner、alias与load依赖，串行和流水choice消费相同起点。
新插入的copy或缺少稳定destination必须在共同输入处解决，不通过“pipeline路径跳过全部copy优化”处理。
原已绑定stage的operation集合只能作为当前调用中的rewrite约束，不能跨mutation保存旧schedule。

### 4.2 循环域和stage物化

用lower/upper/step SSA描述迭代域；支持静态和loop-invariant动态边界、非零起点、非单位正步长，
动态正step须有current-IR证明。动态界包括由外层IV产生的上界，本项不同时扩展模型输入shape或ABI。
原`scf.for`的有序算术和iter_args接续保持，禁止为了流水重排归约数值。
接口保留显式stage数/slot数，本轮load策略首先选择提前一轮，已有多stage/finite-unroll能力继续回归。

算法选型已经对照官方[SCF pipeliner](https://mlir.llvm.org/doxygen/LoopPipelining_8cpp_source.html)
及[变换选项](https://mlir.llvm.org/doxygen/structmlir_1_1scf_1_1PipeliningOption.html)。
仓库pinned `mlir/Dialect/SCF/Transforms/Transforms.h`与`LoopPipelining.cpp`有schedule、dynamic支持和predicate callback；
Wafer需补的是地址/effect、predicate、buffer和直接下游合同，不能把上游新版本行为当作本地API。

采用同一个SCF变换实现；static/dynamic共享schedule和dependency验证。
动态边界的prologue/epilogue使用实际有效迭代predicate；能证明成立时消除，不能证明时保留最小合法执行scope。
不逐条机械套if，不无条件读下一块，也不为未执行的memref结果伪造零值或可读取的占位buffer。
涉及返回值的条件必须通过原init/实际taken path接续；在首次mutation前确认回调能表达该结果语义。
原pinned helper可能在失败后留下部分IR，失败candidate必须整体销毁，不能把部分结果交给下游。

提前一轮的语义例子（R表示读取，C表示计算，槽编号仅示意）：

```text
0步：无R/C，输出原init。
1步：R(0, slot0)；C(0, slot0)。
多步：R(0, slot0)；
      对每个确实存在下一步的i：issue R(i+1, nextSlot)，执行 C(i, currentSlot)；
      C(last, lastSlot)。
```

这只是依赖顺序，不是在此处发wait的recipe。是否同时活跃由最终issue/completion和硬件执行决定。

### 4.3 访问依赖、分支与嵌套

从SSA和标准view/effect/RegionBranch信息收集真实访问。对地址计算、load、consumer、store分别检查
同迭代及跨迭代RAW/WAR/WAW；alias根相同但范围可证明分离时允许，部分重叠/额外observer进入真实依赖。
读取地址依赖当前计算结果时只约束受影响的stage，不把其它独立load一起排除。

- 循环不变`if`：条件本身可安全提前求值时按同一条件形成外部选择，保留各自state和effect；
  空循环不能因此额外执行有effect或不可推测的条件计算。
- 迭代相关`if`：predicate随正确迭代的stage传播，保留conditional memory access；能够证明effect/SSA闭合的
  region可作为一个整体调度。不能因存在region就拒绝，也不能把无法解释的region当pure。
- 嵌套loop：优先对选定内层循环施加同一规则；其它子region按实际依赖处理。
  没有outer-pipeline选择时不同时重排外层，不为支持内层而强行展开整个nest。

不满足证明的形式报告精确的不适用原因；solver/work预算耗尽与语义冲突保持不同结果。
这里的coverage不足归为软件限制，禁止直接标成硬件不支持。

### 4.4 槽、完成、SPM与搜索接入

复用实际load root及原dtype/layout创建所选槽，立即登记typed owner。根据current stage中实际读写的关系验证
槽何时可以再次写入；尾部及循环后的reader引用实际最后槽，动态次数保留SSA，不推算未来lifetime inventory。
empty域不执行“最后一块”的选择/读取；不能让无符号`tripCount-1`在零步时溢出。

流水输出经原Tile→Instr、worker/order和fresh completion；现有same-worker有序保证足够时不增加steady-state join。
有cross-worker或reuse hazard时按真实range/token处理，不能用全局drain补齐分析缺口。
真实双buffer进入唯一SPM planner，只有实际capacity rejection才由controller调整选择。
none/search都接入同一实现，控制scope绑定依靠current typed coordinates和clone mapping；
全可见/边界拆分后也不能按ordinal/name重找loop，串行与流水使用同一优化基线和原预算。

### 4.5 Attention只作为上游适配和验收输入

复用前序已完成的05号visibility拆分：按actual位置与原循环网格拆全可见前缀和边界段，前者移除位置mask，后者保留原mask。
两段访问的并集必须等于原可见域，三项state按KV顺序接续。它生成普通SCF输入，
LoadPipelining/LoopPipelining不读取causal/head标志，不识别固定256/2048。
普通load→逐元素计算→store及load→GEMM→state也须经过同一入口，证明规则不依赖attention。

### 4.6 当前硬件事实与明确范围

跨engine可重叠有[板端证据](../../docs/tx81-current-profile-hardware-behavior.md)，具体收益仍取决于payload、
issue间隙、buffer及完成域；RDMA活动周期不是可直接扣除的总延迟。新增通用动态控制不需要改变厂商指令或timeout配置。
跨Tile DTE通信协议重排、loop-carried异步token和动态SPM allocation不在这两步中扩展，
原因是它们有独立的owner/completion/target合同，不能由本地loop变换替代；这不是硬件无能力的判断。
原native transpose禁用与GS分段合同不变。

## 5. 覆盖矩阵与完成证据

| 输入/分支 | exact要求或typed结果 | 直接下游witness |
| --- | --- | --- |
| 纯职责分拆；旧serial、SCF pipeline、finite-unroll | 适用集合、顺序、结果与失败类别不变 | 既有测试；代表Tile→Instr→target输出对照；canonical/no-op |
| rank≥3、主维1024/1025/1031；静态与动态界；非零lower、非单位/动态已证明正step | 每个有效iteration恰好一次读取/计算/写入，尾部无越界 | 解释实际SCF/Instr访问；数值reference与guard |
| 动态0/1、少于stage数及多步 | 0步init原样返回；1步无next读取；多步prefix/kernel/tail完整 | 实际执行次数、yield/state、最终slot及target address |
| 多个load、非attention逐元素/GEMM、只读共享源 | 只有满足依赖的读取提前；地址和dtype保持 | 同一production入口生成流水；全量数值 |
| loop-invariant条件、随iv改变的条件、嵌套循环 | 分支taken path、conditional load和loop-carried state保持 | 真实分支输入运行；无未执行buffer的读取；Instr/target |
| subview不相交/部分重叠、额外reader、loop-carried RAW/WAR/WAW | 无依赖可流水；冲突要求合法stage或typed不适用 | 当前alias/effect witness；不能只断言compile成功 |
| 未知effect/alias、非法stage、证明预算耗尽 | unsupported/contract failure/indeterminate区分；失败不发布部分candidate | verifier和负例；原输入未被scratch污染 |
| 已有多stage及有限展开 | 能力不因新增动态支持退化，stage/operation coverage正确 | 旧回归及对应actual lifetime |
| actual slots、动态最后槽、SPM刚好可放/真实超容量 | allocation都有owner，容量由同一实际planner决定 | 完成位置和token/participant、SPM offset或带demand的typed rejection |
| 无typed crossing的steady state | 无可避免的逐块join；必要reuse/observable完成保留 | 实际join位置和动态次数，直接lifetime witness |
| causal不同BQ/BK、query offset、主块/tail、S1024/1025/1031 | 全可见与边界无遗漏/重叠，mask和m/l/A接续不变 | visibility→普通流水→Instr/SPM→numeric；原型不能代签 |
| BF16 `[1,28,2048,128]`最终产品 | fresh source/package/no-card、全量输出、guard、completion、厂商正常退出 | 三次无采集普通计时；独立profile归因 |

0/1及短循环测试仍使用真实规模block payload；tiny数据仅在最小负例或有界独立oracle中使用并注明原因。
纯文件迁移复用已有测试；新增测试围绕第二步新增行为和实际访问，不复制实现算法充当reference。
测试、catalog、host build按canonical并行规则执行；设备逐case单进程，每次运行前检查全系统占用。

记录每项的实际work count、pass/analysis timing、wall/RSS；确认流水被选择、原重复copy未回归、
多槽真的进入最终IR。普通总耗时与profile分开，engine活动量不可相加，指令减少不代签性能收益。
最终若合法流水未获收益，保留本版本的实测与选型原因，不靠改容差、强制pipeline或扩大搜索预算包装结果。

## 6. 提交及接续顺序

1. 已有实现收尾：先完成VuVLoop尾块与输出融合的回归和当前版本实卡，再根据实际输出成本完成累加器方向取舍。
   两项可共用同一当前产品包验证，性能按组合结果报告；既有专项只对变动或未覆盖部分补测，不重复运行历史包。
2. 已有上游草稿：完成05号causal前缀/边界拆分的构建、exact覆盖、state接续和产品验证，独立提交。
   本步只生成普通串行循环；后续流水直接消费它，不再单独重做attention拆分。
3. 职责分拆：按第3节完成迁移和行为保持检查，独立提交；对照基线包含前两步已经完成的功能。
4. 剩余存储优化：按性能计划复查残余行状态发布复制，证明和处理仍可消除的部分，保留必要的旧值读取与独立目标。
   已验收的等字节序copy与初始化读取转交不重新实现；这一阶段结果进入串行/流水的共同输入。
5. 通用流水：先共同输入/依赖与动态域，再条件、rotation/completion/SPM和driver接入；按完整矩阵验证后提交。
6. K/V接入及最终性能：复用第2步可见域与第4步存储结果，完成fresh no-card、实卡与独立profile，记录实际收益和剩余瓶颈。

职责分拆的迁移与行为保持证据见第3节；通用流水扩展的实际完成边界见下节验收记录。
稳定设计不因排期改变；全部归同一board-testing，不新增work item，也不重新打开head/tile搜索。

### 通用流水验收（2026-09-22）

none/search及直接调用者已把storage共同优化移到流水选择之前；选择后不再复用旧operation绑定。
动态域和静态短域复用pinned SCF predication，conditional load保留原taken path，轮转槽的动态最后值由真实非空域选择。
嵌套候选优先内层，公共preflight拒绝重叠scope；此前父/子同时绑定导致改写后引用失效的主机崩溃已有独立回归。

访问依赖区分iteration-private allocation、同stage保序、只读共享、NoAlias、实际不相交的invariant subview与轮转槽。
动态per-iteration subview之间若缺少跨迭代距离证明仍返回Unsupported；这属于软件证明范围，不能解释为硬件限制。
无value的未知effect不静默跳过；只有Wafer接口明确的engine摘要及有对应value-bound内存effect的摘要可消去重复统计。

条件预取曾在card-ddr gate被误判越界：StaticIndexRange只消费与常量的比较，遗漏有界SSA条件，且分支收紧后没有重新交原IV网格。
11号共同地址分析已补这两项，保留unsigned非负条件和原DDR descriptor检查。非attention的rank3 1024/1025/1031条件搬运
经过独立SCF访问枚举、Instr、completion、SPM和DDR规划；动态0/1/多步、两槽/三槽的最终结果及空域初值已有执行见证。
仅有loop前初始化、逃逸view、重叠/未知alias和索引位宽溢出有typed负例。

专项StaticIndexRange及Loop/Load/RotatingBuffers实际通过；24组SystemC读取复用按CTest独立进程通过。
直接整进程运行SystemC参数矩阵曾因第二次elaboration被拒，失败记录保留，正式注册的独立进程结果才是有效证据。
Analysis、Transforms、Driver及24组SystemC共27项CTest全部通过；补充整数IV最后槽与原地consumer矩阵后，11项流水专项再次通过。
17项结构/同步lit及60项Instr/target lit通过。当前主块和1031尾块完成fresh no-card；最终LLVM、ELF及ProgramData
与本轮实测包一致，主块的Tile/Instr也逐字节一致。尾块早期IR有差异，不将其称作所有stage相同。

正式串行winner三次普通实卡4.141/4.086/4.077ms，尾块1.313ms，完整数值/guard/完成及厂商退出健康。
保留相同Q/mask准备复用的双槽流水actual候选4.105ms，未见净收益，正式选择仍为串行。
串行/流水独立profile为4.083/4.150ms，各77,280事件完整，动态调用数相同；CT/NE/TDMA活动量基本相同，
不能将RDMA活动量直接当可节省总时间。早期FirstUse候选6.114ms包含准备复用差异，不作为纯流水性能对照。
本步实现与指定产品板端验收已完成；后续用户新增的循环累加器布局修复属于08号独立边界。
证据见[流水验收](../../docs/data/board-performance/attention-loop-pipelining-20260922.json)及
[性能分析](../../docs/board-performance-results.md#2026-09-22通用循环流水与kv重叠验收)。

组合候选的起点问题已由本轮诊断确认：已接受的串行参数派生流水后，等待分支的发现列表被后来的同优先级参数
反复前插；后者容量失败也不会恢复原起点。修复限定为保留actual已接受来源的最低成本起点，不改width/trials、
head/tile域、SPM准入或候选变换。非attention driver回归及产品actual组合已经通过；来源成本只排序发现参数，兄弟自己的actual容量失败仍独立。
同成本保留先到点，不能将本轮产品见证冒充单独的调度状态机穷举测试。

进一步的具体拒绝证据为SPM轮转slot写入与DDR constant/global读取被通用AliasAnalysis返回MayAlias。
两者已有不同的Wafer typed地址域，不能要求它们再通过同域root/range证明；本地流水依赖检查已消费该事实，
rank3整除/tail的跨SPM/DDR global正例与原同域重叠/未知alias负例共同回归。
修复后保持准备复用的流水候选进入完整actual门禁并成功构包，串行仍胜出；以上实卡记录是取舍依据，估时不代替测量。
