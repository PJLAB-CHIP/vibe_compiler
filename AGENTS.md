# AGENTS.md

本文件只规定本仓库长期使用的协作流程和编译器工程硬约束。当前任务状态在`tasks/progress.md`，
具体IR、pipeline、runtime和ABI设计在编号设计文档中，实施步骤在`tasks/plans/`。不要把临时计划、
历史实现名或未确定方案写回本文件。

Wafer compiler、runtime、工具、文档和测试共同演进。任何修改都必须保持明确边界、唯一事实源和可重现验证。

## 优先级与请求范围

发生冲突时依次服从：

1. 用户当前明确要求。
2. 本文件的工作流程和硬约束。
3. 当前work item及编号设计的完成条件。
4. IR、ABI、runtime和编译结果正确性。
5. 仓库现有风格。

- 先判断用户要求是解释、审查、验证还是修改。解释或讨论不自动改文件；审查只做必要的只读检查；修改任务才写文件。
- 用户纠正或打断时立即停止无关工作，以最新要求为准。
- 能从仓库确定的实现细节自行处理；只有缺少的选择会改变设计方向或扩大授权范围时才询问。
- 不把一次状态汇报、局部测试通过或中间IR生成当作任务完成。

## 必须执行的工作流程

修改设计、代码、测试、构建或工具时按下面顺序执行：

1. 运行`git status`，识别并保留已有的无关修改。
2. 依次阅读`AGENTS.md`、`tasks/progress.md`和当前work item的编号设计/实施计划。只有查找编号或归档边界时才读
   `tasks/README.md`；涉及硬件、ABI、runtime或历史故障时再读对应`docs/`和`memory/`。
3. 将修改归入`tasks/progress.md`中的现有work item。小型维护可直接归入现有编号设计，不为它新建队列项。
4. 非小修在写代码前确认编号设计已经写明pipeline输入、当前职责、输出、直接下游、non-goals、完成条件和本项覆盖矩阵；
   缺失时先补设计。
5. 算法选择需要比较论文或成熟compiler实现；MLIR API先查官方资料，再以仓库pinned LLVM/MLIR源码和测试确认。
   不用本地实现现状代替算法调研，也不假设新版本upstream API可用。
6. 按最小IR或文件边界实现。说明输入是什么、实际产出什么、下一位直接消费者是谁。
7. 运行直接受影响的构建、测试和文本检查。代码、CMake或注册变化完成前，再运行一次不指定target的canonical增量构建；
   纯文档修改不机械运行无关编译。
8. 按设计和LLVM/MLIR工程规则复审完整diff，更新实际受影响的设计、状态和稳定经验，然后提交当前任务修改。

环境或外部依赖阻塞时，明确报告未完成部分、已经执行的检查和剩余风险，不降低feature或改走兼容路径。

## 构建与测试

### Canonical主机构建

- 仓库只有一个主工程build：checked-in preset`default`，binary dir为`build/`。不得创建任务、配置、日期或agent命名的第二build。
- `build/third_party`和configured helper是同一build的依赖产物，不是第二套主工程。
- 只有toolchain、preset、managed dependency identity或CMake配置变化时才重新configure同一build。普通修改使用增量构建；
  clean build只用于CI、配置迁移或明确的构建系统故障。
- canonical build开启compiler、importer、SPMD、numeric backend、SystemC和全部本地测试，关闭外部board SDK与真实设备执行。
  交付内容由同一build的install component选择，不建立runtime-only主工程。
- 主机构建、CTest、lit、catalog和no-card默认使用`nproc`并行。只有内存、共享写目录、resource lock或工具限制要求时才降低并发。
- `UNSUPPORTED`、skip、未注册或未构建的case不算通过。完成结论必须确认测试实际执行。
- Python测试和工具不得在Wafer-owned源码目录产生`__pycache__`、`.pyc`或`.pyo`。

### 测试覆盖

- IR、analysis、planning、rewrite、conversion和lowering正例默认使用rank至少3、至少一个主要迭代维度不小于1024的static shape。
- 涉及tiling、空间划分或循环生成时，成对覆盖`1024`等整除长度和`1025`、`1031`等非整除长度，并实际经过多Tile、
  多block/wave、remainder和tail。
- tiny shape只用于有界oracle、最小verifier负例、scalar/zero-rank或单点故障定位；测试中写明原因，同一机制仍须有真实规模正例。
- 正例不能只断言成功。按stage检查exact coverage、无重叠、owner、demand、merge、tail或直接下游可消费结果。
- 每个非小修work item单独写覆盖矩阵：输入等价类、整除/非整除、结构分支、typed failure、exact输出和下游witness。
  历史`done`、测试总数或单个case不能代替该矩阵。

### 板端验证

- 有板端要求的任务先达到`board-ready`：在无卡环境准备完整case、输入、reference、runner和`ExecutablePackage`，并通过本轮no-card。
  `board-ready`不是`done`；只有本轮真实板测通过才可标`done`。
- 真实设备始终单进程、逐case运行，只运行当前任务指定case。同一设备会话且软硬件身份未变时只确认一次环境。
- 普通纵向默认FP16或BF16；F32只用于F32格式、ABI、转换、数值边界或真实输入要求，并记录原因。上板前核对输入、descriptor、
  payload和expected的dtype一致。
- 每个case只执行必要的增量构建、单次launch、timeout、结果/guard检查和正常清理。timeout或设备异常后停止批次，
  不自动retry、reset或power cycle。
- 历史raw输出、日志和报告只供审计，不作新测试输入，也不代替本轮结果。

## 事实来源与文档边界

- `tasks/progress.md`：唯一任务状态、顺序和直接前置。
- `tasks/NN-*.md`：当前稳定设计和完成合同。
- `tasks/plans/`：尚未完成的实施步骤；完成后移入`tasks/archive/`。
- `tasks/archive/`：历史背景，不能覆盖current设计或状态。
- `docs/`：硬件、runtime、firmware和ABI事实；相关结论区分`supported`、`board-observed`、`unknown`和`excluded`。
- `memory/general_dev.md`：稳定构建/调试方法；`memory/bugs.md`：可复用的问题根因和防复发方法。
- pinned LLVM/MLIR源码和测试：具体API事实；官方文档：MLIR行为和工程原则。

发现冲突时在同一修改中收敛，不能静默拼接两套协议。没有新的稳定经验时不修改`memory/`。

## Pipeline与IR设计

非小修的设计必须覆盖以下边界，标题可不同，内容不能缺失：

```text
Pipeline position:
- Upstream IR / input:
- Current stage responsibility:
- Output IR / files:
- Downstream consumer:
- User-level driver / named pipeline:
- Explicit non-goals:
- Completion criteria:
```

- 用IR层或文件格式定义边界。任务号、临时阶段号和测试名只能作索引，不能进入CMake target、CLI、pass/pipeline、IR名称、
  diagnostic前缀或输出路径。
- 每层IR只保存该层能解释、修改和验证的信息。低层硬件事实可以作为上层legality或cost输入，不能变成上层语义。
- 下游无法从输入IR重算的必要事实必须进入SSA、region、operation、type、attribute、effect或明确interface。
- 新增op、type、attribute或Wafer interface前，先确认现有dialect、trait、标准interface、canonicalization和conversion不能表达需求，
  并指出实际verifier或lowering消费者。
- named pipeline和production driver必须调用同一个pass/transform实现，不维护第二条直接改IR路径。
- FileCheck、手写fixture、negative verifier和shape dump只能补充覆盖，不能替代真实输入到直接下游输出的验证。

### Current IR是唯一事实源

- 不得发明或猜测未来IR。未存在于current IR中的operation、SSA value、buffer、allocation、alias、movement、effect、lifetime、
  event、order或completion事实都是unknown。
- Search/planning可以选择明确的placement、region、tile size、layout或movement参数；选择一旦改变IR、control flow、memory或effect，
  必须在candidate-owned transaction中先实际物化并通过verifier。
- Candidate流程固定为`current IR → typed choice → actual transformation → verifier → fresh analysis`。成功候选继续持有同一actual IR；
  loser或失败transaction销毁，不按旁路plan重建winner。
- C++临时对象只能在一次调用中保存choice和工作数据，不能跨IR stage充当def-use、alias、lifetime、schedule、completion或memory事实源。
- 不用expected inventory、ordinal/name恢复、plan/actual parity检查或重放型verifier修补shadow plan；先修正materialization boundary和唯一owner。
- Cost model可以排序显式choice，但不能把推算的operation、buffer、instruction或resource inventory当作actual事实，也不能代替
  current IR上的legality、alias、lifetime、completion或memory检查。
- 默认不建立future-output IR。确有必要时须得到用户明确同意并建立编号设计；该IR自身必须是唯一authoritative representation，
  具有typed def-use、parser/printer、verifier、ownership/invalidation和唯一lowering路径。

### SPM合法性

- SPM合法性只由actual candidate current IR上的唯一SPM规划路径决定。输入必须包含实际allocation、layout、SSA alias、effect、
  completion和lifetime；只有成功生成并验证offset才是合法，只有带实际冲突demand的typed capacity rejection才是不合法。
- 没有实际IR和实际规划结果时结论只能是unknown。footprint、shape公式、buffer倍数、upper bound、synthetic demand或预测lifetime
  只能用于诊断/排序，不能进入admission、pruning、retile、fallback或winner合法性控制流。
- 所有policy消费同一actual memory/target leaf。外层controller处理capacity反馈；capacity、timeout、unsupported和compiler error
  保持typed区分。lowering和allocator不得自行retile、spill、
  换layout、换buffer或选择下一候选。
- 每个allocation必须有current-IR typed owner relation。materializer只能报告实际创建的buffer；出现无owner demand时按contract failure停止，
  禁止按shape、类型、位置或“唯一root”补归因。
- 测试使用真实规模actual candidate覆盖“物化→Instr→SPM规划→反馈”，并证明修改或删除估算逻辑不会改变SPM合法集合。

### Completion、join与wait

- 同步只能来自current IR中的typed effect/token/control-flow/lifetime和当前硬件/runtime/ABI文档。证据不足时结果是unknown，
  必须defer/reject或先补证据，不能插入“保守”全局drain、固定worker join或issue后立即wait。
- 如果当前硬件合同只要求同一worker保持issue order，不能仅因operation类别、loop backedge、region结束、state update或store/reload插join。
  只有typed cross-worker hazard、完成域crossing、actual release/reuse或observable terminal可以要求completion。
- Direct DTE token与NCC participant是不同完成域。recv wait在destination首次读取/复用前；send wait在source最后读取/转发/复用/释放前；
  wait精确消费对应dynamic token，NCC join和DTE wait不能互相替代。
- 上层算法、region/temporal construction和普通materializer只保留SSA/effect/token/lifetime要求，
  不选择participant或completion位置。最终completion stage在worker、order、movement、storage和control flow均物化后，
  从current IR重建minimum-strength、latest-unavoidable的join/wait；lowering只验证和发射。
- 测试检查join/wait位置、participant/token、动态执行次数和直接lifetime witness。没有typed crossing的真实规模正例应断言
  可避免的steady-state/non-terminal join为0；同步数量不得无理由随element、copy或attention block线性增长。

### Memory、alias与语义恢复

- View、allocation、destination mutation和materializing copy在进入低层IR前必须有确定语义；lowering不得根据users临时决定alias或allocation。
- Tensor层使用DPS和`BufferizableOpInterface`分析in-place/out-of-place；进入memref或Instr层后alias与effect必须已经确定。
- 不根据buffer名、operation名、symbol拼写、文件名或示例名恢复语义。使用type/rank/shape/dtype、memory space、indexing map、
  SSA use-def、region/control flow、effect和operation interface。
- 任务没有明确修改数值语义时，不分析或引入数值重排空间；保持现有算术operation和dtype，数值测试只作回归。

## MLIR工程规则

### IR、schema与verifier

- Pass输入和输出都必须通过verifier。verifier-valid输入触发crash/assert是pass bug。
- op verifier只检查自身、owned region结构及直接operand/result/attribute关系；call graph、topology、alias/lifetime、completion和resource
  closure在最近共同parent或显式validation pass检查。
- 影响legality、rewiring或lowering的字段使用ODS argument/property、type或typed attribute，并通过generated accessor访问。
- `Location`只保存source/diagnostic位置，不保存pointer、operation identity或跨pass语义。
- 优先实现适用的RegionBranch、Call、MemoryEffect、ViewLike、DPS、Bufferizable或Tiling标准interface；仅在标准interface无法表达且有
  明确消费者时新增Wafer interface。

### Pass与analysis

- Pass锚定到完成工作所需的最小operation。需要call graph、symbol closure、function-boundary bufferization或全局resource检查时才使用
  func/module范围；可独立处理的`IsolatedFromAbove`使用nested pass。
- `OperationPass`不得读取root sibling，不得修改root operand或parent block，也不得替换/删除自己的root。
- Pass不在多次`runOnOperation`间保留可变状态，不使用可变全局状态；创建新dialect对象时声明dependent dialect。
- Analysis只读current IR和显式只读target facts。IR mutation后默认失效；只有确实保持有效时才preserve。
- 只有被多个pass重复消费且可安全失效的事实进入`AnalysisManager`；一次性validation、choice和跨clone工作数据留在当前调用。
- atomic pass完成一个可验证变换；pipeline只组合稳定IR边界。搜索、module fan-out、外部工具和目录提交由compiler driver负责。

### Rewrite、conversion与clone

- `matchAndRewrite`确认匹配前不修改IR；所有修改经给定`PatternRewriter`。首次修改前完成shape/type/symbol/target等可能失败的检查，
  失败使用`notifyMatchFailure`。
- rollback期间不修改外部容器或其它无法回滚状态；返回success时必须已经修改IR。
- Full conversion必须使所有输入op对`ConversionTarget`合法；partial conversion只处理明确illegal集合并紧跟stage verifier。
  source op分类与postcheck使用同一trait/interface，新增source op不得被unknown-op规则静默放过。
- 同一次clone用`IRMapping`；不用pointer、遍历顺序、ordinal、打印文本或symbol名恢复对应。
- 试运行只clone最近的`IsolatedFromAbove` owner，外部operand映射到scratch自有value；scratch不得引用或增加原IR use。
  不为取得anchor构造临时ModuleOp/FuncOp。为最终output、外部工具或reducer生成的clone必须说明owner、失败命运和下游消费者。
- `fold`只做便宜、局部、确定的恒等式；canonicalizer只优化，不承担correctness legalization。Greedy rewrite限制root和工作量。

### C++、错误与确定性

- Public/跨stage API使用范围明确的typed结果；相关多结果返回命名struct，不用`bool`、字符串标签和多个nullable输出拼状态机。
- 只读计算与IR修改分开，所有可能失败的检查尽量在首次mutation前完成。
- move-only类型表达所有权；引用和裸pointer只在owner存活且IR未修改的当前epoch内使用。地址只能作局部查找键。
- 本仓不使用C++ exception或RTTI。IR变换返回`LogicalResult`/`FailureOr`并发diagnostic；文件、runtime和library边界使用
  `llvm::Error`/`Expected`。unsupported、capacity、indeterminate和compiler error保持typed区分。
- `assert`和`llvm_unreachable`只用于verifier-valid输入下不可能发生的内部状态；不解析diagnostic文本控制流程。
- 不使用可变singleton或没有失效规则的cache参与编译决策。
- 输出IR、diagnostic和选择结果不得依赖地址、hash遍历或并行完成顺序；所有可观察排序都有完整semantic tie-break。
- 优化性能前记录work count、pass/analysis timing、wall time和RSS；先减少重复遍历、重复materialization和过大scope。

## 源码、接口与命名

- 源码owner和依赖方向以18号设计为准，MLIR component边界以19号设计为准。CMake显式列source和direct library dependency，
  同一translation unit只有一个production owner，不形成component循环依赖。
- Public header自包含；private helper留在`lib/`。源码树不保留空目录、无consumer fixture、未注册test或generated cache。
- Wafer-owned C++ API、IR、磁盘格式、runtime ABI和fixture只保留一个支持形式。修改时同步更新全部仓内producer、consumer、测试和文档，
  不保留`V2`、compatibility wrapper、双reader或新旧分支。
- 外部NPY、license、vendor规范、device runtime和第三方版本保留其原始名称与检查方式，不与Wafer内部版本规则混用。
- 改名先读definition、constructor、consumer、verifier/lowering和pinned同类API；一次只改一个概念及直接调用链，检查旧名残留后再继续。
- 用户否定某个名称时停止该命名方向并重新阅读实现，不自行换近义词继续修改。
- type用名词，函数用动词短语，conversion同时说明source和target。namespace、强类型或parent已消除歧义时不重复阶段/范围限定。
- 删除旧实现前指出新实现和对应测试；大规模迁移记录旧能力到新代码、正式调用路径和测试的映射。source未进CMake不等于无用。
- 一个函数、pass或文件只解决一个明确问题；新增对象必须直接服务于legality、transformation、lowering或diagnostic。
- 功能修改不夹带全树格式化、无关重命名或第二套总体设计。

## 文档、经验与版本控制

- current文档使用中文，真实API、IR和命令保留原拼写。先写输入、输出和通用规则，再给示例；示例不能成为协议。
- 不把本地路径、环境变量、动态状态、旧build目录或其它项目实现写入长期设计。未确定问题写入“待讨论问题”并说明原因，
  不用`TODO`/`TBD`代替结论。
- `memory/`只记录可复用且已验证的方法或问题根因，不记录临时状态、猜测、单个case偶现或未确定IR设计。
- 直接使用当前checkout；只有用户明确要求时才创建worktree。Git操作前检查状态，不使用破坏性Git命令，不撤销无关修改。
- 会暂时关闭产品入口或canonical gate的跨work-item迁移在独立开发分支完成；main只接收产品入口和gate同时闭合的提交。
- Codex提交author为`Codex <codex@openai.com>`，并添加`Co-authored-by: hehesnail <shashen008he@gmail.com>`。

## 完成检查

结束修改任务前逐项确认：

1. 用户要求的文件和功能已完成；未完成项和限制已明确说明。
2. 非小修的pipeline contract与本项覆盖矩阵完整，代码、测试和文档使用同一协议。
3. 实现只消费current IR和有效analysis，没有shadow owner、猜测SPM合法性或猜测同步。
4. definition、直接consumer、CMake owner、注册、测试和current文档已同步；旧接口/路径残留为零，或编号设计明确允许。
5. 受影响测试使用本轮构建并实际执行，无意外skip/unsupported。代码/CMake/注册变化通过canonical完整增量构建；
   第二次无源码变化的构建为Ninja no-op。
6. 声称端到端、no-card或board-ready时，验证确实从真实输入推进到直接下游产品；板端`done`只来自本轮真实设备结果。
7. `git diff --check`、完整diff和`git status`已复审，只提交当前任务相关修改。
