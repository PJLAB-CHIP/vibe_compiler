# AGENTS.md

本文件规定本仓库长期使用的协作流程和编译器工程规则。当前任务状态记录在
`tasks/progress.md`，具体设计记录在编号设计文档中。本文件不保存短期计划、临时结论或尚未确定的
IR 设计。

Wafer compiler和runtime的设计、IR、pass、工具和测试会一起演进。每次修改都必须保持明确的
输入输出、可验证的IR以及可重现的测试结果。

## 优先级

发生冲突时，按以下顺序处理：

1. 用户当前明确的要求。
2. 本文件规定的工作流程。
3. 任务的完成条件，包括文档、构建、测试和实际功能。
4. 编译器正确性，包括IR语义、pass边界、analysis失效和lowering结果。
5. 仓库现有风格和已经确定的设计。

执行已经确定的任务时，自行处理可以从仓库得出的细节。设计尚未确定时，说明已知事实、假设和
风险。用户要求只讨论时，不修改文件。

## 工作流程

以下步骤适用于修改设计、代码、测试或工具的任务：

1. 运行`git status`，确认工作树中已有的修改。不得撤销与当前任务无关的改动。
2. 按任务范围阅读`AGENTS.md`、`tasks/progress.md`和相关编号设计文档。只有在查找文档编号或确认
   归档边界时才阅读`tasks/README.md`。根据需要再阅读`docs/`、`memory/`和代码。
3. 将设计或实现工作关联到`tasks/progress.md`中的任务。小型维护可以直接归入已有设计文档，不必新增
   队列项。
4. 非小修必须先确认编号设计文档中的输入、输出和完成条件。缺少这些内容时，先补文档，再写代码。
5. 按IR或文件边界实现修改。说明输入是什么、输出是什么，以及下一个直接使用者是谁。
6. 运行与修改直接相关的构建、测试或文本检查。完成结论必须使用本轮生成的结果。
7. 只更新实际受影响的设计文档、任务状态和`memory/`。提交当前任务的相关改动。

如果环境或外部依赖阻止某一步，报告未完成的部分、已经运行的检查和剩余风险。

### 主机端构建和测试

- 独立的主机端构建、单元测试、`catalog`、`no-card`和静态检查使用机器可用的逻辑CPU并行运行。
  CMake build和CTest默认使用`nproc`给出的并发度。
- 只有内存限制、共享写目录、资源锁或工具限制要求串行时才降低并发度。使用当前环境允许的最大安全
  并发度，并说明限制。
- 编译器IR、analysis、planning、rewrite、conversion和lowering的正例默认使用接近真实workload的static shape：
  rank至少为3，并且至少一个主要迭代维度不小于1024。涉及分块、空间划分或循环生成时，必须成对覆盖
  `1024`等整除长度与`1025`、`1031`等非整除长度，实际经过均匀块、多个Tile、多个block/wave、
  remainder和tail；仅把一个case或个位数shape跑通不能作为主线功能完成证据。
- 小shape只用于确实需要有界逐元素穷举的独立oracle、最小verifier负例、scalar/zero-rank语义或单一边界
  故障定位，并在测试中写明缩小理由。同一机制仍须有上述真实规模覆盖矩阵。正例不能只断言成功，
  必须检查该stage承诺的exact coverage、无重叠、owner、demand、merge、tail或下游可消费结果。shape选择是
  测试覆盖要求，不能进入IR legality、workload识别或compiler策略。
- 每个非小修work item在写代码前，必须在对应编号设计或实施计划的本项小节写出自己的覆盖矩阵：输入等价类、
  整除/非整除、相关结构路径、typed failure、需要精确断言的输出和直接下游witness。全局测试原则不能替代本项矩阵；
  实现完成后逐行核对实际case。规则建立前已经完成、但没有这种逐项证据的work item，必须先安排独立coverage
  closure补齐，不能依靠历史`done`、测试总数或单个成功case继续向下游签发可信前置。
- 真实设备测试始终单进程、逐case运行。

### 板端测试

- 包含板端验证的任务必须先达到`board-ready`。在无卡环境中准备完整测试用例、参考结果和执行脚本，
  生成完整`ExecutablePackage`并通过`no-card`验证。`board-ready`不是`done`；只有本轮真实板测通过后
  才能标记`done`。
- 只运行当前任务指定的板端测试用例。不要重复共享环境检查，也不要顺带运行全量板测。
- 同一重启会话中，如果软硬件身份没有变化，只确认一次设备环境。不为每个测试用例重复版本、反汇编、
  `heartbeat`或其它环境检查。
- 普通编译器/runtime纵向和资格验证默认使用FP16或BF16。只有测试目标是F32格式、ABI、转换、数值边界，
  或者真实输入要求F32时才使用F32，并在测试文档中说明原因。上板前检查输入、program descriptor、
  payload和expected中的dtype一致。
- 普通测试用例只运行必要步骤：增量构建、单进程launch、设置timeout、检查结果和guard并正常清理。
- 不读取或回放历史板端输出。实现、测试用例和环境都未变化时，不为重复确认旧结论而重跑。compiler、
  runtime、`ExecutablePackage`、测试用例或校验逻辑变化并影响该用例后，使用本轮新构建和新输出形成结论。
- 历史raw输出、日志和报告只用于审计，不得作为新测试的输入。
- 每个`board-ready`批次对本轮`ExecutablePackage`运行一次`no-card`和主机端参考结果检查。相关代码、
  合同和环境未变化时不重复。
  只有板端异常需要定位时才进行深入ABI或firmware诊断。
- timeout或设备异常后停止当前批次。不要自动retry、reset或power cycle。

## 处理用户请求

- 先判断请求是解释、审查、验证还是修改。解释、状态、复盘和澄清问题直接使用已有证据回答，不自动
  扩展成实验、构建、文件修改或并行任务。
- 只有用户要求查询、审计、验证、运行或修改时才执行对应操作。上下文不足时，只做回答所需的最小只读
  检查。
- 调用工具前确认它会直接产生用户要求的结果。已有本轮结果时不要重复运行相同验证。
- 用户纠正或打断时，停止无关工作，先处理最新消息。
- 设计讨论先列出已知事实、假设和未确定的问题。给出建议时说明取舍和风险。
- 只有继续工作可能产生错误方向的修改时才询问用户。其它可以从仓库确定的细节自行处理。
- 在用户要求的范围内继续工作到可评审状态。中间结果和局部测试通过不等于任务完成。
- 只在关键判断或下一步发生变化时同步进度。进度说明不是完成证明。

## 仓库导航

- `tasks/progress.md`：任务状态、直接前置和完成条件。
- `tasks/README.md`：编号设计文档和归档文档的索引，不是设计或状态的主要依据。
- `tasks/`：当前编号设计文档，按compiler pipeline排列。
- `tasks/plans/`：当前实施步骤和依赖检查点。任务状态仍以`tasks/progress.md`为准。
- `tasks/archive/`：历史审计、恢复和已完成任务。只作为背景材料。
- `docs/`：硬件、runtime、ABI和逆向资料。
- `docs/tx8-deps-reverse-engineering/`：依赖、runtime、firmware和接口事实。
- `tools/`：辅助工具和检查脚本。
- `memory/general_dev.md`：稳定的构建、测试和调试方法。
- `memory/bugs.md`：问题现象、根因、修复和防复发方法。

阅读顺序：

1. `AGENTS.md`。
2. `tasks/progress.md`。
3. 当前任务的编号设计文档。
4. 需要查找编号或归档边界时阅读`tasks/README.md`。
5. 涉及硬件、runtime、ABI或memory hierarchy时阅读对应`docs/`。
6. 涉及构建、调试或历史问题时阅读相关`memory/`。
7. 修改已有实现时阅读定义、构造位置、直接使用者和测试。
8. 涉及MLIR API时，先查官方文档，再以仓库pinned源码和测试确认可用接口。

任务状态以`tasks/progress.md`为准，设计以当前编号文档为准，硬件和ABI事实以对应`docs/`及本轮验证为准。
`tasks/README.md`、archive和历史记录不能覆盖这些来源。发现冲突时明确指出并在同一修改中收敛，不要
静默合并两套规则。

构建和测试命令以当前CMake、lit配置和`memory/general_dev.md`为准。不要把旧build目录、临时任务名或
本地路径写入长期文档。

## 经验记录

`memory/`只保存可复用的稳定经验。没有新经验时不修改，也不需要在结果中专门说明。

- 将问题现象、根因、修复和防复发方法写入`memory/bugs.md`。
- 将稳定的构建命令、测试入口、调试方法和仓库约定写入`memory/general_dev.md`。
- 如果`memory/`与当前设计或代码冲突，更新或删除过时内容。
- 不记录临时状态、个人推测、未验证workaround、单个case的偶发现象或尚未确定的IR设计。

## 版本控制

- 直接在当前checkout工作。只有用户明确要求时才创建worktree。
- Git操作前检查`git status`。
- 不使用破坏性Git命令，不撤销无关修改。
- Codex提交使用`Codex <codex@openai.com>`作为author，并添加
  `Co-authored-by: hehesnail <shashen008he@gmail.com>`。
- 每个提交只包含当前任务的相关文件。推送前检查共同署名。

## Pipeline文档

非小修必须说明修改位于哪一段compiler pipeline。设计文档或实现说明需要回答以下问题；标题可以使用
等价措辞，但内容不能缺失：

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

- 用IR层或文件格式描述边界。pass名、工具名、测试名和任务号只能作为实现索引。
- 任务号和临时阶段号只能出现在任务文档、计划和历史记录中。不要把它们写入CMake target、cache变量、
  输出路径、CLI、pass名、pipeline名、IR名称、diagnostic前缀或golden output。
- stage关系使用稳定的语义名称，例如`tile-region`、`instr-lowering`、`spm-offsets`和`ddr-offsets`。
- 单个pass可以实现一个stage，但正式编译入口和集成测试必须使用注册的pipeline或compiler API，不长期
  手工拼接pass。
- FileCheck、手写fixture、negative verifier和shape dump只能补充覆盖，不能代替真实输入到下游输出的验证。
- 如果当前上游语义已经要求某项能力，而下游尚未实现但硬件和ABI可以表达，应在对应设计文档中记录后续
  工作。不要把下游缺口描述成上游语义不受支持。硬件可以表达某项能力，并不自动扩大当前任务。

## IR设计

- 每一层IR只保存该层能够解释、修改和验证的信息。低层硬件事实只能作为上层合法性检查或代价模型的
  输入，不能成为上层语义。
- 新增operation、type或attribute前，先检查现有dialect、trait、interface、canonicalization和conversion是否
  已经能够表达该语义。只有现有机制无法表示、验证或lowering时才扩展Wafer IR。
- 分析可以读取当前IR和显式传入的只读目标配置。分析结果必须可以重新计算，并在相关IR修改后失效。
  分析不得修改IR。
- IR变换只能读取输入IR、显式参数和仍然有效的分析结果。不要依赖隐藏的全局状态。
- 如果后续pass需要某项信息，而且无法从输入IR重新计算，就把它表示在IR中。优先使用SSA、region、type、
  effect、operation或明确的attribute。
- 只在一次函数调用中使用的临时计划可以保存在C++对象中。不要把它序列化为旁路文件，也不要在IR已经表达
  执行顺序后再保存一份调度表。

## 接口修改

- Wafer定义的C++ API、IR、磁盘格式、runtime ABI和测试fixture只保留一个受支持的形式。修改时同时更新
  仓库内的写入方、读取方和测试。不要增加`V2`名称、兼容wrapper、双reader或新旧分支。
- Wafer内部schema、record和symbol使用稳定的语义名称，不使用版本号作为标识。parser通过magic、字段、size、
  layout、digest和其它明确约束拒绝旧输入。
- 对仓库控制的接口做破坏性修改时，在同一修改中更新compiler、runtime、CRT、firmware、tools、fixture和文档。
  无法同步升级的外部使用者通过发布流程协调，不在主仓长期保留兼容分支。
- NPY、license、vendor规范、设备runtime和外部toolchain的版本是外部事实，保留其原始名称和检查方式。

## MLIR工程规则

MLIR行为以官方文档为准；具体API以仓库pinned LLVM/MLIR源码和测试为准。不要假设较新upstream的API已经
可用。

### IR有效性和verifier

- Pass的输入必须通过verifier，输出也必须通过verifier。verifier-valid输入触发crash或assert是pass bug。
- Operation verifier只检查operation自身、region结构以及operand/result之间的局部关系。不要在verifier中沿
  def-use链遍历整个module。需要全局信息时，在最近的共同parent上运行显式检查。
- 影响合法性、rewiring或lowering的字段应定义为ODS argument、property、type或attribute，并使用生成的
  accessor。优先使用标准trait和interface。
- `Location`只表示源码位置和diagnostic位置。不要在`Location`中存放指针、operation标识或跨pass语义。

### 命名

- type使用名词，函数使用描述动作的动词短语。conversion名称同时说明源表示和目标表示。
- 只有确实存在歧义时才增加范围或阶段限定词。namespace、强类型、parent operation或pass anchor已经确定
  含义时，不重复这些限定词。
- 改名前阅读定义、字段、构造位置、直接使用者、verifier和lowering，并检查pinned MLIR中的同类API。一次
  只重命名一个概念及其直接调用链。检查完整diff、旧名称残留、构建和相关测试后再继续下一项。
- 用户否定某个名称时，停止该命名方向并重新阅读实现，不自行换一个近义词继续修改。

### Pass和analysis

- 将pass锚定到完成工作所需的最小operation类型。可独立处理的`IsolatedFromAbove` operation使用nested
  pass。需要call graph、symbol closure、function-boundary bufferization或card级资源检查时，保留func、module或
  card范围。
- `OperationPass`不得检查root operation的sibling。pass可以修改root下面的IR，也可以修改root的attribute；
  不得修改root的operand或parent block。替换或删除root operation必须由parent operation上的transformation完成。
- Pass不得在多次`runOnOperation`调用之间保留可变状态，也不得使用可变全局状态。
- 创建新dialect operation、type或attribute的pass必须声明dependent dialect。
- Analysis不得修改operation。pass修改IR后，默认使已有analysis失效；只有确实保持有效时才调用
  `markAnalysesPreserved`或`markAllAnalysesPreserved`。
- 只有会被多个pass重复使用的IR事实才放入`AnalysisManager`。一次性检查、候选赋值和跨clone memo保留为
  当前函数中的C++数据。
- 每个IR变换只有一个实现。`wafer-opt`的named pipeline和正式compiler driver调用同一个实现，不维护另一条
  直接修改IR的路径。
- atomic pass只完成一个可以验证的变换。subpipeline只组合具有稳定输入输出的pass。搜索、module拆分、外部
  工具和目录提交由compiler driver负责。
- `add...Pass`用于向`OpPassManager`加入pass。只有组合出稳定IR边界时才使用`build...Pipeline`名称。

### Rewrite和conversion

- 在`matchAndRewrite`确认匹配成功前不要修改IR。pattern中的所有IR修改都通过给定的
  `PatternRewriter`完成。返回success时必须已经修改IR。
- 尽量在第一次修改前检查shape、type、symbol和target限制。匹配失败使用`notifyMatchFailure`。
- Pattern rollback期间不要修改外部容器、callee集合或其它无法回滚的数据。
- 同一次clone使用`IRMapping`记录对应关系。不要通过指针、遍历顺序、ordinal、打印文本或symbol名称恢复
  operation对应关系。
- 为试运行变换而复制IR时，只复制最近的`IsolatedFromAbove` operation。外部operand通过`IRMapping`映射到
  scratch IR拥有的value。scratch IR不得引用原IR，也不得给原IR增加use。不要为取得module anchor构造临时
  ModuleOp或FuncOp。
- 生成最终输出、外部工具输入、reducer输入或autotuning结果所需的clone不是scratch事务。此类clone必须说明
  谁持有结果、失败后如何处理以及结果是否交给下游。
- Full conversion必须使所有输入operation对`ConversionTarget`合法。Partial conversion只转换明确标记为illegal
  的部分，并在随后运行stage verifier。不要把unknown operation全部标记为legal来模拟full conversion。
- 用一个trait、interface或等价分类标识该stage的source operation，并在`ConversionTarget`和转换后检查中使用
  同一分类。新增source operation必须默认转换或报错，不能被unknown-op规则静默放过。
- `fold`只实现便宜、局部、确定的恒等式。canonicalizer只做优化，不承担正确性所需的legalization。
- 使用greedy rewrite时，限制允许修改的root以及迭代或工作量，避免无界重写。
- 简单的operation-to-operation规则优先使用DRR。复杂的layout、index、resource和region变换使用C++ pattern。
  只有出现明确使用者时才引入PDLL或Transform dialect。

### C++接口、所有权和错误

- Public API和跨stage API使用范围明确的C++类型。一组相关结果返回命名的struct；MLIR惯用的非空输出引用
  可以保留。不要用`bool`、字符串标签和多个nullable输出参数组合状态机。
- 将只读计算和IR修改分成两个函数，使只读部分可以单独测试，并在修改前完成所有可能失败的检查。
- `OwningOpRef`、`unique_ptr`和其它move-only类型表示所有权转移。引用和裸指针只在当前IR未修改且持有IR的对象仍然
  存活时使用。地址只能作为局部查找键，不得传入异步任务或长期cache。
- listener、insertion point和临时修改使用RAII恢复。
- 本仓库不使用C++ exception或C++ RTTI。IR transformation使用`LogicalResult`、`FailureOr`和diagnostic；
  文件、runtime和library边界使用`llvm::Error`或`Expected`。
- 规划或候选检查的返回类型必须区分成功、已证明不可行、无法判定和编译器错误。不要把这些结果压缩成
  `bool`。
- 用户输入错误、unsupported语义、容量不足和环境错误必须返回可处理的错误。`assert`和
  `llvm_unreachable`只用于verifier-valid输入下不可能发生的内部状态。不要解析diagnostic文本决定控制流。
- 不使用可变singleton或没有失效规则的cache参与编译决策。

### 确定性、性能和库边界

- 输出IR、diagnostic和选择结果不得依赖地址、hash table遍历顺序或并行任务完成顺序。所有可观察排序必须有
  完整的语义tie-break。
- 优化性能前先记录工作次数、pass/analysis timing、wall time和RSS。先减少重复遍历、重复materialization和
  不必要的作用范围，再做低层优化。
- Public header必须自包含。implementation先include自己的main header。`Internal.h`保留在`lib/`中。
- CMake显式声明library依赖。不要在compiler driver、IR、analysis、transform和runtime之间形成循环依赖。
- 功能修改不要夹带全树格式化或无关重命名。

### Memory、alias和effect

- View、allocation、destination mutation和materializing copy在lowering前必须有确定语义。lowering不得根据users临时
  决定一个operation是alias还是allocation。
- Tensor层使用DPS和`BufferizableOpInterface`分析in-place与out-of-place。进入memref或Instr层后，alias和
  memory effect必须已经确定。
- 新operation优先实现标准的RegionBranch、Call、MemoryEffect、ViewLike、DPS、Bufferizable或Tiling interface。
  只有标准interface不能表达且已有verifier或lowering使用者时才增加Wafer-specific interface。

### SPM合法性和搜索反馈

- **SPM合法性只能由实际SPM规划结果决定。** 候选必须先具体化为包含实际allocation、layout、SSA alias、effect、
  completion和lifetime的当前IR，再运行唯一的`PlanSPMMemory`/MiniMalloc路径。只有该路径成功生成并验证offset，才能
  认定候选SPM合法；只有该路径返回带实际冲突demand的typed capacity rejection，才能认定当前候选SPM不可行。
- 没有上述实际IR和实际SPM规划结果时，SPM合法性只能是unknown。禁止使用footprint estimate、upper bound、buffer
  数量倍数、shape公式、synthetic demand、预测lifetime或其它近似模型决定candidate admission、pruning、temporal
  refinement、fallback或winner。估算结果不得转换成合法性结论或进入合法性控制流。
- baseline和search都消费同一实际SPM规划结果。实际capacity rejection由外层controller作为当前候选的反馈，决定是否
  构造下一候选；`ResourceExhausted`、timeout、unsupported和compiler error保持各自typed状态，不得伪装成容量不可行。
- BodyEmitter和SPM planner不得在失败后自行retile、spill、改layout、换buffer或选择下一候选。成功候选携带本次实际
  规划结果继续下游，不得另建预测性SPM合同，也不得用重新估算代替实际offset和冲突验证。
- 当前候选的buffer relation由拥有该候选IR的Tile/Card materialization transaction管理。BodyEmitter只能通过caller-owned
  recorder报告本次实际创建的operand、result、output、movement和scratch buffer，不得持有、推断或直接维护跨scope relation
  集合。每个进入实际SPM planner的allocation必须有完整current-IR owner relation；出现无owner demand时按contract failure
  停止，禁止按shape、type、位置或“同region只有一个root”补归因。
- 测试必须用1024级整除/非整除实际候选覆盖“具体化→Instr→实际SPM规划→反馈”链路，并断言删除或扰动任何估算代码
  都不会改变SPM合法集合；只验证估算值、shape或局部pass成功不能作为SPM legality证据。

### 测试和实现迁移

- 每个IR stage的测试集合应覆盖parser/printer roundtrip、verifier正负例、conversion失败、analysis失效、
  named pipeline与正式编译路径的一致性以及`verify-each`。
- 单次修改只运行受影响的测试和必要的下游验证，不机械重跑无关测试。旧build目录或旧generated文件不能
  代替本轮构建。
- source没有进入CMake只说明它不在当前构建中，不说明其中的算法、diagnostic或测试已经无用。
- 删除仍有功能的旧实现前，指出新的实现位置和对应测试。
- 大规模替换要记录旧能力、新实现、正式编译路径中的调用和测试之间的对应关系。只有现行IR和API已经
  不需要、且没有独有测试资产的代码可以直接删除。

## 语义恢复和通用性

- 跨模块共享的常量、attribute key、enum值和字段名只定义一次，使用者引用该定义。
- 不根据buffer名、变量名、operation名、文件名或示例名推断语义。名称只用于diagnostic和调试。
- 优先从type、rank、shape、dtype、memory space、indexing map、SSA use-def、region、control flow、effect和
  operation interface读取语义。
- 如果IR无法区分后续必须区分的语义，扩展operation、type、attribute或interface。不要建立名字匹配或隐藏
  side table。
- 可以从具体测试用例开始设计，但长期规则必须适用于同类IR。说明哪些shape、参数和选择仅用于示例，
  哪些由IR决定，哪些由代价模型选择。
- 当前任务不修改数值语义时，保持现有算术operation和dtype语义。已有数值测试只作为回归。
- 只在能够解释和验证目标细节的lowering层创建低层对象。不要提前把它们表示为字符串或旁路数据。
- 一个函数、pass或文件只解决一个明确问题。新增对象、算法、pass或字段必须直接服务于合法性检查、
  transformation、lowering或diagnostic。

## 文档规则

- 当前文档使用中文。真实API、IR名称和命令保留其原始拼写。
- 当前实施计划放在`tasks/plans/`，历史计划放在`tasks/archive/`。不要按agent、skill或临时工具建立主线
  文档目录。
- 先说明输入、输出和通用规则，再给示例。示例不能成为协议。
- 实现文件、pass和测试只能作为索引，不能代替IR或文件格式的定义。
- 不把其它项目或本地环境特有的路径、环境变量、测试入口、动态状态或runtime实现写入长期设计。
- 尚未确定的问题写在设计文档的“待讨论问题”中，并说明未确定的原因。不要用`TODO`或`TBD`代替结论。
- 修改一个概念时，检查同一文档中的旧名称、重复定义和过度特化描述。

## 禁止事项

- 不新增第二份总体设计文档，除非用户明确要求。导航文档不能复制总体设计。
- 不用无类型字段集合、旁路文件、全局side table、名字约定或临时wrapper长期保存IR语义。
- 不让文档和实现长期使用不同协议。
- 不把analysis写成从名称和偶然结构恢复语义的黑箱。
- 不把单个case的shape、调度或runtime路径写成通用规则。
- 不直接把硬件文档中的历史名称用作compiler IR名称。
- 不在上层IR中加入只能由低层target或runtime解释的字段。

## 设计检查

修改设计时检查：

- 是否写清输入IR、输出IR、直接使用者和完成条件。
- 是否从真实输入IR生成了直接下游实际读取的输出IR或文件，而不只是运行单个pass或fixture。
- 验证是否只重放了产生该输入所需的最小上游链路。
- 每项信息能否从输入IR重新计算；不能时，为什么必须保留。
- 该信息应该表示为SSA、region、operation、type、attribute、effect还是analysis。
- 是否产生重复事实、名字依赖或case特化。
- parser/printer、verifier、canonicalization和lowering分别负责什么。
- 新对象是否直接帮助合法性检查、transformation、lowering或diagnostic。

## 完成任务

收尾时：

1. 只更新实际受影响的设计文档、任务状态和`memory/`。
2. 运行受影响的构建和测试。声明端到端通过前，确认测试实际执行，没有被标记为unsupported或skipped。
3. 报告未完成项和限制。
4. 提交当前任务的相关修改。

完成任务至少需要满足：

- 用户要求的文件已经修改，或者已经说明无法修改的原因。
- 非小修已经写明pipeline输入、输出和完成条件。
- 代码、测试和文档使用同一套IR或文件格式。
- 没有把本地路径、动态状态或其它项目实现写入长期文档。
- 受影响的验证使用本轮构建并实际执行。
- 已知限制已经说明。

一次修改通过测试只说明该修改可以继续推进。只要当前任务范围内仍有旧接口、旧表示或旧路径，就不能
声明任务完成，除非对应设计文档明确允许过渡状态。
