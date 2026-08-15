# AGENTS.md

本文件说明本仓库的稳定协作规则和仓库导航。它不记录当前动态任务状态、短期路线图或尚未收敛的
IR 设计选择。

Wafer AI compiler / runtime 处在设计收敛和实现推进阶段。文档、IR、pass、工具和测试会一起演进，
但工作标准按真实编译器工程执行：设计边界清楚，IR 语义可验证，文档不能把临时讨论、单个 case、
未固定假设或其它项目实现路径写成长期架构。

## 优先级

1. **用户当前明确、无歧义的要求**。
2. **本文件规定的工作流程和仓库规范**：用户目标决定“做什么”，本文件流程决定“怎么做”。除非用户
   明确要求只讨论、不落文件，执行任务时不能跳过上下文读取、设计同步、pipeline contract、验证和提交。
3. **任务完成标准**：文档一致、代码能编译、测试能通过、功能真的工作。
4. **编译器工程原则**：IR 语义显式、pass 边界清楚、analysis 可重算、lowering 可验证。
5. **项目既有风格和最新设计结论**：通过读当前仓库建立，不从其它项目照搬。

执行已收敛任务时，不反复询问可自行判断的事项；设计、方案、架构边界未收敛时，要说明事实、
假设、取舍和风险。用户说“先讨论”“别急着改”时，只讨论，不落文件。

## 工作流程

每次进入执行阶段，按下面顺序推进。它是硬约束，不是收尾建议。

1. 看 `git status`，识别已有改动；不得回滚用户或其它工具的无关改动。
2. 按任务范围读取上下文：`AGENTS.md`、`tasks/progress.md` 任务队列和相关编号设计文档；
   找不到编号文档或需要确认 archive 边界时再查 `tasks/README.md`，必要时读 `docs/`、`memory/` 和代码。
3. 以 `tasks/progress.md` 任务队列作为执行管控入口；当前工作必须能落到一个队列项，或者先更新
   队列和对应设计文档。
4. 非小修、非纯文本错别字任务必须先确认或补齐对应编号设计文档和 pipeline contract，再写代码。
5. 按IR和编译输出边界施工：说明消费哪一层上游IR或输入格式，产出哪一层下游可验证IR或输出格式。
6. 按任务类型运行测试、构建、脚本自检或文本一致性搜索；声称完成前必须有本轮新鲜验证结果。
7. 如果任务状态、设计边界或稳定经验变化，同步对应文档和 `memory/`；任务收尾提交相关改动。

如果某一步因为环境或外部依赖无法完成，结果中必须说明缺口、已做验证和剩余风险。

### 构建和 Host 测试并发

- 独立的 host 构建、单元测试、catalog、no-card 和静态检查默认按机器可用逻辑 CPU 并行执行；
  CMake build / CTest 优先使用 `nproc` 给出的并发度，不得习惯性固定为 `-j2` 等低并发。
- 只有已确认的内存上限、共享可写目录、显式 resource lock 或工具本身不支持并发时才降低并发度；
  降低时要使用当前环境下最大的安全并发并说明原因。
- 本规则不改变下述真实板端测试的串行约束；真实设备 launch 始终单进程、逐 case 执行。

### 板端测试

- 任何包含板端验证的任务，无板阶段都必须推进到`board-ready`：case、oracle和runner完整，实际生成完整
  package并通过no-card；不得留到上板时补。只做当前case的定向验证，禁止重复共享资格或全量审计。
  `board-ready`不是`done`；真实板测通过后才能完成。
- 同一重启会话且软硬件身份未变化时，环境资格只确认一次；不为每个 case 重复版本、反汇编、状态、
  heartbeat 或其它无关 gate。
- 编译器/runtime功能纵向和qualification板测的默认数据类型使用FP16或BF16；只有测试目标本身是
  F32格式、ABI、转换或数值边界，或者真实上游workload明确要求F32时才使用F32，并在对应测试合同中
  写明理由。不得把host unit、no-card或model fixture中的F32机械复制成板测workload；上板前必须单独
  核对source、program descriptor、payload和expected的dtype一致且符合本条默认规则。
- 普通 case 走最小路径：增量构建、单进程串行 launch、bounded timeout、结果 / guard 校验和正常生命周期。
- 禁止读取、重新判定或回放历史板端输出；禁止重新执行已有结论的 case 来“确认”旧证据。代码或测试
  校验逻辑修改后，只能由本轮新构建、新启动和新输出产生硬件结论。
- 历史板端 raw、日志和报告只作为原始审计记录，不得作为测试输入，不得更新当前结论，也不得进入
  待执行批次。
- no-card、host oracle及深入 ABI / firmware诊断只在测试实现变化、环境变化或板端异常需要归因时执行。
- 板端测试不并发；timeout或设备异常后停止当前批次，不自动 retry、reset或power。

## 讨论和推进

- 先区分请求类型。解释、状态、复盘、意见和澄清类问题默认直接复用当前上下文与已有证据回答，不擅自
  升级为审计、实验、重新验证、文件修改或并行子任务。
- 只有用户明确要求查询、审计、验证、运行或修改时，才展开对应执行。现有上下文确实不足以可靠回答时，
  只做不可替代的最小只读确认；不得因此自动启动多 agent 审计、长时间构建或测试。
- 调用工具或分派子任务前，先检查该动作是否直接产生用户当前要求的结果；不能直接产生就不执行。已经有
  本轮 fresh 证据时优先复用，不为解释同一结论重复跑验证。
- 解释型问题先给直接答案。更深入的审计即使可能有价值，也只能作为可选后续说明，未经用户明确要求不得
  自动开始。
- 用户追问、纠正或打断时，立即停止与当前问题无关的后台扩展工作，先响应最新问题。
- 设计讨论先说明当前理解、关键假设、已确认事实和仍需判断的点。
- 对真实取舍给出理由、风险和推荐，不把未收敛问题伪装成结论。
- 只有存在真正歧义且继续工作会产出与用户意图相反的成果时，才停下来问用户。
- 阶段性状态、部分验证通过或单轮实现闭环不等于任务完成；没有方向性阻塞时继续推进到可评审状态。
- 进度同步用于说明关键判断和下一步动作，不是请求许可，也不是完成证明。

## 仓库导航

常用目录：

- `tasks/README.md`：编号设计文档导航和 archive 边界；不是任务状态或设计合同的主要依据。
- `tasks/progress.md`：任务队列，记录每步任务的状态、对应设计文档、要做什么、完成要求和不算完成。
- `tasks/`：当前编号设计文档，按 compiler pipeline 语义顺序排列。
- `tasks/plans/`：当前实施计划，只拆解施工步骤和依赖 checkpoint；状态仍以 `tasks/progress.md` 为准。
- `tasks/archive/`：历史审计、恢复和任务记录，只作背景，不作为当前架构合同。
- `docs/`：硬件、runtime、ABI、反向分析资料。
- `docs/tx8-deps-reverse-engineering/`：依赖、runtime、firmware、接口约束。
- `tools/`：辅助脚本。
- `memory/general_dev.md`：通用开发经验、构建方式、调试入口和稳定 workflow。
- `memory/bugs.md`：问题、失败现象、根因、修复方式和防复发模式。

优先读取：

1. `AGENTS.md`。
2. `tasks/progress.md`。
3. 与当前任务直接相关的编号设计文档。
4. 找不到相关编号文档、需要确认文档编号或 archive 边界时，查 `tasks/README.md`。
5. 涉及硬件、runtime、ABI 或 memory hierarchy 时，读对应 `docs/` 资料。
6. 涉及构建、调试、历史问题或可复用经验时，读 `memory/general_dev.md` 和 `memory/bugs.md`。
7. 涉及已有代码或脚本时，读代码；不要只看文档。
8. 涉及 MLIR dialect、pass、interface、verifier、region 或 conversion 设计时，优先查 MLIR 官方文档。

设计和事实冲突时，优先级为：用户当前要求，本轮已收敛结论，`tasks/progress.md` 和当前编号设计文档，
`docs/` 事实资料，最后才是 `tasks/README.md`、archive 和历史资料。该冲突优先级
不降低本文件流程约束。遇到冲突时不要悄悄合并，要在修改中收敛成清晰边界。

构建和测试入口以当前 CMake / lit 配置以及 `memory/general_dev.md` 最新记录为准。不要把旧 build
目录、临时任务名或本地实验路径写成长期约定。

## Memory 经验沉淀

`memory/` 是稳定经验库，不是任务状态板，也不是设计文档替代品。每次收尾时判断本轮是否产生可复用
经验；有就写，没有就说明无需沉淀。

- `memory/bugs.md`：记录遇到的问题、误导来源、根因、修复方式和防复发模式。
- `memory/general_dev.md`：记录可复用开发模式、构建/测试命令、调试入口、仓库约定和稳定 workflow。
- 发现 `memory/` 与当前编号设计文档、任务队列或代码事实冲突时，同批更新或删除过时 memory。
- 不把临时任务状态、个人推测、未验证 workaround、单个 case 偶然现象或未收敛 IR 设计写进 `memory/`。

## 版本控制

- 默认直接在当前 checkout 工作。
- 不主动新建 worktree，除非用户明确要求。
- Git 操作前先看 `git status`。
- 本仓库由 Codex 执行并提交的共同工作使用 `Codex <codex@openai.com>` 作为 author，并附加
  `Co-authored-by: hehesnail <shashen008he@gmail.com>`；推送前检查共同署名。
- 不使用破坏性 git 命令；不要回滚用户或其它工具做的无关改动。

## 编码和设计规则

### Pipeline Contract

非小修、非纯文本错别字任务，在设计或实现前必须定位compiler pipeline，说明上游IR或编译结果、当前
stage 责任、下游消费和用户级入口。任务文档、设计小节或实现说明必须包含：

```text
Pipeline position:
- Upstream IR / input:
- Current stage responsibility:
- Output IR / files:
- Downstream consumer:
- User-level driver / named pipeline:
- Explicit non-goals:
- Done criteria:
```

约束：

- `pass`、tool、test、文件名和任务号只是实现索引，不能替代IR层、输入输出合同或长期架构对象。
- 任务号、阶段号和临时里程碑编号只能出现在任务文档、历史记录和路线图索引；不能进入 build 目录、
  CMake target/cache variable、output file/directory名、tool/script/CLI/help/docstring、pass/pipeline名称、
  IR op/type/attr/interface/verifier 合同、diagnostic/log 前缀、FileCheck/golden output。
- 需要表达阶段关系时，用稳定语义边界名，如 `tile-region`、`instr-lowering`、`spm-offsets`、
  `ddr-offsets`、`candidate-selection`。
- 单个 pass 可以是实现单元，但主线设计必须说明它在 named pipeline / driver mode 中的位置；
  不能让用户或 integration test 长期手动拼 pass。
- 局部 FileCheck、fixture、negative verifier 或 shape-only dump 只能补覆盖，不能作为主线完成证明。
- 如果下游尚未实现但硬件 / ABI 能表达该语义，应记录为后续任务或扩 IR；不能把下游缺口反写成上游不支持。

### IR 边界

- 设计边界写成 IR 层、显式表示对象和 verifier / lowering 合同，不写成某个 pass 名字。
- 每层只携带自己能稳定解释、变换和验证的信息；低层事实只能作为 legality / cost input，不能提前污染上层语义。
- 新 IR 前先检查已有 dialect、op interface、trait、type、attr、canonicalization 和 conversion
  infrastructure 是否能表达；不能稳定表达、验证或 lowering 时才引入私有 dialect/op/type。
- Analysis 只能从当前 IR 派生局部、可失效、可重算的结果；transformation 只能读取当前 IR 和本 pass
  内局部 analysis，并直接改写当前 IR 或构造下一层 IR。
- 需要跨阶段保留且不能从当前 IR 重算的内容必须进入 IR 自身，但优先用 SSA use-def、region/control-flow、
  op 语义、effect、type 或明确 op 表达，不滥用 attr。
- 不维护影子调度计划，不把 body 已表达的执行结构复制成全局计划 attr。

### MLIR 工程规则

- **先核对版本事实。** MLIR 语义和推荐方式以官方文档为依据，当前可用 API 以仓库 pinned LLVM/MLIR
  headers、实现和测试为准；不得假设 newer upstream 选项、interface 或 driver 行为已经存在。
- **IR 必须自包含。** 影响 legality、rewiring、lowering 或 candidate 结果的事实用 SSA、region、type、effect、
  typed op/attr/interface表达；`Location`只记录source location和诊断位置，不承载指针、稳定语义键或跨pass语义。
- **名称对应真实compiler语义。** type使用语义名词，function使用说明实际动作的动词短语，conversion明确源和目标表示；
  同一概念在ODS、C++、pass、文件和diagnostic中保持一致。发现含糊或误导名称时，先看它处理哪层IR、实际做了什么、
  产出什么，再直接重命名或拆分。
- **限定词只用于必要消歧。** 范围、层级、阶段或实现状态限定词必须对应同一语义域内真实存在的另一类对象，并且去掉后
  会使当前API产生歧义；namespace、强类型、parent op、pass anchor或容器已经确定语义时，不在type、function、文件名中
  重复限定。表示转换边界时明确写出源表示和目标表示，进入目标表示后使用目标对象自身的名称；不得用限定词泛指“更低层”、
  “最终”、“已选择”或“更重要”，也不得用它掩盖一个对象承担多项职责的问题。
- **改名前完成语义核对。** 修改名称前必须读完对象定义、字段或region、构造位置、直接消费者以及相关verifier/lowering，
  并核对仓库pinned LLVM/MLIR中的同类接口；无法用一句话准确说明对象表示什么或函数实际做什么时，禁止改名。一次只收敛
  一个概念及其直接调用链，新名称确认前不得做整仓同义替换。名称被用户否定后，立即停止该命名方向并重新阅读实现，
  不得自行换一个近义抽象词继续修改。每批改名完成后先检查完整diff、旧名称残留、构建和相关测试，通过后才能处理下一批。
- **C++ API 用类型表达合同。** public/跨 stage API 使用窄而强类型的输入；一组字段共同构成一个结果时优先返回 named
  typed result，MLIR 惯用的非空 caller-owned 输出引用可以保留。`bool` 适合 predicate 和二态 option，但不得与字符串
  标签、多个 nullable 输出参数或隐式全局状态拼成状态机协议。只读 query 与 IR mutation/apply 分开，使 query 可测试、
  apply 可原子提交；mutable global、singleton 和无失效边界 cache 禁止进入编译语义。
- **资源控制与lifetime进入类型和接口。** `OwningOpRef`、`unique_ptr`和move-only具体编译结果表示独占控制权转移；引用和普通
  裸指针只表示当前IR epoch内的non-owning reference。地址可以作局部查找键，但不能跨clone、erase、会失效的mutation、
  异步任务或长期cache充当稳定语义键；同次clone对应用`IRMapping`，跨stage语义进入IR自身。
  listener、insertion point、临时 mutation 和 rollback guard 使用 RAII。
- **失败必须可分类且保持有效状态。** 本仓库按LLVM惯例不使用C++ exception/RTTI：IR transformation内部使用
  `LogicalResult`、`FailureOr`和diagnostic，文件/runtime/library边界使用`llvm::Error`/`Expected`，candidate边界使用能区分
  accepted、exact rejection、indeterminate与compiler bug的typed result。`assert`/`llvm_unreachable`只表示verifier-valid
  输入下不可能发生的内部错误；用户输入、unsupported、容量和环境失败必须可恢复传播。不得解析diagnostic字符串决定控制流。
- **确定性和性能都是编译器合同。** DenseMap/DenseSet 和指针键可用于局部 lookup/visited set，但 candidate 顺序、winner、
  diagnostic、IR和具体编译结果不得依赖地址、hash-table遍历、等价元素的不稳定排序或并行结束顺序；可观察边界使用完整
  semantic tie-break。性能判断先记录 work count、pass/analysis timing、wall time 和 RSS，优先修正 scope、重复
  traversal/materialization 和算法复杂度，再做有测量依据的低层优化。
- **遵守LLVM式库边界。** public header自包含，implementation的main header优先include，`Internal.h`留在`lib/`私有目录，
  CMake显式声明实际library依赖；不把compiler driver、IR schema、analysis、transform和runtime helper揉成循环依赖，也不在
  功能改动中夹带全树格式化或无关rename。
- **ODS 和标准 interface 优先。** 稳定 semantic field 进入 ODS typed argument/property并使用 generated accessor；
  优先复用 RegionBranch、Call、MemoryEffect、ViewLike、DPS/Bufferizable、Tiling 等标准 interface。只有标准机制
  无法稳定表达且已有 verifier/lowering consumer 时才新增 Wafer-specific interface。
- **operation hierarchy 决定 pass/analysis scope。** pass/analysis锚定包含所需事实的最窄合法 operation；可独立处理的
  `IsolatedFromAbove` op使用 nested pass。call/symbol closure、function-boundary bufferization、跨region outstanding access、
  card级resource verification等真实全局阶段保留在 func/module/card scope，不机械地下沉。operation pass不得替换自己的root；
  删除wrapper或重接parent SSA由父operation transformation负责。
- **pipeline 只有一个实现。** 一个pass只做一个可命名的IR变换，声明dependent dialect、typed option和analysis
  preservation；production driver与`wafer-opt` named pipeline复用同一query/apply和pipeline builder，不维护direct
  mutation平行实现，也不长期手工拼pass。
- **区分pass、subpipeline与driver。** atomic pass/kernel在最窄合法anchor只完成一个可验证IR变换；semantic
  subpipeline组合一个稳定且verifier-legal的IR边界并可打印、独立重放；compiler driver执行搜索、一对多module拆分、
  外部工具调用和原子目录替换，只调用前两层。顶层composite可以较长，但必须暴露可测试leaf stage；不得机械拆出非法
  中间IR或破坏一次性应用语义。
  仅把一个既有atomic pass加入`OpPassManager`的helper使用`add...Pass`；`build...Pipeline`只用于拥有稳定语义边界和组合合同的
  builder（即使该边界当前恰由一个pass实现），不能用单pass别名伪装新的pipeline层级。
- **analysis 可重算、可失效。** 只由current IR和显式immutable target facts派生且会重复消费的事实进入
  `AnalysisManager`；mutation明确preserve/invalidate。query-local candidate assignment、一次性validation和跨clone memo
  不硬塞成MLIR analysis，也不写回IR。跨全局关系在最近container一次验证，不在每个leaf verifier重复walk module。
- **rewrite 必须事务安全。** pattern callback中的IR修改全部通过`PatternRewriter`；尽量在首次mutation前完成validation，
  failed match使用`notifyMatchFailure`，不得更新rollback之外的`failureReason`、callee set或其它mutable side data。
  clone对应使用`IRMapping`，不用pointer、walk顺序、ordinal、打印字符串或symbol拼写恢复operation对应关系。
- **隔离变换事务只clone最小真实scope。** 只有查询必须消费实际改写后的IR且不得修改原IR时，才clone最近的
  `IsolatedFromAbove` operation；普通lowering直接使用nested pass，不clone。带外部operands的operation必须用`IRMapping`
  映射到scratch-owned SSA，不能让clone交叉引用或新增原IR的use；事务结束只返回typed结果并销毁scratch IR，不为取得
  module anchor构造synthetic Module/Func。
- **conversion legality 要 fail closed。** source op类别使用稳定marker interface/trait或等价单一分类；所有source实现者
  默认illegal，structural/non-executable descriptor/target op显式legal，postcheck复用同一分类。full conversion必须证明source语义全部
  消失；只lower子集时明确使用partial conversion和独立stage verifier，不能用“unknown全legal”伪装full conversion。
- **fold、canonicalization 和声明式rewrite各司其职。** `fold`只放便宜、局部、确定的恒等式；canonicalizer只优化，
  不承担correctness legalization；greedy rewrite限制affected roots和work budget。简单typed op-to-op规则优先DRR，
  复杂layout/index/resource/region逻辑保留C++ pattern；PDLL或Transform dialect只有出现明确consumer时才引入，不设使用配额。
- **alias/effect 在lowering前必须确定。** view、new allocation、destination mutation和materializing copy不能由lowering按
  users临时切换；tensor层用DPS/Bufferizable分析in-place/out-of-place，进入memref/Instr层后alias和memory effect是稳定合同。
- **按MLIR合同验证。** 至少覆盖custom/generic form roundtrip、verifier正负例、conversion failure atomicity、analysis
  invalidation、named/production pipeline parity和`verify-each`；局部pass成功、canonicalizer恰好清掉残留或旧generated
  build能编译，都不能代替fresh source/build和下游IR/file验证。
- **退役实现前先迁移能力。** source未进入CMake只说明它不属于active build，不能据此推断其中算法、proof、diagnostic或
  测试资产已经无用。先逐项确定承接实现；仍被当前或后续合同需要的能力必须迁入active source并受测，之后才能删除旧
  实现。只有已被现行IR/API明确淘汰且没有独有能力的源码可以直接清理。

### 协议和语义恢复

- 跨模块共享的常量、attr key、enum value、field name 定义在一处，消费侧引用定义。
- 协议错误尽量在 ODS、C++ type system、verifier、pass legality 或 dialect conversion 中暴露。
- 创建某个 dialect 的 op/type/attr 的 pass 必须声明 dependent dialects。
- 不用 buffer、var、op、文件名或示例名恢复语义角色、绑定关系或协议分支；名字只能用于日志、调试和讲解。
- 语义恢复优先依赖 IR type/rank/shape/dtype/memory space、structured op semantics、indexing maps、
  SSA def-use、region/control-flow、loop-carried dataflow、op interface、verifier relation 和必要显式 attrs/types。
- 当前 IR 结构无法稳定区分语义时，扩 IR / DSL / attrs / types，不把名字匹配升级成长期分析。

### 通用性和目标层级

- 设计可以 case-driven，但 case 只用于展示 IR 流经 pipeline；协议以通用职责为准。
- 不把 workload、shape、参数顺序、单个 kernel、某个 op 的 tile 选择或某条 runtime 路径固化成协议。
- case 后必须说明哪些是示例参数，哪些来自通用 IR / interface / verifier，哪些只是 cost model /
  planner 候选，泛化情况如何处理。
- 目标相关低层对象只在能解释和验证它们的 lowering 层 materialize；不要提前作为上层字符串列表或旁路数据。
- 一个函数、pass、文件只解决一个问题；描述需要“并且”连接两个不同动作时应拆分。
- 每个新设计对象、算法结构、pass、typed field 或 validator 都要回答它如何帮助 legality、planning、
  lowering、diagnostic，或删除旧 matcher / fallback / 旁路通道。

## 文档规则

- 当前文档默认中文。
- 当前实施计划统一放在 `tasks/plans/`，历史计划移入 `tasks/archive/`；不得按 agent、skill 或临时工具名
  建立主线文档目录，也不得要求某个 skill 才能解释或执行计划。
- 先讲边界和通用方法，再给 case；case 后说明哪些只是示例，不是协议。
- 主线任务文档遵守pipeline contract和长期命名规则；实现入口只能作为索引，不能替代IR、输入格式和输出格式合同。
- 不把其它项目路径、环境变量、测试入口、动态任务状态或 runtime 路线写成当前项目主线。
- 不把尚未收敛的设计选择写进本文件；这类内容留在设计文档讨论和演进。
- 不把 `TODO` / `TBD` 当结论；未定问题写成“待讨论问题”，并说明为什么未定。
- 修改一个概念时，顺手检查同文档其它小节是否有同类过度特化、重复事实源或旧字段残留。

## 禁止事项

- 不新增第二份总体设计文档，除非用户明确要求；导航索引可以存在，但不能复制或改写架构合同。
- 不在 IR 外再造长期语义通道：bag、opaque payload sidecar、side table、名字约定、临时 wrapper 都不能成为协议。
- 不让文档和代码长期协议错位。
- 不把 analysis pass 写成语义恢复黑箱。
- 不把单个 case 的调度结果写成通用架构规则。
- 不把原始硬件文档里的历史命名直接提升为当前 compiler IR 名词。
- 不把上层 IR 污染成低层执行或运行时封装细节。

## 设计检查清单

写设计或改设计时至少检查：

- pipeline contract 是否完整。
- 当前实现是否推进整条 compile pipeline 的一个边界，而不只是让某个 pass 能跑。
- 完成证明是否重放已完成上游链路，并证明当前输出会被下游直接消费。
- 信息是否能从当前 IR 推出；不能推出时，是否真的需要跨 pass 保留。
- 保留方式应该是 op、region、type、attr、effect 还是 analysis。
- 设计是否形成重复事实源，是否绑定单个 case、shape、op 名、路径或历史实现。
- verifier、canonicalization、parser/printer、lowering 的责任是否清楚。
- lower 到目标后端时是否真正帮助 legality、planning、lowering 或 diagnostic。

## 收尾和完成标准

收尾要求：

1. 同步受影响的设计文档、任务队列和 memory。
2. 按任务类型运行验证；声称端到端或主线 gate 通过时，确认相关测试实际执行而不是
   `unsupported` / skipped。`ctest passed` 不能替代 lit unsupported 清单、外部依赖 feature 配置和真实输入链路检查。
3. 沉淀经验：问题、根因和修复模式写 `memory/bugs.md`；可复用命令、入口、检查方式和 workflow 写
   `memory/general_dev.md`；不稳定经验不要硬写。
4. 明确未完成项和限制。
5. 提交本次相关改动。

一次任务完成至少满足：

- 用户要求的文件已实际修改，或明确说明无法修改的原因。
- 非小修任务已有 pipeline contract，且实现、测试和文档没有把单个 pass / fixture 当成主线。
- 设计边界与当前 MLIR/compiler 工程方向一致。
- 没有引入其它项目路径、环境、动态任务状态或测试入口作为长期约定。
- 相关文档没有明显冲突、旧字段残留或重复事实源。
- 做了匹配验证，并说明验证范围。
- 未完成项和限制已经明确写出。

“这一批改动已通过验证”只说明当前批次可继续推进，不等于整个任务完成。当前任务边界仍有旧接口、
旧表示或旧旁路未清理时，不能报完成；主线设计默认按终态原则收口，除非任务文档明确允许过渡例外。
