# 文档整理实施记录

本轮归现有`board-testing`的文档收尾及18号文档组织维护，不新增compiler任务。
产品板测的完成范围不变；搜索收益、三轮调优和Q53完整主机矩阵的取消决定保持。
本记录保存2026-09-26已完成的文档整理，长期合同见[18号设计](../18-source-organization.md#8-文档组织与维护)。

## 输入、输出与范围

- 输入：仓内154份Markdown、当前状态和已记录的代码/测试/板端证据。
- 职责：纠正失效安排，拆分长文，收敛现行合同与历史记录，更新引用。
- 输出：简洁状态入口、按主题组织的设计与经验、按日期组织的测量及完整归档。
- 直接消费者：开发者、审查者和后续任务；无CLI或pipeline变化。
- 非目标：不改源码、IR、算法、ABI、数值和设备资格；不重启取消计划，不重跑板测。
- 完成条件：下表逐项通过；有效合同不只存在于archive；纯文档diff、引用和状态检查通过。

## 迁移与覆盖矩阵

| 原内容 | 去向与精确检查 |
| --- | --- |
| progress历史接续、已完成队列 | 历史记录及completed index；当前状态、取消原因和未通过长LM边界保持 |
| 四份退出调度的计划 | archive；搜索与AccessReuse有效细节及矩阵先进入06号专题，不把未完成性能标done |
| Q57/Q61/Q48候选 | 留在plans；Q53取消后必须先重新明确资格前置，原覆盖门槛保留 |
| 06号长文、11/14号追加内容 | 按既有职责拆分或归位；原算法、pipeline、failure、覆盖逐段对照 |
| 性能记录 | 按日期分篇，原记录正文和证据链接保留；总页只保留口径与索引 |
| bugs与general_dev | 按主题保留根因；修正多个build tree旧规则，流程改为引用AGENTS |
| Explp及依赖说明 | 根据已有证据修正指定范围，不扩大设备资格；修正checker路径 |
| 全部文档引用 | 本地链接/锚点、移动路径和current/archive职责检查；原始JSON/CSV不改 |

## 执行与验证

先保存移动前正文和章节映射，再迁移；逐段核对正文、代码块、表格和引用。
原始证据与无关工作树修改保持。

本轮检查结果：

- 全部仓内Markdown本地链接与章节锚点扫描通过；20份编号设计的pipeline边界齐全。
- 101段实验正文按日期移动后逐段一致，原始实验JSON/CSV未改。
- 215条bug正文逐段一致；4条原记录经复审合并/修正后为3条，共218条按8个主题保存。
- 4份退出调度的计划保留完整历史正文；3份later计划的pipeline及后续覆盖合同完整，Q53取消后的启动条件明确。
- 06、11、14号正文按原行与专题/归档逐一对照：只改失效的状态/迁移引用、Explp指定板测范围及relation结果旧限制，算法、字段、代码块和矩阵保留。
- `python3 -B utils/checks/check_deps.py`通过；MiniMalloc仅因README修正更新distribution digest，algorithm digest不变。
- 完整diff及`git diff --check`通过。纯文档和对应发布文件摘要变更，不运行compiler构建或设备验证。

主要阅读入口的行数变化如下，完整细节仍由所链接专题或历史记录保存：

| 入口 | 整理前 | 整理后 |
| --- | ---: | ---: |
| progress | 430 | 92 |
| tasks索引 | 156 | 84 |
| 06号主文 | 2056 | 325 |
| 11号主文 | 1511 | 942 |
| 14号主文 | 646 | 404 |
| bug索引 | 2279 | 22 |
| 性能记录索引 | 3670 | 41 |

硬件原始资料、有效字段矩阵和未完成合同按原证据强度保留；本轮不重新签发硬件、数值或板端性能资格。

## 内容复审

2026-09-26在入口与归档整理后，按用户追加要求复审AGENTS及全部设计正文。比较基线为`7c10dacf`。
本轮仍归18号；输入为AGENTS、20份编号设计、6份专题及直接消费者，输出为去重并纠正冲突的现行合同，供开发与审查使用。
不修改编译器、算法、IR、ABI或设备行为，不重新签发板端资格，也不恢复取消任务。复审计划已完成，其覆盖结果合并到本记录。

### 逐份结论

| 文档 | 内容处理与保留边界 |
| --- | --- |
| AGENTS | 保留协作流程、构建/板端纪律、current IR与memory/completion硬约束；专业细则由必读表明确交给18/19/20号 |
| 01 | 架构只保留跨层原则、输入输出与发布链；去掉重复pipeline、owner索引和已由06号拥有的搜索细则 |
| 02 | SiLU opmath回到source语义章节；数据身份、schema与验证矩阵保留 |
| 03 | 修正“完整复制source”：IR/metadata隔离与大payload的ProgramDataHandoff分别拥有输入 |
| 04 | 收敛历史身份等式禁令；helper事务引用03号，保留本层mesh/topology核对职责 |
| 05 | 删除重复物化流水线与已结束施工顺序；实测数字回到性能记录引用；修正全屏蔽行公式和饱和/预算截断的幂等范围；算法与矩阵保留 |
| 06主文 | AccessReuse、layout与工程迁移分别引用唯一owner；移除撤回方案在必经pipeline中的声明，保留明确标注的候选合同 |
| 06 structural专题 | 旧施工序号改为实际stage；修复孤立表格行，保留Tensor撤回边界、temporal及规则常量矩阵 |
| 06 physical专题 | PBQP细节并入08号；纠正旧layout-domain裁剪和动态DMA offset限制；保留assignment/clone交接 |
| 06 search专题 | 保留预算、排序、none/search及typed结果合同；区分规则常量物化与其余literal的DDR global输入路径 |
| 06 AccessReuse专题 | 输入窗口、收益筛选、物化、失败与矩阵职责独立，保持正文 |
| 07 | 保留三种Region form、关系与verifier；重复下游算法改为直接输入输出及owner表 |
| 08 | 收齐PBQP唯一合同；full-transfer cleanup归canonical Instr形成后、fresh completion前；修正resident边界 |
| 09 | 去掉两处完整pipeline复述；修正resident lifetime和ABI失败反馈；承接AGENTS的估算不得改变SPM合法集合回归要求 |
| 10 | Attention数学与展开引用05号，保留本层DPS destination写回义务；修正resident支持范围 |
| 11主文与指令专题 | 修正alias reshape、destination insert_slice、动态DMA offset与packed BOOL支持边界；删除重复架构/迁移说明；保留ODS字段和验证条件 |
| 12 | DDR仍独立拥有demand/placement；resident SPM引用07/09号，动态base offset与11号一致，未开放动态shape/stride |
| 13 | 修正peer root在同Tile resident边界的表述；承接同步位置、动态次数与零可避免join的验收要求；保留通信算法矩阵 |
| 14主文与TargetCall专题 | ABI、字段、wrapper及资格边界有独立消费者，保持正文；不以精简为由删除硬件证据条件 |
| 15 | 独立Tile参数行归入entry ABI；移除一次修复叙述，package身份/reader/provider合同保留 |
| 16 | 构建与通用shape纪律引用AGENTS；numeric职责交给17号；去掉旧reference施工顺序与第二build例外；资格矩阵保留 |
| 17 | 分散追加的bitcast、predicate、formal并行及managed-reference内容归回对应章节；承接numeric组件职责 |
| 18 | 用职责表替代重复目录树；承接源码命名、删除/迁移、无孤立产物与无循环依赖规则；补正文维护合同 |
| 19 | 承接OperationPass/clone/C++硬约束；去掉重复e-graph/temporal算法；共享analysis仍要求多个consumer与安全失效；按ODS修正insert_slice无result |
| 20 | 唯一现行接口及外部版本例外已经简洁、边界明确，保持正文 |
| README、docs、memory与历史计划 | 沿用上一轮入口及归档；本轮未改硬件证据、测量数据、经验正文或既有历史计划 |

### 规则承接与事实核对

AGENTS移出的约束分别由19号§3–7/§9（schema、interface、scope、analysis、rewrite、clone、C++）、
18号§2.3/§6（命名、迁移、源码/CMake）、09号§13（actual SPM回归）、13号§7（同步回归）承接；
20号继续拥有唯一现行接口。必读表明确这些细则仍约束实现，不能因入口缩短降低要求。

技术纠错核对了以下current producer/consumer：

- source snapshot：`CompilationOrchestration.cpp`及02号ProgramDataHandoff合同。
- 全屏蔽attention：`OnlineAttentionDecomposition.cpp`、`OnlineAttentionMaterialization.cpp`；只纠正文档公式，不改数值实现。
- PBQP域与canonical transfer cleanup：`LayoutOptimization.cpp`、`ExecutableCompilation.cpp`与08号。
- resident SPM：`TileRegionOps.cpp`、`PlanSPMMemory.cpp`及`LifetimeAnalysis.cpp`；仍拒绝无owner、未知alias与未闭合completion。
- reshape/insert_slice：Tile `ViewOps.td`/`MoveOps.td`、verifier及`MovementLowering.cpp`；view不能临时变allocation，insert直接写dest且无result。
- DMA offset与BOOL：Instr ODS、10/11号已列明的descriptor与整字节证明；不扩大为任意动态geometry或逐bit写入。
- 规则literal：`SpatialRegionMaterialization.cpp`的内容扫描、矩形证明与fill/insert_slice物化；其余literal保持global路径。

### 本轮验证

- 20份编号设计的pipeline合同齐全；算法条件、失败分类及覆盖矩阵在现行owner中保留。
- AGENTS由256行减至190行；AGENTS、20份设计与6份专题合计由14,075行减至13,490行，净减585行；未新增设计或专题文件。
- 全仓Markdown本地路径与章节锚点检查通过；既有101段实验正文、218条经验记录及已归档/延后计划的保留检查通过。
- `python3 -B utils/checks/check_source_organization.py`和`python3 -B utils/checks/check_ir_organization.py`通过。
- 完整diff与`git diff --check`通过；本轮仅Markdown修改，不运行无关编译、no-card或设备测试。

### Q56状态补核（2026-09-29）

此前内容复审遗漏了Q56初始`board-ready`与后续统一板测之间的核对。按用户追问补查Git引入记录、
current package/runtime、回归定义和2026-09-25板测原始JSON后，清理旧独立排程，Q57/Q61依赖改为15号现行合同。
原实现、测试、历史正文和候选覆盖矩阵保留；未增加专项板测完成结论，Q57仍延后。
具体依据见[Q56核对记录](executable-package-and-resident-runtime.md#2026-09-29状态核对)。
检查通过：7份修改文档的122个本地链接/锚点、历史正文及候选范围/覆盖保留、原始JSON摘要和完整diff；
`git diff --check`通过。本次仅修改文档，未运行编译、主机测试或板测。
