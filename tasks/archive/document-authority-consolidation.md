# 文档事实源收敛实施计划

状态：`done`。该work item只整理当前文档的职责、状态和历史归属，
不修改compiler、runtime、IR、ABI或硬件结论。

## Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  当前仓库Markdown、current源码/CMake/test registration，以及编号设计、事实文档和历史归档之间的引用关系。
- Current stage responsibility:
  让任务状态、稳定设计、硬件事实、实施计划和历史证据分别只有一个authoritative owner；删除或改正
  与current实现冲突的表述，并缩短每次开发必须读取的current上下文。
- Output IR / files:
  精简后的README、tasks/progress.md、当前编号设计、current plans、docs事实入口和memory方法文档；
  已完成实施记录移入tasks/archive/。
- Downstream consumer:
  Q52 physical-dataflow实现、Q53 host qualification及后续维护人员。
- User-level driver / named pipeline:
  不适用；本项只改变仓库文档。
- Explicit non-goals:
  不改变任何pipeline行为，不按篇幅删除硬件实验表、Instr ODS/ABI规格或05号算法设计，不把历史材料
  重新声明为current事实，不运行无关构建或板端测试。
- Completion criteria:
  tasks/plans/只保留未完成计划；progress与current plan状态一致；README与当前产品入口一致；
  current文档不把退役符号、归档路径或任务阶段描述为现行实现；本地Markdown引用全部有效；
  stable contract、动态状态、事实证据、工作方法和历史记录各归其owner。
```

## 线性步骤

1. 归档已经完成的计划；把混合在同一文件中的已完成历史和未开始工作拆开。
2. 收缩`tasks/progress.md`为状态、直接前置和完成门禁；更新`tasks/README.md`导航。
3. 修正根README和编号设计中与current源码、注册入口或任务状态冲突的表述。
4. 修正hardware/runtime事实文档中的失效current路径和任务号路由，不删除原始证据。
5. 将Q52已完成checkpoint移出current plan，只保留pending工作、必要前置和逐项门禁。
6. 将`memory/`收缩为可复用构建/调试/防复发方法，详细历史进入archive。
7. 运行文档引用、路径、authority、退役符号、状态一致性和完整diff检查；归档本计划并提交。

## 覆盖矩阵

| 文档类别 | 输入等价类 | 必须精确断言的输出 | 直接下游witness |
| --- | --- | --- | --- |
| 状态与计划 | doing/next/queued/later、done plan、混合计划 | current状态只在progress出现一次；plans中无done计划；pending顺序不变 | Q52/Q53导航可从progress直接到唯一current plan |
| 稳定设计 | pipeline contract、算法规格、实现索引、历史证据 | 输入/输出/consumer/非目标保留；退役class和动态Q状态不再冒充合同 | 编号文档仍覆盖01--20且Pipeline Contract完整 |
| 产品说明 | 可用CLI、明确unavailable policy、测试入口 | README与current CLI和产品测试一致，不声明不存在的model session或shortlist | `wafer-compile --help`及current product tests |
| 硬件事实 | supported、board-observed、unknown、excluded | 原始表和证据保留；current引用指向存在的稳定文档 | completion、SPM、target设计能追到事实源 |
| 方法与历史 | build/debug方法、bug pattern、施工日志 | memory只保留可复用方法；详细历史可从archive找到 | 新任务无需读取历史日志即可执行 |
| 引用与命名 | Markdown链接、反引号路径、current symbol | 本地链接存在；current文档不声称不存在符号；旧计划引用均改到新owner或archive | repository text checks全部通过 |

## 验证

- 解析全部非archive Markdown链接并检查本地目标存在。
- 比较`tasks/progress.md`与current plan中的状态和导航。
- 扫描current文档中的退役符号、失效plan路径和历史stage编号。
- 检查01--20编号文档仍存在且保留完整Pipeline Contract。
- 检查完整diff，确认只有文档归属、表述和导航变化。

## 完成结果

- 三个已完成计划和两个混合历史计划退出`tasks/plans/`；Q57/Q61分别取得独立current plan。
- Q52第1--11项及旧路径账本完整归档；current plan只保留第12--20项与Q53逐项门禁。
- `tasks/progress.md`只保留状态、直接前置和完成门禁；README与current CLI help、none/search typed-unavailable
  边界一致。
- 02、09、11、12、14--17、19、20及两份TX81事实文档不再把动态任务状态、退役符号或失效路径写成current合同。
- `memory/general_dev.md`只保留方法；收敛前全文和MLIR代表实现调研均在archive可追溯。Hardware calibration表、
  Instr规格和05号算法设计未按篇幅删减。
- 非archive Markdown本地链接、current plan路径、退役符号、01--20 Pipeline Contract、source/IR organization和
  `git diff --check`均通过；`wafer-compile --help`核对了公开CLI。纯文档修改未运行无关compiler build。
