# Wafer Compiler Task Queue

本文件是当前任务状态、直接前置和完成门禁的唯一入口。稳定设计在编号文档，实施步骤在`tasks/plans/`，
已完成边界和证据在`tasks/archive/completed-task-index.md`。这里不保存施工日志、算法说明或测试数字。

## 状态规则

- current work item按artifact producer/consumer关系线性排列；同一item只出现一次，全局至多一个`doing`。
- `next`表示直接前置已满足；`queued`表示等待表中前序；`later`表示不在当前主线；板端工作先到
  `board-ready`，真实板测通过后才是`done`。
- `later`只表示保留一个可追溯的候选合同，不表示已经批准实现，也不表示它是当前item的下一步。
  重新启动任何`later`项前，必须根据当前实现、产品调用者和本轮硬件/runtime事实重审其输入、输出、范围和覆盖矩阵；
  旧计划不能直接作为施工授权。
- `cancelled`表示用户明确取消，退出当前排程；不代表完成，重新启动须有新的用户授权。
- 状态变化只改对应行。详细checkpoint、覆盖矩阵和失败修复进入current plan；完成后整个计划移入archive。
- 每项开始前读`AGENTS.md`、本表、编号设计及本项覆盖矩阵；算法调研、pinned API确认、实现、fresh验证和
  设计/LLVM/MLIR规范复审均在本项内闭合。涉及hardware/runtime/ABI/completion/resource时先读对应事实源。
- 失败留在当前item修复，不跳过、不fallback，也不以历史输出代替本轮结果。

## 当前调度

文档内容复审已完成，归18号组织维护：AGENTS、20份编号设计及6份专题已核对职责、重复定义与过时说明，
见[维护记录](archive/documentation-maintenance.md#内容复审)。当前没有自动接续的编译器或板端开发。

恢复版本的指定全矩阵timing复验已完成。完整长LM S1024/1025仍编译失败，未构包、未上板、无timing；
LM/Tensor未完成整改保存在`wip/tensor-subset-materialization`，不作为可交付能力。
独立搜索性能/deep收益验收、三轮模型调优及Q53完整主机矩阵重签已取消，均不自动恢复。

验收范围与逐项证据见[板测归档](archive/board-workload-matrix.md)、[完整计时表](../docs/data/board-performance/board-kernel-timing-20260925.md)
和[撤回记录](archive/tensor-subset-materialization-reverted.md)。历次接续只保存在[历史记录](archive/board-testing-scheduling-history.md)。

## 当前任务队列

`board-testing` 本轮限定的全矩阵 timing 复验已完成；LM/Tensor 整改保存在 WIP 分支，未完成部分不改记为通过。

| 顺序 | Work item | 状态 | Owner | 直接输入 | 完成门禁 | 实施计划 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `board-testing` | `done` | 16；关联15及各现有pipeline owner | 恢复版本的原始Torch XLA case、独立reference、current compiler/runtime及profile元数据 | 本轮80项常规配置、84个step的fresh no-card、单次实卡、完整输出/guard、16 Tile kernel timing及健康全部通过；三项biased-conv紧凑写入与ResNet18可选profile预算修复闭合。额外长LM S1024/1025仅完成编译复查，仍未通过且无timing，不计入80项验收。后续优化与独立完整主机矩阵继续取消。 | `tasks/archive/board-workload-matrix.md` |
| 2 | `production-host-readiness` | `cancelled` | Q53 / 16 | 原定board-testing之后的current compiler、frontend、interface、package/runtime | 2026-09-25用户取消独立完整主机矩阵重签；保留历史证据，不宣称完成，不再排入当前交付。整改直接受影响的主机检查仍由board-testing负责 | `tasks/archive/physical-dataflow-host-readiness.md` |

已完成的Spatial/collective实现边界见[完成索引](archive/completed-task-index.md#文档整理时移出的已完成队列)。


通信实现独立排队项已收拢，提交与范围见[调度收拢记录](archive/mesh-communication-task-reconciliation.md)。
Q56不再独立排程：package/runtime由15号维护，验证归16号及统一板测；旧`board-ready`的
[核对记录](archive/executable-package-and-resident-runtime.md#2026-09-29状态核对)保留原始资格边界。

## 已满足的直接前置

| Owner | 状态 | Current作用 | 证据入口 |
| --- | --- | --- | --- |
| Q52 | `done` | none/search独立current-IR路径、bounded RegionPlan refinement、public search width/trials、actual feedback和逐Tile actual inventory；合法候选比较现按06号统一标量估时合同 | 06；`tasks/archive/physical-dataflow-synthesis-q52-plan-history.md`；`tasks/archive/completed-task-index.md` |
| 01、04、06、07、10、14、16、18--20 | `done` | 架构、device-scope术语、源码/component边界和canonical build | 编号设计；`tasks/archive/completed-task-index.md` |
| Q50.0 | `done` | policy-complete Instr共同消费的actual SPM/DDR/transport/target leaf | 06、09、12--14 |
| Q51/Q50.S | `done` | structural choice/domain算法和attention semantic/decomposition donor；不作为current事实源 | 05、06；历史见completed index |
| Q55 | `done` | current target/package/runtime interface | `tasks/archive/interface-version-consolidation.md` |
| Q60 | `done` | product frontend与portable StableHLO ingestion | `tasks/archive/compiler-entry-productization.md` |

## 延后候选（重新立项前必须重审）

以下工作不会随当前主线自动启动。它们保留在队列中只是为了记录可能的后续方向；启动前必须重新确认当前产品
调用者、硬件/runtime证据、实现边界和逐项覆盖矩阵。没有新的调用者或证据时，不为旧计划继续维护代码。

| Tracking ID | Semantic key | 状态 | 调度判断 | 重新启动条件 | 完成边界 | Owner / plan |
| --- | --- | --- | --- | --- | --- | --- |
| Q57 | `resident-static-execution` | `later` | 低优先级；只在确有常驻调用者时考虑 | 有明确的resident API调用者、板端窗口和fresh输入；Q53已取消，须先重新授权并满足其host/no-card资格前置；15号package/runtime须通过启动时的host/no-card检查并备齐板端case | 同一package的prepare/submit*/close；load-once、data H2D-once、稳定地址、typed completion和poison板端闭合 | 15--17；`tasks/plans/resident-static-execution.md` |
| Q61 | `whole-program-scale-readiness` | `later` | 低优先级；属于规模资格而非核心编译能力 | 冻结完整程序输入、资源预算和验收owner；Q53已取消，须先重新授权并满足其host/no-card资格前置，同时保持Q58/Q60要求 | 普通产品driver处理完整小模型、data-heavy和graph-heavy程序；work、I/O、wall、RSS、disk和package bytes可对账 | 01--02、06、14--18；`tasks/plans/whole-program-scale-readiness.md` |
| Q48 | `semantic-superoptimization` | `later` | 研究/可选方向；不属于FA/FD核心闭合 | 有明确的语义alternative消费者、收益假设和独立等价oracle；Q53已取消，须重新授权并达到原board-ready前置，再重审TensorProgram表示和Q52 handoff合同 | actual structured MLIR alternatives经query-local等价证明后各自进入Q52 current-IR pipeline；只有accepted owner发布 | 05--08、10--11、16--18；`tasks/plans/semantic-superoptimization.md` |
| Q47 | `target-abi-retirement` | `later` | 仅剩板端资格性质；不作为新的compiler开发项 | current interface/package/transaction已闭合且取得真实板端资格窗口；先核对archive中的旧完成边界与current ABI | current package定向重签ordinary与Direct-DTE host/no-card/board纵向；不回放历史package | 11、14--17、20 |

## 外部证据/可选积压（不进入当前线性主线）

以下条目没有当前产品调用者或本地可完成的证据闭环，统一保持`later`，不作为Q53之后的默认开发顺序。
启动前必须先建立新的编号设计和完成门禁，不能直接沿用旧计划。

| Tracking ID | Semantic key | 当前处置 |
| --- | --- | --- |
| Q9.R | `profile-writing-overhead` | 只有profiler写入成为明确瓶颈时重开 |
| Q22.C/E/K/P | target-model qualification | 分别等待numeric corpus、simulator/ISS、vendor provenance或validated timing evidence |
| Q32.T | `compiler-transform-control` | 只有出现明确external control-plane consumer时重开 |
| Q38.W | `multi-worker-production-promotion` | 只有有独立命令链收益假设和matched board A/B时重开 |
| Q3.6 | `crt-writeback-scalar` | 只有Count predicate、wrapper、target/model证据和readback需求明确时重开 |

不在当前DAG中的distributed/executable dialect、MPMD rank class、跨卡coherent variant、WCRE/global registry、
capability lease、跨model migration、shared-weight cache、segmented MoE及70B/100GB stress，恢复时必须先建立编号设计和完成门禁。

## 导航

- 编号设计和archive索引：`tasks/README.md`。
- Q52/Q53历史验收：`tasks/archive/physical-dataflow-host-readiness.md`；详细历史：
  `tasks/archive/physical-dataflow-synthesis-working-history.md`。
- 已完成边界：`tasks/archive/completed-task-index.md`。
- 硬件校准事实：`docs/tx81-compiler-hardware-calibration.md`。
