# 硬件、运行方法与实验证据

按问题选择入口。任务状态只看[progress](../tasks/progress.md)，编译器各层合同只看[编号设计](../tasks/README.md)。
硬件结论须保留`supported`、`board-observed`、`unknown`、`excluded`及其适用版本和输入范围；历史结果不签发当前资格。

## 硬件与ABI事实

| 入口 | 用途 |
| --- | --- |
| [指令集与编程模型](wafer-hardware-instruction-set-and-programming-model.md) | engine、memory、issue order、completion和SDK生命周期 |
| [寄存器规范](wafer-register-level-instruction-spec.md) | 字段、范围、编码和底层发射证据 |
| [编译器硬件校准](tx81-compiler-hardware-calibration.md) | capability、数值与边界资格；区分主机模型和板端观察 |
| [Current profile行为](tx81-current-profile-hardware-behavior.md) | 当前硬件profile的已知行为、限制与证据强度 |
| [TX8逆向与厂商资料索引](tx8-deps-reverse-engineering/README.md) | firmware、runtime、结构体、symbol、CRT对照及原始出处 |

## 操作与诊断

| 入口 | 用途 |
| --- | --- |
| [板端诊断](board-diagnostics.md) | 诊断工具、采集流程和结果解释 |
| [Kernel计时](kernel-timing.md) | 轻量entry计时、参数及与event/Trace的区别 |
| [TDMA故障定位](tx81-tdma-fault-localization.md) | 特定故障的观测与证据链；不扩大为通用硬件结论 |

## 历史实验与复盘

| 入口 | 用途 |
| --- | --- |
| [性能记录索引](board-performance-results.md) | 统一计时口径及按日期保存的完整记录 |
| [2026-09-25完整计时表](data/board-performance/board-kernel-timing-20260925.md) | 恢复版本指定矩阵的逐项结果 |
| [Attention优化复盘](attention-optimization-retrospective.md) | 2026-09-20至22日的观察、优化取舍及归因限制 |
| [TDMA超时复盘](tx81-tdma-timeout-investigation-retrospective.md) | 故障调查经过、确认事实与仍未知的边界 |
| [搜索空间审计](search-space-audit.md) | 2026-09-12冻结版本的预算矩阵和下游审计 |
| [与Agent协作开发编译器](blog/human-agent-compiler.md) | 截至2026-09-10的项目分享 |

`data/`内JSON/CSV保存原始证据，`images/`保存说明图；从对应正文进入即可，不把文件数量或历史通过率当作当前验证。
