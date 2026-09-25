# 板端性能与实验记录

本页索引逐次测量、根因和验收证据。每条记录对应当时的source、compiler、package及运行环境，
保留原始样本和归因限制；历史通过不签发当前版本资格，历史待办不构成执行授权。
当前任务状态只看[progress](../tasks/progress.md)。新增实验写入对应日期文件，并链接原始JSON/CSV。

## 计时口径

- Primary：生产package的device elapsed time，使用TX stream events。它是本记录比较端到端耗时的指标。
- Kernel timing：显式轻量capture的最长Tile main-entry duration，使用厂商微秒时钟，只在entry首尾采样。
  包含内部CPU、搬运、计算、通信及必要等待；独立prepare phase与外层派发在区间外。不能与Primary混作同一指标。
- Engine PMU：另一轮Trace中每Tile的CT/NE/RDMA/WDMA/TDMA累计执行ns。存在重叠，不包含完整的控制与等待过程；不能相加、
  从Primary相减，或据其单独判断端到端热点。
- Trace：本Tile的Kcore `rdcycle`区间，可定位调用、发指令和等待。未校准周期频率及跨Tile时钟，不换算为Primary的ms；
  Trace还包含插桩扰动。调用区间内的`site-control`混有wrapper、同步和插桩，不能全算为计算或全算为可消除开销。
- 一组单次前后观测不称为稳定均值；多个改动一起测量时只报告组合收益，不虚构逐项收益。

## 按日期查阅

| 日期 | 内容 |
| --- | --- |
| [2026-09-25](board-performance/2026-09-25.md) | 全矩阵 kernel timing 与可选 profile 预算；profile metadata 紧凑写入修复 |
| [2026-09-24](board-performance/2026-09-24.md) | Tensor 子集物化的首轮性能保护 |
| [2026-09-23](board-performance/2026-09-23.md) | NCx case 注册与统一计时；Kernel 本体轻量计时；NCx 输入复用与直接 DMA 整改等 |
| [2026-09-22](board-performance/2026-09-22.md) | 全矩阵数值回归与4096 GEMM性能复核；固定指令参数准备；C908缓存配置与计数器实测等 |
| [2026-09-21](board-performance/2026-09-21.md) | 累加器整链转置的性能回退；厂商模块函数表与局部复制；KQ score、分组广播及重复布局求解等 |
| [2026-09-20](board-performance/2026-09-20.md) | 厂商正常退出后的接续与TDMA固件审计；重启后首个4K prefill的PMU观测；直接TDMA fatal的快速单case观测 |
| [2026-09-19](board-performance/2026-09-19.md) | 13项standard单次实卡验收；最终实现的完整standard验证及优化收益；Prefill落选DTE候选实卡对照等 |
| [2026-09-18](board-performance/2026-09-18.md) | 局部拼接版本TDMA故障与CRT GEMM范围修正；TDMA与GEMM/psum局部链定向板测；CRT修正后的原完整block复测仍触发TDMA等 |
| [2026-09-17](board-performance/2026-09-17.md) | TX runtime 5.7接口适配、Add与4096³ GEMM；算子回归与完整LM索引搬运修复；同runtime复现block退化等 |
| [2026-09-14](board-performance/2026-09-14.md) | 复用排序、容量反馈与搜索调度；全部42个默认search配置的实卡计时；扩展矩阵首轮压测与嵌套view地址修复等 |
| [2026-09-13](board-performance/2026-09-13.md) | 用户指定Add复查；循环子集状态修复；搬运清理与One-Shot重复工作等 |
| [2026-09-12](board-performance/2026-09-12.md) | 全workload搜索预算与下游实现审计；搜索空间修正的实现与主机验证（进行中）；既有 profile 与最终 Instr 的根因复核等 |
| [2026-09-09](board-performance/2026-09-09.md) | Decode共享mask访问与单向接收调度；Decode容量及输出写回复验；完整LLaMA block热点与profile报告开销等 |

## 专题与完整数据

- [恢复版本的完整计时表](data/board-performance/board-kernel-timing-20260925.md)：逐项shape、dtype、kernel与event时间。
- [Attention优化复盘](attention-optimization-retrospective.md)：2026-09-20至22日的取舍与累计结果。
- [TDMA故障复盘](tx81-tdma-timeout-investigation-retrospective.md)及[故障证据](tx81-tdma-fault-localization.md)。
- [搜索空间审计](search-space-audit.md)：2026-09-12冻结版本的实验，不代表当前搜索状态。
