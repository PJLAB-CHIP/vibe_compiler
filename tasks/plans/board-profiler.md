# 16-Tile Production Artifact Profiler 实施计划

状态：Q9 profiler foundation 实施中。本文只定义当前施工边界和验证 checkpoint；稳定的
physical-dataflow、target publication、runtime/package 和 verification 合同分别仍由 `tasks/06`、
`tasks/14`、`tasks/15` 和 `tasks/16` 拥有。

## Pipeline contract

```text
Pipeline position:
- Upstream artifact / IR:
  同一 verified source snapshot、ExecutionConfig、TargetProfileId 和完整 runtime launch contract
  形成的最终 Instr / TargetCall、target LLVM bundle 和 verified production package。
- Current stage responsibility:
  wafer-compile 保持普通 production package 逐字节不变，并从该最终 artifact 只派生 count、trace
  两种 profile-only diagnostic clone；wafer-run 在一个 qualified board session 中先执行一次未插桩
  Primary，再串行执行 Count 和 Trace，三次均复用普通 package 的 ResourceId 输入、expected 和 output
  binding。
- Output artifact / IR:
  普通 production package，以及与其 production manifest SHA-256 精确绑定的 final-artifact profile
  companion；一次成功 run 只公开 evidence.json、analysis.json 和 index.html。profile evidence 不进入
  accepted IR。
- Downstream consumer:
  用户先读取 Primary 的单次 submit→all-rank trusted-completion 耗时和正确性，再按 tile、engine、
  target site 和通信事件下钻诊断 capture。下游必须区分 Primary、Count、Trace 的证据角色。
- User-level driver / named pipeline:
  wafer-compile <existing arguments> --profile；随后仍使用原 wafer-run board invocation。
- Explicit non-goals:
  不生成 baseline/winner，不默认运行 benchmark 重复，不新增 wafer-profile executable，不让用户选择
  capture，不依赖 vendor profiler/export format，不修改 firmware，不把 TsmExecute 返回或插桩 Trace
  耗时冒充 Primary，不把未校准 Direct-DTE raw counter 宣称为 elapsed time，不自动回写 compiler cost。
- Completion gate:
  fresh host build/unit/lit/no-card 全部通过；相同 source 的普通 package 与 --profile production package
  逐字节一致；configured board 重启后由本轮新构建、新 package、新 launch、新 output 串行完成一次
  Primary、一次 Count、一次 Trace，闭合正确性、容量、all-and-only 16 tile、六 engine 和 Direct-DTE
  活动门禁。
```

## Artifact 和接口边界

- `--profile` 是唯一 public option，要求 `execution-ranks=16`，与 `--target-model` 冲突。
- `<package>.profile` 只包含一个 `final-artifact` execution binding，以及该 artifact 的 `count`、`trace`
  两个内部 diagnostic capture package。不存在 summary、reserved baseline、winner、alias、候选比较或
  第二份 production execution package。
- Trace record header 已同时携带该次 diagnostic launch 的 rank-local entry span、aggregate PMU
  before/after/recovery 和 event stream；另建 summary clone 会复制相同事实并增加一次 launch，因此不进入
  默认合同。
- Count 只为动态 event 数量提供独立容量预检。当前 trace buffer 为固定容量，event 数不能从静态 site 数
  安全推出；在建立可验证的动态上界前，不能把 Count 删成 trace-only。
- companion 在 transaction 临时目录完整形成，最后写 `activation.json`；activation 精确绑定 production
  manifest 和 `plan.json`、`variants.json`、`site-map.json` 的逐文件 SHA-256。missing、partial、stale
  或 digest mismatch 在任何 board effect 前拒绝。
- `site_id` 只在一个 rank 内有效；site map 只解释 final artifact 的 typed target-call ordinal、engine
  和结构位置。名字只用于诊断，不恢复 IR 语义。

## Device measurement protocol

一个 runner session 固定执行三次，单进程串行，首个 timeout、device anomaly、output mismatch、transport
status、record guard 或 terminal state 错误立即停批，不 retry/reset/power：

1. **Primary**：执行一次未插桩 final production artifact，启用高分辨率 completion observer。唯一用户级
   总耗时是 host steady-clock 从第一次 submit 到 all-rank trusted completion 的本次观测值；不再自动
   warm-up，不计算 median/range，也不把单次值称为稳态统计。
2. **Count**：执行 count diagnostic clone，只取得各 tile 动态 event 数并在 Trace 前验证固定 buffer
   capacity。其耗时不进入 Primary。
3. **Trace**：执行 trace diagnostic clone，取得 entry-local clock、aggregate PMU、typed site event、
   Direct-DTE wait/completion 和 raw DTE activity。其耗时和插桩扰动不进入 Primary。

Primary 必须先通过所有 writable output 校验，并按稳定 semantic key 建立同 session reference；Count 和
Trace 还要先通过各自 external expected，再与 Primary reference 精确比较。缺 external expected 时只能声明
本次 Primary 与两个 diagnostic capture 等价，absolute semantic correctness 仍为 unknown。

Count 的 `next_sequence` 必须不超过 Trace capacity；Trace 的 stored count、next sequence、drop count、
flags、terminal state、guard 和 Count preflight 必须 exact match。任一不一致不得伪造完整 timeline。

高分辨率 observer 记录真实最大 poll gap，不主动 sleep/yield。poll resolution 不满足高分辨率标签时只降级
该标签，不能用全局 `Measurement invalid` 抹掉已经成立的 Primary、输出或局部 counter 证据。

## Timing 和 correlation 语义

- Primary duration、Trace event interval、aggregate counter 和 sampled/raw counter 是不同 measurement
  family，必须分栏展示。
- CT、NE、RDMA、WDMA、TDMA 的 aggregate PMU delta 是本 tile 该 engine 的 measured busy cycles；
  多 engine 可重叠，不能相加为 Primary wall time。
- NCC timeline event 是累计 execution counter 发生增长的 bounded observation window，不是零误差指令
  起止。`TsmExecute` begin/return、compiler ready-order、token、fence 或静态 schedule 都不能替代它。
- Direct-DTE timeline event 是真实 `direct_dte_wait`/completion window；channel 0/1 PMU delta只作为
  uncalibrated raw activity，不能命名为 duration。raw counter不可用时仍可保留已经验证的 wait window。
- 16 个 rank 的本地 cycle 不能直接互比。默认 timeline 是所选 tile 的六条 engine lane 和同一 Trace
  entry-local 横轴；只有独立 clock mapping 合格后才允许跨 tile order、overlap 或 critical path。
- 所有坐标先做 exact unsigned validation；倒序 counter/window 只使对应字段 `Invalid` 或
  `Unavailable`，不得产生负 cycle，也不得连带抹掉 Primary。
- event 到 compiler 的关联只接受 `(logical_rank, site_id, sub_index)` 和已验证 typed site map；不按
  symbol spelling、时间重叠或名字推断。

## Report information architecture

一个离线 `index.html` 提供六个相互链接的视图，不增加公开文件：

1. **Overview**：Primary duration、结果校验、capture 状态、4×4 tile 热图、六 engine 聚合和关键 warning。
2. **Trace Timeline**：Card→Tile→Engine resource tree、tile-local ruler、六 lane interval、engine/filter、
   zoom 和 event detail；默认不画跨 tile 绝对时间轴。
3. **Tile / Engine**：所选 tile 的 measured busy、bounded window、Direct-DTE wait/raw activity、worker
   counter 和明确单位。
4. **Program / Sites**：target-call ordinal、symbol、position、correlation key 与 runtime event 的 typed
   下钻。
5. **Communication / DTE**：send/receive role、wait window、raw PMU 和已知/未知 correlation；没有 DTE
   活动时显示 measured zero/empty，不伪造消息。
6. **Diagnostics / Raw**：identity、clock domain、poll resolution、capacity/drop/guard、counter
   before/after/recovery、capture 方法和原始 JSON 入口。

状态词固定使用 `Measured`、`Sampled`、`Bounded`、`Derived`、`Unavailable`、`Incomplete`、`Invalid`；
空白不等于 idle。近白背景、克制 engine 色、直接标签和状态文字共同编码，不只靠颜色。首屏不出现
candidate、winner、baseline、speedup 或多样本统计。

`<package>.profile` 从 compiler 私有 staging 原子发布前，根目录及全部 directory/regular-file 成员统一
设为 `0777`；失败则不发布 production package 或 companion。`runs/current` 只公开 `index.html`、
`analysis.json`、`evidence.json`，current 目标目录和三项同样为 `0777`。

## Correctness repair gates

- profile CRT不能改变普通 completion 语义。Count 继续调用真实 `TsmWaitfinish()`；只有合法 Trace 在相同
  `TASK_DONE == 1`终止条件下用 PMU observation 包围 polling，并临时 enable/恢复 Direct-DTE PMU。
- split 64-bit读取必须把 high-low-high 稳定性写进 validity。无法稳定读取时保留可独立成立的 wait
  window，但拒绝 raw PMU delta。
- companion schema 版本由 runtime 公共常量拥有；C++ serializer、JSON schema、Python validator 和 fixture
  必须通过跨语言 contract test 消费同一版本。
- site map 从未插桩 production target LLVM 生成；Trace clone只作为 capture executable。发布前比较
  rank/site typed identity、engine、correlation 和目标调用顺序。
- runner 在任何 submit 前解析完整 ordered phase export 集合；provider stream/submission 存活期间不再
  解析新 function。
- 同一 companion 的 campaign/publication 由 runner 单进程串行拥有。报告先在临时目录完整生成，再原子
  替换 `runs/current`；内部 capture package 不作为用户报告产物。

## 实施 checkpoints

1. **Count/Trace companion**：普通 package byte-equivalence；一个 execution binding；Count/Trace
   两个 capture；activation/digest/readback/failure atomicity。
2. **Record 和 instrumentation**：Trace 同时消费 entry、aggregate PMU、五类 NCC activity、Direct-DTE
   wait/raw activity；Count/Trace terminal protocol和decoder negative闭合。
3. **Runtime campaign**：固定三次 launch、Primary 首门槛、Count capacity、Trace exact audit、每次 output
   校验、timeout stop、三文件原子发布。
4. **Analyzer/report**：单次 Primary duration、16 tile、六 engine、per-tile timeline、site/DTE下钻、
   partial validity 和无负 cycle。
5. **Completion replay**：完整 host build/unit/lit/no-card；实现和host门禁稳定后，用户明确要求板测时才以
   本轮新 build、新 package、新 launch 和新 output 串行完成板端 gate。

旧 companion v2 / evidence v4 的 14-launch Add 输出只保留为历史审计记录，不能为本合同的
companion v3 / evidence v5 代签。Q9 在含真实 Direct-DTE wait 的新协议板端 gate 完成前保持 `doing`。
