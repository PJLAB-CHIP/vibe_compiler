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
  binding。Primary 的 production phase 由 TX runtime 在同一 device stream 上用 start/end event pair
  包围，event elapsed time 是最终 kernel/model artifact 的主耗时；host submit 和 host
  launch-to-trusted-completion 只作为分离的诊断量。
- Output artifact / IR:
  普通 production package，以及与其 production manifest SHA-256 精确绑定的 final-artifact profile
  companion；一次成功 run 只公开 evidence.json、analysis.json 和 index.html。profile evidence 不进入
  accepted IR。
- Downstream consumer:
  用户先读取 Primary 的单次 TX stream device-event elapsed time 和正确性，再按 tile、engine、
  target site 和通信事件下钻诊断 capture。host submit、host launch-to-completion、Primary、Count、Trace
  的证据角色和计时域必须分开。
- User-level driver / named pipeline:
  wafer-compile <existing arguments> --profile；随后仍使用原 wafer-run board invocation。
- Explicit non-goals:
  不生成 baseline/winner，不默认运行 benchmark 重复，不新增 wafer-profile executable，不让用户选择
  capture，不依赖 vendor profiler/export format，不修改 firmware，不把 TsmExecute 返回或插桩 Trace
  耗时冒充 Primary，不把 host envelope、`statistics_window`、Kcore `rdcycle`、`tile_clock` metadata
  或未校准 Direct-DTE raw counter 换算/冒充 device elapsed time，不自动回写 compiler cost。
- Completion gate:
  fresh host build/unit/lit/no-card 全部通过；相同 source 的普通 package 与 --profile production package
  逐字节一致；configured board 重启后由本轮新构建、新 package、新 launch、新 output 串行完成一次
  Primary、一次 Count、一次 Trace；Primary 必须取得同 stream、同 production phase 的有效 TX event pair
  和 device elapsed time，并分别保留有效 host submit、host launch-to-completion 及 completion observer
  resolution；同时闭合正确性、容量、all-and-only 16 tile、六 engine 和 Direct-DTE 活动门禁。
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
- 每个 capture binding 除 `record_bytes` 外必须携带并由 loader 精确校验 profiler record ABI。固定
  1 MiB 的旧 Trace CRT 不能仅因 byte size 相同而通过当前 companion；record ABI、companion schema
  或 typed site contract 不一致必须在任何 provider / device effect 前拒绝。
- `site_id` 只在一个 rank 内有效；site map 解释 final artifact 的 typed target-call ordinal、
  `ncc-command`、`ncc-completion` 或 `direct-dte-control` kind、可选 engine 和结构位置。名字只用于
  诊断，不恢复 IR 语义。LocalFence、NCCJoin 和 Direct-DTE control 不能再因没有 NCC issue engine
  而被排除在 profile site contract 外。

## Device measurement protocol

一个 runner session 固定执行三次，单进程串行，首个 timeout、device anomaly、output mismatch、transport
status、record guard 或 terminal state 错误立即停批，不 retry/reset/power：

1. **Primary**：执行一次未插桩 final production artifact，启用高分辨率 completion observer。唯一用户级
   主耗时是 TX runtime 在 production phase 的同一 device stream 上记录的 start/end event elapsed time；
   多 phase invocation 对各 phase 的 device elapsed time 求和。host steady-clock submit 调用耗时、
   从第一次 submit 到 all-rank trusted completion 的 envelope 和实际最大 poll gap 分别保留为诊断，
   不得替代 device elapsed time；不再自动 warm-up，不计算 median/range，也不把单次值称为稳态统计。
2. **Count**：执行 count diagnostic clone，只取得各 tile 动态 event 数并在 Trace 前验证固定 buffer
   capacity。其耗时不进入 Primary。
3. **Trace**：执行 trace diagnostic clone，取得 entry-local clock、aggregate PMU、typed site event、
   Direct-DTE wait/completion 和 raw DTE activity。其耗时和插桩扰动不进入 Primary。

Primary 必须先通过所有 writable output 校验，并按稳定 semantic key 建立同 session reference；Count 和
Trace 还要先通过各自 external expected，再与 Primary reference 精确比较。缺 external expected 时只能声明
本次 Primary 与两个 diagnostic capture 等价，absolute semantic correctness 仍为 unknown。

Count 和 Trace 使用相同的 entry/site/operation hook 结构；Count 不保存 event，也不采 PMU，只让
`next_sequence`逐一计数动态 `TARGET_SITE` container及其child event。因此 Count 的
`next_sequence`必须不超过 Trace capacity，并与 Trace 的 stored count和next sequence精确一致；
Trace的drop count、flags、terminal state和guard还必须独立闭合。completion polling次数只写入一个
completion event的`observation_count`，不是动态event数，不能把poll loop迭代误算成Count/Trace数量差异。
任一preflight不一致、Trace overflow或terminal不完整不得伪造完整timeline。

高分辨率 observer 记录真实最大 poll gap，不主动 sleep/yield。它只资格化 host
launch-to-completion envelope 的终止观察分辨率，不资格化 TX stream device event pair。poll resolution
不满足高分辨率标签时只降级 host diagnostic 标签，不能用全局 `Measurement invalid` 抹掉已经成立的
device elapsed time、Primary 输出或局部 counter 证据。

## Timing 和 correlation 语义

- Primary device elapsed time、host submit、host launch-to-completion、Trace observation interval、
  aggregate execution counter、worker counter 和 Direct-DTE raw counter 是不同 measurement family，
  必须分栏展示。
- Primary 的 TX start/end event 包围同一 stream 上的 production launch，因此它会计入 event 之间的
  device-visible launch/scheduling、Kcore control、NCC issue、completion wait、Direct-DTE lifecycle、
  engine execution 和真实空转；它不是五个 engine busy time 的求和，也不能窄称为纯 engine active
  time。host provider 准备、host submit 调用和 host completion polling 不进入该 device elapsed。
- Trace 是另一份插桩 clone 的另一轮执行。与 production 对应的 phase 在语义上位于 Primary event pair
  内，但其 Trace `rdcycle` 数值只能标为 `proxy`；Trace-only PMU MMIO、event/site bookkeeping、DTE PMU
  开关和替代式 task-status polling 不进入 Primary。每项必须分别携带
  `counts_in_primary_device_elapsed` 与 `magnitude_relation`，不能用一个模糊状态混合“是否计入”和
  “Trace 幅度是否代表生产幅度”。任何由 Trace clone 得出的 cost 都不能提升为 Primary 的精确成本：
  主语义 ledger 只能标为 `proxy`，无法剥离 production 语义与插桩扰动的区间标为 `mixed/proxy`；
  Trace-only cost 只作为不参与主 ledger 求和的 overlay，标为 `no/trace-only`。
- CT、NE、RDMA、WDMA、TDMA 的 aggregate PMU execution delta 按 vendor producer/parser 合同直接以
  nanoseconds 表示本 tile 该 engine 的 measured execution time；不得再标成 cycle，也不得使用
  `tile_clock`换算。多 engine 可重叠，不能相加为 Primary device wall time。
- profiler record 使用显式 event kind 和独立 span，而不是把所有时间塞进一个 engine rectangle。
  `ncc-command`同时携带 typed site envelope、`TsmExecute` submit span 和 engine activity bound；
  completion wait、Direct-DTE aggregate wait、peer-ready、setup/issue、completion wait 和 cleanup
  使用各自 event kind。site envelope、submit、completion 和 DTE phase 均用紧贴真实调用的
  tile-local Kcore `rdcycle` 边界；Trace probe 的 PMU read、event write、site hook、status poll、
  completion-loop bookkeeping、entry setup/teardown使用固定 exclusive cost summary，不得埋入
  production phase。
- canonical site map中的`site_id`是未插桩final target LLVM里的静态关联点；Trace中的
  `TARGET_SITE` container event sequence标识该rank的一次动态site实例，container固定
  `sub_index=0`，其child从1连续递增并共享site ID/envelope。一个动态实例只拥有一个site envelope，
  NCC command与其内部completion wait、Direct-DTE aggregate与其叶子phase都挂在该实例下，不能把
  每条event重新解释成一个site或重复累计同一个envelope。`site_kind`至少区分NCC command、
  NCC completion、Direct-DTE control和Direct-DTE wait；DTE phase只能归属Direct-DTE wait，
  control site允许没有叶子phase。
- NCC activity bound 的一次 sample 必须在五个 counter read 前后各取 `rdcycle`。delta 的安全下界来自
  上一 sample 的 read-begin、上界来自当前 sample 的 read-end；不得再用上一轮 read-end 作为下界。
  zero delta 也保留为 `counter-no-change` observation。连续 same-engine outstanding 无法唯一分配时，
  相关 event 必须标 `attribution-ambiguous`，不能继续把 delta 静默绑到 latest site。
- aggregate/worker counter的validity按`(logical_rank, engine)`局部判定。某一个NCC engine的split read
  无法稳定、恢复值不一致或counter unavailable时，只将该tile该engine的PMU execution字段标为
  `Unavailable`并给出diagnostic；Primary device elapsed、输出正确性、Trace Kcore ledger及其它engine
  counter仍然有效。只有identity/lifecycle/overflow/terminal/output等全局门禁失败时才使整次capture失效，
  不能因一个辅助counter不可用重新制造全局`Measurement invalid`。
- 状态也按measurement field隔离：成对`rdcycle`取得的Trace entry、`TARGET_SITE` envelope、NCC submit /
  completion operation和Direct-DTE operation都标`Measured`；NCC positive-delta activity window才标
  `Bounded`。单个NCC counter不可用时，engine observation标`Unavailable`，同一event的operation span仍保留
  `Measured`，UI必须分栏展示，不能把局部counter状态提升成整个event状态。
- 同一 tile 的 Kcore phase / probe lane 与六条 engine lane 可共享该 tile 的 Trace entry-local
  `rdcycle`横轴。彩色 engine activity 只表示 counter increase 被该 window 包围，不表示矩形内持续
  busy；`counter-no-change`和`attribution-ambiguous`是观测质量 marker，不是可加总 cost。
- Kcore主语义ledger只在同一tile、同一Trace clone、同一`rdcycle` domain内按interval
  union/subtraction做exclusive accounting。entry span依次覆盖：
  `entry-prologue`、动态target-site envelope、`between-site-gap`和`entry-epilogue`；site内再分NCC
  submit、completion wait或Direct-DTE叶子phase，以及`site-control` residual。prologue保存
  `prev=null/next=first-site`，epilogue保存`prev=last-site/next=null`；每个gap必须保存相邻动态实例的
  `prev=(site_id, sub_index)`与`next=(site_id, sub_index)`及boundary/adjacency reason。它们是Trace
  clone中已测得的区间，但prologue、epilogue、gap和site-control都混有尚不能单独归因的调度、控制或插桩
  影响，因此标为`counts_in_primary_device_elapsed=unknown/mixed`与
  `magnitude_relation=mixed/proxy`，不得解释成hardware idle。
- NCC submit、语义completion wait和Direct-DTE叶子phase在production语义上计入Primary，标为
  `counts_in_primary_device_elapsed=yes`；其数值仍来自Trace clone，所以
  `magnitude_relation=proxy`，不能直接从Primary wall time扣除。PMU read、event/site bookkeeping、
  DTE PMU enable/restore、status poll和completion-loop bookkeeping是
  `counts_in_primary_device_elapsed=no`的Trace-only cost；它们只作为非加和overlay展示，不参与上述
  exclusive ledger求和，也不能在没有位置证据时从某个gap/site residual中扣除。entry setup/teardown
  位于entry横轴之外，必须单列，不能冒充entry-prologue/epilogue。
- Direct-DTE aggregate event保留真实wait/completion envelope，并作为raw PMU归属的container；
  叶子phase按peer-ready wait、setup/issue、completion wait和cleanup形成exclusive语义分解，并保留
  send/receive role。同一动态Direct-DTE wait实例有可用叶子phase时，aggregate只作container而不再次
  计入cost；只有没有可用叶子phase时才把aggregate作为明确标注的fallback。channel 0/1 PMU delta只在
  aggregate上作为uncalibrated raw activity，不能命名为duration或下发给叶子phase；raw counter不可用时
  仍可保留已验证的phase window。
- Direct-DTE板端completion gate按manifest中的实际participant集合逐tile检查source event、analysis
  aggregate和至少一个正值`Measured` phase；不能以任意一个tile的活动替代其它participant的动态执行证据。
- 16 个 rank 的本地 cycle 不能直接互比。默认 timeline 是所选 tile 的六条 engine lane 和同一 Trace
  entry-local `rdcycle`横轴；只有独立 clock mapping 合格后才允许跨 tile order、overlap 或 critical path。
- 所有坐标先做 exact unsigned validation；倒序 counter/window 只使对应字段 `Invalid` 或
  `Unavailable`，不得产生负 cycle，也不得连带抹掉 Primary。
- event 到 compiler 的关联只接受
  `(logical_rank, TARGET_SITE.sequence, site_id, sub_index)`动态分组和已验证typed site map；不按
  symbol spelling、时间重叠或名字推断。
- 禁止 `device_elapsed_ns - Σengine_execution_ns`、`host envelope - device elapsed = queue delay`、
  Kcore cycle 与 PMU ns/raw tick 互减、以及跨 tile Kcore cycle 求和。异步 engine execution ns 是
  work volume；它与 Kcore phase breakdown、Primary device wall time分别展示。

## Report information architecture

一个离线 `index.html` 提供七个相互链接的视图，不增加公开文件：

1. **Overview**：Primary TX stream device duration、结果校验、capture 状态、4×4 tile 热图、六 engine
   聚合和关键 warning；host submit/envelope 放在次级诊断，不与主耗时并列命名为 kernel time。
2. **Trace Timeline**：Card→Tile→Engine resource tree、tile-local ruler、Trace-run Kcore语义ledger、
   非加和Trace-only overlay和六engine lane、filter/zoom/event detail；entry-prologue/epilogue、
   site-control和带`prev/next`的between-site-gap均画成可点击区间并解释优化入口，而不是留白；
   NCC engine lane以实心同色系变体画精确`TsmExecute` submit span，以浅色虚线框叠加PMU bounded
   observation envelope；bounded窗口重叠不声明engine并行，真实engine execution ns只作无精确位置的
   work duration。counter-no-change / ambiguous用marker，默认不画跨tile绝对时间轴。
3. **Tile / Engine**：所选 tile 的 measured NCC execution nanoseconds、`rdcycle` bounded window、
   Kcore exclusive cost table、是否计入 Primary、Trace magnitude relation、Direct-DTE phase/raw
   activity、`statistics_window` raw ticks、worker counter 和明确单位。
4. **Program / Sites**：target-call ordinal、symbol、position、correlation key 与 runtime event 的 typed
   下钻。
5. **Communication / DTE**：主界面以“整次通信等待（总计）”和“通信内部步骤”分别展示
   peer-ready、setup/issue、completion wait、cleanup、send/receive role、raw PMU和correlation；
   总计与内部步骤不重复求和，机器字段`aggregate`/`leaf`只在次行保留供审计，没有DTE活动时显示
   measured zero/empty，不伪造消息。
6. **术语说明**：以单一展示词典覆盖全部语义成本、Trace-only成本、measurement status/marker、
   Primary accounting、representativeness、interval role、location、event/site/engine、通信粒度、
   output correctness和validity gate；主视图显示人话名称，同时保留`analysis.json`机器字段。每项说明
   计时边界、单位、是否计入Primary、能否代表engine busy和优化入口。
7. **Diagnostics / Raw**：identity、clock domain、poll resolution、capacity/drop/guard、counter
   before/after/recovery、capture 方法和原始 JSON 入口。

状态词固定使用 `Measured`、`Sampled`、`Bounded`、`Derived`、`Unavailable`、`Incomplete`、`Invalid`；
界面不再用空白表达未知成本。近白背景、克制 engine 色、直接标签、线型/斜纹和状态文字共同编码，
不只靠颜色。首屏不出现 candidate、winner、baseline、speedup 或多样本统计。

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
2. **Record 和 instrumentation**：record ABI 显式携带 typed phase kind、production-correlated /
   trace-only count、site/submit/wait/DTE span、严格 sample bounds、zero-delta/ambiguous 状态和
   exclusive profiler overhead；Count/Trace terminal protocol和decoder negative闭合。
3. **Runtime campaign**：固定三次 launch、Primary 首门槛、Count capacity、Trace exact audit、每次 output
   校验、timeout stop、三文件原子发布。
4. **Analyzer/report**：单次 Primary device duration及分离的host diagnostics、16 tile、六 engine
   execution nanoseconds、per-tile Kcore/Trace/engine timeline、exclusive cost/residual、site/DTE
   下钻、单位矩阵、partial validity 和无负 cycle。
5. **Completion replay**：完整 host build/unit/lit/no-card；实现和host门禁稳定后，用户明确要求板测时才以
   本轮新 build、新 package、新 launch 和新 output 串行完成板端 gate。

旧 companion v2/v3、evidence v4/v6 的 Add 输出以及任何未携带当前 record ABI、Primary TX stream event pair
和Kcore/Trace成本归因的旧输出只保留为历史审计记录，不能为本合同的 companion v4 / evidence v7 代签。
Q9 在含真实 Direct-DTE wait 的
新协议板端 gate 完成前保持 `doing`。

## Historical Add evidence boundary

2026-07-28 的历史 Add campaign 串行执行了 Primary、Count、Trace 三次 launch，未重试或复位；输出
16个resource均 expected-exact，Count/Trace 与 Primary 一致，16 tile 和64个event完整。旧报告中的
`1,029,187 ns`来自 host steady-clock submit→all-rank trusted-completion envelope，只能作为历史 host
diagnostic，不能再称为最终 artifact 的 kernel/model device elapsed time，也不能满足当前 Primary
TX stream event-pair gate。旧报告把 CT、RDMA、WDMA execution delta 标成 cycle 的单位同样已经失效；
这些数值不得直接重标为 ns 或用于当前 per-engine 结论。其 completion observer poll gap 为`11,905 ns`，
只描述该历史 host envelope 的终止观察分辨率。

companion根、内部capture、`runs`、current目标目录和全部regular file均重新逐项验证为`0777`；
current目标仍恰含`index.html`、`analysis.json`、`evidence.json`。该 Add 既没有 Primary TX stream
device-event elapsed time，也没有 Direct-DTE wait，不能用 host envelope 或 measured zero 替代当前
主耗时/Direct-DTE门禁，因此只保留为 stale historical evidence，Q9仍为`doing`。

## Completion evidence

2026-07-28 使用本轮新构建、新package、新launch和新output串行完成最终门禁。板卡heartbeat通过后，
Add与Direct-DTE各执行一个固定Primary→Count→Trace campaign，未并发、未reset或power。Add的未插桩
Primary TX stream device elapsed为`856000 ns`，host submit为`142118 ns`，host
launch-to-completion为`1166716 ns`；Direct-DTE分别为`1328000 ns`、`185727 ns`和`1766194 ns`。
两者均为`valid=true`，16 rank output expected-exact，Count/Trace与Primary一致。

Add每tile Trace entry为`39131–48581 rdcycle`；exclusive语义partition闭合为entry prologue、
NCC submit、site control、between-site gap、completion wait proxy和entry epilogue。Direct-DTE每tile
Trace entry为`197888–238892 rdcycle`，并额外闭合peer-ready wait、setup/issue、completion wait和
cleanup；16个manifest participant均有正值Measured phase。Trace-only overlay按每tile分别记录NCC/DTE
PMU sample、event bookkeeping、status poll、site hook、completion-loop bookkeeping、entry setup和
entry teardown，全部明确为不计入Primary且不与语义partition加和。

两个`<package>.profile`全树非symlink目录和普通文件均为`0777`，每个`runs/current`目标目录恰含
`evidence.json`、`analysis.json`和`index.html`。相关Host unit、report、no-card、device-link /
target-CRT以及SystemC gate均通过；SystemC include-path旧问题本轮未复现，因此没有无依据修改。

## Post-completion timeline clarity repair

Add报告中CT与WDMA的PMU bounded observation envelope共享completion采样上界，因此保守窗口相交；
这不是多buffer证据，也不证明同一SPM上的两个engine真实并行。界面主时间轴改用精确command submit /
operation span，PMU bound只作浅色虚线辅助层，PMU execution ns继续作为无精确起止坐标的measured work
duration。一个engine内的不同事件使用同一engine色系的多档明度、独立边框和site/event详情，不再连成
无法分辨的大色块。机器字段和analysis schema语义保持不变，新增展示角色及operation/activity显式offset
只用于避免UI混淆。每种cost、event、site、engine、status、accounting、reason、correctness和validity均由
同一展示词典驱动；cost卡片给出定义、禁止误读、边界、Primary关系和optimization entry，semantic reason
还显示前后site或边界冲突claimant。Diagnostics对三态semantic correctness使用“通过 / 门禁未满足 /
尚未独立判定”，不再把没有完整independent expected覆盖的`null`误写成`Invalid`。
