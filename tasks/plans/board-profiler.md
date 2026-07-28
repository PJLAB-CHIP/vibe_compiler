# 16-Tile Production Artifact Profiler 实施计划

状态：Q9 profiler foundation 实施中。本文只拆解实施顺序和验证 checkpoint；稳定的
physical-dataflow、target publication、runtime/package 和 verification 合同分别仍由 `tasks/06`、
`tasks/14`、`tasks/15` 和 `tasks/16` 拥有。

## Pipeline contract

```text
Pipeline position:
- Upstream artifact / IR:
  同一 verified source snapshot、ExecutionConfig、TargetProfileId 和完整 runtime launch contract
  形成的最终 Instr / TargetCall、target LLVM bundle 和 verified production package。
- Current stage responsibility:
  wafer-compile 保持普通 production package 逐字节不变，并只从该最终 artifact 生成
  summary/count/trace 三种 profile-only capture clone；wafer-run 复用普通 package 的 ResourceId
  输入、expected 和 output binding，在同一 qualified board session 中串行执行、校验和回传。
- Output artifact / IR:
  普通 production package，以及与其 production manifest SHA-256 精确绑定的 final-artifact
  profile companion；一次成功 run 只公开 evidence.json、analysis.json 和 index.html。
  profile evidence 不进入 accepted IR。
- Downstream consumer:
  用户读取最终 production artifact 的整卡耗时、per-tile entry span、per-engine activity 和
  per-tile timeline；折叠诊断用于解释不可用 counter、poll resolution 和协议错误。
- User-level driver / named pipeline:
  wafer-compile <existing arguments> --profile；随后仍使用原 wafer-run board invocation。
- Explicit non-goals:
  不生成 baseline/winner 比较产物，不新增 wafer-profile executable，不让用户选择 capture，
  不依赖 vendor profiler/library/export format，不修改 firmware，不把 TsmExecute 返回当完成，
  不把未校准的 Direct-DTE raw counter 宣称为 wall-clock latency，不自动回写 compiler cost policy。
- Completion gate:
  fresh host build/unit/lit/no-card 全部通过；同一新构建 source 的普通 package 与 --profile
  production package 逐字节一致；configured live board 重启后先完成普通 production package
  correctness/terminal gate，再由同一 source 的 profile package 完成 10 次总耗时、3 次 capture、
  all-and-only 16 tile record/output/guard 校验和三文件报告。
```

## Artifact 和接口边界

- `--profile` 是唯一 public option，要求 `execution-ranks=16`，与 `--target-model` 冲突。
- `<package>.profile` 只包含一个 `final-artifact` execution binding，以及该 artifact 的
  `summary`、`count`、`trace` 三个内部 capture package。不存在 reserved baseline、winner、
  alias、候选比较或第二份 production execution package。
- companion 在 transaction 临时目录完整形成，最后写 `activation.json`；activation 精确绑定
  production manifest 和 `plan.json`、`variants.json`、`site-map.json` 的逐文件 SHA-256。
  missing、partial、stale 或 digest mismatch 在任何 board effect 前拒绝。
- `site_id` 只在一个 rank 内有效；site map 只解释 final artifact 的 typed target-call ordinal、
  engine 和位置。名字只用于诊断，不恢复 IR 语义。

## Device measurement protocol

- 一个 runner session 只做固定 14 次 launch：1 次普通 production warm-up，10 次相同 production
  package 的高分辨率 submit→all-rank trusted-completion 测量，随后各 1 次 summary、count、trace。
  所有 launch 单进程串行；首个 timeout、device anomaly、output mismatch、transport status、
  record guard 或 terminal state 错误立即停批，不 retry/reset/power。
- warm-up 必须先通过全部 writable output 校验，才能建立同 session reference。10 次测量和 3 次
  capture 都必须再次逐资源精确一致；有 external expected 时还要逐次通过独立 expected。
- 高分辨率 completion observer 忙轮询并记录真实最大 poll gap；不主动 sleep/yield。
  poll resolution 单独展示，不再用笼统的 `Measurement invalid` 抹掉其它有效硬件证据。
- summary 在完整 entry 前后读取 rank-local cycle 和 NCC aggregate PMU；每个 CT/NE/RDMA/WDMA/TDMA
  engine 独立判断 counter stability，稳定的 end-start delta 是该 tile 的 hardware busy cycles。
  engine 可并行，不能相加为 wall time。
- trace 在 typed NCC issue 前后及 profile-only local-completion polling 中采样 cumulative PMU。
  只有 counter 实际增长的相邻观测形成 bounded activity window；window 不是零误差指令起止。
- Direct-DTE 不经过 TsmExecute。trace 在真实 `direct_dte_wait` begin/end 记录 send/receive role、
  rank-local wait/completion window 和 DTE channel 0/1 PMU raw execution-counter delta。当前硬件校准只把
  delta 作为活动证据，不把它命名为 wall-clock latency；即使raw delta为零也保留已捕获的真实wait窗口，
  timeline 必须明确这个measurement basis。
- count 必须先于 trace；trace 的 preflight count、next sequence、stored count、drop count、flags、
  state 和 guard 必须全等且 complete，否则不发布报告。
- 16 个 rank 的本地 cycle 不能直接互比。默认 timeline 是所选 tile 的六条 engine lane 和同一
  entry-local 横轴；只有独立 clock mapping 合格后才允许增加 cross-tile order。

## Analysis 和展示

- 首屏只有一个最终 artifact 结果：10 次样本的 median，总范围和输出正确性。每次 sample 是统计输入，
  不是十个不同“最终耗时结论”。
- Tile 总览展示 16 个 entry span；点选 tile 后显示 CT、NE、RDMA、WDMA、TDMA、Direct-DTE 六行。
  NCC 行显示 PMU busy cycles；Direct-DTE 行显示 wait activity 和 raw PMU delta，并明确单位/资格。
- timeline 只展示实际 counter activity window 或 Direct-DTE wait/completion window。所有坐标先做
  exact unsigned validation；倒序 counter/window 变成不可用诊断，不产生负 cycle。
- HTML 不显示候选、winner、baseline、speedup 或差值；颜色同时配直接 engine 标签，所有值带单位。
- `runs/current`稳定入口只公开 `index.html`、`analysis.json`、`evidence.json`；`runs`、current目标目录和三项
  均可由其它用户访问，三文件为 `0777`。

## Correctness repair gates

- profile CRT不能改变原有completion语义。`summary`、`count`以及缺失/无效record binding都继续调用真实
  `TsmWaitfinish()`；只有`trace`在相同`TASK_DONE == 1`终止条件下用PMU observation包围poll。
  带未知flag或不完整trace状态的binding同样属于非trace fallback，不得启用DTE PMU或绕过真实wait。
  target预处理/反汇编或等价host gate必须覆盖宏展开后的fallback和predicate，不能只检查宏或符号存在。
- 只有消费Direct-DTE counter的`trace` capture可以临时enable并恢复DTE PMU；split 64-bit读取必须把
  high-low-high稳定性写进record validity。无法取得稳定读时保留wait窗口但拒绝raw PMU delta，不能静默拼接
  torn value。
- companion schema版本由runtime公共常量拥有；C++ evidence serializer、JSON schema、Python validator和fixture
  必须通过一条跨语言contract test共同消费当前版本，禁止各自硬编码不同值。
- final site map从未插桩production target LLVM生成。trace clone只作为capture executable；发布前要验证它与
  production的rank/site typed identity、engine、correlation和目标调用顺序一致，不能把插桩后的instruction ordinal
  标成final artifact事实。
- summary entry span和trace entry span来自两次独立diagnostic launch，必须分别保存和展示。每个tile的engine
  activity timeline只使用trace自身entry-local横轴；summary span不得作为trace坐标分母。
- evidence v4由single-final-artifact analyzer直接验证和分析。不得构造ABBA/BAAB、baseline/winner、speedup、
  signed candidate delta或字符串清洗后的旧诊断；任一counter/window倒序只使对应字段不可用，不得产生负耗时。
  Direct-DTE event必须带真实完成的wait窗口；缺失`activity_valid`不能降级成“有效的零窗口”。
- runner在任何submit前解析完整ordered phase export集合；prepare/main phase只负责按既定handle提交和等待，
  provider stream/submission存活期间不再解析新function。campaign第一次普通warm-up必须与普通one-shot共享同一
  implementation和`Normal` completion policy，首次失败不得形成可复用session。
- external expected evidence显式记录`exact`或`relaxed-f16` comparison policy，不把“已通过typed comparator”
  错写成bit-exact。普通`wafer-run` invocation自动消费已验证的sibling companion；不存在companion时仍执行普通
  package，不新增第二个runtime profile mode。
- 每个companion默认只有一个稳定`runs/current` report publication；临时目录失败原子清理，历史保留若以后需要必须
  另设显式policy。`runs`、current指向的目录及其`evidence.json`、`analysis.json`、`index.html`都必须可由其它用户
  穿越/读取，三文件权限为`0777`；回收旧目标前必须用evidence中的`run_id`验证目录身份，不能只凭`run-*`名字递归删。
  同一companion的campaign/publication由runner单进程串行拥有；跨进程并发替换current是显式非目标，不额外发布
  可能残留的锁文件或第四份marker。内部capture package不作为用户报告产物。

## 实施 checkpoints

1. **Final-only companion**：普通 package byte-equivalence；一个 execution binding；三个 capture；
   activation/digest/readback/failure atomicity。
2. **Record 和 instrumentation**：五类 NCC PMU activity、Direct-DTE wait/PMU evidence、count/trace
   terminal protocol、完整 decoder negative。
3. **Runtime campaign**：固定 14 launch、普通 production 首门槛、每次 output exact、timeout stop、
   三文件原子发布。
4. **Analyzer/report**：只输出 final artifact；总耗时、16 tile、六 engine、per-tile timeline、
   partial validity 和无负 cycle。
5. **Completion replay**：完整 host build/unit/lit/no-card；用户重启一次后，以本轮新 build、
   新 package、新 launch 和新 output 串行完成板端 gate。没有 fresh live-board 结果时 Q9 保持
   `doing`。
