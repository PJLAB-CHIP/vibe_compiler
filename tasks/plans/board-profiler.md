# 16-Tile TSM Profiler 实施计划

状态：Q9 profiler foundation 实施中。本文只拆解实施顺序和验证 checkpoint；稳定的
candidate、target publication、runtime/package 和 verification 合同分别仍由 `tasks/06`、
`tasks/14`、`tasks/15` 和 `tasks/16` 拥有。

## Pipeline contract

```text
Pipeline position:
- Upstream artifact / IR:
  同一 verified source snapshot、ExecutionConfig、TargetProfileId 和 launch ABI 形成的
  reserved conservative baseline、production winner、final Instr / TargetCall 及 verified package。
- Current stage responsibility:
  wafer-compile 在不改变普通 production package 的前提下，为同源 baseline / winner 形成
  invocation-local summary/count/trace clones、rank-local TSM site map 和原子 profile companion；
  wafer-run 复用既有 ResourceId resource/expected/output binding，在 configured board 上串行执行、
  校验、回传并分析。
- Output artifact / IR:
  逐字节不变的普通 winner package，以及与其 production manifest SHA-256 精确绑定的 versioned
  profile companion、raw 16-tile measurement evidence、analysis JSON 和离线 HTML report。
  profile evidence 不进入 accepted IR。
- Downstream consumer:
  人工优化评审；Q9 后续只有在 measurement basis、重复稳定性和 held-out 均通过后，才能消费
  冻结 evidence 校准已经合法的 candidate ranking。
- User-level driver / named pipeline:
  wafer-compile <existing arguments> --profile；随后仍使用原 wafer-run board invocation，
  不增加 profiler 专用输入、模式或独立工具。
- Explicit non-goals:
  不新增 wafer-profile executable，不让用户选择 summary/trace，不采集或显示 fence/wait 事件，
  不把 TsmExecute 返回当作 NCC 完成，不把 aggregate PMU 伪装成单指令归因，不修改 firmware，
  不把 heuristic site correlation 解释为稳定 IR provenance 或因果关系，不在本阶段自动回写
  compiler cost policy。
- Completion gate:
  rank-count=16 的 production source-to-package 主链路能同时发布普通 winner 与 profile
  companion；host/no-card 覆盖完整 activation、record decoder、campaign 和 report 协议；configured
  live board 上再以同一 existing resource/expected/output invocation 完成 baseline/winner output
  validation、20 个 primary samples、summary/PMU 和 all-and-only 16-tile TsmExecute count/trace，
  并自动生成 evidence-linked report。没有 fresh live-board 结果时 Q9 保持 `doing`。
```

## Artifact 和接口边界

- `--profile` 是唯一新增 public option；要求 `execution-ranks=16`，与 `--target-model` 冲突。
  `CompilationRequest` 只携带 typed diagnostic intent，不携带输出路径、采样次数或 runtime mode。
- 普通 package 必须与相同输入未开启 profiler 时逐字节一致。`<package>.profile` 先在 transaction
  临时目录形成；`activation.json` 最后写入，精确绑定 production manifest SHA-256，并逐项绑定
  `plan.json`、`variants.json`、`site-map.json` 的 SHA-256。runner 只消费完整且 exact-match 的
  companion；missing、partial、stale 或任一 digest mismatch 均在 board effect 前拒绝。
- companion 只拥有一个自动 protocol。它包含 baseline/winner 两个 variant 的未插桩 execution
  binding，以及每个 variant 各自的 summary/count/trace 三个内部 capture binding，共六个逻辑 capture
  binding。variant 的最终 executable artifact 不同时，它们对应六个独立物理 capture package；若
  reserved baseline 与 winner 是同一 artifact，baseline 的三个 binding 必须复用 winner 已验证的三个
  物理 capture package及其 digest，不制造重复包，并使分析结论保持 inconclusive。它们都不是 user mode，
  也不进入 production manifest schema。
- `site_id` 只在一个 rank 内有效。跨 candidate 展示使用显式版本
  `heuristic-target-call-signature-occurrence-v1`：按 typed target-call signature 及同类 occurrence
  建立启发式 correlation。它不能当作稳定 IR provenance、跨变换 SSA identity 或 causal relation；
  symbol、buffer、resource 名和文件名也不恢复语义。

## Device measurement protocol

- runner 自动发现 sibling companion，复用普通 `wafer-run` 已有的 resource、expected、output 和 board
  参数；不增加 profiler 专用输入、CLI mode、采样参数或另一个可执行文件。所有执行由一个完成环境
  和软硬件 identity 资格检查的固定 session 拥有，单进程串行；首个 timeout、device anomaly、
  writable-output validation、transport status、profile-record guard 或协议异常立即停批，不
  retry/reset/power。
- primary performance 只执行 baseline/winner 的未插桩 execution package：各自 warm-up 后，运行五个
  四次 launch 的平衡 block，顺序在 ABBA/BAAB 之间交替，共 20 个样本；前四个 block 用于拟合，
  第五个完整 hold out。每个样本用 host steady clock 精确包围 submit 到 all-rank trusted completion，
  不包含 allocation、H2D、D2H、load 或 cleanup；只有这些 launch 使用高分辨率 completion observation，
  且每个样本 `max observed poll gap / sample duration <= 0.25%` 才满足 measurement basis。
- summary 是独立 diagnostic launch，只在完整 entry 前后读取 rank-local cycle 和单份 aggregate
  PMU；结束点复用原程序已经验证的 terminal completion，不为测量新增 wait/fence。summary 的 entry
  span、PMU 和可能的 profile/cache 开销不进入 primary speed verdict。
- trace clone 只在 CRT 的 CT、NE、RDMA、WDMA、TDMA 五个 `TsmExecute` 汇聚点记录：
  rank-local sequence、site id、site sub-index、engine、call begin/return cycle、raw adapter result
  和 validity。调用返回跨度只表示 Kcore submit call，不表示 engine execution duration。
- trace 使用 per-rank DDR storage，不保留 SPM。runner 先执行 count preflight，再执行 bounded trace，
  并在 evidence 保留 preflight count、trace `next_sequence`、`dropped_event_count`、raw record flags
  和 terminal state。count 与 trace 必须 exact match；overflow、drop、sequence gap、unknown site、
  guard corruption、非 complete state 或 count mismatch 均 fail closed。
- 每次 launch 都重放相同 input，并完成 all-rank trusted completion、status、D2H、全部 writable
  output 校验和正常 cleanup；capture launch 还必须通过 profile-record guards。若调用方提供 external
  expected，它给出 semantic correctness；否则只允许同一 session 的首个 production-winner warm-up
  按稳定 semantic resource key 建立 exact reference，后续 winner/baseline/capture 全部逐字节比较。
  该 reference 只能证明 repeatability 以及 candidate/instrumentation equivalence，报告必须把 absolute
  correctness 标为 unknown。
- 16 个 rank 的本地 cycle 不能直接比较。当前 foundation 可以显式发布 invalid/unavailable clock
  mapping；此时 16 行 timeline 各自以该 tile 的 entry begin 为原点，不声明 cross-tile ordering、
  cluster issue overlap 或 global critical path。带 uncertainty 的 affine clock mapping 是未来可选增强，
  只有资格合格后才能增加全局顺序视图，不是 foundation 完成前置。

## Analysis 和展示

- primary outcome 只来自 20 个未插桩 execution-package 的 host submit-to-all-completion 样本。
  前四个 block 的 paired effect、95% bootstrap interval 与 `max(1%, 3*MAD noise)`共同给出
  improved/regressed/inconclusive，独立第五 block 必须同方向；输入、poll-gap、协议或 correctness gate
  失败为 invalid。summary/count/trace、tile-local cycle 和 PMU 仅解释变化，绝不代替或修正 speed verdict。
- 跨 candidate 的优化结论要求 production manifest identity、exact companion、environment、output
  equivalence 均合格，而且 baseline/winner executable artifact digest 必须不同。两者 alias 或 artifact
  逐字节相同时，只能报告“无可归因的优化差异”，不得给出 optimization speedup claim。
- 默认 report 先展示 verdict 和 evidence validity，再同时展示 16 行 TSM issue timeline、typed
  physical-topology heatmap、五个 block 的 baseline/winner latency、engine/aggregate-PMU delta、
  changed sites 和 ranked findings。每条 finding 标记 measured、correlated 或 unresolved，并能定位到
  rank-local tile/site/event。clock mapping 无效时，UI 必须同时显示全部 16 个 tile-local 轴和明显的
  “不可跨 tile 排序”说明。
- timeline 没有 fence/wait/ready-order lane，不把 submit API interval画成 engine execution。Direct DTE
  不进入这五类 TSM event count；只有未来另有 typed endpoint/clock 证据时，才可作为独立视图。

## 实施 checkpoints

1. **Typed request 与 companion transaction**：CLI、request、同源 baseline/winner、ordinary package
   identity、最后写入且 exact-hash 的 activation/readback 和 failure atomicity。
2. **Target clone 与 TSM site map**：profile-only target LLVM/CRT clone、五个真实 hook、versioned
   rank-local record ABI、heuristic correlation、count/trace buffer、raw flags/state/drop 和完整 host
   decoder negative。
3. **Runtime campaign**：wafer-run 自动发现、existing ResourceId resource/expected/output 复用、固定 qualified
   session、20 个未插桩 primary samples、独立 summary/count/trace serial launches、16-rank
   output/counter qualification 和 atomic run publication；report generator 由 runner 按
   executable-relative installed resource 定位。
4. **Analyzer 与 report**：deterministic analysis、洞见、16-tile interactive report、invalid/partial
   degradation 和 raw-evidence links。
5. **Completion replay**：普通 compile equivalence、profile source-to-package/no-card、五类 TSM hook、
   synthetic analysis/UI 和 atomic output；configured live board 再执行同源 A/B、held-out 与 diagnostics，
   并确认 hardware gate 未 skipped/unsupported。只完成 no-card foundation 时不得把 Q9 标为完成。
