# Wafer Compiler Task Queue

更新时间：2026-07-30

本文件是任务调度入口，只记录任务状态、前置关系、当前工作、完成门禁和设计/证据owner。具体设计、
pipeline contract、实验结论、测试数字、失败修复过程和历史复盘不在这里重复；分别进入编号设计文档、
`tasks/plans/`、`tasks/archive/`、`docs/`或`memory/`。历史状态变化由Git保留。

## 队列规则

- `Q*`是稳定tracking ID，不表示pipeline层级。
- 全局至多一个`doing`；`next`表示前置已满足但尚未开始，`later`表示不进入当前主线。
- `done`只表示对应设计文档的completion gate已经满足；详细证据只链接owner，不复制到本文件。
- `blocked`必须写明尚缺的外部条件或上游任务，不能用工作过程代替阻塞原因。
- 新任务先进入本表并绑定编号设计owner；非小修再建立`tasks/plans/`实施计划。
- 状态变化只更新对应row；不得追加按日期、轮次或测试批次展开的worklog。
- 后续任务gate只验证直接影响面；除非任务本身改动相关ABI/runtime/firmware，否则禁止全量审计和无关回归。
- 包含板端验证的任务必须把`board-ready`写为无卡阶段门禁；`board-ready`不得标记为`done`。

## 当前调度

```text
Q32 + Q6.B -> Q9 profiler foundation                         [done]
Q32 + Q6.B + Q37 -> Q38 multi-engine software pipelining    [done]
Q38 -> Q39 NoC-resident tile dataflow                        [done]
Q43 Vibe Compiler collaboration review materials             [doing]
Q42 test gate scope reduction                                [done]
Q42 + Q32.C -> Q41 compiler search scalability               [board-ready]
Q42 + Q39 -> Q40 composed search and DTE overlap             [board-ready]
```

| Tracking ID | Semantic key | 状态 | 必须满足的前置 | 当前工作与完成门禁 | 设计 / 计划 owner |
| --- | --- | --- | --- | --- | --- |
| Q9 | `production-artifact-profiler` | `done` | Q32、Q6.B、configured board | 未插桩Primary TX stream launch-to-completion设备包络、分离的host diagnostics、Trace来源的五类NCC per-tile engine active ns/work-volume摘要、独立Direct-DTE cycles/raw activity、Count/Trace、逐次保留真实动态调用与rdcycle的16-tile timeline、exclusive语义成本与非加和Trace-only成本、final Instr静态work与硬件峰值下界对照、exact output、三文件专业UI和profile全树`0777`均闭合；不构造card-wide纯engine elapsed，静态cost不回灌ranking。Ranking feedback保留为后续独立门禁。 | 06、14-16；`tasks/archive/board-profiler.md` |
| Q38 | `multi-engine-software-pipelining` | `done` | Q32、Q6.B、Q37 | complete-rank actual clone、真实fixed-slot multi-buffer、prologue/steady/epilogue、waitfinish normal form、V3 typed worker与Direct-DTE prepare/issue/exact wait-release、issue-time exact range hazard、package/no-card、TargetCall/SystemC、profiler和digest-bound qualification companion均闭合；worker0 rotating SPM FP16/BF16 RDMA+CT+WDMA普通production winner已经fresh板端exact correctness与matched资格。NoC-resident dataflow归Q39；全局choice组合与更广Direct-DTE/compute overlap归Q40。 | 06、08-17；`tasks/plans/multi-engine-software-pipelining.md` |
| Q39 | `noc-resident-tile-dataflow` | `done` | Q38非板端Direct-DTE/fixed-slot合同闭合；Q38板端资格作为独立external gate | complete-rank NoC-resident candidate、静态profitability、package/no-card闭合；K-sharded `4096³`同源baseline/winner板端6/6 exact，winner profile保持16-rank exact并生成有效报告。高wait与未闭合overlap转交Q40，编译搜索耗时转交Q41。 | 02-13、16-17；`tasks/plans/noc-resident-tile-dataflow.md` |
| Q43 | `compiler-collaboration-review-materials` | `doing` | Q9、Q37-Q39完成证据 | 重做面向compiler/runtime/hardware工程师的高密度技术汇报。当前门禁先交付production pipeline、strided descriptor、fixed-slot multi-engine三页代表样稿及PPTX/PDF/逐页预览；每页同时包含结论、主技术图、真实IR或数据、对象级标注、compiler影响和证据边界。旧37页版本不作为当前交付，三页视觉与信息密度确认后再扩展完整汇报。 | 01、06、16；`tasks/plans/vibe-compiler-collaboration-review.md` |
| Q42 | `test-load-reduction` | `done` | 无 | 默认lit/unit/CTest和owner integration均只判直接合同；历史catalog、campaign、model-scale与重复package执行已退出默认入口，保留的source-to-package/no-card seam通过。 | 16；`tasks/plans/test-gate-scope-reduction.md` |
| Q40 | `composed-choice-search-and-dte-overlap` | `board-ready` | Q39、Q42完成 | bounded whole-variant组合、V3 same-block Direct-DTE issue→FP16/BF16 CT/NE→exact wait结构witness、同tuple serialized baseline、16-rank replicated FP16 elementwise完整双package及fresh no-card已闭合；两包保持同source/cluster launch/transport ABI/binding/call inventory并以scheduler顺序区分。真实板端exact-output和matched A/B尚未执行，不得标`done`。 | 06、08-13、16；`tasks/plans/composed-choice-search-and-dte-overlap.md` |
| Q41 | `compiler-search-scalability` | `board-ready` | Q32.C bounded executor、Q42；M-sharded K=1024复现 | 搜索资源与typed optimization configuration均已闭合：18个稳定语义轴支持production/none preset及逐项enable/disable，配置贯穿同一source-to-package pipeline；none唯一保留fully gated baseline，reciprocal单轴四路A/B证明最终ELF结构差异及组合等价，M-sharded K=1024 ordinary/profile production package逐字节一致且fresh no-card通过。搜索仍保持rank-invariant canonical sharding、显式资源上界和wall/RSS观测。真实板端exact-output及winner profile尚未执行，不得标`done`。 | 06、14-16、18；`tasks/plans/compiler-search-scalability.md` |

## Later / External Gates

这些任务不会因当前主线完成自动启动；外部条件满足后先更新状态和对应设计。

| Tracking ID | Semantic key | 状态 | 启动条件 | 完成边界 | 设计 owner |
| --- | --- | --- | --- | --- | --- |
| Q9.R | `profile-publication-overhead` | `later` | Q9、进入profiler publication优化排期 | raw evidence只有一个canonical owner，analysis只发布ID/dictionary引用的摘要，HTML不再重复内嵌全量analysis/evidence，raw按需压缩加载；dense fixture证明规模随事件数线性并通过size gate。板端output readback是独立runtime开销，不计入report publication体积。当前generator和测试仍保留宽对象、pretty JSON及全量内嵌等旧路径，本项尚未实现。 | 16 |
| Q22.C | `target-model-numeric-correlation` | `later` | Q22、Q32、Q6.B、configured numeric corpus | 按capability row冻结board区分向量、held-out和numeric comparator；现有workload证据不能单独代签。 | 16、17 |
| Q22.E | `target-model-package-execution` | `later` | Q18、Q22、Q32、configured simulator/ISS | 原样执行verified package及all-and-only RISC-V ELF；schema升级必须先独立完成。 | 15、16、17 |
| Q22.K | `target-model-packet-provenance` | `later` | Q22、owner-approved vendor package或公开规范 | 建立可引用的CRT/packet/MMIO provenance；缺失不阻塞functional CModel。 | 14、16、17 |
| Q22.P | `target-model-timing-calibration` | `later` | Q32、Q22.C、validated PMU/timing environment | 校准LT/AT；没有RTL/vendor cycle证据不声明cycle accuracy。 | 16、17 |
| Q32.T | `compiler-transform-control` | `later` | Q32、明确的external control-plane consumer | 复用现有rewrite/conversion；Transform IR不保存frontier、不替代all-rank coordinator。 | 01、05-08、10、16、18 |
| Q38.W | `multi-worker-production-promotion` | `later` | V3 typed ABI、actual clone和host/model资格已闭合，且出现需要隔离等待域并可能受益的独立命令链 | 以configured-board matched correctness/performance证明非零worker相对worker0流水的明确收益后才允许normal production promotion；不得为使用worker1/2而拆分已能在worker0并行的流水。 | 08、11、14-17 |
| Q3.6 | `crt-writeback-scalar` | `later` | 明确Count predicate及wrapper/target/model evidence | 独立闭合typed instruction、effect/completion、ABI/CRT、model和必要package readback。 | 11、14-17 |

不在当前DAG中的model/distributed/executable dialect、MPMD/rank class、跨卡coherent variant、
WCRE/global registry、capability lease、跨model state migration、共享weight cache、segmented MoE及
70B/100GB stress，恢复时必须先建立编号设计和completion gate。

## Done Index

本表只保留完成边界和证据入口。实现过程、测试数量和性能样本以对应owner为准。

| Tracking ID | Semantic key | 状态 | 完成边界 | 证据 owner |
| --- | --- | --- | --- | --- |
| Q14 | `architecture-baseline` | `done` | 单卡纵向架构、事实优先级和历史计划边界已重基线。 | 01、14-16；`tasks/archive/12-architecture-evidence-reset.md` |
| Q0 | `target-correctness` | `done` | Target conversion、结构保持、physical legality和原子失败边界闭合。 | 06、07、09、11、14、16 |
| Q5.C | `workload-corpus` | `done` | 固定source/config/payload/CPU expected corpus及独立oracle。 | 02、16 |
| Q15 | `compiler-driver` | `done` | Source到verified rank-local program directory及原子发布闭合；runtime launch kind只保留kernel/model。 | 01-06、14-16 |
| Q16 | `executable-bundle` | `done` | All-and-only rank executable与move-only bundle闭合。 | 03、04、06、09、12、13、16 |
| Q17 | `target-artifact-bundle` | `done` | Single-lowering target module、device link和原子artifact发布闭合。 | 14、16 |
| Q18 | `manifest-runtime` | `done` | Typed manifest、package readback和no-card preflight闭合。 | 15、16 |
| Q6.B | `runtime-board` | `done` | Typed kernel/model、grid/cluster和Direct-DTE launch/runtime lifecycle闭合。 | 13-16；`tasks/archive/runtime-board.md` |
| Q16.T | `direct-dte-transport-activation` | `done` | Direct-DTE binding、completion、target activation和package projection闭合。 | 13-16 |
| Q20 | `single-card-linear-mlp` | `done` | Linear/residual MLP的1/16-rank source、CPU expected和package纵向闭合。 | 01、05、06、10-13、15、16 |
| Q21 | `single-card-tiny-llama` | `done` | 16-rank tiny Llama source、CPU expected和package纵向闭合。 | 01、05、06、10-13、15、16 |
| Q22.R | `target-model-readiness` | `done` | Numeric、bulk、SystemC和host seam readiness完成分级。 | 01、10、11、14-17；`tasks/archive/target-model-readiness.md` |
| Q0.L | `target-command-legality-closure` | `done` | Typed target profile、format legality及map/reduce lowering闭合。 | 01、03、04、06、08、10、11、14-16；`tasks/archive/target-command-legality-closure.md` |
| Q22.N | `target-numeric-foundation` | `done` | Multi-dtype codec、formal numeric policy/kernel和受管oracle依赖闭合。 | 16、17；`tasks/archive/target-numeric-foundation.md` |
| Q22.L | `target-llvm-module-bundle` | `done` | Owner-backed all-rank target LLVM bundle和single-lowering device link闭合。 | 14、16、17；`tasks/archive/target-llvm-module-bundle.md` |
| Q22.B | `target-bulk-qualification` | `done` | oneDNN exact qualification、runtime admission和no-fallback bulk lane闭合。 | 16、17；`tasks/archive/target-bulk-qualification.md` |
| Q22.H | `target-host-call-frontend` | `done` | Same-target-LLVM host frontend、typed decoder和atomic sink闭合。 | 14、16、17；`tasks/archive/target-call-functional-frontend.md` |
| Q22.S | `target-systemc-event-model` | `done` | SystemC functional-event、private memory、Direct-DTE和atomic result闭合。 | 16、17；`tasks/archive/systemc-functional-event-model.md` |
| Q22.V | `target-model-source-verticals` | `done` | Source-backed formal、bulk和multi-rank完整输出组合闭合。 | 01、16、17；`tasks/archive/target-model-source-verticals.md` |
| Q22 | `target-execution-model` | `done` | Model-only untimed functional-numeric capability profile发布完成。 | 01、16、17；`tasks/archive/target-model-completion-audit.md` |
| Q23 | `source-modularity` | `done` | Instruction、tile-region lowering和target numeric按稳定职责拆分。 | 18；`tasks/archive/source-organization-refactor.md` |
| Q24 | `remaining-source-modularity` | `done` | Candidate、target LLVM、numeric、frontend和driver聚合实现拆分。 | 18；`tasks/archive/remaining-source-modularity.md` |
| Q25 | `residual-source-modularity` | `done` | Reference/model、compiler/artifact/package和frontend bridge残余聚合拆分。 | 18；`tasks/archive/residual-source-modularity.md` |
| Q26 | `memory-lifetime-analysis` | `done` | Path-sensitive lifetime/packing、loop backedge和DDR issue-to-typed-completion lifetime闭合。 | 09、11、12、18；`tasks/archive/memory-lifetime-analysis.md` |
| Q34 | `static-memory-packing` | `done` | SPM/DDR共享fixed-capacity canonical packing、validator及限定fallback闭合。 | 09、12、18；`tasks/archive/static-memory-packing.md` |
| Q27 | `reference-executor-retirement` | `done` | Accepted-IR第二解释器和旧CLI退役，CPU expected与model/board differential保留。 | 01、16-18；`tasks/archive/reference-executor-retirement.md` |
| Q29 | `tile-dataflow-scheduling` | `done` | Structured program到bounded rank-local dataflow candidate及whole-rank resource gate闭合。 | 01、06-13、16；`tasks/archive/tile-dataflow-scheduling.md` |
| Q28 | `llama-7b-block-vertical` | `done` | Llama-2 7B单block TP16 source→package→SystemC/PyTorch differential闭合。 | 02、03、06、09、11、12、16、17；`tasks/archive/llama-7b-block-vertical.md` |
| Q30 | `llama-block-production-performance` | `done` | 保持语义不变，收口static movement和physical codec host开销。 | 08、10、11、16-18；`tasks/archive/llama-block-production-performance.md` |
| Q31 | `llama-block-numeric-characterization` | `done` | ProgramTensor边界统计、多seed 7B characterization及source/model gate闭合。 | 02、16-18；`tasks/archive/llama-block-numeric-characterization.md` |
| Q32.I | `mlir-native-implementation-relation-foundation` | `done` | MLIR-native implementation choice和IndexRelation foundation闭合。 | `tasks/archive/mlir-native-implementation-relation-foundation.md` |
| Q32.R | `physical-relation-realization` | `done` | Relation、encoding、transfer和resident handoff realization闭合。 | `tasks/archive/physical-relation-realization.md` |
| Q32.B | `physical-dataflow-test-seam-vertical` | `done` | Production-shaped spill/resident actual-clone seam及late gates闭合。 | `tasks/archive/physical-dataflow-test-seam-vertical.md` |
| Q32.V | `typed-target-capability-vertical` | `done` | Mapped transfer、physical fill和oriented GEMM typed capability纵向闭合。 | `tasks/archive/typed-target-capability-vertical.md` |
| Q32.M | `physical-mechanism-choice-closure` | `done` | Recompute、LICM、numeric、residency、ready-order和communication choice闭合。 | `tasks/archive/physical-mechanism-choice-closure.md` |
| Q32.S | `bounded-joint-physical-dataflow-selection` | `done` | Bounded actual-clone组合、whole-card Pareto和target-owned选择闭合。 | `tasks/archive/bounded-joint-physical-dataflow-selection.md` |
| Q32.G | `physical-dataflow-production-cutover` | `done` | `wafer-compile`成为唯一production decision owner，旧旁路退役。 | `tasks/archive/physical-dataflow-production-cutover.md` |
| Q32 | `physical-dataflow-synthesis` | `done` | Current implementation/relation/layout/storage/order/communication联合选择主线闭合。 | `tasks/archive/physical-dataflow-synthesis-completion-audit.md` |
| Q32.C | `candidate-search-throughput` | `done` | Candidate search有界并发、parse复用和late-gate开销收口。 | `tasks/archive/whole-variant-search-throughput.md` |
| Q32.N | `numeric-algebraic-extension` | `done` | Supported floating algebraic candidates、typed tolerance和整数负例闭合。 | `tasks/plans/numeric-algebraic-extension.md` |
| Q36 | `topology-aware-collective-lowering` | `done` | Typed topology到Ring/ordered Tree lowering和correctness闭合。 | `tasks/archive/topology-aware-collective-lowering.md` |
| Q35 | `k-sharded-gemm-board-vertical` | `done` | 16-rank K-sharded f16 GEMM、local compute、AllReduce和board exact纵向闭合。 | `tasks/archive/k-sharded-gemm-board-vertical.md` |
| Q37 | `tx81-compiler-hardware-calibration` | `done` | Compiler-sensitive硬件行为以supported/observed/unknown/excluded和保守策略闭合。 | `docs/tx81-compiler-hardware-calibration.md` |
| Q1 | `crt-surface-audit` | `done` | Compiler-emitted CRT symbol/prototype surface闭合。 | 11、14、16及对应archive |
| Q2-Q3 | `crt-device-symbol-closure` | `done` | Production CRT symbol和device-link closure闭合。 | 11、14、16及对应archive |
| Q3.5 | `crt-extended-evidence` | `done` | 扩展CRT surface evidence完成分级。 | 11、14、16及对应archive |
| Q13.T | `supporting-doc-tool-decoupling` | `done` | Conformance工具不再解析设计文档marker。 | 01、16 |
| Q13.W | `tool-workflow-consistency` | `done` | SystemC canonical dependency root和CMake cache workflow闭合。 | 16；`tasks/archive/third-party-dependency-root-consistency.md` |
| Q10-Q13 | `historical-design-governance` | `done` | 历史审计、恢复和设计治理已归档。 | `tasks/README.md`、`tasks/archive/` |

## 导航

- 当前计划：`tasks/plans/noc-resident-tile-dataflow.md`与
  `tasks/plans/multi-engine-software-pipelining.md`分别记录Q39、Q38的configured-board external gate。
- Q9完成证据：`tasks/archive/board-profiler.md`。
- 编号设计与归档导航：`tasks/README.md`。
- 硬件校准结论：`docs/tx81-compiler-hardware-calibration.md`。
