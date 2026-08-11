# Wafer Compiler Task Queue

更新时间：2026-08-11

本文件是任务调度入口，只记录任务状态、前置关系、当前工作、完成门禁和设计/证据owner。具体设计、
pipeline contract、实验结论、测试数字、失败修复过程和历史复盘不在这里重复；分别进入编号设计文档、
`tasks/plans/`、`tasks/archive/`、`docs/`或`memory/`。历史状态变化由Git保留。

## 队列规则

- `Q*`是稳定tracking ID，不表示pipeline层级。
- 全局至多一个`doing`；`next`表示前置已满足但尚未开始，`queued`表示已经进入当前主线但仍等待直接前置闭合，
  `later`表示不进入当前主线。
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
Q43 Vibe Compiler collaboration review materials             [done]
Q42 test gate scope reduction                                [done]
Q42 + Q32.C -> Q41 compiler search scalability               [board-ready]
Q42 + Q39 -> Q40 composed search and DTE overlap             [board-ready]
Q15 + Q18 + Q35 -> Q44 PyTorch source board verticals        [board-ready]
Q32 + Q37 -> Q46 layout movement elimination                 [board-ready]
Q46 compiler closure -> Q47 target ABI retirement            [board-ready]
Q32 + Q38-Q41 + Q46/Q47 compiler mechanics
  -> Q49 current whole-card baseline stabilization           [board-ready]
  -> Q50 capability-preserving mechanism migration           [queued]
  -> Q51 unified whole-DAG search correctness                [queued]
  -> Q52 workload-driven search scalability                  [queued]
Q52 + Q44 source mechanics + Q47 compiler ABI closure
  -> Q53 current production board readiness                  [queued]
Q53 board-ready -> Q48 semantic superoptimization             [later]
Q45 compiler terminology and naming                           [later]
```

| Tracking ID | Semantic key | 状态 | 必须满足的前置 | 当前工作与完成门禁 | 设计 / 计划 owner |
| --- | --- | --- | --- | --- | --- |
| Q9 | `production-artifact-profiler` | `done` | Q32、Q6.B、configured board | 未插桩Primary TX stream launch-to-completion设备包络、分离的host diagnostics、Trace来源的五类NCC per-tile engine active ns/work-volume摘要、独立Direct-DTE cycles/raw activity、Count/Trace、逐次保留真实动态调用与rdcycle的16-tile timeline、exclusive语义成本与非加和Trace-only成本、final Instr静态work与硬件峰值下界对照、exact output、schema-v9单一production artifact companion、三文件专业UI和profile全树`0777`均闭合；不构造card-wide纯engine elapsed，静态cost不回灌ranking。Ranking feedback保留为后续独立门禁。 | 06、14-16；`tasks/archive/board-profiler.md` |
| Q38 | `multi-engine-software-pipelining` | `done` | Q32、Q6.B、Q37 | 历史任务已经证明typed multi-buffer、prologue/steady/epilogue、Direct-DTE issue/wait和exact range hazard可表达、可验证；旧fixed-slot candidate、worker selector及其专用测试已删除。Q50迁移机制后，Q51必须把buffering/worker作为同一whole-DAG state的联合动作直接物化到selected physical-Tile IR，不得恢复独立owner。 | 06、08-17；历史计划已归档 |
| Q39 | `noc-resident-tile-dataflow` | `done` | Q38历史表达/验证能力闭合；板端资格作为独立external gate | physical peer materialization、resident/spill mechanics和板端exact资格已经闭合；旧Tile collective/late selector、静态profitability owner及其artifact合同已删除，communication mechanism由Q50保全迁移，choice统一由Q51 whole-DAG search拥有。 | 02-13、16-17；历史计划已归档 |
| Q43 | `compiler-collaboration-review-materials` | `done` | Q9、Q37-Q39完成证据 | 2026-08-03冻结的历史汇报材料已经闭合；它不作为current architecture合同。 | 01、08–16、19；`tasks/archive/vibe-compiler-collaboration-review.md` |
| Q42 | `test-load-reduction` | `done` | 无 | 默认lit/unit/CTest和owner integration均只判直接合同；历史catalog、campaign、model-scale与重复package执行已退出默认入口。 | 16；`tasks/archive/test-gate-scope-reduction.md` |
| Q40 | `composed-choice-search-and-dte-overlap` | `board-ready` | Q39、Q42完成 | Direct-DTE issue→FP16/BF16 compute→exact wait结构witness、serialized baseline和buffered overlap qualification mechanics已经闭合；旧组合owner及package不再是current production evidence。真实板端matched A/B尚未执行，Q50/Q51只能复用其typed completion/overlap机制。 | 06、08-13、16；历史计划已归档 |
| Q41 | `compiler-search-scalability` | `board-ready` | Q32.C bounded executor、Q42；M-sharded K=1024复现 | stage/pipeline/pass/analysis/candidate计时、RSS、bounded executor和低扰动诊断基础已经闭合；pre-current-whole-card model-scale compile与package仅作历史性能背景，current性能证据由Q52、current package证据由Q53 fresh生成。public policy只保留`search`/`none`，剪枝不得按shape/op/name恢复语义。 | 06、14-16、18；`tasks/archive/compiler-search-scalability.md` |
| Q44 | `pytorch-source-board-verticals` | `board-ready` | Q15、Q18、Q35；Q41通用movement lowering与compile scalability | PyTorch/XLA capture、固定seed、case-owned dtype、同module eager oracle和原dtype/shape比较mechanics已经闭合；旧执行域package与ABI fixture已删除，GEMM、KV-cache decode和Llama workload必须由Q53 current whole-card pipeline fresh重建后才形成新的board-ready证据。 | 02、15-16；历史计划已归档 |
| Q46 | `layout-movement-elimination` | `board-ready` | Q32、Q37；复用Q41已闭合的compiler-side有界搜索与计时基础 | 唯一`MemLayout`事实源、exact relation motion、typed layout choice、fanout共享和cross-Tile physical payload gate mechanics已经闭合；其旧候选owner已退出production，Q50迁移机制后layout/physical encoding只作为Q51同一whole-DAG候选的维度，structured op只走唯一direct typed lowering。真实板端matched性能尚未执行。 | 06-08、10-11、13-14、16-17；历史计划已归档 |
| Q49 | `whole-card-baseline-stabilization` | `board-ready` | Q32 relation/actual-clone/exact-gate mechanics；当前whole-card/CardProgram实现 | `none`已由独立deterministic controller闭合：固定16-Tile、逐Linalg op独立temporal tiling、op间compiler-owned DDR、buffer=1、零fusion/零可选edge-action search；spatial shard不一致时只物化typed relation要求的deterministic peer fragments并在consumer端DDR assembly，不搜索通信方案。2026-08-11最终代码上FP16原始DAG prefill、两步functional decode和Llama均fresh生成完整package并通过no-card，且accepted baseline直接复用唯一exact-admitted executable。尚无本轮真实板端输出，因此不是`done`；Q50仍独立迁移搜索能力。 | 01、03-16、18；`tasks/plans/whole-card-tile-dataflow-synthesis.md`；`test/Tools/wafer-compile-whole-card-baseline.test` |
| Q50 | `whole-card-capability-migration` | `queued` | Q49 | 对当前tracked删除与HEAD旧实现建立capability parity：将layout assignment/fanout共享、ready-order、fixed-slot multi-buffer、NCC worker、peer/collective、complete traversal/fusion以及typed Attention/functional decode分析和online/split-K/V actual materializer改造成current IR上的候选生成、transformation和verifier机制；不恢复旧search selector。每项必须有current接入点、selected actual-IR witness及正负测试后，旧入口才能删除。 | 05-13、16、18；`tasks/plans/whole-card-tile-dataflow-synthesis.md` |
| Q51 | `whole-dag-unified-search` | `queued` | Q50 | 建立唯一whole-DAG选择owner：同一可回溯partial state联合表达semantic DAG、跨op/intra-op spatial mapping、完整iterator temporal tile、multi-op fusion、layout/storage、实际SPM residency、buffer count、NoC/DDR/compute资源时序、worker/order和completion；候选从完整合法域惰性生成，seed只排序，actual packing/admission决定最终合法性。小型DAG以完整枚举oracle证明搜索维度没有断开；`search` winner与`none`走同一materialization/exact gate，并证明存在有效多op融合而非仅单边标记。 | 06-13、16、18；`tasks/plans/whole-card-tile-dataflow-synthesis.md` |
| Q52 | `whole-dag-search-scalability` | `queued` | Q51；复用Q41计时/RSS/work ledger | 先在通用DAG、HF prefill/decode和Llama representative load上记录状态数、重复率、失败原因、clone/packing/cost时间与峰值内存，再按热点引入canonical memo、DP、constraint/no-good cache、dominance/branch-and-bound或局部solver；启发式、beam、遗传/退火等trade-off只能在实际组合爆炸后启用，并以小图最优oracle、`none`质量和融合保全约束校准。10分钟以上必须分析热点，允许继续到30分钟；不得用固定tile/fusion/buffer上限冒充优化。 | 06、14-18；`tasks/plans/whole-card-tile-dataflow-synthesis.md` |
| Q53 | `whole-card-production-readiness` | `queued` | Q52、Q44 source/oracle mechanics、Q47 current ABI closure | 冻结`search`策略并fresh生成通用mixed DAG、official HF prefill、functional KV-cache decode和Llama block的FP16/BF16 package、oracle、runner与no-card证据；逐case证明selected actual IR中的多op融合、中间值SPM驻留、无无意义DDR round-trip及数值一致后标`board-ready`。真实板端只串行执行current matched baseline/winner；Llama及至少一个prefill/decode代表获得可重复改善后才`done`。 | 02、06-16、18；`tasks/plans/whole-card-tile-dataflow-synthesis.md` |

## Later / External Gates

这些任务不会因当前主线完成自动启动；外部条件满足后先更新状态和对应设计。

| Tracking ID | Semantic key | 状态 | 启动条件 | 完成边界 | 设计 owner |
| --- | --- | --- | --- | --- | --- |
| Q9.R | `profile-publication-overhead` | `later` | Q9、进入profiler publication优化排期 | raw evidence只有一个canonical owner，analysis只发布ID/dictionary引用的摘要，HTML不再重复内嵌全量analysis/evidence，raw按需压缩加载；dense fixture证明规模随事件数线性并通过size gate。板端output readback是独立runtime开销，不计入report publication体积。当前generator和测试仍保留宽对象、pretty JSON及全量内嵌等旧路径，本项尚未实现。 | 16 |
| Q22.C | `target-model-numeric-correlation` | `later` | Q22、Q32、Q6.B、configured numeric corpus | 按capability row冻结board区分向量、held-out和numeric comparator；现有workload证据不能单独代签。 | 16、17 |
| Q22.E | `target-model-package-execution` | `later` | Q18、Q22、Q32、configured simulator/ISS | 原样执行verified package及all-and-only RISC-V ELF；schema升级必须先独立完成。 | 15、16、17 |
| Q22.K | `target-model-packet-provenance` | `later` | Q22、owner-approved vendor package或公开规范 | 建立可引用的CRT/packet/MMIO provenance；缺失不阻塞functional CModel。 | 14、16、17 |
| Q22.P | `target-model-timing-calibration` | `later` | Q32、Q22.C、validated PMU/timing environment | 校准LT/AT；没有RTL/vendor cycle证据不声明cycle accuracy。 | 16、17 |
| Q32.T | `compiler-transform-control` | `later` | Q32、明确的external control-plane consumer | 复用现有rewrite/conversion；Transform IR不保存whole-DAG search state，也不替代06的唯一whole-card candidate owner。 | 01、05-08、10、16、18 |
| Q45 | `compiler-terminology-and-naming` | `later` | Q51边界稳定、明确独立迁移窗口 | 审计当前source、IR、analysis、transformation、conversion、diagnostic和设计文档中的长期命名；重点清理仍把执行范围、实现过程和临时产物混成一个概念的名称。每个改名先确定pipeline contract与对应IR / analysis / transformation责任，不只换字符串，不改写archive历史。完成门禁是当前代码、文档、CLI/diagnostic与测试使用同一稳定术语，且名称能直接对应可验证的compiler对象。 | 01、18 |
| Q47 | `target-abi-retirement` | `board-ready` | 用户明确将Q46后续验证与ABI收口合并推进 | TargetProfile、旧single-Tile launch form、旧pointer-block entry ABI及多schema reader已经删除；current worker-aware TargetCall/CRT、Grid/Cluster runtime ABI、schema-v8 package和typed completion收口。fresh current source→package→model/no-card及板端普通/DTE路径通过后才能`done`。 | 11、14-17；旧施工记录见`tasks/archive/target-abi-retirement.md` |
| Q48 | `semantic-superoptimization` | `later` | Q53按current whole-DAG/CardProgram/physical-Tile合同达到`board-ready`，且schema-v8 compiler/runtime closure可消费final whole-card Instr/TargetCall；当前前置未满足 | 从actual structured MLIR与current typed Instr自动生成有界actual alternatives，以query-local SMT证明数学value及外部可观察memory/effect/completion等价；所有候选进入Q51定义、Q52优化后的同一global ledger、whole-card exact gates与numeric selection，不增加独立selector。 | 05-08、10-11、16-18；`tasks/plans/semantic-superoptimization.md` |
| Q38.W | `multi-worker-production-promotion` | `later` | current typed ABI、actual clone和host/model资格已闭合，且出现需要隔离等待域并可能受益的独立命令链 | 以configured-board matched correctness/performance证明非零worker相对worker0流水的明确收益后才允许normal production promotion；不得为使用worker1/2而拆分已能在worker0并行的流水。 | 08、11、14-17 |
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
| Q15 | `compiler-driver` | `done` | Source到verified card-partition program及原子发布闭合；runtime launch kind只保留kernel/model。 | 01-06、14-16 |
| Q16 | `executable-bundle` | `done` | All-and-only physical-Tile executable与move-only whole-card bundle闭合。 | 03、04、06、09、12、13、16 |
| Q17 | `target-artifact-bundle` | `done` | Single-lowering target module、device link和原子artifact发布闭合。 | 14、16 |
| Q18 | `manifest-runtime` | `done` | Typed manifest、package readback和no-card preflight闭合。 | 15、16 |
| Q6.B | `runtime-board` | `done` | Typed kernel/model、grid/cluster和Direct-DTE launch/runtime lifecycle闭合。 | 13-16；`tasks/archive/runtime-board.md` |
| Q16.T | `direct-dte-transport-activation` | `done` | Direct-DTE binding、completion、target activation和package projection闭合。 | 13-16 |
| Q20 | `single-card-linear-mlp` | `done` | Linear/residual MLP的source、CPU expected和纵向mechanics闭合；current package证据由Q53重建。 | 01、05、06、10-13、15、16 |
| Q21 | `single-card-tiny-llama` | `done` | tiny Llama source、CPU expected和纵向mechanics闭合；current package证据由Q53重建。 | 01、05、06、10-13、15、16 |
| Q22.R | `target-model-readiness` | `done` | Numeric、bulk、SystemC和host seam readiness完成分级。 | 01、10、11、14-17；`tasks/archive/target-model-readiness.md` |
| Q0.L | `target-command-legality-closure` | `done` | Typed target format legality及map/reduce lowering闭合。 | 01、03、04、06、08、10、11、14-16；`tasks/archive/target-command-legality-closure.md` |
| Q22.N | `target-numeric-foundation` | `done` | Multi-dtype codec、formal numeric policy/kernel和受管oracle依赖闭合。 | 16、17；`tasks/archive/target-numeric-foundation.md` |
| Q22.L | `target-llvm-module-bundle` | `done` | Owner-backed whole-card physical-Tile target LLVM bundle和single-lowering device link闭合。 | 14、16、17；`tasks/archive/target-llvm-module-bundle.md` |
| Q22.B | `target-bulk-qualification` | `done` | oneDNN exact qualification、runtime admission和no-fallback bulk lane闭合。 | 16、17；`tasks/archive/target-bulk-qualification.md` |
| Q22.H | `target-host-call-frontend` | `done` | Same-target-LLVM host frontend、typed decoder和atomic sink闭合。 | 14、16、17；`tasks/archive/target-call-functional-frontend.md` |
| Q22.S | `target-systemc-event-model` | `done` | SystemC functional-event、private memory、Direct-DTE和atomic result闭合。 | 16、17；`tasks/archive/systemc-functional-event-model.md` |
| Q22.V | `target-model-source-verticals` | `done` | Source-backed formal、bulk和multi-Tile完整输出组合mechanics闭合。 | 01、16、17；`tasks/archive/target-model-source-verticals.md` |
| Q22 | `target-execution-model` | `done` | Model-only untimed functional-numeric capability profile发布完成。 | 01、16、17；`tasks/archive/target-model-completion-audit.md` |
| Q23 | `source-modularity` | `done` | Instruction、tile-region lowering和target numeric按稳定职责拆分。 | 18；`tasks/archive/source-organization-refactor.md` |
| Q24 | `remaining-source-modularity` | `done` | Candidate、target LLVM、numeric、frontend和driver聚合实现拆分。 | 18；`tasks/archive/remaining-source-modularity.md` |
| Q25 | `residual-source-modularity` | `done` | Reference/model、compiler/artifact/package和frontend bridge残余聚合拆分。 | 18；`tasks/archive/residual-source-modularity.md` |
| Q26 | `memory-lifetime-analysis` | `done` | Path-sensitive lifetime/packing、loop backedge和DDR issue-to-typed-completion lifetime闭合。 | 09、11、12、18；`tasks/archive/memory-lifetime-analysis.md` |
| Q34 | `static-memory-packing` | `done` | SPM/DDR共享fixed-capacity canonical packing、validator及限定fallback闭合。 | 09、12、18；`tasks/archive/static-memory-packing.md` |
| Q27 | `reference-executor-retirement` | `done` | Accepted-IR第二解释器和旧CLI退役，CPU expected与model/board differential保留。 | 01、16-18；`tasks/archive/reference-executor-retirement.md` |
| Q29 | `tile-dataflow-scheduling` | `done` | Structured tiling与resource-gate mechanics闭合；其旧执行域和独立candidate owner在current whole-card cutover工作树中已删除，但Q50 parity闭合前不构成能力完成迁移的证据。 | 01、06-13、16；`tasks/archive/tile-dataflow-scheduling.md` |
| Q28 | `llama-7b-block-vertical` | `done` | Llama-2 7B单block TP16 source→package→SystemC/PyTorch differential闭合。 | 02、03、06、09、11、12、16、17；`tasks/archive/llama-7b-block-vertical.md` |
| Q30 | `llama-block-production-performance` | `done` | 保持语义不变，收口static movement和physical codec host开销。 | 08、10、11、16-18；`tasks/archive/llama-block-production-performance.md` |
| Q31 | `llama-block-numeric-characterization` | `done` | ProgramTensor边界统计、多seed 7B characterization及source/model gate闭合。 | 02、16-18；`tasks/archive/llama-block-numeric-characterization.md` |
| Q32.I | `mlir-native-implementation-relation-foundation` | `done` | MLIR-native IndexRelation与typed lowering foundation闭合；旧implementation-choice接口在current whole-card cutover工作树中已删除，current替代能力由Q50/Q51闭合。 | `tasks/archive/mlir-native-implementation-relation-foundation.md` |
| Q32.R | `physical-relation-realization` | `done` | Relation、encoding、transfer和resident handoff realization闭合。 | `tasks/archive/physical-relation-realization.md` |
| Q32.B | `physical-dataflow-test-seam-vertical` | `done` | Production-shaped spill/resident actual-clone seam及late gates闭合。 | `tasks/archive/physical-dataflow-test-seam-vertical.md` |
| Q32.V | `typed-target-capability-vertical` | `done` | Mapped transfer、physical fill和oriented GEMM typed capability纵向闭合。 | `tasks/archive/typed-target-capability-vertical.md` |
| Q32.M | `physical-mechanism-choice-closure` | `done` | 历史任务闭合了recompute、LICM、numeric、residency、ready-order和communication的表达/验证实验；旧独立candidate API已删除，current mechanism由Q50迁移，choice只能由Q51/Q48唯一owner接入。 | `tasks/archive/physical-mechanism-choice-closure.md` |
| Q32.S | `bounded-joint-physical-dataflow-selection` | `done` | Bounded actual-clone组合、whole-card Pareto和target-owned选择闭合。 | `tasks/archive/bounded-joint-physical-dataflow-selection.md` |
| Q32.G | `physical-dataflow-production-cutover` | `done` | `wafer-compile`成为唯一production decision owner，旧旁路退役。 | `tasks/archive/physical-dataflow-production-cutover.md` |
| Q32 | `physical-dataflow-synthesis` | `done` | MLIR-native actual-clone、implementation/relation/layout/storage/order/communication机制与exact-gate基础闭合；旧独立frontier和per-task物化在current cutover工作树中已删除，Q50/Q51迁移与统一owner只由06定义。 | `tasks/archive/physical-dataflow-synthesis-completion-audit.md` |
| Q32.C | `candidate-search-throughput` | `done` | Candidate search有界并发、parse复用和late-gate开销收口。 | `tasks/archive/whole-variant-search-throughput.md` |
| Q32.N | `numeric-algebraic-extension` | `done` | Supported floating algebraic candidates、typed tolerance和整数负例闭合。 | `tasks/archive/numeric-algebraic-extension.md` |
| Q36 | `topology-aware-collective-lowering` | `done` | 历史实验曾闭合Typed topology到Ring/ordered Tree mechanics；其Tile collective IR、late lowering和topology helper现已删除，不属于current architecture。 | `tasks/archive/topology-aware-collective-lowering.md` |
| Q35 | `k-sharded-gemm-board-vertical` | `done` | 16-rank K-sharded f16 GEMM、local compute、AllReduce和board exact纵向闭合。 | `tasks/archive/k-sharded-gemm-board-vertical.md` |
| Q37 | `tx81-compiler-hardware-calibration` | `done` | Compiler-sensitive硬件行为以supported/observed/unknown/excluded和保守策略闭合。 | `docs/tx81-compiler-hardware-calibration.md` |
| Q1 | `crt-surface-audit` | `done` | Compiler-emitted CRT symbol/prototype surface闭合。 | 11、14、16及对应archive |
| Q2-Q3 | `crt-device-symbol-closure` | `done` | Production CRT symbol和device-link closure闭合。 | 11、14、16及对应archive |
| Q3.5 | `crt-extended-evidence` | `done` | 扩展CRT surface evidence完成分级。 | 11、14、16及对应archive |
| Q13.T | `supporting-doc-tool-decoupling` | `done` | Conformance工具不再解析设计文档marker。 | 01、16 |
| Q13.W | `tool-workflow-consistency` | `done` | SystemC canonical dependency root和CMake cache workflow闭合。 | 16；`tasks/archive/third-party-dependency-root-consistency.md` |
| Q10-Q13 | `historical-design-governance` | `done` | 历史审计、恢复和设计治理已归档。 | `tasks/README.md`、`tasks/archive/` |

## 导航

- Q49–Q53 whole-card baseline、能力迁移、统一搜索、scalability与production readiness共用实施计划：
  `tasks/plans/whole-card-tile-dataflow-synthesis.md`。
- Q48语义驱动superoptimizer计划：`tasks/plans/semantic-superoptimization.md`。
- Q9完成证据：`tasks/archive/board-profiler.md`。
- 编号设计与归档导航：`tasks/README.md`。
- 硬件校准结论：`docs/tx81-compiler-hardware-calibration.md`。
