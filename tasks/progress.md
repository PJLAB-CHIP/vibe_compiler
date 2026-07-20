# Wafer Compiler Task Queue

更新时间：2026-07-20

本文件只记录当前调度状态、前置关系和紧凑完成索引，不保存逐轮测试数字、实现复盘或历史工作日志。
长期架构与pipeline contract以编号设计文档为准，详细完成证据与实施记录位于`tasks/archive/`，完整导航见
`tasks/README.md`；历史变化由Git保留。

当前发布基线：Q22 repo-owned target-call/SystemC model-only untimed functional-numeric链、Q28标准Llama-2 7B单block
TP16 scale vertical、Q30 production vertical host性能收口及Q31多seed数值表征已经完成；数值纵向直接比较固定source CPU
expected，不再维护
accepted-IR第二套解释器。SystemC受管依赖统一位于`third_party/systemc-model`。板端执行、板端数值相关、exact package执行、
packet provenance和timing仍是独立later/external gate。

共享SPM/DDR static memory packing已从保守greedy升级为默认MiniMalloc fixed-capacity canonical search，并用
确定性宽松work budget、三态结果、独立placement validator和仅限资源耗尽的first-fit fallback闭合编译
资源边界。当前直接进入通用physical-dataflow synthesis；后者先在现有candidate clone和exact gate骨架上完成真实
dependent-tiling/resident rewrite，再以少量MLIR-native候选联合评估implementation、tile、encoding、movement和residency，
最后退役旧decision旁路。当前不建设平行provider/query/schema或通用联合求解器。

## 队列规则

- `Q*`是稳定tracking ID，不表示pipeline层级；执行顺序由状态和前置关系决定。
- 至多一个row标`doing`；`next`表示前置已满足，`later`不会自动进入主线，`blocked`必须写明外部或任务前置。
- `done`只表示对应编号文档中的completion gate已有验证；详细证据不复制到本文件。
- 新任务必须指向编号设计owner；非小修还需在`tasks/plans/`建立实施计划，并确认对应设计文档中的pipeline contract。
- 设计、实现和本队列冲突时，先按`AGENTS.md`优先级收敛事实，再更新状态。

## 当前执行图

```text
已完成compiler/source-oracle/model主线：
Q14
Q0 -> Q15 -> Q16 -> Q17 -> Q18 -> Q16.T
Q5.C + Q16.T -> Q20 -> Q21 -> Q22.R
Q22.R -> Q0.L
  -> Q22.N -> Q22.B
  -> Q22.L -> Q22.H
Q22.N + Q22.H + Q16.T -> Q22.S
Q22.B + Q22.S + Q20 + Q21 -> Q22.V -> Q22

源码模块化：
Q23 -> Q24 -> Q25 -> Q26

static memory packing：
Q26 -> Q34 -> Q32.B

验证consumer收敛：
Q22 -> Q27

tile-dataflow scheduling与7B级单block纵向：
Q22 + Q27 -> Q29 -> Q28 -> Q30 -> Q31

MLIR-native physical-dataflow synthesis：
Q29 + Q28 + Q30 + Q31 -> Q32.I -> Q32.R -> Q32.B -> Q32.M -> Q32.S -> Q32.G -> Q32

later/external：
Q0.L + Q21 + configured board -> Q6.B
Q32 + Q6.B -> Q9
Q22 + Q32 + Q6.B + configured numeric corpus -> Q22.C
Q18 + Q22 + Q32 + configured simulator/ISS -> Q22.E
Q32 + Q22.C + validated PMU/timing environment -> Q22.P
Q22 + owner-approved packet evidence -> Q22.K
Q32 -> Q32.T (optional compiler control plane)
explicit typed target/ABI/model consumer evidence -> Q32.V (independent target capability extensions)
explicit Count semantic/target/model evidence -> Q3.6 (independent typed writeback/ABI/numeric closure)
```

## 当前实施队列

当前无`doing`；Q32.I是唯一`next` row。Q32 umbrella不会让后续row自动进入执行，later/external gate也不会
自动进入主线。

设计就绪边界：Q32.I/R/B/M/S/G采用MLIR interface、可重算analysis、clone rewrite、DialectConversion、现有exact gates和
atomic commit；不再把provider/query/key、shadow frontier、版本化诊断schema、packing objective或target ABI扩展作为主线前置。
下表状态表示实现/验证调度，不表示实现已经完成。board、hardware numeric、simulator/ISS、packet provenance和timing所需
外部事实仍只保留在Later / External Gates。

| Tracking ID | Semantic key | 状态 | 必须满足的前置 | 窄边界 | 设计 owner |
| --- | --- | --- | --- | --- | --- |
| Q32.I | `mlir-native-rewrite-foundation` | `next` | Q29、Q28、Q30、Q31 | fresh重放现有baseline；用source OpInterface/external model和基于MLIR Affine/Presburger/ValueBounds的最小IndexRelation子集，实现第一条dependent-tiling/resident-handoff rewrite；在isolated clone中实际改IR并通过同层verifier。 | 01、06-08、10、16、18；`tasks/plans/physical-dataflow-synthesis.md` A/B |
| Q32.R | `resident-candidate-integration` | `blocked` | Q32.I | 将baseline和少量optimized clone接入现有rank frontier/finalization；每次rewrite后fresh重算analysis，并复用现有tile/instr conversion、SPM/DDR/event/transport/instruction/ABI gates和strict static dominance。 | 06-13、16、18；同计划C |
| Q32.B | `physical-dataflow-test-seam-vertical` | `blocked` | Q32.R、Q34 | 在compiler-private、production-shaped test seam让第一条真实rewrite在rank-count=1/16 source-to-bundle纵向被实际选择；重放package/SystemC/PyTorch、resource和atomic failure gate，baseline在budget或tradeoff时保持合法。此row不切换wafer-compile默认producer，production cutover只由Q32.G/计划F完成。 | 01、06-18；同计划C |
| Q32.M | `physical-rewrite-expansion` | `blocked` | Q32.B | 增加第二条通用shared-input physical-version reuse或projected-view absorption机制；只从SSA/indexing/effect/lifetime识别，候选直接物化typed IR，无机制registry或影子schedule。 | 05-13、16、18；同计划D |
| Q32.S | `bounded-candidate-scaling` | `blocked` | Q32.M | 基于fresh candidate-count/compile-wall证据选择最简单的有限枚举、small frontier或beam；只保留少量planner limits和baseline slot，不做完整IR序列化、packing objective或版本化诊断协议。 | 06、09、12、16、18；同计划E |
| Q32.G | `physical-dataflow-production-cutover` | `blocked` | Q32.S | production只保留MLIR-native candidate owner；删除旧scope/layout/materialization/maximal-resident、未校准scalar-time winner和重复decision facts，重放完整all-rank atomic audit。 | 01、06-18；同计划F |
| Q32 | `physical-dataflow-synthesis` | `blocked` | Q32.G | 至少两类通用rewrite在production winner实际发生；rank-count=1/16、Q28 fixed-seed/Q31 held-out 7B PyTorch/SystemC scale及全部SPM/DDR/event/transport/instruction/ABI/atomic gate fresh通过。不含oriented GEMM、mapped DMA、schema升级、board性能或timing。 | 01、06-18；`tasks/plans/physical-dataflow-synthesis.md` completion audit |

下列later/external gate不会因Q32.I进入`next`自动进入主线。

## Later / External Gates

| Tracking ID | Semantic key | 状态 | 必须满足的前置 / 外部 gate | 窄边界 | 设计 owner |
| --- | --- | --- | --- | --- | --- |
| Q6.B | `runtime-board` | `later` | Q0.L、Q21 + configured board | 用fresh verified package实际执行board lifecycle并比较完整输出。 | 15、16 |
| Q9 | `cost-calibration` | `later` | Q32、Q6.B + profile environment | 只校准Q32合法候选排序，不改变语义合法性。 | 06、16 |
| Q22.C | `target-model-numeric-correlation` | `later` | Q22、Q32、Q6.B + configured numeric corpus | 按capability row用board区分向量和held-out冻结numeric comparator/profile。 | 16、17 |
| Q22.E | `target-model-package-execution` | `later` | Q18、Q22、Q32 + configured simulator/ISS | 原样执行Q32 integrated audit冻结的verified package及all-and-only RISC-V ELF；任何未来schema升级必须先独立完成再作为该gate输入。 | 15、16、17 |
| Q22.K | `target-model-packet-provenance` | `later` | Q22 + owner-approved vendor package或独立公开规范 | 可选关联repo CRT/packet/MMIO；缺失不阻塞数值CModel。 | 14、16、17 |
| Q22.P | `target-model-timing-calibration` | `later` | Q32、Q22.C + validated PMU/timing environment | deferred LT/AT校准；没有RTL/vendor cycle证据不声明cycle accuracy。 | 16、17 |
| Q32.V | `target-capability-extensions` | `later` | explicit typed target/ABI/model consumer evidence | 分别评估mapped DMA、oriented GEMM和winner-derived RequiredCapabilitySet；每项必须有typed Instr/TargetCall、conversion、必要ABI/package readback和model consumer。它们独立于Q32调度，不反向创建planner provider协议，也不作为Q32核心优化前置。 | 08、10、11、14-17 |
| Q32.T | `compiler-transform-control` | `later` | Q32 + explicit external control-plane consumer | 只有出现真实wafer-opt/autotuning consumer后才设计；必须复用同一rewrite/conversion，Transform IR不保存candidate frontier、不替代all-rank coordinator，也不进入wafer-compile或artifact。当前没有冻结param/report schema。 | 01、05-08、10、16、18 |
| Q3.6 | `crt-writeback-scalar` | `later` | explicit Count predicate + wrapper/target/model consumer evidence | static compact-contiguous source到proven-disjoint single-element i32 SPM destination的Count writeback。必须独立闭合typed instruction、effect/completion、ABI/CRT、model evidence和必要package readback；当前predicate/golden/formal-SystemC evidence/board row absent，source/model/board admission保持关闭。它不依赖Q32/Q32.V planner或capability协议。 | 11、14-17 |

新model/distributed/executable dialect、MPMD/rank class、跨卡coherent variant、WCRE/global registry、capability lease、
跨model state migration、共享weight cache、segmented MoE和70B/100GB stress当前不在active DAG；恢复时必须先更新
编号设计和completion gate。

## Done Index

本表只提供状态和证据入口，不复述测试数字或实现过程。

| Tracking ID | Semantic key | 状态 | 完成边界 | 证据 owner |
| --- | --- | --- | --- | --- |
| Q14 | `architecture-baseline` | `done` | 当前单卡纵向架构、事实优先级和历史计划边界已重基线。 | 01、14、15、16；`tasks/archive/12-architecture-evidence-reset.md` |
| Q0 | `target-correctness` | `done` | target conversion、结构保持、physical legality和原子失败窄边界闭合。 | 06、07、09、11、14、16 |
| Q5.C | `workload-corpus` | `done` | 固定PyTorch/XLA source/config/payload/expected corpus及独立CPU oracle。 | 02、16 |
| Q15 | `compiler-driver` | `done` | source到verified rank-local structured tensor program directory及原子发布闭合。 | 01-06、16 |
| Q16 | `executable-bundle` | `done` | all-and-only rank executable与move-only bundle闭合。 | 03、04、06、09、12、13、16 |
| Q17 | `target-artifact-bundle` | `done` | single-lowering target module、device link及原子artifact发布闭合。 | 14、16 |
| Q18 | `manifest-runtime` | `done` | typed manifest、package readback和no-card preflight闭合。 | 15、16 |
| Q16.T | `direct-dte-transport-activation` | `done` | Direct DTE binding、completion、target activation和package投影闭合。 | 13-16 |
| Q20 | `single-card-linear-mlp` | `done` | linear/residual MLP的1/16-rank source/CPU-expected/package纵向链闭合。 | 01、05、06、10-13、15、16 |
| Q21 | `single-card-tiny-llama` | `done` | 16-rank tiny Llama完整source/CPU-expected/package纵向链闭合。 | 01、05、06、10-13、15、16 |
| Q22.R | `target-model-readiness` | `done` | Q21资源与numeric/bulk/SystemC/host seam readiness完成分级。 | 01、10、11、14-17；`tasks/archive/target-model-readiness.md` |
| Q0.L | `target-command-legality-closure` | `done` | typed target profile、format legality、map/reduce lowering和fresh source replay闭合。 | 01、03、04、06、08、10、11、14-16；`tasks/archive/target-command-legality-closure.md` |
| Q22.N | `target-numeric-foundation` | `done` | multi-dtype codec、formal numeric policy/kernel和受管oracle依赖闭合。 | 16、17；`tasks/archive/target-numeric-foundation.md` |
| Q22.L | `target-llvm-module-bundle` | `done` | owner-backed all-rank target LLVM bundle及single-lowering device-link闭合。 | 14、16、17；`tasks/archive/target-llvm-module-bundle.md` |
| Q22.B | `target-bulk-qualification` | `done` | oneDNN exact qualification、runtime admission和no-fallback bulk lane闭合。 | 16、17；`tasks/archive/target-bulk-qualification.md` |
| Q22.H | `target-host-call-frontend` | `done` | same-target-LLVM host frontend、typed decoder和atomic sink闭合。 | 14、16、17；`tasks/archive/target-call-functional-frontend.md` |
| Q22.S | `target-systemc-event-model` | `done` | SystemC functional-event、private memory、Direct DTE和atomic result闭合。 | 16、17；`tasks/archive/systemc-functional-event-model.md` |
| Q22.V | `target-model-source-verticals` | `done` | source-backed formal/bulk/multi-rank完整输出组合闭合。 | 01、16、17；`tasks/archive/target-model-source-verticals.md` |
| Q22 | `target-execution-model` | `done` | model-only untimed functional-numeric capability profile发布完成。 | 01、16、17；`tasks/archive/target-model-completion-audit.md` |
| Q23 | `source-modularity` | `done` | instruction、tile-region到instruction和target numeric按稳定职责拆分，build/test/组织gate闭合且公共语义不变。 | 18；`tasks/archive/source-organization-refactor.md` |
| Q24 | `remaining-source-modularity` | `done` | group/candidate、target LLVM、numeric conformance、frontend program与compiler driver按稳定职责拆分，双配置gate闭合且公共合同不变。 | 18；`tasks/archive/remaining-source-modularity.md` |
| Q25 | `residual-source-modularity` | `done` | reference/model、numeric/bulk、compiler/artifact/package和frontend bridge共11个聚合实现按稳定职责拆分，双配置及真实外部helper gate闭合且公共合同不变。 | 18；`tasks/archive/residual-source-modularity.md` |
| Q26 | `memory-lifetime-analysis` | `done` | instruction loop backedge completion、共享path-sensitive lifetime/packing core、DDR issue-to-fence lifetime及两侧原子offset commit闭合，SPM/DDR各自memory-space、DTE、descriptor和resource合同保持。 | 09、11、12、18；`tasks/archive/memory-lifetime-analysis.md` |
| Q34 | `static-memory-packing` | `done` | SPM/DDR共享packing默认使用受管MiniMalloc fixed-capacity canonical search；精确edge-clique conflict适配、确定性宽松全局work budget、三态result、独立validator和仅限`ResourceExhausted`的first-fit fallback已闭合。 | 09、12、18；`tasks/archive/static-memory-packing.md` |
| Q27 | `reference-executor-retirement` | `done` | accepted-IR第二套解释器、oracle分支和旧CLI退役；CPU expected、typed invocation及target CModel/board differential边界保留。 | 01、16-18；`tasks/archive/reference-executor-retirement.md` |
| Q29 | `tile-dataflow-scheduling` | `done` | structured tensor program直达bounded rank-local task/dataflow candidate、完整traversal、跨region SPM、whole-rank/whole-variant resource gate、旧group executable surface退役及TP16 7B compile-only all-rank package闭合。 | 01、06-13、16；`tasks/archive/tile-dataflow-scheduling.md` |
| Q28 | `llama-7b-block-vertical` | `done` | 标准Llama-2 7B单block TP16从真实source、task-dataflow package到repo-owned SystemC managed-reference执行及完整PyTorch eager output differential闭合；不包含board、exact ELF、性能或timing。 | 02、03、06、09、11、12、16、17；`tasks/archive/llama-7b-block-vertical.md` |
| Q30 | `llama-block-production-performance` | `done` | 保持accepted IR、all-rank package、target command、SystemC行为和完整PyTorch differential不变，收口static movement构造及physical codec重复遍历；7B Release wall time稳定下降。 | 08、10、11、16-18；`tasks/archive/llama-block-production-performance.md` |
| Q31 | `llama-block-numeric-characterization` | `done` | 最终ProgramTensor边界的逐rank abs/ULP统计、非admission多seed 7B重放及预冻结source/model gate收紧闭合；不改变arithmetic、corpus admission或板端policy。 | 02、16-18；`tasks/archive/llama-block-numeric-characterization.md` |
| Q1 | `crt-surface-audit` | `done` | compiler-emitted production CRT symbol/prototype surface审计完成。 | 11、14、16及对应archive |
| Q2-Q3 | `crt-device-symbol-closure` | `done` | production CRT symbol和device-link closure闭合。 | 11、14、16及对应archive |
| Q3.5 | `crt-extended-evidence` | `done` | 扩展CRT surface evidence已分级。 | 11、14、16及对应archive |
| Q13.T | `supporting-doc-tool-decoupling` | `done` | conformance工具从代码事实源推导，不再解析设计文档marker。 | 01、16 |
| Q13.W | `tool-workflow-consistency` | `done` | SystemC canonical third-party root和existing CMake cache切换闭合。 | 16；`tasks/archive/third-party-dependency-root-consistency.md` |
| Q10-Q13 | `historical-design-governance` | `done` | 历史审计、恢复和设计治理工作已归档。 | `tasks/README.md`、`tasks/archive/` |

## 实施计划入口

- Active task：无。
- Next：Q32.I `mlir-native-rewrite-foundation`，计划见`tasks/plans/physical-dataflow-synthesis.md`。Q34证据已归档于
  `tasks/archive/static-memory-packing.md`。
- 新实施计划：`tasks/plans/`。
- 已完成计划和历史证据：`tasks/README.md`的“实施计划导航”和“归档文档”。
