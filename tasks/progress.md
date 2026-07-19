# Wafer Compiler Task Queue

更新时间：2026-07-19

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
资源边界。当前直接进入通用physical-dataflow synthesis；后者先建立policy-free mechanisms，再由唯一solver联合选择implementation、tile、physical
encoding、storage realization、residency和有界局部顺序，并退役旧decision旁路。

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

联合physical-dataflow synthesis：
Q29 + Q28 + Q30 + Q31 -> Q32.I -> Q32.R -> Q32.B -> Q32.V -> Q32.M -> Q32.S -> Q32.G -> Q32

later/external：
Q0.L + Q21 + configured board -> Q6.B
Q32 + Q6.B -> Q9
Q22 + Q32 + Q6.B + configured numeric corpus -> Q22.C
Q18 + Q22 + Q32 + configured simulator/ISS -> Q22.E
Q32 + Q22.C + validated PMU/timing environment -> Q22.P
Q22 + owner-approved packet evidence -> Q22.K
Q32 -> Q32.T (optional compiler control plane)
Q32.V -> Q3.6 (Count typed writeback/ABI mechanical closure)
```

## 当前实施队列

当前无`doing`；Q32.I是唯一`next` row。Q32 umbrella不会让后续row自动进入执行，later/external gate也不会
自动进入主线。

设计就绪边界：Q32.I/R/B/V/M/S/G及Q32 integrated audit、可选Q32.T和Q3.6的**内部**pipeline、typed schema、
provider/search/resource、failure/atomicity与验证合同已经闭合；下表状态仍表示实现/验证调度，不表示这些实现已经完成。
board、hardware numeric、simulator/ISS、packet provenance和timing所需外部事实仍只保留在Later / External Gates。

| Tracking ID | Semantic key | 状态 | 必须满足的前置 | 窄边界 | 设计 owner |
| --- | --- | --- | --- | --- | --- |
| Q32.I | `target-implementation-foundation` | `next` | Q29、Q28、Q30、Q31 | 从当前production structured tensor pipeline冻结fresh baseline，建立versioned SemanticOpDescriptor/scalar DAG、canonical provider query/result/key及current-v1唯一baseline；accepted interface只验证selected合同。 | 01、06-18；`tasks/plans/physical-dataflow-synthesis.md` A/B |
| Q32.R | `physical-relation-route-proof` | `blocked` | Q32.I | 建立closed piecewise quasi-affine IndexRelation、typed outcome/exact-vs-sound/fuel，PhysicalEncoding/view proof、唯一TransferRouteFamily、descriptor cover、InvalidLaneState和完整query key。 | 06-08、10、11、16、18；同计划C |
| Q32.B | `physical-dataflow-baseline-vertical` | `blocked` | Q32.R、Q34 | 通过transaction-local RankFrontierProducer test seam让新provider baseline走同一finalization/all-rank/package/SystemC纵向；reserved baseline原子形成bundle且不调用旧decision owner。 | 01、06-18；同计划D |
| Q32.V | `physical-capability-vertical` | `blocked` | Q32.B | 固定destination-style load/store、两端root-relative DMA offset和staged fallback；闭合invalid-lane、versioned oriented GEMM及schema-v4 RequiredCapabilitySet到model preflight，board数值predicate独立。 | 06、08、10、11、14-18；同计划E |
| Q32.M | `physical-dataflow-rewrite-mechanisms` | `blocked` | Q32.V | 完成按typed IR cut分层的mandatory matrix：tensor relation/tiling/fusion/propagation/absorption，physical 2+ fanout reuse、mapped folding、movement/resident-cut elimination；改写后fresh重算全部analysis/gates。 | 05-13、16、18；同计划F |
| Q32.S | `bounded-physical-dataflow-synthesis` | `blocked` | Q32.M | 只组合通过各自correctness gate的mechanisms，实现bounded/canonical search、resident policy、lazy all-rank join和shared static-packing interval oracle；板端校准前固定UncalibratedStaticOrderV1，reserved baseline与deterministic budget/telemetry闭合。 | 06-13、16、18；同计划G |
| Q32.G | `physical-dataflow-production-cutover` | `blocked` | Q32.S | production只调用新planner；删除旧producer/test selector、scope/layout/materialization/maximal-resident、estimated scalar/discovery fallback及collective schedule options/hard-coded selector与重复facts。 | 01、06-18；同计划H |
| Q32 | `physical-dataflow-synthesis` | `blocked` | Q32.G | 显式v1重放旧profile；显式v2重放rank-count=1/16、mapped/oriented通用source vertical及实际选择新family的Q28 fixed-seed/Q31 held-out 7B PyTorch/SystemC scale gate；再闭合全部SPM/DDR/event/transport/instruction/ABI eligibility/atomic audit。不含board性能或timing。 | 01、06-18；`tasks/plans/physical-dataflow-synthesis.md` completion audit |

下列later/external gate不会因Q32.I进入`next`自动进入主线。

## Later / External Gates

| Tracking ID | Semantic key | 状态 | 必须满足的前置 / 外部 gate | 窄边界 | 设计 owner |
| --- | --- | --- | --- | --- | --- |
| Q6.B | `runtime-board` | `later` | Q0.L、Q21 + configured board | 用fresh verified package实际执行board lifecycle并比较完整输出。 | 15、16 |
| Q9 | `cost-calibration` | `later` | Q32、Q6.B + profile environment | 只校准Q32合法候选排序，不改变语义合法性。 | 06、16 |
| Q22.C | `target-model-numeric-correlation` | `later` | Q22、Q32、Q6.B + configured numeric corpus | 按capability row用board区分向量和held-out冻结numeric comparator/profile。 | 16、17 |
| Q22.E | `target-model-package-execution` | `later` | Q18、Q22、Q32 + configured simulator/ISS | 原样执行Q32 integrated audit冻结的schema-v4 verified package及all-and-only RISC-V ELF；Q22.V v3只作历史证据。 | 15、16、17 |
| Q22.K | `target-model-packet-provenance` | `later` | Q22 + owner-approved vendor package或独立公开规范 | 可选关联repo CRT/packet/MMIO；缺失不阻塞数值CModel。 | 14、16、17 |
| Q22.P | `target-model-timing-calibration` | `later` | Q32、Q22.C + validated PMU/timing environment | deferred LT/AT校准；没有RTL/vendor cycle证据不声明cycle accuracy。 | 16、17 |
| Q32.T | `compiler-transform-control` | `later` | Q32 | 设计已闭合为rank-local structured optimization、显式非production candidate materialization及只读inspection；custom param SSA type实现`TransformParamTypeInterface`并只接收versioned mapped attr。candidate复用rank frontier/materializer和payload可重算的per-rank gates，只发布`RankLocalPlacedPayloadSignatureV1`/rank-local metrics；whole-variant DDR固定`Unavailable(AllRankDDRPlanningNotRun)`，binding与model/board固定`Unverified`。不调用all-rank coordinator、不比较production winner，也不进入wafer-compile或artifact。 | 01、05-08、10、16、18；同计划§5 |
| Q3.6 | `crt-writeback-scalar` | `later` | Q32.V | 内部机械设计已闭合：static compact-contiguous source、proven-disjoint single-element i32 SPM dest、`positive_u32`、I8/F16/BF16/F32 emittable rows、bit-preserving little-endian raw-u32 writeback、registry-driven `SynchronousWriteback`（含ArgMax/ArgMin现有effect欠账同批修复）、v3 110-row per-profile ABI/CRT、fresh non-Count v3 qualification、schema-v4 package identity/readback及唯一`TargetPeripheralCountTransactionV1`。接收未来外部事实的versioned authority registry、`CountSemanticProfileV1`和qualification/admission record也已闭合；当前predicate/golden/formal-SystemC evidence/board row仍absent，因此source/model/board admission保持关闭。 | 11、14-17 |

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
- Next：Q32.I `target-implementation-foundation`，计划见`tasks/plans/physical-dataflow-synthesis.md`。Q34证据已归档于
  `tasks/archive/static-memory-packing.md`。
- 新实施计划：`tasks/plans/`。
- 已完成计划和历史证据：`tasks/README.md`的“实施计划导航”和“归档文档”。
