# Wafer Compiler Task Documents

本文只做 `tasks/` 文档导航，不声明新的架构合同，也不作为任务状态的主要事实源。
当前设计边界以编号设计文档为准；执行状态和下一步以 `tasks/progress.md` 任务队列为准。

## 当前设计文档

编号是稳定的文档导航和contract-owner标识，按compiler pipeline语义大致分组，不按创建日期排列。
它不表示严格的transform拓扑、实施优先级或任务依赖；一个owner文档可以覆盖pipeline中的多个位置。
实际执行顺序只读取`tasks/progress.md`中的状态和“必须满足的前置”；该列可为审计重复列出关键传递gate，不等同最小DAG边集。

| 编号 | 文档 | 范围 |
| --- | --- | --- |
| 01 | `tasks/01-architecture.md` | compiler stack主架构：production pipeline、library/CLI transaction、IR/module/package graph、跨层不变量、consumer分支和职责索引 |
| 02 | `tasks/02-frontend-stablehlo-program.md` | StableHLO program directory、产品adapter边界、ProgramDataSource/ProgramDataRange lifetime与frontend验证 |
| 03 | `tasks/03-shardy-spmd.md` | Shardy/XLA的card-level GSPMD output与`num_partitions`；不绑定片内Tile |
| 04 | `tasks/04-topology-execution-mesh.md` | logical card partition mesh与独立target card/Tile topology |
| 05 | `tasks/05-local-compute-normalization.md` | card-partition-local structured compute normalization、single attention op、graph-level FA/FD算法与tensor collective boundary |
| 06 | `tasks/06-physical-dataflow-synthesis.md` | card-local TensorProgram的spatial/region/temporal choice、actual Card/TileRegion、current-IR layout/movement/execution-structure/Instr与actual admission；预算限制planning工作而不预先截断合法域 |
| 07 | `tasks/07-tile-region.md` | selected tile/dataflow structural IR物化以及structural、layout-resolved、physical TileRegion forms；SPM root不跨界，boundary不自动产生movement或join |
| 08 | `tasks/08-physical-realization.md` | physical encoding attr/type语义、valid domain、view、transfer realizability analysis、descriptor cover和selected physical realization |
| 09 | `tasks/09-spm-memory-planning.md` | SPM lifetime/coexistence、fixed-capacity MiniMalloc legality、all-root coverage、validated placement/headroom和accepted offsets；candidate choice仍由06拥有 |
| 10 | `tasks/10-compute-movement.md` | selected Linalg/Tensor/SCF到typed target-abstract compute/movement的确定性lowering、current Tile execution structure、standard MLIR effects/interface reuse和issue/token/fence/wait |
| 11 | `tasks/11-instruction-ir.md` | complete static Tile instruction program、current descriptor/geometry/range/narrowing legality及mapped/physical-fill/oriented typed extension |
| 12 | `tasks/12-ddr-memory-planning.md` | 当前DDR demand/accepted offsets；多DDR分区、state和streaming延后 |
| 13 | `tasks/13-communication.md` | selected Tile edge到typed p2p/staging/token/wait IR、Direct DTE card-scoped verification和completion；multi-card延后 |
| 14 | `tasks/14-target-code-generation.md` | `CardExecutable`的current target identity/format、accepted immutable data preparation、structure-preserving conversion、CRT ABI和atomic target-module writing |
| 15 | `tasks/15-launch-runtime-package.md` | `CardExecutable -> ExecutablePackage`、ProgramTensor/TargetTensor、program-data.bin、TileEntryArgument、runtime memory plan、no-card和board adapter边界 |
| 16 | `tasks/16-verification-contract.md` | `TensorProgram -> CardModule/TileRegion/Instr -> CardExecutable -> ExecutablePackage`的target correctness、data/whole-program scale、CPU oracle、target-model、no-card和board分层gate |
| 17 | `tasks/17-target-execution-model.md` | `CardExecutable`及其same-invocation target LLVM owner消费、multi-dtype numeric、oneDNN bulk、target-call/SystemC untimed CModel与板端numeric correlation边界 |
| 18 | `tasks/18-source-organization.md` | 跨pipeline的源码ownership、translation unit、compiler library/tool/install、构建依赖和测试镜像组织合同；不改变IR/output语义 |
| 19 | `tasks/19-mlir-engineering.md` | 跨IR层的ODS、standard interface、operation-scoped pass/analysis、rewrite/conversion和named nested pipeline工程合同；不重定义01–18语义 |
| 20 | `tasks/20-interface-evolution.md` | 跨compiler/runtime/tool的内部接口演进、持久化格式与ABI版本owner、集中兼容检查和current-only表示合同 |

### Pipeline Owner 索引

下表只帮助定位边界owner，不复制设计合同，也不把跨阶段owner强行线性化：

| Pipeline boundary | Owner文档 |
| --- | --- |
| verified frontend program、产品adapter、program data ownership和当前program directory | 02 |
| pre-SPMD topology和execution mesh | 04 |
| Shardy/SPMD card-partition output | 03 |
| card-partition-local compute normalization、attention graph algorithm与collective boundary | 05；physical planning和selected decomposition分别由06、07、10消费，跨stage正确性证据由16约束 |
| card-local multi-Tile physical-dataflow planning、selected Card/Tile MPMD materialization和physical encoding/transfer | 06、07、08；source implementation interface由10提供，instruction legality由11提供，exact resource/transport gate由09、12、13提供 |
| policy-free physical-dataflow rewrites | 06、07、08；upstream structured utility由05提供，source/direct typed lowering合同由10提供，源码ownership由18约束 |
| source implementation interface、target-abstract compute/movement和instruction legality | 10、11；transfer realizability/descriptor cover只由08拥有 |
| accepted SPM/DDR allocation、lifetime和offset | 09、12；shared lifetime analysis的源码ownership和测试镜像由18约束 |
| Tile peer/collective materialization、Direct DTE completion、card-scoped acceptance与post-memory transport activation | 13；joint choice由06、target/package/verification consumer由14、15、16约束 |
| card-local multi-Tile时空调度、MPMD executable构造和`CardExecutable` | 06；资源/lifetime边界由09、12、13共同约束 |
| target LLVM、accepted immutable data preparation、CRT/device link和staged target module | 14 |
| target-ready data、typed manifest、RuntimeInvocationPlan和board launch | 15 |
| 横跨上述边界的completion evidence | 16 |
| target execution model、multi-dtype numeric/bulk、same-invocation target module消费、SystemC/CModel capability、板端numeric correlation和deferred timing | 17；target module形成与writing合同由14拥有，target/runtime/verification consumer由14、15、16约束 |
| 跨上述边界的源码与构建模块化 | 18；各IR/output语义仍由01-17拥有 |
| 跨上述IR层的MLIR operation scope、pass/analysis manager、interface和rewrite工程合同 | 19；各层具体语义仍由01-18拥有 |
| 跨compiler/runtime/tool的格式与ABI版本边界 | 20；具体字段语义仍由02、11、14-17拥有 |

## 实施计划导航

本节只列仍由current或later队列消费的实施计划。任务状态、顺序和启动条件只读`tasks/progress.md`；已完成计划
只从下方archive索引定位，不能作为current协议。

| 当前或later范围 | 实施计划 | 稳定设计owner |
| --- | --- | --- |
| Q52 search scalability与Q53 production host readiness | `tasks/plans/physical-dataflow-synthesis.md` | 06；直接下游07–16 |
| Q57 resident static execution | `tasks/plans/executable-package-and-resident-runtime.md` | 15–17 |
| Q61 whole-program scale readiness | `tasks/plans/program-data-and-whole-program-scale.md` | 01–02、06、14–18 |
| Q48 semantic superoptimization | `tasks/plans/semantic-superoptimization.md` | 05–08、10–11、16–18 |

Q52当前同时处理baseline回归、两条policy各自的IR膨胀、verifier职责、actual-IR materialization boundary、
layout/movement、current Tile execution structure/rotating storage和后续search scalability；历史Q51 shadow planning state只作
donor/删除输入，不是current设计。Q53只形成
fresh host/package/no-card与board-ready输入，不运行真实设备。
Q49–Q51详细施工与原Q53板端设想位于
`tasks/archive/physical-dataflow-synthesis-working-history.md`，Q54历史整改位于
`tasks/archive/mlir-engineering-remediation.md`。

## 归档文档

`tasks/archive/` 只保存历史 gap review、recovery、audit 和已收口的任务级记录。归档文档可以作为
实现背景或复盘材料，但不作为当前主线架构合同；如果归档内容和 numbered docs 冲突，以当前 numbered
docs、`tasks/progress.md` 和本轮已收敛设计结论为准。

`tasks/archive/completed-task-index.md`集中保存从current progress移出的已完成任务边界和证据入口；它不是动态状态表，
未完成、`doing`、`queued`、`later`和`board-ready`任务仍只在`tasks/progress.md`维护。

归档正文中的编号文档basename、章节号、line range和命令按当时提交快照解释，不保证仍是当前可解析路径；当前owner路径
只从上面的“当前设计文档”表读取。重命名current owner时不机械改写archive，以免篡改历史审计证据。

| 文档 | 原性质 |
| --- | --- |
| `tasks/archive/completed-task-index.md` | 已完成任务的历史边界与证据入口索引；不参与current调度 |
| `tasks/archive/physical-dataflow-synthesis-working-history.md` | 截至2026-08-25的Q49–Q51详细施工、Q52重基线审计和原Q53板端设想；current Q52/Q53计划已重写，禁止从本文件恢复旧顺序或shared materializer |
| `tasks/archive/mlir-engineering-remediation.md` | 已完成Q54的审计、checkpoint和验证记录；current verifier cleanup由Q52拥有，稳定规则由19和AGENTS拥有 |
| `tasks/archive/whole-card-tile-dataflow-synthesis.md` | 2026-08-11至08-13的旧Q49/Q50完整施工计划；旧任务拆法、shortlist和owner合同不再有效 |
| `tasks/archive/whole-rank-tile-dataflow-synthesis.md` | 2026-08-08的structured-DAG/card历史施工计划；已由current physical-dataflow计划替代，旧public policy与bounded candidate set不再有效 |
| `tasks/archive/whole-variant-search-throughput.md` | 已完成Q32.C的passing-ordinal early stop、bounded persistent candidate executor、accepted-module owner import、exact attempt-plan selective parse和fully-gated Pareto前置late ABI/LLVM，并记录优化后Release单次实测 |
| `tasks/archive/k-sharded-gemm-board-vertical.md` | 已完成Q35 full-4096 f16 K-sharded GEMM的production tiling/SPM/Direct-DTE package、纯tiling隔离及16-rank重复板端raw-exact记录 |
| `tasks/archive/runtime-board.md` | 已完成Q6.B的typed TX board provider、kernel/model多tile launch、cluster Direct DTE、failure lifecycle及真实板端重复exact记录 |
| `tasks/archive/physical-dataflow-synthesis-completion-audit.md` | 已完成Q32的七checkpoint证据映射、双配置全量门禁、fixed/held-out 7B scale重放、单一production owner及剩余边界审计 |
| `tasks/archive/physical-dataflow-synthesis.md` | 已完成Q32 MLIR-native bounded physical-dataflow synthesis的施工checkpoint、hard-cap与integrated completion checklist |
| `tasks/archive/physical-dataflow-production-cutover.md` | 已完成Q32.G默认production winner cutover、旧decision surface删除、all-rank双键correspondence及source/bulk数值回归记录 |
| `tasks/archive/bounded-joint-physical-dataflow-selection.md` | 已完成Q32.S的actual-clone有界联合candidate set、reserved baseline、validated card exact cost、target static policy及逐producer whole-winner记录 |
| `tasks/archive/physical-mechanism-choice-closure.md` | 已完成Q32.M的actual-clone recompute/LICM/integer algebra、partial fanout、spill/resident/ready-order、communication alternatives及重复layout/resource/collective/instruction合同删除记录 |
| `tasks/archive/physical-relation-realization.md` | 已完成Q32.R的rich IndexRelation、physical encoding interface、TransferRealizability、destination-style load、relation-backed resident boundary和fresh 7B TP16数值纵向记录 |
| `tasks/archive/typed-target-capability-vertical.md` | 已完成Q32.V的mapped DMA双端offset、physical-footprint fill、source/Tile/Instr oriented GEMM及formal/SystemC fresh纵向记录 |
| `tasks/archive/mlir-native-implementation-relation-foundation.md` | 已完成Q32.I的fresh baseline、source implementation external model、真实reciprocal/division actual-clone纵向、MLIR-backed IndexRelation foundation和custom interface盘点/首轮删除记录 |
| `tasks/archive/static-memory-packing.md` | Q34历史MiniMalloc施工记录；其中曾有的fallback合同已退役，current只使用MiniMalloc并原样传播`ResourceExhausted` |
| `tasks/archive/llama-block-numeric-characterization.md` | 已完成Q31的ProgramTensor逐rank abs/ULP统计、非verification多seed 7B重放及source/model comparator gate收紧记录 |
| `tasks/archive/llama-block-production-performance.md` | 已完成Q30的static movement/physical codec host性能收口、package等价性和完整7B双replay记录 |
| `tasks/archive/llama-7b-block-vertical.md` | 已完成Q28的标准Llama-2 7B单block TP16 source/package、repo-owned SystemC managed-reference和完整PyTorch eager output differential记录 |
| `tasks/archive/tile-dataflow-scheduling.md` | 已完成Q29历史structured tensor program直达bounded task/dataflow scheduling、当时的跨region SPM合同、card结果写入、旧group executable surface退役及7B TP16 compile-only验证记录；current region语义只看06/07和Q52 current plan |
| `tasks/archive/reference-executor-retirement.md` | 已完成Q27的accepted-IR第二套解释器、oracle分支和旧CLI退役，以及CPU-expected到target CModel纵向gate收敛记录 |
| `tasks/archive/memory-lifetime-analysis.md` | 已完成Q26的共享structured lifetime/packing core、DDR issue-to-fence completion、两侧scope/source relation、offset一次性写入和双配置gate记录 |
| `tasks/archive/residual-source-modularity.md` | 已完成Q25的reference/model、numeric/bulk、compiler/output/package与frontend bridge共11个聚合实现模块化和双配置gate记录 |
| `tasks/archive/remaining-source-modularity.md` | 已完成Q24的group/candidate、target LLVM、numeric conformance、frontend program与compiler driver模块化和双配置gate记录 |
| `tasks/archive/source-organization-refactor.md` | 已完成Q23的instruction、tile-region到instruction、target numeric源码模块化和build/test组织gate记录 |
| `tasks/archive/third-party-dependency-root-consistency.md` | 已完成SystemC canonical third-party root、existing cache切换和双配置重放记录 |
| `tasks/archive/target-model-completion-audit.md` | 已完成Q22各分项字段、legality、runtime evidence、late-rank原子性和双配置全量证据复核记录 |
| `tasks/archive/target-model-source-verticals.md` | 已完成Q22.V的same-lowering product、typed source/model invocation、formal/exact-admitted bulk dispatch和五个固定source vertical实施记录 |
| `tasks/archive/target-bulk-qualification.md` | 已完成Q22.B的受管oneDNN、target-owned adapter、三阶段资格producer和runtime exact-match verification实施记录 |
| `tasks/archive/target-llvm-module-bundle.md` | 已完成Q22.L的all-rank LLVM module生命周期、typed readback和single-lowering device-link接入记录 |
| `tasks/archive/target-numeric-foundation.md` | 已完成Q22.N的typed numeric schema、受管formal依赖、13-format codec和formal kernel实施记录 |
| `tasks/archive/target-call-functional-frontend.md` | 已完成Q22.H的owner-safe host JIT、typed target-call registry/decoder和atomic sink实施记录 |
| `tasks/archive/target-model-readiness.md` | Q21 reduce、numeric/SystemC依赖与host-CRT seam implementation-readiness实证 |
| `tasks/archive/01-design-docs-gap-review.md` | 历史设计缺口审计 |
| `tasks/archive/02-source-organization-recovery.md` | 历史源码组织恢复记录 |
| `tasks/archive/03-dependency-layering-recovery.md` | 历史依赖分层恢复记录 |
| `tasks/archive/04-p0-p6-design-conformance-audit.md` | 历史 conformance audit |
| `tasks/archive/05-p0-p6-recovery-status.md` | 历史 recovery status |
| `tasks/archive/06-r2-recovery.md` | 历史 recovery 记录 |
| `tasks/archive/07-candidate-selection-task-design.md` | 已收口的 candidate-selection 任务记录 |
| `tasks/archive/08-committed-candidate-materialization-task-design.md` | 已收口的selected candidate materialization任务记录 |
| `tasks/archive/09-system-design-implementation-review.md` | 2026-07-10 系统设计与实现审计；只作风险和整改依据，不是架构合同 |
| `tasks/archive/10-target-crt-closure-plan.md` | 已完成并被当前路线替代的 CRT closure 实施记录 |
| `tasks/archive/11-target-crt-conformance-plan.md` | 已完成并被当前路线替代的 CRT conformance 实施记录 |
| `tasks/archive/target-command-legality-closure.md` | 已完成Q0.L的typed target format legality、map/reduce lowering和fresh source replay实施记录 |
| `tasks/archive/12-architecture-evidence-reset.md` | 2026-07-12架构事实重基线审计；只作证据和整改依据 |
| `tasks/archive/single-card-vertical-slice.md` | 已完成的单卡纵向切片实施计划；只保留历史checkpoint和验证记录 |
| `tasks/archive/2026-07-10-long-horizon-plans/` | 已被重基线取代的7份生成式长周期计划；non-normative |
