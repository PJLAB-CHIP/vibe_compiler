# 历史记录索引

本目录保存当时的设计、执行和验收证据，不参与当前调度。归档不等于完成：未通过、board-ready、取消和撤回的边界均保留原结论。
当前状态见[progress](../progress.md)，稳定合同见[编号设计索引](../README.md)。
历史正文中的symbol、命令、版本和“下一步”按当时提交解释；实际超链接指向对应记录。

| 文档 | 范围 |
| --- | --- |
| [board-correctness-qualification.md](board-correctness-qualification.md) | 统一板端覆盖、模型修复与有限产品计时的实施及验收记录 |
| [completed-task-index.md](completed-task-index.md) | 已完成任务的历史边界与证据入口索引；不参与current调度 |
| [document-contract-recovery.md](document-contract-recovery.md) | 纠正current plan过度压缩、恢复Q52第12--20项及Q53规范合同并核对Q57/Q61的定向恢复记录 |
| [document-authority-consolidation.md](document-authority-consolidation.md) | 已完成的current文档事实源、计划归档、README、编号设计、硬件事实和memory收敛记录 |
| [compiler-entry-productization.md](compiler-entry-productization.md) | 已完成Q59/Q60的compiler transaction与frontend产品入口实施记录；稳定合同只读01、02、15、18--20 |
| [compiler-terminology-and-naming.md](compiler-terminology-and-naming.md) | 已完成的源码、IR和component命名整改记录；稳定命名规则只读18、19和AGENTS |
| [interface-version-consolidation.md](interface-version-consolidation.md) | 已完成Q55的current-only接口版本收敛记录；稳定接口规则只读20及各格式owner |
| [executable-package-and-resident-runtime.md](executable-package-and-resident-runtime.md) | 已完成Q56 package/runtime闭合及Q57早期设想；Q57 current计划只读`tasks/plans/resident-static-execution.md` |
| [program-data-and-whole-program-scale.md](program-data-and-whole-program-scale.md) | 已完成Q58 program-data ownership及Q61早期设想；Q61 current计划只读`tasks/plans/whole-program-scale-readiness.md` |
| [physical-dataflow-synthesis-q52-plan-history.md](physical-dataflow-synthesis-q52-plan-history.md) | Q52第1--11项、旧路径删除账本和早期第12--20项展开；后续Q52/Q53验收见`physical-dataflow-host-readiness.md`，当前合同由06号设计拥有 |
| [mlir-engineering-reference-snapshot.md](mlir-engineering-reference-snapshot.md) | 19号MLIR工程合同收敛前的代表实现对照、pinned commit和完整调研链接；不作为current规则 |
| [general-development-reference-snapshot.md](general-development-reference-snapshot.md) | `memory/general_dev.md`收敛前的完整开发经验和历史架构说明；current方法只读memory，设计只读编号文档 |
| [component-dependency-closure.md](component-dependency-closure.md) | 已完成的pass owner、component依赖、tool link、fixture/cache与组织门禁闭合；稳定规则只读18、19 |
| [source-layout-consolidation.md](source-layout-consolidation.md) | 已完成的源码目录、component target、test mirror、tools/runtime路径与current文档收敛计划；稳定规则只读18、19 |
| [target-numeric-contract-reconstruction.md](target-numeric-contract-reconstruction.md) | 已完成Q62的Target数值合同拆分记录；稳定语义只读01、11、14、16–18 |
| [ncc-synchronization-contract-layering.md](ncc-synchronization-contract-layering.md) | 已完成Q63的NCC completion合同分层记录；稳定语义只读11、13、17、19 |
| [source-registration-truth-closure.md](source-registration-truth-closure.md) | 已完成Q64的repo-wide source/test registration mirror与旧一轮owner迁移记录；其中目录分类已由current 18号设计替代 |
| [physical-dataflow-synthesis-working-history.md](physical-dataflow-synthesis-working-history.md) | 截至2026-08-25的Q49–Q51详细施工、Q52重基线审计和原Q53板端设想；后续Q52/Q53计划已归档，禁止从本文件恢复旧顺序或shared materializer |
| [mlir-engineering-remediation.md](mlir-engineering-remediation.md) | 已完成Q54的审计、checkpoint和验证记录；verifier cleanup已由Q52闭合，稳定规则由19和AGENTS拥有 |
| [whole-card-tile-dataflow-synthesis.md](whole-card-tile-dataflow-synthesis.md) | 2026-08-11至08-13的旧Q49/Q50完整施工计划；旧任务拆法、shortlist和owner合同不再有效 |
| [whole-rank-tile-dataflow-synthesis.md](whole-rank-tile-dataflow-synthesis.md) | 2026-08-08的structured-DAG/card历史施工计划；已由current physical-dataflow计划替代，旧public policy与bounded candidate set不再有效 |
| [whole-variant-search-throughput.md](whole-variant-search-throughput.md) | 已完成Q32.C的passing-ordinal early stop、bounded persistent candidate executor、accepted-module owner import、exact attempt-plan selective parse和fully-gated Pareto前置late ABI/LLVM，并记录优化后Release单次实测 |
| [k-sharded-gemm-board-vertical.md](k-sharded-gemm-board-vertical.md) | 已完成Q35 full-4096 f16 K-sharded GEMM的production tiling/SPM/Direct-DTE package、纯tiling隔离及16-rank重复板端raw-exact记录 |
| [runtime-board.md](runtime-board.md) | 已完成Q6.B的typed TX board provider、kernel/model多tile launch、cluster Direct DTE、failure lifecycle及真实板端重复exact记录 |
| [physical-dataflow-synthesis-completion-audit.md](physical-dataflow-synthesis-completion-audit.md) | 已完成Q32的七checkpoint证据映射、双配置全量门禁、fixed/held-out 7B scale重放、单一production owner及剩余边界审计 |
| [physical-dataflow-synthesis.md](physical-dataflow-synthesis.md) | 已完成Q32 MLIR-native bounded physical-dataflow synthesis的施工checkpoint、hard-cap与integrated completion checklist |
| [physical-dataflow-production-cutover.md](physical-dataflow-production-cutover.md) | 已完成Q32.G默认production winner cutover、旧decision surface删除、all-rank双键correspondence及source/bulk数值回归记录 |
| [bounded-joint-physical-dataflow-selection.md](bounded-joint-physical-dataflow-selection.md) | 已完成Q32.S的actual-clone有界联合candidate set、reserved baseline、validated card exact cost、target static policy及逐producer whole-winner记录 |
| [physical-mechanism-choice-closure.md](physical-mechanism-choice-closure.md) | 已完成Q32.M的actual-clone recompute/LICM/integer algebra、partial fanout、spill/resident/ready-order、communication alternatives及重复layout/resource/collective/instruction合同删除记录 |
| [physical-relation-realization.md](physical-relation-realization.md) | 已完成Q32.R的rich IndexRelation、physical encoding interface、TransferRealizability、destination-style load、relation-backed resident boundary和fresh 7B TP16数值纵向记录 |
| [typed-target-capability-vertical.md](typed-target-capability-vertical.md) | 已完成Q32.V的mapped DMA双端offset、physical-footprint fill、source/Tile/Instr oriented GEMM及formal/SystemC fresh纵向记录 |
| [mlir-native-implementation-relation-foundation.md](mlir-native-implementation-relation-foundation.md) | 已完成Q32.I的fresh baseline、source implementation external model、真实reciprocal/division actual-clone纵向、MLIR-backed IndexRelation foundation和custom interface盘点/首轮删除记录 |
| [static-memory-packing.md](static-memory-packing.md) | Q34历史MiniMalloc施工记录；其中曾有的fallback合同已退役，current只使用MiniMalloc并原样传播`ResourceExhausted` |
| [llama-block-numeric-characterization.md](llama-block-numeric-characterization.md) | 已完成Q31的ProgramTensor逐rank abs/ULP统计、非verification多seed 7B重放及source/model comparator gate收紧记录 |
| [llama-block-production-performance.md](llama-block-production-performance.md) | 已完成Q30的static movement/physical codec host性能收口、package等价性和完整7B双replay记录 |
| [llama-7b-block-vertical.md](llama-7b-block-vertical.md) | 已完成Q28的标准Llama-2 7B单block TP16 source/package、repo-owned SystemC managed-reference和完整PyTorch eager output differential记录 |
| [tile-dataflow-scheduling.md](tile-dataflow-scheduling.md) | 已完成Q29历史structured tensor program直达bounded task/dataflow scheduling、当时的跨region SPM合同、card结果写入、旧group executable surface退役及7B TP16 compile-only验证记录；current region语义只看06/07和Q52 current plan |
| [reference-executor-retirement.md](reference-executor-retirement.md) | 已完成Q27的accepted-IR第二套解释器、oracle分支和旧CLI退役，以及CPU-expected到target CModel纵向gate收敛记录 |
| [memory-lifetime-analysis.md](memory-lifetime-analysis.md) | 已完成Q26的共享structured lifetime/packing core、DDR issue-to-fence completion、两侧scope/source relation、offset一次性写入和双配置gate记录 |
| [residual-source-modularity.md](residual-source-modularity.md) | 已完成Q25的reference/model、numeric/bulk、compiler/output/package与frontend bridge共11个聚合实现模块化和双配置gate记录 |
| [remaining-source-modularity.md](remaining-source-modularity.md) | 已完成Q24的group/candidate、target LLVM、numeric conformance、frontend program与compiler driver模块化和双配置gate记录 |
| [source-organization-refactor.md](source-organization-refactor.md) | 已完成Q23的instruction、tile-region到instruction、target numeric源码模块化和build/test组织gate记录 |
| [third-party-dependency-root-consistency.md](third-party-dependency-root-consistency.md) | 已完成SystemC canonical third-party root、existing cache切换和双配置重放记录 |
| [target-model-completion-audit.md](target-model-completion-audit.md) | 已完成Q22各分项字段、legality、runtime evidence、late-rank原子性和双配置全量证据复核记录 |
| [target-model-source-verticals.md](target-model-source-verticals.md) | 已完成Q22.V的same-lowering product、typed source/model invocation、formal/exact-admitted bulk dispatch和五个固定source vertical实施记录 |
| [target-bulk-qualification.md](target-bulk-qualification.md) | 已完成Q22.B的受管oneDNN、target-owned adapter、三阶段资格producer和runtime exact-match verification实施记录 |
| [target-llvm-module-bundle.md](target-llvm-module-bundle.md) | 已完成Q22.L的all-rank LLVM module生命周期、typed readback和single-lowering device-link接入记录 |
| [target-numeric-foundation.md](target-numeric-foundation.md) | 已完成Q22.N的typed numeric schema、受管formal依赖、13-format codec和formal kernel实施记录 |
| [target-call-functional-frontend.md](target-call-functional-frontend.md) | 已完成Q22.H的owner-safe host JIT、typed target-call registry/decoder和atomic sink实施记录 |
| [target-model-readiness.md](target-model-readiness.md) | Q21 reduce、numeric/SystemC依赖与host-CRT seam implementation-readiness实证 |
| [01-design-docs-gap-review.md](01-design-docs-gap-review.md) | 历史设计缺口审计 |
| [02-source-organization-recovery.md](02-source-organization-recovery.md) | 历史源码组织恢复记录 |
| [03-dependency-layering-recovery.md](03-dependency-layering-recovery.md) | 历史依赖分层恢复记录 |
| [04-p0-p6-design-conformance-audit.md](04-p0-p6-design-conformance-audit.md) | 历史 conformance audit |
| [05-p0-p6-recovery-status.md](05-p0-p6-recovery-status.md) | 历史 recovery status |
| [06-r2-recovery.md](06-r2-recovery.md) | 历史 recovery 记录 |
| [07-candidate-selection-task-design.md](07-candidate-selection-task-design.md) | 已收口的 candidate-selection 任务记录 |
| [08-committed-candidate-materialization-task-design.md](08-committed-candidate-materialization-task-design.md) | 已收口的selected candidate materialization任务记录 |
| [09-system-design-implementation-review.md](09-system-design-implementation-review.md) | 2026-07-10 系统设计与实现审计；只作风险和整改依据，不是架构合同 |
| [10-target-crt-closure-plan.md](10-target-crt-closure-plan.md) | 已完成并被当前路线替代的 CRT closure 实施记录 |
| [11-target-crt-conformance-plan.md](11-target-crt-conformance-plan.md) | 已完成并被当前路线替代的 CRT conformance 实施记录 |
| [target-command-legality-closure.md](target-command-legality-closure.md) | 已完成Q0.L的typed target format legality、map/reduce lowering和fresh source replay实施记录 |
| [12-architecture-evidence-reset.md](12-architecture-evidence-reset.md) | 2026-07-12架构事实重基线审计；只作证据和整改依据 |
| [single-card-vertical-slice.md](single-card-vertical-slice.md) | 已完成的单卡纵向切片实施计划；只保留历史checkpoint和验证记录 |
| [2026-07-10-long-horizon-plans/](2026-07-10-long-horizon-plans/) | 已被重基线取代的7份生成式长周期计划；non-normative |
| [access-reuse.md](access-reuse.md) | 访问复用（AccessReuse）统一方案 |
| [board-performance-optimization.md](board-performance-optimization.md) | 模型板端性能优化 |
| [board-profiler.md](board-profiler.md) | 16-Tile Production Artifact Profiler 实施计划 |
| [board-testing-scheduling-history.md](board-testing-scheduling-history.md) | 板测历史接续记录 |
| [board-workload-matrix.md](board-workload-matrix.md) | 板测矩阵、历史调优与2026-09-25计时复验（归档） |
| [collective-hardware-characterization.md](collective-hardware-characterization.md) | Collective Hardware Characterization 实施计划 |
| [compiler-search-scalability.md](compiler-search-scalability.md) | Compiler Search Scalability |
| [composed-choice-search-and-dte-overlap.md](composed-choice-search-and-dte-overlap.md) | Composed Choice Search 与 Direct-DTE/Compute Overlap 实施计划 |
| [layout-movement-elimination.md](layout-movement-elimination.md) | Layout Movement Elimination 实施计划 |
| [loop-body-redundant-transfer-normalization.md](loop-body-redundant-transfer-normalization.md) | 循环内冗余物理传输规范化 |
| [mesh-communication-task-reconciliation.md](mesh-communication-task-reconciliation.md) | 通信实现任务的调度收拢 |
| [multi-engine-software-pipelining.md](multi-engine-software-pipelining.md) | Multi-Engine Software Pipelining 实施计划 |
| [noc-resident-tile-dataflow.md](noc-resident-tile-dataflow.md) | NoC-Resident Tile Dataflow 实施计划 |
| [numeric-algebraic-extension.md](numeric-algebraic-extension.md) | Numeric Algebraic Extension 实施计划 |
| [physical-dataflow-host-readiness.md](physical-dataflow-host-readiness.md) | Physical Dataflow Current-IR实施计划 |
| [physical-dataflow-test-seam-vertical.md](physical-dataflow-test-seam-vertical.md) | Physical-Dataflow Production-Shaped Candidate Seam 收口记录 |
| [physical-search-organization.md](physical-search-organization.md) | 物理搜索效率与 standard/deep 组织实施方案 |
| [production-optimization-board-qualification.md](production-optimization-board-qualification.md) | Production Compiler Optimization Board Qualification 实施计划 |
| [pytorch-source-board-verticals.md](pytorch-source-board-verticals.md) | PyTorch Source Board Verticals 实施计划 |
| [runtime-launch-contract-convergence.md](runtime-launch-contract-convergence.md) | Runtime Launch Contract 收口计划 |
| [spatial-admission-and-typed-outcomes.md](spatial-admission-and-typed-outcomes.md) | Spatial admission 与 relation 协调收敛 |
| [systemc-functional-event-model.md](systemc-functional-event-model.md) | SystemC Functional-Event Model 实施计划 |
| [target-abi-retirement.md](target-abi-retirement.md) | Target ABI 收口实施计划 |
| [tensor-subset-materialization-reverted.md](tensor-subset-materialization-reverted.md) | Tensor 子集物化整改撤回记录 |
| [test-gate-scope-reduction.md](test-gate-scope-reduction.md) | 测试减负 |
| [tile-loop-pipelining.md](tile-loop-pipelining.md) | Tile职责分拆与通用循环流水实施方案 |
| [topology-aware-collective-lowering.md](topology-aware-collective-lowering.md) | Topology-Aware Collective Lowering 实施计划 |
| [vibe-compiler-collaboration-review.md](vibe-compiler-collaboration-review.md) | Vibe Compiler 共同开发汇报与硬件行为导读实施记录 |
| [target-cpu-codegen-experiments.md](target-cpu-codegen-experiments.md) | 已撤回的四组CPU代码生成实验与验收边界 |
| [documentation-maintenance.md](documentation-maintenance.md) | 全仓文档整理、合同迁移与验证记录 |
