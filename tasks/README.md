# Wafer Compiler Task Documents

本文只做 `tasks/` 文档导航，不声明新的架构合同，也不作为任务状态的主要事实源。
当前设计边界以编号设计文档为准；执行状态和下一步以 `tasks/progress.md` 任务队列为准。

## 当前设计文档

编号是稳定的文档导航和contract-owner标识，按compiler pipeline语义大致分组，不按创建日期排列。
它不表示严格的transform拓扑、实施优先级或任务依赖；一个owner文档可以覆盖pipeline中的多个位置。
实际执行顺序只读取`tasks/progress.md`中的状态和“必须满足的前置”；该列可为审计重复列出关键传递gate，不等同最小DAG边集。

| 编号 | 文档 | 范围 |
| --- | --- | --- |
| 01 | `tasks/01-architecture.md` | 当前真实pipeline、近期per-rank static executable bundle和长期扩展边界 |
| 02 | `tasks/02-frontend-stablehlo-program.md` | 当前StableHLO program directory与frontend验证；typed state是后续扩展 |
| 03 | `tasks/03-shardy-spmd.md` | 当前Shardy/XLA SPMD artifact、显式rank identity；MPMD/rank class延后 |
| 04 | `tasks/04-topology-execution-mesh.md` | 当前topology/execution mesh；Q0.L typed target-profile仅随ExecutionConfig透传，不进入mesh IR |
| 05 | `tasks/05-local-compute-normalization.md` | rank-local structured compute和tensor collective handoff |
| 06 | `tasks/06-group.md` | logical group、candidate proposal、完整traversal和bundle commit边界 |
| 07 | `tasks/07-tile-region.md` | complete traversal 内 candidate tile-local execution scope |
| 08 | `tasks/08-layout-materialization.md` | layout proposal、accepted assignment和storage materialization |
| 09 | `tasks/09-spm-memory-planning.md` | SPM lifetime/completion planning和accepted offsets |
| 10 | `tasks/10-compute-movement.md` | target-abstract compute/movement、resource effects 和 issue/token/fence/wait |
| 11 | `tasks/11-instruction-ir.md` | complete static rank instruction program、geometry/range/narrowing legality |
| 12 | `tasks/12-ddr-memory-planning.md` | 当前DDR demand/accepted offsets；multi-arena/state/streaming延后 |
| 13 | `tasks/13-communication.md` | 当前collective到Direct DTE和completion边界；segmented/multi-card延后 |
| 14 | `tasks/14-target-llvm-golden-packet.md` | shared logical/target-format registry、structure-preserving target LLVM、CRT ABI和atomic staged target module |
| 15 | `tasks/15-launch-runtime-package.md` | typed C++ manifest、canonical JSON、no-card RuntimeSession和board adapter边界 |
| 16 | `tasks/16-verification-plan.md` | target correctness、1/16-rank bundle、CPU oracle、target-model、no-card和board分层gate |
| 17 | `tasks/17-target-execution-model.md` | multi-dtype numeric、oneDNN bulk、owner-backed target LLVM bundle、repo-owned target-call/SystemC untimed CModel、optional CRT/packet provenance、Q22.C板端numeric correlation、Q22.E exact-module和deferred Q22.P timing边界 |
| 18 | `tasks/18-source-organization.md` | 跨pipeline的源码ownership、translation unit、内部接口、构建依赖和测试镜像组织合同；不改变IR/artifact语义 |

### Pipeline Owner 索引

下表只帮助定位边界owner，不复制设计合同，也不把跨阶段owner强行线性化：

| Pipeline boundary | Owner文档 |
| --- | --- |
| verified frontend program和当前program directory | 02 |
| pre-SPMD topology和execution mesh | 04 |
| Shardy/SPMD output和显式rank identity | 03 |
| component/rank-local compute normalization | 05 |
| candidate group、tile traversal和layout assignment | 06、07、08 |
| target-abstract compute/movement和instruction legality | 10、11 |
| accepted SPM/DDR allocation、lifetime和offset | 09、12；shared lifetime analysis的源码ownership和测试镜像由18约束 |
| Direct DTE logical schedule/completion与post-memory transport activation | 13；target/package/verification consumer由14、15、16约束 |
| per-rank candidate commit和typed executable bundle | 06；资源/lifetime边界由09、12、13共同约束 |
| target LLVM、CRT/device link和staged target module | 14 |
| typed manifest、launch和RuntimeSession | 15 |
| 横跨上述边界的completion evidence | 16 |
| target execution model、multi-dtype numeric/bulk、target LLVM bundle、SystemC/CModel capability、板端numeric correlation和deferred timing | 17；target/runtime/verification consumer由14、15、16约束 |
| 跨上述边界的源码与构建模块化 | 18；各IR/artifact语义仍由01-17拥有 |

## 实施计划导航

当前无active实施计划。Q27 reference executor退役已归档为`tasks/archive/reference-executor-retirement.md`，
Q26 memory lifetime analysis已归档为`tasks/archive/memory-lifetime-analysis.md`，
Q25剩余聚合边界模块化已归档为`tasks/archive/residual-source-modularity.md`，Q24剩余热点模块化已归档为
`tasks/archive/remaining-source-modularity.md`，Q23首轮源码组织重构已归档为
`tasks/archive/source-organization-refactor.md`；
Q13.W依赖root一致性计划已归档为`tasks/archive/third-party-dependency-root-consistency.md`。Q22.S、Q22.V与最终
完成性审计计划已分别归档为`tasks/archive/systemc-functional-event-model.md`、
`tasks/archive/target-model-source-verticals.md`和`tasks/archive/target-model-completion-audit.md`。
Q22.B/Q22.L/Q22.N/Q0.L/Q22.H计划分别归档为`tasks/archive/target-bulk-qualification.md`、`tasks/archive/target-llvm-module-bundle.md`、
`tasks/archive/target-numeric-foundation.md`、`tasks/archive/target-command-legality-closure.md`和
`tasks/archive/target-call-functional-frontend.md`，Q22.R readiness归档为`tasks/archive/target-model-readiness.md`。
Q22.N+Q22.H已经在Q22.S SystemC event model汇合，Q22.B与Q22.S再由Q22.V source vertical闭合并完成Q22汇总；
后续代码任务开工前都需独立计划。动态执行状态只看`tasks/progress.md`。

## 归档文档

`tasks/archive/` 只保存历史 gap review、recovery、audit 和已收口的任务级记录。归档文档可以作为
实现背景或复盘材料，但不作为当前主线架构合同；如果归档内容和 numbered docs 冲突，以当前 numbered
docs、`tasks/progress.md` 和本轮已收敛设计结论为准。

| 文档 | 原性质 |
| --- | --- |
| `tasks/archive/reference-executor-retirement.md` | 已完成Q27的accepted-IR第二套解释器、oracle分支和旧CLI退役，以及CPU-expected到target CModel纵向gate收敛记录 |
| `tasks/archive/memory-lifetime-analysis.md` | 已完成Q26的共享structured lifetime/packing core、DDR issue-to-fence completion、两侧scope/provenance/原子commit和双配置gate记录 |
| `tasks/archive/residual-source-modularity.md` | 已完成Q25的reference/model、numeric/bulk、compiler/artifact/package与frontend bridge共11个聚合实现模块化和双配置gate记录 |
| `tasks/archive/remaining-source-modularity.md` | 已完成Q24的group/candidate、target LLVM、numeric conformance、frontend program与compiler driver模块化和双配置gate记录 |
| `tasks/archive/source-organization-refactor.md` | 已完成Q23的instruction、tile-region到instruction、target numeric源码模块化和build/test组织gate记录 |
| `tasks/archive/third-party-dependency-root-consistency.md` | 已完成SystemC canonical third-party root、existing cache切换和双配置重放记录 |
| `tasks/archive/target-model-completion-audit.md` | 已完成Q22各分项字段、legality、runtime evidence、late-rank原子性和双配置全量证据复核记录 |
| `tasks/archive/target-model-source-verticals.md` | 已完成Q22.V的same-lowering product、typed source/model invocation、formal/exact-admitted bulk dispatch和五个固定source vertical实施记录 |
| `tasks/archive/target-bulk-qualification.md` | 已完成Q22.B的受管oneDNN、target-owned adapter、三阶段资格producer和runtime exact-match admission实施记录 |
| `tasks/archive/target-llvm-module-bundle.md` | 已完成Q22.L的owner-backed all-rank LLVM module、typed readback和single-lowering device-link接入记录 |
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
| `tasks/archive/08-committed-candidate-materialization-task-design.md` | 已收口的 committed-materialization 任务记录 |
| `tasks/archive/09-system-design-implementation-review.md` | 2026-07-10 系统设计与实现审计；只作风险和整改依据，不是架构合同 |
| `tasks/archive/10-target-crt-closure-plan.md` | 已完成并被当前路线替代的 CRT closure 实施记录 |
| `tasks/archive/11-target-crt-conformance-plan.md` | 已完成并被当前路线替代的 CRT conformance 实施记录 |
| `tasks/archive/target-command-legality-closure.md` | 已完成Q0.L的typed target profile、format legality、map/reduce lowering和fresh source replay实施记录 |
| `tasks/archive/12-architecture-evidence-reset.md` | 2026-07-12架构事实重基线审计；只作证据和整改依据 |
| `tasks/archive/single-card-vertical-slice.md` | 已完成的单卡纵向切片实施计划；只保留历史checkpoint和验证记录 |
| `tasks/archive/2026-07-10-long-horizon-plans/` | 已被重基线取代的7份生成式长周期计划；non-normative |
