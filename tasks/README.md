# Wafer Compiler Task Documents

本文只做 `tasks/` 文档导航，不声明新的架构合同，也不作为任务状态的主要事实源。
当前设计边界以编号设计文档为准；执行状态和下一步以 `tasks/progress.md` 任务队列为准。

## 当前设计文档

编号是稳定的文档导航和contract-owner标识，按compiler pipeline语义大致分组，不按创建日期排列。
它不表示严格的transform拓扑、实施优先级或任务依赖；一个owner文档可以覆盖pipeline中的多个位置。
实际执行顺序只读取`tasks/progress.md`中的状态和直接前置。

| 编号 | 文档 | 范围 |
| --- | --- | --- |
| 01 | `tasks/01-architecture.md` | verified program 到 typed distributed executable、atomic target artifact set 和 RuntimeSession 的总体边界 |
| 02 | `tasks/02-frontend-stablehlo-program.md` | StableHLO program、symbolic bounds、typed IO/parameter/persistent state |
| 03 | `tasks/03-shardy-spmd.md` | Shardy/SDY/MPMD distributed program、parallel coordinates、rank identity/class prerequisites |
| 04 | `tasks/04-topology-execution-mesh.md` | target environment/DDR arenas、topology/mesh、唯一 pinned/relocatable projection owner |
| 05 | `tasks/05-local-compute-normalization.md` | component/rank-local structured compute、state use-def、equal/segmented tensor collective handoff |
| 06 | `tasks/06-group.md` | logical group、candidate-only schedule 和 whole-variant atomic commit boundary |
| 07 | `tasks/07-tile-region.md` | complete traversal 内 candidate tile-local execution scope |
| 08 | `tasks/08-layout-materialization.md` | layout proposal、global accepted assignment 和 storage materialization |
| 09 | `tasks/09-spm-memory-planning.md` | whole-rank-entry SPM lifetime/event planning 和 accepted offsets |
| 10 | `tasks/10-compute-movement.md` | target-abstract compute/movement、resource effects 和 issue/token/fence/wait |
| 11 | `tasks/11-instruction-ir.md` | complete static rank instruction program、geometry/range/narrowing legality |
| 12 | `tasks/12-ddr-memory-planning.md` | typed arena/placement 下 whole-entry IO/weight/state/workspace DDR planning 和 accepted offsets |
| 13 | `tasks/13-communication.md` | logical/segmented collective materialization、post-memory physical transport acceptance 和 completion boundary |
| 14 | `tasks/14-target-llvm-golden-packet.md` | structure-preserving target LLVM、Slot-based Kernel ABI、双 fingerprint 和 atomic TargetArtifactSet |
| 15 | `tasks/15-launch-runtime-package.md` | Protobuf PackageManifest、typed metadata/runtime bootstrap、shared service registry、RuntimeSession、state migration 和 completion DAG |
| 16 | `tasks/16-verification-plan.md` | atomic commit、复杂大模型、target/package/runtime 和 board completion gates |

### Pipeline Owner 索引

下表只帮助定位边界owner，不复制设计合同，也不把跨阶段owner强行线性化：

| Pipeline boundary | Owner文档 |
| --- | --- |
| verified frontend program、typed IO/parameter/state | 02 |
| pre-SPMD target environment、topology和execution mesh | 04 |
| Shardy/SPMD/MPMD distributed program和rank identity | 03 |
| component/rank-local compute normalization | 05 |
| candidate group、tile traversal和layout assignment | 06、07、08 |
| target-abstract compute/movement和instruction legality | 10、11 |
| accepted SPM/DDR allocation、lifetime和offset | 09、12 |
| physical transport、projection和completion relation | 13；projection identity同时由04约束 |
| whole-variant candidate commit和typed executable | 06；资源/lifetime边界由09、12、13共同约束 |
| target LLVM、KAD、ELF和atomic TargetArtifactSet | 14 |
| PackageManifest、launch、RuntimeSession和state migration | 15 |
| 横跨上述边界的completion evidence | 16 |

## 实施计划导航

当前实施集由一份路线图和六份子计划组成，共七份文件：

- `tasks/plans/implementation-roadmap.md`
- `tasks/plans/target-artifact-set.md`
- `tasks/plans/typed-program-distributed-identity.md`
- `tasks/plans/whole-variant-executable.md`
- `tasks/plans/target-correctness-foundation.md`
- `tasks/plans/package-runtime.md`
- `tasks/plans/real-model-board-gates.md`

路线图编排依赖和review checkpoint，子计划只拆解施工文件、测试和提交。它们都不是编号设计文档，
也不能覆盖`tasks/01-16`的IR/ABI合同；动态执行状态仍只看`tasks/progress.md`。

## 归档文档

`tasks/archive/` 只保存历史 gap review、recovery、audit 和已收口的任务级记录。归档文档可以作为
实现背景或复盘材料，但不作为当前主线架构合同；如果归档内容和 numbered docs 冲突，以当前 numbered
docs、`tasks/progress.md` 和本轮已收敛设计结论为准。

| 文档 | 原性质 |
| --- | --- |
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
