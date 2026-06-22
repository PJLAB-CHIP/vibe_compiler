# Wafer Compiler Task Documents

本文只做 `tasks/` 文档导航，不声明新的架构合同。当前设计边界以编号文档和
`tasks/01-architecture.md` 第 8 节为准；执行状态以 `tasks/progress.md` 为准。

## 当前设计文档

编号按 compiler pipeline 的语义顺序排列，不按创建日期排列。

| 编号 | 文档 | 范围 |
| --- | --- | --- |
| 01 | `tasks/01-architecture.md` | 总体 compiler pipeline、IR 分层和子设计边界 |
| 02 | `tasks/02-frontend-stablehlo-program.md` | frontend program、StableHLO import/export、program directory |
| 03 | `tasks/03-shardy-spmd.md` | Shardy propagation、SPMD partition、post-SPMD tensor collective handoff |
| 04 | `tasks/04-topology-execution-mesh-boundary-shards.md` | target topology、execution mesh、boundary shards、encoded tile endpoint |
| 05 | `tasks/05-local-compute-normalization.md` | post-SPMD local compute normalization |
| 06 | `tasks/06-group.md` | logical group、scheduled group、candidate planning boundary |
| 07 | `tasks/07-tile-region.md` | memref-backed tile-local execution boundary |
| 08 | `tasks/08-layout-materialization.md` | physical layout planning and materialization |
| 09 | `tasks/09-spm-memory-planning.md` | SPM memory planning and accepted offset facts |
| 10 | `tasks/10-compute-movement.md` | target-abstract compute / movement IR |
| 11 | `tasks/11-instruction-ir.md` | instruction-level Wafer IR over Wafer-tagged memrefs |
| 12 | `tasks/12-ddr-memory-planning.md` | DDR memory planning and accepted offset facts |
| 13 | `tasks/13-communication.md` | tile-local communication IR and Direct DTE boundary |
| 14 | `tasks/14-abi-golden-packet.md` | C ABI / LLVM / golden packet lowering boundary |
| 15 | `tasks/15-launch-runtime-package.md` | launch op、runtime package、host runtime adapter |
| 16 | `tasks/16-verification-plan.md` | staged verification plan and completion gates |

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
