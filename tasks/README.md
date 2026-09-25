# Wafer Compiler Task Documents

本文只做 `tasks/` 文档导航，不声明新的架构合同，也不作为任务状态的主要事实源。
当前设计边界以编号设计文档为准；执行状态和下一步以 [progress.md](progress.md) 任务队列为准。

## 当前设计文档

编号是稳定的文档导航和contract-owner标识，按compiler pipeline语义大致分组，不按创建日期排列。
它不表示严格的transform拓扑、实施优先级或任务依赖；一个owner文档可以覆盖pipeline中的多个位置。
实际执行顺序与直接前置只读取[progress.md](progress.md)。编号文档链接的专题章节属于同一设计owner。

| 编号 | 文档 | 范围 |
| --- | --- | --- |
| 01 | [01-architecture.md](01-architecture.md) | compiler stack主架构：production pipeline、library/CLI transaction、IR/module/package graph、跨层不变量、consumer分支和职责索引 |
| 02 | [02-frontend-stablehlo-program.md](02-frontend-stablehlo-program.md) | StableHLO program directory、产品adapter边界、ProgramDataSource/ProgramDataRange lifetime与frontend验证 |
| 03 | [03-shardy-spmd.md](03-shardy-spmd.md) | Shardy/XLA的card-level GSPMD output与`num_partitions`；不绑定片内Tile |
| 04 | [04-topology-execution-mesh.md](04-topology-execution-mesh.md) | logical card partition mesh与独立target card/Tile topology |
| 05 | [05-local-compute-normalization.md](05-local-compute-normalization.md) | card-partition-local structured compute normalization、single attention op、graph-level FA/FD算法与tensor collective boundary |
| 06 | [06-physical-dataflow-synthesis.md](06-physical-dataflow-synthesis.md) | card-local TensorProgram的spatial/region/temporal choice、actual TileModule/TileRegion、current-IR layout/movement/execution-structure/Instr与actual admission；预算限制planning工作而不预先截断合法域 |
| 07 | [07-tile-region.md](07-tile-region.md) | selected tile/dataflow structural IR物化以及structural、layout-resolved、physical TileRegion forms；SPM root不跨界，boundary不自动产生movement或join |
| 08 | [08-physical-realization.md](08-physical-realization.md) | physical encoding attr/type语义、valid domain、view、transfer realizability analysis、descriptor cover和selected physical realization |
| 09 | [09-spm-memory-planning.md](09-spm-memory-planning.md) | SPM lifetime/coexistence、fixed-capacity MiniMalloc legality、all-root coverage、validated placement/headroom和accepted offsets；candidate choice仍由06拥有 |
| 10 | [10-compute-movement.md](10-compute-movement.md) | selected Linalg/Tensor/SCF到typed target-abstract compute/movement的确定性lowering、current Tile execution structure、standard MLIR effects/interface reuse和issue/token/fence/wait |
| 11 | [11-instruction-ir.md](11-instruction-ir.md) | complete static Tile instruction program、current descriptor/geometry/range/narrowing legality及mapped/physical-fill/oriented typed extension |
| 12 | [12-ddr-memory-planning.md](12-ddr-memory-planning.md) | 当前DDR demand/accepted offsets；多DDR分区、state和streaming延后 |
| 13 | [13-communication.md](13-communication.md) | selected Tile edge到typed p2p/staging/token/wait IR、Direct DTE card-scoped verification和completion；multi-card延后 |
| 14 | [14-target-code-generation.md](14-target-code-generation.md) | `DeviceExecutable`的current target identity/format、accepted immutable data preparation、structure-preserving conversion、CRT ABI和atomic target-module writing |
| 15 | [15-launch-runtime-package.md](15-launch-runtime-package.md) | `DeviceExecutable -> ExecutablePackage`、ProgramTensor/TargetTensor、program-data.bin、TileEntryArgument、runtime memory plan、no-card和board adapter边界 |
| 16 | [16-verification-contract.md](16-verification-contract.md) | `TensorProgram -> TileModule/TileRegion/Instr -> DeviceExecutable -> ExecutablePackage`的target correctness、data/whole-program scale、CPU oracle、target-model、no-card和board分层gate |
| 17 | [17-target-execution-model.md](17-target-execution-model.md) | `DeviceExecutable`及其same-invocation target LLVM owner消费、multi-dtype `WaferTargetNumericBackend`、target-call/SystemC untimed CModel与板端numeric correlation边界 |
| 18 | [18-source-organization.md](18-source-organization.md) | 跨pipeline的源码ownership、translation unit、compiler library/tool/install、构建依赖、测试镜像与文档组织合同；不改变IR/output语义 |
| 19 | [19-mlir-engineering.md](19-mlir-engineering.md) | 跨IR层的ODS、standard interface、operation-scoped pass/analysis、rewrite/conversion和named nested pipeline工程合同；不重定义01–18语义 |
| 20 | [20-interface-evolution.md](20-interface-evolution.md) | 跨compiler/runtime/tool的内部接口演进、持久化格式与ABI版本owner、集中兼容检查和current-only表示合同 |

### Pipeline Owner 索引

下表只帮助定位边界owner，不复制设计合同，也不把跨阶段owner强行线性化：

| Pipeline boundary | Owner文档 |
| --- | --- |
| verified frontend program、产品adapter、program data ownership和当前program directory | 02 |
| pre-SPMD topology和execution mesh | 04 |
| Shardy/SPMD card-partition output | 03 |
| card-partition-local compute normalization、attention graph algorithm与collective boundary | 05；physical planning和selected decomposition分别由06、07、10消费，跨stage正确性证据由16约束 |
| card-local multi-Tile physical-dataflow planning、selected TileModule/TileRegion MPMD materialization和physical encoding/transfer | 06、07、08；source implementation interface由10提供，instruction legality由11提供，exact resource/transport gate由09、12、13提供 |
| policy-free physical-dataflow rewrites | 06、07、08；upstream structured utility由05提供，source/direct typed lowering合同由10提供，源码ownership由18约束 |
| source implementation interface、target-abstract compute/movement和instruction legality | 10、11；transfer realizability/descriptor cover只由08拥有 |
| accepted SPM/DDR allocation、lifetime和offset | 09、12；shared lifetime analysis的源码ownership和测试镜像由18约束 |
| Tile peer/collective materialization、Direct DTE completion、card-scoped acceptance与post-memory transport activation | 13；joint choice由06、target/package/verification consumer由14、15、16约束 |
| card-local multi-Tile时空调度、MPMD executable构造和`DeviceExecutable` | 06；资源/lifetime边界由09、12、13共同约束 |
| target LLVM、accepted immutable data preparation、CRT/device link和staged target module | 14 |
| target-ready data、typed manifest、RuntimeInvocationPlan和board launch | 15 |
| 横跨上述边界的completion evidence | 16 |
| target execution model、multi-dtype `WaferTargetNumericBackend`、same-invocation target module消费、SystemC/CModel capability、板端numeric correlation和deferred timing | 17；target module形成与writing合同由14拥有，target/runtime/verification consumer由14、15、16约束 |
| 跨上述边界的源码与构建模块化 | 18；各IR/output语义仍由01-17拥有 |
| 跨上述IR层的MLIR operation scope、pass/analysis manager、interface和rewrite工程合同 | 19；各层具体语义仍由01-18拥有 |
| 跨compiler/runtime/tool的格式与ABI版本边界 | 20；具体字段语义仍由02、11、14-17拥有 |

## 实施计划导航

本节只列仍由current或later队列消费的实施计划。任务状态、顺序和启动条件只读[progress.md](progress.md)；已完成计划
只从下方archive索引定位，不能作为current协议。

| 当前或later范围 | 实施计划 | 稳定设计owner |
| --- | --- | --- |
| Q57 resident static execution | [plans/resident-static-execution.md](plans/resident-static-execution.md) | 15–17 |
| Q61 whole-program scale readiness | [plans/whole-program-scale-readiness.md](plans/whole-program-scale-readiness.md) | 01–02、06、14–18 |
| Q48 semantic superoptimization | [plans/semantic-superoptimization.md](plans/semantic-superoptimization.md) | 05–08、10–11、16–18 |

独立Q53完整主机矩阵、搜索收益与三轮调优已取消；相关计划只保留历史证据，见下方归档索引。

## 归档文档

`tasks/archive/` 只保存历史 gap review、recovery、audit 和已收口的任务级记录。归档文档可以作为
实现背景或复盘材料，但不作为当前主线架构合同；如果归档内容和 numbered docs 冲突，以当前 numbered
docs、[progress.md](progress.md) 和本轮已收敛设计结论为准。

[archive/completed-task-index.md](archive/completed-task-index.md)集中保存从current progress移出的已完成任务边界和证据入口；它不是动态状态表，
未完成、`doing`、`queued`、`later`和`board-ready`任务仍只在[progress.md](progress.md)维护。

归档正文中的编号文档basename、章节号、line range和命令按当时提交快照解释，不保证仍是当前可解析路径；当前owner路径
只从上面的“当前设计文档”表读取。重命名current owner时不机械改写archive，以免篡改历史审计证据。

完整清单见[归档索引](archive/README.md)，已完成边界见[completed-task-index](archive/completed-task-index.md)。
