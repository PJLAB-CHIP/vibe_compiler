# Wafer Compiler Task Queue

本文件是当前任务状态、直接前置和完成门禁的唯一入口。稳定设计在编号文档，实施步骤在`tasks/plans/`，
已完成边界和证据在`tasks/archive/completed-task-index.md`。这里不保存施工日志、算法说明或测试数字。

## 状态规则

- current work item按artifact producer/consumer关系线性排列；同一item只出现一次，全局至多一个`doing`。
- `next`表示直接前置已满足；`queued`表示等待表中前序；`later`表示不在当前主线；板端工作先到
  `board-ready`，真实板测通过后才是`done`。
- `later`只表示保留一个可追溯的候选合同，不表示已经批准实现，也不表示它是当前item的下一步。
  重新启动任何`later`项前，必须根据当前实现、产品调用者和本轮硬件/runtime事实重审其输入、输出、范围和覆盖矩阵；
  旧计划不能直接作为施工授权。
- 状态变化只改对应行。详细checkpoint、覆盖矩阵和失败修复进入current plan；完成后整个计划移入archive。
- 每项开始前读`AGENTS.md`、本表、编号设计及本项覆盖矩阵；算法调研、pinned API确认、实现、fresh验证和
  设计/LLVM/MLIR规范复审均在本项内闭合。涉及hardware/runtime/ABI/completion/resource时先读对应事实源。
- 失败留在当前item修复，不跳过、不fallback，也不以历史输出代替本轮结果。

## 当前调度

所有板测统一由`board-testing`这一个总任务管理，包含Add/DTE、通信、GEMM、组合计算、模型和性能。
具体case与结果记录在同一份板测计划中，不再按算子、正确性或性能拆work item。编译器实现任务保留自身范围。

| 顺序 | Work item | 状态 | Owner | 直接输入 | 完成门禁 | 实施计划 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `board-testing` | `doing` | 16；关联13--15、17 | current compiler/runtime、逐case PyTorch source/input/eager reference、通过本轮no-card的原样ExecutablePackage与可用设备会话 | Add、基础Direct-DTE和三组GEMM已有实卡通过证据；区域/选路、跨Tile DDR发布/获取及参数行cache可见性已修复，AllGather/AllToAll/ReduceScatter产品与通信专项的整除/尾部完整PyTorch实卡验收全部通过，最终主机构建、回归和no-card闭合。AllReduce的产品none/search及通信专项整除/尾部验收也已闭合；当前第6项独立板测已确认native归约降rank造成物理布局错读，卷积失败缩小到3×3权重打包/硬件读取边界，1×1及其padding对照均exact。代码尚未修复，留在本项；模型和性能尚未验收。完整矩阵的PyTorch比较、guard/status、正常清理和计划内性能记录齐全后才可完成 | `tasks/plans/board-correctness-qualification.md` |
| 2 | `mesh-communication-materialization` | `queued` | 06、13、14、16 | verified card-local TensorProgram、抽象 collective participant/payload/combine 语义、layout-resolved current TileRegion；target transport 只消费已物化 peer IR | 通用Ring/recursive-doubling/ordered AllToAll/reduction算法与TX81 transport分层；execution/residency、layout/bufferization、frontend helper和search cost的current-IR合同闭合；focused current-IR、SystemC与host/no-card witness通过。真实板测及PyTorch验收由独立板测项拥有 | `tasks/plans/physical-dataflow-synthesis.md` |
| 3 | `production-host-readiness` | `board-ready` | Q53 / 16 | `mesh-communication-materialization`产出的 current-IR 与 target-independent/target transport 分界、frontend、interface、package/runtime | FP16 主纵向 source/IR/package/oracle/runner/no-card/SystemC 矩阵通过；BF16 只执行 dtype/ABI/conversion smoke；fresh 40/40 no-card registry、final FP16 conv/target-model no-card、SystemC 与 canonical build/no-op 通过；2026-09-08板机Release构建、完整check-wafer及3项fresh no-card smoke复核通过；本项不运行真实设备 | `tasks/plans/physical-dataflow-synthesis.md` |
| 4 | `recursive-doubling-feasibility` | `done` | 13、16 | 非native、power-of-two participant complete AllGather；actual contiguous per-Tile gather buffer和现有unicast DTE | 4/16-Tile、1024/1025/1031 actual Instr覆盖精确`log2(P)`轮、每Tile `(P-1)×payload` bytes；fresh completion、MiniMalloc和transport binding通过；记录与Ring/native的actual差异及production接入缺口，不在无板端crossover证据时替换当前算法 | `tasks/plans/physical-dataflow-synthesis.md` |
| 5 | `recursive-doubling-production-choice` | `done` | 06、13、16 | layout-resolved、structured-to-Tile complete AllGather current IR；feasibility已闭合的aggregate-buffer和range-aware completion | search将Ring与recursive doubling分别物化为candidate-owned actual IR并经MiniMalloc/target/cost比较；baseline和native路径不变；4/16-Tile、1024/1025/1031、eligible/ineligible/capacity与actual-cost selection矩阵fresh通过；不新增collective op、shadow plan或板端结论 | `tasks/plans/physical-dataflow-synthesis.md` |
| 6 | `mesh-all-to-all-production-choice` | `done` | 06、13、16 | current Tile peer IR中all-and-only complete personalized exchange；static homogeneous piece及完整rectangular Tile mesh | search比较现行direct/native与二维dimension-ordered aggregate的actual IR；后者只使用current source/receive buffers、actual pack/repack allocation和typed subview，4/16-Tile、1024/1025/1031精确覆盖；总logical bytes不变，4×4每Tile peer message由15降至6；全部candidate经fresh completion、MiniMalloc、target和cost，不发明route或shadow plan | `tasks/plans/physical-dataflow-synthesis.md` |
| 7 | `distributed-reduce-scatter-production-choice` | `done` | 06、07、13、16 | current Tile peer IR中的complete per-destination contribution matrix及closed associative local combine use-def | 将现行central/direct merge与minimum-hop Ring ReduceScatter分别物化为actual peer op和`wafer.tile.elementwise` combine；每个destination shard all-and-only消费全部actual contributions，1024/1025/1031与4/16-Tile经fresh downstream闭合；无法证明combiner closure时不改写 | `tasks/plans/physical-dataflow-synthesis.md` |
| 8 | `distributed-all-reduce-production-choice` | `done` | 06、07、13、16 | current Tile peer IR中的complete full-buffer fanin、closed associative merge和complete result fanout；`distributed-reduce-scatter-production-choice`产出的Ring mechanics | AllReduce Ring复用同一actual ReduceScatter materializer并接现有AllGather，不维护第二套归约算法；ragged contiguous chunk、全部participant replicated result、2(P-1)轮及actual cost selection在1024/1025/1031、4/16-Tile闭合；其它形态保留central merge+fanout | `tasks/plans/physical-dataflow-synthesis.md` |

## 已满足的直接前置

| Owner | 状态 | Current作用 | 证据入口 |
| --- | --- | --- | --- |
| Q52 | `done` | none/search独立current-IR路径、bounded RegionPlan refinement、public search width/trials、actual feedback、safe incomparable delivery和逐Tile actual inventory | 06；`tasks/archive/physical-dataflow-synthesis-q52-plan-history.md`；`tasks/archive/completed-task-index.md` |
| 01、04、06、07、10、14、16、18--20 | `done` | 架构、device-scope术语、源码/component边界和canonical build | 编号设计；`tasks/archive/completed-task-index.md` |
| Q50.0 | `done` | policy-complete Instr共同消费的actual SPM/DDR/transport/target leaf | 06、09、12--14 |
| Q51/Q50.S | `done` | structural choice/domain算法和attention semantic/decomposition donor；不作为current事实源 | 05、06；历史见completed index |
| Q55 | `done` | current target/package/runtime interface | `tasks/archive/interface-version-consolidation.md` |
| Q56 | `board-ready` | current package data与host/no-card合同；真实板端未执行 | `tasks/archive/executable-package-and-resident-runtime.md` |
| Q60 | `done` | product frontend与portable StableHLO ingestion | `tasks/archive/compiler-entry-productization.md` |

## 延后候选（重新立项前必须重审）

以下工作不会随当前主线自动启动。它们保留在队列中只是为了记录可能的后续方向；启动前必须重新确认当前产品
调用者、硬件/runtime证据、实现边界和逐项覆盖矩阵。没有新的调用者或证据时，不为旧计划继续维护代码。

| Tracking ID | Semantic key | 状态 | 调度判断 | 重新启动条件 | 完成边界 | Owner / plan |
| --- | --- | --- | --- | --- | --- | --- |
| Q57 | `resident-static-execution` | `later` | 低优先级；只在确有常驻调用者时考虑 | Q56和Q53均`board-ready`，并且有明确的resident API调用者、板端窗口和fresh输入 | 同一package的prepare/submit*/close；load-once、data H2D-once、稳定地址、typed completion和poison板端闭合 | 15--17；`tasks/plans/resident-static-execution.md` |
| Q61 | `whole-program-scale-readiness` | `later` | 低优先级；属于规模资格而非核心编译能力 | Q53 `board-ready`、Q58、Q60，并冻结完整程序输入、资源预算和验收owner | 普通产品driver处理完整小模型、data-heavy和graph-heavy程序；work、I/O、wall、RSS、disk和package bytes可对账 | 01--02、06、14--18；`tasks/plans/whole-program-scale-readiness.md` |
| Q48 | `semantic-superoptimization` | `later` | 研究/可选方向；不属于FA/FD核心闭合 | 有明确的语义alternative消费者、收益假设和独立等价oracle；先重写TensorProgram表示和Q52 handoff合同 | actual structured MLIR alternatives经query-local等价证明后各自进入Q52 current-IR pipeline；只有accepted owner发布 | 05--08、10--11、16--18；`tasks/plans/semantic-superoptimization.md` |
| Q47 | `target-abi-retirement` | `later` | 仅剩板端资格性质；不作为新的compiler开发项 | current interface/package/transaction已闭合且取得真实板端资格窗口；先核对archive中的旧完成边界与current ABI | current package定向重签ordinary与Direct-DTE host/no-card/board纵向；不回放历史package | 11、14--17、20 |

## 外部证据/可选积压（不进入当前线性主线）

以下条目没有当前产品调用者或本地可完成的证据闭环，统一保持`later`，不作为Q53之后的默认开发顺序。
启动前必须先建立新的编号设计和完成门禁，不能直接沿用旧计划。

| Tracking ID | Semantic key | 当前处置 |
| --- | --- | --- |
| Q9.R | `profile-writing-overhead` | 只有profiler写入成为明确瓶颈时重开 |
| Q22.C/E/K/P | target-model qualification | 分别等待numeric corpus、simulator/ISS、vendor provenance或validated timing evidence |
| Q32.T | `compiler-transform-control` | 只有出现明确external control-plane consumer时重开 |
| Q38.W | `multi-worker-production-promotion` | 只有有独立命令链收益假设和matched board A/B时重开 |
| Q3.6 | `crt-writeback-scalar` | 只有Count predicate、wrapper、target/model证据和readback需求明确时重开 |

不在当前DAG中的distributed/executable dialect、MPMD rank class、跨卡coherent variant、WCRE/global registry、
capability lease、跨model migration、shared-weight cache、segmented MoE及70B/100GB stress，恢复时必须先建立编号设计和完成门禁。

## 导航

- 编号设计和archive索引：`tasks/README.md`。
- Q52/Q53 current plan：`tasks/plans/physical-dataflow-synthesis.md`；详细历史：
  `tasks/archive/physical-dataflow-synthesis-working-history.md`。
- 已完成边界：`tasks/archive/completed-task-index.md`。
- 硬件校准事实：`docs/tx81-compiler-hardware-calibration.md`。
