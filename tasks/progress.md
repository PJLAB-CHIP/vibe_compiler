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
具体case与验收顺序记录在同一份板测计划中，逐次已测性能与根因追加到
[`docs/board-performance-results.md`](../docs/board-performance-results.md)；不再按算子、正确性或性能拆work item。编译器实现任务保留自身范围。

| 顺序 | Work item | 状态 | Owner | 直接输入 | 完成门禁 | 实施计划 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `spatial-admission-and-typed-outcomes` | `done` | 06；关联07、13、14、16 | current TensorProgram、IndexRelation、SpatialPlanDomain与actual memory/target leaf | 正式入口区分unsupported/indeterminate/contract error；双向relation协调与归约init/merge闭合；非均匀反例、actual executable/capacity反馈及同输入search成功；host/fresh no-card门禁 | `tasks/archive/spatial-admission-and-typed-outcomes.md` |
| 2 | `board-testing` | `doing` | 16；关联02、05、06、08--11、13--15、17 | current compiler/runtime、原始Torch XLA模型及actual IR、独立PyTorch reference | 本轮已完成DDR/DTE统一的通信合法候选构造：按实际依赖、完整收发组及资源/alias约束生成可推进顺序，删除逐消息消环循环和独立transport bitmask枚举；原始ResNet默认search、verified package与fresh no-card通过。BN融合与紧凑中间域已完成。共享板测helper和reference/GEMM/MLP工具已调用同一直接XLA入口，产品抓图禁止Dynamo。通用`wafer.tile.pool`以kind区分max/min/sum/avg，支持map证明的二维窗口及缺失N/C轴；原始MaxPool/AvgPool的FP16/BF16 × 1024/1025共8项完整输出、verified package与16 Tile no-card已通过。AvgPool保持F32 opmath、padding计数及原归一化；规则双值矩形常量经既有fill/insert_slice物化。ResNet-18的224输入、完整网络与全部1000个logits已通过baseline构包和16 Tile no-card；局部卷积保留C/KH/KW归约、partial仅有输出维度，全局关系检查移至temporal批次边界，静态Pad guard已修复。融合前默认8/42 search已完成：42个实际候选、13个accepted、29次actual SPM容量拒绝，unsupported/indeterminate为零，生成verified package并通过本轮16 Tile no-card。已修复Region合并跨远端producer、Pad在PBQP后生成未建模计算导致的布局失配；通信proposal在actual Instr上闭合联合依赖，容量反馈优先推进Repair，同一accepted owner只评分一次。编译2147.294秒，含no-card共2153.622秒，峰值RSS 14,907,472 KiB；输入FP16 [1,3,224,224]、输出FP16 [1,1000]完整保留。当前XLA CPU结果相对原PyTorch仍有180/1000超默认容差，不能据此归因为Wafer目标程序。本轮20条BN全部融合，generic由351降至62，保持原scalar运算/转换与dtype；FP16/BF16全输出逐bit一致，实际main/tail Instr/SPM通过。BN阶段默认8/42为18个accepted、24次capacity、unsupported为零；5个Region合并候选可行，最佳评分候选每Tile 86个Region（旧375个），全卡DDR读写约131 MB（旧287 MB）。编译1633.478秒、含no-card 1640.848秒，RSS 4,774,528 KiB；完整428项Transforms及相关前端、pipeline和通信回归、canonical增量构建与no-op通过。本轮通信构造最终默认8/42为18个accepted、24次actual capacity，unsupported/indeterminate均为零；首轮两次查询work耗尽已通过去掉固定操作之间的无效比较解决，预算未调整。编译836.009秒、含no-card总845.348秒，比BN阶段减少48.5%；通信构造803.346秒降至69.325秒，DTE wait重建1463次降至126次。最佳评分候选仍为每Tile 86个Region、51728指令和131267008 DDR bytes，coarse评分相同；峰值RSS 5,175,408 KiB，比BN阶段增加8.4%。完整132项Driver、428项Transforms及修正后7项定向回归、canonical构建/no-op通过；当前最大单项耗时为18次评分共408.023秒。整网设备数值与性能未验收。前序embedding24项及固定FP16 LLaMA block no-card通过属于旧抓图路径，不能代签统一入口后的资格；完整LM、ViT、大GEMM、共享RHS、BF16 LLaMA数值和YOLO仍按原矩阵待办。QKV设备超时后板卡未恢复，本轮没有实卡动作，也没有性能无退化或三轮调优完成结论。详细已测边界及下一步见当前板测实施计划 | `tasks/plans/board-workload-matrix.md` |
| 3 | `mesh-communication-materialization` | `queued` | 06、13、14、16 | verified card-local TensorProgram、抽象 collective participant/payload/combine 语义、layout-resolved current TileRegion；target transport 只消费已物化 peer IR | 通用Ring/recursive-doubling/ordered AllToAll/reduction算法与TX81 transport分层；execution/residency、layout/bufferization、frontend helper和search cost的current-IR合同闭合；focused current-IR、SystemC与host/no-card witness通过。真实板测及PyTorch验收由独立板测项拥有 | `tasks/plans/physical-dataflow-synthesis.md` |
| 4 | `production-host-readiness` | `queued` | Q53 / 16 | `mesh-communication-materialization`产出的 current-IR 与 target-independent/target transport 分界、frontend、interface、package/runtime | 历史主纵向与smoke通过不能代签current模型资格；模型扩展暴露的prefill/decode/LLaMA问题已由board-testing统一修复并完成限定矩阵的FP16实卡验收。等待前置通信实现边界闭合后，按本项合同重签完整source/IR/package/oracle/runner/no-card/SystemC矩阵；板端证据见统一归档，本项不运行真实设备 | `tasks/plans/physical-dataflow-synthesis.md` |
| 5 | `recursive-doubling-feasibility` | `done` | 13、16 | 非native、power-of-two participant complete AllGather；actual contiguous per-Tile gather buffer和现有unicast DTE | 4/16-Tile、1024/1025/1031 actual Instr覆盖精确`log2(P)`轮、每Tile `(P-1)×payload` bytes；fresh completion、MiniMalloc和transport binding通过；记录与Ring/native的actual差异及production接入缺口，不在无板端crossover证据时替换当前算法 | `tasks/plans/physical-dataflow-synthesis.md` |
| 6 | `recursive-doubling-production-choice` | `done` | 06、13、16 | layout-resolved、structured-to-Tile complete AllGather current IR；feasibility已闭合的aggregate-buffer和range-aware completion | search将Ring与recursive doubling分别物化为candidate-owned actual IR并经MiniMalloc/target/cost比较；baseline和native路径不变；4/16-Tile、1024/1025/1031、eligible/ineligible/capacity与actual-cost selection矩阵fresh通过；不新增collective op、shadow plan或板端结论 | `tasks/plans/physical-dataflow-synthesis.md` |
| 7 | `mesh-all-to-all-production-choice` | `done` | 06、13、16 | current Tile peer IR中all-and-only complete personalized exchange；static homogeneous piece及完整rectangular Tile mesh | search比较现行direct/native与二维dimension-ordered aggregate的actual IR；后者只使用current source/receive buffers、actual pack/repack allocation和typed subview，4/16-Tile、1024/1025/1031精确覆盖；总logical bytes不变，4×4每Tile peer message由15降至6；全部candidate经fresh completion、MiniMalloc、target和cost，不发明route或shadow plan | `tasks/plans/physical-dataflow-synthesis.md` |
| 8 | `distributed-reduce-scatter-production-choice` | `done` | 06、07、13、16 | current Tile peer IR中的complete per-destination contribution matrix及closed associative local combine use-def | 将现行central/direct merge与minimum-hop Ring ReduceScatter分别物化为actual peer op和`wafer.tile.elementwise` combine；每个destination shard all-and-only消费全部actual contributions，1024/1025/1031与4/16-Tile经fresh downstream闭合；无法证明combiner closure时不改写 | `tasks/plans/physical-dataflow-synthesis.md` |
| 9 | `distributed-all-reduce-production-choice` | `done` | 06、07、13、16 | current Tile peer IR中的complete full-buffer fanin、closed associative merge和complete result fanout；`distributed-reduce-scatter-production-choice`产出的Ring mechanics | AllReduce Ring复用同一actual ReduceScatter materializer并接现有AllGather，不维护第二套归约算法；ragged contiguous chunk、全部participant replicated result、2(P-1)轮及actual cost selection在1024/1025/1031、4/16-Tile闭合；其它形态保留central merge+fanout | `tasks/plans/physical-dataflow-synthesis.md` |


## 已满足的直接前置

| Owner | 状态 | Current作用 | 证据入口 |
| --- | --- | --- | --- |
| Q52 | `done` | none/search独立current-IR路径、bounded RegionPlan refinement、public search width/trials、actual feedback和逐Tile actual inventory；合法候选比较现按06号统一标量估时合同 | 06；`tasks/archive/physical-dataflow-synthesis-q52-plan-history.md`；`tasks/archive/completed-task-index.md` |
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
