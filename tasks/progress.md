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

用户本轮要求先整理搜索组织调整方案，并增加按方案计费的 deep search 设计；已写入
[06号搜索合同](06-physical-dataflow-synthesis.md#75-主搜索实现分支与-deep-预算)及
[现有搜索组织计划](plans/physical-search-organization.md)。已按用户纠正移除实现分支的基础可行前置条件。
已补入核心及全部已通过case的逐项性能不下降、deep核心实卡收益门槛，详见
[统一验收矩阵](plans/board-workload-matrix.md#搜索组织修改的性能验收)。用户已授权按方案实施到验收闭合，
当前已接入实现选择跨 retile 的绑定、独立分支状态、两种预算及正式入口，已修复 GEMM 驻留搜索和 LLaMA transport/collective 组合回归。
LLaMA及大GEMM配对实卡的完整数值已通过；随后ViT1024 baseline输出通过，但该次运行后的内核检查发现TDMA异常，
批次已停止，用户确认尚未恢复。此前将此异常关联到GEMM4097的记录已按原始日志更正，实际触发操作仍未知。
用户随后指名单次Add检查：当前编译器fresh构包/no-card及FP16全部输出exact通过，运行窗口未发现新内核错误。
当前不再仅以此前告警作为等待重启的前置；继续按具体case验证，TDMA根因仍未知。
全矩阵暴露的共享初始化、降rank切片及矩形证明问题已完成主机修复；全矩阵构包部分完成，余下编译进程已中断、待续跑；
本轮追加13项standard/deep配对板测，96次完整输出均通过且无新设备错误；按当前冻结版本，standard已有18/51项、deep已有13/51项实卡数值记录。
deep本批仅8/2，小方案预算未证明性能收益，LocalReduce测得更慢；正式deep性能验收改用8/42，standard仍为8/42。
扩大deep 8/42后，4/13项完成实卡；standard补编译已完成，冻结版本no-card为51/51。
用户已要求停止在搜任务并收敛细调；9个旧搜索进程、板测等待及后续自动续跑队列均已停止，已有产物保留。
已实现有界细调：每方案最多一轮、每轴左右最近对齐点，移除逐元素细调及反复重启；容量修正与两模式计费不变。
本轮定向主机回归及正式source→package/no-card已通过；新版全矩阵和实卡性能保护尚未重签，详细验证见搜索组织计划。
旧批次不自动恢复，剩余deep/实卡仍未完成。
全部既有通过case的两模式数值、逐项性能与deep收益门槛尚未闭合；
仍归 `board-testing`，不新增队列项，也不代签下述模型、性能和板端未完成门槛。

此前用户授权优先修正局部拼接引入的重复访问，恢复原有共享与性能；本轮已修复复用约束，LLaMA block两种dtype及
三组大GEMM均完成三次完整实卡数值与性能保护。此前ViT S1024/1025默认8/42搜索各有4个actual SPM accepted，
最终target拒绝导出图中的F32 attention GEMM输入；用户已授权并完成02号低精度attention导出，保留原reference和验收标准。
两个完整ViT的XLA数值通过；进一步修复06号空间物化对真实多矩形view image的拒绝后，
两个尺寸均完成默认8/42构包、fresh no-card及各三次完整实卡数值验收。
五项LLaMA/GEMM重新构包并no-card通过，完整包与最新实卡恢复版本相同。
第4项现剩完整LM S1024/1025与4K prefill，ViT性能记录作为后续优化的初始样本，尚未完成三轮调优。
TDMA专项定位暂缓，已有异常结论不改写为已修复；具体步骤与验收仍在同一板测计划。
原授权要求`board-testing`严格依次推进：完整单层LM重测→未融合图数值根因→decode性能恢复→
剩余五配置闭合→逐case性能优化。第1项完整LM重测已完成：用户重启后，两种dtype完整输出及
FP16→BF16连续执行均通过；此前超时未复现、根因未知，不声称已修复。第2项已定位并修复broadcast遗漏非单位轴置换，
冻结未融合图经当前正式后端后FP16/BF16完整实卡数值通过，block两种dtype及三组大GEMM保护均已通过。
当前第3项已修复decode非连续目标导致逐行DMA的通用根因，普通两步完整数值通过；耗时与旧快包仍有差距，健康性能资格待重签。
此前第2步profile发生completion超时；固件记录定位到Trace未返回，旧快包窗口另有先行LSU TDMA fatal，因果未知。
用户重启并授权直接复测后，带诊断日志及未加日志的原包两步均完成Primary/Count/Trace，全部输出及actual KV接续通过，旧超时未复现。
按用户要求清理额外诊断代码、重复检查脚手架和汇总。旧快包普通执行再次报告Tile 1 LSU TDMA异常；随后fresh Add通过，
单独抽出的原1 MiB GatherScatter也完整通过，尚未定位整包异常根因。用户要求暂缓该问题，当前转入第4项，先处理ViT S1024/1025编译失败；
第3项剩余decode性能及TDMA定位保留未完成，第5项待第4项闭合后推进。上板前检查全系统占用、异常即停。所有修改须通用，
LLaMA block及大GEMM的完整数值和匹配性能是共用修改的保护门槛，具体矩阵见同一板测计划。

| 顺序 | Work item | 状态 | Owner | 直接输入 | 完成门禁 | 实施计划 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `spatial-admission-and-typed-outcomes` | `done` | 06；关联07、13、14、16 | current TensorProgram、IndexRelation、SpatialPlanDomain与actual memory/target leaf | 正式入口区分unsupported/indeterminate/contract error；双向relation协调与归约init/merge闭合；非均匀反例、actual executable/capacity反馈及同输入search成功；host/fresh no-card门禁 | `tasks/archive/spatial-admission-and-typed-outcomes.md` |
| 2 | `board-testing` | `doing` | 16；关联02、05、06、08--11、13--15、17 | current compiler/runtime、原始Torch XLA模型及actual IR、独立PyTorch reference | 53项范围中50项已按记录版本实卡完整数值通过；其中完整单层LM S16两种dtype、未融合图数值修复和LLaMA block/大GEMM性能恢复均已闭合。最新低精度attention导出保留原reference和精度门槛，ViT S1024/1025经默认8/42构包、fresh no-card后各三次完整实卡通过；五项LLaMA/GEMM新source及完整包与最新恢复性能版本逐byte一致。第4项剩完整LM S1024/1025编译边界及4K prefill实际workspace容量问题；4K已构包/no-card，但64.0078125 GiB workspace在设备allocation/launch前被拒绝。第3项decode剩余性能和TDMA专项按用户要求暂缓，历史超时根因未知，不由后续通过外推为已修复。第5项及最终同版本全矩阵、三轮性能仍未完成；原最好可复现目标不重置。受影响主机回归已通过，此前扩大lit的六项既有失败仍单独保留。版本、数值、故障及逐次性能证据统一见板测记录。 | `tasks/plans/board-workload-matrix.md` |
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
