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

2026-09-20用户修正板测方式：禁止为性能比较重新运行历史包，直接使用已有健康耗时记录并注明版本、配置及环境差异。
设备恢复核实后的接续先覆盖当前版本尚未验收的case，不再交错旧包或启动原配对队列；必要重复计时只运行当前版本。
用户提出连续运行后偶发异常的可能性，作为待查假设保留；当前仅能确认前18次健康、第19次报告TDMA异常，不能据此归因旧包。
本轮设备异常后的停止状态保持；本次修正不启动设备、不自动retry/reset。
随后按用户要求将CLI成功和所有可返回的失败路径统一交回厂商进程退出清理，移除`_Exit`跳过析构；
已完成主机构建、runtime回归及进程退出回调验证，未启动设备，TDMA根因和实卡退出行为仍待验证。
本项仍归`board-testing`，证据与验证边界见同一实施计划。
用户随后确认再次重启并授权完成剩余板测；本轮先核实新boot与占用，按当前版本未验收配置接续，
完成数值/guard资格后再补当前包必要计时。普通division输入同时收敛为合法有限域，历史包禁止重跑的约束继续有效。
本轮厂商正常退出版本新增FP16普通Q1两步及小prefill双dtype健康资格；随后当前BF16 4K 28-head prefill发生TDMA。
用户明确要求继续后，同workload的FP16仍失败，之后出现AP清理超时。用户最新要求转为定位最初的LSU TDMA，
已停止设备执行并保存现场；当前进行安装包、实际固件/driver/runtime和历史故障的离线审计。
已还原实际AP fatal bit 12→TDMA handler→EID上报路径，确认发行handler没有故障packet/PC快照；
尚未取得故障命令，根因和修复未闭合。下一次设备执行须先准备能缩小问题的采集，不能直接重开矩阵。
用户随后再次授权重启后尽快做单case观测。只读PMU采集器的16 Tile离线映射/数据/拒绝路径验证及
所选当前BF16 4K 28-head prefill的fresh no-card已完成。新boot确认后真实只读访问通过；首个计算即再次报告TDMA，批次停止。
取得64帧16 Tile寄存器：raw从0变为0x4000、命令字段留存，TDMA计数随后继续增长；正常回读/guard和厂商退出完成。
此结果说明本次复现不需要先连续跑多个case，但采样影响未隔离，raw编码和故障packet/PC仍未知。
当前转入离线字段/指令关联，不启动第二个case；70项未签资格及根因修复状态不变，见[实测记录](../docs/data/board-performance/tdma-pmu-observation-20260920.json)。
已将16 Tile实际ELF控制流的110,272次TDMA调用数与PMU对齐，87种SDK descriptor范围/字段检查通过；
last-command与destination出现一致相关性，但未恢复唯一故障指令。当前无需用户再重启，继续保留现场。

当前新增attention方案已按用户要求落入02/05/06/08/10/16号设计及
[统一实施与实卡矩阵](plans/board-workload-matrix.md#attention导出展开与实卡验收)，仍归`board-testing`。
顺序为composite接入→宽状态/causal局部展开→GEMM/layout/movement修正→原PyTorch/HF module新reference→完整实卡与性能保护。
新增Q/K/V均为`[1,28,4096,128]`的causal prefill，FP16/BF16分别验收；全部列出可执行正例都须上板。
独立attention新reference在实现更新后才启用，旧实现通过它不是前置；整网reference和既定门限保持。
用户已授权按方案推进实现及完整验收；composite/SPMD、宽状态、causal局部展开、batched方向及native广播已接入。
本轮22项none配置和原32-head 4K standard配置已完成fresh no-card与首轮完整实卡数值；
随后GQA两种长度/两种dtype、长cache两步/两种dtype、LLaMA block两种dtype、三项大GEMM及ViT1024完成首轮完整实卡；
ViT1025本轮launch窗口出现`NPU LSU TDMA Timeout`，批次已停止，未retry/reset，不签该项通过。
主机已闭合滑窗packed BOOL读取、完整TargetModel数值、guard规划与故障注入，完整lit及component回归通过。
attention性能目标仍未达到；匹配重复计时、实卡guard及最终同版本全矩阵未完成，设备异常处理后才能继续实卡。
用户已确认上述ViT1025异常后人工重启完成，并授权按接续计划推进完整验收；
先完成本轮全部编译和no-card，再核实设备身份、占用和恢复状态并执行实卡验证。scalar VS、规则mask常量模板、
私有DPS复用、GQA单位前缀native归约及DDR通知的精确NCC依赖已实现；此前扩大验证已按用户要求停止、记录保留。
用户现已授权修复layout三项问题：预算中断丢失较优解、缺少完整factor依据的剪枝，以及下游隐藏中间布局选择；
按08号合同先修求解结果保留，再恢复完整合法域，最后提前物化逐元素中间SSA；已随后续退化修复提交为`0e231204`。
该检查点的prefill静态GS由976增至1084；已定位layout名称比较触发的多余搬运，以及逐元素分解阻断select原地复用两处问题。
两处修正及其直接target、私有publication正反例已通过本轮主机验证；扩大回归发现的跨dtype factor与完整blocked BOOL copy缺口也已修复。
修复后168项相关transform、37项conversion及lit通过，canonical完整增量构建后第二次Ninja no-op；原失败与复测记录保留。
Query内物理遍历证明复用已完成定向回归；原8/42配置的prefill、Q1两步及Q2均已fresh构包/no-card通过，
Q2还通过完整TargetModel输出。相同逐Tile GEMM/循环结构下，本轮无未解释的搬运增长，layout修复主机边界已闭合；
板端actual KV接续与attention性能尚未重签，后续源码修改须重签其影响范围。
随后处理prefill/decode展开、mask、CPU与同步成本；之后先完成本轮attention矩阵及受影响保护用例的
全部编译、构包和fresh no-card，再核实设备恢复并逐例实卡验证。具体第5、6步见
[当前实施计划](plans/board-workload-matrix.md#交接与接续计划)；此前停止的其它搜索/验证批次不随此授权重启。
不再通过CPU逐元素生成规则mask；decode已在SPM中的私有rank-0广播源按现有证明直接用于VuV，避免SPM mapping读取。
这不表示scalar数学统一由CT执行。后续已补current Instr的CPU scalar计费和独立register-only F32算术候选，
两种执行方案沿同一actual leaf比较；定向layout/cost/search、正式target LLVM及完整Driver回归已通过。
TargetModel同步接入四种F32算术，精确位型、拒绝边界及完整Simulator/SystemC/lit回归通过。
三个定向产品已fresh no-card通过，Q2完整TargetModel通过；完整source/package与已提交layout版本逐字节相同。
CPU先验仍未校准，prefill的条件内动态工作量仍可为unknown，不能把其counter占位零当作无CPU开销。
最终代码`3a1dbc5a`的82项主机配置已全部完成fresh source/package/no-card；对应76项实卡配置，
另已准备72项可重新执行的历史对照。两项ViT对照保留相同原模型边界和参数，但导出bytecode不同，分别记录该限制。
重启后的runtime API1400、PCI设备、16 Tile映射及当前boot日志已核实；全系统占用仅有已核对身份的日志服务。
本轮Q2两种dtype已完成完整数值、guard及三次匹配计时；对照为旧none方案，不能据此签同预算search收益。
新版BF16普通Q1两步的完整输出、actual KV接续、旧prefix exact及guard通过，尚只有一次资格样本。
随后旧standard对照包首步在北京时间2026-09-19 22:46:48报告Tile-2 `NPU LSU TDMA Timeout`（XID12）；
已立即停止整批，无retry/reset。根因仍unknown，尚余73项配置未启动，新版Q1重复计时及全矩阵性能保护未完成。
主机验证结果及版本边界见同一实施计划。

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
用户随后授权清理旧产物并用新版重新搜索比较；已清理已取消搜索的中间目录，保留完整对照包和日志。
新版原13项standard构包/no-card全部通过，三项Division完成deep 8/42及配对实卡，搜索成本下降但未证明稳定设备收益。
用户随后要求停止：其余10项搜索和后续板测队列均已停止，无在途设备执行。
本轮按用户要求将整个search的效率方案落入06号及现有搜索组织计划：复用实际IR前缀、提前有证据的适用性检查、
统一多尺度参数过程、独立actual容量修正、方案间轮转，明确standard/deep计费、owner失效及性能覆盖。
方案同时明确deep交错与预算收尾不再保证跨预算完整trace前缀；保留同预算确定性和跨预算质量门槛，
尚未证明的集合关系及需固定的尺度/方向细节列入实施前检查。用户已进一步授权按方案实施到验收完成；
实际前缀复用、作用域适用性检查、统一多尺度提案和deep求值间轮转已实现并通过定向主机回归，当前重签产品及性能矩阵；
原13项standard构包/no-card及完整包一致性通过；用户指定的单次实卡批次已完成，多尺度实现冻结版本的13项完整数值均通过，
尚未签发性能结论。三项Division的deep及配对实卡数值通过，尚未证明设备收益。
用户进一步指定补测完整standard并核对优化收益：最终实现已完成既定清单全部构包/no-card及逐项重复实卡数值验证。
变化产物的匹配对照发现prefill尾块收益，同时确认LocalConv S1025与FP16 conv-mixed设备性能退化；
主机配对显示部分编译提速，但LocalConv编译更慢。当前性能不下降与全面效率门槛未通过，先定位并修复这些回归；
波动项及独立guard覆盖缺口保留未完成，具体样本与范围见同一计划。剩余deep编译已中断，本轮未启动，仍待后续验证。
用户指定的prefill落选DTE方案已完成实际候选导出与匹配实卡对照，完整数值通过，但设备耗时高于普通winner；
结果及分块差异见同一搜索组织计划，不据此修改搜索评分或签发DTE收益。
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
| 2 | `board-testing` | `doing` | 16；关联02、05、06、08--11、13--15、17 | current compiler/runtime、原始Torch XLA模型及actual IR、独立PyTorch reference | 既有完整单层LM S16、未融合图及LLaMA/GEMM/ViT资格按记录版本保留。当前82项host配置/86个package已完成fresh no-card；Q2双dtype健康三次计时及BF16普通Q1资格保留。厂商正常退出版本新增FP16普通Q1两步和小prefill双dtype健康资格，随后4K 28-head BF16发生TDMA；用户明确要求接续的FP16也失败。最新完成原始TDMA定位的重启后单次PMU观测，首个当前BF16计算仍告警；64帧寄存器与正常回读/厂商退出已归档，未取得故障packet/PC，根因未知。68项配置尚未启动、2项故障，共70项未签资格；必要重复计时、attention性能目标及全矩阵保护未闭合。完整LM S1024/1025、搜索性能回归/deep收益及三轮模型调优仍未完成；原最好可复现目标不重置。 | `tasks/plans/board-workload-matrix.md` |
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
