# Wafer Compiler Task Queue

更新时间：2026-08-24

本文件是任务调度入口，只记录任务状态、前置关系、当前工作、完成门禁和设计/证据owner。具体设计、
pipeline contract、实验结论、测试数字、失败修复过程和历史复盘不在这里重复；分别进入编号设计文档、
`tasks/plans/`、`tasks/archive/`、`docs/`或`memory/`。已完成任务索引位于
`tasks/archive/completed-task-index.md`，更细的历史状态变化由Git保留。

## 队列规则

- `Q*`只标识稳定设计/能力owner，不表示pipeline层级，也不直接充当可反复进入的施工任务。
- current execution work item使用稳定semantic key；每个work item只有一个输入边界、一个输出边界，在当前调度中只出现一次。
- 全局至多一个work item为`doing`；`next`表示前置已满足但尚未开始，`queued`表示已经进入当前主线但仍等待直接前置闭合，
  `later`表示不进入当前主线。
- work item的`done`只表示该行输出合同已经满足；Q owner的整体完成只由owner map列出的全部work item及production gate汇总，
  不让owner row在`doing/queued`之间往返。
- current职责被后继吸收时，旧任务直接移出当前调度和状态表；可复用的source、oracle、mechanics与验证要求写入
  current owner，历史只由Git和`tasks/archive/`保留，不新增旧任务过渡状态或旧任务索引。
- `blocked`必须写明尚缺的外部条件或上游任务，不能用工作过程代替阻塞原因。
- 新work item先进入本表并绑定编号设计owner；非小修再更新`tasks/plans/`实施计划。
- 状态变化只更新对应row；不得追加按日期、轮次或测试批次展开的worklog。
- 后续任务gate只验证直接影响面；除非任务本身改动相关ABI/runtime/firmware，否则禁止全量审计和无关回归。
- 包含板端验证的任务必须把`board-ready`写为无卡阶段门禁；`board-ready`不得标记为`done`。

## 当前调度

当前执行队列只列一次性work item；顺序已经按artifact producer/consumer关系拓扑排序。Q50.0、Q54、Q55、Q56、Q59、Q60、Q62、
Q63和Q64等前置已满足，不在当前队列中重复展开。2026-08-23重新闭合的第1项保留actual-feedback/SPM机制，同时删除了
production `none`中的per-block/per-element/structural NCC join和无依据的issue后立即DTE await；第2--4项的当前输出仍成立。
原第5--17项只闭合了部分
domain/API、canonical路径或单测mechanism，没有满足各自设计中的selected construction、直接production consumer或完整search门禁，
因此在原语义身份上重新打开，不旁挂第二份closure任务。已经通过的局部单测继续作为实现素材，不能代签这些重新打开项的`done`。

每个work item都必须独立执行下面完整流程；表中逐行重复，不能用全局说明代替本项门禁：先读`AGENTS.md`和本表，再读编号设计及
本项覆盖矩阵；随后调研与问题直接相关的论文、经典算法和成熟编译器实现，比较合法域完整性、复杂度、正确性依据、可维护性及本仓
适用边界，并在本项设计中写清采用方案与未采用方案。凡涉及hardware、runtime、ABI、completion、resource或memory hierarchy，
必须先读对应事实源并把结论标为`supported`、`board-observed`、`unknown`或`excluded`；未证同步语义保持unknown，禁止靠通用经验
补join/wait。只有算法方向明确后，才查官方文档和仓库pinned LLVM/MLIR源码确认具体API，
不能用阅读本地源码代替算法调研。之后修改代码与测试并fresh验证；最后按编号设计以及LLVM/MLIR工程规范复审实现、完整diff和
下游witness，确认无遗漏后才更新状态并提交。

| 顺序 | Work item | 状态 | 设计owner | 直接输入 | 完成输出 | 本项执行流程 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `deterministic-baseline-closure` | `done` | Q49.P | canonical-plan-coverage-closure、Q59、Q63、current hardware/ABI completion facts | 保留actual candidate→SPM→typed feedback闭环；删除per-block/per-element/structural join与unproved immediate await，fresh `none`满足最小completion及动态work gate | 读AGENTS/progress→读编号设计与本项矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→改代码/测试→fresh验证→按设计与LLVM/MLIR规范复审→更新状态并提交 |
| 2 | `spatial-domain` | `done` | Q50.B | spatial-plan-schema、attention-spatial-integration、attention-demand-integration、exact-demand-boundary、Q64 | complete spatial successors、reference enumerator及proposal | 读AGENTS/progress→读编号设计与本项矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→改代码/测试→fresh验证→按设计与LLVM/MLIR规范复审→更新状态并提交 |
| 3 | `search-control-foundation` | `done` | Q51.Core | spatial-domain、exact-demand-boundary、attention-demand-integration | SpatialState frontier/continuation及public `search` routing；missing axis typed incomplete | 读AGENTS/progress→读编号设计与本项矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→改代码/测试→fresh验证→按设计与LLVM/MLIR规范复审→更新状态并提交 |
| 4 | `root-work-domain` | `done` | Q50.C | search-control-foundation、canonical-root-work | full root/merge work domain、Core consumer及complete-candidate emitter输入 | 读AGENTS/progress→读编号设计与本项矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→改代码/测试→fresh验证→按设计与LLVM/MLIR规范复审→更新状态并提交 |
| 5 | `region-execution-domain` | `done` | Q50.D | deterministic-baseline-closure的shared completion/materializer修正、root-work-domain、canonical-region-plan | 完整region/execution/use-binding域、selected RegionPlan直接构造、nested/replica/coupled verifier及actual downstream witness；region builder不选择worker/participant/completion | 读AGENTS/progress→读编号设计与本项矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→改代码/测试→fresh验证→按设计与LLVM/MLIR规范复审→更新状态并提交 |
| 6 | `temporal-domain` | `done` | Q50.E | region-execution-domain、canonical-temporal-plan | complete temporal sizes/orders/tails、top-level/nested/coupled actual loop construction、verifier及Core consumer | 读AGENTS/progress→读编号设计与本项矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→改代码/测试→fresh验证→按设计与LLVM/MLIR规范复审→更新状态并提交 |
| 7 | `partial-feasibility` | `done` | Q50.F | temporal-domain、exact-demand-boundary | 复核A–E结构完整性/missing coordinates及5/6新schema，资源合法性保持unknown | 读AGENTS/progress→读编号设计与本项矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→改代码/测试→fresh验证→按设计与LLVM/MLIR规范复审→更新状态并提交 |
| 8 | `layout-domain` | `done` | Q50.G | partial-feasibility、canonical-representation-plan | operation/interface constraint graph、PBQP精确消元+residual solver、production tuple/alias facts及selected physical-version construction/verifier | 读AGENTS/progress→读编号设计与本项矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→改代码/测试→fresh验证→按设计与LLVM/MLIR规范复审→更新状态并提交 |
| 9 | `movement-domain` | `done` | Q50.H | layout-domain、canonical-movement-plan | local/DDR/direct/relay/fanout/gather current完整域、external every-root reuse、exact multi-piece payload proof、token-only selected construction/verifier及actual surgery donor retirement；不在issue后立即await；raw collective因无current typed primitive而无state | 读AGENTS/progress→读编号设计与本项矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→改代码/测试→fresh验证→按设计与LLVM/MLIR规范复审→更新状态并提交 |
| 10 | `storage-domain` | `done` | Q50.I | movement-domain、canonical-storage-plan | production alias/reuse/`1..U` requirements、peer-relay exact-piece ownership、selected object/multi-axis rotation construction及actual definition/use/completion/release lifetime verifier；零SPM估算或capacity控制流 | 读AGENTS/progress→读编号设计与本项矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→改代码/测试→fresh验证→按设计与LLVM/MLIR规范复审→更新状态并提交 |
| 11 | `event-resource-foundation` | `doing` | Q50.J | storage-domain、Q63、target/effect facts | 完整EventGraph/resource/recurrence/completion facts、Q63/effect接入及fixed-K后同一builder重建J的typed seam；missing contract保持typed unknown而非默认Synchronous | 读AGENTS/progress→读编号设计与本项矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→改代码/测试→fresh验证→按设计与LLVM/MLIR规范复审→更新状态并提交 |
| 12 | `execution-structure-domain` | `queued` | Q50.K | event-resource-foundation、storage-domain、serialized-execution | sound Serialized/Pipelined eligibility与完整有限域、selected phase/loop construction、structure verifier及donor retirement | 读AGENTS/progress→读编号设计与本项矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→改代码/测试→fresh验证→按设计与LLVM/MLIR规范复审→更新状态并提交 |
| 13 | `structure-specific-storage` | `queued` | Q50.I | execution-structure-domain、storage-domain | fixed-K occurrence/slot/lifetime重闭、actual rotating-slot construction，并触发post-K EventGraph重建 | 读AGENTS/progress→读编号设计与本项矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→改代码/测试→fresh验证→按设计与LLVM/MLIR规范复审→更新状态并提交 |
| 14 | `schedule-domain` | `queued` | Q50.J | structure-specific-storage、event-resource-foundation | post-K EventGraph、slot/FSM lifetime、worker/resource/completion完整域及minimum-participant/latest-unavoidable selected wait/join emission/verifier | 读AGENTS/progress→读编号设计与本项矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→改代码/测试→fresh验证→按设计与LLVM/MLIR规范复审→更新状态并提交 |
| 15 | `full-feasibility` | `queued` | Q50.F | schedule-domain及完整B–K→I→J selected construction | 全字段complete-candidate materialization、actual SPM/DDR/transport/target gate、plan/actual join-wait parity与动态work gate、typed rejection及Core反馈 | 读AGENTS/progress→读编号设计与本项矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→改代码/测试→fresh验证→按设计与LLVM/MLIR规范复审→更新状态并提交 |
| 16 | `search-control-closure` | `queued` | Q51.Core | full-feasibility、全部domain work items | all-axis CompleteCandidateKey、actual-result admission、cost/bound、causal no-good、coverage及independent controller oracle | 读AGENTS/progress→读编号设计与本项矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→改代码/测试→fresh验证→按设计与LLVM/MLIR规范复审→更新状态并提交 |
| 17 | `unified-search-closure` | `queued` | Q51 | search-control-closure、attention-selected-decomposition、Q50.0 | parent-by-parent/full-plan oracle、可恢复完整遍历、每complete candidate一次actual evaluation及唯一winner发布 | 读AGENTS/progress→读编号设计与本项矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→改代码/测试→fresh验证→按设计与LLVM/MLIR规范复审→更新状态并提交 |
| 18 | `attention-production-closure` | `queued` | Q50.S | deterministic-baseline-closure、unified-search-closure | donor retirement及prefill/decode的none/search package/no-card；attention algorithm层零固定worker、零per-K2-block completion | 读AGENTS/progress→读编号设计与本项矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→改代码/测试→fresh验证→按设计与LLVM/MLIR规范复审→更新状态并提交 |
| 19 | `search-scalability` | `queued` | Q52 | unified-search-closure、attention-production-closure | measured memo/DP/bound/LNS policy、有限预算LLaMA actual evaluation及唯一winner发布 | 读AGENTS/progress→读编号设计与本项矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→改代码/测试→fresh验证→按设计与LLVM/MLIR规范复审→更新状态并提交 |
| 20 | `production-host-readiness` | `queued` | Q53 | search-scalability、Q60、Q55、Q56 board-ready | fresh source/IR/package/oracle/runner/no-card矩阵；Q53 `board-ready` | 读AGENTS/progress→读编号设计与本项矩阵→读相关硬件/ABI事实并把未证同步语义保留为unknown→调研论文/成熟编译器中的相关算法与实现并比较取舍→查官方文档及pinned LLVM/MLIR确认API→改代码/测试→fresh验证→按设计与LLVM/MLIR规范复审→更新状态并提交 |
失败留在当前work item修复；不跳过、不fallback，也不把owner整体状态提前标为完成。

### 设计owner映射

| 设计owner | 组成work items | Owner整体完成边界 |
| --- | --- | --- |
| Q50.B | spatial-plan-schema、canonical-spatial-assignment、spatial-domain | 三项均通过且spatial-domain在unified search中取得production witness |
| Q50.S | attention-normalization、attention-spatial-integration、attention-demand-integration、attention-work-projection、attention-selected-decomposition、attention-production-closure | 最后一项通过 |
| Q50.A | exact-demand-boundary | work item通过且所有physical consumers迁移 |
| Q50.C | canonical-root-work、root-work-domain | 两项通过且unified search取得selected witness |
| Q50.D | canonical-region-plan、region-execution-domain | domain与selected construction均通过且unified search取得nested/replica/coupled actual witness |
| Q50.E | canonical-temporal-plan、temporal-domain | complete domain与top-level/nested/coupled actual loop construction通过且unified search取得selected witness |
| Q50.F | partial-feasibility、full-feasibility | 两项均通过；前者只签发结构完整性，后者以actual candidate gate签发资源结果 |
| Q50.G | canonical-representation-plan、layout-domain | solver、production constraint producer和selected construction均通过且unified search取得actual witness |
| Q50.H | canonical-movement-plan、movement-domain | 完整domain、selected construction/verifier和donor迁移均通过且unified search取得actual witness |
| Q50.I | canonical-storage-plan、storage-domain、structure-specific-storage | production requirement、selected slot construction、lifetime及K re-entry witness全部闭合 |
| Q50.J | canonical-schedule、event-resource-foundation、schedule-domain | Q63/effect、post-K EventGraph、slot/resource/completion schedule及actual emitter/verifier闭合 |
| Q50.K | serialized-execution、execution-structure-domain | sound domain、selected phase construction/verifier及I/J re-entry闭合 |
| Q49.P | deterministic-baseline-closure | 对应work item通过 |
| Q51.Core | search-control-foundation、search-control-closure | 完整candidate key、actual controller、bound/no-good和coverage oracle通过 |
| Q51 | unified-search-closure | 全轴独立oracle、可恢复遍历、一次actualization和唯一winner/publication通过 |
| Q52 | search-scalability | 对应work item通过 |
| Q53 | production-host-readiness | 对应work item通过即为`board-ready`；真实板端不在当前目标 |

### 当前已满足前置

历史完成明细见`tasks/archive/completed-task-index.md`；这里仅保留当前queue仍直接消费的前置状态。

| Owner | 状态 | 当前作用 | 证据入口 |
| --- | --- | --- | --- |
| Q50.0 | `done` | complete-candidate CardModule→CardExecutable actual compile/verification/admission seam | `tasks/plans/physical-dataflow-synthesis.md` |
| Q54 | `done` | MLIR scope、analysis、pipeline与rewrite infrastructure | `tasks/plans/mlir-engineering-remediation.md` |
| Q55 | `done` | current target/package/runtime interface closure | `tasks/plans/interface-version-consolidation.md` |
| Q58 | `done` | program data ownership/handoff | `tasks/plans/program-data-and-whole-program-scale.md` |
| Q56 | `board-ready` | current package data与host/no-card contract；真实板端尚未执行 | `tasks/plans/executable-package-and-resident-runtime.md` |
| Q59 | `done` | compiler transaction与package commit | `tasks/plans/compiler-entry-productization.md` |
| Q60 | `done` | product frontend与portable StableHLO ingestion | `tasks/plans/compiler-entry-productization.md` |
| Q62 | `done` | current typed target materialization/model boundary | `tasks/plans/target-numeric-contract-reconstruction.md` |
| Q63 | `done` | NCC completion interface与analysis facts | `tasks/plans/ncc-synchronization-contract-layering.md` |
| Q64 | `done` | source registration与library ownership truth | `tasks/plans/source-registration-truth-closure.md` |

## Later / External Gates

这些任务不会因当前主线完成自动启动；外部条件满足后先更新状态和对应设计。

| Tracking ID | Semantic key | 状态 | 启动条件 | 完成边界 | 设计 owner |
| --- | --- | --- | --- | --- | --- |
| Q9.R | `profile-writing-overhead` | `later` | Q9、进入profiler writing优化排期 | raw evidence只有一个canonical owner，analysis只发布ID/dictionary引用的摘要，HTML不再重复内嵌全量analysis/evidence，raw按需压缩加载；dense fixture证明规模随事件数线性并通过size gate。板端output readback是独立runtime开销，不计入report writing体积。当前generator和测试仍保留宽对象、pretty JSON及全量内嵌等旧路径，本项尚未实现。 | 16 |
| Q22.C | `target-model-numeric-correlation` | `later` | Q22、Q32、Q6.B、configured numeric corpus | 按具体target operation、dtype/layout/parameter domain冻结board区分向量、held-out payload和comparator，并绑定current target/environment；不通过通用capability/profile registry签发。现有workload证据不能单独代签。 | 16、17 |
| Q22.E | `target-model-package-execution` | `later` | Q18、Q22、Q32、configured simulator/ISS | 原样执行verified package及all-and-only RISC-V ELF；schema升级必须先独立完成。 | 15、16、17 |
| Q22.K | `target-model-packet-provenance` | `later` | Q22、owner-approved vendor package或公开规范 | 建立可引用的CRT/packet/MMIO provenance；缺失不阻塞functional CModel。 | 14、16、17 |
| Q22.P | `target-model-timing-calibration` | `later` | Q32、Q22.C、validated PMU/timing environment | 校准LT/AT；没有RTL/vendor cycle证据不声明cycle accuracy。 | 16、17 |
| Q32.T | `compiler-transform-control` | `later` | Q32、明确的external control-plane consumer | 复用现有rewrite/conversion；Transform IR不保存physical-dataflow search state，也不替代06的唯一candidate owner。 | 01、05-08、10、16、18 |
| Q47 | `target-abi-retirement` | `later` | Q55、Q56、Q59 current interface/package/transaction闭合，且进入configured-board资格窗口 | 编译器侧ABI retirement已经闭合；启动时只用Q59之后的current package定向重签ordinary与Direct-DTE source→package→model/no-card，随后才标`board-ready`。真实板端只验证current ABI correctness、guard和lifecycle，通过后标`done`；不重复Q53性能A/B，也不执行任何历史package批次。 | 11、14-17、20 |
| Q57 | `resident-static-execution` | `later` | Q56达到`board-ready`且Q53按新package合同达到`board-ready` | 不改变selected `CardExecutable`、`TileEntryArgument`、program-data offset或Tile指令；把Q56的一次执行lifetime延长为`PreparedExecution -> submit* -> close`。PreparedExecution拥有loaded kernel modules、optional non-empty program data `BoardDeviceMemory`、invocation `BoardDeviceMemory`、pointer rows、transport/profile状态和provider failure state；submission/completion携带device generation。TX provider初始只single context、`max_inflight=1`、无cancel；板端证明load-once/run-many、non-empty program data只H2D一次、稳定地址、workspace初始化/覆盖、public terminal和poison quarantine。one-shot只保留同一路径便利封装。request scheduler、runtime-owned KV policy、multi-inflight、persistent device loop、跨卡和external execution-engine policy均不在本项。 | 12、14-17；`tasks/plans/executable-package-and-resident-runtime.md` |
| Q61 | `whole-program-scale-readiness` | `later` | Q53达到`board-ready`、Q58、Q60 | 在主search、产品frontend和package data链闭合后，使用完整程序而非单算子/单block验证source→TensorProgram→search→`CardExecutable`→`ExecutablePackage`的规模行为；同时覆盖完整小模型语义、大型完整parameter inventory和至少一个完整大图，记录IR/op/candidate/work、wall、RSS、disk/IO、target/package bytes及失败分类。Llama-2 7B可作一个scale witness，但模型名、层数和LLM调度不进入合同；本项不以full-model board execution、resident runtime或serving integration作默认完成门禁。 | 01-02、06、14-18；`tasks/plans/program-data-and-whole-program-scale.md` |
| Q48 | `semantic-superoptimization` | `later` | Q53按current CardModule/Tile合同达到`board-ready`，且current compiler/runtime closure可消费final Instr/TargetCall；当前前置未满足 | 从actual structured MLIR生成verifier-legal TensorProgram alternatives并以query-local solver证明外部可观察value、memory与effect等价；alternatives进入Q51 planning state，完整physical candidate通过同一actual admission，只有一个accepted winner进入最终发布。Q48不在Instr层建立第二selector，不预设shortlist或固定候选cap。 | 05-08、10-11、16-18；`tasks/plans/semantic-superoptimization.md` |
| Q38.W | `multi-worker-production-promotion` | `later` | current typed ABI、actual clone和host/model资格已闭合，且出现需要隔离等待域并可能受益的独立命令链 | 以configured-board matched correctness/performance证明非零worker相对worker0流水的明确收益后才允许normal production promotion；不得为使用worker1/2而拆分已能在worker0并行的流水。 | 08、11、14-17 |
| Q3.6 | `crt-writeback-scalar` | `later` | 明确Count predicate及wrapper/target/model evidence | 独立闭合typed instruction、effect/completion、ABI/CRT、model和必要package readback。 | 11、14-17 |

不在当前DAG中的model/distributed/executable dialect、MPMD/rank class、跨卡coherent variant、
WCRE/global registry、capability lease、跨model state migration、共享weight cache、segmented MoE及
70B/100GB stress，恢复时必须先建立编号设计和completion gate。

## 导航

- Q49.P、Q50.0/Q50.S/Q50.A–Q50.K及Q51–Q53的baseline隔离、机制迁移、统一搜索、scalability与production readiness共用实施计划：
  `tasks/plans/physical-dataflow-synthesis.md`。
- Q54 MLIR工程化整改计划：`tasks/plans/mlir-engineering-remediation.md`；稳定工程合同由19拥有。
- Q56 package数据闭合与Q57设备常驻执行共用实施计划：
  `tasks/plans/executable-package-and-resident-runtime.md`；稳定package/runtime字段语义由15拥有。
- Q58 program data ownership与Q61 whole-program scale共用实施计划：
  `tasks/plans/program-data-and-whole-program-scale.md`；稳定source/data/target/verification语义由02、14-16拥有。
- Q59 compiler entry transaction与Q60 frontend production entry共用实施计划：
  `tasks/plans/compiler-entry-productization.md`；稳定driver/frontend/package/source组织语义由01-02、15、18-20拥有。
- Q62 Target数值合同重建计划：`tasks/plans/target-numeric-contract-reconstruction.md`；稳定target operation、
  materialization、model/evidence与源码边界由01、11、14、16-18拥有。
- Q63 NCC completion合同分层计划：`tasks/plans/ncc-synchronization-contract-layering.md`；稳定Instr、communication、
  model和MLIR adapter边界由11、13、17-19拥有。
- Q64 source registration truth闭合计划：`tasks/plans/source-registration-truth-closure.md`；稳定source/library规则由18拥有。
- Q48语义驱动superoptimizer计划：`tasks/plans/semantic-superoptimization.md`。
- Q9完成证据：`tasks/archive/board-profiler.md`。
- 已完成任务边界与证据索引：`tasks/archive/completed-task-index.md`。
- 编号设计与归档导航：`tasks/README.md`。
- 硬件校准结论：`docs/tx81-compiler-hardware-calibration.md`。
