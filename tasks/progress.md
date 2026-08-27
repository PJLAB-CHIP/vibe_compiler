# Wafer Compiler Task Queue

更新时间：2026-08-27

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

当前执行队列只保留尚未完成的一次性work item，并按artifact producer/consumer关系排序。Q49–Q51各项的历史边界与
证据已经移入`tasks/archive/completed-task-index.md`；详细施工记录位于
`tasks/archive/physical-dataflow-synthesis-working-history.md`，不能覆盖current Q52重新发现的baseline回归、双materializer、
IR膨胀、layout未接线和verifier职责问题。

每个work item都必须独立执行下面完整流程；表中逐行重复，不能用全局说明代替本项门禁：先读`AGENTS.md`和本表，再读编号设计及
本项覆盖矩阵；随后调研与问题直接相关的论文、经典算法和成熟编译器实现，比较合法域完整性、复杂度、正确性依据、可维护性及本仓
适用边界，并在本项设计中写清采用方案与未采用方案。凡涉及hardware、runtime、ABI、completion、resource或memory hierarchy，
必须先读对应事实源并把结论标为`supported`、`board-observed`、`unknown`或`excluded`；未证同步语义保持unknown，禁止靠通用经验
补join/wait。只有算法方向明确后，才查官方文档和仓库pinned LLVM/MLIR源码确认具体API，
不能用阅读本地源码代替算法调研。之后修改代码与测试并fresh验证；最后按编号设计以及LLVM/MLIR工程规范复审实现、完整diff和
下游witness，确认无遗漏后才更新状态并提交。

| 顺序 | Work item | 状态 | 设计owner | 直接输入 | 完成输出 | 本项执行流程 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `search-scalability` | `doing` | Q52 | Q51 search controller、structural choice与attention展开donor，current baseline regression，Q50.0 actual memory/target leaf | 按current plan的16个线性checkpoint施工：第6--9项已经闭合actual leaf、Instr completion、execution structure和movement；第10项通过pinned `egg` Rust staticlib和薄C ABI，在05号post-attention边界以bounded access-relation e-graph收敛static reshape/transpose/broadcast/canonical concat、pure elementwise及受限contraction/reduction graph，不自研第二个C++ core；第11项以pinned MLIR SCF tile-and-fuse替换手写三段展开并删除future delivery协议；第12项消费最终current SSA补齐完整layout domain/PBQP assignment、exact view和bufferization，保留已完成的output DPS/copy closure/solver资产但整体状态重新打开；第13项运行baseline gate；第14项原子切换search current-IR production、resource-aware winner并删除旧shadow；第15项闭合e-graph/fusion/layout/movement/Instr inventory；第16项完成同源LLaMA `none`/`search`验收。SPM legality仍只由actual MiniMalloc决定 | 读AGENTS/progress→读05/06及本项16个矩阵→按每个semantic work item分别完成算法调研、pinned API确认、代码/测试、fresh验证和MLIR规范复审→更新状态并提交 |
| 2 | `production-host-readiness` | `queued` | Q53 | search-scalability、Q60产品入口、Q55 current interface、Q56 board-ready package/runtime | 从fresh产品source完成representative source/IR/package/oracle/runner/no-card矩阵并准备无需设备上临时补充的board cases；状态只到`board-ready`，真实板端不在当前目标 | 读AGENTS/progress→读02/06/14–16及本项矩阵→读hardware/runtime/ABI事实→调研成熟compiler的host qualification/board-ready实现→查官方及pinned API→改runner/测试→fresh host/no-card验证→按设计和MLIR/runtime规范复审→更新状态并提交 |
失败留在当前work item修复；不跳过、不fallback，也不把owner整体状态提前标为完成。

### 当前owner完成边界

| 设计owner | Work item | Owner整体完成边界 |
| --- | --- | --- |
| Q52 | search-scalability | 本项16个线性checkpoint全部通过；post-attention global graph和candidate attention展开共同消费05号基于pinned `egg` staticlib与薄C ABI的唯一bounded access-relation e-graph kernel，不存在自研C++/外部进程第二实现；预算耗尽保持原IR，on/off以exact proof保持语义和downstream representability而不要求raw choice identity；tile-and-fuse只从actual current SSA决定producer位置；baseline/search共同使用通过flat oracle的current-IR PBQP layout optimizer，可精确投影时分别使用NE、Vector/CT、movement和instruction-control term，必要事实未知时整组禁用；唯一full-transfer cleanup已接production；actual leaf内bufferization、completion repair和allocation sinking调用为零，普通bufferization/allocation、rotating allocation与Instr completion分别只有layout、execution-structure和completion owner，SPM/DDR high-water只从accepted offsets重算；observable result直接写唯一destination，冗余DDR copy为0、必要copy有witness且typed，Instr不创建copy-only TileRegion；search原子建立structural producer和layout→movement→execution-structure→Instr/order/completion choices并从Accepted current Instr比较winner；旧shadow/future delivery/手写temporal/fusion/copy fallback零production残留；同一current FP16 LLaMA source的两次fresh package/no-card验收通过 |
| Q53 | production-host-readiness | 对应work item通过即为`board-ready`；真实板端不在当前目标 |

### 当前已满足前置

历史完成明细见`tasks/archive/completed-task-index.md`；这里仅保留两个current work item直接消费的前置。

| Owner | 状态 | 当前作用 | 证据入口 |
| --- | --- | --- | --- |
| Q50.0 | `done` | 两条policy各自产生policy-complete Instr后共同消费的actual SPM/DDR/transport/target leaf；不拥有complete materializer | 06、09、12–14；历史见`tasks/archive/completed-task-index.md` |
| Q51/Q50.S | `done` | current search controller、structural choice/domain算法donor和fixed FA/FD semantic/decomposition输入；旧physical-value/movement/storage/schedule domain不再作current事实源，也不能代签Q52 actual-IR pipeline | 05–06；历史见`tasks/archive/completed-task-index.md` |
| Q55 | `done` | current target/package/runtime interface closure | `tasks/plans/interface-version-consolidation.md` |
| Q56 | `board-ready` | current package data与host/no-card contract；真实板端尚未执行 | `tasks/plans/executable-package-and-resident-runtime.md` |
| Q60 | `done` | product frontend与portable StableHLO ingestion | `tasks/plans/compiler-entry-productization.md` |

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
| Q48 | `semantic-superoptimization` | `later` | Q53按current CardModule/Tile合同达到`board-ready`，且current compiler/runtime closure可消费final Instr/TargetCall；当前前置未满足 | 从actual structured MLIR生成verifier-legal TensorProgram alternatives并以query-local solver证明外部可观察value、memory与effect等价；每个alternative作为自己的current TensorProgram进入Q52 actual-IR candidate pipeline，只有一个accepted owner进入最终发布。Q48不在Instr层建立第二selector，不建立shadow planning state，不预设shortlist或固定候选cap。 | 05-08、10-11、16-18；`tasks/plans/semantic-superoptimization.md` |
| Q38.W | `multi-worker-production-promotion` | `later` | current typed ABI、actual clone和host/model资格已闭合，且出现需要隔离等待域并可能受益的独立命令链 | 以configured-board matched correctness/performance证明非零worker相对worker0流水的明确收益后才允许normal production promotion；不得为使用worker1/2而拆分已能在worker0并行的流水。 | 08、11、14-17 |
| Q3.6 | `crt-writeback-scalar` | `later` | 明确Count predicate及wrapper/target/model evidence | 独立闭合typed instruction、effect/completion、ABI/CRT、model和必要package readback。 | 11、14-17 |

不在当前DAG中的model/distributed/executable dialect、MPMD/rank class、跨卡coherent variant、
WCRE/global registry、capability lease、跨model state migration、共享weight cache、segmented MoE及
70B/100GB stress，恢复时必须先建立编号设计和completion gate。

## 导航

- Q52–Q53当前实施计划：`tasks/plans/physical-dataflow-synthesis.md`；Q49–Q51详细施工历史位于
  `tasks/archive/physical-dataflow-synthesis-working-history.md`，稳定设计由06拥有。
- Q54历史实施计划：`tasks/archive/mlir-engineering-remediation.md`；稳定工程合同由19和`AGENTS.md`拥有。
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
