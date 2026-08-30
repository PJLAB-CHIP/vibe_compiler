# Wafer Compiler Task Queue

本文件是当前任务状态、直接前置和完成门禁的唯一入口。稳定设计在编号文档，实施步骤在`tasks/plans/`，
已完成边界和证据在`tasks/archive/completed-task-index.md`。这里不保存施工日志、算法说明或测试数字。

## 状态规则

- current work item按artifact producer/consumer关系线性排列；同一item只出现一次，全局至多一个`doing`。
- `next`表示直接前置已满足；`queued`表示等待表中前序；`later`表示不在当前主线；板端工作先到
  `board-ready`，真实板测通过后才是`done`。
- 状态变化只改对应行。详细checkpoint、覆盖矩阵和失败修复进入current plan；完成后整个计划移入archive。
- 每项开始前读`AGENTS.md`、本表、编号设计及本项覆盖矩阵；算法调研、pinned API确认、实现、fresh验证和
  设计/LLVM/MLIR规范复审均在本项内闭合。涉及hardware/runtime/ABI/completion/resource时先读对应事实源。
- 失败留在当前item修复，不跳过、不fallback，也不以历史输出代替本轮结果。

## 当前调度

| 顺序 | Work item | 状态 | Owner | 直接输入 | 完成门禁 | 实施计划 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `search-scalability` | `doing` | Q52 / 06 | Q51 choice/domain donor、第12项actual structural owner、第13项actual SCF/SSA tile-and-fuse owner、第14项final Linalg/Tensor/SCF owner、第15项layout-resolved/bufferized current owner、current baseline regression、Q50.0 actual memory/target leaf | 当前执行第16项`current-ir-downstream-orchestration`；完成后线性执行第17--20项。none/search独立且无fallback；SPM只由actual MiniMalloc判定；同源LLaMA产品纵向通过 | `tasks/plans/physical-dataflow-synthesis.md` |
| 2 | `production-host-readiness` | `queued` | Q53 / 16 | search scalability、current frontend、interface、package/runtime | fresh source/IR/package/oracle/runner/no-card矩阵通过并达到`board-ready`；本项不运行真实设备 | `tasks/plans/physical-dataflow-synthesis.md` |

Q52当前直接项是第16项`current-ir-downstream-orchestration`。第12项已经形成ordinary contribution/merge/output、FA/FD per-Tile online state、
selected FD merge/finalize和只连接actual endpoints的current relations；第13项已经从live TileRegion operations建立并立即消费temporal
choice，形成ordinary/online K2 actual SCF loop、direct及view-transparent exact fusion、bounded tile-local concat、constant pad和
pack/unpack materialization、three-state recurrence与static main/tail；Spatial与Temporal共用同一static tensor indexing query，covered
producer在第15项后无完整intermediate allocation/copy。第14项已经将每个current
online-attention确定性分解为QK、scale/mask、row max/sum、state scale和PV的actual Linalg/Tensor/arith/math，保留既有loop、spatial merge和
endpoint，layout入口graph/online attention为零。第15项已经从current value/use/op tuple求解并立即应用exact PBQP canonical layout，
形成shared actual conversion、observable DDR subview、cross-Tile source piece及一次function/region-local bufferization；旧node-ID buffer
attribution和第二条Instr bufferization入口已经删除，performance descriptor projection在第16项抽取shared exact query前明确hard-only。
此前的actual leaf、Instr completion、execution
structure、movement、structured logical normalization和legacy shadow-materializer retirement只作为输入，不在current队列重做。
none和search在各自controller重启前均明确返回typed unavailable且不发布package。

## 已满足的直接前置

| Owner | 状态 | Current作用 | 证据入口 |
| --- | --- | --- | --- |
| 01、04、06、07、10、14、16、18--20 | `done` | 架构、device-scope术语、源码/component边界和canonical build | 编号设计；`tasks/archive/completed-task-index.md` |
| Q50.0 | `done` | policy-complete Instr共同消费的actual SPM/DDR/transport/target leaf | 06、09、12--14 |
| Q51/Q50.S | `done` | structural choice/domain算法和attention semantic/decomposition donor；不作为current事实源 | 05、06；历史见completed index |
| Q55 | `done` | current target/package/runtime interface | `tasks/archive/interface-version-consolidation.md` |
| Q56 | `board-ready` | current package data与host/no-card合同；真实板端未执行 | `tasks/archive/executable-package-and-resident-runtime.md` |
| Q60 | `done` | product frontend与portable StableHLO ingestion | `tasks/archive/compiler-entry-productization.md` |

## Later / External Gates

以下工作不会随当前主线自动启动；启动时先建立或更新对应current plan的逐项覆盖矩阵。

| Tracking ID | Semantic key | 状态 | 启动条件 | 完成边界 | Owner / plan |
| --- | --- | --- | --- | --- | --- |
| Q57 | `resident-static-execution` | `later` | Q56和Q53均`board-ready` | 同一package的prepare/submit*/close；load-once、data H2D-once、稳定地址、typed completion和poison板端闭合 | 15--17；`tasks/plans/resident-static-execution.md` |
| Q61 | `whole-program-scale-readiness` | `later` | Q53 `board-ready`、Q58、Q60 | 普通产品driver处理完整小模型、data-heavy和graph-heavy程序；work、I/O、wall、RSS、disk和package bytes可对账 | 01--02、06、14--18；`tasks/plans/whole-program-scale-readiness.md` |
| Q48 | `semantic-superoptimization` | `later` | Q53 `board-ready`且current compiler/runtime可消费final Instr/TargetCall | actual structured MLIR alternatives经query-local等价证明后各自进入Q52 current-IR pipeline；只有accepted owner发布 | 05--08、10--11、16--18；`tasks/plans/semantic-superoptimization.md` |
| Q47 | `target-abi-retirement` | `later` | current interface/package/transaction闭合并进入板端资格窗口 | current package定向重签ordinary与Direct-DTE host/no-card/board纵向；不回放历史package | 11、14--17、20 |
| Q9.R | `profile-writing-overhead` | `later` | profiler writing进入排期 | raw evidence单一owner，report规模随事件数线性且通过size gate | 16 |
| Q22.C | `target-model-numeric-correlation` | `later` | configured numeric corpus和板端环境 | 按具体target op、dtype/layout/parameter domain完成held-out board correlation | 16、17 |
| Q22.E | `target-model-package-execution` | `later` | configured simulator/ISS | 原样执行verified package和all-and-only RISC-V ELF | 15--17 |
| Q22.K | `target-model-packet-provenance` | `later` | owner-approved vendor package或公开规范 | 建立CRT/packet/MMIO provenance；缺失不阻塞functional CModel | 14、16、17 |
| Q22.P | `target-model-timing-calibration` | `later` | validated PMU/timing environment | 有RTL/vendor-cycle证据后校准LT/AT；此前不声明cycle accuracy | 16、17 |
| Q32.T | `compiler-transform-control` | `later` | 明确external control-plane consumer | 复用current rewrite/conversion；Transform IR不保存physical search state | 01、05--08、10、16、18 |
| Q38.W | `multi-worker-production-promotion` | `later` | typed ABI和host/model资格闭合且有独立命令链收益假设 | matched board A/B证明非零worker相对worker0的明确收益 | 08、11、14--17 |
| Q3.6 | `crt-writeback-scalar` | `later` | 明确Count predicate和wrapper/target/model evidence | typed instruction、effect/completion、ABI/CRT、model和必要readback闭合 | 11、14--17 |

不在当前DAG中的distributed/executable dialect、MPMD rank class、跨卡coherent variant、WCRE/global registry、
capability lease、跨model migration、shared-weight cache、segmented MoE及70B/100GB stress，恢复时必须先建立编号设计和完成门禁。

## 导航

- 编号设计和archive索引：`tasks/README.md`。
- Q52/Q53 current plan：`tasks/plans/physical-dataflow-synthesis.md`；详细历史：
  `tasks/archive/physical-dataflow-synthesis-working-history.md`。
- 已完成边界：`tasks/archive/completed-task-index.md`。
- 硬件校准事实：`docs/tx81-compiler-hardware-calibration.md`。
