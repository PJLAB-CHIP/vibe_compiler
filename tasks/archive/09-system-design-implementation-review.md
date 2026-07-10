# Wafer Compiler 系统设计与实现审计

日期：2026-07-10

性质：本文件是一次归档审计记录，不是第二份总体设计文档，也不新增 compiler stage、IR、ABI 或
runtime 合同。后续采纳的结论必须回写相应编号设计文档，再进入实现。

## 1. 审计目标和边界

本轮不以“让一个静态小模型样例跑通”为总体目标，而从未来复杂大模型负载反推 compiler/runtime
需要长期成立的边界。最低目标负载包括：

- dense Transformer 的 prefill 和 token-by-token decode；
- dynamic batch、dynamic sequence、bounded dynamism 和多版本 specialization；
- persistent / paged KV cache、resident weights、跨 invocation 可变状态；
- TP、PP、EP、DP 及其组合，包含多 mesh、MPMD、micro-batch 和 rank-specific execution；
- MoE 路由、ragged token dispatch、all-to-all / all-to-all-v 类通信；
- INT8 / FP8 / mixed precision、量化参数和 target-specific kernel variant；
- compute、DMA、DTE、host/device completion 的异步并行和跨层资源 lifetime；
- compiler-generated artifact 到 package、runtime、device code 和 board 的完整证据链。

最小闭环在本报告中只是一条验证切片：它可以证明某条链路存在，不能定义长期架构，也不能覆盖
复杂负载所需的状态、分布、变体和资源协议。

### Pipeline position

- Upstream artifact / IR: 当前编号设计文档、StableHLO/Shardy program、Wafer 各层 IR、target LLVM、
  device object、package metadata、runtime adapter 和现有测试证据。
- Current stage responsibility: 从总体产品目标反推语义所有权和 pipeline 边界，核对设计、实现、
  verifier、lowering、ABI、runtime 和验证是否一致。
- Output artifact / IR: 风险分级、长期能力压力矩阵、建议的目标边界、整改顺序和可验证完成门槛；
  不产出新的 compiler IR。
- Downstream consumer: `tasks/01` 至 `tasks/16` 的后续设计修订、`tasks/progress.md` 的任务排序，
  以及实现和 integration gate。
- User-level driver / named pipeline: 审计 `wafer-opt` program/target named pipeline、program-directory
  helper、package exporter、device linker 和 `wafer-run` 所组成的用户链路。
- Explicit non-goals: 本轮不直接修复代码，不把审计建议当成已收敛设计，不以单个 case 的 shape、
  rank 数、kernel 或 packet 作为通用协议。
- Completion gate: 关键判断有代码/文档行号或本轮新鲜命令证据；覆盖复杂大模型压力项；区分已确认
  缺陷、架构缺口和推断风险；给出可执行且可验证的整改顺序。

## 2. 执行结论

**总体判断：现有工程已经形成有价值的上半段 compiler 分层，但尚未形成语义正确、单一事实源、
可部署且能扩展到复杂大模型的 executable pipeline。当前状态不应继续以增加 CRT op surface 为最高
优先级。**

具体结论如下：

1. StableHLO/Shardy、topology/mesh、tensor/tile/instruction 分层方向基本合理，值得保留。
2. target LLVM lowering 已存在可复现的静默错误：结构化控制流被展开成无条件、单次调用序列。
   这使 Q0 原“done”结论失效，属于 stop-ship correctness defect。
3. rank identity、instruction geometry、address range、integer narrowing 和 package argument binding
   还没有形成端到端可验证合同；其中多项已能被非法 IR 或 metadata 绕过。
4. candidate selection、target lowering、package export 和 runtime 之间没有唯一的 committed executable
   artifact。多个输入文件和隐含 pass option 共同恢复语义，形成重复事实源。
5. 当前“HF”证据只覆盖一个静态、极小、自定义 decoder block 的 instruction 输出，未覆盖真实模型
   export、selected candidate、target LLVM、package、runtime、数值正确性、KV cache 或 decode。
6. dynamic/stateful execution、TP/PP/EP/DP 组合、MPMD、DTE physical resource assignment、异步事件和
   package variant selection 目前是架构缺口，不是简单补 op 就能解决的功能缺口。

因此，长期通用性不能通过让最低层 instruction op 接受任意动态形态获得。更稳健的方向是：

- 在 program / distributed / executable 层保留 symbolic shape、state、mesh、rank 和 variant 语义；
- 在进入 target instruction 前，以 guard、profile 和 target capability 选择或生成静态合法 variant；
- package 保留 variant 条件、typed resource ABI 和 target fingerprint；
- runtime 只做经过编译器证明的 variant selection、resource binding、launch 和 completion，不重新规划。

这使低层仍可保持可验证和接近硬件，同时避免把 batch、sequence、KV page、expert count 或 rank 数量
固化为单一 case。

## 3. 第一性原理评估准则

### 3.1 正确性先于 surface coverage

编译器的首要义务是保持源程序可观察语义。能发出 105 个 wrapper symbol，不等价于控制流、rank、
address、shape、count、completion 或 error semantics 正确。任何 unsupported 形态都应在改写前
fail closed，不能生成“看起来可链接”的错误程序。

### 3.2 每个长期事实只能有一个所有者

shape、rank、resource binding、offset、transport endpoint、completion 和 target capability 必须存在于
能验证它的 IR/artifact 中。pass option、文件名、参数名、独立 JSON 和 regex 解析都不能成为第二事实源。

### 3.3 分布和状态是程序语义，不是部署注释

rank identity 决定 local slice、collective participant 和通信 peer；KV cache 的 mutation、alias、page
lifetime 决定跨 invocation 结果。它们必须在 SSA、typed resource、region/control-flow 或 executable
contract 中显式存在，不能在 lowering/runtime 末端猜测。

### 3.4 “通用”不等于“所有层都动态”

target packet 和许多 accelerator kernel 本来就需要静态 shape/layout/count。通用性应由上层 bounded
dynamism、specialization policy、guard 和多 variant 组合提供；低层负责精确 legality，而不是接受无法
验证的 symbolic descriptor。

### 3.5 性能来自可验证的资源调度

未来模型的关键不是单 op 峰值，而是 resident weights、KV page、SPM reuse、DMA/DTE overlap、pipeline
micro-batch 和 collective contention。必须先有显式 resource/event/lifetime，再做 cost model 和 profile
校准；估算结果不能倒写成语义事实。

## 4. 建议的长期系统边界

下面是本轮用于压力测试当前实现的参考分层，不是已生效合同。

| 层 | 长期职责 | 当前判断 |
| --- | --- | --- |
| Verified program | StableHLO/structured ops、typed model ABI、shape constraints、immutable parameter、mutable persistent state | StableHLO 基础可保留；state/alias/dynamic ABI 缺失 |
| Target environment | target revision、memory/engine/capability、topology/mesh、supported dtype/layout/packet limits | topology 有基础；capability/profile 缺失 |
| Distributed program | multi-axis sharding、rank semantics、collective、MPMD stage、parameter shard | SPMD 局部存在；rank 被折叠，PP/EP/MPMD 缺失 |
| Local tensor program | local structured compute、local collective、state use-def、bounded dynamic semantics | normalization/group 有基础；嵌套 region 和 dynamic 支持不足 |
| Transactional planning | group/candidate 到 tile/layout/communication/instruction/SPM/DDR 的完整 legality，再原子 commit | selector 思路正确；主线可绕过且缺全局 executable 规划 |
| Executable composition | rank/entrypoint、persistent/transient resource、transport assignment、event、whole-executable lifetime、variant guard | 当前基本缺失 |
| Target lowering | 结构保持的 dialect conversion，最终到 LLVM/CRT/object；unsupported fail closed | 当前直线 op 可发 call；控制流和范围正确性不成立 |
| Thin package | typed ABI、payload/module、variant、mesh/capability、completion、fingerprint；不复制 schedule | 当前 schema/validator/exporter 重复且依赖猜测 |
| Runtime | 校验真实 capability、选 variant/submesh、allocate/import/bind/launch/complete/error/telemetry | 当前主要是 load/symbol gate，不是完整 launcher |

## 5. 风险总表

| ID | 严重度 | 结论 | 类型 |
| --- | --- | --- | --- |
| F-01 | P0 | target LLVM 静默破坏结构化控制流 | 已确认缺陷 |
| F-02 | P0 | rank identity 被常量化或藏在 pass option，分布式语义不成立 | 已确认缺陷/架构阻塞 |
| F-03 | P0 | instruction range、geometry、count 和整数宽度缺少闭合 verifier | 已确认缺陷 |
| F-04 | P0 | package workspace 参数可丢失，binding order 允许不完整/重复 | 已确认缺陷 |
| F-05 | P0 | 主线没有唯一 committed executable artifact | 架构阻塞 |
| F-06 | P1 | Direct DTE 只有 logical schedule，没有 production physical assignment/completion | 已确认缺口 |
| F-07 | P1 | issue-only 指令缺显式 completion，SPM reuse 依赖未进入 artifact 的硬件/runtime 假设 | 高风险合同缺口 |
| F-08 | P1 | dynamic、persistent state、KV cache 和 variant selection 没有端到端表示 | 架构缺口 |
| F-09 | P1 | 当前 deployment model 无法表达 TP/PP/EP/DP 组合和 MPMD | 架构缺口 |
| F-10 | P1 | target profile/capability/fingerprint 缺失，硬件事实散落为常量 | 架构缺口 |
| F-11 | P1 | package/runtime schema 重复、文本猜测 ABI，且 runtime 未真正 launch | 已确认缺陷/缺口 |
| F-12 | P1 | CRT symbol closure 没有证明 status、timeout 和 undefined-symbol policy | 已确认缺口 |
| F-13 | P1 | 当前 HF gate 不能作为真实大模型主线完成证明 | 证据缺口 |
| F-14 | P2 | nested region 可逃过 group 检查，逐元素 movement lowering 不可扩展 | 已确认缺陷/规模风险 |
| F-15 | P2 | 编号文档中的 stage 顺序、artifact 和 completion 叙事仍有冲突 | 设计一致性缺口 |
| F-16 | P2 | dependency/device toolchain 缺完整供应链和持续回归合同 | 工程化风险 |

## 6. P0 发现

### F-01 Target LLVM 静默破坏结构化控制流

**事实。** `lib/Wafer/Transforms/Target/LowerInstrToTargetLLVM.cpp:1057-1080` 对 function 内全部
`wafer.instr.*` 做递归 walk，把 call 统一追加到一个 LLVM entry block，随后删除原 function。这个算法
没有保留 `scf.if`、`scf.for`、多 block、branch 或 call graph 结构。`tasks/11-instruction-ir.md:66-68`、
`:88-119`、`:612` 和 `:648-649` 又明确允许 instruction IR 保留标准结构化控制流。

本轮定向复现把两个分支分别放入 RDMA/WDMA；lowering 后两个 call 都无条件存在。循环 body 同理只被
materialize 一次，不随 trip count 执行。这不是 coverage 缺口，而是 silent miscompile。

**影响。** decode loop、conditional expert route、tail handling、error path 和任何 future control-flow
都会得到错误设备程序。只要该 lowering 还能接收 nested/multiblock function，就不能宣称 target artifact
语义可信。

**整改门槛。**

1. 立即增加 preflight：在开始 mutation 前拒绝所有尚不能结构保持 lower 的 region、multiblock、call。
2. 正式改为结构化 dialect conversion：在原控制流位置改写 instruction op，再按标准 SCF -> CF ->
   LLVM/func lowering 处理容器；不得先 flatten。
3. integration tests 至少覆盖 false branch、不同 loop trip count、nested branch、function call 和失败前
   IR 不被部分改写。
4. full conversion 后 target module 不残留非法 dialect，且语义测试而非只做 FileCheck call 数量。

MLIR 官方 Dialect Conversion 提供 `ConversionTarget`、rewrite pattern、region signature conversion 和
full conversion 机制，适合把 legality 和结构保持放进同一转换合同：
https://mlir.llvm.org/docs/DialectConversion/ 。LLVM translation 本身也要求先形成合法 LLVM dialect：
https://mlir.llvm.org/docs/TargetLLVMIR/ 。

### F-02 Rank identity 在进入执行层前丢失

**事实。** `lib/Wafer/Conversion/StableHLOToLinalg/NormalizeStablehloCollectives.cpp:387-405` 无条件把
`stablehlo.partition_id` 和 `stablehlo.replica_id` 替换成常量 0。与此同时，
`lib/Wafer/Transforms/SPMD/XlaSpmdPartitionerMain.cpp:841-884` 生成单个 SPMD module，把
`num_partitions` 设为 logical rank
count，而 `replica_count` 固定为 1。残留 partition id 因此应当仍是运行 rank 的动态语义，除非 compiler
明确生成每 rank specialization。

`include/Wafer/Conversion/WaferGroupToTileRegion/WaferGroupToTileRegion.h:17-28` 等 API 又把
`currentLogicalRank` 默认设成 0；selector 调用路径可依赖这个默认值。rank 既不稳定存在于 artifact，
也不进入 package cache identity。

**影响。** TP/EP/DP 的 local slice、rank mask、collective peer 和 parameter shard 都可能按 rank 0 编译。
未来即使 DTE packet 正确，也会在错误数据或错误 peer 上执行。

**整改门槛。** 编号设计必须二选一并端到端验证：

- rank-parametric executable：rank/replica identity 是显式 SSA/ABI/resource，lowering 和 runtime bind；或
- per-rank executable variants：compiler 对每个 rank materialize、verify、package，并把 rank、mesh、
  parameter shard 和 artifact digest 纳入 variant identity。

两种模式可以并存，但 pass-only hidden option、默认 0 和 normalization 常量折叠必须退出主线。

### F-03 Instruction verifier 没有证明物理访问和 packet geometry

**事实。**

- `lib/Wafer/IR/Instr/InstructionOps.cpp:756-800` 的 RDMA/WDMA verifier 检查 memory space 和
  descriptor 基本合法性，但不证明 source/destination physical range；同文件 gather/scatter 在
  `:826-849` 已有范围检查，说明统一做法可行。
- 同文件 descriptor 基本检查 `:194-220` 只要求正数和 `inner_bytes <= byte_count`，没有证明
  `byte_count == inner_bytes * product(iterations)`。
- `convert` verifier `:1110-1150` 没有要求 source/destination element count 一致；target lowering
  `lib/Wafer/Transforms/Target/LowerInstrToTargetLLVM.cpp:760-779` 直接从 destination 生成 count。定向复现
  接受 1 element 到 1024
  elements，并发出 count 1024。
- conv/pool/unpool/peripheral 的 shape attrs 在 `lib/Wafer/IR/Instr/InstructionOps.cpp:1269-1306`、
  `:1341-1379`、
  `:1410-1443`、`:1667-1744` 没有和 memref extents 形成完整关系。
- `lib/Wafer/IR/Common/OpVerifierUtils.cpp:413-482` 接受多种 batched GEMM dimension mapping，但 target
  lowering `lib/Wafer/Transforms/Target/LowerInstrToTargetLLVM.cpp:782-805` 只转发地址、M/K/N、batch
  count 和 format，完全不传
  dimension mapping；非 canonical mapping 会被按 canonical GEMM 执行。
- `lib/Wafer/Transforms/Target/LowerInstrToTargetLLVM.cpp:346-363` 把 i64 attrs 直接截断为 i32；
  4294967297 的定向输入被发成 1。
  `runtime/wafer_crt/src/wafer_tx81_crt.c:22-25` 又把 shape 从 uint32 收窄到 uint16，没有上游 bound
  gate。

本轮还复现了 1-element SPM destination 配 16-byte RDMA 仍通过 verifier。

**影响。** 这些路径会形成 OOB、错误 count、错误 packet 或大 shape 截断，直接影响数值正确性和设备
内存安全。复杂模型会显著放大边界 shape、large tensor 和 layout descriptor 的触发概率。

**整改门槛。** 建立一份共享的 typed geometry/descriptor 校验：从 memref type、layout、offset 和
element type 推导 physical interval，证明 byte/count/iteration 等式和所有 narrowing 上界。重复字段应
尽量派生；必须保留时由 verifier 证明相等。ODS/verifier、lowering 和 CRT 不能各自实现不同规则。

### F-04 Package workspace ABI 可以合法地丢失参数

**事实。** `tools/wafer_export_package_metadata.py:668-700` 会根据 LLVM i64 参数数量补出
`workspace0`，但 `:741-751` 的 `binding_order` 只由 model inputs + outputs 组成。Python runtime
`tools/wafer_runtime_adapter.py:263-273` 和 C++ runtime `lib/Wafer/Runtime/HostRuntime.cpp:451-467`
都只按 `binding_order` 构造实参。validator `tools/wafer_package_metadata.py:700-709` 只检查引用名字存在，
不检查唯一性、完整性和与 entrypoint signature 的一一对应。

本轮用带额外 i64 workspace 参数的 LLVM IR 导出 metadata：exporter 正确生成 272-byte workspace，
但 entrypoint binding 只有 `lhs,rhs,out`，validator 仍成功。重复且不完整的 `lhs,lhs` 同样能通过。

**影响。** 一旦 Q4 进入真实 package 主线，设备 entrypoint 会收到少参数或错位地址，风险是立即崩溃或
静默写错内存。

**整改门槛。** 在修复前拒绝所有 workspace-bearing package。正式 ABI 必须由 compiler structured
artifact 导出 exact ordered typed arguments，validator 证明参数数量、顺序、唯一性、resource kind、
access、size/alignment 和 entrypoint signature 完全一致；runtime 不再猜测。

### F-05 缺少唯一 committed executable artifact

**事实。** `tasks/01-architecture.md:29-51` 描述 candidate accepted/rejected/split 后 commit，但当前
target pipeline 在 `lib/Wafer/Pipelines/Pipelines.cpp:92-109` 可走 direct DDR path；selected pipeline
停止在 instruction 一侧。`tools/wafer-opt/wafer-opt.cpp:449-455`、`:617-630` 的 program driver 也只把
program 接到 group，后续链路由不同命令和文件拼接。

package exporter 同时读取 model interface JSON、instruction text、LLVM IR 和 module path，再从文本恢复
entrypoint、memory 和 ABI。这里没有一个 artifact 同时拥有：rank/entrypoint、accepted layout/offset、
transport assignment、persistent/transient resource、event/completion、target profile 和 variants。

**影响。** planner 可以绕过，package 可以和实际 object 不一致，runtime 也无法证明它绑定的是哪个
accepted candidate。未来加入 KV cache、resident weight、PP stage 或多 variant 后，重复事实源会呈
组合爆炸。

**整改门槛。** 所有路径必须经过一次事务式 commit。full-shape direct lowering 可以保留，但只能作为
mandatory candidate policy，而不是平行主线。commit 输出一个可验证 executable composition artifact；
target lowering、package export 和 runtime 只能消费它或它的确定性派生产物。

同一原则也适用于 layout：`tasks/08-layout-materialization.md:74-108` 的早期 `GroupLayoutPlan` 只能提供
domain/demand，不能和 `:657-696` 在 target-abstract op 上运行的正式 layout planner 各自形成 accepted
assignment。最终 assignment 必须只有一份，并属于 committed artifact。

`tasks/06-group.md:61-66` 还把 scheduled-form `wafer.group` 列为正式阶段，但 `:790-806` 的当前 driver
从 candidate evaluation 直接 commit tile/instruction artifact，`:1202-1205` 又尚未决定 group 是否长期
保留。后续必须明确 scheduled group 是拥有 accepted schedule 的真实 IR，还是只保留 logical group 并
删除这层设计草图；不能让它和 committed tile/instruction 同时拥有 schedule。

commit 的作用域也不能停在单 tile-region：`tasks/09-spm-memory-planning.md:457-461` 允许不同
tile-region 复用同一 SPM offset，`tasks/12-ddr-memory-planning.md:203-204` 只规划单 function/candidate，
跨 group lifetime 尚未定义。PP、micro-batch 和跨 region overlap 需要 executable composition 证明 region
之间的 drain/event、并发关系和 persistent/transient resource lifetime。

## 7. P1/P2 发现

### F-06 Direct DTE 尚未形成 production transport contract

`include/Wafer/IR/Instr/DTEOps.td:14-19` 仍把 peer 留在 logical 层，physical endpoint 在后续派生；
`lib/Wafer/Transforms/Target/LowerInstrToTargetLLVM.cpp:981-1011` 只是转发 peer，并删除 topology。
`tasks/14-target-llvm-golden-packet.md:555-573` 也明确列出 local/remote tile、FSM/channel/node、receive
buffer 和 completion 尚缺。DTE symbols 没有进入当前 105-symbol production closure。

需要在 target LLVM 前形成 accepted transport/resource assignment，显式包含 endpoint、channel/FSM、
remote buffer、barrier/event 和 completion；不能由 CRT 或 runtime 根据 peer id 再规划。

package 必须携带该 projection，或携带能确定性重建 projection 的 target/topology artifact identity 和
完整验证条件。当前 `tasks/15-launch-runtime-package.md:99-105` 不序列化 endpoint view，而 exporter 输入
`:380-389` 也没有 topology；target lowering 又会删除 topology。没有这项闭合，多 rank package 无法证明
它对应的 physical deployment。

### F-07 Async completion 和 SPM lifetime 不闭合

多数 compute/movement instruction 只表达 issue side effect，没有 token；DTE 有 token，但 local fence
粒度粗。`lib/Wafer/Transforms/SPM/PlanSPMMemory.cpp:448-498` 只在已知 pending write/fence 上延长
lifetime，`:566-594` 的 region end 没有统一处理未完成 local write。定向 case 中两个 issue-only fill
可以重用同一 SPM offset，而 compiler artifact 本身没有第一个操作已完成的证据。

这项证据确认的是**合同未闭合**，不能单凭 offset 复用断言板端一定误执行。硬件文档
`docs/wafer-hardware-instruction-set-and-programming-model.md:588-590`、`:615-616` 说明：当 runtime 确认
`serial_mode=0` 时，同 worker 的 CT/NE/RDMA/WDMA/TDMA 可由 packet address range 和 busytable 检测依赖；
但它不覆盖 Direct DTE、Stream、Kcore 直接 SPM 访问或多 tile barrier。当前 compiler/package 没有证明
worker mode、packet range 与 busytable scope 满足这项假设。

短期必须二选一：把所需 worker/busytable mode 纳入 target/runtime capability 并用 packet/board gate
证明，或者要求 terminal drain/fence、把未完成 issue lifetime 延长并保守禁止复用。长期应以每 engine
event/token 表达 compute、DMA、DTE completion，使 reuse、double buffering 和 overlap 都由 SSA/event
或显式 target capability 证明。MLIR Async dialect 可作为机制参考，但需先证明它与设备 engine/event
语义匹配：
https://mlir.llvm.org/docs/Dialects/AsyncDialect/ 。

### F-08 Dynamic/state/KV/variant 只在入口局部存在

frontend 在 `lib/Wafer/Frontend/Program.cpp:127-174` 可接受 bounded dynamic，但 program-directory
signature 在 `:621-629` 要求 static，parameter shard 在 `:952-956` 同样要求 static。selector、SPM 和
target lowering 继续要求静态；target function ABI
`lib/Wafer/Transforms/Target/LowerInstrToTargetLLVM.cpp:497-563` 基本只有每个 DDR memref 的 i64 base
address。package shape validator `tools/wafer_package_metadata.py:400-412` 只接受正整数。

因此当前没有 actual size/bound、shape guard、variant selector、KV page table、session lifetime、mutable
alias 或跨 invocation state。package 还把 inputs 视为 read-only，无法自然表达 in-place KV update。

resident/packed weights 也未形成稳定 resource contract：`tasks/08-layout-materialization.md:448-452`、
`:508-511` 一方面不定义 storage encoding attr，另一方面允许替换 `ConstantLike` backing resource；
`:547-570` 的示例仍可能再次 materialize 同一 DDR value。未来需要 immutable content identity、显式
storage encoding、rank/variant shard identity 和 semantic-to-storage verifier，不能只靠 layout marker。

建议在 verified program/executable 层引入 symbolic bounds、typed persistent resources、alias/mutation 和
specialization guards；target instruction variant 仍保持静态。StableHLO 的 dynamism 模型可作为上游语义
参考：https://openxla.org/stablehlo/dynamism 。

### F-09 Deployment model 折叠了未来并行维度

`tools/wafer-opt/wafer-opt.cpp:246-302` 把 logical ranks 限在 1..16，并硬编码 4x4 endpoint mapping；
XLA helper
固定 replica count 为 1。当前只有 total rank count，没有 TP/PP/EP/DP axis identity、stage graph、
micro-batch scheduler、multi-mesh assignment 或 rank-specific artifact composition。MoE 还需要 ragged
dispatch/all-to-all-v 类语义，不能用静态 equal split 的 all-to-all 覆盖。

`wafer.topology` 已能表达 named grids/axes，是可保留基础。后续应先评估 Shardy 的现有 sharding/MPMD
表示，再决定 Wafer 私有 executable extension，避免重复定义上游已有概念：

- https://openxla.org/shardy/sharding_representation
- https://openxla.org/shardy/mpmd/mpmd_dialect
- https://openxla.org/shardy/mpmd/mpmd_optimize_passes

### F-10 Target capability 仍是散落常量

`wafer.target` 目前主要是 target 名；topology 描述 grid/interconnect/unavailable endpoint，但没有 hardware
revision、memory capacity/alignment、engine count、dtype/layout、packet limit 或 errata。
`include/Wafer/Support/TargetPolicy.h:22-38` 仍硬编码 SPM 和 timing 等默认值，部分 pass option 又复制
同类事实。

需要一个由部署环境选择、由 compiler 验证并进入 artifact fingerprint 的 target environment/profile。
在新增私有 capability IR 前，应评估 MLIR DLTI target system/device spec 能承载哪些通用数据：
https://mlir.llvm.org/docs/Dialects/DLTIDialect/ 和 https://mlir.llvm.org/docs/DataLayout/ 。

### F-11 Package/runtime 不是同一合同，也还不是 launcher

Python schema/validator 和 C++ loader接受的字段并不完全相同；C++ unit fixture 可省略 Python 侧要求的
runtime requirements，并接受空 binding order。exporter 在 `tools/wafer_export_package_metadata.py:93-269`、
`:406-575`、`:630-649` 用 regex/text window 解析 MLIR/LLVM，positive fixture 的 function return 甚至
与 target ABI 的 void 形态不一致。

package 还复制完整 instruction list，构成 shadow schedule，随大模型线性膨胀。module validation 主要
检查字符串/`.so` 后缀，没有 digest、ELF target、required exports 或 compiler/target fingerprint。
`tools/wafer-run/wafer-run.cpp:170-192` 目前只 dlopen、查 symbol、打印
`board_launch_gate: not executed`；host
runtime 没有形成 typed allocate/import/bind/launch/complete 调用链。

需要 compiler-side structured manifest、单一生成式 schema/parser/validator、thin package 和实际 artifact
digest。completion 也应是有 scope、timeout、status 和 rank aggregation 的组合关系，而不是一个可由
`kcore_local_drain` 单独满足的字符串。

### F-12 CRT closure 只证明 symbol surface

`byte_count` 在当前 instruction contract 中可以是 logical payload/resource-effect 字段，packet 则由
`inner_bytes` 和 iterations 形成；CRT 不直接消费它本身不必然是错误。真正缺口是 F-03 中 verifier
没有证明二者等价、element-size 整除和 narrowing bounds，导致
`runtime/wafer_crt/src/wafer_tx81_crt.c:63-65` 的转换可静默截断。

此外，`TsmExecute` / `TsmWaitfinish` status 在 CRT `:78-82`、`:664-670`、`:811` 等路径被丢弃，也没有
统一 timeout/error mapping。当前 device link只拒绝 undefined `wafer_tx81_*`，尚无正式非 Wafer
undefined-symbol allowlist。

本轮真实 positive device link 成功生成 RISC-V ELF shared object，但该事实只证明 toolchain/symbol
closure。后续 gate 应增加 compiler-generated descriptor equality/range、status/error、bounded wait、
正式 undefined-symbol allowlist、ELF ABI 和 exports 检查。

### F-13 当前 HF gate 不是大模型完成证明

现有 HF case 是 batch 1、sequence 4、hidden 16、16-rank TP 的自定义 decoder block，结束于 instruction
IR。capture script 手工拼 block，且在 `test/Tools/Inputs/wafer_pytorch_xla_capture.py:250-254` 拒绝
GQA；没有真实 Transformers model、causal serving、KV state、decode、selected commit、target LLVM、
package、runtime 或 numeric gate。

该 case可保留为 importer/SPMD regression，但命名和完成结论必须准确。主线证据应逐步增加真实或忠实
exported model 的 prefill/decode、shared KV、GQA、长 sequence、TP+PP/EP、MoE 和量化 case。

### F-14 Region 检查和 movement lowering 不可扩展

`lib/Wafer/IR/Tensor/GroupOps.cpp:124-148` 与
`lib/Wafer/Analysis/Group/TilingDemandAnalysis.cpp:362-388` 偏向直接 child，nested `scf` 可隐藏 SPM
alloc/instruction，并使 demand 只看到容器。group verifier 和
analysis 必须递归处理允许的 tensor-level subset，或在进入 group 前规范化/拒绝 region。

`lib/Wafer/Conversion/WaferTileRegionToInstr/WaferTileRegionToInstr.cpp:672-755`、`:798-879` 按 logical
element 枚举 movement segment。对长 sequence、KV transpose 和大 tensor 会造成 O(elements) compile time/IR
膨胀。应优先用 affine/indexing relation 和 symbolic descriptor decomposition，仅在必要边界拆 segment。

### F-15 编号文档仍有相互冲突的 pipeline/completion 叙事

`tasks/04-topology-execution-mesh.md:39-73` 要求 topology/mesh 在 SPMD 前 materialize，但同一 contract 把
already-partitioned program 列入 upstream，`:76-103` 又以 partitioned program 作为输入。这里应拆清
pre-SPMD topology/mesh selection 和 post-SPMD launch projection，而不是让一个 stage 同时位于 SPMD
两侧。

审计输入中的 `tasks/16-verification-plan.md` 曾把 direct instruction 称为 committed，并把 SPM/DDR
输入写成 tile region；本批已经同步为 direct memory-planned artifact 和 instruction-level planner input，
同时重新打开 structured-control target gate。仍未收口的是
`tasks/07-tile-region.md:125-141`、`tasks/08-layout-materialization.md:129-147` 把局部 FileCheck/dump 放在
completion gate 的中心，未证明输出被下游主线直接消费。

这些冲突会让队列和 downstream gate 继承错误完成假设。已修正的完成叙事不能回退；其余编号文档在
进入对应实现前仍应统一 stage 顺序、artifact 名义和 completion gate。direct-lowered artifact 与
selected committed artifact必须分开命名，局部 dump 只能作补充覆盖。

### F-16 构建、依赖和 device toolchain 尚缺生产级可复现合同

dependency commit pins 和 `tools/check_deps.py` 是良好基础，但顶层 Python requirements没有 hash lock；
`tools/bootstrap_deps.py:33-40` 直接 pip install，`:70-111` 对预编译包只校验 content length。785 MiB
`third_party/tx8_deps` 本轮未发现 repo-level license/SBOM/digest manifest。

`tools/wafer_device_link.py:29-38` 最终可回退到 PATH 中的 `clang++`；`:172-208` 默认删除 input/CRT 的
`.riscv.attributes`，因此版本/ISA/ABI 不匹配可能被延后到 final ELF 或 board。根 `CMakeLists.txt:35-39`
的 `check-wafer` 依赖 `WaferUnitTests` target，但只保证构建，不执行；本轮也未发现项目级 CI 配置。

整改门槛是：hash-locked Python/dependency payload、TX8 dependency license/SBOM/digest、匹配 compiler
stack 的 clang profile、final ELF attributes/ABI/exports 检查，以及在干净环境实际运行 lit、C++ unit、
Python validator、device positive/negative link 的项目 CI。删除 attributes 若确属 toolchain 兼容要求，
也必须有 final artifact 等价验证，而不是单纯丢弃证据。

## 8. 复杂负载压力矩阵

| 负载/能力 | 当前可用基础 | 关键缺口 | 长期完成证据 |
| --- | --- | --- | --- |
| Dynamic batch/sequence | frontend 局部 bounded dynamic | static program-dir/planner/package；无 guard/variant | 多 shape 共享上层 program，选择已验证 static variants，越界拒绝 |
| Prefill + decode | 静态 block 可到 instruction | 无 decode loop、state、target/package/runtime/numeric | 同一模型 prefill/decode，共享 typed KV，逐 token 数值和 lifetime 正确 |
| Paged KV cache | DDR/SPM resource 基础 | 无 persistent mutable resource、page table、session/alias | page allocate/bind/update/free，跨 invocation 和并发 session 验证 |
| Resident weights | parameter shard 和 DDR 基础 | 无 executable residency/lifetime/import/cache key | 权重一次加载、多 launch 复用，variant/rank/digest 一致 |
| TP/DP | Shardy/SPMD/topology 基础 | rank=0、replica=1、无 rank artifact identity | 不同 rank local slice/collective/parameter shard 可观测正确 |
| PP/MPMD | topology axis 可扩展 | 无 stage graph、multi-mesh、micro-batch/event | 多 stage rank programs、1F1B/其它 schedule 和跨 stage completion |
| MoE/EP | collective op 基础 | 无 ragged route/all-to-all-v/expert residency | 动态 token counts、capacity/drop policy、expert shard 和返回路由正确 |
| Quant/mixed precision | dtype/layout/peripheral surface | capability/scale/zero-point/accumulation/variant 合同不足 | typed quant params、target legality、reference numeric 和 fallback variant |
| Async overlap | issue op、fence、DTE token 局部基础 | completion/lifetime/engine resource 不闭合 | event happens-before、SPM reuse、double buffer、timeout/error 可验证 |
| Multi-card deployment | topology/DTE logical schedule | endpoint/channel/FSM/remote buffer/board completion 缺失 | compiler assignment 到实际 board，rank aggregation 和失败传播 |

## 9. 应保留与应重构的部分

### 建议保留

- StableHLO 作为上游可移植语义，Shardy 负责 sharding propagation/SPMD 的总体方向。
- topology/mesh 使用显式 grid、axis 和 endpoint projection，而不是从名字推断。
- tensor collective、tile buffer collective、DTE schedule 的分层意图。
- layout materialization 与 memory planning 分离，accepted offset 只在合法 candidate 后提交的原则。
- candidate failure 不污染主 IR、analysis 可重算、target wrapper surface fail closed 的工程纪律。
- lit negative verifier、C++ unit、Python validator 和 toolchain scripts 的多层验证框架。

### 必须重构

- 任何 flatten region 的 target lowering、rank 默认 0 和 hidden pass-only semantic context。
- direct path 与 selected path 并列成为事实源；应统一到 mandatory commit。
- package 从四份文本/文件猜 ABI、复制 instruction schedule 的方式。
- 仅靠 issue order 和 coarse fence 推断 completion/lifetime 的方式。
- 用 target 常量、provider 名或文件后缀代替 capability/artifact verification 的方式。

## 10. 建议路线和优先级

### A. Stop-ship correctness closure

1. 重新打开 Q0：target lowering 对 SCF/CF/call 结构保持，未支持形态在 mutation 前拒绝。
2. 恢复 rank identity，明确 rank-parametric 与 per-rank variant 合同。
3. 闭合 instruction physical range、geometry/count equality 和所有 integer narrowing verifier。
4. 修复 issue/completion 与 SPM lifetime；在长期 event 模型前先 fail closed。
5. 阻断 workspace package，修复 exact binding order/typed entrypoint ABI。

在这些项目完成前，不应把新增 `count`/VS/GELU 等 CRT surface 作为唯一 `next`。新增 surface 会扩大
错误 lowering 和不可信 ABI 的覆盖面，不提高主线可信度。

### B. 收敛长期 executable contract

1. 在相关编号设计文档中定义唯一 committed executable composition artifact。
2. 定义 target environment/capability/profile 和 artifact fingerprint。
3. 定义 rank/MPMD/transport physical assignment 和 event/completion ownership。
4. 定义 bounded dynamic + specialization variant、persistent state/KV、resident parameter resource。
5. package 改为 thin structured manifest；runtime 只做验证、选择、绑定和执行。

这一阶段每项都必须先回答 artifact 上游、owner、verifier、lowering、runtime consumer 和 failure mode，
不能一次新增一个不可验证的 opaque attr bag。

### C. 建立真实纵向证据

1. 一个真实或忠实 exported static prefill case：program -> SPMD -> selected commit -> target LLVM ->
   actual device link -> self-contained package -> no-card loader。
2. 多 rank DTE case在真实 board 证明 endpoint、buffer、completion、numeric 和 failure propagation。
3. prefill/decode 共用 KV，覆盖 dynamic bounds 和至少两个 variants。
4. 增加 GQA、MoE/EP、PP micro-batch、quant 和 overlap；每项复用同一 pipeline，不另造 shortcut。

### D. 性能和 extended CRT surface

只有在真实 vertical case需要且 A-C 相应 legality 已成立后，才按 workload demand增加 count、VS、GELU、
MXFP 等 surface，并用 board/profile 校准 cost/overlap。op 数量不是架构成熟度指标。

## 11. 本轮验证证据

本轮在当前 checkout 得到以下新鲜结果：

| 命令/检查 | 结果 | 能证明什么 | 不能证明什么 |
| --- | --- | --- | --- |
| `cmake --build build/wafer-dev --target check-wafer -- -j128` | 205 lit tests：204 passed，1 unsupported | 当前 lit 主配置回归通过 | `check-wafer` 只构建 unit test target，不执行它；不证明 board/numeric |
| `ctest --test-dir build/wafer-dev --output-on-failure` | 3/3 passed | lit、Python runtime adapter、C++ unit 都实际执行 | 测试输入覆盖有限 |
| `lit -sv --show-unsupported build/wafer-dev/test` | 1 个 intentional StableHLO-disabled config test unsupported | 主配置不是大面积 skip | 不证明外部 board/firmware |
| `python3 tools/check_deps.py` | passed | dependency pins 自洽 | 不证明上游新功能已集成 |
| `python3 tools/check_ir_organization.py --root .` | passed | 当前源码组织规则通过 | 不证明 IR 语义正确 |
| target CRT conformance/symbol scripts | 105 production symbols，passed | repo-local surface/定义闭合 | 不证明 packet/status/numeric |
| `wafer_device_link.py` positive link | 生成 ELF64 RISC-V shared object | CRT compile/link 真实可执行 | unresolved non-Wafer symbols尚无正式 allowlist；不证明 board load |
| 定向 SCF lowering case | 两分支均无条件发 call | 确认 F-01 | 尚未验证修复 |
| 定向 descriptor/convert/narrowing cases | 非法范围/数量/大整数可通过并错误发 call | 确认 F-03 | 尚未覆盖每个 op family |
| 定向 package workspace/binding cases | workspace 丢失、重复 binding 可通过 | 确认 F-04 | 尚未验证修复 |

unsupported test 是 `Tools/wafer-compile-stablehlo-disabled.test`，用于 alternate disabled configuration，
不是当前 mainline feature 被静默跳过。实际 board、firmware、真实多卡 numeric、性能和长稳测试环境本轮
不可用，因此报告没有把它们标为通过。

## 12. 最终评级

| 维度 | 评级 | 判断 |
| --- | --- | --- |
| 上游 IR 分层 | B- | 方向合理，dynamic/state 和 nested region contract 需补齐 |
| 分布式语义 | D | rank identity 不成立，MPMD/PP/EP deployment 缺失 |
| Planning/commit | C- | candidate 思路可取，但主线可绕过且缺 executable 全局事务 |
| Instruction legality | D | 多项 range/geometry/width 缺口可产生错误 packet |
| Target lowering | F | 存在已复现 silent control-flow miscompile |
| Package/runtime | D | ABI 可错位，schema 重复，尚未执行 board launch |
| 验证工程 | B- | 基础设施和局部覆盖较好，但纵向真实性不足 |
| 复杂大模型 readiness | D | 需要先完成 correctness 和 executable/state/distribution 架构收敛 |

评级不是对已有投入的否定，而是表示当前 artifact 是否足以承担长期产品合同。最重要的正向资产是分层
意识、显式 topology 和逐层 verifier 框架；最重要的风险是 lower half 把结构化程序变成若干独立文件、
隐式默认值和文本推断。下一阶段应先把语义正确性和唯一 executable 合同立住，再扩能力面。
