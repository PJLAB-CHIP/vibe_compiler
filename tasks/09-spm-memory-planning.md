# Wafer SPM Memory Planning Design

状态：当前合同覆盖instruction IR上的SPM lifetime/range planning；async issue的全部
read/write resource必须活到可信completion。当前实现已闭合non-nested single-`wafer.tile.region`内的path-aware
local/DTE completion proof；跨多个tile-region的whole-entry planning未实现，也不是当前isolation合同的
correctness前置，只保留为后续优化。实现状态以
`tasks/progress.md`为准。accepted fact 为
offset-only `wafer.spm.offset`，size / bank span / alignment 由 memref type、layout 和 target policy 重算。

本文定义 Wafer SPM bufferization、tile-local allocation 和 storage verification。它服务于
`wafer.group` candidate/template planning 的合法性搜索，也负责在完整 static rank entry 中逐个验证
`wafer.tile.region` scope，把其 tile-local value 落到可验证的 memory space、range 和 completion
effect。`wafer.tile.region` 是 `IsolatedFromAbove`，且 verifier 禁止 SPM buffer 作为 region 输入或结果；
SPM planner还在写入任何offset前拒绝nested `wafer.tile.region`，避免inner/outer scope分别从同一physical
arena base规划而覆盖仍存活的outer buffer。因此当前 correctness 边界是每个non-nested region独立规划、
每条region exit完成，再由variant gate汇总全部region，而不是跨region lifetime/reuse。
SPM memory planning 的 instruction-level 输入合同由
`tasks/11-instruction-ir.md` 定义；本文只消费该层暴露的
Wafer-tagged memref / `wafer.instr.*` / effects，不重复定义 instruction op。

本文只负责 `#wafer.memory<spm, *>` 的 tile-local allocation：

- 消费 instruction-level `wafer.instr.*` IR 和 unplaced
  `memref<..., #wafer.memory<spm, layout>>`，并从 memref use-def、effects、queue 和 async policy
  构造 allocation input。
- 对 instruction-level IR 做 SPM memory planning、range/end-address/alignment/bank-span verification 和 failure
  feedback。
- 为Q16 commit前的typed rank-record validation提供accepted offset fact；range/lifetime/alias在candidate IR中
  重算并参与typed C++ bundle materialization，不作为独立attr。post-commit target/package/runtime不从SPM IR重新
  恢复resource semantics。

本文不分配 DDR，不选择 physical layout，不决定 group boundary，不选择 compute/communication
instruction selection，也不生成 runtime package。DDR source/destination range 和 bandwidth 可以作为
legality 或 cost input；DDR declared arena/placement-domain resource 的主设计见
`tasks/12-ddr-memory-planning.md`。SPM allocation 的失败 trace、搜索顺序和
rejected/candidate offset 都是 analysis，不写进长期 IR。

## 1. 核心结论

SPM planning 不能只做 byte-size estimate。候选 tile plan 是否合法，必须跑与下游一致的：

```text
layout materialization
  -> candidate DDR tile-view materialization
  -> instruction legalization / selection
  -> instruction-level IR with unplaced Wafer-tagged memref values
  -> storage requirement collection
  -> liveness/effect analysis
  -> per-isolated-tile-region SPM memory planning + shared physical geometry/range verification
  -> candidate target-ABI address/range/narrowing preflight before commit
  -> target-codegen derivation from the same committed facts; runtime only binds verified manifest ABI slots
```

如果没有合法 allocation，不能生成一个等待下游修复的 scheduled group，也不能保留已通过的
per-group placement。planner 应丢弃整个 variant clone，再回到 tile shape、internal split、layout
assignment、output coverage 或 group boundary 继续搜索。

### 1.1 Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  whole-variant evaluation clone 中所有 static rank entries 的完整 instruction-level
  `wafer.instr.*` structured program，已经完成 candidate DDR tile-view materialization 和 instruction
  legalization；每个non-nested `wafer.tile.region` 是不允许 SPM value 跨边界的独立 storage scope，包含 actual DDR
  tile views、unplaced `memref<..., #wafer.memory<spm, layout>>` values，以及 accepted layout
  assignment、materialization cut、effect/order 和 target SPM policy。
- Current stage responsibility:
  先拒绝nested `wafer.tile.region`，再对每个non-nested isolated scope为
  `#wafer.memory<spm, layout>` memref做SPM memory planning，
  计算 offset/end/bank span、alignment、lifetime/reuse、must-alias/must-not-alias、reserved range
  和 range-end verification；用显式generic async wait、DTE exact token/wait、local fence和terminal drain证明
  该region所有issue completion，并在variant-set gate汇总所有region/rank的通过结果。
- Output artifact / IR:
  仅存在于 complete passing variant clone 中的 same instruction-level IR with offset-only
  `wafer.spm.offset` planning facts on SPM memref definitions，或结构化 allocation failure reason；
  Q16 commit前从该fact、memref use-def、arena/endpoint facts和view relation直接重算并补全resource range，
  验证后写入typed C++ `RankExecutable` record。target LLVM只从committed instruction IR和该record中
  IR-derived entry/resource facts派生ABI address-range参数；package和
  runtime只消费committed executable/manifest，不直接读raw offset facts。本层不新增placed
  memref或影子descriptor中间层。
- Downstream consumer:
  whole-entry DDR memory planning、physical transport acceptance、all-rank transport verification和closed-loop
  whole-variant candidate driver及Q16 typed rank-record validation；atomic commit后target LLVM消费
  committed IR/resource bindings，package只消费committed executable + Q17 verified staged target module
  records，runtime只消费validated manifest。
- User-level driver / named pipeline:
  Q16以后由同一
  `wafer-compile --input-program-dir ... --output-program-dir ... --execution-ranks={1|16} --target-profile=wafer-tx81-single-card-kernel-v1`
  的whole-variant
  candidate loop调用本stage。当前Q15只产出verified grouped program directory，不执行SPM planning；
  `wafer-opt`、`wafer-lower-groups-to-memory-planned-instr`和`wafer-plan-spm-memory`只处理显式IR，
  用于instruction-level replay/lit/debug，不能成为用户stop-stage，也不能把`DirectFullShape`或单group结果直接提交。
- Explicit non-goals:
  不选择 instruction form、不改变 layout assignment、不分配 DDR allocation、不生成 target CRT call 或 packet；
  不把任一 region 的 offset 独立提交，不跨 isolated region 复用 SPM range，也不把 hardware `busytable`
  当作 completion/lifetime 语义；`DirectFullShape` 与其它 tiled candidates 运行同一 per-region planning 和
  whole-variant gate，不设 bypass；不依据presumed rank equivalence复用或跳过任何rank plan。跨region
  whole-entry allocation只是在现有isolation合同改变后才有意义的优化。
- Completion gate:
  对每个合法 `wafer.tile.region` 给出 deterministic memory plan；planned storage 的 size、alignment、
  range/end、lifetime 和 alias relation 能由 region-local IR/effect/verifier 重算；generic `async.call`
  token/value由identity-preserving handle flow上的`async.await`或direct group的`async.await_all`收口，DTE由
  exact token/`wafer.instr.dte_wait`收口，本地issue由local fence收口，且每条region exit没有pending event。
  safe pre-existing loop recurrence可规划，loop-body allocation/task的动态实例或无法证明的async handle flow结构化拒绝。
  任一 region/rank 失败都使
  整个 variant clone 不可提交。即使 distributed 层认为 ranks 等价，也必须验证每个 static rank entry
  中的全部 region。
```

### 1.2 Shared Recomputable Analysis Boundary

SPM与DDR可以共享的不是arena或resource语义，而是从当前structured IR重算的analysis mechanics：path condition、
operation timeline、query-time provenance closure、live segment overlap、generic async completion identity、local
issue/fence completion、weighted conflict priority和deterministic first-fit。compiler-managed allocation由
`RootRef`指向packing demand；caller-owned/external memref由path-qualified `ValueOriginRef`表达，两者都与
`async.call`等producer创建的task identity分离。`rootsAt`/`originsAt`在查询点沿`ViewLikeOpInterface`、
`SelectLikeOpInterface`、`scf.if` yield和`scf.for` init/iter-arg/backedge/result递归闭包，因而loop body中较早
建立的view也能看到最终backedge origin union。该analysis是`WaferTransforms`内部typed对象，每次从当前IR重建；
不写lifetime timestamp、root/origin/task map、conflict graph或packing proposal到IR，也不形成跨pass side table。

generic async合同另以task identity证明terminal completion：`async.call`产生的`!async.token`和
`!async.value`必须由path-covering `async.await`完成；direct `async.create_group` handle可以通过
`async.add_to_group`收集已存在task，再由`async.await_all`完成。mutable group alias、loop body动态创建task后加入
captured group、选择不同task identity的`SelectLike`和非identity-preserving `scf.for`均无法仅凭handle root union
证明完成，必须以`unsupported_async_completion_flow`拒绝；terminal仍有pending task则以
`missing_async_completion`拒绝。`scf.if`只有在task origin本来只存在于对应branch path时，result wait才能完成它；
在分支前已经发起的不同task不能靠if选择隐式取消未选task。

SPM owner继续独占以下合同：以non-nested isolated `wafer.tile.region`为scope、保守拒绝所有loop-carried
`!async.token`、跟踪local Compute/Movement issue及Direct DTE token到path-covering fence/exact wait、验证loop
backedge和terminal drain，并按SPM base/limit/alignment提交`wafer.spm.offset`。共享generic async analysis不替代
DTE origin/wait legality；也不能把DDR的whole-entry/external-root/resource-limit语义反向引入SPM。

当前SPM动态执行scope只接受由`func.func`、`scf.if`和`scf.for`顺序拥有的non-nested tile-region；
async/parallel/unknown region owner均拒绝。SPM value不得成为`func.func`边界，也不得出现在tile-region外；唯一的
helper边界是defined `async.func`的SPM formal，由tile-local call site传入且callee body须独立通过DTE/local/generic
async terminal proof。active/async/parallel SPM scope中的indirect、external/unresolved或call graph上可能执行另一
tile-region的direct `func.call`拒绝；module存在tile-region时external/unresolved `async.call`全局拒绝。该规则不靠
symbol名字判断callee行为，而是在缺少显式arena/resource summary时保守阻止物理SPM arena重入。

## 2. 借鉴点

只吸收这些已有系统的基本方法：

- MLIR Bufferization：从 tensor SSA/effect analysis 改写到 tile-local storage，再把可表达的 storage
  降到 `memref` / LLVM conversion 能消费的 IR。
- XLA BufferAssignment：executable 需要明确 buffer size、reuse、parameter/output/temp 关系。
- TVM USMP / TFLM arena planner：静态 memory planning 可以用 deterministic first-fit 方案先落地。
- register allocation：lifetime 不重叠才可复用，但 Wafer 还要处理 size、alignment、range 和 wait。

## 3. 输入输出

输入：

- transformation-local whole-variant candidate clone 中的完整 static rank programs；其中
  `wafer.tile.region` 是禁止 SPM values 跨边界的独立 allocation scope，rejected clone 必须整体丢弃。
  SPM memory planning 不直接消费
  target-abstract tile-region IR，而消费 R3.2d 生成的 instruction-level IR with unplaced
  Wafer-tagged memref values。
- layout planner 产生的 physical layout assignment 和 materialization demand。
- instruction legalization / selection 产生的 concrete memref value、operand/result/temp/workspace/
  accumulator/psum/staging 分类、instruction family、effect event 和 async policy。
- `WaferCommOpInterface` 或后续 communication instruction selection 提供的 source/destination buffer、byte count、
  token/wait 和 staging storage。
- target policy：SPM range、reserved range、alignment、coloring preference。
- 每个 region 内的 structured control-flow、SSA/alias relation、op effect、token、fence/wait/barrier 和
  terminal completion boundary，以及 variant 中 all-and-only region/rank coverage。

输出：

- `wafer.tile.region` 中带 `#wafer.memory<space, layout>` 的 memref；其中 `#wafer.memory<spm, *>` memref 由本文
  allocator 分配，`#wafer.memory<ddr, *>` memref ownership和accepted range由DDR memory planner负责；
  resource role/scope/entry binding在atomic executable commit时materialize。
- target-codegen从committed IR、typed executable bindings和accepted offset facts派生address/range参数；
  RuntimeSession只为verified manifest slots实例化resource base/handle，不读取或重算SPM plan。
- movement/materialization/compute/sync op。
- pass-local allocation summary：offset/range、size、alignment、lifetime、alias group。
- hardware lowering 需要的 begin/end range 和 dtype storage size。

这些输出属于 committed static rank program 的 SPM / tile-region scopes，不回写到 `wafer.group`；
offset facts 只有作为 complete passing variant 的一部分才能进入主 IR。
其中 allocation summary 只覆盖 `#wafer.memory<spm, *>`；DDR 的 external view/descriptor validation、
constant residency/storage、compiler-managed DDR `memref.alloc`、全局容量、largest contiguous range
和 bandwidth 属于 DDR memory planning，但 movement/scheduler 仍要把 DDR byte footprint 和
bandwidth 作为 cost/legality input。

## 4. Instruction Storage Requirements

allocator 的输入仍可命名为 `BufferDemand`，但它不是直接从 target-abstract `wafer.tile.*` compute /
movement op 猜出来的。它必须由 instruction-level `wafer.instr.*` / Wafer-tagged memref 产生：

```text
BufferDemand {
  id
  kind
  wafer_memory_attr
  physical_layout
  storage_size
  required_alignment
  lifetime
  effects
  alias_group
  overlap_class
}
```

V0 `kind`：

- input/output tile。
- intermediate tile。
- temporary/workspace。
- accumulator/psum。
- layout materialization temp。
- optional double-buffer slot。
- communication staging buffer。
- host-visible writeback staging。

### 4.1 Instruction Storage Providers

SPM allocator 不按 op 名字猜 buffer，也不把一个 target-abstract op 当成一条硬件指令。demand
来源应是 instruction legalization / selection 后的明确 Wafer-tagged memref graph：

- `wafer.tile.*` compute ops / target-abstract movement op：通过 compute/movement 文档定义的接口枚举或选择
  hardware instruction family，例如 NE GEMM、CT elementwise/reduce、TDMA GatherScatter；SPM
  memcpy 是 GatherScatter 的 contiguous descriptor 特例。
  instruction-level IR 再报告 operand/result/temp/workspace/accumulator/psum memref demand、instruction family
  和 async lowering policy。
- `wafer.tile.materialize_layout`：不能只报告“source read / result write”。它必须先选择具体
  materialization instruction lowering，例如可展开成具体 GatherScatter 序列的 ChannelNorm /
  DechannelNorm algorithm、普通 GatherScatter 或 reject；不同 lowering 可产生不同 temp、padding、
  range 和 queue 行为。
- `wafer.tile.*` communication ops p2p op：需要 communication instruction lowering 报告 send source、recv destination、
  communication staging buffer、fixed byte count、token/wait lifetime 和 DTE/FSM resource class。
- `wafer.instr.local_fence` 和后续 sync boundary：报告 local fence、comm wait、group barrier 对 instruction event、buffer lifetime 和
  reuse 的收口。

如果某个 target-abstract op 无法产出可验证 instruction-level IR，不能让 SPM memory planning 用名字或示例
shape 猜测；应先扩 op interface / instruction IR，或保持在更高层 IR。
如果当前只有 R3.2a/R3.2b 的 group-level analysis summary，而没有 R3.2c
`wafer.tile.region` IR，SPM memory planning 不能直接运行；如果只有 R3.2c target-abstract
tile-region IR 而没有 R3.2e candidate DDR tile views 和 R3.2d instruction-level IR，SPM memory planning
同样不能运行。必须先把 tile load/store 的 DDR operand 降成真实 `memref.subview` view，再降到
`wafer.instr.*` / Wafer-tagged memref，让 buffer values、lifetime、effect 和 DMA descriptor
都可由 IR 结构重算。

`storage_size` 必须由 `computeWaferPhysicalTensorInfo(memrefType)` 统一计算，至少包含：

- compact `Tensor/NTensor` vs aligned `Cx/NCx`。
- dtype storage size 和 bool bitpack。
- C0 tail/fold。
- 256B line/layout padding。
- NHWC batch bank alignment。
- wrapper-specific byte count / range-end rule。

`BufferDemand` 的 `wafer_memory_attr`、`physical_layout` 和 `storage_size` 适用于所有 memory space；
但本文 allocator 只为 `#wafer.memory<spm, *>` demand 放置 offset。`#wafer.memory<ddr, *>` demand 不进入 SPM memory planning，只进入 movement legality、
range/bandwidth cost、host-visible lifetime 和 DDR memory ownership 检查。

## 5. Lifetime and Effects

lifetime 从 IR 结构和 effect 推出：

- SSA use-def。
- region/control-flow。
- loop-carried value。
- op memory effects。
- async issue + fence/wait。
- communication wait / group barrier。
- host-visible output boundary。

V0 规则：

- synchronous op 的 input live 到该 op read 完；output live 到最后 use。
- 本地 compute/movement 写入的 destination 在后续 `wafer.instr.local_fence` 前不能被复用；fence
  event 将此前未 fenced 的本地写 lifetime 延伸到该 fence。
- DTE communication issue 的 source/destination 通过返回的 `!async.token` 绑定到
  `wafer.instr.dte_wait`；token 被 wait 消费前，send source 和 recv destination 都不能被复用。
- 其它 async movement/compute/communication 的 source/destination live 到对应 fence/wait。
- 每个 issue 要么返回可追踪的 `!async.token` 并由匹配 wait 消费，要么由显式 local fence 完成；
  effect 只描述 issue/read/write 关系，不能暗示不存在的 completion。
- generic `async.call`返回的token/value同时携带所访问root和独立task identity；handle use可以延长root lifetime，
  但只有identity-preserving `async.await`或direct group的`async.await_all`能完成task。root union不能替代completion proof。
- `wafer.instr.dte_send` 的 source buffer live 到 send completion 或 protocol 允许复用的 wait；`dte_recv`
  destination 在 comm wait 前不能被 compute 读取。
- loop-carried accumulator/psum 跨 backedge live。
- loop body内未跨backedge携带且其全部异步访问已完成的per-iteration temporary可以在下一iteration复用；body内
  allocation一旦通过memref或async handle跨backedge携带，就代表多个动态allocation/task instance，当前没有
  multi-instance/ping-pong placement，必须fail closed。
- 非repeatable branch buffer只有在control-flow证明互斥时才能复用；`scf.for`内的分支每个动态iteration可重新选择，
  其相反branch path对packing仍按may-overlap处理。
- region 内的 nested control-flow 不自动截断 lifetime；`wafer.tile.region` 边界由 IR isolation 和禁止 SPM
  输入/结果的 verifier形成storage边界，但nested `wafer.tile.region`仍共享同一physical arena且当前结构化拒绝；
  不能从名字、group顺序或isolation trait恢复跨region SPM alias。
- region exit 前必须有显式 terminal drain/fence/wait 收口所有 path 上的 pending local/DTE events；
  不能用“后续 package/runtime 会等待”作为 lifetime 证明。

reuse 分类：

- must-alias：DPS/in-place result，必须共享 range。
- may-reuse：lifetime 不重叠，memory/alignment 兼容。
- must-not-alias：lifetime 重叠、async 未 wait、external-visible 或 verifier 禁止 alias。

## 6. Allocation Contract

SPM allocation 的职责是在一个 complete static rank variant、指定 final candidate layout assignment 和
selected instruction lowering 下，逐个 isolated `wafer.tile.region` 放置 `#wafer.memory<spm, *>` memref，
再汇总全部 region 的通过结果。
它不负责全局寻找最佳 group，不选择 compute/movement instruction form，也不把失败方案 materialize
到主 IR。

输入必须足够接近真实 lowering：

- tiled control-flow / event order。
- layout assignment 和 selected materialization / compute / movement instruction form。
- instruction-derived `BufferDemand`。
- effect / async issue / fence / wait / barrier。
- target range、reserved range、alignment、range-end policy。
- optional double-buffer / communication staging policy。

输出：

```text
SPMOffsetResult {
  status
  peak_spm
  allocation_summary
  failure_reasons
  suggested_repairs
}
```

V0 event model：

- 每个 movement / materialization / compute / sync op 产生 issue/read/write/fence/wait event。
- communication p2p op 产生 send/recv issue event，`wafer.instr.dte_wait` 或 lower-level DTE/FSM wait
  产生 completion event。
- 本地 compute/movement issue 的全部 SPM read/write 都进入 pending local access；可信
  `wafer.instr.local_fence` 前不得结束对应 lifetime 或复用buffer。
- async DTE token 将 issue operand refs 延伸到对应 wait；本地 fence 只收口 compute/movement
  issue/read/write，不替代 DTE wait；DTE wait也不收口本地engine issue。
- generic async token/value/group将root lifetime和task identity分别传播；只有受支持的terminal await flow清除task。
- loop backedge 让loop-carried value跨iteration live；body-local allocation/task被携带时拒绝静态单地址规划。
- 非循环分支只有在control-flow可证明互斥时共享lifetime slot；loop-local repeatable branch不能证明全执行互斥。
- dataflow analysis 在单个 `wafer.tile.region` 内运行；nested control-flow不清空pending events，每条
  region exit path都必须证明pending set为空。nested tile-region scope在analysis前拒绝；sibling region之间不共享
  SPM value或pending set，variant gate仍要求all-and-only regions均已规划。

这个 event model 只用于 analysis 和 verifier 可复核的 lowering；它不是新的 schedule attr。

`busytable` 只作为 target capability/legality/cost input：它可以限制可并发 queue、in-flight issue
数量或影响 overlap 成本，但不能替代 `!async.token`、wait/fence、memory effects 或 terminal drain，
也不能作为“硬件会自动处理”而缩短 buffer lifetime。

## 7. Range and Alignment Policy

当前事实和 V0 策略：

- per-tile SPM window 是 3 MiB。
- 普通 tensor allocation 使用保守可用区间，避开 Kcore/runtime reserved range。
- allocator 必须检查真实 end address，不能只检查 base。
- wrapper / packet 路径通常携带 begin/end range；debug/raw lowering 必须能镜像 wrapper 的
  range-end 结果。

当前保守区间：

```text
ordinary tensor allocation: [0x10000, 0x2F0000)
adapter public SPM upper bound: 0x2EFFFF
```

alignment：

- hard alignment 来自 wrapper/op verifier、physical layout、stride/range-end 规则。
- 256B 是 line/layout padding 粒度，也是普通 allocation 的保守 preferred alignment。
- 64KB 是 overlap-critical allocation 的 page/color 粒度，不是所有 packet base address 的硬性
  legality。

V0 对 coloring 的处理：

- 普通 buffer 不做 hard coloring。
- overlap-critical buffer 记录 color class，作为 cost/diagnostic。
- 只有当 scheduler 明确启用 parallel issue 且 target policy 要求隔离时，color conflict 才升级为
  hard legality。

## 8. V0 Allocation Algorithm

V0 只采用一个 deterministic greedy arena allocator。

1. 从 instruction-level IR with unplaced Wafer-tagged memref values 收集 `BufferDemand`。

2. 生成 interval：

```text
Interval {
  demand_id
  start_event
  end_event
  size
  required_alignment
  preferred_alignment
  alias_group
  overlap_class
}
```

3. 合并 must-alias group。

4. 固定 reserved/fixed range。

5. intervals / lifetime segments 排序：

- fixed range 优先。
- size 大优先。
- conflict pressure 高优先，即与其它 live demand overlap 的 physical bytes 多优先。
- lifetime span 长优先。
- materialization temp 靠后，方便失败时移动 cut。

6. lowest-gap allocation：

- 找满足 alignment 的最低可用 offset。
- 不能与 lifetime overlap 的已放置 interval 重叠。
- 不能碰 reserved/forbidden range。
- 若 color policy 不满足，V0 先计入 penalty；hard policy 下返回失败。

7. range/end verification：

- 检查 `base + allocated_size`。
- 检查 wrapper begin/end range。
- 检查 bool bitpack、stride byte count、Cx/NCx padding。

8. 返回 result。

这个算法不保证全局最优，但 deterministic、可诊断、足够服务 group planner 的闭环搜索。

## 9. Failure Feedback

Current structured failure reasons：

- `capacity_overflow`
- `invalid_spm_range`
- `alignment_unsatisfied`
- `unsupported_layout_conversion`
- `lifetime_overlap_conflict`
- `unsupported_lifetime_control_flow`
- `unsupported_lifetime_alias`
- `missing_async_completion`
- `unsupported_async_completion_flow`
- `missing_local_completion`
- `missing_dte_completion`
- `unsupported_completion_control_flow`
- `unsupported_spm_scope_nesting`
- `completion_proof_failure`

`reserved_range_conflict`、`range_end_overflow`、`materialization_peak_too_high`和`color_conflict`只在对应
reserved-range/materialization/coloring policy真正进入当前planner后才成为failure class；不用未实现的诊断名
冒充当前completion证明。

suggested repairs：

- shrink traversal tile。
- request internal split for a specific op/axis。
- move materialization cut。
- choose another layout assignment。
- disable double buffer。
- split group at a producer/consumer edge。

allocator 不做无限 repair；它只返回结构化失败，让 group/layout planner 调整。

## 10. Whole-Variant Commit Model

V0 使用 `plan complete static rank programs before atomic commit`：

```text
complete static rank variant clone + group/tile/layout proposals
  -> materialize every group traversal and its tile-local wafer.tile.region scopes
     (target-abstract compute/comm/load-store/layout/sync/storage/effect)
  -> materialize candidate DDR tile views as memref.subview operands
  -> legalize/select instruction-level wafer.instr.* over unplaced Wafer-tagged memref values
  -> collect BufferDemand + region-local liveness/effect from every isolated tile region
  -> per-region SPM memory planning + terminal completion verification for all ranks
  -> planned offsets for the complete variant or failure feedback
  -> run DDR/instruction/transport/target gates
  -> atomically commit all rank programs and eliminate all wafer.group, or commit nothing
  -> validate candidate IR-derived resources/completion and materialize typed C++ rank records
  -> target codegen derives address-range parameters from committed IR + executable bindings
  -> package/runtime consume committed executable/manifest only
```

原因是 layout、SPM、tile shape 和 target-abstract op selection 强耦合。早期如果只看 logical
group summary 或裸 tensor value，再把真实 allocation 推到更晚的 pass，容易让合法性承诺漂移；
但把 rejected tile-region IR 直接落入主 IR 再回滚也会污染 IR 边界。因此 instruction legalization /
selection 和 SPM memory planning 都应在 transformation-local whole-variant clone 上运行；
`wafer.tile.region` 是该 clone 中的 SPM isolation scope。allocation 使用 region-local instruction-level
memref / lifetime / effect / token 事实源，而不是 target-abstract op 的粗粒度 effect；clone 只负责
all-region atomic acceptance，不把这些隔离的 lifetime 合并成影子 whole-entry plan。

失败的 allocation、offset search trace、cost breakdown 都不进入 IR。任一 group/rank 失败时，
已经计算出的其它 offset facts 与 candidate clone 一起丢弃。

## 11. Runtime / ABI Handoff

Wafer-tagged memref 是 layout / SPM planning 阶段的 tile-local buffer value。SPM bufferization
接受 layout assignment 和 allocation 后，不再新增独立 placed memref / explicit descriptor IR 层。
Q16 commit前的rank-record validation从instruction IR中的memref use-def、view relation、
`wafer.spm.offset`、`wafer.ddr.offset`、arena/placement和
`computeWaferPhysicalTensorInfo(memrefType)`重算resource role/range/scope/alias与entry slot需求，
验证plan一致性、补全range/capacity并materialize typed C++ entry bindings，再随all-rank atomic commit写入bundle；
不能在此新造state consistency、arena/residency policy或ResourceId。

target-codegen之后只能结合committed instruction IR与这些executable bindings派生address/range/
stride参数。package从committed executable序列化runtime-observable resource fields，RuntimeSession只实例化
typed bindings。package/runtime不得再扫描raw memref/offset恢复resource role、scope、alias或lifetime。

派生规则：

- compact `Tensor/NTensor`：address range 由 accepted base offset、logical shape、dtype 和 compact
  layout footprint 重算；不写第二份 range attr。
- aligned `Cx/NCx`：用 `computeWaferPhysicalTensorInfo(memrefType)` 展开 physical extent、padding、
  storage bytes 和 begin/end range；这些是 verifier/codegen 派生值，不作为新 IR 事实源。
- 如果目标 movement/compute instruction 需要 range、logical-to-physical mapping、layout family、
  stride 或 packet field，ABI/packet emission 在 very-late lowering 中从当前 IR 派生对应参数；
  不能把它们作为 R3.4-style access descriptor 在主线 IR 中传递。

这个 handoff 不重新选择 layout，也不重新移动 materialization cut。若某个 Wafer-tagged memref
无法由 accepted facts 派生合法 address/range 参数，说明前面的 layout/SPM plan 不合法，应返回
planner，而不是让 LLVM lowering 才失败。

## 12. IR 表达

SPM bufferization 后，IR 应显式表达：

- Wafer memory attr：address space + physical layout marker。
- buffer ownership / alias relation。
- movement / materialization / compute / sync op。
- necessary effect / wait / barrier。

target-codegen materialization后，bank/color、worker、queue、packet field只在能验证它们的lower-level IR或
conversion-local value中出现；runtime physical base/handle只存在RuntimeSession。主线instruction IR不新增
placed memref、flat backing memref 或 explicit descriptor 事实源。
`wafer.group` 只消费 feasibility 结论，不携带这些字段。

`#wafer.memory<space, layout>` 在 committed instruction IR 中仍是统一语义：`spm` 表示 tile-local
SRAM，`ddr` 表示 device/global DDR address domain。RDMA/WDMA verifier 用 source/destination
address space 检查方向；DDR memory planning 用 `ddr` 继续关联 view/range、compiler-managed DDR
planned range、constant storage/residency 和 declared arena resource。Q16 commit前从同一candidate IR验证并补全
accepted range/arena relation；commit后package/runtime只消费
该唯一owner，不在launch metadata中再次关联或恢复range。

不要把 `wafer.tile.region` body 已经表达的执行结构复制成全局 allocation plan attr。

### 12.1 Accepted SPM Offset Fact

R3.2f V0 在 instruction-level IR 上使用 `wafer.spm.offset` op attr 表达 SPM memory planning
接受的 offset/range fact。该 attr 挂在定义 SPM buffer value 的 `memref.alloc` 上，值为
`#wafer.spm_offset<offset>`：

- `offset` 是 tile-local SPM byte offset。

以下事实不写入 attr，因为它们可由当前 IR 或 target policy 稳定重算：

- `size` 来自 `computeWaferPhysicalTensorInfo(memrefType).physicalBytes`，不是 logical compact bytes。
- `alignment` 来自 target policy 和 `memref.alloc` alignment；planner 用它验证 offset，但 accepted
  IR 不复制该输入。
- bank span 按 256B bank-line 粒度由 `[offset / 256, ceil((offset + size) / 256))` 重算。

每个 logical rank 的 SPM window 仍是 rank-local physical arena。当前 IR 通过 `IsolatedFromAbove` 和
verifier 禁止 SPM buffer 作为 `wafer.tile.region` 输入/结果，planner再拒绝nested tile-region scope，因此每个
non-nested region可以从同一arena policy独立分配offset；正确性不依赖跨sibling region lifetime或range复用。
Q0要求完整traversal中all-and-only regions均完成规划，并由module clone保证任一region/group/rank失败都不部分提交。若未来允许 SPM
value 跨 region，必须先改变 IR 边界并引入可验证的 SSA/alias/completion relation；whole-entry allocator
届时才是必要机制。在当前 isolation 合同下，它只能作为降低 peak/fragmentation 的后续优化，且保持相同
offset-only输出合同。

R3.2f V0 的 dataflow 边界：

- `memref.alloc`产生SPM demand；compiler-managed demand使用`RootRef`，path-qualified semantic origin使用
  `ValueOriginRef`，它们与async task identity分开。view-like和SelectLike result通过query-time closure关联所有
  可能root；未知tracked memref producer不被当成external root，直接以`unsupported_lifetime_alias`拒绝。
- `bufferization.to_tensor/to_memref`只传播已有origin；SPM没有可以由adapter自行建立的external root。任何
  tracked alias/control-flow result都必须解析到allocation或已有semantic origin，generic memref不能仅靠
  `memory_space_cast`或unknown tensor adapter升级成SPM storage。
- `scf.if`为then/else建立path condition；只在非repeatable path上live的SPM buffer可以复用同一offset，yield
  到if result后按分支路径继续延伸。loop内if的branch decision可在后续iteration改变，不能作为packing互斥证明。
- `scf.for`从init、iter-arg、yield和result递归闭合root/origin；发布fixed-point union时去掉loop-local
  repeatable decision。pre-existing loop-carried root覆盖整个loop subtree；未yield且完成全部access的body-local
  temp可按iteration复用，body-local allocation通过memref或async handle跨backedge携带则因缺少multi-instance
  placement而拒绝。
- generic `async.call`的token/value把SPM roots延伸到`async.await`；direct
  `async.create_group`/`async.add_to_group`/`async.await_all`保留同一task identity。group alias、loop动态task加入
  captured group、SelectLike合并不同task或非identity-preserving loop recurrence均拒绝，未await task在terminal失败。
- DTE send/recv产生的`!async.token`把对应SPM refs延伸到精确的`wafer.instr.dte_wait`；token经
  `scf.if` result合并时保留各自origin和path condition。local fence不能消费该completion；SPM当前在DTE
  owner proof之前保守拒绝所有loop-carried async token。
- 带`WaferResourceEffectInterface`的本地compute/movement issue进入pending local issue集合，其全部
  SPM read/write进入pending local access集合；`wafer.instr.local_fence`只在自身path condition覆盖的
  路径上延伸并清除这些状态。DTE wait不能消费本地状态。
- 当前每条`wafer.tile.region` exit path执行terminal check；存在未await的generic task、未等待DTE token或未fence的
  本地issue/read/write时规划失败。可能zero-trip的loop内单一completion不能覆盖loop外pending状态；合法loop本身
  不因此失败，但SPM保守拒绝所有loop-carried async token。
- rejected/candidate offset、search trace、cost estimate 和 repair suggestion 仍是 analysis，不写入
  IR。

实现边界是 R3.2d 已支持的 structured `scf.if` / `scf.for`、single-block `scf.yield` 和
instruction-level async token use。未结构化/多block CFG、超过 64 个 branch/loop decision point，以及未来显式
must-alias group 需要先由 SSA / op interface / verifier 表达，再进入 SPM memory planning；不能靠
名字或旁路协议恢复。

R3.2f SPM memory planning 使用经典静态 memory planning / interval allocation 的保守 baseline，而不是把
alloc event 顺序直接当作 allocation 顺序。算法分两层：

- lifetime analysis 仍由当前 IR 的 SSA、region、path condition 和 async token use 重算，得到可同时
  发生的 lifetime segments。
- allocation order 使用 pressure-weighted offline packing：优先放置 physical size 大、与其它 live
  demand 冲突压力高、lifetime span 长的 demand，再按 alloc event / ordinal 稳定打破平局。

这样可以避免小 buffer 先占低地址造成 arena fragmentation，导致后续大 buffer 在总容量可行时失败。
offset 选择仍保持 deterministic bounded search：只在当前 assigned intervals 形成的合法 gap 中选最低
可行 offset；rejected/candidate offset、candidate 排序权重和搜索 trace 都保持为 analysis，不写入
`wafer.spm.offset`。后续若要引入 graph coloring、ILP、schedule-aware double buffering 或
bank-aware coloring，必须继续保持 same input/output IR contract，只改变 analysis / search。

`wafer.spm.offset` 只保存 accepted offset，不保存 size、alignment、bank span、lifetime 或搜索 trace。
失败原因仍通过 pass diagnostic 返回，
不写进 IR；rejected/candidate offset、lowest-gap 探索过程和 repair suggestion 都保持为 analysis。

## 13. Verifier

SPM / tile-region verifier 至少检查：

- buffer range 不越过可用 SPM window 和 reserved range。
- begin/end range 与 physical layout size 一致。
- Cx/NCx、C0 tail/fold、256B padding、NHWC bank alignment、bool bitpack 计算一致。
- op operand/result 的 Wafer memory attr 满足对应 op verifier。
- must-alias relation 的读写顺序合法。
- may-reuse buffers 的 lifetime 不重叠，或由明确 wait/barrier 收口。
- async buffer 在 fence/wait 前不能复用。
- host-visible writeback 和 communication boundary 有明确 fence/wait/sync。
- 每个 `wafer.tile.region` 不接受/产出 SPM buffer，nested tile-region scope结构化拒绝；non-nested region-local
  aliases/lifetimes已纳入自身plan，且所有region exit path的pending event set为空；variant coverage确保没有漏规划region。
- tile-region只允许sequential func/scf.if/scf.for ownership；active/async/parallel scope中的unknown/external/
  indirect或可能重入SPM的call失败。SPM memref即使先擦成tensor/generic memref也不能经`wafer.tile.yield`、
  raw metadata或无origin的tracked cast/adapter逃逸scope。
- safe pre-existing loop recurrence在memory-planned named pipeline中通过并含backedge fence；loop body创建fresh
  allocation再作为recurrence result携带时，以`unsupported_lifetime_alias`拒绝而不是给所有动态实例同一地址。
- `busytable` 只参与 target capability/legality/cost 查询，不被当作 completion 或 alias proof。
- physical footprint、begin/end、descriptor range、offset addition 和 ABI width narrowing 统一调用 shared
  physical geometry/range/narrowing verifier；禁止 silent truncation 或 consumer-local 重算分叉。
- launch/resource、target LLVM 和 runtime adapter 阶段不能要求额外 placed/access descriptor fact；compact 和 Cx/NCx buffer
  的 address/range/stride 参数必须由 `computeWaferPhysicalTensorInfo`、accepted offset facts、
  allocation range 和 op verifier 一致推出。

## 14. 与 Layout / Group 的关系

全局文档边界见 `tasks/01-architecture.md` 第 8 节。SPM allocation 回答完整 static rank entry 中
每个 isolated tile-region scope 的 `#wafer.memory<spm, *>` allocation 是否可行，并把all-region汇总失败
返回 whole-variant group/layout planner。
闭环顺序：

```text
whole-variant candidate plan
  -> final candidate layout assignment in the clone
  -> complete rank traversal with wafer.tile.region scopes
  -> candidate DDR tile-view materialization
  -> instruction-level wafer.instr.* over unplaced Wafer-tagged memref values
  -> instruction storage / effect demand
  -> per-isolated-region SPM memory planning and terminal completion verification
  -> target-codegen address-range derivation from committed bindings and accepted facts
  -> complete variant accepted or failure feedback; no partial commit
```

layout planner 先尝试移动 materialization cut 或换 flexible layout；SPM 仍失败时，group planner
再缩 tile、请求 internal split 或拆 group。SPM allocation 消费 instruction-level IR 暴露的 demand
和 effect；compute implementation 和 communication algorithm 的选择属于 instruction selection /
closed-loop planner，不在 allocator 内部用名字或 case 猜测。
- local fence、comm wait、group barrier 是不同 sync event。SPM lifetime 可以把它们都建成 event，
  但不能把 NCC local fence 当成 DTE completion 或 multi-tile barrier。

## 15. 后续扩展

这些机制有价值，但不进入 V0 主路径。进入条件必须明确：

- 多 pool allocation：当 ordinary pool、communication staging、runtime-visible buffer 的 reserved
  range 和 lifetime 约束稳定后引入；在此之前用单 pool + reserved range 更容易验证。
- linear scan allocator：当 `wafer.tile.region` 大多是线性 schedule，且 greedy arena 编译成本或
  fragmentation 成为问题时引入。
- graph-coloring / interval-coloring allocator：当 lifetime 图复杂、weighted greedy 产生明显 peak SPM
  浪费，且诊断能保持清楚时引入。
- allocator 内部 repair loop：只允许有限 repair；如果 repair 开始改变 tile shape 或 group
  boundary，应交还 group planner，而不是让 allocator 变成隐藏 scheduler。
- PMU 驱动 bank conflict model：当 board profiling 能稳定解释 blocking time 和 bank/color 关系后，
  替换 V0 的 color penalty。
- worker-local / runtime-visible / communication-only pool：当对应 dialect 和 verifier 已能表达
  ownership、visibility、fence/wait 边界后引入。
- 跨 region whole-entry allocator：只有当 IR 明确允许 SPM value/alias/completion 跨
  `wafer.tile.region`，或测量证明在不改变 isolation 的情况下做全局range coloring有必要时再引入；它是
  peak/fragmentation优化，不是当前per-region correctness gate。

## 16. 参考材料

- MLIR Bufferization / One-Shot Bufferize：<https://mlir.llvm.org/docs/Bufferization/>
- XLA BufferAssignment：<https://openxla.org/xla/hlo_to_thunks>
- TVM USMP：<https://discuss.tvm.apache.org/t/rfc-unified-static-memory-planning/10099>
- TFLM Memory Planner：<https://proceedings.mlsys.org/paper_files/paper/2021/file/6c44dc73014d66ba49b28d483a8f8b0d-Paper.pdf>
