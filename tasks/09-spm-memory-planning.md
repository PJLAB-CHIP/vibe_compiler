# Wafer SPM Memory Planning Design

本文拥有`#wafer.memory<spm, *>`的Tile-local lifetime、fixed-capacity MiniMalloc placement和accepted offset合同。
输入是completion-closed、function-boundary-bufferized的current Instr IR；planner逐Tile从actual allocation roots、SSA use-def、
control flow、effects和completion重建all-and-only fixed problems。它不选择placement、tiling、fusion、layout、movement、
spill或worker/order，也不修改candidate来修复失败。

Passing candidate只增加offset-only `wafer.spm.offset`；size、alignment、range、bank span、lifetime和alias均从current IR、
layout、accepted offset和target policy重算。只有实际MiniMalloc及独立placement validator成功才能认定合法；actual capacity
rejection可以反馈controller，`ResourceExhausted`、unsupported和compiler failure保持原typed状态。High-water/headroom和
bank preference只作accepted placement的诊断或soft ordering，不参与legality。

Physical `wafer.tile.region`是单Tile residency boundary，不是arena或solver query。SPM root/alias不能跨Region；跨Region
shaped value必须通过已物化的DDR store/completion/load。本文只消费11号定义的Wafer-tagged memref、`wafer.instr.*`
和effects，不重复定义instruction op；DDR planning由12号拥有。

## 1. 核心结论

SPM planning 不能只做 byte-size estimate。候选 tile plan 是否合法，必须跑与下游一致的：

```text
selected physical encoding / transfer materialization
  -> selected DDR tile-view materialization
  -> instruction legalization / selection
  -> instruction-level IR with unplaced Wafer-tagged memref values
  -> storage requirement collection
  -> liveness/effect analysis
  -> derive all-and-only fixed SPM allocation problems from roots/lifetime/coexistence
  -> 3 MiB fixed-capacity packing + shared physical geometry/range verification
  -> target-ABI address/range/narrowing validation after selected-plan commit
  -> target-codegen derivation from the selected executable facts; runtime only binds verified TileEntryArguments to planned addresses
```

如果candidate allocation没有被接受，不能生成等待下游修复的program，也不能保留partial placement；未提交top-level TileModule subtrees整体擦除。
actual `ProvenInfeasible`可作为当前candidate的typed capacity rejection；`ResourceExhausted`或internal failure不形成capacity rejection。

### 1.1 Pipeline Contract

```text
Pipeline position:
- Upstream IR / input:
  current-IR pipeline物化的一份完整、尚未放置SPM offset的candidate `builtin.module`。all-and-only topology-available
  `wafer.tile.module`已包含完整`wafer.instr.*`、actual DDR views、unplaced
  SPM memrefs、selected encodings、movement、buffering、effects、order与completion；function boundary已经bufferize，Tile source op已经消失。
  不允许memory stage再补join/wait。不同Tile可以有不同op/loop/tail；nested
  region、SPM alias跨Tile/region或typed opaque clobber fail closed。
- Current stage responsibility:
  要求每个actual allocation有candidate transaction提供的current typed owner relation；逐Tile从current roots、control-flow
  coexistence与conflict重建all-and-only fixed problems；验证
  offset/end、alignment、lifetime/reuse、alias、reserved range和capacity，调用MiniMalloc并独立复验placement。返回validated
  offsets及per-Tile high-water/headroom；任一Tile失败擦除未提交candidate top-level TileModule subtrees并返回typed result。不得运行function-boundary
  bufferization、completion reconstruction、worker/order rewrite或其它会改变lifetime problem的上游变换。
- Output IR / files:
  仅存在于passing candidate TileModule set中的same instruction-level IR；每个SPM memref definition只增加offset-only
  `wafer.spm.offset` accepted fact，或返回结构化terminal outcome。apply前从该fact、memref use-def、
  Tile identity、arena和view relation重算resource range并形成typed Tile executable record。
  target LLVM、package和runtime只消费accepted executable/manifest；solver work与debug统计不写IR。
- Downstream consumer:
  DDR actual placement、cross-Tile transport/resource/ABI verification、target LLVM和package。上游current-Instr stage已经完成
  function-boundary bufferization与completion；failure不允许allocator原地repair、操作planning state或返回planner重选。
- User-level driver / named pipeline:
  `wafer-compile --input-program-dir ... --output-dir ... --num-partitions=1`
  selected execution named pipeline调用本stage；frontend不执行SPM planning。`wafer-opt`只对显式IR测试同一leaf stage，
  不能成为另一条production pipeline或把单Tile结果直接提交。
- Explicit non-goals:
  不选择instruction form、compute/encoding/transfer、TileRegion、retention、spill、buffer、worker/order或completion；不分配DDR；
  不把Tile-local offset独立提交，不从pipeline flag或buffer数量猜lifetime，不用bank phase改变hard feasible set；不拥有objective、
  planning state、proposal、repair或跨invocation solver cache。
- Done criteria:
  对每个合法TileModule set给出deterministic per-Tile memory plan；planned storage的size、alignment、
  range/end、lifetime和alias relation能由Tile-local IR/effect/verifier重算；generic `async.call`
  token/value由identity-preserving handle flow上的`async.await`或direct group的`async.await_all`收口，DTE由
  exact token/`wafer.instr.dte_wait`收口；本地NCC issue按typed worker进入ordered-pending set，同worker
  RAW/WAR/WAW后继可接管访问责任，首个Kcore、DTE、不同worker、unsafe reuse、writing或terminal cut前必须有
  覆盖实际participant的join。completion在root释放/reuse、真实observer和entry terminal按witness验证；region boundary不自动
  生成join，普通traversal/loop/spill点不形成terminal validation boundary。safe pre-existing loop recurrence可规划，
  loop-body allocation/task的动态实例或无法证明的async handle flow结构化拒绝。
  fixed-capacity solver在alignment hole、disconnected component、empty/zero-byte和greedy-packing反例上保持三态结果与独立
  placement validator；resource exhaustion原样传播，不运行其它allocator，也不改写成capacity事实。
  每个complete candidate从current IR独立建立actual problem并调用一次本stage；任一task/Tile失败都使本次candidate
  TileModule set不可提交；即使两个Tile modules字节等价，也必须验证每个Tile的all-root/all-domain coverage。
```

### 1.2 Shared Recomputable Analysis Boundary

SPM与DDR可以共享的不是arena或resource语义，而是从当前structured IR重算的analysis mechanics：path condition、
operation timeline、query-time provenance closure、live segment overlap、generic async completion identity、typed
worker ordered-pending/participant completion，以及由精确pairwise conflict relation驱动的static packing。
fixed-capacity solve和placement validator只消费owner-independent typed problem；两侧owner仍分别决定arena/resource语义和
offset application。compiler-managed allocation由
`RootRef`指向packing demand；caller-owned/external memref由path-qualified `ValueOriginRef`表达，两者都与
`async.call`等producer创建的task identity分离。`rootsAt`/`originsAt`在查询点沿`ViewLikeOpInterface`、
`SelectLikeOpInterface`、`scf.if` yield和`scf.for` init/iter-arg/backedge/result递归闭包，因而loop body中较早
建立的view也能看到最终backedge origin union。该analysis是`WaferTransforms`内部typed对象，每次从当前IR重建；
不写lifetime timestamp、root/origin/task map、conflict graph或packing proposal到IR，也不形成跨pass side table。

generic async合同以task identity传播pending completion：`async.call`产生的`!async.token`和
`!async.value`必须由path-covering `async.await`完成；direct `async.create_group` handle可以通过
`async.add_to_group`收集已存在task，再由`async.await_all`完成。mutable group alias、loop body动态创建task后加入
captured group、选择不同task identity的`SelectLike`和非identity-preserving `scf.for`均无法仅凭handle root union
证明完成，必须以`unsupported_async_completion_flow`拒绝；entry terminal仍有pending task则以
`missing_async_completion`拒绝。`scf.if`只有在task origin本来只存在于对应branch path时，result wait才能完成它；
在分支前已经发起的不同task不能靠if选择隐式取消未选task。

SPM planning以finalized Tile module为scope，对其完整current IR的全部roots统一收集demand、path-aware lifetime、coexistence和
physical arena facts，再形成all-and-only fixed allocation problems。entry含一个或多个non-nested `wafer.tile.region`；nested
region、SPM root/alias跨界及typed opaque clobber输入拒绝。region结构不预设problem/query数量。

actual SPM residency是本stage的late验证结论，不是上游action名称：只有current Instr IR中的producer/consumer root、
SSA/view关系、movement、order及completion共同证明一个value在SPM持续可用，planner才按该lifetime放置它。同region不自动
证明resident或fusion；`wafer.spm.offset`只在所有fixed problems通过后原子写入。

planner跨完整entry跟踪typed worker NCC pending set与Direct DTE token：same-worker exact RAW/WAR/WAW可在安全loop backedge保持
ordered pending；selective spill只结束目标root，不改变其它root lifetime。为真实reuse、observer或worker-domain切换出现的typed
join/wait会更新pending state；region exit只要求仍访问其SPM roots的work完成，entry terminal完成全部observable generic async task、
DTE token和NCC participant。共享generic async analysis不替代DTE
origin/wait legality，也不能把DDR的selected TileModule set/external-root/resource-limit语义反向引入SPM。

终态实现的dynamic ownership只接受由`func.func`、`scf.if`和`scf.for`结构化拥有的non-nested residency regions；
一个entry可有一个或多个，typed opaque clobber和unknown region owner拒绝。若不同regions可能并发，其roots按真实coexistence
联合规划，不能各自假设独占3 MiB。
SPM value不得成为`func.func`或tile-region边界；tile-region的所有data I/O只允许DDR value/view（function external或
compiler-managed materialization），non-data control不能携带SPM alias；与SPM无关的pending completion按typed合同传播。唯一的helper边界是defined
`async.func`的SPM formal，由tile-local call site传入且callee body须独立提供可验证的DTE/local/generic async
effect/token relation，最终pending completion由caller真实observer/root release/entry terminal验证。active/async/parallel SPM scope中的
indirect、external/unresolved或call graph上可能执行另一tile-region的
direct `func.call`拒绝；TileModule-wide lifetime证明某个SPM allocation在兼容路径上跨过顶层/structured
`func.call`或`func.call_indirect`仍live时，callee可能执行tile-region、external/unresolved或indirect call也
fail closed。module存在tile-region时external/unresolved `async.call`全局拒绝。该规则从当前IR重算call graph和
live segment，不靠symbol名字判断callee行为，也不形成跨过程summary；在缺少显式arena/resource summary时保守
阻止物理SPM arena重入。

## 2. 借鉴点

只吸收这些已有系统的基本方法：

- MLIR Bufferization：从 tensor SSA/effect analysis 改写到 tile-local storage，再把可表达的 storage
  降到 `memref` / LLVM conversion 能消费的 IR。
- XLA BufferAssignment：executable 需要明确 buffer size、reuse、parameter/output/temp 关系。
- TVM USMP / TFLM arena planner：静态memory planning应保持离线、确定、offset-only的consumer合同。
- Google MiniMalloc：对固定容量的ML静态buffer使用canonical search、section inference和dominance pruning；Wafer只吸收
  fixed-capacity feasibility core，并在adapter外保留自己的path/lifetime语义、typed outcome和placement validator。
- register allocation：lifetime 不重叠才可复用，但 Wafer 还要处理 size、alignment、range 和 wait。

当前actual memory/target leaf采用上述边界的直接结果：MLIR whole-function bufferization是会改变SSA、alias和allocation的
analysis+rewrite，必须由layout/bufferization stage在进入leaf前完成；MiniMalloc只消费随后从completion-closed current Instr
重建的fixed problem。Leaf复用现有SPM、DDR、transport和target kernel，不再运行组合式“bufferize + rebuild completion”
preparation pipeline，也不移动allocation来制造更短lifetime。Register-allocation类比只用于从actual lifetime判断range reuse，
不采用allocator内部spill、retile、join insertion或其它repair。缺少bufferized boundary或completion时按typed contract failure停止，
而不是由memory stage补齐。

## 3. 输入输出

输入：

- complete candidate TileModule set中的完整Tile modules；每个有assigned work的Tile module含一个或多个non-nested
  physical `wafer.tile.region` SPM residency domains，其中包含selected traversal、tile shape、retention/release与materialization；lifetime由这些actual结构派生。nested region、
  SPM root/alias跨界或typed opaque clobber输入直接拒绝；failure擦除整个未提交top-level TileModule subtrees。
  SPM memory planning 不直接消费
  target-abstract tile-region IR，而消费instruction legalization生成的instruction-level IR with unplaced
  Wafer-tagged memref values。
- selected emitter在该subtree中显式物化的implementation、physical encoding、transfer、residency、layout和movement demand。
- instruction legalization / selection 产生的 concrete memref value、operand/result/temp/workspace/
  accumulator/psum/staging 分类、instruction family、effect event 和 async policy。
- current typed communication instruction、standard memory effects和token operands/results提供的source/destination
  buffer、byte count、wait relation和staging storage。
- target policy：SPM range、reserved range、alignment和bank-phase soft preference。
- 完整entry内的structured control-flow、SPM SSA/alias relation、op effect、token、participant join/exact wait/barrier，
  以及root release、真实observer、entry completion和complete candidate TileModule set中all-and-only Tile coverage。普通traversal/loop/spill/region
  结构不是completion动作来源；region exit只检查仍访问其SPM roots的pending state。

输出：

- `wafer.tile.region` 中带 `#wafer.memory<space, layout>` 的 memref；其中 `#wafer.memory<spm, *>` memref 由本文
  allocator 分配，`#wafer.memory<ddr, *>` memref ownership和accepted range由DDR memory planner负责；
  resource role/scope/entry binding在atomic executable apply时materialize。
- target-codegen从accepted IR、typed executable bindings和accepted offset facts派生address/range参数；
  no-card `RuntimeInvocationPlan`只把verified `TileEntryArgument`映射到checked memory ranges，board execution才取得base/handle；
  两者都不读取或重算SPM plan。
- movement/materialization/compute/sync op。
- pass-local allocation summary：offset/range、size、alignment、lifetime、alias group。
- hardware lowering 需要的 begin/end range 和 dtype storage size。

这些输出属于accepted Tile module的SPM / tile-region scopes，不回写到upstream structured
program；
offset facts只有作为complete passing candidate TileModule set的一部分才能进入本次actual result；最终仅retained winner进入发布IR。
其中 allocation summary 只覆盖 `#wafer.memory<spm, *>`；DDR 的 external view/descriptor validation、
constant residency/storage、compiler-managed DDR `memref.alloc`、全局容量、largest contiguous range
属于 DDR memory planning；movement/scheduler把 exact DDR byte footprint作为cost input。带宽只有qualified PMU事实存在后才能参与
合法候选排序，不能反向改变range/capacity legality。

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

当前 `kind`：

- input/output tile。
- intermediate tile。
- temporary/workspace。
- accumulator/psum。
- layout materialization temp。
- explicit multi-buffer slot；selected physical-dataflow IR必须已经把loop外真实allocation root和loop-carried
  rotation物化进current IR，本planner只消费这些事实，不能由kind或queue depth推断。
- communication staging buffer。
- host-visible writeback staging。

### 4.1 从 Instruction IR 派生 Storage Demand

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
- typed `wafer.instr.ncc_join`和后续sync boundary：报告participant join、comm wait、group barrier对
  instruction event、buffer lifetime和reuse的收口；completion只接受typed NCC join。

如果某个 target-abstract op 无法产出可验证 instruction-level IR，不能让 SPM memory planning 用名字或示例
shape 猜测；应先扩 op interface / instruction IR，或保持在更高层 IR。
如果当前只有planning tensor summary而没有`wafer.tile.region` IR，SPM memory planning不能直接运行；
如果只有target-abstract tile-region IR而没有selected DDR tile views和instruction-level IR，同样不能运行。必须先把
tile load/store的DDR operand降成真实`memref.subview` view，再降到
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
range/capacity/alignment gate、exact byte cost、host-visible lifetime 和 DDR memory ownership 检查。

## 5. Lifetime and Effects

lifetime 从 IR 结构和 effect 推出：

- SSA use-def。
- region/control-flow。
- loop-carried value。
- op memory effects。
- async issue + typed completion/exact wait。
- communication wait / group barrier。
- host-visible output boundary。

当前规则：

- Structured timeline从current `scf.for`的常量上下界和正step证明至少执行一次时，body继承parent路径，
  不再增加“可能零次”的独立decision。不能证明非空时保留原optional-body路径；loop内真实条件分支仍标为repeatable，
  不用于证明跨iteration的packing互斥。该规则只精化只读路径分析，不改变IR、completion或allocation。
  依据[SCF循环语义](https://mlir.llvm.org/docs/Dialects/SCFDialect/#scffor-scfforop)，具体比较与类型以pinned SCF为准。
- synchronous op 的 input live 到该 op read 完；output live 到最后 use。
- 本地 compute/movement issue按typed worker进入ordered-pending set；source/destination access至少活到
  matching participant join，或活到可由same-worker exact RAW/WAR/WAW后继接管的有序访问。不同worker、DTE、
  Kcore、host observer、unknown alias和不安全复用不能使用该接管证明。
- DTE communication issue 的 source/destination 通过返回的 `!async.token` 绑定到
  `wafer.instr.dte_wait`；token 被 wait 消费前，send source 和 recv destination 都不能被复用。
- 其它 async movement/compute/communication 的 source/destination live 到其typed completion或有序后继接管点。
- 每个 issue 要么返回可追踪的 `!async.token`并由匹配wait消费，要么携带typed NCC worker并由ordered-pending/
  participant合同闭合；effect只描述issue/read/write关系，不能暗示不存在的completion。
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
- region内的nested control-flow不自动截断lifetime；`wafer.tile.region` isolation禁止隐式capture，且SPM operand/result
  不得形成cross-region data edge。同一resident allocation root的yield/result/consumer链必须保留在一个region；selected
  spill结束该root，matching reload建立新root；它们在同一region内时不影响其它live roots，在下一个region时则
  要求selected cut已另行结束或materialize切口上的全部SPM roots。typed opaque clobber和
  nested `wafer.tile.region`结构化拒绝，不能从名字、task顺序或isolation trait恢复跨region SPM alias。
- compiler-managed DDR materialization cut按current IR中的managed DDR root、对应WDMA和RDMA到fresh managed SPM root识别。
  fresh completion在store后完成其participant worker，使被spill的SPM source可释放；reload前若同worker仍有intervening
  pending work，再插matching participant join。只有latest issue、fresh allocation和reload位于同一block且顺序可证时，
  该join才可前移到allocation之前以形成packing reuse witness；否则保守地放在reload前。不能为此把通用allocation lifetime
  从allocation event改成first use，也不能把普通region boundary当成completion。
- 普通traversal、loop、spill或local materialization结束不验证terminal completion。为reuse或observer执行的typed
  join/wait只更新current pending set；region exit只验证仍访问其SPM roots的work已被显式typed completion收口，
  region结构本身不插入或执行wait/join；entry terminal用typed participant join/exact
  wait收口all-and-only observable pending NCC/DTE/generic async event，也不能用“后续package/runtime会等待”
  作为lifetime证明。

reuse 分类：

- must-alias：DPS/in-place result，必须共享 range。
- may-reuse：lifetime 不重叠，memory/alignment 兼容。
- must-not-alias：lifetime 重叠、async 未 wait、external-visible 或 verifier 禁止 alias。

## 6. Allocation Contract

SPM allocation的职责是在一个complete candidate TileModule set、指定physical-dataflow realization和
selected instruction lowering下，对全部SPM roots按真实lifetime/coexistence形成fixed problems并在3 MiB arena中放置。
它不负责全局寻找最佳task/dataflow schedule，不选择compute/movement instruction form，也不把失败方案materialize
到主 IR。

输入必须足够接近真实 lowering：

- tiled control-flow / event order。
- actual compute form、physical encoding、transfer/layout materialization和compute/movement instruction form。
- instruction-derived `BufferDemand`。
- effect / async issue / typed participant join / exact wait / barrier。
- target range、reserved range、alignment、range-end policy。
- explicit communication staging policy；multi-buffer policy只有在IR已有两个真实slot和typed completion时才可输入。
  actual buffers、slot relation、worker/order和completion必须由上游selected construction交付到该output边界，planner始终不推断或补建。

输出：

```text
SPMOffsetResult {
  status: Feasible | ProvenInfeasible | ResourceExhausted
  validated_placement?        // present only for Feasible
  actual_high_water_bytes?    // recomputed from validated placement
  failure_reason?
}
```

该结构是owner-private conceptual result，不新增IR type或跨pass output。`Feasible`必须携带覆盖全部demand且通过独立
validator的placement；`actual_high_water_bytes`只能从该placement的`max(offset + size) - arena.begin`重算。
`ProvenInfeasible`只表示fixed-capacity搜索在预算内完成并证明当前arena无解；`ResourceExhausted`只表示搜索资源不足，
不能伪装成容量失败。allocator不返回lower bound、optimality gap、pressure witness、repair action或plan建议。
empty demand返回`Feasible`、empty placement和high-water 0。

调用边界还必须把allocator internal failure作为独立typed failure传播。只有`ProvenInfeasible`可拒绝当前fixed assignment并
形成causal no-good；`ResourceExhausted`与internal failure都是indeterminate，只消耗本次work并保留原合法search state，
不得被planning提升成unsupported、capacity failure或其它exact rejection。

当前event model：

- 每个 movement / materialization / compute / sync op 产生issue/read/write/typed completion event。
- communication p2p op 产生 send/recv issue event，`wafer.instr.dte_wait` 或 lower-level DTE/FSM wait
  产生 completion event。
- 本地 compute/movement issue 的全部SPM read/write按typed worker进入pending local access；same-worker
  exact RAW/WAR/WAW后继可保持issue order并接管访问责任，否则必须活到覆盖该worker的participant join。
- async DTE token将issue operand refs延伸到对应wait；NCC participant join只收口其participant worker，
  不替代DTE wait；DTE wait也不收口NCC issue。
- generic async token/value/group将root lifetime和task identity分别传播；只有受支持的terminal await flow清除task。
- loop backedge 让loop-carried value跨iteration live；body-local allocation/task被携带时拒绝静态单地址规划。
- 非循环分支只有在control-flow可证明互斥时共享lifetime slot；loop-local repeatable branch不能证明全执行互斥。
- dataflow/lifetime analysis在每个Tile module的统一timeline和demand set上运行；nested control-flow、普通traversal、
  loop、spill点和region结构都不自动清空pending events。region exit只要求访问其SPM roots的pending set为空，entry terminal
  闭合observable pending work；nested tile-region在analysis前拒绝，DeviceExecutable verification要求全部roots被all-and-only accepted placement覆盖。

这个 event model 只用于 analysis 和 verifier 可复核的 lowering；它不是新的 schedule attr。

`busytable`只作为same-worker actual-range dependency legality input：它帮助落实IR中已有的RAW/WAR/WAW
issue edge，但不能替代`!async.token`、participant join、exact wait、memory effects或terminal completion proof，也不能
推出queue resident/full、pipeline window或任意缩短buffer lifetime。

## 7. Range and Alignment Policy

当前事实和策略：

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
- 64KB 只是在历史 parallel allocator 中出现过的 page/color 候选粒度；当前 register/library
  证据没有证明它对应物理 bank 或硬件 legality，不能把它固化为overlap-critical allocation的固定粒度。

SPM1 bank placement事实与策略：

- 硬件设计给出8个独立2048-bit bank、LSB interleaving，以及CT/NE/LSU/TMNOC/DTE合计
  10读6写端口；每周期最多可有16个port requests，连续burst8访问的设计目标是平均per-port约90%效率。
  提供的硬件摘录没有进一步给出same-bank仲裁、同bank每周期可完成请求数或端口/stride冲突曲线，不能从“8 banks / 16 ports”
  自行推出精确stall模型。DIDT寄存器还可限制8 bank同时连续活跃比例。这些是physical placement evidence和future cost
  calibration输入，不是新的IR语义或hard allocation legality。
- 由256B bank宽度和LSB interleaving得到目标allocator的粗粒度bank-phase working inference：对256B对齐base，
  `phase = (offset / 256) mod 8`，phase周期为2 KiB。该式只描述连续bank-line的起始相位；确切端口仲裁、
  stride访问分布、DIDT配置和冲突penalty尚未校准，不能把phase相同等价为必然stall。
- allocator只消费fixed-capacity problems并保留validated placements。MiniMalloc在现有单次搜索中遇到多个
  hard-valid offset choices时，稳定排序可用current IR可重算的coarse bank phase作末级soft preference；
  不能让phase改变hard feasible set、进行额外query，或建立独立plan/post-solve relocation。
- bank phase不进入06的fusion/spill/tiling cost，也不得造成allocation failure或改变TileRegion、retention/release/selective spill、
  region partition、DDR movement、worker/order或join。256B alignment保持不变，不引入通用2 KiB或64 KiB对齐；
  accepted IR仍只保存offset，不新增color attr或bank side table。

按本项目1 GHz、并额外假设每个bank每周期贡献一个2048-bit传输做粗略上界算术，raw service envelope为
`8 * 256 B * 1 GHz = 2.048 TB/s`；`2.048 * 0.9 = 1.8432 TB/s`只是再假设per-port效率目标可聚合为全bank利用率的条件说明，不是
operating point或sustained guarantee。hardware-cost point model只取其中一个SPM1 bank的
`256 B * 1 GHz = 256 GB/s/tile`作为保守nominal service prior，不假设8 bank利用率可聚合；它带版本化assumption，
不作为lower bound、proof或allocator legality。该prior与历史SPM0/RAM_ACC的1024-bit接口不同，也不改变bank phase、
port/stride conflict和DIDT penalty仍待校准的事实。

## 8. Static Allocation Algorithm

当前hard placement由受管MiniMalloc的fixed-capacity canonical search实现。allocator只消费本stage从当前IR重算出的demand、
absolute alignment、arena range和pairwise may-overlap relation，不读取op名、workload、tile shape或旧plan历史。

1. 从 instruction-level IR with unplaced Wafer-tagged memref values 收集 `BufferDemand`。

2. 生成path-qualified lifetime segments并通过`lifetimesOverlap`建立精确conflict relation：

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

3. 用synthetic half-open activity slots把任意conflict graph无损编码给MiniMalloc。adapter先按
   `stableOrdinal`规范化正size demand，再对每个conflict connected component构造确定性greedy
   edge-clique cover：每个slot的成员必须两两冲突，每条原conflict edge必须被至少一个slot覆盖，
   non-edge绝不得共享slot。构造后独立fail-closed验证这三项，因此clique合并不会改变原pairwise
   may-overlap语义，同时能让solver直接看到clique capacity lower bound。triangle-free graph的worst case仍可能需要
   一edge一slot；adapter不求解NP-hard的最优clique cover，也不按workload/op/shape特化。同一demand的一条连续
   地址区间覆盖其全部activity slots，中间非活动slot是MiniMalloc gap。不能把multi-segment/path relation
   压成convex hull；无segment且无conflict的demand仍需offset，但使用component-local private slot。zero-byte
   demand不占地址范围，由adapter按absolute alignment直接安置；受管core保持正高度buffer前提，
   两类边界都不向core传empty lifespan/rectangle。

4. 以absolute address求解：nonzero SPM base由fixed prefix range表达，不能先求相对offset再无条件加
   base，否则会破坏absolute alignment。每个conflict component使用一个offset=0、仅覆盖该component
   连续slot区间的prefix；不同prefix在activity time上不相交，因而既约束所有有效placement大于等于
   arena base，又不把本可独立求解的component错误连成一个partition。reserved/fixed range在真正进入target
   policy后以同类显式约束表达。

5. 使用稳定、有限且宽松的全局search work budget。normal-call默认值为
   `min(2^24, 2^21 + 64 * demand_count + 16 * conflict_count)`；budget只按确定性search node消耗并跨
   partition/preordering共享，不以短wall-clock timeout决定语义。显式budget override只是owner-private
   offline/test control，不是用户级pass选项或IR fact。

6. typed outcome只允许：

- `Feasible`：MiniMalloc返回完整placement且通过Wafer独立validator；直接采用。
- `ProvenInfeasible`：在给定node budget内完成完整搜索并证明完整hardware arena无解，可映射
  `capacity_overflow`。
- `ResourceExhausted`：全局work budget耗尽；原样返回，不运行fallback，不得误报capacity。
- invalid input、checked arithmetic overflow或第三方invalid placement是typed contract/internal error。

7. 对MiniMalloc结果独立验证：placement coverage/唯一性、absolute alignment、checked end/range、
   conflict pair不得byte overlap。

8. 接受步骤7验证后的唯一placement；从accepted offsets重算actual high-water/headroom和粗粒度bank phase。
   solver的既有canonical offset ordering可以用bank phase作稳定末级soft key；该key不删除hard-valid
   choices，不执行post-solve relocation、额外query或以phase换算latency。

9. owner继续执行range/end verification：

- 检查 `base + allocated_size`。
- 检查 wrapper begin/end range。
- 检查 bool bitpack、stride byte count、Cx/NCx padding。

10. 只返回typed result；search trace、work count、conflict encoding和bank phase diagnostic都是
    invocation-local diagnostic，不写IR。

固定硬件容量下的feasibility是本stage唯一hard legality primitive。actual high-water和headroom从已经接受的
placement重算并返回owner作容量诊断；它们不替代06从actual IR重算的DDR、GS、compute、completion与tile-utilization成本，
09自身不运行planning objective或改变selected dataflow。

### 8.1 Packing Backend 与 Fixed-Problem 边界

fixed lifetime、physical size、absolute alignment、arena和pairwise conflict在进入本层前已经冻结。production packing
backend只使用MiniMalloc：它只解决这份固定问题，不选择tile、layout、TileRegion、retention/release、selective spill、worker/order或completion，也不把
packing结果反向改写成这些选择。未限制search budget时，当前受管core对adapter表达的fixed-capacity问题保持complete；
production的有界budget必须保留`ResourceExhausted`，不能伪装成不可行。选择MiniMalloc是基于专用搜索、确定性、
三态failure和轻量集成的工程结论，不声称它对所有实例都比通用solver更快。

每个finalized Tile Instr program从完整current IR派生一个或多个fixed allocation problems；每个root被all-and-only
一个problem覆盖，可能并发的regions进入同一coexistence/conflict约束，已证明不重叠的roots可复用3 MiB地址范围。
`Feasible`证明当前固定lifetime/size/alignment/conflict问题能装下；accepted placement的actual high-water不是全局最优证明，也不是继续
二分arena end的理由。plan间真正有意义的working-set差异已经由06选择的tile、buffering和
TileRegion、retention/release、spill/recompute反映在派生lifetime、demand graph与硬capacity结果中；allocator不再用最多8次quality probe放大终态成本。

生产不引入ILP/CP-SAT，也不设置离线小图oracle。仓库内exhaustive/property tests只验证adapter、三态、alignment、
validator和fixed-capacity结果；若真实捕获实例持续`ResourceExhausted`，可以一次性外部诊断solver/backend行为，
但结果不进入依赖、fixture、cost、IR或acceptance。

### 8.2 Complete-Candidate Actual Placement Boundary

对每个complete candidate的finalized Tile Instr program调用本节fixed-capacity路径；每份fresh problem得到：

- `Feasible`：完整placement经独立validator接受，offset可原子写入candidate IR；
- `ProvenInfeasible`：在owner budget内完成搜索并证明硬件arena不可行；
- `ResourceExhausted`：原样返回，不运行其它packing算法；
- invalid input、overflow或invalid solver result：contract/internal failure。

`Feasible` placement的actual high-water/headroom从validated offsets重新计算并返回06作diagnostic与hard-capacity
余量记录，不进入独立quality probe或主Pareto。placement必须原子apply到未提交的candidate complete TileModule set，并重新运行全部
offset-dependent descriptor/range及后续candidate gate。placement以`wafer.spm.offset`进入actual IR，其临时solver state不跨stage缓存，
也不是09发布的proof schema。09不能隐式引入或改变region partition、tile、implementation、encoding、residency、route、
buffering或执行顺序。

candidate construction、bufferization或lifetime/effect变化后，旧placement和所有offset-dependent descriptor/range/cost结果均失效，
必须在同一未提交Card transaction中从current IR重建`StaticPackingProblem`集合；不能原地修补或复用solve结果。只在全部
range/alignment/conflict gate通过后提交`wafer.spm.offset`。每个candidate从自己的current IR重建problem；不得与plan-side problem比较，
也不得复用另一个candidate的placement。


## 9. Failure Feedback

当前structured failure reasons包括：

- `capacity_overflow`（仅完整hardware arena已证明不可行）；
- `packing_search_exhausted`；
- `invalid_packing_result`；
- `invalid_spm_range`；
- `alignment_unsatisfied`；
- `unsupported_layout_conversion`；
- lifetime/alias/control-flow unsupported；
- missing或unsupported async/local/DTE completion；
- unsupported SPM scope nesting。

`capacity_overflow`只来自actual fixed-capacity problem的`ProvenInfeasible`。`packing_search_exhausted`只表示资源耗尽，不是capacity事实。
candidate failure擦除未提交top-level TileModule subtrees；带完整owner relation的capacity rejection可反馈外层controller，其它failure按typed合同处理。
allocator不返回tile、layout、route或retention/release repair recipe，
也不保留partial offsets。

## 10. Complete Candidate TileModule set 到 Actual Result

SPM stage嵌入candidate Card transaction：

```text
complete candidate TileModule set with typed IR
  -> function-boundary and region-local bufferization / layout
  -> movement and physical TileRegion boundary closure
  -> current Tile execution structure / rotating slots
  -> execution-structure-closed TileRegion legalization to Instr
  -> verify actual stage chunks / movement / physical buffers / issue order / completion obligations
  -> relation-backed redundant physical transfer normalization
  -> current Instr worker/order and dependency-driven completion reconstruction
  -> completion-closed canonical unplaced Instr handoff
  -> fresh BufferDemand / lifetime / effect collection
  -> fixed-capacity SPM solve + independent placement validation
  -> atomic offset apply to candidate IR
  -> fresh range / descriptor / completion-consistency revalidation / cost gates
  -> derive and atomically validate DDR placement domains / transport / ABI acceptance
  -> Accepted DeviceExecutable + actual cost, or typed rejection/failure
```

Candidate IR必须已经把TileRegion、retention/release、spill、encoding、transfer、execution structure和instruction sequence显式物化；
SPM owner只从completion-closed Instr重算lifetime与demand。Completion在完整worker assignment形成后由上游current-Instr stage fresh构造；
bufferization、Instr conversion、order或completion改变root、alias、effect或lifetime后，必须在进入本leaf前完成并使旧problem失效；
memory leaf不得再次运行这些变换。旧offset和cost不得复用。packing只允许
对lifetime/conflict证明为不冲突的roots复用range；
accepted offsets形成后由独立physical-alias verifier复核。若复核失败，表示packing结果无效或上游proof不一致；未提交subtree
被擦除并返回internal failure，不能形成exact no-good。offset本身不改变lifetime/completion，allocator不得
就地插join、改slot或反复packing修复。

function-boundary bufferization可能新增或删除buffer/movement，所以它与region-local bufferization在movement、execution structure、
TileRegion relation listener与Instr conversion之前完成；
之后的current-Instr completion stage闭合join/wait，再把finalized
Tile Instr program从final instruction IR派生problems并fresh recost；任一Tile未被接受都擦除整个未提交selected TileModule set，
不产生Tile-local survivor，也不返回planner。actual placement完成后从current explicit DDR arenas/domains重建并原子验证
DDR/transport/resource事实，不消费SPM-side compatibility signature。


## 11. Runtime / ABI Boundary

Wafer-tagged memref是physical-dataflow/SPM planning阶段的tile-local buffer value。SPM bufferization
接受selected realization和allocation后，不再新增独立placed memref/explicit descriptor IR层。
DeviceExecutable verification前的Tile executable validation从instruction IR中的memref use-def、view relation、
`wafer.spm.offset`、`wafer.ddr.offset`、arena/placement和
`computeWaferPhysicalTensorInfo(memrefType)`重算resource role/range/scope/alias与entry slot需求，
验证plan一致性、补全range/capacity并materialize typed C++ entry bindings，再随DeviceExecutable atomic apply写入执行对象；
不能在此新造state consistency、arena/residency policy或ResourceId。

target-codegen之后只能结合accepted instruction IR与这些executable bindings派生address/range/
stride参数。package从accepted executable序列化ProgramTensor/TargetTensor、ports、entry-local requirements与
`TileEntryArgument`；`RuntimeInvocationPlan`只映射checked memory ranges，board execution再取得实际base。
package/runtime不得再扫描raw memref/offset恢复role、alias或lifetime。

派生规则：

- compact `Tensor/NTensor`：address range 由 accepted base offset、logical shape、dtype 和 compact
  layout footprint 重算；不写第二份 range attr。
- aligned `Cx/NCx`：用 `computeWaferPhysicalTensorInfo(memrefType)` 展开 physical extent、padding、
  storage bytes 和 begin/end range；这些是 verifier/codegen 派生值，不作为新 IR 事实源。
- 如果目标 movement/compute instruction 需要 range、logical-to-physical mapping、layout family、
  stride 或 packet field，ABI/packet emission 在 very-late lowering 中从当前 IR 派生对应参数；
  不能把它们作为独立access descriptor在主线IR中传递。

这个 boundary 不重新选择 layout，也不重新移动 materialization cut。若某个 Wafer-tagged memref
无法由 accepted facts 派生合法 address/range 参数，说明前面的 layout/SPM plan 不合法，应返回
planner，而不是让 LLVM lowering 才失败。

## 12. IR 表达

SPM bufferization 后，IR 应显式表达：

- Wafer memory attr：address space + physical layout marker。
- buffer ownership / alias relation。
- movement / materialization / compute / sync op。
- necessary effect / wait / barrier。

target-codegen materialization后，bank/color、worker、queue、packet field只在能验证它们的lower-level IR或
conversion-local value中出现；runtime physical base/handle只存在board invocation或后续`PreparedExecution`。主线instruction IR不新增
placed memref、flat backing memref 或 explicit descriptor 事实源。
上游structured program或调度container不携带这些字段。

`#wafer.memory<space, layout>` 在 accepted instruction IR 中仍是统一语义：`spm` 表示 tile-local
SRAM，`ddr` 表示 device/global DDR address domain。RDMA/WDMA verifier 用 source/destination
address space 检查方向；DDR memory planning 用 `ddr` 继续关联 view/range、compiler-managed DDR
planned range、constant storage/residency 和 declared arena resource。selected `DeviceExecutable` formation前从同一IR验证并补全
accepted range/arena relation；formation后package/runtime只消费
该唯一owner，不在launch metadata中再次关联或恢复range。

不要把 `wafer.tile.region` body 已经表达的执行结构复制成全局 allocation plan attr。

### 12.1 Accepted SPM Offset Fact

当前SPM planning在instruction-level IR上使用`wafer.spm.offset` op attr表达接受的offset fact。
该attr挂在定义SPM buffer value的`memref.alloc`上，值为
`#wafer.spm_offset<offset>`：

- `offset` 是allocation root在当前Tile SPM arena中的byte offset。

以下事实不写入 attr，因为它们可由当前 IR 或 target policy 稳定重算：

- `size` 来自 `computeWaferPhysicalTensorInfo(memrefType).physicalBytes`，不是 logical compact bytes。
- `alignment` 来自 target policy 和 `memref.alloc` alignment；planner 用它验证 offset，但 accepted
  IR 不复制该输入。
- bank-line span按256B粒度由`[offset / 256, ceil((offset + size) / 256))`重算；对256B对齐base的
  working bank phase按`(offset / 256) mod 8`重算。二者都不是accepted attr。

每个Tile的SPM window是Tile-local physical arena。每个finalized Tile module的全部roots从current IR派生fixed
allocation problems并在3 MiB window内all-and-only规划；entry含一个或多个non-nested physical regions。SPM memref/root/alias不得跨region boundary；
region result若是data只能发布DDR value/view，其root可以是function
external或compiler-managed materialization。region内部的selective spill/store结束目标root，matching reload建立distinct root，且
不影响其它live root。任意SPM raw escape、仍访问前一region roots的pending work或无法解析的provenance结构化拒绝。输出仍保持
offset-only，size/alignment/lifetime/bank phase从accepted IR重算。

当前dataflow边界：

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
  `scf.if` result合并时保留各自origin和path condition。NCC participant join不能消费该completion；SPM当前在DTE
  owner proof之前保守拒绝所有loop-carried async token。
- 通过`MemoryEffectOpInterface`在Compute/Movement custom resource上产生effect的本地issue进入pending
  local issue集合，其全部SPM read/write effect按typed worker进入pending local access集合；matching
  participant join只清除覆盖的worker。same-worker exact RAW/WAR/WAW后继可接管同一root的有序访问，DTE wait
  不能消费NCC状态。
- `wafer.tile.region` exit只验证没有仍访问其SPM roots的pending generic task、DTE token或NCC participant；与这些roots无关的
  typed pending state可继续到真实observer或entry terminal。region结构本身不插wait/join。普通traversal/loop/spill点不执行terminal validation；可能zero-trip的loop内completion不能覆盖loop外pending状态，
  safe same-worker NCC pending set可跨backedge，但SPM仍拒绝loop-carried async token。
- rejected/temporary offset、search trace、cost estimate 和 repair suggestion 仍是 analysis，不写入
  IR。

实现边界是instruction IR已支持的structured `scf.if` / `scf.for`、single-block `scf.yield`和
instruction-level async token use。path condition使用按`uint64_t` decision id排序的sparse decision set，
不再存在64个branch/loop decision point上限；当前回归覆盖96个branch及130个decision的conjunction/subtract。
未结构化/多block CFG、decision id域或编译资源耗尽，以及未来显式must-alias group仍需结构化失败，不能靠
线性op顺序、名字或旁路协议恢复。

SPM memory planning使用经典静态memory planning的离线模型，而不是把alloc event顺序直接当作allocation
顺序。算法分两层：

- lifetime analysis 仍由当前 IR 的 SSA、region、path condition 和 async token use 重算，得到可同时
  发生的 lifetime segments。
- packing只由受管MiniMalloc fixed-capacity canonical search完成；path-qualified lifetime先转换成无损pairwise
  conflict relation，再由经fail-closed验证的deterministic edge-clique cover和component-local nonzero-base
  prefix适配给solver。solver全局work budget耗尽保持`ResourceExhausted`；不存在first-fit、best-fit或其它fallback allocator。

solver的accepted placement由Wafer独立validator复验。rejected/temporary offset、solver work和
conflict encoding都保持为analysis/debug统计，不写入`wafer.spm.offset`。后续若评估其它graph-coloring/
interval-coloring backend，必须保持same input/output IR contract和三态结果，只改变analysis/search；不建立常驻
ILP/CP-SAT backend或oracle。
SPM1 bank phase只属于allocator内部offset选择，永远不得升级为hard constraint或改变selected dataflow。
未来若硬件暴露新的独立legality事实，必须另立合同，不能把它追认成phase legality。

`wafer.spm.offset`只保存accepted offset，不保存size、alignment、bank-line span、bank phase、lifetime或搜索trace。
失败原因仍通过 pass diagnostic 返回，
不写进IR；rejected/temporary offset、canonical solver过程和repair suggestion都保持为analysis。

## 13. Verifier

SPM / tile-region verifier 至少检查：

- buffer range 不越过可用 SPM window 和 reserved range。
- begin/end range 与 physical layout size 一致。
- Cx/NCx、C0 tail/fold、256B padding、NHWC bank alignment、bool bitpack 计算一致。
- op operand/result 的 Wafer memory attr 满足对应 op verifier。
- must-alias relation 的读写顺序合法。
- may-reuse buffers 的 lifetime 不重叠，或由明确 wait/barrier 收口。
- async buffer在matching completion前不能复用；仅same-worker exact RAW/WAR/WAW ordered successor可接管访问。
- host-visible writeback和communication boundary有明确participant join/exact wait/sync。
- 每个有assigned work的physical `wafer.tile.module`包含一个或多个non-nested `wafer.tile.region`；nested region或typed opaque clobber输入拒绝。
- physical `wafer.tile.region`不得传递SPM buffer/root/alias；所有data I/O必须是DDR。region内selective spill与cross-region cut必须
  完整表达各自store/completion/load；region exit只验证仍访问其roots的pending work已由显式typed completion清空，
  不因region结构插入或执行wait/join；complete candidate TileModule set coverage确保allocation problems
  all-and-only覆盖每个root。
- bank phase soft preference不参与verifier legality；verifier只重算accepted offset的range/alignment/
  overlap。任何因phase冲突拒绝hard-valid placement、插入spill/region/join或依赖bank attr的实现均违反本合同。
- tile-region只允许sequential func/scf.if/scf.for ownership；active/async/parallel scope中的unknown/external/
  indirect或可能重入SPM的call失败。没有SPM arena/effect的closed scalar direct callee可在resident value live时
  穿过；可能执行tile-region的defined callee、external/unresolved或indirect call因缺少interprocedural
  arena/resource summary而fail closed。SPM memref即使先擦成tensor/generic memref也不能经`wafer.tile.yield`、
  raw metadata或无origin的tracked cast/adapter逃逸scope。
- safe pre-existing loop recurrence在memory-planned named pipeline中可让same-worker ordered pending跨backedge，
  不要求结构性join；loop body创建fresh allocation再作为recurrence result携带时，以
  `unsupported_lifetime_alias`拒绝而不是给所有动态实例同一地址。
- `busytable` 只参与 target capability/legality/cost 查询，不被当作 completion 或 alias proof。
- physical footprint、begin/end、descriptor range、offset addition 和 ABI width narrowing 统一调用 shared
  physical geometry/range/narrowing verifier；禁止 silent truncation 或 consumer-local 重算分叉。
- launch/resource、target LLVM 和 runtime adapter 阶段不能要求额外 placed/access descriptor fact；compact 和 Cx/NCx buffer
  的 address/range/stride 参数必须由 `computeWaferPhysicalTensorInfo`、accepted offset facts、
  allocation range 和 op verifier 一致推出。

## 14. 与 Physical-Dataflow Planning 的关系

全局output DAG见`tasks/01-architecture.md`，planning合同见`tasks/06-physical-dataflow-synthesis.md`。Pre-memory stages不签发SPM合法性；
本层为每个complete candidate TileModule set中每个Tile的`#wafer.memory<spm, *>`分配actual offset或返回typed rejection：

```text
complete candidate TileModule set
  -> actual compute form / encoding / transfer / TileRegion and retention/release already in typed IR
  -> fresh instruction storage / effect demands
  -> fixed-capacity SPM evaluation and validated placement
  -> fresh offset-dependent instruction / descriptor / range / completion gates
  -> actual cost
  -> candidate DDR exact evaluation / transport / ABI coordinator
  -> Accepted DeviceExecutable, or typed rejection/failure with no committed subtree
```

Region、retention、tile/loop、implementation、encoding、transfer、spill/recompute、buffering和order已由上游在current IR中表达；
typed opaque clobber输入直接拒绝。Allocator不创建或修改boundary、不建议或修改choice，也不按名字或case恢复语义；
外层controller比较Accepted results。
NCC participant join、communication wait和DTE completion是不同event，lifetime analysis只能按各自typed
worker/effect/token合同处理；NCC completion只接受typed participant join。


## 15. 后续扩展

这些机制有价值，但不进入 当前主路径。进入条件必须明确：

- 多 pool allocation：当 ordinary pool、communication staging、runtime-visible buffer 的 reserved
  range 和 lifetime 约束稳定后引入；在此之前用单 pool + reserved range 更容易验证。
- linear scan allocator：当fixed-capacity MiniMalloc compile work在真实problem上持续不可接受时才评估；不得仅因
  accepted high-water不是最优就替换backend。
- 其它graph-coloring / interval-coloring backend：只有invocation-local统计证明当前fixed-capacity core在通用
  conflict graph上持续产生不可接受compile work或大量资源耗尽，且新backend保持相同三态和独立validator合同时才评估；
  不能仅因单个case placement更紧凑而替换默认路径。ILP/CP-SAT也只在这个触发条件下用于真实捕获实例的一次性
  诊断，不设常驻oracle。
- allocator内部不引入repair loop；任何会改变tile shape、dataflow cut、layout、movement或buffer的选择都返回
  physical-dataflow selection，避免allocator成为隐藏scheduler。
- PMU驱动bank penalty：当board profiling能稳定解释blocking time、port/stride和bank phase关系后，校准
  soft preference的penalty与DIDT条件；不改变documented LSB interleaving，也不把penalty升级为hard legality。
- worker-local / runtime-visible / communication-only pool：当对应 dialect 和 verifier 已能表达
  ownership、visibility和typed completion边界后引入。

## 16. 参考材料

- SPM1 hardware facts：`docs/wafer-hardware-instruction-set-and-programming-model.md`。
- MLIR Bufferization / One-Shot Bufferize：<https://mlir.llvm.org/docs/Bufferization/>
- XLA BufferAssignment：<https://openxla.org/xla/hlo_to_thunks>
- TVM USMP：<https://discuss.tvm.apache.org/t/rfc-unified-static-memory-planning/10099>
- TFLM Memory Planner：<https://proceedings.mlsys.org/paper_files/paper/2021/file/6c44dc73014d66ba49b28d483a8f8b0d-Paper.pdf>
- Google MiniMalloc：<https://research.google/pubs/minimalloc-a-lightweight-memory-allocator-for-hardware-accelerated-machine-learning/>
- Bounded Memory Scheduling：<https://www.cs.rice.edu/~zoran/Publications_files/PACT2014-BMS.pdf>
