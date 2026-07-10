# Wafer Target Environment, Topology and Execution Mesh Design

状态：本轮长期边界合同已收敛；实现状态以`tasks/progress.md`为准。范围：target environment、target topology、SPMD execution mesh 和 post-SPMD
launch/transport projection。

本文定义 compile-time target legality environment、deployment topology snapshot、SPMD-visible execution
mesh，以及 distributed execution instance 到 Wafer physical tile/transport binding 的投影边界。SPMD 不能
在 abstract full mesh 上先切分，再由后段补坏 tile；它必须消费已经由 target environment 验证并从
topology 中选出的 valid execution mesh。post-SPMD projection 只能绑定已存在的 distributed component /
partition/replica/rank-class 事实，不能重新切分 tensor 或默认到 rank 0。

本文在pre-commit projection中提到的rank class仅指distributed层的target-independent prerequisite class。
projection以canonical `ExecutionInstanceId`和`tasks/01`定义的`CandidateExecutionEntry`为主键；final `RankClassId`由whole-variant commit
结合projection/transport等价性决定，projection不能反向依赖尚未产生的final class。

本文依赖：

- `tasks/03-shardy-spmd.md`
- `tasks/06-group.md`
- `tasks/13-communication.md`
- `tasks/15-launch-runtime-package.md`
- `docs/wafer-hardware-instruction-set-and-programming-model.md`
- `docs/tx8-deps-reverse-engineering/firmware-kuiper-runtime-hardware-analysis.md`
- MLIR Data Layout and Target Interface: <https://mlir.llvm.org/docs/Dialects/DLTIDialect/>

## 1. 目标和非目标

目标：

- 用 `wafer.target.environment` 保存 target revision/ABI 和 compiler legality 所需 capability，并以稳定
  fingerprint 参与 executable compatibility/cache identity。
- 将 capability、topology snapshot 和 calibration profile 分成三个 owner：capability 决定 legality，
  topology 决定可用 endpoint/connectivity，calibration 只影响 cost/candidate ordering。
- 在 SPMD 前 materialize/verify target environment、规则 topology、unavailable endpoint 和 derived
  connectivity，并从中选择 SPMD 可见的 valid execution mesh。
- 在 complete candidate variant 上、whole-variant atomic commit 前，把 distributed component /
  partition / replica / rank class 投影到 physical endpoint、entrypoint/block 和 stage-accepted transport
  assignment，形成 pinned 或 relocatable launch/transport projection。
- 为 communication、target LLVM、package/runtime 提供可验证且自包含的 physical binding。
- 在不能得到合法 execution mesh 或 endpoint projection 时，把原因反馈给 planner 或 compile driver。

非目标：

- 不做 Linalg tiling、group fusion 或 root tile shape search。
- 不分配 `#spm` offset、DDR runtime allocation object 或 workspace slice。
- 不选择 `Cx/NCx` physical layout。
- 不重新切分 tensor，不保存 per-rank tensor slice table。
- 不选择 ring/tree collective algorithm，也不把 DTE allocator/search trace 写进 environment/mesh；本层只
  接受并验证 communication owner 在 candidate clone 中产生的 stage-accepted transport assignment。
- 不让 calibration profile、PMU estimate 或 provider 名称建立 legality。
- 不允许 runtime 根据当前空闲 tile 重做 sharding、rank class、memory plan 或 transport plan。
- 不把 HPGR/KMD/legacy runtime API 细节写成上层 endpoint 语义。

## 2. Pipeline Contract

### 2.1 Pre-SPMD Environment And Mesh Selection

```text
Pipeline position:
- Upstream artifact / IR:
  target descriptor、target capability snapshot、deployment topology/availability snapshot、optional
  calibration profile，以及 verified frontend program 的 logical mesh request / symbolic constraints。
- Current stage responsibility:
  materialize / verify `wafer.target.environment`、`wafer.target.topology` 和 execution mesh rank domain。
  默认 `all_available` policy 从 topology 派生所有 available endpoint；显式 override 才保存 endpoint
  tuples。验证 required capability、rank count、axis role/product、topology membership、availability、
  connectivity 和 explicit physical tile uniqueness。calibration 只供 mesh candidate cost 使用。
- Output artifact / IR:
  `wafer.target.environment` legality capability + fingerprint；`wafer.target.topology` deployment grid /
  availability snapshot + digest；`wafer.execution.mesh` accepted logical rank-domain fact，引用 environment /
  topology 并保存 axes/roles/shape 和 endpoint policy。
- Downstream consumer:
  Shardy MPMD / SPMD partition 和 post-SPMD launch/transport projection。
- User-level driver / named pipeline:
  production 主线由 `wafer-opt --program-pipeline=stablehlo-to-executable` 或等价 driver 在 SPMD 前
  materialize environment/topology/mesh；现有 `stablehlo-spmd*` pipelines 是内部/分阶段 debug 入口，
  default Shardy seed 从 execution mesh rank count / axes/roles 取数。
- Explicit non-goals:
  不重新做 group/candidate/tile shape/layout/SPM/DDR planning；不生成 DTE route、target call、packet、
  object、package、runtime handle 或 physical address；不形成 post-SPMD block/transport binding；不靠
  tensor/axis 名字恢复 shard 或 parallel role。
- Completion gate:
  named pipeline 能重放 target descriptor/capability/topology import -> environment fingerprint -> valid
  execution mesh -> distributed program。environment/mesh 成为唯一 pre-SPMD legality/rank-domain source；
  verifier 拒绝 capability/ABI mismatch、rank count/axis product mismatch、unavailable/duplicate/out-of-topology
  endpoint、disconnected component 和 calibration 被当作 legality 的输入。
```

### 2.2 Post-SPMD Launch And Transport Projection

```text
Pipeline position:
- Upstream artifact / IR:
  verified target environment/topology/execution mesh、globally coherent distributed-program variant、
  完整 candidate clone 中的 static-rank instruction/resource facts、accepted SPM/DDR offsets，以及
  communication owner 产生的 stage-accepted transport/resource assignment。
- Current stage responsibility:
  验证全部 component/partition/replica/canonical execution-instance coverage和distributed prerequisite
  class refs，把 logical execution instances 绑定到 physical
  endpoint、entrypoint/block、parameter/state shard 和 transport resources；声明 projection 是 pinned 还是
  relocatable，并生成确定性 digest / relocation records。
- Output artifact / IR:
  stable `ProjectionSetId`标识的accepted launch/transport projection set：引用environment/topology/mesh/
  distributed variant/candidate-entry semantic digest，
  为每个 execution instance 给出 endpoint、component entrypoint、block id、typed resource binding 和
  collision-free transport assignment。它在 candidate clone 内是 global-commit 输入；只有 whole-variant
  atomic commit 后才成为 executable/package fact，且不复制 tensor sharding 或 memory plan。
- Downstream consumer:
  whole-variant executable verifier/atomic commit；commit 后由 target LLVM、device object composition、
  self-contained package manifest、runtime launch/completion aggregation 消费。
- User-level driver / named pipeline:
  主线 compile driver 在 communication transport acceptance 后、whole-variant atomic commit 前自动执行；
  不要求用户手动串 projection pass。
- Explicit non-goals:
  不重新运行 Shardy/SPMD、改变 component/rank class、选择 dynamic shape variant、重做 layout/SPM/DDR、
  选择 collective algorithm，或让 runtime 根据 provider state 修补 illegal projection。
- Completion gate:
  同一真实 distributed program 的 complete candidate 先生成 projection，再与 memory/transport/event/ABI
  facts 一起通过 atomic commit。pinned case 只接受完全匹配的 environment/topology/mesh digest；relocatable
  case 只通过 compiler-emitted relocation slots 绑定到有限、已验证 projection。缺 rank/component/resource、
  transport 冲突、stale topology、错误 variant 或未填 relocation 必须拒绝整个 variant，不能留下部分提交。
```

## 3. 输入和输出

输入：

```text
pre-SPMD:
  target descriptor + target capability snapshot + topology snapshot + optional calibration profile
  verified frontend program + requested logical axes/roles

post-SPMD:
  distributed program + complete candidate static-rank/resource facts + accepted memory offsets
  + stage-accepted communication transport assignment
  wafer.target.environment + wafer.target.topology + wafer.execution.mesh
```

输出：

```text
wafer.target.environment
  + target/revision/ABI and legality capabilities
  + required feature/limit/errata facts
  + canonical capability fingerprint

wafer.target.topology
  + regular card/tile grid
  + card interconnect kind
  + unavailable endpoint exceptions
  + deployment snapshot digest

wafer.execution.mesh
  + logical mesh axes / roles
  + logical mesh shape
  + endpoint policy
  + optional explicit logical rank -> physical endpoint coordinate
  + target environment / topology reference

accepted launch/transport projection
  + pinned | relocatable mode
  + distributed component/rank-class/artifact coverage
  + endpoint/block/resource/transport binding
  + projection and relocation digest
```

accepted environment/topology/mesh/projection 如果影响 legality、codegen 或 runtime binding，必须进入 IR/artifact；
mesh search trace、rejected embeddings、calibration score breakdown 是 analysis，不写入长期 IR。

## 4. Physical Topology And Tile Model

Wafer physical topology 由规则 card grid 和每张卡内规则 tile grid 组成。当前目标事实不是任意
tile graph dump，而是规则拓扑加少量例外：

```text
card_grid:         [card_rows, card_cols]
card_interconnect: mesh | torus
tile_grid:         [tile_rows, tile_cols]
unavailable_tiles: [card_y, card_x, tile_y, tile_x] tuples
```

已知事实：

- 单卡 `4 x 4 = 16 tile`。
- 32 卡服务器可以抽象成 `4 x 8` card mesh，但实际可用卡、坏卡、PG tile 必须来自 deployment
  topology/availability snapshot。
- 单 tile SPM `3MB` 属于 `wafer.target.environment` capability；NoC/C2C bandwidth/latency 属于带
  provenance 的 calibration profile。二者都不是 topology ownership。

Card/tile adjacency 由规则 grid 派生，不能以 `links` 边表重复保存。单卡 tile mesh 使用同 card 内
`tile_y/tile_x` 四方向邻接；跨卡 C2C 使用 `card_interconnect` 和 `card_grid` 派生 card
四方向邻接，`mesh` 边界不 wrap，`torus` 边界 wrap。若 driver 返回 PG/bad tile 或跨卡 topology
变化，只改 `wafer.target.topology` 或对应 import/rewrite pass。

默认 execution mesh 使用 topology 中所有 available endpoint。显式少用 tile 只作为 override：
用于 debug、bring-up、小 workload 对照、资源隔离或非默认 endpoint override。当前实现优先恢复单卡和
小规模 multi-tile；跨卡 topology 仍应保持可表达。跨卡 route、C2C cost 或 runtime completion
尚未完善时，应形成 mesh/runtime 恢复任务，不能反向要求 SPMD 在无效 mesh 上切分。
当前 `wafer-opt --program-pipeline=stablehlo-spmd* --execution-mesh-ranks=<n>` 只是 program driver
侧生成 explicit `wafer.execution.mesh` 的调试入口，不是 SPMD seed pass / named pipeline 的
`tile-count` 旁路。

单 tile、单卡多 tile 和多卡多 tile 都使用同一个抽象：

```text
logical rank -> physical endpoint coordinate
physical endpoint coordinate -> derived adjacency / availability
```

单卡只是 topology 里的可用 card 维度为 1，多卡只是 topology node 覆盖多个 card；planner 和
verifier 不分裂成两套语义。

## 5. IR Representation

长期合同保留四个相互引用但职责不同的 IR/artifact 事实源，避免 capability、topology、rank domain 和
projection 重复：

```text
wafer.target.environment: target/revision/ABI + legality capability/limit/errata + fingerprint
wafer.target.topology: regular card/tile grid + card interconnect kind + unavailable endpoint exceptions
wafer.execution.mesh:  distributed rank domain + logical mesh axes/roles/shape + optional explicit endpoints
launch/transport projection: candidate distributed instance -> entry/endpoint/block/resource/transport binding
```

calibration profile 是独立 analysis input：它记录带 provenance 的 bandwidth/latency/PMU/cost 参数，可以影响
mesh/candidate 排序，但不能使 capability-illegal program 变合法，也不进入 runtime compatibility 判断。
若 calibration 改变最终选择，accepted plan/projection digest 已反映选择结果；package 可以保存 calibration
digest 作为 provenance，但不能把它当 required capability。

### 5.1 `wafer.target.environment`

`wafer.target.environment` 是 module-level symbol op，保存 compiler 能解释和验证的 target legality snapshot：

- target family、hardware revision、device ABI / target CRT ABI version。
- SPM/DDR capacity/alignment/reserved range、engine/queue/resource count。
- typed DDR arena declarations：每个stable `DdrArenaId`记录memory domain、capacity、largest-contiguous、
  alignment、address width、bandwidth limit、allowed placement-domain kinds和runtime binding ABI；这些字段
  进入environment fingerprint。
- supported dtype、physical layout、instruction/packet/DTE limits、required runtime mode/feature。
- compiler 必须规避的 errata/feature flags。
- 由canonical field encoding派生的`TargetEnvironmentFingerprint`；它覆盖target ABI/capability/arena
  declarations，不覆盖deployment topology/projection，且不是用户任意字符串。

这些字段应优先评估 MLIR DLTI target system/device spec 和当前 Wafer attr 是否能承载；只有不能稳定表达、
验证或 lowering 的 Wafer-specific capability 才进入私有 op。environment 不保存 endpoint availability、
rank mapping、calibration score、runtime handle 或 provider symbol。

概念形式：

```mlir
wafer.target.environment @tx81_env {
  target = "tx81",
  revision = "r1",
  device_abi = "tx-kernel-v0",
  spm_bytes_per_tile = 3145728 : i64,
  // Conceptual typed record: @ddr0, device DDR, per-rank/per-stage capable.
  ddr_arenas = [@ddr0],
  parallel_ncc = true,
  capability_fingerprint = "<canonical digest>"
}
```

### 5.2 Topology And Execution Mesh

`wafer.execution.mesh` 不复制 topology dimensions、unavailable tile table、tile-id codec 或
connectivity links。默认 `all_available` policy 不保存 endpoint section；logical execution ordinal 到 physical
endpoint 的 view 由 `wafer.target.topology` 的规则 grid、`unavailable_tiles` 和固定 rank order
在使用点派生。只有 `explicit` policy 保存 endpoint tuples，用于少用 tile 或非默认 rank endpoint order。

长期不再单独保存 rank count、topology dimensions 或 unavailable tile table；这些由 execution mesh /
topology 派生。distributed component、partition/replica coordinate 和 rank-class membership 来自
`tasks/03-shardy-spmd.md`，不复制进 execution mesh。physical binding 只进入 accepted projection。

`wafer.target.topology` 由 deployment topology/availability snapshot materialize，并引用 compatible
`wafer.target.environment`。它保存规则 grid 和 unavailable endpoint 例外，不展开所有 tile，也不保存
tile-id codec、capability limits、calibration score 或 connectivity links。snapshot digest 从 canonical
grid/interconnect/availability 派生。

V0 形式：

```mlir
wafer.target.topology @target {
  environment = @tx81_env,
  card_grid = array<i64: 4, 8>,
  card_interconnect = "mesh",
  tile_grid = array<i64: 4, 4>,
  unavailable_tiles = array<i64: 0, 0, 0, 1>
}
```

`wafer.target.topology` 是 module-level symbol op。`card_grid` 和 `tile_grid` 都是 `[rows, cols]`；
`card_interconnect` 目前允许 `mesh` / `torus`，默认 target fact 为 `mesh`；`unavailable_tiles` 按
`card_y, card_x, tile_y, tile_x` 4 元 tuple 展开。没有出现在 `unavailable_tiles` 的规则 endpoint
默认 available。该 op 不表达 communication schedule、route choice、DTE packet、cost-model trace
或 runtime endpoint integer encoding。

实现索引：`--wafer-materialize-target-topology` 可以从默认单卡 `4x4 / 16 tile` config 生成规则
topology，也可以通过选项导入 card grid、card interconnect、tile grid 和 unavailable tile 坐标。
该 pass 只是现有 topology materialization 入口，不替代 environment materialization/verification；长期
用户 compile flow 仍由 `wafer-opt` program pipeline 组织，不要求用户手写 pass 串。

`wafer.execution.mesh` 是 SPMD 可见 mesh，必须在 SPMD partition 前存在。它引用
`wafer.target.environment` / `wafer.target.topology`，记录 logical mesh axes / roles / shape 和 endpoint policy：

- `all_available`：默认 policy。rank 数必须等于 topology 中 available endpoint 数，`endpoints`
  为空。rank->endpoint view 按 `card_y, card_x, tile_y, tile_x` row-major 顺序从 topology 派生，
  跳过 `unavailable_tiles`，不写入 IR。
- `explicit`：override policy。`endpoints` 按 logical rank 顺序保存
  `card_y, card_x, tile_y, tile_x` tuple，只用于 debug、资源隔离、小 workload 或非默认 rank
  endpoint order。

SPMD的sharding propagation、parameter shard metadata、tensor collective rank groups、communication
lowering和pre-commit `ExecutableResourceView`都应引用这个execution mesh；committed executable/package只
引用由此验证的typed rank/resource/projection records，而不是假设完整
abstract mesh 或再从其它 op 恢复 rank->tile。

概念形式：

```mlir
wafer.execution.mesh @mesh
    {environment = @tx81_env,
     topology = @target,
     axes = ["dp", "tp"],
     axis_roles = ["dp", "tp"],
     shape = array<i64: 2, 8>,
     policy = "all_available",
     endpoints = array<i64>}

wafer.execution.mesh @debug_mesh {
  environment = @tx81_env,
  topology = @target,
  axes = ["y", "x"],
  axis_roles = ["dp", "tp"],
  shape = array<i64: 2, 2>,
  policy = "explicit",
  endpoints = array<i64: 0, 0, 0, 0,
                          0, 0, 0, 1,
                          0, 0, 1, 0,
                          0, 0, 1, 1>}
```

参数/state logical shard metadata 不在本层重新生成。distributed program verifier 已检查 typed resource、
partition/replica coordinate、rank-class coverage 和 slice bounds；projection 只引用这些 resource identity /
shard relation 并绑定 executable/package typed resource slot，不维护另一套 module-level shard table。

### 5.3 Accepted Launch/Transport Projection

projection 先作为 complete candidate clone 中的 typed fact，不是 tensor IR attr bag。它与其它 candidate
facts 一起通过 whole-variant atomic commit 后，才成为 executable/package artifact。每条 record 至少引用：

- set-level stable `ProjectionSetId`、mode、deterministic member priority和canonical digest；
  `wafer.executable.variant`只引用该ID。

- target environment fingerprint、topology snapshot digest、execution mesh digest、distributed variant /
  component/canonical execution identity、distributed prerequisite class ref和static-rank entry semantic
  digest。final code artifact 在 commit 后反向绑定
  projection digest，不要求 projection 预先知道 module digest。
- canonical partition/replica coordinate、physical endpoint、component entrypoint、block id。
- typed input/output/immutable parameter/persistent state binding；persistent state 保留 alias/mutation/lifetime。
- communication owner 已接受的 channel/FSM/node/remote-buffer/event/completion assignment。

projection mode：

- `pinned`：code 或 transport assignment 嵌入 endpoint/resource fact；environment/topology/mesh/projection
  digest 必须完全匹配，任何 endpoint availability 变化都选择其它预编译 record 或拒绝。
- `relocatable`：共享 code artifact只读取compiler-emitted typed relocation slots，并携带typed union：
  `ConcreteRecordSet`是有序、有限、逐条验证的完整mapping；`FiniteTemplateSet`由一个typed coordinate/
  control-slot substitution template加有限`allowed_bindings`构成，compiler verifier必须展开并验证每个
  binding的rank coverage、availability、transport/resource conflict和digest。runtime按manifest确定性优先级
  选择compatible member并机械填slot，不能生成新origin、搜索endpoint/route或重新分配transport resource。

relocatable 只减少 code cache duplication，不放松 distributed/memory/transport legality。若 exact endpoint、
transport id 或 rank-specific control 已进入 code，artifact 必须标为 pinned，cache key 包含相应 digest。

## 6. Mesh Selection And Launch/Transport Projection

### 6.1 Pre-SPMD Mesh Selection

Mesh selection 必须发生在 SPMD 前。V0 默认 policy 使用所有 available endpoint：

1. 验证 `wafer.target.environment` capability/ABI/fingerprint，并确认 topology snapshot 引用同一 environment。
2. 从 `wafer.target.topology` 派生规则 endpoint 集，并删除 `unavailable_tiles`。
3. 按 `card_interconnect`、`card_grid` 和 `tile_grid` 派生 available endpoint adjacency，再计算
   connected components。
4. 默认 `all_available` 要求所有 available endpoint 属于一个 connected component；logical rank order
   按 `card_y, card_x, tile_y, tile_x` row-major 派生。
5. 生成 `wafer.execution.mesh`，记录 logical mesh axes/roles、shape 和 `all_available` policy，不保存
   endpoints。
6. 显式 override 可以使用 `explicit` policy 保存 endpoint tuples；这些 endpoints 必须存在、
   available、无重复，且属于同一个 available connected component。
7. SPMD partition 只消费这个 valid execution mesh 进行 Shardy MPMD/sharding propagation 和 parameter/state shard
   metadata generation。

### 6.2 Post-SPMD Projection

projection 在 distributed program、complete candidate materialization、whole-entry memory planning 和
communication transport acceptance 之后、whole-variant atomic commit 之前执行：

1. 从distributed program读取component、canonical partition/replica coordinate、distributed prerequisite
   class和全局variant，并从candidate entry取得stable execution-instance identity；
   从 execution mesh/topology 派生 endpoint view。
2. 验证每个 execution instance 恰好映射一个 available endpoint、component entrypoint 和 typed resource
   binding；同一 physical endpoint 不承载两个同时执行 instance，除非显式 time-multiplex contract 存在。
3. 消费 communication owner 的 accepted transport assignment，验证 endpoint、channel/FSM/node、remote
   buffer、event/completion 与 instance/resource coverage，不在本层重新选择 ring/tree/route。
4. rank-parametric code 使用 typed partition/replica/relocation binding；per-rank-static code 的
   specialization coordinate 必须与 projection record 完全一致。
5. 根据 code/transport 是否嵌入 physical fact 标记 pinned 或 relocatable，计算 projection digest，并把
   record 交给 whole-variant verifier；只有 atomic commit 后 target LLVM/package 才能消费。runtime 只能
   选择/绑定 committed record。

cost model 可以考虑：

- 卡内 NoC vs 跨卡 C2C 的相对代价。
- collective group 是否可以映射成局部 ring/tree。
- 每个 physical tile 的 local shard resource pressure。
- unavailable endpoint 分布导致的 hole 和 route penalty。
- logical mesh axis 到 physical graph path 的 shortest-path cost。

这些是候选排序，不是 IR contract。IR/artifact contract 只有 accepted target environment、topology、
execution mesh 和 launch/transport projection。

带洞 mesh 的 V1 算法可以在同一 IR 合同下扩展为 graph-aware logical mesh embedding：

1. 将 available topology 建成带权图 `G=(V,E)`。
2. 在 connected component 内生成 K-node connected subgraph candidates。
3. 尝试 logical mesh shape，例如 `[1,N]`、`[2,N/2]`、`[4,N/4]`。
4. 用 row-major、serpentine、BFS 或 Hilbert-like ordering 建立 rank order。
5. 对每个 logical axis 计算 collective cost；若 axis 跨 disconnected component，则 cost 为 infinity。
6. SPMD cost model 用 axis cost 选择 tensor dim -> mesh axis。

Full co-optimization（SPMD strategy、mesh embedding、endpoint projection、communication algorithm 同时搜索）
不改变上述 artifact ownership；即使未来同一 planner 联合搜索，也只提交最终 accepted environment/mesh /
distributed variant/transport/projection，搜索过程仍不进入 IR。

## 7. 与其它阶段的接口

| 阶段 | Environment / topology / mesh / projection 提供 | 本层不提供 |
| --- | --- | --- |
| frontend / SPMD | environment legality + valid logical mesh axes/roles/rank domain | physical rank default、事后修补 invalid mesh |
| distributed program | environment/mesh reference；post-SPMD projection消费 component/coordinate/rank class | MPMD formation、tensor resharding |
| candidate/layout/memory | typed capability/limit 和 endpoint-local resource context | tile shape、layout、SPM/DDR search |
| communication | endpoint/topology graph 和 projection record schema | collective algorithm、DTE resource search；只接收 accepted assignment |
| target LLVM / package | pinned/relocatable artifact、entrypoint、endpoint、typed resource/transport binding 和 fingerprints | 从文本或文件名重建 ABI/rank |
| runtime | required capability、compatible concrete projection records 和 completion aggregation domain | sharding、placement、transport 或 memory replanning |

Environment / topology / execution mesh 不把 DTE route 或 runtime launch API 写进上层。Communication
lowering 可以基于 accepted endpoint view 选择 protocol/resource；只有 accepted assignment 进入 projection。

## 8. Verifier

`wafer.target.environment` 必须检查：

- target family、revision、device/CRT ABI 和 required capability fields 完整且使用 typed enum/field，
  不接受 opaque string bag。
- capacity/alignment/resource/packet limit 为正且相互一致；required feature 与 errata 不冲突。
- `DdrArenaId`唯一，arena capacity/largest-contiguous/alignment/address width/allowed placement domains合法，
  runtime binding ABI与target revision兼容；environment fingerprint覆盖完整arena declarations。
- capability fingerprint 与 canonical field encoding 一致；不得由输入 metadata 任意指定。
- calibration profile 不参与 legality 或 runtime compatibility predicate。

`wafer.target.topology` 必须检查：

- 引用的 target environment 存在且与 snapshot target/revision compatible。
- `card_grid` 和 `tile_grid` 都是两个正整数。
- `card_interconnect` 是 `mesh` 或 `torus`。
- `unavailable_tiles` 按 `card_y/card_x/tile_y/tile_x` 4 元 tuple 展开。
- 每个 unavailable endpoint 都在 grid 范围内，且没有重复。
- topology snapshot digest 与 canonical grid/interconnect/availability 一致。

`wafer.execution.mesh` 必须检查：

- 引用的 target environment/topology 存在且互相一致。
- `axes` 非空且唯一，`axis_roles` 使用已定义 role，`shape` 为正整数，三者 rank 一致；role 不能从
  axis name 猜测。
- `policy` 是 `all_available` 或 `explicit`。
- `all_available` 不携带 endpoints，且 shape product 等于 topology 中 available endpoint 数。
- `all_available` 的 available endpoints 必须在 derived topology graph 中连通。
- `explicit` 的 endpoint tuple 数量等于 shape product，每个 endpoint 都存在、available 且不重复。
- `explicit` endpoints 必须属于同一个 available connected component。

accepted launch/transport projection 必须检查：

- 引用的 environment/topology/mesh/distributed variant/static-rank entry semantic digest 存在且一致；commit
  后生成的 target artifact 必须绑定同一 projection digest。
- 每个required component + partition/replica coordinate恰好有一个record；candidate entry/distributed
  prerequisite class、entrypoint ABI、
  parameter/state shard 和 endpoint coverage 完整。
- block id 非负且 collision-free；replicated execution 仍使用不同 canonical execution identity。
- transport assignment 的 channel/FSM/node/remote buffer/event/completion 不冲突，并覆盖实际 communication edge。
- rank-parametric relocation slot 类型/数量/owner 完整；per-rank-static specialization coordinate 完全匹配。
- `pinned` record 精确匹配 environment/topology/mesh/projection digest；`relocatable`必须是
  `ConcreteRecordSet`或`FiniteTemplateSet`二选一，所有concrete records/allowed bindings在compile time逐项
  通过同一projection/transport verifier，所有slot可机械填充且runtime无搜索自由度。
- projection 不复制 topology dimensions、unavailable table、tensor shard、layout/SPM/DDR plan 或 runtime handle。

失败必须报告在正确边界：capability/ABI 不满足是 target environment 错误；无法形成 valid execution mesh
是 pre-SPMD mesh selection 错误；component/rank/artifact coverage、stale topology、relocation 或 transport
冲突是 post-SPMD projection 错误。它们不能伪装成 downstream SPM、package schema 或 runtime provider 错误。

## 9. V0 范围

V0 支持：

- typed `wafer.target.environment` materialization，覆盖 target/revision/ABI、SPM/engine/packet/DTE legality
  limit、errata/feature 和 canonical capability fingerprint。
- 默认 target topology materialization，生成规则 card/tile grid、card interconnect kind 和
  unavailable endpoint exceptions。
- topology IR 可表达 unavailable tile；materialization / import 入口先支持默认 single-card 4x4、
  multi-card grid、mesh/torus kind 和 unavailable endpoint coord tuples。
- 默认从 available connected topology materialize `all_available` `wafer.execution.mesh`。
- 显式 `explicit` execution mesh override，用于 debug、小 workload、资源隔离或非默认 endpoint order。
- `wafer-opt --program-pipeline=stablehlo-spmd*` materialize 默认 `wafer.target.environment` /
  `wafer.target.topology` 和
  `wafer.execution.mesh`，`wafer-apply-default-spmd-sharding` 消费 execution mesh 生成 SDY mesh。
- SPMD 基于 environment/mesh 做 Shardy MPMD/sharding propagation 和 parameter/state logical shard generation。
- post-SPMD projection 对全部 component/partition/replica/rank class 形成完整 block/endpoint/resource/transport
  coverage。
- pinned projection；relocatable code artifact只允许从`ConcreteRecordSet`或展开后有限、预验证的
  `FiniteTemplateSet.allowed_bindings`中按确定性优先级选择并机械填relocation slot。
- calibration profile 仅用于 cost/provenance，不参与 verifier legality。

当前不作为 environment / topology / execution mesh / projection 基线通过标准：

- 跨卡 collective 最优 endpoint embedding。
- runtime 临时搜索新 endpoint、动态 unavailable endpoint 重映射或现场 transport allocation。
- serving-level request endpoint policy。
- full SPMD strategy / mesh embedding / communication algorithm co-optimization。

这些不是 logical mesh / rank group 的语义不支持。若上游产出 target environment/topology 可表达的跨卡
mesh，execution mesh / projection 应保留必要 endpoint/capability fact；没有 compatible precompiled
projection 时必须结构化拒绝，而不是让 runtime 改 mapping 或让上游 sharding program 改写语义。
