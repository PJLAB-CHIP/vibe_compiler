# Wafer Topology and Execution Mesh Design

状态：设计草案；范围：target topology、SPMD execution mesh 和 tile endpoint projection。

本文定义 target topology、SPMD-visible execution mesh 和 logical rank 到 Wafer physical tile endpoint
的映射边界。SPMD 不能在 abstract full mesh 上先切分，再由后段补坏 tile；它必须消费已经从
target topology 中选出的 valid execution mesh。参数分片元数据属于 Wafer program directory /
package resource view，不在本层 materialize 成核心 IR op 或 function boundary attr。

本文依赖：

- `tasks/03-shardy-spmd.md`
- `tasks/06-group.md`
- `tasks/13-communication.md`
- `tasks/15-launch-runtime-package.md`
- `docs/wafer-hardware-instruction-set-and-programming-model.md`
- `docs/tx8-deps-reverse-engineering/firmware-kuiper-runtime-hardware-analysis.md`

## 1. 目标和非目标

目标：

- 把 logical rank / block 投影到 physical tile endpoint coordinate。
- 在 SPMD 前 materialize runtime capability、规则 topology、unavailable tile metadata 和 derived
  connectivity，并从中选择 SPMD 可见的 valid execution mesh。
- 为 `wafer.group`、`wafer.tile.region`、后续 `wafer.tile.*` communication materialization 和
  runtime package metadata 提供可验证的 physical mapping。
- 在不能得到合法 execution mesh 或 endpoint projection 时，把原因反馈给 planner 或 compile driver。

非目标：

- 不做 Linalg tiling、group fusion 或 root tile shape search。
- 不分配 `#spm` offset、DDR runtime allocation object 或 workspace slice。
- 不选择 `Cx/NCx` physical layout。
- 不重新切分 tensor，不保存 per-rank tensor slice table。
- 不选择 ring/tree/DTE packet schedule；本层只提供 execution mesh、physical endpoints 和 topology facts。
- 不把 HPGR/KMD/legacy runtime API 细节写成上层 endpoint 语义。

## 2. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  target descriptor / runtime capability / board profile；partitioned StableHLO program directory
  中已经由 SPMD 产出的 parameter shard metadata / payload；committed `wafer.tile.region` /
  `wafer.instr.*` IR at the selected-instr boundary with accepted SPM offsets and DDR demand
  legality facts。
- Current stage responsibility:
  materialize / verify target topology 和 execution mesh rank domain。默认 `all_available` policy
  从 topology 派生所有 available endpoint；显式 override 才保存 endpoint tuples。验证 rank
  count、axis product、topology membership、availability、connectivity 和 explicit physical tile
  uniqueness。
- Output artifact / IR:
  `wafer.target.topology` regular card/tile grid + card interconnect kind + unavailable endpoint
  exceptions；`wafer.execution.mesh` accepted rank-domain fact：它引用 `wafer.target.topology`，
  保存 logical mesh axes / shape 和 endpoint policy；`all_available` 不复制 rank->physical endpoint，
  `explicit` 才保存 endpoint tuples。launch-visible block id 若需要跨阶段保留，应作为薄 launch /
  block binding 表达，不复制 rank->tile mapping。
- Downstream consumer:
  SPMD partition、communication lowering、ABI/LLVM lowering、package metadata 和 runtime adapter。
- User-level driver / named pipeline:
  `wafer-opt --program-pipeline=stablehlo-spmd*` 在 SPMD 前 materialize `wafer.target.topology` /
  `wafer.execution.mesh`，并让默认 SPMD seed 从 execution mesh rank count / axes 取数。
- Explicit non-goals:
  不重新做 group/candidate/tile shape/layout/SPM/DDR planning；不生成 DTE route、ABI call、packet、
  object、package、runtime handle 或 physical address；不靠 tensor 名字恢复 shard 语义。
- Completion gate:
  named pipeline 能重放 target topology materialization -> valid execution mesh selection -> SPMD partition
  -> group formation -> candidate selection -> committed instruction materialization；emitted
  `wafer.execution.mesh` 被 SPMD、communication 和 ABI/package resource view 消费；execution mesh
  成为唯一 rank-domain policy / optional explicit endpoint fact source；verifier 能拒绝 rank count、
  axis product mismatch、unavailable tile、duplicate explicit tile、out-of-topology、disconnected
  available component 和 rank 数不等于可用 tile。
```

## 3. 输入和输出

输入：

```text
partitioned StableHLO / local program
  + wafer.target.topology
  + wafer.execution.mesh
  + logical mesh / rank groups selected from valid topology
  + tensor collective rank groups / communication hints
  + shard shapes and communication volume hints from program metadata
```

输出：

```text
wafer.target.topology
  + regular card/tile grid
  + card interconnect kind
  + unavailable endpoint exceptions

wafer.execution.mesh
  + logical mesh axes
  + logical mesh shape
  + endpoint policy
  + optional explicit logical rank -> physical endpoint coordinate
  + target topology reference
```

accepted execution mesh 如果影响 codegen，必须进入 IR 或 launch metadata；mesh search trace、rejected
embeddings、score breakdown 是 analysis，不写入长期 IR。

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
- 单 tile SPM `3MB`。
- 单卡 tile 间 NoC 带宽高于跨卡 C2C。
- 32 卡服务器可以抽象成 `4 x 8` card mesh，但实际可用卡、坏卡、PG tile 必须来自 runtime
  capability。

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

长期 IR 只保留两个 topology / rank-domain 事实源，避免 topology、rank domain 和 rank->tile mapping 重复：

```text
wafer.target.topology: regular card/tile grid + card interconnect kind + unavailable endpoint exceptions
wafer.execution.mesh:  SPMD rank domain policy + logical mesh axes/shape + optional explicit endpoints
```

`wafer.execution.mesh` 不复制 topology dimensions、unavailable tile table、tile-id codec 或
connectivity links。默认 `all_available` policy 不保存 endpoint table；logical rank 到 physical
endpoint 的 view 由 `wafer.target.topology` 的规则 grid、`unavailable_tiles` 和固定 rank order
在使用点派生。只有 `explicit` policy 保存 endpoint tuples，用于少用 tile 或非默认 rank endpoint order。

长期不再单独保存 rank count、rank->tile、topology dimensions 或 unavailable tile table；这些由
`wafer.execution.mesh` 和 `wafer.target.topology` 派生。若 launch 需要 block id，应在 launch
outline/function 或薄 launch/block binding 中保存 `block_id_per_rank`，但该对象不得再次保存
rank->tile。

`wafer.target.topology` 由 target descriptor、runtime capability snapshot、board profile 或 bring-up
default pass materialize。它保存规则 grid 和 unavailable endpoint 例外，不展开所有 tile，也不保存
tile-id codec、availability partition 或 connectivity links。

V0 形式：

```mlir
wafer.target.topology @target {
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
该 pass 只是 topology materialization 入口；长期用户 compile flow 仍由 `wafer-opt` program
pipeline 组织，不要求用户手写 pass 串。

`wafer.execution.mesh` 是 SPMD 可见 mesh，必须在 SPMD partition 前存在。它引用
`wafer.target.topology`，记录 logical mesh axes / shape 和 endpoint policy：

- `all_available`：默认 policy。rank 数必须等于 topology 中 available endpoint 数，`endpoints`
  为空。rank->endpoint view 按 `card_y, card_x, tile_y, tile_x` row-major 顺序从 topology 派生，
  跳过 `unavailable_tiles`，不写入 IR。
- `explicit`：override policy。`endpoints` 按 logical rank 顺序保存
  `card_y, card_x, tile_y, tile_x` tuple，只用于 debug、资源隔离、小 workload 或非默认 rank
  endpoint order。

SPMD 的 sharding propagation、parameter shard metadata、tensor collective rank groups、
communication lowering 和 ABI/package resource view 都应引用这个 execution mesh，而不是假设完整
abstract mesh 或再从其它 op 恢复 rank->tile。

概念形式：

```mlir
wafer.execution.mesh @mesh
    {topology = @target,
     axes = ["rank"],
     shape = array<i64: 16>,
     policy = "all_available",
     endpoints = array<i64>}

wafer.execution.mesh @debug_mesh {
  topology = @target,
  axes = ["y", "x"],
  shape = array<i64: 2, 2>,
  policy = "explicit",
  endpoints = array<i64: 0, 0, 0, 0,
                          0, 0, 0, 1,
                          0, 0, 1, 0,
                          0, 0, 1, 1>}
```

参数分片元数据不在本层生成新的 IR 对象。`forward.parameter_shards.json` 和
`parameter_shards/<parameter>/rank_XXXXX.npy` 由 frontend/program verifier 校验：function local
argument type、logical rank count、rank coverage、slice bounds 和 NPY payload shape/dtype 必须一致；
logical rank count 必须和 `wafer.execution.mesh` 的 shape product 对齐。后续 ABI/package 需要
launch-visible shard view 时，应在使用点从 program metadata、execution mesh、launch/block binding 和
committed IR 重算，不维护另一套 module-level shard table。

## 6. Mesh Selection and Launch Projection

Mesh selection 必须发生在 SPMD 前。V0 默认 policy 使用所有 available endpoint：

1. 从 `wafer.target.topology` 派生规则 endpoint 集，并删除 `unavailable_tiles`。
2. 按 `card_interconnect`、`card_grid` 和 `tile_grid` 派生 available endpoint adjacency，再计算
   connected components。
3. 默认 `all_available` 要求所有 available endpoint 属于一个 connected component；logical rank order
   按 `card_y, card_x, tile_y, tile_x` row-major 派生。
4. 生成 `wafer.execution.mesh`，记录 logical mesh axes、shape 和 `all_available` policy，不保存
   endpoints。
5. 显式 override 可以使用 `explicit` policy 保存 endpoint tuples；这些 endpoints 必须存在、
   available、无重复，且属于同一个 available connected component。
6. SPMD partition 只消费这个 valid execution mesh 进行 sharding propagation 和 parameter shard
   metadata generation。

Launch projection V0 使用可解释的 deterministic projection，不追求全局最优：

1. 从 `wafer.execution.mesh` 读取 logical rank domain；endpoint view 从 `execution.mesh` policy 和
   `target.topology` 派生。
2. 验证 endpoint view 中的 endpoints 都存在于 `wafer.target.topology`，且仍 available、无重复、
   属于同一 connected component。
3. 按 logical rank 顺序生成同值 `block_id`，并把 block identity 绑定到 launch / outlined function。
4. 后续 cost model 可以在同一 `ExecutionMesh` contract 下替换 mesh selection 策略，但不能把 rejected maps、
   cost breakdown 或通信 route 写入 IR。

cost model 可以考虑：

- 卡内 NoC vs 跨卡 C2C 的相对代价。
- collective group 是否可以映射成局部 ring/tree。
- 每个 physical tile 的 local shard resource pressure。
- unavailable endpoint 分布导致的 hole 和 route penalty。
- logical mesh axis 到 physical graph path 的 shortest-path cost。

这些是候选排序，不是 IR contract。IR contract 只有 accepted target topology 和 execution mesh。

带洞 mesh 的 V1 算法可以在同一 IR 合同下扩展为 graph-aware logical mesh embedding：

1. 将 available topology 建成带权图 `G=(V,E)`。
2. 在 connected component 内生成 K-node connected subgraph candidates。
3. 尝试 logical mesh shape，例如 `[1,N]`、`[2,N/2]`、`[4,N/4]`。
4. 用 row-major、serpentine、BFS 或 Hilbert-like ordering 建立 rank order。
5. 对每个 logical axis 计算 collective cost；若 axis 跨 disconnected component，则 cost 为 infinity。
6. SPMD cost model 用 axis cost 选择 tensor dim -> mesh axis。

Full co-optimization（SPMD strategy、mesh embedding、endpoint projection、communication algorithm 同时搜索）
不作为当前 V0/V1 完成标准。

## 7. 与其它阶段的接口

| 阶段 | Topology / execution mesh 提供 | 本层不提供 |
| --- | --- | --- |
| SPMD partition | 由 `wafer.execution.mesh` 提供 valid logical mesh、rank domain 和 derived / explicit endpoint view | 事后修补 invalid full mesh |
| `wafer.group` | per-rank local shard / block identity、tensor collective rank group 可解释性 | tile shape、fusion boundary |
| `wafer.tile.region` | endpoint coordinate / block id args | SPM offset、layout assignment |
| `wafer.tile.*` communication | tile_region / SPM materialization 后的 logical endpoint 到 physical endpoint mapping；execution mesh 和 topology graph 可用于 route/cost | DTE node/FSM/packet allocation |
| runtime package metadata | cluster membership and launch metadata | runtime allocation mapping、completion source |

Topology / execution mesh 不应把 DTE route 或 runtime launch API 写进上层。Communication lowering 可以基于
accepted endpoint view 选择 ring/tree/unicast protocol。

## 8. Verifier

`wafer.target.topology` 必须检查：

- `card_grid` 和 `tile_grid` 都是两个正整数。
- `card_interconnect` 是 `mesh` 或 `torus`。
- `unavailable_tiles` 按 `card_y/card_x/tile_y/tile_x` 4 元 tuple 展开。
- 每个 unavailable endpoint 都在 grid 范围内，且没有重复。

`wafer.execution.mesh` 必须检查：

- 引用的 target topology 存在。
- `axes` 非空且唯一，`shape` 为正整数，axis 数量与 shape rank 一致。
- `policy` 是 `all_available` 或 `explicit`。
- `all_available` 不携带 endpoints，且 shape product 等于 topology 中 available endpoint 数。
- `all_available` 的 available endpoints 必须在 derived topology graph 中连通。
- `explicit` 的 endpoint tuple 数量等于 shape product，每个 endpoint 都存在、available 且不重复。
- `explicit` endpoints 必须属于同一个 available connected component。

长期 launch/block binding 必须检查：

- 引用的 `wafer.execution.mesh` 和 `wafer.target.topology` 存在且一致。
- logical rank 数量与 `block_ids` 数量一致。
- block id 非负且不重复，除非明确表达 replicated execution。
- referenced execution mesh 满足 launch capability。
- logical rank group 中的 endpoints 都有 physical tile endpoint。
- launch/block binding 中没有 topology dimensions、unavailable tile table、tile id codec、rank->tile、
  SPM offset、DDR address、DTE packet 或 runtime handle。

Planner 失败也必须报告在正确边界：无法形成 valid execution mesh 是 mesh selection / SPMD 前错误；
rank 数超过可用 good tile、endpoint 不存在、mesh axis disconnected 或 topology import 冲突不能
伪装成 downstream SPM、DTE 或 runtime package 错误。

## 9. V0 范围

V0 支持：

- 默认 target topology materialization，生成规则 card/tile grid、card interconnect kind 和
  unavailable endpoint exceptions。
- topology IR 可表达 unavailable tile；materialization / import 入口先支持默认 single-card 4x4、
  multi-card grid、mesh/torus kind 和 unavailable endpoint coord tuples。
- 默认从 available connected topology materialize `all_available` `wafer.execution.mesh`。
- 显式 `explicit` execution mesh override，用于 debug、小 workload、资源隔离或非默认 endpoint order。
- `wafer-opt --program-pipeline=stablehlo-spmd*` materialize 默认 `wafer.target.topology` 和
  `wafer.execution.mesh`，`wafer-apply-default-spmd-sharding` 消费 execution mesh 生成 SDY mesh。
- SPMD 基于 `wafer.execution.mesh` 做 sharding propagation 和 parameter shard metadata generation。
- block id 与 logical rank 的一一绑定。
- local shard 关联/验证只消费已有显式 shard facts；当前 planner 不从 tensor 名字或 payload 恢复
  shard，也不重新切分 tensor。

当前不作为 topology / execution mesh 基线通过标准：

- 跨卡 collective 最优 endpoint embedding。
- runtime 迁移、动态 unavailable endpoint 重映射。
- serving-level request endpoint policy。
- full SPMD strategy / mesh embedding / communication algorithm co-optimization。

这些不是 logical mesh / rank group 的语义不支持。若上游产出硬件 topology 可表达的跨卡 mesh，
execution mesh / resource view 应保留必要 endpoint / capability fact；缺少最优策略或 runtime route
时，失败应定位到 mesh/runtime 恢复项，而不是让上游 sharding program 改写语义。
