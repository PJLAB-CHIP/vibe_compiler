# Wafer Topology, Device Mesh, and Shard Binding Design

状态：设计草案；范围：target topology、SPMD device mesh、shard binding 和 encoded tile endpoint。

本文定义 target topology、SPMD-visible device mesh 和 logical rank 到 Wafer physical tile endpoint 的
映射边界。SPMD 不能在 abstract full mesh 上先切分，再由后段补坏 tile；它必须消费
已经从 target topology 中选出的 valid device mesh。Shard binding 只把已经存在的 logical rank /
rank group 事实投影到这个 device mesh 的 encoded tile endpoint，并为 `wafer.group` /
`wafer.tile.region` / runtime launch 提供物理端点合同。它不负责 tensor tiling、SPM offset、
physical layout、DTE algorithm、DDR allocation 或 C ABI。

本文依赖：

- `tasks/03-shardy-spmd.md`
- `tasks/06-group.md`
- `tasks/13-communication.md`
- `tasks/15-launch-runtime-package.md`
- `docs/wafer-hardware-instruction-set-and-programming-model.md`
- `docs/tx8-deps-reverse-engineering/firmware-kuiper-runtime-hardware-analysis.md`

## 1. 目标和非目标

目标：

- 把 logical rank / block / shard 映射到 encoded physical tile endpoint。
- 在 SPMD 前 materialize runtime capability、topology、tile id mapping、PG/bad-tile metadata 和
  physical connectivity，并从中选择 SPMD 可见的 valid device mesh。
- 为 `wafer.group`、`wafer.tile.region`、后续 `wafer.tile.*` communication materialization 和 `wafer.launch` 提供
  可验证的 physical mapping。
- 在不能得到合法 placement 时，把原因反馈给 planner 或 compile driver。

非目标：

- 不做 Linalg tiling、group fusion 或 root tile shape search。
- 不分配 `#spm` offset、DDR runtime allocation object 或 workspace slice。
- 不选择 `Cx/NCx` physical layout。
- 不选择 ring/tree/DTE packet schedule；placement 只提供 physical endpoints 和 topology facts。
- 不把 HPGR/KMD/legacy runtime API 细节写成上层 placement 语义。

## 2. Pipeline Contract

```text
Pipeline position:
- Upstream artifact / IR:
  `wafer.target.topology` materialized from target descriptor / runtime capability / board profile；
  `wafer.device.mesh` selected from valid connected topology before SPMD；frontend/SPMD materialized
  `wafer.shard.binding` local-shard facts referencing that mesh；committed `wafer.tile.region` /
  `wafer.instr.*` IR at the selected-instr boundary with accepted SPM offsets and DDR demand legality facts。
- Current stage responsibility:
  验证 selected device mesh 与 topology 仍一致，将 logical rank / block 投影到 encoded physical tile id；
  若上游已有显式 local shard facts，则只绑定到 logical rank 并验证 bounds；验证 rank coverage、
  topology membership、availability、connectivity、physical tile uniqueness 和 block id uniqueness。
- Output artifact / IR:
  长期输出是 `wafer.device.mesh` accepted embedding：它引用 `wafer.target.topology`，保存 logical
  mesh axes 与 rank->encoded tile id。launch-visible block id 若需要跨阶段保留，应作为薄 launch /
  block binding 表达，不复制 rank->tile mapping。`wafer.placement.map` 是当前过渡 op，不是最终事实源。
  topology dimensions、bad tile、tile id codec 和 connectivity 只属于 target topology / device mesh。
  local shard 只通过 `wafer.shard.binding` / launch-visible resource view 引用，planner 不重新切分
  tensor，也不复制 memory plan。
- Downstream consumer:
  communication lowering、ABI/LLVM lowering、package manifest 和 runtime adapter。
- User-level driver / named pipeline:
  `wafer-plan-placement` pass 和 `wafer-lower-groups-to-placement` named pipeline。该 pipeline
  在 committed selected-instr boundary 之后追加 placement；用户不应手工拼 placement fixture 作为主线。
- Explicit non-goals:
  不重新做 group/candidate/tile shape/layout/SPM/DDR planning；不生成 DTE route、ABI call、packet、
  object、package、runtime handle 或 physical address；不靠 tensor 名字恢复 shard 语义。
- Completion gate:
  named pipeline 能重放 target topology materialization -> valid device mesh selection -> SPMD partition
  -> group formation -> candidate selection -> committed instruction materialization -> placement；
  emitted `wafer.device.mesh` 被 SPMD、communication verifier 和 ABI/package resource view 消费，并
  成为唯一 rank-domain / rank->encoded endpoint embedding fact source；
  verifier 能拒绝 rank count、unavailable tile、duplicate tile、duplicate block、out-of-topology、
  disconnected mesh axis 和 rank 数超过可用 tile。
```

## 3. 输入和输出

输入：

```text
partitioned StableHLO / local program
  + wafer.target.topology
  + wafer.device.mesh
  + logical mesh / rank groups selected from valid topology
  + tensor collective rank groups / communication hints
  + shard shapes and communication volume hints
```

输出：

```text
wafer.device.mesh
  + logical mesh axes
  + logical rank -> encoded physical tile id
  + target topology reference
  + cluster membership / launch block metadata if retained
```

accepted device mesh 如果影响 codegen，必须进入 IR 或 launch metadata；mesh search trace、
rejected embeddings、score breakdown 是 analysis，不写入长期 IR。

## 4. Physical Topology And Tile Id Model

Wafer physical topology 是一个多卡、多 tile 组成的 mesh graph。每个 tile node 至少有：

```text
coord:        [card_y, card_x, tile_y, tile_x]
tile_id:      hardware / runtime / DTE visible encoded endpoint id
availability: available | bad | pg-disabled
links:        adjacent encoded tile ids with optional cost
```

已知事实：

- 单卡 `4 x 4 = 16 tile`。
- 单 tile SPM `3MB`。
- 单卡 tile 间 NoC 带宽高于跨卡 C2C。
- 32 卡服务器可以抽象成 `4 x 8` card mesh，但实际可用卡、坏卡、PG tile 必须来自 runtime
  capability。

Tile id 编码规则不能散落在 placement planner、verifier、communication lowering 或 ABI lowering
里。默认规则可以存在，但它只能由 target topology materialization / import pass 使用，用来生成显式
`coord -> tile_id` mapping；下游必须通过 topology model 查询 mapping，不能手写 row-major /
card-major 公式。若 driver 返回 remap table、PG 改变可用 tile、跨卡 global id 编码变化，只改
`wafer.target.topology` 或对应 import/rewrite pass。

Placement 不要求所有程序都用完整 16 tile。cluster 可以是单 tile、单卡子集、单卡全 tile 或
多卡 mesh。当前实现优先恢复单卡和小规模 multi-tile；跨卡 topology 仍应保持可表达。跨卡 route、
C2C cost 或 runtime completion 尚未完善时，应形成 placement/runtime 恢复任务，不能反向要求
SPMD 在无效 mesh 上切分。

单 tile、单卡多 tile 和多卡多 tile 都使用同一个抽象：

```text
logical rank -> encoded tile id
encoded tile id -> topology node coord / links / availability
```

单卡只是 topology 里的可用 card 维度为 1，多卡只是 topology node 覆盖多个 card；planner 和
verifier 不分裂成两套语义。

## 5. IR Representation

长期 IR 只保留三个事实源，避免 rank domain 和 rank->tile mapping 重复：

```text
wafer.target.topology: physical tile graph, encoded tile ids, availability, links, id codec provenance
wafer.device.mesh:     SPMD rank domain + accepted logical-rank -> encoded-tile embedding
wafer.shard.binding:   logical rank -> launch-visible tensor slice
```

`wafer.placement.map` 是当前过渡实现。长期不再单独保存 rank count、rank->tile、topology dimensions
或 bad tile table；这些要么属于 `wafer.device.mesh`，要么属于 `wafer.target.topology`。若 launch
需要 block id，应在 launch outline/function 或薄 launch/block binding 中保存 `block_id_per_rank`，
但该对象不得再次保存 rank->tile。

`wafer.target.topology` 由 target descriptor、runtime capability snapshot、board profile 或 bring-up
default pass materialize。它显式保存 tile nodes、encoded tile ids、可用性和 connectivity。默认 tile
id 规则可以是 `row_major_4d` / `card_major_4d` 这类 codec 名称，但 codec 只说明 mapping 的生成来源；
IR 合同是显式的 `coord -> tile_id` 表。

概念形式：

```mlir
wafer.target.topology @target {
  axes = ["card_y", "card_x", "tile_y", "tile_x"],
  id_encoding = #wafer.tile_id_encoding<row_major_4d>,
  tile_coords = array<i64: 0, 0, 0, 0,
                            0, 0, 0, 1>,
  tile_ids = array<i64: 0, 1>,
  available_tile_ids = array<i64: 0, 1>,
  links = array<i64: 0, 1>
}
```

`wafer.device.mesh` 是 SPMD 可见 mesh，必须在 SPMD partition 前存在。它从
`wafer.target.topology` 的 available connected component 中选择 rank domain，并记录 logical rank 到
encoded tile id 的 mapping。SPMD 的 sharding propagation、parameter shard metadata、tensor
collective rank groups、communication lowering 和 ABI/package resource view 都应引用这个 device
mesh，而不是假设完整 abstract mesh 或再从 placement op 恢复 rank->tile。

概念形式：

```mlir
wafer.device.mesh @mesh {
  topology = @target,
  axis_names = ["x", "y"],
  axis_sizes = array<i64: 1, 2>,
  rank_tile_ids = array<i64: 0, 1>
}
```

`wafer.shard.binding` 由 frontend/SPMD program metadata materialize，当前主要来自
`forward.parameter_shards.json`。它引用 entry function symbol 和 argument index，记录该 launch-visible
tensor 的 global shape、rank-local shape、logical rank 覆盖和 per-rank static slice
`offsets/sizes/strides`。它不记录 payload file path、runtime handle、physical tile、SPM/DDR offset
或 package manifest 字段。长期它还必须引用 `wafer.device.mesh`；`logical_rank_count` 只用于校验
coverage，不能作为 rank domain 的根事实源。

概念形式：

```mlir
wafer.shard.binding {
  mesh = @mesh,
  kernel = @forward,
  argument_index = 0 : i64,
  logical_rank_count = 2 : i64,
  global_shape = array<i64: 2, 4>,
  local_shape = array<i64: 1, 4>,
  shard_ranks = array<i64: 0, 1>,
  shard_offsets = array<i64: 0, 0, 1, 0>,
  shard_sizes = array<i64: 1, 4, 1, 4>,
  shard_strides = array<i64: 1, 1, 1, 1>
}
```

数组按 `shard_ranks` 顺序展开。每个 rank 的 slice 维度数必须等于 `global_shape` rank。
`local_shape` 必须匹配引用的 function argument type。payload 文件仍由 program directory verifier
校验，不进入 IR；package emission 在使用点从 shard binding、device mesh、薄 launch/block binding
和 resource view 派生 manifest metadata。

accepted mesh embedding 需要在边界 op 上显式表达。推荐结构：

- 在 SPMD 前生成 `wafer.device.mesh`，作为 rank domain 和 rank->encoded tile id 的唯一事实源。
- 在 `wafer.tile.region` lowering 时把 per-tile logical id / block id / encoded tile id 作为
  region argument、constant-like descriptor 或 launch argument 传入；encoded tile id 来自
  `wafer.device.mesh`。
- 在 `wafer.tile.*` communication lowering 时通过 device mesh 和 topology model 查
  physical source/destination endpoint；这个查询发生在
  tensor collective 已经经 group/tiling 和 tile_region / SPM materialization 之后。

Device mesh 可以保存结构性 rank->tile embedding，因为它不能从 local compute IR 重新推出，且直接影响
SPMD、communication 和 codegen。但它不能保存搜索过程、cost model 中间分数、失败候选或 SPM/DDR
allocation trace。

长期概念字段：

```text
DeviceMesh {
  target_topology_ref
  logical_mesh_axes
  logical_mesh_shape
  rank_tile_ids
}
```

如果后续实现发现 attribute 无法表达 verifier 需要的结构，可以拆成专门 topology / mesh op /
region；但不能退化成名字约定或 side table。

当前已实现的 V0 op 仍把 topology dimensions 和 bad tile ids 临时放在 `wafer.placement.map` 里：

```mlir
wafer.placement.map {
  logical_rank_count = 4 : i64,
  block_ids = array<i64: 0, 1, 2, 3>,
  physical_tile_coords = array<i64: 0, 0, 0, 0,
                                  0, 0, 0, 1,
                                  0, 0, 0, 2,
                                  0, 0, 0, 3>,
  card_y_count = 1 : i64,
  card_x_count = 1 : i64,
  tile_y_count = 4 : i64,
  tile_x_count = 4 : i64,
  bad_tile_ids = array<i64>
}
```

`physical_tile_coords` 按 logical rank 顺序展开，每 4 个整数为
`card_y, card_x, tile_y, tile_x`。logical rank 本身由数组顺序表达，避免再维护一份重复的
rank id 表。`bad_tile_ids` 是当前 capability / PG 过滤后的 flat physical tile id 集合；`block_ids`
按 logical rank 顺序记录 launch-visible block identity，必须非负且唯一。local shard metadata 或
capability reference 后续接到 package/launch metadata，但不能改变 tensor semantics。

该 V0 op 是过渡实现，不是最终合同。收口后 `rank_tile_ids` 只保存在 `wafer.device.mesh`；
topology dimensions、bad tile 集合、connectivity 和 tile id codec 只保存在 `wafer.target.topology`。
`wafer.placement.map` 应被删除，或降级为只保存 `block_ids` 的 launch/block binding；它不能再次保存
rank->tile。

package manifest gate 必须接入 launch-visible endpoint metadata：target topology / device mesh
提供 available / excluded encoded tile ids、connectivity assumption 和 per-rank encoded endpoint；
thin launch/block binding 可提供 per-rank `block_id`；`local_shards` 进入 IR-derived package metadata。
`local_shards` 应引用 launch signature argument/result index 或等价 ABI slot，并记录静态 slice
bounds；tensor name 只能用于诊断/显示，不能作为绑定协议。它不反向修改 tensor IR shape、layout
或 sharding semantics。

## 6. Placement Algorithm

Mesh selection 必须发生在 SPMD 前。V0 推荐使用 valid rectangular submesh 算法：

1. 从 `wafer.target.topology` 删除 bad / PG-disabled tile。
2. 在 available tile graph 上计算 connected components。
3. 在每个 component 内枚举 contiguous rectangular submesh。
4. 根据 requested rank count、preferred mesh shape 和 collective hints 选择最大或最接近的 candidate。
5. 生成 `wafer.device.mesh`，记录 logical mesh axes 和 `rank -> encoded tile id`。
6. SPMD partition 只消费这个 valid device mesh 进行 sharding propagation 和 parameter shard
   materialization。

Placement / launch projection V0 使用可解释的 deterministic projection，不追求全局最优：

1. 从 `wafer.device.mesh` 读取 logical rank domain 和 rank tile ids；无 mesh 的局部 fixture 只能
   显式提供过渡 device mesh config。
2. 验证 rank tile ids 都存在于 `wafer.target.topology`，且仍 available、无重复、属于同一 selected
   connected mesh。
3. 按 logical rank 顺序生成同值 `block_id`，并把 block identity 绑定到 launch / outlined function。
4. 后续 cost model 可以在同一 `DeviceMesh` contract 下替换 mesh selection 策略，但不能把 rejected maps、
   cost breakdown 或通信 route 写入 IR。

cost model 可以考虑：

- 卡内 NoC vs 跨卡 C2C 的相对代价。
- collective group 是否可以映射成局部 ring/tree。
- 每个 physical tile 的 local shard resource pressure。
- good-tile 分布导致的 hole 和 route penalty。
- logical mesh axis 到 physical graph path 的 shortest-path cost。

这些是候选排序，不是 IR contract。IR contract 只有 accepted target topology、device mesh 和
boundary shard binding。

带洞 mesh 的 V1 算法可以在同一 IR 合同下扩展为 graph-aware logical mesh embedding：

1. 将 available topology 建成带权图 `G=(V,E)`。
2. 在 connected component 内生成 K-node connected subgraph candidates。
3. 尝试 logical mesh shape，例如 `[1,N]`、`[2,N/2]`、`[4,N/4]`。
4. 用 row-major、serpentine、BFS 或 Hilbert-like ordering 建立 rank order。
5. 对每个 logical axis 计算 collective cost；若 axis 跨 disconnected component，则 cost 为 infinity。
6. SPMD cost model 用 axis cost 选择 tensor dim -> mesh axis。

Full co-optimization（SPMD strategy、mesh embedding、placement、communication algorithm 同时搜索）
不作为当前 V0/V1 完成标准。

## 7. 与其它阶段的接口

| 阶段 | Placement 提供 | Placement 不提供 |
| --- | --- | --- |
| SPMD partition | 由 `wafer.device.mesh` 提供 valid logical mesh 和 rank tile ids | 事后修补 invalid full mesh |
| `wafer.group` | per-rank local shard / block identity、tensor collective rank group 可解释性 | tile shape、fusion boundary |
| `wafer.tile.region` | encoded tile id / block id args | SPM offset、layout assignment |
| `wafer.tile.*` communication | tile_region / SPM materialization 后的 logical endpoint 到 physical endpoint mapping；device mesh 和 topology graph 可用于 route/cost | DTE node/FSM/packet allocation |
| `wafer.launch` | cluster membership and launch metadata | runtime allocation mapping、completion source |

Placement 不应把 DTE route 或 runtime launch API 写进上层。Communication lowering 可以基于
accepted placement 选择 ring/tree/unicast protocol。

## 8. Verifier

`wafer.target.topology` 必须检查：

- 每个 tile node 的 encoded tile id 唯一。
- tile coordinate rank 与 topology axes 一致。
- available / bad / PG-disabled 状态不互相冲突。
- link endpoints 都引用已知 tile id。
- codec provenance 只用于生成或重建 mapping；verifier 和 lowering 不依赖隐含公式。

`wafer.device.mesh` 必须检查：

- 引用的 target topology 存在。
- `rank_tile_ids` 数量等于 mesh axis product。
- 每个 rank tile id 都存在、available 且不重复。
- mesh axis 上需要通信的 rank pair 在 topology graph 中连通；对 V0 rectangular mesh，rank tile ids
  还必须构成 contiguous rectangular submesh。

`wafer.shard.binding` 必须检查：

- `kernel` symbol 能解析到 `func.func`。
- `argument_index` 指向合法 function argument，且该 argument 是 static ranked tensor。
- `global_shape`、`local_shape` 与 function argument rank / element type 一致；`local_shape` 匹配
  argument type。
- `logical_rank_count` 为正，且与引用的 `wafer.device.mesh` rank 数一致；`shard_ranks` 覆盖
  `0..logical_rank_count-1` 且不重复。
- `shard_offsets` / `shard_sizes` / `shard_strides` 都按 rank-major 展开，长度等于
  `logical_rank_count * global_rank`。
- 每个 slice 不越过 `global_shape`，`sizes` 不超过 `local_shape`，`strides` 为正。
- 同一 function argument 不能有多份 shard binding；若存在 launch/block binding，device mesh /
  rank count 必须一致。

长期 launch/block binding 必须检查：

- 引用的 `wafer.device.mesh` 和 `wafer.target.topology` 存在且一致。
- logical rank 数量与 `block_ids` 数量一致。
- block id 非负且不重复，除非明确表达 replicated execution。
- referenced device mesh 满足 launch capability。
- logical rank group 中的 endpoints 都有 encoded physical tile endpoint。
- launch/block binding 中没有 topology dimensions、bad tile table、tile id codec、rank->tile、
  SPM offset、DDR address、
  DTE packet 或 runtime handle。

Planner 失败也必须报告在正确边界：无法形成 valid device mesh 是 mesh selection / SPMD 前错误；
rank 数超过可用 good tile、tile id 不存在、mesh axis disconnected 或 topology import 冲突不能
伪装成 downstream SPM、DTE 或 runtime package 错误。

## 9. V0 范围

V0 支持：

- 默认 target topology materialization，生成显式 coord -> encoded tile id mapping。
- topology import/rewrite pass 改写 tile id mapping、availability 和 links。
- 从 available connected topology 中选择 valid rectangular `wafer.device.mesh`。
- SPMD 基于 `wafer.device.mesh` 做 sharding propagation 和 parameter shard binding。
- single-tile placement。
- single-card 2/4/8/16 tile cluster。
- 多卡多 tile 的 topology-aware deterministic mapping。
- good-tile / bad-tile / PG-disabled tile 过滤。
- block id 与 logical rank 的一一绑定。
- local shard 绑定/验证只消费已有显式 shard facts；当前 planner 不从 tensor 名字或 payload 恢复
  shard，也不重新切分 tensor。

当前不作为 placement 基线通过标准：

- 跨卡 collective 最优 placement。
- runtime 迁移、动态 bad-tile 重映射。
- serving-level request placement。
- full SPMD strategy / mesh embedding / communication algorithm co-optimization。

这些不是 logical mesh / rank group 的语义不支持。若上游产出硬件 topology 可表达的跨卡 mesh，
placement IR 应保留必要 mapping / capability fact；缺少最优策略或 runtime route 时，失败应定位到
placement/runtime 恢复项，而不是让上游 sharding program 改写语义。
