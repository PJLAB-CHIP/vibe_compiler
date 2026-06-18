# Wafer Placement Design

日期：2026-05-25

状态：设计草案；范围：logical mesh / rank 到 Wafer physical card/tile cluster 的 placement。

本文定义 logical mesh / rank 到 Wafer physical card/tile cluster 的 placement 边界。Placement
消费 Shardy / SPMD 与 Wafer LinalgExt-style tensor collective handoff 保留下来的 logical rank /
rank group 事实，并为 `wafer.group` / `wafer.tile.region` / runtime launch 提供物理映射合同。
它不负责 tensor tiling、SPM offset、physical layout、DTE algorithm、DDR allocation 或 C ABI。

本文依赖：

- `tasks/2026-05-25-wafer-shardy-spmd-design.md`
- `tasks/2026-05-12-wafer-group-design.md`
- `tasks/2026-05-25-wafer-communication-dialect-design.md`
- `tasks/2026-05-25-wafer-launch-runtime-package-design.md`
- `docs/wafer-hardware-instruction-set-and-programming-model.md`
- `docs/tx8-deps-reverse-engineering/firmware-kuiper-runtime-hardware-analysis.md`

## 1. 目标和非目标

目标：

- 把 logical rank / block / shard 映射到 physical card/tile coordinate。
- 吸收 runtime capability、topology、good-tile bitmap、PG/bad-tile metadata 和 card/tile mesh
  信息。
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
  committed `wafer.tile.region` / `wafer.instr.*` IR at the selected-instr boundary，
  with accepted SPM offsets and DDR demand legality facts；frontend/SPMD materialized
  `wafer.shard.binding` local-shard facts；target topology / capability / good-tile / bad-tile metadata。
- Current stage responsibility:
  选择 logical rank / block 到 physical `(card_y, card_x, tile_y, tile_x)` 的 accepted mapping；
  若上游已有显式 local shard facts，则只绑定到 logical rank 并验证 bounds；验证 rank coverage、
  topology bounds、bad tile 过滤、physical tile uniqueness 和 block id uniqueness。
- Output artifact / IR:
  `wafer.placement.map` accepted mapping。它保存 rank->physical coordinate、block id、topology
  dimensions 和 bad tile facts；local shard 只通过 `wafer.shard.binding` / launch-visible resource view 引用，
  placement planner 不重新切分 tensor，也不复制 memory plan。
- Downstream consumer:
  communication lowering、ABI/LLVM lowering、package manifest 和 runtime adapter。
- User-level driver / named pipeline:
  `wafer-plan-placement` pass 和 `wafer-lower-groups-to-placement` named pipeline。该 pipeline
  在 committed selected-instr boundary 之后追加 placement；用户不应手工拼 placement fixture 作为主线。
- Explicit non-goals:
  不重新做 group/candidate/tile shape/layout/SPM/DDR planning；不生成 DTE route、ABI call、packet、
  object、package、runtime handle 或 physical address；不靠 tensor 名字恢复 shard 语义。
- Completion gate:
  named pipeline 能重放 group formation -> candidate selection -> committed instruction materialization
  -> placement；
  emitted `wafer.placement.map` 被 communication verifier 消费，并为 ABI/package resource view 提供
  唯一 placement fact source；
  verifier 能拒绝 rank count、bad tile、duplicate tile、duplicate block、out-of-topology 和 rank 数超过可用 tile。
```

## 3. 输入和输出

输入：

```text
partitioned StableHLO / local program
  + logical mesh / rank groups
  + tensor collective rank groups / communication hints
  + shard shapes and communication volume hints
  + target topology and capability
  + good-tile / PG metadata
```

输出：

```text
accepted placement map
  + logical rank -> physical coordinate
  + cluster membership
  + block id metadata
  + topology dimensions and bad-tile facts
```

accepted placement 如果影响 codegen，必须进入 IR 或 launch metadata；placement search trace、
rejected maps、score breakdown 是 analysis，不写入长期 IR。

## 4. Physical Coordinate Model

Wafer physical topology 使用层级 coordinate，而不是 flat device id：

```text
card_y, card_x, tile_y, tile_x
```

已知事实：

- 单卡 `4 x 4 = 16 tile`。
- 单 tile SPM `3MB`。
- 单卡 tile 间 NoC 带宽高于跨卡 C2C。
- 32 卡服务器可以抽象成 `4 x 8` card mesh，但实际可用卡、坏卡、PG tile 必须来自 runtime
  capability。

Placement 不要求所有程序都用完整 16 tile。cluster 可以是单 tile、单卡子集、单卡全 tile 或
多卡 mesh。当前实现优先恢复单卡和小规模 multi-tile；跨卡 topology 仍应保持可表达。跨卡 route、
C2C cost 或 runtime completion 尚未完善时，应形成 placement/runtime 恢复任务，不能反向要求
SPMD 不产生跨卡 logical mesh 或 rank group。

单 tile、单卡多 tile 和多卡多 tile 都使用同一个抽象：

```text
logical rank -> (card_y, card_x, tile_y, tile_x)
```

单卡只是 `card_y_count = card_x_count = 1`，多卡只是 card 维度大于 1；planner 和 verifier 不
分裂成两套语义。

## 5. IR Representation

accepted placement 和 boundary shard binding 需要分成两个 IR fact source：

```text
wafer.shard.binding:  logical rank -> launch-visible tensor slice
wafer.placement.map:  logical rank -> physical tile coordinate
```

`wafer.shard.binding` 由 frontend/SPMD program metadata materialize，当前主要来自
`forward.parameter_shards.json`。它引用 entry function symbol 和 argument index，记录该 launch-visible
tensor 的 global shape、rank-local shape、logical rank 覆盖和 per-rank static slice
`offsets/sizes/strides`。它不记录 payload file path、runtime handle、physical tile、SPM/DDR offset
或 package manifest 字段。

概念形式：

```mlir
wafer.shard.binding {
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
校验，不进入 IR；package emission 在使用点从 shard binding、placement map 和 resource view 派生
manifest metadata。

accepted placement 需要在边界 op 上显式表达。推荐结构：

- 在 `wafer.launch` 或被 launch outline 的 function 上保存 cluster-level placement mapping。
- 在 `wafer.tile.region` lowering 时把 per-tile logical id / block id / physical coordinate 作为
  region argument、constant-like descriptor 或 launch argument 传入。
- 在 `wafer.tile.*` communication lowering 时通过 placement mapping 查 physical source/destination tile；这个查询发生在
  tensor collective 已经经 group/tiling 和 tile_region / SPM materialization 之后。

Placement attr 可以保存结构性 mapping，因为它不能从 local IR 重新推出，且直接影响 codegen。
但它不能保存搜索过程、cost model 中间分数、失败候选或 SPM/DDR allocation trace。

概念字段：

```text
PlacementMap {
  logical_mesh_axes
  logical_rank_count
  physical_cluster
  logical_rank_to_physical_coord
  block_id_per_rank
  good_tile_bitmap_or_capability_ref
}
```

如果后续实现发现 attribute 无法表达 verifier 需要的结构，可以拆成专门 placement op /
region；但不能退化成名字约定或 side table。

当前 V0 采用专门 op 表达最小 accepted mapping：

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
rank id 表。`bad_tile_ids` 是当前 capability / PG 过滤后的 flat physical tile id 集合；后续
`block_ids` 按 logical rank 顺序记录 launch-visible block identity，必须非负且唯一。local shard
metadata 或 capability reference 后续接到 package/launch metadata，但不能改变 tensor semantics。

package manifest gate 必须接入 launch-visible placement metadata：`good_tile_ids` /
`bad_tile_ids` 表达当前 target capability assumption，per-rank `block_id` 和 `local_shards` 进入
IR-derived package metadata。`local_shards` 只引用 launch signature tensor 名称和静态 slice bounds；它不反向修改
tensor IR shape、layout 或 sharding semantics。

## 6. Placement Algorithm

V0 使用可解释的 deterministic placement，不追求全局最优：

1. 从 `wafer.shard.binding` 推导 logical rank count；若没有 shard binding，显式 driver option 可作为
   bring-up fixture override，缺省退回 single-rank。
2. 从 target capability 构造 card-major / tile-major physical tile sequence，过滤 PG/bad tile。
3. 检查可用 tile 数不少于 logical rank 数。
4. V0 deterministic strategy 按 logical rank 顺序取前 N 个 good tiles，并为每个 rank 生成同值
   `block_id`。该策略已经统一覆盖 single-tile、single-card multi-tile 和 multi-card multi-tile。
5. 后续 cost model 可以在同一 `PlacementMap` contract 下替换排序策略，但不能把 rejected maps、
   cost breakdown 或通信 route 写入 IR。

cost model 可以考虑：

- 卡内 NoC vs 跨卡 C2C 的相对代价。
- collective group 是否可以映射成局部 ring/tree。
- 每个 physical tile 的 local shard resource pressure。
- good-tile 分布导致的 hole 和 route penalty。

这些是候选排序，不是 IR contract。IR contract 只有 accepted placement mapping。

## 7. 与其它阶段的接口

| 阶段 | Placement 提供 | Placement 不提供 |
| --- | --- | --- |
| `wafer.group` | per-rank local shard / block identity、tensor collective rank group 可解释性 | tile shape、fusion boundary |
| `wafer.tile.region` | physical tile coordinate / block id args | SPM offset、layout assignment |
| `wafer.tile.*` communication | tile_region / SPM materialization 后的 logical endpoint 到 physical endpoint mapping | DTE node/FSM/packet allocation |
| `wafer.launch` | cluster membership and launch metadata | runtime allocation mapping、completion source |

Placement 不应把 DTE route 或 runtime launch API 写进上层。Communication lowering 可以基于
accepted placement 选择 ring/tree/unicast protocol。

## 8. Verifier

`wafer.shard.binding` 必须检查：

- `kernel` symbol 能解析到 `func.func`。
- `argument_index` 指向合法 function argument，且该 argument 是 static ranked tensor。
- `global_shape`、`local_shape` 与 function argument rank / element type 一致；`local_shape` 匹配
  argument type。
- `logical_rank_count` 为正；`shard_ranks` 覆盖 `0..logical_rank_count-1` 且不重复。
- `shard_offsets` / `shard_sizes` / `shard_strides` 都按 rank-major 展开，长度等于
  `logical_rank_count * global_rank`。
- 每个 slice 不越过 `global_shape`，`sizes` 不超过 `local_shape`，`strides` 为正。
- 同一 function argument 不能有多份 shard binding；若 module 中已有 `wafer.placement.map`，
  `logical_rank_count` 必须一致。

`wafer.placement.map` 必须检查：

- logical rank 数量与 physical coordinate 数量一致。
- physical coordinate 在 target topology 内。
- 每个 mapped tile 通过 good-tile / PG validation。
- mapping 满足 selected cluster 的 launch capability。
- logical rank group 中的 endpoints 都有 physical coordinate。
- block id / tile id 没有重复冲突，除非明确表达 replicated execution。
- placement attr 中没有 SPM offset、DDR address、DTE packet 或 runtime handle。

Planner 失败也必须报告为 placement 错误，例如 rank 数超过可用 good tile、topology 维度非法或
bad tile list 越界；不应该伪装成 downstream SPM、DTE 或 runtime package 错误。

## 9. V0 范围

V0 支持：

- single-tile placement。
- single-card 2/4/8/16 tile cluster。
- 多卡多 tile 的 topology-aware deterministic mapping。
- good-tile bitmap 过滤。
- block id 与 logical rank 的一一绑定。
- local shard 绑定/验证只消费已有显式 shard facts；当前 planner 不从 tensor 名字或 payload 恢复
  shard，也不重新切分 tensor。

当前不作为 placement 基线通过标准：

- 跨卡 collective 最优 placement。
- runtime 迁移、动态 bad-tile 重映射。
- serving-level request placement。

这些不是 logical mesh / rank group 的语义不支持。若上游产出硬件 topology 可表达的跨卡 mesh，
placement IR 应保留必要 mapping / capability fact；缺少最优策略或 runtime route 时，失败应定位到
placement/runtime 恢复项，而不是让上游 sharding program 改写语义。
