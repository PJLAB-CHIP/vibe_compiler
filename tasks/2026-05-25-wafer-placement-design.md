# Wafer Placement Design

日期：2026-05-25

状态：设计草案；2026-05-25 独立边界收口

本文定义 logical mesh / rank 到 Wafer physical card/tile cluster 的 placement 边界。Placement
是 Shardy / SPMD 之后、`wafer.group` / `wafer.tile_region` / runtime launch 之前的映射合同。
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
- 为 `wafer.tile_region`、`wafer.comm` 和 `wafer.launch` 提供可验证的 physical mapping。
- 在不能得到合法 placement 时，把原因反馈给 planner 或 compile driver。

非目标：

- 不做 Linalg tiling、group fusion 或 root tile shape search。
- 不分配 `#spm` offset、DDR buffer object 或 workspace slice。
- 不选择 `Cx/NCx` physical layout。
- 不选择 ring/tree/DTE packet schedule；placement 只提供 physical endpoints 和 topology cost。
- 不把 HPGR/KMD/legacy runtime API 细节写成上层 placement 语义。

## 2. 输入和输出

输入：

```text
partitioned StableHLO / local program
  + logical mesh / rank groups
  + shard shapes and communication volume hints
  + target topology and capability
  + good-tile / PG metadata
```

输出：

```text
accepted placement map
  + logical rank -> physical coordinate
  + cluster membership
  + local shard / block id metadata
  + topology cost summary
```

accepted placement 如果影响 codegen，必须进入 IR 或 launch metadata；candidate search trace、
rejected maps、score breakdown 是 analysis，不写入长期 IR。

## 3. Physical Coordinate Model

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
多卡 mesh。V0 优先支持单卡和小规模 multi-tile；跨卡只作为 topology 可表达对象，不作为
默认必须跑通的路径。

## 4. IR Representation

accepted placement 需要在边界 op 上显式表达。推荐结构：

- 在 `wafer.launch` 或被 launch outline 的 function 上保存 cluster-level placement mapping。
- 在 `wafer.tile_region` lowering 时把 per-tile logical id / block id / physical coordinate 作为
  region argument、constant-like descriptor 或 launch argument 传入。
- 在 `wafer.comm` lowering 时通过 placement mapping 查 physical source/destination tile。

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
P4.2 可以把 block id、local shard metadata 或 capability reference 接到 package/launch
metadata，但不能改变 tensor semantics。

P4.2 package manifest 已接入 launch-visible placement metadata：`good_tile_ids` / `bad_tile_ids`
表达当前 target capability assumption，per-rank `block_id` 和 `local_shards` 进入 package
metadata。`local_shards` 只引用 launch signature tensor 名称和静态 slice bounds；它不反向修改
tensor IR shape、layout 或 sharding semantics。

## 5. Placement Algorithm

V0 使用可解释的 deterministic placement，不追求全局最优：

1. 从 target capability 构造可用 physical tile set，过滤 PG/bad tile。
2. 根据 logical mesh axes 和 communication volume 构造候选 cluster shape。
3. 优先把通信密集 axis 放在卡内 tile mesh。
4. 对每个候选检查 tile 数、cluster 连通性、rank group endpoint、runtime launch 支持。
5. 选择 cost 最低的合法候选，输出 accepted placement。

cost model 可以考虑：

- 卡内 NoC vs 跨卡 C2C 的相对代价。
- collective group 是否可以映射成局部 ring/tree。
- 每个 physical tile 的 local shard resource pressure。
- good-tile 分布导致的 hole 和 route penalty。

这些是候选排序，不是 IR contract。IR contract 只有 accepted placement mapping。

## 6. 与其它阶段的接口

| 阶段 | Placement 提供 | Placement 不提供 |
| --- | --- | --- |
| `wafer.group` | per-rank local shard / block identity | tile shape、fusion boundary |
| `wafer.tile_region` | physical tile coordinate / block id args | SPM offset、layout assignment |
| `wafer.comm` | logical endpoint 到 physical endpoint mapping | DTE node/FSM/packet allocation |
| `wafer.launch` | cluster membership and launch metadata | buffer object allocation、completion source |

Placement 不应把 DTE route 或 runtime launch API 写进上层。Communication lowering 可以基于
accepted placement 选择 ring/tree/unicast protocol。

## 7. Verifier

必须检查：

- logical rank 数量与 physical coordinate 数量一致。
- physical coordinate 在 target topology 内。
- 每个 mapped tile 通过 good-tile / PG check。
- mapping 满足 selected cluster 的 launch capability。
- logical rank group 中的 endpoints 都有 physical coordinate。
- block id / tile id 没有重复冲突，除非明确表达 replicated execution。
- placement attr 中没有 SPM offset、DDR address、DTE packet 或 runtime handle。

Verifier 失败应报告为 placement 错误，不应该伪装成 downstream SPM、DTE 或 runtime package 错误。

## 8. V0 范围

V0 支持：

- single-tile placement。
- single-card 2/4/8/16 tile cluster。
- logical DP / TP axis 到卡内 tile mesh 的简单映射。
- good-tile bitmap 过滤。

暂不作为 V0 通过标准：

- 跨卡 collective 最优 placement。
- runtime 迁移、动态 bad-tile 重映射。
- serving-level request placement。
