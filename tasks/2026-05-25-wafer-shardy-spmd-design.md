# Wafer Shardy and SPMD Design

日期：2026-05-25

状态：设计草案；2026-05-25 独立边界收口；2026-05-27 明确 P2.S1 主 pipeline 和消费链

本文定义 Wafer compiler 中 Shardy / SPMD 阶段的边界。该阶段负责 global tensor 的逻辑切分、
sharding propagation、SPMD partition 和 logical collective 语义；不负责 physical tile
placement、DTE protocol、SPM buffer、layout materialization 或 runtime launch。

本文依赖：

- `tasks/2026-05-25-wafer-frontend-stablehlo-artifact-design.md`
- `tasks/2026-05-25-wafer-placement-design.md`
- `tasks/2026-05-25-wafer-communication-dialect-design.md`

## 1. 目标和非目标

目标：

- 复用 Shardy / SDY 表达 logical mesh、axis、sharding constraint 和 propagation。
- 从 frontend artifact 中导入旧 sharding annotation。
- 生成 partitioned StableHLO local program 和 logical collective ops。
- 保留 collective group、rank、replica group 和 shard shape 这类后续 placement / comm lowering
  需要的语义事实。

非目标：

- 不选择 card/tile 物理坐标。
- 不决定 tile-goodness、PG/bad-tile fallback 或 cluster placement。
- 不生成 `wafer.comm` DTE send/recv、FSM id、stream id、packet id 或 SPM sync slot。
- 不做 group fusion、SPM allocation、DDR allocation 或 C ABI lowering。

## 2. 输入和输出

输入：

```text
StableHLO module
  + Shardy/SDY annotations or importable sharding attrs
  + logical mesh config
```

输出：

```text
partitioned StableHLO module
  + shard-local function body
  + logical collective ops
  + logical mesh / rank group metadata
```

SPMD 输出仍然是逻辑程序。它只说明“哪些 logical rank 之间需要通信”，不说明“哪两个 physical
tile 通过哪个 DTE resource 通信”。

### 2.1 P2.S1 Shardy Propagation / SPMD Pipeline Contract

P2.S1 在 R3 之前完成。它必须消费 P2.F1 产出的 verified frontend artifact，而不是只 parse
手写 `sdy.mesh` fixture。主 pipeline 边界是：

```text
verified StableHLO / SDY artifact
  -> sharding import / normalization
  -> Shardy propagation
  -> Shardy SPMD partition
  -> per-rank artifact selection
  -> per-rank artifact verifier
  -> Wafer collective / placement input normalization
```

P2.S1 输出仍然是 logical per-rank artifact bundle：

- per-rank StableHLO / func module。
- local rank、global rank、replica group、rank group 和 mesh axis metadata。
- local shard shape、dtype、user-visible input/output shard relation。
- logical collective ops 或能被 collective normalization 解释的 StableHLO / SDY metadata。

这些字段必须来自 IR、SDY attr、StableHLO collective metadata 或 verifier 可检查的 sidecar/config。
不能靠 pass side table、function name、parameter name 或 dump 文件名恢复。

V0 可继续限制 unsupported strategy，但限制必须以 legality diagnostic 形式暴露。例如 multi
replica group、rank selection policy 或 shard slicing 若尚未支持，必须在 P2.S1 verifier /
pipeline legality 中拒绝，不能静默退回到 single group fixture。

## 3. Logical Mesh Contract

Logical mesh 是 model-level parallelism 的语义对象：

- mesh axes 表示数据并行、张量并行、pipeline 并行等逻辑维度。
- rank group 和 replica group 由 sharding propagation / partition 产生。
- logical rank identity 可以作为 placement 输入，但不是 physical tile id。
- logical mesh 可以大于单卡或跨卡；physical feasibility 由 placement 和 runtime capability
  阶段判断。

如果某个 parallel strategy 只能用 physical tile name 表达，说明它不属于 Shardy / SPMD 阶段。

## 4. Collective Contract

V0 关注以下 StableHLO collective 语义：

| collective | SPMD 语义 | 下游 owner |
| --- | --- | --- |
| `collective_permute` | logical point-to-point value movement | placement + `wafer.comm` |
| `all_gather` | shard concat / replication | `wafer.comm` collective lowering |
| `reduce_scatter` | reduce + shard distribution | `wafer.comm` collective lowering |
| `all_reduce` | all-rank reduction | `wafer.comm` collective lowering |

`all_to_all` 不作为 V0 主线。它可以保留为后续扩展，但不能让 V0 communication 设计被它的
复杂 schedule 牵引。

Collective lowering 分两步：

```text
StableHLO logical collective
  -> wafer.comm collective-level op or explicit p2p schedule
  -> Direct DTE / sync / wait lower-level op
```

Shardy / SPMD 只负责第一行之前的 logical collective 生成。

当前实现新增 `--wafer-lower-stablehlo-collectives-to-comm` 作为第一步 normalization：
single-result StableHLO `all_gather`、`all_reduce` 和 `reduce_scatter` 会降到 collective-level
`wafer.comm` op。Pass 通过 `local-rank` option 表达当前 partition 在 replica group 中的 rank，
从 `replica_groups` 推出 group size，并只接受 sum/max/min 这三类 reduction body。StableHLO
`reduce_scatter` 的 V0 输出先显式进入 slot-level `wafer.comm.reduce_scatter`；从 full input 到
local scatter slot 的临时物化由 visible `unrealized_conversion_cast` 表达，后续应由 buffer-slice
IR 或 layout/materialization pass 收敛，不作为隐藏 side table。

2026-05-26 R2.2 恢复了 SDY artifact bridge 的工程入口：`WAFER_ENABLE_SPMD_PARTITIONER_DEPS=ON`
时，`wafer-opt` 和 frontend verifier tool 显式注册 Shardy / SDY dialect，`wafer-opt` 也注册
SDY passes/pipelines。带 `sdy.mesh` / `sdy.sharding` 的 partitioned StableHLO artifact 可以作为
Wafer 输入被 parse/verify。

同一批次把 StableHLO `replica_groups` 的 logical rank group materialize 到
`wafer.comm.all_gather`、`wafer.comm.all_reduce` 和 `wafer.comm.reduce_scatter` 的
`rank_group = array<i64: ...>` attr。`group_size` 只表示 group cardinality，`local_rank` 是当前
partition 在该 rank group 中的 index；ring lowering 通过 `rank_group` 查询 `wafer.placement.map`
的 logical-rank 映射，不再假设 logical rank 总是连续 `0..group_size-1`。V0 仍只接受单个
StableHLO replica group；多 replica-group artifact 需要后续增加全局 rank / group selection
policy。

## 5. 与 Placement 的接口

SPMD 给 placement 的输入是：

- logical mesh shape 和 axis。
- local shard shape、dtype、semantic layout。
- logical rank group / replica group。
- collective communication pattern。
- optional cost hints，例如通信 volume 或 reuse pattern。

Placement 返回的是 accepted physical mapping。这个 mapping 一旦影响 codegen，就属于
placement IR / launch metadata，而不是 Shardy attr 的修改。

SPMD 不应该为了特定 Wafer mesh 重新解释 StableHLO semantics。physical placement 可以拒绝、
重排或拆分 logical mesh，但不能改变 SPMD partition 后的数学语义。

## 6. Pass 合同

实现职责可以拆为：

| 职责 | 输入 | 输出 |
| --- | --- | --- |
| sharding import | StableHLO + old attrs | Shardy/SDY annotations |
| propagation | partially annotated module | fully propagated or diagnosed module |
| SPMD partition | annotated global module | partitioned StableHLO |
| per-rank artifact selection | partitioned StableHLO + rank selection policy | verified local-rank StableHLO artifact |
| collective normalization | partitioned collective ops | downstream-friendly collective IR |

这些 pass 的合法输出不包含 Wafer physical memory space、tile coordinates、DTE resource 或
runtime package metadata。

## 7. Verifier and Diagnostics

必须检查：

- 每个 sharded value 的 shard rank、shape、dtype 与 global type 一致。
- logical mesh axes 和 rank group 可解释。
- collective 的 replica group、source/target rank 和 value type 一致。
- partitioned function boundary 不丢失 user-visible input/output 语义。
- SPMD 输出中没有 physical tile id、SPM offset、DTE resource 或 packet field。

诊断要把问题定位到 logical sharding，不要提前报告为 Wafer SPM/DDR/packet 错误。

## 8. 验证和下游消费

P2.S1 的完成证明必须至少覆盖：

- P2.F1 verified artifact 可以作为 Shardy pipeline 输入。
- Shardy propagation / partitioning 后能得到指定 local rank 的 per-rank artifact。
- per-rank artifact 仍通过 frontend boundary verifier 或等价 verifier，不丢 function boundary、
  dynamic bound、constant sidecar 和 sharding facts。
- StableHLO collective metadata 能继续 lower 到 `wafer.comm` `rank_group`，并被 placement /
  ring lowering 测试消费。

后续 R3/R4/R6 测试应优先复用 P2.S1 的 per-rank artifact 作为输入，逐步验证：

```text
per-rank artifact
  -> local compute normalization
  -> group candidate
  -> tile_region materialization
  -> placement / communication / resource gate
```

手写 `sdy.mesh` / StableHLO collective fixture 只保留为 dialect/verifier/unit 级测试。它不能替代
“verified frontend artifact -> Shardy pipeline -> per-rank artifact -> Wafer lowering”的主链路
证明。
