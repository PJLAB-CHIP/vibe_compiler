# Wafer Shardy and SPMD Design

日期：2026-05-25

状态：设计草案；2026-05-25 独立边界收口；2026-05-27 明确 P2.S1 主 pipeline、执行方法和消费链

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

### 2.1 P2.S1 Shardy Propagation / SPMD Artifact Contract

P2.S1 在 R3 之前完成。它必须消费 P2.F1 产出的 verified frontend artifact，而不是只 parse
手写 `sdy.mesh` fixture。主 pipeline 边界是：

```text
verified StableHLO / SDY artifact
  -> sharding import / normalization
  -> Shardy propagation
  -> per-rank artifact selection
  -> per-rank artifact verifier
  -> logical handoff artifact for placement / communication
```

P2.S1 输出仍然是 logical per-rank artifact bundle：

- per-rank StableHLO / func module。
- local rank、global rank、replica group、rank group 和 mesh axis metadata。
- local shard shape、dtype、user-visible input/output shard relation。
- logical collective ops 或能被 collective normalization 解释的 StableHLO / SDY metadata。

这些字段必须来自 IR、SDY attr、StableHLO collective metadata 或 importer 已 materialize 的
exporter metadata。
不能靠 pass side table、function name、parameter name 或 dump 文件名恢复。

P2.S1 不能以下游当前 lowering、placement 或 runtime 尚未实现为“不支持”依据。若 Shardy/SPMD
产出的合法语义能由 Wafer 硬件通信、存储或同步能力表达，但当前 Wafer IR 还没有清楚表示，P2.S1
必须先补 op / attr / type / verifier contract，或在任务队列中明确把对应下游 lowering 作为恢复项。
multi replica group、rank selection policy、shard slicing 和 collective metadata 都属于这类事实：
它们应进入 per-rank artifact 或后续 handoff IR，而不是被静默退回 single group fixture，也不能
因为某个 ring / placement pass 暂时未覆盖就被当成 SPMD 不支持。

P2.S1 只应拒绝两类输入：exporter / Shardy 产物本身非法或自相矛盾；或者目标硬件 / ABI 证据明确
无法表达该语义，且无法由已有硬件能力组合实现。诊断必须定位到当前拥有该事实的层级。

P2.S1 当前工程 gate 不把 MLIR Mesh dialect 的 `mesh-spmdization` 或 XLA/GSPMD local-body
partitioner 作为完成前置。当前主线 artifact 是 StableHLO / SDY attr 表示的 logical per-rank
handoff：它必须能由 Shardy propagation 消费，并由 Wafer verifier 检查 rank、shard 和 collective
事实。后续若接入真正的 local-body partitioner，必须消费同一份 `sdy.*` / `wafer.spmd.*` facts，
不能另起 sidecar 或把当前下游缺口反向写成 P2.S1 不支持。

#### 2.1.1 P2.S1 真实图和 sharding 覆盖矩阵

P2.S1 的主 gate 继续使用 P2.F1 的真实 4096 matmul 图，不换成小 toy model：

```text
x: tensor<4096x4096xf32>
weight: tensor<4096x4096xf32>
bias: tensor<4096xf32>

y = tanh(x @ weight + bias) + residual
```

Sharding 必须从 framework frontend 提供的 mark 接口进入 artifact，例如 PyTorch/XLA 的
`mark_sharding` 或 export 可追踪的等价前端 op。测试和实现不得手写 `sdy.sharding` 作为主链路
输入，也不得用 Wafer 私有 JSON / sidecar 描述 sharding。每种策略都应导出同一格式的
PyTorch/XLA StableHLO bundle，并由 Shardy / SPMD pipeline 消费：

```text
<strategy>/
  functions/forward.mlir
  functions/forward.meta
  functions/forward.bytecode
  data/weight
  data/bias
```

P2.S1 至少覆盖以下常见 sharding 策略。表中的 `dp` / `tp` 是 logical mesh axis；`None` 表示该
tensor 维度在对应策略中 replicated，不表示缺少 metadata。

| 策略 | 典型 mark | 期望 SPMD 语义 |
| --- | --- | --- |
| data / batch sharding | `x: (dp, None)`；`weight`、`bias` replicated | 输出按 batch 维切分；matmul 本身不需要 collective |
| column parallel / output-feature sharding | `weight: (None, tp)`；`bias: (tp,)`；`x` replicated | 输出按 `N` 维切分；后续若要求 full output 才需要 gather |
| row parallel / contracting-dim sharding | `x: (None, tp)`；`weight: (tp, None)` | `K` 维 partial sum；per-rank artifact 必须保留 reduction collective 语义 |
| 2D output sharding | mesh `dp x tp`；`x: (dp, None)`；`weight: (None, tp)`；`bias: (tp,)` | 输出同时按 `B` / `N` 维切分；保留 2D logical mesh 和 rank group |
| 2D contracting + output sharding | mesh `dp x tp x mp`；`x: (dp, mp)`；`weight: (mp, tp)`；`bias: (tp,)` | 输出按 `B` / `N` 分布，同时 `K` 维沿 `mp` group reduction；`mp` 是 logical reduction axis，不是 physical pipeline/MoE axis |
| partial replication | 某些 tensor 在一个 mesh axis 上 sharded、在另一 axis 上 replicated，例如 `bias` 在 `dp` 上 replicated、在 `tp` 上 sharded | verifier 必须能解释 subgroup replication，不把 replicated axis 丢成默认全复制 |

这些 case 是 P2.S1 的覆盖矩阵，不是新的长期协议对象。长期合同仍是 IR 中的 logical mesh、sharding
annotation、rank group、local shard relation 和 collective metadata。pipeline parallel、MoE /
expert sharding、真实 sequence parallel 和非整除 uneven slicing 不进入第一批 P2.S1 主 gate；它们
需要对应图结构、routing/stage 语义或单独 legality/verifier 覆盖，不能通过在当前 matmul case 上
硬塞名字来冒充支持。

#### 2.1.2 P2.S1 执行方法

P2.S1 的 artifact 生成入口是：

```text
tools/wafer_pytorch_xla_capture.py \
  --emit-p2s1-sharded-matmul \
  --output-root <artifact-root> \
  --global-rank <logical-rank>
```

该命令必须运行在 P2.F1 同一套 importer Python / source-built `torch_xla` 环境里。`--global-rank`
选择 mesh 中要导出的 logical rank；artifact module 中的 `wafer.spmd.global_rank` 保存该 rank，
`wafer.spmd.local_rank` 保存它在当前 `rank_group` 内的 index。对于没有 collective 的策略，
`rank_group` 是完整 logical mesh；对于 row / contracting 或 2D contracting + output，`rank_group`
由 `collective_axis` 推出。

每个 strategy 的前端模型仍然是同一个 4096 matmul smoke module。forward 中只在 framework
边界做三类 mark：

```text
x      -> input_spec
weight -> weight_spec
bias   -> bias_spec
```

当前 PyTorch/XLA 的 `dynamo_mark_sharding` 能被 `torch.export.export` 追踪；本地源码栈中，
直接让 PyTorch/XLA StableHLO bundle exporter 输出带 mark 的文本/bytecode 还不能稳定作为主链路
artifact。因此 adapter 的做法是：

1. 对带 mark 的 module 先跑 `torch.export.export`，证明 sharding 来自 frontend mark，而不是测试
   手写 `sdy.sharding`。
2. 复用 P2.F1 的 PyTorch/XLA StableHLO bundle exporter 生成同一 4096 matmul 的
   `functions/forward.mlir`、`functions/forward.meta`、`functions/forward.bytecode` 和
   `data/weight` / `data/bias`。
3. 在同一个 `forward.mlir` 内 normalize frontend mark：插入 `sdy.mesh`，并给 function
   arguments/results 写入 `sdy.sharding`、`wafer.spmd.global_shape`、`wafer.spmd.local_shape` 和
   `wafer.spmd.shard_offsets`。
4. 在 module attr 写入 `wafer.spmd.strategy`、`wafer.spmd.global_rank`、
   `wafer.spmd.local_rank`、`wafer.spmd.mesh_shape`、`wafer.spmd.rank_group`；需要 reduction 的策略
   额外写入 `wafer.spmd.collective_kind = "all_reduce"` 和 `wafer.spmd.collective_axis`。

这一步不是第二套 sidecar 协议：没有 `wafer_sharding.json`，后续 pass 和 verifier 只消费
StableHLO / SDY / `wafer.spmd.*` IR facts。等 PyTorch/XLA exporter 能直接稳定导出 SDY 或
等价 sharding metadata 时，可以替换掉 normalization 实现，但同一份 per-rank artifact contract
不变。

per-rank verifier 入口是：

```text
wafer-import-model --verify-spmd-bundle <strategy-bundle>
```

该 verifier 先复用 StableHLO bundle metadata / weight file 检查，再检查 SPMD 事实：

- module 必须有 `sdy.mesh`、strategy、global/local rank、mesh shape 和 rank group。
- `wafer.spmd.local_rank` 必须是 `rank_group` 内 index，并且 `rank_group[local_rank]` 必须等于
  `wafer.spmd.global_rank`。
- function boundary 的每个 tensor argument/result 必须有 `sdy.sharding`、global shape、local
  shape 和 shard offset，且 local shard 必须落在 global tensor shape 内。
- collective metadata 目前明确接受 logical `all_reduce`，并要求 rank group 至少两个 rank。
- artifact 中不得出现 physical tile、DTE packet、SPM offset 或 runtime handle 这类下游 lowering
  metadata。

Shardy propagation gate 直接消费每个 `functions/forward.mlir`：

```text
shardy-sdy-opt <strategy>/functions/forward.mlir --sdy-propagation-pipeline
```

这个 gate 用于证明 SDY dialect / propagation pipeline 能读取真实 artifact 中的 sharding facts。
row / contracting 类 case 需要保留 reduction collective metadata；如果后续 local-body partitioner、
collective lowering、placement 或 ring/resource path 还没有完整消费这些事实，应补对应下游任务，
不回头把该策略从 P2.S1 artifact gate 中删掉。

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
| `all_to_all` | split / exchange / concatenate across logical ranks | `wafer.comm` collective lowering or explicit p2p schedule |

`all_to_all` 的高性能算法可以后于 ring all-gather / all-reduce 实现，但 P2.S1 不应因为当前
communication lowering 未实现该算法而丢失或拒绝它的 logical collective 语义。若硬件 data plane
只能通过 unicast Direct DTE 组合实现，per-rank artifact 仍要保留 split / exchange / concat 的
rank group、slice 和 dtype 事实，后续 communication lowering 再选择 p2p schedule。

Collective lowering 分两步：

```text
StableHLO logical collective
  -> wafer.comm collective-level op or explicit p2p schedule
  -> Direct DTE / sync / wait lower-level op
```

Shardy / SPMD 只负责第一行之前的 logical collective 生成。

当前实现已有 `--wafer-lower-stablehlo-collectives-to-comm` 作为第一步 normalization：
single-result StableHLO `all_gather`、`all_reduce` 和 `reduce_scatter` 会降到 collective-level
`wafer.comm` op。Pass 通过 `local-rank` option 表达当前 partition 在 replica group 中的 rank，
从 `replica_groups` 推出 group size，并只接受 sum/max/min 这三类 reduction body。StableHLO
`reduce_scatter` 的 V0 输出先显式进入 slot-level `wafer.comm.reduce_scatter`；从 full input 到
local scatter slot 的临时物化由 visible `unrealized_conversion_cast` 表达，后续应由 buffer-slice
IR 或 layout/materialization pass 收敛，不作为隐藏 side table。这些是当前 bridge 的覆盖状态，
不是 P2.S1 的语义上限；P2.S1 若遇到硬件可表达但当前 bridge 未覆盖的 collective，应扩展
collective-level IR 或保留 StableHLO / SDY metadata，而不是把 lowering 缺口写成 SPMD reject。

2026-05-26 R2.2 恢复了 SDY artifact bridge 的工程入口：`WAFER_ENABLE_SPMD_PARTITIONER_DEPS=ON`
时，`wafer-opt` 和 frontend verifier tool 显式注册 Shardy / SDY dialect，`wafer-opt` 也注册
SDY passes/pipelines。带 `sdy.mesh` / `sdy.sharding` 的 partitioned StableHLO artifact 可以作为
Wafer 输入被 parse/verify。

同一批次把 StableHLO `replica_groups` 的 logical rank group materialize 到
`wafer.comm.all_gather`、`wafer.comm.all_reduce` 和 `wafer.comm.reduce_scatter` 的
`rank_group = array<i64: ...>` attr。`group_size` 只表示 group cardinality，`local_rank` 是当前
partition 在该 rank group 中的 index；ring lowering 通过 `rank_group` 查询 `wafer.placement.map`
的 logical-rank 映射，不再假设 logical rank 总是连续 `0..group_size-1`。当前 bridge 对多
StableHLO replica group 的覆盖仍不完整；P2.S1 的任务是补全全局 rank / group selection policy
和 per-rank artifact 表示，不能把单 replica group 当成长期 SPMD contract。

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

## 8. 验证和导出

P2.S1 的完成证明必须至少覆盖：

- P2.F1 verified artifact 可以作为 Shardy pipeline 输入。
- P2.F1 4096 matmul 图在 data / batch、column parallel、row / contracting、2D output、2D
  contracting + output 和 partial replication 策略下都能通过 frontend mark 导出 sharding
  artifact；主 gate 不以手写 `sdy.sharding` fixture 代替真实导出。
- Shardy propagation 能直接消费每个 strategy 的 `functions/forward.mlir`；指定 global rank 的
  logical per-rank artifact 由同一 bundle 中的 `sdy.*` / `wafer.spmd.*` facts 表示并通过 verifier。
- per-rank artifact 仍通过 frontend boundary verifier 或等价 verifier，不丢 function boundary、
  dynamic bound、bundle-derived constant facts 和 sharding facts。
- per-rank artifact 完整保留 logical collective、replica group / rank group、local rank、local shard
  shape、dtype 和 user-visible input/output shard relation。
- 对当前已有 `wafer.comm` 表示的 collective，可以增加 handoff smoke 证明 metadata 能进入
  `wafer.comm` `rank_group`；但下游 ring / placement / resource lowering 的当前覆盖范围不能作为
  P2.S1 的支持边界。

后续 R3/R4/R6 测试应优先复用 P2.S1 的 per-rank artifact 作为输入，逐步验证：

```text
per-rank artifact
  -> local compute normalization
  -> group candidate
  -> tile_region materialization
  -> placement / communication / resource gate
```

手写 `sdy.mesh` / StableHLO collective fixture 只保留为 dialect/verifier/unit 级测试。它不能替代
“verified frontend artifact -> Shardy pipeline -> per-rank artifact”的主链路证明。

因此，P2.S1 之后每个消费 sharding / per-rank artifact 的任务完成时，都必须继续使用真实图导出的
artifact chain 做端到端 gate。测试不能只构造一个新的手写 per-rank fixture，也不能只检查当前层
dump；必须证明前序 sharding facts 在本任务边界的 verifier、lowering、placement、communication
或 resource 逻辑中被实际使用。若直接下游尚未支持某个硬件可表达语义，应把缺口落成下游恢复任务
或补充 IR 表示，而不是修改上游 artifact 让其避开该语义。
