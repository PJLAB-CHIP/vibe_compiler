# Wafer Shardy and SPMD Design

日期：2026-05-25

状态：设计草案；2026-05-25 独立边界收口；2026-05-27 纠正 P2.S1 主 pipeline、禁止
`wafer.spmd.*` 私有 sharding 协议，并引入 post-SPMD tensor collective handoff

本文定义 Wafer compiler 中 Shardy / SPMD 阶段的边界。该阶段只在 frontend artifact 带有
`mark_sharding` 或其它可解释 sharding annotation 时参与，负责 global tensor 的逻辑切分、
sharding propagation、SPMD partition 和 logical collective 语义；不负责 physical tile
placement、DTE protocol、SPM buffer、layout materialization 或 runtime launch。没有 sharding
annotation 的 artifact 不是 SPMD 错误，语义是未切分的普通 StableHLO local program，应绕过
SPMD partition 分支继续进入 local compute / group / tiling / codegen 主线。

本文依赖：

- `tasks/2026-05-25-wafer-frontend-stablehlo-artifact-design.md`
- `tasks/2026-05-25-wafer-placement-design.md`
- `tasks/2026-05-25-wafer-communication-dialect-design.md`
- `tasks/2026-05-25-wafer-local-compute-normalization-design.md`

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
  + explicit Shardy/SDY annotations or importable sharding attrs
  + logical mesh config when sharding is requested
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

缺少 sharding annotation 时，本阶段的正确行为是 identity / bypass：不生成 partitioned
StableHLO、不生成 logical collective，也不要求存在 mesh metadata。该输入继续作为 unpartitioned
StableHLO local program 被后续 local compute normalization 消费。

### 2.1 P2.S1 Shardy Propagation / SPMD Artifact Contract

P2.S1 在 R3 之前完成。它覆盖“存在显式 sharding 标记”的 artifact 分支，必须消费 P2.F1 产出的
verified frontend artifact，而不是只 parse 手写 `sdy.mesh` fixture。带 sharding 的主 pipeline
边界是：

```text
verified StableHLO / SDY artifact
  -> sharding import / normalization
  -> Shardy propagation
  -> XLA SPMD partitioner / equivalent local-body partitioning service
  -> partitioned StableHLO artifact
  -> per-rank artifact verifier
  -> logical handoff artifact for local compute / tensor collective normalization
```

P2.S1 输出仍然是 logical partitioned artifact bundle：

- partitioned StableHLO / func module，或等价的 per-logical-rank StableHLO module。
- Shardy / StableHLO / exporter 可解释的 logical mesh、replica group、rank group 和 mesh axis metadata。
- local shard shape、dtype、user-visible input/output shard relation。
- StableHLO logical collective ops，例如 `all_gather`、`all_reduce`、`reduce_scatter`、
  `all_to_all` 和 `collective_permute`。

没有 sharding 标记的 P2.F1 artifact 不进入这个 partitioned artifact 合同；它的合同是普通
StableHLO local program 可继续 lower。不能把“没有 sharding metadata”诊断成 P2.S1 verifier
失败，也不能为了让它经过 P2.S1 而补默认私有 sharding。

这些字段必须来自 IR、SDY attr、StableHLO collective metadata 或 importer 已 materialize 的
exporter metadata。
不能靠 pass side table、function name、parameter name 或 dump 文件名恢复。
也不能通过 `wafer.spmd.*` module/function attrs、私有 JSON 或名字约定创造第二套 sharding /
per-rank 协议。若需要保存不能从 StableHLO/SDY 重新推出、且影响 codegen 的 rank/placement
事实，应在 placement / launch 边界定义明确 IR 或 metadata，而不是污染 SPMD artifact。

P2.S1 不能以下游当前 lowering、placement 或 runtime 尚未实现为“不支持”依据。若 Shardy/SPMD
产出的合法语义能由 Wafer 硬件通信、存储或同步能力表达，但当前 Wafer IR 还没有清楚表示，P2.S1
必须先补 op / attr / type / verifier contract，或在任务队列中明确把对应下游 lowering 作为恢复项。
multi replica group、rank selection policy、shard slicing 和 collective metadata 都属于这类事实：
它们应进入 per-rank artifact 或后续 handoff IR，而不是被静默退回 single group fixture，也不能
因为某个 ring / placement pass 暂时未覆盖就被当成 SPMD 不支持。

P2.S1 只应拒绝两类输入：exporter / Shardy 产物本身非法或自相矛盾；或者目标硬件 / ABI 证据明确
无法表达该语义，且无法由已有硬件能力组合实现。诊断必须定位到当前拥有该事实的层级。

P2.S1 当前工程 gate 必须把 XLA SPMD partitioner 或等价 local-body partitioning service 纳入主线
完成证明。只运行 Shardy propagation、只 parse `sdy.mesh`，或只把 frontend mark normalize 成
`sdy.sharding`，都不能证明 per-rank / partitioned body 已产生。若当前第三方版本或 API 暂时无法
稳定导出 partitioned StableHLO，应把它记录为 P2.S1 blocker / recovery task；不得用
`wafer.spmd.*` 临时 attrs 冒充 partitioner 输出。

#### 2.1.1 P2.S1 真实图和 sharding 覆盖矩阵

P2.S1 的 sharding branch 主 gate 继续使用 P2.F1 的真实 4096 matmul 图，不换成小 toy model：

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

同一个 4096 matmul 还应保留 no-sharding 覆盖：不调用 `mark_sharding` 时，导出的 artifact
没有 sharding metadata，pipeline 不进入 P2.S1 partitioner 分支，而是作为未切分 StableHLO 图
继续验证 local compute normalization 和后续 lowering。这个 no-sharding case 不计入上表的
sharding 策略数量，也不应被写成 replicated sharding 的另一种私有表示。

#### 2.1.2 P2.S1 执行方法

P2.S1 的 artifact 生成入口应保持使用同一 4096 matmul source graph，但实现路线必须是：

```text
PyTorch module
  -> torch_xla.distributed.spmd.mark_sharding / torch.ops.xla.dynamo_mark_sharding
  -> PyTorch/XLA StableHLO / SDY artifact
  -> Shardy propagation
  -> XLA SPMD partitioner
  -> partitioned StableHLO artifact
```

该路径必须运行在 P2.F1 同一套 importer Python / source-built `torch_xla` 环境里。`torch_xla`
来自 `third_party/pytorch-xla` 源码构建/安装，并复用本仓库固定的 LLVM/MLIR、StableHLO、Shardy
和 XLA 版本；不得改用 prebuilt `torch_xla` wheel 或新下载另一套 XLA/LLVM。

logical rank / local rank 的选择属于 partitioner invocation、artifact export 或后续 placement /
launch context。它不能通过 `wafer.spmd.global_rank`、`wafer.spmd.local_rank`、
`wafer.spmd.rank_group` 这类 Wafer 私有 attr 写回 StableHLO module。rank group 应优先来自
partitioned StableHLO collective 的 `replica_groups` / `source_target_pairs` / Shardy metadata；
如果某个 rank selection policy 不能从这些事实重建，先补明确 artifact/export contract 或 placement
IR，而不是发明 SPMD-side Wafer attr。

每个 strategy 的前端模型仍然是同一个 4096 matmul smoke module。forward 中只在 framework
边界做三类 mark：

```text
x      -> input_spec
weight -> weight_spec
bias   -> bias_spec
```

当前已知错误路线是：先跑 frontend mark smoke，再在 `forward.mlir` 中手写 `sdy.mesh` /
`sdy.sharding` / `wafer.spmd.*`，然后让后续 verifier 消费这些 Wafer 私有 attrs。这个做法只是在
artifact 中补了一份临时描述，既没有证明 XLA SPMD partitioner 产生了 local body，也会把后续实现
引向错误的协议源。P2.S1 恢复时必须删除这条完成口径。

允许保留的临时测试只有两类：

- frontend mark smoke：证明 PyTorch/XLA `mark_sharding` 可被当前 capture 路径观察或追踪。
- SDY / StableHLO dialect unit fixture：证明工具链能 parse / verify / run Shardy propagation。

这两类测试都不能标记 P2.S1 完成。P2.S1 完成证明必须消费真实 mark 后的 artifact，并得到
partitioned StableHLO 或等价 per-rank StableHLO body。

partitioned artifact verifier 后续可以重新建立工具入口，但它的责任必须是检查 StableHLO / SDY /
exporter-native facts，而不是检查 `wafer.spmd.*`：

- module 中的 Shardy / SDY / StableHLO sharding metadata 必须可由注册 dialect 解释。
- partitioned function boundary 的 tensor argument/result shape、dtype 和 exporter metadata 必须一致。
- local shard relation 必须能从 partitioned StableHLO shape、SDY metadata 或 exporter-native metadata
  解释；不能依赖 parameter/function 名字。
- collective metadata 必须来自 StableHLO op，例如 `replica_groups`、`source_target_pairs`、
  `all_gather_dim`、`scatter_dimension`、`split_dimension` / `concat_dimension` 和 reduction body。
- artifact 中不得出现 physical tile、DTE packet、SPM offset 或 runtime handle 这类下游 lowering
  metadata。

Shardy propagation gate 直接消费每个 `functions/forward.mlir`：

```text
shardy-sdy-opt <strategy>/functions/forward.mlir --sdy-propagation-pipeline
```

这个 gate 只证明 SDY dialect / propagation pipeline 能读取真实 artifact 中的 sharding facts。
它必须和 XLA SPMD partitioner / partitioned StableHLO export gate 配套使用，不能单独作为 P2.S1
完成证明。row / contracting 类 case 需要保留 reduction collective op；如果后续 tensor collective
normalization、placement 或 ring/resource path 还没有完整消费这些事实，应补对应下游任务，不回头
把该策略从 P2.S1 artifact gate 中删掉。

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
| `collective_permute` | logical point-to-point value movement | tensor collective handoff + placement + later `wafer.comm` |
| `all_gather` | shard concat / replication | Wafer LinalgExt-style tensor collective handoff |
| `reduce_scatter` | reduce + shard distribution | Wafer LinalgExt-style tensor collective handoff |
| `all_reduce` | all-rank reduction | Wafer LinalgExt-style tensor collective handoff |
| `all_to_all` | split / exchange / concatenate across logical ranks | Wafer LinalgExt-style tensor collective handoff；later `wafer.comm` p2p schedule |

`all_to_all` 的高性能算法可以后于 ring all-gather / all-reduce 实现，但 P2.S1 不应因为当前
communication lowering 未实现该算法而丢失或拒绝它的 logical collective 语义。若硬件 data plane
只能通过 unicast Direct DTE 组合实现，per-rank artifact 仍要保留 split / exchange / concat 的
rank group、slice 和 dtype 事实，后续 communication lowering 再选择 p2p schedule。

Collective handoff 分三步：

```text
StableHLO logical collective
  -> Wafer LinalgExt-style tensor collective op
  -> tiled tensor collective inside scheduled group / tile_region materialization
  -> wafer.comm collective-level op or explicit p2p schedule
  -> Direct DTE / sync / wait lower-level op
```

Shardy / SPMD 只负责第一行之前的 logical collective 生成。StableHLO collective 不应在 group /
tiling 前直接 lower 成 `wafer.comm`，因为 `wafer.comm` 当前属于 tile-local buffer / communication
IR；它需要 SPM buffer、byte count、placement 和 token/effect 语义。

旧的 StableHLO collective 直降 `wafer.comm.*` pass 已移除。后续不得恢复 group/tiling 前的
StableHLO -> `wafer.comm` 插入点；需要分别实现“StableHLO -> tensor collective”和
“tiled tensor collective -> wafer.comm”两层。

2026-05-26 R2.2 恢复了 SDY artifact bridge 的工程入口：`WAFER_ENABLE_SPMD_PARTITIONER_DEPS=ON`
时，`wafer-opt` 和 frontend verifier tool 显式注册 Shardy / SDY dialect，`wafer-opt` 也注册
SDY passes/pipelines。带 `sdy.mesh` / `sdy.sharding` 的 partitioned StableHLO artifact 可以作为
Wafer 输入被 parse/verify。

同一批次曾把 StableHLO `replica_groups` 的 logical rank group materialize 到 `wafer.comm.*`
`rank_group = array<i64: ...>` attr；该 StableHLO -> `wafer.comm` bridge 已移除，避免后续误把它当成
group/tiling 输入。P2.S1 的任务是通过真实 partitioner 输出和 artifact verifier 保留这些事实，
R2.4/R6 再分别恢复 tensor collective handoff 和 tile-local comm lowering。

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
| collective tensor normalization | partitioned StableHLO collective ops | Wafer LinalgExt-style tensor collective IR |
| communication materialization | tiled tensor collective + tile buffers + placement | `wafer.comm` collective-level op or explicit p2p schedule |

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
- Shardy propagation 能直接消费每个 strategy 的 `functions/forward.mlir`，并且 XLA SPMD
  partitioner / equivalent service 能产出 partitioned StableHLO 或等价 per-rank StableHLO body。
- per-rank artifact 仍通过 frontend boundary verifier 或等价 verifier，不丢 function boundary、
  dynamic bound、bundle-derived constant facts 和 sharding facts。
- per-rank artifact 完整保留 logical collective、replica group / rank group、local rank、local shard
  shape、dtype 和 user-visible input/output shard relation。
- StableHLO collective 能先进入 Wafer LinalgExt-style tensor collective handoff；对当前已有
  `wafer.comm` 表示的 collective，可以保留后段 smoke 证明 metadata 能进入 `wafer.comm`
  `rank_group`，但该 smoke 不能作为 P2.S1 或 group/tiling 完成证明。

另外，no-sharding P2.F1 artifact 必须保持可 lower：没有 `mark_sharding` 时不能要求 Shardy /
SPMD metadata，也不能因为没有 partitioned StableHLO 或 logical collective 而阻塞 local compute
normalization、group/tiling 和后端 lowering。这个 gate 属于普通 local compile 主线，不属于
P2.S1 sharding 覆盖矩阵。

对 sharded 分支，后续 R3/R4/R6 测试应优先复用 P2.S1 的 per-rank artifact 作为输入，逐步验证：

```text
per-rank artifact
  -> local compute normalization
  -> tensor collective normalization if collectives exist
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
