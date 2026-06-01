# Wafer Shardy and SPMD Design

日期：2026-05-25

状态：设计草案；2026-05-25 独立边界收口；2026-05-27 纠正 P2.S1 主 pipeline、禁止
`wafer.spmd.*` 私有 sharding 协议，并引入 post-SPMD tensor collective handoff；2026-06-01
删除误导性的 Python post-SPMD helper，明确 P2.S1/P2.S2/R2.4/local compute 的 pass 接入边界；
2026-06-01 接通真实 pinned-XLA helper 的 P2.S2 artifact gate

本文定义 Wafer compiler 中 Shardy / SPMD 阶段的边界。该阶段负责 global tensor 的逻辑切分、
sharding propagation、SPMD partition 和 logical collective 语义；不负责 physical tile
placement、DTE protocol、SPM buffer、layout materialization 或 runtime launch。

没有用户 `mark_sharding` 或其它可解释 sharding seed 的 artifact 不是 frontend 错误，但也不应
直接绕过 SPMD 去后段补切分。P2.S1 应在 SPMD 层应用 Wafer 默认单卡 sharding policy：默认面向
16 个 tile 生成 Shardy / SDY 可解释的 function-input sharding seed，找不到合适切分维度时生成
16-tile replicated seed；调试时可配置为 1 tile replicated 模式。后续切图、local body 和通信算子
插入仍交给 Shardy / XLA SPMD partitioner。

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
  + optional user Shardy/SDY annotations or importable sharding attrs
  + default Wafer single-card mesh policy when user sharding is absent
```

输出：

```text
partitioned StableHLO module
  + shard-local function body
  + logical collective ops
  + logical mesh / rank group metadata
```

SPMD 输出仍然是逻辑程序。它只说明“哪些 logical rank 之间需要通信”，不说明“哪两个 physical
tile 通过哪个 DTE resource 通信”。当默认 policy 退化为 1 tile 或 replicated local body 时，
partitioner 可以产生等价的 single-program / replicated-local StableHLO；这仍是 SPMD 层的输出，
不是后段 placement/group 私自补 sharding。

### 2.1 P2 Shardy Propagation / SPMD Artifact Contract

本节描述 P2 SPMD artifact 的终态合同；当前实现拆成 P2.S1 Shardy propagation stage 和待做的
P2.S2 Wafer-owned XLA SPMD partition stage。终态覆盖两类 artifact：用户显式标记 sharding 的图，
以及完全没有用户 sharding seed、需要 Wafer 默认单卡 policy 的图。SPMD pipeline 必须消费 P2.F1
产出的 verified frontend artifact，而不是只 parse 手写 `sdy.mesh` fixture。主 pipeline 边界是：

```text
verified StableHLO / optional SDY artifact
  -> if user sharding seed exists:
       sharding import / normalization
     else:
       default Wafer single-card input sharding seed
  -> Shardy propagation
  -> XLA SPMD partitioner / equivalent local-body partitioning service
  -> partitioned StableHLO artifact
  -> per-rank artifact verifier
  -> logical handoff artifact for local compute / tensor collective normalization
```

P2.S2 输出仍然是 logical partitioned artifact bundle：

- partitioned StableHLO / func module，或等价的 per-logical-rank StableHLO module。
- Shardy / StableHLO / exporter 可解释的 logical mesh、replica group、rank group 和 mesh axis metadata。
- local shard shape、dtype、user-visible input/output shard relation。
- StableHLO logical collective ops，例如 `all_gather`、`all_reduce`、`reduce_scatter`、
  `all_to_all` 和 `collective_permute`。

没有用户 sharding seed 的 P2.F1 artifact 也进入 SPMD stage，但只允许由默认 policy 补 Shardy / SDY
可解释的 function-input sharding seed。不能把“没有用户 sharding metadata”诊断成 P2.S1 verifier
失败，也不能补 `wafer.spmd.*`、私有 JSON、名字约定或后段 placement fallback。

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

P2.S2 工程 gate 必须把 XLA SPMD partitioner 或等价 local-body partitioning service 纳入主线
完成证明。只运行 Shardy propagation、只 parse `sdy.mesh`，或只把 frontend mark normalize 成
`sdy.sharding`，都不能证明 per-rank / partitioned body 已产生。若当前第三方版本或 API 暂时无法
稳定导出 partitioned StableHLO，应把它记录为 P2.S2 blocker / recovery task；不得用
`wafer.spmd.*` 临时 attrs 冒充 partitioner 输出。

当前 pass / driver 接入边界如下，pass 名只作为实现入口，不能被提升成 IR 层名词：

- frontend Python capture 只负责导出 exporter-native StableHLO bundle。它可以调用
  PyTorch/XLA `mark_sharding` 产生 pre-SPMD sharding seed，但不能执行 Wafer Shardy propagation、
  XLA SPMD partition 或写 per-rank artifact。
- `wafer-import-model --propagate-stablehlo-sharding` 是 driver 阶段检查入口：先验证
  `functions/forward.mlir` / `forward.meta` / parameter payload，再调用同一套
  `wafer-propagate-stablehlo-sharding` pipeline。
- `wafer-propagate-stablehlo-sharding` 只包含 no-user default input seed 和 Shardy propagation。
  它输出经过 sharding propagation stage 处理后的 StableHLO/SDY IR，不输出 partitioned local body、不写
  `forward.parameter_shards.json`、也不插入 post-SPMD collective。
- P2.S2 必须是 Wafer-owned artifact stage 或等价 library/driver：消费经过 Wafer sharding
  propagation stage 的 StableHLO bundle，显式完成 StableHLO/SDY -> XLA HLO、XLA SPMD
  partitioner、partitioned HLO -> StableHLO round trip，再写 post-SPMD local / replicated-local
  StableHLO bundle 和 rank-local parameter shard binding。`P2.S1` / `P2.S2` 只能作为任务索引，
  不能成为 bundle 目录名、artifact 类型名或长期协议字段。
- R2.4 消费 P2.S2 的 partitioned StableHLO collective，并 normalize 到 Wafer LinalgExt-style
  tensor collective。它不是 XLA SPMD partitioner，也不能从 P2.S1 的 propagated global module
  直接补 `wafer.comm`。
- `wafer-lower-stablehlo-to-linalg` 消费 post-SPMD local body 或 no-sharding replicated local body，
  只做 StableHLO local compute -> Linalg/Tensor/Arith/Math；它不做 sharding propagation、SPMD
  partition、placement、group、SPM/DDR 或 communication materialization。

2026-06-01 当前实现口径：

- frontend Python test generator 的职责只到 PyTorch/XLA StableHLO export：未标记图导出 reference
  bundle；用户 sharding 分支用 `mark_sharding` 标记同一个 4096 matmul 图的 `x`、`weight`、`bias`，
  导出带 `mhlo.sharding` 的 pre-SPMD PyTorch/XLA StableHLO bundle。
- `wafer-propagate-stablehlo-sharding` / `wafer-import-model --propagate-stablehlo-sharding`
  作为 Wafer 侧的 Shardy propagation 入口。它消费 pre-SPMD bundle，补 no-user default input
  seed，并调用 Shardy propagation；它不产生 partitioned local body。
- P2.S2 当前由 Wafer driver mode 加 pinned-XLA helper/service 实现。`wafer-import-model
  --partition-stablehlo-bundle` 先验证输入 bundle，再在 Wafer 侧执行 default input seed + Shardy
  propagation，随后把 propagated bundle 交给 helper。helper 显式执行 StableHLO/SDY -> XLA HLO、
  `SpmdPrepare` / `SpmdPartitioner` / `HloVerifier`、partitioned HLO -> StableHLO round trip，并
  写回 partitioned StableHLO bundle。
- 2026-06-01 直接 CMake link 评估结论：当前 build 虽启用 `WAFER_ENABLE_SPMD_PARTITIONER_DEPS`，
  但 CMake target graph 只包含 Wafer / StableHLO / Shardy，没有 XLA `spmd_partitioner`、HLO
  service、TSL、Abseil 或 generated XLA proto targets。为避免把任务拖进 XLA CMake shim，P2.S2
  采用 `tools/build_xla_spmd_partitioner_helper.py` 生成 pinned `third_party/xla` Bazel overlay 并构建
  helper；helper 只是当前工程接入方式，不是新的 IR 层、bundle 名称或长期协议对象。
- Wafer-owned Shardy / SPMD 实现源码归属 `lib/Wafer/Transforms/SPMD/`。只依赖 MLIR / StableHLO /
  Shardy CMake target 的 pass 应编进 `WaferTransforms`；必须直接使用 XLA HLO service / SPMD
  partitioner / generated proto / TSL 的入口可以从同一 Wafer 源码树映射进 pinned XLA Bazel overlay
  编译。当前 `lib/Wafer/Transforms/SPMD/XlaSpmdPartitionerMain.cpp` 就是这种 Wafer-owned stage
  adapter，`tools/build_xla_spmd_partitioner_helper.py` 只是构建桥。
- 旧 Python post-SPMD helper 和相关 tests 已删除。P2.S2 的 partitioned StableHLO 主链产物只能由
  Wafer-owned artifact stage 保存；不允许用 Python helper、手写 sidecar 或 fixture 冒充这个缺口。
- no-user-sharding 分支当前只在文本 StableHLO/SDY artifact 中用
  `--wafer-apply-default-spmd-sharding` / `wafer-propagate-stablehlo-sharding` 补 function-input seed；
  P2.S2 再消费该 stage 输出 artifact。默认 policy 只标记输入/参数，不给中间 op 或 function
  result 造约束。
- P2.S2 输出的 partitioned bundle 必须由 Wafer-owned artifact stage 保存。`functions/forward.mlir`
  是 local body；`functions/forward.meta` 的 input/output signature 必须匹配 local function boundary；
  `functions/forward.parameter_shards.json` 记录 post-SPMD 后 parameter local argument 到
  `parameter_shards/<parameter>/rank_XXXXX.npy` rank-local payload 的 explicit binding，供后续
  storage/package materialization 消费。binding 的 offsets、sizes、strides 和 replica id 必须来自
  XLA sharding / runtime shard facts，不由 Wafer 从 `partition_spec`、strategy 名或 parameter 名手算。
- 当前 helper 对参数 payload 支持 row-major raw tensor 和 NumPy `.npy` v1/v2 输入，按 XLA
  `HloSharding::TileOffsetForDevice` / `TileLimitForDevice` 为每个 logical rank materialize local
  `.npy` payload；local function signature 从 post-SPMD StableHLO `func.func @main` 的 ranked tensor
  边界回写到 `forward.meta`。

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
  ...
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

同一个 4096 matmul 还应保留 no-user-sharding 覆盖：不调用 `mark_sharding` 时，导出的 artifact
没有用户 sharding seed，P2.S1 默认 policy 应补 function-input sharding seed，并继续进入
Shardy propagation；P2.S2 再进入 XLA SPMD partitioner。这个 case 不计入上表的用户 sharding
策略数量，也不能被写成 Wafer 私有 sharding；它是 SPMD 层的默认 seed policy。

#### 2.1.2 P2.S1 执行方法

P2.S1/P2.S2 的 artifact 生成入口应保持使用同一 4096 matmul source graph，但主线实现路线必须是：

```text
PyTorch module
  -> optional torch_xla.distributed.spmd.mark_sharding / torch.ops.xla.dynamo_mark_sharding
  -> PyTorch/XLA StableHLO / SDY artifact       # frontend Python stops here
  -> if no user sharding seed exists, apply Wafer default function-input seed
  -> Wafer Shardy propagation
  -> Wafer-owned XLA SPMD partition artifact stage
  -> partitioned StableHLO artifact
```

frontend export 路径必须运行在 P2.F1 同一套 importer Python / source-built `torch_xla` 环境里。`torch_xla`
来自 `third_party/pytorch-xla` 源码构建/安装，并复用本仓库固定的 LLVM/MLIR、StableHLO、Shardy
和 XLA 版本；不得改用 prebuilt `torch_xla` wheel 或新下载另一套 XLA/LLVM。

logical rank / local rank 的选择属于 partitioner invocation、artifact export 或后续 placement /
launch context。它不能通过 `wafer.spmd.global_rank`、`wafer.spmd.local_rank`、
`wafer.spmd.rank_group` 这类 Wafer 私有 attr 写回 StableHLO module。rank group 应优先来自
partitioned StableHLO collective 的 `replica_groups` / `source_target_pairs` / Shardy metadata；
如果某个 rank selection policy 不能从这些事实重建，先补明确 artifact/export contract 或 placement
IR，而不是发明 SPMD-side Wafer attr。

每个用户 sharding strategy 的前端模型仍然是同一个 4096 matmul 最小验证 module。forward 中只在
framework 边界做三类 mark：

```text
x      -> input_spec
weight -> weight_spec
bias   -> bias_spec
```

当前已知错误路线是：先跑 frontend mark 最小验证，再在 `forward.mlir` 中手写 `sdy.mesh` /
`sdy.sharding` / `wafer.spmd.*`，然后让后续 verifier 消费这些 Wafer 私有 attrs。这个做法只是在
artifact 中补了一份临时描述，既没有证明 XLA SPMD partitioner 产生了 local body，也会把后续实现
引向错误的协议源。P2.S1/P2.S2 恢复时必须删除这条完成口径。

允许保留的临时测试只有两类：

- frontend mark 最小验证：证明 PyTorch/XLA `mark_sharding` 可被当前 capture 路径观察或追踪。
- SDY / StableHLO dialect unit fixture：证明工具链能 parse / verify / run Shardy propagation。

这些测试都不能标记 P2.S2 完成。P2.S2 完成证明必须消费真实 P2.F1 artifact 和 Wafer Shardy
propagation 输出：用户策略消费真实 mark 后的 artifact；no-user-sharding 策略消费同图未标记
artifact 并由默认 policy 生成 function-input seed。两类策略都必须由 Wafer-owned SPMD partition
stage 得到 partitioned StableHLO、等价 per-rank StableHLO body，或明确的 replicated-local body。

当前验证 gate：

- `test/Pipelines/stablehlo-sharding-propagation.mlir` 和
  `test/Tools/wafer-import-model-sharding-propagation.test` 覆盖 Wafer named pipeline / driver 入口：
  pre-SPMD bundle 先通过 bundle verifier，再执行 default input seed + Shardy propagation；带
  pre-SPMD sharding seed 的 bundle 不能直接进入 C ABI lowering。
- `test/Tools/wafer-pytorch-xla-capture-sharded-bundle.test` 覆盖六种用户策略的真实
  `mark_sharding` -> pre-partition StableHLO bundle，并通过 `wafer-import-model
  --verify-stablehlo-bundle` 校验 bundle metadata / data；同一测试用
  `wafer-import-model --propagate-stablehlo-sharding` 证明 Wafer named pipeline / driver 能读取这些
  真实 artifact。
- `test/Tools/wafer-import-model-xla-spmd-partition-bundle.test` 覆盖 P2.S2 真实 gate：从
  PyTorch/XLA `mark_sharding` bundle 进入 `wafer-import-model --partition-stablehlo-bundle`，由
  Wafer driver 执行 bundle verify + Shardy propagation，再调用 pinned-XLA helper 产出 partitioned
  StableHLO bundle。该 gate 用同一个 matmul 图的 `--size 32` 形态覆盖 data、column、row、
  2d-output、2d-contracting-output 和 partial-replication 六种 strategy；这是为了让本地 helper /
  lit gate 可重放，不改变 P2.F1 4096 export 主图的语义形态。column case 额外检查 rank-local
  signature、StableHLO collective、`forward.parameter_shards.json` 和 rank-local `.npy` payload。
- 旧 Python post-SPMD helper tests 已删除。P2.S2 partitioned StableHLO 主链 gate 不能退回 Python
  helper、`wafer.spmd.*` 私有协议或手写 sidecar。

#### 2.1.3 默认 no-user-sharding policy

当一个 graph 中已经存在任何用户 sharding seed 时，默认 policy 必须完全跳过该 graph。这里的
seed 包括 function argument/result 上的 `sdy.sharding`、中间 value 的 `sdy.sharding` /
`sdy.sharding_constraint` / `sdy.reshard`、manual computation sharding，或 frontend mark 导出的
等价 SDY / StableHLO sharding 表示。用户可能只标记少数关键 op / value，再依赖 Shardy
propagation 推到整图；默认 policy 不能覆盖、补齐或重解释这些用户 seed。

当 graph 完全没有用户 sharding seed 时，P2.S1 默认 policy 创建一个 Wafer 单卡 logical mesh：

```text
@wafer_default_tile_mesh = <["tile" = N]>
```

`N` 默认是 16，用于单卡 4x4 tile；调试和对照验证可以设置 `N = 1`。这个 mesh 是 SPMD logical
mesh，不是 physical placement map，也不包含 tile coordinate、bad-tile、SPM、DTE 或 runtime fact。

默认 policy 主要标记 function inputs / parameters 的 sharding seed，不主动给中间 op 或 function
results 下约束。Shardy propagation 负责把 seed 推到内部 value 和结果，XLA SPMD partitioner
负责产生 local body 和必要 collective / permute。第一版可采用保守 deterministic heuristic：

- 对每个 ranked tensor function input，选择第一个静态且能被 `N` 整除的维度绑定 `tile` axis。
- 如果没有这样的维度，或 `N = 1`，该 input 在 `tile` axis 上 replicated。
- 不依赖 input / parameter 名字；若需要区分 user input、parameter、constant 或 state，必须来自
  frontend 已 materialize 到 IR 的可验证 metadata。
- 默认 policy 只生成 SDY / StableHLO 可解释的 sharding seed，不生成 `wafer.spmd.*`、placement op、
  package manifest 或其它后段协议。

这个 policy 的目标是让未显式标记的单卡图默认利用 16 tile，同时仍把通信插入和 per-rank local body
生成留在 SPMD partitioner 内。找不到合适切分维度时的 16-tile replicated 不是放弃 SPMD；它是
明确的 replicated sharding seed，后续仍可通过同一 SPMD pipeline 产出等价 local program。

partitioned artifact verifier 后续可以重新建立工具入口，但它的责任必须是检查 StableHLO / SDY /
exporter-native facts，而不是检查 `wafer.spmd.*`：

- module 中的 Shardy / SDY / StableHLO sharding metadata 必须可由注册 dialect 解释。
- partitioned function boundary 的 tensor argument/result shape、dtype 和 exporter metadata 必须一致。
- local shard relation 必须能从 partitioned StableHLO shape、SDY metadata 或 exporter-native metadata
  解释；不能依赖 parameter/function 名字。
- 对 resource-backed parameter，post-SPMD artifact 必须显式 materialize `logical rank ->
  rank-local shard payload` 绑定：每个 shard entry 必须有 `file`、offsets、sizes、strides、replica
  id 和 local shape。offsets/sizes/strides 必须来自 PyTorch/XLA runtime / XLA sharding spec 暴露的
  shard indices，不能由 Wafer 根据 `partition_spec`、parameter 名、strategy 名或文件名重新推断。
- collective metadata 必须来自 StableHLO op，例如 `replica_groups`、`source_target_pairs`、
  `all_gather_dim`、`scatter_dimension`、`split_dimension` / `concat_dimension` 和 reduction body。
- artifact 中不得出现 physical tile、DTE packet、SPM offset 或 runtime handle 这类下游 lowering
  metadata。

Shardy propagation gate 通过 Wafer named pipeline / driver 直接消费 `functions/forward.mlir`：

```text
wafer-import-model --propagate-stablehlo-sharding <strategy>
wafer-opt --pass-pipeline='builtin.module(wafer-propagate-stablehlo-sharding)' <strategy>/functions/forward.mlir
```

standalone `shardy-sdy-opt` 可以作为第三方 pipeline 对照，但不应是 Wafer 用户级流程的唯一入口。
这个 gate 只证明 SDY dialect / propagation pipeline 能读取真实 artifact 中的 sharding facts；它必须和
XLA SPMD partitioner / partitioned StableHLO export gate 配套使用，不能单独作为 P2.S2 完成证明。
row / contracting 类 case 需要保留 reduction collective op；如果后续 tensor collective
normalization、placement 或 ring/resource path 还没有完整消费这些事实，应补对应下游任务，不回头把
该策略从 P2 SPMD artifact gate 中删掉。

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

`all_to_all` 的高性能算法可以后于 ring all-gather / all-reduce 实现，但 SPMD artifact stage 不应因为当前
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
group/tiling 输入。SPMD artifact stage 的任务是通过真实 partitioner 输出和 artifact verifier 保留这些事实，
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

P2.S2 的完成证明必须至少覆盖：

- P2.F1 verified artifact 可以作为 Shardy pipeline 输入。
- P2.F1 matmul 图在 data / batch、column parallel、row / contracting、2D output、2D contracting +
  output 和 partial replication 策略下都能通过 frontend mark 导出 sharding artifact；frontend
  export 主图保持 4096 形态，P2.S2 helper / lit gate 可使用同构小尺寸图重放 partition artifact
  chain，避免把验证变成 4096 工作集容量测试。主 gate 不以手写 `sdy.sharding` fixture 代替真实导出。
- Shardy propagation 能直接消费每个 strategy 的 `functions/forward.mlir`，并且 XLA SPMD
  partitioner / equivalent service 能产出 partitioned StableHLO 或等价 per-rank StableHLO body。
- per-rank artifact 仍通过 frontend boundary verifier 或等价 verifier，不丢 function boundary、
  dynamic bound、bundle-derived constant facts 和 sharding facts。
- per-rank artifact 完整保留 logical collective、replica group / rank group、local rank、local shard
  shape、dtype 和 user-visible input/output shard relation。
- P2.S2 输出必须完整保留 StableHLO collective 和 metadata，供 R2.4 进入 Wafer LinalgExt-style
  tensor collective handoff；对当前已有 `wafer.comm` 表示的 collective，可以保留后段 最小验证 证明
  metadata 能进入 `wafer.comm` `rank_group`，但该 最小验证 不能作为 P2.S2、R2.4 或 group/tiling
  完成证明。
- no-user-sharding P2.F1 artifact 必须通过默认 policy 生成 function-input sharding seed：默认
  `tile-count=16`，并覆盖找不到合适切分维度时的 16-tile replicated fallback；`tile-count=1`
  作为调试模式产生 replicated/single-tile seed。
- 默认 policy 遇到 graph 内任意用户 sharding seed 时必须跳过，不覆盖用户只标了关键 op 后由
  Shardy propagation 推导整图的用法。

对 P2.S2 输出，后续 R3/R4/R6 测试应优先复用 per-rank / replicated-local artifact 作为输入，逐步验证：

```text
per-rank artifact
  -> local compute normalization
  -> tensor collective normalization if collectives exist
  -> group candidate
  -> tile_region materialization
  -> placement / communication / resource gate
```

手写 `sdy.mesh` / StableHLO collective fixture 只保留为 dialect/verifier/unit 级测试。它不能替代
“verified frontend artifact -> Wafer Shardy propagation -> Wafer-owned SPMD partition -> per-rank artifact”的主链路证明。

因此，P2.S2 之后每个消费 sharding / per-rank artifact 的任务完成时，都必须继续使用真实图导出的
artifact chain 做端到端 gate。测试不能只构造一个新的手写 per-rank fixture，也不能只检查当前层
dump；必须证明前序 sharding facts 在本任务边界的 verifier、lowering、placement、communication
或 resource 逻辑中被实际使用。若直接下游尚未支持某个硬件可表达语义，应把缺口落成下游恢复任务
或补充 IR 表示，而不是修改上游 artifact 让其避开该语义。
