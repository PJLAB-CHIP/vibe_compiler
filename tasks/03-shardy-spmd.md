# Wafer Shardy and SPMD Design

状态：2026-07-12重基线；当前合同覆盖Shardy propagation、XLA SPMD helper、显式per-rank specialization和
post-SPMD `wafer.linalg_ext.collective.*` handoff。MPMD、distributed dialect和hybrid rank class延后。
实现状态以`tasks/progress.md`为准。

本文定义Wafer compiler中当前Shardy/SPMD阶段的边界。该阶段负责global tensor的逻辑切分、
sharding propagation、SPMD partition、显式logical rank和logical collective语义；MPMD component和
rank class只保留为后续讨论，不属于近期合同。该阶段不负责physical endpoint
mapping、DTE protocol、SPM buffer、layout materialization 或 runtime launch。

没有用户 `mark_sharding` 或其它可解释 sharding seed 的 program 不是 frontend 错误，但也不应
直接绕过 SPMD 去后段补切分。P2.S1 应在 SPMD 层应用 Wafer 默认 sharding policy：该 policy
消费 SPMD 前已经选择出的 `wafer.execution.mesh`，用 mesh rank count / axes 生成 Shardy / SDY
可解释的 function-input sharding seed；找不到合适切分维度时生成同一 mesh 上的 replicated
seed。单卡默认 topology 配置是 4x4 / 16 tile；这个默认必须通过 topology/execution mesh
materialization 进入 SPMD，不是 SPMD 阶段自己写死的常量。默认 execution mesh 使用所有 available
endpoint；显式少用 tile 只能作为 debug、bring-up、小 workload 或资源隔离 override。后续切图、
local body 和通信算子插入仍交给 Shardy / XLA SPMD partitioner。

本文依赖：

- `tasks/02-frontend-stablehlo-program.md`
- `tasks/04-topology-execution-mesh.md`
- `tasks/13-communication.md`
- `tasks/05-local-compute-normalization.md`
- Shardy MPMD dialect: <https://openxla.org/shardy/mpmd/mpmd_dialect>

Pipeline position:

- Upstream artifact / IR:
  verified StableHLO Wafer program with typed model ABI / symbolic constraints、optional user Shardy/SDY
  or imported sharding annotations、typed `wafer.model.program_graph`、explicit `ParallelizationPolicy`、
  `wafer.target.environment`、`wafer.target.topology` 和已验证的`wafer.execution.mesh`。
- Current stage responsibility:
  先把exporter-native graph、可用Shardy MPMD或typed formation policy规范化成verified
  `wafer.parallel.program`，再对每个component应用默认或用户 sharding policy，运行 Shardy propagation 和
  Wafer-owned SPMD partition，形成 distributed program：显式 component graph、`dp/tp/pp/ep`
  logical coordinate、partition/replica identity、target-independent rank class、logical collective 和
  parameter/state shard relation，并保留一个全局 coherent symbolic-shape variant。
- Output artifact / IR:
  verified distributed StableHLO Wafer program 或等价 program serialization，包含 MPMD components、
  component-local function body、StableHLO logical collective、logical mesh / partition / replica group、
  rank-class coverage、typed component ABI、symbolic variant 和 logical parameter/state shard payload relation。
- Downstream consumer:
  `stablehlo-spmd-to-linalg` 的 tensor collective handoff 与 StableHLO-to-Linalg normalization，
  以及 `stablehlo-spmd-to-group` 后续 logical group pipeline。
- User-level driver / named pipeline:
  production 主线由 `wafer-opt --program-pipeline=stablehlo-to-executable` 或等价 driver 自动执行；
  `stablehlo-spmd`、`stablehlo-spmd-to-linalg` 和 `stablehlo-spmd-to-group` 是内部/分阶段 debug 入口。
- Explicit non-goals:
  不选择 physical endpoint、DTE packet/FSM、SPM/DDR allocation、layout、group fusion、
  instruction family、target ABI 或 runtime launch；不根据 target layout/memory/transport plan 合并 rank
  class；不把 `wafer.spmd.*` attr、pass option、文件名或 side table 当成 distributed program，也不提供
  默认 rank 0。
- Completion gate:
  真实 frontend program 经structured component formation和`stablehlo-spmd`产生 verified distributed program，并被 local compute / group
  gate 继续消费。gate 必须证明每个 component 的 partition/replica coordinate、rank-class coverage、typed
  parameter/state shard、logical collective 和全局 variant 一致；至少覆盖一个 rank-parametric class 和一个
  必须 static split 的 class；source-backed PP=2和heterogeneous expert graph必须从typed model/parallel graph进入
  distributed handoff。默认 rank 0、只运行 Shardy propagation、手写distributed fixture或按function name形成component不算完成。

## 1. 目标和非目标

目标：

- 复用 Shardy / SDY 表达 logical mesh、axis、sharding constraint 和 propagation。
- 优先复用 Shardy MPMD dialect / passes 表达 component graph、program partition 和 cross-component relation；
  只有当前上游表示无法承载且 Wafer 下游必须验证的事实才进入 Wafer-owned distributed handoff。
- 从 frontend program 中导入旧 sharding annotation。
- 生成 verified distributed program、component-local StableHLO body 和 logical collective ops。
- 保留 collective group、partition/replica coordinate、rank class、component edge 和 shard shape 这类后续 endpoint / comm lowering
  需要的语义事实。

非目标：

- 不选择 card/tile 物理坐标。
- 不决定 unavailable endpoint fallback 或 cluster endpoint override。
- 不生成 `wafer.tile.*` communication DTE send/recv、FSM id、stream id、packet id 或 SPM sync slot。
- 不做 group fusion、SPM allocation、DDR allocation 或 target CRT lowering。
- 不把 dynamic actual size、shape guard 选择、physical endpoint 或 target capability 当成 rank identity。

## 2. 输入和输出

输入：

```text
StableHLO module
  + typed model ABI / symbolic shape constraint / parameter and state resources
  + typed model program members/edges + explicit ParallelizationPolicy
  + optional user Shardy/SDY annotations or importable sharding attrs
  + wafer.target.environment selected for legality
  + wafer.execution.mesh selected from valid target environment / topology
  + default Wafer mesh policy when user sharding is absent
```

输出：

```text
distributed StableHLO program
  + MPMD component graph and component-local function bodies
  + partition / replica coordinates and target-independent rank classes
  + logical collective ops
  + logical mesh / rank and replica group metadata
  + typed parameter/state logical shard relation
  + globally coherent symbolic-shape variant
```

SPMD 输出仍然是逻辑 distributed program。它只说明“哪些 logical execution instance 运行哪个 component、
哪些 logical rank 之间需要通信”，不说明“哪两个 physical
tile 通过哪个 DTE resource 通信”。当默认 policy 退化为 1 tile 或 replicated local body 时，
partitioner 可以产生一个 component / replicated rank class；这仍是 SPMD 层的输出，
不是后段 endpoint projection / group 私自补 sharding。

### 2.1 Shardy Propagation / SPMD Program Contract

本节描述 P2 SPMD program 的终态合同；当前实现拆成 P2.S1 Shardy propagation stage 和
P2.S2 Wafer-owned XLA SPMD partition stage。P2.S2 的用户级入口是
`wafer-opt --program-pipeline=stablehlo-spmd` program pipeline；它消费 frontend Wafer program，
在 Wafer compiler 侧执行 sharding propagation 并调用 pinned-XLA SPMD helper。StableHLO program
directory 是该 program 当前的序列化形式，不是与 MLIR 分离的第二条编译路径。终态覆盖两类 program：用户显式标记 sharding 的图，
以及完全没有用户 sharding seed、需要 Wafer 默认 mesh policy 的图。SPMD pipeline 必须消费 P2.F1
产出的 verified frontend program，而不是只 parse 手写 `sdy.mesh` 测试输入。主 pipeline 边界是：

```text
verified StableHLO / optional SDY program
  -> target environment / topology import and verification
  -> valid wafer.execution.mesh selection
  -> if user sharding seed exists:
       sharding import / normalization
     else:
       default Wafer input sharding seed from wafer.execution.mesh
  -> Shardy propagation
  -> Shardy MPMD formation when the source/distributed strategy has multiple program components
  -> XLA SPMD partitioner / equivalent local-body partitioning service per component
  -> distributed StableHLO program assembly
  -> distributed program / rank-class verifier
  -> component-local handoff for local compute / tensor collective normalization
```

P2.S2 输出仍然是 logical distributed program；program directory 只是当前 serialization：

- 一个或多个 Shardy MPMD / StableHLO component，以及 component 间 typed value/state/control edge。
- Shardy / StableHLO / exporter 可解释的 logical mesh、partition/replica group、rank group 和
  `dp/tp/pp/ep` coordinate metadata。
- 每个 component 的 local shard shape、dtype、typed input/output/parameter/state relation。
- 覆盖全部 execution instances 的 target-independent rank class；class 声明 rank-parametric 或
  per-rank-static eligibility，但不携带 target memory/transport plan。
- 整个 distributed invocation 共用的 symbolic-shape variant identity / guard constraint。
- StableHLO logical collective ops，例如 `all_gather`、`all_reduce`、`reduce_scatter`、
  `all_to_all` 和 `collective_permute`；MPMD/MoE ragged exchange若不能由StableHLO原生op表达，则使用typed
  component edge进入`wafer.linalg_ext.collective.segmented_all_to_all` handoff，不能退化成side table。

没有用户 sharding seed 的 P2.F1 program 也进入 SPMD stage，但只允许由默认 policy 根据
`wafer.execution.mesh` 补 Shardy / SDY 可解释的 function-input sharding seed。不能把“没有用户 sharding metadata”诊断成 P2.S1 verifier
失败，也不能补 `wafer.spmd.*`、私有 JSON、名字约定或后段 endpoint fallback。

这些字段必须来自 IR、SDY attr、StableHLO collective metadata 或 importer 已 materialize 的
exporter metadata。
不能靠 pass side table、function name、parameter name 或 dump 文件名恢复。
也不能通过 `wafer.spmd.*` module/function attrs、私有 JSON 或名字约定创造第二套 sharding /
rank 协议。优先使用 Shardy MPMD/SDY；只有 component coverage、typed resource edge 或 rank-class
membership 不能从这些上游事实稳定重算时，才在 distributed-program handoff 中增加 verifier-legal
结构。影响 codegen 的 physical endpoint 事实仍属于 resource projection / launch 边界。

P2.S1 不能以下游当前 lowering、endpoint projection 或 runtime 尚未实现为“不支持”依据。若 Shardy/SPMD
产出的合法语义能由 Wafer 硬件通信、存储或同步能力表达，但当前 Wafer IR 还没有清楚表示，P2.S1
必须先补 op / attr / type / verifier contract，或在任务队列中明确把对应下游 lowering 作为恢复项。
multi replica group、component/rank coverage、shard slicing 和 collective metadata 都属于这类事实：
它们应进入 distributed program 或后续 handoff IR，而不是被静默退回 single group 测试输入，也不能
因为某个 ring / endpoint projection pass 暂时未覆盖就被当成 SPMD 不支持。

P2.S1 只应拒绝两类输入：exporter / Shardy 产物本身非法或自相矛盾；或者目标硬件 / ABI 证据明确
无法表达该语义，且无法由已有硬件能力组合实现。诊断必须定位到当前拥有该事实的层级。

P2.S2 工程 gate 必须把 XLA SPMD partitioner 或等价 local-body partitioning service 纳入主线
完成证明。只运行 Shardy propagation、只 parse `sdy.mesh`，或只把 frontend mark normalize 成
`sdy.sharding`，都不能证明 distributed component/local body 已产生。若当前第三方版本或 API 暂时无法
稳定导出 partitioned StableHLO，应把它记录为 P2.S2 blocker / recovery task；不得用
`wafer.spmd.*` 临时 attrs 冒充 partitioner 输出。

当前 pass / tool 接入边界如下，pass 名只作为实现入口，不能被提升成 IR 层名词：

- frontend Python capture 只负责导出 exporter-native StableHLO program directory。它可以调用
  PyTorch/XLA `mark_sharding` 产生 pre-SPMD sharding seed，但不能执行 Wafer Shardy propagation、
  XLA SPMD partition 或写 distributed program。
- `wafer-compile-stablehlo` 是 frontend / StableHLO program directory verifier。它可以注册
  importer/input dialect 来 parse exporter-native annotations，但不拥有 Shardy propagation、XLA
  SPMD partition、distributed component export 或 parameter/state shard materialization。
- `wafer-propagate-stablehlo-sharding` 只包含 no-user default input seed 和 Shardy propagation。
  它由 `wafer-opt` / `WaferPipelines` 注册，输出经过 sharding propagation stage 处理后的
  StableHLO/SDY IR，不输出 partitioned local body、不写 `forward.parameter_shards.json`、也不插入
  post-SPMD collective。
- P2.S2 必须是 Wafer compiler-owned stage、`wafer-opt --program-pipeline=stablehlo-spmd`
  program pipeline 或等价 library entry：消费经过 Wafer
  sharding propagation stage 的 StableHLO/SDY IR 或 program directory，显式完成 StableHLO/SDY ->
  XLA HLO、XLA SPMD partitioner、partitioned HLO -> StableHLO round trip，再写 component-local /
  replicated StableHLO body、distributed coverage 和 logical parameter/state shard metadata。该入口不能挂在
  `wafer-compile-stablehlo` frontend verifier 下。`P2.S1` / `P2.S2` 只能作为任务索引，不能成为
  program directory 目录名、program 类型名或长期协议字段。
- R2.4 消费 P2.S2 的 partitioned StableHLO collective，并 normalize 到
  `wafer.linalg_ext.collective.*`。它不是 XLA SPMD partitioner，也不能从 P2.S1 的
  propagated global module 直接补 tile-local communication。
- `wafer-lower-stablehlo-to-linalg` 逐 component 消费 post-SPMD local body 或 no-sharding replicated body，
  只做 StableHLO local compute -> Linalg/Tensor/Arith/Math；它不做 sharding propagation、SPMD
  partition、endpoint mapping、group、SPM/DDR 或 communication materialization。

实现边界：

- frontend Python test generator 的职责只到 PyTorch/XLA StableHLO export：未标记图导出 reference
  program directory；用户 sharding 分支用 `mark_sharding` 标记同一个 4096 matmul 图的 `x`、`weight`、`bias`，
  导出带 `mhlo.sharding` 的 pre-SPMD PyTorch/XLA StableHLO program directory。
- `wafer-compile-stablehlo` 只保留 `--verify-frontend-program` 和
  `--verify-stablehlo-program`。旧 `--propagate-stablehlo-sharding` 和
  `--partition-stablehlo-program` 入口已删除，因为它们把 compiler SPMD ownership 错挂到了
  frontend verifier tool。
- `wafer-propagate-stablehlo-sharding` 是 Wafer compiler 侧 Shardy propagation named pipeline。它消费
  StableHLO/SDY IR，补 no-user default input seed，并调用 Shardy propagation；它不产生 partitioned
  local body。
- P2.S2 当前由 `wafer-opt --program-pipeline=stablehlo-spmd` program pipeline 加
  pinned-XLA helper/service 实现。`wafer-opt` 先验证输入 program directory，再在 Wafer compiler 侧执行
  default input seed + Shardy propagation，随后把 propagated program directory 交给 helper。helper
  显式执行 StableHLO/SDY -> XLA HLO、`ShardyXLA` / `SpmdPrepare` / `SpmdPartitioner` /
  `HloVerifier`、partitioned HLO -> StableHLO round trip，并写回 partitioned StableHLO program
  directory。helper 会把旧 `mhlo.sharding` / `stablehlo.sharding` 中顺序枚举的
  `{devices=[...]0,1,...}` canonicalize 成 XLA 当前可稳定消费的 iota sharding 形式；这是
  frontend sharding attr 兼容处理，不是 Wafer 私有 sharding 协议。
  `wafer-opt` staging 必须保留未被 propagation 改写的 program members，只重写 propagated IR；
  helper / writer 在产生 distributed program serialization 时负责同步更新 component-local function
  signature、logical parameter/state shard payload、partition/replica coverage 和 rank-class metadata。
  helper 路径由 build-time
  `WAFER_XLA_SPMD_PARTITIONER_HELPER` 配置进入 `wafer-opt`，不是用户级 pipeline flag。
- 2026-06-01 直接 CMake link 评估结论：当前 build 虽启用 `WAFER_ENABLE_SPMD_PARTITIONER_DEPS`，
  但 CMake target graph 只包含 Wafer / StableHLO / Shardy，没有 XLA `spmd_partitioner`、HLO
  service、TSL、Abseil 或 generated XLA proto targets。为避免把任务拖进 XLA CMake shim，P2.S2
  采用 `tools/build_xla_spmd_partitioner_helper.py` 生成 pinned `third_party/xla` Bazel overlay 并构建
  helper；helper 只是当前工程接入方式，不是新的 IR 层、program directory 名称或长期协议对象。
- Wafer-owned Shardy / SPMD 实现源码归属 `lib/Wafer/Transforms/SPMD/`。只依赖 MLIR / StableHLO /
  Shardy CMake target 的 pass 应编进 `WaferTransforms`；必须直接使用 XLA HLO service / SPMD
  partitioner / generated proto / TSL 的入口可以从同一 Wafer 源码树映射进 pinned XLA Bazel overlay
  编译。当前 `lib/Wafer/Transforms/SPMD/XlaSpmdPartitionerMain.cpp` 就是这种 Wafer-owned stage
  adapter，`tools/build_xla_spmd_partitioner_helper.py` 只是构建桥。
- 旧 Python post-SPMD helper 和相关 tests 已删除。P2.S2 的 partitioned StableHLO 主链产物只能由
  Wafer-owned compiler stage 保存；不允许用 Python helper、手写 sidecar 或测试输入冒充这个缺口。
- no-user-sharding 分支当前只在文本 StableHLO/SDY program 中用
  `--wafer-apply-default-spmd-sharding` / `wafer-propagate-stablehlo-sharding` 补 function-input seed；
  P2.S2 再消费该 stage 输出 program。默认 policy 的 rank count / axes 必须来自
  `wafer.execution.mesh`；局部 pass / named pipeline 测试输入必须显式携带
  `wafer.target.environment`、`wafer.target.topology` 和 `wafer.execution.mesh`，不能用 pass option 绕过
  environment / execution mesh。默认
  policy 只标记输入/参数，不给中间 op 或 function
  result 造约束。
- 对 PyTorch/XLA 等 frontend 以 `stablehlo.custom_call @Sharding` / `mhlo.custom_call @Sharding`
  和旧 `mhlo.sharding` / `stablehlo.sharding` 字符串暴露的 sharding seed，Wafer 默认 seed stage
  只做受限导入：支持 replicated，或单个 tensor 维度的切分因子等于当前 execution mesh rank
  count。该导入把 frontend seed 变成 Shardy/SDY 可解释的 tensor sharding；不支持的多维或部分复制
  形态必须在 SPMD 层给出 diagnostic，不能落到后端靠名字或 side table 修复。
- P2.S2 输出的 distributed program serialization 必须由 Wafer-owned compiler stage 保存。
  `functions/forward.mlir` 是当前 single-component serialization 的 local body；多 component program
  必须由 Shardy MPMD/program structure 或等价 structured program member 显式索引，不能从 function 名猜 stage。
  每个 component metadata 的 typed ABI 必须匹配 local function boundary；
  `functions/forward.parameter_shards.json` 当前承接 parameter/state local resource 到
  `parameter_shards/<payload-key>/rank_XXXXX.npy` logical shard payload 的 explicit binding，供后续
  storage/package materialization 消费。binding 的 offsets、sizes、strides、partition/replica coordinate、
  component 和 rank-class coverage 必须来自 XLA/Shardy/exporter-native shard facts，不由 Wafer 从
  `partition_spec`、strategy 名、文件名或 parameter 名手算。
- 当前单 component serialization 使用 parameter-shard schema v3；每个 parameter binding 必须显式声明
  `distribution = replicated | partitioned`。`partitioned` 的 rank slices 必须在 global tensor 上无重叠且
  精确覆盖一次；`replicated` 的每个 rank 必须绑定完整 global tensor、使用完整 replica-id domain，且
  NPY payload byte-identical。当前 schema 不表达 partial replication，helper 必须在导出时结构化失败，
  不能把重复 offsets 当成普通 partition 或靠 verifier 猜测。
- 当前 helper 对参数 payload 支持 row-major raw tensor 和 NumPy `.npy` v1/v2 输入，按 XLA
  `HloSharding::TileOffsetForDevice` / `TileLimitForDevice` 为每个 logical coordinate materialize local
  payload，并按 PyTorch/XLA program directory 权重格式写成 NPY stream；component-local function signature 从 post-SPMD
  StableHLO `func.func @main` 的 ranked tensor 边界回写到 `forward.meta`。NPY stream 是 frontend /
  partitioned program 的 tensor payload 容器，不是 Wafer runtime/package ABI。

#### 2.1.1 真实图和 sharding 覆盖矩阵

P2.S1 的 sharding branch 主 gate 继续使用 P2.F1 的真实 4096 matmul 图，不换成小 toy model：

```text
x: tensor<4096x4096xf32>
weight: tensor<4096x4096xf32>
bias: tensor<4096xf32>

y = tanh(x @ weight + bias) + residual
```

Sharding 必须从 framework frontend 提供的 mark 接口进入 program，例如 PyTorch/XLA 的
`mark_sharding` 或 export 可追踪的等价前端 op。测试和实现不得手写 `sdy.sharding` 作为主链路
输入，也不得用 Wafer 私有 JSON / sidecar 描述 sharding。每种策略都应导出同一格式的
PyTorch/XLA StableHLO program directory，并由 Shardy / SPMD pipeline 消费：

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
| row parallel / contracting-dim sharding | `x: (None, tp)`；`weight: (tp, None)` | `K` 维 partial sum；distributed component 必须保留 reduction collective 语义 |
| 2D output sharding | mesh `dp x tp`；`x: (dp, None)`；`weight: (None, tp)`；`bias: (tp,)` | 输出同时按 `B` / `N` 维切分；保留 2D logical mesh 和 rank group |
| 2D contracting + output sharding | mesh `dp x tp x mp`；`x: (dp, mp)`；`weight: (mp, tp)`；`bias: (tp,)` | 输出按 `B` / `N` 分布，同时 `K` 维沿 `mp` group reduction；`mp` 是 logical reduction axis，不是 physical pipeline/MoE axis |
| partial replication | 某些 tensor 在一个 mesh axis 上 sharded、在另一 axis 上 replicated，例如 `bias` 在 `dp` 上 replicated、在 `tp` 上 sharded | verifier 必须能解释 subgroup replication，不把 replicated axis 丢成默认全复制 |

这些 case 是 P2.S1 的覆盖矩阵，不是新的长期协议对象。长期合同仍是 IR 中的 logical mesh、sharding
annotation、rank group、local shard relation 和 collective metadata。pipeline parallel、MoE /
expert sharding、真实 sequence parallel 和非整除 uneven slicing 不进入第一批 P2.S1 主 gate；它们
需要对应图结构、routing/stage 语义或单独 legality/verifier 覆盖，不能通过在当前 matmul case 上
硬塞名字来冒充支持。

同一个 4096 matmul 还应保留 no-user-sharding 覆盖：不调用 `mark_sharding` 时，导出的 program
没有用户 sharding seed，P2.S1 默认 policy 应补 function-input sharding seed，并继续进入
Shardy propagation；P2.S2 再进入 XLA SPMD partitioner。这个 case 不计入上表的用户 sharding
策略数量，也不能被写成 Wafer 私有 sharding；它是 SPMD 层的默认 seed policy。

#### 2.1.2 执行方法

P2.S1/P2.S2 的 program 生成入口应保持使用同一 4096 matmul source graph，但主线实现路线必须是：

```text
PyTorch module
  -> optional torch_xla.distributed.spmd.mark_sharding / torch.ops.xla.dynamo_mark_sharding
  -> PyTorch/XLA StableHLO / SDY program       # frontend Python stops here
  -> target environment / topology / valid execution mesh materialization
  -> typed model program graph + ParallelizationPolicy -> verified wafer.parallel.program
  -> if no user sharding seed exists, apply Wafer default function-input seed from execution mesh
  -> per-component Wafer Shardy propagation
  -> per-component Wafer-owned XLA SPMD partition compiler stage
  -> distributed StableHLO program assembly and rank-class verification
```

frontend export 路径必须运行在 P2.F1 同一套 importer Python / source-built `torch_xla` 环境里。`torch_xla`
来自 `third_party/pytorch-xla` 源码构建/安装，并复用本仓库固定的 LLVM/MLIR、StableHLO、Shardy
和 XLA 版本；不得改用 prebuilt `torch_xla` wheel 或新下载另一套 XLA/LLVM。

partition/replica identity 必须按第 3 节进入 distributed program。rank-parametric component 保留标准
`stablehlo.partition_id` / `replica_id` 或等价显式 SSA/ABI；per-rank-static component 由 compiler
specialization record 固定 canonical coordinate。它不能通过 `wafer.spmd.global_rank`、
`wafer.spmd.local_rank`、hidden pass option 或 API 默认值写回/恢复。rank group 应优先来自 partitioned
StableHLO collective 的 `replica_groups` / `source_target_pairs` / Shardy metadata；无法重建的 component /
coverage relation进入 structured distributed-program handoff，而不是临时 attr。

每个用户 sharding strategy 的前端模型仍然是同一个 4096 matmul 最小验证 module。forward 中只在
framework 边界做三类 mark：

```text
x      -> input_spec
weight -> weight_spec
bias   -> bias_spec
```

当前已知错误路线是：先跑 frontend mark 最小验证，再在 `forward.mlir` 中手写 `sdy.mesh` /
`sdy.sharding` / `wafer.spmd.*`，然后让后续 verifier 消费这些 Wafer 私有 attrs。这个做法只是在
program 中补了一份临时描述，既没有证明 XLA SPMD partitioner 产生了 local body，也会把后续实现
引向错误的协议源。P2.S1/P2.S2 恢复时必须删除这条完成口径。

允许保留的临时测试只有两类：

- frontend mark 最小验证：证明 PyTorch/XLA `mark_sharding` 可被当前 capture 路径观察或追踪。
- SDY / StableHLO dialect unit 测试输入：证明工具链能 parse / verify / run Shardy propagation。

这些测试都不能标记 P2.S2 完成。P2.S2 完成证明必须消费真实 P2.F1 program 和 Wafer Shardy
propagation 输出：用户策略消费真实 mark 后的 program；no-user-sharding 策略消费同图未标记
program 并由默认 policy 生成 function-input seed。两类策略都必须由 Wafer-owned SPMD partition
stage 得到 component-local partitioned/replicated StableHLO body，并组装成 verified distributed program。

当前验证 gate：

- `test/Pipelines/stablehlo-sharding-propagation.mlir` 和
  `test/Tools/wafer-compile-stablehlo-sharding-propagation.test` 覆盖当前合法边界：pre-SPMD program
  directory 先通过 frontend program directory verifier；Shardy propagation 由 `wafer-opt` named
  pipeline 对 `functions/forward.mlir` 执行。该测试同时确认 `wafer-compile-stablehlo` 不再暴露
  propagation / partition flags。
- `test/Tools/wafer-pytorch-xla-capture-sharded-program.test` 覆盖六种用户策略的真实
  `mark_sharding` -> pre-partition StableHLO program directory，并通过 `wafer-compile-stablehlo
  --verify-stablehlo-program` 校验 program metadata / payload。该测试不执行 Shardy propagation
  或 XLA SPMD partition；它证明 frontend export / verifier 边界，不证明 P2.S2。
- `test/Tools/wafer-opt-spmd-partition.test` 覆盖 P2.S2 真实 gate：从 PyTorch/XLA
  `mark_sharding` program directory 进入 `wafer-opt --program-pipeline=stablehlo-spmd`，由 Wafer
  compiler 的 `wafer-opt` program pipeline 执行 program directory verify + Shardy propagation，再调用 pinned-XLA helper 产出
  partitioned StableHLO program directory。该 gate 用同一个 matmul 图的 `--size 32` 形态覆盖 data、
  column、row、2d-output、2d-contracting-output 和 partial-replication 六种 strategy；这是为了让本地
  helper / lit gate 可重放，不改变 P2.F1 4096 export 主图的语义形态。column case 当前额外检查
  local signature、StableHLO collective、`forward.parameter_shards.json` 和 logical-shard NPY stream
  payload；它还不能单独证明 MPMD、rank-class 或 coherent shape-variant 合同。旧
  `test/Tools/wafer-compile-stablehlo-spmd-partition.test` 已删除，因为它把 P2.S2 主入口
  错误地挂在 frontend verifier tool 下。
- `test/Tools/wafer-opt-hf-megatron-transformer-block.test` 覆盖更接近 LLM 的真实 frontend gate：
  从 HuggingFace Llama config snapshot 构造一个静态 decoder block，使用 PyTorch/XLA
  `mark_sharding` 在 16-rank 单卡 mesh 上标记 Megatron-style tensor parallel 权重和 activation
  tensor-parallel seed，再进入 `wafer-opt --program-pipeline=stablehlo-spmd-to-group`。该 gate
  证明当前 compiler 能消费 attention/RMSNorm/RoPE/SwiGLU 主干、captured constants、parameter
  shards 和 post-SPMD collectives 到 `wafer.group` 边界；Megatron row-parallel/contracting 形态
  必须在 post-SPMD IR 中保留 `all_reduce`，不能退化成只靠 `all_gather` 拼 full tensor。下游
  下游 HF transformer gate 继续消费同一类 HF transformer program，当前覆盖到 group ->
  memory-planned instruction IR；target LLVM call emission 已有 hand-written instr/group gate，HF program-chain
  target LLVM integration、target CRT symbol closure、package/no-card runtime required-symbol gate、真实 board execution、数值 correctness
  和 selected-candidate closed-loop path 仍由后续 gate 覆盖。

#### 2.1.3 默认 no-user-sharding policy

当一个 graph 中已经存在任何用户 sharding seed 时，默认 policy 必须完全跳过该 graph。这里的
seed 包括 function argument/result 上的 `sdy.sharding`、中间 value 的 `sdy.sharding` /
`sdy.sharding_constraint` / `sdy.reshard`、manual computation sharding，或 frontend mark 导出的
等价 SDY / StableHLO sharding 表示。用户可能只标记少数关键 op / value，再依赖 Shardy
propagation 推到整图；默认 policy 不能覆盖、补齐或重解释这些用户 seed。

当 graph 完全没有用户 sharding seed 时，P2.S1 默认 policy 从 `wafer.execution.mesh` 创建
Shardy/SDY 可解释的 logical mesh：

```text
@wafer_default_mesh = <axes and sizes from wafer.execution.mesh>
```

mesh rank count 和 axes 来自 `wafer.execution.mesh`。单卡默认 topology 配置是 4x4 / 16 tile，
默认选择 16-rank mesh；调试、bring-up、小 workload 和资源隔离可以显式选择较小 mesh。这个 mesh 是
SPMD logical mesh 的来源，不是后段 rank->tile map；physical endpoint、unavailable tile 例外和
card/tile adjacency 仍由 `wafer.target.environment` / `wafer.target.topology` /
`wafer.execution.mesh` 的 derived view 派生，
不写入 StableHLO/SDY module。

默认 policy 主要标记 function inputs / parameters 的 sharding seed，不主动给中间 op 或 function
results 下约束。Shardy propagation 负责把 seed 推到内部 value 和结果，XLA SPMD partitioner
负责产生 local body 和必要 collective / permute。第一版可采用保守 deterministic heuristic：

- 对每个 ranked tensor function input，选择第一个静态且能被 execution mesh rank count 整除的维度绑定
  mesh axis。多轴 mesh 可先按 deterministic axis order 选择；更复杂的 dim-to-axis cost model 属于
  后续 SPMD / endpoint co-design，不改变 IR 合同。
- 如果没有这样的维度，或 rank count 为 1，该 input 在对应 mesh axis 上 replicated。
- 不依赖 input / parameter 名字；typed input/output、immutable parameter、persistent state 和 alias/mutation
  必须来自 frontend verified model ABI。persistent state 的 logical shard 与 immutable parameter shard
  分别验证，不能把 state mutation 降级成普通 input/output。
- 默认 policy 只生成 SDY / StableHLO 可解释的 sharding seed，不生成 `wafer.spmd.*`、resource-projection op、
  package metadata 或其它后段协议。

这个 policy 的目标是让未显式标记的图默认利用已选择的 valid execution mesh，同时仍把通信插入、
component-local body 和 rank-class 生成留在 distributed program formation 内。找不到合适切分维度时的 replicated seed 不是
放弃 SPMD；它是明确的 replicated sharding seed，后续仍可通过同一 SPMD pipeline 产出等价 local
component 和 replicated rank class。

distributed program verifier 的责任必须是检查 StableHLO / Shardy MPMD / SDY /
exporter-native facts，而不是检查 `wafer.spmd.*`：

- module 中的 Shardy / SDY / StableHLO sharding metadata 必须可由注册 dialect 解释。
- component function boundary 的 typed role、tensor shape/dtype、symbolic constraint、alias/mutation 和
  exporter metadata 必须一致。
- local shard relation 必须能从 partitioned StableHLO shape、SDY metadata 或 exporter-native metadata
  解释；不能依赖 parameter/function 名字。
- 对 immutable parameter 和 persistent state，post-SPMD distributed program 必须显式 materialize
  `component + partition/replica coordinate -> logical resource shard` 绑定：每个 shard entry 必须有稳定
  resource identity、offsets、sizes、strides、partition/replica coordinate、local shape 和 initializer/import
  relation；文件只作 payload location。offsets/sizes/strides 必须来自 PyTorch/XLA runtime / XLA sharding spec 暴露的
  shard indices，不能由 Wafer 根据 `partition_spec`、parameter 名、strategy 名或文件名重新推断。
- collective metadata 必须来自 StableHLO op，例如 `replica_groups`、`source_target_pairs`、
  `all_gather_dim`、`scatter_dimension`、`split_dimension` / `concat_dimension` 和 reduction body。
- program 中不得出现 physical tile、DTE packet、SPM offset 或 runtime handle 这类下游 lowering
  metadata。

Shardy propagation 的局部 gate 可以通过 Wafer named MLIR pipeline 直接消费 `functions/forward.mlir`：

```text
wafer-opt --pass-pipeline='builtin.module(wafer-propagate-stablehlo-sharding)' <strategy>/functions/forward.mlir
```

standalone `shardy-sdy-opt` 可以作为第三方 pipeline 对照，但不是 Wafer 用户级主线入口。
这个局部 gate 只证明 SDY dialect / propagation pipeline 能读取真实 program 中的 sharding facts；它必须和
Wafer compiler-owned XLA SPMD partitioner / partitioned StableHLO export gate 配套使用，不能单独作为
P2.S2 完成证明。
row / contracting 类 case 需要保留 reduction collective op；如果后续 tensor collective
normalization、endpoint projection 或 ring/resource path 还没有完整消费这些事实，应补对应下游任务，不回头把
该策略从 P2 SPMD program gate 中删掉。

### 2.2 Structured Parallel Program Formation

Shardy MPMD若在pinned版本能完整表达member、port、edge和participant relation，adapter应直接规范化它；当前缺少
完整对象时，V1使用同一Wafer dialect中的transformation-local `wafer.parallel.*` IR，而不是JSON、名字约定或
手写distributed fixture。它位于verified model program与SPMD partition之间，失败时丢弃完整clone，不作为package
或runtime artifact。

| object | 必须字段/关系 | 不拥有 |
| --- | --- | --- |
| `wafer.parallel.program` | model-interface/program-graph ref、execution-mesh ref、typed `ParallelizationPolicy`、ordered component/edge symbol table | physical endpoint、runtime schedule、distributed digest |
| `wafer.parallel.component` | nonzero program-scoped `ComponentId`、source `ModelProgramMemberId` set或compiler-formed source region refs、ordered typed local ABI ports、`LogicalParticipantPredicate` | function/expert/stage name identity、flat default rank |
| `wafer.parallel.edge` | source/destination component+port、tensor/state/control/segmented enum、semantic type/DimId、必要ResourceId/alias-update、participant relation | transport route、buffer、microbatch queue |
| `LogicalParticipantPredicate` | typed `dp/tp/pp/ep` axis-role enum上的checked coordinate equality/range/set和replica relation | axis-name string、physical tile、runtime callback |

`ParallelizationPolicy`是versioned closed typed record，至少区分`preserve_exporter_members`、
`pipeline_partition`、`expert_partition`及其显式组合，并记录source member refs、legal cut/op-interface criteria、stage或
expert domain、participant predicates和resource/state placement constraints。policy可以驱动compiler从一个member
形成多个components，但只能按registered op semantics、SSA use-def、typed resource effect和显式cut constraints；任意
Python callback、function/op name regex和opaque partition payload非法。policy只属于CompilationRequest/build
provenance和formation transaction；distributed IR/semantic identity只保存materialized components、source mapping、
participant predicates、typed edges和其它下游必须验证的结果。不同policy得到逐字段等价的semantic graph时identity
相同；若某个policy choice改变可观察语义，必须materialize成明确output enum/relation，不能复制整份search/cut policy。

formation在一个transaction中构造完整graph，再按explicit ordered component records分配`ComponentId = 1..N`。
exporter member ID与ComponentId不是同一scope：一个member可被合法切成多个components，一个component也可在policy
允许时组合多个members；mapping必须显式。distributed assembly原样保留ComponentId。rename/permutation只要typed
member/edge semantic order不变就不改变formation；改变cut、participant或edge会改变distributed identity。

PP formation显式产生stage components和stage tensor/state/control edges；microbatch count/issue schedule仍由后续
entry iteration domain与completion DAG拥有。EP formation显式产生router、dispatch、one-or-more expert、combine
components；segmented edges携带count/payload ports、expert domain、finite per-peer/destination capacity和count-before-data
relation，actual counts/displacements保留SSA。异构expert必须形成不同component或local ABI class，不能用expert函数名
或buffer名恢复。

formation verifier要求component/port/edge refs闭合，所有model members和required outputs有exact coverage，普通call
不能跨component逃逸typed edge，persistent `ResourceId`/state-group/alias-update relation不丢失，participant predicates
在selected execution mesh上nonempty且无非法overlap。之后每个component独立运行Shardy propagation/SPMD partition，
最终assembly验证cross-component edge两端的local shard、collective/segmented语义和global variant一致。任一步失败不
保留partial component或per-rank output。

### 2.3 Formation and SPMD Operational Limits

parallel/distributed formation接收validated nonidentity `DistributedFormationLimits`，至少限制model members/edges/
ports、parallel components/edges/predicates、mesh coordinates、distributed instances/classes/shards/variants、high-fanout
edge refs、checked instance products、simultaneous clones/workers、peak IR/analysis bytes和diagnostics。formation先从
typed model graph/mesh做checked count/product preflight，再在一个clone中materialize；不能先展开全部coordinates再检查。
充分limits不改变ComponentId、DistributedProgramSemanticId、ExecutionInstanceId、class/shard或edge order；超限无
partial component/distributed IR。

external Shardy/XLA helper还接收`SpmdExecutionLimits`：single/total input/output bytes、output functions/ops/regions/
blocks/values/types/attrs、shard records/instances、workers、wall/CPU/RSS/address-space/process count、stdout/stderr和
kill/reap deadline。helper在sandbox process group中运行，bounded sink防output bomb；timeout/cancel/crash/limit失败
kill/reap并删除partial output。parent先按output bytes限制读取，再重跑structural counters和typed verifier；不得因helper
成功退出跳过。limits只做operational admission，不能改变partition algorithm/options、选择rank prefix或重写sharding。

## 3. Distributed Program Contract

`distributed program` 是 SPMD/MPMD 之后、physical placement 之前的长期逻辑 artifact。它可以继续使用
StableHLO、Shardy MPMD/SDY、func 和 structured program metadata 表达；这个名称不要求新增一套私有
dialect。只有上游 IR 无法稳定表达而下游 legality 必须消费的 component/resource/rank-class relation，
才允许进入 Wafer-owned handoff，并必须有 verifier、parser/printer 和 lowering owner。

### 3.1 Execution Identity And Coordinates

一个 execution instance 的 canonical identity 是：

```text
(distributed_program, component, partition_coordinate, replica_coordinate)
```

其中`ComponentId`是nonzero `uint32`、scoped to one distributed program：由`wafer.parallel.program` formation按
显式ordered component records从1分配，distributed assembly必须原样保留，不从symbol/function name或hash-map iteration得到。program identity
preimage只编码local instance key `(ComponentId, partition_coordinate, replica_coordinate)`，避免把尚未计算的
distributed-program digest自包含；完整program verifier/identity成功后，public `ExecutionInstanceId`才表示为
`(DistributedProgramSemanticId, local instance key)`。package/runtime保留该typed composite，不能只保存flat rank。

- `partition_coordinate` 是 `wafer.execution.mesh` 上参与 tensor partition 的多轴坐标。
- `replica_coordinate` 区分相同 partition program 的复制实例；不能因为当前 XLA helper 使用
  `replica_count = 1` 就从长期合同删除。
- flat logical rank 只是在给定 mesh axis order 下可派生的 ordinal，不是独立事实源，也不能同时代替
  partition、replica、PP stage 或 expert identity。
- `dp/tp/pp/ep` 是 logical axis role。DP/TP/EP 可参与 tensor sharding/replication group；PP 通常通过
  MPMD component/stage assignment 表达。axis role 必须来自 Shardy/SDY/MPMD 或 verified program policy，
  不能从 axis 名字字符串猜测。
- `stablehlo.partition_id` / `replica_id` 等标准语义在 rank-parametric component 中必须保留为显式
  SSA/ABI identity。只有 per-rank-static specialization 记录覆盖唯一 canonical coordinate 时才能常量折叠。
  normalization、pass option、API 默认参数或缺省值都不能把它们替换成 rank 0。

### 3.2 MPMD Components

TP/DP 的同构 local body 可以属于一个 component；PP stage、异构 expert program、host/device control
component 或其它不同 local program 必须形成显式 MPMD component。优先复用 Shardy MPMD dialect、
component formation/optimization pass 和它们的 symbol/use-def relation。每个 component 至少具有：

- stable component identity、typed local ABI 和所覆盖的 execution-instance predicate。
- component 间 typed tensor/state/control edge；persistent state edge 保留 resource identity 和 mutation。
- participant rank/replica groups，以及 collective 所引用的 component-local group。
- symbolic shape constraint 的 local projection；它必须能追溯到 distributed invocation 的全局 variant。

component graph 不保存 micro-batch runtime queue、physical endpoint、DTE route、SPM/DDR offset 或 launch
handle。PP micro-batch 的数学/dataflow relation 属于 component graph；实际 issue schedule、buffer lifetime
和 completion 属于后续 executable planning。

### 3.3 Hybrid Rank Classes

distributed rank class 是 target-independent equivalence class。两个 execution instances 只有在以下事实
全部相同时才能先归入同一 class：

- 同一 component、typed local ABI、local StableHLO/structured body 和 symbolic constraint projection。
- 相同 static local shape/dtype、parameter/state resource role、alias/mutation relation。
- 相同 collective/control role；rank 差异只通过显式 partition/replica identity 和 typed resource binding
  进入，不通过不同 op body 或隐式名字进入。

class 可以声明两种合法 materialization mode：

- rank-parametric：共享 local artifact，partition/replica identity 保留为 SSA/ABI，rank-specific parameter /
  state shard在后续 typed binding 中提供。
- per-rank-static：compiler 为 canonical coordinate 显式 specialization，并保存 specialization/coverage
  record；IR 中不再保留动态 rank，但也没有默认 rank。

SPMD 层形成的 class 只是共享的必要条件。后续 target environment、layout、SPM/DDR、transport 或
entrypoint ABI 不同可以继续拆 class，不能把 distributed program 已判定不同的 class 重新合并。首版实现
可以保守地让每个 coordinate 各成一类；长期 package/cache contract 不因此退化成固定 per-rank 模式。

典型行为：规则 DP replica 和等形 TP rank 可以共享 rank-parametric class；PP stage 必须是不同
component；uneven TP shard、异构 expert shape 或不同 control/collective role 必须 static split。MoE token
count/routing 是动态数据语义，不按每次 token count 创建 rank class。

### 3.4 Globally Coherent Variants

symbolic dynamic shape、state capacity 或 target-specific static specialization 不属于 rank identity。
distributed program 可以包含多个 guardable variant proposal，但一次 distributed invocation 必须先选择
一个全局 coherent variant，再让全部 component/rank class 派生 local static shape。禁止各 rank 独立选择
互不兼容的 shape variant，也禁止 SPMD stage 根据 actual runtime size 直接生成低层 dynamic descriptor。

distributed variant 至少绑定 program semantic digest、symbolic guard、component graph、mesh shape 和全部
rank-class coverage。target capability/topology variant 由 `tasks/04-topology-execution-mesh.md` 继续选择；
runtime 只能从 compiler 已验证的完整 distributed variant 中选择，不能重新分片或补 component。

### 3.5 V1 Typed Distributed Handoff

当前pinned Shardy/SDY能表达mesh/sharding/propagation，但没有可被Wafer后端长期直接消费的
MPMD component、canonical execution instance、resource shard和hybrid prerequisite class完整对象。
因此V1在SPMD partition后引入一个最小Wafer-owned typed handoff；它只补上游无法稳定表达
而下游legality必须消费的relation，不复制StableHLO/SDY body或sharding plan。若后续pinned
Shardy提供等价typed object，importer可将其规范化到同一verifier contract，但不能让两套
component/rank identity并存。

V1 handoff对象为：

| object | 必须字段/关系 | 不拥有 |
| --- | --- | --- |
| `wafer.distributed.program` | stable program symbol、verified model-interface ref、execution-mesh ref、component/instance/class/resource-shard/variant symbol table | local op body副本、physical placement、runtime handle |
| `wafer.distributed.component` | program-local nonzero `uint32 ComponentId`、parallel source mapping、diagnostic symbol、one or more local `func.func` refs、typed local ABI ref、participant predicate、typed component-edge refs | function/symbol-name-derived identity、endpoint、microbatch queue |
| `wafer.distributed.instance` | local key=`ComponentId + typed partition_coordinate + replica_coordinate`、execution-mesh ref；完整program digest验证后与其组成public `ExecutionInstanceId` | flat rank作为第二事实源、self-referential program digest、default rank |
| `wafer.distributed.class` | stable prerequisite-class symbol、唯一component ref、`rank_parametric`/`per_rank_static` enum、covered instance refs、local function/specialization refs、typed ABI/resource/collective-role equivalence proof inputs | final target-dependent `RankClassId`、code cache merge |
| `wafer.distributed.resource_shard` | frontend `ResourceId` ref、instance/class coverage、logical offsets/sizes/strides、dtype/shape、content/shard identity、alias/update relation | payload path作身份、DDR arena/offset、resident handle |
| `wafer.distributed.edge` | source/destination component+port、tensor/state/control enum、type/shape constraint、resource identity/update relation | physical transport、stage issue schedule |
| `wafer.distributed.variant` | typed global shape-guard proposal、program semantic source ref、complete component/instance/class coverage | target capability predicate、projection、rank-local guard |

`wafer.distributed.program`是symbol table container；component-local StableHLO/Linalg body仍由被引用的
`func.func`唯一持有，不嵌回handoff op。`partition_coordinate`的维度与`wafer.execution.mesh`
axis/role一一对应，`replica_coordinate`始终显式存在，即使当前replica count为1。flat
logical rank只是在给定mesh order下从coordinate派生的debug ordinal，不写入另一个identity field。

program-directory `forward.parameter_shards.json`和`rank_XXXXX.npy`只是当前loader/payload locator。
distributed assembly必须在进入group/candidate之前把其中的typed shard relation、content identity和
coverage验证并materialize为上述IR；下游不再parse JSON、匹配`rank_XXXXX`或默认rank 0。
payload locator可留在program container的artifact table，但locator不进入resource/instance identity。

handoff verifier必须检查：所有symbol/ref闭合；every required canonical instance恰好覆盖一次；
class只覆盖同一component并满足本节3.3等价条件；rank-parametric function保留partition/
replica SSA/ABI；per-rank-static specialization只覆盖一个canonical coordinate；resource shard与frontend
resource/type/payload完全一致；variant coverage不留hole且所有component引用同一global guard。
任一physical endpoint、SPM/DDR offset、transport resource、planner trace或runtime field出现都必须失败。

## 4. Logical Mesh Contract

Logical mesh 是 model-level parallelism 的语义对象：

- mesh axes 具有显式 `dp/tp/pp/ep` 等 logical role；axis name 只用于可读性，不能驱动 lowering。
- partition/rank group、replica group 和 component coverage 由 Shardy MPMD / sharding propagation /
  partition 产生。
- logical execution identity 来自 `wafer.execution.mesh` rank domain + distributed component，不是 physical
  tile id。
- logical mesh 可以大于单卡或跨卡；physical feasibility 由 SPMD 前的 target environment / topology /
  execution mesh selection 判断。

如果某个 parallel strategy 只能用 physical tile name 表达，说明它不属于 Shardy / SPMD 阶段。

## 5. Collective Contract

V0 关注以下 StableHLO collective 语义：

| collective | SPMD 语义 | 下游 owner |
| --- | --- | --- |
| `collective_permute` | logical point-to-point value movement | `wafer.linalg_ext.collective.*` handoff + endpoint projection + direct `wafer.instr.dte_*` materialization |
| `all_gather` | shard concat / replication | `wafer.linalg_ext.collective.*` handoff |
| `reduce_scatter` | reduce + shard distribution | `wafer.linalg_ext.collective.*` handoff |
| `all_reduce` | all-rank reduction | `wafer.linalg_ext.collective.*` handoff |
| `all_to_all` | split / exchange / concatenate across logical ranks | `wafer.linalg_ext.collective.*` handoff；当前 V0 direct `wafer.instr.dte_*` materialization，后续可扩 `wafer.tile.*` schedule |
| segmented peer exchange / all-to-all-v | dynamic per-peer counts/displacements under static capacity，适用于MoE token dispatch/combine | `wafer.linalg_ext.collective.segmented_all_to_all` handoff；communication materialization建立count/data phase IR，physical transport只绑定其resources |

`all_to_all` 的高性能算法可以后于 ring all-gather / all-reduce 实现，但 SPMD program stage 不应因为当前
communication lowering 只有 direct p2p correctness path 而丢失或拒绝它的 logical collective 语义。
distributed component 仍要保留 split / exchange / concat 的 rank group、slice 和 dtype 事实，后续
communication lowering 再选择 p2p schedule 或更高性能算法。

segmented peer exchange的peer group、payload dtype、capacity bound和count/data dependence属于distributed
语义；actual token counts/displacements属于invocation SSA data，不进入rank class或variant identity。若上游
MPMD表示只能给出typed expert/component edges，Wafer handoff必须从这些edges构造同一segmented semantic，
不能从expert函数名或router buffer名恢复。

Collective handoff 分三步：

```text
StableHLO logical collective
  -> `wafer.linalg_ext.collective.*` op
  -> tiled tensor collective inside scheduled group / tile_region materialization
  -> `wafer.tile.*` buffer-level collective, or direct instruction body for simple V0 p2p collectives
  -> `wafer.instr.dte_send` / `dte_recv` / `dte_wait`
```

Shardy / SPMD 只负责第一行之前的 logical collective 生成。StableHLO collective 不应在 group /
tiling 前直接 lower 成 `wafer.tile.*` communication，因为 `wafer.tile.*` communication 当前属于 tile-local storage / communication
IR；它需要 SPM buffer、byte count、endpoint 和 token/effect 语义。

R2.4 的用户级 program gate 是 `wafer-opt --program-pipeline=stablehlo-spmd-to-linalg`：该 pipeline
先执行 P2.S2 `stablehlo-spmd` stage，再在同一个 distributed program serialization 中逐 component 把
partitioned / replicated-local StableHLO body lowering 到 Linalg/Tensor IR，并把 StableHLO collective normalize 成
`wafer.linalg_ext.collective.*`。底层 `wafer-lower-stablehlo-to-linalg` named MLIR pipeline 只作为该
program pipeline 的内部构件和局部 debug/unit 覆盖；slot-aligned tile generation、physical rank
endpoint projection、ring/p2p schedule 和 DTE token 仍属于 R3/R6。

旧的 StableHLO collective 直降 `wafer.tile.*` communication ops pass 已移除。后续不得恢复
group/tiling 前的 StableHLO -> tile-local communication 插入点；需要分别实现
“StableHLO -> `wafer.linalg_ext.collective.*`”、“tiled tensor collective -> `wafer.tile.*`
buffer-level collective / direct p2p body”和“buffer-level collective -> `wafer.instr.dte_*`
schedule”。`collective_permute` 和当前 V0 `all_to_all` 可以不额外引入 `wafer.tile.*` wrapper，
只要 split/endpoint/token/body 都在 tile-region / instruction IR 中显式表达。

SDY program bridge 工程入口：`WAFER_ENABLE_SPMD_PARTITIONER_DEPS=ON` 时，`wafer-opt` 和
frontend verifier tool 显式注册 Shardy / SDY dialect，`wafer-opt` 也注册 SDY passes/pipelines。
带 `sdy.mesh` / `sdy.sharding` 的 partitioned StableHLO program 可以作为 Wafer 输入被 parse/verify。

同一批次曾把 StableHLO `replica_groups` 的 logical rank group materialize 到 `wafer.tile.*` communication ops
`rank_group = array<i64: ...>` attr；该 StableHLO -> `wafer.tile.*` communication bridge 已移除，避免后续误把它当成
group/tiling 输入。SPMD program stage 的任务是通过真实 partitioner 输出和 program verifier
保留这些事实，R2.4/R6 再分别恢复 `wafer.linalg_ext.collective.*` handoff、
`wafer.tile.*` buffer-level collective materialization 或 direct p2p body，以及
`wafer.instr.dte_*` schedule lowering。

## 6. 与 Target Environment / Topology / Execution Mesh 的接口

SPMD 消费的 target environment / topology / execution mesh 输入是：

- valid `wafer.execution.mesh` 的 logical mesh shape、axis 和 rank count。
- `wafer.execution.mesh` 引用的 `wafer.target.environment` / `wafer.target.topology`，用于确认 rank domain
  来自 compiler 支持且 available connected 的 topology；SPMD 只消费影响 sharding legality 的 capability，
  不直接消费 physical route、calibration score 或 DTE resource。

SPMD输出给后续program verifier、communication和pre-commit `ExecutableResourceView`的facts是：

- component-local shard shape、dtype、semantic layout 和 symbolic constraint projection。
- component graph、partition/replica coordinate、rank-class coverage 和 logical groups。
- immutable parameter / persistent state logical shard、alias/mutation relation。
- collective communication pattern。

通信volume、reuse pattern等cost hints必须从上述IR/type/group facts按需重算，只属于带provenance的
pass-local analysis或profile输出，不是distributed program协议字段，也不跨stage序列化。

execution mesh 的 derived / explicit endpoint view 一旦影响 codegen，就属于
`wafer.target.environment` + `wafer.execution.mesh` + `wafer.target.topology` 的派生 view，而不是 Shardy
attr。post-SPMD accepted launch/transport projection 是 physical endpoint / block binding 的唯一 owner；它
materialize 该 binding，但不能把同一 mapping 复制回 distributed program 或 execution mesh。

SPMD 不应该为了某个 physical tile id 编码重新解释 StableHLO semantics。mesh selection 可以在
SPMD 前拒绝、重排或拆分 requested logical mesh；SPMD 之后不能再通过 endpoint projection 改变 partition
后的数学语义。

## 7. Pass 合同

实现职责可以拆为：

| 职责 | 输入 | 输出 |
| --- | --- | --- |
| sharding import | StableHLO + old attrs | Shardy/SDY annotations |
| propagation | partially annotated module | fully propagated or diagnosed module |
| structured component formation/import | verified model program graph + Shardy MPMD/SDY + typed policy + execution mesh | verified `wafer.parallel.program` with components、participant predicates和typed cross-component edges |
| SPMD partition | annotated component + valid execution mesh | component-local partitioned StableHLO |
| distributed program assembly | component-local bodies + mesh/shard/group facts | verified components + canonical coordinates + coherent shape-guard proposal |
| logical rank-class formation | distributed program | target-independent rank-parametric/per-rank-static eligibility and full coverage |
| collective tensor normalization | distributed component StableHLO collective ops | component-local `wafer.linalg_ext.collective.*` IR |
| communication materialization | tiled tensor collective + storage values + topology/execution mesh | `wafer.tile.*` buffer-level collective and later `wafer.instr.dte_*` p2p schedule |

这些 pass 的合法输出不包含 Wafer physical memory space、tile coordinates、DTE resource 或
runtime package metadata。

## 8. Verifier and Diagnostics

必须检查：

- 每个 sharded value 的 shard rank、shape、dtype 与 global type 一致。
- parallel formation policy/version可识别，model-member coverage、component IDs、source mapping、local ABI ports、
  participant predicates和cross-component edges完整；名字、JSON或default component不能参与恢复。
- typed component ABI、symbolic constraint、parameter/state shard、alias/mutation 与 frontend model ABI 一致。
- logical mesh axes/roles、partition/replica coordinate、rank group 和 component coverage 与
  `wafer.execution.mesh` 可对齐；每个 required execution instance 恰好由一个 component/rank class 覆盖。
- rank-parametric class 保留显式 partition/replica SSA/ABI；per-rank-static class 有唯一 coordinate 的
  specialization record。任何默认 rank 0、hidden pass option 或无 coverage 常量折叠都非法。
- 同一 distributed variant 的全部 component/rank class 使用兼容的 symbolic guard 和 local projection；
  不允许 rank-local 独立 variant selection。
- rank class 内 local body、static local type/shape、resource role 和 collective/control role 等价；后续
  target-dependent stage 只能拆 class，不能合并当前不等价 class。
- collective 的 replica group、source/target rank 和 value type 一致。
- MPMD component edge 完整、type-compatible，且 persistent state edge 保留同一 resource identity/mutation。
- SPMD 输出中没有 physical tile id、SPM offset、DTE resource 或 packet field。

诊断要把问题定位到 logical sharding，不要提前报告为 Wafer SPM/DDR/packet 错误。

## 9. 验证和导出

P2.S2 的完成证明必须至少覆盖：

- P2.F1 verified program 可以作为 Shardy pipeline 输入。
- P2.F1 matmul 图在 data / batch、column parallel、row / contracting、2D output、2D contracting +
  output 和 partial replication 策略下都能通过 frontend mark 导出 sharding program；frontend
  export 主图保持 4096 形态，P2.S2 helper / lit gate 可使用同构小尺寸图重放 partition program
  chain，避免把验证变成 4096 工作集容量测试。主 gate 不以手写 `sdy.sharding` 测试输入代替真实导出。
- Shardy propagation 能直接消费每个 strategy 的 `functions/forward.mlir`，并且 XLA SPMD
  partitioner / equivalent service 能产出 component-local partitioned StableHLO body，并由 distributed
  program assembly 建立完整 execution-instance coverage。
- source-backed PP=2 case从framework/exporter typed model members进入`wafer.parallel.program`并形成两个stage
  components；heterogeneous expert case形成router/dispatch/expert/combine graph和bounded segmented edges。两者证明
  identity/edge不从function或buffer名恢复，missing/overlap edge/participant negatives在partition前失败；至少一个
  `dp x tp` case保留多轴coordinate和groups。
- `wafer-opt --program-pipeline=stablehlo-spmd-to-linalg` 能从真实 PyTorch/XLA sharded program
  逐 component 继续产出含 `wafer.linalg_ext.collective.*` 的 post-linalg distributed Wafer program，不能要求用户或 lit 手动拼
  `wafer-lower-stablehlo-to-linalg`。
- distributed program 仍通过 frontend typed boundary verifier 或等价 verifier，不丢 input/output、
  immutable parameter、persistent state/alias、symbolic bound、program-directory-derived constant 和 sharding facts。
- rank-parametric case 在两个 coordinate 上复用同一 class 并保留可观测 partition/replica identity；
  per-rank-static case 为每个 coordinate 保存 specialization/coverage record。两者都不允许默认 rank 0。
- 全局 dynamic case 至少有两个 guardable distributed variants，全部 component/rank class 对同一 invocation
  选择一致 variant；越界 actual shape 在进入 local lowering 前失败。
- P2.S2 输出必须完整保留 StableHLO collective 和 metadata，供 R2.4 进入
  `wafer.linalg_ext.collective.*` handoff；后段最小验证只能证明 metadata 能进入
  `wafer.tile.*` buffer-level collective 或 `wafer.instr.dte_*` schedule，不能作为 P2.S2、
  R2.4 或 group/tiling 完成证明。
- no-user-sharding P2.F1 program 必须通过默认 policy 生成 function-input sharding seed：rank count
  和 axes 来自 `wafer.execution.mesh`。单卡 4x4 / 16 tile 是默认 topology 配置；1-rank replicated
  只能作为显式 debug/bring-up execution mesh override 进入，不作为 SPMD 长期协议字段。
- 默认 policy 遇到 graph 内任意用户 sharding seed 时必须跳过，不覆盖用户只标了关键 op 后由
  Shardy propagation 推导整图的用法。

对 P2.S2 输出，后续 R3/R4/R6 测试应优先复用 distributed program 的 component/rank class 作为输入，逐步验证：

```text
distributed program
  -> per-component local compute normalization
  -> tensor collective normalization if collectives exist
  -> logical group
  -> tile_region materialization
  -> target-environment/topology/execution-mesh + typed shard facts / communication / memory planning /
     pre-commit ExecutableResourceView gate
```

手写 `sdy.mesh` / StableHLO collective 测试输入只保留为 dialect/verifier/unit 级测试。它不能替代
“verified frontend program -> Wafer Shardy/MPMD -> Wafer-owned SPMD partition -> distributed program”的主链路证明。

因此，P2.S2 之后每个消费 sharding / distributed program 的任务完成时，都必须继续使用真实图导出的
program chain 做端到端 gate。测试不能只构造一个新的手写 rank/component 输入，也不能只检查当前层
dump；必须证明前序 sharding facts 在本任务边界的 verifier、lowering、endpoint、communication
或 resource 逻辑中被实际使用。若直接下游尚未支持某个硬件可表达语义，应把缺口落成下游恢复任务
或补充 IR 表示，而不是修改上游 program 让其避开该语义。
