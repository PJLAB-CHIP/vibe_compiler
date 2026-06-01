# Wafer R2 Frontend / SPMD / Local Compute Recovery

日期：2026-05-26

状态：R2 完成记录

2026-06-01 复查更新：R2.2 中 StableHLO collective 直接 bridge 到 `wafer.comm` 的实现只保留为
已删除路线的后段 communication coverage，不再作为 P2.S1、group 或 tiling 的主线完成证明。真实 P2.S1 必须走
`frontend export -> StableHLO/SDY bundle -> Wafer Shardy propagation -> Wafer-owned XLA SPMD
partition artifact stage -> partitioned or replicated-local StableHLO`，post-SPMD collective 先进入
Wafer LinalgExt-style tensor collective handoff，再进入 group/tiling。没有用户 sharding 标记时，
P2.S1 在 SPMD 层补默认 single-card function-input sharding seed；不能绕到 placement/group 后再
补切分和通信。旧的 Python post-SPMD export 路线已删除，不能替代 Wafer-owned SPMD partition stage。

## 目标和非目标

R2 的目标是恢复 P2 阶段的主路径边界：frontend artifact 可验证，Shardy / SPMD artifact 能被
Wafer 工具链接收并保留 logical rank facts，local compute normalization 的覆盖状态按 structured
tensor IR 合同记录。

本记录本身不声明 framework-specific PyTorch/JAX capture、完整 Shardy propagation/SPMD partitioner
pipeline、mask/select/dynamic-shape 全覆盖、storage-realized constant slicing、group schedule
completion、SPM/DDR/resource planning 或 runtime/package 闭环完成。2026-05-27 后续 P2.F1/P2.S1
分别补上 source-built PyTorch/XLA capture adapter、真实 `mark_sharding` pre-SPMD artifact 和
Wafer Shardy propagation stage gate；2026-06-01 复查删除了误导性的 Python post-SPMD export
入口，它不是 P2.S1 主链完成证据。当前状态记录在 `tasks/progress.md` 和
`tasks/2026-05-25-wafer-shardy-spmd-design.md`。

2026-06-01 调度结论：framework-specific capture 和完整 Shardy propagation / SPMD partitioner
pipeline 不是放弃项，也不应排在 R3 之后。`P2.F1` 已完成 source-built PyTorch/XLA capture
adapter；`P2.S1` 骨架已接上真实 `mark_sharding` / default-input-seed 到 Wafer Shardy propagation
的 stage gate，但还没有 Wafer-owned XLA SPMD partition artifact stage。`P2.S2` 必须补这个阶段，
不能继续把 Python post-SPMD helper 当成主链。后续 group / tile / resource gate 必须消费真实
frontend/SPMD artifact 来源，而不是继续围绕手写 fixture 自洽。

2026-06-01 调度结论：R2.4 之前先补 R2.4-pre pipeline / driver contract。当前 P2.F1/P2.S1
已经证明 frontend export 和 Wafer Shardy propagation stage 能跑通；post-SPMD partitioned local
body 必须由 P2.S2 的 Wafer-owned stage 生成。后续主链路
必须通过库中注册的 named pipeline 或用户级 driver mode 表达，
bin 只负责注册和调用，单 pass flag 只保留为 unit/debug 入口。用户级 compile target 名称统一为
`wafer`；`tx8` 只保留为硬件/依赖逆向资料中的事实名，不作为 compiler driver target 字符串。

2026-06-01 实现结论：R2.4-pre 已新增 `WaferPipelines` 库并由 `wafer-opt` 注册。当前已收口的
named pipeline 是 `wafer-propagate-stablehlo-sharding`、`wafer-lower-stablehlo-to-linalg`、
`wafer-lower-linalg-to-cabi`、`wafer-lower-stablehlo-to-cabi` 和
`wafer-lower-tile-communication-to-cabi`。这些名字按 IR 输入/输出和职责命名，不按任务号、case
或 workload 命名；`wafer-propagate-stablehlo-sharding` 组合 Wafer default input seed 与
Shardy propagation，不冒充 XLA SPMD partitioner。`wafer-lower-stablehlo-to-cabi`
明确组合 StableHLO->Linalg 与 Linalg->C ABI 两层。`wafer-import-model
--propagate-stablehlo-sharding` 是 pre-SPMD bundle 到 Shardy propagation 的阶段检查入口；
`wafer-import-model --compile-stablehlo-bundle-to-cabi` 是用户级 compile 入口，从 verified local / post-SPMD
partitioned StableHLO bundle 进入 StableHLO->C ABI pipeline，并拒绝带 pre-SPMD sharding seed 但
没有 post-SPMD marker 的 bundle，防止绕过 Shardy/XLA SPMD。当前没有保留 Python post-SPMD
产物入口；P2.S2 必须接上 Wafer-owned XLA SPMD partition stage。
pipeline option `target=wafer` 会 materialize/校验 module 的
`#wafer.target<wafer>`；非 `wafer` target 会诊断。Integration 主链路 gate 已改为分层 named
pipeline 或 driver mode；lit 同时覆盖手写最小 bundle、真实 source-built PyTorch/XLA 小尺寸
export bundle 和 Shardy propagation driver。单 pass flags 继续只作为 Transforms/StageConnections
的局部 unit/debug 覆盖。

2026-06-01 pass 边界复查结论：当前 `test/Spmd` 两个 case 只覆盖 default input seed 和
SDY/Shardy artifact parse/verify，不覆盖 XLA SPMD partitioner，也不输出 rank-local StableHLO。
当前 `test/Frontend` 16 个 case 覆盖 StableHLO / Linalg local compute normalization，其中
softmax、RMSNorm 和 LayerNorm 的输入都是 fine-grained StableHLO staged graph，而不是
`stablehlo.softmax`、`stablehlo.norm` 或 Wafer 私有 high-level op。剩余 lit case 数量主要来自
Dialect verifier、Transforms、Frontend lowering、Pipelines、Integration 和 Tools，不是旧
Python post-SPMD oracle 残留。

## R2.1 Frontend Artifact / Importer Contract

实现边界：

- 新增 `WaferFrontend` target，`tools/wafer-import-model` 不再内联 artifact verifier 逻辑。
- `wafer-import-model --verify-import-result <mlir>` 作为 pre-exported
  StableHLO / MLIR artifact adapter 和 verifier。
- module 级 `wafer.import.graph_break`、`wafer.import.eager_fallback` 继续作为 importer 诊断 marker；
  true 值或 string marker 会被拒绝。
- bounded dynamic shape 通过 function argument/result attr
  `wafer.frontend.dynamic_bounds = [d0, d1, ...]` 表达；rank 必须匹配 tensor rank，dynamic dim
  的 bound 必须为正，static dim 的 bound 必须等于静态维度。没有 bound 的 dynamic shape 仍被拒绝。
- 真实 framework capture 的 artifact metadata 必须来自 exporter-native bundle。PyTorch/XLA 路线使用
  `functions/forward.mlir`、`functions/forward.meta`、`functions/forward.bytecode` 和 pre-SPMD
  `data/<parameter>`；post-SPMD partitioned artifact 额外使用
  `functions/forward.parameter_shards.json` 和 `parameter_shards/<parameter>/rank_XXXXX.npy` 表达
  rank-local parameter payload。

PyTorch/XLA bundle metadata 的 `input_locations` / `input_signature` 必须与 bundle MLIR 中唯一
`func.func` 的 argument ordinal、shape 和 dtype 一致；非 partitioned bundle 中 `parameter`
location 对应的 `data/<name>` 文件必须存在且 payload size 不小于 tensor raw byte size。partitioned
bundle 中 `parameter` location 必须由 post-SPMD shard manifest 和 rank-local payload 校验。
parameter name 只用于定位 exporter artifact 中的 parameter payload，不能成为后端 lowering 分支条件。

## R2.2 Shardy / SPMD Artifact Bridge

实现边界：

- `WAFER_ENABLE_SPMD_PARTITIONER_DEPS=ON` 时，`wafer-opt` 和 `wafer-import-model` 显式注册 SDY
  dialect；`wafer-opt` 同时注册 SDY passes/pipelines。关闭该选项时 core textual tests 不硬依赖
  Shardy。
- `test/Spmd/shardy-artifact-bridge.mlir` 覆盖 `sdy.mesh`、`sdy.sharding`、StableHLO
  `all_gather` textual artifact 和 frontend verifier 的同一 artifact 入口；它只证明 SDY dialect /
  StableHLO collective 可以进入工具链，不证明 XLA SPMD partition 或 rank-local body 已产生。
- 历史 StableHLO collective bridge 曾把 `replica_groups` materialize 到后段 `wafer.comm`
  `rank_group`，用于证明 rank-group metadata 可被 communication verifier 和 ring lowering 消费。
  该入口已经从主线路径删除，不能作为 group/tiling 前的 collective 表示恢复。
- 主线必须拆成 StableHLO -> Wafer LinalgExt-style tensor collective，以及 tiled tensor collective
  + SPM tile buffers -> `wafer.comm` 两层。`wafer.comm` 只能在 tile_region / SPM materialization /
  placement 明确后 materialize。

历史 bridge 覆盖状态：

- 历史 bridge 只覆盖单个 StableHLO replica group；多 replica-group artifact 需要由 SPMD artifact
  和 tensor collective handoff 保留 global/local rank selection policy。这是旧 bridge 覆盖缺口，
  不是 Wafer SPMD 语义不支持。
- R2.2 本身没有把 Shardy propagation/SPMD partitioner 作为 Wafer pass pipeline 主路径跑完；后续
  P2.S1 已用真实 PyTorch/XLA mark artifact 和 Wafer Shardy propagation gate 收回 pre-SPMD stage。
  Python post-SPMD export 入口已删除，不能定义后续主线入口。R2.2 只留下 artifact dialect/metadata
  bridge 的历史证据和后段 communication fixture。
- StableHLO collective 到 tile buffer 的 visible `unrealized_conversion_cast` 仍属于 R6.2 缺口。

## R2.3 Local Compute Normalization Coverage Status

当前 coverage 只说明 local shard structured tensor IR dataflow 可以被识别或 lowering，不说明 group
boundary、tile shape、multi-stage schedule、SPM residency 或 C ABI issue sequence 已完成。

| 子结构 | 当前证据 | 当前结论 | 仍未完成 |
| --- | --- | --- | --- |
| dot / 2D GEMM | `test/Frontend/lower-stablehlo-dot-to-linalg.mlir`、`stablehlo-dot-artifact.mlir`、`linalg-gemm-artifact.mlir` | StableHLO 2D dot 可降到 structured linalg matmul 输入 | accumulator dtype policy 和更宽 batch matmul family 仍需扩展 |
| attention QK^T / AV | `lower-stablehlo-attention-score.mlir`、`lower-stablehlo-attention-value.mlir`、`lower-stablehlo-attention-softmax-value.mlir` | rank-4 attention score/value 的 transpose relation 来自 dot dimension numbers 和 indexing map | 不代表 attention schedule、SPM residency 或 workspace 已完成 |
| elementwise / broadcast | `lower-stablehlo-elementwise.mlir`、`elementwise-broadcast-local-c-abi-issues.mlir`、projection residual gate | add/sub/mul/div/tanh/exp 等当前子集进入 `linalg.generic` / `arith` / `math` dataflow | complex broadcast、compare/select、mask add policy 仍未闭环 |
| reduce | `lower-stablehlo-reduce.mlir`、norm/softmax staged tests | fine-grained StableHLO reduce 到 `linalg.reduce` 的 constant-init 子集保留 reduction dimension 和 kind | non-constant-init reduce、NaN/overflow/approx policy 仍未闭环 |
| softmax | `lower-stablehlo-softmax-staged.mlir`、`softmax-schedule.mlir` | row max、subtract、exp、row sum、divide 的 fine-grained StableHLO SSA dataflow 可 lower 到 `linalg.reduce` / `linalg.generic` staged IR | acceptance pass 不是 schedule completion；multi-stage workspace/materialization 属 R3/R5/R6 后续 |
| norm | `lower-stablehlo-norm-staged.mlir`、`norm-schedule.mlir` | RMSNorm / LayerNorm staged graph 中 last-dim reduce、rsqrt、broadcast mul gate 已记录；没有 `wafer.norm` 或 high-level norm op | LayerNorm/RMSNorm 更宽 decomposition、epsilon policy 和 storage/resource 闭环未完成 |
| RoPE | `lower-stablehlo-rope-mlp-staged.mlir` | RoPE 当前作为 slice/shape/elementwise staged dataflow 覆盖 | sin/cos table 的 bundle/storage slicing 和更宽 shape family 未完成 |
| MLP | `lower-stablehlo-mlp-schedule.mlir`、`lower-stablehlo-local-transformer-block.mlir` | tanh-gated MLP vertical slice 和 full local transformer structured gate 已记录 | GELU/SwiGLU 其它 decomposition、constant slicing 和 package consistency 未完成 |
| shape views | `lower-stablehlo-shape.mlir`、local transformer block gate | static reshape expand/collapse 的 shape-only relation 可进入 local tensor IR | dynamic shape view、layout materialization 和 real movement 属后续层 |

## 验证

本批次新增或扩大了这些 gate：

- `test/Tools/wafer-import-model-reference.test`
- `test/Spmd/shardy-artifact-bridge.mlir`
- `test/Dialect/Wafer/Comm/invalid-comm-rank-group-size.mlir`
- `test/Transforms/ring-all-gather-rank-group.mlir`

2026-05-27 后续清理删除了旧的 `--verify-spmd-bundle`、P2.S1 私有 attr emitter 和 StableHLO
直降 `wafer.comm` pass/tests；上述 R2 记录只保留历史背景，不再表示这些旧入口仍存在。

2026-05-27 后续 P2.S1 新增或扩大了这些 gate：

- `test/Tools/wafer-pytorch-xla-capture-sharded-bundle.test`
- `test/Spmd/default-spmd-input-seed.mlir`

这些 gate 证明真实 source-built PyTorch/XLA `mark_sharding` artifact、default input seed policy 和
Wafer Shardy propagation stage 可被当前工具链验证。它们仍不证明 Wafer-owned propagation ->
XLA SPMD partition 接力、R2.4 tensor collective handoff、R3 group planner、resource planner、
package、runtime 或 board execution。

`test/Spmd/default-spmd-input-seed.mlir` 的输入是手写 StableHLO/SDY module，用来固定 no-user
default seed policy：默认 `tile-count=16`、调试 `tile-count=1`、已有用户 seed 时不覆盖、非法
tile count 诊断。它是 P2.S1 default-seed unit gate，不是 P2.S2 partitioner gate。P2.S2 完成前，
任何 `test/Spmd` 或 `test/Frontend` 的 FileCheck 都不能替代
`frontend artifact -> Wafer Shardy propagation -> Wafer-owned XLA SPMD partition -> per-rank artifact`
的主链路证明。

这些验证证明 R2 artifact/bridge/coverage 状态收敛，不证明 R3 之后的 group planner、resource
planner、package、runtime 或 board execution。

## R2 之后的消费链要求

R2 之后的完成证明必须从单点 fixture 转为跨阶段消费：

```text
P2.F1 framework capture artifact
  -> frontend verifier
  -> if user sharding seed exists:
       P2.S1 Shardy propagation
       -> P2.S2 XLA SPMD partitioner artifact stage
       -> partitioned StableHLO / per-rank artifact verifier
     else:
       P2.S1 default single-card input sharding seed
       -> Shardy propagation
       -> P2.S2 XLA SPMD partitioner artifact stage
       -> partitioned or replicated-local StableHLO / per-rank artifact verifier
  -> StableHLO / local compute normalization
  -> tensor collective normalization if collectives exist
  -> R3 group candidate
  -> R3 tile_region / resource / ABI / package gates
```

每个阶段可以保留手写 MLIR 做 negative verifier 测试，但主线完成证明必须说明该任务边界实际
消费或导出了上游产物中的哪些语义事实。只生成、dump 或 FileCheck 某层输出，而该层 verifier /
lowering / artifact writer 没有使用这些字段，不再作为主线完成依据。

从 P2.F1 开始，后续每个相关任务的完成证明都必须把这条真实图 artifact chain 至少推进到该任务
新增边界，并检查新增事实在该边界可验证、可导出或被直接消费。手写 artifact、单层 FileCheck 和
fixed manifest 只能作为补充覆盖，不能替代端到端可验证性。若下一层尚未实现某个硬件可表达语义，
应产生下游恢复任务或补充 IR contract，不能反向削弱当前层 artifact。

R2.4-pre 完成后，上述消费链不能再依赖用户或 lit 手动串联多个 tool / pass / env。主线 gate
必须通过 named pipeline 或用户级 driver mode 重放上游链路；pipeline 名称按 IR 边界和职责命名，
不能按 P2/R3 任务号、单个 workload 或 case 命名。`wafer-opt` 可以继续暴露单 pass 作为局部
debug/unit 入口，但这些 flag 不能被写成用户级 compile 流程。Python post-SPMD helper 已删除；
不得把 frontend capture 写成 SPMD partition 或用户编译入口。

当前 R2.4-pre 落地的 Wafer pipeline / driver 边界：

- `wafer-import-model --propagate-stablehlo-sharding`：pre-SPMD sharding propagation 检查入口。输入是
  `functions/forward.mlir`、`functions/forward.meta` 和可选 weight payload 组成的 StableHLO bundle；
  driver 先执行 bundle verifier，再调用 `wafer-propagate-stablehlo-sharding` 的 C++ builder，输出
  default input seed + Shardy propagation 后的 StableHLO/SDY module。
- `wafer-import-model --compile-stablehlo-bundle-to-cabi`：local / partitioned StableHLO artifact
  compile 入口。输入是 `functions/forward.mlir`、`functions/forward.meta` 和可选 weight/shard payload 组成的
  StableHLO bundle；driver 先执行 bundle verifier，再调用 `wafer-lower-stablehlo-to-cabi` 的 C++
  builder。带 pre-SPMD sharding seed 但没有 post-SPMD marker 的 bundle 会诊断，不能绕过
  Shardy/XLA SPMD 直接下沉。
- `wafer-propagate-stablehlo-sharding`：StableHLO/SDY sharding seed -> propagated StableHLO/SDY。
  该 pipeline 只负责 no-user default seed 和 Shardy propagation，不产生 partitioned local body；
  P2.S2 必须让 Wafer 自己消费该 propagated artifact 并产出 partitioned bundle。
- `wafer-lower-stablehlo-to-linalg`：StableHLO tensor IR -> Linalg/Tensor/Arith/Math/SCF 结构化
  tensor IR。它不做 group/tile/SPM/DDR/C ABI，也不承载 target 或 tile mapping。
- `wafer-lower-linalg-to-cabi`：已是 structured tensor/linalg 的 tensor compute -> group/tile/SPM/DDR
  -> C ABI issue IR；`tile-mapping=single` 是默认 materialization，
  `tile-mapping=multi-tile-no-comm` 只覆盖已有 no-communication tile materialization 路线。
- `wafer-lower-stablehlo-to-cabi`：StableHLO tensor IR -> Linalg/Tensor IR -> group/tile/SPM/DDR ->
  C ABI issue IR。它组合前两层的 body，不是单独一套隐藏转换。
- `wafer-lower-tile-communication-to-cabi`：已经处在 tile-level `wafer.comm` / p2p IR 的通信 -> C ABI
  issue IR。它不是 R2.4 tensor collective handoff；R2.4 仍要从 partitioned StableHLO collective
  normalizes 到 LinalgExt-style tensor collective，再进入 group/tiling。

P2.F1 主链路 artifact 采用 `4096x4096 @ 4096x4096` f32 matmul + bias + tanh + residual 最小验证。
该规模用于给后续 tiling、SPM/DDR resource、resident constant 和 package gate 提供非 trivial
shape/byte facts；pre-SPMD 大 weight 只能通过 PyTorch/XLA bundle `forward.meta` /
`data/<parameter>` 绑定，post-SPMD 大 weight 通过 `forward.parameter_shards.json` 和 rank-local
shard payload 绑定；不提交 64 MiB weight 到 git 测试文件。

PyTorch/XLA adapter 的完成证明必须使用从 `third_party/pytorch-xla` 源码编译/安装出的
`torch_xla` runtime。prebuilt `torch_xla` wheel、手写 StableHLO artifact 或没有实际运行
source-built runtime 的最小验证不能作为 P2.F1 done 依据。
