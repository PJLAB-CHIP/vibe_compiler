# Wafer R2 Frontend / SPMD / Local Compute Recovery

日期：2026-05-26

状态：R2 完成记录

## 目标和非目标

R2 的目标是恢复 P2 阶段的主路径边界：frontend artifact 可验证，Shardy / SPMD artifact 能被
Wafer 工具链接收并向 communication/placement 传递 logical rank facts，local compute
normalization 的覆盖状态按 structured tensor IR 合同记录。

本记录不声明 framework-specific PyTorch/JAX capture、完整 Shardy propagation/SPMD partitioner
pipeline、mask/select/dynamic-shape 全覆盖、storage-realized constant slicing、group schedule
completion、SPM/DDR/resource planning 或 runtime/package 闭环完成。

## R2.1 Frontend Artifact / Importer Contract

实现边界：

- 新增 `WaferFrontend` target，`tools/wafer-import-model` 不再内联 artifact verifier 逻辑。
- `wafer-import-model --verify-import-result <mlir> [--sidecar <json>]` 作为 pre-exported
  StableHLO / MLIR artifact adapter 和 verifier。
- module 级 `wafer.import.graph_break`、`wafer.import.eager_fallback` 继续作为 importer 诊断 marker；
  true 值或 string marker 会被拒绝。
- bounded dynamic shape 通过 function argument/result attr
  `wafer.frontend.dynamic_bounds = [d0, d1, ...]` 表达；rank 必须匹配 tensor rank，dynamic dim
  的 bound 必须为正，static dim 的 bound 必须等于静态维度。没有 bound 的 dynamic shape 仍被拒绝。
- sidecar manifest V0 是 JSON object：

```json
{
  "version": 0,
  "constants": [
    {
      "function": "weight_artifact",
      "arg": 0,
      "resource_key": "w0",
      "shape": [2, 4],
      "dtype": "f32",
      "byte_size": 32,
      "checksum": "sha256:..."
    }
  ]
}
```

sidecar constant 必须指向 `func.func` argument ordinal；该 argument 必须带
`wafer.frontend.constant = "<resource_key>"`，并且 shape、dtype、byte size 与 argument tensor type
一致。这里的 attr 是 frontend artifact metadata，不是 Wafer low-level constant op，也不表达 DDR
pool、physical layout、BO handle 或 package path。

## R2.2 Shardy / SPMD Artifact Bridge

实现边界：

- `WAFER_ENABLE_SPMD_PARTITIONER_DEPS=ON` 时，`wafer-opt` 和 `wafer-import-model` 显式注册 SDY
  dialect；`wafer-opt` 同时注册 SDY passes/pipelines。关闭该选项时 core textual tests 不硬依赖
  Shardy。
- `test/Spmd/shardy-artifact-bridge.mlir` 覆盖 `sdy.mesh`、`sdy.sharding`、partitioned StableHLO
  `all_gather` 和 frontend verifier 的同一 artifact 入口。
- StableHLO collective bridge 不再只保留 `group_size/local_rank`。`wafer.comm.all_gather`、
  `wafer.comm.all_reduce` 和 `wafer.comm.reduce_scatter` 现在显式携带
  `rank_group = array<i64: ...>`，该值来自 StableHLO `replica_groups`。
- comm verifier 检查 `rank_group` size、非负和唯一性；ring lowering 用 `rank_group` 查询
  `wafer.placement.map` 的 logical-rank 到 physical tile 映射，不再假设 logical rank 连续等于
  `0..group_size-1`。

保留限制：

- V0 bridge 只接受单个 StableHLO replica group；多 replica-group artifact 需要后续引入明确的
  global/local rank selection policy。
- 当前仍没有把 Shardy propagation/SPMD partitioner 作为 Wafer pass pipeline 主路径跑完；R2.2
  只恢复 artifact dialect/metadata bridge 和 logical collective 到 Wafer comm 输入。
- StableHLO collective 到 tile buffer 的 visible `unrealized_conversion_cast` 仍属于 R6.2 缺口。

## R2.3 Local Compute Normalization Coverage Status

当前 coverage 只说明 local shard structured tensor IR dataflow 可以被识别或 lowering，不说明 group
boundary、tile shape、multi-stage schedule、SPM residency 或 C ABI issue sequence 已完成。

| 子结构 | 当前证据 | 当前结论 | 仍未完成 |
| --- | --- | --- | --- |
| dot / 2D GEMM | `test/Frontend/lower-stablehlo-dot-to-linalg.mlir`、`stablehlo-dot-artifact.mlir`、`linalg-gemm-artifact.mlir` | StableHLO 2D dot 可降到 structured linalg matmul 输入 | accumulator dtype policy 和更宽 batch matmul family 仍需扩展 |
| attention QK^T / AV | `lower-stablehlo-attention-score.mlir`、`lower-stablehlo-attention-value.mlir`、`lower-stablehlo-attention-softmax-value.mlir` | rank-4 attention score/value 的 transpose relation 来自 dot dimension numbers 和 indexing map | 不代表 attention schedule、SPM residency 或 workspace 已完成 |
| elementwise / broadcast | `lower-stablehlo-elementwise.mlir`、`elementwise-broadcast-local-c-abi-skeleton.mlir`、projection residual gate | add/sub/mul/div/tanh/exp 等当前子集进入 `linalg.generic` / `arith` / `math` dataflow | complex broadcast、compare/select、mask add policy 仍未闭环 |
| reduce | `lower-stablehlo-reduce.mlir`、norm/softmax staged tests | constant-init reduce 子集保留 reduction dimension 和 kind | non-constant-init reduce、NaN/overflow/approx policy 仍未闭环 |
| softmax | `lower-stablehlo-softmax-staged.mlir`、`softmax-schedule.mlir` | row max、subtract、exp、row sum、divide 的 SSA dataflow gate 已记录 | acceptance pass 不是 schedule completion；multi-stage workspace/materialization 属 R3/R5/R6 后续 |
| norm | `lower-stablehlo-norm-staged.mlir`、`norm-schedule.mlir` | last-dim reduce、rsqrt、broadcast mul staged gate 已记录 | LayerNorm/RMSNorm 更宽 decomposition、epsilon policy 和 storage/resource 闭环未完成 |
| RoPE | `lower-stablehlo-rope-mlp-staged.mlir` | RoPE 当前作为 slice/shape/elementwise staged dataflow 覆盖 | sin/cos table 的 sidecar/storage slicing 和更宽 shape family 未完成 |
| MLP | `lower-stablehlo-mlp-schedule.mlir`、`lower-stablehlo-local-transformer-block.mlir` | tanh-gated MLP vertical slice 和 full local transformer structured gate 已记录 | GELU/SwiGLU 其它 decomposition、constant slicing 和 package consistency 未完成 |
| shape views | `lower-stablehlo-shape.mlir`、local transformer block gate | static reshape expand/collapse 的 shape-only relation 可进入 local tensor IR | dynamic shape view、layout materialization 和 real movement 属后续层 |

## 验证

本批次新增或扩大了这些 gate：

- `test/Tools/wafer-import-model-smoke.test`
- `test/Spmd/shardy-artifact-bridge.mlir`
- `test/Transforms/stablehlo-collectives-to-comm.mlir`
- `test/Dialect/Wafer/Comm/invalid-comm-rank-group-size.mlir`
- `test/Transforms/ring-all-gather-rank-group.mlir`

这些验证证明 R2 artifact/bridge/coverage 状态收敛，不证明 R3 之后的 group planner、resource
planner、package、runtime 或 board execution。
