# Wafer Compiler Progress

更新时间：2026-06-02

本文件只记录当前看板、状态和下一步。设计边界、实现细节、验证记录和历史复盘分别以对应
`tasks/` 设计文档、git commit 和审计文档为准，不在这里重复。

## 状态标记

- `active`：当前优先推进。
- `ready`：前置边界已明确，并且任务已有 pipeline contract，可以开始。
- `pending`：依赖前序任务完成。
- `later`：当前主线之后再做。
- `骨架`：有局部 IR / pass / verifier / fixture，但不代表主路径完成。
- `done`：实现和验证已按当前设计合同完成；P2.F1 之后的相关任务还必须有真实图 artifact chain 证明，
  且不能只用单 pass、手写 fixture、工具脚本或局部 FileCheck 作为主线完成证明。

## 看板约束

主线任务进入 `active` 前必须能在对应设计文档或本看板中找到明确的 pipeline contract：

- upstream artifact / IR。
- current stage responsibility。
- output artifact / IR。
- downstream consumer。
- user-level driver / named pipeline。
- explicit non-goals。
- completion gate。

如果只能说明某个 pass / tool / test 要做什么，而不能说明它在整条 compiler pipeline 中消费什么、
产出什么、由谁继续消费，就不能把该任务推进为 `active` 或 `done`。

## 当前主线

当前 active / ready / done 项：

| ID | 状态 | 任务 | 完成标准 |
| --- | --- | --- | --- |
| P2.S2 | done | 建立 Wafer-owned XLA SPMD partition artifact stage | 消费经过 Wafer sharding propagation stage 的 StableHLO bundle，调用 XLA SPMD partitioner 或等价 stage，输出 post-SPMD local / replicated-local StableHLO bundle、post-SPMD marker、rank-local function signature、collective metadata、`forward.parameter_shards.json` 和 rank-local parameter payload |
| R2.4 | done | Wafer LinalgExt-style tensor collective handoff | 消费 P2.S2 partitioned StableHLO bundle 中的 logical collective 和 rank-local signature，normalize 成后续 local compute / group pipeline 可消费的 `wafer.tensor_collective.*` tensor IR |
| R3.1 | ready | group boundary / candidate contract | 消费 P2.S2/R2.4 真实 frontend/SPMD artifact chain 的 local compute IR 和 tensor collective IR，建立 group candidate 的 IR 边界和完成 gate |

P2.S2 pipeline contract：

- upstream artifact / IR：P2.F1 verified PyTorch/XLA StableHLO bundle，经过 Wafer
  sharding propagation stage 处理后的 StableHLO/SDY IR。
- current stage responsibility：在 Wafer-owned driver / library stage 中调用 XLA SPMD partitioner
  或等价 local-body partitioning service，并导出 partitioned / replicated-local artifact。
- output artifact / IR：partitioned / replicated-local StableHLO bundle、post-SPMD marker、
  rank-local function signature、collective metadata、`forward.parameter_shards.json` 和 rank-local
  parameter payload。
- downstream consumer：R2.4 tensor collective handoff、`wafer-lower-stablehlo-to-linalg` local
  compute normalization、R3 group pipeline 和后续 placement / package stages。
- user-level driver / named pipeline：应由 Wafer driver / named pipeline 消费 sharding propagation
  stage 的输出 artifact；不能
  由 Python capture helper 或 integration test 手动拼外部流程替代。
- explicit non-goals：不发明 `wafer.spmd.*` 私有协议，不在 frontend Python 中做 partition，不把
  下游 group/placement/comm 未完成当作当前不支持。
- completion gate：真实 frontend export -> Wafer sharding propagation artifact chain 能被 P2.S2 消费并产出下游可直接验证的
  partitioned / replicated-local bundle；fixture 只做补充覆盖。

R3.1 pipeline contract：

- upstream artifact / IR：P2.F1/P2.S1/P2.S2/R2.4 真实链路产出的 rank-local `linalg` / `tensor` /
  `scf` local compute IR 和 `wafer.tensor_collective.*` tensor collective IR。
- current stage responsibility：建立 `wafer.group` candidate 边界，说明哪些 tensor SSA value、outs、
  producer/consumer 和 tensor collective 能进入 group 候选，并由 verifier 拒绝 raw StableHLO、
  physical buffer、placement、SPM/DTE 或 runtime metadata。
- output artifact / IR：可验证的 tensor-level `wafer.group` candidate IR；不含 tile buffer、layout
  materialization、`wafer.comm`、C ABI issue 或 package manifest。
- downstream consumer：R3.2 root tile feasibility、R3.3 tile_region materialization、R3.4/R3.5
  layout/SPM/DDR feasibility 和后续 ABI/package stages。
- user-level driver / named pipeline：应由 Wafer named pipeline 或 driver mode 重放 frontend/SPMD/R2.4
  chain 后进入 group candidate gate；不能让 integration test 手动拼 raw StableHLO、tensor collective
  fixture 和 group fixture 作为长期主线。
- explicit non-goals：不做 physical placement、tile shape search、SPM allocation、DTE schedule、
  `wafer.comm` materialization、C ABI 或 package emission。
- completion gate：真实 P2.S2/R2.4 artifact chain 的 local compute + tensor collective 输出能进入
  group candidate gate，且 verifier 证明 group 边界只包含 tensor-level IR；fixture 只做负例和局部覆盖。

P2.S2/R2.4 之后的直接顺序：

1. R3.1：真实 frontend/SPMD artifact 两个分支进入 group candidate gate。
2. R3.2-R3.8：恢复 tile/resource/storage/C ABI/package 主链路。

P7/P8/P9 依赖 P0-P6 主链路恢复，不提前推进。

## 当前边界

- P2.F1 已完成 source-built PyTorch/XLA capture adapter；frontend Python 只导出 reference bundle 或
  pre-SPMD sharded bundle，不负责 Shardy propagation、XLA SPMD partition 或 per-rank artifact。
- P2.S1 当前是 `骨架`：真实 `mark_sharding` / no-user default input seed 已进入 Wafer
  Shardy propagation gate；它不产生 partitioned local body。
- `wafer-propagate-stablehlo-sharding` 只做 default seed + Shardy propagation；不是 XLA SPMD partitioner。
- P2.S2 才拥有 XLA SPMD partition artifact stage。旧 Python post-SPMD helper 和 oracle tests 已删除。
- R2.4 才把 partitioned StableHLO collective normalize 成 tensor-level collective；`wafer.comm` 只能在
  tile_region / SPM materialization / placement 明确后 materialize。
- R2.4 已建立 `wafer.tensor_collective.*` op family 和
  `wafer-normalize-stablehlo-collectives` pass，并接入 `wafer-lower-stablehlo-to-linalg` named
  pipeline。该层使用 destination-style tensor operand/result 和 Wafer tiling demand interface，
  不拥有 physical placement、SPM tile buffer、DTE token、byte schedule 或 runtime handle。
- `wafer-lower-stablehlo-to-linalg` 只做 local compute normalization；不承载 sharding propagation、
  SPMD partition、group、placement、SPM/DDR 或 C ABI。
- `wafer-lower-linalg-to-cabi`、`wafer-lower-stablehlo-to-cabi`、
  `wafer-lower-tile-communication-to-cabi` 和
  `wafer-import-model --compile-stablehlo-bundle-to-cabi` 已删除；旧 single-tile materialization、ring
  lowering、SPM/DDR trial 和 tile_region-to-C-ABI issue lowering 的 explicit unit/debug pass 链也已删除。
  R3/R6/R7 必须按新的 IR contract 从真实 artifact chain 恢复，不能复用这条旧链作为完成证明。
- P2.S2 已完成当前 artifact gate：`wafer-import-model --partition-stablehlo-bundle` 负责 bundle
  verify、in-memory Wafer sharding propagation、临时 propagated bundle、pinned-XLA helper 调用和输出
  bundle verifier。helper 执行 StableHLO/SDY -> XLA HLO、`SpmdPrepare` / `SpmdPartitioner` /
  `HloVerifier`、partitioned HLO -> StableHLO round trip，并写回 rank-local function signature、
  StableHLO collective metadata、`forward.parameter_shards.json` 和
  `parameter_shards/<parameter>/rank_XXXXX.npy` NPY stream payload。
- `test/Spmd` 当前只覆盖 default input seed 和 SDY/Shardy artifact parse/verify，不覆盖 XLA SPMD
  partitioner，也不输出 rank-local StableHLO。
- `test/Frontend` 当前覆盖 StableHLO/Linalg local compute normalization；softmax、RMSNorm、LayerNorm 是
  fine-grained StableHLO staged graph 到 `linalg.reduce` / `linalg.generic` 的 lowering，不是 high-level
  softmax/norm op，也不是 SPMD artifact gate。

## 验证口径

当前本地验证能覆盖：

- CMake / `wafer-opt` / lit / gtest 构建入口。
- MLIR textual tests、FileCheck、verifier negative tests。
- 固定依赖版本、importer backend 隔离和源码依赖层级检查。
- Wafer IR op-family 文件组织、interface/effect/resource 查询合同。
- frontend artifact verifier、PyTorch/XLA bundle metadata / pre-SPMD parameter payload 校验。
- SDY `sdy.mesh` / `sdy.sharding` parse/verify、default input seed、Shardy propagation driver gate。
- P2.S2 pinned-XLA SPMD helper build、six-strategy frontend artifact -> Wafer propagation -> XLA SPMD
  partition -> post-SPMD bundle verification gate。
- StableHLO/Linalg local compute normalization、package manifest validator/stub tool-unit fixture。
- R2.4 StableHLO collective handoff：`all_gather` / `all_reduce` / `reduce_scatter` /
  `all_to_all` / `collective_permute` fixture 能 normalize 成 `wafer.tensor_collective.*`；P2.S2
  真实 partitioned bundle 的 `forward.mlir` 能经 `wafer-lower-stablehlo-to-linalg` 产出
  `wafer.tensor_collective.all_gather`，且不绕到 `wafer.comm`。

不能作为主线完成证明：

- 手写 MLIR fixture、单层 FileCheck 或显式 manifest tool-unit fixture 单独成立。
- Python capture、`test/Spmd` 或 `test/Frontend` 替代 P2.S2 partition artifact stage。
- C stub syntax compile 替代 LLVM IR、object、真实 runtime call 或 wrapper/packet lowering。
- 本地 lit/gtest 替代板端 launch、device completion、数值对比或 PMU/profiling。

## 已完成锚点

这些项只作为当前队列的前置锚点，详细记录见对应设计/恢复文档：

- R0.1-R0.3：P0-P6 状态重读、源码 ownership、依赖层级和版本一致性恢复。
- R1.1-R1.3：Wafer IR op-family 文件边界、interface/effect/resource 合同；历史 stage-connection unit
  gate 已在 2026-06-02 清理中删除，后续 R3 按真实 artifact chain 恢复。
- R2.1-R2.3：frontend artifact verifier、SDY artifact bridge、local compute normalization 覆盖状态。
- P2.F1：source-built PyTorch/XLA capture adapter。
- R2.4-pre：Wafer named pipeline / driver contract。

参考文档：

- `tasks/2026-05-26-wafer-p0-p6-recovery-status.md`
- `tasks/2026-05-26-wafer-p0-p6-design-conformance-audit.md`
- `tasks/2026-05-25-wafer-shardy-spmd-design.md`
- `tasks/2026-05-25-wafer-local-compute-normalization-design.md`
- `tasks/2026-05-26-wafer-r2-recovery.md`

## 恢复队列

| ID | 状态 | 任务 | 依赖 / 说明 |
| --- | --- | --- | --- |
| P2.S2 | done | Wafer-owned XLA SPMD partition artifact stage | `wafer-import-model --partition-stablehlo-bundle` 已接真实 pinned XLA helper/service，并通过六种 sharding strategy 的真实 artifact gate |
| R2.4 | done | Wafer LinalgExt-style tensor collective handoff | `wafer.tensor_collective.*` op / verifier / StableHLO normalization pass 已接入 named pipeline，并通过 P2.S2 真实 artifact handoff gate |
| R3.1 | ready | group boundary / candidate contract | 依赖 P2.S2、R2.4 和真实 frontend/SPMD artifact |
| R3.2 | pending | root tile feasibility oracle | 依赖 R3.1 |
| R3.3 | pending | tile_region materialization contract | 依赖 R3.1/R3.2 |
| R3.4 | pending | layout / SPM feasibility gate | 依赖 R3.3 |
| R3.5 | pending | DDR / resource demand gate | 依赖 R3.3 |
| R3.6 | pending | ABI issue gate | 依赖 storage-realized compute/movement IR |
| R3.7 | pending | package manifest gate | 依赖 ABI issue gate |
| R3.8 | pending | storage-realized / C ABI / golden packet 边界 | 依赖 R3.6/R3.7 |
| R4.1-R4.5 | pending | placement、per-rank identity、shard slicing、writeback、package artifact | 依赖 P2.S2/R2.4/R3 |
| R5.1-R5.2 | pending | transformer local compile 和 compute/package gaps | 依赖 R3/R4 |
| R6.1-R6.2 | pending | communication conformance 和 tiled tensor collective -> comm materialization | 依赖 R2.4/R3/R4 |
| P7/P8/P9 | later | package/runtime/LLVM/object/board/profiling 后续主线 | P0-P6 恢复后再推进 |

## 当前不做

- Serving integration。
- KV cache / paged attention / prefill-decode 调度。
- raw DTE non-unicast collective ABI。
- LLVM dialect / LLVM IR lowering、object emission 或真实 `wafer_*` runtime call emission。
- 自定义 LLVM backend 或 ISA intrinsic lowering。
- 以 importer、runtime path、workload shape 或 parameter 名称作为 IR 合同。

## 下一步

推进 R3.1。下一步消费 `wafer-lower-stablehlo-to-linalg` 输出的 `linalg` / `tensor` / `scf`
local compute IR 和 `wafer.tensor_collective.*` tensor collective IR，建立 group boundary /
candidate contract。不能把 raw StableHLO collective、`wafer.comm`、SPM tile buffer、DTE token、
Python helper 或手写 fixture 当作 group 主线输入。
