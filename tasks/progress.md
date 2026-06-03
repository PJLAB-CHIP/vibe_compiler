# Wafer Compiler Progress

更新时间：2026-06-03

本文件只记录当前看板、主线 pipeline、完成口径和下一步。设计细节、历史复盘和长验证说明放在对应
`tasks/` 设计文档、git commit 和测试里，不在这里重复。

## 状态标记

- `active`：当前优先推进。
- `ready`：前置边界已明确，可以开始。
- `pending`：依赖前序任务完成。
- `later`：当前主线之后再做。
- `done`：实现、测试和文档已按当前 pipeline contract 收口。

## 当前主线

主线用户入口统一为 `wafer-opt` program pipeline。`wafer-compile-stablehlo` 只做 frontend /
StableHLO program verifier；`wafer-propagate-stablehlo-sharding` 和
`wafer-lower-stablehlo-to-linalg` 只作为内部/局部测试用的 named MLIR pipeline，不是用户级编译流程。

```text
PyTorch/XLA StableHLO Wafer program directory
  -> wafer-opt --program-pipeline=stablehlo-spmd
       verify program dir
       Wafer default sharding seed / Shardy propagation
       pinned XLA SPMD helper from build-time WAFER_XLA_SPMD_PARTITIONER_HELPER
       write post-SPMD StableHLO program + parameter shards
  -> wafer-opt --program-pipeline=stablehlo-spmd-to-linalg
       same SPMD stage
       write back Linalg/Tensor/SCF local compute + wafer.tensor_collective.*
  -> R3 group / tiling / placement / package recovery
```

用户级 command 不传 helper path：

```bash
wafer-opt \
  --program-pipeline=stablehlo-spmd-to-linalg \
  --input-program-dir <input.program> \
  --output-program-dir <output.program> \
  --default-tile-count=16
```

## 已完成链路

| ID | 状态 | 输入 | 输出 / 完成 gate |
| --- | --- | --- | --- |
| P2.F1 | done | PyTorch/XLA model + parameter payload | StableHLO Wafer program directory；frontend 只导出 reference / pre-SPMD sharded program，不做 Shardy/SPMD |
| P2.S1 | done | P2.F1 program 或 textual StableHLO/SDY fixture | default input seed + Shardy propagation；作为 `stablehlo-spmd` 内部阶段，不产出 partitioned local body |
| P2.S2 | done | P2.F1 sharded program | `stablehlo-spmd` 产出 post-SPMD StableHLO program、rank-local signature、collective metadata、parameter shard metadata/payload |
| R2.4 | done | post-SPMD StableHLO program | `stablehlo-spmd-to-linalg` 产出 Linalg/Tensor/SCF local compute 和 `wafer.tensor_collective.*`；不生成 `wafer.comm` |

## 恢复队列

| ID | 状态 | 任务 | 输入 / 输出边界 | 完成 gate |
| --- | --- | --- | --- | --- |
| R3.1 | ready | group boundary / candidate contract | 输入：`stablehlo-spmd-to-linalg` tensor-level IR；输出：root-seeded verifier-legal `wafer.group` candidate | 真实 program chain 的 local compute + tensor collective 进入 `stablehlo-spmd-to-group` gate；raw StableHLO/`wafer.comm`/SPM/DTE 被拒绝 |
| R3.2 | pending | root tile feasibility oracle | 输入：`wafer.group` candidate；输出：accepted/rejected root tile candidate 和明确拒绝原因 | op tiling、layout、SPM、DDR、compute/movement legality 接到同一 candidate check |
| R3.3 | pending | tile_region materialization contract | 输入：R3.2 accepted group；输出：`wafer.tile_region` | 只 materialize accepted group；未接受 group 不产生 tile_region |
| R3.4 | pending | layout / SPM feasibility gate | 输入：`wafer.tile_region` tensor effects/liveness；输出：layout + SPM feasibility facts | layout materialization 和 SPM trial 由 effect、range 和 tile buffer lifetime 驱动 |
| R3.5 | pending | DDR / resource demand gate | 输入：storage-aware tile IR；输出：external/workspace/resident-constant/resource demand | 覆盖 pool/domain/capacity/bandwidth demand，不能用 manifest fixture 替代 |
| R3.6 | pending | ABI issue gate | 输入：storage/movement/compute IR；输出：`wafer.abi.*` issue sequence | issue sequence 从 IR 派生，只表达 ABI 参数单位和 wait policy |
| R3.7 | pending | package manifest gate | 输入：ABI issue / launch signature IR；输出：IR-derived package manifest | manifest、C stub、launch signature 从当前 `wafer-opt` 输出导出 |
| R3.8 | pending | storage-realized / C ABI / golden packet boundary | 输入：R3.6/R3.7 输出；输出：wrapper-facing call contract 和 golden packet | 至少 RDMA/WDMA/GEMM 有真实 wrapper-facing call contract 和 packet 对照 |
| R4.1 | pending | placement map / capability gate | 输入：logical rank / group candidate；输出：accepted physical placement | placement 来自 logical rank、good-tile/PG capability 和 coordinate verifier |
| R4.2 | pending | per-rank identity / launch metadata gate | 输入：placement + local shard metadata；输出：rank/block/coord launch metadata | logical rank、block id、physical coord 接到 tile_region/launch/package 边界 |
| R4.3 | pending | shard slicing materialization | 输入：rank-local signature + shard metadata；输出：per-rank input/output slices | multi-tile no-comm materialize local slice，不 clone whole tensor |
| R4.4 | pending | per-rank writeback / merge contract | 输入：per-rank output slices；输出：writeback / host readback / merge responsibility | sharded output 的 IR/package 责任明确 |
| R4.5 | pending | placed package / generated program gate | 输入：placement + local shard + issue sequence；输出：placed package program | placement metadata、local shards、launch args 从 lowering 输出导出 |
| R5.1 | pending | transformer local compile validation | 输入：static transformer local shard IR；输出：accepted schedule 或拒绝原因 | workspace/resident constants/ABI issue sequence 来自 full-block IR dataflow |
| R5.2 | pending | transformer compute coverage gaps | 输入：transformer staged IR gaps；输出：补齐 compute/package consistency | 覆盖 mask/select、dynamic-bound policy、non-constant-init reduce、constant/weight slice |
| R6.1 | pending | communication conformance gate | 输入：tiled tensor collective / placement facts；输出：DTE resource and package/runtime metadata | DTE allocation、collective buffer slice/address offset、metadata 可验证 |
| R6.2 | pending | tensor collective -> comm materialization | 输入：buffer-slice/layout/materialized collective；输出：`wafer.comm` | 移除临时 visible cast，由可验证 buffer-slice/materialization 路径承接 |
| P7/P8/P9 | later | package/runtime/LLVM/object/board/profiling | 输入：P0-P6 恢复后的 package/runtime boundary | P0-P6 主链路恢复后再推进 |

## R3.1 Pipeline Contract

- upstream program / IR：`stablehlo-spmd-to-linalg` 输出的 rank-local `linalg` / `tensor` / `scf`
  local compute IR 和 `wafer.tensor_collective.*` tensor collective IR。
- current stage responsibility：建立 `wafer.group` candidate 边界，说明哪些 tensor SSA value、outs、
  producer/consumer 和 tensor collective 能进入 group 候选。
- output program / IR：可验证的 tensor-level `wafer.group` candidate IR。
- downstream consumer：R3.2 root tile feasibility、R3.3 tile_region materialization、R3.4/R3.5
  layout/SPM/DDR feasibility、R3.6/R3.7 ABI/package stages。
- user-level driver / named pipeline：计划新增
  `wafer-opt --program-pipeline=stablehlo-spmd-to-group`，由该 program pipeline 重放
  frontend/SPMD/R2.4 后进入 group candidate gate；不能让 integration test 手动拼 raw StableHLO、
  tensor collective fixture 和 group fixture 作为长期主线。
- explicit non-goals：不做 physical placement、tile shape search、SPM allocation、DTE schedule、
  `wafer.comm` materialization、C ABI 或 package emission。
- completion gate：真实 P2.S2/R2.4 program chain 的 local compute + tensor collective 输出能进入
  `stablehlo-spmd-to-group` group candidate gate，且 verifier 证明 group 边界只包含 tensor-level IR；
  fixture 只做负例和局部覆盖。

## 当前不做

- Serving integration。
- KV cache / paged attention / prefill-decode 调度。
- raw DTE non-unicast collective ABI。
- LLVM dialect / LLVM IR lowering、object emission 或真实 `wafer_*` runtime call emission。
- 自定义 LLVM backend 或 ISA intrinsic lowering。
- 以 importer、runtime path、workload shape 或 parameter 名称作为 IR 合同。

## 最近验证

当前 `stablehlo-spmd-to-linalg` 主线收口已验证：

- `cmake --build build/r0-deps-pytorch-xla --target check-wafer -- -j8`：102 passed, 1 unsupported。
- `ctest --test-dir build/r0-deps-pytorch-xla --output-on-failure`：2/2 passed。
- `python3 tools/check_deps.py`。
- `python3 tools/check_ir_organization.py --root .`。
- `git diff --check`。

## 下一步

推进 R3.1。新增 `stablehlo-spmd-to-group` program pipeline gate，在重放
`stablehlo-spmd-to-linalg` 真实 program chain 后形成 root-seeded logical `wafer.group`
candidate。输入必须是 `linalg` / `tensor` / `scf` local compute IR 和
`wafer.tensor_collective.*` tensor collective IR。不要把 raw StableHLO collective、`wafer.comm`、
SPM tile buffer、DTE token、Python helper 或手写 fixture 当作 group 主线输入。
