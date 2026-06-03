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

## 看板

| ID | 状态 | 任务 | 当前边界 |
| --- | --- | --- | --- |
| P2.F1 | done | PyTorch/XLA StableHLO Wafer program export | frontend 只导出 reference / pre-SPMD sharded program directory；不做 Shardy、SPMD partition 或 per-rank payload |
| P2.S1 | done | Wafer Shardy propagation stage | default input seed + Shardy propagation 已作为 `stablehlo-spmd` 内部阶段；不产出 partitioned local body |
| P2.S2 | done | Wafer-owned XLA SPMD partition stage | `stablehlo-spmd` 产出 post-SPMD StableHLO program、rank-local signature、collective metadata 和 parameter shard payload |
| R2.4 | done | LinalgExt-style tensor collective handoff | `stablehlo-spmd-to-linalg` 产出 Linalg/Tensor/SCF local compute 和 `wafer.tensor_collective.*`；不生成 `wafer.comm` |
| R3.1 | ready | group boundary / candidate contract | 消费真实 frontend/SPMD/R2.4 program chain 的 tensor-level IR，建立 `wafer.group` candidate 边界 |
| R3.2-R3.8 | pending | tile/resource/storage/C ABI/package recovery | 依赖 R3.1 |
| R4-R6 | pending | placement、shard slicing、communication materialization | 依赖 R3 |
| P7/P8/P9 | later | package/runtime/LLVM/object/board/profiling | P0-P6 主链路恢复后再推进 |

## R3.1 Pipeline Contract

- upstream program / IR：`stablehlo-spmd-to-linalg` 输出的 rank-local `linalg` / `tensor` / `scf`
  local compute IR 和 `wafer.tensor_collective.*` tensor collective IR。
- current stage responsibility：建立 `wafer.group` candidate 边界，说明哪些 tensor SSA value、outs、
  producer/consumer 和 tensor collective 能进入 group 候选。
- output program / IR：可验证的 tensor-level `wafer.group` candidate IR。
- downstream consumer：R3.2 root tile feasibility、R3.3 tile_region materialization、R3.4/R3.5
  layout/SPM/DDR feasibility、R3.6/R3.7 ABI/package stages。
- user-level driver / named pipeline：由 `wafer-opt` program pipeline 重放 frontend/SPMD/R2.4 后进入
  group candidate gate；不能让 integration test 手动拼 raw StableHLO、tensor collective fixture 和
  group fixture 作为长期主线。
- explicit non-goals：不做 physical placement、tile shape search、SPM allocation、DTE schedule、
  `wafer.comm` materialization、C ABI 或 package emission。
- completion gate：真实 P2.S2/R2.4 program chain 的 local compute + tensor collective 输出能进入
  group candidate gate，且 verifier 证明 group 边界只包含 tensor-level IR；fixture 只做负例和局部覆盖。

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

推进 R3.1。输入必须来自 `stablehlo-spmd-to-linalg` 的真实 program chain：`linalg` / `tensor` /
`scf` local compute IR 和 `wafer.tensor_collective.*` tensor collective IR。不要把 raw StableHLO
collective、`wafer.comm`、SPM tile buffer、DTE token、Python helper 或手写 fixture 当作 group
主线输入。
