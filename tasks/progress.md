# Wafer Compiler Progress

更新时间：2026-06-03

本文件只记录当前看板、主线 pipeline、完成口径和下一步。设计细节、历史复盘、测试命令和长验证说明
放在对应 `tasks/` 设计文档、git commit 和测试里，不在这里重复。

## 状态标记

- `active`：当前优先推进。
- `ready`：前置边界已明确，可以开始。
- `pending`：依赖前序任务完成。
- `later`：当前主线之后再做。
- `done`：实现、测试和文档已按当前 pipeline contract 收口。

局部实现、verifier 负例、fixture 或单个 program gate 通过，不自动等于 `done`。若 design gate
仍缺关键语义，必须保留在 `active` / `pending`，并明确已落地和未完成的边界。

## 当前主线

用户级主入口统一为 `wafer-opt` program pipeline：

```text
PyTorch/XLA StableHLO Wafer program directory
  -> stablehlo-spmd
       verify program dir, default sharding seed, Shardy propagation,
       pinned XLA SPMD partition helper, parameter shard metadata/payload
  -> stablehlo-spmd-to-linalg
       post-SPMD StableHLO collective handoff + official StableHLO-to-Linalg
  -> stablehlo-spmd-to-group
       dependency-preserving logical wafer.group candidates
  -> R3.2+ planning / tile-region / placement / package recovery
```

`wafer-compile-stablehlo` 只做 frontend / StableHLO program verifier。
`wafer-propagate-stablehlo-sharding`、`wafer-lower-stablehlo-to-linalg` 和 R3.2a/b dump pass
只是内部或局部测试入口，不是用户级编译流程。

## 已完成链路

| ID | 状态 | 设计来源 | 输入 | 输出 / 完成 gate |
| --- | --- | --- | --- | --- |
| P2.F1 | done | `2026-05-25-wafer-frontend-stablehlo-program-design.md` | PyTorch/XLA model + parameter payload | StableHLO Wafer program directory；frontend 只导出 reference / pre-SPMD sharded program |
| P2.S1 | done | `2026-05-25-wafer-shardy-spmd-design.md` | P2.F1 program 或 textual StableHLO/SDY fixture | default input seed + Shardy propagation；作为 `stablehlo-spmd` 内部阶段 |
| P2.S2 | done | `2026-05-25-wafer-shardy-spmd-design.md` | P2.F1 sharded program | post-SPMD StableHLO program、rank-local signature、collective metadata、parameter shard metadata/payload |
| R2.4 | done | `2026-05-25-wafer-local-compute-normalization-design.md` | post-SPMD StableHLO program | Linalg/Tensor/SCF/Arith/Math local compute + verifier-legal `wafer.tensor_collective.*`；不生成 `wafer.comm` / SPM / ABI；旧本地 StableHLO compute lowering pass 已删除 |
| R3.1 | done | `2026-05-12-wafer-group-design.md` 2.1、9.1-9.5 | R2.4 rank-local structured tensor IR | verifier-legal logical `wafer.group`；覆盖 fill/init、matmul+bias+epilogue、simple elementwise、tensor collective handoff、多 group 和 raw StableHLO / lower-level op 禁止 |
| R3.2a | done | `2026-05-12-wafer-group-design.md` 10.1-10.2 | R3.1 logical `wafer.group` body | `GroupTilingDemand` analysis result；覆盖 boundary/result tile facts、per-op slice、iterator、accumulator/reduction dims、collective demand 和 unsupported-op failure |
| R3.2b | done | `2026-05-21-wafer-layout-materialization-design.md` 3.1.1 | R3.2a `GroupTilingDemand` facts + group SSA use-def | `GroupLayoutPlan` analysis result；覆盖 boundary layout、op layout constraints、broadcast relation、materialization cut/result demand 和 failure forwarding；不修改 IR |

## 当前状态

当前 active task 是 **R3.2c tile-region candidate conversion/pass 边界收口**。

R3.2c 可以消费 R3.1 logical group、R3.2a tiling demand 和 R3.2b layout plan。当前未完成点是：

- op coverage 表仍有限，需要明确 supported / unsupported / deferred。
- 当前 candidate emitter 是否应重构为 conversion pass 还未收口。
- R3.2c 未完成前，R3.2d SPM allocation 不能标 ready。

## 恢复队列

| ID | 状态 | 输入 / 输出边界 | 完成 gate |
| --- | --- | --- | --- |
| R3.2c | active | 输入：R3.1 group + R3.2a demand + R3.2b layout plan；输出：transformation-local provisional `wafer.tile_region` candidate | 明确 conversion/pass 边界和 op coverage；candidate 包含 target-abstract compute / load-store / layout materialization / tile-buffer / effect 结构；rejected candidate 不写主 IR |
| R3.2d | pending | 输入：R3.2c provisional `wafer.tile_region` candidate；输出：SPM allocation result 或结构化失败 | 真实 SPM window placement：alignment、layout padding、scratch/psum/temp、materialization temp、communication staging、lifetime overlap、range/end-address/conflict 都参与 |
| R3.2e | pending | 输入：R3.2c candidate + R3.2d SPM facts；输出：DDR/resource plan 和 compute/movement legality result | 覆盖 external/workspace/resident-constant、pool/domain/capacity/bandwidth/range demand；不能用 manifest fixture 替代 |
| R3.2f | pending | 输入：R3.1 group + R3.2a-e planning results；输出：accepted/rejected/split group planning decision | closed-loop 搜索 group boundary、traversal、tile shape、layout、SPM/DDR/resource plan；未接受 plan 不落 IR；multi-root packing 只作为可证明兼容时的可选策略 |
| R3.3 | pending | 输入：R3.2f accepted plan；输出：committed `wafer.tile_region` | 只 materialize accepted candidate；不重新决定 group 是否可行 |
| R3.4 | pending | 输入：R3.2f accepted layout/SPM facts + R3.3 tile-region；输出：materialized layout/SPM buffer/storage IR | 只落已接受的 layout assignment、materialization cut 和 SPM allocation facts |
| R3.5 | pending | 输入：R3.2f accepted DDR/resource facts + storage-aware tile IR；输出：materialized DDR resource / binding boundary | 只 materialize accepted external/workspace/resident-constant/resource demand |
| R3.6 | pending | 输入：storage/movement/compute IR；输出：`wafer.abi.*` issue sequence | issue sequence 从 IR 派生，只表达 ABI 参数单位和 wait policy |
| R3.7 | pending | 输入：ABI issue / launch signature IR；输出：IR-derived package manifest | manifest、C stub、launch signature 从 current lowering 输出导出，不使用 fixed manifest emitter |
| R3.8 | pending | 输入：R3.6/R3.7 输出；输出：wrapper-facing call contract 和 golden packet | 至少 RDMA/WDMA/GEMM 有真实 wrapper-facing call contract 和 packet 对照 |
| R4.1-R4.5 | pending | placement + local shard + launch/package metadata | placement、rank/block/coord、per-rank slices、writeback 和 placed package 从 lowering 输出导出 |
| R5.1-R5.2 | pending | static transformer local shard IR / staged IR gaps | full-block schedule 或拒绝原因；补 mask/select、dynamic-bound policy、non-constant-init reduce、constant/weight slice |
| R6.1-R6.2 | pending | tiled tensor collective / placement / buffer-slice facts | DTE resource/package metadata 和 `wafer.comm` materialization；移除临时 visible cast 路径 |
| P7/P8/P9 | later | P0-P6 恢复后的 package/runtime boundary | P0-P6 主链路恢复后再推进 LLVM/object、runtime adapter、board/profiling |

## 当前不做

- Serving integration。
- KV cache / paged attention / prefill-decode 调度。
- raw DTE non-unicast collective ABI。
- LLVM dialect / LLVM IR lowering、object emission 或真实 `wafer_*` runtime call emission。
- 自定义 LLVM backend 或 ISA intrinsic lowering。
- 以 importer、runtime path、workload shape 或 parameter 名称作为 IR 合同。

## 下一步

收口 R3.2c conversion coverage 和 pass 边界。R3.2c 完成后再进入 R3.2d，在 provisional
`wafer.tile_region` candidate 上恢复真实 SPM allocation。
