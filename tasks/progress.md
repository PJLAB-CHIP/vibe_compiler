# Wafer Compiler Progress

更新时间：2026-06-04

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
| R3.2c | done | `2026-05-25-wafer-tile-region-design.md` 2.1-2.2 | R3.1 group + R3.2a demand + R3.2b layout plan | MLIR DialectConversion 驱动的 scratch full conversion；输出 provisional `wafer.tile_region` candidate，覆盖 load/store、`wafer.alloc_tile`、layout materialization、`wafer.compute.fill/gemm/elementwise/reduce`、`wafer.move.*` 和 `wafer.view.reshape`；硬件 V0 无承载或缺 placement/local-rank facts 时结构化 failure；不写主 IR、不做 SPM offset |

## 当前状态

当前 active task 是 **R3.2d Wafer instruction legalization / selection on provisional tile-region
candidate**。

R3.2d 可以消费 R3.2c 的 provisional `wafer.tile_region` candidate。当前边界是：

- candidate 中的 `!wafer.tile_buffer`、`wafer.alloc_tile`、`wafer.load_tile`、`wafer.store_tile`、
  `wafer.layout.materialize`、`wafer.compute.*`、`wafer.move.*` 和 `wafer.view.reshape` 已能表达
  target-abstract layout/resource/view 关系，但还不是 SPM allocation 的直接输入。
- R3.2d 只做 Wafer instruction legalization / selection：把 target-abstract op 合法化并选择成
  instruction-level `wafer.instr.*` IR，同时显式生成 unplaced `wafer.storage.*` SSA value；
  不分配 SPM offset，不生成 C ABI call，不重新决定 group formation。
- tensor collective 到 `wafer.comm.*` 仍 deferred，等待 placement/local-rank/buffer facts。

## 恢复队列

| ID | 状态 | 输入 / 输出边界 | 完成 gate |
| --- | --- | --- | --- |
| R3.2d | active | 输入：R3.2c target-abstract `wafer.tile_region` candidate；输出：instruction-level `wafer.instr.*` IR + unplaced `wafer.storage.*` 或结构化失败 | 为 load/store、layout materialize、compute.gemm/reduce/elementwise、move.*、view alias 选择 CT/NE/TDMA/RDMA/WDMA 等硬件指令形态；IR 显式暴露 storage values、queue family、read/write/issue effects、temp/psum/staging、alias/view、storage size policy 和 reject reason；不做 SPM offset、不生成 ABI call |
| R3.2e | pending | 输入：R3.2d instruction-level IR with unplaced storage；输出：同一 instruction-level IR with placed SPM storage 或结构化失败 | 真实 SPM window placement：alignment、layout padding、scratch/psum/temp、materialization temp、communication staging、lifetime overlap、range/end-address/bank span/conflict 都参与 |
| R3.2f | pending | 输入：R3.2e placed instruction-level IR + DDR boundary facts；输出：DDR/resource legality result 和 movement/compute resource validation | 覆盖 external/workspace/resident-constant、pool/domain/capacity/bandwidth/range demand；不能用 manifest fixture 替代；不能回头改变 instruction semantics |
| R3.2g | pending | 输入：R3.1 group + R3.2a-f planning results；输出：accepted/rejected/split group planning decision | closed-loop 搜索 group boundary、traversal、tile shape、layout、instruction selection、SPM/DDR/resource plan；未接受 plan 不落 IR；multi-root packing 只作为可证明兼容时的可选策略 |
| R3.3 | pending | 输入：R3.2g accepted plan；输出：committed `wafer.tile_region` + accepted instruction-level lowering boundary | 只 materialize accepted candidate；不重新决定 group 是否可行 |
| R3.4 | pending | 输入：R3.2g accepted layout/SPM facts + R3.3 tile-region；输出：materialized placed instruction/storage IR | 只落已接受的 layout assignment、materialization cut、instruction storage graph 和 SPM placement facts |
| R3.5 | pending | 输入：R3.2g accepted DDR/resource facts + storage-aware tile IR；输出：materialized DDR resource / binding boundary | 只 materialize accepted external/workspace/resident-constant/resource demand |
| R3.6 | pending | 输入：placed instruction-level IR + launch signature；输出：C ABI call / packet emission 或 debug dump | codegen/emission 从 placed `wafer.instr.*` 派生 ABI 参数单位和 wait policy；`wafer.abi.*` 如保留只作为调试/测试 dump，不是主线 IR 层 |
| R3.7 | pending | 输入：R3.6 emitted call/packet metadata + launch signature；输出：IR-derived package manifest | manifest、C stub、launch signature 从 current lowering 输出导出，不使用 fixed manifest emitter |
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

进入 R3.2d，在 R3.2c provisional `wafer.tile_region` candidate 上恢复 Wafer instruction
legalization / selection，产出 instruction-level `wafer.instr.*` IR 和 unplaced `wafer.storage.*`。
