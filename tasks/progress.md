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

局部实现、verifier 负例、fixture 或单个 program gate 通过，不自动等于 `done`。若 design gate
仍缺关键语义，必须保留在 `active` / `pending`，并明确已落地和未完成的边界。

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
  -> wafer-opt --program-pipeline=stablehlo-spmd-to-group
       same SPMD and R2.4 lowering stages
       form dependency-preserving logical wafer.group candidates
       verify multi-op tensor-level group body legality
  -> R3.2+ tiling / placement / package recovery
```

当前已完成链路的用户级 command 不传 helper path：

```bash
wafer-opt \
  --program-pipeline=stablehlo-spmd-to-linalg \
  --input-program-dir <input.program> \
  --output-program-dir <output.program> \
  --default-tile-count=16
```

R3.1 用户级入口：

```bash
wafer-opt \
  --program-pipeline=stablehlo-spmd-to-group \
  --input-program-dir <input.program> \
  --output-program-dir <output.program> \
  --default-tile-count=16
```

## 已完成链路

| ID | 状态 | 设计来源 | 输入 | 输出 / 完成 gate |
| --- | --- | --- | --- | --- |
| P2.F1 | done | `2026-05-25-wafer-frontend-stablehlo-program-design.md` | PyTorch/XLA model + parameter payload | StableHLO Wafer program directory；frontend 只导出 reference / pre-SPMD sharded program，不做 Shardy/SPMD |
| P2.S1 | done | `2026-05-25-wafer-shardy-spmd-design.md` | P2.F1 program 或 textual StableHLO/SDY fixture | default input seed + Shardy propagation；作为 `stablehlo-spmd` 内部阶段，不产出 partitioned local body |
| P2.S2 | done | `2026-05-25-wafer-shardy-spmd-design.md` | P2.F1 sharded program | `stablehlo-spmd` 产出 post-SPMD StableHLO program、rank-local signature、collective metadata、parameter shard metadata/payload |
| R2.4 | done | `2026-05-25-wafer-local-compute-normalization-design.md` | post-SPMD StableHLO program | `stablehlo-spmd-to-linalg` 产出 Linalg/Tensor/SCF local compute 和 `wafer.tensor_collective.*`；不生成 `wafer.comm` |
| R3.1 | done | `2026-05-12-wafer-group-design.md` 2.1、9.1-9.5 | `stablehlo-spmd-to-linalg` 输出的 rank-local tensor IR | `stablehlo-spmd-to-group` 形成 verifier-legal logical `wafer.group` candidate；支持 dependency-preserving conservative expansion、fill/init、bias/epilogue/simple elementwise、tensor collective handoff 和显式 `ins` / `outs` / `group_yield` 边界 |

## 当前状态

当前没有 active 实现任务。R3.2a op tiling demand analysis 已完成；下一项是 R3.2b layout
planning，消费 R3.2a 的 tile value graph / op layout constraints，恢复 layout assignment、
materialization cut 和 materialization buffer demand。

## 恢复队列

| ID | 状态 | 设计来源 | 输入 / 输出边界 | 完成 gate |
| --- | --- | --- | --- | --- |
| R3.2a | done | `2026-05-12-wafer-group-design.md` 10.1-10.2；MLIR Linalg/TilingInterface；tensor collective docs | 输入：R3.1 logical `wafer.group` body；输出：target-abstract tile plan candidate 和 per-op tile demand | `GroupTilingDemand` analysis + `--wafer-dump-group-tiling-demand` debug gate 已恢复；覆盖 linalg indexing maps、reduction accumulator、tensor collective interface、多 group 和 unsupported op failure；不写 `wafer.group` attr |
| R3.2b | ready | `2026-05-21-wafer-layout-materialization-design.md` | 输入：R3.2a tile value graph 和 op layout constraints；输出：layout assignment、materialization cut 和 materialization buffer demand | 给出可验证 layout assignment、materialization cut 和 materialization buffer demand；失败返回结构化原因；不把 layout plan 写进 `wafer.group` attr |
| R3.2c | pending | `2026-05-21-wafer-spm-bufferization-design.md` | 输入：R3.2a/b 的 tile-local buffer demand、layout、lifetime 和 effect；输出：SPM allocation | 做真实 SPM window placement：alignment、layout padding、scratch/psum/temp、lifetime overlap、range/end-address/conflict 都参与；不是 byte-size estimate；失败返回 planner |
| R3.2d | pending | `2026-05-25-wafer-ddr-resource-allocation-design.md`；compute/movement docs | 输入：R3.2a/b 的 boundary、load/store、constant、workspace、movement 和 compute legality demand；输出：DDR/resource plan 和 compute/movement legality result | 覆盖 external/workspace/resident-constant、pool/domain/capacity/bandwidth/range demand；compute/movement/collective 的 layout、dtype、shape、effect 通过接口/verifier 验证；不能用 manifest fixture 替代 |
| R3.2e | pending | `2026-05-12-wafer-group-design.md` 10.x；R3.2a-d planning / analysis results | 输入：R3.1 logical group 和 R3.2a-d planning results；输出：accepted/rejected/split group planning decision | closed-loop 搜索 group boundary、traversal、tile shape、layout、SPM/DDR/resource plan；accepted plan 带 R3.3 可消费的 traversal/tile/slice/layout/SPM/DDR facts；未接受 plan 不落 IR；可选 multi-root packing 只有在 legality/resource/cost 证明兼容时接受 |
| R3.3 | pending | `2026-05-25-wafer-tile-region-design.md` | 输入：R3.2e accepted plan；输出：`wafer.tile_region` | 只 materialize accepted plan；未接受 group 不产生 `wafer.tile_region` |
| R3.4 | pending | `2026-05-21-wafer-layout-materialization-design.md`；`2026-05-21-wafer-spm-bufferization-design.md` | 输入：R3.2e accepted layout/SPM facts 和 R3.3 `wafer.tile_region`；输出：materialized layout/SPM buffer/storage IR | 只把 R3.2e accepted plan 中的 layout assignment、materialization cut、SPM allocation facts 落到可验证 IR；不在 R3.4 第一次决定 group 是否可行 |
| R3.5 | pending | `2026-05-25-wafer-ddr-resource-allocation-design.md` | 输入：R3.2e accepted DDR/resource facts 和 storage-aware tile IR；输出：materialized DDR resource / binding boundary | 只 materialize R3.2e accepted plan 中的 external/workspace/resident-constant/resource demand；覆盖 pool/domain/capacity/bandwidth，不能用 manifest fixture 替代 |
| R3.6 | pending | `2026-05-25-wafer-c-abi-golden-packet-design.md` | 输入：storage/movement/compute IR；输出：`wafer.abi.*` issue sequence | issue sequence 从 IR 派生，只表达 ABI 参数单位和 wait policy |
| R3.7 | pending | `2026-05-25-wafer-launch-runtime-package-design.md` | 输入：ABI issue / launch signature IR；输出：IR-derived package manifest | manifest、C stub、launch signature 从当前 `wafer-opt` 输出导出，不使用 fixed manifest emitter |
| R3.8 | pending | `2026-05-25-wafer-c-abi-golden-packet-design.md`；launch/runtime package design | 输入：R3.6/R3.7 输出；输出：wrapper-facing call contract 和 golden packet | 至少 RDMA/WDMA/GEMM 有真实 wrapper-facing call contract 和 packet 对照 |
| R4.1 | pending | `2026-05-25-wafer-placement-design.md` | 输入：logical rank / complete group candidate；输出：accepted physical placement | placement 来自 logical rank、good-tile/PG capability 和 coordinate verifier |
| R4.2 | pending | placement design；launch/runtime package design | 输入：placement + local shard metadata；输出：rank/block/coord launch metadata | logical rank、block id、physical coord 接到 tile_region/launch/package 边界 |
| R4.3 | pending | placement design；frontend/SPMD program design | 输入：rank-local signature + shard metadata；输出：per-rank input/output slices | multi-tile no-comm materialize local slice，不 clone whole tensor |
| R4.4 | pending | placement design；launch/runtime package design | 输入：per-rank output slices；输出：writeback / host readback / merge responsibility | sharded output 的 IR/package 责任明确 |
| R4.5 | pending | placement design；launch/runtime package design | 输入：placement + local shard + issue sequence；输出：placed package program | placement metadata、local shards、launch args 从 lowering 输出导出 |
| R5.1 | pending | verification plan；frontend/local compute/package docs | 输入：static transformer local shard IR；输出：accepted schedule 或拒绝原因 | workspace/resident constants/ABI issue sequence 来自 full-block IR dataflow |
| R5.2 | pending | local compute normalization design；verification plan | 输入：transformer staged IR gaps；输出：补齐 compute/package consistency | 覆盖 mask/select、dynamic-bound policy、non-constant-init reduce、constant/weight slice |
| R6.1 | pending | `2026-05-25-wafer-communication-dialect-design.md` | 输入：tiled tensor collective / placement facts；输出：DTE resource and package/runtime metadata | DTE allocation、collective buffer slice/address offset、metadata 可验证 |
| R6.2 | pending | communication design；layout/SPM/materialization docs | 输入：buffer-slice/layout/materialized collective；输出：`wafer.comm` | 移除临时 visible cast，由可验证 buffer-slice/materialization 路径承接 |
| P7/P8/P9 | later | architecture / verification / launch / C ABI docs | 输入：P0-P6 恢复后的 package/runtime boundary | P0-P6 主链路恢复后再推进 LLVM/object、runtime adapter、board/profiling |

R3.2e 明确包含一个可选子目标：multi-root packing / co-scheduling。它的输入不是 raw op list，
而是多个已由 R3.1 形成的 dependency-connected logical groups；只有当 traversal domain、
tile shape、layout、SPM/DDR/resource 和 cost 都可证明兼容时，R3.2e 才能把它们作为一个
co-scheduled candidate 接受。仅因为同 block、同 shape 或语法上可放进同一个 region，不能合并。
若 legality/resource/cost 不成立，保持多个 groups 或按 R3.2e split/reject 规则处理。

## 当前不做

- Serving integration。
- KV cache / paged attention / prefill-decode 调度。
- raw DTE non-unicast collective ABI。
- LLVM dialect / LLVM IR lowering、object emission 或真实 `wafer_*` runtime call emission。
- 自定义 LLVM backend 或 ISA intrinsic lowering。
- 以 importer、runtime path、workload shape 或 parameter 名称作为 IR 合同。

## 下一步

从 R3.2b 开始：消费 R3.2a 的 tile value graph 和 op layout constraints，恢复 layout assignment、
materialization cut 和 materialization buffer demand。完整 R3.2e closed-loop planner 只有在
layout planning、SPM allocation、DDR/resource planning、compute/movement legality analysis
都能真实参与 decision 后才能进入 `ready`。
