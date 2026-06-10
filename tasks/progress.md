# Wafer Compiler Progress

更新时间：2026-06-10

本文件只记录当前看板、主线 pipeline、完成口径和下一步。设计细节、实现复盘、测试命令和长验证说明
放在对应 `tasks/` 设计文档、git commit 和测试里。

## 状态标记

- `active`：当前优先推进。
- `ready`：前置边界已明确，可以开始。
- `pending`：依赖前序任务完成。
- `later`：当前主线之后再做。
- `done`：实现、测试和文档已按当前 pipeline contract 收口。

局部 pass、verifier 负例、fixture 或单个 dump gate 通过，不自动等于 `done`。主线完成证明必须能
重放已完成上游链路，并让当前 stage 输出被下游边界直接消费。

## 当前主线

用户级主入口统一为 `wafer-opt` program pipeline：

```text
PyTorch/XLA StableHLO Wafer program directory
  -> stablehlo-spmd
  -> stablehlo-spmd-to-linalg
  -> stablehlo-spmd-to-group
  -> wafer-lower-groups-to-tile-region / R3.2c
  -> R3.2e candidate DDR tile-view materialization
  -> wafer-lower-tile-region-to-instr / R3.2d
  -> R3.2f SPM placement
  -> R3.2g DDR/resource legality
  -> R3.2h closed-loop planner decision
  -> R3.3+ accepted materialization / launch / package
```

`wafer-compile-stablehlo` 只做 frontend / StableHLO program verifier。
`wafer-propagate-stablehlo-sharding`、`wafer-lower-stablehlo-to-linalg`、R3.2a/b dump pass、
`--wafer-convert-group-to-tile-region` 和 `--wafer-convert-tile-region-to-instr` 是内部或局部测试入口，
不是用户级编译流程。

## 当前 Active

**R3.2e candidate DDR tile-view materialization**

```text
Pipeline position:
- Upstream artifact / IR:
  R3.1 verifier-legal `wafer.group`、R3.2a tiling demand、R3.2b layout plan，以及
  candidate traversal / tile shape / boundary slice proposal。
- Current stage responsibility:
  在 planner scratch IR 中为 tile load/store 从 full DDR boundary memref 显式生成
  `memref.subview` / strided DDR tile operands。
- Output artifact / IR:
  R3.2d 可消费的 scratch `wafer.tile.region` / memref IR，其中 DDR operands 已表达 actual tile view；
  或结构化 failure reason。
- Downstream consumer:
  R3.2d instruction legalization / RDMA-WDMA descriptor lowering、R3.2f SPM placement、
  R3.2g DDR/resource legality、R3.2h closed-loop planner。
- Explicit non-goals:
  不分配 SPM offset，不做 DDR pool/resource planning，不 accept plan，不把 scratch IR commit 到主 IR。
- Completion gate:
  static affine boundary slice 和 simple tiled matmul/elementwise/storeback 能生成 DDR `memref.subview`；
  R3.2d RDMA/WDMA descriptor 从这些 view 推导 stride/iteration；tiled boundary 不能退回 whole-boundary
  DMA。
```

## 任务队列

| ID | 状态 | 边界 |
| --- | --- | --- |
| P2.F1-P2.S2 | done | StableHLO program export、Shardy propagation、SPMD partition、rank-local metadata / payload |
| R2.4 | done | post-SPMD StableHLO -> Linalg/Tensor/SCF/Arith/Math local compute + `wafer.tensor.*` handoff |
| R3.1 | done | verifier-legal logical `wafer.group`；只表达 tensor-level grouping，不做 storage/runtime lowering |
| R3.2a | done | `GroupTilingDemand` analysis；不修改 IR |
| R3.2b | done | `GroupLayoutPlan` analysis；不修改 IR |
| R3.2c | done | memref-backed `wafer.tile.region` + DDR memref function boundary；不从 candidate tile shape 生成 DDR tile subview |
| R3.2d | done | target-abstract tile ops -> `wafer.instr.*`；能消费显式 DDR subview / strided view，不做 SPM offset / ABI |
| R3.2e | active | candidate traversal / tile shape -> DDR `memref.subview` tile operands |
| R3.2f | pending | instruction-level IR with actual DDR tile views -> placed SPM memref / address facts |
| R3.2g | pending | placed instruction IR + DDR tile-view facts -> DDR/resource legality |
| R3.2h | pending | closed-loop accept / reject / split group decision |
| R3.3 | pending | 只 materialize R3.2h accepted plan 为 committed `wafer.tile.region` |
| R3.4-R3.5 | pending | materialize accepted layout/SPM facts and DDR/resource boundary |
| R3.6-R3.8 | pending | placed instruction IR -> C ABI / packet emission / package manifest / golden packet |
| R4.1-R4.5 | pending | placement、rank/block/coord、per-rank slices、writeback、placed package |
| R5.1-R5.2 | pending | static transformer local shard gaps and full-block schedule / rejection reason |
| R6.1-R6.2 | pending | tile communication materialization after placement/local-rank/buffer facts |
| P7/P8/P9 | later | LLVM/object/runtime adapter/board profiling after P0-P6 main chain |

## 边界提醒

- R3.2d RDMA/WDMA 支持 strided DDR view 是 consumer 能力；R3.2e 才是从 tiling candidate 产出
  DDR `memref.subview` 的 producer。
- `wafer-lower-groups-to-instr` 目前是 bring-up / explicit-view lowering 入口，不是完整 tiling
  planner completion proof。
- Tile communication 仍 deferred，等待 placement/local-rank/buffer facts 后再 lower 到 communication /
  DTE / local-drain 边界。

## 下一步

实现 R3.2e：

1. 从 candidate traversal / tile shape 和 R3.2a/R3.2b facts 推导 input/output tile 的
   `memref.subview` offset/sizes/strides。
2. 让 `wafer.tile.load/store` 读写 tile view，而不是 whole group boundary。
3. 用 R3.2d 验证 RDMA/WDMA descriptor 来自 actual DDR tile view。
