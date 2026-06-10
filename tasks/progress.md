# Wafer Compiler Progress

更新时间：2026-06-10

本文件记录当前看板、主线 pipeline、关键 IR 状态、完成口径和下一步。设计细节、实现复盘、测试命令
和长验证说明放在对应 `tasks/` 设计文档、git commit 和测试里；这里不写逐条 worklog。

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
       program verification, default sharding seed, Shardy propagation,
       Wafer-owned SPMD partition output, parameter shard metadata/payload
  -> stablehlo-spmd-to-linalg
       post-SPMD StableHLO collective handoff + official StableHLO-to-Linalg
  -> stablehlo-spmd-to-group
       dependency-preserving logical wafer.group
  -> wafer-lower-groups-to-tile-region / R3.2c
       memref-backed wafer.tile.region + DDR memref function boundary
  -> R3.2e candidate DDR tile-view materialization
       candidate traversal/tile shape -> DDR memref.subview tile operands
  -> wafer-lower-tile-region-to-instr / R3.2d
       target-abstract tile ops -> wafer.instr.* over unplaced Wafer-tagged memref
  -> R3.2f SPM placement
  -> R3.2g DDR/resource legality
  -> R3.2h closed-loop planner decision
  -> R3.3+ accepted materialization / launch / package
```

`wafer-compile-stablehlo` 只做 frontend / StableHLO program verifier。
`wafer-propagate-stablehlo-sharding`、`wafer-lower-stablehlo-to-linalg`、R3.2a/b dump pass、
`--wafer-convert-group-to-tile-region` 和 `--wafer-convert-tile-region-to-instr` 是内部或局部测试入口，
不是用户级编译流程。

## 当前 IR 快照

| 层 | 当前代码已落地 | 当前主线边界 |
| --- | --- | --- |
| Tensor | `wafer.group`、`wafer.group.yield`、`wafer.tensor.*` collective handoff | R3.1 已完成 logical group；tensor collective 到 tile communication 仍 deferred |
| Tile region | `wafer.tile.region`、`wafer.tile.yield` | R3.2c 已是 memref-backed target-abstract tile IR，支持 `scf.if` / `scf.for` 递归 lowering |
| Tile storage / boundary | `wafer.tile.load/store`、`memref.alloc`、`#wafer.memory<space, layout>` | DDR side 使用 `#wafer.memory<ddr, tensor>` memref；SPM side 使用 `#wafer.memory<spm, *>` memref；`!wafer.storage` / `wafer.tile.alloc` 已删除 |
| Tile compute / movement | `wafer.tile.fill/gemm/elementwise/reduce`、layout materialization、copy/slice/broadcast/transpose/reshape | 保持 target-abstract tile ops；layout / movement 语义由 memref type、Wafer memory attr 和 op interface 表达 |
| Instr | `wafer.instr.rdma/wdma/gather_scatter/fill/elementwise/reduce/convert/gemm/local_drain` | R3.2d 已能消费显式 DDR subview / strided view；不做 SPM offset、DDR allocation、ABI call |
| Resource / runtime | `wafer.placement.map`、`wafer.launch` | SPM/DDR policy 不新增单独 storage IR；accepted facts 后续 materialize 到 placed memref / descriptor / launch boundary |

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

## 当前边界

- R3.2c 只 materialize whole logical boundary 的 DDR memref 和 tile-local SPM memref；它不从
  candidate tile shape 自动切 temporal DDR tile view。
- R3.2d RDMA/WDMA 支持 strided DDR view 是 consumer 能力；descriptor 的 offset/stride 必须来自
  IR 中的 memref view，不能从名字、shape 或 whole-boundary memref 推断。
- `wafer-lower-groups-to-instr` 目前是 bring-up / explicit-view lowering 入口，不是完整 tiling
  planner completion proof。
- Tile communication 仍 deferred，等待 placement/local-rank/buffer facts 后再 lower 到 communication /
  DTE / local-drain 边界。

## 已完成主线

| ID | 状态 | 输入 | 输出 / 完成 gate |
| --- | --- | --- | --- |
| P2.F1-P2.S2 | done | PyTorch/XLA model 或 textual StableHLO/SDY fixture | StableHLO Wafer program、Shardy propagation、SPMD partition、rank-local metadata / parameter shard payload |
| R2.4 | done | post-SPMD StableHLO program | Linalg/Tensor/SCF/Arith/Math local compute + verifier-legal `wafer.tensor.*` handoff |
| R3.1 | done | R2.4 rank-local structured tensor IR | verifier-legal logical `wafer.group`；只表达 tensor-level grouping，不做 storage/runtime lowering |
| R3.2a | done | R3.1 logical `wafer.group` body | `GroupTilingDemand` analysis；覆盖 boundary/result tile facts、per-op slice、iterator、accumulator/reduction dims、collective demand 和 unsupported-op failure |
| R3.2b | done | R3.2a demand + group SSA use-def | `GroupLayoutPlan` analysis；覆盖 boundary layout、op layout constraints、broadcast relation、materialization cut/result demand 和 failure forwarding |
| R3.2c | done | R3.1 group + R3.2a/R3.2b facts | memref-backed `wafer.tile.region` + DDR memref function boundary；覆盖 supported compute/movement/view/control-flow；不做 SPM offset，不生成 candidate DDR tile subview |
| R3.2d | done | R3.2c target-abstract tile-region IR；DDR side 是 Wafer DDR memref，SPM side 是 unplaced Wafer SPM memref | instruction-level `wafer.instr.*`；覆盖 RDMA/WDMA/TDMA/CT/NE op contract、structured control-flow body legalization、static movement descriptor packing；不生成 ABI call |

## 后续队列

| ID | 状态 | 输入 | 输出 / 完成 gate |
| --- | --- | --- | --- |
| R3.2e | active | R3.1 group + R3.2a/R3.2b facts + candidate traversal/tile shape | planner scratch IR 中的 DDR `memref.subview` tile operands；R3.2d RDMA/WDMA descriptor 必须从 actual tile view 推导 |
| R3.2f | pending | R3.2e candidate tile-view IR 经 R3.2d legalization 后的 instruction-level IR | placed SPM memref / address facts；覆盖 alignment、range/end-address、bank span/conflict、control-flow lifetime |
| R3.2g | pending | R3.2f placed instruction IR + actual DDR tile-view facts | DDR/resource legality；覆盖 external/compiler-managed/resident-constant、pool/domain/capacity/bandwidth/range demand |
| R3.2h | pending | R3.1 group + R3.2a-g planning results | accepted / rejected / split group decision；未接受 plan 不落 IR |
| R3.3 | pending | R3.2h accepted plan | committed `wafer.tile.region` + accepted instruction-level lowering boundary |
| R3.4 | pending | R3.2h accepted layout/SPM facts + R3.3 tile-region | placed instruction-level IR / placed memref / access descriptor |
| R3.5 | pending | R3.2h accepted DDR/resource facts + memref-aware IR | materialized DDR resource / allocation boundary；不重新做 legality |
| R3.6-R3.8 | pending | placed instruction IR + launch signature | C ABI / packet emission、IR-derived package manifest、wrapper-facing golden packet |
| R4.1-R4.5 | pending | placement + local shard + launch/package metadata | rank/block/coord、per-rank slices、writeback、placed package |
| R5.1-R5.2 | pending | static transformer local shard IR / staged IR gaps | full-block schedule 或拒绝原因；补 mask/select、dynamic-bound policy、constant/weight slice 等 |
| R6.1-R6.2 | pending | tiled tensor collective + placement/local-rank/buffer facts | materialize tile communication 到 communication / DTE / local-drain 边界 |
| P7/P8/P9 | later | P0-P6 主链路输出 | LLVM/object、runtime adapter、board/profiling |

## 当前不做

- Serving integration、KV cache / paged attention / prefill-decode 调度。
- raw DTE non-unicast collective ABI。
- LLVM dialect / LLVM IR lowering、object emission 或真实 `wafer_*` runtime call emission。
- 自定义 LLVM backend 或 ISA intrinsic lowering。
- 以 importer、runtime path、workload shape 或 parameter 名称作为 IR 合同。

## 下一步

实现 R3.2e：

1. 从 candidate traversal / tile shape 和 R3.2a/R3.2b facts 推导 input/output tile 的
   `memref.subview` offset/sizes/strides。
2. 让 `wafer.tile.load/store` 读写 tile view，而不是 whole group boundary。
3. 用 R3.2d 验证 RDMA/WDMA descriptor 来自 actual DDR tile view。
