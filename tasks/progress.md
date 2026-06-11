# Wafer Compiler Progress

更新时间：2026-06-11

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
       candidate traversal/tile shape/boundary slice proposal -> DDR memref.subview tile operands
  -> wafer-lower-tile-region-to-instr / R3.2d
       target-abstract tile ops -> wafer.instr.* over unplaced Wafer-tagged memref
  -> wafer-lower-groups-to-memory-planned-instr / R3.2f SPM memory planning
  -> R3.2g DDR memory plan acceptance
       accept/reject explicit DDR views, descriptors, ownership and resource bounds
  -> R3.2h closed-loop candidate driver
       enumerate candidates, rerun R3.2e-g, choose accepted candidate or split
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
| Tile storage / boundary | `wafer.tile.load/store`、`memref.alloc`、`memref.subview`、`#wafer.memory<space, layout>` | DDR side 使用 `#wafer.memory<ddr, tensor>` memref 或显式 strided tile view；SPM side 使用 `#wafer.memory<spm, *>` memref；`!wafer.storage` / `wafer.tile.alloc` 已删除 |
| Tile compute / movement | `wafer.tile.fill/gemm/elementwise/reduce`、layout materialization、copy/slice/broadcast/transpose/reshape | 保持 target-abstract tile ops；layout / movement 语义由 memref type、Wafer memory attr 和 op interface 表达 |
| Instr | `wafer.instr.rdma/wdma/gather_scatter/fill/elementwise/reduce/convert/gemm/local_drain` | R3.2d 已能消费显式 DDR subview / strided view；不做 SPM offset、DDR allocation、ABI call |
| Resource / runtime | `wafer.placement.map`、`wafer.launch` | SPM/DDR policy 不新增单独 storage IR；accepted facts 后续 materialize 到 placed memref / descriptor / launch boundary |

## 当前 Active

**R3.2g DDR memory plan acceptance**

```text
Pipeline position:
- Upstream artifact / IR:
  R3.2f memory-planned instruction-level `wafer.tile.region` / `wafer.instr.*` IR，其中 SPM `memref.alloc`
  已带 `wafer.spm.offset` accepted fact，DDR tile access 来自 actual `memref.subview` / strided view。
- Current stage responsibility:
  接受或拒绝当前 instruction-level IR 已显式表达的 DDR memory plan：external /
  compiler-managed / resident DDR ownership、DDR range/domain/pool、descriptor 可表达性、
  bandwidth/capacity/resource pressure。所有结论必须从 SSA use-def、memref type/layout、
  `memref.subview` / strided view 和 `wafer.instr.*` descriptor 重算，不能从名字、whole-boundary
  shape 或 pass side table 恢复。
- Output artifact / IR:
  same memory-planned instruction-level IR accepted as DDR-safe，或结构化 failure reason。成功时不额外复制
  一份 `wafer.ddr.plan` / hidden allocation attr；accepted DDR facts 就是当前 IR 中显式存在的
  memref view、descriptor、ownership/policy 和后续可 materialize 的 allocation requirement。
- Downstream consumer:
  R3.2h closed-loop candidate driver 以 R3.2g 成功/失败作为候选合法性 gate；R3.3 accepted
  materialization 和 R3.5 DDR allocation materialization 消费这份已接受的显式 IR/facts。
- Explicit non-goals:
  不重新做 SPM memory planning，不重新推 DDR tile subview，不生成 ABI call，不在 IR 中保存 planner
  搜索 trace、失败候选或重复的 DDR plan attr，不把无法由当前 IR 表达的 compiler-managed DDR offset
  偷藏到 side table。
- Completion gate:
  R3.2f 输出的 simple tiled matmul/elementwise/storeback 能完成 DDR descriptor/range/memory planning
  acceptance；非法 DDR view、pool/domain/capacity/bandwidth 或 descriptor 边界能结构化失败。若某类
  compiler-managed / resident DDR fact 在当前 IR/interface 中不可表达，R3.2g 必须结构化失败或先扩
  IR/interface，不能把它推给后续阶段补协议。
```

## 当前边界

- R3.2c materialize whole logical boundary 的 DDR memref handle；full tensor use 才 lazy
  `wafer.tile.load` 到 SPM。R3.2e 已接入第一批 producer：external boundary 上的 static
  `tensor.extract_slice` 生成 DDR `memref.subview` + tile load，direct output `tensor.insert_slice`
  storeback 生成 DDR `memref.subview` + tile store。
- R3.2e 已接入 candidate evaluation producer：`--wafer-dump-candidate-ddr-tile-views` 接收 candidate
  output tile offsets/sizes，在 candidate evaluation clone 中通过 linalg indexing maps 生成 boundary
  `tensor.extract_slice` / output `tensor.insert_slice` proposal，再 materialize 为 DDR `memref.subview`。
  当前覆盖单结果 destination-style linalg root 的 simple matmul/elementwise；closed-loop traversal /
  tile-shape search 仍归 R3.2h。
- R3.2d RDMA/WDMA 支持 strided DDR view 是 consumer 能力；descriptor 的 offset/stride 必须来自
  IR 中的 memref view，不能从名字、shape 或 whole-boundary memref 推断。
- R3.2f 已接入 `wafer.spm.offset = #wafer.spm_offset<offset, size, alignment, bank_begin,
  bank_limit>` accepted fact；arena 作用域是单个 `wafer.tile.region`，默认普通 SPM range 为
  `[0x10000, 0x2F0000)`，size 来自 `computeWaferPhysicalTensorInfo` 的 physical bytes。V0 已对
  instruction-level IR 建立 region-aware lifetime dataflow：straight-line last-use 后复用、
  `scf.if` path-sensitive branch 互斥复用、`scf.for` loop-carried/backedge lifetime、loop 后 temp
  复用，以及 async token 到 wait/drain 的 lifetime 延伸都由 IR 结构重算；offset 搜索采用
  pressure-weighted offline packing，优先放置 physical size / conflict pressure / lifetime span 更大的
  demand，再在合法 gap 中选最低 offset，避免 alloc-event first-fit 的碎片化失败。
- `wafer-lower-groups-to-instr` 目前是 bring-up / explicit static-view lowering 入口，不是完整
  closed-loop tiling planner completion proof。
- `wafer-lower-groups-to-memory-planned-instr` 是 R3.2f 当前可重放入口：它在 R3.2c/R3.2d 后追加 SPM memory planning，
  证明 instruction-level SPM fact 能被后续 resource/planner 边界消费。
- Tile communication 仍 deferred，等待 memory planning/local-rank/buffer facts 后再 lower 到 communication /
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
| R3.2e.a | done | R3.1 group + explicit static boundary slice facts + R3.2c/R3.2d lowering chain | external boundary `tensor.extract_slice` 和 direct output `tensor.insert_slice` materialize 为 DDR `memref.subview` tile operands；simple tiled elementwise/matmul/storeback 的 RDMA/WDMA descriptor 来自 actual tile view，不退回 whole-boundary DMA |
| R3.2e.b | done | R3.1 group + candidate output tile offsets/sizes + R3.2a/R3.2b facts | planner candidate evaluation 中通过 linalg indexing maps 生成 boundary slice proposal；simple full-tensor matmul group 生成 input/output DDR `memref.subview` tile operands，不写回主 IR |
| R3.2f | done | R3.2e candidate tile-view IR 经 R3.2d legalization 后的 instruction-level IR | `wafer.spm.offset` accepted fact 标注 SPM `memref.alloc`；simple tiled elementwise/matmul/storeback 获得非重叠、256B 对齐、range/end 合法 memory plan；Cx/NCx 使用 physical bytes；straight-line、structured `scf.if` / `scf.for` 和 async token wait 的 lifetime/reuse 由 IR dataflow 重算；pressure-weighted offline packing 避免 alloc-event first-fit 碎片化；capacity/alignment/range failure 结构化诊断 |

## 后续队列

| ID | 状态 | 输入 | 输出 / 完成 gate |
| --- | --- | --- | --- |
| R3.2g | active | R3.2f memory-planned instruction IR + actual DDR tile-view facts | DDR memory plan acceptance；同一 IR 通过 descriptor/range/ownership/pool/domain/capacity/bandwidth gate，或结构化失败 |
| R3.2h | pending | R3.1 group + R3.2a/b facts + R3.2e-g acceptance pipeline | closed-loop candidate driver；枚举 tile/layout/resource 候选，逐个运行 R3.2e/R3.2d/R3.2f/R3.2g，输出 accepted candidate artifact 或 split/retry failure；不重新发明 memory planning |
| R3.3 | pending | R3.2h accepted candidate artifact | committed `wafer.tile.region` + accepted instruction-level lowering boundary；只提交已通过 R3.2e-g gates 的候选 |
| R3.4 | pending | R3.3 committed tile-region + accepted layout/SPM/DDR facts | placed instruction-level IR / placed memref / access descriptor；不重新决定 tile/layout/memory plan |
| R3.5 | pending | R3.4 placed/memref-aware IR + accepted DDR requirements | materialized DDR allocation/import/package boundary；BO alloc/import/query metadata and runtime validation，不重新做 memory planning |
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

实现 R3.2g：

1. 定义 DDR memory plan acceptance 的 IR-derived input 和 failure reason，避免从 whole-boundary
   shape、名字或 side table 恢复 DDR tile view / ownership 语义。
2. 基于 R3.2f memory-planned instruction IR 检查 RDMA/WDMA/TDMA descriptor、DDR memref view range、
   pool/domain ownership 和 bandwidth/capacity demand；成功即接受当前 IR 中显式表达的 DDR plan，
   失败返回结构化原因。
3. 增加 named pipeline，使 tiled matmul/elementwise/storeback 从 R3.2c/R3.2d/R3.2f 直接跑到 R3.2g
   acceptance；证明后续 R3.2h 只需调度/重试候选，不需要再补 DDR planning 语义。
