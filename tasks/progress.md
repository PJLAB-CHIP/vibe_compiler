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
       SPM planned ranges and lifetimes on instruction-level candidate IR
  -> R3.2g DDR memory planning
       DDR external access validation + compiler-managed/resident/inter-group DDR planned ranges
  -> R3.2h closed-loop candidate decision
       enumerate candidates, rerun R3.2e-g, choose passing candidate or split
  -> R3.3 committed materialization
       commit only the accepted candidate into main IR
  -> R3.4 placed instruction-level realization
       placed memref / access descriptor realization from accepted SPM facts and DDR planned ranges
  -> R3.5 launch/runtime DDR materialization
       runtime allocation/import/query/package metadata from accepted DDR plan
  -> R3.6+ C ABI / packet / object / runtime adapter
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
| Instr | `wafer.instr.rdma/wdma/gather_scatter/fill/elementwise/reduce/convert/gemm/local_drain` | R3.2d 已能消费显式 DDR subview / strided view；不做 SPM offset、DDR planned range、ABI call |
| Resource / runtime | `wafer.placement.map`、`wafer.launch` | SPM/DDR planning 不新增单独 storage IR；planned facts 后续 materialize 到 placed memref / descriptor / launch boundary |

## 当前 Active

**R3.2g DDR memory planning**

```text
Pipeline position:
- Upstream artifact / IR:
  R3.2f memory-planned instruction-level candidate IR。SPM side 已有 planned SPM range；
  DDR side 已由 R3.2e/R3.2d 显式表达 external DDR view、`memref.subview` / static strided view、
  RDMA/WDMA descriptor，以及后续需要扩 IR/interface 表达的 compiler-managed/resident/inter-group
  DDR requirement。
- Current stage responsibility:
  从当前 candidate IR 重算 DDR demand；验证 external DDR descriptor 和 view/root byte range；
  为 compiler-managed workspace、resident constant、inter-group DDR temporary 等非 external demand
  在 default DDR arena 内规划 symbolic range/offset/size/alignment 和 lifetime/reuse；验证 capacity、
  largest-contiguous、range overlap、alignment、bandwidth 和 descriptor 对 planned range 的覆盖。
- Output artifact / IR:
  同一 candidate artifact，带显式 DDR planned range / requirement facts；或结构化 rejected reason。
  成功事实必须能被 R3.2h/R3.3/R3.4/R3.5 直接消费，不能只存在 pass-local side table。
- Downstream consumer:
  R3.2h closed-loop candidate decision 使用 R3.2g 成功/失败选择 candidate；
  R3.3 只 commit 已通过 R3.2e-g 的 candidate；
  R3.4 把 accepted DDR planned ranges realize 成 placed memref/access descriptor；
  R3.5 只做 runtime allocation/import/query/package materialization，不重新规划 DDR。
- User-level driver / named pipeline:
  局部 pass 仍是 `wafer-plan-ddr-memory`；
  主线验证入口继续使用从 R3.2c/R3.2d/R3.2f 到 R3.2g 的 named pipeline，但 completion proof
  必须覆盖实际 DDR planned range，不再只覆盖 external DDR validation。
- Explicit non-goals:
  不重新推 DDR tile subview，不重做 SPM memory planning，不选择 tile shape/layout/group boundary，
  不生成 ABI call、packet、physical DDR address 或 runtime handle。
- Completion gate:
  simple matmul/elementwise candidate 中 external input/output DDR views、compiler-managed DDR temporary
  或 resident constant demand 都能获得可验证 DDR planned range；非法 dynamic view、payload/range、
  overlap/capacity/largest-contiguous/bandwidth/alignment failure 能结构化拒绝并返回 R3.2h repair。
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
  bank_limit>` planning fact；arena 作用域是单个 `wafer.tile.region`，默认普通 SPM range 为
  `[0x10000, 0x2F0000)`，size 来自 `computeWaferPhysicalTensorInfo` 的 physical bytes。V0 已对
  instruction-level IR 建立 region-aware lifetime dataflow：straight-line last-use 后复用、
  `scf.if` path-sensitive branch 互斥复用、`scf.for` loop-carried/backedge lifetime、loop 后 temp
  复用，以及 async token 到 wait/drain 的 lifetime 延伸都由 IR 结构重算；offset 搜索采用
  pressure-weighted offline packing。
- R3.2g 当前代码已有 `wafer-plan-ddr-memory` 的 external DDR validation slice：它从 RDMA/WDMA 的
  DDR operand 通过 tile-region block arg、`memref.subview` / view-like chain 回到 root DDR memref，
  验证 descriptor payload、view/root byte range、default DDR arena capacity、largest-contiguous 和
  bandwidth demand，并拒绝 dynamic view / payload/range/resource failure。这个 slice 不是完整
  DDR memory planning completion，因为还没有 compiler-managed/resident/inter-group DDR planned range。
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
| R3.2f | done | R3.2e candidate tile-view IR 经 R3.2d legalization 后的 instruction-level IR | `wafer.spm.offset` planning fact 标注 SPM `memref.alloc`；simple tiled elementwise/matmul/storeback 获得非重叠、256B 对齐、range/end 合法 memory plan；Cx/NCx 使用 physical bytes；straight-line、structured `scf.if` / `scf.for` 和 async token wait 的 lifetime/reuse 由 IR dataflow 重算；pressure-weighted offline packing 避免 alloc-event first-fit 碎片化；capacity/alignment/range failure 结构化诊断 |

## 后续队列

| ID | 状态 | 输入 | 输出 / 完成 gate |
| --- | --- | --- | --- |
| R3.2g | active | R3.2f memory-planned instruction IR + actual DDR tile-view facts + explicit DDR requirements | compiler-side DDR memory planning；external DDR validation、compiler-managed/resident/inter-group DDR planned ranges、lifetime/reuse、capacity/largest-contiguous/bandwidth/alignment/range diagnostics；主线 gate 跑通带 DDR planned range 的 simple candidate |
| R3.2h | pending | R3.1 group + R3.2a/b facts + R3.2e-g gates | closed-loop candidate decision；枚举 tile/layout/resource 候选，逐个运行 R3.2e/R3.2d/R3.2f/R3.2g，输出 passing candidate artifact 或 split/retry failure；不重新发明 memory planning |
| R3.3 | pending | R3.2h passing candidate artifact | committed `wafer.tile.region` + instruction-level lowering boundary；只提交已通过 R3.2e-g gates 的候选 |
| R3.4 | pending | R3.3 committed tile-region + planned layout/SPM/DDR facts | placed instruction-level IR / placed memref / access descriptor；不重新决定 tile/layout/memory plan |
| R3.5 | pending | R3.4 placed/memref-aware IR + accepted DDR planned ranges | runtime allocation/import/query/package materialization；验证 runtime object 满足 accepted plan，不重新做 DDR planning |
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

实现 R3.2g 的真实 DDR memory planning：

1. 扩 IR/interface 表达 compiler-managed/resident/inter-group DDR requirement 和 accepted DDR planned range。
2. 从 instruction-level candidate IR 重算 DDR lifetime/demand，对 default DDR arena 做 symbolic range planning 和 reuse。
3. 把现有 descriptor/view/root/resource validation 接到 planned ranges 上，覆盖 overlap、capacity、
   largest-contiguous、bandwidth、alignment 和 dynamic/unsupported failure。
4. 更新 named pipeline completion proof：不能只证明 external DDR view validation，必须证明 R3.2g 输出能被
   R3.2h/R3.3/R3.4/R3.5 直接消费。
