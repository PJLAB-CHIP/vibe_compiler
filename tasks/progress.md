# Wafer Compiler Progress

更新时间：2026-06-16

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
  -> wafer-lower-groups-to-tile-region / tile-region materialization
       memref-backed wafer.tile.region + DDR memref function boundary
  -> candidate DDR tile-view materialization
       candidate traversal/tile shape/boundary slice proposal -> DDR memref.subview tile operands
  -> wafer-lower-tile-region-to-instr / instruction lowering
       target-abstract tile ops -> wafer.instr.* over unplaced Wafer-tagged memref
  -> wafer-lower-groups-to-memory-planned-instr / SPM offset assignment
       accepted SPM offset facts and lifetimes on instruction-level candidate IR
  -> DDR offset assignment
       DDR external view validation + compiler-managed/resident/inter-group DDR offset facts
  -> wafer-lower-groups-to-selected-instr / candidate-selection + committed materialization
       search shape-driven traversal/reduction refinement frontier, verify same-domain output coverage,
       rerun candidate gates, commit only the selected passing candidate into main IR
  -> R3.4 placed instruction-level realization
       placed memref / access descriptor realization from accepted SPM/DDR offset facts
  -> R3.5 launch/runtime DDR materialization
       runtime allocation/import/query/package metadata from accepted DDR demand
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
| Resource / runtime | `wafer.placement.map`、`wafer.launch` | SPM/DDR planning 不新增单独 storage IR；accepted SPM/DDR offset facts 后续 materialize 到 placed memref / descriptor / launch boundary |

## 当前 Active

**R3.4 placed instruction-level realization**

```text
Pipeline position:
- Upstream artifact / IR:
  committed main IR；已包含 selected candidate 的 `wafer.tile.region`、`wafer.instr.*`、actual DDR tile
  views、accepted SPM offset facts 和 accepted DDR offset facts；原 logical `wafer.group` 已删除。
- Current stage responsibility:
  从 committed IR 中 materialize placed instruction-level realization：把 accepted SPM/DDR offset facts
  和 memref/view demand 转成 R3.5/R3.6 可消费的 placed memref / access descriptor 边界；不重新决定
  tile shape、layout、SPM offset 或 DDR offset。
- Output artifact / IR:
  placed instruction-level IR / placed memref / access descriptor boundary；descriptor 只来自 committed IR
  中可重算的 view、layout、offset 和 intent facts。
- Downstream consumer:
  R3.5 launch/runtime DDR materialization；R3.6+ C ABI / packet / object / runtime adapter。
- User-level driver / named pipeline:
  后续应在 committed selected-instr named pipeline 之后追加 placed realization；用户不应手工拼
  candidate artifact、offset fact 和 descriptor materialization。
- Explicit non-goals:
  不重新做 tile search、layout search、SPM planning 或 DDR planning；不生成 runtime
  allocation/import、ABI call、packet、object 或 physical address。
- Completion gate:
  以 named pipeline 重放 R3.1 -> candidate-selection -> committed materialization 已完成链路，R3.4
  输出由 committed SPM/DDR facts 生成 placed descriptor boundary，且 R3.5 能直接消费 runtime DDR
  demand。
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
  tile-shape search 归 candidate-selection。
- R3.2d RDMA/WDMA 支持 strided DDR view 是 consumer 能力；descriptor 的 offset/stride 必须来自
  IR 中的 memref view，不能从名字、shape 或 whole-boundary memref 推断。
- R3.2f 已接入 offset-only `wafer.spm.offset = #wafer.spm_offset<offset>` planning fact；arena
  作用域是单个 `wafer.tile.region`，默认普通 SPM range 为 `[0x10000, 0x2F0000)`，size、alignment
  和 bank span 由 memref type/layout 与 target policy 重算。V0 已对
  instruction-level IR 建立 region-aware lifetime dataflow：straight-line last-use 后复用、
  `scf.if` path-sensitive branch 互斥复用、`scf.for` loop-carried/backedge lifetime、loop 后 temp
  复用，以及 async token 到 wait/drain 的 lifetime 延伸都由 IR 结构重算；offset 搜索采用
  pressure-weighted offline packing。
- R3.2g 已接入 offset-only `wafer.ddr.offset = #wafer.ddr_offset<offset>` planning fact。DDR
  `memref.alloc` 本身表达 compiler-managed allocation；size、alignment、lifetime 和 read/write role
  由 memref type/layout、target policy、SSA use-def、structured control-flow、async token 和
  RDMA/WDMA descriptor 重算。external DDR function args 不写 access summary attr；descriptor/view/root
  validation 保持 pass-local analysis。`wafer-plan-ddr-memory` 对 function/candidate scope 建立
  structured lifetime dataflow，覆盖 straight-line reuse、`scf.if` path-sensitive 互斥复用、`scf.for`
  loop-carried/backedge lifetime 和 async token 延伸；compiler-managed DDR demand 在 default arena 内做
  pressure-weighted first-fit packing，并覆盖 capacity、largest-contiguous、bandwidth 和 alignment failure。
- R3.2h/R3.3 已接入 `wafer-select-group-tile` 和 `wafer-lower-groups-to-selected-instr`。候选由实际
  static traversal shape 生成每维 refinement options；search 从 full traversal tile/no split 开始，
  gate failure 后按当前 root 的容量压力扩展 traversal 或 reduction refinement neighbor。同 traversal
  domain 的多个输出可共享同一个 traversal tile；direct matmul root 和 supported `linalg.generic`
  reduction root 可进入 reduction split refinement。quick SPM bound 只做最小必要 footprint 下界剪枝，
  真实 SPM 峰值和复用由 SPM offset assignment gate 判定。每个 candidate 的 first /
  last / tail / corner-tail representative tile 都要按 candidate tile-view materialization ->
  instruction lowering -> SPM offset assignment -> DDR offset assignment -> verifier
  通过后才接受。`tile-search=first-legal` 返回第一个 passing candidate；
  `tile-search=min-estimated-time` 只在 passing candidates 中按 lowered instruction IR 的粗估时间排序。
  硬件 SPM/DDR range、alignment 和粗估 timing 由 fixed target policy 提供，不作为
  `wafer-select-group-tile` public option；搜索空间由 `tile-search-effort` preset 和显式 search
  option 控制。`candidate-parallelism` 只在 `min-estimated-time` 下校验和消费，`first-legal` 不读取。
  selected candidate 会 inline commit 回原 `wafer.group` 位置，保留原 parent function 和无关函数；
  输出不生成 `*_selected_group_*` 旁路函数。
  大 shape full-tile control-flow 组合已覆盖 `scf.if` 中 matmul + elementwise chain 和 `scf.for`
  中 elementwise accumulate；`scf.if` 分支内 same-domain multi-output 的 `matmul + elementwise`
  组合已有固定 target SPM range 下的 lit 覆盖，并由专用 heavy runner 覆盖 large-shape traversal tile、
  `K` split 和三输出多 op 组合。需要 tiled materialization 的 `scf.if`
  root 当前仍结构化
  `no_candidate`，等待 output coverage / control-flow tiling interface 扩展。
  不同 output domain、partial scatter coverage、复杂 recompute/cut 和 dynamic reduction range 仍需要
  后续 IR/interface 扩展，当前不能用 side table 或名字伪支持。
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
| R3.2g | done | R3.2f memory-planned instruction IR + actual DDR tile-view facts + DDR `memref.alloc` / external DDR boundary values | `wafer.ddr.offset` 标注 compiler-managed DDR accepted offset；external DDR access summary 不落主 IR；descriptor/view/root validation 从当前 IR 重算；straight-line、`scf.if`、`scf.for` 和 async token lifetime 支持 offset reuse；capacity、largest-contiguous、bandwidth、alignment 等 failure 结构化诊断 |
| R3.2h | done | R3.1 group + R3.2a/b facts + candidate gates | `wafer-select-group-tile` / `wafer-lower-groups-to-selected-instr` 从 full traversal/no split 开始搜索 shape-driven traversal/reduction refinement frontier，验证 same-domain multi-output coverage，逐个重放 candidate tile-view materialization、instruction lowering、SPM offset assignment、DDR offset assignment 和 verifier；支持 `tile-search=first-legal|min-estimated-time`，硬件 range/alignment/timing 来自 fixed target policy，搜索空间 option 可覆盖 `tile-search-effort` preset；选择 passing candidate，不把 rejected plan 或 cost breakdown 写入 committed IR；覆盖 elementwise、static slice、large K=1000 matmul、matmul K split、generic reduction split、same-domain multi-output、固定 target SPM range 下的 multi-output `matmul + elementwise`、large-shape traversal tile / K split 手动 heavy case、reduce、`scf.if`、`scf.for`、representatives 和 gate early-exit |
| R3.3 | done | R3.2h selected candidate evaluation result | selected candidate inline commit 回原 `wafer.group` 位置；原 parent function、无关函数和同函数其它 ops 保留；输出 committed `wafer.tile.region` / `wafer.instr.*` / accepted SPM-DDR offset facts，不生成 `*_selected_group_*` 旁路函数 |

## 后续队列

| ID | 状态 | 输入 | 输出 / 完成 gate |
| --- | --- | --- | --- |
| R3.4 | active | R3.3 committed tile-region + accepted layout/SPM/DDR facts | placed instruction-level IR / placed memref / access descriptor；不重新决定 tile/layout/memory plan |
| R3.5 | pending | R3.4 placed/memref-aware IR + accepted DDR offset facts and descriptor/view demand | runtime allocation/import/query/package materialization；验证 runtime object 满足 committed IR-derived DDR offsets/ranges，不重新做 DDR planning |
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

实现 R3.4 placed instruction-level realization：

1. 定义 committed `wafer.tile.region` / `wafer.instr.*` 中 SPM/DDR offset facts 到 placed memref /
   access descriptor 的 materialization 边界。
2. 明确哪些 descriptor 字段由 memref type/layout、`memref.subview`、`wafer.spm.offset` 和
   `wafer.ddr.offset` 重算，哪些仍留给 R3.5 runtime/launch boundary。
3. 保证 R3.4 不重新做 tile/layout/SPM/DDR planning，只消费 committed IR 中已经 accepted 的 facts。
4. 补 completion proof：R3.1 -> candidate-selection -> committed materialization -> R3.4 的 named
   pipeline 输出能被 R3.5 runtime DDR demand materialization 直接消费。
