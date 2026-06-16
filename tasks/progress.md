# Wafer Compiler Progress

更新时间：2026-06-16

本文件记录当前看板、主线 pipeline、完成口径和下一步。设计细节、实现复盘、测试命令
和长验证说明放在对应 `tasks/` 设计文档、git commit 和测试里；这里不写逐条过程记录。

## 状态标记

- `active`：当前优先推进。
- `ready`：前置边界已明确，可以开始。
- `pending`：依赖前序任务完成。
- `later`：当前主线之后再做。
- `done`：实现、测试和文档已按当前 pipeline contract 收口。
- `removed`：经设计 review 取消为独立主线阶段；必要责任已并入其它可验证边界。

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
  -> placement / local-shard contract
       accepted logical rank/block -> physical coordinate mapping and launch-visible local shard metadata
  -> R3.5 launch/resource contract formation
       launch signature, DDR resource contract, workspace/constant/external binding requirements
       derived from committed IR, accepted offsets and placement
  -> R3.6 ABI / LLVM lowering
       committed `wafer.instr.*`, communication and sync boundary -> LLVM dialect call sequence or
       `wafer_*` C ABI / packet builder input
  -> R3.7 object + package manifest assembly
       object/program id, entrypoint, ABI version, constants, placement and resource metadata
  -> R3.8 runtime adapter / board launch
       allocate/import/query/bind runtime objects, launch program, validate completion and errors
```

`wafer-compile-stablehlo` 只做 frontend / StableHLO program verifier。
`wafer-propagate-stablehlo-sharding`、`wafer-lower-stablehlo-to-linalg`、R3.2a/b dump pass、
`--wafer-convert-group-to-tile-region` 和 `--wafer-convert-tile-region-to-instr` 是内部或局部测试入口，
不是用户级编译流程。

## 当前 Active

**placement / local-shard contract**

```text
Pipeline position:
- Upstream artifact / IR:
  R3.3 committed main IR；已包含 selected candidate 的 `wafer.tile.region`、`wafer.instr.*`、actual DDR
  tile views、accepted SPM offset facts 和 accepted DDR offset facts；同时消费 frontend/SPMD
  保留的 logical rank、block/local shard facts，以及 target topology / capability / good-tile metadata。
- Current stage responsibility:
  materialize accepted placement map 和 launch-visible local shard/block metadata；验证 logical rank 覆盖、
  physical tile 可用性、block id/local shard bounds 和 committed IR 的 per-rank boundary 可解释性。
  该阶段只形成 placement/local-shard contract，不生成 runtime allocation、ABI call、packet、object
  或 package。
- Output artifact / IR:
  accepted placement/local-shard contract；可由 `wafer.placement.*`、launch-visible function/module
  metadata 或后续 `wafer.launch` boundary 承载。它只保存不能从 local IR 重算的 mapping，不复制
  memory plan、packet field、runtime handle 或 search trace。
- Downstream consumer:
  R3.5 launch/resource contract formation、R6 communication lowering、R3.7 package manifest 和 R3.8 runtime adapter。
- User-level driver / named pipeline:
  后续应在 committed selected-instr named pipeline 之后追加 placement/local-shard stage；用户不应
  手工拼 accepted instruction IR、placement fixture 和 package metadata 作为主线 compile flow。
- Explicit non-goals:
  不重新做 tile search、layout search、SPM planning 或 DDR planning；不选择 DTE route、packet
  resource、runtime allocation object、physical address、ABI call 或 object/package 格式。
- Completion gate:
  以 named pipeline 重放 R3.1 -> candidate-selection -> committed materialization -> placement；
  emitted placement/local-shard contract 被 R3.5 launch/resource contract 和 R6 communication lowering
  直接消费，且 verifier 能拒绝 rank 覆盖、bad tile、重复 block/tile 和 local shard 越界。
```

R3.4 placed/access descriptor realization 取消为独立主线阶段。原因是 committed instruction IR 已经
显式包含 instruction operands、DDR tile views、SPM offset facts、DDR offset facts 和 movement
descriptor attrs；把这些可重算事实再 materialize 成 placed memref / access descriptor 中间层会形成
重复事实源。后续 launch/resource contract 和 ABI/LLVM lowering 如需 address/range/stride 参数，应在
对应阶段从 committed IR、accepted offset facts 和 placement/local-shard contract 派生，不通过 R3.4
旁路协议传递。

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

## 已取消主线阶段

| ID | 状态 | 原输入 | 取消原因 / 责任归属 |
| --- | --- | --- | --- |
| R3.4 | removed | R3.3 committed instruction IR 已经包含可重算 memory facts | 取消独立 placed memref / access descriptor materialization；必要 address/range/demand 检查并入 launch/resource contract 和后续 ABI/LLVM lowering 的派生验证 |

## 后续队列

| ID | 状态 | 输入 | 输出 / 完成 gate |
| --- | --- | --- | --- |
| placement/local-shard | active | R3.3 committed tile-region / `wafer.instr.*` + frontend/SPMD logical rank/local shard facts + target topology/capability | accepted placement map、block id、launch-visible local shard metadata；验证 rank coverage、good/bad tile、block/tile uniqueness 和 local shard bounds |
| R3.5 | ready | committed instruction IR + accepted SPM/DDR offset facts + placement/local-shard contract | launch signature + DDR resource contract；导出 workspace/resident constant/external binding requirements，不 allocate/import/query runtime object，不生成 ABI/object/package |
| R6.1-R6.2 | pending | tiled tensor collective + placement/local-rank/buffer facts | materialize tile communication 到 communication / Direct DTE resource / local-drain 边界；结果进入 R3.6 ABI/LLVM lowering |
| R3.6 | pending | committed instruction IR + launch/resource contract + communication/sync lowering | LLVM dialect call sequence 或 `wafer_*` C ABI / packet builder input；固定参数单位、address domain、wait/completion policy 和 ABI version |
| R3.7 | pending | R3.6 ABI/LLVM artifact + placement/resource/constant metadata | object/program id、entrypoint、ABI version 和 IR-derived package manifest；manifest roundtrip 不能替代 object/link 最小验证 |
| R3.8 | pending | R3.7 package + runtime adapter | allocate/import/query/bind runtime objects，launch program，验证 completion、错误传播和 board gate |
| R5.1-R5.2 | pending | static transformer local shard IR / staged IR gaps | full-block schedule 或拒绝原因；补 mask/select、dynamic-bound policy、constant/weight slice 等 |
| P9 | later | board/profile 输出 | overlap、cost model 和 PMU calibration |

## 当前不做

- Serving integration、KV cache / paged attention / prefill-decode 调度。
- raw DTE non-unicast collective ABI。
- 当前 active placement/local-shard 阶段不做 LLVM dialect / LLVM IR lowering、object emission 或真实
  `wafer_*` runtime call emission；这些分别归 R3.6/R3.7/R3.8。
- 自定义 LLVM backend 或 ISA intrinsic lowering。
- 以 importer、runtime path、workload shape 或 parameter 名称作为 IR 合同。

## 下一步

实现 placement / local-shard contract：

1. 从 frontend/SPMD 保留的 logical rank/local shard facts、target topology/capability 和 committed
   `wafer.tile.region` / `wafer.instr.*` 重建 accepted placement map，不引入 placed memref /
   access descriptor 旁路协议。
2. 定义 block id、physical coordinate、good/bad tile assumption 和 launch-visible local shard metadata
   的 IR / launch boundary 表示；这些字段只表达 placement 和 shard bounds，不保存 memory plan 或
   runtime handle。
3. 验证 rank coverage、tile/block uniqueness、bad tile 过滤、local shard bounds 和 committed boundary
   可解释性；失败时给结构化 diagnostic，不重新做 group/candidate/memory planning。
4. 补 completion proof：R3.1 -> candidate-selection -> committed materialization -> placement 的 named
   pipeline 输出能被 R3.5 launch/resource contract 和 R6 communication lowering 直接消费。
