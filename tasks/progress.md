# Wafer Compiler Progress

更新时间：2026-06-18

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
  -> target-topology / device-mesh materialization
       target descriptor / runtime capability / board profile -> wafer.target.topology；
       valid connected topology -> wafer.device.mesh；tile id codec 只在 topology import/materialization
       边界生成或重写 explicit coord -> encoded tile id mapping
  -> stablehlo-spmd
       program verification, default sharding seed, valid device mesh aware Shardy propagation,
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
  -> device mesh / launch-block / local-shard binding
       valid rank domain -> encoded physical tile endpoint embedding；local-shard metadata 引用 device mesh，
       只消费 SPMD / frontend 已显式 materialize 的 shard facts
  -> R3.6 ABI / LLVM lowering
       committed `wafer.instr.*`, accepted offsets, topology/device-mesh/shard-binding, communication
       and sync boundary
       -> LLVM dialect call sequence or `wafer_*` C ABI / packet builder input
  -> R3.7 object + package manifest assembly
       object/program id, entrypoint, ABI version, constants, endpoint and resource metadata
  -> R3.8 runtime adapter / board launch
       allocate/import/query/bind runtime objects, launch program, validate completion and errors
```

`wafer-compile-stablehlo` 只做 frontend / StableHLO program verifier。
`wafer-propagate-stablehlo-sharding`、`wafer-lower-stablehlo-to-linalg`、R3.2a/b dump pass、
`--wafer-convert-group-to-tile-region` 和 `--wafer-convert-tile-region-to-instr` 是内部或局部测试入口，
不是用户级编译流程。

## 当前 Active

**target-topology / device-mesh / shard-binding contract**

```text
Pipeline position:
- Upstream artifact / IR:
  target descriptor / runtime capability / board profile；verified frontend StableHLO program；
  committed selected-instr IR 已包含 selected candidate 的 `wafer.tile.region`、`wafer.instr.*`、
  actual DDR tile views、accepted SPM offset facts 和 DDR demand legality facts。
- Current stage responsibility:
  在 SPMD 前 materialize / import topology，显式记录 coord -> encoded tile id、availability 和 links；
  选择 valid `wafer.device.mesh` 作为 SPMD rank domain 和唯一 rank->encoded tile embedding。若上游
  已有显式 local shard facts，则只绑定并验证 bounds，不重新切分 tensor、不从名字或 payload 恢复
  shard。该阶段形成 topology/device-mesh/shard-binding contract，并迁移当前 placement consumer；
  launch block id 如需跨阶段保留，只能作为薄 binding，不能复制 rank->tile。不生成 runtime
  allocation、ABI call、packet、object 或 package。
- Output artifact / IR:
  `wafer.target.topology` explicit physical tile graph；`wafer.device.mesh` selected valid rank domain；
  `wafer.shard.binding` 引用 device mesh 并表达 boundary tensor slice。`wafer.placement.map` 是当前
  过渡 op，长期不保存 rank count、rank->tile、topology dimensions、bad tile、tile id codec 或
  connectivity。local shard 只在 `wafer.shard.binding` / 后续 launch-visible resource view 中引用，
  不复制 memory plan、packet field、runtime handle 或 search trace。
- Downstream consumer:
  SPMD 先消费 `wafer.device.mesh`；`wafer.shard.binding` 引用同一 mesh；communication verifier、
  ABI/LLVM lowering、package manifest 和 runtime adapter 从同一
  topology/device-mesh/shard-binding fact source 派生。当前 `wafer.placement.map` 只能作为过渡
  projection，不能继续作为长期 consumer 接口。
- User-level driver / named pipeline:
  `wafer-plan-placement` pass 和 `wafer-lower-groups-to-placement` named pipeline；后者在 committed
  selected-instr boundary 之后追加 placement，用户不应手工拼 accepted instruction IR 和 placement
  fixture 作为主线 compile flow。
- Explicit non-goals:
  不重新做 tile search、layout search、SPM planning 或 DDR planning；不选择 DTE route、packet
  resource、runtime allocation object、physical address、ABI call 或 object/package 格式。
- Completion gate:
  named pipeline 重放 target topology materialization -> valid device mesh selection -> SPMD partition
  -> group formation -> candidate selection -> committed instruction materialization -> placement；emitted
  `wafer.device.mesh` 被 SPMD、shard binding、communication verifier 和 ABI/package resource view
  直接消费；verifier 能拒绝 rank count、unavailable tile、重复 block/tile、out-of-topology、
  disconnected mesh axis 和 rank 数超过可用 tile。
```

R3.4 placed/access descriptor realization 取消为独立主线阶段。原因是 committed instruction IR 已经
显式包含 instruction operands、DDR tile views、SPM offset facts、DDR offset facts 和 movement
descriptor attrs；把这些可重算事实再 materialize 成 placed memref / access descriptor 中间层会形成
重复事实源。后续 ABI/LLVM lowering 或 package manifest 如需 launch/resource/address/range/stride
参数，应在使用点从 committed IR、accepted offset facts 和 topology/device-mesh/shard-binding contract 派生，不通过
R3.4 旁路协议传递。

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
| R3.4 | removed | R3.3 committed instruction IR 已经包含可重算 memory facts | 取消独立 placed memref / access descriptor materialization；必要 address/range/demand 检查在 ABI/LLVM lowering、package manifest 和 runtime adapter 使用点从 IR 派生验证 |
| R3.5 | removed | R3.3 committed instruction IR + accepted offsets + topology/device-mesh/shard-binding contract | 取消独立 launch/resource contract materialization；launch signature、external binding、workspace 和 resident constant 等 resource view 由 R3.6/R3.7/R3.8 在使用点通过同一 analysis/verifier 从 IR 重算，不落第二份 metadata |

## 后续队列

| ID | 状态 | 输入 | 输出 / 完成 gate |
| --- | --- | --- | --- |
| topology fact source | active | target descriptor / runtime capability / board profile + requested mesh shape / sharding hints | `wafer.target.topology` explicit tile graph、tile id codec import/rewrite boundary、availability/link verifier |
| device-mesh SPMD integration | pending | `wafer.target.topology` + requested logical mesh policy | `wafer.device.mesh` valid SPMD rank domain + rank->encoded tile id embedding；SPMD default seed 从 mesh rank count / axes 取数 |
| shard-binding migration | pending | partitioned StableHLO program + parameter shard metadata + `wafer.device.mesh` | `wafer.shard.binding` 引用 device mesh；rank coverage 对 mesh 校验，不再对 `wafer.placement.map` 校验 |
| placement-map cleanup | pending | current `wafer.placement.map` transition op + `wafer.device.mesh` + `wafer.target.topology` | 删除 `wafer.placement.map` 的长期 rank->tile / topology 职责；需要 block id 时只保留薄 launch/block binding，且不复制 rank->tile |
| communication mesh consumer | pending | tiled tensor collective + device-mesh/local-rank/buffer facts | peer、route legality 和 p2p schedule 从 topology/device mesh 查询；结果 materialize 到 communication / Direct DTE resource / local-drain 边界 |
| R3.6 | pending | committed instruction IR + accepted SPM/DDR offset facts + topology/device-mesh/shard-binding contract + launch-block binding + communication/sync lowering | LLVM dialect call sequence 或 `wafer_*` C ABI / packet builder input；按需重算 launch/resource view，固定参数单位、address domain、wait/completion policy 和 ABI version |
| R3.7 | pending | R3.6 ABI/LLVM artifact + committed IR + topology/device-mesh/shard-binding contract | object/program id、entrypoint、ABI version 和 IR-derived package manifest；manifest 的 endpoint/resource/constant metadata 由同一 resource view analysis 从 IR 重算，manifest roundtrip 不能替代 object/link 最小验证 |
| R3.8 | pending | R3.7 package + runtime adapter | allocate/import/query/bind runtime objects，launch program，验证 completion、错误传播和 board gate |
| R5.1-R5.2 | pending | static transformer local shard IR / staged IR gaps | full-block schedule 或拒绝原因；补 mask/select、dynamic-bound policy、constant/weight slice 等 |
| P9 | later | board/profile 输出 | overlap、cost model 和 PMU calibration |

## 当前不做

- Serving integration、KV cache / paged attention / prefill-decode 调度。
- raw DTE non-unicast collective ABI。
- 当前 active topology/device-mesh/shard-binding 阶段不做 LLVM dialect / LLVM IR lowering、object emission 或真实
  `wafer_*` runtime call emission；这些分别归 R3.6/R3.7/R3.8。
- 自定义 LLVM backend 或 ISA intrinsic lowering。
- 以 importer、runtime path、workload shape 或 parameter 名称作为 IR 合同。

## 下一步

先补 SPMD 前 topology / device mesh fact source，再做 placement 下游消费：

1. 定义 `wafer.target.topology` 和 topology model，显式保存 tile coord、encoded tile id、availability
   和 links；默认 tile id codec 只用于 materialize mapping，另设 import/rewrite pass 覆盖 driver
   remap、PG/bad tile 和跨卡编码。
2. 定义 `wafer.device.mesh`，从 available connected topology 中选择 valid rectangular submesh，并让
   SPMD / shard binding 引用该 mesh；bad tile 或 disconnected mesh axis 必须在 SPMD 前失败。
3. 让 SPMD 默认 seed policy 从 `wafer.device.mesh` 读取 rank count / axes；单卡默认 topology
   配置是 4x4 / 16 tile，但必须经 topology/device mesh materialization 进入 SPMD，不是 SPMD
   写死常量。
4. 将 `wafer.shard.binding` 迁移为引用 `wafer.device.mesh`，rank coverage 对 mesh 校验，不再通过
   `wafer.placement.map` 对齐。
5. 将当前 `wafer.placement.map` 的 rank count、rank->tile、topology dimensions、bad tile list 和 tile
   id 编码公式职责迁移到 `wafer.device.mesh` / `wafer.target.topology`；需要 block id 时只保留薄
   launch/block binding，不复制 rank->tile。
6. 然后让 communication lowering、ABI/LLVM lowering 和 package manifest 从 topology/device mesh、
   `wafer.shard.binding`、薄 launch/block binding、accepted SPM/DDR offset facts 和 committed
   instruction IR 派生 launch-visible metadata。
