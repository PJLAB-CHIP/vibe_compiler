# Wafer Compiler Progress

更新时间：2026-06-18

本文件只记录当前看板、主线 pipeline 和下一步顺序。详细设计、历史复盘、测试命令和长验证说明
放在对应 `tasks/` 设计文档、git commit 和测试里；这里不维护第二份设计细节。

## 状态标记

- `active`：当前优先推进。
- `ready`：前置边界已明确，可以开始。
- `pending`：依赖前序任务完成。
- `later`：当前主线之后再做。
- `done`：实现、测试和文档已按当前 pipeline contract 收口。

局部 pass、verifier 负例、fixture 或单个 dump gate 通过，不自动等于 `done`。主线完成证明必须能
重放已完成上游链路，并让当前 stage 输出被下游边界直接消费。

## 当前主线

用户级主入口统一为 `wafer-opt` program pipeline。长期 pipeline 按 IR / artifact 边界描述：

```text
PyTorch/XLA StableHLO Wafer program directory
  -> target-topology materialization
       target descriptor / runtime capability / board profile
       -> `wafer.target.topology` explicit coord -> encoded tile id, availability, links
  -> device-mesh selection
       valid connected topology -> `wafer.device.mesh`
       单卡默认 topology 配置是 4x4 / 16 tile；SPMD 只消费 selected mesh，不写死该常量
  -> stablehlo-spmd
       valid device-mesh aware Shardy propagation, Wafer-owned SPMD partition,
       parameter shard metadata/payload
  -> shard-binding materialization
       `wafer.shard.binding` 引用 device mesh；只绑定已有 shard facts，不重新切分 tensor
  -> stablehlo-spmd-to-linalg
       post-SPMD StableHLO collective handoff + official StableHLO-to-Linalg
  -> stablehlo-spmd-to-group
       dependency-preserving logical `wafer.group`
  -> tile-region / instruction / memory-planned candidate pipeline
       memref-backed `wafer.tile.region`
       -> candidate DDR tile-view materialization
       -> instruction-level `wafer.instr.*`
       -> accepted SPM and DDR offset facts
       -> candidate selection and committed materialization
  -> launch-block binding / endpoint projection
       optional per-rank block id；不复制 rank->encoded endpoint
  -> communication mesh consumer
       tiled tensor collective + topology/device mesh -> p2p / Direct DTE / sync boundary
  -> R3.6 ABI / LLVM lowering
       committed instruction IR + accepted offsets + topology/device-mesh/shard-binding
       + communication/sync -> LLVM dialect call sequence or `wafer_*` C ABI / packet builder input
  -> R3.7 object + package manifest assembly
       object/program id, entrypoint, ABI version, constants, endpoint and resource metadata
  -> R3.8 runtime adapter / board launch
       allocate/import/query/bind runtime objects, launch program, validate completion and errors
```

`wafer-compile-stablehlo` 只做 frontend / StableHLO program verifier。`wafer-propagate-stablehlo-sharding`、
`wafer-lower-stablehlo-to-linalg` 和其它单 pass / debug pipeline 是内部或局部测试入口，不是用户级
编译流程。

## 当前 Active

**topology fact source**

```text
Pipeline position:
- Upstream artifact / IR:
  target descriptor / runtime capability / board profile；单卡默认 4x4 / 16 tile topology config；
  optional driver remap、PG-disabled tile、bad tile 和 link/cost facts。
- Current stage responsibility:
  materialize / import `wafer.target.topology`，显式记录 tile coord、encoded tile id、availability、
  links 和 tile id codec provenance。默认 tile id codec 只能在 topology materialization / rewrite
  边界使用；下游通过 topology model 查询 endpoint，不能手写 row-major/card-major 公式。
- Output artifact / IR:
  `wafer.target.topology` explicit physical tile graph。它是 physical topology、availability、links
  和 encoded tile id 的唯一事实源。
- Downstream consumer:
  `wafer.device.mesh` selection、SPMD default seed policy、shard binding verifier、communication
  verifier、ABI/LLVM lowering、package manifest 和 runtime adapter。
- User-level driver / named pipeline:
  `wafer-opt` program pipeline。局部 pass 名只作为实现索引，不能成为长期合同。
- Explicit non-goals:
  不执行 SPMD partition，不重新切分 tensor，不做 tile shape/layout/SPM/DDR planning，不生成
  DTE route、ABI call、packet、object、package 或 runtime allocation object。
- Completion gate:
  default single-card 4x4 / 16 tile topology 可以 materialize 为 explicit tile graph；verifier 能拒绝
  duplicate encoded tile id、malformed coord、unknown link endpoint、availability 冲突和非法 codec
  provenance；rewrite/import 能表达 remap、PG-disabled tile、bad tile 和跨卡编码变化。
```

## 已完成主线

| 范围 | 状态 | 完成口径 |
| --- | --- | --- |
| Frontend / SPMD | done | StableHLO Wafer program、Shardy propagation、Wafer-owned SPMD partition、rank-local metadata / parameter shard payload |
| Local compute normalization | done | post-SPMD StableHLO -> Linalg/Tensor/SCF/Arith/Math + verifier-legal `wafer.tensor.*` handoff |
| Logical group | done | rank-local structured tensor IR -> verifier-legal logical `wafer.group` |
| Tile-region / instruction lowering | done | logical group -> memref-backed `wafer.tile.region` -> instruction-level `wafer.instr.*` over Wafer-tagged memrefs |
| DDR tile-view / SPM / DDR planning | done | candidate DDR `memref.subview` tile operands、accepted SPM offset facts、accepted DDR offset facts、structured failure diagnostics |
| Candidate selection / committed materialization | done | candidate gates rerun before selection；selected candidate committed into main IR；rejected plans and cost traces do not enter IR |

## 后续队列

| ID | 状态 | 输入 | 输出 / 完成 gate |
| --- | --- | --- | --- |
| topology fact source | active | target descriptor / runtime capability / board profile + optional remap/PG/bad-tile/link facts | `wafer.target.topology` explicit tile graph、tile id codec import/rewrite boundary、availability/link verifier |
| device-mesh SPMD integration | pending | `wafer.target.topology` + requested logical mesh policy | `wafer.device.mesh` valid SPMD rank domain + rank->encoded tile id embedding；SPMD default seed 从 mesh rank count / axes 取数 |
| shard-binding migration | pending | partitioned StableHLO program + parameter shard metadata + `wafer.device.mesh` | `wafer.shard.binding` 引用 device mesh；rank coverage 对 mesh 校验，不再对 `wafer.placement.map` 校验 |
| placement-map cleanup | pending | current `wafer.placement.map` transition op + `wafer.device.mesh` + `wafer.target.topology` | 删除 `wafer.placement.map` 的长期 rank->tile / topology 职责；需要 block id 时只保留薄 launch/block binding，且不复制 rank->tile |
| communication mesh consumer | pending | tiled tensor collective + device-mesh/local-rank/buffer facts | peer、route legality 和 p2p schedule 从 topology/device mesh 查询；结果 materialize 到 communication / Direct DTE resource / local-drain 边界 |
| R3.6 | pending | committed instruction IR + accepted SPM/DDR offset facts + topology/device-mesh/shard-binding + launch-block binding + communication/sync lowering | LLVM dialect call sequence 或 `wafer_*` C ABI / packet builder input；按需重算 launch/resource view，固定参数单位、address domain、wait/completion policy 和 ABI version |
| R3.7 | pending | R3.6 ABI/LLVM artifact + committed IR + topology/device-mesh/shard-binding | object/program id、entrypoint、ABI version 和 IR-derived package manifest；endpoint/resource/constant metadata 由同一 resource view analysis 从 IR 重算 |
| R3.8 | pending | R3.7 package + runtime adapter | allocate/import/query/bind runtime objects，launch program，验证 completion、错误传播和 board gate |
| R5.1-R5.2 | pending | static transformer local shard IR / staged IR gaps | full-block schedule 或拒绝原因；补 mask/select、dynamic-bound policy、constant/weight slice 等 |
| P9 | later | board/profile 输出 | overlap、cost model 和 PMU calibration |

## 当前不做

- Serving integration、KV cache / paged attention / prefill-decode 调度。
- raw DTE non-unicast collective ABI。
- LLVM dialect / LLVM IR lowering、object emission 或真实 `wafer_*` runtime call emission；这些分别归
  R3.6/R3.7/R3.8。
- 自定义 LLVM backend 或 ISA intrinsic lowering。
- 以 importer、runtime path、workload shape、parameter 名称或 pass-local side table 作为 IR 合同。

## 下一步

1. 落 `wafer.target.topology`：显式 tile coord、encoded tile id、availability、links、codec
   provenance；单卡默认 4x4 / 16 tile 由该阶段 materialize。
2. 落 `wafer.device.mesh`：从 available connected topology 选择 valid rank domain；bad tile 或
   disconnected mesh axis 必须在 SPMD 前失败。
3. 让 SPMD 默认 seed policy 从 `wafer.device.mesh` 读取 rank count / axes。
4. 将 `wafer.shard.binding` 迁移为引用 `wafer.device.mesh`。
5. 清理 `wafer.placement.map`：长期只允许薄 launch/block binding 保留 block id，不复制 rank->tile、
   topology dimensions、bad tile、codec 或 connectivity。
6. 让 communication、ABI/LLVM 和 package manifest 从 topology/device mesh、`wafer.shard.binding`、
   薄 launch/block binding、accepted offsets 和 committed instruction IR 派生 launch-visible metadata。
