# Wafer Compiler Progress

更新时间：2026-06-18

本文件只记录当前看板、主线 pipeline 和下一步顺序。详细设计、复盘、测试命令和长验证说明
放在对应 `tasks/` 设计文档、git commit 和测试里；这里不维护第二份设计细节。

## 状态标记

- `active`：当前优先推进。
- `pending`：依赖前序任务完成。
- `later`：当前主线之后再做。

局部验证通过不自动等于主线完成。主线完成证明必须能重放已完成上游链路，并让当前 stage 输出被
下游边界直接消费。

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
  -> SPMD partition
       valid device-mesh aware Shardy propagation, Wafer-owned SPMD partition,
       parameter shard metadata/payload
  -> shard-binding materialization
       `wafer.shard.binding` 引用 device mesh；只绑定已有 shard facts，不重新切分 tensor
  -> local compute normalization
       post-SPMD StableHLO collective handoff + StableHLO-to-Linalg
  -> logical group formation
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
  -> ABI / LLVM lowering
       committed instruction IR + accepted offsets + topology/device-mesh/shard-binding
       + communication/sync -> LLVM dialect call sequence or `wafer_*` C ABI / packet builder input
  -> object + package manifest assembly
       object/program id, entrypoint, ABI version, constants, endpoint and resource metadata
  -> runtime adapter / board launch
       allocate/import/query/bind runtime objects, launch program, validate completion and errors
```

`wafer-compile-stablehlo` 只做 frontend / StableHLO program verifier。除 `wafer-opt` program pipeline
之外的工具入口只作为局部验证入口，不在本文件列为主线阶段。

## 当前 Active

**device-mesh SPMD integration**

```text
Pipeline position:
- Upstream artifact / IR:
  `wafer.target.topology` explicit physical tile graph；requested logical mesh policy；SPMD default
  seed policy 需要的 logical mesh rank count / axes。
- Current stage responsibility:
  从 available connected topology 中选择 valid `wafer.device.mesh`，显式记录 logical mesh axes、
  rank count 和 logical rank -> encoded tile id embedding；bad tile、PG-disabled tile 或 disconnected
  mesh axis 必须在 SPMD 前失败。
- Output artifact / IR:
  `wafer.device.mesh` accepted SPMD rank domain。它引用 `wafer.target.topology`，是 rank domain
  和 rank->encoded endpoint embedding 的唯一事实源。
- Downstream consumer:
  SPMD default seed policy、`wafer.shard.binding` verifier、communication verifier、ABI/LLVM lowering、
  package manifest 和 runtime adapter。
- User-level driver / named pipeline:
  `wafer-opt` program pipeline。局部工具入口只作为实现索引，不能成为长期合同。
- Explicit non-goals:
  不执行 Shardy / XLA SPMD partition，不重新切分 tensor，不做 tile shape/layout/SPM/DDR planning，
  不生成 DTE route、ABI call、packet、object、package 或 runtime allocation object。
- Completion gate:
  `wafer.device.mesh` 能从 `wafer.target.topology` 的 available connected component 中 materialize；
  verifier 能拒绝 rank count / axis product mismatch、unavailable tile、duplicate tile、unknown tile、
  disconnected mesh axis 和 rank 数超过 available tile。
```

## 已可依赖的上游边界

| 边界 | 当前可依赖产物 |
| --- | --- |
| Frontend / SPMD | StableHLO Wafer program、Shardy propagation、Wafer-owned SPMD partition、rank-local metadata / parameter shard payload |
| Target topology | `wafer.target.topology` explicit physical tile graph、default single-card 4x4 / 16 tile materialization、availability/link verifier |
| Local compute normalization | Linalg/Tensor/SCF/Arith/Math local compute + verifier-legal `wafer.tensor.*` handoff |
| Logical group | verifier-legal logical `wafer.group` |
| Tile-region / instruction lowering | memref-backed `wafer.tile.region` + instruction-level `wafer.instr.*` over Wafer-tagged memrefs |
| DDR tile-view / SPM / DDR planning | candidate DDR `memref.subview` tile operands、accepted SPM offset facts、accepted DDR offset facts、structured failure diagnostics |
| Candidate selection / committed materialization | selected candidate committed into main IR；rejected plans and cost traces do not enter IR |

## 后续队列

| 阶段 | 状态 | 输入 | 输出 / 完成 gate |
| --- | --- | --- | --- |
| device-mesh SPMD integration | active | `wafer.target.topology` + requested logical mesh policy | `wafer.device.mesh` valid SPMD rank domain + rank->encoded tile id embedding；SPMD default seed 从 mesh rank count / axes 取数 |
| shard-binding migration | pending | partitioned StableHLO program + parameter shard metadata + `wafer.device.mesh` | `wafer.shard.binding` 引用 device mesh；rank coverage 对 mesh 校验，不再对 `wafer.placement.map` 校验 |
| placement-map cleanup | pending | current `wafer.placement.map` transition op + `wafer.device.mesh` + `wafer.target.topology` | 删除 `wafer.placement.map` 的长期 rank->tile / topology 职责；需要 block id 时只保留薄 launch/block binding，且不复制 rank->tile |
| communication mesh consumer | pending | tiled tensor collective + device-mesh/local-rank/buffer facts | peer、route legality 和 p2p schedule 从 topology/device mesh 查询；结果 materialize 到 communication / Direct DTE resource / local-drain 边界 |
| ABI / LLVM lowering | pending | committed instruction IR + accepted SPM/DDR offset facts + topology/device-mesh/shard-binding + launch-block binding + communication/sync lowering | LLVM dialect call sequence 或 `wafer_*` C ABI / packet builder input；按需重算 launch/resource view，固定参数单位、address domain、wait/completion policy 和 ABI version |
| object + package manifest | pending | ABI/LLVM artifact + committed IR + topology/device-mesh/shard-binding | object/program id、entrypoint、ABI version 和 IR-derived package manifest；endpoint/resource/constant metadata 由同一 resource view analysis 从 IR 重算 |
| runtime adapter / board launch | pending | package + runtime adapter | allocate/import/query/bind runtime objects，launch program，验证 completion、错误传播和 board gate |
| transformer staged gaps | pending | static transformer local shard IR / staged IR gaps | full-block schedule 或拒绝原因；补 mask/select、dynamic-bound policy、constant/weight slice 等 |
| overlap / cost calibration | later | board/profile 输出 | overlap、cost model 和 PMU calibration |

## 当前不做

- Serving integration、KV cache / paged attention / prefill-decode 调度。
- raw DTE non-unicast collective ABI。
- LLVM dialect / LLVM IR lowering、object emission 或真实 `wafer_*` runtime call emission；这些归后续
  ABI、package 和 runtime 边界。
- 自定义 LLVM backend 或 ISA intrinsic lowering。
- 以 importer、runtime path、workload shape、parameter 名称或 pass-local side table 作为 IR 合同。

## 下一步

1. 落 `wafer.device.mesh`：从 available connected topology 选择 valid rank domain；bad tile 或
   disconnected mesh axis 必须在 SPMD 前失败。
2. 让 SPMD 默认 seed policy 从 `wafer.device.mesh` 读取 rank count / axes。
3. 将 `wafer.shard.binding` 迁移为引用 `wafer.device.mesh`。
4. 清理 `wafer.placement.map`：长期只允许薄 launch/block binding 保留 block id，不复制 rank->tile、
   topology dimensions、bad tile、codec 或 connectivity。
5. 让 communication、ABI/LLVM 和 package manifest 从 topology/device mesh、`wafer.shard.binding`、
   薄 launch/block binding、accepted offsets 和 committed instruction IR 派生 launch-visible metadata。
