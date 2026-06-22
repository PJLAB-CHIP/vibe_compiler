# Wafer Compiler Progress

更新时间：2026-06-22

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
       -> `wafer.target.topology` regular card/tile grid, card interconnect kind, unavailable endpoint exceptions
  -> execution-mesh selection
       valid connected topology -> `wafer.execution.mesh`
       单卡默认 topology 配置是 4x4 / 16 tile；SPMD 只消费 selected mesh，不写死该常量
  -> SPMD partition
       valid execution-mesh aware Shardy propagation, Wafer-owned SPMD partition,
       parameter shard metadata/payload
  -> shard-binding materialization
       `wafer.shard.binding` 引用 execution mesh；只绑定已有 shard facts，不重新切分 tensor
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
       optional per-rank block id；不复制 rank->physical endpoint
  -> communication mesh consumer
       tiled tensor collective + topology/execution mesh -> p2p / Direct DTE / sync boundary
  -> ABI / LLVM lowering
       committed instruction IR + accepted offsets + topology/execution-mesh/shard-binding
       + communication/sync -> LLVM dialect call sequence or `wafer_*` C ABI / packet builder input
  -> object + package manifest assembly
       object/program id, entrypoint, ABI version, constants, endpoint and resource metadata
  -> runtime adapter / board launch
       allocate/import/query/bind runtime objects, launch program, validate completion and errors
```

`wafer-compile-stablehlo` 只做 frontend / StableHLO program verifier。除 `wafer-opt` program pipeline
之外的工具入口只作为局部验证入口，不在本文件列为主线阶段。

## 当前 Active

**communication mesh consumer**

```text
Pipeline position:
- Upstream artifact / IR:
  tiled tensor collective、`wafer.execution.mesh`、`wafer.target.topology`、local-rank/buffer facts。
- Current stage responsibility:
  peer、route legality 和 p2p schedule 从 topology/execution mesh 查询；不恢复旧 rank->tile side path。
- Output artifact / IR:
  verifier-legal communication / Direct DTE resource / local-drain 边界。
- Downstream consumer:
  communication verifier、ABI/LLVM lowering、package manifest 和 runtime adapter。
- User-level driver / named pipeline:
  `wafer-opt` program pipeline。局部工具入口只作为实现索引，不能成为长期合同。
- Explicit non-goals:
  不执行 Shardy / XLA SPMD partition，不重新切分 tensor，不做 tile shape/layout/SPM/DDR planning，
  不生成 DTE route、ABI call、packet、object、package 或 runtime allocation object。
- Completion gate:
  communication lowering 消费 execution mesh/topology-derived endpoint view，不再依赖独立 rank->tile
  transition op。
```

## 已可依赖的上游边界

| 边界 | 当前可依赖产物 |
| --- | --- |
| Frontend / SPMD | StableHLO Wafer program、Shardy propagation、Wafer-owned SPMD partition、rank-local metadata / parameter shard payload |
| Target topology / execution mesh | `wafer.target.topology` regular card/tile grid、default single-card 4x4 / 16 tile materialization、`wafer.execution.mesh` all_available rank-domain policy、SPMD default seed consumption、unavailable endpoint verifier |
| Local compute normalization | Linalg/Tensor/SCF/Arith/Math local compute + verifier-legal `wafer.tensor.*` handoff |
| Logical group | verifier-legal logical `wafer.group` |
| Tile-region / instruction lowering | memref-backed `wafer.tile.region` + instruction-level `wafer.instr.*` over Wafer-tagged memrefs |
| DDR tile-view / SPM / DDR planning | candidate DDR `memref.subview` tile operands、accepted SPM offset facts、accepted DDR offset facts、structured failure diagnostics |
| Candidate selection / committed materialization | selected candidate committed into main IR；rejected plans and cost traces do not enter IR |

## 后续队列

| 阶段 | 状态 | 输入 | 输出 / 完成 gate |
| --- | --- | --- | --- |
| execution-mesh SPMD integration | done | `wafer.target.topology` + requested logical mesh policy | `wafer.execution.mesh` valid SPMD rank-domain policy + derived/optional endpoint view；SPMD default seed 从 mesh rank count / axes 取数；旧 `tile-count` pass / pipeline fallback 已删除 |
| shard-binding migration | done | partitioned StableHLO program + parameter shard metadata + `wafer.execution.mesh` | `wafer.shard.binding` 引用 execution mesh；rank coverage 对 mesh 校验；IR 不再保存独立 rank count |
| rank-endpoint cleanup | done | legacy rank->tile transition op / pass / package schema | op、pass、pipeline、comm verifier dependency 和 manifest endpoint schema 已删除；需要 block id 时只保留薄 launch/block binding，且不复制 rank->tile |
| communication mesh consumer | active | tiled tensor collective + execution-mesh/local-rank/buffer facts | peer、route legality 和 p2p schedule 从 topology/execution mesh 查询；结果 materialize 到 communication / Direct DTE resource / local-drain 边界 |
| ABI / LLVM lowering | pending | committed instruction IR + accepted SPM/DDR offset facts + topology/execution-mesh/shard-binding + launch-block binding + communication/sync lowering | LLVM dialect call sequence 或 `wafer_*` C ABI / packet builder input；按需重算 launch/resource view，固定参数单位、address domain、wait/completion policy 和 ABI version |
| object + package manifest | pending | ABI/LLVM artifact + committed IR + topology/execution-mesh/shard-binding | object/program id、entrypoint、ABI version 和 IR-derived package manifest；endpoint/resource/constant metadata 由同一 resource view analysis 从 IR 重算 |
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

1. 让 communication、ABI/LLVM 和 package manifest 从 topology/execution mesh、`wafer.shard.binding`、
   薄 launch/block binding、accepted offsets 和 committed instruction IR 派生 launch-visible metadata。
