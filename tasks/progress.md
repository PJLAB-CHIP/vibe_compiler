# Wafer Compiler Progress

更新时间：2026-06-23

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
  -> local compute normalization
       post-SPMD StableHLO collective handoff to `wafer.linalg_ext.collective.*`
       + StableHLO-to-Linalg
  -> logical group formation
       dependency-preserving logical `wafer.group`
  -> tile-region / communication / instruction / memory-planned candidate pipeline
       memref-backed `wafer.tile.region`
       -> buffer-level communication collective materialization
            tiled `wafer.linalg_ext.collective.*` + SPM storage + local-rank facts
            -> `wafer.tile.*` collective over unplaced SPM memrefs
       -> p2p Direct DTE instruction schedule lowering
            `wafer.tile.*` collective + topology/execution-mesh endpoint view
            -> `wafer.instr.dte_send` / `dte_recv` / `dte_wait`
               over unplaced SPM memrefs
       -> candidate DDR tile-view materialization
       -> instruction-level `wafer.instr.*`
       -> accepted SPM and DDR offset facts over compute/movement/communication demand
       -> candidate selection and committed materialization
  -> launch-block binding / endpoint projection
       optional per-rank block id；不复制 rank->physical endpoint
  -> ABI / LLVM lowering
       committed instruction IR + accepted offsets + topology/execution-mesh
       + program parameter shard metadata/resource view + communication/sync
       -> LLVM dialect call sequence or `wafer_*` C ABI / packet builder input
  -> object + package manifest assembly
       object/program id, entrypoint, ABI version, constants, endpoint and resource metadata
  -> runtime adapter / board launch
       allocate/import/query/bind runtime objects, launch program, validate completion and errors
```

`wafer-compile-stablehlo` 只做 frontend / StableHLO program verifier。除 `wafer-opt` program pipeline
之外的工具入口只作为局部验证入口，不在本文件列为主线阶段。

## 当前 Active

**p2p Direct DTE instruction schedule lowering**

```text
Pipeline position:
- Upstream artifact / IR:
  `wafer.tile.*` buffer-level collective、memref-backed `wafer.tile.region` / SPM storage values、
  `wafer.execution.mesh` 和 `wafer.target.topology`。
- Current stage responsibility:
  从 tile collective 的 `rank_group`、`local_rank`、buffer、byte count 和 topology/execution-mesh
  endpoint view 派生 p2p Direct DTE schedule，并把 schedule 显式 rewrite 成
  `wafer.instr.dte_send` / `wafer.instr.dte_recv` / `wafer.instr.dte_wait` 与必要的 local reduce/sync IR。
- Output artifact / IR:
  含 explicit Direct DTE send/recv/wait 和 local reduce/sync 的 instruction-level tile-region IR；
  仍不写 SPM/DDR offset、DTE resource id、packet field 或 ABI/runtime handle。
- Downstream consumer:
  SPM memory planning、DDR memory planning、ABI/LLVM lowering、package manifest 和 runtime adapter。
- User-level driver / named pipeline:
  `wafer-opt` program pipeline；局部 pass 只作为实现索引，不能成为长期合同。
- Explicit non-goals:
  不做 SPM/DDR offset assignment，不生成 ABI call、packet、object、package 或 runtime allocation
  object；不使用 raw DTE non-unicast helper 作为 correctness path。
- Completion gate:
  从已有 StableHLO / LinalgExt collective handoff、group/tile-region 和 `wafer.tile.*` collective
  链路重放到 verifier-legal `wafer.instr.dte_*` schedule；SPM memory planning 能直接消费通信
  token/lifetime/buffer demand，相关 lit、unit test 和组织检查通过。
```

## 已可依赖的上游边界

| 边界 | 当前可依赖产物 |
| --- | --- |
| Frontend / SPMD | StableHLO Wafer program、Shardy propagation、Wafer-owned SPMD partition、rank-local metadata / parameter shard payload |
| Target topology / execution mesh | `wafer.target.topology` regular card/tile grid、default single-card 4x4 / 16 tile materialization、`wafer.execution.mesh` all_available rank-domain policy、SPMD default seed consumption、unavailable endpoint verifier |
| Local compute normalization | Linalg/Tensor/SCF/Arith/Math local compute + verifier-legal `wafer.linalg_ext.collective.*` handoff |
| Logical group | verifier-legal logical `wafer.group` |
| Tile-region / instruction lowering | memref-backed `wafer.tile.region` + instruction-level `wafer.instr.*` over Wafer-tagged memrefs；compute/movement 路径可用，top-level single-result all_gather / reduce_scatter / all_reduce 已能 materialize 成 `wafer.tile.*` collective；compact SPM all_gather、tensor SPM all_reduce 和 full-input reduce_scatter 已能展开成 explicit `wafer.instr.dte_*` schedule |
| DDR tile-view / SPM / DDR planning | candidate DDR `memref.subview` tile operands、accepted SPM offset facts、accepted DDR offset facts、structured failure diagnostics；compute/movement 路径可用，communication staging / token lifetime 必须在 comm lowering 后进入同一 planning gate |
| Candidate selection / committed materialization | selected candidate committed into main IR；rejected plans and cost traces do not enter IR |

## 后续队列

| 阶段 | 状态 | 输入 | 输出 / 完成 gate |
| --- | --- | --- | --- |
| execution-mesh SPMD integration | done | `wafer.target.topology` + requested logical mesh policy | `wafer.execution.mesh` valid SPMD rank-domain policy + derived/optional endpoint view；SPMD default seed 从 mesh rank count / axes 取数；旧 `tile-count` pass / pipeline fallback 已删除 |
| program parameter shard verification | done | partitioned StableHLO program + parameter shard metadata + `wafer.execution.mesh` | program verifier 校验 rank coverage、slice bounds、payload shape/dtype 和 execution mesh rank count；core IR 不 materialize per-rank slice table |
| rank-endpoint cleanup | done | legacy rank->tile transition op / pass / package schema | op、pass、pipeline、comm verifier dependency 和 manifest endpoint schema 已删除；需要 block id 时只保留薄 launch/block binding，且不复制 rank->tile |
| IR naming and communication layer cleanup | done | 已收敛 naming / instruction-family 设计 | `wafer.linalg_ext.collective.*`、`wafer.instr.dte_*`、instruction family + `dte` 全链路一致；文档、代码、测试和 pipeline 不再保留旧合同 |
| buffer-level communication collective materialization | done | tiled `wafer.linalg_ext.collective.*` + unplaced SPM buffer/local-rank facts + execution mesh rank domain | top-level single-result `all_gather` / `reduce_scatter` / `all_reduce` materialize 成 verifier-legal `wafer.tile.*` collective；buffer、bytes、rank_group、local_rank 和 effect 边界来自 IR，不选择 p2p schedule；发生在 SPM memory planning 前 |
| p2p Direct DTE instruction schedule lowering | done | `wafer.tile.*` collective + topology/execution-mesh derived endpoint view | compact SPM all_gather、tensor SPM all_reduce 和 full-input reduce_scatter 已生成 explicit `wafer.instr.dte_send` / `dte_recv` / `dte_wait` schedule，并由 SPM memory planning 消费；peer/route 从 topology/execution mesh 查询，不保存 side table |
| comm-aware memory planning gate | pending | instruction-level compute/movement + `wafer.instr.dte_*` over unplaced SPM memrefs | communication staging、token lifetime、local drain / DTE wait 和 buffer reuse 被 SPM planning 消费；accepted SPM/DDR offset facts 覆盖 comm demand |
| ABI / LLVM lowering | pending | committed instruction IR + accepted SPM/DDR offset facts + topology/execution-mesh + program parameter shard metadata/resource view + launch-block binding + communication/sync lowering | LLVM dialect call sequence 或 `wafer_*` C ABI / packet builder input；按需重算 launch/resource view，固定参数单位、address domain、wait/completion policy 和 ABI version |
| object + package manifest | pending | ABI/LLVM artifact + committed IR + topology/execution-mesh + program parameter shard metadata/resource view | object/program id、entrypoint、ABI version 和 IR-derived package manifest；endpoint/resource/constant metadata 由同一 resource view analysis 从 IR 重算 |
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

1. 让 comm-aware candidate 继续通过 SPM offset assignment 和 DDR offset assignment；
   Direct DTE resource id / concrete address / ABI emission 只能在 accepted offsets 后派生。
2. 再让 ABI/LLVM 和 package manifest 从 topology/execution mesh、program parameter shard metadata /
   resource view、薄 launch/block binding、accepted offsets、committed instruction IR 和 communication/sync
   IR 派生 launch-visible metadata。
