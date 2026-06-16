# Wafer Compiler Progress

更新时间：2026-06-16

本文件是当前看板索引，只记录主线顺序、已完成 artifact、active task、后续输入/完成 gate 和下一步。
设计合同、实现细节和验证命令放在对应 `tasks/` 设计文档、git commit 和测试里。

## 完成口径

局部 pass、fixture 或单个 dump 通过不等于完成；主线完成证明必须能重放已完成上游链路，并让当前
stage 输出被下游边界直接消费。

## 主线顺序

```text
StableHLO program
  -> SPMD
  -> Linalg local compute
  -> logical group
  -> tile-region
  -> instruction IR
  -> SPM offsets
  -> DDR offsets
  -> selected committed instruction IR
  -> placed instruction IR
  -> launch/runtime DDR
  -> C ABI / packet / object / runtime adapter
```

用户级主入口以 `wafer-opt` program pipeline 为准；局部 dump/conversion pass 不替代主线 compile flow。

## 当前 Active

| ID | 状态 | 输入 | 输出 / 完成 gate |
| --- | --- | --- | --- |
| R3.4 | active | R3.3 committed tile-region + accepted layout/SPM/DDR facts | placed instruction-level IR / placed memref / access descriptor；不重新决定 tile/layout/memory plan，并能被 R3.5 直接消费 |

R3.4 只消费 committed IR 中已经 accepted 的 facts，不重新做 tile search、layout search、SPM planning
或 DDR planning。

## 已完成主线

| ID | 输出 artifact / IR |
| --- | --- |
| P2.F1-P2.S2 | StableHLO Wafer program、Shardy/SPMD partition、rank-local metadata / parameter shard payload |
| R2.4 | post-SPMD local compute structured tensor IR + `wafer.tensor.*` handoff |
| R3.1 | logical `wafer.group` |
| R3.2a | `GroupTilingDemand` analysis |
| R3.2b | `GroupLayoutPlan` analysis |
| R3.2c | memref-backed `wafer.tile.region` + DDR memref function boundary |
| R3.2d | instruction-level `wafer.instr.*` over Wafer-tagged memref |
| R3.2e | static/candidate DDR tile-view materialization to actual `memref.subview` |
| R3.2f | accepted `wafer.spm.offset` facts |
| R3.2g | accepted `wafer.ddr.offset` facts |
| R3.2h | candidate search + gate replay |
| R3.3 | selected candidate inline commit 回主 IR |

已完成项的详细 legality、coverage、known gap 和验证入口以对应设计文档为准。

## 后续队列

| ID | 状态 | 输入 | 输出 / 完成 gate |
| --- | --- | --- | --- |
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
- 自定义 LLVM backend。
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
