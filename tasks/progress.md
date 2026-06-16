# Wafer Compiler Progress

更新时间：2026-06-16

本文件只做看板索引：当前任务、主线顺序、完成阶段和下一步。设计合同、实现细节、验证命令和复盘放在
对应 `tasks/` 设计文档、git commit 和测试里。

## 状态

局部 pass、fixture 或单个 dump 通过不等于完成；主线完成证明必须能重放已完成上游链路，并让当前
stage 输出被下游边界直接消费。

## 主线顺序

```text
StableHLO program -> SPMD -> Linalg -> wafer.group -> tile-region -> instr
  -> SPM offsets -> DDR offsets -> selected committed instr -> placed instr
  -> launch/runtime DDR -> C ABI / packet / object / runtime adapter
```

用户级主入口以 `wafer-opt` program pipeline 为准；局部 dump/conversion pass 不替代主线 compile flow。

## 当前 Active

| ID | 当前任务 | 完成 gate |
| --- | --- | --- |
| R3.4 | placed instruction-level realization | 从 R3.3 committed IR 生成 placed memref / access descriptor 边界，并能被 R3.5 直接消费 |

R3.4 只消费 committed IR 中已接受的 layout/SPM/DDR facts，不重新做 tile search、layout search、SPM
planning 或 DDR planning。

## 已完成

| 范围 | 输出边界 |
| --- | --- |
| P2.F1-P2.S2 | StableHLO Wafer program、Shardy/SPMD、rank-local metadata/payload |
| R2.4-R3.1 | post-SPMD local compute + logical `wafer.group` |
| R3.2a-R3.2e | group demand/layout analysis、memref-backed tile-region、instruction IR、DDR tile views |
| R3.2f-R3.2g | accepted SPM/DDR offset facts |
| R3.2h-R3.3 | candidate search/gate replay + selected candidate commit 回主 IR |

细节边界以对应设计文档为准；本表不记录实现流水。

## 后续队列

| ID | 状态 | 输出边界 |
| --- | --- | --- |
| R3.5 | pending | runtime allocation/import/query/package materialization |
| R3.6-R3.8 | pending | C ABI / packet / package manifest |
| R4.1-R4.5 | pending | rank/block/coord、per-rank slices、writeback、placed package |
| R5.1-R5.2 | pending | static transformer local shard IR / staged IR gaps |
| R6.1-R6.2 | pending | tile communication materialization |
| P7/P8/P9 | later | LLVM/object、runtime adapter、board/profiling |

## 当前不做

Serving/KV cache、raw DTE collective ABI、LLVM/object emission、真实 runtime call emission、自定义 LLVM
backend，以及任何基于名字、路径或 workload shape 的 IR 合同。

## 下一步

实现 R3.4：

1. 定义 committed `wafer.tile.region` / `wafer.instr.*` 到 placed memref / access descriptor 的边界。
2. 从 memref type/layout、`memref.subview`、`wafer.spm.offset` 和 `wafer.ddr.offset` 重算 descriptor 字段。
3. 接入 named pipeline，并补 R3.1 -> R3.3 -> R3.4 的主线 completion proof。
