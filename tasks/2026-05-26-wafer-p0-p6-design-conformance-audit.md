# Wafer P0-P6 Design Conformance Audit

日期：2026-05-26

状态：审计记录；用于纠正 P0-P6 历史 `done` 状态的含义

## 结论

当前实现不能按设计文档口径称为 P0-P6 主线完成。更准确的状态是：P0/P1 建立了可运行的工程和
最小 IR skeleton；P2-P6 形成了一批 textual lowering、verifier 和 fixture/smoke gate，但大量 gate
只证明了局部 skeleton 可以运行，尚未证明设计文档要求的主路径闭环。

R0.1 审计时，工程组织也没有按
`tasks/2026-05-11-wafer-ai-compiler-architecture.md` 第 7 节收敛：frontend lowering、
group/tile materialization、layout/SPM/DDR、communication 和 ABI skeleton 基本都堆在
`lib/Wafer/Transforms`。R0.2 已恢复源码 ownership 边界，但 ODS 仍集中在 `WaferOps.td`，verifier
仍集中在 `WaferDialect.cpp`；按 op prefix 拆 IR 文件边界仍是 R1.1。

历史 `tasks/progress.md` 中 P0-P6 的 `done` 应理解为“该 skeleton 批次有对应测试”，不能理解为：

- frontend artifact 由真实 importer adapter 产出。
- group planner 已按设计接入下游 legality/resource oracle。
- tile/layout/SPM/DDR 已实现完整 storage-realized program。
- C ABI 已 lower 到真实 `wafer_*` call / wrapper / packet。
- package manifest 由当前 `wafer-opt` lowering 输出自动生成。
- runtime adapter、BO binding、completion 和板端数值验证已实现。

## 分层审计

| 范围 | 当前实现 | 与设计文档的偏差 |
| --- | --- | --- |
| P0 工程/依赖/组织 | CMake、`wafer-opt`、lit/gtest、dependency pin/checker 已有；R0.2 已把源码 ownership 拆到 Frontend / IR / stage-specific Transforms / Conversion / ABI | 最小工程入口可用，但依赖 ownership、runtime/importer 可见范围和 IR op-prefix 文件边界仍未按设计完成；不代表后端链路完成 |
| P1 Wafer IR skeleton | ODS/type/attr/op/verifier 正负例覆盖了核心 skeleton；多数 op 仍在同一 `WaferOps.td`，多数 verifier 在同一 `WaferDialect.cpp` | 多个 interface/resource/effect 仍是骨架；IR 文件没有按 group/tile_region/layout/SPM/compute/comm/sync/launch 拆开；测试大量使用 `builtin.unrealized_conversion_cast` 构造边界值，只证明 verifier 形态，不证明真实上游/下游连接 |
| P2 frontend / local compute | StableHLO textual lowering 到 linalg/tensor/arith/math 子集已有，部分 transformer staged form 有 FileCheck | `wafer-import-model` 是 synthetic smoke tool，不是真实 importer adapter；sidecar/ConstantLike/storage contract 仍未闭环；部分 slice 是 acceptance pattern，不是完整 frontend artifact pipeline |
| P3 M0 | group formation、root tile check、single-tile materialization、SPM trial、DDR external binding、C ABI skeleton、manifest fixture 均有测试 | group planner 只处理有限 single-op pattern；root tile candidate 基本等于完整静态 result shape，不是候选搜索+下游 oracle；SPM 是顺序 trial；DDR 只有 external compact bytes demand；C ABI 是 skeleton op，不是 `wafer_*` call；golden packet 只是 descriptor builder，不是 wrapper-to-register golden；M0 manifest 由 fixed smoke emitter 生成，不来自当前 IR |
| P4 M1 | placement map verifier、multi-tile no-comm outlining、per-tile launch arg fixture 已有 | multi-tile materialization 克隆同一 whole-tensor tile_region，最后用最后一个 tile_region result 替换 group result；没有真实 local shard slicing、per-rank output merge 或 runtime launch binding；manifest placement metadata 是 fixed smoke fixture |
| P5 transformer block | norm/softmax/projection/MLP acceptance passes、attention QK/AV lowering、M6 textual pipeline 已有 | 多数 pass 是 pattern acceptance，不是 full schedule planner；M6 package manifest 是 fixed fixture，不由 M6 IR 导出；workspace/resident constant metadata 与 IR dataflow 没有自动一致性来源；没有真实 full-block device artifact |
| P6 communication | p2p verifier、ring all-gather/reduce collectives、StableHLO collective bridge、DTE skeleton resource verifier 已有 | Direct DTE resource id 是单调 skeleton 分配，不是目标资源 allocator；collective lowering 没有真实 buffer slice/address offset lowering；StableHLO collective bridge 用 visible cast 衔接 tensor/tile_buffer；communication metadata 没进入真实 package/runtime path |

## 主要根因

1. 任务看板把“skeleton gate 可运行”记录成了“设计主路径完成”。
2. 多个 integration test 把 IR lowering 和 fixed manifest/C stub fixture 放在同一文件里，但二者没有
   use-def 或 artifact 生成关系。
3. 设计文档里明确要求的 LLVM/C ABI/runtime/package 边界没有落成实现，却被 local compile/package
   gate 的措辞覆盖掉了。
4. P3-P6 继续叠加功能时，没有先关闭 P3 的真实 artifact/package 闭环，导致后续 M1/M6/M2-M4 都继承
   skeleton 出口。
5. R0.1 时源码组织没有跟随设计文档的 IR stage / op prefix 边界演进，导致 frontend、planner、
   materialization、resource check、communication 和 ABI lowering 的 ownership 被同一个
   `Transforms` 聚合目录掩盖。R0.2 已修正源码 ownership；op-prefix IR 文件拆分仍在 R1.1。

## 纠正原则

- P0-P6 的历史 `done` 只保留为骨架进度记录，不再作为设计一致性完成声明。
- 后续不继续扩展 P7/P8/P9；先重开 P0-P6 设计一致性恢复队列。`wafer.abi.*` IR 自动导出 package
  manifest 是恢复 P3/P4/P5 local compile gate 的一个子任务，不能被当作 P7 已经可以开始的前提。
- 每个被恢复为 `done` 的任务必须满足对应设计文档的合同，或在任务名/验收里明确收窄为 skeleton。
- fixed smoke manifest 只能作为 tool unit fixture，不能作为 compile pipeline correctness fence。

## 优先修复顺序

1. 逐项重读 P0-P6 对应设计文档，把每个历史 `done` 拆成“设计合同 / 当前 skeleton / 缺口 / 恢复任务”；
   R0.1 结果见 `tasks/2026-05-26-wafer-p0-p6-recovery-status.md`。
2. 修正 `tasks/progress.md` 中 P0-P6 的状态，不再让 skeleton gate 占用设计完成语义。
3. R0.2 已按架构文档第 7 节恢复源码 ownership 边界：IR / Frontend / Transforms / Conversion / ABI
   已分开；Launch/runtime 仍只有 skeleton boundary，真实 adapter 后续恢复。
4. 从 P2/P3 开始恢复主链路：frontend artifact、group planner、tile/layout/SPM/DDR/C ABI/package 的
   每一步都必须按设计合同闭环。
5. 在恢复 M0/M1/M6 local compile gate 时，补 IR-derived package manifest emission：manifest 和
   C stub 必须来自当前 `wafer-opt` pipeline 输出。
6. P0-P6 设计闭环恢复后，再进入后续真实 C ABI / LLVM / runtime / board gate。
