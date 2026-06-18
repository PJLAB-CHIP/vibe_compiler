# Wafer P0-P6 Design Conformance Audit

日期：2026-05-26

状态：审计记录；用于纠正 P0-P6 历史 `done` 状态的含义，并记录 collective handoff 复查结论。

## 结论

当前实现不能按设计文档口径称为 P0-P6 主线完成。更准确的状态是：P0/P1 建立了可运行的工程和
最小 IR 基础实现；P2-P6 形成了一批 textual lowering、verifier 和 fixture/最小验证 gate，但大量
gate 只证明了局部 fixture 可以运行，尚未证明设计文档要求的主路径闭环。

R0.1 审计时，工程组织也没有按
`tasks/2026-05-11-wafer-ai-compiler-architecture.md` 第 7 节收敛：frontend lowering、
group/tile materialization、layout/SPM/DDR、communication 和旧 ABI issue-op lowering 基本都堆在
`lib/Wafer/Transforms`。R0.2 已恢复源码 ownership 边界；R1.1 之后 ODS、op verifier 和 dialect
tests 已按 IR 层组织；R1.2/R1.3 曾补 interface/resource 查询合同和 local compute stage-connection
gate；2026-06-02 后历史 stage-connection gate 和旧 unit/debug pass 链已删除。

主线要求：历史 P2.S1 路线若把 sharding facts 写成 `wafer.spmd.*` 或私有
sidecar，不是主线 SPMD program contract；历史 StableHLO collective 直降 `wafer.tile.*` communication 的 pass
只能算已删除的后段 communication coverage，不能作为 group/tiling 输入。正确主线需要
`frontend export -> StableHLO/SDY -> Wafer Shardy propagation -> Wafer-owned XLA SPMD partitioner
compiler stage -> partitioned StableHLO`，再经 Wafer LinalgExt-style tensor collective handoff 进入
group/tiling。旧 PyTorch/XLA post-SPMD export 测试入口已删除；它不消费 Wafer propagation 输出，
不能作为 P2.S1 主链完成证明。

历史 `tasks/progress.md` 中 P0-P6 的 `done` 应理解为“该局部批次有对应测试”，不能理解为：

- frontend program 由真实 importer adapter 产出。
- group planner 已按设计接入下游 legality analysis 和 resource planning。
- tile/layout/SPM/DDR 已实现完整 committed instruction + placement/resource/ABI program。
- C ABI 已 lower 到真实 `wafer_*` call / wrapper / packet。
- package manifest 由当前 `wafer-opt` lowering 输出自动生成。
- runtime adapter、runtime allocation binding、completion 和板端数值验证已实现。

## 分层审计

| 范围 | 当前实现 | 与设计文档的偏差 |
| --- | --- | --- |
| P0 工程/依赖/组织 | CMake、`wafer-opt`、lit/gtest、dependency pin/layering validation 已有；源码 ownership 拆到 Frontend / IR / Analysis / Transforms / Conversion / ABI；旧聚合 Conversion pass target 已删除；R0.3 已记录并验证依赖层级边界；IR 文件和测试已按层组织 | 最小工程入口可用，不代表后端链路完成 |
| P1 Wafer IR verifier | ODS/type/attr/op/verifier 正负例覆盖了核心基础实现；ODS、verifier 和 dialect tests 已按 IR 层组织；R1.2 已补 interface/resource/effect 查询合同；历史 R1.3 local compute stage-connection gate 已删除 | dialect verifier 的孤立负例仍会使用 `builtin.unrealized_conversion_cast` 构造非法边界值；communication bridge 的 visible cast 仍未清理；instruction/runtime 主链路仍未闭环 |
| P2 frontend / local compute | StableHLO textual lowering 到 linalg/tensor/arith/math 子集已有，部分 transformer staged form 有 FileCheck | `wafer-compile-stablehlo` 早期只是 synthetic 最小验证 tool；PyTorch/XLA program directory capture 已补入 P2.F1，但 ConstantLike/storage contract 仍未闭环；部分 slice 只是 frontend dataflow fixture，不是完整 frontend program pipeline |
| P3 single-tile local compute | 旧 group formation、root tile exploration、single-tile materialization、SPM allocation、DDR external binding、C ABI issue-op unit pass 链已删除；保留 ABI descriptor / manifest tool-unit 覆盖 | group planner、root tile planning、SPM/DDR/instruction-level lowering、ABI/LLVM emission 和 IR-derived package manifest 均需按真实 program chain 恢复 |
| P4 multi-tile no communication | placement map verifier、multi-tile no-comm outlining已有 | multi-tile materialization 克隆同一 whole-tensor tile_region，最后用最后一个 tile_region result 替换 group result；没有真实 local shard slicing、per-rank output merge 或 runtime launch binding；manifest endpoint metadata 尚未由 IR-derived package path 生成 |
| P5 transformer block | norm/softmax/linear-residual/MLP frontend lowering fixtures、attention QK/AV lowering、transformer textual pipeline 已有 | 这些 fixture 只证明 structured tensor dataflow 覆盖，不是 full schedule planner；transformer package manifest 尚未由 transformer IR 导出；workspace/resident constant metadata 与 IR dataflow 没有自动一致性来源；没有真实 full-block device program |
| P6 communication | p2p / collective verifier 和历史 StableHLO collective bridge coverage 曾存在；旧 ring lowering / C ABI issue-op unit pass 链已删除 | Direct DTE resource allocator、collective buffer slice/address offset lowering、tensor collective handoff 到 tiled communication materialization，以及真实 launch/resource/package metadata 均未闭环 |

## 主要根因

1. 任务看板把“局部 fixture gate 可运行”记录成了“设计主路径完成”。
2. 历史上多个 integration test 把 IR lowering 和 fixed manifest/C stub fixture 放在同一文件里，
   但二者没有 use-def 或 program 生成关系；这些测试拼接和 fixed emitter 已删除。
3. 设计文档里明确要求的 LLVM/C ABI/runtime/package 边界没有落成实现，却被 local compile/package
   gate 的措辞覆盖掉了。
4. P3-P6 继续叠加功能时，没有先关闭 P3 的真实 program/package 闭环，导致后续 multi-tile、
   transformer 和 communication gate 都继承
   fixture 出口。
5. R0.1 时源码组织没有跟随设计文档的 IR stage / op prefix 边界演进，导致 frontend、planner、
   materialization、resource planning、communication 和 ABI lowering 的 ownership 被同一个
   `Transforms` 聚合目录掩盖。R0.2 已修正源码 ownership；当前 IR 文件和 dialect tests 已按 IR 层组织。
6. 个别任务口径把“当前下游 lowering/resource/runtime 没实现”混成了“上游语义不支持”。后续恢复
   必须按硬件能力和 IR contract 判断支持范围；当前实现缺口应落成下游恢复任务或 IR 扩展，不能
   反向污染 frontend、SPMD、group 或 placement 语义。

## 纠正原则

- P0-P6 的历史 `done` 只保留为局部进度记录，不再作为设计一致性完成声明。
- 后续不继续扩展 P7/P8/P9；先重开 P0-P6 设计一致性恢复队列。从 committed instruction IR、
  topology/device-mesh/shard-binding contract、resource view analysis 和 ABI/LLVM lowering artifact 自动导出 package manifest 是恢复 P3/P4/P5 local compile gate 的一个子任务，
  不能被当作 P7 已经可以开始的前提。
- 每个被恢复为 `done` 的任务必须满足对应设计文档的合同，或在任务名/验收里明确收窄为局部 fixture。
- 显式 manifest fixture 只能作为 validator/stub tool unit fixture，不能作为 compile pipeline
  correctness fence；fixed manifest emitter 不应恢复。
- 当前下游实现缺口不能作为上游 program / IR 的语义边界。若合法上游语义能由 Wafer 硬件/ABI
  能力组合表达，恢复任务应补 IR contract、verifier、lowering 或 runtime/package gate。

## 优先修复顺序

1. 逐项重读 P0-P6 对应设计文档，把每个历史 `done` 拆成“设计合同 / 当前局部实现 / 缺口 / 恢复任务”；
   R0.1 结果见 `tasks/2026-05-26-wafer-p0-p6-recovery-status.md`。
2. 修正 `tasks/progress.md` 中 P0-P6 的状态，不再让局部 fixture gate 占用设计完成语义。
3. R0.2 已按架构文档第 7 节恢复源码 ownership 边界；R0.3 已补 core/frontend/runtime/test tooling
   dependency target 可见范围；R1.1/R1.2/R1.3 已补 IR 文件边界、interface/resource 查询和 local
   compute stage-connection gate；该历史 gate 已在 2026-06-02 随旧 pass 链删除。Launch/runtime 仍只有
   局部 boundary，真实 adapter 后续恢复。
4. 从 P2/P3 开始恢复主链路：frontend program、group planner、tile/layout/SPM/DDR/C ABI/package 的
   每一步都必须按设计合同闭环。
5. 在恢复 local compile gate 时，补 IR-derived package manifest emission：manifest 和
   C stub 必须来自当前 `wafer-opt` pipeline 输出。
6. P0-P6 设计闭环恢复后，再进入后续真实 C ABI / LLVM / runtime / board gate。
