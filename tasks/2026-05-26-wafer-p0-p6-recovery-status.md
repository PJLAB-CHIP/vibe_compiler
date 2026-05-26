# Wafer P0-P6 Recovery Status

日期：2026-05-26

状态：R0.1 完成记录；用于重写 P0-P6 历史 skeleton 项状态。

本文只记录 P0-P6 的设计合同、当前实现、缺口和恢复任务。它不声明 P0-P6 已完成，也不替代各
设计文档的语义边界。

## 依据

已重读并用于本轮状态重写的主文档：

- `tasks/2026-05-11-wafer-ai-compiler-architecture.md`
- `tasks/2026-05-12-wafer-group-design.md`
- `tasks/2026-05-13-wafer-design-docs-gap-review.md`
- `tasks/2026-05-21-wafer-layout-materialization-design.md`
- `tasks/2026-05-21-wafer-spm-bufferization-design.md`
- `tasks/2026-05-25-wafer-frontend-stablehlo-artifact-design.md`
- `tasks/2026-05-25-wafer-shardy-spmd-design.md`
- `tasks/2026-05-25-wafer-placement-design.md`
- `tasks/2026-05-25-wafer-local-compute-normalization-design.md`
- `tasks/2026-05-25-wafer-tile-region-design.md`
- `tasks/2026-05-25-wafer-compute-dialect-design.md`
- `tasks/2026-05-25-wafer-communication-dialect-design.md`
- `tasks/2026-05-25-wafer-ddr-resource-allocation-design.md`
- `tasks/2026-05-25-wafer-c-abi-golden-packet-design.md`
- `tasks/2026-05-25-wafer-launch-runtime-package-design.md`
- `tasks/2026-05-25-wafer-verification-plan-design.md`

涉及硬件、runtime、ABI、memory hierarchy 的状态判断同时参考：

- `docs/wafer-hardware-instruction-set-and-programming-model.md`
- `docs/wafer-register-level-instruction-spec.md`
- `docs/tx8-deps-reverse-engineering/README.md`
- `docs/tx8-deps-reverse-engineering/tx8-interface-contract.md`
- `docs/tx8-deps-reverse-engineering/firmware-kuiper-runtime-hardware-analysis.md`
- `docs/tx8-deps-reverse-engineering/txda-pytorch-runtime-wheel-analysis.md`

当前源码依据来自 `include/Wafer`、`lib/Wafer`、`tools`、`test`、`unittests` 的现有文件组织和
pass/op/test 覆盖。

## 总体结论

P0-P6 只能保持 `skeleton` 状态。当前代码已经证明一些局部 IR、verifier、pass 和 smoke tests 能跑，
但没有满足架构文档按 IR stage 闭环的完成口径：

- `wafer` dialect 可以继续统一 namespace，但源码未按 op prefix / IR stage 拆分。
- `lib/Wafer/Transforms` 聚合了 frontend lowering、group/materialization、resource check、
  communication 和 ABI skeleton；ownership 边界不清。
- `WaferOps.td` 和 `WaferDialect.cpp` 承载了多数 op/verifier；与第 7 节建议的分层文件组织不符。
- M0/M1/M6 integration gate 把 `wafer-opt` IR FileCheck 和 fixed manifest/C stub fixture 放在同一
  测试文件里，但 package 不是从该次 IR lowering 自动导出。
- 当前 local compile 只到 `wafer.abi.*` skeleton、manifest fixture 和 generated C stub syntax compile；
  没有 LLVM dialect、LLVM IR、object、真实 `wafer_*` call、runtime adapter 或板端 completion。

## P0 工程、依赖、组织

设计合同：

- 建立可配置、可测试的 MLIR 工程入口，包含 `wafer-opt`、lit/FileCheck、gtest 和依赖版本 pin。
- StableHLO/Shardy/importer/runtime 依赖按层隔离；importer-only 依赖不能成为后端 textual tests 的硬依赖。
- 源码按架构文档第 7 节组织：`Frontend`、`IR`、`Transforms`、`Conversion`、`ABI`、launch/runtime
  ownership 清晰；一个 `wafer` dialect namespace 内仍要按 op prefix 拆文件。
- 不把本机路径、checkout path、第三方 C++ API 名或临时 build override 写成 IR contract。

当前实现：

- 有 `CMakeLists.txt`、`cmake/WaferDependencyVersions.cmake`、`wafer-opt`、lit、gtest、
  `tools/check_deps.py` 和 bootstrap 脚本。
- 有 `WAFER_ENABLE_IMPORTER_DEPS` 开关、disabled importer smoke 和 StableHLO/Shardy checkout 检查。
- 当前源码组织仍以 `include/Wafer/Dialect/Wafer/IR`、`lib/Wafer/Dialect/Wafer/IR` 和
  `lib/Wafer/Transforms` 为主，缺少设计文档中的 `Frontend`、stage-specific `Transforms` 和
  `Conversion` 边界。

缺口：

- 工程能跑不等于按设计组织完成；当前目录和库边界会继续掩盖 frontend、planner、resource、
  ABI、launch/runtime 的 ownership。
- StableHLO/Shardy dependency 当前主要服务 textual lowering smoke；真实 importer adapter 和 artifact
  verifier 还没成为独立 frontend 层。
- runtime/driver 头文件和真实 runtime adapter 尚未进入 launch/C ABI 层。

恢复任务：

- R0.2：按架构文档第 7 节恢复源码组织边界，先拆清 ownership，再移动代码。
- R0.3：补依赖层级清单，明确 core compiler、frontend/importer、runtime/driver、test tooling 的
  可见范围和 CMake target 边界。

## P1 Wafer IR Skeleton 和 Verifier

设计合同：

- `wafer.group` 只表达 tensor-level grouping/scheduling boundary，不携带 physical address、SPM
  offset、DTE resource、C ABI 或 planner trace。
- `wafer.tile_region` 表达 post-group device-side execution scope，承载 load/store、layout、
  compute、comm、sync/effect ordering，但不替代 launch。
- layout、SPM、DDR、compute、comm、sync、launch 都应有 op/type/interface/verifier 合同；共享字段只定义一次。
- op interface、type、effect 和 verifier 应暴露协议错误；pass 不能靠 side table 或名字匹配传语义。

当前实现：

- 有 `WaferAttrs.td`、`WaferTypes.td`、`WaferInterfaces.td` 和 `WaferOps.td`，覆盖 target、
  memory space、mem layout、placement、wait policy、elementwise/reduce kind、`!wafer.tile_buffer`
  以及 group/tile/layout/load/store/placement/ABI/compute/comm/sync/launch skeleton ops。
- 有 parser/printer/verifier 正负例；`WaferDialect.cpp` 实现多数 verifier。
- 有 `WaferTilingInterface`、`WaferLayoutOpInterface`、`WaferResourceEffectInterface` skeleton。

缺口：

- ODS 和 verifier 没有按 group、tile_region、layout、SPM、compute、comm、sync、launch 拆文件。
- interface/resource/effect 多数仍是 marker/skeleton，不足以支撑 group planner、layout planner、
  SPM/DDR oracle、compute/comm demand 的真实闭环。
- 测试大量用 `builtin.unrealized_conversion_cast` 构造边界值，只能证明 verifier 形态，不能证明
  上下游 IR 连接。
- 没有 storage-realized memref/descriptor 层，因此 `!wafer.tile_buffer` 还没有按设计消失。

恢复任务：

- R1.1：按 op prefix 拆 ODS、C++ verifier 和测试目录，并保持一个 `wafer` dialect namespace。
- R1.2：把 interface/effect/resource 从 skeleton 扩成 planner/resource/verifier 可调用的合同。
- R1.3：补 stage-connection tests，减少只靠 `unrealized_conversion_cast` 的孤立 verifier 用例。

## P2 Frontend Artifact 和 Local Compute Normalization

设计合同：

- 稳定入口是 verified StableHLO / MLIR artifact 加 optional sidecar 和 compile config，不是某个框架 API。
- frontend 保留 function signature、shape、dtype、dynamic bound、constant/weight 和 sharding annotation；
  不引入 Wafer SPM、DDR、layout materialization、runtime launch 或 private tensor constant op。
- local compute normalization 输出 `linalg` / `tensor` / `scf` / `arith` / `math` structured IR，保留
  dot、batch/head matmul、broadcast、reduce、softmax、norm、RoPE、MLP 的可验证 dataflow。
- pattern 只能来自 StableHLO semantics、types、indexing maps 和 SSA use-def，不能靠名字。

当前实现：

- 有 StableHLO textual artifact tests、`wafer-normalize-constants`、dot/shape/elementwise/reduce
  lowering pass，以及 norm/softmax/projection/MLP acceptance passes。
- 有 attention QK^T / AV rank-4 `dot_general` lowering 和 full local transformer block structured gate。
- `wafer-import-model` 是可诊断 graph break / eager fallback / dynamic shape marker 的 smoke tool。

缺口：

- `wafer-import-model` 不是真实 importer adapter；没有真正的 sidecar manifest、resource-backed
  constant、artifact verifier 和 sharding import source 闭环。
- StableHLO/Shardy logical mesh / SPMD partition 没有形成完整 frontend-to-local-shard pipeline。
- acceptance passes 只识别当前 structured IR pattern，不等于 group schedule planner 或 full
  transformer local compile 已完成。
- mask/select、dynamic shape、非 constant-init reduce、复杂 broadcast、常量 slicing/storage transform
  仍未按设计闭环。

恢复任务：

- R2.1：恢复 frontend artifact / importer contract，包含 importer adapter、sidecar、ConstantLike、
  dynamic bound 和 diagnostics。
- R2.2：补 Shardy/SPMD artifact bridge，保证 logical collective 和 shard-local StableHLO 是 P2/P6 的
  共同输入，而不是 isolated smoke。
- R2.3：把 local compute normalization 的覆盖状态按 op/dataflow contract 重写，不把 acceptance pass
  当 schedule completion。

## P3 M0 Single-Tile Load-GEMM-Store

设计合同：

- M0 必须验证 StableHLO/Linalg -> `wafer.group` -> `wafer.tile_region` -> layout/SPM/DDR ->
  `wafer.compute`/movement -> C ABI/package 的单 tile 闭环。
- `wafer.group` planner 需要 closed-loop search：tile shape、op tiling interface、layout materialization、
  SPM allocation trial、DDR demand/bandwidth、compute/movement legality 都要作为同一候选的检查。
- layout materialization 是真实 movement；SPM allocator 要用 liveness/effect/range/end-address；
  DDR 要有 external binding、workspace、resident constant、pool/domain/capacity/bandwidth。
- C ABI gate 要固定 `wafer_*` 参数单位和 wait policy，并通过 wrapper/golden packet 覆盖；package 要来自
  当前 lowering 输出。

当前实现：

- 有 `wafer-form-groups`、`wafer-check-root-tile-candidates`、`wafer-materialize-single-tile`、
  `wafer-compact-layout-assignment`、`wafer-check-spm-allocation`、`wafer-materialize-ddr-external-bindings`
  和 `wafer-lower-to-c-abi-skeleton`。
- 有 `wafer.abi.rdma_1d`、`wafer.abi.wdma_1d`、`wafer.abi.gemm`、elementwise/reduce skeleton ops。
- 有 `Wafer/ABI/M0Abi.h` descriptor unit tests、M0 integration test、manifest validator 和 generated C
  stub syntax compile。

缺口：

- group formation 主要覆盖有限 single-op / accepted-pattern 子集；没有 closed-loop group planner。
- root tile candidate 基本是静态 result shape 检查，不是 tile shape search + downstream oracle。
- SPM 是顺序 trial，未基于完整 interface demand、async lifetime、storage-realized memref/descriptor。
- DDR 只有 external compact bytes demand；没有 workspace、resident constants、pool/domain、bandwidth
  或 runtime binding。
- C ABI 还是 `wafer.abi.*` skeleton op 和 descriptor builder，不是真实 `wafer_*` call、LLVM lowering
  或 wrapper-to-register golden packet。
- M0 manifest 是 fixed smoke emitter，不由当前 `wafer-opt` output 自动导出。

恢复任务：

- R3.1：恢复 M0 group/tile/layout/SPM/DDR/C ABI 主链路，要求同一 IR pipeline 的候选和资源检查闭环。
- R3.2：恢复 M0 package gate，manifest 和 C stub 从当前 `wafer.abi.*` IR 导出。
- R3.3：恢复 storage realization / C ABI / golden packet 边界，至少让 M0 的 RDMA/WDMA/GEMM 有真实
  wrapper-facing call contract 和 golden packet 对照。

## P4 M1 Multi-Tile No Communication

设计合同：

- Placement 负责 logical rank 到 physical card/tile coordinate、good-tile/PG、block id、local shard
  metadata；不负责 SPM offset、DTE schedule 或 runtime API 细节。
- M1 multi-tile no-comm 需要每个 tile 处理自己的 local shard、写回对应 output slice，并由 package /
  generated artifact 区分 per-tile args。
- `wafer.tile_region` / launch metadata 必须表达 per-tile identity 和 shard slice；不能用 whole-tensor
  clone 代替 shard。

当前实现：

- 有 `wafer.placement.map` verifier，检查 rank count、topology bounds、bad tile 和 duplicate tile。
- 有 `wafer-materialize-multi-tile-no-comm`，按 placement rank 数生成多个 independent tile regions。
- package fixture / C stub 可以生成 per-tile launch arg table 和 placement metadata。

缺口：

- multi-tile materialization 克隆 whole-tensor load-GEMM-store tile_region；没有真实 shard slicing。
- group result 用最后一个 tile_region result 替换，没有 per-rank output merge 或 host-side shard readback。
- block id、physical coord、local shard metadata 没进入 tile_region body 或真实 launch binding。
- manifest placement metadata 是 fixed smoke fixture，不来自 placement op + lowering 输出。
- runtime launch completion 仍未实现，不能证明多 tile 并行或 per-tile args 被真实消费。

恢复任务：

- R4.1：恢复 M1 placement / shard / launch metadata gate，per-rank result 和 package launch args 必须来自 IR。
- R4.2：实现真实 shard slicing、per-rank writeback/merge contract 和 no-comm runtime/package binding。

## P5 Transformer Local Vertical Slice

设计合同：

- M6 要让 static transformer local shard normalized 到 structured tensor IR，并让 group planner 对
  norm、softmax、attention、MLP 给出 accepted schedule 或明确拒绝原因。
- compute 覆盖至少包括 batch/head GEMM、reduce max/sum、elementwise add/sub/mul/div/max/min/neg/
  recip/sqrt/rsqrt/exp、limited broadcast、mask-add 或 compare/select。
- layout/SPM/DDR feasibility 要覆盖所有 accepted groups；constant/weight slice 要能追溯到
  `ConstantLike` + `wafer.load_tile` / storage transform。
- package/runtime metadata 要覆盖 block inputs/outputs、resident constants、workspace；当前无卡环境
  只能做 generated artifact compile，不能声称 device correctness。

当前实现：

- 有 norm、softmax、projection/residual、MLP acceptance passes 和 full local transformer block
  structured gate。
- 有 rank-4 QK^T / AV contraction lowering、same-shape/projected-permutation elementwise、local reduce
  和 batched GEMM skeleton lowering。
- M6 fixed smoke manifest 覆盖 workspace buffers、resident constants 和 82 个 ABI issue skeleton，
  C stub syntax compile 可过。

缺口：

- 多数 transformer pass 是 pattern acceptance，不是 full schedule planner。
- M6 package manifest 是 fixed fixture，不由 M6 IR 自动导出；workspace/resident constant metadata
  与 IR dataflow 无统一事实源。
- mask/select、dynamic shape、非 constant-init reduce、复杂 broadcast 和 true constant slicing/storage
  transform 未闭环。
- layout/SPM/DDR feasibility 未覆盖 full block 所有 accepted groups，只覆盖当前 skeleton 子图。
- 没有真实 full-block device artifact、runtime completion 或数值对比。

恢复任务：

- R5.1：恢复 M6 transformer local compile gate，让 workspace/resident constants/ABI issue sequence
  来自 full-block IR dataflow 和 lowering 输出。
- R5.2：补 transformer compute coverage gaps：mask/select、dynamic-bound policy、non-constant-init reduce、
  constant/weight slice 和 package consistency。

## P6 Communication / Tensor Parallel Path

设计合同：

- Shardy/SPMD 保留 logical collective；placement 提供 physical peer；`wafer.comm` 保留 collective /
  p2p semantics；p2p schedule 用 send/recv/wait/token/effect 显式表达。
- V0 只使用 fixed-size unicast Direct DTE；raw non-unicast DTE 需要单独 ABI 和板端验证。
- DTE resource allocator 要管理 DTE/FSM/packet/stream，comm buffer/staging 要进入 SPM/DDR demand，
  local drain、comm wait、group barrier 要分开。
- communication metadata 最终要进入 C ABI / runtime package，不能停在 isolated IR smoke。

当前实现：

- 有 `wafer.comm.send` / `recv` / `wait` verifier、placement peer check、non-empty wait check。
- 有 `wafer.abi.dte_send` / `recv` / `wait` skeleton 和 block-local resource tuple conflict check。
- 有 ring all-gather、reduce-scatter、all-reduce lowering 到 p2p + local elementwise accumulation。
- 有 StableHLO all_gather/all_reduce/reduce_scatter 到 `wafer.comm` collective-level op 的 normalization。
- 有 M2/M3/M4 integration skeleton gates。

缺口：

- DTE resource id 是单调 skeleton 分配，不是目标 DTE/FSM allocator。
- collective buffer slice / slot / address offset lowering 没有真实 descriptor 或 storage-realized buffer。
- StableHLO collective bridge 仍用 visible `unrealized_conversion_cast` 衔接 tensor/tile_buffer。
- SPM/DDR resource gates 没有和 communication staging / buffer lifetime 完整组合。
- communication plan metadata 没进入真实 package/runtime path；没有 Direct DTE board completion/error gate。

恢复任务：

- R6.1：恢复 communication design-conformance gate，补 DTE resource allocation、collective buffer
  slice/address offset、package/runtime metadata。
- R6.2：移除 StableHLO collective bridge 中的临时 visible cast，改由可验证 buffer-slice /
  layout/materialization 路径承接。

## 状态改写

| ID | 状态 | 理由 |
| --- | --- | --- |
| P0 | skeleton | 工程入口存在，但源码组织和依赖 ownership 未按设计收敛 |
| P1 | skeleton | 核心 op/type/verifier skeleton 存在，但 interface/effect/resource 和文件边界未完成 |
| P2 | skeleton | StableHLO textual lowering 有覆盖，但真实 importer artifact/sidecar/Shardy pipeline 未闭环 |
| P3 | skeleton | M0 local skeleton 可跑，但 group/resource/C ABI/package 主链路未闭环 |
| P4 | skeleton | placement/map 和 multi-tile skeleton 可跑，但真实 shard/merge/launch binding 未闭环 |
| P5 | skeleton | transformer staged acceptance 和 M6 smoke 可跑，但 full schedule/resource/package/device artifact 未闭环 |
| P6 | skeleton | comm/DTE skeleton 可跑，但 resource allocator、buffer slice/address、package/runtime metadata 未闭环 |

R0.1 之后的下一步是 R0.2：先恢复源码组织边界，再进入 P1/P2/P3 的具体实现恢复。
