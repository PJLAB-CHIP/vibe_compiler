# Wafer P0-P6 Recovery Status

日期：2026-05-26

状态：R0.1 完成记录；用于重写 P0-P6 历史未闭环项状态。

本文只记录 P0-P6 的设计合同、当前实现、缺口和恢复任务。它不声明 P0-P6 已完成，也不替代各
设计文档的语义边界。

## 依据

已重读并用于本轮状态重写的主文档：

- `tasks/2026-05-11-wafer-ai-compiler-architecture.md`
- `tasks/2026-05-12-wafer-group-design.md`
- `tasks/2026-05-13-wafer-design-docs-gap-review.md`
- `tasks/2026-05-21-wafer-layout-materialization-design.md`
- `tasks/2026-05-21-wafer-spm-bufferization-design.md`
- `tasks/2026-05-25-wafer-frontend-stablehlo-program-design.md`
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

P0-P6 只能保持 `骨架` 状态。当前代码已经证明一些局部 IR、verifier、pass 和 最小验证 能跑，
但没有满足架构文档按 IR stage 闭环的完成口径：

- R0.2 已把源码 ownership 拆到 `Frontend`、`IR`、`Analysis`、`Transforms`、`Conversion` 和 `ABI`
  边界；R1.1 之后又把 `wafer` dialect 内部 ODS、op verifier 和 dialect tests 按 IR 层收口。
- `WaferOps.td` 现在只作为 TableGen 聚合入口，具体 op 定义在 `include/Wafer/IR/{Tensor,Tile,Resource,Instr,Runtime}/*Ops.td`；
  `WaferDialect.cpp` 只保留 dialect/attr/type/op 注册和 `StorageType` verifier。
- 历史 local integration gate 曾把 `wafer-opt` IR FileCheck 和 fixed manifest/C stub fixture 放在同一
  测试文件里；这些 fixed emitter 和测试拼接已删除，当前 integration 只保留 IR pipeline coverage。
- 当前 local compile 还没有从 placed instruction-level IR 导出 C ABI emission metadata、
  IR-derived package manifest、LLVM dialect、LLVM IR、object、真实 `wafer_*` call、runtime adapter
  或板端 completion。

## P0 工程、依赖、组织

设计合同：

- 建立可配置、可测试的 MLIR 工程入口，包含 `wafer-opt`、lit/FileCheck、gtest 和固定依赖版本。
- StableHLO/Shardy/importer/runtime 依赖按层隔离；importer-only 依赖不能成为后端 textual tests 的硬依赖。
- 源码按架构文档第 7 节组织：`Frontend`、`IR`、`Analysis`、`Transforms`、`Conversion`、`ABI`、
  launch/runtime ownership 清晰；一个 `wafer` dialect namespace 内按 IR 层拆 ODS、verifier 和测试。
- 不把本机路径、checkout path、第三方 C++ API 名或临时 build override 写成 IR contract。

当前实现：

- 有 `CMakeLists.txt`、`cmake/third_party/WaferDependencyVersions.cmake`、`wafer-opt`、lit、gtest、
  `tools/check_deps.py` 和 bootstrap 脚本。
- 有 `WAFER_ENABLE_IMPORTER_DEPS` 开关、disabled importer 最小验证，以及 LLVM/MLIR、StableHLO、
  Shardy、OpenXLA/XLA、googletest public submodule checkout 和 OpenXLA stack 固定版本检查。
- 当前源码使用 `include/Wafer/Frontend`、`include/Wafer/IR`、`include/Wafer/Analysis`、
  `include/Wafer/Conversion`、`lib/Wafer/IR`、`lib/Wafer/Analysis`、`lib/Wafer/Transforms` 和
  `lib/Wafer/Conversion`；旧聚合 `WaferConversion` pass target 已删除，conversion 按 source/target
  IR contract 建 target。
- `WaferTransforms` 只注册当前仍成立的 transform pass；旧 C ABI issue-op lowering 和 `WaferConversion`
  pass target 已删除，后续按 placed instruction-level IR contract 重建。
- R0.3 后 core compiler、frontend/importer、runtime/driver 和 test tools 的 target 可见范围记录在
  `tasks/2026-05-26-wafer-dependency-layering-recovery.md`，并由 `tools/check_deps.py` 检查。
- Shardy/SDY 公共 dialect 与 import/export/propagation passes 已通过 Wafer 顶层 CMake shim 复用
  同一套固定版本 LLVM/MLIR/StableHLO 编译到 `shardy-sdy-opt`；不再以 Shardy standalone Bazel
  workspace 作为 Wafer dependency 编译验证。
- Wafer IR ODS 和 verifier 按层组织：`Tensor` 承载 group/tensor collective，`Tile` 承载
  tile-region/layout/compute/move/view/communication，`Resource` 承载 SPM/placement，
  `Instr` 承载 instruction/sync，`Runtime` 承载 launch；不再保留专门 ABI IR op family。
  `test/Dialect/Wafer` 按相同层级分目录。
- R1.2 后 `WaferTilingInterface`、`WaferLayoutOpInterface`、
  `WaferLayoutMaterializationOpInterface` 和 `WaferResourceEffectInterface` 不再只是 marker；
  group、layout/materialize、SPM、DDR、compute、comm、sync 和 ABI op 可通过接口查询边界值、
  layout requirement 和 resource effect。关键 movement/compute/comm op 也接入 MLIR
  `MemoryEffectOpInterface` 的 Wafer resource。
- R1.3 的历史 stage-connection gate 已在 2026-06-02 清理中删除；后续 group/tile/storage/C ABI
  连接必须由 R3/R6/R7 消费真实 frontend/SPMD program chain 后重新建立。

缺口：

- 工程能跑不等于 P0-P6 主路径完成；R2 已恢复 frontend program verifier 和 SDY program bridge，
  但 group/resource/package/runtime 主链路仍需后续 R3/P8 恢复。
- R1.2 恢复的是可查询合同和局部 legality；它还不是完整 planner/resource planning。closed-loop group
  search、SPM/DDR debug path、storage realization、runtime/package 主链路仍归 R3 之后恢复。
- StableHLO/Shardy dependency 当前主要服务 textual lowering 最小验证 和 SDY program bridge
  dependency boundary；R0.3 依赖栈用 PyTorch/XLA 2.5 的 `WORKSPACE` `xla_hash` 选择 OpenXLA/XLA，再由 XLA
  workspace 选择 LLVM/StableHLO/Shardy base。PyTorch/XLA source 也是后续构建/安装 `torch_xla`
  frontend importer runtime 的源码事实源；Wafer core compiler 代码还没有调用 `torch_xla`，它不能成为
  core compiler public dependency 或第二套 XLA/LLVM/StableHLO 事实源；Shardy 编译验证目标只证明公共
  SPMD 依赖可用，
  R2.2 只额外证明 Wafer 工具能接收 SDY program 并把 StableHLO replica group 显式传入
  `wafer.tile.*` communication。
- runtime/driver 头文件和真实 runtime adapter 尚未进入 launch/C ABI 层。

恢复任务：

- R0.2：已完成源码组织边界恢复；记录见
  `tasks/2026-05-26-wafer-source-organization-recovery.md`。
- R0.3：已完成依赖层级清单、检查和统一 Shardy CMake 编译验证目标；记录见
  `tasks/2026-05-26-wafer-dependency-layering-recovery.md`。
- R1.1：已完成 Wafer IR layer-based 文件边界拆分，并由 `tools/check_ir_organization.py` 检查。

## P1 Wafer IR Skeleton 和 Verifier

设计合同：

- `wafer.group` 只表达 tensor-level grouping/scheduling boundary，不携带 physical address、SPM
  offset、DTE resource、C ABI 或 planner trace。
- `wafer.tile.region` 表达 post-group device-side execution scope，承载 load/store、layout、
  compute、comm、sync/effect ordering，但不替代 launch。
- layout、SPM、DDR、compute、comm、sync、launch 都应有 op/type/interface/verifier 合同；共享字段只定义一次。
- op interface、type、effect 和 verifier 应暴露协议错误；pass 不能靠 side table 或名字匹配传语义。

当前实现：

- 有 `WaferAttrs.td`、`WaferTypes.td`、`WaferInterfaces.td` 和 `WaferOps.td`，覆盖 target、
  memory space、mem layout、placement、wait policy、elementwise/reduce kind、`!wafer.storage`
  以及 group/tile/layout/load/store/placement/ABI/compute/comm/sync/launch temporary ops。
- 有 parser/printer/verifier 正负例；`WaferDialect.cpp` 实现多数 verifier。
- 有 `WaferTilingInterface`、`WaferLayoutOpInterface`、
  `WaferLayoutMaterializationOpInterface`、`WaferResourceEffectInterface` 和 Wafer resource-backed
  MLIR memory effects；接口查询覆盖 group 边界、layout/SPM/DDR、compute/comm/sync/ABI 的局部需求。
- 历史 local compute stage-connection lit gate 已删除；当前不再用 public linalg 手写 source
  拼旧 group/tile/C ABI issue-op pass 链。

缺口：

- ODS、op verifier 和 dialect tests 已按 IR 层拆文件；共享 verifier helper 集中在
  `lib/Wafer/IR/Common/OpVerifierUtils.*`。
- interface/resource/effect 已能作为 planner/verifier 的结构化查询入口，但还没有被 R3 的
  closed-loop group planner、SPM allocation / DDR resource planning 和 placed instruction-level lowering
  全量消费。
- Wafer dialect verifier 的孤立负例仍会使用 `builtin.unrealized_conversion_cast` 构造非法边界值；
  这些用例只能证明 verifier 形态。历史 R1.3 stage-connection gate 已删除；communication bridge
  的 visible cast 仍归 R6.2 清理。
- 没有 placed instruction/storage memref/descriptor 层，因此 `!wafer.storage` 还没有按设计消失。

恢复任务：

- R1.1：已完成；ODS、C++ verifier 和测试目录已按 IR 层组织，并保持一个 `wafer` dialect namespace。
- R1.2：已完成；interface/effect/resource 已从占位定义扩成 planner/resource/verifier 可调用的合同。
- R1.3：历史完成项；local compute stage-connection tests 已随旧 pass 链删除，后续由 R3 重新建立
  真实 program chain gate。

## P2 Frontend Program 和 Local Compute Normalization

设计合同：

- 稳定入口是 verified Wafer program，不是某个框架 API 或 Wafer
  私有伴随 JSON。
- frontend 保留 function signature、shape、dtype、dynamic bound、constant/weight 和 sharding annotation；
  不引入 Wafer SPM、DDR、layout materialization、runtime launch 或 private tensor constant op。
- local compute normalization 输出 `linalg` / `tensor` / `scf` / `arith` / `math` structured IR，保留
  dot、batch/head matmul、broadcast、reduce、softmax、norm、RoPE、MLP 的可验证 dataflow。
- pattern 只能来自 StableHLO semantics、types、indexing maps 和 SSA use-def，不能靠名字。

当前实现：

- StableHLO textual program tests 和 norm/softmax/projection/MLP frontend lowering fixtures 统一通过
  `wafer-lower-stablehlo-to-linalg` named pipeline 覆盖；历史 case-specific acceptance passes 和本地
  `wafer-lower-stablehlo-{dot,elementwise,reduce,shape}` / `wafer-normalize-constants` pass 均已删除。
- attention QK^T / AV rank-4 `dot_general` 由官方 StableHLO-to-Linalg conversion 转成 structured
  generic contraction；full local transformer fixture 只证明 fine-grained StableHLO dataflow 能进入
  structured tensor IR，不证明 group schedule、SPM residency 或 package completion。
- `wafer-compile-stablehlo` 已通过 `WaferFrontend` verifier 接收 pre-exported Wafer program，
  并覆盖 graph break、eager fallback、bounded dynamic shape 诊断；P2.F1 进一步验证 PyTorch/XLA
  program directory `forward.meta` / pre-SPMD `data/<parameter>` 与 MLIR function signature 一致；post-SPMD
  `forward.parameter_shards.json` 与 rank-local shard payload 只有 verifier fixture 覆盖，真实产物
  等 P2.S2 生成。
- `WAFER_ENABLE_SPMD_PARTITIONER_DEPS=ON` 时，`wafer-opt` / `wafer-compile-stablehlo`
  能注册 SDY dialect；
  `sdy.mesh` / `sdy.sharding` program 可被 parse/verify。历史上曾有 StableHLO `replica_groups`
  到 `wafer.tile.*` communication `rank_group` 的 bridge；该路线已退出主线，只作为已删除路径的覆盖记录，
  不再作为 group/tiling 前的 collective handoff。

缺口：

- R2.1/R2.2 已恢复 pre-exported program adapter/verifier 和 SDY program bridge；P2.F1 已补
  source-built PyTorch/XLA program directory capture；P2.S1 已补真实 framework `mark_sharding`、default input
  seed policy 和 Wafer Shardy propagation stage gate。旧 Python post-SPMD export 入口已删除；仍需
  P2.S2 建立 Wafer-owned XLA SPMD partition compiler stage，之后 R2.4 再把 post-SPMD collective
  handoff 成 Wafer LinalgExt-style tensor collective IR。
- frontend fixtures 只覆盖当前 structured IR dataflow，不等于 group schedule planner 或 full
  transformer local compile 已完成。
- mask/select、dynamic shape、非 constant-init reduce、复杂 broadcast、常量 slicing/storage transform
  仍未按设计闭环。

恢复任务：

- R2.1：已完成；记录见 `tasks/2026-05-26-wafer-r2-recovery.md`，包含 frontend program verifier、
  dynamic bound 和 diagnostics。
- R2.2：已完成；记录见 `tasks/2026-05-26-wafer-r2-recovery.md`，logical mesh/sharding program
  可进入工具链。StableHLO replica group materialize 为 `wafer.tile.*` communication `rank_group` 的历史实现只作为
  已删除路线覆盖记录，不作为后段 placement/ring 或 group/tiling 输入。
- R2.3：已完成；记录见 `tasks/2026-05-26-wafer-r2-recovery.md` 和
  `tasks/2026-05-25-wafer-local-compute-normalization-design.md`，local compute coverage 按
  op/dataflow contract 重写，不把 acceptance pass 当 schedule completion。
- P2.F1：已完成；`tools/build_pytorch_xla_runtime.py` 从 `third_party/pytorch-xla` 源码构建/安装
  `torch_xla` 2.5.0，并通过 Bazel override 使用本仓库 `third_party/xla` / `third_party/llvm-project`；
  `test/Tools/Inputs/wafer_pytorch_xla_capture.py` 作为 test program directory generator 调用 `torch.export.export` 和
  `torch_xla.stablehlo.exported_program_to_stablehlo`，为 `4096x4096 @ 4096x4096` f32 matmul +
  bias + tanh + residual 生成 PyTorch/XLA StableHLO program directory，包含 `functions/forward.mlir`、
  `functions/forward.meta`、`functions/forward.bytecode` 和 `data/weight` / `data/bias`；完成证明已跑通真实
  source-built PyTorch/XLA adapter -> program directory -> verifier 链。prebuilt `torch_xla` wheel、手写
  MLIR 或手写 emitter 仍不能作为后续完成证明。
- P2.S1：骨架；`test/Tools/Inputs/wafer_pytorch_xla_capture.py` 用 source-built PyTorch/XLA lazy SPMD runtime
  的 `mark_sharding` 生成六种用户 sharding 策略的真实 PyTorch/XLA StableHLO program directory，program directory 继续由
  frontend verifier 和 Wafer Shardy propagation gate 消费。完全没有用户 seed 时，P2.S1 在 SPMD
  层补默认 single-card function-input sharding seed（默认 16 tile，调试 1 tile，找不到合适切分
  维度时 replicated）。P2.S1 不生成 partitioned StableHLO、rank-local shard payload、
  `wafer.spmd.*`、私有 sharding JSON 或名字约定。P2.S2 必须接上 XLA SPMD partitioner 并产出
  partitioned / replicated-local program；R3.1 依赖 P2.F1/P2.S1/P2.S2/R2.4 提供真实
  frontend/SPMD program 来源和 tensor collective handoff。

## P3 Single-Tile Local Compute

设计合同：

- Single-tile local compute 必须验证 StableHLO/Linalg -> `wafer.group` -> `wafer.tile.region` -> layout/SPM/DDR ->
  `wafer.tile.*` compute/movement -> C ABI/package 的单 tile 闭环。
- `wafer.group` planner 需要 closed-loop search：tile shape、op tiling interface、layout materialization、
  SPM allocation、DDR demand/bandwidth、compute/movement legality 都要作为同一候选的 decision input。
- layout materialization 是真实 movement；SPM allocator 要用 liveness/effect/range/end-address；
  DDR 要有 external binding、workspace、resident constant、pool/domain/capacity/bandwidth。
- C ABI gate 要固定 `wafer_*` 参数单位和 wait policy，并通过 wrapper/golden packet 覆盖；package 要来自
  当前 lowering 输出。

当前实现：

- 2026-06-03 后，Integration 主链路不再注册 C ABI named pipeline。用户级主线统一到
  `wafer-opt --program-pipeline=stablehlo-spmd*`；`WaferPipelines` 只保留内部/局部
  `wafer-propagate-stablehlo-sharding` 和 `wafer-lower-stablehlo-to-linalg`；
  `wafer-lower-linalg-to-cabi`、`wafer-lower-stablehlo-to-cabi`、
  `wafer-lower-tile-communication-to-cabi` 以及
  `wafer-compile-stablehlo --compile-stablehlo-program-to-cabi` 已删除，避免把 R3/R6/R7 的 pending
  skeleton 写成用户级 compile flow。
- 旧 `wafer-form-groups`、`wafer-materialize-single-tile`、`wafer-compact-layout-assignment`、
  `wafer-check-spm-allocation`、`wafer-materialize-ddr-external-bindings`、
  `wafer-lower-ring-*` 和 `wafer-lower-tile-region-to-c-abi` unit/debug pass 链已删除。
- 历史上有 `wafer.instr.rdma`、`wafer.instr.wdma`、`wafer.instr.ne.gemm`、elementwise/reduce ABI issue ops；
  当前主线不把它们作为必经 IR 层。
- 有 `Wafer/ABI/TileAbi.h` descriptor unit tests、manifest validator 和 generated C stub syntax compile。

缺口：

- group formation、root tile planning、SPM allocation、DDR demand 和 ring collective lowering 的旧
  unit pass 已删除；后续必须从真实 program chain 恢复 closed-loop planner / resource planning。
- SPM/DDR 仍未基于完整 interface demand、async lifetime、placed instruction/storage memref/descriptor、
  workspace、resident constants、pool/domain、bandwidth 或 runtime binding 建立主线实现。
- C ABI 仍只有 descriptor builder / stub 工具覆盖，尚未从 placed instruction-level IR 生成真实
  `wafer_*` call、LLVM lowering 或 wrapper-to-register golden packet。
- fixed manifest emitter 已删除；IR-derived package manifest 仍未由当前 `wafer-opt` output 自动导出。

恢复任务：

- R3.1：恢复 logical group boundary contract，保证 `wafer.group` 只表达 local tensor grouping
  和 group boundary，并按 group 设计完成 dependency-preserving conservative expansion。
- R3.2：恢复 root tile planning，把 op tiling、layout、SPM、DDR 和 compute/movement
  legality 接到同一 group planning decision。
- R3.3：恢复 tile_region materialization contract，只把 accepted group materialize 成
  `wafer.tile.region`。
- R3.4：恢复 layout/SPM materialization gate，让 layout materialization 和 SPM allocation 由 effect、
  liveness/range 和 storage lifetime 驱动。
- R3.5：恢复 DDR/resource demand gate，覆盖 external binding、workspace、resident constant、
  pool/domain、capacity/bandwidth demand。
- R3.6：恢复 C ABI / packet emission gate，让 ABI 参数单位和 wait policy 从 placed
  instruction-level IR 派生；不再保留专门 ABI IR op family 作为主线或 debug layer。
- R3.7：恢复 package manifest gate，manifest、C stub 和 launch signature 从当前 `wafer-opt`
  输出导出。
- R3.8：恢复 placed instruction-level IR / C ABI / golden packet 边界，至少让 RDMA/WDMA/GEMM 有真实
  wrapper-facing call contract 和 golden packet 对照。

## P4 Multi-Tile No Communication

设计合同：

- Placement 负责 logical rank 到 physical card/tile coordinate、good-tile/PG、block id、local shard
  metadata；不负责 SPM offset、DTE schedule 或 runtime API 细节。
- Multi-tile no-comm 需要每个 tile 处理自己的 local shard、写回对应 output slice，并由 package /
  generated program 区分 per-tile args。
- `wafer.tile.region` / launch metadata 必须表达 per-tile identity 和 shard slice；不能用 whole-tensor
  clone 代替 shard。

当前实现：

- 有 `wafer.placement.map` verifier，检查 rank count、topology bounds、bad tile 和 duplicate tile。
- `wafer-materialize-multi-tile-no-comm` 已删除；旧实现只是按 placement rank 数 clone whole-tensor
  tile_region，并用最后一个结果替换 group 结果，不能作为 multi-tile no-comm 主线证据。
- package fixture / C stub 可以生成 per-tile launch arg table 和 placement metadata。

缺口：

- multi-tile materialization 克隆 whole-tensor load-GEMM-store tile_region；没有真实 shard slicing。
- group result 用最后一个 tile_region result 替换，没有 per-rank output merge 或 host-side shard readback。
- block id、physical coord、local shard metadata 没进入 tile_region body 或真实 launch binding。
- manifest placement metadata 是 fixed 最小验证 fixture，不来自 placement op + lowering 输出。
- runtime launch completion 仍未实现，不能证明多 tile 并行或 per-tile args 被真实消费。

恢复任务：

- R4.1：恢复 placement map / capability gate，accepted placement 来自 logical rank、good-tile/PG
  capability 和 physical coordinate verifier。
- R4.2：恢复 per-rank identity / launch metadata gate，把 logical rank、block id、physical coord
  和 local shard metadata 接到 tile_region/launch/package 边界。
- R4.3：恢复 shard slicing materialization，multi-tile no-comm 为每个 rank materialize 自己的
  input/output slice，不再 clone whole tensor。
- R4.4：恢复 per-rank writeback / merge contract，明确 sharded output、host-side readback 或
  output merge 的 IR/package 责任。
- R4.5：恢复 package / generated program gate，让 placement metadata、local shards、per-rank
  launch args 和 issue sequence 从当前 lowering 输出导出。

## P5 Transformer Local Vertical Slice

设计合同：

- Transformer local compile 要让 static transformer local shard normalized 到 structured tensor IR，并让 group planner 对
  norm、softmax、attention、MLP 给出 accepted schedule 或明确拒绝原因。
- compute 覆盖至少包括 batch/head GEMM、reduce max/sum、elementwise add/sub/mul/div/max/min/neg/
  recip/sqrt/rsqrt/exp、limited broadcast、mask-add 或 compare/select。
- layout/SPM/DDR feasibility 要覆盖所有 accepted groups；constant/weight slice 要能追溯到
  `ConstantLike` + `wafer.tile.load` / storage transform。
- package/runtime metadata 要覆盖 block inputs/outputs、resident constants、workspace；当前无卡环境
  只能做 generated program compile，不能声称 device correctness。

当前实现：

- 有 norm、softmax、projection/residual、MLP frontend lowering fixtures 和 full local transformer
  block structured fixture；旧 transformer-specific acceptance passes 已删除。
- 有 rank-4 QK^T / AV contraction lowering、same-shape/projected-permutation elementwise、local reduce
  和 batched GEMM issue lowering。
- 旧 local-transformer fixed manifest / C stub gate 已删除；当前 transformer integration 只覆盖
  StableHLO -> structured tensor/local compute pipeline，不覆盖 placed instruction-level IR 或 C ABI emission。

缺口：

- transformer frontend fixtures 只证明 structured tensor dataflow 覆盖，不是 full schedule planner。
- transformer package manifest 尚未由 transformer IR 自动导出；workspace/resident constant metadata
  与 IR dataflow 仍无统一事实源。
- mask/select、dynamic shape、非 constant-init reduce、复杂 broadcast 和 true constant slicing/storage
  transform 未闭环。
- layout/SPM/DDR feasibility 未覆盖 full block 所有 accepted groups，只覆盖当前局部子图。
- 没有真实 full-block device program、runtime completion 或数值对比。

恢复任务：

- R5.1：恢复 transformer local 编译验证，让 workspace/resident constants 和 C ABI emission metadata
  来自 full-block IR dataflow 和 lowering 输出。
- R5.2：补 transformer compute coverage gaps：mask/select、dynamic-bound policy、non-constant-init reduce、
  constant/weight slice 和 package consistency。

## P6 Communication / Tensor Parallel Path

设计合同：

- Shardy/SPMD 保留 logical collective；placement 提供 physical peer；`wafer.tile.*` communication 保留 collective /
  p2p semantics；p2p schedule 用 send/recv/wait/token/effect 显式表达。
- 当前已验证 compiler data plane 只使用 fixed-size unicast Direct DTE；raw non-unicast DTE 需要单独
  ABI 和板端验证。未验证 raw non-unicast 不能成为 logical collective 不支持的理由，collective 可
  先由 unicast p2p schedule 组合表达。
- DTE resource allocator 要管理 DTE/FSM/packet/stream，comm buffer/staging 要进入 SPM/DDR demand，
  local drain、comm wait、group barrier 要分开。
- communication metadata 最终要进入 C ABI / runtime package，不能停在 isolated IR 最小验证。

当前实现：

- 有 `wafer.tile.send` / `recv` / `wait` verifier、placement peer validation、non-empty wait validation。
- 历史上有 旧 ABI debug op `dte_send` / `recv` / `wait` issue op 和 block-local resource tuple conflict
  validation；当前主线应恢复 placed Direct DTE instruction form 和 C ABI emission。
- 有 ring all-gather、reduce-scatter、all-reduce lowering 到 p2p + local elementwise accumulation。
- 旧的 StableHLO all_gather/all_reduce/reduce_scatter 到 `wafer.tile.*` communication collective-level op 的
  normalization 已移除；R2.4 主线已恢复 StableHLO -> Wafer LinalgExt-style tensor collective
  handoff，R6 仍需恢复 tiled tensor collective -> `wafer.tile.*` communication materialization。
- 有 p2p/ring communication fixture 验证；multi-replica collective 需按新的 tensor collective
  handoff 重新建立。

缺口：

- DTE resource id 是单调 placeholder 分配，不是目标 DTE/FSM allocator。
- collective buffer slice / slot / address offset lowering 没有真实 descriptor 或 placed storage buffer。
- tiled tensor collective -> `wafer.tile.*` communication materialization 尚未实现；R6.2 需要在 tile_region / SPM
  materialization 之后补可验证 buffer-slice / layout/materialization 路径。
- SPM/DDR resource 验证没有和 communication staging / buffer lifetime 完整组合。
- communication plan metadata 没进入真实 package/runtime path；没有 Direct DTE board completion/error 验证。

恢复任务：

- R6.1：恢复 communication design-conformance gate，补 DTE resource allocation、collective buffer
  slice/address offset、package/runtime metadata。
- R6.2：移除 StableHLO collective bridge 中的临时 visible cast，改由可验证 buffer-slice /
  layout/materialization 路径承接。

## 状态改写

| ID | 状态 | 理由 |
| --- | --- | --- |
| P0 | 骨架 | 工程入口存在，源码 ownership、依赖层级边界和 IR layer-based 文件边界已由 R0.2/R0.3/R1.1 恢复；仍不代表 frontend/runtime/package 主路径完成 |
| P1 | 骨架 | 核心 op/type/verifier 基础实现存在，interface/effect/resource 已恢复；历史 local compute stage-connection gate 已删除，placed instruction-level 主链路和 communication cast bridge 仍未闭环 |
| P2 | 骨架 | StableHLO textual lowering、frontend program verifier、PyTorch/XLA program directory capture、SDY program bridge、P2.S1 Shardy propagation stage gate、P2.S2 Wafer-owned SPMD partition 和 R2.4 tensor collective handoff 已恢复；dynamic/mask/constant-storage、group/resource/package 和 physical schedule 仍未闭环 |
| P3 | 骨架 | 旧 single-tile local unit path 已删除；group/resource/C ABI/package 主链路未闭环 |
| P4 | 骨架 | placement/map 和 multi-tile fixture 可跑，但真实 shard/merge/launch binding 未闭环 |
| P5 | 骨架 | transformer staged frontend fixtures 和 local fixture 可跑，但 full schedule/resource/package/device program 未闭环 |
| P6 | 骨架 | comm/DTE fixture 可跑，但 resource allocator、buffer slice/address、package/runtime metadata 未闭环 |

R2.4 已建立 Wafer LinalgExt-style tensor collective handoff；当前下一步仍是 R3.1，补完整
group boundary / conservative expansion，再进入 root tile planning 和 tile_region
materialization 主链路。
