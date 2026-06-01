# Wafer Compiler Progress

更新时间：2026-06-01

本文件只记录当前看板、状态和下一步；不再承载实现流水账。历史实现细节以 git commit、设计文档
和审计文档为准。

## 当前状态

- P0-P6 的历史实现只能视为未闭环局部进度（状态仍标记为 `骨架`）：有局部 IR、pass、verifier、fixture 和最小验证，
  但没有按各设计文档完成主路径闭环。
- R0.2/R0.3/R1.1/R1.2/R1.3/R2.1/R2.2/R2.3 已恢复源码 ownership、依赖层级边界、统一 Shardy
  CMake 编译验证目标、Wafer IR op-family 文件边界、Wafer interface/effect/resource 查询合同、
  local compute stage-connection gate、frontend artifact verifier、SDY artifact bridge，以及 local
  compute normalization 覆盖状态。历史 StableHLO replica rank group 到后段 `wafer.comm` 的 bridge
  只作为已删除路线的覆盖记录，不作为当前主线入口。
  P2.F1 已补 source-built PyTorch/XLA capture adapter；P2.S1 已补 framework `mark_sharding`
  pre-partition artifact、no-user 默认 input seed policy 和 Wafer Shardy propagation stage gate。
  过去用 PyTorch/XLA post-SPMD export 生成 partitioned StableHLO 的测试入口已删除，不能再作为
  Wafer-owned SPMD 主链完成依据。
- 设计一致性缺口见
  `tasks/2026-05-26-wafer-p0-p6-design-conformance-audit.md` 和
  `tasks/2026-05-26-wafer-p0-p6-recovery-status.md`。
- framework-specific capture 和 Shardy/SPMD artifact pipeline 不是放弃项；真实前端 artifact 来源
  必须先于 group / tile / resource 主链路恢复。P2.S1 已把 sharded 分支从手写 artifact fixture
  收回到真实 framework mark 和 Wafer Shardy propagation stage，但还没有完成 Wafer-owned
  XLA SPMD partition artifact stage。
- 2026-06-01 复查结论：P2.S1 之前把 sharding facts normalize 成 `wafer.spmd.*` 的路线不是主线
  合同；把 PyTorch/XLA test helper 直接导出 post-SPMD local body 当成 Wafer 主链也不正确。
  正确主链必须是 `frontend export -> StableHLO/SDY bundle -> Wafer Shardy propagation ->
  Wafer-owned XLA SPMD partition artifact stage -> partitioned or replicated-local StableHLO`。
  post-SPMD collective 后续先进入 Wafer LinalgExt-style tensor collective 层，再参与 group/tiling，
  `wafer.comm` 后移到 `wafer.tile_region` / SPM materialization 之后。没有用户 sharding seed 时，
  默认 16 tile / 1 tile 调试策略也属于 SPMD 层，不能放到 placement/group 后段。
- 当前不得推进 P7/P8/P9；必须先恢复 P0-P6 的设计一致性。
- R2.4-pre 已完成库级 Wafer named pipeline / driver contract：`WaferPipelines` 注册
  `wafer-propagate-stablehlo-sharding`、`wafer-lower-stablehlo-to-linalg`、
  `wafer-lower-linalg-to-cabi`、`wafer-lower-stablehlo-to-cabi` 和
  `wafer-lower-tile-communication-to-cabi`。`wafer-propagate-stablehlo-sharding`
  在 SPMD 层应用 no-user 默认 input seed 并调用 Shardy propagation；它不是 XLA SPMD
  partitioner。`wafer-lower-stablehlo-to-cabi` 明确组合 StableHLO->Linalg 与 Linalg->C ABI
  两层。`wafer-import-model --propagate-stablehlo-sharding` 是 sharding propagation 阶段检查入口，
  从 bundle verifier 进入同一条 Shardy propagation pipeline；用户级 compile 入口
  `wafer-import-model --compile-stablehlo-bundle-to-cabi` 从 verified local/partitioned StableHLO
  bundle 进入 StableHLO->C ABI pipeline，并拒绝带 pre-SPMD sharding
  seed 但没有 post-SPMD marker 的 bundle，防止绕过 Shardy/XLA SPMD。当前没有 Wafer-owned
  XLA SPMD partitioner stage，也不保留 Python post-SPMD helper。
  用户级 compile target 名称统一为 `wafer`；`tx8` 只保留为硬件/依赖逆向资料中的事实名，不作为
  compiler driver target 字符串。
- 当前 ready 项是 P2.S2：建立 Wafer-owned SPMD partition artifact stage，消费 Wafer Shardy
  propagation 输出并调用 XLA SPMD partitioner 产出 partitioned StableHLO bundle。R2.4 tensor
  collective handoff 依赖这条 artifact stage 接上后再推进。

## 状态标记

- `ready`：可以直接开始实现。
- `pending`：依赖前序任务完成。
- `later`：当前主线之后再做。
- `骨架`：历史 fixture gate 通过，但不代表设计文档主路径完成。
- `done`：实现和对应验证已经按设计合同完成；P2.F1 之后的相关任务还必须有真实图 artifact chain 的
  端到端消费证明。

## 当前验证口径

当前本地可验证范围：

- CMake / `wafer-opt` / lit / gtest 构建入口。
- MLIR textual tests、FileCheck、verifier negative tests。
- 固定依赖版本 / importer backend 隔离检查。
- Wafer tiling/layout/materialization/resource op interface 查询和 MLIR memory effect resource 查询。
- 从 public linalg source 到 group、tile_region、`wafer.abi.*` issue stage-connection lit gate。
- frontend artifact verifier 的 graph-break/eager fallback/bounded dynamic shape gate，以及 PyTorch/XLA
  bundle `forward.meta` / pre-SPMD `data/<parameter>` / post-SPMD
  `forward.parameter_shards.json` + `parameter_shards/<parameter>/rank_XXXXX.npy` verifier 负例/最小 fixture。
- SDY `sdy.mesh` / `sdy.sharding` artifact parse/verify gate。历史 StableHLO replica `rank_group`
  到后段 `wafer.comm` / ring lowering 的 lit gate 只保留为非主线覆盖记录。
- `test/Spmd` 当前只覆盖 default input seed 和 SDY/Shardy artifact parse/verify，不覆盖 XLA SPMD
  partitioner，也不输出 rank-local StableHLO。`test/Frontend` 当前覆盖 StableHLO/Linalg local
  compute normalization；softmax、RMSNorm、LayerNorm 是 fine-grained StableHLO staged graph 到
  `linalg.reduce` / `linalg.generic` 的 lowering，不是 high-level softmax/norm op 或 SPMD artifact gate。
- C ABI issue ops、package manifest fixture 和 C stub syntax compile。
- P2.F1 主链路已能从 source-built PyTorch/XLA capture adapter 产出 PyTorch/XLA StableHLO bundle，
  并通过 bundle metadata / data verifier。P2.S1 当前覆盖真实 framework `mark_sharding`
  pre-partition artifact、Wafer Shardy/SDY propagation pipeline gate、no-user 默认 `tile-count=16`
  input seed policy 和 `tile-count=1` 调试模式；不再保留 Python post-SPMD helper；
  不生成 `wafer.spmd.*`、私有 JSON、名字约定或临时 module attrs 作为长期 sharding / per-rank
  协议。R3 / R4 / R5 / R6 要逐步消费同一条修正后的 artifact chain。手写 MLIR fixture 只作为
  verifier / unit 测试，不能单独作为主链路完成证明。
- P2.F1 之后每个相关任务标记 `done` 前，都必须新增或更新一条真实 artifact chain gate：从真实
  framework/exporter 图 artifact 进入，重放已有上游链路，并验证本任务边界新增的 IR fact /
  verifier fact / artifact fact 可正确导出和被直接消费。单层 FileCheck、手写 fixture 或静态
  manifest 只能作为补充测试。
- R2.4-pre 已固定主链路 gate 的 named pipeline / driver 入口：pre-SPMD bundle 可通过
  `wafer-import-model --propagate-stablehlo-sharding` 运行 default input seed + Shardy propagation；
  post-SPMD partitioned bundle 或明确 replicated-local StableHLO bundle 通过用户级 compile
  入口 `wafer-import-model --compile-stablehlo-bundle-to-cabi` 进入编译；当前缺口是 P2.S2 还没有
  产出这类 bundle。`wafer-opt` 层的
  Shardy propagation、StableHLO->Linalg、Linalg->C ABI、StableHLO->C ABI 和 tile-level
  communication->C ABI gate 都通过库中注册的 Wafer pipeline 重放。pipeline option
  `target=wafer` 会 materialize/校验 `#wafer.target<wafer>`，非 `wafer` target 会硬诊断。单 pass
  flag 只作为 unit/debug 入口；pipeline 名称按 IR 边界和职责命名，不能按 P2/R3 任务号、单个 case
  或 workload 命名。
- Pass / driver 分层边界：frontend Python 只导出 reference 或 pre-SPMD sharded bundle；
  `wafer-propagate-stablehlo-sharding` 只做 default seed + Shardy propagation；P2.S2 才调用 XLA
  SPMD partitioner 并写 partitioned bundle / parameter shard binding；R2.4 才把 partitioned
  StableHLO collective 转成 tensor-level collective；`wafer-lower-stablehlo-to-linalg` 只做 local
  compute normalization，不承载 sharding propagation 或 partition。
- 真实 artifact chain 有两个 SPMD 输入分支：有用户 sharding seed 时保留用户 seed 并进入
  Wafer Shardy propagation；完全没有用户 seed 时由 P2.S1 默认 policy 补 function-input
  sharding seed 后进入同一 propagation pipeline。P2.S2 再消费 propagation 输出并调用 XLA SPMD
  partitioner。no-user-sharding 分支不能因为缺少用户
  `sdy.sharding` 被 gate 掉，也不能绕开 SPMD 到后段补通信。
- 当前下游 lowering / placement / runtime 未完善，不能反向成为上游“不支持”的语义边界。如果
  Shardy/SPMD、frontend 或 planner 产出的合法语义在 Wafer 硬件/通信/存储模型上可表达，但当前
  IR 或 lowering 尚未覆盖，恢复任务必须补对应 IR contract、verifier 或后续 lowering 任务；只有
  artifact 自身非法，或目标硬件/ABI 证据明确无法表达时，才允许在该层拒绝。

当前不能作为完成证明：

- fixed fixture manifest 不能证明 package 来自当前 `wafer-opt` lowering 输出。
- C stub syntax compile 不能证明 LLVM IR、object、真实 runtime call 或 wrapper/packet lowering。
- 本地验证不能替代板端 launch、device completion、数值对比或 PMU/profiling。

## 历史未闭环状态

这些条目保留为历史局部进度索引，不作为设计完成声明。

| ID | 状态 | 范围 | 当前结论 |
| --- | --- | --- | --- |
| P0 | 骨架 | 工程、依赖、工具、测试入口 | 最小工程入口可用；源码 ownership 和依赖层级边界已恢复；P0 仍只是历史局部记录，不代表 frontend/runtime 主路径完成 |
| P1 | 骨架 | Wafer IR verifier | 核心 op/type/attr 基础实现有测试；ODS/verifier/tests 已按 op family 拆开；interface/effect/resource 已恢复为可查询合同；local compute stage-connection gate 已补，但 storage-realized 主链路仍未闭环 |
| P2 | 骨架 | Frontend artifact 和 local compute normalization | StableHLO textual lowering、frontend artifact verifier、PyTorch/XLA bundle metadata/resource-backed parameter gate、source-built PyTorch/XLA capture adapter、SDY artifact bridge、P2.S1 Shardy propagation stage gate 和 local compute coverage 口径已恢复；P2 仍不代表 Wafer-owned SPMD partition、R2.4 tensor collective handoff、dynamic/mask/constant-storage 或 physical schedule 闭环 |
| P3 | 骨架 | single-tile local compute | 有 group/tile/SPM/DDR/C ABI issue；planner、package、golden packet 和 C ABI 主路径未闭环 |
| P4 | 骨架 | multi-tile no-communication local compute | 有 placement/map 和 clone-style tile_region fixture；真实 shard slicing、merge、runtime launch metadata 未闭环 |
| P5 | 骨架 | transformer local vertical slices | 有 staged pattern acceptance gate；full-block package/device artifact 未从 IR 闭环 |
| P6 | 骨架 | Communication / tensor parallel path | 有 p2p/ring/DTE fixture；buffer slicing、address lowering、resource allocator、package/runtime metadata 未闭环 |

## P0-P6 设计一致性恢复队列

目标：把历史局部进度重新对齐到各设计文档的主路径合同；在这些任务完成前，不进入 P7。

| ID | 状态 | 任务 | 验收 |
| --- | --- | --- | --- |
| R0.1 | done | 逐项重读 P0-P6 对应设计文档并重写任务状态 | 记录见 `tasks/2026-05-26-wafer-p0-p6-recovery-status.md`；每个历史局部项都有“设计合同 / 当前实现 / 缺口 / 恢复任务” |
| R0.2 | done | 恢复源码组织边界 | 记录见 `tasks/2026-05-26-wafer-source-organization-recovery.md`；IR / Frontend / Transforms / Conversion / ABI ownership 已拆开，C ABI issue 不再归属 `WaferTransforms` |
| R0.3 | done | 补依赖层级清单 | 记录见 `tasks/2026-05-26-wafer-dependency-layering-recovery.md`；R0.3 只完成依赖源码拉取、版本固定、层级隔离和编译验证，不表示这些依赖都已经被 Wafer 代码实际调用；`third_party/pytorch-xla` 用于确定 XLA 版本，也是编译/安装 `torch_xla` importer runtime 的源码事实源；P2.F1 已在此基础上完成 source-built PyTorch/XLA capture adapter；`third_party/xla` 仍只作为后续 GSPMD 集成的源码和版本来源，当前不把 XLA/GSPMD 目标编进 Wafer core target；已通过 Wafer CMake 实际编译验证的是同一套 LLVM/MLIR、StableHLO 和 Shardy/SDY 公共 dialect/pass，输出 `shardy-sdy-opt`；core compiler target 不 public link Shardy/XLA/PyTorch-XLA；target 可见范围、固定版本一致性和 Shardy 编译验证目标由 `tools/check_deps.py` 检查 |
| R1.1 | done | 拆分 Wafer IR 文件边界 | `WaferOps.td` 只聚合 `include/Wafer/IR/Ops/*Ops.td`；op verifier 已拆到 `lib/Wafer/IR/Ops/*Ops.cpp`；`test/Dialect/Wafer` 已按 ABI/Attrs/Comm/Compute/DDR/Group/Launch/Layout/Placement/SPM/Sync/TileRegion/Types 分目录；`tools/check_ir_organization.py` 和 lit test 固定结构检查 |
| R1.2 | done | 恢复 interface/effect/resource 合同 | `WaferTilingInterface`、`WaferLayoutOpInterface`、`WaferLayoutMaterializationOpInterface` 和 `WaferResourceEffectInterface` 都提供结构化查询；layout/materialize/SPM/DDR/compute/comm/sync/ABI op 接入 Wafer resource effects，关键 movement/compute/comm op 接入 MLIR memory resource effects；gtest 覆盖 planner-style 查询 |
| R1.3 | done | 补 stage-connection tests | `test/StageConnections/local-compute-pipeline.mlir` 覆盖 linalg matmul/elementwise/reduce 到 group、tile_region、`wafer.abi.*` issue 层连接；`tools/check_stage_connection_tests.py` 固定禁止 stage-connection gate 退回 cast-only 用例 |
| R2.1 | done | 恢复 frontend artifact / importer contract | 记录见 `tasks/2026-05-26-wafer-r2-recovery.md`；`WaferFrontend` verifier、`wafer-import-model --verify-import-result`、`wafer.frontend.dynamic_bounds`、graph break/eager fallback/dynamic shape 诊断已闭环；P2.F1 进一步用 PyTorch/XLA bundle metadata 替代自定义 JSON 作为真实 capture 产物口径 |
| R2.2 | done | 恢复 Shardy/SPMD artifact bridge | 记录见 `tasks/2026-05-26-wafer-r2-recovery.md`；SDY dialect/pass 注册按 `WAFER_ENABLE_SPMD_PARTITIONER_DEPS` 隔离，`sdy.mesh`/`sdy.sharding` artifact 可进入工具链；历史 StableHLO `replica_groups` 到 `wafer.comm` `rank_group` 的 bridge 仅作为已删除路线的覆盖记录，不能作为 group/tiling 前的主线 collective 表示 |
| R2.3 | done | 重写 local compute normalization 覆盖状态 | 记录见 `tasks/2026-05-26-wafer-r2-recovery.md` 和 `tasks/2026-05-25-wafer-local-compute-normalization-design.md`；dot/broadcast/reduce/softmax/norm/RoPE/MLP 覆盖按 structured tensor IR evidence 记录，acceptance pass 不作为 schedule completion |
| P2.F1 | done | 建立 framework-specific capture adapter contract | `tools/build_pytorch_xla_runtime.py` 从 `third_party/pytorch-xla` 源码构建/安装 `torch_xla` 2.5.0，并通过 Bazel override 复用本仓库 `third_party/xla` / `third_party/llvm-project`；`test/Tools/Inputs/wafer_pytorch_xla_capture.py` 作为 test artifact generator 使用 `torch.export.export` + `torch_xla.stablehlo.exported_program_to_stablehlo` 默认生成 `4096x4096 @ 4096x4096` f32 matmul + bias + tanh + residual PyTorch/XLA StableHLO bundle，并支持 `--size` 生成同构小尺寸 compile smoke bundle；bundle 保留 `functions/forward.mlir`、`functions/forward.meta`、`functions/forward.bytecode` 和 `data/weight` / `data/bias`，不提交 64 MiB weight；`wafer-import-model --verify-stablehlo-bundle` 和 lit 最小验证已验证真实 source-built PyTorch/XLA adapter -> bundle -> verifier 链；框架 API、版本路径或自定义 JSON 不进入后端 IR 合同 |
| P2.S1 | 骨架 | Shardy propagation stage gate | 从 P2.F1 4096 matmul图出发：`test/Tools/Inputs/wafer_pytorch_xla_capture.py` 只作为 frontend export test artifact generator，通过 source-built PyTorch/XLA lazy SPMD runtime 的 `mark_sharding` 覆盖 data/batch、column parallel、row/contracting、2D output、2D contracting+output 和 partial replication；pre-partition bundle 保留 `mhlo.sharding` 并由 `wafer-import-model --verify-stablehlo-bundle` 和 `wafer-import-model --propagate-stablehlo-sharding` / `wafer-propagate-stablehlo-sharding` 消费。no-user 分支当前由 Wafer propagation gate 覆盖默认 `tile-count=16` / `tile-count=1` seed policy；不生成 `wafer.spmd.*`、私有 sharding JSON、名字约定或 Python post-SPMD helper 作为 sharding / per-rank routing 协议 |
| P2.S2 | ready | 建立 Wafer-owned XLA SPMD partition artifact stage | 新增 `wafer-spmd-partition` 或等价 driver/library stage：输入是 `wafer-import-model --propagate-stablehlo-sharding` 产出的 propagated StableHLO/SDY artifact，内部显式完成 StableHLO/SDY -> XLA HLO、XLA Shardy/SPMD partitioner、partitioned HLO -> StableHLO round trip，并输出带 post-SPMD marker、collective、rank-local function signature、`functions/forward.parameter_shards.json` 和 `parameter_shards/<parameter>/rank_XXXXX.npy` 的 Wafer bundle；前端 Python 不负责主链 partition |
| R2.4-pre | done | 收敛 Wafer named pipeline / driver contract | 新增 `WaferPipelines` 库并在 `wafer-opt` 注册 `wafer-propagate-stablehlo-sharding`、`wafer-lower-stablehlo-to-linalg`、`wafer-lower-linalg-to-cabi`、`wafer-lower-stablehlo-to-cabi` 和 `wafer-lower-tile-communication-to-cabi`；`wafer-propagate-stablehlo-sharding` 组合 Wafer default input seed 与 Shardy propagation，不冒充 XLA SPMD partitioner；`wafer-lower-stablehlo-to-cabi` 组合 StableHLO->Linalg 与 Linalg->C ABI body；`wafer-import-model --propagate-stablehlo-sharding` 从 StableHLO bundle verifier 进入 Shardy propagation，`--compile-stablehlo-bundle-to-cabi` 从 verified local/partitioned bundle 进入 compile pipeline，并拒绝 pre-SPMD sharding seed 绕过 SPMD；lit 覆盖手写最小 bundle、真实 source-built PyTorch/XLA 小尺寸 export bundle 和 Shardy propagation driver；pipeline option `target=wafer` materialize/校验 `#wafer.target<wafer>`，非 `wafer` target 诊断；单 pass flag 保留给 unit/debug，StageConnections 仍用于逐层连接覆盖 |
| R2.4 | pending | 建立 Wafer LinalgExt-style tensor collective handoff | 依赖 P2.S2。将 partitioned StableHLO collective normalize 成 tensor-level Wafer collective ops；这些 op 实现 `DestinationStyleOpInterface` / `TilingInterface` 和 Wafer collective verifier，和 `linalg` 一起进入 group/tiling；不生成 `tile_buffer`、`wafer.comm` 或 `unrealized_conversion_cast` |
| R3.1 | pending | 恢复 group boundary / candidate contract | `wafer.group` 只表达 local tensor grouping 和 candidate boundary；root op、operands/results、tile candidate shape 和拒绝原因由 IR/interface/verifier 可解释，不靠 pass side table 或名字；依赖 P2.F1/P2.S1/P2.S2/R2.4 的真实 frontend/SPMD artifact 来源和 tensor collective handoff；完成证明必须让真实图 artifact 的用户-sharding 和默认 no-user-sharding 分支都能进入 group candidate gate |
| R3.2 | pending | 恢复 root tile feasibility oracle | root tile candidate 检查必须接入 op tiling contract、layout requirement、SPM demand、DDR demand 和 compute/movement legality；静态 result shape check 只能是其中一个输入 |
| R3.3 | pending | 恢复 tile_region materialization contract | accepted group materialize 成 `wafer.tile_region`，load/store、layout materialize、`wafer.compute.*` 和 tile_yield 全部来自同一 accepted candidate；不得把未接受 plan 落进 IR |
| R3.4 | pending | 恢复 layout/SPM feasibility gate | layout materialization 是显式 movement；SPM trial 使用 resource effects、liveness/range 和 tile buffer lifetime，失败能诊断到具体 op/value |
| R3.5 | pending | 恢复 DDR/resource demand gate | `wafer.load_tile` / `store_tile` 驱动 external binding、workspace、resident constant、pool/domain、capacity/bandwidth demand；compact bytes fixture 不能作为完整 DDR 合同 |
| R3.6 | pending | 恢复 ABI issue gate | `wafer.abi.rdma` / `wdma` / `gemm` / elementwise / reduce issue sequence 从当前 storage/movement/compute IR 派生；ABI issue 保留参数单位和 wait policy，不回写上层 group/tile 语义 |
| R3.7 | pending | 恢复 package manifest gate | package manifest、C stub 和 launch signature 从当前 `wafer-opt` 输出导出；fixed fixture emitter 只作为 tool fixture |
| R3.8 | pending | 恢复 storage-realized / C ABI / golden packet 边界 | RDMA/WDMA/GEMM 有 storage-realized input、真实 `wafer_*` wrapper-facing call contract 和 golden packet 对照；不以 `wafer.abi.*` issue fixture 冒充最终 lower |
| R4.1 | pending | 恢复 placement map / capability gate | accepted placement 来自 logical rank、good-tile/PG/capability 和 physical card/tile coordinate verifier；placement 不携带 SPM offset、DDR address、DTE packet 或 runtime handle |
| R4.2 | pending | 恢复 per-rank identity / launch metadata gate | logical rank、block id、physical coord 和 local shard metadata 进入 tile_region/launch/package 边界，per-rank launch args 来自 IR，不来自 fixed manifest fixture |
| R4.3 | pending | 恢复 shard slicing materialization | multi-tile no-comm 为每个 rank materialize 自己的 input/output slice，不再 clone whole-tensor tile_region |
| R4.4 | pending | 恢复 per-rank writeback / merge contract | 每个 rank 写回对应 output slice；host-side readback、output merge 或 sharded output contract 明确，并由 IR/package metadata 驱动 |
| R4.5 | pending | 恢复 package / generated artifact gate | placement metadata、local shards、per-rank launch args 和 issue sequence 从当前 lowering 输出导出；package fixture 只能保留为 unit fixture |
| R5.1 | pending | 恢复 transformer local 编译验证 | workspace/resident constants/ABI issue sequence 来自 full-block IR dataflow 和 lowering 输出 |
| R5.2 | pending | 补 transformer compute/package gaps | mask/select、dynamic-bound policy、non-constant-init reduce、constant/weight slice 和 package consistency 按设计补齐 |
| R6.1 | pending | 恢复 communication design-conformance gate | DTE resource allocation、collective buffer slice/address offset、communication metadata 与 package/runtime 边界按设计落地 |
| R6.2 | pending | 建立 tiled tensor collective 到 comm 的 materialization | 旧 StableHLO -> `wafer.comm` bridge 已删除；R6.2 需让 tiled tensor collective 在 `wafer.tile_region` / SPM materialization 后转成 `wafer.comm.*`，并用可验证 buffer-slice / layout/materialization 路径表达 |

## 后续队列

P7/P8/P9 只有在 P0-P6 恢复队列完成后才能推进。

| ID | 状态 | 任务 | 验收 |
| --- | --- | --- | --- |
| P7.1 | pending | 建立 `wafer.abi.*` IR 到 package manifest 的导出路径 | single-tile、multi-tile 和 transformer package manifest 的 ABI issue sequence、launch signature 和 placement metadata 来自当前 lowering 输出；fixed fixture emitter 只保留为 unit fixture |
| P7.2 | later | 固定 `wafer.abi.*` 到 `wafer_*` C ABI call contract | 每类 ABI issue op 都有函数名、参数单位、wait/completion 责任和 stub header；unsupported op 有硬诊断 |
| P7.3 | later | 实现 `wafer.abi.*` 到 LLVM dialect call lowering | lowering 后不残留 `wafer.abi.*`；生成 `llvm.call` / symbol declaration；FileCheck 覆盖参数顺序和类型 |
| P7.4 | later | 建立 LLVM IR emission gate | `mlir-translate` 或等价路径能生成 LLVM IR；IR 文本检查 entrypoint、runtime call 和 metadata 引用 |
| P7.5 | later | 建立 object / link syntax gate | 当前 toolchain 能把 LLVM IR 或 generated source 编译成 object，并与 stub runtime ABI shim 做 syntax/link 最小验证 |
| P7.6 | later | 将实物 artifact 接入 package manifest | manifest 记录 LLVM/object artifact id、entrypoint 和 ABI version；C stub-only artifact 不再作为 correctness fence |
| P8.1 | later | 建立 runtime adapter contract 和 stub shielding | runtime path 明确区分真实 device completion 与已知 stub；stub 不能作为 correctness fence |
| P8.2 | later | 接 BO / DDR / launch argument binding | package 中的 tensor、workspace、constant 和 per-tile launch args 能绑定到真实 runtime 资源 |
| P8.3 | later | single-tile board 最小验证 | 实际 launch 成功，completion 可信，最小 GEMM 输出可做数值对比 |
| P8.4 | later | multi-tile / p2p / ring board 最小验证 | 多 tile no-comm、p2p 和 ring collective 有最小板端 completion / error propagation gate |
| P8.5 | later | transformer block board 最小验证 | full local block 产物能 launch；输出数值与参考实现按约定 tolerance 对比 |
| P9.1 | later | 建立 issue/drain placement verifier | overlap 决策由 effect/token 支撑 |
| P9.2 | later | 建立 SPM busy range pressure model | allocator failure 反馈 planner，不写入 IR |
| P9.3 | later | 建立 DDR range/bandwidth pressure model | range conflict / bandwidth cost 可诊断 |
| P9.4 | later | 建立 DTE resource pressure model | FSM / packet / stream pressure 进入 cost model |
| P9.5 | later | 接 PMU/profiling calibration | profiling 只校准 cost model，不作为 IR 语义事实 |

## 当前不做

- Serving integration。
- KV cache / paged attention / prefill-decode 调度。
- raw DTE non-unicast collective ABI。
- LLVM dialect / LLVM IR lowering、object emission 或真实 `wafer_*` runtime call emission。
- 自定义 LLVM backend 或 ISA intrinsic lowering。
- 以某个 importer、runtime path、workload shape 或 parameter 名称作为 IR 合同。

## 下一步

进入 P2.S2：建立 Wafer-owned XLA SPMD partition artifact stage，让 Wafer Shardy propagation
输出接到 partitioned StableHLO bundle。之后再进入 R2.4，建立 Wafer LinalgExt-style tensor
collective handoff；再进入 R3.1，让修正后的真实 frontend artifact 两个分支都能通过主线
pipeline / driver 进入 group candidate gate。
P7/P8/P9 依赖恢复后的 P0-P6 主链路，不提前推进。
