# Wafer Compiler Progress

更新时间：2026-05-27

本文件只记录当前看板、状态和下一步；不再承载实现流水账。历史实现细节以 git commit、设计文档
和审计文档为准。

## 当前状态

- P0-P6 的历史实现只能视为骨架进度（`skeleton`）：有局部 IR、pass、verifier、fixture 和 smoke tests，
  但没有按各设计文档完成主路径闭环。
- R0.2/R0.3/R1.1/R1.2/R1.3/R2.1/R2.2/R2.3 已恢复源码 ownership、依赖层级边界、统一 Shardy
  CMake 编译验证目标、Wafer IR op-family 文件边界、Wafer interface/effect/resource 查询合同、
  local compute stage-connection gate、frontend artifact/sidecar verifier、SDY artifact bridge、
  StableHLO replica rank group 到 `wafer.comm` 的显式表示，以及 local compute normalization 覆盖状态。
- 设计一致性缺口见
  `tasks/2026-05-26-wafer-p0-p6-design-conformance-audit.md` 和
  `tasks/2026-05-26-wafer-p0-p6-recovery-status.md`。
- framework-specific capture 和完整 Shardy SPMD pipeline 不是放弃项；它们已经从 R2 的
  “未覆盖范围”提升为下面的 P2 显式后续项。它们不阻塞当前 R3.1 的 local M0 主链路恢复，但属于
  frontend/SPMD 端到端完成标准。
- 当前不得推进 P7/P8/P9；必须先恢复 P0-P6 的设计一致性。
- 当前唯一 ready 项是 R3.1。

## 状态标记

- `ready`：可以直接开始实现。
- `pending`：依赖前序任务完成。
- `later`：当前主线之后再做。
- `skeleton`：历史 skeleton gate 通过，但不代表设计文档主路径完成。
- `done`：实现和对应验证已经按设计合同完成。

## 当前验证口径

当前本地可验证范围：

- CMake / `wafer-opt` / lit / gtest 构建入口。
- MLIR textual tests、FileCheck、verifier negative tests。
- 固定依赖版本 / importer backend 隔离检查。
- Wafer tiling/layout/materialization/resource op interface 查询和 MLIR memory effect resource 查询。
- 从 public linalg source 到 group、tile_region、`wafer.abi.*` skeleton 的 stage-connection lit gate。
- frontend artifact verifier 的 graph-break/eager fallback/bounded dynamic shape/sidecar constant gate。
- SDY `sdy.mesh` / `sdy.sharding` artifact parse/verify gate，以及 StableHLO replica `rank_group`
  到 `wafer.comm` / ring lowering 的 lit gate。
- C ABI skeleton ops、package manifest fixture 和 C stub syntax compile。

当前不能作为完成证明：

- fixed smoke manifest 不能证明 package 来自当前 `wafer-opt` lowering 输出。
- C stub syntax compile 不能证明 LLVM IR、object、真实 runtime call 或 wrapper/packet lowering。
- 本地验证不能替代板端 launch、device completion、数值对比或 PMU/profiling。

## 历史骨架状态

这些条目保留为历史骨架进度索引，不作为设计完成声明。

| ID | 状态 | 范围 | 当前结论 |
| --- | --- | --- | --- |
| P0 | skeleton | 工程、依赖、工具、测试入口 | 最小工程入口可用；源码 ownership 和依赖层级边界已恢复；P0 仍只是历史 skeleton 记录，不代表 frontend/runtime 主路径完成 |
| P1 | skeleton | Wafer IR skeleton 和 verifier | 核心 op/type/attr skeleton 有测试；ODS/verifier/tests 已按 op family 拆开；interface/effect/resource 已恢复为可查询合同；local compute stage-connection gate 已补，但 storage-realized 主链路仍未闭环 |
| P2 | skeleton | Frontend artifact 和 local compute normalization | StableHLO textual lowering、frontend artifact verifier、sidecar/resource-backed constant metadata、SDY artifact bridge 和 local compute coverage 口径已恢复；P2 仍不代表 framework importer、完整 SPMD partitioner、dynamic/mask/constant-storage 或 physical schedule 闭环 |
| P3 | skeleton | M0 single-tile load-GEMM-store | 有 group/tile/SPM/DDR/C ABI skeleton；planner、package、golden packet 和 C ABI 主路径未闭环 |
| P4 | skeleton | M1 multi-tile no-comm | 有 placement/map 和 clone-style tile_region skeleton；真实 shard slicing、merge、runtime launch metadata 未闭环 |
| P5 | skeleton | Transformer local vertical slices | 有 staged pattern acceptance 和 M6 skeleton gate；full-block package/device artifact 未从 IR 闭环 |
| P6 | skeleton | Communication / tensor parallel path | 有 p2p/ring/DTE skeleton；buffer slicing、address lowering、resource allocator、package/runtime metadata 未闭环 |

## P0-P6 设计一致性恢复队列

目标：把历史骨架进度重新对齐到各设计文档的主路径合同；在这些任务完成前，不进入 P7。

| ID | 状态 | 任务 | 验收 |
| --- | --- | --- | --- |
| R0.1 | done | 逐项重读 P0-P6 对应设计文档并重写任务状态 | 记录见 `tasks/2026-05-26-wafer-p0-p6-recovery-status.md`；每个历史 skeleton 项都有“设计合同 / 当前实现 / 缺口 / 恢复任务” |
| R0.2 | done | 恢复源码组织边界 | 记录见 `tasks/2026-05-26-wafer-source-organization-recovery.md`；IR / Frontend / Transforms / Conversion / ABI ownership 已拆开，C ABI skeleton 不再归属 `WaferTransforms` |
| R0.3 | done | 补依赖层级清单 | 记录见 `tasks/2026-05-26-wafer-dependency-layering-recovery.md`；R0.3 只完成依赖源码拉取、版本固定、层级隔离和编译验证，不表示这些依赖都已经被 Wafer 代码实际调用；`third_party/pytorch-xla` 只用于确定 XLA 版本，当前没有 `torch_xla` frontend importer；`third_party/xla` 只作为后续 GSPMD 集成的源码和版本来源，当前不编译 XLA/GSPMD 目标；已通过 Wafer CMake 实际编译验证的是同一套 LLVM/MLIR、StableHLO 和 Shardy/SDY 公共 dialect/pass，输出 `shardy-sdy-opt`；core compiler target 不 public link Shardy/XLA/PyTorch-XLA；target 可见范围、固定版本一致性和 Shardy 编译验证目标由 `tools/check_deps.py` 检查 |
| R1.1 | done | 拆分 Wafer IR 文件边界 | `WaferOps.td` 只聚合 `include/Wafer/IR/Ops/*Ops.td`；op verifier 已拆到 `lib/Wafer/IR/Ops/*Ops.cpp`；`test/Dialect/Wafer` 已按 ABI/Attrs/Comm/Compute/DDR/Group/Launch/Layout/Placement/SPM/Sync/TileRegion/Types 分目录；`tools/check_ir_organization.py` 和 lit test 固定结构检查 |
| R1.2 | done | 恢复 interface/effect/resource 合同 | `WaferTilingInterface`、`WaferLayoutOpInterface`、`WaferLayoutMaterializationOpInterface` 和 `WaferResourceEffectInterface` 都提供结构化查询；layout/materialize/SPM/DDR/compute/comm/sync/ABI op 接入 Wafer resource effects，关键 movement/compute/comm op 接入 MLIR memory resource effects；gtest 覆盖 planner-style 查询 |
| R1.3 | done | 补 stage-connection tests | `test/StageConnections/local-compute-pipeline.mlir` 覆盖 linalg matmul/elementwise/reduce 到 group、tile_region、`wafer.abi.*` skeleton 的连接；`tools/check_stage_connection_tests.py` 固定禁止 stage-connection gate 退回 cast-only 用例 |
| R2.1 | done | 恢复 frontend artifact / importer contract | 记录见 `tasks/2026-05-26-wafer-r2-recovery.md`；`WaferFrontend` verifier、`wafer-import-model --verify-import-result [--sidecar]`、`wafer.frontend.dynamic_bounds`、`wafer.frontend.constant` sidecar 对齐、graph break/eager fallback/dynamic shape 诊断已闭环 |
| R2.2 | done | 恢复 Shardy/SPMD artifact bridge | 记录见 `tasks/2026-05-26-wafer-r2-recovery.md`；SDY dialect/pass 注册按 `WAFER_ENABLE_SPMD_PARTITIONER_DEPS` 隔离，`sdy.mesh`/`sdy.sharding` artifact 可进入工具链，StableHLO `replica_groups` materialize 为 `wafer.comm` `rank_group`，ring lowering 按 logical rank group 查询 placement |
| R2.3 | done | 重写 local compute normalization 覆盖状态 | 记录见 `tasks/2026-05-26-wafer-r2-recovery.md` 和 `tasks/2026-05-25-wafer-local-compute-normalization-design.md`；dot/broadcast/reduce/softmax/norm/RoPE/MLP 覆盖按 structured tensor IR evidence 记录，acceptance pass 不作为 schedule completion |
| R3.1 | ready | 恢复 M0 group boundary / candidate contract | `wafer.group` 只表达 local tensor grouping 和 candidate boundary；root op、operands/results、tile candidate shape 和拒绝原因由 IR/interface/verifier 可解释，不靠 pass side table 或名字 |
| R3.2 | pending | 恢复 M0 root tile feasibility oracle | root tile candidate 检查必须接入 op tiling contract、layout requirement、SPM demand、DDR demand 和 compute/movement legality；静态 result shape check 只能是其中一个输入 |
| R3.3 | pending | 恢复 M0 tile_region materialization contract | accepted group materialize 成 `wafer.tile_region`，load/store、layout materialize、`wafer.compute.*` 和 tile_yield 全部来自同一 accepted candidate；不得把未接受 plan 落进 IR |
| R3.4 | pending | 恢复 M0 layout/SPM feasibility gate | layout materialization 是显式 movement；SPM trial 使用 resource effects、liveness/range 和 tile buffer lifetime，失败能诊断到具体 op/value |
| R3.5 | pending | 恢复 M0 DDR/resource demand gate | `wafer.load_tile` / `store_tile` 驱动 external binding、workspace、resident constant、pool/domain、capacity/bandwidth demand；compact bytes fixture 不能作为完整 DDR 合同 |
| R3.6 | pending | 恢复 M0 ABI skeleton issue gate | `wafer.abi.rdma` / `wdma` / `gemm` / elementwise / reduce issue sequence 从当前 storage/movement/compute IR 派生；ABI issue 保留参数单位和 wait policy，不回写上层 group/tile 语义 |
| R3.7 | pending | 恢复 M0 package manifest gate | package manifest、C stub 和 launch signature 从当前 `wafer-opt` 输出导出；fixed smoke emitter 只作为 tool fixture |
| R3.8 | pending | 恢复 M0 storage-realized / C ABI / golden packet 边界 | RDMA/WDMA/GEMM 有 storage-realized input、真实 `wafer_*` wrapper-facing call contract 和 golden packet 对照；不以 `wafer.abi.*` skeleton 冒充最终 lower |
| R4.1 | pending | 恢复 M1 placement map / capability gate | accepted placement 来自 logical rank、good-tile/PG/capability 和 physical card/tile coordinate verifier；placement 不携带 SPM offset、DDR address、DTE packet 或 runtime handle |
| R4.2 | pending | 恢复 M1 per-rank identity / launch metadata gate | logical rank、block id、physical coord 和 local shard metadata 进入 tile_region/launch/package 边界，per-rank launch args 来自 IR，不来自 fixed manifest fixture |
| R4.3 | pending | 恢复 M1 shard slicing materialization | multi-tile no-comm 为每个 rank materialize 自己的 input/output slice，不再 clone whole-tensor tile_region |
| R4.4 | pending | 恢复 M1 per-rank writeback / merge contract | 每个 rank 写回对应 output slice；host-side readback、output merge 或 sharded output contract 明确，并由 IR/package metadata 驱动 |
| R4.5 | pending | 恢复 M1 package / generated artifact gate | placement metadata、local shards、per-rank launch args 和 issue sequence 从当前 lowering 输出导出；package fixture 只能保留为 unit fixture |
| R5.1 | pending | 恢复 M6 transformer local 编译验证 | workspace/resident constants/ABI issue sequence 来自 full-block IR dataflow 和 lowering 输出 |
| R5.2 | pending | 补 transformer compute/package gaps | mask/select、dynamic-bound policy、non-constant-init reduce、constant/weight slice 和 package consistency 按设计补齐 |
| R6.1 | pending | 恢复 communication design-conformance gate | DTE resource allocation、collective buffer slice/address offset、communication metadata 与 package/runtime 边界按设计落地 |
| R6.2 | pending | 清理 StableHLO collective bridge 临时 cast | 用可验证 buffer-slice / layout/materialization 路径替代 visible `unrealized_conversion_cast` |

## P2 显式后续项

这些项不是 R2.1-R2.3 的完成条件，但也不能被解释成“不做”。它们需要在端到端 frontend/SPMD
闭环前落地；当前先作为显式队列项保留，不改变 R3.1 是当前 ready 项的事实。

| ID | 状态 | 任务 | 验收 |
| --- | --- | --- | --- |
| P2.F1 | pending | 建立 framework-specific capture adapter contract | PyTorch/JAX 等框架 capture 入口只能产出 verified StableHLO/MLIR artifact、sidecar 和 compile config；graph break、eager fallback、dynamic bound、constant/weight、sharding annotation 的诊断进入 frontend verifier；框架 API、版本路径或参数名不进入后端 IR 合同 |
| P2.S1 | pending | 集成完整 Shardy propagation / SPMD partitioner pipeline | 从未分片或只带 sharding annotation 的 StableHLO/SDY artifact 出发，运行 Shardy propagation/partitioning 并产出 per-rank artifact；multi replica group、rank selection、shard slicing、collective legality 和 placement input 由 IR/attr/verifier 明确表示，不靠 pass side table 或名字 |

## 后续队列

P7/P8/P9 只有在 P0-P6 恢复队列完成后才能推进。

| ID | 状态 | 任务 | 验收 |
| --- | --- | --- | --- |
| P7.1 | pending | 建立 `wafer.abi.*` IR 到 package manifest 的导出路径 | M0/M1/M6 package manifest 的 ABI issue sequence、launch signature 和 placement metadata 来自当前 lowering 输出；fixed smoke emitter 只保留为 unit fixture |
| P7.2 | later | 固定 `wafer.abi.*` 到 `wafer_*` C ABI call contract | 每类 ABI issue op 都有函数名、参数单位、wait/completion 责任和 stub header；unsupported op 有硬诊断 |
| P7.3 | later | 实现 `wafer.abi.*` 到 LLVM dialect call lowering | lowering 后不残留 `wafer.abi.*`；生成 `llvm.call` / symbol declaration；FileCheck 覆盖参数顺序和类型 |
| P7.4 | later | 建立 LLVM IR emission gate | `mlir-translate` 或等价路径能生成 LLVM IR；IR 文本检查 entrypoint、runtime call 和 metadata 引用 |
| P7.5 | later | 建立 object / link syntax gate | 当前 toolchain 能把 LLVM IR 或 generated source 编译成 object，并与 stub runtime ABI shim 做 syntax/link smoke |
| P7.6 | later | 将实物 artifact 接入 package manifest | manifest 记录 LLVM/object artifact id、entrypoint 和 ABI version；C stub-only artifact 不再作为 correctness fence |
| P8.1 | later | 建立 runtime adapter contract 和 stub shielding | runtime path 明确区分真实 device completion 与已知 stub；stub 不能作为 correctness fence |
| P8.2 | later | 接 BO / DDR / launch argument binding | package 中的 tensor、workspace、constant 和 per-tile launch args 能绑定到真实 runtime 资源 |
| P8.3 | later | M0 single-tile board smoke | 实际 launch 成功，completion 可信，最小 GEMM 输出可做数值对比 |
| P8.4 | later | M1/M2/M3/M4 board smoke | 多 tile no-comm、p2p 和 ring collective 有最小板端 completion / error propagation gate |
| P8.5 | later | M6 transformer block board smoke | full local block 产物能 launch；输出数值与参考实现按约定 tolerance 对比 |
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

从 R3.1 开始：恢复 M0 group boundary / candidate contract，再按 R3 顺序恢复 M0 主链路。
P7/P8/P9 依赖恢复后的 P0-P6 主链路，不提前推进。
