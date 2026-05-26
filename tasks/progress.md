# Wafer Compiler Progress

更新时间：2026-05-26

本文件只记录当前看板、状态和下一步；不再承载实现流水账。历史实现细节以 git commit、设计文档
和审计文档为准。

## 当前状态

- P0-P6 的历史实现只能视为骨架进度（`skeleton`）：有局部 IR、pass、verifier、fixture 和 smoke tests，
  但没有按各设计文档完成主路径闭环。
- R0.2/R0.3 已恢复源码 ownership、依赖层级边界和统一 Shardy CMake 编译验证目标；ODS/verifier
  按 op prefix 拆分仍在 R1.1。
- 设计一致性缺口见
  `tasks/2026-05-26-wafer-p0-p6-design-conformance-audit.md` 和
  `tasks/2026-05-26-wafer-p0-p6-recovery-status.md`。
- 当前不得推进 P7/P8/P9；必须先恢复 P0-P6 的设计一致性。
- 当前唯一 ready 项是 R1.1。

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
| P1 | skeleton | Wafer IR skeleton 和 verifier | 核心 op/type/attr skeleton 有测试；ODS/verifier/interface/effect/resource 仍需按 op prefix 和设计合同复核 |
| P2 | skeleton | Frontend artifact 和 local compute normalization | StableHLO textual lowering 有覆盖；真实 importer adapter、sidecar/ConstantLike/storage contract 未闭环 |
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
| R1.1 | ready | 拆分 Wafer IR 文件边界 | ODS、C++ verifier 和 tests 按 group/tile_region/layout/SPM/compute/comm/sync/launch op prefix 组织 |
| R1.2 | pending | 恢复 interface/effect/resource 合同 | planner、layout、SPM/DDR、compute/comm 能通过 op interface / effect 查询需求和合法性 |
| R1.3 | pending | 补 stage-connection tests | 减少只靠 `unrealized_conversion_cast` 的孤立 verifier case，增加上下游连接验证 |
| R2.1 | pending | 恢复 frontend artifact / importer contract | importer adapter、sidecar/ConstantLike、graph break/eager/dynamic shape 诊断按 frontend 设计闭环 |
| R2.2 | pending | 恢复 Shardy/SPMD artifact bridge | logical mesh、rank group、partitioned StableHLO collective 和 shard-local program 成为 placement/comm 输入 |
| R2.3 | pending | 重写 local compute normalization 覆盖状态 | 按 structured tensor IR contract 记录 dot/broadcast/reduce/softmax/norm/RoPE/MLP 覆盖；acceptance pass 不冒充 schedule completion |
| R3.1 | pending | 恢复 M0 group/tile/layout/SPM/DDR/C ABI 主链路 | group planner、root tile feasibility、SPM/DDR demand、C ABI skeleton 均来自同一 IR pipeline 且满足对应设计合同 |
| R3.2 | pending | 恢复 M0 package gate | package manifest 和 C stub 从当前 `wafer-opt` 输出导出；fixed smoke emitter 只作为 tool fixture |
| R3.3 | pending | 恢复 storage realization / C ABI / golden packet 边界 | M0 RDMA/WDMA/GEMM 有 storage-realized input、真实 wrapper-facing call contract 和 golden packet 对照 |
| R4.1 | pending | 恢复 M1 placement / shard / launch metadata gate | multi-tile no-comm 使用真实 shard slicing、per-rank result/metadata 和 package launch args，不用 whole-tensor clone 代替 |
| R4.2 | pending | 恢复 M1 shard slicing / merge | per-rank writeback、host-side readback 或 output merge contract 明确，并由 IR / package metadata 驱动 |
| R5.1 | pending | 恢复 M6 transformer local 编译验证 | workspace/resident constants/ABI issue sequence 来自 full-block IR dataflow 和 lowering 输出 |
| R5.2 | pending | 补 transformer compute/package gaps | mask/select、dynamic-bound policy、non-constant-init reduce、constant/weight slice 和 package consistency 按设计补齐 |
| R6.1 | pending | 恢复 communication design-conformance gate | DTE resource allocation、collective buffer slice/address offset、communication metadata 与 package/runtime 边界按设计落地 |
| R6.2 | pending | 清理 StableHLO collective bridge 临时 cast | 用可验证 buffer-slice / layout/materialization 路径替代 visible `unrealized_conversion_cast` |

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

从 R1.1 开始：先拆 Wafer IR 文件边界，再按 R1/R2/R3 顺序恢复 IR、frontend 和 M0 主链路。P7/P8/P9
依赖恢复后的 P0-P6 主链路，不提前推进。
