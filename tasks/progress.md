# Wafer Compiler Progress

更新时间：2026-05-25

状态：设计文档已经覆盖 compiler core 跑通静态 transformer block vertical slice 所需的主要边界。
下一阶段应转入实现和验证，不继续无边界扩写设计。

## Active Task

把文档中的 IR 分层落成最小可运行 compiler skeleton，并先跑通 M0/M1 load-GEMM-store 闭环。

当前执行焦点：

1. P0.1 到 P0.6：工程、依赖、工具和测试入口。
2. P1.1 到 P1.9：最小 Wafer dialect skeleton、parser/printer/verifier。
3. P3.1 到 P3.10：从手写 StableHLO / Linalg GEMM 输入跑通 M0。

## 当前判断

- 设计层面已经足够支撑开工：frontend artifact、SPMD、placement、local compute normalization、
  `wafer.group`、`wafer.tile_region`、layout、SPM、DDR、compute、communication、C ABI、
  launch/runtime 和 verification plan 都已有独立边界。
- Serving、KV cache、paged attention、prefill/decode 和全模型 runtime integration 暂不进入
  compiler core 跑通主线。
- 如果实现时发现文档和 IR 合同冲突，先修改对应设计文档，再改代码；不要把临时 workaround
  写成长期协议。

## 当前验证口径

在开发环境切到带实际计算卡的服务器之前，当前 gate 以 local compile / package 验证为准：

- compiler pipeline 能从手写 StableHLO / Linalg 或 verified artifact 走到最终编译产物。
- 生成的 C ABI / LLVM / device-code source 或 object 能被当前可用 toolchain 编译。
- runtime package manifest、constant bytes metadata、DDR/SPM/resource summary 能序列化和 roundtrip。
- golden packet / ABI unit tests、FileCheck、verifier negative tests、resource planner tests 通过。
- 不要求当前环境完成板端 launch、device completion、数值对比或 PMU/profiling。

后续迁移到带实际计算卡服务器后，再把 board run 作为对应 milestone 的新增 gate：实际 runtime
launch、可信 completion、输出数值检查、错误传播和 profiling calibration。

## 状态标记

- `ready`：可以直接开始实现。
- `pending`：依赖前序任务完成。
- `later`：当前主线之后再做。
- `done`：实现和对应验证已经完成。

## P0. 工程和依赖骨架

目标：让后续 IR / pass / tests 有稳定工程入口。

| ID | 状态 | 任务 | 验收 |
| --- | --- | --- | --- |
| P0.1 | ready | 建立最小 CMake / build 入口 | 可以配置空项目；不要求已有完整 Wafer IR |
| P0.2 | ready | 建立 LLVM / MLIR 依赖发现和版本 pin 机制 | 依赖由集中配置声明，不在源码里散落 include/library path |
| P0.3 | ready | 加入 StableHLO / Shardy 依赖开关和 dialect registration 入口 | 可关闭 importer-only 依赖并运行 backend textual tests |
| P0.4 | ready | 创建 `include/Wafer/`、`lib/Wafer/`、`tools/wafer-opt` 最小骨架 | `wafer-opt --help` 或等价 smoke test 可运行 |
| P0.5 | ready | 建立 lit / FileCheck 测试目录和最小 test target | 一个空 dialect smoke test 能被 test runner 收集 |
| P0.6 | ready | 建立 gtest 或等价 C++ unit test 入口 | 后续 allocator / storage calculator 有测试落点 |
| P0.7 | pending | 创建 `tools/wafer-import-model` shell | importer-only 依赖缺失时后端仍可构建 |
| P0.8 | pending | 增加依赖一致性检查脚本 | 能检查版本 pin、dialect registration 和 importer/backend 隔离 |

## P1. Wafer IR Skeleton 和 Verifier

目标：核心 IR 对象可以 parse / print / verify，先固定结构边界，不实现复杂 lowering。

| ID | 状态 | 任务 | 验收 |
| --- | --- | --- | --- |
| P1.1 | pending | 定义 `WaferDialect` 和基础 ODS 文件组织 | dialect 可注册；空 module roundtrip |
| P1.2 | pending | 定义共享 attrs/types：target、placement、memory space、mem layout | parser/printer roundtrip；非法 enum 被 verifier 拒绝 |
| P1.3 | pending | 定义 `WaferTilingInterface`、`WaferLayoutOpInterface`、resource/effect 相关接口骨架 | ODS / C++ 编译通过；接口不携带 planner side table |
| P1.4 | pending | 定义最小 `wafer.group` op | region / operand / result contract 可验证；不接受 SPM memref |
| P1.5 | pending | 定义最小 `wafer.tile_region` op 和 tile buffer type | tile-local region boundary、buffer ownership、effect scope 可验证 |
| P1.6 | pending | 定义 layout materialization op 的最小 IR 形态 | 表达真实 data movement；不作为 metadata cast |
| P1.7 | pending | 定义 `wafer.compute` / movement 最小 op family：load tile、store tile、gemm | dtype、shape、layout contract 有 verifier |
| P1.8 | pending | 定义 `wafer.comm` p2p、sync token、local wait 的最小表示 | endpoint、token、wait policy 可验证 |
| P1.9 | pending | 定义 `wafer.launch` 最小 boundary | launch signature、resource summary、package ref 不反向污染 tensor IR |
| P1.10 | pending | 写 parser/printer/verifier 正负例 | 每类 op/type/attr 至少一个 roundtrip 和一个 negative test |

## P2. Frontend Artifact 和 Local Compute Normalization

目标：先从手写 StableHLO / MLIR 输入建立稳定 backend 入口，再接真实 importer。

| ID | 状态 | 任务 | 验收 |
| --- | --- | --- | --- |
| P2.1 | pending | 建立 StableHLO / MLIR textual artifact 输入测试 | parse/roundtrip 保留 function signature、shape、dtype |
| P2.2 | pending | 实现 constant normalization 骨架 | `stablehlo.constant` / sidecar 进入 `arith.constant` 或 `ConstantLike` tensor value |
| P2.3 | pending | Lower `dot_general` / matmul 到 structured tensor IR | indexing map / iterator / DPS 关系可 FileCheck |
| P2.4 | pending | Lower broadcast、reshape、transpose、slice | 只依赖 type、shape、indexing relation，不靠名字 |
| P2.5 | pending | Lower elementwise 和 limited broadcast | 覆盖 add/sub/mul/div/max/min/neg/recip/sqrt/rsqrt/exp 的基础形态 |
| P2.6 | pending | Lower reduce max / reduce sum | 输出 staged reduce IR，可服务 softmax 和 norm |
| P2.7 | pending | 表达 RMSNorm / LayerNorm staged form | reduce + elementwise，不引入 `wafer.norm` 高层 op |
| P2.8 | pending | 表达 softmax staged form | row max、exp、row sum、normalize 状态由 SSA / loop-carried / workspace 表达 |
| P2.9 | pending | 表达 RoPE 和 MLP activation staged form | 只使用 structured tensor IR 和 math/arith 语义 |
| P2.10 | pending | 加 importer smoke test | graph break、eager fallback、unbounded dynamic shape 会被诊断 |

## P3. M0 Single Tile Load-GEMM-Store

目标：跑通最小 device program 链路。

| ID | 状态 | 任务 | 验收 |
| --- | --- | --- | --- |
| P3.1 | pending | 准备手写 StableHLO / Linalg GEMM 输入 case | 输入不依赖模型 importer |
| P3.2 | pending | 实现最小 group formation | 单 GEMM 能形成 `wafer.group`；非法 multi-output/domain 被拒绝或拆分 |
| P3.3 | pending | 实现 root tile shape 候选和 feasibility 调用骨架 | tile shape 是 planner 候选，不写成 IR contract |
| P3.4 | pending | materialize 单 tile `wafer.tile_region` | region 中有 load、compute、store 的 SSA 关系 |
| P3.5 | pending | 实现 compact layout assignment | 不插不必要的 layout conversion |
| P3.6 | pending | 实现 SPM allocation trial | range、alignment、lifetime、end-address 检查通过 |
| P3.7 | pending | 实现 DDR external input/output binding demand | DDR demand 可被 launch/runtime 层消费 |
| P3.8 | pending | Lower `wafer.compute.gemm` / load / store 到 C ABI skeleton | 参数单位、address domain、wait policy 有 verifier |
| P3.9 | pending | 建立 golden packet / ABI unit test | 至少覆盖 M0 用到的 compute/movement family |
| P3.10 | pending | 建立最小 launch/runtime package manifest | manifest roundtrip，package 不依赖已知 stub path 作为 correctness fence |
| P3.11 | pending | M0 local compile gate 汇总测试 | textual pipeline、SPM、DDR、C ABI、generated artifact compile、manifest 检查全部通过 |

## P4. M1 Multi-Tile No Communication

目标：验证 placement、per-tile args 和多 tile launch，不引入 DTE。

| ID | 状态 | 任务 | 验收 |
| --- | --- | --- | --- |
| P4.1 | pending | 定义 logical rank 到 physical tile mapping 表示 | mapping 覆盖所有 rank，不使用 bad tile |
| P4.2 | pending | 接入 good-tile / block id / local shard metadata | metadata 可进入 package，不修改 tensor semantics |
| P4.3 | pending | 支持多 tile 独立 tile-region outlining | 每 tile 独立 load-compute-store，无 tile 间 comm |
| P4.4 | pending | 支持 per-tile launch args | runtime launch 能区分 tile-specific arguments |
| P4.5 | pending | M1 local compile gate 汇总测试 | 多 tile no-comm pipeline、generated artifact compile 和 manifest 检查通过 |

## P5. Transformer Block Local Vertical Slices

目标：在单 shard / 单卡本地路径上逐步覆盖 transformer block。

| ID | 状态 | 任务 | 验收 |
| --- | --- | --- | --- |
| P5.1 | pending | Norm slice：RMSNorm 或 LayerNorm | reduce + elementwise group schedule accepted |
| P5.2 | pending | Softmax slice | row max、exp、row sum、normalize 的 staged schedule accepted |
| P5.3 | pending | Attention score slice：QK^T | batch/head matmul relation 从 StableHLO dimension numbers / indexing map 推出 |
| P5.4 | pending | Attention value slice：softmax + AV | softmax output 到 value accumulation 的状态和 buffer lifetime 可验证 |
| P5.5 | pending | Output projection + residual slice | residual/add/bias 等 elementwise 与 GEMM 边界清晰 |
| P5.6 | pending | MLP slice | GEMM + activation + elementwise multiply + GEMM 可分组或可诊断拆分 |
| P5.7 | pending | Full local transformer block | norm、attention、MLP 串联；layout/SPM/DDR feasibility 全部通过 |
| P5.8 | pending | M6 local compile gate | 单 shard / 单卡完整 block 的 normalization、group split、layout/SPM/DDR、C ABI、generated artifact compile、package 检查通过 |

## P6. Communication / Tensor Parallel Path

目标：只有在 transformer block 需要 tensor parallel 时推进。

| ID | 状态 | 任务 | 验收 |
| --- | --- | --- | --- |
| P6.1 | later | 定义 `wafer.comm` endpoint / token / wait verifier | DTE wait 与 local compute drain 分离 |
| P6.2 | later | Lower fixed-size unicast Direct DTE helper | raw non-unicast DTE 不作为 correctness path |
| P6.3 | later | 管理 FSM / packet / stream resource | resource 不冲突，有 negative tests |
| P6.4 | later | 实现 ring all-gather | 每步 send/recv/wait token 和 buffer lifetime 合法 |
| P6.5 | later | 实现 reduce-scatter / all-reduce | collective 可追溯到 unicast steps |
| P6.6 | later | Lower Shardy/SPMD logical collective 到 `wafer.comm` | placement 和 comm lowering 保留 collective semantics |
| P6.7 | later | M2/M3/M4 gate 汇总测试 | p2p、single-card collective、partitioned collective lowering 分别可验证 |

## P7. Overlap、Cost Model 和 Profiling Calibration

目标：功能链路稳定后再优化。

| ID | 状态 | 任务 | 验收 |
| --- | --- | --- | --- |
| P7.1 | later | 建立 issue/drain placement verifier | overlap 决策由 effect/token 支撑 |
| P7.2 | later | 建立 SPM busy range pressure model | allocator failure 反馈 planner，不写入 IR |
| P7.3 | later | 建立 DDR range/bandwidth pressure model | range conflict / bandwidth cost 可诊断 |
| P7.4 | later | 建立 DTE resource pressure model | FSM / packet / stream pressure 进入 cost model |
| P7.5 | later | 接 PMU/profiling calibration | profiling 只校准 cost model，不作为 IR 语义事实 |

## 当前前置实现项

- 选择并 pin LLVM / MLIR / StableHLO / Shardy 依赖版本。
- 建立 build/test harness；没有 harness 时只能做文档和文本一致性验证，不能宣称执行链路通过。
- 定义最小 Wafer dialect ODS 文件和 shared attrs/types/interfaces。
- 建立至少一条 textual MLIR pipeline，从手写输入开始，不等待完整 model importer。

## 本轮不做

- Serving integration。
- KV cache / paged attention / prefill-decode 调度。
- raw DTE non-unicast collective ABI。
- 自定义 LLVM backend 或 ISA intrinsic lowering。
- 以某个 importer、runtime path、workload shape 或 parameter 名称作为 IR 合同。

## 下一步

从 P0.1 开始落地工程骨架。第一批代码提交应只覆盖 build/dependency/test harness 和最小 Wafer
dialect skeleton，不同时实现 transformer block 逻辑。
