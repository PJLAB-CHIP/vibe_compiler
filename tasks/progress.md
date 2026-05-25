# Wafer Compiler Progress

更新时间：2026-05-25

状态：设计文档已经覆盖 compiler core 跑通静态 transformer block vertical slice 所需的主要边界。
下一阶段应转入实现和验证，不继续无边界扩写设计。

最新实现批次：已落地最小 CMake / MLIR 工程骨架、`wafer-opt`、lit/FileCheck、gtest 入口和
`WaferDialect` + 共享 enum attrs 的 parser/printer/verifier smoke tests。LLVM/MLIR 版本通过集中
pin 和 bootstrap 脚本管理；当前本地验证使用 21.0.0git override，未把本机路径写成项目合同。
`wafer-import-model` disabled stub、依赖一致性检查脚本和 Wafer tiling/layout/resource-effect
interface skeleton 已落地。最小 `wafer.group` / `wafer.group_yield`、region/result/yield verifier
和 SPM memref 拒绝测试已落地。最小 `!wafer.tile_buffer`、`wafer.tile_region` /
`wafer.tile_yield`、边界类型 verifier 和 SPM tile buffer escape 拒绝测试已落地。最小
`wafer.layout.materialize` 及 layout/tensor/memory-space verifier 已落地；下一步进入
compute/movement 之后的 comm/token/wait 表示。`wafer.load_tile`、`wafer.store_tile` 和
`wafer.compute.gemm` 的最小 shape/layout/memory-space verifier 已落地。最小 `wafer.comm.send` /
`wafer.comm.recv` / `wafer.comm.wait` 和 `wafer.sync.local_drain` 已落地，通信 token 使用
`!async.token`，endpoint / byte count / SPM buffer 合同有 verifier。最小 `wafer.launch` boundary
已落地，launch signature、package ref、resource summary 和 host tensor 边界有 verifier。P1
parser/printer/verifier 正负例已补齐到当前 op/type/attr 覆盖口径。P2/P3 输入 artifact 起点已落地：
默认 backend gate 覆盖手写 Linalg GEMM 文本入口，importer-enabled gate 覆盖 pinned StableHLO
`dot_general` 文本入口和 dialect registration。最小 constant normalization pass 已落地，importer
build 中把 `stablehlo.constant` 重写为 `arith.constant`，不引入 Wafer 私有 constant op；独立
sidecar manifest schema 仍归 P3.10 package manifest，不在本批次造临时格式。P2.3 的 M0 范围
2D `stablehlo.dot_general` 到 structured tensor IR lowering 已落地，输出 zero fill + DPS
`linalg.matmul`。P2.4 的 StableHLO shape normalization 已落地：`broadcast_in_dim`、rank-changing
`reshape`、`transpose`、`slice` 降到 `linalg` / `tensor` structured IR，使用 type、shape 和 indexing
关系，不新增 Wafer 私有 op 或名字约定。P2.5 的 StableHLO elementwise normalization 已落地：
add/sub/mul/div/max/min/neg/sqrt/rsqrt/exp 降到 `linalg.elementwise`，`1 / x` 常量形态归一化为
reciprocal kind，单 use `broadcast_in_dim` 输入通过 `indexing_maps` 表达 limited broadcast。P2.6 的
单输入 StableHLO reduce sum/max normalization 已落地：rank-0 init 通过 `tensor.extract` 和
`linalg.fill` materialize 成结果 tensor init，combiner 降到 `linalg.reduce` region。P2.7 的
RMSNorm / LayerNorm staged-form gate 已落地，复用 reduce、elementwise 和 broadcast indexing map
lowering，不引入 `wafer.norm` / `wafer.layer_norm` 高层 op。P2.8 的 softmax staged-form gate 已
落地，row max、shift、exp、row sum 和 normalize 都以 `linalg.reduce` / `linalg.elementwise`
及 broadcast indexing map 表达。P2.9 的 RoPE / MLP activation staged-form gate 已落地：
`stablehlo.concatenate` 降到 `tensor.concat`，`stablehlo.tanh` 降到 `linalg.elementwise`
tanh，RoPE 和 GELU tanh 形态只使用 slice/concat/elementwise structured tensor IR。P2.10 的
importer smoke gate 已落地：importer-enabled `wafer-import-model` 能发出并验证一个静态
StableHLO/MLIR artifact，graph break / eager fallback 通过 importer metadata 硬诊断，
unbounded dynamic shape 通过函数签名类型诊断；backend-only 构建仍保留 disabled importer shell，
不把 StableHLO/Python importer 依赖扩散到后端 textual tests。P3.2 的最小 group formation
已落地：`--wafer-form-groups` 将单结果 tensor `linalg.matmul` 包装成 logical `wafer.group`，
body 只通过显式 block arguments 访问 ins/outs，不写 tile shape、layout 或 resource plan；
多输出 `linalg.generic` 暂不形成 group，留给后续 planner 证明 traversal/domain 后再处理。P3.3 的
root tile candidate / feasibility skeleton 已落地：`--wafer-check-root-tile-candidates` 在
pass-local analysis 中为 M0 rank-2 static group result 建立候选并调用 feasibility checker；
动态 result shape 会被诊断为缺少 bounded tile policy，候选 shape 和资源估计不写入
`wafer.group` attr。P3.4 的 M0 single-tile materialization 已落地：`--wafer-materialize-single-tile`
将单 GEMM logical group 改写为 `wafer.tile_region`，region 内显式建立 external tensor
args 到 `wafer.load_tile`、`wafer.compute.gemm`、`wafer.store_tile` 和 `wafer.tile_yield` 的
SSA 链；为满足当前 `compute.gemm` / `store_tile` verifier，暂时插入 tensor<->cx layout
materialization，后续 P3.5 再做 compact layout assignment 和冗余 conversion 清理。
P3.5 的 compact layout cleanup 已落地：`--wafer-compact-layout-assignment` 删除无其它 use 的
inverse-pair `wafer.layout.materialize(A -> B -> A)`，直接把最终 consumer 改回原始 tile buffer；
该 pass 只做当前 IR 的 layout-aware rewrite，不保存或读取 planner 搜索状态。P3.6 的 SPM
allocation trial 已落地：`--wafer-check-spm-allocation` 在 `wafer.tile_region` 内用 pass-local
sequential trial 分配检查 SPM tile buffer storage size、alignment、lifetime range、end address 和
usable capacity；默认 usable cap 按 `0x2f0000` 建模，trial 不向 IR 写入 offset、range 或 allocation
plan attr。P3.7 的 DDR external binding demand 已落地：新增 `wafer.ddr.external_binding` op
表达 input/output 外部 tensor 的 compact byte size、alignment、read-only 和 host-visible contract，
`--wafer-materialize-ddr-external-bindings` 从 tile-region 内 `load_tile` / `store_tile` 的 boundary
use-def 推导 demand；它不携带 BO handle、physical address、pool/domain placement 或 allocation
trace。P3.8 的 C ABI skeleton lowering 已落地：新增 `wafer.abi.rdma_1d` /
`wafer.abi.wdma_1d` / `wafer.abi.gemm` issue ops，`--wafer-lower-to-c-abi-skeleton` 将
`wafer.load_tile`、`wafer.store_tile` 和 `wafer.compute.gemm` 替换为显式 bytes、M/K/N 和
`issue_only` wait policy 的 ABI skeleton，不生成 raw packet 或 runtime handle。P3.9 的 M0 ABI
unit gate 已落地：`Wafer/ABI/M0Abi.h` descriptor builder 和 gtest 覆盖 RDMA/WDMA 的 DDR/SPM
direction、byte range、exclusive end、SPM usable cap、DDR lower bound，以及 GEMM M/K/N 和
`issue_only` policy；当前固定 ABI argument contract，不声称已经完成真实 wrapper packet bitfield。
P3.10 的最小 package manifest 已落地：`tools/wafer_package_manifest.py` 支持 M0 smoke manifest
emit、validate 和 canonical roundtrip，schema 覆盖 launch signature、DDR external bindings、
resource summary、ABI ops、device-code artifact id 和 runtime completion source，并拒绝已知 stub
completion fence。P3.11 的 M0 local compile gate 已落地：汇总 lit test 从同一个 M0 Linalg GEMM
输入跑完整 textual pipeline 到 DDR binding / SPM allocation / C ABI skeleton，validate manifest，
再从 manifest 生成 C ABI stub 并用本地 C compiler 做 syntax compile。P4.1 的 logical rank 到
physical tile mapping 表示已落地：新增 `wafer.placement.map` accepted mapping op，按 logical
rank 顺序保存 4D physical coordinate，并由 verifier 检查 rank 覆盖、topology bounds、bad tile
过滤和重复 physical tile；该 op 不携带 planner trace、SPM/DDR allocation、DTE packet 或 runtime
handle。P4.2 的 placement package metadata 已落地：runtime manifest 现在记录 target topology、
good/bad tile assumption、per-rank physical coord、block id 和 local shard slice metadata；validator
检查 rank 覆盖、good/bad tile disjoint、mapped tile 必须 good、block/tile 不重复，以及 local shard
不能越过 launch signature tensor shape。P4.3 的 multi-tile no-comm outlining 已落地：
`--wafer-materialize-multi-tile-no-comm` 从唯一 `wafer.placement.map` 读取 `logical_rank_count`，
将单 GEMM group materialize 成多个独立 load-GEMM-store `wafer.tile_region` skeleton，并保持
`wafer.comm` 不进入该路径。P4.4 的 per-tile launch args 已落地：`wafer_emit_c_abi_stub.py`
从 package placement ranks 生成 `wafer_tile_launch_arg_t` C table，包含 logical rank、block id
和 physical coordinate，使 runtime launch artifact 能区分 tile-specific arguments；该表不携带
BO handle、DDR address、SPM offset 或 DTE packet。P4.5 的 M1 local compile gate 已落地：新增
two-tile no-comm integration test，从 textual Linalg GEMM 和 `wafer.placement.map` 跑到两个
independent tile-region 的 C ABI skeleton，validate `--emit-m1-no-comm-smoke` manifest，生成 C
stub 并由本地 C compiler 做 syntax compile。
P5.1 的 norm schedule acceptance gate 已落地：`--wafer-check-norm-schedule` 对 normalized
structured tensor IR 做 pass-local analysis，要求 hidden-dimension `linalg.reduce`、`rsqrt`
elementwise stage 和 rank-2/rank-1 broadcast multiply stage 同时存在；它不引入 `wafer.norm` 或
schedule attr，也不把 planner cost 写进 IR。
P5.2 的 softmax schedule acceptance gate 已落地：`--wafer-check-softmax-schedule` 对 normalized
structured tensor IR 做 pass-local analysis，要求 hidden-dimension reduce max、row max broadcast
subtract、`exp`、hidden-dimension reduce sum 和 final broadcast divide 按 SSA use-def 串联；它不引入
`wafer.softmax`、schedule attr 或 workspace/group split 决策。
P5.3 的 attention score QK^T slice 已落地：`--wafer-lower-stablehlo-dot` 现在支持 rank-4
`dot_general` 的 batch/head attention score 形态，从 StableHLO dimension numbers 验证 batch
dimensions `[0,1]` 和 contracting dimensions `[3]x[3]`，lowering 到带显式 indexing maps 和
parallel/reduction iterator types 的 `linalg.generic` contraction；不靠 query/key 名字恢复语义。

## Active Task

把文档中的 IR 分层落成最小可运行 compiler skeleton，并先跑通 M0/M1 load-GEMM-store 闭环。

当前执行焦点：

1. P0 到 P1：工程、依赖、工具、测试入口和最小 Wafer dialect skeleton。
2. P2 到 P3：从手写 StableHLO / Linalg GEMM 输入跑通 M0 local compile gate。
3. P5.1 到 P5.8：推进 transformer block local vertical slice 的 staged schedule acceptance 和
   local compile gate。

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
| P0.1 | done | 建立最小 CMake / build 入口 | 可以配置空项目；不要求已有完整 Wafer IR |
| P0.2 | done | 建立 LLVM / MLIR 依赖发现和版本 pin 机制 | 依赖由集中配置声明，不在源码里散落 include/library path |
| P0.3 | done | 加入 StableHLO / Shardy 依赖开关和 dialect registration 入口 | 可关闭 importer-only 依赖并运行 backend textual tests |
| P0.4 | done | 创建 `include/Wafer/`、`lib/Wafer/`、`tools/wafer-opt` 最小骨架 | `wafer-opt --help` 或等价 smoke test 可运行 |
| P0.5 | done | 建立 lit / FileCheck 测试目录和最小 test target | 一个空 dialect smoke test 能被 test runner 收集 |
| P0.6 | done | 建立 gtest 或等价 C++ unit test 入口 | 后续 allocator / storage calculator 有测试落点 |
| P0.7 | done | 创建 `tools/wafer-import-model` shell | importer-only 依赖缺失时后端仍可构建 |
| P0.8 | done | 增加依赖一致性检查脚本 | 能检查版本 pin、dialect registration 和 importer/backend 隔离 |

## P1. Wafer IR Skeleton 和 Verifier

目标：核心 IR 对象可以 parse / print / verify，先固定结构边界，不实现复杂 lowering。

| ID | 状态 | 任务 | 验收 |
| --- | --- | --- | --- |
| P1.1 | done | 定义 `WaferDialect` 和基础 ODS 文件组织 | dialect 可注册；空 module roundtrip |
| P1.2 | done | 定义共享 attrs/types：target、placement、memory space、mem layout | parser/printer roundtrip；非法 enum 被 verifier 拒绝 |
| P1.3 | done | 定义 `WaferTilingInterface`、`WaferLayoutOpInterface`、resource/effect 相关接口骨架 | ODS / C++ 编译通过；接口不携带 planner side table |
| P1.4 | done | 定义最小 `wafer.group` op | region / operand / result contract 可验证；不接受 SPM memref |
| P1.5 | done | 定义最小 `wafer.tile_region` op 和 tile buffer type | tile-local region boundary、buffer ownership、effect scope 可验证 |
| P1.6 | done | 定义 layout materialization op 的最小 IR 形态 | 表达真实 data movement；不作为 metadata cast |
| P1.7 | done | 定义 `wafer.compute` / movement 最小 op family：load tile、store tile、gemm | dtype、shape、layout contract 有 verifier |
| P1.8 | done | 定义 `wafer.comm` p2p、sync token、local wait 的最小表示 | endpoint、token、wait policy 可验证 |
| P1.9 | done | 定义 `wafer.launch` 最小 boundary | launch signature、resource summary、package ref 不反向污染 tensor IR |
| P1.10 | done | 写 parser/printer/verifier 正负例 | 每类 op/type/attr 至少一个 roundtrip 和一个 negative test |

## P2. Frontend Artifact 和 Local Compute Normalization

目标：先从手写 StableHLO / MLIR 输入建立稳定 backend 入口，再接真实 importer。

| ID | 状态 | 任务 | 验收 |
| --- | --- | --- | --- |
| P2.1 | done | 建立 StableHLO / MLIR textual artifact 输入测试 | parse/roundtrip 保留 function signature、shape、dtype |
| P2.2 | done | 实现 constant normalization 骨架 | `stablehlo.constant` / sidecar 进入 `arith.constant` 或 `ConstantLike` tensor value |
| P2.3 | done | Lower `dot_general` / matmul 到 structured tensor IR | indexing map / iterator / DPS 关系可 FileCheck |
| P2.4 | done | Lower broadcast、reshape、transpose、slice | 只依赖 type、shape、indexing relation，不靠名字 |
| P2.5 | done | Lower elementwise 和 limited broadcast | 覆盖 add/sub/mul/div/max/min/neg/recip/sqrt/rsqrt/exp 的基础形态 |
| P2.6 | done | Lower reduce max / reduce sum | 输出 staged reduce IR，可服务 softmax 和 norm |
| P2.7 | done | 表达 RMSNorm / LayerNorm staged form | reduce + elementwise，不引入 `wafer.norm` 高层 op |
| P2.8 | done | 表达 softmax staged form | row max、exp、row sum、normalize 状态由 SSA / loop-carried / workspace 表达 |
| P2.9 | done | 表达 RoPE 和 MLP activation staged form | 只使用 structured tensor IR 和 math/arith 语义 |
| P2.10 | done | 加 importer smoke test | graph break、eager fallback、unbounded dynamic shape 会被诊断 |

## P3. M0 Single Tile Load-GEMM-Store

目标：跑通最小 device program 链路。

| ID | 状态 | 任务 | 验收 |
| --- | --- | --- | --- |
| P3.1 | done | 准备手写 StableHLO / Linalg GEMM 输入 case | 输入不依赖模型 importer |
| P3.2 | done | 实现最小 group formation | 单 GEMM 能形成 `wafer.group`；非法 multi-output/domain 被拒绝或拆分 |
| P3.3 | done | 实现 root tile shape 候选和 feasibility 调用骨架 | tile shape 是 planner 候选，不写成 IR contract |
| P3.4 | done | materialize 单 tile `wafer.tile_region` | region 中有 load、compute、store 的 SSA 关系 |
| P3.5 | done | 实现 compact layout assignment | 不插不必要的 layout conversion |
| P3.6 | done | 实现 SPM allocation trial | range、alignment、lifetime、end-address 检查通过 |
| P3.7 | done | 实现 DDR external input/output binding demand | DDR demand 可被 launch/runtime 层消费 |
| P3.8 | done | Lower `wafer.compute.gemm` / load / store 到 C ABI skeleton | 参数单位、address domain、wait policy 有 verifier |
| P3.9 | done | 建立 golden packet / ABI unit test | 至少覆盖 M0 用到的 compute/movement family |
| P3.10 | done | 建立最小 launch/runtime package manifest | manifest roundtrip，package 不依赖已知 stub path 作为 correctness fence |
| P3.11 | done | M0 local compile gate 汇总测试 | textual pipeline、SPM、DDR、C ABI、generated artifact compile、manifest 检查全部通过 |

## P4. M1 Multi-Tile No Communication

目标：验证 placement、per-tile args 和多 tile launch，不引入 DTE。

| ID | 状态 | 任务 | 验收 |
| --- | --- | --- | --- |
| P4.1 | done | 定义 logical rank 到 physical tile mapping 表示 | mapping 覆盖所有 rank，不使用 bad tile |
| P4.2 | done | 接入 good-tile / block id / local shard metadata | metadata 可进入 package，不修改 tensor semantics |
| P4.3 | done | 支持多 tile 独立 tile-region outlining | 每 tile 独立 load-compute-store，无 tile 间 comm |
| P4.4 | done | 支持 per-tile launch args | runtime launch 能区分 tile-specific arguments |
| P4.5 | done | M1 local compile gate 汇总测试 | 多 tile no-comm pipeline、generated artifact compile 和 manifest 检查通过 |

## P5. Transformer Block Local Vertical Slices

目标：在单 shard / 单卡本地路径上逐步覆盖 transformer block。

| ID | 状态 | 任务 | 验收 |
| --- | --- | --- | --- |
| P5.1 | done | Norm slice：RMSNorm 或 LayerNorm | reduce + elementwise group schedule accepted |
| P5.2 | done | Softmax slice | row max、exp、row sum、normalize 的 staged schedule accepted |
| P5.3 | done | Attention score slice：QK^T | batch/head matmul relation 从 StableHLO dimension numbers / indexing map 推出 |
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

继续 P5.4，进入 attention value slice，检查 softmax output 到 value accumulation 的 structured
dataflow 和 buffer lifetime 边界。
不要越过当前 local compile gate 直接实现未验证的整块逻辑。
