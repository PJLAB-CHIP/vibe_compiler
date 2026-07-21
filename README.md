# Wafer AI Compiler

Wafer AI Compiler 是面向 Wafer/TX81 单卡目标的 MLIR 编译器、artifact 和 runtime 验证工程。当前主线接收
PyTorch/XLA 导出的 StableHLO program directory，经 Shardy/XLA SPMD、rank-local structured lowering、物理数据流
选择、SPM/DDR planning、通信和 target conversion，生成完整 rank domain 的 verified package。仓库同时提供
no-card preflight 和 repo-owned TargetCall/SystemC untimed functional model，用于在真实板卡接入前验证 compiler、ABI、
memory、transport 和数值语义。

项目仍处在设计收敛和实现推进阶段。当前 compiler/model-only 主线已经闭合；真实板端执行、板端数值相关、exact-package
ISS/vendor simulator、packet provenance 和 timing calibration 是独立的后续验证层，不能由 SystemC 结果替代。

## 编译流水线

```text
PyTorch/XLA StableHLO program directory
  -> frontend verification
  -> Shardy/XLA SPMD partitioning (1 or 16 logical ranks)
  -> rank-local Linalg/Tensor/SCF structured IR
  -> bounded actual-clone physical-dataflow selection
  -> Tile IR -> Instr IR
  -> SPM/DDR placement + completion + transport gates
  -> whole-card resource/Pareto selection and atomic winner commit
  -> Target LLVM modules + device link
  -> typed manifest + verified package
  -> no-card RuntimeSession / TargetCall + SystemC / configured board
```

最终 artifact 只保留 accepted typed IR、binding、offset、completion 和 transport 事实。候选 frontier、cost、ordinal、
analysis 和 rejected state 都是 compiler-private 的 invocation-local 状态，不进入 package。

## 当前能力

- **Frontend/SPMD**：验证 StableHLO program directory、typed inputs/parameters、Shardy/XLA SPMD 和显式 1/16-rank domain。
- **MLIR-native 选择**：通过 op interface、Affine/Presburger/ValueBounds relation 和 actual clones 联合选择 implementation、
  tile、encoding/view、route、residency、buffering/order 和 communication。
- **物理实现**：支持 Tensor/Cx/NCx physical encoding、metadata view、compact/mapped DMA、SPM gather/scatter、
  relation-backed resident handoff 和 fixed-capacity SPM/DDR packing。
- **计算与数据流候选**：覆盖 share/recompute、静态 LICM、当前 modular-integer algebra 变体、spill/resident、
  movement-first ready order，以及 direct/ring/tree collective alternatives。
- **Typed target capability**：覆盖 mapped RDMA/WDMA offset、physical-footprint fill、GEMM/batched GEMM 和 versioned
  oriented GEMM ABI。
- **原子 artifact**：完整 rank tuple 通过 DDR、NoC、instruction、event、ABI、device-link、manifest 和 readback gate 后，
  才发布 `ExecutableBundle`、Target LLVM modules 和 verified package。
- **功能数值验证**：同一 target lowering 可由 TargetCall/SystemC model 消费，并与独立 CPU expected 比较完整输出；
  标准 Llama-2 7B 单 block TP16 fixed/held-out replay 已作为当前 scale evidence。

当前完成状态和精确边界以 [`tasks/progress.md`](tasks/progress.md) 为准；Q32 physical-dataflow synthesis 的集成证据见
[`tasks/archive/physical-dataflow-synthesis-completion-audit.md`](tasks/archive/physical-dataflow-synthesis-completion-audit.md)。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| `include/Wafer/` | Dialect、interface、analysis、compiler/runtime 公共接口 |
| `lib/Wafer/` | Frontend、SPMD、scheduling、conversion、compiler、artifact、runtime 和 model 实现 |
| `tools/` | `wafer-compile`、`wafer-opt`、StableHLO 工具、依赖 bootstrap 和一致性检查 |
| `test/` | lit/FileCheck、CLI 和 Python tool tests |
| `unittests/` | C++ unit、numeric/bulk 和可选 SystemC tests |
| `tasks/` | 当前编号设计合同、任务队列、实施计划和历史审计 |
| `docs/` | 硬件、runtime、ABI 和依赖逆向事实资料 |
| `memory/` | 稳定构建、调试和防复发经验，不是任务状态或架构合同 |
| `third_party/` | 固定版本 LLVM/MLIR、StableHLO、Shardy/XLA、工具链及受管 model 依赖 |

## 获取依赖

首次 checkout 先同步固定版本 submodules：

```bash
git submodule update --init --recursive
```

依赖版本集中定义在 `cmake/third_party/WaferDependencyVersions.cmake`。常用 bootstrap 入口包括：

```bash
python3 tools/bootstrap_deps.py --python
python3 tools/bootstrap_deps.py --llvm-source
python3 tools/bootstrap_deps.py --numeric-model-deps
python3 tools/bootstrap_deps.py --bulk-model-deps
python3 tools/bootstrap_deps.py --systemc-model-deps
```

这些命令按需执行；core compiler、importer、bulk model 和 SystemC 的完整准备方式不同。开始构建前请阅读
[`third_party/README.md`](third_party/README.md) 和 [`memory/general_dev.md`](memory/general_dev.md)，不要用未固定的系统依赖
冒充正式验证环境。

## 构建与测试

使用与仓库固定 commit 匹配、已经 build/install 的 LLVM/MLIR：

```bash
cmake -S . -B build/wafer-dev -GNinja \
  -DMLIR_DIR=<llvm-install>/lib/cmake/mlir \
  -DLLVM_DIR=<llvm-install>/lib/cmake/llvm

cmake --build build/wafer-dev --target check-wafer -- -j<N>
ctest --test-dir build/wafer-dev --output-on-failure
```

常用验证入口：

- `check-wafer-lit`：lit/FileCheck/Python tool tests；
- `check-wafer-unit`：C++ unit tests；
- `check-wafer`：执行当前配置中实际存在的全部 mandatory 子 gate；
- `ctest --output-on-failure`：运行已注册的配置、依赖、feature 和 integration tests。

启用 importer、Shardy/XLA helper、numeric/bulk model 或 SystemC 时，需要相应的受管依赖记录和 CMake feature。测试报告中的
`unsupported` 必须按当前 feature matrix 单独审计；`ctest passed` 不能替代对关键 source/program gate 是否实际执行的检查。

## 使用默认编译入口

`wafer-compile` 是唯一 production decision owner。它要求显式指定 rank domain 和 target profile：

```bash
build/wafer-dev/bin/wafer-compile \
  --input-program-dir <stablehlo-program-dir> \
  --output-program-dir <verified-package-dir> \
  --execution-ranks 16 \
  --target-profile wafer-tx81-single-card-kernel-v1
```

使用 `--execution-ranks 1` 可运行单 rank domain。完整 TargetCall/SystemC 参数通过
`build/wafer-dev/bin/wafer-compile --help` 查看；target-model 模式必须提供显式 input、CPU expected、数值 policy 和 resource
budget。

`wafer-opt` 只用于局部 MLIR 调试和底层 rewrite/conversion，不拥有另一套 candidate selector，也不能把单 pass 输出直接当成
production package。

## 当前边界

- 当前 production domain 是单卡、显式 1 或 16 logical ranks；cross-card、MPMD、dynamic shape/state、MoE 和持久化权重缓存
  尚未进入主线。
- floating reassociation/tree、generic online reduction 和 non-GEMM FMA contraction 尚未开放；当前 algebraic variants 只覆盖
  已证明的 exact/modular integer 子集。
- SystemC 是 untimed functional-event model，不证明 vendor packet、RISC-V ELF exact execution、板端性能或 cycle accuracy。
- 真实板端必须独立完成 allocation/import、H2D、load、submit、wait/status、D2H、完整输出比较和 cleanup。
- 完整 7B bounded frontier 属于长时间 scale gate，不应作为每次局部修改的日常测试入口。

## 文档与协作

- [`AGENTS.md`](AGENTS.md)：仓库稳定协作、IR、pipeline、验证和提交规则；
- [`tasks/progress.md`](tasks/progress.md)：当前任务状态和前置关系的唯一入口；
- [`tasks/README.md`](tasks/README.md)：编号设计文档、pipeline owner 和历史归档导航；
- [`tasks/01-architecture.md`](tasks/01-architecture.md)：主架构与 artifact DAG；
- [`tasks/16-verification-contract.md`](tasks/16-verification-contract.md)：分层 verification contract；
- [`memory/general_dev.md`](memory/general_dev.md)：本地构建、依赖、调试和验证经验。

提交变更前必须保留无关工作区修改，运行与变更范围匹配的验证，并同步受影响的设计文档、任务队列和稳定经验。
