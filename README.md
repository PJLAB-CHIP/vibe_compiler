# Wafer Compiler

Wafer Compiler 是面向 Wafer/TX81 单卡目标的 MLIR 编译器、目标代码生成器和运行时工具集。当前产品边界是一个
logical card partition、卡内 16 个 Tile 和 static ranked 输入。产品输入来自 PyTorch/XLA exporter 生成的
StableHLO program directory，也可以使用同一格式的 portable StableHLO fixture。

当前生产入口支持两个相互独立的 optimization policy：`none` 使用确定性的 baseline，`search` 在 current IR 上搜索
并比较实际候选。两条 policy 共享经过验证的下游 atomic stages，但不互相调用、fallback 或重建 winner。当前任务状态和
板端资格以 [`tasks/progress.md`](tasks/progress.md) 为准；本 README 只说明稳定的入口和架构边界。

## 端到端编译主线

```text
PyTorch/XLA exporter / portable StableHLO program directory
  -> source and payload verification
  -> XLA SPMD partitioning (logical card partition)
  -> post-SPMD StableHLO readback and shard verification
  -> StableHLO -> Linalg/Tensor/SCF TensorProgram
       - collective normalization and static cleanup
       - one structured attention semantic op
       - bounded structured-graph e-graph normalization
  -> policy split: none | search
  -> Spatial/Region actualization
       - ordinary TileModule/TileRegion
       - FA/FD online-attention state and selected merge endpoints
  -> temporal tiling and fusion from live current operations
  -> online-attention decomposition to actual Linalg/Tensor/SCF
  -> current-IR layout assignment and bufferization
  -> structured compute -> Tile IR
  -> boundary movement and cross-Tile transport closure
  -> execution structure and standalone Tile fanout
  -> TileRegion -> Instr, transfer cleanup and fresh completion
  -> actual MiniMalloc SPM planning, DDR offsets and Direct-DTE/resource gates
  -> DeviceExecutable
  -> Target LLVM conversion, link and readback
  -> atomic ExecutablePackage publication
  -> wafer-run --no-card or --board
```

Current IR 是每一阶段的唯一事实源。planning 只保存显式 transformation choice；operation、SSA、buffer、alias、
lifetime、movement、event、order 和 completion 只有在 candidate-owned transaction 中物化并通过 verifier 后才存在。
SPM 合法性只来自实际 MiniMalloc 结果，不由 footprint、shape 公式或其它预测性估算决定。

## 产品入口

`wafer-compile` 是唯一的 source-to-package production driver：

```bash
build/bin/wafer-compile \
  --input-program-dir <stablehlo-program-dir> \
  --output-dir <package-root> \
  --num-partitions 1 \
  --optimization-policy search
```

可选参数包括 `--optimization-policy none`、`--search-width <count>`、`--search-trials <count>`、
`--compile-timing`、`--dump-compiler-ir <dir>` 和 `--profile`。`--dump-compiler-ir`、target-model qualification
以及测试故障注入都复用同一个 compiler transaction，不形成第二条物理 lowering 路径。

编译成功表示 package 已完成 target/module/manifest strict readback 并通过最后一次原子提交；失败时不会留下可见的
partial package。已经生成的 package 可通过 no-card 入口检查：

```bash
build/bin/wafer-run \
  --package-dir <package-root> \
  --no-card
```

`wafer-opt` 只用于局部 IR 调试、registered pipeline 和 FileCheck replay；它复用相同的 pass/transform 实现，但不拥有
产品 candidate selector。`wafer-verify-program` 只验证 source program directory，不代替 compiler transaction。

## IR 和源码边界

| 路径 | 责任 |
| --- | --- |
| `include/Wafer/IR`、`lib/Wafer/IR` | Wafer dialect、ODS、interface、局部 verifier 和 IR parser/printer |
| `Analysis` | 从 current IR 和显式 target facts 重算的只读分析 |
| `Planning` | Spatial/Region/Temporal choice、PBQP 和有界 search traversal；不拥有 actual IR |
| `Transforms` | 同一主要表示层上的 current-IR rewrite、materialization、layout、movement 和 completion 变换 |
| `Conversion` | 有明确 source/target 表示和 legality 合同的 StableHLO→Linalg、Tile→Instr、Instr→LLVM 转换 |
| `CodeGen` | DeviceExecutable、Target LLVM、ABI preparation、link 和 artifact readback |
| `Driver` | policy routing、candidate transaction、pipeline 编排、外部 helper 和 package publication |
| `Target` | compiler/runtime/Simulator 共享的 target command、format、memory、identity 和 launch 合同 |
| `Simulator` | TargetCall/SystemC/Reference/oneDNN 的同 invocation 功能模型 |
| `Package`、`Runtime` | package schema/readback，以及 no-card/board invocation lifecycle |

`num_partitions` 是 GSPMD 的 card-level 数量，不是 Tile 数量；`tile_id` 和 `launch_slot` 由物理 target/package 层独立
表达。当前生产不覆盖 cross-card transport、dynamic-ranked program、resident execution、MoE 或 persistent weight cache。

## 依赖和 canonical build

固定版本和受管依赖位于 `third_party/`，细节见 [`third_party/README.md`](third_party/README.md)。首次准备依赖可按需执行：

```bash
git submodule update --init --recursive
python3 utils/deps/bootstrap_deps.py --python
python3 utils/deps/bootstrap_deps.py --llvm-source
python3 utils/deps/bootstrap_deps.py --importer-sources --importer-python
python3 utils/deps/bootstrap_deps.py --egraph-sources
python3 utils/deps/bootstrap_deps.py --numeric-model-deps
python3 utils/deps/bootstrap_deps.py --onednn-deps
python3 utils/deps/bootstrap_deps.py --systemc-model-deps
```

这些步骤按实际配置选择，不需要为每个任务建立新的 build。主工程只有一个 canonical binary directory：`build/`。

```bash
cmake --preset default
cmake --build --preset default -j"$(nproc)"
ctest --preset default -j"$(nproc)"
```

default preset 打开 compiler、importer、SPMD、numeric backend、SystemC 和本地测试，关闭外部 board SDK 与真实设备执行。
Pinned egg 由 CMake 以 locked/offline 方式构建；compiler invocation 不启动 Cargo、rustc 或外部 optimizer 进程。

## 测试和证据层级

常用检查入口：

```bash
cmake --build --preset default --target check-wafer -j"$(nproc)"
ctest --test-dir build -R 'Wafer.*UnitTests|wafer-lit' --output-on-failure -j"$(nproc)"
python3 -B utils/checks/check_source_organization.py --root .
python3 -B utils/checks/check_ir_organization.py --root .
python3 -B utils/checks/check_deps.py
```

IR 正例通常使用 rank≥3、主要维度≥1024，并成对覆盖 1024、1025 和 1031 的整除/非整除路径。测试必须检查实际
coverage、owner、demand、merge、tail、copy、completion 或直接下游输出；`skip`、`unsupported`、未注册和旧日志不算通过。

证据按层级区分：host/IR 检查、package readback、no-card、功能模型、真实板端 correctness 和 matched board performance
不能互相代签。SystemC 是 untimed functional model，不证明 vendor packet、cycle accuracy 或板端性能。

## 文档入口

- [`AGENTS.md`](AGENTS.md)：仓库长期协作流程、Current-IR、SPM、completion、MLIR 和提交规则；
- [`tasks/progress.md`](tasks/progress.md)：当前任务状态、顺序和直接前置的唯一入口；
- [`tasks/README.md`](tasks/README.md)：编号设计、pipeline owner 和 archive 导航；
- [`tasks/01-architecture.md`](tasks/01-architecture.md)：整体 pipeline 与 artifact ownership；
- [`tasks/06-physical-dataflow-synthesis.md`](tasks/06-physical-dataflow-synthesis.md)：current-IR physical-dataflow 主合同；
- [`tasks/16-verification-contract.md`](tasks/16-verification-contract.md)：分层验证和 board-ready 规则；
- [`tasks/18-source-organization.md`](tasks/18-source-organization.md)、[`tasks/19-mlir-engineering.md`](tasks/19-mlir-engineering.md)：源码和 MLIR 工程边界；
- [`docs/tx81-compiler-hardware-calibration.md`](docs/tx81-compiler-hardware-calibration.md)：硬件事实及外推限制；
- [`memory/general_dev.md`](memory/general_dev.md)：canonical build、调试和验证方法。
