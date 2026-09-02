<p align="center">
  <img src="docs/images/wafer-compiler-mark.svg" alt="Wafer Compiler" width="180">
</p>

<h1 align="center">Wafer Compiler</h1>

Wafer Compiler 是一个基于 MLIR 的 Wafer 加速器编译器和运行时工具集。它把 PyTorch/XLA 导出的
portable StableHLO program 转换为面向单卡 16-Tile 设备的可执行 package，并提供 host、no-card、
功能模型和板端运行入口。

当前产品路径面向 static ranked、单 logical card partition 的程序。`none` 是确定性的 baseline，
`search` 在同一套 current IR 上搜索并比较实际候选；两者使用同一组下游 lowering 和 package 代码，
不会互相 fallback。当前任务状态和设备资格以 [`tasks/progress.md`](tasks/progress.md) 为准。

## Overview

```mermaid
flowchart LR
  A[StableHLO program] --> B[Verify source and payload]
  B --> C[Card-level SPMD]
  C --> D[Structured TensorProgram]
  D --> E{none | search}
  E --> F[Spatial and temporal IR]
  F --> G[Layout, movement, execution]
  G --> H[Instr and memory planning]
  H --> I[DeviceExecutable]
  I --> J[Target module and package]
  J --> K[no-card | board]
```

Structured graph normalization使用 pinned `egg`，只处理能够从当前 Tensor/Linalg SSA、indexing relation
和 effect 证明的等价变换。FA/FD 在 structured attention 语义已经归一后进入 physical-dataflow pipeline；
后续 Tile、layout、movement、completion 和 target lowering 都只消费实际物化的 current IR。

## Quick start

### Dependencies

固定版本依赖位于 `third_party/`，准备方式见 [`third_party/README.md`](third_party/README.md)。首次 checkout 后：

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

按需准备已有的 pinned 依赖即可；不要为单个任务创建额外 build 目录。

### Configure and build

仓库使用一个 canonical Ninja build，binary directory 固定为 `build/`：

```bash
cmake --preset default
cmake --build --preset default -j"$(nproc)"
```

default preset 打开 compiler、PyTorch/XLA importer、SPMD、target numeric backend、SystemC 和本地测试，
关闭真实 board SDK 和设备执行。Pinned e-graph 依赖由 CMake 以 locked/offline 方式构建；编译器调用不会
启动 Cargo、rustc 或外部 optimizer 进程。

### Compile a program

PyTorch 模型可以通过 `wafer.frontend.export_pytorch_program` 生成 program directory：

```python
from wafer.frontend import export_pytorch_program

export_pytorch_program(module, example_inputs, output_directory)
```

然后使用唯一的 production driver：

```bash
build/bin/wafer-compile \
  --input-program-dir <program-directory> \
  --output-dir <package-directory> \
  --num-partitions 1 \
  --optimization-policy search
```

`--optimization-policy none` 选择 baseline；`search` 还可以设置 `--search-width` 和 `--search-trials`。
`--compile-timing`、`--dump-compiler-ir` 和 `--profile` 是诊断/资格选项，仍然复用同一个 compiler transaction。

编译成功时，`--output-dir` 中会原子发布一个经过 strict readback 的 package；失败时不会留下可见的部分输出。

## Run and verify a package

无设备验证使用同一 package：

```bash
build/bin/wafer-run \
  --package-dir <package-directory> \
  --no-card
```

真实设备运行需要显式的 `--board`、设备身份、runtime digest、输入资源和 expected 输出。默认 build 不包含
board runtime；板端测试只在显式配置并取得资格后注册。

`wafer-opt` 用于局部 MLIR 调试和 registered pipeline replay；`wafer-verify-program` 用于检查 source
program directory。它们都不是第二个 production compiler。

## Repository layout

| Directory | Contents |
| --- | --- |
| `include/Wafer/IR`, `lib/Wafer/IR` | Wafer dialect、ODS、interface 和 verifier |
| `include/Wafer/Analysis`, `lib/Wafer/Analysis` | current IR 的只读分析 |
| `include/Wafer/Planning`, `lib/Wafer/Planning` | physical-dataflow choice、PBQP 和 search traversal |
| `lib/Wafer/Transforms` | structured、Tile、movement、layout、memory 和 completion 变换 |
| `lib/Wafer/Conversion` | StableHLO→Linalg、Tile→Instr、Instr→LLVM 转换 |
| `lib/Wafer/CodeGen`, `lib/Wafer/Target` | DeviceExecutable、Target LLVM、target ABI 和格式 |
| `lib/Wafer/Frontend`, `lib/Wafer/Driver` | source ingestion、payload ownership 和 compiler transaction |
| `lib/Wafer/Package`, `lib/Wafer/Runtime` | package schema、readback 和 invocation lifecycle |
| `lib/Wafer/Simulator` | Reference、oneDNN 和 SystemC 功能模型 |
| `tools/` | `wafer-compile`、`wafer-opt`、`wafer-run`、source verifier 和辅助工具 |
| `test/`, `unittests/` | lit/FileCheck、integration、host/no-card、C++ unit 和 board contract tests |
| `tasks/`, `docs/`, `memory/` | 设计合同、硬件事实和稳定开发方法 |

`num_partitions` 是 card-level SPMD 数量，不是 Tile 数量；`tile_id` 和 `launch_slot` 在 physical target/package
层独立表示。

## Testing

```bash
ctest --preset default -j"$(nproc)"
cmake --build --preset default --target check-wafer -j"$(nproc)"
python3 -B utils/checks/check_source_organization.py --root .
python3 -B utils/checks/check_ir_organization.py --root .
python3 -B utils/checks/check_deps.py
```

主线 IR 测试使用 rank≥3、主要维度≥1024 的 static shape，并覆盖 1024、1025、1031 的整除和非整除路径。
正例需要检查实际 coverage、owner、demand、tail、copy、completion 或直接下游输出；skip、unsupported、
未注册 case 和历史日志不算通过。

测试证据按层级区分：compile/IR、package readback、no-card、功能模型、board correctness 和 board performance
不能互相代签。SystemC 只提供 untimed functional evidence，不证明真实硬件时序或性能。

## Current scope

- 单卡、单 logical card partition、16 Tile、static ranked 输入；
- PyTorch/XLA portable StableHLO ingestion；
- structured graph normalization、FA/FD online-attention lowering、current-IR layout/bufferization、
  movement、Instr、Target LLVM、package 和 no-card 验证；
- cross-card transport、dynamic-ranked program、resident execution、MoE 和 persistent weight cache 不在当前产品范围内。

## Documentation

- [`AGENTS.md`](AGENTS.md)：协作流程、IR 事实源、MLIR 规则和提交约束；
- [`tasks/progress.md`](tasks/progress.md)：任务状态、线性顺序和直接前置；
- [`tasks/README.md`](tasks/README.md)：编号设计与 archive 导航；
- [`tasks/01-architecture.md`](tasks/01-architecture.md)：整体 pipeline 和 artifact ownership；
- [`tasks/06-physical-dataflow-synthesis.md`](tasks/06-physical-dataflow-synthesis.md)：physical-dataflow 主合同；
- [`tasks/16-verification-contract.md`](tasks/16-verification-contract.md)：分层验证与 board-ready 规则；
- [`tasks/18-source-organization.md`](tasks/18-source-organization.md)、[`tasks/19-mlir-engineering.md`](tasks/19-mlir-engineering.md)：源码和 MLIR 工程约束；
- `docs/`：硬件事实和外推限制；
- [`memory/general_dev.md`](memory/general_dev.md)：构建、调试和验证方法。
