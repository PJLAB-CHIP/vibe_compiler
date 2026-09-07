<p align="center">
  <img src="docs/images/wafer-compiler-mark.svg" alt="Wafer Compiler：将程序映射到 Tile 阵列" width="200">
</p>

<h1 align="center">Wafer Compiler</h1>

<p align="center">从张量程序到单卡多 Tile 执行的 MLIR 编译器</p>

Wafer Compiler 接收 PyTorch/XLA 导出的 portable StableHLO program，完成图归一化、Tile 划分、切块与融合、
布局与数据搬运、指令生成和内存规划，最终交付经过验证的 `ExecutablePackage`。
当前 production 后端面向 TX81 单卡 16-Tile 设备；仓库同时包含 package/runtime 工具、功能数值模型和板端 profiler。

通用图算法与硬件传输分层：FA/FD 是 attention 算法，Ring、recursive doubling、dimension-ordered AllToAll 等是
collective 的物化选择；TX81 DTE/NCC 和 runtime ABI 负责消费已经生成的指令与 peer 数据流，不定义上层算法语义。

生产编译入口提供两种策略：

- `none`：按固定规则直接构造 baseline IR，不建立 search frontier；
- `search`：将候选选择实际物化为 IR，验证并重新分析后再比较成本。

两种策略各自持有独立 IR，复用后续变换、lowering 和 package 实现，不互相 fallback。当前任务和设备资格状态记录在
[`tasks/progress.md`](tasks/progress.md)，不在 README 中重复维护。

## 总览

[![Wafer Compiler 架构：真实 IR、编译产物与直接消费者](docs/images/wafer-compiler-pipeline.svg)](docs/images/wafer-compiler-pipeline.svg)

从上往下读：方框是 IR 或产物，箭头旁是该边界执行的变换，右侧是选择、只读输入或独立消费者。
图的稳定合同见 [01 号架构设计](tasks/01-architecture.md) 和 [06 号 physical-dataflow 设计](tasks/06-physical-dataflow-synthesis.md)。

- `num_partitions` 表示 card 级逻辑分区，当前为 `1`；不是把 Tile 数设为 `1`。
- Tile 放置、执行 region、temporal tiling、layout/bufferization、movement 与 completion 各有边界；SPM 合法性只能由
  actual IR 上的 allocation、alias、effect、lifetime 和实际 offset 规划确定。
- `DeviceExecutable` 在 target codegen **之前**形成。成功候选保留同一 actual IR owner，不从旁路计划重建；search 状态不进入 package。
- TargetCall/SystemC 功能模型消费同次 lowering 的 target modules；它不执行 package 内的 RISC-V ELF，也不证明板端性能。

## 快速开始

### 准备依赖

固定版本依赖和可选组件见 [`third_party/README.md`](third_party/README.md)。完整开发环境可以执行：

```bash
git submodule update --init --recursive
python3 -B utils/deps/bootstrap_deps.py --all
```

PyTorch/XLA exporter 还需要按依赖文档构建并安装 pinned `torch_xla`；bootstrap 的 Python 依赖安装不替代该步骤。
如果依赖已经准备好，可以直接配置仓库。

### 配置和构建

仓库只使用一个 canonical Ninja build 目录：`build/`。

```bash
cmake --preset default
cmake --build --preset default -j"$(nproc)"
```

default preset 打开 compiler、importer、SPMD、target numeric backend、功能模型和本地测试；board SDK 集成和真实设备执行需要显式配置。

### 导出并编译程序

Python 前端负责写出 compiler 使用的 program directory：

```python
from wafer.frontend import export_pytorch_program

export_pytorch_program(module, example_inputs, output_directory)
```

这里的 `module` 是 `torch.nn.Module`，`example_inputs` 是静态形状输入，输出目录必须尚不存在。
在源码 checkout 使用时，将 `python/` 加入 `PYTHONPATH`，以 `python3 -B` 运行导出脚本，避免在源码目录生成 Python cache。

使用 production driver 编译：

```bash
build/bin/wafer-compile \
  --input-program-dir <program-directory> \
  --output-dir <package-directory> \
  --num-partitions 1 \
  --optimization-policy search
```

普通编译的 `<package-directory>` 就是交付 package；编译输出目录必须尚不存在。
使用 `--optimization-policy none` 选择 deterministic baseline。`search` 可以用 `--search-width` 和 `--search-trials` 限制搜索工作量；
有界搜索不承诺全局最优。

## 验证和运行 package

no-card 验证会检查 package、launch contract、memory plan 和 transport binding，但不会调用设备 provider：

```bash
build/bin/wafer-run \
  --package-dir <package-directory> \
  --no-card
```

包含 Direct-DTE 的 package 还需给 no-card 指定对应能力：`--direct-dte-status-abi wafer-direct-dte-status --supports-host-watchdog`。
这些参数只声明本次主机验证采用的 runtime 能力，不会开启设备执行。

board 执行使用 `wafer-run --board`，需要显式设备资格、完整输入绑定以及已启用 board runtime 的构建。输入/输出 raw 文件按
manifest 的 port、dtype、physical layout 和 byte count 绑定。参数见 `build/bin/wafer-run --help`，合同见
[15 号 package/runtime 设计](tasks/15-launch-runtime-package.md)。default build 不执行真实设备。

`wafer-opt` 用于本地 MLIR registered pipeline 和 FileCheck case。`wafer-verify-program` 用于验证 source program directory。
两者都不是第二个 production compiler。

## 诊断

编译时用 `--compile-timing` 查看主机编译耗时，`--dump-compiler-ir <dir>` 导出各阶段实际 IR。
`--profile` 生成板端诊断产物；此时运行入口为 `--package-dir <output-dir>/package`，成功板端采集后的报告位于
`<output-dir>/package.profile/runs/current/index.html`。no-card 不进行性能采集。

## 目录结构

| 目录 | 责任 |
| --- | --- |
| `include/Wafer/IR`, `lib/Wafer/IR` | Dialect、ODS、interface 和 verifier |
| `include/Wafer/Analysis`, `lib/Wafer/Analysis` | 针对 current IR 的只读 analysis |
| `include/Wafer/Planning`, `lib/Wafer/Planning` | tiling、region、layout 和 search choice |
| `lib/Wafer/Transforms` | IR 变换、layout、movement、memory 和 completion |
| `lib/Wafer/Conversion` | StableHLO→Linalg、Tile→Instr 和 Instr→LLVM conversion |
| `lib/Wafer/CodeGen`, `lib/Wafer/Target` | Device executable、target module 和 target contract |
| `lib/Wafer/Frontend`, `lib/Wafer/Driver` | 输入导入、transaction 和 pipeline 编排 |
| `lib/Wafer/Package`, `lib/Wafer/Runtime` | package schema、readback 和 invocation lifecycle |
| `lib/Wafer/Simulator` | Reference、oneDNN 和 SystemC 功能模型 |
| `tools/` | compiler、optimizer、verifier、runner 和辅助工具 |
| `test/`, `unittests/` | lit、integration、host/no-card、unit 和 board contract 测试 |
| `tasks/`, `docs/`, `memory/` | 设计合同、硬件事实和开发方法 |

## 测试

```bash
# 按名称定向运行受影响测试
ctest --preset default -j"$(nproc)" -R '<test-name>'

# 完整本地主机测试（包含较重的模型 case）
ctest --preset default -j"$(nproc)"

# 文本与源码组织检查
python3 -B utils/checks/check_source_organization.py --root .
python3 -B utils/checks/check_ir_organization.py --root .
python3 -B utils/checks/check_deps.py
```

IR 变换测试使用接近生产的 static rank，并覆盖整除和 tail 维度。通过条件不仅是命令返回成功，还必须检查本阶段结果、owner
和直接下游 witness；skip、unsupported 和未注册 case 不计入覆盖率。
开发时先做定向回归，重型模型在前层通过后集中验证；具体流程见 [AGENTS.md](AGENTS.md)。

## 当前范围

- 单卡、一个 logical card partition 和 16 个 Tile；
- static ranked program boundary；
- PyTorch/XLA portable StableHLO 导入；
- structured normalization、FA/FD attention lowering、TileRegion 物化、layout/bufferization、movement、Instr、
  target module、package assembly 和 no-card 验证；
- cross-card transport、dynamic-ranked program、resident execution、MoE 和 persistent weight cache 不在当前 production 范围内。

## 文档

- [`AGENTS.md`](AGENTS.md)：开发流程和 compiler 工程规则；
- [`tasks/progress.md`](tasks/progress.md)：当前任务队列和依赖关系；
- [`tasks/README.md`](tasks/README.md)：编号设计文档和 archive 索引；
- [`tasks/01-architecture.md`](tasks/01-architecture.md)：pipeline 和 artifact ownership；
- [`tasks/06-physical-dataflow-synthesis.md`](tasks/06-physical-dataflow-synthesis.md)：physical-dataflow 合同；
- [`tasks/16-verification-contract.md`](tasks/16-verification-contract.md)：验证层级和资格门禁；
- [`tasks/18-source-organization.md`](tasks/18-source-organization.md) 和 [`tasks/19-mlir-engineering.md`](tasks/19-mlir-engineering.md)：源码和 MLIR 组织；
- [`memory/general_dev.md`](memory/general_dev.md)：构建、调试和验证方法。

## 相关项目

- [LLVM](https://github.com/llvm/llvm-project)
- [MLIR](https://mlir.llvm.org/)
- [StableHLO](https://github.com/openxla/stablehlo)
- [OpenXLA XLA](https://github.com/openxla/xla)
- [PyTorch/XLA](https://github.com/pytorch/xla)
- [IREE](https://github.com/iree-org/iree)
