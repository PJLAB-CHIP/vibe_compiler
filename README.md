# Wafer AI Compiler

Wafer AI Compiler 是面向 Wafer/TX81 单卡目标的 MLIR 编译器与runtime。产品输入是PyTorch/XLA导出的
StableHLO program directory；目标流水线在card-level GSPMD之后形成card-local structured IR，选择并物化
multi-Tile dataflow，再生成`DeviceExecutable`和verified package。仓库同时包含no-card validation以及
TargetCall/SystemC untimed functional model，用于分层验证compiler、ABI、memory和transport合同。

当前源码正在重建physical-dataflow的current-IR链。`none`和`search`两个产品policy目前都会明确返回
`operation_not_supported`，不会发布package；可用组件、恢复顺序和完成门禁只以
[`tasks/progress.md`](tasks/progress.md)为准。下面的流水线描述目标架构，不表示每个产品stage当前均已接通。

logical card partition 和 Tile 是两个不同的 domain：`num_partitions` 只属于 GSPMD/global tensor
boundary，card-local MPMD 由builtin module中的16个top-level
`wafer.tile.module(card_id, tile_id)`显式表示。current package schema
只包含 all-and-only Tile executable 和 card resource/launch 合同；不再存在 logical rank 直接绑定
Tile、single-Tile entry ABI 或兼容 reader。

## 目标编译流水线

```text
PyTorch/XLA StableHLO program directory
  -> frontend verification
  -> Shardy/XLA SPMD partitioning (logical card partitions)
  -> card-local Linalg/Tensor/SCF structured DAG
  -> policy-owned spatial/region/temporal choices
  -> actual TileRegion materialization and current-IR transformations
  -> top-level wafer.tile.module(card_id, tile_id)*
  -> standalone per-Tile modules
  -> Tile IR -> Instr IR + completion + SPM/DDR + Direct-DTE gates
  -> module/executable verification and policy acceptance
  -> DeviceExecutable
       ├─ same-lowering Target LLVM -> TargetCall/SystemC
       └─ device link -> typed manifest -> verified package
                            ├─ no-card package validation
                            └─ configured TX81 RuntimeProvider
```

最终IR、target modules和package只保留accepted typed IR、binding、offset、completion和transport事实。
Choice、cost、analysis和rejected candidate state都是query-local compiler state，不进入IR或package。

## 当前仓库边界

- Frontend、StableHLO program验证、Shardy/XLA adapter、Wafer dialect及各atomic transformation/conversion以独立
  library和registered pipeline存在。
- Structured logical normalization、physical relation、layout assignment、movement、execution structure、Instr、
  SPM/DDR、transport、target module、package和runtime分别有明确owner；它们不能代替尚未接通的产品纵向。
- `wafer-opt`用于局部IR调试和registered pipeline验证；它不拥有另一套产品candidate selector。
- TargetCall和SystemC是target-lowering的功能模型入口，不是`wafer-compile`的替代成功路径。
- `wafer-run --no-card`验证已经存在的current package；它不能把历史package或fixture代签为本轮compile结果。
- TX81事实按`supported`、`board-observed`、`unknown`和`excluded`分级；unknown不会被推测成合法性或同步行为。

当前主线是06号设计拥有的physical-dataflow current-IR materialization。动态状态、线性顺序和完成门禁只看
[`tasks/progress.md`](tasks/progress.md)；README不复制实施日志或历史性能结论。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| `include/Wafer/` | IR、Analysis、Planning、Transforms、Conversion、CodeGen、Driver、Target、Simulator、Package、Runtime的稳定typed API |
| `lib/Wafer/` | 与public component对应的实现；analysis/planning/rewrite/conversion/codegen/driver按职责分库 |
| `tools/` | `wafer-compile`、`wafer-run`、`wafer-opt`、`wafer-verify-program`、profile report和device linker等产品入口 |
| `utils/` | 依赖准备、源码/IR/ABI一致性检查和hardware calibration编排等开发脚本 |
| `runtime/crt/` | target CRT public header和source |
| `test/` | lit/FileCheck、CLI、Python、Runtime和Board contracts |
| `unittests/` | 按component拆分的C++ unit、public link smoke、target numeric backend和可选SystemC tests |
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
python3 utils/deps/bootstrap_deps.py --python
python3 utils/deps/bootstrap_deps.py --llvm-source
python3 utils/deps/bootstrap_deps.py --numeric-model-deps
python3 utils/deps/bootstrap_deps.py --onednn-deps
python3 utils/deps/bootstrap_deps.py --systemc-model-deps
```

这些命令按需执行；core compiler、importer、target numeric backend 和 SystemC 的完整准备方式不同。开始构建前请阅读
[`third_party/README.md`](third_party/README.md) 和 [`memory/general_dev.md`](memory/general_dev.md)，不要用未固定的系统依赖
冒充正式验证环境。

## 构建与测试

先准备仓库pinned dependencies，再使用checked-in default preset；主工程binary dir就是`build/`：

```bash
cmake --preset default
cmake --build --preset default -j"$(nproc)"
cmake --build --preset default --target check-wafer -j"$(nproc)"
ctest --preset default -j"$(nproc)"
```

常用验证入口：

- `check-wafer-lit`：Dialect、Frontend、Pipelines、Spmd、Transforms和Tools中的直接tests；
- `check-wafer-unit`：直接C++ unit tests；
- `check-wafer`：执行当前配置中实际存在的全部 mandatory 子 gate；
- `ctest --preset default`：不按label过滤，运行本配置全部已注册的本地tests。

canonical build要求importer、Shardy/XLA helper、target numeric backend和SystemC的受管依赖全部闭合。测试报告中的
`unsupported`、skip或未注册case都不计通过；`ctest passed`不能替代对关键source/program gate是否实际执行的检查。

## 使用默认编译入口

`wafer-compile` 是唯一 production decision owner。当前单卡输入显式指定一个 logical card partition：

```bash
build/bin/wafer-compile \
  --input-program-dir <stablehlo-program-dir> \
  --output-dir <output-dir> \
  --num-partitions 1 \
  --optimization-policy search
```

`search`和`none`是两个独立controller；恢复后它们分别拥有自己的current TileModule/TileRegion/Instr IR，
policy-complete后才共同消费actual memory/target leaf，不能互相调用或fallback。当前二者均在
DeviceExecutable边界返回`operation_not_supported`，所以以上命令不会发布package。

`--compile-timing`、`--dump-compiler-ir <dir>`和`--profile`是产品CLI的显式诊断选项；使用它们不改变成功定义。
流水线恢复后，编译成功当且仅当package完成原子提交和strict readback：退出0时本次package可见，非0时不可见。
普通模式的`<output-dir>`是package root；profile模式使用共同delivery root发布`package`及其profile sibling。
已经存在的current package可做无板卡validation：

```bash
build/bin/wafer-run \
  --package-dir <package-root> \
  --no-card
```

`wafer-opt` 只用于局部 MLIR 调试和底层 rewrite/conversion，不拥有另一套 candidate selector，也不能把单 pass 输出直接当成
production package。

安装后的 production compiler 通过单一 resolver 发现外部工具事实：SPMD helper 与 device linker
script/CRT/ABI 资源位于可执行文件旁的 install 目录，`python3`/`clang++` 从 PATH 解析；pinned TX8
依赖根由环境变量 `TX8_DEPS_ROOT` 显式配置。二进制不烘焙 source/build tree 绝对路径。


## 当前边界

- 目标production domain是单卡、一个logical card partition和card-local 16-Tile MPMD；cross-card、dynamic
  shape/state、MoE 和持久化权重缓存尚未进入主线。card 可以为不同 op/分支生成不同 Tile module，
  这不等于 GSPMD partition。
- SystemC 是 untimed functional-event model，不证明 vendor packet、RISC-V ELF exact execution、板端性能或 cycle accuracy。
- 历史板端证据只证明当时profile、shape、dtype、payload、ABI和runtime identity下的能力；current package和
  性能结论必须由恢复后的产品pipeline fresh产生，不回放旧输出代签。
- Search选择不替代current IR上的legality、actual SPM规划或completion验证。

## 文档与协作

- [`AGENTS.md`](AGENTS.md)：仓库稳定协作、IR、pipeline、验证和提交规则；
- [`tasks/progress.md`](tasks/progress.md)：当前任务状态和前置关系的唯一入口；
- [`tasks/README.md`](tasks/README.md)：编号设计文档、pipeline owner 和历史归档导航；
- [`tasks/01-architecture.md`](tasks/01-architecture.md)：主架构与 IR/target-module/package DAG；
- [`tasks/16-verification-contract.md`](tasks/16-verification-contract.md)：分层 verification contract；
- [`docs/tx81-compiler-hardware-calibration.md`](docs/tx81-compiler-hardware-calibration.md)：当前 profile 可消费的硬件行为与外推边界；
- [`memory/general_dev.md`](memory/general_dev.md)：本地构建、依赖、调试和验证经验。

提交变更前必须保留无关工作区修改，运行与变更范围匹配的验证，并同步受影响的设计文档、任务队列和稳定经验。
