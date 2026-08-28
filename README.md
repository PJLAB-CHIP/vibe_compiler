# Wafer AI Compiler

Wafer AI Compiler 是面向 Wafer/TX81 单卡目标的 MLIR 编译器与 runtime。当前主线接收
PyTorch/XLA 导出的 StableHLO program directory，在 card-level GSPMD 边界之后对整张 structured DAG 联合选择
Tile spatial mapping、temporal tiling、fusion/SPM residency 和 NoC 数据流，生成一个完整的 card
verified package。仓库同时提供 no-card validation 和 repo-owned TargetCall/SystemC untimed functional model，
用于在真实板卡接入前验证 compiler、ABI、memory、transport 和数值语义。

logical card partition 和 Tile 是两个不同的 domain：`num_partitions` 只属于 GSPMD/global tensor
boundary，card-local MPMD 由builtin module中的16个top-level
`wafer.tile.module(card_id, tile_id)`显式表示。current package schema
只包含 all-and-only Tile executable 和 card resource/launch 合同；不再存在 logical rank 直接绑定
Tile、single-Tile entry ABI 或兼容 reader。

## 编译流水线

```text
PyTorch/XLA StableHLO program directory
  -> frontend verification
  -> Shardy/XLA SPMD partitioning (logical card partitions)
  -> card-local Linalg/Tensor/SCF structured DAG
  -> bounded structured-DAG spatial/temporal/dataflow search
  -> top-level wafer.tile.module(card_id, tile_id)*
  -> per-Tile module fan-out
  -> Tile IR -> Instr IR + completion + SPM/DDR + Direct-DTE gates
  -> module/executable verification and candidate selection
  -> DeviceExecutable
       ├─ same-lowering Target LLVM -> TargetCall/SystemC
       └─ device link -> typed manifest -> verified package
                            ├─ no-card RuntimeSession
                            └─ configured TX81 RuntimeProvider
```

最终 IR、target modules 和 package 只保留 accepted typed IR、binding、offset、completion 和 transport 事实。候选集、cost、ordinal、
analysis 和 rejected state 都是 query-local compiler state，不进入 IR 或 package。

## 当前能力

- **Frontend/SPMD**：验证 StableHLO program directory、typed inputs/parameters 和 card-level Shardy/XLA SPMD；单卡当前
  `num_partitions=1`。
- **Structured-DAG 选择**：通过 TilingInterface、SSA use-def、Affine/Presburger relation 和 physical topology 联合选择
  per-op Tile placement、finite temporal tile、local residency 与显式 peer movement；只对有界 shortlist 物化 actual clones。
- **物理实现**：支持 Tensor/Cx/NCx physical encoding、metadata view、compact/mapped DMA、SPM gather/scatter、
  relation-backed resident transfer 和 fixed-capacity SPM/DDR packing。
- **Topology-aware 通信**：cross-Tile edge 从 current SSA/indexing relation 推导 exact demanded domain，只传输 placement
  中缺失的部分；selected Tile modules内使用显式peer send/receive/wait。collective lowering同样由current
  topology 和实际 Tile group 验证。
- **Typed target capability**：覆盖 mapped RDMA/WDMA offset、physical-footprint fill、GEMM/batched GEMM 和 versioned
  oriented GEMM ABI。
- **原子输出**：完整 Tile set 通过 DDR、NoC、instruction、event、ABI、device-link、manifest 和 readback gate 后，
  才写入 `DeviceExecutable`、Target LLVM modules 和 verified package。
- **功能数值验证**：同一 target lowering 可由 TargetCall/SystemC model 消费，并与独立 CPU expected 比较完整输出。
- **板端 runtime**：`wafer-run`对整个 verified package 建立 typed kernel/model session，执行
  allocation/H2D/load/submit/completion/status/D2H/cleanup；不提供选单个 Tile entry 的入口。
- **硬件能力边界**：当前 TX81 profile 对 compiler-sensitive 行为使用
  `supported`/`board-observed`/`unknown`/`excluded`分级；Unknown 不会被猜成 latency、bank、route 或更宽能力。

当前完成状态和精确边界以 [`tasks/progress.md`](tasks/progress.md) 为准；Q32 physical-dataflow synthesis 的集成证据见
[`tasks/archive/physical-dataflow-synthesis-completion-audit.md`](tasks/archive/physical-dataflow-synthesis-completion-audit.md)。

## 当前演进

当前 owner 是06号physical-dataflow主线：Q49.P先从正常上游IR闭合deterministic `none`，Q51再通过Q50各轴把
spatial mapping、temporal tiling、fusion/SPM residency、NoC和compute/communication overlap纳入唯一共同搜索。
旧rank==Tile、late selector、single-entry ABI和专用workload shortcut不再是current owner。动态状态、前置和完成门禁只看 [`tasks/progress.md`](tasks/progress.md)；README
不复制实施日志或历史性能结论。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| `include/Wafer/` | Dialect、interface、analysis、compiler/runtime 公共接口 |
| `lib/Wafer/` | Frontend、SPMD、scheduling、conversion、compiler、target code generation、runtime 和 model 实现 |
| `tools/` | `wafer-compile`、`wafer-run`、`wafer-opt`、StableHLO 工具、profile report、依赖 bootstrap 和一致性检查 |
| `test/` | lit/FileCheck、CLI 和 Python tool tests |
| `unittests/` | C++ unit、target numeric backend 和可选 SystemC tests |
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
python3 tools/bootstrap_deps.py --onednn-deps
python3 tools/bootstrap_deps.py --systemc-model-deps
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

`search`和`none`是两个独立controller，分别构造自己的current TileModule/TileRegion/Instr IR，只在policy-complete后消费同一个actual
memory/target leaf，二者不能互相调用或fallback。当前这两条controller尚未重新接通，会在DeviceExecutable边界明确返回
`operation_not_supported`，不会发布package；状态与恢复顺序以`tasks/progress.md`为准。编译成功当且仅当package已原子提交并readback
验证：CLI 退出 0 时目标 package 必然可见，退出非 0 时本次目标 package 不可见。`--profile` 时输出目录是
共同 delivery root，一次 rename 发布 `<output>/package` 与 `<output>/package.profile`，runtime 的
`<package-root>.profile` sibling 规则不变。target-model 与 compiler IR dump 只属于 internal/test 入口
（`wafer-compile-test`），不进入 production compile status。普通模式的`<output-dir>`就是package root；`--profile`模式下
它是共同delivery root，package root为`<output-dir>/package`。package可先做无板卡validation：

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

- 当前 production domain 是单卡、一个 logical card partition 和 card-local 16-Tile MPMD；cross-card、dynamic
  shape/state、MoE 和持久化权重缓存尚未进入主线。card 可以为不同 op/分支生成不同 Tile module，
  这不等于 GSPMD partition。
- 数学变换只能从 current structured semantics 和显式 proof 合法产生；不授权任意 fast-math、未证明的
  FMA contraction，也不放宽 special value、index、layout、guard 或 physical-span 检查。
- SystemC 是 untimed functional-event model，不证明 vendor packet、RISC-V ELF exact execution、板端性能或 cycle accuracy。
- 历史板端证据只证明当时 profile、shape、dtype、payload、ABI 和 runtime identity 下的能力；Q49.P/Q53 的
  current package 和性能结论必须用新 pipeline fresh 产生，不回放旧输出代签。
- card 理论 cost 只使用 cohort 内全部候选共有的已知 term；不知的硬件参数不进入比较，不产生
  候选局部缺项或候选局部零值；raw collector的`unavailable`只作诊断，不进入最终数值makespan。

## 文档与协作

- [`AGENTS.md`](AGENTS.md)：仓库稳定协作、IR、pipeline、验证和提交规则；
- [`tasks/progress.md`](tasks/progress.md)：当前任务状态和前置关系的唯一入口；
- [`tasks/README.md`](tasks/README.md)：编号设计文档、pipeline owner 和历史归档导航；
- [`tasks/01-architecture.md`](tasks/01-architecture.md)：主架构与 IR/target-module/package DAG；
- [`tasks/16-verification-contract.md`](tasks/16-verification-contract.md)：分层 verification contract；
- [`docs/tx81-compiler-hardware-calibration.md`](docs/tx81-compiler-hardware-calibration.md)：当前 profile 可消费的硬件行为与外推边界；
- [`memory/general_dev.md`](memory/general_dev.md)：本地构建、依赖、调试和验证经验。

提交变更前必须保留无关工作区修改，运行与变更范围匹配的验证，并同步受影响的设计文档、任务队列和稳定经验。
