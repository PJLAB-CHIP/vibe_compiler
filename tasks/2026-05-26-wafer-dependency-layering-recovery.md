# Wafer Dependency Layering Recovery

日期：2026-05-26

状态：R0.3 完成记录；2026-05-26 修正为以 PyTorch/XLA 确定 XLA 版本的依赖栈和统一 Shardy CMake 编译验证目标

## 目标

R0.3 明确当前工程的依赖层级和 CMake target 可见范围，并把第三方依赖声明集中到
`cmake/third_party/`，避免 optional frontend/importer、runtime/driver 或 test tools 依赖穿透到
core compiler target。

本任务只收敛依赖 ownership，不改变 IR 语义、pass pipeline、runtime ABI 或 frontend artifact
合同。

## 术语口径

- 固定版本：git 源码依赖固定到一个具体 commit，Python package 路线固定到一个具体版本号。
  固定版本只说明后续检查会验证“是不是这一个版本”，不说明 Wafer 已经调用了这个依赖。
- 实际使用：Wafer 的 C++ target、pass、tool 或 Python importer 直接 link/import/call 了对应依赖。
  按这个口径，R0.3 中实际进入编译验证的是 LLVM/MLIR、StableHLO 和 Shardy/SDY；PyTorch/XLA 与
  XLA/GSPMD 还没有进入 Wafer 主线代码路径。
- 编译验证目标：一个明确的 build target，用来证明某部分源码能在当前工程里编译和链接。它不是
  feature 完成证明。例如 `wafer-shardy-cmake-gate` 只证明 Shardy/SDY 公共 dialect/pass 能编译，
  不证明 R2.2 的 Shardy/SPMD artifact bridge 已完成。

## 非目标

- 不实现真实 model importer adapter、exporter bundle metadata verifier 或 Shardy SPMD bridge。
- 不引入 HPGR / KMD / legacy `Tsm*` runtime adapter target。
- 不把 `wafer.abi.*` lower 到 LLVM dialect、object 或真实 runtime call。
- 不拆分 Wafer ODS / verifier op-prefix 文件；这是 R1.1。

## 依赖层级

| 层级 | 当前 target / 目录 | 允许依赖 | 禁止依赖 |
| --- | --- | --- | --- |
| Core IR | `WaferIR`、`include/Wafer/IR`、`lib/Wafer/IR` | MLIR IR、interfaces、Async dialect、TableGen 产物 | StableHLO/Shardy importer API、runtime/driver headers、GTest/lit/FileCheck |
| ABI helper | `WaferABI`、`include/Wafer/ABI`、`lib/Wafer/ABI` | C++ standard library 和项目 ABI headers | MLIR dialect API、StableHLO/Shardy、runtime/driver headers、test tools |
| Core transforms | `WaferTransforms`、`lib/Wafer/Transforms` | `WaferIR`、MLIR arith/linalg/tensor/pass/support；`StableHLOToLinalg` 源文件可在 importer enabled 时使用 StableHLO op C++ API | importer framework headers、runtime/driver headers、test tools、C ABI conversion ownership |
| Conversion | `WaferConversion`、`include/Wafer/Conversion`、`lib/Wafer/Conversion` | `WaferIR`、MLIR pass/IR/support；后续 WaferToLLVM 可在本层引入 LLVM dialect | StableHLO/Shardy importer API、test tools；runtime/driver headers 只能在 future launch/runtime adapter 层进入 |
| Frontend/importer | `include/Wafer/Frontend`、`tools/wafer-import-model` | optional StableHLO dialect registration、artifact parsing/verification 依赖；固定版本的 torch / PyTorch/XLA importer Python runtime，后续只允许在 importer 工具中使用 | SPM/layout/runtime/driver target details |
| SPMD bridge | `ShardySdy*` CMake shim target、future Shardy/SPMD pass target | Shardy/SDY source dependency、MLIR dialect registration、import/export/propagation pass 编译验证 | physical tile id、DTE algorithm、runtime package |
| Driver tool | `wafer-opt` | `WaferIR`、`WaferTransforms`、`WaferConversion`、MLIR tool main；optional StableHLO registration | importer framework implementation details、runtime/driver headers |
| Runtime/driver | future `WaferRuntimeAdapter` / launch package layer | HPGR/KMD/legacy runtime headers and libraries, isolated behind adapter | Frontend tensor/group planning dependencies |
| Test tools | `check-wafer-lit`、`WaferUnitTests`、tool tests | lit/FileCheck、GTest、Python test scripts | production library public interfaces |

## 第三方声明目录

- `cmake/third_party/WaferDependencyVersions.cmake` 是 PyTorch/XLA、LLVM/MLIR、StableHLO、Shardy、
  OpenXLA/XLA、GTest、lit 和 importer Python package pin 的唯一事实源。
- `cmake/third_party/WaferThirdParty.cmake` 负责 MLIR/LLVM/Python discovery、固定版本 LLVM fallback、
  optional StableHLO embedded source、统一 Shardy CMake shim、StableHLO test tool import target、framework
  importer Python 环境边界、future runtime/driver SDK roots 和 GTest fallback。
- `third_party/` 是默认 dependency root；source dependencies 作为一级 git submodule 维护：
  `third_party/pytorch-xla`、`third_party/llvm-project`、`third_party/stablehlo`、
  `third_party/shardy`、`third_party/xla`、`third_party/googletest`。Python tools 和 downloads
  也放在该目录；`.deps/` 仅作为旧 build cache 的兼容输入。
- PyTorch/XLA 源码固定版本用于确定后续 frontend importer 要匹配的 XLA 版本，也是在本地构建
  `torch_xla` Python package 和 `_XLAC` extension 的源码事实源；它的 `WORKSPACE` `xla_hash`
  必须等于 `WAFER_OPENXLA_XLA_COMMIT`。当前 PyTorch/XLA `396608c7...` 选择
  OpenXLA/XLA `32ebd694...`。这不表示 Wafer core compiler target 已经调用 `torch_xla` API。
- OpenXLA/XLA 源码固定版本是 C++/MLIR dependency stack 的事实源；`third_party/llvm-project` 和
  `third_party/stablehlo` 必须对齐到 `third_party/xla` workspace 中声明的 LLVM / StableHLO commit。
  `third_party/shardy` 必须使用同一 LLVM / StableHLO stack，并包含 XLA workspace 声明的 Shardy base
  commit；当前使用 `1838e285...`，它是 XLA base `1142aa48...` 的后代并把 Shardy 自身声明的 LLVM 版本对齐到
  XLA 的 `f0b32872...`。不能在同一 Wafer source tree 中同时维护另一套 LLVM/XLA source stack。
- XLA 和 Shardy workspace 都会在其 Bazel external 中对 StableHLO `e51fd95e...` 应用同一份
  `third_party/stablehlo/temporary.patch`。Wafer 不把 patched copy 作为第二个源码 submodule；
  patch 作为对应上游 workspace 的输入存在，并由依赖检查确认 XLA/Shardy patch 内容一致。
- OpenXLA/XLA 源码固定版本是为了后续 GSPMD SPMD partitioner integration。当前主线仍以 Shardy 的
  MLIR sharding representation 作为 R2.2 bridge 边界；XLA/GSPMD 现在没有编进 Wafer target，
  也不能成为 core IR 或 backend library 的 public dependency。
- `requirements-importer.txt` 只固定 frontend importer 和 PyTorch/XLA 源码构建所需的 Python
  packages：`torch==2.5.0`、`torchvision==0.20.0`、`absl-py==2.1.0`、`pyyaml==6.0.1`、
  `requests==2.32.3`。P2.F1 的 `torch_xla` runtime 必须从 `third_party/pytorch-xla`
  源码构建/安装出同版本 package；prebuilt `torch_xla` wheel 不能作为 P2.F1 完成证明。该 runtime
  只用于 importer 工具 / artifact 生成测试，不能穿透到 core compiler public dependency。
- `tools/build_pytorch_xla_runtime.py` 是当前源码构建入口；它为 PyTorch/XLA Bazel build 提供本地
  `@xla`、`@llvm-raw` 和 `@torch` override，使源码构建复用同一套 `third_party/xla`、
  `third_party/llvm-project` 和 importer Python 环境，而不是下载或引入另一套 XLA/LLVM stack。
- 顶层 `CMakeLists.txt` 只 include third-party 配置，不直接拼 StableHLO/Shardy/GTest 发现逻辑。

## CMake 可见范围

- `StablehloOps` 是 `WaferTransforms` 的 private implementation dependency，只服务
  `lib/Wafer/Transforms/StableHLOToLinalg/*` 的 textual frontend lowering skeleton。
- `StablehloRegister` 只由 `wafer-opt` 和 `wafer-import-model` 私有链接，用于工具进程注册 dialect。
- `cmake/third_party/WaferShardyCMake.cmake` 在 `WAFER_ENABLE_SPMD_PARTITIONER_DEPS=ON` 时从
  `third_party/shardy` 生成 SDY ODS TableGen 产物，并编译 `ShardySdyDialect`、
  `ShardySdyRegister`、`ShardySdyImportPasses`、`ShardySdyExportPasses`、
  `ShardySdyPropagationPasses`、`ShardySdyTransforms` 和 `shardy-sdy-opt`。这些 target 复用同一
  top-level LLVM/MLIR 和 embedded StableHLO；不能调用 Shardy 自己的 Bazel workspace 作为 Wafer
  dependency 编译验证。
- `wafer-shardy-cmake-gate` 是 dependency compile 验证目标；开启 SPMD deps 时 `check-wafer` 依赖它。
  该目标只证明公共 Shardy/SPMD 依赖能在同一套固定版本 compiler stack 下编译和注册，不表示 R2.2
  artifact bridge 已完成。
- `WaferConversion` 单独注册 conversion pass；`WaferTransforms` 不再拥有 C ABI conversion pass。
- `WaferIR`、`WaferConversion` 和 `WaferABI` 不链接 StableHLO/Shardy/importer/runtime/test tools。
- GTest 优先使用 `third_party/googletest`，只在 `WaferUnitTests` 中出现；lit/FileCheck 只在 test
  CMake / lit config 中出现。

## 检查

`tools/check_deps.py` 现在默认检查：

- dependency 固定版本是否存在。
- optional importer registration hook 是否仍由工具入口调用。
- PyTorch/XLA、LLVM/MLIR、StableHLO、Shardy、OpenXLA/XLA 和 googletest checkout HEAD 是否匹配固定版本
  （只认 `third_party/*` submodule；legacy `.deps/*` / `.deps/src/*` 只是本地 build cache，不作为版本事实源）。
- LLVM/MLIR、StableHLO 和 Shardy 固定版本是否与 `third_party/xla` workspace 声明的 dependency stack
  一致；Shardy 允许使用包含 XLA base commit 且下层 LLVM/StableHLO 完全一致的 standalone 修正版本。
- XLA 和 Shardy 的 StableHLO `temporary.patch` 是否一致，避免同一 lower stack 上出现两套
  patched StableHLO 语义。
- 第三方依赖声明是否仍集中在 `cmake/third_party/`。
- public source dependency 是否记录为 `third_party/<name>` submodule。
- `third_party/pytorch-xla` 的 `WORKSPACE` `xla_hash` 是否与 `WAFER_OPENXLA_XLA_COMMIT` 一致。
- frontend importer Python package 固定版本是否存在于 `requirements-importer.txt`，且其中不包含
  prebuilt `torch_xla` wheel。
- future frontend importer 和 runtime/driver SDK roots 是否有显式 opt-in CMake 边界。
- StableHLO C++ API 只出现在 frontend hook、StableHLO lowering implementation 和 importer tool。
- runtime/driver header 词项不出现在 production compiler include/lib 源码中。
- `WaferTransforms` 不再以 `PUBLIC` 方式暴露 `StablehloOps`。
- `WaferIR` / `WaferConversion` / `WaferABI` 不含 importer、runtime 或 test tool target 泄漏。
- Shardy CMake shim、`wafer-shardy-cmake-gate` 和 `check-wafer` 的依赖关系存在；Shardy target 使用
  `llvm_update_compile_flags`，避免和固定版本 no-RTTI LLVM/MLIR 产生 ABI/link mismatch。

## 未完成项

- R1.1：按 op prefix 拆 ODS、C++ verifier 和 tests。
- R2.1/R2.2：真实 frontend importer artifact、exporter bundle metadata、Shardy bridge 和 GSPMD-compatible
  partitioner integration 语义；当前只完成 public dependency 固定版本、拉取、隔离检查和统一 Shardy
  CMake compile 验证目标。
- torch-mlir source-tree adapter 的精确 commit 仍属于 R2.1 frontend importer 恢复任务；R0.3
  已先用 PyTorch/XLA 2.5 源码版本收敛 XLA/LLVM/StableHLO/Shardy 版本来源。
- R3.2：从当前 `wafer.abi.*` IR 自动导出 package manifest。
- P8：runtime adapter、BO binding 和 completion source 仍未实现。

## 验证

已验证通过：

- `python3 -m py_compile tools/check_deps.py tools/bootstrap_deps.py`
- `python3 tools/check_deps.py`
- `build/third_party/llvm-install/f0b3287297aeeddcf030e3c1b08d05a69ad465aa/bin/mlir-opt --version`
  - `LLVM version 20.0.0git`
- `cmake -S . -B build/r0-deps-pytorch-xla -GNinja -DMLIR_DIR=$PWD/build/third_party/llvm-install/f0b3287297aeeddcf030e3c1b08d05a69ad465aa/lib/cmake/mlir -DLLVM_DIR=$PWD/build/third_party/llvm-install/f0b3287297aeeddcf030e3c1b08d05a69ad465aa/lib/cmake/llvm -DWAFER_ENABLE_IMPORTER_DEPS=ON -DWAFER_ENABLE_SPMD_PARTITIONER_DEPS=ON`
- `cmake --build build/r0-deps-pytorch-xla --target wafer-shardy-cmake-gate -- -j128`
  - 生成 SDY ODS include。
  - 编译并链接 `ShardySdyDialect`、`ShardySdyRegister`、`ShardySdyImportPasses`、
    `ShardySdyExportPasses`、`ShardySdyPropagationPasses`、`ShardySdyTransforms` 和 `shardy-sdy-opt`。
  - 编译期间 Shardy upstream `basic_factor_propagation.cc` 有一个 `-Wreturn-type` warning；不是本轮
    link/版本错误，后续如果启用 warnings-as-errors 再独立处理。
- `build/r0-deps-pytorch-xla/shardy-sdy-opt third_party/shardy/shardy/dialect/sdy/ir/test/mesh_parse_print.mlir`
- `build/r0-deps-pytorch-xla/shardy-sdy-opt third_party/shardy/shardy/dialect/sdy/transforms/propagation/test/basic_propagation_keep_sharding_rules.mlir -sdy-basic-propagate='keep-sharding-rules=true'`
- `cmake --build build/r0-deps-pytorch-xla --target check-wafer -- -j128`
  - `119` lit tests discovered；`118` passed；`1` unsupported。
- `ctest --test-dir build/r0-deps-pytorch-xla --output-on-failure`
  - `wafer-lit` passed。
  - `WaferUnitTests` passed。
- `git diff --check`

历史背景：

- 本轮复用已安装的固定版本 LLVM/MLIR：
  `cmake --build build/third_party/llvm-project-f0b3 --target install -- -j128`
  - install prefix:
    `build/third_party/llvm-install/f0b3287297aeeddcf030e3c1b08d05a69ad465aa`

明确不再作为当前 Wafer dependency 编译验证的项：

- 不运行 `third_party/shardy` 自己的 Bazel workspace 来证明 Wafer 的 Shardy dependency；该路径会重新
  解析它自己的 external repositories，不能代表 Wafer 统一依赖栈。
- 不运行 `third_party/xla` 自己的 Bazel workspace 作为 Wafer core compile 验证目标；XLA checkout 在
  R0.3 只作为 PyTorch/XLA 选择出来的版本来源和 future GSPMD integration source。
