# Wafer Dependency Layering Recovery

日期：2026-05-26

状态：R0.3 完成记录

## 目标

R0.3 明确当前工程的依赖层级和 CMake target 可见范围，并把第三方依赖声明集中到
`cmake/third_party/`，避免 optional frontend/importer、runtime/driver 或 test tooling 依赖穿透到
core compiler target。

本任务只收敛依赖 ownership，不改变 IR 语义、pass pipeline、runtime ABI 或 frontend artifact
合同。

## 非目标

- 不实现真实 model importer adapter、sidecar manifest 或 Shardy SPMD bridge。
- 不引入 HPGR / KMD / legacy `Tsm*` runtime adapter target。
- 不把 `wafer.abi.*` lower 到 LLVM dialect、object 或真实 runtime call。
- 不拆分 Wafer ODS / verifier op-prefix 文件；这是 R1.1。

## 依赖层级

| 层级 | 当前 target / 目录 | 允许依赖 | 禁止依赖 |
| --- | --- | --- | --- |
| Core IR | `WaferIR`、`include/Wafer/IR`、`lib/Wafer/IR` | MLIR IR、interfaces、Async dialect、TableGen 产物 | StableHLO/Shardy importer API、runtime/driver headers、GTest/lit/FileCheck |
| ABI helper | `WaferABI`、`include/Wafer/ABI`、`lib/Wafer/ABI` | C++ standard library 和项目 ABI headers | MLIR dialect API、StableHLO/Shardy、runtime/driver headers、test tooling |
| Core transforms | `WaferTransforms`、`lib/Wafer/Transforms` | `WaferIR`、MLIR arith/linalg/tensor/pass/support；`StableHLOToLinalg` 源文件可在 importer enabled 时使用 StableHLO op C++ API | importer framework headers、runtime/driver headers、test tooling、C ABI conversion ownership |
| Conversion | `WaferConversion`、`include/Wafer/Conversion`、`lib/Wafer/Conversion` | `WaferIR`、MLIR pass/IR/support；后续 WaferToLLVM 可在本层引入 LLVM dialect | StableHLO/Shardy importer API、test tooling；runtime/driver headers 只能在 future launch/runtime adapter 层进入 |
| Frontend/importer | `include/Wafer/Frontend`、`tools/wafer-import-model` | optional StableHLO dialect registration、artifact parsing/verification 依赖；framework importer roots (`torch-mlir`、PyTorch/XLA) 和 pinned torch exporter wheels | SPM/layout/runtime/driver target details |
| SPMD bridge | future Shardy/SPMD pass target | Shardy/SDY source dependency and MLIR dialect registration | physical tile id、DTE algorithm、runtime package |
| Driver tool | `wafer-opt` | `WaferIR`、`WaferTransforms`、`WaferConversion`、MLIR tool main；optional StableHLO registration | importer framework implementation details、runtime/driver headers |
| Runtime/driver | future `WaferRuntimeAdapter` / launch package layer | HPGR/KMD/legacy runtime headers and libraries, isolated behind adapter | Frontend tensor/group planning dependencies |
| Test tooling | `check-wafer-lit`、`WaferUnitTests`、tool tests | lit/FileCheck、GTest、Python test scripts | production library public interfaces |

## 第三方声明目录

- `cmake/third_party/WaferDependencyVersions.cmake` 是 LLVM/MLIR、StableHLO、Shardy、OpenXLA/XLA、
  PyTorch/XLA、torch-mlir、GTest、lit 和 importer Python wheel pin 的唯一事实源。
- `cmake/third_party/WaferThirdParty.cmake` 负责 MLIR/LLVM/Python discovery、pinned LLVM fallback、
  optional StableHLO/Shardy embedded source、StableHLO test tool import target、framework importer
  source roots、future runtime/driver SDK roots 和 GTest fallback。
- `third_party/` 是默认 dependency root；public source dependencies 作为一级 git submodule 维护：
  `third_party/stablehlo`、`third_party/shardy`、`third_party/xla`、
  `third_party/pytorch-xla`、`third_party/torch-mlir`、`third_party/googletest`。LLVM/MLIR prebuilt、Python tooling 和
  downloads 也放在该目录；`.deps/` 仅作为旧 build cache 的兼容输入。
- OpenXLA/XLA source pin 是为了 future GSPMD SPMD partitioner integration。当前主线仍以 Shardy 的
  MLIR sharding representation 作为 R2.2 bridge 边界；XLA/GSPMD 不能成为 core IR 或 backend
  library 的 public dependency。
- `requirements-importer.txt` 固定 frontend importer Python wheels：`torch==2.5.0`、
  `torchvision==0.20.0`、`torch_xla==2.5.0`。该版本组合来自 OpenXLA 的 PyTorch StableHLO
  export 教程，并与 `third_party/pytorch-xla` 的 `v2.5.0` source pin 对齐。
- 顶层 `CMakeLists.txt` 只 include third-party 配置，不直接拼 StableHLO/Shardy/GTest 发现逻辑。

## CMake 可见范围

- `StablehloOps` 是 `WaferTransforms` 的 private implementation dependency，只服务
  `lib/Wafer/Transforms/StableHLOToLinalg/*` 的 textual frontend lowering skeleton。
- `StablehloRegister` 只由 `wafer-opt` 和 `wafer-import-model` 私有链接，用于工具进程注册 dialect。
- `WaferConversion` 单独注册 conversion pass；`WaferTransforms` 不再拥有 C ABI conversion pass。
- `WaferIR`、`WaferConversion` 和 `WaferABI` 不链接 StableHLO/Shardy/importer/runtime/test tooling。
- GTest 优先使用 `third_party/googletest`，只在 `WaferUnitTests` 中出现；lit/FileCheck 只在 test
  CMake / lit config 中出现。

## 检查

`tools/check_deps.py` 现在默认检查：

- dependency pin 是否存在。
- optional importer registration hook 是否仍由工具入口调用。
- StableHLO、Shardy、OpenXLA/XLA、PyTorch/XLA、torch-mlir 和 googletest checkout HEAD 是否匹配 pin（仅当
  `third_party/*`、legacy `.deps/*` 或 legacy `.deps/src/*` checkout 存在）。
- 第三方依赖声明是否仍集中在 `cmake/third_party/`。
- public source dependency 是否记录为 `third_party/<name>` submodule。
- frontend importer Python wheel pins 是否存在于 `requirements-importer.txt`。
- future frontend importer 和 runtime/driver SDK roots 是否有显式 opt-in CMake 边界。
- StableHLO C++ API 只出现在 frontend hook、StableHLO lowering implementation 和 importer tool。
- runtime/driver header 词项不出现在 production compiler include/lib 源码中。
- `WaferTransforms` 不再以 `PUBLIC` 方式暴露 `StablehloOps`。
- `WaferIR` / `WaferConversion` / `WaferABI` 不含 importer、runtime 或 test tooling target 泄漏。

## 未完成项

- R1.1：按 op prefix 拆 ODS、C++ verifier 和 tests。
- R2.1/R2.2：真实 frontend importer artifact、sidecar、Shardy bridge 和 GSPMD-compatible
  partitioner integration 语义；当前只完成 public dependency pin、拉取和隔离检查。
- R3.2：从当前 `wafer.abi.*` IR 自动导出 package manifest。
- P8：runtime adapter、BO binding 和 completion source 仍未实现。

## 验证

已验证通过：

- `python3 -m py_compile tools/check_deps.py tools/bootstrap_deps.py`
- `python3 tools/check_deps.py`
- `git submodule status`
- fresh dependency build configure:
  `cmake -S . -B build/r0-deps -GNinja -DWAFER_ENABLE_IMPORTER_DEPS=ON -DWAFER_ENABLE_SPMD_PARTITIONER_DEPS=ON -DWAFER_ENABLE_FRAMEWORK_IMPORTER_DEPS=ON ...`
- `cmake --build build/r0-deps --target check-wafer-lit`
- `ctest --test-dir build/r0-deps --output-on-failure`
- `cmake --build build/p0 --target check-wafer-lit`
- `ctest --test-dir build/p0 --output-on-failure`
- `cmake --build build/p2-importer --target check-wafer-lit`
- `ctest --test-dir build/p2-importer --output-on-failure`
