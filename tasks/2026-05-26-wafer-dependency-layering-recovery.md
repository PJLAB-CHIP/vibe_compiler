# Wafer Dependency Layering Recovery

日期：2026-05-26

状态：R0.3 完成记录；2026-05-26 修正为统一 OpenXLA/XLA dependency stack 和 frontend source alignment policy

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
| Frontend/importer | `include/Wafer/Frontend`、`tools/wafer-import-model` | optional StableHLO dialect registration、artifact parsing/verification 依赖；pinned torch exporter wheels / Python tooling | SPM/layout/runtime/driver target details |
| SPMD bridge | future Shardy/SPMD pass target | Shardy/SDY source dependency and MLIR dialect registration | physical tile id、DTE algorithm、runtime package |
| Driver tool | `wafer-opt` | `WaferIR`、`WaferTransforms`、`WaferConversion`、MLIR tool main；optional StableHLO registration | importer framework implementation details、runtime/driver headers |
| Runtime/driver | future `WaferRuntimeAdapter` / launch package layer | HPGR/KMD/legacy runtime headers and libraries, isolated behind adapter | Frontend tensor/group planning dependencies |
| Test tooling | `check-wafer-lit`、`WaferUnitTests`、tool tests | lit/FileCheck、GTest、Python test scripts | production library public interfaces |

## 第三方声明目录

- `cmake/third_party/WaferDependencyVersions.cmake` 是 LLVM/MLIR、StableHLO、Shardy、OpenXLA/XLA、
  GTest、lit 和 importer Python wheel pin 的唯一事实源。
- `cmake/third_party/WaferThirdParty.cmake` 负责 MLIR/LLVM/Python discovery、pinned LLVM fallback、
  optional StableHLO/Shardy embedded source、StableHLO test tool import target、framework importer
  Python tooling boundary、future runtime/driver SDK roots 和 GTest fallback。
- `third_party/` 是默认 dependency root；core compiler public source dependencies 作为一级 git submodule 维护：
  `third_party/llvm-project`、`third_party/stablehlo`、`third_party/shardy`、`third_party/xla`、
  `third_party/googletest`。Python tooling 和 downloads 也放在该目录；`.deps/` 仅作为旧 build
  cache 的兼容输入。
- OpenXLA/XLA source pin 是 C++/MLIR dependency stack 的事实源；`third_party/llvm-project` 和
  `third_party/stablehlo` 必须对齐到 `third_party/xla` workspace 中声明的 LLVM / StableHLO commit。
  `third_party/shardy` 必须使用同一 LLVM / StableHLO stack，并包含 XLA workspace 声明的 Shardy base
  commit；当前使用 `f688d8a6...`，它是 XLA base `4c2a7a07...` 的后代并修正了 standalone
  workspace 依赖。不能在同一 Wafer source tree 中同时维护另一套 LLVM/XLA source stack。
- XLA 和 Shardy workspace 都会在其 Bazel external 中对 StableHLO `e6f81...` 应用同一份
  `third_party/stablehlo/temporary.patch`。Wafer 不把 patched copy 作为第二个源码 submodule；
  patch 作为对应上游 workspace 的输入存在，并由依赖检查确认 XLA/Shardy patch 内容一致。
- OpenXLA/XLA source pin 是为了 future GSPMD SPMD partitioner integration。当前主线仍以 Shardy 的
  MLIR sharding representation 作为 R2.2 bridge 边界；XLA/GSPMD 不能成为 core IR 或 backend
  library 的 public dependency。
- `requirements-importer.txt` 固定 frontend importer Python wheels：`torch==2.5.0`、
  `torchvision==0.20.0`、`torch_xla==2.5.0`。该层只用于 importer tooling / artifact 生成测试；
  PyTorch/XLA 和 torch-mlir 源码树可以作为 optional frontend/importer tooling checkout 放在
  `third_party/<name>`，但不能穿透到 core compiler public dependency。若引入 PyTorch/XLA
  source checkout，必须先保证它的 `WORKSPACE` `xla_hash` 与 `WAFER_OPENXLA_XLA_COMMIT`
  相同；否则只能通过受检查的 wrapper 用
  `--override_repository=xla=$PWD/third_party/xla` 构建，并把该 override gate 作为 importer
  tooling 验证。不能把未对齐的 PyTorch/XLA workspace 作为第二套 XLA/LLVM/StableHLO 事实源。
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
- LLVM/MLIR、StableHLO、Shardy、OpenXLA/XLA 和 googletest checkout HEAD 是否匹配 pin（只认
  `third_party/*` submodule；legacy `.deps/*` / `.deps/src/*` 只是本地 build cache，不作为版本事实源）。
- LLVM/MLIR、StableHLO 和 Shardy pin 是否与 `third_party/xla` workspace 声明的 dependency stack
  一致；Shardy 允许使用包含 XLA base pin 且下层 LLVM/StableHLO 完全一致的 standalone 修正版本。
- XLA 和 Shardy 的 StableHLO `temporary.patch` 是否一致，避免同一 lower stack 上出现两套
  patched StableHLO 语义。
- 第三方依赖声明是否仍集中在 `cmake/third_party/`。
- public source dependency 是否记录为 `third_party/<name>` submodule。
- 如果 `third_party/pytorch-xla` 存在，其 `WORKSPACE` `xla_hash` 是否与
  `WAFER_OPENXLA_XLA_COMMIT` 一致；不一致时不能通过默认 dependency gate。
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
- PyTorch/XLA、torch-mlir source-tree adapter 的精确 commit 仍属于 R2.1 frontend importer
  恢复任务；R0.3 只建立依赖 ownership 和“不得维护第二套 XLA stack”的检查。当前已确认旧
  PyTorch/XLA v2.5.0 source pin 的 `xla_hash=32ebd694...`，PyTorch/XLA upstream master
  的 `xla_hash=9a9aa0e...`，都不等于当前 Wafer `WAFER_OPENXLA_XLA_COMMIT=0ef91e244...`。
- R3.2：从当前 `wafer.abi.*` IR 自动导出 package manifest。
- P8：runtime adapter、BO binding 和 completion source 仍未实现。

## 验证

已验证通过：

- `python3 -m py_compile tools/check_deps.py tools/bootstrap_deps.py`
- `python3 tools/check_deps.py`
- `cmake --build build/third_party/llvm-project-2377 --target install -- -j128`
  - install prefix:
    `build/third_party/llvm-install/2377f82514e12895ea56dcb3c53e305e029a72e8`
  - `mlir-opt --version`: `LLVM version 23.0.0git`
- `cmake --build build/third_party/stablehlo-unified -- -j128`
- `cmake --build build/third_party/stablehlo-xla-patched-test-build -- -j128`
- `cmake --build build/r0-deps-unified --target check-wafer-lit -- -j128`
  - `119` lit tests discovered；`118` passed；`1` unsupported。
- `cmake --build build/r0-deps-unified --target WaferUnitTests -- -j128`
- `ctest --test-dir build/r0-deps-unified --output-on-failure`
  - `wafer-lit` passed。
  - `WaferUnitTests` passed。
- `cd third_party/shardy && /root/.cache/wafer-tools/bin/bazelisk build -c opt --jobs=128 --lockfile_mode=error shardy/...`
  - `503` targets analyzed；build completed successfully。
- `cd third_party/xla && ./configure.py --backend=CPU`
- `cd third_party/xla && /root/.cache/wafer-tools/bin/bazelisk build -c opt --spawn_strategy=sandboxed --test_output=all --lockfile_mode=error //xla/...`
  - `7804` targets analyzed；build completed successfully。
- `git diff --check`

额外事实：

- `cmake --build build/third_party/stablehlo-unified --target check-stablehlo-quick -- -j128`
  编译完成，但 upstream StableHLO quick lit 有 `1` 个 `chlo_legalize_to_stablehlo.mlir` FileCheck
  mismatch；因此不能宣称 StableHLO upstream quick gate 通过。
- 应用 XLA/Shardy StableHLO `temporary.patch` 到 build 目录中的临时 copy 后，StableHLO 纯 build 通过；
  `check-stablehlo-quick` 仍有 upstream test expectation / patch 相关失败。该 patched copy 未作为
  Wafer submodule 或长期源码维护。
