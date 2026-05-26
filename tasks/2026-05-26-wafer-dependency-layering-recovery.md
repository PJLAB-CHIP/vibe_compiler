# Wafer Dependency Layering Recovery

日期：2026-05-26

状态：R0.3 完成记录

## 目标

R0.3 明确当前工程的依赖层级和 CMake target 可见范围，避免 optional frontend/importer、
runtime/driver 或 test tooling 依赖穿透到 core compiler target。

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
| Frontend/importer | `include/Wafer/Frontend`、`tools/wafer-import-model` | optional StableHLO dialect registration、artifact parsing/verification 依赖 | SPM/layout/runtime/driver target details |
| Driver tool | `wafer-opt` | `WaferIR`、`WaferTransforms`、`WaferConversion`、MLIR tool main；optional StableHLO registration | importer framework implementation details、runtime/driver headers |
| Runtime/driver | future `WaferRuntimeAdapter` / launch package layer | HPGR/KMD/legacy runtime headers and libraries, isolated behind adapter | Frontend tensor/group planning dependencies |
| Test tooling | `check-wafer-lit`、`WaferUnitTests`、tool tests | lit/FileCheck、GTest、Python test scripts | production library public interfaces |

## CMake 可见范围

- `StablehloOps` 是 `WaferTransforms` 的 private implementation dependency，只服务
  `lib/Wafer/Transforms/StableHLOToLinalg/*` 的 textual frontend lowering skeleton。
- `StablehloRegister` 只由 `wafer-opt` 和 `wafer-import-model` 私有链接，用于工具进程注册 dialect。
- `WaferConversion` 单独注册 conversion pass；`WaferTransforms` 不再拥有 C ABI conversion pass。
- `WaferIR`、`WaferConversion` 和 `WaferABI` 不链接 StableHLO/Shardy/importer/runtime/test tooling。
- GTest 只在 `WaferUnitTests` 中出现；lit/FileCheck 只在 test CMake / lit config 中出现。

## 检查

`tools/check_deps.py` 现在默认检查：

- dependency pin 是否存在。
- optional importer registration hook 是否仍由工具入口调用。
- StableHLO/Shardy checkout HEAD 是否匹配 pin（仅当 `.deps/src/*` checkout 存在）。
- StableHLO C++ API 只出现在 frontend hook、StableHLO lowering implementation 和 importer tool。
- runtime/driver header 词项不出现在 production compiler include/lib 源码中。
- `WaferTransforms` 不再以 `PUBLIC` 方式暴露 `StablehloOps`。
- `WaferIR` / `WaferConversion` / `WaferABI` 不含 importer、runtime 或 test tooling target 泄漏。

## 未完成项

- R1.1：按 op prefix 拆 ODS、C++ verifier 和 tests。
- R2.1/R2.2：真实 frontend importer artifact、sidecar 和 Shardy bridge。
- R3.2：从当前 `wafer.abi.*` IR 自动导出 package manifest。
- P8：runtime adapter、BO binding 和 completion source 仍未实现。

## 验证

已验证通过：

- `python3 tools/check_deps.py`
- `cmake --build build/p0 --target check-wafer-lit`
- `ctest --test-dir build/p0 --output-on-failure`
- `cmake --build build/p2-importer --target check-wafer-lit`
- `ctest --test-dir build/p2-importer --output-on-failure`
