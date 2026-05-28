# Wafer Source Organization Recovery

日期：2026-05-26

状态：R0.2 完成记录

## 目标

R0.2 只恢复源码 ownership 边界，使当前 prototype 不再把 IR、frontend hook、stage transforms 和
C ABI conversion 混在旧的聚合目录里。

本任务不改变 IR 语义、pass 参数、lowering 行为或测试期望；它只是让后续 R1/R2/R3 能在正确的
目录和 CMake target 边界上继续补合同。

## 非目标

- R0.2 当时不把 `WaferOps.td` / `WaferDialect.cpp` 按 group、tile_region、layout、SPM、compute、
  comm、sync、launch 拆开；该后续项已由 R1.1 完成。
- 不把 interface/effect/resource 基础定义扩成 planner 可查询合同；这是 R1.2。
- 不实现真实 importer adapter、exporter bundle metadata verifier 或 Shardy bridge；这是 R2.1/R2.2。
- 不把 `wafer.abi.*` lower 到 LLVM dialect、LLVM IR、object 或真实 runtime call。
- 不实现 launch/runtime adapter、BO binding 或板端 completion。

## 当前组织

| 边界 | 当前目录 / target | 说明 |
| --- | --- | --- |
| Frontend hook | `include/Wafer/Frontend/InitImporterDialects.h` | 可选 StableHLO dialect 注册入口归到 frontend；真实 model import adapter 仍未实现 |
| Wafer IR | `include/Wafer/IR`、`lib/Wafer/IR`、`WaferIR` | 保持一个 `wafer` dialect namespace；R1.1 后 ODS/verifier/tests 已按 op family 拆分 |
| Transform pipeline | `include/Wafer/Transforms`、`lib/Wafer/Transforms`、`WaferTransforms` | 只注册 tensor/group/tile/resource/comm 等 transform pass，不再拥有 C ABI conversion pass |
| Conversion pipeline | `include/Wafer/Conversion`、`lib/Wafer/Conversion`、`WaferConversion` | 目前只承载 `WaferToCABI/LowerTileRegionToCAbi.cpp`；后续 WaferToLLVM / real C ABI lowering 在这里扩展 |
| ABI helpers | `include/Wafer/ABI`、`lib/Wafer/ABI`、`WaferABI` | 继续承载 tile-level C ABI descriptor/helper，不混入 transform pass |
| Launch/runtime | `lib/Wafer/Transforms/LaunchOutlining` | 当前只保留 DDR external binding demand fixture；真实 launch/runtime/package adapter 仍是后续任务 |

`lib/Wafer/Transforms` 进一步按 stage 分组：

- `StableHLOToLinalg/`：StableHLO textual lowering 和 constant normalization。
- `GroupFormation/`：tensor-level `wafer.group` 形成。
- `GroupScheduling/`：当前 schedule acceptance / candidate checker fixture。
- `TileRegionMaterialization/`：single-tile 和 multi-tile `wafer.tile_region` materialization。
- `SPMBufferize/`：SPM allocation trial checker。
- `LayoutMaterialization/`：layout assignment/materialization fixture。
- `Communication/`：ring collective lowering fixture。
- `LaunchOutlining/`：DDR external binding demand fixture。
- `Support/`：跨 transform stage 的局部 C++ helper，不作为 IR 协议通道。

## 合同

- `wafer-opt` 分别注册 transform pass 和 conversion pass，避免 C ABI lowering 继续由
  `WaferTransforms` ownership 隐式携带。
- pass argument 名称保持不变，已有 lit/FileCheck pipeline 不需要因目录调整改写。
- 可选 StableHLO 依赖仍由 `WAFER_ENABLE_IMPORTER_DEPS` 控制；更细的 dependency target 可见范围由
  R0.3 单独收敛。
- 本次没有新增 side table、名字匹配或跨阶段语义通道。

## 未完成项

- R0.3：补 core compiler、frontend/importer、runtime/driver、test tooling 的 dependency target
  可见范围。
- R1.1：已完成，按 op family 拆 ODS、C++ verifier 和 tests。
- R2.1/R2.2：恢复 frontend artifact / importer / Shardy bridge。
- R3.1/R3.2：恢复 group / root tile 主链路和 IR-derived package gate。

## 验证

已验证通过：

- `cmake --build build/p0 --target wafer-opt`
- `cmake --build build/p0 --target check-wafer-lit`
- `ctest --test-dir build/p0 --output-on-failure`
