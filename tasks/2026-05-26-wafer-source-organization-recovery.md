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
- 不实现真实 importer adapter、Wafer program verifier 或 Shardy bridge；这是 R2.1/R2.2。
- 不把 `wafer.abi.*` lower 到 LLVM dialect、LLVM IR、object 或真实 runtime call。
- 不实现 launch/runtime adapter、BO binding 或板端 completion。

## 当前组织

| 边界 | 当前目录 / target | 说明 |
| --- | --- | --- |
| Frontend hook | `include/Wafer/Frontend/InitImporterDialects.h` | 可选 StableHLO dialect 注册入口归到 frontend；真实 model import adapter 仍未实现 |
| Wafer IR | `include/Wafer/IR`、`lib/Wafer/IR`、`WaferIR` | 保持一个 `wafer` dialect namespace；R1.1 后 ODS/verifier/tests 已按 op family 拆分 |
| Transform pipeline | `include/Wafer/Transforms`、`lib/Wafer/Transforms`、`WaferTransforms` | 只注册当前仍成立的 StableHLO/Linalg normalization 和 Shardy/SPMD propagation helper pass；旧 group/tile/resource/comm unit pass 已删除 |
| Conversion pipeline | 后续重建 | 旧 `include/Wafer/Conversion`、`lib/Wafer/Conversion`、`WaferConversion` 和 `WaferToCABI/LowerTileRegionToCAbi.cpp` 已删除；后续 WaferToLLVM / real C ABI lowering 按 storage-realized contract 重建 |
| ABI helpers | `include/Wafer/ABI`、`lib/Wafer/ABI`、`WaferABI` | 继续承载 tile-level C ABI descriptor/helper，不混入 transform pass |
| Launch/runtime | 后续重建 | 旧 DDR external binding demand fixture 已删除；真实 launch/runtime/package adapter 仍是后续任务 |

`lib/Wafer/Transforms` 进一步按 stage 分组：

- `StableHLOToLinalg/`：StableHLO textual lowering 和 constant normalization。
- 旧 `GroupFormation/`、`TileRegionMaterialization/`、`SPMBufferize/`、`LayoutMaterialization/`、
  `Communication/`、`LaunchOutlining/` 和 `Support/ElementwiseUtils` unit/debug pass 链已删除。
  后续 R3/R6/R7 恢复时按真实 program chain、IR contract 和 CMake ownership 重新建立目录。

## 合同

- `wafer-opt` 当前注册 transform pass、内部/局部 named MLIR pipelines，并承载用户级
  `--program-pipeline=stablehlo-spmd*` program pipeline driver；没有 conversion pass target。
- 旧 unit/debug pass argument 已移除，并由 `test/Transforms/removed-provisional-passes.test`
  负向覆盖。
- 可选 StableHLO 依赖仍由 `WAFER_ENABLE_IMPORTER_DEPS` 控制；更细的 dependency target 可见范围由
  R0.3 单独收敛。
- 本次没有新增 side table、名字匹配或跨阶段语义通道。

## 未完成项

- R0.3：补 core compiler、frontend/importer、runtime/driver、test tooling 的 dependency target
  可见范围。
- R1.1：已完成，按 op family 拆 ODS、C++ verifier 和 tests。
- R2.1/R2.2：恢复 frontend program / importer / Shardy bridge。
- R3.1/R3.2：恢复 group / root tile 主链路和 IR-derived package gate。

## 验证

已验证通过：

- `cmake --build build/p0 --target wafer-opt`
- `cmake --build build/p0 --target check-wafer-lit`
- `ctest --test-dir build/p0 --output-on-failure`
