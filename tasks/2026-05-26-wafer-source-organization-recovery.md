# Wafer Source Organization Recovery

日期：2026-05-26

状态：R0.2 完成记录

## 目标

R0.2 只恢复源码 ownership 边界，使当前 prototype 不再把 IR、frontend hook、stage transforms 和
C ABI conversion 混在旧的聚合目录里。

本任务不改变 IR 语义、pass 参数、lowering 行为或测试期望；它只是让后续 R1/R2/R3 能在正确的
目录和 CMake target 边界上继续补合同。

## 非目标

- R0.2 当时不把 `WaferOps.td` / `WaferDialect.cpp` 拆开；该后续项已由 R1.1 完成，并在后续组织
  清理中收口为按 IR 层组织。
- 不把 interface/effect/resource 基础定义扩成 planner 可查询合同；这是 R1.2。
- 不实现真实 importer adapter、Wafer program verifier 或 Shardy bridge；这是 R2.1/R2.2。
- 不把专门 ABI IR op family 作为主线 IR 层 lower 到 LLVM dialect、LLVM IR、object 或真实 runtime call。
- 不实现 launch/runtime adapter、BO binding 或板端 completion。

## 当前组织

| 边界 | 当前目录 / target | 说明 |
| --- | --- | --- |
| Frontend hook | `include/Wafer/Frontend/InitImporterDialects.h` | 可选 StableHLO dialect 注册入口归到 frontend；真实 model import adapter 仍未实现 |
| Wafer IR | `include/Wafer/IR`、`lib/Wafer/IR`、`WaferIR` | 保持一个 `wafer` dialect namespace；ODS、verifier 和 dialect tests 按 IR 层组织：`Tensor`、`Tile`、`Resource`、`Instr`、`Runtime`、`Common` |
| Analysis | `include/Wafer/Analysis`、`lib/Wafer/Analysis`、`WaferAnalysis` | 承载可从当前 IR 重算的 analysis，例如 group tiling demand 和 layout plan；不修改 IR、不携带 lowering ownership |
| Transform pipeline | `include/Wafer/Transforms`、`lib/Wafer/Transforms`、`WaferTransforms` | 只注册当前仍成立的 group/debug dump、SPMD propagation helper 和 named pipeline glue；不直接编译 conversion 源文件 |
| Conversion pipeline | `include/Wafer/Conversion`、`lib/Wafer/Conversion` | 按 source/target IR contract 分 target：`WaferStableHLOToLinalg`、`WaferGroupToTileRegion`；不把 conversion 放在 `Transforms` 下，也不按临时 artifact 名称组织目录 |
| ABI helpers | `include/Wafer/ABI`、`lib/Wafer/ABI`、`WaferABI` | 继续承载 tile-level C ABI descriptor/helper，不混入 transform pass |
| Launch/runtime | 后续重建 | 旧 DDR external binding demand fixture 已删除；真实 launch/runtime/package adapter 仍是后续任务 |

`lib/Wafer/Conversion` 当前按 conversion contract 分组：

- `StableHLOToLinalg/`：frontend/local compute normalization 的 StableHLO-to-Linalg conversion target。
- `WaferGroupToTileRegion/`：logical `wafer.group` 到 `wafer.tile.region` IR 的
  conversion implementation。

`lib/Wafer/Transforms` 只保留 transform pass 注册、dump/debug pass 和 pipeline glue。旧
`GroupFormation/`、`TileRegionMaterialization/`、`SPMBufferize/`、`LayoutMaterialization/`、
`Communication/`、`LaunchOutlining/` 和 `Support/ElementwiseUtils` unit/debug pass 链已删除。
后续 R3/R6/R7 恢复时按真实 program chain、IR contract 和 CMake ownership 重新建立目录。

## 合同

- `wafer-opt` 当前注册 transform pass、conversion pass、内部/局部 named MLIR pipelines，并承载用户级
  `--program-pipeline=stablehlo-spmd*` program pipeline driver；conversion target 由
  `lib/Wafer/Conversion` 单独 owning，`WaferTransforms` 只链接使用。
- 旧 unit/debug pass argument 已移除，并由 `test/Transforms/removed-legacy-debug-passes.test`
  负向覆盖。
- 可选 StableHLO 依赖仍由 `WAFER_ENABLE_IMPORTER_DEPS` 控制；更细的 dependency target 可见范围由
  R0.3 单独收敛。
- 本次没有新增 side table、名字匹配或跨阶段语义通道。

## 未完成项

- R0.3：已完成 core compiler、frontend/importer、runtime/driver、test tooling 的 dependency target
  可见范围。
- R1.1：已完成，ODS、C++ verifier 和 tests 已按 IR 层组织。
- R2.1/R2.2：恢复 frontend program / importer / Shardy bridge。
- R3.1/R3.2：恢复 group / root tile 主链路；R3.7 恢复 IR-derived package gate。

## 验证

已验证通过：

- `cmake --build build/p0 --target wafer-opt`
- `cmake --build build/p0 --target check-wafer-lit`
- `ctest --test-dir build/p0 --output-on-failure`
