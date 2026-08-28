//===- Pipelines.cpp - TileRegion to Instr pipeline --------------------===//

#include "Wafer/Conversion/TileToInstr/Pipelines.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace wafer {

void buildLowerTileRegionToInstrPipeline(mlir::OpPassManager &pm) {
  pm.nest<mlir::func::FuncOp>().nest<TileRegionOp>().addPass(
      createConvertTileRegionToInstrPass());
  pm.addPass(createConvertBufferizationCopiesToInstrPass());
  pm.nest<mlir::func::FuncOp>().addPass(createRebuildRequiredNCCJoinsPass());
}

} // namespace wafer
