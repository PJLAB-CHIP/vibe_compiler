//===- Pipelines.cpp - TileRegion to Instr pipeline --------------------===//

#include "Wafer/Conversion/WaferTileRegionToInstr/Pipelines.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace wafer {

void buildLowerTileRegionToInstrPipeline(mlir::OpPassManager &pm) {
  mlir::OpPassManager &functionPM = pm.nest<mlir::func::FuncOp>();
  functionPM.nest<TileRegionOp>().addPass(createConvertTileRegionToInstrPass());
  functionPM.addPass(createPlaceRequiredNCCJoinsPass());
}

} // namespace wafer
