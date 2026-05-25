//===- Passes.cpp - Wafer transform pass registration --------------------===//

#include "Wafer/Transforms/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace wafer {

void registerWaferPasses() {
  static bool registered = [] {
    mlir::registerPass(createLowerStablehloDotPass);
    mlir::registerPass(createLowerStablehloElementwisePass);
    mlir::registerPass(createLowerStablehloReducePass);
    mlir::registerPass(createLowerStablehloShapePass);
    mlir::registerPass(createNormalizeConstantsPass);
    mlir::registerPass(createFormGroupsPass);
    mlir::registerPass(createCheckRootTileCandidatesPass);
    mlir::registerPass(createMaterializeSingleTilePass);
    mlir::registerPass(createCompactLayoutAssignmentPass);
    return true;
  }();
  (void)registered;
}

} // namespace wafer
