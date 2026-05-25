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
    mlir::registerPass(createCheckNormSchedulePass);
    mlir::registerPass(createCheckSoftmaxSchedulePass);
    mlir::registerPass(createCheckProjectionResidualSchedulePass);
    mlir::registerPass(createCheckMlpSchedulePass);
    mlir::registerPass(createMaterializeSingleTilePass);
    mlir::registerPass(createMaterializeMultiTileNoCommPass);
    mlir::registerPass(createCompactLayoutAssignmentPass);
    mlir::registerPass(createCheckSPMAllocationPass);
    mlir::registerPass(createMaterializeDDRExternalBindingsPass);
    mlir::registerPass(createLowerRingAllGatherPass);
    mlir::registerPass(createLowerToCAbiSkeletonPass);
    return true;
  }();
  (void)registered;
}

} // namespace wafer
