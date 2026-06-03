//===- Passes.cpp - Wafer transform pass registration --------------------===//

#include "Wafer/Transforms/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace wafer {

void registerWaferTransformPasses() {
  static bool registered = [] {
    mlir::registerPass(createLowerStablehloDotPass);
    mlir::registerPass(createLowerStablehloElementwisePass);
    mlir::registerPass(createFormGroupCandidatesPass);
    mlir::registerPass(createDumpGroupTilingDemandPass);
    mlir::registerPass(createLowerStablehloReducePass);
    mlir::registerPass(createLowerStablehloShapePass);
    mlir::registerPass(createNormalizeStablehloCollectivesPass);
    mlir::registerPass(createNormalizeConstantsPass);
#ifdef WAFER_ENABLE_SHARDY
    mlir::registerPass([] { return createApplyDefaultSpmdShardingPass(); });
#endif
    return true;
  }();
  (void)registered;
}

} // namespace wafer
