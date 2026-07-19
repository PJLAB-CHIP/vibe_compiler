//===- Passes.cpp - Wafer transform pass registration --------------------===//

#include "Wafer/Transforms/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace wafer {

#define GEN_PASS_REGISTRATION
#include "Wafer/Transforms/WaferPasses.h.inc"

void registerWaferTransformPasses() {
  static bool registered = [] {
    registerWaferTransformsPasses();
#ifdef WAFER_ENABLE_SHARDY
    mlir::registerPass([] { return createApplyDefaultSpmdShardingPass(); });
#endif
    return true;
  }();
  (void)registered;
}

} // namespace wafer
