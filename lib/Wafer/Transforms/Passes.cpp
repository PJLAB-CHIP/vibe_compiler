//===- Passes.cpp - Wafer transform pass registration --------------------===//

#include "Wafer/Transforms/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace wafer {

void registerWaferPasses() {
  static bool registered = [] {
    mlir::registerPass(createLowerStablehloDotPass);
    mlir::registerPass(createNormalizeConstantsPass);
    return true;
  }();
  (void)registered;
}

} // namespace wafer
