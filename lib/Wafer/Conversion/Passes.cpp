//===- Passes.cpp - Wafer conversion pass registration -------------------===//

#include "Wafer/Conversion/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace wafer {

void registerWaferConversionPasses() {
  static bool registered = [] {
    mlir::registerPass(createLowerTileRegionToCAbiPass);
    return true;
  }();
  (void)registered;
}

} // namespace wafer
