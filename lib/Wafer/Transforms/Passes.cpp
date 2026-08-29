//===- Passes.cpp - Wafer transform pass registration --------------------===//

#include "Wafer/Transforms/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace wafer {

#define GEN_PASS_REGISTRATION
#include "Wafer/Transforms/WaferTransformPasses.h.inc"

void registerWaferTransformPasses() {
  static bool registered = [] {
    registerWaferTransformsPasses();
    return true;
  }();
  (void)registered;
}

} // namespace wafer
