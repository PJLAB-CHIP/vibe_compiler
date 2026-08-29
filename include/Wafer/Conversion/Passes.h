//===- Passes.h - Wafer conversion pass registration ---------*- C++ -*-===//

#ifndef WAFER_CONVERSION_PASSES_H
#define WAFER_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace wafer {

#define GEN_PASS_DECL
#include "Wafer/Conversion/WaferConversionPasses.h.inc"

std::unique_ptr<mlir::Pass> createLegalizeStablehloToLinalgPass();

#define GEN_PASS_REGISTRATION
#include "Wafer/Conversion/WaferConversionPasses.h.inc"

inline void registerWaferConversionPasses() {
  static bool registered = [] {
    registerWaferConversionsPasses();
    return true;
  }();
  (void)registered;
}

} // namespace wafer

#endif // WAFER_CONVERSION_PASSES_H
