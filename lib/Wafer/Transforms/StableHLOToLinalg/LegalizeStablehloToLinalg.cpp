//===- LegalizeStablehloToLinalg.cpp - Official StableHLO legalization ----===//

#include "Wafer/Transforms/Passes.h"

#include "mlir/Pass/Pass.h"

#ifdef WAFER_ENABLE_STABLEHLO
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#endif

namespace wafer {

std::unique_ptr<mlir::Pass> createLegalizeStablehloToLinalgPass() {
#ifdef WAFER_ENABLE_STABLEHLO
  return mlir::stablehlo::createStablehloLegalizeToLinalgPass();
#else
  return nullptr;
#endif
}

} // namespace wafer
