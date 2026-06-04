//===- LegalizeStablehloToLinalg.cpp - Official StableHLO legalization ----===//

#include "Wafer/Transforms/Passes.h"

#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#ifdef WAFER_ENABLE_STABLEHLO
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#else
namespace {
struct MissingStablehloToLinalgPass
    : public mlir::PassWrapper<MissingStablehloToLinalgPass,
                               mlir::OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MissingStablehloToLinalgPass)

  llvm::StringRef getArgument() const final {
    return "wafer-require-stablehlo-to-linalg";
  }

  llvm::StringRef getDescription() const final {
    return "fail when StableHLO-to-Linalg conversion is unavailable";
  }

  void runOnOperation() final {
    getOperation()->emitError("wafer-lower-stablehlo-to-linalg requires "
                              "WAFER_ENABLE_IMPORTER_DEPS");
    signalPassFailure();
  }
};
} // namespace
#endif

namespace wafer {

std::unique_ptr<mlir::Pass> createLegalizeStablehloToLinalgPass() {
#ifdef WAFER_ENABLE_STABLEHLO
  return mlir::stablehlo::createStablehloLegalizeToLinalgPass();
#else
  return std::make_unique<MissingStablehloToLinalgPass>();
#endif
}

} // namespace wafer
