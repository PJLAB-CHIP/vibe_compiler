//===- NormalizeConstants.cpp - Wafer constant normalization --------------===//

#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/SmallVector.h"

#ifdef WAFER_ENABLE_STABLEHLO
#include "stablehlo/dialect/StablehloOps.h"
#endif

namespace wafer {
namespace {

struct NormalizeConstantsPass
    : public mlir::PassWrapper<NormalizeConstantsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NormalizeConstantsPass)

  llvm::StringRef getArgument() const final {
    return "wafer-normalize-constants";
  }

  llvm::StringRef getDescription() const final {
    return "normalize frontend constants to MLIR ConstantLike tensor values";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::arith::ArithDialect>();
#ifdef WAFER_ENABLE_STABLEHLO
    registry.insert<mlir::stablehlo::StablehloDialect>();
#endif
  }

  void runOnOperation() final {
#ifdef WAFER_ENABLE_STABLEHLO
    llvm::SmallVector<mlir::stablehlo::ConstantOp> constants;
    getOperation().walk([&](mlir::stablehlo::ConstantOp constant) {
      constants.push_back(constant);
    });

    for (mlir::stablehlo::ConstantOp constant : constants) {
      mlir::OpBuilder builder(constant);
      auto replacement = builder.create<mlir::arith::ConstantOp>(
          constant.getLoc(), constant.getType(), constant.getValue());
      constant.getResult().replaceAllUsesWith(replacement.getResult());
      constant.erase();
    }
#endif
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createNormalizeConstantsPass() {
  return std::make_unique<NormalizeConstantsPass>();
}

void registerWaferPasses() {
  static mlir::PassRegistration<NormalizeConstantsPass>
      normalizeConstantsPass;
  (void)normalizeConstantsPass;
}

} // namespace wafer
