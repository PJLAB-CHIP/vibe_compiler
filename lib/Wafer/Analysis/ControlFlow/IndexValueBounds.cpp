//===- IndexValueBounds.cpp - Index range interface models ----------------===//

#include "Wafer/Analysis/ControlFlow/IndexValueBounds.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

namespace wafer::analysis {
namespace {

template <typename Op, bool IsMinimum>
struct ExtremumBounds
    : mlir::ValueBoundsOpInterface::ExternalModel<ExtremumBounds<Op, IsMinimum>,
                                                  Op> {
  void
  populateBoundsForIndexValue(mlir::Operation *operation, mlir::Value value,
                              mlir::ValueBoundsConstraintSet &bounds) const {
    auto op = mlir::cast<Op>(operation);
    auto lhs = bounds.getExpr(op.getLhs());
    auto rhs = bounds.getExpr(op.getRhs());
    if constexpr (IsMinimum) {
      bounds.bound(value) <= lhs;
      bounds.bound(value) <= rhs;
    } else {
      bounds.bound(value) >= lhs;
      bounds.bound(value) >= rhs;
    }
  }
};

struct SignedCeilDivBounds
    : mlir::ValueBoundsOpInterface::ExternalModel<SignedCeilDivBounds,
                                                  mlir::arith::CeilDivSIOp> {
  void
  populateBoundsForIndexValue(mlir::Operation *operation, mlir::Value value,
                              mlir::ValueBoundsConstraintSet &bounds) const {
    auto op = mlir::cast<mlir::arith::CeilDivSIOp>(operation);
    llvm::APInt divisor;
    if (!mlir::matchPattern(op.getRhs(), mlir::m_ConstantInt(&divisor)) ||
        !divisor.isStrictlyPositive() || divisor.getBitWidth() > 64)
      return;
    auto numerator = bounds.getExpr(op.getLhs());
    bounds.bound(value) == numerator.ceilDiv(divisor.getSExtValue());
  }
};

} // namespace

void registerIndexValueBoundsModels(mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *context,
                            mlir::arith::ArithDialect *) {
    mlir::arith::CeilDivSIOp::attachInterface<SignedCeilDivBounds>(*context);
    mlir::arith::MinSIOp::attachInterface<
        ExtremumBounds<mlir::arith::MinSIOp, true>>(*context);
    mlir::arith::MaxSIOp::attachInterface<
        ExtremumBounds<mlir::arith::MaxSIOp, false>>(*context);
  });
}

} // namespace wafer::analysis
