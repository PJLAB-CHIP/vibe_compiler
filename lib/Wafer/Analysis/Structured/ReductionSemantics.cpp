//===- ReductionSemantics.cpp - Structured reduction legality ----------===//

#include "Wafer/Analysis/Structured/ReductionSemantics.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"

namespace wafer::analysis {
namespace {

static void setFailure(std::string *failureReason, llvm::StringRef detail) {
  if (failureReason)
    *failureReason = detail.str();
}

} // namespace

mlir::LogicalResult
verifyReductionPartitionLegality(mlir::linalg::LinalgOp reduction,
                                 bool preservesSequentialReductionOrder,
                                 std::string *failureReason) {
  if (failureReason)
    failureReason->clear();
  if (!reduction || reduction->getNumResults() != 1 ||
      reduction.getNumDpsInits() != 1) {
    setFailure(failureReason,
               "reduction partition requires one result and one accumulator");
    return mlir::failure();
  }
  auto resultType =
      mlir::dyn_cast<mlir::ShapedType>(reduction->getResult(0).getType());
  if (!resultType || !mlir::isa<mlir::IntegerType, mlir::FloatType>(
                         resultType.getElementType())) {
    setFailure(failureReason,
               "reduction partition requires shaped integer or floating-point "
               "results");
    return mlir::failure();
  }
  if (llvm::none_of(reduction.getIteratorTypesArray(),
                    [](auto iterator) {
                      return iterator == mlir::utils::IteratorType::reduction;
                    }) ||
      reduction.getRegionOutputArgs().size() != 1) {
    setFailure(failureReason,
               "reduction partition requires one reduction accumulator");
    return mlir::failure();
  }

  llvm::SmallVector<mlir::Operation *, 1> combinerOps;
  mlir::Value reducedValue = mlir::matchReduction(
      reduction.getRegionOutputArgs(), /*redPos=*/0, combinerOps);
  if (!reducedValue || combinerOps.size() != 1) {
    setFailure(failureReason,
               "reduction partition requires one exact combiner wired to the "
               "reduced value and accumulator");
    return mlir::failure();
  }

  mlir::Operation *combiner = combinerOps.front();
  if (mlir::isa<mlir::arith::AddFOp>(combiner))
    return mlir::success();
  if (auto addi = mlir::dyn_cast<mlir::arith::AddIOp>(combiner)) {
    if (preservesSequentialReductionOrder ||
        addi.getOverflowFlags() == mlir::arith::IntegerOverflowFlags::none)
      return mlir::success();
    setFailure(failureReason,
               "reduction partition cannot preserve integer overflow flags");
    return mlir::failure();
  }
  if (mlir::isa<mlir::arith::MaximumFOp, mlir::arith::MinimumFOp,
                mlir::arith::MaxSIOp, mlir::arith::MinSIOp>(combiner))
    return mlir::success();
  if (mlir::isa<mlir::arith::MaxNumFOp, mlir::arith::MinNumFOp>(combiner)) {
    setFailure(failureReason,
               "reduction partition cannot preserve maxnum/minnum NaN "
               "semantics with the current reduce kind");
    return mlir::failure();
  }
  if (mlir::isa<mlir::arith::MaxUIOp, mlir::arith::MinUIOp>(combiner)) {
    setFailure(failureReason,
               "reduction partition cannot preserve unsigned min/max "
               "semantics with the current reduce kind");
    return mlir::failure();
  }
  setFailure(failureReason,
             "reduction partition requires an exact sum, signed min/max, or "
             "IEEE minimum/maximum combiner");
  return mlir::failure();
}

} // namespace wafer::analysis
