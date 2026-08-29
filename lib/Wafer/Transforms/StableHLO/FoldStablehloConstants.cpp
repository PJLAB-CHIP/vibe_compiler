//===- FoldStablehloConstants.cpp - StableHLO constant normalization ----===//

#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

#ifdef WAFER_ENABLE_STABLEHLO
#include "stablehlo/dialect/StablehloOps.h"
#endif

namespace wafer {
#define GEN_PASS_DEF_FOLDDEFAULTSTABLEHLOEXECUTIONIDSPASS
#define GEN_PASS_DEF_FOLDCONSTANTINTEGERTENSORCASTSPASS
#include "Wafer/Transforms/WaferTransformPasses.h.inc"

namespace {

#ifdef WAFER_ENABLE_STABLEHLO
static bool foldDefaultExecutionId(mlir::Operation *operation) {
  if (operation->getNumResults() != 1)
    return false;
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(operation->getResult(0).getType());
  if (!resultType || resultType.getRank() != 0)
    return false;
  auto integerType =
      mlir::dyn_cast<mlir::IntegerType>(resultType.getElementType());
  if (!integerType)
    return false;

  mlir::OpBuilder builder(operation);
  auto value = mlir::DenseElementsAttr::get(
      resultType, llvm::APInt(integerType.getWidth(), 0));
  auto constant = builder.create<mlir::arith::ConstantOp>(operation->getLoc(),
                                                          resultType, value);
  operation->getResult(0).replaceAllUsesWith(constant.getResult());
  operation->erase();
  return true;
}

static bool
foldConstantIntegerTensorCast(mlir::UnrealizedConversionCastOp cast) {
  if (cast.getInputs().size() != 1 || cast->getNumResults() != 1)
    return false;

  auto constant =
      cast.getInputs().front().getDefiningOp<mlir::arith::ConstantOp>();
  if (!constant)
    return false;
  auto sourceAttr =
      mlir::dyn_cast<mlir::DenseIntElementsAttr>(constant.getValue());
  if (!sourceAttr)
    return false;

  auto sourceType = mlir::dyn_cast<mlir::RankedTensorType>(constant.getType());
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(cast.getResult(0).getType());
  if (!sourceType || !resultType ||
      sourceType.getShape() != resultType.getShape())
    return false;

  auto sourceIntegerType =
      mlir::dyn_cast<mlir::IntegerType>(sourceType.getElementType());
  auto resultIntegerType =
      mlir::dyn_cast<mlir::IntegerType>(resultType.getElementType());
  if (!sourceIntegerType || !resultIntegerType ||
      sourceIntegerType.getWidth() != resultIntegerType.getWidth())
    return false;

  llvm::SmallVector<llvm::APInt> values;
  values.reserve(sourceAttr.getNumElements());
  for (llvm::APInt value : sourceAttr.getValues<llvm::APInt>())
    values.push_back(value);

  mlir::OpBuilder builder(cast);
  auto replacement = builder.create<mlir::arith::ConstantOp>(
      cast.getLoc(), resultType,
      mlir::DenseElementsAttr::get(resultType, values));
  cast.getResult(0).replaceAllUsesWith(replacement.getResult());
  cast.erase();
  return true;
}

static mlir::LogicalResult foldDefaultExecutionIds(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::Operation *> operations;
  module.walk([&](mlir::Operation *operation) {
    if (mlir::isa<mlir::stablehlo::PartitionIdOp, mlir::stablehlo::ReplicaIdOp>(
            operation))
      operations.push_back(operation);
  });
  for (mlir::Operation *operation : operations) {
    if (!foldDefaultExecutionId(operation))
      return operation->emitError(
          "failed to fold default StableHLO execution id");
  }
  return mlir::verify(module);
}

static mlir::LogicalResult
foldConstantIntegerTensorCasts(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::UnrealizedConversionCastOp> casts;
  module.walk(
      [&](mlir::UnrealizedConversionCastOp cast) { casts.push_back(cast); });
  for (mlir::UnrealizedConversionCastOp cast : casts) {
    if (!foldConstantIntegerTensorCast(cast))
      return cast.emitError(
          "failed to fold constant integer tensor unrealized cast");
  }
  return mlir::verify(module);
}
#endif

struct FoldDefaultStablehloExecutionIdsPass
    : public impl::FoldDefaultStablehloExecutionIdsPassBase<
          FoldDefaultStablehloExecutionIdsPass> {
  using impl::FoldDefaultStablehloExecutionIdsPassBase<
      FoldDefaultStablehloExecutionIdsPass>::
      FoldDefaultStablehloExecutionIdsPassBase;

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    impl::FoldDefaultStablehloExecutionIdsPassBase<
        FoldDefaultStablehloExecutionIdsPass>::getDependentDialects(registry);
#ifdef WAFER_ENABLE_STABLEHLO
    registry.insert<mlir::stablehlo::StablehloDialect>();
#endif
  }

  void runOnOperation() final {
#ifdef WAFER_ENABLE_STABLEHLO
    if (mlir::failed(foldDefaultExecutionIds(getOperation())))
      signalPassFailure();
#endif
  }
};

struct FoldConstantIntegerTensorCastsPass
    : public impl::FoldConstantIntegerTensorCastsPassBase<
          FoldConstantIntegerTensorCastsPass> {
  using impl::FoldConstantIntegerTensorCastsPassBase<
      FoldConstantIntegerTensorCastsPass>::
      FoldConstantIntegerTensorCastsPassBase;

  void runOnOperation() final {
#ifdef WAFER_ENABLE_STABLEHLO
    if (mlir::failed(foldConstantIntegerTensorCasts(getOperation())))
      signalPassFailure();
#endif
  }
};

} // namespace
} // namespace wafer
