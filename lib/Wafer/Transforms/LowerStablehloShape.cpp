//===- LowerStablehloShape.cpp - StableHLO shape ops to structured IR -----===//

#include "Wafer/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#ifdef WAFER_ENABLE_STABLEHLO
#include "stablehlo/dialect/StablehloOps.h"
#endif

namespace wafer {
namespace {

#ifdef WAFER_ENABLE_STABLEHLO
static mlir::Value createEmptyTensor(mlir::OpBuilder &builder,
                                     mlir::Location loc,
                                     mlir::RankedTensorType type) {
  return builder
      .create<mlir::tensor::EmptyOp>(loc, type.getShape(),
                                     type.getElementType())
      .getResult();
}

static llvm::SmallVector<int64_t>
getInsertedBroadcastDimensions(mlir::stablehlo::BroadcastInDimOp broadcast) {
  llvm::SmallVector<int64_t> dimensions;
  llvm::ArrayRef<int64_t> operandDimensions = broadcast.getBroadcastDimensions();
  for (int64_t dim = 0; dim < broadcast.getType().getRank(); ++dim)
    if (!llvm::is_contained(operandDimensions, dim))
      dimensions.push_back(dim);
  return dimensions;
}

static llvm::SmallVector<mlir::ReassociationIndices>
getSingleGroupReassociation(int64_t rank) {
  mlir::ReassociationIndices group;
  for (int64_t dim = 0; dim < rank; ++dim)
    group.push_back(dim);
  llvm::SmallVector<mlir::ReassociationIndices> reassociation;
  reassociation.push_back(group);
  return reassociation;
}

static llvm::SmallVector<mlir::OpFoldResult>
getIndexAttrs(mlir::OpBuilder &builder, llvm::ArrayRef<int64_t> values) {
  llvm::SmallVector<mlir::OpFoldResult> attrs;
  for (int64_t value : values)
    attrs.push_back(builder.getIndexAttr(value));
  return attrs;
}

static bool lowerBroadcastInDim(mlir::stablehlo::BroadcastInDimOp broadcast) {
  mlir::RankedTensorType resultType = broadcast.getType();
  if (!resultType.hasStaticShape())
    return false;

  mlir::OpBuilder builder(broadcast);
  mlir::Value empty =
      createEmptyTensor(builder, broadcast.getLoc(), resultType);
  auto lowered = builder.create<mlir::linalg::BroadcastOp>(
      broadcast.getLoc(), broadcast.getOperand(), empty,
      getInsertedBroadcastDimensions(broadcast));

  broadcast.getResult().replaceAllUsesWith(
      lowered.getOperation()->getResult(0));
  broadcast.erase();
  return true;
}

static bool lowerReshape(mlir::stablehlo::ReshapeOp reshape) {
  mlir::RankedTensorType sourceType = reshape.getOperand().getType();
  mlir::RankedTensorType resultType = reshape.getType();
  mlir::OpBuilder builder(reshape);

  if (sourceType.getRank() == 1 && resultType.getRank() > 1) {
    auto lowered = builder.create<mlir::tensor::ExpandShapeOp>(
        reshape.getLoc(), resultType, reshape.getOperand(),
        getSingleGroupReassociation(resultType.getRank()));
    reshape.getResult().replaceAllUsesWith(lowered.getResult());
    reshape.erase();
    return true;
  }

  if (sourceType.getRank() > 1 && resultType.getRank() == 1) {
    auto lowered = builder.create<mlir::tensor::CollapseShapeOp>(
        reshape.getLoc(), resultType, reshape.getOperand(),
        getSingleGroupReassociation(sourceType.getRank()));
    reshape.getResult().replaceAllUsesWith(lowered.getResult());
    reshape.erase();
    return true;
  }

  return false;
}

static bool lowerTranspose(mlir::stablehlo::TransposeOp transpose) {
  mlir::RankedTensorType resultType = transpose.getType();
  if (!resultType.hasStaticShape())
    return false;

  mlir::OpBuilder builder(transpose);
  mlir::Value empty =
      createEmptyTensor(builder, transpose.getLoc(), resultType);
  auto lowered = builder.create<mlir::linalg::TransposeOp>(
      transpose.getLoc(), transpose.getOperand(), empty,
      transpose.getPermutation());

  transpose.getResult().replaceAllUsesWith(
      lowered.getOperation()->getResult(0));
  transpose.erase();
  return true;
}

static bool lowerSlice(mlir::stablehlo::SliceOp slice) {
  mlir::RankedTensorType resultType = slice.getType();
  if (!resultType.hasStaticShape())
    return false;

  llvm::SmallVector<int64_t> sizes;
  for (int64_t dim : resultType.getShape())
    sizes.push_back(dim);

  mlir::OpBuilder builder(slice);
  auto lowered = builder.create<mlir::tensor::ExtractSliceOp>(
      slice.getLoc(), resultType, slice.getOperand(),
      getIndexAttrs(builder, slice.getStartIndices()),
      getIndexAttrs(builder, sizes), getIndexAttrs(builder, slice.getStrides()));

  slice.getResult().replaceAllUsesWith(lowered.getResult());
  slice.erase();
  return true;
}
#endif

struct LowerStablehloShapePass
    : public mlir::PassWrapper<LowerStablehloShapePass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerStablehloShapePass)

  llvm::StringRef getArgument() const final {
    return "wafer-lower-stablehlo-shape";
  }

  llvm::StringRef getDescription() const final {
    return "lower StableHLO broadcast, reshape, transpose, and slice ops to "
           "structured tensor IR";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
#ifdef WAFER_ENABLE_STABLEHLO
    registry.insert<mlir::stablehlo::StablehloDialect>();
#endif
  }

  void runOnOperation() final {
#ifdef WAFER_ENABLE_STABLEHLO
    llvm::SmallVector<mlir::stablehlo::BroadcastInDimOp> broadcasts;
    llvm::SmallVector<mlir::stablehlo::ReshapeOp> reshapes;
    llvm::SmallVector<mlir::stablehlo::TransposeOp> transposes;
    llvm::SmallVector<mlir::stablehlo::SliceOp> slices;

    getOperation().walk([&](mlir::Operation *op) {
      if (auto broadcast =
              mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(op))
        broadcasts.push_back(broadcast);
      else if (auto reshape = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(op))
        reshapes.push_back(reshape);
      else if (auto transpose =
                   mlir::dyn_cast<mlir::stablehlo::TransposeOp>(op))
        transposes.push_back(transpose);
      else if (auto slice = mlir::dyn_cast<mlir::stablehlo::SliceOp>(op))
        slices.push_back(slice);
    });

    for (mlir::stablehlo::BroadcastInDimOp broadcast : broadcasts)
      (void)lowerBroadcastInDim(broadcast);
    for (mlir::stablehlo::ReshapeOp reshape : reshapes)
      (void)lowerReshape(reshape);
    for (mlir::stablehlo::TransposeOp transpose : transposes)
      (void)lowerTranspose(transpose);
    for (mlir::stablehlo::SliceOp slice : slices)
      (void)lowerSlice(slice);
#endif
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createLowerStablehloShapePass() {
  return std::make_unique<LowerStablehloShapePass>();
}

} // namespace wafer
