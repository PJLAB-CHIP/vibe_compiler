//===- GatherLowering.cpp - Selected output blocks and indexed rows -----===//

#include "GatherLowering.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"

namespace wafer::compiler::detail {
namespace {

mlir::Value buildGatherRows(mlir::tensor::GatherOp gather,
                            mlir::IRRewriter &rewriter, mlir::Value output,
                            llvm::SmallVector<mlir::Value> &indices,
                            unsigned dimension) {
  unsigned batchRank = gather.getIndicesType().getRank() - 1;
  auto loc = gather.getLoc();
  auto shape = gather.getIndicesType().getShape();
  if (dimension < batchRank) {
    mlir::Value lower = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value upper =
        rewriter.create<mlir::arith::ConstantIndexOp>(loc, shape[dimension]);
    mlir::Value step = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto loop = rewriter.create<mlir::scf::ForOp>(
        loc, lower, upper, step, mlir::ValueRange{output});
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(loop.getBody());
    indices.push_back(loop.getInductionVar());
    auto updated = buildGatherRows(gather, rewriter,
                                   loop.getRegionIterArgs().front(), indices,
                                   dimension + 1);
    indices.pop_back();
    rewriter.create<mlir::scf::YieldOp>(loc, updated);
    return loop.getResult(0);
  }

  auto sourceType = gather.getSourceType();
  auto gatherDims = gather.getGatherDims();
  unsigned sourceRank = sourceType.getRank();
  bool rankReduced = gather.getResultType().getRank() ==
                     static_cast<int64_t>(batchRank + sourceRank - gatherDims.size());
  llvm::SmallVector<mlir::OpFoldResult> sourceOffsets(
      sourceRank, rewriter.getIndexAttr(0));
  llvm::SmallVector<mlir::OpFoldResult> sourceSizes;
  llvm::SmallVector<int64_t> rowShape;
  for (unsigned dim = 0; dim < sourceRank; ++dim) {
    bool gathered = llvm::is_contained(gatherDims, dim);
    int64_t size = gathered ? 1 : sourceType.getDimSize(dim);
    sourceSizes.push_back(rewriter.getIndexAttr(size));
    if (!gathered || !rankReduced)
      rowShape.push_back(size);
  }
  llvm::SmallVector<mlir::Value> coordinates(indices);
  for (auto [coordinate, sourceDim] : llvm::enumerate(gatherDims)) {
    mlir::Value component =
        rewriter.create<mlir::arith::ConstantIndexOp>(loc, coordinate);
    coordinates.push_back(component);
    mlir::Value index = rewriter.create<mlir::tensor::ExtractOp>(
        loc, gather.getIndices(), coordinates);
    coordinates.pop_back();
    if (!index.getType().isIndex())
      index = rewriter.create<mlir::arith::IndexCastOp>(
          loc, rewriter.getIndexType(), index);
    // Gather defines only in-bounds coordinates. Keep that bound explicit in
    // scalar SSA; address validation must not infer the contents of a buffer.
    mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value upper = rewriter.create<mlir::arith::ConstantIndexOp>(
        loc, sourceType.getDimSize(sourceDim) - 1);
    index = rewriter.create<mlir::arith::MaxSIOp>(loc, index, zero);
    index = rewriter.create<mlir::arith::MinSIOp>(loc, index, upper);
    sourceOffsets[sourceDim] = index;
  }
  auto rowType =
      mlir::RankedTensorType::get(rowShape, sourceType.getElementType());
  auto row = rewriter.create<mlir::tensor::ExtractSliceOp>(
      loc, rowType, gather.getSource(), sourceOffsets, sourceSizes,
      llvm::SmallVector<mlir::OpFoldResult>(sourceRank,
                                            rewriter.getIndexAttr(1)));
  llvm::SmallVector<mlir::OpFoldResult> outputOffsets(indices.begin(), indices.end());
  llvm::SmallVector<mlir::OpFoldResult> outputSizes(batchRank,
                                                  rewriter.getIndexAttr(1));
  for (int64_t size : rowShape) {
    outputOffsets.push_back(rewriter.getIndexAttr(0));
    outputSizes.push_back(rewriter.getIndexAttr(size));
  }
  return rewriter.create<mlir::tensor::InsertSliceOp>(
      loc, row, output, outputOffsets, outputSizes,
      llvm::SmallVector<mlir::OpFoldResult>(outputSizes.size(),
                                            rewriter.getIndexAttr(1)));
}
} // namespace

mlir::LogicalResult lowerTensorGathers(
    mlir::ModuleOp module, StructuredMaterializationRelations &relations) {
  llvm::SmallVector<mlir::tensor::GatherOp> gathers;
  module.walk([&](mlir::tensor::GatherOp gather) {
    if (gather->getParentOfType<TileRegionOp>())
      gathers.push_back(gather);
  });
  for (auto gather : gathers)
    if (!gather.getSourceType().hasStaticShape() ||
        !gather.getIndicesType().hasStaticShape() ||
        !gather.getResultType().hasStaticShape())
      return gather.emitError("selected gather requires static slice extents");

  StructuredBufferReplacementListener listener(relations);
  mlir::IRRewriter rewriter(module.getContext(), &listener);
  for (auto gather : gathers) {
    rewriter.setInsertionPoint(gather);
    // One destination for the selected block. Standard subset bufferization
    // keeps each row as a view of this destination, not a per-index allocation.
    mlir::Value output = rewriter.create<mlir::tensor::EmptyOp>(
        gather.getLoc(), gather.getResultType(), mlir::ValueRange{});
    llvm::SmallVector<mlir::Value> indices;
    mlir::Value result = buildGatherRows(gather, rewriter, output, indices, 0);
    rewriter.replaceOp(gather, result);
  }
  return mlir::success(listener.finalizeAfterRewrite() &&
                       mlir::succeeded(mlir::verify(module)));
}
} // namespace wafer::compiler::detail
