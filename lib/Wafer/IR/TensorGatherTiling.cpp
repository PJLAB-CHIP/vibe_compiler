//===- TensorGatherTiling.cpp - Output-driven indexed tensor slices -----===//

#include "Wafer/IR/WaferInterfaces.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/ADT/STLExtras.h"

namespace wafer {
namespace {

struct GatherTiling final
    : mlir::TilingInterface::ExternalModel<GatherTiling,
                                            mlir::tensor::GatherOp> {
  llvm::SmallVector<mlir::utils::IteratorType>
  getLoopIteratorTypes(mlir::Operation *op) const {
    auto gather = mlir::cast<mlir::tensor::GatherOp>(op);
    return llvm::SmallVector<mlir::utils::IteratorType>(
        gather.getResultType().getRank(), mlir::utils::IteratorType::parallel);
  }

  llvm::SmallVector<mlir::Range>
  getIterationDomain(mlir::Operation *op, mlir::OpBuilder &builder) const {
    auto gather = mlir::cast<mlir::tensor::GatherOp>(op);
    llvm::SmallVector<mlir::Range> ranges;
    for (mlir::OpFoldResult size : mlir::tensor::getMixedSizes(
             builder, op->getLoc(), gather.getResult()))
      ranges.push_back({builder.getIndexAttr(0), size, builder.getIndexAttr(1)});
    return ranges;
  }

  mlir::FailureOr<mlir::TilingResult>
  getTiledImplementation(mlir::Operation *op, mlir::OpBuilder &builder,
                          llvm::ArrayRef<mlir::OpFoldResult> offsets,
                          llvm::ArrayRef<mlir::OpFoldResult> sizes) const {
    auto gather = mlir::cast<mlir::tensor::GatherOp>(op);
    unsigned batchRank = gather.getIndicesType().getRank() - 1;
    unsigned sourceRank = gather.getSourceType().getRank();
    auto dims = gather.getGatherDims();
    bool rankReduced = gather.getResultType().getRank() ==
                       static_cast<int64_t>(batchRank + sourceRank - dims.size());
    if (offsets.size() != sizes.size() ||
        offsets.size() != static_cast<size_t>(gather.getResultType().getRank()))
      return mlir::failure();
    // In rank-preserving form the indexed source axes are unit result axes.
    // Keep those axes whole rather than constructing a mismatched gather.
    if (!rankReduced)
      for (int64_t dim : dims)
        if (mlir::getConstantIntValue(offsets[batchRank + dim]) != 0 ||
            mlir::getConstantIntValue(sizes[batchRank + dim]) != 1)
          return mlir::failure();

    llvm::SmallVector<mlir::OpFoldResult> sourceOffsets, sourceSizes;
    auto fullSourceSizes = mlir::tensor::getMixedSizes(
        builder, op->getLoc(), gather.getSource());
    unsigned resultDimension = batchRank;
    for (unsigned dim = 0; dim < sourceRank; ++dim) {
      if (llvm::is_contained(dims, dim)) {
        sourceOffsets.push_back(builder.getIndexAttr(0));
        sourceSizes.push_back(fullSourceSizes[dim]);
        if (!rankReduced)
          ++resultDimension;
      } else {
        sourceOffsets.push_back(offsets[resultDimension]);
        sourceSizes.push_back(sizes[resultDimension++]);
      }
    }
    llvm::SmallVector<mlir::OpFoldResult> indexOffsets(
        offsets.begin(), offsets.begin() + batchRank);
    llvm::SmallVector<mlir::OpFoldResult> indexSizes(sizes.begin(),
                                                   sizes.begin() + batchRank);
    indexOffsets.push_back(builder.getIndexAttr(0));
    indexSizes.push_back(builder.getIndexAttr(dims.size()));
    auto slice = [&](mlir::Value value,
                     llvm::ArrayRef<mlir::OpFoldResult> sliceOffsets,
                     llvm::ArrayRef<mlir::OpFoldResult> sliceSizes) {
      return builder.create<mlir::tensor::ExtractSliceOp>(
          op->getLoc(), value, sliceOffsets, sliceSizes,
          llvm::SmallVector<mlir::OpFoldResult>(sliceSizes.size(),
                                                builder.getIndexAttr(1)));
    };
    auto source = slice(gather.getSource(), sourceOffsets, sourceSizes);
    auto indices = slice(gather.getIndices(), indexOffsets, indexSizes);
    auto resultType = mlir::tensor::GatherOp::inferResultType(
        source.getType(), indices.getType(), dims, rankReduced);
    auto tiled = builder.create<mlir::tensor::GatherOp>(
        op->getLoc(), resultType, source, indices, dims, gather.getUnique());
    return mlir::TilingResult{{tiled}, {tiled.getResult()}, {source, indices}};
  }

  mlir::LogicalResult getResultTilePosition(
      mlir::Operation *, mlir::OpBuilder &, unsigned resultNumber,
      llvm::ArrayRef<mlir::OpFoldResult> offsets,
      llvm::ArrayRef<mlir::OpFoldResult> sizes,
      llvm::SmallVectorImpl<mlir::OpFoldResult> &resultOffsets,
      llvm::SmallVectorImpl<mlir::OpFoldResult> &resultSizes) const {
    if (resultNumber != 0)
      return mlir::failure();
    resultOffsets.assign(offsets.begin(), offsets.end());
    resultSizes.assign(sizes.begin(), sizes.end());
    return mlir::success();
  }

  mlir::FailureOr<mlir::TilingResult> generateResultTileValue(
      mlir::Operation *op, mlir::OpBuilder &builder, unsigned resultNumber,
      llvm::ArrayRef<mlir::OpFoldResult> offsets,
      llvm::ArrayRef<mlir::OpFoldResult> sizes) const {
    if (resultNumber != 0)
      return mlir::failure();
    return getTiledImplementation(op, builder, offsets, sizes);
  }
};
} // namespace

void registerWaferTensorGatherTilingExternalModels(
    mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *context,
                            mlir::tensor::TensorDialect *) {
    mlir::tensor::GatherOp::attachInterface<GatherTiling>(*context);
  });
}
} // namespace wafer
