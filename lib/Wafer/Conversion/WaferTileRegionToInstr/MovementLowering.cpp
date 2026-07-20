//===- MovementLowering.cpp - Tile-region movement lowering ------------===//

#include "Internal.h"

#include "Wafer/Analysis/PhysicalDataflow/TransferRealizability.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <optional>
#include <string>

using namespace wafer;
using namespace wafer::tile_region_to_instr;

namespace {

constexpr int64_t kNchw2NhwcPermutation[] = {0, 2, 3, 1};
constexpr int64_t kNhwc2NchwPermutation[] = {0, 3, 1, 2};

class TileLoadLowering : public mlir::OpRewritePattern<StorageLoadOp> {
public:
  TileLoadLowering(mlir::MLIRContext *context, std::string *failureReason)
      : mlir::OpRewritePattern<StorageLoadOp>(context),
        failureReason(failureReason) {}

  mlir::LogicalResult
  matchAndRewrite(StorageLoadOp op,
                  mlir::PatternRewriter &rewriter) const final {
    auto sourceType = mlir::cast<mlir::MemRefType>(op.getSource().getType());
    auto destType = mlir::cast<mlir::MemRefType>(op.getDest().getType());
    analysis::IndexRelationResult relation =
        analysis::IndexRelation::identity(destType.getShape());
    if (!relation.isExact() ||
        mlir::failed(analysis::TransferRealizability::proveCompactDma(
            sourceType, destType, *relation.get())))
      return failPattern(rewriter, op, failureReason,
                         "tile.load compact DMA is not exactly realizable");

    mlir::FailureOr<MovementDescriptor> descriptor =
        getStridedTensorDescriptor(rewriter, op, op.getSource().getType(),
                                   failureReason, "tile.load source");
    if (mlir::failed(descriptor))
      return mlir::failure();

    createRDMA(rewriter, op.getLoc(), op.getSource(), op.getDest(),
               *descriptor);
    rewriter.eraseOp(op);
    return mlir::success();
  }

private:
  std::string *failureReason;
};

class TileStoreLowering : public mlir::OpRewritePattern<StorageStoreOp> {
public:
  TileStoreLowering(mlir::MLIRContext *context, std::string *failureReason)
      : mlir::OpRewritePattern<StorageStoreOp>(context),
        failureReason(failureReason) {}

  mlir::LogicalResult
  matchAndRewrite(StorageStoreOp op,
                  mlir::PatternRewriter &rewriter) const final {
    auto sourceType = mlir::cast<mlir::MemRefType>(op.getSource().getType());
    auto destType = mlir::cast<mlir::MemRefType>(op.getDest().getType());
    analysis::IndexRelationResult relation =
        analysis::IndexRelation::identity(destType.getShape());
    if (!relation.isExact() ||
        mlir::failed(analysis::TransferRealizability::proveCompactDma(
            sourceType, destType, *relation.get())))
      return failPattern(rewriter, op, failureReason,
                         "tile.store compact DMA is not exactly realizable");

    mlir::FailureOr<MovementDescriptor> descriptor = getStridedTensorDescriptor(
        rewriter, op, op.getDest().getType(), failureReason, "tile.store dest");
    if (mlir::failed(descriptor))
      return mlir::failure();

    createWDMA(rewriter, op.getLoc(), op.getSource(), op.getDest(),
               *descriptor);
    // A tile store is the last local-engine use of the tile-local source in
    // the current schedule.  Make that completion boundary explicit so SPM
    // planning can end the source lifetime before the next traversal tile.
    // The terminal fence remains a safety net for paths without a store.
    rewriter.create<SyncLocalFenceOp>(op.getLoc());
    rewriter.eraseOp(op);
    return mlir::success();
  }

private:
  std::string *failureReason;
};

class LayoutMaterializeLowering
    : public mlir::OpRewritePattern<LayoutMaterializeOp> {
public:
  LayoutMaterializeLowering(mlir::MLIRContext *context,
                            std::string *failureReason)
      : mlir::OpRewritePattern<LayoutMaterializeOp>(context),
        failureReason(failureReason) {}

  mlir::LogicalResult
  matchAndRewrite(LayoutMaterializeOp op,
                  mlir::PatternRewriter &rewriter) const final {
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!sourceType || !resultType)
      return failPattern(rewriter, op, failureReason,
                         "layout materialize lowering requires memref types");

    analysis::IndexRelationResult relation =
        analysis::IndexRelation::identity(resultType.getShape());
    if (!relation.isExact() ||
        mlir::failed(analysis::TransferRealizability::proveGatherScatter(
            sourceType, resultType, *relation.get())))
      return failPattern(
          rewriter, op, failureReason,
          "layout materialization gather/scatter is not exactly realizable");

    mlir::FailureOr<mlir::Value> dest = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, failureReason);
    if (mlir::failed(dest))
      return mlir::failure();

    mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>> segments =
        getStaticLogicalMovementSegments(rewriter, op, sourceType, resultType,
                                         failureReason,
                                         "layout materialize lowering");
    if (mlir::failed(segments))
      return mlir::failure();

    createGatherScatterSegments(rewriter, op.getLoc(), op.getSource(), *dest,
                                *segments,
                                /*mayReorderDisjointSegments=*/true);
    rewriter.replaceOp(op, *dest);
    return mlir::success();
  }

private:
  std::string *failureReason;
};

class TileCopyLowering : public mlir::OpRewritePattern<MoveCopyOp> {
public:
  TileCopyLowering(mlir::MLIRContext *context, std::string *failureReason)
      : mlir::OpRewritePattern<MoveCopyOp>(context),
        failureReason(failureReason) {}

  mlir::LogicalResult
  matchAndRewrite(MoveCopyOp op, mlir::PatternRewriter &rewriter) const final {
    mlir::FailureOr<mlir::Value> dest = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, failureReason);
    if (mlir::failed(dest))
      return mlir::failure();

    mlir::FailureOr<MovementDescriptor> descriptor = getContiguousDescriptor(
        rewriter, op, op.getSource().getType(), failureReason);
    if (mlir::failed(descriptor))
      return mlir::failure();

    createGatherScatter(rewriter, op.getLoc(), op.getSource(), *dest,
                        *descriptor, *descriptor);
    rewriter.replaceOp(op, *dest);
    return mlir::success();
  }

private:
  std::string *failureReason;
};

class MoveExtractSliceLowering
    : public mlir::OpRewritePattern<MoveExtractSliceOp> {
public:
  MoveExtractSliceLowering(mlir::MLIRContext *context,
                           std::string *failureReason)
      : mlir::OpRewritePattern<MoveExtractSliceOp>(context),
        failureReason(failureReason) {}

  mlir::LogicalResult
  matchAndRewrite(MoveExtractSliceOp op,
                  mlir::PatternRewriter &rewriter) const final {
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!sourceType || !resultType)
      return failPattern(rewriter, op, failureReason,
                         "tile.extract_slice lowering requires memref types");

    llvm::ArrayRef<int64_t> offsets = op.getOffsets();
    llvm::ArrayRef<int64_t> sizes = op.getSizes();
    llvm::ArrayRef<int64_t> strides = op.getStrides();
    llvm::ArrayRef<int64_t> resultShape = resultType.getShape();

    auto sourceIndexFn = [&](llvm::ArrayRef<int64_t> resultIndices,
                             llvm::SmallVectorImpl<int64_t> &sourceIndices) {
      mlir::FailureOr<llvm::SmallVector<int64_t>> fullSliceIndices =
          expandRankReducedSliceIndices(rewriter, op, sizes, resultShape,
                                        resultIndices, failureReason,
                                        "tile.extract_slice lowering");
      if (mlir::failed(fullSliceIndices))
        return mlir::failure();
      sourceIndices.resize(sizes.size(), 0);
      for (size_t dim = 0; dim < sizes.size(); ++dim)
        sourceIndices[dim] =
            offsets[dim] + (*fullSliceIndices)[dim] * strides[dim];
      return mlir::success();
    };
    auto destIndexFn = [](llvm::ArrayRef<int64_t> resultIndices,
                          llvm::SmallVectorImpl<int64_t> &destIndices) {
      destIndices.assign(resultIndices.begin(), resultIndices.end());
      return mlir::success();
    };

    mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>> segments =
        getStaticMappedMovementSegments(
            rewriter, op, sourceType, resultType, resultShape, sourceIndexFn,
            destIndexFn, failureReason, "tile.extract_slice lowering");
    if (mlir::failed(segments))
      return mlir::failure();

    mlir::FailureOr<mlir::Value> dest = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, failureReason);
    if (mlir::failed(dest))
      return mlir::failure();

    createGatherScatterSegments(rewriter, op.getLoc(), op.getSource(), *dest,
                                *segments);
    rewriter.replaceOp(op, *dest);
    return mlir::success();
  }

private:
  std::string *failureReason;
};

class MoveInsertSliceLowering
    : public mlir::OpRewritePattern<MoveInsertSliceOp> {
public:
  MoveInsertSliceLowering(mlir::MLIRContext *context,
                          std::string *failureReason)
      : mlir::OpRewritePattern<MoveInsertSliceOp>(context),
        failureReason(failureReason) {}

  mlir::LogicalResult
  matchAndRewrite(MoveInsertSliceOp op,
                  mlir::PatternRewriter &rewriter) const final {
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    auto destType = mlir::dyn_cast<mlir::MemRefType>(op.getDest().getType());
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!sourceType || !destType || !resultType)
      return failPattern(rewriter, op, failureReason,
                         "tile.insert_slice lowering requires memref types");

    auto identityIndexFn = [](llvm::ArrayRef<int64_t> indices,
                              llvm::SmallVectorImpl<int64_t> &result) {
      result.assign(indices.begin(), indices.end());
      return mlir::success();
    };

    mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>> copySegments =
        getStaticMappedMovementSegments(rewriter, op, destType, resultType,
                                        resultType.getShape(), identityIndexFn,
                                        identityIndexFn, failureReason,
                                        "tile.insert_slice dest copy lowering");
    if (mlir::failed(copySegments))
      return mlir::failure();

    llvm::ArrayRef<int64_t> offsets = op.getOffsets();
    llvm::ArrayRef<int64_t> sizes = op.getSizes();
    llvm::ArrayRef<int64_t> strides = op.getStrides();
    llvm::ArrayRef<int64_t> sourceShape = sourceType.getShape();

    auto sourceIndexFn = [](llvm::ArrayRef<int64_t> sourceIndices,
                            llvm::SmallVectorImpl<int64_t> &result) {
      result.assign(sourceIndices.begin(), sourceIndices.end());
      return mlir::success();
    };
    auto destIndexFn = [&](llvm::ArrayRef<int64_t> sourceIndices,
                           llvm::SmallVectorImpl<int64_t> &destIndices) {
      mlir::FailureOr<llvm::SmallVector<int64_t>> fullSliceIndices =
          expandRankReducedSliceIndices(rewriter, op, sizes, sourceShape,
                                        sourceIndices, failureReason,
                                        "tile.insert_slice lowering");
      if (mlir::failed(fullSliceIndices))
        return mlir::failure();
      destIndices.resize(sizes.size(), 0);
      for (size_t dim = 0; dim < sizes.size(); ++dim)
        destIndices[dim] =
            offsets[dim] + (*fullSliceIndices)[dim] * strides[dim];
      return mlir::success();
    };

    mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>> insertSegments =
        getStaticMappedMovementSegments(
            rewriter, op, sourceType, resultType, sourceShape, sourceIndexFn,
            destIndexFn, failureReason, "tile.insert_slice lowering");
    if (mlir::failed(insertSegments))
      return mlir::failure();

    mlir::FailureOr<mlir::Value> result = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, failureReason);
    if (mlir::failed(result))
      return mlir::failure();

    createGatherScatterSegments(rewriter, op.getLoc(), op.getDest(), *result,
                                *copySegments);
    createGatherScatterSegments(rewriter, op.getLoc(), op.getSource(), *result,
                                *insertSegments);
    rewriter.replaceOp(op, *result);
    return mlir::success();
  }

private:
  std::string *failureReason;
};

class MoveTransposeLowering : public mlir::OpRewritePattern<MoveTransposeOp> {
public:
  MoveTransposeLowering(mlir::MLIRContext *context, std::string *failureReason)
      : mlir::OpRewritePattern<MoveTransposeOp>(context),
        failureReason(failureReason) {}

  mlir::LogicalResult
  matchAndRewrite(MoveTransposeOp op,
                  mlir::PatternRewriter &rewriter) const final {
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!sourceType || !resultType)
      return failPattern(rewriter, op, failureReason,
                         "tile.transpose lowering requires memref types");

    llvm::ArrayRef<int64_t> permutation = op.getPermutation();
    auto sourceIndexFn = [&](llvm::ArrayRef<int64_t> resultIndices,
                             llvm::SmallVectorImpl<int64_t> &sourceIndices) {
      sourceIndices.resize(sourceType.getRank(), 0);
      for (auto [resultDim, sourceDim] : llvm::enumerate(permutation))
        sourceIndices[sourceDim] = resultIndices[resultDim];
      return mlir::success();
    };
    auto destIndexFn = [](llvm::ArrayRef<int64_t> resultIndices,
                          llvm::SmallVectorImpl<int64_t> &destIndices) {
      destIndices.assign(resultIndices.begin(), resultIndices.end());
      return mlir::success();
    };

    mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>> segments =
        getStaticMappedMovementSegments(rewriter, op, sourceType, resultType,
                                        resultType.getShape(), sourceIndexFn,
                                        destIndexFn, failureReason,
                                        "tile.transpose lowering");
    if (mlir::failed(segments))
      return mlir::failure();

    mlir::FailureOr<mlir::Value> dest = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, failureReason);
    if (mlir::failed(dest))
      return mlir::failure();

    createGatherScatterSegments(rewriter, op.getLoc(), op.getSource(), *dest,
                                *segments);
    rewriter.replaceOp(op, *dest);
    return mlir::success();
  }

private:
  std::string *failureReason;
};

class InstrTDMADataMoveLowering
    : public mlir::OpRewritePattern<InstrTDMADataMoveOp> {
public:
  InstrTDMADataMoveLowering(mlir::MLIRContext *context,
                            std::string *failureReason)
      : mlir::OpRewritePattern<InstrTDMADataMoveOp>(context),
        failureReason(failureReason) {}

  mlir::LogicalResult
  matchAndRewrite(InstrTDMADataMoveOp op,
                  mlir::PatternRewriter &rewriter) const final {
    InstrDataMoveKind kind = op.getKindAttr().getValue();
    if (!requiresGatherScatterMaterialization(kind))
      return mlir::failure();

    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    auto destType = mlir::dyn_cast<mlir::MemRefType>(op.getDest().getType());
    if (!sourceType || !destType)
      return failPattern(rewriter, op, failureReason,
                         "tdma_data_move lowering requires memref operands");
    if (mlir::failed(verifyStaticShapeAttrMatchesMemRef(
            rewriter, op, sourceType, op.getSourceShapeAttr(), "source",
            failureReason, "tdma_data_move lowering")) ||
        mlir::failed(verifyStaticShapeAttrMatchesMemRef(
            rewriter, op, destType, op.getDestShapeAttr(), "dest",
            failureReason, "tdma_data_move lowering")))
      return mlir::failure();

    mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>> segments =
        lowerToSegments(op, sourceType, destType, kind, rewriter);
    if (mlir::failed(segments))
      return mlir::failure();

    createGatherScatterSegments(rewriter, op.getLoc(), op.getSource(),
                                op.getDest(), *segments);
    rewriter.eraseOp(op);
    return mlir::success();
  }

private:
  mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>>
  lowerToSegments(InstrTDMADataMoveOp op, mlir::MemRefType sourceType,
                  mlir::MemRefType destType, InstrDataMoveKind kind,
                  mlir::PatternRewriter &rewriter) const {
    switch (kind) {
    case InstrDataMoveKind::Transpose:
      if (!op.getPermutationAttr())
        return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
            rewriter, op, failureReason,
            "tdma_data_move transpose lowering requires permutation attr");
      return getPermutationDataMoveSegments(
          rewriter, op, sourceType, destType,
          op.getPermutationAttr().asArrayRef(), failureReason,
          "tdma_data_move transpose lowering");
    case InstrDataMoveKind::Nchw2Nhwc:
      return getPermutationDataMoveSegments(
          rewriter, op, sourceType, destType,
          llvm::ArrayRef<int64_t>(kNchw2NhwcPermutation), failureReason,
          "tdma_data_move nchw2nhwc lowering");
    case InstrDataMoveKind::Nhwc2Nchw:
      return getPermutationDataMoveSegments(
          rewriter, op, sourceType, destType,
          llvm::ArrayRef<int64_t>(kNhwc2NchwPermutation), failureReason,
          "tdma_data_move nhwc2nchw lowering");
    case InstrDataMoveKind::TensorNom:
      return getStaticLogicalMovementSegments(
          rewriter, op, sourceType, destType, failureReason,
          "tdma_data_move tensor_nom lowering");
    case InstrDataMoveKind::Mirror:
      if (!op.getAxesAttr())
        return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
            rewriter, op, failureReason,
            "tdma_data_move mirror lowering requires axes attr");
      return getMirrorDataMoveSegments(
          rewriter, op, sourceType, destType, op.getAxesAttr().asArrayRef(),
          failureReason, "tdma_data_move mirror lowering");
    case InstrDataMoveKind::Rotate90:
    case InstrDataMoveKind::Rotate180:
    case InstrDataMoveKind::Rotate270:
      if (!op.getAxesAttr())
        return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
            rewriter, op, failureReason,
            "tdma_data_move rotate lowering requires axes attr");
      return getRotateDataMoveSegments(rewriter, op, sourceType, destType, kind,
                                       op.getAxesAttr().asArrayRef(),
                                       failureReason,
                                       "tdma_data_move rotate lowering");
    case InstrDataMoveKind::Pad:
    case InstrDataMoveKind::Img2Col:
      return failFailureOr<llvm::SmallVector<LogicalMovementSegment>>(
          rewriter, op, failureReason,
          "tdma_data_move pad/img2col remains in the production target "
          "surface");
    }
    llvm_unreachable("unknown instr data move kind");
  }

  std::string *failureReason;
};

class MoveBroadcastLowering : public mlir::OpRewritePattern<MoveBroadcastOp> {
public:
  MoveBroadcastLowering(mlir::MLIRContext *context, std::string *failureReason)
      : mlir::OpRewritePattern<MoveBroadcastOp>(context),
        failureReason(failureReason) {}

  mlir::LogicalResult
  matchAndRewrite(MoveBroadcastOp op,
                  mlir::PatternRewriter &rewriter) const final {
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!sourceType || !resultType)
      return failPattern(rewriter, op, failureReason,
                         "tile.broadcast lowering requires memref types");

    llvm::ArrayRef<int64_t> dimensions = op.getDimensions();
    auto sourceIndexFn = [&](llvm::ArrayRef<int64_t> resultIndices,
                             llvm::SmallVectorImpl<int64_t> &sourceIndices) {
      sourceIndices.resize(sourceType.getRank(), 0);
      for (auto [sourceDim, resultDim] : llvm::enumerate(dimensions))
        sourceIndices[sourceDim] = resultIndices[resultDim];
      return mlir::success();
    };
    auto destIndexFn = [](llvm::ArrayRef<int64_t> resultIndices,
                          llvm::SmallVectorImpl<int64_t> &destIndices) {
      destIndices.assign(resultIndices.begin(), resultIndices.end());
      return mlir::success();
    };

    mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>> segments =
        getStaticMappedMovementSegments(rewriter, op, sourceType, resultType,
                                        resultType.getShape(), sourceIndexFn,
                                        destIndexFn, failureReason,
                                        "tile.broadcast lowering");
    if (mlir::failed(segments))
      return mlir::failure();

    mlir::FailureOr<mlir::Value> dest = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, failureReason);
    if (mlir::failed(dest))
      return mlir::failure();

    createGatherScatterSegments(rewriter, op.getLoc(), op.getSource(), *dest,
                                *segments);
    rewriter.replaceOp(op, *dest);
    return mlir::success();
  }

private:
  std::string *failureReason;
};
class ViewReshapeLowering : public mlir::OpRewritePattern<ViewReshapeOp> {
public:
  ViewReshapeLowering(mlir::MLIRContext *context, std::string *failureReason)
      : mlir::OpRewritePattern<ViewReshapeOp>(context),
        failureReason(failureReason) {}

  mlir::LogicalResult
  matchAndRewrite(ViewReshapeOp op,
                  mlir::PatternRewriter &rewriter) const final {
    if (op.getSource().getType() == op.getResult().getType()) {
      rewriter.replaceOp(op, op.getSource());
      return mlir::success();
    }

    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    if (!sourceType)
      return failPattern(rewriter, op, failureReason,
                         "tile.reshape lowering requires memref source type");
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!resultType)
      return failPattern(rewriter, op, failureReason,
                         "tile.reshape lowering requires memref result type");

    MemoryAttr sourceMemory = wafer::getWaferMemoryAttr(sourceType);
    MemoryAttr resultMemory = wafer::getWaferMemoryAttr(resultType);
    if (!sourceMemory || !resultMemory)
      return failPattern(rewriter, op, failureReason,
                         "tile.reshape lowering requires Wafer memref types");

    analysis::IndexRelationResult relation =
        analysis::IndexRelation::staticReshape(resultType.getShape(),
                                               sourceType.getShape());
    if (!relation.isExact())
      return failPattern(rewriter, op, failureReason,
                         "tile.reshape requires an exact index relation");
    if (mlir::succeeded(analysis::TransferRealizability::proveMetadataView(
            sourceType, resultType, *relation.get(),
            /*destinationMayWrite=*/true))) {
      llvm::SmallVector<int64_t> sizes(resultType.getShape().begin(),
                                       resultType.getShape().end());
      mlir::FailureOr<llvm::SmallVector<int64_t>> strides =
          getStaticCompactStrides(rewriter, op, resultType, failureReason);
      if (mlir::failed(strides))
        return mlir::failure();

      auto view = rewriter.create<mlir::memref::ReinterpretCastOp>(
          op.getLoc(), resultType, op.getSource(), /*offset=*/0, sizes,
          *strides);
      rewriter.replaceOp(op, view.getResult());
      return mlir::success();
    }

    if (mlir::failed(analysis::TransferRealizability::proveGatherScatter(
            sourceType, resultType, *relation.get())))
      return failPattern(rewriter, op, failureReason,
                         "tile.reshape movement is not exactly realizable");

    mlir::FailureOr<llvm::SmallVector<LogicalMovementSegment>> segments =
        getStaticLogicalMovementSegments(rewriter, op, sourceType, resultType,
                                         failureReason,
                                         "tile.reshape lowering");
    if (mlir::failed(segments))
      return mlir::failure();

    mlir::FailureOr<mlir::Value> dest = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, failureReason);
    if (mlir::failed(dest))
      return mlir::failure();

    createGatherScatterSegments(rewriter, op.getLoc(), op.getSource(), *dest,
                                *segments);
    rewriter.replaceOp(op, *dest);
    return mlir::success();
  }

private:
  std::string *failureReason;
};

} // namespace

void wafer::tile_region_to_instr::populateMovementLoweringPatterns(
    mlir::RewritePatternSet &patterns, std::string *failureReason) {
  mlir::MLIRContext *context = patterns.getContext();
  patterns.add<TileLoadLowering, TileStoreLowering, LayoutMaterializeLowering,
               TileCopyLowering, MoveExtractSliceLowering,
               MoveInsertSliceLowering, MoveTransposeLowering,
               InstrTDMADataMoveLowering, MoveBroadcastLowering>(context,
                                                                 failureReason);
}

void wafer::tile_region_to_instr::populateViewReshapeLoweringPattern(
    mlir::RewritePatternSet &patterns, std::string *failureReason) {
  patterns.add<ViewReshapeLowering>(patterns.getContext(), failureReason);
}
