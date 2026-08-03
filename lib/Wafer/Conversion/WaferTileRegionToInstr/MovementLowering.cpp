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
    ScopedLoweringPatternTiming timing(op.getOperation());
    auto sourceType = mlir::cast<mlir::MemRefType>(op.getSource().getType());
    auto destType = mlir::cast<mlir::MemRefType>(op.getDest().getType());
    analysis::IndexRelationResult relation =
        analysis::IndexRelation::identity(destType.getShape());
    if (!relation.isExact())
      return failPattern(rewriter, op, failureReason,
                         "tile.load identity relation is not exact");

    if (mlir::succeeded(analysis::TransferRealizability::proveCompactDma(
            sourceType, destType, *relation.get()))) {
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

    mlir::FailureOr<llvm::SmallVector<MovementDescriptorPair>> descriptors =
        getRelationMovementDescriptors(
            rewriter, op, sourceType, destType, destType.getShape(),
            *relation.get(), *relation.get(), MovementEngine::RDMA,
            failureReason, "tile.load");
    if (mlir::failed(descriptors))
      return mlir::failure();

    createMappedRDMADescriptors(rewriter, op.getLoc(), op.getSource(),
                                op.getDest(), *descriptors);
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
    ScopedLoweringPatternTiming timing(op.getOperation());
    auto sourceType = mlir::cast<mlir::MemRefType>(op.getSource().getType());
    auto destType = mlir::cast<mlir::MemRefType>(op.getDest().getType());
    analysis::IndexRelationResult relation =
        analysis::IndexRelation::identity(destType.getShape());
    if (!relation.isExact())
      return failPattern(rewriter, op, failureReason,
                         "tile.store identity relation is not exact");

    if (mlir::succeeded(analysis::TransferRealizability::proveCompactDma(
            sourceType, destType, *relation.get()))) {
      mlir::FailureOr<MovementDescriptor> descriptor =
          getStridedTensorDescriptor(rewriter, op, op.getDest().getType(),
                                     failureReason, "tile.store dest");
      if (mlir::failed(descriptor))
        return mlir::failure();
      createWDMA(rewriter, op.getLoc(), op.getSource(), op.getDest(),
                 *descriptor);
    } else {
      mlir::FailureOr<llvm::SmallVector<MovementDescriptorPair>> descriptors =
          getRelationMovementDescriptors(
              rewriter, op, sourceType, destType, destType.getShape(),
              *relation.get(), *relation.get(), MovementEngine::WDMA,
              failureReason, "tile.store");
      if (mlir::failed(descriptors))
        return mlir::failure();
      createMappedWDMADescriptors(rewriter, op.getLoc(), op.getSource(),
                                  op.getDest(), *descriptors);
    }

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
    ScopedLoweringPatternTiming timing(op.getOperation());
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!sourceType || !resultType)
      return failPattern(rewriter, op, failureReason,
                         "layout materialize lowering requires memref types");

    analysis::IndexRelationResult relation =
        analysis::IndexRelation::identity(resultType.getShape());
    if (!relation.isExact())
      return failPattern(
          rewriter, op, failureReason,
          "layout materialization gather/scatter is not exactly realizable");

    mlir::FailureOr<mlir::Value> dest = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, failureReason);
    if (mlir::failed(dest))
      return mlir::failure();

    mlir::FailureOr<llvm::SmallVector<MovementDescriptorPair>> descriptors =
        getRelationMovementDescriptors(
            rewriter, op, sourceType, resultType, resultType.getShape(),
            *relation.get(), *relation.get(), MovementEngine::GatherScatter,
            failureReason, "layout materialize lowering");
    if (mlir::failed(descriptors))
      return mlir::failure();

    createGatherScatterDescriptors(rewriter, op.getLoc(), op.getSource(),
                                   *dest, *descriptors);
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
    ScopedLoweringPatternTiming timing(op.getOperation());
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!sourceType || !resultType)
      return failPattern(rewriter, op, failureReason,
                         "tile.copy lowering requires memref types");
    analysis::IndexRelationResult relation =
        analysis::IndexRelation::identity(resultType.getShape());
    if (!relation.isExact())
      return failPattern(rewriter, op, failureReason,
                         "tile.copy identity relation is not exact");
    mlir::FailureOr<mlir::Value> dest = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, failureReason);
    if (mlir::failed(dest))
      return mlir::failure();

    mlir::FailureOr<llvm::SmallVector<MovementDescriptorPair>> descriptors =
        getRelationMovementDescriptors(
            rewriter, op, sourceType, resultType, resultType.getShape(),
            *relation.get(), *relation.get(), MovementEngine::GatherScatter,
            failureReason, "tile.copy lowering");
    if (mlir::failed(descriptors))
      return mlir::failure();

    createGatherScatterDescriptors(rewriter, op.getLoc(), op.getSource(),
                                   *dest, *descriptors);
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
    ScopedLoweringPatternTiming timing(op.getOperation());
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
    std::optional<llvm::SmallDenseSet<unsigned>> rankReductionMask =
        mlir::computeRankReductionMask(sizes, resultShape);
    if (!rankReductionMask)
      return failPattern(rewriter, op, failureReason,
                         "tile.extract_slice cannot map rank reduction");
    llvm::SmallVector<mlir::AffineExpr, 4> sourceResults;
    sourceResults.reserve(sourceType.getRank());
    unsigned reducedDim = 0;
    for (unsigned fullDim = 0; fullDim < sizes.size(); ++fullDim) {
      mlir::AffineExpr expression =
          mlir::getAffineConstantExpr(offsets[fullDim],
                                      rewriter.getContext());
      if (!rankReductionMask->contains(fullDim)) {
        expression =
            expression +
            mlir::getAffineDimExpr(reducedDim++, rewriter.getContext()) *
                strides[fullDim];
      }
      sourceResults.push_back(expression);
    }
    analysis::IndexRelationResult sourceRelation =
        analysis::IndexRelation::fromAffineMap(
            mlir::AffineMap::get(resultShape.size(), 0, sourceResults,
                                 rewriter.getContext()),
            resultShape, sourceType.getShape());
    analysis::IndexRelationResult destRelation =
        analysis::IndexRelation::identity(resultShape);
    if (!sourceRelation.isExact() || !destRelation.isExact())
      return failPattern(rewriter, op, failureReason,
                         "tile.extract_slice relation is not exact");

    mlir::FailureOr<llvm::SmallVector<MovementDescriptorPair>> descriptors =
        getRelationMovementDescriptors(
            rewriter, op, sourceType, resultType, resultShape,
            *sourceRelation.get(), *destRelation.get(),
            MovementEngine::GatherScatter, failureReason,
            "tile.extract_slice lowering");
    if (mlir::failed(descriptors))
      return mlir::failure();

    mlir::FailureOr<mlir::Value> dest = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, failureReason);
    if (mlir::failed(dest))
      return mlir::failure();

    createGatherScatterDescriptors(rewriter, op.getLoc(), op.getSource(),
                                   *dest, *descriptors);
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
    ScopedLoweringPatternTiming timing(op.getOperation());
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    auto destType = mlir::dyn_cast<mlir::MemRefType>(op.getDest().getType());
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!sourceType || !destType || !resultType)
      return failPattern(rewriter, op, failureReason,
                         "tile.insert_slice lowering requires memref types");

    analysis::IndexRelationResult copyRelation =
        analysis::IndexRelation::identity(resultType.getShape());
    if (!copyRelation.isExact())
      return failPattern(rewriter, op, failureReason,
                         "tile.insert_slice copy relation is not exact");
    mlir::FailureOr<llvm::SmallVector<MovementDescriptorPair>>
        copyDescriptors = getRelationMovementDescriptors(
            rewriter, op, destType, resultType, resultType.getShape(),
            *copyRelation.get(), *copyRelation.get(),
            MovementEngine::GatherScatter, failureReason,
            "tile.insert_slice dest copy lowering");
    if (mlir::failed(copyDescriptors))
      return mlir::failure();

    llvm::ArrayRef<int64_t> offsets = op.getOffsets();
    llvm::ArrayRef<int64_t> sizes = op.getSizes();
    llvm::ArrayRef<int64_t> strides = op.getStrides();
    llvm::ArrayRef<int64_t> sourceShape = sourceType.getShape();
    std::optional<llvm::SmallDenseSet<unsigned>> rankReductionMask =
        mlir::computeRankReductionMask(sizes, sourceShape);
    if (!rankReductionMask)
      return failPattern(rewriter, op, failureReason,
                         "tile.insert_slice cannot map rank reduction");
    llvm::SmallVector<mlir::AffineExpr, 4> destResults;
    destResults.reserve(resultType.getRank());
    unsigned reducedDim = 0;
    for (unsigned fullDim = 0; fullDim < sizes.size(); ++fullDim) {
      mlir::AffineExpr expression =
          mlir::getAffineConstantExpr(offsets[fullDim],
                                      rewriter.getContext());
      if (!rankReductionMask->contains(fullDim)) {
        expression =
            expression +
            mlir::getAffineDimExpr(reducedDim++, rewriter.getContext()) *
                strides[fullDim];
      }
      destResults.push_back(expression);
    }
    analysis::IndexRelationResult sourceRelation =
        analysis::IndexRelation::identity(sourceShape);
    analysis::IndexRelationResult destRelation =
        analysis::IndexRelation::fromAffineMap(
            mlir::AffineMap::get(sourceShape.size(), 0, destResults,
                                 rewriter.getContext()),
            sourceShape, resultType.getShape());
    if (!sourceRelation.isExact() || !destRelation.isExact())
      return failPattern(rewriter, op, failureReason,
                         "tile.insert_slice relation is not exact");
    mlir::FailureOr<llvm::SmallVector<MovementDescriptorPair>>
        insertDescriptors = getRelationMovementDescriptors(
            rewriter, op, sourceType, resultType, sourceShape,
            *sourceRelation.get(), *destRelation.get(),
            MovementEngine::GatherScatter, failureReason,
            "tile.insert_slice lowering");
    if (mlir::failed(insertDescriptors))
      return mlir::failure();

    mlir::FailureOr<mlir::Value> result = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, failureReason);
    if (mlir::failed(result))
      return mlir::failure();

    createGatherScatterDescriptors(rewriter, op.getLoc(), op.getDest(),
                                   *result, *copyDescriptors);
    createGatherScatterDescriptors(rewriter, op.getLoc(), op.getSource(),
                                   *result, *insertDescriptors);
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
    ScopedLoweringPatternTiming timing(op.getOperation());
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!sourceType || !resultType)
      return failPattern(rewriter, op, failureReason,
                         "tile.transpose lowering requires memref types");

    llvm::ArrayRef<int64_t> permutation = op.getPermutation();
    if (permutation.size() !=
        static_cast<size_t>(sourceType.getRank()))
      return failPattern(rewriter, op, failureReason,
                         "tile.transpose permutation rank mismatch");
    llvm::SmallVector<mlir::AffineExpr, 4> sourceResults(
        sourceType.getRank());
    llvm::SmallVector<bool, 4> seenSourceDims(sourceType.getRank(), false);
    for (auto [resultDim, sourceDim] : llvm::enumerate(permutation)) {
      if (sourceDim < 0 || sourceDim >= sourceType.getRank() ||
          seenSourceDims[sourceDim])
        return failPattern(rewriter, op, failureReason,
                           "tile.transpose permutation is invalid");
      seenSourceDims[sourceDim] = true;
      sourceResults[sourceDim] =
          mlir::getAffineDimExpr(resultDim, rewriter.getContext());
    }
    analysis::IndexRelationResult sourceRelation =
        analysis::IndexRelation::fromAffineMap(
            mlir::AffineMap::get(resultType.getRank(), 0, sourceResults,
                                 rewriter.getContext()),
            resultType.getShape(), sourceType.getShape());
    analysis::IndexRelationResult destRelation =
        analysis::IndexRelation::identity(resultType.getShape());
    if (!sourceRelation.isExact() || !destRelation.isExact())
      return failPattern(rewriter, op, failureReason,
                         "tile.transpose relation is not exact");

    mlir::FailureOr<llvm::SmallVector<MovementDescriptorPair>> descriptors =
        getRelationMovementDescriptors(
            rewriter, op, sourceType, resultType, resultType.getShape(),
            *sourceRelation.get(), *destRelation.get(),
            MovementEngine::GatherScatter, failureReason,
            "tile.transpose lowering");
    if (mlir::failed(descriptors))
      return mlir::failure();

    mlir::FailureOr<mlir::Value> dest = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, failureReason);
    if (mlir::failed(dest))
      return mlir::failure();

    createGatherScatterDescriptors(rewriter, op.getLoc(), op.getSource(),
                                   *dest, *descriptors);
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
    ScopedLoweringPatternTiming timing(op.getOperation());
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

    mlir::FailureOr<llvm::SmallVector<MovementDescriptorPair>> descriptors =
        lowerToDescriptors(op, sourceType, destType, kind, rewriter);
    if (mlir::failed(descriptors))
      return mlir::failure();

    createGatherScatterDescriptors(rewriter, op.getLoc(), op.getSource(),
                                   op.getDest(), *descriptors);
    rewriter.eraseOp(op);
    return mlir::success();
  }

private:
  mlir::FailureOr<llvm::SmallVector<MovementDescriptorPair>>
  lowerToDescriptors(InstrTDMADataMoveOp op, mlir::MemRefType sourceType,
                     mlir::MemRefType destType, InstrDataMoveKind kind,
                     mlir::PatternRewriter &rewriter) const {
    llvm::SmallVector<mlir::AffineExpr, 4> sourceResults(
        sourceType.getRank());
    auto setPermutation = [&](llvm::ArrayRef<int64_t> permutation) {
      if (sourceType.getRank() != destType.getRank() ||
          permutation.size() != static_cast<size_t>(sourceType.getRank()))
        return false;
      llvm::SmallVector<bool, 4> seen(sourceType.getRank(), false);
      for (auto [destDim, sourceDim] : llvm::enumerate(permutation)) {
        if (sourceDim < 0 || sourceDim >= sourceType.getRank() ||
            seen[sourceDim])
          return false;
        seen[sourceDim] = true;
        sourceResults[sourceDim] =
            mlir::getAffineDimExpr(destDim, rewriter.getContext());
      }
      return true;
    };

    llvm::StringRef opLabel = "tdma_data_move lowering";
    switch (kind) {
    case InstrDataMoveKind::Transpose:
      if (!op.getPermutationAttr())
        return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
            rewriter, op, failureReason,
            "tdma_data_move transpose lowering requires permutation attr");
      if (!setPermutation(op.getPermutationAttr().asArrayRef()))
        return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
            rewriter, op, failureReason,
            "tdma_data_move transpose lowering requires a valid permutation");
      opLabel = "tdma_data_move transpose lowering";
      break;
    case InstrDataMoveKind::Nchw2Nhwc:
      if (!setPermutation(kNchw2NhwcPermutation))
        return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
            rewriter, op, failureReason,
            "tdma_data_move nchw2nhwc lowering requires rank four");
      opLabel = "tdma_data_move nchw2nhwc lowering";
      break;
    case InstrDataMoveKind::Nhwc2Nchw:
      if (!setPermutation(kNhwc2NchwPermutation))
        return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
            rewriter, op, failureReason,
            "tdma_data_move nhwc2nchw lowering requires rank four");
      opLabel = "tdma_data_move nhwc2nchw lowering";
      break;
    case InstrDataMoveKind::TensorNom:
      if (!setPermutation(llvm::to_vector(
              llvm::seq<int64_t>(0, sourceType.getRank()))))
        return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
            rewriter, op, failureReason,
            "tdma_data_move tensor_nom lowering requires equal ranks");
      opLabel = "tdma_data_move tensor_nom lowering";
      break;
    case InstrDataMoveKind::Mirror:
      if (!op.getAxesAttr())
        return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
            rewriter, op, failureReason,
            "tdma_data_move mirror lowering requires axes attr");
      if (!setPermutation(llvm::to_vector(
              llvm::seq<int64_t>(0, sourceType.getRank()))))
        return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
            rewriter, op, failureReason,
            "tdma_data_move mirror lowering requires equal ranks");
      for (int64_t axis : op.getAxesAttr().asArrayRef()) {
        if (axis < 0 || axis >= sourceType.getRank())
          return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
              rewriter, op, failureReason,
              "tdma_data_move mirror lowering axis is out of range");
        sourceResults[axis] =
            mlir::getAffineConstantExpr(
                sourceType.getDimSize(axis) - 1, rewriter.getContext()) -
            mlir::getAffineDimExpr(axis, rewriter.getContext());
      }
      opLabel = "tdma_data_move mirror lowering";
      break;
    case InstrDataMoveKind::Rotate90:
    case InstrDataMoveKind::Rotate180:
    case InstrDataMoveKind::Rotate270:
      if (!op.getAxesAttr())
        return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
            rewriter, op, failureReason,
            "tdma_data_move rotate lowering requires axes attr");
      if (!setPermutation(llvm::to_vector(
              llvm::seq<int64_t>(0, sourceType.getRank()))) ||
          op.getAxesAttr().size() != 2)
        return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
            rewriter, op, failureReason,
            "tdma_data_move rotate lowering requires equal ranks and two axes");
      {
        int64_t axis0 = op.getAxesAttr().asArrayRef()[0];
        int64_t axis1 = op.getAxesAttr().asArrayRef()[1];
        if (axis0 < 0 || axis1 < 0 || axis0 >= sourceType.getRank() ||
            axis1 >= sourceType.getRank() || axis0 == axis1)
          return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
              rewriter, op, failureReason,
              "tdma_data_move rotate lowering axes are invalid");
        mlir::AffineExpr d0 =
            mlir::getAffineDimExpr(axis0, rewriter.getContext());
        mlir::AffineExpr d1 =
            mlir::getAffineDimExpr(axis1, rewriter.getContext());
        if (kind == InstrDataMoveKind::Rotate90) {
          sourceResults[axis0] =
              mlir::getAffineConstantExpr(
                  sourceType.getDimSize(axis0) - 1,
                  rewriter.getContext()) -
              d1;
          sourceResults[axis1] = d0;
        } else if (kind == InstrDataMoveKind::Rotate180) {
          sourceResults[axis0] =
              mlir::getAffineConstantExpr(
                  sourceType.getDimSize(axis0) - 1,
                  rewriter.getContext()) -
              d0;
          sourceResults[axis1] =
              mlir::getAffineConstantExpr(
                  sourceType.getDimSize(axis1) - 1,
                  rewriter.getContext()) -
              d1;
        } else {
          sourceResults[axis0] = d1;
          sourceResults[axis1] =
              mlir::getAffineConstantExpr(
                  sourceType.getDimSize(axis1) - 1,
                  rewriter.getContext()) -
              d0;
        }
      }
      opLabel = "tdma_data_move rotate lowering";
      break;
    case InstrDataMoveKind::Pad:
    case InstrDataMoveKind::Img2Col:
      return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
          rewriter, op, failureReason,
          "tdma_data_move pad/img2col remains in the production target "
          "surface");
    }

    analysis::IndexRelationResult sourceRelation =
        analysis::IndexRelation::fromAffineMap(
            mlir::AffineMap::get(destType.getRank(), 0, sourceResults,
                                 rewriter.getContext()),
            destType.getShape(), sourceType.getShape());
    analysis::IndexRelationResult destRelation =
        analysis::IndexRelation::identity(destType.getShape());
    if (!sourceRelation.isExact() || !destRelation.isExact())
      return failFailureOr<llvm::SmallVector<MovementDescriptorPair>>(
          rewriter, op, failureReason,
          llvm::Twine(opLabel).concat(" relation is not exact").str());
    return getRelationMovementDescriptors(
        rewriter, op, sourceType, destType, destType.getShape(),
        *sourceRelation.get(), *destRelation.get(),
        MovementEngine::GatherScatter, failureReason, opLabel);
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
    ScopedLoweringPatternTiming timing(op.getOperation());
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(op.getSource().getType());
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(op.getResult().getType());
    if (!sourceType || !resultType)
      return failPattern(rewriter, op, failureReason,
                         "tile.broadcast lowering requires memref types");

    llvm::ArrayRef<int64_t> dimensions = op.getDimensions();
    if (dimensions.size() !=
        static_cast<size_t>(sourceType.getRank()))
      return failPattern(rewriter, op, failureReason,
                         "tile.broadcast dimension rank mismatch");
    llvm::SmallVector<mlir::AffineExpr, 4> sourceResults;
    sourceResults.reserve(sourceType.getRank());
    llvm::SmallVector<bool, 4> usedResultDims(resultType.getRank(), false);
    for (int64_t resultDim : dimensions) {
      if (resultDim < 0 || resultDim >= resultType.getRank() ||
          usedResultDims[resultDim])
        return failPattern(rewriter, op, failureReason,
                           "tile.broadcast dimensions are invalid");
      usedResultDims[resultDim] = true;
      sourceResults.push_back(
          mlir::getAffineDimExpr(resultDim, rewriter.getContext()));
    }
    analysis::IndexRelationResult sourceRelation =
        analysis::IndexRelation::fromAffineMap(
            mlir::AffineMap::get(resultType.getRank(), 0, sourceResults,
                                 rewriter.getContext()),
            resultType.getShape(), sourceType.getShape());
    analysis::IndexRelationResult destRelation =
        analysis::IndexRelation::identity(resultType.getShape());
    if (!sourceRelation.isExact() || !destRelation.isExact())
      return failPattern(rewriter, op, failureReason,
                         "tile.broadcast relation is not exact");
    mlir::FailureOr<llvm::SmallVector<MovementDescriptorPair>> descriptors =
        getRelationMovementDescriptors(
            rewriter, op, sourceType, resultType, resultType.getShape(),
            *sourceRelation.get(), *destRelation.get(),
            MovementEngine::GatherScatter, failureReason,
            "tile.broadcast lowering");
    if (mlir::failed(descriptors))
      return mlir::failure();

    mlir::FailureOr<mlir::Value> dest = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, failureReason);
    if (mlir::failed(dest))
      return mlir::failure();

    createGatherScatterDescriptors(rewriter, op.getLoc(), op.getSource(),
                                   *dest, *descriptors);
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
    ScopedLoweringPatternTiming timing(op.getOperation());
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

    std::optional<CanonicalReshapeMovementRelations> movementRelations =
        getCanonicalReshapeMovementRelations(
            rewriter.getContext(), sourceType.getShape(),
            resultType.getShape());
    if (!movementRelations)
      return failPattern(
          rewriter, op, failureReason,
          "tile.reshape canonical relation cannot be represented by a "
          "rectangular affine refinement");
    mlir::FailureOr<llvm::SmallVector<MovementDescriptorPair>> descriptors =
        getRelationMovementDescriptors(
            rewriter, op, sourceType, resultType,
            movementRelations->iterationShape,
            movementRelations->iterationToSource,
            movementRelations->iterationToDest,
            MovementEngine::GatherScatter, failureReason,
            "tile.reshape lowering");
    if (mlir::failed(descriptors))
      return mlir::failure();

    mlir::FailureOr<mlir::Value> dest = createDestAlloc(
        op.getLoc(), op.getResult().getType(), rewriter, op, failureReason);
    if (mlir::failed(dest))
      return mlir::failure();

    createGatherScatterDescriptors(rewriter, op.getLoc(), op.getSource(),
                                   *dest, *descriptors);
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
