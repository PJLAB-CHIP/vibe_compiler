//===- TemporalTiling.cpp - Apply live-operation temporal choices -----===//

#include "TemporalTiling.h"

#include "Wafer/Analysis/Linalg/TensorResultIndexing.h"

#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/AffineCanonicalizationUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/TransformUtils.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/MathExtras.h"

#include <functional>

namespace wafer {
namespace {

template <typename T>
mlir::FailureOr<T> fail(TemporalTilingFailure *failure,
                        TemporalTilingFailureKind kind,
                        llvm::StringRef detail) {
  if (failure) {
    failure->kind = kind;
    failure->detail = detail.str();
  }
  return mlir::failure();
}

// Selected parameters remain query-local. Every tile is produced by the
// current source op's standard interface, including its inner reduction.
struct ProducerTiling {
  struct Reduction {
    compiler::detail::TemporalScopeDescriptor descriptor;
    compiler::detail::TemporalScopeChoice choice;
    bool materialized = false;
  };
  llvm::DenseMap<mlir::Operation *, Reduction> reductions;
  TemporalTilingStatistics &statistics;

  explicit ProducerTiling(TemporalTilingStatistics &statistics)
      : statistics(statistics) {}

  mlir::FailureOr<mlir::TilingResult>
  materialize(mlir::IRRewriter &rewriter, mlir::tensor::ExtractSliceOp slice,
              mlir::OpResult producer);
};

// Proofs refer only to live source operations of this call. Erasure invalidates
// membership before an allocator can reuse an address for another operation.
struct LiveFusionSources : mlir::RewriterBase::ForwardingListener {
  struct View {
    mlir::OpResult producer;
    mlir::Value value;
    llvm::SmallVector<compiler::detail::TemporalViewDimensionMapping, 4>
        dimensions;
    std::optional<analysis::IndexRelation> relation;
    llvm::SmallVector<mlir::Operation *, 4> dependencies;
  };
  llvm::DenseMap<mlir::Operation *, bool> sources;
  llvm::DenseMap<mlir::Operation *, View> views;
  explicit LiveFusionSources(mlir::OpBuilder::Listener *listener)
      : ForwardingListener(listener) {}
  void invalidate(mlir::Operation *operation) {
    sources.erase(operation);
    for (auto it = views.begin(); it != views.end();) {
      auto current = it++;
      if (llvm::is_contained(current->second.dependencies, operation))
        views.erase(current);
    }
  }
  void notifyOperationErased(mlir::Operation *operation) override {
    invalidate(operation);
    ForwardingListener::notifyOperationErased(operation);
  }
  void notifyOperationModified(mlir::Operation *operation) override {
    invalidate(operation);
    ForwardingListener::notifyOperationModified(operation);
  }
};

struct ExactFusionInventory {
  llvm::DenseSet<mlir::Value> directEdges;
  llvm::SmallVector<compiler::detail::TemporalFusion, 8> viewPaths;
};

mlir::FailureOr<ExactFusionInventory>
collectExactFusionEdges(TileRegionOp region, TemporalTilingFailure *failure) {
  ExactFusionInventory inventory;
  for (mlir::Operation &operation :
       region.getBody().front().without_terminator()) {
    for (mlir::OpResult result : operation.getResults()) {
      auto query = compiler::detail::queryTemporalFusion(result);
      if (query.kind ==
          compiler::detail::TemporalFusionQueryKind::BrokenContract)
        return fail<ExactFusionInventory>(
            failure, TemporalTilingFailureKind::BrokenContract, query.detail);
      if (!query.isExact() || query.fusion->uses.size() != 1)
        continue;
      const auto &use = query.fusion->uses.front();
      if (use.representation ==
          compiler::detail::TemporalTileRepresentation::RectangularImage)
        continue;
      if (use.consumerValue == result)
        inventory.directEdges.insert(result);
      else
        inventory.viewPaths.push_back(std::move(*query.fusion));
    }
  }
  return inventory;
}

mlir::FailureOr<mlir::Value> reshapeTile(mlir::IRRewriter &rewriter,
                                         mlir::Location location,
                                         mlir::Value source,
                                         mlir::RankedTensorType targetType) {
  auto sourceType = mlir::dyn_cast<mlir::RankedTensorType>(source.getType());
  if (!sourceType || sourceType.getElementType() != targetType.getElementType())
    return mlir::failure();
  if (sourceType == targetType)
    return source;
  if (mlir::tensor::CastOp::areCastCompatible(sourceType, targetType))
    return rewriter.create<mlir::tensor::CastOp>(location, targetType, source)
        .getResult();
  if (sourceType.getRank() == 0 || targetType.getRank() == 0)
    return mlir::failure();
  if (sourceType.hasStaticShape() && targetType.hasStaticShape()) {
    if (sourceType.getNumElements() != targetType.getNumElements())
      return mlir::failure();
  } else {
    // A canonical main/tail loop temporarily has bounded dynamic tile sizes.
    // Inserting/removing unit axes preserves each actual dimension without
    // multiplying or guessing dynamic extents. General dynamic reshape remains
    // outside this materializer; tail specialization later makes these static.
    auto groups =
        mlir::getReassociationIndicesForReshape(sourceType, targetType);
    if (!groups)
      return mlir::failure();
    auto higher =
        sourceType.getRank() > targetType.getRank() ? sourceType : targetType;
    auto lower =
        sourceType.getRank() > targetType.getRank() ? targetType : sourceType;
    for (auto [axis, group] : llvm::enumerate(*groups)) {
      int64_t extent = 1;
      bool found = false;
      for (int64_t member : group) {
        if (higher.getDimSize(member) == 1)
          continue;
        if (found)
          return mlir::failure();
        found = true;
        extent = higher.getDimSize(member);
      }
      if (extent != lower.getDimSize(axis))
        return mlir::failure();
    }
  }

  auto reshape =
      [&](mlir::Value value,
          mlir::RankedTensorType resultType) -> mlir::FailureOr<mlir::Value> {
    auto valueType = mlir::cast<mlir::RankedTensorType>(value.getType());
    auto reassociation =
        mlir::getReassociationIndicesForReshape(valueType, resultType);
    if (!reassociation)
      return mlir::failure();
    if (valueType.getRank() > resultType.getRank())
      return rewriter
          .create<mlir::tensor::CollapseShapeOp>(location, resultType, value,
                                                 *reassociation)
          .getResult();
    if (valueType.getRank() < resultType.getRank())
      return rewriter
          .create<mlir::tensor::ExpandShapeOp>(location, resultType, value,
                                               *reassociation)
          .getResult();
    return mlir::failure();
  };

  return reshape(source, targetType);
}

std::optional<int64_t> resolveStaticIndex(mlir::OpFoldResult value) {
  if (auto attribute = llvm::dyn_cast_if_present<mlir::IntegerAttr>(
          value.dyn_cast<mlir::Attribute>()))
    return attribute.getInt();
  mlir::Value dynamic = value.dyn_cast<mlir::Value>();
  if (!dynamic || !dynamic.getType().isIndex())
    return std::nullopt;
  mlir::FailureOr<int64_t> constant =
      mlir::ValueBoundsConstraintSet::computeConstantBound(
          mlir::presburger::BoundType::EQ,
          mlir::ValueBoundsConstraintSet::Variable(dynamic));
  if (mlir::succeeded(constant))
    return *constant;
  std::function<std::optional<int64_t>(mlir::Value, unsigned)> resolve =
      [&](mlir::Value current, unsigned depth) -> std::optional<int64_t> {
    if (!current || depth > 8)
      return std::nullopt;
    if (std::optional<int64_t> constant = mlir::getConstantIntValue(current))
      return constant;
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(current)) {
      auto loop = argument.getOwner()
                      ? mlir::dyn_cast_or_null<mlir::scf::ForOp>(
                            argument.getOwner()->getParentOp())
                      : mlir::scf::ForOp{};
      if (!loop || argument != loop.getInductionVar())
        return std::nullopt;
      std::optional<int64_t> lower =
          mlir::getConstantIntValue(loop.getLowerBound());
      std::optional<int64_t> upper =
          mlir::getConstantIntValue(loop.getUpperBound());
      std::optional<int64_t> step = mlir::getConstantIntValue(loop.getStep());
      if (!lower || !upper || !step || *step <= 0 || *upper <= *lower ||
          *upper - *lower > *step)
        return std::nullopt;
      return lower;
    }
    mlir::Operation *definition = current.getDefiningOp();
    mlir::AffineMap map;
    mlir::ValueRange operands;
    bool takeMinimum = false;
    bool takeMaximum = false;
    if (auto apply =
            mlir::dyn_cast_or_null<mlir::affine::AffineApplyOp>(definition)) {
      map = apply.getAffineMap();
      operands = apply.getMapOperands();
    } else if (auto minimum = mlir::dyn_cast_or_null<mlir::affine::AffineMinOp>(
                   definition)) {
      map = minimum.getAffineMap();
      operands = minimum.getMapOperands();
      takeMinimum = true;
    } else if (auto maximum = mlir::dyn_cast_or_null<mlir::affine::AffineMaxOp>(
                   definition)) {
      map = maximum.getAffineMap();
      operands = maximum.getMapOperands();
      takeMaximum = true;
    } else {
      return std::nullopt;
    }
    llvm::SmallVector<mlir::Attribute, 4> attributes;
    for (mlir::Value operand : operands) {
      std::optional<int64_t> operandValue = resolve(operand, depth + 1);
      if (!operandValue)
        return std::nullopt;
      attributes.push_back(mlir::IntegerAttr::get(
          mlir::IndexType::get(current.getContext()), *operandValue));
    }
    llvm::SmallVector<mlir::Attribute, 4> folded;
    if (mlir::failed(map.constantFold(attributes, folded)) || folded.empty())
      return std::nullopt;
    std::optional<int64_t> result;
    for (mlir::Attribute attribute : folded) {
      auto integer = mlir::dyn_cast<mlir::IntegerAttr>(attribute);
      if (!integer)
        return std::nullopt;
      if (!result)
        result = integer.getInt();
      else if (takeMinimum)
        result = std::min(*result, integer.getInt());
      else if (takeMaximum)
        result = std::max(*result, integer.getInt());
      else
        return std::nullopt;
    }
    return result;
  };
  return resolve(dynamic, 0);
}

mlir::OpFoldResult addConstantOffset(mlir::IRRewriter &rewriter,
                                     mlir::Location location,
                                     mlir::OpFoldResult base, int64_t offset) {
  if (offset == 0)
    return base;
  if (std::optional<int64_t> constant = mlir::getConstantIntValue(base)) {
    int64_t sum = 0;
    if (!llvm::AddOverflow(*constant, offset, sum))
      return rewriter.getIndexAttr(sum);
  }
  mlir::AffineExpr dimension = mlir::getAffineDimExpr(0, rewriter.getContext());
  mlir::AffineMap map =
      mlir::AffineMap::get(1, 0, dimension + offset, rewriter.getContext());
  return mlir::affine::makeComposedFoldedAffineApply(rewriter, location, map,
                                                     {base});
}

struct ViewFusionRequest {
  mlir::OpResult producer;
  mlir::Value finalView;
  llvm::SmallVector<compiler::detail::TemporalViewDimensionMapping, 4>
      producerDimensions;
  const analysis::IndexRelation *generalReshapeRelation = nullptr;
};

struct ConcatFusionRequest {
  mlir::Value assembledValue;
  llvm::SmallVector<compiler::detail::TemporalConcatSegment, 4> segments;
};

mlir::LogicalResult
fuseViewProducerSlices(mlir::IRRewriter &rewriter, TileRegionOp region,
                       llvm::ArrayRef<ViewFusionRequest> requests,
                       TemporalTilingStatistics &statistics,
                       ProducerTiling &producerTiling) {
  for (const ViewFusionRequest &request : requests) {
    llvm::SmallVector<mlir::tensor::ExtractSliceOp, 4> slices;
    region.walk([&](mlir::tensor::ExtractSliceOp slice) {
      if (slice.getSource() == request.finalView)
        slices.push_back(slice);
    });
    if (slices.empty())
      return mlir::failure();
    for (mlir::tensor::ExtractSliceOp slice : slices) {
      if (!slice || !slice->getBlock())
        return mlir::failure();
      rewriter.setInsertionPoint(slice);
      llvm::SmallVector<mlir::OpFoldResult, 4> viewOffsets =
          slice.getMixedOffsets();
      llvm::SmallVector<mlir::OpFoldResult, 4> viewSizes =
          slice.getMixedSizes();
      llvm::SmallVector<mlir::OpFoldResult, 4> producerOffsets;
      llvm::SmallVector<mlir::OpFoldResult, 4> producerSizes;
      auto producerType =
          mlir::dyn_cast<mlir::RankedTensorType>(request.producer.getType());
      auto viewType =
          mlir::dyn_cast<mlir::RankedTensorType>(request.finalView.getType());
      auto targetType = mlir::dyn_cast<mlir::RankedTensorType>(slice.getType());
      if (!producerType || !viewType || !targetType)
        return mlir::failure();
      if (request.generalReshapeRelation) {
        if (!request.generalReshapeRelation
                 ->hasCanonicalRowMajorReshapeConstruction())
          return mlir::failure();
        std::optional<llvm::SmallVector<mlir::ReassociationIndices>>
            reassociation =
                mlir::getReassociationIndicesForReshape(producerType, viewType);
        if (!reassociation)
          return mlir::failure();
        llvm::SmallVector<int64_t, 6> staticViewOffsets;
        llvm::SmallVector<int64_t, 6> staticViewSizes;
        for (mlir::OpFoldResult offset : viewOffsets) {
          std::optional<int64_t> constant = resolveStaticIndex(offset);
          if (!constant) {
            staticViewOffsets.clear();
            break;
          }
          staticViewOffsets.push_back(*constant);
        }
        if (!staticViewOffsets.empty())
          for (mlir::OpFoldResult size : viewSizes) {
            std::optional<int64_t> constant = resolveStaticIndex(size);
            if (!constant) {
              staticViewSizes.clear();
              break;
            }
            staticViewSizes.push_back(*constant);
          }
        if (staticViewOffsets.size() == viewOffsets.size() &&
            staticViewSizes.size() == viewSizes.size()) {
          analysis::StaticRectangularIndexSetPiecesResult sourcePieces =
              request.generalReshapeRelation
                  ->getExactStaticRectangularImagePieces(staticViewOffsets,
                                                         staticViewSizes);
          analysis::IndexRelationResult inverse =
              request.generalReshapeRelation->inverse();
          if (!sourcePieces.isExact() || sourcePieces.domains.empty() ||
              !inverse.isExact())
            return mlir::failure();
          auto assembledType = mlir::RankedTensorType::get(
              staticViewSizes, targetType.getElementType(),
              targetType.getEncoding());
          mlir::Value assembled = rewriter.create<mlir::tensor::EmptyOp>(
              slice.getLoc(), staticViewSizes, targetType.getElementType());
          for (const analysis::StaticRectangularIndexSet &sourcePiece :
               sourcePieces.domains) {
            analysis::StaticRectangularIndexSetPiecesResult viewPieces =
                inverse.get()->getExactStaticRectangularImagePieces(
                    sourcePiece.offsets, sourcePiece.sizes);
            if (!viewPieces.isExact() || viewPieces.domains.size() != 1)
              return mlir::failure();
            const analysis::StaticRectangularIndexSet &viewPiece =
                viewPieces.domains.front();
            llvm::SmallVector<int64_t, 6> localOffsets;
            for (auto [pieceOffset, requestOffset, pieceSize, requestSize] :
                 llvm::zip_equal(viewPiece.offsets, staticViewOffsets,
                                 viewPiece.sizes, staticViewSizes)) {
              if (pieceOffset < requestOffset || pieceSize <= 0 ||
                  pieceOffset - requestOffset > requestSize - pieceSize)
                return mlir::failure();
              localOffsets.push_back(pieceOffset - requestOffset);
            }
            llvm::SmallVector<mlir::OpFoldResult, 6> sourceOffsets;
            llvm::SmallVector<mlir::OpFoldResult, 6> sourceSizes;
            llvm::SmallVector<mlir::OpFoldResult, 6> sourceStrides(
                sourcePiece.offsets.size(), rewriter.getIndexAttr(1));
            for (int64_t offset : sourcePiece.offsets)
              sourceOffsets.push_back(rewriter.getIndexAttr(offset));
            for (int64_t size : sourcePiece.sizes)
              sourceSizes.push_back(rewriter.getIndexAttr(size));
            auto sourceSlice = rewriter.create<mlir::tensor::ExtractSliceOp>(
                slice.getLoc(), request.producer, sourceOffsets, sourceSizes,
                sourceStrides);
            mlir::FailureOr<mlir::TilingResult> tiled =
                producerTiling.materialize(rewriter, sourceSlice,
                                           request.producer);
            if (mlir::failed(tiled) || tiled->tiledValues.size() != 1)
              return mlir::failure();
            rewriter.eraseOp(sourceSlice);
            auto viewPieceType = mlir::RankedTensorType::get(
                viewPiece.sizes, targetType.getElementType(),
                targetType.getEncoding());
            mlir::FailureOr<mlir::Value> viewValue =
                reshapeTile(rewriter, slice.getLoc(),
                            tiled->tiledValues.front(), viewPieceType);
            if (mlir::failed(viewValue))
              return mlir::failure();
            llvm::SmallVector<mlir::OpFoldResult, 6> insertOffsets;
            llvm::SmallVector<mlir::OpFoldResult, 6> insertSizes;
            llvm::SmallVector<mlir::OpFoldResult, 6> insertStrides(
                localOffsets.size(), rewriter.getIndexAttr(1));
            for (int64_t offset : localOffsets)
              insertOffsets.push_back(rewriter.getIndexAttr(offset));
            for (int64_t size : viewPiece.sizes)
              insertSizes.push_back(rewriter.getIndexAttr(size));
            assembled = rewriter.create<mlir::tensor::InsertSliceOp>(
                slice.getLoc(), *viewValue, assembled, insertOffsets,
                insertSizes, insertStrides);
          }
          mlir::FailureOr<mlir::Value> replacement =
              reshapeTile(rewriter, slice.getLoc(), assembled, targetType);
          if (mlir::failed(replacement) || assembled.getType() != assembledType)
            return mlir::failure();
          rewriter.replaceOp(slice, *replacement);
          continue;
        }
        auto requireFullViewDimension = [&](unsigned dimension) {
          return dimension < viewOffsets.size() &&
                 dimension < viewSizes.size() &&
                 mlir::isConstantIntValue(viewOffsets[dimension], 0) &&
                 mlir::isConstantIntValue(viewSizes[dimension],
                                          viewType.getDimSize(dimension));
        };
        if (producerType.getRank() > viewType.getRank()) {
          for (auto [viewDimension, producerDimensions] :
               llvm::enumerate(*reassociation)) {
            if (producerDimensions.size() == 1) {
              producerOffsets.push_back(viewOffsets[viewDimension]);
              producerSizes.push_back(viewSizes[viewDimension]);
              continue;
            }
            if (!requireFullViewDimension(viewDimension))
              return mlir::failure();
            for (int64_t producerDimension : producerDimensions) {
              producerOffsets.push_back(rewriter.getIndexAttr(0));
              producerSizes.push_back(rewriter.getIndexAttr(
                  producerType.getDimSize(producerDimension)));
            }
          }
        } else {
          for (auto [producerDimension, viewDimensions] :
               llvm::enumerate(*reassociation)) {
            if (viewDimensions.size() == 1) {
              const unsigned viewDimension = viewDimensions.front();
              producerOffsets.push_back(viewOffsets[viewDimension]);
              producerSizes.push_back(viewSizes[viewDimension]);
              continue;
            }
            if (llvm::any_of(viewDimensions, [&](int64_t viewDimension) {
                  return !requireFullViewDimension(viewDimension);
                }))
              return mlir::failure();
            producerOffsets.push_back(rewriter.getIndexAttr(0));
            producerSizes.push_back(rewriter.getIndexAttr(
                producerType.getDimSize(producerDimension)));
          }
        }
      } else {
        for (const auto &mapping : request.producerDimensions) {
          if (mapping.viewDimension < 0) {
            producerOffsets.push_back(rewriter.getIndexAttr(mapping.offset));
            producerSizes.push_back(rewriter.getIndexAttr(1));
            continue;
          }
          const unsigned dimension =
              static_cast<unsigned>(mapping.viewDimension);
          if (dimension >= viewOffsets.size() || dimension >= viewSizes.size())
            return mlir::failure();
          producerOffsets.push_back(addConstantOffset(rewriter, slice.getLoc(),
                                                      viewOffsets[dimension],
                                                      mapping.offset));
          producerSizes.push_back(viewSizes[dimension]);
        }
      }
      if (!producerType ||
          producerOffsets.size() != static_cast<size_t>(producerType.getRank()))
        return mlir::failure();
      llvm::SmallVector<mlir::OpFoldResult, 4> strides(
          producerType.getRank(), rewriter.getIndexAttr(1));
      auto requestSlice = rewriter.create<mlir::tensor::ExtractSliceOp>(
          slice.getLoc(), request.producer, producerOffsets, producerSizes,
          strides);
      mlir::FailureOr<mlir::TilingResult> tiled =
          producerTiling.materialize(rewriter, requestSlice, request.producer);
      if (mlir::failed(tiled) || tiled->tiledValues.empty()) {
        rewriter.eraseOp(requestSlice);
        return mlir::failure();
      }
      mlir::FailureOr<mlir::Value> replacement = mlir::failure();
      if (request.generalReshapeRelation) {
        auto reassociation =
            mlir::getReassociationIndicesForReshape(producerType, viewType);
        if (!reassociation)
          return mlir::failure();
        mlir::Value tiledValue = tiled->tiledValues.front();
        llvm::SmallVector<int64_t, 6> preciseShape(targetType.getShape());
        for (auto [dimension, size] : llvm::enumerate(viewSizes))
          if (std::optional<int64_t> constant = mlir::getConstantIntValue(size))
            preciseShape[dimension] = *constant;
        if (producerType.getRank() > viewType.getRank()) {
          auto tiledType =
              mlir::dyn_cast<mlir::RankedTensorType>(tiledValue.getType());
          if (!tiledType)
            return mlir::failure();
          preciseShape.clear();
          for (llvm::ArrayRef<int64_t> group : *reassociation) {
            int64_t extent = 1;
            for (int64_t dimension : group) {
              int64_t current = tiledType.getDimSize(dimension);
              if (mlir::ShapedType::isDynamic(current)) {
                extent = mlir::ShapedType::kDynamic;
                break;
              }
              if (llvm::MulOverflow(extent, current, extent))
                return mlir::failure();
            }
            preciseShape.push_back(extent);
          }
        } else {
          auto tiledType =
              mlir::dyn_cast<mlir::RankedTensorType>(tiledValue.getType());
          if (!tiledType)
            return mlir::failure();
          for (auto [producerDimension, viewDimensions] :
               llvm::enumerate(*reassociation)) {
            if (viewDimensions.size() == 1) {
              preciseShape[viewDimensions.front()] =
                  tiledType.getDimSize(producerDimension);
              continue;
            }
            for (int64_t viewDimension : viewDimensions)
              preciseShape[viewDimension] = viewType.getDimSize(viewDimension);
          }
        }
        auto preciseType = mlir::RankedTensorType::get(
            preciseShape, targetType.getElementType(),
            targetType.getEncoding());
        mlir::Value preciseValue;
        if (producerType.getRank() > viewType.getRank())
          preciseValue =
              rewriter
                  .create<mlir::tensor::CollapseShapeOp>(
                      slice.getLoc(), preciseType, tiledValue, *reassociation)
                  .getResult();
        else
          preciseValue =
              rewriter
                  .create<mlir::tensor::ExpandShapeOp>(
                      slice.getLoc(), preciseType, tiledValue, *reassociation)
                  .getResult();
        replacement =
            reshapeTile(rewriter, slice.getLoc(), preciseValue, targetType);
      } else {
        // The exact projection identifies each local view axis, including
        // unit axes in a tail. Preserve shape precision across an inner SCF
        // recurrence without guessing a reassociation from dynamic types.
        auto value = tiled->tiledValues.front();
        auto valueType = mlir::cast<mlir::RankedTensorType>(value.getType());
        llvm::SmallVector<int64_t, 6> sourceShape(valueType.getShape());
        llvm::SmallVector<int64_t, 6> viewShape(targetType.getShape());
        for (auto [axis, mapping] :
             llvm::enumerate(request.producerDimensions)) {
          if (mapping.viewDimension < 0)
            continue;
          auto &sourceExtent = sourceShape[axis];
          auto &viewExtent = viewShape[mapping.viewDimension];
          if (mlir::ShapedType::isDynamic(sourceExtent))
            sourceExtent = viewExtent;
          if (!mlir::ShapedType::isDynamic(viewExtent) &&
              !mlir::ShapedType::isDynamic(sourceExtent) &&
              viewExtent != sourceExtent)
            return mlir::failure();
          viewExtent = sourceExtent;
        }
        auto preciseSource = mlir::RankedTensorType::get(
            sourceShape, valueType.getElementType(), valueType.getEncoding());
        if (preciseSource != valueType)
          value = rewriter.create<mlir::tensor::CastOp>(slice.getLoc(),
                                                        preciseSource, value);
        auto preciseView = mlir::RankedTensorType::get(
            viewShape, targetType.getElementType(), targetType.getEncoding());
        auto reshaped =
            reshapeTile(rewriter, slice.getLoc(), value, preciseView);
        if (mlir::failed(reshaped))
          return mlir::failure();
        replacement =
            reshapeTile(rewriter, slice.getLoc(), *reshaped, targetType);
      }
      rewriter.eraseOp(requestSlice);
      if (mlir::failed(replacement))
        return mlir::failure();
      rewriter.replaceOp(slice, *replacement);
    }
    ++statistics.fusedProducers;
    ++statistics.viewTransparentProducers;
  }
  return mlir::success();
}

bool sameSliceRequest(mlir::tensor::ExtractSliceOp left,
                      mlir::tensor::ExtractSliceOp right) {
  return left.getType() == right.getType() &&
         left.getMixedOffsets() == right.getMixedOffsets() &&
         left.getMixedSizes() == right.getMixedSizes() &&
         left.getMixedStrides() == right.getMixedStrides();
}

mlir::LogicalResult
fuseJointProducerSlices(mlir::IRRewriter &rewriter, TileRegionOp region,
                        llvm::ArrayRef<compiler::detail::TemporalFusion> groups,
                        TemporalTilingStatistics &statistics,
                        std::string &detail, ProducerTiling &producerTiling) {
  auto reject = [&](llvm::StringRef message) {
    detail = message.str();
    return mlir::failure();
  };
  for (const auto &group : groups) {
    if (group.uses.size() < 2 || llvm::any_of(group.uses, [](const auto &use) {
          return use.representation ==
                 compiler::detail::TemporalTileRepresentation::RectangularImage;
        }))
      continue;
    const auto &use = group.uses.front();
    if (!group.producer || !use.consumerValue ||
        !group.producer.getOwner()->getBlock())
      return reject("joint producer is no longer current");
    llvm::SmallVector<mlir::tensor::ExtractSliceOp, 8> slices;
    region.walk([&](mlir::tensor::ExtractSliceOp slice) {
      if (slice.getSource() == use.consumerValue)
        slices.push_back(slice);
    });
    if (slices.empty() && !use.consumerValue.use_empty())
      continue;
    if (slices.empty())
      return reject("joint producer has no tiled consumer slice");

    llvm::SmallVector<llvm::SmallVector<mlir::tensor::ExtractSliceOp, 4>, 2>
        equivalentRequests;
    for (mlir::tensor::ExtractSliceOp slice : slices) {
      auto found = llvm::find_if(equivalentRequests, [&](const auto &current) {
        return sameSliceRequest(current.front(), slice);
      });
      if (found == equivalentRequests.end())
        equivalentRequests.push_back({slice});
      else
        found->push_back(slice);
    }
    if (use.consumerValue != group.producer) {
      for (const auto &requestGroup : equivalentRequests) {
        mlir::tensor::ExtractSliceOp representative = requestGroup.front();
        for (mlir::tensor::ExtractSliceOp duplicate :
             llvm::drop_begin(requestGroup))
          rewriter.replaceOp(duplicate, representative.getResult());
      }
      ViewFusionRequest request{
          group.producer, use.consumerValue, use.producerDimensions,
          use.representation ==
                  compiler::detail::TemporalTileRepresentation::ReshapePieces
              ? &*use.viewToProducer
              : nullptr};
      if (mlir::failed(fuseViewProducerSlices(rewriter, region, {request},
                                              statistics, producerTiling)))
        return reject(
            "joint view producer could not materialize its common slices");
      continue;
    }
    for (const auto &requestGroup : equivalentRequests) {
      mlir::tensor::ExtractSliceOp representative = requestGroup.front();
      rewriter.setInsertionPoint(representative);
      auto request = rewriter.create<mlir::tensor::ExtractSliceOp>(
          representative.getLoc(), group.producer,
          representative.getMixedOffsets(), representative.getMixedSizes(),
          representative.getMixedStrides());
      mlir::FailureOr<mlir::TilingResult> tiled =
          producerTiling.materialize(rewriter, request, group.producer);
      if (mlir::failed(tiled) || tiled->tiledValues.size() != 1) {
        rewriter.eraseOp(request);
        return reject(
            "joint producer TilingInterface rejected the common slice");
      }
      mlir::Value sharedTile = tiled->tiledValues.front();
      rewriter.eraseOp(request);
      auto requestedType =
          mlir::dyn_cast<mlir::RankedTensorType>(representative.getType());
      if (!requestedType)
        return reject("joint producer common slice is not a ranked tensor");
      mlir::FailureOr<mlir::Value> replacement = reshapeTile(
          rewriter, representative.getLoc(), sharedTile, requestedType);
      if (mlir::failed(replacement))
        return reject(
            "joint producer tile cannot match the consumer slice type");
      for (mlir::tensor::ExtractSliceOp slice : requestGroup) {
        if (slice.getType() != (*replacement).getType())
          return reject("joint producer slice types differ across consumers");
        rewriter.replaceOp(slice, *replacement);
      }
    }
    if (!group.producer.use_empty())
      return reject("joint producer still has an unfused current use");
    producerTiling.reductions.erase(group.producer.getOwner());
    rewriter.eraseOp(group.producer.getOwner());
    ++statistics.fusedProducers;
  }
  return mlir::success();
}

struct CanonicalLoopGrid {
  mlir::Value induction;
  int64_t lower = 0;
  int64_t upper = 0;
  int64_t step = 0;
};

std::optional<CanonicalLoopGrid>
getCanonicalLoopGrid(mlir::OpFoldResult offset) {
  auto value = mlir::dyn_cast<mlir::Value>(offset);
  auto argument = value ? mlir::dyn_cast<mlir::BlockArgument>(value)
                        : mlir::BlockArgument{};
  auto loop = argument && argument.getOwner()
                  ? mlir::dyn_cast_or_null<mlir::scf::ForOp>(
                        argument.getOwner()->getParentOp())
                  : mlir::scf::ForOp{};
  if (!loop || argument != loop.getInductionVar())
    return std::nullopt;
  std::optional<int64_t> lower =
      mlir::getConstantIntValue(loop.getLowerBound());
  std::optional<int64_t> upper =
      mlir::getConstantIntValue(loop.getUpperBound());
  std::optional<int64_t> step = mlir::getConstantIntValue(loop.getStep());
  if (!lower || !upper || !step || *lower < 0 || *upper <= *lower || *step <= 0)
    return std::nullopt;
  return CanonicalLoopGrid{value, *lower, *upper, *step};
}

std::optional<int64_t> getGridFloor(const CanonicalLoopGrid &grid,
                                    int64_t coordinate) {
  if (coordinate < grid.lower || coordinate >= grid.upper)
    return std::nullopt;
  int64_t delta = coordinate - grid.lower;
  int64_t multiple = 0;
  int64_t value = 0;
  if (llvm::MulOverflow(delta / grid.step, grid.step, multiple) ||
      llvm::AddOverflow(grid.lower, multiple, value) || value < grid.lower ||
      value >= grid.upper)
    return std::nullopt;
  return value;
}

std::optional<int64_t> getGridCeil(const CanonicalLoopGrid &grid,
                                   int64_t coordinate) {
  if (coordinate <= grid.lower)
    return grid.lower;
  if (coordinate >= grid.upper)
    return std::nullopt;
  int64_t delta = coordinate - grid.lower;
  int64_t rounded = 0;
  int64_t multiple = 0;
  int64_t value = 0;
  if (llvm::AddOverflow(delta, grid.step - 1, rounded) ||
      llvm::MulOverflow(rounded / grid.step, grid.step, multiple) ||
      llvm::AddOverflow(grid.lower, multiple, value) || value >= grid.upper)
    return std::nullopt;
  return value;
}

std::optional<unsigned> getConcatPartitionDimension(
    mlir::RankedTensorType assembledType,
    llvm::ArrayRef<compiler::detail::TemporalConcatSegment> segments) {
  if (!assembledType || !assembledType.hasStaticShape() || segments.size() < 2)
    return std::nullopt;
  std::optional<unsigned> partitionDimension;
  for (const auto &segment : segments) {
    if (segment.offsets.size() !=
            static_cast<size_t>(assembledType.getRank()) ||
        segment.sizes.size() != static_cast<size_t>(assembledType.getRank()))
      return std::nullopt;
    for (unsigned dimension = 0; dimension < assembledType.getRank();
         ++dimension) {
      const bool full =
          segment.offsets[dimension] == 0 &&
          segment.sizes[dimension] == assembledType.getDimSize(dimension);
      if (full)
        continue;
      if (partitionDimension && *partitionDimension != dimension)
        return std::nullopt;
      partitionDimension = dimension;
    }
  }
  if (!partitionDimension)
    return std::nullopt;
  llvm::SmallVector<std::pair<int64_t, int64_t>, 4> intervals;
  for (const auto &segment : segments) {
    int64_t begin = segment.offsets[*partitionDimension];
    int64_t end = 0;
    if (begin < 0 || segment.sizes[*partitionDimension] <= 0 ||
        llvm::AddOverflow(begin, segment.sizes[*partitionDimension], end))
      return std::nullopt;
    intervals.push_back({begin, end});
  }
  llvm::sort(intervals);
  int64_t covered = 0;
  for (auto [begin, end] : intervals) {
    if (begin != covered || end <= begin)
      return std::nullopt;
    covered = end;
  }
  if (covered != assembledType.getDimSize(*partitionDimension))
    return std::nullopt;
  return partitionDimension;
}

mlir::FailureOr<mlir::scf::ForOp> splitForLoopAt(mlir::IRRewriter &rewriter,
                                                 mlir::scf::ForOp loop,
                                                 int64_t split) {
  std::optional<int64_t> lower =
      mlir::getConstantIntValue(loop.getLowerBound());
  std::optional<int64_t> upper =
      mlir::getConstantIntValue(loop.getUpperBound());
  std::optional<int64_t> step = mlir::getConstantIntValue(loop.getStep());
  if (!lower || !upper || !step || *step <= 0 || split <= *lower ||
      split >= *upper || (split - *lower) % *step != 0)
    return mlir::failure();

  mlir::RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(loop);
  mlir::Value splitValue =
      rewriter.create<mlir::arith::ConstantIndexOp>(loop.getLoc(), split);
  rewriter.setInsertionPointAfter(loop);
  auto suffix = mlir::cast<mlir::scf::ForOp>(rewriter.clone(*loop));
  rewriter.modifyOpInPlace(
      suffix, [&] { suffix.getLowerBoundMutable().assign(splitValue); });
  rewriter.replaceAllUsesWith(loop.getResults(), suffix.getResults());
  rewriter.modifyOpInPlace(
      suffix, [&] { suffix.getInitArgsMutable().assign(loop.getResults()); });
  rewriter.modifyOpInPlace(
      loop, [&] { loop.getUpperBoundMutable().assign(splitValue); });
  return suffix;
}

mlir::LogicalResult
specializeConcatLoopBoundaries(mlir::IRRewriter &rewriter, TileRegionOp region,
                               llvm::ArrayRef<ConcatFusionRequest> requests,
                               TemporalTilingStatistics &statistics) {
  for (const ConcatFusionRequest &request : requests) {
    auto assembledType = mlir::dyn_cast<mlir::RankedTensorType>(
        request.assembledValue.getType());
    std::optional<unsigned> partitionDimension =
        getConcatPartitionDimension(assembledType, request.segments);
    if (!partitionDimension)
      return mlir::failure();
    llvm::SmallVector<int64_t, 4> boundaries;
    for (const auto &segment : request.segments) {
      int64_t end = 0;
      if (llvm::AddOverflow(segment.offsets[*partitionDimension],
                            segment.sizes[*partitionDimension], end))
        return mlir::failure();
      if (end != assembledType.getDimSize(*partitionDimension))
        boundaries.push_back(end);
    }
    llvm::sort(boundaries);
    boundaries.erase(std::unique(boundaries.begin(), boundaries.end()),
                     boundaries.end());

    llvm::SmallVector<mlir::tensor::ExtractSliceOp, 4> slices;
    region.walk([&](mlir::tensor::ExtractSliceOp slice) {
      if (slice.getSource() == request.assembledValue)
        slices.push_back(slice);
    });
    for (mlir::tensor::ExtractSliceOp slice : slices) {
      llvm::SmallVector<mlir::OpFoldResult, 4> offsets =
          slice.getMixedOffsets();
      llvm::SmallVector<mlir::OpFoldResult, 4> sizes = slice.getMixedSizes();
      if (*partitionDimension >= offsets.size() ||
          *partitionDimension >= sizes.size())
        return mlir::failure();
      if (resolveStaticIndex(offsets[*partitionDimension]))
        continue;
      std::optional<int64_t> requestLength =
          resolveStaticIndex(sizes[*partitionDimension]);
      std::optional<CanonicalLoopGrid> grid =
          getCanonicalLoopGrid(offsets[*partitionDimension]);
      if (!requestLength || *requestLength <= 0 || !grid ||
          grid->step != *requestLength)
        return mlir::failure();

      llvm::SmallVector<int64_t, 4> splitPoints;
      for (int64_t boundary : boundaries) {
        if (boundary <= grid->lower || boundary >= grid->upper)
          continue;
        if ((boundary - grid->lower) % grid->step == 0) {
          splitPoints.push_back(boundary);
          continue;
        }
        std::optional<int64_t> boundaryTile = getGridFloor(*grid, boundary);
        if (!boundaryTile)
          return mlir::failure();
        if (*boundaryTile > grid->lower)
          splitPoints.push_back(*boundaryTile);
        int64_t afterBoundaryTile = 0;
        if (llvm::AddOverflow(*boundaryTile, grid->step, afterBoundaryTile))
          return mlir::failure();
        if (afterBoundaryTile < grid->upper)
          splitPoints.push_back(afterBoundaryTile);
      }
      llvm::sort(splitPoints);
      splitPoints.erase(std::unique(splitPoints.begin(), splitPoints.end()),
                        splitPoints.end());
      mlir::scf::ForOp current = mlir::cast<mlir::scf::ForOp>(
          mlir::cast<mlir::BlockArgument>(grid->induction)
              .getOwner()
              ->getParentOp());
      for (int64_t split : splitPoints) {
        mlir::FailureOr<mlir::scf::ForOp> suffix =
            splitForLoopAt(rewriter, current, split);
        if (mlir::failed(suffix))
          return mlir::failure();
        current = *suffix;
        ++statistics.loops;
        ++statistics.specializedConcatBoundaries;
      }
    }
  }
  return mlir::success();
}

// The assembly query proves the value, not the selected traversal's ability to
// rebuild it with static pieces. Overlapping windows can still read the
// already materialized assembly; they do not require concat fusion to tile.
bool canSpecializeConcatSlices(TileRegionOp region,
                               const ConcatFusionRequest &request) {
  auto type =
      mlir::dyn_cast<mlir::RankedTensorType>(request.assembledValue.getType());
  auto axis = getConcatPartitionDimension(type, request.segments);
  if (!axis)
    return false;
  bool found = false;
  bool supported = true;
  region.walk([&](mlir::tensor::ExtractSliceOp slice) {
    if (slice.getSource() != request.assembledValue)
      return;
    found = true;
    if (slice.getType().getRank() != type.getRank()) {
      supported = false;
      return;
    }
    for (mlir::OpFoldResult size : slice.getMixedSizes()) {
      auto length = resolveStaticIndex(size);
      if (!length || *length <= 0)
        supported = false;
    }
    for (mlir::OpFoldResult stride : slice.getMixedStrides())
      if (resolveStaticIndex(stride) != 1)
        supported = false;
    auto offset = slice.getMixedOffsets()[*axis];
    if (resolveStaticIndex(offset))
      return;
    auto grid = getCanonicalLoopGrid(offset);
    auto length = resolveStaticIndex(slice.getMixedSizes()[*axis]);
    if (!grid || !length || grid->step != *length)
      supported = false;
  });
  return found && supported;
}

mlir::LogicalResult
fuseConcatSlices(mlir::IRRewriter &rewriter, TileRegionOp region,
                 llvm::ArrayRef<ConcatFusionRequest> requests,
                 TemporalTilingStatistics &statistics,
                 ProducerTiling &producerTiling) {
  for (const ConcatFusionRequest &request : requests) {
    auto assembledType = mlir::dyn_cast<mlir::RankedTensorType>(
        request.assembledValue.getType());
    std::optional<unsigned> partitionDimension =
        getConcatPartitionDimension(assembledType, request.segments);
    if (!partitionDimension)
      return mlir::failure();
    llvm::DenseSet<mlir::Operation *> fusedProducerOwners;
    llvm::SmallVector<mlir::tensor::ExtractSliceOp, 4> slices;
    region.walk([&](mlir::tensor::ExtractSliceOp slice) {
      if (slice.getSource() == request.assembledValue)
        slices.push_back(slice);
    });
    if (slices.empty())
      return mlir::failure();
    for (mlir::tensor::ExtractSliceOp slice : slices) {
      auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(slice.getType());
      if (!resultType)
        return mlir::failure();
      rewriter.setInsertionPoint(slice);
      llvm::SmallVector<mlir::OpFoldResult, 4> requestOffsets =
          slice.getMixedOffsets();
      llvm::SmallVector<mlir::OpFoldResult, 4> requestSizes =
          slice.getMixedSizes();
      llvm::SmallVector<int64_t, 4> staticRequestSizes;
      for (mlir::OpFoldResult size : requestSizes) {
        std::optional<int64_t> constant = resolveStaticIndex(size);
        if (!constant || *constant <= 0)
          return mlir::failure();
        staticRequestSizes.push_back(*constant);
      }
      if (staticRequestSizes.size() != requestOffsets.size() ||
          staticRequestSizes.size() !=
              static_cast<size_t>(resultType.getRank()))
        return mlir::failure();
      auto empty = rewriter.create<mlir::tensor::EmptyOp>(
          slice.getLoc(), staticRequestSizes, resultType.getElementType(),
          resultType.getEncoding());
      mlir::Value assembled = empty;
      auto staticResultType = empty.getType();

      const unsigned axis = *partitionDimension;
      const int64_t requestLength = staticRequestSizes[axis];
      std::optional<int64_t> constantRequestBegin =
          resolveStaticIndex(requestOffsets[axis]);
      std::optional<CanonicalLoopGrid> grid;
      if (!constantRequestBegin) {
        grid = getCanonicalLoopGrid(requestOffsets[axis]);
        if (!grid || grid->step != requestLength)
          return mlir::failure();
      }

      auto createConditionForExactBegin = [&](int64_t begin) {
        mlir::Value expected = rewriter.create<mlir::arith::ConstantIndexOp>(
            slice.getLoc(), begin);
        return rewriter.create<mlir::arith::CmpIOp>(
            slice.getLoc(), mlir::arith::CmpIPredicate::eq, grid->induction,
            expected);
      };
      auto createConditionForRange = [&](int64_t first, int64_t last) {
        if (first == last)
          return createConditionForExactBegin(first).getResult();
        mlir::Value lower = rewriter.create<mlir::arith::ConstantIndexOp>(
            slice.getLoc(), first);
        mlir::Value upper =
            rewriter.create<mlir::arith::ConstantIndexOp>(slice.getLoc(), last);
        mlir::Value atLeast = rewriter.create<mlir::arith::CmpIOp>(
            slice.getLoc(), mlir::arith::CmpIPredicate::sge, grid->induction,
            lower);
        mlir::Value atMost = rewriter.create<mlir::arith::CmpIOp>(
            slice.getLoc(), mlir::arith::CmpIPredicate::sle, grid->induction,
            upper);
        return rewriter
            .create<mlir::arith::AndIOp>(slice.getLoc(), atLeast, atMost)
            .getResult();
      };

      auto emitPiece =
          [&](const compiler::detail::TemporalConcatSegment &segment,
              mlir::OpFoldResult sourceAxisOffset,
              mlir::OpFoldResult destinationAxisOffset, int64_t axisSize,
              mlir::Value condition) -> mlir::LogicalResult {
        llvm::SmallVector<mlir::OpFoldResult, 4> sourceOffsets;
        llvm::SmallVector<mlir::OpFoldResult, 4> destinationOffsets;
        llvm::SmallVector<mlir::OpFoldResult, 4> pieceSizes;
        for (unsigned dimension = 0; dimension < staticRequestSizes.size();
             ++dimension) {
          sourceOffsets.push_back(
              dimension == axis ? sourceAxisOffset : requestOffsets[dimension]);
          destinationOffsets.push_back(dimension == axis
                                           ? destinationAxisOffset
                                           : rewriter.getIndexAttr(0));
          pieceSizes.push_back(rewriter.getIndexAttr(
              dimension == axis ? axisSize : staticRequestSizes[dimension]));
        }
        llvm::SmallVector<mlir::OpFoldResult, 4> strides(
            staticRequestSizes.size(), rewriter.getIndexAttr(1));
        mlir::Value current = assembled;
        mlir::scf::IfOp select;
        if (condition) {
          select = rewriter.create<mlir::scf::IfOp>(
              slice.getLoc(), mlir::TypeRange{staticResultType}, condition,
              /*withElseRegion=*/true);
          rewriter.setInsertionPointToStart(&select.getThenRegion().front());
        }
        auto sourceSlice = rewriter.create<mlir::tensor::ExtractSliceOp>(
            slice.getLoc(), segment.source, sourceOffsets, pieceSizes, strides);
        mlir::Value sourcePiece = sourceSlice;
        if (segment.derivedProducer) {
          mlir::FailureOr<mlir::TilingResult> tiled =
              producerTiling.materialize(rewriter, sourceSlice,
                                         *segment.derivedProducer);
          if (mlir::failed(tiled) || tiled->tiledValues.empty())
            return mlir::failure();
          sourcePiece = tiled->tiledValues.front();
          rewriter.eraseOp(sourceSlice);
          fusedProducerOwners.insert(segment.derivedProducer->getOwner());
        }
        mlir::Value inserted = rewriter.create<mlir::tensor::InsertSliceOp>(
            slice.getLoc(), sourcePiece, current, destinationOffsets,
            pieceSizes, strides);
        if (!condition) {
          assembled = inserted;
        } else {
          rewriter.create<mlir::scf::YieldOp>(slice.getLoc(), inserted);
          rewriter.setInsertionPointToStart(&select.getElseRegion().front());
          rewriter.create<mlir::scf::YieldOp>(slice.getLoc(), current);
          assembled = select.getResult(0);
          rewriter.setInsertionPointAfter(select);
        }
        ++statistics.assembledSegments;
        return mlir::success();
      };

      for (const auto &segment : request.segments) {
        auto sourceType =
            mlir::dyn_cast<mlir::RankedTensorType>(segment.source.getType());
        if (!sourceType ||
            segment.offsets.size() !=
                static_cast<size_t>(sourceType.getRank()) ||
            segment.sizes.size() != static_cast<size_t>(sourceType.getRank()))
          return mlir::failure();
        const int64_t segmentBegin = segment.offsets[axis];
        int64_t segmentEnd = 0;
        if (llvm::AddOverflow(segmentBegin, segment.sizes[axis], segmentEnd))
          return mlir::failure();
        if (constantRequestBegin) {
          int64_t requestEnd = 0;
          if (llvm::AddOverflow(*constantRequestBegin, requestLength,
                                requestEnd))
            return mlir::failure();
          const int64_t begin = std::max(*constantRequestBegin, segmentBegin);
          const int64_t end = std::min(requestEnd, segmentEnd);
          if (begin < end &&
              mlir::failed(emitPiece(
                  segment, rewriter.getIndexAttr(begin - segmentBegin),
                  rewriter.getIndexAttr(begin - *constantRequestBegin),
                  end - begin, {})))
            return mlir::failure();
          continue;
        }

        std::optional<int64_t> firstFull = getGridCeil(*grid, segmentBegin);
        const int64_t lastFullCoordinate = segmentEnd - requestLength;
        std::optional<int64_t> lastFull =
            lastFullCoordinate >= grid->lower
                ? getGridFloor(*grid,
                               std::min(lastFullCoordinate, grid->upper - 1))
                : std::nullopt;
        const bool hasFullRange = firstFull.has_value() &&
                                  lastFull.has_value() &&
                                  firstFull.value_or(0) <= lastFull.value_or(0);
        const int64_t firstFullValue = firstFull.value_or(0);
        const int64_t lastFullValue = lastFull.value_or(0);
        if (hasFullRange) {
          std::optional<int64_t> lastGridValue =
              getGridFloor(*grid, grid->upper - 1);
          const bool coversCompleteLoop = firstFullValue == grid->lower &&
                                          lastGridValue &&
                                          lastFullValue == *lastGridValue;
          mlir::Value condition =
              coversCompleteLoop
                  ? mlir::Value{}
                  : createConditionForRange(firstFullValue, lastFullValue);
          mlir::OpFoldResult sourceOffset =
              firstFullValue == lastFullValue && !coversCompleteLoop
                  ? mlir::OpFoldResult(
                        rewriter.getIndexAttr(firstFullValue - segmentBegin))
                  : addConstantOffset(rewriter, slice.getLoc(), grid->induction,
                                      -segmentBegin);
          if (mlir::failed(emitPiece(segment, sourceOffset,
                                     rewriter.getIndexAttr(0), requestLength,
                                     condition)))
            return mlir::failure();
        }

        llvm::SmallVector<int64_t, 2> boundaryBegins;
        if (std::optional<int64_t> begin = getGridFloor(*grid, segmentBegin))
          boundaryBegins.push_back(*begin);
        if (std::optional<int64_t> begin = getGridFloor(*grid, segmentEnd - 1))
          boundaryBegins.push_back(*begin);
        llvm::sort(boundaryBegins);
        boundaryBegins.erase(
            std::unique(boundaryBegins.begin(), boundaryBegins.end()),
            boundaryBegins.end());
        for (int64_t requestBegin : boundaryBegins) {
          if (hasFullRange && requestBegin >= firstFullValue &&
              requestBegin <= lastFullValue)
            continue;
          int64_t requestEnd = 0;
          if (llvm::AddOverflow(requestBegin, requestLength, requestEnd))
            return mlir::failure();
          const int64_t begin = std::max(requestBegin, segmentBegin);
          const int64_t end = std::min(requestEnd, segmentEnd);
          if (begin >= end)
            continue;
          mlir::Value condition = createConditionForExactBegin(requestBegin);
          if (mlir::failed(emitPiece(
                  segment, rewriter.getIndexAttr(begin - segmentBegin),
                  rewriter.getIndexAttr(begin - requestBegin), end - begin,
                  condition)))
            return mlir::failure();
        }
      }
      mlir::FailureOr<mlir::Value> replacement =
          reshapeTile(rewriter, slice.getLoc(), assembled, resultType);
      if (mlir::failed(replacement))
        return mlir::failure();
      rewriter.replaceOp(slice, *replacement);
      ++statistics.tileLocalAssemblies;
    }
    statistics.fusedProducers += fusedProducerOwners.size();
  }
  return mlir::success();
}

mlir::LogicalResult lowerConstantPads(mlir::IRRewriter &rewriter,
                                      TileRegionOp region,
                                      TemporalTilingStatistics &statistics) {
  llvm::SmallVector<mlir::tensor::PadOp, 4> pads;
  region.walk([&](mlir::tensor::PadOp pad) { pads.push_back(pad); });
  for (mlir::tensor::PadOp pad : pads) {
    mlir::Value padding = pad.getConstantPaddingValue();
    auto resultType = pad.getResultType();
    if (!padding)
      continue;
    rewriter.setInsertionPoint(pad);
    llvm::SmallVector<mlir::OpFoldResult, 4> sourceSizes =
        mlir::tensor::getMixedSizes(rewriter, pad.getLoc(), pad.getSource());
    auto low = pad.getMixedLowPad();
    auto high = pad.getMixedHighPad();
    llvm::SmallVector<mlir::Value> dynamicSizes;
    for (int64_t dimension = 0; dimension < resultType.getRank(); ++dimension) {
      if (!resultType.isDynamicDim(dimension))
        continue;
      auto asValue = [&](mlir::OpFoldResult value) {
        return mlir::getValueOrCreateConstantIndexOp(rewriter, pad.getLoc(),
                                                     value);
      };
      mlir::Value size = rewriter.createOrFold<mlir::arith::AddIOp>(
          pad.getLoc(), asValue(sourceSizes[dimension]),
          asValue(low[dimension]));
      dynamicSizes.push_back(rewriter.createOrFold<mlir::arith::AddIOp>(
          pad.getLoc(), size, asValue(high[dimension])));
    }
    mlir::Value empty = rewriter.create<mlir::tensor::EmptyOp>(
        pad.getLoc(), resultType.getShape(), resultType.getElementType(),
        dynamicSizes);
    mlir::Value filled =
        rewriter.create<mlir::linalg::FillOp>(pad.getLoc(), padding, empty)
            .getResult(0);
    llvm::SmallVector<mlir::OpFoldResult, 4> strides(resultType.getRank(),
                                                     rewriter.getIndexAttr(1));
    mlir::Value inserted = rewriter.create<mlir::tensor::InsertSliceOp>(
        pad.getLoc(), pad.getSource(), filled, pad.getMixedLowPad(),
        sourceSizes, strides);
    rewriter.replaceOp(pad, inserted);
    ++statistics.decomposedPads;
  }
  return mlir::success();
}

mlir::LogicalResult
lowerConstantGenerates(mlir::IRRewriter &rewriter, TileRegionOp region,
                       TemporalTilingStatistics &statistics) {
  llvm::SmallVector<mlir::tensor::GenerateOp, 4> operations;
  region.walk([&](mlir::tensor::GenerateOp operation) {
    operations.push_back(operation);
  });
  for (mlir::tensor::GenerateOp operation : operations) {
    auto resultType = operation.getResult().getType();
    auto yield = mlir::dyn_cast<mlir::tensor::YieldOp>(
        operation.getBody().front().getTerminator());
    if (!yield)
      continue;
    mlir::Value value = yield.getValue();
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      if (argument.getOwner() == &operation.getBody().front())
        continue;
    } else if (mlir::Operation *definition = value.getDefiningOp()) {
      if (definition->getParentRegion() == &operation.getRegion())
        continue;
    }
    rewriter.setInsertionPoint(operation);
    mlir::Value empty = rewriter.create<mlir::tensor::EmptyOp>(
        operation.getLoc(), resultType.getShape(), resultType.getElementType(),
        operation.getDynamicExtents());
    mlir::Value filled =
        rewriter.create<mlir::linalg::FillOp>(operation.getLoc(), value, empty)
            .getResult(0);
    rewriter.replaceOp(operation, filled);
    ++statistics.decomposedConstantGenerates;
  }
  return mlir::success();
}

llvm::SmallVector<int64_t, 6>
buildInterchange(const compiler::detail::TemporalScopeDescriptor &descriptor,
                 const compiler::detail::TemporalScopeChoice &choice) {
  llvm::SmallVector<int64_t, 6> interchange;
  interchange.reserve(descriptor.iterationExtents.size());
  for (uint32_t dimension : choice.loopOrder)
    interchange.push_back(dimension);
  for (unsigned dimension = 0; dimension < descriptor.iterationExtents.size();
       ++dimension)
    if (!llvm::is_contained(choice.loopOrder, dimension))
      interchange.push_back(dimension);
  return interchange;
}

mlir::LogicalResult
eraseFusedProducers(mlir::IRRewriter &rewriter,
                    llvm::ArrayRef<mlir::Operation *> fusedProducers) {
  for (mlir::Operation *producer : fusedProducers) {
    bool erasedDeadUser = true;
    while (producer && producer->getBlock() && !producer->use_empty() &&
           erasedDeadUser) {
      erasedDeadUser = false;
      llvm::SmallVector<mlir::Operation *, 4> users;
      for (mlir::Operation *user : producer->getUsers())
        if (!llvm::is_contained(users, user))
          users.push_back(user);
      for (mlir::Operation *user : users)
        if (user->getBlock() && mlir::isOpTriviallyDead(user)) {
          rewriter.eraseOp(user);
          erasedDeadUser = true;
        }
    }
    if (!producer || !producer->getBlock() || !producer->use_empty())
      return mlir::failure();
    rewriter.eraseOp(producer);
  }
  return mlir::success();
}

mlir::LogicalResult specializeRaggedTails(
    mlir::IRRewriter &rewriter,
    const compiler::detail::TemporalScopeDescriptor &descriptor,
    const compiler::detail::TemporalScopeChoice &choice,
    llvm::MutableArrayRef<mlir::LoopLikeOpInterface> loops,
    TemporalTilingStatistics &statistics) {
  if (loops.size() != choice.loopOrder.size())
    return mlir::failure();
  for (size_t reverse = 0; reverse < loops.size(); ++reverse) {
    const size_t position = loops.size() - reverse - 1;
    const unsigned dimension = choice.loopOrder[position];
    const int64_t extent = descriptor.iterationExtents[dimension];
    const int64_t tileSize = choice.iteratorTileSizes[dimension];
    if (extent % tileSize == 0)
      continue;
    auto loop =
        mlir::dyn_cast<mlir::scf::ForOp>(loops[position].getOperation());
    if (!loop)
      return mlir::failure();
    mlir::scf::ForOp partialIteration;
    if (mlir::failed(mlir::scf::peelForLoopAndSimplifyBounds(
            rewriter, loop, partialIteration)) ||
        !partialIteration ||
        mlir::failed(partialIteration.promoteIfSingleIteration(rewriter)))
      return mlir::failure();
    ++statistics.specializedTails;
  }
  return mlir::success();
}

void canonicalizeLoopBoundMinMax(mlir::IRRewriter &rewriter,
                                 TileRegionOp region) {
  llvm::SmallVector<mlir::Operation *, 16> operations;
  region.walk([&](mlir::Operation *operation) {
    if (mlir::isa<mlir::affine::AffineMinOp, mlir::affine::AffineMaxOp>(
            operation))
      operations.push_back(operation);
  });
  for (mlir::Operation *operation : operations)
    if (operation->getBlock())
      (void)mlir::scf::canonicalizeMinMaxOpInLoop(rewriter, operation,
                                                  mlir::scf::matchForLikeLoop);
}

struct JointConsumerTilingResult {
  llvm::SmallVector<mlir::LoopLikeOpInterface, 6> loops;
  unsigned tiledConsumers = 0;
};

mlir::FailureOr<JointConsumerTilingResult> tileJointConsumers(
    mlir::IRRewriter &rewriter, llvm::ArrayRef<mlir::Operation *> consumers,
    llvm::ArrayRef<const compiler::detail::TemporalScopeDescriptor *>
        descriptors,
    llvm::ArrayRef<const compiler::detail::TemporalScopeChoice *> choices) {
  if (consumers.empty() || consumers.size() != descriptors.size() ||
      consumers.size() != choices.size())
    return mlir::failure();
  const auto &descriptor = *descriptors.front();
  const auto &choice = *choices.front();
  for (auto [operation, currentDescriptor, currentChoice] :
       llvm::zip_equal(consumers, descriptors, choices)) {
    auto dps =
        mlir::dyn_cast_if_present<mlir::DestinationStyleOpInterface>(operation);
    if (!operation || !mlir::isa<mlir::TilingInterface>(operation) || !dps ||
        dps.getNumDpsInits() != operation->getNumResults() ||
        currentDescriptor->iterationExtents != descriptor.iterationExtents ||
        currentChoice->iteratorTileSizes != choice.iteratorTileSizes ||
        currentChoice->loopOrder != choice.loopOrder)
      return mlir::failure();
  }

  llvm::SmallVector<mlir::Value, 8> destinations;
  llvm::SmallVector<unsigned, 8> destinationOffsets;
  for (mlir::Operation *operation : consumers) {
    auto tiling = mlir::cast<mlir::TilingInterface>(operation);
    llvm::SmallVector<mlir::Value> current;
    if (mlir::failed(mlir::tensor::getOrCreateDestinations(
            rewriter, operation->getLoc(), tiling, current)) ||
        current.size() != operation->getNumResults())
      return mlir::failure();
    destinationOffsets.push_back(destinations.size());
    llvm::append_range(destinations, current);
  }

  mlir::Location location = consumers.front()->getLoc();
  // Every DPS destination must dominate the common loop. Consumer-local
  // tensor.empty operations are interleaved with the original consumers, so
  // insert immediately before the last consumer and erase all originals only
  // after the replacement loop is complete.
  rewriter.setInsertionPoint(consumers.back());
  llvm::SmallVector<mlir::Value, 6> inductionValues(
      descriptor.iterationExtents.size());
  llvm::SmallVector<mlir::scf::ForOp, 6> loops;
  llvm::SmallVector<mlir::Value, 8> currentDestinations = destinations;
  for (uint32_t dimension : choice.loopOrder) {
    if (dimension >= descriptor.iterationExtents.size() ||
        choice.iteratorTileSizes[dimension] >=
            descriptor.iterationExtents[dimension])
      return mlir::failure();
    mlir::Value lower =
        rewriter.create<mlir::arith::ConstantIndexOp>(location, 0);
    mlir::Value upper = rewriter.create<mlir::arith::ConstantIndexOp>(
        location, descriptor.iterationExtents[dimension]);
    mlir::Value step = rewriter.create<mlir::arith::ConstantIndexOp>(
        location, choice.iteratorTileSizes[dimension]);
    auto loop = rewriter.create<mlir::scf::ForOp>(
        location, lower, upper, step, currentDestinations,
        [](mlir::OpBuilder &builder, mlir::Location nestedLocation, mlir::Value,
           mlir::ValueRange iterArgs) {
          builder.create<mlir::scf::YieldOp>(nestedLocation, iterArgs);
        });
    loops.push_back(loop);
    inductionValues[dimension] = loop.getInductionVar();
    currentDestinations.assign(loop.getRegionIterArgs().begin(),
                               loop.getRegionIterArgs().end());
    rewriter.setInsertionPoint(loop.getBody()->getTerminator());
  }
  if (loops.empty())
    return mlir::failure();

  llvm::SmallVector<mlir::OpFoldResult, 6> tileOffsets;
  llvm::SmallVector<mlir::OpFoldResult, 6> tileSizes;
  for (unsigned dimension = 0; dimension < descriptor.iterationExtents.size();
       ++dimension) {
    const int64_t extent = descriptor.iterationExtents[dimension];
    const int64_t size = choice.iteratorTileSizes[dimension];
    if (size >= extent) {
      tileOffsets.push_back(rewriter.getIndexAttr(0));
      tileSizes.push_back(rewriter.getIndexAttr(extent));
      continue;
    }
    mlir::Value induction = inductionValues[dimension];
    if (!induction)
      return mlir::failure();
    tileOffsets.push_back(induction);
    mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, rewriter.getContext());
    mlir::AffineMap bounded = mlir::AffineMap::get(
        1, 0,
        {mlir::getAffineConstantExpr(size, rewriter.getContext()),
         mlir::getAffineConstantExpr(extent, rewriter.getContext()) - d0},
        rewriter.getContext());
    tileSizes.push_back(mlir::affine::makeComposedFoldedAffineMin(
        rewriter, location, bounded, {induction}));
  }

  llvm::SmallVector<mlir::Value, 8> yielded = currentDestinations;
  for (auto [consumerIndex, operation] : llvm::enumerate(consumers)) {
    auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(operation);
    auto originalTiling = mlir::cast<mlir::TilingInterface>(operation);
    if (!dps || dps.getNumDpsInits() != operation->getNumResults())
      return mlir::failure();
    mlir::IRMapping mapping;
    const unsigned destinationOffset = destinationOffsets[consumerIndex];
    for (unsigned result = 0; result < operation->getNumResults(); ++result)
      mapping.map(dps.getDpsInitOperand(result)->get(),
                  currentDestinations[destinationOffset + result]);
    mlir::Operation *cloned = rewriter.clone(*operation, mapping);
    auto clonedTiling = mlir::cast<mlir::TilingInterface>(cloned);
    mlir::FailureOr<mlir::TilingResult> tiled =
        clonedTiling.getTiledImplementation(rewriter, tileOffsets, tileSizes);
    rewriter.eraseOp(cloned);
    if (mlir::failed(tiled) ||
        tiled->tiledValues.size() != operation->getNumResults())
      return mlir::failure();
    for (auto [resultNumber, tiledValue] :
         llvm::enumerate(tiled->tiledValues)) {
      llvm::SmallVector<mlir::OpFoldResult> resultOffsets;
      llvm::SmallVector<mlir::OpFoldResult> resultSizes;
      if (mlir::failed(originalTiling.getResultTilePosition(
              rewriter, resultNumber, tileOffsets, tileSizes, resultOffsets,
              resultSizes)))
        return mlir::failure();
      llvm::SmallVector<mlir::OpFoldResult, 4> strides(
          resultOffsets.size(), rewriter.getIndexAttr(1));
      yielded[destinationOffset + resultNumber] =
          rewriter.create<mlir::tensor::InsertSliceOp>(
              location, tiledValue,
              currentDestinations[destinationOffset + resultNumber],
              resultOffsets, resultSizes, strides);
    }
  }

  auto innermostYield =
      mlir::cast<mlir::scf::YieldOp>(loops.back().getBody()->getTerminator());
  rewriter.setInsertionPoint(innermostYield);
  rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(innermostYield, yielded);
  for (size_t reverse = 1; reverse < loops.size(); ++reverse) {
    const size_t outerIndex = loops.size() - reverse - 1;
    auto outerYield = mlir::cast<mlir::scf::YieldOp>(
        loops[outerIndex].getBody()->getTerminator());
    rewriter.setInsertionPoint(outerYield);
    rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(
        outerYield, loops[outerIndex + 1].getResults());
  }

  mlir::ValueRange replacements = loops.front().getResults();
  for (auto [consumerIndex, operation] : llvm::enumerate(consumers)) {
    const unsigned offset = destinationOffsets[consumerIndex];
    rewriter.replaceOp(operation,
                       replacements.slice(offset, operation->getNumResults()));
  }
  JointConsumerTilingResult result;
  for (mlir::scf::ForOp loop : loops)
    result.loops.push_back(
        mlir::cast<mlir::LoopLikeOpInterface>(loop.getOperation()));
  result.tiledConsumers = consumers.size();
  return result;
}

mlir::LogicalResult materializeFusionProducer(
    mlir::IRRewriter &rewriter, TileRegionOp region,
    const compiler::detail::TemporalFusion &fusion,
    const compiler::detail::TemporalScopeDescriptor &descriptor,
    const compiler::detail::TemporalScopeChoice &choice,
    const analysis::RectangularTileImage &image,
    llvm::ArrayRef<mlir::LoopLikeOpInterface> loops,
    TemporalTilingStatistics &statistics, std::string &detail,
    ProducerTiling &producerTiling) {
  auto reject = [&](llvm::StringRef message) {
    detail = message.str();
    return mlir::failure();
  };
  if (!fusion.producer || loops.empty() ||
      choice.loopOrder.size() != loops.size())
    return reject("operand fusion no longer matches its current traversal");
  llvm::SmallVector<mlir::tensor::ExtractSliceOp, 4> consumerSlices;
  region.walk([&](mlir::tensor::ExtractSliceOp slice) {
    if (llvm::any_of(fusion.uses, [&](const auto &use) {
          return slice.getSource() == use.consumerValue;
        }))
      consumerSlices.push_back(slice);
  });
  if (consumerSlices.empty()) {
    auto type = mlir::cast<mlir::RankedTensorType>(fusion.producer.getType());
    llvm::SmallVector<int64_t, 6> zero(
        fusion.uses.front().iterationShape.size(), 0);
    auto full =
        fusion.uses.front().iterationToProducer->getExactStaticRectangularImage(
            zero, fusion.uses.front().iterationShape);
    if (full.isExact() && full.domain->sizes == type.getShape() &&
        llvm::all_of(full.domain->offsets,
                     [](int64_t offset) { return offset == 0; }) &&
        llvm::all_of(choice.loopOrder, [&](uint32_t axis) {
          return llvm::is_contained(image.invariantDimensions, axis);
        }))
      // The complete current value already dominates all invariant loops.
      // Any free inner reduction choice is still applied to that same producer.
      return mlir::success();
    return reject("consumer tiling did not expose its exact producer subset");
  }

  llvm::SmallBitVector invariant(descriptor.iterationExtents.size(), false);
  for (uint32_t dimension : image.invariantDimensions) {
    if (dimension >= invariant.size())
      return reject("operand invariant dimension is outside its traversal");
    invariant.set(dimension);
  }
  size_t firstInvariantLoop = loops.size();
  for (auto [position, dimension] : llvm::enumerate(choice.loopOrder))
    if (invariant.test(dimension)) {
      firstInvariantLoop = position;
      break;
    }
  if (firstInvariantLoop < loops.size()) {
    mlir::LoopLikeOpInterface insertionLoop = loops[firstInvariantLoop];
    rewriter.setInsertionPoint(insertionLoop.getOperation());
  } else {
    rewriter.setInsertionPoint(consumerSlices.front());
  }

  llvm::SmallVector<mlir::OpFoldResult, 6> iterationOffsets;
  llvm::SmallVector<mlir::OpFoldResult, 6> iterationSizes;
  mlir::Location location = fusion.producer.getLoc();
  for (unsigned iterator = 0; iterator < descriptor.iterationExtents.size();
       ++iterator) {
    const int64_t extent = descriptor.iterationExtents[iterator];
    const int64_t tileSize = choice.iteratorTileSizes[iterator];
    if (tileSize >= extent || invariant.test(iterator)) {
      iterationOffsets.push_back(rewriter.getIndexAttr(0));
      iterationSizes.push_back(rewriter.getIndexAttr(extent));
      continue;
    }
    auto found = llvm::find(choice.loopOrder, iterator);
    if (found == choice.loopOrder.end())
      return reject("operand dependency has no materialized loop");
    const size_t loopPosition = found - choice.loopOrder.begin();
    if (loopPosition >= firstInvariantLoop)
      return reject("operand tile would be placed above a dependency");
    auto loopInterface = loops[loopPosition];
    auto loop = mlir::dyn_cast<mlir::scf::ForOp>(loopInterface.getOperation());
    if (!loop)
      return reject("operand fusion requires the selected SCF loop");
    auto induction = loop.getInductionVar();
    iterationOffsets.push_back(induction);
    auto d0 = mlir::getAffineDimExpr(0, rewriter.getContext());
    auto bounded = mlir::AffineMap::get(
        1, 0,
        {mlir::getAffineConstantExpr(tileSize, rewriter.getContext()),
         mlir::getAffineConstantExpr(extent, rewriter.getContext()) - d0},
        rewriter.getContext());
    iterationSizes.push_back(mlir::affine::makeComposedFoldedAffineMin(
        rewriter, location, bounded, {induction}));
  }
  llvm::SmallVector<mlir::OpFoldResult, 12> parameters(iterationOffsets.begin(),
                                                       iterationOffsets.end());
  llvm::append_range(parameters, iterationSizes);
  llvm::SmallVector<mlir::OpFoldResult, 4> producerOffsets, producerSizes;
  for (unsigned axis = 0; axis < image.offsetMap.getNumResults(); ++axis) {
    producerOffsets.push_back(mlir::affine::makeComposedFoldedAffineApply(
        rewriter, location, image.offsetMap.getSubMap({axis}), parameters));
    producerSizes.push_back(mlir::affine::makeComposedFoldedAffineApply(
        rewriter, location, image.sizeMap.getSubMap({axis}), iterationSizes));
  }
  auto producerType =
      mlir::dyn_cast<mlir::RankedTensorType>(fusion.producer.getType());
  if (!producerType ||
      producerOffsets.size() != static_cast<size_t>(producerType.getRank()))
    return reject("operand producer rank no longer matches its operand map");
  llvm::SmallVector<mlir::OpFoldResult, 4> strides(producerType.getRank(),
                                                   rewriter.getIndexAttr(1));
  auto request = rewriter.create<mlir::tensor::ExtractSliceOp>(
      location, fusion.producer, producerOffsets, producerSizes, strides);
  mlir::FailureOr<mlir::TilingResult> tiled =
      producerTiling.materialize(rewriter, request, fusion.producer);
  if (mlir::failed(tiled) || tiled->tiledValues.size() != 1) {
    rewriter.eraseOp(request);
    return reject("operand producer rejected its exact relation tile");
  }
  mlir::Value sharedTile = tiled->tiledValues.front();
  rewriter.eraseOp(request);
  for (mlir::tensor::ExtractSliceOp slice : consumerSlices) {
    auto targetType = mlir::dyn_cast<mlir::RankedTensorType>(slice.getType());
    if (!targetType)
      return reject("operand consumer slice is not a ranked tensor");
    mlir::FailureOr<mlir::Value> replacement =
        reshapeTile(rewriter, slice.getLoc(), sharedTile, targetType);
    if (mlir::failed(replacement))
      return reject("operand producer tile does not match its consumer");
    rewriter.replaceOp(slice, *replacement);
  }
  llvm::SmallVector<mlir::Operation *, 8> viewNodes;
  llvm::SmallPtrSet<mlir::Operation *, 8> seenViews;
  for (const auto &use : fusion.uses) {
    mlir::Value value = use.consumerValue;
    while (value != fusion.producer) {
      auto result = mlir::dyn_cast<mlir::OpResult>(value);
      if (!result)
        return reject("fusion lost its transparent source chain");
      auto *operation = result.getOwner();
      if (!seenViews.insert(operation).second)
        break;
      viewNodes.push_back(operation);
      auto indexing = analysis::deriveTensorResultIndexing(result);
      if (!indexing.isExact() || indexing.indexing->operands.size() != 1)
        return reject("fusion lost its transparent source relation");
      value =
          operation->getOperand(indexing.indexing->operands.front().operand);
    }
  }
  llvm::sort(viewNodes,
             [](auto *lhs, auto *rhs) { return rhs->isBeforeInBlock(lhs); });
  for (auto *view : viewNodes) {
    if (!view->use_empty())
      return reject("fusion left an uncaptured view use");
    rewriter.eraseOp(view);
  }
  if (!fusion.producer.use_empty())
    return reject("operand producer still has an unfused current use");
  producerTiling.reductions.erase(fusion.producer.getOwner());
  rewriter.eraseOp(fusion.producer.getOwner());
  ++statistics.fusedProducers;
  return mlir::success();
}

void eraseDeadOperations(mlir::IRRewriter &rewriter, TileRegionOp region) {
  llvm::SmallVector<mlir::Operation *, 32> operations;
  region.walk<mlir::WalkOrder::PostOrder>([&](mlir::Operation *operation) {
    if (operation != region.getOperation() &&
        !operation->hasTrait<mlir::OpTrait::IsTerminator>())
      operations.push_back(operation);
  });
  for (mlir::Operation *operation : operations)
    if (operation->getBlock() && mlir::isOpTriviallyDead(operation))
      rewriter.eraseOp(operation);
}

struct SimplifySlicedUnitCollapse
    : mlir::OpRewritePattern<mlir::tensor::CollapseShapeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::tensor::CollapseShapeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!llvm::any_of(op.getResult().getUsers(), [](mlir::Operation *user) {
          return mlir::isa<mlir::tensor::ExtractSliceOp>(user);
        }))
      return mlir::failure();
    for (llvm::ArrayRef<int64_t> group : op.getReassociationIndices()) {
      unsigned nonUnit = 0;
      for (int64_t dimension : group)
        nonUnit += op.getSrcType().getDimSize(dimension) != 1;
      if (nonUnit > 1)
        return mlir::failure();
    }
    return mlir::success(mlir::succeeded(
        mlir::tensor::simplifyCollapseShapeWithRankReducingExtractSlice(
            op, rewriter)));
  }
};

mlir::LogicalResult
canonicalizeTiledRegion(TileRegionOp region,
                        mlir::RewriterBase::Listener *listener,
                        bool simplifyPackAndUnpack = false) {
  mlir::RewritePatternSet patterns =
      mlir::linalg::getLinalgTilingCanonicalizationPatterns(
          region.getContext());
  patterns.add<SimplifySlicedUnitCollapse>(region.getContext());
  mlir::tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
  mlir::tensor::populateReassociativeReshapeFoldingPatterns(patterns);
  mlir::tensor::populateFoldTensorEmptyPatterns(patterns,
                                                /*foldSingleUseOnly=*/false);
  if (simplifyPackAndUnpack)
    mlir::tensor::populateSimplifyPackAndUnpackPatterns(patterns);
  mlir::GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  config.maxIterations = 10;
  config.scope = &region.getBody();
  config.listener = listener;
  return mlir::applyPatternsAndFoldGreedily(
      region.getBody(), mlir::FrozenRewritePatternSet(std::move(patterns)),
      config);
}

mlir::LogicalResult refineOnlineAttentionStaticTypes(mlir::IRRewriter &rewriter,
                                                     TileRegionOp region) {
  llvm::SmallVector<LinalgExtOnlineAttentionOp, 8> operations;
  region.walk([&](LinalgExtOnlineAttentionOp operation) {
    operations.push_back(operation);
  });
  for (LinalgExtOnlineAttentionOp operation : operations) {
    llvm::SmallVector<mlir::Value, 8> operands(operation->getOperands());
    bool changed = false;
    for (mlir::Value &operand : operands) {
      auto cast = operand.getDefiningOp<mlir::tensor::CastOp>();
      auto sourceType = cast ? mlir::dyn_cast<mlir::RankedTensorType>(
                                   cast.getSource().getType())
                             : mlir::RankedTensorType{};
      auto resultType =
          cast ? mlir::dyn_cast<mlir::RankedTensorType>(cast.getType())
               : mlir::RankedTensorType{};
      if (!cast || !sourceType || !resultType || !sourceType.hasStaticShape() ||
          resultType.hasStaticShape())
        continue;
      operand = cast.getSource();
      changed = true;
    }
    if (!changed)
      continue;
    const unsigned initStart = operation.getMask() ? 5 : 4;
    if (operands.size() != initStart + 3)
      return mlir::failure();
    llvm::SmallVector<mlir::Type, 3> resultTypes{
        operands[initStart].getType(), operands[initStart + 1].getType(),
        operands[initStart + 2].getType()};
    rewriter.setInsertionPoint(operation);
    mlir::Operation *replacement =
        mlir::clone(rewriter, operation.getOperation(), resultTypes, operands);
    rewriter.replaceOp(operation, replacement->getResults());
  }
  return mlir::success();
}

mlir::LogicalResult refinePackUnPackStaticTypes(mlir::IRRewriter &rewriter,
                                                TileRegionOp region) {
  llvm::SmallVector<mlir::Operation *, 8> operations;
  region.walk([&](mlir::Operation *operation) {
    if (mlir::isa<mlir::tensor::PackOp, mlir::tensor::UnPackOp>(operation))
      operations.push_back(operation);
  });
  for (mlir::Operation *operation : operations) {
    auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(operation);
    if (!dps || dps.getNumDpsInits() != operation->getNumResults())
      return mlir::failure();
    llvm::SmallVector<mlir::Value, 8> operands(operation->getOperands());
    bool changed = false;
    for (mlir::Value &operand : operands) {
      auto cast = operand.getDefiningOp<mlir::tensor::CastOp>();
      auto sourceType = cast ? mlir::dyn_cast<mlir::RankedTensorType>(
                                   cast.getSource().getType())
                             : mlir::RankedTensorType{};
      auto resultType =
          cast ? mlir::dyn_cast<mlir::RankedTensorType>(cast.getType())
               : mlir::RankedTensorType{};
      if (!cast || !sourceType || !resultType || !sourceType.hasStaticShape() ||
          resultType.hasStaticShape())
        continue;
      operand = cast.getSource();
      changed = true;
    }
    llvm::SmallVector<mlir::Type, 2> resultTypes;
    for (unsigned index = 0; index < dps.getNumDpsInits(); ++index) {
      mlir::OpResult result = operation->getResult(index);
      mlir::OpOperand *init = dps.getDpsInitOperand(index);
      mlir::Type resultType = result.getType();
      for (mlir::Operation *user : result.getUsers())
        if (auto cast = mlir::dyn_cast<mlir::tensor::CastOp>(user)) {
          auto castType =
              mlir::dyn_cast<mlir::RankedTensorType>(cast.getType());
          if (castType && castType.hasStaticShape()) {
            resultType = castType;
            break;
          }
        }
      mlir::Value &initValue = operands[init->getOperandNumber()];
      auto initType =
          mlir::dyn_cast<mlir::RankedTensorType>(initValue.getType());
      auto desiredType = mlir::dyn_cast<mlir::RankedTensorType>(resultType);
      if (desiredType && !desiredType.hasStaticShape() && initType &&
          initType.hasStaticShape())
        desiredType = initType;
      if (!desiredType || !initType)
        return mlir::failure();
      resultTypes.push_back(desiredType);
      if (initType == desiredType)
        continue;
      if (!mlir::tensor::CastOp::areCastCompatible(initType, desiredType))
        return mlir::failure();
      rewriter.setInsertionPoint(operation);
      initValue = rewriter.create<mlir::tensor::CastOp>(operation->getLoc(),
                                                        desiredType, initValue);
      changed = true;
    }
    if (!changed && llvm::equal(operation->getResultTypes(), resultTypes))
      continue;
    // Pack/UnPack result types are immutable. Replace the one current
    // occurrence after peeling has proved static slice types; the old op is
    // erased immediately and no extra traversal remains.
    rewriter.setInsertionPoint(operation);
    mlir::Operation *replacement =
        mlir::clone(rewriter, operation, resultTypes, operands);
    rewriter.replaceOp(operation, replacement->getResults());
  }
  return mlir::success();
}

mlir::LogicalResult refineReshapeStaticTypes(mlir::IRRewriter &rewriter,
                                             TileRegionOp region) {
  llvm::SmallVector<mlir::Operation *, 8> reshapes;
  region.walk([&](mlir::Operation *operation) {
    if (mlir::isa<mlir::tensor::ExpandShapeOp, mlir::tensor::CollapseShapeOp>(
            operation))
      reshapes.push_back(operation);
  });
  for (auto *operation : reshapes) {
    auto source = operation->getOperand(0);
    auto sourceType = mlir::cast<mlir::RankedTensorType>(source.getType());
    auto targetType =
        mlir::cast<mlir::RankedTensorType>(operation->getResult(0).getType());
    if (!sourceType.hasStaticShape() || targetType.hasStaticShape())
      continue;
    llvm::SmallVector<int64_t, 6> shape(targetType.getShape());
    if (auto expand = mlir::dyn_cast<mlir::tensor::ExpandShapeOp>(operation)) {
      for (auto [axis, group] :
           llvm::enumerate(expand.getReassociationIndices())) {
        int64_t product = 1;
        std::optional<int64_t> dynamic;
        for (int64_t member : group) {
          if (mlir::ShapedType::isDynamic(shape[member])) {
            if (dynamic)
              return mlir::failure();
            dynamic = member;
          } else if (shape[member] <= 0 ||
                     llvm::MulOverflow(product, shape[member], product)) {
            return mlir::failure();
          }
        }
        if (dynamic) {
          if (sourceType.getDimSize(axis) % product)
            return mlir::failure();
          shape[*dynamic] = sourceType.getDimSize(axis) / product;
        }
      }
      auto precise = mlir::RankedTensorType::get(
          shape, targetType.getElementType(), targetType.getEncoding());
      rewriter.setInsertionPoint(operation);
      rewriter.replaceOpWithNewOp<mlir::tensor::ExpandShapeOp>(
          operation, precise, source, expand.getReassociationIndices());
    } else {
      auto collapse = mlir::cast<mlir::tensor::CollapseShapeOp>(operation);
      for (auto [axis, group] :
           llvm::enumerate(collapse.getReassociationIndices())) {
        int64_t product = 1;
        for (int64_t member : group)
          if (llvm::MulOverflow(product, sourceType.getDimSize(member),
                                product))
            return mlir::failure();
        shape[axis] = product;
      }
      auto precise = mlir::RankedTensorType::get(
          shape, targetType.getElementType(), targetType.getEncoding());
      rewriter.setInsertionPoint(operation);
      rewriter.replaceOpWithNewOp<mlir::tensor::CollapseShapeOp>(
          operation, precise, source, collapse.getReassociationIndices());
    }
  }
  return mlir::success();
}

mlir::LogicalResult refineLinalgStaticTypes(mlir::IRRewriter &rewriter,
                                            TileRegionOp region) {
  llvm::SmallVector<mlir::linalg::LinalgOp, 16> operations;
  region.walk([&](mlir::linalg::LinalgOp operation) {
    operations.push_back(operation);
  });
  for (mlir::linalg::LinalgOp operation : operations) {
    if (!operation.hasPureTensorSemantics())
      continue;
    llvm::SmallVector<mlir::Value, 8> operands(operation->getOperands());
    bool changed = false;
    for (mlir::Value &operand : operands) {
      auto cast = operand.getDefiningOp<mlir::tensor::CastOp>();
      auto sourceType = cast ? mlir::dyn_cast<mlir::RankedTensorType>(
                                   cast.getSource().getType())
                             : mlir::RankedTensorType{};
      auto resultType =
          cast ? mlir::dyn_cast<mlir::RankedTensorType>(cast.getType())
               : mlir::RankedTensorType{};
      if (!cast || !sourceType || !resultType || !sourceType.hasStaticShape() ||
          resultType.hasStaticShape())
        continue;
      operand = cast.getSource();
      changed = true;
    }
    if (!changed)
      continue;
    llvm::SmallVector<mlir::Type, 4> resultTypes;
    for (unsigned index = 0; index < operation.getNumDpsInits(); ++index) {
      mlir::OpOperand *init = operation.getDpsInitOperand(index);
      resultTypes.push_back(operands[init->getOperandNumber()].getType());
    }
    rewriter.setInsertionPoint(operation);
    mlir::Operation *replacement =
        mlir::clone(rewriter, operation.getOperation(), resultTypes, operands);
    rewriter.replaceOp(operation, replacement->getResults());
  }
  return refineReshapeStaticTypes(rewriter, region);
}

mlir::FailureOr<mlir::TilingResult>
ProducerTiling::materialize(mlir::IRRewriter &rewriter,
                            mlir::tensor::ExtractSliceOp slice,
                            mlir::OpResult producer) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  auto selected = reductions.find(producer.getOwner());
  auto tiled = mlir::tensor::replaceExtractSliceWithTiledProducer(
      rewriter, slice, producer);
  if (mlir::failed(tiled) || selected == reductions.end())
    return tiled;
  selected->second.materialized = true;
  if (selected->second.choice.loopOrder.empty())
    return tiled;
  if (tiled->tiledValues.empty())
    return mlir::failure();
  auto operation = mlir::dyn_cast_or_null<mlir::TilingInterface>(
      tiled->tiledValues.front().getDefiningOp());
  if (!operation || llvm::any_of(tiled->tiledValues, [&](mlir::Value value) {
        return value.getDefiningOp() != operation.getOperation();
      }))
    return mlir::failure();
  const auto &descriptor = selected->second.descriptor;
  const auto &choice = selected->second.choice;
  llvm::SmallVector<mlir::OpFoldResult, 6> sizes;
  for (auto [extent, size] :
       llvm::zip_equal(descriptor.iterationExtents, choice.iteratorTileSizes))
    sizes.push_back(rewriter.getIndexAttr(size < extent ? size : 0));
  mlir::scf::SCFTilingOptions options;
  options.setTileSizes(sizes);
  options.setInterchange(buildInterchange(descriptor, choice));
  rewriter.setInsertionPoint(operation);
  auto inner = mlir::scf::tileUsingSCF(rewriter, operation, options);
  if (mlir::failed(inner))
    return mlir::failure();
  for (mlir::Value &value : tiled->tiledValues) {
    auto result = mlir::cast<mlir::OpResult>(value);
    value = inner->replacements[result.getResultNumber()];
  }
  rewriter.replaceOp(operation, inner->replacements);
  tiled->tiledOps = inner->tiledOps;
  llvm::append_range(tiled->generatedSlices, inner->generatedSlices);
  statistics.loops += inner->loops.size();
  // Tail specialization runs on current loops after outer fusion is complete,
  // so clones in an output tail receive exactly the same inner treatment.
  return tiled;
}

mlir::LogicalResult
specializeCurrentRaggedLoops(mlir::IRRewriter &rewriter, TileRegionOp region,
                             TemporalTilingStatistics &statistics) {
  llvm::SmallVector<mlir::scf::ForOp> loops;
  region.walk<mlir::WalkOrder::PostOrder>(
      [&](mlir::scf::ForOp loop) { loops.push_back(loop); });
  for (auto loop : loops) {
    if (loop.getInitArgs().empty() ||
        llvm::any_of(loop.getInitArgs(), [](mlir::Value value) {
          return !mlir::isa<mlir::TensorType>(value.getType());
        }))
      continue;
    auto lower = mlir::getConstantIntValue(loop.getLowerBound());
    auto upper = mlir::getConstantIntValue(loop.getUpperBound());
    auto step = mlir::getConstantIntValue(loop.getStep());
    int64_t extent = 0;
    if (!lower || !upper || !step || *step <= 0 ||
        llvm::SubOverflow(*upper, *lower, extent) || extent <= 0 ||
        extent % *step == 0)
      continue;
    mlir::scf::ForOp tail;
    if (mlir::failed(
            mlir::scf::peelForLoopAndSimplifyBounds(rewriter, loop, tail)) ||
        !tail || mlir::failed(tail.promoteIfSingleIteration(rewriter)))
      return mlir::failure();
    ++statistics.specializedTails;
  }
  return mlir::success();
}

mlir::LogicalResult composeCurrentSlices(mlir::IRRewriter &rewriter,
                                         TileRegionOp region) {
  llvm::SmallVector<mlir::Operation *> slices;
  region.walk(
      [&](mlir::tensor::ExtractSliceOp slice) { slices.push_back(slice); });
  mlir::RewritePatternSet patterns(rewriter.getContext());
  mlir::tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
  mlir::GreedyRewriteConfig config;
  config.scope = &region.getBody();
  config.strictMode = mlir::GreedyRewriteStrictness::ExistingAndNewOps;
  config.listener = mlir::dyn_cast_or_null<mlir::RewriterBase::Listener>(
      rewriter.getListener());
  return mlir::applyOpPatternsAndFold(
      slices, mlir::FrozenRewritePatternSet(std::move(patterns)), config);
}

mlir::LogicalResult materializeInitializerTiles(mlir::IRRewriter &rewriter,
                                                TileRegionOp region) {
  llvm::SmallVector<mlir::tensor::ExtractSliceOp> slices;
  region.walk([&](mlir::tensor::ExtractSliceOp slice) {
    if (slice.getSource().getDefiningOp<mlir::linalg::FillOp>())
      slices.push_back(slice);
  });
  for (auto slice : slices) {
    auto fill = slice.getSource().getDefiningOp<mlir::linalg::FillOp>();
    auto type = slice.getType();
    if (!type.hasStaticShape())
      return mlir::failure();
    rewriter.setInsertionPoint(slice);
    auto empty = rewriter.create<mlir::tensor::EmptyOp>(
        slice.getLoc(), type.getShape(), type.getElementType(),
        type.getEncoding());
    auto local = rewriter.create<mlir::linalg::FillOp>(
        slice.getLoc(), fill.getInputs()[0], empty.getResult());
    rewriter.replaceOp(slice, local.getResults());
  }
  return mlir::success();
}

struct StateConsumerRequest {
  mlir::TilingInterface producer;
  mlir::linalg::LinalgOp consumer;
  compiler::detail::TemporalScopeDescriptor producerDescriptor;
  compiler::detail::TemporalScopeDescriptor consumerDescriptor;
  compiler::detail::TemporalScopeChoice producerChoice;
  compiler::detail::TemporalScopeChoice consumerChoice;
  llvm::SmallVector<mlir::AffineMap, 3> resultMaps;
};

std::optional<StateConsumerRequest> getStateConsumerRequest(
    mlir::TilingInterface producer,
    const llvm::DenseMap<mlir::Operation *,
                         const compiler::detail::TemporalScopeDescriptor *>
        &descriptors,
    const llvm::DenseMap<mlir::Operation *,
                         const compiler::detail::TemporalScopeChoice *>
        &choices) {
  auto producerDps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(
      producer.getOperation());
  if (!producerDps || producer->getNumResults() < 2 ||
      !mlir::isMemoryEffectFree(producer))
    return std::nullopt;
  llvm::SmallVector<mlir::AffineMap, 3> resultMaps;
  for (auto result : producer->getResults()) {
    auto map = analysis::getStructuredResultMap(result);
    if (mlir::failed(map))
      return std::nullopt;
    resultMaps.push_back(*map);
  }
  mlir::Operation *soleConsumer = nullptr;
  for (mlir::Operation *user : producer->getUsers()) {
    if (soleConsumer && soleConsumer != user)
      return std::nullopt;
    soleConsumer = user;
  }
  auto consumer = mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(soleConsumer);
  if (!consumer || !consumer.hasPureTensorSemantics() ||
      consumer.getNumReductionLoops() != 0 ||
      consumer->getBlock() != producer->getBlock() ||
      !producer->isBeforeInBlock(consumer) || !choices.count(consumer) ||
      !choices.count(producer))
    return std::nullopt;
  for (unsigned index = 0; index < consumer.getNumDpsInits(); ++index)
    if (consumer.payloadUsesValueFromOperand(consumer.getDpsInitOperand(index)))
      return std::nullopt;
  for (mlir::Value init : producerDps.getDpsInits()) {
    auto fill = init.getDefiningOp<mlir::linalg::FillOp>();
    if (!fill)
      return std::nullopt;
  }
  StateConsumerRequest request{producer,
                               consumer,
                               *descriptors.lookup(producer),
                               *descriptors.lookup(consumer),
                               *choices.lookup(producer),
                               *choices.lookup(consumer),
                               std::move(resultMaps)};
  const auto &producerExtents = request.producerDescriptor.iterationExtents;
  const auto &consumerExtents = request.consumerDescriptor.iterationExtents;
  llvm::SmallVector<int, 6> producerToConsumer(producerExtents.size(), -1);
  llvm::SmallVector<int, 6> consumerToProducer(consumerExtents.size(), -1);
  for (mlir::OpOperand *input : consumer.getDpsInputOperands()) {
    auto result = mlir::dyn_cast<mlir::OpResult>(input->get());
    if (!result || result.getOwner() != producer)
      continue;
    mlir::AffineMap from = request.resultMaps[result.getResultNumber()];
    mlir::AffineMap to = consumer.getMatchingIndexingMap(input);
    if (!from.isProjectedPermutation() || !to.isProjectedPermutation() ||
        from.getNumResults() != to.getNumResults())
      return std::nullopt;
    for (auto [fromExpr, toExpr] :
         llvm::zip_equal(from.getResults(), to.getResults())) {
      unsigned p = mlir::cast<mlir::AffineDimExpr>(fromExpr).getPosition();
      unsigned c = mlir::cast<mlir::AffineDimExpr>(toExpr).getPosition();
      if ((producerToConsumer[p] != -1 &&
           producerToConsumer[p] != static_cast<int>(c)) ||
          (consumerToProducer[c] != -1 &&
           consumerToProducer[c] != static_cast<int>(p)) ||
          producerExtents[p] != consumerExtents[c])
        return std::nullopt;
      producerToConsumer[p] = c;
      consumerToProducer[c] = p;
    }
  }
  auto iterators = producer.getLoopIteratorTypes();
  for (unsigned c : request.consumerChoice.loopOrder) {
    int p = consumerToProducer[c];
    if (p < 0 || iterators[p] != mlir::utils::IteratorType::parallel ||
        llvm::any_of(
            request.resultMaps,
            [&](mlir::AffineMap map) { return !map.isFunctionOfDim(p); }) ||
        request.producerDescriptor.iteratorCapabilities[p] !=
            compiler::detail::IteratorTilingCapability::Tileable)
      return std::nullopt;
  }
  llvm::SmallVector<uint32_t, 4> parallelOrder;
  bool sawReduction = false;
  for (unsigned p : request.producerChoice.loopOrder) {
    if (iterators[p] != mlir::utils::IteratorType::parallel) {
      sawReduction = true;
      continue;
    }
    if (sawReduction || producerToConsumer[p] < 0)
      return std::nullopt;
    parallelOrder.push_back(producerToConsumer[p]);
  }
  if (parallelOrder.empty() ||
      parallelOrder != request.consumerChoice.loopOrder)
    return std::nullopt;
  for (auto [p, c] : llvm::enumerate(producerToConsumer))
    if (c >= 0 && request.producerChoice.iteratorTileSizes[p] !=
                      request.consumerChoice.iteratorTileSizes[c])
      return std::nullopt;
  return request;
}

mlir::LogicalResult tileStateConsumer(mlir::IRRewriter &rewriter,
                                      TileRegionOp region,
                                      StateConsumerRequest request,
                                      TemporalTilingStatistics &statistics) {
  auto producer = request.producer;
  auto consumer = request.consumer;
  // The consumer payload does not read its DPS init. Retaining a state result
  // here would keep a full-size initialization copy outside the output loop.
  rewriter.setInsertionPoint(consumer);
  for (unsigned index = 0; index < consumer.getNumDpsInits(); ++index) {
    mlir::OpOperand *init = consumer.getDpsInitOperand(index);
    auto type = mlir::cast<mlir::RankedTensorType>(init->get().getType());
    auto empty = rewriter.create<mlir::tensor::EmptyOp>(
        consumer.getLoc(), type.getShape(), type.getElementType(),
        type.getEncoding());
    rewriter.modifyOpInPlace(consumer, [&] { init->set(empty); });
  }
  llvm::SmallVector<mlir::OpFoldResult, 6> sizes;
  for (auto [extent, size] :
       llvm::zip_equal(request.consumerDescriptor.iterationExtents,
                       request.consumerChoice.iteratorTileSizes))
    sizes.push_back(rewriter.getIndexAttr(size < extent ? size : 0));
  mlir::scf::SCFTilingOptions options;
  options.setTileSizes(sizes);
  options.setInterchange(
      buildInterchange(request.consumerDescriptor, request.consumerChoice));
  auto outer = mlir::scf::tileUsingSCF(
      rewriter, mlir::cast<mlir::TilingInterface>(consumer.getOperation()),
      options);
  if (mlir::failed(outer))
    return mlir::failure();
  rewriter.replaceOp(consumer, outer->replacements);
  statistics.loops += outer->loops.size();
  if (mlir::failed(specializeRaggedTails(rewriter, request.consumerDescriptor,
                                         request.consumerChoice, outer->loops,
                                         statistics)))
    return mlir::failure();
  canonicalizeLoopBoundMinMax(rewriter, region);

  // Find the actual main/tail consumers through the still-live producer SSA.
  // No clone ordering, symbols or output inventory is used to recover them.
  llvm::SmallVector<mlir::linalg::LinalgOp, 4> consumers;
  region.walk([&](mlir::linalg::LinalgOp generic) {
    for (mlir::Value input : generic.getDpsInputs()) {
      auto slice = input.getDefiningOp<mlir::tensor::ExtractSliceOp>();
      if (slice && slice.getSource().getDefiningOp() == producer) {
        consumers.push_back(generic);
        break;
      }
    }
  });
  if (consumers.empty())
    return mlir::failure();
  for (mlir::linalg::LinalgOp current : consumers) {
    llvm::SmallVector<mlir::OpFoldResult, 6> offsets(
        request.producerDescriptor.iterationExtents.size(),
        rewriter.getIndexAttr(0));
    llvm::SmallVector<mlir::OpFoldResult, 6> tileSizes;
    for (int64_t extent : request.producerDescriptor.iterationExtents)
      tileSizes.push_back(rewriter.getIndexAttr(extent));
    llvm::SmallVector<mlir::tensor::ExtractSliceOp, 3> stateSlices;
    for (mlir::Value input : current.getDpsInputs()) {
      auto slice = input.getDefiningOp<mlir::tensor::ExtractSliceOp>();
      auto result = slice ? mlir::dyn_cast<mlir::OpResult>(slice.getSource())
                          : mlir::OpResult{};
      if (!result || result.getOwner() != producer)
        continue;
      if (!llvm::is_contained(stateSlices, slice))
        stateSlices.push_back(slice);
      auto map = request.resultMaps[result.getResultNumber()];
      for (auto [axis, expression] : llvm::enumerate(map.getResults())) {
        unsigned dimension =
            mlir::cast<mlir::AffineDimExpr>(expression).getPosition();
        auto size = resolveStaticIndex(slice.getMixedSizes()[axis]);
        if (!size || *size <= 0)
          return mlir::failure();
        offsets[dimension] = slice.getMixedOffsets()[axis];
        tileSizes[dimension] = rewriter.getIndexAttr(*size);
      }
    }
    rewriter.setInsertionPoint(current);
    auto tiled = producer.getTiledImplementation(rewriter, offsets, tileSizes);
    if (mlir::failed(tiled) || tiled->tiledOps.size() != 1 ||
        tiled->tiledValues.size() != producer->getNumResults())
      return mlir::failure();
    auto tiledProducer =
        mlir::cast<mlir::TilingInterface>(tiled->tiledOps.front());
    auto dps = mlir::cast<mlir::DestinationStyleOpInterface>(
        tiledProducer.getOperation());
    auto originalDps =
        mlir::cast<mlir::DestinationStyleOpInterface>(producer.getOperation());
    rewriter.setInsertionPoint(tiledProducer);
    for (unsigned index = 0; index < dps.getNumDpsInits(); ++index) {
      auto fill = originalDps.getDpsInitOperand(index)
                      ->get()
                      .getDefiningOp<mlir::linalg::FillOp>();
      auto type = mlir::cast<mlir::RankedTensorType>(
          dps.getDpsInitOperand(index)->get().getType());
      if (!type.hasStaticShape())
        return mlir::failure();
      auto empty = rewriter.create<mlir::tensor::EmptyOp>(
          tiledProducer->getLoc(), type.getShape(), type.getElementType(),
          type.getEncoding());
      auto local = rewriter.create<mlir::linalg::FillOp>(
          tiledProducer->getLoc(), fill.getInputs()[0], empty.getResult());
      rewriter.modifyOpInPlace(tiledProducer, [&] {
        dps.getDpsInitOperand(index)->set(local.getResult(0));
      });
    }
    rewriter.modifyOpInPlace(current, [&] {
      for (mlir::OpOperand *input : current.getDpsInputOperands()) {
        auto slice = input->get().getDefiningOp<mlir::tensor::ExtractSliceOp>();
        auto result = slice ? mlir::dyn_cast<mlir::OpResult>(slice.getSource())
                            : mlir::OpResult{};
        if (result && result.getOwner() == producer)
          input->set(tiledProducer->getResult(result.getResultNumber()));
      }
    });
    for (auto slice : stateSlices)
      if (slice->use_empty())
        rewriter.eraseOp(slice);
    for (mlir::Operation *slice : tiled->generatedSlices)
      if (slice->use_empty())
        rewriter.eraseOp(slice);
    auto reductionChoice = request.producerChoice;
    reductionChoice.loopOrder.clear();
    llvm::SmallVector<mlir::OpFoldResult, 6> reductionSizes;
    auto iterators = producer.getLoopIteratorTypes();
    for (auto [dimension, extent] :
         llvm::enumerate(request.producerDescriptor.iterationExtents)) {
      if (iterators[dimension] == mlir::utils::IteratorType::parallel)
        reductionChoice.iteratorTileSizes[dimension] = extent;
      int64_t size = reductionChoice.iteratorTileSizes[dimension];
      reductionSizes.push_back(rewriter.getIndexAttr(size < extent ? size : 0));
    }
    for (unsigned dimension : request.producerChoice.loopOrder)
      if (iterators[dimension] != mlir::utils::IteratorType::parallel)
        reductionChoice.loopOrder.push_back(dimension);
    if (!reductionChoice.loopOrder.empty()) {
      mlir::scf::SCFTilingOptions reductionOptions;
      reductionOptions.setTileSizes(reductionSizes);
      reductionOptions.setInterchange(
          buildInterchange(request.producerDescriptor, reductionChoice));
      rewriter.setInsertionPoint(tiledProducer);
      auto inner =
          mlir::scf::tileUsingSCF(rewriter, tiledProducer, reductionOptions);
      if (mlir::failed(inner))
        return mlir::failure();
      rewriter.replaceOp(tiledProducer, inner->replacements);
      statistics.loops += inner->loops.size();
      if (mlir::failed(
              specializeRaggedTails(rewriter, request.producerDescriptor,
                                    reductionChoice, inner->loops, statistics)))
        return mlir::failure();
    }
  }
  // Other selected traversals still own live operation handles. Erase only the
  // producer consumed here; the final canonicalization removes its dead inits.
  if (!producer->use_empty())
    return mlir::failure();
  rewriter.eraseOp(producer);
  statistics.tiledTraversals += 2;
  ++statistics.fusedProducers;
  return mlir::success();
}

} // namespace

mlir::FailureOr<TemporalTilingStatistics>
applyTemporalTiling(const compiler::detail::TemporalDomain &domain,
                    const compiler::detail::TemporalChoice &choice,
                    StructuredMaterializationRelations &relations,
                    TemporalTilingFailure *failure) {
  if (failure)
    *failure = {};
  TileRegionOp region = domain.getRegion();
  if (!region || !domain.contains(choice) || mlir::failed(mlir::verify(region)))
    return fail<TemporalTilingStatistics>(
        failure, TemporalTilingFailureKind::BrokenContract,
        "temporal apply requires one live choice from an unchanged TileRegion");
  if (mlir::failed(compiler::detail::checkStructuredBufferRelationsCurrent(
          region->getParentOfType<mlir::ModuleOp>(), relations)))
    return fail<TemporalTilingStatistics>(
        failure, TemporalTilingFailureKind::BrokenContract,
        "temporal apply received stale structural endpoint relations");

  llvm::DenseMap<mlir::Operation *,
                 const compiler::detail::TemporalScopeDescriptor *>
      descriptors;
  for (const auto &descriptor : domain.getScopeDescriptors(choice.kind))
    descriptors.try_emplace(descriptor.operation, &descriptor);
  for (const auto &scope : choice.scopes) {
    auto descriptor = descriptors.find(scope.operation);
    if (!scope.operation || descriptor == descriptors.end() ||
        scope.operation->getParentOfType<TileRegionOp>() != region ||
        scope.operation->getBlock() != &region.getBody().front())
      return fail<TemporalTilingStatistics>(
          failure, TemporalTilingFailureKind::BrokenContract,
          "temporal choice contains a stale traversal operation");
  }

  bool hasAnyActiveDimension = false;
  for (const auto &scope : choice.scopes) {
    const auto &descriptor = *descriptors.find(scope.operation)->second;
    hasAnyActiveDimension |= llvm::any_of(
        llvm::zip_equal(descriptor.iterationExtents, scope.iteratorTileSizes),
        [](auto values) { return std::get<1>(values) < std::get<0>(values); });
  }
  if (!hasAnyActiveDimension)
    return TemporalTilingStatistics{};

  compiler::detail::StructuredBufferReplacementListener listener(relations);
  LiveFusionSources liveSources(&listener);
  mlir::IRRewriter rewriter(region.getContext(), &liveSources);
  TemporalTilingStatistics statistics;
  if (choice.kind == compiler::detail::TemporalTraversalKind::Joint) {
    auto proofs = collectExactFusionEdges(region, failure);
    if (mlir::failed(proofs))
      return mlir::failure();
    auto addSource = [&](mlir::Operation *operation) {
      auto descriptor = descriptors.find(operation);
      if (descriptor == descriptors.end() ||
          descriptor->second->role ==
              compiler::detail::TemporalScopeRole::FusedReduction)
        liveSources.sources.try_emplace(operation, false);
    };
    for (auto value : proofs->directEdges)
      addSource(value.getDefiningOp());
    for (const auto &path : proofs->viewPaths) {
      addSource(path.producer.getOwner());
      if (liveSources.sources.count(path.producer.getOwner())) {
        mlir::Value view = path.uses.front().operand->get();
        llvm::SmallVector<mlir::Operation *, 4> dependencies;
        llvm::SmallVector<mlir::Value, 4> pending{view};
        llvm::SmallPtrSet<mlir::Operation *, 8> seen;
        while (!pending.empty()) {
          auto *operation = pending.pop_back_val().getDefiningOp();
          if (!operation || !seen.insert(operation).second)
            continue;
          dependencies.push_back(operation);
          if (operation != path.producer.getOwner())
            llvm::append_range(pending, operation->getOperands());
        }
        liveSources.views.try_emplace(
            view.getDefiningOp(),
            LiveFusionSources::View{
                path.producer, view, path.uses.front().producerDimensions,
                (path.uses.front().representation ==
                         compiler::detail::TemporalTileRepresentation::
                             ReshapePieces
                     ? path.uses.front().viewToProducer
                     : std::nullopt),
                std::move(dependencies)});
      }
    }
  }
  llvm::DenseMap<mlir::Operation *,
                 const compiler::detail::TemporalScopeChoice *>
      choicesByOperation;
  for (const auto &scope : choice.scopes)
    choicesByOperation.try_emplace(scope.operation, &scope);
  llvm::DenseMap<mlir::Operation *,
                 llvm::SmallVector<const compiler::detail::TemporalFusion *, 2>>
      operandsByConsumer;
  if (choice.kind == compiler::detail::TemporalTraversalKind::Joint)
    for (const auto &fusion : domain.getFusions()) {
      if (!llvm::any_of(fusion.uses, [](const auto &use) {
            return use.representation ==
                   compiler::detail::TemporalTileRepresentation::
                       RectangularImage;
          }))
        continue;
      for (const auto &use : fusion.uses)
        operandsByConsumer[use.operand->getOwner()].push_back(&fusion);
    }
  ProducerTiling producerTiling(statistics);
  for (const auto &scope : choice.scopes) {
    const auto &descriptor = *descriptors.lookup(scope.operation);
    if (descriptor.role == compiler::detail::TemporalScopeRole::FusedReduction)
      producerTiling.reductions.try_emplace(
          scope.operation, ProducerTiling::Reduction{descriptor, scope});
  }
  llvm::SmallPtrSet<mlir::Operation *, 16> groupedOperations;
  if (choice.kind == compiler::detail::TemporalTraversalKind::Joint)
    for (const auto &group : domain.getFusions()) {
      if (group.uses.size() < 2)
        continue;
      groupedOperations.insert(group.producer.getOwner());
      for (const auto &use : group.uses)
        groupedOperations.insert(use.operand->getOwner());
    }
  llvm::SmallPtrSet<mlir::Operation *, 8> commonLoopConsumers;
  for (const auto &scope : choice.scopes) {
    if (commonLoopConsumers.contains(scope.operation))
      continue;
    if (groupedOperations.contains(scope.operation))
      continue;
    auto producer = mlir::dyn_cast<mlir::TilingInterface>(scope.operation);
    if (!producer || producer->getNumResults() < 2)
      continue;
    auto request =
        getStateConsumerRequest(producer, descriptors, choicesByOperation);
    if (!request || groupedOperations.contains(request->consumer))
      continue;
    commonLoopConsumers.insert(request->producer);
    commonLoopConsumers.insert(request->consumer);
    if (mlir::failed(tileStateConsumer(rewriter, region, *request, statistics)))
      return fail<TemporalTilingStatistics>(
          failure, TemporalTilingFailureKind::CompilerFailure,
          "coupled-state consumer could not form its selected output "
          "traversal");
  }
  if (choice.kind == compiler::detail::TemporalTraversalKind::Joint) {
    std::string manualFailure;
    auto hasActiveDimension = [&](mlir::Operation *operation) {
      const auto *descriptor = descriptors.lookup(operation);
      const auto *scopeChoice = choicesByOperation.lookup(operation);
      return descriptor && scopeChoice &&
             llvm::any_of(llvm::zip_equal(descriptor->iterationExtents,
                                          scopeChoice->iteratorTileSizes),
                          [](auto values) {
                            return std::get<1>(values) < std::get<0>(values);
                          });
    };
    auto tileManualConsumers = [&](llvm::ArrayRef<mlir::Operation *> consumers)
        -> mlir::LogicalResult {
      llvm::SmallVector<const compiler::detail::TemporalScopeDescriptor *, 8>
          groupDescriptors;
      llvm::SmallVector<const compiler::detail::TemporalScopeChoice *, 8>
          groupChoices;
      struct OperandApply {
        const compiler::detail::TemporalFusion *fusion;
        const compiler::detail::TemporalScopeDescriptor *descriptor;
        const compiler::detail::TemporalScopeChoice *choice;
        analysis::RectangularTileImage image;
      };
      llvm::SmallVector<OperandApply, 4> operandFusions;
      llvm::SmallPtrSet<const compiler::detail::TemporalFusion *, 8>
          collectedFusions;
      for (mlir::Operation *operation : consumers) {
        const auto *descriptor = descriptors.lookup(operation);
        const auto *scopeChoice = choicesByOperation.lookup(operation);
        if (!descriptor || !scopeChoice) {
          manualFailure =
              "manual joint consumer is absent from the current choice";
          return mlir::failure();
        }
        groupDescriptors.push_back(descriptor);
        groupChoices.push_back(scopeChoice);
        for (const auto *fusion : operandsByConsumer.lookup(operation)) {
          if (!collectedFusions.insert(fusion).second)
            continue;
          auto use = llvm::find_if(fusion->uses, [&](const auto &current) {
            return current.operand->getOwner() == operation;
          });
          if (use == fusion->uses.end())
            return mlir::failure();
          auto image = compiler::detail::queryTemporalFusionTile(*fusion, *use,
                                                                 *scopeChoice);
          if (!image.isExact()) {
            manualFailure = image.reason;
            return mlir::failure();
          }
          operandFusions.push_back(
              {fusion, descriptor, scopeChoice, std::move(*image.image)});
        }
      }
      mlir::FailureOr<JointConsumerTilingResult> tiled = tileJointConsumers(
          rewriter, consumers, groupDescriptors, groupChoices);
      if (mlir::failed(tiled)) {
        manualFailure =
            "compatible joint consumers could not form one common SCF loop";
        return mlir::failure();
      }
      statistics.tiledTraversals += tiled->tiledConsumers;
      statistics.loops += tiled->loops.size();
      for (const OperandApply &operand : operandFusions) {
        std::string detail;
        if (mlir::failed(materializeFusionProducer(
                rewriter, region, *operand.fusion, *operand.descriptor,
                *operand.choice, operand.image, tiled->loops, statistics,
                detail, producerTiling))) {
          manualFailure = "exact operand demand did not materialize: " + detail;
          return mlir::failure();
        }
      }
      if (mlir::failed(specializeRaggedTails(
              rewriter, *groupDescriptors.front(), *groupChoices.front(),
              tiled->loops, statistics))) {
        manualFailure = "joint consumer loop could not form its static tail";
        return mlir::failure();
      }
      return mlir::success();
    };

    llvm::SmallVector<llvm::SmallVector<mlir::Operation *, 8>, 4> cohorts;
    for (const auto &group : domain.getFusions()) {
      llvm::SmallVector<mlir::Operation *, 8> consumers;
      for (const auto &use : group.uses) {
        auto *consumer = use.operand->getOwner();
        if (choicesByOperation.count(consumer) &&
            !llvm::is_contained(consumers, consumer))
          consumers.push_back(consumer);
      }
      if (consumers.size() < 2)
        continue;
      for (size_t i = 0; i < cohorts.size();) {
        if (!llvm::any_of(cohorts[i], [&](auto *op) {
              return llvm::is_contained(consumers, op);
            })) {
          ++i;
          continue;
        }
        for (auto *op : cohorts[i])
          if (!llvm::is_contained(consumers, op))
            consumers.push_back(op);
        cohorts.erase(cohorts.begin() + i);
      }
      llvm::sort(consumers, [](auto *lhs, auto *rhs) {
        return lhs->isBeforeInBlock(rhs);
      });
      cohorts.push_back(std::move(consumers));
    }
    for (const auto &consumers : cohorts) {
      if (!llvm::any_of(consumers, hasActiveDimension))
        continue;
      if (mlir::failed(tileManualConsumers(consumers)))
        return fail<TemporalTilingStatistics>(
            failure, TemporalTilingFailureKind::CompilerFailure,
            "shared fusion traversal: " + manualFailure);
      for (auto *operation : consumers)
        commonLoopConsumers.insert(operation);
    }
    for (const auto &scope : choice.scopes) {
      mlir::Operation *consumer = scope.operation;
      if (!operandsByConsumer.count(consumer))
        continue;
      if (commonLoopConsumers.contains(consumer))
        continue;
      if (!hasActiveDimension(consumer))
        continue;
      llvm::SmallVector<mlir::Operation *, 1> singleton{consumer};
      if (mlir::failed(tileManualConsumers(singleton)))
        return fail<TemporalTilingStatistics>(
            failure, TemporalTilingFailureKind::CompilerFailure,
            "operand consumer: " + manualFailure);
      commonLoopConsumers.insert(consumer);
    }
  }
  for (const auto &scope : llvm::reverse(choice.scopes)) {
    if (commonLoopConsumers.contains(scope.operation) ||
        producerTiling.reductions.count(scope.operation))
      continue;
    const auto &descriptor = *descriptors.find(scope.operation)->second;
    llvm::SmallVector<mlir::OpFoldResult, 6> tileSizes;
    bool hasActiveDimension = false;
    for (auto [extent, size] : llvm::zip_equal(descriptor.iterationExtents,
                                               scope.iteratorTileSizes)) {
      const bool active = size < extent;
      hasActiveDimension |= active;
      tileSizes.push_back(rewriter.getIndexAttr(active ? size : 0));
    }
    if (!hasActiveDimension)
      continue;

    auto tiling = mlir::dyn_cast<mlir::TilingInterface>(scope.operation);
    if (!tiling)
      return fail<TemporalTilingStatistics>(
          failure, TemporalTilingFailureKind::BrokenContract,
          "temporal traversal lost TilingInterface before apply");
    ExactFusionInventory independentInventory;
    mlir::FailureOr<ExactFusionInventory> exactFusionEdges =
        choice.kind == compiler::detail::TemporalTraversalKind::Joint
            ? collectExactFusionEdges(region, failure)
            : mlir::FailureOr<ExactFusionInventory>(
                  std::move(independentInventory));
    if (mlir::failed(exactFusionEdges))
      return mlir::failure();
    llvm::SmallVector<ViewFusionRequest, 4> viewFusionRequests;
    for (const auto &path : exactFusionEdges->viewPaths) {
      if (!path.uses.front().operand ||
          !liveSources.sources.count(path.producer.getOwner()) ||
          path.uses.front().operand->getOwner() != scope.operation)
        continue;
      viewFusionRequests.push_back(
          {path.producer, path.uses.front().operand->get(),
           path.uses.front().producerDimensions,
           path.uses.front().representation ==
                   compiler::detail::TemporalTileRepresentation::ReshapePieces
               ? &*path.uses.front().viewToProducer
               : nullptr});
    }
    llvm::SmallVector<ConcatFusionRequest, 2> concatFusionRequests;
    if (choice.kind == compiler::detail::TemporalTraversalKind::Joint)
      for (mlir::OpOperand &operand : scope.operation->getOpOperands()) {
        compiler::detail::TemporalConcatQueryResult concat =
            compiler::detail::queryTemporalConcatAssembly(operand);
        if (concat.kind ==
            compiler::detail::TemporalConcatQueryKind::BrokenContract)
          return fail<TemporalTilingStatistics>(
              failure, TemporalTilingFailureKind::BrokenContract,
              concat.detail);
        if (concat.isExact())
          concatFusionRequests.push_back(
              {concat.assembledValue, std::move(concat.segments)});
      }
    mlir::scf::SCFTilingOptions tilingOptions;
    tilingOptions.setTileSizes(tileSizes);
    tilingOptions.setInterchange(buildInterchange(descriptor, scope));
    mlir::scf::SCFTileAndFuseOptions options;
    options.setTilingOptions(std::move(tilingOptions));
    options.setFusionControlFn(
        [&](mlir::tensor::ExtractSliceOp, mlir::OpResult producer,
            bool isDestinationOperand)
            -> std::optional<
                mlir::scf::SCFTileAndFuseOptions::ControlFnResult> {
          if (isDestinationOperand ||
              !liveSources.sources.count(producer.getOwner()) ||
              producerTiling.reductions.count(producer.getOwner()) ||
              !exactFusionEdges->directEdges.contains(producer))
            return std::nullopt;
          return mlir::scf::SCFTileAndFuseOptions::ControlFnResult{
              /*yieldProducerReplacement=*/false};
        });

    rewriter.setInsertionPoint(scope.operation);
    mlir::FailureOr<mlir::scf::SCFTileAndFuseResult> tiled =
        mlir::scf::tileConsumerAndFuseProducersUsingSCF(rewriter, tiling,
                                                        options);
    if (mlir::failed(tiled))
      return fail<TemporalTilingStatistics>(
          failure, TemporalTilingFailureKind::CompilerFailure,
          "pinned SCF tile-and-fuse failed after temporal preflight");
    if (tiled->loops.size() != scope.loopOrder.size())
      return fail<TemporalTilingStatistics>(
          failure, TemporalTilingFailureKind::CompilerFailure,
          "pinned SCF tiler returned an unexpected loop nest");
    llvm::SmallVector<mlir::Value, 4> replacements;
    for (mlir::Value result : scope.operation->getResults()) {
      auto replacement = tiled->replacements.find(result);
      if (replacement == tiled->replacements.end())
        return fail<TemporalTilingStatistics>(
            failure, TemporalTilingFailureKind::CompilerFailure,
            "pinned SCF tiler omitted a traversal result replacement");
      replacements.push_back(replacement->second);
    }
    llvm::SmallVector<mlir::Operation *, 8> fusedProducers(
        tiled->fusedProducers.begin(), tiled->fusedProducers.end());
    rewriter.replaceOp(scope.operation, replacements);
    if (mlir::failed(eraseFusedProducers(rewriter, fusedProducers)))
      return fail<TemporalTilingStatistics>(
          failure, TemporalTilingFailureKind::CompilerFailure,
          "exact-derived producer remained live after fusion");
    statistics.tiledTraversals++;
    statistics.loops += tiled->loops.size();
    statistics.fusedProducers += fusedProducers.size();
    if (mlir::failed(specializeRaggedTails(rewriter, descriptor, scope,
                                           tiled->loops, statistics)))
      return fail<TemporalTilingStatistics>(
          failure, TemporalTilingFailureKind::CompilerFailure,
          "ragged temporal loop could not form one static tail");
    canonicalizeLoopBoundMinMax(rewriter, region);
    llvm::erase_if(concatFusionRequests, [&](const auto &request) {
      return !canSpecializeConcatSlices(region, request);
    });
    if (mlir::failed(specializeConcatLoopBoundaries(
            rewriter, region, concatFusionRequests, statistics)))
      return fail<TemporalTilingStatistics>(
          failure, TemporalTilingFailureKind::CompilerFailure,
          "static concat boundary could not specialize its canonical loop");
    if (mlir::failed(fuseViewProducerSlices(
            rewriter, region, viewFusionRequests, statistics, producerTiling)))
      return fail<TemporalTilingStatistics>(
          failure, TemporalTilingFailureKind::CompilerFailure,
          "exact view-derived producer could not be tiled into its consumer");
    if (mlir::failed(fuseConcatSlices(rewriter, region, concatFusionRequests,
                                      statistics, producerTiling)))
      return fail<TemporalTilingStatistics>(
          failure, TemporalTilingFailureKind::CompilerFailure,
          "exact insert assembly could not form one tile-local value");
  }

  if (choice.kind == compiler::detail::TemporalTraversalKind::Joint &&
      !domain.getFusions().empty()) {
    std::string jointFailure;
    if (mlir::failed(fuseJointProducerSlices(rewriter, region,
                                             domain.getFusions(), statistics,
                                             jointFailure, producerTiling)))
      return fail<TemporalTilingStatistics>(
          failure, TemporalTilingFailureKind::CompilerFailure,
          "all-use joint producer did not materialize once in its common "
          "loop: " +
              jointFailure);
  }

  for (const auto &scope : llvm::reverse(choice.scopes)) {
    if (!producerTiling.reductions.count(scope.operation))
      continue;
    if (mlir::failed(
            specializeCurrentRaggedLoops(rewriter, region, statistics)) ||
        mlir::failed(composeCurrentSlices(rewriter, region)))
      return fail<TemporalTilingStatistics>(
          failure, TemporalTilingFailureKind::CompilerFailure,
          "inner traversal could not expose its actual input slices");
    llvm::SmallVector<mlir::tensor::ExtractSliceOp> slices;
    region.walk([&](mlir::tensor::ExtractSliceOp slice) {
      if (slice.getSource().getDefiningOp() == scope.operation &&
          !slice->use_empty())
        slices.push_back(slice);
    });
    for (auto slice : slices) {
      auto producer = mlir::cast<mlir::OpResult>(slice.getSource());
      rewriter.setInsertionPoint(slice);
      auto tiled = producerTiling.materialize(rewriter, slice, producer);
      if (mlir::failed(tiled))
        return fail<TemporalTilingStatistics>(
            failure, TemporalTilingFailureKind::CompilerFailure,
            "fused producer could not materialize its selected inner "
            "traversal");
      auto replacement =
          reshapeTile(rewriter, slice.getLoc(), tiled->tiledValues.front(),
                      slice.getType());
      if (mlir::failed(replacement))
        return fail<TemporalTilingStatistics>(
            failure, TemporalTilingFailureKind::CompilerFailure,
            "fused reduction result does not match its requested slice");
      rewriter.replaceOp(slice, *replacement);
    }
    if (!slices.empty()) {
      ++statistics.fusedProducers;
    } else if (!producerTiling.reductions.find(scope.operation)
                    ->second.materialized &&
               !scope.operation->use_empty() && !scope.loopOrder.empty()) {
      // A full-extent consumer has no slice to fuse. Its producer still owns
      // the selected internal reduction, at this same output scope.
      const auto &descriptor =
          producerTiling.reductions.find(scope.operation)->second.descriptor;
      llvm::SmallVector<mlir::OpFoldResult> sizes;
      for (auto [extent, size] : llvm::zip_equal(descriptor.iterationExtents,
                                                 scope.iteratorTileSizes))
        sizes.push_back(rewriter.getIndexAttr(size < extent ? size : 0));
      mlir::scf::SCFTilingOptions options;
      options.setTileSizes(sizes);
      options.setInterchange(buildInterchange(descriptor, scope));
      rewriter.setInsertionPoint(scope.operation);
      auto tiled = mlir::scf::tileUsingSCF(
          rewriter, mlir::cast<mlir::TilingInterface>(scope.operation),
          options);
      if (mlir::failed(tiled))
        return fail<TemporalTilingStatistics>(
            failure, TemporalTilingFailureKind::CompilerFailure,
            "full-output producer could not materialize its inner reduction");
      rewriter.replaceOp(scope.operation, tiled->replacements);
      statistics.loops += tiled->loops.size();
      ++statistics.tiledTraversals;
    }
  }
  if (mlir::failed(specializeCurrentRaggedLoops(rewriter, region, statistics)))
    return fail<TemporalTilingStatistics>(
        failure, TemporalTilingFailureKind::CompilerFailure,
        "fused inner traversal could not specialize its current tail");
  canonicalizeLoopBoundMinMax(rewriter, region);
  if (mlir::failed(canonicalizeTiledRegion(region, &liveSources)) ||
      mlir::failed(refineLinalgStaticTypes(rewriter, region)) ||
      mlir::failed(refineOnlineAttentionStaticTypes(rewriter, region)) ||
      mlir::failed(refinePackUnPackStaticTypes(rewriter, region)) ||
      mlir::failed(canonicalizeTiledRegion(region, &liveSources,
                                           /*simplifyPackAndUnpack=*/true)) ||
      mlir::failed(refineLinalgStaticTypes(rewriter, region)) ||
      mlir::failed(refineOnlineAttentionStaticTypes(rewriter, region)) ||
      mlir::failed(lowerConstantPads(rewriter, region, statistics)) ||
      mlir::failed(lowerConstantGenerates(rewriter, region, statistics)) ||
      mlir::failed(canonicalizeTiledRegion(region, &liveSources)))
    return fail<TemporalTilingStatistics>(
        failure, TemporalTilingFailureKind::CompilerFailure,
        "bounded temporal canonicalization did not converge");
  while (true) {
    llvm::SmallVector<mlir::tensor::ExtractSliceOp> inputs;
    llvm::SmallVector<mlir::Operation *> views;
    region.walk([&](mlir::tensor::ExtractSliceOp slice) {
      if (slice->use_empty())
        return;
      auto *source = slice.getSource().getDefiningOp();
      if (liveSources.sources.count(source))
        inputs.push_back(slice);
      if (liveSources.views.count(source) && !llvm::is_contained(views, source))
        views.push_back(source);
    });
    if (inputs.empty() && views.empty())
      break;
    for (auto *view : views) {
      const auto &source = liveSources.views.find(view)->second;
      ViewFusionRequest request{source.producer, source.value,
                                source.dimensions,
                                source.relation ? &*source.relation : nullptr};
      if (mlir::failed(fuseViewProducerSlices(rewriter, region, {request},
                                              statistics, producerTiling)))
        return fail<TemporalTilingStatistics>(
            failure, TemporalTilingFailureKind::CompilerFailure,
            "inner traversal view input could not materialize its exact "
            "producer");
    }
    for (auto slice : inputs) {
      auto producer = mlir::cast<mlir::OpResult>(slice.getSource());
      auto source = liveSources.sources.find(producer.getOwner());
      if (!source->second) {
        source->second = true;
        ++statistics.fusedProducers;
      }
      rewriter.setInsertionPoint(slice);
      auto tile = producerTiling.materialize(rewriter, slice, producer);
      if (mlir::failed(tile) || tile->tiledValues.empty())
        return fail<TemporalTilingStatistics>(
            failure, TemporalTilingFailureKind::CompilerFailure,
            "inner traversal input could not materialize its exact producer");
      auto replacement = reshapeTile(
          rewriter, slice.getLoc(), tile->tiledValues.front(), slice.getType());
      if (mlir::failed(replacement))
        return fail<TemporalTilingStatistics>(
            failure, TemporalTilingFailureKind::CompilerFailure,
            "inner traversal input has an incompatible result tile");
      rewriter.replaceOp(slice, *replacement);
    }
    if (mlir::failed(
            specializeCurrentRaggedLoops(rewriter, region, statistics)) ||
        mlir::failed(canonicalizeTiledRegion(region, &liveSources)) ||
        mlir::failed(refineLinalgStaticTypes(rewriter, region)))
      return fail<TemporalTilingStatistics>(
          failure, TemporalTilingFailureKind::CompilerFailure,
          "inner traversal inputs could not reach a static current boundary");
  }
  if (mlir::failed(mlir::verify(region)))
    return fail<TemporalTilingStatistics>(
        failure, TemporalTilingFailureKind::CompilerFailure,
        "temporal tiling was invalid before final common-subexpression "
        "elimination");
  if (mlir::failed(materializeInitializerTiles(rewriter, region)) ||
      mlir::failed(canonicalizeTiledRegion(region, &liveSources)))
    return fail<TemporalTilingStatistics>(
        failure, TemporalTilingFailureKind::CompilerFailure,
        "current initializer tile could not be materialized");
  mlir::DominanceInfo dominance(region);
  mlir::eliminateCommonSubExpressions(rewriter, dominance, region);
  eraseDeadOperations(rewriter, region);
  if (!listener.finalizeAfterRewrite() || mlir::failed(mlir::verify(region)) ||
      mlir::failed(verifyStructuralTileRegions(
          region->getParentOfType<mlir::ModuleOp>())) ||
      mlir::failed(compiler::detail::checkStructuredBufferRelationsCurrent(
          region->getParentOfType<mlir::ModuleOp>(), relations)))
    return fail<TemporalTilingStatistics>(
        failure, TemporalTilingFailureKind::CompilerFailure,
        "temporal tiling produced invalid current structural IR");
  return statistics;
}

} // namespace wafer
