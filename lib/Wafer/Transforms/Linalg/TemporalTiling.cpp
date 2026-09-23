//===- TemporalTiling.cpp - Apply live-operation temporal choices -----===//

#include "TemporalTiling.h"
#include "AttentionVisibility.h"
#include "TensorAssemblyMaterialization.h"
#include "Wafer/Transforms/Tile/TensorInitialization.h"

#include "Wafer/Analysis/Linalg/TensorResultIndexing.h"
#include "Wafer/Transforms/Linalg/StructuredTiling.h"

#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
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
#include "mlir/Interfaces/SubsetOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
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

using compiler::detail::resolveStaticIndex;

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

struct AssemblyReadRequest {
  mlir::Value assembledValue;
};

struct AssemblyProducerFusion {
  mlir::Value assembly;
  mlir::OpResult producer;
};

AssemblyReadRequest getAssemblyReadRequest(
    const compiler::detail::TemporalConcatQueryResult &query) {
  return {query.assembledValue};
}

void collectAssemblyProducerFusions(
    const compiler::detail::TemporalConcatQueryResult &query,
    llvm::SmallVectorImpl<AssemblyProducerFusion> &sources) {
  for (const auto &segment : query.segments)
    if (segment.derivedProducer &&
        llvm::none_of(sources, [&](const auto &selected) {
          return selected.assembly == query.assembledValue &&
                 selected.producer == *segment.derivedProducer;
        }))
      sources.push_back({query.assembledValue, *segment.derivedProducer});
}

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

using analysis::getTensorLoopGridFloor;
using analysis::queryTensorLoopGrid;
using analysis::TensorLoopGrid;

mlir::FailureOr<mlir::scf::ForOp> splitForLoopAt(mlir::IRRewriter &rewriter,
                                                 mlir::scf::ForOp loop,
                                                 int64_t split,
                                                 mlir::IRMapping &mapping) {
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
  auto suffix = mlir::cast<mlir::scf::ForOp>(rewriter.clone(*loop, mapping));
  rewriter.modifyOpInPlace(
      suffix, [&] { suffix.getLowerBoundMutable().assign(splitValue); });
  rewriter.replaceAllUsesWith(loop.getResults(), suffix.getResults());
  rewriter.modifyOpInPlace(
      suffix, [&] { suffix.getInitArgsMutable().assign(loop.getResults()); });
  rewriter.modifyOpInPlace(
      loop, [&] { loop.getUpperBoundMutable().assign(splitValue); });
  return suffix;
}

TensorAssemblyOpportunityKind
assemblyOutcome(analysis::TensorAssemblyStatus status) {
  using Kind = TensorAssemblyOpportunityKind;
  switch (status) {
  case analysis::TensorAssemblyStatus::Exact:
    return Kind::Available;
  case analysis::TensorAssemblyStatus::NotAssembly:
    return Kind::NotApplicable;
  case analysis::TensorAssemblyStatus::Unsupported:
    return Kind::Unsupported;
  case analysis::TensorAssemblyStatus::ResourceExhausted:
    return Kind::ResourceExhausted;
  case analysis::TensorAssemblyStatus::BrokenContract:
    return Kind::BrokenContract;
  }
  llvm_unreachable("unknown assembly status");
}

TemporalTilingFailureKind
assemblyFailureKind(TensorAssemblyOpportunityKind kind) {
  switch (kind) {
  case TensorAssemblyOpportunityKind::Unsupported:
  case TensorAssemblyOpportunityKind::NotApplicable:
    return TemporalTilingFailureKind::Unsupported;
  case TensorAssemblyOpportunityKind::ResourceExhausted:
    return TemporalTilingFailureKind::ResourceExhausted;
  case TensorAssemblyOpportunityKind::BrokenContract:
    return TemporalTilingFailureKind::BrokenContract;
  case TensorAssemblyOpportunityKind::Available:
    return TemporalTilingFailureKind::None;
  }
  llvm_unreachable("unknown assembly read outcome");
}

mlir::LogicalResult specializeConcatLoopBoundaries(
    mlir::IRRewriter &rewriter, TileRegionOp region,
    llvm::ArrayRef<AssemblyReadRequest> requests,
    TemporalTilingStatistics &statistics,
    llvm::SmallVectorImpl<mlir::tensor::ExtractSliceOp> *selectedReads =
        nullptr,
    TemporalTilingFailure *failure = nullptr,
    const analysis::IndexRelationLimits &limits =
        analysis::IndexRelationLimits()) {
  // Collect every current read before mutation. Different affine origins in
  // the same loop require different boundary iterations; the loop identity
  // only groups their union, never suppresses a second read's constraints.
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<int64_t, 8>> loopSplits;
  for (const auto &request : requests) {
    bool invalid = false;
    region.walk([&](mlir::tensor::ExtractSliceOp slice) {
      if (slice.getSource() != request.assembledValue ||
          (selectedReads && !llvm::is_contained(*selectedReads, slice)))
        return;
      auto query = compiler::detail::queryTensorAssemblySlice(slice, limits);
      if (!query.isExact()) {
        if (failure)
          *failure = {assemblyFailureKind(assemblyOutcome(query.status)),
                      query.detail};
        invalid = true;
        return;
      }
      for (const auto &partition : query.loops) {
        auto *loop = mlir::cast<mlir::BlockArgument>(partition.induction)
                         .getOwner()
                         ->getParentOp();
        auto &splits = loopSplits[loop];
        splits.append(partition.boundaries.begin() + 1,
                      partition.boundaries.end() - 1);
      }
    });
    if (invalid)
      return mlir::failure();
  }
  if (loopSplits.empty())
    return mlir::success();
  // Bound cloning work from the actual current body and the union of all
  // selected partitions. This is a rewrite budget, never SPM admission.
  uint64_t work = 0;
  for (auto &entry : loopSplits) {
    auto &splits = entry.second;
    llvm::sort(splits);
    splits.erase(std::unique(splits.begin(), splits.end()), splits.end());
  }
  bool exhausted = false;
  region.walk([&](mlir::Operation *operation) {
    uint64_t copies = 1;
    for (auto *parent = operation->getParentOp(); parent && parent != region;
         parent = parent->getParentOp()) {
      auto found = loopSplits.find(parent);
      if (found == loopSplits.end())
        continue;
      uint64_t intervals = found->second.size() + 1;
      if (copies > limits.maxConstraintWork / intervals) {
        exhausted = true;
        return;
      }
      copies *= intervals;
    }
    if (copies == 1)
      return;
    if (copies > limits.maxConstraintWork - work) {
      exhausted = true;
      return;
    }
    work += copies;
  });
  if (exhausted) {
    if (failure)
      *failure = {
          TemporalTilingFailureKind::ResourceExhausted,
          "selected assembly partitions exceed the rewrite work budget"};
    return mlir::failure();
  }
  // Split nested loops first so a subsequent outer-loop clone contains all
  // already-specialized inner reads. Never replay handles into cloned IR.
  llvm::SmallVector<mlir::scf::ForOp, 8> loops;
  region.walk<mlir::WalkOrder::PostOrder>([&](mlir::scf::ForOp loop) {
    if (loopSplits.contains(loop))
      loops.push_back(loop);
  });
  for (auto loop : loops) {
    auto &splits = loopSplits[loop];
    llvm::sort(splits);
    splits.erase(std::unique(splits.begin(), splits.end()), splits.end());
    auto current = loop;
    for (int64_t split : splits) {
      mlir::IRMapping mapping;
      auto suffix = splitForLoopAt(rewriter, current, split, mapping);
      if (mlir::failed(suffix))
        return mlir::failure();
      if (selectedReads) {
        llvm::SmallVector<mlir::tensor::ExtractSliceOp> clonedReads;
        for (auto read : *selectedReads)
          if (auto *mapped = mapping.lookupOrNull(read.getOperation()))
            clonedReads.push_back(
                mlir::cast<mlir::tensor::ExtractSliceOp>(mapped));
        selectedReads->append(clonedReads);
      }
      current = *suffix;
      ++statistics.loops;
      ++statistics.specializedConcatBoundaries;
    }
  }
  return mlir::success();
}

struct AssemblySliceReuse {
  mlir::Operation *insertionPoint;
  llvm::SmallVector<std::pair<int64_t, int64_t>, 4> bounds;
};

// A static subset proof alone does not authorize replication. Inspect the
// actual loop family between this read and the shared value. An invariant
// suffix can share one local assembly. Repetition around a dependency needs
// an explicit local implementation; it keeps the existing loop order/storage.
std::optional<AssemblySliceReuse>
queryAssemblySliceReuse(mlir::tensor::ExtractSliceOp slice,
                        bool allowRepeatedReads = false) {
  AssemblySliceReuse result{slice.getOperation(), {}};
  llvm::SmallDenseSet<mlir::Value, 4> dependencies;
  for (auto [offset, size] :
       llvm::zip_equal(slice.getMixedOffsets(), slice.getMixedSizes())) {
    auto length = resolveStaticIndex(size);
    if (!length || *length <= 0)
      return std::nullopt;
    int64_t begin = 0, last = 0, end = 0;
    if (auto constant = resolveStaticIndex(offset)) {
      begin = last = *constant;
    } else {
      auto grid = queryTensorLoopGrid(offset).grid;
      if (!grid || (!allowRepeatedReads && grid->step < *length))
        return std::nullopt;
      auto final = getTensorLoopGridFloor(*grid, grid->upper - 1);
      if (!final)
        return std::nullopt;
      begin = grid->lower;
      last = *final;
      dependencies.insert(grid->induction);
    }
    if (llvm::AddOverflow(last, *length, end))
      return std::nullopt;
    result.bounds.push_back({begin, end});
  }
  bool insideDependency = false;
  for (auto *block = slice->getBlock();
       block != slice.getSource().getParentBlock();) {
    auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(block->getParentOp());
    if (!loop)
      return std::nullopt;
    auto grid = queryTensorLoopGrid(loop.getInductionVar()).grid;
    if (!grid)
      return std::nullopt;
    if (dependencies.contains(loop.getInductionVar())) {
      insideDependency = true;
    } else if (!insideDependency) {
      result.insertionPoint = loop;
    } else if (!allowRepeatedReads && grid->upper - grid->lower > grid->step) {
      return std::nullopt;
    }
    // An explicit local implementation may repeat this pure read inside an
    // invariant outer loop. Keep its actual placement: hoisting across a
    // dependency would invent a cache or change its dynamic instance.
    block = loop->getBlock();
  }
  mlir::DominanceInfo dominance;
  if (!dominance.properlyDominates(slice.getSource(), result.insertionPoint) ||
      llvm::any_of(dependencies, [&](mlir::Value induction) {
        return !dominance.properlyDominates(induction, result.insertionPoint);
      }))
    return std::nullopt;
  return result;
}

// The assembly query proves the value, not the selected traversal's ability to
// rebuild it with static pieces. Overlapping windows can still read the
// already materialized assembly; they do not require concat fusion to tile.
TensorAssemblyOpportunity
queryUnsharedAssemblyReads(TileRegionOp region,
                           const AssemblyReadRequest &request) {
  auto type =
      mlir::dyn_cast<mlir::RankedTensorType>(request.assembledValue.getType());
  if (!type || !type.hasStaticShape())
    return {TensorAssemblyOpportunityKind::Unsupported,
            "assembly read requires an explicit shared-use choice"};
  bool found = false;
  bool supported = true;
  std::optional<TensorAssemblyOpportunity> queryFailure;
  llvm::SmallVector<AssemblySliceReuse, 4> reads;
  mlir::OpOperand *nextDestination = nullptr;
  auto current = request.assembledValue;
  while (auto insert =
             current.getDefiningOp<mlir::SubsetInsertionOpInterface>()) {
    if (nextDestination)
      for (auto &use : current.getUses())
        if (&use != nextDestination && !mlir::isOpTriviallyDead(use.getOwner()))
          return {TensorAssemblyOpportunityKind::Unsupported,
                  "assembly read requires an explicit shared-use choice"};
    nextDestination = &insert.getDestinationOperand();
    current = nextDestination->get();
  }
  // A retained full consumer already owns the shared assembly. Rebuilding its
  // fragments for another consumer would introduce unselected duplication.
  for (auto &use : request.assembledValue.getUses()) {
    auto slice = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(use.getOwner());
    if ((!slice || &slice.getSourceMutable() != &use) &&
        !mlir::isOpTriviallyDead(use.getOwner()))
      return {TensorAssemblyOpportunityKind::Unsupported,
              "assembly read requires an explicit shared-use choice"};
  }
  region.walk([&](mlir::tensor::ExtractSliceOp slice) {
    if (slice.getSource() != request.assembledValue)
      return;
    found = true;
    auto reuse = queryAssemblySliceReuse(slice);
    if (!reuse) {
      supported = false;
      return;
    }
    for (auto [bound, extent] : llvm::zip_equal(reuse->bounds, type.getShape()))
      if (bound.first < 0 || bound.second > extent)
        supported = false;
    for (const auto &other : reads) {
      bool disjoint = false;
      for (auto [left, right] : llvm::zip_equal(reuse->bounds, other.bounds))
        disjoint |= left.second <= right.first || right.second <= left.first;
      if (!disjoint)
        supported = false;
    }
    reads.push_back(std::move(*reuse));
    auto query = compiler::detail::queryTensorAssemblySlice(slice);
    if (!query.isExact()) {
      auto kind = assemblyOutcome(query.status);
      if (!queryFailure ||
          kind == TensorAssemblyOpportunityKind::BrokenContract ||
          (kind == TensorAssemblyOpportunityKind::ResourceExhausted &&
           queryFailure->kind != TensorAssemblyOpportunityKind::BrokenContract))
        queryFailure = TensorAssemblyOpportunity{kind, std::move(query.detail)};
    }
  });
  if (queryFailure)
    return std::move(*queryFailure);
  if (!found)
    return {TensorAssemblyOpportunityKind::NotApplicable, {}};
  if (!supported)
    return {TensorAssemblyOpportunityKind::Unsupported,
            "assembly read requires an explicit shared-use choice"};
  return {TensorAssemblyOpportunityKind::Available, {}};
}

mlir::FailureOr<llvm::SmallVector<mlir::tensor::ExtractSliceOp, 4>>
materializeAssemblySlices(
    mlir::IRRewriter &rewriter, TileRegionOp region,
    llvm::ArrayRef<AssemblyReadRequest> requests,
    TemporalTilingStatistics &statistics,
    mlir::tensor::ExtractSliceOp selectedRead = {}) {
  llvm::SmallVector<mlir::tensor::ExtractSliceOp, 4> sourceReads;
  for (const auto &request : requests) {
    llvm::SmallVector<mlir::tensor::ExtractSliceOp, 4> slices;
    region.walk([&](mlir::tensor::ExtractSliceOp slice) {
      if (slice.getSource() == request.assembledValue &&
          (!selectedRead || selectedRead == slice))
        slices.push_back(slice);
    });
    if (slices.empty())
      return mlir::failure();
    for (auto slice : slices) {
      auto query = compiler::detail::queryTensorAssemblySlice(slice);
      auto reuse = queryAssemblySliceReuse(slice, /*allowRepeatedReads=*/true);
      if (!query.isExact() || query.cases.size() != 1 || !reuse)
        return mlir::failure();
      rewriter.setInsertionPoint(reuse->insertionPoint);
      llvm::SmallVector<int64_t, 4> sizes;
      for (auto size : slice.getMixedSizes())
        sizes.push_back(*resolveStaticIndex(size));
      auto generated = compiler::detail::materializeTensorAssemblyRead(
          rewriter, slice.getLoc(), slice.getType(), sizes, query);
      if (mlir::failed(generated))
        return mlir::failure();
      statistics.assembledSegments += generated->sourceReads.size();
      ++statistics.tileLocalAssemblies;
      sourceReads.append(generated->sourceReads);
      rewriter.replaceOp(slice, generated->value);
    }
  }
  return sourceReads;
}

mlir::LogicalResult fuseAssemblySources(
    mlir::IRRewriter &rewriter, TileRegionOp region,
    llvm::ArrayRef<AssemblyReadRequest> requests,
    llvm::ArrayRef<AssemblyProducerFusion> selectedProducers,
    TemporalTilingStatistics &statistics, ProducerTiling &producerTiling) {
  for (const auto &request : requests) {
    auto generated =
        materializeAssemblySlices(rewriter, region, {request}, statistics);
    if (mlir::failed(generated))
      return mlir::failure();
    llvm::DenseSet<mlir::Operation *> fusedProducerOwners;
    for (auto read : *generated) {
      auto producer = mlir::dyn_cast<mlir::OpResult>(read.getSource());
      if (!producer ||
          llvm::none_of(selectedProducers, [&](const auto &selected) {
            return selected.assembly == request.assembledValue &&
                   selected.producer == producer;
          }))
        continue;
      rewriter.setInsertionPoint(read);
      auto tiled = producerTiling.materialize(rewriter, read, producer);
      if (mlir::failed(tiled) || tiled->tiledValues.empty())
        return mlir::failure();
      auto value = reshapeTile(rewriter, read.getLoc(),
                               tiled->tiledValues.front(), read.getType());
      if (mlir::failed(value))
        return mlir::failure();
      rewriter.replaceOp(read, *value);
      fusedProducerOwners.insert(producer.getOwner());
    }
    statistics.fusedProducers += fusedProducerOwners.size();
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

mlir::LogicalResult recordIterationCoordinates(
    mlir::RewriterBase &rewriter,
    const compiler::detail::TemporalScopeDescriptor &descriptor,
    const compiler::detail::TemporalScopeChoice &choice,
    llvm::ArrayRef<mlir::LoopLikeOpInterface> loops) {
  if (!descriptor.identity || loops.size() != choice.loopOrder.size())
    return mlir::failure();
  for (auto [loop, dimension] : llvm::zip_equal(loops, choice.loopOrder)) {
    auto *operation = mlir::LoopLikeOpInterface(loop).getOperation();
    llvm::SmallVector<IterationCoordinateAttr, 2> coordinates;
    if (auto existing = getIterationCoordinates(operation))
      llvm::append_range(coordinates, existing.getCoordinates());
    auto coordinate = IterationCoordinateAttr::get(
        rewriter.getContext(), descriptor.identity, dimension);
    if (llvm::is_contained(coordinates, coordinate))
      continue;
    coordinates.push_back(coordinate);
    rewriter.modifyOpInPlace(operation, [&] {
      operation->setAttr(
          kIterationCoordinatesAttrName,
          IterationCoordinatesAttr::get(rewriter.getContext(), coordinates));
    });
  }
  return mlir::success();
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

// This is a selected-slice transformation, not ordinary graph exploration.
// Compose the view chain before taking its image: an intermediate flatten can
// scatter one compact multi-dimensional window into many disjoint intervals.
struct LocalizeTensorViewSlice
    : mlir::OpInterfaceRewritePattern<mlir::SubsetExtractionOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::SubsetExtractionOpInterface slice,
                  mlir::PatternRewriter &rewriter) const override {
    auto type =
        mlir::dyn_cast<mlir::RankedTensorType>(slice.getResult().getType());
    auto sourceType = mlir::dyn_cast<mlir::RankedTensorType>(
        slice.getSourceOperand().get().getType());
    auto subset = mlir::cast<mlir::SubsetOpInterface>(slice.getOperation())
                      .getAccessedHyperrectangularSlice();
    if (!type || !sourceType || !sourceType.hasStaticShape() ||
        type.getRank() == 0 ||
        mlir::failed(subset))
      return rewriter.notifyMatchFailure(slice,
                                         "requires a static tensor subset");
    llvm::SmallVector<int64_t, 4> staticSizes;
    for (auto size : subset->getMixedSizes()) {
      auto constant = resolveStaticIndex(size);
      if (!constant)
        return rewriter.notifyMatchFailure(slice,
                                           "requires static tile extents");
      staticSizes.push_back(*constant);
    }
    auto resultType = type;
    if (!resultType.hasStaticShape()) {
      if (resultType.getRank() != static_cast<int64_t>(staticSizes.size()))
        return rewriter.notifyMatchFailure(slice,
                                           "dynamic rank reduction is unknown");
      resultType = mlir::RankedTensorType::get(
          staticSizes, type.getElementType(), type.getEncoding());
    }
    if (llvm::any_of(subset->getMixedStrides(), [](auto stride) {
          return !mlir::isConstantIntValue(stride, 1);
        }))
      return rewriter.notifyMatchFailure(slice,
                                         "requires a dense tensor subset");
    auto chain =
        analysis::deriveTensorViewIndexing(slice.getSourceOperand().get());
    if (!chain.isExact())
      return rewriter.notifyMatchFailure(slice, chain.detail);
    auto map = chain.indexing->resultToSource.getProjectedAffineMap(
        rewriter.getContext());
    if (!map)
      return rewriter.notifyMatchFailure(slice, "tile offset is not affine");

    // An affine view translates an exact rectangle without changing its
    // shape. Prove density once; actual offsets remain the current loop SSA.
    llvm::SmallVector<int64_t, 4> zero(staticSizes.size(), 0);
    auto image = analysis::getTensorViewTileSource(
        *chain.indexing, sourceType.getShape(), {zero, staticSizes});
    if (!image.isExact())
      return rewriter.notifyMatchFailure(slice, image.reason);
    auto localSourceType = mlir::RankedTensorType::get(
        image.domain->sizes, type.getElementType(), type.getEncoding());
    if (localSourceType.getNumElements() != resultType.getNumElements() ||
        localSourceType.getRank() == 0)
      return rewriter.notifyMatchFailure(slice,
                                         "view does not preserve tile order");
    llvm::SmallVector<mlir::Attribute, 4> zeroAttributes(
        zero.size(), rewriter.getIndexAttr(0));
    llvm::SmallVector<mlir::Attribute, 4> mappedZero;
    if (mlir::failed(map->constantFold(zeroAttributes, mappedZero)))
      return rewriter.notifyMatchFailure(slice, "affine origin is unavailable");

    auto originalType =
        mlir::cast<mlir::RankedTensorType>(chain.indexing->source.getType());
    if (originalType.getElementType() != type.getElementType() ||
        originalType.getEncoding() != type.getEncoding())
      return rewriter.notifyMatchFailure(slice,
                                         "source representation differs");
    llvm::SmallVector<mlir::AffineMap, 4> translations;
    for (auto [axis, expression] : llvm::enumerate(map->getResults())) {
      int64_t origin = mlir::cast<mlir::IntegerAttr>(mappedZero[axis]).getInt();
      int64_t delta = 0;
      if (llvm::SubOverflow(image.domain->offsets[axis], origin, delta))
        return rewriter.notifyMatchFailure(slice, "tile origin overflows");
      translations.push_back(mlir::AffineMap::get(
          map->getNumDims(), 0, expression + delta, rewriter.getContext()));
    }
    llvm::SmallVector<mlir::OpFoldResult, 4> offsets, sizes, strides;
    for (auto [axis, translation] : llvm::enumerate(translations)) {
      offsets.push_back(mlir::affine::makeComposedFoldedAffineApply(
          rewriter, slice.getLoc(), translation, subset->getMixedOffsets()));
      sizes.push_back(rewriter.getIndexAttr(image.domain->sizes[axis]));
      strides.push_back(rewriter.getIndexAttr(1));
    }
    mlir::Value result = rewriter.create<mlir::tensor::ExtractSliceOp>(
        slice.getLoc(), chain.indexing->source, offsets, sizes, strides);
    result = reshapeStaticTensorTile(rewriter, slice.getLoc(), result, resultType);
    if (resultType != type)
      result = rewriter.create<mlir::tensor::CastOp>(slice.getLoc(), type, result);
    rewriter.replaceOp(slice, result);
    return mlir::success();
  }
};

void localizeAssemblyViewSlices(
    mlir::IRRewriter &rewriter, TileRegionOp region,
    llvm::ArrayRef<AssemblyReadRequest> requests) {
  if (requests.empty())
    return;
  llvm::SmallVector<mlir::SubsetExtractionOpInterface> slices;
  region.walk([&](mlir::SubsetExtractionOpInterface slice) {
    if (llvm::any_of(requests, [&](const auto &request) {
          return slice.getSourceOperand().get() == request.assembledValue;
        })) {
      slices.push_back(slice);
      return;
    }
    auto view = analysis::deriveTensorViewIndexing(slice.getSourceOperand().get());
    if (view.isExact() && llvm::any_of(requests, [&](const auto &request) {
          return view.indexing->source == request.assembledValue;
        }))
      slices.push_back(slice);
  });
  mlir::PatternRewriter patternRewriter(rewriter.getContext());
  patternRewriter.setListener(rewriter.getListener());
  LocalizeTensorViewSlice pattern(rewriter.getContext());
  for (auto slice : slices) {
    patternRewriter.setInsertionPoint(slice);
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(slice.getResult().getType());
    auto sourceType = mlir::dyn_cast<mlir::RankedTensorType>(
        slice.getSourceOperand().get().getType());
    const bool direct = llvm::any_of(requests, [&](const auto &request) {
      return slice.getSourceOperand().get() == request.assembledValue;
    });
    if (direct && type && sourceType && type.hasStaticShape() &&
        type.getRank() < sourceType.getRank()) {
      auto subset = mlir::cast<mlir::SubsetOpInterface>(slice.getOperation())
                        .getAccessedHyperrectangularSlice();
      if (mlir::succeeded(subset)) {
        llvm::SmallVector<mlir::OpFoldResult> sizes;
        llvm::SmallVector<int64_t> shape;
        for (auto size : subset->getMixedSizes()) {
          auto constant = resolveStaticIndex(size);
          if (!constant)
            break;
          shape.push_back(*constant);
          sizes.push_back(patternRewriter.getIndexAttr(*constant));
        }
        if (shape.size() == static_cast<size_t>(sourceType.getRank()) &&
            mlir::computeRankReductionMask(shape, type.getShape())) {
          mlir::Value fullRank =
              patternRewriter.create<mlir::tensor::ExtractSliceOp>(
                  slice.getLoc(), slice.getSourceOperand().get(),
                  subset->getMixedOffsets(), sizes, subset->getMixedStrides());
          auto reshaped = reshapeStaticTensorTile(
              patternRewriter, slice.getLoc(), fullRank, type);
          patternRewriter.replaceOp(slice, reshaped);
          continue;
        }
      }
    }
    // A failed match makes no mutation. The existing exact slice preflight
    // below decides whether this assembly can use the selected loop grid.
    (void)pattern.matchAndRewrite(slice, patternRewriter);
  }
}

mlir::LogicalResult
localizeCurrentAssemblies(mlir::IRRewriter &rewriter, TileRegionOp region,
                          TemporalTilingStatistics &statistics,
                          ProducerTiling &producerTiling,
                          TemporalTilingFailure *failure) {
  // Query actual subsets after all traversals, including fused reductions.
  // Rebuild after each rewrite: loop specialization can replace nested IR.
  while (true) {
    std::optional<AssemblyReadRequest> request;
    llvm::SmallVector<AssemblyProducerFusion, 4> selectedProducers;
    bool broken = false;
    region.walk([&](mlir::SubsetInsertionOpInterface assembly) {
      for (auto &use : assembly.getUpdatedDestination().getUses()) {
        auto consumer =
            mlir::dyn_cast<mlir::SubsetExtractionOpInterface>(use.getOwner());
        if (!consumer || &consumer.getSourceOperand() != &use)
          continue;
        auto query = compiler::detail::queryTemporalConcatAssembly(
            consumer.getSourceOperand());
        if (query.kind ==
            compiler::detail::TemporalConcatQueryKind::BrokenContract) {
          consumer->emitError() << query.detail;
          broken = true;
          return mlir::WalkResult::interrupt();
        }
        if (!query.isExact())
          continue;
        auto candidate = getAssemblyReadRequest(query);
        auto opportunity = queryUnsharedAssemblyReads(region, candidate);
        if (opportunity.kind ==
                TensorAssemblyOpportunityKind::ResourceExhausted ||
            opportunity.kind == TensorAssemblyOpportunityKind::BrokenContract) {
          if (failure)
            *failure = {assemblyFailureKind(opportunity.kind),
                        opportunity.detail};
          broken = true;
          return mlir::WalkResult::interrupt();
        }
        if (opportunity.kind != TensorAssemblyOpportunityKind::Available)
          continue;
        collectAssemblyProducerFusions(query, selectedProducers);
        request = std::move(candidate);
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    if (broken)
      return mlir::failure();
    if (!request)
      break;
    localizeAssemblyViewSlices(rewriter, region, {*request});
    if (mlir::failed(specializeConcatLoopBoundaries(
            rewriter, region, {*request}, statistics, nullptr, failure))) {
      if (failure && failure->kind != TemporalTilingFailureKind::None)
        return mlir::failure();
      return region.emitOpError("assembly boundary specialization failed");
    }
    if (mlir::failed(fuseAssemblySources(rewriter, region, {*request},
                                         selectedProducers, statistics,
                                         producerTiling)))
      return region.emitOpError("assembly slice fusion failed");
  }
  return mlir::success();
}

mlir::LogicalResult
canonicalizeTiledRegion(TileRegionOp region,
                        mlir::RewriterBase::Listener *listener,
                        bool simplifyPackAndUnpack = false) {
  mlir::RewritePatternSet patterns =
      mlir::linalg::getLinalgTilingCanonicalizationPatterns(
          region.getContext());
  // Pinned Pad tiling guards empty source slices with scf.if. Peeling can
  // make its condition constant while casts still hide the static branch
  // shape. Normalize that control flow before the existing type refinement.
  mlir::scf::IfOp::getCanonicalizationPatterns(patterns, region.getContext());
  // Affine composition can expose an IV only after a previous rewrite.
  // Keep its loop-range proof in the same worklist, including late fusion.
  mlir::scf::populateSCFForLoopCanonicalizationPatterns(patterns);
  patterns.add<LocalizeTensorViewSlice>(region.getContext());
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
    llvm::SmallVector<mlir::Type, 3> resultTypes;
    auto destination =
        mlir::cast<mlir::DestinationStyleOpInterface>(operation.getOperation());
    for (int64_t index = 0; index < destination.getNumDpsInits(); ++index)
      resultTypes.push_back(
          operands[destination.getDpsInitOperand(index)->getOperandNumber()]
              .getType());
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

mlir::LogicalResult refineGatherStaticTypes(mlir::IRRewriter &rewriter,
                                            TileRegionOp region) {
  llvm::SmallVector<mlir::tensor::GatherOp, 8> gathers;
  region.walk(
      [&](mlir::tensor::GatherOp gather) { gathers.push_back(gather); });
  for (auto gather : gathers) {
    auto refineOperand = [](mlir::Value value) {
      while (auto cast = value.getDefiningOp<mlir::tensor::CastOp>()) {
        auto type =
            mlir::cast<mlir::RankedTensorType>(cast.getSource().getType());
        if (!type.hasStaticShape())
          break;
        value = cast.getSource();
      }
      return value;
    };
    mlir::Value source = refineOperand(gather.getSource());
    mlir::Value indices = refineOperand(gather.getIndices());
    auto sourceType = mlir::cast<mlir::RankedTensorType>(source.getType());
    auto indicesType = mlir::cast<mlir::RankedTensorType>(indices.getType());
    bool rankReduced = gather.getResultType().getRank() ==
                       indicesType.getRank() - 1 + sourceType.getRank() -
                           static_cast<int64_t>(gather.getGatherDims().size());
    auto resultType = mlir::tensor::GatherOp::inferResultType(
        sourceType, indicesType, gather.getGatherDims(), rankReduced);
    if (source == gather.getSource() && indices == gather.getIndices() &&
        resultType == gather.getResultType())
      continue;
    if (!mlir::tensor::CastOp::areCastCompatible(gather.getResultType(),
                                                 resultType))
      return mlir::failure();
    rewriter.setInsertionPoint(gather);
    rewriter.replaceOpWithNewOp<mlir::tensor::GatherOp>(
        gather, resultType, source, indices, gather.getGatherDims(),
        gather.getUnique());
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
  if (mlir::failed(refineReshapeStaticTypes(rewriter, region)))
    return mlir::failure();
  return refineGatherStaticTypes(rewriter, region);
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
  if (mlir::failed(recordIterationCoordinates(rewriter, descriptor, choice,
                                              inner->loops)))
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
  if (mlir::failed(
          recordIterationCoordinates(rewriter, request.consumerDescriptor,
                                     request.consumerChoice, outer->loops)))
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
      if (mlir::failed(
              recordIterationCoordinates(rewriter, request.producerDescriptor,
                                         reductionChoice, inner->loops)))
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

static mlir::FailureOr<TemporalTilingStatistics> applyRegionTemporalTiling(
    const compiler::detail::TemporalDomain &domain,
    const compiler::detail::TemporalChoice &choice,
    StructuredMaterializationRelations &relations,
    compiler::detail::StructuredBufferReplacementListener &listener,
    TemporalTilingFailure *failure) {
  if (failure)
    *failure = {};
  TileRegionOp region = domain.getRegion();

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
      for (auto [descriptor, selected] :
           llvm::zip_equal(groupDescriptors, groupChoices))
        if (mlir::failed(recordIterationCoordinates(rewriter, *descriptor,
                                                    *selected, tiled->loops))) {
          manualFailure = "joint loop lost its explicit iterator coordinates";
          return mlir::failure();
        }
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
    llvm::SmallVector<AssemblyReadRequest, 2> assemblyRequests;
    llvm::SmallVector<AssemblyProducerFusion, 4> assemblyFusions;
    for (mlir::OpOperand &operand : scope.operation->getOpOperands()) {
      compiler::detail::TemporalConcatQueryResult concat =
          compiler::detail::queryTemporalConcatAssembly(operand);
      if (concat.kind ==
          compiler::detail::TemporalConcatQueryKind::BrokenContract)
        return fail<TemporalTilingStatistics>(
            failure, TemporalTilingFailureKind::BrokenContract, concat.detail);
      if (!concat.isExact())
        continue;
      // Independent traversal keeps computation at its selected scopes, but
      // its actual operand demand still localizes pure subset assembly.
      if (choice.kind == compiler::detail::TemporalTraversalKind::Joint)
        collectAssemblyProducerFusions(concat, assemblyFusions);
      assemblyRequests.push_back(getAssemblyReadRequest(concat));
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
    if (mlir::failed(recordIterationCoordinates(rewriter, descriptor, scope,
                                                tiled->loops)))
      return fail<TemporalTilingStatistics>(
          failure, TemporalTilingFailureKind::CompilerFailure,
          "tiled loop lost its explicit iterator coordinates");
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
    localizeAssemblyViewSlices(rewriter, region, assemblyRequests);
    llvm::SmallVector<AssemblyReadRequest, 2> selectedAssemblies;
    for (const auto &request : assemblyRequests) {
      auto opportunity = queryUnsharedAssemblyReads(region, request);
      if (opportunity.kind ==
              TensorAssemblyOpportunityKind::ResourceExhausted ||
          opportunity.kind == TensorAssemblyOpportunityKind::BrokenContract)
        return fail<TemporalTilingStatistics>(
            failure, assemblyFailureKind(opportunity.kind), opportunity.detail);
      if (opportunity.kind == TensorAssemblyOpportunityKind::Available)
        selectedAssemblies.push_back(request);
    }
    assemblyRequests = std::move(selectedAssemblies);
    if (mlir::failed(
            specializeConcatLoopBoundaries(rewriter, region, assemblyRequests,
                                           statistics, nullptr, failure))) {
      if (failure && failure->kind != TemporalTilingFailureKind::None)
        return mlir::failure();
      return fail<TemporalTilingStatistics>(
          failure, TemporalTilingFailureKind::CompilerFailure,
          "static concat boundary could not specialize its canonical loop");
    }
    if (mlir::failed(fuseViewProducerSlices(
            rewriter, region, viewFusionRequests, statistics, producerTiling)))
      return fail<TemporalTilingStatistics>(
          failure, TemporalTilingFailureKind::CompilerFailure,
          "exact view-derived producer could not be tiled into its consumer");
    if (mlir::failed(fuseAssemblySources(rewriter, region, assemblyRequests,
                                         assemblyFusions, statistics,
                                         producerTiling)))
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
      if (mlir::failed(recordIterationCoordinates(rewriter, descriptor, scope,
                                                  tiled->loops)))
        return fail<TemporalTilingStatistics>(
            failure, TemporalTilingFailureKind::CompilerFailure,
            "inner reduction lost its explicit iterator coordinates");
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
      mlir::failed(canonicalizeTiledRegion(region, &liveSources)))
    return fail<TemporalTilingStatistics>(
        failure, TemporalTilingFailureKind::CompilerFailure,
        "bounded temporal canonicalization did not converge");
  auto initializers = lowerUniformTensorInitializers(rewriter, region);
  if (mlir::failed(initializers))
    return fail<TemporalTilingStatistics>(
        failure, TemporalTilingFailureKind::CompilerFailure,
        "uniform tensor initialization could not be materialized");
  statistics.decomposedPads += initializers->pads;
  statistics.decomposedConstantGenerates += initializers->generates;
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
  if (mlir::failed(localizeCurrentAssemblies(rewriter, region, statistics,
                                             producerTiling, failure))) {
    if (failure && failure->kind != TemporalTilingFailureKind::None)
      return mlir::failure();
    return fail<TemporalTilingStatistics>(
        failure, TemporalTilingFailureKind::CompilerFailure,
        "current assembly tile could not be materialized");
  }
  if (mlir::failed(materializeInitializerTiles(rewriter, region)))
    return fail<TemporalTilingStatistics>(
        failure, TemporalTilingFailureKind::CompilerFailure,
        "current initializer tile could not be materialized");
  if (mlir::failed(canonicalizeTiledRegion(region, &liveSources)))
    return fail<TemporalTilingStatistics>(
        failure, TemporalTilingFailureKind::CompilerFailure,
        "current assembly tile canonicalization did not converge");
  mlir::DominanceInfo dominance(region);
  mlir::eliminateCommonSubExpressions(rewriter, dominance, region);
  eraseDeadOperations(rewriter, region);
  if (mlir::failed(mlir::verify(region)))
    return fail<TemporalTilingStatistics>(
        failure, TemporalTilingFailureKind::CompilerFailure,
        "temporal tiling produced invalid current structural IR");
  return statistics;
}

mlir::FailureOr<TemporalTilingStatistics>
applyTemporalTiling(llvm::ArrayRef<TemporalTilingRequest> requests,
                    StructuredMaterializationRelations &relations,
                    TemporalTilingFailure *failure,
                    mlir::RewriterBase::Listener *externalListener) {
  if (failure)
    *failure = {};
  TemporalTilingStatistics total;
  if (requests.empty())
    return total;
  auto first = requests.front().domain.getRegion();
  auto module =
      first ? first->getParentOfType<mlir::ModuleOp>() : mlir::ModuleOp{};
  if (!module || mlir::failed(mlir::verify(module)) ||
      mlir::failed(compiler::detail::checkStructuredBufferRelationsCurrent(
          module, relations)))
    return fail<TemporalTilingStatistics>(
        failure, TemporalTilingFailureKind::BrokenContract,
        "temporal batch requires verified current IR and live relations");
  llvm::DenseSet<mlir::Operation *> regions;
  for (const auto &request : requests) {
    auto region = request.domain.getRegion();
    if (!region || region->getParentOfType<mlir::ModuleOp>() != module ||
        !regions.insert(region.getOperation()).second ||
        !request.domain.contains(request.choice))
      return fail<TemporalTilingStatistics>(
          failure, TemporalTilingFailureKind::BrokenContract,
          "temporal batch requires distinct live regions and unchanged choices "
          "in one Module");
  }
  compiler::detail::StructuredBufferReplacementListener listener(
      relations, externalListener);
  for (const auto &request : requests) {
    auto applied = applyRegionTemporalTiling(request.domain, request.choice,
                                             relations, listener, failure);
    if (mlir::failed(applied))
      return mlir::failure();
    mlir::IRRewriter rewriter(module.getContext(), &listener);
    if (mlir::failed(materializeAttentionVisibility(
            rewriter, request.domain.getRegion())))
      return fail<TemporalTilingStatistics>(
          failure, TemporalTilingFailureKind::CompilerFailure,
          "selected attention visibility could not be materialized");
    for (auto field : {&TemporalTilingStatistics::tiledTraversals,
                       &TemporalTilingStatistics::loops,
                       &TemporalTilingStatistics::specializedTails,
                       &TemporalTilingStatistics::specializedConcatBoundaries,
                       &TemporalTilingStatistics::fusedProducers,
                       &TemporalTilingStatistics::viewTransparentProducers,
                       &TemporalTilingStatistics::tileLocalAssemblies,
                       &TemporalTilingStatistics::assembledSegments,
                       &TemporalTilingStatistics::decomposedPads,
                       &TemporalTilingStatistics::decomposedConstantGenerates})
      total.*field += (*applied).*field;
  }
  if (!listener.finalizeAfterRewrite() || mlir::failed(mlir::verify(module)) ||
      mlir::failed(verifyStructuralTileRegions(module)) ||
      mlir::failed(compiler::detail::checkStructuredBufferRelationsCurrent(
          module, relations)))
    return fail<TemporalTilingStatistics>(
        failure, TemporalTilingFailureKind::CompilerFailure,
        "temporal batch produced invalid current structural IR");
  return total;
}

TensorAssemblyOpportunity
queryLocalTensorAssemblyRead(mlir::tensor::ExtractSliceOp read,
                             const analysis::IndexRelationLimits &limits) {
  if (!read || !read->getParentOfType<TileRegionOp>() ||
      mlir::failed(mlir::verify(read)))
    return {TensorAssemblyOpportunityKind::BrokenContract,
            "local assembly query requires a verified current Region read"};
  auto query = compiler::detail::queryTensorAssemblySlice(read, limits);
  if (!query.isExact())
    return {assemblyOutcome(query.status), std::move(query.detail)};
  if (!queryAssemblySliceReuse(read, /*allowRepeatedReads=*/true))
    return {TensorAssemblyOpportunityKind::Unsupported,
            "assembly read has no supported current placement"};
  return {TensorAssemblyOpportunityKind::Available, {}};
}

mlir::FailureOr<TemporalTilingStatistics> materializeLocalTensorAssemblyReads(
    TileRegionOp region, llvm::ArrayRef<mlir::tensor::ExtractSliceOp> reads,
    StructuredMaterializationRelations &relations,
    TemporalTilingFailure *failure,
    const analysis::IndexRelationLimits &limits) {
  if (failure)
    *failure = {};
  auto module =
      region ? region->getParentOfType<mlir::ModuleOp>() : mlir::ModuleOp{};
  if (!module || mlir::failed(mlir::verify(region)) || reads.empty() ||
      mlir::failed(compiler::detail::checkStructuredBufferRelationsCurrent(
          module, relations)))
    return fail<TemporalTilingStatistics>(
        failure, TemporalTilingFailureKind::BrokenContract,
        "selected assembly reads require verified IR and live relations");
  llvm::SmallVector<mlir::tensor::ExtractSliceOp> selected;
  llvm::SmallVector<AssemblyReadRequest, 4> requests;
  for (auto read : reads) {
    if (!read || read->getParentOfType<TileRegionOp>() != region ||
        llvm::is_contained(selected, read))
      return fail<TemporalTilingStatistics>(
          failure, TemporalTilingFailureKind::BrokenContract,
          "assembly selection contains a foreign or repeated read");
    auto outcome = queryLocalTensorAssemblyRead(read, limits);
    if (outcome.kind != TensorAssemblyOpportunityKind::Available)
      return fail<TemporalTilingStatistics>(
          failure, assemblyFailureKind(outcome.kind), outcome.detail);
    selected.push_back(read);
    if (llvm::none_of(requests, [&](const auto &request) {
          return request.assembledValue == read.getSource();
        }))
      requests.push_back({read.getSource()});
  }
  TemporalTilingStatistics total;
  compiler::detail::StructuredBufferReplacementListener listener(relations);
  mlir::IRRewriter rewriter(region.getContext(), &listener);
  if (mlir::failed(specializeConcatLoopBoundaries(
          rewriter, region, requests, total, &selected, failure, limits))) {
    if (failure && failure->kind != TemporalTilingFailureKind::None)
      return mlir::failure();
    return fail<TemporalTilingStatistics>(
        failure, TemporalTilingFailureKind::CompilerFailure,
        "selected assembly boundary partition failed after preflight");
  }
  // Splitting clones actual reads with IRMapping. Query those live reads anew;
  // never recover a clone by its walk position, source shape or symbol name.
  for (auto read : selected) {
    auto current = queryLocalTensorAssemblyRead(read, limits);
    // A selected ancestor view may already have exposed the original tensor
    // leaf to this selected read. Its current source is authoritative.
    if (current.kind == TensorAssemblyOpportunityKind::NotApplicable)
      continue;
    if (current.kind != TensorAssemblyOpportunityKind::Available)
      return fail<TemporalTilingStatistics>(
          failure, TemporalTilingFailureKind::CompilerFailure,
          "selected assembly read lost its proved representation");
    if (mlir::failed(materializeAssemblySlices(
            rewriter, region, {{read.getSource()}}, total, read)))
      return fail<TemporalTilingStatistics>(
          failure, TemporalTilingFailureKind::CompilerFailure,
          "selected assembly read failed after preflight");
  }
  if (mlir::failed(canonicalizeTiledRegion(region, &listener)) ||
      !listener.finalizeAfterRewrite() || mlir::failed(mlir::verify(module)) ||
      mlir::failed(verifyStructuralTileRegions(module)) ||
      mlir::failed(compiler::detail::checkStructuredBufferRelationsCurrent(
          module, relations)))
    return fail<TemporalTilingStatistics>(
        failure, TemporalTilingFailureKind::CompilerFailure,
        "selected local assembly produced invalid current structural IR");
  return total;
}

} // namespace wafer
