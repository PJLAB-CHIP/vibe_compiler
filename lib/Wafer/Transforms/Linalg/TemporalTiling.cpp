//===- TemporalTiling.cpp - Apply live-operation temporal choices -----===//

#include "TemporalTiling.h"

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

struct ExactFusionInventory {
  llvm::DenseSet<mlir::Value> directEdges;
  llvm::SmallVector<compiler::detail::TemporalFusionPathResult, 8> viewPaths;
};

mlir::FailureOr<ExactFusionInventory>
collectExactFusionEdges(TileRegionOp region, TemporalTilingFailure *failure) {
  ExactFusionInventory inventory;
  for (mlir::Operation &operation :
       region.getBody().front().without_terminator()) {
    for (mlir::OpResult result : operation.getResults()) {
      compiler::detail::TemporalFusionPathResult query =
          compiler::detail::queryTemporalProducerFusionPath(result);
      if (query.kind ==
          compiler::detail::TemporalFusionQueryKind::BrokenContract)
        return fail<ExactFusionInventory>(
            failure, TemporalTilingFailureKind::BrokenContract, query.detail);
      if (!query.isExact())
        continue;
      if (!query.viewTransparent)
        inventory.directEdges.insert(result);
      else
        inventory.viewPaths.push_back(std::move(query));
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
  if (!sourceType.hasStaticShape() || !targetType.hasStaticShape() ||
      sourceType.getNumElements() != targetType.getNumElements() ||
      sourceType.getRank() == 0 || targetType.getRank() == 0)
    return mlir::failure();

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
                       TemporalTilingStatistics &statistics) {
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
                mlir::tensor::replaceExtractSliceWithTiledProducer(
                    rewriter, sourceSlice, request.producer);
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
          mlir::tensor::replaceExtractSliceWithTiledProducer(
              rewriter, requestSlice, request.producer);
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
        replacement = reshapeTile(rewriter, slice.getLoc(),
                                  tiled->tiledValues.front(), targetType);
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

mlir::LogicalResult fuseJointProducerSlices(
    mlir::IRRewriter &rewriter, TileRegionOp region,
    llvm::ArrayRef<compiler::detail::TemporalJointProducerGroup> groups,
    TemporalTilingStatistics &statistics, std::string &detail) {
  auto reject = [&](llvm::StringRef message) {
    detail = message.str();
    return mlir::failure();
  };
  for (const auto &group : groups) {
    if (!group.producer || !group.consumerValue ||
        !group.producer.getOwner()->getBlock())
      return reject("joint producer is no longer current");
    llvm::SmallVector<mlir::tensor::ExtractSliceOp, 8> slices;
    region.walk([&](mlir::tensor::ExtractSliceOp slice) {
      if (slice.getSource() == group.consumerValue)
        slices.push_back(slice);
    });
    if (slices.empty() && !group.consumerValue.use_empty())
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
    if (group.isViewTransparent()) {
      for (const auto &requestGroup : equivalentRequests) {
        mlir::tensor::ExtractSliceOp representative = requestGroup.front();
        for (mlir::tensor::ExtractSliceOp duplicate :
             llvm::drop_begin(requestGroup))
          rewriter.replaceOp(duplicate, representative.getResult());
      }
      ViewFusionRequest request{
          group.producer, group.consumerValue, group.producerDimensions,
          group.consumerViewToProducer ? &*group.consumerViewToProducer
                                       : nullptr};
      if (mlir::failed(
              fuseViewProducerSlices(rewriter, region, {request}, statistics)))
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
          mlir::tensor::replaceExtractSliceWithTiledProducer(rewriter, request,
                                                             group.producer);
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
  if (!lower || !upper || !step || *lower < 0 || *upper <= *lower ||
      *step <= 0)
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
    if (segment.offsets.size() != static_cast<size_t>(assembledType.getRank()) ||
        segment.sizes.size() != static_cast<size_t>(assembledType.getRank()))
      return std::nullopt;
    for (unsigned dimension = 0; dimension < assembledType.getRank();
         ++dimension) {
      const bool full = segment.offsets[dimension] == 0 &&
                        segment.sizes[dimension] ==
                            assembledType.getDimSize(dimension);
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

mlir::FailureOr<mlir::scf::ForOp>
splitForLoopAt(mlir::IRRewriter &rewriter, mlir::scf::ForOp loop,
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
  mlir::Value splitValue = rewriter.create<mlir::arith::ConstantIndexOp>(
      loop.getLoc(), split);
  rewriter.setInsertionPointAfter(loop);
  auto suffix = mlir::cast<mlir::scf::ForOp>(rewriter.clone(*loop));
  rewriter.modifyOpInPlace(
      suffix, [&] { suffix.getLowerBoundMutable().assign(splitValue); });
  rewriter.replaceAllUsesWith(loop.getResults(), suffix.getResults());
  rewriter.modifyOpInPlace(suffix, [&] {
    suffix.getInitArgsMutable().assign(loop.getResults());
  });
  rewriter.modifyOpInPlace(
      loop, [&] { loop.getUpperBoundMutable().assign(splitValue); });
  return suffix;
}

mlir::LogicalResult specializeConcatLoopBoundaries(
    mlir::IRRewriter &rewriter, TileRegionOp region,
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

mlir::LogicalResult
fuseConcatSlices(mlir::IRRewriter &rewriter, TileRegionOp region,
                 llvm::ArrayRef<ConcatFusionRequest> requests,
                 TemporalTilingStatistics &statistics) {
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
        mlir::Value upper = rewriter.create<mlir::arith::ConstantIndexOp>(
            slice.getLoc(), last);
        mlir::Value atLeast = rewriter.create<mlir::arith::CmpIOp>(
            slice.getLoc(), mlir::arith::CmpIPredicate::sge, grid->induction,
            lower);
        mlir::Value atMost = rewriter.create<mlir::arith::CmpIOp>(
            slice.getLoc(), mlir::arith::CmpIPredicate::sle, grid->induction,
            upper);
        return rewriter.create<mlir::arith::AndIOp>(slice.getLoc(), atLeast,
                                                    atMost)
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
          sourceOffsets.push_back(dimension == axis
                                      ? sourceAxisOffset
                                      : requestOffsets[dimension]);
          destinationOffsets.push_back(
              dimension == axis ? destinationAxisOffset
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
              mlir::tensor::replaceExtractSliceWithTiledProducer(
                  rewriter, sourceSlice, *segment.derivedProducer);
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
                ? getGridFloor(*grid, std::min(lastFullCoordinate,
                                               grid->upper - 1))
                : std::nullopt;
        const bool hasFullRange = firstFull.has_value() &&
                                  lastFull.has_value() &&
                                  firstFull.value_or(0) <= lastFull.value_or(0);
        const int64_t firstFullValue = firstFull.value_or(0);
        const int64_t lastFullValue = lastFull.value_or(0);
        if (hasFullRange) {
          std::optional<int64_t> lastGridValue =
              getGridFloor(*grid, grid->upper - 1);
          const bool coversCompleteLoop =
              firstFullValue == grid->lower && lastGridValue &&
              lastFullValue == *lastGridValue;
          mlir::Value condition =
              coversCompleteLoop
                  ? mlir::Value{}
                  : createConditionForRange(firstFullValue, lastFullValue);
          mlir::OpFoldResult sourceOffset =
              firstFullValue == lastFullValue && !coversCompleteLoop
                  ? mlir::OpFoldResult(
                        rewriter.getIndexAttr(firstFullValue - segmentBegin))
                  : addConstantOffset(rewriter, slice.getLoc(),
                                      grid->induction, -segmentBegin);
          if (mlir::failed(emitPiece(segment, sourceOffset,
                                     rewriter.getIndexAttr(0), requestLength,
                                     condition)))
            return mlir::failure();
        }

        llvm::SmallVector<int64_t, 2> boundaryBegins;
        if (std::optional<int64_t> begin = getGridFloor(*grid, segmentBegin))
          boundaryBegins.push_back(*begin);
        if (std::optional<int64_t> begin =
                getGridFloor(*grid, segmentEnd - 1))
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
          mlir::Value condition =
              createConditionForExactBegin(requestBegin);
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
    if (!padding || !resultType.hasStaticShape())
      continue;
    rewriter.setInsertionPoint(pad);
    mlir::Value empty = rewriter.create<mlir::tensor::EmptyOp>(
        pad.getLoc(), resultType.getShape(), resultType.getElementType());
    mlir::Value filled =
        rewriter.create<mlir::linalg::FillOp>(pad.getLoc(), padding, empty)
            .getResult(0);
    llvm::SmallVector<mlir::OpFoldResult, 4> sourceSizes =
        mlir::tensor::getMixedSizes(rewriter, pad.getLoc(), pad.getSource());
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
    if (!resultType.hasStaticShape() || !yield)
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
        operation.getLoc(), resultType.getShape(), resultType.getElementType());
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
    if (!operation || !mlir::isa<mlir::linalg::LinalgOp>(operation) ||
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

mlir::LogicalResult materializeBroadcastProducer(
    mlir::IRRewriter &rewriter, TileRegionOp region,
    const compiler::detail::TemporalBroadcastFusion &fusion,
    const compiler::detail::TemporalScopeDescriptor &descriptor,
    const compiler::detail::TemporalScopeChoice &choice,
    llvm::ArrayRef<mlir::LoopLikeOpInterface> loops,
    TemporalTilingStatistics &statistics, std::string &detail) {
  auto reject = [&](llvm::StringRef message) {
    detail = message.str();
    return mlir::failure();
  };
  if (!fusion.producer || loops.empty() ||
      choice.loopOrder.size() != loops.size())
    return reject("broadcast fusion no longer matches its current traversal");
  llvm::SmallVector<mlir::tensor::ExtractSliceOp, 4> consumerSlices;
  region.walk([&](mlir::tensor::ExtractSliceOp slice) {
    if (slice.getSource() == fusion.producer)
      consumerSlices.push_back(slice);
  });
  if (consumerSlices.empty())
    return reject("broadcast consumer emitted no producer slice");

  llvm::SmallBitVector invariant(descriptor.iterationExtents.size(), false);
  for (uint32_t dimension : fusion.invariantConsumerDimensions) {
    if (dimension >= invariant.size())
      return reject("broadcast invariant dimension is outside its traversal");
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

  llvm::SmallVector<mlir::OpFoldResult, 4> producerOffsets;
  llvm::SmallVector<mlir::OpFoldResult, 4> producerSizes;
  mlir::Location location = fusion.producer.getLoc();
  for (mlir::AffineExpr expression : fusion.consumerOperandMap.getResults()) {
    auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
    if (!dimension ||
        dimension.getPosition() >= descriptor.iterationExtents.size())
      return reject("broadcast operand map is no longer a projection");
    const unsigned iterator = dimension.getPosition();
    const int64_t extent = descriptor.iterationExtents[iterator];
    const int64_t tileSize = choice.iteratorTileSizes[iterator];
    if (tileSize >= extent) {
      producerOffsets.push_back(rewriter.getIndexAttr(0));
      producerSizes.push_back(rewriter.getIndexAttr(extent));
      continue;
    }
    auto found = llvm::find(choice.loopOrder, iterator);
    if (found == choice.loopOrder.end())
      return reject("broadcast dependent iterator has no materialized loop");
    const size_t loopPosition = found - choice.loopOrder.begin();
    if (loopPosition >= firstInvariantLoop)
      return reject("broadcast producer would be placed above a dependency");
    mlir::LoopLikeOpInterface loopInterface = loops[loopPosition];
    auto loop = mlir::dyn_cast<mlir::scf::ForOp>(loopInterface.getOperation());
    if (!loop)
      return reject("broadcast fusion requires the selected SCF loop");
    mlir::Value induction = loop.getInductionVar();
    producerOffsets.push_back(induction);
    mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, rewriter.getContext());
    mlir::AffineMap bounded = mlir::AffineMap::get(
        1, 0,
        {mlir::getAffineConstantExpr(tileSize, rewriter.getContext()),
         mlir::getAffineConstantExpr(extent, rewriter.getContext()) - d0},
        rewriter.getContext());
    producerSizes.push_back(mlir::affine::makeComposedFoldedAffineMin(
        rewriter, location, bounded, {induction}));
  }
  auto producerType =
      mlir::dyn_cast<mlir::RankedTensorType>(fusion.producer.getType());
  if (!producerType ||
      producerOffsets.size() != static_cast<size_t>(producerType.getRank()))
    return reject("broadcast producer rank no longer matches its operand map");
  llvm::SmallVector<mlir::OpFoldResult, 4> strides(producerType.getRank(),
                                                   rewriter.getIndexAttr(1));
  auto request = rewriter.create<mlir::tensor::ExtractSliceOp>(
      location, fusion.producer, producerOffsets, producerSizes, strides);
  mlir::FailureOr<mlir::TilingResult> tiled =
      mlir::tensor::replaceExtractSliceWithTiledProducer(rewriter, request,
                                                         fusion.producer);
  if (mlir::failed(tiled) || tiled->tiledValues.size() != 1) {
    rewriter.eraseOp(request);
    return reject("broadcast producer rejected its exact projected tile");
  }
  mlir::Value sharedTile = tiled->tiledValues.front();
  rewriter.eraseOp(request);
  for (mlir::tensor::ExtractSliceOp slice : consumerSlices) {
    auto targetType = mlir::dyn_cast<mlir::RankedTensorType>(slice.getType());
    if (!targetType)
      return reject("broadcast consumer slice is not a ranked tensor");
    mlir::FailureOr<mlir::Value> replacement =
        reshapeTile(rewriter, slice.getLoc(), sharedTile, targetType);
    if (mlir::failed(replacement))
      return reject("broadcast producer tile does not match its consumer");
    rewriter.replaceOp(slice, *replacement);
  }
  if (!fusion.producer.use_empty())
    return reject("broadcast producer still has an unfused current use");
  rewriter.eraseOp(fusion.producer.getOwner());
  ++statistics.fusedProducers;
  return mlir::success();
}

mlir::LogicalResult materializeWindowProducers(
    mlir::IRRewriter &rewriter, TileRegionOp region,
    llvm::ArrayRef<const compiler::detail::TemporalWindowFusion *> fusions,
    TemporalTilingStatistics &statistics, std::string &detail) {
  for (const compiler::detail::TemporalWindowFusion *fusion : fusions) {
    if (!fusion || !fusion->producer)
      return mlir::failure();
    compiler::detail::TemporalJointProducerGroup group;
    group.producer = fusion->producer;
    group.consumerValue = fusion->producer;
    if (mlir::failed(fuseJointProducerSlices(rewriter, region, {group},
                                             statistics, detail)))
      return mlir::failure();
  }
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
  return mlir::success();
}

struct OnlineConsumerRequest {
  LinalgExtOnlineAttentionOp producer;
  mlir::linalg::GenericOp consumer;
  compiler::detail::TemporalScopeDescriptor producerDescriptor;
  compiler::detail::TemporalScopeDescriptor consumerDescriptor;
  compiler::detail::TemporalScopeChoice producerChoice;
  compiler::detail::TemporalScopeChoice consumerChoice;
  llvm::SmallVector<mlir::AffineMap, 3> resultMaps;
};

std::optional<OnlineConsumerRequest> getOnlineConsumerRequest(
    LinalgExtOnlineAttentionOp producer,
    const llvm::DenseMap<mlir::Operation *,
                         const compiler::detail::TemporalScopeDescriptor *>
        &descriptors,
    const llvm::DenseMap<mlir::Operation *,
                         const compiler::detail::TemporalScopeChoice *>
        &choices) {
  mlir::Operation *soleConsumer = nullptr;
  for (mlir::Operation *user : producer->getUsers()) {
    if (soleConsumer && soleConsumer != user)
      return std::nullopt;
    soleConsumer = user;
  }
  auto consumer = mlir::dyn_cast_or_null<mlir::linalg::GenericOp>(soleConsumer);
  if (!consumer || !consumer.hasPureTensorSemantics() ||
      consumer.getNumReductionLoops() != 0 ||
      consumer->getBlock() != producer->getBlock() ||
      !producer->isBeforeInBlock(consumer) || !choices.count(consumer) ||
      !choices.count(producer))
    return std::nullopt;
  for (mlir::OpOperand *input : consumer.getDpsInputOperands()) {
    auto result = mlir::dyn_cast<mlir::OpResult>(input->get());
    if (!result || result.getOwner() != producer)
      return std::nullopt;
  }
  for (unsigned index = 0; index < consumer.getNumDpsInits(); ++index)
    if (consumer.payloadUsesValueFromOperand(consumer.getDpsInitOperand(index)))
      return std::nullopt;
  auto producerDps =
      mlir::cast<mlir::DestinationStyleOpInterface>(producer.getOperation());
  for (mlir::Value init : producerDps.getDpsInits()) {
    auto fill = init.getDefiningOp<mlir::linalg::FillOp>();
    if (!fill || !init.hasOneUse())
      return std::nullopt;
  }
  OnlineConsumerRequest request{producer,
                                consumer,
                                *descriptors.lookup(producer),
                                *descriptors.lookup(consumer),
                                *choices.lookup(producer),
                                *choices.lookup(consumer),
                                {producer.getAccumulatorMap(),
                                 producer.getMaximumMap(),
                                 producer.getSumMap()}};
  const auto &producerExtents = request.producerDescriptor.iterationExtents;
  const auto &consumerExtents = request.consumerDescriptor.iterationExtents;
  llvm::SmallVector<int, 6> producerToConsumer(producerExtents.size(), -1);
  llvm::SmallVector<int, 6> consumerToProducer(consumerExtents.size(), -1);
  for (mlir::OpOperand *input : consumer.getDpsInputOperands()) {
    auto result = mlir::cast<mlir::OpResult>(input->get());
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
  auto coupled = producer.getCoupledReductionDescription();
  for (unsigned c : request.consumerChoice.loopOrder) {
    int p = consumerToProducer[c];
    if (p < 0 || iterators[p] != mlir::utils::IteratorType::parallel ||
        coupled.hasReplicatedComponent(p) ||
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

mlir::LogicalResult tileOnlineConsumer(mlir::IRRewriter &rewriter,
                                       TileRegionOp region,
                                       OnlineConsumerRequest request,
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
  llvm::SmallVector<mlir::linalg::GenericOp, 4> consumers;
  region.walk([&](mlir::linalg::GenericOp generic) {
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
  for (mlir::linalg::GenericOp current : consumers) {
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
        return mlir::failure();
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
        tiled->tiledValues.size() != 3)
      return mlir::failure();
    auto online =
        mlir::cast<LinalgExtOnlineAttentionOp>(tiled->tiledOps.front());
    auto dps =
        mlir::cast<mlir::DestinationStyleOpInterface>(online.getOperation());
    auto originalDps =
        mlir::cast<mlir::DestinationStyleOpInterface>(producer.getOperation());
    rewriter.setInsertionPoint(online);
    for (unsigned index = 0; index < dps.getNumDpsInits(); ++index) {
      auto fill = originalDps.getDpsInitOperand(index)
                      ->get()
                      .getDefiningOp<mlir::linalg::FillOp>();
      auto type = mlir::cast<mlir::RankedTensorType>(
          dps.getDpsInitOperand(index)->get().getType());
      if (!type.hasStaticShape())
        return mlir::failure();
      auto empty = rewriter.create<mlir::tensor::EmptyOp>(
          online.getLoc(), type.getShape(), type.getElementType(),
          type.getEncoding());
      auto local = rewriter.create<mlir::linalg::FillOp>(
          online.getLoc(), fill.getInputs()[0], empty.getResult());
      rewriter.modifyOpInPlace(online, [&] {
        dps.getDpsInitOperand(index)->set(local.getResult(0));
      });
    }
    rewriter.modifyOpInPlace(current, [&] {
      for (mlir::OpOperand *input : current.getDpsInputOperands()) {
        auto slice = input->get().getDefiningOp<mlir::tensor::ExtractSliceOp>();
        auto result = mlir::cast<mlir::OpResult>(slice.getSource());
        input->set(online.getResult(result.getResultNumber()));
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
      rewriter.setInsertionPoint(online);
      auto inner = mlir::scf::tileUsingSCF(
          rewriter, mlir::cast<mlir::TilingInterface>(online.getOperation()),
          reductionOptions);
      if (mlir::failed(inner))
        return mlir::failure();
      rewriter.replaceOp(online, inner->replacements);
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
  mlir::IRRewriter rewriter(region.getContext(), &listener);
  TemporalTilingStatistics statistics;
  llvm::DenseMap<mlir::Operation *,
                 const compiler::detail::TemporalScopeChoice *>
      choicesByOperation;
  for (const auto &scope : choice.scopes)
    choicesByOperation.try_emplace(scope.operation, &scope);
  llvm::DenseMap<
      mlir::Operation *,
      llvm::SmallVector<const compiler::detail::TemporalBroadcastFusion *, 2>>
      broadcastByConsumer;
  llvm::DenseMap<
      mlir::Operation *,
      llvm::SmallVector<const compiler::detail::TemporalWindowFusion *, 2>>
      windowByConsumer;
  if (choice.kind == compiler::detail::TemporalTraversalKind::Joint) {
    for (const auto &fusion : domain.getBroadcastFusions()) {
      if (!fusion.consumerOperand)
        return fail<TemporalTilingStatistics>(
            failure, TemporalTilingFailureKind::BrokenContract,
            "broadcast fusion lost its current consumer operand");
      broadcastByConsumer[fusion.consumerOperand->getOwner()].push_back(
          &fusion);
    }
    for (const auto &fusion : domain.getWindowFusions()) {
      if (!fusion.consumerOperand)
        return fail<TemporalTilingStatistics>(
            failure, TemporalTilingFailureKind::BrokenContract,
            "window fusion lost its current consumer operand");
      windowByConsumer[fusion.consumerOperand->getOwner()].push_back(&fusion);
    }
  }
  llvm::SmallPtrSet<mlir::Operation *, 8> commonLoopConsumers;
  for (const auto &scope : choice.scopes) {
    if (commonLoopConsumers.contains(scope.operation))
      continue;
    auto online = mlir::dyn_cast<LinalgExtOnlineAttentionOp>(scope.operation);
    if (!online)
      continue;
    auto request =
        getOnlineConsumerRequest(online, descriptors, choicesByOperation);
    if (!request)
      continue;
    commonLoopConsumers.insert(request->producer);
    commonLoopConsumers.insert(request->consumer);
    if (mlir::failed(
            tileOnlineConsumer(rewriter, region, *request, statistics)))
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
      struct BroadcastApply {
        const compiler::detail::TemporalBroadcastFusion *fusion = nullptr;
        const compiler::detail::TemporalScopeDescriptor *descriptor = nullptr;
        const compiler::detail::TemporalScopeChoice *choice = nullptr;
      };
      llvm::SmallVector<BroadcastApply, 4> broadcastFusions;
      llvm::SmallVector<const compiler::detail::TemporalWindowFusion *, 4>
          windowFusions;
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
        for (const auto *fusion : broadcastByConsumer.lookup(operation))
          broadcastFusions.push_back({fusion, descriptor, scopeChoice});
        llvm::append_range(windowFusions, windowByConsumer.lookup(operation));
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
      for (const BroadcastApply &broadcast : broadcastFusions) {
        std::string detail;
        if (mlir::failed(materializeBroadcastProducer(
                rewriter, region, *broadcast.fusion, *broadcast.descriptor,
                *broadcast.choice, tiled->loops, statistics, detail))) {
          manualFailure =
              "broadcast producer could not be placed outside its invariant "
              "loops: " +
              detail;
          return mlir::failure();
        }
      }
      if (!windowFusions.empty()) {
        std::string detail;
        if (mlir::failed(materializeWindowProducers(
                rewriter, region, windowFusions, statistics, detail))) {
          manualFailure =
              "disjoint window producer did not materialize from its actual "
              "consumer slices: " +
              detail;
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

    for (const auto &group : domain.getJointProducerGroups()) {
      llvm::SmallVector<mlir::Operation *, 8> consumers;
      for (mlir::OpOperand *operand : group.consumerOperands)
        if (operand && choicesByOperation.count(operand->getOwner()) &&
            !llvm::is_contained(consumers, operand->getOwner()))
          consumers.push_back(operand->getOwner());
      if (consumers.size() < 2)
        continue;
      if (!llvm::any_of(consumers, hasActiveDimension))
        continue;
      const bool anyAlreadyGrouped =
          llvm::any_of(consumers, [&](mlir::Operation *operation) {
            return commonLoopConsumers.contains(operation);
          });
      const bool allAlreadyGrouped =
          llvm::all_of(consumers, [&](mlir::Operation *operation) {
            return commonLoopConsumers.contains(operation);
          });
      if (anyAlreadyGrouped && !allAlreadyGrouped)
        return fail<TemporalTilingStatistics>(
            failure, TemporalTilingFailureKind::BrokenContract,
            "joint producer groups overlap only part of a consumer set");
      if (allAlreadyGrouped)
        continue;
      if (mlir::failed(tileManualConsumers(consumers)))
        return fail<TemporalTilingStatistics>(
            failure, TemporalTilingFailureKind::CompilerFailure,
            "all-use producer group: " + manualFailure);
      for (mlir::Operation *operation : consumers)
        commonLoopConsumers.insert(operation);
    }
    for (const auto &scope : choice.scopes) {
      mlir::Operation *consumer = scope.operation;
      if (!broadcastByConsumer.count(consumer))
        continue;
      if (commonLoopConsumers.contains(consumer))
        continue;
      if (!hasActiveDimension(consumer))
        continue;
      llvm::SmallVector<mlir::Operation *, 1> singleton{consumer};
      if (mlir::failed(tileManualConsumers(singleton)))
        return fail<TemporalTilingStatistics>(
            failure, TemporalTilingFailureKind::CompilerFailure,
            "broadcast consumer: " + manualFailure);
      commonLoopConsumers.insert(consumer);
    }
  }
  for (const auto &scope : llvm::reverse(choice.scopes)) {
    if (commonLoopConsumers.contains(scope.operation))
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
    if (choice.kind == compiler::detail::TemporalTraversalKind::Joint)
      for (const auto *fusion : windowByConsumer.lookup(scope.operation))
        if (fusion->producer && fusion->producer.getOwner()->getBlock())
          exactFusionEdges->directEdges.insert(fusion->producer);
    llvm::SmallVector<ViewFusionRequest, 4> viewFusionRequests;
    for (const auto &path : exactFusionEdges->viewPaths) {
      if (!path.consumerOperand ||
          path.consumerOperand->getOwner() != scope.operation)
        continue;
      viewFusionRequests.push_back(
          {path.producer, path.consumerOperand->get(), path.producerDimensions,
           path.consumerViewToProducer ? &*path.consumerViewToProducer
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
    if (mlir::failed(specializeConcatLoopBoundaries(
            rewriter, region, concatFusionRequests, statistics)))
      return fail<TemporalTilingStatistics>(
          failure, TemporalTilingFailureKind::CompilerFailure,
          "static concat boundary could not specialize its canonical loop");
    if (mlir::failed(fuseViewProducerSlices(rewriter, region,
                                            viewFusionRequests, statistics)))
      return fail<TemporalTilingStatistics>(
          failure, TemporalTilingFailureKind::CompilerFailure,
          "exact view-derived producer could not be tiled into its consumer");
    if (mlir::failed(fuseConcatSlices(rewriter, region, concatFusionRequests,
                                      statistics)))
      return fail<TemporalTilingStatistics>(
          failure, TemporalTilingFailureKind::CompilerFailure,
          "exact insert assembly could not form one tile-local value");
  }

  if (choice.kind == compiler::detail::TemporalTraversalKind::Joint &&
      !domain.getJointProducerGroups().empty()) {
    std::string jointFailure;
    if (mlir::failed(fuseJointProducerSlices(rewriter, region,
                                             domain.getJointProducerGroups(),
                                             statistics, jointFailure)))
      return fail<TemporalTilingStatistics>(
          failure, TemporalTilingFailureKind::CompilerFailure,
          "all-use joint producer did not materialize once in its common "
          "loop: " +
              jointFailure);
  }

  canonicalizeLoopBoundMinMax(rewriter, region);
  if (mlir::failed(canonicalizeTiledRegion(region, &listener)) ||
      mlir::failed(refineLinalgStaticTypes(rewriter, region)) ||
      mlir::failed(refineOnlineAttentionStaticTypes(rewriter, region)) ||
      mlir::failed(refinePackUnPackStaticTypes(rewriter, region)) ||
      mlir::failed(canonicalizeTiledRegion(region, &listener,
                                           /*simplifyPackAndUnpack=*/true)) ||
      mlir::failed(refineLinalgStaticTypes(rewriter, region)) ||
      mlir::failed(refineOnlineAttentionStaticTypes(rewriter, region)) ||
      mlir::failed(lowerConstantPads(rewriter, region, statistics)) ||
      mlir::failed(lowerConstantGenerates(rewriter, region, statistics)) ||
      mlir::failed(canonicalizeTiledRegion(region, &listener)))
    return fail<TemporalTilingStatistics>(
        failure, TemporalTilingFailureKind::CompilerFailure,
        "bounded temporal canonicalization did not converge");
  if (mlir::failed(mlir::verify(region)))
    return fail<TemporalTilingStatistics>(
        failure, TemporalTilingFailureKind::CompilerFailure,
        "temporal tiling was invalid before final common-subexpression "
        "elimination");
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
