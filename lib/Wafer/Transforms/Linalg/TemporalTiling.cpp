//===- TemporalTiling.cpp - Apply live-operation temporal choices -----===//

#include "TemporalTiling.h"

#include "Wafer/Transforms/Tile/StructuredBufferRelations.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
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
    if (!group.producer || !group.producer.getOwner()->getBlock())
      return reject("joint producer is no longer current");
    llvm::SmallVector<mlir::tensor::ExtractSliceOp, 8> slices;
    region.walk([&](mlir::tensor::ExtractSliceOp slice) {
      if (slice.getSource() == group.producer)
        slices.push_back(slice);
    });
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

mlir::Value asIndexValue(mlir::IRRewriter &rewriter, mlir::Location location,
                         mlir::OpFoldResult value) {
  return mlir::getValueOrCreateConstantIndexOp(rewriter, location, value);
}

mlir::LogicalResult
fuseConcatSlices(mlir::IRRewriter &rewriter, TileRegionOp region,
                 llvm::ArrayRef<ConcatFusionRequest> requests,
                 TemporalTilingStatistics &statistics) {
  for (const ConcatFusionRequest &request : requests) {
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
      mlir::Value assembled = rewriter.create<mlir::tensor::EmptyOp>(
          slice.getLoc(), requestSizes, resultType.getElementType());
      for (const auto &segment : request.segments) {
        auto sourceType =
            mlir::dyn_cast<mlir::RankedTensorType>(segment.source.getType());
        if (!sourceType ||
            segment.offsets.size() !=
                static_cast<size_t>(sourceType.getRank()) ||
            segment.sizes.size() != static_cast<size_t>(sourceType.getRank()))
          return mlir::failure();
        llvm::SmallVector<mlir::OpFoldResult, 4> sourceOffsets;
        llvm::SmallVector<mlir::OpFoldResult, 4> destinationOffsets;
        llvm::SmallVector<mlir::OpFoldResult, 4> intersectionSizes;
        mlir::Value overlaps =
            rewriter.create<mlir::arith::ConstantIntOp>(slice.getLoc(), 1, 1);
        for (auto [requestOffset, requestSize, segmentOffset, segmentSize] :
             llvm::zip_equal(requestOffsets, requestSizes, segment.offsets,
                             segment.sizes)) {
          mlir::Value requestBegin =
              asIndexValue(rewriter, slice.getLoc(), requestOffset);
          mlir::Value requestLength =
              asIndexValue(rewriter, slice.getLoc(), requestSize);
          mlir::Value requestEnd = rewriter.create<mlir::arith::AddIOp>(
              slice.getLoc(), requestBegin, requestLength);
          mlir::Value segmentBegin =
              rewriter.create<mlir::arith::ConstantIndexOp>(slice.getLoc(),
                                                            segmentOffset);
          mlir::Value segmentEnd =
              rewriter.create<mlir::arith::ConstantIndexOp>(
                  slice.getLoc(), segmentOffset + segmentSize);
          mlir::Value begin = rewriter.create<mlir::arith::MaxSIOp>(
              slice.getLoc(), requestBegin, segmentBegin);
          mlir::Value end = rewriter.create<mlir::arith::MinSIOp>(
              slice.getLoc(), requestEnd, segmentEnd);
          mlir::Value dimensionOverlaps = rewriter.create<mlir::arith::CmpIOp>(
              slice.getLoc(), mlir::arith::CmpIPredicate::slt, begin, end);
          overlaps = rewriter.create<mlir::arith::AndIOp>(
              slice.getLoc(), overlaps, dimensionOverlaps);
          sourceOffsets.push_back(rewriter
                                      .create<mlir::arith::SubIOp>(
                                          slice.getLoc(), begin, segmentBegin)
                                      .getResult());
          destinationOffsets.push_back(
              rewriter
                  .create<mlir::arith::SubIOp>(slice.getLoc(), begin,
                                               requestBegin)
                  .getResult());
          intersectionSizes.push_back(
              rewriter.create<mlir::arith::SubIOp>(slice.getLoc(), end, begin)
                  .getResult());
        }
        llvm::SmallVector<mlir::OpFoldResult, 4> strides(
            sourceType.getRank(), rewriter.getIndexAttr(1));
        auto select = rewriter.create<mlir::scf::IfOp>(
            slice.getLoc(), mlir::TypeRange{assembled.getType()}, overlaps,
            /*withElseRegion=*/true);
        rewriter.setInsertionPointToStart(&select.getThenRegion().front());
        auto sourceSlice = rewriter.create<mlir::tensor::ExtractSliceOp>(
            slice.getLoc(), segment.source, sourceOffsets, intersectionSizes,
            strides);
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
            slice.getLoc(), sourcePiece, assembled, destinationOffsets,
            intersectionSizes, strides);
        rewriter.create<mlir::scf::YieldOp>(slice.getLoc(), inserted);
        rewriter.setInsertionPointToStart(&select.getElseRegion().front());
        rewriter.create<mlir::scf::YieldOp>(slice.getLoc(), assembled);
        assembled = select.getResult(0);
        rewriter.setInsertionPointAfter(select);
        ++statistics.assembledSegments;
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

struct JointConsumerTilingResult {
  llvm::SmallVector<mlir::LoopLikeOpInterface, 6> loops;
  unsigned tiledConsumers = 0;
};

mlir::FailureOr<JointConsumerTilingResult> tileJointConsumers(
    mlir::IRRewriter &rewriter, llvm::ArrayRef<mlir::Operation *> consumers,
    llvm::ArrayRef<const compiler::detail::TemporalScopeDescriptor *>
        descriptors,
    llvm::ArrayRef<const compiler::detail::TemporalScopeChoice *> choices) {
  if (consumers.size() < 2 || consumers.size() != descriptors.size() ||
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

mlir::LogicalResult
canonicalizeTiledRegion(TileRegionOp region,
                        mlir::RewriterBase::Listener *listener,
                        bool simplifyPackAndUnpack = false) {
  mlir::RewritePatternSet patterns =
      mlir::linalg::getLinalgTilingCanonicalizationPatterns(
          region.getContext());
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
  llvm::SmallPtrSet<mlir::Operation *, 8> commonLoopConsumers;
  if (choice.kind == compiler::detail::TemporalTraversalKind::Joint) {
    for (const auto &group : domain.getJointProducerGroups()) {
      llvm::SmallVector<mlir::Operation *, 8> consumers;
      for (mlir::OpOperand *operand : group.consumerOperands)
        if (operand && !llvm::is_contained(consumers, operand->getOwner()))
          consumers.push_back(operand->getOwner());
      if (consumers.size() < 2 ||
          llvm::any_of(consumers, [&](mlir::Operation *operation) {
            return !descriptors.count(operation) ||
                   !choicesByOperation.count(operation);
          }))
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
      llvm::SmallVector<const compiler::detail::TemporalScopeDescriptor *, 8>
          groupDescriptors;
      llvm::SmallVector<const compiler::detail::TemporalScopeChoice *, 8>
          groupChoices;
      for (mlir::Operation *operation : consumers) {
        groupDescriptors.push_back(descriptors.lookup(operation));
        groupChoices.push_back(choicesByOperation.lookup(operation));
      }
      mlir::FailureOr<JointConsumerTilingResult> tiled = tileJointConsumers(
          rewriter, consumers, groupDescriptors, groupChoices);
      if (mlir::failed(tiled))
        return fail<TemporalTilingStatistics>(
            failure, TemporalTilingFailureKind::CompilerFailure,
            "compatible joint consumers could not form one common SCF loop");
      for (mlir::Operation *operation : consumers)
        commonLoopConsumers.insert(operation);
      statistics.tiledTraversals += tiled->tiledConsumers;
      statistics.loops += tiled->loops.size();
      if (mlir::failed(specializeRaggedTails(
              rewriter, *groupDescriptors.front(), *groupChoices.front(),
              tiled->loops, statistics)))
        return fail<TemporalTilingStatistics>(
            failure, TemporalTilingFailureKind::CompilerFailure,
            "joint consumer loop could not form its static tail");
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
