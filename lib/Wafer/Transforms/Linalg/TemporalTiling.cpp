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
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

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
      for (const auto &mapping : request.producerDimensions) {
        if (mapping.viewDimension < 0) {
          producerOffsets.push_back(rewriter.getIndexAttr(mapping.offset));
          producerSizes.push_back(rewriter.getIndexAttr(1));
          continue;
        }
        const unsigned dimension = static_cast<unsigned>(mapping.viewDimension);
        if (dimension >= viewOffsets.size() || dimension >= viewSizes.size())
          return mlir::failure();
        producerOffsets.push_back(addConstantOffset(
            rewriter, slice.getLoc(), viewOffsets[dimension], mapping.offset));
        producerSizes.push_back(viewSizes[dimension]);
      }
      auto producerType =
          mlir::dyn_cast<mlir::RankedTensorType>(request.producer.getType());
      auto targetType = mlir::dyn_cast<mlir::RankedTensorType>(slice.getType());
      if (!producerType || !targetType ||
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
      mlir::FailureOr<mlir::Value> replacement = reshapeTile(
          rewriter, slice.getLoc(), tiled->tiledValues.front(), targetType);
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
  for (const auto &descriptor : domain.getScopeDescriptors())
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
  for (const auto &scope : llvm::reverse(choice.scopes)) {
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
    mlir::FailureOr<ExactFusionInventory> exactFusionEdges =
        collectExactFusionEdges(region, failure);
    if (mlir::failed(exactFusionEdges))
      return mlir::failure();
    llvm::SmallVector<ViewFusionRequest, 4> viewFusionRequests;
    for (const auto &path : exactFusionEdges->viewPaths) {
      if (!path.consumerOperand ||
          path.consumerOperand->getOwner() != scope.operation)
        continue;
      viewFusionRequests.push_back({path.producer, path.consumerOperand->get(),
                                    path.producerDimensions});
    }
    llvm::SmallVector<ConcatFusionRequest, 2> concatFusionRequests;
    for (mlir::OpOperand &operand : scope.operation->getOpOperands()) {
      compiler::detail::TemporalConcatQueryResult concat =
          compiler::detail::queryTemporalConcatAssembly(operand);
      if (concat.kind ==
          compiler::detail::TemporalConcatQueryKind::BrokenContract)
        return fail<TemporalTilingStatistics>(
            failure, TemporalTilingFailureKind::BrokenContract, concat.detail);
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
