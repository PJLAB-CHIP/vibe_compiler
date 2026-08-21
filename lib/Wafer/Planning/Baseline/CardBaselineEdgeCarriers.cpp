//===- CardBaselineEdgeCarriers.cpp ----------------------------------===//

#include "Wafer/Planning/Baseline/CardBaselineEdgeCarriers.h"

#include "Wafer/Analysis/Structured/StructuredOperationTileFootprint.h"
#include "Wafer/Conversion/WaferTensorProgramToTileRegion/DependentDataflow.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>

namespace wafer::compiler::detail {
namespace {

uint64_t saturatingMultiply(uint64_t lhs, uint64_t rhs) {
  const unsigned __int128 product = static_cast<unsigned __int128>(lhs) * rhs;
  return product > std::numeric_limits<uint64_t>::max()
             ? std::numeric_limits<uint64_t>::max()
             : static_cast<uint64_t>(product);
}

mlir::FailureOr<llvm::SmallVector<analysis::StaticRectangularIndexSet, 8>>
getRectangles(const analysis::ExactIndexSet &set) {
  if (set.getForm() == analysis::ExactIndexSetForm::BoxUnion)
    return llvm::SmallVector<analysis::StaticRectangularIndexSet, 8>(
        set.getBoxes().begin(), set.getBoxes().end());
  analysis::IndexSetResult exact{
      analysis::IndexRelationStatus::Exact, set.getPresburgerSet(), {}};
  analysis::StaticRectangularIndexSetPiecesResult pieces =
      exact.getExactStaticRectangularDisjuncts();
  if (!pieces.isExact()) {
    analysis::StaticRectangularIndexSetResult rectangle =
        exact.getExactStaticRectangularDomain();
    if (!rectangle.isExact())
      return mlir::failure();
    pieces.domains.push_back(std::move(*rectangle.domain));
  }
  return std::move(pieces.domains);
}

llvm::SmallVector<analysis::StaticRectangularIndexSet, 8>
splitAtWaveBoundaries(const analysis::StaticRectangularIndexSet &box,
                      llvm::ArrayRef<int64_t> waveSizes) {
  llvm::SmallVector<analysis::StaticRectangularIndexSet, 8> fragments{box};
  if (box.offsets.size() != waveSizes.size() ||
      box.sizes.size() != waveSizes.size())
    return {};
  for (size_t dimension = 0; dimension < waveSizes.size(); ++dimension) {
    const int64_t wave = waveSizes[dimension];
    if (wave <= 0)
      return {};
    llvm::SmallVector<analysis::StaticRectangularIndexSet, 8> next;
    for (const analysis::StaticRectangularIndexSet &fragment : fragments) {
      int64_t remaining = fragment.sizes[dimension];
      int64_t offset = fragment.offsets[dimension];
      while (remaining > 0) {
        analysis::StaticRectangularIndexSet part = fragment;
        const int64_t withinWave = offset % wave;
        const int64_t size = std::min(remaining, wave - withinWave);
        part.offsets[dimension] = offset;
        part.sizes[dimension] = size;
        next.push_back(std::move(part));
        if (next.size() > 100000)
          return {};
        offset += size;
        remaining -= size;
      }
    }
    fragments = std::move(next);
  }
  return fragments;
}

const analysis::SourceDemand *
findSource(const analysis::DestinationDemand &destination,
           const SemanticRootKey &producer, uint32_t result) {
  auto source = llvm::find_if(
      destination.sources, [&](const analysis::SourceDemand &candidate) {
        const auto *structured =
            std::get_if<analysis::StructuredResultSource>(&candidate.source);
        return structured && structured->root == producer &&
               structured->result == result;
      });
  return source == destination.sources.end() ? nullptr : &*source;
}

const analysis::FinalResultOwner *
findOwner(const analysis::ExactDemandProof &proof,
          const analysis::StructuredResultSource &source,
          const analysis::OwnerIntersection &intersection) {
  auto owner = llvm::find_if(
      proof.finalOwners, [&](const analysis::FinalResultOwner &candidate) {
        return candidate.root == source.root &&
               candidate.result == source.result &&
               candidate.tile == intersection.tile &&
               candidate.shard == intersection.ownerShard &&
               candidate.reductionGroup == intersection.reductionGroup;
      });
  return owner == proof.finalOwners.end() ? nullptr : &*owner;
}

bool updateBounds(SpatialEdgeStrategy &strategy,
                  const analysis::StaticRectangularIndexSet &piece) {
  if (strategy.producerOffsets.empty()) {
    strategy.producerOffsets = piece.offsets;
    strategy.producerSizes = piece.sizes;
    return true;
  }
  if (piece.offsets.size() != strategy.producerOffsets.size())
    return false;
  for (size_t dimension = 0; dimension < piece.offsets.size(); ++dimension) {
    const int64_t currentLimit =
        strategy.producerOffsets[dimension] + strategy.producerSizes[dimension];
    const int64_t pieceLimit =
        piece.offsets[dimension] + piece.sizes[dimension];
    const int64_t offset =
        std::min(strategy.producerOffsets[dimension], piece.offsets[dimension]);
    const int64_t limit = std::max(currentLimit, pieceLimit);
    strategy.producerOffsets[dimension] = offset;
    strategy.producerSizes[dimension] = limit - offset;
  }
  return true;
}

mlir::LogicalResult appendCarrier(
    TileMapping &mapping, const StructuredDAGEdge &edge,
    const StructuredDAGNode &producer, const StructuredDAGNode &consumer,
    const SemanticRootKey &producerRoot, const SemanticRootKey &consumerRoot,
    const analysis::DependencyDemand &dependency,
    const analysis::ExactDemandProof &proof, std::string *failureReason) {
  auto producerType = mlir::dyn_cast<mlir::RankedTensorType>(
      producer.operation->getResult(edge.producerResult).getType());
  if (!producerType || !producerType.hasStaticShape() ||
      !producerType.getElementType().isIntOrFloat()) {
    if (failureReason)
      *failureReason = "producer result is not a static numeric tensor";
    return mlir::failure();
  }
  const unsigned elementBits =
      producerType.getElementType().getIntOrFloatBitWidth();
  if (elementBits == 0 || elementBits % 8 != 0)
    return mlir::failure();
  auto temporal = llvm::find_if(mapping.operationTemporalTiles,
                                [&](const StructuredOpTemporalTile &tile) {
                                  return tile.operation == producer.operation;
                                });
  std::optional<llvm::SmallVector<int64_t, 4>> waveSizes =
      temporal != mapping.operationTemporalTiles.end()
          ? getStructuredResultTileShape(producer.operation,
                                         edge.producerResult,
                                         temporal->iteratorTileSizes)
          : std::nullopt;
  if (!waveSizes)
    return mlir::failure();

  int64_t payloadSlice = 0;
  for (const analysis::DestinationDemand &destination :
       dependency.perDestination) {
    const analysis::SourceDemand *source =
        findSource(destination, producerRoot, edge.producerResult);
    if (!source || source->requiredDomain.isEmpty())
      continue;
    mlir::FailureOr<llvm::SmallVector<analysis::StaticRectangularIndexSet, 8>>
        demandPieces = getRectangles(source->requiredDomain);
    if (mlir::failed(demandPieces) || demandPieces->empty()) {
      if (failureReason)
        *failureReason = "producer demand has no finite rectangle union";
      return mlir::failure();
    }

    SpatialEdgeStrategy strategy;
    strategy.producer = producer.operation;
    strategy.producerResult = edge.producerResult;
    strategy.consumer = consumer.operation;
    strategy.consumerOperand = edge.consumerOperand;
    strategy.destinationTile = destination.destinationTile;
    strategy.sourceTile = destination.destinationTile;
    strategy.action = SpatialEdgeAction::PeerFragments;
    strategy.fragmentsDefineProducerDemand = demandPieces->size() > 1;
    for (const analysis::StaticRectangularIndexSet &piece : *demandPieces)
      if (!updateBounds(strategy, piece))
        return mlir::failure();

    for (const analysis::OwnerIntersection &intersection :
         source->eligibleFinalOwners) {
      const auto *structured =
          std::get_if<analysis::StructuredResultSource>(&source->source);
      const analysis::FinalResultOwner *owner =
          structured ? findOwner(proof, *structured, intersection) : nullptr;
      if (!owner)
        return mlir::failure();
      mlir::FailureOr<llvm::SmallVector<analysis::StaticRectangularIndexSet, 8>>
          ownedPieces = getRectangles(intersection.domain);
      if (mlir::failed(ownedPieces))
        return mlir::failure();
      for (const analysis::StaticRectangularIndexSet &piece : *ownedPieces) {
        llvm::SmallVector<analysis::StaticRectangularIndexSet, 8> fragments =
            splitAtWaveBoundaries(piece, *waveSizes);
        if (fragments.empty())
          return mlir::failure();
        for (analysis::StaticRectangularIndexSet &fragment : fragments) {
          const bool resident =
              intersection.tile == destination.destinationTile;
          uint64_t elements = 1;
          for (int64_t size : fragment.sizes)
            elements = saturatingMultiply(elements, size);
          const uint64_t bytes =
              resident ? 0 : saturatingMultiply(elements, elementBits / 8);
          if (!resident &&
              (bytes == 0 || bytes > std::numeric_limits<uint32_t>::max()))
            return mlir::failure();
          strategy.fragments.push_back(SpatialEdgeFragment{
              resident ? SpatialEdgeFragmentKind::Resident
                       : SpatialEdgeFragmentKind::Peer,
              std::move(fragment.offsets), std::move(fragment.sizes),
              intersection.tile, bytes, resident ? 0 : edge.id,
              resident ? 0 : payloadSlice++});
        }
      }
    }
    if (strategy.fragments.empty())
      return mlir::failure();
    auto consumerOwner = llvm::find_if(
        proof.finalOwners, [&](const analysis::FinalResultOwner &owner) {
          return owner.root == consumerRoot && owner.result == 0 &&
                 owner.shard && *owner.shard == destination.destinationShard;
        });
    if (consumerOwner != proof.finalOwners.end() &&
        consumerOwner->domain.getForm() ==
            analysis::ExactIndexSetForm::BoxUnion &&
        consumerOwner->domain.getBoxes().size() == 1) {
      strategy.consumerOffsets =
          consumerOwner->domain.getBoxes().front().offsets;
      strategy.consumerSizes = consumerOwner->domain.getBoxes().front().sizes;
    } else {
      if (failureReason)
        *failureReason =
            "baseline consumer result has no finite final-owner rectangle";
      return mlir::failure();
    }
    mapping.edgeStrategies.push_back(std::move(strategy));
  }
  return mlir::success();
}

} // namespace

mlir::LogicalResult addCardBaselineEdgeCarriers(
    TileMapping &mapping, const SpatialAssignment &spatial,
    const analysis::ExactDemandProof &demand, const StructuredDAGAnalysis &dag,
    std::string *failureReason) {
  mlir::FailureOr<StructuredDemandView> view =
      StructuredDemandView::create(dag, spatial, demand, failureReason);
  if (mlir::failed(view))
    return mlir::failure();
  mapping.edgeStrategies.clear();
  for (const StructuredDAGEdge &edge : dag.getEdges()) {
    const StructuredDAGNode *producer = dag.getNode(edge.producer);
    const StructuredDAGNode *consumer = dag.getNode(edge.consumer);
    const SemanticRootKey *producerRoot = view->getRoot(edge.producer);
    const SemanticRootKey *consumerRoot = view->getRoot(edge.consumer);
    const analysis::DependencyDemand *dependency =
        view->getDependency(edge.consumer, edge.consumerOperand);
    if (!producer || !consumer || !producerRoot || !consumerRoot ||
        !dependency ||
        mlir::failed(appendCarrier(mapping, edge, *producer, *consumer,
                                   *producerRoot, *consumerRoot, *dependency,
                                   demand, failureReason)))
      return mlir::failure();
  }
  return mlir::success();
}

} // namespace wafer::compiler::detail
