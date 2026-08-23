//===- CardBaselineEdgeCarriers.cpp ----------------------------------===//

#include "Wafer/Planning/Baseline/CardBaselineEdgeCarriers.h"

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
  mlir::FailureOr<analysis::ExactIndexSet> normalized =
      analysis::normalizeFiniteExactIndexSet(set);
  if (mlir::failed(normalized))
    return mlir::failure();
  return llvm::SmallVector<analysis::StaticRectangularIndexSet, 8>(
      normalized->getBoxes().begin(), normalized->getBoxes().end());
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
        const bool resident =
            intersection.tile == destination.destinationTile;
        uint64_t elements = 1;
        for (int64_t size : piece.sizes)
          elements = saturatingMultiply(elements, size);
        const uint64_t bytes =
            resident ? 0 : saturatingMultiply(elements, elementBits / 8);
        if (!resident &&
            (bytes == 0 || bytes > std::numeric_limits<uint32_t>::max()))
          return mlir::failure();
        strategy.fragments.push_back(SpatialEdgeFragment{
            resident ? SpatialEdgeFragmentKind::Resident
                     : SpatialEdgeFragmentKind::Peer,
            piece.offsets, piece.sizes, intersection.tile, bytes,
            resident ? 0 : edge.id, resident ? 0 : payloadSlice++});
      }
    }
    if (strategy.fragments.empty())
      return mlir::failure();
    auto consumerOwner = llvm::find_if(
        proof.finalOwners, [&](const analysis::FinalResultOwner &owner) {
          return owner.root == consumerRoot && owner.result == 0 &&
                 owner.shard && *owner.shard == destination.destinationShard;
        });
    if (consumerOwner == proof.finalOwners.end()) {
      if (failureReason)
        *failureReason = "baseline consumer result has no final owner";
      return mlir::failure();
    }
    mlir::FailureOr<llvm::SmallVector<analysis::StaticRectangularIndexSet, 8>>
        consumerPieces = getRectangles(consumerOwner->domain);
    if (mlir::failed(consumerPieces) || consumerPieces->size() != 1) {
      if (failureReason)
        *failureReason =
            "baseline consumer result has no finite final-owner rectangle";
      return mlir::failure();
    }
    strategy.consumerOffsets = consumerPieces->front().offsets;
    strategy.consumerSizes = consumerPieces->front().sizes;
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
