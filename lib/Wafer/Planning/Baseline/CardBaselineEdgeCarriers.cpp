//===- CardBaselineEdgeCarriers.cpp ----------------------------------===//

#include "Wafer/Planning/Baseline/CardBaselineEdgeCarriers.h"

#include "Wafer/Analysis/PhysicalDataflow/StructuredDAGExactDemandQuery.h"
#include "Wafer/Analysis/Structured/StructuredOperationTileFootprint.h"

#include "Wafer/Conversion/WaferTensorProgramToTileRegion/DependentDataflow.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <limits>
#include <optional>

namespace wafer::compiler::detail {
namespace {

uint64_t saturatingMultiply(uint64_t lhs, uint64_t rhs) {
  const unsigned __int128 product = static_cast<unsigned __int128>(lhs) * rhs;
  return product > std::numeric_limits<uint64_t>::max()
             ? std::numeric_limits<uint64_t>::max()
             : static_cast<uint64_t>(product);
}

bool getStaticRectangle(const mlir::presburger::PresburgerSet &set,
                        llvm::SmallVectorImpl<int64_t> &offsets,
                        llvm::SmallVectorImpl<int64_t> &sizes) {
  analysis::IndexSetResult exact{analysis::IndexRelationStatus::Exact, set,
                                 {}};
  analysis::StaticRectangularIndexSetResult rectangle =
      exact.getExactStaticRectangularDomain();
  if (!rectangle.isExact())
    return false;
  offsets.assign(rectangle.domain->offsets.begin(),
                 rectangle.domain->offsets.end());
  sizes.assign(rectangle.domain->sizes.begin(), rectangle.domain->sizes.end());
  return true;
}

std::optional<analysis::StaticRectangularIndexSet> intersectRectangles(
    const analysis::StaticRectangularIndexSet &lhs,
    llvm::ArrayRef<int64_t> rhsOffsets, llvm::ArrayRef<int64_t> rhsSizes) {
  if (lhs.offsets.size() != rhsOffsets.size() ||
      lhs.sizes.size() != rhsSizes.size())
    return std::nullopt;
  analysis::StaticRectangularIndexSet result;
  for (auto [lhsOffset, lhsSize, rhsOffset, rhsSize] :
       llvm::zip_equal(lhs.offsets, lhs.sizes, rhsOffsets, rhsSizes)) {
    int64_t lhsLimit = 0;
    int64_t rhsLimit = 0;
    if (lhsSize <= 0 || rhsSize <= 0 ||
        llvm::AddOverflow(lhsOffset, lhsSize, lhsLimit) ||
        llvm::AddOverflow(rhsOffset, rhsSize, rhsLimit))
      return std::nullopt;
    const int64_t offset = std::max(lhsOffset, rhsOffset);
    const int64_t limit = std::min(lhsLimit, rhsLimit);
    if (offset >= limit)
      return std::nullopt;
    result.offsets.push_back(offset);
    result.sizes.push_back(limit - offset);
  }
  return result;
}

const analysis::LogicalNodeTrial *findNodeTrial(
    const analysis::LogicalShardTrial &trial, StructuredDAGNodeID node) {
  auto found = llvm::find_if(
      trial.nodes,
      [&](const analysis::LogicalNodeTrial &entry) { return entry.node == node; });
  return found == trial.nodes.end() ? nullptr : &*found;
}

const analysis::LogicalTileBinding *findBinding(
    const analysis::LogicalNodeTrial &trial, TileId tile, unsigned result) {
  auto found = llvm::find_if(
      trial.bindings, [&](const analysis::LogicalTileBinding &binding) {
        return binding.tile == tile && binding.resultIndex == result &&
               binding.ownedDomain.has_value();
      });
  return found == trial.bindings.end() ? nullptr : &*found;
}

bool hasCarrier(const TileMapping &mapping, const StructuredDAGEdge &edge,
                mlir::Operation *producer, mlir::Operation *consumer,
                TileId destination) {
  return llvm::any_of(mapping.edgeStrategies,
                      [&](const SpatialEdgeStrategy &strategy) {
                        return strategy.producer == producer &&
                               strategy.producerResult == edge.producerResult &&
                               strategy.consumer == consumer &&
                               strategy.consumerOperand == edge.consumerOperand &&
                               strategy.destinationTile == destination;
                      });
}

bool appendOwnedFragments(
    SpatialEdgeStrategy &strategy,
    llvm::ArrayRef<analysis::StaticRectangularIndexSet> demandPieces,
    const analysis::ExactDestinationDemand &destination,
    const analysis::LogicalNodeTrial &producerTrial,
    const StructuredDAGEdge &edge, unsigned elementBits,
    llvm::ArrayRef<int64_t> producerWaveSizes, int64_t &nextPayloadSlice,
    std::string *failureReason) {
  auto splitAtWaveBoundaries = [&](
                                   const analysis::StaticRectangularIndexSet &box) {
    llvm::SmallVector<analysis::StaticRectangularIndexSet, 8> fragments{box};
    if (box.offsets.size() != producerWaveSizes.size() ||
        box.sizes.size() != producerWaveSizes.size()) {
      fragments.clear();
      return fragments;
    }
    for (size_t dimension = 0; dimension < producerWaveSizes.size();
         ++dimension) {
      const int64_t wave = producerWaveSizes[dimension];
      if (wave <= 0) {
        fragments.clear();
        return fragments;
      }
      llvm::SmallVector<analysis::StaticRectangularIndexSet, 8> next;
      for (const analysis::StaticRectangularIndexSet &fragment : fragments) {
        int64_t remaining = fragment.sizes[dimension];
        int64_t offset = fragment.offsets[dimension];
        while (remaining > 0) {
          analysis::StaticRectangularIndexSet part = fragment;
          const int64_t withinWave = offset % wave;
          const int64_t size =
              std::min(remaining, wave - withinWave);
          part.offsets[dimension] = offset;
          part.sizes[dimension] = size;
          next.push_back(std::move(part));
          if (next.size() > 100000) {
            next.clear();
            return next;
          }
          offset += size;
          remaining -= size;
        }
      }
      fragments = std::move(next);
      if (fragments.empty())
        return fragments;
    }
    return fragments;
  };

  for (const analysis::ExactOwnershipIntersection &intersection :
       destination.ownershipIntersections) {
    if (!intersection.set) {
      if (failureReason)
        *failureReason = "producer ownership intersection is unavailable";
      return false;
    }
    if (intersection.set->isIntegerEmpty())
      continue;
    const analysis::LogicalTileBinding *owner =
        findBinding(producerTrial, intersection.tile, edge.producerResult);
    llvm::SmallVector<int64_t, 4> ownerOffsets;
    llvm::SmallVector<int64_t, 4> ownerSizes;
    if (!owner || !getStaticRectangle(*owner->ownedDomain, ownerOffsets,
                                      ownerSizes)) {
      if (failureReason)
        *failureReason = "producer owner has no finite rectangular domain";
      return false;
    }
    for (const analysis::StaticRectangularIndexSet &piece : demandPieces) {
      std::optional<analysis::StaticRectangularIndexSet> owned =
          intersectRectangles(piece, ownerOffsets, ownerSizes);
      if (!owned)
        continue;
      llvm::SmallVector<analysis::StaticRectangularIndexSet, 8> waveFragments =
          splitAtWaveBoundaries(*owned);
      if (waveFragments.empty()) {
        if (failureReason)
          *failureReason =
              "producer demand cannot be split at temporal wave boundaries";
        return false;
      }
      for (analysis::StaticRectangularIndexSet &waveFragment : waveFragments) {
        if (intersection.tile == destination.destinationTile) {
          strategy.fragments.push_back(SpatialEdgeFragment{
              SpatialEdgeFragmentKind::Resident,
              std::move(waveFragment.offsets), std::move(waveFragment.sizes),
              intersection.tile, /*bytes=*/0, /*communicationId=*/0,
              /*payloadSlice=*/0});
          continue;
        }
        uint64_t elements = 1;
        for (int64_t size : waveFragment.sizes)
          elements =
              saturatingMultiply(elements, static_cast<uint64_t>(size));
        const uint64_t bytes = saturatingMultiply(elements, elementBits / 8);
        if (bytes == 0 || bytes > std::numeric_limits<uint32_t>::max()) {
          if (failureReason)
            *failureReason = "peer fragment exceeds target payload";
          return false;
        }
        strategy.fragments.push_back(SpatialEdgeFragment{
            SpatialEdgeFragmentKind::Peer, std::move(waveFragment.offsets),
            std::move(waveFragment.sizes), intersection.tile, bytes,
            static_cast<int64_t>(edge.id), nextPayloadSlice++});
      }
    }
  }
  if (!strategy.fragments.empty())
    return true;
  if (failureReason)
    *failureReason = "nonempty producer demand has no physical carrier";
  return false;
}

bool appendDestinationCarrier(
    TileMapping &mapping, const StructuredDAGEdge &edge,
    mlir::Operation *producer, mlir::Operation *consumer,
    const analysis::ExactDestinationDemand &destination,
    const analysis::LogicalNodeTrial &producerTrial,
    const analysis::LogicalNodeTrial &consumerTrial,
    StructuredDAGExactDemandQuery &query, unsigned elementBits,
    llvm::ArrayRef<int64_t> producerWaveSizes, int64_t &nextPayloadSlice,
    std::string *failureReason) {
  if (!destination.producerDemand) {
    if (failureReason)
      *failureReason = "destination Tile has no exact producer demand";
    return false;
  }
  if (destination.producerDemand->isIntegerEmpty() ||
      hasCarrier(mapping, edge, producer, consumer,
                 destination.destinationTile))
    return true;
  if (!destination.consumerExecutionDomain) {
    if (failureReason)
      *failureReason = "destination Tile execution domain is unavailable";
    return false;
  }
  analysis::StaticRectangularIndexSetPiecesResult pieces =
      query.getExactProducerDemandPieces(edge.id,
                                         *destination.consumerExecutionDomain);
  if (!pieces.isExact() || pieces.domains.empty()) {
    if (failureReason)
      *failureReason = "producer demand has no finite rectangle decomposition: " +
                       pieces.reason;
    return false;
  }

  SpatialEdgeStrategy strategy;
  strategy.producer = producer;
  strategy.producerResult = edge.producerResult;
  strategy.consumer = consumer;
  strategy.consumerOperand = edge.consumerOperand;
  strategy.destinationTile = destination.destinationTile;
  strategy.sourceTile = destination.destinationTile;
  strategy.action = SpatialEdgeAction::PeerFragments;
  strategy.fragmentsDefineProducerDemand = pieces.domains.size() > 1;
  strategy.producerOffsets = pieces.domains.front().offsets;
  strategy.producerSizes = pieces.domains.front().sizes;
  for (const analysis::StaticRectangularIndexSet &piece :
       llvm::drop_begin(pieces.domains)) {
    for (size_t dimension = 0; dimension < piece.offsets.size(); ++dimension) {
      int64_t currentLimit = 0;
      int64_t pieceLimit = 0;
      if (llvm::AddOverflow(strategy.producerOffsets[dimension],
                            strategy.producerSizes[dimension], currentLimit) ||
          llvm::AddOverflow(piece.offsets[dimension], piece.sizes[dimension],
                            pieceLimit)) {
        if (failureReason)
          *failureReason = "producer carrier bounds overflow";
        return false;
      }
      const int64_t offset =
          std::min(strategy.producerOffsets[dimension], piece.offsets[dimension]);
      const int64_t limit = std::max(currentLimit, pieceLimit);
      strategy.producerOffsets[dimension] = offset;
      strategy.producerSizes[dimension] = limit - offset;
    }
  }
  const analysis::LogicalTileBinding *consumerBinding =
      findBinding(consumerTrial, destination.destinationTile, /*result=*/0);
  if (!consumerBinding ||
      !getStaticRectangle(*consumerBinding->ownedDomain,
                          strategy.consumerOffsets, strategy.consumerSizes)) {
    if (failureReason)
      *failureReason = "consumer result shard is unavailable";
    return false;
  }
  if (!appendOwnedFragments(strategy, pieces.domains, destination,
                            producerTrial, edge, elementBits, producerWaveSizes,
                            nextPayloadSlice, failureReason))
    return false;
  mapping.edgeStrategies.push_back(std::move(strategy));
  return true;
}

bool appendEdgeCarriers(TileMapping &mapping,
                        const analysis::LogicalShardTrial &trial,
                        const StructuredDAGAnalysis &dag,
                        StructuredDAGExactDemandQuery &query,
                        const StructuredDAGEdge &edge,
                        std::string *failureReason) {
  const StructuredDAGNode *producerNode = dag.getNode(edge.producer);
  const StructuredDAGNode *consumerNode = dag.getNode(edge.consumer);
  if (!producerNode || !consumerNode || !producerNode->operation ||
      !consumerNode->operation ||
      edge.producerResult >= producerNode->operation->getNumResults() ||
      edge.consumerOperand >= consumerNode->operation->getNumOperands()) {
    if (failureReason)
      *failureReason = "dependency references an unknown structured node";
    return false;
  }
  if (mlir::failed(traceProducerToConsumerChain(
          producerNode->operation, edge.producerResult,
          consumerNode->operation, edge.consumerOperand, failureReason)))
    return false;
  const analysis::LogicalNodeTrial *producerTrial =
      findNodeTrial(trial, edge.producer);
  const analysis::LogicalNodeTrial *consumerTrial =
      findNodeTrial(trial, edge.consumer);
  if (!producerTrial || !consumerTrial) {
    if (failureReason)
      *failureReason = "producer/consumer logical shard is unavailable";
    return false;
  }
  auto producerType = mlir::dyn_cast<mlir::RankedTensorType>(
      producerNode->operation->getResult(edge.producerResult).getType());
  const unsigned elementBits =
      producerType && producerType.getElementType().isIntOrFloat()
          ? producerType.getElementType().getIntOrFloatBitWidth()
          : 0;
  if (!producerType || !producerType.hasStaticShape() || elementBits == 0 ||
      elementBits % 8 != 0) {
    if (failureReason)
      *failureReason = "producer result is not a static byte-addressable tensor";
    return false;
  }
  analysis::ExactDemandResult demand = query.query(edge.id, trial);
  if (demand.status != analysis::ExactDemandStatus::Satisfied) {
    if (failureReason)
      *failureReason = "exact producer demand failed: " + demand.detail;
    return false;
  }
  auto temporal = llvm::find_if(
      mapping.operationTemporalTiles, [&](const StructuredOpTemporalTile &tile) {
        return tile.operation == producerNode->operation;
      });
  std::optional<llvm::SmallVector<int64_t, 4>> producerWaveSizes =
      temporal != mapping.operationTemporalTiles.end()
          ? getStructuredResultTileShape(producerNode->operation,
                                         edge.producerResult,
                                         temporal->iteratorTileSizes)
          : std::nullopt;
  if (!producerWaveSizes || producerWaveSizes->size() != producerType.getRank()) {
    if (failureReason)
      *failureReason = "producer edge has no temporal result wave";
    return false;
  }
  int64_t nextPayloadSlice = 0;
  for (const analysis::ExactDestinationDemand &destination :
       demand.perDestination)
    if (!appendDestinationCarrier(mapping, edge, producerNode->operation,
                                  consumerNode->operation, destination,
                                  *producerTrial, *consumerTrial, query,
                                  elementBits, *producerWaveSizes,
                                  nextPayloadSlice, failureReason))
      return false;
  return true;
}

mlir::LogicalResult addEdgeCarriers(
    TileMapping &mapping, const analysis::LogicalShardTrial &trial,
    const StructuredDAGAnalysis &dag, StructuredDAGExactDemandQuery &query,
    std::string *failureReason) {
  for (const StructuredDAGEdge &edge : dag.getEdges())
    if (!appendEdgeCarriers(mapping, trial, dag, query, edge, failureReason))
      return mlir::failure();
  return mlir::success();
}

} // namespace

mlir::LogicalResult addCardBaselineEdgeCarriers(
    TileMapping &mapping, const analysis::LogicalShardTrial &trial,
    const StructuredDAGAnalysis &dag, StructuredDAGExactDemandQuery &query,
    std::string *failureReason) {
  return addEdgeCarriers(mapping, trial, dag, query, failureReason);
}

} // namespace wafer::compiler::detail
