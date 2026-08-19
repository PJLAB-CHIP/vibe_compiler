//===- SpatialPlacement.cpp - Structured iterator placement -------------===//

#include "Wafer/Planning/Search/SpatialPlacement.h"

#include "Wafer/Analysis/Structured/ReductionSemantics.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>

namespace wafer::compiler::detail {
namespace {

static bool tileSequenceLess(llvm::ArrayRef<TileId> lhs,
                             llvm::ArrayRef<TileId> rhs) {
  return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(),
                                      rhs.end(), [](TileId left, TileId right) {
                                        return left.getValue() <
                                               right.getValue();
                                      });
}

static std::optional<uint64_t>
getParticipantCount(llvm::ArrayRef<uint32_t> factors) {
  uint64_t count = 1;
  for (uint32_t factor : factors) {
    if (factor == 0 || count > std::numeric_limits<uint64_t>::max() / factor)
      return std::nullopt;
    count *= factor;
  }
  return count;
}

static bool requiresReductionMerge(llvm::ArrayRef<uint32_t> factors,
                                   llvm::ArrayRef<uint8_t> reductions) {
  return llvm::any_of(llvm::zip_equal(factors, reductions), [](auto values) {
    return std::get<0>(values) > 1 && std::get<1>(values) != 0;
  });
}

static bool containsTile(llvm::ArrayRef<TileId> tiles, TileId target) {
  return llvm::is_contained(tiles, target);
}

static bool nextDistinctTileSequence(llvm::ArrayRef<TileId> available,
                                     llvm::SmallVectorImpl<TileId> &tiles) {
  for (size_t reverse = 0; reverse < tiles.size(); ++reverse) {
    const size_t position = tiles.size() - reverse - 1;
    auto current = llvm::find(available, tiles[position]);
    if (current == available.end())
      return false;
    for (auto next = std::next(current); next != available.end(); ++next) {
      if (containsTile(llvm::ArrayRef<TileId>(tiles).take_front(position),
                       *next))
        continue;
      tiles[position] = *next;
      llvm::SmallVector<TileId, 16> used(tiles.begin(),
                                         tiles.begin() + position + 1);
      size_t suffix = position + 1;
      if (suffix == tiles.size())
        return true;
      for (TileId candidate : available) {
        if (containsTile(used, candidate))
          continue;
        tiles[suffix++] = candidate;
        used.push_back(candidate);
        if (suffix == tiles.size())
          return true;
      }
      return false;
    }
  }
  return false;
}

static bool nextFactorVector(llvm::ArrayRef<int64_t> extents,
                             uint64_t maximumParticipants,
                             llvm::SmallVectorImpl<uint32_t> &factors) {
  for (size_t reverse = 0; reverse < factors.size(); ++reverse) {
    const size_t position = factors.size() - reverse - 1;
    for (uint64_t next = static_cast<uint64_t>(factors[position]) + 1;
         next <= static_cast<uint64_t>(extents[position]) &&
         next <= maximumParticipants;
         ++next) {
      llvm::SmallVector<uint32_t, 4> candidate(factors.begin(), factors.end());
      candidate[position] = static_cast<uint32_t>(next);
      std::fill(candidate.begin() + position + 1, candidate.end(), 1);
      std::optional<uint64_t> participants = getParticipantCount(candidate);
      if (participants && *participants <= maximumParticipants) {
        factors.assign(candidate.begin(), candidate.end());
        return true;
      }
    }
    factors[position] = 1;
  }
  return false;
}

} // namespace

bool operator<(const SpatialPlacementAssignment &lhs,
               const SpatialPlacementAssignment &rhs) {
  if (lhs.node != rhs.node)
    return lhs.node < rhs.node;
  if (lhs.iteratorFactors != rhs.iteratorFactors)
    return std::lexicographical_compare(
        lhs.iteratorFactors.begin(), lhs.iteratorFactors.end(),
        rhs.iteratorFactors.begin(), rhs.iteratorFactors.end());
  if (lhs.tiles != rhs.tiles)
    return tileSequenceLess(lhs.tiles, rhs.tiles);
  if (lhs.reductionMergeTile.has_value() != rhs.reductionMergeTile.has_value())
    return !lhs.reductionMergeTile.has_value();
  if (!lhs.reductionMergeTile)
    return false;
  return lhs.reductionMergeTile->getValue() <
         rhs.reductionMergeTile->getValue();
}

StructuredDAGNodePlacement
SpatialPlacementAssignment::getNodePlacement() const {
  StructuredDAGNodePlacement placement;
  placement.node = node;
  placement.iteratorPartitionFactors = iteratorFactors;
  placement.tiles = tiles;
  placement.reductionMergeTile = reductionMergeTile;
  return placement;
}

mlir::FailureOr<SpatialPlacementDomain>
SpatialPlacementDomain::create(const StructuredDAGNode &node,
                               llvm::ArrayRef<TileId> availableTiles) {
  auto linalg = mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(node.operation);
  if (!linalg || availableTiles.empty())
    return mlir::failure();
  llvm::SmallVector<int64_t, 4> extents = linalg.getStaticLoopRanges();
  llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes =
      linalg.getIteratorTypesArray();
  if (extents.size() != iteratorTypes.size() ||
      llvm::any_of(extents, [](int64_t extent) { return extent <= 0; }))
    return mlir::failure();

  llvm::SmallVector<TileId, 16> tiles(availableTiles.begin(),
                                      availableTiles.end());
  llvm::sort(tiles, [](TileId lhs, TileId rhs) {
    return lhs.getValue() < rhs.getValue();
  });
  if (std::adjacent_find(tiles.begin(), tiles.end()) != tiles.end())
    return mlir::failure();
  llvm::SmallVector<uint8_t, 4> reductions;
  reductions.reserve(iteratorTypes.size());
  for (mlir::utils::IteratorType type : iteratorTypes) {
    if (type != mlir::utils::IteratorType::parallel &&
        type != mlir::utils::IteratorType::reduction)
      return mlir::failure();
    reductions.push_back(type == mlir::utils::IteratorType::reduction);
  }
  const bool hasReduction = llvm::is_contained(reductions, uint8_t{1});
  const bool supportsReductionPartitioning =
      !hasReduction ||
      (mlir::isa<mlir::PartialReductionOpInterface>(node.operation) &&
       mlir::succeeded(analysis::verifyReductionPartitionLegality(
           linalg, /*preservesSequentialReductionOrder=*/false)));
  llvm::SmallVector<int64_t, 4> maximumFactors(extents.begin(), extents.end());
  if (!supportsReductionPartitioning)
    for (auto [dimension, reduction] : llvm::enumerate(reductions))
      if (reduction)
        maximumFactors[dimension] = 1;
  return SpatialPlacementDomain(node.id, std::move(extents),
                                std::move(maximumFactors),
                                std::move(reductions), std::move(tiles));
}

SpatialPlacementAssignment SpatialPlacementDomain::getFirstAssignment() const {
  SpatialPlacementAssignment assignment;
  assignment.node = node;
  assignment.iteratorFactors.assign(iteratorExtents.size(), 1);
  assignment.tiles.push_back(availableTiles.front());
  return assignment;
}

bool SpatialPlacementDomain::contains(
    const SpatialPlacementAssignment &assignment) const {
  if (assignment.node != node ||
      assignment.iteratorFactors.size() != iteratorExtents.size())
    return false;
  std::optional<uint64_t> participants =
      getParticipantCount(assignment.iteratorFactors);
  if (!participants || *participants == 0 ||
      *participants != assignment.tiles.size() ||
      *participants > availableTiles.size())
    return false;
  for (auto [factor, extent] :
       llvm::zip_equal(assignment.iteratorFactors, maximumFactors))
    if (factor == 0 || factor > static_cast<uint64_t>(extent))
      return false;
  llvm::SmallVector<TileId, 16> seen;
  for (TileId tile : assignment.tiles) {
    if (!containsTile(availableTiles, tile) || containsTile(seen, tile))
      return false;
    seen.push_back(tile);
  }
  const bool needsMerge =
      requiresReductionMerge(assignment.iteratorFactors, reductionIterators);
  if (needsMerge != assignment.reductionMergeTile.has_value())
    return false;
  if (assignment.reductionMergeTile &&
      !containsTile(availableTiles, *assignment.reductionMergeTile))
    return false;
  return true;
}

mlir::FailureOr<std::optional<SpatialPlacementAssignment>>
SpatialPlacementDomain::getNextAssignment(
    const SpatialPlacementAssignment &assignment) const {
  if (!contains(assignment))
    return mlir::failure();
  SpatialPlacementAssignment next = assignment;
  if (next.reductionMergeTile) {
    auto current = llvm::find(availableTiles, *next.reductionMergeTile);
    if (current == availableTiles.end())
      return mlir::failure();
    if (++current != availableTiles.end()) {
      next.reductionMergeTile = *current;
      return std::optional<SpatialPlacementAssignment>(std::move(next));
    }
  }
  if (nextDistinctTileSequence(availableTiles, next.tiles)) {
    if (requiresReductionMerge(next.iteratorFactors, reductionIterators))
      next.reductionMergeTile = availableTiles.front();
    else
      next.reductionMergeTile.reset();
    return std::optional<SpatialPlacementAssignment>(std::move(next));
  }
  if (!nextFactorVector(maximumFactors, availableTiles.size(),
                        next.iteratorFactors))
    return std::optional<SpatialPlacementAssignment>{};
  std::optional<uint64_t> participants =
      getParticipantCount(next.iteratorFactors);
  if (!participants)
    return mlir::failure();
  next.tiles.assign(availableTiles.begin(),
                    availableTiles.begin() + *participants);
  if (requiresReductionMerge(next.iteratorFactors, reductionIterators))
    next.reductionMergeTile = availableTiles.front();
  else
    next.reductionMergeTile.reset();
  return std::optional<SpatialPlacementAssignment>(std::move(next));
}

mlir::FailureOr<CardSpatialPlacementDomain>
CardSpatialPlacementDomain::create(const StructuredDAGAnalysis &dag,
                                   llvm::ArrayRef<TileId> availableTiles) {
  llvm::SmallVector<SpatialPlacementDomain, 16> domains;
  domains.reserve(dag.getNodes().size());
  for (const StructuredDAGNode &node : dag.getNodes()) {
    mlir::FailureOr<SpatialPlacementDomain> domain =
        SpatialPlacementDomain::create(node, availableTiles);
    if (mlir::failed(domain))
      return mlir::failure();
    domains.push_back(std::move(*domain));
  }
  if (domains.empty())
    return mlir::failure();
  return CardSpatialPlacementDomain(std::move(domains));
}

CardSpatialPlacementAssignment
CardSpatialPlacementDomain::getFirstAssignment() const {
  CardSpatialPlacementAssignment assignment;
  assignment.nodes.reserve(nodeDomains.size());
  for (const SpatialPlacementDomain &domain : nodeDomains)
    assignment.nodes.push_back(domain.getFirstAssignment());
  return assignment;
}

bool CardSpatialPlacementDomain::contains(
    const CardSpatialPlacementAssignment &assignment) const {
  if (assignment.nodes.size() != nodeDomains.size())
    return false;
  return llvm::all_of(
      llvm::zip_equal(nodeDomains, assignment.nodes), [](auto domainAndNode) {
        return std::get<0>(domainAndNode).contains(std::get<1>(domainAndNode));
      });
}

mlir::FailureOr<std::optional<CardSpatialPlacementAssignment>>
CardSpatialPlacementDomain::getNextAssignment(
    const CardSpatialPlacementAssignment &assignment) const {
  if (!contains(assignment))
    return mlir::failure();
  CardSpatialPlacementAssignment next = assignment;
  for (size_t reverse = 0; reverse < nodeDomains.size(); ++reverse) {
    const size_t node = nodeDomains.size() - reverse - 1;
    mlir::FailureOr<std::optional<SpatialPlacementAssignment>> successor =
        nodeDomains[node].getNextAssignment(next.nodes[node]);
    if (mlir::failed(successor))
      return mlir::failure();
    if (!*successor)
      continue;
    next.nodes[node] = std::move(**successor);
    for (size_t suffix = node + 1; suffix < nodeDomains.size(); ++suffix)
      next.nodes[suffix] = nodeDomains[suffix].getFirstAssignment();
    return std::optional<CardSpatialPlacementAssignment>(std::move(next));
  }
  return std::optional<CardSpatialPlacementAssignment>{};
}

llvm::SmallVector<StructuredDAGNodePlacement, 16>
CardSpatialPlacementDomain::getNodePlacements(
    const CardSpatialPlacementAssignment &assignment) const {
  llvm::SmallVector<StructuredDAGNodePlacement, 16> placements;
  if (!contains(assignment))
    return placements;
  placements.reserve(assignment.nodes.size());
  for (const SpatialPlacementAssignment &node : assignment.nodes)
    placements.push_back(node.getNodePlacement());
  return placements;
}

CardSpatialPlacementEvaluation CardSpatialPlacementDomain::evaluate(
    const StructuredDAGAnalysis &dag, analysis::IREpoch epoch,
    const CardSpatialPlacementAssignment &assignment) const {
  CardSpatialPlacementEvaluation evaluation;
  if (!contains(assignment)) {
    evaluation.detail = "spatial assignment is outside its typed domain";
    return evaluation;
  }
  llvm::SmallVector<StructuredDAGNodePlacement, 16> placements =
      getNodePlacements(assignment);
  std::string failureReason;
  mlir::FailureOr<analysis::LogicalShardTrial> trial =
      buildLogicalShardTrial(dag, placements, epoch, &failureReason);
  if (mlir::failed(trial)) {
    evaluation.detail = std::move(failureReason);
    return evaluation;
  }

  StructuredDAGExactDemandQuery query(dag, epoch);
  for (const StructuredDAGEdge &edge : dag.getEdges()) {
    analysis::ExactDemandResult demand = query.query(edge.id, *trial);
    if (demand.status == analysis::ExactDemandStatus::Satisfied)
      continue;
    evaluation.status = demand.status;
    evaluation.detail = std::move(demand.detail);
    return evaluation;
  }
  evaluation.status = analysis::ExactDemandStatus::Satisfied;
  evaluation.trial = std::move(*trial);
  return evaluation;
}

} // namespace wafer::compiler::detail
