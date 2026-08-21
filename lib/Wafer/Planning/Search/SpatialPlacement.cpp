//===- SpatialPlacement.cpp - Structured iterator placement -------------===//

#include "Wafer/Planning/Search/SpatialPlacement.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <vector>

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

mlir::LogicalResult addCanonicalReductionMerges(
    const StructuredDAGNode &node, const SemanticRootKey &root,
    const SpatialPlacementAssignment &assignment, NodeSpatialPlan &plan,
    std::string *failureReason) {
  auto linalg = mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(node.operation);
  if (!linalg)
    return mlir::failure();
  llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes =
      linalg.getIteratorTypesArray();
  const bool partitionsReduction = llvm::any_of(
      llvm::zip_equal(assignment.iteratorFactors, iteratorTypes),
      [](auto values) {
        return std::get<0>(values) > 1 &&
               std::get<1>(values) ==
                   mlir::utils::IteratorType::reduction;
      });
  if (!partitionsReduction)
    return mlir::success();
  if (!mlir::isa<mlir::PartialReductionOpInterface>(node.operation)) {
    if (failureReason)
      *failureReason = "spatial reduction lacks partial-reduction mechanics";
    return mlir::failure();
  }

  for (mlir::OpResult result : node.operation->getResults()) {
    std::set<std::vector<uint32_t>> groups;
    for (size_t cell = 0; cell < assignment.tiles.size(); ++cell) {
      size_t remainder = cell;
      llvm::SmallVector<uint32_t, 4> coordinate(iteratorTypes.size());
      for (size_t reverse = 0; reverse < iteratorTypes.size(); ++reverse) {
        const size_t iterator = iteratorTypes.size() - reverse - 1;
        coordinate[iterator] =
            remainder % assignment.iteratorFactors[iterator];
        remainder /= assignment.iteratorFactors[iterator];
      }
      std::vector<uint32_t> parallelCoordinate;
      for (auto [iterator, type] : llvm::enumerate(iteratorTypes))
        if (type == mlir::utils::IteratorType::parallel)
          parallelCoordinate.push_back(coordinate[iterator]);
      if (!groups.insert(parallelCoordinate).second)
        continue;
      ReductionGroupId group;
      group.root = root;
      group.resultGroup = result.getResultNumber();
      group.parallelCoordinate.assign(parallelCoordinate.begin(),
                                      parallelCoordinate.end());
      plan.reductionMerges.push_back(
          {std::move(group), assignment.tiles[cell]});
    }
  }
  return mlir::success();
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
  return tileSequenceLess(lhs.tiles, rhs.tiles);
}

StructuredDAGNodePlacement
SpatialPlacementAssignment::getNodePlacement() const {
  StructuredDAGNodePlacement placement;
  placement.node = node;
  placement.iteratorPartitionFactors = iteratorFactors;
  placement.tiles = tiles;
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
      mlir::isa<mlir::PartialReductionOpInterface>(node.operation);
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

SpatialPlacementAssignment
SpatialPlacementDomain::getMaximumParticipantAssignment() const {
  return getMaximumParticipantAssignment(availableTiles.size());
}

SpatialPlacementAssignment
SpatialPlacementDomain::getMaximumParticipantAssignment(
    uint64_t maximumParticipants) const {
  struct FactorChoice {
    llvm::SmallVector<uint32_t, 4> factors;
    unsigned reductionSplits = 0;
    unsigned partitionedDimensions = 0;
    uint64_t extentScore = 0;
  };
  const size_t capacity = static_cast<size_t>(std::min<uint64_t>(
      availableTiles.size(), std::max<uint64_t>(maximumParticipants, 1)));
  std::vector<std::optional<FactorChoice>> current(capacity + 1);
  current[1] = FactorChoice{};
  for (size_t dimension = 0; dimension < maximumFactors.size(); ++dimension) {
    std::vector<std::optional<FactorChoice>> next(capacity + 1);
    for (size_t product = 1; product <= capacity; ++product) {
      if (!current[product])
        continue;
      const uint64_t maximum =
          std::min<uint64_t>(maximumFactors[dimension], capacity / product);
      for (uint32_t factor = 1; factor <= maximum; ++factor) {
        const size_t nextProduct = product * factor;
        FactorChoice candidate = *current[product];
        candidate.factors.push_back(factor);
        candidate.reductionSplits +=
            factor > 1 && reductionIterators[dimension] != 0;
        candidate.partitionedDimensions += factor > 1;
        const uint64_t extent =
            static_cast<uint64_t>(iteratorExtents[dimension]);
        const uint64_t weight = static_cast<uint64_t>(factor - 1);
        if (weight != 0)
          candidate.extentScore =
              extent > (std::numeric_limits<uint64_t>::max() -
                        candidate.extentScore) /
                           weight
                  ? std::numeric_limits<uint64_t>::max()
                  : candidate.extentScore + extent * weight;
        auto &incumbent = next[nextProduct];
        if (!incumbent ||
            candidate.reductionSplits < incumbent->reductionSplits ||
            (candidate.reductionSplits == incumbent->reductionSplits &&
             candidate.partitionedDimensions <
                 incumbent->partitionedDimensions) ||
            (candidate.reductionSplits == incumbent->reductionSplits &&
             candidate.partitionedDimensions ==
                 incumbent->partitionedDimensions &&
             candidate.extentScore > incumbent->extentScore) ||
            (candidate.reductionSplits == incumbent->reductionSplits &&
             candidate.partitionedDimensions ==
                 incumbent->partitionedDimensions &&
             candidate.extentScore == incumbent->extentScore &&
             std::lexicographical_compare(
                 incumbent->factors.begin(), incumbent->factors.end(),
                 candidate.factors.begin(), candidate.factors.end())))
          incumbent = std::move(candidate);
      }
    }
    current = std::move(next);
  }

  size_t participants = capacity;
  while (participants > 1 && !current[participants])
    --participants;
  SpatialPlacementAssignment assignment;
  assignment.node = node;
  assignment.iteratorFactors = current[participants]->factors;
  assignment.tiles.assign(availableTiles.begin(),
                          availableTiles.begin() + participants);
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
  return true;
}

mlir::FailureOr<std::optional<SpatialPlacementAssignment>>
SpatialPlacementDomain::getNextAssignment(
    const SpatialPlacementAssignment &assignment) const {
  if (!contains(assignment))
    return mlir::failure();
  SpatialPlacementAssignment next = assignment;
  if (nextDistinctTileSequence(availableTiles, next.tiles))
    return std::optional<SpatialPlacementAssignment>(std::move(next));
  if (!nextFactorVector(maximumFactors, availableTiles.size(),
                        next.iteratorFactors))
    return std::optional<SpatialPlacementAssignment>{};
  std::optional<uint64_t> participants =
      getParticipantCount(next.iteratorFactors);
  if (!participants)
    return mlir::failure();
  next.tiles.assign(availableTiles.begin(),
                    availableTiles.begin() + *participants);
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

CardSpatialPlacementAssignment
CardSpatialPlacementDomain::getMaximumParticipantAssignment() const {
  CardSpatialPlacementAssignment assignment;
  assignment.nodes.reserve(nodeDomains.size());
  for (const SpatialPlacementDomain &domain : nodeDomains)
    assignment.nodes.push_back(domain.getMaximumParticipantAssignment());
  return assignment;
}

mlir::FailureOr<CardSpatialPlacementAssignment>
CardSpatialPlacementDomain::getConstructiveAssignment(
    const StructuredDAGAnalysis &dag, std::string *failureReason) const {
  if (nodeDomains.empty())
    return mlir::failure();
  CardSpatialPlacementAssignment assignment =
      getMaximumParticipantAssignment();
  CardSpatialPlacementEvaluation evaluation = evaluate(dag, assignment);
  if (evaluation.isSatisfied())
    return assignment;
  if (failureReason)
    *failureReason = evaluation.detail;
  return mlir::failure();
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

mlir::FailureOr<SpatialAssignment> CardSpatialPlacementDomain::close(
    const StructuredDAGAnalysis &dag,
    const CardSpatialPlacementAssignment &assignment,
    std::string *failureReason) const {
  if (!contains(assignment)) {
    if (failureReason)
      *failureReason = "spatial assignment is outside its typed domain";
    return mlir::failure();
  }
  mlir::FailureOr<SemanticRootAnalysis> roots =
      SemanticRootAnalysis::create(dag, failureReason);
  if (mlir::failed(roots))
    return mlir::failure();

  llvm::SmallVector<NodeIterationSpace, 16> iterationSpaces;
  SpatialPlan plan;
  for (auto [domain, selected] :
       llvm::zip_equal(nodeDomains, assignment.nodes)) {
    const StructuredDAGNode *dagNode = dag.getNode(selected.node);
    const SemanticRootBinding *root =
        dagNode ? roots->find(dagNode->operation) : nullptr;
    if (!dagNode || !root) {
      if (failureReason)
        *failureReason = "spatial node has no semantic root binding";
      return mlir::failure();
    }
    NodeIterationSpace space;
    space.root = root->key;
    space.iteratorExtents.assign(domain.getIteratorExtents().begin(),
                                 domain.getIteratorExtents().end());
    iterationSpaces.push_back(std::move(space));

    NodeSpatialPlan nodePlan;
    nodePlan.root = root->key;
    for (auto [iterator, factor] :
         llvm::enumerate(selected.iteratorFactors))
      nodePlan.axes.push_back(
          {static_cast<uint32_t>(iterator),
           IteratorPartitionScheme::BalancedParts,
           static_cast<int64_t>(factor)});
    nodePlan.embedding = selected.tiles;
    if (mlir::failed(addCanonicalReductionMerges(
            *dagNode, root->key, selected, nodePlan, failureReason)))
      return mlir::failure();
    plan.nodes.push_back(std::move(nodePlan));
  }
  llvm::sort(plan.nodes, [](const NodeSpatialPlan &lhs,
                            const NodeSpatialPlan &rhs) {
    return lhs.root < rhs.root;
  });
  mlir::FailureOr<SpatialPlanningProblem> problem =
      SpatialPlanningProblem::create(
          iterationSpaces, nodeDomains.front().getAvailableTiles(),
          failureReason);
  if (mlir::failed(problem))
    return mlir::failure();
  return closeSpatialPlanStructure(*problem, plan, failureReason);
}

CardSpatialPlacementEvaluation CardSpatialPlacementDomain::evaluate(
    const StructuredDAGAnalysis &dag,
    const CardSpatialPlacementAssignment &assignment) const {
  CardSpatialPlacementEvaluation evaluation;
  if (!contains(assignment)) {
    evaluation.detail = "spatial assignment is outside its typed domain";
    return evaluation;
  }
  std::string failureReason;
  mlir::FailureOr<SpatialAssignment> closed =
      close(dag, assignment, &failureReason);
  if (mlir::failed(closed)) {
    evaluation.detail = std::move(failureReason);
    return evaluation;
  }
  mlir::FailureOr<DemandPlanningSession> session =
      DemandPlanningSession::create(dag, analysis::IndexRelationLimits(),
                                    &failureReason);
  if (mlir::failed(session)) {
    evaluation.detail = std::move(failureReason);
    return evaluation;
  }
  evaluation.assignment = std::move(*closed);
  evaluation.demand = session->query(*evaluation.assignment);
  if (!evaluation.isSatisfied())
    evaluation.detail = std::visit(
        [](const auto &value) -> std::string {
          using T = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<T, analysis::ExactDemandProof>)
            return {};
          else
            return value.detail;
        },
        *evaluation.demand);
  return evaluation;
}

} // namespace wafer::compiler::detail
