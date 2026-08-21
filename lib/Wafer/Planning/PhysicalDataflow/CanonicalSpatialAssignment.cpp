//===- CanonicalSpatialAssignment.cpp - Direct spatial coordinate ------===//

#include "Wafer/Planning/PhysicalDataflow/CanonicalSpatialAssignment.h"

#include "Wafer/Planning/PhysicalDataflow/AttentionSpatialConstraints.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <optional>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {
namespace {

template <typename T>
mlir::FailureOr<T> fail(std::string *failureReason, llvm::StringRef detail) {
  if (failureReason)
    *failureReason = detail.str();
  return mlir::failure();
}

struct RootFacts {
  SemanticRootKey root;
  mlir::Operation *operation = nullptr;
  llvm::SmallVector<int64_t, 8> iteratorExtents;
  llvm::SmallVector<mlir::utils::IteratorType, 8> iteratorTypes;
  llvm::SmallBitVector resultParallelIterators;
  std::optional<AttentionSpatialConstraints> attention;
};

mlir::FailureOr<RootFacts> deriveRootFacts(const SemanticRootBinding &binding,
                                           std::string *failureReason) {
  auto tiling = mlir::dyn_cast<mlir::TilingInterface>(binding.operation);
  auto destination =
      mlir::dyn_cast<mlir::DestinationStyleOpInterface>(binding.operation);
  if (!tiling || !destination)
    return fail<RootFacts>(
        failureReason, "structured root lacks tiling or destination interface");

  RootFacts facts;
  facts.root = binding.key;
  facts.operation = binding.operation;
  facts.iteratorTypes = tiling.getLoopIteratorTypes();
  llvm::SmallVector<mlir::AffineMap, 4> resultMaps;
  if (auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(binding.operation)) {
    facts.iteratorExtents = linalg.getStaticLoopRanges();
    for (int64_t result = 0; result < destination.getNumDpsInits(); ++result)
      resultMaps.push_back(
          linalg.getMatchingIndexingMap(destination.getDpsInitOperand(result)));
  } else if (auto attention = mlir::dyn_cast<wafer::LinalgExtAttentionOp>(
                 binding.operation)) {
    facts.iteratorExtents = attention.getStaticLoopRanges();
    resultMaps.push_back(attention.getOutputMap());
  } else {
    return fail<RootFacts>(
        failureReason,
        "structured root lacks supported typed indexing semantics");
  }
  if (resultMaps.empty())
    return fail<RootFacts>(failureReason,
                           "structured root has no destination result map");
  if (facts.iteratorExtents.size() != facts.iteratorTypes.size() ||
      facts.iteratorExtents.size() >
          static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
      llvm::any_of(facts.iteratorExtents,
                   [](int64_t extent) { return extent <= 0; }))
    return fail<RootFacts>(
        failureReason,
        "structured root requires positive static iterator extents");

  const size_t rank = facts.iteratorExtents.size();
  facts.resultParallelIterators.resize(rank, false);
  for (mlir::AffineMap map : resultMaps) {
    if (!map || map.getNumDims() != rank || map.getNumSymbols() != 0)
      return fail<RootFacts>(failureReason,
                             "structured result map has invalid loop domain");
    for (mlir::AffineExpr expression : map.getResults()) {
      auto dimension = mlir::dyn_cast<mlir::AffineDimExpr>(expression);
      if (!dimension || dimension.getPosition() >= rank)
        return fail<RootFacts>(
            failureReason,
            "canonical spatial assignment requires projected result maps");
      if (facts.iteratorTypes[dimension.getPosition()] ==
          mlir::utils::IteratorType::parallel)
        facts.resultParallelIterators.set(dimension.getPosition());
    }
  }

  for (mlir::utils::IteratorType type : facts.iteratorTypes)
    if (type != mlir::utils::IteratorType::parallel &&
        type != mlir::utils::IteratorType::reduction)
      return fail<RootFacts>(failureReason,
                             "structured root has unsupported iterator kind");

  if (auto attention =
          mlir::dyn_cast<wafer::LinalgExtAttentionOp>(binding.operation)) {
    mlir::FailureOr<AttentionSpatialConstraints> constraints =
        deriveAttentionSpatialConstraints(attention);
    if (mlir::failed(constraints))
      return fail<RootFacts>(failureReason,
                             "cannot derive attention spatial constraints");
    facts.attention = std::move(*constraints);
  }
  return facts;
}

struct FactorPrefix {
  llvm::SmallVector<int64_t, 8> factors;
  bool partitionsKeyValue = false;
};

bool lexicographicallyGreater(llvm::ArrayRef<int64_t> lhs,
                              llvm::ArrayRef<int64_t> rhs) {
  return std::lexicographical_compare(rhs.begin(), rhs.end(), lhs.begin(),
                                      lhs.end());
}

bool isKeyValueIterator(const RootFacts &facts, size_t iterator) {
  return facts.attention &&
         llvm::is_contained(facts.attention->keyValueReductionIterators,
                            static_cast<unsigned>(iterator));
}

mlir::FailureOr<llvm::SmallVector<int64_t, 8>>
deriveCanonicalFactors(const RootFacts &facts, size_t maximumParticipants,
                       std::string *failureReason) {
  if (maximumParticipants == 0)
    return fail<llvm::SmallVector<int64_t, 8>>(
        failureReason, "canonical spatial assignment has no available Tile");

  const bool requiresKeyValuePartition =
      facts.attention &&
      facts.attention->keyValuePartition ==
          AttentionKeyValuePartitionRequirement::MultipleIntervals;
  using States = std::vector<std::array<std::optional<FactorPrefix>, 2>>;
  States states(maximumParticipants + 1);
  states[1][0] = FactorPrefix{};

  for (size_t iterator = 0; iterator < facts.iteratorExtents.size();
       ++iterator) {
    States next(maximumParticipants + 1);
    const bool keyValue = isKeyValueIterator(facts, iterator);
    const bool mayPartition = facts.resultParallelIterators.test(iterator) ||
                              (requiresKeyValuePartition && keyValue);
    const size_t maximumFactor =
        mayPartition ? std::min<size_t>(maximumParticipants,
                                        facts.iteratorExtents[iterator])
                     : 1;
    for (size_t product = 1; product <= maximumParticipants; ++product) {
      for (unsigned keyValueState = 0; keyValueState < 2; ++keyValueState) {
        if (!states[product][keyValueState])
          continue;
        const size_t factorLimit =
            std::min(maximumFactor, maximumParticipants / product);
        for (size_t factor = 1; factor <= factorLimit; ++factor) {
          FactorPrefix candidate = *states[product][keyValueState];
          candidate.factors.push_back(static_cast<int64_t>(factor));
          candidate.partitionsKeyValue |= keyValue && factor > 1;
          auto &incumbent =
              next[product * factor][candidate.partitionsKeyValue ? 1 : 0];
          if (!incumbent ||
              lexicographicallyGreater(candidate.factors, incumbent->factors))
            incumbent = std::move(candidate);
        }
      }
    }
    states = std::move(next);
  }

  for (size_t product = maximumParticipants; product > 0; --product) {
    auto &selected = states[product][requiresKeyValuePartition ? 1 : 0];
    if (selected)
      return std::move(selected->factors);
  }
  return fail<llvm::SmallVector<int64_t, 8>>(
      failureReason,
      "attention K2 domain cannot form multiple canonical intervals");
}

mlir::FailureOr<llvm::SmallVector<int64_t, 8>>
getIntervalCounts(llvm::ArrayRef<int64_t> extents,
                  llvm::ArrayRef<IteratorPartition> partitions,
                  std::string *failureReason) {
  llvm::SmallVector<int64_t, 8> counts;
  counts.reserve(partitions.size());
  for (auto [extent, partition] : llvm::zip_equal(extents, partitions)) {
    mlir::FailureOr<int64_t> count =
        getIteratorPartitionIntervalCount(extent, partition, failureReason);
    if (mlir::failed(count))
      return mlir::failure();
    counts.push_back(*count);
  }
  return counts;
}

mlir::LogicalResult
addCoupledReductionGroups(const RootFacts &facts,
                          llvm::ArrayRef<int64_t> intervalCounts,
                          NodeSpatialPlan &plan, std::string *failureReason) {
  llvm::SmallVector<unsigned, 2> spatialReductions;
  for (auto [iterator, type] : llvm::enumerate(facts.iteratorTypes))
    if (type == mlir::utils::IteratorType::reduction &&
        intervalCounts[iterator] > 1)
      spatialReductions.push_back(iterator);
  if (spatialReductions.empty())
    return mlir::success();

  auto coupled =
      mlir::dyn_cast<wafer::WaferCoupledReductionOpInterface>(facts.operation);
  if (!coupled) {
    if (failureReason)
      *failureReason = "spatial reduction lacks a coupled reduction interface";
    return mlir::failure();
  }
  wafer::CoupledReductionDescription description =
      coupled.getCoupledReductionDescription();
  llvm::SmallBitVector coupledReductions(intervalCounts.size(), false);
  for (unsigned iterator : description.reductionIterators) {
    if (iterator >= intervalCounts.size() || coupledReductions.test(iterator)) {
      if (failureReason)
        *failureReason = "coupled reduction has an invalid iterator domain";
      return mlir::failure();
    }
    coupledReductions.set(iterator);
  }
  if (description.components.empty() ||
      llvm::any_of(spatialReductions, [&](unsigned iterator) {
        return !coupledReductions.test(iterator);
      })) {
    if (failureReason)
      *failureReason =
          "spatial reduction is not covered by coupled reduction semantics";
    return mlir::failure();
  }

  std::map<std::vector<uint32_t>, TileId> firstContributors;
  for (size_t cell = 0; cell < plan.embedding.size(); ++cell) {
    size_t remainder = cell;
    llvm::SmallVector<uint32_t, 8> coordinate(intervalCounts.size());
    for (size_t reverse = 0; reverse < intervalCounts.size(); ++reverse) {
      const size_t iterator = intervalCounts.size() - reverse - 1;
      coordinate[iterator] = static_cast<uint32_t>(
          remainder % static_cast<size_t>(intervalCounts[iterator]));
      remainder /= static_cast<size_t>(intervalCounts[iterator]);
    }
    std::vector<uint32_t> parallelCoordinate;
    for (int iterator = facts.resultParallelIterators.find_first();
         iterator >= 0;
         iterator = facts.resultParallelIterators.find_next(iterator))
      parallelCoordinate.push_back(coordinate[iterator]);
    firstContributors.try_emplace(std::move(parallelCoordinate),
                                  plan.embedding[cell]);
  }

  for (const auto &[parallelCoordinate, tile] : firstContributors) {
    ReductionGroupId group;
    group.root = facts.root;
    group.resultGroup = 0;
    group.parallelCoordinate.assign(parallelCoordinate.begin(),
                                    parallelCoordinate.end());
    plan.reductionMerges.push_back({std::move(group), tile});
  }
  return mlir::success();
}

llvm::StringRef
describeAttentionViolation(AttentionSpatialConstraintViolation violation) {
  switch (violation) {
  case AttentionSpatialConstraintViolation::None:
    return {};
  case AttentionSpatialConstraintViolation::IteratorDomainMismatch:
    return "canonical attention iterator domain is inconsistent";
  case AttentionSpatialConstraintViolation::FlashAttentionPartitionsKeyValue:
    return "canonical flash attention assignment partitions K2";
  case AttentionSpatialConstraintViolation::
      FlashDecodingLeavesKeyValueUnpartitioned:
    return "canonical flash decoding assignment leaves K2 unpartitioned";
  }
  return "canonical attention constraint has unknown outcome";
}

} // namespace

mlir::FailureOr<CanonicalSpatialCoordinate>
buildCanonicalSpatialAssignment(const StructuredDAGAnalysis &dag,
                                llvm::ArrayRef<TileId> availableTiles,
                                std::string *failureReason) {
  mlir::FailureOr<SemanticRootAnalysis> semanticRoots =
      SemanticRootAnalysis::create(dag, failureReason);
  if (mlir::failed(semanticRoots))
    return mlir::failure();

  llvm::SmallVector<RootFacts, 16> facts;
  llvm::SmallVector<NodeIterationSpace, 16> iterationSpaces;
  facts.reserve(semanticRoots->getRoots().size());
  iterationSpaces.reserve(semanticRoots->getRoots().size());
  for (const SemanticRootBinding &binding : semanticRoots->getRoots()) {
    mlir::FailureOr<RootFacts> root = deriveRootFacts(binding, failureReason);
    if (mlir::failed(root))
      return mlir::failure();
    NodeIterationSpace iterationSpace;
    iterationSpace.root = root->root;
    iterationSpace.iteratorExtents.assign(root->iteratorExtents.begin(),
                                          root->iteratorExtents.end());
    iterationSpaces.push_back(std::move(iterationSpace));
    facts.push_back(std::move(*root));
  }

  mlir::FailureOr<SpatialPlanningProblem> problem =
      SpatialPlanningProblem::create(iterationSpaces, availableTiles,
                                     failureReason);
  if (mlir::failed(problem))
    return mlir::failure();

  SpatialPlan plan;
  plan.nodes.reserve(facts.size());
  for (const RootFacts &root : facts) {
    mlir::FailureOr<llvm::SmallVector<int64_t, 8>> factors =
        deriveCanonicalFactors(root, problem->getAvailableTiles().size(),
                               failureReason);
    if (mlir::failed(factors))
      return mlir::failure();

    NodeSpatialPlan node;
    node.root = root.root;
    size_t cellCount = 1;
    for (auto [iterator, factor] : llvm::enumerate(*factors)) {
      node.axes.push_back({static_cast<uint32_t>(iterator),
                           IteratorPartitionScheme::BalancedParts, factor});
      cellCount *= static_cast<size_t>(factor);
    }
    node.embedding.assign(problem->getAvailableTiles().begin(),
                          problem->getAvailableTiles().begin() + cellCount);

    if (root.attention) {
      AttentionSpatialConstraintViolation violation =
          checkAttentionSpatialConstraints(*root.attention,
                                           root.iteratorExtents, node.axes);
      if (violation != AttentionSpatialConstraintViolation::None)
        return fail<CanonicalSpatialCoordinate>(
            failureReason, describeAttentionViolation(violation));
    }
    mlir::FailureOr<llvm::SmallVector<int64_t, 8>> intervalCounts =
        getIntervalCounts(root.iteratorExtents, node.axes, failureReason);
    if (mlir::failed(intervalCounts))
      return mlir::failure();
    if (mlir::failed(addCoupledReductionGroups(root, *intervalCounts, node,
                                               failureReason)))
      return mlir::failure();
    plan.nodes.push_back(std::move(node));
  }

  mlir::FailureOr<SpatialAssignment> assignment =
      closeSpatialPlanStructure(*problem, plan, failureReason);
  if (mlir::failed(assignment))
    return mlir::failure();
  return CanonicalSpatialCoordinate{std::move(*semanticRoots),
                                    std::move(*problem), std::move(plan),
                                    std::move(*assignment)};
}

} // namespace wafer::compiler::detail
