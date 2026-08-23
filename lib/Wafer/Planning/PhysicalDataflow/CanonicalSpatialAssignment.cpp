//===- CanonicalSpatialAssignment.cpp - Direct spatial coordinate ------===//

#include "Wafer/Planning/PhysicalDataflow/CanonicalSpatialAssignment.h"

#include "Wafer/Planning/PhysicalDataflow/SpatialDomain.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <array>
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

using RootFacts = SpatialRootDomainFacts;

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
    const bool mayPartition =
        facts.partitionableParallelIterators.test(iterator) ||
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
addReductionMergePlacements(const RootFacts &facts,
                            llvm::ArrayRef<int64_t> intervalCounts,
                            NodeSpatialPlan &plan, std::string *failureReason) {
  mlir::FailureOr<llvm::SmallVector<ReductionGroupId, 8>> groups =
      deriveSpatialReductionGroups(facts, plan.axes, failureReason);
  if (mlir::failed(groups))
    return mlir::failure();
  if (groups->empty())
    return mlir::success();

  for (const ReductionGroupId &group : *groups) {
    if (group.resultGroup >= facts.resultParallelIteratorsByGroup.size()) {
      if (failureReason)
        *failureReason = "spatial reduction group has no result map";
      return mlir::failure();
    }
    const llvm::SmallBitVector &resultParallel =
        facts.resultParallelIteratorsByGroup[group.resultGroup];
    std::optional<TileId> contributor;
    for (size_t cell = 0; cell < plan.embedding.size(); ++cell) {
      size_t remainder = cell;
      llvm::SmallVector<uint32_t, 8> coordinate(intervalCounts.size());
      for (size_t reverse = 0; reverse < intervalCounts.size(); ++reverse) {
        const size_t iterator = intervalCounts.size() - reverse - 1;
        coordinate[iterator] = static_cast<uint32_t>(
            remainder % static_cast<size_t>(intervalCounts[iterator]));
        remainder /= static_cast<size_t>(intervalCounts[iterator]);
      }
      llvm::SmallVector<uint32_t, 4> parallelCoordinate;
      for (int iterator = resultParallel.find_first(); iterator >= 0;
           iterator = resultParallel.find_next(iterator))
        parallelCoordinate.push_back(coordinate[iterator]);
      if (parallelCoordinate == group.parallelCoordinate) {
        contributor = plan.embedding[cell];
        break;
      }
    }
    if (!contributor) {
      if (failureReason)
        *failureReason = "spatial reduction group has no contributor";
      return mlir::failure();
    }
    plan.reductionMerges.push_back({group, *contributor});
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
  SpatialDomainProblemResult domainProblem =
      buildSpatialDomainProblem(dag, availableTiles);
  if (!domainProblem.succeeded()) {
    if (failureReason && domainProblem.failure)
      *failureReason = domainProblem.failure->detail;
    return mlir::failure();
  }
  const SpatialDomainProblem &problem = *domainProblem.problem;
  llvm::ArrayRef<RootFacts> facts = problem.getRoots();

  SpatialPlan plan;
  plan.nodes.reserve(facts.size());
  for (const RootFacts &root : facts) {
    mlir::FailureOr<llvm::SmallVector<int64_t, 8>> factors =
        deriveCanonicalFactors(
            root, problem.getStructuralProblem().getAvailableTiles().size(),
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
    llvm::ArrayRef<TileId> tiles =
        problem.getStructuralProblem().getAvailableTiles();
    node.embedding.assign(tiles.begin(), tiles.begin() + cellCount);

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
    if (mlir::failed(addReductionMergePlacements(root, *intervalCounts, node,
                                                 failureReason)))
      return mlir::failure();
    plan.nodes.push_back(std::move(node));
  }

  mlir::FailureOr<SpatialAssignment> assignment = closeSpatialPlanStructure(
      problem.getStructuralProblem(), plan, failureReason);
  if (mlir::failed(assignment))
    return mlir::failure();
  return CanonicalSpatialCoordinate{problem.getSemanticRoots(),
                                    problem.getStructuralProblem(),
                                    std::move(plan), std::move(*assignment)};
}

} // namespace wafer::compiler::detail
