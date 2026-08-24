//===- TemporalDomain.cpp - Complete per-scope temporal domain --------===//

#include "Wafer/Planning/PhysicalDataflow/TemporalDomain.h"

#include "Wafer/Analysis/PhysicalDataflow/IndexRelation.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <tuple>

namespace wafer::compiler::detail {

struct TemporalDomain::NestedTemporalFacts {
  struct Relation {
    RegionExecutionId execution;
    RegionExecutionId parent;
    DemandFragmentId relation;
    analysis::IndexRelation consumerToSource;
    analysis::IndexRelation producerToResult;
    analysis::ExactIndexSet fragmentDomain;
    analysis::StaticRectangularIndexSet producerExecution;
    llvm::SmallVector<IteratorTilingCapability, 4> iteratorCapabilities;
  };

  analysis::IndexRelationLimits limits;
  std::vector<Relation> relations;
};

struct TemporalDomain::Completion {
  TemporalSuccessorKind kind = TemporalSuccessorKind::CompilerBug;
  std::optional<TemporalPlan> plan;
  std::vector<TemporalScopeDescriptor> descriptors;
  std::string detail;
};

namespace {

TemporalDomainResult failed(TemporalDomainFailureKind kind,
                            llvm::StringRef detail,
                            std::optional<TraversalScopeId> scope = {}) {
  return {{}, TemporalDomainFailure{kind, std::move(scope), detail.str()}};
}

bool isAcyclic(unsigned rank,
               llvm::ArrayRef<TemporalPrecedenceEdge> precedence) {
  llvm::SmallVector<unsigned, 8> indegree(rank, 0);
  llvm::SmallVector<llvm::SmallVector<uint32_t, 4>, 8> successors(rank);
  for (const TemporalPrecedenceEdge &edge : precedence) {
    ++indegree[edge.after];
    successors[edge.before].push_back(edge.after);
  }
  llvm::SmallVector<uint32_t, 8> ready;
  for (auto [iterator, degree] : llvm::enumerate(indegree))
    if (degree == 0)
      ready.push_back(static_cast<uint32_t>(iterator));
  unsigned visited = 0;
  while (!ready.empty()) {
    const uint32_t current = ready.pop_back_val();
    ++visited;
    for (uint32_t next : successors[current])
      if (--indegree[next] == 0)
        ready.push_back(next);
  }
  return visited == rank;
}

llvm::SmallVector<uint8_t, 64>
getReachability(unsigned rank,
                llvm::ArrayRef<TemporalPrecedenceEdge> precedence) {
  llvm::SmallVector<uint8_t, 64> result(rank * rank, 0);
  for (const TemporalPrecedenceEdge &edge : precedence)
    result[edge.before * rank + edge.after] = 1;
  for (unsigned via = 0; via < rank; ++via)
    for (unsigned before = 0; before < rank; ++before)
      if (result[before * rank + via])
        for (unsigned after = 0; after < rank; ++after)
          result[before * rank + after] |= result[via * rank + after];
  return result;
}

llvm::SmallVector<uint32_t, 4>
getActiveIterators(const TemporalScopeDescriptor &scope,
                   llvm::ArrayRef<int64_t> sizes) {
  llvm::SmallVector<uint32_t, 4> active;
  for (auto [iterator, extent, size] :
       llvm::enumerate(scope.iterationExtents, sizes))
    if (size < extent)
      active.push_back(static_cast<uint32_t>(iterator));
  return active;
}

bool isTopologicalOrder(const TemporalScopeDescriptor &scope,
                        llvm::ArrayRef<uint32_t> active,
                        llvm::ArrayRef<uint32_t> order) {
  if (active.size() != order.size())
    return false;
  llvm::SmallVector<uint32_t, 4> sorted(order.begin(), order.end());
  llvm::sort(sorted);
  if (sorted != active)
    return false;
  const unsigned rank = scope.iterationExtents.size();
  llvm::SmallVector<uint8_t, 64> reachability =
      getReachability(rank, scope.precedence);
  llvm::SmallVector<unsigned, 8> position(rank, rank);
  for (auto [ordinal, iterator] : llvm::enumerate(order))
    position[iterator] = ordinal;
  for (uint32_t before : active)
    for (uint32_t after : active)
      if (reachability[before * rank + after] &&
          position[before] >= position[after])
        return false;
  return true;
}

bool containsScopePlan(const TemporalScopeDescriptor &descriptor,
                       const TemporalScopePlan &scope) {
  if (!(descriptor.id == scope.id) ||
      scope.iteratorTileSizes.size() != descriptor.iterationExtents.size())
    return false;
  for (auto [size, extent, capability] :
       llvm::zip_equal(scope.iteratorTileSizes, descriptor.iterationExtents,
                       descriptor.iteratorCapabilities))
    if (size <= 0 || size > extent ||
        (capability == IteratorTilingCapability::FullExtentOnly &&
         size != extent))
      return false;
  llvm::SmallVector<uint32_t, 4> active =
      getActiveIterators(descriptor, scope.iteratorTileSizes);
  return isTopologicalOrder(descriptor, active, scope.waveLoopOrder);
}

std::optional<llvm::SmallVector<uint32_t, 4>>
getFirstTopologicalOrder(const TemporalScopeDescriptor &scope,
                         llvm::ArrayRef<uint32_t> active,
                         llvm::ArrayRef<uint32_t> prefix = {}) {
  const unsigned rank = scope.iterationExtents.size();
  llvm::SmallVector<uint8_t, 64> reachability =
      getReachability(rank, scope.precedence);
  llvm::SmallVector<uint32_t, 4> order(prefix.begin(), prefix.end());
  std::set<uint32_t> selected(prefix.begin(), prefix.end());
  if (selected.size() != prefix.size())
    return std::nullopt;
  while (order.size() != active.size()) {
    std::optional<uint32_t> next;
    for (uint32_t candidate : active) {
      if (selected.count(candidate))
        continue;
      bool available = true;
      for (uint32_t predecessor : active)
        if (!selected.count(predecessor) &&
            reachability[predecessor * rank + candidate]) {
          available = false;
          break;
        }
      if (available) {
        next = candidate;
        break;
      }
    }
    if (!next)
      return std::nullopt;
    selected.insert(*next);
    order.push_back(*next);
  }
  return order;
}

std::optional<llvm::SmallVector<uint32_t, 4>>
getNextTopologicalOrder(const TemporalScopeDescriptor &scope,
                        llvm::ArrayRef<uint32_t> active,
                        llvm::ArrayRef<uint32_t> current) {
  if (!isTopologicalOrder(scope, active, current))
    return std::nullopt;
  const unsigned rank = scope.iterationExtents.size();
  llvm::SmallVector<uint8_t, 64> reachability =
      getReachability(rank, scope.precedence);
  for (size_t reverse = 0; reverse < current.size(); ++reverse) {
    const size_t position = current.size() - reverse - 1;
    llvm::SmallVector<uint32_t, 4> prefix(current.begin(),
                                          current.begin() + position);
    std::set<uint32_t> selected(prefix.begin(), prefix.end());
    for (uint32_t candidate : active) {
      if (candidate <= current[position] || selected.count(candidate))
        continue;
      bool available = true;
      for (uint32_t predecessor : active)
        if (!selected.count(predecessor) && predecessor != candidate &&
            reachability[predecessor * rank + candidate]) {
          available = false;
          break;
        }
      if (!available)
        continue;
      prefix.push_back(candidate);
      std::optional<llvm::SmallVector<uint32_t, 4>> completed =
          getFirstTopologicalOrder(scope, active, prefix);
      if (completed)
        return completed;
      prefix.pop_back();
    }
  }
  return std::nullopt;
}

const analysis::RootExecutionWork *
findExecution(const analysis::RootRegionWork &work,
              const RequiredRootExecution &required) {
  auto found = llvm::find_if(work.execution,
                             [&](const analysis::RootExecutionWork &candidate) {
                               return candidate.shard == required.shard;
                             });
  return found == work.execution.end() ? nullptr : &*found;
}

std::optional<TemporalScopeDescriptor>
makeTopLevelDescriptor(const RegionExecutionId &id,
                       const RequiredRootExecution &required,
                       const analysis::RootRegionWork &work,
                       const analysis::RootExecutionWork &execution,
                       TemporalDomainFailure &failure) {
  TemporalScopeDescriptor descriptor;
  descriptor.id.execution = id;
  descriptor.id.invocation = TopLevelWorkPieceId{0};
  for (const IteratorInterval &interval : execution.iterationDomain) {
    if (interval.size <= 0) {
      failure = {TemporalDomainFailureKind::BrokenContract, descriptor.id,
                 "temporal execution has a non-positive local extent"};
      return std::nullopt;
    }
    descriptor.iterationOffsets.push_back(interval.offset);
    descriptor.iterationExtents.push_back(interval.size);
  }
  auto tiling =
      mlir::dyn_cast_or_null<mlir::TilingInterface>(work.rootOperation);
  if (!tiling || tiling.getLoopIteratorTypes().size() !=
                     descriptor.iterationExtents.size()) {
    failure = {TemporalDomainFailureKind::UnsupportedSemantics, descriptor.id,
               "temporal execution lacks a matching typed iterator domain"};
    return std::nullopt;
  }
  descriptor.iteratorCapabilities.assign(descriptor.iterationExtents.size(),
                                         IteratorTilingCapability::Tileable);
  (void)required;
  return descriptor;
}

std::optional<llvm::SmallVector<int64_t, 6>>
getStaticLoopRanges(mlir::Operation *operation) {
  if (auto linalg = mlir::dyn_cast_or_null<mlir::linalg::LinalgOp>(operation)) {
    llvm::SmallVector<int64_t, 6> ranges = linalg.getStaticLoopRanges();
    if (llvm::any_of(ranges, [](int64_t value) { return value <= 0; }))
      return std::nullopt;
    return ranges;
  }
  if (auto attention =
          mlir::dyn_cast_or_null<LinalgExtAttentionOp>(operation)) {
    llvm::SmallVector<int64_t, 6> ranges = attention.getStaticLoopRanges();
    if (llvm::any_of(ranges, [](int64_t value) { return value <= 0; }))
      return std::nullopt;
    return ranges;
  }
  return std::nullopt;
}

std::optional<mlir::AffineMap> getOperandMap(mlir::Operation *operation,
                                             unsigned operand) {
  if (!operation || operand >= operation->getNumOperands())
    return std::nullopt;
  if (auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(operation))
    return linalg.getMatchingIndexingMap(&linalg->getOpOperand(operand));
  if (auto attention = mlir::dyn_cast<LinalgExtAttentionOp>(operation)) {
    llvm::SmallVector<mlir::AffineMap, 6> maps =
        attention.getIndexingMapsArray();
    if (operand < maps.size())
      return maps[operand];
  }
  return std::nullopt;
}

std::optional<mlir::AffineMap> getResultMap(mlir::Operation *operation,
                                            unsigned result) {
  if (!operation || result >= operation->getNumResults())
    return std::nullopt;
  if (auto linalg = mlir::dyn_cast<mlir::linalg::LinalgOp>(operation))
    return linalg.getIndexingMapMatchingResult(operation->getResult(result));
  if (auto attention = mlir::dyn_cast<LinalgExtAttentionOp>(operation)) {
    if (result == 0)
      return attention.getOutputMap();
  }
  return std::nullopt;
}

const analysis::RootBoundaryWork *
findBoundary(const analysis::RootRegionWork &work,
             const analysis::RootBoundaryId &id) {
  auto found = llvm::find_if(work.boundaries,
                             [&](const analysis::RootBoundaryWork &boundary) {
                               return boundary.id == id;
                             });
  return found == work.boundaries.end() ? nullptr : &*found;
}

const analysis::RootBoundaryUseWork *
findBoundaryUse(const analysis::RootBoundaryWork &boundary,
                const analysis::RootUseId &id) {
  auto found = llvm::find_if(
      boundary.consumerUses,
      [&](const analysis::RootBoundaryUseWork &use) { return use.id == id; });
  return found == boundary.consumerUses.end() ? nullptr : &*found;
}

const analysis::ExactIndexSet *
findFragmentDomain(const analysis::RootBoundaryUseWork &use,
                   const DemandFragmentId &fragment) {
  if (!fragment.ownerTile)
    return use.requiredDomain ? &*use.requiredDomain : nullptr;
  auto owner = llvm::find_if(
      use.eligibleFinalOwners,
      [&](const analysis::OwnerIntersection &candidate) {
        return candidate.ownerShard == fragment.ownerShard &&
               candidate.reductionGroup == fragment.reductionGroup &&
               candidate.tile == *fragment.ownerTile;
      });
  return owner == use.eligibleFinalOwners.end() ? nullptr : &owner->domain;
}

struct NestedDerivationFailure {
  TemporalSuccessorKind kind = TemporalSuccessorKind::CompilerBug;
  std::string detail;
};

template <typename T> struct NestedDerivationResult {
  std::optional<T> value;
  std::optional<NestedDerivationFailure> failure;
};

NestedDerivationFailure relationFailure(analysis::IndexRelationStatus status,
                                        llvm::StringRef detail) {
  TemporalSuccessorKind kind = TemporalSuccessorKind::CompilerBug;
  if (status == analysis::IndexRelationStatus::ResourceExhausted)
    kind = TemporalSuccessorKind::Indeterminate;
  else if (status == analysis::IndexRelationStatus::Unsupported ||
           status == analysis::IndexRelationStatus::SoundBound)
    kind = TemporalSuccessorKind::Unsupported;
  return {kind, detail.str()};
}

NestedDerivationResult<
    llvm::SmallVector<analysis::StaticRectangularIndexSet, 8>>
buildWaveRectangles(const TemporalScopeDescriptor &descriptor,
                    const TemporalScopePlan &plan, uint64_t limit) {
  if (descriptor.iterationExtents.size() != plan.iteratorTileSizes.size() ||
      descriptor.iterationOffsets.size() != descriptor.iterationExtents.size())
    return {{},
            NestedDerivationFailure{
                TemporalSuccessorKind::CompilerBug,
                "nested temporal parent descriptor rank is invalid"}};
  llvm::SmallVector<llvm::SmallVector<IteratorInterval, 8>, 6> axes;
  uint64_t count = 1;
  for (auto [offset, extent, size] :
       llvm::zip_equal(descriptor.iterationOffsets, descriptor.iterationExtents,
                       plan.iteratorTileSizes)) {
    std::string detail;
    auto waves = buildTemporalAxisWaves({offset, extent}, size, &detail);
    if (mlir::failed(waves))
      return {{},
              NestedDerivationFailure{TemporalSuccessorKind::CompilerBug,
                                      std::move(detail)}};
    if (!waves->empty() && count > limit / waves->size()) {
      return {{},
              NestedDerivationFailure{
                  TemporalSuccessorKind::Indeterminate,
                  "nested temporal wave classes exceed the relation work "
                  "limit"}};
    }
    count *= waves->size();
    axes.push_back(std::move(*waves));
  }
  if (count > limit) {
    return {{},
            NestedDerivationFailure{
                TemporalSuccessorKind::Indeterminate,
                "nested temporal wave classes exceed the relation work "
                "limit"}};
  }
  llvm::SmallVector<analysis::StaticRectangularIndexSet, 8> rectangles(1);
  for (llvm::ArrayRef<IteratorInterval> axis : axes) {
    llvm::SmallVector<analysis::StaticRectangularIndexSet, 8> expanded;
    expanded.reserve(rectangles.size() * axis.size());
    for (const analysis::StaticRectangularIndexSet &prefix : rectangles)
      for (const IteratorInterval &interval : axis) {
        analysis::StaticRectangularIndexSet next = prefix;
        next.offsets.push_back(interval.offset);
        next.sizes.push_back(interval.size);
        expanded.push_back(std::move(next));
      }
    rectangles = std::move(expanded);
  }
  return {std::move(rectangles), {}};
}

NestedDerivationResult<
    llvm::SmallVector<analysis::StaticRectangularIndexSet, 8>>
intersectAndNormalize(const mlir::presburger::PresburgerSet &lhs,
                      const mlir::presburger::PresburgerSet &rhs,
                      const analysis::IndexRelationLimits &limits) {
  analysis::ExactIndexSet intersection(
      lhs.intersect(rhs), analysis::ExactIndexSetForm::GeneralPresburger);
  if (intersection.isEmpty())
    return {llvm::SmallVector<analysis::StaticRectangularIndexSet, 8>{}, {}};
  auto normalized = analysis::normalizeFiniteExactIndexSet(intersection);
  if (mlir::failed(normalized))
    return {{},
            NestedDerivationFailure{
                TemporalSuccessorKind::Unsupported,
                "nested temporal relation is not an exact box union"}};
  if (normalized->getBoxes().size() > limits.maxRectangularPieces)
    return {{},
            NestedDerivationFailure{
                TemporalSuccessorKind::Indeterminate,
                "nested temporal relation exceeds the box work limit"}};
  return {llvm::SmallVector<analysis::StaticRectangularIndexSet, 8>(
              normalized->getBoxes().begin(), normalized->getBoxes().end()),
          {}};
}

template <typename RelationT>
NestedDerivationResult<std::vector<TemporalScopeDescriptor>>
deriveNestedDescriptors(llvm::ArrayRef<RelationT> relations,
                        llvm::ArrayRef<TemporalScopeDescriptor> descriptors,
                        const TemporalPlan &plan,
                        const analysis::IndexRelationLimits &limits) {
  std::vector<TemporalScopeDescriptor> result;
  for (const RelationT &relation : relations) {
    for (const TemporalScopeDescriptor &parent : descriptors) {
      if (!(parent.id.execution == relation.parent))
        continue;
      auto parentPlan =
          llvm::find_if(plan.scopes, [&](const TemporalScopePlan &scope) {
            return scope.id == parent.id;
          });
      if (parentPlan == plan.scopes.end())
        continue;
      auto waves =
          buildWaveRectangles(parent, *parentPlan, limits.maxRectangularPieces);
      if (!waves.value)
        return {{}, std::move(waves.failure)};
      for (const analysis::StaticRectangularIndexSet &wave : *waves.value) {
        auto requested =
            relation.consumerToSource.getExactStaticRectangularImagePieces(
                wave.offsets, wave.sizes, limits);
        if (!requested.isExact()) {
          return {{}, relationFailure(requested.status, requested.reason)};
        }
        for (const analysis::StaticRectangularIndexSet &request :
             requested.domains) {
          analysis::IndexSetResult requestSet =
              analysis::IndexRelation::staticRectangularDomain(
                  request.offsets, request.sizes, limits);
          if (!requestSet.isExact()) {
            return {{}, relationFailure(requestSet.status, requestSet.reason)};
          }
          auto clippedRequests = intersectAndNormalize(
              *requestSet.set, relation.fragmentDomain.getPresburgerSet(),
              limits);
          if (!clippedRequests.value)
            return {{}, std::move(clippedRequests.failure)};
          for (const analysis::StaticRectangularIndexSet &clipped :
               *clippedRequests.value) {
            analysis::IndexSetResult clippedSet =
                analysis::IndexRelation::staticRectangularDomain(
                    clipped.offsets, clipped.sizes, limits);
            if (!clippedSet.isExact())
              return {{},
                      relationFailure(clippedSet.status, clippedSet.reason)};
            analysis::IndexSetResult producer =
                relation.producerToResult.preimage(*clippedSet.set, limits);
            analysis::IndexSetResult execution =
                analysis::IndexRelation::staticRectangularDomain(
                    relation.producerExecution.offsets,
                    relation.producerExecution.sizes, limits);
            if (!producer.isExact() || !execution.isExact()) {
              const analysis::IndexSetResult &failed =
                  !producer.isExact() ? producer : execution;
              return {{}, relationFailure(failed.status, failed.reason)};
            }
            auto producerPieces =
                intersectAndNormalize(*producer.set, *execution.set, limits);
            if (!producerPieces.value)
              return {{}, std::move(producerPieces.failure)};
            for (const analysis::StaticRectangularIndexSet &piece :
                 *producerPieces.value) {
              NestedInvocationClassId invocation;
              invocation.parent = relation.parent;
              // Invocation classes describe translation-equivalent work, not
              // one state per parent-wave ordinal. The actual parent leaf
              // supplies absolute offsets; only local extents and the typed
              // use relation select the child temporal plan.
              invocation.uses.push_back(
                  {relation.relation,
                   llvm::SmallVector<int64_t, 4>(clipped.offsets.size(), 0),
                   clipped.sizes});
              invocation.producerOffsets.assign(piece.offsets.size(), 0);
              invocation.producerExtents = piece.sizes;
              TemporalScopeDescriptor child;
              child.id.execution = relation.execution;
              child.id.invocation = std::move(invocation);
              child.iterationOffsets.assign(piece.offsets.size(), 0);
              child.iterationExtents = piece.sizes;
              child.iteratorCapabilities = relation.iteratorCapabilities;
              child.parentScope = parent.id;
              result.push_back(std::move(child));
              if (result.size() > limits.maxRectangularPieces) {
                return {{},
                        NestedDerivationFailure{
                            TemporalSuccessorKind::Indeterminate,
                            "nested temporal invocation classes exceed "
                            "the relation work limit"}};
              }
            }
          }
        }
      }
    }
  }
  llvm::sort(result, [](const TemporalScopeDescriptor &lhs,
                        const TemporalScopeDescriptor &rhs) {
    return lhs.id < rhs.id;
  });
  result.erase(std::unique(result.begin(), result.end(),
                           [](const TemporalScopeDescriptor &lhs,
                              const TemporalScopeDescriptor &rhs) {
                             return lhs == rhs;
                           }),
               result.end());
  return {std::move(result), {}};
}

} // namespace

TemporalScopePlan
TemporalDomain::getFirstScopePlan(const TemporalScopeDescriptor &scope) {
  TemporalScopePlan result;
  result.id = scope.id;
  result.iteratorTileSizes = scope.iterationExtents;
  return result;
}

bool TemporalDomain::advanceScopePlan(const TemporalScopeDescriptor &scope,
                                      TemporalScopePlan &plan) {
  llvm::SmallVector<uint32_t, 4> active =
      getActiveIterators(scope, plan.iteratorTileSizes);
  if (std::optional<llvm::SmallVector<uint32_t, 4>> next =
          getNextTopologicalOrder(scope, active, plan.waveLoopOrder)) {
    plan.waveLoopOrder = std::move(*next);
    return true;
  }
  for (size_t reverse = 0; reverse < scope.iterationExtents.size(); ++reverse) {
    const size_t dimension = scope.iterationExtents.size() - reverse - 1;
    if (scope.iteratorCapabilities[dimension] ==
            IteratorTilingCapability::FullExtentOnly ||
        plan.iteratorTileSizes[dimension] <= 1)
      continue;
    --plan.iteratorTileSizes[dimension];
    for (size_t reset = dimension + 1; reset < scope.iterationExtents.size();
         ++reset)
      plan.iteratorTileSizes[reset] = scope.iterationExtents[reset];
    active = getActiveIterators(scope, plan.iteratorTileSizes);
    std::optional<llvm::SmallVector<uint32_t, 4>> first =
        getFirstTopologicalOrder(scope, active);
    if (!first)
      return false;
    plan.waveLoopOrder = std::move(*first);
    return true;
  }
  return false;
}

TemporalDomain::Completion
TemporalDomain::completePlan(llvm::ArrayRef<TemporalScopePlan> prefix) const {
  Completion completion;
  completion.kind = TemporalSuccessorKind::Plan;
  completion.descriptors = scopes;

  auto appendPlans = [&](size_t from) -> bool {
    for (size_t index = from; index < completion.descriptors.size(); ++index) {
      const TemporalScopeDescriptor &descriptor = completion.descriptors[index];
      if (completion.plan && completion.plan->scopes.size() < prefix.size()) {
        const TemporalScopePlan &selected =
            prefix[completion.plan->scopes.size()];
        if (!containsScopePlan(descriptor, selected)) {
          completion.kind = TemporalSuccessorKind::CompilerBug;
          completion.detail =
              "temporal prefix does not match its derived scope descriptor";
          return false;
        }
        completion.plan->scopes.push_back(selected);
      } else {
        completion.plan->scopes.push_back(getFirstScopePlan(descriptor));
      }
    }
    return true;
  };

  completion.plan.emplace();
  if (!appendPlans(/*from=*/0))
    return completion;
  if (!nestedFacts) {
    if (prefix.size() > completion.plan->scopes.size()) {
      completion.kind = TemporalSuccessorKind::CompilerBug;
      completion.detail = "temporal prefix has an unexpected extra scope";
      completion.plan.reset();
    }
    return completion;
  }

  std::set<RegionExecutionId> derivedExecutions;
  for (const TemporalScopeDescriptor &descriptor : completion.descriptors)
    derivedExecutions.insert(descriptor.id.execution);
  std::set<RegionExecutionId> nestedExecutions;
  for (const NestedTemporalFacts::Relation &relation : nestedFacts->relations)
    nestedExecutions.insert(relation.execution);

  while (true) {
    std::set<RegionExecutionId> ready;
    for (const NestedTemporalFacts::Relation &relation : nestedFacts->relations)
      if (!derivedExecutions.count(relation.execution) &&
          derivedExecutions.count(relation.parent))
        ready.insert(relation.execution);
    if (ready.empty())
      break;

    const size_t previousSize = completion.descriptors.size();
    for (const RegionExecutionId &execution : ready) {
      std::vector<TemporalScopeDescriptor> children;
      for (const NestedTemporalFacts::Relation &relation :
           nestedFacts->relations) {
        if (!(relation.execution == execution))
          continue;
        auto derived = deriveNestedDescriptors(
            llvm::ArrayRef<NestedTemporalFacts::Relation>(&relation, 1),
            completion.descriptors, *completion.plan, nestedFacts->limits);
        if (!derived.value) {
          completion.kind = derived.failure
                                ? derived.failure->kind
                                : TemporalSuccessorKind::CompilerBug;
          completion.detail =
              derived.failure
                  ? std::move(derived.failure->detail)
                  : "nested temporal derivation returned no typed outcome";
          if (completion.kind != TemporalSuccessorKind::Unsupported)
            completion.plan.reset();
          return completion;
        }
        children.insert(children.end(),
                        std::make_move_iterator(derived.value->begin()),
                        std::make_move_iterator(derived.value->end()));
        if (children.size() > nestedFacts->limits.maxRectangularPieces) {
          completion.kind = TemporalSuccessorKind::Indeterminate;
          completion.detail =
              "nested temporal invocation classes exceed the relation work "
              "limit";
          completion.plan.reset();
          return completion;
        }
      }
      std::vector<TemporalScopeDescriptor> merged;
      for (TemporalScopeDescriptor &child : children) {
        auto *invocation =
            std::get_if<NestedInvocationClassId>(&child.id.invocation);
        if (!invocation) {
          completion.kind = TemporalSuccessorKind::CompilerBug;
          completion.detail =
              "derived nested scope has no invocation-class identity";
          completion.plan.reset();
          return completion;
        }
        auto existing = llvm::find_if(
            merged, [&](const TemporalScopeDescriptor &candidate) {
              const auto *candidateInvocation =
                  std::get_if<NestedInvocationClassId>(
                      &candidate.id.invocation);
              return candidateInvocation &&
                     candidate.id.execution == child.id.execution &&
                     candidate.parentScope == child.parentScope &&
                     candidate.iterationOffsets == child.iterationOffsets &&
                     candidate.iterationExtents == child.iterationExtents &&
                     candidate.iteratorCapabilities ==
                         child.iteratorCapabilities &&
                     candidate.precedence == child.precedence;
            });
        if (existing == merged.end()) {
          merged.push_back(std::move(child));
          continue;
        }
        auto &existingInvocation =
            std::get<NestedInvocationClassId>(existing->id.invocation);
        existingInvocation.uses.insert(existingInvocation.uses.end(),
                                       invocation->uses.begin(),
                                       invocation->uses.end());
      }
      children = std::move(merged);
      for (TemporalScopeDescriptor &child : children) {
        auto &invocation =
            std::get<NestedInvocationClassId>(child.id.invocation);
        llvm::sort(invocation.uses);
        invocation.uses.erase(
            std::unique(invocation.uses.begin(), invocation.uses.end()),
            invocation.uses.end());
      }
      llvm::sort(children, [](const TemporalScopeDescriptor &lhs,
                              const TemporalScopeDescriptor &rhs) {
        return lhs.id < rhs.id;
      });
      auto duplicate =
          std::adjacent_find(children.begin(), children.end(),
                             [](const TemporalScopeDescriptor &lhs,
                                const TemporalScopeDescriptor &rhs) {
                               return lhs.id == rhs.id && !(lhs == rhs);
                             });
      if (duplicate != children.end() || children.empty()) {
        completion.kind = TemporalSuccessorKind::CompilerBug;
        completion.detail = children.empty()
                                ? "nested execution has no exact invocation "
                                  "class"
                                : "nested invocation identity has conflicting "
                                  "scope descriptors";
        completion.plan.reset();
        return completion;
      }
      children.erase(std::unique(children.begin(), children.end(),
                                 [](const TemporalScopeDescriptor &lhs,
                                    const TemporalScopeDescriptor &rhs) {
                                   return lhs == rhs;
                                 }),
                     children.end());
      completion.descriptors.insert(completion.descriptors.end(),
                                    std::make_move_iterator(children.begin()),
                                    std::make_move_iterator(children.end()));
      derivedExecutions.insert(execution);
    }
    if (!appendPlans(previousSize))
      return completion;
  }
  if (!std::includes(derivedExecutions.begin(), derivedExecutions.end(),
                     nestedExecutions.begin(), nestedExecutions.end())) {
    completion.kind = TemporalSuccessorKind::CompilerBug;
    completion.detail = "nested temporal execution dependency is missing or "
                        "cyclic";
    completion.plan.reset();
    return completion;
  }
  if (prefix.size() > completion.plan->scopes.size()) {
    completion.kind = TemporalSuccessorKind::CompilerBug;
    completion.detail = "temporal prefix has an unexpected extra scope";
    completion.plan.reset();
  }
  return completion;
}

TemporalSuccessor TemporalDomain::getFirstPlan() const {
  return completePrefix(TemporalPlan{});
}

TemporalSuccessor
TemporalDomain::completePrefix(const TemporalPlan &prefix) const {
  Completion completed = completePlan(prefix.scopes);
  if (completed.kind != TemporalSuccessorKind::Plan || !completed.plan) {
    if (completed.kind == TemporalSuccessorKind::Unsupported &&
        completed.plan) {
      TemporalCursor cursor;
      cursor.plan = *completed.plan;
      return {
          completed.kind, {}, std::move(cursor), std::move(completed.detail)};
    }
    return {completed.kind, {}, {}, std::move(completed.detail)};
  }
  TemporalCursor cursor;
  cursor.plan = *completed.plan;
  return {TemporalSuccessorKind::Plan, std::move(completed.plan),
          std::move(cursor)};
}

TemporalSuccessor
TemporalDomain::getNextPlan(const TemporalCursor &cursor) const {
  Completion current = completePlan(cursor.plan.scopes);
  if ((current.kind != TemporalSuccessorKind::Plan &&
       current.kind != TemporalSuccessorKind::Unsupported) ||
      !current.plan || !(*current.plan == cursor.plan))
    return {TemporalSuccessorKind::CompilerBug,
            {},
            {},
            "temporal cursor cannot be replayed from current facts"};
  for (size_t reverse = 0; reverse < current.descriptors.size(); ++reverse) {
    const size_t index = current.descriptors.size() - reverse - 1;
    TemporalPlan prefix;
    prefix.scopes.assign(cursor.plan.scopes.begin(),
                         cursor.plan.scopes.begin() + index + 1);
    if (!advanceScopePlan(current.descriptors[index], prefix.scopes.back()))
      continue;
    Completion next = completePlan(prefix.scopes);
    if (next.kind != TemporalSuccessorKind::Plan || !next.plan) {
      if (next.kind == TemporalSuccessorKind::Unsupported && next.plan) {
        TemporalCursor nextCursor;
        nextCursor.plan = *next.plan;
        return {next.kind, {}, std::move(nextCursor), std::move(next.detail)};
      }
      return {next.kind, {}, {}, std::move(next.detail)};
    }
    TemporalCursor nextCursor;
    nextCursor.plan = *next.plan;
    return {TemporalSuccessorKind::Plan, std::move(next.plan),
            std::move(nextCursor)};
  }
  return {TemporalSuccessorKind::End};
}

bool TemporalDomain::contains(const TemporalPlan &plan) const {
  Completion completed = completePlan(plan.scopes);
  return completed.kind == TemporalSuccessorKind::Plan && completed.plan &&
         *completed.plan == plan;
}

TemporalDomainResult
buildTemporalDomain(llvm::ArrayRef<TemporalScopeDescriptor> input) {
  std::vector<TemporalScopeDescriptor> scopes(input.begin(), input.end());
  std::map<TraversalScopeId, size_t> original;
  for (auto [index, scope] : llvm::enumerate(scopes)) {
    if (!original.try_emplace(scope.id, index).second)
      return failed(TemporalDomainFailureKind::BrokenContract,
                    "temporal domain has a duplicate traversal scope",
                    scope.id);
    const size_t rank = scope.iterationExtents.size();
    if (scope.iterationOffsets.size() != rank ||
        scope.iteratorCapabilities.size() != rank)
      return failed(TemporalDomainFailureKind::BrokenContract,
                    "temporal scope descriptor ranks do not agree", scope.id);
    if (llvm::any_of(scope.iterationExtents,
                     [](int64_t extent) { return extent <= 0; }))
      return failed(TemporalDomainFailureKind::BrokenContract,
                    "temporal scope has a non-positive extent", scope.id);
    llvm::sort(scope.precedence);
    if (std::adjacent_find(scope.precedence.begin(), scope.precedence.end()) !=
        scope.precedence.end())
      return failed(TemporalDomainFailureKind::BrokenContract,
                    "temporal precedence has a duplicate edge", scope.id);
    for (const TemporalPrecedenceEdge &edge : scope.precedence)
      if (edge.before >= rank || edge.after >= rank ||
          edge.before == edge.after)
        return failed(TemporalDomainFailureKind::BrokenContract,
                      "temporal precedence edge is outside iterator rank",
                      scope.id);
    if (!isAcyclic(rank, scope.precedence))
      return failed(TemporalDomainFailureKind::BrokenContract,
                    "temporal precedence graph has a cycle", scope.id);
    const bool nested =
        std::holds_alternative<NestedInvocationClassId>(scope.id.invocation);
    if (nested != scope.parentScope.has_value())
      return failed(TemporalDomainFailureKind::BrokenContract,
                    "nested temporal scope and parent relation disagree",
                    scope.id);
    if (scope.parentScope) {
      const auto &invocation =
          std::get<NestedInvocationClassId>(scope.id.invocation);
      if (!(invocation.parent == scope.parentScope->execution))
        return failed(TemporalDomainFailureKind::BrokenContract,
                      "nested invocation names a different parent execution",
                      scope.id);
      if (!llvm::is_sorted(invocation.uses) ||
          std::adjacent_find(invocation.uses.begin(), invocation.uses.end()) !=
              invocation.uses.end())
        return failed(TemporalDomainFailureKind::BrokenContract,
                      "nested invocation uses are not canonical", scope.id);
      if (invocation.producerOffsets != scope.iterationOffsets ||
          invocation.producerExtents != scope.iterationExtents ||
          invocation.uses.empty() ||
          llvm::any_of(invocation.uses, [](const NestedUseClassId &use) {
            return use.requestedOffsets.size() != use.requestedExtents.size() ||
                   llvm::any_of(use.requestedExtents,
                                [](int64_t extent) { return extent <= 0; });
          }))
        return failed(TemporalDomainFailureKind::BrokenContract,
                      "nested invocation rectangles and scope disagree",
                      scope.id);
    }
  }

  // Stable dependency order starts with every top-level variable, then adds
  // nested depth layers. This prevents a small child identity from moving in
  // front of an unrelated top-level variable when parent choices change.
  std::vector<TemporalScopeDescriptor> ordered;
  std::set<TraversalScopeId> selected;
  llvm::SmallVector<const TemporalScopeDescriptor *, 16> topLevel;
  for (const TemporalScopeDescriptor &scope : scopes)
    if (!scope.parentScope)
      topLevel.push_back(&scope);
  llvm::sort(topLevel, [](const TemporalScopeDescriptor *lhs,
                          const TemporalScopeDescriptor *rhs) {
    return lhs->id < rhs->id;
  });
  for (const TemporalScopeDescriptor *scope : topLevel) {
    selected.insert(scope->id);
    ordered.push_back(*scope);
  }
  while (ordered.size() != scopes.size()) {
    llvm::SmallVector<const TemporalScopeDescriptor *, 16> ready;
    for (const TemporalScopeDescriptor &scope : scopes) {
      if (selected.count(scope.id) || !scope.parentScope ||
          !selected.count(*scope.parentScope))
        continue;
      ready.push_back(&scope);
    }
    if (ready.empty())
      return failed(TemporalDomainFailureKind::BrokenContract,
                    "temporal scope dependency is missing or cyclic");
    llvm::sort(ready, [](const TemporalScopeDescriptor *lhs,
                         const TemporalScopeDescriptor *rhs) {
      return lhs->id < rhs->id;
    });
    for (const TemporalScopeDescriptor *scope : ready) {
      selected.insert(scope->id);
      ordered.push_back(*scope);
    }
  }
  return {TemporalDomain(std::move(ordered)), {}};
}

TemporalDomainResult
buildTemporalDomain(const RegionPlan &regions,
                    llvm::ArrayRef<analysis::RootRegionWork> rootWorks) {
  if (regions.groups.empty() || rootWorks.empty())
    return failed(TemporalDomainFailureKind::BrokenContract,
                  "temporal input requires region and root work");
  std::map<analysis::RootRegionWorkId, const analysis::RootRegionWork *> works;
  for (const analysis::RootRegionWork &work : rootWorks)
    if (!works.try_emplace(work.id, &work).second)
      return failed(TemporalDomainFailureKind::BrokenContract,
                    "temporal input has duplicate root work");

  struct PendingNestedRelation {
    RegionExecutionId execution;
    ExecutionInstanceId parent;
    RequiredRootExecution producer;
    const RegionGroupPlan *group = nullptr;
  };

  std::vector<TemporalScopeDescriptor> scopes;
  std::vector<PendingNestedRelation> pendingNested;
  std::set<RegionExecutionId> observed;
  for (const RegionGroupPlan &group : regions.groups) {
    for (const ExecutionInstancePlan &execution : group.executions) {
      if (!observed.insert(RegionExecutionId(execution.id)).second)
        return failed(TemporalDomainFailureKind::BrokenContract,
                      "temporal input has duplicate required execution");
      const auto *required =
          std::get_if<RequiredRootExecution>(&execution.id.source);
      if (!required)
        continue;
      if (const auto *nested = std::get_if<ExecutionInstancePlan::NestedUnder>(
              &execution.placement)) {
        pendingNested.push_back({RegionExecutionId(execution.id),
                                 ExecutionInstanceId{nested->consumer},
                                 *required, &group});
        continue;
      }
      auto work = works.find(required->work);
      const analysis::RootExecutionWork *piece =
          work == works.end() ? nullptr
                              : findExecution(*work->second, *required);
      if (!piece)
        return failed(TemporalDomainFailureKind::BrokenContract,
                      "temporal required execution has no exact root work");
      TemporalDomainFailure failure;
      std::optional<TemporalScopeDescriptor> descriptor =
          makeTopLevelDescriptor(RegionExecutionId(execution.id), *required,
                                 *work->second, *piece, failure);
      if (!descriptor)
        return {{}, std::move(failure)};
      scopes.push_back(std::move(*descriptor));
    }
    for (const ReplicaExecutionPlan &replica : group.replicas) {
      if (!observed.insert(RegionExecutionId(replica.id)).second)
        return failed(TemporalDomainFailureKind::BrokenContract,
                      "temporal input has duplicate replica execution");
      if (const auto *nested = std::get_if<ExecutionInstancePlan::NestedUnder>(
              &replica.placement)) {
        pendingNested.push_back({RegionExecutionId(replica.id),
                                 ExecutionInstanceId{nested->consumer},
                                 replica.id.producer, &group});
        continue;
      }
      auto work = works.find(replica.id.producer.work);
      const analysis::RootExecutionWork *piece =
          work == works.end()
              ? nullptr
              : findExecution(*work->second, replica.id.producer);
      if (!piece)
        return failed(TemporalDomainFailureKind::BrokenContract,
                      "temporal replica execution has no producer root work");
      TemporalDomainFailure failure;
      std::optional<TemporalScopeDescriptor> descriptor =
          makeTopLevelDescriptor(RegionExecutionId(replica.id),
                                 replica.id.producer, *work->second, *piece,
                                 failure);
      if (!descriptor)
        return {{}, std::move(failure)};
      scopes.push_back(std::move(*descriptor));
    }
  }

  auto nestedFacts = std::make_shared<TemporalDomain::NestedTemporalFacts>();
  for (const PendingNestedRelation &pending : pendingNested) {
    if (!observed.count(RegionExecutionId(pending.parent)))
      return failed(TemporalDomainFailureKind::BrokenContract,
                    "nested temporal execution names a missing parent");
    const auto *parentRequired =
        std::get_if<RequiredRootExecution>(&pending.parent.source);
    auto consumerWork =
        parentRequired ? works.find(parentRequired->work) : works.end();
    auto producerWork = works.find(pending.producer.work);
    const analysis::RootExecutionWork *consumerExecution =
        consumerWork == works.end() || !parentRequired
            ? nullptr
            : findExecution(*consumerWork->second, *parentRequired);
    const analysis::RootExecutionWork *producerExecution =
        producerWork == works.end()
            ? nullptr
            : findExecution(*producerWork->second, pending.producer);
    if (!consumerExecution || !producerExecution || !pending.group)
      return failed(TemporalDomainFailureKind::BrokenContract,
                    "nested temporal execution has no exact root work");

    llvm::SmallVector<const LocalUseBinding *, 2> bindings;
    for (const LocalUseBinding &binding : pending.group->localBindings)
      if (binding.producer == pending.execution &&
          binding.delivery == LocalUseDelivery::DirectNestedValue)
        bindings.push_back(&binding);
    if (bindings.empty())
      return failed(TemporalDomainFailureKind::BrokenContract,
                    "nested temporal execution has no direct region-use "
                    "binding");

    for (const LocalUseBinding *binding : bindings) {
      const analysis::RootBoundaryWork *boundary =
          findBoundary(*consumerWork->second, binding->fragment.source);
      const analysis::RootBoundaryUseWork *use =
          boundary ? findBoundaryUse(*boundary, binding->fragment.use)
                   : nullptr;
      const analysis::ExactIndexSet *fragmentDomain =
          use ? findFragmentDomain(*use, binding->fragment) : nullptr;
      mlir::Operation *consumer = consumerWork->second->rootOperation;
      mlir::Operation *producer = producerWork->second->rootOperation;
      if (!boundary || !use || !fragmentDomain || fragmentDomain->isEmpty() ||
          !consumer || !producer ||
          binding->fragment.use.operand >= consumer->getNumOperands())
        return failed(TemporalDomainFailureKind::BrokenContract,
                      "nested region use has no exact nonempty boundary "
                      "relation");
      if (consumer->getOperand(binding->fragment.use.operand) !=
          boundary->sourceValue)
        return failed(
            TemporalDomainFailureKind::UnsupportedSemantics,
            "nested temporal relation crosses an explicit reconstruction; "
            "D must provide its composed exact relation");
      auto sourceResult = mlir::dyn_cast<mlir::OpResult>(boundary->sourceValue);
      if (!sourceResult || sourceResult.getOwner() != producer ||
          sourceResult.getResultNumber() != binding->fragment.source.index)
        return failed(TemporalDomainFailureKind::BrokenContract,
                      "nested boundary does not name the selected producer "
                      "result");

      std::optional<llvm::SmallVector<int64_t, 6>> consumerRanges =
          getStaticLoopRanges(consumer);
      std::optional<llvm::SmallVector<int64_t, 6>> producerRanges =
          getStaticLoopRanges(producer);
      std::optional<mlir::AffineMap> consumerMap =
          getOperandMap(consumer, binding->fragment.use.operand);
      std::optional<mlir::AffineMap> producerMap =
          getResultMap(producer, sourceResult.getResultNumber());
      auto sourceType = mlir::dyn_cast<mlir::RankedTensorType>(
          boundary->sourceValue.getType());
      if (!consumerRanges || !producerRanges || !consumerMap || !producerMap ||
          !sourceType || !sourceType.hasStaticShape())
        return failed(TemporalDomainFailureKind::UnsupportedSemantics,
                      "nested temporal relation lacks static structured "
                      "indexing facts");
      if (consumerExecution->iterationDomain.size() != consumerRanges->size() ||
          fragmentDomain->getRank() !=
              static_cast<unsigned>(sourceType.getRank()))
        return failed(TemporalDomainFailureKind::BrokenContract,
                      "nested execution and fragment ranks are inconsistent");

      analysis::IndexRelationResult consumerToSource =
          analysis::IndexRelation::fromAffineMap(*consumerMap, *consumerRanges,
                                                 sourceType.getShape(),
                                                 nestedFacts->limits);
      analysis::IndexRelationResult producerToResult =
          analysis::IndexRelation::fromAffineMap(*producerMap, *producerRanges,
                                                 sourceType.getShape(),
                                                 nestedFacts->limits);
      if (!consumerToSource.isExact() || !producerToResult.isExact()) {
        const analysis::IndexRelationStatus status =
            !consumerToSource.isExact() ? consumerToSource.status
                                        : producerToResult.status;
        return failed(status == analysis::IndexRelationStatus::ResourceExhausted
                          ? TemporalDomainFailureKind::Indeterminate
                          : TemporalDomainFailureKind::UnsupportedSemantics,
                      !consumerToSource.isExact() ? consumerToSource.reason
                                                  : producerToResult.reason);
      }

      analysis::StaticRectangularIndexSet producerDomain;
      for (const IteratorInterval &interval :
           producerExecution->iterationDomain) {
        producerDomain.offsets.push_back(interval.offset);
        producerDomain.sizes.push_back(interval.size);
      }
      if (producerDomain.sizes.size() != producerRanges->size())
        return failed(TemporalDomainFailureKind::BrokenContract,
                      "nested producer execution rank is inconsistent");
      llvm::SmallVector<IteratorTilingCapability, 4> capabilities(
          producerRanges->size(), IteratorTilingCapability::Tileable);
      nestedFacts->relations.push_back(
          {pending.execution, RegionExecutionId(pending.parent),
           binding->fragment, std::move(*consumerToSource.relation),
           std::move(*producerToResult.relation), *fragmentDomain,
           std::move(producerDomain), std::move(capabilities)});
    }
  }
  llvm::sort(nestedFacts->relations,
             [](const TemporalDomain::NestedTemporalFacts::Relation &lhs,
                const TemporalDomain::NestedTemporalFacts::Relation &rhs) {
               return std::tie(lhs.execution, lhs.parent, lhs.relation) <
                      std::tie(rhs.execution, rhs.parent, rhs.relation);
             });

  TemporalDomainResult base =
      buildTemporalDomain(llvm::ArrayRef<TemporalScopeDescriptor>(scopes));
  if (!base.succeeded())
    return base;
  return {
      TemporalDomain(std::move(base.domain->scopes),
                     pendingNested.empty() ? nullptr : std::move(nestedFacts)),
      {}};
}

mlir::FailureOr<TemporalIntervalChildren>
splitTemporalSizeInterval(TemporalSizeInterval interval,
                          std::optional<int64_t> proposal,
                          std::string *failureReason) {
  if (interval.lower <= 0 || interval.upper < interval.lower ||
      (proposal &&
       (*proposal < interval.lower || *proposal > interval.upper))) {
    if (failureReason)
      *failureReason = "temporal size interval or proposal is invalid";
    return mlir::failure();
  }
  const int64_t point = proposal.value_or(static_cast<int64_t>(
      static_cast<__int128>(interval.lower) +
      (static_cast<__int128>(interval.upper) - interval.lower) / 2));
  TemporalIntervalChildren result;
  result.singleton = {point, point};
  if (point < interval.upper)
    result.above = TemporalSizeInterval{point + 1, interval.upper};
  if (point > interval.lower)
    result.below = TemporalSizeInterval{interval.lower, point - 1};
  return result;
}

mlir::FailureOr<llvm::SmallVector<uint32_t, 4>> buildFirstTemporalWaveLoopOrder(
    llvm::ArrayRef<int64_t> iteratorExtents,
    llvm::ArrayRef<int64_t> iteratorTileSizes,
    llvm::ArrayRef<TemporalPrecedenceEdge> precedence,
    std::string *failureReason) {
  if (iteratorExtents.size() != iteratorTileSizes.size() ||
      llvm::any_of(llvm::zip_equal(iteratorExtents, iteratorTileSizes),
                   [](auto values) {
                     auto [extent, size] = values;
                     return extent <= 0 || size <= 0 || size > extent;
                   })) {
    if (failureReason)
      *failureReason = "temporal size vector is outside its iterator domain";
    return mlir::failure();
  }
  TemporalScopeDescriptor descriptor;
  descriptor.iterationExtents.assign(iteratorExtents.begin(),
                                     iteratorExtents.end());
  descriptor.precedence.assign(precedence.begin(), precedence.end());
  llvm::sort(descriptor.precedence);
  for (const TemporalPrecedenceEdge &edge : descriptor.precedence)
    if (edge.before >= iteratorExtents.size() ||
        edge.after >= iteratorExtents.size() || edge.before == edge.after) {
      if (failureReason)
        *failureReason = "temporal precedence edge is outside iterator rank";
      return mlir::failure();
    }
  if (!isAcyclic(iteratorExtents.size(), descriptor.precedence)) {
    if (failureReason)
      *failureReason = "temporal precedence graph has a cycle";
    return mlir::failure();
  }
  llvm::SmallVector<uint32_t, 4> active =
      getActiveIterators(descriptor, iteratorTileSizes);
  auto order = getFirstTopologicalOrder(descriptor, active);
  if (!order) {
    if (failureReason)
      *failureReason = "temporal precedence graph has no linear extension";
    return mlir::failure();
  }
  return std::move(*order);
}

mlir::FailureOr<llvm::SmallVector<IteratorInterval, 8>>
buildTemporalAxisWaves(IteratorInterval interval, int64_t tileSize,
                       std::string *failureReason) {
  if (interval.size <= 0 || tileSize <= 0 || tileSize > interval.size) {
    if (failureReason)
      *failureReason = "temporal wave request has an invalid extent or size";
    return mlir::failure();
  }
  const __int128 end = static_cast<__int128>(interval.offset) + interval.size;
  if (end < std::numeric_limits<int64_t>::min() ||
      end > std::numeric_limits<int64_t>::max()) {
    if (failureReason)
      *failureReason = "temporal wave interval overflows index range";
    return mlir::failure();
  }
  llvm::SmallVector<IteratorInterval, 8> waves;
  int64_t consumed = 0;
  while (consumed < interval.size) {
    const int64_t size = std::min(tileSize, interval.size - consumed);
    const __int128 offset = static_cast<__int128>(interval.offset) + consumed;
    if (offset < std::numeric_limits<int64_t>::min() ||
        offset > std::numeric_limits<int64_t>::max()) {
      if (failureReason)
        *failureReason = "temporal wave offset overflows index range";
      return mlir::failure();
    }
    waves.push_back({static_cast<int64_t>(offset), size});
    consumed += size;
  }
  return waves;
}

mlir::FailureOr<bool> refineTemporalPlanFromActualSPMFeedback(
    TemporalPlan &temporal, llvm::ArrayRef<analysis::RootRegionWork> rootWorks,
    llvm::ArrayRef<SemanticRootKey> affectedRoots, std::string *failureReason) {
  if (affectedRoots.empty()) {
    if (failureReason)
      *failureReason = "actual SPM rejection has no attributed semantic root";
    return mlir::failure();
  }
  auto workOf = [](const RegionExecutionId &execution)
      -> std::optional<analysis::RootRegionWorkId> {
    if (const auto *required = std::get_if<ExecutionInstanceId>(&execution))
      return std::visit([](const auto &source) { return source.work; },
                        required->source);
    return std::get<ReplicaExecutionId>(execution).producer.work;
  };
  auto rootExecution =
      [](const RegionExecutionId &execution) -> const RequiredRootExecution * {
    if (const auto *required = std::get_if<ExecutionInstanceId>(&execution))
      return std::get_if<RequiredRootExecution>(&required->source);
    return &std::get<ReplicaExecutionId>(execution).producer;
  };

  std::set<SemanticRootKey> uniqueRoots(affectedRoots.begin(),
                                        affectedRoots.end());
  std::map<analysis::RootRegionWorkId, const analysis::RootRegionWork *> works;
  for (const analysis::RootRegionWork &work : rootWorks)
    works.try_emplace(work.id, &work);

  TemporalPlan candidate = temporal;
  bool changed = false;
  for (const SemanticRootKey &root : uniqueRoots) {
    mlir::Operation *operation = nullptr;
    for (const analysis::RootRegionWork &work : rootWorks)
      if (work.id.root == root && work.rootOperation) {
        operation = work.rootOperation;
        break;
      }
    if (!operation) {
      if (failureReason)
        *failureReason = "actual SPM rejection names an unknown root";
      return mlir::failure();
    }

    llvm::SmallVector<unsigned, 4> allowedAxes;
    if (auto attention = mlir::dyn_cast<LinalgExtAttentionOp>(operation)) {
      mlir::FailureOr<AttentionIterationRoles> roles =
          attention.getIterationRoles();
      if (mlir::failed(roles))
        return mlir::failure();
      allowedAxes.assign(roles->keyValueReduction.begin(),
                         roles->keyValueReduction.end());
    } else {
      auto tiling = mlir::dyn_cast<mlir::TilingInterface>(operation);
      if (!tiling) {
        if (failureReason)
          *failureReason =
              "actual SPM rejection root is not temporally tileable";
        return mlir::failure();
      }
      for (unsigned axis = 0; axis < tiling.getLoopIteratorTypes().size();
           ++axis)
        allowedAxes.push_back(axis);
    }

    std::optional<unsigned> selectedAxis;
    int64_t largestCurrent = 1;
    for (unsigned axis : allowedAxes)
      for (const TemporalScopePlan &scope : candidate.scopes) {
        std::optional<analysis::RootRegionWorkId> work =
            workOf(scope.id.execution);
        if (!isTopLevelScope(scope.id) || !work || work->root != root ||
            axis >= scope.iteratorTileSizes.size())
          continue;
        if (scope.iteratorTileSizes[axis] > largestCurrent) {
          largestCurrent = scope.iteratorTileSizes[axis];
          selectedAxis = axis;
        }
      }
    if (!selectedAxis)
      continue;

    for (TemporalScopePlan &scope : candidate.scopes) {
      if (!isTopLevelScope(scope.id))
        continue;
      std::optional<analysis::RootRegionWorkId> workId =
          workOf(scope.id.execution);
      const RequiredRootExecution *required = rootExecution(scope.id.execution);
      if (!workId || !required)
        return mlir::failure();
      if (workId->root != root ||
          *selectedAxis >= scope.iteratorTileSizes.size())
        continue;
      auto work = works.find(*workId);
      if (work == works.end())
        return mlir::failure();
      auto execution =
          llvm::find_if(work->second->execution,
                        [&](const analysis::RootExecutionWork &entry) {
                          return entry.shard == required->shard;
                        });
      if (execution == work->second->execution.end() ||
          *selectedAxis >= execution->iterationDomain.size())
        return mlir::failure();
      const int64_t current = scope.iteratorTileSizes[*selectedAxis];
      if (current <= 1)
        continue;
      llvm::SmallVector<int64_t, 4> candidateSizes = scope.iteratorTileSizes;
      candidateSizes[*selectedAxis] = (current + 1) / 2;
      llvm::SmallVector<int64_t, 4> extents;
      for (const IteratorInterval &interval : execution->iterationDomain)
        extents.push_back(interval.size);
      auto order = buildFirstTemporalWaveLoopOrder(extents, candidateSizes, {},
                                                   failureReason);
      if (mlir::failed(order))
        return mlir::failure();
      scope.iteratorTileSizes = std::move(candidateSizes);
      scope.waveLoopOrder = std::move(*order);
      changed = true;
    }
  }
  if (changed) {
    auto firstNested = llvm::find_if(candidate.scopes, [](const auto &scope) {
      return !isTopLevelScope(scope.id);
    });
    candidate.scopes.erase(firstNested, candidate.scopes.end());
    temporal = std::move(candidate);
  }
  return changed;
}

} // namespace wafer::compiler::detail
