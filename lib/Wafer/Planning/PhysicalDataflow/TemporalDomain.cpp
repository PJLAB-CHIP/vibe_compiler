//===- TemporalDomain.cpp - Complete free temporal domain -------------===//

#include "Wafer/Planning/PhysicalDataflow/TemporalDomain.h"

#include "Wafer/IR/WaferDialect.h"

#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>

namespace wafer::compiler::detail {

struct TemporalDomain::Completion {
  TemporalSuccessorKind kind = TemporalSuccessorKind::CompilerBug;
  std::optional<TemporalPlan> plan;
  std::string detail;
};

namespace {

TemporalDomainResult failed(TemporalDomainFailureKind kind,
                            llvm::StringRef detail,
                            std::optional<TemporalScopeId> scope = {}) {
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
      if (auto completed = getFirstTopologicalOrder(scope, active, prefix))
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
makeDescriptor(const RegionExecutionId &id,
               const analysis::RootRegionWork &work,
               const analysis::RootExecutionWork &execution,
               TemporalDomainFailure &failure) {
  TemporalScopeDescriptor descriptor;
  descriptor.id.execution = id;
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
  return descriptor;
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
  if (auto next = getNextTopologicalOrder(scope, active, plan.waveLoopOrder)) {
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
    auto first = getFirstTopologicalOrder(scope, active);
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
  if (prefix.size() > scopes.size()) {
    completion.kind = TemporalSuccessorKind::CompilerBug;
    completion.detail = "temporal prefix has an unexpected extra scope";
    return completion;
  }
  completion.plan.emplace();
  for (auto [index, descriptor] : llvm::enumerate(scopes)) {
    if (index < prefix.size()) {
      if (!containsScopePlan(descriptor, prefix[index])) {
        completion.kind = TemporalSuccessorKind::CompilerBug;
        completion.detail =
            "temporal prefix does not match its scope descriptor";
        completion.plan.reset();
        return completion;
      }
      completion.plan->scopes.push_back(prefix[index]);
    } else {
      completion.plan->scopes.push_back(getFirstScopePlan(descriptor));
    }
  }
  return completion;
}

TemporalSuccessor TemporalDomain::getFirstPlan() const {
  return completePrefix(TemporalPlan{});
}

TemporalSuccessor
TemporalDomain::completePrefix(const TemporalPlan &prefix) const {
  Completion completed = completePlan(prefix.scopes);
  if (completed.kind != TemporalSuccessorKind::Plan || !completed.plan)
    return {completed.kind, {}, {}, std::move(completed.detail)};
  TemporalCursor cursor;
  cursor.plan = *completed.plan;
  return {TemporalSuccessorKind::Plan, std::move(completed.plan),
          std::move(cursor)};
}

TemporalSuccessor
TemporalDomain::getNextPlan(const TemporalCursor &cursor) const {
  Completion current = completePlan(cursor.plan.scopes);
  if (current.kind != TemporalSuccessorKind::Plan || !current.plan ||
      !(*current.plan == cursor.plan))
    return {TemporalSuccessorKind::CompilerBug,
            {},
            {},
            "temporal cursor cannot be replayed from current facts"};
  for (size_t reverse = 0; reverse < scopes.size(); ++reverse) {
    const size_t index = scopes.size() - reverse - 1;
    TemporalPlan prefix;
    prefix.scopes.assign(cursor.plan.scopes.begin(),
                         cursor.plan.scopes.begin() + index + 1);
    if (!advanceScopePlan(scopes[index], prefix.scopes.back()))
      continue;
    Completion next = completePlan(prefix.scopes);
    if (next.kind != TemporalSuccessorKind::Plan || !next.plan)
      return {next.kind, {}, {}, std::move(next.detail)};
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
  std::map<TemporalScopeId, size_t> original;
  for (auto [index, scope] : llvm::enumerate(scopes)) {
    if (!original.try_emplace(scope.id, index).second)
      return failed(TemporalDomainFailureKind::BrokenContract,
                    "temporal domain has a duplicate execution scope",
                    scope.id);
    const size_t rank = scope.iterationExtents.size();
    if (rank == 0 || scope.iterationOffsets.size() != rank ||
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
  }
  llvm::sort(scopes, [](const TemporalScopeDescriptor &lhs,
                        const TemporalScopeDescriptor &rhs) {
    return lhs.id < rhs.id;
  });
  return {TemporalDomain(std::move(scopes)), {}};
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

  std::vector<TemporalScopeDescriptor> scopes;
  std::set<RegionExecutionId> observed;
  auto append = [&](RegionExecutionId id, const RequiredRootExecution &required)
      -> std::optional<TemporalDomainFailure> {
    if (!observed.insert(id).second)
      return TemporalDomainFailure{TemporalDomainFailureKind::BrokenContract,
                                   TemporalScopeId{id},
                                   "temporal input has a duplicate execution"};
    auto work = works.find(required.work);
    const analysis::RootExecutionWork *piece =
        work == works.end() ? nullptr : findExecution(*work->second, required);
    if (!piece)
      return TemporalDomainFailure{TemporalDomainFailureKind::BrokenContract,
                                   TemporalScopeId{id},
                                   "temporal execution has no exact root work"};
    TemporalDomainFailure failure;
    auto descriptor = makeDescriptor(id, *work->second, *piece, failure);
    if (!descriptor)
      return failure;
    scopes.push_back(std::move(*descriptor));
    return std::nullopt;
  };

  for (const RegionGroupPlan &group : regions.groups) {
    for (const ExecutionInstancePlan &execution : group.executions) {
      const auto *required =
          std::get_if<RequiredRootExecution>(&execution.id.source);
      if (!required)
        continue;
      if (auto failure = append(RegionExecutionId{execution.id}, *required))
        return {{}, std::move(*failure)};
    }
    for (const ReplicaExecutionPlan &replica : group.replicas)
      if (auto failure =
              append(RegionExecutionId{replica.id}, replica.id.producer))
        return {{}, std::move(*failure)};
  }
  if (scopes.empty())
    return failed(TemporalDomainFailureKind::BrokenContract,
                  "temporal input has no tileable execution");
  return buildTemporalDomain(scopes);
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
    llvm::ArrayRef<SemanticRootKey> affectedRoots, std::string *failureReason,
    bool preferReductionAxes) {
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
      llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes =
          tiling.getLoopIteratorTypes();
      if (preferReductionAxes)
        for (auto [axis, iteratorType] : llvm::enumerate(iteratorTypes))
          if (iteratorType == mlir::utils::IteratorType::reduction)
            allowedAxes.push_back(static_cast<unsigned>(axis));
      if (allowedAxes.empty())
        for (unsigned axis = 0; axis < iteratorTypes.size(); ++axis)
          allowedAxes.push_back(axis);
    }

    std::optional<unsigned> selectedAxis;
    int64_t largestCurrent = 1;
    for (unsigned axis : allowedAxes)
      for (const TemporalScopePlan &scope : candidate.scopes) {
        std::optional<analysis::RootRegionWorkId> work =
            workOf(scope.id.execution);
        if (!work || work->root != root ||
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
  if (changed)
    temporal = std::move(candidate);
  return changed;
}

} // namespace wafer::compiler::detail
