//===- BaselineTemporalPlan.cpp - Actual-feedback temporal plan ------===//

#include "Wafer/Planning/Baseline/BaselineTemporalPlan.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalTileShape.h"

#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/STLExtras.h"

#include <map>
#include <set>

namespace wafer::compiler::detail {
namespace {

CanonicalTemporalPlanOutcome
broken(BrokenTemporalPlanReason reason, llvm::StringRef detail,
       std::optional<analysis::RootRegionWorkId> work = std::nullopt) {
  return BrokenTemporalPlan{reason, std::move(work), detail.str()};
}

analysis::RootRegionWorkId workOf(const ExecutionInstanceId &execution) {
  return std::visit([](const auto &source) { return source.work; },
                    execution.source);
}

} // namespace

CanonicalTemporalPlanOutcome
buildBaselineTemporalPlan(const RegionPlan &regions,
                          llvm::ArrayRef<analysis::RootRegionWork> rootWorks) {
  std::map<analysis::RootRegionWorkId, const analysis::RootRegionWork *> works;
  for (const analysis::RootRegionWork &work : rootWorks)
    if (!works.try_emplace(work.id, &work).second)
      return broken(BrokenTemporalPlanReason::DuplicateRootWork,
                    "baseline temporal input has duplicate root work", work.id);

  std::set<ExecutionInstanceId> observed;
  TemporalPlan result;
  for (const RegionGroupPlan &group : regions.groups) {
    if (group.mandatoryRoots.size() != 1)
      return broken(BrokenTemporalPlanReason::RegionWorkMismatch,
                    "baseline temporal input is not singleton regions");
    auto work = works.find(group.mandatoryRoots.front());
    if (work == works.end())
      return broken(BrokenTemporalPlanReason::RegionWorkMismatch,
                    "baseline temporal region has no root work");
    for (const ExecutionInstancePlan &instance : group.executions) {
      const auto *root =
          std::get_if<RequiredRootExecution>(&instance.id.source);
      if (!root)
        continue;
      if (!observed.insert(instance.id).second)
        return broken(BrokenTemporalPlanReason::DuplicateScope,
                      "baseline temporal execution is duplicated", root->work);
      auto execution = llvm::find_if(
          work->second->execution,
          [&](const analysis::RootExecutionWork &candidate) {
            return candidate.shard == root->shard;
          });
      if (execution == work->second->execution.end())
        return broken(BrokenTemporalPlanReason::MissingExecution,
                      "baseline temporal execution has no root work",
                      root->work);
      llvm::SmallVector<int64_t, 4> local;
      for (const IteratorInterval &interval : execution->iterationDomain)
        local.push_back(interval.size);
      if (llvm::is_contained(local, int64_t{0}))
        return broken(BrokenTemporalPlanReason::InvalidLocalExtent,
                      "baseline temporal local extent is invalid", root->work);
      result.scopes.push_back({instance.id, std::move(local), {}});
    }
  }
  llvm::sort(result.scopes,
             [](const TemporalScopePlan &lhs, const TemporalScopePlan &rhs) {
               return lhs.execution < rhs.execution;
             });
  return result;
}

mlir::FailureOr<bool>
refineBaselineTemporalPlan(
    TemporalPlan &temporal,
    llvm::ArrayRef<analysis::RootRegionWork> rootWorks,
    llvm::ArrayRef<SemanticRootKey> affectedRoots,
    std::string *failureReason) {
  if (affectedRoots.empty()) {
    if (failureReason)
      *failureReason =
          "actual SPM rejection has no attributed semantic root";
    return mlir::failure();
  }
  std::set<SemanticRootKey> uniqueRoots(affectedRoots.begin(),
                                       affectedRoots.end());
  std::map<analysis::RootRegionWorkId, const analysis::RootRegionWork *> works;
  for (const analysis::RootRegionWork &work : rootWorks)
    works.try_emplace(work.id, &work);

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
          *failureReason = "actual SPM rejection root is not temporally tileable";
        return mlir::failure();
      }
      for (unsigned axis = 0; axis < tiling.getLoopIteratorTypes().size();
           ++axis)
        allowedAxes.push_back(axis);
    }

    std::optional<unsigned> selectedAxis;
    int64_t largestCurrent = 1;
    for (unsigned axis : allowedAxes)
      for (const TemporalScopePlan &scope : temporal.scopes) {
        if (workOf(scope.execution).root != root ||
            axis >= scope.iteratorTileSizes.size())
          continue;
        if (scope.iteratorTileSizes[axis] > largestCurrent) {
          largestCurrent = scope.iteratorTileSizes[axis];
          selectedAxis = axis;
        }
      }
    if (!selectedAxis)
      continue;

    for (TemporalScopePlan &scope : temporal.scopes) {
      const analysis::RootRegionWorkId workId = workOf(scope.execution);
      if (workId.root != root ||
          *selectedAxis >= scope.iteratorTileSizes.size())
        continue;
      auto work = works.find(workId);
      const auto *required =
          std::get_if<RequiredRootExecution>(&scope.execution.source);
      if (work == works.end() || !required)
        return mlir::failure();
      auto execution = llvm::find_if(
          work->second->execution,
          [&](const analysis::RootExecutionWork &candidate) {
            return candidate.shard == required->shard;
          });
      if (execution == work->second->execution.end() ||
          *selectedAxis >= execution->iterationDomain.size())
        return mlir::failure();
      const int64_t current = scope.iteratorTileSizes[*selectedAxis];
      if (current <= 1)
        continue;
      scope.iteratorTileSizes[*selectedAxis] = (current + 1) / 2;
      changed = true;
    }
  }
  return changed;
}

} // namespace wafer::compiler::detail
