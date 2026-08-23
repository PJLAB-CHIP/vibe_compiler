//===- BaselineTemporalPlan.cpp - Actual-feedback temporal plan ------===//

#include "Wafer/Planning/Baseline/BaselineTemporalPlan.h"

#include "Wafer/IR/WaferDialect.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalDomain.h"

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
  for (const RegionGroupPlan &group : regions.groups)
    if (group.mandatoryRoots.size() != 1)
      return broken(BrokenTemporalPlanReason::RegionWorkMismatch,
                    "baseline temporal input is not singleton regions");
  TemporalDomainResult domain = buildTemporalDomain(regions, rootWorks);
  if (!domain.succeeded()) {
    const std::string detail =
        domain.failure ? domain.failure->detail
                       : "baseline temporal domain returned no detail";
    const BrokenTemporalPlanReason reason =
        detail.find("non-positive") != std::string::npos
            ? BrokenTemporalPlanReason::InvalidLocalExtent
            : BrokenTemporalPlanReason::RegionWorkMismatch;
    return broken(reason, detail);
  }
  TemporalSuccessor first = domain.domain->getFirstPlan();
  if (first.getKind() != TemporalSuccessorKind::Plan || !first.getPlan())
    return broken(BrokenTemporalPlanReason::RegionWorkMismatch,
                  first.getDetail().empty()
                      ? "baseline temporal domain has no full-local point"
                      : first.getDetail());
  return *first.getPlan();
}

mlir::FailureOr<bool> refineBaselineTemporalPlan(
    TemporalPlan &temporal, llvm::ArrayRef<analysis::RootRegionWork> rootWorks,
    llvm::ArrayRef<SemanticRootKey> affectedRoots, std::string *failureReason) {
  if (affectedRoots.empty()) {
    if (failureReason)
      *failureReason = "actual SPM rejection has no attributed semantic root";
    return mlir::failure();
  }
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
        const ExecutionInstanceId *execution = getRequiredExecution(scope.id);
        if (!execution || workOf(*execution).root != root ||
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
      const ExecutionInstanceId *scopeExecution =
          getRequiredExecution(scope.id);
      if (!scopeExecution || !isTopLevelScope(scope.id))
        return mlir::failure();
      const analysis::RootRegionWorkId workId = workOf(*scopeExecution);
      if (workId.root != root ||
          *selectedAxis >= scope.iteratorTileSizes.size())
        continue;
      auto work = works.find(workId);
      const auto *required =
          std::get_if<RequiredRootExecution>(&scopeExecution->source);
      if (work == works.end() || !required)
        return mlir::failure();
      auto execution =
          llvm::find_if(work->second->execution,
                        [&](const analysis::RootExecutionWork &candidate) {
                          return candidate.shard == required->shard;
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
