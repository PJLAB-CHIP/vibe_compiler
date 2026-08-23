//===- CanonicalTemporalPlan.cpp - Full-local temporal point ----------===//

#include "Wafer/Planning/PhysicalDataflow/CanonicalTemporalPlan.h"

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

} // namespace

CanonicalTemporalPlanOutcome
buildCanonicalTemporalPlan(const RegionPlan &regions,
                           llvm::ArrayRef<analysis::RootRegionWork> rootWorks) {
  if (regions.groups.empty() || rootWorks.empty())
    return broken(BrokenTemporalPlanReason::RegionWorkMismatch,
                  "canonical temporal plan requires regions and root work");

  std::map<analysis::RootRegionWorkId, const analysis::RootRegionWork *> works;
  std::set<ExecutionInstanceId> expectedRootExecutions;
  std::set<ExecutionInstanceId> expectedMergeExecutions;
  for (const analysis::RootRegionWork &work : rootWorks) {
    if (!works.try_emplace(work.id, &work).second)
      return broken(BrokenTemporalPlanReason::DuplicateRootWork,
                    "canonical temporal input has duplicate root work",
                    work.id);
    for (const analysis::RootExecutionWork &execution : work.execution)
      expectedRootExecutions.insert(
          {RequiredRootExecution{work.id, execution.shard}});
    for (const analysis::ReductionMergeRequirement &merge : work.merges)
      expectedMergeExecutions.insert(
          {RequiredMergeExecution{work.id, merge.group}});
  }

  std::set<ExecutionInstanceId> observedRootExecutions;
  std::set<ExecutionInstanceId> observedMergeExecutions;
  TemporalPlan result;
  for (const RegionGroupPlan &group : regions.groups) {
    if (group.mandatoryRoots.size() != 1 ||
        group.tile != group.mandatoryRoots.front().tile)
      return broken(BrokenTemporalPlanReason::RegionWorkMismatch,
                    "canonical temporal input is not a singleton region");
    const analysis::RootRegionWorkId workId = group.mandatoryRoots.front();
    auto work = works.find(workId);
    if (work == works.end())
      return broken(BrokenTemporalPlanReason::RegionWorkMismatch,
                    "region references missing root work", workId);

    for (const ExecutionInstancePlan &instance : group.executions) {
      if (const auto *root =
              std::get_if<RequiredRootExecution>(&instance.id.source)) {
        if (root->work != workId ||
            !observedRootExecutions.insert(instance.id).second)
          return broken(BrokenTemporalPlanReason::DuplicateScope,
                        "root execution is duplicated or belongs to another "
                        "region",
                        workId);
        auto execution =
            llvm::find_if(work->second->execution,
                          [&](const analysis::RootExecutionWork &candidate) {
                            return candidate.shard == root->shard;
                          });
        if (execution == work->second->execution.end())
          return broken(BrokenTemporalPlanReason::MissingExecution,
                        "region root execution has no exact work piece",
                        workId);
        TemporalScopePlan scope;
        scope.id.execution = RegionExecutionId(instance.id);
        scope.id.invocation = TopLevelWorkPieceId{0};
        for (const IteratorInterval &interval : execution->iterationDomain) {
          if (interval.size <= 0)
            return broken(BrokenTemporalPlanReason::InvalidLocalExtent,
                          "temporal scope has a non-positive local extent",
                          workId);
          scope.iteratorTileSizes.push_back(interval.size);
        }
        result.scopes.push_back(std::move(scope));
        continue;
      }

      const auto &merge = std::get<RequiredMergeExecution>(instance.id.source);
      if (merge.work != workId ||
          !observedMergeExecutions.insert(instance.id).second ||
          llvm::none_of(work->second->merges,
                        [&](const analysis::ReductionMergeRequirement &entry) {
                          return entry.group == merge.group;
                        }))
        return broken(BrokenTemporalPlanReason::MissingExecution,
                      "region merge execution has no exact merge work", workId);
    }
  }

  if (observedRootExecutions != expectedRootExecutions ||
      observedMergeExecutions != expectedMergeExecutions)
    return broken(BrokenTemporalPlanReason::MissingExecution,
                  "canonical region plan does not cover every execution");
  llvm::sort(result.scopes,
             [](const TemporalScopePlan &lhs, const TemporalScopePlan &rhs) {
               return lhs.id < rhs.id;
             });
  return result;
}

} // namespace wafer::compiler::detail
