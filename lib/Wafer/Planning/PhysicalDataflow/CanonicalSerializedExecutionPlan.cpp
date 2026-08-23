//===- CanonicalSerializedExecutionPlan.cpp - Serialized point -------===//

#include "Wafer/Planning/PhysicalDataflow/CanonicalSerializedExecutionPlan.h"

#include "llvm/ADT/StringRef.h"

#include <set>
#include <utility>

namespace wafer::compiler::detail {
namespace {

CanonicalSerializedExecutionPlanOutcome
broken(BrokenSerializedExecutionPlanReason reason, llvm::StringRef detail,
       std::optional<ExecutionInstanceId> execution = std::nullopt) {
  return BrokenSerializedExecutionPlan{reason, std::move(execution),
                                       detail.str()};
}

} // namespace

CanonicalSerializedExecutionPlanOutcome
buildCanonicalSerializedExecutionPlan(const RegionPlan &regions,
                                      const TemporalPlan &temporal) {
  std::set<ExecutionInstanceId> allExecutions;
  std::set<ExecutionInstanceId> rootExecutions;
  for (const RegionGroupPlan &group : regions.groups) {
    for (const ExecutionInstancePlan &instance : group.executions) {
      if (!allExecutions.insert(instance.id).second)
        return broken(BrokenSerializedExecutionPlanReason::DuplicateExecution,
                      "serialized input has a duplicate region execution",
                      instance.id);
      if (std::holds_alternative<RequiredRootExecution>(instance.id.source))
        rootExecutions.insert(instance.id);
    }
  }
  if (allExecutions.empty())
    return broken(BrokenSerializedExecutionPlanReason::EmptyExecutionSet,
                  "serialized input has no required execution");

  std::set<ExecutionInstanceId> observedTemporalScopes;
  for (const TemporalScopePlan &scope : temporal.scopes) {
    const ExecutionInstanceId *execution = getRequiredExecution(scope.id);
    if (!execution || !isTopLevelScope(scope.id) ||
        !std::holds_alternative<RequiredRootExecution>(execution->source) ||
        !rootExecutions.count(*execution))
      return broken(
          BrokenSerializedExecutionPlanReason::UnexpectedTemporalScope,
          "temporal scope does not name a required root execution",
          execution ? std::optional<ExecutionInstanceId>(*execution)
                    : std::nullopt);
    if (!observedTemporalScopes.insert(*execution).second)
      return broken(BrokenSerializedExecutionPlanReason::DuplicateTemporalScope,
                    "serialized input has a duplicate temporal scope",
                    *execution);
  }

  if (observedTemporalScopes != rootExecutions) {
    for (const ExecutionInstanceId &execution : rootExecutions)
      if (!observedTemporalScopes.count(execution))
        return broken(BrokenSerializedExecutionPlanReason::MissingTemporalScope,
                      "required root execution has no temporal scope",
                      execution);
  }

  SerializedExecutionPlan result;
  result.executions.assign(allExecutions.begin(), allExecutions.end());
  return result;
}

} // namespace wafer::compiler::detail
