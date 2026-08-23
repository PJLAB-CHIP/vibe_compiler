//===- TemporalPlan.cpp - Per-execution temporal plan schema ----------===//

#include "Wafer/Planning/PhysicalDataflow/TemporalPlan.h"

namespace wafer::compiler::detail {

const TemporalPlan *
getTemporalPlan(const CanonicalTemporalPlanOutcome &outcome) {
  return std::get_if<TemporalPlan>(&outcome);
}

const ExecutionInstanceId *getRequiredExecution(const TraversalScopeId &scope) {
  return std::get_if<ExecutionInstanceId>(&scope.execution);
}

bool isTopLevelScope(const TraversalScopeId &scope) {
  return std::holds_alternative<TopLevelWorkPieceId>(scope.invocation);
}

} // namespace wafer::compiler::detail
