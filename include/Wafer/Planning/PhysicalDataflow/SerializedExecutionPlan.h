//===- SerializedExecutionPlan.h - Serialized execution coverage -*- C++
//-*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_SERIALIZEDEXECUTIONPLAN_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_SERIALIZEDEXECUTIONPLAN_H

#include "Wafer/Planning/PhysicalDataflow/RegionPlan.h"

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace wafer::compiler::detail {

/// Canonical execution-structure point. The existing ExecutionInstanceId is
/// the identity; this plan only states that every required execution occurs
/// once in serialized form. It carries no stage, slot, event, or worker facts.
struct SerializedExecutionPlan {
  std::vector<ExecutionInstanceId> executions;
};

enum class BrokenSerializedExecutionPlanReason : uint8_t {
  EmptyExecutionSet,
  DuplicateExecution,
  MissingTemporalScope,
  DuplicateTemporalScope,
  UnexpectedTemporalScope,
};

struct BrokenSerializedExecutionPlan {
  BrokenSerializedExecutionPlanReason reason =
      BrokenSerializedExecutionPlanReason::EmptyExecutionSet;
  std::optional<ExecutionInstanceId> execution;
  std::string detail;
};

using CanonicalSerializedExecutionPlanOutcome =
    std::variant<SerializedExecutionPlan, BrokenSerializedExecutionPlan>;

const SerializedExecutionPlan *getSerializedExecutionPlan(
    const CanonicalSerializedExecutionPlanOutcome &outcome);

} // namespace wafer::compiler::detail

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_SERIALIZEDEXECUTIONPLAN_H
