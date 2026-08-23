//===- ExecutionStructurePlan.h - Typed cyclic structure ----*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_EXECUTIONSTRUCTUREPLAN_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_EXECUTIONSTRUCTUREPLAN_H

#include "Wafer/Planning/PhysicalDataflow/EventGraph.h"
#include "Wafer/Planning/PhysicalDataflow/ExecutionOccurrence.h"

#include <cstdint>
#include <tuple>
#include <variant>
#include <vector>

namespace wafer::compiler::detail {

/// One connected event component and every temporal recurrence represented
/// in it. A component with zero or multiple recurrences is a valid Serialized
/// scope but is not advertised as one cyclic pipeline.
struct PipelineScopeId {
  std::vector<EventId> events;
  std::vector<OccurrenceRelationId> recurrences;

  friend bool operator==(const PipelineScopeId &lhs,
                         const PipelineScopeId &rhs) {
    return lhs.events == rhs.events && lhs.recurrences == rhs.recurrences;
  }
  friend bool operator<(const PipelineScopeId &lhs,
                        const PipelineScopeId &rhs) {
    return std::tie(lhs.events, lhs.recurrences) <
           std::tie(rhs.events, rhs.recurrences);
  }
};

class StageId {
public:
  explicit constexpr StageId(uint32_t value) : value(value) {}
  constexpr uint32_t getValue() const { return value; }

  friend constexpr bool operator==(StageId lhs, StageId rhs) {
    return lhs.value == rhs.value;
  }
  friend constexpr bool operator!=(StageId lhs, StageId rhs) {
    return !(lhs == rhs);
  }
  friend constexpr bool operator<(StageId lhs, StageId rhs) {
    return lhs.value < rhs.value;
  }

private:
  uint32_t value;
};

struct EventStageAssignment {
  EventId event;
  StageId stage{0};

  friend bool operator==(const EventStageAssignment &lhs,
                         const EventStageAssignment &rhs) {
    return lhs.event == rhs.event && lhs.stage == rhs.stage;
  }
  friend bool operator<(const EventStageAssignment &lhs,
                        const EventStageAssignment &rhs) {
    return std::tie(lhs.event, lhs.stage) < std::tie(rhs.event, rhs.stage);
  }
};

struct SerializedExecutionStructure {
  PipelineScopeId scope;

  friend bool operator==(const SerializedExecutionStructure &lhs,
                         const SerializedExecutionStructure &rhs) {
    return lhs.scope == rhs.scope;
  }
  friend bool operator<(const SerializedExecutionStructure &lhs,
                        const SerializedExecutionStructure &rhs) {
    return lhs.scope < rhs.scope;
  }
};

struct PipelinedExecutionStructure {
  PipelineScopeId scope;
  OccurrenceRelationId recurrence;
  uint64_t launchDistance = 1;
  std::vector<EventStageAssignment> eventStages;

  friend bool operator==(const PipelinedExecutionStructure &lhs,
                         const PipelinedExecutionStructure &rhs) {
    return lhs.scope == rhs.scope && lhs.recurrence == rhs.recurrence &&
           lhs.launchDistance == rhs.launchDistance &&
           lhs.eventStages == rhs.eventStages;
  }
  friend bool operator<(const PipelinedExecutionStructure &lhs,
                        const PipelinedExecutionStructure &rhs) {
    return std::tie(lhs.scope, lhs.recurrence, lhs.launchDistance,
                    lhs.eventStages) < std::tie(rhs.scope, rhs.recurrence,
                                                rhs.launchDistance,
                                                rhs.eventStages);
  }
};

using ExecutionStructureChoice =
    std::variant<SerializedExecutionStructure, PipelinedExecutionStructure>;

struct ExecutionStructurePlan {
  std::vector<ExecutionStructureChoice> scopes;

  friend bool operator==(const ExecutionStructurePlan &lhs,
                         const ExecutionStructurePlan &rhs) {
    return lhs.scopes == rhs.scopes;
  }
  friend bool operator<(const ExecutionStructurePlan &lhs,
                        const ExecutionStructurePlan &rhs) {
    return lhs.scopes < rhs.scopes;
  }
};

const PipelineScopeId &getPipelineScope(const ExecutionStructureChoice &choice);

} // namespace wafer::compiler::detail

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_EXECUTIONSTRUCTUREPLAN_H
