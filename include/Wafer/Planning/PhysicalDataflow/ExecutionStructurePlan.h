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

enum class PipelineDependenceKind : uint8_t {
  DataReady,
  AsyncCompletion,
  BufferLifetime,
  Effect,
};

/// A dependence between two recurring event classes. Distance zero belongs to
/// one logical iteration; distance one consumes the immediately preceding
/// iteration. Current lowering supports no larger distance.
struct PipelineDependence {
  EventId source;
  EventId destination;
  uint32_t iterationDistance = 0;
  PipelineDependenceKind kind = PipelineDependenceKind::DataReady;

  friend bool operator==(const PipelineDependence &lhs,
                         const PipelineDependence &rhs) {
    return std::tie(lhs.source, lhs.destination, lhs.iterationDistance,
                    lhs.kind) == std::tie(rhs.source, rhs.destination,
                                          rhs.iterationDistance, rhs.kind);
  }
  friend bool operator<(const PipelineDependence &lhs,
                        const PipelineDependence &rhs) {
    return std::tie(lhs.source, lhs.destination, lhs.iterationDistance,
                    lhs.kind) < std::tie(rhs.source, rhs.destination,
                                         rhs.iterationDistance, rhs.kind);
  }
};

/// Exact decomposition of one selected temporal axis as emitted by the E
/// compact traversal: one prefix full wave, a static steady loop, and at most
/// one remainder wave. Outer axes repeat this same structure.
struct PipelineIterationClass {
  uint32_t recurrenceAxis = 0;
  uint64_t prefixCount = 1;
  uint64_t steadyTripCount = 0;
  uint64_t tailCount = 0;

  friend bool operator==(const PipelineIterationClass &lhs,
                         const PipelineIterationClass &rhs) {
    return std::tie(lhs.recurrenceAxis, lhs.prefixCount, lhs.steadyTripCount,
                    lhs.tailCount) ==
           std::tie(rhs.recurrenceAxis, rhs.prefixCount, rhs.steadyTripCount,
                    rhs.tailCount);
  }
  friend bool operator<(const PipelineIterationClass &lhs,
                        const PipelineIterationClass &rhs) {
    return std::tie(lhs.recurrenceAxis, lhs.prefixCount, lhs.steadyTripCount,
                    lhs.tailCount) <
           std::tie(rhs.recurrenceAxis, rhs.prefixCount, rhs.steadyTripCount,
                    rhs.tailCount);
  }
};

enum class ExecutionStructureLowering : uint8_t {
  SCFDistanceOne,
  FiniteUnrolled,
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
  PipelineIterationClass iteration;
  uint64_t launchDistance = 1;
  std::vector<EventStageAssignment> eventStages;
  std::vector<PipelineDependence> dependences;
  std::vector<CompletionObligation> completionObligations;
  ExecutionStructureLowering lowering =
      ExecutionStructureLowering::SCFDistanceOne;

  friend bool operator==(const PipelinedExecutionStructure &lhs,
                         const PipelinedExecutionStructure &rhs) {
    return lhs.scope == rhs.scope && lhs.recurrence == rhs.recurrence &&
           lhs.iteration == rhs.iteration &&
           lhs.launchDistance == rhs.launchDistance &&
           lhs.eventStages == rhs.eventStages &&
           lhs.dependences == rhs.dependences &&
           lhs.completionObligations == rhs.completionObligations &&
           lhs.lowering == rhs.lowering;
  }
  friend bool operator<(const PipelinedExecutionStructure &lhs,
                        const PipelinedExecutionStructure &rhs) {
    return std::tie(lhs.scope, lhs.recurrence, lhs.iteration,
                    lhs.launchDistance, lhs.eventStages, lhs.dependences,
                    lhs.completionObligations, lhs.lowering) <
           std::tie(rhs.scope, rhs.recurrence, rhs.iteration,
                    rhs.launchDistance, rhs.eventStages, rhs.dependences,
                    rhs.completionObligations, rhs.lowering);
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
