//===- ExecutionOccurrence.h - Exact finite occurrence -------*- C++ -*-===//

#ifndef WAFER_PLANNING_PHYSICALDATAFLOW_EXECUTIONOCCURRENCE_H
#define WAFER_PLANNING_PHYSICALDATAFLOW_EXECUTIONOCCURRENCE_H

#include "Wafer/Planning/PhysicalDataflow/TemporalPlan.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <tuple>

namespace wafer::compiler::detail {

/// Exact finite occurrence class derived from one selected temporal scope.
/// Per-axis counts retain a multi-dimensional recurrence without flattening
/// its semantic identity into a loop ordinal.
struct OccurrenceRelationId {
  TraversalScopeId scope;
  llvm::SmallVector<uint64_t, 4> axisOccurrences;

  friend bool operator==(const OccurrenceRelationId &lhs,
                         const OccurrenceRelationId &rhs) {
    return lhs.scope == rhs.scope && lhs.axisOccurrences == rhs.axisOccurrences;
  }
  friend bool operator<(const OccurrenceRelationId &lhs,
                        const OccurrenceRelationId &rhs) {
    return std::tie(lhs.scope, lhs.axisOccurrences) <
           std::tie(rhs.scope, rhs.axisOccurrences);
  }
};

} // namespace wafer::compiler::detail

#endif // WAFER_PLANNING_PHYSICALDATAFLOW_EXECUTIONOCCURRENCE_H
