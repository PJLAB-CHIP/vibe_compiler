//===- CanonicalSerializedExecutionPlan.h - Serialized point -*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CANONICALSERIALIZEDEXECUTIONPLAN_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CANONICALSERIALIZEDEXECUTIONPLAN_H

#include "Wafer/Planning/PhysicalDataflow/SerializedExecutionPlan.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalPlan.h"

namespace wafer::compiler::detail {

/// Builds the canonical serialized execution point from closed region and
/// temporal plans. The query emits no IR and introduces no pipeline scope.
CanonicalSerializedExecutionPlanOutcome
buildCanonicalSerializedExecutionPlan(const RegionPlan &regions,
                                      const TemporalPlan &temporal);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CANONICALSERIALIZEDEXECUTIONPLAN_H
