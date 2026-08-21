//===- CanonicalSchedulePlan.h - Stable worker0 order --------*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CANONICALSCHEDULEPLAN_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CANONICALSCHEDULEPLAN_H

#include "Wafer/Planning/PhysicalDataflow/SchedulePlan.h"
#include "Wafer/Planning/PhysicalDataflow/SerializedExecutionPlan.h"

namespace wafer::compiler::detail {

/// Builds the stable worker0 topological point from canonical storage
/// definition/use lifetimes. The query emits no IR and introduces no event,
/// completion, resource, timestamp, or pipeline facts.
CanonicalSchedulePlanOutcome
buildCanonicalSchedulePlan(const CanonicalStorageCoordinate &storage,
                           const SerializedExecutionPlan &serialized);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CANONICALSCHEDULEPLAN_H
