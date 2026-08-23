//===- CanonicalStoragePlan.h - Fresh single-slot storage ----*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CANONICALSTORAGEPLAN_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CANONICALSTORAGEPLAN_H

#include "Wafer/Planning/PhysicalDataflow/BufferPlan.h"
#include "Wafer/Planning/PhysicalDataflow/SerializedExecutionPlan.h"

namespace wafer::compiler::detail {

/// Builds the canonical fresh, single-slot storage point and its typed
/// definition/use lifetimes. The query emits no IR and chooses no alias,
/// reuse, rotation, order, worker, or offset facts.
CanonicalStoragePlanOutcome buildCanonicalStoragePlan(
    const CanonicalRepresentationCoordinate &representations,
    const CanonicalMovementCoordinate &movements,
    const SerializedExecutionPlan &serialized);

/// Recomputes the maximum resident slice for each canonical object from a
/// fixed temporal plan while retaining the complete logical exactDomain.
CanonicalStoragePlanOutcome recloseCanonicalStorageForTemporal(
    const CanonicalStorageCoordinate &storage, const TemporalPlan &temporal,
    llvm::ArrayRef<analysis::RootRegionWork> rootWorks);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CANONICALSTORAGEPLAN_H
