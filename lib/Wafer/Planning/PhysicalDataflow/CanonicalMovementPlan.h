//===- CanonicalMovementPlan.h - Deterministic DDR carrier ---*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CANONICALMOVEMENTPLAN_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CANONICALMOVEMENTPLAN_H

#include "Wafer/Planning/PhysicalDataflow/MovementPlan.h"

#include "llvm/ADT/ArrayRef.h"

namespace wafer::compiler::detail {

/// Builds the deterministic external-load/DDR/gather/publication carrier for
/// the canonical B--G coordinate. The query emits no IR and chooses no peer,
/// relay, reuse, buffer, event, or schedule facts.
CanonicalMovementPlanOutcome buildCanonicalMovementPlan(
    const RegionPlan &regions,
    const CanonicalRepresentationCoordinate &representations,
    llvm::ArrayRef<analysis::RootRegionWork> rootWorks);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CANONICALMOVEMENTPLAN_H
