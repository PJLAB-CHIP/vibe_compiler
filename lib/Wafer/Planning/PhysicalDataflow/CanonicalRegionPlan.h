//===- CanonicalRegionPlan.h - Singleton region plan ---------*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CANONICALREGIONPLAN_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CANONICALREGIONPLAN_H

#include "Wafer/Planning/PhysicalDataflow/RegionPlan.h"

#include "llvm/ADT/ArrayRef.h"

namespace wafer::compiler::detail {

/// Builds the deterministic singleton region coordinate from already-derived
/// root work. The query emits no IR and makes no temporal, representation,
/// movement, storage, schedule, or search-domain choice.
CanonicalRegionPlanOutcome
buildCanonicalRegionPlan(llvm::ArrayRef<analysis::RootRegionWork> rootWorks);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CANONICALREGIONPLAN_H
