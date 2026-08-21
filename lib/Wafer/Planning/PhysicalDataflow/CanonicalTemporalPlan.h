//===- CanonicalTemporalPlan.h - Full-local temporal point ---*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CANONICALTEMPORALPLAN_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CANONICALTEMPORALPLAN_H

#include "Wafer/Planning/PhysicalDataflow/TemporalPlan.h"

#include "llvm/ADT/ArrayRef.h"

namespace wafer::compiler::detail {

/// Builds the full-local, one-wave temporal point for a canonical singleton
/// region plan. The query emits no IR and reads no target or capacity facts.
CanonicalTemporalPlanOutcome
buildCanonicalTemporalPlan(const RegionPlan &regions,
                           llvm::ArrayRef<analysis::RootRegionWork> rootWorks);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_CANONICALTEMPORALPLAN_H
