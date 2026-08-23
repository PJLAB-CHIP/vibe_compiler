//===- BaselineTemporalPlan.h - Deterministic temporal start -*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_BASELINE_BASELINETEMPORALPLAN_H
#define WAFER_COMPILER_PLANNING_BASELINE_BASELINETEMPORALPLAN_H

#include "Wafer/Planning/PhysicalDataflow/TemporalPlan.h"

#include "llvm/ADT/ArrayRef.h"

namespace wafer::compiler::detail {

/// Starts from the complete local iterator domain. This function does not
/// inspect SPM capacity or predict lowering allocations.
CanonicalTemporalPlanOutcome
buildBaselineTemporalPlan(const RegionPlan &regions,
                          llvm::ArrayRef<analysis::RootRegionWork> rootWorks);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_BASELINE_BASELINETEMPORALPLAN_H
