//===- BaselineTemporalPlan.h - Actual-feedback temporal plan -*- C++ -*-===//

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

/// Advances one deterministic temporal coordinate after the actual Instr SPM
/// planner attributed an exact capacity rejection to `affectedRoots`.
/// Returns false when every implicated scope is already at its minimum.
mlir::FailureOr<bool>
refineBaselineTemporalPlan(
    TemporalPlan &temporal,
    llvm::ArrayRef<analysis::RootRegionWork> rootWorks,
    llvm::ArrayRef<SemanticRootKey> affectedRoots,
    std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_BASELINE_BASELINETEMPORALPLAN_H
