//===- CanonicalBaselinePlan.h - Resolved deterministic plan -*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_BASELINE_CANONICALBASELINEPLAN_H
#define WAFER_COMPILER_PLANNING_BASELINE_CANONICALBASELINEPLAN_H

#include "Wafer/Analysis/Structured/CardProgramAnalysis.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalAttentionWorkProjection.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalFeasibilityProof.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalMovementPlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalRepresentationPlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalSchedulePlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalSerializedExecutionPlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalStoragePlan.h"
#include "Wafer/Planning/PhysicalDataflow/SelectedAttentionDecomposition.h"

namespace wafer::compiler::detail {

/// Complete resolved canonical coordinate. It is query-local, borrows current
/// root Operation handles through RootRegionWork, and must be consumed before
/// the first source mutation. No field is a search candidate or actual IR.
struct CanonicalBaselinePlan {
  SpatialAssignment spatial;
  analysis::ExactDemandProof demand;
  std::vector<analysis::RootRegionWork> rootWorks;
  RegionPlan regions;
  TemporalPlan temporal;
  CanonicalRepresentationCoordinate representations;
  CanonicalMovementCoordinate movements;
  SerializedExecutionPlan serialized;
  CanonicalStorageCoordinate storage;
  CanonicalScheduleCoordinate schedule;
  CanonicalAttentionWorkCoordinate attention;
  CanonicalResourceProblem resourceProblem;
  FullFeasibilityProof feasibility;
  PreparedAttentionDecomposition preparedAttention;
};

mlir::FailureOr<CanonicalBaselinePlan>
buildCanonicalBaselinePlan(const CardProgramAnalysis &program,
                           const TargetMemoryPolicy &memory,
                           std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_BASELINE_CANONICALBASELINEPLAN_H
