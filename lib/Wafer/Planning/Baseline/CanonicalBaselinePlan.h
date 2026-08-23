//===- CanonicalBaselinePlan.h - Deterministic candidate plan -*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_BASELINE_CANONICALBASELINEPLAN_H
#define WAFER_COMPILER_PLANNING_BASELINE_CANONICALBASELINEPLAN_H

#include "Wafer/Analysis/Structured/CardProgramAnalysis.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalAttentionWorkProjection.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalMovementPlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalRepresentationPlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalSchedulePlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalSerializedExecutionPlan.h"
#include "Wafer/Planning/PhysicalDataflow/CanonicalStoragePlan.h"
#include "Wafer/Planning/PhysicalDataflow/SelectedAttentionDecomposition.h"

namespace wafer::compiler::detail {

/// One complete deterministic candidate before materialization. This object
/// contains semantic choices only; it never claims SPM legality. SPM legality
/// is established after materialization by the actual Instr memory planner.
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
  PreparedAttentionDecomposition preparedAttention;
};

mlir::FailureOr<CanonicalBaselinePlan>
buildCanonicalBaselinePlan(const CardProgramAnalysis &program,
                           std::string *failureReason = nullptr);

/// Rebuilds every temporal-dependent semantic component after the outer
/// controller changes `plan.temporal` in response to an actual SPM rejection.
/// This function performs no resource prediction and creates no IR.
mlir::LogicalResult
recloseCanonicalBaselinePlan(CanonicalBaselinePlan &plan,
                             std::string *failureReason = nullptr);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_BASELINE_CANONICALBASELINEPLAN_H
