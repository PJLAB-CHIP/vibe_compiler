//===- PlanningProblem.cpp - Immutable physical planning input ----------===//

#include "Wafer/Planning/PhysicalDataflow/Search/PlanningProblem.h"

namespace wafer::compiler::detail {

mlir::FailureOr<PhysicalDataflowPlanningProblem>
PhysicalDataflowPlanningProblem::create(
    const StructuredProgramAnalysis &program, CardId cardId,
    const analysis::IndexRelationLimits &relationLimits,
    std::string *failureReason) {
  SpatialPlanDomainResult spatial =
      buildSpatialPlanDomain(program.dag, program.topology, cardId);
  if (!spatial.succeeded()) {
    if (failureReason)
      *failureReason = spatial.failure
                           ? spatial.failure->detail
                           : "spatial planning problem returned no detail";
    return mlir::failure();
  }
  return PhysicalDataflowPlanningProblem(
      program, cardId, std::move(*spatial.domain), relationLimits);
}

} // namespace wafer::compiler::detail
