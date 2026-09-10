//===- PlanningProblem.cpp - Immutable physical planning input ----------===//

#include "Wafer/Planning/PhysicalDataflow/PlanningProblem.h"

namespace wafer::compiler::detail {

std::variant<PhysicalDataflowPlanningProblem, SpatialDomainFailure>
PhysicalDataflowPlanningProblem::create(
    const StructuredProgramAnalysis &program, CardId cardId,
    const analysis::IndexRelationLimits &relationLimits) {
  SpatialPlanDomainResult spatial =
      buildSpatialPlanDomain(program.dag, program.topology, cardId);
  if (!spatial.succeeded()) {
    return spatial.failure.value_or(SpatialDomainFailure{
        SpatialDomainFailureKind::BrokenContract, std::nullopt,
        "spatial planning problem returned no detail"});
  }
  return PhysicalDataflowPlanningProblem(
      program, cardId, std::move(*spatial.domain), relationLimits);
}

} // namespace wafer::compiler::detail
