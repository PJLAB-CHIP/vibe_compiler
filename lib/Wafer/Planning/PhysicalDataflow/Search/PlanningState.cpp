//===- PlanningState.cpp - Closed physical planning prefixes -----------===//

#include "Wafer/Planning/PhysicalDataflow/Search/PlanningState.h"

#include "Wafer/Planning/PhysicalDataflow/MovementDomain.h"
#include "Wafer/Planning/PhysicalDataflow/RegionDomain.h"
#include "Wafer/Planning/PhysicalDataflow/RepresentationDomain.h"
#include "Wafer/Planning/PhysicalDataflow/Search/PlanningProblem.h"
#include "Wafer/Planning/PhysicalDataflow/TemporalDomain.h"

namespace wafer::compiler::detail {

mlir::FailureOr<SpatialState>
SpatialState::create(const PhysicalDataflowPlanningProblem &problem,
                     SpatialPlan plan, std::string *failureReason) {
  if (!problem.getSpatialDomain().contains(plan)) {
    if (failureReason)
      *failureReason = "SpatialState plan is outside the current domain";
    return mlir::failure();
  }
  return SpatialState(std::move(plan));
}

mlir::FailureOr<RegionState> RegionState::create(const RegionDomain &domain,
                                                 SpatialState spatial,
                                                 RegionPlan regions,
                                                 std::string *failureReason) {
  if (!domain.contains(regions)) {
    if (failureReason)
      *failureReason = "RegionState plan is outside the current domain";
    return mlir::failure();
  }
  return RegionState(std::move(spatial), std::move(regions));
}

mlir::FailureOr<TemporalState>
TemporalState::create(const TemporalDomain &domain, RegionState region,
                      TemporalPlan temporal, std::string *failureReason) {
  if (!domain.contains(temporal)) {
    if (failureReason)
      *failureReason = "TemporalState plan is outside the current domain";
    return mlir::failure();
  }
  return TemporalState(std::move(region), std::move(temporal));
}

mlir::FailureOr<RepresentationState> RepresentationState::create(
    const RepresentationDomain &domain, TemporalState temporal,
    RepresentationPlan representations, std::string *failureReason) {
  if (!domain.contains(representations)) {
    if (failureReason)
      *failureReason = "RepresentationState plan is outside the current domain";
    return mlir::failure();
  }
  return RepresentationState(std::move(temporal), std::move(representations));
}

mlir::FailureOr<MovementState>
MovementState::create(const MovementDomain &domain,
                      RepresentationState representations,
                      MovementPlan movement, std::string *failureReason) {
  if (!domain.contains(movement)) {
    if (failureReason)
      *failureReason = "MovementState plan is outside the current domain";
    return mlir::failure();
  }
  return MovementState(std::move(representations), std::move(movement));
}

} // namespace wafer::compiler::detail
