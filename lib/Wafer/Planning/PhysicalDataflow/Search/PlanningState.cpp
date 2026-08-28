//===- PlanningState.cpp - Closed structural planning prefixes --------===//

#include "Wafer/Planning/PhysicalDataflow/Search/PlanningState.h"

#include "Wafer/Planning/PhysicalDataflow/RegionDomain.h"
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

} // namespace wafer::compiler::detail
