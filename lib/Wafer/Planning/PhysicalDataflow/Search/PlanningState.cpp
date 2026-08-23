//===- PlanningState.cpp - Closed physical planning prefixes -----------===//

#include "Wafer/Planning/PhysicalDataflow/Search/PlanningState.h"

#include "Wafer/Planning/PhysicalDataflow/Search/PlanningProblem.h"

namespace wafer::compiler::detail {

llvm::StringRef
stringifyRequiredPlanningCoordinate(RequiredPlanningCoordinate coordinate) {
  switch (coordinate) {
  case RequiredPlanningCoordinate::Region:
    return "region";
  }
  return "unknown";
}

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

} // namespace wafer::compiler::detail
