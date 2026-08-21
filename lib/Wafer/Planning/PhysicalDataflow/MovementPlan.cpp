//===- MovementPlan.cpp - Explicit canonical movement schema ----------===//

#include "Wafer/Planning/PhysicalDataflow/MovementPlan.h"

namespace wafer::compiler::detail {

const CanonicalMovementCoordinate *
getCanonicalMovementCoordinate(const CanonicalMovementPlanOutcome &outcome) {
  return std::get_if<CanonicalMovementCoordinate>(&outcome);
}

} // namespace wafer::compiler::detail
