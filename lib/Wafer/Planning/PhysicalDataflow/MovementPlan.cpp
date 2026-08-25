//===- MovementPlan.cpp - Explicit canonical movement schema ----------===//

#include "Wafer/Planning/PhysicalDataflow/MovementPlan.h"

namespace wafer::compiler::detail {

llvm::StringRef stringifyMovementActionKind(const MovementActionId &action) {
  if (std::holds_alternative<ExternalLoadId>(action))
    return "external-load";
  if (std::holds_alternative<DDRBoundaryTransferId>(action))
    return "ddr-boundary-transfer";
  if (std::holds_alternative<ReductionGatherId>(action))
    return "reduction-gather";
  return "result-publication";
}

const CanonicalMovementCoordinate *
getCanonicalMovementCoordinate(const CanonicalMovementPlanOutcome &outcome) {
  return std::get_if<CanonicalMovementCoordinate>(&outcome);
}

} // namespace wafer::compiler::detail
