//===- FeasibilityProof.cpp - Canonical resource proof schema ---------===//

#include "Wafer/Planning/PhysicalDataflow/FeasibilityProof.h"

namespace wafer::compiler::detail {

const CanonicalFeasibilityCoordinate *
getCanonicalFeasibilityCoordinate(const CanonicalFeasibilityOutcome &outcome) {
  return std::get_if<CanonicalFeasibilityCoordinate>(&outcome);
}

} // namespace wafer::compiler::detail
