//===- RepresentationPlan.cpp - Logical and physical versions --------===//

#include "Wafer/Planning/PhysicalDataflow/RepresentationPlan.h"

namespace wafer::compiler::detail {

const CanonicalRepresentationCoordinate *getCanonicalRepresentationCoordinate(
    const CanonicalRepresentationPlanOutcome &outcome) {
  return std::get_if<CanonicalRepresentationCoordinate>(&outcome);
}

} // namespace wafer::compiler::detail
