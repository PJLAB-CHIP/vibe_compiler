//===- BufferPlan.cpp - Physical storage planning schema --------------===//

#include "Wafer/Planning/PhysicalDataflow/BufferPlan.h"

namespace wafer::compiler::detail {

const CanonicalStorageCoordinate *
getCanonicalStorageCoordinate(const CanonicalStoragePlanOutcome &outcome) {
  return std::get_if<CanonicalStorageCoordinate>(&outcome);
}

} // namespace wafer::compiler::detail
