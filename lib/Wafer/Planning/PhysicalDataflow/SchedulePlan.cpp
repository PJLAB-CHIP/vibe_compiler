//===- SchedulePlan.cpp - Canonical schedule planning schema ----------===//

#include "Wafer/Planning/PhysicalDataflow/SchedulePlan.h"

namespace wafer::compiler::detail {

const CanonicalScheduleCoordinate *
getCanonicalScheduleCoordinate(const CanonicalSchedulePlanOutcome &outcome) {
  return std::get_if<CanonicalScheduleCoordinate>(&outcome);
}

} // namespace wafer::compiler::detail
