//===- TemporalPlan.cpp - Per-execution temporal plan schema ----------===//

#include "Wafer/Planning/PhysicalDataflow/TemporalPlan.h"

namespace wafer::compiler::detail {

const TemporalPlan *
getTemporalPlan(const CanonicalTemporalPlanOutcome &outcome) {
  return std::get_if<TemporalPlan>(&outcome);
}

} // namespace wafer::compiler::detail
