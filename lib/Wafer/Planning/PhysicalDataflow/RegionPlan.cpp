//===- RegionPlan.cpp - Region and execution planning schema -----------===//

#include "Wafer/Planning/PhysicalDataflow/RegionPlan.h"

namespace wafer::compiler::detail {

const RegionPlan *getRegionPlan(const CanonicalRegionPlanOutcome &outcome) {
  return std::get_if<RegionPlan>(&outcome);
}

} // namespace wafer::compiler::detail
