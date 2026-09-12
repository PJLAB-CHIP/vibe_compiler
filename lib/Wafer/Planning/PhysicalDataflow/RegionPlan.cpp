//===- RegionPlan.cpp - Region and execution planning schema -----------===//

#include "Wafer/Planning/PhysicalDataflow/RegionPlan.h"
#include "llvm/ADT/STLExtras.h"

namespace wafer::compiler::detail {

std::vector<ExternalUseBinding>
getRegionExternalInputs(const RegionGroupPlan &group) {
  std::vector<ExternalUseBinding> inputs = group.externalBindings;
  for (const auto &replica : group.replicas)
    llvm::append_range(inputs, replica.inputs);
  llvm::sort(inputs);
  inputs.erase(std::unique(inputs.begin(), inputs.end()), inputs.end());
  return inputs;
}

const RegionPlan *getRegionPlan(const CanonicalRegionPlanOutcome &outcome) {
  return std::get_if<RegionPlan>(&outcome);
}

} // namespace wafer::compiler::detail
