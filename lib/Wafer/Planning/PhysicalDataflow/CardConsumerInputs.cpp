//===- CardConsumerInputs.cpp - Selected boundary inputs --------------===//

#include "Wafer/Planning/PhysicalDataflow/CardConsumerInputs.h"

namespace wafer::compiler::detail {

mlir::LogicalResult
addCardConsumerInputs(TileMapping &mapping,
                      const analysis::ExactDemandProof &demand,
                      std::string *failureReason) {
  (void)failureReason;
  mapping.operandDemands.assign(demand.dependencyDemands.begin(),
                                demand.dependencyDemands.end());
  return mlir::success();
}

} // namespace wafer::compiler::detail
