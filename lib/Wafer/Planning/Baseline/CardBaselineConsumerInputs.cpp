//===- CardBaselineConsumerInputs.cpp --------------------------------===//

#include "Wafer/Planning/Baseline/CardBaselineConsumerInputs.h"

namespace wafer::compiler::detail {

mlir::LogicalResult addCardBaselineConsumerInputs(
    TileMapping &mapping, const analysis::ExactDemandProof &demand,
    std::string *failureReason) {
  (void)failureReason;
  mapping.operandDemands.assign(demand.dependencyDemands.begin(),
                                demand.dependencyDemands.end());
  return mlir::success();
}

} // namespace wafer::compiler::detail
