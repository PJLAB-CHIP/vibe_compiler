//===- CardBaselineConsumerInputs.cpp --------------------------------===//

#include "Wafer/Compiler/Baseline/CardBaselineConsumerInputs.h"

#include <set>

namespace wafer::compiler::detail {

mlir::LogicalResult addCardBaselineConsumerInputs(
    TileMapping &mapping, const StructuredDAGAnalysis &dag,
    const analysis::LogicalShardTrial &trial,
    StructuredDAGExactDemandQuery &query, std::string *failureReason) {
  mapping.operandDemands.clear();

  std::set<std::pair<StructuredDAGNodeID, uint32_t>> inputs;
  for (const StructuredDAGEdge &edge : dag.getEdges())
    inputs.emplace(edge.consumer, edge.consumerOperand);

  for (const auto &[consumer, operand] : inputs) {
    analysis::ConsumerInputDemand demand =
        query.queryOperand(consumer, operand, trial);
    if (demand.status != analysis::ExactDemandStatus::Satisfied) {
      if (failureReason)
        *failureReason = "consumer input demand failed: " + demand.detail;
      return mlir::failure();
    }
    mapping.operandDemands.push_back(std::move(demand));
  }
  return mlir::success();
}

} // namespace wafer::compiler::detail
