//===- CardBaselineConsumerInputs.cpp --------------------------------===//

#include "Wafer/Planning/Baseline/CardBaselineConsumerInputs.h"
#include "Wafer/Support/CompileTiming.h"

#include "llvm/ADT/Twine.h"

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
    std::string detail;
    if (wafer::support::getActiveCompileTimingSession())
      detail = (llvm::Twine("consumer=") + llvm::Twine(consumer) +
                " operand=" + llvm::Twine(operand))
                   .str();
    analysis::ConsumerInputDemand demand = [&]() {
      wafer::support::ScopedCompileTimingSpan timing(
          "query", "deterministic-baseline", "construct-consumer-input",
          detail);
      return query.queryOperand(consumer, operand, trial);
    }();
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
