//===- ExactDemand.cpp - Exact logical dependency proof ----------------===//

#include "Wafer/Analysis/PhysicalDataflow/ExactDemand.h"

namespace wafer::analysis {

ExactIndexSet::ExactIndexSet()
    : set(mlir::presburger::PresburgerSet::getEmpty(
          mlir::presburger::PresburgerSpace::getSetSpace(0))) {}

ExactIndexSet::ExactIndexSet(
    mlir::presburger::PresburgerSet set, ExactIndexSetForm form,
    llvm::ArrayRef<StaticRectangularIndexSet> boxes)
    : set(std::move(set)), form(form), boxes(boxes.begin(), boxes.end()) {}

ExactDemandOutcomeCategory
classifyExactDemandOutcome(const ExactDemandOutcome &outcome) {
  return std::visit(
      [](const auto &value) -> ExactDemandOutcomeCategory {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, ExactDemandProof>)
          return ExactDemandOutcomeCategory::Satisfied;
        if constexpr (std::is_same_v<T, UnsupportedDemandSemantics>)
          return ExactDemandOutcomeCategory::UnsupportedSemantics;
        if constexpr (std::is_same_v<T, DemandWorkLimitReached>)
          return ExactDemandOutcomeCategory::IndeterminateResourceExhaustion;
        return ExactDemandOutcomeCategory::CompilerContractError;
      },
      outcome);
}

const ExactDemandProof *getExactDemandProof(const ExactDemandOutcome &outcome) {
  return std::get_if<ExactDemandProof>(&outcome);
}

} // namespace wafer::analysis
