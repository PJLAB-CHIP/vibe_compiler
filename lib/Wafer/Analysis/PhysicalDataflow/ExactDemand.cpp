//===- ExactDemand.cpp - Exact logical dependency proof ----------------===//

#include "Wafer/Analysis/PhysicalDataflow/ExactDemand.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>

namespace wafer::analysis {

namespace {

bool boxesOverlap(const StaticRectangularIndexSet &lhs,
                  const StaticRectangularIndexSet &rhs) {
  if (lhs.offsets.size() != rhs.offsets.size())
    return true;
  for (auto [lhsOffset, lhsSize, rhsOffset, rhsSize] :
       llvm::zip_equal(lhs.offsets, lhs.sizes, rhs.offsets, rhs.sizes)) {
    int64_t lhsEnd = 0;
    int64_t rhsEnd = 0;
    if (llvm::AddOverflow(lhsOffset, lhsSize, lhsEnd) ||
        llvm::AddOverflow(rhsOffset, rhsSize, rhsEnd))
      return true;
    if (lhsEnd <= rhsOffset || rhsEnd <= lhsOffset)
      return false;
  }
  return true;
}

class EqualityLocalEliminatingPolyhedron
    : public mlir::presburger::IntegerPolyhedron {
public:
  explicit EqualityLocalEliminatingPolyhedron(
      const mlir::presburger::IntegerRelation &relation)
      : IntegerPolyhedron(relation) {}

  void eliminateEqualityLocals() { removeRedundantLocalVars(); }
};

} // namespace

ExactIndexSet::ExactIndexSet()
    : set(mlir::presburger::PresburgerSet::getEmpty(
          mlir::presburger::PresburgerSpace::getSetSpace(0))) {}

ExactIndexSet::ExactIndexSet(
    mlir::presburger::PresburgerSet set, ExactIndexSetForm form,
    llvm::ArrayRef<StaticRectangularIndexSet> boxes)
    : set(std::move(set)), form(form), boxes(boxes.begin(), boxes.end()) {}

mlir::FailureOr<ExactIndexSet>
normalizeFiniteExactIndexSet(const ExactIndexSet &domain) {
  if (domain.getForm() == ExactIndexSetForm::BoxUnion)
    return domain;
  if (domain.isEmpty())
    return ExactIndexSet(domain.getPresburgerSet(),
                         ExactIndexSetForm::BoxUnion);

  std::optional<mlir::presburger::PresburgerSet> simplified;
  for (const mlir::presburger::IntegerRelation &disjunct :
       domain.getPresburgerSet().getAllDisjuncts()) {
    EqualityLocalEliminatingPolyhedron polyhedron(disjunct);
    polyhedron.eliminateEqualityLocals();
    polyhedron.simplify();
    mlir::presburger::PresburgerSet piece(polyhedron);
    if (!simplified)
      simplified = std::move(piece);
    else
      simplified->unionInPlace(piece);
  }
  if (!simplified)
    return mlir::failure();
  IndexSetResult exact{IndexRelationStatus::Exact, std::move(*simplified), {}};
  StaticRectangularIndexSetPiecesResult pieces =
      exact.getExactStaticRectangularDisjuncts();
  if (pieces.isExact() && !pieces.domains.empty()) {
    llvm::sort(pieces.domains,
               [](const StaticRectangularIndexSet &lhs,
                  const StaticRectangularIndexSet &rhs) {
                 if (lhs.offsets != rhs.offsets)
                   return std::lexicographical_compare(
                       lhs.offsets.begin(), lhs.offsets.end(),
                       rhs.offsets.begin(), rhs.offsets.end());
                 return std::lexicographical_compare(
                     lhs.sizes.begin(), lhs.sizes.end(), rhs.sizes.begin(),
                     rhs.sizes.end());
               });
    bool disjoint = true;
    for (size_t lhs = 0; lhs < pieces.domains.size() && disjoint; ++lhs)
      for (size_t rhs = lhs + 1; rhs < pieces.domains.size(); ++rhs)
        if (boxesOverlap(pieces.domains[lhs], pieces.domains[rhs])) {
          disjoint = false;
          break;
        }
    if (disjoint)
      return ExactIndexSet(*exact.set, ExactIndexSetForm::BoxUnion,
                           pieces.domains);
  }

  StaticRectangularIndexSetResult rectangle =
      exact.getExactStaticRectangularDomain();
  if (!rectangle.isExact())
    return mlir::failure();
  return ExactIndexSet(*exact.set, ExactIndexSetForm::BoxUnion,
                       {*rectangle.domain});
}

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
