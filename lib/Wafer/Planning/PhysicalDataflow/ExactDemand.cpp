//===- ExactDemand.cpp - Exact logical dependency proof ----------------===//

#include "Wafer/Planning/PhysicalDataflow/ExactDemand.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>

namespace wafer::analysis {

namespace {

bool boxLess(const StaticRectangularIndexSet &lhs,
             const StaticRectangularIndexSet &rhs) {
  if (lhs.offsets != rhs.offsets)
    return std::lexicographical_compare(lhs.offsets.begin(), lhs.offsets.end(),
                                        rhs.offsets.begin(), rhs.offsets.end());
  return std::lexicographical_compare(lhs.sizes.begin(), lhs.sizes.end(),
                                      rhs.sizes.begin(), rhs.sizes.end());
}

// Only combine intervals with identical cross sections. Every merge is an
// exact union and strictly decreases the worklist; no bounding-box or general
// Presburger proof is needed. A fixed axis order makes the result independent
// of the input disjunct order (without promising a minimum box partition).
mlir::LogicalResult
coalesceBoxes(llvm::SmallVectorImpl<StaticRectangularIndexSet> &boxes,
              unsigned rank) {
  for (const auto &box : boxes) {
    if (box.offsets.size() != rank || box.sizes.size() != rank)
      return mlir::failure();
    for (unsigned axis = 0; axis < rank; ++axis) {
      int64_t end = 0;
      if (box.sizes[axis] <= 0 ||
          llvm::AddOverflow(box.offsets[axis], box.sizes[axis], end))
        return mlir::failure();
    }
  }
  if (rank == 0) {
    if (!boxes.empty())
      boxes.resize(1);
    return mlir::success();
  }
  size_t previousSize;
  do {
    previousSize = boxes.size();
    for (unsigned axis = rank; axis-- > 0;) {
      auto sameCrossSection = [axis, rank](const auto &lhs, const auto &rhs) {
        for (unsigned other = 0; other < rank; ++other)
          if (other != axis && (lhs.offsets[other] != rhs.offsets[other] ||
                                lhs.sizes[other] != rhs.sizes[other]))
            return false;
        return true;
      };
      llvm::sort(boxes, [axis, rank](const auto &lhs, const auto &rhs) {
        for (unsigned other = 0; other < rank; ++other) {
          if (other == axis)
            continue;
          if (lhs.offsets[other] != rhs.offsets[other])
            return lhs.offsets[other] < rhs.offsets[other];
          if (lhs.sizes[other] != rhs.sizes[other])
            return lhs.sizes[other] < rhs.sizes[other];
        }
        if (lhs.offsets[axis] != rhs.offsets[axis])
          return lhs.offsets[axis] < rhs.offsets[axis];
        return lhs.sizes[axis] < rhs.sizes[axis];
      });
      size_t written = 0;
      for (size_t read = 0; read < boxes.size(); ++read) {
        if (written && sameCrossSection(boxes[written - 1], boxes[read])) {
          auto &previous = boxes[written - 1];
          const auto &next = boxes[read];
          // Individual endpoints were checked above. A union spanning a
          // signed-size overflow remains representable by its original boxes.
          int64_t end = previous.offsets[axis] + previous.sizes[axis];
          int64_t nextEnd = next.offsets[axis] + next.sizes[axis];
          int64_t size = 0;
          if (next.offsets[axis] <= end &&
              !llvm::SubOverflow(std::max(end, nextEnd), previous.offsets[axis],
                                 size)) {
            previous.sizes[axis] = size;
            continue;
          }
        }
        if (written != read)
          boxes[written] = std::move(boxes[read]);
        ++written;
      }
      boxes.resize(written);
    }
  } while (boxes.size() < previousSize);
  llvm::sort(boxes, boxLess);
  return mlir::success();
}

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
  if (domain.getForm() == ExactIndexSetForm::BoxUnion) {
    auto boxes = llvm::to_vector(domain.getBoxes());
    if (mlir::failed(coalesceBoxes(boxes, domain.getRank())))
      return mlir::failure();
    return ExactIndexSet(domain.getPresburgerSet(), ExactIndexSetForm::BoxUnion,
                         boxes);
  }
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
    if (mlir::failed(coalesceBoxes(pieces.domains, domain.getRank())))
      return mlir::failure();
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
