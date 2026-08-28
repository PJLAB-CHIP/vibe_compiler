//===- SemanticRoot.h - Query-local semantic root identity ----*- C++ -*-===//

#ifndef WAFER_ANALYSIS_LINALG_SEMANTICROOT_H
#define WAFER_ANALYSIS_LINALG_SEMANTICROOT_H

#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>

namespace wafer::compiler::detail {

enum class SemanticRootAnchorKind : uint8_t {
  FunctionResult,
  EffectBoundary,
};

enum class SemanticRootPathRelation : uint8_t {
  SSAUseDef,
  RegionBranch,
  EffectDependency,
};

struct SemanticRootPathStep {
  SemanticRootPathRelation relation = SemanticRootPathRelation::SSAUseDef;
  uint32_t producerResult = 0;
  uint32_t consumerOperand = 0;

  friend bool operator==(const SemanticRootPathStep &lhs,
                         const SemanticRootPathStep &rhs) {
    return lhs.relation == rhs.relation &&
           lhs.producerResult == rhs.producerResult &&
           lhs.consumerOperand == rhs.consumerOperand;
  }
  friend bool operator<(const SemanticRootPathStep &lhs,
                        const SemanticRootPathStep &rhs) {
    if (lhs.relation != rhs.relation)
      return lhs.relation < rhs.relation;
    if (lhs.producerResult != rhs.producerResult)
      return lhs.producerResult < rhs.producerResult;
    return lhs.consumerOperand < rhs.consumerOperand;
  }
};

/// Canonical root identity valid for one immutable TensorProgram borrow. The
/// path is ordered from the observable boundary toward the root and contains
/// no operation address, symbol spelling, location, or walk ordinal.
struct SemanticRootKey {
  SemanticRootAnchorKind anchorKind = SemanticRootAnchorKind::FunctionResult;
  uint32_t anchorIndex = 0;
  llvm::SmallVector<SemanticRootPathStep, 4> path;

  friend bool operator==(const SemanticRootKey &lhs,
                         const SemanticRootKey &rhs) {
    return lhs.anchorKind == rhs.anchorKind &&
           lhs.anchorIndex == rhs.anchorIndex && lhs.path == rhs.path;
  }
  friend bool operator!=(const SemanticRootKey &lhs,
                         const SemanticRootKey &rhs) {
    return !(lhs == rhs);
  }
  friend bool operator<(const SemanticRootKey &lhs,
                        const SemanticRootKey &rhs) {
    if (lhs.anchorKind != rhs.anchorKind)
      return lhs.anchorKind < rhs.anchorKind;
    if (lhs.anchorIndex != rhs.anchorIndex)
      return lhs.anchorIndex < rhs.anchorIndex;
    return std::lexicographical_compare(lhs.path.begin(), lhs.path.end(),
                                        rhs.path.begin(), rhs.path.end());
  }
};

} // namespace wafer::compiler::detail

#endif // WAFER_ANALYSIS_LINALG_SEMANTICROOT_H
