//===- CoordinatedVariantSelection.h - Final all-rank choice -*- C++ -*-===//

#ifndef WAFER_COMPILER_COORDINATEDVARIANTSELECTION_H
#define WAFER_COMPILER_COORDINATEDVARIANTSELECTION_H

#include "CoordinatedTerminalEvaluation.h"

#include "Wafer/Analysis/NoCProfitabilityAnalysis.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace wafer::compiler::detail {

/// Invocation-local view of one C3 fully gated variant. The view owns no IR
/// and is never serialized; it exists only so C4 can be unit-tested without
/// constructing target artifacts.
struct CoordinatedVariantCostView {
  int64_t stableSemanticOrdinal = 0;
  bool reservedBaseline = false;
  const analysis::WholeCardInstructionProgramCost *resourceCost = nullptr;
  uint32_t terminalActionOrdinal = 0;
};

enum class CoordinatedHardwareSelectionEvidence : uint8_t {
  ReservedBaseline,
  EstimatedBenefit,
  ProvenBenefit,
};

/// Pure C4 decision. Indices refer to the caller's fully gated input and do
/// not become an artifact or a second candidate owner.
struct CoordinatedVariantSelectionPlan {
  size_t baselineIndex = 0;
  size_t selectedIndex = 0;
  llvm::SmallVector<size_t, 16> paretoIndices;
  CoordinatedHardwareSelectionEvidence evidence =
      CoordinatedHardwareSelectionEvidence::ReservedBaseline;
  analysis::WholeCardResourceDurationEstimate selectedDuration;
};

/// Retains the exact-cost Pareto frontier, then selects at most one candidate
/// using current target duration parameters and the production promotion
/// margin. This function is pure: it performs no lowering, placement, repair,
/// ABI validation, or IR mutation.
mlir::FailureOr<CoordinatedVariantSelectionPlan>
planCoordinatedVariantSelection(
    llvm::ArrayRef<CoordinatedVariantCostView> variants,
    WholeVariantSelectionMode selectionMode);

/// Applies the pure plan and moves exactly one already accepted C3 variant to
/// the caller. Every rejected variant is destroyed with its own complete rank
/// domain; no rank-local result can be committed independently.
mlir::FailureOr<FullyGatedCoordinatedVariant>
selectFullyGatedCoordinatedVariant(
    std::vector<FullyGatedCoordinatedVariant> variants,
    WholeVariantSelectionMode selectionMode, llvm::raw_ostream &diagnostics,
    WholeVariantSelectionStatistics *statistics = nullptr);

llvm::StringRef stringifyCoordinatedHardwareSelectionEvidence(
    CoordinatedHardwareSelectionEvidence evidence);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_COORDINATEDVARIANTSELECTION_H
