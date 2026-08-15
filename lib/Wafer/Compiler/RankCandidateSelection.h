//===- RankCandidateSelection.h - Rank candidate selection -*- C++ -*-===//

#ifndef WAFER_COMPILER_RANKCANDIDATESELECTION_H
#define WAFER_COMPILER_RANKCANDIDATESELECTION_H

#include "RankCandidateEvaluation.h"

#include "Wafer/Analysis/NoCProfitabilityAnalysis.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace wafer::compiler::detail {

/// Invocation-local cost view of one evaluated card candidate. It holds
/// no IR and is never serialized.
struct RankCandidateCostView {
  int64_t stableSemanticOrdinal = 0;
  bool reservedBaseline = false;
  const analysis::CardInstructionProgramCost *resourceCost = nullptr;
  uint32_t scheduleActionOrdinal = 0;
};

enum class RankSelectionBasis : uint8_t {
  ReservedBaseline,
  EstimatedBenefit,
  ProvenBenefit,
};

/// Pure selection result. Indices refer to the caller's evaluated candidates
/// and do not create another candidate representation.
struct RankCandidateSelection {
  size_t baselineIndex = 0;
  size_t selectedIndex = 0;
  llvm::SmallVector<size_t, 16> paretoIndices;
  RankSelectionBasis basis = RankSelectionBasis::ReservedBaseline;
  analysis::ProgramDurationEstimate selectedDuration;
};

/// Pure compiler-private qualification decision over already validated
/// final Instr executables. Matching is derived from typed current IR and
/// the scheduling-action key; costs and production promotion are absent.
struct RankQualificationSelection {
  size_t selectedIndex = 0;
  llvm::SmallVector<size_t, 16> matchingIndices;
};

/// Retains the exact-cost Pareto candidates, then selects at most one candidate
/// using current target duration parameters and the production promotion
/// margin. This function is pure: it performs no lowering, placement, repair,
/// ABI validation, or IR mutation.
mlir::FailureOr<RankCandidateSelection> selectRankCandidateByCost(
    llvm::ArrayRef<RankCandidateCostView> variants,
    RankCandidateSelectionMode selectionMode);

/// Filters a complete evaluated candidates for one qualification or collective
/// characterization mode, then selects the lowest stable semantic/action
/// ordinal. Input order is never a tie-break. The unique admitted baseline is
/// validated as an invocation invariant but is eligible only when the specific
/// mode's typed predicate allows it.
mlir::FailureOr<RankQualificationSelection>
selectRankQualificationCandidate(
    llvm::ArrayRef<EvaluatedRankCandidate> variants,
    RankCandidateSelectionMode selectionMode);

/// Drops exact-cost dominated or deterministically equivalent evaluated
/// executable variants in place. In production mode it also drops candidates
/// that the complete current baseline proves can never satisfy the final
/// promotion contract (for example, an uncalibrated regression). A reserved
/// baseline, when present, is always retained; unlike final selection this
/// reducer also accepts a candidates that has not reached the baseline yet. It
/// performs no hardware promotion or IR change. Qualification candidatess
/// are only key-validated here and remain unpruned until their dedicated
/// final-IR selector runs.
mlir::LogicalResult reduceEvaluatedRankCandidates(
    std::vector<EvaluatedRankCandidate> &variants,
    RankCandidateSelectionMode selectionMode,
    const analysis::CardInstructionProgramCost *externalBaselineCost =
        nullptr);

/// Applies the pure result and moves exactly one selected candidate to
/// the caller. Every rejected variant is destroyed with its own complete rank
/// domain; no rank-local result can be returned independently.
mlir::FailureOr<EvaluatedRankCandidate> selectEvaluatedRankCandidate(
    std::vector<EvaluatedRankCandidate> variants,
    RankCandidateSelectionMode selectionMode, llvm::raw_ostream &diagnostics,
    RankCandidateSearchStatistics *statistics = nullptr);

llvm::StringRef stringifyRankSelectionBasis(RankSelectionBasis basis);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_RANKCANDIDATESELECTION_H
