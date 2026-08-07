//===- CoordinatedVariantSelection.h - Final all-rank choice -*- C++ -*-===//

#ifndef WAFER_COMPILER_COORDINATEDVARIANTSELECTION_H
#define WAFER_COMPILER_COORDINATEDVARIANTSELECTION_H

#include "CoordinatedExecutableFinalization.h"

#include "Wafer/Analysis/NoCProfitabilityAnalysis.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace wafer::compiler::detail {

/// Invocation-local view of one C3 admitted executable variant. The view owns
/// no IR and is never serialized; it exists only so C4 can be unit-tested
/// without constructing target artifacts.
struct CoordinatedVariantCostView {
  int64_t stableSemanticOrdinal = 0;
  bool reservedBaseline = false;
  const analysis::WholeCardInstructionProgramCost *resourceCost = nullptr;
  uint32_t scheduleActionOrdinal = 0;
};

enum class CoordinatedHardwareSelectionEvidence : uint8_t {
  ReservedBaseline,
  EstimatedBenefit,
  ProvenBenefit,
};

/// Pure C4 decision. Indices refer to the caller's admitted executable input
/// and do not become an artifact or a second candidate owner.
struct CoordinatedVariantSelectionPlan {
  size_t baselineIndex = 0;
  size_t selectedIndex = 0;
  llvm::SmallVector<size_t, 16> paretoIndices;
  CoordinatedHardwareSelectionEvidence evidence =
      CoordinatedHardwareSelectionEvidence::ReservedBaseline;
  analysis::WholeCardResourceDurationEstimate selectedDuration;
};

/// Pure compiler-private qualification decision over already exact-admitted
/// final Instr executables. Matching is derived from typed current IR and
/// action identity; costs and production promotion are intentionally absent.
struct CoordinatedQualificationSelectionPlan {
  size_t selectedIndex = 0;
  llvm::SmallVector<size_t, 16> matchingIndices;
};

/// Retains the exact-cost Pareto frontier, then selects at most one candidate
/// using current target duration parameters and the production promotion
/// margin. This function is pure: it performs no lowering, placement, repair,
/// ABI validation, or IR mutation.
mlir::FailureOr<CoordinatedVariantSelectionPlan>
planCoordinatedVariantSelection(
    llvm::ArrayRef<CoordinatedVariantCostView> variants,
    WholeVariantSelectionMode selectionMode);

/// Filters a complete admitted frontier for one qualification or collective
/// characterization mode, then selects the lowest stable semantic/action
/// ordinal. Input order is never a tie-break. The unique admitted baseline is
/// validated as an invocation invariant but is eligible only when the specific
/// mode's typed predicate allows it.
mlir::FailureOr<CoordinatedQualificationSelectionPlan>
planCoordinatedQualificationSelection(
    llvm::ArrayRef<AdmittedCoordinatedExecutable> variants,
    WholeVariantSelectionMode selectionMode);

/// Drops exact-cost dominated or deterministically equivalent admitted
/// executable variants in place. In production mode it also drops candidates
/// that the complete current baseline proves can never satisfy the final
/// promotion contract (for example, an uncalibrated regression). A reserved
/// baseline, when present, is always retained; unlike final selection this
/// reducer also accepts a frontier that has not reached the baseline yet. It
/// performs no hardware promotion or IR change. Qualification frontiers are
/// only identity-validated here and remain unpruned until their dedicated
/// admitted-final-IR selector runs.
mlir::LogicalResult reduceAdmittedExecutableFrontier(
    std::vector<AdmittedCoordinatedExecutable> &variants,
    WholeVariantSelectionMode selectionMode,
    const analysis::WholeCardInstructionProgramCost *externalBaselineCost =
        nullptr);

/// Applies the pure plan and moves exactly one already accepted C3 variant to
/// the caller. Every rejected variant is destroyed with its own complete rank
/// domain; no rank-local result can be committed independently.
mlir::FailureOr<AdmittedCoordinatedExecutable>
selectAdmittedCoordinatedExecutable(
    std::vector<AdmittedCoordinatedExecutable> variants,
    WholeVariantSelectionMode selectionMode, llvm::raw_ostream &diagnostics,
    WholeVariantSelectionStatistics *statistics = nullptr);

llvm::StringRef stringifyCoordinatedHardwareSelectionEvidence(
    CoordinatedHardwareSelectionEvidence evidence);

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_COORDINATEDVARIANTSELECTION_H
