//===- CostModel.h - Instruction program performance model -*- C++ -*-===//

#ifndef WAFER_ANALYSIS_INSTR_COSTMODEL_H
#define WAFER_ANALYSIS_INSTR_COSTMODEL_H

#include "Wafer/Analysis/Instr/ScheduleCostAnalysis.h"

#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>

namespace wafer::analysis {

enum class SearchCostProfileProvenance : uint8_t {
  BuiltInEstimate,
  CalibratedTarget,
};

/// Performance-only rates shared by one accepted-candidate comparison cohort.
/// They never participate in IR legality or memory admission.
struct SearchCostPolicy {
  /// Stable provenance is part of the comparison cohort. A candidate scored
  /// with one profile can never silently outrank a candidate scored with
  /// another profile.
  uint64_t profileIdentity = 3;
  SearchCostProfileProvenance profileProvenance =
      SearchCostProfileProvenance::BuiltInEstimate;
  uint64_t ddrNominalBytesPerSecond = 150'000'000'000ULL;
  uint64_t directionalNoCBytesPerSecond = 128'000'000'000ULL;
  uint64_t dteEndpointBytesPerSecondEstimate = 128'000'000'000ULL;
  // Sender lifecycle estimates: first call includes cold control/setup work;
  // subsequent calls use the warmed path. Both exclude payload serialization.
  uint64_t dteFirstMessagePicosecondsEstimate = 13'000'000ULL;
  uint64_t dteMessageStartupPicosecondsEstimate = 1'500'000ULL;
  uint64_t noCHopPicosecondsEstimate = 1'000ULL;
  uint64_t instructionFixedPicosecondsEstimate = 1'000ULL;
  // Coarse service prior only when detailed work is not available/calibrated.
  // This includes estimated work, unlike the small issue-only control term.
  uint64_t unmodeledInstructionPicosecondsEstimate = 13'000'000ULL;
  uint64_t dteWaitedEventPicosecondsEstimate = 1'000ULL;
  // Idle NCC control: one call/ordering cost plus incremental worker polling.
  // Pending engine work is separate from these point estimates.
  uint64_t nccJoinPicosecondsEstimate = 140'000ULL;
  uint64_t nccParticipantWaitPicosecondsEstimate = 45'000ULL;
  uint64_t f16Bf16NPULogicalOpsPerSecondPerTile = 8'000'000'000'000ULL;
  uint64_t f16Bf16VectorLogicalOpsPerSecondPerTile = 64'000'000'000ULL;
  uint64_t f32VectorLogicalOpsPerSecondPerTile = 32'000'000'000ULL;
  uint64_t spmExplicitMovementBytesPerSecondPerTileEstimate =
      256'000'000'000ULL;
};

class SearchCostCohort {
public:
  static mlir::FailureOr<SearchCostCohort>
  create(const SearchCostPolicy &policy, std::string *failureReason = nullptr);

  const SearchCostPolicy &getPolicy() const { return policy; }

  friend bool operator==(const SearchCostCohort &lhs,
                         const SearchCostCohort &rhs) {
    const auto &left = lhs.policy;
    const auto &right = rhs.policy;
    return left.profileIdentity == right.profileIdentity &&
           left.profileProvenance == right.profileProvenance &&
           left.ddrNominalBytesPerSecond == right.ddrNominalBytesPerSecond &&
           left.directionalNoCBytesPerSecond ==
               right.directionalNoCBytesPerSecond &&
           left.dteEndpointBytesPerSecondEstimate ==
               right.dteEndpointBytesPerSecondEstimate &&
           left.dteFirstMessagePicosecondsEstimate ==
               right.dteFirstMessagePicosecondsEstimate &&
           left.dteMessageStartupPicosecondsEstimate ==
               right.dteMessageStartupPicosecondsEstimate &&
           left.noCHopPicosecondsEstimate == right.noCHopPicosecondsEstimate &&
           left.instructionFixedPicosecondsEstimate ==
               right.instructionFixedPicosecondsEstimate &&
           left.unmodeledInstructionPicosecondsEstimate ==
               right.unmodeledInstructionPicosecondsEstimate &&
           left.dteWaitedEventPicosecondsEstimate ==
               right.dteWaitedEventPicosecondsEstimate &&
           left.nccJoinPicosecondsEstimate ==
               right.nccJoinPicosecondsEstimate &&
           left.nccParticipantWaitPicosecondsEstimate ==
               right.nccParticipantWaitPicosecondsEstimate &&
           left.f16Bf16NPULogicalOpsPerSecondPerTile ==
               right.f16Bf16NPULogicalOpsPerSecondPerTile &&
           left.f16Bf16VectorLogicalOpsPerSecondPerTile ==
               right.f16Bf16VectorLogicalOpsPerSecondPerTile &&
           left.f32VectorLogicalOpsPerSecondPerTile ==
               right.f32VectorLogicalOpsPerSecondPerTile &&
           left.spmExplicitMovementBytesPerSecondPerTileEstimate ==
               right.spmExplicitMovementBytesPerSecondPerTileEstimate;
  }

private:
  explicit SearchCostCohort(SearchCostPolicy policy) : policy(policy) {}

  SearchCostPolicy policy;
};

/// Resource service estimates retained for diagnostics. Candidate ranking uses
/// one estimated duration; these terms are not independent Pareto objectives.
struct SearchResourceDurations {
  uint64_t neF16Bf16Picoseconds = 0;
  uint64_t vectorF16Bf16Picoseconds = 0;
  uint64_t vectorF32Picoseconds = 0;
  uint64_t ddrPicoseconds = 0;
  uint64_t nocPicoseconds = 0;
  uint64_t dteEndpointPicoseconds = 0;
  uint64_t dteStartupPicoseconds = 0;
  uint64_t nocHopPicoseconds = 0;
  uint64_t spmMovementPicoseconds = 0;
  uint64_t instructionControlPicoseconds = 0;
  uint64_t dteWaitControlPicoseconds = 0;
  uint64_t nccWaitControlPicoseconds = 0;
  /// Actual storage facts used only to break equal-duration ties. They never
  /// stand in for capacity admission.
  uint64_t spmHighWaterBytes = 0;
  uint64_t ddrHighWaterBytes = 0;
  uint64_t spmBufferCount = 0;
  uint64_t ddrBufferCount = 0;

  friend bool operator==(const SearchResourceDurations &lhs,
                         const SearchResourceDurations &rhs) {
    return std::tie(
               lhs.neF16Bf16Picoseconds, lhs.vectorF16Bf16Picoseconds,
               lhs.vectorF32Picoseconds, lhs.ddrPicoseconds, lhs.nocPicoseconds,
               lhs.dteEndpointPicoseconds, lhs.dteStartupPicoseconds,
               lhs.nocHopPicoseconds, lhs.spmMovementPicoseconds,
               lhs.instructionControlPicoseconds, lhs.dteWaitControlPicoseconds,
               lhs.nccWaitControlPicoseconds, lhs.spmHighWaterBytes,
               lhs.ddrHighWaterBytes, lhs.spmBufferCount, lhs.ddrBufferCount) ==
           std::tie(
               rhs.neF16Bf16Picoseconds, rhs.vectorF16Bf16Picoseconds,
               rhs.vectorF32Picoseconds, rhs.ddrPicoseconds, rhs.nocPicoseconds,
               rhs.dteEndpointPicoseconds, rhs.dteStartupPicoseconds,
               rhs.nocHopPicoseconds, rhs.spmMovementPicoseconds,
               rhs.instructionControlPicoseconds, rhs.dteWaitControlPicoseconds,
               rhs.nccWaitControlPicoseconds, rhs.spmHighWaterBytes,
               rhs.ddrHighWaterBytes, rhs.spmBufferCount, rhs.ddrBufferCount);
  }
};

struct KnownSearchObjective {
  SearchResourceDurations durations;
  SearchCostCohort cohort;
  uint64_t estimatedDurationPicoseconds = 0;
  /// Detailed work was unavailable, uncalibrated, or saturated. This changes
  /// diagnostic confidence, not whether a legal candidate can be ranked.
  bool usesCoarseEstimate = false;
};

enum class SearchObjectiveUnknownReason : uint8_t {
  NoCohort,
  MetricUnavailable,
  UncalibratedWork,
  ArithmeticOverflow,
};

struct UnknownSearchObjective {
  SearchObjectiveUnknownReason reason = SearchObjectiveUnknownReason::NoCohort;
};

using SearchObjective =
    std::variant<KnownSearchObjective, UnknownSearchObjective>;

SearchObjective
deriveSearchObjective(const InstructionProgramAggregateCost &cost,
                      const std::optional<SearchCostCohort> &cohort);

enum class SearchObjectiveComparison : uint8_t {
  Better,
  Worse,
  Equivalent,
  Incomparable,
};

SearchObjectiveComparison compareSearchObjectives(const SearchObjective &lhs,
                                                  const SearchObjective &rhs);

} // namespace wafer::analysis

#endif // WAFER_ANALYSIS_INSTR_COSTMODEL_H
