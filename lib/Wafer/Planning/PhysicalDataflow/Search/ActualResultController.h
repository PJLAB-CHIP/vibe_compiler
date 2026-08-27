//===- ActualResultController.h - Typed accepted-result control -*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_ACTUALRESULTCONTROLLER_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_ACTUALRESULTCONTROLLER_H

#include "Wafer/Analysis/ScheduleCost/TheoreticalScheduleCostAnalysis.h"
#include "Wafer/Planning/PhysicalDataflow/FullFeasibility.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace wafer::compiler::detail {

/// Stable identity of one structural search transaction. Facts created after
/// TileRegion materialization (layout, buffers, movement, events, schedule and
/// offsets) cannot enter this key.
class StructuralCandidateKey {
public:
  static StructuralCandidateKey create(const TemporalState &state) {
    return StructuralCandidateKey(state.getSpatialPlan(), state.getRegionPlan(),
                                  state.getTemporalPlan());
  }
  static StructuralCandidateKey create(SpatialPlan spatial,
                                       RegionPlan regions,
                                       TemporalPlan temporal) {
    return StructuralCandidateKey(std::move(spatial), std::move(regions),
                                  std::move(temporal));
  }

  const SpatialPlan &getSpatialPlan() const { return spatial; }
  const RegionPlan &getRegionPlan() const { return regions; }
  const TemporalPlan &getTemporalPlan() const { return temporal; }

  friend bool operator==(const StructuralCandidateKey &lhs,
                         const StructuralCandidateKey &rhs) {
    return std::tie(lhs.spatial, lhs.regions, lhs.temporal) ==
           std::tie(rhs.spatial, rhs.regions, rhs.temporal);
  }
  friend bool operator<(const StructuralCandidateKey &lhs,
                        const StructuralCandidateKey &rhs) {
    return std::tie(lhs.spatial, lhs.regions, lhs.temporal) <
           std::tie(rhs.spatial, rhs.regions, rhs.temporal);
  }

private:
  StructuralCandidateKey(SpatialPlan spatial, RegionPlan regions,
                         TemporalPlan temporal)
      : spatial(std::move(spatial)), regions(std::move(regions)),
        temporal(std::move(temporal)) {}

  SpatialPlan spatial;
  RegionPlan regions;
  TemporalPlan temporal;
};

class SearchCostCohort {
public:
  static mlir::FailureOr<SearchCostCohort>
  create(const analysis::ScheduleEstimatePolicy &policy,
         std::string *failureReason = nullptr);

  const analysis::ScheduleEstimatePolicy &getPolicy() const { return policy; }

  friend bool operator==(const SearchCostCohort &lhs,
                         const SearchCostCohort &rhs) {
    const auto &left = lhs.policy;
    const auto &right = rhs.policy;
    return left.cardDDRNominalBytesPerSecond ==
               right.cardDDRNominalBytesPerSecond &&
           left.directionalNoCBytesPerSecond ==
               right.directionalNoCBytesPerSecond &&
           left.dteEndpointBytesPerSecondEstimate ==
               right.dteEndpointBytesPerSecondEstimate &&
           left.dteMessageStartupPicosecondsEstimate ==
               right.dteMessageStartupPicosecondsEstimate &&
           left.noCHopPicosecondsEstimate ==
               right.noCHopPicosecondsEstimate &&
           left.instructionFixedPicosecondsEstimate ==
               right.instructionFixedPicosecondsEstimate &&
           left.dteWaitedEventPicosecondsEstimate ==
               right.dteWaitedEventPicosecondsEstimate &&
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
  explicit SearchCostCohort(analysis::ScheduleEstimatePolicy policy)
      : policy(policy) {}

  analysis::ScheduleEstimatePolicy policy;
};

/// Independently comparable service dimensions derived from the final actual
/// Instr program. NE and Vector/CT are deliberately not collapsed into one
/// compute number: without a proved inter-engine schedule, a trade-off between
/// them has no total order.
struct SearchResourceDurations {
  uint64_t neF16Bf16Picoseconds = 0;
  uint64_t vectorF16Bf16Picoseconds = 0;
  uint64_t vectorF32Picoseconds = 0;
  uint64_t ddrPicoseconds = 0;
  uint64_t nocPicoseconds = 0;
  uint64_t spmMovementPicoseconds = 0;
  uint64_t instructionControlPicoseconds = 0;
  uint64_t dteWaitControlPicoseconds = 0;
  uint64_t nccWaitControlPicoseconds = 0;

  friend bool operator==(const SearchResourceDurations &lhs,
                         const SearchResourceDurations &rhs) {
    return std::tie(lhs.neF16Bf16Picoseconds,
                    lhs.vectorF16Bf16Picoseconds,
                    lhs.vectorF32Picoseconds, lhs.ddrPicoseconds,
                    lhs.nocPicoseconds, lhs.spmMovementPicoseconds,
                    lhs.instructionControlPicoseconds,
                    lhs.dteWaitControlPicoseconds,
                    lhs.nccWaitControlPicoseconds) ==
           std::tie(rhs.neF16Bf16Picoseconds,
                    rhs.vectorF16Bf16Picoseconds,
                    rhs.vectorF32Picoseconds, rhs.ddrPicoseconds,
                    rhs.nocPicoseconds, rhs.spmMovementPicoseconds,
                    rhs.instructionControlPicoseconds,
                    rhs.dteWaitControlPicoseconds,
                    rhs.nccWaitControlPicoseconds);
  }
};

struct KnownSearchObjective {
  SearchResourceDurations durations;
  SearchCostCohort cohort;
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
deriveSearchObjective(const analysis::CardInstructionProgramCost &cost,
                      const std::optional<SearchCostCohort> &cohort);

enum class SearchObjectiveComparison : uint8_t {
  Better,
  Worse,
  Equivalent,
  Incomparable,
};

SearchObjectiveComparison compareSearchObjectives(const SearchObjective &lhs,
                                                  const SearchObjective &rhs);

enum class CandidateReservation : uint8_t {
  Granted,
  Exhausted,
  Duplicate,
  Closed,
};
enum class CandidateRecordOutcome : uint8_t {
  Accepted,
  ExactRejection,
  Unsupported,
  Indeterminate,
  CompilerBug,
};

enum class ExactCompleteRejectionKind : uint8_t {
  SPMCapacity,
  ExecutableGate,
};

struct ExactCompleteRejection {
  StructuralCandidateKey key;
  ExactCompleteRejectionKind kind = ExactCompleteRejectionKind::ExecutableGate;
  std::vector<SemanticRootKey> causalRoots;

  friend bool operator<(const ExactCompleteRejection &lhs,
                        const ExactCompleteRejection &rhs) {
    return lhs.key < rhs.key;
  }
};

struct RetainedSearchCandidate {
  StructuralCandidateKey key;
  SearchObjective objective;
  CardExecutableCompilationResult compilation;

  CardExecutableLoweringResult takeExecutable() {
    return compilation.takeExecutable();
  }
};

enum class SearchControllerCoverage : uint8_t {
  ComparableBest,
  FeasibleUnranked,
  FeasiblePartial,
  NoFeasible,
  IncompleteNoCandidate,
  Failed,
};

enum class SearchFrontierStatus : uint8_t {
  Exhausted,
  Incomplete,
};

enum class ExactRejectionCachePolicy : uint8_t {
  Enabled,
  Disabled,
};

struct ActualResultControllerOptions {
  uint64_t actualizationCredits = 0;
  std::optional<SearchCostCohort> cohort;
  ExactRejectionCachePolicy exactRejectionCache =
      ExactRejectionCachePolicy::Enabled;
};

/// A caller assertion that `objective` is an admissible lower bound for the
/// named complete assignment. Q51.Core only compares the typed value; the
/// producer and its proof are owned by the later search-policy stage.
struct SearchLowerBound {
  StructuralCandidateKey key;
  SearchObjective objective;
};

llvm::StringRef
stringifySearchControllerCoverage(SearchControllerCoverage coverage);

struct SearchControllerStatistics {
  uint64_t reserved = 0;
  uint64_t duplicateReservations = 0;
  uint64_t exhaustedReservations = 0;
  uint64_t closedReservations = 0;
  uint64_t accepted = 0;
  uint64_t exactRejected = 0;
  uint64_t unsupported = 0;
  uint64_t indeterminate = 0;
  uint64_t compilerBugs = 0;
};

struct SearchControllerResult {
  SearchControllerCoverage coverage = SearchControllerCoverage::Failed;
  std::optional<RetainedSearchCandidate> winner;
  SearchControllerStatistics statistics;
  size_t exactCompleteRejections = 0;
};

class ActualResultController {
public:
  explicit ActualResultController(ActualResultControllerOptions options)
      : remainingCredits(options.actualizationCredits),
        cohort(std::move(options.cohort)),
        exactRejectionCache(options.exactRejectionCache) {}

  CandidateReservation reserve(const StructuralCandidateKey &key);
  CandidateRecordOutcome record(const StructuralCandidateKey &key,
                                FullFeasibilityResult result);

  bool isForbidden(const StructuralCandidateKey &key) const;
  const ExactCompleteRejection *
  findExactCompleteRejection(const StructuralCandidateKey &key) const;
  bool canPrune(const SearchLowerBound &lowerBound) const;
  void markCompilerBug();
  uint64_t getRemainingCredits() const { return remainingCredits; }
  const SearchControllerStatistics &getStatistics() const { return statistics; }
  size_t getExactCompleteRejectionCount() const { return forbidden.size(); }

  SearchControllerResult finish(SearchFrontierStatus frontier);

private:
  CandidateRecordOutcome failCompilerBug();

  uint64_t remainingCredits;
  std::optional<SearchCostCohort> cohort;
  ExactRejectionCachePolicy exactRejectionCache;
  std::set<StructuralCandidateKey> reserved;
  std::set<StructuralCandidateKey> completed;
  std::set<ExactCompleteRejection> forbidden;
  std::optional<RetainedSearchCandidate> incumbent;
  SearchControllerStatistics statistics;
  bool sawUnknownOrIncomparable = false;
  bool poisoned = false;
  bool finished = false;
};

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_ACTUALRESULTCONTROLLER_H
