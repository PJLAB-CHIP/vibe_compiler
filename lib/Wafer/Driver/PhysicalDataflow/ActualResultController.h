//===- ActualResultController.h - Typed accepted-result control -*- C++ -*-===//

#ifndef WAFER_DRIVER_PHYSICALDATAFLOW_ACTUALRESULTCONTROLLER_H
#define WAFER_DRIVER_PHYSICALDATAFLOW_ACTUALRESULTCONTROLLER_H

#include "Wafer/Driver/ExecutableCompilation.h"
#include "Wafer/Planning/PhysicalDataflow/PlanningState.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace wafer::compiler::detail {

/// Performance-only rates shared by one accepted-candidate comparison cohort.
/// They never participate in IR legality or memory admission.
struct SearchCostPolicy {
  uint64_t ddrNominalBytesPerSecond = 150'000'000'000ULL;
  uint64_t directionalNoCBytesPerSecond = 128'000'000'000ULL;
  uint64_t dteEndpointBytesPerSecondEstimate = 128'000'000'000ULL;
  uint64_t dteMessageStartupPicosecondsEstimate = 10'000'000ULL;
  uint64_t noCHopPicosecondsEstimate = 1'000ULL;
  uint64_t instructionFixedPicosecondsEstimate = 1'000ULL;
  uint64_t dteWaitedEventPicosecondsEstimate = 1'000ULL;
  uint64_t nccParticipantWaitPicosecondsEstimate = 1'000ULL;
  uint64_t f16Bf16NPULogicalOpsPerSecondPerTile = 8'000'000'000'000ULL;
  uint64_t f16Bf16VectorLogicalOpsPerSecondPerTile = 64'000'000'000ULL;
  uint64_t f32VectorLogicalOpsPerSecondPerTile = 32'000'000'000ULL;
  uint64_t spmExplicitMovementBytesPerSecondPerTileEstimate =
      256'000'000'000ULL;
};

/// Stable identity of one structural search transaction. Facts created after
/// TileRegion materialization (layout, buffers, movement, events, schedule and
/// offsets) cannot enter this key.
class StructuralCandidateKey {
public:
  static StructuralCandidateKey create(const RegionState &state) {
    return StructuralCandidateKey(state.getSpatialPlan(),
                                  state.getRegionPlan());
  }
  static StructuralCandidateKey create(SpatialPlan spatial,
                                       RegionPlan regions) {
    return StructuralCandidateKey(std::move(spatial), std::move(regions));
  }

  const SpatialPlan &getSpatialPlan() const { return spatial; }
  const RegionPlan &getRegionPlan() const { return regions; }
  friend bool operator==(const StructuralCandidateKey &lhs,
                         const StructuralCandidateKey &rhs) {
    return std::tie(lhs.spatial, lhs.regions) ==
           std::tie(rhs.spatial, rhs.regions);
  }
  friend bool operator<(const StructuralCandidateKey &lhs,
                        const StructuralCandidateKey &rhs) {
    return std::tie(lhs.spatial, lhs.regions) <
           std::tie(rhs.spatial, rhs.regions);
  }

private:
  StructuralCandidateKey(SpatialPlan spatial, RegionPlan regions)
      : spatial(std::move(spatial)), regions(std::move(regions)) {}

  SpatialPlan spatial;
  RegionPlan regions;
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
    return left.ddrNominalBytesPerSecond == right.ddrNominalBytesPerSecond &&
           left.directionalNoCBytesPerSecond ==
               right.directionalNoCBytesPerSecond &&
           left.dteEndpointBytesPerSecondEstimate ==
               right.dteEndpointBytesPerSecondEstimate &&
           left.dteMessageStartupPicosecondsEstimate ==
               right.dteMessageStartupPicosecondsEstimate &&
           left.noCHopPicosecondsEstimate == right.noCHopPicosecondsEstimate &&
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
  explicit SearchCostCohort(SearchCostPolicy policy) : policy(policy) {}

  SearchCostPolicy policy;
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
    return std::tie(
               lhs.neF16Bf16Picoseconds, lhs.vectorF16Bf16Picoseconds,
               lhs.vectorF32Picoseconds, lhs.ddrPicoseconds, lhs.nocPicoseconds,
               lhs.spmMovementPicoseconds, lhs.instructionControlPicoseconds,
               lhs.dteWaitControlPicoseconds, lhs.nccWaitControlPicoseconds) ==
           std::tie(
               rhs.neF16Bf16Picoseconds, rhs.vectorF16Bf16Picoseconds,
               rhs.vectorF32Picoseconds, rhs.ddrPicoseconds, rhs.nocPicoseconds,
               rhs.spmMovementPicoseconds, rhs.instructionControlPicoseconds,
               rhs.dteWaitControlPicoseconds, rhs.nccWaitControlPicoseconds);
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
deriveSearchObjective(const analysis::InstructionProgramAggregateCost &cost,
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

enum class ActualCandidateStatus : uint8_t {
  Accepted,
  ExactRejection,
  Unsupported,
  Indeterminate,
  CompilerBug,
};

/// Result returned by the caller-owned current-IR actualizer. Accepted keeps
/// the original actual Tile/Instr owner; rejected results contain only typed
/// witness facts. No structural or downstream shadow plan enters this type.
struct ActualCandidateResult {
  ActualCandidateStatus status = ActualCandidateStatus::CompilerBug;
  std::optional<ExecutableCompilationResult> compilation;
  std::vector<SemanticRootKey> causalRoots;
  std::string detail;

  bool isAccepted() const {
    return status == ActualCandidateStatus::Accepted && compilation &&
           compilation->isAccepted() && compilation->executable.has_value();
  }
  bool isExactRejection() const {
    return status == ActualCandidateStatus::ExactRejection;
  }
};

struct RetainedSearchCandidate {
  StructuralCandidateKey key;
  SearchObjective objective;
  ExecutableCompilationResult compilation;

  ExecutableLoweringResult takeExecutable() {
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
/// named complete assignment. The controller only compares the typed value;
/// the producer and its proof are owned by the search policy.
struct SearchLowerBound {
  StructuralCandidateKey key;
  SearchObjective objective;
};

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
                                ActualCandidateResult result);

  bool isForbidden(const StructuralCandidateKey &key) const;
  const ExactCompleteRejection *
  findExactCompleteRejection(const StructuralCandidateKey &key) const;
  bool canPrune(const SearchLowerBound &lowerBound) const;
  void markCompilerBug();
  uint64_t getRemainingCredits() const { return remainingCredits; }
  const StructuralCandidateKey *getIncumbentKey() const {
    return incumbent ? &incumbent->key : nullptr;
  }
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
  std::optional<SearchObjective> referenceObjective;
  SearchControllerStatistics statistics;
  bool sawUnknownOrIncomparable = false;
  bool poisoned = false;
  bool finished = false;
};

} // namespace wafer::compiler::detail

#endif // WAFER_DRIVER_PHYSICALDATAFLOW_ACTUALRESULTCONTROLLER_H
