//===- ActualResultController.h - Typed accepted-result control -*- C++ -*-===//

#ifndef WAFER_DRIVER_PHYSICALDATAFLOW_ACTUALRESULTCONTROLLER_H
#define WAFER_DRIVER_PHYSICALDATAFLOW_ACTUALRESULTCONTROLLER_H

#include "Wafer/Analysis/Instr/CostModel.h"
#include "Wafer/Driver/ExecutableCompilation.h"
#include "Wafer/Planning/PhysicalDataflow/PlanningState.h"

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

/// Pass the still-owned instruction modules to the sole cost implementation.
analysis::SearchObjective deriveExecutableSearchObjective(
    const ExecutableLoweringResult &executable,
    const std::optional<analysis::SearchCostCohort> &cohort);

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

enum class CandidateDomainState : uint8_t { Open, Closed };

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
  struct EvaluatedObjective {
    std::optional<analysis::SearchCostCohort> cohort;
    analysis::SearchObjective value;
  };
  ActualCandidateStatus status = ActualCandidateStatus::CompilerBug;
  std::optional<ExecutableCompilationResult> compilation;
  std::vector<SemanticRootKey> causalRoots;
  std::string detail;
  // Valid only for the still-owned, unchanged accepted Instr. Replacing or
  // mutating that IR invalidates this scalar ranking result.
  std::optional<EvaluatedObjective> objective;

  bool isAccepted() const {
    return status == ActualCandidateStatus::Accepted && compilation &&
           compilation->isAccepted() && compilation->executable.has_value();
  }
  bool isExactRejection() const {
    return status == ActualCandidateStatus::ExactRejection;
  }
};

const analysis::SearchObjective &getActualCandidateObjective(
    ActualCandidateResult &result,
    const std::optional<analysis::SearchCostCohort> &cohort);

struct RetainedSearchCandidate {
  StructuralCandidateKey key;
  analysis::SearchObjective objective;
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
  std::optional<analysis::SearchCostCohort> cohort;
  ExactRejectionCachePolicy exactRejectionCache =
      ExactRejectionCachePolicy::Enabled;
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
  CandidateRecordOutcome
  record(const StructuralCandidateKey &key, ActualCandidateResult result,
         CandidateDomainState domain = CandidateDomainState::Closed);
  /// Closes a session separately from recording an actual leaf.
  void close(const StructuralCandidateKey &key, SearchFrontierStatus domain);

  bool isForbidden(const StructuralCandidateKey &key) const;
  const ExactCompleteRejection *
  findExactCompleteRejection(const StructuralCandidateKey &key) const;
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
  std::optional<analysis::SearchCostCohort> cohort;
  ExactRejectionCachePolicy exactRejectionCache;
  std::set<StructuralCandidateKey> reserved;
  std::set<StructuralCandidateKey> completed;
  std::set<ExactCompleteRejection> forbidden;
  std::optional<RetainedSearchCandidate> incumbent;
  SearchControllerStatistics statistics;
  bool sawUnknownOrIncomparable = false;
  bool sawIncompleteDomain = false;
  bool poisoned = false;
  bool finished = false;
};

} // namespace wafer::compiler::detail

#endif // WAFER_DRIVER_PHYSICALDATAFLOW_ACTUALRESULTCONTROLLER_H
