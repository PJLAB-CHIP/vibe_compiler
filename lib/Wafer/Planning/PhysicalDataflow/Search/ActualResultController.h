//===- ActualResultController.h - Typed accepted-result control -*- C++ -*-===//

#ifndef WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_ACTUALRESULTCONTROLLER_H
#define WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_ACTUALRESULTCONTROLLER_H

#include "Wafer/Planning/PhysicalDataflow/FullFeasibility.h"
#include "Wafer/Planning/PhysicalDataflow/Search/CompleteCandidateKey.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace wafer::compiler::detail {

class SearchCostCohort {
public:
  static mlir::FailureOr<SearchCostCohort>
  create(uint64_t instructionTick, uint64_t ddrReadByteTick,
         uint64_t ddrWriteByteTick, uint64_t nocMinimumHopByteTick,
         std::string *failureReason = nullptr);

  uint64_t getInstructionTick() const { return instructionTick; }
  uint64_t getDDRReadByteTick() const { return ddrReadByteTick; }
  uint64_t getDDRWriteByteTick() const { return ddrWriteByteTick; }
  uint64_t getNoCMinimumHopByteTick() const { return nocMinimumHopByteTick; }

  friend bool operator==(const SearchCostCohort &lhs,
                         const SearchCostCohort &rhs) {
    return lhs.instructionTick == rhs.instructionTick &&
           lhs.ddrReadByteTick == rhs.ddrReadByteTick &&
           lhs.ddrWriteByteTick == rhs.ddrWriteByteTick &&
           lhs.nocMinimumHopByteTick == rhs.nocMinimumHopByteTick;
  }

private:
  SearchCostCohort(uint64_t instructionTick, uint64_t ddrReadByteTick,
                   uint64_t ddrWriteByteTick, uint64_t nocMinimumHopByteTick)
      : instructionTick(instructionTick), ddrReadByteTick(ddrReadByteTick),
        ddrWriteByteTick(ddrWriteByteTick),
        nocMinimumHopByteTick(nocMinimumHopByteTick) {}

  uint64_t instructionTick;
  uint64_t ddrReadByteTick;
  uint64_t ddrWriteByteTick;
  uint64_t nocMinimumHopByteTick;
};

struct KnownSearchObjective {
  uint64_t ticks = 0;
  SearchCostCohort cohort;
};

enum class SearchObjectiveUnknownReason : uint8_t {
  NoCohort,
  MetricUnavailable,
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
  CompleteCandidateKey key;
  ExactCompleteRejectionKind kind = ExactCompleteRejectionKind::ExecutableGate;
  std::vector<SemanticRootKey> causalRoots;

  friend bool operator<(const ExactCompleteRejection &lhs,
                        const ExactCompleteRejection &rhs) {
    return lhs.key < rhs.key;
  }
};

struct RetainedSearchCandidate {
  CompleteCandidateKey key;
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
  CompleteCandidateKey key;
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

  CandidateReservation reserve(const CompleteCandidateKey &key);
  CandidateRecordOutcome record(const CompleteCandidateKey &key,
                                FullFeasibilityResult result);

  bool isForbidden(const CompleteCandidateKey &key) const;
  const ExactCompleteRejection *
  findExactCompleteRejection(const CompleteCandidateKey &key) const;
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
  std::set<CompleteCandidateKey> reserved;
  std::set<CompleteCandidateKey> completed;
  std::set<ExactCompleteRejection> forbidden;
  std::optional<RetainedSearchCandidate> incumbent;
  SearchControllerStatistics statistics;
  bool sawUnknownOrIncomparable = false;
  bool poisoned = false;
  bool finished = false;
};

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_PLANNING_PHYSICALDATAFLOW_SEARCH_ACTUALRESULTCONTROLLER_H
