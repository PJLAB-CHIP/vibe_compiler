//===- SearchControl.h - Deterministic search control --------*- C++ -*-===//

#ifndef WAFER_COMPILER_SEARCH_SEARCHCONTROL_H
#define WAFER_COMPILER_SEARCH_SEARCHCONTROL_H

#include "mlir/Support/LogicalResult.h"

#include <cassert>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace wafer::compiler::detail {

/// A required, session-local limit. Unlimited execution must be requested
/// explicitly; there is no hidden default search effort.
struct SearchWorkBudget {
  static SearchWorkBudget unlimited() { return {std::nullopt, std::nullopt}; }

  static SearchWorkBudget bounded(uint64_t evaluations,
                                  uint64_t expansions) {
    return {evaluations, expansions};
  }

  std::optional<uint64_t> maximumEvaluations;
  std::optional<uint64_t> maximumExpansions;
};

enum class SearchEvaluationKind : uint8_t {
  Expandable,
  Accepted,
  ExactRejection,
  Deferred,
  Indeterminate,
  Fatal,
};

/// Typed result of evaluating one concrete assignment. Diagnostic strings do
/// not participate in control flow. Only Accepted owns an executable result.
template <typename Accepted> class SearchEvaluation {
public:
  static SearchEvaluation expandable() {
    return SearchEvaluation(SearchEvaluationKind::Expandable);
  }
  static SearchEvaluation accepted(Accepted result) {
    SearchEvaluation evaluation(SearchEvaluationKind::Accepted);
    evaluation.acceptedResult.emplace(std::move(result));
    return evaluation;
  }
  static SearchEvaluation exactRejection() {
    return SearchEvaluation(SearchEvaluationKind::ExactRejection);
  }
  static SearchEvaluation deferred() {
    return SearchEvaluation(SearchEvaluationKind::Deferred);
  }
  static SearchEvaluation indeterminate() {
    return SearchEvaluation(SearchEvaluationKind::Indeterminate);
  }
  static SearchEvaluation fatal() {
    return SearchEvaluation(SearchEvaluationKind::Fatal);
  }

  SearchEvaluationKind getKind() const { return kind; }

  Accepted takeAccepted() {
    assert(kind == SearchEvaluationKind::Accepted && acceptedResult);
    return std::move(*acceptedResult);
  }

private:
  explicit SearchEvaluation(SearchEvaluationKind kind) : kind(kind) {}

  SearchEvaluationKind kind;
  std::optional<Accepted> acceptedResult;
};

/// Small semantic work summary. It is returned to the caller for budget and
/// coverage evidence; the compiler does not print or persist it by default.
struct SearchWorkCounts {
  uint64_t generated = 0;
  uint64_t deduplicated = 0;
  uint64_t evaluated = 0;
  uint64_t expanded = 0;
  uint64_t accepted = 0;
  uint64_t exactRejected = 0;
  uint64_t deferred = 0;
  uint64_t indeterminate = 0;
};

template <typename Key, typename Accepted> struct SearchControlResult {
  SearchControlResult(Accepted incumbent, SearchWorkCounts work,
                      std::vector<Key> unresolvedKeys,
                      bool frontierExhausted)
      : incumbent(std::move(incumbent)), work(work),
        unresolvedKeys(std::move(unresolvedKeys)),
        frontierExhausted(frontierExhausted) {}

  Accepted incumbent;
  SearchWorkCounts work;
  std::vector<Key> unresolvedKeys;
  bool frontierExhausted;
};

/// Runs one deterministic, non-dropping search over a concrete typed State.
/// `keyOf` supplies its canonical semantic key; `expand` returns all immediate
/// children; `evaluate` classifies the current state; `isBetter` compares
/// complete accepted results in one frozen cost cohort. The baseline
/// incumbent is never inserted into the candidate frontier or work counts.
template <typename State, typename Key, typename Accepted, typename KeyOf,
          typename Expand, typename Evaluate, typename IsBetter>
mlir::FailureOr<SearchControlResult<Key, Accepted>> runSearchControl(
    std::vector<State> initialStates, Accepted baseline,
    SearchWorkBudget budget, KeyOf keyOf, Expand expand, Evaluate evaluate,
    IsBetter isBetter) {
  std::map<Key, State> frontier;
  std::set<Key> seen;
  SearchWorkCounts work;

  auto insert = [&](State state) {
    Key key = keyOf(state);
    if (!seen.insert(key).second) {
      ++work.deduplicated;
      return;
    }
    ++work.generated;
    frontier.emplace(std::move(key), std::move(state));
  };
  for (State &state : initialStates)
    insert(std::move(state));

  std::vector<Key> unresolved;
  auto appendRemaining = [&] {
    for (const auto &entry : frontier)
      unresolved.push_back(entry.first);
  };

  while (!frontier.empty()) {
    auto node = frontier.extract(frontier.begin());
    Key key = node.key();
    State state = std::move(node.mapped());

    if (budget.maximumEvaluations &&
        work.evaluated >= *budget.maximumEvaluations) {
      unresolved.push_back(std::move(key));
      appendRemaining();
      return SearchControlResult<Key, Accepted>(
          std::move(baseline), work, std::move(unresolved),
          /*frontierExhausted=*/false);
    }

    ++work.evaluated;
    SearchEvaluation<Accepted> outcome = evaluate(state);
    switch (outcome.getKind()) {
    case SearchEvaluationKind::Expandable: {
      if (budget.maximumExpansions &&
          work.expanded >= *budget.maximumExpansions) {
        unresolved.push_back(std::move(key));
        appendRemaining();
        return SearchControlResult<Key, Accepted>(
            std::move(baseline), work, std::move(unresolved),
            /*frontierExhausted=*/false);
      }
      ++work.expanded;
      std::vector<State> children = expand(state);
      for (State &child : children)
        insert(std::move(child));
      break;
    }
    case SearchEvaluationKind::Accepted: {
      ++work.accepted;
      Accepted candidate = outcome.takeAccepted();
      if (isBetter(candidate, baseline))
        baseline = std::move(candidate);
      break;
    }
    case SearchEvaluationKind::ExactRejection:
      ++work.exactRejected;
      break;
    case SearchEvaluationKind::Deferred:
      ++work.deferred;
      unresolved.push_back(std::move(key));
      break;
    case SearchEvaluationKind::Indeterminate:
      ++work.indeterminate;
      unresolved.push_back(std::move(key));
      break;
    case SearchEvaluationKind::Fatal:
      return mlir::failure();
    }
  }

  return SearchControlResult<Key, Accepted>(
      std::move(baseline), work, std::move(unresolved),
      /*frontierExhausted=*/true);
}

} // namespace wafer::compiler::detail

#endif // WAFER_COMPILER_SEARCH_SEARCHCONTROL_H
