//===- SearchControlTest.cpp ------------------------------------------===//

#include "Wafer/Planning/Search/SearchControl.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <map>
#include <utility>
#include <vector>

namespace {

struct State {
  int key;
};

struct Accepted {
  int key;
  int cost;
};

using Evaluation =
    wafer::compiler::detail::SearchEvaluation<Accepted>;

static bool better(const Accepted &candidate, const Accepted &incumbent) {
  return std::tie(candidate.cost, candidate.key) <
         std::tie(incumbent.cost, incumbent.key);
}

TEST(SearchControlTest, EmptyDomainReturnsTheBaselineWithoutCandidateWork) {
  auto result = wafer::compiler::detail::runSearchControl<State, int, Accepted>(
      {}, Accepted{-1, 100},
      wafer::compiler::detail::SearchWorkBudget::bounded(0, 0),
      [](const State &state) { return state.key; },
      [](const State &) { return std::vector<State>{}; },
      [](const State &) { return Evaluation::fatal(); }, better);

  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->incumbent.key, -1);
  EXPECT_TRUE(result->frontierExhausted);
  EXPECT_TRUE(result->unresolvedKeys.empty());
  EXPECT_EQ(result->work.generated, 0u);
  EXPECT_EQ(result->work.evaluated, 0u);
}

TEST(SearchControlTest, DiamondIsDeterministicAndDeduplicatesTheJoin) {
  auto run = [](std::vector<State> firstChildren) {
    std::vector<int> evaluationOrder;
    auto result =
        wafer::compiler::detail::runSearchControl<State, int, Accepted>(
            {{0}}, Accepted{-1, 100},
            wafer::compiler::detail::SearchWorkBudget::unlimited(),
            [](const State &state) { return state.key; },
            [&](const State &state) {
              if (state.key == 0)
                return firstChildren;
              if (state.key == 1 || state.key == 2)
                return std::vector<State>{{3}};
              return std::vector<State>{};
            },
            [&](const State &state) {
              evaluationOrder.push_back(state.key);
              if (state.key == 3)
                return Evaluation::accepted({3, 10});
              return Evaluation::expandable();
            },
            better);
    return std::make_pair(std::move(result), std::move(evaluationOrder));
  };

  auto [forward, forwardOrder] = run({{1}, {2}});
  auto [reverse, reverseOrder] = run({{2}, {1}});
  ASSERT_TRUE(mlir::succeeded(forward));
  ASSERT_TRUE(mlir::succeeded(reverse));
  EXPECT_EQ(forwardOrder, (std::vector<int>{0, 1, 2, 3}));
  EXPECT_EQ(reverseOrder, forwardOrder);
  EXPECT_EQ(forward->incumbent.key, 3);
  EXPECT_EQ(reverse->incumbent.key, 3);
  EXPECT_EQ(forward->work.generated, 4u);
  EXPECT_EQ(forward->work.deduplicated, 1u);
  EXPECT_TRUE(forward->frontierExhausted);
}

TEST(SearchControlTest, ExactRejectionDoesNotDropAnAcceptedSibling) {
  auto result = wafer::compiler::detail::runSearchControl<State, int, Accepted>(
      {{0}}, Accepted{-1, 100},
      wafer::compiler::detail::SearchWorkBudget::unlimited(),
      [](const State &state) { return state.key; },
      [](const State &state) {
        if (state.key == 0)
          return std::vector<State>{{1}, {2}};
        return std::vector<State>{};
      },
      [](const State &state) {
        if (state.key == 1)
          return Evaluation::exactRejection();
        if (state.key == 2)
          return Evaluation::accepted({2, 5});
        return Evaluation::expandable();
      },
      better);

  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->incumbent.key, 2);
  EXPECT_EQ(result->work.exactRejected, 1u);
  EXPECT_EQ(result->work.accepted, 1u);
}

TEST(SearchControlTest, DeferredAndIndeterminateStatesRemainUnresolved) {
  auto result = wafer::compiler::detail::runSearchControl<State, int, Accepted>(
      {{2}, {1}}, Accepted{-1, 100},
      wafer::compiler::detail::SearchWorkBudget::unlimited(),
      [](const State &state) { return state.key; },
      [](const State &) { return std::vector<State>{}; },
      [](const State &state) {
        return state.key == 1 ? Evaluation::deferred()
                              : Evaluation::indeterminate();
      },
      better);

  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_EQ(result->unresolvedKeys, (std::vector<int>{1, 2}));
  EXPECT_TRUE(result->frontierExhausted);
  EXPECT_EQ(result->work.deferred, 1u);
  EXPECT_EQ(result->work.indeterminate, 1u);
}

TEST(SearchControlTest, BudgetStopPreservesTheCurrentAndRemainingStates) {
  auto result = wafer::compiler::detail::runSearchControl<State, int, Accepted>(
      {{0}}, Accepted{-1, 100},
      wafer::compiler::detail::SearchWorkBudget::bounded(
          /*evaluations=*/2, /*expansions=*/2),
      [](const State &state) { return state.key; },
      [](const State &state) {
        if (state.key == 0)
          return std::vector<State>{{1}, {2}};
        return std::vector<State>{};
      },
      [](const State &) { return Evaluation::expandable(); }, better);

  ASSERT_TRUE(mlir::succeeded(result));
  EXPECT_FALSE(result->frontierExhausted);
  EXPECT_EQ(result->unresolvedKeys, (std::vector<int>{2}));
  EXPECT_EQ(result->work.evaluated, 2u);
  EXPECT_EQ(result->work.expanded, 2u);
}

TEST(SearchControlTest, FatalEvaluationFailsInsteadOfReturningTheBaseline) {
  auto result = wafer::compiler::detail::runSearchControl<State, int, Accepted>(
      {{0}}, Accepted{-1, 100},
      wafer::compiler::detail::SearchWorkBudget::unlimited(),
      [](const State &state) { return state.key; },
      [](const State &) { return std::vector<State>{}; },
      [](const State &) { return Evaluation::fatal(); }, better);

  EXPECT_TRUE(mlir::failed(result));
}

} // namespace
