//===- ExactPBQPSolverTest.cpp -----------------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/ExactPBQPSolver.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <vector>

namespace {

using namespace wafer::compiler::detail;

ExactPBQPBinaryFactor
factor(uint32_t lhs, uint32_t rhs, uint32_t states,
       llvm::function_ref<ExactPBQPCost(uint32_t, uint32_t)> cost) {
  ExactPBQPBinaryFactor result;
  result.lhs = lhs;
  result.rhs = rhs;
  result.lhsStates = states;
  result.rhsStates = states;
  for (uint32_t i = 0; i < states; ++i)
    for (uint32_t j = 0; j < states; ++j)
      result.costs.push_back(cost(i, j));
  return result;
}

ExactPBQPBinaryFactor
rectFactor(uint32_t lhs, uint32_t rhs, uint32_t lhsStates, uint32_t rhsStates,
           llvm::function_ref<ExactPBQPCost(uint32_t, uint32_t)> cost) {
  ExactPBQPBinaryFactor result;
  result.lhs = lhs;
  result.rhs = rhs;
  result.lhsStates = lhsStates;
  result.rhsStates = rhsStates;
  for (uint32_t i = 0; i < lhsStates; ++i)
    for (uint32_t j = 0; j < rhsStates; ++j)
      result.costs.push_back(cost(i, j));
  return result;
}

std::optional<std::pair<ExactPBQPCost, std::vector<uint32_t>>>
bruteForce(const ExactPBQPProblem &problem) {
  std::optional<std::pair<ExactPBQPCost, std::vector<uint32_t>>> best;
  std::vector<uint32_t> assignment(problem.variables.size());
  std::function<void(size_t)> visit = [&](size_t variable) {
    if (variable != problem.variables.size()) {
      for (uint32_t state = 0;
           state < problem.variables[variable].unaryCosts.size(); ++state) {
        assignment[variable] = state;
        visit(variable + 1);
      }
      return;
    }
    ExactPBQPCost cost = 0;
    auto add = [&](ExactPBQPCost next) {
      if (cost == kExactPBQPInfinity || next == kExactPBQPInfinity ||
          next > kExactPBQPInfinity - cost)
        cost = kExactPBQPInfinity;
      else
        cost += next;
    };
    for (auto [index, state] : llvm::enumerate(assignment))
      add(problem.variables[index].unaryCosts[state]);
    for (const ExactPBQPBinaryFactor &edge : problem.factors)
      add(edge.costs[static_cast<size_t>(assignment[edge.lhs]) *
                         edge.rhsStates +
                     assignment[edge.rhs]]);
    if (cost == kExactPBQPInfinity)
      return;
    if (!best || cost < best->first ||
        (cost == best->first && assignment < best->second))
      best = {cost, assignment};
  };
  visit(0);
  return best;
}

ExactPBQPProblem makeProblem(unsigned nodes, unsigned states) {
  ExactPBQPProblem problem;
  for (unsigned node = 0; node < nodes; ++node) {
    ExactPBQPVariable variable;
    for (unsigned state = 0; state < states; ++state)
      variable.unaryCosts.push_back((node * 7 + state * 3) % 11);
    problem.variables.push_back(std::move(variable));
  }
  return problem;
}

ExactPBQPResult
solve(const ExactPBQPProblem &problem, uint64_t workLimit,
      uint32_t semanticTieVariableCount = std::numeric_limits<uint32_t>::max(),
      std::optional<std::vector<uint32_t>> initialFeasibleAssignment =
          std::nullopt) {
  ExactPBQPSolveOptions options;
  options.workLimit = workLimit;
  options.semanticTieVariableCount = semanticTieVariableCount;
  options.initialFeasibleAssignment = std::move(initialFeasibleAssignment);
  return solveExactPBQP(problem, options);
}

void expectOracle(const ExactPBQPProblem &problem) {
  auto expected = bruteForce(problem);
  ASSERT_TRUE(expected);
  ExactPBQPResult actual = solve(problem, /*workLimit=*/1000000);
  ASSERT_EQ(actual.status, ExactPBQPStatus::Optimal);
  ASSERT_TRUE(actual.cost);
  EXPECT_EQ(*actual.cost, expected->first);
  EXPECT_EQ(actual.assignment, expected->second);
  EXPECT_EQ(actual.lowerBound, expected->first);
}

TEST(ExactPBQPSolverTest, RepeatedComponentsReuseAnExactOptimumWithinBudget) {
  // A small exhaustive oracle checks the mathematical solver, independently
  // of tensor sizes; the production witness supplies the real-scale IR.
  auto component = makeProblem(5, 3);
  for (uint32_t lhs = 0; lhs < 5; ++lhs)
    for (uint32_t rhs = lhs + 1; rhs < 5; ++rhs)
      component.factors.push_back(
          factor(lhs, rhs, 3, [](auto a, auto b) { return a == b ? 4 : 0; }));
  auto oracle = bruteForce(component);
  ASSERT_TRUE(oracle);
  std::vector<uint32_t> initial(5, 0);
  auto single = solve(component, 1000000, 5, initial);
  ASSERT_EQ(single.status, ExactPBQPStatus::Optimal);
  ASSERT_EQ(single.assignment, oracle->second);
  ExactPBQPProblem repeated;
  constexpr unsigned copies = 16;
  std::vector<uint32_t> expected;
  for (unsigned i = 0; i < copies; ++i) {
    auto offset = static_cast<uint32_t>(repeated.variables.size());
    llvm::append_range(repeated.variables, component.variables);
    for (auto edge : component.factors) {
      edge.lhs += offset;
      edge.rhs += offset;
      repeated.factors.push_back(std::move(edge));
    }
    llvm::append_range(expected, oracle->second);
  }
  initial.resize(copies * 5, 0);
  const uint64_t budget = single.work * 2 + copies * 200;
  auto solved = solve(repeated, budget, copies * 5, initial);
  ASSERT_EQ(solved.status, ExactPBQPStatus::Optimal);
  EXPECT_EQ(solved.assignment, expected);
  EXPECT_EQ(solved.cost, oracle->first * copies);
  EXPECT_LE(solved.work, budget);
  EXPECT_LT(solved.work, single.work * copies);
  auto again = solve(repeated, budget, copies * 5, initial);
  EXPECT_EQ(again.assignment, solved.assignment);
  EXPECT_EQ(again.work, solved.work);

  // A single changed cost must invalidate the identical-component lookup.
  repeated.variables.back().unaryCosts[oracle->second.back()] += 17;
  component.variables.back() = repeated.variables.back();
  auto changedOracle = bruteForce(component);
  ASSERT_TRUE(changedOracle);
  auto changed = solve(repeated, 1000000, copies * 5, initial);
  ASSERT_EQ(changed.status, ExactPBQPStatus::Optimal);
  std::copy(changedOracle->second.begin(), changedOracle->second.end(),
            expected.end() - 5);
  EXPECT_EQ(changed.assignment, expected);
  EXPECT_EQ(changed.cost, oracle->first * (copies - 1) + changedOracle->first);
  auto zero = solve(repeated, 0, copies * 5, initial);
  EXPECT_EQ(zero.status, ExactPBQPStatus::Feasible);
  EXPECT_EQ(zero.assignment, initial);
  EXPECT_EQ(zero.work, 0u);
}

TEST(ExactPBQPSolverTest, R0R1R2PathAndResidualCycleMatchFlatOracle) {
  ExactPBQPProblem path = makeProblem(/*nodes=*/6, /*states=*/3);
  for (uint32_t node = 0; node + 1 < path.variables.size(); ++node)
    path.factors.push_back(
        factor(node, node + 1, 3, [=](uint32_t lhs, uint32_t rhs) {
          return lhs == rhs ? node + 1 : 0;
        }));
  expectOracle(path);

  ExactPBQPProblem cycle = makeProblem(/*nodes=*/5, /*states=*/3);
  for (uint32_t node = 0; node < cycle.variables.size(); ++node)
    cycle.factors.push_back(factor(node, (node + 1) % cycle.variables.size(), 3,
                                   [=](uint32_t lhs, uint32_t rhs) {
                                     return lhs == rhs ? kExactPBQPInfinity
                                                       : (lhs + rhs + node) % 5;
                                   }));
  expectOracle(cycle);
}

TEST(ExactPBQPSolverTest, ResidualCliqueUsesStableOptimalTieBreak) {
  ExactPBQPProblem clique = makeProblem(/*nodes=*/4, /*states=*/3);
  for (uint32_t lhs = 0; lhs < clique.variables.size(); ++lhs)
    for (uint32_t rhs = lhs + 1; rhs < clique.variables.size(); ++rhs)
      clique.factors.push_back(
          factor(lhs, rhs, 3, [](uint32_t left, uint32_t right) {
            return left == right ? 2 : 0;
          }));
  expectOracle(clique);
}

TEST(ExactPBQPSolverTest,
     ReductionReconstructionUsesTheGlobalAssignmentTieBreak) {
  ExactPBQPProblem problem = makeProblem(/*nodes=*/2, /*states=*/2);
  for (ExactPBQPVariable &variable : problem.variables)
    std::fill(variable.unaryCosts.begin(), variable.unaryCosts.end(), 0);
  problem.factors.push_back(factor(0, 1, 2, [](uint32_t lhs, uint32_t rhs) {
    return lhs == rhs ? ExactPBQPCost{1} : ExactPBQPCost{0};
  }));
  ExactPBQPResult solved = solve(problem, /*workLimit=*/1000);
  ASSERT_EQ(solved.status, ExactPBQPStatus::Optimal);
  EXPECT_EQ(solved.cost, 0u);
  EXPECT_EQ(solved.assignment, (std::vector<uint32_t>{0, 1}));
}

TEST(ExactPBQPSolverTest, NoSolutionBudgetAndMalformedProblemRemainDistinct) {
  ExactPBQPProblem noSolution = makeProblem(/*nodes=*/2, /*states=*/2);
  noSolution.factors.push_back(
      factor(0, 1, 2, [](uint32_t, uint32_t) { return kExactPBQPInfinity; }));
  EXPECT_EQ(solve(noSolution, 1000).status, ExactPBQPStatus::NoSolution);
  EXPECT_EQ(solve(makeProblem(5, 4), 1).status, ExactPBQPStatus::Indeterminate);

  ExactPBQPProblem malformed = makeProblem(2, 2);
  malformed.factors.push_back(
      factor(0, 1, 2, [](uint32_t, uint32_t) { return 0; }));
  malformed.factors.push_back(malformed.factors.front());
  EXPECT_EQ(solve(malformed, 1000).status, ExactPBQPStatus::BrokenContract);
}

TEST(ExactPBQPSolverTest,
     ValidIncumbentSurvivesBudgetExhaustionWithoutClaimingOptimality) {
  ExactPBQPProblem problem = makeProblem(/*nodes=*/5, /*states=*/4);
  std::vector<uint32_t> incumbent{3, 2, 1, 0, 3};
  ExactPBQPResult fallback =
      solve(problem, /*workLimit=*/0, std::numeric_limits<uint32_t>::max(),
            incumbent);
  ASSERT_EQ(fallback.status, ExactPBQPStatus::Feasible);
  EXPECT_EQ(fallback.assignment, incumbent);
  ASSERT_TRUE(fallback.cost);
  EXPECT_EQ(fallback.lowerBound, 0u);

  auto expected = bruteForce(problem);
  ASSERT_TRUE(expected);
  ExactPBQPResult optimized =
      solve(problem, /*workLimit=*/1000000,
            std::numeric_limits<uint32_t>::max(), incumbent);
  ASSERT_EQ(optimized.status, ExactPBQPStatus::Optimal);
  EXPECT_EQ(optimized.assignment, expected->second);
  EXPECT_EQ(optimized.cost, expected->first);
}

TEST(ExactPBQPSolverTest, InvalidIncumbentIsABrokenCallerContract) {
  ExactPBQPProblem problem = makeProblem(/*nodes=*/2, /*states=*/2);
  problem.factors.push_back(factor(0, 1, 2, [](uint32_t lhs, uint32_t rhs) {
    return lhs == rhs ? kExactPBQPInfinity : ExactPBQPCost{0};
  }));
  EXPECT_EQ(solve(problem, /*workLimit=*/0,
                  std::numeric_limits<uint32_t>::max(),
                  std::vector<uint32_t>{0, 0})
                .status,
            ExactPBQPStatus::BrokenContract);
}

TEST(ExactPBQPSolverTest, ExhaustionRetainsSearchComponentsAndNumericOptima) {
  // Tiny finite graphs are intentional: enumerate all assignments as an
  // independent oracle, and interrupt every solver work boundary.
  for (unsigned componentCount : {1u, 2u}) {
    ExactPBQPProblem problem;
    // Distinct components force independent numeric and tie phases; identical
    // components can now advance directly to a previously proved optimum.
    for (unsigned node = 0; node < 4 * componentCount; ++node)
      problem.variables.push_back({{9, node / 4, 4}});
    for (unsigned component = 0; component < componentCount; ++component)
      for (unsigned lhs = 0; lhs < 4; ++lhs)
        for (unsigned rhs = lhs + 1; rhs < 4; ++rhs)
          problem.factors.push_back(
              factor(component * 4 + lhs, component * 4 + rhs, 3,
                     [](uint32_t a, uint32_t b) { return a == b ? 0 : 2; }));
    auto expected = bruteForce(problem);
    ASSERT_TRUE(expected);
    std::vector<uint32_t> initial(problem.variables.size(), 0);
    ExactPBQPCost previous = 9 * initial.size();
    bool keptPartialSearch = false, keptNumericOptimum = false;
    bool keptComponent = false;
    for (uint64_t budget = 0; budget < 1600; ++budget) {
      auto solved = solve(problem, budget, initial.size(), initial);
      SCOPED_TRACE(::testing::Message() << componentCount << "/" << budget);
      ASSERT_TRUE(solved.status == ExactPBQPStatus::Optimal ||
                  solved.status == ExactPBQPStatus::Feasible);
      ASSERT_EQ(solved.assignment.size(), initial.size());
      ASSERT_TRUE(solved.cost);
      EXPECT_LE(solved.work, budget);
      EXPECT_LE(*solved.cost, previous);
      previous = *solved.cost;
      // Evaluate this exact returned assignment through the independent oracle.
      auto fixed = problem;
      for (unsigned node = 0; node < fixed.variables.size(); ++node)
        for (unsigned state = 0; state < 3; ++state)
          if (state != solved.assignment[node])
            fixed.variables[node].unaryCosts[state] = kExactPBQPInfinity;
      auto checked = bruteForce(fixed);
      ASSERT_TRUE(checked);
      EXPECT_EQ(solved.cost, checked->first);
      EXPECT_LE(solved.lowerBound, *solved.cost);
      if (solved.status == ExactPBQPStatus::Optimal) {
        EXPECT_EQ(solved.cost, expected->first);
        EXPECT_EQ(solved.assignment, expected->second);
      } else {
        keptPartialSearch |=
            *solved.cost > expected->first && *solved.cost < 9 * initial.size();
        keptNumericOptimum |= solved.cost == expected->first &&
                              solved.lowerBound == expected->first;
        if (componentCount == 2)
          keptComponent |=
              llvm::all_of(llvm::ArrayRef(solved.assignment).take_front(4),
                           [](uint32_t state) { return state == 1; }) &&
              llvm::all_of(llvm::ArrayRef(solved.assignment).drop_front(4),
                           [](uint32_t state) { return state == 0; });
      }
      if (budget % 31 == 0) {
        auto repeated = solve(problem, budget, initial.size(), initial);
        EXPECT_EQ(solved.assignment, repeated.assignment);
        EXPECT_EQ(solved.status, repeated.status);
        EXPECT_EQ(solved.work, repeated.work);
      }
    }
    EXPECT_TRUE(keptPartialSearch);
    EXPECT_TRUE(keptNumericOptimum);
    if (componentCount == 2) {
      EXPECT_TRUE(keptComponent);
    }
  }
}

TEST(ExactPBQPSolverTest,
     DisconnectedComponentsPreserveGlobalCostAndAssignmentTie) {
  ExactPBQPProblem problem = makeProblem(/*nodes=*/6, /*states=*/2);
  problem.factors.push_back(factor(0, 1, 2, [](uint32_t lhs, uint32_t rhs) {
    return lhs == rhs ? ExactPBQPCost{0} : ExactPBQPCost{3};
  }));
  problem.factors.push_back(factor(1, 2, 2, [](uint32_t lhs, uint32_t rhs) {
    return lhs == rhs ? ExactPBQPCost{2} : ExactPBQPCost{0};
  }));
  problem.factors.push_back(factor(3, 4, 2, [](uint32_t lhs, uint32_t rhs) {
    return lhs == rhs ? ExactPBQPCost{1} : ExactPBQPCost{0};
  }));
  problem.factors.push_back(factor(4, 5, 2, [](uint32_t lhs, uint32_t rhs) {
    return lhs == rhs ? ExactPBQPCost{0} : ExactPBQPCost{4};
  }));
  expectOracle(problem);
}

TEST(ExactPBQPSolverTest, OneStateHubReducesAtArbitraryDegree) {
  ExactPBQPProblem problem;
  problem.variables.push_back({{0}});
  constexpr uint32_t leaves = 64;
  for (uint32_t leaf = 0; leaf < leaves; ++leaf) {
    problem.variables.push_back({{7, 5, 3, 1}});
    problem.factors.push_back(
        rectFactor(0, leaf + 1, 1, 4, [=](uint32_t, uint32_t state) {
          return state == leaf % 4 ? ExactPBQPCost{0} : ExactPBQPCost{2};
        }));
  }
  ExactPBQPResult solved = solve(problem, /*workLimit=*/10000,
                                 /*semanticTieVariableCount=*/0);
  ASSERT_EQ(solved.status, ExactPBQPStatus::Optimal);
  ASSERT_EQ(solved.assignment.size(), leaves + 1);
  EXPECT_EQ(solved.assignment.front(), 0u);
  EXPECT_LT(solved.work, 10000u);
}

TEST(ExactPBQPSolverTest,
     AuxiliaryVariablesRemainDeterministicOutsideSemanticTiePrefix) {
  ExactPBQPProblem problem = makeProblem(/*nodes=*/2, /*states=*/2);
  for (ExactPBQPVariable &variable : problem.variables)
    std::fill(variable.unaryCosts.begin(), variable.unaryCosts.end(), 0);
  problem.factors.push_back(factor(0, 1, 2, [](uint32_t lhs, uint32_t rhs) {
    return lhs == rhs ? ExactPBQPCost{1} : ExactPBQPCost{0};
  }));
  ExactPBQPResult first = solve(problem, /*workLimit=*/1000,
                                /*semanticTieVariableCount=*/1);
  ExactPBQPResult second = solve(problem, /*workLimit=*/1000,
                                 /*semanticTieVariableCount=*/1);
  ASSERT_EQ(first.status, ExactPBQPStatus::Optimal);
  EXPECT_EQ(first.assignment.front(), 0u);
  EXPECT_EQ(first.assignment, second.assignment);
  EXPECT_EQ(first.cost, second.cost);
  auto seeded =
      solve(problem, /*workLimit=*/1000,
            /*semanticTieVariableCount=*/0, std::vector<uint32_t>{0, 1});
  EXPECT_EQ(seeded.status, ExactPBQPStatus::Optimal);
  EXPECT_EQ(seeded.cost, first.cost);
}

} // namespace
