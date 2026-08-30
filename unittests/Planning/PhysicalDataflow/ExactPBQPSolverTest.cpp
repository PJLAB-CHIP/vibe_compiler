//===- ExactPBQPSolverTest.cpp -----------------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/ExactPBQPSolver.h"

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

void expectOracle(const ExactPBQPProblem &problem) {
  auto expected = bruteForce(problem);
  ASSERT_TRUE(expected);
  ExactPBQPResult actual = solveExactPBQP(problem, /*workLimit=*/1000000);
  ASSERT_EQ(actual.status, ExactPBQPStatus::Optimal);
  ASSERT_TRUE(actual.cost);
  EXPECT_EQ(*actual.cost, expected->first);
  EXPECT_EQ(actual.assignment, expected->second);
  EXPECT_EQ(actual.lowerBound, expected->first);
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
  ExactPBQPResult solved = solveExactPBQP(problem, /*workLimit=*/1000);
  ASSERT_EQ(solved.status, ExactPBQPStatus::Optimal);
  EXPECT_EQ(solved.cost, 0u);
  EXPECT_EQ(solved.assignment, (std::vector<uint32_t>{0, 1}));
}

TEST(ExactPBQPSolverTest, NoSolutionBudgetAndMalformedProblemRemainDistinct) {
  ExactPBQPProblem noSolution = makeProblem(/*nodes=*/2, /*states=*/2);
  noSolution.factors.push_back(
      factor(0, 1, 2, [](uint32_t, uint32_t) { return kExactPBQPInfinity; }));
  EXPECT_EQ(solveExactPBQP(noSolution, 1000).status,
            ExactPBQPStatus::NoSolution);
  EXPECT_EQ(solveExactPBQP(makeProblem(5, 4), 1).status,
            ExactPBQPStatus::Indeterminate);

  ExactPBQPProblem malformed = makeProblem(2, 2);
  malformed.factors.push_back(
      factor(0, 1, 2, [](uint32_t, uint32_t) { return 0; }));
  malformed.factors.push_back(malformed.factors.front());
  EXPECT_EQ(solveExactPBQP(malformed, 1000).status,
            ExactPBQPStatus::BrokenContract);
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
  ExactPBQPResult solved = solveExactPBQP(problem, /*workLimit=*/10000,
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
  ExactPBQPResult first = solveExactPBQP(problem, /*workLimit=*/1000,
                                         /*semanticTieVariableCount=*/1);
  ExactPBQPResult second = solveExactPBQP(problem, /*workLimit=*/1000,
                                          /*semanticTieVariableCount=*/1);
  ASSERT_EQ(first.status, ExactPBQPStatus::Optimal);
  EXPECT_EQ(first.assignment.front(), 0u);
  EXPECT_EQ(first.assignment, second.assignment);
  EXPECT_EQ(first.cost, second.cost);
}

} // namespace
