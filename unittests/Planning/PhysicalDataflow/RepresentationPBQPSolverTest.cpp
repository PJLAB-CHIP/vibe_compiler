//===- RepresentationPBQPSolverTest.cpp ------------------------------===//

#include "Wafer/Planning/PhysicalDataflow/RepresentationPBQPSolver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include "gtest/gtest.h"

#include <functional>
#include <limits>
#include <optional>
#include <vector>

namespace {

using namespace wafer::compiler::detail;

RepresentationPBQPBinaryFactor
factor(uint32_t lhs, uint32_t rhs, uint32_t states,
       llvm::function_ref<RepresentationPBQPCost(uint32_t, uint32_t)> cost) {
  RepresentationPBQPBinaryFactor result;
  result.lhs = lhs;
  result.rhs = rhs;
  result.lhsStates = states;
  result.rhsStates = states;
  for (uint32_t i = 0; i < states; ++i)
    for (uint32_t j = 0; j < states; ++j)
      result.costs.push_back(cost(i, j));
  return result;
}

std::optional<std::pair<RepresentationPBQPCost, std::vector<uint32_t>>>
bruteForce(const RepresentationPBQPProblem &problem) {
  std::optional<std::pair<RepresentationPBQPCost, std::vector<uint32_t>>> best;
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
    RepresentationPBQPCost cost = 0;
    auto add = [&](RepresentationPBQPCost next) {
      if (cost == kRepresentationPBQPInfinity ||
          next == kRepresentationPBQPInfinity ||
          next > kRepresentationPBQPInfinity - cost)
        cost = kRepresentationPBQPInfinity;
      else
        cost += next;
    };
    for (auto [index, state] : llvm::enumerate(assignment))
      add(problem.variables[index].unaryCosts[state]);
    for (const RepresentationPBQPBinaryFactor &edge : problem.factors)
      add(edge.costs[static_cast<size_t>(assignment[edge.lhs]) *
                         edge.rhsStates +
                     assignment[edge.rhs]]);
    if (cost == kRepresentationPBQPInfinity)
      return;
    if (!best || cost < best->first ||
        (cost == best->first && assignment < best->second))
      best = {cost, assignment};
  };
  visit(0);
  return best;
}

RepresentationPBQPProblem makeProblem(unsigned nodes, unsigned states) {
  RepresentationPBQPProblem problem;
  for (unsigned node = 0; node < nodes; ++node) {
    RepresentationPBQPVariable variable;
    for (unsigned state = 0; state < states; ++state)
      variable.unaryCosts.push_back((node * 7 + state * 3) % 11);
    problem.variables.push_back(std::move(variable));
  }
  return problem;
}

void expectOracle(const RepresentationPBQPProblem &problem) {
  auto expected = bruteForce(problem);
  ASSERT_TRUE(expected);
  RepresentationPBQPResult actual =
      solveRepresentationPBQP(problem, /*workLimit=*/1000000);
  ASSERT_EQ(actual.status, RepresentationPBQPStatus::Optimal);
  ASSERT_TRUE(actual.cost);
  EXPECT_EQ(*actual.cost, expected->first);
  EXPECT_EQ(actual.assignment, expected->second);
  EXPECT_EQ(actual.lowerBound, expected->first);
}

TEST(RepresentationPBQPSolverTest, R0R1R2PathAndResidualCycleMatchFlatOracle) {
  RepresentationPBQPProblem path = makeProblem(/*nodes=*/6, /*states=*/3);
  for (uint32_t node = 0; node + 1 < path.variables.size(); ++node)
    path.factors.push_back(
        factor(node, node + 1, 3, [=](uint32_t lhs, uint32_t rhs) {
          return lhs == rhs ? node + 1 : 0;
        }));
  expectOracle(path);

  RepresentationPBQPProblem cycle = makeProblem(/*nodes=*/5, /*states=*/3);
  for (uint32_t node = 0; node < cycle.variables.size(); ++node)
    cycle.factors.push_back(factor(node, (node + 1) % cycle.variables.size(), 3,
                                   [=](uint32_t lhs, uint32_t rhs) {
                                     return lhs == rhs
                                                ? kRepresentationPBQPInfinity
                                                : (lhs + rhs + node) % 5;
                                   }));
  expectOracle(cycle);
}

TEST(RepresentationPBQPSolverTest, ResidualCliqueUsesStableOptimalTieBreak) {
  RepresentationPBQPProblem clique = makeProblem(/*nodes=*/4, /*states=*/3);
  for (uint32_t lhs = 0; lhs < clique.variables.size(); ++lhs)
    for (uint32_t rhs = lhs + 1; rhs < clique.variables.size(); ++rhs)
      clique.factors.push_back(
          factor(lhs, rhs, 3, [](uint32_t left, uint32_t right) {
            return left == right ? 2 : 0;
          }));
  expectOracle(clique);
}

TEST(RepresentationPBQPSolverTest,
     NoSolutionBudgetAndMalformedProblemRemainDistinct) {
  RepresentationPBQPProblem noSolution = makeProblem(/*nodes=*/2, /*states=*/2);
  noSolution.factors.push_back(factor(
      0, 1, 2, [](uint32_t, uint32_t) { return kRepresentationPBQPInfinity; }));
  EXPECT_EQ(solveRepresentationPBQP(noSolution, 1000).status,
            RepresentationPBQPStatus::NoSolution);
  EXPECT_EQ(solveRepresentationPBQP(makeProblem(5, 4), 1).status,
            RepresentationPBQPStatus::Indeterminate);

  RepresentationPBQPProblem malformed = makeProblem(2, 2);
  malformed.factors.push_back(
      factor(0, 1, 2, [](uint32_t, uint32_t) { return 0; }));
  malformed.factors.push_back(malformed.factors.front());
  EXPECT_EQ(solveRepresentationPBQP(malformed, 1000).status,
            RepresentationPBQPStatus::BrokenContract);
}

} // namespace
