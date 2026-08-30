//===- ExactPBQPSolver.cpp - Exact finite PBQP ------------------------===//

#include "Wafer/Planning/PhysicalDataflow/ExactPBQPSolver.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <functional>
#include <set>
#include <tuple>

namespace wafer::compiler::detail {
namespace {

struct Budget {
  uint64_t limit = 0;
  uint64_t used = 0;

  bool consume(uint64_t amount = 1) {
    if (amount > limit - std::min(limit, used))
      return false;
    used += amount;
    return true;
  }
};

ExactPBQPCost addCost(ExactPBQPCost lhs, ExactPBQPCost rhs) {
  if (lhs == kExactPBQPInfinity || rhs == kExactPBQPInfinity ||
      rhs > kExactPBQPInfinity - lhs)
    return kExactPBQPInfinity;
  return lhs + rhs;
}

struct Edge {
  uint32_t lhs = 0;
  uint32_t rhs = 0;
  uint32_t lhsStates = 0;
  uint32_t rhsStates = 0;
  std::vector<ExactPBQPCost> costs;
  bool active = true;

  ExactPBQPCost &at(uint32_t lhsState, uint32_t rhsState) {
    return costs[lhsState * rhsStates + rhsState];
  }
  ExactPBQPCost at(uint32_t lhsState, uint32_t rhsState) const {
    return costs[lhsState * rhsStates + rhsState];
  }
};

struct Graph {
  std::vector<std::vector<ExactPBQPCost>> nodes;
  std::vector<Edge> edges;
};

ExactPBQPCost edgeCost(const Edge &edge, uint32_t node, uint32_t state,
                       uint32_t neighbor, uint32_t neighborState) {
  return edge.lhs == node && edge.rhs == neighbor
             ? edge.at(state, neighborState)
             : edge.at(neighborState, state);
}

llvm::SmallVector<uint32_t, 4>
neighbors(const Graph &graph, llvm::ArrayRef<bool> active, uint32_t node) {
  llvm::SmallVector<uint32_t, 4> result;
  for (const Edge &edge : graph.edges) {
    if (!edge.active)
      continue;
    if (edge.lhs == node && active[edge.rhs])
      result.push_back(edge.rhs);
    else if (edge.rhs == node && active[edge.lhs])
      result.push_back(edge.lhs);
  }
  llvm::sort(result);
  return result;
}

Edge *findEdge(Graph &graph, uint32_t lhs, uint32_t rhs) {
  if (lhs > rhs)
    std::swap(lhs, rhs);
  auto found = llvm::find_if(graph.edges, [&](const Edge &edge) {
    return edge.active && edge.lhs == lhs && edge.rhs == rhs;
  });
  return found == graph.edges.end() ? nullptr : &*found;
}

std::optional<uint32_t> minimumState(llvm::ArrayRef<ExactPBQPCost> costs,
                                     Budget &budget) {
  std::optional<uint32_t> selected;
  for (auto [state, cost] : llvm::enumerate(costs)) {
    if (!budget.consume())
      return std::nullopt;
    if (cost == kExactPBQPInfinity)
      continue;
    if (!selected || cost < costs[*selected])
      selected = static_cast<uint32_t>(state);
  }
  return selected;
}

struct ReductionRecord {
  uint32_t node = 0;
  llvm::SmallVector<uint32_t, 2> neighbors;
  std::vector<uint32_t> choices;
  bool fixedState = false;
};

bool validateAndBuild(const ExactPBQPProblem &problem, Graph &graph) {
  if (problem.variables.empty())
    return false;
  graph.nodes.reserve(problem.variables.size());
  for (const ExactPBQPVariable &variable : problem.variables) {
    if (variable.unaryCosts.empty())
      return false;
    graph.nodes.push_back(variable.unaryCosts);
  }
  std::set<std::pair<uint32_t, uint32_t>> pairs;
  for (const ExactPBQPBinaryFactor &factor : problem.factors) {
    if (factor.lhs >= graph.nodes.size() || factor.rhs >= graph.nodes.size() ||
        factor.lhs == factor.rhs ||
        factor.lhsStates != graph.nodes[factor.lhs].size() ||
        factor.rhsStates != graph.nodes[factor.rhs].size() ||
        factor.costs.size() !=
            static_cast<size_t>(factor.lhsStates) * factor.rhsStates)
      return false;
    uint32_t lhs = factor.lhs;
    uint32_t rhs = factor.rhs;
    bool transpose = lhs > rhs;
    if (transpose)
      std::swap(lhs, rhs);
    if (!pairs.insert({lhs, rhs}).second)
      return false;
    Edge edge;
    edge.lhs = lhs;
    edge.rhs = rhs;
    edge.lhsStates = static_cast<uint32_t>(graph.nodes[lhs].size());
    edge.rhsStates = static_cast<uint32_t>(graph.nodes[rhs].size());
    edge.costs.resize(static_cast<size_t>(edge.lhsStates) * edge.rhsStates);
    for (uint32_t i = 0; i < factor.lhsStates; ++i)
      for (uint32_t j = 0; j < factor.rhsStates; ++j) {
        ExactPBQPCost cost =
            factor.costs[static_cast<size_t>(i) * factor.rhsStates + j];
        if (!transpose)
          edge.at(i, j) = cost;
        else
          edge.at(j, i) = cost;
      }
    graph.edges.push_back(std::move(edge));
  }
  return true;
}

std::optional<ExactPBQPCost> evaluate(const ExactPBQPProblem &problem,
                                      llvm::ArrayRef<uint32_t> assignment) {
  if (assignment.size() != problem.variables.size())
    return std::nullopt;
  ExactPBQPCost result = 0;
  for (auto [index, state] : llvm::enumerate(assignment)) {
    if (state >= problem.variables[index].unaryCosts.size())
      return std::nullopt;
    result = addCost(result, problem.variables[index].unaryCosts[state]);
  }
  for (const ExactPBQPBinaryFactor &factor : problem.factors)
    result = addCost(result,
                     factor.costs[static_cast<size_t>(assignment[factor.lhs]) *
                                      factor.rhsStates +
                                  assignment[factor.rhs]]);
  return result == kExactPBQPInfinity ? std::nullopt
                                      : std::optional<ExactPBQPCost>(result);
}

} // namespace

static std::optional<std::vector<std::vector<uint32_t>>>
getConnectedComponents(const ExactPBQPProblem &problem) {
  if (problem.variables.empty() ||
      llvm::any_of(problem.variables, [](const ExactPBQPVariable &variable) {
        return variable.unaryCosts.empty();
      }))
    return std::nullopt;
  std::vector<std::vector<uint32_t>> adjacency(problem.variables.size());
  for (const ExactPBQPBinaryFactor &factor : problem.factors) {
    if (factor.lhs >= problem.variables.size() ||
        factor.rhs >= problem.variables.size() || factor.lhs == factor.rhs)
      return std::nullopt;
    adjacency[factor.lhs].push_back(factor.rhs);
    adjacency[factor.rhs].push_back(factor.lhs);
  }
  std::vector<bool> visited(problem.variables.size(), false);
  std::vector<std::vector<uint32_t>> components;
  for (uint32_t start = 0; start < problem.variables.size(); ++start) {
    if (visited[start])
      continue;
    std::vector<uint32_t> component;
    llvm::SmallVector<uint32_t, 16> pending{start};
    visited[start] = true;
    while (!pending.empty()) {
      uint32_t node = pending.pop_back_val();
      component.push_back(node);
      llvm::sort(adjacency[node]);
      for (uint32_t neighbor : adjacency[node])
        if (!visited[neighbor]) {
          visited[neighbor] = true;
          pending.push_back(neighbor);
        }
    }
    llvm::sort(component);
    components.push_back(std::move(component));
  }
  llvm::sort(components, [](const auto &lhs, const auto &rhs) {
    return lhs.front() < rhs.front();
  });
  return components;
}

static ExactPBQPResult solveNumericExactPBQP(const ExactPBQPProblem &problem,
                                             Budget &budget) {
  ExactPBQPResult result;
  Graph graph;
  if (budget.limit == 0 || !validateAndBuild(problem, graph))
    return result;
  llvm::SmallVector<bool, 16> active(graph.nodes.size(), true);
  llvm::SmallVector<ReductionRecord, 16> records;
  bool exhausted = false;

  while (true) {
    std::optional<uint32_t> selected;
    llvm::SmallVector<uint32_t, 2> adjacent;
    bool fixedState = false;
    for (uint32_t node = 0; node < graph.nodes.size(); ++node) {
      if (!active[node])
        continue;
      if (graph.nodes[node].size() == 1) {
        selected = node;
        fixedState = true;
        break;
      }
      llvm::SmallVector<uint32_t, 4> current = neighbors(graph, active, node);
      if (current.size() <= 2) {
        selected = node;
        adjacent.assign(current.begin(), current.end());
        break;
      }
    }
    if (!selected)
      break;

    ReductionRecord record;
    record.node = *selected;
    record.neighbors = adjacent;
    record.fixedState = fixedState;
    if (fixedState) {
      if (!budget.consume() ||
          graph.nodes[*selected].front() == kExactPBQPInfinity) {
        exhausted = budget.used >= budget.limit;
        if (!exhausted) {
          result.status = ExactPBQPStatus::NoSolution;
          result.work = budget.used;
          return result;
        }
        break;
      }
      record.choices.push_back(0);
      // A one-state node can be removed at any degree. Its unary cost is a
      // constant and is recovered by final evaluation; each incident edge is
      // propagated exactly into the neighboring unary vector.
      for (const Edge &edge : graph.edges) {
        if (!edge.active || (edge.lhs != *selected && edge.rhs != *selected))
          continue;
        uint32_t neighbor = edge.lhs == *selected ? edge.rhs : edge.lhs;
        if (!active[neighbor])
          continue;
        for (uint32_t state = 0; state < graph.nodes[neighbor].size();
             ++state) {
          if (!budget.consume()) {
            exhausted = true;
            break;
          }
          graph.nodes[neighbor][state] =
              addCost(graph.nodes[neighbor][state],
                      edgeCost(edge, *selected, 0, neighbor, state));
        }
        if (exhausted)
          break;
      }
    } else if (adjacent.empty()) {
      std::optional<uint32_t> state =
          minimumState(graph.nodes[*selected], budget);
      if (!state) {
        exhausted = budget.used >= budget.limit;
        if (!exhausted) {
          result.status = ExactPBQPStatus::NoSolution;
          result.work = budget.used;
          return result;
        }
        break;
      }
      record.choices.push_back(*state);
    } else if (adjacent.size() == 1) {
      uint32_t neighbor = adjacent.front();
      Edge *edge = findEdge(graph, *selected, neighbor);
      if (!edge)
        return result;
      record.choices.resize(graph.nodes[neighbor].size());
      for (uint32_t neighborState = 0;
           neighborState < graph.nodes[neighbor].size(); ++neighborState) {
        std::vector<ExactPBQPCost> costs;
        for (uint32_t state = 0; state < graph.nodes[*selected].size();
             ++state) {
          if (!budget.consume()) {
            exhausted = true;
            break;
          }
          costs.push_back(addCost(
              graph.nodes[*selected][state],
              edgeCost(*edge, *selected, state, neighbor, neighborState)));
        }
        if (exhausted)
          break;
        std::optional<uint32_t> state = minimumState(costs, budget);
        if (!state) {
          if (budget.used >= budget.limit) {
            exhausted = true;
            break;
          }
          record.choices[neighborState] = 0;
          graph.nodes[neighbor][neighborState] = kExactPBQPInfinity;
          continue;
        }
        record.choices[neighborState] = *state;
        graph.nodes[neighbor][neighborState] =
            addCost(graph.nodes[neighbor][neighborState], costs[*state]);
      }
    } else {
      uint32_t lhs = adjacent[0];
      uint32_t rhs = adjacent[1];
      Edge *lhsEdge = findEdge(graph, *selected, lhs);
      Edge *rhsEdge = findEdge(graph, *selected, rhs);
      if (!lhsEdge || !rhsEdge)
        return result;
      const uint32_t lhsStates = graph.nodes[lhs].size();
      const uint32_t rhsStates = graph.nodes[rhs].size();
      record.choices.resize(static_cast<size_t>(lhsStates) * rhsStates);
      std::vector<ExactPBQPCost> fill(record.choices.size());
      for (uint32_t lhsState = 0; lhsState < lhsStates && !exhausted;
           ++lhsState)
        for (uint32_t rhsState = 0; rhsState < rhsStates; ++rhsState) {
          std::vector<ExactPBQPCost> costs;
          for (uint32_t state = 0; state < graph.nodes[*selected].size();
               ++state) {
            if (!budget.consume()) {
              exhausted = true;
              break;
            }
            costs.push_back(addCost(
                graph.nodes[*selected][state],
                addCost(edgeCost(*lhsEdge, *selected, state, lhs, lhsState),
                        edgeCost(*rhsEdge, *selected, state, rhs, rhsState))));
          }
          if (exhausted)
            break;
          std::optional<uint32_t> state = minimumState(costs, budget);
          if (!state) {
            if (budget.used >= budget.limit) {
              exhausted = true;
              break;
            }
            const size_t offset =
                static_cast<size_t>(lhsState) * rhsStates + rhsState;
            record.choices[offset] = 0;
            fill[offset] = kExactPBQPInfinity;
            continue;
          }
          const size_t offset =
              static_cast<size_t>(lhsState) * rhsStates + rhsState;
          record.choices[offset] = *state;
          fill[offset] = costs[*state];
        }
      if (!exhausted) {
        uint32_t edgeLhs = std::min(lhs, rhs);
        uint32_t edgeRhs = std::max(lhs, rhs);
        Edge *neighborEdge = findEdge(graph, edgeLhs, edgeRhs);
        if (!neighborEdge) {
          Edge edge;
          edge.lhs = edgeLhs;
          edge.rhs = edgeRhs;
          edge.lhsStates = graph.nodes[edgeLhs].size();
          edge.rhsStates = graph.nodes[edgeRhs].size();
          edge.costs.assign(
              static_cast<size_t>(edge.lhsStates) * edge.rhsStates, 0);
          graph.edges.push_back(std::move(edge));
          neighborEdge = &graph.edges.back();
        }
        for (uint32_t lhsState = 0; lhsState < lhsStates; ++lhsState)
          for (uint32_t rhsState = 0; rhsState < rhsStates; ++rhsState) {
            const size_t offset =
                static_cast<size_t>(lhsState) * rhsStates + rhsState;
            ExactPBQPCost &target = lhs < rhs
                                        ? neighborEdge->at(lhsState, rhsState)
                                        : neighborEdge->at(rhsState, lhsState);
            target = addCost(target, fill[offset]);
          }
      }
    }
    if (exhausted)
      break;
    records.push_back(std::move(record));
    active[*selected] = false;
    for (Edge &edge : graph.edges)
      if (edge.active && (edge.lhs == *selected || edge.rhs == *selected))
        edge.active = false;
  }

  if (exhausted) {
    result.status = ExactPBQPStatus::Indeterminate;
    result.work = budget.used;
    return result;
  }

  llvm::SmallVector<uint32_t, 16> core;
  for (uint32_t node = 0; node < active.size(); ++node)
    if (active[node])
      core.push_back(node);
  llvm::sort(core, [&](uint32_t lhs, uint32_t rhs) {
    const auto lhsKey = std::tuple(
        graph.nodes[lhs].size(),
        -static_cast<int64_t>(neighbors(graph, active, lhs).size()), lhs);
    const auto rhsKey = std::tuple(
        graph.nodes[rhs].size(),
        -static_cast<int64_t>(neighbors(graph, active, rhs).size()), rhs);
    return lhsKey < rhsKey;
  });

  std::vector<uint32_t> assignment(graph.nodes.size(),
                                   std::numeric_limits<uint32_t>::max());
  std::optional<ExactPBQPCost> bestCost;
  std::vector<uint32_t> bestAssignment;
  auto reconstruct = [&](std::vector<uint32_t> candidate) {
    for (const ReductionRecord &record : llvm::reverse(records)) {
      if (record.fixedState || record.neighbors.empty())
        candidate[record.node] = record.choices.front();
      else if (record.neighbors.size() == 1)
        candidate[record.node] =
            record.choices[candidate[record.neighbors.front()]];
      else {
        const uint32_t rhsStates = graph.nodes[record.neighbors[1]].size();
        candidate[record.node] =
            record.choices[static_cast<size_t>(candidate[record.neighbors[0]]) *
                               rhsStates +
                           candidate[record.neighbors[1]]];
      }
    }
    return candidate;
  };
  std::function<void(size_t, ExactPBQPCost)> search =
      [&](size_t position, ExactPBQPCost partial) {
        if (exhausted)
          return;
        if (position == core.size()) {
          std::vector<uint32_t> complete = reconstruct(assignment);
          if (!bestCost || partial < *bestCost ||
              (partial == *bestCost && complete < bestAssignment)) {
            bestCost = partial;
            bestAssignment = std::move(complete);
          }
          return;
        }
        const uint32_t node = core[position];
        for (uint32_t state = 0; state < graph.nodes[node].size(); ++state) {
          if (!budget.consume()) {
            exhausted = true;
            return;
          }
          ExactPBQPCost next = addCost(partial, graph.nodes[node][state]);
          for (const Edge &edge : graph.edges) {
            if (!edge.active || (edge.lhs != node && edge.rhs != node))
              continue;
            const uint32_t neighbor = edge.lhs == node ? edge.rhs : edge.lhs;
            if (assignment[neighbor] == std::numeric_limits<uint32_t>::max())
              continue;
            next = addCost(next, edgeCost(edge, node, state, neighbor,
                                          assignment[neighbor]));
          }
          if (next == kExactPBQPInfinity || (bestCost && next > *bestCost))
            continue;
          assignment[node] = state;
          search(position + 1, next);
          assignment[node] = std::numeric_limits<uint32_t>::max();
        }
      };
  search(0, 0);
  if (exhausted) {
    result.status = ExactPBQPStatus::Indeterminate;
    result.work = budget.used;
    return result;
  }
  if (!bestCost) {
    result.status = ExactPBQPStatus::NoSolution;
    result.work = budget.used;
    return result;
  }
  assignment = std::move(bestAssignment);
  std::optional<ExactPBQPCost> checked = evaluate(problem, assignment);
  if (!checked)
    return result;
  result.status = ExactPBQPStatus::Optimal;
  result.assignment = std::move(assignment);
  result.cost = *checked;
  result.lowerBound = *checked;
  result.work = budget.used;
  return result;
}

ExactPBQPResult solveExactPBQP(const ExactPBQPProblem &problem,
                               uint64_t workLimit,
                               uint32_t semanticTieVariableCount) {
  const uint64_t variableWork = problem.variables.size();
  const uint64_t factorCount = problem.factors.size();
  if (factorCount > (std::numeric_limits<uint64_t>::max() - variableWork) / 2) {
    ExactPBQPResult exhausted;
    exhausted.status = ExactPBQPStatus::Indeterminate;
    exhausted.work = workLimit;
    return exhausted;
  }
  const uint64_t structuralWork = variableWork + factorCount * 2;
  if (structuralWork > workLimit) {
    ExactPBQPResult exhausted;
    exhausted.status = ExactPBQPStatus::Indeterminate;
    exhausted.work = workLimit;
    return exhausted;
  }
  std::optional<std::vector<std::vector<uint32_t>>> components =
      getConnectedComponents(problem);
  if (!components) {
    ExactPBQPResult broken;
    return broken;
  }
  if (components->size() > 1) {
    ExactPBQPResult combined;
    combined.status = ExactPBQPStatus::Optimal;
    combined.assignment.resize(problem.variables.size());
    combined.cost = 0;
    combined.lowerBound = 0;
    uint64_t used = structuralWork;
    for (const std::vector<uint32_t> &component : *components) {
      if (used >= workLimit) {
        combined.status = ExactPBQPStatus::Indeterminate;
        combined.assignment.clear();
        combined.cost.reset();
        combined.work = used;
        return combined;
      }
      ExactPBQPProblem local;
      local.variables.reserve(component.size());
      llvm::DenseMap<uint32_t, uint32_t> localIndex;
      uint32_t localSemanticVariables = 0;
      for (uint32_t original : component) {
        localIndex.try_emplace(original, local.variables.size());
        local.variables.push_back(problem.variables[original]);
        localSemanticVariables += original < semanticTieVariableCount;
      }
      for (const ExactPBQPBinaryFactor &factor : problem.factors) {
        auto lhs = localIndex.find(factor.lhs);
        if (lhs == localIndex.end())
          continue;
        auto rhs = localIndex.find(factor.rhs);
        if (rhs == localIndex.end()) {
          combined.status = ExactPBQPStatus::BrokenContract;
          combined.assignment.clear();
          combined.cost.reset();
          combined.work = used;
          return combined;
        }
        ExactPBQPBinaryFactor remapped = factor;
        remapped.lhs = lhs->second;
        remapped.rhs = rhs->second;
        local.factors.push_back(std::move(remapped));
      }
      ExactPBQPResult solved =
          solveExactPBQP(local, workLimit - used, localSemanticVariables);
      used += solved.work;
      if (solved.status != ExactPBQPStatus::Optimal || !solved.cost ||
          solved.assignment.size() != component.size()) {
        solved.work = used;
        return solved;
      }
      ExactPBQPCost nextCost = addCost(*combined.cost, *solved.cost);
      ExactPBQPCost nextLowerBound =
          addCost(combined.lowerBound, solved.lowerBound);
      if (nextCost == kExactPBQPInfinity ||
          nextLowerBound == kExactPBQPInfinity) {
        combined.status = ExactPBQPStatus::Indeterminate;
        combined.assignment.clear();
        combined.cost.reset();
        combined.work = used;
        return combined;
      }
      combined.cost = nextCost;
      combined.lowerBound = nextLowerBound;
      for (auto [localVariable, original] : llvm::enumerate(component))
        combined.assignment[original] = solved.assignment[localVariable];
    }
    combined.work = used;
    return combined;
  }

  Budget budget{workLimit, structuralWork};
  ExactPBQPResult optimum = solveNumericExactPBQP(problem, budget);
  if (optimum.status != ExactPBQPStatus::Optimal)
    return optimum;

  // A reduction may have several states with the same primary cost. Keeping
  // only the smallest state at that reduction is not a global tie-break: an
  // earlier variable can have been reduced through the state being selected.
  // Recover the lexicographically first complete optimum by fixing variables
  // in observable order and proving that the remaining PBQP still reaches the
  // already established optimum. All probes share the caller's work budget.
  const ExactPBQPCost targetCost = *optimum.cost;
  ExactPBQPProblem constrained = problem;
  ExactPBQPResult selected = optimum;
  const uint32_t tieVariables = std::min<uint32_t>(
      semanticTieVariableCount, constrained.variables.size());
  for (uint32_t variable = 0; variable < tieVariables; ++variable) {
    const uint32_t stateCount = static_cast<uint32_t>(
        constrained.variables[variable].unaryCosts.size());
    if (selected.assignment.size() != constrained.variables.size() ||
        selected.assignment[variable] >= stateCount) {
      ExactPBQPResult broken;
      broken.work = budget.used;
      return broken;
    }

    // `selected` is already a witness for its current state. Only states that
    // would improve the lexicographic result need another exact solve.
    uint32_t fixedState = selected.assignment[variable];
    for (uint32_t state = 0; state < fixedState; ++state) {
      ExactPBQPProblem trial = constrained;
      for (uint32_t other = 0; other < stateCount; ++other)
        if (other != state)
          trial.variables[variable].unaryCosts[other] = kExactPBQPInfinity;

      ExactPBQPResult candidate = solveNumericExactPBQP(trial, budget);
      if (candidate.status == ExactPBQPStatus::Indeterminate) {
        candidate.work = budget.used;
        return candidate;
      }
      if (candidate.status == ExactPBQPStatus::BrokenContract) {
        candidate.work = budget.used;
        return candidate;
      }
      if (candidate.status != ExactPBQPStatus::Optimal ||
          candidate.cost != targetCost)
        continue;

      constrained = std::move(trial);
      selected = std::move(candidate);
      fixedState = state;
      break;
    }

    // Preserve the established witness without spending another solve on the
    // state it already proves feasible.
    for (uint32_t other = 0; other < stateCount; ++other)
      if (other != fixedState)
        constrained.variables[variable].unaryCosts[other] = kExactPBQPInfinity;
  }

  selected.work = budget.used;
  return selected;
}

} // namespace wafer::compiler::detail
