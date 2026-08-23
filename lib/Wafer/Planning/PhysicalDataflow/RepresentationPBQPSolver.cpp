//===- RepresentationPBQPSolver.cpp - Exact finite PBQP ---------------===//

#include "Wafer/Planning/PhysicalDataflow/RepresentationPBQPSolver.h"

#include "llvm/ADT/ArrayRef.h"
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

RepresentationPBQPCost addCost(RepresentationPBQPCost lhs,
                               RepresentationPBQPCost rhs) {
  if (lhs == kRepresentationPBQPInfinity ||
      rhs == kRepresentationPBQPInfinity ||
      rhs > kRepresentationPBQPInfinity - lhs)
    return kRepresentationPBQPInfinity;
  return lhs + rhs;
}

struct Edge {
  uint32_t lhs = 0;
  uint32_t rhs = 0;
  uint32_t lhsStates = 0;
  uint32_t rhsStates = 0;
  std::vector<RepresentationPBQPCost> costs;
  bool active = true;

  RepresentationPBQPCost &at(uint32_t lhsState, uint32_t rhsState) {
    return costs[lhsState * rhsStates + rhsState];
  }
  RepresentationPBQPCost at(uint32_t lhsState, uint32_t rhsState) const {
    return costs[lhsState * rhsStates + rhsState];
  }
};

struct Graph {
  std::vector<std::vector<RepresentationPBQPCost>> nodes;
  std::vector<Edge> edges;
};

RepresentationPBQPCost edgeCost(const Edge &edge, uint32_t node, uint32_t state,
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

std::optional<uint32_t>
minimumState(llvm::ArrayRef<RepresentationPBQPCost> costs, Budget &budget) {
  std::optional<uint32_t> selected;
  for (auto [state, cost] : llvm::enumerate(costs)) {
    if (!budget.consume())
      return std::nullopt;
    if (cost == kRepresentationPBQPInfinity)
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
};

bool validateAndBuild(const RepresentationPBQPProblem &problem, Graph &graph) {
  if (problem.variables.empty())
    return false;
  graph.nodes.reserve(problem.variables.size());
  for (const RepresentationPBQPVariable &variable : problem.variables) {
    if (variable.unaryCosts.empty())
      return false;
    graph.nodes.push_back(variable.unaryCosts);
  }
  std::set<std::pair<uint32_t, uint32_t>> pairs;
  for (const RepresentationPBQPBinaryFactor &factor : problem.factors) {
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
        RepresentationPBQPCost cost =
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

std::optional<RepresentationPBQPCost>
evaluate(const RepresentationPBQPProblem &problem,
         llvm::ArrayRef<uint32_t> assignment) {
  if (assignment.size() != problem.variables.size())
    return std::nullopt;
  RepresentationPBQPCost result = 0;
  for (auto [index, state] : llvm::enumerate(assignment)) {
    if (state >= problem.variables[index].unaryCosts.size())
      return std::nullopt;
    result = addCost(result, problem.variables[index].unaryCosts[state]);
  }
  for (const RepresentationPBQPBinaryFactor &factor : problem.factors)
    result = addCost(result,
                     factor.costs[static_cast<size_t>(assignment[factor.lhs]) *
                                      factor.rhsStates +
                                  assignment[factor.rhs]]);
  return result == kRepresentationPBQPInfinity
             ? std::nullopt
             : std::optional<RepresentationPBQPCost>(result);
}

} // namespace

RepresentationPBQPResult
solveRepresentationPBQP(const RepresentationPBQPProblem &problem,
                        uint64_t workLimit) {
  RepresentationPBQPResult result;
  Graph graph;
  if (workLimit == 0 || !validateAndBuild(problem, graph))
    return result;
  Budget budget{workLimit, 0};
  llvm::SmallVector<bool, 16> active(graph.nodes.size(), true);
  llvm::SmallVector<ReductionRecord, 16> records;
  bool exhausted = false;

  while (true) {
    std::optional<uint32_t> selected;
    llvm::SmallVector<uint32_t, 2> adjacent;
    for (uint32_t node = 0; node < graph.nodes.size(); ++node) {
      if (!active[node])
        continue;
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
    if (adjacent.empty()) {
      std::optional<uint32_t> state =
          minimumState(graph.nodes[*selected], budget);
      if (!state) {
        exhausted = budget.used >= budget.limit;
        if (!exhausted) {
          result.status = RepresentationPBQPStatus::NoSolution;
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
        std::vector<RepresentationPBQPCost> costs;
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
          graph.nodes[neighbor][neighborState] = kRepresentationPBQPInfinity;
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
      std::vector<RepresentationPBQPCost> fill(record.choices.size());
      for (uint32_t lhsState = 0; lhsState < lhsStates && !exhausted;
           ++lhsState)
        for (uint32_t rhsState = 0; rhsState < rhsStates; ++rhsState) {
          std::vector<RepresentationPBQPCost> costs;
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
            fill[offset] = kRepresentationPBQPInfinity;
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
            RepresentationPBQPCost &target =
                lhs < rhs ? neighborEdge->at(lhsState, rhsState)
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
    result.status = RepresentationPBQPStatus::Indeterminate;
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
  std::optional<RepresentationPBQPCost> bestCost;
  std::vector<uint32_t> bestAssignment;
  std::function<void(size_t, RepresentationPBQPCost)> search =
      [&](size_t position, RepresentationPBQPCost partial) {
        if (exhausted)
          return;
        if (position == core.size()) {
          if (!bestCost || partial < *bestCost ||
              (partial == *bestCost && assignment < bestAssignment)) {
            bestCost = partial;
            bestAssignment = assignment;
          }
          return;
        }
        const uint32_t node = core[position];
        for (uint32_t state = 0; state < graph.nodes[node].size(); ++state) {
          if (!budget.consume()) {
            exhausted = true;
            return;
          }
          RepresentationPBQPCost next =
              addCost(partial, graph.nodes[node][state]);
          for (const Edge &edge : graph.edges) {
            if (!edge.active || (edge.lhs != node && edge.rhs != node))
              continue;
            const uint32_t neighbor = edge.lhs == node ? edge.rhs : edge.lhs;
            if (assignment[neighbor] == std::numeric_limits<uint32_t>::max())
              continue;
            next = addCost(next, edgeCost(edge, node, state, neighbor,
                                          assignment[neighbor]));
          }
          if (next == kRepresentationPBQPInfinity ||
              (bestCost && next > *bestCost))
            continue;
          assignment[node] = state;
          search(position + 1, next);
          assignment[node] = std::numeric_limits<uint32_t>::max();
        }
      };
  search(0, 0);
  if (exhausted) {
    result.status = RepresentationPBQPStatus::Indeterminate;
    result.work = budget.used;
    return result;
  }
  if (!bestCost) {
    result.status = RepresentationPBQPStatus::NoSolution;
    result.work = budget.used;
    return result;
  }
  assignment = std::move(bestAssignment);
  for (const ReductionRecord &record : llvm::reverse(records)) {
    if (record.neighbors.empty())
      assignment[record.node] = record.choices.front();
    else if (record.neighbors.size() == 1)
      assignment[record.node] =
          record.choices[assignment[record.neighbors.front()]];
    else {
      const uint32_t rhsStates = graph.nodes[record.neighbors[1]].size();
      assignment[record.node] =
          record.choices[static_cast<size_t>(assignment[record.neighbors[0]]) *
                             rhsStates +
                         assignment[record.neighbors[1]]];
    }
  }
  std::optional<RepresentationPBQPCost> checked = evaluate(problem, assignment);
  if (!checked)
    return result;
  result.status = RepresentationPBQPStatus::Optimal;
  result.assignment = std::move(assignment);
  result.cost = *checked;
  result.lowerBound = *checked;
  result.work = budget.used;
  return result;
}

} // namespace wafer::compiler::detail
