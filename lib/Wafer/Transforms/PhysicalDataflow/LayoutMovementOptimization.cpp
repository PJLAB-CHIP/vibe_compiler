//===- LayoutMovementOptimization.cpp - Typed physical layout proposals --===//

#include "Wafer/Transforms/PhysicalDataflow.h"

#include "Wafer/Analysis/PhysicalDataflow/TransferRealizability.h"
#include "Wafer/IR/WaferDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace wafer {
namespace {

constexpr uint64_t kSolverWorkBudget = UINT64_C(1048576);
constexpr uint64_t kSearchExpansionBudget = UINT64_C(4096);
constexpr unsigned kMaximumProposals = 4;

struct ProposalCost {
  uint64_t movementBytes = 0;
  uint64_t unknownCommandCount = 0;
  uint64_t knownCommandLowerBound = 0;
  uint64_t extraFootprintBytes = 0;
  bool infinite = false;

  static ProposalCost infinity() {
    ProposalCost result;
    result.infinite = true;
    return result;
  }
};

static bool rawLess(const ProposalCost &lhs, const ProposalCost &rhs) {
  if (lhs.infinite != rhs.infinite)
    return !lhs.infinite;
  if (lhs.infinite)
    return false;
  return std::tie(lhs.movementBytes, lhs.unknownCommandCount,
                  lhs.knownCommandLowerBound, lhs.extraFootprintBytes) <
         std::tie(rhs.movementBytes, rhs.unknownCommandCount,
                  rhs.knownCommandLowerBound, rhs.extraFootprintBytes);
}

static ProposalCost rawAdd(ProposalCost lhs, ProposalCost rhs) {
  if (lhs.infinite || rhs.infinite)
    return ProposalCost::infinity();
  auto add = [](uint64_t lhsValue, uint64_t rhsValue, uint64_t &result) {
    if (rhsValue > std::numeric_limits<uint64_t>::max() - lhsValue)
      return false;
    result = lhsValue + rhsValue;
    return true;
  };
  ProposalCost result;
  if (!add(lhs.movementBytes, rhs.movementBytes, result.movementBytes) ||
      !add(lhs.unknownCommandCount, rhs.unknownCommandCount,
           result.unknownCommandCount) ||
      !add(lhs.knownCommandLowerBound, rhs.knownCommandLowerBound,
           result.knownCommandLowerBound) ||
      !add(lhs.extraFootprintBytes, rhs.extraFootprintBytes,
           result.extraFootprintBytes))
    return ProposalCost::infinity();
  return result;
}

struct SolverBudget {
  uint64_t workRemaining = kSolverWorkBudget;
  uint64_t expansionsRemaining = kSearchExpansionBudget;

  bool consumeWork(uint64_t amount = 1) {
    if (amount > workRemaining)
      return false;
    workRemaining -= amount;
    return true;
  }

  bool consumeExpansion() {
    if (expansionsRemaining == 0)
      return false;
    --expansionsRemaining;
    return true;
  }
};

static std::optional<ProposalCost>
checkedAdd(ProposalCost lhs, ProposalCost rhs, SolverBudget &budget) {
  if (!budget.consumeWork())
    return std::nullopt;
  if (lhs.infinite || rhs.infinite)
    return ProposalCost::infinity();
  auto add = [](uint64_t lhsValue,
                uint64_t rhsValue) -> std::optional<uint64_t> {
    if (rhsValue > std::numeric_limits<uint64_t>::max() - lhsValue)
      return std::nullopt;
    return lhsValue + rhsValue;
  };
  std::optional<uint64_t> movement = add(lhs.movementBytes, rhs.movementBytes);
  std::optional<uint64_t> unknown =
      add(lhs.unknownCommandCount, rhs.unknownCommandCount);
  std::optional<uint64_t> known =
      add(lhs.knownCommandLowerBound, rhs.knownCommandLowerBound);
  std::optional<uint64_t> footprint =
      add(lhs.extraFootprintBytes, rhs.extraFootprintBytes);
  if (!movement || !unknown || !known || !footprint)
    return ProposalCost::infinity();
  return ProposalCost{*movement, *unknown, *known, *footprint, false};
}

static std::optional<bool> checkedLess(const ProposalCost &lhs,
                                       const ProposalCost &rhs,
                                       SolverBudget &budget) {
  if (!budget.consumeWork())
    return std::nullopt;
  return rawLess(lhs, rhs);
}

struct PBQPNode {
  mlir::Operation *operation = nullptr;
  llvm::SmallVector<MemLayout, 4> states;
  llvm::SmallVector<ProposalCost, 4> unary;
};

struct PBQPEdge {
  unsigned lhs = 0;
  unsigned rhs = 0;
  unsigned lhsStates = 0;
  unsigned rhsStates = 0;
  std::vector<ProposalCost> matrix;
  bool active = true;

  ProposalCost &at(unsigned lhsState, unsigned rhsState) {
    return matrix[lhsState * rhsStates + rhsState];
  }
  const ProposalCost &at(unsigned lhsState, unsigned rhsState) const {
    return matrix[lhsState * rhsStates + rhsState];
  }
};

struct PBQPGraph {
  llvm::SmallVector<PBQPNode, 16> nodes;
  llvm::SmallVector<PBQPEdge, 24> edges;
};

struct DomainConstraints {
  std::vector<uint8_t> allowedMasks;

  friend bool operator<(const DomainConstraints &lhs,
                        const DomainConstraints &rhs) {
    return lhs.allowedMasks < rhs.allowedMasks;
  }
};

struct PBQPSolution {
  ProposalCost cost;
  std::vector<unsigned> states;
};

static bool solutionLess(const PBQPSolution &lhs, const PBQPSolution &rhs) {
  if (rawLess(lhs.cost, rhs.cost))
    return true;
  if (rawLess(rhs.cost, lhs.cost))
    return false;
  return lhs.states < rhs.states;
}

static PBQPEdge *findActiveEdge(PBQPGraph &graph, unsigned lhs, unsigned rhs) {
  if (lhs > rhs)
    std::swap(lhs, rhs);
  for (PBQPEdge &edge : graph.edges)
    if (edge.active && edge.lhs == lhs && edge.rhs == rhs)
      return &edge;
  return nullptr;
}

static const PBQPEdge *findEdge(const PBQPGraph &graph, unsigned lhs,
                                unsigned rhs) {
  bool swapped = lhs > rhs;
  if (swapped)
    std::swap(lhs, rhs);
  for (const PBQPEdge &edge : graph.edges)
    if (edge.lhs == lhs && edge.rhs == rhs)
      return &edge;
  return nullptr;
}

static ProposalCost getEdgeCost(const PBQPEdge &edge, unsigned lhsNode,
                                unsigned lhsState, unsigned rhsNode,
                                unsigned rhsState) {
  if (edge.lhs == lhsNode && edge.rhs == rhsNode)
    return edge.at(lhsState, rhsState);
  return edge.at(rhsState, lhsState);
}

struct EliminationRecord {
  unsigned node = 0;
  llvm::SmallVector<unsigned, 2> neighbors;
  std::vector<unsigned> choices;
};

static llvm::SmallVector<unsigned, 4>
getActiveNeighbors(const PBQPGraph &graph, llvm::ArrayRef<bool> active,
                   unsigned node) {
  llvm::SmallVector<unsigned, 4> neighbors;
  for (const PBQPEdge &edge : graph.edges) {
    if (!edge.active)
      continue;
    if (edge.lhs == node && active[edge.rhs])
      neighbors.push_back(edge.rhs);
    else if (edge.rhs == node && active[edge.lhs])
      neighbors.push_back(edge.lhs);
  }
  llvm::sort(neighbors);
  return neighbors;
}

static std::optional<unsigned>
chooseBestState(llvm::ArrayRef<ProposalCost> costs, SolverBudget &budget) {
  std::optional<unsigned> best;
  for (unsigned state = 0; state < costs.size(); ++state) {
    if (costs[state].infinite)
      continue;
    if (!best) {
      best = state;
      continue;
    }
    std::optional<bool> less = checkedLess(costs[state], costs[*best], budget);
    if (!less)
      return std::nullopt;
    if (*less)
      best = state;
  }
  return best;
}

static bool applyDominancePruning(PBQPGraph &graph, SolverBudget &budget) {
  for (unsigned nodeIndex = 0; nodeIndex < graph.nodes.size(); ++nodeIndex) {
    PBQPNode &node = graph.nodes[nodeIndex];
    for (unsigned dominated = 0; dominated < node.states.size(); ++dominated) {
      if (node.unary[dominated].infinite)
        continue;
      for (unsigned candidate = 0; candidate < node.states.size();
           ++candidate) {
        if (candidate == dominated || node.unary[candidate].infinite)
          continue;
        uint64_t comparisonCount = 1;
        for (const PBQPEdge &edge : graph.edges) {
          if (edge.lhs == nodeIndex)
            comparisonCount += edge.rhsStates;
          else if (edge.rhs == nodeIndex)
            comparisonCount += edge.lhsStates;
        }
        if (!budget.consumeWork(comparisonCount))
          return false;

        bool noWorse = !rawLess(node.unary[dominated], node.unary[candidate]);
        bool strictlyBetter =
            rawLess(node.unary[candidate], node.unary[dominated]);
        if (!noWorse)
          continue;
        for (const PBQPEdge &edge : graph.edges) {
          if (edge.lhs != nodeIndex && edge.rhs != nodeIndex)
            continue;
          unsigned neighbor = edge.lhs == nodeIndex ? edge.rhs : edge.lhs;
          for (unsigned neighborState = 0;
               neighborState < graph.nodes[neighbor].states.size();
               ++neighborState) {
            ProposalCost candidateCost = getEdgeCost(edge, nodeIndex, candidate,
                                                     neighbor, neighborState);
            ProposalCost dominatedCost = getEdgeCost(edge, nodeIndex, dominated,
                                                     neighbor, neighborState);
            if (rawLess(dominatedCost, candidateCost)) {
              noWorse = false;
              break;
            }
            strictlyBetter |= rawLess(candidateCost, dominatedCost);
          }
          if (!noWorse)
            break;
        }
        if (noWorse && strictlyBetter) {
          node.unary[dominated] = ProposalCost::infinity();
          break;
        }
      }
    }
    if (llvm::none_of(node.unary,
                      [](const ProposalCost &cost) { return !cost.infinite; }))
      return false;
  }
  return true;
}

static std::optional<ProposalCost>
evaluateAssignment(const PBQPGraph &graph, llvm::ArrayRef<unsigned> assignment,
                   SolverBudget &budget) {
  if (assignment.size() != graph.nodes.size())
    return std::nullopt;
  uint64_t additionCount = graph.nodes.size() + graph.edges.size();
  if (!budget.consumeWork(additionCount))
    return std::nullopt;
  ProposalCost total;
  for (auto [node, state] : llvm::enumerate(assignment)) {
    if (state >= graph.nodes[node].unary.size())
      return std::nullopt;
    total = rawAdd(total, graph.nodes[node].unary[state]);
  }
  for (const PBQPEdge &edge : graph.edges)
    total = rawAdd(total, edge.at(assignment[edge.lhs], assignment[edge.rhs]));
  if (total.infinite)
    return std::nullopt;
  return total;
}

static std::optional<PBQPSolution>
solvePBQP(const PBQPGraph &input, const DomainConstraints &constraints,
          SolverBudget &budget) {
  PBQPGraph graph = input;
  if (constraints.allowedMasks.size() != graph.nodes.size())
    return std::nullopt;
  for (auto [nodeIndex, node] : llvm::enumerate(graph.nodes)) {
    uint8_t mask = constraints.allowedMasks[nodeIndex];
    for (unsigned state = 0; state < node.states.size(); ++state)
      if ((mask & (uint8_t{1} << state)) == 0)
        node.unary[state] = ProposalCost::infinity();
  }
  if (!applyDominancePruning(graph, budget))
    return std::nullopt;

  llvm::SmallVector<bool, 16> active(graph.nodes.size(), true);
  llvm::SmallVector<EliminationRecord, 16> records;
  while (true) {
    std::optional<unsigned> selected;
    llvm::SmallVector<unsigned, 4> selectedNeighbors;
    for (unsigned node = 0; node < graph.nodes.size(); ++node) {
      if (!active[node])
        continue;
      llvm::SmallVector<unsigned, 4> neighbors =
          getActiveNeighbors(graph, active, node);
      if (neighbors.size() <= 2) {
        selected = node;
        selectedNeighbors = std::move(neighbors);
        break;
      }
    }
    if (!selected)
      break;

    unsigned node = *selected;
    EliminationRecord record;
    record.node = node;
    record.neighbors.assign(selectedNeighbors.begin(), selectedNeighbors.end());
    if (selectedNeighbors.empty()) {
      std::optional<unsigned> best =
          chooseBestState(graph.nodes[node].unary, budget);
      if (!best)
        return std::nullopt;
      record.choices.push_back(*best);
    } else if (selectedNeighbors.size() == 1) {
      unsigned neighbor = selectedNeighbors.front();
      PBQPEdge *edge = findActiveEdge(graph, node, neighbor);
      if (!edge)
        return std::nullopt;
      record.choices.resize(graph.nodes[neighbor].states.size());
      for (unsigned neighborState = 0;
           neighborState < graph.nodes[neighbor].states.size();
           ++neighborState) {
        llvm::SmallVector<ProposalCost, 4> candidates;
        for (unsigned state = 0; state < graph.nodes[node].states.size();
             ++state) {
          std::optional<ProposalCost> combined = checkedAdd(
              graph.nodes[node].unary[state],
              getEdgeCost(*edge, node, state, neighbor, neighborState), budget);
          if (!combined)
            return std::nullopt;
          candidates.push_back(*combined);
        }
        std::optional<unsigned> best = chooseBestState(candidates, budget);
        if (!best)
          return std::nullopt;
        record.choices[neighborState] = *best;
        std::optional<ProposalCost> updated =
            checkedAdd(graph.nodes[neighbor].unary[neighborState],
                       candidates[*best], budget);
        if (!updated)
          return std::nullopt;
        graph.nodes[neighbor].unary[neighborState] = *updated;
      }
    } else {
      unsigned lhs = selectedNeighbors[0];
      unsigned rhs = selectedNeighbors[1];
      PBQPEdge *lhsEdge = findActiveEdge(graph, node, lhs);
      PBQPEdge *rhsEdge = findActiveEdge(graph, node, rhs);
      if (!lhsEdge || !rhsEdge)
        return std::nullopt;
      unsigned lhsStateCount = graph.nodes[lhs].states.size();
      unsigned rhsStateCount = graph.nodes[rhs].states.size();
      record.choices.resize(lhsStateCount * rhsStateCount);
      std::vector<ProposalCost> fill(lhsStateCount * rhsStateCount);
      for (unsigned lhsState = 0; lhsState < lhsStateCount; ++lhsState) {
        for (unsigned rhsState = 0; rhsState < rhsStateCount; ++rhsState) {
          llvm::SmallVector<ProposalCost, 4> candidates;
          for (unsigned state = 0; state < graph.nodes[node].states.size();
               ++state) {
            std::optional<ProposalCost> lhsCost = checkedAdd(
                graph.nodes[node].unary[state],
                getEdgeCost(*lhsEdge, node, state, lhs, lhsState), budget);
            if (!lhsCost)
              return std::nullopt;
            std::optional<ProposalCost> total = checkedAdd(
                *lhsCost, getEdgeCost(*rhsEdge, node, state, rhs, rhsState),
                budget);
            if (!total)
              return std::nullopt;
            candidates.push_back(*total);
          }
          std::optional<unsigned> best = chooseBestState(candidates, budget);
          if (!best)
            return std::nullopt;
          unsigned offset = lhsState * rhsStateCount + rhsState;
          record.choices[offset] = *best;
          fill[offset] = candidates[*best];
        }
      }

      unsigned edgeLhs = std::min(lhs, rhs);
      unsigned edgeRhs = std::max(lhs, rhs);
      PBQPEdge *neighborEdge = findActiveEdge(graph, edgeLhs, edgeRhs);
      if (!neighborEdge) {
        PBQPEdge edge;
        edge.lhs = edgeLhs;
        edge.rhs = edgeRhs;
        edge.lhsStates = graph.nodes[edgeLhs].states.size();
        edge.rhsStates = graph.nodes[edgeRhs].states.size();
        edge.matrix.resize(edge.lhsStates * edge.rhsStates);
        graph.edges.push_back(std::move(edge));
        neighborEdge = &graph.edges.back();
      }
      for (unsigned lhsState = 0; lhsState < lhsStateCount; ++lhsState) {
        for (unsigned rhsState = 0; rhsState < rhsStateCount; ++rhsState) {
          unsigned fillOffset = lhsState * rhsStateCount + rhsState;
          ProposalCost &target = lhs < rhs
                                     ? neighborEdge->at(lhsState, rhsState)
                                     : neighborEdge->at(rhsState, lhsState);
          std::optional<ProposalCost> updated =
              checkedAdd(target, fill[fillOffset], budget);
          if (!updated)
            return std::nullopt;
          target = *updated;
        }
      }
    }

    records.push_back(std::move(record));
    active[node] = false;
    for (PBQPEdge &edge : graph.edges)
      if (edge.active && (edge.lhs == node || edge.rhs == node))
        edge.active = false;
  }

  llvm::SmallVector<unsigned, 16> core;
  for (unsigned node = 0; node < active.size(); ++node)
    if (active[node])
      core.push_back(node);
  llvm::stable_sort(core, [&](unsigned lhs, unsigned rhs) {
    size_t lhsDegree = getActiveNeighbors(graph, active, lhs).size();
    size_t rhsDegree = getActiveNeighbors(graph, active, rhs).size();
    return lhsDegree != rhsDegree ? lhsDegree > rhsDegree : lhs < rhs;
  });

  std::vector<unsigned> assignment(graph.nodes.size(),
                                   std::numeric_limits<unsigned>::max());
  std::optional<ProposalCost> bestCoreCost;
  std::vector<unsigned> bestCoreAssignment;
  std::function<bool(unsigned, ProposalCost)> search =
      [&](unsigned position, ProposalCost partial) -> bool {
    if (position == core.size()) {
      if (!bestCoreCost) {
        bestCoreCost = partial;
        bestCoreAssignment = assignment;
        return true;
      }
      std::optional<bool> less = checkedLess(partial, *bestCoreCost, budget);
      if (!less)
        return false;
      if (*less || (!rawLess(*bestCoreCost, partial) &&
                    assignment < bestCoreAssignment)) {
        bestCoreCost = partial;
        bestCoreAssignment = assignment;
      }
      return true;
    }
    unsigned node = core[position];
    for (unsigned state = 0; state < graph.nodes[node].states.size(); ++state) {
      if (!budget.consumeExpansion())
        return static_cast<bool>(bestCoreCost);
      ProposalCost cost = graph.nodes[node].unary[state];
      for (const PBQPEdge &edge : graph.edges) {
        if (!edge.active || (edge.lhs != node && edge.rhs != node))
          continue;
        unsigned neighbor = edge.lhs == node ? edge.rhs : edge.lhs;
        if (assignment[neighbor] == std::numeric_limits<unsigned>::max())
          continue;
        std::optional<ProposalCost> updated = checkedAdd(
            cost,
            getEdgeCost(edge, node, state, neighbor, assignment[neighbor]),
            budget);
        if (!updated)
          return false;
        cost = *updated;
      }
      std::optional<ProposalCost> updated = checkedAdd(partial, cost, budget);
      if (!updated)
        return false;
      assignment[node] = state;
      if (!updated->infinite && !search(position + 1, *updated))
        return false;
      assignment[node] = std::numeric_limits<unsigned>::max();
    }
    return true;
  };

  if (core.empty()) {
    bestCoreCost = ProposalCost{};
    bestCoreAssignment = assignment;
  } else if (!search(0, ProposalCost{}) || !bestCoreCost) {
    return std::nullopt;
  }
  assignment = std::move(bestCoreAssignment);

  for (const EliminationRecord &record : llvm::reverse(records)) {
    if (record.neighbors.empty()) {
      assignment[record.node] = record.choices.front();
    } else if (record.neighbors.size() == 1) {
      unsigned neighborState = assignment[record.neighbors.front()];
      assignment[record.node] = record.choices[neighborState];
    } else {
      unsigned lhsState = assignment[record.neighbors[0]];
      unsigned rhsState = assignment[record.neighbors[1]];
      unsigned rhsCount = graph.nodes[record.neighbors[1]].states.size();
      assignment[record.node] = record.choices[lhsState * rhsCount + rhsState];
    }
  }

  std::optional<ProposalCost> incumbent =
      evaluateAssignment(input, assignment, budget);
  if (!incumbent)
    return std::nullopt;

  auto tryLocalAssignment = [&](std::vector<unsigned> candidate) {
    if (!budget.consumeExpansion())
      return false;
    std::optional<ProposalCost> candidateCost =
        evaluateAssignment(input, candidate, budget);
    if (!candidateCost)
      return false;
    std::optional<bool> less = checkedLess(*candidateCost, *incumbent, budget);
    if (!less)
      return false;
    if (*less ||
        (!rawLess(*incumbent, *candidateCost) && candidate < assignment)) {
      assignment = std::move(candidate);
      incumbent = *candidateCost;
    }
    return true;
  };
  bool continueLocalSearch = true;
  for (unsigned node = 0; continueLocalSearch && node < input.nodes.size();
       ++node) {
    for (unsigned state = 0; state < input.nodes[node].states.size(); ++state) {
      if ((constraints.allowedMasks[node] & (uint8_t{1} << state)) == 0 ||
          state == assignment[node])
        continue;
      std::vector<unsigned> candidate = assignment;
      candidate[node] = state;
      if (!tryLocalAssignment(std::move(candidate))) {
        continueLocalSearch = false;
        break;
      }
    }
  }
  for (unsigned lhs = 0; continueLocalSearch && lhs < input.nodes.size();
       ++lhs) {
    for (unsigned rhs = lhs + 1;
         continueLocalSearch && rhs < input.nodes.size(); ++rhs) {
      for (unsigned lhsState = 0;
           continueLocalSearch && lhsState < input.nodes[lhs].states.size();
           ++lhsState) {
        if ((constraints.allowedMasks[lhs] & (uint8_t{1} << lhsState)) == 0 ||
            lhsState == assignment[lhs])
          continue;
        for (unsigned rhsState = 0; rhsState < input.nodes[rhs].states.size();
             ++rhsState) {
          if ((constraints.allowedMasks[rhs] & (uint8_t{1} << rhsState)) == 0 ||
              rhsState == assignment[rhs])
            continue;
          std::vector<unsigned> candidate = assignment;
          candidate[lhs] = lhsState;
          candidate[rhs] = rhsState;
          if (!tryLocalAssignment(std::move(candidate))) {
            continueLocalSearch = false;
            break;
          }
        }
      }
    }
  }
  return PBQPSolution{*incumbent, std::move(assignment)};
}

struct HeapEntry {
  DomainConstraints constraints;
  PBQPSolution solution;
};

static bool heapEntryLess(const HeapEntry &lhs, const HeapEntry &rhs) {
  if (solutionLess(lhs.solution, rhs.solution))
    return true;
  if (solutionLess(rhs.solution, lhs.solution))
    return false;
  return lhs.constraints.allowedMasks < rhs.constraints.allowedMasks;
}

static bool heapPush(std::vector<HeapEntry> &heap, HeapEntry entry,
                     SolverBudget &budget) {
  // Sift on a disposable copy so exhausting the shared comparison budget can
  // never leave a partially ordered heap behind. The heap is deliberately
  // small (it only feeds four proposals), so this also keeps the failure
  // boundary straightforward and auditable.
  std::vector<HeapEntry> candidate = heap;
  SolverBudget candidateBudget = budget;
  candidate.push_back(std::move(entry));
  size_t index = candidate.size() - 1;
  while (index > 0) {
    size_t parent = (index - 1) / 2;
    if (!candidateBudget.consumeWork())
      return false;
    if (!heapEntryLess(candidate[index], candidate[parent]))
      break;
    std::swap(candidate[index], candidate[parent]);
    index = parent;
  }
  heap = std::move(candidate);
  budget = candidateBudget;
  return true;
}

static std::optional<HeapEntry> heapPop(std::vector<HeapEntry> &heap,
                                        SolverBudget &budget) {
  if (heap.empty())
    return std::nullopt;
  std::vector<HeapEntry> candidate = heap;
  SolverBudget candidateBudget = budget;
  HeapEntry result = std::move(candidate.front());
  if (candidate.size() == 1) {
    candidate.pop_back();
    heap = std::move(candidate);
    return result;
  }
  candidate.front() = std::move(candidate.back());
  candidate.pop_back();
  size_t index = 0;
  while (true) {
    size_t lhs = index * 2 + 1;
    if (lhs >= candidate.size())
      break;
    size_t rhs = lhs + 1;
    size_t child = lhs;
    if (rhs < candidate.size()) {
      if (!candidateBudget.consumeWork())
        return std::nullopt;
      if (heapEntryLess(candidate[rhs], candidate[lhs]))
        child = rhs;
    }
    if (!candidateBudget.consumeWork())
      return std::nullopt;
    if (!heapEntryLess(candidate[child], candidate[index]))
      break;
    std::swap(candidate[index], candidate[child]);
    index = child;
  }
  heap = std::move(candidate);
  budget = candidateBudget;
  return result;
}

static llvm::SmallVector<PBQPSolution, kMaximumProposals>
solveTopProposals(const PBQPGraph &graph) {
  llvm::SmallVector<PBQPSolution, kMaximumProposals> proposals;
  if (graph.nodes.empty())
    return proposals;
  SolverBudget budget;
  DomainConstraints root;
  root.allowedMasks.reserve(graph.nodes.size());
  for (const PBQPNode &node : graph.nodes)
    root.allowedMasks.push_back(
        static_cast<uint8_t>((uint16_t{1} << node.states.size()) - 1));
  std::optional<PBQPSolution> rootSolution = solvePBQP(graph, root, budget);
  if (!rootSolution)
    return proposals;

  std::vector<HeapEntry> heap;
  if (!heapPush(heap, HeapEntry{root, std::move(*rootSolution)}, budget))
    return proposals;
  std::set<DomainConstraints> seenConstraints;
  seenConstraints.insert(root);
  std::set<std::vector<unsigned>> seenSolutions;
  while (proposals.size() < kMaximumProposals) {
    std::optional<HeapEntry> entry = heapPop(heap, budget);
    if (!entry)
      break;
    if (seenSolutions.insert(entry->solution.states).second)
      proposals.push_back(entry->solution);

    for (unsigned position = 0; position < graph.nodes.size(); ++position) {
      DomainConstraints child = entry->constraints;
      bool valid = true;
      for (unsigned prefix = 0; prefix < position; ++prefix)
        child.allowedMasks[prefix] =
            static_cast<uint8_t>(uint8_t{1} << entry->solution.states[prefix]);
      child.allowedMasks[position] &= static_cast<uint8_t>(
          ~(uint8_t{1} << entry->solution.states[position]));
      if (child.allowedMasks[position] == 0)
        valid = false;
      if (!valid || !seenConstraints.insert(child).second)
        continue;
      std::optional<PBQPSolution> solution = solvePBQP(graph, child, budget);
      if (solution &&
          !heapPush(heap, HeapEntry{std::move(child), std::move(*solution)},
                    budget))
        return proposals;
    }
  }
  return proposals;
}

static mlir::MemRefType withLayout(mlir::MemRefType type, MemLayout layout) {
  MemoryAttr memory = getWaferMemoryAttr(type);
  if (!memory)
    return {};
  return mlir::MemRefType::get(
      type.getShape(), type.getElementType(), type.getLayout(),
      MemoryAttr::get(type.getContext(), memory.getSpace(), layout));
}

static std::optional<MemLayout> getLayout(mlir::Type type) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  MemoryAttr memory =
      memrefType ? getWaferMemoryAttr(memrefType) : MemoryAttr{};
  if (!memory)
    return std::nullopt;
  return memory.getLayout();
}

static std::optional<MemorySpace> getMemorySpace(mlir::Type type) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  MemoryAttr memory =
      memrefType ? getWaferMemoryAttr(memrefType) : MemoryAttr{};
  if (!memory)
    return std::nullopt;
  return memory.getSpace();
}

static mlir::Value stripLayoutMaterializations(mlir::Value value) {
  while (auto materialize = value.getDefiningOp<LayoutMaterializeOp>())
    value = materialize.getSource();
  return value;
}

static std::optional<ProposalCost>
getMaterializationCost(mlir::MemRefType sourceType, mlir::MemRefType destType) {
  if (!sourceType || !destType ||
      sourceType.getShape() != destType.getShape() ||
      sourceType.getElementType() != destType.getElementType())
    return std::nullopt;
  if (getMemorySpace(sourceType) != getMemorySpace(destType))
    return std::nullopt;
  if (getLayout(sourceType) == getLayout(destType))
    return ProposalCost{};
  std::optional<WaferPhysicalTensorInfo> sourceInfo =
      computeWaferPhysicalTensorInfo(sourceType);
  std::optional<WaferPhysicalTensorInfo> destInfo =
      computeWaferPhysicalTensorInfo(destType);
  analysis::IndexRelationResult identity =
      analysis::IndexRelation::identity(destType.getShape());
  if (!sourceInfo || !destInfo || sourceInfo->physicalBytes <= 0 ||
      destInfo->physicalBytes <= 0 || sourceInfo->elementBytes <= 0 ||
      sourceInfo->elementBytes != destInfo->elementBytes ||
      sourceInfo->bitPackedElement || destInfo->bitPackedElement ||
      !identity.isExact() ||
      mlir::failed(analysis::TransferRealizability::proveMappedTransfer(
          sourceType, destType, destType.getShape(), *identity.get(),
          *identity.get())))
    return std::nullopt;
  return ProposalCost{static_cast<uint64_t>(destInfo->physicalBytes), 0, 1,
                      static_cast<uint64_t>(destInfo->physicalBytes), false};
}

static bool isEligiblePointwiseCompute(mlir::Operation *operation) {
  if (auto elementwise = mlir::dyn_cast<ComputeElementwiseOp>(operation))
    return !elementwise.getIndexingMapsAttr() &&
           elementwise->getNumResults() == 1;
  return mlir::isa<ComputeConvertOp>(operation) &&
         operation->getNumResults() == 1;
}

static bool isLayoutAssignableOperation(mlir::Operation *operation) {
  return isEligiblePointwiseCompute(operation) ||
         mlir::isa<CommAllReduceOp>(operation);
}

static bool isLayoutStateLegal(mlir::Operation *operation, MemLayout layout) {
  if (!isLayoutAssignableOperation(operation))
    return false;
  auto resultType =
      mlir::dyn_cast<mlir::MemRefType>(operation->getResult(0).getType());
  resultType = resultType ? withLayout(resultType, layout) : mlir::MemRefType{};
  if (!resultType || !resultType.hasStaticShape())
    return false;
  if (mlir::isa<CommAllReduceOp>(operation)) {
    std::optional<WaferPhysicalTensorInfo> physical =
        computeWaferPhysicalTensorInfo(resultType);
    if (!physical || physical->physicalBytes <= 0 || physical->bitPackedElement)
      return false;
  }
  analysis::IndexRelationResult identity =
      analysis::IndexRelation::identity(resultType.getShape());
  if (!identity.isExact())
    return false;
  for (mlir::Value operand : operation->getOperands()) {
    auto operandType = mlir::dyn_cast<mlir::MemRefType>(operand.getType());
    operandType =
        operandType ? withLayout(operandType, layout) : mlir::MemRefType{};
    if (!operandType || operandType.getShape() != resultType.getShape() ||
        mlir::failed(analysis::TransferRealizability::provePhysicalTraversal(
            operandType, resultType, resultType.getShape(), *identity.get(),
            *identity.get())))
      return false;
  }
  return true;
}

static analysis::IndexRelationResult getTransposeRelation(
    mlir::MLIRContext *context, mlir::MemRefType destinationType,
    mlir::MemRefType sourceType, llvm::ArrayRef<int64_t> permutation) {
  if (destinationType.getRank() != sourceType.getRank() ||
      permutation.size() != static_cast<size_t>(destinationType.getRank()))
    return {};
  llvm::SmallVector<mlir::AffineExpr, 4> sourceCoordinates(
      sourceType.getRank());
  for (auto [destinationDim, sourceDim] : llvm::enumerate(permutation)) {
    if (sourceDim < 0 || sourceDim >= sourceType.getRank() ||
        sourceCoordinates[sourceDim])
      return {};
    sourceCoordinates[sourceDim] =
        mlir::getAffineDimExpr(destinationDim, context);
  }
  mlir::AffineMap map = mlir::AffineMap::get(destinationType.getRank(), 0,
                                             sourceCoordinates, context);
  return analysis::IndexRelation::fromAffineMap(map, destinationType.getShape(),
                                                sourceType.getShape());
}

struct ExactUnaryView {
  mlir::Value source;
  analysis::IndexRelation relation;
};

static std::optional<ExactUnaryView>
getExactUnaryView(mlir::Operation *operation) {
  if (!operation || operation->getNumResults() != 1)
    return std::nullopt;
  auto resultType =
      mlir::dyn_cast<mlir::MemRefType>(operation->getResult(0).getType());
  if (!resultType || !resultType.hasStaticShape())
    return std::nullopt;

  mlir::Value source;
  analysis::IndexRelationResult relation;
  if (auto reshape = mlir::dyn_cast<ViewReshapeOp>(operation)) {
    source = reshape.getSource();
    auto sourceType = mlir::dyn_cast<mlir::MemRefType>(source.getType());
    if (!sourceType || !sourceType.hasStaticShape())
      return std::nullopt;
    relation = analysis::IndexRelation::staticReshape(resultType.getShape(),
                                                      sourceType.getShape());
  } else if (auto transpose = mlir::dyn_cast<MoveTransposeOp>(operation)) {
    source = transpose.getSource();
    auto sourceType = mlir::dyn_cast<mlir::MemRefType>(source.getType());
    if (!sourceType || !sourceType.hasStaticShape())
      return std::nullopt;
    relation =
        getTransposeRelation(operation->getContext(), resultType, sourceType,
                             transpose.getPermutationAttr().asArrayRef());
  } else if (auto extract = mlir::dyn_cast<MoveExtractSliceOp>(operation)) {
    source = extract.getSource();
    auto sourceType = mlir::dyn_cast<mlir::MemRefType>(source.getType());
    if (!sourceType || !sourceType.hasStaticShape())
      return std::nullopt;
    llvm::ArrayRef<int64_t> sizes = extract.getSizes();
    std::optional<llvm::SmallDenseSet<unsigned>> rankReductionMask =
        mlir::computeRankReductionMask(sizes, resultType.getShape());
    if (!rankReductionMask)
      return std::nullopt;
    llvm::SmallVector<mlir::AffineExpr, 4> sourceCoordinates;
    sourceCoordinates.reserve(sourceType.getRank());
    unsigned resultDim = 0;
    for (unsigned sourceDim = 0; sourceDim < sizes.size(); ++sourceDim) {
      mlir::AffineExpr coordinate = mlir::getAffineConstantExpr(
          extract.getOffsets()[sourceDim], operation->getContext());
      if (!rankReductionMask->contains(sourceDim))
        coordinate = coordinate + mlir::getAffineDimExpr(
                                      resultDim++, operation->getContext()) *
                                      extract.getStrides()[sourceDim];
      sourceCoordinates.push_back(coordinate);
    }
    relation = analysis::IndexRelation::fromAffineMap(
        mlir::AffineMap::get(resultType.getRank(), 0, sourceCoordinates,
                             operation->getContext()),
        resultType.getShape(), sourceType.getShape());
  } else if (auto broadcast = mlir::dyn_cast<MoveBroadcastOp>(operation)) {
    source = broadcast.getSource();
    auto sourceType = mlir::dyn_cast<mlir::MemRefType>(source.getType());
    if (!sourceType || !sourceType.hasStaticShape() ||
        broadcast.getDimensions().size() !=
            static_cast<size_t>(sourceType.getRank()))
      return std::nullopt;
    llvm::SmallVector<mlir::AffineExpr, 4> sourceCoordinates;
    sourceCoordinates.reserve(sourceType.getRank());
    for (int64_t resultDim : broadcast.getDimensions()) {
      if (resultDim < 0 || resultDim >= resultType.getRank())
        return std::nullopt;
      sourceCoordinates.push_back(
          mlir::getAffineDimExpr(resultDim, operation->getContext()));
    }
    relation = analysis::IndexRelation::fromAffineMap(
        mlir::AffineMap::get(resultType.getRank(), 0, sourceCoordinates,
                             operation->getContext()),
        resultType.getShape(), sourceType.getShape());
  } else if (auto copy = mlir::dyn_cast<MoveCopyOp>(operation)) {
    source = copy.getSource();
    auto sourceType = mlir::dyn_cast<mlir::MemRefType>(source.getType());
    if (!sourceType || sourceType.getShape() != resultType.getShape())
      return std::nullopt;
    relation = analysis::IndexRelation::identity(resultType.getShape());
  } else {
    return std::nullopt;
  }
  if (!relation.isExact())
    return std::nullopt;
  return ExactUnaryView{source, std::move(*relation.get())};
}

static bool mayMutateValue(mlir::Operation *operation, mlir::Value value) {
  if (mlir::isMemoryEffectFree(operation))
    return false;
  auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
  if (!effects)
    return true;
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 8> instances;
  effects.getEffects(instances);
  for (const mlir::MemoryEffects::EffectInstance &effect : instances) {
    if (effect.getValue() != value)
      continue;
    if (!mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect()))
      return true;
  }
  // Every exact unary movement reads its explicit source and writes a fresh
  // result. Its rootless SPM/movement-resource effects do not mutate an
  // unrelated source captured by another pre-view in the same diamond.
  return !getExactUnaryView(operation).has_value();
}

static bool hasPotentialMutationBetween(mlir::Operation *before,
                                        mlir::Operation *after,
                                        mlir::Value value) {
  if (!before || !after || before->getBlock() != after->getBlock() ||
      !before->isBeforeInBlock(after))
    return true;
  for (mlir::Operation *operation = before->getNextNode();
       operation && operation != after; operation = operation->getNextNode())
    if (mayMutateValue(operation, value))
      return true;
  return false;
}

static unsigned eliminateInversePointwiseViews(mlir::ModuleOp module) {
  unsigned changed = 0;
  bool madeProgress = true;
  uint64_t valueAndUseCount = 0;
  module.walk([&](mlir::Operation *operation) {
    valueAndUseCount += operation->getNumResults();
    valueAndUseCount += operation->getNumOperands();
  });
  constexpr uint64_t maxDisjuncts = 8;
  uint64_t workRemaining =
      std::max<uint64_t>(1, maxDisjuncts * valueAndUseCount);
  while (madeProgress && workRemaining > 0) {
    madeProgress = false;
    llvm::SmallVector<mlir::Operation *, 8> computes;
    module.walk([&](mlir::Operation *operation) {
      if (isEligiblePointwiseCompute(operation))
        computes.push_back(operation);
    });
    for (mlir::Operation *compute : computes) {
      if (workRemaining-- == 0)
        break;
      if (!compute->getResult(0).hasOneUse())
        continue;
      mlir::Operation *post = *compute->getResult(0).getUsers().begin();
      std::optional<ExactUnaryView> postView = getExactUnaryView(post);
      if (!postView || post->getBlock() != compute->getBlock() ||
          postView->source != compute->getResult(0) ||
          hasPotentialMutationBetween(compute, post, compute->getResult(0)))
        continue;
      auto destinationType =
          mlir::dyn_cast<mlir::MemRefType>(post->getResult(0).getType());
      if (!destinationType)
        continue;
      analysis::IndexRelationResult destinationIdentity =
          analysis::IndexRelation::identity(destinationType.getShape());
      if (!destinationIdentity.isExact())
        continue;
      analysis::IndexRelationQueryResult postIsIdentity =
          postView->relation.isEquivalentTo(*destinationIdentity.get());

      llvm::SmallVector<mlir::Value, 4> replacementOperands;
      llvm::SmallVector<mlir::Operation *, 4> preViews;
      bool exact = true;
      for (mlir::Value operand : compute->getOperands()) {
        auto operandType = mlir::dyn_cast<mlir::MemRefType>(operand.getType());
        if (!operandType) {
          exact = false;
          break;
        }
        if (operandType.getShape() == destinationType.getShape() &&
            postIsIdentity.isProvenTrue()) {
          replacementOperands.push_back(operand);
          continue;
        }
        mlir::Operation *pre = operand.getDefiningOp();
        std::optional<ExactUnaryView> preView = getExactUnaryView(pre);
        auto sourceType =
            preView
                ? mlir::dyn_cast<mlir::MemRefType>(preView->source.getType())
                : mlir::MemRefType{};
        if (!preView || pre->getBlock() != compute->getBlock() || !sourceType ||
            sourceType.getShape() != destinationType.getShape() ||
            hasPotentialMutationBetween(pre, compute, preView->source)) {
          exact = false;
          break;
        }
        analysis::IndexRelationResult composed =
            postView->relation.compose(preView->relation);
        analysis::IndexRelationQueryResult equivalent =
            composed.isExact()
                ? composed.get()->isEquivalentTo(*destinationIdentity.get())
                : analysis::IndexRelationQueryResult{};
        if (!composed.isExact() || !equivalent.isProvenTrue()) {
          exact = false;
          break;
        }
        replacementOperands.push_back(preView->source);
        preViews.push_back(pre);
      }
      if (!exact || replacementOperands.size() != compute->getNumOperands())
        continue;

      mlir::OpBuilder builder(compute);
      mlir::IRMapping mapping;
      mlir::Operation *replacement = builder.clone(*compute, mapping);
      replacement->setOperands(replacementOperands);
      replacement->getResult(0).setType(destinationType);
      if (mlir::failed(mlir::verify(replacement))) {
        replacement->erase();
        continue;
      }
      post->getResult(0).replaceAllUsesWith(replacement->getResult(0));
      post->erase();
      compute->erase();
      for (mlir::Operation *pre : preViews)
        if (pre->use_empty())
          pre->erase();
      ++changed;
      madeProgress = true;
    }
  }
  return changed;
}

static PBQPEdge &getOrCreateEdge(PBQPGraph &graph, unsigned lhs, unsigned rhs) {
  if (lhs > rhs)
    std::swap(lhs, rhs);
  for (PBQPEdge &edge : graph.edges)
    if (edge.lhs == lhs && edge.rhs == rhs)
      return edge;
  PBQPEdge edge;
  edge.lhs = lhs;
  edge.rhs = rhs;
  edge.lhsStates = graph.nodes[lhs].states.size();
  edge.rhsStates = graph.nodes[rhs].states.size();
  edge.matrix.resize(edge.lhsStates * edge.rhsStates);
  graph.edges.push_back(std::move(edge));
  return graph.edges.back();
}

static void addUnaryCost(PBQPNode &node, unsigned state,
                         std::optional<ProposalCost> cost) {
  if (!cost) {
    node.unary[state] = ProposalCost::infinity();
    return;
  }
  node.unary[state] = rawAdd(node.unary[state], *cost);
}

static void collectExternalLayoutRequirements(
    mlir::Value value, const llvm::DenseMap<mlir::Operation *, unsigned> &nodes,
    llvm::DenseSet<mlir::Value> &visited,
    llvm::SmallDenseSet<MemLayout, 4> &requirements) {
  if (!visited.insert(value).second)
    return;
  for (mlir::OpOperand &use : value.getUses()) {
    mlir::Operation *owner = use.getOwner();
    if (nodes.count(owner))
      continue;
    if (auto materialize = mlir::dyn_cast<LayoutMaterializeOp>(owner)) {
      collectExternalLayoutRequirements(materialize.getResult(), nodes, visited,
                                        requirements);
      continue;
    }
    std::optional<MemLayout> layout = getLayout(value.getType());
    if (layout)
      requirements.insert(*layout);
  }
}

static PBQPGraph buildLayoutGraph(mlir::ModuleOp module) {
  PBQPGraph graph;
  llvm::DenseMap<mlir::Operation *, unsigned> nodeIndices;
  module.walk([&](mlir::Operation *operation) {
    if (!isLayoutAssignableOperation(operation))
      return;
    PBQPNode node;
    node.operation = operation;
    constexpr MemLayout layouts[] = {MemLayout::Tensor, MemLayout::NTensor,
                                     MemLayout::Cx, MemLayout::NCx};
    for (MemLayout layout : layouts) {
      if (!isLayoutStateLegal(operation, layout))
        continue;
      node.states.push_back(layout);
      node.unary.push_back(ProposalCost{});
    }
    if (node.states.size() < 2)
      return;
    nodeIndices[operation] = graph.nodes.size();
    graph.nodes.push_back(std::move(node));
  });

  llvm::DenseSet<std::pair<unsigned, unsigned>> accountedProducerInputs;
  for (auto [nodeIndex, node] : llvm::enumerate(graph.nodes)) {
    for (auto [operandIndex, operand] :
         llvm::enumerate(node.operation->getOperands())) {
      mlir::Value root = stripLayoutMaterializations(operand);
      mlir::Operation *producer = root.getDefiningOp();
      auto producerIt = nodeIndices.find(producer);
      bool sameBlockProducer =
          producerIt != nodeIndices.end() &&
          producer->getBlock() == node.operation->getBlock();
      if (sameBlockProducer) {
        unsigned producerIndex = producerIt->second;
        if (!accountedProducerInputs.insert({producerIndex, nodeIndex}).second)
          continue;
        PBQPEdge &edge = getOrCreateEdge(graph, producerIndex, nodeIndex);
        for (unsigned producerState = 0;
             producerState < graph.nodes[producerIndex].states.size();
             ++producerState) {
          auto sourceType = mlir::cast<mlir::MemRefType>(root.getType());
          sourceType = withLayout(
              sourceType, graph.nodes[producerIndex].states[producerState]);
          for (unsigned consumerState = 0; consumerState < node.states.size();
               ++consumerState) {
            auto destType = mlir::cast<mlir::MemRefType>(operand.getType());
            destType = withLayout(destType, node.states[consumerState]);
            std::optional<ProposalCost> cost =
                getMaterializationCost(sourceType, destType);
            ProposalCost value = cost ? *cost : ProposalCost::infinity();
            ProposalCost &target = producerIndex < nodeIndex
                                       ? edge.at(producerState, consumerState)
                                       : edge.at(consumerState, producerState);
            target = rawAdd(target, value);
          }
        }
        continue;
      }

      auto sourceType = mlir::dyn_cast<mlir::MemRefType>(root.getType());
      if (!sourceType)
        continue;
      for (unsigned state = 0; state < node.states.size(); ++state) {
        auto destType = mlir::cast<mlir::MemRefType>(operand.getType());
        destType = withLayout(destType, node.states[state]);
        if (mlir::isa<CommAllReduceOp>(node.operation) && operandIndex == 1 &&
            root.getDefiningOp<mlir::memref::AllocOp>())
          continue;
        addUnaryCost(node, state, getMaterializationCost(sourceType, destType));
      }
    }

    llvm::DenseSet<mlir::Value> visited;
    llvm::SmallDenseSet<MemLayout, 4> requirements;
    collectExternalLayoutRequirements(node.operation->getResult(0), nodeIndices,
                                      visited, requirements);
    auto resultType =
        mlir::cast<mlir::MemRefType>(node.operation->getResult(0).getType());
    for (MemLayout required : requirements) {
      auto destType = withLayout(resultType, required);
      for (unsigned state = 0; state < node.states.size(); ++state) {
        auto sourceType = withLayout(resultType, node.states[state]);
        addUnaryCost(node, state, getMaterializationCost(sourceType, destType));
      }
    }
  }
  return graph;
}

struct MovementStats {
  uint64_t bytes = 0;
  unsigned commands = 0;
};

static MovementStats getLayoutMovementStats(mlir::ModuleOp module) {
  MovementStats stats;
  auto addResultMovement = [&](mlir::Value result) {
    auto type = mlir::dyn_cast<mlir::MemRefType>(result.getType());
    std::optional<WaferPhysicalTensorInfo> info =
        type ? computeWaferPhysicalTensorInfo(type) : std::nullopt;
    if (!info || info->physicalBytes <= 0) {
      stats.bytes = std::numeric_limits<uint64_t>::max();
      return;
    }
    uint64_t bytes = static_cast<uint64_t>(info->physicalBytes);
    if (bytes > std::numeric_limits<uint64_t>::max() - stats.bytes)
      stats.bytes = std::numeric_limits<uint64_t>::max();
    else
      stats.bytes += bytes;
    if (stats.commands != std::numeric_limits<unsigned>::max())
      ++stats.commands;
  };
  module.walk([&](mlir::Operation *operation) {
    if (mlir::isa<LayoutMaterializeOp, MoveExtractSliceOp, MoveInsertSliceOp,
                  MoveCopyOp, MoveTransposeOp, MoveBroadcastOp>(operation))
      addResultMovement(operation->getResult(0));
  });
  module.walk([&](ViewReshapeOp reshape) {
    auto sourceType =
        mlir::dyn_cast<mlir::MemRefType>(reshape.getSource().getType());
    auto resultType =
        mlir::dyn_cast<mlir::MemRefType>(reshape.getResult().getType());
    if (!sourceType || !resultType ||
        mlir::succeeded(
            analysis::TransferRealizability::proveStaticReshapeMetadataView(
                sourceType, resultType, /*destinationMayWrite=*/true)))
      return;
    addResultMovement(reshape.getResult());
  });
  return stats;
}

struct LayoutVersions {
  mlir::Value tensor;
  mlir::Value nTensor;
  mlir::Value cx;
  mlir::Value nCx;
};

static mlir::Value &getVersionSlot(LayoutVersions &versions, MemLayout layout) {
  switch (layout) {
  case MemLayout::Tensor:
    return versions.tensor;
  case MemLayout::NTensor:
    return versions.nTensor;
  case MemLayout::Cx:
    return versions.cx;
  case MemLayout::NCx:
    return versions.nCx;
  }
  llvm_unreachable("unknown memory layout");
}

static mlir::Value getAnyVersion(LayoutVersions &versions) {
  if (versions.tensor)
    return versions.tensor;
  if (versions.nTensor)
    return versions.nTensor;
  if (versions.cx)
    return versions.cx;
  return versions.nCx;
}

static unsigned getVersionCount(const LayoutVersions &versions) {
  return static_cast<unsigned>(static_cast<bool>(versions.tensor)) +
         static_cast<unsigned>(static_cast<bool>(versions.nTensor)) +
         static_cast<unsigned>(static_cast<bool>(versions.cx)) +
         static_cast<unsigned>(static_cast<bool>(versions.nCx));
}

static mlir::FailureOr<mlir::Value>
getOrCreateVersion(mlir::Value root, MemLayout layout,
                   mlir::Operation *insertionPoint,
                   llvm::DenseMap<mlir::Value, LayoutVersions> &versions) {
  LayoutVersions &rootVersions = versions[root];
  mlir::Value &slot = getVersionSlot(rootVersions, layout);
  if (slot)
    return slot;
  if (getVersionCount(rootVersions) >= 2)
    return mlir::failure();
  mlir::Value source = getAnyVersion(rootVersions);
  if (!source)
    source = root;
  auto sourceType = mlir::dyn_cast<mlir::MemRefType>(source.getType());
  mlir::MemRefType destType =
      sourceType ? withLayout(sourceType, layout) : mlir::MemRefType{};
  if (!destType || !getMaterializationCost(sourceType, destType))
    return mlir::failure();
  mlir::OpBuilder builder(insertionPoint);
  slot = builder
             .create<LayoutMaterializeOp>(insertionPoint->getLoc(), destType,
                                          source)
             .getResult();
  return slot;
}

static mlir::LogicalResult applyAssignment(mlir::ModuleOp module,
                                           const PBQPGraph &graph,
                                           const PBQPSolution &solution) {
  llvm::DenseMap<mlir::Operation *, unsigned> nodeIndices;
  llvm::DenseSet<mlir::Value> relevantRoots;
  for (auto [index, node] : llvm::enumerate(graph.nodes)) {
    nodeIndices[node.operation] = index;
    relevantRoots.insert(node.operation->getResult(0));
    for (mlir::Value operand : node.operation->getOperands())
      relevantRoots.insert(stripLayoutMaterializations(operand));
  }

  llvm::SmallVector<mlir::Operation *, 64> originalOperations;
  module.walk([&](mlir::Operation *operation) {
    if (operation != module.getOperation())
      originalOperations.push_back(operation);
  });
  llvm::DenseMap<mlir::Value, LayoutVersions> versions;
  llvm::SmallVector<mlir::Operation *, 32> oldOperations;

  for (mlir::Operation *operation : originalOperations) {
    if (!operation->getBlock())
      continue;
    if (auto materialize = mlir::dyn_cast<LayoutMaterializeOp>(operation)) {
      mlir::Value root = stripLayoutMaterializations(materialize.getSource());
      if (relevantRoots.contains(root))
        oldOperations.push_back(operation);
      continue;
    }

    auto nodeIt = nodeIndices.find(operation);
    if (nodeIt != nodeIndices.end()) {
      unsigned nodeIndex = nodeIt->second;
      MemLayout layout =
          graph.nodes[nodeIndex].states[solution.states[nodeIndex]];
      llvm::SmallVector<mlir::Value, 4> operands;
      for (auto [operandIndex, operand] :
           llvm::enumerate(operation->getOperands())) {
        mlir::Value root = stripLayoutMaterializations(operand);
        std::optional<MemLayout> rootLayout = getLayout(root.getType());
        if (mlir::isa<CommAllReduceOp>(operation) && operandIndex == 1 &&
            rootLayout && *rootLayout != layout) {
          auto oldAlloc = root.getDefiningOp<mlir::memref::AllocOp>();
          auto rootType = mlir::dyn_cast<mlir::MemRefType>(root.getType());
          mlir::MemRefType destType =
              rootType ? withLayout(rootType, layout) : mlir::MemRefType{};
          if (!oldAlloc || !destType)
            return mlir::failure();
          LayoutVersions &rootVersions = versions[root];
          mlir::Value &slot = getVersionSlot(rootVersions, layout);
          if (!slot) {
            mlir::OpBuilder builder(operation);
            slot = builder
                       .create<mlir::memref::AllocOp>(operation->getLoc(),
                                                      destType)
                       .getResult();
          }
          operands.push_back(slot);
          oldOperations.push_back(oldAlloc);
          continue;
        }
        LayoutVersions &rootVersions = versions[root];
        if (!getAnyVersion(rootVersions)) {
          if (!rootLayout)
            return mlir::failure();
          getVersionSlot(rootVersions, *rootLayout) = root;
        }
        mlir::FailureOr<mlir::Value> version =
            getOrCreateVersion(root, layout, operation, versions);
        if (mlir::failed(version))
          return mlir::failure();
        operands.push_back(*version);
      }

      mlir::OpBuilder builder(operation);
      mlir::IRMapping mapping;
      mlir::Operation *clone = builder.clone(*operation, mapping);
      clone->setOperands(operands);
      auto oldResultType =
          mlir::cast<mlir::MemRefType>(operation->getResult(0).getType());
      mlir::MemRefType newResultType = withLayout(oldResultType, layout);
      clone->getResult(0).setType(newResultType);
      if (mlir::isa<CommAllReduceOp>(clone)) {
        std::optional<WaferPhysicalTensorInfo> physical =
            computeWaferPhysicalTensorInfo(newResultType);
        if (!physical || physical->physicalBytes <= 0)
          return mlir::failure();
        clone->setAttr("bytes",
                       builder.getI64IntegerAttr(physical->physicalBytes));
      }
      mlir::Value root = operation->getResult(0);
      LayoutVersions &rootVersions = versions[root];
      getVersionSlot(rootVersions, layout) = clone->getResult(0);
      oldOperations.push_back(operation);
      continue;
    }

    for (mlir::OpOperand &operand : operation->getOpOperands()) {
      mlir::Value root = stripLayoutMaterializations(operand.get());
      if (!relevantRoots.contains(root))
        continue;
      auto versionsIt = versions.find(root);
      if (versionsIt == versions.end() || !getAnyVersion(versionsIt->second))
        continue;
      std::optional<MemLayout> required = getLayout(operand.get().getType());
      if (!required)
        continue;
      mlir::FailureOr<mlir::Value> replacement =
          getOrCreateVersion(root, *required, operation, versions);
      if (mlir::failed(replacement))
        return mlir::failure();
      operand.set(*replacement);
    }
  }

  bool erased = true;
  while (erased && !oldOperations.empty()) {
    erased = false;
    for (auto it = oldOperations.begin(); it != oldOperations.end();) {
      mlir::Operation *operation = *it;
      if (operation->use_empty()) {
        operation->erase();
        it = oldOperations.erase(it);
        erased = true;
      } else {
        ++it;
      }
    }
  }
  if (!oldOperations.empty())
    return mlir::failure();
  return mlir::verify(module);
}

static std::vector<unsigned> getBaselineStateVector(const PBQPGraph &graph) {
  std::vector<unsigned> baseline;
  baseline.reserve(graph.nodes.size());
  for (const PBQPNode &node : graph.nodes) {
    std::optional<MemLayout> current =
        getLayout(node.operation->getResult(0).getType());
    unsigned state = 0;
    for (; state < node.states.size(); ++state)
      if (node.states[state] == current)
        break;
    baseline.push_back(state);
  }
  return baseline;
}

} // namespace

static mlir::LogicalResult
applyPhysicalLayoutProposalImpl(mlir::ModuleOp module, unsigned proposalOrdinal,
                                PhysicalLayoutProposalResult *result,
                                std::string *failureReason) {
  auto fail = [&](llvm::StringRef reason) {
    if (failureReason)
      *failureReason = reason.str();
    return mlir::failure();
  };
  MovementStats before = getLayoutMovementStats(module);
  unsigned relationRewrites = eliminateInversePointwiseViews(module);
  PBQPGraph graph = buildLayoutGraph(module);
  if (graph.nodes.empty()) {
    if (relationRewrites == 0 || proposalOrdinal != 0)
      return fail("physical layout projection has no requested typed compute "
                  "proposal");
    MovementStats after = getLayoutMovementStats(module);
    if (after.bytes >= before.bytes && after.commands >= before.commands)
      return fail("relation-guided proposal does not reduce actual movement");
    if (mlir::failed(mlir::verify(module)))
      return fail("relation-guided proposal failed actual-clone verification");
    if (result) {
      result->proposalCount = 1;
      result->appliedProposalOrdinal = 0;
      result->movementBytesBefore = before.bytes;
      result->movementBytesAfter = after.bytes;
      result->movementCommandsBefore = before.commands;
      result->movementCommandsAfter = after.commands;
    }
    return mlir::success();
  }
  std::vector<unsigned> baseline = getBaselineStateVector(graph);
  llvm::SmallVector<PBQPSolution, kMaximumProposals> solved =
      solveTopProposals(graph);
  llvm::SmallVector<PBQPSolution, kMaximumProposals> proposals;
  for (PBQPSolution &solution : solved) {
    if (solution.states != baseline || relationRewrites != 0)
      proposals.push_back(std::move(solution));
  }
  if (result)
    result->proposalCount = proposals.size();
  if (proposalOrdinal >= proposals.size())
    return fail("requested physical layout proposal ordinal is unavailable");

  if (mlir::failed(applyAssignment(module, graph, proposals[proposalOrdinal])))
    return fail("physical layout proposal failed typed actual-clone "
                "materialization or verification");
  MovementStats after = getLayoutMovementStats(module);
  if (after.bytes >= before.bytes && after.commands >= before.commands)
    return fail("physical layout proposal does not reduce actual layout "
                "movement");
  if (result) {
    result->appliedProposalOrdinal = proposalOrdinal;
    result->movementBytesBefore = before.bytes;
    result->movementBytesAfter = after.bytes;
    result->movementCommandsBefore = before.commands;
    result->movementCommandsAfter = after.commands;
  }
  return mlir::success();
}

mlir::LogicalResult
applyPhysicalLayoutProposal(mlir::ModuleOp module, unsigned proposalOrdinal,
                            PhysicalLayoutProposalResult *result,
                            std::string *failureReason) {
  if (!module) {
    if (failureReason)
      *failureReason = "physical layout proposal requires a module";
    return mlir::failure();
  }
  mlir::OwningOpRef<mlir::ModuleOp> candidate =
      mlir::cast<mlir::ModuleOp>(module->clone());
  PhysicalLayoutProposalResult candidateResult;
  if (mlir::failed(applyPhysicalLayoutProposalImpl(
          *candidate, proposalOrdinal, &candidateResult, failureReason)))
    return mlir::failure();

  // Commit only a completely materialized and verified proposal. A rejected
  // ordinal, exhausted proof/solver budget, or failed typed verifier leaves
  // the caller's actual clone byte-for-byte structurally unchanged.
  module->setAttrs((*candidate)->getAttrs());
  module.getBodyRegion().takeBody(candidate->getBodyRegion());
  if (result)
    *result = candidateResult;
  return mlir::success();
}

} // namespace wafer
