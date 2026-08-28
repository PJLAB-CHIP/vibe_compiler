//===- MiniMallocPacking.cpp - MiniMalloc static packing adapter ---------===//

#include "Wafer/Transforms/Instr/MiniMallocPacking.h"

#include "wafer_third_party/minimalloc/minimalloc.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace wafer::memory_planning::detail {
namespace {

namespace mm = wafer_third_party::minimalloc;

PackingResult makeFailure(PackingStatus status, uint64_t searchNodes = 0) {
  PackingResult result;
  result.status = status;
  result.searchNodes = searchNodes;
  return result;
}

std::optional<int64_t> alignUp(int64_t value, int64_t alignment) {
  int64_t remainder = value % alignment;
  if (remainder == 0)
    return value;
  int64_t result = 0;
  if (llvm::AddOverflow(value, alignment - remainder, result))
    return std::nullopt;
  return result;
}

} // namespace

PackingResult solveWithMiniMalloc(const StaticPackingProblem &problem,
                                  uint64_t searchNodeBudget) {
  PackingResult result;
  result.status = PackingStatus::Feasible;
  if (std::optional<PackingValidationFailure> failure =
          validatePackingProblem(problem)) {
    result.status = PackingStatus::InvalidProblem;
    result.demandIndex = failure->demandIndex;
    return result;
  }
  if (problem.demands.empty())
    return result;

  if (problem.demands.size() > std::numeric_limits<unsigned>::max())
    return makeFailure(PackingStatus::InvalidProblem);

  // MiniMalloc's rectangle-search completeness assumes positive spatial
  // height. Zero-byte Wafer demands do not occupy an address range, so place
  // them independently at the first absolutely aligned address in the arena
  // and exclude them from both the synthetic conflict graph and the solver.
  // This also makes a zero-only problem feasible without consuming search
  // fuel.
  llvm::SmallVector<Placement, 4> zeroPlacements;
  std::vector<unsigned> originalForCanonical;
  originalForCanonical.reserve(problem.demands.size());
  std::vector<unsigned> originalsByStableOrdinal(problem.demands.size());
  std::iota(originalsByStableOrdinal.begin(), originalsByStableOrdinal.end(),
            0);
  std::sort(originalsByStableOrdinal.begin(), originalsByStableOrdinal.end(),
            [&](unsigned lhs, unsigned rhs) {
              return problem.demands[lhs].stableOrdinal <
                     problem.demands[rhs].stableOrdinal;
            });
  for (unsigned original : originalsByStableOrdinal) {
    const StaticPackingDemand &demand = problem.demands[original];
    if (demand.sizeBytes > 0) {
      originalForCanonical.push_back(original);
      continue;
    }
    std::optional<int64_t> offset =
        alignUp(problem.arena.begin, demand.alignmentBytes);
    if (!offset || *offset > problem.arena.end) {
      result.status = PackingStatus::ProvenInfeasible;
      result.demandIndex = original;
      result.placements.clear();
      return result;
    }
    zeroPlacements.push_back(Placement{original, *offset, *offset});
  }
  if (originalForCanonical.empty()) {
    result.placements = std::move(zeroPlacements);
    return result;
  }

  const uint64_t arenaCapacity =
      static_cast<uint64_t>(problem.arena.end - problem.arena.begin);
  for (unsigned original : originalForCanonical) {
    if (static_cast<uint64_t>(problem.demands[original].sizeBytes) <=
        arenaCapacity)
      continue;
    result.individuallyOversizedDemandIndices.push_back(original);
  }
  if (!result.individuallyOversizedDemandIndices.empty()) {
    llvm::sort(result.individuallyOversizedDemandIndices,
               [&](unsigned lhs, unsigned rhs) {
                 const StaticPackingDemand &lhsDemand = problem.demands[lhs];
                 const StaticPackingDemand &rhsDemand = problem.demands[rhs];
                 if (lhsDemand.sizeBytes != rhsDemand.sizeBytes)
                   return lhsDemand.sizeBytes > rhsDemand.sizeBytes;
                 return lhsDemand.stableOrdinal < rhsDemand.stableOrdinal;
               });
    result.status = PackingStatus::ProvenInfeasible;
    result.demandIndex = result.individuallyOversizedDemandIndices.front();
    // Any single member is also a size-one over-capacity clique certificate.
    result.capacityConflictDemandIndices.push_back(*result.demandIndex);
    result.placements.clear();
    return result;
  }

  // Canonicalize positive-size demands independently of input vector order.
  // The source demand index is restored after solving, so owner diagnostics
  // and assigned offsets still address the original LifetimeDemand.
  constexpr unsigned kNotCanonical = std::numeric_limits<unsigned>::max();
  std::vector<unsigned> canonicalForOriginal(problem.demands.size(),
                                             kNotCanonical);
  for (unsigned canonical = 0; canonical < originalForCanonical.size();
       ++canonical)
    canonicalForOriginal[originalForCanonical[canonical]] = canonical;

  struct CanonicalConflict {
    unsigned lhs = 0;
    unsigned rhs = 0;
  };
  std::vector<CanonicalConflict> conflicts;
  conflicts.reserve(problem.conflicts.size());
  for (const PackingConflict &conflict : problem.conflicts) {
    unsigned lhs = canonicalForOriginal[conflict.lhsDemandIndex];
    unsigned rhs = canonicalForOriginal[conflict.rhsDemandIndex];
    if (lhs == kNotCanonical || rhs == kNotCanonical)
      continue;
    if (lhs > rhs)
      std::swap(lhs, rhs);
    conflicts.push_back(CanonicalConflict{lhs, rhs});
  }
  std::sort(conflicts.begin(), conflicts.end(),
            [](const CanonicalConflict &lhs, const CanonicalConflict &rhs) {
              if (lhs.lhs != rhs.lhs)
                return lhs.lhs < rhs.lhs;
              return lhs.rhs < rhs.rhs;
            });

  // Any undirected conflict graph is represented exactly. Each half-open
  // activity slot is a graph clique, and a deterministic greedy edge-clique
  // cover ensures every conflict edge shares at least one slot. Non-edges are
  // never placed in the same slot. Compared with one slot per edge, preserving
  // whole cliques also exposes strong capacity lower bounds to MiniMalloc and
  // avoids needlessly fragmenting interval-like production graphs.
  std::vector<std::vector<int64_t>> activeSlots(originalForCanonical.size());
  if (conflicts.size() >
      static_cast<size_t>(std::numeric_limits<int64_t>::max()))
    return makeFailure(PackingStatus::ArithmeticOverflow);

  std::vector<llvm::BitVector> adjacency(
      originalForCanonical.size(),
      llvm::BitVector(originalForCanonical.size(), false));
  std::vector<llvm::BitVector> uncovered(
      originalForCanonical.size(),
      llvm::BitVector(originalForCanonical.size(), false));
  for (const CanonicalConflict &conflict : conflicts) {
    adjacency[conflict.lhs].set(conflict.rhs);
    adjacency[conflict.rhs].set(conflict.lhs);
    uncovered[conflict.lhs].set(conflict.rhs);
    uncovered[conflict.rhs].set(conflict.lhs);
  }

  struct ActivityComponent {
    std::vector<unsigned> members;
    std::vector<CanonicalConflict> conflicts;
    int64_t slotBegin = 0;
    int64_t slotEnd = 0;
  };
  std::vector<ActivityComponent> components;
  std::vector<unsigned> componentForCanonical(originalForCanonical.size(),
                                              kNotCanonical);

  // A clique whose raw byte sum exceeds the usable arena is already a
  // complete fixed-problem infeasibility proof.  Keep a deterministic,
  // size-descending minimal prefix as a causal certificate for the upstream
  // joint search.  This is derived from the exact final lifetime conflict
  // graph; it neither estimates residency nor changes which packing problems
  // are legal.
  auto recordOverCapacityClique =
      [&](llvm::ArrayRef<unsigned> canonicalClique) {
        llvm::SmallVector<unsigned, 8> originals;
        originals.reserve(canonicalClique.size());
        for (unsigned canonical : canonicalClique)
          originals.push_back(originalForCanonical[canonical]);
        llvm::sort(originals, [&](unsigned lhs, unsigned rhs) {
          const StaticPackingDemand &lhsDemand = problem.demands[lhs];
          const StaticPackingDemand &rhsDemand = problem.demands[rhs];
          if (lhsDemand.sizeBytes != rhsDemand.sizeBytes)
            return lhsDemand.sizeBytes > rhsDemand.sizeBytes;
          return lhsDemand.stableOrdinal < rhsDemand.stableOrdinal;
        });

        uint64_t totalBytes = 0;
        llvm::SmallVector<unsigned, 8> certificate;
        for (unsigned original : originals) {
          const uint64_t bytes =
              static_cast<uint64_t>(problem.demands[original].sizeBytes);
          totalBytes = totalBytes > std::numeric_limits<uint64_t>::max() - bytes
                           ? std::numeric_limits<uint64_t>::max()
                           : totalBytes + bytes;
          certificate.push_back(original);
          if (totalBytes <= arenaCapacity)
            continue;
          result.status = PackingStatus::ProvenInfeasible;
          result.demandIndex = certificate.front();
          result.capacityConflictDemandIndices = std::move(certificate);
          result.placements.clear();
          return true;
        }
        return false;
      };
  for (unsigned root = 0; root < originalForCanonical.size(); ++root) {
    if (componentForCanonical[root] != kNotCanonical)
      continue;
    unsigned componentIndex = components.size();
    components.emplace_back();
    std::vector<unsigned> worklist{root};
    componentForCanonical[root] = componentIndex;
    for (size_t next = 0; next < worklist.size(); ++next) {
      unsigned member = worklist[next];
      components.back().members.push_back(member);
      for (int neighbor = adjacency[member].find_first(); neighbor >= 0;
           neighbor = adjacency[member].find_next(neighbor)) {
        unsigned canonicalNeighbor = static_cast<unsigned>(neighbor);
        if (componentForCanonical[canonicalNeighbor] != kNotCanonical)
          continue;
        componentForCanonical[canonicalNeighbor] = componentIndex;
        worklist.push_back(canonicalNeighbor);
      }
    }
    std::sort(components.back().members.begin(),
              components.back().members.end());
  }
  for (const CanonicalConflict &conflict : conflicts) {
    if (componentForCanonical[conflict.lhs] !=
        componentForCanonical[conflict.rhs])
      return makeFailure(PackingStatus::InvalidSolverResult);
    components[componentForCanonical[conflict.lhs]].conflicts.push_back(
        conflict);
  }

  int64_t nextSlot = 0;
  for (ActivityComponent &component : components) {
    component.slotBegin = nextSlot;
    for (const CanonicalConflict &conflict : component.conflicts) {
      if (!uncovered[conflict.lhs].test(conflict.rhs))
        continue;

      std::vector<unsigned> clique{conflict.lhs, conflict.rhs};
      llvm::BitVector candidates = adjacency[conflict.lhs];
      candidates &= adjacency[conflict.rhs];
      for (int candidate = candidates.find_first(); candidate >= 0;
           candidate = candidates.find_next(candidate)) {
        clique.push_back(static_cast<unsigned>(candidate));
        candidates &= adjacency[candidate];
      }

      if (nextSlot == std::numeric_limits<int64_t>::max())
        return makeFailure(PackingStatus::ArithmeticOverflow);
      for (size_t lhs = 0; lhs < clique.size(); ++lhs)
        for (size_t rhs = lhs + 1; rhs < clique.size(); ++rhs)
          if (!adjacency[clique[lhs]].test(clique[rhs]))
            return makeFailure(PackingStatus::InvalidSolverResult);
      if (recordOverCapacityClique(clique))
        return result;
      for (unsigned member : clique)
        activeSlots[member].push_back(nextSlot);
      for (size_t lhs = 0; lhs < clique.size(); ++lhs) {
        for (size_t rhs = lhs + 1; rhs < clique.size(); ++rhs) {
          uncovered[clique[lhs]].reset(clique[rhs]);
          uncovered[clique[rhs]].reset(clique[lhs]);
        }
      }
      ++nextSlot;
    }
    for (unsigned member : component.members) {
      if (!activeSlots[member].empty())
        continue;
      if (nextSlot == std::numeric_limits<int64_t>::max())
        return makeFailure(PackingStatus::ArithmeticOverflow);
      activeSlots[member].push_back(nextSlot++);
    }
    component.slotEnd = nextSlot;
  }
  for (const CanonicalConflict &conflict : conflicts)
    if (uncovered[conflict.lhs].test(conflict.rhs))
      return makeFailure(PackingStatus::InvalidSolverResult);

  mm::Problem miniProblem;
  miniProblem.capacity = problem.arena.end;
  miniProblem.buffers.reserve(
      originalForCanonical.size() +
      (problem.arena.begin > 0 ? components.size() : 0));
  for (unsigned canonical = 0; canonical < originalForCanonical.size();
       ++canonical) {
    unsigned original = originalForCanonical[canonical];
    const StaticPackingDemand &demand = problem.demands[original];
    const std::vector<int64_t> &slots = activeSlots[canonical];

    mm::Buffer buffer;
    buffer.id = std::to_string(demand.stableOrdinal);
    buffer.lifespan = mm::Lifespan{slots.front(), slots.back() + 1};
    buffer.size = demand.sizeBytes;
    buffer.alignment = demand.alignmentBytes;
    for (size_t slotIndex = 1; slotIndex < slots.size(); ++slotIndex) {
      int64_t gapBegin = slots[slotIndex - 1] + 1;
      int64_t gapEnd = slots[slotIndex];
      if (gapBegin < gapEnd)
        buffer.gaps.push_back(
            mm::Gap{mm::Lifespan{gapBegin, gapEnd}, std::nullopt});
    }
    miniProblem.buffers.push_back(std::move(buffer));
  }

  // MiniMalloc aligns absolute offsets to zero. Fixed prefix buffers make the
  // nonzero Wafer arena base an explicit absolute-address constraint; solving
  // relative offsets and adding base would be wrong when base is not divisible
  // by every demand alignment. One prefix per disjoint conflict component
  // keeps their activity ranges independent, so the solver can decompose them.
  if (problem.arena.begin > 0) {
    for (auto [componentIndex, component] : llvm::enumerate(components)) {
      mm::Buffer prefix;
      prefix.id = "wafer-reserved-prefix-" + std::to_string(componentIndex);
      prefix.lifespan = mm::Lifespan{component.slotBegin, component.slotEnd};
      prefix.size = problem.arena.begin;
      prefix.alignment = 1;
      prefix.offset = 0;
      miniProblem.buffers.push_back(std::move(prefix));
    }
  }

  mm::SolveOptions options;
  options.nodeBudget = searchNodeBudget;
  mm::SolveResult miniResult = mm::Solve(miniProblem, options);
  result.searchNodes = miniResult.searchNodes;
  switch (miniResult.status) {
  case mm::SolveStatus::kProvenInfeasible:
    result.status = PackingStatus::ProvenInfeasible;
    return result;
  case mm::SolveStatus::kResourceExhausted:
    result.status = PackingStatus::ResourceExhausted;
    return result;
  case mm::SolveStatus::kInvalidProblem:
    // The Wafer problem and every generated solver field were already checked
    // above. Rejection here therefore indicates an adapter/core contract bug,
    // not malformed user IR.
    result.status = PackingStatus::InvalidSolverResult;
    return result;
  case mm::SolveStatus::kInternalError:
    result.status = PackingStatus::InvalidSolverResult;
    return result;
  case mm::SolveStatus::kFeasible:
    break;
  }

  if (miniResult.solution.offsets.size() != miniProblem.buffers.size())
    return makeFailure(PackingStatus::InvalidSolverResult,
                       miniResult.searchNodes);

  result.placements = std::move(zeroPlacements);
  result.placements.reserve(problem.demands.size());
  for (unsigned canonical = 0; canonical < originalForCanonical.size();
       ++canonical) {
    unsigned original = originalForCanonical[canonical];
    int64_t offset = miniResult.solution.offsets[canonical];
    int64_t end = 0;
    if (llvm::AddOverflow(offset, problem.demands[original].sizeBytes, end))
      return makeFailure(PackingStatus::ArithmeticOverflow,
                         miniResult.searchNodes);
    result.placements.push_back(Placement{original, offset, end});
  }
  llvm::sort(result.placements,
             [&](const Placement &lhs, const Placement &rhs) {
               return problem.demands[lhs.demandIndex].stableOrdinal <
                      problem.demands[rhs.demandIndex].stableOrdinal;
             });
  result.status = PackingStatus::Feasible;
  return result;
}

} // namespace wafer::memory_planning::detail
