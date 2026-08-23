//===- StaticMemoryPacking.cpp - Static MiniMalloc contract ----------===//

#include "MemoryPlanning/StaticMemoryPacking.h"

#include "MemoryPlanning/MiniMallocPacking.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <limits>

namespace wafer::memory_planning::detail {
namespace {

bool checkedAdd(int64_t lhs, int64_t rhs, int64_t &result) {
  return !llvm::AddOverflow(lhs, rhs, result);
}

bool byteRangesOverlap(int64_t lhsBegin, int64_t lhsEnd, int64_t rhsBegin,
                       int64_t rhsEnd) {
  if (lhsBegin >= lhsEnd || rhsBegin >= rhsEnd)
    return false;
  return lhsBegin < rhsEnd && rhsBegin < lhsEnd;
}

uint64_t conflictKey(unsigned lhs, unsigned rhs) {
  return (uint64_t{lhs} << 32) | uint64_t{rhs};
}

PackingResult
validationFailureResult(const PackingValidationFailure &failure) {
  PackingResult result;
  result.demandIndex = failure.demandIndex;
  result.status = failure.kind == PackingValidationFailureKind::RangeOverflow
                      ? PackingStatus::ArithmeticOverflow
                      : PackingStatus::InvalidProblem;
  return result;
}

PackingResult invalidSolverResult(const PackingResult &solverResult,
                                  const PackingValidationFailure &failure) {
  PackingResult result;
  result.status = PackingStatus::InvalidSolverResult;
  result.searchNodes = solverResult.searchNodes;
  result.demandIndex = failure.demandIndex;
  return result;
}

} // namespace

StaticPackingProblem
buildStaticPackingProblem(llvm::ArrayRef<LifetimeDemand> lifetimeDemands,
                          ArenaRange arena) {
  StaticPackingProblem problem;
  problem.arena = arena;
  problem.demands.reserve(lifetimeDemands.size());
  for (const LifetimeDemand &demand : lifetimeDemands) {
    int64_t minEvent = std::numeric_limits<int64_t>::max();
    int64_t maxEvent = std::numeric_limits<int64_t>::min();
    for (const LiveSegment &segment : demand.segments) {
      minEvent = std::min(minEvent, segment.beginEvent);
      maxEvent = std::max(maxEvent, segment.endEvent);
    }
    int64_t span = 0;
    if (!demand.segments.empty() && maxEvent > minEvent &&
        llvm::SubOverflow(maxEvent, minEvent, span))
      span = std::numeric_limits<int64_t>::max();
    problem.demands.push_back(StaticPackingDemand{
        demand.sizeBytes, demand.alignmentBytes, span,
        demand.allocationPoint.event, demand.stableOrdinal});
  }

  for (unsigned lhs = 0; lhs < lifetimeDemands.size(); ++lhs)
    for (unsigned rhs = lhs + 1; rhs < lifetimeDemands.size(); ++rhs)
      if (lifetimesOverlap(lifetimeDemands[lhs], lifetimeDemands[rhs]))
        problem.conflicts.push_back(PackingConflict{lhs, rhs});
  return problem;
}

uint64_t defaultPackingSearchNodeBudget(const StaticPackingProblem &problem) {
  constexpr uint64_t kDemandScale = 64;
  constexpr uint64_t kConflictScale = 16;
  uint64_t budget = kBasePackingSearchNodes;
  auto addCapped = [&](uint64_t count, uint64_t scale) {
    if (budget >= kMaxDefaultPackingSearchNodes)
      return;
    uint64_t remaining = kMaxDefaultPackingSearchNodes - budget;
    if (count > remaining / scale) {
      budget = kMaxDefaultPackingSearchNodes;
      return;
    }
    budget += count * scale;
  };
  addCapped(problem.demands.size(), kDemandScale);
  addCapped(problem.conflicts.size(), kConflictScale);
  return budget;
}

std::optional<PackingValidationFailure>
validatePackingProblem(const StaticPackingProblem &problem) {
  if (problem.arena.begin < 0 || problem.arena.end < problem.arena.begin)
    return PackingValidationFailure{PackingValidationFailureKind::InvalidArena,
                                    std::nullopt};

  llvm::DenseSet<unsigned> ordinals;
  for (auto [index, demand] : llvm::enumerate(problem.demands)) {
    if (demand.sizeBytes < 0 || demand.alignmentBytes <= 0 ||
        !ordinals.insert(demand.stableOrdinal).second)
      return PackingValidationFailure{
          PackingValidationFailureKind::InvalidDemand,
          static_cast<unsigned>(index)};
  }

  llvm::DenseSet<uint64_t> conflicts;
  for (const PackingConflict &conflict : problem.conflicts) {
    if (conflict.lhsDemandIndex >= problem.demands.size() ||
        conflict.rhsDemandIndex >= problem.demands.size() ||
        conflict.lhsDemandIndex >= conflict.rhsDemandIndex ||
        !conflicts
             .insert(
                 conflictKey(conflict.lhsDemandIndex, conflict.rhsDemandIndex))
             .second)
      return PackingValidationFailure{
          PackingValidationFailureKind::InvalidConflict, std::nullopt};
  }
  return std::nullopt;
}

std::optional<PackingValidationFailure>
validatePlacements(const StaticPackingProblem &problem,
                   llvm::ArrayRef<Placement> placements) {
  if (std::optional<PackingValidationFailure> failure =
          validatePackingProblem(problem))
    return failure;

  llvm::SmallVector<const Placement *, 8> byDemand(problem.demands.size(),
                                                   nullptr);
  for (const Placement &placement : placements) {
    if (placement.demandIndex >= problem.demands.size())
      return PackingValidationFailure{
          PackingValidationFailureKind::DuplicatePlacement, std::nullopt};
    if (byDemand[placement.demandIndex])
      return PackingValidationFailure{
          PackingValidationFailureKind::DuplicatePlacement,
          placement.demandIndex};
    byDemand[placement.demandIndex] = &placement;
  }

  for (auto [index, demand] : llvm::enumerate(problem.demands)) {
    const Placement *placement = byDemand[index];
    if (!placement)
      return PackingValidationFailure{
          PackingValidationFailureKind::MissingPlacement,
          static_cast<unsigned>(index)};
    int64_t expectedEnd = 0;
    if (!checkedAdd(placement->offsetBytes, demand.sizeBytes, expectedEnd))
      return PackingValidationFailure{
          PackingValidationFailureKind::RangeOverflow,
          static_cast<unsigned>(index)};
    if (placement->endBytes != expectedEnd)
      return PackingValidationFailure{PackingValidationFailureKind::EndMismatch,
                                      static_cast<unsigned>(index)};
    if (placement->offsetBytes < problem.arena.begin ||
        placement->endBytes > problem.arena.end)
      return PackingValidationFailure{PackingValidationFailureKind::OutOfRange,
                                      static_cast<unsigned>(index)};
    if (placement->offsetBytes % demand.alignmentBytes != 0)
      return PackingValidationFailure{PackingValidationFailureKind::Misaligned,
                                      static_cast<unsigned>(index)};
  }

  for (const PackingConflict &conflict : problem.conflicts) {
    const Placement &lhs = *byDemand[conflict.lhsDemandIndex];
    const Placement &rhs = *byDemand[conflict.rhsDemandIndex];
    if (byteRangesOverlap(lhs.offsetBytes, lhs.endBytes, rhs.offsetBytes,
                          rhs.endBytes))
      return PackingValidationFailure{
          PackingValidationFailureKind::ConflictingRanges,
          conflict.rhsDemandIndex};
  }
  return std::nullopt;
}

PackingResult packStaticMemory(const StaticPackingProblem &problem,
                               std::optional<uint64_t> searchNodeBudget) {
  if (std::optional<PackingValidationFailure> failure =
          validatePackingProblem(problem))
    return validationFailureResult(*failure);

  const uint64_t budget =
      searchNodeBudget.value_or(defaultPackingSearchNodeBudget(problem));
  PackingResult result = solveWithMiniMalloc(problem, budget);
  if (result.status == PackingStatus::Feasible)
    if (std::optional<PackingValidationFailure> failure =
            validatePlacements(problem, result.placements))
      return invalidSolverResult(result, *failure);
  return result;
}

PackingResult packStaticMemory(llvm::ArrayRef<LifetimeDemand> demands,
                               ArenaRange arena,
                               std::optional<uint64_t> searchNodeBudget) {
  return packStaticMemory(buildStaticPackingProblem(demands, arena),
                          searchNodeBudget);
}

} // namespace wafer::memory_planning::detail
