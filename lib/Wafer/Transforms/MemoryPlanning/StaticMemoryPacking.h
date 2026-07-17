//===- StaticMemoryPacking.h - Static arena packing contract ----*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_MEMORYPLANNING_STATICMEMORYPACKING_H
#define WAFER_TRANSFORMS_MEMORYPLANNING_STATICMEMORYPACKING_H

#include "MemoryPlanning/LifetimeAnalysis.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace wafer::memory_planning::detail {

struct ArenaRange {
  int64_t begin = 0;
  int64_t end = 0;
};

struct StaticPackingDemand {
  int64_t sizeBytes = 0;
  int64_t alignmentBytes = 0;
  int64_t lifetimeSpan = 0;
  int64_t allocationEvent = 0;
  unsigned stableOrdinal = 0;
};

struct PackingConflict {
  unsigned lhsDemandIndex = 0;
  unsigned rhsDemandIndex = 0;
};

/// A pure, owner-independent packing problem. Demand indices match the input
/// LifetimeDemand order, while stableOrdinal provides permutation-independent
/// canonical ordering.
struct StaticPackingProblem {
  ArenaRange arena;
  llvm::SmallVector<StaticPackingDemand, 8> demands;
  llvm::SmallVector<PackingConflict, 16> conflicts;
};

struct Placement {
  unsigned demandIndex = 0;
  int64_t offsetBytes = 0;
  int64_t endBytes = 0;
};

enum class PackingStatus {
  Feasible,
  ProvenInfeasible,
  ResourceExhausted,
  HeuristicNoFit,
  InvalidProblem,
  ArithmeticOverflow,
  InvalidSolverResult,
};

enum class PackingBackend {
  MiniMalloc,
  FirstFitFallback,
};

struct PackingResult {
  PackingStatus status = PackingStatus::InvalidProblem;
  PackingBackend backend = PackingBackend::MiniMalloc;
  llvm::SmallVector<Placement, 8> placements;
  uint64_t searchNodes = 0;
  bool fallbackAttempted = false;
  std::optional<unsigned> demandIndex;

  bool succeeded() const { return status == PackingStatus::Feasible; }
};

enum class PackingValidationFailureKind {
  InvalidArena,
  InvalidDemand,
  InvalidConflict,
  MissingPlacement,
  DuplicatePlacement,
  RangeOverflow,
  OutOfRange,
  Misaligned,
  ConflictingRanges,
  EndMismatch,
};

struct PackingValidationFailure {
  PackingValidationFailureKind kind =
      PackingValidationFailureKind::InvalidArena;
  std::optional<unsigned> demandIndex;
};

/// The default is deliberately generous: the fixed base budget is over 13x
/// the largest official MiniMalloc challenging-case node count measured for
/// the pinned source, then grows modestly with problem size up to a stable cap.
constexpr uint64_t kBasePackingSearchNodes = uint64_t{1} << 21;
constexpr uint64_t kMaxDefaultPackingSearchNodes = uint64_t{1} << 24;

StaticPackingProblem
buildStaticPackingProblem(llvm::ArrayRef<LifetimeDemand> demands,
                          ArenaRange arena);

uint64_t defaultPackingSearchNodeBudget(const StaticPackingProblem &problem);

std::optional<PackingValidationFailure>
validatePackingProblem(const StaticPackingProblem &problem);

std::optional<PackingValidationFailure>
validatePlacements(const StaticPackingProblem &problem,
                   llvm::ArrayRef<Placement> placements);

/// Production policy: MiniMalloc is always attempted first. Deterministic
/// first-fit is consulted only after MiniMalloc reports ResourceExhausted.
/// Passing an explicit budget is an internal/offline and test control; normal
/// callers use the computed generous default.
PackingResult
packStaticMemory(const StaticPackingProblem &problem,
                 std::optional<uint64_t> searchNodeBudget = std::nullopt);

PackingResult
packStaticMemory(llvm::ArrayRef<LifetimeDemand> demands, ArenaRange arena,
                 std::optional<uint64_t> searchNodeBudget = std::nullopt);

/// Retained safety fallback and test oracle. A NoFit-like result from this
/// heuristic is not a proof that the fixed-capacity problem is infeasible.
PackingResult packFirstFit(const StaticPackingProblem &problem);

} // namespace wafer::memory_planning::detail

#endif // WAFER_TRANSFORMS_MEMORYPLANNING_STATICMEMORYPACKING_H
