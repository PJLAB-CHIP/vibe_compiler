//===- StaticMemoryPacking.h - Static MiniMalloc contract -*- C++ -*-===//

#ifndef WAFER_TRANSFORMS_INSTR_STATICMEMORYPACKING_H
#define WAFER_TRANSFORMS_INSTR_STATICMEMORYPACKING_H

#include "Wafer/Transforms/Instr/LifetimeAnalysis.h"

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

/// Exact static packing input derived from actual allocation lifetimes.
/// Demand indices match the LifetimeDemand order; stableOrdinal provides a
/// permutation-independent deterministic key.
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
  InvalidProblem,
  ArithmeticOverflow,
  InvalidSolverResult,
};

struct PackingResult {
  PackingStatus status = PackingStatus::InvalidProblem;
  llvm::SmallVector<Placement, 8> placements;
  uint64_t searchNodes = 0;
  std::optional<unsigned> demandIndex;
  /// Deterministic exact capacity certificate. Every listed demand belongs to
  /// one conflict-graph clique whose byte sum exceeds the usable arena.
  llvm::SmallVector<unsigned, 8> capacityConflictDemandIndices;
  /// Actual demands that individually exceed the usable arena.
  llvm::SmallVector<unsigned, 8> individuallyOversizedDemandIndices;

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

/// Runs MiniMalloc on one exact problem. Resource exhaustion is returned as
/// indeterminate and never invokes another allocator or heuristic fallback.
PackingResult
packStaticMemory(const StaticPackingProblem &problem,
                 std::optional<uint64_t> searchNodeBudget = std::nullopt);

PackingResult
packStaticMemory(llvm::ArrayRef<LifetimeDemand> demands, ArenaRange arena,
                 std::optional<uint64_t> searchNodeBudget = std::nullopt);

} // namespace wafer::memory_planning::detail

#endif // WAFER_TRANSFORMS_INSTR_STATICMEMORYPACKING_H
