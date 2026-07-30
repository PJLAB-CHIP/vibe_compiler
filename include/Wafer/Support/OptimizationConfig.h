//===- OptimizationConfig.h - Typed compiler optimization set --*- C++ -*-===//

#ifndef WAFER_SUPPORT_OPTIMIZATIONCONFIG_H
#define WAFER_SUPPORT_OPTIMIZATIONCONFIG_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace wafer {

/// Stable semantic alternatives owned by the production candidate pipeline.
/// These are not pass names: disabling one kind prevents that alternative
/// family from entering the bounded candidate domain while preserving all
/// mandatory normalization, legality, placement, lowering, and publication
/// gates.
enum class OptimizationKind : uint8_t {
  ConsumerLocalRecomputation,
  LoopInvariantCodeMotion,
  AlgebraicReassociation,
  ReductionTreeBalancing,
  AlgebraicDistribution,
  AlgebraicFactorization,
  ImplementationSelection,
  TileSearchAlternatives,
  ScopeComposition,
  CollectiveAlgorithmSelection,
  DirectMappedBoundaryTransfer,
  ConcurrentWorkingSetSelection,
  FullBufferTransferElision,
  FullBufferResidency,
  ReadyOrderScheduling,
  StaticFixedSlotBuffering,
  DisjointWorkerPlacement,
  NoCResidentDataflow,
  Count,
};

inline constexpr std::array<OptimizationKind, 18> kSupportedOptimizationKinds =
    {
        OptimizationKind::ConsumerLocalRecomputation,
        OptimizationKind::LoopInvariantCodeMotion,
        OptimizationKind::AlgebraicReassociation,
        OptimizationKind::ReductionTreeBalancing,
        OptimizationKind::AlgebraicDistribution,
        OptimizationKind::AlgebraicFactorization,
        OptimizationKind::ImplementationSelection,
        OptimizationKind::TileSearchAlternatives,
        OptimizationKind::ScopeComposition,
        OptimizationKind::CollectiveAlgorithmSelection,
        OptimizationKind::DirectMappedBoundaryTransfer,
        OptimizationKind::ConcurrentWorkingSetSelection,
        OptimizationKind::FullBufferTransferElision,
        OptimizationKind::FullBufferResidency,
        OptimizationKind::ReadyOrderScheduling,
        OptimizationKind::StaticFixedSlotBuffering,
        OptimizationKind::DisjointWorkerPlacement,
        OptimizationKind::NoCResidentDataflow,
};

inline llvm::ArrayRef<OptimizationKind> getSupportedOptimizationKinds() {
  return kSupportedOptimizationKinds;
}

inline llvm::StringRef stringifyOptimizationKind(OptimizationKind kind) {
  switch (kind) {
  case OptimizationKind::ConsumerLocalRecomputation:
    return "consumer-local-recomputation";
  case OptimizationKind::LoopInvariantCodeMotion:
    return "loop-invariant-code-motion";
  case OptimizationKind::AlgebraicReassociation:
    return "algebraic-reassociation";
  case OptimizationKind::ReductionTreeBalancing:
    return "reduction-tree-balancing";
  case OptimizationKind::AlgebraicDistribution:
    return "algebraic-distribution";
  case OptimizationKind::AlgebraicFactorization:
    return "algebraic-factorization";
  case OptimizationKind::ImplementationSelection:
    return "implementation-selection";
  case OptimizationKind::TileSearchAlternatives:
    return "tile-search-alternatives";
  case OptimizationKind::ScopeComposition:
    return "scope-composition";
  case OptimizationKind::CollectiveAlgorithmSelection:
    return "collective-algorithm-selection";
  case OptimizationKind::DirectMappedBoundaryTransfer:
    return "direct-mapped-boundary-transfer";
  case OptimizationKind::ConcurrentWorkingSetSelection:
    return "concurrent-working-set-selection";
  case OptimizationKind::FullBufferTransferElision:
    return "full-buffer-transfer-elision";
  case OptimizationKind::FullBufferResidency:
    return "full-buffer-residency";
  case OptimizationKind::ReadyOrderScheduling:
    return "ready-order-scheduling";
  case OptimizationKind::StaticFixedSlotBuffering:
    return "static-fixed-slot-buffering";
  case OptimizationKind::DisjointWorkerPlacement:
    return "disjoint-worker-placement";
  case OptimizationKind::NoCResidentDataflow:
    return "noc-resident-dataflow";
  case OptimizationKind::Count:
    return "";
  }
  return "";
}

inline std::optional<OptimizationKind>
parseOptimizationKind(llvm::StringRef name) {
  for (OptimizationKind kind : getSupportedOptimizationKinds())
    if (stringifyOptimizationKind(kind) == name)
      return kind;
  return std::nullopt;
}

/// Invocation-local set of optional semantic candidate producers.
class OptimizationConfig {
public:
  static constexpr OptimizationConfig production() {
    return OptimizationConfig(allMask());
  }
  static constexpr OptimizationConfig none() {
    return OptimizationConfig(uint64_t{0});
  }

  constexpr bool isEnabled(OptimizationKind kind) const {
    return kind != OptimizationKind::Count && (enabledMask & bit(kind)) != 0;
  }

  constexpr void enable(OptimizationKind kind) {
    if (kind != OptimizationKind::Count)
      enabledMask |= bit(kind);
  }
  constexpr void disable(OptimizationKind kind) {
    if (kind != OptimizationKind::Count)
      enabledMask &= ~bit(kind);
  }

  friend constexpr bool operator==(OptimizationConfig lhs,
                                   OptimizationConfig rhs) {
    return lhs.enabledMask == rhs.enabledMask;
  }
  friend constexpr bool operator!=(OptimizationConfig lhs,
                                   OptimizationConfig rhs) {
    return !(lhs == rhs);
  }

private:
  static constexpr uint64_t bit(OptimizationKind kind) {
    return uint64_t{1} << static_cast<uint8_t>(kind);
  }
  static constexpr uint64_t allMask() {
    return bit(OptimizationKind::Count) - 1;
  }

  explicit constexpr OptimizationConfig(uint64_t enabledMask)
      : enabledMask(enabledMask) {}

  uint64_t enabledMask;
};

static_assert(kSupportedOptimizationKinds.size() ==
              static_cast<size_t>(OptimizationKind::Count));
static_assert(static_cast<unsigned>(OptimizationKind::Count) < 64,
              "optimization bitset exceeds its fixed storage");

} // namespace wafer

#endif // WAFER_SUPPORT_OPTIMIZATIONCONFIG_H
