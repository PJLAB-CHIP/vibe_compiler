//===- TargetPolicy.h - Wafer target and search policy ----------*- C++ -*-===//

#ifndef WAFER_SUPPORT_TARGETPOLICY_H
#define WAFER_SUPPORT_TARGETPOLICY_H

#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <limits>

namespace wafer {

enum class TileSearchEffort { Quick, Default, Deep };

struct TileSearchPolicy {
  llvm::SmallVector<int64_t, 8> preferredTileSizes;
  // Zero means keep all shape-driven refinement sizes for each dimension.
  int64_t maxCandidatesPerDim = 0;
  int64_t maxSearchCandidates = 16;
  int64_t searchBeamWidth = 8;
};

struct TargetMemoryPolicy {
  int64_t spmBase = 65536;
  int64_t spmLimit = 3080192;
  int64_t spmAlignment = 256;
  int64_t ddrCapacityBytes = std::numeric_limits<int64_t>::max();
  int64_t ddrLargestContiguousBytes = std::numeric_limits<int64_t>::max();
  int64_t ddrBandwidthLimitBytes = std::numeric_limits<int64_t>::max();
  int64_t ddrAlignmentBytes = 256;
};

struct TargetTimingPolicy {
  int64_t computeOpsPerCycle = 1024;
  int64_t ddrBytesPerCycle = 256;
  int64_t spmBytesPerCycle = 1024;
  int64_t instrIssueCycles = 1;
  bool assumeDdrComputeOverlap = false;
};

enum class TargetStaticTradeoffPolicy {
  /// Before calibrated timing is available, prefer a strict reduction in the
  /// highest-priority known resource class: DDR movement, NoC movement, SPM
  /// movement, issued work, then static dataflow depth/order. A tradeoff
  /// within one class, or any Unknown fact, remains conservative.
  ExternalMovementFirst,
  Conservative,
};

struct TargetStaticSelectionPolicy {
  TargetStaticTradeoffPolicy tradeoff =
      TargetStaticTradeoffPolicy::ExternalMovementFirst;
};

struct WaferTargetPolicy {
  TileSearchPolicy tileSearch;
  TargetMemoryPolicy memory;
  TargetTimingPolicy timing;
  TargetStaticSelectionPolicy staticSelection;
};

inline TileSearchPolicy getTileSearchPolicy(TileSearchEffort effort) {
  TileSearchPolicy policy;
  policy.preferredTileSizes = {128, 64, 32, 16, 8, 4, 2, 1};

  switch (effort) {
  case TileSearchEffort::Quick:
    policy.maxCandidatesPerDim = 0;
    policy.maxSearchCandidates = 8;
    policy.searchBeamWidth = 4;
    break;
  case TileSearchEffort::Default:
    policy.maxCandidatesPerDim = 0;
    policy.maxSearchCandidates = 16;
    policy.searchBeamWidth = 8;
    break;
  case TileSearchEffort::Deep:
    policy.maxCandidatesPerDim = 0;
    policy.maxSearchCandidates = 64;
    policy.searchBeamWidth = 16;
    break;
  }

  return policy;
}

inline WaferTargetPolicy getDefaultWaferTargetPolicy(
    TileSearchEffort effort = TileSearchEffort::Default) {
  WaferTargetPolicy policy;
  policy.tileSearch = getTileSearchPolicy(effort);
  return policy;
}

} // namespace wafer

#endif // WAFER_SUPPORT_TARGETPOLICY_H
