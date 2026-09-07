//===- DirectDTE.h - Pure Direct DTE resource facts ---------*- C++ -*-===//

#ifndef WAFER_TARGET_DIRECTDTE_H
#define WAFER_TARGET_DIRECTDTE_H

#include <cstdint>

namespace wafer {

/// Current target-visible Direct DTE resources per Tile. These are hard ABI
/// facts shared by planning, binding and verification; they are not queue
/// depth estimates or profitability parameters.
struct TargetDirectDTEResourceLimits {
  static constexpr uint32_t senderSlotsPerTile = 1;
  static constexpr uint32_t receiverFSMsPerTile = 4;
  // direct_sync_post/wait share one non-counting notification per peer pair.
  static constexpr uint32_t receiverReadySlotsPerPeer = 1;
};

} // namespace wafer

#endif // WAFER_TARGET_DIRECTDTE_H
