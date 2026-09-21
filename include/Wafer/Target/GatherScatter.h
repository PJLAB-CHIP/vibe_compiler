//===- GatherScatter.h - TX81 GS issue granularity -------------*- C++ -*-===//

#ifndef WAFER_TARGET_GATHERSCATTER_H
#define WAFER_TARGET_GATHERSCATTER_H

#include <cstdint>

namespace wafer::target {

/// Compiler issue policy, not the iteration field width or a watchdog timing
/// formula. Tight-issue board qualification is tracked separately from address
/// correctness. Keep the policy shared by materialization and target checking.
inline constexpr int64_t kGatherScatterMaxInnerTransfers = 16384;
inline constexpr int64_t kGatherScatterMaxPayloadBytes = 1024 * 1024;

inline bool isGatherScatterIssueBounded(int64_t bytes, int64_t innerBytes) {
  return bytes > 0 && innerBytes > 0 && bytes % innerBytes == 0 &&
         bytes <= kGatherScatterMaxPayloadBytes &&
         bytes / innerBytes <= kGatherScatterMaxInnerTransfers;
}

} // namespace wafer::target

#endif // WAFER_TARGET_GATHERSCATTER_H
