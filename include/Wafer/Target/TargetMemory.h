//===- TargetMemory.h - Fixed target memory constraints -------*- C++ -*-===//

#ifndef WAFER_TARGET_CORE_TARGETMEMORY_H
#define WAFER_TARGET_CORE_TARGETMEMORY_H

#include <cstdint>
#include <limits>

namespace wafer {

/// Hard address, capacity and alignment facts consumed by legality and
/// allocation. Search budgets, preferred shapes and performance estimates do
/// not belong to this contract.
struct TargetMemoryPolicy {
  int64_t spmBase = 65536;
  int64_t spmLimit = 3080192;
  int64_t spmAlignment = 256;
  int64_t ddrCapacityBytes = std::numeric_limits<int64_t>::max();
  int64_t ddrLargestContiguousBytes = std::numeric_limits<int64_t>::max();
  int64_t ddrBandwidthLimitBytes = std::numeric_limits<int64_t>::max();
  int64_t ddrAlignmentBytes = 256;
};

inline TargetMemoryPolicy getTargetMemoryPolicy() { return {}; }

} // namespace wafer

#endif // WAFER_TARGET_CORE_TARGETMEMORY_H
