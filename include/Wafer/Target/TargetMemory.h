//===- TargetMemory.h - Fixed target memory constraints -------*- C++ -*-===//

#ifndef WAFER_TARGET_CORE_TARGETMEMORY_H
#define WAFER_TARGET_CORE_TARGETMEMORY_H

#include <cstdint>
#include <limits>

namespace wafer {

inline constexpr char kTargetKcoreReleaseScope[] = "wafer.kcore";
// The pinned C908 SDK encodes sync as 0x0180000b. LLVM's RISC-V assembler
// lacks that vendor mnemonic; emit the same instruction word.
inline constexpr char kTargetKcoreReleaseAssembly[] = "fence iorw, iorw\n.word 0x0180000b";

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
