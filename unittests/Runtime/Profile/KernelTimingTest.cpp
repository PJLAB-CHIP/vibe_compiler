#include "Wafer/Runtime/Profile/ProfilerRecord.h"

#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cstring>

namespace {
using namespace wafer::runtime;

uint64_t clockMicroseconds = 0;
unsigned clockReads = 0;
unsigned cacheInvalidations = 0;
unsigned cacheWritebacks = 0;
uint64_t wafer_profile_microseconds() {
  ++clockReads;
  return clockMicroseconds;
}
void wafer_profile_cache_range(uint64_t, uint64_t bytes, uint32_t invalidate) {
  EXPECT_EQ(bytes, WAFER_TX81_KERNEL_TIMING_RECORD_BYTES);
  if (invalidate)
    ++cacheInvalidations;
  else
    ++cacheWritebacks;
}

// The production timing state machine has no PMU or command/wait dependencies.
#include "wafer_tx81_kernel_timing_impl.inc"

TEST(KernelTimingTest, CapturesOnlyEndpointsAndRebindsEveryInvocation) {
  alignas(64) std::array<uint8_t, 64> buffer{};
  std::vector<WaferTx81KernelTimingRecord> records;
  clockReads = cacheInvalidations = cacheWritebacks = 0;
  for (uint32_t tile = 0; tile < 16; ++tile) {
    auto image = buildTx81ProfilerLaunchImage(buffer.size(), tile,
                                              Tx81ProfilerCaptureKind::Timing);
    ASSERT_TRUE(static_cast<bool>(image)) << llvm::toString(image.takeError());
    std::copy(image->begin(), image->end(), buffer.begin());
    clockMicroseconds = 1000000 + tile * 100000;
    wafer_tx81_profile_entry_begin_from_config(
        reinterpret_cast<uintptr_t>(buffer.data()));
    // Different clock origins intentionally cannot be treated as one timeline.
    clockMicroseconds += 1024 + tile;
    wafer_tx81_profile_entry_end();
    auto decoded = decodeTx81KernelTimingRecord(buffer);
    ASSERT_TRUE(static_cast<bool>(decoded))
        << llvm::toString(decoded.takeError());
    EXPECT_EQ(decoded->tile_id, tile);
    EXPECT_EQ(decoded->entry_end_us - decoded->entry_begin_us, 1024 + tile);
    records.push_back(*decoded);
  }
  EXPECT_EQ(clockReads, 32u);
  EXPECT_EQ(cacheInvalidations, 16u);
  EXPECT_EQ(cacheWritebacks, 16u);
  std::reverse(records.begin(), records.end());
  auto summary = summarizeTx81KernelTiming(records);
  ASSERT_TRUE(static_cast<bool>(summary))
      << llvm::toString(summary.takeError());
  EXPECT_EQ(summary->longestTile, 15u);
  EXPECT_EQ(summary->longestTileMicroseconds, 1039u);
  EXPECT_EQ(summary->tileMicroseconds[1], 1025u);
  EXPECT_EQ(summary->tileMicroseconds[7], 1031u);

  const auto previous = buffer;
  wafer_tx81_profile_entry_end();
  EXPECT_EQ(buffer, previous);
  // A rejected next config must not retain a pointer to the prior record.
  wafer_tx81_profile_entry_begin_from_config(
      reinterpret_cast<uintptr_t>(buffer.data()));
  wafer_tx81_profile_entry_end();
  EXPECT_EQ(buffer, previous);
  EXPECT_EQ(cacheWritebacks, 16u);

  records[0] = records[1];
  auto duplicate = summarizeTx81KernelTiming(records);
  ASSERT_FALSE(static_cast<bool>(duplicate));
  llvm::consumeError(duplicate.takeError());
  records.pop_back();
  auto missing = summarizeTx81KernelTiming(records);
  ASSERT_FALSE(static_cast<bool>(missing));
  llvm::consumeError(missing.takeError());
}

TEST(KernelTimingTest, RejectsWrongKindSizeAndIncompleteOrCorruptedRecord) {
  alignas(64) std::array<uint8_t, 64> buffer{};
  auto image = buildTx81ProfilerLaunchImage(buffer.size(), 0,
                                            Tx81ProfilerCaptureKind::Timing);
  ASSERT_TRUE(static_cast<bool>(image)) << llvm::toString(image.takeError());
  std::copy(image->begin(), image->end(), buffer.begin());
  auto incomplete = decodeTx81KernelTimingRecord(buffer);
  ASSERT_FALSE(static_cast<bool>(incomplete));
  llvm::consumeError(incomplete.takeError());
  clockMicroseconds = 200;
  wafer_tx81_profile_entry_begin_from_config(
      reinterpret_cast<uintptr_t>(buffer.data()));
  clockMicroseconds = 250;
  wafer_tx81_profile_entry_end();
  for (unsigned offset : {0u, 8u, 12u, 32u, 56u}) {
    auto corrupted = buffer;
    corrupted[offset] = 0xff;
    auto result = decodeTx81KernelTimingRecord(corrupted);
    ASSERT_FALSE(static_cast<bool>(result)) << offset;
    llvm::consumeError(result.takeError());
  }
  buffer[24] = 100; // End precedes start.
  auto reversed = decodeTx81KernelTimingRecord(buffer);
  ASSERT_FALSE(static_cast<bool>(reversed));
  llvm::consumeError(reversed.takeError());
  auto truncated =
      decodeTx81KernelTimingRecord(llvm::ArrayRef(buffer).drop_back());
  ASSERT_FALSE(static_cast<bool>(truncated));
  llvm::consumeError(truncated.takeError());
  for (auto args : {std::pair{64u, 1u}, std::pair{832u, 0u}}) {
    auto rejected = buildTx81ProfilerLaunchImage(
        args.first, 0, Tx81ProfilerCaptureKind::Timing, args.second);
    ASSERT_FALSE(static_cast<bool>(rejected));
    llvm::consumeError(rejected.takeError());
  }
}
} // namespace
