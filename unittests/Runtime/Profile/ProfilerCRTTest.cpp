// Execute the production record state machine with only hardware I/O mocked.
#include "Wafer/ABI/Tx81NCCABI.h"
#include "Wafer/Runtime/Profile/ProfilerRecord.h"
#include "instr_def.h"
#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace {

std::array<uint32_t, 1024> nccRegisters{};
std::array<uint32_t, 1024> dteRegisters{};
uint64_t modelCycle = 1;
uint64_t issueCalls = 0;
uint64_t statusPolls = 0;

static uint64_t wafer_profile_cycle() { return ++modelCycle; }
static uint32_t wafer_profile_read_pmu32(uint32_t offset) {
  return nccRegisters[offset / 4];
}
static uint32_t wafer_profile_read_dte_pmu32(uint32_t offset) {
  return dteRegisters[offset / 4];
}
static void wafer_profile_write_pmu32(uint32_t offset, uint32_t value) {
  nccRegisters[offset / 4] = value;
}
static void wafer_profile_write_dte_pmu32(uint32_t offset, uint32_t value) {
  dteRegisters[offset / 4] = value;
}
static void wafer_profile_cache_range(uint64_t, uint64_t, uint32_t) {
  ++modelCycle;
}
static void wafer_profile_fence_io() { ++modelCycle; }
static uint64_t TsmExecute(void *) {
  ++issueCalls;
  modelCycle += 3;
  nccRegisters[GR_PMU_RDMA_EXE_TIME / 4] += 5;
  return 17;
}
static uint32_t TsmWaitfinish_bywork(uint32_t) { return 1; }
static uint32_t TsmGetCsrTaskstatus_bywork(uint32_t) {
  modelCycle += 2;
  return ++statusPolls % 3 == 0;
}
// The profiler receives these private CRT event identities, never pointers.
enum {
  WAFER_DIRECT_DTE_SEND_EVENT = 0x100,
  WAFER_DIRECT_DTE_RECV_EVENT_BASE = 0x200,
  WAFER_DIRECT_DTE_MAX_RECEIVERS = 4,
};

#include "wafer_tx81_profiler_impl.inc"

void executeProfiledProgram(uint32_t iterations) {
  for (uint32_t i = 0; i < iterations; ++i) {
    wafer_tx81_profile_site_begin(7);
    EXPECT_EQ(
        wafer_profile_execute_ncc(nullptr, 0, WAFER_TX81_PROFILER_ENGINE_RDMA),
        17u);
    EXPECT_EQ(wafer_profile_wait_ncc_worker_completion(0), 1u);
    wafer_profile_direct_dte_begin(WAFER_DIRECT_DTE_SEND_EVENT,
                                   WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_ISSUE);
    for (uint8_t kind : {WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_PEER_READY_WAIT,
                         WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_SETUP_ISSUE}) {
      uint32_t event = wafer_profile_direct_dte_phase_begin(kind);
      modelCycle += 2;
      wafer_profile_direct_dte_phase_end(event);
    }
    wafer_profile_direct_dte_end();
    wafer_profile_direct_dte_begin(WAFER_DIRECT_DTE_SEND_EVENT,
                                   WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_WAIT);
    for (uint8_t kind : {WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_COMPLETION_WAIT,
                         WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_CLEANUP}) {
      uint32_t event = wafer_profile_direct_dte_phase_begin(kind);
      modelCycle += 2;
      wafer_profile_direct_dte_phase_end(event);
    }
    wafer_profile_direct_dte_end();
    wafer_tx81_profile_site_end(7);
  }
}

TEST(ProfilerCRTTest, PrefixPreservesCommandsWaitsAndDTEStateOutsideTheRecord) {
  for (uint32_t iterations : {1024u, 1025u, 1031u})
    for (uint32_t limit : {0u, 1u, 4u, 5u, 7u, 8u, 1024u}) {
      SCOPED_TRACE(iterations);
      SCOPED_TRACE(limit);
      nccRegisters.fill(0);
      dteRegisters.fill(0);
      nccRegisters[GR_PMU_EN / 4] = 0x10;
      dteRegisters[DTE_PMU_EN / 4] = 0x40;
      modelCycle = 1;
      issueCalls = statusPolls = 0;
      const uint32_t total = iterations * 9;
      const uint32_t count = limit == 0 ? total : limit;
      const size_t bytes = (WAFER_TX81_PROFILER_HEADER_BYTES +
                            static_cast<size_t>(count) *
                                WAFER_TX81_PROFILER_TSM_CALL_EVENT_BYTES +
                            WAFER_TX81_PROFILER_BUFFER_GUARD_BYTES + 63) /
                           64 * 64;
      auto image = wafer::runtime::buildTx81ProfilerLaunchImage(
          bytes, 0, wafer::runtime::Tx81ProfilerCaptureKind::Trace, limit);
      ASSERT_TRUE(static_cast<bool>(image))
          << llvm::toString(image.takeError());
      std::unique_ptr<void, decltype(&std::free)> storage(
          std::aligned_alloc(64, bytes), &std::free);
      ASSERT_TRUE(storage);
      std::memcpy(storage.get(), image->data(), bytes);
      wafer_tx81_profile_entry_begin_from_config(
          reinterpret_cast<uint64_t>(storage.get()));
      executeProfiledProgram(iterations);
      wafer_tx81_profile_entry_end();
      EXPECT_EQ(issueCalls, iterations);
      EXPECT_EQ(statusPolls, iterations * 3u);
      EXPECT_EQ(nccRegisters[GR_PMU_EN / 4], 0x10u);
      EXPECT_EQ(dteRegisters[DTE_PMU_EN / 4], 0x40u);
      auto decoded = wafer::runtime::decodeTx81ProfilerRecord(
          llvm::ArrayRef(static_cast<const uint8_t *>(storage.get()), bytes));
      ASSERT_TRUE(static_cast<bool>(decoded))
          << llvm::toString(decoded.takeError());
      EXPECT_EQ(decoded->header.trace_event_limit, limit);
      EXPECT_EQ(decoded->header.next_sequence, total);
      EXPECT_EQ(decoded->header.event_count, count);
      EXPECT_EQ(decoded->header.dropped_event_count, 0u);
      EXPECT_EQ(decoded->header.trace_state,
                WAFER_TX81_PROFILER_TRACE_COMPLETE);
      EXPECT_EQ(decoded->header.flags &
                    WAFER_TX81_PROFILER_RECORD_SITE_PROTOCOL_ERROR,
                0u);
    }
}

TEST(ProfilerCRTTest, CountExecutesTheSameProgramWithoutStoredEvents) {
  for (uint32_t iterations : {1024u, 1025u, 1031u}) {
    nccRegisters.fill(0);
    dteRegisters.fill(0);
    nccRegisters[GR_PMU_EN / 4] = 0x10;
    dteRegisters[DTE_PMU_EN / 4] = 0x40;
    issueCalls = statusPolls = 0;
    constexpr size_t bytes = WAFER_TX81_PROFILER_MIN_BUFFER_BYTES;
    auto image = wafer::runtime::buildTx81ProfilerLaunchImage(
        bytes, 0, wafer::runtime::Tx81ProfilerCaptureKind::Count);
    ASSERT_TRUE(static_cast<bool>(image));
    std::unique_ptr<void, decltype(&std::free)> storage(
        std::aligned_alloc(64, bytes), &std::free);
    ASSERT_TRUE(storage);
    std::memcpy(storage.get(), image->data(), bytes);
    wafer_tx81_profile_entry_begin_from_config(
        reinterpret_cast<uint64_t>(storage.get()));
    executeProfiledProgram(iterations);
    wafer_tx81_profile_entry_end();
    EXPECT_EQ(issueCalls, iterations);
    EXPECT_EQ(statusPolls, 0u);
    EXPECT_EQ(nccRegisters[GR_PMU_EN / 4], 0x10u);
    EXPECT_EQ(dteRegisters[DTE_PMU_EN / 4], 0x40u);
    auto decoded = wafer::runtime::decodeTx81ProfilerRecord(
        llvm::ArrayRef(static_cast<const uint8_t *>(storage.get()), bytes));
    ASSERT_TRUE(static_cast<bool>(decoded))
        << llvm::toString(decoded.takeError());
    EXPECT_EQ(decoded->header.next_sequence, iterations * 9u);
    EXPECT_TRUE(decoded->events.empty());
  }
}

TEST(ProfilerCRTTest, UnrecordedDTEStillRequiresItsMatchingEnd) {
  constexpr size_t bytes = 896; // One event: isolate the missing-end fault.
  auto image = wafer::runtime::buildTx81ProfilerLaunchImage(
      bytes, 0, wafer::runtime::Tx81ProfilerCaptureKind::Trace, 1);
  ASSERT_TRUE(static_cast<bool>(image));
  std::unique_ptr<void, decltype(&std::free)> storage(
      std::aligned_alloc(64, bytes), &std::free);
  ASSERT_TRUE(storage);
  std::memcpy(storage.get(), image->data(), bytes);
  wafer_tx81_profile_entry_begin_from_config(
      reinterpret_cast<uint64_t>(storage.get()));
  wafer_tx81_profile_site_begin(7);
  wafer_profile_direct_dte_begin(WAFER_DIRECT_DTE_SEND_EVENT,
                                 WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_ISSUE);
  wafer_tx81_profile_site_end(7);
  wafer_tx81_profile_entry_end();
  auto *header =
      static_cast<const WaferTx81ProfilerRecordHeader *>(storage.get());
  EXPECT_EQ(header->trace_state, WAFER_TX81_PROFILER_TRACE_INVALID);
  EXPECT_NE(header->flags & WAFER_TX81_PROFILER_RECORD_SITE_PROTOCOL_ERROR, 0u);
}

} // namespace
