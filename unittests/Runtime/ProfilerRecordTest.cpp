#include "Wafer/Runtime/ProfilerRecord.h"

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

constexpr uint64_t kRecordBytes = 1024;

std::vector<uint8_t> makeRecord(uint32_t tile, bool trace = true) {
  std::vector<uint8_t> bytes(kRecordBytes, 0);
  WaferTx81ProfilerRecordHeader header{};
  header.magic = WAFER_TX81_PROFILER_RECORD_MAGIC;
  header.schema_version = WAFER_TX81_PROFILER_SCHEMA_VERSION;
  header.header_bytes = WAFER_TX81_PROFILER_HEADER_BYTES;
  header.event_bytes = WAFER_TX81_PROFILER_TSM_CALL_EVENT_BYTES;
  header.events_offset = WAFER_TX81_PROFILER_EVENTS_OFFSET;
  header.buffer_bytes = bytes.size();
  header.buffer_address = UINT64_C(0x10000000) + tile * kRecordBytes;
  header.tile_id = tile;
  header.flags = WAFER_TX81_PROFILER_RECORD_ENTRY_BEGUN |
                 WAFER_TX81_PROFILER_RECORD_ENTRY_ENDED |
                 WAFER_TX81_PROFILER_RECORD_PUBLISHED;
  if (trace)
    header.flags |= WAFER_TX81_PROFILER_RECORD_TRACE_ENABLED;
  header.trace_state = WAFER_TX81_PROFILER_TRACE_COMPLETE;
  header.event_capacity = (bytes.size() - WAFER_TX81_PROFILER_HEADER_BYTES -
                           WAFER_TX81_PROFILER_BUFFER_GUARD_BYTES) /
                          WAFER_TX81_PROFILER_TSM_CALL_EVENT_BYTES;
  header.event_count = trace ? 1 : 0;
  header.next_sequence = header.event_count;
  header.entry_begin_cycle = 100;
  header.entry_end_cycle = 200;
  header.active_site_id = WAFER_TX81_PROFILER_INVALID_SITE_ID;
  header.summary_validity = WAFER_TX81_PROFILER_SUMMARY_ENTRY_CYCLES |
                            WAFER_TX81_PROFILER_SUMMARY_PMU_BEFORE_CAPTURED |
                            WAFER_TX81_PROFILER_SUMMARY_PMU_AFTER_CAPTURED |
                            WAFER_TX81_PROFILER_SUMMARY_PMU_RECOVERY_CAPTURED |
                            WAFER_TX81_PROFILER_SUMMARY_PMU_ENABLE_UNCHANGED |
                            WAFER_TX81_PROFILER_SUMMARY_PMU_RECOVERED;
  header.header_guard = WAFER_TX81_PROFILER_HEADER_GUARD;
  header.buffer_guard_offset =
      bytes.size() - WAFER_TX81_PROFILER_BUFFER_GUARD_BYTES;
  header.header_footer_guard = WAFER_TX81_PROFILER_HEADER_GUARD;
  std::memcpy(bytes.data(), &header, sizeof(header));

  if (trace) {
    WaferTx81ProfilerTSMCallEvent event{};
    event.sequence = 0;
    event.begin_cycle = 120;
    event.end_cycle = 130;
    // This remains an opaque adapter result, not a completion assertion.
    event.raw_return = UINT64_C(0x123456789abcdef0);
    event.site_id = 7;
    event.sub_index = 2;
    event.engine = WAFER_TX81_PROFILER_ENGINE_RDMA;
    event.metadata = WAFER_TX81_PROFILER_EVENT_SITE_VALID;
    std::memcpy(bytes.data() + WAFER_TX81_PROFILER_EVENTS_OFFSET, &event,
                sizeof(event));
  }
  uint64_t guard = WAFER_TX81_PROFILER_BUFFER_GUARD;
  std::memcpy(bytes.data() + header.buffer_guard_offset, &guard, sizeof(guard));
  return bytes;
}

TEST(ProfilerRecordTest, BuildsLaunchConfigurationWithoutExposingASchemaSlot) {
  auto image = wafer::runtime::buildTx81ProfilerLaunchImage(
      kRecordBytes, 5, wafer::runtime::Tx81ProfilerCaptureKind::Trace);
  ASSERT_TRUE(static_cast<bool>(image)) << llvm::toString(image.takeError());
  ASSERT_EQ(image->size(), kRecordBytes);
  WaferTx81ProfilerLaunchConfig config{};
  std::memcpy(&config, image->data(), sizeof(config));
  EXPECT_EQ(config.magic, WAFER_TX81_PROFILER_LAUNCH_CONFIG_MAGIC);
  EXPECT_EQ(config.schema_version, WAFER_TX81_PROFILER_SCHEMA_VERSION);
  EXPECT_EQ(config.config_bytes, WAFER_TX81_PROFILER_LAUNCH_CONFIG_BYTES);
  EXPECT_EQ(config.record_bytes, kRecordBytes);
  EXPECT_EQ(config.tile_id, 5u);
  EXPECT_EQ(config.flags, WAFER_TX81_PROFILER_ENTRY_TRACE_ENABLED);
  EXPECT_EQ(config.guard, WAFER_TX81_PROFILER_LAUNCH_CONFIG_GUARD);
  EXPECT_EQ(config.reserved[0], 0u);
  EXPECT_EQ((*image)[WAFER_TX81_PROFILER_LAUNCH_CONFIG_BYTES], UINT8_C(0xa5));
}

TEST(ProfilerRecordTest, CountOnlyCarriesRequiredCapacityWithoutEvents) {
  std::vector<uint8_t> bytes = makeRecord(0, /*trace=*/false);
  auto *header =
      reinterpret_cast<WaferTx81ProfilerRecordHeader *>(bytes.data());
  header->flags |= WAFER_TX81_PROFILER_RECORD_COUNT_ONLY;
  header->next_sequence = 19;
  auto decoded = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_TRUE(static_cast<bool>(decoded))
      << llvm::toString(decoded.takeError());
  EXPECT_TRUE(decoded->events.empty());
  EXPECT_EQ(decoded->header.next_sequence, 19u);

  header->flags |= WAFER_TX81_PROFILER_RECORD_TRACE_ENABLED;
  auto invalid = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(invalid));
  EXPECT_NE(llvm::toString(invalid.takeError()).find("both enabled"),
            std::string::npos);

  header->flags &= ~WAFER_TX81_PROFILER_RECORD_TRACE_ENABLED;
  header->event_count = 1;
  auto storedEvent = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(storedEvent));
  EXPECT_NE(llvm::toString(storedEvent.takeError()).find("stored events"),
            std::string::npos);
}

TEST(ProfilerRecordTest, DecodesRawEventAndPreservesUnknownWorker) {
  auto decoded = wafer::runtime::decodeTx81ProfilerRecord(makeRecord(3));
  ASSERT_TRUE(static_cast<bool>(decoded))
      << llvm::toString(decoded.takeError());
  ASSERT_EQ(decoded->events.size(), 1u);
  const auto &event = decoded->events.front();
  EXPECT_EQ(event.raw_return, UINT64_C(0x123456789abcdef0));
  EXPECT_TRUE(wafer::runtime::isTx81ProfilerSiteValid(event));
  EXPECT_FALSE(wafer::runtime::isTx81ProfilerWorkerValid(event));
  EXPECT_EQ(wafer::runtime::getTx81ProfilerWorker(event), 0u);
}

TEST(ProfilerRecordTest, RejectsGuardCorruption) {
  std::vector<uint8_t> bytes = makeRecord(0);
  bytes.back() ^= 1;
  auto decoded = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(decoded));
  EXPECT_NE(llvm::toString(decoded.takeError()).find("tail guard"),
            std::string::npos);
}

TEST(ProfilerRecordTest, RejectsInconsistentPMUValidity) {
  std::vector<uint8_t> bytes = makeRecord(0);
  auto *header =
      reinterpret_cast<WaferTx81ProfilerRecordHeader *>(bytes.data());
  header->summary_validity &= ~WAFER_TX81_PROFILER_SUMMARY_PMU_RECOVERED;
  auto decoded = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(decoded));
  EXPECT_NE(llvm::toString(decoded.takeError()).find("PMU validity"),
            std::string::npos);
}

TEST(ProfilerRecordTest, RequiresAllAndOnlySixteenCompleteTiles) {
  std::vector<wafer::runtime::Tx81ProfilerRecord> records;
  for (uint32_t tile = 0; tile < WAFER_TX81_PROFILER_TILE_COUNT; ++tile) {
    auto decoded = wafer::runtime::decodeTx81ProfilerRecord(makeRecord(tile));
    ASSERT_TRUE(static_cast<bool>(decoded))
        << llvm::toString(decoded.takeError());
    records.push_back(std::move(*decoded));
  }
  EXPECT_FALSE(
      static_cast<bool>(wafer::runtime::verifyTx81ProfilerTileDomain(records)));

  records.back().header.tile_id = 0;
  llvm::Error duplicate = wafer::runtime::verifyTx81ProfilerTileDomain(records);
  ASSERT_TRUE(static_cast<bool>(duplicate));
  EXPECT_NE(llvm::toString(std::move(duplicate)).find("duplicated"),
            std::string::npos);
}

TEST(ProfilerRecordTest, StructurallyDecodesButDomainRejectsOverflow) {
  std::vector<uint8_t> bytes = makeRecord(0);
  auto *header =
      reinterpret_cast<WaferTx81ProfilerRecordHeader *>(bytes.data());
  header->flags |= WAFER_TX81_PROFILER_RECORD_OVERFLOW;
  header->trace_state = WAFER_TX81_PROFILER_TRACE_OVERFLOW;
  header->dropped_event_count = 1;
  header->next_sequence = 2;
  auto decoded = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_TRUE(static_cast<bool>(decoded))
      << llvm::toString(decoded.takeError());

  std::vector<wafer::runtime::Tx81ProfilerRecord> records;
  records.push_back(std::move(*decoded));
  for (uint32_t tile = 1; tile < WAFER_TX81_PROFILER_TILE_COUNT; ++tile) {
    auto other = wafer::runtime::decodeTx81ProfilerRecord(makeRecord(tile));
    ASSERT_TRUE(static_cast<bool>(other)) << llvm::toString(other.takeError());
    records.push_back(std::move(*other));
  }
  llvm::Error error = wafer::runtime::verifyTx81ProfilerTileDomain(records);
  ASSERT_TRUE(static_cast<bool>(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("incomplete or invalid"),
            std::string::npos);
}

} // namespace
