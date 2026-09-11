#include "Wafer/Runtime/Profile/ProfilerRecord.h"

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <vector>

namespace {

constexpr uint64_t kRecordBytes = 1024;

WaferTx81ProfilerTSMCallEvent makeTargetSiteEvent(uint32_t siteId = 7,
                                                  uint64_t siteBegin = 110,
                                                  uint64_t siteEnd = 150) {
  WaferTx81ProfilerTSMCallEvent event{};
  event.site_begin_cycle = siteBegin;
  event.site_end_cycle = siteEnd;
  event.site_id = siteId;
  event.sub_index = 0;
  event.engine = WAFER_TX81_PROFILER_ENGINE_NONE;
  event.kind = WAFER_TX81_PROFILER_EVENT_TARGET_SITE;
  event.metadata = WAFER_TX81_PROFILER_EVENT_SITE_VALID |
                   WAFER_TX81_PROFILER_EVENT_SITE_SPAN_VALID;
  return event;
}

std::vector<uint8_t> makeRecord(uint32_t tile, bool trace = true) {
  std::vector<uint8_t> bytes(kRecordBytes, 0);
  WaferTx81ProfilerRecordHeader header{};
  header.magic = WAFER_TX81_PROFILER_RECORD_MAGIC;
  header.header_bytes = WAFER_TX81_PROFILER_HEADER_BYTES;
  header.event_bytes = WAFER_TX81_PROFILER_TSM_CALL_EVENT_BYTES;
  header.events_offset = WAFER_TX81_PROFILER_EVENTS_OFFSET;
  header.buffer_bytes = bytes.size();
  header.buffer_address = UINT64_C(0x10000000) + tile * kRecordBytes;
  header.tile_id = tile;
  header.flags = WAFER_TX81_PROFILER_RECORD_ENTRY_BEGUN |
                 WAFER_TX81_PROFILER_RECORD_ENTRY_ENDED |
                 WAFER_TX81_PROFILER_RECORD_COMPLETE;
  if (trace)
    header.flags |= WAFER_TX81_PROFILER_RECORD_TRACE_ENABLED;
  else
    header.flags |= WAFER_TX81_PROFILER_RECORD_COUNT_ONLY;
  header.trace_state = WAFER_TX81_PROFILER_TRACE_COMPLETE;
  header.event_capacity = (bytes.size() - WAFER_TX81_PROFILER_HEADER_BYTES -
                           WAFER_TX81_PROFILER_BUFFER_GUARD_BYTES) /
                          WAFER_TX81_PROFILER_TSM_CALL_EVENT_BYTES;
  header.event_count = trace ? 2 : 0;
  header.next_sequence = header.event_count;
  header.entry_begin_cycle = 100;
  header.entry_end_cycle = 200;
  header.active_site_id = WAFER_TX81_PROFILER_INVALID_SITE_ID;
  header.summary_validity = WAFER_TX81_PROFILER_SUMMARY_ENTRY_CYCLES;
  if (trace)
    header.summary_validity |=
        WAFER_TX81_PROFILER_SUMMARY_PMU_BEFORE_CAPTURED |
        WAFER_TX81_PROFILER_SUMMARY_PMU_AFTER_CAPTURED |
        WAFER_TX81_PROFILER_SUMMARY_PMU_RECOVERY_CAPTURED |
        WAFER_TX81_PROFILER_SUMMARY_PMU_ENABLE_UNCHANGED |
        WAFER_TX81_PROFILER_SUMMARY_PMU_RECOVERED |
        WAFER_TX81_PROFILER_SUMMARY_NCC_PMU_RESTORE_VERIFIED |
        WAFER_TX81_PROFILER_SUMMARY_DTE_PMU_RESTORE_VERIFIED;
  header.header_guard = WAFER_TX81_PROFILER_HEADER_GUARD;
  header.buffer_guard_offset =
      bytes.size() - WAFER_TX81_PROFILER_BUFFER_GUARD_BYTES;
  header.header_footer_guard = WAFER_TX81_PROFILER_HEADER_GUARD;
  std::memcpy(bytes.data(), &header, sizeof(header));

  if (trace) {
    WaferTx81ProfilerTSMCallEvent container = makeTargetSiteEvent();
    container.sequence = 0;
    std::memcpy(bytes.data() + WAFER_TX81_PROFILER_EVENTS_OFFSET, &container,
                sizeof(container));

    WaferTx81ProfilerTSMCallEvent event{};
    event.sequence = 1;
    event.observed_begin_cycle = 112;
    event.observed_end_cycle = 135;
    event.counter_delta = 7;
    event.site_begin_cycle = 110;
    event.site_end_cycle = 150;
    event.operation_begin_cycle = 118;
    event.operation_end_cycle = 122;
    event.site_id = 7;
    event.observation_count = 2;
    event.sub_index = 1;
    event.engine = WAFER_TX81_PROFILER_ENGINE_RDMA;
    event.kind = WAFER_TX81_PROFILER_EVENT_NCC_COMMAND;
    event.metadata = WAFER_TX81_PROFILER_EVENT_SITE_VALID |
                     WAFER_TX81_PROFILER_EVENT_OBSERVATION_SPAN_VALID |
                     WAFER_TX81_PROFILER_EVENT_SITE_SPAN_VALID |
                     WAFER_TX81_PROFILER_EVENT_OPERATION_SPAN_VALID |
                     WAFER_TX81_PROFILER_EVENT_COUNTER_DELTA_POSITIVE |
                     WAFER_TX81_PROFILER_EVENT_NCC_COUNTER_VALID;
    std::memcpy(bytes.data() + WAFER_TX81_PROFILER_EVENTS_OFFSET +
                    sizeof(container),
                &event, sizeof(event));
  }
  uint64_t guard = WAFER_TX81_PROFILER_BUFFER_GUARD;
  std::memcpy(bytes.data() + header.buffer_guard_offset, &guard, sizeof(guard));
  return bytes;
}

WaferTx81ProfilerTSMCallEvent *eventAt(std::vector<uint8_t> &bytes,
                                       size_t index) {
  return reinterpret_cast<WaferTx81ProfilerTSMCallEvent *>(
             bytes.data() + WAFER_TX81_PROFILER_EVENTS_OFFSET) +
         index;
}

WaferTx81ProfilerTSMCallEvent makeDirectDTEEvent(uint8_t kind, uint32_t role,
                                                 uint64_t operationBegin,
                                                 uint64_t operationEnd,
                                                 uint16_t subIndex) {
  WaferTx81ProfilerTSMCallEvent event{};
  event.site_begin_cycle = 110;
  event.site_end_cycle = 150;
  event.operation_begin_cycle = operationBegin;
  event.operation_end_cycle = operationEnd;
  event.site_id = 7;
  event.sub_index = subIndex;
  event.engine = WAFER_TX81_PROFILER_ENGINE_DIRECT_DTE;
  event.kind = kind;
  event.metadata = WAFER_TX81_PROFILER_EVENT_SITE_VALID |
                   WAFER_TX81_PROFILER_EVENT_SITE_SPAN_VALID |
                   WAFER_TX81_PROFILER_EVENT_OPERATION_SPAN_VALID | role;
  if (kind == WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_ISSUE ||
      kind == WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_WAIT) {
    event.observed_begin_cycle = 112;
    event.observed_end_cycle = 145;
    event.counter_delta = 7;
    event.observation_count = 2;
    event.metadata |= WAFER_TX81_PROFILER_EVENT_OBSERVATION_SPAN_VALID |
                      WAFER_TX81_PROFILER_EVENT_COUNTER_DELTA_POSITIVE |
                      WAFER_TX81_PROFILER_EVENT_DTE_COUNTER_VALID;
  }
  return event;
}

void setEvents(std::vector<uint8_t> &bytes,
               std::initializer_list<WaferTx81ProfilerTSMCallEvent> newEvents) {
  auto *header =
      reinterpret_cast<WaferTx81ProfilerRecordHeader *>(bytes.data());
  ASSERT_LE(newEvents.size(), header->event_capacity);
  header->event_count = newEvents.size();
  header->next_sequence = newEvents.size();
  size_t index = 0;
  for (WaferTx81ProfilerTSMCallEvent event : newEvents) {
    event.sequence = index;
    std::memcpy(bytes.data() + WAFER_TX81_PROFILER_EVENTS_OFFSET +
                    index * sizeof(event),
                &event, sizeof(event));
    ++index;
  }
}

TEST(ProfilerRecordTest, BuildsLaunchConfigurationWithoutExposingASchemaSlot) {
  auto image = wafer::runtime::buildTx81ProfilerLaunchImage(
      kRecordBytes, 5, wafer::runtime::Tx81ProfilerCaptureKind::Trace);
  ASSERT_TRUE(static_cast<bool>(image)) << llvm::toString(image.takeError());
  ASSERT_EQ(image->size(), kRecordBytes);
  WaferTx81ProfilerLaunchConfig config{};
  std::memcpy(&config, image->data(), sizeof(config));
  EXPECT_EQ(config.magic, WAFER_TX81_PROFILER_LAUNCH_CONFIG_MAGIC);
  EXPECT_EQ(config.reserved0, 0u);
  EXPECT_EQ(config.config_bytes, WAFER_TX81_PROFILER_LAUNCH_CONFIG_BYTES);
  EXPECT_EQ(config.record_bytes, kRecordBytes);
  EXPECT_EQ(config.tile_id, 5u);
  EXPECT_EQ(config.flags, WAFER_TX81_PROFILER_ENTRY_TRACE_ENABLED);
  EXPECT_EQ(config.guard, WAFER_TX81_PROFILER_LAUNCH_CONFIG_GUARD);
  EXPECT_EQ(config.reserved[0], 0u);
  EXPECT_EQ((*image)[WAFER_TX81_PROFILER_LAUNCH_CONFIG_BYTES], UINT8_C(0xa5));
}

TEST(ProfilerRecordTest, BoundedPrefixKeepsTheCompleteDynamicCount) {
  for (uint32_t total : {1024u, 1025u, 1031u})
    for (uint32_t limit : {0u, 1024u, 1025u}) {
      SCOPED_TRACE(total);
      SCOPED_TRACE(limit);
      auto original = makeRecord(0);
      auto command = *eventAt(original, 1);
      WaferTx81ProfilerRecordHeader header{};
      std::memcpy(&header, original.data(), sizeof(header));
      uint32_t count = limit == 0 ? total : std::min(total, limit);
      uint64_t recordBytes =
          (WAFER_TX81_PROFILER_HEADER_BYTES +
           static_cast<uint64_t>(std::max(count, limit)) * sizeof(command) +
           WAFER_TX81_PROFILER_BUFFER_GUARD_BYTES + 63) /
          64 * 64;
      auto image = wafer::runtime::buildTx81ProfilerLaunchImage(
          recordBytes, 0, wafer::runtime::Tx81ProfilerCaptureKind::Trace,
          limit);
      ASSERT_TRUE(static_cast<bool>(image))
          << llvm::toString(image.takeError());
      WaferTx81ProfilerLaunchConfig config{};
      std::memcpy(&config, image->data(), sizeof(config));
      EXPECT_EQ(config.trace_event_limit, limit);
      header.buffer_bytes = recordBytes;
      header.buffer_guard_offset = recordBytes - sizeof(uint64_t);
      header.event_capacity =
          (recordBytes - WAFER_TX81_PROFILER_EVENTS_OFFSET - sizeof(uint64_t)) /
          sizeof(command);
      header.trace_event_limit = limit;
      header.event_count = count;
      header.next_sequence = total;
      header.entry_end_cycle = 200 + total * 10;
      std::memcpy(image->data(), &header, sizeof(header));
      uint64_t guard = WAFER_TX81_PROFILER_BUFFER_GUARD;
      std::memcpy(image->data() + header.buffer_guard_offset, &guard,
                  sizeof(guard));
      for (uint32_t i = 0; i < count; ++i) {
        uint64_t begin = 110 + (i / 2) * 10;
        auto event =
            i % 2 == 0 ? makeTargetSiteEvent(7, begin, begin + 8) : command;
        event.sequence = i;
        if (i % 2 != 0) {
          event.site_begin_cycle = begin;
          event.site_end_cycle = begin + 8;
          event.observed_begin_cycle = begin + 1;
          event.observed_end_cycle = begin + 6;
          event.operation_begin_cycle = begin + 2;
          event.operation_end_cycle = begin + 4;
        }
        std::memcpy(image->data() + WAFER_TX81_PROFILER_EVENTS_OFFSET +
                        i * sizeof(event),
                    &event, sizeof(event));
      }
      auto decoded = wafer::runtime::decodeTx81ProfilerRecord(*image);
      ASSERT_TRUE(static_cast<bool>(decoded))
          << llvm::toString(decoded.takeError());
      EXPECT_EQ(decoded->events.size(), count);
      EXPECT_EQ(decoded->header.next_sequence, total);
      EXPECT_EQ(decoded->header.dropped_event_count, 0u);
      for (uint32_t i = 0; i < count; ++i)
        EXPECT_EQ(decoded->events[i].sequence, i);
      auto *stored =
          reinterpret_cast<WaferTx81ProfilerRecordHeader *>(image->data());
      stored->trace_event_limit = header.event_capacity + 1;
      auto badLimit = wafer::runtime::decodeTx81ProfilerRecord(*image);
      ASSERT_FALSE(static_cast<bool>(badLimit));
      llvm::consumeError(badLimit.takeError());
      stored->trace_event_limit = limit;
      --stored->event_count;
      auto missing = wafer::runtime::decodeTx81ProfilerRecord(*image);
      ASSERT_FALSE(static_cast<bool>(missing));
      llvm::consumeError(missing.takeError());
    }
}

TEST(ProfilerRecordTest, InvalidPrefixIsRejectedBeforeBuildingTheImage) {
  auto capacity = wafer::runtime::getTx81ProfilerEventCapacity(kRecordBytes);
  ASSERT_TRUE(static_cast<bool>(capacity));
  for (auto kind : {wafer::runtime::Tx81ProfilerCaptureKind::Count,
                    wafer::runtime::Tx81ProfilerCaptureKind::Trace}) {
    auto bad = wafer::runtime::buildTx81ProfilerLaunchImage(
        kRecordBytes, 0, kind,
        kind == wafer::runtime::Tx81ProfilerCaptureKind::Count ? 1
                                                               : *capacity + 1);
    ASSERT_FALSE(static_cast<bool>(bad));
    llvm::consumeError(bad.takeError());
  }
  auto bytes = makeRecord(0, false);
  auto *header =
      reinterpret_cast<WaferTx81ProfilerRecordHeader *>(bytes.data());
  header->trace_event_limit = 1;
  auto bad = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(bad));
  llvm::consumeError(bad.takeError());
}

TEST(ProfilerRecordTest, CountOnlyCarriesRequiredCapacityWithoutEvents) {
  std::vector<uint8_t> bytes = makeRecord(0, /*trace=*/false);
  auto *header =
      reinterpret_cast<WaferTx81ProfilerRecordHeader *>(bytes.data());
  header->next_sequence = 19;
  auto decoded = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_TRUE(static_cast<bool>(decoded))
      << llvm::toString(decoded.takeError());
  EXPECT_TRUE(decoded->events.empty());
  EXPECT_EQ(decoded->header.next_sequence, 19u);

  header->flags |= WAFER_TX81_PROFILER_RECORD_TRACE_ENABLED;
  auto invalid = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(invalid));
  EXPECT_NE(llvm::toString(invalid.takeError()).find("exactly one"),
            std::string::npos);

  header->flags &= ~WAFER_TX81_PROFILER_RECORD_TRACE_ENABLED;
  header->event_count = 1;
  auto storedEvent = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(storedEvent));
  EXPECT_NE(llvm::toString(storedEvent.takeError()).find("stored events"),
            std::string::npos);

  header->event_count = 0;
  header->flags &= ~WAFER_TX81_PROFILER_RECORD_COUNT_ONLY;
  auto missingMode = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(missingMode));
  EXPECT_NE(llvm::toString(missingMode.takeError()).find("exactly one"),
            std::string::npos);
}

TEST(ProfilerRecordTest, DecodesHardwareObservationAndPreservesUnknownWorker) {
  auto decoded = wafer::runtime::decodeTx81ProfilerRecord(makeRecord(3));
  ASSERT_TRUE(static_cast<bool>(decoded))
      << llvm::toString(decoded.takeError());
  ASSERT_EQ(decoded->events.size(), 2u);
  const auto &event = decoded->events[1];
  EXPECT_EQ(event.counter_delta, 7u);
  EXPECT_EQ(event.observation_count, 2u);
  EXPECT_TRUE(wafer::runtime::isTx81ProfilerObservationSpanValid(event));
  EXPECT_TRUE(wafer::runtime::isTx81ProfilerOperationSpanValid(event));
  EXPECT_TRUE(wafer::runtime::isTx81ProfilerSiteSpanValid(event));
  EXPECT_TRUE(wafer::runtime::isTx81ProfilerCounterDeltaPositive(event));
  EXPECT_TRUE(wafer::runtime::isTx81ProfilerSiteValid(event));
  EXPECT_FALSE(wafer::runtime::isTx81ProfilerWorkerValid(event));
  EXPECT_EQ(wafer::runtime::getTx81ProfilerWorker(event), 0u);
}

TEST(ProfilerRecordTest, ValidatesDirectDTERoleAndActivity) {
  std::vector<uint8_t> bytes = makeRecord(3);
  auto *event = eventAt(bytes, 1);
  event->engine = WAFER_TX81_PROFILER_ENGINE_DIRECT_DTE;
  event->kind = WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_WAIT;
  event->metadata &= ~WAFER_TX81_PROFILER_EVENT_NCC_COUNTER_VALID;
  event->metadata |= WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_SEND |
                     WAFER_TX81_PROFILER_EVENT_DTE_COUNTER_VALID;
  auto decoded = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_TRUE(static_cast<bool>(decoded))
      << llvm::toString(decoded.takeError());
  EXPECT_TRUE(wafer::runtime::isTx81ProfilerDirectDTESend(decoded->events[1]));
  EXPECT_TRUE(
      wafer::runtime::isTx81ProfilerDirectDTECounterValid(decoded->events[1]));

  event->counter_delta = 0;
  event->metadata &= ~WAFER_TX81_PROFILER_EVENT_COUNTER_DELTA_POSITIVE;
  auto zeroRawCounter = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_TRUE(static_cast<bool>(zeroRawCounter))
      << llvm::toString(zeroRawCounter.takeError());
  EXPECT_TRUE(wafer::runtime::isTx81ProfilerObservationSpanValid(
      zeroRawCounter->events[1]));
  EXPECT_TRUE(wafer::runtime::isTx81ProfilerDirectDTECounterValid(
      zeroRawCounter->events[1]));

  event->counter_delta = 7;
  event->metadata |= WAFER_TX81_PROFILER_EVENT_COUNTER_DELTA_POSITIVE;
  event->metadata |= WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_RECV;
  auto bothRoles = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(bothRoles));
  EXPECT_NE(llvm::toString(bothRoles.takeError()).find("combination"),
            std::string::npos);

  event->metadata &= ~WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_RECV;
  event->metadata &= ~WAFER_TX81_PROFILER_EVENT_OBSERVATION_SPAN_VALID;
  event->metadata &= ~WAFER_TX81_PROFILER_EVENT_DTE_COUNTER_VALID;
  event->metadata &= ~WAFER_TX81_PROFILER_EVENT_COUNTER_DELTA_POSITIVE;
  event->observed_begin_cycle = 0;
  event->observed_end_cycle = 0;
  event->observation_count = 0;
  event->counter_delta = 0;
  auto nonzeroInvalid = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(nonzeroInvalid));
  EXPECT_NE(llvm::toString(nonzeroInvalid.takeError()).find("combination"),
            std::string::npos);

  auto missingWaitWindow = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(missingWaitWindow));
  EXPECT_NE(llvm::toString(missingWaitWindow.takeError()).find("combination"),
            std::string::npos);

  event->observed_begin_cycle = 112;
  event->observed_end_cycle = 135;
  event->observation_count = 2;
  event->metadata |= WAFER_TX81_PROFILER_EVENT_OBSERVATION_SPAN_VALID;
  auto unavailableCounter = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_TRUE(static_cast<bool>(unavailableCounter))
      << llvm::toString(unavailableCounter.takeError());
  EXPECT_FALSE(wafer::runtime::isTx81ProfilerDirectDTECounterValid(
      unavailableCounter->events[1]));

  event->counter_delta = 9;
  event->metadata |= WAFER_TX81_PROFILER_EVENT_COUNTER_DELTA_POSITIVE;
  auto tornCounter = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(tornCounter));
  EXPECT_NE(llvm::toString(tornCounter.takeError()).find("combination"),
            std::string::npos);
}

TEST(ProfilerRecordTest, PreservesZeroDeltaNCCObservation) {
  std::vector<uint8_t> bytes = makeRecord(3);
  auto *event = eventAt(bytes, 1);
  event->counter_delta = 0;
  event->metadata &= ~WAFER_TX81_PROFILER_EVENT_COUNTER_DELTA_POSITIVE;
  event->metadata |= WAFER_TX81_PROFILER_EVENT_SAME_ENGINE_AMBIGUOUS;
  auto decoded = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_TRUE(static_cast<bool>(decoded))
      << llvm::toString(decoded.takeError());
  EXPECT_EQ(decoded->events[1].observation_count, 2u);
  EXPECT_TRUE(
      wafer::runtime::isTx81ProfilerObservationSpanValid(decoded->events[1]));
  EXPECT_FALSE(
      wafer::runtime::isTx81ProfilerCounterDeltaPositive(decoded->events[1]));
  EXPECT_TRUE(
      wafer::runtime::isTx81ProfilerSameEngineAmbiguous(decoded->events[1]));
}

TEST(ProfilerRecordTest, PreservesNCCCommandWhenCounterIsUnavailable) {
  std::vector<uint8_t> bytes = makeRecord(3);
  auto *event = eventAt(bytes, 1);
  event->observed_begin_cycle = 0;
  event->observed_end_cycle = 0;
  event->counter_delta = 0;
  event->observation_count = 0;
  event->metadata &= ~(WAFER_TX81_PROFILER_EVENT_OBSERVATION_SPAN_VALID |
                       WAFER_TX81_PROFILER_EVENT_COUNTER_DELTA_POSITIVE |
                       WAFER_TX81_PROFILER_EVENT_NCC_COUNTER_VALID);
  auto unavailable = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_TRUE(static_cast<bool>(unavailable))
      << llvm::toString(unavailable.takeError());
  EXPECT_TRUE(wafer::runtime::isTx81ProfilerSiteValid(unavailable->events[1]));
  EXPECT_TRUE(
      wafer::runtime::isTx81ProfilerOperationSpanValid(unavailable->events[1]));
  EXPECT_FALSE(wafer::runtime::isTx81ProfilerObservationSpanValid(
      unavailable->events[1]));

  event->observed_begin_cycle = 112;
  event->observed_end_cycle = 135;
  event->observation_count = 2;
  event->metadata |= WAFER_TX81_PROFILER_EVENT_OBSERVATION_SPAN_VALID;
  auto observationWithoutValidity =
      wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(observationWithoutValidity));
  EXPECT_NE(llvm::toString(observationWithoutValidity.takeError())
                .find("NCC command"),
            std::string::npos);

  event->metadata |= WAFER_TX81_PROFILER_EVENT_NCC_COUNTER_VALID;
  event->metadata &= ~WAFER_TX81_PROFILER_EVENT_OBSERVATION_SPAN_VALID;
  event->observed_begin_cycle = 0;
  event->observed_end_cycle = 0;
  event->observation_count = 0;
  auto validityWithoutObservation =
      wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(validityWithoutObservation));
  EXPECT_NE(llvm::toString(validityWithoutObservation.takeError())
                .find("NCC command"),
            std::string::npos);

  event->metadata &= ~WAFER_TX81_PROFILER_EVENT_NCC_COUNTER_VALID;
  event->counter_delta = 1;
  event->metadata |= WAFER_TX81_PROFILER_EVENT_COUNTER_DELTA_POSITIVE;
  auto deltaWithoutValidity = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(deltaWithoutValidity));
  EXPECT_NE(
      llvm::toString(deltaWithoutValidity.takeError()).find("NCC command"),
      std::string::npos);
}

TEST(ProfilerRecordTest, ValidatesTargetSiteContainer) {
  std::vector<uint8_t> bytes = makeRecord(3);
  auto *event = eventAt(bytes, 0);
  event->observed_begin_cycle = 0;
  event->observed_end_cycle = 0;
  event->counter_delta = 0;
  event->operation_begin_cycle = 0;
  event->operation_end_cycle = 0;
  event->observation_count = 0;
  event->engine = WAFER_TX81_PROFILER_ENGINE_NONE;
  event->kind = WAFER_TX81_PROFILER_EVENT_TARGET_SITE;
  event->metadata = WAFER_TX81_PROFILER_EVENT_SITE_VALID |
                    WAFER_TX81_PROFILER_EVENT_SITE_SPAN_VALID;
  auto container = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_TRUE(static_cast<bool>(container))
      << llvm::toString(container.takeError());
  EXPECT_TRUE(
      wafer::runtime::isTx81ProfilerSiteValid(container->events.front()));
  EXPECT_TRUE(
      wafer::runtime::isTx81ProfilerSiteSpanValid(container->events.front()));
  EXPECT_FALSE(wafer::runtime::isTx81ProfilerOperationSpanValid(
      container->events.front()));

  event->operation_begin_cycle = 118;
  event->operation_end_cycle = 122;
  event->metadata |= WAFER_TX81_PROFILER_EVENT_OPERATION_SPAN_VALID;
  auto operation = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(operation));
  EXPECT_NE(llvm::toString(operation.takeError()).find("target-site"),
            std::string::npos);

  event->operation_begin_cycle = 0;
  event->operation_end_cycle = 0;
  event->metadata &= ~WAFER_TX81_PROFILER_EVENT_OPERATION_SPAN_VALID;
  event->metadata |= WAFER_TX81_PROFILER_EVENT_WORKER_VALID;
  auto worker = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(worker));
  EXPECT_NE(llvm::toString(worker.takeError()).find("target-site"),
            std::string::npos);

  event->metadata &= ~WAFER_TX81_PROFILER_EVENT_WORKER_VALID;
  event->sub_index = 1;
  auto nonzeroSubIndex = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(nonzeroSubIndex));
  EXPECT_NE(llvm::toString(nonzeroSubIndex.takeError()).find("target-site"),
            std::string::npos);
}

TEST(ProfilerRecordTest, RequiresContiguousDynamicSiteGroups) {
  std::vector<uint8_t> bytes = makeRecord(3);
  const auto container = *eventAt(bytes, 0);
  const auto operation = *eventAt(bytes, 1);
  auto valid = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_TRUE(static_cast<bool>(valid)) << llvm::toString(valid.takeError());

  setEvents(bytes, {operation});
  auto missingContainer = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(missingContainer));
  EXPECT_NE(llvm::toString(missingContainer.takeError())
                .find("no preceding target-site"),
            std::string::npos);

  auto mismatchedSite = operation;
  mismatchedSite.site_id = 8;
  setEvents(bytes, {container, mismatchedSite});
  auto wrongSite = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(wrongSite));
  EXPECT_NE(llvm::toString(wrongSite.takeError()).find("conflicts"),
            std::string::npos);

  auto mismatchedSpan = operation;
  ++mismatchedSpan.site_begin_cycle;
  setEvents(bytes, {container, mismatchedSpan});
  auto wrongSpan = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(wrongSpan));
  EXPECT_NE(llvm::toString(wrongSpan.takeError()).find("conflicts"),
            std::string::npos);

  auto skippedSubIndex = operation;
  skippedSubIndex.sub_index = 2;
  setEvents(bytes, {container, skippedSubIndex});
  auto noncontiguous = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(noncontiguous));
  EXPECT_NE(llvm::toString(noncontiguous.takeError()).find("not contiguous"),
            std::string::npos);

  auto nextContainer = makeTargetSiteEvent(8, 155, 190);
  auto nextOperation = operation;
  nextOperation.site_id = 8;
  nextOperation.site_begin_cycle = 155;
  nextOperation.site_end_cycle = 190;
  nextOperation.operation_begin_cycle = 160;
  nextOperation.operation_end_cycle = 171;
  nextOperation.observed_begin_cycle = 162;
  nextOperation.observed_end_cycle = 169;
  nextOperation.sub_index = 1;
  setEvents(bytes, {container, nextContainer, nextOperation});
  auto nextGroup = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_TRUE(static_cast<bool>(nextGroup))
      << llvm::toString(nextGroup.takeError());
}

TEST(ProfilerRecordTest, ValidatesExplicitCompletionWait) {
  std::vector<uint8_t> bytes = makeRecord(3);
  auto *event = eventAt(bytes, 1);
  event->engine = WAFER_TX81_PROFILER_ENGINE_NONE;
  event->kind = WAFER_TX81_PROFILER_EVENT_NCC_COMPLETION_WAIT;
  event->observed_begin_cycle = 0;
  event->observed_end_cycle = 0;
  event->counter_delta = 0;
  event->observation_count = 4;
  event->metadata = WAFER_TX81_PROFILER_EVENT_SITE_VALID |
                    WAFER_TX81_PROFILER_EVENT_SITE_SPAN_VALID |
                    WAFER_TX81_PROFILER_EVENT_OPERATION_SPAN_VALID |
                    WAFER_TX81_PROFILER_EVENT_WAIT_LOCAL;
  auto local = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_TRUE(static_cast<bool>(local)) << llvm::toString(local.takeError());
  EXPECT_TRUE(wafer::runtime::isTx81ProfilerLocalWait(local->events[1]));

  event->metadata &= ~WAFER_TX81_PROFILER_EVENT_WAIT_LOCAL;
  event->metadata |= WAFER_TX81_PROFILER_EVENT_WAIT_WORKER |
                     WAFER_TX81_PROFILER_EVENT_WORKER_VALID | 2U;
  auto worker = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_TRUE(static_cast<bool>(worker)) << llvm::toString(worker.takeError());
  EXPECT_TRUE(wafer::runtime::isTx81ProfilerWorkerWait(worker->events[1]));
  EXPECT_EQ(wafer::runtime::getTx81ProfilerWorker(worker->events[1]), 2u);

  event->metadata |= WAFER_TX81_PROFILER_EVENT_WAIT_LOCAL;
  auto conflicting = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(conflicting));
  EXPECT_NE(llvm::toString(conflicting.takeError()).find("completion-wait"),
            std::string::npos);

  event->metadata = WAFER_TX81_PROFILER_EVENT_OPERATION_SPAN_VALID |
                    WAFER_TX81_PROFILER_EVENT_WAIT_LOCAL;
  event->site_id = WAFER_TX81_PROFILER_INVALID_SITE_ID;
  event->sub_index = WAFER_TX81_PROFILER_INVALID_SUB_INDEX;
  event->site_begin_cycle = 0;
  event->site_end_cycle = 0;
  auto unbracketed = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(unbracketed));
  EXPECT_NE(llvm::toString(unbracketed.takeError()).find("completion-wait"),
            std::string::npos);
}

TEST(ProfilerRecordTest, ValidatesDirectDTEPhaseRoles) {
  std::vector<uint8_t> bytes = makeRecord(3);
  auto container = makeTargetSiteEvent();
  auto aggregate = makeDirectDTEEvent(WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_WAIT,
                                      WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_RECV,
                                      115, 145, 1);
  auto cleanup = makeDirectDTEEvent(
      WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_CLEANUP,
      WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_RECV, 130, 140, 2);
  setEvents(bytes, {container, aggregate, cleanup});
  auto decodedCleanup = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_TRUE(static_cast<bool>(decodedCleanup))
      << llvm::toString(decodedCleanup.takeError());

  eventAt(bytes, 2)->kind = WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_SETUP_ISSUE;
  auto receiverSetup = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(receiverSetup));
  EXPECT_NE(llvm::toString(receiverSetup.takeError()).find("sender-only"),
            std::string::npos);
}

TEST(ProfilerRecordTest, RequiresDirectDTEPhaseInsideMatchingAggregate) {
  std::vector<uint8_t> bytes = makeRecord(3);
  auto container = makeTargetSiteEvent();
  auto aggregate = makeDirectDTEEvent(WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_WAIT,
                                      WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_SEND,
                                      115, 145, 1);
  auto phase = makeDirectDTEEvent(
      WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_COMPLETION_WAIT,
      WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_SEND, 125, 135, 2);
  setEvents(bytes, {container, aggregate, phase});
  auto valid = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_TRUE(static_cast<bool>(valid)) << llvm::toString(valid.takeError());

  phase.sub_index = 1;
  setEvents(bytes, {container, phase});
  auto orphaned = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(orphaned));
  EXPECT_NE(llvm::toString(orphaned.takeError()).find("matching aggregate"),
            std::string::npos);

  phase.sub_index = 2;
  phase.operation_end_cycle = aggregate.operation_end_cycle + 1;
  setEvents(bytes, {container, aggregate, phase});
  auto outOfBounds = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(outOfBounds));
  EXPECT_NE(llvm::toString(outOfBounds.takeError()).find("matching aggregate"),
            std::string::npos);

  phase.operation_end_cycle = 135;
  phase.metadata &= ~WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_SEND;
  phase.metadata |= WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_RECV;
  setEvents(bytes, {container, aggregate, phase});
  auto wrongRole = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(wrongRole));
  EXPECT_NE(llvm::toString(wrongRole.takeError()).find("matching aggregate"),
            std::string::npos);
}

TEST(ProfilerRecordTest, SeparatesDirectDTEIssueFromCompletionPhases) {
  std::vector<uint8_t> bytes = makeRecord(3);
  auto container = makeTargetSiteEvent();
  auto aggregate = makeDirectDTEEvent(
      WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_ISSUE,
      WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_SEND, 115, 145, 1);
  auto phase = makeDirectDTEEvent(
      WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_SETUP_ISSUE,
      WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_SEND, 125, 135, 2);
  setEvents(bytes, {container, aggregate, phase});
  auto valid = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_TRUE(static_cast<bool>(valid)) << llvm::toString(valid.takeError());

  aggregate.kind = WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_WAIT;
  setEvents(bytes, {container, aggregate, phase});
  auto issuePhaseUnderWait = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(issuePhaseUnderWait));
  EXPECT_NE(llvm::toString(issuePhaseUnderWait.takeError())
                .find("matching aggregate"),
            std::string::npos);

  aggregate.kind = WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_ISSUE;
  aggregate.metadata &= ~WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_SEND;
  aggregate.metadata |= WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_RECV;
  setEvents(bytes, {container, aggregate, phase});
  auto receiverIssue = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(receiverIssue));
  EXPECT_NE(llvm::toString(receiverIssue.takeError()).find("aggregate"),
            std::string::npos);
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

TEST(ProfilerRecordTest, RequiresPMURestoreVerificationForTrace) {
  std::vector<uint8_t> bytes = makeRecord(0);
  auto *header =
      reinterpret_cast<WaferTx81ProfilerRecordHeader *>(bytes.data());
  header->summary_validity &=
      ~(WAFER_TX81_PROFILER_SUMMARY_NCC_PMU_RESTORE_VERIFIED |
        WAFER_TX81_PROFILER_SUMMARY_DTE_PMU_RESTORE_VERIFIED);
  auto decoded = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(decoded));
  EXPECT_NE(llvm::toString(decoded.takeError()).find("terminal lifecycle"),
            std::string::npos);
}

TEST(ProfilerRecordTest, RetainsCompleteRecordWhenPMUAdvancesAfterEnd) {
  std::vector<uint8_t> bytes = makeRecord(0);
  auto *header =
      reinterpret_cast<WaferTx81ProfilerRecordHeader *>(bytes.data());
  header->pmu_recovery.counters[0] =
      header->pmu_after.counters[0] + UINT64_C(1);
  header->summary_validity &= ~WAFER_TX81_PROFILER_SUMMARY_PMU_RECOVERED;

  auto decoded = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_TRUE(static_cast<bool>(decoded))
      << llvm::toString(decoded.takeError());
  EXPECT_EQ(decoded->header.summary_validity &
                WAFER_TX81_PROFILER_SUMMARY_PMU_RECOVERED,
            0u);
  EXPECT_NE(decoded->header.pmu_after.counters[0],
            decoded->header.pmu_recovery.counters[0]);
}

TEST(ProfilerRecordTest, RetainsRecoveredRecordWithPartialPMUStableMask) {
  std::vector<uint8_t> bytes = makeRecord(0);
  auto *header =
      reinterpret_cast<WaferTx81ProfilerRecordHeader *>(bytes.data());
  header->pmu_after.stable_mask = UINT32_C(1) << 0;
  header->pmu_recovery.stable_mask = UINT32_C(1) << 0;
  header->pmu_after.counters[4] = 91;
  header->pmu_recovery.counters[4] = 91;
  auto decoded = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_TRUE(static_cast<bool>(decoded))
      << llvm::toString(decoded.takeError());
  EXPECT_EQ(decoded->header.pmu_after.stable_mask, UINT32_C(1));
  EXPECT_EQ(decoded->header.pmu_after.counters[4], 91u);
  EXPECT_EQ(decoded->header.summary_validity &
                WAFER_TX81_PROFILER_SUMMARY_PMU_AFTER_STABLE,
            0u);
}

TEST(ProfilerRecordTest, ValidatesExclusiveTraceCostSummary) {
  std::vector<uint8_t> bytes = makeRecord(0);
  auto *header =
      reinterpret_cast<WaferTx81ProfilerRecordHeader *>(bytes.data());
  header->cost_summary.ncc_pmu_sample_cycles = 11;
  header->cost_summary.dte_pmu_sample_cycles = 12;
  header->cost_summary.event_bookkeeping_cycles = 13;
  header->cost_summary.status_poll_cycles = 14;
  header->cost_summary.site_hook_cycles = 15;
  header->cost_summary.completion_loop_bookkeeping_cycles = 16;
  header->cost_summary.entry_setup_cycles = 17;
  header->cost_summary.entry_teardown_cycles = 18;
  auto decoded = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_TRUE(static_cast<bool>(decoded))
      << llvm::toString(decoded.takeError());
  EXPECT_EQ(decoded->header.cost_summary.status_poll_cycles, 14u);

  header->cost_summary.completion_loop_bookkeeping_cycles = 36;
  auto oversizedEntryCost = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(oversizedEntryCost));
  EXPECT_NE(llvm::toString(oversizedEntryCost.takeError()).find("entry span"),
            std::string::npos);

  header->cost_summary.completion_loop_bookkeeping_cycles = 35;
  header->cost_summary.entry_setup_cycles = 80;
  header->cost_summary.entry_teardown_cycles = 90;
  auto exactEntryCost = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_TRUE(static_cast<bool>(exactEntryCost))
      << llvm::toString(exactEntryCost.takeError());

  header->cost_summary.entry_teardown_cycles = UINT64_MAX;
  auto saturated = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(saturated));
  EXPECT_NE(llvm::toString(saturated.takeError()).find("cost summary"),
            std::string::npos);

  bytes = makeRecord(0, /*trace=*/false);
  header = reinterpret_cast<WaferTx81ProfilerRecordHeader *>(bytes.data());
  header->cost_summary.site_hook_cycles = 1;
  auto nonTrace = wafer::runtime::decodeTx81ProfilerRecord(bytes);
  ASSERT_FALSE(static_cast<bool>(nonTrace));
  EXPECT_NE(llvm::toString(nonTrace.takeError()).find("cost summary"),
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
  header->next_sequence = 3;
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
