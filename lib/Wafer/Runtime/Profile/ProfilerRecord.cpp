//===- ProfilerRecord.cpp - TX81 profiler host contract ----------------===//

#include "Wafer/Runtime/Profile/ProfilerRecord.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Errc.h"

#include <cstddef>
#include <cstring>
#include <limits>
#include <type_traits>

namespace wafer::runtime {
namespace {

llvm::Error invalid(llvm::Twine message) {
  return llvm::createStringError(llvm::errc::invalid_argument, "%s",
                                 message.str().c_str());
}

template <typename T>
void writeLittleEndian(std::vector<uint8_t> &bytes, size_t offset, T value) {
  static_assert(std::is_unsigned_v<T>);
  for (size_t index = 0; index < sizeof(T); ++index)
    bytes[offset + index] =
        static_cast<uint8_t>(value >> static_cast<unsigned>(index * 8));
}

bool hasOnlyBits(uint32_t value, uint32_t mask) { return (value & ~mask) == 0; }

constexpr uint32_t kKnownRecordFlags =
    WAFER_TX81_PROFILER_RECORD_TRACE_ENABLED |
    WAFER_TX81_PROFILER_RECORD_ENTRY_BEGUN |
    WAFER_TX81_PROFILER_RECORD_ENTRY_ENDED |
    WAFER_TX81_PROFILER_RECORD_OVERFLOW |
    WAFER_TX81_PROFILER_RECORD_SITE_PROTOCOL_ERROR |
    WAFER_TX81_PROFILER_RECORD_SUB_INDEX_OVERFLOW |
    WAFER_TX81_PROFILER_RECORD_COMPLETE | WAFER_TX81_PROFILER_RECORD_COUNT_ONLY;

constexpr uint32_t kKnownSummaryValidity =
    WAFER_TX81_PROFILER_SUMMARY_ENTRY_CYCLES |
    WAFER_TX81_PROFILER_SUMMARY_PMU_BEFORE_CAPTURED |
    WAFER_TX81_PROFILER_SUMMARY_PMU_AFTER_CAPTURED |
    WAFER_TX81_PROFILER_SUMMARY_PMU_BEFORE_STABLE |
    WAFER_TX81_PROFILER_SUMMARY_PMU_AFTER_STABLE |
    WAFER_TX81_PROFILER_SUMMARY_PMU_ENABLE_UNCHANGED |
    WAFER_TX81_PROFILER_SUMMARY_PMU_RECOVERY_CAPTURED |
    WAFER_TX81_PROFILER_SUMMARY_PMU_RECOVERY_STABLE |
    WAFER_TX81_PROFILER_SUMMARY_PMU_RECOVERED |
    WAFER_TX81_PROFILER_SUMMARY_NCC_PMU_RESTORE_VERIFIED |
    WAFER_TX81_PROFILER_SUMMARY_DTE_PMU_RESTORE_VERIFIED;

constexpr uint32_t kKnownEventMetadata =
    WAFER_TX81_PROFILER_EVENT_WORKER_MASK |
    WAFER_TX81_PROFILER_EVENT_WORKER_VALID |
    WAFER_TX81_PROFILER_EVENT_SITE_VALID |
    WAFER_TX81_PROFILER_EVENT_OBSERVATION_SPAN_VALID |
    WAFER_TX81_PROFILER_EVENT_SITE_SPAN_VALID |
    WAFER_TX81_PROFILER_EVENT_OPERATION_SPAN_VALID |
    WAFER_TX81_PROFILER_EVENT_COUNTER_DELTA_POSITIVE |
    WAFER_TX81_PROFILER_EVENT_SAME_ENGINE_AMBIGUOUS |
    WAFER_TX81_PROFILER_EVENT_WAIT_LOCAL |
    WAFER_TX81_PROFILER_EVENT_WAIT_WORKER |
    WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_SEND |
    WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_RECV |
    WAFER_TX81_PROFILER_EVENT_DTE_COUNTER_VALID |
    WAFER_TX81_PROFILER_EVENT_NCC_COUNTER_VALID;

bool costSummaryIsZero(const WaferTx81ProfilerCostSummary &summary) {
  return summary.ncc_pmu_sample_cycles == 0 &&
         summary.dte_pmu_sample_cycles == 0 &&
         summary.event_bookkeeping_cycles == 0 &&
         summary.status_poll_cycles == 0 && summary.site_hook_cycles == 0 &&
         summary.completion_loop_bookkeeping_cycles == 0 &&
         summary.entry_setup_cycles == 0 && summary.entry_teardown_cycles == 0;
}

bool costSummaryIsFinite(const WaferTx81ProfilerCostSummary &summary) {
  const uint64_t values[] = {
      summary.ncc_pmu_sample_cycles,
      summary.dte_pmu_sample_cycles,
      summary.event_bookkeeping_cycles,
      summary.status_poll_cycles,
      summary.site_hook_cycles,
      summary.completion_loop_bookkeeping_cycles,
      summary.entry_setup_cycles,
      summary.entry_teardown_cycles,
  };
  uint64_t total = 0;
  for (uint64_t value : values) {
    if (value == std::numeric_limits<uint64_t>::max() ||
        total > std::numeric_limits<uint64_t>::max() - value)
      return false;
    total += value;
  }
  return true;
}

bool entryCostSummaryFitsSpan(const WaferTx81ProfilerCostSummary &summary,
                              uint64_t entryCycles) {
  const uint64_t values[] = {
      summary.ncc_pmu_sample_cycles,
      summary.dte_pmu_sample_cycles,
      summary.event_bookkeeping_cycles,
      summary.status_poll_cycles,
      summary.site_hook_cycles,
      summary.completion_loop_bookkeeping_cycles,
  };
  uint64_t total = 0;
  for (uint64_t value : values) {
    // This comparison both saturates an otherwise overflowing addition and
    // proves that the exclusive, entry-internal categories fit on the entry
    // axis. Entry setup and teardown deliberately live outside that axis.
    if (value > entryCycles - total)
      return false;
    total += value;
  }
  return true;
}

bool isNCCEngine(uint8_t engine) {
  return engine <= WAFER_TX81_PROFILER_ENGINE_TDMA;
}

bool isDirectDTEKind(uint8_t kind) {
  return (kind >= WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_WAIT &&
          kind <= WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_CLEANUP) ||
         kind == WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_ISSUE;
}

bool snapshotStableMaskIsValid(const WaferTx81ProfilerPMUSnapshot &snapshot) {
  constexpr uint32_t known =
      (UINT32_C(1) << WAFER_TX81_PROFILER_PMU64_COUNTERS) - 1U;
  return (snapshot.stable_mask & ~known) == 0;
}

bool snapshotsRecovered(const WaferTx81ProfilerPMUSnapshot &end,
                        const WaferTx81ProfilerPMUSnapshot &recovery) {
  for (size_t index = 0; index < WAFER_TX81_PROFILER_PMU64_COUNTERS; ++index)
    if (end.counters[index] != recovery.counters[index])
      return false;
  for (size_t worker = 0; worker < WAFER_TX81_PROFILER_WORKERS; ++worker)
    for (size_t queue = 0; queue < WAFER_TX81_PROFILER_QUEUES; ++queue)
      if (end.instructions[worker][queue] !=
              recovery.instructions[worker][queue] ||
          end.blocking[worker][queue] != recovery.blocking[worker][queue])
        return false;
  return true;
}

} // namespace

llvm::Expected<std::vector<uint8_t>>
buildTx81ProfilerLaunchImage(uint64_t recordBytes, uint32_t tileId,
                             Tx81ProfilerCaptureKind kind) {
  if (recordBytes < WAFER_TX81_PROFILER_MIN_BUFFER_BYTES ||
      recordBytes % WAFER_TX81_PROFILER_BUFFER_ALIGNMENT != 0)
    return invalid("TX81 profiler record bytes are too small or misaligned");
  if (recordBytes > std::numeric_limits<size_t>::max())
    return invalid("TX81 profiler record bytes exceed host addressability");
  if (tileId >= WAFER_TX81_PROFILER_TILE_COUNT)
    return invalid("TX81 profiler tile id is outside 0..15");

  std::vector<uint8_t> image(static_cast<size_t>(recordBytes), UINT8_C(0xa5));
  writeLittleEndian<uint64_t>(image,
                              offsetof(WaferTx81ProfilerLaunchConfig, magic),
                              WAFER_TX81_PROFILER_LAUNCH_CONFIG_MAGIC);
  writeLittleEndian<uint32_t>(
      image, offsetof(WaferTx81ProfilerLaunchConfig, reserved0), 0);
  writeLittleEndian<uint32_t>(
      image, offsetof(WaferTx81ProfilerLaunchConfig, config_bytes),
      WAFER_TX81_PROFILER_LAUNCH_CONFIG_BYTES);
  writeLittleEndian<uint64_t>(
      image, offsetof(WaferTx81ProfilerLaunchConfig, record_bytes),
      recordBytes);
  writeLittleEndian<uint32_t>(
      image, offsetof(WaferTx81ProfilerLaunchConfig, tile_id), tileId);
  uint32_t flags = 0;
  switch (kind) {
  case Tx81ProfilerCaptureKind::Count:
    flags = WAFER_TX81_PROFILER_ENTRY_COUNT_ONLY;
    break;
  case Tx81ProfilerCaptureKind::Trace:
    flags = WAFER_TX81_PROFILER_ENTRY_TRACE_ENABLED;
    break;
  default:
    return invalid("TX81 profiler capture kind is invalid");
  }
  writeLittleEndian<uint32_t>(
      image, offsetof(WaferTx81ProfilerLaunchConfig, flags), flags);
  writeLittleEndian<uint64_t>(image,
                              offsetof(WaferTx81ProfilerLaunchConfig, guard),
                              WAFER_TX81_PROFILER_LAUNCH_CONFIG_GUARD);
  for (size_t index = 0; index < 3; ++index)
    writeLittleEndian<uint64_t>(
        image,
        offsetof(WaferTx81ProfilerLaunchConfig, reserved) +
            index * sizeof(uint64_t),
        0);
  return image;
}

llvm::Expected<Tx81ProfilerRecord>
decodeTx81ProfilerRecord(llvm::ArrayRef<uint8_t> bytes) {
  if (llvm::endianness::native != llvm::endianness::little)
    return invalid("TX81 profiler decoder requires a little-endian host");
  if (bytes.size() < WAFER_TX81_PROFILER_MIN_BUFFER_BYTES ||
      bytes.size() % WAFER_TX81_PROFILER_BUFFER_ALIGNMENT != 0)
    return invalid("TX81 profiler buffer byte count is too small or unaligned");

  Tx81ProfilerRecord record;
  std::memcpy(&record.header, bytes.data(), sizeof(record.header));
  const WaferTx81ProfilerRecordHeader &header = record.header;
  if (header.magic != WAFER_TX81_PROFILER_RECORD_MAGIC ||
      header.reserved0 != 0)
    return invalid("TX81 profiler record magic or reserved field is invalid");
  if (header.header_bytes != WAFER_TX81_PROFILER_HEADER_BYTES ||
      header.event_bytes != WAFER_TX81_PROFILER_TSM_CALL_EVENT_BYTES ||
      header.events_offset != WAFER_TX81_PROFILER_EVENTS_OFFSET)
    return invalid("TX81 profiler record byte layout is invalid");
  if (header.buffer_bytes != bytes.size() || header.buffer_address == 0 ||
      header.buffer_address % WAFER_TX81_PROFILER_BUFFER_ALIGNMENT != 0)
    return invalid("TX81 profiler record buffer identity is invalid");
  if (header.tile_id >= WAFER_TX81_PROFILER_TILE_COUNT)
    return invalid("TX81 profiler record tile id is outside 0..15");
  if (!hasOnlyBits(header.flags, kKnownRecordFlags) ||
      !hasOnlyBits(header.summary_validity, kKnownSummaryValidity) ||
      header.reserved1 != 0)
    return invalid("TX81 profiler record contains unknown flags");
  if (header.header_guard != WAFER_TX81_PROFILER_HEADER_GUARD ||
      header.header_footer_guard != WAFER_TX81_PROFILER_HEADER_GUARD)
    return invalid("TX81 profiler header guard is corrupt");
  const uint64_t expectedGuardOffset =
      bytes.size() - WAFER_TX81_PROFILER_BUFFER_GUARD_BYTES;
  if (header.buffer_guard_offset != expectedGuardOffset)
    return invalid("TX81 profiler tail guard offset is invalid");
  uint64_t tailGuard = 0;
  std::memcpy(&tailGuard, bytes.data() + expectedGuardOffset,
              sizeof(tailGuard));
  if (tailGuard != WAFER_TX81_PROFILER_BUFFER_GUARD)
    return invalid("TX81 profiler tail guard is corrupt");

  const uint64_t eventStorage = bytes.size() -
                                WAFER_TX81_PROFILER_EVENTS_OFFSET -
                                WAFER_TX81_PROFILER_BUFFER_GUARD_BYTES;
  const uint64_t expectedCapacity =
      eventStorage / WAFER_TX81_PROFILER_TSM_CALL_EVENT_BYTES;
  if (expectedCapacity > std::numeric_limits<uint32_t>::max() ||
      header.event_capacity != expectedCapacity ||
      header.event_count > header.event_capacity)
    return invalid("TX81 profiler event capacity or count is invalid");
  const bool traceEnabled =
      (header.flags & WAFER_TX81_PROFILER_RECORD_TRACE_ENABLED) != 0;
  const bool countOnly =
      (header.flags & WAFER_TX81_PROFILER_RECORD_COUNT_ONLY) != 0;
  if (traceEnabled == countOnly)
    return invalid("TX81 profiler record must enable exactly one capture mode");
  if (traceEnabled &&
      header.next_sequence != static_cast<uint64_t>(header.event_count) +
                                  header.dropped_event_count)
    return invalid("TX81 profiler event sequence accounting is inconsistent");
  if (countOnly && (header.event_count != 0 || header.dropped_event_count != 0))
    return invalid("TX81 profiler count record contains stored events");
  if ((!traceEnabled && !costSummaryIsZero(header.cost_summary)) ||
      !costSummaryIsFinite(header.cost_summary))
    return invalid("TX81 profiler cost summary is invalid");

  constexpr uint32_t terminalFlags = WAFER_TX81_PROFILER_RECORD_ENTRY_BEGUN |
                                     WAFER_TX81_PROFILER_RECORD_ENTRY_ENDED |
                                     WAFER_TX81_PROFILER_RECORD_COMPLETE;
  const uint32_t pmuValidityBits =
      WAFER_TX81_PROFILER_SUMMARY_PMU_BEFORE_CAPTURED |
      WAFER_TX81_PROFILER_SUMMARY_PMU_AFTER_CAPTURED |
      WAFER_TX81_PROFILER_SUMMARY_PMU_BEFORE_STABLE |
      WAFER_TX81_PROFILER_SUMMARY_PMU_AFTER_STABLE |
      WAFER_TX81_PROFILER_SUMMARY_PMU_ENABLE_UNCHANGED |
      WAFER_TX81_PROFILER_SUMMARY_PMU_RECOVERY_CAPTURED |
      WAFER_TX81_PROFILER_SUMMARY_PMU_RECOVERY_STABLE |
      WAFER_TX81_PROFILER_SUMMARY_PMU_RECOVERED |
      WAFER_TX81_PROFILER_SUMMARY_NCC_PMU_RESTORE_VERIFIED |
      WAFER_TX81_PROFILER_SUMMARY_DTE_PMU_RESTORE_VERIFIED;
  const uint32_t requiredSummary =
      WAFER_TX81_PROFILER_SUMMARY_ENTRY_CYCLES |
      (traceEnabled ? (WAFER_TX81_PROFILER_SUMMARY_PMU_BEFORE_CAPTURED |
                       WAFER_TX81_PROFILER_SUMMARY_PMU_AFTER_CAPTURED |
                       WAFER_TX81_PROFILER_SUMMARY_PMU_RECOVERY_CAPTURED |
                       WAFER_TX81_PROFILER_SUMMARY_NCC_PMU_RESTORE_VERIFIED |
                       WAFER_TX81_PROFILER_SUMMARY_DTE_PMU_RESTORE_VERIFIED)
                    : 0U);
  if ((header.flags & terminalFlags) != terminalFlags ||
      (header.summary_validity & requiredSummary) != requiredSummary ||
      (countOnly && (header.summary_validity & pmuValidityBits) != 0U) ||
      header.entry_end_cycle < header.entry_begin_cycle)
    return invalid("TX81 profiler record terminal lifecycle is incomplete");
  if (!entryCostSummaryFitsSpan(header.cost_summary,
                                header.entry_end_cycle -
                                    header.entry_begin_cycle))
    return invalid(
        "TX81 profiler entry-internal cost summary exceeds the entry span");
  if (traceEnabled) {
    if (!snapshotStableMaskIsValid(header.pmu_before) ||
        !snapshotStableMaskIsValid(header.pmu_after) ||
        !snapshotStableMaskIsValid(header.pmu_recovery))
      return invalid("TX81 profiler PMU stable mask contains unknown bits");
    constexpr uint32_t allStable =
        (UINT32_C(1) << WAFER_TX81_PROFILER_PMU64_COUNTERS) - 1U;
    if (((header.summary_validity &
          WAFER_TX81_PROFILER_SUMMARY_PMU_BEFORE_STABLE) != 0) !=
            (header.pmu_before.stable_mask == allStable) ||
        ((header.summary_validity &
          WAFER_TX81_PROFILER_SUMMARY_PMU_AFTER_STABLE) != 0) !=
            (header.pmu_after.stable_mask == allStable) ||
        ((header.summary_validity &
          WAFER_TX81_PROFILER_SUMMARY_PMU_RECOVERY_STABLE) != 0) !=
            (header.pmu_recovery.stable_mask == allStable) ||
        ((header.summary_validity &
          WAFER_TX81_PROFILER_SUMMARY_PMU_ENABLE_UNCHANGED) != 0) !=
            (header.pmu_before.enable == header.pmu_after.enable &&
             header.pmu_after.enable == header.pmu_recovery.enable) ||
        ((header.summary_validity & WAFER_TX81_PROFILER_SUMMARY_PMU_RECOVERED) !=
         0) != snapshotsRecovered(header.pmu_after, header.pmu_recovery))
      return invalid("TX81 profiler PMU validity is inconsistent");
  }

  const bool overflow =
      (header.flags & WAFER_TX81_PROFILER_RECORD_OVERFLOW) != 0;
  const bool protocolInvalid =
      (header.flags & WAFER_TX81_PROFILER_RECORD_SITE_PROTOCOL_ERROR) != 0;
  if (overflow != (header.dropped_event_count != 0))
    return invalid("TX81 profiler overflow accounting is inconsistent");
  if (protocolInvalid) {
    if (header.trace_state != WAFER_TX81_PROFILER_TRACE_INVALID)
      return invalid("TX81 profiler protocol error has wrong terminal state");
  } else if (overflow) {
    if (header.trace_state != WAFER_TX81_PROFILER_TRACE_OVERFLOW)
      return invalid("TX81 profiler overflow has wrong terminal state");
  } else if (header.trace_state != WAFER_TX81_PROFILER_TRACE_COMPLETE) {
    return invalid("TX81 profiler record has no terminal trace state");
  }

  record.events.resize(header.event_count);
  if (!record.events.empty())
    std::memcpy(
        record.events.data(), bytes.data() + WAFER_TX81_PROFILER_EVENTS_OFFSET,
        record.events.size() * WAFER_TX81_PROFILER_TSM_CALL_EVENT_BYTES);
  for (auto [index, event] : llvm::enumerate(record.events)) {
    if (event.sequence != index)
      return invalid("TX81 profiler stored event sequence is not contiguous");
    if (!hasOnlyBits(event.metadata, kKnownEventMetadata))
      return invalid("TX81 profiler event metadata contains unknown bits");

    const bool observationValid = isTx81ProfilerObservationSpanValid(event);
    const bool siteSpanValid = isTx81ProfilerSiteSpanValid(event);
    const bool operationValid = isTx81ProfilerOperationSpanValid(event);
    const bool positiveDelta = isTx81ProfilerCounterDeltaPositive(event);
    const bool ambiguous = isTx81ProfilerSameEngineAmbiguous(event);
    const bool localWait = isTx81ProfilerLocalWait(event);
    const bool workerWait = isTx81ProfilerWorkerWait(event);
    const bool directSend = isTx81ProfilerDirectDTESend(event);
    const bool directReceive = isTx81ProfilerDirectDTEReceive(event);
    const bool directCounterValid = isTx81ProfilerDirectDTECounterValid(event);
    const bool nccCounterValid =
        (event.metadata & WAFER_TX81_PROFILER_EVENT_NCC_COUNTER_VALID) != 0;

    if (observationValid) {
      if (event.observation_count == 0 ||
          event.observed_end_cycle < event.observed_begin_cycle)
        return invalid("TX81 profiler observation span is invalid");
    } else if (event.observed_begin_cycle != 0 ||
               event.observed_end_cycle != 0) {
      return invalid(
          "TX81 profiler event has fields for an unavailable observation");
    } else if (event.observation_count != 0 &&
               event.kind != WAFER_TX81_PROFILER_EVENT_NCC_COMPLETION_WAIT) {
      return invalid("TX81 profiler event has an unexpected observation count");
    }
    if (operationValid) {
      if (event.operation_end_cycle < event.operation_begin_cycle)
        return invalid("TX81 profiler operation span is reversed");
    } else if (event.operation_begin_cycle != 0 ||
               event.operation_end_cycle != 0) {
      return invalid(
          "TX81 profiler event has fields for an unavailable operation");
    }
    if (positiveDelta != (event.counter_delta != 0))
      return invalid("TX81 profiler counter delta validity is inconsistent");

    if (!isTx81ProfilerWorkerValid(event) && getTx81ProfilerWorker(event) != 0)
      return invalid("TX81 profiler invalid worker has nonzero identity bits");
    if (isTx81ProfilerWorkerValid(event) &&
        getTx81ProfilerWorker(event) >= WAFER_TX81_PROFILER_WORKERS)
      return invalid("TX81 profiler valid worker is outside 0..2");
    if (isTx81ProfilerSiteValid(event)) {
      if (event.site_id == WAFER_TX81_PROFILER_INVALID_SITE_ID ||
          event.sub_index == WAFER_TX81_PROFILER_INVALID_SUB_INDEX)
        return invalid("TX81 profiler valid site uses a sentinel identity");
      if (!siteSpanValid || event.site_end_cycle < event.site_begin_cycle ||
          (operationValid &&
           (event.operation_begin_cycle < event.site_begin_cycle ||
            event.operation_end_cycle > event.site_end_cycle)))
        return invalid("TX81 profiler valid site span is inconsistent");
    } else if (event.site_id != WAFER_TX81_PROFILER_INVALID_SITE_ID ||
               event.sub_index != WAFER_TX81_PROFILER_INVALID_SUB_INDEX ||
               siteSpanValid || event.site_begin_cycle != 0 ||
               event.site_end_cycle != 0) {
      return invalid("TX81 profiler invalid site does not use sentinels");
    }

    switch (event.kind) {
    case WAFER_TX81_PROFILER_EVENT_NCC_COMMAND:
      if (!isNCCEngine(event.engine) || !operationValid ||
          !isTx81ProfilerSiteValid(event) || localWait || workerWait ||
          directSend || directReceive || directCounterValid ||
          nccCounterValid != observationValid ||
          (!nccCounterValid && (event.observation_count != 0 ||
                                event.counter_delta != 0 || positiveDelta)))
        return invalid(
            "TX81 profiler NCC command field combination is invalid");
      break;
    case WAFER_TX81_PROFILER_EVENT_NCC_COMPLETION_WAIT:
      if (event.engine != WAFER_TX81_PROFILER_ENGINE_NONE || observationValid ||
          !operationValid || !isTx81ProfilerSiteValid(event) || ambiguous ||
          localWait == workerWait || directSend || directReceive ||
          directCounterValid || event.counter_delta != 0 ||
          event.observation_count == 0 ||
          (localWait && isTx81ProfilerWorkerValid(event)) ||
          (workerWait && !isTx81ProfilerWorkerValid(event)))
        return invalid(
            "TX81 profiler NCC completion-wait field combination is invalid");
      break;
    case WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_WAIT:
    case WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_ISSUE:
      if (event.engine != WAFER_TX81_PROFILER_ENGINE_DIRECT_DTE ||
          !observationValid || event.observation_count != 2 ||
          !operationValid || !isTx81ProfilerSiteValid(event) || ambiguous ||
          localWait || workerWait || directSend == directReceive ||
          isTx81ProfilerWorkerValid(event) ||
          (!directCounterValid && event.counter_delta != 0) ||
          (event.kind == WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_ISSUE &&
           !directSend))
        return invalid(
            "TX81 profiler Direct-DTE aggregate field combination is invalid");
      break;
    case WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_PEER_READY_WAIT:
    case WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_SETUP_ISSUE:
    case WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_COMPLETION_WAIT:
    case WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_CLEANUP:
      if (event.engine != WAFER_TX81_PROFILER_ENGINE_DIRECT_DTE ||
          observationValid || !operationValid ||
          !isTx81ProfilerSiteValid(event) || ambiguous || localWait ||
          workerWait || directSend == directReceive ||
          isTx81ProfilerWorkerValid(event) || directCounterValid ||
          event.counter_delta != 0 || event.observation_count != 0)
        return invalid(
            "TX81 profiler Direct-DTE phase field combination is invalid");
      if ((event.kind == WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_PEER_READY_WAIT ||
           event.kind == WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_SETUP_ISSUE) &&
          !directSend)
        return invalid(
            "TX81 profiler receiver contains a sender-only Direct-DTE phase");
      break;
    case WAFER_TX81_PROFILER_EVENT_TARGET_SITE:
      if (event.engine != WAFER_TX81_PROFILER_ENGINE_NONE ||
          event.sub_index != 0 ||
          event.metadata != (WAFER_TX81_PROFILER_EVENT_SITE_VALID |
                             WAFER_TX81_PROFILER_EVENT_SITE_SPAN_VALID) ||
          observationValid || operationValid || event.counter_delta != 0 ||
          event.observation_count != 0)
        return invalid(
            "TX81 profiler target-site field combination is invalid");
      break;
    default:
      return invalid("TX81 profiler event kind is invalid");
    }
    if (!isDirectDTEKind(event.kind) &&
        (directSend || directReceive || directCounterValid))
      return invalid(
          "TX81 profiler non-DTE event contains Direct-DTE metadata");
    if (event.kind != WAFER_TX81_PROFILER_EVENT_NCC_COMMAND && nccCounterValid)
      return invalid(
          "TX81 profiler non-NCC event contains NCC counter validity");
  }
  const WaferTx81ProfilerTSMCallEvent *siteContainer = nullptr;
  uint32_t nextSubIndex = 0;
  for (const auto &event : record.events) {
    if (event.kind == WAFER_TX81_PROFILER_EVENT_TARGET_SITE) {
      siteContainer = &event;
      nextSubIndex = 1;
      continue;
    }
    if (siteContainer == nullptr)
      return invalid(
          "TX81 profiler event has no preceding target-site container");
    if (event.site_id != siteContainer->site_id ||
        event.site_begin_cycle != siteContainer->site_begin_cycle ||
        event.site_end_cycle != siteContainer->site_end_cycle)
      return invalid(
          "TX81 profiler event conflicts with its target-site container");
    if (event.sub_index != nextSubIndex)
      return invalid(
          "TX81 profiler target-site child sub-index is not contiguous");
    ++nextSubIndex;
  }
  const WaferTx81ProfilerTSMCallEvent *directAggregate = nullptr;
  for (const auto &event : record.events) {
    if (event.kind == WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_ISSUE ||
        event.kind == WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_WAIT) {
      directAggregate = &event;
      continue;
    }
    if (event.kind < WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_PEER_READY_WAIT ||
        event.kind > WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_CLEANUP) {
      directAggregate = nullptr;
      continue;
    }
    const bool issuePhase =
        event.kind == WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_PEER_READY_WAIT ||
        event.kind == WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_SETUP_ISSUE;
    const bool issueAggregate =
        directAggregate != nullptr &&
        directAggregate->kind == WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_ISSUE;
    if (directAggregate == nullptr || issuePhase != issueAggregate ||
        directAggregate->site_id != event.site_id ||
        directAggregate->site_begin_cycle != event.site_begin_cycle ||
        directAggregate->site_end_cycle != event.site_end_cycle ||
        isTx81ProfilerDirectDTESend(*directAggregate) !=
            isTx81ProfilerDirectDTESend(event) ||
        isTx81ProfilerDirectDTEReceive(*directAggregate) !=
            isTx81ProfilerDirectDTEReceive(event) ||
        directAggregate->operation_begin_cycle > event.operation_begin_cycle ||
        event.operation_end_cycle > directAggregate->operation_end_cycle)
      return invalid(
          "TX81 profiler Direct-DTE phase is not contained by a matching "
          "aggregate");
  }
  return record;
}

llvm::Error
verifyTx81ProfilerTileDomain(llvm::ArrayRef<Tx81ProfilerRecord> records) {
  if (records.size() != WAFER_TX81_PROFILER_TILE_COUNT)
    return invalid("TX81 profiler requires exactly 16 tile records");

  llvm::SmallBitVector tiles(WAFER_TX81_PROFILER_TILE_COUNT);
  llvm::SmallSet<uint64_t, WAFER_TX81_PROFILER_TILE_COUNT> bufferAddresses;
  const uint64_t bufferBytes = records.front().header.buffer_bytes;
  const bool traceEnabled = (records.front().header.flags &
                             WAFER_TX81_PROFILER_RECORD_TRACE_ENABLED) != 0;
  const bool countOnly = (records.front().header.flags &
                          WAFER_TX81_PROFILER_RECORD_COUNT_ONLY) != 0;
  for (const Tx81ProfilerRecord &record : records) {
    const WaferTx81ProfilerRecordHeader &header = record.header;
    if (tiles.test(header.tile_id))
      return invalid("TX81 profiler tile record is duplicated");
    tiles.set(header.tile_id);
    if (!bufferAddresses.insert(header.buffer_address).second)
      return invalid("TX81 profiler buffer address is duplicated");
    if (header.buffer_bytes != bufferBytes ||
        ((header.flags & WAFER_TX81_PROFILER_RECORD_TRACE_ENABLED) != 0) !=
            traceEnabled ||
        ((header.flags & WAFER_TX81_PROFILER_RECORD_COUNT_ONLY) != 0) !=
            countOnly)
      return invalid("TX81 profiler tile records use different contracts");
    if (header.trace_state != WAFER_TX81_PROFILER_TRACE_COMPLETE ||
        (header.flags & (WAFER_TX81_PROFILER_RECORD_OVERFLOW |
                         WAFER_TX81_PROFILER_RECORD_SITE_PROTOCOL_ERROR |
                         WAFER_TX81_PROFILER_RECORD_SUB_INDEX_OVERFLOW)) != 0 ||
        header.dropped_event_count != 0 || header.active_site_depth != 0 ||
        header.active_site_id != WAFER_TX81_PROFILER_INVALID_SITE_ID)
      return invalid("TX81 profiler tile record is incomplete or invalid");
    if (traceEnabled && llvm::any_of(record.events, [](const auto &event) {
          return !isTx81ProfilerSiteValid(event);
        }))
      return invalid("TX81 profiler trace contains an unknown typed site");
  }
  if (tiles.count() != WAFER_TX81_PROFILER_TILE_COUNT)
    return invalid("TX81 profiler tile domain is not all-and-only 0..15");
  return llvm::Error::success();
}

} // namespace wafer::runtime
