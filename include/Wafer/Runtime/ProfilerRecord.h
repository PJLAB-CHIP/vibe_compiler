//===- ProfilerRecord.h - TX81 profiler host contract ---------*- C++ -*-===//

#ifndef WAFER_RUNTIME_PROFILERRECORD_H
#define WAFER_RUNTIME_PROFILERRECORD_H

#include "Wafer/ABI/Tx81ProfilerABI.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace wafer::runtime {

/// One structurally decoded per-tile profiler record. The current schema
/// separates
/// conservative PMU observation, typed-site and exact target-operation spans.
/// NCC counter_delta is the vendor execution-time counter delta in nanoseconds;
/// Direct-DTE counter_delta remains raw and is usable only when its validity
/// bit is set.
struct Tx81ProfilerRecord {
  WaferTx81ProfilerRecordHeader header{};
  std::vector<WaferTx81ProfilerTSMCallEvent> events;
};

enum class Tx81ProfilerCaptureKind {
  Count,
  Trace,
};

/// Builds the exact host image uploaded to an owned, aligned per-tile DDR
/// profiler allocation before launch. The profiling-only device wrapper
/// receives the allocation address through its hidden final pointer-table
/// slot; the CRT replaces this configuration with the output record.
llvm::Expected<std::vector<uint8_t>>
buildTx81ProfilerLaunchImage(uint64_t recordBytes, uint32_t tileId,
                             Tx81ProfilerCaptureKind kind);

/// Decodes and verifies one terminal record, including version, byte layout,
/// guards, event bounds and internal lifecycle consistency. Overflow and
/// protocol-invalid records are retained as structurally decoded evidence;
/// verifyTx81ProfilerTileDomain rejects them for a successful collection.
llvm::Expected<Tx81ProfilerRecord>
decodeTx81ProfilerRecord(llvm::ArrayRef<uint8_t> bytes);

/// Requires all-and-only tile ids 0..15, a common record contract and
/// complete, non-overflowing records. Every stored event requires a static
/// profiled-package site identity; an unbracketed completion wait is an
/// instrumentation-contract failure rather than anonymous evidence. Worker
/// attribution remains optional and is governed solely by validity bits.
llvm::Error
verifyTx81ProfilerTileDomain(llvm::ArrayRef<Tx81ProfilerRecord> records);

inline bool
isTx81ProfilerWorkerValid(const WaferTx81ProfilerTSMCallEvent &event) {
  return (event.metadata & WAFER_TX81_PROFILER_EVENT_WORKER_VALID) != 0;
}

inline uint8_t
getTx81ProfilerWorker(const WaferTx81ProfilerTSMCallEvent &event) {
  return event.metadata & WAFER_TX81_PROFILER_EVENT_WORKER_MASK;
}

inline bool
isTx81ProfilerSiteValid(const WaferTx81ProfilerTSMCallEvent &event) {
  return (event.metadata & WAFER_TX81_PROFILER_EVENT_SITE_VALID) != 0;
}

inline bool
isTx81ProfilerObservationSpanValid(const WaferTx81ProfilerTSMCallEvent &event) {
  return (event.metadata & WAFER_TX81_PROFILER_EVENT_OBSERVATION_SPAN_VALID) !=
         0;
}

inline bool
isTx81ProfilerSiteSpanValid(const WaferTx81ProfilerTSMCallEvent &event) {
  return (event.metadata & WAFER_TX81_PROFILER_EVENT_SITE_SPAN_VALID) != 0;
}

inline bool
isTx81ProfilerOperationSpanValid(const WaferTx81ProfilerTSMCallEvent &event) {
  return (event.metadata & WAFER_TX81_PROFILER_EVENT_OPERATION_SPAN_VALID) != 0;
}

inline bool
isTx81ProfilerCounterDeltaPositive(const WaferTx81ProfilerTSMCallEvent &event) {
  return (event.metadata & WAFER_TX81_PROFILER_EVENT_COUNTER_DELTA_POSITIVE) !=
         0;
}

inline bool
isTx81ProfilerSameEngineAmbiguous(const WaferTx81ProfilerTSMCallEvent &event) {
  return (event.metadata & WAFER_TX81_PROFILER_EVENT_SAME_ENGINE_AMBIGUOUS) !=
         0;
}

inline bool
isTx81ProfilerLocalWait(const WaferTx81ProfilerTSMCallEvent &event) {
  return (event.metadata & WAFER_TX81_PROFILER_EVENT_WAIT_LOCAL) != 0;
}

inline bool
isTx81ProfilerWorkerWait(const WaferTx81ProfilerTSMCallEvent &event) {
  return (event.metadata & WAFER_TX81_PROFILER_EVENT_WAIT_WORKER) != 0;
}

inline bool
isTx81ProfilerDirectDTESend(const WaferTx81ProfilerTSMCallEvent &event) {
  return (event.metadata & WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_SEND) != 0;
}

inline bool
isTx81ProfilerDirectDTEReceive(const WaferTx81ProfilerTSMCallEvent &event) {
  return (event.metadata & WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_RECV) != 0;
}

inline bool isTx81ProfilerDirectDTECounterValid(
    const WaferTx81ProfilerTSMCallEvent &event) {
  return (event.metadata & WAFER_TX81_PROFILER_EVENT_DTE_COUNTER_VALID) != 0;
}

inline bool
isTx81ProfilerNCCCounterValid(const WaferTx81ProfilerTSMCallEvent &event) {
  return (event.metadata & WAFER_TX81_PROFILER_EVENT_NCC_COUNTER_VALID) != 0;
}

} // namespace wafer::runtime

#endif // WAFER_RUNTIME_PROFILERRECORD_H
