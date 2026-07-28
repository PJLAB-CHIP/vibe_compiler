//===- ProfilerRecord.h - TX81 profiler host contract ---------*- C++ -*-===//

#ifndef WAFER_RUNTIME_PROFILERRECORD_H
#define WAFER_RUNTIME_PROFILERRECORD_H

#include "Wafer/ABI/Tx81ProfilerABI.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace wafer::runtime {

/// One structurally decoded per-tile profiler record. In schema v2 the fixed
/// event field named raw_return contains an exact NCC busy-cycle delta or, only
/// when isTx81ProfilerDirectDTECounterValid is true, an uncalibrated
/// Direct-DTE PMU delta. begin/end bound the NCC observation or the real
/// Direct-DTE wait/completion window.
struct Tx81ProfilerRecord {
  WaferTx81ProfilerRecordHeader header{};
  std::vector<WaferTx81ProfilerTSMCallEvent> events;
};

enum class Tx81ProfilerCaptureKind {
  Summary,
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
/// verifyTx81ProfilerTileDomain rejects them for a successful campaign.
llvm::Expected<Tx81ProfilerRecord>
decodeTx81ProfilerRecord(llvm::ArrayRef<uint8_t> bytes);

/// Requires all-and-only tile ids 0..15, a common record contract and
/// complete, non-overflowing records. Trace records additionally require a
/// valid static site identity on every stored event. Worker attribution
/// remains optional and is governed solely by the event validity bit.
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
isTx81ProfilerActivityValid(const WaferTx81ProfilerTSMCallEvent &event) {
  return (event.metadata & WAFER_TX81_PROFILER_EVENT_ACTIVITY_VALID) != 0;
}

inline bool
isTx81ProfilerDirectDTESend(const WaferTx81ProfilerTSMCallEvent &event) {
  return (event.metadata & WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_SEND) != 0;
}

inline bool
isTx81ProfilerDirectDTEReceive(const WaferTx81ProfilerTSMCallEvent &event) {
  return (event.metadata & WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_RECV) != 0;
}

inline bool
isTx81ProfilerDirectDTECounterValid(
    const WaferTx81ProfilerTSMCallEvent &event) {
  return (event.metadata & WAFER_TX81_PROFILER_EVENT_DTE_COUNTER_VALID) != 0;
}

} // namespace wafer::runtime

#endif // WAFER_RUNTIME_PROFILERRECORD_H
