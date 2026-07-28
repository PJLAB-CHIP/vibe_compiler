//===- Tx81ProfilerABI.h - TX81 device profiler record ABI ------*- C -*-===//

#ifndef WAFER_ABI_TX81PROFILERABI_H
#define WAFER_ABI_TX81PROFILERABI_H

#include <stddef.h>
#include <stdint.h>

/*
 * Pipeline position:
 * - Upstream artifact / IR: a profiling-only target module, its accepted
 *   instruction sites, and one owned per-tile DDR output allocation.
 * - Current stage responsibility: preserve raw Kcore cycle samples, NCC PMU
 *   snapshots, five NCC engine activity observations, and Direct-DTE
 *   completion observations.
 * - Output artifact / IR: a versioned per-tile byte record.  It is evidence,
 *   not compiler IR, a scheduler hint, or a target package side channel.
 * - Downstream consumer: the host decoder and profile-scoped calibration.
 * - User-level driver / named pipeline: an explicitly selected profiling
 *   target-artifact build followed by the normal board runtime lifecycle.
 * - Explicit non-goals: no completion meaning for an NCC submit return,
 *   no vendor-profiler dependency, no cross-tile clock alignment, and no
 *   production-CRT instrumentation.
 * - Completion gate: structural decode, guards, exact terminal lifecycle, and
 *   separately qualified counter semantics must all succeed before analysis.
 */

#define WAFER_TX81_PROFILER_RECORD_ABI_V3 "wafer-tx81-profiler-record-v3"

#define WAFER_TX81_PROFILER_RECORD_MAGIC UINT64_C(0x3152464f52505757)
#define WAFER_TX81_PROFILER_HEADER_GUARD UINT64_C(0xa3d95f672cb184e0)
#define WAFER_TX81_PROFILER_BUFFER_GUARD UINT64_C(0x6e2ac4d13975bf08)

#define WAFER_TX81_PROFILER_SCHEMA_VERSION 3U
#define WAFER_TX81_PROFILER_LAUNCH_CONFIG_MAGIC UINT64_C(0x3147464352505757)
#define WAFER_TX81_PROFILER_LAUNCH_CONFIG_GUARD UINT64_C(0xd28c4f6173a950be)
#define WAFER_TX81_PROFILER_LAUNCH_CONFIG_BYTES 64U
#define WAFER_TX81_PROFILER_CACHE_LINE_BYTES 64U
#define WAFER_TX81_PROFILER_BUFFER_ALIGNMENT 64U
#define WAFER_TX81_PROFILER_HEADER_BYTES 768U
#define WAFER_TX81_PROFILER_PMU_SNAPSHOT_BYTES 192U
#define WAFER_TX81_PROFILER_COST_SUMMARY_BYTES 64U
#define WAFER_TX81_PROFILER_TSM_CALL_EVENT_BYTES 80U
#define WAFER_TX81_PROFILER_EVENTS_OFFSET WAFER_TX81_PROFILER_HEADER_BYTES
#define WAFER_TX81_PROFILER_BUFFER_GUARD_BYTES 8U
#define WAFER_TX81_PROFILER_MIN_BUFFER_BYTES 832U
#define WAFER_TX81_PROFILER_TRACE_BUFFER_BYTES UINT64_C(1048576)

#define WAFER_TX81_PROFILER_WORKERS 3U
#define WAFER_TX81_PROFILER_QUEUES 5U
#define WAFER_TX81_PROFILER_PMU64_COUNTERS 8U
#define WAFER_TX81_PROFILER_TILE_COUNT 16U

#define WAFER_TX81_PROFILER_INVALID_SITE_ID UINT32_MAX
#define WAFER_TX81_PROFILER_INVALID_SUB_INDEX UINT16_MAX

enum WaferTx81ProfilerEngine {
  WAFER_TX81_PROFILER_ENGINE_CT = 0,
  WAFER_TX81_PROFILER_ENGINE_NE = 1,
  WAFER_TX81_PROFILER_ENGINE_RDMA = 2,
  WAFER_TX81_PROFILER_ENGINE_WDMA = 3,
  WAFER_TX81_PROFILER_ENGINE_TDMA = 4,
  WAFER_TX81_PROFILER_ENGINE_DIRECT_DTE = 5,
  WAFER_TX81_PROFILER_ENGINE_NONE = UINT8_MAX,
};

enum WaferTx81ProfilerEventKind {
  WAFER_TX81_PROFILER_EVENT_NCC_COMMAND = 0,
  WAFER_TX81_PROFILER_EVENT_NCC_COMPLETION_WAIT = 1,
  WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_WAIT = 2,
  WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_PEER_READY_WAIT = 3,
  WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_SETUP_ISSUE = 4,
  WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_COMPLETION_WAIT = 5,
  WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_CLEANUP = 6,
  WAFER_TX81_PROFILER_EVENT_TARGET_SITE = 7,
};

enum WaferTx81ProfilerTraceState {
  WAFER_TX81_PROFILER_TRACE_UNBOUND = 0,
  WAFER_TX81_PROFILER_TRACE_RECORDING = 1,
  WAFER_TX81_PROFILER_TRACE_COMPLETE = 2,
  WAFER_TX81_PROFILER_TRACE_OVERFLOW = 3,
  WAFER_TX81_PROFILER_TRACE_INVALID = 4,
};

enum WaferTx81ProfilerEntryFlag {
  WAFER_TX81_PROFILER_ENTRY_TRACE_ENABLED = UINT32_C(1) << 0,
  WAFER_TX81_PROFILER_ENTRY_COUNT_ONLY = UINT32_C(1) << 1,
};

enum WaferTx81ProfilerRecordFlag {
  WAFER_TX81_PROFILER_RECORD_TRACE_ENABLED = UINT32_C(1) << 0,
  WAFER_TX81_PROFILER_RECORD_ENTRY_BEGUN = UINT32_C(1) << 1,
  WAFER_TX81_PROFILER_RECORD_ENTRY_ENDED = UINT32_C(1) << 2,
  WAFER_TX81_PROFILER_RECORD_OVERFLOW = UINT32_C(1) << 3,
  WAFER_TX81_PROFILER_RECORD_SITE_PROTOCOL_ERROR = UINT32_C(1) << 4,
  WAFER_TX81_PROFILER_RECORD_SUB_INDEX_OVERFLOW = UINT32_C(1) << 5,
  WAFER_TX81_PROFILER_RECORD_PUBLISHED = UINT32_C(1) << 6,
  WAFER_TX81_PROFILER_RECORD_COUNT_ONLY = UINT32_C(1) << 7,
};

enum WaferTx81ProfilerSummaryValidity {
  WAFER_TX81_PROFILER_SUMMARY_ENTRY_CYCLES = UINT32_C(1) << 0,
  WAFER_TX81_PROFILER_SUMMARY_PMU_BEFORE_CAPTURED = UINT32_C(1) << 1,
  WAFER_TX81_PROFILER_SUMMARY_PMU_AFTER_CAPTURED = UINT32_C(1) << 2,
  WAFER_TX81_PROFILER_SUMMARY_PMU_BEFORE_STABLE = UINT32_C(1) << 3,
  WAFER_TX81_PROFILER_SUMMARY_PMU_AFTER_STABLE = UINT32_C(1) << 4,
  WAFER_TX81_PROFILER_SUMMARY_PMU_ENABLE_UNCHANGED = UINT32_C(1) << 5,
  WAFER_TX81_PROFILER_SUMMARY_PMU_RECOVERY_CAPTURED = UINT32_C(1) << 6,
  WAFER_TX81_PROFILER_SUMMARY_PMU_RECOVERY_STABLE = UINT32_C(1) << 7,
  WAFER_TX81_PROFILER_SUMMARY_PMU_RECOVERED = UINT32_C(1) << 8,
};

enum WaferTx81ProfilerEventMetadata {
  WAFER_TX81_PROFILER_EVENT_WORKER_MASK = UINT32_C(0x03),
  WAFER_TX81_PROFILER_EVENT_WORKER_VALID = UINT32_C(1) << 2,
  WAFER_TX81_PROFILER_EVENT_SITE_VALID = UINT32_C(1) << 3,
  WAFER_TX81_PROFILER_EVENT_OBSERVATION_SPAN_VALID = UINT32_C(1) << 4,
  WAFER_TX81_PROFILER_EVENT_SITE_SPAN_VALID = UINT32_C(1) << 5,
  WAFER_TX81_PROFILER_EVENT_OPERATION_SPAN_VALID = UINT32_C(1) << 6,
  WAFER_TX81_PROFILER_EVENT_COUNTER_DELTA_POSITIVE = UINT32_C(1) << 7,
  WAFER_TX81_PROFILER_EVENT_SAME_ENGINE_AMBIGUOUS = UINT32_C(1) << 8,
  WAFER_TX81_PROFILER_EVENT_WAIT_LOCAL = UINT32_C(1) << 9,
  WAFER_TX81_PROFILER_EVENT_WAIT_WORKER = UINT32_C(1) << 10,
  WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_SEND = UINT32_C(1) << 11,
  WAFER_TX81_PROFILER_EVENT_DIRECT_DTE_RECV = UINT32_C(1) << 12,
  WAFER_TX81_PROFILER_EVENT_DTE_COUNTER_VALID = UINT32_C(1) << 13,
  WAFER_TX81_PROFILER_EVENT_NCC_COUNTER_VALID = UINT32_C(1) << 14,
};

/*
 * A profiling-only target wrapper receives one additional pointer-table slot
 * after the normal typed ABI slots.  The slot points at the same owned DDR
 * allocation that becomes the output record.  Before launch the host writes
 * this cache-line-sized configuration at offset zero; entry_begin_from_config
 * copies it locally before replacing it with the record header.  Production
 * wrappers do not have this extra slot.
 */
typedef struct WaferTx81ProfilerLaunchConfig {
  uint64_t magic;
  uint32_t schema_version;
  uint32_t config_bytes;
  uint64_t record_bytes;
  uint32_t tile_id;
  uint32_t flags;
  uint64_t guard;
  uint64_t reserved[3];
} WaferTx81ProfilerLaunchConfig;

/*
 * Counter order is statistics-window, FU union, CT, NE, RDMA, WDMA, TDMA,
 * scalar.  Instruction and blocking arrays are worker-major, then
 * CT/NE/RDMA/WDMA/TDMA.  Values remain raw until their measurement basis is
 * qualified for the exact target profile.
 */
typedef struct WaferTx81ProfilerPMUSnapshot {
  uint64_t counters[WAFER_TX81_PROFILER_PMU64_COUNTERS];
  uint32_t instructions[WAFER_TX81_PROFILER_WORKERS]
                       [WAFER_TX81_PROFILER_QUEUES];
  uint32_t blocking[WAFER_TX81_PROFILER_WORKERS][WAFER_TX81_PROFILER_QUEUES];
  uint32_t enable;
  uint32_t stable_mask;
} WaferTx81ProfilerPMUSnapshot;

/*
 * Trace-run exclusive cost.  The categories are mutually exclusive with each
 * other, but they are overlays on semantic event/site/operation spans and must
 * not be added to those spans.  Status polling and completion-loop
 * bookkeeping belong to the Trace-only replacement for the production
 * vendor blocking NCC completion primitive; the production-common
 * completion-wait semantic is represented separately by its operation span.
 * All inside-entry categories here are therefore instrumentation-only.  Entry
 * setup is measured before entry_begin_cycle; entry teardown begins at
 * entry_end_cycle and ends before the final record publication.  They
 * therefore lie outside the entry axis.
 */
typedef struct WaferTx81ProfilerCostSummary {
  uint64_t ncc_pmu_sample_cycles;
  uint64_t dte_pmu_sample_cycles;
  uint64_t event_bookkeeping_cycles;
  uint64_t status_poll_cycles;
  uint64_t site_hook_cycles;
  uint64_t completion_loop_bookkeeping_cycles;
  uint64_t entry_setup_cycles;
  uint64_t entry_teardown_cycles;
} WaferTx81ProfilerCostSummary;

/*
 * An NCC command event separates the exact vendor NCC issue-call span from
 * the conservative PMU observation window.  counter_delta is the vendor NCC
 * execution-time counter delta in nanoseconds, not Kcore cycles.
 * observation_count remains meaningful for a zero-delta observation.
 * NCC_COUNTER_VALID means every sample needed for this event's bounded
 * counter attribution was stable.  If one engine's split read is unstable,
 * only that engine event loses its counter/window evidence; its site and exact
 * NCC issue-operation span, the other engines, and the record remain usable.
 *
 * Direct-DTE aggregate events separate their productive operation span from
 * split PMU reads.  DTE_COUNTER_VALID requires stable endpoint reads,
 * monotonic channels and a non-overflowing sum; only then is counter_delta the
 * uncalibrated channel-0/channel-1 sum.  Direct-DTE phase and NCC completion
 * events carry operation spans and no PMU delta.  A target-site event is a
 * container carrying the complete typed target-call site span even when the
 * site produces no engine or wait event; it carries no operation/counter span.
 */
typedef struct WaferTx81ProfilerTSMCallEvent {
  uint64_t sequence;
  uint64_t observed_begin_cycle;
  uint64_t observed_end_cycle;
  uint64_t counter_delta;
  uint64_t site_begin_cycle;
  uint64_t site_end_cycle;
  uint64_t operation_begin_cycle;
  uint64_t operation_end_cycle;
  uint32_t site_id;
  uint32_t observation_count;
  uint16_t sub_index;
  uint8_t engine;
  uint8_t kind;
  uint32_t metadata;
} WaferTx81ProfilerTSMCallEvent;

typedef struct WaferTx81ProfilerRecordHeader {
  uint64_t magic;
  uint32_t schema_version;
  uint32_t header_bytes;
  uint32_t event_bytes;
  uint32_t events_offset;
  uint64_t buffer_bytes;
  uint64_t buffer_address;
  uint32_t tile_id;
  uint32_t flags;
  uint32_t trace_state;
  uint32_t event_capacity;
  uint32_t event_count;
  uint32_t dropped_event_count;
  uint64_t next_sequence;
  uint64_t entry_begin_cycle;
  uint64_t entry_end_cycle;
  uint32_t active_site_id;
  uint16_t next_sub_index;
  uint16_t active_site_depth;
  uint32_t summary_validity;
  uint32_t reserved0;
  uint64_t header_guard;
  uint64_t buffer_guard_offset;
  WaferTx81ProfilerPMUSnapshot pmu_before;
  WaferTx81ProfilerPMUSnapshot pmu_after;
  WaferTx81ProfilerPMUSnapshot pmu_recovery;
  WaferTx81ProfilerCostSummary cost_summary;
  uint64_t header_footer_guard;
} WaferTx81ProfilerRecordHeader;

#if defined(__cplusplus)
#define WAFER_TX81_PROFILER_STATIC_ASSERT static_assert
#else
#define WAFER_TX81_PROFILER_STATIC_ASSERT _Static_assert
#endif

WAFER_TX81_PROFILER_STATIC_ASSERT(
    sizeof(WaferTx81ProfilerLaunchConfig) ==
        WAFER_TX81_PROFILER_LAUNCH_CONFIG_BYTES,
    "TX81 profiler launch configuration ABI size changed");
WAFER_TX81_PROFILER_STATIC_ASSERT(
    sizeof(WaferTx81ProfilerPMUSnapshot) ==
        WAFER_TX81_PROFILER_PMU_SNAPSHOT_BYTES,
    "TX81 profiler PMU snapshot ABI size changed");
WAFER_TX81_PROFILER_STATIC_ASSERT(
    sizeof(WaferTx81ProfilerCostSummary) ==
        WAFER_TX81_PROFILER_COST_SUMMARY_BYTES,
    "TX81 profiler cost summary ABI size changed");
WAFER_TX81_PROFILER_STATIC_ASSERT(
    sizeof(WaferTx81ProfilerTSMCallEvent) ==
        WAFER_TX81_PROFILER_TSM_CALL_EVENT_BYTES,
    "TX81 profiler engine event ABI size changed");
WAFER_TX81_PROFILER_STATIC_ASSERT(
    offsetof(WaferTx81ProfilerTSMCallEvent, observed_begin_cycle) == 8U &&
        offsetof(WaferTx81ProfilerTSMCallEvent, counter_delta) == 24U &&
        offsetof(WaferTx81ProfilerTSMCallEvent, site_begin_cycle) == 32U &&
        offsetof(WaferTx81ProfilerTSMCallEvent, operation_begin_cycle) == 48U &&
        offsetof(WaferTx81ProfilerTSMCallEvent, site_id) == 64U &&
        offsetof(WaferTx81ProfilerTSMCallEvent, observation_count) == 68U &&
        offsetof(WaferTx81ProfilerTSMCallEvent, sub_index) == 72U &&
        offsetof(WaferTx81ProfilerTSMCallEvent, metadata) == 76U,
    "TX81 profiler engine event ABI offsets changed");
WAFER_TX81_PROFILER_STATIC_ASSERT(sizeof(WaferTx81ProfilerRecordHeader) ==
                                      WAFER_TX81_PROFILER_HEADER_BYTES,
                                  "TX81 profiler header ABI size changed");
WAFER_TX81_PROFILER_STATIC_ASSERT(offsetof(WaferTx81ProfilerRecordHeader,
                                           pmu_before) == 120U,
                                  "TX81 profiler PMU-before offset changed");
WAFER_TX81_PROFILER_STATIC_ASSERT(offsetof(WaferTx81ProfilerRecordHeader,
                                           pmu_after) == 312U,
                                  "TX81 profiler PMU-after offset changed");
WAFER_TX81_PROFILER_STATIC_ASSERT(offsetof(WaferTx81ProfilerRecordHeader,
                                           pmu_recovery) == 504U,
                                  "TX81 profiler PMU-recovery offset changed");
WAFER_TX81_PROFILER_STATIC_ASSERT(offsetof(WaferTx81ProfilerRecordHeader,
                                           cost_summary) == 696U,
                                  "TX81 profiler cost summary offset changed");
WAFER_TX81_PROFILER_STATIC_ASSERT(offsetof(WaferTx81ProfilerRecordHeader,
                                           header_footer_guard) == 760U,
                                  "TX81 profiler header footer offset changed");
WAFER_TX81_PROFILER_STATIC_ASSERT(
    (WAFER_TX81_PROFILER_BUFFER_ALIGNMENT &
     (WAFER_TX81_PROFILER_BUFFER_ALIGNMENT - 1U)) == 0U,
    "TX81 profiler buffer alignment must be a power of two");

#undef WAFER_TX81_PROFILER_STATIC_ASSERT

#endif // WAFER_ABI_TX81PROFILERABI_H
