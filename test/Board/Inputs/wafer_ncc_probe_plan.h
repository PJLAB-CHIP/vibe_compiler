#ifndef WAFER_NCC_PROBE_PLAN_H
#define WAFER_NCC_PROBE_PLAN_H

#include "wafer_ncc_probe_protocol.h"

#include <stdint.h>

typedef struct WaferNccProbeLane {
  uint32_t engine;
  uint32_t worker;
  uint32_t issue_mode;
  uint32_t transfer_bytes;
  uint32_t element_format;
  uint32_t layout_kind;
  uint32_t layout_inner_bytes;
  uint32_t layout_stride0_bytes;
  uint32_t layout_stride1_bytes;
  uint32_t layout_stride2_bytes;
  uint32_t layout_iteration0;
  uint32_t layout_iteration1;
  uint32_t layout_iteration2;
  uint32_t flags;
} WaferNccProbeLane;

typedef struct WaferNccProbeRequest {
  uint32_t command;
  uint32_t lane_count;
  uint32_t rounds;
  uint32_t effect_relation;
  uint32_t range_relation;
  uint32_t schedule;
  uint32_t wait_kind;
  uint32_t wait_worker_mask;
  uint32_t first_operand;
  uint32_t second_operand;
  uint32_t issue_limit;
  uint64_t seed;
  uint64_t sample;
  uint64_t flags;
  WaferNccProbeLane lanes[WAFER_NCC_PROTOCOL_MAX_LANES];
} WaferNccProbeRequest;

typedef struct WaferNccProbeIssue {
  uint32_t ordinal;
  uint32_t lane;
  uint32_t round;
  uint32_t slot;
  uint32_t engine;
  uint32_t worker;
  uint64_t tag;
  const WaferNccProbeLane *lane_spec;
} WaferNccProbeIssue;

typedef struct WaferNccProbeObservation {
  uint64_t execute_rc;
  uint64_t inter_type;
  uint64_t read0_begin;
  uint64_t read0_end;
  uint64_t read1_begin;
  uint64_t read1_end;
  uint64_t write_begin;
  uint64_t write_end;
  uint64_t flags;
} WaferNccProbeObservation;

/*
 * An adapter owns all engine-specific packet, address and golden behavior.
 * The generic executor never switches on engine or recovers semantics from a
 * case name.  Adapter storage is indexed by the issue's unique slot.
 */
typedef struct WaferNccProbeEngineAdapter {
  uint32_t engine;
  uint32_t queue_depth;
  uint32_t issue_mode_mask;
  uint32_t (*memory_effects)(const WaferNccProbeRequest *request,
                             const WaferNccProbeLane *lane);
  int (*seed)(void *context, const WaferNccProbeRequest *request,
              const WaferNccProbeIssue *issue);
  int (*prepare)(void *context, const WaferNccProbeRequest *request,
                 const WaferNccProbeIssue *issue,
                 uint64_t *preparation_flags);
  int (*issue)(void *context, const WaferNccProbeRequest *request,
               const WaferNccProbeIssue *issue, uint64_t *execute_rc);
  int (*observe)(void *context, const WaferNccProbeRequest *request,
                 const WaferNccProbeIssue *issue,
                 WaferNccProbeObservation *observation);
  int (*oracle)(void *context, const WaferNccProbeRequest *request,
                const WaferNccProbeIssue *issue, uint32_t phase,
                uint64_t *result_mismatches, uint64_t *guard_mismatches);
  void (*release)(void *context, const WaferNccProbeIssue *issue);
} WaferNccProbeEngineAdapter;

enum WaferNccProbeSnapshotPhase {
  WAFER_NCC_SNAPSHOT_BEFORE = 0,
  WAFER_NCC_SNAPSHOT_BOUNDARY = 1,
  WAFER_NCC_SNAPSHOT_FINAL = 2,
};

typedef struct WaferNccProbeExecutionHooks {
  /*
   * Publish all seed-side memory effects before packet construction or issue.
   * Device probes use this boundary to make weak-order Kcore SPM stores
   * visible to NCC; it is deliberately outside the measured issue window.
   */
  int (*seed_complete)(void *context);
  int (*snapshot)(void *context, uint32_t phase,
                  volatile uint64_t *record_words);
  int (*serial_drain)(void *context, uint32_t worker_mask);
  int (*requested_wait)(void *context, uint32_t wait_kind,
                        uint32_t worker_mask);
  int (*safety_drain)(void *context, uint32_t worker_mask);
  uint64_t (*read_cycle)(void *context);
  uint64_t (*read_worker_control)(void *context, uint32_t worker);
} WaferNccProbeExecutionHooks;

uint32_t wafer_ncc_probe_decode_request(const volatile uint64_t *request_words,
                                        WaferNccProbeRequest *request);

uint32_t wafer_ncc_probe_validate_plan(
    const WaferNccProbeRequest *request,
    const WaferNccProbeEngineAdapter *adapters, uint32_t adapter_count);

int wafer_ncc_probe_is_strided_dependency(
    const WaferNccProbeRequest *request);

uint32_t wafer_ncc_probe_execute_plan(
    const WaferNccProbeRequest *request,
    const WaferNccProbeEngineAdapter *adapters, uint32_t adapter_count,
    const WaferNccProbeExecutionHooks *hooks, void *context,
    volatile uint64_t *record_words);

#endif
