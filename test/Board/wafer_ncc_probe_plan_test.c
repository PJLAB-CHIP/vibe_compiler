#include "wafer_ncc_probe_plan.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

/* Keep test checks active in Release builds where CMake defines NDEBUG. */
#undef assert
#define assert(condition)                                                       \
  do {                                                                          \
    if (!(condition))                                                           \
      abort();                                                                  \
  } while (0)

enum MockEventKind {
  MOCK_SEED = 1000,
  MOCK_PREPARE = 2000,
  MOCK_SNAPSHOT_BEFORE = 3000,
  MOCK_SNAPSHOT_BOUNDARY = 3001,
  MOCK_SNAPSHOT_FINAL = 3002,
  MOCK_ISSUE = 4000,
  MOCK_READ_CONTROL = 4500,
  MOCK_OBSERVE = 5000,
  MOCK_SERIAL_DRAIN = 6000,
  MOCK_REQUESTED_WAIT = 7000,
  MOCK_ORACLE_BOUNDARY = 8000,
  MOCK_SAFETY_DRAIN = 9000,
  MOCK_ORACLE_FINAL = 10000,
  MOCK_RELEASE = 11000,
};

typedef struct MockContext {
  uint32_t events[256];
  uint32_t event_count;
  uint64_t tags[WAFER_NCC_PROTOCOL_MAX_ISSUES];
  uint32_t seeded_mask;
  uint32_t prepared_mask;
  uint32_t issued_mask;
  uint32_t safety_drain_calls;
  uint64_t cycle;
  int fail_safety_drain;
} MockContext;

static void mock_event(MockContext *context, uint32_t event) {
  assert(context->event_count <
         sizeof(context->events) / sizeof(context->events[0]));
  context->events[context->event_count++] = event;
}

static int mock_seed(void *opaque, const WaferNccProbeRequest *request,
                     const WaferNccProbeIssue *issue) {
  (void)request;
  MockContext *context = (MockContext *)opaque;
  uint32_t bit = UINT32_C(1) << issue->slot;
  assert((context->seeded_mask & bit) == 0);
  assert(issue->tag != 0);
  for (uint32_t slot = 0; slot < WAFER_NCC_PROTOCOL_MAX_ISSUES; ++slot)
    if ((context->seeded_mask & (UINT32_C(1) << slot)) != 0)
      assert(context->tags[slot] != issue->tag);
  context->tags[issue->slot] = issue->tag;
  context->seeded_mask |= bit;
  mock_event(context, MOCK_SEED + issue->slot);
  return 0;
}

static int mock_prepare(void *opaque, const WaferNccProbeRequest *request,
                        const WaferNccProbeIssue *issue) {
  (void)request;
  MockContext *context = (MockContext *)opaque;
  uint32_t bit = UINT32_C(1) << issue->slot;
  assert((context->seeded_mask & bit) != 0);
  assert((context->prepared_mask & bit) == 0);
  context->prepared_mask |= bit;
  mock_event(context, MOCK_PREPARE + issue->slot);
  return 0;
}

static int mock_issue(void *opaque, const WaferNccProbeRequest *request,
                      const WaferNccProbeIssue *issue, uint64_t *execute_rc) {
  (void)request;
  MockContext *context = (MockContext *)opaque;
  uint32_t bit = UINT32_C(1) << issue->slot;
  assert((context->prepared_mask & bit) != 0);
  assert((context->issued_mask & bit) == 0);
  context->issued_mask |= bit;
  *execute_rc = UINT64_C(0x100) + issue->slot;
  mock_event(context, MOCK_ISSUE + issue->slot);
  return 0;
}

static int mock_observe(void *opaque, const WaferNccProbeRequest *request,
                        const WaferNccProbeIssue *issue,
                        WaferNccProbeObservation *observation) {
  (void)request;
  MockContext *context = (MockContext *)opaque;
  assert((context->issued_mask & (UINT32_C(1) << issue->slot)) != 0);
  observation->inter_type =
      (uint64_t)issue->engine | ((uint64_t)issue->worker << 8);
  observation->read0_begin = UINT64_C(0x10000) + issue->slot * 0x100U;
  observation->read0_end = observation->read0_begin + 0xffU;
  observation->read1_begin = UINT64_C(0x18000) + issue->slot * 0x100U;
  observation->read1_end = observation->read1_begin + 0xffU;
  observation->write_begin =
      UINT64_C(0x20000) + issue->slot * 0x100U;
  observation->write_end = observation->write_begin + 0xffU;
  observation->flags =
      WAFER_NCC_ISSUE_READ0_VALID | WAFER_NCC_ISSUE_READ1_VALID |
      WAFER_NCC_ISSUE_WRITE_VALID | WAFER_NCC_ISSUE_PACKET_OBSERVED;
  mock_event(context, MOCK_OBSERVE + issue->slot);
  return 0;
}

static int mock_oracle(void *opaque, const WaferNccProbeRequest *request,
                       const WaferNccProbeIssue *issue, uint32_t phase,
                       uint64_t *result_mismatches,
                       uint64_t *guard_mismatches) {
  (void)request;
  MockContext *context = (MockContext *)opaque;
  assert((context->issued_mask & (UINT32_C(1) << issue->slot)) != 0);
  *result_mismatches = 0;
  *guard_mismatches = 0;
  if (phase == WAFER_NCC_ORACLE_BOUNDARY)
    mock_event(context, MOCK_ORACLE_BOUNDARY + issue->slot);
  else {
    assert(phase == WAFER_NCC_ORACLE_FINAL);
    mock_event(context, MOCK_ORACLE_FINAL + issue->slot);
  }
  return 0;
}

static void mock_release(void *opaque, const WaferNccProbeIssue *issue) {
  MockContext *context = (MockContext *)opaque;
  assert((context->prepared_mask & (UINT32_C(1) << issue->slot)) != 0);
  context->prepared_mask &= ~(UINT32_C(1) << issue->slot);
  mock_event(context, MOCK_RELEASE + issue->slot);
}

static int mock_snapshot(void *opaque, uint32_t phase,
                         volatile uint64_t *record) {
  (void)record;
  MockContext *context = (MockContext *)opaque;
  if (phase == WAFER_NCC_SNAPSHOT_BEFORE)
    mock_event(context, MOCK_SNAPSHOT_BEFORE);
  else if (phase == WAFER_NCC_SNAPSHOT_BOUNDARY)
    mock_event(context, MOCK_SNAPSHOT_BOUNDARY);
  else {
    assert(phase == WAFER_NCC_SNAPSHOT_FINAL);
    mock_event(context, MOCK_SNAPSHOT_FINAL);
  }
  return 0;
}

static int mock_serial_drain(void *opaque, uint32_t worker_mask) {
  MockContext *context = (MockContext *)opaque;
  assert(worker_mask != 0);
  mock_event(context, MOCK_SERIAL_DRAIN + worker_mask);
  return 0;
}

static int mock_requested_wait(void *opaque, uint32_t wait_kind,
                               uint32_t worker_mask) {
  MockContext *context = (MockContext *)opaque;
  assert(wait_kind != WAFER_NCC_WAIT_NONE);
  if (wait_kind == WAFER_NCC_WAIT_BY_WORKER)
    assert(worker_mask != 0);
  mock_event(context, MOCK_REQUESTED_WAIT);
  return 0;
}

static int mock_safety_drain(void *opaque, uint32_t worker_mask) {
  MockContext *context = (MockContext *)opaque;
  assert(worker_mask != 0);
  ++context->safety_drain_calls;
  mock_event(context, MOCK_SAFETY_DRAIN);
  return context->fail_safety_drain;
}

static uint64_t mock_read_cycle(void *opaque) {
  MockContext *context = (MockContext *)opaque;
  context->cycle += 11;
  return context->cycle;
}

static uint64_t mock_read_worker_control(void *opaque, uint32_t worker) {
  MockContext *context = (MockContext *)opaque;
  uint32_t issued = 0;
  for (uint32_t slot = 0; slot < WAFER_NCC_PROTOCOL_MAX_ISSUES; ++slot)
    issued += (context->issued_mask >> slot) & 1U;
  mock_event(context, MOCK_READ_CONTROL + issued - 1U);
  return (uint64_t)(issued & UINT8_MAX) | ((uint64_t)worker << 16);
}

static const WaferNccProbeExecutionHooks hooks = {
    mock_snapshot,
    mock_serial_drain,
    mock_requested_wait,
    mock_safety_drain,
    mock_read_cycle,
    mock_read_worker_control,
};

static uint32_t mock_effects(const WaferNccProbeRequest *request,
                             const WaferNccProbeLane *lane_spec) {
  (void)request;
  (void)lane_spec;
  return WAFER_NCC_MEMORY_READ0 | WAFER_NCC_MEMORY_READ1 |
         WAFER_NCC_MEMORY_WRITE;
}

static const WaferNccProbeEngineAdapter adapters[] = {
    {WAFER_NCC_ENGINE_CT, 6, UINT32_C(3), mock_effects, mock_seed, mock_prepare,
     mock_issue, mock_observe, mock_oracle, mock_release},
    {WAFER_NCC_ENGINE_NE, 6, UINT32_C(3), mock_effects, mock_seed, mock_prepare,
     mock_issue, mock_observe, mock_oracle, mock_release},
    {WAFER_NCC_ENGINE_RDMA, 6, UINT32_C(3), mock_effects, mock_seed,
     mock_prepare, mock_issue, mock_observe, mock_oracle, mock_release},
    {WAFER_NCC_ENGINE_WDMA, 6, UINT32_C(3), mock_effects, mock_seed,
     mock_prepare, mock_issue, mock_observe, mock_oracle, mock_release},
    {WAFER_NCC_ENGINE_TDMA, 4, UINT32_C(3), mock_effects, mock_seed,
     mock_prepare, mock_issue, mock_observe, mock_oracle, mock_release},
};

static WaferNccProbeLane lane(uint32_t engine, uint32_t worker) {
  WaferNccProbeLane result = {0};
  result.engine = engine;
  result.worker = worker;
  result.issue_mode = WAFER_NCC_ISSUE_RAW;
  result.transfer_bytes = 4096;
  result.element_format = 1;
  result.layout_kind = WAFER_NCC_LAYOUT_CONTIGUOUS;
  return result;
}

static WaferNccProbeRequest request(uint32_t lanes, uint32_t rounds,
                                    uint32_t schedule) {
  WaferNccProbeRequest result = {0};
  result.command = WAFER_NCC_COMMAND_EXECUTE;
  result.lane_count = lanes;
  result.rounds = rounds;
  result.effect_relation = WAFER_NCC_EFFECT_NONE;
  result.range_relation = WAFER_NCC_RANGE_DISJOINT;
  result.schedule = schedule;
  result.wait_kind = WAFER_NCC_WAIT_BY_WORKER;
  result.wait_worker_mask = 1;
  result.first_operand = WAFER_NCC_OPERAND_AUTO;
  result.second_operand = WAFER_NCC_OPERAND_AUTO;
  result.seed = UINT64_C(0x123456789abcdef0);
  return result;
}

static uint32_t find_event(const MockContext *context, uint32_t event) {
  for (uint32_t index = 0; index < context->event_count; ++index)
    if (context->events[index] == event)
      return index;
  assert(0 && "expected event is missing");
  return UINT32_MAX;
}

static void test_dual_lane_boundary_order(void) {
  WaferNccProbeRequest plan = request(2, 4, WAFER_NCC_SCHEDULE_WINDOW);
  plan.lanes[0] = lane(WAFER_NCC_ENGINE_RDMA, 0);
  plan.lanes[1] = lane(WAFER_NCC_ENGINE_CT, 1);
  plan.wait_worker_mask = 3;
  plan.effect_relation = WAFER_NCC_EFFECT_RAW;
  plan.range_relation = WAFER_NCC_RANGE_EXACT;
  plan.first_operand = WAFER_NCC_OPERAND_WRITE;
  plan.second_operand = WAFER_NCC_OPERAND_READ0;

  MockContext context = {0};
  uint64_t record[WAFER_NCC_PROTOCOL_RECORD_WORDS];
  assert(wafer_ncc_probe_execute_plan(
             &plan, adapters, sizeof(adapters) / sizeof(adapters[0]), &hooks,
             &context, record) == WAFER_NCC_STATUS_OK);
  assert(record[WAFER_NCC_REC_ISSUE_COUNT] == 8);
  assert(record[WAFER_NCC_REC_STATUS] == WAFER_NCC_STATUS_OK);
  assert(record[WAFER_NCC_REC_FLAGS] ==
         (WAFER_NCC_RECORD_BEFORE_CAPTURED |
          WAFER_NCC_RECORD_REQUESTED_WAIT_DONE |
          WAFER_NCC_RECORD_BOUNDARY_CAPTURED |
          WAFER_NCC_RECORD_BOUNDARY_ORACLE_DONE |
          WAFER_NCC_RECORD_SAFETY_DRAIN_DONE |
          WAFER_NCC_RECORD_FINAL_CAPTURED |
          WAFER_NCC_RECORD_FINAL_ORACLE_DONE));

  const uint32_t expected_slots[] = {0, 4, 1, 5, 2, 6, 3, 7};
  uint32_t previous = 0;
  for (uint32_t index = 0;
       index < sizeof(expected_slots) / sizeof(expected_slots[0]); ++index) {
    uint32_t position =
        find_event(&context, MOCK_ISSUE + expected_slots[index]);
    uint32_t control =
        find_event(&context, MOCK_READ_CONTROL + index);
    uint32_t observe =
        find_event(&context, MOCK_OBSERVE + expected_slots[index]);
    assert(position < control);
    assert(control < observe);
    if (index != 0)
      assert(position > previous);
    previous = position;
  }
  uint32_t boundary_snapshot = find_event(&context, MOCK_SNAPSHOT_BOUNDARY);
  uint32_t safety_drain = find_event(&context, MOCK_SAFETY_DRAIN);
  uint32_t final_snapshot = find_event(&context, MOCK_SNAPSHOT_FINAL);
  assert(boundary_snapshot < safety_drain);
  for (uint32_t slot = 0; slot < 8; ++slot)
    assert(find_event(&context, MOCK_ORACLE_BOUNDARY + slot) < safety_drain);
  assert(safety_drain < final_snapshot);
  for (uint32_t slot = 0; slot < 8; ++slot)
    assert(find_event(&context, MOCK_ORACLE_FINAL + slot) > safety_drain);

  uint64_t seen_tags[8] = {0};
  for (uint32_t issue = 0; issue < 8; ++issue) {
    uint32_t base =
        wafer_ncc_protocol_issue_word(issue, WAFER_NCC_ISSUE_ORDINAL);
    assert(record[base + WAFER_NCC_ISSUE_ORDINAL] == issue);
    assert(record[base + WAFER_NCC_ISSUE_SLOT] < 8);
    seen_tags[issue] = record[base + WAFER_NCC_ISSUE_TAG];
    for (uint32_t previous_tag = 0; previous_tag < issue; ++previous_tag)
      assert(seen_tags[previous_tag] != seen_tags[issue]);
  }
  assert(context.prepared_mask == 0);
}

static void test_three_lane_disjoint_window_and_serial(void) {
  const uint32_t schedules[] = {WAFER_NCC_SCHEDULE_WINDOW,
                                WAFER_NCC_SCHEDULE_SERIAL};
  for (uint32_t schedule_index = 0; schedule_index < 2; ++schedule_index) {
    WaferNccProbeRequest plan = request(3, 4, schedules[schedule_index]);
    plan.lanes[0] = lane(WAFER_NCC_ENGINE_RDMA, 0);
    plan.lanes[1] = lane(WAFER_NCC_ENGINE_CT, 0);
    plan.lanes[2] = lane(WAFER_NCC_ENGINE_WDMA, 0);
    MockContext context = {0};
    uint64_t record[WAFER_NCC_PROTOCOL_RECORD_WORDS];
    assert(wafer_ncc_probe_execute_plan(
               &plan, adapters, sizeof(adapters) / sizeof(adapters[0]), &hooks,
               &context, record) == WAFER_NCC_STATUS_OK);
    assert(record[WAFER_NCC_REC_ISSUE_COUNT] == 12);
    const uint32_t expected_slots[] = {0, 4, 8, 1, 5, 9,
                                       2, 6, 10, 3, 7, 11};
    uint32_t previous = 0;
    for (uint32_t index = 0; index < 12; ++index) {
      uint32_t position =
          find_event(&context, MOCK_ISSUE + expected_slots[index]);
      if (index != 0)
        assert(position > previous);
      previous = position;
    }
    uint32_t serial_drains = 0;
    for (uint32_t index = 0; index < context.event_count; ++index)
      serial_drains +=
          context.events[index] >= MOCK_SERIAL_DRAIN &&
          context.events[index] < MOCK_REQUESTED_WAIT;
    assert(serial_drains ==
           (schedules[schedule_index] == WAFER_NCC_SCHEDULE_SERIAL ? 12U
                                                                   : 0U));
  }
}

static void test_validation_bounds(void) {
  WaferNccProbeRequest unsafe = request(2, 4, WAFER_NCC_SCHEDULE_WINDOW);
  unsafe.lanes[0] = lane(WAFER_NCC_ENGINE_CT, 0);
  unsafe.lanes[1] = lane(WAFER_NCC_ENGINE_CT, 0);
  assert(wafer_ncc_probe_validate_plan(
             &unsafe, adapters,
             sizeof(adapters) / sizeof(adapters[0])) ==
         WAFER_NCC_STATUS_UNSAFE_WINDOW);
  unsafe.schedule = WAFER_NCC_SCHEDULE_SERIAL;
  assert(wafer_ncc_probe_validate_plan(
             &unsafe, adapters,
             sizeof(adapters) / sizeof(adapters[0])) ==
         WAFER_NCC_STATUS_OK);

  WaferNccProbeRequest ambiguous = request(3, 2, WAFER_NCC_SCHEDULE_WINDOW);
  ambiguous.lanes[0] = lane(WAFER_NCC_ENGINE_RDMA, 0);
  ambiguous.lanes[1] = lane(WAFER_NCC_ENGINE_CT, 0);
  ambiguous.lanes[2] = lane(WAFER_NCC_ENGINE_WDMA, 0);
  ambiguous.effect_relation = WAFER_NCC_EFFECT_RAW;
  ambiguous.range_relation = WAFER_NCC_RANGE_EXACT;
  assert(wafer_ncc_probe_validate_plan(
             &ambiguous, adapters,
             sizeof(adapters) / sizeof(adapters[0])) ==
         WAFER_NCC_STATUS_UNSUPPORTED_COMBINATION);

  ambiguous.effect_relation = WAFER_NCC_EFFECT_NONE;
  ambiguous.range_relation = WAFER_NCC_RANGE_DISJOINT;
  ambiguous.lanes[0].flags = 1;
  assert(wafer_ncc_probe_validate_plan(
             &ambiguous, adapters,
             sizeof(adapters) / sizeof(adapters[0])) ==
         WAFER_NCC_STATUS_BAD_REQUEST);

  WaferNccProbeRequest raw = request(2, 2, WAFER_NCC_SCHEDULE_WINDOW);
  raw.lanes[0] = lane(WAFER_NCC_ENGINE_RDMA, 0);
  raw.lanes[1] = lane(WAFER_NCC_ENGINE_CT, 0);
  raw.effect_relation = WAFER_NCC_EFFECT_RAW;
  raw.range_relation = WAFER_NCC_RANGE_EXACT;
  raw.first_operand = WAFER_NCC_OPERAND_WRITE;
  raw.second_operand = WAFER_NCC_OPERAND_READ0;
  assert(wafer_ncc_probe_validate_plan(
             &raw, adapters, sizeof(adapters) / sizeof(adapters[0])) ==
         WAFER_NCC_STATUS_OK);
  raw.second_operand = WAFER_NCC_OPERAND_WRITE;
  assert(wafer_ncc_probe_validate_plan(
             &raw, adapters, sizeof(adapters) / sizeof(adapters[0])) ==
         WAFER_NCC_STATUS_UNSUPPORTED_COMBINATION);
}

static void test_wire_decode(void) {
  uint64_t words[WAFER_NCC_PROTOCOL_REQUEST_WORDS] = {0};
  words[WAFER_NCC_REQ_MAGIC] = WAFER_NCC_PROTOCOL_REQUEST_MAGIC;
  words[WAFER_NCC_REQ_SCHEMA_AND_WORDS] =
      ((uint64_t)WAFER_NCC_PROTOCOL_SCHEMA << 32) |
      WAFER_NCC_PROTOCOL_REQUEST_WORDS;
  words[WAFER_NCC_REQ_COMMAND] = WAFER_NCC_COMMAND_EXECUTE;
  words[WAFER_NCC_REQ_LANE_COUNT] = 1;
  words[WAFER_NCC_REQ_ROUNDS] = 2;
  words[WAFER_NCC_REQ_EFFECT_RELATION] = WAFER_NCC_EFFECT_NONE;
  words[WAFER_NCC_REQ_RANGE_RELATION] = WAFER_NCC_RANGE_DISJOINT;
  words[WAFER_NCC_REQ_SCHEDULE] = WAFER_NCC_SCHEDULE_WINDOW;
  words[WAFER_NCC_REQ_WAIT_KIND] = WAFER_NCC_WAIT_BY_WORKER;
  words[WAFER_NCC_REQ_WAIT_WORKER_MASK] = 1;
  words[WAFER_NCC_REQ_SEED] = 17;
  words[WAFER_NCC_REQ_FIRST_OPERAND] = WAFER_NCC_OPERAND_AUTO;
  words[WAFER_NCC_REQ_SECOND_OPERAND] = WAFER_NCC_OPERAND_AUTO;
  words[wafer_ncc_protocol_lane_word(0, WAFER_NCC_LANE_ENGINE)] =
      WAFER_NCC_ENGINE_TDMA;
  words[wafer_ncc_protocol_lane_word(0, WAFER_NCC_LANE_WORKER)] = 0;
  words[wafer_ncc_protocol_lane_word(0, WAFER_NCC_LANE_ISSUE_MODE)] =
      WAFER_NCC_ISSUE_RAW;
  words[wafer_ncc_protocol_lane_word(0, WAFER_NCC_LANE_TRANSFER_BYTES)] = 256;
  words[wafer_ncc_protocol_lane_word(0, WAFER_NCC_LANE_ELEMENT_FORMAT)] = 1;
  words[wafer_ncc_protocol_lane_word(0, WAFER_NCC_LANE_LAYOUT_KIND)] =
      WAFER_NCC_LAYOUT_CONTIGUOUS;

  WaferNccProbeRequest decoded;
  assert(wafer_ncc_probe_decode_request(words, &decoded) ==
         WAFER_NCC_STATUS_OK);
  assert(decoded.lane_count == 1);
  assert(decoded.rounds == 2);
  assert(decoded.lanes[0].engine == WAFER_NCC_ENGINE_TDMA);
  assert(decoded.lanes[0].transfer_bytes == 256);
  assert(wafer_ncc_probe_validate_plan(
             &decoded, adapters,
             sizeof(adapters) / sizeof(adapters[0])) ==
         WAFER_NCC_STATUS_OK);

  words[WAFER_NCC_REQ_COMMAND] |= UINT64_C(1) << 32;
  assert(wafer_ncc_probe_decode_request(words, &decoded) ==
         WAFER_NCC_STATUS_BAD_REQUEST);
  words[WAFER_NCC_REQ_COMMAND] &= UINT32_MAX;

  words[WAFER_NCC_REQ_ISSUE_LIMIT] = UINT64_C(1) << 32;
  assert(wafer_ncc_probe_decode_request(words, &decoded) ==
         WAFER_NCC_STATUS_BAD_REQUEST);
  words[WAFER_NCC_REQ_ISSUE_LIMIT] = 0;

  words[wafer_ncc_protocol_lane_word(1, WAFER_NCC_LANE_ENGINE)] = 1;
  assert(wafer_ncc_probe_decode_request(words, &decoded) ==
         WAFER_NCC_STATUS_BAD_REQUEST);
  words[wafer_ncc_protocol_lane_word(1, WAFER_NCC_LANE_ENGINE)] = 0;

  words[wafer_ncc_protocol_lane_word(0, WAFER_NCC_LANE_WORKER)] =
      UINT64_C(1) << 32;
  assert(wafer_ncc_probe_decode_request(words, &decoded) ==
         WAFER_NCC_STATUS_BAD_REQUEST);
}

static void test_failed_safety_drain_is_not_retried(void) {
  WaferNccProbeRequest plan = request(1, 1, WAFER_NCC_SCHEDULE_WINDOW);
  plan.lanes[0] = lane(WAFER_NCC_ENGINE_CT, 0);
  MockContext context = {0};
  context.fail_safety_drain = 1;
  uint64_t record[WAFER_NCC_PROTOCOL_RECORD_WORDS];
  assert(wafer_ncc_probe_execute_plan(
             &plan, adapters, sizeof(adapters) / sizeof(adapters[0]), &hooks,
             &context, record) == WAFER_NCC_STATUS_DRAIN_FAILED);
  assert(context.safety_drain_calls == 1);
  assert((record[WAFER_NCC_REC_FLAGS] &
          WAFER_NCC_RECORD_SAFETY_DRAIN_DONE) == 0);
  assert(context.prepared_mask == 0);
}

static void test_depth_plus_one_is_tight_and_waited(void) {
  WaferNccProbeRequest plan = request(2, 4, WAFER_NCC_SCHEDULE_WINDOW);
  plan.lanes[0] = lane(WAFER_NCC_ENGINE_CT, 0);
  plan.lanes[1] = plan.lanes[0];
  plan.flags = WAFER_NCC_REQUEST_TIGHT_DEPTH_PLUS_ONE;
  plan.issue_limit = 7;

  MockContext context = {0};
  uint64_t record[WAFER_NCC_PROTOCOL_RECORD_WORDS];
  assert(wafer_ncc_probe_execute_plan(
             &plan, adapters, sizeof(adapters) / sizeof(adapters[0]), &hooks,
             &context, record) == WAFER_NCC_STATUS_OK);
  assert(record[WAFER_NCC_REC_ISSUE_COUNT] == 7);
  assert(record[WAFER_NCC_REC_ISSUE_LIMIT] == 7);
  static const uint32_t expected_slots[] = {0, 4, 1, 5, 2, 6, 3};
  uint32_t last_issue =
      find_event(&context, MOCK_ISSUE + expected_slots[6]);
  uint32_t control = find_event(&context, MOCK_READ_CONTROL + 6);
  assert(last_issue < control);
  for (uint32_t index = 0; index < 7; ++index) {
    assert(find_event(&context, MOCK_ISSUE + expected_slots[index]) < control);
    assert(find_event(&context, MOCK_OBSERVE + expected_slots[index]) >
           control);
    uint32_t base =
        wafer_ncc_protocol_issue_word(index, WAFER_NCC_ISSUE_ORDINAL);
    assert(record[base + WAFER_NCC_ISSUE_EXECUTE_CYCLES] == 11);
    assert(record[base + WAFER_NCC_ISSUE_CONTROL_AFTER_ISSUE] ==
           (index == 3 ? 7U : 0U));
    assert((record[base + WAFER_NCC_ISSUE_FLAGS] &
            WAFER_NCC_ISSUE_WINDOW_CONTROL_VALID) ==
           (index == 3 ? WAFER_NCC_ISSUE_WINDOW_CONTROL_VALID : 0U));
  }
  assert(context.prepared_mask == 0);

  plan.wait_kind = WAFER_NCC_WAIT_NONE;
  plan.wait_worker_mask = 0;
  assert(wafer_ncc_probe_validate_plan(
             &plan, adapters, sizeof(adapters) / sizeof(adapters[0])) ==
         WAFER_NCC_STATUS_UNSUPPORTED_COMBINATION);
}

static void test_full_depth_window_is_accepted_without_overflow(void) {
  WaferNccProbeRequest plan = request(1, 4, WAFER_NCC_SCHEDULE_WINDOW);
  plan.lanes[0] = lane(WAFER_NCC_ENGINE_TDMA, 0);
  assert(wafer_ncc_probe_validate_plan(
             &plan, adapters, sizeof(adapters) / sizeof(adapters[0])) ==
         WAFER_NCC_STATUS_OK);

  plan = request(2, 3, WAFER_NCC_SCHEDULE_WINDOW);
  plan.lanes[0] = lane(WAFER_NCC_ENGINE_CT, 0);
  plan.lanes[1] = plan.lanes[0];
  assert(wafer_ncc_probe_validate_plan(
             &plan, adapters, sizeof(adapters) / sizeof(adapters[0])) ==
         WAFER_NCC_STATUS_OK);

  plan.rounds = 4;
  assert(wafer_ncc_probe_validate_plan(
             &plan, adapters, sizeof(adapters) / sizeof(adapters[0])) ==
         WAFER_NCC_STATUS_UNSAFE_WINDOW);
}

int main(void) {
  test_dual_lane_boundary_order();
  test_three_lane_disjoint_window_and_serial();
  test_validation_bounds();
  test_wire_decode();
  test_failed_safety_drain_is_not_retried();
  test_depth_plus_one_is_tight_and_waited();
  test_full_depth_window_is_accepted_without_overflow();
  return 0;
}
