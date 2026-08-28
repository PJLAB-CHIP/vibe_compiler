#include "wafer_ncc_probe_plan.h"

#include <stddef.h>

static void wafer_ncc_probe_zero(volatile void *pointer, uint32_t bytes) {
  volatile uint8_t *output = (volatile uint8_t *)pointer;
  for (uint32_t index = 0; index < bytes; ++index)
    output[index] = 0;
}

static int wafer_ncc_probe_decode_u32(const volatile uint64_t *words,
                                      uint32_t index, uint32_t *value) {
  uint64_t encoded = words[index];
  if (encoded > UINT32_MAX)
    return 0;
  *value = (uint32_t)encoded;
  return 1;
}

static int wafer_ncc_probe_words_are_zero(const volatile uint64_t *words,
                                          uint32_t begin, uint32_t count) {
  for (uint32_t index = 0; index < count; ++index)
    if (words[begin + index] != 0)
      return 0;
  return 1;
}

static int wafer_ncc_probe_checked_mul(uint64_t lhs, uint64_t rhs,
                                       uint64_t *result) {
  if (rhs != 0 && lhs > UINT64_MAX / rhs)
    return 0;
  *result = lhs * rhs;
  return 1;
}

static int wafer_ncc_probe_checked_add(uint64_t lhs, uint64_t rhs,
                                       uint64_t *result) {
  if (lhs > UINT64_MAX - rhs)
    return 0;
  *result = lhs + rhs;
  return 1;
}

static int wafer_ncc_probe_dma_lane_is_valid(const WaferNccProbeLane *lane) {
  if ((lane->engine != WAFER_NCC_ENGINE_RDMA &&
       lane->engine != WAFER_NCC_ENGINE_WDMA) ||
      lane->issue_mode != WAFER_NCC_ISSUE_WRAPPER ||
      lane->element_format != WAFER_NCC_PROTOCOL_DMA_FORMAT_FP16 ||
      lane->layout_inner_bytes == 0 || lane->layout_iteration0 == 0 ||
      lane->layout_iteration1 == 0 || lane->layout_iteration2 == 0 ||
      (lane->layout_inner_bytes | lane->layout_stride0_bytes |
       lane->layout_stride1_bytes | lane->layout_stride2_bytes) &
          UINT32_C(1))
    return 0;

  uint64_t compact_bytes = lane->layout_inner_bytes;
  uint64_t row_bytes = 0;
  uint64_t plane_bytes = 0;
  uint64_t envelope_bytes = 0;
  uint64_t product = 0;
  if (!wafer_ncc_probe_checked_mul(compact_bytes, lane->layout_iteration0,
                                   &compact_bytes) ||
      !wafer_ncc_probe_checked_mul(compact_bytes, lane->layout_iteration1,
                                   &compact_bytes) ||
      !wafer_ncc_probe_checked_mul(compact_bytes, lane->layout_iteration2,
                                   &compact_bytes) ||
      !wafer_ncc_probe_checked_mul(lane->layout_iteration0 - 1U,
                                   lane->layout_stride0_bytes, &product) ||
      !wafer_ncc_probe_checked_add(product, lane->layout_inner_bytes,
                                   &row_bytes) ||
      !wafer_ncc_probe_checked_mul(lane->layout_iteration1 - 1U,
                                   lane->layout_stride1_bytes, &product) ||
      !wafer_ncc_probe_checked_add(product, row_bytes, &plane_bytes) ||
      !wafer_ncc_probe_checked_mul(lane->layout_iteration2 - 1U,
                                   lane->layout_stride2_bytes, &product) ||
      !wafer_ncc_probe_checked_add(product, plane_bytes, &envelope_bytes))
    return 0;
  return compact_bytes == lane->transfer_bytes &&
         (lane->layout_iteration0 == 1 ||
          lane->layout_stride0_bytes >= lane->layout_inner_bytes) &&
         (lane->layout_iteration1 == 1 ||
          lane->layout_stride1_bytes >= row_bytes) &&
         (lane->layout_iteration2 == 1 ||
          lane->layout_stride2_bytes >= plane_bytes) &&
         envelope_bytes <= WAFER_NCC_PROTOCOL_MAX_DMA_ENVELOPE_BYTES;
}

static int wafer_ncc_probe_dma_layout_equal(const WaferNccProbeLane *lhs,
                                            const WaferNccProbeLane *rhs) {
  return lhs->layout_kind == WAFER_NCC_LAYOUT_DMA_STRIDED &&
         rhs->layout_kind == WAFER_NCC_LAYOUT_DMA_STRIDED &&
         lhs->transfer_bytes == rhs->transfer_bytes &&
         lhs->element_format == rhs->element_format &&
         lhs->layout_inner_bytes == rhs->layout_inner_bytes &&
         lhs->layout_stride0_bytes == rhs->layout_stride0_bytes &&
         lhs->layout_stride1_bytes == rhs->layout_stride1_bytes &&
         lhs->layout_stride2_bytes == rhs->layout_stride2_bytes &&
         lhs->layout_iteration0 == rhs->layout_iteration0 &&
         lhs->layout_iteration1 == rhs->layout_iteration1 &&
         lhs->layout_iteration2 == rhs->layout_iteration2;
}

static int
wafer_ncc_probe_is_serial_dma_roundtrip(const WaferNccProbeRequest *request) {
  return request->lane_count == 2 && request->rounds == 1 &&
         request->effect_relation == WAFER_NCC_EFFECT_RAW &&
         request->range_relation == WAFER_NCC_RANGE_EXACT &&
         request->schedule == WAFER_NCC_SCHEDULE_SERIAL &&
         request->wait_kind == WAFER_NCC_WAIT_BY_WORKER &&
         request->wait_worker_mask == 1 &&
         request->first_operand == WAFER_NCC_OPERAND_WRITE &&
         request->second_operand == WAFER_NCC_OPERAND_READ0 &&
         request->lanes[0].engine == WAFER_NCC_ENGINE_RDMA &&
         request->lanes[1].engine == WAFER_NCC_ENGINE_WDMA &&
         request->lanes[0].worker == 0 && request->lanes[1].worker == 0 &&
         wafer_ncc_probe_dma_layout_equal(&request->lanes[0],
                                          &request->lanes[1]);
}

int wafer_ncc_probe_is_strided_dependency(const WaferNccProbeRequest *request) {
  if (request->lane_count != 2 || request->rounds != 1 ||
      request->schedule > WAFER_NCC_SCHEDULE_SERIAL ||
      request->wait_kind != WAFER_NCC_WAIT_BY_WORKER ||
      request->wait_worker_mask != 1 || request->flags != 0 ||
      request->issue_limit != 0 || request->lanes[0].worker != 0 ||
      request->lanes[1].worker != 0 ||
      request->lanes[0].issue_mode != WAFER_NCC_ISSUE_WRAPPER ||
      request->lanes[1].issue_mode != WAFER_NCC_ISSUE_WRAPPER ||
      request->lanes[0].element_format != WAFER_NCC_PROTOCOL_DMA_FORMAT_FP16 ||
      request->lanes[1].element_format != WAFER_NCC_PROTOCOL_DMA_FORMAT_FP16 ||
      !wafer_ncc_probe_dma_layout_equal(&request->lanes[0], &request->lanes[1]))
    return 0;

  uint32_t first = request->lanes[0].engine;
  uint32_t second = request->lanes[1].engine;
  switch (request->effect_relation) {
  case WAFER_NCC_EFFECT_RAW:
    return request->range_relation == WAFER_NCC_RANGE_STRIDED_ENVELOPE &&
           first == WAFER_NCC_ENGINE_RDMA && second == WAFER_NCC_ENGINE_WDMA &&
           request->first_operand == WAFER_NCC_OPERAND_WRITE &&
           request->second_operand == WAFER_NCC_OPERAND_READ0;
  case WAFER_NCC_EFFECT_WAR:
    return request->range_relation == WAFER_NCC_RANGE_STRIDED_ENVELOPE &&
           first == WAFER_NCC_ENGINE_WDMA && second == WAFER_NCC_ENGINE_RDMA &&
           request->first_operand == WAFER_NCC_OPERAND_READ0 &&
           request->second_operand == WAFER_NCC_OPERAND_WRITE;
  case WAFER_NCC_EFFECT_WAW: {
    uint64_t chunks = (uint64_t)request->lanes[0].layout_iteration0 *
                      request->lanes[0].layout_iteration1 *
                      request->lanes[0].layout_iteration2;
    return request->range_relation >= WAFER_NCC_RANGE_EXACT &&
           request->range_relation <= WAFER_NCC_RANGE_ADJACENT &&
           (request->range_relation != WAFER_NCC_RANGE_PARTIAL || chunks > 1) &&
           first == WAFER_NCC_ENGINE_RDMA && second == WAFER_NCC_ENGINE_RDMA &&
           request->first_operand == WAFER_NCC_OPERAND_WRITE &&
           request->second_operand == WAFER_NCC_OPERAND_WRITE;
  }
  case WAFER_NCC_EFFECT_RAR:
    return request->range_relation == WAFER_NCC_RANGE_STRIDED_ENVELOPE &&
           first == WAFER_NCC_ENGINE_WDMA && second == WAFER_NCC_ENGINE_WDMA &&
           request->first_operand == WAFER_NCC_OPERAND_READ0 &&
           request->second_operand == WAFER_NCC_OPERAND_READ0;
  default:
    return 0;
  }
}

static int wafer_ncc_probe_is_constructor_observation(
    const WaferNccProbeRequest *request) {
  const WaferNccProbeLane *lane = &request->lanes[0];
  return request->flags == WAFER_NCC_REQUEST_CONSTRUCTOR_OBSERVATION &&
         request->lane_count == 1 && request->rounds == 1 &&
         request->effect_relation == WAFER_NCC_EFFECT_NONE &&
         request->range_relation == WAFER_NCC_RANGE_DISJOINT &&
         request->schedule == WAFER_NCC_SCHEDULE_SERIAL &&
         request->wait_kind == WAFER_NCC_WAIT_BY_WORKER &&
         request->wait_worker_mask == 1 &&
         request->first_operand == WAFER_NCC_OPERAND_AUTO &&
         request->second_operand == WAFER_NCC_OPERAND_AUTO &&
         request->issue_limit == 0 && lane->engine == WAFER_NCC_ENGINE_CT &&
         lane->worker == 0 && lane->issue_mode == WAFER_NCC_ISSUE_RAW;
}

static int wafer_ncc_probe_is_ordered_producer_consumer(
    const WaferNccProbeRequest *request) {
  if (request->flags != WAFER_NCC_REQUEST_ORDERED_PRODUCER_CONSUMER ||
      request->lane_count != 2 || request->rounds != 1 ||
      request->effect_relation != WAFER_NCC_EFFECT_RAW ||
      request->range_relation != WAFER_NCC_RANGE_EXACT ||
      request->schedule != WAFER_NCC_SCHEDULE_SERIAL ||
      request->wait_kind != WAFER_NCC_WAIT_BY_WORKER ||
      request->first_operand != WAFER_NCC_OPERAND_WRITE ||
      request->second_operand != WAFER_NCC_OPERAND_READ0 ||
      request->issue_limit != 0)
    return 0;
  const WaferNccProbeLane *first = &request->lanes[0];
  const WaferNccProbeLane *second = &request->lanes[1];
  uint32_t participants =
      (UINT32_C(1) << first->worker) | (UINT32_C(1) << second->worker);
  if (request->wait_worker_mask != participants)
    return 0;
  return (first->engine == WAFER_NCC_ENGINE_RDMA &&
          second->engine == WAFER_NCC_ENGINE_CT) ||
         (first->engine == WAFER_NCC_ENGINE_CT &&
          second->engine == WAFER_NCC_ENGINE_WDMA) ||
         (first->engine == WAFER_NCC_ENGINE_NE &&
          second->engine == WAFER_NCC_ENGINE_WDMA) ||
         (first->engine == WAFER_NCC_ENGINE_TDMA &&
          second->engine == WAFER_NCC_ENGINE_CT) ||
         (first->engine == WAFER_NCC_ENGINE_TDMA &&
          second->engine == WAFER_NCC_ENGINE_NE);
}

static int wafer_ncc_probe_is_double_slot_observation(
    const WaferNccProbeRequest *request) {
  if (request->flags != WAFER_NCC_REQUEST_DOUBLE_SLOT_OBSERVATION ||
      request->lane_count != 3 || request->rounds == 0 ||
      request->rounds > WAFER_NCC_PROTOCOL_MAX_THREE_LANE_ROUNDS ||
      request->effect_relation != WAFER_NCC_EFFECT_NONE ||
      request->range_relation != WAFER_NCC_RANGE_DISJOINT ||
      request->schedule > WAFER_NCC_SCHEDULE_SERIAL ||
      request->wait_kind != WAFER_NCC_WAIT_BY_WORKER ||
      request->wait_worker_mask != 1 ||
      request->first_operand != WAFER_NCC_OPERAND_AUTO ||
      request->second_operand != WAFER_NCC_OPERAND_AUTO ||
      request->issue_limit != 0 ||
      request->lanes[0].engine != WAFER_NCC_ENGINE_RDMA ||
      request->lanes[1].engine != WAFER_NCC_ENGINE_CT ||
      request->lanes[2].engine != WAFER_NCC_ENGINE_WDMA)
    return 0;
  uint32_t bytes = request->lanes[0].transfer_bytes;
  for (uint32_t lane = 0; lane < 3; ++lane)
    if (request->lanes[lane].worker != 0 ||
        request->lanes[lane].transfer_bytes != bytes ||
        request->lanes[lane].element_format !=
            WAFER_NCC_PROTOCOL_DMA_FORMAT_FP16 ||
        request->lanes[lane].layout_kind != WAFER_NCC_LAYOUT_CONTIGUOUS)
      return 0;
  return 1;
}

static int
wafer_ncc_probe_is_bounded_pair_window(const WaferNccProbeRequest *request) {
  if (request->flags != WAFER_NCC_REQUEST_BOUNDED_PAIR_WINDOW ||
      request->lane_count != 2 ||
      request->rounds != WAFER_NCC_PROTOCOL_MAX_ROUNDS ||
      request->effect_relation != WAFER_NCC_EFFECT_NONE ||
      request->range_relation != WAFER_NCC_RANGE_DISJOINT ||
      request->schedule != WAFER_NCC_SCHEDULE_WINDOW ||
      request->wait_kind != WAFER_NCC_WAIT_BY_WORKER ||
      request->first_operand != WAFER_NCC_OPERAND_AUTO ||
      request->second_operand != WAFER_NCC_OPERAND_AUTO ||
      request->issue_limit != 0 ||
      request->lanes[0].engine == request->lanes[1].engine)
    return 0;
  uint32_t participants = (UINT32_C(1) << request->lanes[0].worker) |
                          (UINT32_C(1) << request->lanes[1].worker);
  return request->wait_worker_mask == participants;
}

static int wafer_ncc_probe_is_mapped_spm_kcore_write_observation(
    const WaferNccProbeRequest *request) {
  uint64_t allowed_flags = WAFER_NCC_REQUEST_MAPPED_SPM_KCORE_WRITE |
                           WAFER_NCC_REQUEST_PREISSUE_LOCAL_WAIT;
  const WaferNccProbeLane *consumer = &request->lanes[0];
  return (request->flags == WAFER_NCC_REQUEST_MAPPED_SPM_KCORE_WRITE ||
          request->flags == allowed_flags) &&
         request->lane_count == 1 && request->rounds == 1 &&
         request->effect_relation == WAFER_NCC_EFFECT_NONE &&
         request->range_relation == WAFER_NCC_RANGE_DISJOINT &&
         request->schedule == WAFER_NCC_SCHEDULE_WINDOW &&
         request->wait_kind == WAFER_NCC_WAIT_BY_WORKER &&
         request->wait_worker_mask == 1 &&
         request->first_operand == WAFER_NCC_OPERAND_AUTO &&
         request->second_operand == WAFER_NCC_OPERAND_AUTO &&
         request->issue_limit == 0 && consumer->engine == WAFER_NCC_ENGINE_CT &&
         consumer->worker == 0 && consumer->issue_mode == WAFER_NCC_ISSUE_RAW &&
         consumer->element_format == WAFER_NCC_PROTOCOL_DMA_FORMAT_FP16 &&
         consumer->layout_kind == WAFER_NCC_LAYOUT_CONTIGUOUS &&
         consumer->transfer_bytes % sizeof(uint16_t) == 0;
}

static const WaferNccProbeEngineAdapter *
wafer_ncc_probe_find_adapter(const WaferNccProbeEngineAdapter *adapters,
                             uint32_t adapter_count, uint32_t engine) {
  for (uint32_t index = 0; index < adapter_count; ++index)
    if (adapters[index].engine == engine)
      return &adapters[index];
  return NULL;
}

static uint32_t
wafer_ncc_probe_participant_mask(const WaferNccProbeRequest *request) {
  uint32_t mask = 0;
  for (uint32_t lane = 0; lane < request->lane_count; ++lane)
    mask |= UINT32_C(1) << request->lanes[lane].worker;
  return mask;
}

static uint32_t wafer_ncc_probe_operand_effect(uint32_t operand) {
  switch (operand) {
  case WAFER_NCC_OPERAND_READ0:
    return WAFER_NCC_MEMORY_READ0;
  case WAFER_NCC_OPERAND_READ1:
    return WAFER_NCC_MEMORY_READ1;
  case WAFER_NCC_OPERAND_WRITE:
    return WAFER_NCC_MEMORY_WRITE;
  default:
    return 0;
  }
}

static uint64_t wafer_ncc_probe_issue_tag(uint64_t seed, uint32_t slot) {
  /*
   * Slot occupies the low byte, so all issues in one bounded plan are unique
   * even when the high bits of the caller seed are identical.
   */
  return ((seed * UINT64_C(0x9e3779b97f4a7c15)) &
          UINT64_C(0xffffffffffffff00)) |
         (uint64_t)(slot + 1U);
}

static uint32_t
wafer_ncc_probe_initialize_issues(const WaferNccProbeRequest *request,
                                  WaferNccProbeIssue *issues) {
  uint32_t count = 0;
  for (uint32_t lane = 0; lane < request->lane_count; ++lane) {
    for (uint32_t round = 0; round < request->rounds; ++round) {
      WaferNccProbeIssue *issue = &issues[count];
      issue->ordinal = count;
      issue->lane = lane;
      issue->round = round;
      issue->slot = wafer_ncc_protocol_slot(lane, round, request->rounds);
      issue->engine = request->lanes[lane].engine;
      issue->worker = request->lanes[lane].worker;
      issue->tag = wafer_ncc_probe_issue_tag(request->seed, issue->slot);
      issue->lane_spec = &request->lanes[lane];
      ++count;
      if (request->issue_limit != 0 && count == request->issue_limit)
        return count;
    }
  }
  return count;
}

static void
wafer_ncc_probe_write_issue_identity(volatile uint64_t *record,
                                     const WaferNccProbeIssue *issue) {
  uint32_t base =
      wafer_ncc_protocol_issue_word(issue->ordinal, WAFER_NCC_ISSUE_ORDINAL);
  record[base + WAFER_NCC_ISSUE_ORDINAL] = issue->ordinal;
  record[base + WAFER_NCC_ISSUE_LANE] = issue->lane;
  record[base + WAFER_NCC_ISSUE_ROUND] = issue->round;
  record[base + WAFER_NCC_ISSUE_SLOT] = issue->slot;
  record[base + WAFER_NCC_ISSUE_ENGINE] = issue->engine;
  record[base + WAFER_NCC_ISSUE_WORKER] = issue->worker;
  record[base + WAFER_NCC_ISSUE_TAG] = issue->tag;
}

static void
wafer_ncc_probe_write_observation(volatile uint64_t *record,
                                  const WaferNccProbeIssue *issue,
                                  const WaferNccProbeObservation *observation) {
  uint32_t base =
      wafer_ncc_protocol_issue_word(issue->ordinal, WAFER_NCC_ISSUE_ORDINAL);
  record[base + WAFER_NCC_ISSUE_EXECUTE_RC] = observation->execute_rc;
  record[base + WAFER_NCC_ISSUE_INTER_TYPE] = observation->inter_type;
  record[base + WAFER_NCC_ISSUE_READ0_BEGIN] = observation->read0_begin;
  record[base + WAFER_NCC_ISSUE_READ0_END] = observation->read0_end;
  record[base + WAFER_NCC_ISSUE_READ1_BEGIN] = observation->read1_begin;
  record[base + WAFER_NCC_ISSUE_READ1_END] = observation->read1_end;
  record[base + WAFER_NCC_ISSUE_WRITE_BEGIN] = observation->write_begin;
  record[base + WAFER_NCC_ISSUE_WRITE_END] = observation->write_end;
  record[base + WAFER_NCC_ISSUE_FLAGS] = observation->flags;
}

static int wafer_ncc_probe_observe_deferred(
    const WaferNccProbeRequest *request,
    const WaferNccProbeEngineAdapter *adapters, uint32_t adapter_count,
    void *context, volatile uint64_t *record, WaferNccProbeIssue *issues,
    uint32_t issue_count) {
  for (uint32_t index = 0; index < issue_count; ++index) {
    WaferNccProbeIssue *issue = &issues[index];
    const WaferNccProbeEngineAdapter *adapter =
        wafer_ncc_probe_find_adapter(adapters, adapter_count, issue->engine);
    if (adapter == NULL)
      return 1;
    uint32_t base =
        wafer_ncc_protocol_issue_word(issue->ordinal, WAFER_NCC_ISSUE_ORDINAL);
    WaferNccProbeObservation observation;
    wafer_ncc_probe_zero(&observation, sizeof(observation));
    observation.execute_rc = record[base + WAFER_NCC_ISSUE_EXECUTE_RC];
    if (adapter->observe(context, request, issue, &observation) != 0)
      return 1;
    wafer_ncc_probe_write_observation(record, issue, &observation);
  }
  return 0;
}

static void wafer_ncc_probe_release_prepared(
    const WaferNccProbeEngineAdapter *adapters, uint32_t adapter_count,
    void *context, WaferNccProbeIssue *issues, uint32_t prepared_count) {
  while (prepared_count != 0) {
    WaferNccProbeIssue *issue = &issues[--prepared_count];
    const WaferNccProbeEngineAdapter *adapter =
        wafer_ncc_probe_find_adapter(adapters, adapter_count, issue->engine);
    if (adapter != NULL && adapter->release != NULL)
      adapter->release(context, issue);
  }
}

static uint32_t wafer_ncc_probe_finish_after_failure(
    uint32_t status, const WaferNccProbeRequest *request,
    const WaferNccProbeEngineAdapter *adapters, uint32_t adapter_count,
    const WaferNccProbeExecutionHooks *hooks, void *context,
    volatile uint64_t *record, WaferNccProbeIssue *issues,
    uint32_t prepared_count, int issued_any) {
  if (issued_any && hooks->safety_drain != NULL) {
    uint32_t mask = wafer_ncc_probe_participant_mask(request);
    if (hooks->safety_drain(context, mask) == 0)
      record[WAFER_NCC_REC_FLAGS] |= WAFER_NCC_RECORD_SAFETY_DRAIN_DONE;
  }
  wafer_ncc_probe_release_prepared(adapters, adapter_count, context, issues,
                                   prepared_count);
  record[WAFER_NCC_REC_STATUS] = status;
  return status;
}

uint32_t wafer_ncc_probe_decode_request(const volatile uint64_t *request_words,
                                        WaferNccProbeRequest *request) {
  if (request_words == NULL || request == NULL)
    return WAFER_NCC_STATUS_BAD_REQUEST;
  uint64_t words = request_words[WAFER_NCC_REQ_WORD_COUNT];
  if (request_words[WAFER_NCC_REQ_MAGIC] != WAFER_NCC_PROTOCOL_REQUEST_MAGIC ||
      words != WAFER_NCC_PROTOCOL_REQUEST_WORDS)
    return WAFER_NCC_STATUS_BAD_REQUEST;

  wafer_ncc_probe_zero(request, sizeof(*request));
  if (!wafer_ncc_probe_decode_u32(request_words, WAFER_NCC_REQ_COMMAND,
                                  &request->command) ||
      !wafer_ncc_probe_decode_u32(request_words, WAFER_NCC_REQ_LANE_COUNT,
                                  &request->lane_count) ||
      !wafer_ncc_probe_decode_u32(request_words, WAFER_NCC_REQ_ROUNDS,
                                  &request->rounds) ||
      !wafer_ncc_probe_decode_u32(request_words, WAFER_NCC_REQ_EFFECT_RELATION,
                                  &request->effect_relation) ||
      !wafer_ncc_probe_decode_u32(request_words, WAFER_NCC_REQ_RANGE_RELATION,
                                  &request->range_relation) ||
      !wafer_ncc_probe_decode_u32(request_words, WAFER_NCC_REQ_SCHEDULE,
                                  &request->schedule) ||
      !wafer_ncc_probe_decode_u32(request_words, WAFER_NCC_REQ_WAIT_KIND,
                                  &request->wait_kind) ||
      !wafer_ncc_probe_decode_u32(request_words, WAFER_NCC_REQ_WAIT_WORKER_MASK,
                                  &request->wait_worker_mask) ||
      !wafer_ncc_probe_decode_u32(request_words, WAFER_NCC_REQ_FIRST_OPERAND,
                                  &request->first_operand) ||
      !wafer_ncc_probe_decode_u32(request_words, WAFER_NCC_REQ_SECOND_OPERAND,
                                  &request->second_operand) ||
      !wafer_ncc_probe_decode_u32(request_words, WAFER_NCC_REQ_ISSUE_LIMIT,
                                  &request->issue_limit) ||
      !wafer_ncc_probe_words_are_zero(
          request_words, WAFER_NCC_PROTOCOL_REQUEST_RESERVED_BASE,
          WAFER_NCC_PROTOCOL_REQUEST_RESERVED_WORDS))
    return WAFER_NCC_STATUS_BAD_REQUEST;
  request->seed = request_words[WAFER_NCC_REQ_SEED];
  request->sample = request_words[WAFER_NCC_REQ_SAMPLE];
  request->flags = request_words[WAFER_NCC_REQ_FLAGS];
  for (uint32_t lane = 0; lane < WAFER_NCC_PROTOCOL_MAX_LANES; ++lane) {
    WaferNccProbeLane *decoded = &request->lanes[lane];
    uint32_t base = wafer_ncc_protocol_lane_word(lane, 0);
    if (!wafer_ncc_probe_decode_u32(request_words, base + WAFER_NCC_LANE_ENGINE,
                                    &decoded->engine) ||
        !wafer_ncc_probe_decode_u32(request_words, base + WAFER_NCC_LANE_WORKER,
                                    &decoded->worker) ||
        !wafer_ncc_probe_decode_u32(request_words,
                                    base + WAFER_NCC_LANE_ISSUE_MODE,
                                    &decoded->issue_mode) ||
        !wafer_ncc_probe_decode_u32(request_words,
                                    base + WAFER_NCC_LANE_TRANSFER_BYTES,
                                    &decoded->transfer_bytes) ||
        !wafer_ncc_probe_decode_u32(request_words,
                                    base + WAFER_NCC_LANE_ELEMENT_FORMAT,
                                    &decoded->element_format) ||
        !wafer_ncc_probe_decode_u32(request_words,
                                    base + WAFER_NCC_LANE_LAYOUT_KIND,
                                    &decoded->layout_kind) ||
        !wafer_ncc_probe_decode_u32(request_words,
                                    base + WAFER_NCC_LANE_LAYOUT_INNER_BYTES,
                                    &decoded->layout_inner_bytes) ||
        !wafer_ncc_probe_decode_u32(request_words,
                                    base + WAFER_NCC_LANE_LAYOUT_STRIDE0_BYTES,
                                    &decoded->layout_stride0_bytes) ||
        !wafer_ncc_probe_decode_u32(request_words,
                                    base + WAFER_NCC_LANE_LAYOUT_STRIDE1_BYTES,
                                    &decoded->layout_stride1_bytes) ||
        !wafer_ncc_probe_decode_u32(request_words,
                                    base + WAFER_NCC_LANE_LAYOUT_STRIDE2_BYTES,
                                    &decoded->layout_stride2_bytes) ||
        !wafer_ncc_probe_decode_u32(request_words,
                                    base + WAFER_NCC_LANE_LAYOUT_ITERATION0,
                                    &decoded->layout_iteration0) ||
        !wafer_ncc_probe_decode_u32(request_words,
                                    base + WAFER_NCC_LANE_LAYOUT_ITERATION1,
                                    &decoded->layout_iteration1) ||
        !wafer_ncc_probe_decode_u32(request_words,
                                    base + WAFER_NCC_LANE_LAYOUT_ITERATION2,
                                    &decoded->layout_iteration2) ||
        !wafer_ncc_probe_decode_u32(request_words, base + WAFER_NCC_LANE_FLAGS,
                                    &decoded->flags))
      return WAFER_NCC_STATUS_BAD_REQUEST;
  }
  if (request->lane_count <= WAFER_NCC_PROTOCOL_MAX_LANES &&
      !wafer_ncc_probe_words_are_zero(
          request_words,
          WAFER_NCC_PROTOCOL_LANE_BASE +
              request->lane_count * WAFER_NCC_PROTOCOL_LANE_STRIDE,
          (WAFER_NCC_PROTOCOL_MAX_LANES - request->lane_count) *
              WAFER_NCC_PROTOCOL_LANE_STRIDE))
    return WAFER_NCC_STATUS_BAD_REQUEST;
  return WAFER_NCC_STATUS_OK;
}

uint32_t
wafer_ncc_probe_validate_plan(const WaferNccProbeRequest *request,
                              const WaferNccProbeEngineAdapter *adapters,
                              uint32_t adapter_count) {
  if (request == NULL || (adapter_count != 0 && adapters == NULL))
    return WAFER_NCC_STATUS_BAD_REQUEST;
  if (request->command > WAFER_NCC_COMMAND_EXECUTE ||
      (request->flags != 0 &&
       request->flags != WAFER_NCC_REQUEST_TIGHT_DEPTH_PLUS_ONE &&
       request->flags != WAFER_NCC_REQUEST_CONSTRUCTOR_OBSERVATION &&
       request->flags != WAFER_NCC_REQUEST_ORDERED_PRODUCER_CONSUMER &&
       request->flags != WAFER_NCC_REQUEST_DOUBLE_SLOT_OBSERVATION &&
       request->flags != WAFER_NCC_REQUEST_MAPPED_SPM_KCORE_WRITE &&
       request->flags != WAFER_NCC_REQUEST_TIGHT_KCORE_BOUNDARY &&
       request->flags != WAFER_NCC_REQUEST_TIGHT_QUEUE_SATURATION &&
       request->flags != WAFER_NCC_REQUEST_TIGHT_WORKER_SCOPE &&
       request->flags != WAFER_NCC_REQUEST_BOUNDED_PAIR_WINDOW &&
       request->flags != (WAFER_NCC_REQUEST_MAPPED_SPM_KCORE_WRITE |
                          WAFER_NCC_REQUEST_PREISSUE_LOCAL_WAIT)))
    return WAFER_NCC_STATUS_BAD_REQUEST;
  if (request->command == WAFER_NCC_COMMAND_QUALIFY) {
    if (request->lane_count != 0 || request->rounds != 0 ||
        request->effect_relation != WAFER_NCC_EFFECT_NONE ||
        request->range_relation != WAFER_NCC_RANGE_DISJOINT ||
        request->schedule != WAFER_NCC_SCHEDULE_WINDOW ||
        request->wait_kind != WAFER_NCC_WAIT_NONE ||
        request->wait_worker_mask != 0 ||
        request->first_operand != WAFER_NCC_OPERAND_AUTO ||
        request->second_operand != WAFER_NCC_OPERAND_AUTO ||
        request->issue_limit != 0 || request->flags != 0)
      return WAFER_NCC_STATUS_BAD_REQUEST;
    return WAFER_NCC_STATUS_OK;
  }

  if (request->lane_count == 0 ||
      request->lane_count > WAFER_NCC_PROTOCOL_MAX_LANES ||
      request->rounds == 0 || request->rounds > WAFER_NCC_PROTOCOL_MAX_ROUNDS ||
      (request->lane_count == 3 &&
       request->rounds > WAFER_NCC_PROTOCOL_MAX_THREE_LANE_ROUNDS) ||
      request->effect_relation > WAFER_NCC_EFFECT_RAR ||
      request->range_relation > WAFER_NCC_RANGE_STRIDED_ENVELOPE ||
      request->schedule > WAFER_NCC_SCHEDULE_SERIAL ||
      request->wait_kind > WAFER_NCC_WAIT_LOCAL_FENCE ||
      (request->wait_worker_mask &
       ~((UINT32_C(1) << WAFER_NCC_PROTOCOL_WORKERS) - 1U)) != 0)
    return WAFER_NCC_STATUS_BAD_REQUEST;

  if (request->lane_count == 1 &&
      (request->effect_relation != WAFER_NCC_EFFECT_NONE ||
       request->range_relation != WAFER_NCC_RANGE_DISJOINT))
    return WAFER_NCC_STATUS_UNSUPPORTED_COMBINATION;
  if (request->effect_relation == WAFER_NCC_EFFECT_NONE) {
    if (request->first_operand != WAFER_NCC_OPERAND_AUTO ||
        request->second_operand != WAFER_NCC_OPERAND_AUTO)
      return WAFER_NCC_STATUS_BAD_REQUEST;
  } else if (request->lane_count != 2 ||
             wafer_ncc_probe_operand_effect(request->first_operand) == 0 ||
             wafer_ncc_probe_operand_effect(request->second_operand) == 0) {
    return WAFER_NCC_STATUS_UNSUPPORTED_COMBINATION;
  }
  if (request->lane_count == 3 &&
      (request->effect_relation != WAFER_NCC_EFFECT_NONE ||
       request->range_relation != WAFER_NCC_RANGE_DISJOINT))
    return WAFER_NCC_STATUS_UNSUPPORTED_COMBINATION;

  uint32_t participant_mask = 0;
  int has_strided_lane = 0;
  int has_dma_strided_lane = 0;
  for (uint32_t lane = 0; lane < request->lane_count; ++lane) {
    const WaferNccProbeLane *lane_spec = &request->lanes[lane];
    if (lane_spec->engine > WAFER_NCC_ENGINE_TDMA ||
        lane_spec->worker >= WAFER_NCC_PROTOCOL_WORKERS ||
        lane_spec->issue_mode > WAFER_NCC_ISSUE_WRAPPER ||
        lane_spec->transfer_bytes == 0 ||
        lane_spec->layout_kind > WAFER_NCC_LAYOUT_DMA_STRIDED ||
        lane_spec->flags != 0)
      return WAFER_NCC_STATUS_BAD_REQUEST;
    if ((lane_spec->engine != WAFER_NCC_ENGINE_TDMA &&
         lane_spec->element_format != WAFER_NCC_PROTOCOL_DMA_FORMAT_FP16) ||
        (lane_spec->element_format == WAFER_NCC_PROTOCOL_DMA_FORMAT_FP16 &&
         lane_spec->transfer_bytes % sizeof(uint16_t) != 0))
      return WAFER_NCC_STATUS_UNSUPPORTED_COMBINATION;
    if (lane_spec->layout_kind != WAFER_NCC_LAYOUT_DMA_STRIDED &&
        (lane_spec->layout_stride0_bytes != 0 ||
         lane_spec->layout_stride1_bytes != 0 ||
         lane_spec->layout_stride2_bytes != 0 ||
         lane_spec->layout_iteration0 != 0 ||
         lane_spec->layout_iteration1 != 0 ||
         lane_spec->layout_iteration2 != 0))
      return WAFER_NCC_STATUS_BAD_REQUEST;
    if (lane_spec->layout_kind == WAFER_NCC_LAYOUT_CONTIGUOUS &&
        lane_spec->layout_inner_bytes != 0)
      return WAFER_NCC_STATUS_BAD_REQUEST;
    if (lane_spec->layout_kind == WAFER_NCC_LAYOUT_INNER_STRIDED) {
      has_strided_lane = 1;
      if (lane_spec->layout_inner_bytes == 0 ||
          lane_spec->layout_inner_bytes > lane_spec->transfer_bytes ||
          lane_spec->transfer_bytes % lane_spec->layout_inner_bytes != 0)
        return WAFER_NCC_STATUS_BAD_REQUEST;
    } else if (lane_spec->layout_kind == WAFER_NCC_LAYOUT_DMA_STRIDED) {
      has_strided_lane = 1;
      has_dma_strided_lane = 1;
      if (!wafer_ncc_probe_dma_lane_is_valid(lane_spec))
        return WAFER_NCC_STATUS_UNSUPPORTED_COMBINATION;
    }
    participant_mask |= UINT32_C(1) << lane_spec->worker;

    const WaferNccProbeEngineAdapter *adapter = wafer_ncc_probe_find_adapter(
        adapters, adapter_count, lane_spec->engine);
    if (adapter == NULL)
      return WAFER_NCC_STATUS_MISSING_ADAPTER;
    if (adapter->queue_depth == 0 || adapter->memory_effects == NULL ||
        adapter->seed == NULL || adapter->prepare == NULL ||
        adapter->issue == NULL || adapter->observe == NULL ||
        adapter->oracle == NULL || adapter->release == NULL ||
        (adapter->issue_mode_mask & (UINT32_C(1) << lane_spec->issue_mode)) ==
            0)
      return WAFER_NCC_STATUS_UNSUPPORTED_COMBINATION;
  }
  if (request->range_relation == WAFER_NCC_RANGE_STRIDED_ENVELOPE &&
      !has_strided_lane)
    return WAFER_NCC_STATUS_UNSUPPORTED_COMBINATION;
  if (has_dma_strided_lane &&
      !wafer_ncc_probe_is_serial_dma_roundtrip(request) &&
      !wafer_ncc_probe_is_strided_dependency(request))
    return WAFER_NCC_STATUS_UNSUPPORTED_COMBINATION;

  if (request->lane_count == 2 &&
      request->effect_relation != WAFER_NCC_EFFECT_NONE) {
    const WaferNccProbeEngineAdapter *first = wafer_ncc_probe_find_adapter(
        adapters, adapter_count, request->lanes[0].engine);
    const WaferNccProbeEngineAdapter *second = wafer_ncc_probe_find_adapter(
        adapters, adapter_count, request->lanes[1].engine);
    uint32_t first_effects = first->memory_effects(request, &request->lanes[0]);
    uint32_t second_effects =
        second->memory_effects(request, &request->lanes[1]);
    uint32_t first_selected =
        wafer_ncc_probe_operand_effect(request->first_operand);
    uint32_t second_selected =
        wafer_ncc_probe_operand_effect(request->second_operand);
    uint32_t read_mask = WAFER_NCC_MEMORY_READ0 | WAFER_NCC_MEMORY_READ1;
    int supported = 0;
    if ((first_effects & first_selected) == 0 ||
        (second_effects & second_selected) == 0)
      return WAFER_NCC_STATUS_UNSUPPORTED_COMBINATION;
    switch (request->effect_relation) {
    case WAFER_NCC_EFFECT_RAW:
      supported = first_selected == WAFER_NCC_MEMORY_WRITE &&
                  (second_selected & read_mask) != 0;
      break;
    case WAFER_NCC_EFFECT_WAR:
      supported = (first_selected & read_mask) != 0 &&
                  second_selected == WAFER_NCC_MEMORY_WRITE;
      break;
    case WAFER_NCC_EFFECT_WAW:
      supported = first_selected == WAFER_NCC_MEMORY_WRITE &&
                  second_selected == WAFER_NCC_MEMORY_WRITE;
      break;
    case WAFER_NCC_EFFECT_RAR:
      supported = (first_selected & read_mask) != 0 &&
                  (second_selected & read_mask) != 0;
      break;
    default:
      break;
    }
    if (!supported)
      return WAFER_NCC_STATUS_UNSUPPORTED_COMBINATION;
  }

  for (uint32_t first = 0; first < adapter_count; ++first)
    for (uint32_t second = first + 1; second < adapter_count; ++second)
      if (adapters[first].engine == adapters[second].engine)
        return WAFER_NCC_STATUS_BAD_REQUEST;

  if (request->wait_kind == WAFER_NCC_WAIT_BY_WORKER) {
    if (request->wait_worker_mask == 0 ||
        (request->wait_worker_mask & ~participant_mask) != 0)
      return WAFER_NCC_STATUS_BAD_REQUEST;
  } else if (request->wait_worker_mask != 0) {
    return WAFER_NCC_STATUS_BAD_REQUEST;
  }

  int tight_depth_plus_one =
      request->flags == WAFER_NCC_REQUEST_TIGHT_DEPTH_PLUS_ONE;
  int tight_kcore_boundary =
      request->flags == WAFER_NCC_REQUEST_TIGHT_KCORE_BOUNDARY;
  int tight_queue_saturation =
      request->flags == WAFER_NCC_REQUEST_TIGHT_QUEUE_SATURATION;
  int tight_worker_scope =
      request->flags == WAFER_NCC_REQUEST_TIGHT_WORKER_SCOPE;
  int bounded_pair_window =
      request->flags == WAFER_NCC_REQUEST_BOUNDED_PAIR_WINDOW;
  int special_queue_bound = tight_depth_plus_one || tight_kcore_boundary ||
                            tight_queue_saturation || bounded_pair_window;
  if (tight_depth_plus_one || tight_queue_saturation) {
    const WaferNccProbeLane *first = &request->lanes[0];
    const WaferNccProbeLane *second = &request->lanes[1];
    const WaferNccProbeEngineAdapter *adapter =
        wafer_ncc_probe_find_adapter(adapters, adapter_count, first->engine);
    int issue_limit_is_expected =
        adapter != NULL &&
        (request->issue_limit == adapter->queue_depth + 1U ||
         (tight_queue_saturation &&
          (request->issue_limit == adapter->queue_depth ||
           request->issue_limit + 1U == adapter->queue_depth)));
    if (request->lane_count != 2 ||
        request->effect_relation != WAFER_NCC_EFFECT_NONE ||
        request->range_relation != WAFER_NCC_RANGE_DISJOINT ||
        request->schedule != WAFER_NCC_SCHEDULE_WINDOW ||
        request->wait_kind != WAFER_NCC_WAIT_BY_WORKER ||
        request->wait_worker_mask != 1U || first->worker != 0 ||
        first->issue_mode != WAFER_NCC_ISSUE_RAW ||
        second->engine != first->engine || second->worker != first->worker ||
        second->issue_mode != first->issue_mode ||
        second->transfer_bytes != first->transfer_bytes ||
        second->element_format != first->element_format ||
        second->layout_kind != first->layout_kind ||
        second->layout_inner_bytes != first->layout_inner_bytes ||
        second->layout_stride0_bytes != first->layout_stride0_bytes ||
        second->layout_stride1_bytes != first->layout_stride1_bytes ||
        second->layout_stride2_bytes != first->layout_stride2_bytes ||
        second->layout_iteration0 != first->layout_iteration0 ||
        second->layout_iteration1 != first->layout_iteration1 ||
        second->layout_iteration2 != first->layout_iteration2 ||
        second->flags != first->flags || !issue_limit_is_expected ||
        request->issue_limit > request->lane_count * request->rounds ||
        request->issue_limit <= (request->lane_count - 1U) * request->rounds)
      return WAFER_NCC_STATUS_UNSUPPORTED_COMBINATION;
  } else if (tight_kcore_boundary) {
    const WaferNccProbeLane *producer = &request->lanes[0];
    const WaferNccProbeEngineAdapter *adapter =
        wafer_ncc_probe_find_adapter(adapters, adapter_count, producer->engine);
    if (request->lane_count != 1 ||
        request->effect_relation != WAFER_NCC_EFFECT_NONE ||
        request->range_relation != WAFER_NCC_RANGE_DISJOINT ||
        request->schedule != WAFER_NCC_SCHEDULE_WINDOW ||
        (request->wait_kind != WAFER_NCC_WAIT_NONE &&
         request->wait_kind != WAFER_NCC_WAIT_LOCAL_FENCE) ||
        request->wait_worker_mask != 0U || producer->worker != 0 ||
        request->first_operand != WAFER_NCC_OPERAND_AUTO ||
        request->second_operand != WAFER_NCC_OPERAND_AUTO ||
        producer->issue_mode != WAFER_NCC_ISSUE_RAW || adapter == NULL ||
        request->rounds > adapter->queue_depth || request->issue_limit != 0)
      return WAFER_NCC_STATUS_UNSUPPORTED_COMBINATION;
  } else if (tight_worker_scope) {
    if (request->lane_count != 3 ||
        request->effect_relation != WAFER_NCC_EFFECT_NONE ||
        request->range_relation != WAFER_NCC_RANGE_DISJOINT ||
        request->schedule != WAFER_NCC_SCHEDULE_WINDOW ||
        request->wait_kind == WAFER_NCC_WAIT_NONE ||
        request->issue_limit != 0 ||
        (participant_mask & (participant_mask - 1U)) == 0)
      return WAFER_NCC_STATUS_UNSUPPORTED_COMBINATION;
    for (uint32_t lane = 0; lane < request->lane_count; ++lane)
      if (request->lanes[lane].issue_mode != WAFER_NCC_ISSUE_RAW)
        return WAFER_NCC_STATUS_UNSUPPORTED_COMBINATION;
  } else if (bounded_pair_window) {
    if (!wafer_ncc_probe_is_bounded_pair_window(request))
      return WAFER_NCC_STATUS_UNSUPPORTED_COMBINATION;
  } else if (request->issue_limit != 0) {
    return WAFER_NCC_STATUS_BAD_REQUEST;
  }
  if (request->flags == WAFER_NCC_REQUEST_CONSTRUCTOR_OBSERVATION &&
      !wafer_ncc_probe_is_constructor_observation(request))
    return WAFER_NCC_STATUS_UNSUPPORTED_COMBINATION;
  if (request->flags == WAFER_NCC_REQUEST_ORDERED_PRODUCER_CONSUMER &&
      !wafer_ncc_probe_is_ordered_producer_consumer(request))
    return WAFER_NCC_STATUS_UNSUPPORTED_COMBINATION;
  if (request->flags == WAFER_NCC_REQUEST_DOUBLE_SLOT_OBSERVATION &&
      !wafer_ncc_probe_is_double_slot_observation(request))
    return WAFER_NCC_STATUS_UNSUPPORTED_COMBINATION;
  if ((request->flags & WAFER_NCC_REQUEST_MAPPED_SPM_KCORE_WRITE) != 0 &&
      !wafer_ncc_probe_is_mapped_spm_kcore_write_observation(request))
    return WAFER_NCC_STATUS_UNSUPPORTED_COMBINATION;

  if (request->schedule == WAFER_NCC_SCHEDULE_WINDOW && !special_queue_bound) {
    for (uint32_t lane = 0; lane < request->lane_count; ++lane) {
      const WaferNccProbeLane *current = &request->lanes[lane];
      uint32_t outstanding = request->rounds;
      for (uint32_t previous = 0; previous < lane; ++previous) {
        const WaferNccProbeLane *other = &request->lanes[previous];
        if (other->engine == current->engine &&
            other->worker == current->worker)
          outstanding += request->rounds;
      }
      const WaferNccProbeEngineAdapter *adapter = wafer_ncc_probe_find_adapter(
          adapters, adapter_count, current->engine);
      if (outstanding > adapter->queue_depth)
        return WAFER_NCC_STATUS_UNSAFE_WINDOW;
    }
  }
  return WAFER_NCC_STATUS_OK;
}

uint32_t
wafer_ncc_probe_execute_plan(const WaferNccProbeRequest *request,
                             const WaferNccProbeEngineAdapter *adapters,
                             uint32_t adapter_count,
                             const WaferNccProbeExecutionHooks *hooks,
                             void *context, volatile uint64_t *record) {
  if (record == NULL || hooks == NULL || hooks->snapshot == NULL ||
      hooks->seed_complete == NULL || hooks->serial_drain == NULL ||
      hooks->requested_wait == NULL || hooks->safety_drain == NULL ||
      hooks->read_cycle == NULL || hooks->read_worker_control == NULL)
    return WAFER_NCC_STATUS_BAD_REQUEST;

  wafer_ncc_probe_zero(record,
                       WAFER_NCC_PROTOCOL_RECORD_WORDS * sizeof(uint64_t));
  record[WAFER_NCC_REC_MAGIC] = WAFER_NCC_PROTOCOL_RECORD_MAGIC;
  record[WAFER_NCC_REC_WORD_COUNT] = WAFER_NCC_PROTOCOL_RECORD_WORDS;
  record[WAFER_NCC_REC_STATUS] = WAFER_NCC_STATUS_BAD_REQUEST;

  uint32_t status =
      wafer_ncc_probe_validate_plan(request, adapters, adapter_count);
  if (status != WAFER_NCC_STATUS_OK) {
    record[WAFER_NCC_REC_STATUS] = status;
    return status;
  }

  record[WAFER_NCC_REC_COMMAND] = request->command;
  record[WAFER_NCC_REC_LANE_COUNT] = request->lane_count;
  record[WAFER_NCC_REC_ROUNDS] = request->rounds;
  record[WAFER_NCC_REC_ISSUE_COUNT] =
      request->issue_limit != 0 ? request->issue_limit
                                : request->lane_count * request->rounds;
  record[WAFER_NCC_REC_EFFECT_RELATION] = request->effect_relation;
  record[WAFER_NCC_REC_RANGE_RELATION] = request->range_relation;
  record[WAFER_NCC_REC_SCHEDULE] = request->schedule;
  record[WAFER_NCC_REC_WAIT_KIND] = request->wait_kind;
  record[WAFER_NCC_REC_WAIT_WORKER_MASK] = request->wait_worker_mask;
  record[WAFER_NCC_REC_FIRST_OPERAND] = request->first_operand;
  record[WAFER_NCC_REC_SECOND_OPERAND] = request->second_operand;
  record[WAFER_NCC_REC_ISSUE_LIMIT] = request->issue_limit;
  record[WAFER_NCC_REC_SAFETY_WORKER_MASK] =
      wafer_ncc_probe_participant_mask(request);
  record[WAFER_NCC_REC_SEED] = request->seed;
  record[WAFER_NCC_REC_SAMPLE] = request->sample;

  WaferNccProbeIssue issues[WAFER_NCC_PROTOCOL_MAX_ISSUES];
  wafer_ncc_probe_zero(issues, sizeof(issues));
  uint32_t issue_count = wafer_ncc_probe_initialize_issues(request, issues);
  for (uint32_t index = 0; index < issue_count; ++index)
    wafer_ncc_probe_write_issue_identity(record, &issues[index]);

  uint32_t prepared_count = 0;
  for (uint32_t index = 0; index < issue_count; ++index) {
    WaferNccProbeIssue *issue = &issues[index];
    const WaferNccProbeEngineAdapter *adapter =
        wafer_ncc_probe_find_adapter(adapters, adapter_count, issue->engine);
    if (adapter->seed(context, request, issue) != 0)
      return wafer_ncc_probe_finish_after_failure(
          WAFER_NCC_STATUS_SEED_FAILED, request, adapters, adapter_count, hooks,
          context, record, issues, prepared_count, 0);
  }
  /*
   * Every seed hook may have written operands through a weak-order Kcore
   * mapping.  One explicit visibility boundary after the complete seed set
   * prevents the first NCC consumer from racing those stores.  Preparation,
   * PMU snapshots, and the measured issue window all begin afterward.
   */
  if (hooks->seed_complete(context) != 0)
    return wafer_ncc_probe_finish_after_failure(
        WAFER_NCC_STATUS_SEED_FAILED, request, adapters, adapter_count, hooks,
        context, record, issues, prepared_count, 0);
  for (uint32_t index = 0; index < issue_count; ++index) {
    WaferNccProbeIssue *issue = &issues[index];
    const WaferNccProbeEngineAdapter *adapter =
        wafer_ncc_probe_find_adapter(adapters, adapter_count, issue->engine);
    uint32_t base =
        wafer_ncc_protocol_issue_word(issue->ordinal, WAFER_NCC_ISSUE_ORDINAL);
    uint64_t preparation_flags = WAFER_NCC_ISSUE_PREPARE_ENTERED;
    if (adapter->prepare(context, request, issue, &preparation_flags) != 0) {
      record[base + WAFER_NCC_ISSUE_FLAGS] = preparation_flags;
      return wafer_ncc_probe_finish_after_failure(
          WAFER_NCC_STATUS_PREPARE_FAILED, request, adapters, adapter_count,
          hooks, context, record, issues, prepared_count, 0);
    }
    record[base + WAFER_NCC_ISSUE_FLAGS] =
        preparation_flags | WAFER_NCC_ISSUE_PREPARE_COMPLETED;
    ++prepared_count;
  }

  if (hooks->snapshot(context, WAFER_NCC_SNAPSHOT_BEFORE, record) != 0)
    return wafer_ncc_probe_finish_after_failure(
        WAFER_NCC_STATUS_OBSERVATION_FAILED, request, adapters, adapter_count,
        hooks, context, record, issues, prepared_count, 0);
  record[WAFER_NCC_REC_FLAGS] |= WAFER_NCC_RECORD_BEFORE_CAPTURED;

  int tight_depth_plus_one =
      request->flags == WAFER_NCC_REQUEST_TIGHT_DEPTH_PLUS_ONE;
  int tight_kcore_boundary =
      request->flags == WAFER_NCC_REQUEST_TIGHT_KCORE_BOUNDARY;
  int tight_queue_saturation =
      request->flags == WAFER_NCC_REQUEST_TIGHT_QUEUE_SATURATION;
  int tight_worker_scope =
      request->flags == WAFER_NCC_REQUEST_TIGHT_WORKER_SCOPE;
  int bounded_pair_window =
      request->flags == WAFER_NCC_REQUEST_BOUNDED_PAIR_WINDOW;
  int tight_submission = tight_depth_plus_one || tight_kcore_boundary ||
                         tight_queue_saturation || tight_worker_scope;
  int issued_any = 0;
  WaferNccProbeIssue *last_issued = NULL;
  uint64_t tight_worker_scope_target_control = 0;
  int tight_worker_scope_target_captured = 0;
  uint64_t plan_cycle_before = hooks->read_cycle(context);
  uint32_t schedule[WAFER_NCC_PROTOCOL_MAX_ISSUES];
  uint32_t schedule_count = 0;
  if (request->flags == WAFER_NCC_REQUEST_DOUBLE_SLOT_OBSERVATION) {
    for (uint32_t step = 0; step < request->rounds + 2U; ++step) {
      for (uint32_t lane = 0; lane < request->lane_count; ++lane) {
        if (step < lane)
          continue;
        uint32_t round = step - lane;
        if (round < request->rounds)
          schedule[schedule_count++] = lane * request->rounds + round;
      }
    }
  } else {
    for (uint32_t round = 0; round < request->rounds; ++round)
      for (uint32_t lane = 0; lane < request->lane_count; ++lane) {
        uint32_t ordinal = lane * request->rounds + round;
        if (ordinal < issue_count)
          schedule[schedule_count++] = ordinal;
      }
  }
  for (uint32_t scheduled = 0; scheduled < schedule_count; ++scheduled) {
    WaferNccProbeIssue *issue = &issues[schedule[scheduled]];
    const WaferNccProbeEngineAdapter *adapter =
        wafer_ncc_probe_find_adapter(adapters, adapter_count, issue->engine);
    uint64_t execute_rc = 0;
    uint64_t cycle_before = hooks->read_cycle(context);
    if (adapter->issue(context, request, issue, &execute_rc) != 0)
      return wafer_ncc_probe_finish_after_failure(
          WAFER_NCC_STATUS_ISSUE_FAILED, request, adapters, adapter_count,
          hooks, context, record, issues, prepared_count, issued_any);
    uint64_t cycle_after = hooks->read_cycle(context);
    if (tight_worker_scope && scheduled + 1U == schedule_count) {
      /*
       * The final issue belongs to the scope target by construction.  Capture
       * that worker first, before record bookkeeping or any
       * instrumentation-worker MMIO, so a bounded target backlog cannot drain
       * merely while the probe walks unrelated CSR windows.
       */
      tight_worker_scope_target_control =
          hooks->read_worker_control(context, issue->worker);
      tight_worker_scope_target_captured = 1;
    }
    issued_any = 1;

    uint32_t base =
        wafer_ncc_protocol_issue_word(issue->ordinal, WAFER_NCC_ISSUE_ORDINAL);
    record[base + WAFER_NCC_ISSUE_EXECUTE_RC] = execute_rc;
    record[base + WAFER_NCC_ISSUE_EXECUTE_CYCLES] = cycle_after - cycle_before;
    last_issued = issue;

    if (tight_submission)
      continue;

    record[base + WAFER_NCC_ISSUE_CONTROL_AFTER_ISSUE] =
        hooks->read_worker_control(context, issue->worker);
    WaferNccProbeObservation observation;
    wafer_ncc_probe_zero(&observation, sizeof(observation));
    observation.execute_rc = execute_rc;
    if (adapter->observe(context, request, issue, &observation) != 0)
      return wafer_ncc_probe_finish_after_failure(
          WAFER_NCC_STATUS_OBSERVATION_FAILED, request, adapters, adapter_count,
          hooks, context, record, issues, prepared_count, 1);
    wafer_ncc_probe_write_observation(record, issue, &observation);

    if (request->schedule == WAFER_NCC_SCHEDULE_SERIAL) {
      uint64_t serial_wait_before = hooks->read_cycle(context);
      int serial_wait_rc =
          hooks->serial_drain(context, UINT32_C(1) << issue->worker);
      uint64_t serial_wait_after = hooks->read_cycle(context);
      uint64_t serial_wait_cycles = serial_wait_after - serial_wait_before;
      uint64_t serial_wait_index = record[WAFER_NCC_REC_SERIAL_WAIT_COUNT];
      if (serial_wait_index < WAFER_NCC_PROTOCOL_MAX_WAIT_SAMPLES)
        record[WAFER_NCC_PROTOCOL_WAIT_SAMPLE_BASE + serial_wait_index] =
            serial_wait_cycles;
      record[WAFER_NCC_REC_SERIAL_WAIT_CYCLES] += serial_wait_cycles;
      ++record[WAFER_NCC_REC_SERIAL_WAIT_COUNT];
      if (serial_wait_rc != 0)
        return wafer_ncc_probe_finish_after_failure(
            WAFER_NCC_STATUS_DRAIN_FAILED, request, adapters, adapter_count,
            hooks, context, record, issues, prepared_count, 1);
    }
    if (bounded_pair_window &&
        scheduled + 1U ==
            request->lane_count * WAFER_NCC_PROTOCOL_MAX_THREE_LANE_ROUNDS) {
      uint64_t drain_before = hooks->read_cycle(context);
      int drain_rc = hooks->serial_drain(context, request->wait_worker_mask);
      uint64_t drain_after = hooks->read_cycle(context);
      record[WAFER_NCC_REC_BOUNDED_WINDOW_DRAIN_COUNT] = 1;
      record[WAFER_NCC_REC_BOUNDED_WINDOW_DRAIN_CYCLES] =
          drain_after - drain_before;
      if (drain_rc != 0)
        return wafer_ncc_probe_finish_after_failure(
            WAFER_NCC_STATUS_DRAIN_FAILED, request, adapters, adapter_count,
            hooks, context, record, issues, prepared_count, 1);
    }
  }

  if (tight_depth_plus_one) {
    uint32_t last_base = wafer_ncc_protocol_issue_word(last_issued->ordinal,
                                                       WAFER_NCC_ISSUE_ORDINAL);
    record[last_base + WAFER_NCC_ISSUE_CONTROL_AFTER_ISSUE] =
        hooks->read_worker_control(context, last_issued->worker);
    if (wafer_ncc_probe_observe_deferred(request, adapters, adapter_count,
                                         context, record, issues,
                                         issue_count) != 0)
      return wafer_ncc_probe_finish_after_failure(
          WAFER_NCC_STATUS_OBSERVATION_FAILED, request, adapters, adapter_count,
          hooks, context, record, issues, prepared_count, 1);
    record[last_base + WAFER_NCC_ISSUE_FLAGS] |=
        WAFER_NCC_ISSUE_WINDOW_CONTROL_VALID;
  }
  if (tight_queue_saturation || tight_worker_scope) {
    if (tight_worker_scope) {
      if (!tight_worker_scope_target_captured || last_issued == NULL)
        return wafer_ncc_probe_finish_after_failure(
            WAFER_NCC_STATUS_OBSERVATION_FAILED, request, adapters,
            adapter_count, hooks, context, record, issues, prepared_count,
            issued_any);
      uint32_t target_worker = last_issued->worker;
      record[WAFER_NCC_REC_CONTROL_PRE_WAIT + target_worker] =
          tight_worker_scope_target_control;
      for (uint32_t offset = 1; offset < WAFER_NCC_PROTOCOL_WORKERS; ++offset) {
        uint32_t worker = (target_worker + offset) % WAFER_NCC_PROTOCOL_WORKERS;
        record[WAFER_NCC_REC_CONTROL_PRE_WAIT + worker] =
            hooks->read_worker_control(context, worker);
      }
    } else {
      for (uint32_t worker = 0; worker < WAFER_NCC_PROTOCOL_WORKERS; ++worker)
        record[WAFER_NCC_REC_CONTROL_PRE_WAIT + worker] =
            hooks->read_worker_control(context, worker);
    }
    record[WAFER_NCC_REC_FLAGS] |= WAFER_NCC_RECORD_PRE_WAIT_CAPTURED;
  }

  uint64_t plan_cycle_after = 0;
  if (request->wait_kind != WAFER_NCC_WAIT_NONE) {
    uint64_t wait_before = hooks->read_cycle(context);
    int wait_rc = hooks->requested_wait(context, request->wait_kind,
                                        request->wait_worker_mask);
    uint64_t wait_after = hooks->read_cycle(context);
    plan_cycle_after = wait_after;
    record[WAFER_NCC_REC_WAIT_CYCLES] = wait_after - wait_before;
    if (wait_rc != 0)
      return wafer_ncc_probe_finish_after_failure(
          WAFER_NCC_STATUS_WAIT_FAILED, request, adapters, adapter_count, hooks,
          context, record, issues, prepared_count, issued_any);
  } else
    plan_cycle_after = hooks->read_cycle(context);
  record[WAFER_NCC_REC_PLAN_CYCLES] = plan_cycle_after - plan_cycle_before;
  record[WAFER_NCC_REC_FLAGS] |= WAFER_NCC_RECORD_REQUESTED_WAIT_DONE;

  if (hooks->snapshot(context, WAFER_NCC_SNAPSHOT_BOUNDARY, record) != 0)
    return wafer_ncc_probe_finish_after_failure(
        WAFER_NCC_STATUS_OBSERVATION_FAILED, request, adapters, adapter_count,
        hooks, context, record, issues, prepared_count, issued_any);
  record[WAFER_NCC_REC_FLAGS] |= WAFER_NCC_RECORD_BOUNDARY_CAPTURED;

  for (uint32_t index = 0; index < issue_count; ++index) {
    WaferNccProbeIssue *issue = &issues[index];
    const WaferNccProbeEngineAdapter *adapter =
        wafer_ncc_probe_find_adapter(adapters, adapter_count, issue->engine);
    uint64_t result_mismatches = 0;
    uint64_t guard_mismatches = 0;
    if (adapter->oracle(context, request, issue, WAFER_NCC_ORACLE_BOUNDARY,
                        &result_mismatches, &guard_mismatches) != 0)
      return wafer_ncc_probe_finish_after_failure(
          WAFER_NCC_STATUS_OBSERVATION_FAILED, request, adapters, adapter_count,
          hooks, context, record, issues, prepared_count, issued_any);
    uint32_t base =
        wafer_ncc_protocol_issue_word(index, WAFER_NCC_ISSUE_ORDINAL);
    record[base + WAFER_NCC_ISSUE_BOUNDARY_MISMATCHES] = result_mismatches;
    record[base + WAFER_NCC_ISSUE_BOUNDARY_GUARD_MISMATCHES] = guard_mismatches;
    record[WAFER_NCC_REC_BOUNDARY_MISMATCHES] += result_mismatches;
    record[WAFER_NCC_REC_BOUNDARY_GUARD_MISMATCHES] += guard_mismatches;
  }
  record[WAFER_NCC_REC_FLAGS] |= WAFER_NCC_RECORD_BOUNDARY_ORACLE_DONE;

  /*
   * Sample worker control, the mapped-SPM marker, and the boundary Kcore read
   * immediately after the tight NCC window (or its requested local wait).
   * Packet bookkeeping is deliberately deferred so a short workload cannot
   * drain merely while the probe records observations.
   */
  if ((tight_kcore_boundary || tight_queue_saturation || tight_worker_scope) &&
      wafer_ncc_probe_observe_deferred(request, adapters, adapter_count,
                                       context, record, issues,
                                       issue_count) != 0)
    return wafer_ncc_probe_finish_after_failure(
        WAFER_NCC_STATUS_OBSERVATION_FAILED, request, adapters, adapter_count,
        hooks, context, record, issues, prepared_count, issued_any);

  uint32_t participant_mask = wafer_ncc_probe_participant_mask(request);
  if (participant_mask != 0 &&
      hooks->safety_drain(context, participant_mask) != 0)
    return wafer_ncc_probe_finish_after_failure(
        WAFER_NCC_STATUS_DRAIN_FAILED, request, adapters, adapter_count, hooks,
        context, record, issues, prepared_count,
        /*issued_any=*/0);
  record[WAFER_NCC_REC_FLAGS] |= WAFER_NCC_RECORD_SAFETY_DRAIN_DONE;

  if (hooks->snapshot(context, WAFER_NCC_SNAPSHOT_FINAL, record) != 0)
    return wafer_ncc_probe_finish_after_failure(
        WAFER_NCC_STATUS_OBSERVATION_FAILED, request, adapters, adapter_count,
        hooks, context, record, issues, prepared_count, 0);
  record[WAFER_NCC_REC_FLAGS] |= WAFER_NCC_RECORD_FINAL_CAPTURED;

  for (uint32_t index = 0; index < issue_count; ++index) {
    WaferNccProbeIssue *issue = &issues[index];
    const WaferNccProbeEngineAdapter *adapter =
        wafer_ncc_probe_find_adapter(adapters, adapter_count, issue->engine);
    uint64_t result_mismatches = 0;
    uint64_t guard_mismatches = 0;
    if (adapter->oracle(context, request, issue, WAFER_NCC_ORACLE_FINAL,
                        &result_mismatches, &guard_mismatches) != 0)
      return wafer_ncc_probe_finish_after_failure(
          WAFER_NCC_STATUS_OBSERVATION_FAILED, request, adapters, adapter_count,
          hooks, context, record, issues, prepared_count, 0);
    uint32_t base =
        wafer_ncc_protocol_issue_word(index, WAFER_NCC_ISSUE_ORDINAL);
    record[base + WAFER_NCC_ISSUE_FINAL_MISMATCHES] = result_mismatches;
    record[base + WAFER_NCC_ISSUE_FINAL_GUARD_MISMATCHES] = guard_mismatches;
    record[WAFER_NCC_REC_FINAL_MISMATCHES] += result_mismatches;
    record[WAFER_NCC_REC_FINAL_GUARD_MISMATCHES] += guard_mismatches;
  }
  record[WAFER_NCC_REC_FLAGS] |= WAFER_NCC_RECORD_FINAL_ORACLE_DONE;

  wafer_ncc_probe_release_prepared(adapters, adapter_count, context, issues,
                                   prepared_count);
  record[WAFER_NCC_REC_STATUS] = WAFER_NCC_STATUS_OK;
  return WAFER_NCC_STATUS_OK;
}
