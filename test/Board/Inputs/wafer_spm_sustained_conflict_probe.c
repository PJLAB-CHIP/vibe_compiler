/*
 * Reuse the already-qualified TX81 packet builders, PMU sampler, cache
 * publication, and bounded worker drain from the memory-descriptor probe.
 * The legacy entry points are renamed and garbage-collected; this file exports
 * only the rank-one sustained-conflict entry consumed by the standard loader.
 */
#define wafer_tx81_instruction_family_probe                              \
  wafer_tx81_mdc_legacy_probe_not_exported
#define wafer_tx81_spm_cross_tile_conflict_probe                         \
  wafer_tx81_mdc_cross_tile_probe_not_exported
#include "wafer_memory_descriptor_calibration_probe.c"
#undef wafer_tx81_spm_cross_tile_conflict_probe
#undef wafer_tx81_instruction_family_probe

#include "wafer_spm_sustained_conflict_probe_protocol.h"

#define WAFER_SSC_MAX_ROUNDS 4U
#define WAFER_SSC_ENGINES 5U
#define WAFER_SSC_WORKER 0U
#define WAFER_SSC_MAX_PENDING_SETUP 4U
#define WAFER_SSC_MAX_DMA_CHUNK 65536U

typedef struct WaferSSCRequest {
  uint32_t group;
  uint32_t work_bytes;
  uint32_t rounds;
  uint32_t issue_order;
  uint32_t sample;
} WaferSSCRequest;

static void wafer_ssc_init_record(volatile uint64_t *record,
                                  uint32_t status) {
  for (uint32_t word = 0; word < WAFER_SSC_RECORD_WORDS; ++word)
    record[word] = 0U;
  record[WAFER_SSC_REC_MAGIC] = WAFER_SSC_RECORD_MAGIC;
  record[WAFER_SSC_REC_SCHEMA_AND_WORDS] =
      ((uint64_t)WAFER_SSC_SCHEMA << 32) | WAFER_SSC_RECORD_WORDS;
  record[WAFER_SSC_REC_STATUS] = status;
  record[WAFER_SSC_REC_ROWS] = WAFER_SSC_ROW_COUNT;
  record[WAFER_SSC_REC_TRANSFER_BYTES] = WAFER_SSC_TRANSFER_BYTES;
  record[WAFER_SSC_REC_SPM_BASE] = WAFER_SSC_SPM_BASE;
  record[WAFER_SSC_REC_REQUEST_GUARD] = WAFER_SSC_REQUEST_GUARD;
  record[WAFER_SSC_REC_RECORD_GUARD] = WAFER_SSC_RECORD_GUARD;
}

static volatile uint64_t *
wafer_ssc_row_record(volatile uint64_t *record, uint32_t row) {
  return record + WAFER_SSC_ROW_RECORD_BASE_WORD +
         row * WAFER_SSC_ROW_RECORD_WORDS;
}

static void wafer_ssc_init_row(volatile uint64_t *row_record,
                               uint32_t status) {
  for (uint32_t word = 0; word < WAFER_SSC_ROW_RECORD_WORDS; ++word)
    row_record[word] = 0U;
  row_record[WAFER_SSC_ROW_MAGIC_WORD] = WAFER_SSC_ROW_MAGIC;
  row_record[WAFER_SSC_ROW_SCHEMA_AND_WORDS] =
      ((uint64_t)WAFER_SSC_SCHEMA << 32) |
      WAFER_SSC_ROW_RECORD_WORDS;
  row_record[WAFER_SSC_ROW_STATUS] = status;
  row_record[WAFER_SSC_ROW_REQUEST_GUARD] =
      WAFER_SSC_REQUEST_GUARD;
  row_record[WAFER_SSC_ROW_GUARD_WORD] =
      WAFER_SSC_ROW_RECORD_GUARD;
}

static uint32_t wafer_ssc_decode(const volatile uint64_t *wire,
                                 WaferSSCRequest *request) {
  if (wire[WAFER_SSC_REQ_MAGIC] != WAFER_SSC_REQUEST_MAGIC ||
      wire[WAFER_SSC_REQ_SCHEMA_AND_WORDS] !=
          (((uint64_t)WAFER_SSC_SCHEMA << 32) |
           WAFER_SSC_REQUEST_WORDS) ||
      wire[WAFER_SSC_REQ_REPEATS] != WAFER_SSC_REPEATS ||
      wire[WAFER_SSC_REQ_ROWS] != WAFER_SSC_ROW_COUNT ||
      wire[WAFER_SSC_REQ_TRANSFER_BYTES] !=
          WAFER_SSC_TRANSFER_BYTES ||
      wire[WAFER_SSC_REQ_SPM_BASE] != WAFER_SSC_SPM_BASE ||
      wire[WAFER_SSC_REQ_SPM_CELL_STRIDE] !=
          WAFER_SSC_SPM_CELL_STRIDE ||
      wire[WAFER_SSC_REQ_SPM_READ0_OFFSET] !=
          WAFER_SSC_SPM_READ0_OFFSET ||
      wire[WAFER_SSC_REQ_SPM_READ1_OFFSET] !=
          WAFER_SSC_SPM_READ1_OFFSET ||
      wire[WAFER_SSC_REQ_SPM_WRITE_OFFSET] !=
          WAFER_SSC_SPM_WRITE_OFFSET ||
      wire[WAFER_SSC_REQ_CANDIDATE_OFFSET] !=
          WAFER_SSC_CANDIDATE_OFFSET ||
      wire[WAFER_SSC_REQ_CONTROL_OFFSET] !=
          WAFER_SSC_CONTROL_OFFSET ||
      wire[WAFER_SSC_REQ_RESOURCE_BYTES] !=
          WAFER_SSC_RESOURCE_BYTES ||
      wire[WAFER_SSC_REQ_PAYLOAD_READ0_OFFSET] !=
          WAFER_SSC_PAYLOAD_READ0_OFFSET ||
      wire[WAFER_SSC_REQ_PAYLOAD_READ1_OFFSET] !=
          WAFER_SSC_PAYLOAD_READ1_OFFSET ||
      wire[WAFER_SSC_REQ_PAYLOAD_RDMA_OFFSET] !=
          WAFER_SSC_PAYLOAD_RDMA_OFFSET ||
      wire[WAFER_SSC_REQ_PAYLOAD_CANARY_OFFSET] !=
          WAFER_SSC_PAYLOAD_CANARY_OFFSET ||
      wire[WAFER_SSC_REQ_PAYLOAD_CANARY_BYTES] !=
          WAFER_SSC_PAYLOAD_CANARY_BYTES ||
      wire[WAFER_SSC_REQ_OUTPUT_ARCHIVE_BASE] !=
          WAFER_SSC_OUTPUT_ARCHIVE_BASE ||
      wire[WAFER_SSC_REQ_OUTPUT_ARCHIVE_STRIDE] !=
          WAFER_SSC_OUTPUT_ARCHIVE_STRIDE ||
      wire[WAFER_SSC_REQ_ROW_RECORD_BASE_WORD] !=
          WAFER_SSC_ROW_RECORD_BASE_WORD ||
      wire[WAFER_SSC_REQ_ROW_RECORD_WORDS] !=
          WAFER_SSC_ROW_RECORD_WORDS ||
      wire[WAFER_SSC_REQ_RESOURCE_CANARY] !=
          WAFER_SSC_RESOURCE_CANARY ||
      wire[WAFER_SSC_REQ_SPM_CANARY] != WAFER_SSC_SPM_CANARY ||
      wire[WAFER_SSC_REQ_GUARD] != WAFER_SSC_REQUEST_GUARD ||
      wire[WAFER_SSC_REQ_RESERVED] != 0U)
    return WAFER_SSC_STATUS_BAD_REQUEST;

  request->group = (uint32_t)wire[WAFER_SSC_REQ_GROUP];
  request->work_bytes = (uint32_t)wire[WAFER_SSC_REQ_WORK_BYTES];
  request->rounds = (uint32_t)wire[WAFER_SSC_REQ_ROUNDS];
  request->issue_order = (uint32_t)wire[WAFER_SSC_REQ_ISSUE_ORDER];
  request->sample = (uint32_t)wire[WAFER_SSC_REQ_SAMPLE];
  uint32_t work_index =
      request->work_bytes == 4096U
          ? 0U
          : (request->work_bytes == 16384U ? 1U : UINT32_MAX);
  if (work_index == UINT32_MAX ||
      request->rounds !=
          request->work_bytes / WAFER_SSC_TRANSFER_BYTES ||
      request->rounds == 0U ||
      request->rounds > WAFER_SSC_MAX_ROUNDS ||
      request->issue_order > 1U ||
      request->sample >= WAFER_SSC_REPEATS ||
      request->group != work_index * 2U + request->issue_order ||
      wire[WAFER_SSC_REQ_OWNED_SPM_BYTES] !=
          (uint64_t)request->rounds * WAFER_SSC_SPM_CELL_STRIDE)
    return WAFER_SSC_STATUS_BAD_REQUEST;
  uint64_t owned_bytes =
      (uint64_t)request->rounds * WAFER_SSC_SPM_CELL_STRIDE;
  if (WAFER_SSC_SPM_BASE <
          WAFER_MDC_SPM_ALLOCATABLE_BEGIN ||
      owned_bytes == 0U ||
      WAFER_SSC_SPM_BASE + owned_bytes >
          WAFER_MDC_SPM_ALLOCATABLE_END ||
      WAFER_SSC_OUTPUT_ARCHIVE_BASE +
              WAFER_SSC_ROW_COUNT *
                  WAFER_SSC_OUTPUT_ARCHIVE_STRIDE >
          WAFER_SSC_RESOURCE_BYTES ||
      WAFER_SSC_OUTPUT_ARCHIVE_STRIDE < owned_bytes)
    return WAFER_SSC_STATUS_BAD_REQUEST;
  return WAFER_SSC_STATUS_OK;
}

static uint32_t wafer_ssc_drain_setup(uint32_t *pending,
                                      uint64_t *control) {
  if (*pending == 0U)
    return 1U;
  if (!wafer_mdc_drain_worker(WAFER_SSC_WORKER, control))
    return 0U;
  *pending = 0U;
  return 1U;
}

static uint32_t wafer_ssc_issue_seed_chunk(uint64_t payload_ddr,
                                           uint64_t spm,
                                           uint32_t bytes,
                                           uint32_t *pending,
                                           uint64_t *control) {
  wafer_tx81_rdma(
      payload_ddr + WAFER_SSC_PAYLOAD_CANARY_OFFSET, spm, bytes, bytes,
      0U, 0U, 0U, 1U, 1U, 1U, Fmt_UINT8);
  ++*pending;
  return *pending < WAFER_SSC_MAX_PENDING_SETUP ||
         wafer_ssc_drain_setup(pending, control);
}

static uint32_t wafer_ssc_seed_owned_spm(const WaferSSCRequest *request,
                                         uint64_t payload_ddr) {
  uint64_t owned_bytes =
      (uint64_t)request->rounds * WAFER_SSC_SPM_CELL_STRIDE;
  uint64_t cursor = 0U;
  uint32_t pending = 0U;
  uint64_t control = 0U;
  while (cursor < owned_bytes) {
    uint32_t chunk =
        owned_bytes - cursor > WAFER_SSC_MAX_DMA_CHUNK
            ? WAFER_SSC_MAX_DMA_CHUNK
            : (uint32_t)(owned_bytes - cursor);
    if (!wafer_ssc_issue_seed_chunk(
            payload_ddr, WAFER_SSC_SPM_BASE + cursor, chunk, &pending,
            &control))
      return 0U;
    cursor += chunk;
  }
  for (uint32_t round = 0; round < request->rounds; ++round) {
    uint64_t cell =
        WAFER_SSC_SPM_BASE +
        (uint64_t)round * WAFER_SSC_SPM_CELL_STRIDE;
    wafer_tx81_rdma(
        payload_ddr + WAFER_SSC_PAYLOAD_READ0_OFFSET +
            (uint64_t)round * WAFER_SSC_TRANSFER_BYTES,
        cell + WAFER_SSC_SPM_READ0_OFFSET,
        WAFER_SSC_TRANSFER_BYTES, WAFER_SSC_TRANSFER_BYTES, 0U, 0U, 0U,
        1U, 1U, 1U, Fmt_UINT8);
    if (++pending == WAFER_SSC_MAX_PENDING_SETUP &&
        !wafer_ssc_drain_setup(&pending, &control))
      return 0U;
    wafer_tx81_rdma(
        payload_ddr + WAFER_SSC_PAYLOAD_READ1_OFFSET +
            (uint64_t)round * WAFER_SSC_TRANSFER_BYTES,
        cell + WAFER_SSC_SPM_READ1_OFFSET,
        WAFER_SSC_TRANSFER_BYTES, WAFER_SSC_TRANSFER_BYTES, 0U, 0U, 0U,
        1U, 1U, 1U, Fmt_UINT8);
    if (++pending == WAFER_SSC_MAX_PENDING_SETUP &&
        !wafer_ssc_drain_setup(&pending, &control))
      return 0U;
  }
  return wafer_ssc_drain_setup(&pending, &control);
}

static uint32_t wafer_ssc_prepare_packets(
    const WaferSSCRequest *request, uint64_t payload_ddr,
    uint64_t relative_offset,
    WaferMDCPreparedInstruction
        packets[2][WAFER_SSC_MAX_ROUNDS]) {
  for (uint32_t round = 0; round < request->rounds; ++round) {
    uint64_t cell =
        WAFER_SSC_SPM_BASE +
        (uint64_t)round * WAFER_SSC_SPM_CELL_STRIDE;
    uint64_t read0 = cell + WAFER_SSC_SPM_READ0_OFFSET;
    uint64_t read1 = cell + WAFER_SSC_SPM_READ1_OFFSET;
    uint64_t write = cell + WAFER_SSC_SPM_WRITE_OFFSET;

    packets[0][round].engine = WAFER_MDC_ENGINE_CT;
    TsmArith *arith = TsmNewArith();
    if (arith == 0)
      return 0U;
    arith->AddVV(
        &packets[0][round].packet.ct, read0, read1, write,
        WAFER_SSC_TRANSFER_BYTES / sizeof(uint16_t), RND_NEAREST_EVEN,
        Fmt_FP16);
    wafer_mdc_set_packet_worker(
        &packets[0][round].packet.ct.inter_type, WAFER_SSC_WORKER);
    TsmDeleteArith(arith);

    packets[1][round].engine = WAFER_MDC_ENGINE_RDMA;
    TsmRdma *rdma = TsmNewRdma();
    if (rdma == 0)
      return 0U;
    rdma->AddSrcDst(
        &packets[1][round].packet.rdma,
        payload_ddr + WAFER_SSC_PAYLOAD_RDMA_OFFSET +
            (uint64_t)round * WAFER_SSC_TRANSFER_BYTES,
        write + relative_offset, Fmt_UINT8);
    rdma->ConfigStrideIteration(
        &packets[1][round].packet.rdma, WAFER_SSC_TRANSFER_BYTES, 0U, 1U,
        0U, 1U, 0U, 1U);
    wafer_mdc_set_packet_worker(
        &packets[1][round].packet.rdma.inter_type, WAFER_SSC_WORKER);
    TsmDeleteRdma(rdma);
  }
  return 1U;
}

static uint32_t wafer_ssc_issue_lane(
    WaferMDCPreparedInstruction
        packets[2][WAFER_SSC_MAX_ROUNDS],
    uint32_t lane, uint32_t rounds) {
  for (uint32_t round = 0; round < rounds; ++round)
    if (TsmExecute(
            wafer_mdc_prepared_packet(&packets[lane][round])) != 1)
      return 0U;
  return 1U;
}

static uint32_t wafer_ssc_issue_measured(
    const WaferSSCRequest *request, uint32_t schedule,
    WaferMDCPreparedInstruction
        packets[2][WAFER_SSC_MAX_ROUNDS],
    uint64_t *final_control) {
  uint32_t first = request->issue_order == 0U ? 0U : 1U;
  uint32_t second = first ^ 1U;
  if (schedule == WAFER_MDC_SCHEDULE_SERIAL) {
    if (!wafer_ssc_issue_lane(packets, first, request->rounds) ||
        !wafer_mdc_drain_worker(WAFER_SSC_WORKER, final_control) ||
        !wafer_ssc_issue_lane(packets, second, request->rounds) ||
        !wafer_mdc_drain_worker(WAFER_SSC_WORKER, final_control))
      return 0U;
    return 1U;
  }
  for (uint32_t round = 0; round < request->rounds; ++round)
    if (TsmExecute(
            wafer_mdc_prepared_packet(&packets[first][round])) != 1 ||
        TsmExecute(
            wafer_mdc_prepared_packet(&packets[second][round])) != 1)
      return 0U;
  return wafer_mdc_drain_worker(WAFER_SSC_WORKER, final_control);
}

static uint32_t wafer_ssc_archive_owned_spm(
    const WaferSSCRequest *request, uint64_t output_ddr,
    uint64_t archive_offset) {
  uint64_t owned_bytes =
      (uint64_t)request->rounds * WAFER_SSC_SPM_CELL_STRIDE;
  uint64_t cursor = 0U;
  uint32_t pending = 0U;
  uint64_t control = 0U;
  while (cursor < owned_bytes) {
    uint32_t chunk =
        owned_bytes - cursor > WAFER_SSC_MAX_DMA_CHUNK
            ? WAFER_SSC_MAX_DMA_CHUNK
            : (uint32_t)(owned_bytes - cursor);
    wafer_tx81_wdma(
        WAFER_SSC_SPM_BASE + cursor,
        output_ddr + archive_offset + cursor, chunk, chunk, 0U, 0U, 0U,
        1U, 1U, 1U, Fmt_UINT8);
    cursor += chunk;
    if (++pending == WAFER_SSC_MAX_PENDING_SETUP &&
        !wafer_ssc_drain_setup(&pending, &control))
      return 0U;
  }
  return wafer_ssc_drain_setup(&pending, &control);
}

static void wafer_ssc_write_row_contract(
    volatile uint64_t *row_record, const WaferSSCRequest *request,
    uint32_t row, uint32_t ordinal, uint64_t relative_offset,
    uint32_t schedule, uint64_t request_ddr, uint64_t payload_ddr,
    uint64_t output_ddr) {
  uint32_t cell = request->group * WAFER_SSC_ROW_COUNT + row;
  uint64_t owned_bytes =
      (uint64_t)request->rounds * WAFER_SSC_SPM_CELL_STRIDE;
  row_record[WAFER_SSC_ROW_CELL] = cell;
  row_record[WAFER_SSC_ROW_ADDRESS_CLASS] = row / 2U;
  row_record[WAFER_SSC_ROW_RELATIVE_OFFSET] = relative_offset;
  row_record[WAFER_SSC_ROW_SCHEDULE] = schedule;
  row_record[WAFER_SSC_ROW_ISSUE_ORDER] = request->issue_order;
  row_record[WAFER_SSC_ROW_SAMPLE] = request->sample;
  row_record[WAFER_SSC_ROW_EXECUTION_ORDINAL] = ordinal;
  row_record[WAFER_SSC_ROW_ROUNDS] = request->rounds;
  row_record[WAFER_SSC_ROW_TRANSFER_BYTES] =
      WAFER_SSC_TRANSFER_BYTES;
  row_record[WAFER_SSC_ROW_SPM_BASE] = WAFER_SSC_SPM_BASE;
  row_record[WAFER_SSC_ROW_OWNED_SPM_BYTES] = owned_bytes;
  row_record[WAFER_SSC_ROW_SPM_CELL_STRIDE] =
      WAFER_SSC_SPM_CELL_STRIDE;
  row_record[WAFER_SSC_ROW_SPM_READ0_OFFSET] =
      WAFER_SSC_SPM_READ0_OFFSET;
  row_record[WAFER_SSC_ROW_SPM_READ1_OFFSET] =
      WAFER_SSC_SPM_READ1_OFFSET;
  row_record[WAFER_SSC_ROW_SPM_WRITE_OFFSET] =
      WAFER_SSC_SPM_WRITE_OFFSET;
  row_record[WAFER_SSC_ROW_RDMA_WRITE_OFFSET] =
      WAFER_SSC_SPM_WRITE_OFFSET + relative_offset;
  row_record[WAFER_SSC_ROW_ARCHIVE_OFFSET] =
      WAFER_SSC_OUTPUT_ARCHIVE_BASE +
      (uint64_t)row * WAFER_SSC_OUTPUT_ARCHIVE_STRIDE;
  row_record[WAFER_SSC_ROW_REQUEST_DDR] = request_ddr;
  row_record[WAFER_SSC_ROW_PAYLOAD_DDR] = payload_ddr;
  row_record[WAFER_SSC_ROW_OUTPUT_DDR] = output_ddr;
}

static uint32_t wafer_ssc_execute_row(
    const WaferSSCRequest *request, uint32_t row, uint32_t ordinal,
    uint64_t request_ddr, uint64_t payload_ddr, uint64_t output_ddr,
    volatile uint64_t *row_record) {
  uint32_t address_class = row / 2U;
  uint32_t schedule = row % 2U;
  uint64_t relative_offset =
      address_class == 0U ? WAFER_SSC_CANDIDATE_OFFSET
                          : WAFER_SSC_CONTROL_OFFSET;
  wafer_ssc_init_row(row_record, WAFER_SSC_STATUS_PREPARE_FAILED);
  wafer_ssc_write_row_contract(
      row_record, request, row, ordinal, relative_offset, schedule,
      request_ddr, payload_ddr, output_ddr);

  if (!wafer_ssc_seed_owned_spm(request, payload_ddr)) {
    row_record[WAFER_SSC_ROW_STATUS] =
        WAFER_SSC_STATUS_PREPARE_FAILED;
    return WAFER_SSC_STATUS_PREPARE_FAILED;
  }
  WaferMDCPreparedInstruction
      packets[2][WAFER_SSC_MAX_ROUNDS] = {0};
  if (!wafer_ssc_prepare_packets(
          request, payload_ddr, relative_offset, packets)) {
    row_record[WAFER_SSC_ROW_STATUS] =
        WAFER_SSC_STATUS_PREPARE_FAILED;
    return WAFER_SSC_STATUS_PREPARE_FAILED;
  }

  WaferMDCPMU before = wafer_mdc_read_pmu();
  uint64_t plan_begin = wafer_mdc_cycle();
  uint64_t final_control = 0U;
  if (!wafer_ssc_issue_measured(
          request, schedule, packets, &final_control)) {
    row_record[WAFER_SSC_ROW_STATUS] =
        WAFER_SSC_STATUS_EXECUTE_FAILED;
    return WAFER_SSC_STATUS_EXECUTE_FAILED;
  }
  uint64_t plan_end = wafer_mdc_cycle();
  WaferMDCPMU after = wafer_mdc_read_pmu();

  uint32_t other_instructions = 0U;
  for (uint32_t engine = 0; engine < WAFER_SSC_ENGINES; ++engine)
    if (engine != WAFER_MDC_ENGINE_CT &&
        engine != WAFER_MDC_ENGINE_RDMA)
      other_instructions +=
          (uint32_t)(after.instructions[engine] -
                     before.instructions[engine]);
  row_record[WAFER_SSC_ROW_CT_INST_DELTA] =
      (uint32_t)(after.instructions[WAFER_MDC_ENGINE_CT] -
                 before.instructions[WAFER_MDC_ENGINE_CT]);
  row_record[WAFER_SSC_ROW_RDMA_INST_DELTA] =
      (uint32_t)(after.instructions[WAFER_MDC_ENGINE_RDMA] -
                 before.instructions[WAFER_MDC_ENGINE_RDMA]);
  row_record[WAFER_SSC_ROW_OTHER_INST_DELTA] = other_instructions;
  row_record[WAFER_SSC_ROW_WORKER_CT_INST_DELTA] =
      (uint32_t)(
          after.worker_instructions[WAFER_SSC_WORKER]
                                   [WAFER_MDC_ENGINE_CT] -
          before.worker_instructions[WAFER_SSC_WORKER]
                                    [WAFER_MDC_ENGINE_CT]);
  row_record[WAFER_SSC_ROW_WORKER_RDMA_INST_DELTA] =
      (uint32_t)(
          after.worker_instructions[WAFER_SSC_WORKER]
                                   [WAFER_MDC_ENGINE_RDMA] -
          before.worker_instructions[WAFER_SSC_WORKER]
                                    [WAFER_MDC_ENGINE_RDMA]);
  row_record[WAFER_SSC_ROW_CT_EXEC_DELTA] =
      after.execution[WAFER_MDC_ENGINE_CT] -
      before.execution[WAFER_MDC_ENGINE_CT];
  row_record[WAFER_SSC_ROW_RDMA_EXEC_DELTA] =
      after.execution[WAFER_MDC_ENGINE_RDMA] -
      before.execution[WAFER_MDC_ENGINE_RDMA];
  row_record[WAFER_SSC_ROW_FULL_EXEC_DELTA] =
      after.full_execution - before.full_execution;
  row_record[WAFER_SSC_ROW_CT_BLOCKING_DELTA] =
      (uint32_t)(after.blocking[WAFER_MDC_ENGINE_CT] -
                 before.blocking[WAFER_MDC_ENGINE_CT]);
  row_record[WAFER_SSC_ROW_RDMA_BLOCKING_DELTA] =
      (uint32_t)(after.blocking[WAFER_MDC_ENGINE_RDMA] -
                 before.blocking[WAFER_MDC_ENGINE_RDMA]);
  row_record[WAFER_SSC_ROW_PLAN_CYCLES] = plan_end - plan_begin;
  row_record[WAFER_SSC_ROW_PMU_ENABLE] = wafer_mdc_read32(GR_PMU_EN);
  row_record[WAFER_SSC_ROW_SERIAL_MODE] =
      get_ncc_reg(WAFER_SSC_WORKER, GR_CSR_SERIAL_MODE_ADDR);
  row_record[WAFER_SSC_ROW_STABLE_BEFORE] = before.stable_mask;
  row_record[WAFER_SSC_ROW_STABLE_AFTER] = after.stable_mask;
  row_record[WAFER_SSC_ROW_FINAL_CONTROL] = final_control;
  row_record[WAFER_SSC_ROW_FLAGS] =
      WAFER_SSC_FLAG_PREPARED | WAFER_SSC_FLAG_ISSUED |
      WAFER_SSC_FLAG_COMPLETED | WAFER_SSC_FLAG_PMU_RECORDED;

  uint64_t archive_offset =
      WAFER_SSC_OUTPUT_ARCHIVE_BASE +
      (uint64_t)row * WAFER_SSC_OUTPUT_ARCHIVE_STRIDE;
  if (!wafer_ssc_archive_owned_spm(
          request, output_ddr, archive_offset)) {
    row_record[WAFER_SSC_ROW_STATUS] =
        WAFER_SSC_STATUS_READBACK_FAILED;
    return WAFER_SSC_STATUS_READBACK_FAILED;
  }
  row_record[WAFER_SSC_ROW_FLAGS] = WAFER_SSC_REQUIRED_FLAGS;
  row_record[WAFER_SSC_ROW_STATUS] = WAFER_SSC_STATUS_OK;
  return WAFER_SSC_STATUS_OK;
}

__attribute__((visibility("hidden"))) void
wafer_tx81_instruction_family_probe(uint64_t request_ddr,
                                    uint64_t payload_ddr,
                                    uint64_t output_ddr) {
  wafer_mdc_cache_range(
      request_ddr, WAFER_SSC_RESOURCE_BYTES, 1U);
  wafer_mdc_cache_range(
      payload_ddr, WAFER_SSC_RESOURCE_BYTES, 1U);
  const volatile uint64_t *wire =
      (const volatile uint64_t *)(uintptr_t)request_ddr;
  volatile uint64_t *record =
      (volatile uint64_t *)(uintptr_t)output_ddr;
  WaferSSCRequest request = {0};
  uint32_t status = wafer_ssc_decode(wire, &request);
  wafer_ssc_init_record(record, status);
  for (uint32_t row = 0; row < WAFER_SSC_ROW_COUNT; ++row)
    wafer_ssc_init_row(
        wafer_ssc_row_record(record, row), status);

  if (status == WAFER_SSC_STATUS_OK) {
    record[WAFER_SSC_REC_GROUP] = request.group;
    record[WAFER_SSC_REC_WORK_BYTES] = request.work_bytes;
    record[WAFER_SSC_REC_ROUNDS] = request.rounds;
    record[WAFER_SSC_REC_ISSUE_ORDER] = request.issue_order;
    record[WAFER_SSC_REC_SAMPLE] = request.sample;
    record[WAFER_SSC_REC_OWNED_SPM_BYTES] =
        (uint64_t)request.rounds * WAFER_SSC_SPM_CELL_STRIDE;
    record[WAFER_SSC_REC_REQUEST_DDR] = request_ddr;
    record[WAFER_SSC_REC_PAYLOAD_DDR] = payload_ddr;
    record[WAFER_SSC_REC_OUTPUT_DDR] = output_ddr;
    record[WAFER_SSC_REC_EXECUTION_ROTATION] = request.sample;
    for (uint32_t ordinal = 0; ordinal < WAFER_SSC_ROW_COUNT;
         ++ordinal) {
      uint32_t row =
          (ordinal + request.sample) % WAFER_SSC_ROW_COUNT;
      uint32_t row_status = wafer_ssc_execute_row(
          &request, row, ordinal, request_ddr, payload_ddr, output_ddr,
          wafer_ssc_row_record(record, row));
      if (row_status != WAFER_SSC_STATUS_OK) {
        status = row_status;
        break;
      }
      record[WAFER_SSC_REC_COMPLETED_ROWS] = ordinal + 1U;
    }
    if (status == WAFER_SSC_STATUS_OK)
      record[WAFER_SSC_REC_FLAGS] = WAFER_SSC_REQUIRED_FLAGS;
    record[WAFER_SSC_REC_STATUS] = status;
  }
  wafer_mdc_cache_range(
      output_ddr,
      (WAFER_SSC_ROW_RECORD_BASE_WORD +
       WAFER_SSC_ROW_COUNT * WAFER_SSC_ROW_RECORD_WORDS) *
          sizeof(uint64_t),
      0U);
}
