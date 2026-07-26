/*
 * Reuse the qualified 16-rank direct-DTE entry ABI, PMU reader, cache
 * maintenance, DMA wrappers, barriers, and status-v2 lifecycle from the
 * physical-tile DDR probe.  Rename its entry and provide the active-mask
 * experiment below; both functions stay in one device translation unit.
 */
#define wafer_tx81_ddr_tile_offset_probe                                       \
  wafer_tx81_ddr_tile_offset_probe_sequential_reference
#include "wafer_ddr_tile_offset_probe.c"
#undef wafer_tx81_ddr_tile_offset_probe

#include "wafer_ddr_active_rank_contention_probe_protocol.h"

static uint64_t wafer_dar_read64_stable(uint32_t low_offset,
                                        uint32_t *stable) {
  uint32_t low = 0U;
  uint32_t high_after = 0U;
  *stable = 0U;
  for (uint32_t retry = 0; retry < WAFER_DDR_TILE_PMU_STABLE_RETRIES;
       ++retry) {
    uint32_t high_before = wafer_ddr_tile_read32(low_offset + 4U);
    low = wafer_ddr_tile_read32(low_offset);
    high_after = wafer_ddr_tile_read32(low_offset + 4U);
    if (high_before == high_after) {
      *stable = 1U;
      break;
    }
  }
  return ((uint64_t)high_after << 32) | low;
}

static WaferDDRTilePMU wafer_dar_read_pmu(uint32_t *stable) {
  WaferDDRTilePMU result;
  uint32_t counter_stable = 0U;
  *stable = 1U;
  result.rdma_instructions = wafer_ddr_tile_read32(GR_PMU_RDMA_INST_NUMS);
  result.wdma_instructions = wafer_ddr_tile_read32(GR_PMU_WDMA_INST_NUMS);
  result.rdma_blocking = wafer_ddr_tile_read32(GR_PMU_RDMA_BLOCKING_TIME);
  result.wdma_blocking = wafer_ddr_tile_read32(GR_PMU_WDMA_BLOCKING_TIME);
  result.rdma_execution =
      wafer_dar_read64_stable(GR_PMU_RDMA_EXE_TIME, &counter_stable);
  *stable &= counter_stable;
  result.wdma_execution =
      wafer_dar_read64_stable(GR_PMU_WDMA_EXE_TIME, &counter_stable);
  *stable &= counter_stable;
  result.fu_execution =
      wafer_dar_read64_stable(GR_PMU_FU_EXE_TIME, &counter_stable);
  *stable &= counter_stable;
  result.statistics_window =
      wafer_dar_read64_stable(GR_PMU_STATISTICS_WINDOW, &counter_stable);
  *stable &= counter_stable;
  return result;
}

static uint32_t wafer_dar_decode(uint32_t rank,
                                 const volatile uint64_t *request) {
  if (request[WAFER_DAR_REQ_MAGIC] != WAFER_DAR_REQUEST_MAGIC ||
      request[WAFER_DAR_REQ_SCHEMA_AND_WORDS] !=
          (((uint64_t)WAFER_DAR_SCHEMA << 32) | WAFER_DAR_REQUEST_WORDS) ||
      request[WAFER_DAR_REQ_RANK] != rank ||
      request[WAFER_DAR_REQ_GUARD] != WAFER_DAR_REQUEST_GUARD ||
      request[WAFER_DAR_REQ_RESOURCE_BYTES] != WAFER_DAR_RESOURCE_BYTES ||
      request[WAFER_DAR_REQ_PAYLOAD_SEED] == 0U ||
      request[WAFER_DAR_REQ_ACTIVE_MASK] == 0U ||
      (request[WAFER_DAR_REQ_ACTIVE_MASK] &
       ~((UINT64_C(1) << WAFER_DAR_RANKS) - 1U)) != 0U ||
      request[WAFER_DAR_REQ_DIRECTION] < WAFER_DAR_DIRECTION_RDMA ||
      request[WAFER_DAR_REQ_DIRECTION] > WAFER_DAR_DIRECTION_BIDIRECTIONAL ||
      (request[WAFER_DAR_REQ_PAYLOAD_BYTES] != 4096U &&
       request[WAFER_DAR_REQ_PAYLOAD_BYTES] != 65536U) ||
      request[WAFER_DAR_REQ_INNER_BYTES] == 0U ||
      request[WAFER_DAR_REQ_ITERATION0] == 0U ||
      request[WAFER_DAR_REQ_INNER_BYTES] * request[WAFER_DAR_REQ_ITERATION0] !=
          request[WAFER_DAR_REQ_PAYLOAD_BYTES] ||
      request[WAFER_DAR_REQ_ISSUE_ORDER] > 1U)
    return WAFER_DAR_STATUS_BAD_REQUEST;

  if (request[WAFER_DAR_REQ_ITERATION0] == 1U) {
    if (request[WAFER_DAR_REQ_INNER_BYTES] !=
            request[WAFER_DAR_REQ_PAYLOAD_BYTES] ||
        request[WAFER_DAR_REQ_STRIDE0] != 0U)
      return WAFER_DAR_STATUS_BAD_REQUEST;
  } else if (request[WAFER_DAR_REQ_INNER_BYTES] != 4096U ||
             request[WAFER_DAR_REQ_STRIDE0] != 8192U ||
             request[WAFER_DAR_REQ_ITERATION0] != 16U) {
    return WAFER_DAR_STATUS_BAD_REQUEST;
  }

  uint32_t direction = (uint32_t)request[WAFER_DAR_REQ_DIRECTION];
  uint32_t expected_rdma = direction == WAFER_DAR_DIRECTION_WDMA ? 0U : 1U;
  uint32_t expected_wdma = direction == WAFER_DAR_DIRECTION_RDMA ? 0U : 1U;
  if (request[WAFER_DAR_REQ_EXPECTED_RDMA] != expected_rdma ||
      request[WAFER_DAR_REQ_EXPECTED_WDMA] != expected_wdma)
    return WAFER_DAR_STATUS_BAD_REQUEST;
  return WAFER_DAR_STATUS_OK;
}

static void wafer_dar_target_dma(uint32_t direction, uint64_t source,
                                 uint64_t destination, uint32_t payload_bytes,
                                 uint32_t inner_bytes, uint32_t stride0,
                                 uint32_t iteration0) {
  if (direction == WAFER_DAR_DIRECTION_RDMA)
    wafer_tx81_rdma(source, destination, payload_bytes, inner_bytes, stride0,
                    0U, 0U, iteration0, 1U, 1U, Fmt_UINT8);
  else
    wafer_tx81_wdma(source, destination, payload_bytes, inner_bytes, stride0,
                    0U, 0U, iteration0, 1U, 1U, Fmt_UINT8);
}

static void wafer_dar_copy_chunks(uint32_t direction, uint64_t source,
                                  uint64_t destination, uint32_t bytes) {
  while (bytes != 0U) {
    uint32_t chunk = bytes > WAFER_DAR_MAX_CONTIGUOUS_DMA_BYTES
                         ? WAFER_DAR_MAX_CONTIGUOUS_DMA_BYTES
                         : bytes;
    if (direction == WAFER_DAR_DIRECTION_RDMA)
      wafer_ddr_tile_rdma(source, destination, chunk);
    else
      wafer_ddr_tile_wdma(source, destination, chunk);
    source += chunk;
    destination += chunk;
    bytes -= chunk;
  }
}

static uint64_t wafer_dar_canary_word(uint32_t rank, uint32_t sample,
                                      uint32_t word) {
  return UINT64_C(0xC39A57E10D2468BF) ^ ((uint64_t)rank << 48) ^
         ((uint64_t)sample << 24) ^
         ((uint64_t)word * UINT64_C(0x9E3779B97F4A7C15));
}

static void wafer_dar_write_inactive_canary(uint64_t output, uint32_t rank,
                                            uint32_t sample) {
  volatile uint64_t *words = (volatile uint64_t *)(uintptr_t)output;
  for (uint32_t word = 0;
       word < WAFER_DAR_INACTIVE_CANARY_BYTES / sizeof(uint64_t); ++word)
    words[word] = wafer_dar_canary_word(rank, sample, word);
  wafer_ddr_tile_cache_range(output, WAFER_DAR_INACTIVE_CANARY_BYTES, 0U);
}

static void wafer_dar_write_record(
    volatile uint64_t *record, uint32_t status, uint32_t rank,
    const volatile uint64_t *request, uint32_t active, uint32_t envelope_bytes,
    const WaferDDRTilePMU *before, const WaferDDRTilePMU *after,
    uint32_t before_stable, uint32_t after_stable, uint64_t completion_cycles,
    uint64_t input0, uint64_t input1, uint64_t output0, uint64_t output1) {
  for (uint32_t index = 0; index < WAFER_DAR_RECORD_WORDS; ++index)
    record[index] = 0U;
  record[WAFER_DAR_REC_MAGIC] = WAFER_DAR_RECORD_MAGIC;
  record[WAFER_DAR_REC_SCHEMA_AND_WORDS] =
      ((uint64_t)WAFER_DAR_SCHEMA << 32) | WAFER_DAR_RECORD_WORDS;
  record[WAFER_DAR_REC_STATUS] = status;
  record[WAFER_DAR_REC_RANK] = rank;
  record[WAFER_DAR_REC_SAMPLE] = request[WAFER_DAR_REQ_SAMPLE];
  record[WAFER_DAR_REC_ACTIVE_MASK] = request[WAFER_DAR_REQ_ACTIVE_MASK];
  record[WAFER_DAR_REC_ACTIVE] = active;
  record[WAFER_DAR_REC_DIRECTION] = request[WAFER_DAR_REQ_DIRECTION];
  record[WAFER_DAR_REC_PAYLOAD_BYTES] = request[WAFER_DAR_REQ_PAYLOAD_BYTES];
  record[WAFER_DAR_REC_INNER_BYTES] = request[WAFER_DAR_REQ_INNER_BYTES];
  record[WAFER_DAR_REC_STRIDE0] = request[WAFER_DAR_REQ_STRIDE0];
  record[WAFER_DAR_REC_ITERATION0] = request[WAFER_DAR_REQ_ITERATION0];
  record[WAFER_DAR_REC_ENVELOPE_BYTES] = envelope_bytes;
  record[WAFER_DAR_REC_RDMA_INST_DELTA] =
      after->rdma_instructions - before->rdma_instructions;
  record[WAFER_DAR_REC_WDMA_INST_DELTA] =
      after->wdma_instructions - before->wdma_instructions;
  record[WAFER_DAR_REC_RDMA_BLOCKING_DELTA] =
      after->rdma_blocking - before->rdma_blocking;
  record[WAFER_DAR_REC_WDMA_BLOCKING_DELTA] =
      after->wdma_blocking - before->wdma_blocking;
  record[WAFER_DAR_REC_RDMA_EXEC_DELTA] =
      after->rdma_execution - before->rdma_execution;
  record[WAFER_DAR_REC_WDMA_EXEC_DELTA] =
      after->wdma_execution - before->wdma_execution;
  record[WAFER_DAR_REC_FU_EXEC_DELTA] =
      after->fu_execution - before->fu_execution;
  record[WAFER_DAR_REC_WINDOW_DELTA] =
      after->statistics_window - before->statistics_window;
  record[WAFER_DAR_REC_COMPLETION_CYCLES] = completion_cycles;
  record[WAFER_DAR_REC_INPUT0_BASE] = input0;
  record[WAFER_DAR_REC_INPUT1_BASE] = input1;
  record[WAFER_DAR_REC_OUTPUT0_BASE] = output0;
  record[WAFER_DAR_REC_OUTPUT1_BASE] = output1;
  record[WAFER_DAR_REC_SPM_RDMA] = WAFER_DAR_SPM_RDMA;
  record[WAFER_DAR_REC_SPM_WDMA] = WAFER_DAR_SPM_WDMA;
  record[WAFER_DAR_REC_ISSUE_ORDER] = request[WAFER_DAR_REQ_ISSUE_ORDER];
  record[WAFER_DAR_REC_REQUEST_GUARD] = WAFER_DAR_REQUEST_GUARD;
  record[WAFER_DAR_REC_PAYLOAD_SEED] = request[WAFER_DAR_REQ_PAYLOAD_SEED];
  record[WAFER_DAR_REC_TARGET_ONLY_WINDOW] = 1U;
  record[WAFER_DAR_REC_PMU_BEFORE_STABLE] = before_stable;
  record[WAFER_DAR_REC_PMU_AFTER_STABLE] = after_stable;
  record[WAFER_DAR_REC_RECORD_GUARD] = WAFER_DAR_RECORD_GUARD;
}

__attribute__((visibility("hidden"))) void
wafer_tx81_ddr_tile_offset_probe(uint32_t rank, uint64_t input0,
                                 uint64_t input1, uint64_t output0,
                                 uint64_t output1, uint64_t status_ddr) {
  wafer_tx81_direct_dte_begin_after_prepare(status_ddr, WAFER_DAR_RANKS);
  wafer_ddr_tile_cache_range(input0, WAFER_DAR_REQUEST_WORDS * 8U, 1U);
  const volatile uint64_t *request =
      (const volatile uint64_t *)(uintptr_t)input0;
  uint32_t status = wafer_dar_decode(rank, request);
  uint32_t active =
      (uint32_t)((request[WAFER_DAR_REQ_ACTIVE_MASK] >> rank) & 1U);
  uint32_t sample = (uint32_t)request[WAFER_DAR_REQ_SAMPLE];
  uint32_t payload_bytes = (uint32_t)request[WAFER_DAR_REQ_PAYLOAD_BYTES];
  uint32_t inner_bytes = (uint32_t)request[WAFER_DAR_REQ_INNER_BYTES];
  uint32_t stride0 = (uint32_t)request[WAFER_DAR_REQ_STRIDE0];
  uint32_t iteration0 = (uint32_t)request[WAFER_DAR_REQ_ITERATION0];
  uint32_t envelope_bytes = inner_bytes + stride0 * (iteration0 - 1U);
  uint32_t direction = (uint32_t)request[WAFER_DAR_REQ_DIRECTION];
  uint32_t span = envelope_bytes + 2U * WAFER_DAR_GUARD_BYTES;

  wafer_dar_write_inactive_canary(output1, rank, sample);
  if (status == WAFER_DAR_STATUS_OK && active != 0U) {
    wafer_dar_copy_chunks(WAFER_DAR_DIRECTION_RDMA,
                          input1 + WAFER_DAR_SEED_RDMA_OFFSET,
                          WAFER_DAR_SPM_RDMA - WAFER_DAR_GUARD_BYTES, span);
    if (direction != WAFER_DAR_DIRECTION_RDMA)
      wafer_dar_copy_chunks(WAFER_DAR_DIRECTION_RDMA,
                            input1 + WAFER_DAR_SEED_WDMA_OFFSET,
                            WAFER_DAR_SPM_WDMA - WAFER_DAR_GUARD_BYTES, span);
    if (direction != WAFER_DAR_DIRECTION_RDMA)
      wafer_dar_copy_chunks(WAFER_DAR_DIRECTION_WDMA,
                            WAFER_DAR_SPM_RDMA - WAFER_DAR_GUARD_BYTES,
                            output0 + WAFER_DAR_WDMA_TARGET_OFFSET, span);
    wafer_tx81_local_fence();
  }

  hrt_barrier();
  uint32_t before_stable = 0U;
  uint32_t after_stable = 0U;
  WaferDDRTilePMU before = wafer_dar_read_pmu(&before_stable);
  uint64_t begin = wafer_ddr_tile_cycle();
  if (status == WAFER_DAR_STATUS_OK && active != 0U) {
    uint32_t issue_order = (uint32_t)request[WAFER_DAR_REQ_ISSUE_ORDER];
    for (uint32_t position = 0; position < 2U; ++position) {
      uint32_t selected = position ^ issue_order;
      if (selected == 0U && direction != WAFER_DAR_DIRECTION_WDMA)
        wafer_dar_target_dma(WAFER_DAR_DIRECTION_RDMA,
                             input0 + WAFER_DAR_SOURCE_OFFSET +
                                 WAFER_DAR_GUARD_BYTES,
                             WAFER_DAR_SPM_RDMA, payload_bytes, inner_bytes,
                             stride0, iteration0);
      if (selected == 1U && direction != WAFER_DAR_DIRECTION_RDMA)
        wafer_dar_target_dma(WAFER_DAR_DIRECTION_WDMA, WAFER_DAR_SPM_WDMA,
                             output0 + WAFER_DAR_WDMA_TARGET_OFFSET +
                                 WAFER_DAR_GUARD_BYTES,
                             payload_bytes, inner_bytes, stride0, iteration0);
    }
    wafer_tx81_local_fence();
  }
  uint64_t completion_cycles = wafer_ddr_tile_cycle() - begin;
  WaferDDRTilePMU after = wafer_dar_read_pmu(&after_stable);
  hrt_barrier();

  uint32_t expected_rdma =
      active != 0U && direction != WAFER_DAR_DIRECTION_WDMA ? 1U : 0U;
  uint32_t expected_wdma =
      active != 0U && direction != WAFER_DAR_DIRECTION_RDMA ? 1U : 0U;
  if (status == WAFER_DAR_STATUS_OK &&
      (before_stable == 0U || after_stable == 0U))
    status = WAFER_DAR_STATUS_PMU_UNSTABLE;
  if (status == WAFER_DAR_STATUS_OK &&
      (after.rdma_instructions - before.rdma_instructions != expected_rdma ||
       after.wdma_instructions - before.wdma_instructions != expected_wdma))
    status = WAFER_DAR_STATUS_TARGET_COUNT_MISMATCH;

  if (status == WAFER_DAR_STATUS_OK && active != 0U &&
      direction != WAFER_DAR_DIRECTION_WDMA) {
    wafer_dar_copy_chunks(WAFER_DAR_DIRECTION_WDMA,
                          WAFER_DAR_SPM_RDMA - WAFER_DAR_GUARD_BYTES,
                          output0 + WAFER_DAR_RDMA_ARCHIVE_OFFSET, span);
    wafer_tx81_local_fence();
  }
  hrt_barrier();

  wafer_dar_write_record((volatile uint64_t *)(uintptr_t)output0, status, rank,
                         request, active, envelope_bytes, &before, &after,
                         before_stable, after_stable, completion_cycles, input0,
                         input1, output0, output1);
  wafer_ddr_tile_cache_range(output0, WAFER_DAR_RECORD_WORDS * sizeof(uint64_t),
                             0U);
  wafer_tx81_direct_dte_finish();
}
