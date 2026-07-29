#include "wafer_full_card_barrier_probe_protocol.h"
#include "wafer_tx81_crt.h"

#include <stdint.h>

extern void hrt_barrier(void);

__attribute__((visibility("hidden"))) const uint64_t
    wafer_barrier_slots_per_rank = WAFER_BARRIER_SLOTS_PER_RANK;
__attribute__((visibility("hidden"))) const uint64_t
    wafer_barrier_input_slot = WAFER_BARRIER_INPUT_SLOT;
__attribute__((visibility("hidden"))) const uint64_t
    wafer_barrier_output_slot = WAFER_BARRIER_OUTPUT_SLOT;
__attribute__((visibility("hidden"))) const uint64_t
    wafer_barrier_status_slot = WAFER_BARRIER_STATUS_SLOT;

static void wafer_barrier_cache_range(uint64_t begin, uint32_t bytes,
                                      uint32_t invalidate_only) {
  enum {
    WAFER_TX81_SUPERVISOR_MODE = 1,
    WAFER_TX81_MACHINE_MODE = 3,
  };
  uintptr_t mode;
  __asm__ volatile("fence" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
  __asm__ volatile("csrr %0, mxstatus" : "=r"(mode));
  mode = (mode >> 30) & 3U;
  for (uintptr_t address = (uintptr_t)begin;
       address < (uintptr_t)begin + bytes;
       address += WAFER_BARRIER_CACHE_LINE_BYTES) {
    if (mode == WAFER_TX81_MACHINE_MODE) {
      if (invalidate_only != 0)
        __asm__ volatile("dcache.ipa %0" : : "r"(address) : "memory");
      else
        __asm__ volatile("dcache.cipa %0" : : "r"(address) : "memory");
    } else if (mode == WAFER_TX81_SUPERVISOR_MODE) {
      if (invalidate_only != 0)
        __asm__ volatile("dcache.iva %0" : : "r"(address) : "memory");
      else
        __asm__ volatile("dcache.civa %0" : : "r"(address) : "memory");
    }
  }
  __asm__ volatile("sync.is" ::: "memory");
  __asm__ volatile("fence" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
}

static uint64_t wafer_barrier_cycle(void) {
  uint64_t cycle;
  __asm__ volatile("rdcycle %0" : "=r"(cycle));
  return cycle;
}

static void wafer_barrier_delay(uint64_t iterations) {
  for (volatile uint64_t index = 0; index < iterations; ++index)
    __asm__ volatile("" ::: "memory");
}

static uint64_t wafer_barrier_epoch1_marker(uint32_t rank) {
  return WAFER_BARRIER_EPOCH1_BASE | rank;
}

static uint64_t wafer_barrier_epoch2_marker(uint32_t rank) {
  return WAFER_BARRIER_EPOCH2_BASE | rank;
}

static uint64_t wafer_barrier_metadata(uint32_t rank, uint64_t status) {
  return WAFER_BARRIER_RECORD_SCHEMA | ((uint64_t)rank << 16) |
         (status << 24) | ((uint64_t)WAFER_BARRIER_RANKS << 32);
}

static uint64_t wafer_barrier_verify_epoch(
    const volatile uint64_t *rank_major_slots, uint32_t slot_offset,
    uint64_t marker_base) {
  uint64_t mismatches = 0;
  for (uint32_t peer = 0; peer < WAFER_BARRIER_RANKS; ++peer) {
    uint64_t peer_output =
        rank_major_slots[peer * wafer_barrier_slots_per_rank +
                         wafer_barrier_output_slot];
    wafer_barrier_cache_range(peer_output + slot_offset,
                              WAFER_BARRIER_CACHE_LINE_BYTES, 1);
    const volatile uint64_t *marker =
        (const volatile uint64_t *)(uintptr_t)(peer_output + slot_offset);
    mismatches += *marker != (marker_base | peer);
  }
  return mismatches;
}

__attribute__((visibility("hidden"))) void wafer_tx81_full_card_barrier_probe(
    uint32_t rank, uint64_t input_ddr, uint64_t output_ddr,
    uint64_t status_ddr, uint64_t rank_major_slots_address) {
  const volatile uint64_t *rank_major_slots =
      (const volatile uint64_t *)(uintptr_t)rank_major_slots_address;
  const volatile uint64_t *request =
      (const volatile uint64_t *)(uintptr_t)input_ddr;
  volatile uint8_t *output =
      (volatile uint8_t *)(uintptr_t)output_ddr;

  wafer_tx81_direct_dte_begin_after_prepare(status_ddr,
                                            WAFER_BARRIER_RANKS);
  wafer_barrier_cache_range(input_ddr, WAFER_BARRIER_CACHE_LINE_BYTES, 1);
  uint32_t request_valid =
      request[WAFER_BARRIER_REQ_MAGIC] == WAFER_BARRIER_REQUEST_MAGIC &&
      request[WAFER_BARRIER_REQ_SCHEMA] == WAFER_BARRIER_REQUEST_SCHEMA &&
      request[WAFER_BARRIER_REQ_RANK] == rank &&
      request[WAFER_BARRIER_REQ_GUARD] == WAFER_BARRIER_REQUEST_GUARD;

  for (uint32_t index = 0; index < WAFER_BARRIER_RESOURCE_BYTES; ++index)
    output[index] = WAFER_BARRIER_OUTPUT_CANARY;
  uint64_t step_flags = WAFER_BARRIER_STEP_OUTPUT_INITIALIZED;

  uint64_t delay1 = ((uint64_t)rank + 1) * WAFER_BARRIER_DELAY1_SCALE;
  wafer_barrier_delay(delay1);
  volatile uint64_t *epoch1 =
      (volatile uint64_t *)(uintptr_t)(
          output_ddr + WAFER_BARRIER_EPOCH1_OFFSET);
  *epoch1 = wafer_barrier_epoch1_marker(rank);
  wafer_barrier_cache_range(output_ddr + WAFER_BARRIER_EPOCH1_OFFSET,
                            WAFER_BARRIER_CACHE_LINE_BYTES, 0);
  step_flags |= WAFER_BARRIER_STEP_EPOCH1_PUBLISHED;
  uint64_t begin1 = wafer_barrier_cycle();
  hrt_barrier();
  uint64_t duration1 = wafer_barrier_cycle() - begin1;
  step_flags |= WAFER_BARRIER_STEP_EPOCH1_COMPLETED;

  uint64_t epoch1_mismatches = wafer_barrier_verify_epoch(
      rank_major_slots, WAFER_BARRIER_EPOCH1_OFFSET,
      WAFER_BARRIER_EPOCH1_BASE);
  if (epoch1_mismatches == 0)
    step_flags |= WAFER_BARRIER_STEP_EPOCH1_VISIBLE;

  uint64_t delay2 =
      ((uint64_t)WAFER_BARRIER_RANKS - rank) *
      WAFER_BARRIER_DELAY2_SCALE;
  wafer_barrier_delay(delay2);
  volatile uint64_t *epoch2 =
      (volatile uint64_t *)(uintptr_t)(
          output_ddr + WAFER_BARRIER_EPOCH2_OFFSET);
  *epoch2 = wafer_barrier_epoch2_marker(rank);
  wafer_barrier_cache_range(output_ddr + WAFER_BARRIER_EPOCH2_OFFSET,
                            WAFER_BARRIER_CACHE_LINE_BYTES, 0);
  step_flags |= WAFER_BARRIER_STEP_EPOCH2_PUBLISHED;
  uint64_t begin2 = wafer_barrier_cycle();
  hrt_barrier();
  uint64_t duration2 = wafer_barrier_cycle() - begin2;
  step_flags |= WAFER_BARRIER_STEP_EPOCH2_COMPLETED;

  uint64_t epoch2_mismatches = wafer_barrier_verify_epoch(
      rank_major_slots, WAFER_BARRIER_EPOCH2_OFFSET,
      WAFER_BARRIER_EPOCH2_BASE);
  if (epoch2_mismatches == 0)
    step_flags |= WAFER_BARRIER_STEP_EPOCH2_VISIBLE;
  uint64_t epoch1_crosstalk = wafer_barrier_verify_epoch(
      rank_major_slots, WAFER_BARRIER_EPOCH1_OFFSET,
      WAFER_BARRIER_EPOCH1_BASE);
  if (epoch1_crosstalk == 0)
    step_flags |= WAFER_BARRIER_STEP_EPOCH1_STABLE;

  uint64_t status =
      request_valid == 0
          ? WAFER_BARRIER_STATUS_INVALID_REQUEST
          : epoch1_mismatches == 0 && epoch2_mismatches == 0 &&
                    epoch1_crosstalk == 0
                ? WAFER_BARRIER_STATUS_OK
                : WAFER_BARRIER_STATUS_ORACLE_FAILED;
  volatile uint64_t *record =
      (volatile uint64_t *)(uintptr_t)(
          output_ddr + WAFER_BARRIER_RECORD_OFFSET);
  record[WAFER_BARRIER_REC_MAGIC] = WAFER_BARRIER_RECORD_MAGIC;
  record[WAFER_BARRIER_REC_METADATA] =
      wafer_barrier_metadata(rank, status);
  record[WAFER_BARRIER_REC_EPOCH1_MISMATCHES] = epoch1_mismatches;
  record[WAFER_BARRIER_REC_EPOCH2_MISMATCHES] = epoch2_mismatches;
  record[WAFER_BARRIER_REC_EPOCH1_CROSSTALK] = epoch1_crosstalk;
  record[WAFER_BARRIER_REC_EPOCH1_MARKER] =
      wafer_barrier_epoch1_marker(rank);
  record[WAFER_BARRIER_REC_EPOCH2_MARKER] =
      wafer_barrier_epoch2_marker(rank);
  record[WAFER_BARRIER_REC_DELAY1] = delay1;
  record[WAFER_BARRIER_REC_DELAY2] = delay2;
  record[WAFER_BARRIER_REC_DURATION1] = duration1;
  record[WAFER_BARRIER_REC_DURATION2] = duration2;
  record[WAFER_BARRIER_REC_CALLS] = 2;
  record[WAFER_BARRIER_REC_STEP_FLAGS] = step_flags;
  record[WAFER_BARRIER_REC_REQUEST_GUARD] =
      request[WAFER_BARRIER_REQ_GUARD];
  record[WAFER_BARRIER_REC_PARTICIPANTS] = WAFER_BARRIER_RANKS;
  record[WAFER_BARRIER_REC_GUARD] = WAFER_BARRIER_RECORD_GUARD;
  wafer_barrier_cache_range(output_ddr, WAFER_BARRIER_RESOURCE_BYTES, 0);
  wafer_tx81_direct_dte_finish();
}
