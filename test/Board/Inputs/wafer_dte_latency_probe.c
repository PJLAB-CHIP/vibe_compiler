#include "instr_def.h"
#include "wafer_tx81_crt.h"
#include <stdint.h>

/* Test-only record: 256 u64 words per Tile, followed by two guarded SPM
 * captures. The seed's 32768-byte output slice contains all three ranges. */
#define MAGIC UINT64_C(0x4454454c4154454e)
#define GUARD UINT64_C(0xa5a5a5a5a5a5a5a5)
#define RECORD_BYTES 2048U
#define GUARD_BYTES 256U
#define SOURCE UINT64_C(0x10000)
#define RECEIVE UINT64_C(0x20000)

static uint64_t cycle(void) {
  uint64_t value;
  __asm__ volatile("rdcycle %0" : "=r"(value)::"memory");
  return value;
}

/* Same CLINT MTIME address and split read as the pinned C908 core timer.
 * Its frequency is measured against StreamEvents, never assumed here. */
static uint64_t timer(void) {
  const volatile uint32_t *value =
      (const volatile uint32_t *)UINT64_C(0x1400bff8);
  for (unsigned retry = 0; retry < 8; ++retry) {
    uint32_t high = value[1], low = value[0];
    if (high == value[1])
      return ((uint64_t)high << 32) | low;
  }
  return 0;
}

static uint64_t measure_cycles(uint64_t target) {
  uint64_t begin = cycle(), end = begin;
  for (uint64_t i = 0; i < UINT64_C(200000000); ++i) {
    end = cycle();
    if (end - begin >= target)
      break;
  }
  return end - begin;
}

static void flush(uint64_t base) {
  uintptr_t mode;
  __asm__ volatile("fence\nsync\ncsrr %0, mxstatus" : "=r"(mode)::"memory");
  for (uintptr_t p = base; p < base + RECORD_BYTES; p += 64) {
    if (((mode >> 30) & 3U) == 3U)
      __asm__ volatile("dcache.cipa %0" ::"r"(p) : "memory");
    else
      __asm__ volatile("dcache.civa %0" ::"r"(p) : "memory");
  }
  __asm__ volatile("sync.is\nfence\nsync" ::: "memory");
}

void wafer_dte_latency_probe(uint32_t tile, uint64_t input, uint64_t output,
                             uint64_t status) {
  volatile uint64_t *record = (volatile uint64_t *)(uintptr_t)output;
  for (unsigned i = 0; i < RECORD_BYTES / 8; ++i)
    record[i] = GUARD;
  record[0] = MAGIC;
  record[1] = tile;
  record[2] = WAFER_LATENCY_CLOCK_CYCLES;
  record[5] = WAFER_LATENCY_CLOCK_CYCLES ? 0 : 32;
  record[6] = WAFER_LATENCY_PAYLOAD_BYTES;
  if (WAFER_LATENCY_NCC_SYNC)
    record[8] = 1;
  if (WAFER_LATENCY_NCC_PARTICIPANTS > 1)
    record[9] = WAFER_LATENCY_NCC_PARTICIPANTS;
  uint64_t empty = UINT64_MAX;
  for (unsigned i = 0; i < 32; ++i) {
    uint64_t begin = cycle(), end = cycle();
    if (end - begin < empty)
      empty = end - begin;
  }
  record[7] = empty;
  wafer_tx81_direct_dte_begin_after_prepare(status, 16);
  if (WAFER_LATENCY_CLOCK_CYCLES) {
    uint64_t t0 = timer();
    record[3] = measure_cycles(WAFER_LATENCY_CLOCK_CYCLES);
    record[4] = timer() - t0;
  } else {
    const uint32_t bytes = WAFER_LATENCY_PAYLOAD_BYTES;
    const uint32_t slot = bytes + 2 * GUARD_BYTES;
    wafer_tx81_rdma(input, SOURCE, slot, slot, 0, 0, 0, 1, 1, 1, Fmt_FP16, 0);
    wafer_tx81_rdma(input, RECEIVE, slot, slot, 0, 0, 0, 1, 1, 1, Fmt_FP16, 0);
    wafer_tx81_ncc_join(1);
    if (WAFER_LATENCY_NCC_PARTICIPANTS > 1) {
      for (unsigned worker = 0; worker < 3; ++worker)
        wafer_tx81_wdma(SOURCE, output + RECORD_BYTES + worker * slot, slot,
                        slot, 0, 0, 0, 1, 1, 1, Fmt_FP16, 0);
      wafer_tx81_ncc_join(1);
    }
    /* Same receiver-first, issue/send-wait/receive-wait order as the existing
     * qualified DTE probe. Both waits precede next-iteration buffer reuse. */
    for (unsigned sample = 0; sample < 32; ++sample) {
      if (WAFER_LATENCY_NCC_PARTICIPANTS > 1) {
        const unsigned participants = WAFER_LATENCY_NCC_PARTICIPANTS;
        uint64_t stamp[5], durations[5];
        stamp[0] = cycle();
        wafer_tx81_ncc_join((1U << participants) - 1);
        stamp[1] = cycle();
        for (unsigned worker = 0; worker < participants; ++worker)
          wafer_tx81_ncc_join(1U << worker);
        stamp[2] = cycle();
        /* Shared read-only source, disjoint guarded destinations. */
        for (unsigned worker = 0; worker < participants; ++worker)
          wafer_tx81_wdma(SOURCE + GUARD_BYTES,
                          output + RECORD_BYTES + worker * slot + GUARD_BYTES,
                          bytes, bytes, 0, 0, 0, 1, 1, 1, Fmt_FP16, worker);
        stamp[3] = cycle();
        wafer_tx81_ncc_join((1U << participants) - 1);
        stamp[4] = cycle();
        for (unsigned phase = 0; phase < 4; ++phase)
          durations[phase] = stamp[phase + 1] - stamp[phase];
        for (unsigned worker = 0; worker < participants; ++worker)
          wafer_tx81_wdma(SOURCE + GUARD_BYTES,
                          output + RECORD_BYTES + worker * slot + GUARD_BYTES,
                          bytes, bytes, 0, 0, 0, 1, 1, 1, Fmt_FP16, worker);
        uint64_t begin = cycle();
        for (unsigned worker = 0; worker < participants; ++worker)
          wafer_tx81_ncc_join(1U << worker);
        durations[4] = cycle() - begin;
        for (unsigned phase = 0; phase < 5; ++phase)
          record[16 + sample * 5 + phase] = durations[phase];
        continue;
      }
      uint64_t stamp[6];
      stamp[0] = cycle();
      if (WAFER_LATENCY_NCC_SYNC) {
        /* Worker 0 is idle on entry. Each DMA's matching join completes it
         * before the next issue or buffer reuse; no worker overlap is assumed.
         */
        wafer_tx81_ncc_join(1);
        stamp[1] = cycle();
        wafer_tx81_rdma(input + GUARD_BYTES, SOURCE + GUARD_BYTES, bytes, bytes,
                        0, 0, 0, 1, 1, 1, Fmt_FP16, 0);
        stamp[2] = cycle();
        wafer_tx81_ncc_join(1);
        stamp[3] = cycle();
        wafer_tx81_wdma(SOURCE + GUARD_BYTES,
                        output + RECORD_BYTES + GUARD_BYTES, bytes, bytes, 0, 0,
                        0, 1, 1, 1, Fmt_FP16, 0);
        stamp[4] = cycle();
        wafer_tx81_ncc_join(1);
        stamp[5] = cycle();
      } else {
        uint64_t receive = wafer_tx81_direct_dte_recv_prepare(
            RECEIVE + GUARD_BYTES, bytes, tile, (tile + 15) % 16, 0);
        stamp[1] = cycle();
        uint64_t send = wafer_tx81_direct_dte_send_prepare(
            SOURCE + GUARD_BYTES, RECEIVE + GUARD_BYTES, bytes, tile,
            (tile + 1) % 16, 0, 0);
        stamp[2] = cycle();
        wafer_tx81_direct_dte_send_issue(send);
        stamp[3] = cycle();
        wafer_tx81_direct_dte_wait(send);
        stamp[4] = cycle();
        wafer_tx81_direct_dte_wait(receive);
        stamp[5] = cycle();
      }
      for (unsigned phase = 0; phase < 5; ++phase)
        record[16 + sample * 5 + phase] = stamp[phase + 1] - stamp[phase];
    }
    if (WAFER_LATENCY_NCC_PARTICIPANTS > 1) {
      /* Capture source guards separately, preserving every destination guard.
       */
      wafer_tx81_wdma(SOURCE, output + RECORD_BYTES + 3 * slot, GUARD_BYTES,
                      GUARD_BYTES, 0, 0, 0, 1, 1, 1, Fmt_FP16, 0);
      wafer_tx81_wdma(SOURCE + GUARD_BYTES + bytes,
                      output + RECORD_BYTES + 3 * slot + GUARD_BYTES,
                      GUARD_BYTES, GUARD_BYTES, 0, 0, 0, 1, 1, 1, Fmt_FP16, 0);
    } else {
      wafer_tx81_wdma(SOURCE, output + RECORD_BYTES, slot, slot, 0, 0, 0, 1, 1,
                      1, Fmt_FP16, 0);
      wafer_tx81_wdma(RECEIVE, output + RECORD_BYTES + slot, slot, slot, 0, 0,
                      0, 1, 1, 1, Fmt_FP16, 0);
    }
    wafer_tx81_ncc_join(1);
  }
  wafer_tx81_direct_dte_finish();
  flush(output);
}
