#include "Wafer/ABI/Tx81DirectDTEStatusABI.h"
#include "instr_def.h"
#include "pmu/pmu_reg.h"
#include "wafer_tx81_crt.h"

#include <stddef.h>
#include <stdint.h>

extern int8_t *get_spm_memory_mapping(uint64_t offset);

#define WAFER_PROBE_RANK_COUNT UINT32_C(16)
#define WAFER_PROBE_HEADER_BYTES UINT32_C(128)
#define WAFER_PROBE_INPUT_BYTES UINT32_C(128)
#define WAFER_PROBE_MAX_PAYLOAD_BYTES UINT32_C(64)
#define WAFER_PROBE_DISJOINT_BYTES UINT32_C(32)
#define WAFER_PROBE_OUTPUT_GUARD_BYTES UINT32_C(32)
#define WAFER_PROBE_SPM_GUARD_BYTES UINT32_C(256)
#define WAFER_PROBE_DISJOINT_ELEMENTS                                          \
  (WAFER_PROBE_DISJOINT_BYTES / (uint32_t)sizeof(uint16_t))

#define WAFER_PROBE_SPM_INPUT (UINT64_C(0x10000) + WAFER_PROBE_SPM_GUARD_BYTES)
#define WAFER_PROBE_SPM_PRODUCED                                               \
  (UINT64_C(0x20000) + WAFER_PROBE_SPM_GUARD_BYTES)
#define WAFER_PROBE_SPM_DTE_RECV                                               \
  (UINT64_C(0x40000) + WAFER_PROBE_SPM_GUARD_BYTES)
#define WAFER_PROBE_SPM_DTE_RECV_SECOND                                        \
  (UINT64_C(0x50000) + WAFER_PROBE_SPM_GUARD_BYTES)
#define WAFER_PROBE_SPM_DISJOINT_INPUT                                         \
  (UINT64_C(0x60000) + WAFER_PROBE_SPM_GUARD_BYTES)
#define WAFER_PROBE_SPM_DISJOINT_OUTPUT                                        \
  (UINT64_C(0x70000) + WAFER_PROBE_SPM_GUARD_BYTES)

#define WAFER_PROBE_PMU_BASE UINT64_C(0x590000)
#define WAFER_PROBE_MAGIC UINT64_C(0x3143434e45544457)
#define WAFER_PROBE_CANARY UINT64_C(0xd7e0ca11d7e0ca11)
#define WAFER_PROBE_SCHEMA UINT64_C(6)
#define WAFER_PROBE_STABLE_RETRIES UINT32_C(8)
#define WAFER_PROBE_TRANSPORT_COUNTER_COUNT UINT32_C(6)
#define WAFER_PROBE_TRANSPORT_STABLE_MASK                                      \
  ((UINT32_C(1) << WAFER_PROBE_TRANSPORT_COUNTER_COUNT) - UINT32_C(1))

#define WAFER_PROBE_INPUT_GUARD_BEFORE UINT8_C(0x31)
#define WAFER_PROBE_INPUT_GUARD_AFTER UINT8_C(0x32)
#define WAFER_PROBE_PRODUCED_GUARD_BEFORE UINT8_C(0x41)
#define WAFER_PROBE_PRODUCED_GUARD_AFTER UINT8_C(0x42)
#define WAFER_PROBE_RECV_GUARD_BEFORE UINT8_C(0x51)
#define WAFER_PROBE_RECV_GUARD_AFTER UINT8_C(0x52)
#define WAFER_PROBE_RECV_SECOND_GUARD_BEFORE UINT8_C(0x53)
#define WAFER_PROBE_RECV_SECOND_GUARD_AFTER UINT8_C(0x54)
#define WAFER_PROBE_DISJOINT_INPUT_GUARD_BEFORE UINT8_C(0x61)
#define WAFER_PROBE_DISJOINT_INPUT_GUARD_AFTER UINT8_C(0x62)
#define WAFER_PROBE_DISJOINT_OUTPUT_GUARD_BEFORE UINT8_C(0x71)
#define WAFER_PROBE_DISJOINT_OUTPUT_GUARD_AFTER UINT8_C(0x72)
#define WAFER_PROBE_SPM_PAYLOAD_POISON UINT8_C(0xc3)

#if WAFER_PROBE_HEADER_BYTES + 2U * WAFER_PROBE_OUTPUT_GUARD_BYTES +           \
        WAFER_PROBE_MAX_PAYLOAD_BYTES !=                                       \
    256U
#error "DTE/NCC probe record must fit the existing 256-byte output resource"
#endif

#if WAFER_PROBE_HEADER_BYTES + WAFER_PROBE_INPUT_BYTES > 256U
#error "DTE/NCC input data must fit the existing 256-byte input resource"
#endif

#if WAFER_PROBE_HEADER_BYTES %                                                 \
        WAFER_TX81_DIRECT_DTE_STATUS_V2_CACHE_LINE_BYTES !=                    \
    0U
#error "DTE/NCC probe header must own complete cache lines"
#endif

#if WAFER_PROBE_MAX_PAYLOAD_BYTES % 2U != 0U ||                                \
    WAFER_PROBE_DISJOINT_BYTES % 2U != 0U
#error "DTE/NCC f16 payloads must contain complete elements"
#endif

_Static_assert(sizeof(uint16_t) == 2U,
               "DTE/NCC f16 oracle requires 16-bit storage");

enum WaferDteNccProbeMode {
  WAFER_PROBE_NCC_PRODUCER_DTE = 1,
  WAFER_PROBE_DTE_NCC_CONSUMER = 2,
  WAFER_PROBE_DISJOINT_LOCAL_WAIT_FIRST = 3,
  WAFER_PROBE_DISJOINT_DTE_WAIT_FIRST = 4,
  WAFER_PROBE_DTE_REUSE_AFTER_EVENTS = 5,
  WAFER_PROBE_DTE_TWO_DESTINATION_BROADCAST = 6,
  WAFER_PROBE_DTE_REUSE_BEFORE_SEND_EVENT_ERROR = 7,
  WAFER_PROBE_DTE_INVALID_FSM_ERROR = 8,
  WAFER_PROBE_DTE_WAIT_UNKNOWN_EVENT_ERROR = 9,
};

enum WaferDteContractEvidence {
  WAFER_PROBE_CONTRACT_STATUS_ERROR = UINT32_C(1) << 0,
  WAFER_PROBE_CONTRACT_VALID_SEND_EVENT = UINT32_C(1) << 1,
  WAFER_PROBE_CONTRACT_VALID_RECV_EVENT = UINT32_C(1) << 2,
  WAFER_PROBE_CONTRACT_REJECTED_SEND_EVENT = UINT32_C(1) << 3,
  WAFER_PROBE_CONTRACT_REJECTED_RECV_EVENT = UINT32_C(1) << 4,
  WAFER_PROBE_CONTRACT_UNKNOWN_WAIT_RETURNED = UINT32_C(1) << 5,
};

typedef struct {
  uint64_t full;
  uint64_t ct;
  uint64_t rdma;
  uint64_t wdma;
  uint32_t ct_count;
  uint32_t rdma_count;
  uint32_t wdma_count;
  uint32_t stable_mask;
} WaferProbePmu;

typedef struct {
  uint64_t dte_channel0_transfer;
  uint64_t dte_channel1_transfer;
  uint64_t dte_channel0_execution;
  uint64_t dte_channel1_execution;
  uint64_t spm_dte_t2_port8;
  uint64_t spm_dte_t3_port8;
  uint32_t dte_enable;
  uint32_t spm_enable;
  uint32_t stable_mask;
} WaferProbeTransportPmu;

typedef struct {
  uint32_t source_guard_mismatches;
  uint32_t receive_guard_mismatches;
  uint32_t compute_guard_mismatches;
  uint32_t compute_result_mismatches;
} WaferProbeOracle;

static volatile uint8_t *wafer_probe_spm8(uint64_t offset) {
  return (volatile uint8_t *)(void *)get_spm_memory_mapping(offset);
}

static volatile uint16_t *wafer_probe_spm16(uint64_t offset) {
  return (volatile uint16_t *)(void *)get_spm_memory_mapping(offset);
}

static void wafer_probe_fill8(volatile uint8_t *destination, uint8_t value,
                              uint32_t bytes) {
  for (uint32_t index = 0; index < bytes; ++index)
    destination[index] = value;
}

static uint32_t wafer_probe_mismatch8(const volatile uint8_t *actual,
                                      uint8_t expected, uint32_t bytes) {
  uint32_t mismatches = 0;
  for (uint32_t index = 0; index < bytes; ++index)
    mismatches += actual[index] != expected;
  return mismatches;
}

static void wafer_probe_seed_guarded_region(uint64_t payload, uint32_t bytes,
                                            uint8_t guard_before,
                                            uint8_t guard_after) {
  wafer_probe_fill8(wafer_probe_spm8(payload - WAFER_PROBE_SPM_GUARD_BYTES),
                    guard_before, WAFER_PROBE_SPM_GUARD_BYTES);
  wafer_probe_fill8(wafer_probe_spm8(payload), WAFER_PROBE_SPM_PAYLOAD_POISON,
                    bytes);
  wafer_probe_fill8(wafer_probe_spm8(payload + bytes), guard_after,
                    WAFER_PROBE_SPM_GUARD_BYTES);
}

static uint32_t wafer_probe_guard_mismatches(uint64_t payload, uint32_t bytes,
                                             uint8_t guard_before,
                                             uint8_t guard_after) {
  return wafer_probe_mismatch8(
             wafer_probe_spm8(payload - WAFER_PROBE_SPM_GUARD_BYTES),
             guard_before, WAFER_PROBE_SPM_GUARD_BYTES) +
         wafer_probe_mismatch8(wafer_probe_spm8(payload + bytes), guard_after,
                               WAFER_PROBE_SPM_GUARD_BYTES);
}

static uint16_t wafer_probe_positive_integer_f16(uint32_t value) {
  if (value == 0 || value > UINT32_C(1024))
    return value == 0 ? UINT16_C(0) : UINT16_MAX;
  uint32_t exponent = 0;
  while ((UINT32_C(1) << (exponent + 1U)) <= value)
    ++exponent;
  uint32_t leading = UINT32_C(1) << exponent;
  uint32_t mantissa = (value - leading) << (10U - exponent);
  return (uint16_t)(((exponent + 15U) << 10U) | mantissa);
}

static uint32_t wafer_probe_compute_mismatches(uint64_t payload,
                                               uint32_t source_rank,
                                               uint32_t first_lane,
                                               uint32_t elements) {
  const volatile uint16_t *actual = wafer_probe_spm16(payload);
  uint32_t mismatches = 0;
  for (uint32_t index = 0; index < elements; ++index) {
    uint32_t input =
        source_rank * UINT32_C(4) + first_lane + index + UINT32_C(1);
    uint16_t expected = wafer_probe_positive_integer_f16(input * UINT32_C(2));
    mismatches += actual[index] != expected;
  }
  return mismatches;
}

static uint32_t wafer_probe_copy_mismatches(uint64_t payload,
                                            uint32_t source_rank,
                                            uint32_t first_lane,
                                            uint32_t elements) {
  const volatile uint16_t *actual = wafer_probe_spm16(payload);
  uint32_t mismatches = 0;
  for (uint32_t index = 0; index < elements; ++index) {
    uint32_t input =
        source_rank * UINT32_C(4) + first_lane + index + UINT32_C(1);
    uint16_t expected = wafer_probe_positive_integer_f16(input);
    mismatches += actual[index] != expected;
  }
  return mismatches;
}

static uint32_t wafer_probe_u8_saturated(uint32_t value) {
  return value > UINT8_MAX ? UINT8_MAX : value;
}

static int wafer_probe_payload_bytes_are_valid(uint32_t payload_bytes) {
  return payload_bytes == UINT32_C(16) || payload_bytes == UINT32_C(32) ||
         payload_bytes == WAFER_PROBE_MAX_PAYLOAD_BYTES;
}

static uint32_t wafer_probe_mmio_read32(uint64_t base, uint32_t offset) {
  const volatile uint32_t *address =
      (const volatile uint32_t *)(uintptr_t)(base + offset);
  return *address;
}

static uint64_t wafer_probe_mmio_read64(uint64_t base, uint32_t low_offset,
                                        uint32_t stable_bit,
                                        uint32_t *stable_mask) {
  uint32_t low = 0;
  uint32_t high_after = 0;
  for (uint32_t retry = 0; retry < WAFER_PROBE_STABLE_RETRIES; ++retry) {
    uint32_t high_before = wafer_probe_mmio_read32(base, low_offset + 4);
    low = wafer_probe_mmio_read32(base, low_offset);
    high_after = wafer_probe_mmio_read32(base, low_offset + 4);
    if (high_before == high_after) {
      *stable_mask |= stable_bit;
      break;
    }
  }
  return ((uint64_t)high_after << 32) | low;
}

static WaferProbePmu wafer_probe_read_pmu(void) {
  WaferProbePmu sample = {0};
  sample.full =
      wafer_probe_mmio_read64(WAFER_PROBE_PMU_BASE, GR_PMU_FU_EXE_TIME,
                              UINT32_C(1) << 0, &sample.stable_mask);
  sample.ct = wafer_probe_mmio_read64(WAFER_PROBE_PMU_BASE, GR_PMU_CT_EXE_TIME,
                                      UINT32_C(1) << 1, &sample.stable_mask);
  sample.rdma =
      wafer_probe_mmio_read64(WAFER_PROBE_PMU_BASE, GR_PMU_RDMA_EXE_TIME,
                              UINT32_C(1) << 2, &sample.stable_mask);
  sample.wdma =
      wafer_probe_mmio_read64(WAFER_PROBE_PMU_BASE, GR_PMU_WDMA_EXE_TIME,
                              UINT32_C(1) << 3, &sample.stable_mask);
  sample.ct_count =
      wafer_probe_mmio_read32(WAFER_PROBE_PMU_BASE, GR_PMU_CT_INST_NUMS);
  sample.rdma_count =
      wafer_probe_mmio_read32(WAFER_PROBE_PMU_BASE, GR_PMU_RDMA_INST_NUMS);
  sample.wdma_count =
      wafer_probe_mmio_read32(WAFER_PROBE_PMU_BASE, GR_PMU_WDMA_INST_NUMS);
  return sample;
}

static WaferProbeTransportPmu wafer_probe_read_transport_pmu(void) {
  WaferProbeTransportPmu sample = {0};
  __asm__ volatile("fence iorw, iorw" ::: "memory");
  sample.dte_enable = wafer_probe_mmio_read32(SCT_REG_BASE_DTE, SCT_DTE_PMU_EN);
  sample.spm_enable = wafer_probe_mmio_read32(SCT_REG_BASE_SPM, SPM1_PMU_EN);
  sample.dte_channel0_transfer =
      wafer_probe_mmio_read64(SCT_REG_BASE_DTE, SCT_DTE_CH0_TRANS_DATA_L,
                              UINT32_C(1) << 0, &sample.stable_mask);
  sample.dte_channel1_transfer =
      wafer_probe_mmio_read64(SCT_REG_BASE_DTE, SCT_DTE_CH1_TRANS_DATA_L,
                              UINT32_C(1) << 1, &sample.stable_mask);
  sample.dte_channel0_execution =
      wafer_probe_mmio_read64(SCT_REG_BASE_DTE, SCT_DTE_CH0_CLK_COUNTER_L,
                              UINT32_C(1) << 2, &sample.stable_mask);
  sample.dte_channel1_execution =
      wafer_probe_mmio_read64(SCT_REG_BASE_DTE, SCT_DTE_CH1_CLK_COUNTER_L,
                              UINT32_C(1) << 3, &sample.stable_mask);
  sample.spm_dte_t2_port8 =
      wafer_probe_mmio_read64(SCT_REG_BASE_SPM, SPM1_PMU_T2_8_31_0,
                              UINT32_C(1) << 4, &sample.stable_mask);
  sample.spm_dte_t3_port8 =
      wafer_probe_mmio_read64(SCT_REG_BASE_SPM, SPM1_PMU_T3_8_31_0,
                              UINT32_C(1) << 5, &sample.stable_mask);
  __asm__ volatile("fence iorw, iorw" ::: "memory");
  return sample;
}

static void wafer_probe_seed_regions(void) {
  wafer_probe_seed_guarded_region(
      WAFER_PROBE_SPM_INPUT, WAFER_PROBE_MAX_PAYLOAD_BYTES,
      WAFER_PROBE_INPUT_GUARD_BEFORE, WAFER_PROBE_INPUT_GUARD_AFTER);
  wafer_probe_seed_guarded_region(
      WAFER_PROBE_SPM_PRODUCED, WAFER_PROBE_MAX_PAYLOAD_BYTES,
      WAFER_PROBE_PRODUCED_GUARD_BEFORE, WAFER_PROBE_PRODUCED_GUARD_AFTER);
  wafer_probe_seed_guarded_region(
      WAFER_PROBE_SPM_DTE_RECV, WAFER_PROBE_MAX_PAYLOAD_BYTES,
      WAFER_PROBE_RECV_GUARD_BEFORE, WAFER_PROBE_RECV_GUARD_AFTER);
  wafer_probe_seed_guarded_region(
      WAFER_PROBE_SPM_DTE_RECV_SECOND, WAFER_PROBE_MAX_PAYLOAD_BYTES,
      WAFER_PROBE_RECV_SECOND_GUARD_BEFORE,
      WAFER_PROBE_RECV_SECOND_GUARD_AFTER);
  wafer_probe_seed_guarded_region(WAFER_PROBE_SPM_DISJOINT_INPUT,
                                  WAFER_PROBE_DISJOINT_BYTES,
                                  WAFER_PROBE_DISJOINT_INPUT_GUARD_BEFORE,
                                  WAFER_PROBE_DISJOINT_INPUT_GUARD_AFTER);
  wafer_probe_seed_guarded_region(WAFER_PROBE_SPM_DISJOINT_OUTPUT,
                                  WAFER_PROBE_DISJOINT_BYTES,
                                  WAFER_PROBE_DISJOINT_OUTPUT_GUARD_BEFORE,
                                  WAFER_PROBE_DISJOINT_OUTPUT_GUARD_AFTER);
  __asm__ volatile("fence iorw, iorw" ::: "memory");
}

static WaferProbeOracle wafer_probe_read_oracle(uint32_t rank, uint32_t mode,
                                                uint32_t payload_bytes) {
  WaferProbeOracle oracle = {0};
  uint32_t predecessor =
      (rank + WAFER_PROBE_RANK_COUNT - 1U) % WAFER_PROBE_RANK_COUNT;
  uint64_t source_payload = WAFER_PROBE_SPM_INPUT;
  uint8_t source_guard_before = WAFER_PROBE_INPUT_GUARD_BEFORE;
  uint8_t source_guard_after = WAFER_PROBE_INPUT_GUARD_AFTER;
  uint64_t compute_payload = WAFER_PROBE_SPM_PRODUCED;
  uint32_t compute_bytes = WAFER_PROBE_MAX_PAYLOAD_BYTES;
  uint8_t compute_guard_before = WAFER_PROBE_PRODUCED_GUARD_BEFORE;
  uint8_t compute_guard_after = WAFER_PROBE_PRODUCED_GUARD_AFTER;
  uint32_t compute_source_rank = rank;
  uint32_t compute_first_lane = 0;
  uint32_t compute_elements = payload_bytes / (uint32_t)sizeof(uint16_t);

  if (!wafer_probe_payload_bytes_are_valid(payload_bytes)) {
    oracle.source_guard_mismatches = UINT8_MAX;
    oracle.receive_guard_mismatches = UINT8_MAX;
    oracle.compute_guard_mismatches = UINT8_MAX;
    oracle.compute_result_mismatches = UINT8_MAX;
    return oracle;
  } else if (mode == WAFER_PROBE_NCC_PRODUCER_DTE) {
    source_payload = WAFER_PROBE_SPM_PRODUCED;
    source_guard_before = WAFER_PROBE_PRODUCED_GUARD_BEFORE;
    source_guard_after = WAFER_PROBE_PRODUCED_GUARD_AFTER;
  } else if (mode == WAFER_PROBE_DTE_NCC_CONSUMER) {
    compute_source_rank = predecessor;
  } else if (mode == WAFER_PROBE_DTE_REUSE_AFTER_EVENTS) {
    compute_payload = WAFER_PROBE_SPM_DTE_RECV;
    compute_bytes = WAFER_PROBE_MAX_PAYLOAD_BYTES;
    compute_guard_before = WAFER_PROBE_RECV_GUARD_BEFORE;
    compute_guard_after = WAFER_PROBE_RECV_GUARD_AFTER;
    compute_source_rank = predecessor;
    compute_first_lane =
        WAFER_PROBE_MAX_PAYLOAD_BYTES / (uint32_t)sizeof(uint16_t);
  } else if (mode == WAFER_PROBE_DTE_REUSE_BEFORE_SEND_EVENT_ERROR) {
    compute_payload = WAFER_PROBE_SPM_DTE_RECV;
    compute_bytes = WAFER_PROBE_MAX_PAYLOAD_BYTES;
    compute_guard_before = WAFER_PROBE_RECV_GUARD_BEFORE;
    compute_guard_after = WAFER_PROBE_RECV_GUARD_AFTER;
    compute_source_rank = predecessor;
  } else if (mode == WAFER_PROBE_DTE_TWO_DESTINATION_BROADCAST) {
    uint32_t second_predecessor =
        (rank + WAFER_PROBE_RANK_COUNT - 2U) % WAFER_PROBE_RANK_COUNT;
    oracle.compute_result_mismatches += wafer_probe_copy_mismatches(
        WAFER_PROBE_SPM_DTE_RECV, predecessor, 0, compute_elements);
    oracle.compute_result_mismatches += wafer_probe_copy_mismatches(
        WAFER_PROBE_SPM_DTE_RECV_SECOND, second_predecessor, 0,
        compute_elements);
    compute_payload = WAFER_PROBE_SPM_DTE_RECV;
    compute_bytes = WAFER_PROBE_MAX_PAYLOAD_BYTES;
    compute_guard_before = WAFER_PROBE_RECV_GUARD_BEFORE;
    compute_guard_after = WAFER_PROBE_RECV_GUARD_AFTER;
    compute_elements = 0;
  } else if (mode == WAFER_PROBE_DISJOINT_LOCAL_WAIT_FIRST ||
             mode == WAFER_PROBE_DISJOINT_DTE_WAIT_FIRST) {
    compute_payload = WAFER_PROBE_SPM_DISJOINT_OUTPUT;
    compute_bytes = WAFER_PROBE_DISJOINT_BYTES;
    compute_guard_before = WAFER_PROBE_DISJOINT_OUTPUT_GUARD_BEFORE;
    compute_guard_after = WAFER_PROBE_DISJOINT_OUTPUT_GUARD_AFTER;
    compute_first_lane =
        WAFER_PROBE_MAX_PAYLOAD_BYTES / (uint32_t)sizeof(uint16_t);
    compute_elements = WAFER_PROBE_DISJOINT_ELEMENTS;
  } else if (mode == WAFER_PROBE_DTE_INVALID_FSM_ERROR ||
             mode == WAFER_PROBE_DTE_WAIT_UNKNOWN_EVENT_ERROR) {
    compute_payload = WAFER_PROBE_SPM_DTE_RECV;
    compute_bytes = WAFER_PROBE_MAX_PAYLOAD_BYTES;
    compute_guard_before = WAFER_PROBE_RECV_GUARD_BEFORE;
    compute_guard_after = WAFER_PROBE_RECV_GUARD_AFTER;
    compute_elements = 0;
  } else {
    oracle.source_guard_mismatches = UINT8_MAX;
    oracle.receive_guard_mismatches = UINT8_MAX;
    oracle.compute_guard_mismatches = UINT8_MAX;
    oracle.compute_result_mismatches = UINT8_MAX;
    return oracle;
  }

  oracle.source_guard_mismatches = wafer_probe_guard_mismatches(
      source_payload, WAFER_PROBE_MAX_PAYLOAD_BYTES, source_guard_before,
      source_guard_after);
  oracle.receive_guard_mismatches = wafer_probe_guard_mismatches(
      WAFER_PROBE_SPM_DTE_RECV, WAFER_PROBE_MAX_PAYLOAD_BYTES,
      WAFER_PROBE_RECV_GUARD_BEFORE, WAFER_PROBE_RECV_GUARD_AFTER);
  if (mode == WAFER_PROBE_DTE_TWO_DESTINATION_BROADCAST)
    oracle.receive_guard_mismatches += wafer_probe_guard_mismatches(
        WAFER_PROBE_SPM_DTE_RECV_SECOND, WAFER_PROBE_MAX_PAYLOAD_BYTES,
        WAFER_PROBE_RECV_SECOND_GUARD_BEFORE,
        WAFER_PROBE_RECV_SECOND_GUARD_AFTER);
  oracle.compute_guard_mismatches =
      wafer_probe_guard_mismatches(compute_payload, compute_bytes,
                                   compute_guard_before, compute_guard_after);
  oracle.compute_result_mismatches +=
      mode == WAFER_PROBE_DTE_REUSE_AFTER_EVENTS ||
              mode == WAFER_PROBE_DTE_REUSE_BEFORE_SEND_EVENT_ERROR
          ? wafer_probe_copy_mismatches(
                compute_payload, compute_source_rank, compute_first_lane,
                compute_elements)
          : wafer_probe_compute_mismatches(
                compute_payload, compute_source_rank, compute_first_lane,
                compute_elements);
  return oracle;
}

static void wafer_probe_dma_read(uint64_t source_ddr, uint64_t dest_spm,
                                 uint32_t bytes) {
  wafer_tx81_rdma(source_ddr, dest_spm, bytes, bytes, 0, 0, 0, 1, 1, 1,
                  Fmt_FP16);
}

static void wafer_probe_dma_write(uint64_t source_spm, uint64_t dest_ddr,
                                  uint32_t bytes) {
  wafer_tx81_wdma(source_spm, dest_ddr, bytes, bytes, 0, 0, 0, 1, 1, 1,
                  Fmt_FP16);
}

static void wafer_probe_dte_ring(uint32_t rank, uint64_t source_spm,
                                 uint32_t bytes) {
  uint32_t predecessor =
      (rank + WAFER_PROBE_RANK_COUNT - 1) % WAFER_PROBE_RANK_COUNT;
  uint32_t successor = (rank + 1) % WAFER_PROBE_RANK_COUNT;
  uint64_t receive = wafer_tx81_direct_dte_recv_prepare(
      WAFER_PROBE_SPM_DTE_RECV, bytes, rank, predecessor, 0);
  uint64_t send = wafer_tx81_direct_dte_send_prepare(
      source_spm, WAFER_PROBE_SPM_DTE_RECV, bytes, rank, successor, 0, 0);
  /*
   * Every rank has posted its receive before entering the sender wait.  The
   * reverse order would make all ranks wait for an arrival that nobody has
   * started, so it is intentionally outside this positive-only probe.
   */
  wafer_tx81_direct_dte_wait(send);
  wafer_tx81_direct_dte_wait(receive);
}

static void wafer_probe_dte_two_destination_broadcast(
    uint32_t rank, uint64_t source_spm, uint32_t bytes) {
  uint32_t predecessor1 =
      (rank + WAFER_PROBE_RANK_COUNT - 1U) % WAFER_PROBE_RANK_COUNT;
  uint32_t predecessor2 =
      (rank + WAFER_PROBE_RANK_COUNT - 2U) % WAFER_PROBE_RANK_COUNT;
  uint32_t successor1 = (rank + 1U) % WAFER_PROBE_RANK_COUNT;
  uint32_t successor2 = (rank + 2U) % WAFER_PROBE_RANK_COUNT;
  uint64_t receive1 = wafer_tx81_direct_dte_recv_prepare(
      WAFER_PROBE_SPM_DTE_RECV, bytes, rank, predecessor1, 0);
  uint64_t receive2 = wafer_tx81_direct_dte_recv_prepare(
      WAFER_PROBE_SPM_DTE_RECV_SECOND, bytes, rank, predecessor2, 1);

  /*
   * The CRT owns one sender at a time.  Both destinations are prepared
   * before either send, then the same source is reused only after send1's
   * event completes.  This is a bounded fanout, not an assumption of a
   * native multicast packet.
   */
  uint64_t send1 = wafer_tx81_direct_dte_send_prepare(
      source_spm, WAFER_PROBE_SPM_DTE_RECV, bytes, rank, successor1, 0, 0);
  wafer_tx81_direct_dte_wait(send1);
  uint64_t send2 = wafer_tx81_direct_dte_send_prepare(
      source_spm, WAFER_PROBE_SPM_DTE_RECV_SECOND, bytes, rank, successor2, 1,
      0);
  wafer_tx81_direct_dte_wait(send2);
  wafer_tx81_direct_dte_wait(receive1);
  wafer_tx81_direct_dte_wait(receive2);
}

static uint32_t wafer_probe_dte_reuse_before_send_event(
    uint32_t rank, uint64_t source_spm, uint32_t bytes) {
  uint32_t predecessor =
      (rank + WAFER_PROBE_RANK_COUNT - 1U) % WAFER_PROBE_RANK_COUNT;
  uint32_t successor = (rank + 1U) % WAFER_PROBE_RANK_COUNT;
  uint64_t receive = wafer_tx81_direct_dte_recv_prepare(
      WAFER_PROBE_SPM_DTE_RECV, bytes, rank, predecessor, 0);
  uint64_t send = wafer_tx81_direct_dte_send_prepare(
      source_spm, WAFER_PROBE_SPM_DTE_RECV, bytes, rank, successor, 0, 0);
  uint64_t rejected = wafer_tx81_direct_dte_send_prepare(
      source_spm, WAFER_PROBE_SPM_DTE_RECV, bytes, rank, successor, 0, 0);
  uint32_t evidence = 0;
  if (send == UINT64_C(0x100))
    evidence |= WAFER_PROBE_CONTRACT_VALID_SEND_EVENT;
  if (receive == UINT64_C(0x200))
    evidence |= WAFER_PROBE_CONTRACT_VALID_RECV_EVENT;
  if (rejected == 0)
    evidence |= WAFER_PROBE_CONTRACT_REJECTED_SEND_EVENT;

  /*
   * The rejected duplicate prepare never attaches a transport node.  The
   * original valid send/receive pair still owns live state, so complete both
   * events before ending the shadow-status lifecycle.
   */
  wafer_tx81_direct_dte_wait(send);
  wafer_tx81_direct_dte_wait(receive);
  return evidence;
}

static uint32_t wafer_probe_dte_invalid_fsm(uint32_t rank, uint64_t source_spm,
                                            uint32_t bytes) {
  uint32_t predecessor =
      (rank + WAFER_PROBE_RANK_COUNT - 1U) % WAFER_PROBE_RANK_COUNT;
  uint32_t successor = (rank + 1U) % WAFER_PROBE_RANK_COUNT;
  uint64_t rejected_send = wafer_tx81_direct_dte_send_prepare(
      source_spm, WAFER_PROBE_SPM_DTE_RECV, bytes, rank, successor, 4, 0);
  uint64_t rejected_receive = wafer_tx81_direct_dte_recv_prepare(
      WAFER_PROBE_SPM_DTE_RECV, bytes, rank, predecessor, 4);
  uint32_t evidence = 0;
  if (rejected_send == 0)
    evidence |= WAFER_PROBE_CONTRACT_REJECTED_SEND_EVENT;
  if (rejected_receive == 0)
    evidence |= WAFER_PROBE_CONTRACT_REJECTED_RECV_EVENT;
  return evidence;
}

static void wafer_probe_publish_header(uint64_t output_ddr, uint32_t rank,
                                       uint32_t mode, uint32_t payload_bytes,
                                       uint32_t status,
                                       uint32_t contract_status,
                                       uint32_t contract_evidence,
                                       WaferProbePmu before,
                                       WaferProbePmu after,
                                       WaferProbeTransportPmu transport_before,
                                       WaferProbeTransportPmu transport_after,
                                       WaferProbeOracle oracle) {
  enum {
    WAFER_TX81_SUPERVISOR_MODE = 1,
    WAFER_TX81_MACHINE_MODE = 3,
  };
  volatile uint64_t *record = (volatile uint64_t *)(uintptr_t)output_ddr;
  uint32_t stable_mask = before.stable_mask & after.stable_mask;
  uint32_t transport_stable_mask =
      transport_before.stable_mask & transport_after.stable_mask;
  uint32_t scope_change =
      (transport_before.dte_enable != transport_after.dte_enable
           ? UINT32_C(1)
           : UINT32_C(0)) |
      (transport_before.spm_enable != transport_after.spm_enable ? UINT32_C(2)
                                                                 : UINT32_C(0));
  uint64_t metadata =
      (WAFER_PROBE_SCHEMA << 32) | ((uint64_t)mode << 24) |
      ((uint64_t)rank << 16) | payload_bytes |
      ((uint64_t)(stable_mask & UINT8_MAX) << 48) |
      ((uint64_t)(contract_status & UINT8_MAX) << 56);
  uint64_t counts =
      (uint64_t)(status & UINT8_MAX) |
      ((uint64_t)wafer_probe_u8_saturated(after.ct_count - before.ct_count)
       << 8) |
      ((uint64_t)wafer_probe_u8_saturated(after.rdma_count - before.rdma_count)
       << 16) |
      ((uint64_t)wafer_probe_u8_saturated(after.wdma_count - before.wdma_count)
       << 24) |
      ((uint64_t)wafer_probe_u8_saturated(oracle.source_guard_mismatches)
       << 32) |
      ((uint64_t)wafer_probe_u8_saturated(oracle.receive_guard_mismatches)
       << 40) |
      ((uint64_t)wafer_probe_u8_saturated(oracle.compute_guard_mismatches)
       << 48) |
      ((uint64_t)wafer_probe_u8_saturated(oracle.compute_result_mismatches)
       << 56);
  uintptr_t mode_bits;

  record[0] = WAFER_PROBE_MAGIC;
  record[1] = metadata;
  record[2] = WAFER_PROBE_CANARY;
  record[3] = counts;
  record[4] = after.full - before.full;
  record[5] = after.ct - before.ct;
  record[6] = after.rdma - before.rdma;
  record[7] = after.wdma - before.wdma;
  record[8] = (uint64_t)transport_stable_mask |
              ((uint64_t)WAFER_PROBE_TRANSPORT_STABLE_MASK << 16) |
              ((uint64_t)scope_change << 32) | (UINT64_C(1) << 40) |
              ((uint64_t)(contract_evidence & UINT8_MAX) << 48);
  record[9] = (uint64_t)transport_before.dte_enable |
              ((uint64_t)transport_before.spm_enable << 32);
  record[10] = transport_after.dte_channel0_transfer -
               transport_before.dte_channel0_transfer;
  record[11] = transport_after.dte_channel1_transfer -
               transport_before.dte_channel1_transfer;
  record[12] = transport_after.dte_channel0_execution -
               transport_before.dte_channel0_execution;
  record[13] = transport_after.dte_channel1_execution -
               transport_before.dte_channel1_execution;
  record[14] =
      transport_after.spm_dte_t2_port8 - transport_before.spm_dte_t2_port8;
  record[15] =
      transport_after.spm_dte_t3_port8 - transport_before.spm_dte_t3_port8;

  __asm__ volatile("fence" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
  __asm__ volatile("csrr %0, mxstatus" : "=r"(mode_bits));
  mode_bits = (mode_bits >> 30) & 3U;
  for (uintptr_t address = output_ddr;
       address < output_ddr + WAFER_PROBE_HEADER_BYTES;
       address += WAFER_TX81_DIRECT_DTE_STATUS_V2_CACHE_LINE_BYTES) {
    if (mode_bits == WAFER_TX81_MACHINE_MODE)
      __asm__ volatile("dcache.cipa %0" : : "r"(address) : "memory");
    else if (mode_bits == WAFER_TX81_SUPERVISOR_MODE)
      __asm__ volatile("dcache.civa %0" : : "r"(address) : "memory");
  }
  __asm__ volatile("sync.is" ::: "memory");
  __asm__ volatile("fence" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
}

static void wafer_probe_invalidate_input_header(uint64_t input_ddr) {
  enum {
    WAFER_TX81_SUPERVISOR_MODE = 1,
    WAFER_TX81_MACHINE_MODE = 3,
  };
  uintptr_t mode_bits;
  __asm__ volatile("fence" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
  __asm__ volatile("csrr %0, mxstatus" : "=r"(mode_bits));
  mode_bits = (mode_bits >> 30) & 3U;
  if (mode_bits == WAFER_TX81_MACHINE_MODE)
    __asm__ volatile("dcache.ipa %0" : : "r"(input_ddr) : "memory");
  else if (mode_bits == WAFER_TX81_SUPERVISOR_MODE)
    __asm__ volatile("dcache.iva %0" : : "r"(input_ddr) : "memory");
  __asm__ volatile("sync.is" ::: "memory");
  __asm__ volatile("fence" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
}

__attribute__((visibility("hidden"))) void
wafer_tx81_dte_ncc_execution_probe(uint32_t rank, uint64_t input_ddr,
                                   uint64_t output_ddr, uint64_t status_ddr) {
  const volatile uint32_t *input =
      (const volatile uint32_t *)(uintptr_t)input_ddr;
  uint32_t mode;
  uint32_t payload_bytes;
  uint64_t contract_status_ddr;
  uint64_t input_payload = input_ddr + WAFER_PROBE_HEADER_BYTES;
  uint64_t output_payload =
      output_ddr + WAFER_PROBE_HEADER_BYTES + WAFER_PROBE_OUTPUT_GUARD_BYTES;
  WaferProbePmu before;
  WaferProbePmu after;
  WaferProbeTransportPmu transport_before;
  WaferProbeTransportPmu transport_after;
  WaferProbeOracle oracle;
  uint32_t intermediate_mismatches = 0;
  uint32_t contract_status;
  uint32_t contract_evidence = 0;
  uint32_t runtime_status;

  /*
   * Host H2D and Kcore cached loads are not coherent on this profile.  Only
   * the request words are read by Kcore; tensor payload remains an RDMA
   * oracle.
   */
  wafer_probe_invalidate_input_header(input_ddr);
  mode = input[0];
  payload_bytes = input[1];
  contract_status_ddr =
      mode >= WAFER_PROBE_DTE_REUSE_BEFORE_SEND_EVENT_ERROR
          ? output_ddr
          : status_ddr;
  wafer_tx81_direct_dte_begin_after_prepare(contract_status_ddr,
                                            WAFER_PROBE_RANK_COUNT);
  wafer_probe_seed_regions();
  before = wafer_probe_read_pmu();
  transport_before = wafer_probe_read_transport_pmu();

  if (wafer_probe_payload_bytes_are_valid(payload_bytes)) {
    switch ((enum WaferDteNccProbeMode)mode) {
    case WAFER_PROBE_NCC_PRODUCER_DTE:
      wafer_probe_dma_read(input_payload, WAFER_PROBE_SPM_INPUT, payload_bytes);
      wafer_tx81_elementwise_add(WAFER_PROBE_SPM_INPUT, WAFER_PROBE_SPM_INPUT,
                                 WAFER_PROBE_SPM_PRODUCED,
                                 payload_bytes / (uint32_t)sizeof(uint16_t),
                                 Fmt_FP16);
      wafer_tx81_local_fence();
      wafer_probe_dte_ring(rank, WAFER_PROBE_SPM_PRODUCED, payload_bytes);
      wafer_probe_dma_write(WAFER_PROBE_SPM_DTE_RECV, output_payload,
                            WAFER_PROBE_MAX_PAYLOAD_BYTES);
      wafer_tx81_local_fence();
      break;

    case WAFER_PROBE_DTE_NCC_CONSUMER:
      wafer_probe_dma_read(input_payload, WAFER_PROBE_SPM_INPUT, payload_bytes);
      wafer_tx81_local_fence();
      wafer_probe_dte_ring(rank, WAFER_PROBE_SPM_INPUT, payload_bytes);
      wafer_tx81_elementwise_add(
          WAFER_PROBE_SPM_DTE_RECV, WAFER_PROBE_SPM_DTE_RECV,
          WAFER_PROBE_SPM_PRODUCED, payload_bytes / (uint32_t)sizeof(uint16_t),
          Fmt_FP16);
      wafer_tx81_local_fence();
      wafer_probe_dma_write(WAFER_PROBE_SPM_PRODUCED, output_payload,
                            WAFER_PROBE_MAX_PAYLOAD_BYTES);
      wafer_tx81_local_fence();
      break;

    case WAFER_PROBE_DISJOINT_LOCAL_WAIT_FIRST:
    case WAFER_PROBE_DISJOINT_DTE_WAIT_FIRST:
      wafer_probe_dma_read(input_payload, WAFER_PROBE_SPM_INPUT, payload_bytes);
      wafer_probe_dma_read(input_payload + WAFER_PROBE_MAX_PAYLOAD_BYTES,
                           WAFER_PROBE_SPM_DISJOINT_INPUT,
                           WAFER_PROBE_DISJOINT_BYTES);
      wafer_tx81_local_fence();
      wafer_tx81_elementwise_add(WAFER_PROBE_SPM_DISJOINT_INPUT,
                                 WAFER_PROBE_SPM_DISJOINT_INPUT,
                                 WAFER_PROBE_SPM_DISJOINT_OUTPUT,
                                 WAFER_PROBE_DISJOINT_ELEMENTS, Fmt_FP16);
      if (mode == WAFER_PROBE_DISJOINT_LOCAL_WAIT_FIRST) {
        wafer_tx81_local_fence();
        wafer_probe_dte_ring(rank, WAFER_PROBE_SPM_INPUT, payload_bytes);
      } else {
        wafer_probe_dte_ring(rank, WAFER_PROBE_SPM_INPUT, payload_bytes);
        wafer_tx81_local_fence();
      }
      wafer_probe_dma_write(WAFER_PROBE_SPM_DTE_RECV, output_payload,
                            WAFER_PROBE_MAX_PAYLOAD_BYTES);
      wafer_tx81_local_fence();
      break;

    case WAFER_PROBE_DTE_REUSE_AFTER_EVENTS: {
      uint32_t predecessor =
          (rank + WAFER_PROBE_RANK_COUNT - 1U) % WAFER_PROBE_RANK_COUNT;
      uint32_t elements =
          payload_bytes / (uint32_t)sizeof(uint16_t);
      wafer_probe_dma_read(input_payload, WAFER_PROBE_SPM_INPUT,
                           payload_bytes);
      wafer_tx81_local_fence();
      wafer_probe_dte_ring(rank, WAFER_PROBE_SPM_INPUT, payload_bytes);
      intermediate_mismatches = wafer_probe_copy_mismatches(
          WAFER_PROBE_SPM_DTE_RECV, predecessor, 0, elements);

      /*
       * Both send and receive events have completed before the same source
       * and destination slots are reused with a distinguishable payload.
       */
      wafer_probe_dma_read(input_payload + WAFER_PROBE_MAX_PAYLOAD_BYTES,
                           WAFER_PROBE_SPM_INPUT, payload_bytes);
      wafer_tx81_local_fence();
      wafer_probe_dte_ring(rank, WAFER_PROBE_SPM_INPUT, payload_bytes);
      wafer_probe_dma_write(WAFER_PROBE_SPM_DTE_RECV, output_payload,
                            WAFER_PROBE_MAX_PAYLOAD_BYTES);
      wafer_tx81_local_fence();
      break;
    }

    case WAFER_PROBE_DTE_TWO_DESTINATION_BROADCAST:
      wafer_probe_dma_read(input_payload, WAFER_PROBE_SPM_INPUT,
                           payload_bytes);
      wafer_tx81_local_fence();
      wafer_probe_dte_two_destination_broadcast(
          rank, WAFER_PROBE_SPM_INPUT, payload_bytes);
      wafer_probe_dma_write(WAFER_PROBE_SPM_DTE_RECV, output_payload,
                            WAFER_PROBE_MAX_PAYLOAD_BYTES);
      wafer_tx81_local_fence();
      break;

    case WAFER_PROBE_DTE_REUSE_BEFORE_SEND_EVENT_ERROR:
      wafer_probe_dma_read(input_payload, WAFER_PROBE_SPM_INPUT, payload_bytes);
      wafer_tx81_local_fence();
      contract_evidence = wafer_probe_dte_reuse_before_send_event(
          rank, WAFER_PROBE_SPM_INPUT, payload_bytes);
      wafer_probe_dma_write(WAFER_PROBE_SPM_DTE_RECV, output_payload,
                            WAFER_PROBE_MAX_PAYLOAD_BYTES);
      wafer_tx81_local_fence();
      break;

    case WAFER_PROBE_DTE_INVALID_FSM_ERROR:
      contract_evidence =
          wafer_probe_dte_invalid_fsm(rank, WAFER_PROBE_SPM_INPUT,
                                      payload_bytes);
      wafer_probe_dma_write(WAFER_PROBE_SPM_DTE_RECV, output_payload,
                            WAFER_PROBE_MAX_PAYLOAD_BYTES);
      wafer_tx81_local_fence();
      break;

    case WAFER_PROBE_DTE_WAIT_UNKNOWN_EVENT_ERROR:
      wafer_tx81_direct_dte_wait(UINT64_C(0xdeadbeef));
      contract_evidence |= WAFER_PROBE_CONTRACT_UNKNOWN_WAIT_RETURNED;
      wafer_probe_dma_write(WAFER_PROBE_SPM_DTE_RECV, output_payload,
                            WAFER_PROBE_MAX_PAYLOAD_BYTES);
      wafer_tx81_local_fence();
      break;
    }
  }

  after = wafer_probe_read_pmu();
  transport_after = wafer_probe_read_transport_pmu();
  oracle = wafer_probe_read_oracle(rank, mode, payload_bytes);
  oracle.compute_result_mismatches += intermediate_mismatches;
  wafer_tx81_direct_dte_finish();
  contract_status =
      *(const volatile uint32_t *)(uintptr_t)contract_status_ddr;
  if (contract_status == WAFER_TX81_DIRECT_DTE_STATUS_TRANSPORT_ERROR)
    contract_evidence |= WAFER_PROBE_CONTRACT_STATUS_ERROR;
  if (contract_status_ddr != status_ddr) {
    /*
     * BoardRuntime correctly rejects a transport-error terminal status.  The
     * shadow cache line above preserves the error observation, while this
     * fresh empty lifecycle proves all guarded state was cleaned and gives
     * the owning runtime status resource its required SUCCESS terminal.
     */
    wafer_tx81_direct_dte_begin_after_prepare(status_ddr,
                                              WAFER_PROBE_RANK_COUNT);
    wafer_tx81_direct_dte_finish();
  }
  runtime_status = *(const volatile uint32_t *)(uintptr_t)status_ddr;
  wafer_probe_publish_header(output_ddr, rank, mode, payload_bytes,
                             runtime_status, contract_status,
                             contract_evidence,
                             before, after, transport_before, transport_after,
                             oracle);
}
