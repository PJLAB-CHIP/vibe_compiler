#include "Wafer/ABI/Tx81DirectDTEStatusABI.h"
#include "direct_dte_and_fsm.h"
#include "instr_def.h"
#include "pmu/pmu_reg.h"
#include "tx81_spm.h"
#include "wafer_tx81_crt.h"

#include <stddef.h>
#include <stdint.h>

#define WAFER_PROBE_RANK_COUNT UINT32_C(16)
#define WAFER_PROBE_HEADER_BYTES UINT32_C(128)
#define WAFER_PROBE_MAX_PAYLOAD_BYTES UINT32_C(4096)
#define WAFER_PROBE_DISJOINT_BYTES UINT32_C(32)
#define WAFER_PROBE_SPM_GUARD_BYTES UINT32_C(256)
#define WAFER_PROBE_ASYNC_TRANSPORT_BYTES UINT32_C(65536)
#define WAFER_PROBE_HOST_SLOT_BYTES                                        \
  (WAFER_PROBE_ASYNC_TRANSPORT_BYTES + 2U * WAFER_PROBE_SPM_GUARD_BYTES)
#define WAFER_PROBE_HOST_SLOT_COUNT UINT32_C(4)
#define WAFER_PROBE_RESOURCE_BYTES UINT32_C(264448)
#define WAFER_PROBE_DISJOINT_ELEMENTS                                          \
  (WAFER_PROBE_DISJOINT_BYTES / (uint32_t)sizeof(uint16_t))
#define WAFER_PROBE_ASYNC_TRANSPORT_ELEMENTS                                   \
  (WAFER_PROBE_ASYNC_TRANSPORT_BYTES / (uint32_t)sizeof(uint16_t))

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
#define WAFER_PROBE_SPM_ASYNC_SOURCE                                           \
  (UINT64_C(0x100000) + WAFER_PROBE_SPM_GUARD_BYTES)
#define WAFER_PROBE_SPM_ASYNC_RECV                                             \
  (UINT64_C(0x120000) + WAFER_PROBE_SPM_GUARD_BYTES)
#define WAFER_PROBE_SPM_ASYNC_COMPUTE_INPUT                                    \
  (UINT64_C(0x140000) + WAFER_PROBE_SPM_GUARD_BYTES)
#define WAFER_PROBE_SPM_ASYNC_COMPUTE_OUTPUT                                   \
  (UINT64_C(0x160000) + WAFER_PROBE_SPM_GUARD_BYTES)

#define WAFER_PROBE_PMU_BASE UINT64_C(0x590000)
#define WAFER_PROBE_MAGIC UINT64_C(0x3143434e45544457)
#define WAFER_PROBE_CANARY UINT64_C(0xd7e0ca11d7e0ca11)
#define WAFER_PROBE_SCHEMA UINT64_C(7)
#define WAFER_PROBE_STABLE_RETRIES UINT32_C(8)
#define WAFER_PROBE_TRANSPORT_COUNTER_COUNT UINT32_C(6)
#define WAFER_PROBE_TRANSPORT_STABLE_MASK                                      \
  ((UINT32_C(1) << WAFER_PROBE_TRANSPORT_COUNTER_COUNT) - UINT32_C(1))
#define WAFER_PROBE_RAW_ASYNC_RC_MARKER UINT32_C(0x4153594e)

#define WAFER_PROBE_INITIAL_CANARY UINT8_C(0xa5)

#if WAFER_PROBE_HEADER_BYTES +                                                  \
        WAFER_PROBE_HOST_SLOT_COUNT * WAFER_PROBE_HOST_SLOT_BYTES >             \
    WAFER_PROBE_RESOURCE_BYTES
#error "DTE/NCC probe record exceeds its output resource"
#endif

#if WAFER_PROBE_HEADER_BYTES + 2U * WAFER_PROBE_HOST_SLOT_BYTES >               \
    WAFER_PROBE_RESOURCE_BYTES
#error "DTE/NCC input data exceeds its input resource"
#endif

#if WAFER_PROBE_RESOURCE_BYTES % 256U != 0U
#error "DTE/NCC host resources must remain 256-byte aligned"
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
  WAFER_PROBE_DTE_SENDER_RAW_ASYNC_SERIAL = 10,
  WAFER_PROBE_DTE_SENDER_RAW_ASYNC_WINDOW = 11,
  WAFER_PROBE_DTE_REUSE_BEFORE_RECV_EVENT_ERROR = 12,
  WAFER_PROBE_DTE_RECEIVER_UNPREPARED = 13,
};

enum WaferDteContractEvidence {
  WAFER_PROBE_CONTRACT_STATUS_ERROR = UINT32_C(1) << 0,
  WAFER_PROBE_CONTRACT_VALID_SEND_EVENT = UINT32_C(1) << 1,
  WAFER_PROBE_CONTRACT_VALID_RECV_EVENT = UINT32_C(1) << 2,
  WAFER_PROBE_CONTRACT_REJECTED_SEND_EVENT = UINT32_C(1) << 3,
  WAFER_PROBE_CONTRACT_REJECTED_RECV_EVENT = UINT32_C(1) << 4,
  WAFER_PROBE_CONTRACT_UNKNOWN_WAIT_RETURNED = UINT32_C(1) << 5,
  WAFER_PROBE_CONTRACT_RAW_ASYNC_ISSUED = UINT32_C(1) << 6,
  WAFER_PROBE_CONTRACT_RAW_ASYNC_COMPLETED = UINT32_C(1) << 7,
};

_Static_assert(sizeof(DirectDTESendInfo) == 64U,
               "raw sender probe requires version-matched SendInfo layout");
_Static_assert(offsetof(DirectDTESendInfo, stride_iterations) == 28U,
               "raw sender probe requires version-matched stride layout");
_Static_assert(offsetof(DirectDTESendInfo, dte_node) == 56U,
               "raw sender probe requires version-matched node layout");

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

typedef struct {
  uint32_t evidence;
  int32_t send_result;
  int32_t wait_result;
  int32_t release_result;
} WaferProbeRawAsyncResult;

static uint32_t wafer_probe_u8_saturated(uint32_t value) {
  return value > UINT8_MAX ? UINT8_MAX : value;
}

static int wafer_probe_payload_bytes_are_valid(uint32_t payload_bytes) {
  return payload_bytes == UINT32_C(16) || payload_bytes == UINT32_C(32) ||
         payload_bytes == UINT32_C(64) || payload_bytes == UINT32_C(256) ||
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

static uint64_t wafer_probe_host_slot(uint64_t base, uint32_t slot) {
  return base + WAFER_PROBE_HEADER_BYTES +
         (uint64_t)slot * WAFER_PROBE_HOST_SLOT_BYTES;
}

static void wafer_probe_seed_guarded_region(uint64_t source_ddr,
                                            uint64_t payload_spm,
                                            uint32_t payload_bytes) {
  wafer_probe_dma_read(source_ddr,
                       payload_spm - WAFER_PROBE_SPM_GUARD_BYTES,
                       payload_bytes + 2U * WAFER_PROBE_SPM_GUARD_BYTES);
}

static void wafer_probe_seed_regions(uint64_t input_ddr, uint64_t output_ddr,
                                     uint32_t mode) {
  uint64_t canary = wafer_probe_host_slot(output_ddr, 0);
  if (mode == WAFER_PROBE_DTE_SENDER_RAW_ASYNC_SERIAL ||
      mode == WAFER_PROBE_DTE_SENDER_RAW_ASYNC_WINDOW) {
    wafer_probe_seed_guarded_region(wafer_probe_host_slot(input_ddr, 0),
                                    WAFER_PROBE_SPM_ASYNC_SOURCE,
                                    WAFER_PROBE_ASYNC_TRANSPORT_BYTES);
    wafer_probe_seed_guarded_region(canary, WAFER_PROBE_SPM_ASYNC_RECV,
                                    WAFER_PROBE_ASYNC_TRANSPORT_BYTES);
    wafer_probe_seed_guarded_region(wafer_probe_host_slot(input_ddr, 1),
                                    WAFER_PROBE_SPM_ASYNC_COMPUTE_INPUT,
                                    WAFER_PROBE_ASYNC_TRANSPORT_BYTES);
    wafer_probe_seed_guarded_region(canary,
                                    WAFER_PROBE_SPM_ASYNC_COMPUTE_OUTPUT,
                                    WAFER_PROBE_ASYNC_TRANSPORT_BYTES);
  } else {
    wafer_probe_seed_guarded_region(canary, WAFER_PROBE_SPM_INPUT,
                                    WAFER_PROBE_MAX_PAYLOAD_BYTES);
    wafer_probe_seed_guarded_region(canary, WAFER_PROBE_SPM_PRODUCED,
                                    WAFER_PROBE_MAX_PAYLOAD_BYTES);
    wafer_probe_seed_guarded_region(canary, WAFER_PROBE_SPM_DTE_RECV,
                                    WAFER_PROBE_MAX_PAYLOAD_BYTES);
    wafer_probe_seed_guarded_region(canary, WAFER_PROBE_SPM_DTE_RECV_SECOND,
                                    WAFER_PROBE_MAX_PAYLOAD_BYTES);
    wafer_probe_seed_guarded_region(canary, WAFER_PROBE_SPM_DISJOINT_INPUT,
                                    WAFER_PROBE_DISJOINT_BYTES);
    wafer_probe_seed_guarded_region(canary, WAFER_PROBE_SPM_DISJOINT_OUTPUT,
                                    WAFER_PROBE_DISJOINT_BYTES);
  }
  /*
   * Setup is deliberately outside the measured window.  This drain is the
   * real NCC-to-Kcore PMU boundary, not a mapped-SPM visibility workaround.
   */
  wafer_tx81_local_fence();
}

static void wafer_probe_capture_region(uint64_t output_ddr, uint32_t slot,
                                       uint64_t payload_spm,
                                       uint32_t payload_bytes) {
  wafer_probe_dma_write(payload_spm - WAFER_PROBE_SPM_GUARD_BYTES,
                        wafer_probe_host_slot(output_ddr, slot),
                        payload_bytes + 2U * WAFER_PROBE_SPM_GUARD_BYTES);
}

static void wafer_probe_capture_results(uint64_t output_ddr, uint32_t mode) {
  switch ((enum WaferDteNccProbeMode)mode) {
  case WAFER_PROBE_NCC_PRODUCER_DTE:
    wafer_probe_capture_region(output_ddr, 0, WAFER_PROBE_SPM_PRODUCED,
                               WAFER_PROBE_MAX_PAYLOAD_BYTES);
    wafer_probe_capture_region(output_ddr, 1, WAFER_PROBE_SPM_DTE_RECV,
                               WAFER_PROBE_MAX_PAYLOAD_BYTES);
    wafer_probe_capture_region(output_ddr, 2, WAFER_PROBE_SPM_PRODUCED,
                               WAFER_PROBE_MAX_PAYLOAD_BYTES);
    wafer_probe_capture_region(output_ddr, 3, WAFER_PROBE_SPM_DTE_RECV_SECOND,
                               WAFER_PROBE_MAX_PAYLOAD_BYTES);
    break;
  case WAFER_PROBE_DISJOINT_LOCAL_WAIT_FIRST:
  case WAFER_PROBE_DISJOINT_DTE_WAIT_FIRST:
    wafer_probe_capture_region(output_ddr, 0, WAFER_PROBE_SPM_INPUT,
                               WAFER_PROBE_MAX_PAYLOAD_BYTES);
    wafer_probe_capture_region(output_ddr, 1, WAFER_PROBE_SPM_DTE_RECV,
                               WAFER_PROBE_MAX_PAYLOAD_BYTES);
    wafer_probe_capture_region(output_ddr, 2, WAFER_PROBE_SPM_DISJOINT_OUTPUT,
                               WAFER_PROBE_DISJOINT_BYTES);
    wafer_probe_capture_region(output_ddr, 3, WAFER_PROBE_SPM_DISJOINT_INPUT,
                               WAFER_PROBE_DISJOINT_BYTES);
    break;
  case WAFER_PROBE_DTE_SENDER_RAW_ASYNC_SERIAL:
  case WAFER_PROBE_DTE_SENDER_RAW_ASYNC_WINDOW:
    wafer_probe_capture_region(output_ddr, 0, WAFER_PROBE_SPM_ASYNC_SOURCE,
                               WAFER_PROBE_ASYNC_TRANSPORT_BYTES);
    wafer_probe_capture_region(output_ddr, 1, WAFER_PROBE_SPM_ASYNC_RECV,
                               WAFER_PROBE_ASYNC_TRANSPORT_BYTES);
    wafer_probe_capture_region(output_ddr, 2,
                               WAFER_PROBE_SPM_ASYNC_COMPUTE_INPUT,
                               WAFER_PROBE_ASYNC_TRANSPORT_BYTES);
    wafer_probe_capture_region(output_ddr, 3,
                               WAFER_PROBE_SPM_ASYNC_COMPUTE_OUTPUT,
                               WAFER_PROBE_ASYNC_TRANSPORT_BYTES);
    break;
  case WAFER_PROBE_DTE_REUSE_AFTER_EVENTS:
    wafer_probe_capture_region(output_ddr, 0, WAFER_PROBE_SPM_INPUT,
                               WAFER_PROBE_MAX_PAYLOAD_BYTES);
    wafer_probe_capture_region(output_ddr, 1, WAFER_PROBE_SPM_DTE_RECV,
                               WAFER_PROBE_MAX_PAYLOAD_BYTES);
    wafer_probe_capture_region(output_ddr, 2, WAFER_PROBE_SPM_DTE_RECV,
                               WAFER_PROBE_MAX_PAYLOAD_BYTES);
    break;
  case WAFER_PROBE_DTE_NCC_CONSUMER:
    wafer_probe_capture_region(output_ddr, 0, WAFER_PROBE_SPM_INPUT,
                               WAFER_PROBE_MAX_PAYLOAD_BYTES);
    wafer_probe_capture_region(output_ddr, 1, WAFER_PROBE_SPM_DTE_RECV,
                               WAFER_PROBE_MAX_PAYLOAD_BYTES);
    wafer_probe_capture_region(output_ddr, 2, WAFER_PROBE_SPM_PRODUCED,
                               WAFER_PROBE_MAX_PAYLOAD_BYTES);
    wafer_probe_capture_region(output_ddr, 3, WAFER_PROBE_SPM_DTE_RECV_SECOND,
                               WAFER_PROBE_MAX_PAYLOAD_BYTES);
    break;
  default:
    wafer_probe_capture_region(output_ddr, 0, WAFER_PROBE_SPM_INPUT,
                               WAFER_PROBE_MAX_PAYLOAD_BYTES);
    wafer_probe_capture_region(output_ddr, 1, WAFER_PROBE_SPM_DTE_RECV,
                               WAFER_PROBE_MAX_PAYLOAD_BYTES);
    wafer_probe_capture_region(output_ddr, 2, WAFER_PROBE_SPM_DTE_RECV,
                               WAFER_PROBE_MAX_PAYLOAD_BYTES);
    wafer_probe_capture_region(output_ddr, 3, WAFER_PROBE_SPM_DTE_RECV_SECOND,
                               WAFER_PROBE_MAX_PAYLOAD_BYTES);
    break;
  }
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

static void wafer_probe_dte_receiver_unprepared(uint32_t rank,
                                                uint64_t source_spm,
                                                uint32_t bytes) {
  uint32_t successor = (rank + 1U) % WAFER_PROBE_RANK_COUNT;
  uint64_t send = wafer_tx81_direct_dte_send_prepare(
      source_spm, WAFER_PROBE_SPM_DTE_RECV, bytes, rank, successor, 0, 0);
  wafer_tx81_direct_dte_wait(send);
}

/*
 * Test-only raw sender seam.  Production continues to use the CRT event API,
 * whose sender wait intentionally owns attach, async issue, wait, and release
 * as one synchronous operation.  This seam keeps receiver-first ordering and
 * the same bounded lifecycle, but exposes one disjoint CT issue between
 * send_async and wait_done for comparison with the serial control.
 */
static WaferProbeRawAsyncResult
wafer_probe_dte_sender_raw_async(uint32_t rank, uint32_t issue_window) {
  uint32_t predecessor =
      (rank + WAFER_PROBE_RANK_COUNT - 1U) % WAFER_PROBE_RANK_COUNT;
  uint32_t successor = (rank + 1U) % WAFER_PROBE_RANK_COUNT;
  WaferProbeRawAsyncResult result = {
      .evidence = 0,
      .send_result = INT32_MIN,
      .wait_result = INT32_MIN,
      .release_result = INT32_MIN,
  };
  uint64_t receive = wafer_tx81_direct_dte_recv_prepare(
      WAFER_PROBE_SPM_ASYNC_RECV, WAFER_PROBE_ASYNC_TRANSPORT_BYTES, rank,
      predecessor, 0);
  if (receive == UINT64_C(0x200))
    result.evidence |= WAFER_PROBE_CONTRACT_VALID_RECV_EVENT;

  direct_sync_wait(rank, successor);
  DirectDTESendInfo info = {0};
  info.src_addr = (uintptr_t)WAFER_PROBE_SPM_ASYNC_SOURCE;
  info.dst_addr =
      (uintptr_t)(get_tile_spm_addr_base(successor, 4, 4) +
                  WAFER_PROBE_SPM_ASYNC_RECV);
  info.length = WAFER_PROBE_ASYNC_TRANSPORT_BYTES;
  info.remote_fsm_id = 0;
  info.mode = 0;
  info.dst_tile = (uint16_t)successor;
  info.tile_this = (uint16_t)rank;
  info.dte_node = direct_dte_attach(0);

  if (info.dte_node != NULL) {
    result.send_result = direct_dte_send_async(&info);
    if (result.send_result == 0)
      result.evidence |= WAFER_PROBE_CONTRACT_RAW_ASYNC_ISSUED;
  }

  if (result.send_result == 0 && issue_window != 0U)
    wafer_tx81_elementwise_add(
        WAFER_PROBE_SPM_ASYNC_COMPUTE_INPUT,
        WAFER_PROBE_SPM_ASYNC_COMPUTE_INPUT,
        WAFER_PROBE_SPM_ASYNC_COMPUTE_OUTPUT,
        WAFER_PROBE_ASYNC_TRANSPORT_ELEMENTS, Fmt_FP16);

  if (result.send_result == 0)
    result.wait_result = direct_dte_wait_done(&info);
  if (info.dte_node != NULL)
    result.release_result = direct_dte_release(info.dte_node);
  wafer_tx81_direct_dte_wait(receive);

  if (result.send_result == 0 && issue_window == 0U)
    wafer_tx81_elementwise_add(
        WAFER_PROBE_SPM_ASYNC_COMPUTE_INPUT,
        WAFER_PROBE_SPM_ASYNC_COMPUTE_INPUT,
        WAFER_PROBE_SPM_ASYNC_COMPUTE_OUTPUT,
        WAFER_PROBE_ASYNC_TRANSPORT_ELEMENTS, Fmt_FP16);
  wafer_tx81_local_fence();

  if (result.send_result == 0 && result.wait_result == 0 &&
      result.release_result == 0)
    result.evidence |= WAFER_PROBE_CONTRACT_RAW_ASYNC_COMPLETED;
  return result;
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

static uint32_t wafer_probe_dte_reuse_before_recv_event(
    uint32_t rank, uint64_t source_spm, uint32_t bytes) {
  uint32_t predecessor =
      (rank + WAFER_PROBE_RANK_COUNT - 1U) % WAFER_PROBE_RANK_COUNT;
  uint32_t successor = (rank + 1U) % WAFER_PROBE_RANK_COUNT;
  uint64_t receive = wafer_tx81_direct_dte_recv_prepare(
      WAFER_PROBE_SPM_DTE_RECV, bytes, rank, predecessor, 0);
  uint64_t rejected = wafer_tx81_direct_dte_recv_prepare(
      WAFER_PROBE_SPM_DTE_RECV_SECOND, bytes, rank, predecessor, 0);
  uint64_t send = wafer_tx81_direct_dte_send_prepare(
      source_spm, WAFER_PROBE_SPM_DTE_RECV, bytes, rank, successor, 0, 0);
  uint32_t evidence = 0;
  if (send == UINT64_C(0x100))
    evidence |= WAFER_PROBE_CONTRACT_VALID_SEND_EVENT;
  if (receive == UINT64_C(0x200))
    evidence |= WAFER_PROBE_CONTRACT_VALID_RECV_EVENT;
  if (rejected == 0)
    evidence |= WAFER_PROBE_CONTRACT_REJECTED_RECV_EVENT;

  /*
   * The rejected duplicate never replaces the active receiver handle or
   * posts a second readiness token.  Complete the original matching pair so
   * the shadow-status lifecycle remains bounded and owns normal cleanup.
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
                                       WaferProbeOracle oracle,
                                       WaferProbeRawAsyncResult raw_async) {
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
  if (mode == WAFER_PROBE_DTE_SENDER_RAW_ASYNC_SERIAL ||
      mode == WAFER_PROBE_DTE_SENDER_RAW_ASYNC_WINDOW) {
    record[6] = (uint64_t)(uint32_t)raw_async.send_result |
                ((uint64_t)(uint32_t)raw_async.wait_result << 32);
    record[7] = (uint64_t)(uint32_t)raw_async.release_result |
                ((uint64_t)WAFER_PROBE_RAW_ASYNC_RC_MARKER << 32);
  } else {
    record[6] = after.rdma - before.rdma;
    record[7] = after.wdma - before.wdma;
  }
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
  uint64_t input_payload =
      wafer_probe_host_slot(input_ddr, 0) + WAFER_PROBE_SPM_GUARD_BYTES;
  WaferProbePmu before;
  WaferProbePmu after;
  WaferProbeTransportPmu transport_before;
  WaferProbeTransportPmu transport_after;
  WaferProbeOracle oracle = {0};
  uint32_t contract_status;
  uint32_t contract_evidence = 0;
  uint32_t runtime_status;
  WaferProbeRawAsyncResult raw_async = {
      .evidence = 0,
      .send_result = INT32_MIN,
      .wait_result = INT32_MIN,
      .release_result = INT32_MIN,
  };

  /*
   * Host H2D and Kcore cached loads are not coherent on this profile.  Only
   * the request words are read by Kcore; tensor payload remains an RDMA
   * oracle.
   */
  wafer_probe_invalidate_input_header(input_ddr);
  mode = input[0];
  payload_bytes = input[1];
  contract_status_ddr =
      (mode >= WAFER_PROBE_DTE_REUSE_BEFORE_SEND_EVENT_ERROR &&
       mode <= WAFER_PROBE_DTE_WAIT_UNKNOWN_EVENT_ERROR) ||
              mode == WAFER_PROBE_DTE_REUSE_BEFORE_RECV_EVENT_ERROR
          ? output_ddr
          : status_ddr;
  wafer_tx81_direct_dte_begin_after_prepare(contract_status_ddr,
                                            WAFER_PROBE_RANK_COUNT);
  wafer_probe_seed_regions(input_ddr, output_ddr, mode);
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
      break;

    case WAFER_PROBE_DTE_REUSE_AFTER_EVENTS: {
      wafer_probe_dma_read(input_payload, WAFER_PROBE_SPM_INPUT,
                           payload_bytes);
      wafer_tx81_local_fence();
      wafer_probe_dte_ring(rank, WAFER_PROBE_SPM_INPUT, payload_bytes);
      wafer_probe_capture_region(output_ddr, 3, WAFER_PROBE_SPM_DTE_RECV,
                                 WAFER_PROBE_MAX_PAYLOAD_BYTES);
      /*
       * The capture must finish before Direct DTE reuses the same receive
       * slot.  This is a real NCC-to-DTE completion-domain boundary.
       */
      wafer_tx81_local_fence();

      /*
       * Both send and receive events have completed before the same source
       * and destination slots are reused with a distinguishable payload.
       */
      wafer_probe_dma_read(input_payload + WAFER_PROBE_MAX_PAYLOAD_BYTES,
                           WAFER_PROBE_SPM_INPUT, payload_bytes);
      wafer_tx81_local_fence();
      wafer_probe_dte_ring(rank, WAFER_PROBE_SPM_INPUT, payload_bytes);
      break;
    }

    case WAFER_PROBE_DTE_TWO_DESTINATION_BROADCAST:
      wafer_probe_dma_read(input_payload, WAFER_PROBE_SPM_INPUT,
                           payload_bytes);
      wafer_tx81_local_fence();
      wafer_probe_dte_two_destination_broadcast(
          rank, WAFER_PROBE_SPM_INPUT, payload_bytes);
      break;

    case WAFER_PROBE_DTE_REUSE_BEFORE_SEND_EVENT_ERROR:
      wafer_probe_dma_read(input_payload, WAFER_PROBE_SPM_INPUT, payload_bytes);
      wafer_tx81_local_fence();
      contract_evidence = wafer_probe_dte_reuse_before_send_event(
          rank, WAFER_PROBE_SPM_INPUT, payload_bytes);
      break;

    case WAFER_PROBE_DTE_REUSE_BEFORE_RECV_EVENT_ERROR:
      wafer_probe_dma_read(input_payload, WAFER_PROBE_SPM_INPUT, payload_bytes);
      wafer_tx81_local_fence();
      contract_evidence = wafer_probe_dte_reuse_before_recv_event(
          rank, WAFER_PROBE_SPM_INPUT, payload_bytes);
      break;

    case WAFER_PROBE_DTE_INVALID_FSM_ERROR:
      contract_evidence =
          wafer_probe_dte_invalid_fsm(rank, WAFER_PROBE_SPM_INPUT,
                                      payload_bytes);
      break;

    case WAFER_PROBE_DTE_WAIT_UNKNOWN_EVENT_ERROR:
      wafer_tx81_direct_dte_wait(UINT64_C(0xdeadbeef));
      contract_evidence |= WAFER_PROBE_CONTRACT_UNKNOWN_WAIT_RETURNED;
      break;

    case WAFER_PROBE_DTE_SENDER_RAW_ASYNC_SERIAL:
    case WAFER_PROBE_DTE_SENDER_RAW_ASYNC_WINDOW:
      raw_async = wafer_probe_dte_sender_raw_async(
          rank, mode == WAFER_PROBE_DTE_SENDER_RAW_ASYNC_WINDOW);
      contract_evidence = raw_async.evidence;
      break;

    case WAFER_PROBE_DTE_RECEIVER_UNPREPARED:
      wafer_probe_dma_read(input_payload, WAFER_PROBE_SPM_INPUT, payload_bytes);
      wafer_tx81_local_fence();
      wafer_probe_dte_receiver_unprepared(rank, WAFER_PROBE_SPM_INPUT,
                                          payload_bytes);
      break;
    }
  }

  after = wafer_probe_read_pmu();
  transport_after = wafer_probe_read_transport_pmu();
  wafer_probe_capture_results(output_ddr, mode);
  /*
   * Readback is outside the measurement window.  One terminal drain publishes
   * all four guarded slots to host DDR before Kcore writes the result header.
   */
  wafer_tx81_local_fence();
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
                             oracle, raw_async);
}
