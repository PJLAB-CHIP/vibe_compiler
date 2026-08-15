#include "instr_def.h"
#include "wafer_ct_convert_calibration_probe_protocol.h"
#include "wafer_tx81_crt.h"

#include <stdint.h>

#define WAFER_CTC_PMU_BASE UINT64_C(0x590000)
#define WAFER_CTC_STABLE_RETRIES 8U

typedef struct WaferCTCCase {
  uint32_t case_id;
  uint32_t opcode;
  uint32_t source_type;
  uint32_t destination_type;
  uint32_t elements;
  uint32_t input_bytes;
  uint32_t result_bytes;
  uint32_t output_span;
  uint32_t rounding_mode;
  uint32_t zero_point;
  uint32_t domain;
  uint32_t disposition;
} WaferCTCCase;

typedef struct WaferCTCPMU {
  uint32_t instructions;
  uint32_t blocking;
  uint64_t execution;
} WaferCTCPMU;

static const uint8_t wafer_ctc_source_types[36] = {
    WAFER_CTC_INT8,  WAFER_CTC_INT8,  WAFER_CTC_INT8,  WAFER_CTC_INT8,
    WAFER_CTC_INT16, WAFER_CTC_INT16, WAFER_CTC_INT16, WAFER_CTC_INT16,
    WAFER_CTC_INT32, WAFER_CTC_INT32, WAFER_CTC_INT32, WAFER_CTC_INT32,
    WAFER_CTC_BF16,  WAFER_CTC_BF16,  WAFER_CTC_BF16,  WAFER_CTC_BF16,
    WAFER_CTC_BF16,  WAFER_CTC_BF16,  WAFER_CTC_FP16,  WAFER_CTC_FP16,
    WAFER_CTC_FP16,  WAFER_CTC_FP16,  WAFER_CTC_FP16,  WAFER_CTC_FP16,
    WAFER_CTC_FP32,  WAFER_CTC_FP32,  WAFER_CTC_FP32,  WAFER_CTC_FP32,
    WAFER_CTC_FP32,  WAFER_CTC_FP32,  WAFER_CTC_TF32,  WAFER_CTC_TF32,
    WAFER_CTC_TF32,  WAFER_CTC_TF32,  WAFER_CTC_TF32,  WAFER_CTC_TF32,
};

static const uint8_t wafer_ctc_destination_types[36] = {
    WAFER_CTC_FP16,  WAFER_CTC_BF16,  WAFER_CTC_FP32,  WAFER_CTC_TF32,
    WAFER_CTC_FP16,  WAFER_CTC_BF16,  WAFER_CTC_FP32,  WAFER_CTC_TF32,
    WAFER_CTC_FP16,  WAFER_CTC_BF16,  WAFER_CTC_FP32,  WAFER_CTC_TF32,
    WAFER_CTC_INT8,  WAFER_CTC_INT16, WAFER_CTC_INT32, WAFER_CTC_FP16,
    WAFER_CTC_FP32,  WAFER_CTC_TF32,  WAFER_CTC_INT8,  WAFER_CTC_INT16,
    WAFER_CTC_INT32, WAFER_CTC_BF16,  WAFER_CTC_FP32,  WAFER_CTC_TF32,
    WAFER_CTC_INT8,  WAFER_CTC_INT16, WAFER_CTC_INT32, WAFER_CTC_FP16,
    WAFER_CTC_BF16,  WAFER_CTC_TF32,  WAFER_CTC_INT8,  WAFER_CTC_INT16,
    WAFER_CTC_INT32, WAFER_CTC_FP16,  WAFER_CTC_BF16,  WAFER_CTC_FP32,
};

static const uint8_t wafer_ctc_type_bytes[7] = {1U, 2U, 4U, 2U, 2U, 4U, 4U};

static uint32_t wafer_ctc_zero_point_route(uint32_t route) {
  return route < 4U;
}

static uint32_t wafer_ctc_plain_route(uint32_t route) {
  switch (route) {
  case 4U:
  case 12U:
  case 15U:
  case 16U:
  case 17U:
  case 22U:
  case 23U:
  case 33U:
  case 35U:
    return 1U;
  default:
    return 0U;
  }
}

static uint32_t wafer_ctc_rounding_route(uint32_t route) {
  return wafer_ctc_zero_point_route(route) == 0U &&
         wafer_ctc_plain_route(route) == 0U;
}

static uint32_t wafer_ctc_extrema_disposition(uint32_t source_type,
                                               uint32_t destination_type) {
  if (destination_type <= WAFER_CTC_INT32)
    return WAFER_CTC_BOARD_OBSERVED;
  if (destination_type == WAFER_CTC_FP16 &&
      (source_type == WAFER_CTC_INT32 ||
       source_type == WAFER_CTC_BF16 ||
       source_type == WAFER_CTC_FP32 ||
       source_type == WAFER_CTC_TF32))
    return WAFER_CTC_BOARD_OBSERVED;
  if (destination_type == WAFER_CTC_BF16 &&
      (source_type == WAFER_CTC_FP32 ||
       source_type == WAFER_CTC_TF32))
    return WAFER_CTC_BOARD_OBSERVED;
  if (destination_type == WAFER_CTC_TF32 &&
      source_type == WAFER_CTC_FP32)
    return WAFER_CTC_BOARD_OBSERVED;
  return WAFER_CTC_BOARD_EXACT;
}

static void wafer_ctc_cache_range(uint64_t begin, uint32_t bytes,
                                  uint32_t invalidate_only) {
  enum {
    WAFER_CTC_SUPERVISOR_MODE = 1,
    WAFER_CTC_MACHINE_MODE = 3,
    WAFER_CTC_CACHE_LINE_BYTES = 64,
  };
  uintptr_t mode;
  __asm__ volatile("fence" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
  __asm__ volatile("csrr %0, mxstatus" : "=r"(mode));
  mode = (mode >> 30) & 3U;
  for (uintptr_t address = (uintptr_t)begin; address < (uintptr_t)begin + bytes;
       address += WAFER_CTC_CACHE_LINE_BYTES) {
    if (mode == WAFER_CTC_MACHINE_MODE) {
      if (invalidate_only != 0U)
        __asm__ volatile("dcache.ipa %0" : : "r"(address) : "memory");
      else
        __asm__ volatile("dcache.cipa %0" : : "r"(address) : "memory");
    } else if (mode == WAFER_CTC_SUPERVISOR_MODE) {
      if (invalidate_only != 0U)
        __asm__ volatile("dcache.iva %0" : : "r"(address) : "memory");
      else
        __asm__ volatile("dcache.civa %0" : : "r"(address) : "memory");
    }
  }
  __asm__ volatile("sync.is" ::: "memory");
  __asm__ volatile("fence" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
}

static uint32_t wafer_ctc_read32(uint32_t offset) {
  return *(const volatile uint32_t *)(uintptr_t)(WAFER_CTC_PMU_BASE + offset);
}

static uint64_t wafer_ctc_read64(uint32_t low_offset) {
  uint32_t low = 0;
  uint32_t high_after = 0;
  for (uint32_t retry = 0; retry < WAFER_CTC_STABLE_RETRIES; ++retry) {
    uint32_t high_before = wafer_ctc_read32(low_offset + 4U);
    low = wafer_ctc_read32(low_offset);
    high_after = wafer_ctc_read32(low_offset + 4U);
    if (high_before == high_after)
      break;
  }
  return ((uint64_t)high_after << 32) | low;
}

static WaferCTCPMU wafer_ctc_read_pmu(void) {
  WaferCTCPMU result;
  result.instructions = wafer_ctc_read32(GR_PMU_CT_INST_NUMS);
  result.blocking = wafer_ctc_read32(GR_PMU_CT_BLOCKING_TIME);
  result.execution = wafer_ctc_read64(GR_PMU_CT_EXE_TIME);
  return result;
}

static uint32_t wafer_ctc_decode(const volatile uint64_t *request,
                                 WaferCTCCase *selected) {
  if (request[WAFER_CTC_REQ_MAGIC] != WAFER_CTC_REQUEST_MAGIC ||
      request[WAFER_CTC_REQ_WORD_COUNT] !=
          (WAFER_CTC_REQUEST_WORDS) ||
      request[WAFER_CTC_REQ_RESOURCE_BYTES] != WAFER_CTC_RESOURCE_BYTES ||
      request[WAFER_CTC_REQ_SLOT_BYTES] != WAFER_CTC_SLOT_BYTES ||
      request[WAFER_CTC_REQ_BODY_OFFSET] != WAFER_CTC_BODY_OFFSET ||
      request[WAFER_CTC_REQ_GUARD] != WAFER_CTC_REQUEST_GUARD)
    return WAFER_CTC_STATUS_BAD_REQUEST;

  uint32_t opcode = (uint32_t)request[WAFER_CTC_REQ_OPCODE];
  uint32_t case_id = (uint32_t)request[WAFER_CTC_REQ_CASE];
  if (opcode < 139U || opcode > 174U)
    return WAFER_CTC_STATUS_BAD_REQUEST;
  uint32_t route = opcode - 139U;
  uint32_t elements = (uint32_t)request[WAFER_CTC_REQ_ELEMENTS];
  if (elements != WAFER_CTC_MAIN_ELEMENTS &&
      elements != WAFER_CTC_TAIL_ELEMENTS)
    return WAFER_CTC_STATUS_BAD_REQUEST;
  uint32_t source_type = wafer_ctc_source_types[route];
  uint32_t destination_type = wafer_ctc_destination_types[route];
  uint32_t domain = (uint32_t)request[WAFER_CTC_REQ_DOMAIN];
  uint32_t rounding_mode = (uint32_t)request[WAFER_CTC_REQ_ROUNDING];
  uint32_t zero_point = (uint32_t)request[WAFER_CTC_REQ_ZERO_POINT];
  uint32_t disposition =
      (uint32_t)request[WAFER_CTC_REQ_DISPOSITION];
  uint32_t expected_case = 0U;
  uint32_t expected_disposition = WAFER_CTC_BOARD_EXACT;
  if (domain == WAFER_CTC_DOMAIN_NORMAL) {
    expected_case = route * 2U +
                    (elements == WAFER_CTC_TAIL_ELEMENTS);
    if (rounding_mode != RND_NEAREST_EVEN || zero_point != 0U)
      return WAFER_CTC_STATUS_BAD_REQUEST;
  } else if (domain == WAFER_CTC_DOMAIN_DIRECTED) {
    if (wafer_ctc_rounding_route(route) == 0U ||
        elements != WAFER_CTC_MAIN_ELEMENTS ||
        rounding_mode < RND_ZERO || rounding_mode > RND_NEG_INF ||
        zero_point != 0U)
      return WAFER_CTC_STATUS_BAD_REQUEST;
    expected_case = 1000U + route * 3U + rounding_mode - 1U;
  } else if (domain == WAFER_CTC_DOMAIN_ZERO_POINT) {
    if (wafer_ctc_zero_point_route(route) == 0U ||
        elements != WAFER_CTC_MAIN_ELEMENTS ||
        rounding_mode != RND_NEAREST_EVEN || zero_point != 7U)
      return WAFER_CTC_STATUS_BAD_REQUEST;
    expected_case = 2000U + route;
    expected_disposition = WAFER_CTC_BOARD_OBSERVED;
  } else if (domain == WAFER_CTC_DOMAIN_EXTREMA) {
    if (elements != WAFER_CTC_MAIN_ELEMENTS ||
        rounding_mode != RND_NEAREST_EVEN || zero_point != 0U)
      return WAFER_CTC_STATUS_BAD_REQUEST;
    expected_case = 3000U + route;
    expected_disposition =
        wafer_ctc_extrema_disposition(source_type, destination_type);
  } else if (domain == WAFER_CTC_DOMAIN_STOCHASTIC) {
    if (wafer_ctc_rounding_route(route) == 0U ||
        elements != WAFER_CTC_MAIN_ELEMENTS ||
        rounding_mode != RND_STOCHASTIC || zero_point != 0U)
      return WAFER_CTC_STATUS_BAD_REQUEST;
    expected_case = 4000U + route;
    expected_disposition = WAFER_CTC_BOARD_OBSERVED;
  } else {
    return WAFER_CTC_STATUS_UNSUPPORTED_CASE;
  }
  uint32_t input_bytes = elements * wafer_ctc_type_bytes[source_type];
  uint32_t result_bytes = elements * wafer_ctc_type_bytes[destination_type];
  uint32_t output_span = (result_bytes + 255U) & ~UINT32_C(255);
  if (request[WAFER_CTC_REQ_SRC_TYPE] != source_type ||
      request[WAFER_CTC_REQ_DST_TYPE] != destination_type ||
      request[WAFER_CTC_REQ_INPUT_BYTES] != input_bytes ||
      request[WAFER_CTC_REQ_RESULT_BYTES] != result_bytes ||
      request[WAFER_CTC_REQ_OUTPUT_SPAN] != output_span ||
      case_id != expected_case ||
      disposition != expected_disposition ||
      WAFER_CTC_BODY_OFFSET + input_bytes > WAFER_CTC_SLOT_BYTES ||
      WAFER_CTC_BODY_OFFSET + output_span > WAFER_CTC_SLOT_BYTES)
    return WAFER_CTC_STATUS_BAD_REQUEST;

  selected->case_id = case_id;
  selected->opcode = opcode;
  selected->source_type = source_type;
  selected->destination_type = destination_type;
  selected->elements = elements;
  selected->input_bytes = input_bytes;
  selected->result_bytes = result_bytes;
  selected->output_span = output_span;
  selected->rounding_mode = rounding_mode;
  selected->zero_point = zero_point;
  selected->domain = domain;
  selected->disposition = disposition;
  return WAFER_CTC_STATUS_OK;
}

#define WAFER_CTC_DISPATCH(OPCODE, SYMBOL)                                     \
  case OPCODE:                                                                 \
    SYMBOL(source, destination, selected->elements, selected->zero_point,      \
           selected->rounding_mode, 0U);                                       \
    break

static uint32_t wafer_ctc_issue(const WaferCTCCase *selected) {
  uint64_t source = WAFER_CTC_SPM_INPUT + WAFER_CTC_BODY_OFFSET;
  uint64_t destination = WAFER_CTC_SPM_OUTPUT + WAFER_CTC_BODY_OFFSET;
  switch (selected->opcode) {
    WAFER_CTC_DISPATCH(139U, wafer_tx81_convert_int8_fp16);
    WAFER_CTC_DISPATCH(140U, wafer_tx81_convert_int8_bf16);
    WAFER_CTC_DISPATCH(141U, wafer_tx81_convert_int8_fp32);
    WAFER_CTC_DISPATCH(142U, wafer_tx81_convert_int8_tf32);
    WAFER_CTC_DISPATCH(143U, wafer_tx81_convert_int16_fp16);
    WAFER_CTC_DISPATCH(144U, wafer_tx81_convert_int16_bf16);
    WAFER_CTC_DISPATCH(145U, wafer_tx81_convert_int16_fp32);
    WAFER_CTC_DISPATCH(146U, wafer_tx81_convert_int16_tf32);
    WAFER_CTC_DISPATCH(147U, wafer_tx81_convert_int32_fp16);
    WAFER_CTC_DISPATCH(148U, wafer_tx81_convert_int32_bf16);
    WAFER_CTC_DISPATCH(149U, wafer_tx81_convert_int32_fp32);
    WAFER_CTC_DISPATCH(150U, wafer_tx81_convert_int32_tf32);
    WAFER_CTC_DISPATCH(151U, wafer_tx81_convert_bf16_int8);
    WAFER_CTC_DISPATCH(152U, wafer_tx81_convert_bf16_int16);
    WAFER_CTC_DISPATCH(153U, wafer_tx81_convert_bf16_int32);
    WAFER_CTC_DISPATCH(154U, wafer_tx81_convert_bf16_fp16);
    WAFER_CTC_DISPATCH(155U, wafer_tx81_convert_bf16_fp32);
    WAFER_CTC_DISPATCH(156U, wafer_tx81_convert_bf16_tf32);
    WAFER_CTC_DISPATCH(157U, wafer_tx81_convert_fp16_int8);
    WAFER_CTC_DISPATCH(158U, wafer_tx81_convert_fp16_int16);
    WAFER_CTC_DISPATCH(159U, wafer_tx81_convert_fp16_int32);
    WAFER_CTC_DISPATCH(160U, wafer_tx81_convert_fp16_bf16);
    WAFER_CTC_DISPATCH(161U, wafer_tx81_convert_fp16_fp32);
    WAFER_CTC_DISPATCH(162U, wafer_tx81_convert_fp16_tf32);
    WAFER_CTC_DISPATCH(163U, wafer_tx81_convert_fp32_int8);
    WAFER_CTC_DISPATCH(164U, wafer_tx81_convert_fp32_int16);
    WAFER_CTC_DISPATCH(165U, wafer_tx81_convert_fp32_int32);
    WAFER_CTC_DISPATCH(166U, wafer_tx81_convert_fp32_fp16);
    WAFER_CTC_DISPATCH(167U, wafer_tx81_convert_fp32_bf16);
    WAFER_CTC_DISPATCH(168U, wafer_tx81_convert_fp32_tf32);
    WAFER_CTC_DISPATCH(169U, wafer_tx81_convert_tf32_int8);
    WAFER_CTC_DISPATCH(170U, wafer_tx81_convert_tf32_int16);
    WAFER_CTC_DISPATCH(171U, wafer_tx81_convert_tf32_int32);
    WAFER_CTC_DISPATCH(172U, wafer_tx81_convert_tf32_fp16);
    WAFER_CTC_DISPATCH(173U, wafer_tx81_convert_tf32_bf16);
    WAFER_CTC_DISPATCH(174U, wafer_tx81_convert_tf32_fp32);
  default:
    return 0U;
  }
  return 1U;
}

#undef WAFER_CTC_DISPATCH

static void wafer_ctc_init_record(volatile uint64_t *record, uint32_t status) {
  for (uint32_t index = 0; index < WAFER_CTC_RECORD_WORDS; ++index)
    record[index] = 0;
  record[WAFER_CTC_REC_MAGIC] = WAFER_CTC_RECORD_MAGIC;
  record[WAFER_CTC_REC_WORD_COUNT] =
      WAFER_CTC_RECORD_WORDS;
  record[WAFER_CTC_REC_STATUS] = status;
  record[WAFER_CTC_REC_OUTPUT_DDR_OFFSET] = WAFER_CTC_OUTPUT_DDR_OFFSET;
  record[WAFER_CTC_REC_SLOT_BYTES] = WAFER_CTC_SLOT_BYTES;
  record[WAFER_CTC_REC_BODY_OFFSET] = WAFER_CTC_BODY_OFFSET;
  record[WAFER_CTC_REC_RECORD_GUARD] = WAFER_CTC_RECORD_GUARD;
}

__attribute__((visibility("hidden"))) void
wafer_tx81_instruction_family_probe(uint64_t request_ddr, uint64_t payload_ddr,
                                    uint64_t output_ddr) {
  wafer_ctc_cache_range(request_ddr, WAFER_CTC_RESOURCE_BYTES, 1U);
  wafer_ctc_cache_range(payload_ddr, WAFER_CTC_RESOURCE_BYTES, 1U);
  const volatile uint64_t *request =
      (const volatile uint64_t *)(uintptr_t)request_ddr;
  volatile uint64_t *record = (volatile uint64_t *)(uintptr_t)output_ddr;
  WaferCTCCase selected = {0};
  uint32_t status = wafer_ctc_decode(request, &selected);
  wafer_ctc_init_record(record, status);
  if (status == WAFER_CTC_STATUS_OK) {
    record[WAFER_CTC_REC_CASE] = selected.case_id;
    record[WAFER_CTC_REC_OPCODE] = selected.opcode;
    record[WAFER_CTC_REC_SRC_TYPE] = selected.source_type;
    record[WAFER_CTC_REC_DST_TYPE] = selected.destination_type;
    record[WAFER_CTC_REC_ELEMENTS] = selected.elements;
    record[WAFER_CTC_REC_INPUT_BYTES] = selected.input_bytes;
    record[WAFER_CTC_REC_RESULT_BYTES] = selected.result_bytes;
    record[WAFER_CTC_REC_OUTPUT_SPAN] = selected.output_span;
    record[WAFER_CTC_REC_ROUNDING] = selected.rounding_mode;
    record[WAFER_CTC_REC_SAMPLE] = request[WAFER_CTC_REQ_SAMPLE];
    record[WAFER_CTC_REC_DOMAIN] = selected.domain;
    record[WAFER_CTC_REC_ZERO_POINT] = selected.zero_point;
    record[WAFER_CTC_REC_DISPOSITION] = selected.disposition;
    record[WAFER_CTC_REC_REQUEST_GUARD] = request[WAFER_CTC_REQ_GUARD];

    WaferCTCPMU before = wafer_ctc_read_pmu();
    wafer_tx81_rdma(payload_ddr, WAFER_CTC_SPM_INPUT, WAFER_CTC_SLOT_BYTES,
                    WAFER_CTC_SLOT_BYTES, 0, 0, 0, 1, 1, 1, Fmt_UINT8, 0U);
    /*
     * The request resource's second slot is host-initialized canary data.
     * Seed the whole output through NCC so setup, convert, and readback remain
     * one dependency-ordered issue window.
     */
    wafer_tx81_rdma(request_ddr + WAFER_CTC_SLOT_BYTES,
                    WAFER_CTC_SPM_OUTPUT, WAFER_CTC_SLOT_BYTES,
                    WAFER_CTC_SLOT_BYTES, 0, 0, 0, 1, 1, 1, Fmt_UINT8, 0U);
    if (wafer_ctc_issue(&selected) == 0U) {
      status = WAFER_CTC_STATUS_EXECUTE_FAILED;
    } else {
      wafer_tx81_wdma(WAFER_CTC_SPM_OUTPUT,
                      output_ddr + WAFER_CTC_OUTPUT_DDR_OFFSET,
                      WAFER_CTC_SLOT_BYTES, WAFER_CTC_SLOT_BYTES, 0, 0, 0, 1, 1,
                      1, Fmt_UINT8, 0U);
      wafer_tx81_ncc_join(1U);
      WaferCTCPMU after = wafer_ctc_read_pmu();
      record[WAFER_CTC_REC_CT_INST_DELTA] =
          (uint32_t)(after.instructions - before.instructions);
      record[WAFER_CTC_REC_CT_EXEC_DELTA] = after.execution - before.execution;
      record[WAFER_CTC_REC_CT_BLOCKING_DELTA] =
          (uint32_t)(after.blocking - before.blocking);
    }
    record[WAFER_CTC_REC_STATUS] = status;
  }
  wafer_ctc_cache_range(output_ddr, WAFER_CTC_RECORD_WORDS * sizeof(uint64_t),
                        0U);
}
