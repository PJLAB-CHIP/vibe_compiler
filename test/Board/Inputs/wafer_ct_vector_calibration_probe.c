#include "instr_adapter.h"
#include "instr_def.h"
#include "wafer_ct_vector_calibration_probe_protocol.h"
#include "wafer_tx81_crt.h"

#include <stddef.h>
#include <stdint.h>

extern int8_t *get_spm_memory_mapping(uint64_t offset);

typedef struct WaferCTVCase {
  uint32_t case_id;
  uint32_t disposition;
  uint32_t family;
  uint32_t dtype;
  uint32_t opcode;
  uint32_t result_bytes;
  uint32_t output_span;
  uint32_t elements;
  uint32_t input_a_bytes;
  uint32_t input_b_bytes;
  uint32_t scalar_bits;
  uint32_t unit_elements;
} WaferCTVCase;

static void wafer_ctv_cache_range(uint64_t begin, uint32_t bytes,
                                  uint32_t invalidate_only) {
  enum {
    WAFER_TX81_SUPERVISOR_MODE = 1,
    WAFER_TX81_MACHINE_MODE = 3,
    WAFER_TX81_CACHE_LINE_BYTES = 64,
  };
  uintptr_t mode;
  __asm__ volatile("fence" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
  __asm__ volatile("csrr %0, mxstatus" : "=r"(mode));
  mode = (mode >> 30) & 3U;
  for (uintptr_t address = (uintptr_t)begin;
       address < (uintptr_t)begin + bytes;
       address += WAFER_TX81_CACHE_LINE_BYTES) {
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

static void wafer_ctv_invalidate(uint64_t begin, uint32_t bytes) {
  wafer_ctv_cache_range(begin, bytes, 1);
}

static void wafer_ctv_publish(volatile uint64_t *record) {
  wafer_ctv_cache_range((uint64_t)(uintptr_t)record,
                        WAFER_CTV_RECORD_WORDS * sizeof(uint64_t), 0);
}

static uint32_t wafer_ctv_dtype_bytes(uint32_t dtype) {
  return dtype == WAFER_CTV_F32 ? 4U : 2U;
}

static uint32_t wafer_ctv_format(uint32_t dtype) {
  switch (dtype) {
  case WAFER_CTV_F16:
    return Fmt_FP16;
  case WAFER_CTV_BF16:
    return Fmt_BF16;
  case WAFER_CTV_F32:
    return Fmt_FP32;
  default:
    return Fmt_UNUSED;
  }
}

static uint32_t wafer_ctv_family(uint32_t opcode) {
  if (opcode <= 5U)
    return WAFER_CTV_UNARY;
  if (opcode <= 29U)
    return WAFER_CTV_BINARY;
  if (opcode <= 77U)
    return WAFER_CTV_RELATION;
  if (opcode <= 87U)
    return WAFER_CTV_LOGIC_VALUE;
  if (opcode <= 97U)
    return WAFER_CTV_LOGIC_BOOL;
  if (opcode <= 104U)
    return WAFER_CTV_TRANSCENDENTAL;
  if (opcode <= 110U)
    return WAFER_CTV_ACTIVATION;
  return UINT32_MAX;
}

static uint32_t wafer_ctv_disposition(uint32_t opcode) {
  if (opcode == 108U || opcode == 109U)
    return WAFER_CTV_BOARD_OBSERVED;
  if (opcode == 1U || opcode == 3U || opcode == 4U ||
      (opcode >= 98U && opcode <= 106U) || opcode == 110U)
    return WAFER_CTV_BOARD_TOLERANCE;
  return WAFER_CTV_BOARD_EXACT;
}

static uint32_t wafer_ctv_bool_output(uint32_t opcode) {
  return ((opcode >= 30U && opcode <= 77U &&
           ((opcode - 30U) % 8U == 1U ||
            (opcode - 30U) % 8U == 3U ||
            (opcode - 30U) % 8U == 6U ||
            (opcode - 30U) % 8U == 7U)) ||
          (opcode >= 88U && opcode <= 97U));
}

static uint32_t wafer_ctv_vs(uint32_t opcode) {
  return ((opcode >= 6U && opcode <= 29U &&
           (opcode - 6U) % 4U == 1U) ||
          (opcode >= 30U && opcode <= 77U &&
           ((opcode - 30U) % 8U == 2U ||
            (opcode - 30U) % 8U == 3U)));
}

static uint32_t wafer_ctv_vuv(uint32_t opcode) {
  return ((opcode >= 6U && opcode <= 29U &&
           ((opcode - 6U) % 4U == 2U ||
            (opcode - 6U) % 4U == 3U)) ||
          (opcode >= 30U && opcode <= 77U &&
           (opcode - 30U) % 8U >= 4U) ||
          (opcode >= 82U && opcode <= 87U) ||
          (opcode >= 92U && opcode <= 97U));
}

static uint32_t wafer_ctv_has_second_operand(uint32_t opcode) {
  if (opcode <= 5U || opcode == 78U || opcode == 88U ||
      (opcode >= 98U && opcode <= 110U))
    return 0;
  return wafer_ctv_vs(opcode) == 0;
}

static uint32_t wafer_ctv_scalar_bits(uint32_t dtype) {
  return dtype == WAFER_CTV_F32 ? UINT32_C(0x40000000)
                                : UINT32_C(0x4000);
}

static uint32_t wafer_ctv_decode(const volatile uint64_t *request,
                                 WaferCTVCase *decoded) {
  if (request[WAFER_CTV_REQ_MAGIC] != WAFER_CTV_REQUEST_MAGIC ||
      request[WAFER_CTV_REQ_SCHEMA_AND_WORDS] !=
          (((uint64_t)WAFER_CTV_SCHEMA << 32) |
           WAFER_CTV_REQUEST_WORDS) ||
      request[WAFER_CTV_REQ_GUARD] != WAFER_CTV_REQUEST_GUARD ||
      request[WAFER_CTV_REQ_RESOURCE_BYTES] !=
          WAFER_CTV_RESOURCE_BYTES ||
      request[WAFER_CTV_REQ_SLOT_BYTES] != WAFER_CTV_SLOT_BYTES ||
      request[WAFER_CTV_REQ_BODY_OFFSET] != WAFER_CTV_BODY_OFFSET)
    return WAFER_CTV_STATUS_BAD_REQUEST;

  uint32_t opcode = (uint32_t)request[WAFER_CTV_REQ_OPCODE];
  uint32_t dtype = (uint32_t)request[WAFER_CTV_REQ_DTYPE];
  uint32_t elements = (uint32_t)request[WAFER_CTV_REQ_ELEMENTS];
  if (opcode > 110U || dtype > WAFER_CTV_F32 ||
      (elements != WAFER_CTV_MAIN_ELEMENTS &&
       elements != WAFER_CTV_TAIL_ELEMENTS) ||
      (opcode >= 88U && opcode <= 97U && dtype != WAFER_CTV_F16))
    return WAFER_CTV_STATUS_UNSUPPORTED_CASE;

  uint32_t shape = elements == WAFER_CTV_TAIL_ELEMENTS;
  uint32_t expected_case =
      WAFER_CTV_CASE_BASE + opcode * WAFER_CTV_CASE_STRIDE +
      dtype * 2U + shape;
  uint32_t dtype_bytes = wafer_ctv_dtype_bytes(dtype);
  uint32_t bool_output = wafer_ctv_bool_output(opcode);
  uint32_t result_bytes =
      bool_output != 0 ? (elements + 7U) / 8U : elements * dtype_bytes;
  uint32_t output_span = (result_bytes + 255U) & ~UINT32_C(255);
  uint32_t family = wafer_ctv_family(opcode);
  uint32_t disposition = wafer_ctv_disposition(opcode);
  if (request[WAFER_CTV_REQ_CASE] != expected_case ||
      request[WAFER_CTV_REQ_FAMILY] != family ||
      request[WAFER_CTV_REQ_DISPOSITION] != disposition ||
      request[WAFER_CTV_REQ_RESULT_BYTES] != result_bytes ||
      request[WAFER_CTV_REQ_OUTPUT_SPAN] != output_span ||
      output_span > WAFER_CTV_SLOT_BYTES - WAFER_CTV_BODY_OFFSET)
    return WAFER_CTV_STATUS_BAD_REQUEST;

  uint32_t bool_input = family == WAFER_CTV_LOGIC_BOOL;
  uint32_t input_a_bytes =
      bool_input != 0 ? (elements + 7U) / 8U : elements * dtype_bytes;
  uint32_t input_b_bytes = 0;
  if (wafer_ctv_has_second_operand(opcode) != 0) {
    uint32_t second_elements =
        wafer_ctv_vuv(opcode) != 0 ? WAFER_CTV_UNIT_ELEMENTS : elements;
    input_b_bytes = bool_input != 0 ? (second_elements + 7U) / 8U
                                   : second_elements * dtype_bytes;
  }
  if (input_a_bytes > WAFER_CTV_SLOT_BYTES ||
      input_b_bytes > WAFER_CTV_SLOT_BYTES)
    return WAFER_CTV_STATUS_UNSUPPORTED_CASE;

  decoded->case_id = expected_case;
  decoded->disposition = disposition;
  decoded->family = family;
  decoded->dtype = dtype;
  decoded->opcode = opcode;
  decoded->result_bytes = result_bytes;
  decoded->output_span = output_span;
  decoded->elements = elements;
  decoded->input_a_bytes = input_a_bytes;
  decoded->input_b_bytes = input_b_bytes;
  decoded->scalar_bits =
      wafer_ctv_vs(opcode) != 0 ? wafer_ctv_scalar_bits(dtype) : 0;
  decoded->unit_elements =
      wafer_ctv_vuv(opcode) != 0 ? WAFER_CTV_UNIT_ELEMENTS : 0;
  return WAFER_CTV_STATUS_OK;
}

static void wafer_ctv_fill_output(void) {
  volatile uint8_t *output = (volatile uint8_t *)(void *)
      get_spm_memory_mapping(WAFER_CTV_SPM_OUTPUT);
  for (uint32_t index = 0; index < WAFER_CTV_SLOT_BYTES; ++index)
    output[index] = WAFER_CTV_SLOT_CANARY;
}

static uint64_t wafer_ctv_guard_mismatches(const WaferCTVCase *selected) {
  const volatile uint8_t *output =
      (const volatile uint8_t *)(const void *)
          get_spm_memory_mapping(WAFER_CTV_SPM_OUTPUT);
  uint32_t allowed_begin = WAFER_CTV_BODY_OFFSET;
  uint32_t allowed_end = allowed_begin + selected->output_span;
  uint64_t mismatches = 0;
  for (uint32_t index = 0; index < WAFER_CTV_SLOT_BYTES; ++index)
    if ((index < allowed_begin || index >= allowed_end) &&
        output[index] != WAFER_CTV_SLOT_CANARY)
      ++mismatches;
  return mismatches;
}

static uint64_t wafer_ctv_issue(const WaferCTVCase *selected) {
  CT_Param instruction = {0};
  uint32_t output = (uint32_t)(WAFER_CTV_SPM_OUTPUT +
                               WAFER_CTV_BODY_OFFSET);
  instruction.inter_type = I_CGRA | I_WORKER0;
  instruction.ctrl.rnd_mode = RND_NEAREST_EVEN;
  instruction.ctrl.src0_format =
      (uint8_t)wafer_ctv_format(selected->dtype);
  instruction.ctrl.opcode = (uint8_t)selected->opcode;
  instruction.param.src0 = (uint32_t)WAFER_CTV_SPM_A;
  instruction.param.dst0 = output;
  instruction.param.elem_count = selected->elements;
  instruction.param.src0_end =
      (uint32_t)WAFER_CTV_SPM_A + selected->input_a_bytes - 1U;
  instruction.param.dst0_end =
      output + selected->output_span - 1U;
  if (wafer_ctv_vs(selected->opcode) != 0) {
    instruction.param.src1 = selected->scalar_bits;
  } else if (selected->input_b_bytes != 0) {
    instruction.param.src1 = (uint32_t)WAFER_CTV_SPM_B;
    instruction.param.src1_end =
        (uint32_t)WAFER_CTV_SPM_B + selected->input_b_bytes - 1U;
  }
  if (selected->unit_elements != 0) {
    instruction.param.unit_elem_count = selected->unit_elements;
    /*
     * The loop opcodes consume the same logical main/tail vector, while the
     * full fields make the dependency footprint explicit and nonzero.
     */
    instruction.param.full_elem_count = selected->elements;
    instruction.param.full_unit_elem_count =
        selected->unit_elements;
  }
  uint64_t result = TsmExecute(&instruction);
  (void)TsmWaitfinish_bywork(0);
  return result;
}

static void wafer_ctv_init_record(volatile uint64_t *record,
                                  uint32_t status) {
  for (uint32_t index = 0; index < WAFER_CTV_RECORD_WORDS; ++index)
    record[index] = 0;
  record[WAFER_CTV_REC_MAGIC] = WAFER_CTV_RECORD_MAGIC;
  record[WAFER_CTV_REC_SCHEMA_AND_WORDS] =
      ((uint64_t)WAFER_CTV_SCHEMA << 32) | WAFER_CTV_RECORD_WORDS;
  record[WAFER_CTV_REC_STATUS] = status;
  record[WAFER_CTV_REC_OUTPUT_DDR_OFFSET] =
      WAFER_CTV_OUTPUT_DDR_OFFSET;
  record[WAFER_CTV_REC_SLOT_BYTES] = WAFER_CTV_SLOT_BYTES;
  record[WAFER_CTV_REC_BODY_OFFSET] = WAFER_CTV_BODY_OFFSET;
  record[WAFER_CTV_REC_RECORD_GUARD] = WAFER_CTV_RECORD_GUARD;
}

__attribute__((visibility("hidden"))) void
wafer_tx81_instruction_family_probe(uint64_t request_ddr,
                                    uint64_t payload_ddr,
                                    uint64_t output_ddr) {
  wafer_ctv_invalidate(request_ddr, WAFER_CTV_RESOURCE_BYTES);
  wafer_ctv_invalidate(payload_ddr, WAFER_CTV_RESOURCE_BYTES);
  const volatile uint64_t *request =
      (const volatile uint64_t *)(uintptr_t)request_ddr;
  volatile uint64_t *record =
      (volatile uint64_t *)(uintptr_t)output_ddr;
  WaferCTVCase selected = {0};
  uint32_t status = wafer_ctv_decode(request, &selected);
  wafer_ctv_init_record(record, status);

  if (status == WAFER_CTV_STATUS_OK) {
    record[WAFER_CTV_REC_CASE] = selected.case_id;
    record[WAFER_CTV_REC_DISPOSITION] = selected.disposition;
    record[WAFER_CTV_REC_FAMILY] = selected.family;
    record[WAFER_CTV_REC_DTYPE] = selected.dtype;
    record[WAFER_CTV_REC_OPCODE] = selected.opcode;
    record[WAFER_CTV_REC_RESULT_BYTES] = selected.result_bytes;
    record[WAFER_CTV_REC_OUTPUT_SPAN] = selected.output_span;
    record[WAFER_CTV_REC_ELEMENTS] = selected.elements;
    record[WAFER_CTV_REC_SAMPLE] = request[WAFER_CTV_REQ_SAMPLE];
    record[WAFER_CTV_REC_REQUEST_GUARD] =
        request[WAFER_CTV_REQ_GUARD];
    record[WAFER_CTV_REC_INPUT_A_BYTES] =
        selected.input_a_bytes;
    record[WAFER_CTV_REC_INPUT_B_BYTES] =
        selected.input_b_bytes;
    record[WAFER_CTV_REC_SCALAR_BITS] = selected.scalar_bits;
    record[WAFER_CTV_REC_UNIT_ELEMENTS] =
        selected.unit_elements;

    wafer_tx81_rdma(payload_ddr, WAFER_CTV_SPM_A,
                    WAFER_CTV_SLOT_BYTES, WAFER_CTV_SLOT_BYTES,
                    0, 0, 0, 1, 1, 1, Fmt_UINT8);
    wafer_tx81_rdma(payload_ddr + WAFER_CTV_SLOT_BYTES,
                    WAFER_CTV_SPM_B, WAFER_CTV_SLOT_BYTES,
                    WAFER_CTV_SLOT_BYTES, 0, 0, 0, 1, 1, 1,
                    Fmt_UINT8);
    wafer_tx81_local_fence();
    wafer_ctv_fill_output();
    uint64_t execute_result = wafer_ctv_issue(&selected);
    record[WAFER_CTV_REC_EXECUTE_RESULT] = execute_result;
    if (execute_result == 0) {
      status = WAFER_CTV_STATUS_EXECUTE_FAILED;
    } else {
      record[WAFER_CTV_REC_OUTPUT_GUARD_MISMATCHES] =
          wafer_ctv_guard_mismatches(&selected);
      wafer_tx81_wdma(WAFER_CTV_SPM_OUTPUT,
                      output_ddr + WAFER_CTV_OUTPUT_DDR_OFFSET,
                      WAFER_CTV_SLOT_BYTES, WAFER_CTV_SLOT_BYTES,
                      0, 0, 0, 1, 1, 1, Fmt_UINT8);
      wafer_tx81_local_fence();
    }
    record[WAFER_CTV_REC_STATUS] = status;
  }
  wafer_ctv_publish(record);
}
