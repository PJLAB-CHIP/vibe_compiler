#include "instr_adapter.h"
#include "instr_def.h"
#include "wafer_ct_vuvloop_semantics_protocol.h"
#include "wafer_tx81_crt.h"

#include <stdint.h>

typedef struct WaferCTVLSelection {
  uint32_t case_id;
  uint32_t disposition;
  uint32_t elem_count;
  uint32_t unit_elem_count;
  uint32_t full_elem_count;
  uint32_t full_unit_elem_count;
} WaferCTVLSelection;

static void wafer_ctvl_cache_range(uint64_t begin, uint32_t bytes,
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

static void wafer_ctvl_invalidate(uint64_t begin, uint32_t bytes) {
  wafer_ctvl_cache_range(begin, bytes, 1);
}

static void wafer_ctvl_publish(volatile uint64_t *record) {
  wafer_ctvl_cache_range((uint64_t)(uintptr_t)record,
                         WAFER_CTVL_RECORD_WORDS * sizeof(uint64_t), 0);
}

static uint32_t wafer_ctvl_decode(const volatile uint64_t *request,
                                  WaferCTVLSelection *selected) {
  if (request[WAFER_CTVL_REQ_MAGIC] != WAFER_CTVL_REQUEST_MAGIC ||
      request[WAFER_CTVL_REQ_SCHEMA_AND_WORDS] !=
          (((uint64_t)WAFER_CTVL_SCHEMA << 32) |
           WAFER_CTVL_REQUEST_WORDS) ||
      request[WAFER_CTVL_REQ_OPCODE] != WAFER_CTVL_OPCODE ||
      request[WAFER_CTVL_REQ_DTYPE] != WAFER_CTVL_DTYPE ||
      request[WAFER_CTVL_REQ_LHS_OWNED_BYTES] !=
          WAFER_CTVL_LHS_OWNED_BYTES ||
      request[WAFER_CTVL_REQ_RHS_OWNED_BYTES] !=
          WAFER_CTVL_RHS_OWNED_BYTES ||
      request[WAFER_CTVL_REQ_OUTPUT_CAPTURE_BYTES] !=
          WAFER_CTVL_OUTPUT_CAPTURE_BYTES ||
      request[WAFER_CTVL_REQ_GUARD_BYTES] != WAFER_CTVL_GUARD_BYTES ||
      request[WAFER_CTVL_REQ_GUARD] != WAFER_CTVL_REQUEST_GUARD)
    return WAFER_CTVL_STATUS_BAD_REQUEST;

  selected->case_id = (uint32_t)request[WAFER_CTVL_REQ_CASE];
  switch (selected->case_id) {
  case WAFER_CTVL_RECTANGULAR_CONTROL:
    selected->disposition = WAFER_CTVL_EXACT_CONTROL;
    selected->elem_count = 128U;
    selected->unit_elem_count = 64U;
    selected->full_elem_count = 384U;
    selected->full_unit_elem_count = 192U;
    break;
  case WAFER_CTVL_SUPPORTED_OUTER2_CONTROL:
    selected->disposition = WAFER_CTVL_EXACT_CONTROL;
    selected->elem_count = 192U;
    selected->unit_elem_count = 64U;
    selected->full_elem_count = 384U;
    selected->full_unit_elem_count = 128U;
    break;
  default:
    return WAFER_CTVL_STATUS_UNSUPPORTED_CASE;
  }

  if (selected->unit_elem_count != 64U ||
      (uint64_t)selected->full_elem_count *
              selected->unit_elem_count !=
          (uint64_t)selected->elem_count *
              selected->full_unit_elem_count ||
      request[WAFER_CTVL_REQ_DISPOSITION] != selected->disposition ||
      request[WAFER_CTVL_REQ_ELEM_COUNT] != selected->elem_count ||
      request[WAFER_CTVL_REQ_UNIT_ELEM_COUNT] !=
          selected->unit_elem_count ||
      request[WAFER_CTVL_REQ_FULL_ELEM_COUNT] !=
          selected->full_elem_count ||
      request[WAFER_CTVL_REQ_FULL_UNIT_ELEM_COUNT] !=
          selected->full_unit_elem_count)
    return WAFER_CTVL_STATUS_BAD_REQUEST;
  return WAFER_CTVL_STATUS_OK;
}

static uint32_t wafer_ctvl_output_physical_span(
    const WaferCTVLSelection *selected) {
  uint32_t bytes =
      selected->full_elem_count * WAFER_CTVL_ELEMENT_BYTES;
  return (bytes + WAFER_CTVL_PHYSICAL_BLOCK_BYTES - 1U) &
         ~(WAFER_CTVL_PHYSICAL_BLOCK_BYTES - 1U);
}

static uint64_t wafer_ctvl_issue(const WaferCTVLSelection *selected) {
  CT_Param instruction = {0};
  uint32_t lhs =
      (uint32_t)(WAFER_CTVL_SPM_LHS + WAFER_CTVL_BODY_OFFSET);
  uint32_t rhs =
      (uint32_t)(WAFER_CTVL_SPM_RHS + WAFER_CTVL_BODY_OFFSET);
  uint32_t output =
      (uint32_t)(WAFER_CTVL_SPM_OUTPUT + WAFER_CTVL_BODY_OFFSET);
  instruction.inter_type = I_CGRA | I_WORKER0;
  instruction.ctrl.rnd_mode = RND_NEAREST_EVEN;
  instruction.ctrl.src0_format = Fmt_FP16;
  instruction.ctrl.opcode = WAFER_CTVL_OPCODE;
  instruction.param.src0 = lhs;
  instruction.param.src1 = rhs;
  instruction.param.dst0 = output;
  instruction.param.src0_end =
      lhs + selected->full_elem_count * WAFER_CTVL_ELEMENT_BYTES - 1U;
  instruction.param.src1_end =
      rhs + selected->full_unit_elem_count * WAFER_CTVL_ELEMENT_BYTES - 1U;
  instruction.param.dst0_end =
      output + wafer_ctvl_output_physical_span(selected) - 1U;
  instruction.param.elem_count = selected->elem_count;
  instruction.param.unit_elem_count = selected->unit_elem_count;
  instruction.param.full_elem_count = selected->full_elem_count;
  instruction.param.full_unit_elem_count =
      selected->full_unit_elem_count;
  return TsmExecute(&instruction);
}

static void wafer_ctvl_init_record(volatile uint64_t *record,
                                   uint32_t status) {
  for (uint32_t index = 0; index < WAFER_CTVL_RECORD_WORDS; ++index)
    record[index] = 0;
  record[WAFER_CTVL_REC_MAGIC] = WAFER_CTVL_RECORD_MAGIC;
  record[WAFER_CTVL_REC_SCHEMA_AND_WORDS] =
      ((uint64_t)WAFER_CTVL_SCHEMA << 32) |
      WAFER_CTVL_RECORD_WORDS;
  record[WAFER_CTVL_REC_STATUS] = status;
  record[WAFER_CTVL_REC_OUTPUT_DDR_OFFSET] =
      WAFER_CTVL_OUTPUT_DDR_OFFSET;
  record[WAFER_CTVL_REC_OUTPUT_SLOT_BYTES] =
      WAFER_CTVL_OUTPUT_SLOT_BYTES;
  record[WAFER_CTVL_REC_BODY_OFFSET] = WAFER_CTVL_BODY_OFFSET;
  record[WAFER_CTVL_REC_OUTPUT_OWNED_BYTES] =
      WAFER_CTVL_OUTPUT_CAPTURE_BYTES;
  record[WAFER_CTVL_REC_RESOURCE_BYTES] = WAFER_CTVL_RESOURCE_BYTES;
  record[WAFER_CTVL_REC_RECORD_GUARD] = WAFER_CTVL_RECORD_GUARD;
}

__attribute__((visibility("hidden"))) void
wafer_tx81_instruction_family_probe(uint64_t request_ddr,
                                    uint64_t payload_ddr,
                                    uint64_t output_ddr) {
  wafer_ctvl_invalidate(request_ddr, WAFER_CTVL_REQUEST_WORDS *
                                        sizeof(uint64_t));
  const volatile uint64_t *request =
      (const volatile uint64_t *)(uintptr_t)request_ddr;
  volatile uint64_t *record =
      (volatile uint64_t *)(uintptr_t)output_ddr;
  WaferCTVLSelection selected = {0};
  uint32_t status = wafer_ctvl_decode(request, &selected);
  wafer_ctvl_init_record(record, status);

  if (status == WAFER_CTVL_STATUS_OK) {
    record[WAFER_CTVL_REC_CASE] = selected.case_id;
    record[WAFER_CTVL_REC_DISPOSITION] = selected.disposition;
    record[WAFER_CTVL_REC_OPCODE] = WAFER_CTVL_OPCODE;
    record[WAFER_CTVL_REC_DTYPE] = WAFER_CTVL_DTYPE;
    record[WAFER_CTVL_REC_ELEM_COUNT] = selected.elem_count;
    record[WAFER_CTVL_REC_UNIT_ELEM_COUNT] =
        selected.unit_elem_count;
    record[WAFER_CTVL_REC_FULL_ELEM_COUNT] =
        selected.full_elem_count;
    record[WAFER_CTVL_REC_FULL_UNIT_ELEM_COUNT] =
        selected.full_unit_elem_count;
    record[WAFER_CTVL_REC_SAMPLE] =
        request[WAFER_CTVL_REQ_SAMPLE];
    record[WAFER_CTVL_REC_REQUEST_GUARD] =
        request[WAFER_CTVL_REQ_GUARD];
    record[WAFER_CTVL_REC_OUTPUT_PHYSICAL_SPAN] =
        wafer_ctvl_output_physical_span(&selected);

    wafer_tx81_rdma_v3(payload_ddr + WAFER_CTVL_PAYLOAD_LHS_OFFSET,
                    WAFER_CTVL_SPM_LHS, WAFER_CTVL_LHS_SLOT_BYTES,
                    WAFER_CTVL_LHS_SLOT_BYTES, 0, 0, 0, 1, 1, 1,
                    Fmt_UINT8, 0U);
    wafer_tx81_rdma_v3(payload_ddr + WAFER_CTVL_PAYLOAD_RHS_OFFSET,
                    WAFER_CTVL_SPM_RHS, WAFER_CTVL_RHS_SLOT_BYTES,
                    WAFER_CTVL_RHS_SLOT_BYTES, 0, 0, 0, 1, 1, 1,
                    Fmt_UINT8, 0U);
    wafer_tx81_rdma_v3(
        payload_ddr + WAFER_CTVL_PAYLOAD_OUTPUT_SEED_OFFSET,
        WAFER_CTVL_SPM_OUTPUT, WAFER_CTVL_OUTPUT_SLOT_BYTES,
        WAFER_CTVL_OUTPUT_SLOT_BYTES, 0, 0, 0, 1, 1, 1,
        Fmt_UINT8, 0U);
    uint64_t execute_result = wafer_ctvl_issue(&selected);
    record[WAFER_CTVL_REC_EXECUTE_RESULT] = execute_result;
    wafer_tx81_wdma_v3(WAFER_CTVL_SPM_OUTPUT,
                    output_ddr + WAFER_CTVL_OUTPUT_DDR_OFFSET,
                    WAFER_CTVL_OUTPUT_SLOT_BYTES,
                    WAFER_CTVL_OUTPUT_SLOT_BYTES, 0, 0, 0, 1, 1, 1,
                    Fmt_UINT8, 0U);
    wafer_tx81_ncc_join(1U);
    record[WAFER_CTVL_REC_COMPLETION_SEEN] = 1;
    record[WAFER_CTVL_REC_TERMINAL_FENCE_COUNT] = 1;
    if (execute_result == 0)
      status = WAFER_CTVL_STATUS_EXECUTE_REJECTED;
    record[WAFER_CTVL_REC_STATUS] = status;
  }
  wafer_ctvl_publish(record);
}
