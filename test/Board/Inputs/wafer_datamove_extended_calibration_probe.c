#include "instr_adapter.h"
#include "instr_adapter_plat.h"
#include "instr_def.h"
#include "wafer_datamove_extended_calibration_probe_protocol.h"
#include "wafer_tx81_crt.h"

#include <stdint.h>

#define WAFER_DMX_PMU_BASE UINT64_C(0x590000)
#define WAFER_DMX_STABLE_RETRIES 8U

typedef struct WaferDMXCase {
  uint32_t case_id;
  uint32_t input_bytes;
  uint32_t result_bytes;
  uint32_t output_span;
  uint32_t tdma_instructions;
  uint32_t ct_instructions;
  uint32_t ne_instructions;
  uint32_t oracle;
} WaferDMXCase;

typedef struct WaferDMXPMU {
  uint32_t tdma_instructions;
  uint32_t ct_instructions;
  uint32_t ne_instructions;
  uint64_t tdma_execution;
  uint64_t ct_execution;
  uint64_t ne_execution;
} WaferDMXPMU;

static const WaferDMXCase wafer_dmx_cases[] = {
    {0U, 24576U, 16380U, 24576U, 0U, 1U, 0U, 1U},
    {1U, 17408U, 16380U, 17408U, 0U, 1U, 0U, 1U},
    {2U, 17920U, 16380U, 17920U, 0U, 1U, 0U, 1U},
    {4U, 9728U, 19456U, 19456U, 1U, 0U, 0U, 0U},
    {5U, 27136U, 88576U, 88576U, 1U, 0U, 0U, 0U},
    {6U, 4608U, 260U, 512U, 0U, 1U, 0U, 1U},
    {7U, 4608U, 260U, 512U, 0U, 1U, 0U, 1U},
    {8U, 16380U, 17408U, 17408U, 1U, 0U, 0U, 1U},
    {9U, 512U, 260U, 512U, 2U, 1U, 0U, 0U},
    {10U, 512U, 260U, 512U, 4U, 1U, 0U, 0U},
    {11U, 4608U, 32U, 256U, 2U, 0U, 1U, 0U},
    {12U, 0U, 8064U, 8192U, 1U, 0U, 0U, 0U},
    {13U, 0U, 8128U, 8192U, 1U, 0U, 0U, 0U},
    {14U, 0U, 4096U, 4096U, 1U, 0U, 0U, 0U},
    {15U, 0U, 4096U, 4096U, 1U, 0U, 0U, 0U},
    {16U, 0U, 4096U, 4096U, 1U, 0U, 0U, 0U},
    {17U, 0U, 4096U, 4096U, 1U, 0U, 0U, 0U},
};

static void wafer_dmx_cache_range(uint64_t begin, uint32_t bytes,
                                  uint32_t invalidate_only) {
  enum {
    WAFER_DMX_SUPERVISOR_MODE = 1,
    WAFER_DMX_MACHINE_MODE = 3,
    WAFER_DMX_CACHE_LINE_BYTES = 64,
  };
  uintptr_t mode;
  __asm__ volatile("fence" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
  __asm__ volatile("csrr %0, mxstatus" : "=r"(mode));
  mode = (mode >> 30) & 3U;
  for (uintptr_t address = (uintptr_t)begin;
       address < (uintptr_t)begin + bytes;
       address += WAFER_DMX_CACHE_LINE_BYTES) {
    if (mode == WAFER_DMX_MACHINE_MODE) {
      if (invalidate_only != 0U)
        __asm__ volatile("dcache.ipa %0" : : "r"(address) : "memory");
      else
        __asm__ volatile("dcache.cipa %0" : : "r"(address) : "memory");
    } else if (mode == WAFER_DMX_SUPERVISOR_MODE) {
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

static uint32_t wafer_dmx_read32(uint32_t offset) {
  return *(const volatile uint32_t *)(uintptr_t)(WAFER_DMX_PMU_BASE + offset);
}

static uint64_t wafer_dmx_read64(uint32_t low_offset) {
  uint32_t low = 0;
  uint32_t high_after = 0;
  for (uint32_t retry = 0; retry < WAFER_DMX_STABLE_RETRIES; ++retry) {
    uint32_t high_before = wafer_dmx_read32(low_offset + 4U);
    low = wafer_dmx_read32(low_offset);
    high_after = wafer_dmx_read32(low_offset + 4U);
    if (high_before == high_after)
      break;
  }
  return ((uint64_t)high_after << 32) | low;
}

static WaferDMXPMU wafer_dmx_read_pmu(void) {
  WaferDMXPMU result;
  result.tdma_instructions = wafer_dmx_read32(GR_PMU_TDMA_INST_NUMS);
  result.ct_instructions = wafer_dmx_read32(GR_PMU_CT_INST_NUMS);
  result.ne_instructions = wafer_dmx_read32(GR_PMU_NE_INST_NUMS);
  result.tdma_execution = wafer_dmx_read64(GR_PMU_TDMA_EXE_TIME);
  result.ct_execution = wafer_dmx_read64(GR_PMU_CT_EXE_TIME);
  result.ne_execution = wafer_dmx_read64(GR_PMU_NE_EXE_TIME);
  return result;
}

static uint32_t wafer_dmx_decode(const volatile uint64_t *request,
                                 WaferDMXCase *selected) {
  if (request[WAFER_DMX_REQ_MAGIC] != WAFER_DMX_REQUEST_MAGIC ||
      request[WAFER_DMX_REQ_WORD_COUNT] !=
          (WAFER_DMX_REQUEST_WORDS) ||
      request[WAFER_DMX_REQ_RESOURCE_BYTES] != WAFER_DMX_RESOURCE_BYTES ||
      request[WAFER_DMX_REQ_SLOT_BYTES] != WAFER_DMX_SLOT_BYTES ||
      request[WAFER_DMX_REQ_BODY_OFFSET] != WAFER_DMX_BODY_OFFSET ||
      request[WAFER_DMX_REQ_GUARD] != WAFER_DMX_REQUEST_GUARD)
    return WAFER_DMX_STATUS_BAD_REQUEST;
  uint32_t case_id = (uint32_t)request[WAFER_DMX_REQ_CASE];
  const WaferDMXCase *matched = 0;
  for (uint32_t index = 0U;
       index < sizeof(wafer_dmx_cases) / sizeof(wafer_dmx_cases[0]);
       ++index) {
    if (wafer_dmx_cases[index].case_id == case_id) {
      matched = &wafer_dmx_cases[index];
      break;
    }
  }
  if (matched == 0)
    return WAFER_DMX_STATUS_BAD_REQUEST;
  *selected = *matched;
  if (request[WAFER_DMX_REQ_INPUT_BYTES] != selected->input_bytes ||
      request[WAFER_DMX_REQ_RESULT_BYTES] != selected->result_bytes ||
      request[WAFER_DMX_REQ_OUTPUT_SPAN] != selected->output_span ||
      request[WAFER_DMX_REQ_TDMA_INSTRUCTIONS] !=
          selected->tdma_instructions ||
      request[WAFER_DMX_REQ_CT_INSTRUCTIONS] != selected->ct_instructions ||
      request[WAFER_DMX_REQ_NE_INSTRUCTIONS] != selected->ne_instructions ||
      request[WAFER_DMX_REQ_ORACLE] != selected->oracle ||
      WAFER_DMX_BODY_OFFSET + selected->input_bytes > WAFER_DMX_SLOT_BYTES ||
      WAFER_DMX_BODY_OFFSET + selected->output_span > WAFER_DMX_SLOT_BYTES)
    return WAFER_DMX_STATUS_BAD_REQUEST;
  return WAFER_DMX_STATUS_OK;
}

static Data_Shape wafer_dmx_shape(uint32_t n, uint32_t h, uint32_t w,
                                  uint32_t c) {
  Data_Shape result = {(uint16_t)n, (uint16_t)h, (uint16_t)w, (uint16_t)c};
  return result;
}

static void wafer_dmx_gather(uint64_t source, uint64_t destination,
                             uint32_t bytes, uint32_t inner_bytes,
                             uint32_t source_stride0,
                             uint32_t source_iteration0,
                             uint32_t destination_stride0,
                             uint32_t destination_iteration0) {
  wafer_tx81_gather_scatter(source, destination, bytes, inner_bytes,
                            source_stride0, 0U, 0U, source_iteration0, 1U, 1U,
                            destination_stride0, 0U, 0U,
                            destination_iteration0, 1U, 1U, 0U);
}

static uint64_t wafer_dmx_raw_concat(uint64_t source0, Data_Shape shape0,
                                     uint64_t source1, Data_Shape shape1,
                                     uint64_t destination,
                                     Data_Shape destination_shape,
                                     uint32_t dimension) {
  if (dimension > 2U)
    return 0U;
  TsmMoveInstr instruction = {0};
  TsmDataMove *move = TsmNewDataMove();
  move->Concat(&instruction, source0, shape0, source1, shape1, destination,
               destination_shape, dimension, Fmt_FP16);
  uint64_t result = TsmExecute(&instruction);
  TsmDeleteDataMove(move);
  return result;
}

static uint64_t wafer_dmx_raw_mask_gather(uint64_t source, uint32_t index,
                                          uint64_t destination,
                                          uint32_t bit_vector) {
  TsmMaskDataMoveInstr instruction = {0};
  TsmMaskDataMove *move = TsmNewMaskDataMove();
  if (bit_vector != 0U)
    move->MaskGather_bV(&instruction, source, index, destination, 130U,
                        Fmt_FP16);
  else
    move->MaskGather(&instruction, source, index, destination, 130U, Fmt_FP16);
  uint64_t result = TsmExecute(&instruction);
  TsmDeleteMaskDataMove(move);
  return result;
}

static uint64_t wafer_dmx_raw_tensor_nom(uint64_t source,
                                         uint64_t destination) {
  TsmDataMoveInstr instruction = {0};
  TsmDataMove *move = TsmNewDataMove();
  move->TensorNom(&instruction, source, wafer_dmx_shape(2U, 7U, 9U, 65U),
                  destination, wafer_dmx_shape(2U, 7U, 9U, 65U), Fmt_FP16);
  uint64_t result = TsmExecute(&instruction);
  TsmDeleteDataMove(move);
  return result;
}

static uint64_t wafer_dmx_raw_memset(uint64_t destination, uint32_t value,
                                     uint32_t elements, uint32_t stride,
                                     uint32_t iterations,
                                     Data_Format format) {
  St_StrideIteration descriptor = {stride, iterations, 0U, 1U, 0U, 1U};
  TsmDataMoveInstr instruction = {0};
  TsmPeripheral *peripheral = TsmNewPeripheral();
  peripheral->Memset(&instruction, destination, value, elements, &descriptor,
                     format);
  uint64_t result = TsmExecute(&instruction);
  TsmDeletePeripheral(peripheral);
  return result;
}

static uint32_t wafer_dmx_issue(const WaferDMXCase *selected,
                                uint64_t *raw_execute_rc) {
  const uint64_t input = WAFER_DMX_SPM_INPUT + WAFER_DMX_BODY_OFFSET;
  const uint64_t output = WAFER_DMX_SPM_OUTPUT + WAFER_DMX_BODY_OFFSET;
  const uint64_t auxiliary0 = WAFER_DMX_SPM_AUX0 + WAFER_DMX_BODY_OFFSET;
  const uint64_t auxiliary1 = WAFER_DMX_SPM_AUX1 + WAFER_DMX_BODY_OFFSET;
  *raw_execute_rc = UINT64_MAX;
  switch (selected->case_id) {
  case 0U:
    *raw_execute_rc = wafer_dmx_raw_concat(
        input, wafer_dmx_shape(2U, 7U, 9U, 33U), input + 16384U,
        wafer_dmx_shape(2U, 7U, 9U, 32U), output,
        wafer_dmx_shape(2U, 7U, 9U, 65U), 0U);
    break;
  case 1U:
    *raw_execute_rc = wafer_dmx_raw_concat(
        input, wafer_dmx_shape(2U, 7U, 4U, 65U), input + 7680U,
        wafer_dmx_shape(2U, 7U, 5U, 65U), output,
        wafer_dmx_shape(2U, 7U, 9U, 65U), 1U);
    break;
  case 2U:
    *raw_execute_rc = wafer_dmx_raw_concat(
        input, wafer_dmx_shape(2U, 3U, 9U, 65U), input + 7680U,
        wafer_dmx_shape(2U, 4U, 9U, 65U), output,
        wafer_dmx_shape(2U, 7U, 9U, 65U), 2U);
    break;
  case 4U:
    wafer_tx81_tdma_pad(input, output, 2U, 5U, 7U, 65U, 2U, 7U, 10U, 65U,
                        1U, 1U, 2U, 1U, Fmt_FP16, 0U);
    break;
  case 5U:
    wafer_tx81_tdma_img2col(input, output, 2U, 9U, 11U, 65U, 2U, 6U, 54U,
                            65U, 1U, 0U, 2U, 1U, 3U, 2U, 2U, 1U, Fmt_FP16, 0U);
    break;
  case 6U:
    *raw_execute_rc = wafer_dmx_raw_mask_gather(
        input, (uint32_t)(input + 4096U), output, 0U);
    break;
  case 7U:
    *raw_execute_rc = wafer_dmx_raw_mask_gather(
        input, (uint32_t)(input + 4096U), output, 1U);
    break;
  case 8U:
    *raw_execute_rc = wafer_dmx_raw_tensor_nom(input, output);
    break;
  case 9U:
    wafer_dmx_gather(input, auxiliary0, 256U, 128U, 128U, 2U, 130U, 2U);
    wafer_dmx_gather(input + 256U, auxiliary0 + 128U, 4U, 2U, 8U, 2U, 130U,
                     2U);
    wafer_tx81_elementwise_add(auxiliary0, auxiliary0, output, 130U,
                               Fmt_FP16, 0U);
    break;
  case 10U:
    for (uint32_t batch = 0; batch < 2U; ++batch) {
      wafer_dmx_gather(input + batch * 256U, auxiliary0 + batch * 130U, 128U,
                       128U, 0U, 1U, 0U, 1U);
      wafer_dmx_gather(input + batch * 256U + 128U,
                       auxiliary0 + batch * 130U + 128U, 2U, 2U, 0U, 1U, 0U,
                       1U);
    }
    wafer_tx81_elementwise_add(auxiliary0, auxiliary0, output, 130U,
                               Fmt_FP16, 0U);
    break;
  case 11U:
    wafer_dmx_gather(input, auxiliary0, 256U, 256U, 0U, 1U, 0U, 1U);
    wafer_dmx_gather(input + 4096U, auxiliary1, 512U, 512U, 0U, 1U, 0U, 1U);
    wafer_tx81_gemm(auxiliary0, auxiliary1, output, 0, 1U, 16U, 16U, 1U,
                    Fmt_FP16, Fmt_FP16, Fmt_UNUSED, 0U);
    break;
  case 12U:
    *raw_execute_rc =
        wafer_dmx_raw_memset(output, UINT32_C(0x3c00), 64U, 256U, 32U,
                             Fmt_FP16);
    break;
  case 13U:
    *raw_execute_rc =
        wafer_dmx_raw_memset(output, UINT32_C(0x3f80), 32U, 128U, 64U,
                             Fmt_BF16);
    break;
  case 14U:
    *raw_execute_rc =
        wafer_dmx_raw_memset(output, UINT32_C(0x3c00), 2048U, 4096U, 1U,
                             Fmt_FP16);
    break;
  case 15U:
    wafer_tx81_memset(output, UINT32_C(0x3c00), 2048U, Fmt_FP16, 0U);
    break;
  case 16U:
    *raw_execute_rc =
        wafer_dmx_raw_memset(output, UINT32_C(0x3f80), 2048U, 4096U, 1U,
                             Fmt_BF16);
    break;
  case 17U:
    wafer_tx81_memset(output, UINT32_C(0x3f80), 2048U, Fmt_BF16, 0U);
    break;
  default:
    return 0U;
  }
  return 1U;
}

static void wafer_dmx_init_record(volatile uint64_t *record,
                                  uint32_t status) {
  for (uint32_t index = 0; index < WAFER_DMX_RECORD_WORDS; ++index)
    record[index] = 0;
  record[WAFER_DMX_REC_MAGIC] = WAFER_DMX_RECORD_MAGIC;
  record[WAFER_DMX_REC_WORD_COUNT] =
      WAFER_DMX_RECORD_WORDS;
  record[WAFER_DMX_REC_STATUS] = status;
  record[WAFER_DMX_REC_OUTPUT_DDR_OFFSET] = WAFER_DMX_OUTPUT_DDR_OFFSET;
  record[WAFER_DMX_REC_SLOT_BYTES] = WAFER_DMX_SLOT_BYTES;
  record[WAFER_DMX_REC_BODY_OFFSET] = WAFER_DMX_BODY_OFFSET;
  record[WAFER_DMX_REC_RAW_EXECUTE_RC] = UINT64_MAX;
  record[WAFER_DMX_REC_RECORD_GUARD] = WAFER_DMX_RECORD_GUARD;
}

__attribute__((visibility("hidden"))) void
wafer_tx81_instruction_family_probe(uint64_t request_ddr,
                                    uint64_t payload_ddr,
                                    uint64_t output_ddr) {
  wafer_dmx_cache_range(request_ddr, WAFER_DMX_RESOURCE_BYTES, 1U);
  wafer_dmx_cache_range(payload_ddr, WAFER_DMX_RESOURCE_BYTES, 1U);
  const volatile uint64_t *request =
      (const volatile uint64_t *)(uintptr_t)request_ddr;
  volatile uint64_t *record = (volatile uint64_t *)(uintptr_t)output_ddr;
  WaferDMXCase selected = {0};
  uint32_t status = wafer_dmx_decode(request, &selected);
  wafer_dmx_init_record(record, status);
  if (status == WAFER_DMX_STATUS_OK) {
    record[WAFER_DMX_REC_CASE] = selected.case_id;
    record[WAFER_DMX_REC_INPUT_BYTES] = selected.input_bytes;
    record[WAFER_DMX_REC_RESULT_BYTES] = selected.result_bytes;
    record[WAFER_DMX_REC_OUTPUT_SPAN] = selected.output_span;
    record[WAFER_DMX_REC_TDMA_INSTRUCTIONS] = selected.tdma_instructions;
    record[WAFER_DMX_REC_CT_INSTRUCTIONS] = selected.ct_instructions;
    record[WAFER_DMX_REC_NE_INSTRUCTIONS] = selected.ne_instructions;
    record[WAFER_DMX_REC_ORACLE] = selected.oracle;
    record[WAFER_DMX_REC_SAMPLE] = request[WAFER_DMX_REQ_SAMPLE];
    record[WAFER_DMX_REC_REQUEST_GUARD] = request[WAFER_DMX_REQ_GUARD];

    uint32_t staged_bytes = WAFER_DMX_BODY_OFFSET + selected.input_bytes;
    wafer_tx81_rdma(payload_ddr, WAFER_DMX_SPM_INPUT, staged_bytes,
                    staged_bytes, 0U, 0U, 0U, 1U, 1U, 1U, Fmt_UINT8, 0U);
    wafer_tx81_rdma(request_ddr + WAFER_DMX_SLOT_BYTES,
                    WAFER_DMX_SPM_OUTPUT, WAFER_DMX_SLOT_BYTES,
                    WAFER_DMX_SLOT_BYTES, 0U, 0U, 0U, 1U, 1U, 1U,
                    Fmt_UINT8, 0U);
    if (selected.case_id == 11U) {
      wafer_tx81_rdma(request_ddr + WAFER_DMX_SLOT_BYTES,
                      WAFER_DMX_SPM_AUX0,
                      WAFER_DMX_BODY_OFFSET + 256U,
                      WAFER_DMX_BODY_OFFSET + 256U, 0U, 0U, 0U, 1U, 1U, 1U,
                      Fmt_UINT8, 0U);
      wafer_tx81_rdma(request_ddr + WAFER_DMX_SLOT_BYTES,
                      WAFER_DMX_SPM_AUX1,
                      WAFER_DMX_BODY_OFFSET + 512U,
                      WAFER_DMX_BODY_OFFSET + 512U, 0U, 0U, 0U, 1U, 1U, 1U,
                      Fmt_UINT8, 0U);
    }
    WaferDMXPMU before = wafer_dmx_read_pmu();
    uint64_t raw_execute_rc = UINT64_MAX;
    if (wafer_dmx_issue(&selected, &raw_execute_rc) == 0U) {
      status = WAFER_DMX_STATUS_EXECUTE_FAILED;
    } else {
      wafer_tx81_wdma(WAFER_DMX_SPM_OUTPUT,
                      output_ddr + WAFER_DMX_OUTPUT_DDR_OFFSET,
                      WAFER_DMX_SLOT_BYTES, WAFER_DMX_SLOT_BYTES, 0U, 0U, 0U,
                      1U, 1U, 1U, Fmt_UINT8, 0U);
      wafer_tx81_ncc_join(1U);
      WaferDMXPMU after = wafer_dmx_read_pmu();
      record[WAFER_DMX_REC_TDMA_INST_DELTA] =
          (uint32_t)(after.tdma_instructions - before.tdma_instructions);
      record[WAFER_DMX_REC_CT_INST_DELTA] =
          (uint32_t)(after.ct_instructions - before.ct_instructions);
      record[WAFER_DMX_REC_NE_INST_DELTA] =
          (uint32_t)(after.ne_instructions - before.ne_instructions);
      record[WAFER_DMX_REC_TDMA_EXEC_DELTA] =
          after.tdma_execution - before.tdma_execution;
      record[WAFER_DMX_REC_CT_EXEC_DELTA] =
          after.ct_execution - before.ct_execution;
      record[WAFER_DMX_REC_NE_EXEC_DELTA] =
          after.ne_execution - before.ne_execution;
      record[WAFER_DMX_REC_RAW_EXECUTE_RC] = raw_execute_rc;
    }
    record[WAFER_DMX_REC_STATUS] = status;
  }
  wafer_dmx_cache_range(output_ddr,
                        WAFER_DMX_RECORD_WORDS * sizeof(uint64_t), 0U);
}
