#include "instr_def.h"
#include "wafer_ddr_sparse_high_offset_probe_protocol.h"
#include "wafer_tx81_crt.h"

#include <stdint.h>

#define WAFER_DDR_SPARSE_CACHE_LINE_BYTES 64U
#define WAFER_DDR_SPARSE_ADDRESS_ALIGNMENT UINT64_C(256)

static uint64_t wafer_ddr_sparse_offset(uint32_t index,
                                        uint64_t workspace_bytes) {
  static const uint64_t fixed_offsets[] = {
      UINT64_C(0x000000000),
      UINT64_C(0x100000000),
      UINT64_C(0x400000000),
      UINT64_C(0x800000000),
  };
  if (index < sizeof(fixed_offsets) / sizeof(fixed_offsets[0]))
    return fixed_offsets[index];
  if (index == 4U)
    return workspace_bytes - WAFER_DDR_SPARSE_NEAR_END_DISTANCE;
  return workspace_bytes - WAFER_DDR_SPARSE_SLOT_BYTES;
}

static void wafer_ddr_sparse_cache_range(uint64_t begin, uint32_t bytes,
                                         uint32_t invalidate_only) {
  enum {
    WAFER_DDR_SPARSE_SUPERVISOR_MODE = 1,
    WAFER_DDR_SPARSE_MACHINE_MODE = 3,
  };
  uintptr_t mode;
  __asm__ volatile("fence" ::: "memory");
  __asm__ volatile("sync" ::: "memory");
  __asm__ volatile("csrr %0, mxstatus" : "=r"(mode));
  mode = (mode >> 30) & 3U;
  for (uintptr_t address = (uintptr_t)begin;
       address < (uintptr_t)begin + bytes;
       address += WAFER_DDR_SPARSE_CACHE_LINE_BYTES) {
    if (mode == WAFER_DDR_SPARSE_MACHINE_MODE) {
      if (invalidate_only != 0U)
        __asm__ volatile("dcache.ipa %0" : : "r"(address) : "memory");
      else
        __asm__ volatile("dcache.cipa %0" : : "r"(address) : "memory");
    } else if (mode == WAFER_DDR_SPARSE_SUPERVISOR_MODE) {
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

static void wafer_ddr_sparse_rdma(uint64_t source, uint64_t destination) {
  wafer_tx81_rdma_v3(source, destination, WAFER_DDR_SPARSE_SLOT_BYTES,
                  WAFER_DDR_SPARSE_SLOT_BYTES, 0U, 0U, 0U, 1U, 1U, 1U,
                  Fmt_UINT8, 0U);
}

static void wafer_ddr_sparse_wdma(uint64_t source, uint64_t destination) {
  wafer_tx81_wdma_v3(source, destination, WAFER_DDR_SPARSE_SLOT_BYTES,
                  WAFER_DDR_SPARSE_SLOT_BYTES, 0U, 0U, 0U, 1U, 1U, 1U,
                  Fmt_UINT8, 0U);
}

static uint32_t wafer_ddr_sparse_request_status(
    const volatile uint64_t *request, uint64_t input, uint64_t output,
    uint64_t workspace) {
  uint64_t workspace_bytes =
      request[WAFER_DDR_SPARSE_REQ_WORKSPACE_BYTES];
  if (request[WAFER_DDR_SPARSE_REQ_MAGIC] !=
          WAFER_DDR_SPARSE_REQUEST_MAGIC ||
      request[WAFER_DDR_SPARSE_REQ_SCHEMA] != WAFER_DDR_SPARSE_SCHEMA ||
      workspace_bytes < WAFER_DDR_SPARSE_MIN_WORKSPACE_BYTES ||
      workspace_bytes > WAFER_DDR_SPARSE_MAX_WORKSPACE_BYTES ||
      workspace_bytes % WAFER_DDR_SPARSE_ADDRESS_ALIGNMENT != 0U ||
      request[WAFER_DDR_SPARSE_REQ_RESOURCE_BYTES] !=
          WAFER_DDR_SPARSE_RESOURCE_BYTES ||
      request[WAFER_DDR_SPARSE_REQ_PAYLOAD_BYTES] !=
          WAFER_DDR_SPARSE_PAYLOAD_BYTES ||
      request[WAFER_DDR_SPARSE_REQ_GUARD_BYTES] !=
          WAFER_DDR_SPARSE_GUARD_BYTES ||
      request[WAFER_DDR_SPARSE_REQ_OFFSET_COUNT] !=
          WAFER_DDR_SPARSE_OFFSET_COUNT ||
      request[WAFER_DDR_SPARSE_REQ_GUARD] !=
          WAFER_DDR_SPARSE_REQUEST_GUARD)
    return WAFER_DDR_SPARSE_STATUS_BAD_REQUEST;
  if (input == 0U || output == 0U || workspace == 0U ||
      input % WAFER_DDR_SPARSE_ADDRESS_ALIGNMENT != 0U ||
      output % WAFER_DDR_SPARSE_ADDRESS_ALIGNMENT != 0U ||
      workspace % WAFER_DDR_SPARSE_ADDRESS_ALIGNMENT != 0U)
    return WAFER_DDR_SPARSE_STATUS_BAD_BINDING;
  if (input > UINT64_MAX - WAFER_DDR_SPARSE_RESOURCE_BYTES ||
      output > UINT64_MAX - WAFER_DDR_SPARSE_RESOURCE_BYTES ||
      workspace > UINT64_MAX - workspace_bytes)
    return WAFER_DDR_SPARSE_STATUS_ADDRESS_OVERFLOW;
  return WAFER_DDR_SPARSE_STATUS_OK;
}

static void wafer_ddr_sparse_write_header(
    volatile uint64_t *header, const volatile uint64_t *request,
    uint32_t status, uint32_t row_count, uint64_t input, uint64_t output,
    uint64_t workspace) {
  for (uint32_t word = 0; word < WAFER_DDR_SPARSE_HEADER_WORDS; ++word)
    header[word] = 0U;
  header[WAFER_DDR_SPARSE_HDR_MAGIC] = WAFER_DDR_SPARSE_RECORD_MAGIC;
  header[WAFER_DDR_SPARSE_HDR_SCHEMA_AND_WORDS] =
      ((uint64_t)WAFER_DDR_SPARSE_SCHEMA << 32) |
      WAFER_DDR_SPARSE_HEADER_WORDS;
  header[WAFER_DDR_SPARSE_HDR_STATUS] = status;
  header[WAFER_DDR_SPARSE_HDR_SAMPLE] =
      request[WAFER_DDR_SPARSE_REQ_SAMPLE];
  header[WAFER_DDR_SPARSE_HDR_ROW_COUNT] = row_count;
  header[WAFER_DDR_SPARSE_HDR_WORKSPACE_BASE] = workspace;
  header[WAFER_DDR_SPARSE_HDR_WORKSPACE_BYTES] =
      request[WAFER_DDR_SPARSE_REQ_WORKSPACE_BYTES];
  header[WAFER_DDR_SPARSE_HDR_INPUT_BASE] = input;
  header[WAFER_DDR_SPARSE_HDR_OUTPUT_BASE] = output;
  header[WAFER_DDR_SPARSE_HDR_REQUEST_GUARD] =
      request[WAFER_DDR_SPARSE_REQ_GUARD];
  header[WAFER_DDR_SPARSE_HDR_PAYLOAD_BYTES] =
      WAFER_DDR_SPARSE_PAYLOAD_BYTES;
  header[WAFER_DDR_SPARSE_HDR_GUARD_BYTES] =
      WAFER_DDR_SPARSE_GUARD_BYTES;
  header[WAFER_DDR_SPARSE_HDR_SLOT_BYTES] = WAFER_DDR_SPARSE_SLOT_BYTES;
  header[WAFER_DDR_SPARSE_HDR_RECORD_GUARD] =
      WAFER_DDR_SPARSE_RECORD_GUARD;
}

static void wafer_ddr_sparse_write_row(volatile uint64_t *row,
                                       uint32_t index, uint64_t workspace,
                                       uint64_t workspace_bytes) {
  uint64_t relative_offset =
      wafer_ddr_sparse_offset(index, workspace_bytes);
  uint64_t input_offset =
      WAFER_DDR_SPARSE_INPUT_BASE +
      (uint64_t)index * WAFER_DDR_SPARSE_SLOT_BYTES;
  uint64_t archive_offset =
      WAFER_DDR_SPARSE_ARCHIVE_BASE +
      (uint64_t)index * WAFER_DDR_SPARSE_SLOT_BYTES;
  row[WAFER_DDR_SPARSE_ROW_MAGIC_WORD] = WAFER_DDR_SPARSE_ROW_MAGIC;
  row[WAFER_DDR_SPARSE_ROW_INDEX] = index;
  row[WAFER_DDR_SPARSE_ROW_RELATIVE_OFFSET] = relative_offset;
  row[WAFER_DDR_SPARSE_ROW_WORKSPACE_ADDRESS] =
      workspace + relative_offset;
  row[WAFER_DDR_SPARSE_ROW_INPUT_OFFSET] = input_offset;
  row[WAFER_DDR_SPARSE_ROW_ARCHIVE_OFFSET] = archive_offset;
  row[WAFER_DDR_SPARSE_ROW_SLOT_BYTES] = WAFER_DDR_SPARSE_SLOT_BYTES;
  row[WAFER_DDR_SPARSE_ROW_GUARD_WORD] = WAFER_DDR_SPARSE_ROW_GUARD;
}

__attribute__((visibility("hidden"))) void
wafer_tx81_ddr_sparse_high_offset_probe(uint64_t input, uint64_t output,
                                        uint64_t workspace) {
  wafer_ddr_sparse_cache_range(
      input, WAFER_DDR_SPARSE_REQUEST_WORDS * sizeof(uint64_t), 1U);
  wafer_ddr_sparse_cache_range(output, WAFER_DDR_SPARSE_RESOURCE_BYTES, 1U);
  const volatile uint64_t *request =
      (const volatile uint64_t *)(uintptr_t)input;
  volatile uint64_t *header = (volatile uint64_t *)(uintptr_t)output;
  volatile uint64_t *rows = header + WAFER_DDR_SPARSE_HEADER_WORDS;
  uint32_t status =
      wafer_ddr_sparse_request_status(request, input, output, workspace);
  uint32_t completed = 0U;
  uint64_t workspace_bytes =
      request[WAFER_DDR_SPARSE_REQ_WORKSPACE_BYTES];

  if (status == WAFER_DDR_SPARSE_STATUS_OK) {
    for (uint32_t index = 0; index < WAFER_DDR_SPARSE_OFFSET_COUNT; ++index) {
      uint64_t input_offset =
          WAFER_DDR_SPARSE_INPUT_BASE +
          (uint64_t)index * WAFER_DDR_SPARSE_SLOT_BYTES;
      uint64_t archive_offset =
          WAFER_DDR_SPARSE_ARCHIVE_BASE +
          (uint64_t)index * WAFER_DDR_SPARSE_SLOT_BYTES;
      uint64_t workspace_address =
          workspace + wafer_ddr_sparse_offset(index, workspace_bytes);

      wafer_ddr_sparse_rdma(input + input_offset,
                            WAFER_DDR_SPARSE_SPM_SOURCE);
      wafer_tx81_ncc_join(1U);
      wafer_ddr_sparse_wdma(WAFER_DDR_SPARSE_SPM_SOURCE,
                            workspace_address);
      wafer_tx81_ncc_join(1U);
      wafer_ddr_sparse_rdma(workspace_address,
                            WAFER_DDR_SPARSE_SPM_READBACK);
      wafer_tx81_ncc_join(1U);
      wafer_ddr_sparse_wdma(WAFER_DDR_SPARSE_SPM_READBACK,
                            output + archive_offset);
      wafer_tx81_ncc_join(1U);
      wafer_ddr_sparse_write_row(
          rows + index * WAFER_DDR_SPARSE_ROW_WORDS, index, workspace,
          workspace_bytes);
      ++completed;
    }
  }

  wafer_ddr_sparse_write_header(header, request, status, completed, input,
                                output, workspace);
  wafer_ddr_sparse_cache_range(output, WAFER_DDR_SPARSE_RECORD_BYTES, 0U);
}
