#include "wafer_cabi_shim.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifndef WAFER_CABI_SHIM_CAPTURE
#ifndef CONFIG_NO_PLATFORM_HOOK_H
#define CONFIG_NO_PLATFORM_HOOK_H 1
#endif
#ifndef USING_RISCV
#define USING_RISCV 1
#endif
#include "instr_adapter.h"
#endif

enum {
  WAFER_CABI_STATUS_OK = 0,
  WAFER_CABI_STATUS_INVALID_ARGUMENT = 1,
  WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE = 2,
};

enum {
  WAFER_CABI_INTER_TYPE_CGRA = 0,
  WAFER_CABI_INTER_TYPE_NEUR = 1,
  WAFER_CABI_INTER_TYPE_RDMA = 2,
  WAFER_CABI_INTER_TYPE_WDMA = 3,
  WAFER_CABI_INTER_TYPE_TDMA = 4,
  WAFER_CABI_NE_TYPE_GEMM = 3,
  WAFER_CABI_TDMA_GATHER_SCATTER_OPCODE = 135,
  WAFER_CABI_PERIPHERAL_MEMSET_OPCODE = 179,
};

enum {
  WAFER_CABI_ELEMENTWISE_ADD = 0,
  WAFER_CABI_ELEMENTWISE_SUB = 1,
  WAFER_CABI_ELEMENTWISE_MUL = 2,
  WAFER_CABI_ELEMENTWISE_DIV = 3,
  WAFER_CABI_ELEMENTWISE_MAX = 4,
  WAFER_CABI_ELEMENTWISE_MIN = 5,
  WAFER_CABI_ELEMENTWISE_NEG = 6,
  WAFER_CABI_ELEMENTWISE_RECIP = 7,
  WAFER_CABI_ELEMENTWISE_SQRT = 8,
  WAFER_CABI_ELEMENTWISE_RSQRT = 9,
  WAFER_CABI_ELEMENTWISE_EXP = 10,
  WAFER_CABI_ELEMENTWISE_TANH = 11,
  WAFER_CABI_ELEMENTWISE_EQ = 12,
  WAFER_CABI_ELEMENTWISE_NE = 13,
  WAFER_CABI_ELEMENTWISE_LT = 14,
  WAFER_CABI_ELEMENTWISE_LE = 15,
  WAFER_CABI_ELEMENTWISE_GT = 16,
  WAFER_CABI_ELEMENTWISE_GE = 17,
};

enum {
  WAFER_CABI_REDUCE_SUM = 0,
  WAFER_CABI_REDUCE_MAX = 1,
  WAFER_CABI_REDUCE_MIN = 2,
  WAFER_CABI_REDUCE_AVG = 3,
};

typedef struct wafer_cabi_stride_iteration {
  uint32_t stride0;
  uint32_t logical_iteration0;
  uint32_t encoded_iteration0;
  uint32_t stride1;
  uint32_t logical_iteration1;
  uint32_t encoded_iteration1;
  uint32_t stride2;
  uint32_t logical_iteration2;
  uint32_t encoded_iteration2;
} wafer_cabi_stride_iteration_t;

static int wafer_checked_add_u64(uint64_t lhs, uint64_t rhs, uint64_t *result) {
  if (rhs > UINT64_MAX - lhs)
    return 0;
  *result = lhs + rhs;
  return 1;
}

static int wafer_checked_mul_u64(uint64_t lhs, uint64_t rhs, uint64_t *result) {
  if (lhs != 0 && rhs > UINT64_MAX / lhs)
    return 0;
  *result = lhs * rhs;
  return 1;
}

static int wafer_checked_mul_add_u64(uint64_t lhs, uint64_t rhs,
                                     uint64_t addend, uint64_t *result) {
  uint64_t product = 0;
  if (!wafer_checked_mul_u64(lhs, rhs, &product))
    return 0;
  return wafer_checked_add_u64(product, addend, result);
}

static int wafer_checked_u32(uint64_t value, uint32_t *result) {
  if (value > UINT32_MAX)
    return 0;
  *result = (uint32_t)value;
  return 1;
}

static int wafer_checked_u16(uint64_t value, uint16_t *result) {
  if (value > UINT16_MAX)
    return 0;
  *result = (uint16_t)value;
  return 1;
}

static int wafer_checked_end_inclusive(uint64_t base, uint64_t span,
                                       uint64_t *end) {
  uint64_t exclusive_end = 0;
  if (span == 0)
    return 0;
  if (!wafer_checked_add_u64(base, span, &exclusive_end))
    return 0;
  *end = exclusive_end - 1;
  return 1;
}

static int wafer_format_element_bytes(uint32_t data_format,
                                      uint64_t *element_bytes,
                                      int *bit_packed) {
  *bit_packed = 0;
  switch (data_format) {
  case 0:
  case 8:
    *element_bytes = 1;
    return 1;
  case 1:
  case 2:
  case 3:
  case 9:
    *element_bytes = 2;
    return 1;
  case 4:
  case 5:
  case 6:
  case 10:
    *element_bytes = 4;
    return 1;
  case 11:
  case 12:
    *element_bytes = 8;
    return 1;
  case 7:
    *element_bytes = 1;
    *bit_packed = 1;
    return 1;
  default:
    return 0;
  }
}

static int wafer_inner_elem_count(uint64_t inner_bytes, uint32_t data_format,
                                  uint32_t *elem_count) {
  uint64_t element_bytes = 0;
  int bit_packed = 0;
  uint64_t count = 0;

  if (inner_bytes == 0)
    return 0;
  if (!wafer_format_element_bytes(data_format, &element_bytes, &bit_packed))
    return 0;
  if (bit_packed) {
    if (!wafer_checked_mul_u64(inner_bytes, 8, &count))
      return 0;
  } else {
    if (inner_bytes % element_bytes != 0)
      return 0;
    count = inner_bytes / element_bytes;
  }
  return wafer_checked_u32(count, elem_count);
}

static int wafer_validate_element_count(uint64_t element_count,
                                        uint32_t data_format,
                                        uint32_t *elem_count) {
  uint64_t element_bytes = 0;
  int bit_packed = 0;
  if (element_count == 0 || !wafer_checked_u32(element_count, elem_count) ||
      !wafer_format_element_bytes(data_format, &element_bytes, &bit_packed))
    return 0;
  return 1;
}

static int wafer_elementwise_opcode(uint32_t kind, uint32_t output_format,
                                    uint32_t *opcode) {
  int bool_output = output_format == 7;
  switch (kind) {
  case WAFER_CABI_ELEMENTWISE_ADD:
    *opcode = 14;
    return 1;
  case WAFER_CABI_ELEMENTWISE_SUB:
    *opcode = 18;
    return 1;
  case WAFER_CABI_ELEMENTWISE_MUL:
    *opcode = 22;
    return 1;
  case WAFER_CABI_ELEMENTWISE_DIV:
    *opcode = 26;
    return 1;
  case WAFER_CABI_ELEMENTWISE_MAX:
    *opcode = 6;
    return 1;
  case WAFER_CABI_ELEMENTWISE_MIN:
    *opcode = 10;
    return 1;
  case WAFER_CABI_ELEMENTWISE_NEG:
    *opcode = 5;
    return 1;
  case WAFER_CABI_ELEMENTWISE_RECIP:
    *opcode = 1;
    return 1;
  case WAFER_CABI_ELEMENTWISE_SQRT:
    *opcode = 3;
    return 1;
  case WAFER_CABI_ELEMENTWISE_RSQRT:
    *opcode = 4;
    return 1;
  case WAFER_CABI_ELEMENTWISE_EXP:
    *opcode = 101;
    return 1;
  case WAFER_CABI_ELEMENTWISE_TANH:
    *opcode = 105;
    return 1;
  case WAFER_CABI_ELEMENTWISE_EQ:
    *opcode = bool_output ? 31 : 30;
    return 1;
  case WAFER_CABI_ELEMENTWISE_NE:
    *opcode = bool_output ? 39 : 38;
    return 1;
  case WAFER_CABI_ELEMENTWISE_GE:
    *opcode = bool_output ? 47 : 46;
    return 1;
  case WAFER_CABI_ELEMENTWISE_GT:
    *opcode = bool_output ? 55 : 54;
    return 1;
  case WAFER_CABI_ELEMENTWISE_LE:
    *opcode = bool_output ? 63 : 62;
    return 1;
  case WAFER_CABI_ELEMENTWISE_LT:
    *opcode = bool_output ? 71 : 70;
    return 1;
  default:
    return 0;
  }
}

static int wafer_reduce_opcode(uint32_t kind, uint32_t *opcode) {
  switch (kind) {
  case WAFER_CABI_REDUCE_SUM:
    *opcode = 111;
    return 1;
  case WAFER_CABI_REDUCE_AVG:
    *opcode = 112;
    return 1;
  case WAFER_CABI_REDUCE_MAX:
    *opcode = 113;
    return 1;
  case WAFER_CABI_REDUCE_MIN:
    *opcode = 114;
    return 1;
  default:
    return 0;
  }
}

static int wafer_convert_opcode(uint32_t src_format, uint32_t dst_format,
                                uint32_t *opcode) {
  if (src_format == dst_format) {
    *opcode = WAFER_CABI_TDMA_GATHER_SCATTER_OPCODE;
    return 1;
  }

  switch (src_format) {
  case 0:
    if (dst_format >= 2 && dst_format <= 6 && dst_format != 4) {
      *opcode = 139 + (dst_format == 2   ? 0
                       : dst_format == 3 ? 1
                       : dst_format == 5 ? 2
                                         : 3);
      return 1;
    }
    break;
  case 1:
    if (dst_format == 2 || dst_format == 3 || dst_format == 5 ||
        dst_format == 6) {
      *opcode = 143 + (dst_format == 2   ? 0
                       : dst_format == 3 ? 1
                       : dst_format == 5 ? 2
                                         : 3);
      return 1;
    }
    break;
  case 4:
    if (dst_format == 2 || dst_format == 3 || dst_format == 5 ||
        dst_format == 6) {
      *opcode = 147 + (dst_format == 2   ? 0
                       : dst_format == 3 ? 1
                       : dst_format == 5 ? 2
                                         : 3);
      return 1;
    }
    break;
  case 3:
    if (dst_format == 0 || dst_format == 1 || dst_format == 4 ||
        dst_format == 2 || dst_format == 5 || dst_format == 6) {
      *opcode = dst_format == 0   ? 151
                : dst_format == 1 ? 152
                : dst_format == 4 ? 153
                : dst_format == 2 ? 154
                : dst_format == 5 ? 155
                                  : 156;
      return 1;
    }
    break;
  case 2:
    if (dst_format == 0 || dst_format == 1 || dst_format == 4 ||
        dst_format == 3 || dst_format == 5 || dst_format == 6) {
      *opcode = dst_format == 0   ? 157
                : dst_format == 1 ? 158
                : dst_format == 4 ? 159
                : dst_format == 3 ? 160
                : dst_format == 5 ? 161
                                  : 162;
      return 1;
    }
    break;
  case 5:
    if (dst_format == 0 || dst_format == 1 || dst_format == 4 ||
        dst_format == 2 || dst_format == 3 || dst_format == 6) {
      *opcode = dst_format == 0   ? 163
                : dst_format == 1 ? 164
                : dst_format == 4 ? 165
                : dst_format == 2 ? 166
                : dst_format == 3 ? 167
                                  : 168;
      return 1;
    }
    break;
  case 6:
    if (dst_format == 0 || dst_format == 1 || dst_format == 4 ||
        dst_format == 2 || dst_format == 3 || dst_format == 5) {
      *opcode = dst_format == 0   ? 169
                : dst_format == 1 ? 170
                : dst_format == 4 ? 171
                : dst_format == 2 ? 172
                : dst_format == 3 ? 173
                                  : 174;
      return 1;
    }
    break;
  default:
    break;
  }
  return 0;
}

static int wafer_build_stride_iteration(uint64_t stride0, uint64_t stride1,
                                        uint64_t stride2, uint64_t iteration0,
                                        uint64_t iteration1,
                                        uint64_t iteration2,
                                        wafer_cabi_stride_iteration_t *result) {
  if (iteration0 == 0 || iteration1 == 0 || iteration2 == 0)
    return 0;
  if (!wafer_checked_u32(stride0, &result->stride0) ||
      !wafer_checked_u32(stride1, &result->stride1) ||
      !wafer_checked_u32(stride2, &result->stride2) ||
      !wafer_checked_u32(iteration0, &result->logical_iteration0) ||
      !wafer_checked_u32(iteration1, &result->logical_iteration1) ||
      !wafer_checked_u32(iteration2, &result->logical_iteration2))
    return 0;

  result->encoded_iteration0 = result->logical_iteration0 - 1;
  result->encoded_iteration1 = result->logical_iteration1 - 1;
  result->encoded_iteration2 = result->logical_iteration2 - 1;
  return 1;
}

static int wafer_strided_span(const wafer_cabi_stride_iteration_t *si,
                              uint64_t inner_bytes, uint64_t *span) {
  uint64_t max_offset = 0;
  if (!wafer_checked_mul_add_u64(si->logical_iteration0 - 1, si->stride0,
                                 max_offset, &max_offset))
    return 0;
  if (!wafer_checked_mul_add_u64(si->logical_iteration1 - 1, si->stride1,
                                 max_offset, &max_offset))
    return 0;
  if (!wafer_checked_mul_add_u64(si->logical_iteration2 - 1, si->stride2,
                                 max_offset, &max_offset))
    return 0;
  return wafer_checked_add_u64(max_offset, inner_bytes, span);
}

static int wafer_iteration_product(const wafer_cabi_stride_iteration_t *si,
                                   uint64_t *product) {
  uint64_t partial = 0;
  if (!wafer_checked_mul_u64(si->logical_iteration0, si->logical_iteration1,
                             &partial))
    return 0;
  return wafer_checked_mul_u64(partial, si->logical_iteration2, product);
}

static int wafer_validate_dma(uint64_t byte_count, uint64_t inner_bytes,
                              uint32_t data_format,
                              const wafer_cabi_stride_iteration_t *si,
                              uint32_t *elem_count, uint64_t *strided_span) {
  if (byte_count == 0 || inner_bytes == 0 || inner_bytes > byte_count)
    return 0;
  if (!wafer_inner_elem_count(inner_bytes, data_format, elem_count))
    return 0;
  return wafer_strided_span(si, inner_bytes, strided_span);
}

#ifdef WAFER_CABI_SHIM_CAPTURE
static wafer_cabi_last_issue_t wafer_last_issue;

void wafer_cabi_reset_capture(void) {
  memset(&wafer_last_issue, 0, sizeof(wafer_last_issue));
}

const wafer_cabi_last_issue_t *wafer_cabi_get_last_issue(void) {
  return &wafer_last_issue;
}

static void wafer_capture_status(uint32_t kind, int32_t status) {
  memset(&wafer_last_issue, 0, sizeof(wafer_last_issue));
  wafer_last_issue.kind = kind;
  wafer_last_issue.status = status;
}
#else
static int32_t wafer_status_from_u64(uint64_t status) {
  if (status > INT32_MAX)
    return INT32_MAX;
  return (int32_t)status;
}
#endif

int32_t wafer_rdma(uint64_t ddr_src_addr, uint32_t spm_dst_offset,
                   uint64_t byte_count, uint64_t inner_bytes,
                   uint64_t src_stride0_b, uint64_t src_stride1_b,
                   uint64_t src_stride2_b, uint64_t src_iter0,
                   uint64_t src_iter1, uint64_t src_iter2,
                   uint32_t data_format) {
  wafer_cabi_stride_iteration_t si;
  uint32_t elem_count = 0;
  uint64_t strided_span = 0;
  int32_t status = WAFER_CABI_STATUS_OK;

  if (!wafer_build_stride_iteration(src_stride0_b, src_stride1_b, src_stride2_b,
                                    src_iter0, src_iter1, src_iter2, &si) ||
      !wafer_validate_dma(byte_count, inner_bytes, data_format, &si,
                          &elem_count, &strided_span)) {
    status = WAFER_CABI_STATUS_INVALID_ARGUMENT;
#ifdef WAFER_CABI_SHIM_CAPTURE
    wafer_capture_status(WAFER_CABI_CAPTURE_RDMA, status);
#endif
    return status;
  }

#ifdef WAFER_CABI_SHIM_CAPTURE
  uint64_t src_end = 0;
  uint64_t dst_end = 0;
  if (!wafer_checked_end_inclusive(ddr_src_addr, strided_span, &src_end) ||
      !wafer_checked_end_inclusive(spm_dst_offset, byte_count, &dst_end)) {
    status = WAFER_CABI_STATUS_INVALID_ARGUMENT;
    wafer_capture_status(WAFER_CABI_CAPTURE_RDMA, status);
    return status;
  }
  wafer_capture_status(WAFER_CABI_CAPTURE_RDMA, status);
  wafer_last_issue.payload.dma =
      (wafer_cabi_dma_capture_t){WAFER_CABI_INTER_TYPE_RDMA,
                                 ddr_src_addr,
                                 spm_dst_offset,
                                 si.stride0,
                                 si.encoded_iteration0,
                                 si.stride1,
                                 si.encoded_iteration1,
                                 si.stride2,
                                 si.encoded_iteration2,
                                 elem_count,
                                 data_format,
                                 src_end,
                                 dst_end,
                                 WAFER_CABI_WAIT_ISSUE_ONLY};
  return status;
#else
  TsmRdma *rdma = TsmNewRdma();
  if (rdma == NULL || rdma->AddSrcDst == NULL ||
      rdma->ConfigStrideIteration == NULL) {
    if (rdma != NULL)
      TsmDeleteRdma(rdma);
    return WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
  }

  TsmRdmaInstr instr = {0};
  rdma->AddSrcDst(&instr, ddr_src_addr, spm_dst_offset,
                  (Data_Format)data_format);
  rdma->ConfigStrideIteration(
      &instr, elem_count, si.stride0, si.logical_iteration0, si.stride1,
      si.logical_iteration1, si.stride2, si.logical_iteration2);
  status = wafer_status_from_u64(TsmExecute(&instr));
  TsmDeleteRdma(rdma);
  return status;
#endif
}

int32_t wafer_wdma(uint32_t spm_src_offset, uint64_t ddr_dst_addr,
                   uint64_t byte_count, uint64_t inner_bytes,
                   uint64_t dst_stride0_b, uint64_t dst_stride1_b,
                   uint64_t dst_stride2_b, uint64_t dst_iter0,
                   uint64_t dst_iter1, uint64_t dst_iter2,
                   uint32_t data_format) {
  wafer_cabi_stride_iteration_t si;
  uint32_t elem_count = 0;
  uint64_t strided_span = 0;
  int32_t status = WAFER_CABI_STATUS_OK;

  if (!wafer_build_stride_iteration(dst_stride0_b, dst_stride1_b, dst_stride2_b,
                                    dst_iter0, dst_iter1, dst_iter2, &si) ||
      !wafer_validate_dma(byte_count, inner_bytes, data_format, &si,
                          &elem_count, &strided_span)) {
    status = WAFER_CABI_STATUS_INVALID_ARGUMENT;
#ifdef WAFER_CABI_SHIM_CAPTURE
    wafer_capture_status(WAFER_CABI_CAPTURE_WDMA, status);
#endif
    return status;
  }

#ifdef WAFER_CABI_SHIM_CAPTURE
  uint64_t src_end = 0;
  uint64_t dst_end = 0;
  if (!wafer_checked_end_inclusive(spm_src_offset, byte_count, &src_end) ||
      !wafer_checked_end_inclusive(ddr_dst_addr, strided_span, &dst_end)) {
    status = WAFER_CABI_STATUS_INVALID_ARGUMENT;
    wafer_capture_status(WAFER_CABI_CAPTURE_WDMA, status);
    return status;
  }
  wafer_capture_status(WAFER_CABI_CAPTURE_WDMA, status);
  wafer_last_issue.payload.dma =
      (wafer_cabi_dma_capture_t){WAFER_CABI_INTER_TYPE_WDMA,
                                 spm_src_offset,
                                 ddr_dst_addr,
                                 si.stride0,
                                 si.encoded_iteration0,
                                 si.stride1,
                                 si.encoded_iteration1,
                                 si.stride2,
                                 si.encoded_iteration2,
                                 elem_count,
                                 data_format,
                                 src_end,
                                 dst_end,
                                 WAFER_CABI_WAIT_ISSUE_ONLY};
  return status;
#else
  TsmWdma *wdma = TsmNewWdma();
  if (wdma == NULL || wdma->AddSrcDst == NULL ||
      wdma->ConfigStrideIteration == NULL) {
    if (wdma != NULL)
      TsmDeleteWdma(wdma);
    return WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
  }

  TsmWdmaInstr instr = {0};
  wdma->AddSrcDst(&instr, spm_src_offset, ddr_dst_addr,
                  (Data_Format)data_format);
  wdma->ConfigStrideIteration(
      &instr, elem_count, si.stride0, si.logical_iteration0, si.stride1,
      si.logical_iteration1, si.stride2, si.logical_iteration2);
  status = wafer_status_from_u64(TsmExecute(&instr));
  TsmDeleteWdma(wdma);
  return status;
#endif
}

int32_t wafer_gather_scatter(uint32_t spm_src_offset, uint32_t spm_dst_offset,
                             uint64_t byte_count, uint64_t inner_bytes,
                             uint64_t src_stride0_b, uint64_t src_stride1_b,
                             uint64_t src_stride2_b, uint64_t src_iter0,
                             uint64_t src_iter1, uint64_t src_iter2,
                             uint64_t dst_stride0_b, uint64_t dst_stride1_b,
                             uint64_t dst_stride2_b, uint64_t dst_iter0,
                             uint64_t dst_iter1, uint64_t dst_iter2) {
  wafer_cabi_stride_iteration_t src_si;
  wafer_cabi_stride_iteration_t dst_si;
  uint64_t src_span = 0;
  uint64_t dst_span = 0;
  uint64_t src_product = 0;
  uint64_t dst_product = 0;
  uint64_t moved_bytes = 0;
  uint32_t inner_bytes_u32 = 0;
  int32_t status = WAFER_CABI_STATUS_OK;

  if (byte_count == 0 || inner_bytes == 0 || inner_bytes > byte_count ||
      !wafer_checked_u32(inner_bytes, &inner_bytes_u32) ||
      !wafer_build_stride_iteration(src_stride0_b, src_stride1_b, src_stride2_b,
                                    src_iter0, src_iter1, src_iter2, &src_si) ||
      !wafer_build_stride_iteration(dst_stride0_b, dst_stride1_b, dst_stride2_b,
                                    dst_iter0, dst_iter1, dst_iter2, &dst_si) ||
      !wafer_strided_span(&src_si, inner_bytes, &src_span) ||
      !wafer_strided_span(&dst_si, inner_bytes, &dst_span) ||
      !wafer_iteration_product(&src_si, &src_product) ||
      !wafer_iteration_product(&dst_si, &dst_product) ||
      src_product != dst_product ||
      !wafer_checked_mul_u64(inner_bytes, src_product, &moved_bytes) ||
      moved_bytes != byte_count) {
    status = WAFER_CABI_STATUS_INVALID_ARGUMENT;
#ifdef WAFER_CABI_SHIM_CAPTURE
    wafer_capture_status(WAFER_CABI_CAPTURE_GATHER_SCATTER, status);
#endif
    return status;
  }

#ifdef WAFER_CABI_SHIM_CAPTURE
  uint64_t src_end = 0;
  uint64_t dst_end = 0;
  uint32_t src_end_u32 = 0;
  uint32_t dst_end_u32 = 0;
  if (!wafer_checked_end_inclusive(spm_src_offset, src_span, &src_end) ||
      !wafer_checked_end_inclusive(spm_dst_offset, dst_span, &dst_end) ||
      !wafer_checked_u32(src_end, &src_end_u32) ||
      !wafer_checked_u32(dst_end, &dst_end_u32)) {
    status = WAFER_CABI_STATUS_INVALID_ARGUMENT;
    wafer_capture_status(WAFER_CABI_CAPTURE_GATHER_SCATTER, status);
    return status;
  }

  wafer_capture_status(WAFER_CABI_CAPTURE_GATHER_SCATTER, status);
  wafer_last_issue.payload.gather_scatter =
      (wafer_cabi_gather_scatter_capture_t){
          WAFER_CABI_INTER_TYPE_TDMA,
          WAFER_CABI_TDMA_GATHER_SCATTER_OPCODE,
          spm_src_offset,
          spm_dst_offset,
          inner_bytes_u32,
          src_si.stride0,
          src_si.encoded_iteration0,
          src_si.stride1,
          src_si.encoded_iteration1,
          src_si.stride2,
          src_si.encoded_iteration2,
          dst_si.stride0,
          dst_si.encoded_iteration0,
          dst_si.stride1,
          dst_si.encoded_iteration1,
          dst_si.stride2,
          dst_si.encoded_iteration2,
          src_end_u32,
          dst_end_u32,
          WAFER_CABI_WAIT_ISSUE_ONLY};
  return status;
#else
  TsmDataMove *data_move = TsmNewDataMove();
  if (data_move == NULL || data_move->GatherScatter == NULL) {
    if (data_move != NULL)
      TsmDeleteDataMove(data_move);
    return WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
  }

  St_StrideIteration src = {src_si.stride0, src_si.logical_iteration0,
                            src_si.stride1, src_si.logical_iteration1,
                            src_si.stride2, src_si.logical_iteration2};
  St_StrideIteration dst = {dst_si.stride0, dst_si.logical_iteration0,
                            dst_si.stride1, dst_si.logical_iteration1,
                            dst_si.stride2, dst_si.logical_iteration2};
  TsmDataMoveInstr instr = {0};
  data_move->GatherScatter(&instr, spm_src_offset, spm_dst_offset,
                           inner_bytes_u32, &src, &dst);
  status = wafer_status_from_u64(TsmExecute(&instr));
  TsmDeleteDataMove(data_move);
  return status;
#endif
}

int32_t wafer_gemm(uint32_t lhs_spm_offset, uint32_t rhs_spm_offset,
                   uint32_t dst_spm_offset, uint64_t m, uint64_t k, uint64_t n,
                   uint32_t data_format) {
  uint32_t m_u32 = 0;
  uint32_t k_u32 = 0;
  uint32_t n_u32 = 0;
  uint64_t element_bytes = 0;
  int bit_packed = 0;
  int32_t status = WAFER_CABI_STATUS_OK;

  if (m == 0 || k == 0 || n == 0 || !wafer_checked_u32(m, &m_u32) ||
      !wafer_checked_u32(k, &k_u32) || !wafer_checked_u32(n, &n_u32) ||
      !wafer_format_element_bytes(data_format, &element_bytes, &bit_packed)) {
    status = WAFER_CABI_STATUS_INVALID_ARGUMENT;
#ifdef WAFER_CABI_SHIM_CAPTURE
    wafer_capture_status(WAFER_CABI_CAPTURE_GEMM, status);
#endif
    return status;
  }

#ifdef WAFER_CABI_SHIM_CAPTURE
  wafer_capture_status(WAFER_CABI_CAPTURE_GEMM, status);
  wafer_last_issue.payload.gemm =
      (wafer_cabi_gemm_capture_t){WAFER_CABI_INTER_TYPE_NEUR,
                                  WAFER_CABI_NE_TYPE_GEMM,
                                  lhs_spm_offset,
                                  rhs_spm_offset,
                                  dst_spm_offset,
                                  m_u32,
                                  k_u32,
                                  n_u32,
                                  1,
                                  1,
                                  0,
                                  0,
                                  data_format,
                                  data_format,
                                  0,
                                  WAFER_CABI_WAIT_ISSUE_ONLY};
  return status;
#else
  TsmGemm *gemm = TsmNewGemm();
  if (gemm == NULL || gemm->AddInput == NULL || gemm->ConfigMKN == NULL ||
      gemm->ConfigBatch == NULL || gemm->AddOutput == NULL ||
      gemm->SetPsum == NULL || gemm->SetTransflag == NULL) {
    if (gemm != NULL)
      TsmDeleteGemm(gemm);
    return WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
  }

  TsmNeInstr instr = {0};
  gemm->AddInput(&instr, lhs_spm_offset, rhs_spm_offset,
                 (Data_Format)data_format);
  gemm->ConfigMKN(&instr, m_u32, k_u32, n_u32);
  gemm->ConfigBatch(&instr, 1, 1);
  gemm->AddOutput(&instr, dst_spm_offset, (Data_Format)data_format);
  gemm->SetPsum(&instr, 0, 0, (Data_Format)data_format);
  gemm->SetTransflag(&instr, 0, 0);
  if (gemm->DisableRelu != NULL)
    gemm->DisableRelu(&instr);
  if (gemm->DisableLeakyRelu != NULL)
    gemm->DisableLeakyRelu(&instr);
  status = wafer_status_from_u64(TsmExecute(&instr));
  TsmDeleteGemm(gemm);
  return status;
#endif
}

int32_t wafer_fill(uint32_t dst_spm_offset, uint64_t value_bits,
                   uint64_t element_count, uint32_t data_format) {
  uint32_t elem_count = 0;
  uint64_t element_bytes = 0;
  int bit_packed = 0;
  int32_t status = WAFER_CABI_STATUS_OK;

  if (!wafer_validate_element_count(element_count, data_format, &elem_count) ||
      !wafer_format_element_bytes(data_format, &element_bytes, &bit_packed) ||
      value_bits > UINT32_MAX) {
    status = WAFER_CABI_STATUS_INVALID_ARGUMENT;
#ifdef WAFER_CABI_SHIM_CAPTURE
    wafer_capture_status(WAFER_CABI_CAPTURE_FILL, status);
#endif
    return status;
  }

#ifdef WAFER_CABI_SHIM_CAPTURE
  wafer_capture_status(WAFER_CABI_CAPTURE_FILL, status);
  wafer_last_issue.payload.fill =
      (wafer_cabi_fill_capture_t){WAFER_CABI_INTER_TYPE_TDMA,
                                  WAFER_CABI_PERIPHERAL_MEMSET_OPCODE,
                                  dst_spm_offset,
                                  value_bits,
                                  elem_count,
                                  data_format,
                                  WAFER_CABI_WAIT_ISSUE_ONLY};
  return status;
#else
  TsmPeripheral *peripheral = TsmNewPeripheral();
  if (peripheral == NULL || peripheral->Memset == NULL) {
    if (peripheral != NULL)
      TsmDeletePeripheral(peripheral);
    return WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
  }
  TsmDataMoveInstr instr = {0};
  St_StrideIteration si = {0, 1, 0, 1, 0, 1};
  peripheral->Memset(&instr, dst_spm_offset, (uint32_t)value_bits, elem_count,
                     &si, (Data_Format)data_format);
  status = wafer_status_from_u64(TsmExecute(&instr));
  TsmDeletePeripheral(peripheral);
  return status;
#endif
}

int32_t wafer_elementwise(uint32_t elementwise_kind, uint32_t dst_spm_offset,
                          uint32_t src0_spm_offset, uint32_t src1_spm_offset,
                          uint64_t element_count, uint32_t input_format,
                          uint32_t output_format) {
  uint32_t elem_count = 0;
  uint32_t opcode = 0;
  uint64_t element_bytes = 0;
  int bit_packed = 0;
  int32_t status = WAFER_CABI_STATUS_OK;
  int is_relation = elementwise_kind >= WAFER_CABI_ELEMENTWISE_EQ &&
                    elementwise_kind <= WAFER_CABI_ELEMENTWISE_GE;

  if (!wafer_validate_element_count(element_count, input_format, &elem_count) ||
      !wafer_format_element_bytes(output_format, &element_bytes, &bit_packed) ||
      !wafer_elementwise_opcode(elementwise_kind, output_format, &opcode) ||
      (!is_relation && input_format != output_format) ||
      (is_relation && output_format != input_format && output_format != 7)) {
    status = WAFER_CABI_STATUS_INVALID_ARGUMENT;
#ifdef WAFER_CABI_SHIM_CAPTURE
    wafer_capture_status(WAFER_CABI_CAPTURE_ELEMENTWISE, status);
#endif
    return status;
  }

#ifdef WAFER_CABI_SHIM_CAPTURE
  wafer_capture_status(WAFER_CABI_CAPTURE_ELEMENTWISE, status);
  wafer_last_issue.payload.elementwise =
      (wafer_cabi_elementwise_capture_t){WAFER_CABI_INTER_TYPE_CGRA,
                                         opcode,
                                         elementwise_kind,
                                         dst_spm_offset,
                                         src0_spm_offset,
                                         src1_spm_offset,
                                         elem_count,
                                         input_format,
                                         output_format,
                                         WAFER_CABI_WAIT_ISSUE_ONLY};
  return status;
#else
  if (elementwise_kind <= WAFER_CABI_ELEMENTWISE_RSQRT) {
    TsmArith *arith = TsmNewArith();
    if (arith == NULL) {
      return WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
    }
    TsmArithInstr instr = {0};
    switch (elementwise_kind) {
    case WAFER_CABI_ELEMENTWISE_ADD:
      if (arith->AddVV == NULL)
        status = WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
      else
        arith->AddVV(&instr, src0_spm_offset, src1_spm_offset, dst_spm_offset,
                     elem_count, (RND_MODE)0, (Data_Format)input_format);
      break;
    case WAFER_CABI_ELEMENTWISE_SUB:
      if (arith->SubVV == NULL)
        status = WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
      else
        arith->SubVV(&instr, src0_spm_offset, src1_spm_offset, dst_spm_offset,
                     elem_count, (RND_MODE)0, (Data_Format)input_format);
      break;
    case WAFER_CABI_ELEMENTWISE_MUL:
      if (arith->MulVV == NULL)
        status = WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
      else
        arith->MulVV(&instr, src0_spm_offset, src1_spm_offset, dst_spm_offset,
                     elem_count, (RND_MODE)0, (Data_Format)input_format);
      break;
    case WAFER_CABI_ELEMENTWISE_DIV:
      if (arith->DivVV == NULL)
        status = WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
      else
        arith->DivVV(&instr, src0_spm_offset, src1_spm_offset, dst_spm_offset,
                     elem_count, (RND_MODE)0, (Data_Format)input_format);
      break;
    case WAFER_CABI_ELEMENTWISE_MAX:
      if (arith->MaxVV == NULL)
        status = WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
      else
        arith->MaxVV(&instr, src0_spm_offset, src1_spm_offset, dst_spm_offset,
                     elem_count, (RND_MODE)0, (Data_Format)input_format);
      break;
    case WAFER_CABI_ELEMENTWISE_MIN:
      if (arith->MinVV == NULL)
        status = WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
      else
        arith->MinVV(&instr, src0_spm_offset, src1_spm_offset, dst_spm_offset,
                     elem_count, (RND_MODE)0, (Data_Format)input_format);
      break;
    case WAFER_CABI_ELEMENTWISE_NEG:
      if (arith->NegVV == NULL)
        status = WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
      else
        arith->NegVV(&instr, src0_spm_offset, dst_spm_offset, elem_count,
                     (Data_Format)input_format);
      break;
    case WAFER_CABI_ELEMENTWISE_RECIP:
      if (arith->RecipVV == NULL)
        status = WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
      else
        arith->RecipVV(&instr, src0_spm_offset, dst_spm_offset, elem_count,
                       (Data_Format)input_format);
      break;
    case WAFER_CABI_ELEMENTWISE_SQRT:
      if (arith->SqrtVV == NULL)
        status = WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
      else
        arith->SqrtVV(&instr, src0_spm_offset, dst_spm_offset, elem_count,
                      (Data_Format)input_format);
      break;
    case WAFER_CABI_ELEMENTWISE_RSQRT:
      if (arith->RsqrtVV == NULL)
        status = WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
      else
        arith->RsqrtVV(&instr, src0_spm_offset, dst_spm_offset, elem_count,
                       (Data_Format)input_format);
      break;
    default:
      status = WAFER_CABI_STATUS_INVALID_ARGUMENT;
      break;
    }
    if (status == WAFER_CABI_STATUS_OK)
      status = wafer_status_from_u64(TsmExecute(&instr));
    TsmDeleteArith(arith);
    return status;
  }

  if (elementwise_kind == WAFER_CABI_ELEMENTWISE_EXP) {
    TsmTranscendental *trans = TsmNewTranscendental();
    if (trans == NULL || trans->Exp == NULL) {
      if (trans != NULL)
        TsmDeleteTranscendental(trans);
      return WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
    }
    TsmTranscendentalInstr instr = {0};
    trans->Exp(&instr, src0_spm_offset, dst_spm_offset, elem_count,
               (Data_Format)input_format);
    status = wafer_status_from_u64(TsmExecute(&instr));
    TsmDeleteTranscendental(trans);
    return status;
  }

  if (elementwise_kind == WAFER_CABI_ELEMENTWISE_TANH) {
    TsmActivation *activation = TsmNewActivation();
    if (activation == NULL || activation->Tanh == NULL) {
      if (activation != NULL)
        TsmDeleteActivation(activation);
      return WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
    }
    TsmActivationInstr instr = {0};
    activation->Tanh(&instr, src0_spm_offset, dst_spm_offset, elem_count,
                     (Data_Format)input_format);
    status = wafer_status_from_u64(TsmExecute(&instr));
    TsmDeleteActivation(activation);
    return status;
  }

  TsmRelation *relation = TsmNewRelation();
  if (relation == NULL) {
    return WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
  }
  TsmRelationInstr instr = {0};
  int bool_output = output_format == 7;
  switch (elementwise_kind) {
  case WAFER_CABI_ELEMENTWISE_EQ:
    if (bool_output && relation->BoolEqualVV != NULL)
      relation->BoolEqualVV(&instr, src0_spm_offset, src1_spm_offset,
                            dst_spm_offset, elem_count,
                            (Data_Format)input_format);
    else if (!bool_output && relation->EqualVV != NULL)
      relation->EqualVV(&instr, src0_spm_offset, src1_spm_offset,
                        dst_spm_offset, elem_count, (Data_Format)input_format);
    else
      status = WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
    break;
  case WAFER_CABI_ELEMENTWISE_NE:
    if (bool_output && relation->BoolUnEqualVV != NULL)
      relation->BoolUnEqualVV(&instr, src0_spm_offset, src1_spm_offset,
                              dst_spm_offset, elem_count,
                              (Data_Format)input_format);
    else if (!bool_output && relation->UnEqualVV != NULL)
      relation->UnEqualVV(&instr, src0_spm_offset, src1_spm_offset,
                          dst_spm_offset, elem_count,
                          (Data_Format)input_format);
    else
      status = WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
    break;
  case WAFER_CABI_ELEMENTWISE_LT:
    if (bool_output && relation->BoolLessThenVV != NULL)
      relation->BoolLessThenVV(&instr, src0_spm_offset, src1_spm_offset,
                               dst_spm_offset, elem_count,
                               (Data_Format)input_format);
    else if (!bool_output && relation->LessThenVV != NULL)
      relation->LessThenVV(&instr, src0_spm_offset, src1_spm_offset,
                           dst_spm_offset, elem_count,
                           (Data_Format)input_format);
    else
      status = WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
    break;
  case WAFER_CABI_ELEMENTWISE_LE:
    if (bool_output && relation->BoolLessEqualVV != NULL)
      relation->BoolLessEqualVV(&instr, src0_spm_offset, src1_spm_offset,
                                dst_spm_offset, elem_count,
                                (Data_Format)input_format);
    else if (!bool_output && relation->LessEqualVV != NULL)
      relation->LessEqualVV(&instr, src0_spm_offset, src1_spm_offset,
                            dst_spm_offset, elem_count,
                            (Data_Format)input_format);
    else
      status = WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
    break;
  case WAFER_CABI_ELEMENTWISE_GT:
    if (bool_output && relation->BoolGreaterVV != NULL)
      relation->BoolGreaterVV(&instr, src0_spm_offset, src1_spm_offset,
                              dst_spm_offset, elem_count,
                              (Data_Format)input_format);
    else if (!bool_output && relation->GreaterVV != NULL)
      relation->GreaterVV(&instr, src0_spm_offset, src1_spm_offset,
                          dst_spm_offset, elem_count,
                          (Data_Format)input_format);
    else
      status = WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
    break;
  case WAFER_CABI_ELEMENTWISE_GE:
    if (bool_output && relation->BoolGreaterEqualVV != NULL)
      relation->BoolGreaterEqualVV(&instr, src0_spm_offset, src1_spm_offset,
                                   dst_spm_offset, elem_count,
                                   (Data_Format)input_format);
    else if (!bool_output && relation->GreaterEqualVV != NULL)
      relation->GreaterEqualVV(&instr, src0_spm_offset, src1_spm_offset,
                               dst_spm_offset, elem_count,
                               (Data_Format)input_format);
    else
      status = WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
    break;
  default:
    status = WAFER_CABI_STATUS_INVALID_ARGUMENT;
    break;
  }
  if (status == WAFER_CABI_STATUS_OK)
    status = wafer_status_from_u64(TsmExecute(&instr));
  TsmDeleteRelation(relation);
  return status;
#endif
}

int32_t wafer_reduce(uint32_t reduce_kind, uint32_t src_spm_offset,
                     uint32_t dst_spm_offset, uint32_t dim, uint64_t n,
                     uint64_t h, uint64_t w, uint64_t c, uint32_t data_format) {
  uint32_t opcode = 0;
  uint16_t n_u16 = 0;
  uint16_t h_u16 = 0;
  uint16_t w_u16 = 0;
  uint16_t c_u16 = 0;
  uint64_t element_bytes = 0;
  int bit_packed = 0;
  int32_t status = WAFER_CABI_STATUS_OK;

  if (dim > 5 || n == 0 || h == 0 || w == 0 || c == 0 ||
      !wafer_reduce_opcode(reduce_kind, &opcode) ||
      !wafer_checked_u16(n, &n_u16) || !wafer_checked_u16(h, &h_u16) ||
      !wafer_checked_u16(w, &w_u16) || !wafer_checked_u16(c, &c_u16) ||
      !wafer_format_element_bytes(data_format, &element_bytes, &bit_packed)) {
    status = WAFER_CABI_STATUS_INVALID_ARGUMENT;
#ifdef WAFER_CABI_SHIM_CAPTURE
    wafer_capture_status(WAFER_CABI_CAPTURE_REDUCE, status);
#endif
    return status;
  }

#ifdef WAFER_CABI_SHIM_CAPTURE
  wafer_capture_status(WAFER_CABI_CAPTURE_REDUCE, status);
  wafer_last_issue.payload.reduce =
      (wafer_cabi_reduce_capture_t){WAFER_CABI_INTER_TYPE_CGRA,
                                    opcode,
                                    reduce_kind,
                                    src_spm_offset,
                                    dst_spm_offset,
                                    dim,
                                    n_u16,
                                    h_u16,
                                    w_u16,
                                    c_u16,
                                    data_format,
                                    WAFER_CABI_WAIT_ISSUE_ONLY};
  return status;
#else
  TsmReduce *reduce = TsmNewReduce();
  if (reduce == NULL) {
    return WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
  }
  TsmReduceInstr instr = {0};
  Data_Shape shape = {n_u16, h_u16, w_u16, c_u16};
  switch (reduce_kind) {
  case WAFER_CABI_REDUCE_SUM:
    if (reduce->ReduceSum == NULL)
      status = WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
    else
      reduce->ReduceSum(&instr, src_spm_offset, dst_spm_offset, dim, shape,
                        (Data_Format)data_format);
    break;
  case WAFER_CABI_REDUCE_AVG:
    if (reduce->ReduceAvg == NULL)
      status = WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
    else
      reduce->ReduceAvg(&instr, src_spm_offset, dst_spm_offset, dim, shape,
                        (Data_Format)data_format);
    break;
  case WAFER_CABI_REDUCE_MAX:
    if (reduce->ReduceMax == NULL)
      status = WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
    else
      reduce->ReduceMax(&instr, src_spm_offset, dst_spm_offset, dim, shape,
                        (Data_Format)data_format);
    break;
  case WAFER_CABI_REDUCE_MIN:
    if (reduce->ReduceMin == NULL)
      status = WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
    else
      reduce->ReduceMin(&instr, src_spm_offset, dst_spm_offset, dim, shape,
                        (Data_Format)data_format);
    break;
  default:
    status = WAFER_CABI_STATUS_INVALID_ARGUMENT;
    break;
  }
  if (status == WAFER_CABI_STATUS_OK)
    status = wafer_status_from_u64(TsmExecute(&instr));
  TsmDeleteReduce(reduce);
  return status;
#endif
}

int32_t wafer_convert(uint32_t src_format, uint32_t dst_format,
                      uint32_t src_spm_offset, uint32_t dst_spm_offset,
                      uint64_t element_count) {
  uint32_t elem_count = 0;
  uint32_t opcode = 0;
  uint64_t element_bytes = 0;
  int bit_packed = 0;
  int32_t status = WAFER_CABI_STATUS_OK;

  if (!wafer_validate_element_count(element_count, src_format, &elem_count) ||
      !wafer_format_element_bytes(dst_format, &element_bytes, &bit_packed) ||
      !wafer_convert_opcode(src_format, dst_format, &opcode)) {
    status = WAFER_CABI_STATUS_INVALID_ARGUMENT;
#ifdef WAFER_CABI_SHIM_CAPTURE
    wafer_capture_status(WAFER_CABI_CAPTURE_CONVERT, status);
#endif
    return status;
  }

#ifdef WAFER_CABI_SHIM_CAPTURE
  wafer_capture_status(WAFER_CABI_CAPTURE_CONVERT, status);
  wafer_last_issue.payload.convert =
      (wafer_cabi_convert_capture_t){WAFER_CABI_INTER_TYPE_CGRA,
                                     opcode,
                                     src_format,
                                     dst_format,
                                     src_spm_offset,
                                     dst_spm_offset,
                                     elem_count,
                                     WAFER_CABI_WAIT_ISSUE_ONLY};
  return status;
#else
  if (src_format == dst_format) {
    uint64_t bytes = 0;
    if (bit_packed) {
      bytes = (element_count + 7) / 8;
    } else if (!wafer_checked_mul_u64(element_count, element_bytes, &bytes)) {
      return WAFER_CABI_STATUS_INVALID_ARGUMENT;
    }
    if (bytes > UINT32_MAX)
      return WAFER_CABI_STATUS_INVALID_ARGUMENT;
    TsmDataMove *data_move = TsmNewDataMove();
    if (data_move == NULL || data_move->GatherScatter == NULL) {
      if (data_move != NULL)
        TsmDeleteDataMove(data_move);
      return WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
    }
    St_StrideIteration si = {0, 1, 0, 1, 0, 1};
    TsmDataMoveInstr instr = {0};
    data_move->GatherScatter(&instr, src_spm_offset, dst_spm_offset,
                             (uint32_t)bytes, &si, &si);
    status = wafer_status_from_u64(TsmExecute(&instr));
    TsmDeleteDataMove(data_move);
    return status;
  }

  TsmConvert *convert = TsmNewConvert();
  if (convert == NULL)
    return WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
  TsmConvertInstr instr = {0};

#define WAFER_CALL_CONVERT_ZP(FN)                                              \
  do {                                                                         \
    if (convert->FN == NULL)                                                   \
      status = WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;                          \
    else                                                                       \
      convert->FN(&instr, src_spm_offset, 0, dst_spm_offset, elem_count);      \
  } while (0)
#define WAFER_CALL_CONVERT(FN)                                                 \
  do {                                                                         \
    if (convert->FN == NULL)                                                   \
      status = WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;                          \
    else                                                                       \
      convert->FN(&instr, src_spm_offset, dst_spm_offset, elem_count);         \
  } while (0)
#define WAFER_CALL_CONVERT_RND(FN)                                             \
  do {                                                                         \
    if (convert->FN == NULL)                                                   \
      status = WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;                          \
    else                                                                       \
      convert->FN(&instr, src_spm_offset, dst_spm_offset, elem_count,          \
                  (RND_MODE)0);                                                \
  } while (0)

  switch (opcode) {
  case 139:
    WAFER_CALL_CONVERT_ZP(INT8_FP16);
    break;
  case 140:
    WAFER_CALL_CONVERT_ZP(INT8_BF16);
    break;
  case 141:
    WAFER_CALL_CONVERT_ZP(INT8_FP32);
    break;
  case 142:
    WAFER_CALL_CONVERT_ZP(INT8_TF32);
    break;
  case 143:
    WAFER_CALL_CONVERT(INT16_FP16);
    break;
  case 144:
    WAFER_CALL_CONVERT_RND(INT16_BF16);
    break;
  case 145:
    WAFER_CALL_CONVERT_RND(INT16_FP32);
    break;
  case 146:
    WAFER_CALL_CONVERT_RND(INT16_TF32);
    break;
  case 147:
    WAFER_CALL_CONVERT_RND(INT32_FP16);
    break;
  case 148:
    WAFER_CALL_CONVERT_RND(INT32_BF16);
    break;
  case 149:
    WAFER_CALL_CONVERT_RND(INT32_FP32);
    break;
  case 150:
    WAFER_CALL_CONVERT_RND(INT32_TF32);
    break;
  case 151:
    WAFER_CALL_CONVERT(BF16_INT8);
    break;
  case 152:
    WAFER_CALL_CONVERT_RND(BF16_INT16);
    break;
  case 153:
    WAFER_CALL_CONVERT_RND(BF16_INT32);
    break;
  case 154:
    WAFER_CALL_CONVERT(BF16_FP16);
    break;
  case 155:
    WAFER_CALL_CONVERT(BF16_FP32);
    break;
  case 156:
    WAFER_CALL_CONVERT(BF16_TF32);
    break;
  case 157:
    WAFER_CALL_CONVERT_RND(FP16_INT8);
    break;
  case 158:
    WAFER_CALL_CONVERT_RND(FP16_INT16);
    break;
  case 159:
    WAFER_CALL_CONVERT_RND(FP16_INT32);
    break;
  case 160:
    WAFER_CALL_CONVERT_RND(FP16_BF16);
    break;
  case 161:
    WAFER_CALL_CONVERT(FP16_FP32);
    break;
  case 162:
    WAFER_CALL_CONVERT(FP16_TF32);
    break;
  case 163:
    WAFER_CALL_CONVERT_RND(FP32_INT8);
    break;
  case 164:
    WAFER_CALL_CONVERT_RND(FP32_INT16);
    break;
  case 165:
    WAFER_CALL_CONVERT_RND(FP32_INT32);
    break;
  case 166:
    WAFER_CALL_CONVERT_RND(FP32_FP16);
    break;
  case 167:
    WAFER_CALL_CONVERT_RND(FP32_BF16);
    break;
  case 168:
    WAFER_CALL_CONVERT_RND(FP32_TF32);
    break;
  case 169:
    WAFER_CALL_CONVERT_RND(TF32_INT8);
    break;
  case 170:
    WAFER_CALL_CONVERT_RND(TF32_INT16);
    break;
  case 171:
    WAFER_CALL_CONVERT_RND(TF32_INT32);
    break;
  case 172:
    WAFER_CALL_CONVERT(TF32_FP16);
    break;
  case 173:
    WAFER_CALL_CONVERT_RND(TF32_BF16);
    break;
  case 174:
    WAFER_CALL_CONVERT(TF32_FP32);
    break;
  default:
    status = WAFER_CABI_STATUS_INVALID_ARGUMENT;
    break;
  }

#undef WAFER_CALL_CONVERT_ZP
#undef WAFER_CALL_CONVERT
#undef WAFER_CALL_CONVERT_RND

  if (status == WAFER_CABI_STATUS_OK)
    status = wafer_status_from_u64(TsmExecute(&instr));
  TsmDeleteConvert(convert);
  return status;
#endif
}

int32_t wafer_dte_send(uint32_t buffer_spm_offset, uint32_t peer,
                       uint64_t byte_count) {
  uint32_t bytes = 0;
  int32_t status = WAFER_CABI_STATUS_OK;
  if (byte_count == 0 || !wafer_checked_u32(byte_count, &bytes)) {
    status = WAFER_CABI_STATUS_INVALID_ARGUMENT;
#ifdef WAFER_CABI_SHIM_CAPTURE
    wafer_capture_status(WAFER_CABI_CAPTURE_DTE_SEND, status);
#endif
    return status;
  }

#ifdef WAFER_CABI_SHIM_CAPTURE
  wafer_capture_status(WAFER_CABI_CAPTURE_DTE_SEND, status);
  wafer_last_issue.payload.dte = (wafer_cabi_dte_capture_t){
      1, buffer_spm_offset, peer, bytes, WAFER_CABI_WAIT_ISSUE_ONLY};
  return status;
#else
  (void)buffer_spm_offset;
  (void)peer;
  return WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
#endif
}

int32_t wafer_dte_recv(uint32_t buffer_spm_offset, uint32_t peer,
                       uint64_t byte_count) {
  uint32_t bytes = 0;
  int32_t status = WAFER_CABI_STATUS_OK;
  if (byte_count == 0 || !wafer_checked_u32(byte_count, &bytes)) {
    status = WAFER_CABI_STATUS_INVALID_ARGUMENT;
#ifdef WAFER_CABI_SHIM_CAPTURE
    wafer_capture_status(WAFER_CABI_CAPTURE_DTE_RECV, status);
#endif
    return status;
  }

#ifdef WAFER_CABI_SHIM_CAPTURE
  wafer_capture_status(WAFER_CABI_CAPTURE_DTE_RECV, status);
  wafer_last_issue.payload.dte = (wafer_cabi_dte_capture_t){
      0, buffer_spm_offset, peer, bytes, WAFER_CABI_WAIT_ISSUE_ONLY};
  return status;
#else
  (void)buffer_spm_offset;
  (void)peer;
  return WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
#endif
}

int32_t wafer_dte_wait(uint32_t token_count) {
  if (token_count == 0) {
#ifdef WAFER_CABI_SHIM_CAPTURE
    wafer_capture_status(WAFER_CABI_CAPTURE_DTE_WAIT,
                         WAFER_CABI_STATUS_INVALID_ARGUMENT);
    wafer_last_issue.payload.dte_wait = (wafer_cabi_dte_wait_capture_t){
        token_count, WAFER_CABI_WAIT_LOCAL_WAIT};
#else
    (void)token_count;
#endif
    return WAFER_CABI_STATUS_INVALID_ARGUMENT;
  }

#ifdef WAFER_CABI_SHIM_CAPTURE
  wafer_capture_status(WAFER_CABI_CAPTURE_DTE_WAIT, WAFER_CABI_STATUS_OK);
  wafer_last_issue.payload.dte_wait =
      (wafer_cabi_dte_wait_capture_t){token_count, WAFER_CABI_WAIT_LOCAL_WAIT};
  return WAFER_CABI_STATUS_OK;
#else
  (void)token_count;
  return WAFER_CABI_STATUS_WRAPPER_UNAVAILABLE;
#endif
}

int32_t wafer_local_fence(void) {
#ifdef WAFER_CABI_SHIM_CAPTURE
  wafer_capture_status(WAFER_CABI_CAPTURE_LOCAL_FENCE, WAFER_CABI_STATUS_OK);
  wafer_last_issue.payload.local_fence =
      (wafer_cabi_local_fence_capture_t){1, WAFER_CABI_WAIT_LOCAL_WAIT};
  return WAFER_CABI_STATUS_OK;
#else
  return (int32_t)TsmWaitfinish();
#endif
}
