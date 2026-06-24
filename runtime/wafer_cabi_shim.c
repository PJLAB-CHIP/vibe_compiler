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
  WAFER_CABI_INTER_TYPE_NEUR = 1,
  WAFER_CABI_INTER_TYPE_RDMA = 2,
  WAFER_CABI_INTER_TYPE_WDMA = 3,
  WAFER_CABI_INTER_TYPE_TDMA = 4,
  WAFER_CABI_NE_TYPE_GEMM = 3,
  WAFER_CABI_TDMA_GATHER_SCATTER_OPCODE = 135,
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

static int wafer_checked_add_u64(uint64_t lhs, uint64_t rhs,
                                 uint64_t *result) {
  if (rhs > UINT64_MAX - lhs)
    return 0;
  *result = lhs + rhs;
  return 1;
}

static int wafer_checked_mul_u64(uint64_t lhs, uint64_t rhs,
                                 uint64_t *result) {
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

static int wafer_build_stride_iteration(
    uint64_t stride0, uint64_t stride1, uint64_t stride2, uint64_t iteration0,
    uint64_t iteration1, uint64_t iteration2,
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

  if (!wafer_build_stride_iteration(src_stride0_b, src_stride1_b,
                                    src_stride2_b, src_iter0, src_iter1,
                                    src_iter2, &si) ||
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
  rdma->ConfigStrideIteration(&instr, elem_count, si.stride0,
                              si.logical_iteration0, si.stride1,
                              si.logical_iteration1, si.stride2,
                              si.logical_iteration2);
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

  if (!wafer_build_stride_iteration(dst_stride0_b, dst_stride1_b,
                                    dst_stride2_b, dst_iter0, dst_iter1,
                                    dst_iter2, &si) ||
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
  wdma->ConfigStrideIteration(&instr, elem_count, si.stride0,
                              si.logical_iteration0, si.stride1,
                              si.logical_iteration1, si.stride2,
                              si.logical_iteration2);
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
      !wafer_build_stride_iteration(src_stride0_b, src_stride1_b,
                                    src_stride2_b, src_iter0, src_iter1,
                                    src_iter2, &src_si) ||
      !wafer_build_stride_iteration(dst_stride0_b, dst_stride1_b,
                                    dst_stride2_b, dst_iter0, dst_iter1,
                                    dst_iter2, &dst_si) ||
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

  St_StrideIteration src = {src_si.stride0,
                            src_si.logical_iteration0,
                            src_si.stride1,
                            src_si.logical_iteration1,
                            src_si.stride2,
                            src_si.logical_iteration2};
  St_StrideIteration dst = {dst_si.stride0,
                            dst_si.logical_iteration0,
                            dst_si.stride1,
                            dst_si.logical_iteration1,
                            dst_si.stride2,
                            dst_si.logical_iteration2};
  TsmDataMoveInstr instr = {0};
  data_move->GatherScatter(&instr, spm_src_offset, spm_dst_offset,
                           inner_bytes_u32, &src, &dst);
  status = wafer_status_from_u64(TsmExecute(&instr));
  TsmDeleteDataMove(data_move);
  return status;
#endif
}

int32_t wafer_gemm(uint32_t lhs_spm_offset, uint32_t rhs_spm_offset,
                   uint32_t dst_spm_offset, uint64_t m, uint64_t k,
                   uint64_t n, uint32_t data_format) {
  uint32_t m_u32 = 0;
  uint32_t k_u32 = 0;
  uint32_t n_u32 = 0;
  uint64_t element_bytes = 0;
  int bit_packed = 0;
  int32_t status = WAFER_CABI_STATUS_OK;

  if (m == 0 || k == 0 || n == 0 ||
      !wafer_checked_u32(m, &m_u32) || !wafer_checked_u32(k, &k_u32) ||
      !wafer_checked_u32(n, &n_u32) ||
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
