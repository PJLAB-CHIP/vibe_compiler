#ifndef WAFER_CABI_SHIM_H
#define WAFER_CABI_SHIM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int32_t wafer_rdma(uint64_t ddr_src_addr, uint32_t spm_dst_offset,
                   uint64_t byte_count, uint64_t inner_bytes,
                   uint64_t src_stride0_b, uint64_t src_stride1_b,
                   uint64_t src_stride2_b, uint64_t src_iter0,
                   uint64_t src_iter1, uint64_t src_iter2,
                   uint32_t data_format);

int32_t wafer_wdma(uint32_t spm_src_offset, uint64_t ddr_dst_addr,
                   uint64_t byte_count, uint64_t inner_bytes,
                   uint64_t dst_stride0_b, uint64_t dst_stride1_b,
                   uint64_t dst_stride2_b, uint64_t dst_iter0,
                   uint64_t dst_iter1, uint64_t dst_iter2,
                   uint32_t data_format);

int32_t wafer_gather_scatter(uint32_t spm_src_offset, uint32_t spm_dst_offset,
                             uint64_t byte_count, uint64_t inner_bytes,
                             uint64_t src_stride0_b, uint64_t src_stride1_b,
                             uint64_t src_stride2_b, uint64_t src_iter0,
                             uint64_t src_iter1, uint64_t src_iter2,
                             uint64_t dst_stride0_b, uint64_t dst_stride1_b,
                             uint64_t dst_stride2_b, uint64_t dst_iter0,
                             uint64_t dst_iter1, uint64_t dst_iter2);

int32_t wafer_gemm(uint32_t lhs_spm_offset, uint32_t rhs_spm_offset,
                   uint32_t dst_spm_offset, uint64_t m, uint64_t k, uint64_t n,
                   uint32_t data_format);

int32_t wafer_fill(uint32_t dst_spm_offset, uint64_t value_bits,
                   uint64_t element_count, uint32_t data_format);

int32_t wafer_elementwise(uint32_t elementwise_kind, uint32_t dst_spm_offset,
                          uint32_t src0_spm_offset, uint32_t src1_spm_offset,
                          uint64_t element_count, uint32_t input_format,
                          uint32_t output_format);

int32_t wafer_select(uint32_t dst_spm_offset, uint32_t predicate_spm_offset,
                     uint32_t true_spm_offset, uint32_t false_spm_offset,
                     uint64_t element_count, uint32_t value_format);

int32_t wafer_reduce(uint32_t reduce_kind, uint32_t src_spm_offset,
                     uint32_t dst_spm_offset, uint32_t dim, uint64_t n,
                     uint64_t h, uint64_t w, uint64_t c, uint32_t data_format);

int32_t wafer_convert(uint32_t src_format, uint32_t dst_format,
                      uint32_t src_spm_offset, uint32_t dst_spm_offset,
                      uint64_t element_count);

int32_t wafer_dte_send(uint32_t buffer_spm_offset, uint32_t peer,
                       uint64_t byte_count);
int32_t wafer_dte_recv(uint32_t buffer_spm_offset, uint32_t peer,
                       uint64_t byte_count);
int32_t wafer_dte_wait(uint32_t token_count);

int32_t wafer_local_fence(void);

#ifdef WAFER_CABI_SHIM_CAPTURE

enum {
  WAFER_CABI_CAPTURE_NONE = 0,
  WAFER_CABI_CAPTURE_RDMA = 1,
  WAFER_CABI_CAPTURE_WDMA = 2,
  WAFER_CABI_CAPTURE_GATHER_SCATTER = 3,
  WAFER_CABI_CAPTURE_GEMM = 4,
  WAFER_CABI_CAPTURE_LOCAL_FENCE = 5,
  WAFER_CABI_CAPTURE_FILL = 6,
  WAFER_CABI_CAPTURE_ELEMENTWISE = 7,
  WAFER_CABI_CAPTURE_REDUCE = 8,
  WAFER_CABI_CAPTURE_CONVERT = 9,
  WAFER_CABI_CAPTURE_DTE_SEND = 10,
  WAFER_CABI_CAPTURE_DTE_RECV = 11,
  WAFER_CABI_CAPTURE_DTE_WAIT = 12,
};

enum {
  WAFER_CABI_WAIT_ISSUE_ONLY = 0,
  WAFER_CABI_WAIT_LOCAL_WAIT = 1,
};

typedef struct wafer_cabi_dma_capture {
  uint32_t inter_type;
  uint64_t src;
  uint64_t dst;
  uint32_t stride0;
  uint32_t iteration0;
  uint32_t stride1;
  uint32_t iteration1;
  uint32_t stride2;
  uint32_t iteration2;
  uint32_t elem_count;
  uint32_t format;
  uint64_t src_end;
  uint64_t dst_end;
  uint32_t wait_policy;
} wafer_cabi_dma_capture_t;

typedef struct wafer_cabi_gather_scatter_capture {
  uint32_t inter_type;
  uint32_t opcode;
  uint32_t src0;
  uint32_t dst;
  uint32_t elem_count;
  uint32_t src_stride0;
  uint32_t src_iteration0;
  uint32_t src_stride1;
  uint32_t src_iteration1;
  uint32_t src_stride2;
  uint32_t src_iteration2;
  uint32_t dst_stride0;
  uint32_t dst_iteration0;
  uint32_t dst_stride1;
  uint32_t dst_iteration1;
  uint32_t dst_stride2;
  uint32_t dst_iteration2;
  uint32_t src0_end;
  uint32_t dst_end;
  uint32_t wait_policy;
} wafer_cabi_gather_scatter_capture_t;

typedef struct wafer_cabi_gemm_capture {
  uint32_t inter_type;
  uint32_t type;
  uint32_t lhs;
  uint32_t rhs;
  uint32_t dest;
  uint32_t gemm_m;
  uint32_t gemm_k;
  uint32_t gemm_n;
  uint32_t left_batch;
  uint32_t right_batch;
  uint32_t left_trans;
  uint32_t right_trans;
  uint32_t input_format;
  uint32_t output_format;
  uint32_t psum_enabled;
  uint32_t wait_policy;
} wafer_cabi_gemm_capture_t;

typedef struct wafer_cabi_fill_capture {
  uint32_t inter_type;
  uint32_t opcode;
  uint32_t dst;
  uint64_t value_bits;
  uint32_t elem_count;
  uint32_t format;
  uint32_t wait_policy;
} wafer_cabi_fill_capture_t;

typedef struct wafer_cabi_elementwise_capture {
  uint32_t inter_type;
  uint32_t opcode;
  uint32_t kind;
  uint32_t dst;
  uint32_t src0;
  uint32_t src1;
  uint32_t src2;
  uint32_t elem_count;
  uint32_t input_format;
  uint32_t output_format;
  uint32_t wait_policy;
} wafer_cabi_elementwise_capture_t;

typedef struct wafer_cabi_reduce_capture {
  uint32_t inter_type;
  uint32_t opcode;
  uint32_t kind;
  uint32_t src;
  uint32_t dst;
  uint32_t dim;
  uint32_t n;
  uint32_t h;
  uint32_t w;
  uint32_t c;
  uint32_t format;
  uint32_t wait_policy;
} wafer_cabi_reduce_capture_t;

typedef struct wafer_cabi_convert_capture {
  uint32_t inter_type;
  uint32_t opcode;
  uint32_t src_format;
  uint32_t dst_format;
  uint32_t src;
  uint32_t dst;
  uint32_t elem_count;
  uint32_t wait_policy;
} wafer_cabi_convert_capture_t;

typedef struct wafer_cabi_dte_capture {
  uint32_t is_send;
  uint32_t buffer;
  uint32_t peer;
  uint32_t bytes;
  uint32_t wait_policy;
} wafer_cabi_dte_capture_t;

typedef struct wafer_cabi_dte_wait_capture {
  uint32_t token_count;
  uint32_t wait_policy;
} wafer_cabi_dte_wait_capture_t;

typedef struct wafer_cabi_local_fence_capture {
  uint32_t calls_local_wait;
  uint32_t wait_policy;
} wafer_cabi_local_fence_capture_t;

typedef struct wafer_cabi_last_issue {
  uint32_t kind;
  int32_t status;
  union {
    wafer_cabi_dma_capture_t dma;
    wafer_cabi_gather_scatter_capture_t gather_scatter;
    wafer_cabi_gemm_capture_t gemm;
    wafer_cabi_fill_capture_t fill;
    wafer_cabi_elementwise_capture_t elementwise;
    wafer_cabi_reduce_capture_t reduce;
    wafer_cabi_convert_capture_t convert;
    wafer_cabi_dte_capture_t dte;
    wafer_cabi_dte_wait_capture_t dte_wait;
    wafer_cabi_local_fence_capture_t local_fence;
  } payload;
} wafer_cabi_last_issue_t;

void wafer_cabi_reset_capture(void);
const wafer_cabi_last_issue_t *wafer_cabi_get_last_issue(void);

#endif

#ifdef __cplusplus
}
#endif

#endif
