#include "wafer_cabi_shim.h"

#include <stdio.h>

int main(void) {
  const wafer_cabi_last_issue_t *issue = NULL;

  wafer_cabi_reset_capture();
  printf("rdma status=%d\n",
         wafer_rdma(0x280000200ULL, 65536, 512, 64, 128, 0, 0, 4, 1, 1, 2));
  issue = wafer_cabi_get_last_issue();
  printf("rdma kind=%u inter=%u src=%llu dst=%llu elem=%u stride0=%u "
         "iter0=%u fmt=%u src_end=%llu dst_end=%llu wait=%u\n",
         issue->kind, issue->payload.dma.inter_type,
         (unsigned long long)issue->payload.dma.src,
         (unsigned long long)issue->payload.dma.dst,
         issue->payload.dma.elem_count, issue->payload.dma.stride0,
         issue->payload.dma.iteration0, issue->payload.dma.format,
         (unsigned long long)issue->payload.dma.src_end,
         (unsigned long long)issue->payload.dma.dst_end,
         issue->payload.dma.wait_policy);

  printf("wdma status=%d\n",
         wafer_wdma(2048, 0x280001000ULL, 128, 32, 64, 0, 0, 4, 1, 1, 2));
  issue = wafer_cabi_get_last_issue();
  printf("wdma kind=%u inter=%u src=%llu dst=%llu elem=%u iter0=%u "
         "src_end=%llu dst_end=%llu wait=%u\n",
         issue->kind, issue->payload.dma.inter_type,
         (unsigned long long)issue->payload.dma.src,
         (unsigned long long)issue->payload.dma.dst,
         issue->payload.dma.elem_count, issue->payload.dma.iteration0,
         (unsigned long long)issue->payload.dma.src_end,
         (unsigned long long)issue->payload.dma.dst_end,
         issue->payload.dma.wait_policy);

  printf("gather_scatter status=%d\n",
         wafer_gather_scatter(4096, 8192, 128, 32, 64, 0, 0, 4, 1, 1,
                              96, 0, 0, 4, 1, 1));
  issue = wafer_cabi_get_last_issue();
  printf("gather_scatter kind=%u inter=%u opcode=%u src=%u dst=%u "
         "bytes=%u src_iter0=%u dst_iter0=%u src_end=%u dst_end=%u wait=%u\n",
         issue->kind, issue->payload.gather_scatter.inter_type,
         issue->payload.gather_scatter.opcode,
         issue->payload.gather_scatter.src0,
         issue->payload.gather_scatter.dst,
         issue->payload.gather_scatter.elem_count,
         issue->payload.gather_scatter.src_iteration0,
         issue->payload.gather_scatter.dst_iteration0,
         issue->payload.gather_scatter.src0_end,
         issue->payload.gather_scatter.dst_end,
         issue->payload.gather_scatter.wait_policy);

  printf("gemm status=%d\n", wafer_gemm(65536, 65792, 66048, 4, 8, 16, 2));
  issue = wafer_cabi_get_last_issue();
  printf("gemm kind=%u inter=%u type=%u lhs=%u rhs=%u dst=%u m=%u k=%u n=%u "
         "in_fmt=%u out_fmt=%u wait=%u\n",
         issue->kind, issue->payload.gemm.inter_type, issue->payload.gemm.type,
         issue->payload.gemm.lhs, issue->payload.gemm.rhs,
         issue->payload.gemm.dest, issue->payload.gemm.gemm_m,
         issue->payload.gemm.gemm_k, issue->payload.gemm.gemm_n,
         issue->payload.gemm.input_format, issue->payload.gemm.output_format,
         issue->payload.gemm.wait_policy);

  printf("local_fence status=%d\n", wafer_local_fence());
  issue = wafer_cabi_get_last_issue();
  printf("local_fence kind=%u wait=%u local_wait=%u\n", issue->kind,
         issue->payload.local_fence.wait_policy,
         issue->payload.local_fence.calls_local_wait);

  return 0;
}
