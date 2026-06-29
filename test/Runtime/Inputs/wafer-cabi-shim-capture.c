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

  printf("fill status=%d\n", wafer_fill(66176, 0, 8, 5));
  issue = wafer_cabi_get_last_issue();
  printf("fill kind=%u inter=%u opcode=%u dst=%u value=%llu elements=%u "
         "fmt=%u wait=%u\n",
         issue->kind, issue->payload.fill.inter_type,
         issue->payload.fill.opcode, issue->payload.fill.dst,
         (unsigned long long)issue->payload.fill.value_bits,
         issue->payload.fill.elem_count, issue->payload.fill.format,
         issue->payload.fill.wait_policy);

  printf("elementwise status=%d\n",
         wafer_elementwise(0, 66560, 66176, 66304, 8, 5, 5));
  issue = wafer_cabi_get_last_issue();
  printf("elementwise kind=%u inter=%u opcode=%u op=%u dst=%u src0=%u "
         "src1=%u elements=%u in_fmt=%u out_fmt=%u wait=%u\n",
         issue->kind, issue->payload.elementwise.inter_type,
         issue->payload.elementwise.opcode, issue->payload.elementwise.kind,
         issue->payload.elementwise.dst, issue->payload.elementwise.src0,
         issue->payload.elementwise.src1, issue->payload.elementwise.elem_count,
         issue->payload.elementwise.input_format,
         issue->payload.elementwise.output_format,
         issue->payload.elementwise.wait_policy);

  printf("select status=%d\n", wafer_select(67072, 67136, 67264, 67392, 8, 5));
  issue = wafer_cabi_get_last_issue();
  printf("select kind=%u inter=%u opcode=%u op=%u dst=%u pred=%u true=%u "
         "false=%u elements=%u pred_fmt=%u out_fmt=%u wait=%u\n",
         issue->kind, issue->payload.elementwise.inter_type,
         issue->payload.elementwise.opcode, issue->payload.elementwise.kind,
         issue->payload.elementwise.dst, issue->payload.elementwise.src0,
         issue->payload.elementwise.src1, issue->payload.elementwise.src2,
         issue->payload.elementwise.elem_count,
         issue->payload.elementwise.input_format,
         issue->payload.elementwise.output_format,
         issue->payload.elementwise.wait_policy);

  printf("reduce status=%d\n", wafer_reduce(0, 66560, 66688, 0, 1, 1, 1, 8, 5));
  issue = wafer_cabi_get_last_issue();
  printf("reduce kind=%u inter=%u opcode=%u op=%u src=%u dst=%u dim=%u "
         "shape=%ux%ux%ux%u fmt=%u wait=%u\n",
         issue->kind, issue->payload.reduce.inter_type,
         issue->payload.reduce.opcode, issue->payload.reduce.kind,
         issue->payload.reduce.src, issue->payload.reduce.dst,
         issue->payload.reduce.dim, issue->payload.reduce.n,
         issue->payload.reduce.h, issue->payload.reduce.w,
         issue->payload.reduce.c, issue->payload.reduce.format,
         issue->payload.reduce.wait_policy);

  printf("convert status=%d\n", wafer_convert(5, 4, 66560, 66816, 8));
  issue = wafer_cabi_get_last_issue();
  printf("convert kind=%u inter=%u opcode=%u src_fmt=%u dst_fmt=%u src=%u "
         "dst=%u elements=%u wait=%u\n",
         issue->kind, issue->payload.convert.inter_type,
         issue->payload.convert.opcode, issue->payload.convert.src_format,
         issue->payload.convert.dst_format, issue->payload.convert.src,
         issue->payload.convert.dst, issue->payload.convert.elem_count,
         issue->payload.convert.wait_policy);

  printf("dte_send status=%d\n", wafer_dte_send(66560, 1, 32));
  issue = wafer_cabi_get_last_issue();
  printf("dte_send kind=%u op=%s buffer=%u peer=%u bytes=%u wait=%u\n",
         issue->kind, issue->payload.dte.is_send ? "send" : "recv",
         issue->payload.dte.buffer, issue->payload.dte.peer,
         issue->payload.dte.bytes, issue->payload.dte.wait_policy);

  printf("dte_recv status=%d\n", wafer_dte_recv(66304, 1, 32));
  issue = wafer_cabi_get_last_issue();
  printf("dte_recv kind=%u op=%s buffer=%u peer=%u bytes=%u wait=%u\n",
         issue->kind, issue->payload.dte.is_send ? "send" : "recv",
         issue->payload.dte.buffer, issue->payload.dte.peer,
         issue->payload.dte.bytes, issue->payload.dte.wait_policy);

  printf("dte_wait status=%d\n", wafer_dte_wait(2));
  issue = wafer_cabi_get_last_issue();
  printf("dte_wait kind=%u tokens=%u wait=%u\n", issue->kind,
         issue->payload.dte_wait.token_count,
         issue->payload.dte_wait.wait_policy);

  printf("local_fence status=%d\n", wafer_local_fence());
  issue = wafer_cabi_get_last_issue();
  printf("local_fence kind=%u wait=%u local_wait=%u\n", issue->kind,
         issue->payload.local_fence.wait_policy,
         issue->payload.local_fence.calls_local_wait);

  return 0;
}
