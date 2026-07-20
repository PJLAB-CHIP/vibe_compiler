// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-schedule-tensor-program{logical-rank=0 tile-search=min-estimated-time tile-search-effort=quick max-search-candidates=4 print-candidate-summary=true})' %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=SUMMARY
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-schedule-tensor-program{logical-rank=0 tile-search=min-estimated-time tile-search-effort=quick max-search-candidates=4})' %s | FileCheck %s --check-prefix=IR

// The full fused scope exceeds SPM while each of the two complete tasks is
// legal. The selected whole-rank alternative proves that the complete static
// result/view relation covers the consumer input, then carries the producer's
// SPM SSA result across the task boundary. The shapes only make that general
// capacity boundary reproducible; they are not part of the handoff protocol.
func.func @resident_collective_edge(
    %lhs: tensor<256x128xf16>,
    %rhs: tensor<128x1024xf16>) -> tensor<256x1024xf16> {
  %zero = arith.constant 0.0 : f16
  %matmul_empty = tensor.empty() : tensor<256x1024xf16>
  %matmul_init = linalg.fill ins(%zero : f16)
      outs(%matmul_empty : tensor<256x1024xf16>) -> tensor<256x1024xf16>
  %matmul = linalg.matmul
      ins(%lhs, %rhs : tensor<256x128xf16>, tensor<128x1024xf16>)
      outs(%matmul_init : tensor<256x1024xf16>) -> tensor<256x1024xf16>
  %collective_out = tensor.empty() : tensor<256x1024xf16>
  %collective = wafer.linalg_ext.collective.all_reduce
      ins(%matmul : tensor<256x1024xf16>)
      outs(%collective_out : tensor<256x1024xf16>) {
  ^bb0(%lhs_value: f16, %rhs_value: f16):
    %sum = arith.addf %lhs_value, %rhs_value : f16
    wafer.linalg_ext.collective.yield %sum : f16
  } {channel_id = 43 : i64, rank_group = array<i64: 0, 1>}
      -> tensor<256x1024xf16>
  return %collective : tensor<256x1024xf16>
}

// SUMMARY: wafer.schedule_tensor_program selected 1 full-buffer SPM handoff(s) for rank variant
// SUMMARY: wafer.schedule_tensor_program selected task @resident_collective_edge#0
// SUMMARY-SAME: tile=[256,1024]
// SUMMARY: wafer.schedule_tensor_program selected task @resident_collective_edge#1
// SUMMARY-SAME: tile=[256,1024]

// IR-LABEL: func.func @resident_collective_edge
// IR: %[[PRODUCED:.+]] = wafer.tile.region
// IR-SAME: -> (memref<256x1024xf16, #wafer.memory<spm, tensor>>)
// IR-NOT: wafer.instr.wdma
// IR: wafer.tile.yield
// IR: %{{.+}} = wafer.tile.region(%[[PRODUCED]],
// IR-NOT: wafer.instr.rdma
// IR: wafer.instr.gather_scatter
// IR: wafer.instr.wdma
