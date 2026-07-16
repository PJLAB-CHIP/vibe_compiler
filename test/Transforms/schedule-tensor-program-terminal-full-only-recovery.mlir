// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-schedule-tensor-program{logical-rank=0 tile-search=min-estimated-time tile-search-effort=quick max-search-candidates=2 print-candidate-summary=true})' %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=SUMMARY
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-schedule-tensor-program{logical-rank=0 tile-search=min-estimated-time tile-search-effort=quick max-search-candidates=2})' %s | FileCheck %s --check-prefix=IR --implicit-check-not=linalg. --implicit-check-not=wafer.linalg_ext

// The aggressive dataflow scope ends at all_reduce and therefore supports
// only its full traversal. The full fused traversal includes a 256x4096 f16
// weight and exceeds the SPM window. The one bounded root-capability recovery
// policy cuts only the terminal collective: the matmul may then refine N once,
// while the collective remains a full-traversal singleton.
func.func @terminal_collective_recovery(
    %lhs: tensor<16x256xf16>,
    %rhs: tensor<256x4096xf16>) -> tensor<16x4096xf16> {
  %zero = arith.constant 0.0 : f16
  %matmul_empty = tensor.empty() : tensor<16x4096xf16>
  %matmul_init = linalg.fill ins(%zero : f16)
      outs(%matmul_empty : tensor<16x4096xf16>) -> tensor<16x4096xf16>
  %matmul = linalg.matmul
      ins(%lhs, %rhs : tensor<16x256xf16>, tensor<256x4096xf16>)
      outs(%matmul_init : tensor<16x4096xf16>) -> tensor<16x4096xf16>
  %collective_out = tensor.empty() : tensor<16x4096xf16>
  %collective = wafer.linalg_ext.collective.all_reduce
      ins(%matmul : tensor<16x4096xf16>)
      outs(%collective_out : tensor<16x4096xf16>) {
  ^bb0(%lhs_value: f16, %rhs_value: f16):
    %sum = arith.addf %lhs_value, %rhs_value : f16
    wafer.linalg_ext.collective.yield %sum : f16
  } {channel_id = 41 : i64, rank_group = array<i64: 0, 1>}
      -> tensor<16x4096xf16>
  return %collective : tensor<16x4096xf16>
}

// A collective-only task must not manufacture smaller traversal candidates:
// only the full candidate can use the complete-traversal fallback.
func.func @singleton_collective(
    %input: tensor<16x4096xf16>) -> tensor<16x4096xf16> {
  %out = tensor.empty() : tensor<16x4096xf16>
  %collective = wafer.linalg_ext.collective.all_reduce
      ins(%input : tensor<16x4096xf16>)
      outs(%out : tensor<16x4096xf16>) {
  ^bb0(%lhs_value: f16, %rhs_value: f16):
    %sum = arith.addf %lhs_value, %rhs_value : f16
    wafer.linalg_ext.collective.yield %sum : f16
  } {channel_id = 42 : i64, rank_group = array<i64: 0, 1>}
      -> tensor<16x4096xf16>
  return %collective : tensor<16x4096xf16>
}

// SUMMARY: wafer.schedule_tensor_program selected task @terminal_collective_recovery#0
// SUMMARY-SAME: tile=[16,2048]
// SUMMARY-SAME: candidates=2
// SUMMARY-SAME: rejected=1
// SUMMARY: wafer.schedule_tensor_program selected task @terminal_collective_recovery#1
// SUMMARY-SAME: tile=[16,4096]
// SUMMARY-SAME: candidates=1
// SUMMARY-SAME: rejected=0
// SUMMARY: wafer.schedule_tensor_program selected task @singleton_collective#0
// SUMMARY-SAME: tile=[16,4096]
// SUMMARY-SAME: candidates=1
// SUMMARY-SAME: rejected=0

// IR-LABEL: func.func @terminal_collective_recovery
// IR-COUNT-2: wafer.tile.region
// IR: return
// IR-LABEL: func.func @singleton_collective
// IR-COUNT-1: wafer.tile.region
// IR: return
