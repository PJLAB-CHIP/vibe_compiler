// RUN: wafer-opt %s | FileCheck %s

module {
  func.func @batched_attention_score(
      %query: memref<2x3x5x8xf32, #wafer.memory<spm, cx>>,
      %key: memref<2x3x7x8xf32, #wafer.memory<spm, cx>>) {
    %score = wafer.tile.gemm %query, %key
        {batch_count = 6 : i64,
         lhs_batch_dims = array<i64: 0, 1>,
         lhs_contracting_dim = 3 : i64,
         lhs_m_dim = 2 : i64,
         result_batch_dims = array<i64: 0, 1>,
         result_m_dim = 2 : i64,
         result_n_dim = 3 : i64,
         rhs_batch_dims = array<i64: 0, 1>,
         rhs_contracting_dim = 3 : i64,
         rhs_n_dim = 2 : i64}
        : (memref<2x3x5x8xf32, #wafer.memory<spm, cx>>,
           memref<2x3x7x8xf32, #wafer.memory<spm, cx>>)
       -> memref<2x3x5x7xf32, #wafer.memory<spm, cx>>
    return
  }
}

// CHECK-LABEL: func.func @batched_attention_score
// CHECK: wafer.tile.gemm
// CHECK-SAME: batch_count = 6 : i64
// CHECK-SAME: lhs_contracting_dim = 3 : i64
// CHECK-SAME: rhs_contracting_dim = 3 : i64
