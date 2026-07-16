// RUN: wafer-opt %s | FileCheck %s

module {
  func.func @batched_attention_score(
      %query: memref<6x5x8xf32, #wafer.memory<spm, ncx>>,
      %key: memref<6x8x7xf32, #wafer.memory<spm, ncx>>) {
    %score = wafer.tile.gemm %query, %key
        {batch_count = 6 : i64,
         lhs_batch_dims = array<i64: 0>,
         lhs_contracting_dim = 2 : i64,
         lhs_m_dim = 1 : i64,
         result_batch_dims = array<i64: 0>,
         result_m_dim = 1 : i64,
         result_n_dim = 2 : i64,
         rhs_batch_dims = array<i64: 0>,
         rhs_contracting_dim = 1 : i64,
         rhs_n_dim = 2 : i64}
        : (memref<6x5x8xf32, #wafer.memory<spm, ncx>>,
           memref<6x8x7xf32, #wafer.memory<spm, ncx>>)
       -> memref<6x5x7xf32, #wafer.memory<spm, ncx>>
    return
  }
}

// CHECK-LABEL: func.func @batched_attention_score
// CHECK: wafer.tile.gemm
// CHECK-SAME: batch_count = 6 : i64
// CHECK-SAME: lhs_contracting_dim = 2 : i64
// CHECK-SAME: rhs_contracting_dim = 1 : i64
