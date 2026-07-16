// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  func.func @batched_gemm_with_permuted_dimensions(
      %lhs: memref<5x2x8xf32, #wafer.memory<spm, ncx>>,
      %rhs: memref<7x2x8xf32, #wafer.memory<spm, ncx>>) {
    %result = wafer.tile.gemm %lhs, %rhs
        {batch_count = 2 : i64,
         lhs_batch_dims = array<i64: 1>,
         lhs_contracting_dim = 2 : i64,
         lhs_m_dim = 0 : i64,
         result_batch_dims = array<i64: 1>,
         result_m_dim = 0 : i64,
         result_n_dim = 2 : i64,
         rhs_batch_dims = array<i64: 1>,
         rhs_contracting_dim = 2 : i64,
         rhs_n_dim = 0 : i64}
        : (memref<5x2x8xf32, #wafer.memory<spm, ncx>>,
           memref<7x2x8xf32, #wafer.memory<spm, ncx>>)
       -> memref<5x2x7xf32, #wafer.memory<spm, ncx>>
    return
  }
}

// CHECK: error: 'wafer.tile.gemm' op GEMM batched form requires canonical [B,M,K] x [B,K,N] -> [B,M,N] dimension attrs
