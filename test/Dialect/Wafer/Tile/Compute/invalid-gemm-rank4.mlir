// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  func.func @rank4_batched_gemm(
      %lhs: memref<2x3x5x8xf32, #wafer.memory<spm, ncx>>,
      %rhs: memref<2x3x7x8xf32, #wafer.memory<spm, ncx>>) {
    %result = wafer.tile.gemm %lhs, %rhs
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
        : (memref<2x3x5x8xf32, #wafer.memory<spm, ncx>>,
           memref<2x3x7x8xf32, #wafer.memory<spm, ncx>>)
       -> memref<2x3x5x7xf32, #wafer.memory<spm, ncx>>
    return
  }
}

// CHECK: error: 'wafer.tile.gemm' op GEMM batched form expects operands and result to have rank exactly 3
