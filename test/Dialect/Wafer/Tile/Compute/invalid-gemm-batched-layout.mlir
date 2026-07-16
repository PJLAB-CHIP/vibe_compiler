// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  func.func @batched_gemm_with_plain_layout(
      %lhs: memref<2x5x8xf32, #wafer.memory<spm, cx>>,
      %rhs: memref<2x8x7xf32, #wafer.memory<spm, cx>>) {
    %result = wafer.tile.gemm %lhs, %rhs
        {batch_count = 2 : i64,
         lhs_batch_dims = array<i64: 0>,
         lhs_contracting_dim = 2 : i64,
         lhs_m_dim = 1 : i64,
         result_batch_dims = array<i64: 0>,
         result_m_dim = 1 : i64,
         result_n_dim = 2 : i64,
         rhs_batch_dims = array<i64: 0>,
         rhs_contracting_dim = 1 : i64,
         rhs_n_dim = 2 : i64}
        : (memref<2x5x8xf32, #wafer.memory<spm, cx>>,
           memref<2x8x7xf32, #wafer.memory<spm, cx>>)
       -> memref<2x5x7xf32, #wafer.memory<spm, cx>>
    return
  }
}

// CHECK: error: 'wafer.tile.gemm' op batched gemm storage values must use ncx layout
