// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %lhs = "builtin.unrealized_conversion_cast"()
      : () -> memref<2x3x5x8xf32, #wafer.memory<spm, ncx>>
  %rhs = "builtin.unrealized_conversion_cast"()
      : () -> memref<2x3x7x8xf32, #wafer.memory<spm, ncx>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<2x3x5x7xf32, #wafer.memory<spm, ncx>>
  wafer.instr.gemm %lhs, %rhs into %dst
      {m = 5 : i64, k = 8 : i64, n = 7 : i64,
       batch_count = 6 : i64,
       lhs_batch_dims = array<i64: 0, 1>,
       lhs_contracting_dim = 3 : i64,
       lhs_m_dim = 2 : i64,
       result_batch_dims = array<i64: 0, 1>,
       result_m_dim = 2 : i64,
       result_n_dim = 3 : i64,
       rhs_batch_dims = array<i64: 0, 1>,
       rhs_contracting_dim = 3 : i64,
       rhs_n_dim = 2 : i64}
      : memref<2x3x5x8xf32, #wafer.memory<spm, ncx>>,
        memref<2x3x7x8xf32, #wafer.memory<spm, ncx>>
    into memref<2x3x5x7xf32, #wafer.memory<spm, ncx>>
}

// CHECK: error: 'wafer.instr.gemm' op GEMM batched form expects operands and result to have rank exactly 3
