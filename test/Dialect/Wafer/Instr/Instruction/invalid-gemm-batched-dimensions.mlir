// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %lhs = "builtin.unrealized_conversion_cast"()
      : () -> memref<5x2x8xf32, #wafer.memory<spm, ncx>>
  %rhs = "builtin.unrealized_conversion_cast"()
      : () -> memref<7x2x8xf32, #wafer.memory<spm, ncx>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<5x2x7xf32, #wafer.memory<spm, ncx>>
  wafer.instr.gemm %lhs, %rhs into %dst
      {m = 5 : i64, k = 8 : i64, n = 7 : i64,
       batch_count = 2 : i64,
       lhs_batch_dims = array<i64: 1>,
       lhs_contracting_dim = 2 : i64,
       lhs_m_dim = 0 : i64,
       result_batch_dims = array<i64: 1>,
       result_m_dim = 0 : i64,
       result_n_dim = 2 : i64,
       rhs_batch_dims = array<i64: 1>,
       rhs_contracting_dim = 2 : i64,
       rhs_n_dim = 0 : i64}
      : memref<5x2x8xf32, #wafer.memory<spm, ncx>>,
        memref<7x2x8xf32, #wafer.memory<spm, ncx>>
    into memref<5x2x7xf32, #wafer.memory<spm, ncx>>
}

// CHECK: error: 'wafer.instr.gemm' op GEMM batched form dimension attrs must match the explicit stored operand orientations and canonical [B,M,N] result
