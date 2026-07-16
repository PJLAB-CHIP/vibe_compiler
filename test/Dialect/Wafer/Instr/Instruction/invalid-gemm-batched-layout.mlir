// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %lhs = "builtin.unrealized_conversion_cast"()
      : () -> memref<2x5x8xf32, #wafer.memory<spm, cx>>
  %rhs = "builtin.unrealized_conversion_cast"()
      : () -> memref<2x8x7xf32, #wafer.memory<spm, cx>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<2x5x7xf32, #wafer.memory<spm, cx>>
  wafer.instr.gemm %lhs, %rhs into %dst
      {m = 5 : i64, k = 8 : i64, n = 7 : i64,
       batch_count = 2 : i64,
       lhs_batch_dims = array<i64: 0>,
       lhs_contracting_dim = 2 : i64,
       lhs_m_dim = 1 : i64,
       result_batch_dims = array<i64: 0>,
       result_m_dim = 1 : i64,
       result_n_dim = 2 : i64,
       rhs_batch_dims = array<i64: 0>,
       rhs_contracting_dim = 1 : i64,
       rhs_n_dim = 2 : i64}
      : memref<2x5x8xf32, #wafer.memory<spm, cx>>,
        memref<2x8x7xf32, #wafer.memory<spm, cx>>
    into memref<2x5x7xf32, #wafer.memory<spm, cx>>
}

// CHECK: error: 'wafer.instr.gemm' op lhs rank > 2 must use ncx layout
