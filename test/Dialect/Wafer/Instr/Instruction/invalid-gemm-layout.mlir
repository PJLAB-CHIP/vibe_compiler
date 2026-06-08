// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %lhs = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<spm, tensor>>
  %rhs = "builtin.unrealized_conversion_cast"()
      : () -> memref<8x16xf16, #wafer.memory<spm, cx>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x16xf16, #wafer.memory<spm, cx>>
  wafer.instr.gemm %lhs, %rhs into %dst
      {m = 4 : i64, k = 8 : i64, n = 16 : i64}
      : memref<4x8xf16, #wafer.memory<spm, tensor>>,
        memref<8x16xf16, #wafer.memory<spm, cx>>
    into memref<4x16xf16, #wafer.memory<spm, cx>>
}

// CHECK: error: 'wafer.instr.gemm' op lhs, rhs and dest must use aligned SPM layouts
