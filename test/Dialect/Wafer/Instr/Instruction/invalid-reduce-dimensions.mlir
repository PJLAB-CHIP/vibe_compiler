// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<spm, cx>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf16, #wafer.memory<spm, cx>>
  wafer.instr.reduce #wafer.instr_reduce_kind<sum> %input into %dst
      {dim = 2 : i64}
      : memref<4x8xf16, #wafer.memory<spm, cx>>
    into memref<4xf16, #wafer.memory<spm, cx>>
}

// CHECK: error: 'wafer.instr.reduce' op reduce dim is not valid for input rank
