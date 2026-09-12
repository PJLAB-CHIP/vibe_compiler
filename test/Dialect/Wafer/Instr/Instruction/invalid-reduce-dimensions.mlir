// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1x4x8xf16, #wafer.memory<spm, ncx>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1x4x1xf16, #wafer.memory<spm, ncx>>
  wafer.instr.reduce #wafer.instr_reduce_kind<sum> %input into %dst
      {dim = 6 : i64}
      : memref<1x1x4x8xf16, #wafer.memory<spm, ncx>>
    into memref<1x1x4x1xf16, #wafer.memory<spm, ncx>>
}

// CHECK: error: 'wafer.instr.reduce' op reduce dim must be a target code in [0, 5]
