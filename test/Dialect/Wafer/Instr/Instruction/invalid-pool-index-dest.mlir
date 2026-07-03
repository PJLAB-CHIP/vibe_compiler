// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x8x8x64xf16, #wafer.memory<spm, ncx>>
  %out = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x4x4x64xf16, #wafer.memory<spm, ncx>>

  wafer.instr.pool #wafer.instr_pool_kind<indexedmax> %input into %out
      {source_shape = array<i64: 1, 8, 8, 64>,
       dest_shape = array<i64: 1, 4, 4, 64>,
       pads = array<i64: 0, 0, 0, 0>,
       kernel_strides = array<i64: 2, 2, 2, 2>}
      : memref<1x8x8x64xf16, #wafer.memory<spm, ncx>>
    into memref<1x4x4x64xf16, #wafer.memory<spm, ncx>>
}

// CHECK: pool kind expects 2 dest operand
