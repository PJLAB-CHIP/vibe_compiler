// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x4x4x64xf16, #wafer.memory<spm, ncx>>
  %dest = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x8x8x64xf16, #wafer.memory<spm, ncx>>

  wafer.instr.unpool #wafer.instr_unpool_kind<mask> %input into %dest
      {source_shape = array<i64: 1, 4, 4, 64>,
       dest_shape = array<i64: 1, 8, 8, 64>,
       kernel_strides = array<i64: 2, 2, 2, 2>}
      : memref<1x4x4x64xf16, #wafer.memory<spm, ncx>>
    into memref<1x8x8x64xf16, #wafer.memory<spm, ncx>>
}

// CHECK: error: 'wafer.instr.unpool' op unpool kind requires scalar index attr
