// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x8x8x64xf16, #wafer.memory<spm, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x8x8x64xf16, #wafer.memory<spm, tensor>>

  wafer.instr.tdma_data_move #wafer.instr_data_move_kind<img2col> %src into %dst
      {source_shape = array<i64: 1, 8, 8, 64>,
       dest_shape = array<i64: 1, 8, 8, 64>,
       kernel_strides = array<i64: 3, 3, 1, 1>}
      : memref<1x8x8x64xf16, #wafer.memory<spm, tensor>>
     to memref<1x8x8x64xf16, #wafer.memory<spm, tensor>>
}

// CHECK: error: 'wafer.instr.tdma_data_move' op img2col data_move requires pads attr
