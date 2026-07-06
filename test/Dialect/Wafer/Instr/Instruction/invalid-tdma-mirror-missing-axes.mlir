// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>

  wafer.instr.tdma_data_move #wafer.instr_data_move_kind<mirror> %src into %dst
      {source_shape = array<i64: 1, 1, 2, 3>,
       dest_shape = array<i64: 1, 1, 2, 3>}
      : memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
     to memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
}

// CHECK: error: 'wafer.instr.tdma_data_move' op mirror data_move requires axes attr
