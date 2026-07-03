// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x8x8x64xf16, #wafer.memory<spm, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x8x64x8xf16, #wafer.memory<spm, tensor>>

  wafer.instr.tdma_data_move #wafer.instr_data_move_kind<transpose> %src into %dst
      {source_shape = array<i64: 1, 8, 8, 64>,
       dest_shape = array<i64: 1, 8, 64, 8>,
       permutation = array<i64: 0, 1, 3, 2>}
      : memref<1x8x8x64xf16, #wafer.memory<spm, tensor>>
     to memref<1x8x64x8xf16, #wafer.memory<spm, tensor>>
}

// CHECK: error: 'wafer.instr.tdma_data_move' op transpose-like data_move kind is not V0 production legal; use gather_scatter lowering
