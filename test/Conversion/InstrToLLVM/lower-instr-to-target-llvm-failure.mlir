// RUN: not wafer-opt --wafer-lower-instr-to-target-llvm %s 2>&1 | FileCheck %s

func.func @reject_transform_like_tdma() {
  %src = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
  %dst = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
      : memref<1x1x3x2xf16, #wafer.memory<spm, tensor>>
  wafer.instr.tdma_data_move #wafer.instr_data_move_kind<transpose> %src into %dst
      {source_shape = array<i64: 1, 1, 2, 3>,
       dest_shape = array<i64: 1, 1, 3, 2>,
       permutation = array<i64: 0, 1, 3, 2>}
      : memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
     to memref<1x1x3x2xf16, #wafer.memory<spm, tensor>>
  return
}

// CHECK: unsupported_target_instr: transform-like tdma_data_move kind reached target LLVM lowering
