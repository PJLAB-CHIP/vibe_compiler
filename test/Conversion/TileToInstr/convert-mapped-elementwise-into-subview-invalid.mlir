// RUN: not wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %s 2>&1 | FileCheck %s

// A uniform RHS still needs a proved physical traversal. A compact scratch
// cannot make CT skip the six untouched parent elements between logical rows.
// CHECK: failed to legalize operation 'wafer.tile.elementwise_into'
func.func @gapped_destination() {
  wafer.tile.region() -> () {
    %base = memref.alloc() : memref<1x2x4x1031xf16, #wafer.memory<spm, tensor>>
    %rhs = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %two = arith.constant 2.0 : f16
    wafer.tile.fill %rhs, %two : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    %dest = memref.subview %base[0, 1, 0, 0] [1, 1, 4, 1025] [1, 1, 1, 1]
        : memref<1x2x4x1031xf16, #wafer.memory<spm, tensor>>
       to memref<1x1x4x1025xf16, strided<[8248, 4124, 1031, 1], offset: 4124>, #wafer.memory<spm, tensor>>
    wafer.tile.elementwise_into <add> %dest, %rhs into %dest
        {indexing_maps = [affine_map<(a,b,c,d)->(a,b,c,d)>, affine_map<(a,b,c,d)->(c)>, affine_map<(a,b,c,d)->(a,b,c,d)>]}
        : memref<1x1x4x1025xf16, strided<[8248, 4124, 1031, 1], offset: 4124>, #wafer.memory<spm, tensor>>,
          memref<4xf16, #wafer.memory<spm, tensor>>
       into memref<1x1x4x1025xf16, strided<[8248, 4124, 1031, 1], offset: 4124>, #wafer.memory<spm, tensor>>
    wafer.tile.yield
  }
  return
}
