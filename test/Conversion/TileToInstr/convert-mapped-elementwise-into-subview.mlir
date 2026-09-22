// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %s | FileCheck %s

// Independent operand scratch must not inherit the destination view offset.
// Keep the original destination address for CT and the exact mapped input for GS.

// CHECK-LABEL: func.func @permutation_1024
// CHECK: %[[BASE1024:[^ ]+]] = memref.alloc() : memref<1x2x4x1024xbf16, #wafer.memory<spm, tensor>>
// CHECK: %[[RHS1024:[^ ]+]] = memref.alloc() : memref<1x1x1024x4xbf16, #wafer.memory<spm, tensor>>
// CHECK: %[[DEST1024:[^ ]+]] = memref.subview %[[BASE1024]][0, 1, 0, 0] [1, 1, 4, 1024]
// CHECK: %[[SCRATCH1024:[^ ]+]] = memref.alloc() : memref<1x1x4x1024xbf16, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.gather_scatter %[[RHS1024]] to %[[SCRATCH1024]] {byte_count = 8192 : i64
// CHECK-SAME: dst_strides = array<i64: 2, 2048, 0>
// CHECK-SAME: src_strides = array<i64: 8, 2, 0>
// CHECK: wafer.instr.elementwise <add> %[[DEST1024]], %[[SCRATCH1024]] into %[[DEST1024]]
// CHECK-NOT: memref.alloc
// CHECK: return
func.func @permutation_1024() {
  wafer.tile.region() -> () {
    %base = memref.alloc() : memref<1x2x4x1024xbf16, #wafer.memory<spm, tensor>>
    %rhs = memref.alloc() : memref<1x1x1024x4xbf16, #wafer.memory<spm, tensor>>
    %dest = memref.subview %base[0, 1, 0, 0] [1, 1, 4, 1024] [1, 1, 1, 1] : memref<1x2x4x1024xbf16, #wafer.memory<spm, tensor>> to memref<1x1x4x1024xbf16, strided<[8192, 4096, 1024, 1], offset: 4096>, #wafer.memory<spm, tensor>>
    wafer.tile.elementwise_into <add> %dest, %rhs into %dest
        {indexing_maps = [affine_map<(a,b,c,d)->(a,b,c,d)>, affine_map<(a,b,c,d)->(a,b,d,c)>, affine_map<(a,b,c,d)->(a,b,c,d)>]}
        : memref<1x1x4x1024xbf16, strided<[8192, 4096, 1024, 1], offset: 4096>, #wafer.memory<spm, tensor>>, memref<1x1x1024x4xbf16, #wafer.memory<spm, tensor>> into memref<1x1x4x1024xbf16, strided<[8192, 4096, 1024, 1], offset: 4096>, #wafer.memory<spm, tensor>>
    wafer.tile.yield
  }
  return
}

// CHECK-LABEL: func.func @broadcast_1025
// CHECK: %[[BASE1025:[^ ]+]] = memref.alloc() : memref<1x2x4x1025xf32, #wafer.memory<spm, tensor>>
// CHECK: %[[RHS1025:[^ ]+]] = memref.alloc() : memref<4xf32, #wafer.memory<spm, tensor>>
// CHECK: scf.for
// CHECK: %[[DEST1025:[^ ]+]] = memref.subview %[[BASE1025]]
// CHECK: %[[SCRATCH1025:[^ ]+]] = memref.alloc() : memref<1x1x4x1025xf32, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.gather_scatter %[[RHS1025]] to %[[SCRATCH1025]] {byte_count = 16400 : i64
// CHECK-SAME: dst_strides = array<i64: 4, 4100, 0>
// CHECK-SAME: src_strides = array<i64: 0, 4, 0>
// CHECK: wafer.instr.elementwise <add> %[[DEST1025]], %[[SCRATCH1025]] into %[[DEST1025]]
// CHECK-NOT: memref.alloc
// CHECK: return
func.func @broadcast_1025() {
  wafer.tile.region() -> () {
    %base = memref.alloc() : memref<1x2x4x1025xf32, #wafer.memory<spm, tensor>>
    %rhs = memref.alloc() : memref<4xf32, #wafer.memory<spm, tensor>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %i = %c0 to %c2 step %c1 {
      %dest = memref.subview %base[0, %i, 0, 0] [1, 1, 4, 1025] [1, 1, 1, 1] : memref<1x2x4x1025xf32, #wafer.memory<spm, tensor>> to memref<1x1x4x1025xf32, strided<[8200, 4100, 1025, 1], offset: ?>, #wafer.memory<spm, tensor>>
      wafer.tile.elementwise_into <add> %dest, %rhs into %dest
          {indexing_maps = [affine_map<(a,b,c,d)->(a,b,c,d)>, affine_map<(a,b,c,d)->(c)>, affine_map<(a,b,c,d)->(a,b,c,d)>]}
          : memref<1x1x4x1025xf32, strided<[8200, 4100, 1025, 1], offset: ?>, #wafer.memory<spm, tensor>>, memref<4xf32, #wafer.memory<spm, tensor>> into memref<1x1x4x1025xf32, strided<[8200, 4100, 1025, 1], offset: ?>, #wafer.memory<spm, tensor>>
    }
    wafer.tile.yield
  }
  return
}

// CHECK-LABEL: func.func @fill_1031
// CHECK: %[[BASE1031:[^ ]+]] = memref.alloc() : memref<1x2x4x1031xf16, #wafer.memory<spm, tensor>>
// CHECK: scf.for
// CHECK: %[[DEST1031:[^ ]+]] = memref.subview %[[BASE1031]]
// CHECK: %[[SCRATCH1031:[^ ]+]] = memref.alloc() : memref<1x1x4x1031xf16, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.fill %[[SCRATCH1031]],
// CHECK-NOT: wafer.instr.gather_scatter
// CHECK: wafer.instr.elementwise <add> %[[DEST1031]], %[[SCRATCH1031]] into %[[DEST1031]]
// CHECK-NOT: memref.alloc
// CHECK: return
func.func @fill_1031() {
  wafer.tile.region() -> () {
    %base = memref.alloc() : memref<1x2x4x1031xf16, #wafer.memory<spm, tensor>>
    %rhs = memref.alloc() : memref<4xf16, #wafer.memory<spm, tensor>>
    %two = arith.constant 2.0 : f16
    wafer.tile.fill %rhs, %two : memref<4xf16, #wafer.memory<spm, tensor>>, f16
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %i = %c0 to %c2 step %c1 {
      %dest = memref.subview %base[0, %i, 0, 0] [1, 1, 4, 1031] [1, 1, 1, 1] : memref<1x2x4x1031xf16, #wafer.memory<spm, tensor>> to memref<1x1x4x1031xf16, strided<[8248, 4124, 1031, 1], offset: ?>, #wafer.memory<spm, tensor>>
      wafer.tile.elementwise_into <add> %dest, %rhs into %dest
          {indexing_maps = [affine_map<(a,b,c,d)->(a,b,c,d)>, affine_map<(a,b,c,d)->(c)>, affine_map<(a,b,c,d)->(a,b,c,d)>]}
          : memref<1x1x4x1031xf16, strided<[8248, 4124, 1031, 1], offset: ?>, #wafer.memory<spm, tensor>>, memref<4xf16, #wafer.memory<spm, tensor>> into memref<1x1x4x1031xf16, strided<[8248, 4124, 1031, 1], offset: ?>, #wafer.memory<spm, tensor>>
    }
    wafer.tile.yield
  }
  return
}
