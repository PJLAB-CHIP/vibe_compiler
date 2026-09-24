// RUN: split-file %s %t
// RUN: wafer-opt --wafer-lower-tile-region-to-instr %t/full.mlir | FileCheck %s --check-prefix=FULL
// RUN: wafer-opt --wafer-lower-tile-region-to-instr %t/partial.mlir | FileCheck %s --check-prefix=PARTIAL
// RUN: wafer-opt --wafer-lower-tile-region-to-instr %t/unaligned.mlir | FileCheck %s --check-prefix=UNALIGNED
// RUN: wafer-opt --wafer-lower-tile-region-to-instr --wafer-lower-instr-to-target-llvm %t/dynamic.mlir | FileCheck %s --check-prefix=DYNAMIC
// RUN: wafer-opt --wafer-lower-tile-region-to-instr --wafer-lower-instr-to-target-llvm %t/strided.mlir | FileCheck %s --check-prefix=STRIDED

//--- full.mlir
func.func @full_1024() {
  %token = arith.constant false
  %r = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%done: i1):
    %src = memref.alloc() : memref<1x1x1024xi1, #wafer.memory<spm, tensor>>
    %dst = memref.alloc() : memref<1x1x1024xi1, #wafer.memory<spm, tensor>>
    %init = arith.constant true
    wafer.tile.fill %dst, %init : memref<1x1x1024xi1, #wafer.memory<spm, tensor>>, i1
    memref.copy %src, %dst : memref<1x1x1024xi1, #wafer.memory<spm, tensor>> to memref<1x1x1024xi1, #wafer.memory<spm, tensor>>
    wafer.tile.copy_into %src into %dst : memref<1x1x1024xi1, #wafer.memory<spm, tensor>> into memref<1x1x1024xi1, #wafer.memory<spm, tensor>>
    wafer.tile.yield %done : i1
  }
  return
}
// FULL-LABEL: func.func @full_1024
// FULL: wafer.instr.fill
// FULL-SAME: physical_footprint
// FULL: wafer.instr.gather_scatter
// FULL-SAME: byte_count = 128 : i64
// FULL-SAME: dst_iterations = array<i64: 1, 1, 1>
// FULL-SAME: dst_strides = array<i64: 0, 0, 0>
// FULL-SAME: inner_bytes = 128 : i64
// FULL-SAME: src_iterations = array<i64: 1, 1, 1>
// FULL-SAME: src_strides = array<i64: 0, 0, 0>
// FULL: wafer.instr.gather_scatter
// FULL-SAME: byte_count = 128 : i64
// FULL-NOT: memref.copy
// FULL-NOT: wafer.tile.copy_into

func.func @full_1025() {
  %token = arith.constant false
  %r = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%done: i1):
    %src = memref.alloc() : memref<1x1x1025xi1, #wafer.memory<spm, tensor>>
    %dst = memref.alloc() : memref<1x1x1025xi1, #wafer.memory<spm, tensor>>
    %init = arith.constant true
    wafer.tile.fill %dst, %init : memref<1x1x1025xi1, #wafer.memory<spm, tensor>>, i1
    memref.copy %src, %dst : memref<1x1x1025xi1, #wafer.memory<spm, tensor>> to memref<1x1x1025xi1, #wafer.memory<spm, tensor>>
    wafer.tile.copy_into %src into %dst : memref<1x1x1025xi1, #wafer.memory<spm, tensor>> into memref<1x1x1025xi1, #wafer.memory<spm, tensor>>
    wafer.tile.yield %done : i1
  }
  return
}
// FULL-LABEL: func.func @full_1025
// FULL: wafer.instr.fill
// FULL-SAME: physical_footprint
// FULL: wafer.instr.gather_scatter
// FULL-SAME: byte_count = 129 : i64
// FULL-SAME: dst_iterations = array<i64: 1, 1, 1>
// FULL-SAME: dst_strides = array<i64: 0, 0, 0>
// FULL-SAME: inner_bytes = 129 : i64
// FULL-SAME: src_iterations = array<i64: 1, 1, 1>
// FULL-SAME: src_strides = array<i64: 0, 0, 0>
// FULL: wafer.instr.gather_scatter
// FULL-SAME: byte_count = 129 : i64
// FULL-NOT: memref.copy
// FULL-NOT: wafer.tile.copy_into

func.func @full_1031() {
  %token = arith.constant false
  %r = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%done: i1):
    %src = memref.alloc() : memref<1x1x1031xi1, #wafer.memory<spm, tensor>>
    %dst = memref.alloc() : memref<1x1x1031xi1, #wafer.memory<spm, tensor>>
    %init = arith.constant true
    wafer.tile.fill %dst, %init : memref<1x1x1031xi1, #wafer.memory<spm, tensor>>, i1
    memref.copy %src, %dst : memref<1x1x1031xi1, #wafer.memory<spm, tensor>> to memref<1x1x1031xi1, #wafer.memory<spm, tensor>>
    wafer.tile.copy_into %src into %dst : memref<1x1x1031xi1, #wafer.memory<spm, tensor>> into memref<1x1x1031xi1, #wafer.memory<spm, tensor>>
    wafer.tile.yield %done : i1
  }
  return
}
// FULL-LABEL: func.func @full_1031
// FULL: wafer.instr.fill
// FULL-SAME: physical_footprint
// FULL: wafer.instr.gather_scatter
// FULL-SAME: byte_count = 129 : i64
// FULL-SAME: dst_iterations = array<i64: 1, 1, 1>
// FULL-SAME: dst_strides = array<i64: 0, 0, 0>
// FULL-SAME: inner_bytes = 129 : i64
// FULL-SAME: src_iterations = array<i64: 1, 1, 1>
// FULL-SAME: src_strides = array<i64: 0, 0, 0>
// FULL: wafer.instr.gather_scatter
// FULL-SAME: byte_count = 129 : i64
// FULL-NOT: memref.copy
// FULL-NOT: wafer.tile.copy_into

//--- partial.mlir
// The view must not overwrite live neighboring bits.
func.func @partial() {
  %token = arith.constant false
  %r = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%done: i1):
    %src = memref.alloc() : memref<1x1x1031xi1, #wafer.memory<spm, tensor>>
    %storage = memref.alloc() : memref<1x1x1040xi1, #wafer.memory<spm, tensor>>
    %view = memref.subview %storage[0, 0, 0] [1, 1, 1031] [1, 1, 1]
      : memref<1x1x1040xi1, #wafer.memory<spm, tensor>> to memref<1x1x1031xi1, strided<[1040, 1040, 1]>, #wafer.memory<spm, tensor>>
    memref.copy %src, %view : memref<1x1x1031xi1, #wafer.memory<spm, tensor>> to memref<1x1x1031xi1, strided<[1040, 1040, 1]>, #wafer.memory<spm, tensor>>
    wafer.tile.yield %done : i1
  }
  return
}
// PARTIAL: wafer.instr.gather_scatter
// PARTIAL-SAME: byte_count = 129 : i64
// PARTIAL-COUNT-2: wafer.instr.bit2fp
// PARTIAL: wafer.instr.gather_scatter
// PARTIAL-SAME: byte_count = 2062 : i64
// PARTIAL: wafer.instr.elementwise <ne>
// PARTIAL: wafer.instr.gather_scatter
// PARTIAL-SAME: byte_count = 129 : i64
// PARTIAL-NOT: memref.copy

//--- unaligned.mlir
// The view must not overwrite live neighboring bits.
func.func @unaligned() {
  %token = arith.constant false
  %r = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%done: i1):
    %src = memref.alloc() : memref<1x1x1024xi1, #wafer.memory<spm, tensor>>
    %storage = memref.alloc() : memref<1x1x1040xi1, #wafer.memory<spm, tensor>>
    %view = memref.subview %storage[0, 0, 1] [1, 1, 1024] [1, 1, 1]
      : memref<1x1x1040xi1, #wafer.memory<spm, tensor>> to memref<1x1x1024xi1, strided<[1040, 1040, 1], offset: 1>, #wafer.memory<spm, tensor>>
    wafer.tile.copy_into %src into %view : memref<1x1x1024xi1, #wafer.memory<spm, tensor>> into memref<1x1x1024xi1, strided<[1040, 1040, 1], offset: 1>, #wafer.memory<spm, tensor>>
    wafer.tile.yield %done : i1
  }
  return
}
// UNALIGNED: wafer.instr.gather_scatter
// UNALIGNED-SAME: byte_count = 129 : i64
// UNALIGNED-COUNT-2: wafer.instr.bit2fp
// UNALIGNED: wafer.instr.gather_scatter
// UNALIGNED-SAME: byte_count = 2048 : i64
// UNALIGNED-SAME: dst_offset = 2 : i64
// UNALIGNED: wafer.instr.elementwise <ne>
// UNALIGNED: wafer.instr.gather_scatter
// UNALIGNED-SAME: byte_count = 129 : i64
// UNALIGNED-NOT: wafer.tile.copy_into

//--- dynamic.mlir
func.func @dynamic() {
  wafer.tile.region() -> () {
    %src = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<1x1x1024xi1, #wafer.memory<spm, tensor>>
    %dst = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<131072>}
      : memref<1x1x1040xi1, #wafer.memory<spm, tensor>>
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c17 = arith.constant 17 : index
    scf.for %i = %c0 to %c17 step %c8 {
      %view = memref.subview %dst[0, 0, %i] [1, 1, 1024] [1, 1, 1]
        : memref<1x1x1040xi1, #wafer.memory<spm, tensor>>
        to memref<1x1x1024xi1, strided<[1040, 1040, 1], offset: ?>, #wafer.memory<spm, tensor>>
      memref.copy %src, %view : memref<1x1x1024xi1, #wafer.memory<spm, tensor>>
        to memref<1x1x1024xi1, strided<[1040, 1040, 1], offset: ?>, #wafer.memory<spm, tensor>>
      wafer.tile.copy_into %view into %src
        : memref<1x1x1024xi1, strided<[1040, 1040, 1], offset: ?>, #wafer.memory<spm, tensor>>
        into memref<1x1x1024xi1, #wafer.memory<spm, tensor>>
    }
    wafer.tile.yield
  }
  return
}
// DYNAMIC-LABEL: llvm.func @dynamic
// DYNAMIC: llvm.udiv
// DYNAMIC: llvm.call @wafer_tx81_gather_scatter
// DYNAMIC: llvm.call @wafer_tx81_gather_scatter

//--- strided.mlir
func.func @strided() {
  wafer.tile.region() -> () {
    %src = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<1x1025x512xi1, #wafer.memory<spm, tensor>>
    %dst = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<262144>}
      : memref<1x1025x1024xi1, #wafer.memory<spm, tensor>>
    %view = memref.subview %dst[0, 0, 512] [1, 1025, 512] [1, 1, 1]
      : memref<1x1025x1024xi1, #wafer.memory<spm, tensor>>
      to memref<1x1025x512xi1, strided<[1049600, 1024, 1], offset: 512>, #wafer.memory<spm, tensor>>
    memref.copy %src, %view : memref<1x1025x512xi1, #wafer.memory<spm, tensor>>
      to memref<1x1025x512xi1, strided<[1049600, 1024, 1], offset: 512>, #wafer.memory<spm, tensor>>
    wafer.tile.copy_into %view into %src
      : memref<1x1025x512xi1, strided<[1049600, 1024, 1], offset: 512>, #wafer.memory<spm, tensor>>
      into memref<1x1025x512xi1, #wafer.memory<spm, tensor>>
    wafer.tile.yield
  }
  return
}
// STRIDED-LABEL: llvm.func @strided
// STRIDED: llvm.mlir.constant(65600 : i32)
// STRIDED: llvm.mlir.constant(64 : i32)
// STRIDED: llvm.call @wafer_tx81_gather_scatter
// STRIDED: llvm.call @wafer_tx81_gather_scatter
