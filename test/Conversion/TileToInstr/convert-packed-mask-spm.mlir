// RUN: split-file %s %t
// RUN: wafer-opt --wafer-lower-tile-region-to-instr %t/full.mlir | FileCheck %s --check-prefix=FULL
// RUN: not wafer-opt --wafer-lower-tile-region-to-instr %t/partial.mlir 2>&1 | FileCheck %s --check-prefix=PARTIAL
// RUN: not wafer-opt --wafer-lower-tile-region-to-instr %t/unaligned.mlir 2>&1 | FileCheck %s --check-prefix=UNALIGNED

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
// PARTIAL: failed to legalize operation 'memref.copy'

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
// UNALIGNED: failed to legalize operation 'wafer.tile.copy_into'

