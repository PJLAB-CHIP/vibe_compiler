// RUN: wafer-opt %s | FileCheck %s

module {
  %wide = "builtin.unrealized_conversion_cast"()
      : () -> memref<8xf32, #wafer.memory<spm, tensor>>
  %patch = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf32, #wafer.memory<spm, tensor>>
  %slice = wafer.tile.extract_slice %wide
      {offsets = array<i64: 2>, sizes = array<i64: 4>, strides = array<i64: 1>}
      : memref<8xf32, #wafer.memory<spm, tensor>>
     -> memref<4xf32, #wafer.memory<spm, tensor>>
  %inserted = wafer.tile.insert_slice %slice into %wide
      {offsets = array<i64: 2>, sizes = array<i64: 4>, strides = array<i64: 1>}
      : memref<4xf32, #wafer.memory<spm, tensor>>
       into memref<8xf32, #wafer.memory<spm, tensor>>
     -> memref<8xf32, #wafer.memory<spm, tensor>>
  %matrix = "builtin.unrealized_conversion_cast"()
      : () -> memref<2x4xf32, #wafer.memory<spm, tensor>>
  %row = wafer.tile.extract_slice %matrix
      {offsets = array<i64: 0, 0>, sizes = array<i64: 1, 4>, strides = array<i64: 1, 1>}
      : memref<2x4xf32, #wafer.memory<spm, tensor>>
     -> memref<4xf32, #wafer.memory<spm, tensor>>
  %row_inserted = wafer.tile.insert_slice %row into %matrix
      {offsets = array<i64: 1, 0>, sizes = array<i64: 1, 4>, strides = array<i64: 1, 1>}
      : memref<4xf32, #wafer.memory<spm, tensor>>
       into memref<2x4xf32, #wafer.memory<spm, tensor>>
     -> memref<2x4xf32, #wafer.memory<spm, tensor>>
  %copy = wafer.tile.copy %patch
      : memref<4xf32, #wafer.memory<spm, tensor>>
     -> memref<4xf32, #wafer.memory<spm, tensor>>
  %transposed = "builtin.unrealized_conversion_cast"()
      : () -> memref<3x2xf32, #wafer.memory<spm, tensor>>
  %transpose = wafer.tile.transpose %transposed
      {permutation = array<i64: 1, 0>}
      : memref<3x2xf32, #wafer.memory<spm, tensor>>
     -> memref<2x3xf32, #wafer.memory<spm, tensor>>
  %broadcast = wafer.tile.broadcast %patch
      {dimensions = array<i64: 1>}
      : memref<4xf32, #wafer.memory<spm, tensor>>
     -> memref<2x4xf32, #wafer.memory<spm, tensor>>
}

// CHECK: wafer.tile.extract_slice
// CHECK: wafer.tile.insert_slice
// CHECK: wafer.tile.extract_slice
// CHECK: wafer.tile.insert_slice
// CHECK: wafer.tile.copy
// CHECK: wafer.tile.transpose
// CHECK: wafer.tile.broadcast
