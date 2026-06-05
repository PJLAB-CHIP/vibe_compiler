// RUN: wafer-opt %s | FileCheck %s

module {
  %wide = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.storage<tensor<8xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %patch = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.storage<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %slice = wafer.move.extract_slice %wide
      {offsets = array<i64: 2>, sizes = array<i64: 4>, strides = array<i64: 1>}
      : !wafer.storage<tensor<8xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
     -> !wafer.storage<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %inserted = wafer.move.insert_slice %slice into %wide
      {offsets = array<i64: 2>, sizes = array<i64: 4>, strides = array<i64: 1>}
      : !wafer.storage<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
       into !wafer.storage<tensor<8xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
     -> !wafer.storage<tensor<8xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %matrix = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.storage<tensor<2x4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %row = wafer.move.extract_slice %matrix
      {offsets = array<i64: 0, 0>, sizes = array<i64: 1, 4>, strides = array<i64: 1, 1>}
      : !wafer.storage<tensor<2x4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
     -> !wafer.storage<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %row_inserted = wafer.move.insert_slice %row into %matrix
      {offsets = array<i64: 1, 0>, sizes = array<i64: 1, 4>, strides = array<i64: 1, 1>}
      : !wafer.storage<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
       into !wafer.storage<tensor<2x4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
     -> !wafer.storage<tensor<2x4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %copy = wafer.move.copy %patch
      : !wafer.storage<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
     -> !wafer.storage<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %transposed = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.storage<tensor<3x2xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %transpose = wafer.move.transpose %transposed
      {permutation = array<i64: 1, 0>}
      : !wafer.storage<tensor<3x2xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
     -> !wafer.storage<tensor<2x3xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %broadcast = wafer.move.broadcast %patch
      {dimensions = array<i64: 1>}
      : !wafer.storage<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
     -> !wafer.storage<tensor<2x4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
}

// CHECK: wafer.move.extract_slice
// CHECK: wafer.move.insert_slice
// CHECK: wafer.move.extract_slice
// CHECK: wafer.move.insert_slice
// CHECK: wafer.move.copy
// CHECK: wafer.move.transpose
// CHECK: wafer.move.broadcast
