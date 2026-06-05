// RUN: wafer-opt --pass-pipeline='builtin.module(canonicalize)' %s | FileCheck %s

func.func @fold_same_type_view(%arg0: !wafer.storage<tensor<4x8xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>) -> !wafer.storage<tensor<4x8xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>> {
  %0 = wafer.tile.reshape %arg0 : !wafer.storage<tensor<4x8xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>> -> !wafer.storage<tensor<4x8xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  return %0 : !wafer.storage<tensor<4x8xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
}

// CHECK-LABEL: func.func @fold_same_type_view
// CHECK-NOT: wafer.tile.reshape
// CHECK: return %arg0

func.func @keep_shape_change_view(%arg0: !wafer.storage<tensor<4x8xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>) -> !wafer.storage<tensor<2x16xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>> {
  %0 = wafer.tile.reshape %arg0 : !wafer.storage<tensor<4x8xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>> -> !wafer.storage<tensor<2x16xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  return %0 : !wafer.storage<tensor<2x16xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
}

// CHECK-LABEL: func.func @keep_shape_change_view
// CHECK: wafer.tile.reshape
