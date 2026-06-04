// RUN: wafer-opt --pass-pipeline='builtin.module(canonicalize)' %s | FileCheck %s

func.func @fold_same_type_view(%arg0: !wafer.tile_buffer<tensor<4x8xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>) -> !wafer.tile_buffer<tensor<4x8xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>> {
  %0 = wafer.view.reshape %arg0 : !wafer.tile_buffer<tensor<4x8xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>> -> !wafer.tile_buffer<tensor<4x8xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  return %0 : !wafer.tile_buffer<tensor<4x8xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
}

// CHECK-LABEL: func.func @fold_same_type_view
// CHECK-NOT: wafer.view.reshape
// CHECK: return %arg0

func.func @keep_shape_change_view(%arg0: !wafer.tile_buffer<tensor<4x8xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>) -> !wafer.tile_buffer<tensor<2x16xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>> {
  %0 = wafer.view.reshape %arg0 : !wafer.tile_buffer<tensor<4x8xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>> -> !wafer.tile_buffer<tensor<2x16xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  return %0 : !wafer.tile_buffer<tensor<2x16xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
}

// CHECK-LABEL: func.func @keep_shape_change_view
// CHECK: wafer.view.reshape
