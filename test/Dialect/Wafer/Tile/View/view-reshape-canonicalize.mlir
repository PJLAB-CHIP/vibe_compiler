// RUN: wafer-opt --pass-pipeline='builtin.module(canonicalize)' %s | FileCheck %s

func.func @fold_same_type_view(%arg0: memref<4x8xf32, #wafer.memory<spm, tensor>>) -> memref<4x8xf32, #wafer.memory<spm, tensor>> {
  %0 = wafer.tile.reshape %arg0 : memref<4x8xf32, #wafer.memory<spm, tensor>> -> memref<4x8xf32, #wafer.memory<spm, tensor>>
  return %0 : memref<4x8xf32, #wafer.memory<spm, tensor>>
}

// CHECK-LABEL: func.func @fold_same_type_view
// CHECK-NOT: wafer.tile.reshape
// CHECK: return %arg0

func.func @keep_shape_change_view(%arg0: memref<4x8xf32, #wafer.memory<spm, tensor>>) -> memref<2x16xf32, #wafer.memory<spm, tensor>> {
  %0 = wafer.tile.reshape %arg0 : memref<4x8xf32, #wafer.memory<spm, tensor>> -> memref<2x16xf32, #wafer.memory<spm, tensor>>
  return %0 : memref<2x16xf32, #wafer.memory<spm, tensor>>
}

// CHECK-LABEL: func.func @keep_shape_change_view
// CHECK: wafer.tile.reshape
