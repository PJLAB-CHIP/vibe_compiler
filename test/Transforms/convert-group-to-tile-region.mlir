// RUN: wafer-opt --wafer-convert-group-to-tile-region %s | FileCheck %s

func.func @fill_group(%arg0: f32) -> tensor<4xf32> {
  %out = tensor.empty() : tensor<4xf32>
  %group = wafer.group ins(%arg0 : f32) outs(%out : tensor<4xf32>) {
  ^bb0(%value: f32, %dest: tensor<4xf32>):
    %filled = linalg.fill ins(%value : f32) outs(%dest : tensor<4xf32>) -> tensor<4xf32>
    wafer.group_yield %filled : tensor<4xf32>
  } : tensor<4xf32>
  return %group : tensor<4xf32>
}

// CHECK-LABEL: func.func @fill_group
// CHECK-NOT: wafer.group
// CHECK: wafer.tile_region
// CHECK: wafer.storage.load
// CHECK: wafer.compute.fill
// CHECK: wafer.storage.store
// CHECK: wafer.tile_yield
