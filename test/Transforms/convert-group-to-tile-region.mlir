// RUN: wafer-opt --wafer-convert-group-to-tile-region='logical-rank=0' %s | FileCheck %s

func.func @fill_group(%arg0: f32) -> tensor<4xf32> {
  %out = tensor.empty() : tensor<4xf32>
  %group = wafer.group ins(%arg0 : f32) outs(%out : tensor<4xf32>) {
  ^bb0(%value: f32, %dest: tensor<4xf32>):
    %filled = linalg.fill ins(%value : f32) outs(%dest : tensor<4xf32>) -> tensor<4xf32>
    wafer.group.yield %filled : tensor<4xf32>
  } : tensor<4xf32>
  return %group : tensor<4xf32>
}

// CHECK-LABEL: func.func @fill_group
// CHECK-NOT: wafer.group
// CHECK: memref.alloc
// CHECK-SAME: memref<4xf32, #wafer.memory<ddr, tensor>>
// CHECK: wafer.tile.region
// CHECK-SAME: memref<4xf32, #wafer.memory<ddr, tensor>>
// CHECK-NOT: wafer.tile.load
// CHECK: %[[FILL_DEST:.*]] = memref.alloc()
// CHECK-SAME: memref<4xf32, #wafer.memory<spm, tensor>>
// CHECK: wafer.tile.fill %[[FILL_DEST]]
// CHECK-NOT: wafer.tile.load
// CHECK: wafer.tile.store %[[FILL_DEST]]
// CHECK-SAME: memref<4xf32, #wafer.memory<ddr, tensor>>
// CHECK: wafer.tile.yield
// CHECK: bufferization.to_tensor
