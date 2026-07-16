// RUN: wafer-opt %s | FileCheck %s

module {
  %source = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %0 = wafer.tile.region(%source : tensor<4xf32>) -> (tensor<4xf32>) {
  ^bb0(%arg0: tensor<4xf32>):
    %local = "builtin.unrealized_conversion_cast"()
        : () -> memref<4xf32, #wafer.memory<spm, tensor>>
    wafer.tile.yield %arg0 : tensor<4xf32>
  }

  %resident = wafer.tile.region(%source : tensor<4xf32>) ->
      (memref<4xf32, #wafer.memory<spm, tensor>>) {
  ^bb0(%unused: tensor<4xf32>):
    %local = memref.alloc()
        : memref<4xf32, #wafer.memory<spm, tensor>>
    wafer.tile.yield %local
        : memref<4xf32, #wafer.memory<spm, tensor>>
  }
  %forwarded = wafer.tile.region(%resident
      : memref<4xf32, #wafer.memory<spm, tensor>>) ->
      (memref<2x2xf32, #wafer.memory<spm, tensor>>) {
  ^bb0(%arg0: memref<4xf32, #wafer.memory<spm, tensor>>):
    %reshaped = wafer.tile.reshape %arg0
        : memref<4xf32, #wafer.memory<spm, tensor>> ->
          memref<2x2xf32, #wafer.memory<spm, tensor>>
    wafer.tile.yield %reshaped
        : memref<2x2xf32, #wafer.memory<spm, tensor>>
  }
}

// CHECK: wafer.tile.region(%{{.+}} : tensor<4xf32>) -> (tensor<4xf32>) {
// CHECK: memref<4xf32, #wafer.memory<spm, tensor>>
// CHECK: wafer.tile.yield %{{.+}} : tensor<4xf32>
// CHECK: %[[RESIDENT:.+]] = wafer.tile.region(%{{.+}} : tensor<4xf32>) -> (memref<4xf32, #wafer.memory<spm, tensor>>) {
// CHECK: %[[LOCAL:.+]] = memref.alloc() : memref<4xf32, #wafer.memory<spm, tensor>>
// CHECK: wafer.tile.yield %[[LOCAL]] : memref<4xf32, #wafer.memory<spm, tensor>>
// CHECK: wafer.tile.region(%[[RESIDENT]] : memref<4xf32, #wafer.memory<spm, tensor>>) -> (memref<2x2xf32, #wafer.memory<spm, tensor>>) {
// CHECK: wafer.tile.reshape
