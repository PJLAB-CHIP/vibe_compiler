// RUN: wafer-opt %s | FileCheck %s

module {
  %source = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %0 = wafer.tile.region(%source : tensor<4xf32>) -> (tensor<4xf32>) {
  ^bb0(%arg0: tensor<4xf32>):
    %local = "builtin.unrealized_conversion_cast"()
        : () -> memref<4xf32, #wafer.memory<spm, tensor>>
    wafer.tile.yield %arg0 : tensor<4xf32>
  }
}

// CHECK: wafer.tile.region(%{{.+}} : tensor<4xf32>) -> (tensor<4xf32>) {
// CHECK: memref<4xf32, #wafer.memory<spm, tensor>>
// CHECK: wafer.tile.yield %{{.+}} : tensor<4xf32>
