// RUN: wafer-opt %s | FileCheck %s

module {
  %source = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %0 = wafer.tile.region(%source : tensor<4xf32>) -> (tensor<4xf32>) {
  ^bb0(%arg0: tensor<4xf32>):
    %src = "builtin.unrealized_conversion_cast"()
        : () -> memref<4xf32, #wafer.memory<spm, tensor>>
    %dst = wafer.tile.materialize_layout %src
        : memref<4xf32, #wafer.memory<spm, tensor>>
       -> memref<4xf32, #wafer.memory<spm, cx>>
    wafer.tile.yield %arg0 : tensor<4xf32>
  }
}

// CHECK: %[[SRC:.+]] = {{.*}}unrealized_conversion_cast to memref<4xf32, #wafer.memory<spm, tensor>>
// CHECK: wafer.tile.materialize_layout %[[SRC]] : memref<4xf32, #wafer.memory<spm, tensor>> -> memref<4xf32, #wafer.memory<spm, cx>>
