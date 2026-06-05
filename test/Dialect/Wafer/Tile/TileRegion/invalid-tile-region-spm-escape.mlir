// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %source = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %0 = wafer.tile.region(%source : tensor<4xf32>)
      -> (memref<4xf32, #wafer.memory<spm, tensor>>) {
  ^bb0(%arg0: tensor<4xf32>):
    %local = "builtin.unrealized_conversion_cast"()
        : () -> memref<4xf32, #wafer.memory<spm, tensor>>
    wafer.tile.yield %local
        : memref<4xf32, #wafer.memory<spm, tensor>>
  }
}

// CHECK: SPM buffer values cannot cross wafer.tile.region boundaries
