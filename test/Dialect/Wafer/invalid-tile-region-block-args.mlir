// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %source = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %0 = wafer.tile_region(%source : tensor<4xf32>) -> (tensor<4xf32>) {
  ^bb0:
    wafer.tile_yield %source : tensor<4xf32>
  }
}

// CHECK: expected 1 body block arguments matching tile_region inputs, got 0
