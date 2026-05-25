// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %source = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %wrong = "builtin.unrealized_conversion_cast"() : () -> tensor<8xf32>
  %0 = wafer.tile_region(%source, %wrong : tensor<4xf32>, tensor<8xf32>) -> (tensor<4xf32>) {
  ^bb0(%arg0: tensor<4xf32>, %bad: tensor<8xf32>):
    wafer.tile_yield %bad : tensor<8xf32>
  }
}

// CHECK: tile_yield type 'tensor<8xf32>' does not match tile_region result type 'tensor<4xf32>' at index 0
