// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %source = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %0:2 = wafer.group ins(%source : tensor<4xf32>)
                      outs(%source : tensor<4xf32>) {
  ^bb0(%in: tensor<4xf32>, %out: tensor<4xf32>):
    wafer.group.yield %in, %out : tensor<4xf32>, tensor<4xf32>
  } : tensor<4xf32>, tensor<4xf32>
}

// CHECK: expected result count to match outs count, got 2 results and 1 outs
