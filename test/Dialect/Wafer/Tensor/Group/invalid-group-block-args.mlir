// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %source = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %dest = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %0 = wafer.group ins(%source : tensor<4xf32>)
                    outs(%dest : tensor<4xf32>) {
  ^bb0(%in: tensor<4xf32>):
    wafer.group.yield %in : tensor<4xf32>
  } : tensor<4xf32>
}

// CHECK: expected 2 body block arguments matching group ins plus outs, got 1
