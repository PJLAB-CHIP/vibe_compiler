// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %source = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %wrong = "builtin.unrealized_conversion_cast"() : () -> tensor<8xf32>
  %0 = wafer.group ins(%source, %wrong : tensor<4xf32>, tensor<8xf32>)
                    outs(%source : tensor<4xf32>) {
  ^bb0(%in: tensor<4xf32>, %bad: tensor<8xf32>, %out: tensor<4xf32>):
    wafer.group_yield %bad : tensor<8xf32>
  } : tensor<4xf32>
}

// CHECK: group yield type 'tensor<8xf32>' does not match result type 'tensor<4xf32>' at index 0
