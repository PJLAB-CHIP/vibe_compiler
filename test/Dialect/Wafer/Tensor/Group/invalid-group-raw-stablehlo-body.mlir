// REQUIRES: stablehlo
// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %source = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %0 = wafer.group ins(%source : tensor<4xf32>)
                    outs(%source : tensor<4xf32>) {
  ^bb0(%in: tensor<4xf32>, %out: tensor<4xf32>):
    %sum = stablehlo.add %in, %in : tensor<4xf32>
    wafer.group_yield %sum : tensor<4xf32>
  } : tensor<4xf32>
}

// CHECK: 'wafer.group' op body cannot contain raw StableHLO op 'stablehlo.add'
