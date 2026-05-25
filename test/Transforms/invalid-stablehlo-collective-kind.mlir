// REQUIRES: stablehlo
// RUN: not wafer-opt --wafer-lower-stablehlo-collectives-to-comm %s 2>&1 | FileCheck %s

module {
  func.func @unsupported_all_reduce(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "stablehlo.all_reduce"(%arg0) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %prod = stablehlo.multiply %lhs, %rhs : tensor<f32>
      "stablehlo.return"(%prod) : (tensor<f32>) -> ()
    }) {
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// CHECK: only sum/max/min StableHLO collective reductions are supported
