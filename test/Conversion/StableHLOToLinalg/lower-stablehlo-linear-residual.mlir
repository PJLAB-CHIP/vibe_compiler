// REQUIRES: stablehlo
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-stablehlo-to-linalg)' %s | FileCheck %s

module {
  func.func @linear_residual(
      %hidden: tensor<4x8xf32>,
      %weight: tensor<8x16xf32>,
      %bias: tensor<16xf32>,
      %residual: tensor<4x16xf32>) -> tensor<4x16xf32> {
    %linear = "stablehlo.dot_general"(%hidden, %weight) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [0]
      >,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    %bias_bcast = "stablehlo.broadcast_in_dim"(%bias) {
      broadcast_dimensions = array<i64: 1>
    } : (tensor<16xf32>) -> tensor<4x16xf32>
    %biased = stablehlo.add %linear, %bias_bcast : tensor<4x16xf32>
    %out = stablehlo.add %biased, %residual : tensor<4x16xf32>
    return %out : tensor<4x16xf32>
  }
}

// CHECK-LABEL: func.func @linear_residual
// CHECK-NOT: stablehlo.
// CHECK: %[[LINEAR:.+]] = linalg.matmul
// CHECK: %[[BIASED:.+]] = linalg.generic
// CHECK-SAME: ins(%[[LINEAR]], %arg2 : tensor<4x16xf32>, tensor<16xf32>)
// CHECK: arith.addf
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[BIASED]], %arg3 : tensor<4x16xf32>, tensor<4x16xf32>)
// CHECK: arith.addf
// CHECK-NOT: wafer.
// CHECK: return %{{.+}} : tensor<4x16xf32>
