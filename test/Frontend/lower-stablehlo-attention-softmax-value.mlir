// REQUIRES: stablehlo
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-stablehlo-to-linalg)' %s | FileCheck %s

module {
  func.func @attention_softmax_value(
      %scores: tensor<2x3x5x7xf32>,
      %value: tensor<2x3x7x8xf32>) -> tensor<2x3x5x8xf32> {
    %neg_inf = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %row_max = "stablehlo.reduce"(%scores, %neg_inf) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %max = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %max : tensor<f32>
    }) {
      dimensions = array<i64: 3>
    } : (tensor<2x3x5x7xf32>, tensor<f32>) -> tensor<2x3x5xf32>
    %row_max_bcast = "stablehlo.broadcast_in_dim"(%row_max) {
      broadcast_dimensions = array<i64: 0, 1, 2>
    } : (tensor<2x3x5xf32>) -> tensor<2x3x5x7xf32>
    %shifted = stablehlo.subtract %scores, %row_max_bcast : tensor<2x3x5x7xf32>
    %exp_scores = stablehlo.exponential %shifted : tensor<2x3x5x7xf32>
    %row_sum = "stablehlo.reduce"(%exp_scores, %zero) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }) {
      dimensions = array<i64: 3>
    } : (tensor<2x3x5x7xf32>, tensor<f32>) -> tensor<2x3x5xf32>
    %row_sum_bcast = "stablehlo.broadcast_in_dim"(%row_sum) {
      broadcast_dimensions = array<i64: 0, 1, 2>
    } : (tensor<2x3x5xf32>) -> tensor<2x3x5x7xf32>
    %prob = stablehlo.divide %exp_scores, %row_sum_bcast : tensor<2x3x5x7xf32>
    %out = "stablehlo.dot_general"(%prob, %value) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_batching_dimensions = [0, 1],
        rhs_batching_dimensions = [0, 1],
        lhs_contracting_dimensions = [3],
        rhs_contracting_dimensions = [2]
      >,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<2x3x5x7xf32>, tensor<2x3x7x8xf32>) -> tensor<2x3x5x8xf32>
    return %out : tensor<2x3x5x8xf32>
  }
}

// CHECK-LABEL: func.func @attention_softmax_value
// CHECK-NOT: stablehlo.
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK: arith.maximumf
// CHECK: linalg.generic
// CHECK: math.exp
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK: arith.addf
// CHECK: linalg.generic
// CHECK: arith.divf
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: return %{{.+}} : tensor<2x3x5x8xf32>
