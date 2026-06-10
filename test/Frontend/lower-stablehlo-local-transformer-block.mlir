// REQUIRES: stablehlo
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-stablehlo-to-linalg)' %s | FileCheck %s

module {
  func.func @local_transformer_block(
      %x: tensor<2x3x5x8xf32>,
      %hidden_count: tensor<2x3x5xf32>,
      %eps: tensor<2x3x5xf32>,
      %norm_weight: tensor<8xf32>,
      %key: tensor<2x3x7x8xf32>,
      %value: tensor<2x3x7x8xf32>,
      %out_linear_w: tensor<8x8xf32>,
      %out_linear_bias: tensor<8xf32>,
      %gate_w: tensor<8x16xf32>,
      %up_w: tensor<8x16xf32>,
      %down_w: tensor<16x8xf32>) -> tensor<30x8xf32> {
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %neg_inf = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>

    %square = stablehlo.multiply %x, %x : tensor<2x3x5x8xf32>
    %sum = "stablehlo.reduce"(%square, %zero) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %add = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %add : tensor<f32>
    }) {
      dimensions = array<i64: 3>
    } : (tensor<2x3x5x8xf32>, tensor<f32>) -> tensor<2x3x5xf32>
    %mean = stablehlo.divide %sum, %hidden_count : tensor<2x3x5xf32>
    %mean_eps = stablehlo.add %mean, %eps : tensor<2x3x5xf32>
    %rms = stablehlo.rsqrt %mean_eps : tensor<2x3x5xf32>
    %rms_bcast = "stablehlo.broadcast_in_dim"(%rms) {
      broadcast_dimensions = array<i64: 0, 1, 2>
    } : (tensor<2x3x5xf32>) -> tensor<2x3x5x8xf32>
    %normed0 = stablehlo.multiply %x, %rms_bcast : tensor<2x3x5x8xf32>
    %weight_bcast = "stablehlo.broadcast_in_dim"(%norm_weight) {
      broadcast_dimensions = array<i64: 3>
    } : (tensor<8xf32>) -> tensor<2x3x5x8xf32>
    %normed = stablehlo.multiply %normed0, %weight_bcast : tensor<2x3x5x8xf32>

    %scores = "stablehlo.dot_general"(%normed, %key) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_batching_dimensions = [0, 1],
        rhs_batching_dimensions = [0, 1],
        lhs_contracting_dimensions = [3],
        rhs_contracting_dimensions = [3]
      >,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<2x3x5x8xf32>, tensor<2x3x7x8xf32>) -> tensor<2x3x5x7xf32>
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
      %add = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %add : tensor<f32>
    }) {
      dimensions = array<i64: 3>
    } : (tensor<2x3x5x7xf32>, tensor<f32>) -> tensor<2x3x5xf32>
    %row_sum_bcast = "stablehlo.broadcast_in_dim"(%row_sum) {
      broadcast_dimensions = array<i64: 0, 1, 2>
    } : (tensor<2x3x5xf32>) -> tensor<2x3x5x7xf32>
    %prob = stablehlo.divide %exp_scores, %row_sum_bcast : tensor<2x3x5x7xf32>
    %attn = "stablehlo.dot_general"(%prob, %value) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_batching_dimensions = [0, 1],
        rhs_batching_dimensions = [0, 1],
        lhs_contracting_dimensions = [3],
        rhs_contracting_dimensions = [2]
      >,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<2x3x5x7xf32>, tensor<2x3x7x8xf32>) -> tensor<2x3x5x8xf32>

    %attn2d = "stablehlo.reshape"(%attn) : (tensor<2x3x5x8xf32>) -> tensor<30x8xf32>
    %residual2d = "stablehlo.reshape"(%normed) : (tensor<2x3x5x8xf32>) -> tensor<30x8xf32>
    %out_linear = "stablehlo.dot_general"(%attn2d, %out_linear_w) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [0]
      >,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<30x8xf32>, tensor<8x8xf32>) -> tensor<30x8xf32>
    %out_linear_bias_bcast = "stablehlo.broadcast_in_dim"(%out_linear_bias) {
      broadcast_dimensions = array<i64: 1>
    } : (tensor<8xf32>) -> tensor<30x8xf32>
    %biased = stablehlo.add %out_linear, %out_linear_bias_bcast : tensor<30x8xf32>
    %resid = stablehlo.add %biased, %residual2d : tensor<30x8xf32>

    %gate = "stablehlo.dot_general"(%resid, %gate_w) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [0]
      >,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<30x8xf32>, tensor<8x16xf32>) -> tensor<30x16xf32>
    %up = "stablehlo.dot_general"(%resid, %up_w) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [0]
      >,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<30x8xf32>, tensor<8x16xf32>) -> tensor<30x16xf32>
    %activated = stablehlo.tanh %gate : tensor<30x16xf32>
    %gated = stablehlo.multiply %activated, %up : tensor<30x16xf32>
    %out = "stablehlo.dot_general"(%gated, %down_w) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [0]
      >,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<30x16xf32>, tensor<16x8xf32>) -> tensor<30x8xf32>
    return %out : tensor<30x8xf32>
  }
}

// CHECK-LABEL: func.func @local_transformer_block
// CHECK-NOT: stablehlo.
// CHECK: math.rsqrt
// CHECK: linalg.generic
// CHECK: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
// CHECK: math.exp
// CHECK: linalg.generic
// CHECK: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
// CHECK: tensor.collapse_shape
// CHECK: linalg.matmul
// CHECK: math.tanh
// CHECK: linalg.matmul
// CHECK: return %{{.+}} : tensor<30x8xf32>
