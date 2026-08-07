// REQUIRES: stablehlo
// RUN: not wafer-opt --pass-pipeline='builtin.module(wafer-lower-stablehlo-to-linalg,wafer-materialize-flash-decoding{output-tile-sizes=1,1,1,8 key-value-tile-size=4 split-count=2})' %s 2>&1 | FileCheck %s

// A one-token query is still ordinary read-only attention. Without exact
// past/new appends, attention consumption, and returned updated K/V state it
// must never enter FlashDecoding.
module {
  func.func @one_token_read_only_attention(
      %scores: tensor<1x1x1x7xf32>,
      %value: tensor<1x1x7x8xf32>) -> tensor<1x1x1x8xf32> {
    %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %row_max = "stablehlo.reduce"(%scores, %neg_inf) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %max = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %max : tensor<f32>
    }) {dimensions = array<i64: 3>}
        : (tensor<1x1x1x7xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    %row_max_bcast = "stablehlo.broadcast_in_dim"(%row_max) {
      broadcast_dimensions = array<i64: 0, 1, 2>
    } : (tensor<1x1x1xf32>) -> tensor<1x1x1x7xf32>
    %shifted = stablehlo.subtract %scores, %row_max_bcast
        : tensor<1x1x1x7xf32>
    %exponential = stablehlo.exponential %shifted : tensor<1x1x1x7xf32>
    %row_sum = "stablehlo.reduce"(%exponential, %zero) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }) {dimensions = array<i64: 3>}
        : (tensor<1x1x1x7xf32>, tensor<f32>) -> tensor<1x1x1xf32>
    %row_sum_bcast = "stablehlo.broadcast_in_dim"(%row_sum) {
      broadcast_dimensions = array<i64: 0, 1, 2>
    } : (tensor<1x1x1xf32>) -> tensor<1x1x1x7xf32>
    %probability = stablehlo.divide %exponential, %row_sum_bcast
        : tensor<1x1x1x7xf32>
    %output = "stablehlo.dot_general"(%probability, %value) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_batching_dimensions = [0, 1],
        rhs_batching_dimensions = [0, 1],
        lhs_contracting_dimensions = [3],
        rhs_contracting_dimensions = [2]
      >,
      precision_config = [#stablehlo<precision DEFAULT>,
                          #stablehlo<precision DEFAULT>]
    } : (tensor<1x1x1x7xf32>, tensor<1x1x7x8xf32>)
        -> tensor<1x1x1x8xf32>
    return %output : tensor<1x1x1x8xf32>
  }
}

// CHECK: flash_decoding_materialization_failed:
// CHECK-SAME: decode requires returned K/V appends consumed by attention
