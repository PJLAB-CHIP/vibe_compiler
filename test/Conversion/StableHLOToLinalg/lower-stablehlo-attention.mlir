// REQUIRES: stablehlo
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-stablehlo-to-linalg)' %s | FileCheck %s

module {
  func.func @scaled_masked_attention(
      %query: tensor<2x4xf16>, %key: tensor<5x4xf16>,
      %value: tensor<5x6xf16>, %mask: tensor<2x5xf16>)
      -> tensor<2x6xf16> {
    %zero = stablehlo.constant dense<0.0> : tensor<f16>
    %lowest = stablehlo.constant dense<-6.5504E+4> : tensor<f16>
    %scale = stablehlo.constant dense<0.5> : tensor<f16>
    %scores = "stablehlo.dot_general"(%query, %key) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [1]>,
      precision_config = [#stablehlo<precision DEFAULT>,
                          #stablehlo<precision DEFAULT>]
    } : (tensor<2x4xf16>, tensor<5x4xf16>) -> tensor<2x5xf16>
    %scale_broadcast = "stablehlo.broadcast_in_dim"(%scale) {
      broadcast_dimensions = array<i64>
    } : (tensor<f16>) -> tensor<2x5xf16>
    %scaled = stablehlo.multiply %scores, %scale_broadcast
        : tensor<2x5xf16>
    %masked = stablehlo.add %scaled, %mask : tensor<2x5xf16>
    %row_max = "stablehlo.reduce"(%masked, %lowest) ({
    ^bb0(%lhs: tensor<f16>, %rhs: tensor<f16>):
      %maximum = stablehlo.maximum %lhs, %rhs : tensor<f16>
      stablehlo.return %maximum : tensor<f16>
    }) {dimensions = array<i64: 1>}
        : (tensor<2x5xf16>, tensor<f16>) -> tensor<2xf16>
    %max_broadcast = "stablehlo.broadcast_in_dim"(%row_max) {
      broadcast_dimensions = array<i64: 0>
    } : (tensor<2xf16>) -> tensor<2x5xf16>
    %shifted = stablehlo.subtract %masked, %max_broadcast
        : tensor<2x5xf16>
    %exponential = stablehlo.exponential %shifted : tensor<2x5xf16>
    %row_sum = "stablehlo.reduce"(%exponential, %zero) ({
    ^bb0(%lhs: tensor<f16>, %rhs: tensor<f16>):
      %sum = stablehlo.add %lhs, %rhs : tensor<f16>
      stablehlo.return %sum : tensor<f16>
    }) {dimensions = array<i64: 1>}
        : (tensor<2x5xf16>, tensor<f16>) -> tensor<2xf16>
    %sum_broadcast = "stablehlo.broadcast_in_dim"(%row_sum) {
      broadcast_dimensions = array<i64: 0>
    } : (tensor<2xf16>) -> tensor<2x5xf16>
    %probability = stablehlo.divide %exponential, %sum_broadcast
        : tensor<2x5xf16>
    %result = "stablehlo.dot_general"(%probability, %value) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [0]>,
      precision_config = [#stablehlo<precision DEFAULT>,
                          #stablehlo<precision DEFAULT>]
    } : (tensor<2x5xf16>, tensor<5x6xf16>) -> tensor<2x6xf16>
    return %result : tensor<2x6xf16>
  }

  // Minimized from the current PyTorch/XLA attention export shape family:
  // batch/head views are collapsed around the contractions while scale and
  // softmax use a promoted element type in the logical score domain.
  func.func @pytorch_viewed_attention(
      %query: tensor<1x2x3x4xf16>, %key: tensor<1x2x5x4xf16>,
      %value: tensor<1x2x5x6xf16>, %mask: tensor<1x2x3x5xf16>)
      -> tensor<1x2x3x6xf16> {
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    %lowest = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %scale = stablehlo.constant dense<0.125>
        : tensor<1x2x3x5xf32>
    %query_view = stablehlo.reshape %query
        : (tensor<1x2x3x4xf16>) -> tensor<2x3x4xf16>
    %key_transposed = stablehlo.transpose %key, dims = [0, 1, 3, 2]
        : (tensor<1x2x5x4xf16>) -> tensor<1x2x4x5xf16>
    %key_view = stablehlo.reshape %key_transposed
        : (tensor<1x2x4x5xf16>) -> tensor<2x4x5xf16>
    %scores_view = "stablehlo.dot_general"(%query_view, %key_view) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_batching_dimensions = [0], rhs_batching_dimensions = [0],
        lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
      precision_config = [#stablehlo<precision DEFAULT>,
                          #stablehlo<precision DEFAULT>]
    } : (tensor<2x3x4xf16>, tensor<2x4x5xf16>) -> tensor<2x3x5xf16>
    %scores = stablehlo.reshape %scores_view
        : (tensor<2x3x5xf16>) -> tensor<1x2x3x5xf16>
    %scores_f32 = stablehlo.convert %scores
        : (tensor<1x2x3x5xf16>) -> tensor<1x2x3x5xf32>
    %scaled_f32 = stablehlo.multiply %scores_f32, %scale
        : tensor<1x2x3x5xf32>
    %scaled = stablehlo.convert %scaled_f32
        : (tensor<1x2x3x5xf32>) -> tensor<1x2x3x5xf16>
    %masked = stablehlo.add %scaled, %mask : tensor<1x2x3x5xf16>
    %masked_f32 = stablehlo.convert %masked
        : (tensor<1x2x3x5xf16>) -> tensor<1x2x3x5xf32>
    %row_max = "stablehlo.reduce"(%masked_f32, %lowest) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %maximum = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %maximum : tensor<f32>
    }) {dimensions = array<i64: 3>}
        : (tensor<1x2x3x5xf32>, tensor<f32>) -> tensor<1x2x3xf32>
    %max_broadcast = "stablehlo.broadcast_in_dim"(%row_max) {
      broadcast_dimensions = array<i64: 0, 1, 2>
    } : (tensor<1x2x3xf32>) -> tensor<1x2x3x5xf32>
    %shifted = stablehlo.subtract %masked_f32, %max_broadcast
        : tensor<1x2x3x5xf32>
    %exponential = stablehlo.exponential %shifted
        : tensor<1x2x3x5xf32>
    %row_sum = "stablehlo.reduce"(%exponential, %zero) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }) {dimensions = array<i64: 3>}
        : (tensor<1x2x3x5xf32>, tensor<f32>) -> tensor<1x2x3xf32>
    %sum_broadcast = "stablehlo.broadcast_in_dim"(%row_sum) {
      broadcast_dimensions = array<i64: 0, 1, 2>
    } : (tensor<1x2x3xf32>) -> tensor<1x2x3x5xf32>
    %probability_f32 = stablehlo.divide %exponential, %sum_broadcast
        : tensor<1x2x3x5xf32>
    %probability = stablehlo.convert %probability_f32
        : (tensor<1x2x3x5xf32>) -> tensor<1x2x3x5xf16>
    %probability_view = stablehlo.reshape %probability
        : (tensor<1x2x3x5xf16>) -> tensor<2x3x5xf16>
    %value_view = stablehlo.reshape %value
        : (tensor<1x2x5x6xf16>) -> tensor<2x5x6xf16>
    %result_view = "stablehlo.dot_general"(%probability_view, %value_view) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_batching_dimensions = [0], rhs_batching_dimensions = [0],
        lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
      precision_config = [#stablehlo<precision DEFAULT>,
                          #stablehlo<precision DEFAULT>]
    } : (tensor<2x3x5xf16>, tensor<2x5x6xf16>) -> tensor<2x3x6xf16>
    %result = stablehlo.reshape %result_view
        : (tensor<2x3x6xf16>) -> tensor<1x2x3x6xf16>
    return %result : tensor<1x2x3x6xf16>
  }
}

// CHECK-LABEL: func.func @scaled_masked_attention
// CHECK: %[[SCALE:.+]] = tensor.extract
// CHECK: %[[ATTENTION:.+]] = wafer.linalg_ext.attention
// CHECK-SAME: algorithm(<flash_attention>)
// CHECK-SAME: indexing_maps = [
// CHECK: return %[[ATTENTION]] : tensor<2x6xf16>
// CHECK-NOT: stablehlo.
// CHECK-NOT: linalg.matmul
// CHECK-NOT: math.exp

// CHECK-LABEL: func.func @pytorch_viewed_attention
// CHECK: %[[VIEWED_ATTENTION:.+]] = wafer.linalg_ext.attention
// CHECK-SAME: ins(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}} : tensor<1x2x3x4xf16>, tensor<1x2x5x4xf16>, tensor<1x2x5x6xf16>, f32, tensor<1x2x3x5xf16>)
// CHECK-SAME: algorithm(<flash_attention>)
// CHECK: return %[[VIEWED_ATTENTION]] : tensor<1x2x3x6xf16>
