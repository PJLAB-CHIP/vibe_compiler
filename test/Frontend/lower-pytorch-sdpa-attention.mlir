// REQUIRES: stablehlo
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-stablehlo-to-linalg)' %s | FileCheck %s

// Minimized typed witness from the current product exporter for
// torch.nn.functional.scaled_dot_product_attention(is_causal=true). It keeps
// the exporter-produced split Q/K scaling and compare/select causal mask.
module {
  func.func @native_sdpa(
      %value: tensor<1x2x4x8xf16>, %enabled: tensor<i1>,
      %key: tensor<1x2x4x8xf16>, %query: tensor<1x2x4x8xf16>)
      -> tensor<1x2x4x8xf16> {
    %mask_zero = stablehlo.constant dense<0.0> : tensor<4x4xf32>
    %mask_lowest = stablehlo.constant dense<0xFF800000> : tensor<4x4xf32>
    %int_zero = stablehlo.constant dense<0> : tensor<4x4xi64>
    %key_scale = stablehlo.constant dense<0.594603539>
        : tensor<1x2x8x4xf32>
    %query_scale = stablehlo.constant dense<0.594603539>
        : tensor<1x2x4x8xf32>
    %sum_zero = stablehlo.constant dense<0.0> : tensor<f32>
    %max_lowest = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %indices = stablehlo.constant dense<[0, 1, 2, 3]> : tensor<4xi64>
    %query_f32 = stablehlo.convert %query
        : (tensor<1x2x4x8xf16>) -> tensor<1x2x4x8xf32>
    %scaled_query = stablehlo.multiply %query_f32, %query_scale
        : tensor<1x2x4x8xf32>
    %query_view = stablehlo.reshape %scaled_query
        : (tensor<1x2x4x8xf32>) -> tensor<2x4x8xf32>
    %key_f32 = stablehlo.convert %key
        : (tensor<1x2x4x8xf16>) -> tensor<1x2x4x8xf32>
    %key_transpose = stablehlo.transpose %key_f32, dims = [0, 1, 3, 2]
        : (tensor<1x2x4x8xf32>) -> tensor<1x2x8x4xf32>
    %scaled_key = stablehlo.multiply %key_transpose, %key_scale
        : tensor<1x2x8x4xf32>
    %key_view = stablehlo.reshape %scaled_key
        : (tensor<1x2x8x4xf32>) -> tensor<2x8x4xf32>
    %scores_view = stablehlo.dot_general %query_view, %key_view,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<2x4x8xf32>, tensor<2x8x4xf32>) -> tensor<2x4x4xf32>
    %scores = stablehlo.reshape %scores_view
        : (tensor<2x4x4xf32>) -> tensor<1x2x4x4xf32>
    %rows = stablehlo.broadcast_in_dim %indices, dims = [1]
        : (tensor<4xi64>) -> tensor<4x4xi64>
    %columns = stablehlo.broadcast_in_dim %indices, dims = [0]
        : (tensor<4xi64>) -> tensor<4x4xi64>
    %distance = stablehlo.subtract %rows, %columns : tensor<4x4xi64>
    %causal = stablehlo.compare LE, %distance, %int_zero
        : (tensor<4x4xi64>, tensor<4x4xi64>) -> tensor<4x4xi1>
    %enabled_mask = stablehlo.broadcast_in_dim %enabled, dims = []
        : (tensor<i1>) -> tensor<4x4xi1>
    %condition = stablehlo.and %causal, %enabled_mask : tensor<4x4xi1>
    %mask = stablehlo.select %condition, %mask_zero, %mask_lowest
        : tensor<4x4xi1>, tensor<4x4xf32>
    %mask_broadcast = stablehlo.broadcast_in_dim %mask, dims = [2, 3]
        : (tensor<4x4xf32>) -> tensor<1x2x4x4xf32>
    %masked = stablehlo.add %scores, %mask_broadcast
        : tensor<1x2x4x4xf32>
    %row_max = stablehlo.reduce(%masked init: %max_lowest)
        applies stablehlo.maximum across dimensions = [3]
        : (tensor<1x2x4x4xf32>, tensor<f32>) -> tensor<1x2x4xf32>
    %max_broadcast = stablehlo.broadcast_in_dim %row_max, dims = [0, 1, 2]
        : (tensor<1x2x4xf32>) -> tensor<1x2x4x4xf32>
    %shifted = stablehlo.subtract %masked, %max_broadcast
        : tensor<1x2x4x4xf32>
    %exponential = stablehlo.exponential %shifted : tensor<1x2x4x4xf32>
    %row_sum = stablehlo.reduce(%exponential init: %sum_zero)
        applies stablehlo.add across dimensions = [3]
        : (tensor<1x2x4x4xf32>, tensor<f32>) -> tensor<1x2x4xf32>
    %sum_broadcast = stablehlo.broadcast_in_dim %row_sum, dims = [0, 1, 2]
        : (tensor<1x2x4xf32>) -> tensor<1x2x4x4xf32>
    %probability = stablehlo.divide %exponential, %sum_broadcast
        : tensor<1x2x4x4xf32>
    %probability_view = stablehlo.reshape %probability
        : (tensor<1x2x4x4xf32>) -> tensor<2x4x4xf32>
    %value_f32 = stablehlo.convert %value
        : (tensor<1x2x4x8xf16>) -> tensor<1x2x4x8xf32>
    %value_view = stablehlo.reshape %value_f32
        : (tensor<1x2x4x8xf32>) -> tensor<2x4x8xf32>
    %result_view = stablehlo.dot_general %probability_view, %value_view,
        batching_dims = [0] x [0], contracting_dims = [2] x [1]
        : (tensor<2x4x4xf32>, tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
    %result_f32 = stablehlo.reshape %result_view
        : (tensor<2x4x8xf32>) -> tensor<1x2x4x8xf32>
    %result = stablehlo.convert %result_f32
        : (tensor<1x2x4x8xf32>) -> tensor<1x2x4x8xf16>
    return %result : tensor<1x2x4x8xf16>
  }
}

// CHECK-LABEL: func.func @native_sdpa
// CHECK: arith.select
// CHECK: %[[ATTENTION:.+]] = wafer.linalg_ext.attention
// CHECK-SAME: tensor<1x2x4x8xf32>, tensor<1x2x8x4xf32>, tensor<1x2x4x8xf32>, f32, tensor<1x2x4x4xf32>
// CHECK-SAME: algorithm(<flash_attention>)
// CHECK: arith.truncf
// CHECK: return %{{.+}} : tensor<1x2x4x8xf16>
// CHECK-NOT: math.exp
