// REQUIRES: stablehlo
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-stablehlo-to-linalg,wafer-form-logical-groups)' %s | FileCheck %s

module {
  func.func @matmul_bias_relu(
      %lhs: tensor<4x8xf32>,
      %rhs: tensor<8x16xf32>,
      %bias: tensor<16xf32>) -> tensor<4x16xf32> {
    %mm = "stablehlo.dot_general"(%lhs, %rhs) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [0]
      >,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    %bias_bcast = "stablehlo.broadcast_in_dim"(%bias) {
      broadcast_dimensions = array<i64: 1>
    } : (tensor<16xf32>) -> tensor<4x16xf32>
    %biased = stablehlo.add %mm, %bias_bcast : tensor<4x16xf32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<4x16xf32>
    %relu = stablehlo.maximum %biased, %zero : tensor<4x16xf32>
    return %relu : tensor<4x16xf32>
  }
}

// CHECK-LABEL: func.func @matmul_bias_relu
// CHECK: %[[OUT:.+]] = tensor.empty() : tensor<4x16xf32>
// CHECK: %[[GROUP:.+]] = wafer.group ins(%arg0, %arg1, %arg2 : tensor<4x8xf32>, tensor<8x16xf32>, tensor<16xf32>) outs(%[[OUT]] : tensor<4x16xf32>)
// CHECK: linalg.fill
// CHECK: linalg.matmul
// CHECK: linalg.generic
// CHECK: arith.addf
// CHECK: linalg.generic
// CHECK: arith.maximumf
// CHECK: wafer.group.yield
// CHECK-NOT: wafer.group
// CHECK: return %[[GROUP]] : tensor<4x16xf32>
