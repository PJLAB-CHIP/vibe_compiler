// REQUIRES: stablehlo
// RUN: wafer-opt --wafer-lower-stablehlo-dot --wafer-lower-stablehlo-elementwise --wafer-check-mlp-schedule %s | FileCheck %s

module {
  func.func @mlp_tanh_gate(
      %hidden: tensor<4x8xf32>,
      %gate_w: tensor<8x16xf32>,
      %up_w: tensor<8x16xf32>,
      %down_w: tensor<16x8xf32>) -> tensor<4x8xf32> {
    %gate = "stablehlo.dot_general"(%hidden, %gate_w) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [0]
      >,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    %up = "stablehlo.dot_general"(%hidden, %up_w) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [0]
      >,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    %activated = stablehlo.tanh %gate : tensor<4x16xf32>
    %gated = stablehlo.multiply %activated, %up : tensor<4x16xf32>
    %out = "stablehlo.dot_general"(%gated, %down_w) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [0]
      >,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<4x16xf32>, tensor<16x8xf32>) -> tensor<4x8xf32>
    return %out : tensor<4x8xf32>
  }
}

// CHECK-LABEL: func.func @mlp_tanh_gate
// CHECK-NOT: stablehlo.
// CHECK: linalg.matmul
// CHECK: linalg.matmul
// CHECK: linalg.generic
// CHECK: math.tanh
// CHECK: linalg.generic
// CHECK: arith.mulf
// CHECK: linalg.matmul
// CHECK: return %{{.+}} : tensor<4x8xf32>
