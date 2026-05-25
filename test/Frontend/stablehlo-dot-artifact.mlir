// REQUIRES: stablehlo
// RUN: wafer-opt %s | FileCheck %s

module {
  func.func @m0_stablehlo_dot(
      %lhs: tensor<4x8xf16>,
      %rhs: tensor<8x16xf16>) -> tensor<4x16xf16> {
    %0 = "stablehlo.dot_general"(%lhs, %rhs) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_contracting_dimensions = [1],
        rhs_contracting_dimensions = [0]
      >,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<4x8xf16>, tensor<8x16xf16>) -> tensor<4x16xf16>
    return %0 : tensor<4x16xf16>
  }
}

// CHECK-LABEL: func.func @m0_stablehlo_dot(
// CHECK-SAME: %{{[^:]+}}: tensor<4x8xf16>
// CHECK-SAME: %{{[^:]+}}: tensor<8x16xf16>
// CHECK-SAME: ) -> tensor<4x16xf16>
// CHECK: stablehlo.dot_general
// CHECK-SAME: contracting_dims = [1] x [0]
// CHECK-SAME: tensor<4x16xf16>
