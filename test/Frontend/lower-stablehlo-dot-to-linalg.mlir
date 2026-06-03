// REQUIRES: stablehlo
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-stablehlo-to-linalg)' %s | FileCheck %s

module {
  func.func @single_tile_lower_dot(
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

// CHECK-LABEL: func.func @single_tile_lower_dot
// CHECK-NOT: stablehlo.dot_general
// CHECK: tensor.empty() : tensor<4x16xf16>
// CHECK: linalg.fill
// CHECK: linalg.matmul
// CHECK: return %{{.+}} : tensor<4x16xf16>
