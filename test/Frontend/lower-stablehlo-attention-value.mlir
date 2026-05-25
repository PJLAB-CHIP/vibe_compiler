// REQUIRES: stablehlo
// RUN: wafer-opt --wafer-lower-stablehlo-dot %s | FileCheck %s

module {
  func.func @attention_value_av(
      %prob: tensor<2x3x5x7xf16>,
      %value: tensor<2x3x7x8xf16>) -> tensor<2x3x5x8xf16> {
    %0 = "stablehlo.dot_general"(%prob, %value) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_batching_dimensions = [0, 1],
        rhs_batching_dimensions = [0, 1],
        lhs_contracting_dimensions = [3],
        rhs_contracting_dimensions = [2]
      >,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    } : (tensor<2x3x5x7xf16>, tensor<2x3x7x8xf16>) -> tensor<2x3x5x8xf16>
    return %0 : tensor<2x3x5x8xf16>
  }
}

// CHECK-DAG: #[[LHS:map[0-9]*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
// CHECK-DAG: #[[RHS:map[0-9]*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
// CHECK-DAG: #[[OUT:map[0-9]*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>

// CHECK-LABEL: func.func @attention_value_av
// CHECK-NOT: stablehlo.dot_general
// CHECK: tensor.empty() : tensor<2x3x5x8xf16>
// CHECK: linalg.fill
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[LHS]], #[[RHS]], #[[OUT]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: return %{{.+}} : tensor<2x3x5x8xf16>
