// RUN: wafer-opt --wafer-form-groups %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>

module {
  func.func @same_shape_elementwise(
      %lhs: tensor<4x8xf16>,
      %rhs: tensor<4x8xf16>,
      %out: tensor<4x8xf16>) -> tensor<4x8xf16> {
    %0 = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%lhs, %rhs : tensor<4x8xf16>, tensor<4x8xf16>)
        outs(%out : tensor<4x8xf16>) {
      ^bb0(%lhs_s: f16, %rhs_s: f16, %out_s: f16):
        %sum = arith.addf %lhs_s, %rhs_s : f16
        linalg.yield %sum : f16
    } -> tensor<4x8xf16>
    return %0 : tensor<4x8xf16>
  }

  func.func @row_broadcast_elementwise(
      %lhs: tensor<4x8xf16>,
      %rhs: tensor<4xf16>,
      %out: tensor<4x8xf16>) -> tensor<4x8xf16> {
    %0 = linalg.generic {
        indexing_maps = [#map, #row, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%lhs, %rhs : tensor<4x8xf16>, tensor<4xf16>)
        outs(%out : tensor<4x8xf16>) {
      ^bb0(%lhs_s: f16, %rhs_s: f16, %out_s: f16):
        %sum = arith.addf %lhs_s, %rhs_s : f16
        linalg.yield %sum : f16
    } -> tensor<4x8xf16>
    return %0 : tensor<4x8xf16>
  }
}

// CHECK-LABEL: func.func @same_shape_elementwise(
// CHECK: %[[GROUP:.+]] = wafer.group
// CHECK: linalg.generic
// CHECK: arith.addf
// CHECK: wafer.group_yield
// CHECK: return %[[GROUP]] : tensor<4x8xf16>

// CHECK-LABEL: func.func @row_broadcast_elementwise(
// CHECK: %[[BGROUP:.+]] = wafer.group
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps
// CHECK: wafer.group_yield
// CHECK: return %[[BGROUP]] : tensor<4x8xf16>
