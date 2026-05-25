// RUN: wafer-opt --wafer-form-groups %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>

module {
  func.func @same_shape_elementwise(
      %lhs: tensor<4x8xf16>,
      %rhs: tensor<4x8xf16>,
      %out: tensor<4x8xf16>) -> tensor<4x8xf16> {
    %0 = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%lhs, %rhs : tensor<4x8xf16>, tensor<4x8xf16>)
        outs(%out : tensor<4x8xf16>) -> tensor<4x8xf16>
    return %0 : tensor<4x8xf16>
  }

  func.func @row_broadcast_elementwise(
      %lhs: tensor<4x8xf16>,
      %rhs: tensor<4xf16>,
      %out: tensor<4x8xf16>) -> tensor<4x8xf16> {
    %0 = linalg.elementwise kind=#linalg.elementwise_kind<add>
        indexing_maps = [#map, #row, #map]
        ins(%lhs, %rhs : tensor<4x8xf16>, tensor<4xf16>)
        outs(%out : tensor<4x8xf16>) -> tensor<4x8xf16>
    return %0 : tensor<4x8xf16>
  }
}

// CHECK-LABEL: func.func @same_shape_elementwise(
// CHECK: %[[GROUP:.+]] = wafer.group
// CHECK: linalg.elementwise
// CHECK-SAME: kind=#linalg.elementwise_kind<add>
// CHECK: wafer.group_yield
// CHECK: return %[[GROUP]] : tensor<4x8xf16>

// CHECK-LABEL: func.func @row_broadcast_elementwise(
// CHECK: %[[BGROUP:.+]] = wafer.group
// CHECK: linalg.elementwise
// CHECK-SAME: indexing_maps
// CHECK: wafer.group_yield
// CHECK: return %[[BGROUP]] : tensor<4x8xf16>
