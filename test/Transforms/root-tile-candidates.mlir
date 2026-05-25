// RUN: wafer-opt --wafer-form-groups --wafer-check-root-tile-candidates %s | FileCheck %s

module {
  func.func @single_matmul(
      %lhs: tensor<4x8xf16>,
      %rhs: tensor<8x16xf16>,
      %out: tensor<4x16xf16>) -> tensor<4x16xf16> {
    %0 = linalg.matmul
        ins(%lhs, %rhs : tensor<4x8xf16>, tensor<8x16xf16>)
        outs(%out : tensor<4x16xf16>) -> tensor<4x16xf16>
    return %0 : tensor<4x16xf16>
  }
}

// CHECK-LABEL: func.func @single_matmul(
// CHECK: %[[GROUP:.+]] = wafer.group
// CHECK-SAME: ins(
// CHECK-SAME: outs(
// CHECK-NOT: root_tile
// CHECK-NOT: tile_shape
// CHECK: {
// CHECK: linalg.matmul
// CHECK: wafer.group_yield
// CHECK: return %[[GROUP]] : tensor<4x16xf16>
