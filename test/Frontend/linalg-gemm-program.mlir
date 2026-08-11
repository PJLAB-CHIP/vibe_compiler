// RUN: wafer-opt %s | FileCheck %s

module {
  func.func @single_tile_linalg_gemm(
      %lhs: tensor<4x8xf16>,
      %rhs: tensor<8x16xf16>) -> tensor<4x16xf16> {
    %out = tensor.empty() : tensor<4x16xf16>
    %0 = linalg.matmul
        ins(%lhs, %rhs : tensor<4x8xf16>, tensor<8x16xf16>)
        outs(%out : tensor<4x16xf16>) -> tensor<4x16xf16>
    return %0 : tensor<4x16xf16>
  }
}

// CHECK-LABEL: func.func @single_tile_linalg_gemm(
// CHECK-SAME: %{{[^:]+}}: tensor<4x8xf16>
// CHECK-SAME: %{{[^:]+}}: tensor<8x16xf16>
// CHECK-SAME: ) -> tensor<4x16xf16>
// CHECK: tensor.empty() : tensor<4x16xf16>
// CHECK: linalg.matmul
// CHECK: return %{{.+}} : tensor<4x16xf16>
