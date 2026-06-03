// RUN: wafer-opt --wafer-form-group-candidates %s | FileCheck %s

module {
  func.func @matmul_candidate(
      %lhs: tensor<4x8xf32>,
      %rhs: tensor<8x16xf32>) -> tensor<4x16xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<4x16xf32>
    %init = linalg.fill
        ins(%cst : f32)
        outs(%empty : tensor<4x16xf32>) -> tensor<4x16xf32>
    %0 = linalg.matmul
        ins(%lhs, %rhs : tensor<4x8xf32>, tensor<8x16xf32>)
        outs(%init : tensor<4x16xf32>) -> tensor<4x16xf32>
    return %0 : tensor<4x16xf32>
  }

  func.func @tensor_collective_candidate(
      %input: tensor<4xf32>,
      %out: tensor<8xf32>) -> tensor<8xf32> {
    %0 = wafer.tensor_collective.all_gather
        ins(%input : tensor<4xf32>)
        outs(%out : tensor<8xf32>)
        {axis = 0 : i64, rank_group = array<i64: 0, 1>}
        -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}

// CHECK-LABEL: func.func @matmul_candidate
// CHECK: %[[GROUP:.+]] = wafer.group
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<4x8xf32>, tensor<8x16xf32>)
// CHECK-SAME: outs(%{{.+}} : tensor<4x16xf32>)
// CHECK: linalg.matmul
// CHECK: wafer.group_yield
// CHECK: return %[[GROUP]] : tensor<4x16xf32>

// CHECK-LABEL: func.func @tensor_collective_candidate
// CHECK: %[[TCGROUP:.+]] = wafer.group
// CHECK-SAME: ins(%{{.+}} : tensor<4xf32>)
// CHECK-SAME: outs(%{{.+}} : tensor<8xf32>)
// CHECK: wafer.tensor_collective.all_gather
// CHECK-SAME: rank_group = array<i64: 0, 1>
// CHECK: wafer.group_yield
// CHECK: return %[[TCGROUP]] : tensor<8xf32>
