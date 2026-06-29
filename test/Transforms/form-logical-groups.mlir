// RUN: wafer-opt --wafer-form-logical-groups %s | FileCheck %s

module {
  func.func @matmul_logical_group(
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

  func.func @linalg_ext_collective_logical_group(
      %input: tensor<4xf32>,
      %out: tensor<8xf32>) -> tensor<8xf32> {
    %0 = wafer.linalg_ext.collective.all_gather
        ins(%input : tensor<4xf32>)
        outs(%out : tensor<8xf32>)
        {axis = 0 : i64, rank_group = array<i64: 0, 1>}
        -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }

  func.func @collective_with_static_insert_slice_producer(
      %lhs: tensor<1x1x4x1xf32>,
      %rhs: tensor<1x1x4x1xf32>,
      %out: tensor<1x1x4x6xf32>) -> tensor<1x1x4x6xf32> {
    %zero = arith.constant dense<0.000000e+00> : tensor<1x1x4x6xf32>
    %insert0 = tensor.insert_slice %lhs into %zero[0, 0, 0, 0] [1, 1, 4, 1] [1, 1, 1, 1]
        : tensor<1x1x4x1xf32> into tensor<1x1x4x6xf32>
    %insert1 = tensor.insert_slice %rhs into %insert0[0, 0, 0, 2] [1, 1, 4, 1] [1, 1, 1, 1]
        : tensor<1x1x4x1xf32> into tensor<1x1x4x6xf32>
    %0 = wafer.linalg_ext.collective.all_reduce
        ins(%insert1 : tensor<1x1x4x6xf32>)
        outs(%out : tensor<1x1x4x6xf32>) {
    ^bb0(%lhs_scalar: f32, %rhs_scalar: f32):
      %sum = arith.addf %lhs_scalar, %rhs_scalar : f32
      wafer.linalg_ext.collective.yield %sum : f32
    } {rank_group = array<i64: 0, 1>}
        -> tensor<1x1x4x6xf32>
    return %0 : tensor<1x1x4x6xf32>
  }
}

// CHECK-LABEL: func.func @matmul_logical_group
// CHECK: %[[GROUP:.+]] = wafer.group
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<4x8xf32>, tensor<8x16xf32>)
// CHECK-SAME: outs(%{{.+}} : tensor<4x16xf32>)
// CHECK: linalg.matmul
// CHECK: wafer.group.yield
// CHECK: return %[[GROUP]] : tensor<4x16xf32>

// CHECK-LABEL: func.func @linalg_ext_collective_logical_group
// CHECK: %[[TCGROUP:.+]] = wafer.group
// CHECK-SAME: ins(%{{.+}} : tensor<4xf32>)
// CHECK-SAME: outs(%{{.+}} : tensor<8xf32>)
// CHECK: wafer.linalg_ext.collective.all_gather
// CHECK-SAME: rank_group = array<i64: 0, 1>
// CHECK: wafer.group.yield
// CHECK: return %[[TCGROUP]] : tensor<8xf32>

// CHECK-LABEL: func.func @collective_with_static_insert_slice_producer
// CHECK: %[[GROUP:.+]] = wafer.group
// CHECK-SAME: ins(%{{.+}}, %{{.+}} : tensor<1x1x4x1xf32>, tensor<1x1x4x1xf32>)
// CHECK-SAME: outs(%{{.+}} : tensor<1x1x4x6xf32>)
// CHECK: arith.constant dense<0.000000e+00> : tensor<1x1x4x6xf32>
// CHECK: tensor.insert_slice
// CHECK: tensor.insert_slice
// CHECK: wafer.linalg_ext.collective.all_reduce
// CHECK: rank_group = array<i64: 0, 1>
// CHECK: wafer.group.yield
// CHECK: return %[[GROUP]] : tensor<1x1x4x6xf32>
