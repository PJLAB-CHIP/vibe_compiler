// REQUIRES: stablehlo
// RUN: wafer-opt --wafer-lower-stablehlo-collectives-to-comm='local-rank=1' %s | FileCheck %s

module {
  func.func @all_gather(%arg0: tensor<4xf32>) -> tensor<8xf32> {
    %0 = "stablehlo.all_gather"(%arg0) {
      all_gather_dim = 0 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<4xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }

  func.func @all_reduce(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "stablehlo.all_reduce"(%arg0) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      "stablehlo.return"(%sum) : (tensor<f32>) -> ()
    }) {
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  func.func @reduce_scatter(%arg0: tensor<8xf32>) -> tensor<4xf32> {
    %0 = "stablehlo.reduce_scatter"(%arg0) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %max = stablehlo.maximum %lhs, %rhs : tensor<f32>
      "stablehlo.return"(%max) : (tensor<f32>) -> ()
    }) {
      scatter_dimension = 0 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<8xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @all_gather
// CHECK-NOT: stablehlo.all_gather
// CHECK: wafer.comm.all_gather %{{.*}} into %{{.*}} {bytes = 16 : i64, group_size = 2 : i64, local_rank = 1 : i64, rank_group = array<i64: 0, 1>}

// CHECK-LABEL: func.func @all_reduce
// CHECK-NOT: stablehlo.all_reduce
// CHECK: wafer.comm.all_reduce <sum> %{{.*}} using %{{.*}} {bytes = 16 : i64, group_size = 2 : i64, local_rank = 1 : i64, rank_group = array<i64: 0, 1>}

// CHECK-LABEL: func.func @reduce_scatter
// CHECK-NOT: stablehlo.reduce_scatter
// CHECK: wafer.comm.reduce_scatter <max> %{{.*}} using %{{.*}} {bytes = 16 : i64, group_size = 2 : i64, local_rank = 1 : i64, rank_group = array<i64: 0, 1>}
