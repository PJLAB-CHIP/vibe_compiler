// REQUIRES: stablehlo
// RUN: wafer-opt --wafer-lower-stablehlo-collectives-to-comm='local-rank=0' --wafer-lower-ring-all-gather --wafer-lower-ring-reduce-collectives --wafer-lower-to-c-abi-skeleton %s | FileCheck %s

module {
  wafer.placement.map
      {bad_tile_ids = array<i64>,
       card_x_count = 1 : i64,
       card_y_count = 1 : i64,
       logical_rank_count = 2 : i64,
       physical_tile_coords = array<i64: 0, 0, 0, 0,
                                      0, 0, 0, 1>,
       tile_x_count = 4 : i64,
       tile_y_count = 4 : i64}

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
      %min = stablehlo.minimum %lhs, %rhs : tensor<f32>
      "stablehlo.return"(%min) : (tensor<f32>) -> ()
    }) {
      scatter_dimension = 0 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<8xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// CHECK-LABEL: module
// CHECK-NOT: stablehlo.
// CHECK-NOT: wafer.comm.
// CHECK-NOT: wafer.compute.elementwise
// CHECK: func.func @all_gather
// CHECK: wafer.abi.dte_send
// CHECK: func.func @all_reduce
// CHECK: wafer.abi.elementwise <issue_only> <add>
// CHECK: func.func @reduce_scatter
// CHECK: wafer.abi.elementwise <issue_only> <min>
