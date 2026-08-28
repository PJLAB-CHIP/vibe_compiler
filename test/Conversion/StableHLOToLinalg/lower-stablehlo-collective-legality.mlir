// REQUIRES: stablehlo
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-stablehlo-to-linalg)' %s | FileCheck %s

module {
  func.func @multi_replica_group(%input: tensor<4xf32>) -> tensor<8xf32> {
    %0 = "stablehlo.all_gather"(%input) {
      all_gather_dim = 0 : i64,
      replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 7, type = 1>
    } : (tensor<4xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}

// CHECK: wafer.linalg_ext.collective.all_gather
// CHECK-SAME: axis = 0 : i64
// CHECK-SAME: channel_id = 7 : i64
// CHECK-SAME: partition_groups = dense<{{.*}}> : tensor<2x2xi64>
