// REQUIRES: stablehlo
// RUN: not wafer-opt --pass-pipeline='builtin.module(wafer-lower-stablehlo-to-linalg)' %s 2>&1 | FileCheck %s

module {
  func.func @unsupported_replica_group(%input: tensor<4xf32>) -> tensor<8xf32> {
    %0 = "stablehlo.all_gather"(%input) {
      all_gather_dim = 0 : i64,
      replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 7, type = 1>
    } : (tensor<4xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }
}

// CHECK: failed to normalize StableHLO collective to wafer.tensor handoff
