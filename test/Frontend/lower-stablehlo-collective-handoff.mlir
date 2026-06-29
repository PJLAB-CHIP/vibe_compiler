// REQUIRES: stablehlo
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-stablehlo-to-linalg)' %s | FileCheck %s --check-prefix=IR

module {
  func.func @partitioned_all_gather(%input: tensor<4xf32>) -> tensor<8xf32> {
    %0 = "stablehlo.all_gather"(%input) {
      all_gather_dim = 0 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>
    } : (tensor<4xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }

  func.func @partitioned_all_reduce(%input: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "stablehlo.all_reduce"(%input) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      "stablehlo.return"(%sum) : (tensor<f32>) -> ()
    }) {
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>
    } : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  func.func @partitioned_all_reduce_rank_groups(%input: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "stablehlo.all_reduce"(%input) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      "stablehlo.return"(%sum) : (tensor<f32>) -> ()
    }) {
      replica_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 6, type = 1>,
      use_global_device_ids
    } : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  func.func @partitioned_reduce_scatter(%input: tensor<8x4xf32>) -> tensor<4x4xf32> {
    %0 = "stablehlo.reduce_scatter"(%input) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      "stablehlo.return"(%sum) : (tensor<f32>) -> ()
    }) {
      scatter_dimension = 0 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 3, type = 1>
    } : (tensor<8x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  func.func @partitioned_all_to_all(%input: tensor<8x4xf32>) -> tensor<4x8xf32> {
    %0 = "stablehlo.all_to_all"(%input) {
      split_dimension = 0 : i64,
      concat_dimension = 1 : i64,
      split_count = 2 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 4, type = 1>
    } : (tensor<8x4xf32>) -> tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
  }

  func.func @partitioned_collective_permute(%input: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "stablehlo.collective_permute"(%input) {
      source_target_pairs = dense<[[0, 1], [1, 0]]> : tensor<2x2xi64>,
      channel_handle = #stablehlo.channel_handle<handle = 5, type = 1>
    } : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// IR-LABEL: func.func @partitioned_all_gather
// IR: tensor.empty() : tensor<8xf32>
// IR: wafer.linalg_ext.collective.all_gather
// IR-SAME: axis = 0 : i64
// IR-SAME: channel_id = 1 : i64
// IR-SAME: rank_group = array<i64: 0, 1>
// IR-NOT: stablehlo.all_gather
// IR-NOT: wafer.instr.dte_send

// IR-LABEL: func.func @partitioned_all_reduce
// IR: tensor.empty() : tensor<4xf32>
// IR: wafer.linalg_ext.collective.all_reduce
// IR: ^bb0(%{{.*}}: f32, %{{.*}}: f32):
// IR: arith.addf
// IR: wafer.linalg_ext.collective.yield
// IR: channel_id = 2 : i64
// IR-SAME: rank_group = array<i64: 0, 1>
// IR-NOT: stablehlo.all_reduce
// IR-NOT: wafer.instr.dte_send

// IR-LABEL: func.func @partitioned_all_reduce_rank_groups
// IR: wafer.linalg_ext.collective.all_reduce
// IR: channel_id = 6 : i64
// IR-SAME: rank_groups = dense<{{\[\[}}0, 1], [2, 3]]> : tensor<2x2xi64>
// IR-SAME: use_global_device_ids = true
// IR-NOT: stablehlo.all_reduce

// IR-LABEL: func.func @partitioned_reduce_scatter
// IR: tensor.empty() : tensor<4x4xf32>
// IR: wafer.linalg_ext.collective.reduce_scatter
// IR: channel_id = 3 : i64
// IR-SAME: rank_group = array<i64: 0, 1>
// IR-NOT: stablehlo.reduce_scatter
// IR-NOT: wafer.instr.dte_send

// IR-LABEL: func.func @partitioned_all_to_all
// IR: tensor.empty() : tensor<4x8xf32>
// IR: wafer.linalg_ext.collective.all_to_all
// IR-SAME: channel_id = 4 : i64
// IR-SAME: split_count = 2 : i64
// IR-NOT: stablehlo.all_to_all
// IR-NOT: wafer.instr.dte_send

// IR-LABEL: func.func @partitioned_collective_permute
// IR: tensor.empty() : tensor<4xf32>
// IR: wafer.linalg_ext.collective.collective_permute
// IR-SAME: channel_id = 5 : i64
// IR-SAME: source_target_pairs = array<i64: 0, 1, 1, 0>
// IR-NOT: stablehlo.collective_permute
// IR-NOT: wafer.instr.dte_send
