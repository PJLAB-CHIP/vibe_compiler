// RUN: wafer-opt %s | FileCheck %s

func.func @all_gather(
    %input: tensor<2x4xf32>,
    %out: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %0 = wafer.tensor_collective.all_gather
      ins(%input : tensor<2x4xf32>)
      outs(%out : tensor<8x4xf32>)
      {axis = 0 : i64, rank_group = array<i64: 0, 1, 2, 3>}
      -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// CHECK-LABEL: func.func @all_gather
// CHECK: wafer.tensor_collective.all_gather
// CHECK-SAME: axis = 0 : i64
// CHECK-SAME: rank_group = array<i64: 0, 1, 2, 3>
// CHECK-NOT: wafer.comm

func.func @all_reduce(
    %input: tensor<4xf32>,
    %out: tensor<4xf32>) -> tensor<4xf32> {
  %0 = wafer.tensor_collective.all_reduce
      ins(%input : tensor<4xf32>)
      outs(%out : tensor<4xf32>)
      {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      wafer.tensor_collective.yield %sum : f32
    } {rank_group = array<i64: 0, 1>} -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @all_reduce
// CHECK: wafer.tensor_collective.all_reduce
// CHECK: wafer.tensor_collective.yield
// CHECK-NOT: wafer.comm

func.func @reduce_scatter(
    %input: tensor<8x4xf32>,
    %out: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = wafer.tensor_collective.reduce_scatter
      ins(%input : tensor<8x4xf32>)
      outs(%out : tensor<2x4xf32>)
      {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      wafer.tensor_collective.yield %sum : f32
    } {axis = 0 : i64, rank_group = array<i64: 0, 1, 2, 3>} -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func.func @reduce_scatter
// CHECK: wafer.tensor_collective.reduce_scatter

func.func @all_to_all(
    %input: tensor<8x4xf32>,
    %out: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = wafer.tensor_collective.all_to_all
      ins(%input : tensor<8x4xf32>)
      outs(%out : tensor<4x8xf32>)
      {split_axis = 0 : i64, concat_axis = 1 : i64,
       split_count = 2 : i64,
       rank_group = array<i64: 0, 1>}
      -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @all_to_all
// CHECK: wafer.tensor_collective.all_to_all

func.func @collective_permute(
    %input: tensor<2x4xf32>,
    %out: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = wafer.tensor_collective.collective_permute
      ins(%input : tensor<2x4xf32>)
      outs(%out : tensor<2x4xf32>)
      {source_target_pairs = array<i64: 0, 1, 1, 0>}
      -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func.func @collective_permute
// CHECK: wafer.tensor_collective.collective_permute
