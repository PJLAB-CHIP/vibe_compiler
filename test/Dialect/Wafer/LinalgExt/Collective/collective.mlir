// RUN: wafer-opt %s | FileCheck %s

func.func @all_gather(
    %input: tensor<2x4xf32>) -> tensor<8x4xf32> {
  %out = tensor.empty() : tensor<8x4xf32>
  %0 = wafer.linalg_ext.collective.all_gather
      ins(%input : tensor<2x4xf32>)
      outs(%out : tensor<8x4xf32>)
      {axis = 0 : i64, partition_group = array<i64: 0, 1, 2, 3>}
      -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// CHECK-LABEL: func.func @all_gather
// CHECK: wafer.linalg_ext.collective.all_gather
// CHECK-SAME: axis = 0 : i64
// CHECK-SAME: partition_group = array<i64: 0, 1, 2, 3>
// CHECK-NOT: wafer.instr.dte_send

func.func @all_reduce(
    %input: tensor<4xf32>) -> tensor<4xf32> {
  %out = tensor.empty() : tensor<4xf32>
  %0 = wafer.linalg_ext.collective.all_reduce
      ins(%input : tensor<4xf32>)
      outs(%out : tensor<4xf32>)
      {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      wafer.linalg_ext.collective.yield %sum : f32
    } {partition_group = array<i64: 0, 1>} -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @all_reduce
// CHECK: wafer.linalg_ext.collective.all_reduce
// CHECK: wafer.linalg_ext.collective.yield
// CHECK-NOT: wafer.instr.dte_send

func.func @all_reduce_partition_groups(
    %input: tensor<4xf32>) -> tensor<4xf32> {
  %out = tensor.empty() : tensor<4xf32>
  %0 = wafer.linalg_ext.collective.all_reduce
      ins(%input : tensor<4xf32>)
      outs(%out : tensor<4xf32>)
      {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      wafer.linalg_ext.collective.yield %sum : f32
    } {partition_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}
      -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @all_reduce_partition_groups
// CHECK: wafer.linalg_ext.collective.all_reduce
// CHECK: partition_groups = dense<{{\[\[}}0, 1], [2, 3]]> : tensor<2x2xi64>

func.func @all_reduce_promoted(
    %input: tensor<4xf16>) -> tensor<4xf32> {
  %out = tensor.empty() : tensor<4xf32>
  %0 = wafer.linalg_ext.collective.all_reduce
      ins(%input : tensor<4xf16>)
      outs(%out : tensor<4xf32>)
      {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      wafer.linalg_ext.collective.yield %sum : f32
    } {partition_group = array<i64: 0, 1>} -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @all_reduce_promoted
// CHECK: wafer.linalg_ext.collective.all_reduce
// CHECK-SAME: ins(%{{.*}} : tensor<4xf16>)
// CHECK-SAME: outs(%{{.*}} : tensor<4xf32>)
// CHECK: ^bb0(%{{.*}}: f32, %{{.*}}: f32):

func.func @reduce_scatter(
    %input: tensor<8x4xf32>) -> tensor<2x4xf32> {
  %out = tensor.empty() : tensor<2x4xf32>
  %0 = wafer.linalg_ext.collective.reduce_scatter
      ins(%input : tensor<8x4xf32>)
      outs(%out : tensor<2x4xf32>)
      {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      wafer.linalg_ext.collective.yield %sum : f32
    } {axis = 0 : i64, partition_group = array<i64: 0, 1, 2, 3>} -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func.func @reduce_scatter
// CHECK: wafer.linalg_ext.collective.reduce_scatter

func.func @reduce_scatter_promoted(
    %input: tensor<8x4xbf16>) -> tensor<2x4xf32> {
  %out = tensor.empty() : tensor<2x4xf32>
  %0 = wafer.linalg_ext.collective.reduce_scatter
      ins(%input : tensor<8x4xbf16>)
      outs(%out : tensor<2x4xf32>)
      {
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      wafer.linalg_ext.collective.yield %sum : f32
    } {axis = 0 : i64, partition_group = array<i64: 0, 1, 2, 3>}
      -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func.func @reduce_scatter_promoted
// CHECK: wafer.linalg_ext.collective.reduce_scatter
// CHECK-SAME: ins(%{{.*}} : tensor<8x4xbf16>)
// CHECK-SAME: outs(%{{.*}} : tensor<2x4xf32>)
// CHECK: ^bb0(%{{.*}}: f32, %{{.*}}: f32):

func.func @all_to_all(
    %input: tensor<8x4xf32>) -> tensor<4x8xf32> {
  %out = tensor.empty() : tensor<4x8xf32>
  %0 = wafer.linalg_ext.collective.all_to_all
      ins(%input : tensor<8x4xf32>)
      outs(%out : tensor<4x8xf32>)
      {split_axis = 0 : i64, concat_axis = 1 : i64,
       split_count = 2 : i64,
       partition_group = array<i64: 0, 1>}
      -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @all_to_all
// CHECK: wafer.linalg_ext.collective.all_to_all

func.func @collective_permute(
    %input: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %out = tensor.empty() : tensor<2x4xf32>
  %0 = wafer.linalg_ext.collective.collective_permute
      ins(%input : tensor<2x4xf32>)
      outs(%out : tensor<2x4xf32>)
      {source_target_pairs = array<i64: 0, 1, 1, 0>}
      -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK-LABEL: func.func @collective_permute
// CHECK: wafer.linalg_ext.collective.collective_permute
