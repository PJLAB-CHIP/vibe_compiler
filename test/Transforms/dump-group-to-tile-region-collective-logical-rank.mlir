// RUN: wafer-opt --wafer-dump-group-to-tile-region='logical-rank=1' %s 2>&1 | FileCheck %s

module {
  func.func @all_gather_rank_one(%input: tensor<4xf32>, %out: tensor<8xf32>)
      -> tensor<8xf32> {
    %0 = wafer.group ins(%input : tensor<4xf32>) outs(%out : tensor<8xf32>) {
    ^bb0(%arg0: tensor<4xf32>, %arg1: tensor<8xf32>):
      %ag = wafer.linalg_ext.collective.all_gather
          ins(%arg0 : tensor<4xf32>)
          outs(%arg1 : tensor<8xf32>)
          {axis = 0 : i64, rank_group = array<i64: 0, 1>}
          -> tensor<8xf32>
      wafer.group.yield %ag : tensor<8xf32>
    } : tensor<8xf32>
    return %0 : tensor<8xf32>
  }

  func.func @reduce_scatter_rank_one(%input: tensor<8xf32>,
                                     %out: tensor<4xf32>)
      -> tensor<4xf32> {
    %0 = wafer.group ins(%input : tensor<8xf32>) outs(%out : tensor<4xf32>) {
    ^bb0(%arg0: tensor<8xf32>, %arg1: tensor<4xf32>):
      %rs = wafer.linalg_ext.collective.reduce_scatter
          ins(%arg0 : tensor<8xf32>)
          outs(%arg1 : tensor<4xf32>)
          {
          ^bb0(%lhs: f32, %rhs: f32):
            %sum = arith.addf %lhs, %rhs : f32
            wafer.linalg_ext.collective.yield %sum : f32
          } {axis = 0 : i64, rank_group = array<i64: 0, 1>} -> tensor<4xf32>
      wafer.group.yield %rs : tensor<4xf32>
    } : tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  func.func @all_reduce_rank_groups_rank_one(%input: tensor<4xf32>,
                                             %out: tensor<4xf32>)
      -> tensor<4xf32> {
    %0 = wafer.group ins(%input : tensor<4xf32>) outs(%out : tensor<4xf32>) {
    ^bb0(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>):
      %ar = wafer.linalg_ext.collective.all_reduce
          ins(%arg0 : tensor<4xf32>)
          outs(%arg1 : tensor<4xf32>)
          {
          ^bb0(%lhs: f32, %rhs: f32):
            %sum = arith.addf %lhs, %rhs : f32
            wafer.linalg_ext.collective.yield %sum : f32
          } {rank_groups = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}
          -> tensor<4xf32>
      wafer.group.yield %ar : tensor<4xf32>
    } : tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  func.func @all_to_all_rank_one(%input: tensor<4x4xf32>,
                                 %out: tensor<2x8xf32>)
      -> tensor<2x8xf32> {
    %0 = wafer.group ins(%input : tensor<4x4xf32>)
        outs(%out : tensor<2x8xf32>) {
    ^bb0(%arg0: tensor<4x4xf32>, %arg1: tensor<2x8xf32>):
      %a2a = wafer.linalg_ext.collective.all_to_all
          ins(%arg0 : tensor<4x4xf32>)
          outs(%arg1 : tensor<2x8xf32>)
          {split_axis = 0 : i64, concat_axis = 1 : i64,
           split_count = 2 : i64, rank_group = array<i64: 0, 1>}
          -> tensor<2x8xf32>
      wafer.group.yield %a2a : tensor<2x8xf32>
    } : tensor<2x8xf32>
    return %0 : tensor<2x8xf32>
  }
}

// CHECK-LABEL: wafer.group_to_tile_region group @all_gather_rank_one#0
// CHECK: wafer.tile.all_gather
// CHECK-SAME: group_size = 2 : i64
// CHECK-SAME: local_rank = 1 : i64
// CHECK-SAME: rank_group = array<i64: 0, 1>
// CHECK-LABEL: wafer.group_to_tile_region group @reduce_scatter_rank_one#0
// CHECK: wafer.tile.reduce_scatter <sum>
// CHECK-SAME: axis = 0 : i64
// CHECK-SAME: group_size = 2 : i64
// CHECK-SAME: local_rank = 1 : i64
// CHECK-SAME: rank_group = array<i64: 0, 1>
// CHECK-LABEL: wafer.group_to_tile_region group @all_reduce_rank_groups_rank_one#0
// CHECK: wafer.tile.all_reduce <sum>
// CHECK-SAME: group_size = 2 : i64
// CHECK-SAME: local_rank = 1 : i64
// CHECK-SAME: rank_group = array<i64: 0, 1>
// CHECK-LABEL: wafer.group_to_tile_region group @all_to_all_rank_one#0
// CHECK: wafer.instr.dte_send
// CHECK-SAME: peer = 0 : i64
// CHECK: wafer.instr.dte_recv
// CHECK-SAME: peer = 0 : i64
// CHECK: wafer.instr.dte_wait
