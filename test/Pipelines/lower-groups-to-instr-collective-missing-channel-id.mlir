// RUN: not wafer-opt --pass-pipeline='builtin.module(wafer-lower-groups-to-instr)' %s 2>&1 | FileCheck %s

func.func @missing_channel_id(%input: tensor<4xf32>, %out: tensor<8xf32>)
    -> tensor<8xf32> {
  %group = wafer.group ins(%input : tensor<4xf32>)
      outs(%out : tensor<8xf32>) {
  ^bb0(%arg0: tensor<4xf32>, %arg1: tensor<8xf32>):
    %ag = wafer.linalg_ext.collective.all_gather
        ins(%arg0 : tensor<4xf32>)
        outs(%arg1 : tensor<8xf32>)
        {axis = 0 : i64, rank_group = array<i64: 0, 1>}
        -> tensor<8xf32>
    wafer.group.yield %ag : tensor<8xf32>
  } : tensor<8xf32>
  return %group : tensor<8xf32>
}

// CHECK: all_gather materialization requires channel_id for stable DTE identity
