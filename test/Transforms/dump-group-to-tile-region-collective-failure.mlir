// RUN: wafer-opt --wafer-dump-group-to-tile-region='logical-rank=0' %s 2>&1 | FileCheck %s

func.func @collective_ignores_accumulator(%input: tensor<4xf32>,
                                          %out: tensor<4xf32>)
    -> tensor<4xf32> {
  %group = wafer.group ins(%input : tensor<4xf32>)
      outs(%out : tensor<4xf32>) {
  ^bb0(%input_arg: tensor<4xf32>, %out_arg: tensor<4xf32>):
    %result = wafer.linalg_ext.collective.all_reduce
        ins(%input_arg : tensor<4xf32>)
        outs(%out_arg : tensor<4xf32>)
        {
        ^bb0(%value: f32, %acc: f32):
          %bad = arith.addf %value, %value : f32
          wafer.linalg_ext.collective.yield %bad : f32
        } {rank_group = array<i64: 0, 1>} -> tensor<4xf32>
    wafer.group.yield %result : tensor<4xf32>
  } : tensor<4xf32>
  return %group : tensor<4xf32>
}

// CHECK: wafer.group_to_tile_region group @collective_ignores_accumulator#0
// CHECK-NEXT: failure collective reduction materialization requires one exact combiner wired to the reduced value and accumulator
