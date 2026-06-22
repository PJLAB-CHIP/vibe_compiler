// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-groups-to-placement{logical-rank-count=2 card-y=1 card-x=1 tile-y=1 tile-x=2})' %s | FileCheck --implicit-check-not=selected_group %s

func.func @pipeline_elementwise(%lhs: tensor<8xf32>, %rhs: tensor<8xf32>,
                                %out: tensor<8xf32>) -> tensor<8xf32> {
  %group = wafer.group ins(%lhs, %rhs : tensor<8xf32>, tensor<8xf32>)
      outs(%out : tensor<8xf32>) {
  ^bb0(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>):
    %sum = linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>
        ],
        iterator_types = ["parallel"]
      } ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
        outs(%arg2 : tensor<8xf32>) {
      ^bb0(%lhs_el: f32, %rhs_el: f32, %out_el: f32):
        %add = arith.addf %lhs_el, %rhs_el : f32
        linalg.yield %add : f32
      } -> tensor<8xf32>
    wafer.group.yield %sum : tensor<8xf32>
  } : tensor<8xf32>
  return %group : tensor<8xf32>
}

// CHECK: wafer.placement.map
// CHECK-SAME: block_ids = array<i64: 0, 1>
// CHECK-SAME: logical_rank_count = 2 : i64
// CHECK-SAME: physical_tile_coords = array<i64: 0, 0, 0, 0, 0, 0, 0, 1>
// CHECK-LABEL: func.func @pipeline_elementwise
// CHECK-NOT: wafer.group
// CHECK: wafer.tile.region
// CHECK: wafer.spm.offset
// CHECK: wafer.instr.elementwise <add>
