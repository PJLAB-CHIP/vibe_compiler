// RUN: wafer-opt --wafer-dump-group-to-tile-region='logical-rank=0' %s 2>&1 | FileCheck %s

func.func @canonical_concat(%first: tensor<2x2xf32>,
                            %second: tensor<2x2xf32>,
                            %out: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %group = wafer.group
      ins(%first, %second : tensor<2x2xf32>, tensor<2x2xf32>)
      outs(%out : tensor<2x4xf32>) {
  ^bb0(%first_arg: tensor<2x2xf32>, %second_arg: tensor<2x2xf32>,
       %out_arg: tensor<2x4xf32>):
    %c2 = arith.constant 2 : index
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } outs(%out_arg : tensor<2x4xf32>) {
      ^bb0(%init: f32):
        %row = linalg.index 0 : index
        %column = linalg.index 1 : index
        %in_first = arith.cmpi ult, %column, %c2 : index
        %value = scf.if %in_first -> f32 {
          %first_value = tensor.extract %first_arg[%row, %column]
              : tensor<2x2xf32>
          scf.yield %first_value : f32
        } else {
          %second_column = arith.subi %column, %c2 : index
          %second_value = tensor.extract %second_arg[%row, %second_column]
              : tensor<2x2xf32>
          scf.yield %second_value : f32
        }
        linalg.yield %value : f32
      } -> tensor<2x4xf32>
    wafer.group.yield %result : tensor<2x4xf32>
  } : tensor<2x4xf32>
  return %group : tensor<2x4xf32>
}

func.func @arbitrary_flag_is_not_concat(%first: tensor<2x2xf32>,
                                        %second: tensor<2x2xf32>,
                                        %out: tensor<2x4xf32>, %flag: i1)
    -> tensor<2x4xf32> {
  %group = wafer.group
      ins(%first, %second, %flag
          : tensor<2x2xf32>, tensor<2x2xf32>, i1)
      outs(%out : tensor<2x4xf32>) {
  ^bb0(%first_arg: tensor<2x2xf32>, %second_arg: tensor<2x2xf32>,
       %flag_arg: i1, %out_arg: tensor<2x4xf32>):
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      } outs(%out_arg : tensor<2x4xf32>) {
      ^bb0(%init: f32):
        %row = linalg.index 0 : index
        %column = linalg.index 1 : index
        %value = scf.if %flag_arg -> f32 {
          %first_value = tensor.extract %first_arg[%row, %column]
              : tensor<2x2xf32>
          scf.yield %first_value : f32
        } else {
          %second_value = tensor.extract %second_arg[%row, %column]
              : tensor<2x2xf32>
          scf.yield %second_value : f32
        }
        linalg.yield %value : f32
      } -> tensor<2x4xf32>
    wafer.group.yield %result : tensor<2x4xf32>
  } : tensor<2x4xf32>
  return %group : tensor<2x4xf32>
}

// CHECK: wafer.group_to_tile_region group @canonical_concat#0
// CHECK: wafer.tile.insert_slice
// CHECK-SAME: offsets = array<i64: 0, 0>
// CHECK: wafer.tile.insert_slice
// CHECK-SAME: offsets = array<i64: 0, 2>
// CHECK: wafer.tile.store
// CHECK: wafer.group_to_tile_region group @arbitrary_flag_is_not_concat#0
// CHECK-NEXT: failure unsupported linalg.generic body op linalg.index
