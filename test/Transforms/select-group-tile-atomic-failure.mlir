// RUN: not wafer-opt --wafer-select-group-tile='logical-rank=0' --mlir-print-ir-after-failure %s 2>&1 | FileCheck --implicit-check-not=wafer.tile.region --implicit-check-not=wafer.instr --check-prefixes=DIAG,IR %s

func.func @first_group_must_remain_source(
    %lhs: tensor<4xf32>, %rhs: tensor<4xf32>, %out: tensor<4xf32>)
    -> tensor<4xf32> {
  %group = wafer.group
      ins(%lhs, %rhs : tensor<4xf32>, tensor<4xf32>)
      outs(%out : tensor<4xf32>) {
  ^bb0(%lhs_arg: tensor<4xf32>, %rhs_arg: tensor<4xf32>,
       %out_arg: tensor<4xf32>):
    %sum = linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>
        ],
        iterator_types = ["parallel"]
      } ins(%lhs_arg, %rhs_arg : tensor<4xf32>, tensor<4xf32>)
        outs(%out_arg : tensor<4xf32>) {
    ^bb0(%left: f32, %right: f32, %old: f32):
      %value = arith.addf %left, %right : f32
      linalg.yield %value : f32
    } -> tensor<4xf32>
    wafer.group.yield %sum : tensor<4xf32>
  } : tensor<4xf32>
  return %group : tensor<4xf32>
}

func.func @second_group_fails_late_evaluation(
    %lhs: tensor<?xf32>, %rhs: tensor<?xf32>, %out: tensor<?xf32>)
    -> tensor<?xf32> {
  %group = wafer.group
      ins(%lhs, %rhs : tensor<?xf32>, tensor<?xf32>)
      outs(%out : tensor<?xf32>) {
  ^bb0(%lhs_arg: tensor<?xf32>, %rhs_arg: tensor<?xf32>,
       %out_arg: tensor<?xf32>):
    %sum = linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>
        ],
        iterator_types = ["parallel"]
      } ins(%lhs_arg, %rhs_arg : tensor<?xf32>, tensor<?xf32>)
        outs(%out_arg : tensor<?xf32>) {
    ^bb0(%left: f32, %right: f32, %old: f32):
      %value = arith.addf %left, %right : f32
      linalg.yield %value : f32
    } -> tensor<?xf32>
    wafer.group.yield %sum : tensor<?xf32>
  } : tensor<?xf32>
  return %group : tensor<?xf32>
}

// DIAG: no_candidate: tile selection requires static ranked group results with one traversal shape

// IR: IR Dump After SelectGroupTilePass Failed
// IR-LABEL: func.func @first_group_must_remain_source
// IR: wafer.group
// IR: linalg.generic
// IR: wafer.group.yield
// IR: return
// IR-LABEL: func.func @second_group_fails_late_evaluation
// IR: wafer.group
// IR: linalg.generic
// IR: wafer.group.yield
// IR: return
