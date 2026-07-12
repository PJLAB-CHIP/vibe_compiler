// RUN: wafer-opt --wafer-dump-group-to-tile-region='logical-rank=0' %s 2>&1 | FileCheck %s

func.func @maxnum_nan_semantics(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>,
                                %out: tensor<4xf32>) -> tensor<4xf32> {
  %group = wafer.group ins(%lhs, %rhs : tensor<4xf32>, tensor<4xf32>)
      outs(%out : tensor<4xf32>) {
  ^bb0(%lhs_arg: tensor<4xf32>, %rhs_arg: tensor<4xf32>,
       %out_arg: tensor<4xf32>):
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>
        ],
        iterator_types = ["parallel"]
      } ins(%lhs_arg, %rhs_arg : tensor<4xf32>, tensor<4xf32>)
        outs(%out_arg : tensor<4xf32>) {
      ^bb0(%left: f32, %right: f32, %init: f32):
        %max = arith.maxnumf %left, %right : f32
        linalg.yield %max : f32
      } -> tensor<4xf32>
    wafer.group.yield %result : tensor<4xf32>
  } : tensor<4xf32>
  return %group : tensor<4xf32>
}

func.func @unsigned_division(%lhs: tensor<4xi32>, %rhs: tensor<4xi32>,
                             %out: tensor<4xi32>) -> tensor<4xi32> {
  %group = wafer.group ins(%lhs, %rhs : tensor<4xi32>, tensor<4xi32>)
      outs(%out : tensor<4xi32>) {
  ^bb0(%lhs_arg: tensor<4xi32>, %rhs_arg: tensor<4xi32>,
       %out_arg: tensor<4xi32>):
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>
        ],
        iterator_types = ["parallel"]
      } ins(%lhs_arg, %rhs_arg : tensor<4xi32>, tensor<4xi32>)
        outs(%out_arg : tensor<4xi32>) {
      ^bb0(%left: i32, %right: i32, %init: i32):
        %div = arith.divui %left, %right : i32
        linalg.yield %div : i32
      } -> tensor<4xi32>
    wafer.group.yield %result : tensor<4xi32>
  } : tensor<4xi32>
  return %group : tensor<4xi32>
}

func.func @unsigned_compare(%lhs: tensor<4xi32>, %rhs: tensor<4xi32>,
                            %out: tensor<4xi1>) -> tensor<4xi1> {
  %group = wafer.group ins(%lhs, %rhs : tensor<4xi32>, tensor<4xi32>)
      outs(%out : tensor<4xi1>) {
  ^bb0(%lhs_arg: tensor<4xi32>, %rhs_arg: tensor<4xi32>,
       %out_arg: tensor<4xi1>):
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>
        ],
        iterator_types = ["parallel"]
      } ins(%lhs_arg, %rhs_arg : tensor<4xi32>, tensor<4xi32>)
        outs(%out_arg : tensor<4xi1>) {
      ^bb0(%left: i32, %right: i32, %init: i1):
        %less = arith.cmpi ult, %left, %right : i32
        linalg.yield %less : i1
      } -> tensor<4xi1>
    wafer.group.yield %result : tensor<4xi1>
  } : tensor<4xi1>
  return %group : tensor<4xi1>
}

func.func @unordered_float_compare(%lhs: tensor<4xf32>,
                                   %rhs: tensor<4xf32>,
                                   %out: tensor<4xi1>) -> tensor<4xi1> {
  %group = wafer.group ins(%lhs, %rhs : tensor<4xf32>, tensor<4xf32>)
      outs(%out : tensor<4xi1>) {
  ^bb0(%lhs_arg: tensor<4xf32>, %rhs_arg: tensor<4xf32>,
       %out_arg: tensor<4xi1>):
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>,
          affine_map<(d0) -> (d0)>
        ],
        iterator_types = ["parallel"]
      } ins(%lhs_arg, %rhs_arg : tensor<4xf32>, tensor<4xf32>)
        outs(%out_arg : tensor<4xi1>) {
      ^bb0(%left: f32, %right: f32, %init: i1):
        %less = arith.cmpf ult, %left, %right : f32
        linalg.yield %less : i1
      } -> tensor<4xi1>
    wafer.group.yield %result : tensor<4xi1>
  } : tensor<4xi1>
  return %group : tensor<4xi1>
}

// CHECK: wafer.group_to_tile_region group @maxnum_nan_semantics#0
// CHECK-NEXT: failure arith.maxnumf NaN semantics cannot be represented by the current target elementwise kind
// CHECK: wafer.group_to_tile_region group @unsigned_division#0
// CHECK-NEXT: failure unsigned integer division cannot be represented by the current target elementwise kind
// CHECK: wafer.group_to_tile_region group @unsigned_compare#0
// CHECK-NEXT: failure unsigned arith.cmpi predicate cannot be represented by the current signed target comparison kind
// CHECK: wafer.group_to_tile_region group @unordered_float_compare#0
// CHECK-NEXT: failure arith.cmpf predicate does not match the current target comparison NaN semantics
