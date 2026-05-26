// REQUIRES: stablehlo
// RUN: wafer-opt --wafer-lower-stablehlo-reduce --wafer-normalize-constants --wafer-lower-stablehlo-elementwise --wafer-lower-stablehlo-shape %s | FileCheck %s

module {
  func.func @softmax_staged(%scores: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %neg_inf = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %row_max = "stablehlo.reduce"(%scores, %neg_inf) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %max = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %max : tensor<f32>
    }) {
      dimensions = array<i64: 1>
    } : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    %row_max_bcast = "stablehlo.broadcast_in_dim"(%row_max) {
      broadcast_dimensions = array<i64: 0>
    } : (tensor<2xf32>) -> tensor<2x4xf32>
    %shifted = stablehlo.subtract %scores, %row_max_bcast : tensor<2x4xf32>
    %exp_scores = stablehlo.exponential %shifted : tensor<2x4xf32>
    %row_sum = "stablehlo.reduce"(%exp_scores, %zero) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }) {
      dimensions = array<i64: 1>
    } : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    %row_sum_bcast = "stablehlo.broadcast_in_dim"(%row_sum) {
      broadcast_dimensions = array<i64: 0>
    } : (tensor<2xf32>) -> tensor<2x4xf32>
    %prob = stablehlo.divide %exp_scores, %row_sum_bcast : tensor<2x4xf32>
    return %prob : tensor<2x4xf32>
  }
}

// CHECK-DAG: #[[ROW_MAP:map[0-9]*]] = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG: #[[IDENTITY_MAP:map[0-9]*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @softmax_staged
// CHECK-NOT: stablehlo.
// CHECK: linalg.reduce
// CHECK-SAME: arith.maximumf
// CHECK-SAME: dimensions = [1]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[IDENTITY_MAP]], #[[ROW_MAP]], #[[IDENTITY_MAP]]]
// CHECK: arith.subf
// CHECK: linalg.generic
// CHECK: math.exp
// CHECK: linalg.reduce
// CHECK-SAME: arith.addf
// CHECK-SAME: dimensions = [1]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[IDENTITY_MAP]], #[[ROW_MAP]], #[[IDENTITY_MAP]]]
// CHECK: arith.divf
