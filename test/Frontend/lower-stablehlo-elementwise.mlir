// REQUIRES: stablehlo
// RUN: wafer-opt --wafer-normalize-constants --wafer-lower-stablehlo-elementwise %s | FileCheck %s

module {
  func.func @lower_binary(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>)
      -> tensor<4xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
    %1 = stablehlo.subtract %0, %arg1 : tensor<4xf32>
    %2 = stablehlo.multiply %1, %arg1 : tensor<4xf32>
    %3 = stablehlo.divide %2, %arg1 : tensor<4xf32>
    %4 = stablehlo.maximum %3, %arg1 : tensor<4xf32>
    %5 = stablehlo.minimum %4, %arg1 : tensor<4xf32>
    return %5 : tensor<4xf32>
  }

  func.func @lower_unary(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.negate %arg0 : tensor<4xf32>
    %1 = stablehlo.sqrt %0 : tensor<4xf32>
    %2 = stablehlo.rsqrt %1 : tensor<4xf32>
    %3 = stablehlo.exponential %2 : tensor<4xf32>
    return %3 : tensor<4xf32>
  }

  func.func @lower_reciprocal(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<4xf32>
    %0 = stablehlo.divide %cst, %arg0 : tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  func.func @lower_broadcasted_add(%arg0: tensor<4xf32>,
                                   %arg1: tensor<2x4xf32>)
      -> tensor<2x4xf32> {
    %0 = "stablehlo.broadcast_in_dim"(%arg0) {
      broadcast_dimensions = array<i64: 1>
    } : (tensor<4xf32>) -> tensor<2x4xf32>
    %1 = stablehlo.add %0, %arg1 : tensor<2x4xf32>
    return %1 : tensor<2x4xf32>
  }
}

// CHECK-DAG: #[[BCAST_MAP:map[0-9]*]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG: #[[IDENTITY_MAP:map[0-9]*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @lower_binary
// CHECK-NOT: stablehlo.
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<add>
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<sub>
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<mul>
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<div>
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<max_signed>
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<min_signed>

// CHECK-LABEL: func.func @lower_unary
// CHECK-NOT: stablehlo.
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<negf>
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<sqrt>
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<rsqrt>
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<exp>

// CHECK-LABEL: func.func @lower_reciprocal
// CHECK-NOT: stablehlo.
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<reciprocal>

// CHECK-LABEL: func.func @lower_broadcasted_add
// CHECK-NOT: stablehlo.
// CHECK: linalg.elementwise
// CHECK-SAME: kind=#linalg.elementwise_kind<add>
// CHECK-SAME: indexing_maps = [#[[BCAST_MAP]], #[[IDENTITY_MAP]], #[[IDENTITY_MAP]]]
