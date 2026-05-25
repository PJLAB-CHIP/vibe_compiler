// REQUIRES: stablehlo
// RUN: wafer-opt --wafer-lower-stablehlo-reduce %s | FileCheck %s

module {
  func.func @lower_reduce_sum(%arg0: tensor<2x4xf32>,
                              %init: tensor<f32>) -> tensor<2xf32> {
    %0 = "stablehlo.reduce"(%arg0, %init) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %1 = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) {
      dimensions = array<i64: 1>
    } : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }

  func.func @lower_reduce_max(%arg0: tensor<2x4xf32>,
                              %init: tensor<f32>) -> tensor<4xf32> {
    %0 = "stablehlo.reduce"(%arg0, %init) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %1 = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) {
      dimensions = array<i64: 0>
    } : (tensor<2x4xf32>, tensor<f32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @lower_reduce_sum
// CHECK-NOT: stablehlo.reduce
// CHECK: tensor.extract %arg1[]
// CHECK: tensor.empty() : tensor<2xf32>
// CHECK: linalg.fill
// CHECK: linalg.reduce
// CHECK-SAME: arith.addf
// CHECK-SAME: dimensions = [1]

// CHECK-LABEL: func.func @lower_reduce_max
// CHECK-NOT: stablehlo.reduce
// CHECK: tensor.extract %arg1[]
// CHECK: tensor.empty() : tensor<4xf32>
// CHECK: linalg.fill
// CHECK: linalg.reduce
// CHECK-SAME: arith.maximumf
// CHECK-SAME: dimensions = [0]
