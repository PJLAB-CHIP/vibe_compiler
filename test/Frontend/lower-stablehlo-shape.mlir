// REQUIRES: stablehlo
// RUN: wafer-opt --wafer-lower-stablehlo-shape %s | FileCheck %s

module {
  func.func @lower_broadcast(%arg0: tensor<4xf32>) -> tensor<2x4xf32> {
    %0 = "stablehlo.broadcast_in_dim"(%arg0) {
      broadcast_dimensions = array<i64: 1>
    } : (tensor<4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }

  func.func @lower_expand_reshape(%arg0: tensor<6xf32>) -> tensor<2x3xf32> {
    %0 = "stablehlo.reshape"(%arg0) : (tensor<6xf32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }

  func.func @lower_collapse_reshape(%arg0: tensor<2x3xf32>) -> tensor<6xf32> {
    %0 = "stablehlo.reshape"(%arg0) : (tensor<2x3xf32>) -> tensor<6xf32>
    return %0 : tensor<6xf32>
  }

  func.func @lower_transpose(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
    %0 = "stablehlo.transpose"(%arg0) {
      permutation = array<i64: 1, 0>
    } : (tensor<2x3xf32>) -> tensor<3x2xf32>
    return %0 : tensor<3x2xf32>
  }

  func.func @lower_slice(%arg0: tensor<3x4xi32>) -> tensor<1x2xi32> {
    %0 = "stablehlo.slice"(%arg0) {
      start_indices = array<i64: 1, 0>,
      limit_indices = array<i64: 2, 4>,
      strides = array<i64: 1, 2>
    } : (tensor<3x4xi32>) -> tensor<1x2xi32>
    return %0 : tensor<1x2xi32>
  }
}

// CHECK-LABEL: func.func @lower_broadcast
// CHECK-NOT: stablehlo.broadcast_in_dim
// CHECK: tensor.empty() : tensor<2x4xf32>
// CHECK: linalg.broadcast
// CHECK-SAME: dimensions = [0]

// CHECK-LABEL: func.func @lower_expand_reshape
// CHECK-NOT: stablehlo.reshape
// CHECK: tensor.expand_shape
// CHECK-SAME: {{\[\[}}0, 1{{\]\]}}

// CHECK-LABEL: func.func @lower_collapse_reshape
// CHECK-NOT: stablehlo.reshape
// CHECK: tensor.collapse_shape
// CHECK-SAME: {{\[\[}}0, 1{{\]\]}}

// CHECK-LABEL: func.func @lower_transpose
// CHECK-NOT: stablehlo.transpose
// CHECK: tensor.empty() : tensor<3x2xf32>
// CHECK: linalg.transpose
// CHECK-SAME: permutation = [1, 0]

// CHECK-LABEL: func.func @lower_slice
// CHECK-NOT: stablehlo.slice
// CHECK: tensor.extract_slice
// CHECK-SAME: [1, 0] [1, 2] [1, 2]
