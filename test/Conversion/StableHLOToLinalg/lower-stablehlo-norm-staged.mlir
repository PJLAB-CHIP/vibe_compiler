// REQUIRES: stablehlo
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-stablehlo-to-linalg)' %s | FileCheck %s

module {
  func.func @rmsnorm_staged(%x: tensor<2x4xf32>,
                            %weight: tensor<4xf32>) -> tensor<2x4xf32> {
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %hidden = stablehlo.constant dense<4.000000e+00> : tensor<2xf32>
    %eps = stablehlo.constant dense<1.000000e-05> : tensor<2xf32>
    %square = stablehlo.multiply %x, %x : tensor<2x4xf32>
    %sum = "stablehlo.reduce"(%square, %zero) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %add = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %add : tensor<f32>
    }) {
      dimensions = array<i64: 1>
    } : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    %mean = stablehlo.divide %sum, %hidden : tensor<2xf32>
    %mean_eps = stablehlo.add %mean, %eps : tensor<2xf32>
    %rms = stablehlo.rsqrt %mean_eps : tensor<2xf32>
    %rms_bcast = "stablehlo.broadcast_in_dim"(%rms) {
      broadcast_dimensions = array<i64: 0>
    } : (tensor<2xf32>) -> tensor<2x4xf32>
    %normed = stablehlo.multiply %x, %rms_bcast : tensor<2x4xf32>
    %weight_bcast = "stablehlo.broadcast_in_dim"(%weight) {
      broadcast_dimensions = array<i64: 1>
    } : (tensor<4xf32>) -> tensor<2x4xf32>
    %scaled = stablehlo.multiply %normed, %weight_bcast : tensor<2x4xf32>
    return %scaled : tensor<2x4xf32>
  }

  func.func @layernorm_staged(%x: tensor<2x4xf32>,
                              %weight: tensor<4xf32>,
                              %bias: tensor<4xf32>) -> tensor<2x4xf32> {
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %hidden = stablehlo.constant dense<4.000000e+00> : tensor<2xf32>
    %eps = stablehlo.constant dense<1.000000e-05> : tensor<2xf32>
    %sum = "stablehlo.reduce"(%x, %zero) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %add = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %add : tensor<f32>
    }) {
      dimensions = array<i64: 1>
    } : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    %mean = stablehlo.divide %sum, %hidden : tensor<2xf32>
    %mean_bcast = "stablehlo.broadcast_in_dim"(%mean) {
      broadcast_dimensions = array<i64: 0>
    } : (tensor<2xf32>) -> tensor<2x4xf32>
    %centered = stablehlo.subtract %x, %mean_bcast : tensor<2x4xf32>
    %square = stablehlo.multiply %centered, %centered : tensor<2x4xf32>
    %var_sum = "stablehlo.reduce"(%square, %zero) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %add = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %add : tensor<f32>
    }) {
      dimensions = array<i64: 1>
    } : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    %var = stablehlo.divide %var_sum, %hidden : tensor<2xf32>
    %var_eps = stablehlo.add %var, %eps : tensor<2xf32>
    %inv = stablehlo.rsqrt %var_eps : tensor<2xf32>
    %inv_bcast = "stablehlo.broadcast_in_dim"(%inv) {
      broadcast_dimensions = array<i64: 0>
    } : (tensor<2xf32>) -> tensor<2x4xf32>
    %normed = stablehlo.multiply %centered, %inv_bcast : tensor<2x4xf32>
    %weight_bcast = "stablehlo.broadcast_in_dim"(%weight) {
      broadcast_dimensions = array<i64: 1>
    } : (tensor<4xf32>) -> tensor<2x4xf32>
    %scaled = stablehlo.multiply %normed, %weight_bcast : tensor<2x4xf32>
    %bias_bcast = "stablehlo.broadcast_in_dim"(%bias) {
      broadcast_dimensions = array<i64: 1>
    } : (tensor<4xf32>) -> tensor<2x4xf32>
    %shifted = stablehlo.add %scaled, %bias_bcast : tensor<2x4xf32>
    return %shifted : tensor<2x4xf32>
  }
}

// CHECK-LABEL: func.func @rmsnorm_staged
// CHECK-NOT: stablehlo.
// CHECK: arith.mulf
// CHECK: linalg.generic
// CHECK: iterator_types = ["parallel", "reduction"]
// CHECK: arith.addf
// CHECK: arith.divf
// CHECK: arith.addf
// CHECK: math.rsqrt
// CHECK: linalg.generic
// CHECK: arith.mulf
// CHECK: linalg.generic
// CHECK: arith.mulf

// CHECK-LABEL: func.func @layernorm_staged
// CHECK-NOT: stablehlo.
// CHECK: linalg.generic
// CHECK: iterator_types = ["parallel", "reduction"]
// CHECK: arith.addf
// CHECK: arith.subf
// CHECK: linalg.generic
// CHECK: iterator_types = ["parallel", "reduction"]
// CHECK: arith.addf
// CHECK: math.rsqrt
// CHECK: linalg.generic
// CHECK: arith.addf
