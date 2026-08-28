// REQUIRES: stablehlo
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-stablehlo-to-linalg)' %s | FileCheck %s

module {
  func.func @rope_staged(%x: tensor<2x4xf32>,
                         %cos: tensor<2x2xf32>,
                         %sin: tensor<2x2xf32>) -> tensor<2x4xf32> {
    %x0 = "stablehlo.slice"(%x) {
      start_indices = array<i64: 0, 0>,
      limit_indices = array<i64: 2, 2>,
      strides = array<i64: 1, 1>
    } : (tensor<2x4xf32>) -> tensor<2x2xf32>
    %x1 = "stablehlo.slice"(%x) {
      start_indices = array<i64: 0, 2>,
      limit_indices = array<i64: 2, 4>,
      strides = array<i64: 1, 1>
    } : (tensor<2x4xf32>) -> tensor<2x2xf32>
    %x0_cos = stablehlo.multiply %x0, %cos : tensor<2x2xf32>
    %x1_sin = stablehlo.multiply %x1, %sin : tensor<2x2xf32>
    %rot0 = stablehlo.subtract %x0_cos, %x1_sin : tensor<2x2xf32>
    %x0_sin = stablehlo.multiply %x0, %sin : tensor<2x2xf32>
    %x1_cos = stablehlo.multiply %x1, %cos : tensor<2x2xf32>
    %rot1 = stablehlo.add %x0_sin, %x1_cos : tensor<2x2xf32>
    %out = stablehlo.concatenate %rot0, %rot1, dim = 1
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x4xf32>
    return %out : tensor<2x4xf32>
  }

  func.func @gelu_tanh_staged(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %half = stablehlo.constant dense<5.000000e-01> : tensor<2x4xf32>
    %one = stablehlo.constant dense<1.000000e+00> : tensor<2x4xf32>
    %cube_coeff = stablehlo.constant dense<4.471500e-02> : tensor<2x4xf32>
    %sqrt_2_over_pi = stablehlo.constant dense<7.978846e-01> : tensor<2x4xf32>
    %x2 = stablehlo.multiply %x, %x : tensor<2x4xf32>
    %x3 = stablehlo.multiply %x2, %x : tensor<2x4xf32>
    %cubic = stablehlo.multiply %cube_coeff, %x3 : tensor<2x4xf32>
    %inner = stablehlo.add %x, %cubic : tensor<2x4xf32>
    %scaled = stablehlo.multiply %sqrt_2_over_pi, %inner : tensor<2x4xf32>
    %tanh = stablehlo.tanh %scaled : tensor<2x4xf32>
    %gate = stablehlo.add %one, %tanh : tensor<2x4xf32>
    %half_x = stablehlo.multiply %half, %x : tensor<2x4xf32>
    %gelu = stablehlo.multiply %half_x, %gate : tensor<2x4xf32>
    return %gelu : tensor<2x4xf32>
  }
}

// CHECK-LABEL: func.func @rope_staged
// CHECK-NOT: stablehlo.
// CHECK: tensor.extract_slice
// CHECK-SAME: [0, 0] [2, 2] [1, 1]
// CHECK: tensor.extract_slice
// CHECK-SAME: [0, 2] [2, 2] [1, 1]
// CHECK: arith.mulf
// CHECK: arith.subf
// CHECK: arith.addf
// CHECK: tensor.empty() : tensor<2x4xf32>
// CHECK: tensor.insert_slice {{.*}}[0, 0] [2, 2] [1, 1]
// CHECK: tensor.insert_slice {{.*}}[0, 2] [2, 2] [1, 1]
// CHECK-NOT: scf.if
// CHECK-NOT: linalg.index

// CHECK-LABEL: func.func @gelu_tanh_staged
// CHECK-NOT: stablehlo.
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: math.tanh
// CHECK: arith.mulf
