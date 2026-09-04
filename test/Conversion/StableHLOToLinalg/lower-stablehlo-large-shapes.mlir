// REQUIRES: stablehlo
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-stablehlo-to-linalg)' %s | FileCheck %s

// These are structural conversion witnesses at production-scale extents. The
// file intentionally keeps the bodies simple so failures identify the
// StableHLO-to-Linalg boundary rather than a later physical-dataflow stage.
module {
  func.func @large_softmax(%scores: tensor<1x1024x1025xf32>)
      -> tensor<1x1024x1025xf32> {
    %neg_inf = stablehlo.constant dense<-3.40282347E+38> : tensor<f32>
    %zero = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %row_max = "stablehlo.reduce"(%scores, %neg_inf) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %max = stablehlo.maximum %lhs, %rhs : tensor<f32>
      stablehlo.return %max : tensor<f32>
    }) {
      dimensions = array<i64: 2>
    } : (tensor<1x1024x1025xf32>, tensor<f32>) -> tensor<1x1024xf32>
    %row_max_bcast = "stablehlo.broadcast_in_dim"(%row_max) {
      broadcast_dimensions = array<i64: 0, 1>
    } : (tensor<1x1024xf32>) -> tensor<1x1024x1025xf32>
    %shifted = stablehlo.subtract %scores, %row_max_bcast
        : tensor<1x1024x1025xf32>
    %exp_scores = stablehlo.exponential %shifted : tensor<1x1024x1025xf32>
    %row_sum = "stablehlo.reduce"(%exp_scores, %zero) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }) {
      dimensions = array<i64: 2>
    } : (tensor<1x1024x1025xf32>, tensor<f32>) -> tensor<1x1024xf32>
    %row_sum_bcast = "stablehlo.broadcast_in_dim"(%row_sum) {
      broadcast_dimensions = array<i64: 0, 1>
    } : (tensor<1x1024xf32>) -> tensor<1x1024x1025xf32>
    %prob = stablehlo.divide %exp_scores, %row_sum_bcast
        : tensor<1x1024x1025xf32>
    return %prob : tensor<1x1024x1025xf32>
  }

  func.func @large_reduce(%arg0: tensor<1x1024x1025xf32>,
                           %init: tensor<f32>) -> tensor<1x1024xf32> {
    %result = "stablehlo.reduce"(%arg0, %init) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }) {
      dimensions = array<i64: 2>
    } : (tensor<1x1024x1025xf32>, tensor<f32>) -> tensor<1x1024xf32>
    return %result : tensor<1x1024xf32>
  }

  func.func @large_dot(%lhs: tensor<1x1024x1025xf16>,
                        %rhs: tensor<1x1025x1024xf16>)
      -> tensor<1x1024x1024xf16> {
    %result = "stablehlo.dot_general"(%lhs, %rhs) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_batching_dimensions = [0], rhs_batching_dimensions = [0],
        lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
      precision_config = [#stablehlo<precision DEFAULT>,
                          #stablehlo<precision DEFAULT>]
    } : (tensor<1x1024x1025xf16>, tensor<1x1025x1024xf16>)
        -> tensor<1x1024x1024xf16>
    return %result : tensor<1x1024x1024xf16>
  }

  func.func @large_attention(
      %query: tensor<1x1024x64xf16>, %key: tensor<1x1025x64xf16>,
      %value: tensor<1x1025x128xf16>, %mask: tensor<1x1024x1025xf16>)
      -> tensor<1x1024x128xf16> {
    %zero = stablehlo.constant dense<0.0> : tensor<f16>
    %lowest = stablehlo.constant dense<-6.5504E+4> : tensor<f16>
    %scale = stablehlo.constant dense<0.125> : tensor<f16>
    %scale_broadcast = stablehlo.broadcast_in_dim %scale, dims = []
        : (tensor<f16>) -> tensor<1x1024x1025xf16>
    %scores = "stablehlo.dot_general"(%query, %key) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_batching_dimensions = [0], rhs_batching_dimensions = [0],
        lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [2]>,
      precision_config = [#stablehlo<precision DEFAULT>,
                          #stablehlo<precision DEFAULT>]
    } : (tensor<1x1024x64xf16>, tensor<1x1025x64xf16>)
        -> tensor<1x1024x1025xf16>
    %scaled = stablehlo.multiply %scores, %scale_broadcast
        : tensor<1x1024x1025xf16>
    %masked = stablehlo.add %scaled, %mask : tensor<1x1024x1025xf16>
    %row_max = "stablehlo.reduce"(%masked, %lowest) ({
    ^bb0(%lhs: tensor<f16>, %rhs: tensor<f16>):
      %maximum = stablehlo.maximum %lhs, %rhs : tensor<f16>
      stablehlo.return %maximum : tensor<f16>
    }) {dimensions = array<i64: 2>}
        : (tensor<1x1024x1025xf16>, tensor<f16>) -> tensor<1x1024xf16>
    %max_broadcast = "stablehlo.broadcast_in_dim"(%row_max) {
      broadcast_dimensions = array<i64: 0, 1>
    } : (tensor<1x1024xf16>) -> tensor<1x1024x1025xf16>
    %shifted = stablehlo.subtract %masked, %max_broadcast
        : tensor<1x1024x1025xf16>
    %exponential = stablehlo.exponential %shifted
        : tensor<1x1024x1025xf16>
    %row_sum = "stablehlo.reduce"(%exponential, %zero) ({
    ^bb0(%lhs: tensor<f16>, %rhs: tensor<f16>):
      %sum = stablehlo.add %lhs, %rhs : tensor<f16>
      stablehlo.return %sum : tensor<f16>
    }) {dimensions = array<i64: 2>}
        : (tensor<1x1024x1025xf16>, tensor<f16>) -> tensor<1x1024xf16>
    %sum_broadcast = "stablehlo.broadcast_in_dim"(%row_sum) {
      broadcast_dimensions = array<i64: 0, 1>
    } : (tensor<1x1024xf16>) -> tensor<1x1024x1025xf16>
    %probability = stablehlo.divide %exponential, %sum_broadcast
        : tensor<1x1024x1025xf16>
    %result = "stablehlo.dot_general"(%probability, %value) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_batching_dimensions = [0], rhs_batching_dimensions = [0],
        lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>,
      precision_config = [#stablehlo<precision DEFAULT>,
                          #stablehlo<precision DEFAULT>]
    } : (tensor<1x1024x1025xf16>, tensor<1x1025x128xf16>)
        -> tensor<1x1024x128xf16>
    return %result : tensor<1x1024x128xf16>
  }
}

// CHECK-LABEL: func.func @large_softmax
// CHECK-NOT: stablehlo.
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK: math.exp
// CHECK: arith.divf

// CHECK-LABEL: func.func @large_reduce
// CHECK-NOT: stablehlo.
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK: arith.addf

// CHECK-LABEL: func.func @large_dot
// CHECK-NOT: stablehlo.
// CHECK: linalg.batch_matmul
// CHECK-SAME: tensor<1x1024x1024xf16>

// CHECK-LABEL: func.func @large_attention
// CHECK-NOT: stablehlo.
// CHECK: wafer.linalg_ext.attention
// CHECK-SAME: algorithm(<flash_attention>)
// CHECK: tensor<1x1024x128xf16>
