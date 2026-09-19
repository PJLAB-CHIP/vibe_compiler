// REQUIRES: stablehlo
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-stablehlo-to-linalg)' %s | FileCheck %s

// The index producer must remain an explicit dependency after official gather
// legalization. The data-dependent table read and StableHLO clamping remain.
// This is source-to-Linalg coverage, not a claim of target gather support.
module {
  func.func @gather_1024(%table: tensor<4096x8xf16>, %ids: tensor<2x1024x1xi64>)
      -> tensor<2x1024x8xf16> {
    %indices = stablehlo.convert %ids : (tensor<2x1024x1xi64>) -> tensor<2x1024x1xi32>
    %result = "stablehlo.gather"(%table, %indices) {
      dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0],
          start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 8>
    } : (tensor<4096x8xf16>, tensor<2x1024x1xi32>) -> tensor<2x1024x8xf16>
    return %result : tensor<2x1024x8xf16>
  }
  func.func @gather_1025(%table: tensor<4096x8xbf16>, %ids: tensor<2x1025x1xi64>)
      -> tensor<2x1025x8xbf16> {
    %indices = stablehlo.convert %ids : (tensor<2x1025x1xi64>) -> tensor<2x1025x1xi32>
    %result = "stablehlo.gather"(%table, %indices) {
      dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0],
          start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 8>
    } : (tensor<4096x8xbf16>, tensor<2x1025x1xi32>) -> tensor<2x1025x8xbf16>
    return %result : tensor<2x1025x8xbf16>
  }
  func.func @gather_1031(%table: tensor<4096x8xf16>, %ids: tensor<2x1031x1xi64>)
      -> tensor<2x1031x8xf16> {
    %indices = stablehlo.convert %ids : (tensor<2x1031x1xi64>) -> tensor<2x1031x1xi32>
    %result = "stablehlo.gather"(%table, %indices) {
      dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0],
          start_index_map = [0], index_vector_dim = 2>, slice_sizes = array<i64: 1, 8>
    } : (tensor<4096x8xf16>, tensor<2x1031x1xi32>) -> tensor<2x1031x8xf16>
    return %result : tensor<2x1031x8xf16>
  }
}

// CHECK-LABEL: func.func @gather_1024
// CHECK: %[[IDS0:.*]] = linalg.generic
// CHECK: arith.trunci
// CHECK: %[[CLAMP0:.*]] = linalg.generic {{.*}} ins(%[[IDS0]] : tensor<2x1024x1xi32>)
// CHECK: %[[LOW0:.*]] = arith.maxsi {{.*}}, %c0_i32 : i32
// CHECK-NEXT: %[[HIGH0:.*]] = arith.minsi %[[LOW0]], %c4095_i32 : i32
// CHECK-NEXT: linalg.yield %[[HIGH0]] : i32
// CHECK: %[[RESULT0:.*]] = tensor.gather %arg0[%[[CLAMP0]]] gather_dims([0]) : (tensor<4096x8xf16>, tensor<2x1024x1xi32>) -> tensor<2x1024x8xf16>
// CHECK-NEXT: return %[[RESULT0]] : tensor<2x1024x8xf16>
// CHECK-LABEL: func.func @gather_1025
// CHECK: %[[IDS1:.*]] = linalg.generic
// CHECK: arith.trunci
// CHECK: %[[CLAMP1:.*]] = linalg.generic {{.*}} ins(%[[IDS1]] : tensor<2x1025x1xi32>)
// CHECK: %[[LOW1:.*]] = arith.maxsi {{.*}}, %c0_i32 : i32
// CHECK-NEXT: %[[HIGH1:.*]] = arith.minsi %[[LOW1]], %c4095_i32 : i32
// CHECK-NEXT: linalg.yield %[[HIGH1]] : i32
// CHECK: %[[RESULT1:.*]] = tensor.gather %arg0[%[[CLAMP1]]] gather_dims([0]) : (tensor<4096x8xbf16>, tensor<2x1025x1xi32>) -> tensor<2x1025x8xbf16>
// CHECK-NEXT: return %[[RESULT1]] : tensor<2x1025x8xbf16>
// CHECK-LABEL: func.func @gather_1031
// CHECK: %[[IDS2:.*]] = linalg.generic
// CHECK: arith.trunci
// CHECK: %[[CLAMP2:.*]] = linalg.generic {{.*}} ins(%[[IDS2]] : tensor<2x1031x1xi32>)
// CHECK: %[[LOW2:.*]] = arith.maxsi {{.*}}, %c0_i32 : i32
// CHECK-NEXT: %[[HIGH2:.*]] = arith.minsi %[[LOW2]], %c4095_i32 : i32
// CHECK-NEXT: linalg.yield %[[HIGH2]] : i32
// CHECK: %[[RESULT2:.*]] = tensor.gather %arg0[%[[CLAMP2]]] gather_dims([0]) : (tensor<4096x8xf16>, tensor<2x1031x1xi32>) -> tensor<2x1031x8xf16>
// CHECK-NEXT: return %[[RESULT2]] : tensor<2x1031x8xf16>
