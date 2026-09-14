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
// CHECK: linalg.generic {{.*}} ins(%[[IDS0]] : tensor<2x1024x1xi32>)
// CHECK-NOT: tensor.extract %[[IDS0]]
// CHECK: arith.index_cast
// CHECK: arith.maxsi
// CHECK: arith.minsi
// CHECK: tensor.extract {{.*}} : tensor<4096x8xf16>
// CHECK: return {{.*}} : tensor<2x1024x8xf16>
// CHECK-LABEL: func.func @gather_1025
// CHECK: %[[IDS1:.*]] = linalg.generic
// CHECK: arith.trunci
// CHECK: linalg.generic {{.*}} ins(%[[IDS1]] : tensor<2x1025x1xi32>)
// CHECK-NOT: tensor.extract %[[IDS1]]
// CHECK: arith.index_cast
// CHECK: arith.maxsi
// CHECK: arith.minsi
// CHECK: tensor.extract {{.*}} : tensor<4096x8xbf16>
// CHECK: return {{.*}} : tensor<2x1025x8xbf16>
// CHECK-LABEL: func.func @gather_1031
// CHECK: %[[IDS2:.*]] = linalg.generic
// CHECK: arith.trunci
// CHECK: linalg.generic {{.*}} ins(%[[IDS2]] : tensor<2x1031x1xi32>)
// CHECK-NOT: tensor.extract %[[IDS2]]
// CHECK: arith.index_cast
// CHECK: arith.maxsi
// CHECK: arith.minsi
// CHECK: tensor.extract {{.*}} : tensor<4096x8xf16>
// CHECK: return {{.*}} : tensor<2x1031x8xf16>
