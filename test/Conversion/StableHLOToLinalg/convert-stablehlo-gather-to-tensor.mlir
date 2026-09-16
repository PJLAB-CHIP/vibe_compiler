// RUN: wafer-opt %s --wafer-convert-stablehlo-gather-to-tensor --split-input-file --verify-each | FileCheck %s
// RUN: wafer-opt %s --wafer-lower-stablehlo-to-linalg --split-input-file --verify-each | FileCheck %s --check-prefix=PIPELINE

// CHECK-LABEL: func.func @rows
// CHECK: linalg.generic
// CHECK-SAME: ins(%arg1 : tensor<2x1024xi64>)
// CHECK: arith.constant 0 : i64
// CHECK: arith.constant 2047 : i64
// CHECK: arith.maxsi
// CHECK: arith.minsi
// CHECK: tensor.expand_shape
// CHECK: tensor.gather %arg0[%{{.*}}] gather_dims([0])
// CHECK-NOT: unique
// CHECK: return
// PIPELINE-LABEL: func.func @rows
// PIPELINE: tensor.gather
// PIPELINE-NOT: stablehlo.gather
// PIPELINE: return
func.func @rows(%table: tensor<2048x64xf16>, %ids: tensor<2x1024xi64>) -> tensor<2x1024x64xf16> {
  %result = "stablehlo.gather"(%table, %ids) {
    dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 64>
  } : (tensor<2048x64xf16>, tensor<2x1024xi64>) -> tensor<2x1024x64xf16>
  return %result : tensor<2x1024x64xf16>
}

// -----

// CHECK-LABEL: func.func @rows_tail
// CHECK: linalg.generic
// CHECK-SAME: ins(%arg1 : tensor<2x1025x1xi32>)
// CHECK: arith.constant 2047 : i32
// CHECK: arith.maxsi
// CHECK: arith.minsi
// CHECK-NOT: tensor.expand_shape
// CHECK: tensor.gather %arg0[%{{.*}}] gather_dims([0])
// CHECK-NOT: unique
// CHECK: return
// PIPELINE-LABEL: func.func @rows_tail
// PIPELINE: tensor.gather
// PIPELINE: return
func.func @rows_tail(%table: tensor<2048x64xbf16>, %ids: tensor<2x1025x1xi32>) -> tensor<2x1025x64xbf16> {
  %result = "stablehlo.gather"(%table, %ids) {
    dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 64>
  } : (tensor<2048x64xbf16>, tensor<2x1025x1xi32>) -> tensor<2x1025x64xbf16>
  return %result : tensor<2x1025x64xbf16>
}

// -----

// CHECK-LABEL: func.func @partial_window
// CHECK: stablehlo.gather
// CHECK-NOT: tensor.gather
// CHECK: return
// PIPELINE-LABEL: func.func @partial_window
// PIPELINE: tensor.extract
// PIPELINE-NOT: stablehlo.gather
// PIPELINE: return
func.func @partial_window(%table: tensor<2048x64xf16>, %ids: tensor<2x1031xi64>) -> tensor<2x1031x32xf16> {
  %result = "stablehlo.gather"(%table, %ids) {
    dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>,
    indices_are_sorted = false,
    slice_sizes = array<i64: 1, 32>
  } : (tensor<2048x64xf16>, tensor<2x1031xi64>) -> tensor<2x1031x32xf16>
  return %result : tensor<2x1031x32xf16>
}
