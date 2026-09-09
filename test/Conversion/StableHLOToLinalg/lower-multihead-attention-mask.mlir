// REQUIRES: stablehlo
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-stablehlo-to-linalg)' %s | FileCheck %s
// RUN: sed 's/1024/1025/g' %s | wafer-opt --pass-pipeline='builtin.module(wafer-lower-stablehlo-to-linalg)' | FileCheck %s
// RUN: sed 's/1024/1031/g; s/f16/bf16/g' %s | wafer-opt --pass-pipeline='builtin.module(wafer-lower-stablehlo-to-linalg)' | FileCheck %s
// Product-derived HF prefill, with two heads sharing one causal mask.
// The broad mask and scale must not survive as score-sized materializations.
// CHECK-DAG: #[[MASK_MAP:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d4)>
// CHECK-LABEL: func.func @main
// CHECK-NOT: linalg.generic
// CHECK-NOT: tensor.extract
// CHECK: %[[MASK:.*]] = tensor.collapse_shape %arg1
// CHECK: %[[SCALE:.*]] = arith.constant 0.0883883461 : f32
// CHECK-NOT: linalg.generic
// CHECK: wafer.linalg_ext.attention ins(%arg3, %arg2, %arg0, %[[SCALE]], %[[MASK]]
// CHECK-SAME: indexing_maps = [{{.*}}, #[[MASK_MAP]], {{.*}}]
// CHECK-NOT: linalg.generic
// CHECK: return

module @multihead_attention {
  func.func @main(%arg0: tensor<1x2x1024x128xf16>, %arg1: tensor<1x1x1024x1024xf16>, %arg2: tensor<1x2x1024x128xf16>, %arg3: tensor<1x2x1024x128xf16>) -> tensor<1x2x1024x128xf16> {
    %cst = stablehlo.constant dense<0.0883883461> : tensor<1x2x1024x1024xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.reshape %arg3 : (tensor<1x2x1024x128xf16>) -> tensor<2x1024x128xf16>
    %1 = stablehlo.transpose %arg2, dims = [0, 1, 3, 2] : (tensor<1x2x1024x128xf16>) -> tensor<1x2x128x1024xf16>
    %2 = stablehlo.reshape %1 : (tensor<1x2x128x1024xf16>) -> tensor<2x128x1024xf16>
    %3 = stablehlo.dot_general %0, %2, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x1024x128xf16>, tensor<2x128x1024xf16>) -> tensor<2x1024x1024xf16>
    %4 = stablehlo.reshape %3 : (tensor<2x1024x1024xf16>) -> tensor<1x2x1024x1024xf16>
    %5 = stablehlo.convert %4 : (tensor<1x2x1024x1024xf16>) -> tensor<1x2x1024x1024xf32>
    %6 = stablehlo.multiply %5, %cst : tensor<1x2x1024x1024xf32>
    %7 = stablehlo.convert %6 : (tensor<1x2x1024x1024xf32>) -> tensor<1x2x1024x1024xf16>
    %8 = stablehlo.reshape %arg1 : (tensor<1x1x1024x1024xf16>) -> tensor<1x1024x1024xf16>
    %9 = stablehlo.broadcast_in_dim %8, dims = [0, 2, 3] : (tensor<1x1024x1024xf16>) -> tensor<1x2x1024x1024xf16>
    %10 = stablehlo.add %7, %9 : tensor<1x2x1024x1024xf16>
    %11 = stablehlo.convert %10 : (tensor<1x2x1024x1024xf16>) -> tensor<1x2x1024x1024xf32>
    %12 = stablehlo.reduce(%11 init: %cst_1) applies stablehlo.maximum across dimensions = [3] : (tensor<1x2x1024x1024xf32>, tensor<f32>) -> tensor<1x2x1024xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1, 2] : (tensor<1x2x1024xf32>) -> tensor<1x2x1024x1024xf32>
    %14 = stablehlo.subtract %11, %13 : tensor<1x2x1024x1024xf32>
    %15 = stablehlo.exponential %14 : tensor<1x2x1024x1024xf32>
    %16 = stablehlo.reduce(%15 init: %cst_0) applies stablehlo.add across dimensions = [3] : (tensor<1x2x1024x1024xf32>, tensor<f32>) -> tensor<1x2x1024xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1, 2] : (tensor<1x2x1024xf32>) -> tensor<1x2x1024x1024xf32>
    %18 = stablehlo.divide %15, %17 : tensor<1x2x1024x1024xf32>
    %19 = stablehlo.convert %18 : (tensor<1x2x1024x1024xf32>) -> tensor<1x2x1024x1024xf16>
    %20 = stablehlo.reshape %19 : (tensor<1x2x1024x1024xf16>) -> tensor<2x1024x1024xf16>
    %21 = stablehlo.reshape %arg0 : (tensor<1x2x1024x128xf16>) -> tensor<2x1024x128xf16>
    %22 = stablehlo.dot_general %20, %21, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x1024x1024xf16>, tensor<2x1024x128xf16>) -> tensor<2x1024x128xf16>
    %23 = stablehlo.reshape %22 : (tensor<2x1024x128xf16>) -> tensor<1x2x1024x128xf16>
    return %23 : tensor<1x2x1024x128xf16>
  }
}
