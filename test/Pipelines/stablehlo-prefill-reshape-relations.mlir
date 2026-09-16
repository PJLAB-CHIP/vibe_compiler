// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-stablehlo-to-linalg)' %s -o %t.once
// RUN: wafer-opt --pass-pipeline='builtin.module(func.func(wafer-normalize-structured-tensor-graph))' %t.once -o %t.twice
// RUN: FileCheck %s < %t.once
// RUN: FileCheck %s < %t.twice
// RUN: sed 's/4096/4097/g' %s | wafer-opt --pass-pipeline='builtin.module(wafer-lower-stablehlo-to-linalg)' | FileCheck %s --check-prefix=TAIL

// A real prefill source graph grows the relation store while reparameterizing
// reshapes through mixed-width compute. Keep both contractions, reductions,
// arithmetic and result shape; the old callback read freed relation storage.
// CHECK: #{{.*}} = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)>
// CHECK-LABEL: func.func @main
// CHECK: linalg.batch_matmul
// CHECK: arith.extf
// CHECK: arith.mulf
// CHECK: arith.truncf
// CHECK: arith.addf
// CHECK: arith.extf
// CHECK: arith.maximumf
// CHECK: arith.subf
// CHECK: math.exp
// CHECK: arith.addf
// CHECK: arith.divf
// CHECK: arith.truncf
// CHECK: linalg.batch_matmul
// CHECK: tensor.expand_shape
// CHECK: return {{.*}} : tensor<1x32x4096x128xf16>
// TAIL-LABEL: func.func @main
// TAIL: linalg.batch_matmul
// TAIL: arith.maximumf
// TAIL: math.exp
// TAIL: arith.divf
// TAIL: linalg.batch_matmul
// TAIL: return {{.*}} : tensor<1x32x4097x128xf16>

module {
  func.func @main(%arg0: tensor<1x32x4096x128xf16>, %arg1: tensor<1x1x4096x4096xf16>, %arg2: tensor<f32>, %arg3: tensor<1x32x4096x128xf16>, %arg4: tensor<1x32x4096x128xf16>) -> tensor<1x32x4096x128xf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %0 = stablehlo.reshape %arg4 : (tensor<1x32x4096x128xf16>) -> tensor<32x4096x128xf16>
    %1 = stablehlo.transpose %arg3, dims = [0, 1, 3, 2] : (tensor<1x32x4096x128xf16>) -> tensor<1x32x128x4096xf16>
    %2 = stablehlo.reshape %1 : (tensor<1x32x128x4096xf16>) -> tensor<32x128x4096xf16>
    %3 = stablehlo.dot_general %0, %2, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x4096x128xf16>, tensor<32x128x4096xf16>) -> tensor<32x4096x4096xf16>
    %4 = stablehlo.reshape %3 : (tensor<32x4096x4096xf16>) -> tensor<1x32x4096x4096xf16>
    %5 = stablehlo.convert %4 : (tensor<1x32x4096x4096xf16>) -> tensor<1x32x4096x4096xf32>
    %6 = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<1x32x4096x4096xf32>
    %7 = stablehlo.multiply %5, %6 : tensor<1x32x4096x4096xf32>
    %8 = stablehlo.convert %7 : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x4096xf16>
    %9 = stablehlo.reshape %arg1 : (tensor<1x1x4096x4096xf16>) -> tensor<1x4096x4096xf16>
    %10 = stablehlo.broadcast_in_dim %9, dims = [0, 2, 3] : (tensor<1x4096x4096xf16>) -> tensor<1x32x4096x4096xf16>
    %11 = stablehlo.add %8, %10 : tensor<1x32x4096x4096xf16>
    %12 = stablehlo.convert %11 : (tensor<1x32x4096x4096xf16>) -> tensor<1x32x4096x4096xf32>
    %13 = stablehlo.reduce(%12 init: %cst_0) applies stablehlo.maximum across dimensions = [3] : (tensor<1x32x4096x4096xf32>, tensor<f32>) -> tensor<1x32x4096xf32>
    %14 = stablehlo.broadcast_in_dim %13, dims = [0, 1, 2] : (tensor<1x32x4096xf32>) -> tensor<1x32x4096x4096xf32>
    %15 = stablehlo.subtract %12, %14 : tensor<1x32x4096x4096xf32>
    %16 = stablehlo.exponential %15 : tensor<1x32x4096x4096xf32>
    %17 = stablehlo.reduce(%16 init: %cst) applies stablehlo.add across dimensions = [3] : (tensor<1x32x4096x4096xf32>, tensor<f32>) -> tensor<1x32x4096xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1, 2] : (tensor<1x32x4096xf32>) -> tensor<1x32x4096x4096xf32>
    %19 = stablehlo.divide %16, %18 : tensor<1x32x4096x4096xf32>
    %20 = stablehlo.convert %19 : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x4096xf16>
    %21 = stablehlo.reshape %20 : (tensor<1x32x4096x4096xf16>) -> tensor<32x4096x4096xf16>
    %22 = stablehlo.reshape %arg0 : (tensor<1x32x4096x128xf16>) -> tensor<32x4096x128xf16>
    %23 = stablehlo.dot_general %21, %22, batching_dims = [0] x [0], contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x4096x4096xf16>, tensor<32x4096x128xf16>) -> tensor<32x4096x128xf16>
    %24 = stablehlo.reshape %23 : (tensor<32x4096x128xf16>) -> tensor<1x32x4096x128xf16>
    return %24 : tensor<1x32x4096x128xf16>
  }
}
