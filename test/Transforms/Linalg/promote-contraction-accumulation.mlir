// RUN: wafer-opt --wafer-promote-contraction-accumulation %s | FileCheck %s

module {
  func.func @shared_nonzero_init(%a: tensor<2x16x1025xbf16>, %b: tensor<2x1025x32xbf16>) -> (tensor<2x16x32xbf16>, tensor<2x16x32xbf16>) {
    %c = arith.constant 1.5 : bf16
    %empty = tensor.empty() : tensor<2x16x32xbf16>
    %init = linalg.fill ins(%c : bf16) outs(%empty : tensor<2x16x32xbf16>) -> tensor<2x16x32xbf16>
    %r = linalg.batch_matmul ins(%a, %b : tensor<2x16x1025xbf16>, tensor<2x1025x32xbf16>) outs(%init : tensor<2x16x32xbf16>) -> tensor<2x16x32xbf16>
    return %r, %init : tensor<2x16x32xbf16>, tensor<2x16x32xbf16>
  }

  func.func @already_wide(%a: tensor<2x16x1024xf16>, %b: tensor<2x1024x32xf16>, %init: tensor<2x16x32xf32>) -> tensor<2x16x32xf32> {
    %r = linalg.batch_matmul ins(%a, %b : tensor<2x16x1024xf16>, tensor<2x1024x32xf16>) outs(%init : tensor<2x16x32xf32>) -> tensor<2x16x32xf32>
    return %r : tensor<2x16x32xf32>
  }

  func.func @different_payload(%a: tensor<2x16x1031xf16>, %b: tensor<2x1031x32xf16>, %init: tensor<2x16x32xf16>) -> tensor<2x16x32xf16> {
    %r = linalg.generic {indexing_maps = [affine_map<(b,m,n,k)->(b,m,k)>, affine_map<(b,m,n,k)->(b,k,n)>, affine_map<(b,m,n,k)->(b,m,n)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%a, %b : tensor<2x16x1031xf16>, tensor<2x1031x32xf16>) outs(%init : tensor<2x16x32xf16>) {
    ^bb0(%x: f16, %y: f16, %old: f16):
      %p = arith.mulf %x, %y : f16
      %s = arith.subf %old, %p : f16
      linalg.yield %s : f16
    } -> tensor<2x16x32xf16>
    return %r : tensor<2x16x32xf16>
  }
}

// CHECK-LABEL: func.func @shared_nonzero_init
// CHECK: %[[NARROW:.*]] = linalg.fill {{.*}} -> tensor<2x16x32xbf16>
// CHECK: arith.constant 1.500000e+00 : f32
// CHECK: %[[INIT:.*]] = linalg.fill {{.*}} -> tensor<2x16x32xf32>
// CHECK: %[[WIDE:.*]] = linalg.generic {{.*}} outs(%[[INIT]] : tensor<2x16x32xf32>)
// CHECK: arith.extf {{.*}} : bf16 to f32
// CHECK: arith.extf {{.*}} : bf16 to f32
// CHECK: arith.mulf {{.*}} : f32
// CHECK: arith.addf {{.*}} : f32
// CHECK: %[[RESULT:.*]] = linalg.generic {{.*}} ins(%[[WIDE]] : tensor<2x16x32xf32>)
// CHECK: arith.truncf {{.*}} : f32 to bf16
// CHECK: return %[[RESULT]], %[[NARROW]]
// CHECK-LABEL: func.func @already_wide
// CHECK-NOT: arith.truncf
// CHECK: linalg.batch_matmul {{.*}} outs({{.*}} : tensor<2x16x32xf32>)
// CHECK: return
// CHECK-LABEL: func.func @different_payload
// CHECK-NOT: arith.extf
// CHECK: arith.mulf {{.*}} : f16
// CHECK: arith.subf {{.*}} : f16
// CHECK: return
