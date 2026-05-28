// RUN: wafer-opt --wafer-form-groups --wafer-check-root-tile-candidates --wafer-materialize-single-tile --wafer-check-spm-allocation --wafer-materialize-ddr-external-bindings --wafer-lower-tile-region-to-c-abi %s | FileCheck %s

#qk_maps = [
  affine_map<(b, h, q, k, d) -> (b, h, q, d)>,
  affine_map<(b, h, q, k, d) -> (b, h, k, d)>,
  affine_map<(b, h, q, k, d) -> (b, h, q, k)>
]

module {
  func.func @attention_score(
      %query: tensor<2x3x5x8xf32>,
      %key: tensor<2x3x7x8xf32>,
      %out: tensor<2x3x5x7xf32>) -> tensor<2x3x5x7xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<2x3x5x7xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x3x5x7xf32>) -> tensor<2x3x5x7xf32>
    %score = linalg.generic {
        indexing_maps = #qk_maps,
        iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
        ins(%query, %key : tensor<2x3x5x8xf32>, tensor<2x3x7x8xf32>)
        outs(%init : tensor<2x3x5x7xf32>) {
    ^bb0(%q: f32, %k: f32, %acc: f32):
      %mul = arith.mulf %q, %k : f32
      %add = arith.addf %acc, %mul : f32
      linalg.yield %add : f32
    } -> tensor<2x3x5x7xf32>
    return %score : tensor<2x3x5x7xf32>
  }
}

// CHECK-LABEL: func.func @attention_score
// CHECK-NOT: linalg.generic
// CHECK-NOT: wafer.compute.gemm
// CHECK: wafer.abi.gemm <issue_only>
// CHECK-SAME: batch_count = 6 : i64
// CHECK-SAME: k = 8 : i64
// CHECK-SAME: m = 5 : i64
// CHECK-SAME: n = 7 : i64
