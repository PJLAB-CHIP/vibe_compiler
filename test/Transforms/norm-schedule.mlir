// RUN: wafer-opt --wafer-check-norm-schedule %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
#col = affine_map<(d0, d1) -> (d1)>
#vec = affine_map<(d0) -> (d0)>

module {
  func.func @rmsnorm_schedule(
      %x: tensor<2x4xf32>,
      %hidden: tensor<2xf32>,
      %eps: tensor<2xf32>,
      %weight: tensor<4xf32>) -> tensor<2x4xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %square_init = tensor.empty() : tensor<2x4xf32>
    %square = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%x, %x : tensor<2x4xf32>, tensor<2x4xf32>)
        outs(%square_init : tensor<2x4xf32>) {
      ^bb0(%x0: f32, %x1: f32, %out_s: f32):
        %product = arith.mulf %x0, %x1 : f32
        linalg.yield %product : f32
    } -> tensor<2x4xf32>
    %sum_init_empty = tensor.empty() : tensor<2xf32>
    %sum_init = linalg.fill ins(%zero : f32)
        outs(%sum_init_empty : tensor<2xf32>) -> tensor<2xf32>
    %sum = linalg.reduce { arith.addf }
        ins(%square : tensor<2x4xf32>)
        outs(%sum_init : tensor<2xf32>)
        dimensions = [1]
    %mean_init = tensor.empty() : tensor<2xf32>
    %mean = linalg.generic {
        indexing_maps = [#vec, #vec, #vec],
        iterator_types = ["parallel"]}
        ins(%sum, %hidden : tensor<2xf32>, tensor<2xf32>)
        outs(%mean_init : tensor<2xf32>) {
      ^bb0(%sum_s: f32, %hidden_s: f32, %out_s: f32):
        %div = arith.divf %sum_s, %hidden_s : f32
        linalg.yield %div : f32
    } -> tensor<2xf32>
    %eps_init = tensor.empty() : tensor<2xf32>
    %mean_eps = linalg.generic {
        indexing_maps = [#vec, #vec, #vec],
        iterator_types = ["parallel"]}
        ins(%mean, %eps : tensor<2xf32>, tensor<2xf32>)
        outs(%eps_init : tensor<2xf32>) {
      ^bb0(%mean_s: f32, %eps_s: f32, %out_s: f32):
        %sum_eps = arith.addf %mean_s, %eps_s : f32
        linalg.yield %sum_eps : f32
    } -> tensor<2xf32>
    %rms_init = tensor.empty() : tensor<2xf32>
    %rms = linalg.generic {
        indexing_maps = [#vec, #vec],
        iterator_types = ["parallel"]}
        ins(%mean_eps : tensor<2xf32>)
        outs(%rms_init : tensor<2xf32>) {
      ^bb0(%mean_eps_s: f32, %out_s: f32):
        %rsqrt = math.rsqrt %mean_eps_s : f32
        linalg.yield %rsqrt : f32
    } -> tensor<2xf32>
    %norm_init = tensor.empty() : tensor<2x4xf32>
    %normed = linalg.generic {
        indexing_maps = [#map, #row, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%x, %rms : tensor<2x4xf32>, tensor<2xf32>)
        outs(%norm_init : tensor<2x4xf32>) {
      ^bb0(%x_s: f32, %rms_s: f32, %out_s: f32):
        %product = arith.mulf %x_s, %rms_s : f32
        linalg.yield %product : f32
    } -> tensor<2x4xf32>
    %scaled_init = tensor.empty() : tensor<2x4xf32>
    %scaled = linalg.generic {
        indexing_maps = [#map, #col, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%normed, %weight : tensor<2x4xf32>, tensor<4xf32>)
        outs(%scaled_init : tensor<2x4xf32>) {
      ^bb0(%normed_s: f32, %weight_s: f32, %out_s: f32):
        %product = arith.mulf %normed_s, %weight_s : f32
        linalg.yield %product : f32
    } -> tensor<2x4xf32>
    return %scaled : tensor<2x4xf32>
  }
}

// CHECK-LABEL: func.func @rmsnorm_schedule
// CHECK: linalg.reduce
// CHECK: math.rsqrt
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#map, #map{{[0-9]+}}, #map]
// CHECK-NOT: wafer.norm
