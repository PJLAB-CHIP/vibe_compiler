// RUN: wafer-opt --wafer-check-norm-schedule %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
#col = affine_map<(d0, d1) -> (d1)>

module {
  func.func @rmsnorm_schedule(
      %x: tensor<2x4xf32>,
      %hidden: tensor<2xf32>,
      %eps: tensor<2xf32>,
      %weight: tensor<4xf32>) -> tensor<2x4xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %square_init = tensor.empty() : tensor<2x4xf32>
    %square = linalg.elementwise kind=#linalg.elementwise_kind<mul>
        ins(%x, %x : tensor<2x4xf32>, tensor<2x4xf32>)
        outs(%square_init : tensor<2x4xf32>) -> tensor<2x4xf32>
    %sum_init_empty = tensor.empty() : tensor<2xf32>
    %sum_init = linalg.fill ins(%zero : f32)
        outs(%sum_init_empty : tensor<2xf32>) -> tensor<2xf32>
    %sum = linalg.reduce { arith.addf }
        ins(%square : tensor<2x4xf32>)
        outs(%sum_init : tensor<2xf32>)
        dimensions = [1]
    %mean_init = tensor.empty() : tensor<2xf32>
    %mean = linalg.elementwise kind=#linalg.elementwise_kind<div>
        ins(%sum, %hidden : tensor<2xf32>, tensor<2xf32>)
        outs(%mean_init : tensor<2xf32>) -> tensor<2xf32>
    %eps_init = tensor.empty() : tensor<2xf32>
    %mean_eps = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%mean, %eps : tensor<2xf32>, tensor<2xf32>)
        outs(%eps_init : tensor<2xf32>) -> tensor<2xf32>
    %rms_init = tensor.empty() : tensor<2xf32>
    %rms = linalg.elementwise kind=#linalg.elementwise_kind<rsqrt>
        ins(%mean_eps : tensor<2xf32>)
        outs(%rms_init : tensor<2xf32>) -> tensor<2xf32>
    %norm_init = tensor.empty() : tensor<2x4xf32>
    %normed = linalg.elementwise kind=#linalg.elementwise_kind<mul>
        indexing_maps = [#map, #row, #map]
        ins(%x, %rms : tensor<2x4xf32>, tensor<2xf32>)
        outs(%norm_init : tensor<2x4xf32>) -> tensor<2x4xf32>
    %scaled_init = tensor.empty() : tensor<2x4xf32>
    %scaled = linalg.elementwise kind=#linalg.elementwise_kind<mul>
        indexing_maps = [#map, #col, #map]
        ins(%normed, %weight : tensor<2x4xf32>, tensor<4xf32>)
        outs(%scaled_init : tensor<2x4xf32>) -> tensor<2x4xf32>
    return %scaled : tensor<2x4xf32>
  }
}

// CHECK-LABEL: func.func @rmsnorm_schedule
// CHECK: linalg.reduce
// CHECK: linalg.elementwise kind=#linalg.elementwise_kind<rsqrt>
// CHECK: linalg.elementwise
// CHECK-SAME: indexing_maps = [#map, #map{{[0-9]+}}, #map]
// CHECK-NOT: wafer.norm
