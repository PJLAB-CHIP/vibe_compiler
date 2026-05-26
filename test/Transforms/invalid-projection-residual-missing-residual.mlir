// RUN: not wafer-opt --wafer-check-projection-residual-schedule %s 2>&1 | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#col = affine_map<(d0, d1) -> (d1)>

module {
  func.func @missing_residual(
      %hidden: tensor<4x8xf32>,
      %weight: tensor<8x16xf32>,
      %bias: tensor<16xf32>,
      %out: tensor<4x16xf32>) -> tensor<4x16xf32> {
    %projected = linalg.matmul
        ins(%hidden, %weight : tensor<4x8xf32>, tensor<8x16xf32>)
        outs(%out : tensor<4x16xf32>) -> tensor<4x16xf32>
    %biased_init = tensor.empty() : tensor<4x16xf32>
    %biased = linalg.generic {
        indexing_maps = [#map, #col, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%projected, %bias : tensor<4x16xf32>, tensor<16xf32>)
        outs(%biased_init : tensor<4x16xf32>) {
      ^bb0(%projected_s: f32, %bias_s: f32, %out_s: f32):
        %sum = arith.addf %projected_s, %bias_s : f32
        linalg.yield %sum : f32
    } -> tensor<4x16xf32>
    return %biased : tensor<4x16xf32>
  }
}

// CHECK: projection residual schedule requires a rank-2 residual add stage
