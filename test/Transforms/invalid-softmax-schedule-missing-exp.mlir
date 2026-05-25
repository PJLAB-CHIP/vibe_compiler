// RUN: not wafer-opt --wafer-check-softmax-schedule %s 2>&1 | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>

module {
  func.func @missing_exp(%scores: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %neg_inf = arith.constant -3.40282347E+38 : f32
    %zero = arith.constant 0.000000e+00 : f32
    %max_init_empty = tensor.empty() : tensor<2xf32>
    %max_init = linalg.fill ins(%neg_inf : f32)
        outs(%max_init_empty : tensor<2xf32>) -> tensor<2xf32>
    %row_max = linalg.reduce { arith.maximumf }
        ins(%scores : tensor<2x4xf32>)
        outs(%max_init : tensor<2xf32>)
        dimensions = [1]
    %shifted_init = tensor.empty() : tensor<2x4xf32>
    %shifted = linalg.elementwise kind=#linalg.elementwise_kind<sub>
        indexing_maps = [#map, #row, #map]
        ins(%scores, %row_max : tensor<2x4xf32>, tensor<2xf32>)
        outs(%shifted_init : tensor<2x4xf32>) -> tensor<2x4xf32>
    %sum_init_empty = tensor.empty() : tensor<2xf32>
    %sum_init = linalg.fill ins(%zero : f32)
        outs(%sum_init_empty : tensor<2xf32>) -> tensor<2xf32>
    %row_sum = linalg.reduce { arith.addf }
        ins(%shifted : tensor<2x4xf32>)
        outs(%sum_init : tensor<2xf32>)
        dimensions = [1]
    %prob_init = tensor.empty() : tensor<2x4xf32>
    %prob = linalg.elementwise kind=#linalg.elementwise_kind<div>
        indexing_maps = [#map, #row, #map]
        ins(%shifted, %row_sum : tensor<2x4xf32>, tensor<2xf32>)
        outs(%prob_init : tensor<2x4xf32>) -> tensor<2x4xf32>
    return %prob : tensor<2x4xf32>
  }
}

// CHECK: softmax schedule requires an exp elementwise stage after row max subtraction
