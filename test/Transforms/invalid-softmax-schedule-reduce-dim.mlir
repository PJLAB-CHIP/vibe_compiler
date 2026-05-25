// RUN: not wafer-opt --wafer-check-softmax-schedule %s 2>&1 | FileCheck %s

module {
  func.func @bad_reduce_dim(%scores: tensor<2x4xf32>) -> tensor<4xf32> {
    %neg_inf = arith.constant -3.40282347E+38 : f32
    %max_init_empty = tensor.empty() : tensor<4xf32>
    %max_init = linalg.fill ins(%neg_inf : f32)
        outs(%max_init_empty : tensor<4xf32>) -> tensor<4xf32>
    %row_max = linalg.reduce { arith.maximumf }
        ins(%scores : tensor<2x4xf32>)
        outs(%max_init : tensor<4xf32>)
        dimensions = [0]
    return %row_max : tensor<4xf32>
  }
}

// CHECK: softmax schedule requires a hidden-dimension reduce max stage
