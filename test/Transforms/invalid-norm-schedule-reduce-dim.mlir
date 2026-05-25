// RUN: not wafer-opt --wafer-check-norm-schedule %s 2>&1 | FileCheck %s

module {
  func.func @bad_reduce_dim(%x: tensor<2x4xf32>) -> tensor<4xf32> {
    %zero = arith.constant 0.000000e+00 : f32
    %sum_init_empty = tensor.empty() : tensor<4xf32>
    %sum_init = linalg.fill ins(%zero : f32)
        outs(%sum_init_empty : tensor<4xf32>) -> tensor<4xf32>
    %sum = linalg.reduce { arith.addf }
        ins(%x : tensor<2x4xf32>)
        outs(%sum_init : tensor<4xf32>)
        dimensions = [0]
    return %sum : tensor<4xf32>
  }
}

// CHECK: norm schedule requires a hidden-dimension reduction
