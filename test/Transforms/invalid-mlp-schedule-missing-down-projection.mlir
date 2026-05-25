// RUN: not wafer-opt --wafer-check-mlp-schedule %s 2>&1 | FileCheck %s

module {
  func.func @missing_down_projection(
      %hidden: tensor<4x8xf32>,
      %gate_w: tensor<8x16xf32>,
      %up_w: tensor<8x16xf32>,
      %gate_out: tensor<4x16xf32>,
      %up_out: tensor<4x16xf32>) -> tensor<4x16xf32> {
    %gate = linalg.matmul
        ins(%hidden, %gate_w : tensor<4x8xf32>, tensor<8x16xf32>)
        outs(%gate_out : tensor<4x16xf32>) -> tensor<4x16xf32>
    %up = linalg.matmul
        ins(%hidden, %up_w : tensor<4x8xf32>, tensor<8x16xf32>)
        outs(%up_out : tensor<4x16xf32>) -> tensor<4x16xf32>
    %activated_init = tensor.empty() : tensor<4x16xf32>
    %activated = linalg.elementwise kind=#linalg.elementwise_kind<tanh>
        ins(%gate : tensor<4x16xf32>)
        outs(%activated_init : tensor<4x16xf32>) -> tensor<4x16xf32>
    %gated_init = tensor.empty() : tensor<4x16xf32>
    %gated = linalg.elementwise kind=#linalg.elementwise_kind<mul>
        ins(%activated, %up : tensor<4x16xf32>, tensor<4x16xf32>)
        outs(%gated_init : tensor<4x16xf32>) -> tensor<4x16xf32>
    return %gated : tensor<4x16xf32>
  }
}

// CHECK: MLP schedule requires a down projection matmul consuming the gated activation
