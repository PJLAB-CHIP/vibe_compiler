// RUN: not wafer-opt %s 2>&1 | FileCheck %s

func.func @invalid_axis(
    %input: tensor<2x4xf32>,
    %out: tensor<8x4xf32>) -> tensor<8x4xf32> {
  // CHECK: collective axis must be within tensor rank
  %0 = wafer.linalg_ext.collective.all_gather
      ins(%input : tensor<2x4xf32>)
      outs(%out : tensor<8x4xf32>)
      {axis = 2 : i64, rank_group = array<i64: 0, 1, 2, 3>}
      -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}
