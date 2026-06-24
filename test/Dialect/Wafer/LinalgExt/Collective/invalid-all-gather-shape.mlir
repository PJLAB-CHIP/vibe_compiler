// RUN: not wafer-opt %s 2>&1 | FileCheck %s

func.func @invalid_all_gather_shape(
    %input: tensor<2x4xf32>,
    %out: tensor<7x4xf32>) -> tensor<7x4xf32> {
  // CHECK: all_gather result dimension along axis must equal input dimension times rank_group size
  %0 = wafer.linalg_ext.collective.all_gather
      ins(%input : tensor<2x4xf32>)
      outs(%out : tensor<7x4xf32>)
      {axis = 0 : i64, rank_group = array<i64: 0, 1, 2, 3>}
      -> tensor<7x4xf32>
  return %0 : tensor<7x4xf32>
}
