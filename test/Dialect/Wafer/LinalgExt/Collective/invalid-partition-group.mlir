// RUN: not wafer-opt %s 2>&1 | FileCheck %s

func.func @invalid_partition_group(
    %input: tensor<2x4xf32>) -> tensor<8x4xf32> {
  %out = tensor.empty() : tensor<8x4xf32>
  // CHECK: partition_group entries must be unique
  %0 = wafer.linalg_ext.collective.all_gather
      ins(%input : tensor<2x4xf32>)
      outs(%out : tensor<8x4xf32>)
      {axis = 0 : i64, partition_group = array<i64: 0, 1, 1, 3>}
      -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}
