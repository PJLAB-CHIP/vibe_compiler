// RUN: not wafer-opt %s 2>&1 | FileCheck %s

func.func @invalid_physical_buffer(
    %input: memref<2x4xf32, #wafer.memory<spm, tensor>>,
    %out: tensor<8x4xf32>) -> tensor<8x4xf32> {
  // CHECK: linalg-ext collective operands and results must be ranked tensors
  %0 = wafer_linalg_ext.collective.all_gather
      ins(%input : memref<2x4xf32, #wafer.memory<spm, tensor>>)
      outs(%out : tensor<8x4xf32>)
      {axis = 0 : i64, rank_group = array<i64: 0, 1, 2, 3>}
      -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}
