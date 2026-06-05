// RUN: not wafer-opt %s 2>&1 | FileCheck %s

func.func @invalid_physical_buffer(
    %input: !wafer.storage<tensor<2x4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>,
    %out: tensor<8x4xf32>) -> tensor<8x4xf32> {
  // CHECK: tensor collective operands and results must be ranked tensors
  %0 = wafer.tensor.all_gather
      ins(%input : !wafer.storage<tensor<2x4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>)
      outs(%out : tensor<8x4xf32>)
      {axis = 0 : i64, rank_group = array<i64: 0, 1, 2, 3>}
      -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}
