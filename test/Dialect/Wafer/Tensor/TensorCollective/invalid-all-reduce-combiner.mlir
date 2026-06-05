// RUN: not wafer-opt %s 2>&1 | FileCheck %s

func.func @invalid_all_reduce_combiner(
    %input: tensor<4xf32>,
    %out: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: combiner must yield exactly one scalar value
  %0 = wafer.tensor.all_reduce
      ins(%input : tensor<4xf32>)
      outs(%out : tensor<4xf32>)
      {
    ^bb0(%lhs: f32, %rhs: f32):
      wafer.tensor.yield
    } {rank_group = array<i64: 0, 1>} -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
