// RUN: not wafer-opt %s 2>&1 | FileCheck %s

func.func @invalid_all_reduce_demotion(
    %input: tensor<4xf32>,
    %out: tensor<4xf16>) -> tensor<4xf16> {
  // CHECK: input element type must match the out element type or use a supported floating promotion
  %0 = wafer.linalg_ext.collective.all_reduce
      ins(%input : tensor<4xf32>)
      outs(%out : tensor<4xf16>)
      {
    ^bb0(%lhs: f16, %rhs: f16):
      %sum = arith.addf %lhs, %rhs : f16
      wafer.linalg_ext.collective.yield %sum : f16
    } {rank_group = array<i64: 0, 1>} -> tensor<4xf16>
  return %0 : tensor<4xf16>
}
