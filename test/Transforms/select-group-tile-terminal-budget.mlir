// RUN: not wafer-opt --mlir-print-ir-after-failure --wafer-select-group-tile='logical-rank=0' %s 2>&1 | FileCheck %s

func.func @two_ordered_reductions_exceed_rank_budget(
    %input0: tensor<1x511xf32>, %out0: tensor<1xf32>,
    %input1: tensor<1x511xf32>, %out1: tensor<1xf32>)
    -> (tensor<1xf32>, tensor<1xf32>) {
  %result0 = wafer.group ins(%input0 : tensor<1x511xf32>)
      outs(%out0 : tensor<1xf32>) {
  ^bb0(%input: tensor<1x511xf32>, %out: tensor<1xf32>):
    %init_scalar = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<1xf32>
    %init = linalg.fill ins(%init_scalar : f32)
        outs(%empty : tensor<1xf32>) -> tensor<1xf32>
    %sum = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0)>
        ],
        iterator_types = ["parallel", "reduction"]
      } ins(%input : tensor<1x511xf32>)
        outs(%init : tensor<1xf32>) {
      ^bb0(%value: f32, %acc: f32):
        %next = arith.addf %value, %acc : f32
        linalg.yield %next : f32
      } -> tensor<1xf32>
    wafer.group.yield %sum : tensor<1xf32>
  } : tensor<1xf32>

  %result1 = wafer.group ins(%input1 : tensor<1x511xf32>)
      outs(%out1 : tensor<1xf32>) {
  ^bb0(%input: tensor<1x511xf32>, %out: tensor<1xf32>):
    %init_scalar = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<1xf32>
    %init = linalg.fill ins(%init_scalar : f32)
        outs(%empty : tensor<1xf32>) -> tensor<1xf32>
    %sum = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0)>
        ],
        iterator_types = ["parallel", "reduction"]
      } ins(%input : tensor<1x511xf32>)
        outs(%init : tensor<1xf32>) {
      ^bb0(%value: f32, %acc: f32):
        %next = arith.addf %value, %acc : f32
        linalg.yield %next : f32
      } -> tensor<1xf32>
    wafer.group.yield %sum : tensor<1xf32>
  } : tensor<1xf32>

  return %result0, %result1 : tensor<1xf32>, tensor<1xf32>
}

// CHECK: static_terminal_budget_exceeded: selected rank requires
// CHECK-SAME: terminal instruction issue/completion operations; limit is 4096
// CHECK: IR Dump After SelectGroupTilePass Failed
// CHECK: func.func @two_ordered_reductions_exceed_rank_budget
// CHECK: wafer.group
// CHECK: linalg.generic
// CHECK: wafer.group
// CHECK: linalg.generic
// CHECK-NOT: wafer.instr.
