// RUN: wafer-opt --wafer-dump-group-layout-plan %s 2>&1 | FileCheck %s

module {
  func.func @matmul_bias_relu(
      %lhs: tensor<4x8xf32>,
      %rhs: tensor<8x16xf32>,
      %bias: tensor<16xf32>,
      %out: tensor<4x16xf32>) -> tensor<4x16xf32> {
    %0 = wafer.group ins(%lhs, %rhs, %bias : tensor<4x8xf32>, tensor<8x16xf32>, tensor<16xf32>)
        outs(%out : tensor<4x16xf32>) {
    ^bb0(%arg0: tensor<4x8xf32>, %arg1: tensor<8x16xf32>,
         %arg2: tensor<16xf32>, %arg3: tensor<4x16xf32>):
      %c0 = arith.constant 0.000000e+00 : f32
      %init = linalg.fill
          ins(%c0 : f32)
          outs(%arg3 : tensor<4x16xf32>) -> tensor<4x16xf32>
      %mm = linalg.matmul
          ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<8x16xf32>)
          outs(%init : tensor<4x16xf32>) -> tensor<4x16xf32>
      %biased = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
          ],
          iterator_types = ["parallel", "parallel"]
        } ins(%mm, %arg2 : tensor<4x16xf32>, tensor<16xf32>)
          outs(%arg3 : tensor<4x16xf32>) {
        ^bb0(%lhs_el: f32, %bias_el: f32, %out_el: f32):
          %sum = arith.addf %lhs_el, %bias_el : f32
          linalg.yield %sum : f32
        } -> tensor<4x16xf32>
      %zero = arith.constant dense<0.000000e+00> : tensor<4x16xf32>
      %relu = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
          ],
          iterator_types = ["parallel", "parallel"]
        } ins(%biased, %zero : tensor<4x16xf32>, tensor<4x16xf32>)
          outs(%arg3 : tensor<4x16xf32>) {
        ^bb0(%value: f32, %zero_el: f32, %out_el: f32):
          %max = arith.maximumf %value, %zero_el : f32
          linalg.yield %max : f32
        } -> tensor<4x16xf32>
      wafer.group_yield %relu : tensor<4x16xf32>
    } : tensor<4x16xf32>
    return %0 : tensor<4x16xf32>
  }

  func.func @two_independent_groups(
      %a0: tensor<4xf32>, %b0: tensor<4xf32>, %out0: tensor<4xf32>,
      %a1: tensor<8xf32>, %b1: tensor<8xf32>, %out1: tensor<8xf32>)
      -> (tensor<4xf32>, tensor<8xf32>) {
    %0 = wafer.group ins(%a0, %b0 : tensor<4xf32>, tensor<4xf32>)
        outs(%out0 : tensor<4xf32>) {
    ^bb0(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>):
      %sum = linalg.generic {
          indexing_maps = [
            affine_map<(d0) -> (d0)>,
            affine_map<(d0) -> (d0)>,
            affine_map<(d0) -> (d0)>
          ],
          iterator_types = ["parallel"]
        } ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>)
          outs(%arg2 : tensor<4xf32>) {
        ^bb0(%lhs_el: f32, %rhs_el: f32, %out_el: f32):
          %add = arith.addf %lhs_el, %rhs_el : f32
          linalg.yield %add : f32
        } -> tensor<4xf32>
      wafer.group_yield %sum : tensor<4xf32>
    } : tensor<4xf32>
    %1 = wafer.group ins(%a1, %b1 : tensor<8xf32>, tensor<8xf32>)
        outs(%out1 : tensor<8xf32>) {
    ^bb0(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>):
      %sum = linalg.generic {
          indexing_maps = [
            affine_map<(d0) -> (d0)>,
            affine_map<(d0) -> (d0)>,
            affine_map<(d0) -> (d0)>
          ],
          iterator_types = ["parallel"]
        } ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>)
          outs(%arg2 : tensor<8xf32>) {
        ^bb0(%lhs_el: f32, %rhs_el: f32, %out_el: f32):
          %add = arith.addf %lhs_el, %rhs_el : f32
          linalg.yield %add : f32
        } -> tensor<8xf32>
      wafer.group_yield %sum : tensor<8xf32>
    } : tensor<8xf32>
    return %0, %1 : tensor<4xf32>, tensor<8xf32>
  }

  func.func @collective_group(%input: tensor<4xf32>, %out: tensor<4xf32>)
      -> tensor<4xf32> {
    %0 = wafer.group ins(%input : tensor<4xf32>) outs(%out : tensor<4xf32>) {
    ^bb0(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>):
      %ar = wafer.tensor_collective.all_reduce
          ins(%arg0 : tensor<4xf32>)
          outs(%arg1 : tensor<4xf32>)
          {
          ^bb0(%lhs: f32, %rhs: f32):
            %sum = arith.addf %lhs, %rhs : f32
            wafer.tensor_collective.yield %sum : f32
          } {rank_group = array<i64: 0, 1>} -> tensor<4xf32>
      wafer.group_yield %ar : tensor<4xf32>
    } : tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  func.func @unsupported_body_op(%input: tensor<8xf32>, %out: tensor<4xf32>)
      -> tensor<4xf32> {
    %0 = wafer.group ins(%input : tensor<8xf32>) outs(%out : tensor<4xf32>) {
    ^bb0(%arg0: tensor<8xf32>, %arg1: tensor<4xf32>):
      %slice = tensor.extract_slice %arg0[0] [4] [1]
          : tensor<8xf32> to tensor<4xf32>
      wafer.group_yield %slice : tensor<4xf32>
    } : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// CHECK-LABEL: wafer.layout_plan group @matmul_bias_relu#0
// CHECK: boundary input #0 layout=tensor type=tensor<4x8xf32>
// CHECK: boundary input #1 layout=tensor type=tensor<8x16xf32>
// CHECK: op #2 linalg.matmul
// CHECK: input #0 layout=cx slice=[d0,d2]
// CHECK: input #1 layout=cx slice=[d2,d1]
// CHECK: output #0 layout=cx slice=[d0,d1]
// CHECK: result #0 layout=cx slice=[d0,d1]
// CHECK: accumulator result #0 layout=cx reduction_dims=[d2]
// CHECK: materialization #{{[0-9]+}} boundary input #0 -> op #2 input #0 tensor->cx type=tensor<4x8xf32>
// CHECK: materialization #{{[0-9]+}} boundary input #1 -> op #2 input #1 tensor->cx type=tensor<8x16xf32>
// CHECK: op #3 linalg.generic
// CHECK: input #1 layout=tensor slice=[d1] relation=broadcast
// CHECK: op #5 linalg.generic
// CHECK: result #0 layout=cx slice=[d0,d1]
// CHECK: materialization #{{[0-9]+}} op #5 result #0 -> group result #0 cx->tensor type=tensor<4x16xf32>
// CHECK-LABEL: wafer.layout_plan group @two_independent_groups#0
// CHECK: result #0 layout=tensor slice=[d0]
// CHECK-LABEL: wafer.layout_plan group @two_independent_groups#1
// CHECK: result #0 layout=tensor slice=[d0]
// CHECK-LABEL: wafer.layout_plan group @collective_group#0
// CHECK: op #0 wafer.tensor_collective.all_reduce
// CHECK: collective kind=all_reduce rank_group=[0,1]
// CHECK: input #0 layout=tensor slice=[d0]
// CHECK: result #0 layout=tensor slice=[d0]
// CHECK-LABEL: wafer.layout_plan group @unsupported_body_op#0
// CHECK: failure tiling demand failed: unsupported op tensor.extract_slice
