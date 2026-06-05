// RUN: wafer-opt --wafer-dump-group-tiling-demand %s 2>&1 | FileCheck %s

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
      wafer.group.yield %relu : tensor<4x16xf32>
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
      wafer.group.yield %sum : tensor<4xf32>
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
      wafer.group.yield %sum : tensor<8xf32>
    } : tensor<8xf32>
    return %0, %1 : tensor<4xf32>, tensor<8xf32>
  }

  func.func @collective_group(%input: tensor<4xf32>, %out: tensor<4xf32>)
      -> tensor<4xf32> {
    %0 = wafer.group ins(%input : tensor<4xf32>) outs(%out : tensor<4xf32>) {
    ^bb0(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>):
      %ar = wafer.tensor.all_reduce
          ins(%arg0 : tensor<4xf32>)
          outs(%arg1 : tensor<4xf32>)
          {
          ^bb0(%lhs: f32, %rhs: f32):
            %sum = arith.addf %lhs, %rhs : f32
            wafer.tensor.yield %sum : f32
          } {rank_group = array<i64: 0, 1>} -> tensor<4xf32>
      wafer.group.yield %ar : tensor<4xf32>
    } : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}

// CHECK-LABEL: wafer.tiling_demand group @matmul_bias_relu#0
// CHECK: result_tile #0 type=tensor<4x16xf32> slice=[d0,d1]
// CHECK: op #1 linalg.fill
// CHECK: output #0 slice=[d0,d1]
// CHECK: op #2 linalg.matmul
// CHECK: iterators=parallel(d0,d1) reduction(d2)
// CHECK: input #0 slice=[d0,d2]
// CHECK: input #1 slice=[d2,d1]
// CHECK: output #0 slice=[d0,d1]
// CHECK: result #0 slice=[d0,d1]
// CHECK: accumulator result #0 reduction_dims=[d2]
// CHECK: op #3 linalg.generic
// CHECK: input #1 slice=[d1]
// CHECK: op #5 linalg.generic
// CHECK: input #0 slice=[d0,d1]
// CHECK-LABEL: wafer.tiling_demand group @two_independent_groups#0
// CHECK: result_tile #0 type=tensor<4xf32> slice=[d0]
// CHECK-LABEL: wafer.tiling_demand group @two_independent_groups#1
// CHECK: result_tile #0 type=tensor<8xf32> slice=[d0]
// CHECK-LABEL: wafer.tiling_demand group @collective_group#0
// CHECK: op #0 wafer.tensor.all_reduce
// CHECK: collective kind=all_reduce rank_group=[0,1]
// CHECK: input #0 slice=[d0]
// CHECK: result #0 slice=[d0]
