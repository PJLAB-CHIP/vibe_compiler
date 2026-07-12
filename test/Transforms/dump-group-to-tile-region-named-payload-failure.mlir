// RUN: wafer-opt --wafer-dump-group-to-tile-region='logical-rank=0' %s 2>&1 | FileCheck %s

#mm_lhs = affine_map<(d0, d1, d2) -> (d0, d2)>
#mm_rhs = affine_map<(d0, d1, d2) -> (d2, d1)>
#mm_out = affine_map<(d0, d1, d2) -> (d0, d1)>
#bmm_lhs = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#bmm_rhs = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#bmm_out = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

func.func @fill_wrong_yield(%out: tensor<4xf32>) -> tensor<4xf32> {
  %group = wafer.group ins() outs(%out : tensor<4xf32>) {
  ^bb0(%out_arg: tensor<4xf32>):
    %zero = arith.constant 0.0 : f32
    %filled = "linalg.fill"(%zero, %out_arg) <{
        operandSegmentSizes = array<i32: 1, 1>
      }> ({
      ^bb0(%value: f32, %init: f32):
        "linalg.yield"(%init) : (f32) -> ()
      }) : (f32, tensor<4xf32>) -> tensor<4xf32>
    wafer.group.yield %filled : tensor<4xf32>
  } : tensor<4xf32>
  return %group : tensor<4xf32>
}

func.func @matmul_extra_payload(%lhs: tensor<2x4xf16>,
                                %rhs: tensor<4x3xf16>,
                                %out: tensor<2x3xf16>) -> tensor<2x3xf16> {
  %group = wafer.group
      ins(%lhs, %rhs : tensor<2x4xf16>, tensor<4x3xf16>)
      outs(%out : tensor<2x3xf16>) {
  ^bb0(%lhs_arg: tensor<2x4xf16>, %rhs_arg: tensor<4x3xf16>,
       %out_arg: tensor<2x3xf16>):
    %zero = arith.constant 0.0 : f16
    %init = linalg.fill ins(%zero : f16)
        outs(%out_arg : tensor<2x3xf16>) -> tensor<2x3xf16>
    %result = "linalg.matmul"(%lhs_arg, %rhs_arg, %init) <{
        operandSegmentSizes = array<i32: 2, 1>
      }> ({
      ^bb0(%left: f16, %right: f16, %acc: f16):
        %product = arith.mulf %left, %right : f16
        %extra = arith.subf %left, %right : f16
        %sum = arith.addf %acc, %product : f16
        "linalg.yield"(%sum) : (f16) -> ()
      }) {linalg.memoized_indexing_maps = [#mm_lhs, #mm_rhs, #mm_out]}
      : (tensor<2x4xf16>, tensor<4x3xf16>, tensor<2x3xf16>)
          -> tensor<2x3xf16>
    wafer.group.yield %result : tensor<2x3xf16>
  } : tensor<2x3xf16>
  return %group : tensor<2x3xf16>
}

func.func @batch_matmul_extra_payload(%lhs: tensor<2x3x4xf32>,
                                      %rhs: tensor<2x4x5xf32>,
                                      %out: tensor<2x3x5xf32>)
    -> tensor<2x3x5xf32> {
  %group = wafer.group
      ins(%lhs, %rhs : tensor<2x3x4xf32>, tensor<2x4x5xf32>)
      outs(%out : tensor<2x3x5xf32>) {
  ^bb0(%lhs_arg: tensor<2x3x4xf32>, %rhs_arg: tensor<2x4x5xf32>,
       %out_arg: tensor<2x3x5xf32>):
    %zero = arith.constant 0.0 : f32
    %init = linalg.fill ins(%zero : f32)
        outs(%out_arg : tensor<2x3x5xf32>) -> tensor<2x3x5xf32>
    %result = "linalg.batch_matmul"(%lhs_arg, %rhs_arg, %init) <{
        operandSegmentSizes = array<i32: 2, 1>
      }> ({
      ^bb0(%left: f32, %right: f32, %acc: f32):
        %product = arith.mulf %left, %right : f32
        %extra = arith.subf %left, %right : f32
        %sum = arith.addf %acc, %product : f32
        "linalg.yield"(%sum) : (f32) -> ()
      }) {linalg.memoized_indexing_maps = [#bmm_lhs, #bmm_rhs, #bmm_out]}
      : (tensor<2x3x4xf32>, tensor<2x4x5xf32>, tensor<2x3x5xf32>)
          -> tensor<2x3x5xf32>
    wafer.group.yield %result : tensor<2x3x5xf32>
  } : tensor<2x3x5xf32>
  return %group : tensor<2x3x5xf32>
}

// CHECK: wafer.group_to_tile_region group @fill_wrong_yield#0
// CHECK-NEXT: failure linalg.fill requires the canonical scalar payload
// CHECK: wafer.group_to_tile_region group @matmul_extra_payload#0
// CHECK-NEXT: failure matmul requires an exact multiply-accumulate payload
// CHECK: wafer.group_to_tile_region group @batch_matmul_extra_payload#0
// CHECK-NEXT: failure batch matmul requires an exact multiply-accumulate payload
