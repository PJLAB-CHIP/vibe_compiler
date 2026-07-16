// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-schedule-tensor-program{logical-rank=0 tile-search-effort=quick})' %s | FileCheck %s --implicit-check-not=linalg. --implicit-check-not=tensor.

#identity_2d = affine_map<(d0, d1) -> (d0, d1)>
#identity_3d = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// Shape changes are not scheduling boundaries.  The two producers and the
// final add form one complete dataflow scope even though %middle is expanded
// from 2-D to 3-D before it meets %early.
func.func @cross_shape_producer_chain_fuses(
    %early_input: tensor<1x4x4xf32>,
    %middle_input: tensor<4x4xf32>,
    %early_init: tensor<1x4x4xf32>,
    %middle_init: tensor<4x4xf32>,
    %result_init: tensor<1x4x4xf32>) -> tensor<1x4x4xf32> {
  %early = linalg.generic {
      indexing_maps = [#identity_3d, #identity_3d],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%early_input : tensor<1x4x4xf32>)
      outs(%early_init : tensor<1x4x4xf32>) {
  ^bb0(%input: f32, %old: f32):
    linalg.yield %input : f32
  } -> tensor<1x4x4xf32>

  %middle = linalg.generic {
      indexing_maps = [#identity_2d, #identity_2d],
      iterator_types = ["parallel", "parallel"]
    } ins(%middle_input : tensor<4x4xf32>)
      outs(%middle_init : tensor<4x4xf32>) {
  ^bb0(%input: f32, %old: f32):
    linalg.yield %input : f32
  } -> tensor<4x4xf32>
  %middle_expanded = tensor.expand_shape %middle [[0, 1], [2]]
      output_shape [1, 4, 4]
      : tensor<4x4xf32> into tensor<1x4x4xf32>

  %result = linalg.generic {
      indexing_maps = [#identity_3d, #identity_3d, #identity_3d],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%early, %middle_expanded
          : tensor<1x4x4xf32>, tensor<1x4x4xf32>)
      outs(%result_init : tensor<1x4x4xf32>) {
  ^bb0(%lhs: f32, %rhs: f32, %old: f32):
    %sum = arith.addf %lhs, %rhs : f32
    linalg.yield %sum : f32
  } -> tensor<1x4x4xf32>
  return %result : tensor<1x4x4xf32>
}

// CHECK-LABEL: func.func @cross_shape_producer_chain_fuses
// CHECK-COUNT-1: wafer.tile.region
// CHECK: memref.expand_shape
// CHECK: wafer.instr.elementwise <add>
// CHECK: bufferization.to_tensor
// CHECK: return
