// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-schedule-tensor-program{logical-rank=0 tile-search-effort=quick})' %s | FileCheck %s

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#id1 = affine_map<(d0) -> (d0)>

module {
  func.func @constant_and_reshape_have_typed_ddr_roots(
      %input: tensor<2x2xf32>, %first_init: tensor<2x2xf32>,
      %second_init: tensor<4xf32>) -> (tensor<2x2xf32>, tensor<4xf32>) {
    %constant = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]>
        : tensor<2x2xf32>
    %first = linalg.generic {
        indexing_maps = [#id2, #id2, #id2],
        iterator_types = ["parallel", "parallel"]
      } ins(%input, %constant : tensor<2x2xf32>, tensor<2x2xf32>)
        outs(%first_init : tensor<2x2xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %old: f32):
      %sum = arith.addf %lhs, %rhs : f32
      linalg.yield %sum : f32
    } -> tensor<2x2xf32>
    %flat = tensor.collapse_shape %first [[0, 1]]
        : tensor<2x2xf32> into tensor<4xf32>
    %second = linalg.generic {
        indexing_maps = [#id1, #id1], iterator_types = ["parallel"]
      } ins(%flat : tensor<4xf32>) outs(%second_init : tensor<4xf32>) {
    ^bb0(%value: f32, %old: f32):
      linalg.yield %value : f32
    } -> tensor<4xf32>
    // Keeping the pre-reshape value live forces a real task/DDR boundary;
    // the scheduler must express the reshape as a memref view of that root.
    return %first, %second : tensor<2x2xf32>, tensor<4xf32>
  }
}

// CHECK: memref.global "private" constant @__wafer_constant
// CHECK-LABEL: func.func @constant_and_reshape_have_typed_ddr_roots
// CHECK: %[[CONSTANT:.+]] = memref.get_global @__wafer_constant
// CHECK: wafer.tile.region({{.*}}%[[CONSTANT]]
// CHECK: %[[RESHAPED:.+]] = memref.collapse_shape
// CHECK: memref.cast %[[RESHAPED]]
// CHECK-NOT: bufferization.to_memref %{{.*}} : memref<4xf32
// CHECK: return
