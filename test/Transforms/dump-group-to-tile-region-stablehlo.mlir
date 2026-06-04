// REQUIRES: stablehlo
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-stablehlo-to-linalg,wafer-form-logical-groups,wafer-dump-group-to-tile-region)' %s 2>&1 | FileCheck %s

module {
  func.func @lower_reduce_sum(%arg0: tensor<2x4xf32>,
                              %init: tensor<f32>) -> tensor<2xf32> {
    %0 = "stablehlo.reduce"(%arg0, %init) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %1 = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) {
      dimensions = array<i64: 1>
    } : (tensor<2x4xf32>, tensor<f32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }

  func.func @lower_broadcast(%arg0: tensor<4xf32>) -> tensor<2x4xf32> {
    %0 = "stablehlo.broadcast_in_dim"(%arg0) {
      broadcast_dimensions = array<i64: 1>
    } : (tensor<4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }

  func.func @lower_compare(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>)
      -> tensor<4xi1> {
    %0 = "stablehlo.compare"(%lhs, %rhs) {
      comparison_direction = #stablehlo<comparison_direction GT>
    } : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
    return %0 : tensor<4xi1>
  }
}

// CHECK-LABEL: wafer.group_to_tile_region group @lower_reduce_sum#0
// CHECK: wafer.compute.fill
// CHECK: wafer.layout.materialize
// CHECK: #wafer.mem_layout<cx>
// CHECK: wafer.compute.reduce <sum>
// CHECK-SAME: dimensions = array<i64: 1>
// CHECK: f32) -> !wafer.tile_buffer<tensor<2xf32>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
// CHECK: wafer.store_tile
// CHECK-LABEL: wafer.group_to_tile_region group @lower_broadcast#0
// CHECK: wafer.move.broadcast
// CHECK-SAME: dimensions = array<i64: 1>
// CHECK: wafer.store_tile
// CHECK-LABEL: wafer.group_to_tile_region group @lower_compare#0
// CHECK: wafer.compute.elementwise <gt>
// CHECK: wafer.store_tile
