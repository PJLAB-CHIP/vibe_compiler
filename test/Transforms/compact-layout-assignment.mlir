// RUN: wafer-opt --wafer-compact-layout-assignment %s | FileCheck %s

module {
  func.func @remove_inverse_layout_pair(
      %src: tensor<4xf32>,
      %dest: tensor<4xf32>) -> tensor<4xf32> {
    %0 = wafer.tile_region(%src, %dest : tensor<4xf32>, tensor<4xf32>)
        -> (tensor<4xf32>) {
    ^bb0(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>):
      %tile = wafer.load_tile %arg0
          : tensor<4xf32>
         -> !wafer.tile_buffer<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
      %cx = wafer.layout.materialize %tile
          : !wafer.tile_buffer<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
         -> !wafer.tile_buffer<tensor<4xf32>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
      %tensor = wafer.layout.materialize %cx
          : !wafer.tile_buffer<tensor<4xf32>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
         -> !wafer.tile_buffer<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
      wafer.store_tile %tensor, %arg1
          : !wafer.tile_buffer<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
         -> tensor<4xf32>
      wafer.tile_yield %arg1 : tensor<4xf32>
    }
    return %0 : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @remove_inverse_layout_pair(
// CHECK: %[[TILE:.+]] = wafer.load_tile
// CHECK-NOT: wafer.layout.materialize
// CHECK: wafer.store_tile %[[TILE]],
// CHECK: wafer.tile_yield
