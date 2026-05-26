// RUN: wafer-opt --wafer-lower-ring-all-gather %s | FileCheck %s

module {
  wafer.placement.map
      {bad_tile_ids = array<i64>,
       card_x_count = 1 : i64,
       card_y_count = 1 : i64,
       logical_rank_count = 4 : i64,
       physical_tile_coords = array<i64: 0, 0, 0, 0,
                                      0, 0, 0, 1,
                                      0, 0, 0, 2,
                                      0, 0, 0, 3>,
       tile_x_count = 4 : i64,
       tile_y_count = 4 : i64}

  %source = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %0 = wafer.tile_region(%source : tensor<4xf32>) -> (tensor<4xf32>) {
  ^bb0(%arg0: tensor<4xf32>):
    %local = "builtin.unrealized_conversion_cast"()
        : () -> !wafer.tile_buffer<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
    %gather = "builtin.unrealized_conversion_cast"()
        : () -> !wafer.tile_buffer<tensor<8xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
    wafer.comm.all_gather %local into %gather
        {bytes = 16 : i64, group_size = 2 : i64, local_rank = 0 : i64,
         rank_group = array<i64: 1, 3>}
        : !wafer.tile_buffer<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
        -> !wafer.tile_buffer<tensor<8xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
    wafer.tile_yield %arg0 : tensor<4xf32>
  }
}

// CHECK-LABEL: module
// CHECK-NOT: wafer.comm.all_gather
// CHECK: wafer.comm.send %{{.*}} {bytes = 16 : i64, peer = 3 : i64, slot = 0 : i64}
// CHECK: wafer.comm.recv %{{.*}} {bytes = 16 : i64, peer = 3 : i64, slot = 1 : i64}
