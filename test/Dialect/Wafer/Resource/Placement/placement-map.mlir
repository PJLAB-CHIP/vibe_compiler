// RUN: wafer-opt %s | FileCheck %s

module {
  wafer.placement.map
      {bad_tile_ids = array<i64>,
       block_ids = array<i64: 0, 1, 2, 3>,
       card_x_count = 1 : i64,
       card_y_count = 1 : i64,
       logical_rank_count = 4 : i64,
       physical_tile_coords = array<i64: 0, 0, 0, 0,
                                      0, 0, 0, 1,
                                      0, 0, 0, 2,
                                      0, 0, 0, 3>,
       tile_x_count = 4 : i64,
       tile_y_count = 4 : i64}
}

// CHECK: wafer.placement.map
// CHECK-SAME: block_ids = array<i64: 0, 1, 2, 3>
// CHECK-SAME: logical_rank_count = 4 : i64
// CHECK-SAME: physical_tile_coords = array<i64: 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3>
