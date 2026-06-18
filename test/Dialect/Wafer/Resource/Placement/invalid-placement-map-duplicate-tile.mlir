// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  wafer.placement.map
      {bad_tile_ids = array<i64>,
       block_ids = array<i64: 0, 1>,
       card_x_count = 1 : i64,
       card_y_count = 1 : i64,
       logical_rank_count = 2 : i64,
       physical_tile_coords = array<i64: 0, 0, 0, 0,
                                      0, 0, 0, 0>,
       tile_x_count = 4 : i64,
       tile_y_count = 4 : i64}
}

// CHECK: maps multiple logical ranks to physical tile id 0
