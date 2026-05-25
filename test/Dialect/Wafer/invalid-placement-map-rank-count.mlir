// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  wafer.placement.map
      {bad_tile_ids = array<i64>,
       card_x_count = 1 : i64,
       card_y_count = 1 : i64,
       logical_rank_count = 4 : i64,
       physical_tile_coords = array<i64: 0, 0, 0, 0,
                                      0, 0, 0, 1,
                                      0, 0, 0, 2>,
       tile_x_count = 4 : i64,
       tile_y_count = 4 : i64}
}

// CHECK: physical tile mapping must contain one 4D coordinate per logical rank
