// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  wafer.placement.map
      {bad_tile_ids = array<i64>,
       block_ids = array<i64: 0, 1>,
       card_x_count = 1 : i64,
       card_y_count = 1 : i64,
       logical_rank_count = 2 : i64,
       physical_tile_coords = array<i64: 0, 0, 0, 0,
                                      0, 0, 0, 1>,
       tile_x_count = 4 : i64,
       tile_y_count = 4 : i64}

  %buf = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf32, #wafer.memory<spm, tensor>>
  %send = wafer.tile.send %buf {peer = 2 : i64, bytes = 16 : i64}
      : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
}

// CHECK: comm peer must refer to an active placement tile
