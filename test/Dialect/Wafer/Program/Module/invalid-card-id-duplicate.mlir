// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>,
       unavailable_tiles = array<i64>}
  wafer.card.module card_id = 0 {
    wafer.tile.module tile_id = 0 {}
  }
  wafer.card.module card_id = 0 {
    wafer.tile.module tile_id = 0 {}
  }
}

// CHECK: card_id must be unique in its module; duplicate 0
