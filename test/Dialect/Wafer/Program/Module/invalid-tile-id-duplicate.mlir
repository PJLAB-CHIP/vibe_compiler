// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>,
       unavailable_tiles = array<i64>}
  wafer.card.module card_id = 0 {
    wafer.tile.module tile_id = 0 {}
    wafer.tile.module tile_id = 0 {}
  }
}

// CHECK: tile_id must be unique in a card module; duplicate 0
