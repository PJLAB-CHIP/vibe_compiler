// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>,
       unavailable_tiles = array<i64: 0, 0, 0, 1>}
  wafer.card.program card_id = 0 {
    wafer.tile.program tile_id = 0 {}
    wafer.tile.program tile_id = 1 {}
  }
}

// CHECK: tile_id 1 is unavailable for card_id 0
