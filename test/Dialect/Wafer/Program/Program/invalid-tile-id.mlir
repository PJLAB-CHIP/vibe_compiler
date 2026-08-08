// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  wafer.target.topology @target
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>,
       unavailable_tiles = array<i64>}
  wafer.card.program card_id = 0 {
    wafer.tile.program tile_id = 1 {}
  }
}

// CHECK: tile_id 1 is outside target Tile grid [0, 1)
