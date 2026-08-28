// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  wafer.target.topology @default
      {card_grid = array<i64: 4, 8>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>,
       unavailable_tiles = array<i64: 0, 0, 1>}
}

// CHECK: unavailable_tiles must contain card_y/card_x/tile_y/tile_x tuples
