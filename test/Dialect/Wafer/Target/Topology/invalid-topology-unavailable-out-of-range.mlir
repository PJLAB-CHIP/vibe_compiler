// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  wafer.target.topology @default
      {card_grid = array<i64: 4, 8>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>,
       unavailable_tiles = array<i64: 0, 0, 4, 0>}
}

// CHECK: unavailable_tiles tile_y coordinate 4 is outside tile_grid
