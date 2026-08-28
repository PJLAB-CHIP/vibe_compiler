// RUN: wafer-opt %s | FileCheck %s

module {
  wafer.target.topology @default
      {card_grid = array<i64: 4, 8>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>,
       unavailable_tiles = array<i64: 0, 0, 0, 1,
                                      0, 0, 1, 0>}
}

// CHECK: wafer.target.topology @default
// CHECK-SAME: card_grid = array<i64: 4, 8>
// CHECK-SAME: card_interconnect = "mesh"
// CHECK-SAME: tile_grid = array<i64: 4, 4>
// CHECK-SAME: unavailable_tiles = array<i64: 0, 0, 0, 1, 0, 0, 1, 0>
