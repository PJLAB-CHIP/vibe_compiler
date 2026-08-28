// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  wafer.target.topology @default
      {card_grid = array<i64: 4, 8>,
       card_interconnect = "ring",
       tile_grid = array<i64: 4, 4>,
       unavailable_tiles = array<i64>}
}

// CHECK: card_interconnect must be mesh or torus
