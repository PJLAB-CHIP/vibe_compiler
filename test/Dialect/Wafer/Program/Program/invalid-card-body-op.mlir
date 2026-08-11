// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  wafer.target.topology @target {
    card_grid = array<i64: 1, 1>,
    card_interconnect = "mesh",
    tile_grid = array<i64: 1, 1>,
    unavailable_tiles = array<i64>
  }
  wafer.card.program card_id = 0 {
    // expected-error @+1 {{body may contain only card-shared declarations and wafer.tile.program operations}}
    %value = arith.constant 1 : i32
    wafer.tile.program tile_id = 0 {}
  }
}

// CHECK: body may contain only card-shared declarations and wafer.tile.program operations
