// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  wafer.card.program card_id = 0 {
    wafer.tile.program tile_id = 0 {}
  }
}

// CHECK: expected exactly one direct wafer.target.topology in source module
