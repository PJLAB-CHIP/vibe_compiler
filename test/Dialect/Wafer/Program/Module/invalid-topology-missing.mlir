// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  wafer.card.module card_id = 0 {
    wafer.tile.module tile_id = 0 {}
  }
}

// CHECK: expected exactly one direct wafer.target.topology in source module
