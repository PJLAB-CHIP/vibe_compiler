// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  wafer.tile.module tile_id = 0 {}
}

// CHECK: 'wafer.tile.module' op expects parent op 'wafer.card.module'
