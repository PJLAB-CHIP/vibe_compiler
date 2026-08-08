// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  wafer.tile.program tile_id = 0 {}
}

// CHECK: 'wafer.tile.program' op expects parent op 'wafer.card.program'
