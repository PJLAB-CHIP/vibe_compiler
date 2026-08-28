// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  func.func @wrong_scope() {
    wafer.tile.module card_id = 0 tile_id = 0 {}
    return
  }
}

// CHECK: 'wafer.tile.module' op must be directly nested under a builtin.module
