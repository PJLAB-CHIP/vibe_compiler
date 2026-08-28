// RUN: wafer-opt --wafer-strip-target-metadata %s | FileCheck %s

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card_partition"], shape = array<i64: 1>}

  func.func @keep() {
    return
  }
}

// CHECK-NOT: wafer.target.topology
// CHECK-NOT: wafer.execution.mesh
// CHECK: func.func @keep
