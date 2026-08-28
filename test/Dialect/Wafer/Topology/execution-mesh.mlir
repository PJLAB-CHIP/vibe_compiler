// RUN: wafer-opt %s | FileCheck %s

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>,
       unavailable_tiles = array<i64>}

  wafer.execution.mesh @default_mesh
      {axes = ["card_partition"],
       shape = array<i64: 1>}

  wafer.execution.mesh @debug_mesh
      {axes = ["y", "x"],
       shape = array<i64: 2, 2>}
}

// CHECK: wafer.execution.mesh @default_mesh
// CHECK-SAME: axes = ["card_partition"]
// CHECK-SAME: shape = array<i64: 1>
// CHECK: wafer.execution.mesh @debug_mesh
// CHECK-SAME: axes = ["y", "x"]
// CHECK-SAME: shape = array<i64: 2, 2>
