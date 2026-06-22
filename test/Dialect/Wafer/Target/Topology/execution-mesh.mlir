// RUN: wafer-opt %s | FileCheck %s

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>,
       unavailable_tiles = array<i64>}

  wafer.execution.mesh @default_mesh
      {topology = @default,
       axes = ["rank"],
       shape = array<i64: 16>,
       policy = "all_available",
       endpoints = array<i64>}

  wafer.execution.mesh @debug_mesh
      {topology = @default,
       axes = ["y", "x"],
       shape = array<i64: 2, 2>,
       policy = "explicit",
       endpoints = array<i64: 0, 0, 0, 0,
                               0, 0, 0, 1,
                               0, 0, 1, 0,
                               0, 0, 1, 1>}
}

// CHECK: wafer.execution.mesh @default_mesh
// CHECK-SAME: axes = ["rank"]
// CHECK-SAME: endpoints = array<i64>
// CHECK-SAME: policy = "all_available"
// CHECK-SAME: shape = array<i64: 16>
// CHECK-SAME: topology = @default
// CHECK: wafer.execution.mesh @debug_mesh
// CHECK-SAME: axes = ["y", "x"]
// CHECK-SAME: endpoints = array<i64: 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1>
// CHECK-SAME: policy = "explicit"
// CHECK-SAME: shape = array<i64: 2, 2>
// CHECK-SAME: topology = @default
