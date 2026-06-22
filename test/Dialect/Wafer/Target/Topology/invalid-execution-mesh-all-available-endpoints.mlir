// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>,
       unavailable_tiles = array<i64>}

  wafer.execution.mesh @mesh
      {topology = @default,
       axes = ["rank"],
       shape = array<i64: 16>,
       policy = "all_available",
       endpoints = array<i64: 0, 0, 0, 0>}
}

// CHECK: all_available execution mesh must not carry explicit endpoints
