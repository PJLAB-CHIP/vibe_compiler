// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>,
       unavailable_tiles = array<i64: 0, 0, 0, 1>}

  wafer.execution.mesh @mesh
      {topology = @default,
       axes = ["rank"],
       shape = array<i64: 1>,
       policy = "explicit",
       endpoints = array<i64: 0, 0, 0, 1>}
}

// CHECK: explicit endpoint references unavailable tile coordinate
