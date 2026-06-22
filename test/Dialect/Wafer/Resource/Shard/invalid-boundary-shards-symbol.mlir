// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 1, 1>,
       unavailable_tiles = array<i64>}

  wafer.execution.mesh @default_mesh
      {topology = @default,
       axes = ["rank"],
       shape = array<i64: 1>,
       policy = "explicit",
       endpoints = array<i64: 0, 0, 0, 0>}

  func.func @forward(%arg0: tensor<1x4xf32> {wafer.boundary_shards = @missing_shards}) {
    return
  }
}

// CHECK: argument #0 attribute 'wafer.boundary_shards' references unknown wafer.boundary.shards symbol @missing_shards
