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

  func.func @forward(%arg0: tensor<1x4xf32> {wafer.boundary_shards = @arg0_shards}) {
    return
  }

  wafer.boundary.shards @arg0_shards
      {execution_mesh = @default_mesh,
       global_shape = array<i64: 2, 4>,
       local_shape = array<i64: 1, 4>,
       shard_offsets = array<i64: 2, 0>,
       shard_ranks = array<i64: 0>,
       shard_sizes = array<i64: 1, 4>,
       shard_strides = array<i64: 1, 1>}
}

// CHECK: shard slice for logical rank 0 exceeds global shape
