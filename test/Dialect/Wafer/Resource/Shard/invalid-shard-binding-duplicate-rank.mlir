// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>,
       card_interconnect = "mesh",
       tile_grid = array<i64: 1, 2>,
       unavailable_tiles = array<i64>}

  wafer.execution.mesh @default_mesh
      {topology = @default,
       axes = ["rank"],
       shape = array<i64: 2>,
       policy = "explicit",
       endpoints = array<i64: 0, 0, 0, 0, 0, 0, 0, 1>}

  func.func @forward(%arg0: tensor<1x4xf32>) {
    return
  }

  wafer.shard.binding
      {argument_index = 0 : i64,
       execution_mesh = @default_mesh,
       global_shape = array<i64: 2, 4>,
       kernel = @forward,
       local_shape = array<i64: 1, 4>,
       shard_offsets = array<i64: 0, 0, 1, 0>,
       shard_ranks = array<i64: 0, 0>,
       shard_sizes = array<i64: 1, 4, 1, 4>,
       shard_strides = array<i64: 1, 1, 1, 1>}
}

// CHECK: maps multiple shard slices to logical rank 0
