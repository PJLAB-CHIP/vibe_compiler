// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  func.func @forward(%arg0: tensor<1x4xf32>) {
    return
  }

  wafer.shard.binding
      {argument_index = 0 : i64,
       global_shape = array<i64: 2, 4>,
       kernel = @forward,
       local_shape = array<i64: 1, 4>,
       logical_rank_count = 2 : i64,
       shard_offsets = array<i64: 0, 0, 1, 0>,
       shard_ranks = array<i64: 0, 1>,
       shard_sizes = array<i64: 1, 4, 1, 4>,
       shard_strides = array<i64: 1, 1, 1, 1>}

  wafer.placement.map
      {bad_tile_ids = array<i64>,
       block_ids = array<i64: 0>,
       card_x_count = 1 : i64,
       card_y_count = 1 : i64,
       logical_rank_count = 1 : i64,
       physical_tile_coords = array<i64: 0, 0, 0, 0>,
       tile_x_count = 4 : i64,
       tile_y_count = 4 : i64}
}

// CHECK: shard binding logical rank count 2 does not match placement map logical rank count 1
