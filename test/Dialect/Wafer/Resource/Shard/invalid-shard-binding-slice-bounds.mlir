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
       logical_rank_count = 1 : i64,
       shard_offsets = array<i64: 2, 0>,
       shard_ranks = array<i64: 0>,
       shard_sizes = array<i64: 1, 4>,
       shard_strides = array<i64: 1, 1>}
}

// CHECK: shard slice for logical rank 0 exceeds global shape
