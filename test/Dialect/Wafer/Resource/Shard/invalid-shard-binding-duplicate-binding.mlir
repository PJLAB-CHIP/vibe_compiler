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
}

// CHECK: duplicates shard binding for referenced function argument
