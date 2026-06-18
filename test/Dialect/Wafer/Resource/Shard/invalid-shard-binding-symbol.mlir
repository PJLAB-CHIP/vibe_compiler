// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  wafer.shard.binding
      {argument_index = 0 : i64,
       global_shape = array<i64: 2, 4>,
       kernel = @missing,
       local_shape = array<i64: 1, 4>,
       logical_rank_count = 1 : i64,
       shard_offsets = array<i64: 0, 0>,
       shard_ranks = array<i64: 0>,
       shard_sizes = array<i64: 1, 4>,
       shard_strides = array<i64: 1, 1>}
}

// CHECK: references unknown function symbol @missing
