// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  func.func @forward(%arg0: tensor<1x4xf32>) {
    return
  }

  wafer.shard.binding
      {argument_index = 0 : i64,
       execution_mesh = @missing_mesh,
       global_shape = array<i64: 1, 4>,
       kernel = @forward,
       local_shape = array<i64: 1, 4>,
       shard_offsets = array<i64: 0, 0>,
       shard_ranks = array<i64: 0>,
       shard_sizes = array<i64: 1, 4>,
       shard_strides = array<i64: 1, 1>}
}

// CHECK: references unknown execution mesh symbol @missing_mesh
