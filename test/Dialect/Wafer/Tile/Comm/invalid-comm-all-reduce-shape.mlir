// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf32, #wafer.memory<spm, tensor>>
  %recv = "builtin.unrealized_conversion_cast"()
      : () -> memref<8xf32, #wafer.memory<spm, tensor>>
  %result = wafer.tile.all_reduce #wafer.reduce_kind<sum> %input using %recv
      {bytes = 16 : i64, group_size = 4 : i64, local_rank = 0 : i64,
       rank_group = array<i64: 0, 1, 2, 3>}
      : (memref<4xf32, #wafer.memory<spm, tensor>>,
         memref<8xf32, #wafer.memory<spm, tensor>>)
     -> memref<4xf32, #wafer.memory<spm, tensor>>
}

// CHECK: all_reduce input, recv buffer, and result types must match
