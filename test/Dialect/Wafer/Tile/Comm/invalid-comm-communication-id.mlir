// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %local = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf32, #wafer.memory<spm, tensor>>
  %gather = "builtin.unrealized_conversion_cast"()
      : () -> memref<8xf32, #wafer.memory<spm, tensor>>
  wafer.tile.all_gather %local into %gather
      {bytes = 16 : i64, group_size = 2 : i64, local_rank = 0 : i64,
       rank_group = array<i64: 0, 1>, communication_id = -1 : i64}
      : memref<4xf32, #wafer.memory<spm, tensor>>
     -> memref<8xf32, #wafer.memory<spm, tensor>>
}

// CHECK: communication_id must be non-negative
