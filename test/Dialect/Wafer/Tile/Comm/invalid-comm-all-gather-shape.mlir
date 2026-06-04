// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %local = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.tile_buffer<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %gather = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.tile_buffer<tensor<8xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  wafer.comm.all_gather %local into %gather
      {bytes = 16 : i64, group_size = 4 : i64, local_rank = 0 : i64,
       rank_group = array<i64: 0, 1, 2, 3>}
      : !wafer.tile_buffer<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
      -> !wafer.tile_buffer<tensor<8xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
}

// CHECK: all_gather gather buffer compact byte size must equal bytes times group_size
