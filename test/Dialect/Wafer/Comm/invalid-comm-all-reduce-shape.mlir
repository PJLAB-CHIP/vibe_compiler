// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.tile_buffer<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %recv = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.tile_buffer<tensor<8xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %result = wafer.comm.all_reduce #wafer.reduce_kind<sum> %input using %recv
      {bytes = 16 : i64, group_size = 4 : i64, local_rank = 0 : i64}
      : (!wafer.tile_buffer<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>,
         !wafer.tile_buffer<tensor<8xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>)
     -> !wafer.tile_buffer<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
}

// CHECK: all_reduce input, recv buffer, and result types must match
