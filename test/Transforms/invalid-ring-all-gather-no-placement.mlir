// RUN: not wafer-opt --wafer-lower-ring-all-gather %s 2>&1 | FileCheck %s

module {
  %source = "builtin.unrealized_conversion_cast"() : () -> tensor<4xf32>
  %0 = wafer.tile_region(%source : tensor<4xf32>) -> (tensor<4xf32>) {
  ^bb0(%arg0: tensor<4xf32>):
    %local = "builtin.unrealized_conversion_cast"()
        : () -> !wafer.tile_buffer<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
    %gather = "builtin.unrealized_conversion_cast"()
        : () -> !wafer.tile_buffer<tensor<16xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
    wafer.comm.all_gather %local into %gather
        {bytes = 16 : i64, group_size = 4 : i64, local_rank = 0 : i64}
        : !wafer.tile_buffer<tensor<4xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
        -> !wafer.tile_buffer<tensor<16xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
    wafer.tile_yield %arg0 : tensor<4xf32>
  }
}

// CHECK: ring all-gather lowering requires exactly one placement map
