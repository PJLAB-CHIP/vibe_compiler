// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %source = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.tile_buffer<tensor<4x16xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
  %dest = "builtin.unrealized_conversion_cast"() : () -> tensor<4x16xf16>
  wafer.store_tile %source, %dest
      : !wafer.tile_buffer<tensor<4x16xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
     -> tensor<4x16xf16>
}

// CHECK: store_tile source must use tensor mem_layout for external writeback
