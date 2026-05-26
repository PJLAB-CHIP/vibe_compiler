// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %source = "builtin.unrealized_conversion_cast"() : () -> tensor<4x8xf16>
  %0 = wafer.load_tile %source
      : tensor<4x8xf16>
     -> !wafer.tile_buffer<tensor<4x16xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
}

// CHECK: load_tile result tensor type must match source tensor type
