// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %bad = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.tile_buffer<tensor<*xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
}

// CHECK: tile_buffer logical type must be a ranked tensor
