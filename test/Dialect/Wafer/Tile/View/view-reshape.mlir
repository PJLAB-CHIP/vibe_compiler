// RUN: wafer-opt %s | FileCheck %s

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.tile_buffer<tensor<6xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %reshaped = wafer.view.reshape %input
      : !wafer.tile_buffer<tensor<6xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
     -> !wafer.tile_buffer<tensor<2x3xf32>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
}

// CHECK: wafer.view.reshape
