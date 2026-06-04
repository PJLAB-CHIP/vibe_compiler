// RUN: wafer-opt %s | FileCheck %s

module {
  %spm = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.tile_buffer<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %ddr = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.tile_buffer<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<ddr>>
}

// CHECK: !wafer.tile_buffer<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
// CHECK: !wafer.tile_buffer<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<ddr>>
