// RUN: wafer-opt %s -split-input-file -verify-diagnostics

module {
  // expected-error @below {{'wafer.alloc_tile' op result must use SPM memory space}}
  %0 = wafer.alloc_tile
      : !wafer.tile_buffer<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<ddr>>
}
