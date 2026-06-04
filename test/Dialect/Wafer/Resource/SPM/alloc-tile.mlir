// RUN: wafer-opt %s | FileCheck %s

module {
  %0 = wafer.alloc_tile
      : !wafer.tile_buffer<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
}

// CHECK: wafer.alloc_tile
