// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %a = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.tile_buffer<tensor<4x7xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
  %b = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.tile_buffer<tensor<8x16xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
  %0 = wafer.compute.gemm %a, %b
      : (!wafer.tile_buffer<tensor<4x7xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>,
         !wafer.tile_buffer<tensor<8x16xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>)
     -> !wafer.tile_buffer<tensor<4x16xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
}

// CHECK: gemm lhs K dimension must match rhs K dimension
