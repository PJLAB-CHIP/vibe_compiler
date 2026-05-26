// RUN: wafer-opt %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>

module {
  %a = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.tile_buffer<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %b = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.tile_buffer<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %row = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.tile_buffer<tensor<4xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %sum = wafer.compute.elementwise #wafer.elementwise_kind<add> %a, %b
      : (!wafer.tile_buffer<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>,
         !wafer.tile_buffer<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>)
     -> !wafer.tile_buffer<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %exp = wafer.compute.elementwise #wafer.elementwise_kind<exp> %sum
      : (!wafer.tile_buffer<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>)
     -> !wafer.tile_buffer<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %row_add = wafer.compute.elementwise #wafer.elementwise_kind<add> %a, %row
      {indexing_maps = [#map, #row, #map]}
      : (!wafer.tile_buffer<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>,
         !wafer.tile_buffer<tensor<4xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>)
     -> !wafer.tile_buffer<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
}

// CHECK: wafer.compute.elementwise <add>
// CHECK: wafer.compute.elementwise <exp>
// CHECK: wafer.compute.elementwise <add>
// CHECK-SAME: indexing_maps
