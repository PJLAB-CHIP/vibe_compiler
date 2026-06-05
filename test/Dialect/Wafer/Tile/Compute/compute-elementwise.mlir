// RUN: wafer-opt %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>

module {
  %a = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %b = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %row = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.storage<tensor<4xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %bool = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.storage<tensor<4x8xi1>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %sum = wafer.tile.elementwise #wafer.elementwise_kind<add> %a, %b
      : (!wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>,
         !wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>)
     -> !wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %exp = wafer.tile.elementwise #wafer.elementwise_kind<exp> %sum
      : (!wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>)
     -> !wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %row_add = wafer.tile.elementwise #wafer.elementwise_kind<add> %a, %row
      {indexing_maps = [#map, #row, #map]}
      : (!wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>,
         !wafer.storage<tensor<4xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>)
     -> !wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %gt = wafer.tile.elementwise #wafer.elementwise_kind<gt> %a, %b
      : (!wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>,
         !wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>)
     -> !wafer.storage<tensor<4x8xi1>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
}

// CHECK: wafer.tile.elementwise <add>
// CHECK: wafer.tile.elementwise <exp>
// CHECK: wafer.tile.elementwise <add>
// CHECK-SAME: indexing_maps
// CHECK: wafer.tile.elementwise <gt>
