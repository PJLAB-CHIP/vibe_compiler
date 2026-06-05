// RUN: not wafer-opt %s 2>&1 | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#col = affine_map<(d0, d1) -> (d1)>

module {
  %a = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %b = "builtin.unrealized_conversion_cast"()
      : () -> !wafer.storage<tensor<4xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
  %sum = wafer.compute.elementwise #wafer.elementwise_kind<add> %a, %b
      {indexing_maps = [#map, #col, #map]}
      : (!wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>,
         !wafer.storage<tensor<4xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>)
     -> !wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
}

// CHECK: elementwise indexing map dimension must match tensor shape
