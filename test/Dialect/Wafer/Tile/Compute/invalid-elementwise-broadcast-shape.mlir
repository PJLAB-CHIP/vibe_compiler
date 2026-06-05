// RUN: not wafer-opt %s 2>&1 | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#col = affine_map<(d0, d1) -> (d1)>

module {
  %a = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<spm, tensor>>
  %b = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf16, #wafer.memory<spm, tensor>>
  %sum = wafer.tile.elementwise #wafer.elementwise_kind<add> %a, %b
      {indexing_maps = [#map, #col, #map]}
      : (memref<4x8xf16, #wafer.memory<spm, tensor>>,
         memref<4xf16, #wafer.memory<spm, tensor>>)
     -> memref<4x8xf16, #wafer.memory<spm, tensor>>
}

// CHECK: elementwise indexing map dimension must match tensor shape
