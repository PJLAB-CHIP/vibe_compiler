// RUN: wafer-opt %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>

module {
  %a = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<spm, tensor>>
  %b = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<spm, tensor>>
  %row = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf16, #wafer.memory<spm, tensor>>
  %bool = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xi1, #wafer.memory<spm, tensor>>
  %sum = wafer.tile.elementwise #wafer.elementwise_kind<add> %a, %b
      : (memref<4x8xf16, #wafer.memory<spm, tensor>>,
         memref<4x8xf16, #wafer.memory<spm, tensor>>)
     -> memref<4x8xf16, #wafer.memory<spm, tensor>>
  %exp = wafer.tile.elementwise #wafer.elementwise_kind<exp> %sum
      : (memref<4x8xf16, #wafer.memory<spm, tensor>>)
     -> memref<4x8xf16, #wafer.memory<spm, tensor>>
  %row_add = wafer.tile.elementwise #wafer.elementwise_kind<add> %a, %row
      {indexing_maps = [#map, #row, #map]}
      : (memref<4x8xf16, #wafer.memory<spm, tensor>>,
         memref<4xf16, #wafer.memory<spm, tensor>>)
     -> memref<4x8xf16, #wafer.memory<spm, tensor>>
  %gt = wafer.tile.elementwise #wafer.elementwise_kind<gt> %a, %b
      : (memref<4x8xf16, #wafer.memory<spm, tensor>>,
         memref<4x8xf16, #wafer.memory<spm, tensor>>)
     -> memref<4x8xi1, #wafer.memory<spm, tensor>>
}

// CHECK: wafer.tile.elementwise <add>
// CHECK: wafer.tile.elementwise <exp>
// CHECK: wafer.tile.elementwise <add>
// CHECK-SAME: indexing_maps
// CHECK: wafer.tile.elementwise <gt>
