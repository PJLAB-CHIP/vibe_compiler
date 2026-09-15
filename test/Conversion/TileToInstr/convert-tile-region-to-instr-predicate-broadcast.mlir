// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %s | FileCheck %s

#id2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#column = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @bitpacked_predicate_broadcast() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %predicate = "builtin.unrealized_conversion_cast"()
      : () -> memref<2x1025xi1, #wafer.memory<spm, tensor>>
  %true_value = "builtin.unrealized_conversion_cast"()
      : () -> memref<2x1025x16xf32, #wafer.memory<spm, tensor>>
  %false_value = "builtin.unrealized_conversion_cast"()
      : () -> memref<2x1025x16xf32, #wafer.memory<spm, tensor>>
  %selected = wafer.tile.elementwise #wafer.elementwise_kind<select>
      %predicate, %true_value, %false_value
      {indexing_maps = [#column, #id2, #id2, #id2]}
      : (memref<2x1025xi1, #wafer.memory<spm, tensor>>,
         memref<2x1025x16xf32, #wafer.memory<spm, tensor>>,
         memref<2x1025x16xf32, #wafer.memory<spm, tensor>>)
     -> memref<2x1025x16xf32, #wafer.memory<spm, tensor>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK: wafer.instr.bit2fp
// CHECK: wafer.instr.gather_scatter
// CHECK: wafer.instr.mask_move
