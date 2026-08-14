// RUN: not wafer-opt --mlir-disable-threading --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' \
// RUN:   --mlir-print-ir-after-failure --mlir-print-ir-module-scope -o /dev/null %s 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=wafer.instr

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#column = affine_map<(d0, d1) -> (d0)>

func.func @reject_bitpacked_predicate_broadcast() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %predicate = "builtin.unrealized_conversion_cast"()
      : () -> memref<2xi1, #wafer.memory<spm, tensor>>
  %true_value = "builtin.unrealized_conversion_cast"()
      : () -> memref<2x3xf32, #wafer.memory<spm, tensor>>
  %false_value = "builtin.unrealized_conversion_cast"()
      : () -> memref<2x3xf32, #wafer.memory<spm, tensor>>
  %selected = wafer.tile.elementwise #wafer.elementwise_kind<select>
      %predicate, %true_value, %false_value
      {indexing_maps = [#column, #id2, #id2, #id2]}
      : (memref<2xi1, #wafer.memory<spm, tensor>>,
         memref<2x3xf32, #wafer.memory<spm, tensor>>,
         memref<2x3xf32, #wafer.memory<spm, tensor>>)
     -> memref<2x3xf32, #wafer.memory<spm, tensor>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK: tile-region to instruction conversion failed
// CHECK: wafer.tile.elementwise
