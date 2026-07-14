// RUN: not wafer-opt --mlir-disable-threading --wafer-convert-tile-region-to-instr \
// RUN:   --mlir-print-ir-after-failure --mlir-print-ir-module-scope -o /dev/null %s 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=wafer.instr

#id2 = affine_map<(d0, d1) -> (d0, d1)>
#column = affine_map<(d0, d1) -> (d0)>

func.func @reject_bitpacked_predicate_broadcast() {
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
  return
}

// CHECK: tile.elementwise indexing map materialization requires byte-addressable elements
// CHECK: wafer.tile.elementwise
