// RUN: not wafer-opt --mlir-print-ir-after-failure --split-input-file --wafer-convert-tile-region-to-instr %s 2>&1 | FileCheck %s

func.func @reject_avg(
    %input: memref<2x2xf32, #wafer.memory<spm, cx>>)
    -> memref<2xf32, #wafer.memory<spm, cx>> {
  %result = wafer.tile.reduce #wafer.reduce_kind<avg> %input
      {dimensions = array<i64: 1>, init_value = 0.000000e+00 : f32}
      : (memref<2x2xf32, #wafer.memory<spm, cx>>)
     -> memref<2xf32, #wafer.memory<spm, cx>>
  return %result : memref<2xf32, #wafer.memory<spm, cx>>
}

// CHECK: tile.reduce lowering does not support avg accumulation
// CHECK: IR Dump After ConvertTileRegionToInstrPass Failed
// CHECK: func.func @reject_avg
// CHECK: wafer.tile.reduce <avg>
// CHECK-NOT: wafer.instr.

// -----

func.func @reject_static_terminal_budget(
    %input: memref<1024x1xf32, #wafer.memory<spm, cx>>)
    -> memref<1xf32, #wafer.memory<spm, cx>> {
  %result = wafer.tile.reduce #wafer.reduce_kind<sum> %input
      {dimensions = array<i64: 0>, init_value = 1.000000e+00 : f32}
      : (memref<1024x1xf32, #wafer.memory<spm, cx>>)
     -> memref<1xf32, #wafer.memory<spm, cx>>
  return %result : memref<1xf32, #wafer.memory<spm, cx>>
}

// CHECK: static_terminal_budget_exceeded: ordered tile.reduce minimum terminal operation count exceeds 4096
// CHECK: IR Dump After ConvertTileRegionToInstrPass Failed
// CHECK: func.func @reject_static_terminal_budget
// CHECK: wafer.tile.reduce <sum>
// CHECK-NOT: wafer.instr.

// -----

func.func @reject_unencodable_dtype(
    %input: memref<2x2xi4, #wafer.memory<spm, cx>>)
    -> memref<2xi4, #wafer.memory<spm, cx>> {
  %result = wafer.tile.reduce #wafer.reduce_kind<sum> %input
      {dimensions = array<i64: 1>, init_value = 0 : i4}
      : (memref<2x2xi4, #wafer.memory<spm, cx>>)
     -> memref<2xi4, #wafer.memory<spm, cx>>
  return %result : memref<2xi4, #wafer.memory<spm, cx>>
}

// CHECK: tile.reduce element type is not encodable by the target data-format ABI
// CHECK: IR Dump After ConvertTileRegionToInstrPass Failed
// CHECK: func.func @reject_unencodable_dtype
// CHECK: wafer.tile.reduce <sum>
// CHECK-NOT: wafer.instr.

// -----

func.func @reject_bitpacked(
    %input: memref<2x2xi1, #wafer.memory<spm, cx>>)
    -> memref<2xi1, #wafer.memory<spm, cx>> {
  %result = wafer.tile.reduce #wafer.reduce_kind<sum> %input
      {dimensions = array<i64: 1>, init_value = false}
      : (memref<2x2xi1, #wafer.memory<spm, cx>>)
     -> memref<2xi1, #wafer.memory<spm, cx>>
  return %result : memref<2xi1, #wafer.memory<spm, cx>>
}

// CHECK: tile.reduce lowering requires byte-addressable elements
// CHECK: IR Dump After ConvertTileRegionToInstrPass Failed
// CHECK: func.func @reject_bitpacked
// CHECK: wafer.tile.reduce <sum>
// CHECK-NOT: wafer.instr.
