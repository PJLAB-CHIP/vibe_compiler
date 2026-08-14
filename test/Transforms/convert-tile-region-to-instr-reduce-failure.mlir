// RUN: not wafer-opt --mlir-print-ir-after-failure --split-input-file --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %s 2>&1 | FileCheck %s

func.func @reject_avg() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %input = memref.alloc() : memref<2x2xf32, #wafer.memory<spm, cx>>
  %result = wafer.tile.reduce #wafer.reduce_kind<avg> %input
      {dimensions = array<i64: 1>, init_value = 0.000000e+00 : f32}
      : (memref<2x2xf32, #wafer.memory<spm, cx>>)
     -> memref<2xf32, #wafer.memory<spm, cx>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK: tile-region to instruction conversion failed
// CHECK: IR Dump After ConvertTileRegionToInstrPass Failed
// CHECK: wafer.tile.reduce <avg>
// CHECK-NOT: wafer.instr.

// -----

func.func @reject_unencodable_dtype() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %input = memref.alloc() : memref<2x2xi4, #wafer.memory<spm, cx>>
  %result = wafer.tile.reduce #wafer.reduce_kind<sum> %input
      {dimensions = array<i64: 1>, init_value = 0 : i4}
      : (memref<2x2xi4, #wafer.memory<spm, cx>>)
     -> memref<2xi4, #wafer.memory<spm, cx>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK: tile-region to instruction conversion failed
// CHECK: IR Dump After ConvertTileRegionToInstrPass Failed
// CHECK: wafer.tile.reduce <sum>
// CHECK-NOT: wafer.instr.

// -----

func.func @reject_bitpacked() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %input = memref.alloc() : memref<2x2xi1, #wafer.memory<spm, cx>>
  %result = wafer.tile.reduce #wafer.reduce_kind<sum> %input
      {dimensions = array<i64: 1>, init_value = false}
      : (memref<2x2xi1, #wafer.memory<spm, cx>>)
     -> memref<2xi1, #wafer.memory<spm, cx>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK: tile-region to instruction conversion failed
// CHECK: IR Dump After ConvertTileRegionToInstrPass Failed
// CHECK: wafer.tile.reduce <sum>
// CHECK-NOT: wafer.instr.
