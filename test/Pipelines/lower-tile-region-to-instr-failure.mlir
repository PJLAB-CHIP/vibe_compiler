// RUN: not wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %s 2>&1 | FileCheck %s

func.func @reject_unsupported_reduce_layout(
    %boundary: memref<4xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(
      %boundary : memref<4xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<4xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<4xf16, #wafer.memory<ddr, tensor>>):
    %input = memref.alloc() : memref<2x2xf32, #wafer.memory<spm, cx>>
    %result = wafer.tile.reduce <avg> %input
        {dimensions = array<i64: 1>, init_value = 0.000000e+00 : f32}
        : (memref<2x2xf32, #wafer.memory<spm, cx>>)
       -> memref<2xf32, #wafer.memory<spm, cx>>
    wafer.tile.yield %arg0 : memref<4xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK: tile-region to instruction conversion failed
