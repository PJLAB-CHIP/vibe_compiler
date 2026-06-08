// RUN: not wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %s 2>&1 | FileCheck %s

func.func @reject_transpose(
    %input: memref<4x4xf16, #wafer.memory<ddr, tensor>>,
    %output: memref<4x4xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%input, %output
      : memref<4x4xf16, #wafer.memory<ddr, tensor>>,
        memref<4x4xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<4x4xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%in: memref<4x4xf16, #wafer.memory<ddr, tensor>>,
       %out: memref<4x4xf16, #wafer.memory<ddr, tensor>>):
    %loaded = wafer.tile.load %in
        : memref<4x4xf16, #wafer.memory<ddr, tensor>>
       -> memref<4x4xf16, #wafer.memory<spm, tensor>>
    %transposed = wafer.tile.transpose %loaded
        {permutation = array<i64: 1, 0>}
        : memref<4x4xf16, #wafer.memory<spm, tensor>>
       -> memref<4x4xf16, #wafer.memory<spm, tensor>>
    wafer.tile.store %transposed, %out
        : memref<4x4xf16, #wafer.memory<spm, tensor>>
       -> memref<4x4xf16, #wafer.memory<ddr, tensor>>
    wafer.tile.yield %out
        : memref<4x4xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK: wafer.tile.transpose cannot lower to TDMA-backed wafer.instr.gather_scatter: unsupported permutation
