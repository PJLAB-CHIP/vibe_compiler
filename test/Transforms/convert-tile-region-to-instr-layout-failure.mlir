// RUN: not wafer-opt --wafer-convert-tile-region-to-instr %s 2>&1 | FileCheck %s

func.func @reject_padded_layout_materialize(
    %boundary: memref<4x8xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<4x8xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<4x8xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<4x8xf16, #wafer.memory<ddr, tensor>>):
    %loaded = wafer.tile.load %arg0
        : memref<4x8xf16, #wafer.memory<ddr, tensor>>
       -> memref<4x8xf16, #wafer.memory<spm, tensor>>
    %cx = wafer.tile.materialize_layout %loaded
        : memref<4x8xf16, #wafer.memory<spm, tensor>>
       -> memref<4x8xf16, #wafer.memory<spm, cx>>
    wafer.tile.yield %arg0
        : memref<4x8xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK: layout materialize lowering requires equal static physical byte counts
