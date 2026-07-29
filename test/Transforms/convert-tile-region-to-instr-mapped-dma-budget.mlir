// RUN: not wafer-opt --wafer-convert-tile-region-to-instr %s 2>&1 | FileCheck %s

func.func @reject_large_identity_tensor_to_cx_mapping(
    %input: memref<1024x4096xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%input
      : memref<1024x4096xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<1024x4096xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<1024x4096xf16, #wafer.memory<ddr, tensor>>):
    %cx = memref.alloc()
        : memref<1024x4096xf16, #wafer.memory<spm, cx>>
    wafer.tile.load %arg0 into %cx
        : memref<1024x4096xf16, #wafer.memory<ddr, tensor>>
      into memref<1024x4096xf16, #wafer.memory<spm, cx>>
    wafer.tile.yield %arg0
        : memref<1024x4096xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK: static_terminal_budget_exceeded: tile.load mapped DMA command count exceeds 4096
