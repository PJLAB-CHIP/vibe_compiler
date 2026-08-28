// RUN: not wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %s 2>&1 | FileCheck %s

func.func @unowned_ddr_copy() {
  %source = memref.alloc()
      : memref<2x1024x64xf16, #wafer.memory<ddr, tensor>>
  %target = memref.alloc()
      : memref<2x1024x64xf16, #wafer.memory<ddr, tensor>>
  memref.copy %source, %target
      : memref<2x1024x64xf16, #wafer.memory<ddr, tensor>>
        to memref<2x1024x64xf16, #wafer.memory<ddr, tensor>>
  return
}

// CHECK: failed to legalize operation 'memref.copy'
