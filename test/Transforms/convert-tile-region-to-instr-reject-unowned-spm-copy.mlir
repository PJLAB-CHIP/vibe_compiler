// RUN: not wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %s 2>&1 | FileCheck %s

func.func @unowned_spm_copy() {
  %source = memref.alloc()
      : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
  %target = memref.alloc()
      : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
  memref.copy %source, %target
      : memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
        to memref<2x1025x64xf16, #wafer.memory<spm, tensor>>
  return
}

// CHECK: failed to legalize operation 'memref.copy'
