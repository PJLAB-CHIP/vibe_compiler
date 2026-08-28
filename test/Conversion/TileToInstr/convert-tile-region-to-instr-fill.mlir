// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %s | FileCheck %s

func.func @fill(
    %boundary: memref<16xf16, #wafer.memory<ddr, tensor>>) {
  %unused = wafer.tile.region(
      %boundary : memref<16xf16, #wafer.memory<ddr, tensor>>) ->
      (memref<16xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%ddr: memref<16xf16, #wafer.memory<ddr, tensor>>):
    %dest = memref.alloc()
        : memref<16xf16, #wafer.memory<spm, tensor>>
    %zero = arith.constant 0.000000e+00 : f16
    wafer.tile.fill %dest, %zero
        : memref<16xf16, #wafer.memory<spm, tensor>>, f16
    wafer.tile.store %dest, %ddr
        : memref<16xf16, #wafer.memory<spm, tensor>>
          -> memref<16xf16, #wafer.memory<ddr, tensor>>
    wafer.tile.yield %ddr
        : memref<16xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @fill
// CHECK: wafer.instr.fill %{{.+}}, %{{.+}}
// CHECK-NOT: wafer.tile.fill
