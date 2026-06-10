// RUN: wafer-opt --wafer-place-spm-buffers='spm-base=65536 spm-limit=65792' %s | FileCheck %s

func.func @reuse_after_last_use(%boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %zero = arith.constant 0.000000e+00 : f16
    %first = memref.alloc() : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %first, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    %second = memref.alloc() : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %second, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    wafer.tile.yield %arg0 : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @reuse_after_last_use
// CHECK: %[[FIRST:.+]] = memref.alloc() {wafer.spm.placement = #wafer.spm_placement<65536, 256, 256, 256, 257>} : memref<128xf16, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.fill %[[FIRST]]
// CHECK: %[[SECOND:.+]] = memref.alloc() {wafer.spm.placement = #wafer.spm_placement<65536, 256, 256, 256, 257>} : memref<128xf16, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.fill %[[SECOND]]
