// RUN: wafer-opt --wafer-place-spm-buffers='spm-base=65536 spm-limit=66560' %s | FileCheck %s

func.func @pressure_weighted_packing_avoids_fragmentation(
    %boundary: memref<256xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<256xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<256xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<256xf16, #wafer.memory<ddr, tensor>>):
    %zero = arith.constant 0.000000e+00 : f16
    %early_dead = memref.alloc() : memref<128xf16, #wafer.memory<spm, tensor>>
    %left = memref.alloc() : memref<128xf16, #wafer.memory<spm, tensor>>
    %right = memref.alloc() : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %early_dead, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    %large = memref.alloc() : memref<256xf16, #wafer.memory<spm, tensor>>
    wafer.instr.fill %left, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.fill %right, %zero
        : memref<128xf16, #wafer.memory<spm, tensor>>, f16
    wafer.instr.fill %large, %zero
        : memref<256xf16, #wafer.memory<spm, tensor>>, f16
    wafer.tile.yield %arg0 : memref<256xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @pressure_weighted_packing_avoids_fragmentation
// CHECK: %[[EARLY:.+]] = memref.alloc() {wafer.spm.placement = #wafer.spm_placement<65536, 256, 256, 256, 257>} : memref<128xf16, #wafer.memory<spm, tensor>>
// CHECK: %[[LEFT:.+]] = memref.alloc() {wafer.spm.placement = #wafer.spm_placement<66048, 256, 256, 258, 259>} : memref<128xf16, #wafer.memory<spm, tensor>>
// CHECK: %[[RIGHT:.+]] = memref.alloc() {wafer.spm.placement = #wafer.spm_placement<66304, 256, 256, 259, 260>} : memref<128xf16, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.fill %[[EARLY]]
// CHECK: %[[LARGE:.+]] = memref.alloc() {wafer.spm.placement = #wafer.spm_placement<65536, 512, 256, 256, 258>} : memref<256xf16, #wafer.memory<spm, tensor>>
// CHECK: wafer.instr.fill %[[LEFT]]
// CHECK: wafer.instr.fill %[[RIGHT]]
// CHECK: wafer.instr.fill %[[LARGE]]
