// RUN: wafer-opt --wafer-plan-spm-memory='spm-base=65536 spm-limit=66304' %s | FileCheck %s

func.func @subview_use_extends_spm_root_lifetime(
    %boundary: memref<128xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<128xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<128xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<128xf16, #wafer.memory<ddr, tensor>>):
    %root = memref.alloc()
        : memref<256xf16, #wafer.memory<spm, tensor>>
    %view = memref.subview %root[64] [128] [1]
        : memref<256xf16, #wafer.memory<spm, tensor>>
       to memref<128xf16, strided<[1], offset: 64>, #wafer.memory<spm, tensor>>

    %overlapping = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    %c0 = arith.constant 0 : index
    %value = memref.load %view[%c0]
        : memref<128xf16, strided<[1], offset: 64>, #wafer.memory<spm, tensor>>
    %overlapping_value = memref.load %overlapping[%c0]
        : memref<128xf16, #wafer.memory<spm, tensor>>

    %after = memref.alloc()
        : memref<128xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %arg0 : memref<128xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @subview_use_extends_spm_root_lifetime
// CHECK: %[[ROOT:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<256xf16, #wafer.memory<spm, tensor>>
// CHECK: %[[VIEW:.+]] = memref.subview %[[ROOT]]
// CHECK: %[[OVERLAPPING:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66048>} : memref<128xf16, #wafer.memory<spm, tensor>>
// CHECK: memref.load %[[VIEW]]
// CHECK: memref.load %[[OVERLAPPING]]
// CHECK: %[[AFTER:.+]] = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<128xf16, #wafer.memory<spm, tensor>>
