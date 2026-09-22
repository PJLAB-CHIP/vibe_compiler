// RUN: wafer-opt %s | FileCheck %s

// External DDR storage carries its selected physical encoding. NCx movement
// address coverage is checked by BlockedDDRWindowsCopyExactPhysicalBytes.
module {
  func.func @store(%src: memref<2x1024x128xbf16, #wafer.memory<spm, ntensor>>,
                   %plain: memref<2x1024x128xbf16, #wafer.memory<ddr, ntensor>>,
                   %blocked: memref<2x1024x128xbf16, #wafer.memory<ddr, ncx>>) {
    wafer.tile.store %src, %plain
        : memref<2x1024x128xbf16, #wafer.memory<spm, ntensor>>
          -> memref<2x1024x128xbf16, #wafer.memory<ddr, ntensor>>
    wafer.tile.store %src, %blocked
        : memref<2x1024x128xbf16, #wafer.memory<spm, ntensor>>
          -> memref<2x1024x128xbf16, #wafer.memory<ddr, ncx>>
    return
  }
}

// CHECK-LABEL: func.func @store
// CHECK: wafer.tile.store {{.*}} #wafer.memory<ddr, ntensor>
// CHECK: wafer.tile.store {{.*}} #wafer.memory<ddr, ncx>
