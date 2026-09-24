// RUN: wafer-opt --wafer-convert-private-packed-updates-to-instr %s | FileCheck %s --check-prefix=UNCHANGED
// RUN: not wafer-opt --wafer-lower-tile-region-to-instr %s 2>&1 | FileCheck %s
// A verifier-valid view with overlapping rows is not an injective write.
// UNCHANGED: memref.reinterpret_cast
// UNCHANGED-NOT: wafer.instr
// UNCHANGED: wafer.tile.store
// CHECK: failed to legalize operation 'wafer.tile.store'

func.func @overlap() {
  %root = memref.alloc() : memref<1x1025x1031xi1, #wafer.memory<ddr, tensor>>
  wafer.tile.region(%root : memref<1x1025x1031xi1, #wafer.memory<ddr, tensor>>) -> () {
  ^bb0(%dest: memref<1x1025x1031xi1, #wafer.memory<ddr, tensor>>):
    %view = memref.reinterpret_cast %dest to offset: [0], sizes: [1, 1024, 32], strides: [1024, 1, 1]
      : memref<1x1025x1031xi1, #wafer.memory<ddr, tensor>>
      to memref<1x1024x32xi1, strided<[1024, 1, 1]>, #wafer.memory<ddr, tensor>>
    %source = memref.alloc() : memref<1x1024x32xi1, #wafer.memory<spm, tensor>>
    wafer.tile.store %source, %view
      : memref<1x1024x32xi1, #wafer.memory<spm, tensor>>
      -> memref<1x1024x32xi1, strided<[1024, 1, 1]>, #wafer.memory<ddr, tensor>>
    wafer.tile.yield
  }
  return
}
