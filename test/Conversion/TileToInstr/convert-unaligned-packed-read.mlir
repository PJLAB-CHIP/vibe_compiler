// RUN: split-file %s %t
// RUN: wafer-opt --wafer-lower-tile-region-to-instr %t/read.mlir | FileCheck %s
// RUN: wafer-opt --wafer-lower-tile-region-to-instr --wafer-plan-spm-memory --wafer-plan-ddr-memory --wafer-lower-instr-to-target-llvm %t/read.mlir | FileCheck %s --check-prefix=TARGET
// RUN: not wafer-opt --wafer-lower-tile-region-to-instr %t/unknown.mlir 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: not wafer-opt --wafer-lower-tile-region-to-instr %t/outside.mlir 2>&1 | FileCheck %s --check-prefix=REJECT

// CHECK-LABEL: func.func @read
// CHECK: memref.reinterpret_cast
// CHECK: wafer.instr.rdma
// CHECK: wafer.instr.bit2fp
// CHECK: wafer.instr.gather_scatter
// CHECK-SAME: src_offset = 8 : i64
// CHECK: wafer.instr.elementwise <ne>
// CHECK-SAME: rhs_unit_elements = 1 : i64
// TARGET: llvm.call @wafer_tx81_rdma
// TARGET: llvm.call @wafer_tx81_bit2fp
// TARGET: llvm.call @wafer_tx81_gather_scatter
// TARGET: llvm.call @wafer_tx81_elementwise_ne
// REJECT: failed to legalize operation 'wafer.tile.load'

//--- read.mlir
func.func @read(%input: memref<1x1031x1031xi1, #wafer.memory<ddr, tensor>>) {
  wafer.tile.region(%input : memref<1x1031x1031xi1, #wafer.memory<ddr, tensor>>) -> () {
  ^bb0(%source: memref<1x1031x1031xi1, #wafer.memory<ddr, tensor>>):
    %view = memref.subview %source[0, 7, 3] [1, 1024, 32] [1, 1, 1]
      : memref<1x1031x1031xi1, #wafer.memory<ddr, tensor>>
      to memref<1x1024x32xi1, strided<[1062961, 1031, 1], offset: 7220>, #wafer.memory<ddr, tensor>>
    %flat = memref.collapse_shape %view [[0, 1], [2]]
      : memref<1x1024x32xi1, strided<[1062961, 1031, 1], offset: 7220>, #wafer.memory<ddr, tensor>>
      into memref<1024x32xi1, strided<[1031, 1], offset: 7220>, #wafer.memory<ddr, tensor>>
    %output = memref.alloc() : memref<1024x32xi1, #wafer.memory<spm, tensor>>
    wafer.tile.load %flat into %output
      : memref<1024x32xi1, strided<[1031, 1], offset: 7220>, #wafer.memory<ddr, tensor>>
      into memref<1024x32xi1, #wafer.memory<spm, tensor>>
    // Observable end; no extra wait is introduced by packed extraction.
    wafer.instr.ncc_join [0]
    wafer.tile.yield
  }
  return
}

//--- unknown.mlir
func.func @unknown(%input: memref<1x1031x1031xi1, #wafer.memory<ddr, tensor>>, %column: index) {
  wafer.tile.region(%input, %column : memref<1x1031x1031xi1, #wafer.memory<ddr, tensor>>, index) -> () {
  ^bb0(%source: memref<1x1031x1031xi1, #wafer.memory<ddr, tensor>>, %offset: index):
    %view = memref.subview %source[0, 0, %offset] [1, 1024, 32] [1, 1, 1]
      : memref<1x1031x1031xi1, #wafer.memory<ddr, tensor>>
      to memref<1x1024x32xi1, strided<[1062961, 1031, 1], offset: ?>, #wafer.memory<ddr, tensor>>
    %output = memref.alloc() : memref<1x1024x32xi1, #wafer.memory<spm, tensor>>
    wafer.tile.load %view into %output
      : memref<1x1024x32xi1, strided<[1062961, 1031, 1], offset: ?>, #wafer.memory<ddr, tensor>>
      into memref<1x1024x32xi1, #wafer.memory<spm, tensor>>
    wafer.tile.yield
  }
  return
}

//--- outside.mlir
// A verifier-valid static view is not proof that its enclosing byte window
// lies in the original allocation; check that bound before materializing.
func.func @outside(%input: memref<1x1024x1031xi1, #wafer.memory<ddr, tensor>>) {
  wafer.tile.region(%input : memref<1x1024x1031xi1, #wafer.memory<ddr, tensor>>) -> () {
  ^bb0(%source: memref<1x1024x1031xi1, #wafer.memory<ddr, tensor>>):
    %view = memref.subview %source[0, 1, 3] [1, 1024, 32] [1, 1, 1]
      : memref<1x1024x1031xi1, #wafer.memory<ddr, tensor>>
      to memref<1x1024x32xi1, strided<[1055744, 1031, 1], offset: 1034>, #wafer.memory<ddr, tensor>>
    %output = memref.alloc() : memref<1x1024x32xi1, #wafer.memory<spm, tensor>>
    wafer.tile.load %view into %output
      : memref<1x1024x32xi1, strided<[1055744, 1031, 1], offset: 1034>, #wafer.memory<ddr, tensor>>
      into memref<1x1024x32xi1, #wafer.memory<spm, tensor>>
    wafer.tile.yield
  }
  return
}
