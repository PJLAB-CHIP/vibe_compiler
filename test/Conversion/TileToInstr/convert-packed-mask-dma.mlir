// RUN: split-file %s %t
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %t/read.mlir | FileCheck %s --check-prefix=READ
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %t/tail.mlir | FileCheck %s --check-prefix=TAIL
// RUN: not wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %t/partial-tail.mlir 2>&1 | FileCheck %s --check-prefix=PARTIAL
// RUN: wafer-opt --wafer-plan-ddr-memory %t/aligned.mlir | wafer-opt --wafer-lower-instr-to-target-llvm | FileCheck %s --check-prefix=ADDRESS
// RUN: not wafer-opt --wafer-plan-ddr-memory %t/unaligned.mlir 2>&1 | FileCheck %s --check-prefix=ALIGNMENT
// RUN: not wafer-opt --wafer-lower-instr-to-target-llvm %t/unaligned.mlir 2>&1 | FileCheck %s --check-prefix=TARGET-ALIGNMENT

// READ: wafer.instr.rdma
// READ-SAME: byte_count = 2048 : i64
// READ-NOT: memref.copy
// TAIL: wafer.instr.rdma
// TAIL-SAME: byte_count = 129 : i64
// PARTIAL: failed to legalize operation 'memref.copy'
// ADDRESS: llvm.mlir.constant(14 : i64)
// ADDRESS: llvm.call @wafer_tx81_rdma
// ALIGNMENT: bitpacked DDR view requires a statically byte-aligned offset
// TARGET-ALIGNMENT: bitpacked view requires a byte-aligned offset

//--- read.mlir
func.func @read(%input: memref<1x1031x16xi1, #wafer.memory<ddr, tensor>>) {
  wafer.tile.region(%input : memref<1x1031x16xi1, #wafer.memory<ddr, tensor>>) -> () {
  ^bb0(%source: memref<1x1031x16xi1, #wafer.memory<ddr, tensor>>):
    %view = memref.subview %source[0, 7, 0] [1, 1024, 16] [1, 1, 1]
      : memref<1x1031x16xi1, #wafer.memory<ddr, tensor>>
      to memref<1x1024x16xi1, strided<[16496, 16, 1], offset: 112>, #wafer.memory<ddr, tensor>>
    %out = memref.alloc() : memref<1x1024x16xi1, #wafer.memory<spm, tensor>>
    memref.copy %view, %out
      : memref<1x1024x16xi1, strided<[16496, 16, 1], offset: 112>, #wafer.memory<ddr, tensor>>
      to memref<1x1024x16xi1, #wafer.memory<spm, tensor>>
    wafer.tile.yield
  }
  return
}

//--- tail.mlir
func.func @tail(%input: memref<1x1x1031xi1, #wafer.memory<ddr, tensor>>) {
  wafer.tile.region(%input : memref<1x1x1031xi1, #wafer.memory<ddr, tensor>>) -> () {
  ^bb0(%source: memref<1x1x1031xi1, #wafer.memory<ddr, tensor>>):
    %out = memref.alloc() : memref<1x1x1031xi1, #wafer.memory<spm, tensor>>
    memref.copy %source, %out : memref<1x1x1031xi1, #wafer.memory<ddr, tensor>>
      to memref<1x1x1031xi1, #wafer.memory<spm, tensor>>
    wafer.tile.yield
  }
  return
}

//--- partial-tail.mlir
// The last destination byte also contains live bits outside this view.
func.func @partial_tail(%input: memref<1x1x1031xi1, #wafer.memory<ddr, tensor>>) {
  wafer.tile.region(%input : memref<1x1x1031xi1, #wafer.memory<ddr, tensor>>) -> () {
  ^bb0(%source: memref<1x1x1031xi1, #wafer.memory<ddr, tensor>>):
    %storage = memref.alloc() : memref<1x1x1040xi1, #wafer.memory<spm, tensor>>
    %out = memref.subview %storage[0, 0, 0] [1, 1, 1031] [1, 1, 1]
      : memref<1x1x1040xi1, #wafer.memory<spm, tensor>>
      to memref<1x1x1031xi1, strided<[1040, 1040, 1]>, #wafer.memory<spm, tensor>>
    memref.copy %source, %out : memref<1x1x1031xi1, #wafer.memory<ddr, tensor>>
      to memref<1x1x1031xi1, strided<[1040, 1040, 1]>, #wafer.memory<spm, tensor>>
    wafer.tile.yield
  }
  return
}

//--- aligned.mlir
func.func @aligned(%input: memref<1x1031x16xi1, #wafer.memory<ddr, tensor>>) {
  %view = memref.subview %input[0, 7, 0] [1, 1024, 16] [1, 1, 1]
    : memref<1x1031x16xi1, #wafer.memory<ddr, tensor>>
    to memref<1x1024x16xi1, strided<[16496, 16, 1], offset: 112>, #wafer.memory<ddr, tensor>>
  %out = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
    : memref<1x1024x16xi1, #wafer.memory<spm, tensor>>
  wafer.instr.rdma %view to %out {byte_count = 2048 : i64, inner_bytes = 2048 : i64,
    src_iterations = array<i64: 1, 1, 1>, src_strides = array<i64: 0, 0, 0>}
    : memref<1x1024x16xi1, strided<[16496, 16, 1], offset: 112>, #wafer.memory<ddr, tensor>>
    to memref<1x1024x16xi1, #wafer.memory<spm, tensor>>
  wafer.instr.ncc_join [0]
  return
}

//--- unaligned.mlir
func.func @unaligned(%input: memref<1x1x1031xi1, #wafer.memory<ddr, tensor>>) {
  %view = memref.subview %input[0, 0, 1] [1, 1, 1024] [1, 1, 1]
    : memref<1x1x1031xi1, #wafer.memory<ddr, tensor>>
    to memref<1x1x1024xi1, strided<[1031, 1031, 1], offset: 1>, #wafer.memory<ddr, tensor>>
  %out = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
    : memref<1x1x1024xi1, #wafer.memory<spm, tensor>>
  wafer.instr.rdma %view to %out {byte_count = 128 : i64, inner_bytes = 128 : i64,
    src_iterations = array<i64: 1, 1, 1>, src_strides = array<i64: 0, 0, 0>}
    : memref<1x1x1024xi1, strided<[1031, 1031, 1], offset: 1>, #wafer.memory<ddr, tensor>>
    to memref<1x1x1024xi1, #wafer.memory<spm, tensor>>
  wafer.instr.ncc_join [0]
  return
}
