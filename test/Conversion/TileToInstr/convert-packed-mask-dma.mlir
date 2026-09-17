// RUN: split-file %s %t
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %t/read.mlir | FileCheck %s --check-prefix=READ
// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %t/tail.mlir | FileCheck %s --check-prefix=TAIL
// RUN: not wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %t/partial-tail.mlir 2>&1 | FileCheck %s --check-prefix=PARTIAL
// RUN: wafer-opt --wafer-plan-ddr-memory %t/aligned.mlir | wafer-opt --wafer-lower-instr-to-target-llvm | FileCheck %s --check-prefix=ADDRESS
// RUN: not wafer-opt --wafer-plan-ddr-memory %t/unaligned.mlir 2>&1 | FileCheck %s --check-prefix=ALIGNMENT
// RUN: not wafer-opt --wafer-lower-instr-to-target-llvm %t/unaligned.mlir 2>&1 | FileCheck %s --check-prefix=TARGET-ALIGNMENT
// RUN: wafer-opt --wafer-lower-tile-region-to-instr --wafer-plan-ddr-memory --wafer-lower-instr-to-target-llvm %t/dynamic.mlir | FileCheck %s --check-prefix=DYNAMIC
// RUN: not wafer-opt --wafer-lower-tile-region-to-instr --wafer-plan-ddr-memory %t/step-one.mlir 2>&1 | FileCheck %s --check-prefix=STEP
// RUN: not wafer-opt --wafer-lower-tile-region-to-instr --wafer-lower-instr-to-target-llvm %t/step-one.mlir 2>&1 | FileCheck %s --check-prefix=TARGET-STEP
// RUN: wafer-opt --wafer-lower-tile-region-to-instr %t/strided.mlir | FileCheck %s --check-prefix=STRIDED
// RUN: not wafer-opt --wafer-lower-tile-region-to-instr %t/unaligned-stride.mlir 2>&1 | FileCheck %s --check-prefix=ROW

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

//--- dynamic.mlir
func.func @dynamic(%input: memref<1x1x1048xi1, #wafer.memory<ddr, tensor>>) {
  %base = memref.subview %input[0, 0, 8] [1, 1, 1040] [1, 1, 1]
    : memref<1x1x1048xi1, #wafer.memory<ddr, tensor>>
    to memref<1x1x1040xi1, strided<[1048, 1048, 1], offset: 8>, #wafer.memory<ddr, tensor>>
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c17 = arith.constant 17 : index
  scf.for %i = %c0 to %c17 step %c8 {
    %view = memref.subview %base[0, 0, %i] [1, 1, 1024] [1, 1, 1]
      : memref<1x1x1040xi1, strided<[1048, 1048, 1], offset: 8>, #wafer.memory<ddr, tensor>>
      to memref<1x1x1024xi1, strided<[1048, 1048, 1], offset: ?>, #wafer.memory<ddr, tensor>>
    wafer.tile.region(%view : memref<1x1x1024xi1, strided<[1048, 1048, 1], offset: ?>, #wafer.memory<ddr, tensor>>) -> () {
    ^bb0(%source: memref<1x1x1024xi1, strided<[1048, 1048, 1], offset: ?>, #wafer.memory<ddr, tensor>>):
      %out = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<1x1x1024xi1, #wafer.memory<spm, tensor>>
      wafer.tile.load %source into %out
        : memref<1x1x1024xi1, strided<[1048, 1048, 1], offset: ?>, #wafer.memory<ddr, tensor>>
        into memref<1x1x1024xi1, #wafer.memory<spm, tensor>>
      wafer.tile.store %out, %source
        : memref<1x1x1024xi1, #wafer.memory<spm, tensor>>
        -> memref<1x1x1024xi1, strided<[1048, 1048, 1], offset: ?>, #wafer.memory<ddr, tensor>>
      wafer.tile.yield
    }
  }
  return
}
// DYNAMIC-LABEL: llvm.func @dynamic
// DYNAMIC: llvm.udiv
// DYNAMIC: llvm.call @wafer_tx81_rdma
// DYNAMIC: llvm.call @wafer_tx81_wdma

//--- step-one.mlir
// Aligned minimum and maximum do not prove the intervening indices aligned.
func.func @step_one(%input: memref<1x1x1040xi1, #wafer.memory<ddr, tensor>>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 1 : index
  %c17 = arith.constant 17 : index
  scf.for %i = %c0 to %c17 step %c8 {
    %view = memref.subview %input[0, 0, %i] [1, 1, 1024] [1, 1, 1]
      : memref<1x1x1040xi1, #wafer.memory<ddr, tensor>>
      to memref<1x1x1024xi1, strided<[1040, 1040, 1], offset: ?>, #wafer.memory<ddr, tensor>>
    wafer.tile.region(%view : memref<1x1x1024xi1, strided<[1040, 1040, 1], offset: ?>, #wafer.memory<ddr, tensor>>) -> () {
    ^bb0(%source: memref<1x1x1024xi1, strided<[1040, 1040, 1], offset: ?>, #wafer.memory<ddr, tensor>>):
      %out = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<1x1x1024xi1, #wafer.memory<spm, tensor>>
      wafer.tile.load %source into %out
        : memref<1x1x1024xi1, strided<[1040, 1040, 1], offset: ?>, #wafer.memory<ddr, tensor>>
        into memref<1x1x1024xi1, #wafer.memory<spm, tensor>>
      wafer.tile.store %out, %source
        : memref<1x1x1024xi1, #wafer.memory<spm, tensor>>
        -> memref<1x1x1024xi1, strided<[1040, 1040, 1], offset: ?>, #wafer.memory<ddr, tensor>>
      wafer.tile.yield
    }
  }
  return
}
// STEP: bitpacked DDR view requires a statically byte-aligned offset or proven dynamic alignment
// TARGET-STEP: packed subview byte alignment is not proven from current SSA

//--- strided.mlir
func.func @strided_1024(%input: memref<1x1024x1024xi1, #wafer.memory<ddr, tensor>>) {
  %view = memref.subview %input[0, 0, 512] [1, 1024, 512] [1, 1, 1]
    : memref<1x1024x1024xi1, #wafer.memory<ddr, tensor>> to memref<1x1024x512xi1, strided<[1048576, 1024, 1], offset: 512>, #wafer.memory<ddr, tensor>>
  wafer.tile.region(%view : memref<1x1024x512xi1, strided<[1048576, 1024, 1], offset: 512>, #wafer.memory<ddr, tensor>>) -> () {
  ^bb0(%source: memref<1x1024x512xi1, strided<[1048576, 1024, 1], offset: 512>, #wafer.memory<ddr, tensor>>):
    %out = memref.alloc() : memref<1x1024x512xi1, #wafer.memory<spm, tensor>>
    wafer.tile.load %source into %out : memref<1x1024x512xi1, strided<[1048576, 1024, 1], offset: 512>, #wafer.memory<ddr, tensor>> into memref<1x1024x512xi1, #wafer.memory<spm, tensor>>
    wafer.tile.store %out, %source : memref<1x1024x512xi1, #wafer.memory<spm, tensor>> -> memref<1x1024x512xi1, strided<[1048576, 1024, 1], offset: 512>, #wafer.memory<ddr, tensor>>
    wafer.tile.yield
  }
  return
}
// STRIDED-LABEL: func.func @strided_1024
// STRIDED: wafer.instr.rdma
// STRIDED-SAME: byte_count = 65536 : i64
// STRIDED-SAME: inner_bytes = 64 : i64
// STRIDED-SAME: src_iterations = array<i64: 1024, 1, 1>
// STRIDED-SAME: src_strides = array<i64: 128, 0, 0>
// STRIDED: wafer.instr.wdma
// STRIDED-SAME: byte_count = 65536 : i64
// STRIDED-SAME: dst_iterations = array<i64: 1024, 1, 1>
// STRIDED-SAME: dst_strides = array<i64: 128, 0, 0>
// STRIDED-SAME: inner_bytes = 64 : i64

func.func @strided_1025(%input: memref<1x1025x1024xi1, #wafer.memory<ddr, tensor>>) {
  %view = memref.subview %input[0, 0, 512] [1, 1025, 512] [1, 1, 1]
    : memref<1x1025x1024xi1, #wafer.memory<ddr, tensor>> to memref<1x1025x512xi1, strided<[1049600, 1024, 1], offset: 512>, #wafer.memory<ddr, tensor>>
  wafer.tile.region(%view : memref<1x1025x512xi1, strided<[1049600, 1024, 1], offset: 512>, #wafer.memory<ddr, tensor>>) -> () {
  ^bb0(%source: memref<1x1025x512xi1, strided<[1049600, 1024, 1], offset: 512>, #wafer.memory<ddr, tensor>>):
    %out = memref.alloc() : memref<1x1025x512xi1, #wafer.memory<spm, tensor>>
    wafer.tile.load %source into %out : memref<1x1025x512xi1, strided<[1049600, 1024, 1], offset: 512>, #wafer.memory<ddr, tensor>> into memref<1x1025x512xi1, #wafer.memory<spm, tensor>>
    wafer.tile.store %out, %source : memref<1x1025x512xi1, #wafer.memory<spm, tensor>> -> memref<1x1025x512xi1, strided<[1049600, 1024, 1], offset: 512>, #wafer.memory<ddr, tensor>>
    wafer.tile.yield
  }
  return
}
// STRIDED-LABEL: func.func @strided_1025
// STRIDED: wafer.instr.rdma
// STRIDED-SAME: byte_count = 65600 : i64
// STRIDED-SAME: inner_bytes = 64 : i64
// STRIDED-SAME: src_iterations = array<i64: 1025, 1, 1>
// STRIDED-SAME: src_strides = array<i64: 128, 0, 0>
// STRIDED: wafer.instr.wdma
// STRIDED-SAME: byte_count = 65600 : i64
// STRIDED-SAME: dst_iterations = array<i64: 1025, 1, 1>
// STRIDED-SAME: dst_strides = array<i64: 128, 0, 0>
// STRIDED-SAME: inner_bytes = 64 : i64

func.func @strided_1031(%input: memref<1x1031x1024xi1, #wafer.memory<ddr, tensor>>) {
  %view = memref.subview %input[0, 0, 512] [1, 1031, 512] [1, 1, 1]
    : memref<1x1031x1024xi1, #wafer.memory<ddr, tensor>> to memref<1x1031x512xi1, strided<[1055744, 1024, 1], offset: 512>, #wafer.memory<ddr, tensor>>
  wafer.tile.region(%view : memref<1x1031x512xi1, strided<[1055744, 1024, 1], offset: 512>, #wafer.memory<ddr, tensor>>) -> () {
  ^bb0(%source: memref<1x1031x512xi1, strided<[1055744, 1024, 1], offset: 512>, #wafer.memory<ddr, tensor>>):
    %out = memref.alloc() : memref<1x1031x512xi1, #wafer.memory<spm, tensor>>
    wafer.tile.load %source into %out : memref<1x1031x512xi1, strided<[1055744, 1024, 1], offset: 512>, #wafer.memory<ddr, tensor>> into memref<1x1031x512xi1, #wafer.memory<spm, tensor>>
    wafer.tile.store %out, %source : memref<1x1031x512xi1, #wafer.memory<spm, tensor>> -> memref<1x1031x512xi1, strided<[1055744, 1024, 1], offset: 512>, #wafer.memory<ddr, tensor>>
    wafer.tile.yield
  }
  return
}
// STRIDED-LABEL: func.func @strided_1031
// STRIDED: wafer.instr.rdma
// STRIDED-SAME: byte_count = 65984 : i64
// STRIDED-SAME: inner_bytes = 64 : i64
// STRIDED-SAME: src_iterations = array<i64: 1031, 1, 1>
// STRIDED-SAME: src_strides = array<i64: 128, 0, 0>
// STRIDED: wafer.instr.wdma
// STRIDED-SAME: byte_count = 65984 : i64
// STRIDED-SAME: dst_iterations = array<i64: 1031, 1, 1>
// STRIDED-SAME: dst_strides = array<i64: 128, 0, 0>
// STRIDED-SAME: inner_bytes = 64 : i64

//--- unaligned-stride.mlir
// Whole-byte row lengths do not permit a fractional-byte row stride.
func.func @unaligned_rows(%input: memref<1x1024x1025xi1, #wafer.memory<ddr, tensor>>) {
  %view = memref.subview %input[0, 0, 512] [1, 1024, 512] [1, 1, 1]
    : memref<1x1024x1025xi1, #wafer.memory<ddr, tensor>> to memref<1x1024x512xi1, strided<[1049600, 1025, 1], offset: 512>, #wafer.memory<ddr, tensor>>
  wafer.tile.region(%view : memref<1x1024x512xi1, strided<[1049600, 1025, 1], offset: 512>, #wafer.memory<ddr, tensor>>) -> () {
  ^bb0(%source: memref<1x1024x512xi1, strided<[1049600, 1025, 1], offset: 512>, #wafer.memory<ddr, tensor>>):
    %out = memref.alloc() : memref<1x1024x512xi1, #wafer.memory<spm, tensor>>
    wafer.tile.load %source into %out : memref<1x1024x512xi1, strided<[1049600, 1025, 1], offset: 512>, #wafer.memory<ddr, tensor>> into memref<1x1024x512xi1, #wafer.memory<spm, tensor>>
    wafer.tile.yield
  }
  return
}
// ROW: failed to legalize operation 'wafer.tile.load'
