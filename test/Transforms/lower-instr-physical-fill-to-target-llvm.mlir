// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %s | FileCheck %s

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card_partition"], shape = array<i64: 1>}

  func.func @physical_fill() {
    %cx = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<2x3xf16, #wafer.memory<spm, cx>>
    %bits = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<9xi1, #wafer.memory<spm, tensor>>
    %half = arith.constant 3.33251953125E-1 : f16
    %true = arith.constant true
    wafer.instr.fill %cx, %half
        {fill_domain = #wafer.fill_domain<physical_footprint>}
        : memref<2x3xf16, #wafer.memory<spm, cx>>, f16
    wafer.instr.fill %bits, %true
        {fill_domain = #wafer.fill_domain<physical_footprint>}
        : memref<9xi1, #wafer.memory<spm, tensor>>, i1
    return
  }
}

// CHECK-LABEL: llvm.func @physical_fill
// CHECK: %[[CX:.*]] = llvm.mlir.constant(65536 : i64) : i64
// CHECK: %[[BITS:.*]] = llvm.mlir.constant(65792 : i64) : i64
// CHECK: %[[HALF_RAW:.*]] = llvm.mlir.constant(13653 : i32) : i32
// CHECK: %[[CX_COUNT:.*]] = llvm.mlir.constant(128 : i32) : i32
// CHECK: %[[F16_FORMAT:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK: %[[FIRST_WORKER:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: llvm.call @wafer_tx81_memset_v3(%[[CX]], %[[HALF_RAW]], %[[CX_COUNT]], %[[F16_FORMAT]], %[[FIRST_WORKER]])
// CHECK: %[[TRUE_RAW:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[BIT_COUNT:.*]] = llvm.mlir.constant(16 : i32) : i32
// CHECK: %[[BOOL_FORMAT:.*]] = llvm.mlir.constant(7 : i32) : i32
// CHECK: %[[SECOND_WORKER:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: llvm.call @wafer_tx81_memset_v3(%[[BITS]], %[[TRUE_RAW]], %[[BIT_COUNT]], %[[BOOL_FORMAT]], %[[SECOND_WORKER]])
