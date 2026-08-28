// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %s | FileCheck %s

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card_partition"], shape = array<i64: 1>}

  func.func @oriented_gemm() {
    %lhs = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<3x2xf16, #wafer.memory<spm, cx>>
    %rhs = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65792>}
        : memref<4x3xf16, #wafer.memory<spm, cx>>
    %dst = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<66048>}
        : memref<2x4xf16, #wafer.memory<spm, cx>>
    wafer.instr.gemm %lhs, %rhs into %dst
        {m = 2 : i64, k = 3 : i64, n = 4 : i64,
         lhs_orientation = #wafer.gemm_orientation<transpose>,
         rhs_orientation = #wafer.gemm_orientation<transpose>}
        : memref<3x2xf16, #wafer.memory<spm, cx>>,
          memref<4x3xf16, #wafer.memory<spm, cx>>
      into memref<2x4xf16, #wafer.memory<spm, cx>>
    return
  }
}

// CHECK-LABEL: llvm.func @oriented_gemm
// CHECK: %[[LHS:.*]] = llvm.mlir.constant(65536 : i64) : i64
// CHECK: %[[RHS:.*]] = llvm.mlir.constant(65792 : i64) : i64
// CHECK: %[[DST:.*]] = llvm.mlir.constant(66048 : i64) : i64
// CHECK: %[[M:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK: %[[K:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK: %[[N:.*]] = llvm.mlir.constant(4 : i32) : i32
// CHECK: %[[BATCH:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[FORMAT:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK: %[[LHS_ORIENTATION:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[RHS_ORIENTATION:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[WORKER:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: llvm.call @wafer_tx81_gemm_oriented(%[[LHS]], %[[RHS]], %[[DST]], %[[M]], %[[K]], %[[N]], %[[BATCH]], %[[FORMAT]], %[[LHS_ORIENTATION]], %[[RHS_ORIENTATION]], %[[WORKER]])
