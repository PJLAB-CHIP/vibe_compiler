// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %s | FileCheck %s

func.func @transpose_transpose() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %tile_lhs = memref.alloc() : memref<3x2xf16, #wafer.memory<spm, cx>>
  %tile_rhs = memref.alloc() : memref<4x3xf16, #wafer.memory<spm, cx>>
  %result = wafer.tile.gemm %tile_lhs, %tile_rhs
      {lhs_orientation = #wafer.gemm_orientation<transpose>,
       rhs_orientation = #wafer.gemm_orientation<transpose>}
      : (memref<3x2xf16, #wafer.memory<spm, cx>>,
         memref<4x3xf16, #wafer.memory<spm, cx>>)
     -> memref<2x4xf16, #wafer.memory<spm, cx>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @transpose_transpose
// CHECK: %[[DEST:.+]] = memref.alloc() : memref<2x4xf16, #wafer.memory<spm, cx>>
// CHECK: wafer.instr.gemm %{{.+}}, %{{.+}} into %[[DEST]]
// CHECK-SAME: k = 3 : i64
// CHECK-SAME: lhs_orientation = #wafer.gemm_orientation<transpose>
// CHECK-SAME: m = 2 : i64
// CHECK-SAME: n = 4 : i64
// CHECK-SAME: rhs_orientation = #wafer.gemm_orientation<transpose>
