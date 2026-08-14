// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %s | FileCheck %s

func.func @ordered_trailing_tensor_sum() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %tile_input = memref.alloc()
      : memref<2x2x512xf16, #wafer.memory<spm, ncx>>
  %result = wafer.tile.reduce #wafer.reduce_kind<sum> %tile_input
      {dimensions = array<i64: 2>, init_value = 0.000000e+00 : f16}
      : (memref<2x2x512xf16, #wafer.memory<spm, ncx>>)
     -> memref<2x2xf16, #wafer.memory<spm, cx>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @ordered_trailing_tensor_sum
// CHECK-COUNT-512: wafer.instr.gather_scatter %{{.+}} to %{{.+}}
// CHECK-NOT: wafer.tile.reduce
// CHECK: return
