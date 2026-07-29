// RUN: wafer-opt --wafer-convert-tile-region-to-instr %s | FileCheck %s

func.func @ordered_trailing_tensor_sum(
    %input: memref<2x2x512xf16, #wafer.memory<spm, ncx>>)
    -> memref<2x2xf16, #wafer.memory<spm, cx>> {
  %result = wafer.tile.reduce #wafer.reduce_kind<sum> %input
      {dimensions = array<i64: 2>, init_value = 0.000000e+00 : f16}
      : (memref<2x2x512xf16, #wafer.memory<spm, ncx>>)
     -> memref<2x2xf16, #wafer.memory<spm, cx>>
  return %result : memref<2x2xf16, #wafer.memory<spm, cx>>
}

// CHECK-LABEL: func.func @ordered_trailing_tensor_sum
// CHECK: %[[SLICE:.*]] = memref.alloc() : memref<2x2xf16, #wafer.memory<spm, tensor>>
// CHECK-COUNT-512: wafer.instr.gather_scatter %arg0 to %[[SLICE]]
// CHECK-NOT: wafer.tile.reduce
// CHECK: return
