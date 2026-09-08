// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %s | FileCheck %s

func.func @ordinary_conv() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %tile_input = memref.alloc()
      : memref<1x7x11x5xf16, #wafer.memory<spm, ncx>>
  %tile_weight = memref.alloc()
      : memref<2x3x7x5xf16, #wafer.memory<spm, cx>>
  %result = wafer.tile.conv %tile_input, %tile_weight
      {pads = array<i64: 1, 0, 2, 1>,
       unpads = array<i64: 0, 0, 0, 0>,
       strides = array<i64: 3, 2>,
       dilations = array<i64: 1, 2>}
      : (memref<1x7x11x5xf16, #wafer.memory<spm, ncx>>,
         memref<2x3x7x5xf16, #wafer.memory<spm, cx>>)
     -> memref<1x3x5x7xf16, #wafer.memory<spm, ncx>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @ordinary_conv
// CHECK: %[[DEST:.+]] = memref.alloc() : memref<1x3x5x7xf16, #wafer.memory<spm, ncx>>
// CHECK: wafer.instr.conv <conv> %{{.+}}, %{{.+}} into %[[DEST]]
// CHECK-SAME: dilations = array<i64: 2, 1>
// CHECK-SAME: input_shape = array<i64: 1, 7, 11, 5>
// CHECK-SAME: kernel_strides = array<i64: 3, 2, 2, 3>
// CHECK-SAME: output_shape = array<i64: 1, 3, 5, 7>
// CHECK-SAME: pads = array<i64: 1, 0, 2, 1>
// CHECK-SAME: weight_shape = array<i64: 2, 3, 7, 5>
