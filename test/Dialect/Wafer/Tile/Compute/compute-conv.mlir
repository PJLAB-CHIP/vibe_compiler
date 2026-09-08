// RUN: wafer-opt %s | FileCheck %s

func.func @ordinary_conv(
    %input: memref<1x7x11x5xf16, #wafer.memory<spm, ncx>>,
    %weight: memref<2x3x7x5xf16, #wafer.memory<spm, cx>>) {
  %result = wafer.tile.conv %input, %weight
      {pads = array<i64: 1, 0, 2, 1>,
       unpads = array<i64: 0, 0, 0, 0>,
       strides = array<i64: 3, 2>,
       dilations = array<i64: 1, 2>}
      : (memref<1x7x11x5xf16, #wafer.memory<spm, ncx>>,
         memref<2x3x7x5xf16, #wafer.memory<spm, cx>>)
     -> memref<1x3x5x7xf16, #wafer.memory<spm, ncx>>
  return
}

// CHECK-LABEL: func.func @ordinary_conv
// CHECK: wafer.tile.conv
// CHECK-SAME: dilations = array<i64: 1, 2>
// CHECK-SAME: pads = array<i64: 1, 0, 2, 1>
// CHECK-SAME: strides = array<i64: 3, 2>
