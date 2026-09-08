// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %act = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x8x8x64xf16, #wafer.memory<spm, tensor>>
  %weight = "builtin.unrealized_conversion_cast"()
      : () -> memref<3x3x64x64xf16, #wafer.memory<spm, cx>>
  %out = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x8x8x64xf16, #wafer.memory<spm, ncx>>

  wafer.instr.conv #wafer.instr_conv_kind<conv> %act, %weight into %out
      {input_shape = array<i64: 1, 8, 8, 64>,
       weight_shape = array<i64: 3, 3, 64, 64>,
       output_shape = array<i64: 1, 8, 8, 64>,
       pads = array<i64: 1, 1, 1, 1>,
       unpads = array<i64: 0, 0, 0, 0>,
       kernel_strides = array<i64: 3, 3, 1, 1>,
       dilations = array<i64: 1, 1>}
      : memref<1x8x8x64xf16, #wafer.memory<spm, tensor>>,
        memref<3x3x64x64xf16, #wafer.memory<spm, cx>>
    into memref<1x8x8x64xf16, #wafer.memory<spm, ncx>>
}

// CHECK: input must use cx/ncx SPM layout
