// RUN: wafer-opt --split-input-file --verify-diagnostics %s

func.func @instruction_weight_must_be_one_cx_volume(
    %input: memref<1x3x1024x16xf16, #wafer.memory<spm, ncx>>,
    %weight: memref<2x3x24x16xf16, #wafer.memory<spm, ncx>>,
    %output: memref<1x2x1022x24xf16, #wafer.memory<spm, ncx>>) {
  // expected-error @below {{convolution input/dest must use ncx and weight must use cx layout}}
  wafer.instr.conv #wafer.instr_conv_kind<conv> %input, %weight into %output
      {input_shape = array<i64: 1, 3, 1024, 16>,
       weight_shape = array<i64: 2, 3, 24, 16>,
       output_shape = array<i64: 1, 2, 1022, 24>,
       pads = array<i64: 0, 0, 0, 0>,
       unpads = array<i64: 0, 0, 0, 0>,
       kernel_strides = array<i64: 3, 2, 1, 1>,
       dilations = array<i64: 1, 1>}
      : memref<1x3x1024x16xf16, #wafer.memory<spm, ncx>>,
        memref<2x3x24x16xf16, #wafer.memory<spm, ncx>>
    into memref<1x2x1022x24xf16, #wafer.memory<spm, ncx>>
  return
}

// -----

func.func @tile_weight_must_be_one_cx_volume(
    %input: memref<1x3x1024x16xf16, #wafer.memory<spm, ncx>>,
    %weight: memref<2x3x24x16xf16, #wafer.memory<spm, ncx>>) {
  // expected-error @below {{convolution input/result must use ncx and weight must use cx layout}}
  %result = wafer.tile.conv %input, %weight
      {pads = array<i64: 0, 0, 0, 0>, unpads = array<i64: 0, 0, 0, 0>,
       strides = array<i64: 1, 1>, dilations = array<i64: 1, 1>}
      : (memref<1x3x1024x16xf16, #wafer.memory<spm, ncx>>,
         memref<2x3x24x16xf16, #wafer.memory<spm, ncx>>)
     -> memref<1x2x1022x24xf16, #wafer.memory<spm, ncx>>
  return
}
