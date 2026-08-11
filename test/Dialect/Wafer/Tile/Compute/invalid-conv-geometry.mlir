// RUN: wafer-opt --verify-diagnostics %s

func.func @invalid_output(
    %input: memref<1x7x11x5xf16, #wafer.memory<spm, ncx>>,
    %weight: memref<3x2x7x5xf16, #wafer.memory<spm, ncx>>) {
  // expected-error @below {{convolution output spatial shape does not match input/kernel/stride/dilation/pad/unpad}}
  %result = wafer.tile.conv %input, %weight
      {pads = array<i64: 1, 0, 2, 1>,
       unpads = array<i64: 0, 0, 0, 0>,
       strides = array<i64: 3, 2>,
       dilations = array<i64: 1, 2>}
      : (memref<1x7x11x5xf16, #wafer.memory<spm, ncx>>,
         memref<3x2x7x5xf16, #wafer.memory<spm, ncx>>)
     -> memref<1x4x5x7xf16, #wafer.memory<spm, ncx>>
  return
}
