// RUN: wafer-opt --split-input-file --verify-diagnostics %s

func.func @wrong_output(%input: memref<1x1024x16x8xf16, #wafer.memory<spm, ncx>>) {
  // expected-error @+1 {{result extent disagrees with its input window geometry}}
  %r = wafer.tile.pool #wafer.reduce_kind<avg> %input
      {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>, dilations = array<i64: 1, 1>}
      : memref<1x1024x16x8xf16, #wafer.memory<spm, ncx>>
     to memref<1x513x8x8xf16, #wafer.memory<spm, ncx>>
  return
}

// -----

func.func @wrong_channels(%input: memref<1x1024x16x8xbf16, #wafer.memory<spm, ncx>>) {
  // expected-error @+1 {{must preserve batch and channel extents}}
  %r = wafer.tile.pool #wafer.reduce_kind<max> %input
      {kernel = array<i64: 2, 2>, strides = array<i64: 2, 2>, dilations = array<i64: 1, 1>}
      : memref<1x1024x16x8xbf16, #wafer.memory<spm, ncx>>
     to memref<1x512x8x9xbf16, #wafer.memory<spm, ncx>>
  return
}

// -----

func.func @zero_kernel(%input: memref<1x1024x16x8xf16, #wafer.memory<spm, ncx>>) {
  // expected-error @+1 {{kernel, stride and dilation must be positive}}
  %r = wafer.tile.pool #wafer.reduce_kind<sum> %input
      {kernel = array<i64: 0, 2>, strides = array<i64: 2, 2>, dilations = array<i64: 1, 1>}
      : memref<1x1024x16x8xf16, #wafer.memory<spm, ncx>>
     to memref<1x512x8x8xf16, #wafer.memory<spm, ncx>>
  return
}
