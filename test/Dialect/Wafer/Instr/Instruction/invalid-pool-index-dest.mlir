// RUN: wafer-opt -split-input-file -verify-diagnostics %s

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x8x8x64xf16, #wafer.memory<spm, ncx>>
  %out = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x4x4x64xf16, #wafer.memory<spm, ncx>>

  // expected-error @below {{pool kind expects 2 dest operand}}
  wafer.instr.pool #wafer.instr_pool_kind<indexedmax> %input into %out
      {source_shape = array<i64: 1, 8, 8, 64>,
       dest_shape = array<i64: 1, 4, 4, 64>,
       pads = array<i64: 0, 0, 0, 0>,
       kernel_strides = array<i64: 2, 2, 2, 2>}
      : memref<1x8x8x64xf16, #wafer.memory<spm, ncx>>
    into memref<1x4x4x64xf16, #wafer.memory<spm, ncx>>
}

// -----

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x8x8x64xf16, #wafer.memory<spm, ncx>>
  %out = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x4x4x64xf16, #wafer.memory<spm, ncx>>
  %index = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x4x4x64xi32, #wafer.memory<spm, ncx>>

  // expected-error @below {{indexed pool index dest element type must be i16}}
  wafer.instr.pool #wafer.instr_pool_kind<indexedmin> %input into %out, %index
      {source_shape = array<i64: 1, 8, 8, 64>,
       dest_shape = array<i64: 1, 4, 4, 64>,
       pads = array<i64: 0, 0, 0, 0>,
       kernel_strides = array<i64: 2, 2, 2, 2>}
      : memref<1x8x8x64xf16, #wafer.memory<spm, ncx>>
    into memref<1x4x4x64xf16, #wafer.memory<spm, ncx>>,
         memref<1x4x4x64xi32, #wafer.memory<spm, ncx>>
}
