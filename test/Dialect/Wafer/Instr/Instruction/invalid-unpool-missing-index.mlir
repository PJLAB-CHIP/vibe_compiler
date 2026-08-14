// RUN: wafer-opt -split-input-file -verify-diagnostics %s

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x4x4x64xf16, #wafer.memory<spm, ncx>>
  %dest = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x8x8x64xf16, #wafer.memory<spm, ncx>>

  // expected-error @below {{unpool kind requires index operand}}
  wafer.instr.unpool #wafer.instr_unpool_kind<mask> %input into %dest
      {source_shape = array<i64: 1, 4, 4, 64>,
       dest_shape = array<i64: 1, 8, 8, 64>,
       kernel_strides = array<i64: 2, 2, 2, 2>}
      : memref<1x4x4x64xf16, #wafer.memory<spm, ncx>>
    into memref<1x8x8x64xf16, #wafer.memory<spm, ncx>>
}

// -----

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x4x4x64xf16, #wafer.memory<spm, ncx>>
  %index = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x4x4x64xi32, #wafer.memory<spm, ncx>>
  %dest = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x8x8x64xf16, #wafer.memory<spm, ncx>>

  // expected-error @below {{index element type must be i16}}
  wafer.instr.unpool #wafer.instr_unpool_kind<mask> %input, %index into %dest
      {source_shape = array<i64: 1, 4, 4, 64>,
       dest_shape = array<i64: 1, 8, 8, 64>,
       kernel_strides = array<i64: 2, 2, 2, 2>}
      : memref<1x4x4x64xf16, #wafer.memory<spm, ncx>>,
        memref<1x4x4x64xi32, #wafer.memory<spm, ncx>>
    into memref<1x8x8x64xf16, #wafer.memory<spm, ncx>>
}

// -----

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x4x4x64xf16, #wafer.memory<spm, ncx>>
  %index = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x4x4x32xi16, #wafer.memory<spm, ncx>>
  %dest = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x8x8x64xf16, #wafer.memory<spm, ncx>>

  // expected-error @below {{index source_shape must match the logical buffer shape}}
  wafer.instr.unpool #wafer.instr_unpool_kind<mask> %input, %index into %dest
      {source_shape = array<i64: 1, 4, 4, 64>,
       dest_shape = array<i64: 1, 8, 8, 64>,
       kernel_strides = array<i64: 2, 2, 2, 2>}
      : memref<1x4x4x64xf16, #wafer.memory<spm, ncx>>,
        memref<1x4x4x32xi16, #wafer.memory<spm, ncx>>
    into memref<1x8x8x64xf16, #wafer.memory<spm, ncx>>
}

// -----

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x4x4x64xf16, #wafer.memory<spm, ncx>>
  %index = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x4x4x64xi16, #wafer.memory<spm, ncx>>
  %dest = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x8x8x64xf16, #wafer.memory<spm, ncx>>

  // expected-error @below {{avg unpool must not have index operand}}
  wafer.instr.unpool #wafer.instr_unpool_kind<avg> %input, %index into %dest
      {source_shape = array<i64: 1, 4, 4, 64>,
       dest_shape = array<i64: 1, 8, 8, 64>,
       kernel_strides = array<i64: 2, 2, 2, 2>}
      : memref<1x4x4x64xf16, #wafer.memory<spm, ncx>>,
        memref<1x4x4x64xi16, #wafer.memory<spm, ncx>>
    into memref<1x8x8x64xf16, #wafer.memory<spm, ncx>>
}

// -----

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x4x4x64xf16, #wafer.memory<spm, ncx>>
  %dest = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x8x8x64xf16, #wafer.memory<spm, ncx>>

  // expected-error @below {{does not accept schema-free semantic attribute 'index'}}
  wafer.instr.unpool #wafer.instr_unpool_kind<avg> %input into %dest
      {source_shape = array<i64: 1, 4, 4, 64>,
       dest_shape = array<i64: 1, 8, 8, 64>,
       kernel_strides = array<i64: 2, 2, 2, 2>,
       index = 0 : i64}
      : memref<1x4x4x64xf16, #wafer.memory<spm, ncx>>
    into memref<1x8x8x64xf16, #wafer.memory<spm, ncx>>
}
