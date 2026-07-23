// RUN: wafer-opt -split-input-file -verify-diagnostics %s -o /dev/null

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<4096x1x1x1xf16, #wafer.memory<spm, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<4096x1x1x1xf16, #wafer.memory<spm, tensor>>
  wafer.instr.peripheral #wafer.instr_peripheral_kind<bilinear> %src into %dst
      {elem_count = 4096 : i64,
       source_shape = array<i64: 4096, 1, 1, 1>,
       dest_shape = array<i64: 4096, 1, 1, 1>}
      : memref<4096x1x1x1xf16, #wafer.memory<spm, tensor>>
    into memref<4096x1x1x1xf16, #wafer.memory<spm, tensor>>
}

// -----

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x4096x1x1xf16, #wafer.memory<spm, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x4096x1x1xf16, #wafer.memory<spm, tensor>>
  wafer.instr.peripheral #wafer.instr_peripheral_kind<bilinear> %src into %dst
      {elem_count = 4096 : i64,
       source_shape = array<i64: 1, 4096, 1, 1>,
       dest_shape = array<i64: 1, 4096, 1, 1>}
      : memref<1x4096x1x1xf16, #wafer.memory<spm, tensor>>
    into memref<1x4096x1x1xf16, #wafer.memory<spm, tensor>>
}

// -----

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1x4096x1xf16, #wafer.memory<spm, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1x4096x1xf16, #wafer.memory<spm, tensor>>
  wafer.instr.peripheral #wafer.instr_peripheral_kind<bilinear> %src into %dst
      {elem_count = 4096 : i64,
       source_shape = array<i64: 1, 1, 4096, 1>,
       dest_shape = array<i64: 1, 1, 4096, 1>}
      : memref<1x1x4096x1xf16, #wafer.memory<spm, tensor>>
    into memref<1x1x4096x1xf16, #wafer.memory<spm, tensor>>
}

// -----

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1x1x16384xf16, #wafer.memory<spm, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1x1x16384xf16, #wafer.memory<spm, tensor>>
  wafer.instr.peripheral #wafer.instr_peripheral_kind<bilinear> %src into %dst
      {elem_count = 16384 : i64,
       source_shape = array<i64: 1, 1, 1, 16384>,
       dest_shape = array<i64: 1, 1, 1, 16384>}
      : memref<1x1x1x16384xf16, #wafer.memory<spm, tensor>>
    into memref<1x1x1x16384xf16, #wafer.memory<spm, tensor>>
}

// -----

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1x1x1xf16, #wafer.memory<spm, ncx>>
  %weight = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1x1x1xf16, #wafer.memory<spm, ncx>>
  %output = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1x1x1xf16, #wafer.memory<spm, ncx>>
  wafer.instr.conv #wafer.instr_conv_kind<conv> %input, %weight into %output
      {input_shape = array<i64: 1, 1, 1, 1>,
       weight_shape = array<i64: 1, 1, 1, 1>,
       output_shape = array<i64: 1, 1, 1, 1>,
       pads = array<i64: 1023, 0, 0, 0>,
       unpads = array<i64: 1023, 0, 0, 0>,
       kernel_strides = array<i64: 1, 1, 1, 1>,
       dilations = array<i64: 1, 1>}
      : memref<1x1x1x1xf16, #wafer.memory<spm, ncx>>,
        memref<1x1x1x1xf16, #wafer.memory<spm, ncx>>
    into memref<1x1x1x1xf16, #wafer.memory<spm, ncx>>
}

// -----

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x255x255x1xf16, #wafer.memory<spm, ncx>>
  %weight = "builtin.unrealized_conversion_cast"()
      : () -> memref<255x255x1x1xf16, #wafer.memory<spm, ncx>>
  %output = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1x1x1xf16, #wafer.memory<spm, ncx>>
  wafer.instr.conv #wafer.instr_conv_kind<conv> %input, %weight into %output
      {input_shape = array<i64: 1, 255, 255, 1>,
       weight_shape = array<i64: 255, 255, 1, 1>,
       output_shape = array<i64: 1, 1, 1, 1>,
       pads = array<i64: 0, 0, 0, 0>,
       unpads = array<i64: 0, 0, 0, 0>,
       kernel_strides = array<i64: 255, 255, 1023, 1023>,
       dilations = array<i64: 1, 1>}
      : memref<1x255x255x1xf16, #wafer.memory<spm, ncx>>,
        memref<255x255x1x1xf16, #wafer.memory<spm, ncx>>
    into memref<1x1x1x1xf16, #wafer.memory<spm, ncx>>
}

// -----

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1024x1x1xf16, #wafer.memory<spm, ncx>>
  %weight = "builtin.unrealized_conversion_cast"()
      : () -> memref<2x1x1x1xf16, #wafer.memory<spm, ncx>>
  %output = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1x1x1xf16, #wafer.memory<spm, ncx>>
  wafer.instr.conv #wafer.instr_conv_kind<conv> %input, %weight into %output
      {input_shape = array<i64: 1, 1024, 1, 1>,
       weight_shape = array<i64: 2, 1, 1, 1>,
       output_shape = array<i64: 1, 1, 1, 1>,
       pads = array<i64: 0, 0, 0, 0>,
       unpads = array<i64: 0, 0, 0, 0>,
       kernel_strides = array<i64: 2, 1, 1, 1>,
       dilations = array<i64: 1023, 1023>}
      : memref<1x1024x1x1xf16, #wafer.memory<spm, ncx>>,
        memref<2x1x1x1xf16, #wafer.memory<spm, ncx>>
    into memref<1x1x1x1xf16, #wafer.memory<spm, ncx>>
}

// -----

module {
  %lhs = "builtin.unrealized_conversion_cast"()
      : () -> memref<4096x1x16384xf16, #wafer.memory<spm, ncx>>
  %rhs = "builtin.unrealized_conversion_cast"()
      : () -> memref<4096x16384x1xf16, #wafer.memory<spm, ncx>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<4096x1x1xf16, #wafer.memory<spm, ncx>>
  wafer.instr.gemm %lhs, %rhs into %dst
      {m = 1 : i64, k = 16384 : i64, n = 1 : i64,
       batch_count = 4096 : i64,
       lhs_batch_dims = array<i64: 0>,
       lhs_m_dim = 1 : i64,
       lhs_contracting_dim = 2 : i64,
       rhs_batch_dims = array<i64: 0>,
       rhs_contracting_dim = 1 : i64,
       rhs_n_dim = 2 : i64,
       result_batch_dims = array<i64: 0>,
       result_m_dim = 1 : i64,
       result_n_dim = 2 : i64}
      : memref<4096x1x16384xf16, #wafer.memory<spm, ncx>>,
        memref<4096x16384x1xf16, #wafer.memory<spm, ncx>>
    into memref<4096x1x1xf16, #wafer.memory<spm, ncx>>
}
