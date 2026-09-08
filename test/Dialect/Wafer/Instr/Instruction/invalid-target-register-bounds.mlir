// RUN: wafer-opt -split-input-file -verify-diagnostics %s

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<4097x1x1x1xf16, #wafer.memory<spm, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<4097x1x1x1xf16, #wafer.memory<spm, tensor>>
  // expected-error @below {{source_shape N dimension must be in [1, 4096]}}
  wafer.instr.peripheral #wafer.instr_peripheral_kind<bilinear> %src into %dst
      {elem_count = 4097 : i64,
       source_shape = array<i64: 4097, 1, 1, 1>,
       dest_shape = array<i64: 4097, 1, 1, 1>}
      : memref<4097x1x1x1xf16, #wafer.memory<spm, tensor>>
    into memref<4097x1x1x1xf16, #wafer.memory<spm, tensor>>
}

// -----

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x4097x1x1xf16, #wafer.memory<spm, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x4097x1x1xf16, #wafer.memory<spm, tensor>>
  // expected-error @below {{source_shape H dimension must be in [1, 4096]}}
  wafer.instr.peripheral #wafer.instr_peripheral_kind<bilinear> %src into %dst
      {elem_count = 4097 : i64,
       source_shape = array<i64: 1, 4097, 1, 1>,
       dest_shape = array<i64: 1, 4097, 1, 1>}
      : memref<1x4097x1x1xf16, #wafer.memory<spm, tensor>>
    into memref<1x4097x1x1xf16, #wafer.memory<spm, tensor>>
}

// -----

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1x4097x1xf16, #wafer.memory<spm, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1x4097x1xf16, #wafer.memory<spm, tensor>>
  // expected-error @below {{source_shape W dimension must be in [1, 4096]}}
  wafer.instr.peripheral #wafer.instr_peripheral_kind<bilinear> %src into %dst
      {elem_count = 4097 : i64,
       source_shape = array<i64: 1, 1, 4097, 1>,
       dest_shape = array<i64: 1, 1, 4097, 1>}
      : memref<1x1x4097x1xf16, #wafer.memory<spm, tensor>>
    into memref<1x1x4097x1xf16, #wafer.memory<spm, tensor>>
}

// -----

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1x1x16385xf16, #wafer.memory<spm, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1x1x16385xf16, #wafer.memory<spm, tensor>>
  // expected-error @below {{source_shape C dimension must be in [1, 16384]}}
  wafer.instr.peripheral #wafer.instr_peripheral_kind<bilinear> %src into %dst
      {elem_count = 16385 : i64,
       source_shape = array<i64: 1, 1, 1, 16385>,
       dest_shape = array<i64: 1, 1, 1, 16385>}
      : memref<1x1x1x16385xf16, #wafer.memory<spm, tensor>>
    into memref<1x1x1x16385xf16, #wafer.memory<spm, tensor>>
}

// -----

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1x1x1xf16, #wafer.memory<spm, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1025x1x1xf16, #wafer.memory<spm, tensor>>
  // expected-error @below {{pads entries must be in [0, 1023]}}
  wafer.instr.tdma_data_move #wafer.instr_data_move_kind<pad> %src into %dst
      {source_shape = array<i64: 1, 1, 1, 1>,
       dest_shape = array<i64: 1, 1025, 1, 1>,
       pads = array<i64: 1024, 0, 0, 0>}
      : memref<1x1x1x1xf16, #wafer.memory<spm, tensor>>
     to memref<1x1025x1x1xf16, #wafer.memory<spm, tensor>>
}

// -----

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x2049x1x1xf16, #wafer.memory<spm, ncx>>
  %weight = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1x1x1xf16, #wafer.memory<spm, cx>>
  %output = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1025x1x1xf16, #wafer.memory<spm, ncx>>
  // expected-error @below {{unpads entries must be in [0, 1023]}}
  wafer.instr.conv #wafer.instr_conv_kind<conv> %input, %weight into %output
      {input_shape = array<i64: 1, 2049, 1, 1>,
       weight_shape = array<i64: 1, 1, 1, 1>,
       output_shape = array<i64: 1, 1025, 1, 1>,
       pads = array<i64: 0, 0, 0, 0>,
       unpads = array<i64: 1024, 0, 0, 0>,
       kernel_strides = array<i64: 1, 1, 1, 1>,
       dilations = array<i64: 1, 1>}
      : memref<1x2049x1x1xf16, #wafer.memory<spm, ncx>>,
        memref<1x1x1x1xf16, #wafer.memory<spm, cx>>
    into memref<1x1025x1x1xf16, #wafer.memory<spm, ncx>>
}

// -----

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x256x1x1xf16, #wafer.memory<spm, ncx>>
  %weight = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x256x1x1xf16, #wafer.memory<spm, cx>>
  %output = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1x1x1xf16, #wafer.memory<spm, ncx>>
  // expected-error @below {{kernel_strides kernel entries must be in [1, 255]}}
  wafer.instr.conv #wafer.instr_conv_kind<conv> %input, %weight into %output
      {input_shape = array<i64: 1, 256, 1, 1>,
       weight_shape = array<i64: 1, 256, 1, 1>,
       output_shape = array<i64: 1, 1, 1, 1>,
       pads = array<i64: 0, 0, 0, 0>,
       unpads = array<i64: 0, 0, 0, 0>,
       kernel_strides = array<i64: 256, 1, 1, 1>,
       dilations = array<i64: 1, 1>}
      : memref<1x256x1x1xf16, #wafer.memory<spm, ncx>>,
        memref<1x256x1x1xf16, #wafer.memory<spm, cx>>
    into memref<1x1x1x1xf16, #wafer.memory<spm, ncx>>
}

// -----

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1025x1x1xf16, #wafer.memory<spm, ncx>>
  %weight = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1x1x1xf16, #wafer.memory<spm, cx>>
  %output = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x2x1x1xf16, #wafer.memory<spm, ncx>>
  // expected-error @below {{kernel_strides stride entries must be in [1, 1023]}}
  wafer.instr.conv #wafer.instr_conv_kind<conv> %input, %weight into %output
      {input_shape = array<i64: 1, 1025, 1, 1>,
       weight_shape = array<i64: 1, 1, 1, 1>,
       output_shape = array<i64: 1, 2, 1, 1>,
       pads = array<i64: 0, 0, 0, 0>,
       unpads = array<i64: 0, 0, 0, 0>,
       kernel_strides = array<i64: 1, 1, 1024, 1>,
       dilations = array<i64: 1, 1>}
      : memref<1x1025x1x1xf16, #wafer.memory<spm, ncx>>,
        memref<1x1x1x1xf16, #wafer.memory<spm, cx>>
    into memref<1x2x1x1xf16, #wafer.memory<spm, ncx>>
}

// -----

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1025x1x1xf16, #wafer.memory<spm, ncx>>
  %weight = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x2x1x1xf16, #wafer.memory<spm, cx>>
  %output = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1x1x1xf16, #wafer.memory<spm, ncx>>
  // expected-error @below {{dilations entries must be in [1, 1023]}}
  wafer.instr.conv #wafer.instr_conv_kind<conv> %input, %weight into %output
      {input_shape = array<i64: 1, 1025, 1, 1>,
       weight_shape = array<i64: 1, 2, 1, 1>,
       output_shape = array<i64: 1, 1, 1, 1>,
       pads = array<i64: 0, 0, 0, 0>,
       unpads = array<i64: 0, 0, 0, 0>,
       kernel_strides = array<i64: 2, 1, 1, 1>,
       dilations = array<i64: 1024, 1>}
      : memref<1x1025x1x1xf16, #wafer.memory<spm, ncx>>,
        memref<1x2x1x1xf16, #wafer.memory<spm, cx>>
    into memref<1x1x1x1xf16, #wafer.memory<spm, ncx>>
}

// -----

module {
  %lhs = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x16385xf16, #wafer.memory<spm, cx>>
  %rhs = "builtin.unrealized_conversion_cast"()
      : () -> memref<16385x1xf16, #wafer.memory<spm, cx>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1xf16, #wafer.memory<spm, cx>>
  // expected-error @below {{k must be in [1, 16384]}}
  wafer.instr.gemm %lhs, %rhs into %dst
      {m = 1 : i64, k = 16385 : i64, n = 1 : i64}
      : memref<1x16385xf16, #wafer.memory<spm, cx>>,
        memref<16385x1xf16, #wafer.memory<spm, cx>>
    into memref<1x1xf16, #wafer.memory<spm, cx>>
}

// -----

module {
  %lhs = "builtin.unrealized_conversion_cast"()
      : () -> memref<4097x1x1xf16, #wafer.memory<spm, ncx>>
  %rhs = "builtin.unrealized_conversion_cast"()
      : () -> memref<4097x1x1xf16, #wafer.memory<spm, ncx>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<4097x1x1xf16, #wafer.memory<spm, ncx>>
  // expected-error @below {{batch_count must be in [1, 4096]}}
  wafer.instr.gemm %lhs, %rhs into %dst
      {m = 1 : i64, k = 1 : i64, n = 1 : i64,
       batch_count = 4097 : i64,
       lhs_batch_dims = array<i64: 0>,
       lhs_m_dim = 1 : i64,
       lhs_contracting_dim = 2 : i64,
       rhs_batch_dims = array<i64: 0>,
       rhs_contracting_dim = 1 : i64,
       rhs_n_dim = 2 : i64,
       result_batch_dims = array<i64: 0>,
       result_m_dim = 1 : i64,
       result_n_dim = 2 : i64}
      : memref<4097x1x1xf16, #wafer.memory<spm, ncx>>,
        memref<4097x1x1xf16, #wafer.memory<spm, ncx>>
    into memref<4097x1x1xf16, #wafer.memory<spm, ncx>>
}
