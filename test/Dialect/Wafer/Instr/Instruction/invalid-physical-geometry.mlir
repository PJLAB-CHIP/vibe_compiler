// RUN: wafer-opt -split-input-file -verify-diagnostics %s

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<8xf16, #wafer.memory<ddr, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<1xf16, #wafer.memory<spm, tensor>>
  // expected-error @below {{destination byte count exceeds physical byte size}}
  wafer.instr.rdma %src to %dst
      {byte_count = 16 : i64, inner_bytes = 16 : i64,
       src_strides = array<i64: 0, 0, 0>,
       src_iterations = array<i64: 1, 1, 1>}
      : memref<8xf16, #wafer.memory<ddr, tensor>>
     to memref<1xf16, #wafer.memory<spm, tensor>>
}

// -----

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x8x8x64xf16, #wafer.memory<spm, ncx>>
  %weight = "builtin.unrealized_conversion_cast"()
      : () -> memref<3x3x64x64xf16, #wafer.memory<spm, ncx>>
  %output = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x8x8x64xf16, #wafer.memory<spm, ncx>>
  // expected-error @below {{input_shape must match the logical buffer shape}}
  wafer.instr.conv #wafer.instr_conv_kind<conv> %input, %weight into %output
      {input_shape = array<i64: 1, 4, 8, 64>,
       weight_shape = array<i64: 3, 3, 64, 64>,
       output_shape = array<i64: 1, 8, 8, 64>,
       pads = array<i64: 1, 1, 1, 1>,
       unpads = array<i64: 0, 0, 0, 0>,
       kernel_strides = array<i64: 3, 3, 1, 1>,
       dilations = array<i64: 1, 1>}
      : memref<1x8x8x64xf16, #wafer.memory<spm, ncx>>,
        memref<3x3x64x64xf16, #wafer.memory<spm, ncx>>
    into memref<1x8x8x64xf16, #wafer.memory<spm, ncx>>
}

// -----

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<1xf16, #wafer.memory<spm, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<8xf16, #wafer.memory<ddr, tensor>>
  // expected-error @below {{source byte count exceeds physical byte size}}
  wafer.instr.wdma %src to %dst
      {byte_count = 16 : i64, inner_bytes = 16 : i64,
       dst_strides = array<i64: 0, 0, 0>,
       dst_iterations = array<i64: 1, 1, 1>}
      : memref<1xf16, #wafer.memory<spm, tensor>>
     to memref<8xf16, #wafer.memory<ddr, tensor>>
}

// -----

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf16, #wafer.memory<spm, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<8xf32, #wafer.memory<spm, tensor>>
  // expected-error @below {{convert source and destination element counts must match}}
  wafer.instr.convert #wafer.instr_convert_kind<fp16_fp32> %src into %dst
      : memref<4xf16, #wafer.memory<spm, tensor>>
     to memref<8xf32, #wafer.memory<spm, tensor>>
}

// -----

module {
  %buf = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf16, #wafer.memory<spm, tensor>>
  // expected-error @below {{DTE byte count exceeds buffer physical byte size}}
  %token = wafer.instr.dte_send %buf {peer = 1 : i64, bytes = 16 : i64,
      message = #wafer.dte_message<communication = 0, phase = collective_permute, round = 0, slice = 0>}
      : memref<4xf16, #wafer.memory<spm, tensor>> -> !async.token
}

// -----

module {
  %lhs = "builtin.unrealized_conversion_cast"()
      : () -> memref<65536x1xf16, #wafer.memory<spm, cx>>
  %rhs = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1xf16, #wafer.memory<spm, cx>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<65536x1xf16, #wafer.memory<spm, cx>>
  // expected-error @below {{m must fit uint16_t}}
  wafer.instr.gemm %lhs, %rhs into %dst
      {m = 65536 : i64, k = 1 : i64, n = 1 : i64}
      : memref<65536x1xf16, #wafer.memory<spm, cx>>,
        memref<1x1xf16, #wafer.memory<spm, cx>>
    into memref<65536x1xf16, #wafer.memory<spm, cx>>
}

// -----

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<?xf16, #wafer.memory<ddr, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<1xf16, #wafer.memory<spm, tensor>>
  // expected-error @below {{source physical byte size must be statically known}}
  wafer.instr.rdma %src to %dst
      {byte_count = 2 : i64, inner_bytes = 2 : i64,
       src_strides = array<i64: 0, 0, 0>,
       src_iterations = array<i64: 1, 1, 1>}
      : memref<?xf16, #wafer.memory<ddr, tensor>>
     to memref<1xf16, #wafer.memory<spm, tensor>>
}

// -----

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<1xf16, #wafer.memory<spm, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<1xf16, #wafer.memory<spm, tensor>>
  // expected-error @below {{descriptor stride must fit uint32_t}}
  wafer.instr.gather_scatter %src to %dst
      {byte_count = 2 : i64, inner_bytes = 2 : i64,
       src_strides = array<i64: 4294967296, 0, 0>,
       src_iterations = array<i64: 1, 1, 1>,
       dst_strides = array<i64: 0, 0, 0>,
       dst_iterations = array<i64: 1, 1, 1>}
      : memref<1xf16, #wafer.memory<spm, tensor>>
     to memref<1xf16, #wafer.memory<spm, tensor>>
}

// -----

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<1xf16, #wafer.memory<spm, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<1xf16, #wafer.memory<spm, tensor>>
  // expected-error @below {{descriptor stride must fit uint32_t}}
  wafer.instr.gather_scatter %src to %dst
      {byte_count = 2 : i64, inner_bytes = 2 : i64,
       src_strides = array<i64: 0, 0, 0>,
       src_iterations = array<i64: 1, 1, 1>,
       dst_strides = array<i64: 4294967296, 0, 0>,
       dst_iterations = array<i64: 1, 1, 1>}
      : memref<1xf16, #wafer.memory<spm, tensor>>
     to memref<1xf16, #wafer.memory<spm, tensor>>
}

// -----

module {
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<4294967296xi8, #wafer.memory<spm, tensor>>
  %value = arith.constant 0 : i8
  // expected-error @below {{fill dest element count must fit uint32_t}}
  wafer.instr.fill %dst, %value
      : memref<4294967296xi8, #wafer.memory<spm, tensor>>, i8
}

// -----

module {
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<1xindex, #wafer.memory<spm, tensor>>
  %value = arith.constant 0 : index
  // expected-error @below {{fill value type must be a target-encodable integer or float}}
  wafer.instr.fill %dst, %value
      : memref<1xindex, #wafer.memory<spm, tensor>>, index
}

// -----

module {
  %buffer = "builtin.unrealized_conversion_cast"()
      : () -> memref<4294967296xi8, #wafer.memory<spm, tensor>>
  // expected-error @below {{elementwise dest element count must fit uint32_t}}
  wafer.instr.elementwise #wafer.instr_elementwise_kind<add> %buffer, %buffer into %buffer
      : memref<4294967296xi8, #wafer.memory<spm, tensor>>,
        memref<4294967296xi8, #wafer.memory<spm, tensor>>
    into memref<4294967296xi8, #wafer.memory<spm, tensor>>
}

// -----

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<4294967296xi1, #wafer.memory<spm, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<4294967296xf16, #wafer.memory<spm, tensor>>
  // expected-error @below {{bit2fp dest element count must fit uint32_t}}
  wafer.instr.bit2fp %src into %dst
      : memref<4294967296xi1, #wafer.memory<spm, tensor>>
     to memref<4294967296xf16, #wafer.memory<spm, tensor>>
}

// -----

module {
  %buffer = "builtin.unrealized_conversion_cast"()
      : () -> memref<4294967296xf16, #wafer.memory<spm, tensor>>
  // expected-error @below {{mask_move dest element count must fit uint32_t}}
  wafer.instr.mask_move %buffer, %buffer into %buffer
      : memref<4294967296xf16, #wafer.memory<spm, tensor>>,
        memref<4294967296xf16, #wafer.memory<spm, tensor>>
    into memref<4294967296xf16, #wafer.memory<spm, tensor>>
}

// -----

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x65536xf16, #wafer.memory<spm, cx>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<1xf16, #wafer.memory<spm, cx>>
  // expected-error @below {{reduce input shape dimension must fit uint16_t}}
  wafer.instr.reduce #wafer.instr_reduce_kind<sum> %input into %dst
      {dim = 0 : i64}
      : memref<1x65536xf16, #wafer.memory<spm, cx>>
    into memref<1xf16, #wafer.memory<spm, cx>>
}

// -----

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<4294967296xf16, #wafer.memory<spm, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<4294967296xf32, #wafer.memory<spm, tensor>>
  // expected-error @below {{convert dest element count must fit uint32_t}}
  wafer.instr.convert #wafer.instr_convert_kind<fp16_fp32> %src into %dst
      : memref<4294967296xf16, #wafer.memory<spm, tensor>>
     to memref<4294967296xf32, #wafer.memory<spm, tensor>>
}

// -----

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<1xf16, #wafer.memory<spm, tensor>>
  %value = "builtin.unrealized_conversion_cast"()
      : () -> memref<1xf16, #wafer.memory<spm, tensor>>
  %index = "builtin.unrealized_conversion_cast"()
      : () -> memref<1xi32, #wafer.memory<spm, tensor>>
  // expected-error @below {{elem_count must fit uint32_t}}
  wafer.instr.peripheral #wafer.instr_peripheral_kind<argmax> %input into %value, %index
      {elem_count = 4294967296 : i64}
      : memref<1xf16, #wafer.memory<spm, tensor>>
    into memref<1xf16, #wafer.memory<spm, tensor>>,
         memref<1xi32, #wafer.memory<spm, tensor>>
}

// -----

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1x1x65536xf16, #wafer.memory<spm, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1x1x65536xf16, #wafer.memory<spm, tensor>>
  // expected-error @below {{source_shape entries must fit uint16_t}}
  wafer.instr.peripheral #wafer.instr_peripheral_kind<bilinear> %src into %dst
      {elem_count = 65536 : i64,
       source_shape = array<i64: 1, 1, 1, 65536>,
       dest_shape = array<i64: 1, 1, 1, 65536>}
      : memref<1x1x1x65536xf16, #wafer.memory<spm, tensor>>
    into memref<1x1x1x65536xf16, #wafer.memory<spm, tensor>>
}

// -----

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1x1x1xf16, #wafer.memory<spm, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1x1x1xf16, #wafer.memory<spm, tensor>>
  // expected-error @below {{pads entries must fit uint16_t}}
  wafer.instr.tdma_data_move #wafer.instr_data_move_kind<pad> %src into %dst
      {source_shape = array<i64: 1, 1, 1, 1>,
       dest_shape = array<i64: 1, 1, 1, 1>,
       pads = array<i64: 65536, 0, 0, 0>}
      : memref<1x1x1x1xf16, #wafer.memory<spm, tensor>>
     to memref<1x1x1x1xf16, #wafer.memory<spm, tensor>>
}

// -----

module {
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<4294967295xi8, #wafer.memory<spm, tensor>>
  %value = arith.constant 0 : i8
  wafer.instr.fill %dst, %value
      : memref<4294967295xi8, #wafer.memory<spm, tensor>>, i8
}

// -----

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<4294967296xi8, #wafer.memory<ddr, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<4294967296xi8, #wafer.memory<spm, tensor>>
  // expected-error @below {{byte_count must fit uint32_t}}
  wafer.instr.rdma %src to %dst
      {byte_count = 4294967296 : i64, inner_bytes = 4294967296 : i64,
       src_strides = array<i64: 0, 0, 0>,
       src_iterations = array<i64: 1, 1, 1>}
      : memref<4294967296xi8, #wafer.memory<ddr, tensor>>
     to memref<4294967296xi8, #wafer.memory<spm, tensor>>
}

// -----

module {
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<1xi64, #wafer.memory<spm, tensor>>
  %value = arith.constant 0 : i64
  // expected-error @below {{fill value type exceeds the uint32_t target scalar ABI}}
  wafer.instr.fill %dst, %value
      : memref<1xi64, #wafer.memory<spm, tensor>>, i64
}

// -----

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1x1x65536xf16, #wafer.memory<spm, ncx>>
  %weight = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1x1x1xf16, #wafer.memory<spm, ncx>>
  %output = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1x1x65536xf16, #wafer.memory<spm, ncx>>
  // expected-error @below {{input_shape entries must fit uint16_t}}
  wafer.instr.conv #wafer.instr_conv_kind<conv> %input, %weight into %output
      {input_shape = array<i64: 1, 1, 1, 65536>,
       weight_shape = array<i64: 1, 1, 1, 1>,
       output_shape = array<i64: 1, 1, 1, 65536>,
       pads = array<i64: 0, 0, 0, 0>,
       unpads = array<i64: 0, 0, 0, 0>,
       kernel_strides = array<i64: 1, 1, 1, 1>,
       dilations = array<i64: 1, 1>}
      : memref<1x1x1x65536xf16, #wafer.memory<spm, ncx>>,
        memref<1x1x1x1xf16, #wafer.memory<spm, ncx>>
    into memref<1x1x1x65536xf16, #wafer.memory<spm, ncx>>
}

// -----

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<1xf16, #wafer.memory<spm, tensor>>
  %lut = "builtin.unrealized_conversion_cast"()
      : () -> memref<1xf16, #wafer.memory<spm, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<1xf16, #wafer.memory<spm, tensor>>
  // expected-error @below {{lut_elem_count must fit uint32_t}}
  wafer.instr.peripheral #wafer.instr_peripheral_kind<lut16> %input, %lut into %dst
      {elem_count = 1 : i64, lut_elem_count = -1 : i64}
      : memref<1xf16, #wafer.memory<spm, tensor>>,
        memref<1xf16, #wafer.memory<spm, tensor>>
    into memref<1xf16, #wafer.memory<spm, tensor>>
}

// -----

module {
  %buffer = "builtin.unrealized_conversion_cast"()
      : () -> memref<1xf16, #wafer.memory<spm, tensor>>
  // expected-error @below {{DTE peer must fit uint32_t}}
  %token = wafer.instr.dte_send %buffer
      {peer = 4294967296 : i64, bytes = 2 : i64,
       message = #wafer.dte_message<communication = 0, phase = collective_permute, round = 0, slice = 0>}
      : memref<1xf16, #wafer.memory<spm, tensor>> -> !async.token
}

// -----

module {
  %buffer = "builtin.unrealized_conversion_cast"()
      : () -> memref<1xf16, #wafer.memory<spm, tensor>>
  // expected-error @below {{DTE slot must fit uint32_t}}
  %token = wafer.instr.dte_recv %buffer
      {peer = 1 : i64, bytes = 2 : i64, slot = 4294967296 : i64,
       message = #wafer.dte_message<communication = 0, phase = collective_permute, round = 0, slice = 0>}
      : memref<1xf16, #wafer.memory<spm, tensor>> -> !async.token
}

// -----

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<1xf16, #wafer.memory<spm, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<?xf16, #wafer.memory<ddr, tensor>>
  // expected-error @below {{target_geometry_mismatch: destination physical byte size must be statically known}}
  wafer.instr.wdma %src to %dst
      {byte_count = 2 : i64, inner_bytes = 2 : i64,
       dst_strides = array<i64: 0, 0, 0>,
       dst_iterations = array<i64: 1, 1, 1>}
      : memref<1xf16, #wafer.memory<spm, tensor>>
     to memref<?xf16, #wafer.memory<ddr, tensor>>
}

// -----

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<?xf16, #wafer.memory<spm, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<1xf16, #wafer.memory<spm, tensor>>
  // expected-error @below {{target_geometry_mismatch: source physical byte size must be statically known}}
  wafer.instr.gather_scatter %src to %dst
      {byte_count = 2 : i64, inner_bytes = 2 : i64,
       src_strides = array<i64: 0, 0, 0>,
       src_iterations = array<i64: 1, 1, 1>,
       dst_strides = array<i64: 0, 0, 0>,
       dst_iterations = array<i64: 1, 1, 1>}
      : memref<?xf16, #wafer.memory<spm, tensor>>
     to memref<1xf16, #wafer.memory<spm, tensor>>
}

// -----

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf16, #wafer.memory<spm, tensor>>
  %value = "builtin.unrealized_conversion_cast"()
      : () -> memref<1xf16, #wafer.memory<spm, tensor>>
  %index = "builtin.unrealized_conversion_cast"()
      : () -> memref<1xi32, #wafer.memory<spm, tensor>>
  // expected-error @below {{target_geometry_mismatch: peripheral primary input element count must equal elem_count}}
  wafer.instr.peripheral #wafer.instr_peripheral_kind<argmax> %input into %value, %index
      {elem_count = 3 : i64}
      : memref<4xf16, #wafer.memory<spm, tensor>>
    into memref<1xf16, #wafer.memory<spm, tensor>>,
         memref<1xi32, #wafer.memory<spm, tensor>>
}

// -----

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf16, #wafer.memory<spm, tensor>>
  %lut = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf16, #wafer.memory<spm, tensor>>
  %dest = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf16, #wafer.memory<spm, tensor>>
  // expected-error @below {{target_geometry_mismatch: LUT table element count must equal elem_count}}
  wafer.instr.peripheral #wafer.instr_peripheral_kind<lut16> %input, %lut into %dest
      {elem_count = 4 : i64, lut_elem_count = 3 : i64}
      : memref<4xf16, #wafer.memory<spm, tensor>>,
        memref<4xf16, #wafer.memory<spm, tensor>>
    into memref<4xf16, #wafer.memory<spm, tensor>>
}

// -----

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<2xf16, #wafer.memory<ddr, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<2xf16, #wafer.memory<spm, tensor>>
  // expected-error @below {{target_geometry_mismatch: inner_bytes must be divisible by the target element byte width}}
  wafer.instr.rdma %src to %dst
      {byte_count = 3 : i64, inner_bytes = 3 : i64,
       src_strides = array<i64: 0, 0, 0>,
       src_iterations = array<i64: 1, 1, 1>}
      : memref<2xf16, #wafer.memory<ddr, tensor>>
     to memref<2xf16, #wafer.memory<spm, tensor>>
}

// -----

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<2xf32, #wafer.memory<spm, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<2xf16, #wafer.memory<ddr, tensor>>
  // expected-error @below {{target_geometry_mismatch: movement source and destination element types must match}}
  wafer.instr.wdma %src to %dst
      {byte_count = 4 : i64, inner_bytes = 4 : i64,
       dst_strides = array<i64: 0, 0, 0>,
       dst_iterations = array<i64: 1, 1, 1>}
      : memref<2xf32, #wafer.memory<spm, tensor>>
     to memref<2xf16, #wafer.memory<ddr, tensor>>
}

// -----

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<4294967296xi1, #wafer.memory<ddr, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<4294967296xi1, #wafer.memory<spm, tensor>>
  // expected-error @below {{target_abi_narrowing: bitpacked BOOL inner_bytes cannot be converted to a uint32_t logical element count}}
  wafer.instr.rdma %src to %dst
      {byte_count = 536870912 : i64, inner_bytes = 536870912 : i64,
       src_strides = array<i64: 0, 0, 0>,
       src_iterations = array<i64: 1, 1, 1>}
      : memref<4294967296xi1, #wafer.memory<ddr, tensor>>
     to memref<4294967296xi1, #wafer.memory<spm, tensor>>
}

// -----

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x8x8x64xf16, #wafer.memory<spm, ncx>>
  %weight = "builtin.unrealized_conversion_cast"()
      : () -> memref<3x3x64x64xf16, #wafer.memory<spm, ncx>>
  %output = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x99x99x64xf16, #wafer.memory<spm, ncx>>
  // expected-error @below {{target_geometry_mismatch: convolution output spatial shape does not match input/kernel/stride/dilation/pad/unpad}}
  wafer.instr.conv #wafer.instr_conv_kind<conv> %input, %weight into %output
      {input_shape = array<i64: 1, 8, 8, 64>,
       weight_shape = array<i64: 3, 3, 64, 64>,
       output_shape = array<i64: 1, 99, 99, 64>,
       pads = array<i64: 1, 1, 1, 1>,
       unpads = array<i64: 0, 0, 0, 0>,
       kernel_strides = array<i64: 3, 3, 1, 1>,
       dilations = array<i64: 1, 1>}
      : memref<1x8x8x64xf16, #wafer.memory<spm, ncx>>,
        memref<3x3x64x64xf16, #wafer.memory<spm, ncx>>
    into memref<1x99x99x64xf16, #wafer.memory<spm, ncx>>
}

// -----

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x8x8x16xf16, #wafer.memory<spm, ncx>>
  %output = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x5x4x16xf16, #wafer.memory<spm, ncx>>
  // expected-error @below {{target_geometry_mismatch: pool destination spatial shape does not match source/kernel/stride/pad}}
  wafer.instr.pool #wafer.instr_pool_kind<max> %input into %output
      {source_shape = array<i64: 1, 8, 8, 16>,
       dest_shape = array<i64: 1, 5, 4, 16>,
       pads = array<i64: 0, 0, 0, 0>,
       kernel_strides = array<i64: 2, 2, 2, 2>}
      : memref<1x8x8x16xf16, #wafer.memory<spm, ncx>>
    into memref<1x5x4x16xf16, #wafer.memory<spm, ncx>>
}

// -----

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x2x2x4xf16, #wafer.memory<spm, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x4x5x4xf16, #wafer.memory<spm, tensor>>
  // expected-error @below {{target_geometry_mismatch: pad destination shape does not match source and pads}}
  wafer.instr.tdma_data_move #wafer.instr_data_move_kind<pad> %src into %dst
      {source_shape = array<i64: 1, 2, 2, 4>,
       dest_shape = array<i64: 1, 4, 5, 4>,
       pads = array<i64: 1, 1, 1, 1>}
      : memref<1x2x2x4xf16, #wafer.memory<spm, tensor>>
     to memref<1x4x5x4xf16, #wafer.memory<spm, tensor>>
}
