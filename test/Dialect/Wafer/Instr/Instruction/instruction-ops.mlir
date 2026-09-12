// RUN: wafer-opt %s | FileCheck %s

module {
  %ddr_in = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<ddr, tensor>>
  %ddr_out = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<ddr, tensor>>
  %tensor = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<spm, tensor>>
  %converted = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf32, #wafer.memory<spm, tensor>>
  %i8_tensor = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xi8, #wafer.memory<spm, tensor>>
  %cx = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<spm, cx>>
  %reduce_input = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1x4x8xf16, #wafer.memory<spm, ncx>>
  %reduce_out = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1x4x1xf16, #wafer.memory<spm, ncx>>
  %lhs = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<spm, cx>>
  %rhs = "builtin.unrealized_conversion_cast"()
      : () -> memref<8x16xf16, #wafer.memory<spm, cx>>
  %gemm_out = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x16xf16, #wafer.memory<spm, cx>>
  %f16 = arith.constant 0.000000e+00 : f16

  wafer.instr.rdma %ddr_in to %tensor
      {byte_count = 64 : i64, inner_bytes = 64 : i64,
       src_strides = array<i64: 0, 0, 0>,
       src_iterations = array<i64: 1, 1, 1>}
      : memref<4x8xf16, #wafer.memory<ddr, tensor>>
     to memref<4x8xf16, #wafer.memory<spm, tensor>>

  wafer.instr.gather_scatter %tensor to %cx
      {byte_count = 64 : i64, inner_bytes = 16 : i64,
       src_strides = array<i64: 16, 0, 0>,
       src_iterations = array<i64: 4, 1, 1>,
       dst_strides = array<i64: 16, 0, 0>,
       dst_iterations = array<i64: 4, 1, 1>}
      : memref<4x8xf16, #wafer.memory<spm, tensor>>
     to memref<4x8xf16, #wafer.memory<spm, cx>>

  wafer.instr.gather_scatter %tensor to %cx
      {byte_count = 16 : i64, inner_bytes = 16 : i64,
       src_offset = 16 : i64, dst_offset = 128 : i64,
       src_strides = array<i64: 0, 0, 0>,
       src_iterations = array<i64: 1, 1, 1>,
       dst_strides = array<i64: 0, 0, 0>,
       dst_iterations = array<i64: 1, 1, 1>}
      : memref<4x8xf16, #wafer.memory<spm, tensor>>
     to memref<4x8xf16, #wafer.memory<spm, cx>>

  wafer.instr.fill %tensor, %f16
      : memref<4x8xf16, #wafer.memory<spm, tensor>>, f16

  wafer.instr.elementwise #wafer.instr_elementwise_kind<add> %tensor, %tensor into %tensor
      : memref<4x8xf16, #wafer.memory<spm, tensor>>,
        memref<4x8xf16, #wafer.memory<spm, tensor>>
    into memref<4x8xf16, #wafer.memory<spm, tensor>>

  wafer.instr.reduce #wafer.instr_reduce_kind<sum> %reduce_input into %reduce_out
      {dim = 0 : i64}
      : memref<1x1x4x8xf16, #wafer.memory<spm, ncx>>
    into memref<1x1x4x1xf16, #wafer.memory<spm, ncx>>

  wafer.instr.convert #wafer.instr_convert_kind<fp16_fp32> %tensor into %converted
      : memref<4x8xf16, #wafer.memory<spm, tensor>>
     to memref<4x8xf32, #wafer.memory<spm, tensor>>

  wafer.instr.convert #wafer.instr_convert_kind<fp32_fp16> %converted into %tensor
      {rounding_mode = 0 : i64}
      : memref<4x8xf32, #wafer.memory<spm, tensor>>
     to memref<4x8xf16, #wafer.memory<spm, tensor>>

  wafer.instr.convert #wafer.instr_convert_kind<int8_fp16> %i8_tensor into %tensor
      {zero_point = 0 : i64}
      : memref<4x8xi8, #wafer.memory<spm, tensor>>
     to memref<4x8xf16, #wafer.memory<spm, tensor>>

  wafer.instr.gemm %lhs, %rhs into %gemm_out
      {m = 4 : i64, k = 8 : i64, n = 16 : i64}
      : memref<4x8xf16, #wafer.memory<spm, cx>>,
        memref<8x16xf16, #wafer.memory<spm, cx>>
    into memref<4x16xf16, #wafer.memory<spm, cx>>

  wafer.instr.wdma %tensor to %ddr_out
      {byte_count = 64 : i64, inner_bytes = 64 : i64,
       dst_strides = array<i64: 0, 0, 0>,
       dst_iterations = array<i64: 1, 1, 1>}
      : memref<4x8xf16, #wafer.memory<spm, tensor>>
     to memref<4x8xf16, #wafer.memory<ddr, tensor>>
}

// CHECK: wafer.instr.rdma
// CHECK-SAME: byte_count = 64 : i64
// CHECK: wafer.instr.gather_scatter
// CHECK-SAME: dst_iterations = array<i64: 4, 1, 1>
// CHECK: wafer.instr.gather_scatter
// CHECK-SAME: dst_offset = 128 : i64
// CHECK-SAME: src_offset = 16 : i64
// CHECK: wafer.instr.fill
// CHECK: wafer.instr.elementwise <add>
// CHECK-NOT: indexing_maps
// CHECK: wafer.instr.reduce <sum>
// CHECK-SAME: dim = 0 : i64
// CHECK: wafer.instr.convert <fp16_fp32>
// CHECK: wafer.instr.convert <fp32_fp16>
// CHECK-SAME: rounding_mode = 0 : i64
// CHECK: wafer.instr.convert <int8_fp16>
// CHECK-SAME: zero_point = 0 : i64
// CHECK: wafer.instr.gemm
// CHECK-SAME: m = 4 : i64
// CHECK: wafer.instr.wdma
