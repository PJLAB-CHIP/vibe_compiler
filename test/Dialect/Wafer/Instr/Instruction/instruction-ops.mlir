// RUN: wafer-opt %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

module {
  %ddr_in = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<ddr, tensor>>
  %ddr_out = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<ddr, tensor>>
  %tensor = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<spm, tensor>>
  %cx = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<spm, cx>>
  %reduce_out = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf16, #wafer.memory<spm, cx>>
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

  wafer.instr.fill %tensor, %f16
      : memref<4x8xf16, #wafer.memory<spm, tensor>>, f16

  wafer.instr.elementwise #wafer.elementwise_kind<add> %tensor, %tensor into %tensor
      {indexing_maps = [#map, #map, #map]}
      : memref<4x8xf16, #wafer.memory<spm, tensor>>,
        memref<4x8xf16, #wafer.memory<spm, tensor>>
    into memref<4x8xf16, #wafer.memory<spm, tensor>>

  wafer.instr.reduce #wafer.reduce_kind<sum> %cx into %reduce_out, %f16 : f16
      {dimensions = array<i64: 1>}
      : memref<4x8xf16, #wafer.memory<spm, cx>>
    into memref<4xf16, #wafer.memory<spm, cx>>

  wafer.instr.convert %tensor into %cx
      {src_dtype = f16, dst_dtype = f16}
      : memref<4x8xf16, #wafer.memory<spm, tensor>>
     to memref<4x8xf16, #wafer.memory<spm, cx>>

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
// CHECK: wafer.instr.fill
// CHECK: wafer.instr.elementwise <add>
// CHECK-SAME: indexing_maps
// CHECK: wafer.instr.reduce <sum>
// CHECK-SAME: dimensions = array<i64: 1>
// CHECK: wafer.instr.convert
// CHECK-SAME: dst_dtype = f16
// CHECK-SAME: src_dtype = f16
// CHECK: wafer.instr.gemm
// CHECK-SAME: m = 4 : i64
// CHECK: wafer.instr.wdma
