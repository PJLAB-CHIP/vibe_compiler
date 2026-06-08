// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<spm, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<spm, cx>>
  wafer.instr.gather_scatter %src to %dst
      {byte_count = 16 : i64, inner_bytes = 16 : i64,
       src_offset = 64 : i64,
       src_strides = array<i64: 0, 0, 0>,
       src_iterations = array<i64: 1, 1, 1>,
       dst_strides = array<i64: 0, 0, 0>,
       dst_iterations = array<i64: 1, 1, 1>}
      : memref<4x8xf16, #wafer.memory<spm, tensor>>
     to memref<4x8xf16, #wafer.memory<spm, cx>>
}

// CHECK: error: 'wafer.instr.gather_scatter' op source descriptor byte range exceeds physical byte size
