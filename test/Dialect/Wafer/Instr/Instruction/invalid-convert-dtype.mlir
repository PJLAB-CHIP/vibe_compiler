// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<spm, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf32, #wafer.memory<spm, tensor>>
  wafer.instr.convert %src into %dst
      {src_dtype = f32, dst_dtype = f32}
      : memref<4x8xf16, #wafer.memory<spm, tensor>>
     to memref<4x8xf32, #wafer.memory<spm, tensor>>
}

// CHECK: error: 'wafer.instr.convert' op src_dtype must match source element type
