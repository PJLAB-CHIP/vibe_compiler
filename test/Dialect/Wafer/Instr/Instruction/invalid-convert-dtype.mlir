// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<spm, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf32, #wafer.memory<spm, tensor>>
  wafer.instr.convert #wafer.instr_convert_kind<fp32_int32> %src into %dst
      : memref<4x8xf16, #wafer.memory<spm, tensor>>
     to memref<4x8xf32, #wafer.memory<spm, tensor>>
}

// CHECK: error: 'wafer.instr.convert' op convert kind source type does not match source element type
