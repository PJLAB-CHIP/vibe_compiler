// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xi8, #wafer.memory<spm, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<spm, tensor>>
  wafer.instr.convert #wafer.instr_convert_kind<int8_fp16> %src into %dst
      : memref<4x8xi8, #wafer.memory<spm, tensor>>
     to memref<4x8xf16, #wafer.memory<spm, tensor>>
}

// CHECK: error: 'wafer.instr.convert' op convert kind requires zero_point attr
