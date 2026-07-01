// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %lhs = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<spm, tensor>>
  %rhs = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x7xf16, #wafer.memory<spm, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<spm, tensor>>
  wafer.instr.elementwise #wafer.instr_elementwise_kind<add> %lhs, %rhs into %dst
      : memref<4x8xf16, #wafer.memory<spm, tensor>>,
        memref<4x7xf16, #wafer.memory<spm, tensor>>
    into memref<4x8xf16, #wafer.memory<spm, tensor>>
}

// CHECK: error: 'wafer.instr.elementwise' op elementwise operand tensor types must match result tensor type
