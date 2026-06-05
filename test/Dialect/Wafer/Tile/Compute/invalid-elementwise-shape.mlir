// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %a = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<spm, tensor>>
  %b = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x7xf16, #wafer.memory<spm, tensor>>
  %sum = wafer.tile.elementwise #wafer.elementwise_kind<add> %a, %b
      : (memref<4x8xf16, #wafer.memory<spm, tensor>>,
         memref<4x7xf16, #wafer.memory<spm, tensor>>)
     -> memref<4x8xf16, #wafer.memory<spm, tensor>>
}

// CHECK: elementwise operand tensor types must match result tensor type
