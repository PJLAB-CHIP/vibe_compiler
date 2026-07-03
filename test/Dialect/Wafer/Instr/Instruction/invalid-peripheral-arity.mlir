// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<16xf16, #wafer.memory<spm, tensor>>
  %value = "builtin.unrealized_conversion_cast"()
      : () -> memref<1xf16, #wafer.memory<spm, tensor>>

  wafer.instr.peripheral #wafer.instr_peripheral_kind<argmax> %input into %value
      {elem_count = 16 : i64}
      : memref<16xf16, #wafer.memory<spm, tensor>>
    into memref<1xf16, #wafer.memory<spm, tensor>>
}

// CHECK: peripheral kind expects 2 dest operand
