// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<16xf16, #wafer.memory<spm, tensor>>
  %lut = "builtin.unrealized_conversion_cast"()
      : () -> memref<16xf16, #wafer.memory<spm, tensor>>
  %dest = "builtin.unrealized_conversion_cast"()
      : () -> memref<16xf16, #wafer.memory<spm, tensor>>

  wafer.instr.peripheral #wafer.instr_peripheral_kind<lut16> %input, %lut into %dest
      {elem_count = 16 : i64}
      : memref<16xf16, #wafer.memory<spm, tensor>>,
        memref<16xf16, #wafer.memory<spm, tensor>>
    into memref<16xf16, #wafer.memory<spm, tensor>>
}

// CHECK: error: 'wafer.instr.peripheral' op peripheral kind requires lut_elem_count attr
