// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<16xf16, #wafer.memory<spm, tensor>>

  "wafer.instr.peripheral"(%input) <{
      kind = #wafer.instr_peripheral_kind<count>,
      elem_count = 16 : i64,
      operandSegmentSizes = array<i32: 1, 0>
    }> : (memref<16xf16, #wafer.memory<spm, tensor>>) -> ()
}

// CHECK: error: 'wafer.instr.peripheral' op count peripheral writeback is not represented in instruction IR
