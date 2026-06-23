// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %buf = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf32, #wafer.memory<ddr, tensor>>
  %send = wafer.instr.dte_send %buf {peer = 1 : i64, bytes = 16 : i64}
      : memref<4xf32, #wafer.memory<ddr, tensor>> -> !async.token
}

// CHECK: operand #0 must be Wafer SPM memref
