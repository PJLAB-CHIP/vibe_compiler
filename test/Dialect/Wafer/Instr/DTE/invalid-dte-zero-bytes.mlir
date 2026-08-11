// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %buf = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf32, #wafer.memory<spm, tensor>>
  %recv = wafer.instr.dte_recv %buf {peer = 0 : i64, bytes = 0 : i64,
      message = #wafer.dte_message<communication = 0, round = 0, slice = 0>}
      : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
}

// CHECK: DTE byte count must be positive
