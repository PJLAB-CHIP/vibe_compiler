// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %buffer = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf32, #wafer.memory<spm, tensor>>
  %token = wafer.instr.dte_send %buffer
      {peer = 1 : i64, bytes = 16 : i64,
       message = #wafer.dte_message<communication = 0, phase = peer_dataflow,
                                    round = 0, slice = 0>}
      : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
}

// CHECK: expected 'round'
