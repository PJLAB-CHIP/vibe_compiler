// RUN: wafer-opt %s | FileCheck %s

module {
  %buffer = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1025x128xf16, #wafer.memory<spm, tensor>>
  %broadcast = wafer.instr.dte_broadcast %buffer
      {source_offset = 0 : i64, peers = array<i64: 1, 2>, bytes = 256 : i64,
       messages = [#wafer.dte_message<communication = 10, round = 0, slice = 0>,
                   #wafer.dte_message<communication = 10, round = 0, slice = 1>]}
      : memref<1x1025x128xf16, #wafer.memory<spm, tensor>> -> !async.token
  %scatter = wafer.instr.dte_scatter %buffer
      {source_offset = 256 : i64, peers = array<i64: 1, 8, 15, 7>,
       bytes = 256 : i64,
       messages = [#wafer.dte_message<communication = 11, round = 1, slice = 0>,
                   #wafer.dte_message<communication = 11, round = 1, slice = 1>,
                   #wafer.dte_message<communication = 11, round = 1, slice = 2>,
                   #wafer.dte_message<communication = 11, round = 1, slice = 3>]}
      : memref<1x1025x128xf16, #wafer.memory<spm, tensor>> -> !async.token
}

// CHECK: wafer.instr.dte_broadcast
// CHECK-SAME: bytes = 256 : i64
// CHECK-SAME: peers = array<i64: 1, 2>
// CHECK-SAME: source_offset = 0 : i64
// CHECK: wafer.instr.dte_scatter
// CHECK-SAME: peers = array<i64: 1, 8, 15, 7>
// CHECK-SAME: source_offset = 256 : i64
