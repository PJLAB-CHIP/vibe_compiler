// RUN: not wafer-opt --split-input-file %s 2>&1 | FileCheck %s

module {
  %buffer = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1025x128xf16, #wafer.memory<spm, tensor>>
  %bad = wafer.instr.dte_broadcast %buffer
      {source_offset = 0 : i64, peers = array<i64: 1, 2, 3>, bytes = 256 : i64,
       messages = [#wafer.dte_message<communication = 1, round = 0, slice = 0>,
                   #wafer.dte_message<communication = 1, round = 0, slice = 1>,
                   #wafer.dte_message<communication = 1, round = 0, slice = 2>]}
      : memref<1x1025x128xf16, #wafer.memory<spm, tensor>> -> !async.token
}

// CHECK: native DTE multi-send destination count must be 2, 4, 8, or 15

// -----

module {
  %buffer = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1025x128xf16, #wafer.memory<spm, tensor>>
  %bad = wafer.instr.dte_scatter %buffer
      {source_offset = 0 : i64, peers = array<i64: 1, 2>, bytes = 257 : i64,
       messages = [#wafer.dte_message<communication = 2, round = 0, slice = 0>,
                   #wafer.dte_message<communication = 2, round = 0, slice = 1>]}
      : memref<1x1025x128xf16, #wafer.memory<spm, tensor>> -> !async.token
}

// CHECK: native DTE multi-send requires exactly 256 bytes per destination

// -----

module {
  %buffer = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1025x128xf16, #wafer.memory<spm, tensor>>
  %bad = wafer.instr.dte_broadcast %buffer
      {source_offset = 0 : i64, peers = array<i64: 1, 1>, bytes = 256 : i64,
       messages = [#wafer.dte_message<communication = 3, round = 0, slice = 0>,
                   #wafer.dte_message<communication = 3, round = 0, slice = 1>]}
      : memref<1x1025x128xf16, #wafer.memory<spm, tensor>> -> !async.token
}

// CHECK: native DTE multi-send peers must be unique non-negative uint32_t values
