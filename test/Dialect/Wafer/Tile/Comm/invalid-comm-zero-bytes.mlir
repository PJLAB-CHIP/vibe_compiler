// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %buf = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf32, #wafer.memory<spm, tensor>>
  %recv = wafer.tile.recv %buf {peer = 0 : i64, bytes = 0 : i64}
      : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
}

// CHECK: comm byte count must be positive
