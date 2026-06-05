// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %buf = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf32, #wafer.memory<spm, tensor>>
  %send = wafer.tile.send %buf {peer = -1 : i64, bytes = 16 : i64}
      : memref<4xf32, #wafer.memory<spm, tensor>> -> !async.token
}

// CHECK: comm peer must be non-negative
