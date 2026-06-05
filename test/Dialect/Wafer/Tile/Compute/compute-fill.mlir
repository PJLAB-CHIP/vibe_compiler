// RUN: wafer-opt %s | FileCheck %s

module {
  %out = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<spm, tensor>>
  %cst = arith.constant 0.000000e+00 : f16
  wafer.tile.fill %out, %cst
      : memref<4x8xf16, #wafer.memory<spm, tensor>>, f16
}

// CHECK: wafer.tile.fill
