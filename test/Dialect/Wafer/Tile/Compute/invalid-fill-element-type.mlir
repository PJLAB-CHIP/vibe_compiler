// RUN: wafer-opt %s -split-input-file -verify-diagnostics

module {
  %out = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<spm, tensor>>
  %cst = arith.constant 0.000000e+00 : f32
  // expected-error @below {{'wafer.tile.fill' op fill value type must match destination tensor element type}}
  wafer.tile.fill %out, %cst
      : memref<4x8xf16, #wafer.memory<spm, tensor>>, f32
}
