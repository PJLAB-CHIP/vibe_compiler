// RUN: wafer-opt %s -split-input-file -verify-diagnostics

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<spm, cx>>
  %init = arith.constant 0.000000e+00 : f16
  // expected-error @+1 {{requires exactly one of scalar init operand or init_value attr}}
  %sum = wafer.tile.reduce #wafer.reduce_kind<sum> %input, %init
      {dimensions = array<i64: 1>, init_value = 0.000000e+00 : f16}
      : (memref<4x8xf16, #wafer.memory<spm, cx>>, f16)
     -> memref<4xf16, #wafer.memory<spm, cx>>
}

// -----

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<spm, cx>>
  // expected-error @+1 {{requires exactly one of scalar init operand or init_value attr}}
  %sum = wafer.tile.reduce #wafer.reduce_kind<sum> %input
      {dimensions = array<i64: 1>}
      : (memref<4x8xf16, #wafer.memory<spm, cx>>)
     -> memref<4xf16, #wafer.memory<spm, cx>>
}

// -----

func.func @dynamic_init(
    %init: f16,
    %input: memref<4x8xf16, #wafer.memory<spm, cx>>)
    -> memref<4xf16, #wafer.memory<spm, cx>> {
  // expected-error @+1 {{reduce init operand must be defined by arith.constant}}
  %sum = wafer.tile.reduce #wafer.reduce_kind<sum> %input, %init
      {dimensions = array<i64: 1>}
      : (memref<4x8xf16, #wafer.memory<spm, cx>>, f16)
     -> memref<4xf16, #wafer.memory<spm, cx>>
  return %sum : memref<4xf16, #wafer.memory<spm, cx>>
}

// -----

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<?x8xf16, #wafer.memory<spm, cx>>
  // expected-error @+1 {{reduce input/result shapes must be static and positive}}
  %sum = wafer.tile.reduce #wafer.reduce_kind<sum> %input
      {dimensions = array<i64: 0>, init_value = 0.000000e+00 : f16}
      : (memref<?x8xf16, #wafer.memory<spm, cx>>)
     -> memref<8xf16, #wafer.memory<spm, cx>>
}
