// RUN: wafer-opt %s -verify-diagnostics

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<spm, cx>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<4xf16, #wafer.memory<spm, cx>>
  // expected-error @+1 {{does not accept schema-free semantic attribute 'init_value'}}
  wafer.instr.reduce #wafer.instr_reduce_kind<sum> %input into %dst
      {dim = 0 : i64, init_value = 0.000000e+00 : f16}
      : memref<4x8xf16, #wafer.memory<spm, cx>>
    into memref<4xf16, #wafer.memory<spm, cx>>
}
