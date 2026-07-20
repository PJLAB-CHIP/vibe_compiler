// RUN: wafer-opt -split-input-file -verify-diagnostics %s

module {
  %lhs = "builtin.unrealized_conversion_cast"()
      : () -> memref<2x3xf16, #wafer.memory<spm, cx>>
  %rhs = "builtin.unrealized_conversion_cast"()
      : () -> memref<3x4xf16, #wafer.memory<spm, cx>>
  // expected-error @below {{lhs_orientation and rhs_orientation must either both be present}}
  %result = wafer.tile.gemm %lhs, %rhs
      {lhs_orientation = #wafer.gemm_orientation<normal>}
      : (memref<2x3xf16, #wafer.memory<spm, cx>>,
         memref<3x4xf16, #wafer.memory<spm, cx>>)
     -> memref<2x4xf16, #wafer.memory<spm, cx>>
}

// -----

module {
  %lhs = "builtin.unrealized_conversion_cast"()
      : () -> memref<2x3xf16, #wafer.memory<spm, cx>>
  %rhs = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x2xf16, #wafer.memory<spm, cx>>
  // expected-error @below {{gemm result shape must be lhs M by rhs N}}
  %result = wafer.tile.gemm %lhs, %rhs
      {lhs_orientation = #wafer.gemm_orientation<transpose>,
       rhs_orientation = #wafer.gemm_orientation<transpose>}
      : (memref<2x3xf16, #wafer.memory<spm, cx>>,
         memref<4x2xf16, #wafer.memory<spm, cx>>)
     -> memref<2x4xf16, #wafer.memory<spm, cx>>
}
