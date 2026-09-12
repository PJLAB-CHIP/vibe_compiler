// RUN: wafer-opt %s -split-input-file -verify-diagnostics

module {
  %lhs = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<spm, tensor>>
  %rhs = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<spm, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<spm, tensor>>
  // expected-error @+1 {{attribute 'kind' failed to satisfy constraint: Wafer instruction CT elementwise target kind}}
  "wafer.instr.elementwise"(%lhs, %rhs, %dst)
      {kind = #wafer.elementwise_kind<add>}
      : (memref<4x8xf16, #wafer.memory<spm, tensor>>,
         memref<4x8xf16, #wafer.memory<spm, tensor>>,
         memref<4x8xf16, #wafer.memory<spm, tensor>>) -> ()
}

// -----

module {
  %input = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1x4x8xf16, #wafer.memory<spm, ncx>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<1x1x4x1xf16, #wafer.memory<spm, ncx>>
  // expected-error @+1 {{attribute 'kind' failed to satisfy constraint: Wafer instruction CT reduce target kind}}
  "wafer.instr.reduce"(%input, %dst)
      {kind = #wafer.reduce_kind<sum>, dim = 0 : i64}
      : (memref<1x1x4x8xf16, #wafer.memory<spm, ncx>>,
         memref<1x1x4x1xf16, #wafer.memory<spm, ncx>>) -> ()
}

// -----

module {
  %src = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf32, #wafer.memory<spm, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xi32, #wafer.memory<spm, tensor>>
  // expected-error @+1 {{requires attribute 'kind'}}
  "wafer.instr.convert"(%src, %dst)
      {src_dtype = f32, dst_dtype = i32}
      : (memref<4x8xf32, #wafer.memory<spm, tensor>>,
         memref<4x8xi32, #wafer.memory<spm, tensor>>) -> ()
}
