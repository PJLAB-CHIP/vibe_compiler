// RUN: not wafer-opt %s 2>&1 | FileCheck %s

#identity = affine_map<(d0, d1) -> (d0, d1)>

module {
  %lhs = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<spm, tensor>>
  %rhs = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<spm, tensor>>
  %dst = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<spm, tensor>>
  wafer.instr.elementwise #wafer.instr_elementwise_kind<add> %lhs, %rhs into %dst
      {indexing_maps = [#identity, #identity, #identity]}
      : memref<4x8xf16, #wafer.memory<spm, tensor>>,
        memref<4x8xf16, #wafer.memory<spm, tensor>>
    into memref<4x8xf16, #wafer.memory<spm, tensor>>
}

// CHECK: error: 'wafer.instr.elementwise' op does not accept schema-free semantic attribute 'indexing_maps'
