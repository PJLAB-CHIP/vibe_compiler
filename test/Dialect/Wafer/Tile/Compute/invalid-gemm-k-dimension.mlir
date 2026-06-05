// RUN: not wafer-opt %s 2>&1 | FileCheck %s

module {
  %a = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x7xf16, #wafer.memory<spm, cx>>
  %b = "builtin.unrealized_conversion_cast"()
      : () -> memref<8x16xf16, #wafer.memory<spm, cx>>
  %0 = wafer.tile.gemm %a, %b
      : (memref<4x7xf16, #wafer.memory<spm, cx>>,
         memref<8x16xf16, #wafer.memory<spm, cx>>)
     -> memref<4x16xf16, #wafer.memory<spm, cx>>
}

// CHECK: gemm lhs K dimension must match rhs K dimension
