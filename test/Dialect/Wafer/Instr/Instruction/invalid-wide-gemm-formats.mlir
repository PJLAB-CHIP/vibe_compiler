// RUN: wafer-opt --split-input-file --verify-diagnostics %s

func.func @invalid_output(%a: memref<2x16x1025xf16, #wafer.memory<spm, ncx>>, %b: memref<2x1025x32xf16, #wafer.memory<spm, ncx>>, %c: memref<2x16x32xbf16, #wafer.memory<spm, ncx>>) {
  // expected-error @+1 {{gemm requires equal input types and the same output type or f16/bf16 inputs with f32 output}}
  wafer.instr.gemm %a, %b into %c {m = 16 : i64, k = 1025 : i64, n = 32 : i64} : memref<2x16x1025xf16, #wafer.memory<spm, ncx>>, memref<2x1025x32xf16, #wafer.memory<spm, ncx>> into memref<2x16x32xbf16, #wafer.memory<spm, ncx>>
  return
}

// -----

func.func @mismatched_inputs(%a: memref<2x16x1031xf16, #wafer.memory<spm, ncx>>, %b: memref<2x1031x32xbf16, #wafer.memory<spm, ncx>>, %c: memref<2x16x32xf32, #wafer.memory<spm, ncx>>) {
  // expected-error @+1 {{gemm lhs and rhs element types must match}}
  wafer.instr.gemm %a, %b into %c {m = 16 : i64, k = 1031 : i64, n = 32 : i64} : memref<2x16x1031xf16, #wafer.memory<spm, ncx>>, memref<2x1031x32xbf16, #wafer.memory<spm, ncx>> into memref<2x16x32xf32, #wafer.memory<spm, ncx>>
  return
}
