// RUN: wafer-opt --split-input-file --verify-diagnostics %s -o /dev/null

// A rank-preserving C reduction writes every retained scalar into C0=4
// storage. A rank-dropped declaration must not hide these physical strides.
func.func @rank_dropped(%input: memref<1x2x8x1024xf16, #wafer.memory<spm, ncx>>,
                        %dest: memref<1x2x8xf16, #wafer.memory<spm, ncx>>) {
  // expected-error @below {{reduce operands/dest must use rank4 NHWC geometry}}
  wafer.instr.reduce <sum> %input into %dest {dim = 0 : i64}
      : memref<1x2x8x1024xf16, #wafer.memory<spm, ncx>>
    into memref<1x2x8xf16, #wafer.memory<spm, ncx>>
  return
}

// -----

// Rank3 NCx has independent outer slices; left-padding its shape in the CRT
// call would lose their bank alignment. Keep this real-scale negative even
// when both buffers otherwise satisfy the logical H reduction shape.
func.func @implicit_nhwc(%input: memref<16x1025x1xf16, #wafer.memory<spm, ncx>>,
                         %dest: memref<1x1025x1xf16, #wafer.memory<spm, ncx>>) {
  // expected-error @below {{reduce operands/dest must use rank4 NHWC geometry}}
  wafer.instr.reduce <sum> %input into %dest {dim = 2 : i64}
      : memref<16x1025x1xf16, #wafer.memory<spm, ncx>>
    into memref<1x1025x1xf16, #wafer.memory<spm, ncx>>
  return
}

// -----

func.func @wrong_extent(%input: memref<1x2x8x1025xf16, #wafer.memory<spm, ncx>>,
                        %dest: memref<1x2x8x2xf16, #wafer.memory<spm, ncx>>) {
  // expected-error @below {{reduce dest must retain non-reduced dimensions and set reduced dimensions to one}}
  wafer.instr.reduce <sum> %input into %dest {dim = 0 : i64}
      : memref<1x2x8x1025xf16, #wafer.memory<spm, ncx>>
    into memref<1x2x8x2xf16, #wafer.memory<spm, ncx>>
  return
}

// -----

func.func @unsupported_n(%input: memref<2x3x8x1031xf16, #wafer.memory<spm, ncx>>,
                         %dest: memref<1x3x8x1031xf16, #wafer.memory<spm, ncx>>) {
  // expected-error @below {{reduce N/HWC axes have no supported target contract}}
  wafer.instr.reduce <sum> %input into %dest {dim = 3 : i64}
      : memref<2x3x8x1031xf16, #wafer.memory<spm, ncx>>
    into memref<1x3x8x1031xf16, #wafer.memory<spm, ncx>>
  return
}

// -----

func.func @unsupported_hwc(%input: memref<2x3x8x1031xf16, #wafer.memory<spm, ncx>>,
                           %dest: memref<2x1x1x1xf16, #wafer.memory<spm, ncx>>) {
  // expected-error @below {{reduce N/HWC axes have no supported target contract}}
  wafer.instr.reduce <sum> %input into %dest {dim = 5 : i64}
      : memref<2x3x8x1031xf16, #wafer.memory<spm, ncx>>
    into memref<2x1x1x1xf16, #wafer.memory<spm, ncx>>
  return
}
