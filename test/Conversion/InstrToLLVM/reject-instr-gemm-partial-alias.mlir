// RUN: wafer-opt --wafer-lower-instr-to-target-llvm --verify-diagnostics %s

module {
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh {axes = ["card_partition"], shape = array<i64: 1>}
  func.func @partial_f16_nn() {
    %a = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<2x2x1025xf16, #wafer.memory<spm, ncx>>
    %b = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<196608>} : memref<2x1025x4xf16, #wafer.memory<spm, ncx>>
    %c = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<458752>} : memref<2x2x4xf16, #wafer.memory<spm, ncx>>
    %p = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<458752>} : memref<2x2x4xf32, #wafer.memory<spm, ncx>>
    // expected-error @+1 {{unsupported_target_alias: GEMM psum and destination must have disjoint physical storage}}
    wafer.instr.gemm %a, %b psum(%p : memref<2x2x4xf32, #wafer.memory<spm, ncx>>) into %c {
      m = 2 : i64, k = 1025 : i64, n = 4 : i64, batch_count = 2 : i64,
      lhs_batch_dims = array<i64: 0>, rhs_batch_dims = array<i64: 0>, result_batch_dims = array<i64: 0>,
      lhs_m_dim = 1 : i64, lhs_contracting_dim = 2 : i64,
      rhs_contracting_dim = 1 : i64, rhs_n_dim = 2 : i64,
      result_m_dim = 1 : i64, result_n_dim = 2 : i64
    } : memref<2x2x1025xf16, #wafer.memory<spm, ncx>>, memref<2x1025x4xf16, #wafer.memory<spm, ncx>> into memref<2x2x4xf16, #wafer.memory<spm, ncx>>
    return
  }
}
