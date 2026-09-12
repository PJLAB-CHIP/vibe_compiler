// RUN: wafer-opt --wafer-lower-instr-to-target-llvm --verify-diagnostics %s

module {
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh {axes = ["card_partition"], shape = array<i64: 1>}
func.func @overlapping_backedge() {
    %a = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<2x2x1025xbf16, #wafer.memory<spm, ncx>>
    %b = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<196608>} : memref<2x1025x4xbf16, #wafer.memory<spm, ncx>>
    %c = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<458752>} : memref<2x2x4xbf16, #wafer.memory<spm, ncx>>
    %p = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<524288>} : memref<2x2x4xf32, #wafer.memory<spm, ncx>>
    %other = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<458752>} : memref<2x2x4xf32, #wafer.memory<spm, ncx>>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %end = arith.constant 3 : index
    %result = scf.for %iv = %zero to %end step %one iter_args(%current = %p) -> (memref<2x2x4xf32, #wafer.memory<spm, ncx>>) {
      %condition = arith.cmpi eq, %iv, %zero : index
      %chosen = arith.select %condition, %current, %other : memref<2x2x4xf32, #wafer.memory<spm, ncx>>
    // expected-error @+1 {{unsupported_target_alias: GEMM psum and destination must have disjoint physical storage}}
    wafer.instr.gemm %a, %b psum(%current : memref<2x2x4xf32, #wafer.memory<spm, ncx>>) into %c {
      m = 2 : i64, k = 1025 : i64, n = 4 : i64, batch_count = 2 : i64,
      lhs_batch_dims = array<i64: 0>, rhs_batch_dims = array<i64: 0>, result_batch_dims = array<i64: 0>,
      lhs_m_dim = 1 : i64, lhs_contracting_dim = 2 : i64,
      rhs_contracting_dim = 1 : i64, rhs_n_dim = 2 : i64,
      result_m_dim = 1 : i64, result_n_dim = 2 : i64
    } : memref<2x2x1025xbf16, #wafer.memory<spm, ncx>>, memref<2x1025x4xbf16, #wafer.memory<spm, ncx>> into memref<2x2x4xbf16, #wafer.memory<spm, ncx>>
      %next = scf.if %condition -> (memref<2x2x4xf32, #wafer.memory<spm, ncx>>) {
        scf.yield %other : memref<2x2x4xf32, #wafer.memory<spm, ncx>>
      } else {
        scf.yield %current : memref<2x2x4xf32, #wafer.memory<spm, ncx>>
      }
      scf.yield %next : memref<2x2x4xf32, #wafer.memory<spm, ncx>>
    }
    return
  }
}
