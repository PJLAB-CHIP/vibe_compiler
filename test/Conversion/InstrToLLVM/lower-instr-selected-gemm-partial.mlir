// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %s | FileCheck %s

module {
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh {axes = ["card_partition"], shape = array<i64: 1>}
func.func @selected_partial_1024() {
    %a = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<2x2x1024xf16, #wafer.memory<spm, ncx>>
    %b = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<196608>} : memref<2x1024x4xf16, #wafer.memory<spm, ncx>>
    %c = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<458752>} : memref<2x2x4xf16, #wafer.memory<spm, ncx>>
    %p = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<524288>} : memref<2x2x4xf32, #wafer.memory<spm, ncx>>
    %other = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<589824>} : memref<2x2x4xf32, #wafer.memory<spm, ncx>>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %end = arith.constant 3 : index
    %result = scf.for %iv = %zero to %end step %one iter_args(%current = %p) -> (memref<2x2x4xf32, #wafer.memory<spm, ncx>>) {
      %condition = arith.cmpi eq, %iv, %zero : index
      %chosen = arith.select %condition, %current, %other : memref<2x2x4xf32, #wafer.memory<spm, ncx>>
    wafer.instr.gemm %a, %b psum(%chosen : memref<2x2x4xf32, #wafer.memory<spm, ncx>>) into %c {
      m = 2 : i64, k = 1024 : i64, n = 4 : i64, batch_count = 2 : i64,
      lhs_batch_dims = array<i64: 0>, rhs_batch_dims = array<i64: 0>, result_batch_dims = array<i64: 0>,
      lhs_m_dim = 1 : i64, lhs_contracting_dim = 2 : i64,
      rhs_contracting_dim = 1 : i64, rhs_n_dim = 2 : i64,
      result_m_dim = 1 : i64, result_n_dim = 2 : i64
    } : memref<2x2x1024xf16, #wafer.memory<spm, ncx>>, memref<2x1024x4xf16, #wafer.memory<spm, ncx>> into memref<2x2x4xf16, #wafer.memory<spm, ncx>>
      %next = scf.if %condition -> (memref<2x2x4xf32, #wafer.memory<spm, ncx>>) {
        scf.yield %other : memref<2x2x4xf32, #wafer.memory<spm, ncx>>
      } else {
        scf.yield %current : memref<2x2x4xf32, #wafer.memory<spm, ncx>>
      }
      scf.yield %next : memref<2x2x4xf32, #wafer.memory<spm, ncx>>
    }
    wafer.instr.gemm %a, %b psum(%result : memref<2x2x4xf32, #wafer.memory<spm, ncx>>) into %c {
      m = 2 : i64, k = 1024 : i64, n = 4 : i64, batch_count = 2 : i64,
      lhs_batch_dims = array<i64: 0>, rhs_batch_dims = array<i64: 0>, result_batch_dims = array<i64: 0>,
      lhs_m_dim = 1 : i64, lhs_contracting_dim = 2 : i64,
      rhs_contracting_dim = 1 : i64, rhs_n_dim = 2 : i64,
      result_m_dim = 1 : i64, result_n_dim = 2 : i64
    } : memref<2x2x1024xf16, #wafer.memory<spm, ncx>>, memref<2x1024x4xf16, #wafer.memory<spm, ncx>> into memref<2x2x4xf16, #wafer.memory<spm, ncx>>
    return
  }

func.func @selected_partial_1025() {
    %a = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<2x2x1025xbf16, #wafer.memory<spm, ncx>>
    %b = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<196608>} : memref<2x1025x4xbf16, #wafer.memory<spm, ncx>>
    %c = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<458752>} : memref<2x2x4xbf16, #wafer.memory<spm, ncx>>
    %p = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<524288>} : memref<2x2x4xf32, #wafer.memory<spm, ncx>>
    %other = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<589824>} : memref<2x2x4xf32, #wafer.memory<spm, ncx>>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %end = arith.constant 3 : index
    %result = scf.for %iv = %zero to %end step %one iter_args(%current = %p) -> (memref<2x2x4xf32, #wafer.memory<spm, ncx>>) {
      %condition = arith.cmpi eq, %iv, %zero : index
      %chosen = arith.select %condition, %current, %other : memref<2x2x4xf32, #wafer.memory<spm, ncx>>
    wafer.instr.gemm %a, %b psum(%chosen : memref<2x2x4xf32, #wafer.memory<spm, ncx>>) into %c {
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
    wafer.instr.gemm %a, %b psum(%result : memref<2x2x4xf32, #wafer.memory<spm, ncx>>) into %c {
      m = 2 : i64, k = 1025 : i64, n = 4 : i64, batch_count = 2 : i64,
      lhs_batch_dims = array<i64: 0>, rhs_batch_dims = array<i64: 0>, result_batch_dims = array<i64: 0>,
      lhs_m_dim = 1 : i64, lhs_contracting_dim = 2 : i64,
      rhs_contracting_dim = 1 : i64, rhs_n_dim = 2 : i64,
      result_m_dim = 1 : i64, result_n_dim = 2 : i64
    } : memref<2x2x1025xbf16, #wafer.memory<spm, ncx>>, memref<2x1025x4xbf16, #wafer.memory<spm, ncx>> into memref<2x2x4xbf16, #wafer.memory<spm, ncx>>
    return
  }

func.func @selected_partial_1031() {
    %a = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<2x2x1031xf16, #wafer.memory<spm, ncx>>
    %b = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<196608>} : memref<2x1031x4xf16, #wafer.memory<spm, ncx>>
    %c = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<458752>} : memref<2x2x4xf16, #wafer.memory<spm, ncx>>
    %p = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<524288>} : memref<2x2x4xf32, #wafer.memory<spm, ncx>>
    %other = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<589824>} : memref<2x2x4xf32, #wafer.memory<spm, ncx>>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %end = arith.constant 3 : index
    %result = scf.for %iv = %zero to %end step %one iter_args(%current = %p) -> (memref<2x2x4xf32, #wafer.memory<spm, ncx>>) {
      %condition = arith.cmpi eq, %iv, %zero : index
      %chosen = arith.select %condition, %current, %other : memref<2x2x4xf32, #wafer.memory<spm, ncx>>
    wafer.instr.gemm %a, %b psum(%chosen : memref<2x2x4xf32, #wafer.memory<spm, ncx>>) into %c {
      m = 2 : i64, k = 1031 : i64, n = 4 : i64, batch_count = 2 : i64,
      lhs_batch_dims = array<i64: 0>, rhs_batch_dims = array<i64: 0>, result_batch_dims = array<i64: 0>,
      lhs_m_dim = 1 : i64, lhs_contracting_dim = 2 : i64,
      rhs_contracting_dim = 1 : i64, rhs_n_dim = 2 : i64,
      result_m_dim = 1 : i64, result_n_dim = 2 : i64
    } : memref<2x2x1031xf16, #wafer.memory<spm, ncx>>, memref<2x1031x4xf16, #wafer.memory<spm, ncx>> into memref<2x2x4xf16, #wafer.memory<spm, ncx>>
      %next = scf.if %condition -> (memref<2x2x4xf32, #wafer.memory<spm, ncx>>) {
        scf.yield %other : memref<2x2x4xf32, #wafer.memory<spm, ncx>>
      } else {
        scf.yield %current : memref<2x2x4xf32, #wafer.memory<spm, ncx>>
      }
      scf.yield %next : memref<2x2x4xf32, #wafer.memory<spm, ncx>>
    }
    wafer.instr.gemm %a, %b psum(%result : memref<2x2x4xf32, #wafer.memory<spm, ncx>>) into %c {
      m = 2 : i64, k = 1031 : i64, n = 4 : i64, batch_count = 2 : i64,
      lhs_batch_dims = array<i64: 0>, rhs_batch_dims = array<i64: 0>, result_batch_dims = array<i64: 0>,
      lhs_m_dim = 1 : i64, lhs_contracting_dim = 2 : i64,
      rhs_contracting_dim = 1 : i64, rhs_n_dim = 2 : i64,
      result_m_dim = 1 : i64, result_n_dim = 2 : i64
    } : memref<2x2x1031xf16, #wafer.memory<spm, ncx>>, memref<2x1031x4xf16, #wafer.memory<spm, ncx>> into memref<2x2x4xf16, #wafer.memory<spm, ncx>>
    return
  }
}

// CHECK-LABEL: llvm.func @selected_partial_1024
// CHECK: %[[DEST1024:.*]] = llvm.mlir.constant(458752 : i64) : i64
// CHECK: %[[INIT1024:.*]] = llvm.mlir.constant(524288 : i64) : i64
// CHECK: %[[OTHER1024:.*]] = llvm.mlir.constant(589824 : i64) : i64
// CHECK: llvm.br ^bb{{[0-9]+}}({{.*}}, %[[INIT1024]] : i64, i64)
// CHECK: ^bb{{[0-9]+}}(%{{.*}}: i64, %[[ITER1024:.*]]: i64):
// CHECK: %[[CHOSEN1024:.*]] = llvm.select %{{.*}}, %[[ITER1024]], %[[OTHER1024]] : i1, i64
// CHECK: llvm.call @wafer_tx81_gemm(%{{.*}}, %{{.*}}, %[[DEST1024]], %[[CHOSEN1024]],
// CHECK: llvm.call @wafer_tx81_gemm(%{{.*}}, %{{.*}}, %[[DEST1024]], %[[ITER1024]],

// CHECK-LABEL: llvm.func @selected_partial_1025
// CHECK: %[[DEST1025:.*]] = llvm.mlir.constant(458752 : i64) : i64
// CHECK: %[[INIT1025:.*]] = llvm.mlir.constant(524288 : i64) : i64
// CHECK: %[[OTHER1025:.*]] = llvm.mlir.constant(589824 : i64) : i64
// CHECK: llvm.br ^bb{{[0-9]+}}({{.*}}, %[[INIT1025]] : i64, i64)
// CHECK: ^bb{{[0-9]+}}(%{{.*}}: i64, %[[ITER1025:.*]]: i64):
// CHECK: %[[CHOSEN1025:.*]] = llvm.select %{{.*}}, %[[ITER1025]], %[[OTHER1025]] : i1, i64
// CHECK: llvm.call @wafer_tx81_gemm(%{{.*}}, %{{.*}}, %[[DEST1025]], %[[CHOSEN1025]],
// CHECK: llvm.call @wafer_tx81_gemm(%{{.*}}, %{{.*}}, %[[DEST1025]], %[[ITER1025]],

// CHECK-LABEL: llvm.func @selected_partial_1031
// CHECK: %[[DEST1031:.*]] = llvm.mlir.constant(458752 : i64) : i64
// CHECK: %[[INIT1031:.*]] = llvm.mlir.constant(524288 : i64) : i64
// CHECK: %[[OTHER1031:.*]] = llvm.mlir.constant(589824 : i64) : i64
// CHECK: llvm.br ^bb{{[0-9]+}}({{.*}}, %[[INIT1031]] : i64, i64)
// CHECK: ^bb{{[0-9]+}}(%{{.*}}: i64, %[[ITER1031:.*]]: i64):
// CHECK: %[[CHOSEN1031:.*]] = llvm.select %{{.*}}, %[[ITER1031]], %[[OTHER1031]] : i1, i64
// CHECK: llvm.call @wafer_tx81_gemm(%{{.*}}, %{{.*}}, %[[DEST1031]], %[[CHOSEN1031]],
// CHECK: llvm.call @wafer_tx81_gemm(%{{.*}}, %{{.*}}, %[[DEST1031]], %[[ITER1031]],
