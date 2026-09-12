// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %s | FileCheck %s

module {
  wafer.target.topology @default {card_grid = array<i64: 1, 1>, card_interconnect = "mesh", tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh {axes = ["card_partition"], shape = array<i64: 1>}
  func.func @wide_f16_nn() {
    %a = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<2x2x1025xf16, #wafer.memory<spm, ncx>>
    %b = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<196608>} : memref<2x1025x4xf16, #wafer.memory<spm, ncx>>
    %c = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<458752>} : memref<2x2x4xf32, #wafer.memory<spm, ncx>>
    wafer.instr.gemm %a, %b into %c {
      m = 2 : i64, k = 1025 : i64, n = 4 : i64, batch_count = 2 : i64,
      lhs_batch_dims = array<i64: 0>, rhs_batch_dims = array<i64: 0>, result_batch_dims = array<i64: 0>,
      lhs_m_dim = 1 : i64, lhs_contracting_dim = 2 : i64,
      rhs_contracting_dim = 1 : i64, rhs_n_dim = 2 : i64,
      result_m_dim = 1 : i64, result_n_dim = 2 : i64
    } : memref<2x2x1025xf16, #wafer.memory<spm, ncx>>, memref<2x1025x4xf16, #wafer.memory<spm, ncx>> into memref<2x2x4xf32, #wafer.memory<spm, ncx>>
    return
  }

  func.func @wide_f16_nt() {
    %a = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<2x2x1025xf16, #wafer.memory<spm, ncx>>
    %b = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<196608>} : memref<2x4x1025xf16, #wafer.memory<spm, ncx>>
    %c = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<458752>} : memref<2x2x4xf32, #wafer.memory<spm, ncx>>
    wafer.instr.gemm %a, %b into %c {
      m = 2 : i64, k = 1025 : i64, n = 4 : i64, batch_count = 2 : i64,
      lhs_batch_dims = array<i64: 0>, rhs_batch_dims = array<i64: 0>, result_batch_dims = array<i64: 0>,
      lhs_m_dim = 1 : i64, lhs_contracting_dim = 2 : i64,
      rhs_contracting_dim = 2 : i64, rhs_n_dim = 1 : i64,
      result_m_dim = 1 : i64, result_n_dim = 2 : i64,
      lhs_orientation = #wafer.gemm_orientation<normal>,
      rhs_orientation = #wafer.gemm_orientation<transpose>
    } : memref<2x2x1025xf16, #wafer.memory<spm, ncx>>, memref<2x4x1025xf16, #wafer.memory<spm, ncx>> into memref<2x2x4xf32, #wafer.memory<spm, ncx>>
    return
  }

  func.func @wide_f16_tn() {
    %a = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<2x1025x2xf16, #wafer.memory<spm, ncx>>
    %b = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<196608>} : memref<2x1025x4xf16, #wafer.memory<spm, ncx>>
    %c = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<458752>} : memref<2x2x4xf32, #wafer.memory<spm, ncx>>
    wafer.instr.gemm %a, %b into %c {
      m = 2 : i64, k = 1025 : i64, n = 4 : i64, batch_count = 2 : i64,
      lhs_batch_dims = array<i64: 0>, rhs_batch_dims = array<i64: 0>, result_batch_dims = array<i64: 0>,
      lhs_m_dim = 2 : i64, lhs_contracting_dim = 1 : i64,
      rhs_contracting_dim = 1 : i64, rhs_n_dim = 2 : i64,
      result_m_dim = 1 : i64, result_n_dim = 2 : i64,
      lhs_orientation = #wafer.gemm_orientation<transpose>,
      rhs_orientation = #wafer.gemm_orientation<normal>
    } : memref<2x1025x2xf16, #wafer.memory<spm, ncx>>, memref<2x1025x4xf16, #wafer.memory<spm, ncx>> into memref<2x2x4xf32, #wafer.memory<spm, ncx>>
    return
  }

  func.func @wide_f16_tt() {
    %a = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<2x1025x2xf16, #wafer.memory<spm, ncx>>
    %b = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<196608>} : memref<2x4x1025xf16, #wafer.memory<spm, ncx>>
    %c = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<458752>} : memref<2x2x4xf32, #wafer.memory<spm, ncx>>
    wafer.instr.gemm %a, %b into %c {
      m = 2 : i64, k = 1025 : i64, n = 4 : i64, batch_count = 2 : i64,
      lhs_batch_dims = array<i64: 0>, rhs_batch_dims = array<i64: 0>, result_batch_dims = array<i64: 0>,
      lhs_m_dim = 2 : i64, lhs_contracting_dim = 1 : i64,
      rhs_contracting_dim = 2 : i64, rhs_n_dim = 1 : i64,
      result_m_dim = 1 : i64, result_n_dim = 2 : i64,
      lhs_orientation = #wafer.gemm_orientation<transpose>,
      rhs_orientation = #wafer.gemm_orientation<transpose>
    } : memref<2x1025x2xf16, #wafer.memory<spm, ncx>>, memref<2x4x1025xf16, #wafer.memory<spm, ncx>> into memref<2x2x4xf32, #wafer.memory<spm, ncx>>
    return
  }

  func.func @wide_bf16_nn() {
    %a = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<2x2x1025xbf16, #wafer.memory<spm, ncx>>
    %b = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<196608>} : memref<2x1025x4xbf16, #wafer.memory<spm, ncx>>
    %c = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<458752>} : memref<2x2x4xf32, #wafer.memory<spm, ncx>>
    wafer.instr.gemm %a, %b into %c {
      m = 2 : i64, k = 1025 : i64, n = 4 : i64, batch_count = 2 : i64,
      lhs_batch_dims = array<i64: 0>, rhs_batch_dims = array<i64: 0>, result_batch_dims = array<i64: 0>,
      lhs_m_dim = 1 : i64, lhs_contracting_dim = 2 : i64,
      rhs_contracting_dim = 1 : i64, rhs_n_dim = 2 : i64,
      result_m_dim = 1 : i64, result_n_dim = 2 : i64
    } : memref<2x2x1025xbf16, #wafer.memory<spm, ncx>>, memref<2x1025x4xbf16, #wafer.memory<spm, ncx>> into memref<2x2x4xf32, #wafer.memory<spm, ncx>>
    return
  }

  func.func @wide_bf16_nt() {
    %a = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<2x2x1025xbf16, #wafer.memory<spm, ncx>>
    %b = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<196608>} : memref<2x4x1025xbf16, #wafer.memory<spm, ncx>>
    %c = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<458752>} : memref<2x2x4xf32, #wafer.memory<spm, ncx>>
    wafer.instr.gemm %a, %b into %c {
      m = 2 : i64, k = 1025 : i64, n = 4 : i64, batch_count = 2 : i64,
      lhs_batch_dims = array<i64: 0>, rhs_batch_dims = array<i64: 0>, result_batch_dims = array<i64: 0>,
      lhs_m_dim = 1 : i64, lhs_contracting_dim = 2 : i64,
      rhs_contracting_dim = 2 : i64, rhs_n_dim = 1 : i64,
      result_m_dim = 1 : i64, result_n_dim = 2 : i64,
      lhs_orientation = #wafer.gemm_orientation<normal>,
      rhs_orientation = #wafer.gemm_orientation<transpose>
    } : memref<2x2x1025xbf16, #wafer.memory<spm, ncx>>, memref<2x4x1025xbf16, #wafer.memory<spm, ncx>> into memref<2x2x4xf32, #wafer.memory<spm, ncx>>
    return
  }

  func.func @wide_bf16_tn() {
    %a = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<2x1025x2xbf16, #wafer.memory<spm, ncx>>
    %b = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<196608>} : memref<2x1025x4xbf16, #wafer.memory<spm, ncx>>
    %c = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<458752>} : memref<2x2x4xf32, #wafer.memory<spm, ncx>>
    wafer.instr.gemm %a, %b into %c {
      m = 2 : i64, k = 1025 : i64, n = 4 : i64, batch_count = 2 : i64,
      lhs_batch_dims = array<i64: 0>, rhs_batch_dims = array<i64: 0>, result_batch_dims = array<i64: 0>,
      lhs_m_dim = 2 : i64, lhs_contracting_dim = 1 : i64,
      rhs_contracting_dim = 1 : i64, rhs_n_dim = 2 : i64,
      result_m_dim = 1 : i64, result_n_dim = 2 : i64,
      lhs_orientation = #wafer.gemm_orientation<transpose>,
      rhs_orientation = #wafer.gemm_orientation<normal>
    } : memref<2x1025x2xbf16, #wafer.memory<spm, ncx>>, memref<2x1025x4xbf16, #wafer.memory<spm, ncx>> into memref<2x2x4xf32, #wafer.memory<spm, ncx>>
    return
  }

  func.func @wide_bf16_tt() {
    %a = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>} : memref<2x1025x2xbf16, #wafer.memory<spm, ncx>>
    %b = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<196608>} : memref<2x4x1025xbf16, #wafer.memory<spm, ncx>>
    %c = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<458752>} : memref<2x2x4xf32, #wafer.memory<spm, ncx>>
    wafer.instr.gemm %a, %b into %c {
      m = 2 : i64, k = 1025 : i64, n = 4 : i64, batch_count = 2 : i64,
      lhs_batch_dims = array<i64: 0>, rhs_batch_dims = array<i64: 0>, result_batch_dims = array<i64: 0>,
      lhs_m_dim = 2 : i64, lhs_contracting_dim = 1 : i64,
      rhs_contracting_dim = 2 : i64, rhs_n_dim = 1 : i64,
      result_m_dim = 1 : i64, result_n_dim = 2 : i64,
      lhs_orientation = #wafer.gemm_orientation<transpose>,
      rhs_orientation = #wafer.gemm_orientation<transpose>
    } : memref<2x1025x2xbf16, #wafer.memory<spm, ncx>>, memref<2x4x1025xbf16, #wafer.memory<spm, ncx>> into memref<2x2x4xf32, #wafer.memory<spm, ncx>>
    return
  }

}

// CHECK-LABEL: llvm.func @wide_f16_nn
// CHECK: llvm.mlir.constant(4 : i32) : i32
// CHECK-NEXT: %{{.*}} = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT: %[[INPUT:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT: %[[OUTPUT:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK: llvm.call @wafer_tx81_gemm({{.*}}, %[[INPUT]], %[[OUTPUT]], {{.*}})

// CHECK-LABEL: llvm.func @wide_f16_nt
// CHECK: llvm.mlir.constant(4 : i32) : i32
// CHECK-NEXT: %{{.*}} = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT: %[[INPUT:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT: %[[OUTPUT:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK-NEXT: %[[LHS:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[RHS:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: llvm.call @wafer_tx81_gemm_oriented({{.*}}, %[[INPUT]], %[[OUTPUT]], %[[LHS]], %[[RHS]], {{.*}})

// CHECK-LABEL: llvm.func @wide_f16_tn
// CHECK: llvm.mlir.constant(4 : i32) : i32
// CHECK-NEXT: %{{.*}} = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT: %[[INPUT:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT: %[[OUTPUT:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK-NEXT: %[[LHS:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[RHS:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: llvm.call @wafer_tx81_gemm_oriented({{.*}}, %[[INPUT]], %[[OUTPUT]], %[[LHS]], %[[RHS]], {{.*}})

// CHECK-LABEL: llvm.func @wide_f16_tt
// CHECK: llvm.mlir.constant(4 : i32) : i32
// CHECK-NEXT: %{{.*}} = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT: %[[INPUT:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT: %[[OUTPUT:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK-NEXT: %[[LHS:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[RHS:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: llvm.call @wafer_tx81_gemm_oriented({{.*}}, %[[INPUT]], %[[OUTPUT]], %[[LHS]], %[[RHS]], {{.*}})

// CHECK-LABEL: llvm.func @wide_bf16_nn
// CHECK: llvm.mlir.constant(4 : i32) : i32
// CHECK-NEXT: %{{.*}} = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT: %[[INPUT:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK-NEXT: %[[OUTPUT:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK: llvm.call @wafer_tx81_gemm({{.*}}, %[[INPUT]], %[[OUTPUT]], {{.*}})

// CHECK-LABEL: llvm.func @wide_bf16_nt
// CHECK: llvm.mlir.constant(4 : i32) : i32
// CHECK-NEXT: %{{.*}} = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT: %[[INPUT:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK-NEXT: %[[OUTPUT:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK-NEXT: %[[LHS:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: %[[RHS:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: llvm.call @wafer_tx81_gemm_oriented({{.*}}, %[[INPUT]], %[[OUTPUT]], %[[LHS]], %[[RHS]], {{.*}})

// CHECK-LABEL: llvm.func @wide_bf16_tn
// CHECK: llvm.mlir.constant(4 : i32) : i32
// CHECK-NEXT: %{{.*}} = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT: %[[INPUT:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK-NEXT: %[[OUTPUT:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK-NEXT: %[[LHS:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[RHS:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: llvm.call @wafer_tx81_gemm_oriented({{.*}}, %[[INPUT]], %[[OUTPUT]], %[[LHS]], %[[RHS]], {{.*}})

// CHECK-LABEL: llvm.func @wide_bf16_tt
// CHECK: llvm.mlir.constant(4 : i32) : i32
// CHECK-NEXT: %{{.*}} = llvm.mlir.constant(2 : i32) : i32
// CHECK-NEXT: %[[INPUT:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK-NEXT: %[[OUTPUT:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK-NEXT: %[[LHS:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK-NEXT: %[[RHS:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: llvm.call @wafer_tx81_gemm_oriented({{.*}}, %[[INPUT]], %[[OUTPUT]], %[[LHS]], %[[RHS]], {{.*}})
