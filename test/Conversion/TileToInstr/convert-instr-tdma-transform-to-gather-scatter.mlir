// RUN: wafer-opt --pass-pipeline='builtin.module(wafer-lower-tile-region-to-instr)' %s | FileCheck %s

func.func @instr_tdma_transpose_materializes() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %src = memref.alloc() : memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
  %dst = memref.alloc() : memref<1x1x3x2xf16, #wafer.memory<spm, tensor>>
  wafer.instr.tdma_data_move #wafer.instr_data_move_kind<transpose> %src into %dst
      {source_shape = array<i64: 1, 1, 2, 3>,
       dest_shape = array<i64: 1, 1, 3, 2>,
       permutation = array<i64: 0, 1, 3, 2>}
      : memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
     to memref<1x1x3x2xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @instr_tdma_transpose_materializes
// CHECK: %[[SRC:.+]] = memref.alloc() : memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
// CHECK: %[[DST:.+]] = memref.alloc() : memref<1x1x3x2xf16, #wafer.memory<spm, tensor>>
// CHECK-NOT: wafer.instr.tdma_data_move
// CHECK: wafer.instr.gather_scatter %[[SRC]] to %[[DST]]
// CHECK-SAME: byte_count = 12 : i64
// CHECK-NOT: wafer.instr.tdma_data_move

func.func @instr_tdma_large_transpose_uses_loop_stride() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %src = memref.alloc() : memref<1x1x4096x4096xf16, #wafer.memory<spm, tensor>>
  %dst = memref.alloc() : memref<1x1x4096x4096xf16, #wafer.memory<spm, tensor>>
  wafer.instr.tdma_data_move #wafer.instr_data_move_kind<transpose> %src into %dst
      {source_shape = array<i64: 1, 1, 4096, 4096>,
       dest_shape = array<i64: 1, 1, 4096, 4096>,
       permutation = array<i64: 0, 1, 3, 2>}
      : memref<1x1x4096x4096xf16, #wafer.memory<spm, tensor>>
     to memref<1x1x4096x4096xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @instr_tdma_large_transpose_uses_loop_stride
// CHECK-COUNT-1: wafer.instr.gather_scatter
// CHECK-SAME: byte_count = 33554432 : i64
// CHECK-SAME: dst_iterations = array<i64: 4096, 4096, 1>
// CHECK-SAME: dst_strides = array<i64: 2, 8192, 0>
// CHECK-SAME: inner_bytes = 2 : i64
// CHECK-SAME: src_iterations = array<i64: 4096, 4096, 1>
// CHECK-SAME: src_strides = array<i64: 8192, 2, 0>

func.func @instr_tdma_nchw2nhwc_materializes() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %src = memref.alloc() : memref<1x2x3x4xf16, #wafer.memory<spm, tensor>>
  %dst = memref.alloc() : memref<1x3x4x2xf16, #wafer.memory<spm, tensor>>
  wafer.instr.tdma_data_move #wafer.instr_data_move_kind<nchw2nhwc> %src into %dst
      {source_shape = array<i64: 1, 2, 3, 4>,
       dest_shape = array<i64: 1, 3, 4, 2>}
      : memref<1x2x3x4xf16, #wafer.memory<spm, tensor>>
     to memref<1x3x4x2xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @instr_tdma_nchw2nhwc_materializes
// CHECK: %[[NCHW_SRC:.+]] = memref.alloc() : memref<1x2x3x4xf16, #wafer.memory<spm, tensor>>
// CHECK: %[[NHWC_DST:.+]] = memref.alloc() : memref<1x3x4x2xf16, #wafer.memory<spm, tensor>>
// CHECK-NOT: wafer.instr.tdma_data_move
// CHECK: wafer.instr.gather_scatter %[[NCHW_SRC]] to %[[NHWC_DST]]
// CHECK-NOT: wafer.instr.tdma_data_move

func.func @instr_tdma_nhwc2nchw_materializes() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %src = memref.alloc() : memref<1x3x4x2xf16, #wafer.memory<spm, tensor>>
  %dst = memref.alloc() : memref<1x2x3x4xf16, #wafer.memory<spm, tensor>>
  wafer.instr.tdma_data_move #wafer.instr_data_move_kind<nhwc2nchw> %src into %dst
      {source_shape = array<i64: 1, 3, 4, 2>,
       dest_shape = array<i64: 1, 2, 3, 4>}
      : memref<1x3x4x2xf16, #wafer.memory<spm, tensor>>
     to memref<1x2x3x4xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @instr_tdma_nhwc2nchw_materializes
// CHECK: %[[NHWC_SRC:.+]] = memref.alloc() : memref<1x3x4x2xf16, #wafer.memory<spm, tensor>>
// CHECK: %[[NCHW_DST:.+]] = memref.alloc() : memref<1x2x3x4xf16, #wafer.memory<spm, tensor>>
// CHECK-NOT: wafer.instr.tdma_data_move
// CHECK: wafer.instr.gather_scatter %[[NHWC_SRC]] to %[[NCHW_DST]]
// CHECK-NOT: wafer.instr.tdma_data_move

func.func @instr_tdma_tensor_nom_materializes() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %src = memref.alloc() : memref<1x1x2x64xf16, #wafer.memory<spm, tensor>>
  %dst = memref.alloc() : memref<1x1x2x64xf16, #wafer.memory<spm, cx>>
  wafer.instr.tdma_data_move #wafer.instr_data_move_kind<tensor_nom> %src into %dst
      {source_shape = array<i64: 1, 1, 2, 64>,
       dest_shape = array<i64: 1, 1, 2, 64>}
      : memref<1x1x2x64xf16, #wafer.memory<spm, tensor>>
     to memref<1x1x2x64xf16, #wafer.memory<spm, cx>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @instr_tdma_tensor_nom_materializes
// CHECK: %[[TENSOR_SRC:.+]] = memref.alloc() : memref<1x1x2x64xf16, #wafer.memory<spm, tensor>>
// CHECK: %[[CX_DST:.+]] = memref.alloc() : memref<1x1x2x64xf16, #wafer.memory<spm, cx>>
// CHECK-NOT: wafer.instr.tdma_data_move
// CHECK: wafer.instr.gather_scatter %[[TENSOR_SRC]] to %[[CX_DST]]
// CHECK-NOT: wafer.instr.tdma_data_move

func.func @instr_tdma_mirror_materializes() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %src = memref.alloc() : memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
  %dst = memref.alloc() : memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
  wafer.instr.tdma_data_move #wafer.instr_data_move_kind<mirror> %src into %dst
      {source_shape = array<i64: 1, 1, 2, 3>,
       dest_shape = array<i64: 1, 1, 2, 3>,
       axes = array<i64: 3>}
      : memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
     to memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @instr_tdma_mirror_materializes
// CHECK: %[[MIRROR_SRC:.+]] = memref.alloc() : memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
// CHECK: %[[MIRROR_DST:.+]] = memref.alloc() : memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
// CHECK-NOT: wafer.instr.tdma_data_move
// CHECK: wafer.instr.gather_scatter %[[MIRROR_SRC]] to %[[MIRROR_DST]]
// CHECK-NOT: wafer.instr.tdma_data_move

func.func @instr_tdma_rotate90_materializes() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %src = memref.alloc() : memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
  %dst = memref.alloc() : memref<1x1x3x2xf16, #wafer.memory<spm, tensor>>
  wafer.instr.tdma_data_move #wafer.instr_data_move_kind<rotate90> %src into %dst
      {source_shape = array<i64: 1, 1, 2, 3>,
       dest_shape = array<i64: 1, 1, 3, 2>,
       axes = array<i64: 2, 3>}
      : memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
     to memref<1x1x3x2xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @instr_tdma_rotate90_materializes
// CHECK: %[[ROTATE_SRC:.+]] = memref.alloc() : memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
// CHECK: %[[ROTATE_DST:.+]] = memref.alloc() : memref<1x1x3x2xf16, #wafer.memory<spm, tensor>>
// CHECK-NOT: wafer.instr.tdma_data_move
// CHECK: wafer.instr.gather_scatter %[[ROTATE_SRC]] to %[[ROTATE_DST]]
// CHECK-NOT: wafer.instr.tdma_data_move

func.func @instr_tdma_rotate180_materializes() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %src = memref.alloc() : memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
  %dst = memref.alloc() : memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
  wafer.instr.tdma_data_move #wafer.instr_data_move_kind<rotate180> %src into %dst
      {source_shape = array<i64: 1, 1, 2, 3>,
       dest_shape = array<i64: 1, 1, 2, 3>,
       axes = array<i64: 2, 3>}
      : memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
     to memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @instr_tdma_rotate180_materializes
// CHECK: %[[ROTATE180_SRC:.+]] = memref.alloc() : memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
// CHECK: %[[ROTATE180_DST:.+]] = memref.alloc() : memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
// CHECK-NOT: wafer.instr.tdma_data_move
// CHECK: wafer.instr.gather_scatter %[[ROTATE180_SRC]] to %[[ROTATE180_DST]]
// CHECK-NOT: wafer.instr.tdma_data_move

func.func @instr_tdma_rotate270_materializes() {
  %token = arith.constant false
  %unused = wafer.tile.region(%token : i1) -> (i1) {
  ^bb0(%tile_token: i1):
  %src = memref.alloc() : memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
  %dst = memref.alloc() : memref<1x1x3x2xf16, #wafer.memory<spm, tensor>>
  wafer.instr.tdma_data_move #wafer.instr_data_move_kind<rotate270> %src into %dst
      {source_shape = array<i64: 1, 1, 2, 3>,
       dest_shape = array<i64: 1, 1, 3, 2>,
       axes = array<i64: 2, 3>}
      : memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
     to memref<1x1x3x2xf16, #wafer.memory<spm, tensor>>
    wafer.tile.yield %tile_token : i1
  }
  return
}

// CHECK-LABEL: func.func @instr_tdma_rotate270_materializes
// CHECK: %[[ROTATE270_SRC:.+]] = memref.alloc() : memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
// CHECK: %[[ROTATE270_DST:.+]] = memref.alloc() : memref<1x1x3x2xf16, #wafer.memory<spm, tensor>>
// CHECK-NOT: wafer.instr.tdma_data_move
// CHECK: wafer.instr.gather_scatter %[[ROTATE270_SRC]] to %[[ROTATE270_DST]]
// CHECK-NOT: wafer.instr.tdma_data_move
