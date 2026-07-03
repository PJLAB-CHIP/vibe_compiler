// RUN: wafer-opt --wafer-convert-tile-region-to-instr %s | FileCheck %s

func.func @instr_tdma_transpose_materializes(
    %src: memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>,
    %dst: memref<1x1x3x2xf16, #wafer.memory<spm, tensor>>) {
  wafer.instr.tdma_data_move #wafer.instr_data_move_kind<transpose> %src into %dst
      {source_shape = array<i64: 1, 1, 2, 3>,
       dest_shape = array<i64: 1, 1, 3, 2>,
       permutation = array<i64: 0, 1, 3, 2>}
      : memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
     to memref<1x1x3x2xf16, #wafer.memory<spm, tensor>>
  return
}

// CHECK-LABEL: func.func @instr_tdma_transpose_materializes
// CHECK-SAME: %[[SRC:[a-zA-Z0-9_]+]]: memref<1x1x2x3xf16, #wafer.memory<spm, tensor>>
// CHECK-SAME: %[[DST:[a-zA-Z0-9_]+]]: memref<1x1x3x2xf16, #wafer.memory<spm, tensor>>
// CHECK-NOT: wafer.instr.tdma_data_move
// CHECK: wafer.instr.gather_scatter %[[SRC]] to %[[DST]]
// CHECK-SAME: byte_count = 12 : i64
// CHECK-NOT: wafer.instr.tdma_data_move

func.func @instr_tdma_nchw2nhwc_materializes(
    %src: memref<1x2x3x4xf16, #wafer.memory<spm, tensor>>,
    %dst: memref<1x3x4x2xf16, #wafer.memory<spm, tensor>>) {
  wafer.instr.tdma_data_move #wafer.instr_data_move_kind<nchw2nhwc> %src into %dst
      {source_shape = array<i64: 1, 2, 3, 4>,
       dest_shape = array<i64: 1, 3, 4, 2>}
      : memref<1x2x3x4xf16, #wafer.memory<spm, tensor>>
     to memref<1x3x4x2xf16, #wafer.memory<spm, tensor>>
  return
}

// CHECK-LABEL: func.func @instr_tdma_nchw2nhwc_materializes
// CHECK-SAME: %[[NCHW_SRC:[a-zA-Z0-9_]+]]: memref<1x2x3x4xf16, #wafer.memory<spm, tensor>>
// CHECK-SAME: %[[NHWC_DST:[a-zA-Z0-9_]+]]: memref<1x3x4x2xf16, #wafer.memory<spm, tensor>>
// CHECK-NOT: wafer.instr.tdma_data_move
// CHECK: wafer.instr.gather_scatter %[[NCHW_SRC]] to %[[NHWC_DST]]
// CHECK-NOT: wafer.instr.tdma_data_move

func.func @instr_tdma_nhwc2nchw_materializes(
    %src: memref<1x3x4x2xf16, #wafer.memory<spm, tensor>>,
    %dst: memref<1x2x3x4xf16, #wafer.memory<spm, tensor>>) {
  wafer.instr.tdma_data_move #wafer.instr_data_move_kind<nhwc2nchw> %src into %dst
      {source_shape = array<i64: 1, 3, 4, 2>,
       dest_shape = array<i64: 1, 2, 3, 4>}
      : memref<1x3x4x2xf16, #wafer.memory<spm, tensor>>
     to memref<1x2x3x4xf16, #wafer.memory<spm, tensor>>
  return
}

// CHECK-LABEL: func.func @instr_tdma_nhwc2nchw_materializes
// CHECK-SAME: %[[NHWC_SRC:[a-zA-Z0-9_]+]]: memref<1x3x4x2xf16, #wafer.memory<spm, tensor>>
// CHECK-SAME: %[[NCHW_DST:[a-zA-Z0-9_]+]]: memref<1x2x3x4xf16, #wafer.memory<spm, tensor>>
// CHECK-NOT: wafer.instr.tdma_data_move
// CHECK: wafer.instr.gather_scatter %[[NHWC_SRC]] to %[[NCHW_DST]]
// CHECK-NOT: wafer.instr.tdma_data_move

func.func @instr_tdma_tensor_nom_materializes(
    %src: memref<1x1x2x64xf16, #wafer.memory<spm, tensor>>,
    %dst: memref<1x1x2x64xf16, #wafer.memory<spm, cx>>) {
  wafer.instr.tdma_data_move #wafer.instr_data_move_kind<tensor_nom> %src into %dst
      {source_shape = array<i64: 1, 1, 2, 64>,
       dest_shape = array<i64: 1, 1, 2, 64>}
      : memref<1x1x2x64xf16, #wafer.memory<spm, tensor>>
     to memref<1x1x2x64xf16, #wafer.memory<spm, cx>>
  return
}

// CHECK-LABEL: func.func @instr_tdma_tensor_nom_materializes
// CHECK-SAME: %[[TENSOR_SRC:[a-zA-Z0-9_]+]]: memref<1x1x2x64xf16, #wafer.memory<spm, tensor>>
// CHECK-SAME: %[[CX_DST:[a-zA-Z0-9_]+]]: memref<1x1x2x64xf16, #wafer.memory<spm, cx>>
// CHECK-NOT: wafer.instr.tdma_data_move
// CHECK: wafer.instr.gather_scatter %[[TENSOR_SRC]] to %[[CX_DST]]
// CHECK-NOT: wafer.instr.tdma_data_move
