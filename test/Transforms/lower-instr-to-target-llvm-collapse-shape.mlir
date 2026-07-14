// RUN: split-file %s %t
// RUN: wafer-opt --wafer-lower-instr-to-target-llvm='target-profile=wafer-tx81-single-card-kernel-v1' %t/compact.mlir | FileCheck %s --check-prefix=COMPACT
// RUN: not wafer-opt --wafer-lower-instr-to-target-llvm='target-profile=wafer-tx81-single-card-kernel-v1' %t/aligned.mlir 2>&1 | FileCheck %s --check-prefix=ALIGNED

//--- compact.mlir

func.func @compact_collapse(
    %input: memref<1x4x16xf32, #wafer.memory<ddr, tensor>>,
    %output: memref<4x16xf32, #wafer.memory<ddr, tensor>>) {
  %collapsed = memref.collapse_shape %input [[0, 1], [2]]
      : memref<1x4x16xf32, #wafer.memory<ddr, tensor>>
     into memref<4x16xf32, #wafer.memory<ddr, tensor>>
  %tile = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
      : memref<4x16xf32, #wafer.memory<spm, tensor>>
  wafer.instr.rdma %collapsed to %tile
      {byte_count = 256 : i64, inner_bytes = 256 : i64,
       src_strides = array<i64: 0, 0, 0>,
       src_iterations = array<i64: 1, 1, 1>}
      : memref<4x16xf32, #wafer.memory<ddr, tensor>>
     to memref<4x16xf32, #wafer.memory<spm, tensor>>
  wafer.instr.wdma %tile to %output
      {byte_count = 256 : i64, inner_bytes = 256 : i64,
       dst_strides = array<i64: 0, 0, 0>,
       dst_iterations = array<i64: 1, 1, 1>}
      : memref<4x16xf32, #wafer.memory<spm, tensor>>
     to memref<4x16xf32, #wafer.memory<ddr, tensor>>
  wafer.instr.local_fence
  return
}

// COMPACT-LABEL: llvm.func @compact_collapse(
// COMPACT-SAME: %[[INPUT:.+]]: i64, %[[OUTPUT:.+]]: i64
// COMPACT-NOT: memref.collapse_shape
// COMPACT: llvm.call @wafer_tx81_rdma(%[[INPUT]],
// COMPACT: llvm.call @wafer_tx81_wdma({{.*}}%[[OUTPUT]]
// COMPACT: llvm.return

//--- aligned.mlir

func.func @reject_aligned_collapse(
    %input: memref<1x4x16xf32, #wafer.memory<ddr, cx>>)
    -> memref<4x16xf32, #wafer.memory<ddr, cx>> {
  %collapsed = memref.collapse_shape %input [[0, 1], [2]]
      : memref<1x4x16xf32, #wafer.memory<ddr, cx>>
     into memref<4x16xf32, #wafer.memory<ddr, cx>>
  return %collapsed : memref<4x16xf32, #wafer.memory<ddr, cx>>
}

// ALIGNED: unsupported_target_address: collapse_shape requires matching Wafer tensor-layout memory and element types
