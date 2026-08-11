// RUN: wafer-opt --wafer-lower-instr-to-target-llvm %s | FileCheck %s

module {
  wafer.target.topology @default
      {card_grid = array<i64: 1, 1>, card_interconnect = "mesh",
       tile_grid = array<i64: 4, 4>, unavailable_tiles = array<i64>}
  wafer.execution.mesh @default_mesh
      {axes = ["card_partition"], shape = array<i64: 1>}

  func.func @mapped_dma(
      %input: memref<8xf16, #wafer.memory<ddr, tensor>>,
      %output: memref<8xf16, #wafer.memory<ddr, tensor>>) {
    %spm = memref.alloc() {wafer.spm.offset = #wafer.spm_offset<65536>}
        : memref<8xf16, #wafer.memory<spm, tensor>>
    wafer.instr.rdma %input to %spm
        {byte_count = 4 : i64, inner_bytes = 4 : i64,
         src_offset = 4 : i64, dst_offset = 2 : i64,
         src_strides = array<i64: 0, 0, 0>,
         src_iterations = array<i64: 1, 1, 1>}
        : memref<8xf16, #wafer.memory<ddr, tensor>>
       to memref<8xf16, #wafer.memory<spm, tensor>>
    wafer.instr.wdma %spm to %output
        {byte_count = 4 : i64, inner_bytes = 4 : i64,
         src_offset = 6 : i64, dst_offset = 8 : i64,
         dst_strides = array<i64: 0, 0, 0>,
         dst_iterations = array<i64: 1, 1, 1>}
        : memref<8xf16, #wafer.memory<spm, tensor>>
       to memref<8xf16, #wafer.memory<ddr, tensor>>
    return
  }
}

// CHECK-LABEL: llvm.func @mapped_dma
// CHECK: %[[IN_OFFSET:.+]] = llvm.mlir.constant(4 : i64)
// CHECK: %[[IN:.+]] = llvm.add %{{.+}}, %[[IN_OFFSET]] : i64
// CHECK: %[[SPM_READ_OFFSET:.+]] = llvm.mlir.constant(2 : i64)
// CHECK: %[[SPM_READ:.+]] = llvm.add %{{.+}}, %[[SPM_READ_OFFSET]] : i64
// CHECK: llvm.call @wafer_tx81_rdma_v3(%[[IN]], %[[SPM_READ]]
// CHECK: %[[SPM_WRITE_OFFSET:.+]] = llvm.mlir.constant(6 : i64)
// CHECK: %[[SPM_WRITE:.+]] = llvm.add %{{.+}}, %[[SPM_WRITE_OFFSET]] : i64
// CHECK: %[[OUT_OFFSET:.+]] = llvm.mlir.constant(8 : i64)
// CHECK: %[[OUT:.+]] = llvm.add %{{.+}}, %[[OUT_OFFSET]] : i64
// CHECK: llvm.call @wafer_tx81_wdma_v3(%[[SPM_WRITE]], %[[OUT]]
