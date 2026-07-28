// RUN: wafer-opt --wafer-convert-tile-region-to-instr %s | FileCheck %s

func.func @mapped_load_store(
    %input: memref<2x3xf16, #wafer.memory<ddr, tensor>>,
    %output: memref<2x3xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%input, %output
      : memref<2x3xf16, #wafer.memory<ddr, tensor>>,
        memref<2x3xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<2x3xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%in: memref<2x3xf16, #wafer.memory<ddr, tensor>>,
       %out: memref<2x3xf16, #wafer.memory<ddr, tensor>>):
    %cx = memref.alloc() : memref<2x3xf16, #wafer.memory<spm, cx>>
    wafer.tile.load %in into %cx
        : memref<2x3xf16, #wafer.memory<ddr, tensor>>
      into memref<2x3xf16, #wafer.memory<spm, cx>>
    wafer.tile.store %cx, %out
        : memref<2x3xf16, #wafer.memory<spm, cx>>
       -> memref<2x3xf16, #wafer.memory<ddr, tensor>>
    wafer.tile.yield %out
        : memref<2x3xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @mapped_load_store
// CHECK: wafer.instr.rdma
// CHECK-SAME: dst_offset = 0 : i64
// CHECK-SAME: src_offset = 0 : i64
// CHECK-NEXT: wafer.instr.rdma
// CHECK-SAME: dst_offset = 8 : i64
// CHECK-SAME: src_offset = 6 : i64
// CHECK-NEXT: wafer.instr.wdma
// CHECK-SAME: dst_offset = 0 : i64
// CHECK-SAME: src_offset = 0 : i64
// CHECK-NEXT: wafer.instr.wdma
// CHECK-SAME: dst_offset = 6 : i64
// CHECK-SAME: src_offset = 8 : i64
// CHECK: wafer.instr.ncc_join [0]
// CHECK-NOT: wafer.instr.local_fence
