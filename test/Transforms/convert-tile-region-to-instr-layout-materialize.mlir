// RUN: wafer-opt --wafer-convert-tile-region-to-instr %s | FileCheck %s

func.func @materialize_padded_layout(
    %boundary: memref<4x8xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<4x8xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<4x8xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<4x8xf16, #wafer.memory<ddr, tensor>>):
    %loaded = memref.alloc()
        : memref<4x8xf16, #wafer.memory<spm, tensor>>
    wafer.tile.load %arg0 into %loaded
        : memref<4x8xf16, #wafer.memory<ddr, tensor>>
      into memref<4x8xf16, #wafer.memory<spm, tensor>>
    %cx = wafer.tile.materialize_layout %loaded
        : memref<4x8xf16, #wafer.memory<spm, tensor>>
       -> memref<4x8xf16, #wafer.memory<spm, cx>>
    wafer.tile.yield %arg0
        : memref<4x8xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

func.func @materialize_retained_tail_layout(
    %boundary: memref<4x86xf16, #wafer.memory<ddr, tensor>>) {
  %region = wafer.tile.region(%boundary
      : memref<4x86xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<4x86xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<4x86xf16, #wafer.memory<ddr, tensor>>):
    %loaded = memref.alloc()
        : memref<4x86xf16, #wafer.memory<spm, tensor>>
    wafer.tile.load %arg0 into %loaded
        : memref<4x86xf16, #wafer.memory<ddr, tensor>>
      into memref<4x86xf16, #wafer.memory<spm, tensor>>
    %cx = wafer.tile.materialize_layout %loaded
        : memref<4x86xf16, #wafer.memory<spm, tensor>>
       -> memref<4x86xf16, #wafer.memory<spm, cx>>
    wafer.tile.yield %arg0
        : memref<4x86xf16, #wafer.memory<ddr, tensor>>
  }
  return
}

// CHECK-LABEL: func.func @materialize_padded_layout
// CHECK-NOT: wafer.tile.materialize_layout
// CHECK: memref.alloc() : memref<4x8xf16, #wafer.memory<spm, tensor>>
// CHECK: %[[CX:.+]] = memref.alloc() : memref<4x8xf16, #wafer.memory<spm, cx>>
// CHECK: wafer.instr.gather_scatter %{{.+}} to %[[CX]]
// CHECK-SAME: byte_count = 64 : i64
// CHECK-SAME: inner_bytes = 64 : i64

// CHECK-LABEL: func.func @materialize_retained_tail_layout
// CHECK: %[[TAIL_CX:.+]] = memref.alloc() : memref<4x86xf16, #wafer.memory<spm, cx>>
// CHECK-COUNT-2: wafer.instr.gather_scatter %{{.+}} to %[[TAIL_CX]]
// CHECK-NOT: wafer.instr.gather_scatter %{{.+}} to %[[TAIL_CX]]
