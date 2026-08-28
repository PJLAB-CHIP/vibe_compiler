// RUN: wafer-opt %s | FileCheck %s

module {
  %a = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x8xf16, #wafer.memory<ddr, tensor>>
  %b = "builtin.unrealized_conversion_cast"()
      : () -> memref<8x16xf16, #wafer.memory<ddr, tensor>>
  %c = "builtin.unrealized_conversion_cast"()
      : () -> memref<4x16xf16, #wafer.memory<ddr, tensor>>
  %0 = wafer.tile.region(%a, %b, %c
      : memref<4x8xf16, #wafer.memory<ddr, tensor>>,
        memref<8x16xf16, #wafer.memory<ddr, tensor>>,
        memref<4x16xf16, #wafer.memory<ddr, tensor>>)
      -> (memref<4x16xf16, #wafer.memory<ddr, tensor>>) {
  ^bb0(%arg0: memref<4x8xf16, #wafer.memory<ddr, tensor>>,
       %arg1: memref<8x16xf16, #wafer.memory<ddr, tensor>>,
       %arg2: memref<4x16xf16, #wafer.memory<ddr, tensor>>):
    %a_t = memref.alloc() : memref<4x8xf16, #wafer.memory<spm, tensor>>
    wafer.tile.load %arg0 into %a_t
        : memref<4x8xf16, #wafer.memory<ddr, tensor>>
      into memref<4x8xf16, #wafer.memory<spm, tensor>>
    %b_t = memref.alloc() : memref<8x16xf16, #wafer.memory<spm, tensor>>
    wafer.tile.load %arg1 into %b_t
        : memref<8x16xf16, #wafer.memory<ddr, tensor>>
      into memref<8x16xf16, #wafer.memory<spm, tensor>>
    %a_cx = wafer.tile.materialize_layout %a_t
        : memref<4x8xf16, #wafer.memory<spm, tensor>>
       -> memref<4x8xf16, #wafer.memory<spm, cx>>
    %b_cx = wafer.tile.materialize_layout %b_t
        : memref<8x16xf16, #wafer.memory<spm, tensor>>
       -> memref<8x16xf16, #wafer.memory<spm, cx>>
    %mm = wafer.tile.gemm %a_cx, %b_cx
        : (memref<4x8xf16, #wafer.memory<spm, cx>>,
           memref<8x16xf16, #wafer.memory<spm, cx>>)
       -> memref<4x16xf16, #wafer.memory<spm, cx>>
    %mm_tensor = wafer.tile.materialize_layout %mm
        : memref<4x16xf16, #wafer.memory<spm, cx>>
       -> memref<4x16xf16, #wafer.memory<spm, tensor>>
    wafer.tile.store %mm_tensor, %arg2
        : memref<4x16xf16, #wafer.memory<spm, tensor>>
       -> memref<4x16xf16, #wafer.memory<ddr, tensor>>
    wafer.tile.yield %arg2
        : memref<4x16xf16, #wafer.memory<ddr, tensor>>
  }
}

// CHECK: wafer.tile.load
// CHECK: wafer.tile.gemm
// CHECK: wafer.tile.store
