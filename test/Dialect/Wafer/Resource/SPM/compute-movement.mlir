// RUN: wafer-opt %s | FileCheck %s

module {
  %a = "builtin.unrealized_conversion_cast"() : () -> tensor<4x8xf16>
  %b = "builtin.unrealized_conversion_cast"() : () -> tensor<8x16xf16>
  %c = "builtin.unrealized_conversion_cast"() : () -> tensor<4x16xf16>
  %0 = wafer.tile.region(%a, %b, %c
      : tensor<4x8xf16>, tensor<8x16xf16>, tensor<4x16xf16>)
      -> (tensor<4x16xf16>) {
  ^bb0(%arg0: tensor<4x8xf16>, %arg1: tensor<8x16xf16>, %arg2: tensor<4x16xf16>):
    %a_t = wafer.tile.load %arg0
        : tensor<4x8xf16>
       -> memref<4x8xf16, #wafer.memory<spm, tensor>>
    %b_t = wafer.tile.load %arg1
        : tensor<8x16xf16>
       -> memref<8x16xf16, #wafer.memory<spm, tensor>>
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
       -> tensor<4x16xf16>
    wafer.tile.yield %arg2 : tensor<4x16xf16>
  }
}

// CHECK: wafer.tile.load
// CHECK: wafer.tile.gemm
// CHECK: wafer.tile.store
