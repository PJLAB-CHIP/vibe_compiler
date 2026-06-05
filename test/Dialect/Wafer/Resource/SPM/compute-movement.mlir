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
       -> !wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
    %b_t = wafer.tile.load %arg1
        : tensor<8x16xf16>
       -> !wafer.storage<tensor<8x16xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
    %a_cx = wafer.tile.materialize_layout %a_t
        : !wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
       -> !wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
    %b_cx = wafer.tile.materialize_layout %b_t
        : !wafer.storage<tensor<8x16xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
       -> !wafer.storage<tensor<8x16xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
    %mm = wafer.tile.gemm %a_cx, %b_cx
        : (!wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>,
           !wafer.storage<tensor<8x16xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>)
       -> !wafer.storage<tensor<4x16xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
    %mm_tensor = wafer.tile.materialize_layout %mm
        : !wafer.storage<tensor<4x16xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
       -> !wafer.storage<tensor<4x16xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
    wafer.tile.store %mm_tensor, %arg2
        : !wafer.storage<tensor<4x16xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
       -> tensor<4x16xf16>
    wafer.tile.yield %arg2 : tensor<4x16xf16>
  }
}

// CHECK: wafer.tile.load
// CHECK: wafer.tile.gemm
// CHECK: wafer.tile.store
