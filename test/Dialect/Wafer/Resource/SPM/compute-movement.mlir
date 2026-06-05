// RUN: wafer-opt %s | FileCheck %s

module {
  %a = "builtin.unrealized_conversion_cast"() : () -> tensor<4x8xf16>
  %b = "builtin.unrealized_conversion_cast"() : () -> tensor<8x16xf16>
  %c = "builtin.unrealized_conversion_cast"() : () -> tensor<4x16xf16>
  %0 = wafer.tile_region(%a, %b, %c
      : tensor<4x8xf16>, tensor<8x16xf16>, tensor<4x16xf16>)
      -> (tensor<4x16xf16>) {
  ^bb0(%arg0: tensor<4x8xf16>, %arg1: tensor<8x16xf16>, %arg2: tensor<4x16xf16>):
    %a_t = wafer.storage.load %arg0
        : tensor<4x8xf16>
       -> !wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
    %b_t = wafer.storage.load %arg1
        : tensor<8x16xf16>
       -> !wafer.storage<tensor<8x16xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
    %a_cx = wafer.layout.materialize %a_t
        : !wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
       -> !wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
    %b_cx = wafer.layout.materialize %b_t
        : !wafer.storage<tensor<8x16xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
       -> !wafer.storage<tensor<8x16xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
    %mm = wafer.compute.gemm %a_cx, %b_cx
        : (!wafer.storage<tensor<4x8xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>,
           !wafer.storage<tensor<8x16xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>)
       -> !wafer.storage<tensor<4x16xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
    %mm_tensor = wafer.layout.materialize %mm
        : !wafer.storage<tensor<4x16xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
       -> !wafer.storage<tensor<4x16xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
    wafer.storage.store %mm_tensor, %arg2
        : !wafer.storage<tensor<4x16xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
       -> tensor<4x16xf16>
    wafer.tile_yield %arg2 : tensor<4x16xf16>
  }
}

// CHECK: wafer.storage.load
// CHECK: wafer.compute.gemm
// CHECK: wafer.storage.store
