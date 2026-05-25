// RUN: not wafer-opt --wafer-check-spm-allocation='spm-capacity-bytes=256' %s 2>&1 | FileCheck %s

module {
  func.func @spm_overflow(
      %lhs: tensor<16x16xf16>,
      %rhs: tensor<16x16xf16>,
      %out: tensor<16x16xf16>) -> tensor<16x16xf16> {
    %0 = wafer.tile_region(%lhs, %rhs, %out : tensor<16x16xf16>, tensor<16x16xf16>, tensor<16x16xf16>)
        -> (tensor<16x16xf16>) {
    ^bb0(%arg0: tensor<16x16xf16>, %arg1: tensor<16x16xf16>, %arg2: tensor<16x16xf16>):
      %lhs_t = wafer.load_tile %arg0
          : tensor<16x16xf16>
         -> !wafer.tile_buffer<tensor<16x16xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
      %rhs_t = wafer.load_tile %arg1
          : tensor<16x16xf16>
         -> !wafer.tile_buffer<tensor<16x16xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
      %lhs_cx = wafer.layout.materialize %lhs_t
          : !wafer.tile_buffer<tensor<16x16xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
         -> !wafer.tile_buffer<tensor<16x16xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
      %rhs_cx = wafer.layout.materialize %rhs_t
          : !wafer.tile_buffer<tensor<16x16xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
         -> !wafer.tile_buffer<tensor<16x16xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
      %mm = wafer.compute.gemm %lhs_cx, %rhs_cx
          : (!wafer.tile_buffer<tensor<16x16xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>,
             !wafer.tile_buffer<tensor<16x16xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>)
         -> !wafer.tile_buffer<tensor<16x16xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
      %mm_t = wafer.layout.materialize %mm
          : !wafer.tile_buffer<tensor<16x16xf16>, #wafer.mem_layout<cx>, #wafer.memory_space<spm>>
         -> !wafer.tile_buffer<tensor<16x16xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
      wafer.store_tile %mm_t, %arg2
          : !wafer.tile_buffer<tensor<16x16xf16>, #wafer.mem_layout<tensor>, #wafer.memory_space<spm>>
         -> tensor<16x16xf16>
      wafer.tile_yield %arg2 : tensor<16x16xf16>
    }
    return %0 : tensor<16x16xf16>
  }
}

// CHECK: SPM allocation trial failed
// CHECK: capacity_overflow
